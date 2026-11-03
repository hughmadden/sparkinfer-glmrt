"""SM89 V4.1 coordinator FP8 projections through the exported pointer ABI.

For each projection and capacity this compiles exactly what the exporter
compiles (row/grouped quant + K32 block-scaled GEMM), runs them the way the
native shim does (activation values and row scales in scratch, the MMA-scale
pointer aliasing the row scales on SM89, grouped output rearranged), and
compares against a torch reference: bit-exact quantization replica, then the
dequantized product in FP64. Timing replays a CUDA graph of back-to-back GEMM
launches over weight copies that together exceed L2 (clean L2, graph launch
gaps included, no host dispatch).

Usage: v41_fp8_check.py [--projections q_a,kv,...] [--rows 1,16,...] [--time]
"""

import argparse
import math
import sys

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from _bench import time_graph
from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot
from b12x._lib.intrinsics import pow2_ceil_ue8m0_torch
from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid
from b12x._lib.sm89_fp8 import sm89_block32_plan
from b12x._lib.utils import make_ptr
from b12x.gemm.wo_projection._quant_cute import compile_wo_grouped_quant_aot

PROJECTIONS = (
    ('engram', 25600, 6144), ('ffn_up', 2304, 5120), ('ffn_down', 5120, 2304),
    ('main', 5120, 15360), ('q_a', 1280, 5120), ('q_b', 32768, 1280),
    ('kv', 512, 5120), ('o_b', 5120, 8192), ('o_a', 8192, 32768), ('index_q', 4096, 1280),
    ('q_b_tp2', 16384, 1280), ('o_b_tp2', 2560, 8192),
    ('ffn_tp2_up', 1152, 5120), ('ffn_tp2_down', 5120, 1152),
)
CAPACITIES = (1, 16, 64, 80, 256, 1024, 4096)
INV_E4M3_MAX = torch.tensor(1.0 / 448.0, dtype=torch.float32)
GBPS_CEILING, TFLOPS_CEILING = 978.0, 597.0


def quant_reference(x: torch.Tensor, floor: float):
    """Bit-exact replica of the K32 row quantizer: UE8M0 ceil scales of
    amax(32) (floored), E4M3 RNE of x * 2^(127 - e); zero blocks scale 127."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // 32, 32)
    amax = blocks.abs().amax(-1)
    if floor > 0:
        amax = amax.clamp_min(floor)
    _, byte = pow2_ceil_ue8m0_torch(amax * INV_E4M3_MAX.to(x.device))
    byte = torch.where(amax == 0, torch.full_like(byte, 127), byte)
    inv = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    values = (blocks * inv[..., None]).to(torch.float8_e4m3fn).reshape(rows, k)
    return values, byte


def rope_reference(x: torch.Tensor, freqs: torch.Tensor):
    """Inverse RoPE of the grouped quantizer (last 64 of each 512 columns),
    rounded to BF16: fma_rn(v0, c, round(v1*s)) and fma_rn(v1, c, round(-v0*s))."""
    rows, k = x.shape
    v = x.float().reshape(rows, k // 512, 512).clone()
    tail = v[..., 448:].reshape(rows, k // 512, 32, 2)
    c = freqs[:, None, :, 0].float()
    s = freqs[:, None, :, 1].float()
    v0, v1 = tail[..., 0], tail[..., 1]
    t0 = (v1 * s).double()
    t1 = (-(v0 * s)).double()
    r0 = (v0.double() * c.double() + t0).float().bfloat16().float()
    r1 = (v1.double() * c.double() + t1).float().bfloat16().float()
    v[..., 448:] = torch.stack((r0, r1), -1).reshape(rows, k // 512, 64)
    return v.reshape(rows, k).bfloat16()


def ptr(dtype, t):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def check(name, n, k, capacity, rows, sms, stream, time_it):
    groups = 8 if name == 'o_a' else 1
    gk, gn = k // groups, n // groups
    dev = torch.device('cuda')
    torch.manual_seed(hash((name, capacity)) & 0xFFFF)
    x = (torch.randn(rows, k, device=dev) * torch.rand(rows, 1, device=dev) * 4).bfloat16()
    w = (torch.randn(groups, gn, gk, device=dev) * 2).to(torch.float8_e4m3fn)
    sb = torch.randint(118, 128, (groups, gn // 32, gk // 32), device=dev, dtype=torch.uint8)
    values = torch.zeros(groups * capacity * gk, device=dev, dtype=torch.uint8)
    scale_rows = torch.zeros(groups * capacity * (gk // 32) + 16, device=dev, dtype=torch.uint8)
    # C is [L, live rows, N]: the group stride follows live M.
    out_flat = torch.zeros(groups * capacity * gn, device=dev, dtype=torch.bfloat16)
    out = out_flat[:groups * rows * gn].view(groups, rows, gn)
    alpha = torch.ones(1, device=dev)
    expected_m = 16 if (name in ('o_b', 'o_b_tp2') and capacity <= 80) else capacity
    freqs = None
    if groups > 1:
        quant = compile_wo_grouped_quant_aot(groups=groups, group_width=gk)
        grid = min(rows * 32, sms * 4)
        if capacity == 80:
            # Exercise the inverse-RoPE variant on one capacity.
            freqs = torch.randn(rows, 32, 2, device=dev, dtype=torch.float32)
            quant = compile_wo_grouped_quant_aot(groups=groups, group_width=gk, row_frequencies=True)
        src = ptr(cutlass.BFloat16, x) if freqs is None else ptr(cutlass.Float32, freqs)
        quant(ptr(cutlass.BFloat16, x), ptr(cutlass.Int64, x), src,
              ptr(cutlass.Uint32, values), ptr(cutlass.Uint8, scale_rows), ptr(cutlass.Uint8, scale_rows),
              Int32(rows), Int32(rows * 64 if freqs is not None else 1), Int32(grid), stream)
    else:
        quant = compile_mxfp8_rows_quant_aot(size_k=k, scale_block_size=32, expected_m=expected_m,
                                             amax_floor=1e-4)
        grid = mxfp8_rows_quant_aot_grid(size_k=k, rows=rows, expected_m=expected_m, sm_count=sms)
        quant(ptr(cutlass.BFloat16, x), ptr(cutlass.Uint32, values), ptr(cutlass.Uint8, scale_rows),
              ptr(cutlass.Uint8, scale_rows), Int32(rows), Int32(grid), stream)
    gemm, split = compile_dense_gemm_mxfp8_aot(
        size_m=capacity, size_n=gn, size_k=gk, num_groups=groups, expected_m=expected_m,
        sfb_k_replicated=False, return_split_k_metadata=True, sm_count=sms)
    assert split == 1
    gptrs = [ptr(cutlass.Float8E4M3FN, values), ptr(cutlass.Float8E4M3FN, w),
             ptr(cutlass.Float8E8M0FNU, scale_rows), ptr(cutlass.Float8E8M0FNU, sb),
             ptr(cutlass.BFloat16, out_flat)]
    aptr = ptr(cutlass.Float32, alpha)
    run = lambda wp, st=stream: gemm(gptrs[0], wp, gptrs[2], gptrs[3], gptrs[4], gptrs[4], gptrs[4],
                                     gptrs[4], aptr, Int32(rows), st)
    run(gptrs[1])
    torch.cuda.synchronize()

    # Reference: bit-exact activation quantization, then FP64 dequantized product.
    xin = x if freqs is None else rope_reference(x, freqs)
    floor = 0.0 if groups > 1 else 1e-4
    q_ref, s_ref = [], []
    for gidx in range(groups):
        qv, qs = quant_reference(xin[:, gidx * gk:(gidx + 1) * gk], floor)
        q_ref.append(qv)
        s_ref.append(qs)
    got_v = values[:groups * rows * gk].view(groups, rows, gk)
    got_s = scale_rows[:groups * rows * (gk // 32)].view(groups, rows, gk // 32)
    quant_mismatch = sum(int((got_v[i].view(torch.float8_e4m3fn).view(torch.uint8)
                              != q_ref[i].view(torch.uint8)).sum()) +
                         int((got_s[i] != s_ref[i]).sum()) for i in range(groups))
    worst = 0.0
    for gidx in range(groups):
        a = (q_ref[gidx].double().reshape(rows, gk // 32, 32) *
             torch.exp2(s_ref[gidx].double() - 127)[..., None]).reshape(rows, gk)
        b = (w[gidx].double().reshape(gn // 32, 32, gk // 32, 32) *
             torch.exp2(sb[gidx].double() - 127)[:, None, :, None]).reshape(gn, gk)
        ref = a @ b.T
        err = ((out[gidx].double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
        worst = max(worst, err)
    tail_clean = bool((out_flat[groups * rows * gn:] == 0).all())
    ok = quant_mismatch == 0 and worst < 8e-3 and tail_clean and math.isfinite(worst)
    plan = sm89_block32_plan(size_m=capacity, size_n=gn, size_k=gk, num_groups=groups, sm_count=sms)
    line = (f"{name:12s} N={n:5d} K={k:5d} cap={capacity:4d} rows={rows:4d} "
            f"{plan['kind']:5s} quant_mismatch={quant_mismatch} rel_err={worst:.2e} ok={ok}")
    if time_it:
        weight_bytes = n * k // groups
        copies = max(2, math.ceil(160e6 / weight_bytes))
        ws = [w] + [w.clone() for _ in range(copies - 1)]
        wps = [ptr(cutlass.Float8E4M3FN, c) for c in ws]
        us = time_graph([(lambda h, wp=wp: run(wp, cuda.CUstream(h))) for wp in wps],
                        rounds=max(2, 64 // copies))
        flops = 2.0 * rows * weight_bytes
        gbps = weight_bytes / (us * 1e-6) / 1e9
        tfs = flops / (us * 1e-6) / 1e12
        line += (f" | gemm {us:8.1f} us {gbps:6.0f} GB/s ({100 * gbps / GBPS_CEILING:3.0f}%) "
                 f"{tfs:6.1f} TF/s ({100 * tfs / TFLOPS_CEILING:3.0f}%)")
        del ws, wps
    print(line, flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--projections', default=','.join(p[0] for p in PROJECTIONS))
    ap.add_argument('--rows', default=','.join(map(str, CAPACITIES)))
    ap.add_argument('--live', default='full', help="'full' (rows = capacity) or 'odd' (capacity - 3)")
    ap.add_argument('--time', action='store_true')
    args = ap.parse_args()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device={torch.cuda.get_device_name(0)} sms={sms} cutlass-dsl={cutlass.__version__}")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    wanted = set(args.projections.split(','))
    ok = True
    for name, n, k in PROJECTIONS:
        if name not in wanted:
            continue
        for capacity in (int(r) for r in args.rows.split(',')):
            rows = capacity if args.live == 'full' else max(1, capacity - 3)
            ok &= check(name, n, k, capacity, rows, sms, stream, args.time)
    print('V41_FP8', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
