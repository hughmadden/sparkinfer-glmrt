"""SM89 probe 3: FP8-weight decode GEMV (``MmaFp8Gemv``) at rows 1, 4, 16.

E4M3 weights with FP32 128x128 block scales, BF16 activations, BF16 MMA
(weights widened through FP16, legal on SM89), split-K warps folded through
shared memory. Timed L2-cold by rotating back-to-back launches through six
weight copies (300 MB against the 72 MB L2), as a decode step walks layers.
(``triton.testing.do_bench`` flushes L2 with a memset that leaves 72 MB of
dirty lines, whose write-back costs a 50 MB read ~24 us; it is not used.)
Bandwidth counts weight bytes only.
"""

import argparse
import sys

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from _bench import time_rotating
from b12x._lib.utils import make_ptr
from b12x.integration.cuteafd._fp8_weights import FP8_GEMV_CONFIG, MmaFp8Gemv, gemv_warps

CEILING_GBPS = 978.0
COPIES = 6


def run(n, k, rows_list, warps, groups, dev, load_hint="stream"):
    torch.manual_seed(0)
    w8 = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand((n + 127) // 128, (k + 127) // 128, device=dev) * 0.02 + 0.01
    w_ref = (w8.float().view(n // 128, 128, k // 128, 128) * sb[:, None, :, None]).view(n, k)
    max_rows = max(rows_list)
    gemv = MmaFp8Gemv(n, k, max_rows=max_rows, warps=warps, groups=groups, load_hint=load_hint)
    x = torch.randn(max_rows, k, device=dev).to(torch.bfloat16)
    out = torch.zeros(max_rows, n, device=dev, dtype=torch.bfloat16)
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(cutlass.BFloat16, x.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, w8.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float32, sb.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemv, *ptrs, Int32(max_rows), stream)
    copies = [w8] + [torch.empty_like(w8).copy_(w8) for _ in range(COPIES - 1)]
    copy_ptrs = [make_ptr(cutlass.Float8E4M3FN, c.data_ptr(), gmem, assumed_align=16) for c in copies]
    ok_all = True
    for rows in rows_list:
        out.zero_()
        compiled(*ptrs, Int32(rows), stream)
        torch.cuda.synchronize()
        # Reference: BF16-rounded dequantized weight, as the kernel feeds the MMA.
        ref = x[:rows].float() @ w_ref.to(torch.bfloat16).float().T
        rel = ((out[:rows].float() - ref).abs().max() / ref.abs().max()).item()
        ok = rel < 1e-2
        ok_all &= ok
        us = time_rotating([
            (lambda wp=wp: compiled(ptrs[0], wp, ptrs[2], ptrs[3], Int32(rows), stream))
            for wp in copy_ptrs])
        gbps = n * k / (us * 1e-6) / 1e9
        print(f"MmaFp8Gemv N={n} K={k} rows={rows:2d} warps={warps} groups={groups} {load_hint}: {us:6.1f} us "
              f"{gbps:5.0f} GB/s ({100 * gbps / CEILING_GBPS:.0f}% of {CEILING_GBPS:.0f}) "
              f"maxrel={rel:.2e} ok={ok}", flush=True)
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="12288x4096,4096x12288")
    ap.add_argument("--rows", default="1,4,16")
    ap.add_argument("--groups", default="")
    ap.add_argument("--warps", default="")
    ap.add_argument("--hints", default="stream")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    print(f"device={torch.cuda.get_device_name(dev)} cutlass-dsl={cutlass.__version__}")
    rows = [int(r) for r in args.rows.split(",")]
    ok = True
    for shape in args.shapes.split(","):
        n, k = (int(v) for v in shape.split("x"))
        shipped = FP8_GEMV_CONFIG.get((n, k))
        group_opts = [int(g) for g in args.groups.split(",")] if args.groups else [1]
        warp_opts = [int(w) for w in args.warps.split(",")] if args.warps else [gemv_warps(k)]
        for warps in warp_opts:
            for groups in group_opts:
                for hint in args.hints.split(","):
                    ok &= run(n, k, rows, warps, groups, dev, hint)
        if shipped:
            print(f"  (shipped SM120 config for {n}x{k}: {shipped})")
    print("Q3", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
