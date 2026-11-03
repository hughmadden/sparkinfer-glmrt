"""SM89 probe 4: V4.1 router scores (BF16 [rows,5120] @ gate[E,5120].T -> FP32).

Compiles ``compile_v41_router_scores_aot(experts=E, activated=True)`` -- the
exported program: skinny GEMV with fused sqrt(softplus) up to 64 live rows,
raw logits from the MMA route above -- and checks it against a float64 torch
reference. Timing replays a CUDA graph of back-to-back launches that rotate
through enough (x, gate, out) copies to exceed L2 several times, so every call
streams its operands from DRAM (launch gaps included). Ceiling: the larger of
bytes / 978 GB/s and FLOPs / 301 TF/s (RTX 4090 D, measured).

Usage: q4_v41_router_sm89.py [--experts 128,384] [--rows 1,5,...] [--mma-only]
       [--tile WARPS_M,WARPS_N,TILE_M,TILE_N,STAGES] [--no-time]
"""

import argparse
import math
import sys
import time

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F

from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared import v41_router as router

HIDDEN = 5120
ROWS = (1, 5, 16, 64, 65, 512, 2048, 4096)
PEAK_BYTES = 978e9
PEAK_FLOPS = 301e12
L2_DEFEAT_BYTES = 3 * 72 * 2**20


def ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def ceiling_us(rows, experts):
    nbytes = experts * HIDDEN * 2 + rows * HIDDEN * 2 + rows * experts * 4
    flops = 2.0 * rows * experts * HIDDEN
    return max(nbytes / PEAK_BYTES, flops / PEAK_FLOPS) * 1e6, nbytes


def time_graph(compiled, operands, rows, min_launches=600):
    """Mean us per call: one graph rotates through every operand copy."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for x, w, out in operands:  # warm (module load, first-touch)
            compiled(ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16),
                     ptr(out, cutlass.Float32), cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for x, w, out in operands:
                compiled(ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16),
                         ptr(out, cutlass.Float32), cutlass.Int32(rows), current_cuda_stream())
        replays = max(3, math.ceil(min_launches / len(operands)))
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(replays):
            graph.replay()
        end.record(stream)
        end.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * len(operands))


def run(experts, rows_list, *, mma_only, timed):
    torch.manual_seed(experts)
    w = (torch.randn(experts, HIDDEN, device="cuda") * 0.02).bfloat16()
    from b12x._lib.compiler import compile as compile_kernel

    def compile_route(kernel):
        return compile_kernel(
            kernel,
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            cutlass.Int32(1), current_cuda_stream())

    tiles = router.v41_router_mma_tiles(experts)
    t0 = time.time()
    if mma_only:
        compiled = compile_route(router._router_mma(experts, HIDDEN))
        label = f"mma tiles={tiles}"
    else:
        compiled = router.compile_v41_router_scores_aot(experts=experts, activated=True)
        label = f"export skinny<={router.V41_ROUTER_SKINNY_MAX_ROWS} mma tiles={tiles}"
    print(f"E={experts} {label} compile={time.time() - t0:.1f}s", flush=True)
    # Bits of the plain SM120-shaped tile (64x16, 4x1 warps): every row band
    # retiles the same per-output K order, so the MMA route must match it.
    canonical = None
    if router.target_is_pre_hopper() and len(tiles) > 1:
        canonical = compile_route(router._V41RouterScoresCpAsync(experts, hidden=HIDDEN))
    ok = True
    for rows in rows_list:
        x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
        out = torch.full((rows, experts), float("nan"), device="cuda")
        compiled(ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16), ptr(out, cutlass.Float32),
                 cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        logits = x.double() @ w.double().T
        skinny = not mma_only and rows <= router.V41_ROUTER_SKINNY_MAX_ROWS
        ref = torch.sqrt(F.softplus(logits)) if skinny else logits
        diff = (out.double() - ref).abs()
        rel = (diff.max() / ref.abs().max()).item()
        elem = (diff / ref.abs().clamp_min(1e-3 * ref.abs().max())).max().item()
        finite = bool(torch.isfinite(out).all())
        case_ok = finite and rel < 1e-3
        bits = ""
        if canonical is not None and not skinny:
            same = torch.full_like(out, float("nan"))
            canonical(ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16), ptr(same, cutlass.Float32),
                      cutlass.Int32(rows), current_cuda_stream())
            torch.cuda.synchronize()
            identical = torch.equal(out.view(torch.int32), same.view(torch.int32))
            case_ok &= identical
            bits = f" bits=={'64x16' if identical else 'DIFFER'}"
        ok &= case_ok
        line = (f"  E={experts} rows={rows:5d} route={'skinny' if skinny else 'mma':6s} "
                f"maxrel={rel:.2e} elemrel={elem:.2e} finite={finite}{bits} ok={case_ok}")
        if timed:
            ceil_us, nbytes = ceiling_us(rows, experts)
            copies = max(2, min(256, math.ceil(L2_DEFEAT_BYTES / nbytes)))
            operands = [(x if i == 0 else x.clone(), w if i == 0 else w.clone(),
                         torch.empty_like(out)) for i in range(copies)]
            us = time_graph(compiled, operands, rows)
            line += (f" | {us:8.2f} us ceiling {ceil_us:7.2f} us ({100 * ceil_us / us:5.1f}%) "
                     f"{nbytes / us / 1e3:6.1f} GB/s {2.0 * rows * experts * HIDDEN / us / 1e6:6.1f} TF/s "
                     f"copies={copies}")
            del operands
        print(line, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="128,384")
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--mma-only", action="store_true")
    ap.add_argument("--tile", help="one pre-Hopper MMA tile for every width: WARPS_M,WARPS_N,TILE_M,TILE_N,STAGES")
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
          f"sms={torch.cuda.get_device_properties(0).multi_processor_count} "
          f"cutlass-dsl={cutlass.__version__} pre_hopper={router.target_is_pre_hopper()}", flush=True)
    rows_list = [int(r) for r in args.rows.split(",")]
    ok = True
    for experts in (int(e) for e in args.experts.split(",")):
        if args.tile:
            router._PRE_HOPPER_TILES[(experts, HIDDEN)] = (
                (4096, *(int(v) for v in args.tile.split(","))),)
        ok &= run(experts, rows_list, mma_only=args.mma_only, timed=not args.no_time)
    print("Q4", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
