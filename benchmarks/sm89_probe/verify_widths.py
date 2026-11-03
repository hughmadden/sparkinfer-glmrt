"""SM89 V4.1 verify widths (rows 17-96): GEMV and GEMM plans per projection.

Builds each candidate kernel directly through the native pointer ABI
(activation row scales and weight block scales in plain layouts) and times it
at the requested live rows from a CUDA graph of back-to-back launches over
weight copies exceeding L2 (clean L2). Every timed run is checked against the
FP64 dequantized product.

Plans: gemv:W:C (Sm89Block32Gemv, W warps, C column groups, capacity = the
smallest of --caps holding the rows; the exported verify capacities are 64 and
80) and dense:MxN (cp.async DenseGemmKernel block-32 tile, capacity 256).

Plan "bundle" times what the exporter builds: the variant of the smallest
capacity of 64/80/256 holding the rows (several kernels split by live rows).

Usage: verify_widths.py [--shapes main,o_b] [--rows 24,48,64,80,96] [--plans gemv:8:4,dense:128x64]
                        [--caps 64,80,96]
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
from b12x._lib.sm89_fp8 import Sm89Block32Gemv, Sm89DenseBlock32Launch
from b12x._lib.utils import make_ptr

SHAPES = {
    "engram": (25600, 6144, 1), "ffn_up": (2304, 5120, 1), "ffn_down": (5120, 2304, 1),
    "main": (5120, 15360, 1), "q_a": (1280, 5120, 1), "q_b": (32768, 1280, 1),
    "kv": (512, 5120, 1), "o_b": (5120, 8192, 1), "o_a": (8192, 32768, 8),
    "index_q": (4096, 1280, 1), "q_b_tp2": (16384, 1280, 1), "o_b_tp2": (2560, 8192, 1),
    "ffn_tp2_up": (1152, 5120, 1), "ffn_tp2_down": (5120, 1152, 1),
}


def ptr(dtype, t):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class _Exported:
    """The exported variant for ``rows``: the smallest capacity of 64/80/256
    holding it, built by the SM89 plan (o_b and o_b_tp2 on the fixed recipe)."""

    def __init__(self, n, k, groups, cap, fixed):
        from b12x._lib.sm89_fp8 import sm89_block32_plan, _part_launch, Sm89RowSplitLaunch

        plan = sm89_block32_plan(size_m=cap, size_n=n, size_k=k, num_groups=groups,
                                 sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                                 fixed_rows=fixed)
        parts = [(part["rows"], _part_launch(part, n, k, groups)) for part in plan["parts"]]
        self.launch = parts[0][1] if len(parts) == 1 else Sm89RowSplitLaunch(parts)
        self.desc = "+".join(
            "gemv" if part["kind"] == "gemv" else "x".join(map(str, part["tile_mn"]))
            for part in plan["parts"])


def build(plan, n, k, groups, rows, caps, name=""):
    kind, *rest = plan.split(":")
    if kind == "bundle":
        cap = next(c for c in (64, 80, 256) if c >= rows)
        exported = _Exported(n, k, groups, cap, name in ("o_b", "o_b_tp2") and cap <= 80)
        return exported.launch, cap
    if kind == "gemv":
        warps, cg = int(rest[0]), int(rest[1])
        cap = next(c for c in caps if c >= rows)
        return Sm89Block32Gemv(n, k, groups=groups, max_rows=cap, warps=warps, col_groups=cg), cap
    atom_24 = rest[0].endswith("a24")
    tm, tn = (int(v) for v in rest[0].removesuffix("a24").split("x"))
    extra = {"atom_24": True} if atom_24 else {}
    return Sm89DenseBlock32Launch(n, k, groups=groups, tile_mn=(tm, tn), **extra), 256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--rows", default="24,32,48,64,72,80,88,96")
    ap.add_argument("--plans", default="gemv:8:4,dense:128x64")
    ap.add_argument("--caps", default="64,80,96")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]
    dev = torch.device("cuda", 0)
    stream0 = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    rows_list = [int(r) for r in args.rows.split(",")]
    for name in args.shapes.split(","):
        n, k, groups = SHAPES[name]
        gn, gk = n // groups, k // groups
        torch.manual_seed(n + k)
        cap = 256
        a = (torch.randn(groups, cap, gk, device=dev) * 3).to(torch.float8_e4m3fn)
        sa = torch.randint(110, 135, (groups, cap, gk // 32), device=dev, dtype=torch.uint8)
        w = (torch.randn(groups, gn, gk, device=dev) * 2).to(torch.float8_e4m3fn)
        sb = torch.randint(115, 130, (groups, gn // 32, gk // 32), device=dev, dtype=torch.uint8)
        weight_bytes = gn * gk * groups
        copies = max(2, math.ceil(160e6 / weight_bytes))
        ws = [w] + [w.clone() for _ in range(copies - 1)]
        wps = [ptr(cutlass.Float8E4M3FN, t) for t in ws]
        out = torch.zeros(groups * cap * gn, device=dev, dtype=torch.bfloat16)
        for plan in args.plans.split(","):
            line = f"{name:12s} {n:5d}x{k:<5d} {plan:14s}"
            if plan.startswith("dense") and groups > 1 and "x32" in plan:
                continue  # 32-wide dense tiles do not support grouped launches
            cache = {}
            for rows in rows_list:
                if plan.startswith("gemv") and rows > caps[-1]:
                    line += f" | r{rows:<3d}      -     "
                    continue
                launch, kcap = build(plan, gn, gk, groups, rows, caps, name)
                # Live rows packed group-major with live M, as the quantizers write them.
                av = a[:, :rows].contiguous()
                sav = sa[:, :rows].contiguous()
                p = [ptr(cutlass.Float8E4M3FN, av), wps[0], ptr(cutlass.Float8E8M0FNU, sav),
                     ptr(cutlass.Float8E8M0FNU, sb), ptr(cutlass.BFloat16, out)]
                if kcap not in cache:
                    cache[kcap] = cute.compile(launch, *p, p[4], p[4], p[4], ptr(cutlass.Float32, out),
                                               Int32(rows), stream0)
                compiled = cache[kcap]
                call = lambda h, wp, rows=rows: compiled(p[0], wp, p[2], p[3], p[4], p[4], p[4], p[4],
                                                         p[4], Int32(rows), cuda.CUstream(h))
                out.zero_()
                call(torch.cuda.current_stream().cuda_stream, wps[0])
                torch.cuda.synchronize()
                worst = 0.0
                got = out[: groups * rows * gn].view(groups, rows, gn).double()
                for g in range(groups):
                    da = (av[g].double().view(rows, gk // 32, 32)
                          * torch.exp2(sav[g].double() - 127)[..., None]).view(rows, gk)
                    db = (w[g].double().view(gn // 32, 32, gk // 32, 32)
                          * torch.exp2(sb[g].double() - 127)[:, None, :, None]).view(gn, gk)
                    ref = da @ db.T
                    worst = max(worst, ((got[g] - ref).abs().max() / ref.abs().max()).item())
                us = time_graph([(lambda h, wp=wp: call(h, wp)) for wp in wps],
                                rounds=max(2, 48 // copies))
                gbps = weight_bytes / (us * 1e-6) / 1e9
                bad = "" if worst < 8e-3 else f"!{worst:.0e}"
                line += f" | r{rows:<3d}{us:7.1f}us {gbps:4.0f}{bad}"
            print(line, flush=True)
        del ws, wps
    return 0


if __name__ == "__main__":
    sys.exit(main())
