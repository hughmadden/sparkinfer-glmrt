"""SM89 V4.1 decode GEMV: warps x split-K plans for the small-N projections.

Times ``Sm89Block32Gemv`` per (warps, slices) at rows 1/4/8/16, including the
fixed-order second-pass reduce (a CuTe twin of the native ``reduce_splits``:
sum slices 0..S-1 in FP32, BF16 RNE) when split. CUDA graph of back-to-back
launches over weight copies exceeding L2. Checks every plan against the FP64
dequantized product.

Usage: decode_split.py [--shapes kv,q_a,...] [--rows 1,4,8,16] [--plans 8x1x1,16x1x2,...]
(plan = warps x slices x col_groups)
"""

import argparse
import math
import sys

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from _bench import time_graph
from b12x._lib.sm89_fp8 import Sm89Block32Gemv
from b12x._lib.utils import make_ptr

SHAPES = {"kv": (512, 5120), "q_a": (1280, 5120), "index_q": (4096, 1280), "ffn_up": (2304, 5120),
          "ffn_tp2_up": (1152, 5120), "ffn_tp2_down": (5120, 1152), "o_b_tp2": (2560, 8192),
          "q_b_tp2": (16384, 1280), "ffn_down": (5120, 2304), "o_b": (5120, 8192)}
READ_CEILING = {"kv": 626, "q_a": 794, "index_q": 759, "ffn_up": 864, "ffn_tp2_up": 777,
                "ffn_tp2_down": 777, "o_b_tp2": 913, "q_b_tp2": 915}


class ReduceSplits:
    """CuTe twin of the native reduce_splits: out[i] = bf16(sum_s partials[s][i])."""

    def __init__(self, slices: int):
        self.slices = slices

    @cute.jit
    def __call__(self, partials: cute.Pointer, out: cute.Pointer, elements: Int32,
                 stream: cuda.CUstream):
        self.kernel(partials, out, elements).launch(
            grid=((elements + Int32(255)) // Int32(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, partials: cute.Pointer, out: cute.Pointer, elements: Int32):
        i = Int32(cute.arch.block_idx()[0]) * Int32(256) + Int32(cute.arch.thread_idx()[0])
        if i < elements:
            total = Float32(0.0)
            for sl in cutlass.range_constexpr(self.slices):
                at = Int64(partials.toint()) + (Int64(sl) * Int64(elements) + Int64(i)) * Int64(4)
                total = total + cute.make_ptr(Float32, at, cute.AddressSpace.gmem, assumed_align=4)[0]
            dst = cute.make_ptr(cutlass.BFloat16, Int64(out.toint()) + Int64(i) * Int64(2),
                                cute.AddressSpace.gmem, assumed_align=2)
            dst[0] = total.to(cutlass.BFloat16)


def ptr(dtype, t):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="kv,q_a,index_q,ffn_up,ffn_tp2_up,ffn_tp2_down,o_b_tp2,q_b_tp2")
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--plans", default="8x1,16x1,8x2,16x2,8x4,16x4,4x4")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    stream0 = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    rows_list = [int(r) for r in args.rows.split(",")]
    plans = [tuple(int(v) for v in (p + "x1" if p.count("x") == 1 else p).split("x"))
             for p in args.plans.split(",")]
    for name in args.shapes.split(","):
        n, k = SHAPES[name]
        torch.manual_seed(n + k)
        cap = max(rows_list)
        a = (torch.randn(cap, k, device=dev) * 3).to(torch.float8_e4m3fn)
        sa = torch.randint(110, 135, (cap, k // 32), device=dev, dtype=torch.uint8)
        w = (torch.randn(n, k, device=dev) * 2).to(torch.float8_e4m3fn)
        sb = torch.randint(115, 130, (n // 32, k // 32), device=dev, dtype=torch.uint8)
        copies = max(2, math.ceil(160e6 / (n * k)))
        ws = [w] + [w.clone() for _ in range(copies - 1)]
        da = (a.double().view(cap, k // 32, 32) * torch.exp2(sa.double() - 127)[..., None]).view(cap, k)
        db = (w.double().view(n // 32, 32, k // 32, 32)
              * torch.exp2(sb.double() - 127)[:, None, :, None]).view(n, k)
        ref = da @ db.T
        partials = torch.zeros(4 * cap * n, device=dev, dtype=torch.float32)
        out = torch.zeros(cap * n, device=dev, dtype=torch.bfloat16)
        for warps, slices, cg in plans:
            if k // 128 < slices:
                continue
            gemv = Sm89Block32Gemv(n, k, max_rows=16, warps=warps, slices=slices, col_groups=cg)
            c = partials if slices > 1 else out
            cdt = cutlass.Float32 if slices > 1 else cutlass.BFloat16
            ptrs = [ptr(cutlass.Float8E4M3FN, a), ptr(cutlass.Float8E4M3FN, w),
                    ptr(cutlass.Float8E8M0FNU, sa), ptr(cutlass.Float8E8M0FNU, sb), ptr(cdt, c)]
            compiled = cute.compile(gemv, *ptrs, ptrs[4], ptrs[4], ptrs[4],
                                    ptr(cutlass.Float32, partials), Int32(1), stream0)
            red = None
            if slices > 1:
                red = cute.compile(ReduceSplits(slices), ptr(cutlass.Float32, partials),
                                   ptr(cutlass.BFloat16, out), Int32(1), stream0)
            wps = [ptr(cutlass.Float8E4M3FN, t) for t in ws]
            line = f"{name:12s} {n:5d}x{k:<5d} warps={warps:2d} slices={slices} cg={cg}"
            for rows in rows_list:
                def call(h, wp, rows=rows):
                    st = cuda.CUstream(h)
                    compiled(ptrs[0], wp, ptrs[2], ptrs[3], ptrs[4], ptrs[4], ptrs[4], ptrs[4],
                             ptrs[4], Int32(rows), st)
                    if red is not None:
                        red(ptr(cutlass.Float32, partials), ptr(cutlass.BFloat16, out),
                            Int32(rows * n), st)
                out.zero_()
                call(torch.cuda.current_stream().cuda_stream, ptrs[1])
                torch.cuda.synchronize()
                got = out[:rows * n].view(rows, n).double()
                rel = ((got - ref[:rows]).abs().max() / ref[:rows].abs().max()).item()
                us = time_graph([(lambda h, wp=wp: call(h, wp)) for wp in wps],
                                rounds=max(2, 64 // copies))
                gbps = n * k / (us * 1e-6) / 1e9
                ok = "" if rel < 8e-3 else f" BAD rel={rel:.1e}"
                line += f" | r{rows} {us:6.2f}us {gbps:4.0f}GB/s {100 * gbps / 978:3.0f}%{ok}"
            print(line + f" | read ceiling {READ_CEILING.get(name, 0)}", flush=True)
        del ws
    return 0


if __name__ == "__main__":
    sys.exit(main())
