"""SM89 prefill projections: multistage block-32 GEMM against the references.

Kernels, all FP8 E4M3 with BF16 output:

* plain   - ``DenseGemmKernel`` plain FP8 (one FP32 alpha, no block scales),
            128x128x128 warp-specialized cp.async: the plain-FP8 reference.
* ws      - ``Sm89DenseBlock32Launch``: the same kernel family with K32
            activation and 32x32 weight scales (per-K32 promotion), 128x64.
* ms      - ``Sm89Block32Gemm``: multistage main loop, every warp loads and
            computes (``ms:BMxBN:stages:WMxWN``; a ``:plain`` suffix runs it
            without scales, the same loop's plain-FP8 reference).

Times a CUDA graph of back-to-back launches over weight copies exceeding L2;
checks block-scaled outputs against the FP64 dequantized product.

Usage: prefill_gemm.py [--m 2048,4096] [--shapes 4096x12288,2304x5120,8192x32768g8]
                       [--kernels plain,ws,ms:128x128:3:2x4]
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
from q2_dense_gemm_sm89 import PlainDenseLaunch
from b12x._lib.sm89_fp8 import Sm89Block32Gemm, Sm89DenseBlock32Launch
from b12x._lib.utils import make_ptr

PEAK = 597.0


def ptr(dtype, t):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="2048,4096")
    ap.add_argument("--shapes", default="12288x4096,4096x12288,2304x5120,5120x15360,25600x6144")
    ap.add_argument("--kernels", default="plain,ws,ms:128x128:3:2x4")
    ap.add_argument("--check", type=int, default=1)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    stream0 = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for shape in args.shapes.split(","):
        # NxK, or NxKgG: G groups of (N/G)x(K/G), rows packed group-major.
        groups = int(shape.split("g")[1]) if "g" in shape else 1
        n, k = (int(v) // groups for v in shape.split("g")[0].split("x"))
        torch.manual_seed(n + k)
        w = (torch.randn(groups, n, k, device=dev) * 2).to(torch.float8_e4m3fn)
        sb = torch.randint(118, 128, (groups, n // 32, k // 32), device=dev, dtype=torch.uint8)
        copies = max(2, math.ceil(160e6 / (groups * n * k)))
        ws_ = [w] + [w.clone() for _ in range(copies - 1)]
        alpha = torch.ones(1, device=dev)
        for m in (int(v) for v in args.m.split(",")):
            a = (torch.randn(groups, m, k, device=dev) * 3).to(torch.float8_e4m3fn)
            sa = torch.randint(110, 135, (groups, m, k // 32), device=dev, dtype=torch.uint8)
            out = torch.zeros(groups, m, n, device=dev, dtype=torch.bfloat16)
            ref = None
            line = f"M={m:5d} {shape:12s}"
            base_us = None
            for spec in args.kernels.split(","):
                name, *rest = spec.split(":")
                if name == "plain":
                    if groups > 1:
                        continue
                    launch = PlainDenseLaunch("fp8", n, k, (128, 128), 128, sms)
                    p = [ptr(cutlass.Float8E4M3FN, a), None, ptr(cutlass.BFloat16, out),
                         ptr(cutlass.Float32, alpha), ptr(cutlass.Float32, alpha)]
                    p[1] = ptr(cutlass.Float8E4M3FN, w)
                    compiled = cute.compile(launch, *p, Int32(m), stream0)
                    calls = [(lambda h, wt=wt: compiled(p[0], ptr(cutlass.Float8E4M3FN, wt), *p[2:],
                                                        Int32(m), cuda.CUstream(h))) for wt in ws_]
                    scaled = False
                else:
                    if name == "ws":
                        launch = Sm89DenseBlock32Launch(n, k, groups=groups, tile_mn=(128, 64))
                    else:
                        tile = tuple(int(v) for v in rest[0].split("x"))
                        layout = tuple(int(v) for v in rest[2].split("x"))
                        launch = Sm89Block32Gemm(n, k, groups=groups, tile_mn=tile,
                                                 stages=int(rest[1]), warp_layout=layout,
                                                 promote=not spec.endswith(":plain"))
                    p = [ptr(cutlass.Float8E4M3FN, a), ptr(cutlass.Float8E4M3FN, w),
                         ptr(cutlass.Float8E8M0FNU, sa), ptr(cutlass.Float8E8M0FNU, sb),
                         ptr(cutlass.BFloat16, out)]
                    compiled = cute.compile(launch, *p, p[4], p[4], p[4], ptr(cutlass.Float32, alpha),
                                            Int32(m), stream0)
                    calls = [(lambda h, wt=wt: compiled(p[0], ptr(cutlass.Float8E4M3FN, wt), *p[2:],
                                                        p[4], p[4], p[4], ptr(cutlass.Float32, alpha),
                                                        Int32(m), cuda.CUstream(h))) for wt in ws_]
                    scaled = not spec.endswith(":plain")
                bad = ""
                if scaled and args.check:
                    out.zero_()
                    calls[0](torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    if ref is None:
                        # First 256 rows of the last group.
                        rows = min(m, 256)
                        da = (a[-1, :rows].double().view(rows, k // 32, 32)
                              * torch.exp2(sa[-1, :rows].double() - 127)[..., None]).view(rows, k)
                        db = (w[-1].double().view(n // 32, 32, k // 32, 32)
                              * torch.exp2(sb[-1].double() - 127)[:, None, :, None]).view(n, k)
                        ref = da @ db.T
                    got = out[-1, : ref.shape[0]].double()
                    rel = ((got - ref).abs().max() / ref.abs().max()).item()
                    tail_rows = (out[-1, -1].abs().sum() > 0).item()
                    bad = "" if rel < 8e-3 and tail_rows else f" !{rel:.0e}/{tail_rows}"
                us = time_graph(calls, rounds=max(2, 24 // copies))
                tf = 2.0 * groups * m * n * k / (us * 1e-6) / 1e12
                if base_us is None:
                    base_us = us
                line += f" | {spec} {us:7.1f}us {tf:4.0f}TF x{us / base_us:.2f}{bad}"
            print(line, flush=True)
        del ws_
    return 0


if __name__ == "__main__":
    sys.exit(main())
