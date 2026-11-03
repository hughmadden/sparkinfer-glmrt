"""SM89 probe 3b: W8A8 E4M3 decode GEMV, plain per-tensor FP8 against 32x32
UE8M0 block-scaled weights (per-K32 promotion), rows 1, 4, 16.

Same kernel both ways (``fp8_w8a8_gemv.Fp8W8A8Gemv``), same bytes read; only
the MMA accumulation differs. Timed back-to-back over six weight copies
(clean L2). Bandwidth counts weight bytes only (scales add 1/1024).
"""

import argparse
import sys

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from _bench import time_rotating
from fp8_w8a8_gemv import Fp8W8A8Gemv
from b12x._lib.utils import make_ptr

CEILING_GBPS = 978.0
COPIES = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="12288x4096,4096x12288")
    ap.add_argument("--rows", default="1,4,16")
    ap.add_argument("--warps", default="4,8")
    ap.add_argument("--hint", default="stream")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    print(f"device={torch.cuda.get_device_name(dev)} cutlass-dsl={cutlass.__version__}")
    gmem = cute.AddressSpace.gmem
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ok_all = True
    for shape in args.shapes.split(","):
        n, k = (int(v) for v in shape.split("x"))
        torch.manual_seed(0)
        w8 = (torch.randn(n, k, device=dev) * 2).to(torch.float8_e4m3fn)
        scales = torch.randint(121, 134, (n // 32, k // 32), device=dev, dtype=torch.uint8)
        w_scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, 0).repeat_interleave(32, 1)
        w_deq = w8.float() * w_scale
        alpha = torch.full((1,), 0.05, device=dev)
        x8 = (torch.randn(16, k, device=dev) * 2).to(torch.float8_e4m3fn)
        out = torch.zeros(16, n, device=dev, dtype=torch.bfloat16)
        copies = [w8] + [w8.clone() for _ in range(COPIES - 1)]
        cptrs = [make_ptr(cutlass.Float8E4M3FN, c.data_ptr(), gmem, assumed_align=16) for c in copies]
        xp = make_ptr(cutlass.Float8E4M3FN, x8.data_ptr(), gmem, assumed_align=16)
        sp = make_ptr(cutlass.Uint8, scales.data_ptr(), gmem, assumed_align=16)
        ap_ = make_ptr(cutlass.Float32, alpha.data_ptr(), gmem, assumed_align=16)
        op = make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=16)
        for warps in (int(v) for v in args.warps.split(",")):
            if k % (128 * warps):
                continue
            base_us = {}
            for block32 in (False, True):
                gemv = Fp8W8A8Gemv(n, k, warps=warps, block32=block32, hint=args.hint)
                compiled = cute.compile(gemv, xp, cptrs[0], sp, ap_, op, Int32(16), stream)
                for rows in (int(r) for r in args.rows.split(",")):
                    out.zero_()
                    compiled(xp, cptrs[0], sp, ap_, op, Int32(rows), stream)
                    torch.cuda.synchronize()
                    ref = (x8[:rows].float() @ (w_deq if block32 else w8.float()).T) * 0.05
                    rel = ((out[:rows].float() - ref).abs().max() / ref.abs().max()).item()
                    tail_zero = bool((out[rows:] == 0).all())
                    ok = rel < 1e-2 and tail_zero
                    ok_all &= ok
                    us = time_rotating([
                        (lambda wp=wp: compiled(xp, wp, sp, ap_, op, Int32(rows), stream)) for wp in cptrs])
                    gbps = n * k / (us * 1e-6) / 1e9
                    tag = "block32" if block32 else "plain  "
                    extra = ""
                    if block32:
                        extra = f" x{us / base_us[rows]:.3f} vs plain"
                    else:
                        base_us[rows] = us
                    print(f"w8a8_gemv {tag} N={n} K={k} rows={rows:2d} warps={warps}: {us:6.1f} us "
                          f"{gbps:5.0f} GB/s ({100 * gbps / CEILING_GBPS:.0f}% of {CEILING_GBPS:.0f}){extra} "
                          f"maxrel={rel:.2e} ok={ok}", flush=True)
        del copies
    print("Q3b", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
