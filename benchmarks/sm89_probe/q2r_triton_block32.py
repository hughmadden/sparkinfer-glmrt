"""SM89 probe 2 reference: Triton FP8 GEMM with 32x32 UE8M0 weight blocks.

Calibration only (b12x core kernels stay CuTe DSL): each K32 ``tl.dot`` runs
into a fresh FP32 partial that is scaled by its column block's power-of-two
scale and added to the accumulator, against the same kernel without scales.
Timed like the CuTe probe (two weight copies, back-to-back).
"""

import sys

import torch
import triton
import triton.language as tl

from _bench import time_rotating

M = 4096
SHAPES = ((12288, 4096), (4096, 12288))


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "G": 8}, num_stages=s, num_warps=w)
        for s in (4, 6) for w in (4, 8)
    ] + [
        triton.Config({"BM": 128, "BN": 256, "G": 8}, num_stages=s, num_warps=8) for s in (3, 4)
    ],
    key=["M", "N", "K", "BLOCK32"],
)
@triton.jit
def _mm32(a, b, sb, c, alpha, M, N, K, BLOCK32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          G: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = pid // (G * npn)
    first_m = group * G
    gm = min(npm - first_m, G)
    pm = first_m + (pid % (G * npn)) % gm
    pn = (pid % (G * npn)) // gm
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, 32)
    pa = a + rm[:, None] * K + rk[None, :]
    pb = b + rn[None, :] * K + rk[:, None]
    ps = sb + (rn // 32) * (K // 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K // 32):
        if BLOCK32:
            e = tl.load(ps + kk).to(tl.int32)
            s = (e << 23).to(tl.float32, bitcast=True)
            acc += tl.dot(tl.load(pa), tl.load(pb)) * s[None, :]
        else:
            acc = tl.dot(tl.load(pa), tl.load(pb), acc)
        pa += 32
        pb += 32
    acc = acc * tl.load(alpha)
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


def main() -> int:
    dev = torch.device("cuda", 0)
    print(f"device={torch.cuda.get_device_name(dev)} triton={triton.__version__}")
    for n, k in SHAPES:
        torch.manual_seed(0)
        a8 = torch.randn(M, k, device=dev).to(torch.float8_e4m3fn)
        b8 = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        sb = torch.randint(121, 134, (n // 32, k // 32), device=dev, dtype=torch.uint8)
        w_scale = torch.exp2(sb.float() - 127.0).repeat_interleave(32, 0).repeat_interleave(32, 1)
        alpha = torch.full((1,), 0.05, device=dev)
        c = torch.empty(M, n, device=dev, dtype=torch.bfloat16)
        b2 = b8.clone()
        base = None
        for block32 in (False, True):
            grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
            run = lambda bb: _mm32[grid](a8, bb, sb, c, alpha, M, n, k, BLOCK32=block32)
            run(b8)
            torch.cuda.synchronize()
            ref = (a8.float() @ ((b8.float() * w_scale) if block32 else b8.float()).T) * 0.05
            rel = ((c.float() - ref).abs().max() / ref.abs().max()).item()
            us = time_rotating([lambda: run(b8), lambda: run(b2)], rounds=10)
            tf = 2.0 * M * n * k / (us * 1e-6) / 1e12
            ratio = "" if base is None else f" x{us / base:.3f} vs plain"
            base = us if base is None else base
            print(f"triton_fp8_k32 {'block32' if block32 else 'plain  '} M={M} N={n} K={k}: {us / 1e3:.3f} ms "
                  f"{tf:.1f} TF/s ({100 * tf / 597:.0f}% of 597){ratio} maxrel={rel:.2e} cfg={_mm32.best_config}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
