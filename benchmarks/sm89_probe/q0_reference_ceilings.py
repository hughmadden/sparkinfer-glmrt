"""SM89 probe 0: library reference points for the prefill shapes.

cuBLAS BF16 (``torch.mm``), cuBLASLt per-tensor FP8 (``torch._scaled_mm``) and
a plain Triton matmul, timed like the CuTe probes (back-to-back over two
weight copies). These calibrate the CuTe DSL numbers; they are not serving
paths. Decode-side references are in ``q3_stream_reference.py``.
"""

import sys

import torch
import triton
import triton.language as tl

from _bench import time_rotating

M = 4096
SHAPES = ((12288, 4096), (4096, 12288))  # (N, K): weight is [N, K]


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "G": 8}, num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K", "IS_FP8"],
)
@triton.jit
def _mm(a, b, c, scale, M, N, K, IS_FP8: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr, G: tl.constexpr):
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
    rk = tl.arange(0, BK)
    pa = a + rm[:, None] * K + rk[None, :]
    pb = b + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(pa), tl.load(pb), acc)
        pa += BK
        pb += BK
    if IS_FP8:
        acc = acc * tl.load(scale)
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


def triton_mm(a, b, scale, is_fp8):
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _mm[grid](a, b, c, scale, m, n, k, IS_FP8=is_fp8)
    return c


def tflops(ms, n, k):
    return 2.0 * M * n * k / (ms * 1e-3) / 1e12


def do_bench(fn_of_b, bs):
    """Mean ms per call rotating over the weight copies in ``bs``."""
    return time_rotating([(lambda b=b: fn_of_b(b)) for b in bs], rounds=10) * 1e-3


def main() -> int:
    dev = torch.device("cuda", 0)
    print(f"device={torch.cuda.get_device_name(dev)} torch={torch.__version__} triton={triton.__version__}")
    torch.manual_seed(0)
    one = torch.ones((), device=dev)
    for n, k in SHAPES:
        a = torch.randn(M, k, device=dev).to(torch.bfloat16)
        b = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
        bs = (b, b.clone())
        ref = a.float() @ b.float().T
        out = a @ b.T
        ms = do_bench(lambda bb: a @ bb.T, bs)
        print(f"cublas_bf16   M={M} N={n} K={k}: {ms:.3f} ms {tflops(ms, n, k):.1f} TF/s "
              f"maxrel={((out.float() - ref).abs().max() / ref.abs().max()).item():.2e}")
        out = triton_mm(a, b, one, False)
        ms = do_bench(lambda bb: triton_mm(a, bb, one, False), bs)
        print(f"triton_bf16   M={M} N={n} K={k}: {ms:.3f} ms {tflops(ms, n, k):.1f} TF/s "
              f"maxrel={((out.float() - ref).abs().max() / ref.abs().max()).item():.2e} cfg={_mm.best_config}")
        a8 = a.to(torch.float8_e4m3fn)
        b8 = (b * 20).to(torch.float8_e4m3fn)
        b8s = (b8, b8.clone())
        sa = torch.ones((), device=dev)
        sb = torch.full((), 0.05, device=dev)
        ref8 = (a8.float() @ b8.float().T) * 0.05
        out = torch._scaled_mm(a8, b8.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        ms = do_bench(lambda bb: torch._scaled_mm(a8, bb.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), b8s)
        print(f"cublaslt_fp8  M={M} N={n} K={k}: {ms:.3f} ms {tflops(ms, n, k):.1f} TF/s "
              f"maxrel={((out.float() - ref8).abs().max() / ref8.abs().max()).item():.2e}")
        out = triton_mm(a8, b8, sb, True)
        ms = do_bench(lambda bb: triton_mm(a8, bb, sb, True), b8s)
        print(f"triton_fp8    M={M} N={n} K={k}: {ms:.3f} ms {tflops(ms, n, k):.1f} TF/s "
              f"maxrel={((out.float() - ref8).abs().max() / ref8.abs().max()).item():.2e} cfg={_mm.best_config}")
        del a, b, bs, a8, b8, b8s, ref, ref8, out
    return 0


if __name__ == "__main__":
    sys.exit(main())
