"""SM89 probe 1: can the pinned CuTe-DSL compile and run kernels for sm_89?

Runs a trivial elementwise kernel, then single-CTA warp-MMA tiles with
``MmaF16BF16Op`` (BF16, m16n8k16) and ``MmaFP8Op`` (E4M3, m16n8k32), each
checked against a torch reference.
"""

import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack


def _arch():
    from cutlass.cutlass_dsl import CuTeDSL

    return CuTeDSL._get_dsl().get_arch_enum()


@cute.kernel
def _add_one_kernel(gx: cute.Tensor, gy: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    gy[i] = gx[i] + Float32(1.0)


@cute.jit
def add_one(mx: cute.Tensor, my: cute.Tensor):
    n = cute.size(mx)
    _add_one_kernel(mx, my).launch(grid=(n // 128, 1, 1), block=(128, 1, 1))


class WarpMmaTile:
    """One CTA computes C[M,N] = A[M,K] @ B[N,K]^T straight from registers."""

    def __init__(self, op_kind: str, m: int = 64, n: int = 64, k: int = 64):
        self.op_kind = op_kind
        self.m, self.n, self.k = m, n, k

    def _tiled_mma(self):
        if self.op_kind == "bf16":
            op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16))
            inst_k = 16
        else:
            op = cute.nvgpu.warp.MmaFP8Op(cutlass.Float8E4M3FN, Float32, (16, 8, 32))
            inst_k = 32
        return cute.make_tiled_mma(op, (2, 2, 1), permutation_mnk=(32, 32, inst_k))

    @cute.jit
    def __call__(self, ma: cute.Tensor, mb: cute.Tensor, mc: cute.Tensor):
        tiled_mma = self._tiled_mma()
        self.kernel(ma, mb, mc, tiled_mma).launch(grid=(1, 1, 1), block=(128, 1, 1))

    @cute.kernel
    def kernel(self, ma: cute.Tensor, mb: cute.Tensor, mc: cute.Tensor, tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        thr_mma = tiled_mma.get_slice(tidx)
        tCgA = thr_mma.partition_A(ma)
        tCgB = thr_mma.partition_B(mb)
        tCgC = thr_mma.partition_C(mc)
        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        cute.autovec_copy(tCgA, tCrA)
        cute.autovec_copy(tCgB, tCrB)
        tCrC.fill(0.0)
        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)
        cute.autovec_copy(tCrC, tCgC)


def _as_cute(t: torch.Tensor, dtype=None):
    r = from_dlpack(t, assumed_align=16)
    if dtype is not None:
        r.element_type = dtype
    return r


def main() -> int:
    dev = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(dev)
    print(f"device={torch.cuda.get_device_name(dev)} cc={cap} cutlass-dsl={cutlass.__version__}")
    ok = True

    x = torch.arange(1024, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    add_one(_as_cute(x), _as_cute(y))
    torch.cuda.synchronize()
    good = torch.equal(y, x + 1)
    print(f"dsl_arch={_arch()} trivial_kernel ok={good}")
    ok &= good

    torch.manual_seed(0)
    m = n = k = 64
    a = torch.randn(m, k, device=dev).to(torch.bfloat16)
    b = torch.randn(n, k, device=dev).to(torch.bfloat16)
    c = torch.zeros(m, n, device=dev, dtype=torch.float32)
    WarpMmaTile("bf16")(_as_cute(a), _as_cute(b), _as_cute(c))
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    err = (c - ref).abs().max().item()
    good = err < 1e-3
    print(f"MmaF16BF16Op m16n8k16 64x64x64 max_abs_err={err:.3e} ok={good}")
    ok &= good

    # Small integers first: every product and partial sum is exact, so any
    # error is a fragment-layout bug, not accumulator precision.
    fp8_tile = WarpMmaTile("fp8")
    for label, a8, b8 in (
        ("int", torch.randint(-3, 4, (m, k), device=dev).float().to(torch.float8_e4m3fn),
         torch.randint(-3, 4, (n, k), device=dev).float().to(torch.float8_e4m3fn)),
        ("randn", (torch.randn(m, k, device=dev) * 2).to(torch.float8_e4m3fn),
         (torch.randn(n, k, device=dev) * 2).to(torch.float8_e4m3fn)),
    ):
        c.zero_()
        fp8_tile(
            _as_cute(a8.view(torch.uint8), cutlass.Float8E4M3FN),
            _as_cute(b8.view(torch.uint8), cutlass.Float8E4M3FN),
            _as_cute(c),
        )
        torch.cuda.synchronize()
        ref = a8.double() @ b8.double().T
        err = (c.double() - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        # FP8 MMA accumulates with reduced internal precision on this
        # generation; exact inputs must be exact, random inputs within ~1e-3.
        good = err == 0.0 if label == "int" else rel < 2e-3
        print(f"MmaFP8Op m16n8k32 e4m3 64x64x64 {label}: max_abs_err={err:.3e} rel={rel:.2e} ok={good}")
        ok &= good
    print("Q1", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
