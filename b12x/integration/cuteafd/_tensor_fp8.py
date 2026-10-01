"""Per-tensor FP8 W8A8 prefill projections (ModelOpt ``quant_algo: FP8``).

A ModelOpt FP8 linear stores E4M3 ``[N, K]`` with one FP32 ``weight_scale`` and
one calibrated FP32 ``input_scale``; it is served as static W8A8:

* activations: ``q = satfinite(rn(x / input_scale))`` (E4M3, no amax pass);
* GEMM: plain E4M3 tensor-core MMAs (``DenseGemmKernel`` ``plain_fp8``, no
  per-block promotion) with FP32 accumulation, then one multiply by
  ``alpha = input_scale * weight_scale`` and one BF16 rounding.

The scales live on the device in a small FP32 operand per projection
(``tensor_scales``: FP32 ``[12]``, ``input_scale`` at 0, ``alpha_0`` at 4 and
``alpha_1`` at 8, each 16-byte aligned, the rest zero; ``alpha_1`` is the second
row part's alpha when two weights with different ``weight_scale`` are
row-concatenated, e.g. ``gate_proj | up_proj``).

``QuantRowsStatic``   BF16 ``x [rows, K]`` to E4M3 with the static input scale.
``SwiGLUStaticFp8``   ``hidden = bf16(bf16(silu(gate)) * up)`` from separate gate and
    up planes, quantized the same way (the down projection's input scale).
``TensorFp8Gemm``     the plain-FP8 GEMM pointer ABI into BF16 ``out [rows, N]``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from b12x._lib.intrinsics import (cvt_f32x4_to_e4m3x4, div_rn_f32, ld_global_nc_f32, ld_global_v4_u32,
                                  pack_f32x2_to_bfloat2, st_global_v2_u32)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

__all__ = ["QuantRowsStatic", "SwiGLUStaticFp8", "TensorFp8Gemm", "tensor_scales_operand"]

# Offsets (FP32 elements, 16-byte aligned) inside a projection's ``tensor_scales`` operand.
INPUT_SCALE, ALPHA0, ALPHA1 = 0, 4, 8
TENSOR_SCALES = 12


def tensor_scales_operand(name: str):
    from ._common import Operand
    import torch

    return Operand(f"{name}_tscale", torch.float32, f"[{TENSOR_SCALES}]", align=16,
                   note="input_scale at 0, input_scale*weight_scale at 4 (first part) and 8 (second part)")


@cute.jit
def _round_bf16(value: Float32) -> Float32:
    return _bf16_lo(pack_f32x2_to_bfloat2(value, value))


class QuantRowsStatic:
    """``q[i] = satfinite(rn(x[i] / s))`` over ``rows * K`` BF16 values, ``s`` the FP32 at
    ``scales[INPUT_SCALE]``; eight values per thread."""

    threads = 256

    def __init__(self, k: int):
        self.k = int(k)
        if self.k % 8:
            raise ValueError("static FP8 rows need K divisible by 8")

    def key(self) -> tuple:
        return ("static", self.k)

    @cute.jit
    def __call__(self, x: cute.Pointer, q: cute.Pointer, scales: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        groups = Int64(rows) * Int64(self.k // 8)
        blocks = (groups + Int64(self.threads - 1)) // Int64(self.threads)
        self.kernel(x, q, scales, groups).launch(grid=(Int32(blocks), 1, 1), block=(self.threads, 1, 1),
                                                 stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, q: cute.Pointer, scales: cute.Pointer, groups: Int64):
        group = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if group < groups:
            s = ld_global_nc_f32(Int64(scales.toint()) + Int64(4 * INPUT_SCALE))
            words = ld_global_v4_u32(Int64(x.toint()) + group * Int64(16))
            v = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
            for i in cutlass.range_constexpr(4):
                v[2 * i] = div_rn_f32(_bf16_lo(words[i]), s)
                v[2 * i + 1] = div_rn_f32(_bf16_hi(words[i]), s)
            lo = cvt_f32x4_to_e4m3x4(v[0], v[1], v[2], v[3])
            hi = cvt_f32x4_to_e4m3x4(v[4], v[5], v[6], v[7])
            st_global_v2_u32(Int64(q.toint()) + group * Int64(8), lo, hi)


class SwiGLUStaticFp8:
    """``q = satfinite(rn(bf16(bf16(silu(g)) * u) / s))`` for ``g`` / ``u`` the gate and up
    planes (BF16 ``[rows, I]`` each), ``s`` the FP32 at ``scales[INPUT_SCALE]``; the
    rounding points of ``GlmSwiGLU`` (no clamp), eight values per thread."""

    threads = 256

    def __init__(self, inter: int):
        self.inter = int(inter)
        if self.inter % 8:
            raise ValueError("SwiGLU FP8 rows need I divisible by 8")

    def key(self) -> tuple:
        return ("swiglu_static", self.inter)

    @cute.jit
    def __call__(self, gate: cute.Pointer, up: cute.Pointer, q: cute.Pointer, scales: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        groups = Int64(rows) * Int64(self.inter // 8)
        blocks = (groups + Int64(self.threads - 1)) // Int64(self.threads)
        self.kernel(gate, up, q, scales, groups).launch(grid=(Int32(blocks), 1, 1), block=(self.threads, 1, 1),
                                                        stream=stream)

    @cute.kernel
    def kernel(self, gate: cute.Pointer, up: cute.Pointer, q: cute.Pointer, scales: cute.Pointer, groups: Int64):
        group = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if group < groups:
            s = ld_global_nc_f32(Int64(scales.toint()) + Int64(4 * INPUT_SCALE))
            g = ld_global_v4_u32(Int64(gate.toint()) + group * Int64(16))
            u = ld_global_v4_u32(Int64(up.toint()) + group * Int64(16))
            v = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
            for i in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
                    gv = _bf16_lo(g[i]) if half == 0 else _bf16_hi(g[i])
                    uv = _bf16_lo(u[i]) if half == 0 else _bf16_hi(u[i])
                    silu = _round_bf16(div_rn_f32(gv, Float32(1.0) + cute.math.exp(-gv, fastmath=False)))
                    v[2 * i + half] = div_rn_f32(_round_bf16(silu * uv), s)
            lo = cvt_f32x4_to_e4m3x4(v[0], v[1], v[2], v[3])
            hi = cvt_f32x4_to_e4m3x4(v[4], v[5], v[6], v[7])
            st_global_v2_u32(Int64(q.toint()) + group * Int64(8), lo, hi)


def tensor_fp8_gemm_launch(n: int, k: int, max_rows: int, *, sm_count: int | None = None,
                           tile_k: int | None = None, tile: tuple[int, int] | None = None):
    """The ``dense_gemm(..., plain_fp8=True)`` launch for ``[rows <= max_rows, K] x [N, K]^T`` to
    BF16 with a device FP32 ``alpha``, as a pointer ABI:
    ``(a, b, sfa, sfb, c, qc_values, qc_rows, qc_mma, alpha, rows, stream)``."""
    import torch

    from b12x._lib import dense_gemm as dg
    from b12x._lib.utils import get_num_sm

    n, k, max_rows = int(n), int(k), int(max_rows)
    if sm_count is None:
        sm_count = get_num_sm(torch.device("cuda", torch.cuda.current_device()))
    plan = dg._select_default_dense_gemm_plan(max_rows, n, k, sm_count, is_mxfp8=True, expected_m=max_rows)
    if plan.swap_ab or plan.load_path != "tma":
        raise ValueError(f"tensor-FP8 prefill wants the unswapped TMA plan, got {plan}")
    if tile_k is None:
        tile_k = dg._select_mxfp8_tile_k(max_rows, n, k, max_rows, sm_count)
    policy = dg._dense_gemm_policy_for(
        m=max_rows, n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
        mma_tiler_mn=tile or plan.mma_tiler_mn, cluster_shape_mn=(1, 1), sm_count=sm_count, tile_k=tile_k,
        expected_m=max_rows)
    if policy.split_k_slices != 1:
        raise ValueError(f"tensor-FP8 prefill at {max_rows} rows chose split-K {policy.split_k_slices}")
    return dg._new_dense_gemm_launch(
        n=n, k=k, l=1, c_l=1, a_major="k", b_major="k", c_major="n", ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU, c_dtype=cutlass.BFloat16, alpha_dtype=cutlass.Float32, sf_vec_size=32,
        mma_k=32, tile_k=tile_k, mma_tiler_mn=tile or plan.mma_tiler_mn, cluster_shape_mn=(1, 1), policy=policy,
        sm_count=sm_count, sm_version="sm_120", load_path="tma", swap_ab=False, sfb_k_reuse=False,
        b_tile_major=False, alpha_is_one=False, plain_fp8=True, block_fp8=False)


class TensorFp8Gemm:
    """``out[rows, N] = bf16(alpha * (q @ w^T))`` over E4M3 ``q [rows, K]`` and ``w [N, K]``,
    ``alpha`` the FP32 at ``alpha`` (a device pointer)."""

    def __init__(self, n: int, k: int, max_rows: int, **launch):
        self.n, self.k = int(n), int(k)
        self.gemm = tensor_fp8_gemm_launch(self.n, self.k, max_rows, **launch)

    def key(self) -> tuple:
        return ("tensor_fp8", self.n, self.k, self.gemm.compile_key())

    @cute.jit
    def __call__(self, q: Int64, w_fp8: Int64, alpha: Int64, out: Int64, rows: Int32, stream: cuda.CUstream):
        gmem = cute.AddressSpace.gmem
        a = cute.make_ptr(cutlass.Float8E4M3FN, q, gmem, assumed_align=16)
        b = cute.make_ptr(cutlass.Float8E4M3FN, w_fp8, gmem, assumed_align=16)
        c = cute.make_ptr(cutlass.BFloat16, out, gmem, assumed_align=16)
        al = cute.make_ptr(Float32, alpha, gmem, assumed_align=16)
        # Scale-factor and quantized-output operands are unused by the plain-FP8 GEMM.
        sf = cute.make_ptr(cutlass.Float8E8M0FNU, alpha, gmem, assumed_align=16)
        qc = cute.make_ptr(cutlass.Float8E4M3FN, alpha, gmem, assumed_align=16)
        self.gemm(a, b, sf, sf, c, qc, sf, sf, al, rows, stream)
