"""Block-FP8 prefill projections for the GLM 5.3 Flash (``glmf``) programs.

Prefill rows project through E4M3 tensor-core MMAs (``DenseGemmKernel`` with
``block_fp8``: every K128 block's products summed in FP32, then scaled by the
row's activation scale times the weight's block scale and added to the FP32
accumulator), as the official FP8 release is served (DeepSeek-style
``w8a8_block_fp8_matmul``):

* activations: ``QuantRows128`` quantizes each row's 128-wide K blocks to E4M3
  with an FP32 scale ``amax / 448`` (1.0 for an all-zero block), values
  ``satfinite(rn(x / scale))``;
* weights: E4M3 ``[N, K]`` with the checkpoint's FP32 ``[N/128, K/128]`` block
  scales (MLA, dense and shared-expert projections: the official weights
  exactly), or FP32 ``[N, K/128]`` per-row scales (``row_scales``: the KDA
  projections quantized per row at load, as their decode copies).

``BlockFp8Projection`` is the pointer ABI the programs embed: quantize ``x``
into caller scratch, then the GEMM into BF16 ``out [rows, N]``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32, fabs_f32, fmax_f32, ld_global_v4_u32, st_global_v2_u32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

FP8_MAX = 448.0
_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class QuantRows128:
    """BF16 ``x [rows, K]`` (row stride ``stride`` elements) to E4M3 ``q [rows, K]``
    and FP32 ``scale [rows, K/128]``. Sixteen threads own a 128-wide block (8
    values each); a CTA of 256 threads quantizes 16 blocks."""

    threads = 256
    lanes = 16

    def __init__(self, k: int, stride: int | None = None):
        self.k = int(k)
        self.stride = self.k if stride is None else int(stride)
        if self.k % 128 or self.stride % 8:
            raise ValueError("block-FP8 rows need K divisible by 128 and a 16-byte row stride")
        self.groups = self.k // 128

    def key(self) -> tuple:
        return (self.k, self.stride)

    @cute.jit
    def __call__(self, x: cute.Pointer, q: cute.Pointer, scale: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        blocks = rows * Int32(self.groups)
        per_cta = Int32(self.threads // self.lanes)
        self.kernel(x, q, scale, blocks).launch(grid=((blocks + per_cta - Int32(1)) // per_cta, 1, 1),
                                                block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, q: cute.Pointer, scale: cute.Pointer, blocks: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        block = Int32(cute.arch.block_idx()[0]) * Int32(self.threads // self.lanes) + tidx // Int32(self.lanes)
        lane = tidx % Int32(self.lanes)
        values = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(8):
            values[i] = Float32(0.0)
        row = Int64(block // Int32(self.groups))
        group = Int64(block % Int32(self.groups))
        col = group * Int64(128) + Int64(lane) * Int64(8)
        valid = block < blocks
        if valid:
            words = ld_global_v4_u32(Int64(x.toint()) + (row * Int64(self.stride) + col) * Int64(2))
            for i in cutlass.range_constexpr(4):
                values[2 * i] = _bf16_lo(words[i])
                values[2 * i + 1] = _bf16_hi(words[i])
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            amax = fmax_f32(amax, fabs_f32(values[i]))
        for shift in cutlass.range_constexpr(4):
            amax = fmax_f32(amax, Float32(cute.arch.shuffle_sync_bfly(amax, offset=8 >> shift)))
        s = Float32(1.0)
        if amax > Float32(0.0):
            s = div_rn_f32(amax, Float32(FP8_MAX))
        if valid:
            for i in cutlass.range_constexpr(8):
                values[i] = div_rn_f32(values[i], s)
            lo = cvt_f32x4_to_e4m3x4(values[0], values[1], values[2], values[3])
            hi = cvt_f32x4_to_e4m3x4(values[4], values[5], values[6], values[7])
            st_global_v2_u32(Int64(q.toint()) + row * Int64(self.k) + col, lo, hi)
            if lane == Int32(0):
                at = cute.make_tensor(cute.make_ptr(Float32, Int64(scale.toint()) + (row * Int64(self.groups) + group)
                                                    * Int64(4), cute.AddressSpace.gmem, assumed_align=4),
                                      cute.make_layout((1,)))
                at[0] = s


def block_fp8_gemm_launch(n: int, k: int, max_rows: int, *, row_scales: bool = False, sm_count: int | None = None):
    """The ``dense_gemm(..., block_fp8=True)`` launch for ``[rows <= max_rows, K] x [N, K]^T`` to BF16,
    as a pointer ABI: ``(a, b, sfa, sfb, c, qc_values, qc_rows, qc_mma, alpha, rows, stream)``."""
    import torch

    from b12x._lib import dense_gemm as dg
    from b12x._lib.utils import get_num_sm

    n, k, max_rows = int(n), int(k), int(max_rows)
    if sm_count is None:
        sm_count = get_num_sm(torch.device("cuda", torch.cuda.current_device()))
    plan_n = -(-n // 128) * 128
    plan = dg._select_default_dense_gemm_plan(max_rows, plan_n, k, sm_count, is_mxfp8=True, block_fp8=True,
                                              expected_m=max_rows)
    if plan.swap_ab or plan.load_path != "tma":
        raise ValueError(f"block-FP8 prefill wants the unswapped TMA plan, got {plan}")
    policy = dg._dense_gemm_policy_for(
        m=max_rows, n=plan_n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn, cluster_shape_mn=(1, 1), sm_count=sm_count, tile_k=128,
        expected_m=max_rows, generalize_block_fp8_split_k=True)
    if policy.split_k_slices != 1:
        raise ValueError(f"block-FP8 prefill at {max_rows} rows chose split-K {policy.split_k_slices}")
    return dg._new_dense_gemm_launch(
        n=n, k=k, l=1, c_l=1, a_major="k", b_major="k", c_major="n", ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float32, c_dtype=cutlass.BFloat16, alpha_dtype=cutlass.Float32, sf_vec_size=128,
        mma_k=32, tile_k=128, mma_tiler_mn=plan.mma_tiler_mn, cluster_shape_mn=(1, 1), policy=policy,
        sm_count=sm_count, sm_version="sm_120", load_path="tma", swap_ab=False, sfb_k_reuse=False,
        b_tile_major=False, alpha_is_one=True, plain_fp8=True, block_fp8=True, block_fp8_row_b=row_scales)


def quant_scratch_bytes(k: int, rows: int) -> int:
    """E4M3 rows ``[rows, K]`` then FP32 scales ``[rows, K/128]``."""
    rows = max(int(rows), 1)
    return _align(rows * int(k)) + _align(rows * (int(k) // 128) * 4)


class BlockFp8Projection:
    """``out = x @ dequant(w)^T`` over E4M3 weights: ``x`` quantized per row and 128-K block into
    ``scratch`` (``quant_scratch_bytes``), then the block-FP8 GEMM into BF16 ``out [rows, N]``."""

    def __init__(self, n: int, k: int, max_rows: int, *, row_scales: bool = False, x_stride: int | None = None):
        self.n, self.k = int(n), int(k)
        self.row_scales = bool(row_scales)
        self.quant = QuantRows128(self.k, x_stride)
        self.gemm = block_fp8_gemm_launch(self.n, self.k, max_rows, row_scales=row_scales)

    def key(self) -> tuple:
        return (self.n, self.k, self.row_scales, self.quant.key(), self.gemm.compile_key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, w_scale: cute.Pointer, out: cute.Pointer,
                 scratch: Int64, rows: Int32, stream: cuda.CUstream):
        gmem = cute.AddressSpace.gmem
        q = cute.make_ptr(cutlass.Float8E4M3FN, scratch, gmem, assumed_align=16)
        s_at = scratch + _align_i64(Int64(rows) * Int64(self.k))
        s = cute.make_ptr(Float32, s_at, gmem, assumed_align=16)
        self.quant(x, q, s, rows, stream)
        b = cute.make_ptr(cutlass.Float8E4M3FN, Int64(w_fp8.toint()), gmem, assumed_align=16)
        sb = cute.make_ptr(Float32, Int64(w_scale.toint()), gmem, assumed_align=16)
        c = cute.make_ptr(cutlass.BFloat16, Int64(out.toint()), gmem, assumed_align=16)
        # The quantized-output and alpha operands are unused (alpha is one).
        unused = Int64(w_scale.toint())
        self.gemm(q, b, s, sb, c, cute.make_ptr(cutlass.Float8E4M3FN, unused, gmem, assumed_align=16),
                  cute.make_ptr(cutlass.Float8E8M0FNU, unused, gmem, assumed_align=16),
                  cute.make_ptr(cutlass.Float8E8M0FNU, unused, gmem, assumed_align=16),
                  cute.make_ptr(Float32, unused, gmem, assumed_align=16), rows, stream)
