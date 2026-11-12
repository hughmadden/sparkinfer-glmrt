"""MiniMax residual norm, FP32 router and paged attention KV producers.

Arithmetic/layout reference: Hugh Madden, hughmadden/cuteafd@2cbf6873e,
with hughmadden/sparkinfer-glmrt@4390f106. The residual norm and FP32-weight
router use MiniMax-specific kernels; KV packing preserves request/head
descales and skips negative token slots.
All live counts are launch scalars; callers own buffers and graph lifetimes.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4, div_rn_f32, ld_global_v4_f32, pack_f32x2_to_bfloat2, st_global_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo, _ld_cached
from ._common import Operand, Scalar, compile_program
from .minimax_router import minimax_router_scores
from ._glm_kernels import _block_sum, _reduction_storage, _rsqrt
from .pointwise import _add, _fma, _mul, _view


class _ResidualNorm:
    """FP32 residual -> BF16 ``(1 + w) * RMSNorm`` in one pass over the row.

    One CTA per row: thread ``t`` owns the eight-element groups ``j * threads
    + t``, so every warp access is a contiguous 1 KiB (FP32 residual) or
    512 B (BF16 weight) segment instead of a strided scalar. The residual
    stays in registers across the block reduction -- the row is read once,
    which is what the wide rows are limited by -- and the product is rounded
    to BF16 once, at the packed 16-byte store. The BF16 weight is a
    read-only broadcast load shared by every row; ``1 + w`` is the
    checkpoint's zero-centered gamma and there is no mean or bias term.
    """

    threads = 256

    def __init__(self, width: int, eps: float):
        self.width = int(width)
        if self.width <= 0 or self.width % (8 * self.threads):
            raise ValueError("RMSNorm width must be a positive multiple of 8 * CTA width")
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        self.vectors = self.width // (8 * self.threads)
        self.eps = float(eps)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 cos: cute.Pointer, sin: cute.Pointer, positions: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        # bias/cos/sin/positions are ABI placeholders: this norm has no bias,
        # no rotary and no subtract-mean term.
        self.kernel(x, weight, out).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _load(self, x_row: Int64, weight_base: Int64, values: cute.Tensor, weights: cute.Tensor):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            at = Int64(j * self.threads) + tidx
            v0, v1, v2, v3 = ld_global_v4_f32(x_row + at * Int64(32))
            v4, v5, v6, v7 = ld_global_v4_f32(x_row + at * Int64(32) + Int64(16))
            values[8 * j + 0] = v0
            values[8 * j + 1] = v1
            values[8 * j + 2] = v2
            values[8 * j + 3] = v3
            values[8 * j + 4] = v4
            values[8 * j + 5] = v5
            values[8 * j + 6] = v6
            values[8 * j + 7] = v7
            w0, w1, w2, w3 = _ld_cached(weight_base + at * Int64(16))
            weights[8 * j + 0] = _bf16_lo(w0)
            weights[8 * j + 1] = _bf16_hi(w0)
            weights[8 * j + 2] = _bf16_lo(w1)
            weights[8 * j + 3] = _bf16_hi(w1)
            weights[8 * j + 4] = _bf16_lo(w2)
            weights[8 * j + 5] = _bf16_hi(w2)
            weights[8 * j + 6] = _bf16_lo(w3)
            weights[8 * j + 7] = _bf16_hi(w3)

    @cute.jit
    def _scaled(self, value: Float32, weight: Float32, inv: Float32) -> Float32:
        return _mul(_mul(value, inv), _add(weight, Float32(1.0)))

    @cute.jit
    def _store(self, out_row: Int64, values: cute.Tensor, weights: cute.Tensor, inv: Float32):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            at = Int64(j * self.threads) + tidx
            first = 8 * j
            st_global_v4_u32(
                out_row + at * Int64(16),
                pack_f32x2_to_bfloat2(self._scaled(values[first], weights[first], inv),
                                      self._scaled(values[first + 1], weights[first + 1], inv)),
                pack_f32x2_to_bfloat2(self._scaled(values[first + 2], weights[first + 2], inv),
                                      self._scaled(values[first + 3], weights[first + 3], inv)),
                pack_f32x2_to_bfloat2(self._scaled(values[first + 4], weights[first + 4], inv),
                                      self._scaled(values[first + 5], weights[first + 5], inv)),
                pack_f32x2_to_bfloat2(self._scaled(values[first + 6], weights[first + 6], inv),
                                      self._scaled(values[first + 7], weights[first + 7], inv)))

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(1, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((1, self.warps), stride=(self.warps, 1)))
        count = 8 * self.vectors
        values = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        weights = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        x_row = Int64(x.toint()) + row * Int64(self.width * 4)
        out_row = Int64(out.toint()) + row * Int64(self.width * 2)
        self._load(x_row, Int64(weight.toint()), values, weights)
        square = Float32(0.0)
        for k in cutlass.range_constexpr(count):
            square = _fma(values[k], values[k], square)
        total = _block_sum(square, sums, Int32(0), self.warps)
        inv = _rsqrt(_add(div_rn_f32(total, Float32(self.width)), Float32(self.eps)))
        self._store(out_row, values, weights, inv)


class _RouterScores:
    def __init__(self):
        self.project = minimax_router_scores(128, 6144)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, logits: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.project(x, weight, logits, rows, stream)


class _KvPack:
    @cute.jit
    def __call__(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
                 batch_ids: cute.Pointer, k_descale: cute.Pointer, v_descale: cute.Pointer,
                 key_cache: cute.Pointer, value_cache: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(k, v, slots, batch_ids, k_descale, v_descale, key_cache, value_cache).launch(
            grid=(rows, 4, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
               batch_ids: cute.Pointer, k_descale: cute.Pointer, v_descale: cute.Pointer,
               key_cache: cute.Pointer, value_cache: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        lane = Int64(cute.arch.thread_idx()[0])
        k, v, slots, batch_ids = _view(k), _view(v), _view(slots), _view(batch_ids)
        k_descale, v_descale = _view(k_descale), _view(v_descale)
        slot = Int64(slots[Int64(row)])
        if slot >= Int64(0):
            scale_at = Int64(batch_ids[Int64(row)]) * Int64(4) + Int64(head)
            ki = div_rn_f32(Float32(1.0), Float32(k_descale[scale_at]))
            vi = div_rn_f32(Float32(1.0), Float32(v_descale[scale_at]))
            src = (Int64(row) * Int64(4) + Int64(head)) * Int64(128) + lane * Int64(4)
            dst = (slot * Int64(4) + Int64(head)) * Int64(128)
            kp = cute.make_ptr(Uint32, Int64(key_cache.toint()) + dst,
                               cute.AddressSpace.gmem, assumed_align=16)
            vp = cute.make_ptr(Uint32, Int64(value_cache.toint()) + dst,
                               cute.AddressSpace.gmem, assumed_align=16)
            kp[lane] = cvt_f32x4_to_e4m3x4(Float32(k[src]) * ki, Float32(k[src + Int64(1)]) * ki,
                                          Float32(k[src + Int64(2)]) * ki, Float32(k[src + Int64(3)]) * ki)
            vp[lane] = cvt_f32x4_to_e4m3x4(Float32(v[src]) * vi, Float32(v[src + Int64(1)]) * vi,
                                          Float32(v[src + Int64(2)]) * vi, Float32(v[src + Int64(3)]) * vi)


def compile_minimax_norm_aot():
    """FP32 residual to BF16 Gemma norm, with the shared seven-pointer ABI."""
    # Same program name, operand records and geometry dict the shared AffineNorm
    # exported: the engine validates the manifest's pointers and reads heads,
    # bias and rotary alongside width, epsilon, zero_centered and subtract_mean.
    # Version 2 keeps this block-per-row kernel off the old scalar kernel's
    # compile identity, which pairs the identical key tuple with version 1.
    return compile_program(_ResidualNorm(width=6144, eps=1e-6), name="affine_norm",
        operands=(
            Operand("x", torch.float32, "[rows,1,6144]"),
            Operand("weight", torch.bfloat16, "[6144]"),
            Operand("bias", torch.bfloat16, "[6144] (ignored unless bias)"),
            Operand("cos", torch.bfloat16, "[positions,0] (ignored unless rotary)"),
            Operand("sin", torch.bfloat16, "[positions,0] (ignored unless rotary)"),
            Operand("positions", torch.int64, "[rows] (ignored unless rotary)"),
            Operand("out", torch.bfloat16, "[rows,1,6144]", "out")),
        scalars=(Scalar("rows"),),
        key=(6144, 1, 1e-6, True, False, False, 0, torch.float32, torch.bfloat16),
        version=2,
        geometry=dict(width=6144, heads=1, eps=1e-6, zero_centered=True,
                      subtract_mean=False, bias=False, rotary=0))


def compile_minimax_router_scores_aot():
    """Retain the checkpoint's FP32 router weights and FP32 logits."""
    return compile_program(_RouterScores(), name="minimax_router_scores", operands=(
        Operand("x", torch.bfloat16, "[rows,6144]"),
        Operand("weight", torch.float32, "[128,6144]"),
        Operand("logits", torch.float32, "[rows,128]", "out")),
        scalars=(Scalar("rows"),), key=(), geometry=dict(hidden=6144, experts=128))


def compile_minimax_kv_pack_aot():
    """Positive request/head descales must match attention and remain fixed."""
    return compile_program(_KvPack(), name="minimax_kv_pack", operands=(
        Operand("k", torch.bfloat16, "[rows,4,128]"),
        Operand("v", torch.bfloat16, "[rows,4,128]"),
        Operand("slots", torch.int64, "[rows] (physical slots; negative skips writes)"),
        Operand("batch_ids", torch.int32, "[rows]"),
        Operand("k_descale", torch.float32, "[batch,4]"),
        Operand("v_descale", torch.float32, "[batch,4]"),
        Operand("key_cache", torch.uint8, "[pages,64,4,128]", "out"),
        Operand("value_cache", torch.uint8, "[pages,64,4,128]", "out")),
        scalars=(Scalar("rows"),), key=(), geometry=dict(heads=4, head_dim=128, page_rows=64))
