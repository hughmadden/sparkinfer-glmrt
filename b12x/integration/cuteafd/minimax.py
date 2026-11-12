"""MiniMax residual norm, FP32 router and paged attention KV producers.

Arithmetic/layout reference: Hugh Madden, hughmadden/cuteafd@2cbf6873e,
with hughmadden/sparkinfer-glmrt@4390f106. Reuse the shared norm and GEMV;
KV packing preserves request/head descales and skips negative token slots.
All live counts are launch scalars; callers own buffers and graph lifetimes.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32
from b12x.gemm.bf16_gemv._kernel import SmallNGemvKernel
from ._common import Operand, Scalar, compile_program
from .pointwise import _view, compile_affine_norm_aot


class _RouterScores:
    def __init__(self):
        self.project = SmallNGemvKernel(128, 6144, False, False)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, logits: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.project(x, weight, weight, logits, rows, Int64(6144), Int64(6144), Int64(128),
                     Int64(1), Int64(1), Int32(0), stream)


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
    return compile_affine_norm_aot(width=6144, eps=1e-6, zero_centered=True,
                                  input_dtype=torch.float32, parameter_dtype=torch.bfloat16)


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
