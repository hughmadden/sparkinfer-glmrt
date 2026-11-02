"""GLM Flash projection seams over the existing fused operators.

Only packing is new: recurrence, rank norm/cache, UK/UV and clamped SwiGLU
are the very same primitives as glmf.py. Projected rows stay BF16.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import GLM53_FLASH, GLMFGeometry, Operand, Scalar, compile_program
from .glmf import (_MlaProducer, _GlmfOutput, _GlmfFfn, _check_rows, _align, _ptr)
from .kimi_split import compile_kda_core_aot, compile_projected_kda_core_aot
from .glm_attention import compile_glm_attention_stage_aot


class _Join:
    def __init__(self, left, right):
        self.left, self.right = left, right

    @cute.jit
    def __call__(self, left: cute.Pointer, right: cute.Pointer, output: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(left, right, output).launch(
            grid=((self.left + self.right + 255) // 256, rows, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, left: cute.Pointer, right: cute.Pointer, output: cute.Pointer):
        col = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        row = Int64(cute.arch.block_idx()[1])
        if col < Int64(self.left + self.right):
            value = BFloat16(0)
            if col < Int64(self.left):
                value = left[row * Int64(self.left) + col]
            else:
                value = right[row * Int64(self.right) + col - Int64(self.left)]
            output[row * Int64(self.left + self.right) + col] = value


class _RankNorm:
    def __init__(self, g, max_rows):
        self.pack = _Join(g.q_lora_rank, g.kv_lora_rank)
        self.norm = _MlaProducer(g, max_rows=max_rows).pack

    @cute.jit
    def __call__(self, q_a: cute.Pointer, kv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, kv_slots: cute.Pointer, q_resid: cute.Pointer,
                 kv_cache: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        packed = cute.recast_ptr(scratch, dtype=BFloat16)
        self.pack(q_a, kv_a, packed, rows, stream)
        self.norm(packed, q_a_norm, kv_a_norm, kv_slots, kv_slots,
                  _ptr(Float32, Int64(kv_slots.toint()), 4), q_resid, kv_cache, rows, stream)


class _SwiGLU:
    def __init__(self, g, inter, max_rows):
        self.join = _Join(inter, inter)
        self.activation = _GlmfFfn(g, inter, False, max_rows).swiglu

    @cute.jit
    def __call__(self, gate: cute.Pointer, up: cute.Pointer, hidden: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        packed = cute.recast_ptr(scratch, dtype=BFloat16)
        self.join(gate, up, packed, rows, stream)
        self.activation(packed, hidden, rows, stream)


def compile_glmf_stage_aot(g=GLM53_FLASH, *, stage, max_rows=64, inter=2048):
    """Projection-free stages, keeping the fused Flash geometry and numerics."""
    if not isinstance(g, GLMFGeometry):
        raise ValueError("GLM Flash stages require GLMFGeometry")
    max_rows = _check_rows(max_rows)
    scratch = {}
    if stage == "rank_norm":
        launch = _RankNorm(g, max_rows)
        operands = (
            Operand("q_a", torch.bfloat16, f"[rows,{g.q_lora_rank}]"),
            Operand("kv_a", torch.bfloat16, f"[rows,{g.kv_lora_rank}]"),
            Operand("q_a_norm", torch.bfloat16, f"[{g.q_lora_rank}]"),
            Operand("kv_a_norm", torch.bfloat16, f"[{g.kv_lora_rank}]"),
            Operand("kv_slots", torch.int64, "[rows]", align=8),
            Operand("q_resid", torch.bfloat16, f"[rows,{g.q_lora_rank}]", "out"),
            Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
            Operand("scratch", torch.uint8, "[rank_norm_scratch]", "scratch"))
        scratch = {"scratch": lambda rows: _align(rows * g.qkv_a_width * 2)}
    elif stage in ("absorb", "values"):
        absorb = stage == "absorb"
        launch = (_MlaProducer(g, max_rows=max_rows).absorb if absorb else
                  _GlmfOutput(g, False, max_rows).uv)
        operands = (
            Operand("a", torch.bfloat16, f"[rows,{g.heads},{g.qk_head_dim if absorb else g.kv_lora_rank}]"),
            Operand("w", torch.bfloat16, f"[{g.heads},{g.kv_lora_rank},{g.qk_nope_dim}]" if absorb
                    else f"[{g.heads},{g.v_head_dim},{g.kv_lora_rank}]"),
            Operand("out", torch.bfloat16, f"[rows,{g.heads},{g.latent_dim if absorb else g.v_head_dim}]", "out"))
    elif stage == "swiglu":
        if type(inter) is not int or inter <= 0:
            raise ValueError("positive intermediate width required")
        launch = _SwiGLU(g, inter, max_rows)
        operands = (
            Operand("gate", torch.bfloat16, f"[rows,{inter}]"),
            Operand("up", torch.bfloat16, f"[rows,{inter}]"),
            Operand("hidden", torch.bfloat16, f"[rows,{inter}]", "out"),
            Operand("scratch", torch.uint8, "[swiglu_scratch]", "scratch"))
        scratch = {"scratch": lambda rows: _align(rows * inter * 4)}
    else:
        raise ValueError("unknown GLM Flash stage")
    return compile_program(launch, name="glmf_stage_" + stage, operands=operands,
        scalars=(Scalar("rows"),), key=(g, stage, max_rows, inter), scratch=scratch,
        geometry={**vars(g), "stage": stage, "max_rows": max_rows, "inter": inter})


def compile_glmf_kda_core_aot(g=GLM53_FLASH, *, max_rows=64, decode=True,
                               state_dtype="float32", state_rounding="window"):
    """q/k/v are projected; f_a/g_a/b and f_b/g_b remain the original low-rank gates.

    The shared core's gate input is unused for this low-rank geometry; callers
    bind any live BF16 buffer there. Do not substitute KDAGeometry: Flash's
    recurrent beta is rounded through BF16 before the recurrence.
    """
    if not isinstance(g, GLMFGeometry):
        raise ValueError("GLM Flash KDA requires its low-rank GLMFGeometry")
    return compile_kda_core_aot(g, max_rows=max_rows, decode=decode,
        state_dtype=state_dtype, state_rounding=state_rounding)


def compile_glmf_projected_kda_aot(g=GLM53_FLASH, *, max_rows=64, decode=True,
                                   state_dtype="float32", state_rounding="window"):
    """Native-weight caller supplies q/k/v, f_b, g_b and raw beta BF16 rows."""
    if not isinstance(g, GLMFGeometry):
        raise ValueError("GLM Flash KDA requires its low-rank GLMFGeometry")
    return compile_projected_kda_core_aot(g, max_rows=max_rows, decode=decode,
        state_dtype=state_dtype, state_rounding=state_rounding)


def compile_glmf_fp8_stage_aot(g=GLM53_FLASH, *, stage, max_rows=64, mode="decode",
                               query_row=None):
    """Reuse GLM's native FP8 UK/UV primitive with Flash's own NoPE geometry."""
    if not isinstance(g, GLMFGeometry):
        raise ValueError("GLM Flash FP8 stages require GLMFGeometry")
    if stage not in ("absorb", "values"):
        raise ValueError("GLM Flash FP8 stage must be absorb or values")
    if mode not in ("decode", "prefill"):
        raise ValueError("GLM Flash FP8 mode must be decode or prefill")
    return compile_glm_attention_stage_aot(g, stage=stage, max_rows=max_rows,
        fp8_only=mode, query_row=query_row)
