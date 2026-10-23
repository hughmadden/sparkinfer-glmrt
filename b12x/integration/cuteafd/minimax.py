"""MiniMax-M3.1-Flash coordinator programs; every live count is a scalar.

KV projection concatenates K then V rows. Gate/up projections concatenate
gate then up rows. MXFP8 weights retain the checkpoint E4M3 payload; run
scale_prep once on their per-row K32 UE8M0 scales. All projection outputs and
norm/activation boundaries are BF16; the residual and router math are FP32.
"""
from __future__ import annotations

import torch

from ._common import Operand, Scalar, compile_program
from ._minimax_kernels import IndexQuant, KvPack, KvSplit, Route, RouterScores, SwigluOai
from ._mxfp8_linear import Mxfp8ScalePrep
from .pointwise import compile_affine_norm_aot, compile_residual_add_aot
from .rope import compile_rope_tables_aot
from .scaled_linear import compile_scaled_linear_aot

LINEAR_SHAPES = {
    "q": (6144, 8192), "kv": (6144, 1024), "o": (8192, 6144),
    "index_q": (6144, 512), "index_k": (6144, 128),
    "shared_gate_up": (6144, 6144), "shared_down": (3072, 6144),
    "dense_gate_up": (6144, 24576), "dense_down": (12288, 6144),
}


def compile_minimax_linear_aot(projection, *, max_rows=4096):
    if projection not in LINEAR_SHAPES:
        raise ValueError(f"unknown MiniMax projection {projection!r}")
    k, n = LINEAR_SHAPES[projection]
    return compile_scaled_linear_aot(input_features=k, output_features=n,
        encoding="mxfp8", activation_encoding="mxfp8", max_rows=max_rows)


def compile_minimax_scale_prep_aot():
    """Load-time byte permutation; outputs multiple128, input_groups=K/32 multiple4."""
    return compile_program(Mxfp8ScalePrep(), name="minimax_mxfp8_scale_prep", operands=(
        Operand("scale", torch.uint8, "[outputs,input_groups] (checkpoint UE8M0)"),
        Operand("scale_mma", torch.uint8, "[outputs*input_groups]", "out")),
        scalars=(Scalar("outputs"), Scalar("input_groups")), key=(),
        geometry=dict(scale_block=32, rows_per_tile=128, groups_per_tile=4))


def compile_minimax_norm_aot():
    return compile_affine_norm_aot(width=6144, eps=1e-6, zero_centered=True,
                                  input_dtype=torch.float32, parameter_dtype=torch.bfloat16)


def compile_minimax_qk_norm_rope_aot(*, heads):
    if heads not in (1, 4, 64):
        raise ValueError("MiniMax Q/K norms require 1, 4 or 64 heads")
    return compile_affine_norm_aot(width=128, heads=heads, eps=1e-6, zero_centered=True,
                                  rotary=64, input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16)


def compile_minimax_rope_tables_aot():
    """Pass theta=1e7 and positions=the planned position capacity."""
    return compile_rope_tables_aot(rotary_span=64)


def compile_minimax_residual_aot():
    return compile_residual_add_aot(width=6144, delta_dtype=torch.bfloat16)


def compile_minimax_kv_split_aot():
    return compile_program(KvSplit(), name="minimax_kv_split", operands=(
        Operand("kv", torch.bfloat16, "[rows,1024] (K first)"),
        Operand("k", torch.bfloat16, "[rows,4,128]", "out"),
        Operand("v", torch.bfloat16, "[rows,4,128]", "out")),
        scalars=(Scalar("rows"),), key=(), geometry=dict(kv_heads=4, head_dim=128))


def compile_minimax_swiglu_aot(*, width):
    if width not in (3072, 12288):
        raise ValueError("MiniMax shared/dense SwiGLU widths are 3072/12288")
    return compile_program(SwigluOai(width), name="minimax_swiglu_oai", operands=(
        Operand("gate_up", torch.bfloat16, f"[rows,{2 * width}] (gate first)"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,), geometry=dict(width=width, alpha=1.702, limit=7.0))


def compile_minimax_router_scores_aot():
    return compile_program(RouterScores(), name="minimax_router_scores", operands=(
        Operand("x", torch.bfloat16, "[rows,6144]"),
        Operand("weight", torch.float32, "[128,6144] (checkpoint FP32)"),
        Operand("logits", torch.float32, "[rows,128]", "out")), scalars=(Scalar("rows"),),
        key=(), geometry=dict(hidden=6144, experts=128))


def compile_minimax_route_aot():
    return compile_program(Route(), name="minimax_sigmoid_route", operands=(
        Operand("logits", torch.float32, "[rows,128]"),
        Operand("bias", torch.float32, "[128] (selection only)"),
        Operand("ids", torch.int32, "[rows,4] (descending score, lower id breaks ties)", "out"),
        Operand("weights", torch.float32, "[rows,4] (unbiased, renormalized, times2)", "out")),
        scalars=(Scalar("rows"),), key=(), geometry=dict(experts=128, topk=4, factor=2.0))


def compile_minimax_index_quant_aot(*, heads=4, paged=False):
    """Contiguous Q/K rows, or 64-row paged index K; negative slots skip writes.

    For contiguous prefill K use heads=1; reserve and zero-pad K rows as
    required by the indexer. Paged cache uses payload then FP32 row scales.
    """
    if heads not in (1, 4) or (paged and heads != 1):
        raise ValueError("index quant requires 1/4 heads; paged K requires one")
    return compile_program(IndexQuant(heads, paged), name="minimax_index_quant", operands=(
        Operand("x", torch.bfloat16, f"[rows,{heads},128]"),
        Operand("slots", torch.int64, "[rows] (physical token slots; ignored unless paged)"),
        Operand("values", torch.uint8, "[pool_pages,8448]" if paged else f"[rows,{heads},128]", "out"),
        Operand("scales", torch.float32, "[0] (ignored)" if paged else f"[rows,{heads}]", "out")),
        scalars=(Scalar("rows"),), key=(heads, paged),
        geometry=dict(heads=heads, head_dim=128, paged=paged, page_rows=64 if paged else 0))


def compile_minimax_kv_pack_aot():
    """Use the same positive request/head descales as attention; keep them fixed for the request."""
    return compile_program(KvPack(), name="minimax_kv_pack", operands=(
        Operand("k", torch.bfloat16, "[rows,4,128]"),
        Operand("v", torch.bfloat16, "[rows,4,128]"),
        Operand("slots", torch.int64, "[rows] (physical token slots; negative skips writes)"),
        Operand("batch_ids", torch.int32, "[rows]"),
        Operand("k_descale", torch.float32, "[batch,4]"),
        Operand("v_descale", torch.float32, "[batch,4]"),
        Operand("key_cache", torch.uint8, "[pool_pages,64,4,128]", "out"),
        Operand("value_cache", torch.uint8, "[pool_pages,64,4,128]", "out")),
        scalars=(Scalar("rows"),), key=(), geometry=dict(heads=4, head_dim=128, page_rows=64))
