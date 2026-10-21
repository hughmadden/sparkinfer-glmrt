"""Native pointer ABIs for staged region compression and selected paged GQA.

Cache pages, selected logical positions and transaction publication belong to
callers. These adapters do not commit state or allocate Python workspaces.
"""
from __future__ import annotations

import torch

from ._common import Operand, Scalar, compile_program


def compile_region_compress_aot():
    """Complete eight-token spans only; incomplete output slots stay untouched."""
    from b12x.attention.paged._weighted_regions import _CompressRegions

    return compile_program(_CompressRegions(), name="region_compress", operands=(
        Operand("keys", torch.bfloat16, "[tokens,256]"),
        Operand("logits", torch.bfloat16, "[tokens,256]"),
        Operand("starts", torch.int32, "[regions]", align=4),
        Operand("counts", torch.int32, "[regions] (eight or ignored)", align=4),
        Operand("output", torch.float8_e4m3fn, "[regions,256]", "inout"),
        Operand("unrounded", torch.float32, "[regions,256]", "inout")),
        scalars=(Scalar("regions"), Scalar("limit", "int64", "raw staging token extent")),
        key=(8, 256), geometry=dict(region_tokens=8, proxy_dim=256))


def compile_region_scores_aot(*, groups, capacity):
    """Signed F32 mixture weights; visibility excludes the current partial region."""
    from b12x.attention.paged._weighted_regions import _WeightedRegionScore

    if groups <= 0 or capacity <= 0:
        raise ValueError("provider groups and summary capacity must be positive")
    return compile_program(_WeightedRegionScore(int(groups), int(capacity)),
        name="region_scores", operands=(
            Operand("query", torch.bfloat16, f"[rows,{groups},4,256]"),
            Operand("weights", torch.float32, f"[rows,{groups},4] (signed)"),
            Operand("summaries", torch.float8_e4m3fn, f"[requests,{capacity},256]"),
            Operand("requests", torch.int32, "[rows]", align=4),
            Operand("positions", torch.int64, "[rows]", align=8),
            Operand("scores", torch.float32, f"[rows,{groups},{capacity}]", "out")),
        scalars=(Scalar("rows"), Scalar("limit", "int64", "summary request extent")),
        key=(int(groups), int(capacity)),
        geometry=dict(groups=int(groups), capacity=int(capacity), proxy_dim=256, heads_per_group=4))


def compile_selected_gqa_aot(*, query_heads, kv_heads, page_rows=64,
                             selection_width=4104, selection_groups=None, head_dim=192,
                             kv_format="bf16"):
    """BF16 or per-token/head FP8 planar pages, direct output without split scratch.

    Block tables are [requests,table_width]; table_batch=1 is the shared-table
    layout. For SWA pass selection_width=512 and selection_groups=1. The owner
    maps ring pages plus speculative staging into this read-only view. CSA uses
    per-KV-head provider selections [rows,kv_heads,4104]. Unused descales and
    split partials remain explicit ABI placeholders, matching the shared kernel.
    """
    from b12x.attention.paged.forward_extend_generic import PagedForwardKernel

    if selection_groups is None:
        selection_groups = kv_heads
    if head_dim not in (192, 256) or min(query_heads, kv_heads, page_rows, selection_width) <= 0:
        raise ValueError("positive selected GQA geometry and D192/D256 required")
    if kv_format not in ("bf16", "fp8"):
        raise ValueError("KV format must be bf16 or fp8")
    fp8 = kv_format == "fp8"
    token_stride = int(kv_heads) * int(head_dim)
    if fp8:
        from .pointwise import fp8_kv_plane_bytes
        token_stride = fp8_kv_plane_bytes(kv_heads, head_dim)
    strides = (int(page_rows) * token_stride, token_stride, int(head_dim))
    scale_offset = int(kv_heads) * int(head_dim)
    launch = PagedForwardKernel.selected_positions(q_heads=int(query_heads), kv_heads=int(kv_heads),
        kv_is_fp8=fp8, direct_output=True, kv_warps=2, page_size=int(page_rows),
        key_strides=strides, value_strides=strides, selection_width=int(selection_width),
        head_dim=int(head_dim), selection_groups=int(selection_groups),
        inline_scale_offsets=(scale_offset, scale_offset) if fp8 else None)
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    cache_shape = (f"[num_cache_pages,{page_rows},{token_stride}]" if fp8 else
                   f"[num_cache_pages,{page_rows},{kv_heads},{head_dim}]")
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{query_heads},{head_dim}]"),
        Operand("key_cache", cache_dtype, cache_shape),
        Operand("value_cache", cache_dtype, cache_shape),
        Operand("k_descale", torch.float32, "ignored: BF16 or inline token/head descales"),
        Operand("v_descale", torch.float32, "ignored: BF16 or inline token/head descales"),
        Operand("block_table", torch.int32, "[table_batch,table_width]", align=4),
        Operand("request_ids", torch.int32, "[rows]", align=4),
        Operand("selected_positions", torch.int32,
                f"[rows,{selection_groups},{selection_width}]", align=4),
        Operand("query_positions", torch.int64, "[rows]", align=8),
        Operand("partial_output", torch.float32, "ignored for direct output"),
        Operand("partial_lse", torch.float32, "ignored for direct output"),
        Operand("output", torch.bfloat16, f"[rows,{query_heads},{head_dim}]", "out"))
    scalars = (Scalar("num_cache_pages", "int64"), Scalar("table_batch", "int64"),
               Scalar("table_width", "int64"), Scalar("softmax_scale", "float32"),
               Scalar("rows"), Scalar("splits", note="one for direct output"))
    return compile_program(launch, name="selected_paged_gqa", operands=operands, scalars=scalars,
        key=(int(query_heads), int(kv_heads), int(page_rows), int(selection_width),
             int(selection_groups), int(head_dim), kv_format),
        geometry=dict(query_heads=int(query_heads), kv_heads=int(kv_heads), page_rows=int(page_rows),
                      selection_width=int(selection_width), selection_groups=int(selection_groups),
                      head_dim=int(head_dim), key_strides=strides, value_strides=strides,
                      kv_format=kv_format, record_bytes=token_stride * (1 if fp8 else 2)))
