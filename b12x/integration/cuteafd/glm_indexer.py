"""Native AOT GLM 5.x DSA index top-k (``attention.dsa_indexer``, 32 heads).

``compile_glm_index_topk_aot(g, max_rows=R, max_pages=P, mode=...)`` is the
DeepSeek V4 index top-k program (``dsv4_indexer``: the route the prepared
``dsa_indexer`` plan selects for this capacity) at GLM's 32 index heads and
top-2048, over the index cache ``glm_index_producer`` writes. The index
cache and the latent cache share page ids (64 rows per page), so the output
physical slots ``page*64 + row`` address the latent records directly.

ABI (``rows <= max_rows``)::

    q_fp8          fp8  [rows,32,128]       in   glm_index_producer q_fp8
    weights        f32  [rows,32]           in   glm_index_producer head_weights
    index_k_cache  u8   [pool_pages,8448]   in   64 x 128 E4M3 keys, then 64 FP32 scales
    page_table     i32  prefill: [table_width] (one sequence, shared by every row)
                        decode:  [rows, table_stride] (first table_width columns used)
    cache_lengths  i32  [rows]              in   index rows visible to each query row
                                                 (causal: position + 1)
    output_indices i32  [rows,2048]         out  physical slots, -1 padded
    scratch        u8   [index_topk_scratch_bytes]  zero-filled once at allocation
    rows           int32
    table_width    int32  live page-table columns, 1 <= table_width <= max_pages
    table_stride   int32  decode: int32 elements between page-table rows; prefill: ignored

Row ``i`` keeps the top 2048 of ``sum_h relu(q_h . k_j * k_scale_j) * w_h``
over ``j < cache_lengths[i]``; when ``cache_lengths[i] <= 2048`` it selects
every visible row. Shared (``indexer_types == "shared"``) layers reuse the
previous full layer's ``output_indices``.
"""

from __future__ import annotations

import torch

from ._common import GLM53, GLMGeometry, Operand, Scalar, compile_program

__all__ = ["compile_glm_index_topk_aot", "glm_index_topk_scratch_bytes"]


def glm_index_topk_scratch_bytes(g: GLMGeometry = GLM53, *, max_rows: int, max_pages: int, mode: str) -> int:
    from .dsv4_indexer import _prepared_layout

    return int(_prepared_layout(g, max_rows, max_pages, mode, g.index_heads).nbytes)


def compile_glm_index_topk_aot(g: GLMGeometry = GLM53, *, max_rows: int, max_pages: int,
                               mode: str = "prefill"):
    """GLM index top-k for ``rows <= max_rows`` and ``table_width <= max_pages``."""
    from .dsv4_indexer import _TopK

    if mode not in ("prefill", "decode"):
        raise ValueError("mode must be prefill or decode")
    if int(max_rows) <= 0 or int(max_pages) <= 0:
        raise ValueError("max_rows and max_pages must be positive")
    launch = _TopK(g, int(max_rows), int(max_pages), mode, heads=g.index_heads)
    k, i = launch.topk, g.index_heads
    table_shape = "[table_width]" if mode == "prefill" else "[rows,table_stride]"
    operands = (
        Operand("q_fp8", torch.float8_e4m3fn, f"[rows,{i},128]"),
        Operand("weights", torch.float32, f"[rows,{i}]"),
        Operand("index_k_cache", torch.uint8, f"[pool_pages,{g.index_page_bytes}]"),
        Operand("page_table", torch.int32, table_shape, align=4),
        Operand("cache_lengths", torch.int32, "[rows]", align=4),
        Operand("output_indices", torch.int32, f"[rows,{k}]", "out", align=4),
        Operand("scratch", torch.uint8, "[index_topk_scratch_bytes]", "scratch"),
    )
    nbytes = int(launch.layout.nbytes)
    return compile_program(
        launch, name=f"glm_index_topk_{mode}", operands=operands,
        scalars=(Scalar("rows"), Scalar("table_width"), Scalar("table_stride")),
        key=launch.key(),
        geometry={"topk": k, "heads": i, "max_rows": int(max_rows), "max_pages": int(max_pages),
                  "mode": mode, "route": launch.route, "supertile_tokens": launch.supertile},
        scratch={"scratch": lambda rows: nbytes},
        doc=__doc__,
    )
