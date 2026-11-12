"""Bounded paged index-cache staging shared by MiniMax, GLM and Kimi.

Cache records use the shared 64 x 128 E4M3 bytes + 64 FP32 scales ABI.
The gather copies raw words without dequantization or scale conversion.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from b12x.attention.dsa_indexer._gather_cute import SharedSupertileGather
from ._common import Operand, Scalar, compile_program


class _IndexCacheGather:
    def __init__(self):
        self.gather = SharedSupertileGather(supertile_tokens=256)

    @cute.jit
    def __call__(self, cache: cute.Pointer, pages: cute.Pointer, lengths: cute.Pointer,
                 keys: cute.Pointer, scales: cute.Pointer, starts: cute.Pointer,
                 ends: cute.Pointer, rows: Int32, table_width: Int32, page_begin: Int32,
                 pool_pages: Int64, live_tokens: Int32, output_tokens: Int32,
                 stream: cuda.CUstream):
        self.gather.bounded(cache, pages, lengths, keys, scales, starts, ends,
                            rows, table_width, page_begin, pool_pages, Int64(live_tokens),
                            output_tokens, stream)


def compile_index_cache_gather_aot():
    """Stage one request's causal prefix into an admitted contiguous owner.

    Scalars must satisfy 0 <= live_tokens <= output_tokens, output_tokens is a
    positive multiple of 256, and all host extents match the scalar capacities.
    Negative or out-of-pool page IDs and out-of-table columns are zero-filled;
    the padded suffix is always cleared even when its pages contain old keys.
    Lengths produce local bounds clamped to [0, live_tokens]. page_begin may
    select an earlier shared-table chunk; live_tokens is local to that chunk.
    All input/output storage and the stream must survive queued work and replay.
    """
    return compile_program(_IndexCacheGather(), name="index_cache_gather", operands=(
        Operand("cache", torch.uint8, "[pool_pages,8448]"),
        Operand("pages", torch.int32, "[table_width]", align=4),
        Operand("lengths", torch.int32, "[rows]", align=4),
        Operand("keys", torch.uint8, "[output_tokens,128]", "out"),
        Operand("scales", torch.float32, "[output_tokens]", "out", align=4),
        Operand("starts", torch.int32, "[rows]", "out", align=4),
        Operand("ends", torch.int32, "[rows]", "out", align=4)),
        scalars=(Scalar("rows"), Scalar("table_width"), Scalar("page_begin"),
                 Scalar("pool_pages", "int64"), Scalar("live_tokens"), Scalar("output_tokens")),
        key=("bounded-shared-index-cache",),
        geometry=dict(page_rows=64, head_dim=128, page_bytes=8448, padding_multiple=256),
        doc=compile_index_cache_gather_aot.__doc__)
