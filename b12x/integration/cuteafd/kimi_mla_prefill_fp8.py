"""Kimi K3 MLA prefill attention over compact FP8 latent records (AOT program).

The extend route of ``kimi_mla`` for ``kv_cache="fp8"``: one request, causal,
no split, reading the 656-byte GLM-format record natively (E4M3 latent widened
exactly into BF16 tensor-core math, FP32 group scales applied in FP32).
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
import torch

from b12x.attention.dense_mla._prefill_fp8 import RECORD_BYTES, DenseMlaFp8PrefillKernel
from ._common import Operand, Scalar, compile_program

# (chunk, stages, query_tile, prefetch, min_blocks).  Two query rows share
# each gathered 64-record chunk: the measured best of eight variants at
# h48/h96, rows 1408/4096, prefixes 0..256K (see the dense_mla learnings).
DEFAULT_FP8_PREFILL_TILE = (64, 1, 2, 0, 1)


class _Fp8PrefillAttention:
    def __init__(self, g, tile):
        self.g = g
        chunk, stages, query_tile, prefetch, min_blocks = tile
        self.kernel = DenseMlaFp8PrefillKernel(
            num_heads=g.heads, page_size=64, chunk=chunk, stages=stages,
            query_tile=query_tile, prefetch=prefetch, min_blocks=min_blocks,
        )

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                 cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                 lse: cute.Pointer, rows: Int32, batch: Int32, page_table_width: Int64,
                 stream: cuda.CUstream):
        self.run(query, kv_cache, page_table, cache_seqlens, cu_seqlens_q, attn, lse,
                 rows, batch, page_table_width, stream)

    @cute.jit
    def run(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
            cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
            lse: cute.Pointer, rows: Int32, batch: Int32, page_table_width: Int64,
            stream: cuda.CUstream):
        g = self.g
        q = cute.make_tensor(cute.recast_ptr(query, dtype=cutlass.Uint8), cute.make_layout((1,)))
        kv = cute.make_tensor(cute.recast_ptr(kv_cache, dtype=cutlass.Uint8), cute.make_layout((1,)))
        table = cute.make_tensor(page_table, cute.make_layout((Int64(batch) * page_table_width,)))
        lengths = cute.make_tensor(cache_seqlens, cute.make_layout((batch,)))
        cu = cute.make_tensor(cu_seqlens_q, cute.make_layout((batch + Int32(1),)))
        out = cute.make_tensor(attn, cute.make_layout((rows, g.heads, 512), stride=(g.heads * 512, 512, 1)))
        lse_out = cute.make_tensor(lse, cute.make_layout((rows, g.heads), stride=(g.heads, 1)))
        self.kernel(q, kv, table, lengths, cu, out, lse_out,
                    Float32(g.softmax_scale * math.log2(math.e)),
                    Int64(g.heads * 576 * 2), Int64(64 * RECORD_BYTES), page_table_width,
                    rows, batch, stream)


class _Fp8PrefillAsDenseMla(_Fp8PrefillAttention):
    """The ``kimi_dense_mla`` extend ABI: scratch and active_splits are unused
    (the route never splits), so the engine's call site is unchanged."""

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                 cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                 lse: cute.Pointer, scratch: cute.Pointer, rows: Int32, batch: Int32,
                 page_table_width: Int64, active_splits: Int32, stream: cuda.CUstream):
        self.run(query, kv_cache, page_table, cache_seqlens, cu_seqlens_q, attn, lse,
                 rows, batch, page_table_width, stream)


def compile_kimi_fp8_prefill_aot(g, *, max_rows=4096, tile=DEFAULT_FP8_PREFILL_TILE):
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand("kv_cache", torch.uint8, "[pages,64,656]"),
        Operand("page_table", torch.int32, "[batch,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[batch]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out"))
    return compile_program(_Fp8PrefillAttention(g, tuple(tile)), name="kimi_mla_fp8_prefill",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64")),
        key=(g, max_rows, tuple(tile)),
        geometry={**vars(g), "max_rows": max_rows, "tile": tuple(tile)})


__all__ = ["DEFAULT_FP8_PREFILL_TILE", "_Fp8PrefillAsDenseMla", "compile_kimi_fp8_prefill_aot"]
