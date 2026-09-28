"""CuTe port of the shared-page-table supertile gather (packed-contiguous route).

Equivalent to the Triton ``_gather_shared_paged_supertile_kernel`` in
``paged.py``: for one supertile chunk starting at page-table column
``page_begin`` it copies ``supertile_tokens`` index-K rows (128 FP8 bytes plus
the 4-byte FP32 scale) from the paged FP8 index cache into contiguous scratch,
zero-filling rows whose column is past ``table_width`` or whose page id is
negative, and writes per-query-row local bounds ``k_start = 0`` and
``k_end = clamp(length - page_begin * 64, 0, supertile_tokens)``.

The page table is row-shared: only its first row (``table_width`` int32 page
ids) is read. Pool offsets are 64-bit.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

_PAGE = 64
_HEAD_DIM = 128
_PAGE_BYTES = _PAGE * (_HEAD_DIM + 4)
_DATA_BYTES = _PAGE * _HEAD_DIM
_THREADS = 256
_TOKENS_PER_CTA = _THREADS // 32  # one warp copies one 128-byte row + scale


class SharedSupertileGather:
    def __init__(self, *, supertile_tokens: int):
        self.supertile = int(supertile_tokens)

    @cute.jit
    def __call__(self, index_k_cache: cute.Pointer, page_table: cute.Pointer,
                 lengths: cute.Pointer, k_quant: cute.Pointer, k_scale: cute.Pointer,
                 k_start: cute.Pointer, k_end: cute.Pointer, rows: Int32,
                 table_width: Int32, page_begin: Int32, stream: cuda.CUstream):
        token_ctas = (self.supertile + _TOKENS_PER_CTA - 1) // _TOKENS_PER_CTA
        row_ctas = (rows + Int32(_THREADS - 1)) // Int32(_THREADS)
        grid = Int32(token_ctas)
        if row_ctas > grid:
            grid = row_ctas
        self.kernel(index_k_cache, page_table, lengths, k_quant, k_scale, k_start, k_end,
                    rows, table_width, page_begin).launch(
            grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, index_k_cache: cute.Pointer, page_table: cute.Pointer,
               lengths: cute.Pointer, k_quant: cute.Pointer, k_scale: cute.Pointer,
               k_start: cute.Pointer, k_end: cute.Pointer, rows: Int32,
               table_width: Int32, page_begin: Int32):
        bidx = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        cache_base = Int64(index_k_cache.toint())
        token = bidx * Int32(_TOKENS_PER_CTA) + warp
        if token < Int32(self.supertile):
            column = page_begin + token // Int32(_PAGE)
            slot = token % Int32(_PAGE)
            page_id = Int32(-1)
            if column < table_width:
                table = cute.make_ptr(Int32, Int64(page_table.toint()), cute.AddressSpace.gmem,
                                      assumed_align=4)
                page_id = Int32(table[column])
            word = Uint32(0)
            scale = Uint32(0)
            if page_id >= Int32(0):
                page_base = cache_base + Int64(page_id) * Int64(_PAGE_BYTES)
                src = cute.make_ptr(Uint32, page_base + Int64(slot) * Int64(_HEAD_DIM),
                                    cute.AddressSpace.gmem, assumed_align=4)
                word = Uint32(src[lane])
                if lane == Int32(0):
                    src_scale = cute.make_ptr(Uint32, page_base + Int64(_DATA_BYTES) + Int64(slot) * Int64(4),
                                              cute.AddressSpace.gmem, assumed_align=4)
                    scale = Uint32(src_scale[0])
            dst = cute.make_ptr(Uint32, Int64(k_quant.toint()) + Int64(token) * Int64(_HEAD_DIM),
                                cute.AddressSpace.gmem, assumed_align=4)
            dst[lane] = word
            if lane == Int32(0):
                dst_scale = cute.make_ptr(Uint32, Int64(k_scale.toint()) + Int64(token) * Int64(4),
                                          cute.AddressSpace.gmem, assumed_align=4)
                dst_scale[0] = scale
        row = bidx * Int32(_THREADS) + tidx
        if row < rows:
            src_len = cute.make_ptr(Int32, Int64(lengths.toint()), cute.AddressSpace.gmem, assumed_align=4)
            local = Int32(src_len[row]) - page_begin * Int32(_PAGE)
            if local < Int32(0):
                local = Int32(0)
            if local > Int32(self.supertile):
                local = Int32(self.supertile)
            starts = cute.make_ptr(Int32, Int64(k_start.toint()), cute.AddressSpace.gmem, assumed_align=4)
            ends = cute.make_ptr(Int32, Int64(k_end.toint()), cute.AddressSpace.gmem, assumed_align=4)
            starts[row] = Int32(0)
            ends[row] = local


class WriteActiveWidth:
    """One-thread prologue: active_width[0] = table_width * 64 (paged scorer cap)."""

    @cute.jit
    def __call__(self, active_width: cute.Pointer, table_width: Int32, stream: cuda.CUstream):
        self.kernel(active_width, table_width).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, active_width: cute.Pointer, table_width: Int32):
        if Int32(cute.arch.thread_idx()[0]) == Int32(0):
            active_width[0] = table_width * Int32(_PAGE)


__all__ = ["SharedSupertileGather", "WriteActiveWidth"]
