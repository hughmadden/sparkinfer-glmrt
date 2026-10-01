"""Native AOT C4 index top-k (``attention.dsa_indexer``, physical-slot output).

``compile_dsv4_index_topk_aot(geometry, max_rows=R, max_pages=P, mode=...)``
compiles exactly the route the prepared ``dsa_indexer`` plan selects for
``Caps(num_q_heads=64, max_q_rows=R, max_page_table_width=P,
topk=geometry.index_topk, mode=mode, output_index_space="physical")``:

* ``mode="prefill"`` (row-shared page table, one sequence): the
  packed-contiguous route. Per supertile chunk of the page table, a CuTe
  gather (port of the Triton ``_gather_shared_paged_supertile_kernel``)
  packs the chunk's index-K rows contiguously, the contiguous FP8 prefill
  scorer writes tiled logits, and the tiled radix top-k folds the chunk into
  a ping-pong carry; the final chunk emits physical slots.
* ``mode="decode"`` (per-row page tables): the fused score+select kernel
  when the plan chooses it (Flash: up to 16 rows, Pro: up to 6), otherwise
  the paged streaming supertile scorer + the same tiled top-k fold.

ABI (``rows <= max_rows`` live query rows, ``K`` = geometry.index_topk)::

    q_fp8          fp8  [rows,64,128]       in   index query (index producer output)
    weights        f32  [rows,64]           in   learned head weights
    index_k_cache  u8   [pool_pages,8448]   in   64 rows x 128 FP8, then 64 FP32 scales
    page_table     i32  prefill: [table_width] (shared by every row)
                        decode:  [rows, table_stride] (first table_width columns used)
    cache_lengths  i32  [rows]              in   live index rows visible to each query row
    output_indices i32  [rows,K]            out  physical slots (page*64+row), -1 padded
    scratch        u8   [index_topk_scratch_bytes(...)]  zero-filled once at allocation
    rows           int32
    table_width    int32  live page-table columns, 1 <= table_width <= max_pages
    table_stride   int32  decode: int32 elements between page-table rows; prefill: ignored

Selection semantics are the prepared path's: row ``i`` scores index rows
``[0, cache_lengths[i])`` of its page table, keeps the top ``K`` by
``sum_h relu(q_h . k) * w_h`` and pads with ``-1``. The scratch layout is
the prepared plan's layout for the same capacity (``max_rows``,
``max_pages``); zero it once when allocating (the fused route's cross-CTA
merge state must start zeroed and restores itself after every launch).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from b12x.attention.dsa_indexer._gather_cute import SharedSupertileGather, WriteActiveWidth

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program

__all__ = ["compile_dsv4_index_topk_aot", "index_topk_route", "index_topk_scratch_bytes"]

_HEADS = 64
_HEAD_DIM = 128
_PAGE = 64
_PAGE_BYTES = 8448
_DATA_BYTES = _PAGE * _HEAD_DIM
_TILE_BLOCK_Q = 32  # paged.py _PAGED_INDEX_TILE_BLOCK_Q
_PAGED_TILE_BLOCK_K = 512  # paged.py _PAGED_INDEX_TILE_BLOCK_K


def _descriptor(shape, dtype, strides=None):
    if strides is None:
        strides, step = [], 1
        for dim in reversed(shape):
            strides.append(step)
            step *= dim
        strides = tuple(reversed(strides))
    return {"shape": tuple(shape), "strides": tuple(strides), "dtype": dtype, "alignment": 16}


def _prepared_layout(geometry: DSV4Geometry, max_rows: int, max_pages: int, mode: str,
                     heads: int = _HEADS):
    """The scratch layout (and route) of the prepared plan for this capacity."""
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer import _preparation as prep
    from b12x.attention.dsa_indexer._tuning import TUNING
    from b12x.attention.dsa_indexer.scratch import plan_indexer_scratch
    from b12x.preparation import detect_device

    device = torch.device("cuda", torch.cuda.current_device())
    rows, pages, topk = int(max_rows), int(max_pages), int(geometry.index_topk)
    caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=rows,
                            max_page_table_width=pages, topk=topk, mode=mode,
                            output_index_space="physical")
    operands = dict(
        q_fp8=_descriptor((rows, heads, _HEAD_DIM), "float8_e4m3fn"),
        query_weights=_descriptor((rows, heads), "float32"),
        index_k_cache=_descriptor((pages, _PAGE_BYTES), "uint8"),
        page_table=_descriptor((rows, pages), "int32", (0, 1) if mode == "prefill" else None),
        cache_lengths=_descriptor((rows,), "int32"),
        active_width=_descriptor((1,), "int32"),
        output_indices=_descriptor((rows, topk), "int32"),
        output_scores=None,
    )
    query = prep._query(caps, prep.invocation_from_descriptors(caps, operands=operands))
    configuration = TUNING.configure(query, device=detect_device(device).identity)
    config = configuration.default if configuration.pinned is None else configuration.pinned
    plan = plan_indexer_scratch(prep._scratch_caps(query, device=device, config=config),
                                fused_merge=config.fused_merge)
    return plan.layout


def index_topk_route(geometry: DSV4Geometry, *, max_rows: int, max_pages: int, mode: str) -> str:
    return _prepared_layout(geometry, max_rows, max_pages, mode).route


def index_topk_scratch_bytes(geometry: DSV4Geometry, *, max_rows: int, max_pages: int,
                             mode: str) -> int:
    """Bytes of the ``scratch`` pointer (the prepared plan's scratch spec).

    Regions (each 1024-byte aligned, see dsa_indexer.scratch): gathered K
    ``S*128`` + scales ``S*4`` (prefill only, ``S`` = supertile tokens),
    k_start/k_end ``2 * R * 4``, tiled logits
    ``ceil(R/32) * (S/BK) * 32 * BK * 4``, top-k values/indices ``2 * R*K*4``,
    carry ping-pong ``2 * 2 * R*K*4`` (when more than one chunk), active width
    ``4``, and for the fused decode route the merge pack
    ``2 * num_sms*K*4`` and merge state ``R * 8968 * 4``.
    """
    return int(_prepared_layout(geometry, max_rows, max_pages, mode).nbytes)


class _TopK:
    """``heads`` index heads (64 for DeepSeek V4, 32 for GLM 5.x)."""

    def __init__(self, geometry: DSV4Geometry, max_rows: int, max_pages: int, mode: str,
                 heads: int = _HEADS):
        from b12x.attention.dsa_indexer.tiled_topk import (
            _build_tiled_topk_kernel,
            _resolve_smem_candidate_capacity,
            deterministic_topk,
        )

        self.heads = int(heads)
        layout = _prepared_layout(geometry, max_rows, max_pages, mode, self.heads)
        self.layout = layout
        self.route = layout.route
        self.mode = mode
        self.topk = int(geometry.index_topk)
        self.max_rows = int(max_rows)
        self.max_pages = int(max_pages)
        self.num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.supertile = int(layout.supertile_tokens)
        self.supertile_pages = max(1, self.supertile // _PAGE)
        if self.route == "paged_fused":
            from b12x.attention.dsa_indexer.fused_indexer import KV_LAYOUT_PAGED, _build_fused_indexer_kernel

            self.fused = _build_fused_indexer_kernel(
                KV_LAYOUT_PAGED, self.heads, self.topk, True, int(layout.fused_ctas_per_group),
                num_sms=self.num_sms, merge_threshold=int(layout.fused_merge_threshold),
                k_quant_page_stride=_PAGE_BYTES, k_scales_row_stride=_PAGE_BYTES // 4,
                max_seq_capacity=self.max_pages * _PAGE, vectorized_q_load=True,
                q_row_stride_bytes=self.heads * _HEAD_DIM,
            )
            return
        if self.route == "packed_contiguous":
            from b12x.attention.dsa_indexer.contiguous_kernel import (
                _PREFILL512_BLOCK_K,
                _build_dsa_contiguous_prefill512_kernel,
                _build_dsa_contiguous_prefill_kernel,
            )

            self.block_k = int(layout.prefill_block_k)
            self.prefill512 = self.block_k == _PREFILL512_BLOCK_K
            self.scorer = (
                _build_dsa_contiguous_prefill512_kernel(tiled_output=True, num_heads=self.heads)
                if self.prefill512 else _build_dsa_contiguous_prefill_kernel(tiled_output=True)
            )
            self.gather = SharedSupertileGather(supertile_tokens=self.supertile)
        elif self.route == "paged_tiled":
            from b12x.attention.dsa_indexer.kernel import _build_dsa_paged_stream_supertile_kernel

            if not layout.stream_scorer:
                raise NotImplementedError("the non-streaming paged scorer is not exported")
            self.block_k = _PAGED_TILE_BLOCK_K
            self.scorer = _build_dsa_paged_stream_supertile_kernel(
                int(layout.stream_scorer_ctas), self.heads, _TILE_BLOCK_Q, _PAGED_TILE_BLOCK_K,
                _PAGE_BYTES, _PAGE_BYTES // 4)
            self.active_width = WriteActiveWidth()
        else:
            raise NotImplementedError(f"unsupported indexer route {self.route!r}")
        self.k_tiles = self.supertile // self.block_k
        capacity = _resolve_smem_candidate_capacity(topk=self.topk)
        # (is_first, physical output); non-final chunks keep values for the carry.
        # Deterministic select (index tie-break, canonical ascending output) unless
        # B12X_DSA_TOPK_DETERMINISTIC=0 (A/B builds).
        self.deterministic = deterministic_topk()
        self.topk_kernels = {
            (first, phys): _build_tiled_topk_kernel(
                _TILE_BLOCK_Q, self.block_k, self.topk, True, first, phys, 1, capacity, not phys,
                self.deterministic)
            for first in (True, False) for phys in (True, False)
        }

    def key(self) -> tuple:
        L = self.layout
        key = (self.route, self.topk, self.max_rows, self.max_pages, self.supertile,
               L.prefill_block_k, L.fused_ctas_per_group, L.fused_merge_threshold,
               L.stream_scorer_ctas, L.max_chunks, L.nbytes, self.num_sms)
        key = key if self.heads == _HEADS else key + (self.heads,)
        return key if getattr(self, "deterministic", True) else key + ("arrival-order",)

    # -- helpers ------------------------------------------------------------
    @cute.jit
    def _ptr(self, dtype: cutlass.Constexpr, base: Int64, offset: cutlass.Constexpr, align: cutlass.Constexpr = 16):
        return cute.make_ptr(dtype, base + Int64(offset), cute.AddressSpace.gmem, assumed_align=align)

    @cute.jit
    def _flat(self, dtype: cutlass.Constexpr, base: Int64, offset: cutlass.Constexpr, size, align: cutlass.Constexpr = 4):
        return cute.make_tensor(self._ptr(dtype, base, offset, align), cute.make_layout((size,)))

    @cute.jit
    def __call__(self, q_fp8: cute.Pointer, weights: cute.Pointer, index_k_cache: cute.Pointer,
                 page_table: cute.Pointer, cache_lengths: cute.Pointer,
                 output_indices: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_width: Int32, table_stride: Int32, stream: cuda.CUstream):
        L = self.layout
        base = Int64(scratch.toint())
        cache = Int64(index_k_cache.toint())
        m = rows
        q_bytes = cute.make_tensor(
            cute.make_ptr(cutlass.Uint8, Int64(q_fp8.toint()), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, self.heads, _HEAD_DIM), stride=(self.heads * _HEAD_DIM, _HEAD_DIM, 1)))
        w = cute.make_tensor(weights, cute.make_layout((m, self.heads), stride=(self.heads, 1)))
        lengths = cute.make_tensor(cache_lengths, cute.make_layout((m,)))
        topk = self.topk
        out = cute.make_tensor(output_indices, cute.make_layout((m, topk), stride=(topk, 1)))
        if const_expr(self.route == "paged_fused"):
            k_quant = cute.make_tensor(
                cute.make_ptr(cutlass.Uint8, cache, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((Int32(1 << 30), _PAGE, _HEAD_DIM), stride=(_PAGE_BYTES, _HEAD_DIM, 1)))
            k_scales = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, cache + Int64(_DATA_BYTES), cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((Int32(1 << 30), _PAGE), stride=(_PAGE_BYTES // 4, 1)))
            table = cute.make_tensor(page_table, cute.make_layout((m, table_width), stride=(table_stride, 1)))
            values = cute.make_tensor(self._ptr(cutlass.Float32, base, L.topk_values_offset_bytes),
                                      cute.make_layout((m, topk), stride=(topk, 1)))
            pack_values = self._flat(cutlass.Float32, base, L.fused_pack_values_offset_bytes, L.fused_pack_elements)
            pack_indices = self._flat(cutlass.Int32, base, L.fused_pack_indices_offset_bytes, L.fused_pack_elements)
            state = self._flat(cutlass.Int32, base, L.fused_merge_state_offset_bytes, L.fused_state_words)
            self.fused(q_bytes, w, k_quant, k_scales, table, lengths, lengths, lengths, out, values,
                       pack_values, pack_indices, state, stream)
        else:
            self._chunked(q_fp8, w, cache, q_bytes, page_table, lengths, cache_lengths,
                          output_indices, base, rows, table_width, table_stride, stream)

    @cute.jit
    def _chunked(self, q_fp8: cute.Pointer, w: cute.Tensor, cache: Int64, q_bytes: cute.Tensor,
                 page_table: cute.Pointer, lengths: cute.Tensor, cache_lengths: cute.Pointer,
                 output_indices: cute.Pointer, base: Int64, rows: Int32, table_width: Int32,
                 table_stride: Int32, stream: cuda.CUstream):
        L = self.layout
        topk = self.topk
        m = rows
        tile_logits = self._flat(cutlass.Float32, base, L.tile_logits_offset_bytes, L.tile_logits_elements)
        live = Int32(m) * Int32(topk)
        final_values = cute.make_tensor(self._ptr(cutlass.Float32, base, L.topk_values_offset_bytes),
                                        cute.make_layout((live,)))
        final_indices = cute.make_tensor(output_indices, cute.make_layout((live,)))
        half = self.max_rows * topk
        carry_v0 = cute.make_tensor(self._ptr(cutlass.Float32, base, L.candidate_values_offset_bytes), cute.make_layout((live,)))
        carry_v1 = cute.make_tensor(self._ptr(cutlass.Float32, base, L.candidate_values_offset_bytes + half * 4), cute.make_layout((live,)))
        carry_i0 = cute.make_tensor(self._ptr(cutlass.Int32, base, L.candidate_indices_offset_bytes), cute.make_layout((live,)))
        carry_i1 = cute.make_tensor(self._ptr(cutlass.Int32, base, L.candidate_indices_offset_bytes + half * 4), cute.make_layout((live,)))
        if const_expr(self.route == "packed_contiguous"):
            # Physical output rows all read the shared first page-table row.
            table_flat = cute.make_tensor(page_table, cute.make_layout((table_width,)))
            table_row_stride = Int32(0)
        else:
            table_flat = cute.make_tensor(page_table, cute.make_layout((m * table_stride,)))
            table_row_stride = table_stride
            self.active_width(self._ptr(cutlass.Int32, base, L.active_width_offset_bytes, 4), table_width, stream)
        dummy_desc = cute.make_tensor(self._ptr(cutlass.Int64, base, L.active_width_offset_bytes, 8), cute.make_layout((1,)))
        chunks = (table_width + Int32(self.supertile_pages - 1)) // Int32(self.supertile_pages)
        chunk = Int32(0)
        while chunk < chunks:
            page_begin = chunk * Int32(self.supertile_pages)
            pages_here = table_width - page_begin
            if pages_here > Int32(self.supertile_pages):
                pages_here = Int32(self.supertile_pages)
            token_begin = page_begin * Int32(_PAGE)
            tokens_here = pages_here * Int32(_PAGE)
            self._score(q_fp8, w, cache, q_bytes, page_table, lengths, cache_lengths, base,
                        rows, table_width, table_stride, page_begin, tile_logits, dummy_desc, stream)
            is_first = chunk == Int32(0)
            is_last = chunk == chunks - Int32(1)
            odd = (chunk % Int32(2)) == Int32(1)
            if is_first:
                if is_last:
                    self._fold(True, True, tile_logits, lengths, final_values, final_indices,
                               final_values, final_indices, table_flat, table_row_stride, rows,
                               token_begin, tokens_here, stream)
                else:
                    self._fold(True, False, tile_logits, lengths, carry_v0, carry_i0,
                               carry_v0, carry_i0, table_flat, table_row_stride, rows,
                               token_begin, tokens_here, stream)
            else:
                if is_last:
                    if odd:
                        self._fold(False, True, tile_logits, lengths, final_values, final_indices,
                                   carry_v0, carry_i0, table_flat, table_row_stride, rows,
                                   token_begin, tokens_here, stream)
                    else:
                        self._fold(False, True, tile_logits, lengths, final_values, final_indices,
                                   carry_v1, carry_i1, table_flat, table_row_stride, rows,
                                   token_begin, tokens_here, stream)
                else:
                    if odd:
                        self._fold(False, False, tile_logits, lengths, carry_v1, carry_i1,
                                   carry_v0, carry_i0, table_flat, table_row_stride, rows,
                                   token_begin, tokens_here, stream)
                    else:
                        self._fold(False, False, tile_logits, lengths, carry_v0, carry_i0,
                                   carry_v1, carry_i1, table_flat, table_row_stride, rows,
                                   token_begin, tokens_here, stream)
            chunk = chunk + Int32(1)

    @cute.jit
    def _score(self, q_fp8: cute.Pointer, w: cute.Tensor, cache: Int64, q_bytes: cute.Tensor,
               page_table: cute.Pointer, lengths: cute.Tensor, cache_lengths: cute.Pointer,
               base: Int64, rows: Int32, table_width: Int32, table_stride: Int32,
               page_begin: Int32, tile_logits: cute.Tensor, dummy_desc: cute.Tensor,
               stream: cuda.CUstream):
        L = self.layout
        m = rows
        if const_expr(self.route == "packed_contiguous"):
            s = self.supertile
            self.gather(
                cute.make_ptr(cutlass.Uint8, cache, cute.AddressSpace.gmem, assumed_align=16),
                page_table, cache_lengths,
                self._ptr(cutlass.Uint8, base, L.gather_k_quant_offset_bytes),
                self._ptr(cutlass.Uint8, base, L.gather_k_scale_offset_bytes),
                self._ptr(cutlass.Int32, base, L.contiguous_lengths_offset_bytes, 4),
                self._ptr(cutlass.Int32, base, L.runtime_lengths_offset_bytes, 4),
                rows, table_width, page_begin, stream)
            q_u32 = cute.make_tensor(
                cute.make_ptr(cutlass.Uint32, Int64(q_fp8.toint()), cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((m, self.heads, _HEAD_DIM // 4), stride=(self.heads * _HEAD_DIM // 4, _HEAD_DIM // 4, 1)))
            k_quant = cute.make_tensor(self._ptr(cutlass.Uint8, base, L.gather_k_quant_offset_bytes),
                                       cute.make_layout((s, _HEAD_DIM), stride=(_HEAD_DIM, 1)))
            k_scale = self._flat(cutlass.Float32, base, L.gather_k_scale_offset_bytes, s)
            k_start = cute.make_tensor(self._ptr(cutlass.Int32, base, L.contiguous_lengths_offset_bytes, 4),
                                       cute.make_layout((m,)))
            k_end = cute.make_tensor(self._ptr(cutlass.Int32, base, L.runtime_lengths_offset_bytes, 4),
                                     cute.make_layout((m,)))
            dummy_out = cute.make_tensor(self._ptr(cutlass.Float32, base, L.topk_values_offset_bytes),
                                         cute.make_layout((1, 1), stride=(1, 1)))
            if const_expr(self.prefill512):
                self.scorer(q_u32, w, k_quant, dummy_desc, k_scale, k_start, k_end, dummy_out,
                            tile_logits, rows, Int32(s), Int32(0), Int32(self.k_tiles), Int32(0), stream)
            else:
                dummy_blocks = cute.make_tensor(self._ptr(cutlass.Float32, base, L.topk_values_offset_bytes),
                                                cute.make_layout((1, 1, 1), stride=(1, 1, 1)))
                self.scorer(q_u32, w, k_quant, dummy_desc, k_scale, k_start, k_end, dummy_out,
                            tile_logits, dummy_blocks, rows, Int32(s), Int32(0), Int32(0),
                            Int32(self.k_tiles), Int32(0), stream)
        else:
            k_quant = cute.make_tensor(
                cute.make_ptr(cutlass.Uint8, cache, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((Int32(1 << 30), _PAGE, _HEAD_DIM), stride=(_PAGE_BYTES, _HEAD_DIM, 1)))
            k_scales = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, cache + Int64(_DATA_BYTES), cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((Int32(1 << 30), _PAGE), stride=(_PAGE_BYTES // 4, 1)))
            table = cute.make_tensor(page_table, cute.make_layout((m, table_width), stride=(table_stride, 1)))
            active = cute.make_tensor(self._ptr(cutlass.Int32, base, L.active_width_offset_bytes, 4),
                                      cute.make_layout((1,)))
            scalar_flag = active
            self.scorer(q_bytes, w, k_quant, dummy_desc, scalar_flag, k_scales, table, lengths,
                        active, page_begin, Int32(self.supertile), tile_logits, stream)

    @cute.jit
    def _fold(self, first: cutlass.Constexpr, phys: cutlass.Constexpr, tile_logits: cute.Tensor,
              lengths: cute.Tensor, values: cute.Tensor, indices: cute.Tensor,
              carry_values: cute.Tensor, carry_indices: cute.Tensor, table_flat: cute.Tensor,
              table_row_stride: Int32, rows: Int32, token_begin: Int32, tokens_here: Int32,
              stream: cuda.CUstream):
        kernel = self.topk_kernels[(first, phys)]
        if const_expr(phys):
            kernel(tile_logits, lengths, lengths, values, indices, carry_values, carry_indices,
                   table_flat, table_row_stride, rows, Int32(0), Int32(self.k_tiles), Int32(0),
                   Int32(_TILE_BLOCK_Q), Int32(self.block_k), Int32(self.topk), token_begin,
                   tokens_here, token_begin, Int32(_PAGE), Int32(1), Int32(0), stream)
        else:
            kernel(tile_logits, lengths, lengths, values, indices, carry_values, carry_indices,
                   lengths, Int32(0), rows, Int32(0), Int32(self.k_tiles), Int32(0),
                   Int32(_TILE_BLOCK_Q), Int32(self.block_k), Int32(self.topk), token_begin,
                   tokens_here, token_begin, Int32(1), Int32(1), Int32(0), stream)


def compile_dsv4_index_topk_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int, max_pages: int,
                                mode: str = "prefill"):
    """C4 index top-k for ``rows <= max_rows`` and ``table_width <= max_pages``."""
    if mode not in ("prefill", "decode"):
        raise ValueError("mode must be prefill or decode")
    if int(max_rows) <= 0 or int(max_pages) <= 0:
        raise ValueError("max_rows and max_pages must be positive")
    launch = _TopK(geometry, int(max_rows), int(max_pages), mode)
    k = launch.topk
    table_shape = "[table_width]" if mode == "prefill" else "[rows,table_stride]"
    operands = (
        Operand("q_fp8", torch.float8_e4m3fn, "[rows,64,128]"),
        Operand("weights", torch.float32, "[rows,64]"),
        Operand("index_k_cache", torch.uint8, "[pool_pages,8448]"),
        Operand("page_table", torch.int32, table_shape, align=4),
        Operand("cache_lengths", torch.int32, "[rows]", align=4),
        Operand("output_indices", torch.int32, f"[rows,{k}]", "out", align=4),
        Operand("scratch", torch.uint8, "[index_topk_scratch_bytes]", "scratch"),
    )
    nbytes = int(launch.layout.nbytes)
    return compile_program(
        launch, name=f"dsv4_index_topk_{mode}", operands=operands,
        scalars=(Scalar("rows"), Scalar("table_width"), Scalar("table_stride")),
        key=launch.key(),
        geometry={"topk": k, "max_rows": int(max_rows), "max_pages": int(max_pages), "mode": mode,
                  "route": launch.route, "supertile_tokens": launch.supertile},
        scratch={"scratch": lambda rows: nbytes},
        doc=__doc__,
    )
