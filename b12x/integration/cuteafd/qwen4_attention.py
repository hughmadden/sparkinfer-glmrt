"""Native AOT Qwen 3.8 Flash Next (``qwen4_exp``) full-attention programs, family ``qwen4``.

Twelve layers (every fourth) run GQA with the QSA indexer and a sigmoid
output gate (see :class:`Qwen4Geometry`): ``H`` hidden 2560, ``N`` 24 query
heads and ``G`` 2 KV heads of 256, ``I`` 4 index heads of 128, blocks of 4
tokens, a budget of 512 blocks (every token up to 2051 visible tokens). All
weights are BF16 as the checkpoints store them. ``rows`` is the live row
count (``rows <= max_rows``); scratch regions are laid out from it (size
with ``rows = max_rows``), each 1024-byte aligned.

Caches (64-row pages; a token's record slot is ``page * 64 + position % 64``,
so a 4-token block's slots are consecutive and block-aligned):

* ``kv_cache``    u8 ``[pages, 64 * 2048]``: per token K ``[2, 256]`` BF16
  (after ``k_norm`` and RoPE) then V ``[2, 256]`` BF16;
* ``token_keys``  BF16 ``[pages * 64, 128]``: the raw (pre-norm) index key of
  every token at its record slot;
* ``index_cache`` BF16 ``[pool_pages * 64, 128]``: one key per complete block
  (``pool_slot = pool_page * 64 + (position // 4) % 64``; 64 blocks = 256
  tokens per pool page): ``rope(k_layernorm(bf16(mean(raw keys))), first
  position of the block)``.

``compile_qwen4_attn_producer_aot(g, max_rows=R)``::

    x           bf16 [rows,H]            in   attn_hyper_connection mixed input
    w_in        bf16 [13952,H]           in   cat(q_proj (per head 256 q then 256 gate rows), k_proj,
                                              v_proj, indexer.index_qk_proj (4x128 q then 128 key))
    q_norm      bf16 [256]               in   self_attn.q_norm.weight  (all norms: (1 + w), eps 1e-6)
    k_norm      bf16 [256]               in   self_attn.k_norm.weight
    iq_norm     bf16 [128]               in   indexer.q_layernorm.weight
    ik_norm     bf16 [128]               in   indexer.k_layernorm.weight
    positions   i64  [rows]              in   token position (RoPE; theta 1e7 on dims 0:64, NeoX halves)
    kv_slots    i64  [rows]              in   record slot (<0 skips every cache write of the row)
    pool_slots  i64  [rows]              in   index_cache slot of the block the row completes
                                              (position % 4 == 3), else -1
    kv_cache    u8   [pages,131072]      inout
    token_keys  bf16 [record_slots,128]  inout
    index_cache bf16 [pool_slots,128]    inout
    query       bf16 [rows,N,256]        out  q_norm + RoPE
    gate        bf16 [rows,N*256]        out  the output gate (pre-sigmoid)
    index_q     bf16 [rows,I,128]        out  q_layernorm + RoPE
    scratch     u8   attn_producer_scratch_bytes(rows): in-projection BF16 [rows, 13952]
    rows        int32

The block keys are built after every row's raw key is written, so a block
completed inside a step may use keys of earlier rows of the same step (its
other tokens must be in the cache already or in this step).

``compile_qwen4_index_topk_aot(g, max_rows=R, max_context=C)`` (only when some
row sees more than 2051 tokens)::

    index_q     bf16 [rows,I,128]        in   producer index_q
    positions   i64  [rows]              in   (the row sees blocks 0 .. (position + 1) // 4 - 1)
    index_cache bf16 [pool_slots,128]    in
    page_table  i32  [rows|1,stride]     in   pool pages of the row's sequence (row r's table at
                                              r * table_stride; table_stride 0: every row reads row 0's)
    output_indices i32 [rows,512]        out  selected block ids (logical, ascending), -1 padded
    scratch     u8   index_topk_scratch_bytes(rows)
    rows, table_stride int32

Score per block: ``sum_h relu(q_h . k) / sqrt(128)`` in FP32 over BF16 inputs
(b12x ``qsa`` representative scorer, tensor cores); exact radix top-512, ties
to the lower block id. ``max_context`` bounds the blocks a row may see.

``compile_qwen4_index_expand_aot(g)``::

    positions   i64  [rows]              in
    blocks      i32  [rows,512]          in   output_indices (ignored for rows that see <= 2051 tokens)
    indices     i32  [rows,2112]         out  selected logical positions, -1 padded
    lengths     i32  [rows]              out  selected count
    rows        int32

``compile_qwen4_sparse_gqa_aot(g, max_rows=R)``::

    query       bf16 [rows,N,256]        in
    kv_cache    u8   [pages,131072]      in
    positions   i64  [rows]              in
    page_table  i32  [rows|1,stride]     in   KV pages of the row's sequence (as above)
    indices     i32  [rows,2112]         in   index_expand indices
    out         bf16 [rows,N*256]        out  softmax(q.k / 16) . v over the selected tokens
    scratch     u8   sparse_gqa_scratch_bytes(rows)
    rows, table_width, table_stride int32   table_width: pages in the shared table (stride 0);
                                            with table_stride > 0 each row reads table_stride entries

b12x's selected-position paged GQA (``attention.paged._selected_forward``:
FP32 scores and online softmax, BF16 probabilities, FP32 PV); rows <= 64 split
every row's selection over 64/32/16 CTAs (1 / <=4 / <=64 rows) and merge the
FP32 partials, more rows write BF16 directly.

``compile_qwen4_attn_o_aot(g, max_rows=R)``::

    attn        bf16 [rows,N*256]        in   sparse_gqa out
    gate        bf16 [rows,N*256]        in   producer gate
    w_o         bf16 [H,N*256]           in   o_proj
    out         bf16 [rows,H]            out  attention output (before the hyper-connection injection)
    scratch     u8   attn_o_scratch_bytes(rows): gated BF16 [rows, N*256]
    rows        int32
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ._fp8_weights import fp8_operands, projection
from ._common import QWEN38_FLASH_NEXT, AotProgram, Operand, Qwen4Geometry, Scalar, compile_program
from ._glm_kernels import glm_projection
from ._qwen4_attention_kernels import (
    Qwen4AttnPost,
    Qwen4BlockTopK,
    Qwen4GateMul,
    Qwen4IndexExpand,
    Qwen4PoolKeys,
    Qwen4RowTables,
)

__all__ = [
    "attn_o_scratch_bytes",
    "attn_producer_scratch_bytes",
    "compile_qwen4_attn_o_aot",
    "compile_qwen4_attn_producer_aot",
    "compile_qwen4_index_expand_aot",
    "compile_qwen4_index_topk_aot",
    "compile_qwen4_sparse_gqa_aot",
    "index_topk_scratch_bytes",
    "sparse_gqa_scratch_bytes",
    "split_count",
]

_ALIGN = 1024
SPLIT_MAX_ROWS = 64
TOPK_CHUNK_ROWS = 256


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _ptr(dtype, address, align: int = 16):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _check_rows(max_rows: int) -> int:
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(max_rows)


def split_count(rows: int) -> int:
    """Splits per row of the split (``rows <= 64``) attention route."""
    rows = int(rows)
    return 64 if rows == 1 else 32 if rows <= 4 else 16


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


def attn_producer_scratch_bytes(g: Qwen4Geometry, rows: int) -> int:
    return _align(max(int(rows), 1) * g.attn_in_width * 2)


class _Producer:
    def __init__(self, g: Qwen4Geometry, fp8: bool = False):
        self.g, self.fp8 = g, bool(fp8)
        self.proj = projection(g.attn_in_width, g.hidden, self.fp8)
        self.post = Qwen4AttnPost(g)
        self.pool = Qwen4PoolKeys(g)

    def key(self) -> tuple:
        return (self.proj.key(), self.g, self.fp8)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, q_norm: cute.Pointer, k_norm: cute.Pointer,
                 iq_norm: cute.Pointer, ik_norm: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer,
                 pool_slots: cute.Pointer, kv_cache: cute.Pointer, token_keys: cute.Pointer,
                 index_cache: cute.Pointer, query: cute.Pointer, gate: cute.Pointer, index_q: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in, w_in, q_norm, k_norm, iq_norm, ik_norm, positions, kv_slots, pool_slots, kv_cache,
                  token_keys, index_cache, query, gate, index_q, scratch, rows, stream)

    @cute.jit
    def body(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
             q_norm: cute.Pointer, k_norm: cute.Pointer, iq_norm: cute.Pointer, ik_norm: cute.Pointer,
             positions: cute.Pointer, kv_slots: cute.Pointer, pool_slots: cute.Pointer, kv_cache: cute.Pointer,
             token_keys: cute.Pointer, index_cache: cute.Pointer, query: cute.Pointer, gate: cute.Pointer,
             index_q: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        proj = _ptr(cutlass.BFloat16, Int64(scratch.toint()))
        self.proj(x, w_in, w_in_fp8, w_in_scale, proj, rows, stream)
        self.post(proj, q_norm, k_norm, iq_norm, ik_norm, positions, kv_slots, kv_cache, token_keys, query, gate,
                  index_q, rows, stream)
        self.pool(positions, kv_slots, pool_slots, ik_norm, token_keys, index_cache, rows, stream)


class _ProducerFp8(_Producer):
    def __init__(self, g: Qwen4Geometry):
        super().__init__(g, fp8=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
                 q_norm: cute.Pointer, k_norm: cute.Pointer, iq_norm: cute.Pointer, ik_norm: cute.Pointer,
                 positions: cute.Pointer, kv_slots: cute.Pointer, pool_slots: cute.Pointer, kv_cache: cute.Pointer,
                 token_keys: cute.Pointer, index_cache: cute.Pointer, query: cute.Pointer, gate: cute.Pointer,
                 index_q: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in_fp8, w_in_scale, q_norm, k_norm, iq_norm, ik_norm, positions, kv_slots, pool_slots,
                  kv_cache, token_keys, index_cache, query, gate, index_q, scratch, rows, stream)


def compile_qwen4_attn_producer_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, max_rows: int,
                                    fp8: bool = False) -> AotProgram:
    """Full-attention producer for ``rows <= max_rows``; see the module docstring.
    ``fp8`` adds ``w_in_fp8``/``w_in_scale`` (E4M3, FP32 128x128 block scales) for rows <= 16."""
    max_rows = _check_rows(max_rows)
    launch = (_ProducerFp8 if fp8 else _Producer)(g)
    h, n, d, ih = g.hidden, g.heads, g.head_dim, g.index_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_in", torch.bfloat16, f"[{g.attn_in_width},{h}]"),
        *(fp8_operands("w_in", g.attn_in_width, h) if fp8 else ()),
        Operand("q_norm", torch.bfloat16, f"[{d}]"),
        Operand("k_norm", torch.bfloat16, f"[{d}]"),
        Operand("iq_norm", torch.bfloat16, f"[{g.index_head_dim}]"),
        Operand("ik_norm", torch.bfloat16, f"[{g.index_head_dim}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("pool_slots", torch.int64, "[rows]", align=8),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
        Operand("token_keys", torch.bfloat16, f"[record_slots,{g.index_head_dim}]", "inout"),
        Operand("index_cache", torch.bfloat16, f"[pool_slots,{g.index_head_dim}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{d}]", "out"),
        Operand("gate", torch.bfloat16, f"[rows,{n * d}]", "out"),
        Operand("index_q", torch.bfloat16, f"[rows,{ih},{g.index_head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[attn_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="qwen4_attn_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "heads": n, "kv_heads": g.kv_heads, "head_dim": d, "index_heads": ih,
                  "rope_dim": g.rope_dim, "rope_theta": g.rope_theta, "record_bytes": g.record_bytes,
                  "page_rows": g.page_rows, "eps": g.norm_eps, "max_rows": max_rows, "fp8_weights": fp8},
        scratch={"scratch": lambda rows: attn_producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# QSA block top-k and expansion
# ---------------------------------------------------------------------------


def _topk_layout(g: Qwen4Geometry, max_rows: int, max_context: int) -> dict[str, int]:
    rows = max(int(max_rows), 1)
    chunk = min(rows, TOPK_CHUNK_ROWS)
    groups = int(max_context) // g.index_block
    offsets, at = {}, 0
    for name, nbytes in (("requests", rows * 4), ("lengths", rows * 4), ("counts", chunk * 4),
                         ("merges", chunk * 4), ("scores", chunk * groups * 4)):
        offsets[name] = at
        at += _align(nbytes)
    offsets["nbytes"] = at
    offsets["chunk"] = chunk
    offsets["groups"] = groups
    return offsets


def index_topk_scratch_bytes(g: Qwen4Geometry, rows: int, max_context: int) -> int:
    """requests/lengths i32 [rows], counts/merges i32 [chunk], FP32 scores [chunk, max_context / 4]
    (``chunk = min(rows, 256)``)."""
    return _topk_layout(g, rows, max_context)["nbytes"]


class _IndexTopK:
    def __init__(self, g: Qwen4Geometry, max_rows: int, max_context: int):
        from b12x.attention.qsa._score_cute import _RepresentativeScoreKernel

        self.g = g
        self.layout = _topk_layout(g, max_rows, max_context)
        groups = self.layout["groups"]
        self.score = _RepresentativeScoreKernel(g.index_heads, g.index_head_dim, g.index_block, g.page_rows,
                                                groups, g.index_blocks)
        self.topk = Qwen4BlockTopK(k=g.index_blocks, block=g.index_block, max_groups=groups)
        self.tables = Qwen4RowTables()

    def key(self) -> tuple:
        return (tuple(sorted(self.layout.items())), self.g)

    @cute.jit
    def __call__(self, index_q: cute.Pointer, positions: cute.Pointer, index_cache: cute.Pointer,
                 page_table: cute.Pointer, output_indices: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_stride: Int32, stream: cuda.CUstream):
        L = self.layout
        g = self.g
        base = Int64(scratch.toint())
        requests = _ptr(Int32, base + Int64(L["requests"]), 4)
        lengths = _ptr(Int32, base + Int64(L["lengths"]), 4)
        counts = _ptr(Int32, base + Int64(L["counts"]), 4)
        merges = _ptr(Int32, base + Int64(L["merges"]), 4)
        scores = _ptr(Float32, base + Int64(L["scores"]), 4)
        self.tables(positions, requests, lengths, rows, Int32(0), stream)
        q_row = g.index_heads * g.index_head_dim * 2
        start = Int32(0)
        while start < rows:
            n = cutlass.min(Int32(L["chunk"]), rows - start)
            s64 = Int64(start)
            q = _ptr(cutlass.BFloat16, Int64(index_q.toint()) + s64 * Int64(q_row))
            pos = _ptr(Int64, Int64(positions.toint()) + s64 * Int64(8), 8)
            req = _ptr(Int32, Int64(requests.toint()) + s64 * Int64(4), 4)
            out = _ptr(Int32, Int64(output_indices.toint()) + s64 * Int64(g.index_blocks * 4), 4)
            self.score((q, pos, req, lengths, index_cache, page_table, scores, counts, merges),
                       (Int64(g.page_rows * g.index_head_dim), Int64(g.index_head_dim), Int64(table_stride),
                        Int64(L["groups"])),
                       n, Int32(0), Int32(L["groups"]), stream)
            self.topk(pos, scores, out, n, Int64(L["groups"]), stream)
            start = start + n


def compile_qwen4_index_topk_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, max_rows: int,
                                 max_context: int = 65536) -> AotProgram:
    """QSA top-512 blocks per row; see the module docstring."""
    max_rows = _check_rows(max_rows)
    if int(max_context) % (g.index_block * g.page_rows):
        raise ValueError("max_context must be a multiple of 256")
    launch = _IndexTopK(g, max_rows, int(max_context))
    ih, dd = g.index_heads, g.index_head_dim
    operands = (
        Operand("index_q", torch.bfloat16, f"[rows,{ih},{dd}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("index_cache", torch.bfloat16, f"[pool_slots,{dd}]"),
        Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4),
        Operand("output_indices", torch.int32, f"[rows,{g.index_blocks}]", "out", align=4),
        Operand("scratch", torch.uint8, "[index_topk_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="qwen4_index_topk", operands=operands,
        scalars=(Scalar("rows"), Scalar("table_stride")), key=(max_rows, launch.key()),
        geometry={"index_heads": ih, "index_head_dim": dd, "blocks": g.index_blocks, "block": g.index_block,
                  "max_rows": max_rows, "max_context": int(max_context), "chunk_rows": launch.layout["chunk"]},
        scratch={"scratch": lambda rows: index_topk_scratch_bytes(g, rows, max_context)},
        doc=__doc__,
    )


def compile_qwen4_index_expand_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """Selected blocks (or every token up to 2051) to logical positions plus the open tail."""
    launch = Qwen4IndexExpand(pools=g.index_blocks, width=g.sparse_topk, dense_limit=g.dense_limit,
                              kpool=g.index_block)
    return compile_program(
        launch, name="qwen4_index_expand",
        operands=(Operand("positions", torch.int64, "[rows]", align=8),
                  Operand("blocks", torch.int32, f"[rows,{g.index_blocks}]", align=4),
                  Operand("indices", torch.int32, f"[rows,{g.sparse_topk}]", "out", align=4),
                  Operand("lengths", torch.int32, "[rows]", "out", align=4)),
        scalars=(Scalar("rows"),), key=(g.index_blocks, g.sparse_topk, g.dense_limit),
        geometry={"blocks": g.index_blocks, "width": g.sparse_topk, "dense_limit": g.dense_limit},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# Sparse GQA
# ---------------------------------------------------------------------------


def _gqa_layout(g: Qwen4Geometry, max_rows: int) -> dict[str, int]:
    rows = max(int(max_rows), 1)
    split_rows = min(rows, SPLIT_MAX_ROWS)
    row_splits = max(r * split_count(r) for r in range(1, split_rows + 1))
    offsets, at = {}, 0
    for name, nbytes in (("requests", rows * 4), ("lengths", rows * 4),
                         ("partial", row_splits * g.heads * g.head_dim * 4), ("lse", row_splits * g.heads * 4)):
        offsets[name] = at
        at += _align(nbytes)
    offsets["nbytes"] = at
    return offsets


def sparse_gqa_scratch_bytes(g: Qwen4Geometry, rows: int) -> int:
    """requests/lengths i32 [rows], then FP32 split partials [rows * splits, N, 256] and LSE
    [rows * splits, N] for the split route (at most 1024 row-splits)."""
    return _gqa_layout(g, rows)["nbytes"]


class _SparseGqa:
    def __init__(self, g: Qwen4Geometry, max_rows: int, direct_kv_warps: int = 2, split_kv_warps: int = 2):
        from b12x.attention.paged._selected_forward import _ShapeAdaptiveSparseGqaMergeKernel
        from b12x.attention.paged.forward_extend_generic import PagedForwardKernel

        self.g = g
        self.max_rows = int(max_rows)
        self.layout = _gqa_layout(g, max_rows)
        token = g.record_bytes // 2
        strides = (g.page_rows * token, token, g.head_dim)
        common = dict(q_heads=g.heads, kv_heads=g.kv_heads, kv_is_fp8=False, page_size=g.page_rows,
                      key_strides=strides, value_strides=strides, selection_width=g.sparse_topk)
        self.kv_warps = (int(split_kv_warps), int(direct_kv_warps))
        self.split = PagedForwardKernel.selected_positions(direct_output=False, kv_warps=self.kv_warps[0], **common)
        self.direct = (PagedForwardKernel.selected_positions(direct_output=True, kv_warps=self.kv_warps[1], **common)
                       if self.max_rows > SPLIT_MAX_ROWS else None)
        self.merge = _ShapeAdaptiveSparseGqaMergeKernel(q_heads=g.heads)
        self.tables = Qwen4RowTables()

    def key(self) -> tuple:
        return (tuple(sorted(self.layout.items())), self.max_rows, self.kv_warps, self.g)

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer,
                 page_table: cute.Pointer, indices: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, table_width: Int32, table_stride: Int32, stream: cuda.CUstream):
        g = self.g
        L = self.layout
        base = Int64(scratch.toint())
        requests = _ptr(Int32, base + Int64(L["requests"]), 4)
        lengths = _ptr(Int32, base + Int64(L["lengths"]), 4)
        partial = _ptr(Float32, base + Int64(L["partial"]), 16)
        lse = _ptr(Float32, base + Int64(L["lse"]), 16)
        shared = Int32(0)
        width = table_stride
        if table_stride == Int32(0):
            shared = Int32(1)
            width = table_width
        self.tables(positions, requests, lengths, rows, shared, stream)
        keys = _ptr(cutlass.BFloat16, Int64(kv_cache.toint()))
        values = _ptr(cutlass.BFloat16, Int64(kv_cache.toint()) + Int64(g.kv_heads * g.head_dim * 2))
        dummy = _ptr(Float32, base + Int64(L["lse"]), 16)
        scale = Float32(g.softmax_scale)
        pages = Int64(1 << 40)
        if cutlass.const_expr(self.direct is None):
            splits = self._splits(rows)
            self.split(query, keys, values, dummy, dummy, page_table, requests, indices, positions, partial, lse,
                       out, pages, Int64(rows), Int64(width), scale, rows, splits, stream)
            self.merge(partial, lse, out, rows, splits, stream)
        else:
            if rows <= Int32(SPLIT_MAX_ROWS):
                splits = self._splits(rows)
                self.split(query, keys, values, dummy, dummy, page_table, requests, indices, positions, partial,
                           lse, out, pages, Int64(rows), Int64(width), scale, rows, splits, stream)
                self.merge(partial, lse, out, rows, splits, stream)
            else:
                self.direct(query, keys, values, dummy, dummy, page_table, requests, indices, positions, partial,
                            lse, out, pages, Int64(rows), Int64(width), scale, rows, Int32(1), stream)

    @cute.jit
    def _splits(self, rows: Int32) -> Int32:
        splits = Int32(16)
        if rows <= Int32(4):
            splits = Int32(32)
        if rows == Int32(1):
            splits = Int32(64)
        return splits


def compile_qwen4_sparse_gqa_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, max_rows: int,
                                 direct_kv_warps: int = 2, split_kv_warps: int = 2) -> AotProgram:
    """GQA over each row's selected positions; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _SparseGqa(g, max_rows, direct_kv_warps, split_kv_warps)
    n, d = g.heads, g.head_dim
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{n},{d}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4),
        Operand("indices", torch.int32, f"[rows,{g.sparse_topk}]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{n * d}]", "out"),
        Operand("scratch", torch.uint8, "[sparse_gqa_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="qwen4_sparse_gqa", operands=operands,
        scalars=(Scalar("rows"), Scalar("table_width"), Scalar("table_stride")), key=(max_rows, launch.key()),
        geometry={"heads": n, "kv_heads": g.kv_heads, "head_dim": d, "width": g.sparse_topk,
                  "softmax_scale": g.softmax_scale, "max_rows": max_rows, "split_max_rows": SPLIT_MAX_ROWS,
                  "kv_warps": list(launch.kv_warps)},
        scratch={"scratch": lambda rows: sparse_gqa_scratch_bytes(g, rows)},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# Gated output projection
# ---------------------------------------------------------------------------


def attn_o_scratch_bytes(g: Qwen4Geometry, rows: int) -> int:
    return _align(max(int(rows), 1) * g.heads * g.head_dim * 2)


class _AttnO:
    def __init__(self, g: Qwen4Geometry, fp8: bool = False):
        self.g, self.fp8 = g, bool(fp8)
        self.gate = Qwen4GateMul(g.heads * g.head_dim)
        self.o_proj = projection(g.hidden, g.heads * g.head_dim, self.fp8)

    def key(self) -> tuple:
        return (self.o_proj.key(), self.g, self.fp8)

    @cute.jit
    def __call__(self, attn: cute.Pointer, gate: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(attn, gate, w_o, w_o, w_o, out, scratch, rows, stream)

    @cute.jit
    def body(self, attn: cute.Pointer, gate: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
             w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        gated = _ptr(cutlass.BFloat16, Int64(scratch.toint()))
        self.gate(attn, gate, gated, rows, stream)
        self.o_proj(gated, w_o, w_o_fp8, w_o_scale, out, rows, stream)


class _AttnOFp8(_AttnO):
    def __init__(self, g: Qwen4Geometry):
        super().__init__(g, fp8=True)

    @cute.jit
    def __call__(self, attn: cute.Pointer, gate: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
                 w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(attn, gate, w_o, w_o_fp8, w_o_scale, out, scratch, rows, stream)


def compile_qwen4_attn_o_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, max_rows: int,
                             fp8: bool = False) -> AotProgram:
    """``o_proj(attn * sigmoid(gate))``; see the module docstring. ``fp8`` adds
    ``w_o_fp8``/``w_o_scale`` for rows <= 16."""
    max_rows = _check_rows(max_rows)
    launch = (_AttnOFp8 if fp8 else _AttnO)(g)
    h, w = g.hidden, g.heads * g.head_dim
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{w}]"),
        Operand("gate", torch.bfloat16, f"[rows,{w}]"),
        Operand("w_o", torch.bfloat16, f"[{h},{w}]"),
        *(fp8_operands("w_o", h, w) if fp8 else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[attn_o_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="qwen4_attn_o", operands=operands, scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": h, "width": w, "max_rows": max_rows, "fp8_weights": fp8},
        scratch={"scratch": lambda rows: attn_o_scratch_bytes(g, rows)},
        doc=__doc__,
    )
