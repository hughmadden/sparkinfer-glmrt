"""Native AOT MiMo V2 attention programs: QKV producer, GQA attention, o_proj.

Two attention kinds (``kind``): ``full`` (64 query / 4 KV heads, causal over
a paged cache, no sink, RoPE theta 5e6) and ``swa`` (64 / 8 heads, 128-token
window, learned per-head sink, theta 1e4). ``H`` hidden 4096, ``N`` 64 query
heads, ``G`` KV heads of the kind, ``R = G * 320`` BF16 elements per KV record
(keys ``[G, 192]`` then values ``[G, 128]``, see ``_mimo_kernels``). Weights
are BF16 (the checkpoint's FP8 blocks times their FP32 scales, dequantized at
load; ``o_proj`` is stored BF16). Projections are ``RoutedBf16Projection``
(skinny GEMV for few rows, TMA tensor-core GEMM above; FP32 accumulation).
Scratch is laid out from the live row count (size it with ``rows =
max_rows``); regions start 1024-byte aligned.

``compile_mimo_producer_aot(g, kind=K, max_rows=M)``::

    x           bf16 [rows,H]         in     input_layernorm output
    positions   i64  [rows]           in     RoPE position per row
    kv_slots    i64  [rows]           in     record slot per row (<0 skips): full = page*64+row
                                             of the paged cache; swa = the row's index in kv_step
    cos_sin     f32  [P,64]           in     cos(32) | sin(32) of theta^(-2i/64) (theta of the kind)
    w_qkv       bf16 [N*192+R,H]      in     cat(q_proj, k_proj, v_proj)
    kv_cache    bf16 [slots,R]        inout  record at kv_slots[row] (full: paged cache; swa: kv_step)
    query       bf16 [rows,N,192]     out    RoPE on dims 0:64 of each head
    scratch     u8   producer_scratch_bytes(kind, rows): qkv BF16 [rows, N*192+R]
    rows        int32

``compile_mimo_attention_aot(g, kind="full", route=..., max_rows=M)``::

    q           bf16 [rows,N,192]     in     producer query
    kv_cache    bf16 [pages*64,R]     in     paged records (slot page*64+row)
    positions   i64  [rows]           in
    page_table  i32  [rows|1,stride]  in     row r's pages at page_table[r*table_stride]
    out         bf16 [rows,N,128]     out
    scratch     u8   attention_scratch_bytes(...)
    rows, table_stride int32 (0: every row shares row 0's table, the prefill layout)
    splits      int32 (decode route only, 1..max_splits)

``compile_mimo_attention_aot(g, kind="swa", route=..., max_rows=M)``::

    q           bf16 [rows,N,192]     in     producer query
    kv_step     bf16 [rows,R]         in     this step's records (producer kv_slots = row index)
    ring        bf16 [rings*256,R]    inout  per-sequence rings; step rows are committed after
                                             attention (the last 256 of each sequence's step)
    positions   i64  [rows]           in
    ring_slots  i64  [rows]           in     ring*256 + position % 256
    seq_first   i32  [rows]           in     index of the first step row of the row's sequence
    sinks       bf16 [N]              in     self_attn.attention_sink_bias
    out         bf16 [rows,N,128]     out
    scratch     u8   attention_scratch_bytes(...)   (unused; kept for a uniform ABI)
    rows        int32

Rows of one sequence are consecutive with consecutive positions. Keys at or
past the sequence's first step position come from ``kv_step``, older ones
from the ring, so a step may be longer than the ring. The ``prefill`` route
packs several rows per CTA and requires one sequence per launch; the
``decode`` route takes any mix (one row per CTA). Full decode splits every
row's key range over ``splits`` CTAs (FP32 partials + LSE, then a merge);
scratch holds ``partials f32 [rows, S, N, 128]`` then (1024-aligned)
``lse f32 [rows, S, N]`` for ``S = max_splits``.

Softmax: ``softmax(q.k * 192^-0.5 [; sink]) . v`` (the sink is a raw extra
logit that takes probability mass but no value), FP32 scores, BF16
probabilities, FP32 PV accumulation.

``compile_mimo_o_aot(g, max_rows=M)``::

    attn        bf16 [rows,N*128]     in     attention output (heads major)
    w_o         bf16 [H,N*128]        in     o_proj (BF16 in the checkpoint)
    out         bf16 [rows,H]         out    before the residual add
    rows        int32
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import MIMO_V2_FLASH, MiMoGeometry, Operand, Scalar, compile_program
from ._glm_kernels import glm_projection
from ._mimo_kernels import MAX_SPLITS, MimoGqaAttention, MimoQkvRope, MimoRingCommit, MimoSplitMerge

__all__ = [
    "attention_scratch_bytes",
    "compile_mimo_attention_aot",
    "compile_mimo_o_aot",
    "compile_mimo_producer_aot",
    "producer_scratch_bytes",
    "prefill_tokens",
]

_ALIGN = 1024
DEFAULT_MAX_SPLITS = 32


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _check(kind: str, max_rows: int) -> int:
    if kind not in ("full", "swa"):
        raise ValueError("kind is 'full' or 'swa'")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(max_rows)


def producer_scratch_bytes(g: MiMoGeometry, kind: str, rows: int) -> int:
    return _align(max(int(rows), 1) * g.qkv_width(kind) * 2)


def attention_scratch_bytes(g: MiMoGeometry, kind: str, route: str, rows: int,
                            max_splits: int = DEFAULT_MAX_SPLITS) -> int:
    rows = max(int(rows), 1)
    if kind == "full" and route == "decode":
        return _align(rows * max_splits * g.heads * g.v_head_dim * 4) + _align(rows * max_splits * g.heads * 4)
    return _ALIGN


def prefill_tokens(g: MiMoGeometry, kind: str) -> int:
    """Rows per CTA of the prefill route: a 64-row M tile of (token, head)."""
    return 64 // (g.heads // g.kv_heads(kind))


class _Producer:
    def __init__(self, g: MiMoGeometry, kind: str):
        self.g, self.kind = g, kind
        self.qkv = glm_projection(g.qkv_width(kind), g.hidden)
        self.post = MimoQkvRope(heads=g.heads, kv_heads=g.kv_heads(kind), v_scale=g.v_scale,
                                head=g.qk_head_dim, v_head=g.v_head_dim)

    def key(self) -> tuple:
        return (self.qkv.key(), self.kind, self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        qkv = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.qkv(x, w_qkv, qkv, rows, stream)
        self.post(qkv, positions, cos_sin, kv_slots, query, kv_cache, rows, stream)


def compile_mimo_producer_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, kind: str, max_rows: int):
    """QKV projection, partial RoPE and the KV record write; see the module docstring."""
    max_rows = _check(kind, max_rows)
    launch = _Producer(g, kind)
    h, n, r = g.hidden, g.heads, g.record_elems(kind)
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_qkv", torch.bfloat16, f"[{g.qkv_width(kind)},{h}]"),
        Operand("kv_cache", torch.bfloat16, f"[slots,{r}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.qk_head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"mimo_{kind}_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"kind": kind, "hidden": h, "heads": n, "kv_heads": g.kv_heads(kind), "record_elems": r,
                  "max_rows": max_rows, "v_scale": g.v_scale, "rope_theta": g.rope_theta(kind)},
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, kind, rows)},
        doc=__doc__,
    )


class _FullAttention:
    def __init__(self, g: MiMoGeometry, route: str, max_splits: int):
        self.g, self.route = g, route
        decode = route == "decode"
        self.attn = MimoGqaAttention(
            heads=g.heads, kv_heads=g.full_kv_heads, tokens=1 if decode else prefill_tokens(g, "full"),
            window=0, paged=True, sink=False, direct=not decode, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, ring_rows=g.ring_rows, head=g.qk_head_dim, v_head=g.v_head_dim)
        self.merge = MimoSplitMerge(heads=g.heads, v_head=g.v_head_dim, sink=False) if decode else None
        self.max_splits = int(max_splits)

    def key(self) -> tuple:
        return (self.attn.key(), self.route, self.max_splits)

    @cute.jit
    def run(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
            out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32, splits: Int32,
            stream: cuda.CUstream):
        base = Int64(scratch.toint())
        partials = cute.make_ptr(cutlass.Float32, base, cute.AddressSpace.gmem, assumed_align=16)
        lse_off = base + (Int64(rows) * Int64(self.max_splits * self.g.heads * self.g.v_head_dim * 4)
                          + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        lse = cute.make_ptr(cutlass.Float32, lse_off, cute.AddressSpace.gmem, assumed_align=16)
        # Unused SWA operands take any valid address.
        self.attn(q, kv_cache, kv_cache, positions, page_table, page_table, page_table, q, out, partials, lse,
                  rows, table_stride, splits, stream)
        if cutlass.const_expr(self.merge is not None):
            self.merge(partials, lse, q, out, rows, splits, stream)


class _FullAttentionDecode(_FullAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32, splits: Int32,
                 stream: cuda.CUstream):
        clamped = splits
        if clamped > Int32(self.max_splits):
            clamped = Int32(self.max_splits)
        if clamped < Int32(1):
            clamped = Int32(1)
        self.run(q, kv_cache, positions, page_table, out, scratch, rows, table_stride, clamped, stream)


class _FullAttentionPrefill(_FullAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32,
                 stream: cuda.CUstream):
        self.run(q, kv_cache, positions, page_table, out, scratch, rows, table_stride, Int32(1), stream)


class _SwaAttention:
    def __init__(self, g: MiMoGeometry, route: str):
        self.g, self.route = g, route
        self.attn = MimoGqaAttention(
            heads=g.heads, kv_heads=g.swa_kv_heads, tokens=1 if route == "decode" else prefill_tokens(g, "swa"),
            window=g.window, paged=False, sink=True, direct=True, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, ring_rows=g.ring_rows, head=g.qk_head_dim, v_head=g.v_head_dim)
        self.commit = MimoRingCommit(record=g.record_elems("swa"), ring_rows=g.ring_rows)

    def key(self) -> tuple:
        return (self.attn.key(), self.route)

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_step: cute.Pointer, ring: cute.Pointer, positions: cute.Pointer,
                 ring_slots: cute.Pointer, seq_first: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(cutlass.Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        i32 = cute.make_ptr(cutlass.Int32, Int64(seq_first.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.attn(q, ring, kv_step, positions, i32, ring_slots, seq_first, sinks, out, f32, f32, rows, Int32(0),
                  Int32(1), stream)
        self.commit(kv_step, ring, ring_slots, seq_first, rows, stream)


def compile_mimo_attention_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, kind: str, route: str, max_rows: int,
                               max_splits: int = DEFAULT_MAX_SPLITS):
    """GQA attention of ``kind`` for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check(kind, max_rows)
    if route not in ("decode", "prefill"):
        raise ValueError("route is 'decode' or 'prefill'")
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    n, r = g.heads, g.record_elems(kind)
    q = Operand("q", torch.bfloat16, f"[rows,{n},{g.qk_head_dim}]")
    out = Operand("out", torch.bfloat16, f"[rows,{n},{g.v_head_dim}]", "out")
    scratch = Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch")
    if kind == "full":
        launch = (_FullAttentionDecode if route == "decode" else _FullAttentionPrefill)(g, route, max_splits)
        operands = (q, Operand("kv_cache", torch.bfloat16, f"[pages*{g.page_rows},{r}]"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4), out, scratch)
        scalars = (Scalar("rows"), Scalar("table_stride"))
        if route == "decode":
            scalars += (Scalar("splits"),)
    else:
        launch = _SwaAttention(g, route)
        operands = (q, Operand("kv_step", torch.bfloat16, f"[rows,{r}]"),
                    Operand("ring", torch.bfloat16, f"[rings*{g.ring_rows},{r}]", "inout"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("ring_slots", torch.int64, "[rows]", align=8),
                    Operand("seq_first", torch.int32, "[rows]", align=4),
                    Operand("sinks", torch.bfloat16, f"[{n}]", align=2), out, scratch)
        scalars = (Scalar("rows"),)
    return compile_program(
        launch, name=f"mimo_{kind}_attention", operands=operands, scalars=scalars,
        key=(max_rows, kind, route, launch.key()),
        geometry={"kind": kind, "route": route, "heads": n, "kv_heads": g.kv_heads(kind), "record_elems": r,
                  "max_rows": max_rows, "window": g.window if kind == "swa" else 0, "ring_rows": g.ring_rows,
                  "page_rows": g.page_rows, "max_splits": int(max_splits) if kind == "full" and route == "decode"
                  else 1, "rows_per_cta": launch.attn.tokens},
        scratch={"scratch": lambda rows: attention_scratch_bytes(g, kind, route, rows, max_splits)},
        doc=__doc__,
    )


class _Output:
    def __init__(self, g: MiMoGeometry):
        self.o = glm_projection(g.hidden, g.heads * g.v_head_dim)

    def key(self) -> tuple:
        return self.o.key()

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.o(attn, w_o, out, rows, stream)


def compile_mimo_o_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, max_rows: int):
    """o_proj for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check("full", max_rows)
    launch = _Output(g)
    h, w = g.hidden, g.heads * g.v_head_dim
    return compile_program(
        launch, name="mimo_o",
        operands=(Operand("attn", torch.bfloat16, f"[rows,{w}]"), Operand("w_o", torch.bfloat16, f"[{h},{w}]"),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": h, "width": w, "max_rows": max_rows}, doc=__doc__,
    )
