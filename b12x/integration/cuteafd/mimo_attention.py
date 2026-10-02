"""Native AOT MiMo V2 attention programs: QKV producer, GQA attention, o_proj.

Two attention kinds (``kind``): ``full`` (64 query / 4 KV heads, causal over
a paged cache, no sink, RoPE theta 5e6) and ``swa`` (64 / 8 heads, 128-token
window, learned per-head sink, theta 1e4). ``H`` hidden 4096, ``N`` 64 query
heads, ``G`` KV heads of the kind, ``R = G * 320`` BF16 elements per KV record
(keys ``[G, 192]`` then values ``[G, 128]``, see ``_mimo_kernels``). Weights
in the legacy ABI are BF16 (FP8 source blocks dequantized with their FP32
scales; ``o_proj`` is stored BF16 in the supported checkpoints). Legacy
projections are ``RoutedBf16Projection`` (skinny GEMV for few rows, TMA
tensor-core GEMM above; FP32 accumulation).
With ``fp8_only`` (the exported producers) ``w_qkv`` is ``w_qkv_fp8`` E4M3 +
FP32 per-row x 128-K scales only (``w_qkv_scale [W, K/128]`` in decode programs,
``w_qkv_kscale [K/128, W]`` in prefill programs; ``mimo_w8``: decode GEMVs to 32
rows then W8A16; prefill W8A8 or W8A16 on the ``fp8_rows`` switch).
The optional FP8-only output projection uses the same layouts and dispatch
with ``w_o_fp8`` and ``w_o_scale`` / ``w_o_kscale``, and no BF16 operand.
Quantizing a BF16 checkpoint output weight requires model quality validation
separate from validating the FP8 projection against its dequantized weights.
Scratch is laid out from the live row count (size it with ``rows =
max_rows``); regions start 1024-byte aligned.

``compile_mimo_producer_aot(g, kind=K, max_rows=M)``::

    x           bf16 [rows,H]         in     input_layernorm output
    positions   i64  [rows]           in     RoPE position per row
    kv_slots    i64  [rows]           in     record slot per row (<0 skips): full = page*64+row
                                             of the paged cache; swa = the row's index in kv_step
    cos_sin     f32  [P,64]           in     cos(32) | sin(32) of theta^(-2i/64) (theta of the kind)
    w_qkv       bf16 [W,H]            in     cat(q_proj, k_proj, v_proj), W = N*192 + G*(k_stride+128)
                                             (keys k_stride = 192 apart, or 256: V2.6 Pro zero-pads
                                             every key to whole 128-row blocks)
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

KV records (``kv``): ``"bf16"`` as above; ``"int8"`` / ``"fp8"`` 8-bit records
(signed bytes with an FP32 scale per 32 dims of each head's key and value, or
E4M3 with one per 64 dims; ``record_bytes``, layout in ``_mimo_kernels``): the
producer quantizes the record it writes, attention widens keys and values to
BF16 in shared memory, the SWA commit copies whole records. ``kv_cache``,
``kv_step`` and ``ring`` are then ``u8 [.., record_bytes]``. The 8-bit full
prefill program takes ``kv_wide bf16 [keys, R]`` and the scalar ``keys`` (last
row position + 1) after ``table_stride``: it widens the sequence's keys there
once, then runs the BF16 prefill attention over that copy.

Softmax: ``softmax(q.k * 192^-0.5 [; sink]) . v`` (the sink is a raw extra
logit that takes probability mass but no value), FP32 scores, BF16
probabilities, FP32 PV accumulation.

``compile_mimo_o_aot(g, max_rows=M)``::

    attn        bf16 [rows,N*128]     in     attention output (heads major)
    w_o         bf16 [H,N*128]        in     o_proj (BF16 in the checkpoint)
    out         bf16 [rows,H]         out    before the residual add
    rows        int32

FP8 decode weights (``fp8=True``, producer and o): each BF16 weight ``w`` is
followed by ``w_fp8`` (E4M3 ``[N, K]``) and ``w_scale`` (FP32 ``[N, K/128]``,
one scale per output row and 128-wide K block, so the checkpoint's 128x128
grids and the per-head full-attention ``k_proj`` grid expand exactly, and a
BF16 weight such as ``o_proj`` quantizes per row), and the scalar
``fp8_rows`` follows ``rows``: steps of ``rows <= min(fp8_rows, MIMO_FP8_ROWS)``
read the E4M3 copy through ``MmaFp8Gemv`` (one 16-row tile up to 16 rows, two
above; weights widened to ``bf16(w * s)``, the BF16 program's dequantized
weight), longer steps the BF16 weight; ``fp8_rows`` 0 turns FP8 off.

``compile_mimo_head_fp8_aot(g)``: FP32 logits of up to 16 decode rows over an
E4M3 LM head with per-row x 128-K scales::

    x           bf16 [rows,H]         in     final norm output
    w_fp8       e4m3 [V,H]            in
    scale       f32  [V,H/128]        in
    logits      f32  [rows,V]         out
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
from .glmf import FP8_GEMV_CONFIG, FP8_ROWS, _Fp8Switch, _HeadFp8, fp8_ops
from ._mimo_kernels import (
    MAX_SPLITS, MimoGqaAttention, MimoKvWiden, MimoQkvRope, MimoRingCommit, MimoSplitMerge, fp8_record_bytes, fp8_scale_counts,
)

# 8-bit KV records (``kv="fp8"``: E4M3, ``kv="int8"``: signed bytes): one FP32 scale per this
# many dims of each head's key and value (0: one per key, one per value). See ``_mimo_kernels``.
KV_FP8_GROUP = 64
KV_GROUPS = {"fp8": 64, "int8": 32}
_KV8 = {"bf16": None, "fp8": "e4m3", "int8": "s8"}

# Most rows a MiMo decode program reads the E4M3 copies for (qkv, o, dense FFN): DFlash
# verify steps of 17-32 rows stay on FP8 (V2.6 Pro coordinator alone, 1 RTX PRO 6000:
# 16 rows 23.3 ms, 24 rows 44.6 ms on the BF16 projections).
MIMO_FP8_ROWS = 32

__all__ = [
    "KV_FP8_GROUP",
    "attention_scratch_bytes",
    "compile_mimo_attention_aot",
    "compile_mimo_head_fp8_aot",
    "compile_mimo_o_aot",
    "compile_mimo_producer_aot",
    "producer_scratch_bytes",
    "prefill_tokens",
    "record_bytes",
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


def _check_kv(kv: str) -> str:
    if kv not in _KV8:
        raise ValueError(f"kv is one of {tuple(_KV8)}")
    return kv


def _group(kv: str, kv_group: int | None) -> int:
    return KV_GROUPS.get(kv, KV_FP8_GROUP) if kv_group is None else int(kv_group)


def record_bytes(g: MiMoGeometry, kind: str, kv: str = "bf16", kv_group: int | None = None) -> int:
    """Bytes of one token's KV record of ``kind`` (``kv`` "bf16", "fp8" or "int8", see ``_mimo_kernels``)."""
    if _check_kv(kv) != "bf16":
        return fp8_record_bytes(g.kv_heads(kind), _group(kv, kv_group), g.qk_head_dim, g.v_head_dim)
    return g.record_elems(kind) * 2


def _kv_operand(name: str, g: MiMoGeometry, kind: str, kv: str, rows: str, mode: str = "in",
                kv_group: int | None = None):
    if kv != "bf16":
        return Operand(name, torch.uint8, f"[{rows},{record_bytes(g, kind, kv, kv_group)}]", mode)
    return Operand(name, torch.bfloat16, f"[{rows},{g.record_elems(kind)}]", mode)


def _kv_geometry(g: MiMoGeometry, kind: str, kv: str, kv_group: int | None) -> dict:
    kv_group = _group(kv, kv_group)
    out = {"kv": kv, "record_bytes": record_bytes(g, kind, kv, kv_group)}
    if kv != "bf16":
        sk, sv = fp8_scale_counts(kv_group, g.qk_head_dim, g.v_head_dim)
        out.update({"kv_group": kv_group, "key_scales": sk, "value_scales": sv})
    return out


class _Producer:
    def __init__(self, g: MiMoGeometry, kind: str, fp8: bool = False, kv: str = "bf16", kv_group: int | None = None):
        self.g, self.kind, self.fp8 = g, kind, bool(fp8)
        self.kv = _check_kv(kv)
        self.kv_group = _group(kv, kv_group)
        self.qkv = _Fp8Switch(g.qkv_width(kind), g.hidden, fp8=True, row_scales=True, wide_rows=MIMO_FP8_ROWS) if fp8 \
            else glm_projection(g.qkv_width(kind), g.hidden)
        self.post = MimoQkvRope(heads=g.heads, kv_heads=g.kv_heads(kind), v_scale=g.v_scale,
                                head=g.qk_head_dim, v_head=g.v_head_dim, k_stride=g.qkv_k_stride,
                                kv8=_KV8[self.kv], kv_group=self.kv_group)

    def key(self) -> tuple:
        return (self.qkv.key(), self.kind, self.g, self.fp8, self.kv, self.kv_group)

    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        qkv = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.qkv(x, w_qkv, qkv, rows, stream)
        self.post(qkv, positions, cos_sin, kv_slots, query, kv_cache, rows, stream)


class _ProducerFp8(_Producer):
    def __init__(self, g: MiMoGeometry, kind: str, kv: str = "bf16", kv_group: int | None = None):
        super().__init__(g, kind, fp8=True, kv=kv, kv_group=kv_group)

    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv: cute.Pointer, w_qkv_fp8: cute.Pointer, w_qkv_scale: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        qkv = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.qkv(x, w_qkv, w_qkv_fp8, w_qkv_scale, qkv, rows, fp8_rows, stream)
        self.post(qkv, positions, cos_sin, kv_slots, query, kv_cache, rows, stream)


def mimo_w8(n: int, k: int, prefill_rows: int | None):
    """An ``Fp8Projection`` over a per-row x 128-K scaled E4M3 weight (MiMo's FP8 tensors: the
    checkpoint's E4M3 bytes with its 128x128 or per-head grid expanded per row; decode programs
    take the scales row major, ``{w}_scale [N, K/128]``, prefill programs K-block major,
    ``{w}_kscale [K/128, N]``): decode rows run the 16-row GEMV, the two-tile GEMV up to
    ``MIMO_FP8_ROWS`` and the W8A16 TMA GEMM above; prefill rows run W8A8 (``fp8_rows`` nonzero)
    or W8A16. (A K-block-major decode GEMV reads its scales one cache line per K block: SM120,
    16 rows of 14848x4096 back to back over 40 weights, 40.5 -> 48.7 us.)"""
    from ._fp8_weights import Fp8Projection, gemv_warps

    warps, groups = FP8_GEMV_CONFIG.get((n, k), (gemv_warps(k), 4))
    return Fp8Projection(n, k, prefill_rows=prefill_rows, row_scales=True, gemv_rows=FP8_ROWS,
                         wide_rows=MIMO_FP8_ROWS, warps=warps, groups=groups)


def mimo_w8_scratch_bytes(k: int, rows: int, prefill: bool) -> int:
    from ._glmf_fp8 import quant_scratch_bytes

    return quant_scratch_bytes(k, rows) if prefill else 0


class _ProducerW8(_Producer):
    """The QKV producer over the FP8-only ``w_qkv`` (no BF16 copy)."""

    def __init__(self, g: MiMoGeometry, kind: str, max_rows: int, prefill: bool, kv: str = "bf16",
                 kv_group: int | None = None):
        super().__init__(g, kind, fp8=False, kv=kv, kv_group=kv_group)
        self.fp8 = "only"
        self.qkv = mimo_w8(g.qkv_width(kind), g.hidden, int(max_rows) if prefill else None)

    @cute.jit
    def body(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
             w_qkv_fp8: cute.Pointer, w_qkv_scale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        qkv = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        # The quantized rows follow the projection output (producer_scratch_bytes).
        qkv_bytes = Int64(rows) * Int64(self.g.qkv_width(self.kind) * 2)
        qscratch = Int64(scratch.toint()) + (qkv_bytes + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        self.qkv(x, w_qkv_fp8, w_qkv_scale, qkv, rows, fp8_rows, qscratch, stream)
        self.post(qkv, positions, cos_sin, kv_slots, query, kv_cache, rows, stream)


class _ProducerW8Decode(_ProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv_fp8: cute.Pointer, w_qkv_scale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(x, positions, kv_slots, cos_sin, w_qkv_fp8, w_qkv_scale, kv_cache, query, scratch, rows, fp8_rows,
                  stream)


class _ProducerW8Prefill(_ProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv_fp8: cute.Pointer, w_qkv_kscale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(x, positions, kv_slots, cos_sin, w_qkv_fp8, w_qkv_kscale, kv_cache, query, scratch, rows, fp8_rows,
                  stream)


def _compile_mimo_producer_w8(g: MiMoGeometry, kind: str, max_rows: int, mode: str, kv: str, kv_group: int):
    from ._fp8_weights import check_w8_mode, fp8_only_operands

    prefill = check_w8_mode(mode) == "prefill"
    launch = (_ProducerW8Prefill if prefill else _ProducerW8Decode)(g, kind, max_rows, prefill, kv, kv_group)
    h, n, r, w = g.hidden, g.heads, g.record_elems(kind), g.qkv_width(kind)
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        *fp8_only_operands("w_qkv", w, h, row_scales=True, prefill=prefill),
        _kv_operand("kv_cache", g, kind, kv, "slots", "inout", kv_group),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.qk_head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"mimo_{kind}_producer", operands=operands,
        scalars=(Scalar("rows"), Scalar("fp8_rows", note="decode: most rows on the FP8 GEMV; prefill: nonzero "
                                                         "runs W8A8, 0 W8A16")),
        key=(max_rows, mode, launch.key()),
        geometry={"kind": kind, "hidden": h, "heads": n, "kv_heads": g.kv_heads(kind), "record_elems": r,
                  "max_rows": max_rows, "v_scale": g.v_scale, "rope_theta": g.rope_theta(kind), "fp8_weights": "only",
                  "mode": mode, "qkv_width": w, "k_stride": g.qkv_k_stride, **_kv_geometry(g, kind, kv, kv_group)},
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, kind, rows) + mimo_w8_scratch_bytes(h, rows, prefill)},
        doc=__doc__,
    )


def compile_mimo_producer_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, kind: str, max_rows: int, fp8: bool = False,
                              fp8_only: str | None = None, kv: str = "bf16", kv_group: int | None = None):
    """QKV projection, partial RoPE and the KV record write; see the module docstring
    (``fp8``: the E4M3 ``w_qkv`` copy for decode rows; ``fp8_only`` ``"decode"`` /
    ``"prefill"``: ``w_qkv`` as E4M3 + per-row scales only, see ``mimo_w8``; ``kv``:
    ``"int8"`` / ``"fp8"`` write 8-bit records, see ``_mimo_kernels``)."""
    max_rows = _check(kind, max_rows)
    _check_kv(kv)
    if fp8_only is not None:
        return _compile_mimo_producer_w8(g, kind, max_rows, fp8_only, kv, kv_group)
    launch = (_ProducerFp8(g, kind, kv, kv_group) if fp8 else _Producer(g, kind, kv=kv, kv_group=kv_group))
    h, n, r = g.hidden, g.heads, g.record_elems(kind)
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_qkv", torch.bfloat16, f"[{g.qkv_width(kind)},{h}]"),
        *(fp8_ops("w_qkv", g.qkv_width(kind), h, True) if fp8 else ()),
        _kv_operand("kv_cache", g, kind, kv, "slots", "inout", kv_group),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.qk_head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"mimo_{kind}_producer", operands=operands,
        scalars=(Scalar("rows"), Scalar("fp8_rows")) if fp8 else (Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"kind": kind, "hidden": h, "heads": n, "kv_heads": g.kv_heads(kind), "record_elems": r,
                  "max_rows": max_rows, "v_scale": g.v_scale, "rope_theta": g.rope_theta(kind), "fp8_weights": fp8,
                  "qkv_width": g.qkv_width(kind), "k_stride": g.qkv_k_stride, **_kv_geometry(g, kind, kv, kv_group)},
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, kind, rows)},
        doc=__doc__,
    )


class _FullAttention:
    def __init__(self, g: MiMoGeometry, route: str, max_splits: int, kv: str = "bf16", kv_group: int | None = None):
        self.g, self.route = g, route
        decode = route == "decode"
        self.attn = MimoGqaAttention(
            heads=g.heads, kv_heads=g.full_kv_heads, tokens=1 if decode else prefill_tokens(g, "full"),
            window=0, paged=True, sink=False, direct=not decode, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, ring_rows=g.ring_rows, head=g.qk_head_dim, v_head=g.v_head_dim,
            kv8=_KV8[_check_kv(kv)], kv_group=_group(kv, kv_group))
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
    def __init__(self, g: MiMoGeometry, route: str, kv: str = "bf16", kv_group: int | None = None):
        self.g, self.route = g, route
        self.attn = MimoGqaAttention(
            heads=g.heads, kv_heads=g.swa_kv_heads, tokens=1 if route == "decode" else prefill_tokens(g, "swa"),
            window=g.window, paged=False, sink=True, direct=True, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, ring_rows=g.ring_rows, head=g.qk_head_dim, v_head=g.v_head_dim,
            kv8=_KV8[_check_kv(kv)], kv_group=_group(kv, kv_group))
        self.commit = MimoRingCommit(record=g.record_elems("swa"), ring_rows=g.ring_rows,
                                     record_bytes=record_bytes(g, "swa", kv, kv_group))

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


class _FullAttentionPrefillWide(_FullAttention):
    """Prefill over 8-bit records: :class:`MimoKvWiden` copies the sequence's keys ``< keys`` to
    BF16 ``kv_wide`` once, then the BF16 prefill attention reads that copy (``contiguous``)."""

    def __init__(self, g: MiMoGeometry, route: str, max_splits: int, kv: str, kv_group: int | None):
        super().__init__(g, route, max_splits)
        self.kv, self.kv_group = kv, _group(kv, kv_group)
        self.attn = MimoGqaAttention(
            heads=g.heads, kv_heads=g.full_kv_heads, tokens=prefill_tokens(g, "full"), window=0, paged=True,
            sink=False, direct=True, softmax_scale=g.softmax_scale, page_rows=g.page_rows, ring_rows=g.ring_rows,
            head=g.qk_head_dim, v_head=g.v_head_dim, contiguous=True)
        self.widen = MimoKvWiden(kv_heads=g.full_kv_heads, kv8=_KV8[kv], kv_group=self.kv_group,
                                 page_rows=g.page_rows, head=g.qk_head_dim, v_head=g.v_head_dim)

    def key(self) -> tuple:
        return (self.attn.key(), self.route, self.kv, self.kv_group, "wide")

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 kv_wide: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32,
                 keys: Int32, stream: cuda.CUstream):
        self.widen(kv_cache, page_table, kv_wide, keys, stream)
        self.run(q, kv_wide, positions, page_table, out, scratch, rows, table_stride, Int32(1), stream)


def compile_mimo_attention_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, kind: str, route: str, max_rows: int,
                               max_splits: int = DEFAULT_MAX_SPLITS, kv: str = "bf16", kv_group: int | None = None):
    """GQA attention of ``kind`` for ``rows <= max_rows``; see the module docstring (``kv``
    ``"int8"`` / ``"fp8"``: 8-bit records, see ``_mimo_kernels``)."""
    max_rows = _check(kind, max_rows)
    _check_kv(kv)
    if route not in ("decode", "prefill"):
        raise ValueError("route is 'decode' or 'prefill'")
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    n, r = g.heads, g.record_elems(kind)
    q = Operand("q", torch.bfloat16, f"[rows,{n},{g.qk_head_dim}]")
    out = Operand("out", torch.bfloat16, f"[rows,{n},{g.v_head_dim}]", "out")
    scratch = Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch")
    if kind == "full":
        wide = route == "prefill" and kv != "bf16"
        cls = _FullAttentionDecode if route == "decode" else _FullAttentionPrefillWide if wide else _FullAttentionPrefill
        launch = cls(g, route, max_splits, kv, kv_group)
        operands = (q, _kv_operand("kv_cache", g, kind, kv, f"pages*{g.page_rows}", "in", kv_group),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4))
        if wide:
            operands += (Operand("kv_wide", torch.bfloat16, f"[keys,{r}]", "out"),)
        operands += (out, scratch)
        scalars = (Scalar("rows"), Scalar("table_stride"))
        if route == "decode":
            scalars += (Scalar("splits"),)
        if wide:
            scalars += (Scalar("keys", note="the sequence's last row position + 1 (keys widened into kv_wide)"),)
    else:
        launch = _SwaAttention(g, route, kv, kv_group)
        operands = (q, _kv_operand("kv_step", g, kind, kv, "rows", "in", kv_group),
                    _kv_operand("ring", g, kind, kv, f"rings*{g.ring_rows}", "inout", kv_group),
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
                  else 1, "rows_per_cta": launch.attn.tokens, **_kv_geometry(g, kind, kv, kv_group)},
        scratch={"scratch": lambda rows: attention_scratch_bytes(g, kind, route, rows, max_splits)},
        doc=__doc__,
    )


class _Output:
    def __init__(self, g: MiMoGeometry, fp8: bool = False):
        self.fp8 = bool(fp8)
        self.o = _Fp8Switch(g.hidden, g.heads * g.v_head_dim, fp8=True, row_scales=True, wide_rows=MIMO_FP8_ROWS) if fp8 \
            else glm_projection(g.hidden, g.heads * g.v_head_dim)

    def key(self) -> tuple:
        return (self.o.key(), self.fp8)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.o(attn, w_o, out, rows, stream)


class _OutputFp8(_Output):
    def __init__(self, g: MiMoGeometry):
        super().__init__(g, fp8=True)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer, w_o_scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.o(attn, w_o, w_o_fp8, w_o_scale, out, rows, fp8_rows, stream)


class _OutputW8:
    def __init__(self, g: MiMoGeometry, max_rows: int, prefill: bool):
        self.o = mimo_w8(g.hidden, g.heads * g.v_head_dim, int(max_rows) if prefill else None)

    def key(self) -> tuple:
        return self.o.key()

    @cute.jit
    def body(self, attn: cute.Pointer, w_o_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.o(attn, w_o_fp8, scale, out, rows, fp8_rows, Int64(scratch.toint()), stream)


class _OutputW8Decode(_OutputW8):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_o_fp8: cute.Pointer, w_o_scale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(attn, w_o_fp8, w_o_scale, out, scratch, rows, fp8_rows, stream)


class _OutputW8Prefill(_OutputW8):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_o_fp8: cute.Pointer, w_o_kscale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(attn, w_o_fp8, w_o_kscale, out, scratch, rows, fp8_rows, stream)


def compile_mimo_o_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, max_rows: int, fp8: bool = False,
                       fp8_only: str | None = None):
    """o_proj for ``rows <= max_rows``; see the module docstring (``fp8``: the
    E4M3 per-row-scaled copy for decode rows; ``fp8_only``: ``"decode"`` or
    ``"prefill"`` over one E4M3 weight, with no BF16 weight operand).

    FP8-only prefill uses W8A16 when ``fp8_rows=0``. A nonzero value enables
    activation quantization, which requires its own model quality gate.
    """
    max_rows = _check("full", max_rows)
    if fp8_only is not None:
        from ._fp8_weights import check_w8_mode, fp8_only_operands

        prefill = check_w8_mode(fp8_only) == "prefill"
        launch = (_OutputW8Prefill if prefill else _OutputW8Decode)(g, max_rows, prefill)
        h, w = g.hidden, g.heads * g.v_head_dim
        return compile_program(
            launch, name="mimo_o",
            operands=(Operand("attn", torch.bfloat16, f"[rows,{w}]"),
                      *fp8_only_operands("w_o", h, w, row_scales=True, prefill=prefill),
                      Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
                      Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch")),
            scalars=(Scalar("rows"), Scalar("fp8_rows")), key=(max_rows, fp8_only, launch.key()),
            geometry={"hidden": h, "width": w, "max_rows": max_rows, "fp8_weights": "only", "mode": fp8_only},
            scratch={"scratch": lambda rows: mimo_w8_scratch_bytes(w, rows, prefill)}, doc=__doc__,
        )
    launch = _OutputFp8(g) if fp8 else _Output(g)
    h, w = g.hidden, g.heads * g.v_head_dim
    return compile_program(
        launch, name="mimo_o",
        operands=(Operand("attn", torch.bfloat16, f"[rows,{w}]"), Operand("w_o", torch.bfloat16, f"[{h},{w}]"),
                  *(fp8_ops("w_o", h, w, True) if fp8 else ()),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"), Scalar("fp8_rows")) if fp8 else (Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": h, "width": w, "max_rows": max_rows, "fp8_weights": fp8}, doc=__doc__,
    )


def compile_mimo_head_fp8_aot(g: MiMoGeometry = MIMO_V2_FLASH):
    """FP32 logits of up to 16 decode rows over the E4M3 LM head; see the module docstring."""
    h, v = g.hidden, g.vocab_size
    launch = _HeadFp8(v, h)
    return compile_program(
        launch, name="mimo_head_fp8",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w_fp8", torch.float8_e4m3fn, f"[{v},{h}]"),
                  Operand("scale", torch.float32, f"[{v},{h // 128}]", align=4),
                  Operand("logits", torch.float32, f"[rows,{v}]", "out")),
        scalars=(Scalar("rows"),), key=(v, launch.key()),
        geometry={"hidden": h, "vocab": v, "max_rows": FP8_ROWS}, doc=__doc__,
    )
