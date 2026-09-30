"""Native AOT Qwen 3.8 Flash Next (``qwen4_exp``) Gated DeltaNet program, family ``qwen4``.

Weights are BF16 as stored (every Qwen 3.8 Flash Next checkpoint keeps the
GDN tensors in BF16); ``H`` hidden 2560, ``K`` key width 16 x 128, ``V``
value width 48 x 128, ``C = 2K + V`` short-conv channels 10240, ``P`` in-
projection rows ``C + V + 2 * 48`` = 16480. ``rows`` is the live row count
(``rows <= max_rows``); scratch regions are laid out from it (size with
``rows = max_rows``), each 1024-byte aligned.

``compile_qwen4_gdn_aot(g, max_rows=R)`` (one GDN layer, ``qwen4_gdn_m{R}``)::

    x          bf16 [rows,H]             in   attn_hyper_connection mixed input
    w_in       bf16 [P,H]                in   cat(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
    conv_w     f32  [C,4]                in   linear_attn.conv1d.weight (the 1 axis dropped)
    a_log      f32  [48]                 in   A_log
    dt_bias    f32  [48]                 in
    norm_w     bf16 [128]                in   linear_attn.norm.weight (used as w, not 1 + w)
    w_out      bf16 [H,V]                in   out_proj
    conv_state bf16 [slots,3,C]          inout  last three in-projection q|k|v rows per sequence
    state      f32  [slots,48,128,128]   inout  recurrent state [head, v, k] per sequence
    slots      i32  [rows]               in   state slot of each row's sequence (<0: zero state, not kept)
    seq_first  i32  [rows]               in   first step row of each row's sequence (rows of a sequence
                                              are contiguous and in position order)
    out        bf16 [rows,H]             out  attention output (before the hyper-connection injection)
    replay     f32  [replay_bytes / 4]   out  decode capacities only: the step's replay record
    scratch    u8   qwen4_gdn_scratch_bytes(rows)
    rows       int32
    spec       int32                          decode capacities only: nonzero leaves the conv and
                                              recurrent state as they were and records every row's
                                              replay inputs (``gdn_replay_layout``) for
                                              ``qwen4_gdn_commit``

``compile_qwen4_gdn_commit_aot(g)`` (``qwen4_gdn_commit``) applies each
sequence's accepted rows of a speculative step to every GDN layer's recurrent
and conv state in one launch; the state is bit-identical to serial steps over
those rows.

The recurrence is token-sequential (``Qwen4GdnRecurrent``: any mix of
sequences, e.g. decode or verify steps) unless the program was built for more
than ``CHUNKED_MIN_ROWS`` rows and the step has more than that many: then all
rows must be ONE sequence (a prefill step) and b12x's chunked delta-rule
prefill (``is_gdn``: 16-token tiles on tensor cores, initial = final state
``slots[0]``) runs it. Both paths read and write the same state layout, so a
sequence may alternate between them. A new sequence's slot must be zeroed
(conv state and recurrent state) before its first step.
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
from ._glmf_kernels import GlmfKdaConvState
from ._qwen4_gdn_kernels import (
    REPLAY_ROWS,
    Qwen4GdnCommit,
    Qwen4GdnConv,
    Qwen4GdnConvCommit,
    Qwen4GdnGatedNorm,
    Qwen4GdnRecurrent,
    gdn_replay_layout,
)
from .glmf import _SequenceMeta

__all__ = ["CHUNKED_MIN_ROWS", "compile_qwen4_gdn_aot", "compile_qwen4_gdn_commit_aot", "gdn_replay_layout",
           "gdn_scratch_bytes"]

_ALIGN = 1024

# Live rows above which a prefill-capacity program takes the chunked recurrence.
CHUNKED_MIN_ROWS = 64


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


def _ptr(dtype, address: Int64, align: int = 16):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


class _GdnChunked:
    """One sequence's GDN recurrence through b12x's chunked delta-rule prefill
    (prologue, then prepare + recurrence per window of tiles on one stream)."""

    def __init__(self, g: Qwen4Geometry, max_rows: int):
        from b12x.sequence._shared.delta_prefill import _cute_kernels as dk
        from b12x.sequence._shared.delta_prefill.contract import materialize_layout
        from b12x.sequence._shared.delta_prefill.workspace import default_window_tiles
        from b12x.sequence.gdn_prefill._impl import Caps, _Layout

        heads = g.gdn_value_heads
        caps = Caps(device=torch.device("cuda", torch.cuda.current_device()), max_tokens=int(max_rows), max_seqs=1,
                    max_state_slots=1, key_heads=g.gdn_key_heads, value_heads=heads)
        # The default GdnPrefillConfig: sequential, v_split 64, k_split 1, 3 stages.
        layout = materialize_layout(caps, layout_type=_Layout, v_split=64, k_split=1, stages=3,
                                    window_tiles=default_window_tiles(heads, int(max_rows), 1))
        self.layout, self.heads = layout, heads
        self.key_width, self.conv_width = g.gdn_key_width, g.gdn_conv_width
        self.tiles_per_window = layout.window_tiles
        self.meta = _SequenceMeta()
        self.prologue = dk._PrologueKernel(
            max_seqs=1, tiles_capacity=caps.tiles_capacity, window_tiles=layout.window_tiles,
            max_windows=layout.max_windows, flag_count=layout.workspace_windows * layout.window_tiles * heads)
        self.prepare = dk._PrepareKernel(
            heads=heads, key_heads=g.gdn_key_heads, is_gdn=True, tiles_capacity=caps.tiles_capacity,
            window_tiles=layout.window_tiles, qk_l2norm=True, a_log_type=cutlass.Float32,
            dt_bias_type=cutlass.Float32)
        self.recurrence = dk._RecurrenceKernel(
            heads=heads, tiles_capacity=caps.tiles_capacity, window_tiles=layout.window_tiles,
            rows=layout.recurrence_rows, v_split=64, k_split=1, stages=3, checkpoint_export=False,
            null_state_index=None, index_type=Int32, max_sequence_tiles=layout.max_sequence_tiles, is_gdn=True)
        self.nbytes = _align(int(layout.scratch_specs()[0].nbytes)) + _align(8 * 4)

    def key(self) -> tuple:
        lay = self.layout
        return (lay.window_tiles, lay.max_windows, lay.recurrence_rows, lay.v_split, lay.k_split, lay.stages)

    @cute.jit
    def __call__(self, qkv: cute.Pointer, a_raw: cute.Pointer, b_raw: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, out: cute.Pointer,
                 scratch: Int64, ab_stride: cutlass.Constexpr, rows: Int32, stream: cuda.CUstream):
        lay = self.layout
        off = lay.offsets
        meta_base = scratch + Int64(_align(int(lay.scratch_specs()[0].nbytes)))
        i32 = lambda at: _ptr(Int32, at, 4)  # noqa: E731
        self.meta(slots, i32(meta_base), rows, stream)
        cu, num_seqs = i32(meta_base), i32(meta_base + Int64(8))
        initial, final = i32(meta_base + Int64(16)), i32(meta_base + Int64(20))
        checkpoint, checkpoint_offsets = i32(meta_base + Int64(24)), i32(meta_base + Int64(28))
        band_base, sorted_seq = i32(scratch + Int64(off["band_base"])), i32(scratch + Int64(off["sorted_seq"]))
        rank_of, pos_seq = i32(scratch + Int64(off["rank_of"])), i32(scratch + Int64(off["pos_seq"]))
        pos_local, window_table = i32(scratch + Int64(off["pos_local"])), i32(scratch + Int64(off["window_table"]))
        ready = i32(scratch + Int64(off["ready_flags"]))
        ws = scratch + Int64(off["ws"])
        base = Int64(qkv.toint())
        q = _ptr(cutlass.BFloat16, base)
        k = _ptr(cutlass.BFloat16, base + Int64(self.key_width * 2))
        v = _ptr(cutlass.BFloat16, base + Int64(2 * self.key_width * 2))
        self.prologue(cu, num_seqs, band_base, sorted_seq, rank_of, pos_seq, pos_local, window_table, ready,
                      Int32(1), stream)
        tiles = (rows + Int32(15)) // Int32(16)
        windows = (tiles + Int32(self.tiles_per_window - 1)) // Int32(self.tiles_per_window)
        for w in cutlass.range_constexpr(lay.max_windows):
            if Int32(w) < windows:
                self.prepare(q, k, a_raw, b_raw, a_log, dt_bias, cu, pos_seq, pos_local, ready,
                             _ptr(cutlass.BFloat16, ws), _ptr(cutlass.Float32, ws),
                             Int64(self.conv_width), Int64(self.conv_width), Int64(ab_stride), Int64(ab_stride),
                             Int64(1), Float32(128.0 ** -0.5), Float32(0.0), Float32(1.0e-6), Int32(w), stream)
                self.recurrence(v, cu, band_base, sorted_seq, window_table, initial, final, checkpoint,
                                checkpoint_offsets, num_seqs, ready, _ptr(cutlass.Int8, ws, 16),
                                state, out, Int64(self.conv_width), Int64(self.heads * 128),
                                Int64(self.heads * 128 * 128), Int64(1), Int32(w), stream)


def gdn_scratch_bytes(g: Qwen4Geometry, rows: int, chunked: int = 0) -> int:
    """proj [rows,P], conv q|k|v [rows,C], o [rows,V], y [rows,V] (BF16), then the chunked
    recurrence's workspace (``chunked`` bytes, prefill capacities)."""
    rows = max(int(rows), 1)
    return (_align(rows * g.gdn_in_width * 2) + _align(rows * g.gdn_conv_width * 2)
            + 2 * _align(rows * g.gdn_value_width * 2) + _align(chunked))


class _Gdn:
    def __init__(self, g: Qwen4Geometry, max_rows: int = 64, fp8: bool = False):
        self.g = g
        self.chunked = _GdnChunked(g, max_rows) if int(max_rows) > CHUNKED_MIN_ROWS else None
        c, v, p = g.gdn_conv_width, g.gdn_value_width, g.gdn_in_width
        self.c, self.v, self.p = c, v, p
        self.fp8 = bool(fp8)
        self.in_proj = projection(p, g.hidden, self.fp8)
        self.conv = Qwen4GdnConv(channels=c, proj_width=p)
        self.conv_state = GlmfKdaConvState(channels=c, proj_width=p)
        self.recurrent = Qwen4GdnRecurrent(heads=g.gdn_value_heads, key_heads=g.gdn_key_heads, ab_stride=p)
        self.norm = Qwen4GdnGatedNorm(heads=g.gdn_value_heads, eps=g.norm_eps, gate_stride=p)
        self.o_proj = projection(g.hidden, v, self.fp8)
        self.replay = gdn_replay_layout(g.gdn_key_heads, g.gdn_value_heads, c)
        # Speculative steps (replay records) are decode-capacity programs only.
        self.spec = self.chunked is None

    def key(self) -> tuple:
        return (self.in_proj.key(), self.o_proj.key(), self.g, self.fp8,
                None if self.chunked is None else self.chunked.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, norm_w: cute.Pointer, w_out: cute.Pointer, conv_state: cute.Pointer,
                 state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in, w_in, conv_w, a_log, dt_bias, norm_w, w_out, w_out, w_out, conv_state, state,
                  slots, seq_first, out, state, scratch, rows, Int32(0), stream)

    @cute.jit
    def body(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
             conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer, norm_w: cute.Pointer,
             w_out: cute.Pointer, w_out_fp8: cute.Pointer, w_out_scale: cute.Pointer, conv_state: cute.Pointer,
             state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer,
             replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, spec: Int32, stream: cuda.CUstream):
        c, v, p = self.c, self.v, self.p
        m = Int64(rows)
        base = Int64(scratch.toint())
        qkv_off = base + _align_i64(m * Int64(p * 2))
        o_off = qkv_off + _align_i64(m * Int64(c * 2))
        y_off = o_off + _align_i64(m * Int64(v * 2))
        bf16 = cutlass.BFloat16
        proj = _ptr(bf16, base)
        self.in_proj(x, w_in, w_in_fp8, w_in_scale, proj, rows, stream)
        self.conv(proj, conv_w, conv_state, slots, seq_first, _ptr(bf16, qkv_off), rows, stream)
        # spec 0: the conv window advances in place; otherwise the rows go to the replay record.
        self.conv_state(proj, conv_state, slots, seq_first,
                        _ptr(bf16, Int64(replay.toint()) + Int64(self.replay["proj"])), spec, rows, stream)
        z = _ptr(bf16, base + Int64(c * 2))
        b_raw = _ptr(bf16, base + Int64((c + v) * 2), 2)
        a_raw = _ptr(bf16, base + Int64((c + v + self.g.gdn_value_heads) * 2), 2)
        if cutlass.const_expr(self.chunked is None):
            self.recurrent(_ptr(bf16, qkv_off), a_raw, b_raw, a_log, dt_bias, state, slots, _ptr(bf16, o_off),
                           replay, spec, rows, stream)
        else:
            if rows > Int32(CHUNKED_MIN_ROWS):
                self.chunked(_ptr(bf16, qkv_off), a_raw, b_raw, a_log, dt_bias, state, slots, _ptr(bf16, o_off),
                             y_off + _align_i64(m * Int64(v * 2)), p, rows, stream)
            else:
                self.recurrent(_ptr(bf16, qkv_off), a_raw, b_raw, a_log, dt_bias, state, slots,
                               _ptr(bf16, o_off), replay, spec, rows, stream)
        self.norm(_ptr(bf16, o_off), z, norm_w, _ptr(bf16, y_off), rows, stream)
        self.o_proj(_ptr(bf16, y_off), w_out, w_out_fp8, w_out_scale, out, rows, stream)


class _GdnSpec(_Gdn):
    """The BF16 decode-capacity GDN layer with the speculative replay record."""

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, norm_w: cute.Pointer, w_out: cute.Pointer, conv_state: cute.Pointer,
                 state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer,
                 replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, spec: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in, w_in, conv_w, a_log, dt_bias, norm_w, w_out, w_out, w_out, conv_state, state,
                  slots, seq_first, out, replay, scratch, rows, spec, stream)


class _GdnFp8(_Gdn):
    """The GDN layer with FP8 in/out projections for decode rows (<= 16 live
    rows read the E4M3 copies, more rows the BF16 weights) and the speculative
    replay record."""

    def __init__(self, g: Qwen4Geometry, max_rows: int = 64):
        super().__init__(g, max_rows, fp8=True)
        if not self.spec:
            raise ValueError("FP8 GDN programs are decode capacities (<= CHUNKED_MIN_ROWS rows)")

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
                 conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer, norm_w: cute.Pointer,
                 w_out: cute.Pointer, w_out_fp8: cute.Pointer, w_out_scale: cute.Pointer, conv_state: cute.Pointer,
                 state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer,
                 replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, spec: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in_fp8, w_in_scale, conv_w, a_log, dt_bias, norm_w, w_out, w_out_fp8, w_out_scale,
                  conv_state, state, slots, seq_first, out, replay, scratch, rows, spec, stream)


def compile_qwen4_gdn_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, max_rows: int, fp8: bool = False) -> AotProgram:
    """One GDN layer for ``rows <= max_rows``; see the module docstring. ``fp8``
    adds ``w_in_fp8``/``w_in_scale`` and ``w_out_fp8``/``w_out_scale`` (E4M3 with
    FP32 128x128 block scales) after each BF16 weight for rows <= 16."""
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    spec = max_rows <= CHUNKED_MIN_ROWS
    launch = (_GdnFp8 if fp8 else _GdnSpec if spec else _Gdn)(g, max_rows)
    chunked = 0 if launch.chunked is None else launch.chunked.nbytes
    record = launch.replay["bytes"]
    h, c, v, p, heads = g.hidden, g.gdn_conv_width, g.gdn_value_width, g.gdn_in_width, g.gdn_value_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_in", torch.bfloat16, f"[{p},{h}]",
                note="cat(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)"),
        *(fp8_operands("w_in", p, h) if fp8 else ()),
        Operand("conv_w", torch.float32, f"[{c},4]", align=4),
        Operand("a_log", torch.float32, f"[{heads}]", align=4),
        Operand("dt_bias", torch.float32, f"[{heads}]", align=4),
        Operand("norm_w", torch.bfloat16, f"[{g.gdn_head_dim}]"),
        Operand("w_out", torch.bfloat16, f"[{h},{v}]"),
        *(fp8_operands("w_out", h, v) if fp8 else ()),
        Operand("conv_state", torch.bfloat16, f"[slots,3,{c}]", "inout", align=2),
        Operand("state", torch.float32, f"[slots,{heads},128,128]", "inout"),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        *((Operand("replay", torch.float32, f"[{record // 4}]", "out",
                   note="speculative replay record (gdn_replay_layout)"),) if spec else ()),
        Operand("scratch", torch.uint8, "[qwen4_gdn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="qwen4_gdn", operands=operands,
        scalars=(Scalar("rows"), Scalar("spec")) if spec else (Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "key_heads": g.gdn_key_heads, "value_heads": heads, "head_dim": g.gdn_head_dim,
                  "max_rows": max_rows, "in_width": p, "eps": g.norm_eps, "fp8_weights": fp8,
                  "chunked_min_rows": CHUNKED_MIN_ROWS if chunked else None,
                  "replay_rows": REPLAY_ROWS if spec else None, "replay_bytes": record if spec else None},
        scratch={"scratch": lambda rows: gdn_scratch_bytes(g, rows, chunked)},
        doc=__doc__,
    )


class _GdnCommit:
    def __init__(self, g: Qwen4Geometry):
        self.g = g
        c = g.gdn_conv_width
        self.recurrent = Qwen4GdnCommit(heads=g.gdn_value_heads, key_heads=g.gdn_key_heads, channels=c)
        self.conv = Qwen4GdnConvCommit(heads=g.gdn_value_heads, key_heads=g.gdn_key_heads, channels=c)

    @cute.jit
    def __call__(self, state: cute.Pointer, conv_state: cute.Pointer, replay: cute.Pointer, tables: cute.Pointer,
                 sequences: Int32, layers: Int32, slots: Int32, stream: cuda.CUstream):
        self.recurrent(state, replay, tables, sequences, layers, slots, stream)
        self.conv(conv_state, replay, tables, sequences, layers, slots, stream)


def compile_qwen4_gdn_commit_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """Verify-by-replay for the GDN layers: after a speculative decode step
    (``qwen4_gdn`` with ``spec`` = 1, which records each row's replay inputs and
    leaves the state alone), apply each sequence's accepted rows to its
    recurrent and conv state in every GDN layer.

    ``state`` FP32 ``[layers, slots, 48, 128, 128]`` and ``conv_state`` BF16
    ``[layers, slots, 3, C]`` (the layers' pools back to back), ``replay`` the
    layers' records back to back (``replay_bytes`` each), ``tables`` i32
    ``[3, sequences]``: state slot (negative: skip), first step row and
    accepted rows per sequence. The recurrent update repeats ``qwen4_gdn``'s
    arithmetic, so the state is bit-identical to serial steps over those rows.
    """
    launch = _GdnCommit(g)
    heads, c = g.gdn_value_heads, g.gdn_conv_width
    record = gdn_replay_layout(g.gdn_key_heads, heads, c)["bytes"]
    return compile_program(
        launch, name="qwen4_gdn_commit",
        operands=(Operand("state", torch.float32, f"[layers,slots,{heads},128,128]", "inout"),
                  Operand("conv_state", torch.bfloat16, f"[layers,slots,3,{c}]", "inout", align=2),
                  Operand("replay", torch.float32, f"[layers,{record // 4}]"),
                  Operand("tables", torch.int32, "[3,sequences]", align=4)),
        scalars=(Scalar("sequences"), Scalar("layers"), Scalar("slots")),
        key=(heads, g.gdn_key_heads, c, REPLAY_ROWS),
        geometry={"value_heads": heads, "key_heads": g.gdn_key_heads, "channels": c, "replay_rows": REPLAY_ROWS,
                  "replay_bytes": record},
        doc=compile_qwen4_gdn_commit_aot.__doc__,
    )
