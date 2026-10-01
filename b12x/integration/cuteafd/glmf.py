"""Native AOT GLM 5.3 Flash (``glm5_next``) coordinator programs, family ``glmf``.

Weights are BF16 (the EXL3 checkpoints' dense tensors, or FP8 blocks x FP32
scales dequantized at load), except with ``fp8_only`` (the exported MLA
producer, o and FFN programs): ``w_qkv_a``, ``w_q_b``, ``w_o``, ``w_gate_up``
and ``w_down`` are then ``{w}_fp8`` E4M3 + ``{w}_scale`` FP32 128x128 grids
only (the official FP8 release's tensors), decode rows up to ``fp8_rows`` on
the FP8 GEMV and W8A16 GEMMs above, prefill W8A8 (``fp8_rows`` nonzero) or
W8A16; ``H`` hidden 4096, ``D`` KDA width 64 x 128,
``P`` KDA in-projection rows ``3D + 2*128 + 64``, ``N`` MLA heads 64, ``Q``
q_lora 1536, ``E`` routed experts 288. ``rows`` is the live row count
(``rows <= max_rows``); scratch regions are laid out from it (size with
``rows = max_rows``), each 1024-byte aligned.

mHC (four BF16 streams ``[rows, 4, H]``): the DeepSeek V4 programs
(``dsv4_mhc``) at this width with eps 1e-5 (stream RMS and the fused
input/post-attention norms) and hc_eps 1e-6: ``mhc_pre`` (first layer),
``mhc_post_pre`` (sublayer output back into the streams + the next site's
collapse and norm), ``mhc_post`` (last layer). The head is the unweighted
stream mean (``compile_glmf_head_aot``).

``compile_glmf_kda_aot(g, max_rows=R)`` (one Kimi Delta Attention layer)::

    x           bf16 [rows,H]            in   input_layernorm output (mhc y)
    w_in        bf16 [P,H]               in   cat(q_proj, k_proj, v_proj, f_a_proj, g_a_proj, b_proj)
    w_fg        bf16 [2,D,128]           in   cat(f_b_proj, g_b_proj)
    conv_w      f32  [3D,4]              in   cat(q_conv1d, k_conv1d, v_conv1d) (the 1 axis dropped)
    a_log       f32  [64]                in   A_log
    dt_bias     f32  [D]                 in
    o_norm      bf16 [128]               in   o_norm.weight
    w_o         bf16 [H,D]               in   o_proj
    conv_state  bf16 [slots,3,3D]        inout  last three in-projection q|k|v rows per sequence
    state       f32  [slots,64,128,128]  inout  recurrent state [head, v, k] per sequence
    slots       i32  [rows]              in   state slot of each row's sequence (<0: zero state, not kept)
    seq_first   i32  [rows]              in   first step row of each row's sequence (rows of a sequence
                                              are contiguous and in position order)
    out         bf16 [rows,H]            out  attention output (before the mHC post)
    scratch     u8   kda_scratch_bytes(rows)
    rows        int32

``compile_glmf_mla_producer_aot(g, max_rows=R)`` (MLA without RoPE)::

    x           bf16 [rows,H]            in
    kv_slots    i64  [rows]              in   latent record slot page*64+row (<0 skips)
    w_qkv_a     bf16 [Q+512,H]           in   cat(q_a_proj, kv_a_proj_with_mqa)
    q_a_norm    bf16 [Q]                 in
    kv_a_norm   bf16 [512]               in
    w_q_b       bf16 [N*256,Q]           in
    w_uk        bf16 [N,512,256]         in   kv_b_proj key rows per head, transposed
    kv_cache    u8   [pages,64*528]      inout  512 E4M3 + 4 FP32 group scales per record
    query       bf16 [rows,N,512]        out  q_nope @ W_UK
    q_resid     bf16 [rows,Q]            out  q_a_layernorm(q_a(x)), for the indexer
    scratch     u8   mla_producer_scratch_bytes(rows)
    rows        int32

``glmf_sparse_mla_{prefill,decode}`` is ``glm_sparse_mla`` at this geometry
(``ModelType.GLM_NEXT``: 512-wide query, 528-byte records, 2112-slot index
rows whose ``lengths`` stay <= 2051), ``glmf_o`` is ``glm_o`` (W_UV per
head then o_proj).

FFN side: ``glmf_ffn`` (SwiGLU clamped at 10: dense I=12288, shared expert
I=2048), ``glmf_router_scores`` (FP32 logits ``x @ gate^T`` over the BF16
gate weight; the sigmoid top-8 stays native), ``glmf_expert_input_quant``
(FP8 K32 wire rows), ``glmf_add`` (``bf16(routed + shared)``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ._common import FLASH, GLM53_FLASH, AotProgram, GLMFGeometry, Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, GlmRankNormPackKV, GlmSwiGLU, glm_projection
from ._glmf_kernels import (
    GlmfAdd,
    GlmfIndexExpand,
    GlmfIndexPost,
    GlmfPoolKeys,
    REPLAY_ROWS,
    GlmfKdaCommit,
    GlmfKdaConv,
    GlmfKdaConvCommit,
    GlmfKdaConvRows,
    GlmfKdaConvState,
    GlmfKdaGatedNorm,
    GlmfKdaRecurrent,
    GlmfMeanNorm,
    kda_replay_layout,
)

__all__ = [
    "compile_glmf_add_aot",
    "compile_glmf_expert_input_quant_aot",
    "compile_glmf_ffn_aot",
    "compile_glmf_head_aot",
    "compile_glmf_head_fp8_aot",
    "compile_glmf_kda_commit_aot",
    "compile_glmf_index_expand_aot",
    "compile_glmf_index_producer_aot",
    "compile_glmf_index_topk_aot",
    "compile_glmf_kda_aot",
    "compile_glmf_mla_producer_aot",
    "compile_glmf_o_aot",
    "compile_glmf_router_scores_aot",
    "kda_scratch_bytes",
    "mhc_geometry",
    "mla_producer_scratch_bytes",
]

_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


def _check_rows(max_rows: int) -> int:
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(max_rows)


def mhc_geometry(g: GLMFGeometry = GLM53_FLASH):
    """The DeepSeek V4 mHC programs' geometry at GLM 5.3 Flash's width and epsilons."""
    return replace(FLASH, name="glmf", hidden=g.hidden, norm_eps=g.norm_eps, hc_eps=g.hc_eps,
                   hc_sinkhorn_iters=g.hc_sinkhorn_iters)


class _Fp8Switch:
    """A projection whose rows may read FP8. Decode programs: ``MmaFp8Gemv``
    when ``rows <= fp8_rows`` (a launch scalar, 0 turns FP8 off; at most
    ``FP8_ROWS``), else the BF16 projection. ``row_scales``: FP32 scales
    ``[N, K/128]`` (per-row quantization of a BF16 weight) instead of the
    checkpoint's 128x128 block grid.

    Prefill programs (``prefill_rows``): rows past the skinny BF16 GEMV run
    the block-FP8 GEMM (``BlockFp8Projection``: E4M3 activations per row and
    128-K block, FP32 scales) when ``fp8_rows`` is nonzero; per-row weight
    scales are then stored K-block major, ``[K/128, N]``. ``qscratch`` holds
    the quantized rows (``quant_scratch_bytes``); ``prefill_mask`` selects the
    ``fp8_rows`` bits that turn this projection's FP8 route on."""

    def __init__(self, n: int, k: int, *, fp8: bool, row_scales: bool = False, prefill_rows: int | None = None,
                 prefill_mask: int = 0xFF, wide_rows: int = 0):
        from ._fp8_weights import MmaFp8Gemv, gemv_warps
        from ._glmf_fp8 import BlockFp8Projection

        self.n, self.k = int(n), int(k)
        self.bf16 = glm_projection(self.n, self.k, wide=prefill_rows is not None)
        self.prefill = None
        self.fp8 = None
        self.fp8_wide = None
        self.prefill_mask = int(prefill_mask)
        if fp8 and prefill_rows is not None:
            self.prefill = BlockFp8Projection(self.n, self.k, int(prefill_rows), row_scales=row_scales)
        elif fp8:
            warps, groups = FP8_GEMV_CONFIG.get((self.n, self.k), (gemv_warps(self.k), 4))
            self.fp8 = MmaFp8Gemv(self.n, self.k, max_rows=FP8_ROWS, warps=warps, groups=groups,
                                  row_scales=row_scales)
            # wide_rows: steps of FP8_ROWS < rows <= wide_rows (up to fp8_rows) run a
            # multi-tile GEMV over the same E4M3 copy instead of the BF16 projection.
            if int(wide_rows) > FP8_ROWS:
                self.fp8_wide = MmaFp8Gemv(self.n, self.k, max_rows=int(wide_rows), warps=warps, groups=groups,
                                           row_scales=row_scales)
        self.max_fp8_rows = FP8_ROWS if self.fp8_wide is None else int(wide_rows)

    def key(self) -> tuple:
        return (self.bf16.key(), None if self.fp8 is None else self.fp8.key(),
                None if self.prefill is None else (self.prefill.key(), self.prefill_mask),
                None if self.fp8_wide is None else self.fp8_wide.key())

    @cute.jit
    def run(self, x: cute.Pointer, w: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
            rows: Int32, fp8_rows: Int32, qscratch: Int64, stream: cuda.CUstream):
        """``__call__`` with the prefill route's quantization scratch."""
        if cutlass.const_expr(self.prefill is not None):
            if (fp8_rows & Int32(self.prefill_mask)) != Int32(0) and rows > Int32(self.bf16.max_skinny_rows):
                self.prefill(x, w_fp8, scale, out, qscratch, rows, stream)
            else:
                self.bf16(x, w, out, rows, stream)
        else:
            self(x, w, w_fp8, scale, out, rows, fp8_rows, stream)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.fp8 is None):
            self.bf16(x, w, out, rows, stream)
        else:
            limit = fp8_rows
            if limit > Int32(self.max_fp8_rows):
                limit = Int32(self.max_fp8_rows)
            if rows <= limit:
                if cutlass.const_expr(self.fp8_wide is None):
                    self.fp8(x, w_fp8, scale, out, rows, stream)
                else:
                    if rows <= Int32(FP8_ROWS):
                        self.fp8(x, w_fp8, scale, out, rows, stream)
                    else:
                        self.fp8_wide(x, w_fp8, scale, out, rows, stream)
            else:
                self.bf16(x, w, out, rows, stream)


# Most rows a decode program runs through the FP8 GEMV (MmaFp8Gemv's M-tile ceiling).
FP8_ROWS = 16
# MmaFp8Gemv (warps, groups) per GLM 5.3 Flash projection (RTX PRO 6000 at 325 W, one
# row, L2-cold, us, default 4 warps x 4 groups -> chosen): KDA in 24896x4096 67.8 -> 62.7,
# KDA o 4096x8192 23.5 -> 22.5, MLA o 4096x16384 44.6 -> 44.0, q_b 16384x1536 18.6 -> 16.9,
# qkv_a 2048x4096 12.4 -> 12.0, shared gate_up 4096x4096 13.7 -> 13.0, shared down
# 4096x2048 12.4 -> 12.0, dense gate_up 24576x4096 69.9 -> 64.5, dense down 4096x12288
# 38.4 -> 34.5, FP32 LM head 154880x4096 387 -> 385 (BF16 cuBLAS head: 801).
FP8_GEMV_CONFIG = {
    (24896, 4096): (8, 2), (4096, 8192): (8, 2), (4096, 16384): (4, 2), (16384, 1536): (2, 2),
    (2048, 4096): (8, 4), (4096, 4096): (8, 2), (4096, 2048): (2, 2), (24576, 4096): (8, 4),
    (4096, 12288): (8, 2), (154880, 4096): (8, 2),
    # MiMo V2 Flash, per-row scales (one row / 16 rows, us, default -> chosen): full qkv
    # 13568x4096 41.8/54.0 -> 35.8/43.9, SWA qkv 14848x4096 42.3/57.8 -> 38.6/47.9, dense
    # gate_up 32768x4096 80.2 -> 78.4, LM head 152576x4096 380.8 -> 378.9.
    (13568, 4096): (2, 2), (14848, 4096): (2, 2), (32768, 4096): (8, 2), (152576, 4096): (8, 2),
}


def fp8_ops(name: str, n: int, k: int, row_scales: bool, prefill: bool = False) -> tuple:
    """``{name}_fp8`` E4M3 [n, k] and ``{name}_scale`` FP32 (``[n, k/128]`` row scales or the 128x128
    grid); prefill programs take per-row scales K-block major as ``{name}_kscale`` ``[k/128, n]``."""
    kb = -(-int(k) // 128)
    if prefill and row_scales:
        return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]"),
                Operand(f"{name}_kscale", torch.float32, f"[{kb},{n}]", note="per-row x 128-K scales, K-block major"))
    rows = int(n) if row_scales else -(-int(n) // 128)
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]"),
            Operand(f"{name}_scale", torch.float32, f"[{rows},{kb}]", align=16 if prefill else 4,
                    note="per-row x 128-K scales" if row_scales else "checkpoint 128x128 block scales"))


def _fp8_mode(fp8) -> tuple[bool, bool]:
    """``fp8`` of the compile functions: False, True (decode FP8 GEMVs) or "prefill"
    (block-FP8 GEMMs); returns (fp8, prefill)."""
    if fp8 not in (False, True, "prefill"):
        raise ValueError(f"fp8 must be False, True or 'prefill', got {fp8!r}")
    return bool(fp8), fp8 == "prefill"


def _fp8_scalars(fp8, prefill: bool) -> tuple:
    if prefill:
        return (Scalar("rows"), Scalar("fp8_rows", note="prefill: nonzero runs rows past the skinny GEMV "
                                                        "through the block-FP8 GEMMs"))
    return (Scalar("rows"), Scalar("fp8_rows")) if fp8 else (Scalar("rows"),)


def _w8_mode(fp8_only) -> bool | None:
    """``fp8_only`` of the compile functions: None (the ``fp8`` routes over BF16 + FP8 copies),
    ``"decode"`` or ``"prefill"`` (E4M3 weights only); returns whether a prefill program."""
    if fp8_only is None:
        return None
    from ._fp8_weights import check_w8_mode

    return check_w8_mode(fp8_only) == "prefill"


def _w8(n: int, k: int, prefill_rows: int | None):
    """An ``Fp8Projection`` over the checkpoint's 128x128 grid with this family's GEMV tuning."""
    from ._fp8_weights import Fp8Projection, gemv_warps

    warps, groups = FP8_GEMV_CONFIG.get((n, k), (gemv_warps(k), 4))
    return Fp8Projection(n, k, prefill_rows=prefill_rows, gemv_rows=FP8_ROWS, warps=warps, groups=groups)


def _w8_operands(name: str, n: int, k: int, prefill: bool) -> tuple:
    from ._fp8_weights import fp8_only_operands

    return fp8_only_operands(name, n, k, prefill=prefill)


def _batched_warps(max_rows: int) -> int:
    """MMA warps (16 rows each) of the per-head batched BF16 GEMMs: 128-row
    tiles at prefill capacities (4096 rows: fg 125 -> 96 us, absorb 373 -> 308,
    uv 300 -> 276 on SM120), 64 below."""
    return 8 if int(max_rows) > CHUNKED_MIN_ROWS else 4


def _ptr(dtype, address: Int64, align: int = 16):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


# ---------------------------------------------------------------------------
# KDA layer
# ---------------------------------------------------------------------------


# Live rows above which a prefill-capacity KDA program takes the chunked
# recurrence (b12x delta_prefill: 16-token tiles on tensor cores) instead of
# the token-sequential one.
CHUNKED_MIN_ROWS = 64


class _SequenceMeta:
    """Writes one sequence's packed-prefill metadata: ``cu_seqlens = [0,
    rows]``, ``num_seqs = 1``, ``num_tokens = rows``, initial = final state
    slot ``slots[0]``, no checkpoint (``[-1]``, offset 0)."""

    @cute.jit
    def __call__(self, slots: cute.Pointer, meta: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(slots, meta, rows).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, slots: cute.Pointer, meta: cute.Pointer, rows: Int32):
        if cute.arch.thread_idx()[0] == 0:
            s = cute.make_tensor(slots, cute.make_layout((1,)))
            m = cute.make_tensor(meta, cute.make_layout((8,)))
            m[0] = Int32(0)
            m[1] = rows
            m[2] = Int32(1)
            m[3] = rows
            m[4] = s[0]
            m[5] = s[0]
            m[6] = Int32(-1)
            m[7] = Int32(0)


class _KdaChunked:
    """One sequence's KDA recurrence through b12x's chunked delta-rule
    prefill (prologue, then prepare + recurrence per window of tiles on one
    stream), over the same q|k|v, gate, beta, state and output buffers as
    ``GlmfKdaRecurrent``. Beta is ``sigmoid(b)`` in FP32 (the reference rounds
    it to BF16 first)."""

    def __init__(self, g: GLMFGeometry, max_rows: int):
        from b12x.sequence._shared.delta_prefill import _cute_kernels as dk
        from b12x.sequence._shared.delta_prefill.contract import materialize_layout
        from b12x.sequence._shared.delta_prefill.workspace import default_window_tiles
        from b12x.sequence.kda_prefill._impl import Caps, _Layout

        heads, d = g.kda_heads, g.kda_width
        caps = Caps(device=torch.device("cuda", torch.cuda.current_device()), max_tokens=int(max_rows), max_seqs=1,
                    max_state_slots=1, heads=heads)
        # The default KdaPrefillConfig: v_split 64, k_split 1, 3 stages.
        layout = materialize_layout(caps, layout_type=_Layout, v_split=64, k_split=1, stages=3,
                                    window_tiles=default_window_tiles(heads, int(max_rows), 1))
        self.layout, self.heads, self.d = layout, heads, d
        self.tiles_per_window = layout.window_tiles
        self.lower_bound = float(g.gate_lower_bound)
        self.meta = _SequenceMeta()
        self.prologue = dk._PrologueKernel(
            max_seqs=1, tiles_capacity=caps.tiles_capacity, window_tiles=layout.window_tiles,
            max_windows=layout.max_windows, flag_count=layout.workspace_windows * layout.window_tiles * heads)
        self.prepare = dk._PrepareKernel(
            heads=heads, key_heads=heads, is_gdn=False, tiles_capacity=caps.tiles_capacity,
            window_tiles=layout.window_tiles, qk_l2norm=True, a_log_type=cutlass.Float32,
            dt_bias_type=cutlass.Float32)
        self.recurrence = dk._RecurrenceKernel(
            heads=heads, tiles_capacity=caps.tiles_capacity, window_tiles=layout.window_tiles,
            rows=layout.recurrence_rows, v_split=64, k_split=1, stages=3, checkpoint_export=False,
            null_state_index=None, index_type=Int32, max_sequence_tiles=layout.max_sequence_tiles, is_gdn=False)
        self.nbytes = _align(int(layout.scratch_specs()[0].nbytes)) + _align(8 * 4)

    def key(self) -> tuple:
        lay = self.layout
        return (lay.window_tiles, lay.max_windows, lay.recurrence_rows, lay.v_split, lay.k_split, lay.stages)

    @cute.jit
    def __call__(self, qkv: cute.Pointer, g_raw: cute.Pointer, b_raw: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, out: cute.Pointer,
                 scratch: Int64, qkv_stride: cutlass.Constexpr, g_stride: cutlass.Constexpr,
                 b_stride: cutlass.Constexpr, rows: Int32, stream: cuda.CUstream):
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
        d = self.d
        q = _ptr(cutlass.BFloat16, Int64(qkv.toint()))
        k = _ptr(cutlass.BFloat16, Int64(qkv.toint()) + Int64(d * 2))
        v = _ptr(cutlass.BFloat16, Int64(qkv.toint()) + Int64(2 * d * 2))
        self.prologue(cu, num_seqs, band_base, sorted_seq, rank_of, pos_seq, pos_local, window_table, ready,
                      Int32(1), stream)
        tiles = (rows + Int32(15)) // Int32(16)
        windows = (tiles + Int32(self.tiles_per_window - 1)) // Int32(self.tiles_per_window)
        for w in cutlass.range_constexpr(lay.max_windows):
            if Int32(w) < windows:
                self.prepare(q, k, g_raw, b_raw, a_log, dt_bias, cu, pos_seq, pos_local, ready,
                             _ptr(cutlass.BFloat16, ws), _ptr(cutlass.Float32, ws),
                             Int64(qkv_stride), Int64(qkv_stride), Int64(g_stride), Int64(b_stride), Int64(1),
                             Float32(128.0 ** -0.5), Float32(self.lower_bound * 1.4426950408889634),
                             Float32(1.0e-6), Int32(w), stream)
                self.recurrence(v, cu, band_base, sorted_seq, window_table, initial, final, checkpoint,
                                checkpoint_offsets, num_seqs, ready, _ptr(cutlass.Int8, ws, 16),
                                state, out, Int64(qkv_stride), Int64(d), Int64(self.heads * 128 * 128), Int64(1),
                                Int32(w), stream)


def kda_scratch_bytes(g: GLMFGeometry, rows: int, chunked: int = 0, prefill_fp8: bool = False) -> int:
    """proj [rows,P], f|gate [rows,2,D], conv q|k|v [rows,3D], o [rows,D], y [rows,D] (BF16), then
    the chunked recurrence's workspace (``chunked`` bytes, prefill capacities), then the quantized
    rows of the block-FP8 prefill projections (``prefill_fp8``)."""
    from ._glmf_fp8 import quant_scratch_bytes

    rows = max(int(rows), 1)
    d = g.kda_width
    quant = quant_scratch_bytes(max(g.hidden, d), rows) if prefill_fp8 else 0
    return (_align(rows * g.kda_in_width * 2) + _align(rows * 2 * d * 2) + _align(rows * 3 * d * 2)
            + 2 * _align(rows * d * 2) + _align(chunked) + quant)


class _Kda:
    def __init__(self, g: GLMFGeometry, max_rows: int = 64, fp8=False):
        self.g = g
        fp8, prefill = _fp8_mode(fp8)
        self.chunked = _KdaChunked(g, max_rows) if int(max_rows) > CHUNKED_MIN_ROWS else None
        d, p = g.kda_width, g.kda_in_width
        self.d, self.p = d, p
        prefill_rows = int(max_rows) if prefill else None
        self.chunked_bytes = 0 if self.chunked is None else self.chunked.nbytes
        # Prefill: fp8_rows bit 0 turns the in-projection's FP8 GEMM on, bit 1 o_proj's.
        self.in_proj = _Fp8Switch(p, g.hidden, fp8=fp8, row_scales=True, prefill_rows=prefill_rows, prefill_mask=1)
        # f_b(f_a) and g_b(g_a): two 128 -> D products off the in-projection row.
        self.fg = BatchedBf16Gemm(n=d, k=g.kda_head_dim, batch=2, a_row=p, a_batch=g.kda_head_dim,
                                  o_row=2 * d, o_batch=d, compute_warps=_batched_warps(max_rows))
        # Prefill capacities: the row-blocked conv and the scanning conv-state update.
        wide = int(max_rows) > CHUNKED_MIN_ROWS
        self.conv = (GlmfKdaConvRows if wide else GlmfKdaConv)(channels=3 * d, proj_width=p)
        self.conv_state = GlmfKdaConvState(channels=3 * d, proj_width=p, row_block=64 if wide else 0)
        self.recurrent = GlmfKdaRecurrent(heads=g.kda_heads, lower_bound=g.gate_lower_bound, qkv_width=3 * d,
                                          g_stride=2 * d, b_stride=p)
        self.norm = GlmfKdaGatedNorm(heads=g.kda_heads, eps=g.norm_eps, gate_stride=2 * d)
        self.o_proj = _Fp8Switch(g.hidden, d, fp8=fp8, row_scales=True, prefill_rows=prefill_rows, prefill_mask=2)

    def key(self) -> tuple:
        return (self.in_proj.key(), self.fg.key(), self.o_proj.key(), self.g,
                None if self.chunked is None else self.chunked.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_fg: cute.Pointer, conv_w: cute.Pointer,
                 a_log: cute.Pointer, dt_bias: cute.Pointer, o_norm: cute.Pointer, w_o: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in, w_in, w_fg, conv_w, a_log, dt_bias, o_norm, w_o, w_o, w_o, conv_state, state, slots,
                  seq_first, out, state, scratch, rows, Int32(0), Int32(0), stream)

    @cute.jit
    def body(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
             w_fg: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
             o_norm: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer, w_o_scale: cute.Pointer,
             conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
             out: cute.Pointer, replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             spec: Int32, stream: cuda.CUstream):
        d, p = self.d, self.p
        m = Int64(rows)
        base = Int64(scratch.toint())
        fg_off = base + _align_i64(m * Int64(p * 2))
        qkv_off = fg_off + _align_i64(m * Int64(4 * d))
        o_off = qkv_off + _align_i64(m * Int64(6 * d))
        y_off = o_off + _align_i64(m * Int64(2 * d))
        # Quantized prefill rows follow the chunked recurrence's workspace.
        qscratch = y_off + _align_i64(m * Int64(2 * d)) + Int64(_align(self.chunked_bytes))
        bf16 = cutlass.BFloat16
        proj = _ptr(bf16, base)
        self.in_proj.run(x, w_in, w_in_fp8, w_in_scale, proj, rows, fp8_rows, qscratch, stream)
        self.fg(_ptr(bf16, base + Int64(3 * d * 2)), w_fg, _ptr(bf16, fg_off), rows, stream)
        self.conv(proj, conv_w, conv_state, slots, seq_first, _ptr(bf16, qkv_off), rows, stream)
        _, proj_off, _ = kda_replay_layout(self.g.kda_heads, 3 * d)
        self.conv_state(proj, conv_state, slots, seq_first, _ptr(bf16, Int64(replay.toint()) + Int64(proj_off)), spec,
                        rows, stream)
        b_raw = _ptr(bf16, base + Int64((3 * d + 256) * 2), 2)
        if cutlass.const_expr(self.chunked is None):
            self.recurrent(_ptr(bf16, qkv_off), _ptr(bf16, fg_off), b_raw, a_log, dt_bias, state, slots,
                           _ptr(bf16, o_off), replay, spec, rows, stream)
        else:
            if rows > Int32(CHUNKED_MIN_ROWS):
                self.chunked(_ptr(bf16, qkv_off), _ptr(bf16, fg_off), b_raw, a_log, dt_bias, state, slots,
                             _ptr(bf16, o_off), y_off + _align_i64(m * Int64(2 * d)), 3 * d, 2 * d, p, rows, stream)
            else:
                self.recurrent(_ptr(bf16, qkv_off), _ptr(bf16, fg_off), b_raw, a_log, dt_bias, state, slots,
                               _ptr(bf16, o_off), replay, spec, rows, stream)
        self.norm(_ptr(bf16, o_off), _ptr(bf16, fg_off + Int64(d * 2)), o_norm, _ptr(bf16, y_off), rows, stream)
        self.o_proj.run(_ptr(bf16, y_off), w_o, w_o_fp8, w_o_scale, out, rows, fp8_rows, qscratch, stream)


class _KdaFp8(_Kda):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
                 w_fg: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                 o_norm: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer, w_o_scale: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 spec: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in_fp8, w_in_scale, w_fg, conv_w, a_log, dt_bias, o_norm, w_o, w_o_fp8, w_o_scale,
                  conv_state, state, slots, seq_first, out, replay, scratch, rows, fp8_rows, spec, stream)


class _KdaFp8Prefill(_Kda):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_in_fp8: cute.Pointer, w_in_kscale: cute.Pointer,
                 w_fg: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                 o_norm: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer, w_o_kscale: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(x, w_in, w_in_fp8, w_in_kscale, w_fg, conv_w, a_log, dt_bias, o_norm, w_o, w_o_fp8, w_o_kscale,
                  conv_state, state, slots, seq_first, out, state, scratch, rows, fp8_rows, Int32(0), stream)


def compile_glmf_kda_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int, fp8=False) -> AotProgram:
    """One KDA layer for ``rows <= max_rows``; see the module docstring. ``fp8``
    adds ``w_in``/``w_o`` E4M3 copies with per-row scales and the ``fp8_rows``
    scalar (decode steps up to that many rows read them); ``fp8="prefill"``
    adds them with K-block-major scales (``w_in_kscale``/``w_o_kscale``) and
    the ``fp8_rows`` scalar as an on/off switch (block-FP8 GEMMs, no replay record)."""
    max_rows = _check_rows(max_rows)
    fp8_on, prefill = _fp8_mode(fp8)
    launch = (_KdaFp8Prefill if prefill else _KdaFp8 if fp8_on else _Kda)(g, max_rows, fp8)
    chunked = 0 if launch.chunked is None else launch.chunked.nbytes
    h, d, p, heads = g.hidden, g.kda_width, g.kda_in_width, g.kda_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_in", torch.bfloat16, f"[{p},{h}]"),
        *(fp8_ops("w_in", p, h, True, prefill) if fp8_on else ()),
        Operand("w_fg", torch.bfloat16, f"[2,{d},{g.kda_head_dim}]"),
        Operand("conv_w", torch.float32, f"[{3 * d},4]", align=4),
        Operand("a_log", torch.float32, f"[{heads}]", align=4),
        Operand("dt_bias", torch.float32, f"[{d}]", align=4),
        Operand("o_norm", torch.bfloat16, f"[{g.kda_head_dim}]"),
        Operand("w_o", torch.bfloat16, f"[{h},{d}]"),
        *(fp8_ops("w_o", h, d, True, prefill) if fp8_on else ()),
        Operand("conv_state", torch.bfloat16, f"[slots,3,{3 * d}]", "inout", align=2),
        Operand("state", torch.float32, f"[slots,{heads},128,128]", "inout"),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        *((Operand("replay", torch.float32, f"[{kda_replay_layout(heads, 3 * d)[2] // 4}]", "inout"),)
          if fp8_on and not prefill else ()),
        Operand("scratch", torch.uint8, "[kda_scratch_bytes]", "scratch"),
    )
    scalars = (_fp8_scalars(fp8, True) if prefill else
               (Scalar("rows"), Scalar("fp8_rows"), Scalar("spec")) if fp8_on else (Scalar("rows"),))
    return compile_program(
        launch, name="glmf_kda", operands=operands, scalars=scalars,
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "heads": heads, "head_dim": g.kda_head_dim, "max_rows": max_rows,
                  "in_width": p, "lower_bound": g.gate_lower_bound, "eps": g.norm_eps,
                  "chunked_min_rows": CHUNKED_MIN_ROWS if chunked else None},
        scratch={"scratch": lambda rows: kda_scratch_bytes(g, rows, chunked, prefill)},
        doc=__doc__,
    )


class _KdaCommit:
    def __init__(self, g: GLMFGeometry):
        self.g = g
        self.recurrent = GlmfKdaCommit(heads=g.kda_heads, channels=3 * g.kda_width)
        self.conv = GlmfKdaConvCommit(heads=g.kda_heads, channels=3 * g.kda_width)

    @cute.jit
    def __call__(self, state: cute.Pointer, conv_state: cute.Pointer, replay: cute.Pointer, tables: cute.Pointer,
                 sequences: Int32, layers: Int32, slots: Int32, stream: cuda.CUstream):
        self.recurrent(state, replay, tables, sequences, layers, slots, stream)
        self.conv(conv_state, replay, tables, sequences, layers, slots, stream)


def compile_glmf_kda_commit_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Verify-by-replay for KDA: after a speculative decode step (``glmf_kda``
    with ``spec`` = 1, which records each row's replay inputs and leaves the
    state alone), apply each sequence's accepted rows to its recurrent and
    conv state in every KDA layer.

    ``state`` FP32 ``[layers, slots, H, 128, 128]`` and ``conv_state`` BF16
    ``[layers, slots, 3, 3D]`` (the layers' pools back to back), ``replay``
    the layers' records back to back (``replay_bytes`` each), ``tables`` i32
    ``[3, sequences]``: state slot (negative: skip), first step row and
    accepted rows per sequence. The recurrent update repeats ``glmf_kda``'s
    arithmetic, so the state is bit-identical to serial steps over those rows.
    """
    launch = _KdaCommit(g)
    heads, d = g.kda_heads, g.kda_width
    record = kda_replay_layout(heads, 3 * d)[2]
    return compile_program(
        launch, name="glmf_kda_commit",
        operands=(Operand("state", torch.float32, f"[layers,slots,{heads},128,128]", "inout"),
                  Operand("conv_state", torch.bfloat16, f"[layers,slots,3,{3 * d}]", "inout", align=2),
                  Operand("replay", torch.float32, f"[layers,{record // 4}]"),
                  Operand("tables", torch.int32, "[3,sequences]", align=4)),
        scalars=(Scalar("sequences"), Scalar("layers"), Scalar("slots")),
        key=(heads, d, REPLAY_ROWS),
        geometry={"heads": heads, "channels": 3 * d, "replay_rows": REPLAY_ROWS, "replay_bytes": record},
        doc=compile_glmf_kda_commit_aot.__doc__,
    )


# ---------------------------------------------------------------------------
# MLA producer (no RoPE)
# ---------------------------------------------------------------------------


def mla_producer_scratch_bytes(g: GLMFGeometry, rows: int, prefill_fp8: bool = False) -> int:
    """qkv_a BF16 [rows, Q+512] then q_b BF16 [rows, N*256], then the quantized rows of the
    block-FP8 prefill projections (``prefill_fp8``)."""
    from ._glmf_fp8 import quant_scratch_bytes

    rows = max(int(rows), 1)
    quant = quant_scratch_bytes(max(g.hidden, g.q_lora_rank), rows) if prefill_fp8 else 0
    return _align(rows * g.qkv_a_width * 2) + _align(rows * g.heads * g.qk_head_dim * 2) + quant


class _MlaProducer:
    def __init__(self, g: GLMFGeometry, fp8=False, max_rows: int = 64):
        self.g = g
        n, q = g.heads, g.q_lora_rank
        fp8, prefill = _fp8_mode(fp8)
        prefill_rows = int(max_rows) if prefill else None
        self.qkv_a = _Fp8Switch(g.qkv_a_width, g.hidden, fp8=fp8, prefill_rows=prefill_rows)
        self.q_b = _Fp8Switch(n * g.qk_head_dim, q, fp8=fp8, prefill_rows=prefill_rows)
        self.pack = GlmRankNormPackKV(q_rank=q, eps=g.norm_eps, page_rows=g.page_rows,
                                      record_bytes=g.record_bytes, rope=0)
        self.absorb = BatchedBf16Gemm(n=g.kv_lora_rank, k=g.qk_nope_dim, batch=n, a_row=n * g.qk_head_dim,
                                      a_batch=g.qk_head_dim, o_row=n * g.latent_dim, o_batch=g.latent_dim,
                                      compute_warps=_batched_warps(max_rows))

    def key(self) -> tuple:
        return (self.qkv_a.key(), self.q_b.key(), self.absorb.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b: cute.Pointer, w_uk: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, kv_slots, w_qkv_a, w_qkv_a, w_qkv_a, q_a_norm, kv_a_norm, w_q_b, w_q_b, w_q_b, w_uk, kv_cache,
                  query, q_resid, scratch, rows, Int32(0), stream)

    @cute.jit
    def body(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a: cute.Pointer, w_qkv_a_fp8: cute.Pointer,
             w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer, w_q_b: cute.Pointer,
             w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer, w_uk: cute.Pointer, kv_cache: cute.Pointer,
             query: cute.Pointer, q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = _ptr(cutlass.BFloat16, base)
        q_at = base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2))
        q = _ptr(cutlass.BFloat16, q_at)
        qscratch = q_at + _align_i64(Int64(rows) * Int64(g.heads * g.qk_head_dim * 2))
        self.qkv_a.run(x, w_qkv_a, w_qkv_a_fp8, w_qkv_a_scale, qkv, rows, fp8_rows, qscratch, stream)
        # No RoPE: the pack kernel never reads positions or cos_sin.
        self.pack(qkv, q_a_norm, kv_a_norm, kv_slots, kv_slots, _ptr(cutlass.Float32, Int64(kv_slots.toint()), 4),
                  q_resid, kv_cache, rows, stream)
        self.q_b.run(q_resid, w_q_b, w_q_b_fp8, w_q_b_scale, q, rows, fp8_rows, qscratch, stream)
        self.absorb(q, w_uk, query, rows, stream)


class _MlaProducerFp8(_MlaProducer):
    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a: cute.Pointer, w_qkv_a_fp8: cute.Pointer,
                 w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer, w_q_b: cute.Pointer,
                 w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer, w_uk: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, kv_slots, w_qkv_a, w_qkv_a_fp8, w_qkv_a_scale, q_a_norm, kv_a_norm, w_q_b, w_q_b_fp8,
                  w_q_b_scale, w_uk, kv_cache, query, q_resid, scratch, rows, fp8_rows, stream)


class _MlaProducerW8(_MlaProducer):
    """The MLA producer over FP8-only ``q_a|kv_a`` and ``q_b`` (``Fp8Projection``)."""

    def __init__(self, g: GLMFGeometry, max_rows: int, prefill: bool):
        super().__init__(g, False, max_rows)
        prefill_rows = int(max_rows) if prefill else None
        self.qkv_a = _w8(g.qkv_a_width, g.hidden, prefill_rows)
        self.q_b = _w8(g.heads * g.qk_head_dim, g.q_lora_rank, prefill_rows)

    def key(self) -> tuple:
        return ("w8",) + super().key()

    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a_fp8: cute.Pointer,
                 w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer,
                 w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer, w_uk: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = _ptr(cutlass.BFloat16, base)
        q_at = base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2))
        q = _ptr(cutlass.BFloat16, q_at)
        qscratch = q_at + _align_i64(Int64(rows) * Int64(g.heads * g.qk_head_dim * 2))
        self.qkv_a(x, w_qkv_a_fp8, w_qkv_a_scale, qkv, rows, fp8_rows, qscratch, stream)
        self.pack(qkv, q_a_norm, kv_a_norm, kv_slots, kv_slots, _ptr(cutlass.Float32, Int64(kv_slots.toint()), 4),
                  q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b_fp8, w_q_b_scale, q, rows, fp8_rows, qscratch, stream)
        self.absorb(q, w_uk, query, rows, stream)


def compile_glmf_mla_producer_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int, fp8=False,
                                  fp8_only: str | None = None) -> AotProgram:
    """MLA producer for ``rows <= max_rows``; see the module docstring. ``fp8``
    adds the checkpoint's E4M3 ``w_qkv_a``/``w_q_b`` with 128x128 scales and ``fp8_rows``
    (``fp8="prefill"``: ``fp8_rows`` switches block-FP8 GEMMs). ``fp8_only`` (``"decode"`` /
    ``"prefill"``) takes ``w_qkv_a``/``w_q_b`` as E4M3 + scales only: decode rows up to
    ``fp8_rows`` run the GEMV, the rest the W8A16 TMA GEMM; prefill rows run W8A8 when
    ``fp8_rows`` is nonzero, else W8A16."""
    max_rows = _check_rows(max_rows)
    w8_prefill = _w8_mode(fp8_only)
    if w8_prefill is not None:
        launch = _MlaProducerW8(g, max_rows, w8_prefill)
        h, q, n = g.hidden, g.q_lora_rank, g.heads
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{h}]"),
            Operand("kv_slots", torch.int64, "[rows]", align=8),
            *_w8_operands("w_qkv_a", g.qkv_a_width, h, w8_prefill),
            Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
            Operand("kv_a_norm", torch.bfloat16, f"[{g.kv_lora_rank}]"),
            *_w8_operands("w_q_b", n * g.qk_head_dim, q, w8_prefill),
            Operand("w_uk", torch.bfloat16, f"[{n},{g.kv_lora_rank},{g.qk_nope_dim}]"),
            Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
            Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
            Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
            Operand("scratch", torch.uint8, "[mla_producer_scratch_bytes]", "scratch"),
        )
        return compile_program(
            launch, name="glmf_mla_producer", operands=operands, scalars=_fp8_scalars(True, w8_prefill),
            key=(max_rows, "w8", fp8_only, launch.key()),
            geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows, "fp8_weights": "only",
                      "record_bytes": g.record_bytes, "page_rows": g.page_rows, "eps": g.norm_eps},
            scratch={"scratch": lambda rows: mla_producer_scratch_bytes(g, rows, w8_prefill)},
            doc=__doc__,
        )
    fp8_on, prefill = _fp8_mode(fp8)
    launch = (_MlaProducerFp8 if fp8_on else _MlaProducer)(g, fp8, max_rows)
    h, q, n = g.hidden, g.q_lora_rank, g.heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{h}]"),
        *(fp8_ops("w_qkv_a", g.qkv_a_width, h, False, prefill) if fp8_on else ()),
        Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
        Operand("kv_a_norm", torch.bfloat16, f"[{g.kv_lora_rank}]"),
        Operand("w_q_b", torch.bfloat16, f"[{n * g.qk_head_dim},{q}]"),
        *(fp8_ops("w_q_b", n * g.qk_head_dim, q, False, prefill) if fp8_on else ()),
        Operand("w_uk", torch.bfloat16, f"[{n},{g.kv_lora_rank},{g.qk_nope_dim}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
        Operand("scratch", torch.uint8, "[mla_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_mla_producer", operands=operands, scalars=_fp8_scalars(fp8_on, prefill),
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows,
                  "record_bytes": g.record_bytes, "page_rows": g.page_rows, "eps": g.norm_eps},
        scratch={"scratch": lambda rows: mla_producer_scratch_bytes(g, rows, prefill)},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# FFN side, head
# ---------------------------------------------------------------------------


class _GlmfFfn:
    def __init__(self, g: GLMFGeometry, inter: int, fp8, max_rows: int = 64):
        self.h, self.i = g.hidden, int(inter)
        fp8, prefill = _fp8_mode(fp8)
        prefill_rows = int(max_rows) if prefill else None
        self.gate_up = _Fp8Switch(2 * self.i, self.h, fp8=fp8, prefill_rows=prefill_rows)
        self.down = _Fp8Switch(self.h, self.i, fp8=fp8, prefill_rows=prefill_rows)
        self.swiglu = GlmSwiGLU(self.i, limit=g.swiglu_limit)

    def key(self) -> tuple:
        return (self.h, self.i, self.gate_up.key(), self.down.key(), self.swiglu.limit)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_down: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_gate_up, w_gate_up, w_gate_up, w_down, w_down, w_down, out, scratch, rows, Int32(0), stream)

    @cute.jit
    def body(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_gate_up_fp8: cute.Pointer,
             w_gate_up_scale: cute.Pointer, w_down: cute.Pointer, w_down_fp8: cute.Pointer,
             w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             stream: cuda.CUstream):
        base = Int64(scratch.toint())
        gate_up = _ptr(cutlass.BFloat16, base)
        hidden_at = base + _align_i64(Int64(rows) * Int64(4 * self.i))
        hidden = _ptr(cutlass.BFloat16, hidden_at)
        qscratch = hidden_at + _align_i64(Int64(rows) * Int64(2 * self.i))
        self.gate_up.run(x, w_gate_up, w_gate_up_fp8, w_gate_up_scale, gate_up, rows, fp8_rows, qscratch, stream)
        self.swiglu(gate_up, hidden, rows, stream)
        self.down.run(hidden, w_down, w_down_fp8, w_down_scale, out, rows, fp8_rows, qscratch, stream)


class _GlmfFfnFp8(_GlmfFfn):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_gate_up_fp8: cute.Pointer,
                 w_gate_up_scale: cute.Pointer, w_down: cute.Pointer, w_down_fp8: cute.Pointer,
                 w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, w_gate_up, w_gate_up_fp8, w_gate_up_scale, w_down, w_down_fp8, w_down_scale, out, scratch,
                  rows, fp8_rows, stream)


class _GlmfFfnW8(_GlmfFfn):
    def __init__(self, g: GLMFGeometry, inter: int, max_rows: int, prefill: bool):
        super().__init__(g, inter, False, max_rows)
        prefill_rows = int(max_rows) if prefill else None
        self.gate_up = _w8(2 * self.i, self.h, prefill_rows)
        self.down = _w8(self.h, self.i, prefill_rows)

    def key(self) -> tuple:
        return ("w8",) + super().key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up_fp8: cute.Pointer, w_gate_up_scale: cute.Pointer,
                 w_down_fp8: cute.Pointer, w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        gate_up = _ptr(cutlass.BFloat16, base)
        hidden_at = base + _align_i64(Int64(rows) * Int64(4 * self.i))
        hidden = _ptr(cutlass.BFloat16, hidden_at)
        qscratch = hidden_at + _align_i64(Int64(rows) * Int64(2 * self.i))
        self.gate_up(x, w_gate_up_fp8, w_gate_up_scale, gate_up, rows, fp8_rows, qscratch, stream)
        self.swiglu(gate_up, hidden, rows, stream)
        self.down(hidden, w_down_fp8, w_down_scale, out, rows, fp8_rows, qscratch, stream)


def compile_glmf_ffn_aot(g: GLMFGeometry = GLM53_FLASH, *, inter: int, max_rows: int, fp8=False,
                         fp8_only: str | None = None) -> AotProgram:
    """Clamped SwiGLU MLP: the ``glm_ffn`` ABI (x, w_gate_up, w_down, out, scratch; rows); ``fp8`` adds
    the checkpoint's E4M3 weights with 128x128 scales and ``fp8_rows`` (``fp8="prefill"``:
    ``fp8_rows`` switches block-FP8 GEMMs). ``fp8_only`` takes the weights as E4M3 + scales only
    (see ``compile_glmf_mla_producer_aot``)."""
    from ._glmf_fp8 import quant_scratch_bytes
    from .glm_ffn import ffn_scratch_bytes

    max_rows = _check_rows(max_rows)
    i = int(inter)
    w8_prefill = _w8_mode(fp8_only)
    if w8_prefill is not None:
        launch = _GlmfFfnW8(g, i, max_rows, w8_prefill)
        h = g.hidden
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{h}]"),
            *_w8_operands("w_gate_up", 2 * i, h, w8_prefill),
            *_w8_operands("w_down", h, i, w8_prefill),
            Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
            Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
        )
        return compile_program(
            launch, name="glmf_ffn", operands=operands, scalars=_fp8_scalars(True, w8_prefill),
            key=(max_rows, "w8", fp8_only, launch.key()),
            geometry={"hidden": h, "inter": i, "max_rows": max_rows, "swiglu_limit": g.swiglu_limit,
                      "fp8_weights": "only"},
            scratch={"scratch": lambda rows: ffn_scratch_bytes(i, rows)
                     + (quant_scratch_bytes(max(h, i), rows) if w8_prefill else 0)},
            doc=__doc__,
        )
    fp8_on, prefill = _fp8_mode(fp8)
    launch = (_GlmfFfnFp8 if fp8_on else _GlmfFfn)(g, i, fp8, max_rows)
    h = g.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{2 * i},{h}]"),
        *(fp8_ops("w_gate_up", 2 * i, h, False, prefill) if fp8_on else ()),
        Operand("w_down", torch.bfloat16, f"[{h},{i}]"),
        *(fp8_ops("w_down", h, i, False, prefill) if fp8_on else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_ffn", operands=operands, scalars=_fp8_scalars(fp8_on, prefill),
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "inter": i, "max_rows": max_rows, "swiglu_limit": g.swiglu_limit},
        scratch={"scratch": lambda rows: ffn_scratch_bytes(i, rows)
                 + (quant_scratch_bytes(max(h, i), rows) if prefill else 0)},
        doc=__doc__,
    )


class _RouterScores:
    def __init__(self, e: int, h: int):
        from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

        self.proj = RoutedBf16Projection(e, h, out_dtype=cutlass.Float32)

    def key(self) -> tuple:
        return self.proj.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, logits: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.proj(x, w, logits, rows, stream)


def compile_glmf_router_scores_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """FP32 router logits ``x @ gate^T`` (BF16 operands, FP32 accumulation)."""
    e, h = g.routed_experts, g.hidden
    launch = _RouterScores(e, h)
    return compile_program(
        launch, name="glmf_router_scores",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w", torch.bfloat16, f"[{e},{h}]", note="mlp.gate.weight"),
                  Operand("logits", torch.float32, f"[rows,{e}]", "out")),
        scalars=(Scalar("rows"),), key=(launch.key(),),
        geometry={"hidden": h, "experts": e, "top_k": g.top_k}, doc=__doc__,
    )


def compile_glmf_expert_input_quant_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Routed-expert BF16 rows -> FP8 K32 wire rows (H E4M3 bytes then H/32 UE8M0)."""
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    h = g.hidden
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="glmf_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4},
        doc=__doc__,
    )


def compile_glmf_add_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """``out = bf16(a + b)`` over ``[rows, H]`` (routed + shared expert outputs)."""
    h = g.hidden
    return compile_program(
        GlmfAdd(h), name="glmf_add",
        operands=(Operand("a", torch.bfloat16, f"[rows,{h}]"), Operand("b", torch.bfloat16, f"[rows,{h}]"),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"),), key=(h,), geometry={"hidden": h}, doc=__doc__,
    )


def compile_glmf_head_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Unweighted mean of the four streams, then ``model.norm``: ``out = w * rmsnorm(mean)``."""
    h = g.hidden
    return compile_program(
        GlmfMeanNorm(h, g.norm_eps, g.hc_mult), name="glmf_head",
        operands=(Operand("streams", torch.bfloat16, f"[rows,{g.hc_mult},{h}]"),
                  Operand("weight", torch.bfloat16, f"[{h}]"),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"),), key=(h, g.norm_eps, g.hc_mult), geometry={"hidden": h, "eps": g.norm_eps},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# DSA indexer over 4-token key pools
# ---------------------------------------------------------------------------


def index_producer_scratch_bytes(g: GLMFGeometry, rows: int) -> int:
    """Index query BF16 [rows, I*128] then [wk | weights_proj | gate] BF16 [rows, 256+I]."""
    rows = max(int(rows), 1)
    return _align(rows * g.index_heads * 128 * 2) + _align(rows * (256 + g.index_heads) * 2)


class _IndexProducer:
    def __init__(self, g: GLMFGeometry, max_rows: int = 64):
        self.g = g
        i = g.index_heads
        self.wq = glm_projection(i * 128, g.q_lora_rank, wide=int(max_rows) > CHUNKED_MIN_ROWS)
        self.wk = glm_projection(256 + i, g.hidden)
        self.post = GlmfIndexPost(heads=i, eps=g.index_norm_eps, weight_scale=float(i) ** -0.5 * 128.0 ** -0.5)
        self.pool = GlmfPoolKeys(kpool=g.index_kpool, page_rows=g.page_rows)

    def key(self) -> tuple:
        return (self.wq.key(), self.wk.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, q_resid: cute.Pointer, slots: cute.Pointer, pool_slots: cute.Pointer,
                 w_iq: cute.Pointer, w_ik: cute.Pointer, k_norm_w: cute.Pointer, k_norm_b: cute.Pointer,
                 ape: cute.Pointer, token_keys: cute.Pointer, index_cache: cute.Pointer, q_fp8: cute.Pointer,
                 head_weights: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        iq = _ptr(cutlass.BFloat16, base)
        kw = _ptr(cutlass.BFloat16, base + _align_i64(Int64(rows) * Int64(g.index_heads * 128 * 2)))
        self.wq(q_resid, w_iq, iq, rows, stream)
        self.wk(x, w_ik, kw, rows, stream)
        self.post(iq, kw, slots, k_norm_w, k_norm_b, q_fp8, head_weights, token_keys, rows, stream)
        self.pool(slots, pool_slots, ape, token_keys, index_cache, rows, stream)


def compile_glmf_index_producer_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int) -> AotProgram:
    """DSA index query, per-token keys and completed pool keys.

    ``x`` bf16 [rows,H], ``q_resid`` bf16 [rows,Q] (mla producer), ``slots``
    i64 [rows] (the rows' MLA record slots, also their token-key rows),
    ``pool_slots`` i64 [rows] (index-cache slot of the pool a row completes,
    else -1), ``w_iq`` bf16 [I*128,Q] (indexer.wq_b), ``w_ik`` bf16
    [256+I,H] (cat(indexer.wk, indexer.weights_proj,
    indexer.index_kpool_compress_gate)), ``k_norm_w``/``k_norm_b`` bf16
    [128], ``ape`` bf16 [4,128] (index_kpool_compress_ape), ``token_keys``
    bf16 [record slots,256] inout, ``index_cache`` u8 [pool_pages,8448]
    inout, ``q_fp8`` fp8 [rows,I,128] out, ``head_weights`` f32 [rows,I] out.
    """
    max_rows = _check_rows(max_rows)
    launch = _IndexProducer(g, max_rows)
    h, q, i = g.hidden, g.q_lora_rank, g.index_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]"),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("pool_slots", torch.int64, "[rows]", align=8),
        Operand("w_iq", torch.bfloat16, f"[{i * 128},{q}]"),
        Operand("w_ik", torch.bfloat16, f"[{256 + i},{h}]"),
        Operand("k_norm_w", torch.bfloat16, "[128]"),
        Operand("k_norm_b", torch.bfloat16, "[128]"),
        Operand("ape", torch.bfloat16, f"[{g.index_kpool},128]"),
        Operand("token_keys", torch.bfloat16, "[record_slots,256]", "inout"),
        Operand("index_cache", torch.uint8, "[pool_pages,8448]", "inout"),
        Operand("q_fp8", torch.float8_e4m3fn, f"[rows,{i},128]", "out"),
        Operand("head_weights", torch.float32, f"[rows,{i}]", "out"),
        Operand("scratch", torch.uint8, "[index_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_index_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "index_heads": i, "max_rows": max_rows, "kpool": g.index_kpool,
                  "eps": g.index_norm_eps},
        scratch={"scratch": lambda rows: index_producer_scratch_bytes(g, rows)},
        doc=compile_glmf_index_producer_aot.__doc__,
    )


@dataclass(frozen=True)
class _PoolTopK:
    """The DSA top-k over pools: ``index_topk / kpool`` of them."""

    index_topk: int
    index_heads: int
    index_page_bytes: int = 64 * (128 + 4)


def compile_glmf_index_topk_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int, max_pages: int,
                                mode: str = "prefill") -> AotProgram:
    """Top ``index_topk / kpool`` (512) pools per row: the GLM index top-k
    (``glm_index_topk``) over the pool index cache; ``cache_lengths`` are the
    complete pools each row sees (``(position + 1) // 4``)."""
    from .glm_indexer import compile_glm_index_topk_aot as topk

    pools = _PoolTopK(index_topk=g.index_topk // g.index_kpool, index_heads=g.index_heads)
    return topk(pools, max_rows=max_rows, max_pages=max_pages, mode=mode)


def compile_glmf_index_expand_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Top-k pools (or every earlier token up to ``index_topk + kpool - 1``)
    to MLA record slots plus the open tail pool.

    ``positions`` i64 [rows], ``pools`` i32 [rows, index_topk/kpool]
    (glmf_index_topk output), ``pool_logical`` i32 [pool_pages] (logical
    page of each pool-cache page in its sequence), ``page_table`` i32 (MLA
    record pages; per row ``stride`` entries, 0 = one shared table),
    ``indices`` i32 [rows, sparse_topk] out, ``lengths`` i32 [rows] out;
    scalars ``rows``, ``stride``.
    """
    pools = g.index_topk // g.index_kpool
    launch = GlmfIndexExpand(pools=pools, width=g.sparse_topk, dense_limit=g.index_topk + g.index_kpool - 1,
                             kpool=g.index_kpool)
    return compile_program(
        launch, name="glmf_index_expand",
        operands=(Operand("positions", torch.int64, "[rows]", align=8),
                  Operand("pools", torch.int32, f"[rows,{pools}]", align=4),
                  Operand("pool_logical", torch.int32, "[pool_pages]", align=4),
                  Operand("page_table", torch.int32, "[rows,stride]", align=4),
                  Operand("indices", torch.int32, f"[rows,{g.sparse_topk}]", "out", align=4),
                  Operand("lengths", torch.int32, "[rows]", "out", align=4)),
        scalars=(Scalar("rows"), Scalar("stride")), key=(pools, g.sparse_topk, g.index_kpool),
        geometry={"pools": pools, "width": g.sparse_topk, "dense_limit": g.index_topk + g.index_kpool - 1},
        doc=compile_glmf_index_expand_aot.__doc__,
    )


# ---------------------------------------------------------------------------
# MLA output: W_UV per head, then o_proj (FP8-capable)
# ---------------------------------------------------------------------------


class _GlmfOutput:
    def __init__(self, g: GLMFGeometry, fp8, max_rows: int = 64):
        n, v = g.heads, g.v_head_dim
        self.width = n * v
        fp8, prefill = _fp8_mode(fp8)
        self.uv = BatchedBf16Gemm(n=v, k=g.kv_lora_rank, batch=n, a_row=n * g.kv_lora_rank,
                                  a_batch=g.kv_lora_rank, o_row=n * v, o_batch=v,
                                  compute_warps=_batched_warps(max_rows))
        self.o = _Fp8Switch(g.hidden, n * v, fp8=fp8, prefill_rows=int(max_rows) if prefill else None)

    def key(self) -> tuple:
        return (self.uv.key(), self.o.key())

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(attn, w_uv, w_o, w_o, w_o, out, scratch, rows, Int32(0), stream)

    @cute.jit
    def body(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
             w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             stream: cuda.CUstream):
        values = _ptr(cutlass.BFloat16, Int64(scratch.toint()))
        qscratch = Int64(scratch.toint()) + _align_i64(Int64(rows) * Int64(self.width * 2))
        self.uv(attn, w_uv, values, rows, stream)
        self.o.run(values, w_o, w_o_fp8, w_o_scale, out, rows, fp8_rows, qscratch, stream)


class _GlmfOutputFp8(_GlmfOutput):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
                 w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(attn, w_uv, w_o, w_o_fp8, w_o_scale, out, scratch, rows, fp8_rows, stream)


class _GlmfOutputW8(_GlmfOutput):
    def __init__(self, g: GLMFGeometry, max_rows: int, prefill: bool):
        super().__init__(g, False, max_rows)
        self.o = _w8(g.hidden, self.width, int(max_rows) if prefill else None)

    def key(self) -> tuple:
        return ("w8",) + super().key()

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o_fp8: cute.Pointer, w_o_scale: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        values = _ptr(cutlass.BFloat16, Int64(scratch.toint()))
        qscratch = Int64(scratch.toint()) + _align_i64(Int64(rows) * Int64(self.width * 2))
        self.uv(attn, w_uv, values, rows, stream)
        self.o(values, w_o_fp8, w_o_scale, out, rows, fp8_rows, qscratch, stream)


def compile_glmf_o_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int, fp8=False,
                       fp8_only: str | None = None) -> AotProgram:
    """``glm_o`` (attn, w_uv, w_o, out, scratch; rows) with the FP8 o_proj switch when ``fp8``
    (``fp8="prefill"``: ``fp8_rows`` switches the block-FP8 GEMM). ``fp8_only`` takes o_proj
    as E4M3 + scales only (see ``compile_glmf_mla_producer_aot``)."""
    from ._glmf_fp8 import quant_scratch_bytes
    from .glm_attention import o_scratch_bytes

    max_rows = _check_rows(max_rows)
    w8_prefill = _w8_mode(fp8_only)
    if w8_prefill is not None:
        launch = _GlmfOutputW8(g, max_rows, w8_prefill)
        h, n, v = g.hidden, g.heads, g.v_head_dim
        operands = (
            Operand("attn", torch.bfloat16, f"[rows,{n},{g.kv_lora_rank}]"),
            Operand("w_uv", torch.bfloat16, f"[{n},{v},{g.kv_lora_rank}]"),
            *_w8_operands("w_o", h, n * v, w8_prefill),
            Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
            Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch"),
        )
        return compile_program(
            launch, name="glmf_o", operands=operands, scalars=_fp8_scalars(True, w8_prefill),
            key=(max_rows, "w8", fp8_only, launch.key()),
            geometry={"hidden": h, "heads": n, "v_head_dim": v, "max_rows": max_rows, "fp8_weights": "only"},
            scratch={"scratch": lambda rows: o_scratch_bytes(g, rows)
                     + (quant_scratch_bytes(n * v, rows) if w8_prefill else 0)},
            doc=compile_glmf_o_aot.__doc__,
        )
    fp8_on, prefill = _fp8_mode(fp8)
    launch = (_GlmfOutputFp8 if fp8_on else _GlmfOutput)(g, fp8, max_rows)
    h, n, v = g.hidden, g.heads, g.v_head_dim
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{n},{g.kv_lora_rank}]"),
        Operand("w_uv", torch.bfloat16, f"[{n},{v},{g.kv_lora_rank}]"),
        Operand("w_o", torch.bfloat16, f"[{h},{n * v}]"),
        *(fp8_ops("w_o", h, n * v, False, prefill) if fp8_on else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_o", operands=operands, scalars=_fp8_scalars(fp8_on, prefill),
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "heads": n, "v_head_dim": v, "max_rows": max_rows, "fp8_weights": fp8},
        scratch={"scratch": lambda rows: o_scratch_bytes(g, rows)
                 + (quant_scratch_bytes(n * v, rows) if prefill else 0)},
        doc=compile_glmf_o_aot.__doc__,
    )


# ---------------------------------------------------------------------------
# LM head for decode rows over an FP8 copy
# ---------------------------------------------------------------------------


class _HeadFp8:
    def __init__(self, vocab: int, hidden: int):
        from ._fp8_weights import MmaFp8Gemv

        warps, groups = FP8_GEMV_CONFIG.get((vocab, hidden), (4, 4))
        self.gemv = MmaFp8Gemv(vocab, hidden, max_rows=FP8_ROWS, warps=warps, groups=groups,
                               out_dtype=cutlass.Float32, row_scales=True)

    def key(self) -> tuple:
        return self.gemv.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, logits: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.gemv(x, w_fp8, scale, logits, rows, stream)


def compile_glmf_head_fp8_aot(g: GLMFGeometry = GLM53_FLASH, *, vocab: int = 154880) -> AotProgram:
    """FP32 logits of up to 16 decode rows over an E4M3 LM head with per-row
    x 128-K scales: x bf16 [rows,H] (the head norm's output), w_fp8 [V,H],
    scale f32 [V,H/128], logits f32 [rows,V]."""
    h = g.hidden
    launch = _HeadFp8(int(vocab), h)
    return compile_program(
        launch, name="glmf_head_fp8",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w_fp8", torch.float8_e4m3fn, f"[{vocab},{h}]"),
                  Operand("scale", torch.float32, f"[{vocab},{h // 128}]", align=4),
                  Operand("logits", torch.float32, f"[rows,{vocab}]", "out")),
        scalars=(Scalar("rows"),), key=(int(vocab), launch.key()),
        geometry={"hidden": h, "vocab": int(vocab), "max_rows": FP8_ROWS},
        doc=compile_glmf_head_fp8_aot.__doc__,
    )
