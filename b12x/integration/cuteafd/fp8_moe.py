"""Native AOT exact FP8 routed experts (checkpoint E4M3 + FP32 128x128 scales),
MXFP4 ones (``Fp8MoeGeometry.weights = "mxfp4"``: packed E2M1 ``w [E, N,
K/2]`` with UE8M0 ``s [E, N, K/32]``, MiMo V2.6 Pro; ``_mxfp4_moe_kernels``)
and NVIDIA ModelOpt NVFP4 ones (``weights = "nvfp4"``: packed E2M1 ``w [E, N,
K/2]`` with E4M3 ``s [E, N, K/16]`` followed by the experts' FP32
``weight_scale_2`` ``[E]``, run W4A16; ``_nvfp4_moe_kernels``).

``compile_fp8_moe_aot(geometry, route=..., max_rows=M, wire=True)``: one
program computing a (TP slice of a) sigmoid-routed MoE layer's routed
experts for ``rows <= M`` input rows, ``out[r] = sum_k w[r,k] *
down_e(act(gate_e(x_r), up_e(x_r)))`` with ``e = ids[r,k]``. The weights are
the checkpoint's E4M3 tensors times their FP32 block scales, widened to
``bf16(w * s)`` inside the GEMMs (see ``_fp8_moe_kernels``); nothing is
re-quantized. ``H`` hidden, ``I`` the slice of the intermediate this program
serves (``Fp8MoeGeometry.slice``), ``E`` experts, ``k`` routes per row::

    x          u8   [rows, H + H/32]  in   FP8 K32 wire rows (E4M3 then UE8M0),
                                           or bf16 [rows, H] when wire=False
    ids        i32  [rows, k]         in   expert ids (out of range: route skipped)
    weights    f32  [rows, k]         in   route weights (already normalized/scaled)
    w1         e4m3 [E, I, H]         in   gate_proj rows of the slice
    s1         f32  [E, I/128, H/128] in   their weight_scale_inv blocks
    w3         e4m3 [E, I, H]         in   up_proj
    s3         f32  [E, I/128, H/128] in
    w2         e4m3 [E, H, I]         in   down_proj columns of the slice
    s2         f32  [E, H/128, I/128] in
    out        bf16 [rows, H]         out  sum over the row's routes (FP32, one rounding)
    scratch    u8   fp8_moe_scratch_bytes(g, route, rows)
    rows       int32

A TP slice's ``out`` is the rank partial of the Spark response contract
(BF16 ``[rows, H]``, summed over ranks by the coordinator). ``activation =
bf16(bf16(silu(g)) * u)`` with the optional clamp ``g.swiglu_limit`` (0 = none).

Routes: ``decode`` groups the (row, route) pairs by expert without padding and
streams each active expert's weights through the grouped FP8 GEMV (64-row
chunks per expert; weight-bandwidth bound, the choice up to ~1024 rows);
``prefill`` pads each expert's group to 64 rows and runs the grouped TMA GEMM;
``stream`` (FP8 K32 wire input only) groups the pairs without padding into
128-row chunks and runs the expert-stationary streaming GEMMs of
``_fp8_moe_stream`` (each expert weight byte read once per layer; gate/up
straight from the wire rows with SwiGLU fused; both over widened ``bf16(w *
s)``, the same semantics as the other routes); ``auto`` branches on the live
row count (the large route above ``auto_large_rows``: ``stream`` for wire
input on GB10, ``prefill`` otherwise) and sizes its scratch for both.
Scratch (1024-aligned regions, live row layout): metadata, ``pair_row`` / ``pair_pos``, BF16 activations (reused
for the down projection's output), ``gate|up`` BF16, SwiGLU BF16; ``stream``:
metadata, ``pair_row`` / ``pair_pos``, SwiGLU BF16 ``[max_tiles * 128, I]``
(chunk-padded rows), down output BF16 ``[pairs, H]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import AotProgram, Operand, Scalar, compile_program
from ._fp8_moe_kernels import (
    GatherRows, GroupedFp8Gemm, GroupedFp8Gemv, MoeCombine, MoePrep, MoeSwiGLU, meta_words,
)
from ._fp8_moe_stream import STREAM_TILE_M, StreamFp8Down, StreamFp8GateUp, stream_max_tiles
from ._mxfp4_moe_kernels import GroupedMxfp4Gemv
from ._mxfp4_moe_stream import StreamMxfp4Down, StreamMxfp4GateUp
from ._nvfp4_moe_kernels import GroupedNvfp4Gemv, nvfp4_alpha_offset
from ._nvfp4_moe_stream import ChunkRows, ChunkSwiGLU, StreamNvfp4Linear
from ._nvfp4_moe_a4 import ChunkRowsA4, ChunkSwiGLUA4, StreamNvfp4LinearA4

__all__ = ["Fp8MoeGeometry", "GEOMETRIES", "compile_fp8_moe_aot", "fp8_moe_scratch_bytes"]

_ALIGN = 1024
TILE_M = 64
DECODE_MAX_ROWS = 64
# Live rows above which ``auto`` takes the grouped GEMM (RTX PRO 6000, MiMo
# TP4 slice, us: 1024 rows GEMV 1753 / GEMM 2059; 4096 rows 5728 / 5374).
AUTO_PREFILL_ROWS = 2048
# Live rows above which ``auto`` takes the streaming route for wire input
# (GB10, TP4 slices, us, GEMV / stream: MiMo 1024 rows 8467 / 8263, 2048 rows
# 14109 / 9185; GLM 1024 rows 12932 / 13124, 2048 rows 20433 / 14426).
AUTO_STREAM_ROWS = 1024
# MXFP4 (MiMo V2.6 Pro) live rows above which ``auto`` streams (wire input);
# RTX PRO 6000 (us, GEMV / stream): TP6 1024 rows 1662 / 1729, 2048 2598 / 1907;
# TP1 512 8965 / 7952, 1024 13654 / 10653.
AUTO_MXFP4_STREAM_ROWS = 1024
# GB10 (SM121), TP6 slice, us, GEMV / stream (block-scaled gate/up): 512 rows
# 7458 / 8537, 768 8771 / 8454, 1024 10057 / 8946.
AUTO_MXFP4_STREAM_ROWS_SM121 = 640
# The MXFP4 stream gate/up multiplies the E4M3 wire rows by the packed E2M1
# weights with block-scaled MMAs (UE8M0 per 32 on both operands: the same
# products, FP32 accumulation) instead of widening both to BF16. GB10 TP6
# (us, widen / block-scaled): 4096 rows 7175 / 5833, 1024 5012 / 4794; the
# layer outputs match the widening kernel's in all but ~1e-4 of BF16 values.
MXFP4_STREAM_QMMA = True
# Column groups (8 columns each) per CTA of the decode gate/up GEMV: one for
# single-row programs (GB10, us, interleaved x3, 4 groups -> 1: MiMo TP4
# 236 -> 230, GLM TP4 347 -> 341, MiMo TP2 450 -> 445; bitwise equal; the
# CTAs even out over the 48 SMs), four above (4 and 16 rows: within noise).
DECODE_GATE_UP_GROUPS = 4


# NVFP4 live rows above which ``auto`` takes the stream route (BF16 or wire
# input). RTX PRO 6000 (us, GEMV / stream, CUDA graphs): GLM 5.3 Flash TP1
# 1024 rows 6577 / 8236, 2048 10533 / 9789, 4096 20352 / 13830; Qwen 3.8 TP1
# 1024 2389 / 3064, 2048 3474 / 3582, 4096 6384 / 4610; GLM Flash TP4 wire
# 2048 2867 / 2759, 4096 5454 / 3854.
AUTO_NVFP4_STREAM_ROWS = 2048
# W4A4 (``activations="a4"``) live rows above which ``auto`` streams. RTX PRO
# 6000 (us, W4A16 GEMV / W4A4 stream, CUDA graphs): GLM 5.3 Flash TP1 256
# rows 2719 / 4453, 512 4726 / 4204, 1024 6616 / 4459, 4096 20544 / 6372;
# Qwen 3.8 TP1 512 2334 / 1430, 4096 6475 / 2323; GLM Flash TP4 wire 512
# 975 / 1218, 1024 1585 / 1302, 4096 5446 / 1994.
AUTO_NVFP4A4_STREAM_ROWS = 512
# GB10 (SM121, us, wire input, W4A16 GEMV / W4A16 stream / W4A4 stream): GLM 5.3
# Flash TP4 1024 rows 7303 / 9353 / 7713, 2048 11646 / 10800 / 8642, 4096
# 21077 / 14439 / 10752; GLM 5.3 TP6 1024 8906 / 11279 / 8244; Qwen 3.8 TP3
# 2048 5359 / 8179 / 5679, 4096 9817 / 10268 / 7193.
AUTO_NVFP4A4_STREAM_ROWS_SM121 = 1024


def _gate_up_groups(tile_rows: int) -> int:
    return 1 if int(tile_rows) == 1 else DECODE_GATE_UP_GROUPS


def _streams(wire: bool, weights: str = "fp8") -> bool:
    """``auto`` streams wire input on GB10 (SM121, weight-bandwidth bound);
    SM120 (RTX PRO 6000) keeps the grouped GEMM it was measured with. MXFP4
    weights have only the streaming large route (wire input), on both."""
    if weights == "mxfp4":
        return bool(wire)
    if weights in ("nvfp4", "nvfp4a4"):
        return True
    return bool(wire) and tuple(torch.cuda.get_device_capability()) == (12, 1)


def auto_large_rows(wire: bool, weights: str = "fp8") -> int:
    """Live-row threshold of ``auto``'s large route (``stream`` or ``prefill``)."""
    if weights == "nvfp4":
        return AUTO_NVFP4_STREAM_ROWS
    if weights == "nvfp4a4":
        gb10 = tuple(torch.cuda.get_device_capability()) == (12, 1)
        return AUTO_NVFP4A4_STREAM_ROWS_SM121 if gb10 else AUTO_NVFP4A4_STREAM_ROWS
    if weights == "mxfp4":
        gb10 = tuple(torch.cuda.get_device_capability()) == (12, 1)
        return AUTO_MXFP4_STREAM_ROWS_SM121 if gb10 else AUTO_MXFP4_STREAM_ROWS
    return AUTO_STREAM_ROWS if _streams(wire) else AUTO_PREFILL_ROWS


def _large_route(wire: bool, weights: str = "fp8") -> str:
    return "stream" if _streams(wire, weights) else "prefill"


@dataclass(frozen=True)
class Fp8MoeGeometry:
    """Routed-expert geometry of one program: the model's hidden, experts and
    top-k, the full intermediate and the ``tp`` degree whose slice it serves."""

    name: str
    hidden: int
    experts: int
    top_k: int
    intermediate: int
    tp: int = 1
    swiglu_limit: float = 0.0
    # "fp8": E4M3 + FP32 128x128 scales; "mxfp4": packed E2M1 + UE8M0 per 32;
    # "nvfp4": packed E2M1 + E4M3 per 16 + an FP32 alpha per expert.
    weights: str = "fp8"
    # NVFP4 only: "a16" (BF16 activations: W4A16, exact widening) or "a4" (the
    # stream route quantizes activations to NVFP4 with the checkpoint's static
    # input_scale and runs block-scaled FP4 MMAs: W4A4; the GEMV stays W4A16).
    activations: str = "a16"

    def __post_init__(self) -> None:
        if self.activations not in ("a16", "a4") or (self.activations == "a4" and self.weights != "nvfp4"):
            raise ValueError("activations is 'a16', or 'a4' with NVFP4 weights")
        if self.weights not in ("fp8", "mxfp4", "nvfp4"):
            raise ValueError("weights is 'fp8', 'mxfp4' or 'nvfp4'")
        if self.weights == "nvfp4" and (self.hidden % 128 or self.intermediate % 16 or self.intermediate // 16 < self.tp):
            raise ValueError("NVFP4 experts need a 128-aligned hidden and whole 16-blocks per rank")
        if self.weights == "fp8" and (self.hidden % 128 or self.intermediate % 128 or self.intermediate // 128 < self.tp):
            raise ValueError("FP8 experts need a 128-aligned hidden and intermediate with a 128-block per rank")
        if self.weights == "mxfp4" and (self.hidden % 128 or self.intermediate % 32 or self.intermediate // 32 < self.tp):
            raise ValueError("MXFP4 experts need a 128-aligned hidden and whole 32-blocks per rank")
        if not 1 <= self.top_k <= 16:
            raise ValueError("top_k must be 1..16")

    @property
    def slice(self) -> int:
        """Stored intermediate width of every rank. MXFP4 ranks own whole
        32-blocks as evenly as possible, zero-padded to one 128-aligned width
        (TP6 of 2048: 352/320 real rows in 384). FP8 ranks own whole 128x128
        scale blocks the same way (TP6 of 2048: 3, 3, 3, 3, 2, 2 blocks), each
        stored as the widest (384): zero gate/up rows give SiLU(0) * 0 = 0 and
        zero down columns add nothing, so the padding is exact."""
        if self.weights == "mxfp4":
            return -(-(-(-self.intermediate // 32) // self.tp) * 32 // 128) * 128
        if self.weights == "nvfp4":
            # Whole 16-value scale blocks per rank (TP6 of 2048: 352/336 rows in 384).
            return -(-(-(-self.intermediate // 16) // self.tp) * 16 // 128) * 128
        return -(-(self.intermediate // 128) // self.tp) * 128

    @property
    def kind(self) -> str:
        """The weight format, ``nvfp4a4`` for NVFP4 with the W4A4 stream route."""
        return "nvfp4a4" if self.activations == "a4" else self.weights

    def with_tp(self, tp: int) -> "Fp8MoeGeometry":
        return Fp8MoeGeometry(self.name, self.hidden, self.experts, self.top_k, self.intermediate, int(tp),
                              self.swiglu_limit, self.weights, self.activations)


# MiMo V2 Flash (no SwiGLU clamp in its config) and the GLM 5.3 MTP layer.
GEOMETRIES = {
    "mimo": Fp8MoeGeometry("mimo", hidden=4096, experts=256, top_k=8, intermediate=2048),
    "glm": Fp8MoeGeometry("glm", hidden=6144, experts=256, top_k=8, intermediate=2048),
    # GLM 5.3 Flash: 288 experts, SwiGLU clamped at 10 as its config says.
    "glmf": Fp8MoeGeometry("glmf", hidden=4096, experts=288, top_k=8, intermediate=2048, swiglu_limit=10.0),
    # Qwen 3.8 Flash Next: 512 experts, softmax top-10, unclamped SiLU.
    "qwen4": Fp8MoeGeometry("qwen4", hidden=2560, experts=512, top_k=10, intermediate=640),
    # MiMo V2.6 Pro: MXFP4 experts, 384 of them, top-8, unclamped SiLU.
    "mimop": Fp8MoeGeometry("mimop", hidden=6144, experts=384, top_k=8, intermediate=2048, weights="mxfp4"),
    # NVIDIA ModelOpt NVFP4 releases of GLM 5.3, GLM 5.3 Flash and Qwen 3.8 Flash Next (W4A16).
    "glm_nvfp4": Fp8MoeGeometry("glm_nvfp4", hidden=6144, experts=256, top_k=8, intermediate=2048, weights="nvfp4"),
    "glmf_nvfp4": Fp8MoeGeometry("glmf_nvfp4", hidden=4096, experts=288, top_k=8, intermediate=2048,
                                 swiglu_limit=10.0, weights="nvfp4"),
    "qwen4_nvfp4": Fp8MoeGeometry("qwen4_nvfp4", hidden=2560, experts=512, top_k=10, intermediate=640,
                                  weights="nvfp4"),
    # GLM 5.3 Flash's NVFP4 dense MLPs (layers 0-2 of nvidia/GLM-5.3-Flash-NVFP4)
    # as one always-selected expert: the coordinator runs them with ids 0, weight 1.
    "glmfdense_nvfp4": Fp8MoeGeometry("glmfdense_nvfp4", hidden=4096, experts=1, top_k=1, intermediate=12288,
                                      swiglu_limit=10.0, weights="nvfp4"),
    "glmfdense_nvfp4a4": Fp8MoeGeometry("glmfdense_nvfp4a4", hidden=4096, experts=1, top_k=1, intermediate=12288,
                                        swiglu_limit=10.0, weights="nvfp4", activations="a4"),
    # The same with W4A4 large-row (stream) steps.
    "glm_nvfp4a4": Fp8MoeGeometry("glm_nvfp4a4", hidden=6144, experts=256, top_k=8, intermediate=2048,
                                  weights="nvfp4", activations="a4"),
    "glmf_nvfp4a4": Fp8MoeGeometry("glmf_nvfp4a4", hidden=4096, experts=288, top_k=8, intermediate=2048,
                                   swiglu_limit=10.0, weights="nvfp4", activations="a4"),
    "qwen4_nvfp4a4": Fp8MoeGeometry("qwen4_nvfp4a4", hidden=2560, experts=512, top_k=10, intermediate=640,
                                    weights="nvfp4", activations="a4"),
}


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _grouped_rows(g: Fp8MoeGeometry, route: str, rows: int) -> int:
    pairs = max(int(rows), 1) * g.top_k
    if route == "decode":
        return pairs
    return TILE_M * (-(-pairs // TILE_M) + min(g.experts, pairs))


def _regions(g: Fp8MoeGeometry, route: str, rows: int) -> list[int]:
    rows = max(int(rows), 1)
    if route == "stream" and g.activations == "a4":
        pairs = rows * g.top_k
        padded = stream_max_tiles(g.experts, pairs) * STREAM_TILE_M
        return [
            meta_words(g.experts, stream_max_tiles(g.experts, pairs)) * 4,
            pairs * 4,                                 # pair_row
            pairs * 4,                                 # pair_pos
            padded * g.hidden // 2,                    # packed input rows
            padded * g.hidden // 16,                   # their E4M3 scales
            pairs * g.slice * 2,                       # gate
            pairs * g.slice * 2,                       # up
            padded * g.slice // 2,                     # packed SwiGLU rows
            padded * g.slice // 16,                    # their scales
            pairs * g.hidden * 2,                      # down output
        ]
    if route == "stream" and g.weights == "nvfp4":
        pairs = rows * g.top_k
        padded = stream_max_tiles(g.experts, pairs) * STREAM_TILE_M
        return [
            meta_words(g.experts, stream_max_tiles(g.experts, pairs)) * 4,
            pairs * 4,                                 # pair_row
            pairs * 4,                                 # pair_pos
            padded * g.hidden * 2,                     # chunk-padded input rows, then the down output
            pairs * g.slice * 2,                       # gate
            pairs * g.slice * 2,                       # up
            padded * g.slice * 2,                      # chunk-padded SwiGLU
        ]
    if route == "stream":
        pairs = rows * g.top_k
        return [
            meta_words(g.experts, stream_max_tiles(g.experts, pairs)) * 4,
            pairs * 4,                                 # pair_row
            pairs * 4,                                 # pair_pos
            stream_max_tiles(g.experts, pairs) * STREAM_TILE_M * g.slice * 2,  # SwiGLU, chunk-padded rows
            pairs * g.hidden * 2,                      # down output
        ]
    grouped = _grouped_rows(g, route, rows)
    act_rows = rows if route == "decode" else grouped
    return [
        meta_words(g.experts, grouped // TILE_M if route == "prefill" else 1) * 4,
        grouped * 4,                                   # pair_row
        rows * g.top_k * 4,                            # pair_pos
        max(act_rows, grouped) * g.hidden * 2,         # BF16 activations, then the down output
        grouped * 2 * g.slice * 2,                     # gate | up
        grouped * g.slice * 2,                         # SwiGLU
    ]


def _resolve(route: str, max_rows: int, wire: bool = True, weights: str = "fp8") -> str:
    """``auto`` compiles only the GEMV when the capacity never reaches the large
    route (always, for BF16-input MXFP4 packages: their only large route
    streams wire rows); MXFP4 and NVFP4 have no grouped-GEMM ``prefill``
    route (NVFP4 streams BF16 or wire input above ``AUTO_NVFP4_STREAM_ROWS``)."""
    if weights in ("nvfp4", "nvfp4a4") and route == "prefill":
        raise ValueError("NVFP4 experts have no prefill route (use stream)")
    if weights == "mxfp4":
        if route == "prefill":
            raise ValueError("MXFP4 experts have no prefill route (use stream)")
        if route == "auto" and not wire:
            return "decode"
    return "decode" if route == "auto" and int(max_rows) <= auto_large_rows(wire, weights) else route


def fp8_moe_scratch_bytes(g: Fp8MoeGeometry, route: str, rows: int, wire: bool = True) -> int:
    route = _resolve(route, rows, wire, g.kind)
    if route == "auto":
        return max(fp8_moe_scratch_bytes(g, "decode", rows, wire),
                   fp8_moe_scratch_bytes(g, _large_route(wire, g.kind), rows, wire))
    return sum(_align(b) for b in _regions(g, route, rows))


@cute.jit
def _al(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class _Route:
    """One route's kernels (``decode``: grouped GEMV; ``prefill``: grouped TMA GEMM)."""

    def __init__(self, g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool):
        self.g, self.route = g, route
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        decode = route == "decode"
        self.max_tiles = 1 if decode else _grouped_rows(g, route, max_rows) // TILE_M
        self.prep = MoePrep(experts=e, top_k=k, pad=1 if decode else TILE_M, max_tiles=self.max_tiles)
        self.rows_in = GatherRows(k=h, wire=wire, gather=not decode)
        if decode:
            tile_rows = min(int(max_rows), DECODE_MAX_ROWS)
            warps_h = 8 if h >= 6144 else 4
            gemv = {"mxfp4": GroupedMxfp4Gemv, "nvfp4": GroupedNvfp4Gemv}.get(g.weights, GroupedFp8Gemv)
            self.gate_up = gemv(n=2 * i, k=h, experts=e, split=i, max_rows=tile_rows, warps=warps_h,
                                groups=_gate_up_groups(tile_rows), gather=True)
            # K split over warps in whole 128 blocks: 4 where they divide evenly,
            # else the largest divisor <= 8 of the blocks (Qwen's 640: 5 warps).
            warps_i = 4 if i % 512 == 0 else max(d for d in range(1, 9) if (i // 128) % d == 0)
            self.down = gemv(n=h, k=i, experts=e, max_rows=tile_rows, warps=warps_i)
        else:
            self.gate_up = GroupedFp8Gemm(n=i, k=h, experts=e, halves=2)
            self.down = GroupedFp8Gemm(n=h, k=i, experts=e, halves=1)
        self.swiglu = MoeSwiGLU(inter=i, limit=g.swiglu_limit)
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return (self.route, self.max_tiles, self.rows_in.key(), self.gate_up.key(), self.down.key(),
                self.swiglu.key(), self.combine.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        pairs = rows * Int32(g.top_k)
        if cutlass.const_expr(self.route == "decode"):
            grouped = pairs
            act_rows = rows
            max_tiles = Int32(1)
        else:
            groups = pairs
            if groups > Int32(g.experts):
                groups = Int32(g.experts)
            max_tiles = (pairs + Int32(TILE_M - 1)) // Int32(TILE_M) + groups
            grouped = max_tiles * Int32(TILE_M)
            act_rows = grouped
        base = Int64(scratch.toint())
        meta_at = base
        pair_row_at = meta_at + _al(Int64(4 * meta_words(g.experts, self.max_tiles)))
        pair_pos_at = pair_row_at + _al(Int64(grouped) * Int64(4))
        xb_at = pair_pos_at + _al(Int64(pairs) * Int64(4))
        big = Int64(act_rows)
        if Int64(grouped) > big:
            big = Int64(grouped)
        gu_at = xb_at + _al(big * Int64(g.hidden * 2))
        act_at = gu_at + _al(Int64(grouped) * Int64(4 * g.slice))
        ptr = lambda dtype, at: cute.make_ptr(dtype, at, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        meta = ptr(cutlass.Int32, meta_at)
        pair_row = ptr(cutlass.Int32, pair_row_at)
        pair_pos = ptr(cutlass.Int32, pair_pos_at)
        xb = ptr(cutlass.BFloat16, xb_at)
        gu = ptr(cutlass.BFloat16, gu_at)
        act = ptr(cutlass.BFloat16, act_at)
        ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.prep(ids_i, meta, pair_row, pair_pos, rows, grouped, stream)
        self.rows_in(x, pair_row, meta, xb, act_rows, stream)
        if cutlass.const_expr(self.route == "decode"):
            max_groups = pairs
            if max_groups > Int32(g.experts):
                max_groups = Int32(g.experts)
            self.gate_up(xb, pair_row, meta, w1, s1, w3, s3, gu, max_groups, stream)
            self.swiglu(gu, meta, act, grouped, stream)
            self.down(act, pair_row, meta, w2, s2, w2, s2, xb, max_groups, stream)
        else:
            self.gate_up(xb, meta, w1, s1, w3, s3, gu, max_tiles, stream)
            self.swiglu(gu, meta, act, grouped, stream)
            self.down(act, meta, w2, s2, w2, s2, xb, max_tiles, stream)
        self.combine(xb, pair_pos, weights, out, rows, stream)


class _StreamRoute:
    """The ``stream`` route: compact expert groups in 128-row chunks, streaming
    gate/up (FP8 wire rows, SwiGLU fused) and down GEMMs, route combine."""

    def __init__(self, g: Fp8MoeGeometry, max_rows: int, wire: bool):
        if not wire:
            raise ValueError("the stream route takes FP8 K32 wire rows")
        self.g = g
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        self.max_tiles = stream_max_tiles(e, int(max_rows) * k)
        self.prep = MoePrep(experts=e, top_k=k, pad=1, max_tiles=self.max_tiles, tile_rows=STREAM_TILE_M,
                            chunked=True)
        if g.weights == "mxfp4":
            self.gate_up = StreamMxfp4GateUp(inter=i, hidden=h, experts=e, limit=g.swiglu_limit,
                                             qmma=MXFP4_STREAM_QMMA)
            self.down = StreamMxfp4Down(hidden=h, inter=i, experts=e)
        else:
            self.gate_up = StreamFp8GateUp(inter=i, hidden=h, experts=e, limit=g.swiglu_limit)
            self.down = StreamFp8Down(hidden=h, inter=i, experts=e)
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return ("stream", self.max_tiles, self.gate_up.key(), self.down.key(), self.combine.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        pairs = rows * Int32(g.top_k)
        groups = pairs
        if groups > Int32(g.experts):
            groups = Int32(g.experts)
        max_tiles = (pairs + Int32(STREAM_TILE_M - 1)) // Int32(STREAM_TILE_M) + groups
        base = Int64(scratch.toint())
        pair_row_at = base + _al(Int64(4 * meta_words(g.experts, self.max_tiles)))
        pair_pos_at = pair_row_at + _al(Int64(pairs) * Int64(4))
        act_at = pair_pos_at + _al(Int64(pairs) * Int64(4))
        y_at = act_at + _al(Int64(max_tiles) * Int64(STREAM_TILE_M * 2 * g.slice))
        ptr = lambda dtype, at: cute.make_ptr(dtype, at, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        meta = ptr(cutlass.Int32, base)
        pair_row = ptr(cutlass.Int32, pair_row_at)
        pair_pos = ptr(cutlass.Int32, pair_pos_at)
        act = ptr(cutlass.BFloat16, act_at)
        y = ptr(cutlass.BFloat16, y_at)
        ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.prep(ids_i, meta, pair_row, pair_pos, rows, pairs, stream)
        self.gate_up(x, pair_row, meta, w1, s1, w3, s3, act, max_tiles, stream)
        self.down(act, meta, w2, s2, y, max_tiles, stream)
        self.combine(y, pair_pos, weights, out, rows, stream)


class _StreamNvfp4Route:
    """The NVFP4 ``stream`` route: compact expert groups in 128-row chunks,
    chunk-padded BF16 input rows, gate and up streaming GEMMs, SwiGLU, the
    down streaming GEMM, route combine (``_nvfp4_moe_stream``)."""

    def __init__(self, g: Fp8MoeGeometry, max_rows: int, wire: bool):
        self.g = g
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        self.max_tiles = stream_max_tiles(e, int(max_rows) * k)
        self.prep = MoePrep(experts=e, top_k=k, pad=1, max_tiles=self.max_tiles, tile_rows=STREAM_TILE_M,
                            chunked=True)
        self.rows_in = ChunkRows(k=h, wire=wire, experts=e)
        self.gate_up = StreamNvfp4Linear(n=i, k=h, experts=e)
        self.swiglu = ChunkSwiGLU(inter=i, experts=e, limit=g.swiglu_limit)
        self.down = StreamNvfp4Linear(n=h, k=i, experts=e)
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return ("stream_nvfp4", self.max_tiles, self.rows_in.key(), self.gate_up.key(), self.swiglu.key(),
                self.down.key(), self.combine.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        pairs = rows * Int32(g.top_k)
        groups = pairs
        if groups > Int32(g.experts):
            groups = Int32(g.experts)
        max_tiles = (pairs + Int32(STREAM_TILE_M - 1)) // Int32(STREAM_TILE_M) + groups
        padded = Int64(max_tiles) * Int64(STREAM_TILE_M)
        base = Int64(scratch.toint())
        pair_row_at = base + _al(Int64(4 * meta_words(g.experts, self.max_tiles)))
        pair_pos_at = pair_row_at + _al(Int64(pairs) * Int64(4))
        xb_at = pair_pos_at + _al(Int64(pairs) * Int64(4))
        gate_at = xb_at + _al(padded * Int64(g.hidden * 2))
        up_at = gate_at + _al(Int64(pairs) * Int64(g.slice * 2))
        act_at = up_at + _al(Int64(pairs) * Int64(g.slice * 2))
        ptr = lambda dtype, at: cute.make_ptr(dtype, at, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        meta = ptr(cutlass.Int32, base)
        pair_row = ptr(cutlass.Int32, pair_row_at)
        pair_pos = ptr(cutlass.Int32, pair_pos_at)
        xb = ptr(cutlass.BFloat16, xb_at)
        gate = ptr(cutlass.BFloat16, gate_at)
        up = ptr(cutlass.BFloat16, up_at)
        act = ptr(cutlass.BFloat16, act_at)
        ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.prep(ids_i, meta, pair_row, pair_pos, rows, pairs, stream)
        self.rows_in(x, pair_row, meta, xb, max_tiles, stream)
        self.gate_up(xb, meta, w1, s1, gate, max_tiles, stream)
        self.gate_up(xb, meta, w3, s3, up, max_tiles, stream)
        self.swiglu(gate, up, meta, act, max_tiles, stream)
        # The down output reuses the input rows' region (pairs <= chunk-padded rows).
        self.down(act, meta, w2, s2, xb, max_tiles, stream)
        self.combine(xb, pair_pos, weights, out, rows, stream)


class _StreamNvfp4A4Route:
    """The W4A4 NVFP4 ``stream`` route (``_nvfp4_moe_a4``): chunk-padded NVFP4
    input rows, gate and up block-scaled FP4 GEMMs, SwiGLU quantized to NVFP4,
    the down GEMM, route combine."""

    def __init__(self, g: Fp8MoeGeometry, max_rows: int, wire: bool):
        self.g = g
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        self.max_tiles = stream_max_tiles(e, int(max_rows) * k)
        self.prep = MoePrep(experts=e, top_k=k, pad=1, max_tiles=self.max_tiles, tile_rows=STREAM_TILE_M,
                            chunked=True)
        self.rows_in = ChunkRowsA4(k=h, wire=wire, experts=e, scale_rows=i)
        self.gate_up = StreamNvfp4LinearA4(n=i, k=h, experts=e)
        self.swiglu = ChunkSwiGLUA4(inter=i, hidden=h, experts=e, limit=g.swiglu_limit)
        self.down = StreamNvfp4LinearA4(n=h, k=i, experts=e)
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return ("stream_nvfp4_a4", self.max_tiles, self.rows_in.key(), self.gate_up.key(), self.swiglu.key(),
                self.down.key(), self.combine.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        pairs = rows * Int32(g.top_k)
        groups = pairs
        if groups > Int32(g.experts):
            groups = Int32(g.experts)
        max_tiles = (pairs + Int32(STREAM_TILE_M - 1)) // Int32(STREAM_TILE_M) + groups
        padded = Int64(max_tiles) * Int64(STREAM_TILE_M)
        base = Int64(scratch.toint())
        pair_row_at = base + _al(Int64(4 * meta_words(g.experts, self.max_tiles)))
        pair_pos_at = pair_row_at + _al(Int64(pairs) * Int64(4))
        xq_at = pair_pos_at + _al(Int64(pairs) * Int64(4))
        xs_at = xq_at + _al(padded * Int64(g.hidden // 2))
        gate_at = xs_at + _al(padded * Int64(g.hidden // 16))
        up_at = gate_at + _al(Int64(pairs) * Int64(g.slice * 2))
        aq_at = up_at + _al(Int64(pairs) * Int64(g.slice * 2))
        as_at = aq_at + _al(padded * Int64(g.slice // 2))
        y_at = as_at + _al(padded * Int64(g.slice // 16))
        ptr = lambda dtype, at: cute.make_ptr(dtype, at, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        meta = ptr(cutlass.Int32, base)
        pair_row = ptr(cutlass.Int32, pair_row_at)
        pair_pos = ptr(cutlass.Int32, pair_pos_at)
        xq, xs = ptr(cutlass.Uint8, xq_at), ptr(cutlass.Uint8, xs_at)
        gate, up = ptr(cutlass.BFloat16, gate_at), ptr(cutlass.BFloat16, up_at)
        aq, a_s = ptr(cutlass.Uint8, aq_at), ptr(cutlass.Uint8, as_at)
        y = ptr(cutlass.BFloat16, y_at)
        ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.prep(ids_i, meta, pair_row, pair_pos, rows, pairs, stream)
        self.rows_in(x, pair_row, meta, s1, xq, xs, max_tiles, stream)
        self.gate_up(xq, xs, meta, w1, s1, gate, max_tiles, stream)
        self.gate_up(xq, xs, meta, w3, s3, up, max_tiles, stream)
        self.swiglu(gate, up, meta, s2, aq, a_s, max_tiles, stream)
        self.down(aq, a_s, meta, w2, s2, y, max_tiles, stream)
        self.combine(y, pair_pos, weights, out, rows, stream)


def _make_route(g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool):
    if route == "stream" and g.activations == "a4":
        return _StreamNvfp4A4Route(g, max_rows, wire)
    if route == "stream" and g.weights == "nvfp4":
        return _StreamNvfp4Route(g, max_rows, wire)
    return _StreamRoute(g, max_rows, wire) if route == "stream" else _Route(g, route, max_rows, wire)


class _Fp8Moe:
    def __init__(self, g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool):
        self.route = route
        self.threshold = auto_large_rows(wire, g.kind)
        self.decode = _Route(g, "decode", max_rows, wire) if route in ("decode", "auto") else None
        large = _large_route(wire, g.kind) if route == "auto" else route
        self.prefill = _make_route(g, large, max_rows, wire) if route in ("prefill", "stream", "auto") else None

    def key(self) -> tuple:
        return (self.threshold,) + tuple(part.key() if part is not None else None
                                         for part in (self.decode, self.prefill))

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.route == "decode"):
            self.decode(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
        elif cutlass.const_expr(self.route != "auto"):
            self.prefill(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
        else:
            if rows <= Int32(self.threshold):
                self.decode(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
            else:
                self.prefill(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)


def compile_fp8_moe_aot(g: Fp8MoeGeometry, *, route: str, max_rows: int, wire: bool = True) -> AotProgram:
    """Routed FP8 experts of ``g`` for ``rows <= max_rows``; see the module docstring."""
    if route not in ("decode", "prefill", "stream", "auto"):
        raise ValueError("route is 'decode', 'prefill', 'stream' or 'auto'")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    requested, route = route, _resolve(route, max_rows, wire, g.kind)
    launch = _Fp8Moe(g, route, max_rows, wire)
    h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
    x = (Operand("x", torch.uint8, f"[rows,{h + h // 32}]", note="FP8 K32 wire rows") if wire
         else Operand("x", torch.bfloat16, f"[rows,{h}]"))
    if g.weights == "nvfp4":
        gate_up_scales = nvfp4_alpha_offset(e, i, h) + 8 * e
        down_scales = nvfp4_alpha_offset(e, h, i) + 8 * e
        weights = (
            Operand("w1", torch.uint8, f"[{e},{i},{h // 2}]", note="packed E2M1 (even element low)"),
            Operand("s1", torch.uint8, f"[{gate_up_scales}]", align=4,
                    note=f"E4M3 [{e},{i},{h // 16}] then FP32 alpha [{e}] and input_scale [{e}]"),
            Operand("w3", torch.uint8, f"[{e},{i},{h // 2}]"),
            Operand("s3", torch.uint8, f"[{gate_up_scales}]", align=4),
            Operand("w2", torch.uint8, f"[{e},{h},{i // 2}]"),
            Operand("s2", torch.uint8, f"[{down_scales}]", align=4,
                    note=f"E4M3 [{e},{h},{i // 16}] then FP32 alpha [{e}] and input_scale [{e}]"),
        )
    elif g.weights == "mxfp4":
        weights = (
            Operand("w1", torch.uint8, f"[{e},{i},{h // 2}]", note="packed E2M1 (even element low)"),
            Operand("s1", torch.uint8, f"[{e},{i},{h // 32}]", align=4, note="UE8M0 per 32"),
            Operand("w3", torch.uint8, f"[{e},{i},{h // 2}]"),
            Operand("s3", torch.uint8, f"[{e},{i},{h // 32}]", align=4),
            Operand("w2", torch.uint8, f"[{e},{h},{i // 2}]"),
            Operand("s2", torch.uint8, f"[{e},{h},{i // 32}]", align=4),
        )
    else:
        weights = (
            Operand("w1", torch.float8_e4m3fn, f"[{e},{i},{h}]"),
            Operand("s1", torch.float32, f"[{e},{i // 128},{h // 128}]", align=4),
            Operand("w3", torch.float8_e4m3fn, f"[{e},{i},{h}]"),
            Operand("s3", torch.float32, f"[{e},{i // 128},{h // 128}]", align=4),
            Operand("w2", torch.float8_e4m3fn, f"[{e},{h},{i}]"),
            Operand("s2", torch.float32, f"[{e},{h // 128},{i // 128}]", align=4),
        )
    operands = (
        x,
        Operand("ids", torch.int32, f"[rows,{k}]", align=4),
        Operand("weights", torch.float32, f"[rows,{k}]", align=4),
        *weights,
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[fp8_moe_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"fp8_moe_{g.name}_tp{g.tp}_{route}", operands=operands, scalars=(Scalar("rows"),),
        key=launch.key(),
        geometry={"requested_route": requested, "hidden": h, "experts": e, "top_k": k, "intermediate": g.intermediate, "tp": g.tp,
                  "slice": i, "swiglu_limit": g.swiglu_limit, "route": route, "max_rows": int(max_rows),
                  "wire": bool(wire), "weights": g.weights, "activations": g.activations},
        scratch={"scratch": lambda rows: fp8_moe_scratch_bytes(g, route, rows, wire)},
        doc=__doc__,
    )
