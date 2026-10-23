"""Native AOT exact FP8 routed experts (checkpoint E4M3 + FP32 128x128 scales),
MXFP4 ones (``Fp8MoeGeometry.weights = "mxfp4"``: packed E2M1 ``w [E, N,
K/2]`` with UE8M0 ``s [E, N, K/32]``, MiMo V2.6 Pro; ``_mxfp4_moe_kernels``)
and NVIDIA ModelOpt NVFP4 ones (``weights = "nvfp4"``: packed E2M1 ``w [E, N,
K/2]`` with E4M3 ``s [E, N, K/16]`` followed by the experts' FP32
``weight_scale_2`` and calibrated ``input_scale`` ``[E]``. They run W4A4
from original BF16 inputs at every row count (``_nvfp4_moe_a4``).

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
MiniMax's ``swigluoai_uninterleave`` uses gate ``w1``, up ``w3`` and
``bf16(g * sigmoid(1.702 * g) * (u + 1))`` after the same clamp, with one
BF16 rounding of the activation product.

Routes: ``decode`` groups the (row, route) pairs by expert without padding and
streams each active expert's weights through the grouped FP8 GEMV (64-row
chunks per expert; weight-bandwidth bound, the choice up to ~1024 rows);
``prefill`` pads each expert's group to 64 rows and runs the grouped TMA GEMM;
``stream`` groups the pairs without padding into 128-row chunks and runs the
expert-stationary streaming GEMMs of ``_fp8_moe_stream`` (each expert weight
byte read once per layer; gate/up straight from the FP8 K32 wire rows with
SwiGLU fused; BF16 input rows are first quantized to wire rows in scratch).
For FP8 weights its gate/up is W8A8: block-scaled E4M3 x E4M3 MMAs with the
FP32 weight block scales promoted per 128-K step, i.e. ``x . (w * s)`` with
the exact checkpoint weights; the down projection keeps BF16 activations over
widened ``bf16(w * s)``. ``prefill`` picks the large-row form of FP8
programs: ``"auto"`` (default) runs wire input W8A8 and BF16 input as before
(W8A16, its rows exact); ``"w8a8"`` also quantizes BF16 rows to wire rows
and streams them W8A8; ``"w8a16"`` is the former programs exactly (gate/up
over widened ``bf16(w * s)``). ``auto`` branches on the live row count (the
large route above ``auto_large_rows``: the W8A8 ``stream``; else the W8A16
``stream`` for wire input on GB10 and ``prefill`` otherwise) and sizes its
scratch for both.
Scratch (1024-aligned regions, live row layout): metadata, ``pair_row`` / ``pair_pos``, BF16 activations (reused
for the down projection's output), ``gate|up`` BF16, SwiGLU BF16; ``stream``:
metadata, ``pair_row`` / ``pair_pos``, SwiGLU BF16 ``[max_tiles * 128, I]``
(chunk-padded rows), down output BF16 ``[pairs, H]``, (BF16 input) wire rows.
MXFP4 wire-input stream programs run the down projection A8 by default
(``MXFP4_STREAM_DOWN_A8``): SwiGLU is rounded to the same BF16 values, then
each row's K32 blocks are quantized to E4M3 with UE8M0 scales and the down
GEMM uses direct block-scaled E4M3 x E2M1 MMAs. ``prefill="w8a16"`` keeps the
former BF16 down projection (``mxfp4_down_a8=False`` forces it too); decode
is unchanged. Native BF16 MXFP4 programs read original BF16 activations and
retain BF16 SwiGLU/down math at every row count; only the kernel schedule
changes above the runtime row threshold. Scratch covers both routes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int32, Int64

from b12x._lib.quant.mxfp8_rows import _THREADS as _QUANT_THREADS
from b12x._lib.quant.mxfp8_rows import _WARP_SUBGROUP_WIDTH, _MXFP8RowsQuantLaunch
from b12x._lib.utils import runtime_num_sm

from ._common import AotProgram, Operand, Scalar, compile_program
from ._fp8_moe_kernels import (
    GatherRows, GroupedFp8Gemm, GroupedFp8Gemv, MoeCombine, MoePrep, MoeSwiGLU, meta_words,
)
from ._fp8_moe_stream import STREAM_TILE_M, StreamFp8Down, StreamFp8GateUp, stream_max_tiles
from ._mxfp4_moe_kernels import GroupedMxfp4Gemv
from ._mxfp4_moe_stream import StreamMxfp4Down, StreamMxfp4GateUp
from ._mxfp4_moe_a8 import StreamMxfp4DownA8
from ._mxfp4_down_a8_plan import mxfp8_down_row_bytes, mxfp4_scale_row_bytes
from ._nvfp4_moe_kernels import nvfp4_alpha_offset
from ._nvfp4_moe_a4 import ChunkRowsA4, ChunkSwiGLUA4, StreamNvfp4GateUpA4, StreamNvfp4LinearA4

__all__ = ["Fp8MoeGeometry", "GEOMETRIES", "PREFILL_FORMS", "compile_fp8_moe_aot", "fp8_moe_scratch_bytes",
           "prefill_forms"]

_ALIGN = 1024
TILE_M = 64
DECODE_MAX_ROWS = 64
# Sparse gate/up cp.async ring depth. The sparse tile's SMEM (16 x 128 B of
# activations + tile_n x 128 B of weights per K256 stage) is far below the
# 101,376-byte CTA ceiling, so the sparse class stages deeper than the wide one;
# the extra stages are what keep more than one operand batch in flight at decode
# occupancy.
SMALL_GATE_STAGES = 3
# CTAs per queried SM the sparsest valid routing (top_k experts) must reach for
# the sparse gate/up N tile to be used; narrower tiles add CTAs, never K order.
SMALL_GATE_CTAS_PER_SM = 4
# Live rows above which ``auto`` takes the grouped GEMM (RTX PRO 6000, MiMo
# TP4 slice, us: 1024 rows GEMV 1753 / GEMM 2059; 4096 rows 5728 / 5374).
# W8A16 programs (``prefill="w8a16"``, and BF16 input under "auto") keep
# these routes and thresholds exactly.
AUTO_PREFILL_ROWS = 2048
# Live rows above which ``auto`` takes the W8A16 streaming route for wire
# input (GB10, TP4 slices, us, GEMV / stream: MiMo 1024 rows 8467 / 8263, 2048
# rows 14109 / 9185; GLM 1024 rows 12932 / 13124, 2048 rows 20433 / 14426).
AUTO_STREAM_ROWS = 1024
# W8A8 stream (the default for FP8 weights): live rows above which ``auto``
# streams. GB10 wire input, us, GEMV / stream: MiMo TP4 768 rows 8097 / 8198,
# 1024 8591 / 8439; MiMo TP2 768 15870 / 14971; GLM TP4 768 12597 / 12236,
# 1024 13032 / 12577; GLM Flash TP4 768 9016 / 9049, 1024 9353 / 9293.
AUTO_W8A8_STREAM_ROWS_SM121 = 768
# RTX PRO 6000 (SM120): wire input, and BF16 input with ``prefill="w8a8"``
# (the program quantizes its rows to the wire format).
# MiMo TP1, BF16 input, us, GEMV / W8A8 stream / grouped GEMM: 512 rows 4060 /
# 4101 / 7300; 1024 5724 / 4178 / 7492; 2048 10528 / 4438 / 10246; 4096 21160 /
# 5916 / 16882.
AUTO_W8A8_STREAM_ROWS_SM120 = 512
# MXFP4 live rows above which ``auto`` streams on SM120 (``rows <= threshold``
# takes GEMV; ``rows > threshold`` streams). Coordinator packages ship ``tp1``
# and ``tp2`` (``tp6`` is a Spark/SM121 layout governed by
# ``AUTO_MXFP4_STREAM_ROWS_SM121 = 640`` and is shown below for context only).
# Native BF16 coordinator input (--input bf16, --graph, --check, 2x20
# interleaved decode/stream replays, RTX PRO 6000 188 SMs, 450 W cap, zero
# throttle; pass-mean GEMV / stream us):
#   coarse      256          384          512          640          768          1024         1536          2048
#   mimop tp1   5317 / 7552  5568 / 7677  5764 / 7629  6202 / 7733  6849 / 7824  8083 / 8021  10202 / 8683  12425 / 9689
#   mimop tp2   2563 / 3814  2688 / 3853  2859 / 3896  3098 / 3915  3375 / 3939  3956 / 3994   4990 / 4215   6168 / 4633
#   mimof tp1   2174 / 3384  2324 / 3421  2602 / 3459  2953 / 3500  3207 / 3520  3928 / 3667   5463 / 4072   7298 / 4927
#   mimof tp2   1123 / 1751  1197 / 1768  1336 / 1786  1486 / 1804  1592 / 1820  1894 / 1879   2621 / 2014   3546 / 2404
#   (context)
#   mimop tp6   1160 / 1542  1201 / 1566  1249 / 1582  1324 / 1598  1428 / 1616  1655 / 1646   2107 / 1725   2606 / 1827
#   mimof tp6    528 /  716   554 /  725   611 /  736   677 /  748   724 /  760   839 /  789   1099 /  852   1492 / 1006
# Fine boundary screen across the four coordinator layouts (GEMV / stream us,
# with stream/decode at row in parentheses):
#   fine        832                  896                  928                  960                  992                  1024
#   mimop tp1   7265 / 8002 (1.101)  7581 / 8011 (1.057)  7644 / 7921 (1.036)  7761 / 7934 (1.022)  7950 / 7984 (1.004)  8110 / 8046 (0.992)
#   mimop tp2   3576 / 3950 (1.104)  3742 / 3984 (1.065)  3798 / 3995 (1.052)  3897 / 4010 (1.029)  3946 / 4032 (1.022)  3968 / 3994 (1.007)
#   mimof tp1   3385 / 3556 (1.050)  3598 / 3638 (1.011)  3696 / 3667 (0.992)  3812 / 3701 (0.971)  3923 / 3744 (0.954)  3928 / 3666 (0.933)
#   mimof tp2   1653 / 1828 (1.106)  1730 / 1841 (1.064)  1752 / 1851 (1.057)  1813 / 1859 (1.026)  1850 / 1868 (1.009)  1891 / 1879 (0.994)
# Routing geomean across the 4 coordinator layouts x 6 fine-band rows
# (routed / decode: decode for rows <= threshold, stream for rows > threshold):
#   threshold   896     928     960     992     1000    1024
#   geomean     1.0038  0.9983  0.9963  0.9968  0.9968  1.0000
# At 928..992 three of the four coordinator layouts still favour GEMV (mimop
# tp1 1.036..1.004, mimop tp2 1.052..1.022, mimof tp2 1.057..1.009), so
# streaming below 1000 taxes prefill tail chunks on those layouts. Setting
# 1000 keeps rows <= 1000 on GEMV and routes full 1024-row prefill chunks to
# stream (stream/decode at 1024: mimop tp1 0.992, mimop tp2 1.007, mimof tp1
# 0.933, mimof tp2 0.994; 4-layout geomean 0.981).
AUTO_MXFP4_STREAM_ROWS = 1000
# GB10 (SM121), TP6 slice, us, GEMV / stream (block-scaled gate/up): 512 rows
# 7458 / 8537, 768 8771 / 8454, 1024 10057 / 8946.
AUTO_MXFP4_STREAM_ROWS_SM121 = 640
# The MXFP4 stream gate/up multiplies the E4M3 wire rows by the packed E2M1
# weights with block-scaled MMAs (UE8M0 per 32 on both operands: the same
# products, FP32 accumulation) instead of widening both to BF16. GB10 TP6
# (us, widen / block-scaled): 4096 rows 7175 / 5833, 1024 5012 / 4794; the
# layer outputs match the widening kernel's in all but ~1e-4 of BF16 values.
MXFP4_STREAM_QMMA = True
# The MXFP4 stream down projection over MXFP8 rows (the SwiGLU output's K32
# blocks as E4M3 with UE8M0 scales, block-scaled MMAs) instead of widening the
# E2M1 weights for BF16 MMAs over BF16 rows. GB10 TP6, 4096 rows, per layer
# (us, BF16 / A8 down): 13379 / 12531; MiMo V2.6 Pro golden prefill NLL
# 2.4190 -> 2.4119, KL to the reference 0.0446 -> 0.0457.
MXFP4_STREAM_DOWN_A8 = True
# The FP8 stream gate/up of W8A8 programs: block-scaled E4M3 x E4M3 MMAs
# (False: the W8A16 widening kernel everywhere, as ``prefill="w8a16"``).
FP8_STREAM_QMMA = True
# Column groups (8 columns each) per CTA of the decode gate/up GEMV: one for
# single-row programs (GB10, us, interleaved x3, 4 groups -> 1: MiMo TP4
# 236 -> 230, GLM TP4 347 -> 341, MiMo TP2 450 -> 445; bitwise equal; the
# CTAs even out over the 48 SMs), four above (4 and 16 rows: within noise).
DECODE_GATE_UP_GROUPS = 4
# CTAs of the in-program BF16 -> wire row quantizer (grid-stride over rows).
_QUANT_MAX_CTAS = 512


def _gate_up_groups(tile_rows: int) -> int:
    return 1 if int(tile_rows) == 1 else DECODE_GATE_UP_GROUPS


def _sm121() -> bool:
    return tuple(torch.cuda.get_device_capability()) == (12, 1)


# Large-row forms of FP8 programs (``compile_fp8_moe_aot(prefill=...)``).
PREFILL_FORMS = ("auto", "w8a8", "w8a16")


def _w8a8(wire: bool, weights: str, prefill: str) -> bool:
    """FP8 programs whose large route is the W8A8 stream: wire input unless
    ``prefill="w8a16"``; BF16 input only with ``prefill="w8a8"`` (its rows are
    then quantized to wire rows: measurably less exact than BF16 rows)."""
    if prefill not in PREFILL_FORMS:
        raise ValueError(f"prefill is one of {PREFILL_FORMS}")
    if weights != "fp8" or prefill == "w8a16":
        return False
    return bool(wire) or prefill == "w8a8"


def _mxfp4_down_a8(wire: bool, weights: str, prefill: str, explicit: "bool | None" = None) -> bool:
    """Whether an MXFP4 wire-input stream runs its down projection A8: the
    default unless ``prefill="w8a16"`` (the former BF16 down), or ``explicit``."""
    if prefill not in PREFILL_FORMS:
        raise ValueError(f"prefill is one of {PREFILL_FORMS}")
    if explicit is not None:
        if explicit and (weights != "mxfp4" or not wire):
            raise ValueError("MXFP4 A8 down requires MXFP4 weights and wire input")
        return bool(explicit)
    return weights == "mxfp4" and bool(wire) and MXFP4_STREAM_DOWN_A8 and prefill != "w8a16"


def _streams(wire: bool, weights: str = "fp8", prefill: str = "auto") -> bool:
    """Whether ``auto``'s large route is ``stream``: MXFP4 weights (wire
    or native BF16 input; their only large route), NVFP4 (original BF16 input), W8A8 FP8
    programs (``_w8a8``), and W8A16 FP8 wire input on GB10 (weight-bandwidth
    bound); otherwise the grouped GEMM SM120 was measured with."""
    if weights == "mxfp4":
        return True
    if weights in ("nvfp4", "nvfp4a4"):
        return True
    if _w8a8(wire, weights, prefill):
        return True
    return bool(wire) and _sm121()


def auto_large_rows(wire: bool, weights: str = "fp8", prefill: str = "auto") -> int:
    """Live-row threshold of ``auto``'s large route (``stream`` or ``prefill``)."""
    if weights in ("nvfp4", "nvfp4a4"):
        return 0  # Declared W4A4 arithmetic is independent of the live-row count.
    if weights == "mxfp4":
        return AUTO_MXFP4_STREAM_ROWS_SM121 if _sm121() else AUTO_MXFP4_STREAM_ROWS
    if _w8a8(wire, weights, prefill):
        return AUTO_W8A8_STREAM_ROWS_SM121 if _sm121() else AUTO_W8A8_STREAM_ROWS_SM120
    return AUTO_STREAM_ROWS if _streams(wire, weights, prefill) else AUTO_PREFILL_ROWS


def _large_route(wire: bool, weights: str = "fp8", prefill: str = "auto") -> str:
    return "stream" if _streams(wire, weights, prefill) else "prefill"


def prefill_forms(g: "Fp8MoeGeometry", max_rows: int, wire: bool = True) -> list[str]:
    """The ``prefill`` forms besides "auto" whose program of ``g`` at capacity
    ``max_rows`` differs from the "auto" one (packages carry those too)."""
    def policy(prefill):
        return (_large_route(wire, g.kind, prefill), auto_large_rows(wire, g.kind, prefill),
                _w8a8(wire, g.kind, prefill), _mxfp4_down_a8(wire, g.kind, prefill))

    base = policy("auto")
    return [form for form in PREFILL_FORMS[1:] if policy(form) != base
            and int(max_rows) > min(base[1], policy(form)[1])]


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
    # NVFP4 requires A4 at every row count; a16 remains for other weight formats.
    activations: str = "a16"
    activation: str = "silu_bf16_product"
    projection: str = "bf16"

    def __post_init__(self) -> None:
        import math

        if self.activation not in ("silu_bf16_product", "silu_post_clamp_fp32_product", "swigluoai_uninterleave"):
            raise ValueError("unknown expert activation contract")
        if self.projection != "bf16":
            raise ValueError("expert projections must round to BF16 before activation/route sum")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("SwiGLU limit must be finite and nonnegative")
        if self.weights == "mxfp4" and self.activation != "silu_bf16_product":
            raise ValueError("MXFP4 supports only the pre-SiLU BF16 product contract")
        if self.activation == "swigluoai_uninterleave" and self.weights != "nvfp4":
            raise ValueError("SwiGLU-OAI experts require native NVFP4 A4")
        if self.activations not in ("a16", "a4") or (self.activations == "a4" and self.weights != "nvfp4"):
            raise ValueError("activations is 'a16', or 'a4' with NVFP4 weights")
        if self.weights == "nvfp4" and self.activations != "a4":
            raise ValueError("declared NVFP4 experts require W4A4; W4A16 is not a serving mode")
        if self.weights not in ("fp8", "mxfp4", "nvfp4", "bf16"):
            raise ValueError("weights is fp8, mxfp4, nvfp4 or bf16")
        if min(self.hidden, self.experts, self.intermediate, self.tp) <= 0:
            raise ValueError("expert geometry dimensions must be positive")
        if self.weights == "bf16" and (self.hidden % 128 or self.intermediate % 128 or self.intermediate // 128 < self.tp):
            raise ValueError("BF16 experts need 128-aligned hidden/intermediate and a block per rank")
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
                              self.swiglu_limit, self.weights, self.activations, self.activation, self.projection)


# MiMo V2 Flash (no SwiGLU clamp in its config) and the GLM 5.3 MTP layer.
GEOMETRIES = {
    "step5": Fp8MoeGeometry("step5", hidden=4096, experts=352, top_k=8, intermediate=1536,
                              swiglu_limit=7.0, activation="silu_post_clamp_fp32_product"),
    "step5_bf16": Fp8MoeGeometry("step5_bf16", hidden=4096, experts=352, top_k=8, intermediate=1536,
                                   swiglu_limit=7.0, weights="bf16", activation="silu_post_clamp_fp32_product"),
    "step5_nvfp4": Fp8MoeGeometry("step5_nvfp4", hidden=4096, experts=352, top_k=8, intermediate=1536,
                                    swiglu_limit=7.0, weights="nvfp4", activations="a4",
                                    activation="silu_post_clamp_fp32_product"),
    "minimax_nvfp4": Fp8MoeGeometry("minimax_nvfp4", hidden=6144, experts=128, top_k=4, intermediate=3072,
                                      swiglu_limit=7.0, weights="nvfp4", activations="a4",
                                      activation="swigluoai_uninterleave"),
    "mimo": Fp8MoeGeometry("mimo", hidden=4096, experts=256, top_k=8, intermediate=2048),
    "glm": Fp8MoeGeometry("glm", hidden=6144, experts=256, top_k=8, intermediate=2048),
    # GLM 5.3 Flash: 288 experts, SwiGLU clamped at 10 as its config says.
    "glmf": Fp8MoeGeometry("glmf", hidden=4096, experts=288, top_k=8, intermediate=2048, swiglu_limit=10.0),
    # Qwen 3.8 Flash Next: 512 experts, softmax top-10, unclamped SiLU.
    "qwen4": Fp8MoeGeometry("qwen4", hidden=2560, experts=512, top_k=10, intermediate=640),
    # MiMo V2.6 Pro: MXFP4 experts, 384 of them, top-8, unclamped SiLU.
    "mimop": Fp8MoeGeometry("mimop", hidden=6144, experts=384, top_k=8, intermediate=2048, weights="mxfp4"),
    "mimof": Fp8MoeGeometry("mimof", hidden=4096, experts=256, top_k=8, intermediate=2048, weights="mxfp4"),
    # NVIDIA ModelOpt NVFP4 releases: native W4A4 at every row count.
    "glm_nvfp4": Fp8MoeGeometry("glm_nvfp4", hidden=6144, experts=256, top_k=8, intermediate=2048, weights="nvfp4", activations="a4"),
    "glmf_nvfp4": Fp8MoeGeometry("glmf_nvfp4", hidden=4096, experts=288, top_k=8, intermediate=2048,
                                 swiglu_limit=10.0, weights="nvfp4", activations="a4"),
    "qwen4_nvfp4": Fp8MoeGeometry("qwen4_nvfp4", hidden=2560, experts=512, top_k=10, intermediate=640,
                                  weights="nvfp4", activations="a4"),
    # GLM 5.3 Flash's NVFP4 dense MLPs (layers 0-2 of nvidia/GLM-5.3-Flash-NVFP4)
    # as one always-selected expert: the coordinator runs them with ids 0, weight 1.
    "glmfdense_nvfp4": Fp8MoeGeometry("glmfdense_nvfp4", hidden=4096, experts=1, top_k=1, intermediate=12288,
                                      swiglu_limit=10.0, weights="nvfp4", activations="a4"),
    "glmfdense_nvfp4a4": Fp8MoeGeometry("glmfdense_nvfp4a4", hidden=4096, experts=1, top_k=1, intermediate=12288,
                                        swiglu_limit=10.0, weights="nvfp4", activations="a4"),
    # Explicit A4 names retained for package compatibility.
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


def _regions(g: Fp8MoeGeometry, route: str, rows: int, wire: bool = True,
             mxfp4_down_a8: bool = False) -> list[int]:
    rows = max(int(rows), 1)
    if route == "stream" and g.activations == "a4":
        pairs = rows * g.top_k
        padded = stream_max_tiles(g.experts, pairs) * STREAM_TILE_M
        split = 0 if nvfp4_a4_fused() else pairs * g.slice * 2
        return [
            meta_words(g.experts, stream_max_tiles(g.experts, pairs)) * 4,
            pairs * 4,                                 # pair_row
            pairs * 4,                                 # pair_pos
            padded * g.hidden // 2,                    # packed input rows
            padded * g.hidden // 16,                   # their E4M3 scales
            split,                                     # gate (unfused route only)
            split,                                     # up (unfused route only)
            padded * g.slice // 2,                     # packed SwiGLU rows
            padded * g.slice // 16,                    # their scales
            pairs * g.hidden * 2,                      # down output
        ]
    if route == "stream":
        pairs = rows * g.top_k
        act_bytes = mxfp8_down_row_bytes(g.slice) if mxfp4_down_a8 else 2 * g.slice
        return [
            meta_words(g.experts, stream_max_tiles(g.experts, pairs)) * 4,
            pairs * 4,                                 # pair_row
            pairs * 4,                                 # pair_pos
            stream_max_tiles(g.experts, pairs) * STREAM_TILE_M * act_bytes,  # SwiGLU, chunk-padded rows
            pairs * g.hidden * 2,                      # down output
        ] + ([] if wire or g.weights == "mxfp4" else [rows * (g.hidden + g.hidden // 32)])
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


def nvfp4_a4_fused() -> bool:
    """The W4A4 stream route's gate/up: one fused gate/up + SwiGLU + NVFP4 quantization
    launch (``StreamNvfp4GateUpA4``, default), or with ``B12X_NVFP4_A4_FUSED=0`` at
    export the separate gate GEMM, up GEMM and ``ChunkSwiGLUA4`` (A/B)."""
    import os

    return os.environ.get("B12X_NVFP4_A4_FUSED", "1") != "0"


def _resolve(route: str, max_rows: int, wire: bool = True, weights: str = "fp8", prefill: str = "auto") -> str:
    """Select one serving route. NVFP4 always streams calibrated A4 from BF16;
    other formats retain their row-count-dependent policies."""
    # A4 is a numerical contract, not a large-row optimization. Chunk padding
    # also supports a single live row; never fall back to the A16 GEMV.
    if weights in ("nvfp4", "nvfp4a4"):
        if wire:
            raise ValueError("calibrated NVFP4 A4 requires original BF16 input")
        return "stream"
    if weights == "bf16":
        if wire or route == "stream":
            raise ValueError("native BF16 experts require BF16 input and decode/prefill/auto route")
    if weights == "mxfp4":
        if route == "prefill":
            raise ValueError("MXFP4 experts have no prefill route (use stream)")
    large = auto_large_rows(wire, weights, prefill)
    return "decode" if route == "auto" and int(max_rows) <= large else route


def fp8_moe_scratch_bytes(g: Fp8MoeGeometry, route: str, rows: int, wire: bool = True,
                          prefill: str = "auto", mxfp4_down_a8: "bool | None" = None) -> int:
    mxfp4_down_a8 = _mxfp4_down_a8(wire, g.kind, prefill, mxfp4_down_a8)
    route = _resolve(route, rows, wire, g.kind, prefill)
    if route == "auto":
        return max(fp8_moe_scratch_bytes(g, "decode", rows, wire, prefill, mxfp4_down_a8),
                   fp8_moe_scratch_bytes(g, _large_route(wire, g.kind, prefill), rows, wire, prefill, mxfp4_down_a8))
    return sum(_align(b) for b in _regions(g, route, rows, wire, mxfp4_down_a8))


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
            gemv = GroupedMxfp4Gemv if g.weights == "mxfp4" else GroupedFp8Gemv
            native = {"native_bf16": True} if g.weights == "bf16" else {}
            self.gate_up = gemv(n=2 * i, k=h, experts=e, split=i, max_rows=tile_rows, warps=warps_h,
                                groups=_gate_up_groups(tile_rows), gather=True, **native)
            # K split over warps in whole 128 blocks: 4 where they divide evenly,
            # else the largest divisor <= 8 of the blocks (Qwen's 640: 5 warps).
            warps_i = 4 if i % 512 == 0 else max(d for d in range(1, 9) if (i // 128) % d == 0)
            if g.weights == "mxfp4" and i % 128:
                warps_i = min(4, (i + 127) // 128)
            self.down = gemv(n=h, k=i, experts=e, max_rows=tile_rows, warps=warps_i, **native)
        else:
            self.gate_up = GroupedFp8Gemm(n=i, k=h, experts=e, halves=2, native_bf16=g.weights == "bf16")
            self.down = GroupedFp8Gemm(n=h, k=i, experts=e, halves=1, native_bf16=g.weights == "bf16")
        self.swiglu = MoeSwiGLU(inter=i, limit=g.swiglu_limit, activation=g.activation)
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

    def __init__(self, g: Fp8MoeGeometry, max_rows: int, wire: bool, prefill: str = "auto",
                 mxfp4_down_a8: bool = False):
        self.g, self.wire = g, bool(wire)
        self.mxfp4_down_a8 = bool(mxfp4_down_a8)
        self.act_bytes = mxfp8_down_row_bytes(g.slice) if self.mxfp4_down_a8 else 2 * g.slice
        # BF16 input: the rows are first quantized to FP8 K32 wire rows (the
        # coordinator's expert-input rule: UE8M0 per 32, amax floor 1e-4).
        self.quant = None if wire or g.weights == "mxfp4" else _MXFP8RowsQuantLaunch(
            g.hidden, BFloat16, _WARP_SUBGROUP_WIDTH, _QUANT_THREADS, 32, False, 1e-4, wire_rows=True)
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        self.max_tiles = stream_max_tiles(e, int(max_rows) * k)
        self.prep = MoePrep(experts=e, top_k=k, pad=1, max_tiles=self.max_tiles, tile_rows=STREAM_TILE_M,
                            chunked=True)
        if g.weights == "mxfp4":
            self.gate_up = StreamMxfp4GateUp(inter=i, hidden=h, experts=e, limit=g.swiglu_limit,
                                             qmma=MXFP4_STREAM_QMMA and wire,
                                             act_mxfp8=self.mxfp4_down_a8, wire=wire)
            self.down = (StreamMxfp4DownA8 if self.mxfp4_down_a8 else StreamMxfp4Down)(hidden=h, inter=i, experts=e)
        else:
            self.gate_up = StreamFp8GateUp(inter=i, hidden=h, experts=e, limit=g.swiglu_limit, activation=g.activation,
                                           qmma=FP8_STREAM_QMMA and prefill != "w8a16")
            self.down = StreamFp8Down(hidden=h, inter=i, experts=e)
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return ("stream", self.max_tiles, self.wire, self.gate_up.key(), self.down.key(), self.combine.key())

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
        y_at = act_at + _al(Int64(max_tiles) * Int64(STREAM_TILE_M * self.act_bytes))
        ptr = lambda dtype, at: cute.make_ptr(dtype, at, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        rows_x = x
        if cutlass.const_expr(self.quant is not None):
            wire_at = y_at + _al(Int64(pairs) * Int64(g.hidden * 2))
            # One warp quantizes 8 K32 groups of a row (subgroups of 4 lanes); grid-stride over rows.
            tasks = rows * Int32((g.hidden // 32 + 7) // 8)
            grid = (tasks + Int32(_QUANT_THREADS // 32 - 1)) // Int32(_QUANT_THREADS // 32)
            if grid > Int32(_QUANT_MAX_CTAS):
                grid = Int32(_QUANT_MAX_CTAS)
            self.quant(cute.make_ptr(cutlass.BFloat16, Int64(x.toint()), cute.AddressSpace.gmem, assumed_align=16),
                       ptr(cutlass.Uint32, wire_at), ptr(cutlass.Uint8, wire_at + Int64(g.hidden)),
                       ptr(cutlass.Uint8, wire_at), rows, Int32(g.hidden), grid, stream)
            rows_x = ptr(cutlass.Uint8, wire_at)
        meta = ptr(cutlass.Int32, base)
        pair_row = ptr(cutlass.Int32, pair_row_at)
        pair_pos = ptr(cutlass.Int32, pair_pos_at)
        act = ptr(cutlass.Uint8 if self.mxfp4_down_a8 else cutlass.BFloat16, act_at)
        y = ptr(cutlass.BFloat16, y_at)
        ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.prep(ids_i, meta, pair_row, pair_pos, rows, pairs, stream)
        self.gate_up(rows_x, pair_row, meta, w1, s1, w3, s3, act, max_tiles, stream)
        self.down(act, meta, w2, s2, y, max_tiles, stream)
        self.combine(y, pair_pos, weights, out, rows, stream)


class _StreamNvfp4A4Route:
    """The W4A4 NVFP4 ``stream`` route (``_nvfp4_moe_a4``): chunk-padded NVFP4
    input rows, gate/up block-scaled FP4 GEMMs with SwiGLU and its NVFP4
    quantization fused (``StreamNvfp4GateUpA4``; ``nvfp4_a4_fused``), the down
    GEMM, route combine."""

    def __init__(self, g: Fp8MoeGeometry, max_rows: int, wire: bool):
        self.g = g
        h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
        self.max_tiles = stream_max_tiles(e, int(max_rows) * k)
        self.prep = MoePrep(experts=e, top_k=k, pad=1, max_tiles=self.max_tiles, tile_rows=STREAM_TILE_M,
                            chunked=True)
        self.rows_in = ChunkRowsA4(k=h, wire=wire, experts=e, scale_rows=i)
        self.fused = nvfp4_a4_fused()
        if self.fused:
            self.gate_up = StreamNvfp4GateUpA4(inter=i, hidden=h, experts=e, limit=g.swiglu_limit, activation=g.activation)
            self.swiglu = None
        else:
            self.gate_up = StreamNvfp4LinearA4(n=i, k=h, experts=e)
            self.swiglu = ChunkSwiGLUA4(inter=i, hidden=h, experts=e, limit=g.swiglu_limit, activation=g.activation)
        self.down = StreamNvfp4LinearA4(n=h, k=i, experts=e)
        self.small_gate = ()
        self.small_down = ()
        if self.fused:
            # Separate kernels keep the wide tile's SMEM reservation out of the
            # sparse-expert launch. Every specialization is planned before capture.
            self.gate_up = StreamNvfp4GateUpA4(inter=i, hidden=h, experts=e, limit=g.swiglu_limit, activation=g.activation,
                                               live_min=17)
            self.down = StreamNvfp4LinearA4(n=h, k=i, experts=e, live_min=17)
            self.small_gate = tuple(StreamNvfp4GateUpA4(inter=i, hidden=h, experts=e, limit=g.swiglu_limit, activation=g.activation,
                                                        tile_m=16, tile_n=n, live_max=16, stages=SMALL_GATE_STAGES)
                                    for n in (32, 64, 128))
            self.small_down = tuple(StreamNvfp4LinearA4(n=h, k=i, experts=e, tile_m=16, tile_n=n,
                                                        live_max=16) for n in (32, 64, 128))
        self.combine = MoeCombine(hidden=h, top_k=k)

    def key(self) -> tuple:
        return ("stream_nvfp4_a4", self.max_tiles, self.rows_in.key(), self.gate_up.key(),
                None if self.swiglu is None else self.swiglu.key(), self.down.key(), self.combine.key(),
                tuple(p.key() for p in self.small_gate), tuple(p.key() for p in self.small_down))

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
        split = Int64(0) if self.fused else Int64(pairs) * Int64(g.slice * 2)
        up_at = gate_at + _al(split)
        aq_at = up_at + _al(split)
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
        self.rows_in(x, pair_row, meta, s1, xq, xs, pairs, stream)
        if cutlass.const_expr(self.fused):
            sms = runtime_num_sm()
            # The sparsest valid routing has top_k experts, one tile each, so the
            # sparse grid is row-count independent and at one row sits far below
            # this device's warp capacity. Take the narrowest N tile whose grid
            # reaches the target CTAs per queried SM: halving N doubles the CTAs
            # and splits only output columns, never an output's K order.
            target = sms * Int32(SMALL_GATE_CTAS_PER_SM)
            if Int32(g.top_k * (g.slice // 16)) >= target:
                self.small_gate[0](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, max_tiles, stream)
            elif Int32(g.top_k * (g.slice // 32)) >= target:
                self.small_gate[1](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, max_tiles, stream)
            else:
                self.small_gate[2](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, max_tiles, stream)
            # Bound by pairs, so duplicate expert ids cannot bypass the wide path.
            if pairs > Int32(16):
                self.gate_up(xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, max_tiles, stream)
        else:
            self.gate_up(xq, xs, meta, w1, s1, gate, max_tiles, stream)
            self.gate_up(xq, xs, meta, w3, s3, up, max_tiles, stream)
            self.swiglu(gate, up, meta, s2, aq, a_s, max_tiles, stream)
        if cutlass.const_expr(self.fused):
            if Int32(g.top_k * (g.hidden // 128)) >= sms * Int32(2):
                self.small_down[2](aq, a_s, meta, w2, s2, y, max_tiles, stream)
            elif Int32(g.top_k * (g.hidden // 64)) >= sms * Int32(2):
                self.small_down[1](aq, a_s, meta, w2, s2, y, max_tiles, stream)
            else:
                self.small_down[0](aq, a_s, meta, w2, s2, y, max_tiles, stream)
            if pairs > Int32(16):
                self.down(aq, a_s, meta, w2, s2, y, max_tiles, stream)
        else:
            self.down(aq, a_s, meta, w2, s2, y, max_tiles, stream)
        self.combine(y, pair_pos, weights, out, rows, stream)


def _make_route(g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool, prefill: str = "auto",
                mxfp4_down_a8: bool = False):
    if route == "stream" and g.activations == "a4":
        return _StreamNvfp4A4Route(g, max_rows, wire)
    if route == "stream":
        return _StreamRoute(g, max_rows, wire, prefill, mxfp4_down_a8)
    return _Route(g, route, max_rows, wire)


class _Fp8Moe:
    def __init__(self, g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool, prefill: str = "auto",
                 mxfp4_down_a8: bool = False):
        self.route = route
        self.threshold = auto_large_rows(wire, g.kind, prefill)
        self.decode = _Route(g, "decode", max_rows, wire) if route in ("decode", "auto") else None
        large = _large_route(wire, g.kind, prefill) if route == "auto" else route
        self.prefill = (_make_route(g, large, max_rows, wire, prefill, mxfp4_down_a8)
                        if route in ("prefill", "stream", "auto") else None)

    def key(self) -> tuple:
        return (self.threshold,) + tuple(part.key() if part is not None else None
                                         for part in (self.decode, self.prefill))

    def weight_read_grouping(self, weights: str, max_rows: int) -> dict:
        """Static accounting only; never part of the compiler key or launch."""
        if weights == "nvfp4":
            # Sparse/wide W4A4 tiles depend on expert occupancy, not only rows.
            # One full slice per routed occurrence is an explicit traffic bound.
            return {"contract": "conservative_route_upper_bound",
                    "segments": [{"end_rows": int(max_rows), "group_rows": 1}]}

        def group(part):
            if isinstance(part, _Route):
                gate, down = part.gate_up, part.down
                values = ((16 * gate.m_tiles, 16 * down.m_tiles) if part.route == "decode"
                          else (gate.tile_m, down.tile_m))
                if values[0] != values[1]:
                    raise ValueError("expert gate/up and down weight-read groups differ")
                return int(values[0])
            if isinstance(part, _StreamRoute):
                return int(part.prep.tile_rows)
            raise ValueError("expert route has no exact weight-read grouping contract")

        segments = []
        if self.route == "auto":
            cutoff = min(int(max_rows), int(self.threshold))
            segments.append({"end_rows": cutoff, "group_rows": group(self.decode)})
            if cutoff < int(max_rows):
                large = group(self.prefill)
                if large == segments[-1]["group_rows"]:
                    segments[-1]["end_rows"] = int(max_rows)
                else:
                    segments.append({"end_rows": int(max_rows), "group_rows": large})
        else:
            part = self.decode if self.route == "decode" else self.prefill
            segments.append({"end_rows": int(max_rows), "group_rows": group(part)})
        return {"contract": "exact_grouping", "segments": segments}

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


def compile_fp8_moe_aot(g: Fp8MoeGeometry, *, route: str, max_rows: int, wire: bool = True,
                        prefill: str = "auto", mxfp4_down_a8: "bool | None" = None) -> AotProgram:
    """Routed FP8 experts of ``g`` for ``rows <= max_rows``; see the module docstring
    (``prefill``: the large-row form of FP8 programs, "auto", "w8a8" or "w8a16")."""
    if route not in ("decode", "prefill", "stream", "auto"):
        raise ValueError("route is 'decode', 'prefill', 'stream' or 'auto'")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    mxfp4_down_a8 = _mxfp4_down_a8(wire, g.kind, prefill, mxfp4_down_a8)
    requested, route = route, _resolve(route, max_rows, wire, g.kind, prefill)
    launch = _Fp8Moe(g, route, max_rows, wire, prefill, mxfp4_down_a8)
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
    elif g.weights == "bf16":
        weights = (
            Operand("w1", torch.bfloat16, f"[{e},{i},{h}]"),
            Operand("s1", torch.uint8, "[0]", note="unused; zero scale storage"),
            Operand("w3", torch.bfloat16, f"[{e},{i},{h}]"),
            Operand("s3", torch.uint8, "[0]", note="unused; zero scale storage"),
            Operand("w2", torch.bfloat16, f"[{e},{h},{i}]"),
            Operand("s2", torch.uint8, "[0]", note="unused; zero scale storage"),
        )
    elif g.weights == "mxfp4":
        weights = (
            Operand("w1", torch.uint8, f"[{e},{i},{h // 2}]", note="packed E2M1 (even element low)"),
            Operand("s1", torch.uint8, f"[{e},{i},{h // 32}]", align=4, note="UE8M0 per 32"),
            Operand("w3", torch.uint8, f"[{e},{i},{h // 2}]"),
            Operand("s3", torch.uint8, f"[{e},{i},{h // 32}]", align=4),
            Operand("w2", torch.uint8, f"[{e},{h},{i // 2}]"),
            Operand("s2", torch.uint8, f"[{e},{h},{mxfp4_scale_row_bytes(i)}]", align=4,
                    note="UE8M0 per 32, each row padded to four bytes"),
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
                  "wire": bool(wire), "weights": g.weights, "activations": g.activations,
                  "activation": g.activation, "projection": g.projection,
                  "prefill": prefill, "mxfp4_down_a8": bool(mxfp4_down_a8),
                  "weight_read_grouping": launch.weight_read_grouping(g.weights, max_rows)},
        scratch={"scratch": lambda rows: fp8_moe_scratch_bytes(g, route, rows, wire, prefill, mxfp4_down_a8)},
        doc=__doc__,
    )
