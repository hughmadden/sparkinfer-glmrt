"""Native AOT exact FP8 routed experts (checkpoint E4M3 + FP32 128x128 scales).

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
``auto`` branches on the live row count (grouped GEMM above
``AUTO_PREFILL_ROWS``) and sizes its scratch for both. Scratch (1024-aligned regions, live
row layout): metadata, ``pair_row`` / ``pair_pos``, BF16 activations (reused
for the down projection's output), ``gate|up`` BF16, SwiGLU BF16.
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

__all__ = ["Fp8MoeGeometry", "GEOMETRIES", "compile_fp8_moe_aot", "fp8_moe_scratch_bytes"]

_ALIGN = 1024
TILE_M = 64
DECODE_MAX_ROWS = 64
# Live rows above which ``auto`` takes the grouped GEMM (RTX PRO 6000, MiMo
# TP4 slice, us: 1024 rows GEMV 1753 / GEMM 2059; 4096 rows 5728 / 5374).
AUTO_PREFILL_ROWS = 2048


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

    def __post_init__(self) -> None:
        if self.hidden % 128 or self.intermediate % (128 * self.tp):
            raise ValueError("hidden and the intermediate slice must be 128-aligned")
        if not 1 <= self.top_k <= 16:
            raise ValueError("top_k must be 1..16")

    @property
    def slice(self) -> int:
        return self.intermediate // self.tp

    def with_tp(self, tp: int) -> "Fp8MoeGeometry":
        return Fp8MoeGeometry(self.name, self.hidden, self.experts, self.top_k, self.intermediate, int(tp),
                              self.swiglu_limit)


# MiMo V2 Flash (no SwiGLU clamp in its config) and the GLM 5.3 MTP layer.
GEOMETRIES = {
    "mimo": Fp8MoeGeometry("mimo", hidden=4096, experts=256, top_k=8, intermediate=2048),
    "glm": Fp8MoeGeometry("glm", hidden=6144, experts=256, top_k=8, intermediate=2048),
    # GLM 5.3 Flash: 288 experts, SwiGLU clamped at 10 as its config says.
    "glmf": Fp8MoeGeometry("glmf", hidden=4096, experts=288, top_k=8, intermediate=2048, swiglu_limit=10.0),
    # Qwen 3.8 Flash Next: 512 experts, softmax top-10, unclamped SiLU.
    "qwen4": Fp8MoeGeometry("qwen4", hidden=2560, experts=512, top_k=10, intermediate=640),
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


def _resolve(route: str, max_rows: int) -> str:
    """``auto`` compiles only the GEMV when the capacity never reaches the GEMM."""
    return "decode" if route == "auto" and int(max_rows) <= AUTO_PREFILL_ROWS else route


def fp8_moe_scratch_bytes(g: Fp8MoeGeometry, route: str, rows: int) -> int:
    route = _resolve(route, rows)
    if route == "auto":
        return max(fp8_moe_scratch_bytes(g, "decode", rows), fp8_moe_scratch_bytes(g, "prefill", rows))
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
            self.gate_up = GroupedFp8Gemv(n=2 * i, k=h, experts=e, split=i, max_rows=tile_rows, warps=warps_h,
                                          gather=True)
            # K split over warps in whole 128 blocks: 4 where they divide evenly,
            # else the largest divisor <= 8 of the blocks (Qwen's 640: 5 warps).
            warps_i = 4 if i % 512 == 0 else max(d for d in range(1, 9) if (i // 128) % d == 0)
            self.down = GroupedFp8Gemv(n=h, k=i, experts=e, max_rows=tile_rows, warps=warps_i)
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


class _Fp8Moe:
    def __init__(self, g: Fp8MoeGeometry, route: str, max_rows: int, wire: bool):
        self.route = route
        self.decode = _Route(g, "decode", max_rows, wire) if route in ("decode", "auto") else None
        self.prefill = _Route(g, "prefill", max_rows, wire) if route in ("prefill", "auto") else None

    def key(self) -> tuple:
        return tuple(part.key() if part is not None else None for part in (self.decode, self.prefill))

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.route == "decode"):
            self.decode(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
        elif cutlass.const_expr(self.route == "prefill"):
            self.prefill(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
        else:
            if rows <= Int32(AUTO_PREFILL_ROWS):
                self.decode(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)
            else:
                self.prefill(x, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch, rows, stream)


def compile_fp8_moe_aot(g: Fp8MoeGeometry, *, route: str, max_rows: int, wire: bool = True) -> AotProgram:
    """Routed FP8 experts of ``g`` for ``rows <= max_rows``; see the module docstring."""
    if route not in ("decode", "prefill", "auto"):
        raise ValueError("route is 'decode', 'prefill' or 'auto'")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    requested, route = route, _resolve(route, max_rows)
    launch = _Fp8Moe(g, route, max_rows, wire)
    h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
    x = (Operand("x", torch.uint8, f"[rows,{h + h // 32}]", note="FP8 K32 wire rows") if wire
         else Operand("x", torch.bfloat16, f"[rows,{h}]"))
    operands = (
        x,
        Operand("ids", torch.int32, f"[rows,{k}]", align=4),
        Operand("weights", torch.float32, f"[rows,{k}]", align=4),
        Operand("w1", torch.float8_e4m3fn, f"[{e},{i},{h}]"),
        Operand("s1", torch.float32, f"[{e},{i // 128},{h // 128}]", align=4),
        Operand("w3", torch.float8_e4m3fn, f"[{e},{i},{h}]"),
        Operand("s3", torch.float32, f"[{e},{i // 128},{h // 128}]", align=4),
        Operand("w2", torch.float8_e4m3fn, f"[{e},{h},{i}]"),
        Operand("s2", torch.float32, f"[{e},{h // 128},{i // 128}]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[fp8_moe_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"fp8_moe_{g.name}_tp{g.tp}_{route}", operands=operands, scalars=(Scalar("rows"),),
        key=launch.key(),
        geometry={"requested_route": requested, "hidden": h, "experts": e, "top_k": k, "intermediate": g.intermediate, "tp": g.tp,
                  "slice": i, "swiglu_limit": g.swiglu_limit, "route": route, "max_rows": int(max_rows),
                  "wire": bool(wire)},
        scratch={"scratch": lambda rows: fp8_moe_scratch_bytes(g, route, rows)},
        doc=__doc__,
    )
