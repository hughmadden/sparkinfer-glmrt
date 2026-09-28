"""Native AOT DeepSeek V4 FFN-side programs: shared expert, router scores,
routed-expert input quantizer.

``compile_dsv4_shared_ffn_aot(geometry, max_rows=...)`` equals the
prototype's shared expert (``b12x_model.py`` DeepseekV4Layer.ffn, 4c)::

    gate_up = block_fp8_linear(x, [w1; w3])            # gate rows first, BF16
    up      = clamp(up, -10, 10); gate = min(gate, 10)  # FP32
    hidden  = bf16(silu(gate) * up)                     # silu = g / (1 + exp(-g))
    out     = block_fp8_linear(hidden, w2)              # BF16 [rows, H]

Both GEMMs are ``Fp8LinearStage`` (per-token K128 activation quantization,
the prepared plan's default lowering for ``max_rows``). ABI (``I`` =
moe_inter, 2048 Flash / 3072 Pro)::

    x          bf16 [rows,H]      in
    w13        fp8  [2I,H]        in   pack_weight(cat(w1, w3)).weight.values (gate rows first)
    w13_scale  u8   ceil(2I/128)*ceil(H/128)*512 bytes   .weight.scale_mma storage
    w2         fp8  [H,I]         in   pack_weight(w2).weight.values
    w2_scale   u8   ceil(H/128)*ceil(I/128)*512 bytes    .weight.scale_mma storage
    out        bf16 [rows,H]      out
    scratch    u8   shared_ffn_scratch_bytes(geometry, rows, max_rows)
    rows       int32 (1 <= rows <= max_rows)

``compile_dsv4_router_scores_aot(geometry)``: FP32-accumulated BF16 scores.
Live rows <= 32 run a bandwidth-bound skinny GEMV (FP32 output); more rows run
the V4.1 TMA-fed warp-MMA kernel (``_V41RouterScores``); the branch is on the
``live_rows`` scalar inside the program. V4.1 header names::

    x          bf16 [rows,H]      in   ffn input (after ffn_norm)
    w          bf16 [E,H]         in   ffn.gate.weight
    logits     f32  [rows,E]      out  raw scores x . gate^T (score transform,
                                        bias and top-k are separate)
    live_rows  int32

``compile_dsv4_expert_input_quant_aot(geometry)`` wraps
``compile_mxfp8_rows_quant_aot(size_k=H, expected_m=80, amax_floor=1e-4,
wire_rows=True)``: BF16 rows -> wire rows of ``H`` E4M3 bytes followed by
``H/32`` UE8M0 K32 scale bytes (row stride ``H + H/32``: 4224 Flash, 7392
Pro), scale = 2^ceil(log2(max(amax, 1e-4) / 448)). Header names::

    source_ptr     bf16 [rows,H]              in
    values_ptr     u8   [rows,H+H/32]         out  wire rows (row payload)
    scale_rows_ptr u8   values_ptr + H        out  must be exactly values_ptr + H
    scale_mma_ptr  u8   any 16-B address       unused with wire rows
    m              int32  live rows
    grid_x         int32  CTAs; any value >= 1 is correct (grid-stride loop).
                          Native choice: expert_input_quant_grid(H, rows, sm_count)
                          = min(ceil(rows * ceil(H/32/8) / 8), 4 * sm_count)
                          (``mxfp8_rows_quant_aot_grid(expected_m=80)``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import div_rn_f32, fmax_f32, fmin_f32

from ._common import FLASH, AotProgram, DSV4Geometry, Operand, Scalar, compile_program
from ._linear import Fp8LinearStage, block_fp8_lowering, stage_scratch_bytes

__all__ = [
    "compile_dsv4_expert_input_quant_aot",
    "compile_dsv4_router_scores_aot",
    "compile_dsv4_shared_ffn_aot",
    "expert_input_quant_grid",
    "shared_ffn_scratch_bytes",
]

_ALIGN = 1024
_SWIGLU_THREADS = 256


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _slices(max_rows: int, k: int, n: int) -> int:
    return int(block_fp8_lowering(max_rows=max_rows, in_features=k, out_features=n).policy.split_k_slices)


def shared_ffn_scratch_bytes(geometry: DSV4Geometry, rows: int, max_rows: int) -> int:
    """gate_up BF16 [rows,2I], hidden BF16 [rows,I], one reused FP8 stage region."""
    h, i = geometry.hidden, geometry.moe_inter
    return (_align(rows * 2 * i * 2) + _align(rows * i * 2)
            + max(stage_scratch_bytes(h, 2 * i, rows, _slices(max_rows, h, 2 * i)),
                  stage_scratch_bytes(i, h, rows, _slices(max_rows, i, h))))


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class ClampedSwiGLU:
    """hidden[r, j] = bf16(silu(min(g, L)) * clamp(u, -L, L)); g, u from [gate | up]."""

    def __init__(self, inter: int, limit: float):
        self.inter = int(inter)
        self.limit = float(limit)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, hidden: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        i = self.inter
        self.kernel(cute.make_tensor(gate_up, cute.make_layout((m, 2 * i), stride=(2 * i, 1))),
                    cute.make_tensor(hidden, cute.make_layout((m, i), stride=(i, 1)))).launch(
            grid=(rows, (i + _SWIGLU_THREADS - 1) // _SWIGLU_THREADS, 1),
            block=(_SWIGLU_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Tensor, hidden: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(_SWIGLU_THREADS) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.inter):
            gate = fmin_f32(Float32(gate_up[row, col]), Float32(self.limit))
            up = fmin_f32(fmax_f32(Float32(gate_up[row, Int64(self.inter) + col]), Float32(-self.limit)),
                          Float32(self.limit))
            silu = div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False))
            hidden[row, col] = (silu * up).to(BFloat16)


class _SharedFFN:
    def __init__(self, geometry: DSV4Geometry, max_rows: int):
        self.h, self.i = geometry.hidden, geometry.moe_inter
        self.w13 = Fp8LinearStage(in_features=self.h, out_features=2 * self.i, max_rows=max_rows)
        self.w2 = Fp8LinearStage(in_features=self.i, out_features=self.h, max_rows=max_rows)
        self.swiglu = ClampedSwiGLU(self.i, geometry.swiglu_limit)

    @cute.jit
    def __call__(self, x: cute.Pointer, w13: cute.Pointer, w13_scale: cute.Pointer,
                 w2: cute.Pointer, w2_scale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        base = Int64(scratch.toint())
        gate_up = cute.make_ptr(BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        hidden_off = base + _align_i64(m * Int64(4 * self.i))
        hidden = cute.make_ptr(BFloat16, hidden_off, cute.AddressSpace.gmem, assumed_align=16)
        stage = hidden_off + _align_i64(m * Int64(2 * self.i))
        self.w13(x, w13, w13_scale, gate_up, stage, rows, stream)
        self.swiglu(gate_up, hidden, rows, stream)
        self.w2(hidden, w2, w2_scale, out, stage, rows, stream)


def compile_dsv4_shared_ffn_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int) -> AotProgram:
    """V4 shared expert for ``rows <= max_rows``; see module docstring."""
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    launch = _SharedFFN(geometry, max_rows)
    h, i = geometry.hidden, geometry.moe_inter
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w13", torch.float8_e4m3fn, f"[{2 * i},{h}]"),
        Operand("w13_scale", torch.uint8, f"[{(2 * i + 127) // 128 * ((h + 127) // 128) * 512}]"),
        Operand("w2", torch.float8_e4m3fn, f"[{h},{i}]"),
        Operand("w2_scale", torch.uint8, f"[{(h + 127) // 128 * ((i + 127) // 128) * 512}]"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[shared_ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="dsv4_shared_ffn", operands=operands, scalars=(Scalar("rows"),),
        key=(h, i, geometry.swiglu_limit, launch.w13.key(), launch.w2.key()),
        geometry={"hidden": h, "moe_inter": i, "max_rows": max_rows,
                  "swiglu_limit": geometry.swiglu_limit},
        scratch={"scratch": lambda rows: shared_ffn_scratch_bytes(geometry, rows, max_rows)},
        doc=__doc__,
    )


ROUTER_SKINNY_MAX_ROWS = 32


class _RouterScores:
    """Skinny GEMV (FP32 out) for live_rows <= 32, else the V4.1 TMA MMA kernel."""

    def __init__(self, experts: int, hidden: int):
        from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, skinny_config
        from b12x.moe._shared.v41_router import _V41RouterScores

        self.skinny = SkinnyBf16Gemv(experts, hidden, out_dtype=cutlass.Float32,
                                     **skinny_config(experts, hidden, cutlass.Float32))
        self.mma = _V41RouterScores(experts, hidden=hidden)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, logits: cute.Pointer, live_rows: Int32,
                 stream: cuda.CUstream):
        if live_rows <= Int32(ROUTER_SKINNY_MAX_ROWS):
            self.skinny(x, w, logits, live_rows, stream)
        else:
            self.mma(x, w, logits, live_rows, stream)


def compile_dsv4_router_scores_aot(geometry: DSV4Geometry = FLASH, experts: int | None = None) -> AotProgram:
    """Raw FP32 router scores ``x @ gate^T``; see module docstring."""
    e = geometry.routed_experts if experts is None else int(experts)
    h = geometry.hidden
    if e not in (256, 384) or h not in (4096, 7168):
        raise ValueError("DSV4 router scores support 256/384 experts over hidden 4096/7168")
    launch = _RouterScores(e, h)
    s = launch.skinny
    return compile_program(
        launch, name="dsv4_router_scores",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w", torch.bfloat16, f"[{e},{h}]", note="ffn.gate.weight"),
                  Operand("logits", torch.float32, f"[rows,{e}]", "out")),
        scalars=(Scalar("live_rows"),),
        key=(e, h, ROUTER_SKINNY_MAX_ROWS, s.cols, s.rows_per_tile, s.threads),
        geometry={"hidden": h, "experts": e, "skinny_max_rows": ROUTER_SKINNY_MAX_ROWS}, doc=__doc__,
    )


def expert_input_quant_grid(hidden: int, rows: int, sm_count: int) -> int:
    """Native CTA count for the expert input quantizer (``expected_m=80``)."""
    from b12x._lib.quant.mxfp8_rows import mxfp8_rows_quant_aot_grid

    return mxfp8_rows_quant_aot_grid(size_k=hidden, rows=rows, expected_m=80, sm_count=sm_count)


def compile_dsv4_expert_input_quant_aot(geometry: DSV4Geometry = FLASH) -> AotProgram:
    """Routed-expert BF16 -> FP8 K32 wire-row quantizer; see module docstring."""
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    h = geometry.hidden
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="dsv4_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4},
        doc=__doc__,
    )
