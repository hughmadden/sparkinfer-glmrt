"""Native AOT GLM 5.x norm and FFN-side programs.

``compile_glm_norm_aot(g)``: residual add + RMSNorm (``input_layernorm``,
``post_attention_layernorm``, the next layer's ``input_layernorm`` fused with
the previous residual add, and the final ``model.norm``)::

    residual   bf16 [rows,H]   inout  updated in place when deltas > 0
    delta0     bf16 [rows,H]   in     attention / FFN output (ignored when deltas == 0)
    delta1     bf16 [rows,H]   in     second FFN output (shared expert; deltas == 2 only)
    weight     bf16 [H]        in     the norm weight
    out        bf16 [rows,H]   out    w * bf16(residual * rsqrt(mean(residual^2) + eps))
    rows       int32
    deltas     int32   0: norm only; 1: residual = bf16(residual + delta0);
                       2: residual = bf16(residual + bf16(delta0 + delta1))

``compile_glm_ffn_aot(g, inter=I, max_rows=R)``: SwiGLU MLP (dense layers
``I = 12288``; the shared expert ``I = 2048``), no clamp::

    x          bf16 [rows,H]   in    post_attention_layernorm output
    w_gate_up  bf16 [2I,H]     in    cat(gate_proj, up_proj) (gate rows first)
    w_down     bf16 [H,I]      in    down_proj
    out        bf16 [rows,H]   out
    scratch    u8   ffn_scratch_bytes(I, rows): gate_up BF16 [rows,2I], hidden BF16 [rows,I]
    rows       int32

``hidden = bf16(bf16(silu(gate)) * up)``, the reference's rounding points.

``compile_glm_router_scores_aot(g)``: FP32 router logits ``x @ gate^T``
(``F.linear`` of FP32-promoted BF16 operands); the sigmoid, bias, top-8 and
x2.5 stay native::

    x          bf16 [rows,H]   in    post_attention_layernorm output
    w          bf16 [E,H]      in    mlp.gate.weight
    logits     f32  [rows,E]   out
    live_rows  int32

``compile_glm_expert_input_quant_aot(g)``: routed-expert BF16 rows -> FP8
K32 wire rows of ``H`` E4M3 bytes then ``H/32`` UE8M0 scales (row stride
6336 at H = 6144); the DeepSeek V4 ABI (``dsv4_ffn`` docstring): pointers
``source_ptr, values_ptr, scale_rows_ptr (= values_ptr + H), scale_mma_ptr``,
scalars ``m, grid_x``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import GLM53, AotProgram, GLMGeometry, Operand, Scalar, compile_program
from ._fp8_weights import FP8_GEMV_ROWS, Fp8Projection, check_w8_mode, fp8_only_operands, fp8_operands, projection, \
    w8_scalars
from ._glm_kernels import GlmAddRmsNorm, GlmSwiGLU

__all__ = [
    "compile_glm_expert_input_quant_aot",
    "compile_glm_ffn_aot",
    "compile_glm_norm_aot",
    "compile_glm_router_scores_aot",
    "ffn_scratch_bytes",
]

_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


def compile_glm_norm_aot(g: GLMGeometry = GLM53) -> AotProgram:
    """Residual add + RMSNorm at the model width; see the module docstring."""
    launch = GlmAddRmsNorm(g.hidden, g.norm_eps)
    h = g.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,{h}]", "inout"),
        Operand("delta0", torch.bfloat16, f"[rows,{h}]"),
        Operand("delta1", torch.bfloat16, f"[rows,{h}]"),
        Operand("weight", torch.bfloat16, f"[{h}]"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    return compile_program(
        launch, name="glm_norm", operands=operands, scalars=(Scalar("rows"), Scalar("deltas")),
        key=(h, g.norm_eps), geometry={"hidden": h, "eps": g.norm_eps}, doc=__doc__,
    )


def ffn_scratch_bytes(inter: int, rows: int) -> int:
    rows = max(int(rows), 1)
    return _align(rows * 2 * int(inter) * 2) + _align(rows * int(inter) * 2)


class _Ffn:
    def __init__(self, g: GLMGeometry, inter: int, fp8: bool = False):
        self.h, self.i = g.hidden, int(inter)
        self.gate_up = projection(2 * self.i, self.h, fp8)
        self.down = projection(self.h, self.i, fp8)
        self.swiglu = GlmSwiGLU(self.i)

    def key(self) -> tuple:
        return (self.h, self.i, self.gate_up.key(), self.down.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_down: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, w_gate_up, w_gate_up, w_gate_up, w_down, w_down, w_down, out, scratch, rows, stream)

    @cute.jit
    def body(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_gate_up_fp8: cute.Pointer,
             w_gate_up_scale: cute.Pointer, w_down: cute.Pointer, w_down_fp8: cute.Pointer,
             w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
             stream: cuda.CUstream):
        base = Int64(scratch.toint())
        gate_up = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        hidden = cute.make_ptr(cutlass.BFloat16, base + _align_i64(Int64(rows) * Int64(4 * self.i)),
                               cute.AddressSpace.gmem, assumed_align=16)
        self.gate_up(x, w_gate_up, w_gate_up_fp8, w_gate_up_scale, gate_up, rows, stream)
        self.swiglu(gate_up, hidden, rows, stream)
        self.down(hidden, w_down, w_down_fp8, w_down_scale, out, rows, stream)


class _FfnFp8(_Ffn):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_gate_up_fp8: cute.Pointer,
                 w_gate_up_scale: cute.Pointer, w_down: cute.Pointer, w_down_fp8: cute.Pointer,
                 w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, w_gate_up, w_gate_up_fp8, w_gate_up_scale, w_down, w_down_fp8, w_down_scale, out,
                  scratch, rows, stream)


class _FfnW8:
    """The SwiGLU MLP over FP8-only weights (``Fp8Projection``: decode GEMV / W8A16 TMA, or
    prefill W8A8 / W8A16 on the ``fp8_rows`` switch)."""

    def __init__(self, g: GLMGeometry, inter: int, prefill_rows: int | None):
        self.h, self.i = g.hidden, int(inter)
        self.prefill = prefill_rows is not None
        self.gate_up = Fp8Projection(2 * self.i, self.h, prefill_rows=prefill_rows)
        self.down = Fp8Projection(self.h, self.i, prefill_rows=prefill_rows)
        self.swiglu = GlmSwiGLU(self.i)

    def key(self) -> tuple:
        return ("w8", self.h, self.i, self.gate_up.key(), self.down.key())

    def scratch_bytes(self, rows: int) -> int:
        return ffn_scratch_bytes(self.i, rows) + self.gate_up.prefill_scratch_bytes(rows) \
            + self.down.prefill_scratch_bytes(rows)

    @cute.jit
    def body(self, x: cute.Pointer, w_gate_up_fp8: cute.Pointer, w_gate_up_scale: cute.Pointer,
             w_down_fp8: cute.Pointer, w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
             rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        gate_up = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        hidden_at = base + _align_i64(Int64(rows) * Int64(4 * self.i))
        hidden = cute.make_ptr(cutlass.BFloat16, hidden_at, cute.AddressSpace.gmem, assumed_align=16)
        qscratch = hidden_at + _align_i64(Int64(rows) * Int64(2 * self.i))
        self.gate_up(x, w_gate_up_fp8, w_gate_up_scale, gate_up, rows, fp8_rows, qscratch, stream)
        self.swiglu(gate_up, hidden, rows, stream)
        self.down(hidden, w_down_fp8, w_down_scale, out, rows, fp8_rows, qscratch, stream)


class _FfnW8Decode(_FfnW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up_fp8: cute.Pointer, w_gate_up_scale: cute.Pointer,
                 w_down_fp8: cute.Pointer, w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.body(x, w_gate_up_fp8, w_gate_up_scale, w_down_fp8, w_down_scale, out, scratch, rows,
                  Int32(FP8_GEMV_ROWS), stream)


class _FfnW8Prefill(_FfnW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up_fp8: cute.Pointer, w_gate_up_scale: cute.Pointer,
                 w_down_fp8: cute.Pointer, w_down_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(x, w_gate_up_fp8, w_gate_up_scale, w_down_fp8, w_down_scale, out, scratch, rows, fp8_rows,
                  stream)


def _compile_glm_ffn_w8(g: GLMGeometry, inter: int, max_rows: int, mode: str) -> AotProgram:
    prefill = mode == "prefill"
    launch = (_FfnW8Prefill if prefill else _FfnW8Decode)(g, inter, int(max_rows) if prefill else None)
    h, i = g.hidden, int(inter)
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        *fp8_only_operands("w_gate_up", 2 * i, h, prefill=prefill),
        *fp8_only_operands("w_down", h, i, prefill=prefill),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_ffn", operands=operands, scalars=w8_scalars(prefill),
        key=(int(max_rows), mode, launch.key()),
        geometry={"hidden": h, "inter": i, "max_rows": int(max_rows), "fp8_weights": "only", "mode": mode},
        scratch={"scratch": launch.scratch_bytes},
        doc=__doc__,
    )


def compile_glm_ffn_aot(g: GLMGeometry = GLM53, *, inter: int, max_rows: int, fp8: bool = False,
                        fp8_only: str | None = None) -> AotProgram:
    """SwiGLU MLP of intermediate ``inter`` for ``rows <= max_rows``; ``fp8``
    adds the decode FP8 weight operands. ``fp8_only`` (``"decode"`` or ``"prefill"``)
    takes the weights as E4M3 + FP32 128x128 scales only, no BF16 copies (see
    ``glm_attention``'s docstring)."""
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    if fp8_only is not None:
        return _compile_glm_ffn_w8(g, inter, max_rows, check_w8_mode(fp8_only))
    launch = (_FfnFp8 if fp8 else _Ffn)(g, inter, fp8)
    h, i = g.hidden, int(inter)
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{2 * i},{h}]"),
        *(fp8_operands("w_gate_up", 2 * i, h) if fp8 else ()),
        Operand("w_down", torch.bfloat16, f"[{h},{i}]"),
        *(fp8_operands("w_down", h, i) if fp8 else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_ffn", operands=operands, scalars=(Scalar("rows"),),
        key=(int(max_rows), fp8, launch.key()),
        geometry={"hidden": h, "inter": i, "max_rows": int(max_rows), "fp8_weights": fp8},
        scratch={"scratch": lambda rows: ffn_scratch_bytes(i, rows)},
        doc=__doc__,
    )


def compile_glm_router_scores_aot(g: GLMGeometry = GLM53) -> AotProgram:
    """FP32 router logits (skinny GEMV <= 32 rows, TMA warp-MMA above)."""
    from .dsv4_ffn import ROUTER_SKINNY_MAX_ROWS, _RouterScores

    e, h = g.routed_experts, g.hidden
    launch = _RouterScores(e, h)
    s = launch.skinny
    return compile_program(
        launch, name="glm_router_scores",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w", torch.bfloat16, f"[{e},{h}]", note="mlp.gate.weight"),
                  Operand("logits", torch.float32, f"[rows,{e}]", "out")),
        scalars=(Scalar("live_rows"),),
        key=(e, h, ROUTER_SKINNY_MAX_ROWS, s.cols, s.rows_per_tile, s.threads),
        geometry={"hidden": h, "experts": e, "skinny_max_rows": ROUTER_SKINNY_MAX_ROWS}, doc=__doc__,
    )


def compile_glm_expert_input_quant_aot(g: GLMGeometry = GLM53) -> AotProgram:
    """Routed-expert BF16 -> FP8 K32 wire rows at the GLM width."""
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    h = g.hidden
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="glm_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4},
        doc=__doc__,
    )
