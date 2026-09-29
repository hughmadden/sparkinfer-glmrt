"""Native AOT MiMo V2 norm and FFN-side programs (``H`` = 4096 Flash, 6144 V2.6 Pro).

``compile_mimo_norm_aot(g)``: residual add + RMSNorm (eps 1e-5), the
``glm_norm`` kernel and ABI at the MiMo width::

    residual   bf16 [rows,H]   inout  updated in place when deltas > 0
    delta0     bf16 [rows,H]   in     attention / FFN output (ignored when deltas == 0)
    delta1     bf16 [rows,H]   in     second delta (deltas == 2 only)
    weight     bf16 [H]        in     the norm weight
    out        bf16 [rows,H]   out    w * bf16(residual * rsqrt(mean(residual^2) + eps))
    rows       int32
    deltas     int32   0: norm only; 1: residual = bf16(residual + delta0);
                       2: residual = bf16(residual + bf16(delta0 + delta1))

``compile_mimo_ffn_aot(g, max_rows=R)``: the SwiGLU dense MLP of layer 0 (and
the MTP layers), ``I = 16384``, no clamp; the ``glm_ffn`` ABI::

    x          bf16 [rows,H]   in    post_attention_layernorm output
    w_gate_up  bf16 [2I,H]     in    cat(gate_proj, up_proj) (gate rows first)
    w_down     bf16 [H,I]      in    down_proj
    out        bf16 [rows,H]   out
    scratch    u8   ffn_scratch_bytes(I, rows): gate_up BF16 [rows,2I], hidden BF16 [rows,I]
    rows       int32

``fp8=True`` (decode steps): ``w_gate_up_fp8``/``w_gate_up_scale`` and
``w_down_fp8``/``w_down_scale`` (E4M3, FP32 ``[N, K/128]`` per-row scales, the
checkpoint's 128x128 grid expanded) follow their BF16 weights and the scalar
``fp8_rows`` follows ``rows`` (see ``mimo_attention``).

``compile_mimo_router_scores_aot(g)``: router logits ``x @ gate^T`` against
the checkpoint's FP32 gate weight, as ``F.linear`` of FP32-promoted operands.
The weight is split at load into BF16 high and low parts (``w_hi = bf16(w)``,
``w_lo = bf16(w - w_hi)``, 16 significant bits) and both are accumulated in
FP32; the sigmoid, bias, top-8 and normalization stay native
(``cuteafd_router_select``, sigmoid, no routed scale)::

    x          bf16 [rows,H]    in    post_attention_layernorm output
    w_hilo     bf16 [2E,H]      in    cat(w_hi, w_lo)
    logits     f32  [rows,E]    out
    scratch    u8   [rows, 2E] FP32 (router_scratch_bytes)
    rows       int32

``compile_mimo_expert_input_quant_aot(g)``: routed-expert BF16 rows -> FP8
K32 wire rows (``H`` E4M3 bytes then ``H/32`` UE8M0 scales, row stride 4224);
the DeepSeek V4 ABI (``dsv4_ffn``): ``source_ptr, values_ptr, scale_rows_ptr
(= values_ptr + H), scale_mma_ptr``, scalars ``m, grid_x``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import MIMO_V2_FLASH, AotProgram, MiMoGeometry, Operand, Scalar, compile_program
from ._glm_kernels import GlmAddRmsNorm
from ._mimo_kernels import RouterHiLoAdd
from .glm_ffn import _Ffn, ffn_scratch_bytes
from .glmf import _GlmfFfnFp8, _Fp8Switch, fp8_ops

__all__ = [
    "compile_mimo_expert_input_quant_aot",
    "compile_mimo_ffn_aot",
    "compile_mimo_norm_aot",
    "compile_mimo_router_scores_aot",
    "ffn_scratch_bytes",
    "router_scratch_bytes",
    "split_router_weight",
]


def compile_mimo_norm_aot(g: MiMoGeometry = MIMO_V2_FLASH) -> AotProgram:
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
        launch, name="mimo_norm", operands=operands, scalars=(Scalar("rows"), Scalar("deltas")),
        key=(h, g.norm_eps), geometry={"hidden": h, "eps": g.norm_eps}, doc=__doc__,
    )


class _FfnFp8(_GlmfFfnFp8):
    """The unclamped SwiGLU MLP over per-row-scaled E4M3 copies for decode rows."""

    def __init__(self, g: MiMoGeometry, inter: int):
        from ._glm_kernels import GlmSwiGLU

        self.h, self.i = g.hidden, int(inter)
        self.gate_up = _Fp8Switch(2 * self.i, self.h, fp8=True, row_scales=True)
        self.down = _Fp8Switch(self.h, self.i, fp8=True, row_scales=True)
        self.swiglu = GlmSwiGLU(self.i)


def compile_mimo_ffn_aot(g: MiMoGeometry = MIMO_V2_FLASH, *, max_rows: int, inter: int | None = None,
                         fp8: bool = False) -> AotProgram:
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    i = int(g.dense_inter if inter is None else inter)
    launch = _FfnFp8(g, i) if fp8 else _Ffn(g, i)
    h = g.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{2 * i},{h}]"),
        *(fp8_ops("w_gate_up", 2 * i, h, True) if fp8 else ()),
        Operand("w_down", torch.bfloat16, f"[{h},{i}]"),
        *(fp8_ops("w_down", h, i, True) if fp8 else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="mimo_ffn", operands=operands,
        scalars=(Scalar("rows"), Scalar("fp8_rows")) if fp8 else (Scalar("rows"),),
        key=(int(max_rows), fp8, launch.key()),
        geometry={"hidden": h, "inter": i, "max_rows": int(max_rows), "fp8_weights": fp8},
        scratch={"scratch": lambda rows: ffn_scratch_bytes(i, rows)},
        doc=__doc__,
    )


def router_scratch_bytes(g: MiMoGeometry, rows: int) -> int:
    return (max(int(rows), 1) * 2 * g.routed_experts * 4 + 1023) // 1024 * 1024


def split_router_weight(w: torch.Tensor) -> torch.Tensor:
    """FP32 ``[E, H]`` -> BF16 ``[2E, H]`` = ``cat(bf16(w), bf16(w - bf16(w)))``."""
    hi = w.float().bfloat16()
    lo = (w.float() - hi.float()).bfloat16()
    return torch.cat([hi, lo], 0).contiguous()


class _RouterHiLo:
    def __init__(self, g: MiMoGeometry):
        from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

        self.e = g.routed_experts
        self.proj = RoutedBf16Projection(2 * self.e, g.hidden, out_dtype=cutlass.Float32)
        self.add = RouterHiLoAdd(self.e)

    def key(self) -> tuple:
        return (self.proj.key(), self.e)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_hilo: cute.Pointer, logits: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        hilo = cute.make_ptr(cutlass.Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.proj(x, w_hilo, hilo, rows, stream)
        self.add(hilo, logits, rows, stream)


class _RouterBf16:
    """BF16 router weight (V2.6 Pro): one BF16 product, FP32 accumulation."""

    def __init__(self, g: MiMoGeometry):
        from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

        self.proj = RoutedBf16Projection(g.routed_experts, g.hidden, out_dtype=cutlass.Float32)

    def key(self) -> tuple:
        return ("bf16", self.proj.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_router: cute.Pointer, logits: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.proj(x, w_router, logits, rows, stream)


def compile_mimo_router_scores_aot(g: MiMoGeometry = MIMO_V2_FLASH) -> AotProgram:
    """FP32 router weight: ``w_hilo`` (BF16 hi + lo); BF16 router weight
    (``g.router_fp32`` False): ``w_router`` BF16 ``[E, H]`` as stored."""
    e, h = g.routed_experts, g.hidden
    if g.router_fp32:
        launch = _RouterHiLo(g)
        weight = Operand("w_hilo", torch.bfloat16, f"[{2 * e},{h}]", note="split mlp.gate.weight (FP32)")
    else:
        launch = _RouterBf16(g)
        weight = Operand("w_router", torch.bfloat16, f"[{e},{h}]", note="mlp.gate.weight (BF16)")
    return compile_program(
        launch, name="mimo_router_scores",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"), weight,
                  Operand("logits", torch.float32, f"[rows,{e}]", "out"),
                  Operand("scratch", torch.uint8, "[router_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(launch.key(),),
        geometry={"hidden": h, "experts": e, "top_k": g.top_k, "router_fp32": g.router_fp32},
        scratch={"scratch": lambda rows: router_scratch_bytes(g, rows)}, doc=__doc__,
    )


def compile_mimo_expert_input_quant_aot(g: MiMoGeometry = MIMO_V2_FLASH) -> AotProgram:
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    h = g.hidden
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="mimo_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4},
        doc=__doc__,
    )
