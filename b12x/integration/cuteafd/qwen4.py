"""Native AOT Qwen 3.8 Flash Next (``qwen4_exp``) coordinator programs, family ``qwen4``.

Weights are BF16 (every Qwen checkpoint stores the coordinator tensors in
BF16); ``H`` hidden 2560, ``C = 4H`` the four hyper-connection streams of a
row, ``R`` the hyper-connection rank 320. ``rows`` is the live row count;
scratch regions are laid out from it (size with the capacity), each
1024-byte aligned. Arithmetic and rounding: see ``_qwen4_kernels``.

Hyper-connections (``Qwen4ExpTextGatedResidual``; streams BF16 ``[rows,4,H]``)::

    qwen4_hc_pre       residual [rows,4,H] in, norm bf16 [C] (hc_norm), w_di bf16 [R+4,C]
                       (cat(input_mix_weight_down, block_inject_weight)), w_up bf16 [C,R]
                       (input_mix_weight_up), y bf16 [rows,H] out (sublayer input),
                       inject bf16 [rows,4] out, scratch
    qwen4_hc_post_pre  x bf16 [rows,H] (sublayer output), residual [rows,4,H], inject bf16
                       [rows,4] inout (the finished site's weights in, the next site's out),
                       norm, w_di, w_up (next site), residual_out [rows,4,H] out, y out, scratch
    qwen4_hc_post      x, residual, inject, out [rows,4,H] (the last sublayer)
    qwen4_head         streams [rows,4,H], norm, w_down bf16 [R,C], w_up (hyper_connection_mixer),
                       out bf16 [rows,H] (lm_head input; this model has no final norm), scratch

MoE front: ``qwen4_router_scores`` (FP32 logits ``x @ gate^T``; the native
softmax top-10 rounds them to BF16 first, as the reference's BF16 linear),
``qwen4_shared`` (shared expert with its sigmoid gate: ``w_gate_up`` bf16
[1296,H] = cat(gate_proj, up_proj, shared_expert_gate, 15 zero rows),
``w_down`` bf16 [H,640], out = bf16(bf16(sigmoid(g)) * down(swiglu))),
``qwen4_expert_input_quant`` (FP8 K32 wire rows), ``qwen4_add``.

PLE (layer 1, ``Qwen4ExpTextPLELayer``) ``qwen4_ple_{bf16,fp8}``::

    streams    bf16 [rows,4,H]           inout  += the PLE output
    ids        i64  [rows,16]            in     table row of each n-gram head (host hashed)
    table      bf16|fp8 [T,160]          in     the n-gram table (device or host-mapped)
    scale      f32  [1]                  in     FP8 table scale (ignored for BF16)
    w_kv       bf16 [C+H,H]              in     cat(key_proj, value_proj)
    norm_key, norm_query, norm_conv bf16 [C]
    conv_w     f32  [C,4]                in     ple.conv1d.weight (dilation 3)
    conv_state bf16 [slots,9,C]          inout  the sequence's last 9 norm_conv(gv) rows
    slots, seq_first i32 [rows]          in     as the GDN program
    scratch
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import QWEN38_FLASH_NEXT, AotProgram, Operand, Qwen4Geometry, Scalar, compile_program
from ._qwen4_kernels import (
    Qwen4Add,
    Qwen4HcGate,
    Qwen4HcMix,
    Qwen4HcNorm,
    Qwen4HcPost,
    Qwen4PleConv,
    Qwen4PleConvState,
    Qwen4PleGate,
    Qwen4PleGather,
    Qwen4SharedGate,
    Qwen4SwiGLU,
)

__all__ = [
    "compile_qwen4_add_aot",
    "compile_qwen4_expert_input_quant_aot",
    "compile_qwen4_hc_post_aot",
    "compile_qwen4_hc_post_pre_aot",
    "compile_qwen4_hc_pre_aot",
    "compile_qwen4_head_aot",
    "compile_qwen4_ple_aot",
    "compile_qwen4_router_scores_aot",
    "compile_qwen4_shared_aot",
    "hc_scratch_bytes",
    "ple_scratch_bytes",
    "shared_scratch_bytes",
]

_ALIGN = 1024
SHARED_ROWS = 1296


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


def _ptr(dtype, address: Int64, align: int = 16):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _projection(n: int, k: int, out_dtype=cutlass.BFloat16):
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection, TmaBf16Projection

    # K = R (320) has no skinny GEMV split (K/8 = 40 threads); the TMA route serves every row count.
    if (k // 8) % 32:
        return TmaBf16Projection(n, k, out_dtype=out_dtype)
    return RoutedBf16Projection(n, k, out_dtype=out_dtype)


def _key(projection) -> tuple:
    return projection.key() if hasattr(projection, "key") else (projection.n, projection.k,
                                                                tuple(sorted(projection.config.items())))


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------


def hc_scratch_bytes(g: Qwen4Geometry, rows: int, inject: bool = True) -> int:
    """normed [rows,C], d [rows,R(+4)], a [rows,R], u [rows,C] (BF16)."""
    rows = max(int(rows), 1)
    c, r = g.hc_width, g.hc_lowrank
    return (_align(rows * c * 2) + _align(rows * (r + (g.hc_count if inject else 0)) * 2)
            + _align(rows * r * 2) + _align(rows * c * 2))


class _Hc:
    """Norm (optionally after the previous site's post), low-rank mix, and the injection weights."""

    def __init__(self, g: Qwen4Geometry, *, post: bool, inject: bool):
        self.g, self.post, self.inject = g, post, inject
        c, r = g.hc_width, g.hc_lowrank
        self.d_width = r + (g.hc_count if inject else 0)
        self.norm = Qwen4HcNorm(g.hidden, g.hc_count, g.norm_eps, post)
        self.di = _projection(self.d_width, c)
        self.gate = Qwen4HcGate(r, g.hc_count, inject)
        self.up = _projection(c, r)
        self.mix = Qwen4HcMix(g.hidden, g.hc_count)

    def key(self) -> tuple:
        return (self.g, self.post, self.inject, _key(self.di), _key(self.up))

    @cute.jit
    def body(self, delta: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, norm: cute.Pointer,
             w_di: cute.Pointer, w_up: cute.Pointer, residual_out: cute.Pointer, y: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        m = Int64(rows)
        base = Int64(scratch.toint())
        d_off = base + _align_i64(m * Int64(g.hc_width * 2))
        a_off = d_off + _align_i64(m * Int64(self.d_width * 2))
        u_off = a_off + _align_i64(m * Int64(g.hc_lowrank * 2))
        bf16 = cutlass.BFloat16
        normed = _ptr(bf16, base)
        self.norm(delta, residual, inject, norm, residual_out, normed, rows, stream)
        self.di(normed, w_di, _ptr(bf16, d_off), rows, stream)
        self.gate(_ptr(bf16, d_off), _ptr(bf16, a_off), inject, rows, stream)
        self.up(_ptr(bf16, a_off), w_up, _ptr(bf16, u_off), rows, stream)
        self.mix(_ptr(bf16, u_off), normed, y, rows, stream)


class _HcPre(_Hc):
    def __init__(self, g: Qwen4Geometry):
        super().__init__(g, post=False, inject=True)

    @cute.jit
    def __call__(self, residual: cute.Pointer, norm: cute.Pointer, w_di: cute.Pointer, w_up: cute.Pointer,
                 y: cute.Pointer, inject: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(residual, residual, inject, norm, w_di, w_up, residual, y, scratch, rows, stream)


class _HcPostPre(_Hc):
    def __init__(self, g: Qwen4Geometry):
        super().__init__(g, post=True, inject=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, norm: cute.Pointer,
                 w_di: cute.Pointer, w_up: cute.Pointer, residual_out: cute.Pointer, y: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, residual, inject, norm, w_di, w_up, residual_out, y, scratch, rows, stream)


class _Head(_Hc):
    def __init__(self, g: Qwen4Geometry):
        super().__init__(g, post=False, inject=False)

    @cute.jit
    def __call__(self, streams: cute.Pointer, norm: cute.Pointer, w_down: cute.Pointer, w_up: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(streams, streams, streams, norm, w_down, w_up, streams, out, scratch, rows, stream)


def _hc_operands(g: Qwen4Geometry, inject: bool = True) -> dict[str, Operand]:
    c, r, h = g.hc_width, g.hc_lowrank, g.hidden
    di = r + g.hc_count if inject else r
    return {
        "norm": Operand("norm", torch.bfloat16, f"[{c}]", note="hc_norm.weight"),
        "w_di": Operand("w_di", torch.bfloat16, f"[{di},{c}]",
                        note="cat(input_mix_weight_down, block_inject_weight)" if inject else "input_mix_weight_down"),
        "w_up": Operand("w_up", torch.bfloat16, f"[{c},{r}]", note="input_mix_weight_up"),
        "streams": Operand("residual", torch.bfloat16, f"[rows,{g.hc_count},{h}]"),
        "y": Operand("y", torch.bfloat16, f"[rows,{h}]", "out"),
        "scratch": Operand("scratch", torch.uint8, "[hc_scratch_bytes]", "scratch"),
    }


def _hc_geometry(g: Qwen4Geometry) -> dict:
    return {"hidden": g.hidden, "streams": g.hc_count, "lowrank": g.hc_lowrank, "eps": g.norm_eps}


def compile_qwen4_hc_pre_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """A site's hyper-connection input: see the module docstring."""
    launch = _HcPre(g)
    o = _hc_operands(g)
    operands = (o["streams"], o["norm"], Operand("w_di", torch.bfloat16, o["w_di"].shape, note=o["w_di"].note),
                o["w_up"], o["y"], Operand("inject", torch.bfloat16, f"[rows,{g.hc_count}]", "out", align=2),
                o["scratch"])
    return compile_program(launch, name="qwen4_hc_pre", operands=operands, scalars=(Scalar("rows"),),
                           key=launch.key(), geometry=_hc_geometry(g),
                           scratch={"scratch": lambda rows: hc_scratch_bytes(g, rows)}, doc=__doc__)


def compile_qwen4_hc_post_pre_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """A sublayer output into the streams, then the next site's input: see the module docstring."""
    launch = _HcPostPre(g)
    o = _hc_operands(g)
    h = g.hidden
    operands = (Operand("x", torch.bfloat16, f"[rows,{h}]"), o["streams"],
                Operand("inject", torch.bfloat16, f"[rows,{g.hc_count}]", "inout", align=2),
                o["norm"], o["w_di"], o["w_up"],
                Operand("residual_out", torch.bfloat16, f"[rows,{g.hc_count},{h}]", "out"), o["y"], o["scratch"])
    return compile_program(launch, name="qwen4_hc_post_pre", operands=operands, scalars=(Scalar("rows"),),
                           key=launch.key(), geometry=_hc_geometry(g),
                           scratch={"scratch": lambda rows: hc_scratch_bytes(g, rows)}, doc=__doc__)


def compile_qwen4_hc_post_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """``out_s = bf16(residual_s + bf16(x * inject_s))``."""
    h = g.hidden
    launch = Qwen4HcPost(h, g.hc_count)
    operands = (Operand("x", torch.bfloat16, f"[rows,{h}]"),
                Operand("residual", torch.bfloat16, f"[rows,{g.hc_count},{h}]"),
                Operand("inject", torch.bfloat16, f"[rows,{g.hc_count}]", align=2),
                Operand("out", torch.bfloat16, f"[rows,{g.hc_count},{h}]", "out"))
    return compile_program(launch, name="qwen4_hc_post", operands=operands, scalars=(Scalar("rows"),),
                           key=(h, g.hc_count), geometry=_hc_geometry(g), doc=__doc__)


def compile_qwen4_head_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """The final stream mixer (no injection): the lm_head input."""
    launch = _Head(g)
    o = _hc_operands(g, inject=False)
    operands = (Operand("streams", torch.bfloat16, f"[rows,{g.hc_count},{g.hidden}]"), o["norm"],
                Operand("w_down", torch.bfloat16, o["w_di"].shape, note="input_mix_weight_down"), o["w_up"],
                Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"), o["scratch"])
    return compile_program(launch, name="qwen4_head", operands=operands, scalars=(Scalar("rows"),),
                           key=launch.key(), geometry=_hc_geometry(g),
                           scratch={"scratch": lambda rows: hc_scratch_bytes(g, rows, inject=False)}, doc=__doc__)


# ---------------------------------------------------------------------------
# MoE front
# ---------------------------------------------------------------------------


def compile_qwen4_router_scores_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """FP32 router logits ``x @ gate^T`` (BF16 operands, FP32 accumulation)."""
    from .glmf import _RouterScores

    e, h = g.routed_experts, g.hidden
    launch = _RouterScores(e, h)
    return compile_program(
        launch, name="qwen4_router_scores",
        operands=(Operand("x", torch.bfloat16, f"[rows,{h}]"),
                  Operand("w", torch.bfloat16, f"[{e},{h}]", note="mlp.gate.weight"),
                  Operand("logits", torch.float32, f"[rows,{e}]", "out")),
        scalars=(Scalar("rows"),), key=(launch.key(),),
        geometry={"hidden": h, "experts": e, "top_k": g.top_k}, doc=__doc__,
    )


def shared_scratch_bytes(g: Qwen4Geometry, rows: int) -> int:
    """gate|up|gate-logit [rows,1296], hidden [rows,I], down [rows,H] (BF16)."""
    rows = max(int(rows), 1)
    return _align(rows * SHARED_ROWS * 2) + _align(rows * g.shared_inter * 2) + _align(rows * g.hidden * 2)


class _Shared:
    def __init__(self, g: Qwen4Geometry):
        self.g = g
        i = g.shared_inter
        if SHARED_ROWS < 2 * i + 1:
            raise ValueError("shared expert rows do not fit the gate/up/gate block")
        self.gate_up = _projection(SHARED_ROWS, g.hidden)
        self.swiglu = Qwen4SwiGLU(i, SHARED_ROWS)
        self.down = _projection(g.hidden, i)
        self.scale = Qwen4SharedGate(g.hidden, SHARED_ROWS, 2 * i)

    def key(self) -> tuple:
        return (self.g, _key(self.gate_up), _key(self.down))

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_down: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        m = Int64(rows)
        base = Int64(scratch.toint())
        hid = base + _align_i64(m * Int64(SHARED_ROWS * 2))
        down = hid + _align_i64(m * Int64(g.shared_inter * 2))
        bf16 = cutlass.BFloat16
        self.gate_up(x, w_gate_up, _ptr(bf16, base), rows, stream)
        self.swiglu(_ptr(bf16, base), _ptr(bf16, hid), rows, stream)
        self.down(_ptr(bf16, hid), w_down, _ptr(bf16, down), rows, stream)
        self.scale(_ptr(bf16, base), _ptr(bf16, down), out, rows, stream)


def compile_qwen4_shared_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """Shared expert times its sigmoid gate (see the module docstring)."""
    launch = _Shared(g)
    h, i = g.hidden, g.shared_inter
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{SHARED_ROWS},{h}]",
                note="cat(shared_expert.gate_proj, shared_expert.up_proj, shared_expert_gate, zeros)"),
        Operand("w_down", torch.bfloat16, f"[{h},{i}]"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[shared_scratch_bytes]", "scratch"),
    )
    return compile_program(launch, name="qwen4_shared", operands=operands, scalars=(Scalar("rows"),),
                           key=launch.key(), geometry={"hidden": h, "inter": i, "rows": SHARED_ROWS},
                           scratch={"scratch": lambda rows: shared_scratch_bytes(g, rows)}, doc=__doc__)


def compile_qwen4_expert_input_quant_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """Routed-expert BF16 rows -> FP8 K32 wire rows (H E4M3 bytes then H/32 UE8M0)."""
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot

    h = g.hidden
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="qwen4_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4},
        doc=__doc__,
    )


def compile_qwen4_add_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT) -> AotProgram:
    """``out = bf16(a + b)`` over ``[rows, H]`` (routed + shared expert outputs)."""
    h = g.hidden
    return compile_program(
        Qwen4Add(h), name="qwen4_add",
        operands=(Operand("a", torch.bfloat16, f"[rows,{h}]"), Operand("b", torch.bfloat16, f"[rows,{h}]"),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"),), key=(h,), geometry={"hidden": h}, doc=__doc__,
    )


# ---------------------------------------------------------------------------
# PLE
# ---------------------------------------------------------------------------


def ple_scratch_bytes(g: Qwen4Geometry, rows: int) -> int:
    """emb [rows,H], kv [rows,C+H], gv [rows,C], gvn [rows,C] (BF16)."""
    rows = max(int(rows), 1)
    c, h = g.hc_width, g.hidden
    return _align(rows * g.ple_dim * 2) + _align(rows * (c + h) * 2) + 2 * _align(rows * c * 2)


class _Ple:
    def __init__(self, g: Qwen4Geometry, fp8: bool):
        self.g, self.fp8 = g, fp8
        c, h = g.hc_width, g.hidden
        if g.ple_dim != g.ple_rows * g.ple_row_dim:
            raise ValueError("PLE rows do not tile the PLE width")
        self.gather = Qwen4PleGather(g.ple_rows, g.ple_row_dim, fp8)
        self.kv = _projection(c + h, g.ple_dim)
        self.gate = Qwen4PleGate(h, g.hc_count, g.norm_eps)
        self.conv = Qwen4PleConv(c, g.ple_conv, g.ngram_size)
        self.state = Qwen4PleConvState(c, g.ple_state_rows)

    def key(self) -> tuple:
        return (self.g, self.fp8, _key(self.kv))

    @cute.jit
    def __call__(self, streams: cute.Pointer, ids: cute.Pointer, table: cute.Pointer, scale: cute.Pointer,
                 w_kv: cute.Pointer, norm_key: cute.Pointer, norm_query: cute.Pointer, norm_conv: cute.Pointer,
                 conv_w: cute.Pointer, conv_state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        m = Int64(rows)
        c, h = g.hc_width, g.hidden
        base = Int64(scratch.toint())
        kv_off = base + _align_i64(m * Int64(g.ple_dim * 2))
        gv_off = kv_off + _align_i64(m * Int64((c + h) * 2))
        gvn_off = gv_off + _align_i64(m * Int64(c * 2))
        bf16 = cutlass.BFloat16
        self.gather(ids, table, scale, _ptr(bf16, base), rows, stream)
        self.kv(_ptr(bf16, base), w_kv, _ptr(bf16, kv_off), rows, stream)
        self.gate(_ptr(bf16, kv_off), streams, norm_key, norm_query, norm_conv, _ptr(bf16, gv_off),
                  _ptr(bf16, gvn_off), rows, stream)
        self.conv(_ptr(bf16, gv_off), _ptr(bf16, gvn_off), conv_w, conv_state, slots, seq_first, streams, rows,
                  stream)
        self.state(_ptr(bf16, gvn_off), conv_state, slots, seq_first, rows, stream)


def compile_qwen4_ple_aot(g: Qwen4Geometry = QWEN38_FLASH_NEXT, *, fp8: bool) -> AotProgram:
    """The PLE layer (see the module docstring); ``fp8`` selects an E4M3 table."""
    launch = _Ple(g, fp8)
    c, h = g.hc_width, g.hidden
    table = torch.float8_e4m3fn if fp8 else torch.bfloat16
    operands = (
        Operand("streams", torch.bfloat16, f"[rows,{g.hc_count},{h}]", "inout"),
        Operand("ids", torch.int64, f"[rows,{g.ple_rows}]", align=8),
        Operand("table", table, f"[table_rows,{g.ple_row_dim}]", note="ngram_embedding shards, one array"),
        Operand("scale", torch.float32, "[1]", align=4, note="ngram_embedding.weight_scale (FP8)"),
        Operand("w_kv", torch.bfloat16, f"[{c + h},{g.ple_dim}]", note="cat(ple.key_proj, ple.value_proj)"),
        Operand("norm_key", torch.bfloat16, f"[{c}]"),
        Operand("norm_query", torch.bfloat16, f"[{c}]"),
        Operand("norm_conv", torch.bfloat16, f"[{c}]"),
        Operand("conv_w", torch.float32, f"[{c},{g.ple_conv}]", align=4),
        Operand("conv_state", torch.bfloat16, f"[slots,{g.ple_state_rows},{c}]", "inout", align=2),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("scratch", torch.uint8, "[ple_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"qwen4_ple_{'fp8' if fp8 else 'bf16'}", operands=operands, scalars=(Scalar("rows"),),
        key=launch.key(),
        geometry={"hidden": h, "streams": g.hc_count, "rows_per_token": g.ple_rows, "row_dim": g.ple_row_dim,
                  "taps": g.ple_conv, "dilation": g.ngram_size, "state_rows": g.ple_state_rows, "fp8": fp8},
        scratch={"scratch": lambda rows: ple_scratch_bytes(g, rows)}, doc=__doc__,
    )
