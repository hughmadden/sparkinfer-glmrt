"""Native AOT GLM 5.3 Flash (``glm5_next``) coordinator programs, family ``glmf``.

Weights are BF16 (the EXL3 checkpoints' dense tensors, or FP8 blocks x FP32
scales dequantized at load); ``H`` hidden 4096, ``D`` KDA width 64 x 128,
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

from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import FLASH, GLM53_FLASH, AotProgram, GLMFGeometry, Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, GlmRankNormPackKV, GlmSwiGLU, glm_projection
from ._glmf_kernels import (
    GlmfAdd,
    GlmfKdaConv,
    GlmfKdaConvState,
    GlmfKdaGatedNorm,
    GlmfKdaRecurrent,
    GlmfMeanNorm,
)

__all__ = [
    "compile_glmf_add_aot",
    "compile_glmf_expert_input_quant_aot",
    "compile_glmf_ffn_aot",
    "compile_glmf_head_aot",
    "compile_glmf_kda_aot",
    "compile_glmf_mla_producer_aot",
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


def _ptr(dtype, address: Int64, align: int = 16):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


# ---------------------------------------------------------------------------
# KDA layer
# ---------------------------------------------------------------------------


def kda_scratch_bytes(g: GLMFGeometry, rows: int) -> int:
    """proj [rows,P], f|gate [rows,2,D], conv q|k|v [rows,3D], o [rows,D], y [rows,D] (BF16)."""
    rows = max(int(rows), 1)
    d = g.kda_width
    return (_align(rows * g.kda_in_width * 2) + _align(rows * 2 * d * 2) + _align(rows * 3 * d * 2)
            + 2 * _align(rows * d * 2))


class _Kda:
    def __init__(self, g: GLMFGeometry):
        self.g = g
        d, p = g.kda_width, g.kda_in_width
        self.d, self.p = d, p
        self.in_proj = glm_projection(p, g.hidden)
        # f_b(f_a) and g_b(g_a): two 128 -> D products off the in-projection row.
        self.fg = BatchedBf16Gemm(n=d, k=g.kda_head_dim, batch=2, a_row=p, a_batch=g.kda_head_dim,
                                  o_row=2 * d, o_batch=d)
        self.conv = GlmfKdaConv(channels=3 * d, proj_width=p)
        self.conv_state = GlmfKdaConvState(channels=3 * d, proj_width=p)
        self.recurrent = GlmfKdaRecurrent(heads=g.kda_heads, lower_bound=g.gate_lower_bound, qkv_width=3 * d,
                                          g_stride=2 * d, b_stride=p)
        self.norm = GlmfKdaGatedNorm(heads=g.kda_heads, eps=g.norm_eps, gate_stride=2 * d)
        self.o_proj = glm_projection(g.hidden, d)

    def key(self) -> tuple:
        return (self.in_proj.key(), self.fg.key(), self.o_proj.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, w_fg: cute.Pointer, conv_w: cute.Pointer,
                 a_log: cute.Pointer, dt_bias: cute.Pointer, o_norm: cute.Pointer, w_o: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        d, p = self.d, self.p
        m = Int64(rows)
        base = Int64(scratch.toint())
        fg_off = base + _align_i64(m * Int64(p * 2))
        qkv_off = fg_off + _align_i64(m * Int64(4 * d))
        o_off = qkv_off + _align_i64(m * Int64(6 * d))
        y_off = o_off + _align_i64(m * Int64(2 * d))
        bf16 = cutlass.BFloat16
        proj = _ptr(bf16, base)
        self.in_proj(x, w_in, proj, rows, stream)
        self.fg(_ptr(bf16, base + Int64(3 * d * 2)), w_fg, _ptr(bf16, fg_off), rows, stream)
        self.conv(proj, conv_w, conv_state, slots, seq_first, _ptr(bf16, qkv_off), rows, stream)
        self.conv_state(proj, conv_state, slots, seq_first, rows, stream)
        self.recurrent(_ptr(bf16, qkv_off), _ptr(bf16, fg_off), _ptr(bf16, base + Int64((3 * d + 256) * 2), 2),
                       a_log, dt_bias, state, slots, _ptr(bf16, o_off), rows, stream)
        self.norm(_ptr(bf16, o_off), _ptr(bf16, fg_off + Int64(d * 2)), o_norm, _ptr(bf16, y_off), rows, stream)
        self.o_proj(_ptr(bf16, y_off), w_o, out, rows, stream)


def compile_glmf_kda_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int) -> AotProgram:
    """One KDA layer for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _Kda(g)
    h, d, p, heads = g.hidden, g.kda_width, g.kda_in_width, g.kda_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_in", torch.bfloat16, f"[{p},{h}]"),
        Operand("w_fg", torch.bfloat16, f"[2,{d},{g.kda_head_dim}]"),
        Operand("conv_w", torch.float32, f"[{3 * d},4]", align=4),
        Operand("a_log", torch.float32, f"[{heads}]", align=4),
        Operand("dt_bias", torch.float32, f"[{d}]", align=4),
        Operand("o_norm", torch.bfloat16, f"[{g.kda_head_dim}]"),
        Operand("w_o", torch.bfloat16, f"[{h},{d}]"),
        Operand("conv_state", torch.bfloat16, f"[slots,3,{3 * d}]", "inout", align=2),
        Operand("state", torch.float32, f"[slots,{heads},128,128]", "inout"),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[kda_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_kda", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "heads": heads, "head_dim": g.kda_head_dim, "max_rows": max_rows,
                  "in_width": p, "lower_bound": g.gate_lower_bound, "eps": g.norm_eps},
        scratch={"scratch": lambda rows: kda_scratch_bytes(g, rows)},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# MLA producer (no RoPE)
# ---------------------------------------------------------------------------


def mla_producer_scratch_bytes(g: GLMFGeometry, rows: int) -> int:
    """qkv_a BF16 [rows, Q+512] then q_b BF16 [rows, N*256]."""
    rows = max(int(rows), 1)
    return _align(rows * g.qkv_a_width * 2) + _align(rows * g.heads * g.qk_head_dim * 2)


class _MlaProducer:
    def __init__(self, g: GLMFGeometry):
        self.g = g
        n, q = g.heads, g.q_lora_rank
        self.qkv_a = glm_projection(g.qkv_a_width, g.hidden)
        self.q_b = glm_projection(n * g.qk_head_dim, q)
        self.pack = GlmRankNormPackKV(q_rank=q, eps=g.norm_eps, page_rows=g.page_rows,
                                      record_bytes=g.record_bytes, rope=0)
        self.absorb = BatchedBf16Gemm(n=g.kv_lora_rank, k=g.qk_nope_dim, batch=n, a_row=n * g.qk_head_dim,
                                      a_batch=g.qk_head_dim, o_row=n * g.latent_dim, o_batch=g.latent_dim)

    def key(self) -> tuple:
        return (self.qkv_a.key(), self.q_b.key(), self.absorb.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b: cute.Pointer, w_uk: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = _ptr(cutlass.BFloat16, base)
        q = _ptr(cutlass.BFloat16, base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2)))
        self.qkv_a(x, w_qkv_a, qkv, rows, stream)
        # No RoPE: the pack kernel never reads positions or cos_sin.
        self.pack(qkv, q_a_norm, kv_a_norm, kv_slots, kv_slots, _ptr(cutlass.Float32, Int64(kv_slots.toint()), 4),
                  q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b, q, rows, stream)
        self.absorb(q, w_uk, query, rows, stream)


def compile_glmf_mla_producer_aot(g: GLMFGeometry = GLM53_FLASH, *, max_rows: int) -> AotProgram:
    """MLA producer for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _MlaProducer(g)
    h, q, n = g.hidden, g.q_lora_rank, g.heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{h}]"),
        Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
        Operand("kv_a_norm", torch.bfloat16, f"[{g.kv_lora_rank}]"),
        Operand("w_q_b", torch.bfloat16, f"[{n * g.qk_head_dim},{q}]"),
        Operand("w_uk", torch.bfloat16, f"[{n},{g.kv_lora_rank},{g.qk_nope_dim}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
        Operand("scratch", torch.uint8, "[mla_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_mla_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows,
                  "record_bytes": g.record_bytes, "page_rows": g.page_rows, "eps": g.norm_eps},
        scratch={"scratch": lambda rows: mla_producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


# ---------------------------------------------------------------------------
# FFN side, head
# ---------------------------------------------------------------------------


def compile_glmf_ffn_aot(g: GLMFGeometry = GLM53_FLASH, *, inter: int, max_rows: int) -> AotProgram:
    """Clamped SwiGLU MLP: the ``glm_ffn`` ABI (x, w_gate_up, w_down, out, scratch; rows)."""
    from .glm_ffn import _Ffn, ffn_scratch_bytes

    max_rows = _check_rows(max_rows)
    i = int(inter)
    launch = _Ffn(g, i)
    launch.swiglu = GlmSwiGLU(i, limit=g.swiglu_limit)
    h = g.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{2 * i},{h}]"),
        Operand("w_down", torch.bfloat16, f"[{h},{i}]"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glmf_ffn", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key(), g.swiglu_limit),
        geometry={"hidden": h, "inter": i, "max_rows": max_rows, "swiglu_limit": g.swiglu_limit},
        scratch={"scratch": lambda rows: ffn_scratch_bytes(i, rows)},
        doc=__doc__,
    )


def compile_glmf_router_scores_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """FP32 router logits ``x @ gate^T`` (BF16 operands, FP32 accumulation)."""
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

    e, h = g.routed_experts, g.hidden
    launch = RoutedBf16Projection(e, h, out_dtype=cutlass.Float32)
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
