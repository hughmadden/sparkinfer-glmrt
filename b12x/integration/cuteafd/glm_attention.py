"""Native AOT GLM 5.x attention producers and output projection.

Weights are BF16 (the checkpoint's FP8 128x128 blocks times their FP32
scales, dequantized at load); every projection is ``RoutedBf16Projection``
(skinny GEMV up to 8 live rows when N >= 2048, else up to its default; the
TMA tensor-core GEMM above; FP32 accumulation, BF16 out). ``H`` hidden 6144, ``N`` heads 64, ``Q`` q_lora
2048, ``D`` qk_nope 192, ``V`` v_head 256, ``I`` index heads 32; ``rows``
live tokens (``rows <= max_rows``); ``P`` positions in the RoPE table.
Scratch is laid out from the live row count (size it with ``rows =
max_rows``); every region starts 1024-byte aligned.

``compile_glm_producer_aot(g, max_rows=R)`` (MLA query/key producer)::

    x             bf16 [rows,H]           in   input_layernorm output
    positions     i64  [rows]             in   RoPE position per row
    kv_slots      i64  [rows]             in   latent cache slot page*64+row (<0 skips)
    cos_sin       f32  [P,64]             in   cos(32) | sin(32), theta**(-2i/64)
    w_qkv_a       bf16 [Q+576,H]          in   cat(q_a_proj, kv_a_proj_with_mqa)
    q_a_norm      bf16 [Q]                in   q_a_layernorm.weight
    kv_a_norm     bf16 [512]              in   kv_a_layernorm.weight
    w_q_b         bf16 [N*256,Q]          in   q_b_proj
    w_uk          bf16 [N,512,D]          in   kv_b_proj k rows, per head transposed:
                                               w_uk[h,c,d] = kv_b[h*448+d, c]
    kv_cache      u8   [pages,64*656]     inout  record at kv_slots[row]
    query         bf16 [rows,N,576]       out  [q_nope @ W_UK (512) | rope(q_rot) (64)]
    q_resid       bf16 [rows,Q]           out  q_a_layernorm(q_a(x)), for the indexer
    scratch       u8   producer_scratch_bytes(rows)
    rows          int32

Record (656 bytes, 64 per page, page base ``page*41984``): 512 E4M3 of
``kv_a_layernorm(latent)`` in four 128-groups, the four FP32 group scales
(``amax/448``), then 64 BF16 of the rotated key RoPE. This is the b12x
GLM_NSA sparse-MLA record (``pack_mla_kv_cache_reference``).

``compile_glm_index_producer_aot(g, max_rows=R)`` (DSA indexer, full layers)::

    x             bf16 [rows,H]           in   input_layernorm output
    q_resid       bf16 [rows,Q]           in   producer output
    positions     i64  [rows]             in
    index_slots   i64  [rows]             in   index cache slot page*64+row (<0 skips)
    cos_sin       f32  [P,64]             in
    w_iq          bf16 [I*128,Q]          in   indexer.wq_b
    w_ik          bf16 [128+I,H]          in   cat(indexer.wk, indexer.weights_proj)
    k_norm_w      bf16 [128]              in   indexer.k_norm.weight
    k_norm_b      bf16 [128]              in   indexer.k_norm.bias
    index_cache   u8   [pages,8448]       inout  64 x 128 E4M3 keys then 64 FP32 scales
    q_fp8         fp8  [rows,I,128]       out  rotated query / q_scale
    head_weights  f32  [rows,I]           out  bf16(weights_proj) * I^-0.5 * 128^-0.5 * q_scale
    scratch       u8   index_producer_scratch_bytes(rows)
    rows          int32

``compile_glm_o_aot(g, max_rows=R)`` (value up-projection, then o_proj)::

    attn          bf16 [rows,N,512]       in   sparse MLA output (latent per head)
    w_uv          bf16 [N,V,512]          in   kv_b_proj v rows per head: kv_b[h*448+192+v, c]
    w_o           bf16 [H,N*V]            in   o_proj
    out           bf16 [rows,H]           out  attention output (before the residual add)
    scratch       u8   o_scratch_bytes(rows)
    rows          int32
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import GLM53, GLMGeometry, Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, GlmIndexPost, GlmQueryRope, GlmRankNormPackKV, glm_projection

__all__ = [
    "compile_glm_index_producer_aot",
    "compile_glm_o_aot",
    "compile_glm_producer_aot",
    "index_producer_scratch_bytes",
    "o_scratch_bytes",
    "producer_scratch_bytes",
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


def producer_scratch_bytes(g: GLMGeometry, rows: int) -> int:
    """qkv_a BF16 [rows, Q+576] then q_b BF16 [rows, N*256]."""
    rows = max(int(rows), 1)
    return _align(rows * g.qkv_a_width * 2) + _align(rows * g.heads * g.qk_head_dim * 2)


def index_producer_scratch_bytes(g: GLMGeometry, rows: int) -> int:
    """Index query BF16 [rows, I*128] then [wk | weights_proj] BF16 [rows, 128+I]."""
    rows = max(int(rows), 1)
    return _align(rows * g.index_heads * 128 * 2) + _align(rows * (128 + g.index_heads) * 2)


def o_scratch_bytes(g: GLMGeometry, rows: int) -> int:
    """Per-head values BF16 [rows, N*V]."""
    return _align(max(int(rows), 1) * g.heads * g.v_head_dim * 2)


class _Producer:
    def __init__(self, g: GLMGeometry):
        self.g = g
        n, q = g.heads, g.q_lora_rank
        self.qkv_a = glm_projection(g.qkv_a_width, g.hidden)
        self.q_b = glm_projection(n * g.qk_head_dim, q)
        self.pack = GlmRankNormPackKV(q_rank=q, eps=g.norm_eps, page_rows=g.page_rows,
                                      record_bytes=g.record_bytes)
        # q_nope [rows, N, 192] (row stride N*256) @ W_UK^T -> query[:, :, :512].
        self.absorb = BatchedBf16Gemm(
            n=g.kv_lora_rank, k=g.qk_nope_dim, batch=n, a_row=n * g.qk_head_dim,
            a_batch=g.qk_head_dim, o_row=n * g.latent_dim, o_batch=g.latent_dim)
        self.rope = GlmQueryRope(heads=n, qk_nope=g.qk_nope_dim, qk_head=g.qk_head_dim,
                                 latent=g.latent_dim)

    def key(self) -> tuple:
        return (self.qkv_a.key(), self.q_b.key(), self.absorb.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer,
                 cos_sin: cute.Pointer, w_qkv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b: cute.Pointer, w_uk: cute.Pointer,
                 kv_cache: cute.Pointer, query: cute.Pointer, q_resid: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        q_off = base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2))
        q = cute.make_ptr(cutlass.BFloat16, q_off, cute.AddressSpace.gmem, assumed_align=16)
        self.qkv_a(x, w_qkv_a, qkv, rows, stream)
        self.pack(qkv, q_a_norm, kv_a_norm, positions, kv_slots, cos_sin, q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b, q, rows, stream)
        self.absorb(q, w_uk, query, rows, stream)
        self.rope(q, positions, cos_sin, query, rows, stream)


class _IndexProducer:
    def __init__(self, g: GLMGeometry):
        self.g = g
        i = g.index_heads
        self.wq = glm_projection(i * 128, g.q_lora_rank)
        self.wk = glm_projection(128 + i, g.hidden)
        self.post = GlmIndexPost(heads=i, eps=g.index_norm_eps,
                                 weight_scale=float(i) ** -0.5 * 128.0 ** -0.5, page_rows=g.page_rows)

    def key(self) -> tuple:
        return (self.wq.key(), self.wk.key(), self.g)

    @cute.jit
    def __call__(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer,
                 index_slots: cute.Pointer, cos_sin: cute.Pointer, w_iq: cute.Pointer,
                 w_ik: cute.Pointer, k_norm_w: cute.Pointer, k_norm_b: cute.Pointer,
                 index_cache: cute.Pointer, q_fp8: cute.Pointer, head_weights: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        iq = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        kw_off = base + _align_i64(Int64(rows) * Int64(g.index_heads * 128 * 2))
        kw = cute.make_ptr(cutlass.BFloat16, kw_off, cute.AddressSpace.gmem, assumed_align=16)
        self.wq(q_resid, w_iq, iq, rows, stream)
        self.wk(x, w_ik, kw, rows, stream)
        self.post(iq, kw, positions, index_slots, cos_sin, k_norm_w, k_norm_b, q_fp8, head_weights,
                  index_cache, rows, stream)


class _Output:
    def __init__(self, g: GLMGeometry):
        self.g = g
        n, v = g.heads, g.v_head_dim
        self.uv = BatchedBf16Gemm(n=v, k=g.kv_lora_rank, batch=n, a_row=n * g.kv_lora_rank,
                                  a_batch=g.kv_lora_rank, o_row=n * v, o_batch=v)
        self.o = glm_projection(g.hidden, n * v)

    def key(self) -> tuple:
        return (self.uv.key(), self.o.key(), self.g)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        values = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem,
                               assumed_align=16)
        self.uv(attn, w_uv, values, rows, stream)
        self.o(values, w_o, out, rows, stream)


def compile_glm_producer_aot(g: GLMGeometry = GLM53, *, max_rows: int):
    """MLA producer for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _Producer(g)
    h, q, n = g.hidden, g.q_lora_rank, g.heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{h}]"),
        Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("w_q_b", torch.bfloat16, f"[{n * g.qk_head_dim},{q}]"),
        Operand("w_uk", torch.bfloat16, f"[{n},512,{g.qk_nope_dim}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows,
                  "record_bytes": g.record_bytes, "page_rows": g.page_rows, "eps": g.norm_eps},
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


def compile_glm_index_producer_aot(g: GLMGeometry = GLM53, *, max_rows: int):
    """DSA index query/key producer for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _IndexProducer(g)
    h, q, i = g.hidden, g.q_lora_rank, g.index_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("index_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_iq", torch.bfloat16, f"[{i * 128},{q}]"),
        Operand("w_ik", torch.bfloat16, f"[{128 + i},{h}]"),
        Operand("k_norm_w", torch.bfloat16, "[128]"),
        Operand("k_norm_b", torch.bfloat16, "[128]"),
        Operand("index_cache", torch.uint8, f"[pages,{g.index_page_bytes}]", "inout"),
        Operand("q_fp8", torch.float8_e4m3fn, f"[rows,{i},128]", "out"),
        Operand("head_weights", torch.float32, f"[rows,{i}]", "out"),
        Operand("scratch", torch.uint8, "[index_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_index_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "index_heads": i, "max_rows": max_rows,
                  "eps": g.index_norm_eps},
        scratch={"scratch": lambda rows: index_producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


def compile_glm_o_aot(g: GLMGeometry = GLM53, *, max_rows: int):
    """W_UV per head then o_proj for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check_rows(max_rows)
    launch = _Output(g)
    h, n, v = g.hidden, g.heads, g.v_head_dim
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{n},512]"),
        Operand("w_uv", torch.bfloat16, f"[{n},{v},512]"),
        Operand("w_o", torch.bfloat16, f"[{h},{n * v}]"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_o", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, launch.key()),
        geometry={"hidden": h, "heads": n, "v_head_dim": v, "max_rows": max_rows},
        scratch={"scratch": lambda rows: o_scratch_bytes(g, rows)},
        doc=__doc__,
    )
