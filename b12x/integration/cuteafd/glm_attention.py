"""Native AOT GLM 5.x attention producers and output projection.

Weights are BF16 (the checkpoint's FP8 128x128 blocks times their FP32
scales, dequantized at load); every projection is ``RoutedBf16Projection``
(skinny GEMV up to 8 live rows when N >= 2048, else up to its default; the
TMA tensor-core GEMM above; FP32 accumulation, BF16 out).

``fp8_only="decode"`` / ``"prefill"`` (the exported programs): every FP8
checkpoint weight is passed only as ``{w}_fp8`` E4M3 plus ``{w}_scale`` (its
128x128 grid) in place of the BF16 operand, and ``w_uk`` / ``w_uv`` as
``{w}_fp8`` per-head E4M3 slices of ``kv_b_proj`` with ``{w}_scale`` one FP32
scale per weight row and 64-wide K tile (``[N,512,3]``, ``[N,256,8]``). Decode
programs run the FP8 GEMV up to 16 rows and W8A16 GEMMs above (bitwise the
BF16 programs over the dequantized weights); prefill programs take a second
scalar ``fp8_rows`` (nonzero: W8A8, E4M3 activations per row and 128-K block;
0: W8A16) and read ``q_a|kv_a`` with per-row scales K-block major
(``w_qkv_a_kscale [48, 2624]``). ``w_ik`` stays BF16 (``wk`` FP8 and
``weights_proj`` BF16 in the release). ``H`` hidden 6144, ``N`` heads 64, ``Q`` q_lora
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
from ._fp8_weights import FP8_GEMV_ROWS, Fp8Projection, check_w8_mode, fp8_only_operands, fp8_operands, projection, \
    w8_scalars
from ._glm_kernels import (BatchedBf16Gemm, BatchedFp8Gemm, GlmIndexPost, GlmQueryRope, GlmRankNormPackKV,
                           glm_projection)

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
    def __init__(self, g: GLMGeometry, fp8: bool = False):
        self.g = g
        n, q = g.heads, g.q_lora_rank
        self.qkv_a = projection(g.qkv_a_width, g.hidden, fp8)
        self.q_b = projection(n * g.qk_head_dim, q, fp8)
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
        self.body(x, positions, kv_slots, cos_sin, w_qkv_a, w_qkv_a, w_qkv_a, q_a_norm, kv_a_norm,
                  w_q_b, w_q_b, w_q_b, w_uk, kv_cache, query, q_resid, scratch, rows, stream)

    @cute.jit
    def body(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer,
             cos_sin: cute.Pointer, w_qkv_a: cute.Pointer, w_qkv_a_fp8: cute.Pointer,
             w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer,
             w_q_b: cute.Pointer, w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer,
             w_uk: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer, q_resid: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        q_off = base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2))
        q = cute.make_ptr(cutlass.BFloat16, q_off, cute.AddressSpace.gmem, assumed_align=16)
        self.qkv_a(x, w_qkv_a, w_qkv_a_fp8, w_qkv_a_scale, qkv, rows, stream)
        self.pack(qkv, q_a_norm, kv_a_norm, positions, kv_slots, cos_sin, q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b, w_q_b_fp8, w_q_b_scale, q, rows, stream)
        self.absorb(q, w_uk, query, rows, stream)
        self.rope(q, positions, cos_sin, query, rows, stream)


class _ProducerFp8(_Producer):
    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer,
                 cos_sin: cute.Pointer, w_qkv_a: cute.Pointer, w_qkv_a_fp8: cute.Pointer,
                 w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer,
                 w_q_b: cute.Pointer, w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer,
                 w_uk: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer, q_resid: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, positions, kv_slots, cos_sin, w_qkv_a, w_qkv_a_fp8, w_qkv_a_scale, q_a_norm,
                  kv_a_norm, w_q_b, w_q_b_fp8, w_q_b_scale, w_uk, kv_cache, query, q_resid, scratch,
                  rows, stream)


class _IndexProducer:
    def __init__(self, g: GLMGeometry, fp8: bool = False):
        self.g = g
        i = g.index_heads
        self.wq = projection(i * 128, g.q_lora_rank, fp8)
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
        self.body(x, q_resid, positions, index_slots, cos_sin, w_iq, w_iq, w_iq, w_ik, k_norm_w, k_norm_b,
                  index_cache, q_fp8, head_weights, scratch, rows, stream)

    @cute.jit
    def body(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer,
             index_slots: cute.Pointer, cos_sin: cute.Pointer, w_iq: cute.Pointer, w_iq_fp8: cute.Pointer,
             w_iq_scale: cute.Pointer, w_ik: cute.Pointer, k_norm_w: cute.Pointer, k_norm_b: cute.Pointer,
             index_cache: cute.Pointer, q_fp8: cute.Pointer, head_weights: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        iq = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        kw_off = base + _align_i64(Int64(rows) * Int64(g.index_heads * 128 * 2))
        kw = cute.make_ptr(cutlass.BFloat16, kw_off, cute.AddressSpace.gmem, assumed_align=16)
        self.wq(q_resid, w_iq, w_iq_fp8, w_iq_scale, iq, rows, stream)
        self.wk(x, w_ik, kw, rows, stream)
        self.post(iq, kw, positions, index_slots, cos_sin, k_norm_w, k_norm_b, q_fp8, head_weights,
                  index_cache, rows, stream)


class _IndexProducerFp8(_IndexProducer):
    @cute.jit
    def __call__(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer,
                 index_slots: cute.Pointer, cos_sin: cute.Pointer, w_iq: cute.Pointer,
                 w_iq_fp8: cute.Pointer, w_iq_scale: cute.Pointer, w_ik: cute.Pointer,
                 k_norm_w: cute.Pointer, k_norm_b: cute.Pointer, index_cache: cute.Pointer,
                 q_fp8: cute.Pointer, head_weights: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, q_resid, positions, index_slots, cos_sin, w_iq, w_iq_fp8, w_iq_scale, w_ik, k_norm_w,
                  k_norm_b, index_cache, q_fp8, head_weights, scratch, rows, stream)


class _Output:
    def __init__(self, g: GLMGeometry, fp8: bool = False):
        self.g = g
        n, v = g.heads, g.v_head_dim
        self.uv = BatchedBf16Gemm(n=v, k=g.kv_lora_rank, batch=n, a_row=n * g.kv_lora_rank,
                                  a_batch=g.kv_lora_rank, o_row=n * v, o_batch=v)
        self.o = projection(g.hidden, n * v, fp8)

    def key(self) -> tuple:
        return (self.uv.key(), self.o.key(), self.g)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(attn, w_uv, w_o, w_o, w_o, out, scratch, rows, stream)

    @cute.jit
    def body(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
             w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
             stream: cuda.CUstream):
        values = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem,
                               assumed_align=16)
        self.uv(attn, w_uv, values, rows, stream)
        self.o(values, w_o, w_o_fp8, w_o_scale, out, rows, stream)


class _OutputFp8(_Output):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer, w_o_fp8: cute.Pointer,
                 w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(attn, w_uv, w_o, w_o_fp8, w_o_scale, out, scratch, rows, stream)


# ---------------------------------------------------------------------------
# FP8-only weights (``fp8_only``: no BF16 copies)
# ---------------------------------------------------------------------------


def kv_b_fp8_operands(name: str, g: GLMGeometry) -> tuple:
    """``w_uk`` / ``w_uv`` as E4M3 per-head slices of ``kv_b_proj`` with one FP32 scale per
    weight row and 64-wide K tile (the 128x128 grid's value there; heads span 3.5 blocks)."""
    n, c = g.heads, g.kv_lora_rank
    if name == "w_uk":
        shape, scale = f"[{n},{c},{g.qk_nope_dim}]", f"[{n},{c},{g.qk_nope_dim // 64}]"
    else:
        shape, scale = f"[{n},{g.v_head_dim},{c}]", f"[{n},{g.v_head_dim},{c // 64}]"
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, shape, note="kv_b_proj E4M3, per head"),
            Operand(f"{name}_scale", torch.float32, scale, align=4,
                    note="kv_b_proj block scale per weight row and 64-K tile"))


def _w8_batched_warps(prefill_rows: int | None) -> int:
    """MMA warps of the FP8 kv_b GEMMs: 128-row tiles in prefill programs (the E4M3 tile is
    widened once per 128 rows; SM120 at 325 W, 4096 rows, us: absorb 412 -> 262, values
    354 -> 269; BF16 with 4 warps 318 and 304), 64 in decode programs."""
    return 4 if prefill_rows is None else 8


def _w8_projection(n: int, k: int, prefill_rows: int | None, row_scales: bool = False) -> Fp8Projection:
    return Fp8Projection(n, k, prefill_rows=prefill_rows, row_scales=row_scales and prefill_rows is not None)


class _ProducerW8:
    """MLA producer over FP8-only weights. Prefill programs take ``q_a|kv_a`` (2624 rows: the
    block-FP8 GEMM wants whole 128-row blocks) with per-row scales, K-block major."""

    def __init__(self, g: GLMGeometry, prefill_rows: int | None):
        self.g = g
        n, q = g.heads, g.q_lora_rank
        self.qkv_a = _w8_projection(g.qkv_a_width, g.hidden, prefill_rows, row_scales=True)
        self.q_b = _w8_projection(n * g.qk_head_dim, q, prefill_rows)
        self.pack = GlmRankNormPackKV(q_rank=q, eps=g.norm_eps, page_rows=g.page_rows,
                                      record_bytes=g.record_bytes)
        self.absorb = BatchedFp8Gemm(
            n=g.kv_lora_rank, k=g.qk_nope_dim, batch=n, a_row=n * g.qk_head_dim,
            a_batch=g.qk_head_dim, o_row=n * g.latent_dim, o_batch=g.latent_dim,
            compute_warps=_w8_batched_warps(prefill_rows))
        self.rope = GlmQueryRope(heads=n, qk_nope=g.qk_nope_dim, qk_head=g.qk_head_dim,
                                 latent=g.latent_dim)

    def key(self) -> tuple:
        return ("w8", self.qkv_a.key(), self.q_b.key(), self.absorb.key(), self.g)

    def scratch_bytes(self, rows: int) -> int:
        return producer_scratch_bytes(self.g, rows) + self.qkv_a.prefill_scratch_bytes(rows)

    @cute.jit
    def body(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
             w_qkv_a_fp8: cute.Pointer, w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer,
             kv_a_norm: cute.Pointer, w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer,
             w_uk_fp8: cute.Pointer, w_uk_scale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
             q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        qkv = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        q_off = base + _align_i64(Int64(rows) * Int64(g.qkv_a_width * 2))
        q = cute.make_ptr(cutlass.BFloat16, q_off, cute.AddressSpace.gmem, assumed_align=16)
        qscratch = q_off + _align_i64(Int64(rows) * Int64(g.heads * g.qk_head_dim * 2))
        self.qkv_a(x, w_qkv_a_fp8, w_qkv_a_scale, qkv, rows, fp8_rows, qscratch, stream)
        self.pack(qkv, q_a_norm, kv_a_norm, positions, kv_slots, cos_sin, q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b_fp8, w_q_b_scale, q, rows, fp8_rows, qscratch, stream)
        self.absorb(q, w_uk_fp8, w_uk_scale, query, rows, stream)
        self.rope(q, positions, cos_sin, query, rows, stream)


class _ProducerW8Decode(_ProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv_a_fp8: cute.Pointer, w_qkv_a_scale: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer,
                 w_uk_fp8: cute.Pointer, w_uk_scale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
                 q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, positions, kv_slots, cos_sin, w_qkv_a_fp8, w_qkv_a_scale, q_a_norm, kv_a_norm, w_q_b_fp8,
                  w_q_b_scale, w_uk_fp8, w_uk_scale, kv_cache, query, q_resid, scratch, rows,
                  Int32(FP8_GEMV_ROWS), stream)


class _ProducerW8Prefill(_ProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, cos_sin: cute.Pointer,
                 w_qkv_a_fp8: cute.Pointer, w_qkv_a_kscale: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b_fp8: cute.Pointer, w_q_b_scale: cute.Pointer,
                 w_uk_fp8: cute.Pointer, w_uk_scale: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
                 q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, positions, kv_slots, cos_sin, w_qkv_a_fp8, w_qkv_a_kscale, q_a_norm, kv_a_norm, w_q_b_fp8,
                  w_q_b_scale, w_uk_fp8, w_uk_scale, kv_cache, query, q_resid, scratch, rows, fp8_rows, stream)


class _IndexProducerW8:
    def __init__(self, g: GLMGeometry, prefill_rows: int | None):
        self.g = g
        i = g.index_heads
        self.wq = _w8_projection(i * 128, g.q_lora_rank, prefill_rows)
        self.wk = glm_projection(128 + i, g.hidden)
        self.post = GlmIndexPost(heads=i, eps=g.index_norm_eps,
                                 weight_scale=float(i) ** -0.5 * 128.0 ** -0.5, page_rows=g.page_rows)

    def key(self) -> tuple:
        return ("w8", self.wq.key(), self.wk.key(), self.g)

    def scratch_bytes(self, rows: int) -> int:
        return index_producer_scratch_bytes(self.g, rows) + self.wq.prefill_scratch_bytes(rows)

    @cute.jit
    def body(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer, index_slots: cute.Pointer,
             cos_sin: cute.Pointer, w_iq_fp8: cute.Pointer, w_iq_scale: cute.Pointer, w_ik: cute.Pointer,
             k_norm_w: cute.Pointer, k_norm_b: cute.Pointer, index_cache: cute.Pointer, q_fp8: cute.Pointer,
             head_weights: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             stream: cuda.CUstream):
        g = self.g
        base = Int64(scratch.toint())
        iq = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        kw_off = base + _align_i64(Int64(rows) * Int64(g.index_heads * 128 * 2))
        kw = cute.make_ptr(cutlass.BFloat16, kw_off, cute.AddressSpace.gmem, assumed_align=16)
        qscratch = kw_off + _align_i64(Int64(rows) * Int64((128 + g.index_heads) * 2))
        self.wq(q_resid, w_iq_fp8, w_iq_scale, iq, rows, fp8_rows, qscratch, stream)
        self.wk(x, w_ik, kw, rows, stream)
        self.post(iq, kw, positions, index_slots, cos_sin, k_norm_w, k_norm_b, q_fp8, head_weights,
                  index_cache, rows, stream)


class _IndexProducerW8Decode(_IndexProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer, index_slots: cute.Pointer,
                 cos_sin: cute.Pointer, w_iq_fp8: cute.Pointer, w_iq_scale: cute.Pointer, w_ik: cute.Pointer,
                 k_norm_w: cute.Pointer, k_norm_b: cute.Pointer, index_cache: cute.Pointer, q_fp8: cute.Pointer,
                 head_weights: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body(x, q_resid, positions, index_slots, cos_sin, w_iq_fp8, w_iq_scale, w_ik, k_norm_w, k_norm_b,
                  index_cache, q_fp8, head_weights, scratch, rows, Int32(FP8_GEMV_ROWS), stream)


class _IndexProducerW8Prefill(_IndexProducerW8):
    @cute.jit
    def __call__(self, x: cute.Pointer, q_resid: cute.Pointer, positions: cute.Pointer, index_slots: cute.Pointer,
                 cos_sin: cute.Pointer, w_iq_fp8: cute.Pointer, w_iq_scale: cute.Pointer, w_ik: cute.Pointer,
                 k_norm_w: cute.Pointer, k_norm_b: cute.Pointer, index_cache: cute.Pointer, q_fp8: cute.Pointer,
                 head_weights: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(x, q_resid, positions, index_slots, cos_sin, w_iq_fp8, w_iq_scale, w_ik, k_norm_w, k_norm_b,
                  index_cache, q_fp8, head_weights, scratch, rows, fp8_rows, stream)


class _OutputW8:
    def __init__(self, g: GLMGeometry, prefill_rows: int | None):
        self.g = g
        n, v = g.heads, g.v_head_dim
        self.uv = BatchedFp8Gemm(n=v, k=g.kv_lora_rank, batch=n, a_row=n * g.kv_lora_rank,
                                 a_batch=g.kv_lora_rank, o_row=n * v, o_batch=v,
                                 compute_warps=_w8_batched_warps(prefill_rows))
        self.o = _w8_projection(g.hidden, n * v, prefill_rows)

    def key(self) -> tuple:
        return ("w8", self.uv.key(), self.o.key(), self.g)

    def scratch_bytes(self, rows: int) -> int:
        return o_scratch_bytes(self.g, rows) + self.o.prefill_scratch_bytes(rows)

    @cute.jit
    def body(self, attn: cute.Pointer, w_uv_fp8: cute.Pointer, w_uv_scale: cute.Pointer, w_o_fp8: cute.Pointer,
             w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
             stream: cuda.CUstream):
        values = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem,
                               assumed_align=16)
        qscratch = Int64(scratch.toint()) + _align_i64(Int64(rows) * Int64(self.g.heads * self.g.v_head_dim * 2))
        self.uv(attn, w_uv_fp8, w_uv_scale, values, rows, stream)
        self.o(values, w_o_fp8, w_o_scale, out, rows, fp8_rows, qscratch, stream)


class _OutputW8Decode(_OutputW8):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv_fp8: cute.Pointer, w_uv_scale: cute.Pointer, w_o_fp8: cute.Pointer,
                 w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.body(attn, w_uv_fp8, w_uv_scale, w_o_fp8, w_o_scale, out, scratch, rows, Int32(FP8_GEMV_ROWS), stream)


class _OutputW8Prefill(_OutputW8):
    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv_fp8: cute.Pointer, w_uv_scale: cute.Pointer, w_o_fp8: cute.Pointer,
                 w_o_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32, fp8_rows: Int32,
                 stream: cuda.CUstream):
        self.body(attn, w_uv_fp8, w_uv_scale, w_o_fp8, w_o_scale, out, scratch, rows, fp8_rows, stream)


def _compile_w8(kind: str, g: GLMGeometry, max_rows: int, mode: str):
    prefill = check_w8_mode(mode) == "prefill"
    rows_cap = int(max_rows) if prefill else None
    h, q, n, i = g.hidden, g.q_lora_rank, g.heads, g.index_heads
    w8 = lambda name, n_, k_, row=False: fp8_only_operands(name, n_, k_, row_scales=row and prefill,  # noqa: E731
                                                         prefill=prefill)
    if kind == "producer":
        launch = (_ProducerW8Prefill if prefill else _ProducerW8Decode)(g, rows_cap)
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{h}]"),
            Operand("positions", torch.int64, "[rows]", align=8),
            Operand("kv_slots", torch.int64, "[rows]", align=8),
            Operand("cos_sin", torch.float32, "[P,64]", align=4),
            *w8("w_qkv_a", g.qkv_a_width, h, row=True),
            Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
            Operand("kv_a_norm", torch.bfloat16, "[512]"),
            *w8("w_q_b", n * g.qk_head_dim, q),
            *kv_b_fp8_operands("w_uk", g),
            Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
            Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
            Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
            Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
        )
        name, geometry = "glm_producer", {"hidden": h, "q_lora_rank": q, "heads": n, "record_bytes": g.record_bytes,
                                          "page_rows": g.page_rows, "eps": g.norm_eps}
    elif kind == "index_producer":
        launch = (_IndexProducerW8Prefill if prefill else _IndexProducerW8Decode)(g, rows_cap)
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{h}]"),
            Operand("q_resid", torch.bfloat16, f"[rows,{q}]"),
            Operand("positions", torch.int64, "[rows]", align=8),
            Operand("index_slots", torch.int64, "[rows]", align=8),
            Operand("cos_sin", torch.float32, "[P,64]", align=4),
            *w8("w_iq", i * 128, q),
            Operand("w_ik", torch.bfloat16, f"[{128 + i},{h}]", note="cat(wk, weights_proj): BF16 (mixed in the release)"),
            Operand("k_norm_w", torch.bfloat16, "[128]"),
            Operand("k_norm_b", torch.bfloat16, "[128]"),
            Operand("index_cache", torch.uint8, f"[pages,{g.index_page_bytes}]", "inout"),
            Operand("q_fp8", torch.float8_e4m3fn, f"[rows,{i},128]", "out"),
            Operand("head_weights", torch.float32, f"[rows,{i}]", "out"),
            Operand("scratch", torch.uint8, "[index_producer_scratch_bytes]", "scratch"),
        )
        name, geometry = "glm_index_producer", {"hidden": h, "q_lora_rank": q, "index_heads": i,
                                                "eps": g.index_norm_eps}
    else:
        launch = (_OutputW8Prefill if prefill else _OutputW8Decode)(g, rows_cap)
        operands = (
            Operand("attn", torch.bfloat16, f"[rows,{n},512]"),
            *kv_b_fp8_operands("w_uv", g),
            *w8("w_o", h, n * g.v_head_dim),
            Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
            Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch"),
        )
        name, geometry = "glm_o", {"hidden": h, "heads": n, "v_head_dim": g.v_head_dim}
    geometry.update(max_rows=int(max_rows), fp8_weights="only", mode=mode)
    return compile_program(
        launch, name=name, operands=operands, scalars=w8_scalars(prefill), key=(int(max_rows), mode, launch.key()),
        geometry=geometry, scratch={"scratch": launch.scratch_bytes}, doc=__doc__,
    )


def compile_glm_producer_aot(g: GLMGeometry = GLM53, *, max_rows: int, fp8: bool = False,
                             fp8_only: str | None = None):
    """MLA producer for ``rows <= max_rows``; see the module docstring. ``fp8``
    adds the decode FP8 weight operands."""
    if fp8_only is not None:
        return _compile_w8("producer", g, max_rows, fp8_only)
    max_rows = _check_rows(max_rows)
    launch = (_ProducerFp8 if fp8 else _Producer)(g, fp8)
    h, q, n = g.hidden, g.q_lora_rank, g.heads
    extra = lambda name, n_, k_: fp8_operands(name, n_, k_) if fp8 else ()  # noqa: E731
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{h}]"),
        *extra("w_qkv_a", g.qkv_a_width, h),
        Operand("q_a_norm", torch.bfloat16, f"[{q}]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("w_q_b", torch.bfloat16, f"[{n * g.qk_head_dim},{q}]"),
        *extra("w_q_b", n * g.qk_head_dim, q),
        Operand("w_uk", torch.bfloat16, f"[{n},512,{g.qk_nope_dim}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},{g.latent_dim}]", "out"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows, "fp8_weights": fp8,
                  "record_bytes": g.record_bytes, "page_rows": g.page_rows, "eps": g.norm_eps},
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


def compile_glm_index_producer_aot(g: GLMGeometry = GLM53, *, max_rows: int, fp8: bool = False,
                                   fp8_only: str | None = None):
    """DSA index query/key producer for ``rows <= max_rows``; see the module
    docstring. ``fp8`` adds the decode FP8 ``w_iq`` operands."""
    if fp8_only is not None:
        return _compile_w8("index_producer", g, max_rows, fp8_only)
    max_rows = _check_rows(max_rows)
    launch = (_IndexProducerFp8 if fp8 else _IndexProducer)(g, fp8)
    h, q, i = g.hidden, g.q_lora_rank, g.index_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("q_resid", torch.bfloat16, f"[rows,{q}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("index_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_iq", torch.bfloat16, f"[{i * 128},{q}]"),
        *(fp8_operands("w_iq", i * 128, q) if fp8 else ()),
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
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "q_lora_rank": q, "index_heads": i, "max_rows": max_rows, "fp8_weights": fp8,
                  "eps": g.index_norm_eps},
        scratch={"scratch": lambda rows: index_producer_scratch_bytes(g, rows)},
        doc=__doc__,
    )


def compile_glm_o_aot(g: GLMGeometry = GLM53, *, max_rows: int, fp8: bool = False,
                      fp8_only: str | None = None):
    """W_UV per head then o_proj for ``rows <= max_rows``; see the module
    docstring. ``fp8`` adds the decode FP8 ``w_o`` operands."""
    if fp8_only is not None:
        return _compile_w8("o", g, max_rows, fp8_only)
    max_rows = _check_rows(max_rows)
    launch = (_OutputFp8 if fp8 else _Output)(g, fp8)
    h, n, v = g.hidden, g.heads, g.v_head_dim
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{n},512]"),
        Operand("w_uv", torch.bfloat16, f"[{n},{v},512]"),
        Operand("w_o", torch.bfloat16, f"[{h},{n * v}]"),
        *(fp8_operands("w_o", h, n * v) if fp8 else ()),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.uint8, "[o_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="glm_o", operands=operands, scalars=(Scalar("rows"),),
        key=(max_rows, fp8, launch.key()),
        geometry={"hidden": h, "heads": n, "v_head_dim": v, "max_rows": max_rows, "fp8_weights": fp8},
        scratch={"scratch": lambda rows: o_scratch_bytes(g, rows)},
        doc=__doc__,
    )
