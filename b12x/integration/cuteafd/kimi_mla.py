"""Kimi K3 NoPE MLA with caller-owned BF16 or compact GLM-format records.

Records hold 512 normalized latents (BF16 or grouped E4M3/FP32 scales) and
64 raw BF16 key values. The shared MLA drafter additionally rotates the tail.
The query is absorbed through UK; UV, sigmoid(g_proj(x)) and o_proj follow
full causal dense_mla. Compact storage widens only the current shared-memory
tile to BF16 arithmetic; native FP8 tensor-core attention is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32, fabs_f32, fmax_f32
import torch

from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
from b12x.attention.dense_mla._layout import choose_layout
from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, _bf16, _block_sum, _fp8_scale, _reduction_storage, _rsqrt, _warp_max, glm_projection
from ._qwen4_attention_kernels import Qwen4GateMul


@dataclass(frozen=True)
class KimiMlaGeometry:
    name: str = "kimi_k3_mla"
    hidden: int = 7168
    heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_dim: int = 128
    qk_rope_dim: int = 64
    v_head_dim: int = 128
    norm_eps: float = 1e-6
    page_rows: int = 64
    kv_cache: str = "bf16"

    def __post_init__(self):
        if self.kv_cache not in ("bf16", "fp8"):
            raise ValueError("MLA KV cache must be bf16 or fp8")
        if self.heads not in (48, 96):
            raise ValueError("Kimi MLA requires 48 or 96 heads")
        if (self.hidden, self.q_lora_rank, self.kv_lora_rank, self.qk_nope_dim,
            self.qk_rope_dim, self.v_head_dim, self.page_rows, self.norm_eps) != (
                7168, 1536, 512, 128, 64, 128, 64, 1e-6):
            raise ValueError("unsupported Kimi K3 MLA geometry or rank-norm epsilon")

    @property
    def qkv_a_width(self):
        return self.q_lora_rank + 576

    @property
    def record_bytes(self):
        return 656 if self.kv_cache == "fp8" else 1152

    @property
    def softmax_scale(self):
        return 192 ** -0.5


@dataclass(frozen=True)
class MlaDraftGeometry(KimiMlaGeometry):
    """Checkpoint-native block drafter, distinct from the NoPE target.

    The shared loader supplies the YaRN-adjusted score scale, frequencies and
    rotary magnitude factor as runtime inputs, never inferred here.
    """

    name: str = "mla_block_draft"
    heads: int = 64
    norm_eps: float = 1e-5
    window_size: int | None = None

    def __post_init__(self):
        if self.kv_cache not in ("bf16", "fp8"):
            raise ValueError("MLA draft KV cache must be bf16 or fp8")
        if (self.hidden, self.heads, self.q_lora_rank, self.kv_lora_rank,
            self.qk_nope_dim, self.qk_rope_dim, self.v_head_dim,
            self.page_rows, self.norm_eps) != (
                7168, 64, 1536, 512, 128, 64, 128, 64, 1e-5):
            raise ValueError("unsupported MLA block-draft geometry")
        if self.window_size not in (None, 4096):
            raise ValueError("MLA block draft supports full context or window 4096")


def _align(n):
    return (int(n) + 1023) // 1024 * 1024


@cute.jit
def _offset(n: Int64) -> Int64:
    return (n + Int64(1023)) // Int64(1024) * Int64(1024)


def producer_scratch_bytes(g, rows):
    return _align(rows * g.qkv_a_width * 2) + _align(rows * g.heads * 192 * 2)


def output_scratch_bytes(g, rows):
    return 2 * _align(rows * g.heads * 128 * 2)


def dense_mla_layout(g, rows, *, mode):
    """The one specialization this geometry serves in this mode."""
    return choose_layout(
        mode=mode,
        fp8=False,
        qk_dim=576,
        value_dim=512,
        single_request=True,
        windowed=isinstance(g, MlaDraftGeometry),
        query_rows=rows,
    )


def attention_layout(g, rows, *, max_context, mode):
    if mode not in ("decode", "extend", "verify"):
        raise ValueError("mode must be decode, extend or verify")
    if not 1 <= int(max_context) <= 1048576 or int(rows) <= 0:
        raise ValueError("invalid context or row capacity")
    chunk = dense_mla_layout(g, rows, mode=mode).candidates_per_chunk
    chunks_total = (max_context + chunk - 1) // chunk
    splits = min(16, chunks_total) if mode != "extend" else 1
    chunks = (chunks_total + splits - 1) // splits
    part = _align(rows * g.heads * splits * 512 * 2) if splits > 1 else 0
    lse = _align(rows * g.heads * splits * 4) if splits > 1 else 0
    return dict(num_splits=splits, chunks_per_split=chunks,
                partial_output_bytes=part, partial_lse_bytes=lse,
                scratch_bytes=part + lse)


class _RankNormPack:
    threads = 256

    def __init__(self, g):
        self.g = g

    @cute.jit
    def __call__(self, qkv: cute.Pointer, q_norm: cute.Pointer, kv_norm: cute.Pointer,
                 slots: cute.Pointer, q_resid: cute.Pointer, cache: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(qkv, q_norm, kv_norm, slots, q_resid, cache).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Pointer, q_norm: cute.Pointer, kv_norm: cute.Pointer,
               slots: cute.Pointer, q_resid: cute.Pointer, cache: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, 8))
        sums = storage.sums.get_tensor(cute.make_layout((2, 8), stride=(8, 1)))
        q = cute.make_rmem_tensor((6,), Float32)
        kv = cute.make_rmem_tensor((2,), Float32)
        qs, ks = Float32(0), Float32(0)
        for j in cutlass.range_constexpr(6):
            d = Int64(j * 256) + Int64(tid)
            q[j] = Float32(qkv[row * Int64(2112) + d])
            qs += q[j] * q[j]
        for j in cutlass.range_constexpr(2):
            d = Int64(j * 256) + Int64(tid)
            kv[j] = Float32(qkv[row * Int64(2112) + Int64(1536) + d])
            ks += kv[j] * kv[j]
        qi = _rsqrt(_block_sum(qs, sums, Int32(0), 8) / Float32(1536) + Float32(self.g.norm_eps))
        ki = _rsqrt(_block_sum(ks, sums, Int32(1), 8) / Float32(512) + Float32(self.g.norm_eps))
        for j in cutlass.range_constexpr(6):
            d = Int64(j * 256) + Int64(tid)
            q_resid[row * Int64(1536) + d] = (Float32(q_norm[d]) * _bf16(q[j] * qi)).to(BFloat16)
        slot = Int64(slots[row])
        if slot >= Int64(0):
            if cutlass.const_expr(self.g.kv_cache == "fp8"):
                record = Int64(cache.toint()) + slot * Int64(self.g.record_bytes)
                # Preserve Kimi's rank-norm sum order; only its rounded result
                # is regrouped into GLM's four independent 128-value blocks.
                if tid < Int32(128):
                    normalized = cute.make_rmem_tensor((4,), Float32)
                    local = Float32(0)
                    for j in cutlass.range_constexpr(4):
                        d = Int64(4) * Int64(tid) + Int64(j)
                        value = Float32(qkv[row * Int64(2112) + Int64(1536) + d])
                        normalized[j] = _bf16(Float32(kv_norm[d]) * _bf16(value * ki))
                        local = fmax_f32(local, fabs_f32(normalized[j]))
                    scale = _fp8_scale(_warp_max(local))
                    words = cute.make_ptr(Uint32, record, cute.AddressSpace.gmem, assumed_align=4)
                    words[tid] = cvt_f32x4_to_e4m3x4(
                        div_rn_f32(normalized[0], scale), div_rn_f32(normalized[1], scale),
                        div_rn_f32(normalized[2], scale), div_rn_f32(normalized[3], scale))
                    if tid % Int32(32) == Int32(0):
                        scales = cute.make_ptr(Float32, record + Int64(512), cute.AddressSpace.gmem, assumed_align=4)
                        scales[tid // Int32(32)] = scale
                if tid < Int32(64):
                    tail = cute.make_ptr(BFloat16, record + Int64(528), cute.AddressSpace.gmem, assumed_align=2)
                    tail[tid] = qkv[row * Int64(2112) + Int64(2048) + Int64(tid)]
            else:
                for j in cutlass.range_constexpr(2):
                    d = Int64(j * 256) + Int64(tid)
                    cache[slot * Int64(576) + d] = (Float32(kv_norm[d]) * _bf16(kv[j] * ki)).to(BFloat16)
                if tid < Int32(64):
                    cache[slot * Int64(576) + Int64(512) + Int64(tid)] = qkv[row * Int64(2112) + Int64(2048) + Int64(tid)]


class _QueryTail:
    def __init__(self, heads):
        self.heads = heads

    @cute.jit
    def __call__(self, q: cute.Pointer, query: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(q, query).launch(grid=(rows, self.heads, 1), block=(64, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Pointer, query: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        d = Int64(cute.arch.thread_idx()[0])
        h = Int64(row) * Int64(self.heads) + Int64(head)
        query[h * Int64(576) + Int64(512) + d] = q[h * Int64(192) + Int64(128) + d]


class _Producer:
    def __init__(self, g):
        self.g = g
        self.qkv = glm_projection(g.qkv_a_width, g.hidden)
        self.norm = _RankNormPack(g)
        self.q_b = glm_projection(g.heads * 192, g.q_lora_rank)
        self.uk = BatchedBf16Gemm(n=512, k=128, batch=g.heads, a_row=g.heads * 192,
                                   a_batch=192, o_row=g.heads * 576, o_batch=576)
        self.tail = _QueryTail(g.heads)

    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, w_qkv_a: cute.Pointer,
                 q_a_norm: cute.Pointer, kv_a_norm: cute.Pointer, w_q_b: cute.Pointer,
                 w_uk: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer,
                 q_resid: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        qkv = cute.recast_ptr(scratch, dtype=BFloat16)
        q = cute.make_ptr(BFloat16, Int64(scratch.toint()) + _offset(Int64(rows) * Int64(self.g.qkv_a_width * 2)),
                          cute.AddressSpace.gmem, assumed_align=16)
        self.qkv(x, w_qkv_a, qkv, rows, stream)
        self.norm(qkv, q_a_norm, kv_a_norm, kv_slots, q_resid, kv_cache, rows, stream)
        self.q_b(q_resid, w_q_b, q, rows, stream)
        self.uk(q, w_uk, query, rows, stream)
        self.tail(q, query, rows, stream)


class _DraftRotary:
    """Interleaved (GPT-J) rotary-64, matching the MLA checkpoint layout."""

    def __init__(self, g):
        self.heads = g.heads
        self.record_bytes = g.record_bytes
        self.tail_offset = 528 if g.kv_cache == "fp8" else 1024

    @cute.jit
    def __call__(self, query: cute.Pointer, cache: cute.Pointer, slots: cute.Pointer,
                 positions: cute.Pointer, inv_freq: cute.Pointer, factor: Float32,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(query, cache, slots, positions, inv_freq, factor).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, query: cute.Pointer, cache: cute.Pointer, slots: cute.Pointer,
               positions: cute.Pointer, inv_freq: cute.Pointer, factor: Float32):
        row, head, _ = cute.arch.block_idx()
        pair = Int64(cute.arch.thread_idx()[0])
        angle = Float32(positions[Int64(row)]) * Float32(inv_freq[pair])
        c = _bf16(cute.math.cos(angle, fastmath=False) * factor)
        s = _bf16(cute.math.sin(angle, fastmath=False) * factor)
        base = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(576) + Int64(512) + pair * Int64(2)
        x, y = Float32(query[base]), Float32(query[base + Int64(1)])
        query[base] = (_bf16(x * c) - _bf16(y * s)).to(BFloat16)
        query[base + Int64(1)] = (_bf16(y * c) + _bf16(x * s)).to(BFloat16)
        slot = Int64(slots[Int64(row)])
        if head == Int32(0) and slot >= Int64(0):
            tail = cute.make_ptr(BFloat16, Int64(cache.toint()) + slot * Int64(self.record_bytes)
                + Int64(self.tail_offset), cute.AddressSpace.gmem, assumed_align=2)
            base = pair * Int64(2)
            x, y = Float32(tail[base]), Float32(tail[base + Int64(1)])
            tail[base] = (_bf16(x * c) - _bf16(y * s)).to(BFloat16)
            tail[base + Int64(1)] = (_bf16(y * c) + _bf16(x * s)).to(BFloat16)


class _DraftProducer:
    def __init__(self, g):
        self.producer = _Producer(g)
        self.rotary = _DraftRotary(g)

    @cute.jit
    def __call__(self, x: cute.Pointer, kv_slots: cute.Pointer, positions: cute.Pointer,
                 inv_freq: cute.Pointer, w_qkv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, w_q_b: cute.Pointer, w_uk: cute.Pointer,
                 kv_cache: cute.Pointer, query: cute.Pointer, q_resid: cute.Pointer,
                 scratch: cute.Pointer, rotary_factor: Float32, rows: Int32, stream: cuda.CUstream):
        self.producer(x, kv_slots, w_qkv_a, q_a_norm, kv_a_norm, w_q_b, w_uk,
                      kv_cache, query, q_resid, scratch, rows, stream)
        self.rotary(query, kv_cache, kv_slots, positions, inv_freq, rotary_factor, rows, stream)


class _DraftOutput:
    def __init__(self, g):
        self.uv = BatchedBf16Gemm(n=128, k=512, batch=g.heads, a_row=g.heads * 512,
                                   a_batch=512, o_row=g.heads * 128, o_batch=128)
        self.o = glm_projection(g.hidden, g.heads * 128)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, w_o: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        values = cute.recast_ptr(scratch, dtype=BFloat16)
        self.uv(attn, w_uv, values, rows, stream)
        self.o(values, w_o, out, rows, stream)


class _Output:
    def __init__(self, g):
        self.g = g
        self.uv = BatchedBf16Gemm(n=128, k=512, batch=g.heads, a_row=g.heads * 512,
                                   a_batch=512, o_row=g.heads * 128, o_batch=128)
        self.gate_proj = glm_projection(g.heads * 128, g.hidden)
        self.gate_mul = Qwen4GateMul(g.heads * 128)
        self.o = glm_projection(g.hidden, g.heads * 128)

    @cute.jit
    def __call__(self, attn: cute.Pointer, x: cute.Pointer, w_uv: cute.Pointer,
                 w_g: cute.Pointer, w_o: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        values = cute.recast_ptr(scratch, dtype=BFloat16)
        gates = cute.make_ptr(BFloat16, Int64(scratch.toint()) + _offset(Int64(rows) * Int64(self.g.heads * 128 * 2)),
                              cute.AddressSpace.gmem, assumed_align=16)
        self.uv(attn, w_uv, values, rows, stream)
        self.gate_proj(x, w_g, gates, rows, stream)
        self.gate_mul(values, gates, values, rows, stream)
        self.o(values, w_o, out, rows, stream)


class _Attention:
    def __init__(self, g, rows, max_context, mode):
        self.g = g
        self.layout = attention_layout(g, rows, max_context=max_context, mode=mode)
        self.splits = self.layout["num_splits"]
        # Decode and verify keep one query row per tile; a prefill tile shares
        # one gathered KV window across all of its query rows.
        self.smem = dense_mla_layout(g, rows, mode=mode)
        self.forward = DenseMlaForwardKernel(layout=self.smem,
            page_size=64, num_heads=g.heads, num_splits=self.splits,
            chunks_per_split=self.layout["chunks_per_split"],
            query_tile=self.smem.query_tile, fp8=False,
            qk_dim=576, value_dim=512,
            window_size=g.window_size if isinstance(g, MlaDraftGeometry) else None,
            noncausal_block=isinstance(g, MlaDraftGeometry),
            ds_mla_record=g.kv_cache == "fp8")
        self.merge = DenseMlaMergeKernel(num_splits=self.splits, value_dim=512) if self.splits > 1 else None

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                 cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                 lse: cute.Pointer, scratch: cute.Pointer, rows: Int32, batch: Int32,
                 page_table_width: Int64, active_splits: Int32, stream: cuda.CUstream):
        self.run_attention(query, kv_cache, page_table, cache_seqlens, cu_seqlens_q,
            attn, lse, scratch, rows, batch, page_table_width, active_splits,
            Float32(self.g.softmax_scale * math.log2(math.e)), stream)

    @cute.jit
    def run_attention(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                      cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                      lse: cute.Pointer, scratch: cute.Pointer, rows: Int32, batch: Int32,
                      page_table_width: Int64, active_splits: Int32, score_scale_log2: Float32,
                      stream: cuda.CUstream):
        g = self.g
        q = cute.make_tensor(cute.recast_ptr(query, dtype=cutlass.Uint8), cute.make_layout((1,)))
        kv = cute.make_tensor(cute.recast_ptr(kv_cache, dtype=cutlass.Uint8), cute.make_layout((1,)))
        table = cute.make_tensor(page_table, cute.make_layout((Int64(batch) * page_table_width,)))
        lengths = cute.make_tensor(cache_seqlens, cute.make_layout((batch,)))
        cu = cute.make_tensor(cu_seqlens_q, cute.make_layout((batch + Int32(1),)))
        out = cute.make_tensor(attn, cute.make_layout((rows, g.heads, 512), stride=(g.heads * 512, 512, 1)))
        lse_out = cute.make_tensor(lse, cute.make_layout((rows, g.heads), stride=(g.heads, 1)))
        if cutlass.const_expr(self.splits > 1):
            po = cute.make_tensor(cute.recast_ptr(scratch, dtype=BFloat16),
                cute.make_layout((rows, g.heads, self.splits, 512), stride=(g.heads * self.splits * 512, self.splits * 512, 512, 1)))
            pl_ptr = cute.make_ptr(Float32, Int64(scratch.toint()) + _offset(Int64(rows) * Int64(g.heads * self.splits * 512 * 2)),
                                   cute.AddressSpace.gmem, assumed_align=16)
            pl = cute.make_tensor(pl_ptr, cute.make_layout((rows, g.heads, self.splits), stride=(g.heads * self.splits, self.splits, 1)))
        else:
            po, pl = out, lse_out
        self.forward(q, kv, table, lengths, cu, out, lse_out, po, pl, lse_out, lse_out,
            score_scale_log2, Int64(g.heads * 576 * 2), Int64(576 * 2),
            Int64(64 * g.record_bytes), Int64(g.record_bytes), page_table_width, rows, batch, active_splits, stream)
        if cutlass.const_expr(self.splits > 1):
            if active_splits > Int32(1):
                self.merge(po, pl, out, lse_out, active_splits, stream)


class _DraftAttention(_Attention):
    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                 cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                 lse: cute.Pointer, scratch: cute.Pointer, rows: Int32, batch: Int32,
                 page_table_width: Int64, active_splits: Int32, attention_scale: Float32,
                 stream: cuda.CUstream):
        self.run_attention(query, kv_cache, page_table, cache_seqlens, cu_seqlens_q,
            attn, lse, scratch, rows, batch, page_table_width, active_splits,
            attention_scale * Float32(math.log2(math.e)), stream)


def compile_mla_draft_producer_aot(g: MlaDraftGeometry, *, max_rows=64):
    if not isinstance(g, MlaDraftGeometry):
        raise TypeError("MLA draft requires MlaDraftGeometry")
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{g.hidden}]"),
        Operand("q_a_norm", torch.bfloat16, "[1536]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("w_q_b", torch.bfloat16, f"[{g.heads * 192},1536]"),
        Operand("w_uk", torch.bfloat16, f"[{g.heads},512,128]"),
        Operand("kv_cache", torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
                "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]", "out"),
        Operand("q_resid", torch.bfloat16, "[rows,1536]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"))
    return compile_program(_DraftProducer(g), name="mla_draft_producer", operands=operands,
        scalars=(Scalar("rotary_factor", "float32"), Scalar("rows")), key=(g, max_rows),
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, rows)},
        geometry={**vars(g), "rotary_style": "interleaved", "rotary_dim": 64})


def compile_mla_draft_output_aot(g: MlaDraftGeometry, *, max_rows=64):
    if not isinstance(g, MlaDraftGeometry):
        raise TypeError("MLA draft requires MlaDraftGeometry")
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]"),
        Operand("w_uv", torch.bfloat16, f"[{g.heads},128,512]"),
        Operand("w_o", torch.bfloat16, f"[{g.hidden},{g.heads * 128}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"),
        Operand("scratch", torch.uint8, "[output_scratch_bytes]", "scratch"))
    return compile_program(_DraftOutput(g), name="mla_draft_output", operands=operands,
        scalars=(Scalar("rows"),), key=(g, max_rows),
        scratch={"scratch": lambda rows: _align(rows * g.heads * 128 * 2)},
        geometry={**vars(g), "output_gate": False})


def compile_mla_draft_attention_aot(g: MlaDraftGeometry, *, max_rows=64, max_context=1048576):
    if not isinstance(g, MlaDraftGeometry):
        raise TypeError("MLA draft requires MlaDraftGeometry")
    layout = attention_layout(g, max_rows, max_context=max_context, mode="verify")
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand("kv_cache", torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
                "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]"),
        Operand("page_table", torch.int32, "[batch,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[batch]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out"),
        Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch"))
    return compile_program(_DraftAttention(g, max_rows, max_context, "verify"), name="mla_draft_attention",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits"),
                 Scalar("attention_scale", "float32")),
        key=(g, max_rows, max_context),
        scratch={"scratch": lambda rows: attention_layout(g, rows, max_context=max_context, mode="verify")["scratch_bytes"]},
        geometry={**vars(g), **layout, "noncausal_block": True, "max_context": max_context})


def compile_kimi_mla_producer_aot(g=KimiMlaGeometry(), *, max_rows=64):
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{g.hidden}]"),
        Operand("q_a_norm", torch.bfloat16, "[1536]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("w_q_b", torch.bfloat16, f"[{g.heads * 192},1536]"),
        Operand("w_uk", torch.bfloat16, f"[{g.heads},512,128]"),
        Operand("kv_cache", torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
                "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]", "out"),
        Operand("q_resid", torch.bfloat16, "[rows,1536]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"))
    return compile_program(_Producer(g), name="kimi_mla_producer", operands=operands,
        scalars=(Scalar("rows"),), key=(g, max_rows),
        scratch={"scratch": lambda rows: producer_scratch_bytes(g, rows)},
        geometry=vars(g))


def compile_kimi_mla_output_aot(g=KimiMlaGeometry(), *, max_rows=64):
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]"),
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("w_uv", torch.bfloat16, f"[{g.heads},128,512]"),
        Operand("w_g", torch.bfloat16, f"[{g.heads * 128},{g.hidden}]"),
        Operand("w_o", torch.bfloat16, f"[{g.hidden},{g.heads * 128}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"),
        Operand("scratch", torch.uint8, "[output_scratch_bytes]", "scratch"))
    return compile_program(_Output(g), name="kimi_mla_output", operands=operands,
        scalars=(Scalar("rows"),), key=(g, max_rows),
        scratch={"scratch": lambda rows: output_scratch_bytes(g, rows)},
        geometry=vars(g))


def compile_kimi_dense_mla_aot(g=KimiMlaGeometry(), *, max_rows=64, max_context=1048576, mode="decode"):
    layout = attention_layout(g, max_rows, max_context=max_context, mode=mode)
    smem = dense_mla_layout(g, max_rows, mode=mode)
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand("kv_cache", torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
                "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]"),
        Operand("page_table", torch.int32, "[batch,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[batch]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out"),
        Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch"))
    native_fp8_extend = (g.kv_cache == "fp8" and mode == "extend"
                         and not isinstance(g, MlaDraftGeometry))
    if native_fp8_extend:
        # Compact records on the causal extend route read FP8 natively (no
        # split, no scratch); decode/verify keep the shared widening reader.
        from .kimi_mla_prefill_fp8 import DEFAULT_FP8_PREFILL_TILE, _Fp8PrefillAsDenseMla
        layout = {**layout, "num_splits": 1, "chunks_per_split": layout["num_splits"] * layout["chunks_per_split"],
                  "partial_output_bytes": 0, "partial_lse_bytes": 0, "scratch_bytes": 0}
        return compile_program(_Fp8PrefillAsDenseMla(g, DEFAULT_FP8_PREFILL_TILE), name="kimi_dense_mla",
            operands=operands, scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
            key=(g, max_rows, max_context, mode, "native_fp8_prefill", DEFAULT_FP8_PREFILL_TILE),
            scratch={"scratch": lambda rows: 0},
            geometry={**vars(g), **layout, "mode": mode, "max_context": max_context,
                      "native_fp8_prefill": DEFAULT_FP8_PREFILL_TILE})
    return compile_program(_Attention(g, max_rows, max_context, mode), name="kimi_dense_mla",
        operands=operands, scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        # The tile is part of the compiled kernel, so it has to be part of the
        # program and disk cache key too: mode and context happen to determine
        # it today, and a future planner change must not silently reuse a
        # cubin built for a different tile.
        key=(g, max_rows, max_context, mode, smem.query_tile, smem.fp8,
             smem.kv_stages, smem.heads_per_tile, smem.candidates_per_chunk,
             smem.math_warps_per_query),
        scratch={"scratch": lambda rows: attention_layout(g, rows, max_context=max_context, mode=mode)["scratch_bytes"]},
        geometry={**vars(g), **layout, "mode": mode, "max_context": max_context})
