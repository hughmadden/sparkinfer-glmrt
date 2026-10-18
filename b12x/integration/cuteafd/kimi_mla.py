"""Kimi K3 NoPE MLA: lossless BF16 records and caller-owned AOT storage.

Records contain 512 normalized latent values followed by 64 *raw* key values.
The query is absorbed through UK; UV, sigmoid(g_proj(x)) and o_proj follow
full causal dense_mla. No rotary tables, positions, indexer or allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
import torch

from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
from b12x.attention.dense_mla._layout import make_smem_layout
from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, _bf16, _block_sum, _reduction_storage, _rsqrt, glm_projection
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

    def __post_init__(self):
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
        return 1152

    @property
    def softmax_scale(self):
        return 192 ** -0.5


def _align(n):
    return (int(n) + 1023) // 1024 * 1024


@cute.jit
def _offset(n: Int64) -> Int64:
    return (n + Int64(1023)) // Int64(1024) * Int64(1024)


def producer_scratch_bytes(g, rows):
    return _align(rows * g.qkv_a_width * 2) + _align(rows * g.heads * 192 * 2)


def output_scratch_bytes(g, rows):
    return 2 * _align(rows * g.heads * 128 * 2)


def attention_layout(g, rows, *, max_context, mode):
    if mode not in ("decode", "extend", "verify"):
        raise ValueError("mode must be decode, extend or verify")
    if not 1 <= int(max_context) <= 1048576 or int(rows) <= 0:
        raise ValueError("invalid context or row capacity")
    splits = min(16, (max_context + 63) // 64) if mode != "extend" else 1
    chunks = ((max_context + 63) // 64 + splits - 1) // splits
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
        # One query per tile supports ragged batches in all three modes.
        self.forward = DenseMlaForwardKernel(layout=make_smem_layout(query_tile=1, fp8=False),
            page_size=64, num_heads=g.heads, num_splits=self.splits,
            chunks_per_split=self.layout["chunks_per_split"], query_tile=1, fp8=False,
            qk_dim=576, value_dim=512, window_size=None)
        self.merge = DenseMlaMergeKernel(num_splits=self.splits, value_dim=512) if self.splits > 1 else None

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, page_table: cute.Pointer,
                 cache_seqlens: cute.Pointer, cu_seqlens_q: cute.Pointer, attn: cute.Pointer,
                 lse: cute.Pointer, scratch: cute.Pointer, rows: Int32, batch: Int32,
                 page_table_width: Int64, active_splits: Int32, stream: cuda.CUstream):
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
            Float32(g.softmax_scale * math.log2(math.e)), Int64(g.heads * 576 * 2), Int64(576 * 2),
            Int64(64 * 1152), Int64(1152), page_table_width, rows, batch, active_splits, stream)
        if cutlass.const_expr(self.splits > 1):
            if active_splits > Int32(1):
                self.merge(po, pl, out, lse_out, active_splits, stream)


def compile_kimi_mla_producer_aot(g=KimiMlaGeometry(), *, max_rows=64):
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("w_qkv_a", torch.bfloat16, f"[{g.qkv_a_width},{g.hidden}]"),
        Operand("q_a_norm", torch.bfloat16, "[1536]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("w_q_b", torch.bfloat16, f"[{g.heads * 192},1536]"),
        Operand("w_uk", torch.bfloat16, f"[{g.heads},512,128]"),
        Operand("kv_cache", torch.bfloat16, "[pages,64,576]", "inout"),
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
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand("kv_cache", torch.bfloat16, "[pages,64,576]"),
        Operand("page_table", torch.int32, "[batch,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[batch]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out"),
        Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch"))
    return compile_program(_Attention(g, max_rows, max_context, mode), name="kimi_dense_mla",
        operands=operands, scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        key=(g, max_rows, max_context, mode),
        scratch={"scratch": lambda rows: attention_layout(g, rows, max_context=max_context, mode=mode)["scratch_bytes"]},
        geometry={**vars(g), **layout, "mode": mode, "max_context": max_context})
