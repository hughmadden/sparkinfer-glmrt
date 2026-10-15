"""Dense Qwen hybrid attention, sharing Flash Next GDN and Q/gate/RoPE kernels.

No mHC, PLE, QSA or expert geometry is fabricated. Full attention traverses all
causal KV pages, not a selected-position list. BF16 weights remain native;
9B GDN norm weights remain FP32 and 27B norm weights remain BF16.
"""
from __future__ import annotations

from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import QWEN35_9B, QwenDenseGeometry, Operand, Scalar, compile_program
from ._fp8_weights import projection
from ._mimo_kernels import MAX_SPLITS, MimoGqaAttention, MimoSplitMerge
from ._qwen4_attention_kernels import Qwen4AttnPost
from .qwen4_attention import compile_qwen4_attn_o_aot
from .qwen4_gdn import compile_qwen4_gdn_aot, compile_qwen4_gdn_commit_aot


def _align(n: int) -> int:
    return (int(n) + 1023) // 1024 * 1024


def _rows(max_rows: int) -> int:
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(max_rows)


def compile_qwen_dense_gdn_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int,
                               norm_dtype: torch.dtype):
    """Native BF16 projections and SiLU output, preserving the norm tensor dtype.

    Packed projection rows are [qkv, z, b, a], exactly as the shared GDN ABI.
    norm_dtype must come from the tensor header, not from hidden width.
    """
    program = compile_qwen4_gdn_aot(g, max_rows=max_rows, activation="silu", norm_dtype=norm_dtype)
    return replace(program, name="qwen_dense_gdn", geometry={
        **program.geometry, "output_gate": "silu", "norm_dtype": str(norm_dtype)})


def compile_qwen_dense_gdn_commit_aot(g: QwenDenseGeometry = QWEN35_9B):
    """Exact recurrent/convolution verify replay using the shared update arithmetic."""
    program = compile_qwen4_gdn_commit_aot(g)
    return replace(program, name="qwen_dense_gdn_commit")


class _Producer:
    def __init__(self, g: QwenDenseGeometry):
        self.g = g
        self.proj = projection(g.attn_in_width, g.hidden, False)
        self.post = Qwen4AttnPost(g, indexed=False)

    def key(self) -> tuple:
        return (self.g, self.proj.key(), "unrestricted")

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in: cute.Pointer, q_norm: cute.Pointer, k_norm: cute.Pointer,
                 rope_positions: cute.Pointer, kv_slots: cute.Pointer, kv_cache: cute.Pointer,
                 query: cute.Pointer, gate: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        proj = cute.make_ptr(cutlass.BFloat16, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.proj(x, w_in, w_in, w_in, proj, rows, stream)
        # No index items are launched, so these ABI placeholders are never read.
        self.post(proj, q_norm, k_norm, q_norm, k_norm, rope_positions, kv_slots, kv_cache,
                  query, query, gate, query, rows, stream)


def compile_qwen_dense_producer_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int):
    """Q/gate packed per head, zero-centered Q/K norm and checkpoint M-RoPE.

    kv_slots is an Int64 record offset (page*page_rows+in_page); negative
    slots suppress writes. The target owns causal positions and cache pages.
    """
    max_rows = _rows(max_rows)
    launch = _Producer(g)
    return compile_program(launch, name="qwen_dense_producer", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("w_in", torch.bfloat16, f"[{g.attn_in_width},{g.hidden}]"),
        Operand("q_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("k_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("rope_positions", torch.int32, "[rows,3]", align=4),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("kv_cache", torch.uint8, f"[record_slots,{g.record_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]", "out"),
        Operand("gate", torch.bfloat16, f"[rows,{g.heads * g.head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": g.hidden, "heads": g.heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
                  "in_width": g.attn_in_width, "record_bytes": g.record_bytes, "max_rows": max_rows,
                  "rope_dim": g.rope_dim, "rope_theta": g.rope_theta, "indexed": False},
        scratch={"scratch": lambda rows: _align(max(rows, 1) * g.attn_in_width * 2)})


def full_attention_scratch_bytes(g: QwenDenseGeometry, rows: int, max_splits: int) -> int:
    return (_align(max(rows, 1) * max_splits * g.heads * g.head_dim * 4)
            + _align(max(rows, 1) * max_splits * g.heads * 4))


class _FullAttention:
    def __init__(self, g: QwenDenseGeometry, max_splits: int):
        self.g, self.max_splits = g, max_splits
        # One query token per CTA permits arbitrary mixed lanes and verify widths.
        self.attn = MimoGqaAttention(heads=g.heads, kv_heads=g.kv_heads, tokens=1,
            window=0, paged=True, sink=False, direct=False, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, head=g.head_dim, v_head=g.head_dim)
        self.merge = MimoSplitMerge(heads=g.heads, v_head=g.head_dim)

    def key(self) -> tuple:
        return (self.g, self.max_splits, self.attn.key())

    @cute.jit
    def __call__(self, query: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer,
                 page_table: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        partial = cute.make_ptr(cutlass.Float32, base, cute.AddressSpace.gmem, assumed_align=16)
        size = Int64(rows) * Int64(self.max_splits * self.g.heads * self.g.head_dim * 4)
        lse = cute.make_ptr(cutlass.Float32, base + (size + Int64(1023)) // Int64(1024) * Int64(1024),
                            cute.AddressSpace.gmem, assumed_align=16)
        self.attn(query, kv_cache, kv_cache, positions, page_table, page_table, page_table, query,
                  out, partial, lse, rows, table_stride, splits, stream)
        self.merge(partial, lse, query, out, rows, splits, stream)


def compile_qwen_dense_full_gqa_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int, max_splits: int = 32):
    """Unrestricted causal paged GQA over K|V BF16 records, including long context.

    positions is Int64 per physical query. page_table is Int32 [rows,stride],
    or a single shared row when table_stride=0. All page-scaled addresses are
    Int64. Caller chooses 1 <= splits <= max_splits before capture.
    """
    max_rows = _rows(max_rows)
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    launch = _FullAttention(g, int(max_splits))
    return compile_program(launch, name="qwen_dense_full_gqa", operands=(
        Operand("query", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]"),
        Operand("kv_cache", torch.uint8, f"[record_slots,{g.record_bytes}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[full_attention_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"), Scalar("table_stride"), Scalar("splits")),
        key=(max_rows, launch.key()),
        geometry={"heads": g.heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
                  "page_rows": g.page_rows, "record_bytes": g.record_bytes, "max_rows": max_rows,
                  "max_splits": int(max_splits), "causal": True, "indexed": False},
        scratch={"scratch": lambda rows: full_attention_scratch_bytes(g, rows, int(max_splits))})


def compile_qwen_dense_attn_o_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int):
    """Reuse the same BF16 sigmoid output gate and output projection as Flash Next."""
    program = compile_qwen4_attn_o_aot(g, max_rows=max_rows)
    return replace(program, name="qwen_dense_attn_o")
