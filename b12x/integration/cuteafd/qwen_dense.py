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
from ._qwen4_kernels import Qwen4Add, Qwen4HcNorm, Qwen4SwiGLU
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
    def __init__(self, g: QwenDenseGeometry, kv_format: str = "bf16"):
        self.g, self.kv_format = g, kv_format
        self.proj = projection(g.attn_in_width, g.hidden, False)
        self.post = Qwen4AttnPost(g, kv_format=kv_format, indexed=False)

    def key(self) -> tuple:
        return (self.g, self.kv_format, self.proj.key(), "unrestricted")

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


def compile_qwen_dense_producer_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int,
                                     kv_format: str = "bf16"):
    """Q/gate packed per head, zero-centered Q/K norm and checkpoint M-RoPE.

    kv_slots is an Int64 record offset (page*page_rows+in_page); negative
    slots suppress writes. The target owns causal positions and cache pages.
    """
    max_rows = _rows(max_rows)
    record_bytes = g.kv_record_bytes(kv_format)
    launch = _Producer(g, kv_format)
    return compile_program(launch, name="qwen_dense_producer", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("w_in", torch.bfloat16, f"[{g.attn_in_width},{g.hidden}]"),
        Operand("q_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("k_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("rope_positions", torch.int32, "[rows,3]", align=4),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("kv_cache", torch.uint8, f"[record_slots,{record_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]", "out"),
        Operand("gate", torch.bfloat16, f"[rows,{g.heads * g.head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": g.hidden, "heads": g.heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
                  "in_width": g.attn_in_width, "record_bytes": record_bytes, "kv_format": kv_format, "max_rows": max_rows,
                  "rope_dim": g.rope_dim, "rope_theta": g.rope_theta, "indexed": False},
        scratch={"scratch": lambda rows: _align(max(rows, 1) * g.attn_in_width * 2)})


def full_attention_scratch_bytes(g: QwenDenseGeometry, rows: int, max_splits: int) -> int:
    return (_align(max(rows, 1) * max_splits * g.heads * g.head_dim * 4)
            + _align(max(rows, 1) * max_splits * g.heads * 4))


class _FullAttention:
    def __init__(self, g: QwenDenseGeometry, max_splits: int, kv_format: str = "bf16", *, max_rows: int):
        self.g, self.max_splits = g, max_splits
        g.kv_record_bytes(kv_format)
        # Large programs serve one contiguous prefill; decode keeps mixed-lane
        # queries independent. Preserve the existing BF16 reduction geometry.
        tokens = 8 if kv_format == "fp8" and max_rows > 64 else 1
        self.attn = MimoGqaAttention(heads=g.heads, kv_heads=g.kv_heads, tokens=tokens,
            window=0, paged=True, sink=False, direct=False, softmax_scale=g.softmax_scale,
            page_rows=g.page_rows, head=g.head_dim, v_head=g.head_dim,
            kv8="e4m3" if kv_format == "fp8" else None, kv_group=0)
        self.merge = MimoSplitMerge(heads=g.heads, v_head=g.head_dim, empty_rows_zero=True)

    def key(self) -> tuple:
        return (self.g, self.max_splits, self.attn.key(), "empty_rows_zero")

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


def compile_qwen_dense_full_gqa_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int, max_splits: int = 32,
                                     kv_format: str = "bf16"):
    """Unrestricted causal paged GQA over BF16 or per-head E4M3 records.

    Compact records use the shared MiMo reader: on-chip reconstruction only,
    with no global BF16 shadow or context-sized widening allocation.

    positions is Int64 per physical query. page_table is Int32 [rows,stride],
    or a single shared row when table_stride=0. All page-scaled addresses are
    Int64. A negative position is an inactive query with zero output and no KV
    reads. Caller chooses 1 <= splits <= max_splits before capture.
    """
    max_rows = _rows(max_rows)
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    record_bytes = g.kv_record_bytes(kv_format)
    launch = _FullAttention(g, int(max_splits), kv_format, max_rows=max_rows)
    return compile_program(launch, name="qwen_dense_full_gqa", operands=(
        Operand("query", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]"),
        Operand("kv_cache", torch.uint8, f"[record_slots,{record_bytes}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[full_attention_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"), Scalar("table_stride"), Scalar("splits")),
        key=(max_rows, launch.key()),
        geometry={"heads": g.heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
                  "page_rows": g.page_rows, "record_bytes": record_bytes, "kv_format": kv_format, "max_rows": max_rows,
                  "max_splits": int(max_splits), "causal": True, "indexed": False,
                  "empty_rows_zero": True},
        scratch={"scratch": lambda rows: full_attention_scratch_bytes(g, rows, int(max_splits))})


def compile_qwen_dense_attn_o_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int):
    """Reuse the same BF16 sigmoid output gate and output projection as Flash Next."""
    program = compile_qwen4_attn_o_aot(g, max_rows=max_rows)
    return replace(program, name="qwen_dense_attn_o")


class _Norm:
    def __init__(self, g: QwenDenseGeometry):
        self.g = g
        # One ordinary residual stream; no hyper-connection mix/injection.
        self.norm = Qwen4HcNorm(g.hidden, 1, g.norm_eps, False)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.norm(x, x, x, weight, out, out, rows, stream)


def compile_qwen_dense_norm_aot(g: QwenDenseGeometry = QWEN35_9B):
    """Zero-centered RMSNorm, including final norm; FP32 (1+w) before BF16 cast."""
    return compile_program(_Norm(g), name="qwen_dense_norm", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("weight", torch.bfloat16, f"[{g.hidden}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out")),
        scalars=(Scalar("rows"),), key=(g,),
        geometry={"hidden": g.hidden, "eps": g.norm_eps, "zero_centered": True})


def compile_qwen_dense_add_aot(g: QwenDenseGeometry = QWEN35_9B):
    return compile_program(Qwen4Add(g.hidden), name="qwen_dense_add", operands=(
        Operand("a", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("b", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out")),
        scalars=(Scalar("rows"),), key=(g.hidden,), geometry={"hidden": g.hidden})


class _Ffn:
    def __init__(self, g: QwenDenseGeometry):
        self.g = g
        self.up = projection(2 * g.intermediate, g.hidden, False)
        self.act = Qwen4SwiGLU(g.intermediate, 2 * g.intermediate)
        self.down = projection(g.hidden, g.intermediate, False)

    def key(self):
        return (self.g, self.up.key(), self.down.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up: cute.Pointer, w_down: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        size = Int64(rows) * Int64(4 * self.g.intermediate)
        hidden = cute.make_ptr(cutlass.BFloat16,
            base + (size + Int64(1023)) // Int64(1024) * Int64(1024),
            cute.AddressSpace.gmem, assumed_align=16)
        gate_up = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        self.up(x, w_gate_up, w_gate_up, w_gate_up, gate_up, rows, stream)
        self.act(gate_up, hidden, rows, stream)
        self.down(hidden, w_down, w_down, w_down, out, rows, stream)


def compile_qwen_dense_ffn_aot(g: QwenDenseGeometry = QWEN35_9B, *, max_rows: int):
    max_rows = _rows(max_rows)
    launch = _Ffn(g)
    return compile_program(launch, name="qwen_dense_ffn", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        Operand("w_gate_up", torch.bfloat16, f"[{2 * g.intermediate},{g.hidden}]"),
        Operand("w_down", torch.bfloat16, f"[{g.hidden},{g.intermediate}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={"hidden": g.hidden, "intermediate": g.intermediate, "max_rows": max_rows},
        scratch={"scratch": lambda rows: _align(max(1, rows) * g.intermediate * 4)
                 + _align(max(1, rows) * g.intermediate * 2)})
