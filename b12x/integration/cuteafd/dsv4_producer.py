"""Native AOT DeepSeek V4 attention producer and C4 index-query producer.

``compile_dsv4_producer_aot(geometry, max_rows=...)`` is one program equal
to ``b12x.attention.dsv4_producer.run`` (fp8 cache format):

1. block-FP8 GEMM ``[wq_a; wkv]`` (per-token K128 activation quantization);
2. q_rank = RMSNorm(q_a) * q_norm; KV RMSNorm * kv_norm, RoPE, FP8 584-byte
   record written into the window/main cache row ``main_slots[row]``;
3. block-FP8 GEMM ``wq_b`` straight into ``query``;
4. per-head RMSNorm (no weight) + partial RoPE on ``query`` in place.

ABI (``H`` hidden, ``Q`` q_lora_rank, ``N`` heads; ``rows`` live tokens,
``rows <= max_rows``; ``P`` positions in the RoPE table)::

    hidden        bf16 [rows,H]                in
    positions     i64  [rows]                  in   RoPE position per row
    main_slots    i64  [rows]                  in   physical cache slot (page*256+row)
    cos_sin       f32  [P,64]                  in   cos(32) | sin(32)
    w_qkv         fp8  [Q+512,H]               in   pack_weights(...).qkv_rank.weight.values
    w_qkv_scale   u8   ceil((Q+512)/128)*ceil(H/128)*512 bytes   .qkv_rank.weight.scale_mma storage
    w_q           fp8  [N*512,Q]               in   .q.weight.values
    w_q_scale     u8   ceil(N*512/128)*ceil(Q/128)*512 bytes     .q.weight.scale_mma storage
    q_norm        bf16 [Q]                     in
    kv_norm       bf16 [512]                   in
    main_kv_cache u8   [pages,149760]          inout (only row main_slots[i] written)
    query         bf16 [rows,N,512]            out
    q_rank        bf16 [rows,Q]                out  (consumed by the index producer)
    scratch       u8   producer_scratch_bytes(geometry, rows, max_rows)
    rows          int32

``compile_dsv4_index_producer_aot(geometry, max_rows=...)`` equals
``dsv4_producer.run_indexer``: block-FP8 GEMM of the index ``wq_b`` over
``q_rank``, the BF16 head-weight projection ``hidden @ weights_proj^T``
(FP32 accumulate, BF16 out like ``torch.mm``: skinny GEMV for <= 160 live
rows, TMA tensor-core GEMM above), partial RoPE, Hadamard, FP4 QAT::

    q_rank        bf16 [rows,Q]                in
    hidden        bf16 [rows,H]                in
    positions     i64  [rows]                  in
    cos_sin       f32  [P,64]                  in
    w_q           fp8  [8192,Q]                in   pack_indexer_weights(...).q.weight.values
    w_q_scale     u8   64*ceil(Q/128)*512 bytes          .q.weight.scale_mma storage
    w_proj        bf16 [64,H]                  in   indexer.weights_proj.weight
    query         fp8  [rows,64,128]           out
    head_weights  f32  [rows,64]               out
    scratch       u8   index_producer_scratch_bytes(geometry, rows, max_rows)
    rows          int32

``max_rows`` is the planned capacity: it selects the GEMM tiles exactly as
the prepared ``gemm.block_fp8_linear`` plan of that capacity does. Scratch is
laid out from the live row count; size it with ``rows = max_rows``. Scratch
and every output must be disjoint from the inputs; the cache and scratch
bases must be 256-byte aligned.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from b12x.attention.dsv4_producer._cute import (
    DSV4IndexerQueryPost,
    DSV4QueryNormRope,
    DSV4RankNormPackKV,
)
from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program
from ._linear import Fp8LinearStage, block_fp8_lowering, stage_scratch_bytes

__all__ = [
    "compile_dsv4_index_producer_aot",
    "compile_dsv4_producer_aot",
    "index_producer_scratch_bytes",
    "producer_scratch_bytes",
]

_ALIGN = 1024
_INDEX_WIDTH = 64 * 128
_INDEX_WEIGHT_SCALE = (128 * 64) ** -0.5


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _slices(max_rows: int, k: int, n: int) -> int:
    return int(block_fp8_lowering(max_rows=max_rows, in_features=k, out_features=n).policy.split_k_slices)


def producer_scratch_bytes(geometry: DSV4Geometry, rows: int, max_rows: int) -> int:
    """qkv BF16 [rows, Q+512] then one reused block-FP8 stage region."""
    h, q, n = geometry.hidden, geometry.q_lora_rank, geometry.heads * geometry.head_dim
    qkv = _align(rows * (q + 512) * 2)
    stage = max(stage_scratch_bytes(h, q + 512, rows, _slices(max_rows, h, q + 512)),
                stage_scratch_bytes(q, n, rows, _slices(max_rows, q, n)))
    return qkv + stage


def index_producer_scratch_bytes(geometry: DSV4Geometry, rows: int, max_rows: int) -> int:
    """Raw index query BF16 [rows, 8192], raw head weights BF16 [rows, 64], stage."""
    q = geometry.q_lora_rank
    return (_align(rows * _INDEX_WIDTH * 2) + _align(rows * 64 * 2)
            + stage_scratch_bytes(q, _INDEX_WIDTH, rows, _slices(max_rows, q, _INDEX_WIDTH)))


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class _Producer:
    def __init__(self, geometry: DSV4Geometry, max_rows: int):
        self.h, self.q = geometry.hidden, geometry.q_lora_rank
        self.heads = geometry.heads
        self.qkv = Fp8LinearStage(in_features=self.h, out_features=self.q + 512, max_rows=max_rows)
        self.wq_b = Fp8LinearStage(in_features=self.q, out_features=self.heads * 512, max_rows=max_rows)
        self.pack = DSV4RankNormPackKV(q_rank=self.q, eps=geometry.norm_eps)
        self.query_post = DSV4QueryNormRope(heads=self.heads, eps=geometry.norm_eps)

    @cute.jit
    def __call__(self, hidden: cute.Pointer, positions: cute.Pointer, main_slots: cute.Pointer,
                 cos_sin: cute.Pointer, w_qkv: cute.Pointer, w_qkv_scale: cute.Pointer,
                 w_q: cute.Pointer, w_q_scale: cute.Pointer, q_norm: cute.Pointer,
                 kv_norm: cute.Pointer, main_kv_cache: cute.Pointer, query: cute.Pointer,
                 q_rank: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        base = Int64(scratch.toint())
        qkv = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        stage = base + _align_i64(Int64(rows) * Int64((self.q + 512) * 2))
        self.qkv(hidden, w_qkv, w_qkv_scale, qkv, stage, rows, stream)
        self.pack(qkv, q_norm, kv_norm, positions, main_slots, cos_sin, q_rank, main_kv_cache,
                  rows, stream)
        self.wq_b(q_rank, w_q, w_q_scale, query, stage, rows, stream)
        self.query_post(query, positions, cos_sin, rows, stream)


class _IndexProducer:
    def __init__(self, geometry: DSV4Geometry, max_rows: int):
        self.h, self.q = geometry.hidden, geometry.q_lora_rank
        self.wq_b = Fp8LinearStage(in_features=self.q, out_features=_INDEX_WIDTH, max_rows=max_rows)
        self.proj = RoutedBf16Projection(64, self.h)
        self.post = DSV4IndexerQueryPost(heads=64, weight_scale=_INDEX_WEIGHT_SCALE)

    @cute.jit
    def __call__(self, q_rank: cute.Pointer, hidden: cute.Pointer, positions: cute.Pointer,
                 cos_sin: cute.Pointer, w_q: cute.Pointer, w_q_scale: cute.Pointer,
                 w_proj: cute.Pointer, query: cute.Pointer, head_weights: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        m = Int64(rows)
        raw_query = cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16)
        weights_off = base + _align_i64(m * Int64(_INDEX_WIDTH * 2))
        raw_weights = cute.make_ptr(cutlass.BFloat16, weights_off, cute.AddressSpace.gmem, assumed_align=16)
        stage = weights_off + _align_i64(m * Int64(64 * 2))
        self.wq_b(q_rank, w_q, w_q_scale, raw_query, stage, rows, stream)
        self.proj(hidden, w_proj, raw_weights, rows, stream)
        self.post(raw_query, raw_weights, positions, cos_sin, query, head_weights, rows, stream)


def _check_capacity(max_rows: int) -> int:
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    return max_rows


def compile_dsv4_producer_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int):
    """Fused Q/KV producer for ``rows <= max_rows``; see module docstring."""
    max_rows = _check_capacity(max_rows)
    launch = _Producer(geometry, max_rows)
    h, q, n = geometry.hidden, geometry.q_lora_rank, geometry.heads
    operands = (
        Operand("hidden", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("main_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_qkv", torch.float8_e4m3fn, f"[{q + 512},{h}]"),
        Operand("w_qkv_scale", torch.uint8, f"[{(q + 512 + 127) // 128 * ((h + 127) // 128) * 512}]"),
        Operand("w_q", torch.float8_e4m3fn, f"[{n * 512},{q}]"),
        Operand("w_q_scale", torch.uint8, f"[{n * 4 * ((q + 127) // 128) * 512}]"),
        Operand("q_norm", torch.bfloat16, f"[{q}]"),
        Operand("kv_norm", torch.bfloat16, "[512]"),
        Operand("main_kv_cache", torch.uint8, "[pages,149760]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{n},512]", "out"),
        Operand("q_rank", torch.bfloat16, f"[rows,{q}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="dsv4_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(h, q, n, geometry.norm_eps, max_rows <= 8, launch.qkv.key(), launch.wq_b.key()),
        geometry={"hidden": h, "q_lora_rank": q, "heads": n, "max_rows": max_rows,
                  "cache_format": "fp8", "eps": geometry.norm_eps},
        scratch={"scratch": lambda rows: producer_scratch_bytes(geometry, rows, max_rows)},
        doc=__doc__,
    )


def compile_dsv4_index_producer_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int):
    """C4 index-query producer for ``rows <= max_rows``; see module docstring."""
    max_rows = _check_capacity(max_rows)
    launch = _IndexProducer(geometry, max_rows)
    h, q = geometry.hidden, geometry.q_lora_rank
    operands = (
        Operand("q_rank", torch.bfloat16, f"[rows,{q}]"),
        Operand("hidden", torch.bfloat16, f"[rows,{h}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_q", torch.float8_e4m3fn, f"[{_INDEX_WIDTH},{q}]"),
        Operand("w_q_scale", torch.uint8, f"[{64 * ((q + 127) // 128) * 512}]"),
        Operand("w_proj", torch.bfloat16, f"[64,{h}]"),
        Operand("query", torch.float8_e4m3fn, "[rows,64,128]", "out"),
        Operand("head_weights", torch.float32, "[rows,64]", "out"),
        Operand("scratch", torch.uint8, "[index_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="dsv4_index_producer", operands=operands, scalars=(Scalar("rows"),),
        key=(h, q, launch.wq_b.key(), launch.proj.key()),
        geometry={"hidden": h, "q_lora_rank": q, "max_rows": max_rows},
        scratch={"scratch": lambda rows: index_producer_scratch_bytes(geometry, rows, max_rows)},
        doc=__doc__,
    )
