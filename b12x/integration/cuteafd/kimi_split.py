"""Projection-only seams for Kimi core quantization; fused BF16 exports stay unchanged.

All projected inputs/outputs are contiguous BF16 rows. KDA retains f_a/f_b/b
and calls the original body with its original workspace, recurrence and replay.
MLA retains the original rank norm/cache, UK/query tail and UV/gate multiply.
No entry applies an o projection or changes attention arithmetic.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int32, Int64

from ._common import Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, glm_projection
from ._glmf_kernels import kda_replay_layout
from ._kda_contract import KDAGeometry
from ._qwen4_attention_kernels import Qwen4GateMul
from .glmf import (_Kda, _check_rows, _kda_state, _ptr, _state_geometry,
                   _replay_geometry, kda_scratch_bytes)
from .kimi_mla import KimiMlaGeometry, _RankNormPack, _QueryTail, _align


class _NoProjection:
    @cute.jit
    def run(self, x: cute.Pointer, w: cute.Pointer, w_fp8: cute.Pointer,
            scale: cute.Pointer, out: cute.Pointer, rows: Int32,
            fp8_rows: Int32, qscratch: Int64, stream: cuda.CUstream):
        pass


class _KdaPack:
    def __init__(self, g):
        self.d, self.hd, self.heads, self.p = g.kda_width, g.kda_head_dim, g.kda_heads, g.kda_in_width

    @cute.jit
    def __call__(self, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 gate: cute.Pointer, fab: cute.Pointer, proj: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(q, k, v, gate, fab, proj).launch(
            grid=((self.p + 255) // 256, rows, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
               gate: cute.Pointer, fab: cute.Pointer, proj: cute.Pointer):
        tile, row, _ = cute.arch.block_idx()
        col = Int64(tile) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        r, d, hd = Int64(row), Int64(self.d), Int64(self.hd)
        if col < Int64(self.p):
            value = BFloat16(0)
            if col < d:
                value = q[r * d + col]
            elif col < Int64(2) * d:
                value = k[r * d + col - d]
            elif col < Int64(3) * d:
                value = v[r * d + col - Int64(2) * d]
            elif col < Int64(3) * d + hd:
                value = fab[r * Int64(self.hd + self.heads) + col - Int64(3) * d]
            elif col < Int64(4) * d + hd:
                value = gate[r * d + col - Int64(3) * d - hd]
            else:
                value = fab[r * Int64(self.hd + self.heads) + col - Int64(4) * d]
            proj[r * Int64(self.p) + col] = value


class _KdaCore:
    def __init__(self, g, max_rows, decode, state_dtype, state_rounding):
        self.core = _Kda(g, max_rows, state_dtype=state_dtype,
                         state_rounding=state_rounding, decode=decode)
        self.core.in_proj = _NoProjection()
        self.core.output_norm = True
        self.fab = glm_projection(g.kda_head_dim + g.kda_heads, g.hidden)
        self.pack = _KdaPack(g)
        self.g = g
        self.fab_offset = _align(kda_scratch_bytes(g, max_rows, self.core.chunked_bytes))

    @cute.jit
    def __call__(self, x: cute.Pointer, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 gate: cute.Pointer, w_fab: cute.Pointer, w_f: cute.Pointer,
                 conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                 o_norm: cute.Pointer, conv_state: cute.Pointer, state: cute.Pointer,
                 slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer,
                 replay: cute.Pointer, scratch: cute.Pointer, rows: Int32, spec: Int32,
                 stream: cuda.CUstream):
        fab = _ptr(BFloat16, Int64(scratch.toint()) + Int64(self.fab_offset))
        self.fab(x, w_fab, fab, rows, stream)
        self.pack(q, k, v, gate, fab, cute.recast_ptr(scratch, dtype=BFloat16), rows, stream)
        self.core.body(x, w_fab, w_fab, w_fab, w_f, conv_w, a_log, dt_bias,
            o_norm, w_fab, w_fab, w_fab, conv_state, state, slots, seq_first,
            out, replay, scratch, rows, Int32(0), spec, stream)


def compile_kimi_kda_core_aot(g=KDAGeometry(hidden=7168, kda_heads=48), *,
                               max_rows=64, decode=True, state_dtype="float32",
                               state_rounding="window"):
    """q/k/v/g are independent [rows,D] BF16 projections; f_a and b stay native.

    w_fab is cat(f_a,b) [128+heads,H], w_f is f_b [1,D,128]. Other operands
    are the fused norm-output KDA ABI. The replay pointer is always present;
    prefill uses spec=0 and decode supports the unchanged speculative replay.
    """
    max_rows = _check_rows(max_rows)
    if not isinstance(g, KDAGeometry) or g.output_gate != "full_rank":
        raise ValueError("Kimi split KDA requires full-rank KDAGeometry")
    state_torch = _kda_state(state_dtype, state_rounding)[1]
    launch = _KdaCore(g, max_rows, decode, state_dtype, state_rounding)
    h, d, heads = g.hidden, g.kda_width, g.kda_heads
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        *(Operand(n, torch.bfloat16, f"[rows,{d}]") for n in ("q", "k", "v", "gate")),
        Operand("w_fab", torch.bfloat16, f"[{128 + heads},{h}]"),
        Operand("w_f", torch.bfloat16, f"[1,{d},128]"),
        Operand("conv_w", torch.float32, f"[{3 * d},4]", align=4),
        Operand("a_log", torch.float32, f"[{heads}]", align=4),
        Operand("dt_bias", torch.float32, f"[{d}]", align=4),
        Operand("o_norm", getattr(torch, g.o_norm_dtype), "[128]"),
        Operand("conv_state", torch.bfloat16, f"[slots,3,{3 * d}]", "inout", align=2),
        Operand("state", state_torch, f"[slots,{heads},128,128]", "inout"),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
        Operand("replay", torch.float32,
                f"[{kda_replay_layout(heads, 3 * d, launch.core.replay_rows)[2] // 4}]", "inout"),
        Operand("scratch", torch.uint8, "[split_kda_scratch_bytes]", "scratch"))
    return compile_program(launch, name="kimi_kda_core", operands=operands,
        scalars=(Scalar("rows"), Scalar("spec")),
        key=(g, max_rows, decode, state_dtype, state_rounding, launch.fab.key()),
        scratch={"scratch": lambda rows: launch.fab_offset + _align(rows * (128 + heads) * 2)},
        geometry={**vars(g), "max_rows": max_rows, "decode": decode, "output_kind": "norm",
                  "fab_offset": launch.fab_offset, **_state_geometry(state_dtype, state_rounding),
                  **_replay_geometry(launch.core)})


class _RankPack:
    def __init__(self, kv_stride):
        self.kv_stride = kv_stride

    @cute.jit
    def __call__(self, q_a: cute.Pointer, kv_a: cute.Pointer, packed: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(q_a, kv_a, packed).launch(grid=(9, rows, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q_a: cute.Pointer, kv_a: cute.Pointer, packed: cute.Pointer):
        tile, row, _ = cute.arch.block_idx()
        col = Int64(tile) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        r = Int64(row)
        if col < Int64(2112):
            value = BFloat16(0)
            if col < Int64(1536):
                value = q_a[r * Int64(1536) + col]
            else:
                value = kv_a[r * Int64(self.kv_stride) + col - Int64(1536)]
            packed[r * Int64(2112) + col] = value


class _MlaRankNorm:
    def __init__(self, g, kv_stride):
        self.pack, self.norm = _RankPack(kv_stride), _RankNormPack(g)

    @cute.jit
    def __call__(self, q_a: cute.Pointer, kv_a: cute.Pointer, q_a_norm: cute.Pointer,
                 kv_a_norm: cute.Pointer, kv_slots: cute.Pointer, q_resid: cute.Pointer,
                 kv_cache: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        packed = cute.recast_ptr(scratch, dtype=BFloat16)
        self.pack(q_a, kv_a, packed, rows, stream)
        self.norm(packed, q_a_norm, kv_a_norm, kv_slots, q_resid, kv_cache, rows, stream)


class _MlaQuery:
    def __init__(self, g):
        self.uk = BatchedBf16Gemm(n=512, k=128, batch=g.heads, a_row=g.heads * 192,
                                 a_batch=192, o_row=g.heads * 576, o_batch=576)
        self.tail = _QueryTail(g.heads)

    @cute.jit
    def __call__(self, q: cute.Pointer, w_uk: cute.Pointer, query: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.uk(q, w_uk, query, rows, stream)
        self.tail(q, query, rows, stream)


class _MlaOutput:
    def __init__(self, g):
        self.uv = BatchedBf16Gemm(n=128, k=512, batch=g.heads, a_row=g.heads * 512,
                                 a_batch=512, o_row=g.heads * 128, o_batch=128)
        self.gate_mul = Qwen4GateMul(g.heads * 128)

    @cute.jit
    def __call__(self, attn: cute.Pointer, w_uv: cute.Pointer, gates: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.uv(attn, w_uv, out, rows, stream)
        self.gate_mul(out, gates, out, rows, stream)


def _mla_geometry(g, max_rows):
    if type(g) is not KimiMlaGeometry:
        raise ValueError("split MLA requires KimiMlaGeometry (NoPE)")
    return {**vars(g), "max_rows": _check_rows(max_rows)}


def compile_kimi_mla_rank_norm_aot(g=KimiMlaGeometry(), *, max_rows=64, kv_stride=576):
    """Separate q_a [rows,1536], kv_a [rows,576|640]; padded kv tail is ignored."""
    geometry = _mla_geometry(g, max_rows)
    if kv_stride not in (576, 640):
        raise ValueError("kv_stride must be logical 576 or padded 640")
    operands = (
        Operand("q_a", torch.bfloat16, "[rows,1536]"),
        Operand("kv_a", torch.bfloat16, f"[rows,{kv_stride}]"),
        Operand("q_a_norm", torch.bfloat16, "[1536]"),
        Operand("kv_a_norm", torch.bfloat16, "[512]"),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("q_resid", torch.bfloat16, "[rows,1536]", "out"),
        Operand("kv_cache", torch.bfloat16, "[pages,64,576]", "inout"),
        Operand("scratch", torch.uint8, "[align(rows*2112*2)]", "scratch"))
    return compile_program(_MlaRankNorm(g, kv_stride), name="kimi_mla_rank_norm",
        operands=operands, scalars=(Scalar("rows"),), key=(g, max_rows, kv_stride),
        scratch={"scratch": lambda rows: _align(rows * 2112 * 2)},
        geometry={**geometry, "kv_stride": kv_stride})


def compile_kimi_mla_query_aot(g=KimiMlaGeometry(), *, max_rows=64):
    operands = (
        Operand("q", torch.bfloat16, f"[rows,{g.heads},192]"),
        Operand("w_uk", torch.bfloat16, f"[{g.heads},512,128]"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]", "out"))
    return compile_program(_MlaQuery(g), name="kimi_mla_query", operands=operands,
        scalars=(Scalar("rows"),), key=(g, max_rows), geometry=_mla_geometry(g, max_rows))


def compile_kimi_mla_gate_output_aot(g=KimiMlaGeometry(), *, max_rows=64):
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]"),
        Operand("w_uv", torch.bfloat16, f"[{g.heads},128,512]"),
        Operand("gates", torch.bfloat16, f"[rows,{g.heads},128]"),
        Operand("out", torch.bfloat16, f"[rows,{g.heads},128]", "out"))
    return compile_program(_MlaOutput(g), name="kimi_mla_gate_output", operands=operands,
        scalars=(Scalar("rows"),), key=(g, max_rows), geometry=_mla_geometry(g, max_rows))
