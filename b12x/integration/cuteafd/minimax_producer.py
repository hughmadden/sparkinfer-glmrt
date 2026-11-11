"""MiniMax producers composed from existing Gemma, NeoX and OAI arithmetic.

Norm materializes BF16 before the FP32 partial rotation. Index quantization
uses RN reciprocal then multiply (not x / scale), with scale=1 on zero rows.
The caller owns all scratch; live row counts never specialize a program.
Math/layout reference: Hugh Madden's MiniMax implementation.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.activation import gated_activation_value
from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _fp8_scale, _warp_max
from ._mimo_kernels import MimoQkvRope
from ._qwen4_attention_kernels import Qwen4AttnPost


class _Norm(Qwen4AttnPost):
    def __init__(self, heads):
        # Only _head128(..., rope=False) is used; Qwen's RoPE is not MiniMax's.
        self.heads, self.eps = int(heads), 1e-6

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        layout = cute.make_layout((m, self.heads, 128), stride=(self.heads * 128, 128, 1))
        self.norm(cute.make_tensor(x, layout), weight, cute.make_tensor(out, layout)).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def norm(self, x: cute.Tensor, weight: cute.Pointer, out: cute.Tensor):
        row, head = Int64(cute.arch.block_idx()[0]), Int64(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        self._head128(x[row, head, None], weight, x[row, head, None], lane,
                      out[row, head, None], False)


class _NormRope:
    def __init__(self, heads):
        self.heads = int(heads)
        self.norm = _Norm(heads)
        self.rope = MimoQkvRope(heads=heads, kv_heads=1, v_scale=1.0, head=128, v_head=128)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, positions: cute.Pointer,
                 cos_sin: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.norm(x, weight, scratch, rows, stream)
        m = Int64(rows)
        layout = cute.make_layout((m, self.heads * 128), stride=(self.heads * 128, 1))
        self.rotate(cute.make_tensor(scratch, layout), positions, cos_sin,
                    cute.make_tensor(out, layout)).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def rotate(self, src: cute.Tensor, positions: cute.Pointer, cos_sin: cute.Pointer, out: cute.Tensor):
        row, head = Int64(cute.arch.block_idx()[0]), Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        position = Int64(positions[row])
        cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                           cute.AddressSpace.gmem, assumed_align=4)
        base = head * Int64(128)
        self.rope._rope(src, row, base, out, row, base, Float32(cs[lane]),
                        Float32(cs[Int64(32) + lane]), lane)
        out[row, base + Int64(64) + lane] = src[row, base + Int64(64) + lane]
        out[row, base + Int64(96) + lane] = src[row, base + Int64(96) + lane]


class _Index:
    heads = 4

    def __init__(self):
        self.query, self.key = _NormRope(4), _NormRope(1)

    @cute.jit
    def __call__(self, iq: cute.Pointer, ik: cute.Pointer, iq_norm: cute.Pointer,
                 ik_norm: cute.Pointer, positions: cute.Pointer, slots: cute.Pointer,
                 cos_sin: cute.Pointer, q_fp8: cute.Pointer, q_scale: cute.Pointer,
                 k_fp8: cute.Pointer, k_scale: cute.Pointer, cache: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        # Normalized temporary followed by rotated Q and K, all BF16.
        rotated_q = scratch + m * Int64(4 * 128)
        rotated_k = rotated_q + m * Int64(4 * 128)
        self.query(iq, iq_norm, positions, cos_sin, rotated_q, scratch, rows, stream)
        self.key(ik, ik_norm, positions, cos_sin, rotated_k, scratch, rows, stream)
        self.pack(rotated_q, rotated_k, slots, q_fp8, q_scale, k_fp8, k_scale, cache).launch(
            grid=(rows, 5, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def pack(self, iq: cute.Pointer, ik: cute.Pointer, slots: cute.Pointer,
             q_fp8: cute.Pointer, q_scale: cute.Pointer, k_fp8: cute.Pointer,
             k_scale: cute.Pointer, cache: cute.Pointer):
        row, head = Int64(cute.arch.block_idx()[0]), Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        source = ik + row * Int64(128)
        if head < Int64(4):
            source = iq + (row * Int64(4) + head) * Int64(128)
        value = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        maximum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            value[j] = Float32(source[lane * Int64(4) + Int64(j)])
            maximum = cutlass.max(maximum, cute.math.abs(value[j]))
        scale = _fp8_scale(_warp_max(maximum))
        inverse = div_rn_f32(Float32(1.0), scale)
        packed = cvt_f32x4_to_e4m3x4(value[0] * inverse, value[1] * inverse,
                                    value[2] * inverse, value[3] * inverse)
        destination = Int64(k_fp8.toint()) + row * Int64(128)
        if head < Int64(4):
            destination = Int64(q_fp8.toint()) + (row * Int64(4) + head) * Int64(128)
            if lane == Int64(0):
                q_scale[row * Int64(4) + head] = scale
        else:
            if lane == Int64(0):
                k_scale[row] = scale
            slot = Int64(slots[row])
            if slot >= Int64(0):
                page, offset = slot // Int64(64), slot % Int64(64)
                page_base = Int64(cache.toint()) + page * Int64(64 * 132)
                word = cute.make_ptr(Uint32, page_base + offset * Int64(128) + lane * Int64(4),
                                     cute.AddressSpace.gmem, assumed_align=4)
                word[0] = packed
                if lane == Int64(0):
                    scales = cute.make_ptr(Float32, page_base + Int64(8192) + offset * Int64(4),
                                           cute.AddressSpace.gmem, assumed_align=4)
                    scales[0] = scale
        word = cute.make_ptr(Uint32, destination + lane * Int64(4),
                             cute.AddressSpace.gmem, assumed_align=4)
        word[0] = packed


class _OaiSplit:
    def __init__(self, width):
        self.width = int(width)

    @cute.jit
    def __call__(self, gate: cute.Pointer, up: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(gate, up, out, rows).launch(
            grid=((Int64(rows) * Int64(self.width) + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate: cute.Pointer, up: cute.Pointer, out: cute.Pointer, rows: Int32):
        idx = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if idx < Int64(rows) * Int64(self.width):
            out[idx] = gated_activation_value(Float32(gate[idx]), Float32(up[idx]),
                has_limit=True, limit_value=7.0, is_swigluoai=True,
                alpha=1.702, offset=1.0).to(BFloat16)


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def compile_minimax_norm_rope_aot(*, heads):
    _positive_int("heads", heads)
    return compile_program(_NormRope(heads), name="minimax_norm_rope", operands=(
        Operand("x", torch.bfloat16, f"[rows,{heads},128]"),
        Operand("weight", torch.bfloat16, "[128] Gemma offset"),
        Operand("positions", torch.int64, "[rows]"),
        Operand("cos_sin", torch.float32, "[positions,64] (cos32 then sin32)"),
        Operand("out", torch.bfloat16, f"[rows,{heads},128]", "out"),
        Operand("scratch", torch.bfloat16, f"[rows,{heads},128]", "scratch")),
        scalars=(Scalar("rows"),), key=(heads,),
        scratch={"scratch": lambda rows: int(rows) * heads * 128 * 2},
        geometry=dict(heads=heads, head_dim=128, rotary_dim=64, eps=1e-6))


def compile_minimax_index_producer_aot():
    return compile_program(_Index(), name="minimax_index_producer", operands=(
        Operand("iq", torch.bfloat16, "[rows,4,128]"),
        Operand("ik", torch.bfloat16, "[rows,128]"),
        Operand("iq_norm", torch.bfloat16, "[128] Gemma offset"),
        Operand("ik_norm", torch.bfloat16, "[128] Gemma offset"),
        Operand("positions", torch.int64, "[rows]"),
        Operand("slots", torch.int64, "[rows] negative skips paged write"),
        Operand("cos_sin", torch.float32, "[positions,64] (cos32 then sin32)"),
        Operand("q_fp8", torch.float8_e4m3fn, "[rows,4,128]", "out"),
        Operand("q_scale", torch.float32, "[rows,4]", "out"),
        Operand("k_fp8", torch.float8_e4m3fn, "[rows,128]", "out"),
        Operand("k_scale", torch.float32, "[rows]", "out"),
        Operand("cache", torch.uint8, "[pool_pages,8448] keys then FP32 scales", "inout"),
        Operand("scratch", torch.bfloat16, "[rows,9,128]", "scratch")),
        scalars=(Scalar("rows"),), key=(),
        scratch={"scratch": lambda rows: int(rows) * 9 * 128 * 2},
        geometry=dict(heads=4, head_dim=128, rotary_dim=64, eps=1e-6, page_rows=64))


def compile_minimax_oai_split_aot(*, width):
    _positive_int("width", width)
    return compile_program(_OaiSplit(width), name="minimax_oai_split", operands=(
        Operand("gate", torch.bfloat16, f"[rows,{width}]"),
        Operand("up", torch.bfloat16, f"[rows,{width}]"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,),
        geometry=dict(width=width, alpha=1.702, beta=1.0, limit=7.0))
