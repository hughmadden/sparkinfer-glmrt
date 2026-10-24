"""Shared native pointwise boundaries with explicit operand precision.

No allocations, live-size specialization, or checkpoint conversions at launch.
The dense residual remains F32; BF16 materialization is explicit at projection,
Q/K norm -> RoPE, sigmoid gate, and post-SiLU product boundaries. Indexer norm
parameters may be F32 on disk but must enter these programs as BF16 operands.
"""
from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4, div_rn_f32, ld_global_v4_f32, pack_f32x2_to_bfloat2, rsqrt_approx_ftz_f32,
    st_global_v4_u32,
)
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _bf16, _fp8_scale, _warp_max, _warp_sum
from ._fp8_moe_kernels import _post_silu_product
from ._qwen4_kernels import _sigmoid


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip)], "mul.rn.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip)], "add.rn.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _sub(a: Float32, b: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip)], "sub.rn.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip), Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _view(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


def _positive(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _dtype(value):
    if value not in (torch.float32, torch.bfloat16):
        raise ValueError("pointwise dtype must be float32 or bfloat16")
    return value


class AffineNorm:
    """One warp per head; native D192 Q/K uses the deployed fixed lane fold.

    Inputs/weights are typed by the ABI. D192/R64: 16 logical lanes, each
    three adjacent four-value vectors. D192/R192: 32 lanes, eight contiguous
    values with zero padding. Both use ascending XOR reduction and FMA lanes.
    Other norms use a fixed striped scalar sum independent of live rows.
    """
    def __init__(self, width, heads, eps, zero_centered, subtract_mean, bias, rotary):
        if width <= 0 or heads <= 0 or rotary < 0 or rotary > width or rotary % 2:
            raise ValueError("invalid affine norm geometry")
        if width == 192 and rotary not in (0, 64, 192):
            raise ValueError("D192 Q/K rotary span must be 64 or 192")
        _positive("epsilon", eps)
        self.width, self.heads, self.eps = int(width), int(heads), float(eps)
        self.zero_centered, self.subtract_mean, self.bias = zero_centered, subtract_mean, bias
        self.rotary = int(rotary)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 cos: cute.Pointer, sin: cute.Pointer, positions: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, weight, bias, cos, sin, positions, out).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def _normalized(self, x: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor,
                    base: Int64, dim: Int32, mean: Float32, inv: Float32):
        w = Float32(weight[dim])
        if cutlass.const_expr(self.zero_centered):
            w = _add(w, Float32(1.0))
        value = _mul(_mul(_sub(Float32(x[base + Int64(dim)]), mean), inv), w)
        if cutlass.const_expr(self.bias):
            value = _add(value, Float32(bias[dim]))
        return _bf16(value)

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
               cos: cute.Pointer, sin: cute.Pointer, positions: cute.Pointer, out: cute.Pointer):
        x, weight, bias = _view(x), _view(weight), _view(bias)
        cos, sin, positions, out = _view(cos), _view(sin), _view(positions), _view(out)
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        base = (row * Int64(self.heads) + head) * Int64(self.width)
        squares, total = Float32(0.0), Float32(0.0)
        if cutlass.const_expr(self.width == 192 and self.rotary == 64):
            if lane < Int32(16):
                for block in cutlass.range_constexpr(3):
                    for element in cutlass.range_constexpr(4):
                        dim = Int64(block * 64 + element) + Int64(lane) * Int64(4)
                        v = Float32(x[base + dim])
                        squares = _fma(v, v, squares)
        elif cutlass.const_expr(self.width == 192 and self.rotary == 192):
            for element in cutlass.range_constexpr(8):
                dim = lane * Int32(8) + Int32(element)
                v = Float32(0.0)
                if dim < Int32(192):
                    v = Float32(x[base + Int64(dim)])
                squares = _fma(v, v, squares)
        else:
            for chunk in cutlass.range_constexpr((self.width + 31) // 32):
                dim = Int32(chunk * 32) + lane
                if dim < Int32(self.width):
                    v = Float32(x[base + Int64(dim)])
                    squares = _add(squares, _mul(v, v))
                    total = _add(total, v)
        squares = _warp_sum(squares)
        mean = Float32(0.0)
        variance = div_rn_f32(squares, Float32(self.width))
        if cutlass.const_expr(self.subtract_mean):
            mean = div_rn_f32(_warp_sum(total), Float32(self.width))
            variance = _sub(variance, _mul(mean, mean))
        inv = rsqrt_approx_ftz_f32(_add(variance, Float32(self.eps)))
        for chunk in cutlass.range_constexpr((self.width + 31) // 32):
            dim = Int32(chunk * 32) + lane
            if dim < Int32(self.width):
                value = self._normalized(x, weight, bias, base, dim, mean, inv)
                if cutlass.const_expr(self.rotary > 0):
                    half = Int32(self.rotary // 2)
                    if dim < Int32(self.rotary):
                        pair = dim % half
                        real = self._normalized(x, weight, bias, base, pair, mean, inv)
                        imag = self._normalized(x, weight, bias, base, pair + half, mean, inv)
                        position = Int64(positions[row])
                        cache_at = position * Int64(self.rotary // 2) + Int64(pair)
                        c, s = Float32(cos[cache_at]), Float32(sin[cache_at])
                        if dim < half:
                            value = _sub(_mul(real, c), _mul(imag, s))
                        else:
                            value = _fma(real, s, _mul(imag, c))
                out[base + Int64(dim)] = value.to(BFloat16)


class RuntimeAffineNorm(AffineNorm):
    """Runtime epsilon ABI; keep the legacy constant kernel unchanged for AOT users."""

    def __init__(self, width, heads, eps, zero_centered, subtract_mean, bias, rotary,
                 vector_io=False):
        super().__init__(width, heads, eps, zero_centered, subtract_mean, bias, rotary)
        self.vector_io = bool(vector_io)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 cos: cute.Pointer, sin: cute.Pointer, positions: cute.Pointer, out: cute.Pointer,
                 rows: Int32, epsilon: Float32, stream: cuda.CUstream):
        self.kernel(x, weight, bias, cos, sin, positions, out, epsilon).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def _wide(self, x_addr: Int64, w_addr: Int64, out_addr: Int64, x: cute.Tensor,
              weight: cute.Tensor, out: cute.Tensor, base: Int64, lane: Int32,
              epsilon: Float32):
        # Keep the reduction's striped operands in registers before the chain;
        # the per-lane accumulation order is the generic path's, so the
        # reduction stays bit-identical.
        held = []
        squares, total = Float32(0.0), Float32(0.0)
        for chunk in cutlass.range_constexpr(self.width // 32):
            held.append(Float32(x[base + Int64(chunk * 32) + Int64(lane)]))
        for value in held:
            squares = _add(squares, _mul(value, value))
            if cutlass.const_expr(self.subtract_mean):
                total = _add(total, value)
        squares = _warp_sum(squares)
        mean = Float32(0.0)
        variance = div_rn_f32(squares, Float32(self.width))
        if cutlass.const_expr(self.subtract_mean):
            mean = div_rn_f32(_warp_sum(total), Float32(self.width))
            variance = _sub(variance, _mul(mean, mean))
        inv = rsqrt_approx_ftz_f32(_add(variance, epsilon))
        self._vector_rows(x_addr, w_addr, out_addr, x, weight, out, base, lane, mean, inv)

    @cute.jit
    def _vector_rows(self, x_addr: Int64, w_addr: Int64, out_addr: Int64, x: cute.Tensor,
                     weight: cute.Tensor, out: cute.Tensor, base: Int64, lane: Int32,
                     mean: Float32, inv: Float32):
        # Per-lane contiguous vector chunks; the per-element math and its
        # rounding are the scalar path's.  compile_program declares every
        # operand pointer 16-byte aligned, and width % 256 == 0 makes each row
        # base a multiple of 1024 B (F32) / 512 B (BF16), so this chunk's
        # accesses are 16-byte aligned.
        for chunk in cutlass.range_constexpr(self.width // 256):
            first = Int64(chunk * 256) + Int64(lane) * Int64(8)
            x0, x1, x2, x3 = ld_global_v4_f32(x_addr + (base + first) * Int64(4))
            x4, x5, x6, x7 = ld_global_v4_f32(x_addr + (base + first) * Int64(4) + Int64(16))
            w0, w1, w2, w3 = ld_global_v4_f32(w_addr + first * Int64(4))
            w4, w5, w6, w7 = ld_global_v4_f32(w_addr + first * Int64(4) + Int64(16))
            normalized = []
            for value, weight_value in ((x0, w0), (x1, w1), (x2, w2), (x3, w3),
                                        (x4, w4), (x5, w5), (x6, w6), (x7, w7)):
                if cutlass.const_expr(self.zero_centered):
                    weight_value = _add(weight_value, Float32(1.0))
                normalized.append(_mul(_mul(_sub(value, mean), inv), weight_value))
            st_global_v4_u32(out_addr + (base + first) * Int64(2),
                pack_f32x2_to_bfloat2(normalized[0], normalized[1]),
                pack_f32x2_to_bfloat2(normalized[2], normalized[3]),
                pack_f32x2_to_bfloat2(normalized[4], normalized[5]),
                pack_f32x2_to_bfloat2(normalized[6], normalized[7]))

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
               cos: cute.Pointer, sin: cute.Pointer, positions: cute.Pointer, out: cute.Pointer, epsilon: Float32):
        x_addr, w_addr, out_addr = (Int64(x.toint()), Int64(weight.toint()),
                                    Int64(out.toint()))
        x, weight, bias = _view(x), _view(weight), _view(bias)
        cos, sin, positions, out = _view(cos), _view(sin), _view(positions), _view(out)
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        base = (row * Int64(self.heads) + head) * Int64(self.width)
        wide = (self.vector_io and self.rotary == 0 and not self.bias
                and self.width % 256 == 0)
        if cutlass.const_expr(wide):
            self._wide(x_addr, w_addr, out_addr, x, weight, out, base, lane, epsilon)
        else:
            squares, total = Float32(0.0), Float32(0.0)
            if cutlass.const_expr(self.width == 192 and self.rotary == 64):
                if lane < Int32(16):
                    for block in cutlass.range_constexpr(3):
                        for element in cutlass.range_constexpr(4):
                            dim = Int64(block * 64 + element) + Int64(lane) * Int64(4)
                            v = Float32(x[base + dim])
                            squares = _fma(v, v, squares)
            elif cutlass.const_expr(self.width == 192 and self.rotary == 192):
                for element in cutlass.range_constexpr(8):
                    dim = lane * Int32(8) + Int32(element)
                    v = Float32(0.0)
                    if dim < Int32(192):
                        v = Float32(x[base + Int64(dim)])
                    squares = _fma(v, v, squares)
            else:
                for chunk in cutlass.range_constexpr((self.width + 31) // 32):
                    dim = Int32(chunk * 32) + lane
                    if dim < Int32(self.width):
                        v = Float32(x[base + Int64(dim)])
                        squares = _add(squares, _mul(v, v))
                        total = _add(total, v)
            squares = _warp_sum(squares)
            mean = Float32(0.0)
            variance = div_rn_f32(squares, Float32(self.width))
            if cutlass.const_expr(self.subtract_mean):
                mean = div_rn_f32(_warp_sum(total), Float32(self.width))
                variance = _sub(variance, _mul(mean, mean))
            inv = rsqrt_approx_ftz_f32(_add(variance, epsilon))
            for chunk in cutlass.range_constexpr((self.width + 31) // 32):
                dim = Int32(chunk * 32) + lane
                if dim < Int32(self.width):
                    value = self._normalized(x, weight, bias, base, dim, mean, inv)
                    if cutlass.const_expr(self.rotary > 0):
                        half = Int32(self.rotary // 2)
                        if dim < Int32(self.rotary):
                            pair = dim % half
                            real = self._normalized(x, weight, bias, base, pair, mean, inv)
                            imag = self._normalized(x, weight, bias, base, pair + half, mean, inv)
                            position = Int64(positions[row])
                            cache_at = position * Int64(self.rotary // 2) + Int64(pair)
                            c, s = Float32(cos[cache_at]), Float32(sin[cache_at])
                            if dim < half:
                                value = _sub(_mul(real, c), _mul(imag, s))
                            else:
                                value = _fma(real, s, _mul(imag, c))
                    out[base + Int64(dim)] = value.to(BFloat16)


class ResidualAdd:
    def __init__(self, width):
        self.width = int(width)

    @cute.jit
    def __call__(self, residual: cute.Pointer, delta: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(residual, delta, out, Int64(rows) * Int64(self.width)).launch(
            grid=((Int64(rows) * Int64(self.width) + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, residual: cute.Pointer, delta: cute.Pointer, out: cute.Pointer, count: Int64):
        residual, delta, out = _view(residual), _view(delta), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            out[i] = _add(Float32(residual[i]), Float32(delta[i]))


class HeadGate:
    def __init__(self, heads, dim):
        self.heads, self.dim = int(heads), int(dim)

    @cute.jit
    def __call__(self, x: cute.Pointer, logits: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, logits, out).launch(grid=(rows, self.heads, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, logits: cute.Pointer, out: cute.Pointer):
        x, logits, out = _view(x), _view(logits), _view(out)
        head = Int64(cute.arch.block_idx()[0]) * Int64(self.heads) + Int64(cute.arch.block_idx()[1])
        gate = _bf16(_sigmoid(Float32(logits[head])))
        for chunk in cutlass.range_constexpr((self.dim + 255) // 256):
            dim = Int64(chunk * 256) + Int64(cute.arch.thread_idx()[0])
            if dim < Int64(self.dim):
                out[head * Int64(self.dim) + dim] = _mul(Float32(x[head * Int64(self.dim) + dim]), gate).to(BFloat16)


class RouteWeights:
    """Shared selector owns ids; retain the subtract-after-add mixture rounding."""
    def __init__(self, experts, topk, epsilon, factor):
        self.experts, self.topk = int(experts), int(topk)
        self.epsilon, self.factor = float(epsilon), float(factor)

    @cute.jit
    def __call__(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(logits, bias, ids, out).launch(grid=(rows, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer, out: cute.Pointer):
        logits, bias, ids, out = _view(logits), _view(bias), _view(ids), _view(out)
        row = Int64(cute.arch.block_idx()[0])
        if Int32(cute.arch.thread_idx()[0]) == Int32(0):
            values = cute.make_rmem_tensor(cute.make_layout((self.topk,), stride=(1,)), Float32)
            denominator = Float32(0.0)
            for slot in cutlass.range_constexpr(self.topk):
                expert = Int64(ids[row * Int64(self.topk) + Int64(slot)])
                b = Float32(bias[expert])
                p = _sigmoid(Float32(logits[row * Int64(self.experts) + expert]))
                values[slot] = _sub(_add(p, b), b)
                denominator = _add(denominator, values[slot])
            denominator = _add(denominator, Float32(self.epsilon))
            for slot in cutlass.range_constexpr(self.topk):
                out[row * Int64(self.topk) + Int64(slot)] = _mul(div_rn_f32(values[slot], denominator), Float32(self.factor))


class PostSilu:
    def __init__(self, width, limit):
        self.width, self.limit = int(width), float(limit)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(gate_up, out, Int64(rows) * Int64(self.width)).launch(
            grid=((Int64(rows) * Int64(self.width) + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Pointer, out: cute.Pointer, count: Int64):
        gate_up, out = _view(gate_up), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            row, col = i // Int64(self.width), i % Int64(self.width)
            base = row * Int64(2 * self.width) + col
            out[i] = Float32(_post_silu_product(Float32(gate_up[base]),
                Float32(gate_up[base + Int64(self.width)]), self.limit)).to(BFloat16)


class ScaleProjection:
    def __init__(self, width, outputs, prescale):
        from ._glm_kernels import glm_projection

        self.outputs, self.prescale = int(outputs), float(prescale)
        self.projection = glm_projection(outputs, width)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.projection(x, weight, scratch, rows, stream)
        self.scale(scratch, out, Int64(rows) * Int64(self.outputs)).launch(
            grid=((Int64(rows) * Int64(self.outputs) + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def scale(self, projected: cute.Pointer, out: cute.Pointer, count: Int64):
        projected, out = _view(projected), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            out[i] = _mul(Float32(projected[i]), Float32(self.prescale))


def fp8_kv_plane_bytes(kv_heads, head_dim):
    """One planar K or V token: E4M3 heads, FP32 descales, zero alignment tail."""
    if kv_heads <= 0 or head_dim not in (192, 256):
        raise ValueError("positive KV heads and D192/D256 required")
    return (int(kv_heads) * (int(head_dim) + 4) + 15) // 16 * 16


class PackFp8KvPlane:
    """Qwen's per-token/head FP8 contract with a planar, aligned storage owner."""

    def __init__(self, kv_heads, head_dim):
        self.heads, self.dim = int(kv_heads), int(head_dim)
        self.row_bytes = fp8_kv_plane_bytes(kv_heads, head_dim)
        self.payload_bytes = self.heads * self.dim
        self.values_per_lane = self.dim // 32

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, out).launch(grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        source = _view(x)
        values = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        base = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(self.dim)
        amax = Float32(0.0)
        for item in cutlass.range_constexpr(self.values_per_lane):
            values[item] = Float32(source[base + Int64(lane) + Int64(item * 32)])
            amax = cutlass.max(amax, cute.math.abs(values[item]))
        scale = _fp8_scale(_warp_max(amax))
        inverse = div_rn_f32(Float32(1.0), scale)
        row_address = Int64(out.toint()) + Int64(row) * Int64(self.row_bytes)
        if lane == Int32(0):
            scales = cute.make_tensor(cute.make_ptr(Float32,
                row_address + Int64(self.payload_bytes), cute.AddressSpace.gmem, assumed_align=4),
                cute.make_layout((self.heads,)))
            scales[head] = scale
        words = cute.make_tensor(cute.make_ptr(Uint32,
            row_address + Int64(head) * Int64(self.dim), cute.AddressSpace.gmem, assumed_align=4),
            cute.make_layout((self.dim // 4,)))
        # A lane owns one dimension in each 32-wide stripe; shuffle groups of four
        # into one packed store without requiring D192 to divide into eight lanes.
        for item in cutlass.range_constexpr(self.values_per_lane):
            value = values[item] * inverse
            v1 = cute.arch.shuffle_sync_bfly(value, offset=1)
            v2 = cute.arch.shuffle_sync_bfly(value, offset=2)
            v3 = cute.arch.shuffle_sync_bfly(v1, offset=2)
            if lane % Int32(4) == Int32(0):
                words[(lane + Int32(item * 32)) // Int32(4)] = cvt_f32x4_to_e4m3x4(value, v1, v2, v3)
        if head == Int32(0):
            tail = self.row_bytes - self.payload_bytes - self.heads * 4
            if lane < Int32(tail):
                padding = cute.make_tensor(cute.make_ptr(Uint8,
                    row_address + Int64(self.payload_bytes + self.heads * 4),
                    cute.AddressSpace.gmem, assumed_align=1), cute.make_layout((max(1, tail),)))
                padding[lane] = Uint8(0)


def compile_pack_fp8_kv_plane_aot(*, kv_heads, head_dim=192):
    launch = PackFp8KvPlane(kv_heads, head_dim)
    return compile_program(launch, name="pack_fp8_kv_plane", operands=(
        Operand("x", torch.bfloat16, f"[rows,{kv_heads},{head_dim}]"),
        Operand("out", torch.uint8, f"[rows,{launch.row_bytes}]", "out")),
        scalars=(Scalar("rows"),), key=(int(kv_heads), int(head_dim)),
        geometry=dict(kv_heads=int(kv_heads), head_dim=int(head_dim), kv_format="fp8",
                      record_bytes=launch.row_bytes, scale_offset=launch.payload_bytes))


def compile_post_silu_aot(*, width, limit):
    if width <= 0:
        raise ValueError("width must be positive")
    _positive("limit", limit)
    return compile_program(PostSilu(width, limit), name="post_silu_product", operands=(
        Operand("gate_up", torch.bfloat16, f"[rows,{2 * width}] (gate first)"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width, limit), geometry=dict(width=width, limit=limit))


def compile_bf16_projection_aot(*, width, outputs, output_dtype=torch.bfloat16, compensated=False):
    """Canonical BF16 MMA projection, with one explicit output boundary.

    FP32 output preserves the accumulator (router/partial sums); BF16 output
    materializes projection rows before norm, activation or residual updates.
    ``compensated`` opts into K16 compensated FP32 accumulation, with the
    measured live-row tiles of ``COMPENSATED_TILES_BY_ROWS`` where the weight
    shape has them (the entries retile the same canonical accumulation, so
    they emit the canonical kernel's bits) and fixed prefill tiles otherwise;
    the default route and its compile identity stay unchanged.
    """
    from b12x.gemm.bf16_gemv._skinny import (
        COMPENSATED_TILES_BY_ROWS,
        RoutedBf16Projection,
        TmaBf16Projection,
    )

    if width <= 0 or outputs <= 0 or width % 8 or outputs % 8:
        raise ValueError("projection dimensions must be positive multiples of eight")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("projection output is BF16 or FP32")
    if not isinstance(compensated, bool):
        raise ValueError("projection compensated must be boolean")
    dtype = BFloat16 if output_dtype == torch.bfloat16 else Float32
    tiles_by_rows = COMPENSATED_TILES_BY_ROWS.get((int(outputs), int(width)), ()) if compensated else ()
    launch = TmaBf16Projection(outputs, width, out_dtype=dtype, tiles_by_rows=tiles_by_rows,
                               compensated=True) \
        if compensated else RoutedBf16Projection(outputs, width, out_dtype=dtype, max_skinny_rows=0)
    geometry = dict(width=width, outputs=outputs, output_dtype=str(output_dtype), skinny_rows=0)
    if compensated:
        geometry["compensated"] = True
        if tiles_by_rows:
            geometry["tiles_by_rows"] = tiles_by_rows
    return compile_program(launch, name="bf16_projection", operands=(
        Operand("x", torch.bfloat16, f"[rows,{width}]"),
        Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
        Operand("out", output_dtype, f"[rows,{outputs}]", "out")),
        scalars=(Scalar("rows"),), key=(width, outputs, str(output_dtype), launch.key()),
        geometry=geometry)


def compile_bf16_scaled_projection_aot(*, width, outputs, prescale):
    if width <= 0 or outputs <= 0:
        raise ValueError("projection dimensions must be positive")
    _positive("prescale", prescale)
    launch = ScaleProjection(width, outputs, prescale)
    return compile_program(launch, name="bf16_scaled_projection", operands=(
        Operand("x", torch.bfloat16, f"[rows,{width}]"),
        Operand("weight", torch.bfloat16, f"[{outputs},{width}] (rounded checkpoint weight)"),
        Operand("out", torch.float32, f"[rows,{outputs}]", "out"),
        Operand("scratch", torch.bfloat16, f"[rows,{outputs}]", "scratch")),
        scalars=(Scalar("rows"),), key=(width, outputs, prescale),
        scratch={"scratch": lambda rows: int(rows) * int(outputs) * 2},
        geometry=dict(width=width, outputs=outputs, prescale=prescale))


def compile_affine_norm_aot(*, width, heads=1, eps=1e-5, zero_centered=False,
                            subtract_mean=False, bias=False, rotary=0,
                            input_dtype=torch.float32, parameter_dtype=torch.float32):
    # None explicitly selects the scalar ABI; the historical default stays constant.
    launch = (RuntimeAffineNorm(width, heads, 1.0, zero_centered, subtract_mean, bias, rotary,
                                vector_io=input_dtype is torch.float32
                                and parameter_dtype is torch.float32)
              if eps is None else
              AffineNorm(width, heads, eps, zero_centered, subtract_mean, bias, rotary))
    if width == 192 and rotary and (subtract_mean or bias):
        raise ValueError("fixed D192 Q/K reduction requires RMSNorm without bias")
    operands = (Operand("x", _dtype(input_dtype), f"[rows,{heads},{width}]"),
        Operand("weight", _dtype(parameter_dtype), f"[{width}]"),
        Operand("bias", parameter_dtype, f"[{width}] (ignored unless bias)"),
        Operand("cos", torch.bfloat16, f"[positions,{rotary // 2}] (ignored unless rotary)"),
        Operand("sin", torch.bfloat16, f"[positions,{rotary // 2}] (ignored unless rotary)"),
        Operand("positions", torch.int64, "[rows] (ignored unless rotary)"),
        Operand("out", torch.bfloat16, f"[rows,{heads},{width}]", "out"))
    if eps is None:
        return compile_program(launch, name="affine_norm_runtime_eps", operands=operands,
            scalars=(Scalar("rows"), Scalar("epsilon", "float32")),
            key=(width, heads, zero_centered, subtract_mean, bias, rotary, input_dtype, parameter_dtype),
            version=2,
            geometry=dict(width=width, heads=heads, epsilon="runtime_float32", zero_centered=zero_centered,
                subtract_mean=subtract_mean, bias=bias, rotary=rotary))
    return compile_program(launch, name="affine_norm", operands=operands, scalars=(Scalar("rows"),),
        key=(width, heads, eps, zero_centered, subtract_mean, bias, rotary, input_dtype, parameter_dtype),
        geometry=dict(width=width, heads=heads, eps=eps, zero_centered=zero_centered,
            subtract_mean=subtract_mean, bias=bias, rotary=rotary))


def compile_residual_add_aot(*, width, delta_dtype=torch.bfloat16):
    if width <= 0:
        raise ValueError("width must be positive")
    return compile_program(ResidualAdd(width), name="residual_add_f32", operands=(
        Operand("residual", torch.float32, f"[rows,{width}]"),
        Operand("delta", _dtype(delta_dtype), f"[rows,{width}]"),
        Operand("out", torch.float32, f"[rows,{width}]", "out")), scalars=(Scalar("rows"),),
        key=(width, delta_dtype), geometry=dict(width=width))


def compile_head_gate_aot(*, heads, dim, logits_dtype=torch.bfloat16):
    if heads <= 0 or dim <= 0:
        raise ValueError("head gate dimensions must be positive")
    return compile_program(HeadGate(heads, dim), name="head_sigmoid_gate", operands=(
        Operand("x", torch.bfloat16, f"[rows,{heads},{dim}]"),
        Operand("logits", _dtype(logits_dtype), f"[rows,{heads}]"),
        Operand("out", torch.bfloat16, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"),), key=(heads, dim, logits_dtype), geometry=dict(heads=heads, dim=dim))


def compile_route_weights_aot(*, experts, topk, epsilon=1e-20, factor=3.0):
    if topk <= 0 or experts < topk:
        raise ValueError("invalid router dimensions")
    _positive("epsilon", epsilon)
    _positive("factor", factor)
    return compile_program(RouteWeights(experts, topk, epsilon, factor), name="sigmoid_route_weights",
        operands=(Operand("logits", torch.float32, f"[rows,{experts}]"),
            Operand("bias", torch.float32, f"[{experts}]"),
            Operand("ids", torch.int32, f"[rows,{topk}] (valid ids from shared selector)"),
            Operand("out", torch.float32, f"[rows,{topk}]", "out")),
        scalars=(Scalar("rows"),), key=(experts, topk, epsilon, factor),
        geometry=dict(experts=experts, topk=topk, epsilon=epsilon, factor=factor))
