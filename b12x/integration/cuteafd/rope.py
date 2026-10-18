"""Load-time BF16 rotary tables with explicit FP32 arithmetic boundaries."""
from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _mul, _sub, _view


class RopeTables:
    def __init__(self, rotary_span, llama3):
        self.rotary_span = rotary_span
        self.llama3 = llama3

    @cute.jit
    def __call__(self, cos: cute.Pointer, sin: cute.Pointer,
                 positions: Int32, theta: Float32, factor: Float32,
                 original_max: Float32, low_factor: Float32, high_factor: Float32,
                 stream: cuda.CUstream):
        count = Int64(positions) * Int64(self.rotary_span // 2)
        self.kernel(cos, sin, count, theta, factor, original_max, low_factor, high_factor).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, cos: cute.Pointer, sin: cute.Pointer, count: Int64,
               theta: Float32, factor: Float32, original_max: Float32,
               low_factor: Float32, high_factor: Float32):
        cos, sin = _view(cos), _view(sin)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            position = i // Int64(self.rotary_span // 2)
            dim = i % Int64(self.rotary_span // 2)
            # Torch tensor/scalar division multiplies by the FP32 reciprocal.
            exponent = _mul(Float32(dim * Int64(2)), Float32(1.0 / self.rotary_span))
            power = cute.math.pow(theta, exponent, fastmath=False)
            inv_freq = div_rn_f32(Float32(1.0), power)
            if cutlass.const_expr(self.llama3):
                low_wave = div_rn_f32(original_max, low_factor)
                high_wave = div_rn_f32(original_max, high_factor)
                wave = _mul(div_rn_f32(Float32(1.0), inv_freq), Float32(2 * math.pi))
                scaled = inv_freq
                if wave > low_wave:
                    scaled = _mul(inv_freq, div_rn_f32(Float32(1.0), factor))
                smooth = _mul(_sub(_mul(div_rn_f32(Float32(1.0), wave), original_max), low_factor),
                              div_rn_f32(Float32(1.0), _sub(high_factor, low_factor)))
                smoothed = _add(_mul(_mul(_sub(Float32(1.0), smooth), scaled),
                                     div_rn_f32(Float32(1.0), factor)), _mul(smooth, scaled))
                if wave >= high_wave and wave <= low_wave:
                    inv_freq = smoothed
                else:
                    inv_freq = scaled
            frequency = _mul(Float32(position), inv_freq)
            cos[i] = cute.math.cos(frequency, fastmath=False).to(BFloat16)
            sin[i] = cute.math.sin(frequency, fastmath=False).to(BFloat16)


def compile_rope_tables_aot(*, rotary_span, llama3=False):
    """Compile once; extent and all checkpoint recipe values are launch scalars."""
    if not isinstance(rotary_span, int) or rotary_span <= 0 or rotary_span % 2:
        raise ValueError("rotary_span must be a positive even integer")
    return compile_program(RopeTables(rotary_span, bool(llama3)), name="rope_tables_bf16",
        operands=(Operand("cos", torch.bfloat16, f"[positions,{rotary_span // 2}]", "out"),
                  Operand("sin", torch.bfloat16, f"[positions,{rotary_span // 2}]", "out")),
        scalars=(Scalar("positions"), Scalar("theta", "float32"), Scalar("factor", "float32"),
                 Scalar("original_max", "float32"), Scalar("low_factor", "float32"),
                 Scalar("high_factor", "float32")),
        key=(rotary_span, bool(llama3)), geometry=dict(rotary_span=rotary_span, llama3=bool(llama3)))
