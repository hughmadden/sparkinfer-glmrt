"""Three-byte FP32 partials for the GLM Flash two-GPU peer sum.

BF24 retains the sign, all eight exponent bits and fifteen fraction bits.
Each value is rounded to nearest even before its low byte is discarded.
Four little-endian 24-bit values occupy three consecutive uint32 words.
No exponent rescaling or saturation is applied; NaNs remain NaNs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cvt_f32_to_bf16_bits,
    ld_global_v4_u32,
    st_global_u32,
    st_global_v2_u32,
    u32_as_f32,
)
from ._common import (
    AotProgram, GLM53_FLASH, GLMFGeometry, Operand, Scalar, compile_program,
)

__all__ = ["compile_glmf_pack_bf24_aot", "compile_glmf_add_bf24_aot"]


@cute.jit
def _round_bf24(bits: Uint32) -> Uint32:
    rounded = (bits + Uint32(127) + ((bits >> Uint32(8)) & Uint32(1))) >> Uint32(8)
    if (bits & Uint32(0x7F800000)) == Uint32(0x7F800000):
        rounded = bits >> Uint32(8)
        # A NaN whose payload occupies only the discarded byte must not
        # become infinity. Otherwise preserve its retained payload and sign.
        if (bits & Uint32(0x007FFFFF)) != Uint32(0) and (rounded & Uint32(0x007FFF)) == Uint32(0):
            rounded = rounded | Uint32(1)
    return rounded


@cute.jit
def _load_bf24x4(packed: cute.Pointer, group: Int64):
    words = cute.make_ptr(Uint32, Int64(packed.toint()), cute.AddressSpace.gmem, assumed_align=4)
    at = group * Int64(3)
    w0, w1, w2 = words[at], words[at + Int64(1)], words[at + Int64(2)]
    return (
        u32_as_f32(w0 << Uint32(8)),
        u32_as_f32(((w0 >> Uint32(24)) | (w1 << Uint32(8))) << Uint32(8)),
        u32_as_f32(((w1 >> Uint32(16)) | (w2 << Uint32(16))) << Uint32(8)),
        u32_as_f32(w2 & Uint32(0xFFFFFF00)),
    )


class GlmfPackBf24:
    threads = 256

    def __init__(self, width: int):
        self.width = int(width)
        if self.width % 4:
            raise ValueError("BF24 packing requires hidden % 4 == 0")

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        groups = Int64(rows) * Int64(self.width // 4)
        self.kernel(x, out, groups).launch(
            grid=((groups + Int64(self.threads - 1)) // Int64(self.threads), 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, groups: Int64):
        group = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if group < groups:
            x0, x1, x2, x3 = ld_global_v4_u32(Int64(x.toint()) + group * Int64(16))
            q0, q1, q2, q3 = _round_bf24(x0), _round_bf24(x1), _round_bf24(x2), _round_bf24(x3)
            address = Int64(out.toint()) + group * Int64(12)
            st_global_u32(address, q0 | (q1 << Uint32(24)))
            st_global_u32(address + Int64(4), (q1 >> Uint32(8)) | (q2 << Uint32(16)))
            st_global_u32(address + Int64(8), (q2 >> Uint32(16)) | (q3 << Uint32(8)))


class GlmfAddBf24:
    threads = 256

    def __init__(self, width: int):
        self.width = int(width)
        if self.width % 4:
            raise ValueError("BF24 sum requires hidden % 4 == 0")

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        groups = Int64(rows) * Int64(self.width // 4)
        self.kernel(a, b, out, groups).launch(
            grid=((groups + Int64(self.threads - 1)) // Int64(self.threads), 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, groups: Int64):
        group = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if group < groups:
            a0, a1, a2, a3 = _load_bf24x4(a, group)
            b0, b1, b2, b3 = _load_bf24x4(b, group)
            s0, s1 = cvt_f32_to_bf16_bits(a0 + b0), cvt_f32_to_bf16_bits(a1 + b1)
            s2, s3 = cvt_f32_to_bf16_bits(a2 + b2), cvt_f32_to_bf16_bits(a3 + b3)
            st_global_v2_u32(Int64(out.toint()) + group * Int64(8),
                             s0 | (s1 << Uint32(16)), s2 | (s3 << Uint32(16)))


def compile_glmf_pack_bf24_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Round FP32 ``x[rows,H]`` to little-endian BF24 ``out[rows,H,3]``."""
    h = g.hidden
    return compile_program(
        GlmfPackBf24(h), name="glmf_pack_bf24",
        operands=(Operand("x", torch.float32, f"[rows,{h}]"),
                  Operand("out", torch.uint8, f"[rows,{h},3]", "out")),
        scalars=(Scalar("rows"),), key=(h,), geometry={"hidden": h, "bytes_per_partial": 3},
        doc=compile_glmf_pack_bf24_aot.__doc__)


def compile_glmf_add_bf24_aot(g: GLMFGeometry = GLM53_FLASH) -> AotProgram:
    """Decode BF24 partials, add in FP32, and round once to BF16 ``out[rows,H]``."""
    h = g.hidden
    return compile_program(
        GlmfAddBf24(h), name="glmf_add_bf24",
        operands=(Operand("a", torch.uint8, f"[rows,{h},3]"),
                  Operand("b", torch.uint8, f"[rows,{h},3]"),
                  Operand("out", torch.bfloat16, f"[rows,{h}]", "out")),
        scalars=(Scalar("rows"),), key=(h,), geometry={"hidden": h, "bytes_per_partial": 3},
        doc=compile_glmf_add_bf24_aot.__doc__)
