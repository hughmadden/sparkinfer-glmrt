"""INT8 tensor-core helpers for the warp-specialized mixed-Trellis A8 prototype.

EXL3 MCG decode yields FP16 values on a fixed grid bounded by |w| <= 3.9492, so
``round(32 w)`` fits int8 exactly-scaled (relative weight MSE 5e-5, against 7e-4
for an E4M3 requantization). Activations are quantized per (row, 128-wide K
block) after the H128 rotation, where they are close to Gaussian.

K order inside each 32-wide K step follows the FP16 B-fragment lanes: MMA
position 4t + j (and 16 + 4t + j) holds physical k = (2t, 2t+1, 2t+8, 2t+9)[j]
(+16), so a decoded m16n8k16 B fragment pair packs into an m16n8k32 int8
fragment without shuffles; the producer writes activations in that order.
"""

from __future__ import annotations

from typing import Tuple

from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Float32 bit pattern 0x4B400000 = 12582912.0 = 1.5 * 2^23: an int32 accumulator
# initialised to this word holds 12582912 + acc in float32 bits for |acc| < 2^22.
A8_MAGIC_BITS = 0x4B400000
A8_MAGIC_F32 = 12582912.0
A8_W_SCALE = 32.0


def _asm(ret, args, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        ret, args, text, constraints,
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def imma_m16n8k32_s8(
    c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`` (accumulators as u32 bits)."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        loc=loc, ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def f16x2_pair_to_s8x4(lo, hi, *, loc=None, ip=None) -> Uint32:
    """Two f16x2 words (k 2t,2t+1 and 2t+8,2t+9) -> int8x4 of round(32 w):
    fma(w, 32, 1536) lands the integer in the low mantissa byte."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 m, c, x, y;\n"
            "mov.b32 m, 0x50005000;\n"  # (32.0, 32.0)
            "mov.b32 c, 0x66006600;\n"  # (1536.0, 1536.0)
            "fma.rn.f16x2 x, $1, m, c;\n"
            "fma.rn.f16x2 y, $2, m, c;\n"
            "prmt.b32 $0, x, y, 0x6420; }",
            "=r,r,r",
            loc=loc, ip=ip,
        )
    )


@dsl_user_op
def f32x4_to_s8x4(v0, v1, v2, v3, inv, *, loc=None, ip=None) -> Uint32:
    """int8x4 of round(v * inv) for |v * inv| <= 127 (magic-add rounding)."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (v0, v1, v2, v3, inv)],
            "{ .reg .f32 a, b, c, d; .reg .b32 m, p, q;\n"
            "mov.b32 m, 0x4B400000;\n"
            "fma.rn.f32 a, $1, $5, m;\n"
            "fma.rn.f32 b, $2, $5, m;\n"
            "fma.rn.f32 c, $3, $5, m;\n"
            "fma.rn.f32 d, $4, $5, m;\n"
            "prmt.b32 p, a, b, 0x0040;\n"
            "prmt.b32 q, c, d, 0x0040;\n"
            "prmt.b32 $0, p, q, 0x5410; }",
            "=r,f,f,f,f,f",
            loc=loc, ip=ip,
        )
    )


@dsl_user_op
def prmt_b32(a, b, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, $2, {int(sel):#x};",
            "=r,r,r",
            loc=loc, ip=ip,
        )
    )


@dsl_user_op
def a8_rescale(acc, bits, scale, *, loc=None, ip=None) -> Float32:
    """acc + (float(bits) - 12582912) * scale for a magic-initialised int32 block sum."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(acc).ir_value(loc=loc, ip=ip), Uint32(bits).ir_value(loc=loc, ip=ip),
             Float32(scale).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t; mov.b32 t, $2;\n"
            "sub.rn.f32 t, t, 0f4B400000;\n"
            "fma.rn.f32 $0, t, $3, $1; }",
            "=f,f,r,f",
            loc=loc, ip=ip,
        )
    )


@dsl_user_op
def st_shared_v2_u32_a8(addr, a, b, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip),
         Uint32(b).ir_value(loc=loc, ip=ip)],
        "st.shared.v2.u32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
