"""Native trellis linear on pre-rotated inputs; fixed-order FP32 partials.

Both native output-unit and input-unit storage are consumed without repacking.
Rotations and channel scales belong to the caller, allowing several operands to
share one input rotation. A block scale multiplies each K128 accumulator, never
the E4M3/F16 decoded fragments. Live rows never select a different arithmetic.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, _f16x2_to_bf16x2,
    f16_mma_m16n8k16_f32, ld_global_nc_u32, ld_global_nc_u32_l2hint, ld_global_nc_v4_u32,
    weight_stream_policy,
    ld_shared_u32, ld_shared_f32, st_shared_f32, packed_dequant_trellis_to_half2x4,
    packed_dequant_trellis_stream_to_half2x4,
    packed_dequant_trellis_mul1_to_half2x4,
    shared_ptr_to_u32, st_global_v2_f32, st_shared_v4_u32,
    trellis_align_stream_u32x2,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a16.exlr8 import _e4m3_round_half2, _side_value
from b12x.moe._shared.kernels.w4a16.mixed_trellis_ws import (
    _cp_async_mbar_arrive_noinc,
    _fence_mbar_init,
    _mbar_arrive,
    _mbar_init,
    _mbar_wait,
)


# Gemini's consumer; E4M3 weights are exact in both F16 and BF16.
_WS_CONSUMER_THREADS = 256
_WS_PRODUCER_THREADS = 128
_WS_CONSUMER_REGS = 192
_WS_PRODUCER_REGS = 120
_WS_STAGES = 4
_WS_A_STAGE_BYTES = 8192
_WS_MIN_ROWS = 130
_PREFILL_MIN_ROWS = 130
_PREFILL_STAGES = 2
_PREFILL_PRODUCER_REGS = 104
_PREFILL_CONSUMER_REGS = 192
_PREFILL_BM = 128
_PREFILL_BN = 64
_PREFILL_CONSUMER_THREADS = _WS_CONSUMER_THREADS
_PREFILL_PRODUCER_THREADS = _WS_PRODUCER_THREADS


@dsl_user_op
def _mbar_init(addr: Int32, count: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "fence.mbarrier_init.release.cluster;",
        "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _mbar_arrive(addr: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _cp_async_mbar_arrive_noinc(addr: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _mbar_wait(addr: Int32, parity: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; WS_WAIT: "
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; "
        "@!p bra WS_WAIT; }",
        "r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_global_nc_u32_offset(base_ptr: Int64, offset: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            f"ld.global.nc.u32 $0, [$1 + {int(offset)}];",
            "=r,l",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def cp_async4_zfill_shared_global_offset(
    smem_base: Int32, smem_off: int, gmem_base: Int64, gmem_off: int, src_size: Int32,
    *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Int64(gmem_base).ir_value(loc=loc, ip=ip),
            Int32(src_size).ir_value(loc=loc, ip=ip),
        ],
        f"cp.async.cg.shared.global [$0 + {int(smem_off)}], [$1 + {int(gmem_off)}], 16, $2;",
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def st_shared_v2_f32_offset(
    smem_base: Int32, offset: int, v0: Float32, v1: Float32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        f"st.shared.v2.f32 [$0 + {int(offset)}], {{$1, $2}};",
        "r,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_shared_v2_f32_offset(
    smem_base: Int32, offset: int, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_base).ir_value(loc=loc, ip=ip)],
        f"ld.shared.v2.f32 {{$0, $1}}, [$2 + {int(offset)}];",
        "=f,=f,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_shared_v2_u32_offset(
    smem_base: Int32, offset: int, v0: Uint32, v1: Uint32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
        ],
        f"st.shared.v2.u32 [$0 + {int(offset)}], {{$1, $2}};",
        "r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_shared_v2_u32_offset(
    smem_base: Int32, offset: int, *, loc=None, ip=None
) -> tuple[Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_base).ir_value(loc=loc, ip=ip)],
        f"ld.shared.v2.u32 {{$0, $1}}, [$2 + {int(offset)}];",
        "=r,=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def ldmatrix_m8n8x4_b16_offset(base_addr: Int32, offset: int, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0, $1, $2, $3}}, [$4 + {int(offset)}];",
        "=r,=r,=r,=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f16x2_pair_to_e4m3x4(v0: Uint32, v1: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(v0).ir_value(loc=loc, ip=ip), Uint32(v1).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 lo, hi; "
            "cvt.rn.satfinite.e4m3x2.f16x2 lo, $1; "
            "cvt.rn.satfinite.e4m3x2.f16x2 hi, $2; "
            "mov.b32 $0, {lo, hi}; }",
            "=r,r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def unpack_e4m3x4_to_f16x2_pair(packed: Uint32, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi; "
        "mov.b32 {lo, hi}, $2; "
        "cvt.rn.f16x2.e4m3x2 $0, lo; "
        "cvt.rn.f16x2.e4m3x2 $1, hi; }",
        "=r,=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def add_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def ld_global_v2_f32(addr: Int64, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_global_v2_f32(addr: Int64, v0: Float32, v1: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )



@dsl_user_op
def _ld_global_nc_u32_offset(base_ptr: Int64, offset: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .b64 p; createpolicy.fractional.L2::evict_last.b64 p, 1.0; "
        f"ld.global.nc.L2::cache_hint.u32 $0, [$1 + {int(offset)}], p; }}",
        "=r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _st_global_cs_v2_f32_offset(base_ptr: Int64, offset: int, v0: Float32, v1: Float32,
                                *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip),
         Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip)],
        f"st.global.cs.v2.f32 [$0 + {int(offset)}], {{$1, $2}};",
        "l,f,f", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _f16x2x2_to_e4m3x4(b0: Uint32, b1: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(b0).ir_value(loc=loc, ip=ip), Uint32(b1).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f16x2 lo, $1; "
        "cvt.rn.satfinite.e4m3x2.f16x2 hi, $2; mov.b32 $0, {lo, hi}; }",
        "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _e4m3x4_to_f16x2x2(word: Uint32, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
        "cvt.rn.f16x2.e4m3x2 $0, lo; cvt.rn.f16x2.e4m3x2 $1, hi; }",
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _cp_async4_zfill_offset(smem_addr: Int32, smem_offset: int,
                            gmem_addr: Int64, gmem_offset: int,
                            rem_rows: Int32, row_off: int,
                            *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(rem_rows).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .pred p; .reg .b32 sz; setp.gt.s32 p, $2, {int(row_off)}; selp.b32 sz, 16, 0, p; "
        f"cp.async.cg.shared.global [$0 + {int(smem_offset)}], [$1 + {int(gmem_offset)}], 16, sz; }}",
        "r,l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_v2_u32_offset(smem_addr: Int32, offset: int, v0: Uint32, v1: Uint32,
                             *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Uint32(v0).ir_value(loc=loc, ip=ip),
         Uint32(v1).ir_value(loc=loc, ip=ip)],
        f"st.shared.v2.b32 [$0 + {int(offset)}], {{$1, $2}};",
        "r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_v4_u32_offset(smem_addr: Int32, offset: int, v0: Uint32, v1: Uint32,
                             v2: Uint32, v3: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Uint32(v0).ir_value(loc=loc, ip=ip),
         Uint32(v1).ir_value(loc=loc, ip=ip),
         Uint32(v2).ir_value(loc=loc, ip=ip),
         Uint32(v3).ir_value(loc=loc, ip=ip)],
        f"st.shared.v4.b32 [$0 + {int(offset)}], {{$1, $2, $3, $4}};",
        "r,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _ld_ring_v2_u32_offset(smem_addr: Int32, offset: int, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.v2.b32 {{$0, $1}}, [$2 + {int(offset)}];",
        "=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _ld_ring_v4_u32_offset(smem_addr: Int32, offset: int,
                           *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.v4.b32 {{$0, $1, $2, $3}}, [$4 + {int(offset)}];",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)))


@dsl_user_op
def _ldmatrix_m8n8x4_b16_offset(base_addr: Int32, offset: int,
                                *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0, $1, $2, $3}}, [$4 + {int(offset)}];",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)))


def _load_word(address, policy):
    """A trellis payload word, with ``policy`` as its L2 cache policy when given."""
    if policy is None:
        return ld_global_nc_u32(address)
    return ld_global_nc_u32_l2hint(address, policy)


@cute.jit
def decode_tile_raw(address: Int64, tile: Int32, lane: Int32,
                    bits: cutlass.Constexpr, mul1: cutlass.Constexpr = False, policy=None):
    """Decode exact FP16 weights, retaining the caller's L2 payload policy."""
    base = address + tile.to(Int64) * Int64(32 * bits)

    if cutlass.const_expr(bits == 4):
        b = _load_word(base + lane.to(Int64) * Int64(4), policy)
        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
        wa, wb = b, (a << Uint32(16)) | (b >> Uint32(16))
    elif cutlass.const_expr(bits <= 4):
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = _load_word(base + ia.to(Int64) * Int64(4), policy)
        b = _load_word(base + ib.to(Int64) * Int64(4), policy)
        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
        wa = (window >> shift.to(Int64)).to(Uint32)
        wb = (window >> (shift + Int32(4 * bits)).to(Int64)).to(Uint32)
    elif cutlass.const_expr(bits == 5):
        # Four K5 windows span 31 bits; all eight may span three ring words.
        ib0, ib1, sb, _ = trellis256_lane_geom_bits(lane, 0, 4, bits)
        ia0, ia1, sa, _ = trellis256_lane_geom_bits(lane, 4, 4, bits)
        b0 = _load_word(base + ib0.to(Int64) * Int64(4), policy)
        b1 = _load_word(base + ib1.to(Int64) * Int64(4), policy)
        a0 = _load_word(base + ia0.to(Int64) * Int64(4), policy)
        a1 = _load_word(base + ia1.to(Int64) * Int64(4), policy)
        wa = (((a0.to(Int64) << Int64(32)) | a1.to(Int64)) >> sa.to(Int64)).to(Uint32)
        wb = (((b0.to(Int64) << Int64(32)) | b1.to(Int64)) >> sb.to(Int64)).to(Uint32)
    else:
        # Even four K6 windows need 34 bits. Retain the aligned 64-bit stream.
        i0, i2, shift, delta = trellis256_lane_geom_bits(lane, 0, 8, bits)
        i1 = i0 + Int32(1)
        i1 -= Int32(8 * bits) * (i1 >= Int32(8 * bits)).to(Int32)
        z0 = _load_word(base + i0.to(Int64) * Int64(4), policy)
        z1 = _load_word(base + i1.to(Int64) * Int64(4), policy)
        z2 = _load_word(base + i2.to(Int64) * Int64(4), policy)
        wa, wb = trellis_align_stream_u32x2(z0, z1, z2, shift, delta)
    if cutlass.const_expr(mul1):
        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, bits)
    elif cutlass.const_expr(bits == 6):
        b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, bits)
    else:
        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, bits)
    return b0, b1, b2, b3


@cute.jit
def decode_tile(address: Int64, tile: Int32, lane: Int32,
                bits: cutlass.Constexpr, e4m3: cutlass.Constexpr,
                mul1: cutlass.Constexpr = False, policy=None):
    b0, b1, b2, b3 = decode_tile_raw(address, tile, lane, bits, mul1, policy)
    if cutlass.const_expr(e4m3):
        b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
        b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
    return b0, b1, b2, b3


class NativeTrellisLinear:
    ABI_VERSION = 1
    CTAS_PER_SM = 4

    def __init__(self, g):
        self.k, self.n = g["input_features"], g["output_features"]
        self.bits, self.e4m3 = g["bits"], g["codebook"] == "e4m3"
        self.mul1 = g["codebook"] == "mul1"
        self.input_bf16 = g.get("input_dtype", "f16") == "bf16"
        # High-rate windows need more live registers than the frozen K2--K4 path.
        self.min_blocks_per_mp = 2 if self.bits >= 5 else self.CTAS_PER_SM
        self.input_units = g["sharded_side"] == 2
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        self.splits = g["splits"]
        self.ws_b_stage_bytes = 4096 if self.e4m3 else 8192
        self.ws_stage_bytes = _WS_A_STAGE_BYTES + self.ws_b_stage_bytes
        self.ws_bar_off = _WS_STAGES * self.ws_stage_bytes
        self.ws_sc_m_start = 4 if self.splits <= 1 else (1 if self.e4m3 else 2)
        self.ws_sc_off = self.ws_bar_off + 128
        self.ws_smem_bytes = (
            self.ws_bar_off + 2 * _WS_STAGES * 8
            if self.splits <= 1
            else self.ws_sc_off + (4 - self.ws_sc_m_start) * 16384
        )


    @cute.jit
    def mma(self, d0: Float32, d1: Float32, d2: Float32, d3: Float32,
            a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32,
            b0: Uint32, b1: Uint32):
        # Decoders retain their native half2 contract. E4M3 is exact in BF16;
        # other codebooks round once here, before the BF16 MMA.
        if cutlass.const_expr(self.input_bf16):
            b0, b1 = _f16x2_to_bf16x2(b0), _f16x2_to_bf16x2(b1)
            return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        else:
            return f16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    def _storage(self, active: int):
        class Storage:
            pass
        Storage.__annotations__ = {
            "x": cute.struct.Align[cute.struct.MemRange[Uint32, 16 * (16 * active + 1) * 4], 16],
        }
        return cute.struct(Storage)

    def _ws_storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "words": cute.struct.Align[
                cute.struct.MemRange[Uint32, (self.ws_smem_bytes + 3) // 4], 1024
            ],
        }
        return cute.struct(Storage)


    def _prefill_storage(self):
        w_tile_bytes = 256 if self.e4m3 else 512
        stage_bytes = _PREFILL_BM * 64 * 2 + 16 * w_tile_bytes
        total_bytes = _PREFILL_STAGES * stage_bytes + 128
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, total_bytes], 128],
        }
        return cute.struct(Storage)


    @cute.jit
    def __call__(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations) and cta_limit > Int32(0):
            if cutlass.const_expr(self.e4m3 and self.max_rows > 64):
                if rows > Int32(64):
                    tiles = ((rows + Int32(127)) // Int32(128)) * Int32(1 * (self.n // 128))
                    self.multiply_prefill_ws(inputs, payload, scales, output, rows, rotation_id).launch(
                        grid=(min(tiles, max(Int32(1), cta_limit // Int32(self.CTAS_PER_SM))), 1, 1),
                        block=(_WS_CONSUMER_THREADS + _WS_PRODUCER_THREADS, 1, 1),
                        min_blocks_per_mp=1, stream=stream)
                else:
                    self._generic_call(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)
            else:
                self._generic_call(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)

    @cute.jit
    def _generic_call(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations) and cta_limit > Int32(0):
            if cutlass.const_expr(self.max_rows >= _PREFILL_MIN_ROWS):
                if rows >= Int32(_PREFILL_MIN_ROWS):
                    prefill_tiles = ((rows + Int32(_PREFILL_BM - 1)) // Int32(_PREFILL_BM)) * Int32(self.n // _PREFILL_BN)
                    self.multiply_prefill(inputs, payload, scales, output, rows, rotation_id).launch(
                        grid=(min(prefill_tiles, max(Int32(1), cta_limit // Int32(4))), 1, 1),
                        block=(_PREFILL_CONSUMER_THREADS + _PREFILL_PRODUCER_THREADS, 1, 1),
                        min_blocks_per_mp=1, stream=stream)
                else:
                    self._decode(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)
            else:
                self._decode(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)

    @cute.jit
    def _decode(self, inputs: cute.Tensor, payload: cute.Tensor,
                scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                rows: Int32, rotation_id: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.bits == 4 and self.e4m3 and self.splits <= 16):
            if rows <= Int32(16):
                if cutlass.const_expr(self.splits <= 2):
                    self.multiply_folded(inputs, payload, scales, output, rows, rotation_id).launch(
                        grid=(min(Int32(self.n // 32), cta_limit), 1, 1),
                        block=(32 * self.splits, 1, 1), min_blocks_per_mp=1, stream=stream)
                else:
                    self._legacy(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)
            elif rows > Int32(48) and rows <= Int32(64):
                self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 4, 4).launch(
                    grid=(min(Int32(self.n // 128 * self.splits), cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=1, stream=stream)
                self.reduce(partial, output, rows).launch(
                    grid=(min(rows * Int32(self.n // 128), cta_limit), 1, 1),
                    block=(128, 1, 1), stream=stream)
            else:
                self._legacy(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)
        else:
            self._legacy(inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream)

    @cute.jit
    def _legacy(self, inputs: cute.Tensor, payload: cute.Tensor,
                scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                rows: Int32, rotation_id: Int32, cta_limit: Int32, stream: cuda.CUstream):
        tiles = Int32(self.n // 128 * self.splits)
        if cutlass.const_expr(self.bits < 5):
            self.multiply(inputs, payload, scales, partial, rows, rotation_id).launch(
                grid=(min(((rows + Int32(15)) // Int32(16)) * tiles, cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
        elif rows <= Int32(16):
            self.multiply(inputs, payload, scales, partial, rows, rotation_id).launch(
                grid=(min(tiles, cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
        elif rows <= Int32(32):
            self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 2, 8).launch(
                grid=(min(tiles, cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
        elif rows <= Int32(48):
            self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 3, 8).launch(
                grid=(min(tiles, cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
        else:
            self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 4, 4).launch(
                grid=(min(((rows + Int32(63)) // Int32(64)) * tiles, cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
        self.reduce(partial, output, rows).launch(
            grid=(min(rows * Int32(self.n // 128), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)

    def _fold_storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "fold": cute.struct.Align[cute.struct.MemRange[Float32, self.splits * 512], 16],
        }
        return cute.struct(Storage)

    @cute.kernel
    def multiply_folded(self, inputs: cute.Tensor, payload: cute.Tensor,
                        scales: cute.Tensor, output: cute.Tensor,
                        rows: Int32, rotation_id: Int32):
        """Warp per original K split; preserve its fold order inside one CTA."""
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, split = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        storage = cutlass_utils.SmemAllocator().allocate(self._fold_storage())
        fold = shared_ptr_to_u32(storage.fold.data_ptr())
        x = inputs[rotation_id]
        while task < ((rows + Int32(15)) // Int32(16)) * Int32(self.n // 32):
            nb = task % Int32(self.n // 32)
            group = task // Int32(self.n // 32)
            policy = weight_stream_policy((group + Int32(1) == (rows + Int32(15)) // Int32(16)).to(Int32))
            row0 = group * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(2):
                            tile = (nb // Int32(4)) * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + (nb % Int32(4)) * Int32(2) + Int32(n16)
                            if cutlass.const_expr(self.input_units):
                                tile = kb * Int32(self.n // 16 * 8) + (nb // Int32(4)) * Int32(64) + Int32(k16 * 8) + (nb % Int32(4)) * Int32(2) + Int32(n16)
                            b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for n in cutlass.range_constexpr(4):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
            for n in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(4):
                    st_shared_f32(fold + (split * Int32(512) + Int32(n*128+v*32) + lane) * Int32(4), total[n,v])
            cute.arch.sync_threads()
            if split == Int32(0):
                for n in cutlass.range_constexpr(4):
                    result = cute.make_rmem_tensor((4,), Float32)
                    result.fill(Float32(0.0))
                    for sp in cutlass.range_constexpr(self.splits):
                        for v in cutlass.range_constexpr(4):
                            result[v] += ld_shared_f32(fold + (Int32(sp*512+n*128+v*32) + lane) * Int32(4))
                    col = nb * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                    if row0 < rows:
                        base = row0.to(Int64) * Int64(self.n) + col.to(Int64)
                        output[base], output[base+Int64(1)] = result[0], result[1]
                    if row1 < rows:
                        base = row1.to(Int64) * Int64(self.n) + col.to(Int64)
                        output[base], output[base+Int64(1)] = result[2], result[3]
            cute.arch.sync_threads()
            task += step

    @cute.kernel
    def multiply(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor,
                 rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        x = inputs[rotation_id]
        while task < ((rows + Int32(15)) // Int32(16)) * Int32(self.n // 128 * self.splits):
            split = task % Int32(self.splits)
            nb = (task // Int32(self.splits)) % Int32(self.n // 128)
            group = task // Int32(self.splits * (self.n // 128))
            # Each 16-row group re-reads the weights; the last one evicts them first.
            policy = weight_stream_policy((group + Int32(1) == (rows + Int32(15)) // Int32(16)).to(Int32))
            row0 = group * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(2):
                            tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                            if cutlass.const_expr(self.input_units):
                                tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                            b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for n in cutlass.range_constexpr(4):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
            for n in cutlass.range_constexpr(4):
                col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                base = (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                if row0 < rows:
                    partial[base], partial[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 < rows:
                    base += Int64(8 * self.splits * self.n)
                    partial[base], partial[base + Int64(1)] = total[n, 2], total[n, 3]
            task += step

    @cute.kernel
    def multiply_wide(self, inputs: cute.Tensor, payload: cute.Tensor,
                      scales: cute.Tensor, partial: cute.Tensor,
                      rows: Int32, rotation_id: Int32,
                      active: cutlass.Constexpr, unroll_k16: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        c, r_grp = tid & Int32(15), tid >> Int32(4)
        x = inputs[rotation_id]
        M = 16 * active
        M_PAD = M + 1
        tiles = Int32(self.n // 128 * self.splits)
        limit = tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * tiles
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage(active))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)
        while task < limit:
            split = task % Int32(self.splits)
            nb = (task // Int32(self.splits)) % Int32(self.n // 128)
            group = Int32(0) if cutlass.const_expr(active < 4) else task // tiles
            # One pass over the weights below 49 rows; else the last 64-row group evicts.
            last_reader = Int32(1)
            if cutlass.const_expr(active == 4):
                last_reader = (group + Int32(1) == (rows + Int32(M - 1)) // Int32(M)).to(Int32)
            policy = weight_stream_policy(last_reader)
            row_base = (lane >> Int32(2)) if cutlass.const_expr(active < 4) else group * Int32(M) + (lane >> Int32(2))
            total = cute.make_rmem_tensor((active, 4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    k_col = kb * Int32(128) + c * Int32(8)
                    for s in cutlass.range_constexpr(2 * active):
                        row = r_grp + Int32(s * 8) if cutlass.const_expr(active < 4) else group * Int32(M) + r_grp + Int32(s * 8)
                        v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if cutlass.const_expr(active in (2, 3) and s < 2 * (active - 1)):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        else:
                            if row < rows:
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                    cute.arch.sync_threads()
                    if cutlass.const_expr(active <= 2):
                        acc = cute.make_rmem_tensor((active, 4, 4), Float32)
                        acc.fill(Float32(0.0))
                        for k16 in cutlass.range_constexpr(8):
                            a = cute.make_rmem_tensor((active, 4), Uint32)
                            for mb in cutlass.range_constexpr(active):
                                a[mb, 0] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                a[mb, 1] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                a[mb, 2] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                a[mb, 3] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                            for n16 in cutlass.range_constexpr(2):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 2 * n16, 0], acc[mb, 2 * n16, 1], acc[mb, 2 * n16, 2], acc[mb, 2 * n16, 3],
                                        a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b0, b1)
                                    acc[mb, 2 * n16, 0], acc[mb, 2 * n16, 1], acc[mb, 2 * n16, 2], acc[mb, 2 * n16, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 2 * n16 + 1, 0], acc[mb, 2 * n16 + 1, 1], acc[mb, 2 * n16 + 1, 2], acc[mb, 2 * n16 + 1, 3],
                                        a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b2, b3)
                                    acc[mb, 2 * n16 + 1, 0], acc[mb, 2 * n16 + 1, 1], acc[mb, 2 * n16 + 1, 2], acc[mb, 2 * n16 + 1, 3] = d0, d1, d2, d3
                        scale = _side_value(scales.iterator.toint(), kb)
                        for mb in cutlass.range_constexpr(active):
                            for n in cutlass.range_constexpr(4):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, n, v] = total[mb, n, v] + acc[mb, n, v] * scale
                    elif cutlass.const_expr(active == 3):
                        scale = _side_value(scales.iterator.toint(), kb)
                        for n16 in cutlass.range_constexpr(2):
                            acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                            acc.fill(Float32(0.0))
                            for k16 in cutlass.range_constexpr(8):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    a0 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                    a1 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                    a2 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                    a3 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                        a0, a1, a2, a3, b2, b3)
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                            for mb in cutlass.range_constexpr(active):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                    total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    else:
                        scale = _side_value(scales.iterator.toint(), kb)
                        for n16 in cutlass.range_constexpr(2):
                            acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                            acc.fill(Float32(0.0))
                            for ko in cutlass.range(8 // unroll_k16, unroll=1):
                                sx_k = sx_read + ko * Int32(2 * unroll_k16 * M_PAD * 16)
                                for ki in cutlass.range_constexpr(unroll_k16):
                                    k16 = ko * Int32(unroll_k16) + Int32(ki)
                                    tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + k16) * Int32(8) + warp * Int32(2) + Int32(n16)
                                    if cutlass.const_expr(self.input_units):
                                        tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + k16 * Int32(8) + warp * Int32(2) + Int32(n16)
                                    b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                    for mb in cutlass.range_constexpr(active):
                                        a0 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16) * 16))
                                        a1 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16 + 8) * 16))
                                        a2 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16) * 16))
                                        a3 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16 + 8) * 16))
                                        d0, d1, d2, d3 = self.mma(
                                            acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                            a0, a1, a2, a3, b0, b1)
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                        d0, d1, d2, d3 = self.mma(
                                            acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                            a0, a1, a2, a3, b2, b3)
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                            for mb in cutlass.range_constexpr(active):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                    total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    cute.arch.sync_threads()
            for mb in cutlass.range_constexpr(active):
                row0 = row_base + Int32(mb * 16)
                row1 = row0 + Int32(8)
                for n in cutlass.range_constexpr(4):
                    col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                    base = (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                    t0, t1, t2, t3 = total[mb, n, 0], total[mb, n, 1], total[mb, n, 2], total[mb, n, 3]
                    if cutlass.const_expr(active < 4 and mb < active - 1):
                        partial[base], partial[base + Int64(1)] = t0, t1
                        base += Int64(8 * self.splits * self.n)
                        partial[base], partial[base + Int64(1)] = t2, t3
                    else:
                        if row0 < rows:
                            partial[base], partial[base + Int64(1)] = t0, t1
                        if row1 < rows:
                            base += Int64(8 * self.splits * self.n)
                            partial[base], partial[base + Int64(1)] = t2, t3
            task += step

    @cute.kernel
    def reduce(self, partial: cute.Tensor, output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        while task < rows * Int32(self.n // 128):
            row, col = task // Int32(self.n // 128), (task % Int32(self.n // 128)) * Int32(128) + Int32(tid)
            total = Float32(0.0)
            for split in cutlass.range(self.splits):
                total += partial[(row.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)]
            output[row.to(Int64) * Int64(self.n) + col.to(Int64)] = total
            task += step

    @cute.jit
    def _prefill_produce_tile(self, p_tid: Int32, base: Int32,
                              x_ptr: Int64, payload_ptr: Int64,
                              rows: Int32, m0: Int32, nb64: Int32,
                              stage_cnt: Int32) -> Int32:
        w_tile_bytes = 256 if self.e4m3 else 512
        a_stage_bytes = _PREFILL_BM * 64 * 2
        stage_bytes = a_stage_bytes + 16 * w_tile_bytes
        bar_full = base + Int32(_PREFILL_STAGES * stage_bytes)
        bar_empty = bar_full + Int32(8 * _PREFILL_STAGES)
        lane, pw = p_tid & Int32(31), p_tid >> Int32(5)
        r0, c = p_tid >> Int32(3), p_tid & Int32(7)
        sa_lane_off = r0 * Int32(128) + ((c ^ (r0 & Int32(7))) << Int32(4))
        sw_pw_off = Int32(a_stage_bytes) + pw * Int32(w_tile_bytes) + lane * Int32(8 if self.e4m3 else 16)
        nb128 = nb64 >> Int32(1)
        n16_in_128 = (nb64 & Int32(1)) * Int32(4) + pw
        tile_init = nb128 * Int32(64 if self.input_units else (self.k // 16 * 8)) + n16_in_128
        kb_w_stride = Int64((self.n // 2 if self.input_units else 64) * 32 * self.bits)
        w_tile_base = payload_ptr + tile_init.to(Int64) * Int64(32 * self.bits)
        prev_lane = (lane + Int32(31)) & Int32(31)
        i0, i2, shift, delta = trellis256_lane_geom_bits(lane, 0, 8, self.bits)
        i1 = i0 + Int32(1)
        i1 -= Int32(8 * self.bits) * (i1 >= Int32(8 * self.bits)).to(Int32)
        p0_cur = w_tile_base + (lane if self.bits == 4 else i0).to(Int64) * Int64(4)
        p1_cur = w_tile_base + i1.to(Int64) * Int64(4)
        p2_cur = w_tile_base + i2.to(Int64) * Int64(4)
        rem_rows = rows - (m0 + r0)
        x_cur = x_ptr + (m0 + r0).to(Int64) * Int64(self.k * 2) + (c * Int32(8)).to(Int64) * Int64(2)
        w_words = 1 if self.bits == 4 else 3
        w_s0 = cute.make_rmem_tensor((4, w_words), Uint32)
        w_s1 = cute.make_rmem_tensor((4, w_words), Uint32)
        for k16_in_64 in cutlass.range_constexpr(4):
            off0 = k16_in_64 * 256 * self.bits
            off1 = (4 + k16_in_64) * 256 * self.bits
            w_s0[k16_in_64, 0] = _ld_global_nc_u32_offset(p0_cur, off0)
            w_s1[k16_in_64, 0] = _ld_global_nc_u32_offset(p0_cur, off1)
            if cutlass.const_expr(self.bits != 4):
                w_s0[k16_in_64, 1] = _ld_global_nc_u32_offset(p1_cur, off0)
                w_s0[k16_in_64, 2] = _ld_global_nc_u32_offset(p2_cur, off0)
                w_s1[k16_in_64, 1] = _ld_global_nc_u32_offset(p1_cur, off1)
                w_s1[k16_in_64, 2] = _ld_global_nc_u32_offset(p2_cur, off1)
        sa_base0 = base + sa_lane_off
        sw_base0 = base + sw_pw_off
        num_kb = Int32(self.k // 128)
        for kb in cutlass.range(self.k // 128, unroll=1):
            phase = (stage_cnt & Int32(1)) ^ Int32(1)
            # Stage s = 0 (slot = 0)
            _mbar_wait(bar_empty, phase)
            for i in cutlass.range_constexpr(8):
                _cp_async4_zfill_offset(
                    sa_base0, i * 2048,
                    x_cur, i * 32 * self.k, rem_rows, i * 16)
            _cp_async_mbar_arrive_noinc(bar_full)
            for k16_in_64 in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.bits == 4):
                    b = w_s0[k16_in_64, 0]
                    a = cute.arch.shuffle_sync(b, prev_lane)
                    wa, wb = b, (a << Uint32(16)) | (b >> Uint32(16))
                    if cutlass.const_expr(self.mul1):
                        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, 4)
                    else:
                        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 4)
                else:
                    wa, wb = trellis_align_stream_u32x2(
                        w_s0[k16_in_64, 0], w_s0[k16_in_64, 1], w_s0[k16_in_64, 2], shift, delta)
                    if cutlass.const_expr(self.mul1):
                        # K2--K5 MUL1 consumes two four-window words, not lo/hi.
                        if cutlass.const_expr(self.bits < 6):
                            wb = (wa >> Uint32(4 * self.bits)) | (wb << Uint32(32 - 4 * self.bits))
                        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, self.bits)
                    else:
                        b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, self.bits)
                if cutlass.const_expr(self.e4m3):
                    e01 = _f16x2x2_to_e4m3x4(b0, b1)
                    e23 = _f16x2x2_to_e4m3x4(b2, b3)
                    _st_shared_v2_u32_offset(sw_base0, k16_in_64 * 4 * w_tile_bytes, e01, e23)
                else:
                    _st_shared_v4_u32_offset(sw_base0, k16_in_64 * 4 * w_tile_bytes, b0, b1, b2, b3)
            _mbar_arrive(bar_full)
            if kb + Int32(1) < num_kb:
                p0_cur = p0_cur + kb_w_stride
                p1_cur = p1_cur + kb_w_stride
                p2_cur = p2_cur + kb_w_stride
            for k16_in_64 in cutlass.range_constexpr(4):
                off0 = k16_in_64 * 256 * self.bits
                w_s0[k16_in_64, 0] = _ld_global_nc_u32_offset(p0_cur, off0)
                if cutlass.const_expr(self.bits != 4):
                    w_s0[k16_in_64, 1] = _ld_global_nc_u32_offset(p1_cur, off0)
                    w_s0[k16_in_64, 2] = _ld_global_nc_u32_offset(p2_cur, off0)

            # Stage s = 1 (slot = 1)
            _mbar_wait(bar_empty + Int32(8), phase)
            for i in cutlass.range_constexpr(8):
                _cp_async4_zfill_offset(
                    sa_base0, stage_bytes + i * 2048,
                    x_cur, 128 + i * 32 * self.k, rem_rows, i * 16)
            _cp_async_mbar_arrive_noinc(bar_full + Int32(8))
            x_cur = x_cur + Int64(256)
            for k16_in_64 in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.bits == 4):
                    b = w_s1[k16_in_64, 0]
                    a = cute.arch.shuffle_sync(b, prev_lane)
                    wa, wb = b, (a << Uint32(16)) | (b >> Uint32(16))
                    if cutlass.const_expr(self.mul1):
                        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, 4)
                    else:
                        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 4)
                else:
                    wa, wb = trellis_align_stream_u32x2(
                        w_s1[k16_in_64, 0], w_s1[k16_in_64, 1], w_s1[k16_in_64, 2], shift, delta)
                    if cutlass.const_expr(self.mul1):
                        # K2--K5 MUL1 consumes two four-window words, not lo/hi.
                        if cutlass.const_expr(self.bits < 6):
                            wb = (wa >> Uint32(4 * self.bits)) | (wb << Uint32(32 - 4 * self.bits))
                        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, self.bits)
                    else:
                        b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, self.bits)
                if cutlass.const_expr(self.e4m3):
                    e01 = _f16x2x2_to_e4m3x4(b0, b1)
                    e23 = _f16x2x2_to_e4m3x4(b2, b3)
                    _st_shared_v2_u32_offset(sw_base0, stage_bytes + k16_in_64 * 4 * w_tile_bytes, e01, e23)
                else:
                    _st_shared_v4_u32_offset(sw_base0, stage_bytes + k16_in_64 * 4 * w_tile_bytes, b0, b1, b2, b3)
            _mbar_arrive(bar_full + Int32(8))
            for k16_in_64 in cutlass.range_constexpr(4):
                off1 = (4 + k16_in_64) * 256 * self.bits
                w_s1[k16_in_64, 0] = _ld_global_nc_u32_offset(p0_cur, off1)
                if cutlass.const_expr(self.bits != 4):
                    w_s1[k16_in_64, 1] = _ld_global_nc_u32_offset(p1_cur, off1)
                    w_s1[k16_in_64, 2] = _ld_global_nc_u32_offset(p2_cur, off1)
            stage_cnt = stage_cnt + Int32(1)
        return stage_cnt


    @cute.jit
    def _prefill_consume_tile(self, tid: Int32, base: Int32,
                              scales_ptr: Int64, output_ptr: Int64,
                              rows: Int32, m0: Int32, nb64: Int32,
                              stage_cnt: Int32) -> Int32:
        w_tile_bytes = 256 if self.e4m3 else 512
        a_stage_bytes = _PREFILL_BM * 64 * 2
        stage_bytes = a_stage_bytes + 16 * w_tile_bytes
        bar_full = base + Int32(_PREFILL_STAGES * stage_bytes)
        bar_empty = bar_full + Int32(8 * _PREFILL_STAGES)
        lane, cw = tid & Int32(31), tid >> Int32(5)
        cw_m, cw_n = cw >> Int32(1), cw & Int32(1)
        l7, l15, l_sh4 = lane & Int32(7), lane & Int32(15), lane >> Int32(4)
        a_lane_off = (cw_m * Int32(32) + l15) * Int32(128) + ((l_sh4 ^ l7) << Int32(4))
        sw0 = base + Int32(a_stage_bytes) + cw_n * Int32(2 * w_tile_bytes) + lane * Int32(8 if self.e4m3 else 16)
        grand_total = cute.make_rmem_tensor((8, 4), Float32)
        grand_total.fill(Float32(0.0))
        for split in cutlass.range(self.splits, unroll=1):
            split_total = cute.make_rmem_tensor((8, 4), Float32)
            split_total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    scale = _side_value(scales_ptr, kb)
                    acc = cute.make_rmem_tensor((8, 4), Float32)
                    acc.fill(Float32(0.0))
                    phase = stage_cnt & Int32(1)
                    for s in cutlass.range_constexpr(2):
                        s_off = s * stage_bytes
                        _mbar_wait(bar_full + Int32(8 * s), phase)
                        for k16_in_64 in cutlass.range_constexpr(4):
                            a_addr = base + (a_lane_off ^ Int32(k16_in_64 * 32))
                            a0_0, a1_0, a2_0, a3_0 = _ldmatrix_m8n8x4_b16_offset(a_addr, s_off)
                            a0_1, a1_1, a2_1, a3_1 = _ldmatrix_m8n8x4_b16_offset(a_addr, s_off + 2048)
                            for n_local in cutlass.range_constexpr(2):
                                w_off = s_off + (k16_in_64 * 4 + n_local) * w_tile_bytes
                                if cutlass.const_expr(self.e4m3):
                                    e01, e23 = _ld_ring_v2_u32_offset(sw0, w_off)
                                    b0, b1 = _e4m3x4_to_f16x2x2(e01)
                                    b2, b3 = _e4m3x4_to_f16x2x2(e23)
                                else:
                                    b0, b1, b2, b3 = _ld_ring_v4_u32_offset(sw0, w_off)
                                i0 = 2 * n_local
                                i1 = 2 * n_local + 1
                                d0, d1, d2, d3 = self.mma(
                                    acc[i0, 0], acc[i0, 1], acc[i0, 2], acc[i0, 3],
                                    a0_0, a1_0, a2_0, a3_0, b0, b1)
                                acc[i0, 0], acc[i0, 1], acc[i0, 2], acc[i0, 3] = d0, d1, d2, d3
                                d0, d1, d2, d3 = self.mma(
                                    acc[i1, 0], acc[i1, 1], acc[i1, 2], acc[i1, 3],
                                    a0_0, a1_0, a2_0, a3_0, b2, b3)
                                acc[i1, 0], acc[i1, 1], acc[i1, 2], acc[i1, 3] = d0, d1, d2, d3
                                j0 = 4 + 2 * n_local
                                j1 = 4 + 2 * n_local + 1
                                d0, d1, d2, d3 = self.mma(
                                    acc[j0, 0], acc[j0, 1], acc[j0, 2], acc[j0, 3],
                                    a0_1, a1_1, a2_1, a3_1, b0, b1)
                                acc[j0, 0], acc[j0, 1], acc[j0, 2], acc[j0, 3] = d0, d1, d2, d3
                                d0, d1, d2, d3 = self.mma(
                                    acc[j1, 0], acc[j1, 1], acc[j1, 2], acc[j1, 3],
                                    a0_1, a1_1, a2_1, a3_1, b2, b3)
                                acc[j1, 0], acc[j1, 1], acc[j1, 2], acc[j1, 3] = d0, d1, d2, d3
                        _mbar_arrive(bar_empty + Int32(8 * s))
                    for n8 in cutlass.range_constexpr(8):
                        for v in cutlass.range_constexpr(4):
                            split_total[n8, v] = split_total[n8, v] + acc[n8, v] * scale
                    stage_cnt = stage_cnt + Int32(1)
            for n8 in cutlass.range_constexpr(8):
                for v in cutlass.range_constexpr(4):
                    grand_total[n8, v] = grand_total[n8, v] + split_total[n8, v]
        col_base = nb64 * Int32(64) + cw_n * Int32(32) + (lane & Int32(3)) * Int32(2)
        for m_sub in cutlass.range_constexpr(2):
            row0 = m0 + cw_m * Int32(32) + Int32(m_sub * 16) + (lane >> Int32(2))
            row1 = row0 + Int32(8)
            if row0 < rows:
                addr0 = output_ptr + (row0.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                for n8 in cutlass.range_constexpr(4):
                    idx = m_sub * 4 + n8
                    _st_global_cs_v2_f32_offset(addr0, n8 * 32, grand_total[idx, 0], grand_total[idx, 1])
            if row1 < rows:
                addr1 = output_ptr + (row1.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                for n8 in cutlass.range_constexpr(4):
                    idx = m_sub * 4 + n8
                    _st_global_cs_v2_f32_offset(addr1, n8 * 32, grand_total[idx, 2], grand_total[idx, 3])
        return stage_cnt


    @cute.kernel
    def multiply_prefill(self, inputs: cute.Tensor, payload: cute.Tensor,
                         scales: cute.Tensor, output: cute.Tensor,
                         rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._prefill_storage())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        w_tile_bytes = 256 if self.e4m3 else 512
        stage_bytes = _PREFILL_BM * 64 * 2 + 16 * w_tile_bytes
        bar_full = base + Int32(_PREFILL_STAGES * stage_bytes)
        bar_empty = bar_full + Int32(8 * _PREFILL_STAGES)
        if tid == Int32(0):
            for s in cutlass.range_constexpr(_PREFILL_STAGES):
                _mbar_init(bar_full + Int32(8 * s), Int32(2 * _PREFILL_PRODUCER_THREADS))
                _mbar_init(bar_empty + Int32(8 * s), Int32(_PREFILL_CONSUMER_THREADS))
            _fence_mbar_init()
        cute.arch.sync_threads()
        n_tiles = Int32(self.n // _PREFILL_BN)
        total_tiles = ((rows + Int32(_PREFILL_BM - 1)) // Int32(_PREFILL_BM)) * n_tiles
        x_ptr = inputs[rotation_id]
        payload_ptr = payload.iterator.toint()
        scales_ptr = scales.iterator.toint()
        output_ptr = output.iterator.toint()
        if tid >= Int32(_PREFILL_CONSUMER_THREADS):
            p_tid = tid - Int32(_PREFILL_CONSUMER_THREADS)
            stage_cnt = Int32(0)
            while task < total_tiles:
                m0 = (task // n_tiles) * Int32(_PREFILL_BM)
                nb64 = task % n_tiles
                stage_cnt = self._prefill_produce_tile(p_tid, base, x_ptr, payload_ptr, rows, m0, nb64, stage_cnt)
                task += step
        else:
            stage_cnt = Int32(0)
            while task < total_tiles:
                m0 = (task // n_tiles) * Int32(_PREFILL_BM)
                nb64 = task % n_tiles
                stage_cnt = self._prefill_consume_tile(tid, base, scales_ptr, output_ptr, rows, m0, nb64, stage_cnt)
                task += step


    @cute.kernel
    def multiply_prefill_grouped(self, inputs: cute.Tensor,
                                 payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                                 scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                                 outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                                 rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._prefill_storage())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        w_tile_bytes = 256 if self.e4m3 else 512
        stage_bytes = _PREFILL_BM * 64 * 2 + 16 * w_tile_bytes
        bar_full = base + Int32(_PREFILL_STAGES * stage_bytes)
        bar_empty = bar_full + Int32(8 * _PREFILL_STAGES)
        if tid == Int32(0):
            for s in cutlass.range_constexpr(_PREFILL_STAGES):
                _mbar_init(bar_full + Int32(8 * s), Int32(2 * _PREFILL_PRODUCER_THREADS))
                _mbar_init(bar_empty + Int32(8 * s), Int32(_PREFILL_CONSUMER_THREADS))
            _fence_mbar_init()
        cute.arch.sync_threads()
        n_tiles = Int32(self.n // _PREFILL_BN)
        proj_tiles = ((rows + Int32(_PREFILL_BM - 1)) // Int32(_PREFILL_BM)) * n_tiles
        total_tiles = Int32(4) * proj_tiles
        if tid >= Int32(_PREFILL_CONSUMER_THREADS):
            p_tid = tid - Int32(_PREFILL_CONSUMER_THREADS)
            stage_cnt = Int32(0)
            while task < total_tiles:
                proj = task // proj_tiles
                rem = task % proj_tiles
                m0 = (rem // n_tiles) * Int32(_PREFILL_BM)
                nb64 = rem % n_tiles
                x_ptr = inputs[rotation_id + proj]
                payload_ptr = select4(
                    proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                    payloads[2].iterator.toint(), payloads[3].iterator.toint())
                stage_cnt = self._prefill_produce_tile(p_tid, base, x_ptr, payload_ptr, rows, m0, nb64, stage_cnt)
                task += step
        else:
            stage_cnt = Int32(0)
            while task < total_tiles:
                proj = task // proj_tiles
                rem = task % proj_tiles
                m0 = (rem // n_tiles) * Int32(_PREFILL_BM)
                nb64 = rem % n_tiles
                scales_ptr = select4(
                    proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                    scales[2].iterator.toint(), scales[3].iterator.toint())
                output_ptr = select4(
                    proj, outputs[0].iterator.toint(), outputs[1].iterator.toint(),
                    outputs[2].iterator.toint(), outputs[3].iterator.toint())
                stage_cnt = self._prefill_consume_tile(tid, base, scales_ptr, output_ptr, rows, m0, nb64, stage_cnt)
                task += step


    @cute.jit
    def ws_produce_tile(
        self,
        m_block: Int32,
        nb: Int32,
        x_addr: Int64,
        payload_ptr: Int64,
        rows: Int32,
        pw: Int32,
        r_p: Int32,
        c_p: Int32,
        smem_a_base: Int32,
        prod_b_base: Int32,
        bar_full: Int32,
        bar_empty: Int32,
        off0: Int64,
        off1: Int64,
        off2: Int64,
        off3: Int64,
        shift0: Int32,
        shift1: Int32,
        step_k128: Int32,
    ) -> Int32:
        m_row0 = m_block * Int32(128) + r_p
        x_tile_base = x_addr + (m_row0.to(Int64) * Int64(self.k) + (c_p * Int32(8)).to(Int64)) * Int64(2)
        sz0 = (m_row0 < rows).to(Int32) * Int32(16)
        sz1 = ((m_row0 + Int32(32)) < rows).to(Int32) * Int32(16)
        sz2 = ((m_row0 + Int32(64)) < rows).to(Int32) * Int32(16)
        sz3 = ((m_row0 + Int32(96)) < rows).to(Int32) * Int32(16)
        kb_byte_stride = (
            (self.n // 16 * 8) * 32 * self.bits
            if self.input_units
            else 64 * 32 * self.bits
        )
        base_unit_tile = (
            nb * Int32(64)
            if cutlass.const_expr(self.input_units)
            else nb * Int32(self.k // 16 * 8)
        )
        pay_kb_base = payload_ptr + (base_unit_tile + pw * Int32(2)).to(Int64) * Int64(32 * self.bits)
        p0 = pay_kb_base + off0
        p1 = pay_kb_base + off1
        p2 = pay_kb_base + off2
        p3 = pay_kb_base + off3
        raw0 = cute.make_rmem_tensor(4, Uint32)
        raw1 = cute.make_rmem_tensor(4, Uint32)
        raw2 = cute.make_rmem_tensor(4, Uint32)
        raw3 = cute.make_rmem_tensor(4, Uint32)
        cur0 = cute.make_rmem_tensor(4, Uint32)
        cur1 = cute.make_rmem_tensor(4, Uint32)
        cur2 = cute.make_rmem_tensor(4, Uint32)
        cur3 = cute.make_rmem_tensor(4, Uint32)
        out_b0 = cute.make_rmem_tensor(4, Uint32)
        out_b1 = cute.make_rmem_tensor(4, Uint32)
        out_b2 = cute.make_rmem_tensor(4, Uint32)
        out_b3 = cute.make_rmem_tensor(4, Uint32)
        for t in cutlass.range_constexpr(4):
            dk = t // 2
            j = t % 2
            tile_off = (dk * 8 + j) * 32 * self.bits
            if cutlass.const_expr(self.bits == 4):
                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
            elif cutlass.const_expr(self.bits < 4):
                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
            elif cutlass.const_expr(self.bits == 5):
                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
                raw3[t] = ld_global_nc_u32_offset(p3, tile_off)
            else:
                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
        for kb in cutlass.range(self.k // 128, unroll=1):
            x_kb_base = x_tile_base + kb.to(Int64) * Int64(256)
            prod_parity = (step_k128 & Int32(1)) ^ Int32(1)
            for sub in cutlass.range_constexpr(4):
                for t in cutlass.range_constexpr(4):
                    cur0[t] = raw0[t]
                    if cutlass.const_expr(self.bits < 4):
                        cur1[t] = raw1[t]
                    elif cutlass.const_expr(self.bits == 5):
                        cur1[t] = raw1[t]
                        cur2[t] = raw2[t]
                        cur3[t] = raw3[t]
                    elif cutlass.const_expr(self.bits > 4):
                        cur1[t] = raw1[t]
                        cur2[t] = raw2[t]
                if cutlass.const_expr(sub < 3):
                    for t in cutlass.range_constexpr(4):
                        dk = t // 2
                        j = t % 2
                        tile_off = ((2 * (sub + 1) + dk) * 8 + j) * 32 * self.bits
                        if cutlass.const_expr(self.bits == 4):
                            raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                        elif cutlass.const_expr(self.bits < 4):
                            raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                            raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                        elif cutlass.const_expr(self.bits == 5):
                            raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                            raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                            raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
                            raw3[t] = ld_global_nc_u32_offset(p3, tile_off)
                        else:
                            raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                            raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                            raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
                else:
                    p0 += Int64(kb_byte_stride)
                    p1 += Int64(kb_byte_stride)
                    p2 += Int64(kb_byte_stride)
                    p3 += Int64(kb_byte_stride)
                    if kb + Int32(1) < Int32(self.k // 128):
                        for t in cutlass.range_constexpr(4):
                            dk = t // 2
                            j = t % 2
                            tile_off = (dk * 8 + j) * 32 * self.bits
                            if cutlass.const_expr(self.bits == 4):
                                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                            elif cutlass.const_expr(self.bits < 4):
                                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                            elif cutlass.const_expr(self.bits == 5):
                                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                                raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
                                raw3[t] = ld_global_nc_u32_offset(p3, tile_off)
                            else:
                                raw0[t] = ld_global_nc_u32_offset(p0, tile_off)
                                raw1[t] = ld_global_nc_u32_offset(p1, tile_off)
                                raw2[t] = ld_global_nc_u32_offset(p2, tile_off)
                _mbar_wait(bar_empty + Int32(sub * 8), prod_parity)
                cp_async4_zfill_shared_global_offset(
                    smem_a_base, sub * self.ws_stage_bytes + 0 * 2048,
                    x_kb_base, 0 * 32 * self.k * 2 + sub * 64, sz0,
                )
                cp_async4_zfill_shared_global_offset(
                    smem_a_base, sub * self.ws_stage_bytes + 1 * 2048,
                    x_kb_base, 1 * 32 * self.k * 2 + sub * 64, sz1,
                )
                cp_async4_zfill_shared_global_offset(
                    smem_a_base, sub * self.ws_stage_bytes + 2 * 2048,
                    x_kb_base, 2 * 32 * self.k * 2 + sub * 64, sz2,
                )
                cp_async4_zfill_shared_global_offset(
                    smem_a_base, sub * self.ws_stage_bytes + 3 * 2048,
                    x_kb_base, 3 * 32 * self.k * 2 + sub * 64, sz3,
                )
                _cp_async_mbar_arrive_noinc(bar_full + Int32(sub * 8))
                for t in cutlass.range_constexpr(4):
                    wa = Uint32(0)
                    wb = Uint32(0)
                    if cutlass.const_expr(self.bits == 4):
                        b = cur0[t]
                        a = cute.arch.shuffle_sync(b, shift0)
                        wa = b
                        wb = (a << Uint32(16)) | (b >> Uint32(16))
                    elif cutlass.const_expr(self.bits < 4):
                        window = (cur0[t].to(Int64) << Int64(32)) | cur1[t].to(Int64)
                        wa = (window >> shift0.to(Int64)).to(Uint32)
                        wb = (window >> shift1.to(Int64)).to(Uint32)
                    elif cutlass.const_expr(self.bits == 5):
                        wa = (((cur0[t].to(Int64) << Int64(32)) | cur1[t].to(Int64)) >> shift0.to(Int64)).to(Uint32)
                        wb = (((cur2[t].to(Int64) << Int64(32)) | cur3[t].to(Int64)) >> shift1.to(Int64)).to(Uint32)
                    else:
                        wa, wb = trellis_align_stream_u32x2(cur0[t], cur1[t], cur2[t], shift0, shift1)
                    if cutlass.const_expr(self.mul1):
                        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, self.bits)
                    elif cutlass.const_expr(self.bits == 6):
                        b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, self.bits)
                    else:
                        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, self.bits)
                    if cutlass.const_expr(self.e4m3):
                        out_b0[t] = pack_f16x2_pair_to_e4m3x4(b0, b1)
                        out_b1[t] = pack_f16x2_pair_to_e4m3x4(b2, b3)
                    else:
                        out_b0[t], out_b1[t], out_b2[t], out_b3[t] = b0, b1, b2, b3
                for dk in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(2):
                        t = 2 * dk + j
                        if cutlass.const_expr(self.e4m3):
                            st_shared_v2_u32_offset(
                                prod_b_base,
                                sub * self.ws_stage_bytes + dk * 2048 + j * 256,
                                out_b0[t],
                                out_b1[t],
                            )
                        else:
                            st_shared_v2_u32_offset(
                                prod_b_base,
                                sub * self.ws_stage_bytes + dk * 4096 + j * 512,
                                out_b0[t],
                                out_b1[t],
                            )
                            st_shared_v2_u32_offset(
                                prod_b_base,
                                sub * self.ws_stage_bytes + dk * 4096 + j * 512 + 256,
                                out_b2[t],
                                out_b3[t],
                            )
                _mbar_arrive(bar_full + Int32(sub * 8))
            step_k128 += Int32(1)
        return step_k128


    @cute.jit
    def ws_consume_tile(
        self,
        total: cute.Tensor,
        out_lo: cute.Tensor,
        m_block: Int32,
        nb: Int32,
        scales_ptr: Int64,
        output_ptr: Int64,
        rows: Int32,
        ring: Int32,
        cons_a_base0: Int32,
        cons_a_base1: Int32,
        cons_b_base: Int32,
        bar_full: Int32,
        step_k128: Int32,
    ) -> Int32:
        for split in cutlass.range(self.splits, unroll=1):
            total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    scale = _side_value(scales_ptr, kb)
                    acc = cute.make_rmem_tensor((4, 4, 4), Float32)
                    acc.fill(Float32(0.0))
                    cons_parity = step_k128 & Int32(1)
                    a0_frag = cute.make_rmem_tensor((4, 4), Uint32)
                    a1_frag = cute.make_rmem_tensor((4, 4), Uint32)
                    e0_frag = cute.make_rmem_tensor((2, 2), Uint32)
                    e1_frag = cute.make_rmem_tensor((2, 2), Uint32)
                    b0_frag = cute.make_rmem_tensor((2, 4), Uint32)
                    b1_frag = cute.make_rmem_tensor((2, 4), Uint32)
                    _mbar_wait(bar_full + Int32(0), cons_parity)
                    for m_local in cutlass.range_constexpr(4):
                        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(
                            cons_a_base0, m_local * 1024
                        )
                        a0_frag[m_local, 0] = a0
                        a0_frag[m_local, 1] = a1
                        a0_frag[m_local, 2] = a2
                        a0_frag[m_local, 3] = a3
                    for j in cutlass.range_constexpr(2):
                        if cutlass.const_expr(self.e4m3):
                            e01, e23 = ld_shared_v2_u32_offset(
                                cons_b_base, j * 256
                            )
                            e0_frag[j, 0] = e01
                            e0_frag[j, 1] = e23
                        else:
                            b0, b1 = ld_shared_v2_u32_offset(
                                cons_b_base, j * 512
                            )
                            b2, b3 = ld_shared_v2_u32_offset(
                                cons_b_base, j * 512 + 256
                            )
                            b0_frag[j, 0] = b0
                            b0_frag[j, 1] = b1
                            b0_frag[j, 2] = b2
                            b0_frag[j, 3] = b3
                    for sub in cutlass.range_constexpr(4):
                        for m_local in cutlass.range_constexpr(4):
                            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(
                                cons_a_base1, sub * self.ws_stage_bytes + m_local * 1024
                            )
                            a1_frag[m_local, 0] = a0
                            a1_frag[m_local, 1] = a1
                            a1_frag[m_local, 2] = a2
                            a1_frag[m_local, 3] = a3
                        for j in cutlass.range_constexpr(2):
                            if cutlass.const_expr(self.e4m3):
                                e01, e23 = ld_shared_v2_u32_offset(
                                    cons_b_base, sub * self.ws_stage_bytes + 2048 + j * 256
                                )
                                e1_frag[j, 0] = e01
                                e1_frag[j, 1] = e23
                            else:
                                b0, b1 = ld_shared_v2_u32_offset(
                                    cons_b_base, sub * self.ws_stage_bytes + 4096 + j * 512
                                )
                                b2, b3 = ld_shared_v2_u32_offset(
                                    cons_b_base, sub * self.ws_stage_bytes + 4096 + j * 512 + 256
                                )
                                b1_frag[j, 0] = b0
                                b1_frag[j, 1] = b1
                                b1_frag[j, 2] = b2
                                b1_frag[j, 3] = b3
                        for j in cutlass.range_constexpr(2):
                            if cutlass.const_expr(self.e4m3):
                                b0, b1 = unpack_e4m3x4_to_f16x2_pair(e0_frag[j, 0])
                                b2, b3 = unpack_e4m3x4_to_f16x2_pair(e0_frag[j, 1])
                            else:
                                b0, b1 = b0_frag[j, 0], b0_frag[j, 1]
                                b2, b3 = b0_frag[j, 2], b0_frag[j, 3]
                            for m_local in cutlass.range_constexpr(4):
                                a0 = a0_frag[m_local, 0]
                                a1 = a0_frag[m_local, 1]
                                a2 = a0_frag[m_local, 2]
                                a3 = a0_frag[m_local, 3]
                                d0, d1, d2, d3 = self.mma(
                                    acc[m_local, 2 * j, 0],
                                    acc[m_local, 2 * j, 1],
                                    acc[m_local, 2 * j, 2],
                                    acc[m_local, 2 * j, 3],
                                    a0, a1, a2, a3, b0, b1,
                                )
                                acc[m_local, 2 * j, 0], acc[m_local, 2 * j, 1], acc[m_local, 2 * j, 2], acc[m_local, 2 * j, 3] = d0, d1, d2, d3
                                d4, d5, d6, d7 = self.mma(
                                    acc[m_local, 2 * j + 1, 0],
                                    acc[m_local, 2 * j + 1, 1],
                                    acc[m_local, 2 * j + 1, 2],
                                    acc[m_local, 2 * j + 1, 3],
                                    a0, a1, a2, a3, b2, b3,
                                )
                                acc[m_local, 2 * j + 1, 0], acc[m_local, 2 * j + 1, 1], acc[m_local, 2 * j + 1, 2], acc[m_local, 2 * j + 1, 3] = d4, d5, d6, d7
                        _mbar_arrive(bar_full + Int32(sub * 8 + 8 * _WS_STAGES))
                        if cutlass.const_expr(sub < 3):
                            _mbar_wait(bar_full + Int32((sub + 1) * 8), cons_parity)
                            for m_local in cutlass.range_constexpr(4):
                                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(
                                    cons_a_base0, (sub + 1) * self.ws_stage_bytes + m_local * 1024
                                )
                                a0_frag[m_local, 0] = a0
                                a0_frag[m_local, 1] = a1
                                a0_frag[m_local, 2] = a2
                                a0_frag[m_local, 3] = a3
                            for j in cutlass.range_constexpr(2):
                                if cutlass.const_expr(self.e4m3):
                                    e01, e23 = ld_shared_v2_u32_offset(
                                        cons_b_base, (sub + 1) * self.ws_stage_bytes + j * 256
                                    )
                                    e0_frag[j, 0] = e01
                                    e0_frag[j, 1] = e23
                                else:
                                    b0, b1 = ld_shared_v2_u32_offset(
                                        cons_b_base, (sub + 1) * self.ws_stage_bytes + j * 512
                                    )
                                    b2, b3 = ld_shared_v2_u32_offset(
                                        cons_b_base, (sub + 1) * self.ws_stage_bytes + j * 512 + 256
                                    )
                                    b0_frag[j, 0] = b0
                                    b0_frag[j, 1] = b1
                                    b0_frag[j, 2] = b2
                                    b0_frag[j, 3] = b3
                        for j in cutlass.range_constexpr(2):
                            if cutlass.const_expr(self.e4m3):
                                b0, b1 = unpack_e4m3x4_to_f16x2_pair(e1_frag[j, 0])
                                b2, b3 = unpack_e4m3x4_to_f16x2_pair(e1_frag[j, 1])
                            else:
                                b0, b1 = b1_frag[j, 0], b1_frag[j, 1]
                                b2, b3 = b1_frag[j, 2], b1_frag[j, 3]
                            for m_local in cutlass.range_constexpr(4):
                                a0 = a1_frag[m_local, 0]
                                a1 = a1_frag[m_local, 1]
                                a2 = a1_frag[m_local, 2]
                                a3 = a1_frag[m_local, 3]
                                d0, d1, d2, d3 = self.mma(
                                    acc[m_local, 2 * j, 0],
                                    acc[m_local, 2 * j, 1],
                                    acc[m_local, 2 * j, 2],
                                    acc[m_local, 2 * j, 3],
                                    a0, a1, a2, a3, b0, b1,
                                )
                                acc[m_local, 2 * j, 0], acc[m_local, 2 * j, 1], acc[m_local, 2 * j, 2], acc[m_local, 2 * j, 3] = d0, d1, d2, d3
                                d4, d5, d6, d7 = self.mma(
                                    acc[m_local, 2 * j + 1, 0],
                                    acc[m_local, 2 * j + 1, 1],
                                    acc[m_local, 2 * j + 1, 2],
                                    acc[m_local, 2 * j + 1, 3],
                                    a0, a1, a2, a3, b2, b3,
                                )
                                acc[m_local, 2 * j + 1, 0], acc[m_local, 2 * j + 1, 1], acc[m_local, 2 * j + 1, 2], acc[m_local, 2 * j + 1, 3] = d4, d5, d6, d7
                    for m_local in cutlass.range_constexpr(4):
                        for n in cutlass.range_constexpr(4):
                            for v in cutlass.range_constexpr(4):
                                total[m_local, n, v] = total[m_local, n, v] + acc[m_local, n, v] * scale
                    step_k128 += Int32(1)
            if cutlass.const_expr(self.splits > 1):
                cons_sc_base = ring + Int32(self.ws_sc_off) + Int32(cute.arch.thread_idx()[0]) * Int32(8)
                if split == Int32(0):
                    for m_local in cutlass.range_constexpr(self.ws_sc_m_start):
                        for n in cutlass.range_constexpr(4):
                            for v in cutlass.range_constexpr(4):
                                out_lo[m_local, n, v] = add_rn_f32(Float32(0.0), total[m_local, n, v])
                    for sc_idx in cutlass.range_constexpr(4 - self.ws_sc_m_start):
                        m_idx = self.ws_sc_m_start + sc_idx
                        for n in cutlass.range_constexpr(4):
                            st_shared_v2_f32_offset(
                                cons_sc_base,
                                (sc_idx * 8 + n * 2) * 2048,
                                add_rn_f32(Float32(0.0), total[m_idx, n, 0]),
                                add_rn_f32(Float32(0.0), total[m_idx, n, 1]),
                            )
                            st_shared_v2_f32_offset(
                                cons_sc_base,
                                (sc_idx * 8 + n * 2 + 1) * 2048,
                                add_rn_f32(Float32(0.0), total[m_idx, n, 2]),
                                add_rn_f32(Float32(0.0), total[m_idx, n, 3]),
                            )
                else:
                    for m_local in cutlass.range_constexpr(self.ws_sc_m_start):
                        for n in cutlass.range_constexpr(4):
                            for v in cutlass.range_constexpr(4):
                                out_lo[m_local, n, v] = add_rn_f32(out_lo[m_local, n, v], total[m_local, n, v])
                    for sc_idx in cutlass.range_constexpr(4 - self.ws_sc_m_start):
                        m_idx = self.ws_sc_m_start + sc_idx
                        sc_frag = cute.make_rmem_tensor((4, 4), Float32)
                        for n in cutlass.range_constexpr(4):
                            sc_frag[n, 0], sc_frag[n, 1] = ld_shared_v2_f32_offset(
                                cons_sc_base, (sc_idx * 8 + n * 2) * 2048
                            )
                            sc_frag[n, 2], sc_frag[n, 3] = ld_shared_v2_f32_offset(
                                cons_sc_base, (sc_idx * 8 + n * 2 + 1) * 2048
                            )
                        for n in cutlass.range_constexpr(4):
                            st_shared_v2_f32_offset(
                                cons_sc_base,
                                (sc_idx * 8 + n * 2) * 2048,
                                add_rn_f32(sc_frag[n, 0], total[m_idx, n, 0]),
                                add_rn_f32(sc_frag[n, 1], total[m_idx, n, 1]),
                            )
                            st_shared_v2_f32_offset(
                                cons_sc_base,
                                (sc_idx * 8 + n * 2 + 1) * 2048,
                                add_rn_f32(sc_frag[n, 2], total[m_idx, n, 2]),
                                add_rn_f32(sc_frag[n, 3], total[m_idx, n, 3]),
                            )
        tid_out = Int32(cute.arch.thread_idx()[0])
        lane_out = tid_out & Int32(31)
        cw_out = tid_out >> Int32(5)
        row_base = m_block * Int32(128) + (cw_out >> Int32(2)) * Int32(64) + (lane_out >> Int32(2))
        col_base = nb * Int32(128) + (cw_out & Int32(3)) * Int32(32) + (lane_out & Int32(3)) * Int32(2)
        out_col_base = output_ptr + col_base.to(Int64) * Int64(4)
        cons_sc_base = ring + Int32(self.ws_sc_off) + tid_out * Int32(8)
        for m_local in cutlass.range_constexpr(4):
            row0 = row_base + Int32(m_local * 16)
            row1 = row0 + Int32(8)
            if cutlass.const_expr(self.splits <= 1):
                if row0 < rows:
                    r0_addr = out_col_base + row0.to(Int64) * Int64(self.n * 4)
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(
                            r0_addr + Int64(n * 32),
                            add_rn_f32(Float32(0.0), total[m_local, n, 0]),
                            add_rn_f32(Float32(0.0), total[m_local, n, 1]),
                        )
                if row1 < rows:
                    r1_addr = out_col_base + row1.to(Int64) * Int64(self.n * 4)
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(
                            r1_addr + Int64(n * 32),
                            add_rn_f32(Float32(0.0), total[m_local, n, 2]),
                            add_rn_f32(Float32(0.0), total[m_local, n, 3]),
                        )
            elif cutlass.const_expr(m_local < self.ws_sc_m_start):
                if row0 < rows:
                    r0_addr = out_col_base + row0.to(Int64) * Int64(self.n * 4)
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(
                            r0_addr + Int64(n * 32),
                            out_lo[m_local, n, 0],
                            out_lo[m_local, n, 1],
                        )
                if row1 < rows:
                    r1_addr = out_col_base + row1.to(Int64) * Int64(self.n * 4)
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(
                            r1_addr + Int64(n * 32),
                            out_lo[m_local, n, 2],
                            out_lo[m_local, n, 3],
                        )
            else:
                sc_idx = m_local - self.ws_sc_m_start
                if row0 < rows:
                    r0_addr = out_col_base + row0.to(Int64) * Int64(self.n * 4)
                    sc0 = cute.make_rmem_tensor((4, 2), Float32)
                    for n in cutlass.range_constexpr(4):
                        sc0[n, 0], sc0[n, 1] = ld_shared_v2_f32_offset(
                            cons_sc_base, (sc_idx * 8 + n * 2) * 2048
                        )
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(r0_addr + Int64(n * 32), sc0[n, 0], sc0[n, 1])
                if row1 < rows:
                    r1_addr = out_col_base + row1.to(Int64) * Int64(self.n * 4)
                    sc1 = cute.make_rmem_tensor((4, 2), Float32)
                    for n in cutlass.range_constexpr(4):
                        sc1[n, 0], sc1[n, 1] = ld_shared_v2_f32_offset(
                            cons_sc_base, (sc_idx * 8 + n * 2 + 1) * 2048
                        )
                    for n in cutlass.range_constexpr(4):
                        st_global_v2_f32(r1_addr + Int64(n * 32), sc1[n, 0], sc1[n, 1])
        return step_k128


    @cute.kernel
    def multiply_prefill_ws(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
    ):
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._ws_storage())
        ring = shared_ptr_to_u32(storage.words.data_ptr())
        bar_full = ring + Int32(self.ws_bar_off)
        bar_empty = bar_full + Int32(8 * _WS_STAGES)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        if tid == Int32(0):
            for s in cutlass.range_constexpr(_WS_STAGES):
                _mbar_init(bar_full + Int32(8 * s), Int32(2 * _WS_PRODUCER_THREADS))
                _mbar_init(bar_empty + Int32(8 * s), Int32(_WS_CONSUMER_THREADS))
            _fence_mbar_init()
        cute.arch.sync_threads()
        m_tiles = (rows + Int32(127)) // Int32(128)
        n_tiles = Int32(self.n // 128)
        total_tasks = m_tiles * n_tiles
        step_k128 = Int32(0)
        if tid >= Int32(_WS_CONSUMER_THREADS):
            cute.arch.warpgroup_reg_dealloc(_WS_PRODUCER_REGS)
            ptid = tid - Int32(_WS_CONSUMER_THREADS)
            pw = ptid >> Int32(5)
            lane = ptid & Int32(31)
            r_p = ptid >> Int32(2)
            c_p = ptid & Int32(3)
            smem_a_base = ring + r_p * Int32(64) + (c_p ^ ((r_p >> Int32(1)) & Int32(3))) * Int32(16)
            prod_b_base = (
                ring + Int32(_WS_A_STAGE_BYTES) + pw * Int32(512) + lane * Int32(8)
                if cutlass.const_expr(self.e4m3)
                else ring + Int32(_WS_A_STAGE_BYTES) + pw * Int32(1024) + lane * Int32(8)
            )
            off0 = Int64(0)
            off1 = Int64(0)
            off2 = Int64(0)
            off3 = Int64(0)
            shift0 = Int32(0)
            shift1 = Int32(0)
            if cutlass.const_expr(self.bits == 4):
                off0 = lane.to(Int64) * Int64(4)
                shift0 = (lane + Int32(31)) & Int32(31)
            elif cutlass.const_expr(self.bits < 4):
                ia, ib, shift0, _ = trellis256_lane_geom_bits(lane, 0, 8, self.bits)
                off0 = ia.to(Int64) * Int64(4)
                off1 = ib.to(Int64) * Int64(4)
                shift1 = shift0 + Int32(4 * self.bits)
            elif cutlass.const_expr(self.bits == 5):
                ib0, ib1, shift1, _ = trellis256_lane_geom_bits(lane, 0, 4, self.bits)
                ia0, ia1, shift0, _ = trellis256_lane_geom_bits(lane, 4, 4, self.bits)
                off0 = ia0.to(Int64) * Int64(4)
                off1 = ia1.to(Int64) * Int64(4)
                off2 = ib0.to(Int64) * Int64(4)
                off3 = ib1.to(Int64) * Int64(4)
            else:
                i0, i2, shift0, shift1 = trellis256_lane_geom_bits(lane, 0, 8, self.bits)
                i1 = i0 + Int32(1)
                i1 -= Int32(8 * self.bits) * (i1 >= Int32(8 * self.bits)).to(Int32)
                off0 = i0.to(Int64) * Int64(4)
                off1 = i1.to(Int64) * Int64(4)
                off2 = i2.to(Int64) * Int64(4)
            x_addr = inputs[rotation_id]
            payload_ptr = payload.iterator.toint()
            while task < total_tasks:
                g_m_base = (task // (n_tiles * Int32(4))) * Int32(4)
                rem = task - g_m_base * n_tiles
                nb = rem >> Int32(2)
                m_block = g_m_base + (rem & Int32(3))
                if g_m_base + Int32(4) > m_tiles:
                    tile_m = m_tiles - g_m_base
                    nb = rem // tile_m
                    m_block = g_m_base + (rem % tile_m)
                step_k128 = self.ws_produce_tile(
                    m_block, nb, x_addr, payload_ptr, rows, pw, r_p, c_p,
                    smem_a_base, prod_b_base, bar_full, bar_empty,
                    off0, off1, off2, off3, shift0, shift1, step_k128,
                )
                task += step
        else:
            cute.arch.warpgroup_reg_alloc(_WS_CONSUMER_REGS)
            lane = tid & Int32(31)
            cw = tid >> Int32(5)
            m_warp = cw >> Int32(2)
            n_warp = cw & Int32(3)
            r_m = lane & Int32(15)
            c_half = lane >> Int32(4)
            cons_a_base0 = ring + m_warp * Int32(4096) + r_m * Int32(64) + (c_half ^ ((r_m >> Int32(1)) & Int32(3))) * Int32(16)
            cons_a_base1 = ring + m_warp * Int32(4096) + r_m * Int32(64) + ((Int32(2) + c_half) ^ ((r_m >> Int32(1)) & Int32(3))) * Int32(16)
            cons_b_base = (
                ring + Int32(_WS_A_STAGE_BYTES) + n_warp * Int32(512) + lane * Int32(8)
                if cutlass.const_expr(self.e4m3)
                else ring + Int32(_WS_A_STAGE_BYTES) + n_warp * Int32(1024) + lane * Int32(8)
            )
            scales_ptr = scales.iterator.toint()
            output_ptr = output.iterator.toint()
            total = cute.make_rmem_tensor((4, 4, 4), Float32)
            out_lo = cute.make_rmem_tensor((max(1, self.ws_sc_m_start), 4, 4), Float32)
            while task < total_tasks:
                g_m_base = (task // (n_tiles * Int32(4))) * Int32(4)
                rem = task - g_m_base * n_tiles
                nb = rem >> Int32(2)
                m_block = g_m_base + (rem & Int32(3))
                if g_m_base + Int32(4) > m_tiles:
                    tile_m = m_tiles - g_m_base
                    nb = rem // tile_m
                    m_block = g_m_base + (rem % tile_m)
                step_k128 = self.ws_consume_tile(
                    total, out_lo, m_block, nb, scales_ptr, output_ptr, rows,
                    ring, cons_a_base0, cons_a_base1, cons_b_base,
                    bar_full, step_k128,
                )
                task += step


    @cute.kernel
    def multiply_prefill_ws_grouped(
        self,
        inputs: cute.Tensor,
        payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        rows: Int32,
        rotation_id: Int32,
    ):
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._ws_storage())
        ring = shared_ptr_to_u32(storage.words.data_ptr())
        bar_full = ring + Int32(self.ws_bar_off)
        bar_empty = bar_full + Int32(8 * _WS_STAGES)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        if tid == Int32(0):
            for s in cutlass.range_constexpr(_WS_STAGES):
                _mbar_init(bar_full + Int32(8 * s), Int32(2 * _WS_PRODUCER_THREADS))
                _mbar_init(bar_empty + Int32(8 * s), Int32(_WS_CONSUMER_THREADS))
            _fence_mbar_init()
        cute.arch.sync_threads()
        m_tiles = (rows + Int32(127)) // Int32(128)
        n_per_proj = Int32(self.n // 128)
        group_n_tiles = Int32(4 * (self.n // 128))
        total_tasks = m_tiles * group_n_tiles
        step_k128 = Int32(0)
        if tid >= Int32(_WS_CONSUMER_THREADS):
            cute.arch.warpgroup_reg_dealloc(_WS_PRODUCER_REGS)
            ptid = tid - Int32(_WS_CONSUMER_THREADS)
            pw = ptid >> Int32(5)
            lane = ptid & Int32(31)
            r_p = ptid >> Int32(2)
            c_p = ptid & Int32(3)
            smem_a_base = ring + r_p * Int32(64) + (c_p ^ ((r_p >> Int32(1)) & Int32(3))) * Int32(16)
            prod_b_base = (
                ring + Int32(_WS_A_STAGE_BYTES) + pw * Int32(512) + lane * Int32(8)
                if cutlass.const_expr(self.e4m3)
                else ring + Int32(_WS_A_STAGE_BYTES) + pw * Int32(1024) + lane * Int32(8)
            )
            off0 = Int64(0)
            off1 = Int64(0)
            off2 = Int64(0)
            off3 = Int64(0)
            shift0 = Int32(0)
            shift1 = Int32(0)
            if cutlass.const_expr(self.bits == 4):
                off0 = lane.to(Int64) * Int64(4)
                shift0 = (lane + Int32(31)) & Int32(31)
            elif cutlass.const_expr(self.bits < 4):
                ia, ib, shift0, _ = trellis256_lane_geom_bits(lane, 0, 8, self.bits)
                off0 = ia.to(Int64) * Int64(4)
                off1 = ib.to(Int64) * Int64(4)
                shift1 = shift0 + Int32(4 * self.bits)
            elif cutlass.const_expr(self.bits == 5):
                ib0, ib1, shift1, _ = trellis256_lane_geom_bits(lane, 0, 4, self.bits)
                ia0, ia1, shift0, _ = trellis256_lane_geom_bits(lane, 4, 4, self.bits)
                off0 = ia0.to(Int64) * Int64(4)
                off1 = ia1.to(Int64) * Int64(4)
                off2 = ib0.to(Int64) * Int64(4)
                off3 = ib1.to(Int64) * Int64(4)
            else:
                i0, i2, shift0, shift1 = trellis256_lane_geom_bits(lane, 0, 8, self.bits)
                i1 = i0 + Int32(1)
                i1 -= Int32(8 * self.bits) * (i1 >= Int32(8 * self.bits)).to(Int32)
                off0 = i0.to(Int64) * Int64(4)
                off1 = i1.to(Int64) * Int64(4)
                off2 = i2.to(Int64) * Int64(4)
            while task < total_tasks:
                g_m_base = (task // (group_n_tiles * Int32(4))) * Int32(4)
                rem = task - g_m_base * group_n_tiles
                global_nb = rem >> Int32(2)
                m_block = g_m_base + (rem & Int32(3))
                if g_m_base + Int32(4) > m_tiles:
                    tile_m = m_tiles - g_m_base
                    global_nb = rem // tile_m
                    m_block = g_m_base + (rem % tile_m)
                proj = global_nb // n_per_proj
                nb = global_nb % n_per_proj
                x_addr = inputs[rotation_id + proj]
                payload_ptr = select4(
                    proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                    payloads[2].iterator.toint(), payloads[3].iterator.toint(),
                )
                step_k128 = self.ws_produce_tile(
                    m_block, nb, x_addr, payload_ptr, rows, pw, r_p, c_p,
                    smem_a_base, prod_b_base, bar_full, bar_empty,
                    off0, off1, off2, off3, shift0, shift1, step_k128,
                )
                task += step
        else:
            cute.arch.warpgroup_reg_alloc(_WS_CONSUMER_REGS)
            lane = tid & Int32(31)
            cw = tid >> Int32(5)
            m_warp = cw >> Int32(2)
            n_warp = cw & Int32(3)
            r_m = lane & Int32(15)
            c_half = lane >> Int32(4)
            cons_a_base0 = ring + m_warp * Int32(4096) + r_m * Int32(64) + (c_half ^ ((r_m >> Int32(1)) & Int32(3))) * Int32(16)
            cons_a_base1 = ring + m_warp * Int32(4096) + r_m * Int32(64) + ((Int32(2) + c_half) ^ ((r_m >> Int32(1)) & Int32(3))) * Int32(16)
            cons_b_base = (
                ring + Int32(_WS_A_STAGE_BYTES) + n_warp * Int32(512) + lane * Int32(8)
                if cutlass.const_expr(self.e4m3)
                else ring + Int32(_WS_A_STAGE_BYTES) + n_warp * Int32(1024) + lane * Int32(8)
            )
            total = cute.make_rmem_tensor((4, 4, 4), Float32)
            out_lo = cute.make_rmem_tensor((max(1, self.ws_sc_m_start), 4, 4), Float32)
            while task < total_tasks:
                g_m_base = (task // (group_n_tiles * Int32(4))) * Int32(4)
                rem = task - g_m_base * group_n_tiles
                global_nb = rem >> Int32(2)
                m_block = g_m_base + (rem & Int32(3))
                if g_m_base + Int32(4) > m_tiles:
                    tile_m = m_tiles - g_m_base
                    global_nb = rem // tile_m
                    m_block = g_m_base + (rem % tile_m)
                proj = global_nb // n_per_proj
                nb = global_nb % n_per_proj
                scales_ptr = select4(
                    proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                    scales[2].iterator.toint(), scales[3].iterator.toint(),
                )
                output_ptr = select4(
                    proj, outputs[0].iterator.toint(), outputs[1].iterator.toint(),
                    outputs[2].iterator.toint(), outputs[3].iterator.toint(),
                )
                step_k128 = self.ws_consume_tile(
                    total, out_lo, m_block, nb, scales_ptr, output_ptr, rows,
                    ring, cons_a_base0, cons_a_base1, cons_b_base,
                    bar_full, step_k128,
                )
                task += step


    @cute.jit
    def call_grouped(self, inputs: cute.Tensor,
                     payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     partial: cute.Tensor,
                     outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     rows: Int32, rotation_id: Int32, cta_limit: Int32,
                     stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id + Int32(3) < Int32(self.rotations) and cta_limit > Int32(0):
            if cutlass.const_expr(self.e4m3 and self.max_rows > 64):
                if rows > Int32(64):
                    tiles = ((rows + Int32(127)) // Int32(128)) * Int32(4 * (self.n // 128))
                    self.multiply_prefill_ws_grouped(inputs, payloads, scales, outputs, rows, rotation_id).launch(
                        grid=(min(tiles, max(Int32(1), cta_limit // Int32(self.CTAS_PER_SM))), 1, 1),
                        block=(_WS_CONSUMER_THREADS + _WS_PRODUCER_THREADS, 1, 1),
                        min_blocks_per_mp=1, stream=stream)
                else:
                    self._generic_grouped(inputs, payloads, scales, partial, outputs, rows, rotation_id, cta_limit, stream)
            else:
                self._generic_grouped(inputs, payloads, scales, partial, outputs, rows, rotation_id, cta_limit, stream)

    @cute.jit
    def _generic_grouped(self, inputs: cute.Tensor,
                     payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     partial: cute.Tensor,
                     outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     rows: Int32, rotation_id: Int32, cta_limit: Int32,
                     stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id + Int32(3) < Int32(self.rotations) and cta_limit > Int32(0):
            group_tiles = Int32(4 * (self.n // 128) * self.splits)
            if cutlass.const_expr(self.max_rows >= _PREFILL_MIN_ROWS):
                if rows >= Int32(_PREFILL_MIN_ROWS):
                    prefill_tiles = Int32(4) * ((rows + Int32(_PREFILL_BM - 1)) // Int32(_PREFILL_BM)) * Int32(self.n // _PREFILL_BN)
                    self.multiply_prefill_grouped(inputs, payloads, scales, outputs, rows, rotation_id).launch(
                        grid=(min(prefill_tiles, max(Int32(1), cta_limit // Int32(4))), 1, 1),
                        block=(_PREFILL_CONSUMER_THREADS + _PREFILL_PRODUCER_THREADS, 1, 1),
                        min_blocks_per_mp=1, stream=stream)
                else:
                    if cutlass.const_expr(self.bits < 5):
                        self.multiply_grouped(inputs, payloads, scales, partial, rows, rotation_id).launch(
                            grid=(min(((rows + Int32(15)) // Int32(16)) * group_tiles, cta_limit), 1, 1),
                            block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
                    elif rows <= Int32(16):
                        self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 1, 8).launch(
                            grid=(min(group_tiles, cta_limit), 1, 1),
                            block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
                    elif rows <= Int32(32):
                        self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 2, 4).launch(
                            grid=(min(group_tiles, cta_limit), 1, 1),
                            block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
                    elif rows <= Int32(48):
                        self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 3, 8 if self.mul1 else 4).launch(
                            grid=(min(group_tiles, cta_limit), 1, 1),
                            block=(128, 1, 1), min_blocks_per_mp=2 if self.mul1 else 3, stream=stream)
                    else:
                        self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 4, 4).launch(
                            grid=(min(((rows + Int32(63)) // Int32(64)) * group_tiles, cta_limit), 1, 1),
                            block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
                    self.reduce_grouped(partial, outputs, rows).launch(
                        grid=(min(rows * Int32(4 * (self.n // 128)), cta_limit), 1, 1),
                        block=(128, 1, 1), stream=stream)
            else:
                if cutlass.const_expr(self.bits < 5):
                    self.multiply_grouped(inputs, payloads, scales, partial, rows, rotation_id).launch(
                        grid=(min(((rows + Int32(15)) // Int32(16)) * group_tiles, cta_limit), 1, 1),
                        block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
                elif rows <= Int32(16):
                    self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 1, 8).launch(
                        grid=(min(group_tiles, cta_limit), 1, 1),
                        block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
                elif rows <= Int32(32):
                    self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 2, 4).launch(
                        grid=(min(group_tiles, cta_limit), 1, 1),
                        block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
                elif rows <= Int32(48):
                    self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 3, 8 if self.mul1 else 4).launch(
                        grid=(min(group_tiles, cta_limit), 1, 1),
                        block=(128, 1, 1), min_blocks_per_mp=2 if self.mul1 else 3, stream=stream)
                else:
                    self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 4, 4).launch(
                        grid=(min(((rows + Int32(63)) // Int32(64)) * group_tiles, cta_limit), 1, 1),
                        block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
                self.reduce_grouped(partial, outputs, rows).launch(
                    grid=(min(rows * Int32(4 * (self.n // 128)), cta_limit), 1, 1),
                    block=(128, 1, 1), stream=stream)

    @cute.kernel
    def multiply_grouped(self, inputs: cute.Tensor,
                         payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                         scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                         partial: cute.Tensor, rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        tiles = Int32(self.n // 128 * self.splits)
        group_tiles = Int32(4 * (self.n // 128) * self.splits)
        while task < ((rows + Int32(15)) // Int32(16)) * group_tiles:
            group = task // group_tiles
            policy = weight_stream_policy((group + Int32(1) == (rows + Int32(15)) // Int32(16)).to(Int32))
            rem = task % group_tiles
            proj = rem // tiles
            tile_in_proj = rem % tiles
            split = tile_in_proj % Int32(self.splits)
            nb = tile_in_proj // Int32(self.splits)
            x = inputs[rotation_id + proj]
            payload_ptr = select4(
                proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                payloads[2].iterator.toint(), payloads[3].iterator.toint(),
            )
            scales_ptr = select4(
                proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                scales[2].iterator.toint(), scales[3].iterator.toint(),
            )
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            row0 = group * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(2):
                            tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                            if cutlass.const_expr(self.input_units):
                                tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                            b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = self.mma(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales_ptr, kb)
                    for n in cutlass.range_constexpr(4):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
            for n in cutlass.range_constexpr(4):
                col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                base = proj_partial + (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                if row0 < rows:
                    partial[base], partial[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 < rows:
                    base += Int64(8 * self.splits * self.n)
                    partial[base], partial[base + Int64(1)] = total[n, 2], total[n, 3]
            task += step

    @cute.kernel
    def multiply_wide_grouped(self, inputs: cute.Tensor,
                              payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                              scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                              partial: cute.Tensor, rows: Int32, rotation_id: Int32,
                              active: cutlass.Constexpr, unroll_k16: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        c, r_grp = tid & Int32(15), tid >> Int32(4)
        M = 16 * active
        M_PAD = M + 1
        tiles = Int32(self.n // 128 * self.splits)
        group_tiles = Int32(4 * (self.n // 128) * self.splits)
        limit = group_tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * group_tiles
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage(active))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)
        while task < limit:
            group = Int32(0) if cutlass.const_expr(active < 4) else task // group_tiles
            last_reader = Int32(1)
            if cutlass.const_expr(active == 4):
                last_reader = (group + Int32(1) == (rows + Int32(M - 1)) // Int32(M)).to(Int32)
            policy = weight_stream_policy(last_reader)
            rem = task if cutlass.const_expr(active < 4) else task % group_tiles
            proj = rem // tiles
            tile_in_proj = rem % tiles
            split = tile_in_proj % Int32(self.splits)
            nb = tile_in_proj // Int32(self.splits)
            total = cute.make_rmem_tensor((active, 4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    x = inputs[rotation_id + proj]
                    k_col = kb * Int32(128) + c * Int32(8)
                    for s in cutlass.range_constexpr(2 * active):
                        row = r_grp + Int32(s * 8) if cutlass.const_expr(active < 4) else group * Int32(M) + r_grp + Int32(s * 8)
                        v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if cutlass.const_expr(active in (2, 3) and s < 2 * (active - 1)):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        else:
                            if row < rows:
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                    cute.arch.sync_threads()
                    scale = _side_value(select4(
                        proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                        scales[2].iterator.toint(), scales[3].iterator.toint(),
                    ), kb)
                    payload_ptr = select4(
                        proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                        payloads[2].iterator.toint(), payloads[3].iterator.toint(),
                    )
                    for n16 in cutlass.range_constexpr(2):
                        acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                        acc.fill(Float32(0.0))
                        if cutlass.const_expr(unroll_k16 == 8):
                            for k16 in cutlass.range_constexpr(8):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    a0 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                    a1 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                    a2 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                    a3 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = self.mma(
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                        a0, a1, a2, a3, b2, b3)
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                        else:
                            for ko in cutlass.range(8 // unroll_k16, unroll=1):
                                sx_k = sx_read + ko * Int32(2 * unroll_k16 * M_PAD * 16)
                                for ki in cutlass.range_constexpr(unroll_k16):
                                    k16 = ko * Int32(unroll_k16) + Int32(ki)
                                    tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + k16) * Int32(8) + warp * Int32(2) + Int32(n16)
                                    if cutlass.const_expr(self.input_units):
                                        tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + k16 * Int32(8) + warp * Int32(2) + Int32(n16)
                                    b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                    for mb in cutlass.range_constexpr(active):
                                        a0 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16) * 16))
                                        a1 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16 + 8) * 16))
                                        a2 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16) * 16))
                                        a3 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16 + 8) * 16))
                                        d0, d1, d2, d3 = self.mma(
                                            acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                            a0, a1, a2, a3, b0, b1)
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                        d0, d1, d2, d3 = self.mma(
                                            acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                            a0, a1, a2, a3, b2, b3)
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                        for mb in cutlass.range_constexpr(active):
                            for v in cutlass.range_constexpr(4):
                                total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    cute.arch.sync_threads()
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            row_base = (lane >> Int32(2)) if cutlass.const_expr(active < 4) else group * Int32(M) + (lane >> Int32(2))
            for mb in cutlass.range_constexpr(active):
                row0 = row_base + Int32(mb * 16)
                row1 = row0 + Int32(8)
                for n in cutlass.range_constexpr(4):
                    col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                    base = proj_partial + (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                    if cutlass.const_expr(active < 4 and mb < active - 1):
                        partial[base], partial[base + Int64(1)] = total[mb, n, 0], total[mb, n, 1]
                        base += Int64(8 * self.splits * self.n)
                        partial[base], partial[base + Int64(1)] = total[mb, n, 2], total[mb, n, 3]
                    else:
                        if row0 < rows:
                            partial[base], partial[base + Int64(1)] = total[mb, n, 0], total[mb, n, 1]
                        if row1 < rows:
                            base += Int64(8 * self.splits * self.n)
                            partial[base], partial[base + Int64(1)] = total[mb, n, 2], total[mb, n, 3]
            task += step

    @cute.kernel
    def reduce_grouped(self, partial: cute.Tensor,
                       outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                       rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        n_blocks = Int32(self.n // 128)
        group_blocks = Int32(4 * (self.n // 128))
        while task < rows * group_blocks:
            row = task // group_blocks
            rem = task % group_blocks
            proj = rem // n_blocks
            col = (rem % n_blocks) * Int32(128) + Int32(tid)
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            total = Float32(0.0)
            for split in cutlass.range_constexpr(self.splits):
                total += partial[proj_partial + (row.to(Int64) * Int64(self.splits) + Int64(split)) * Int64(self.n) + col.to(Int64)]
            out_idx = row.to(Int64) * Int64(self.n) + col.to(Int64)
            if proj == Int32(0):
                outputs[0][out_idx] = total
            elif proj == Int32(1):
                outputs[1][out_idx] = total
            elif proj == Int32(2):
                outputs[2][out_idx] = total
            else:
                outputs[3][out_idx] = total
            task += step


@cute.jit
def select4(index: Int32, v0: Int64, v1: Int64, v2: Int64, v3: Int64) -> Int64:
    out = v3
    if index == Int32(0):
        out = v0
    elif index == Int32(1):
        out = v1
    elif index == Int32(2):
        out = v2
    return out
