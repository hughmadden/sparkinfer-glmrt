"""Expert-major M64 EXLR8 prefill, retaining the K1 reduction tree.

Each 8-warp CTA owns a full N128 unit (N16 per warp) and up to four M16 tiles.
Each warp decodes its full 16x16 trellis tile once per K16 step and feeds both
N8 MMA halves from a single ldmatrix of A. Live M16 tiles are compacted per
group, A and B are staged with cp.async, a K128 accumulator is scaled
with one pinned FMA, and FC1 split partials fold in ascending split order.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_bulk_g2s_mbar,
    cp_async_bulk_g2s_mbar_l2hint,
    create_l2_evict_first_policy,
    ld_shared_i32_relaxed,
    mma_m16n8k32_f32_e4m3,
    st_shared_i32,
    st_global_u32,
    st_global_v4_u32,
    ld_global_nc_v2_u32,
    f16x2_to_f32x2,
    f16_mma_m16n8k16_f32,
    half2_to_float2_scaled,
    ld_global_b16,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    packed_dequant_trellis_to_half2x4,
    shared_ptr_to_u32,
    st_global_v2_u32,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad
from .mixed_trellis_ws import (
    _mbar_init, _fence_mbar_init, _mbar_wait, _mbar_arrive,
    _mbar_arrive_expect_tx, _cp_async_mbar_arrive_noinc,
)

# FC1 warp-specialized layout (K4's mixed_trellis_ws shape): 8 consumer warps
# (two warpgroups) own N32 each of a gate|up N256 tile; one producer warpgroup
# streams the tile's K128 blocks through an mbarrier ring.
_FC1_CONSUMER_THREADS = 256
_FC1_PRODUCER_THREADS = 128
_FC1_PRODUCER_REGS = 88
_FC1_CONSUMER_REGS = 208
_FC1_CTA_MAX_REGS = 3 * 168
# Full FP32 accumulators trade producer registers for two larger consumers.
_FC1_FULL_GRAND_PRODUCER_REGS = 40
_FC1_FULL_GRAND_CONSUMER_REGS = 232
assert (_FC1_FULL_GRAND_PRODUCER_REGS + 2 * _FC1_FULL_GRAND_CONSUMER_REGS
        == _FC1_CTA_MAX_REGS), "FC1 full-grand register split must equal CTAPOOL (504)"
assert _FC1_PRODUCER_REGS + 2 * _FC1_CONSUMER_REGS == _FC1_CTA_MAX_REGS, (
    f"FC1 warpgroup register split {_FC1_PRODUCER_REGS} + 2*{_FC1_CONSUMER_REGS} "
    f"must equal CTAPOOL {_FC1_CTA_MAX_REGS}"
)
_FC1_TILE_STAGES = 4
_FC1_HEADER_WORDS = 8
from .exlr8 import Exlr8A16Kernel, _e4m3_round_half2, _side_value, signed_byte, store_bf16x4


@dsl_user_op
def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f32x2_to_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_shared_u32_offset(base_addr: Int32, offset: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(), [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.b32 $0, [$1 + {int(offset)}];", "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_ring_u32_offset(base_addr: Int32, offset: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(), [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.b32 $0, [$1 + {int(offset)}];", "=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_global_v2_f32(addr: Int64, *, loc=None, ip=None):
    """Coherent (non-.nc) 8-byte load: re-reads this thread's own earlier
    st.global of an FC1 split partial."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];", "=f,=f,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def ballot_sync_u32(pred: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(), [Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
        "=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ldmatrix_m8n8x4_b16_offset(base_addr: Int32, offset: int, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0, $1, $2, $3}}, [$4 + {int(offset)}];",
        "=r,=r,=r,=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)))


@dsl_user_op
def cp_async4_shared_global_ca_pred(smem_addr: Int32, gmem_addr: Int64, pred: Int32,
                                    *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p cp.async.ca.shared.global [$0], [$1], 16; }",
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async4_shared_global_cg_pred(smem_addr: Int32, gmem_addr: Int64, pred: Int32,
                                    *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p cp.async.cg.shared.global [$0], [$1], 16; }",
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def st_global_v2_f32(addr: Int64, v0: Float32, v1: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip),
         Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip)],
        "st.global.v2.f32 [$0], {$1, $2};", "l,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bind_loop_i32(val: Int32, dep: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(val).ir_value(loc=loc, ip=ip), Int32(dep).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def shfl_sync_idx_i64(val: Int64, src_lane: Int32, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(
        T.i64(),
        [Int64(val).ir_value(loc=loc, ip=ip), Int32(src_lane).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 lo, hi; mov.b64 {lo, hi}, $1; shfl.sync.idx.b32 lo, lo, $2, 31, -1; shfl.sync.idx.b32 hi, hi, $2, 31, -1; mov.b64 $0, {lo, hi}; }",
        "=l,l,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_global_lu_v2_u32_offset(base_addr: Int64, offset: int, *, loc=None, ip=None):
    v = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(base_addr).ir_value(loc=loc, ip=ip)],
        f"ld.global.lu.v2.u32 {{$0, $1}}, [$2 + {int(offset)}];",
        "=r,=r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), v, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), v, [1], loc=loc, ip=ip)))


@dsl_user_op
def ld_global_lu_v4_f32_offset(base_addr: Int64, offset: int, *, loc=None, ip=None):
    v = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_addr).ir_value(loc=loc, ip=ip)],
        f"ld.global.lu.v4.f32 {{$0, $1, $2, $3}}, [$4 + {int(offset)}];",
        "=f,=f,=f,=f,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), v, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [1], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [2], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [3], loc=loc, ip=ip)))


@dsl_user_op
def ld_shared_v4_f32(smem_addr: Int32, *, loc=None, ip=None):
    v = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), v, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [1], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [2], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), v, [3], loc=loc, ip=ip)))


@dsl_user_op
def st_shared_v4_f32(smem_addr: Int32, v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                     *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip),
         Float32(v2).ir_value(loc=loc, ip=ip),
         Float32(v3).ir_value(loc=loc, ip=ip)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def ld_ring_f32_offset(base_addr: Int32, offset: int, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.f32 $0, [$1 + {int(offset)}];", "=f,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_shared_f32(smem_addr: Int32, val: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Float32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;", "r,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def st_shared_v2_u32(smem_addr: Int32, v0: Uint32, v1: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Uint32(v0).ir_value(loc=loc, ip=ip),
         Uint32(v1).ir_value(loc=loc, ip=ip)],
        "st.shared.v2.b32 [$0], {$1, $2};", "r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def st_global_f32(addr: Int64, val: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip),
         Float32(val).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;", "l,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async4_shared_global_cg_zfill(smem_addr: Int32, gmem_addr: Int64, valid_copy: Int32,
                                     active_pred: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(valid_copy).ir_value(loc=loc, ip=ip),
         Int32(active_pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p, q; .reg .b32 sz; setp.ne.b32 p, $3, 0; setp.ne.b32 q, $2, 0; selp.b32 sz, 16, 0, q; @p cp.async.cg.shared.global [$0], [$1], 16, sz; }",
        "r,l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def ld_global_nc_v4_u32_pred(addr: Int64, pred: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; @p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)))


@dsl_user_op
def _e4m3x4_from_half2x2(h01: Uint32, h23: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(h01).ir_value(loc=loc, ip=ip), Uint32(h23).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 e01, e23; cvt.rn.satfinite.e4m3x2.f16x2 e01, $1; cvt.rn.satfinite.e4m3x2.f16x2 e23, $2; mov.b32 $0, {e01, e23}; }",
        "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f16x8_absmax_f32(x0: Uint32, x1: Uint32, x2: Uint32, x3: Uint32,
                      *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Uint32(x0).ir_value(loc=loc, ip=ip), Uint32(x1).ir_value(loc=loc, ip=ip),
         Uint32(x2).ir_value(loc=loc, ip=ip), Uint32(x3).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 a0, a1, a2, a3, m01, m23, m;
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            and.b32 a0, $1, 0x7fff7fff;
            and.b32 a1, $2, 0x7fff7fff;
            and.b32 a2, $3, 0x7fff7fff;
            and.b32 a3, $4, 0x7fff7fff;
            max.f16x2 m01, a0, a1;
            max.f16x2 m23, a2, a3;
            max.f16x2 m, m01, m23;
            mov.b32 {h0, h1}, m;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            max.f32 $0, f0, f1;
        }
        """,
        "=f,r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f16x4_absmax_f32(x0: Uint32, x1: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Uint32(x0).ir_value(loc=loc, ip=ip), Uint32(x1).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 a0, a1, m;
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            and.b32 a0, $1, 0x7fff7fff;
            and.b32 a1, $2, 0x7fff7fff;
            max.f16x2 m, a0, a1;
            mov.b32 {h0, h1}, m;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            max.f32 $0, f0, f1;
        }
        """,
        "=f,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _a8_scale_and_inv(amax: Float32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(amax).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred pz;
            .reg .f32 s, q;
            mul.rn.f32 s, $2, 0f3b124925;
            setp.le.f32 pz, $2, 0f00000000;
            div.rn.f32 q, 0f43e00000, $2;
            mov.f32 $0, s;
            selp.f32 $1, 0f00000000, q, pz;
        }
        """,
        "=f,=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _quantize_f16x8_to_e4m3x8(
    x0: Uint32, x1: Uint32, x2: Uint32, x3: Uint32, inv_s: Float32, *, loc=None, ip=None
):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(x0).ir_value(loc=loc, ip=ip), Uint32(x1).ir_value(loc=loc, ip=ip),
         Uint32(x2).ir_value(loc=loc, ip=ip), Uint32(x3).ir_value(loc=loc, ip=ip),
         Float32(inv_s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            .reg .b16 e01, e23, e45, e67;
            mov.b32 {h0, h1}, $2;
            mov.b32 {h2, h3}, $3;
            mov.b32 {h4, h5}, $4;
            mov.b32 {h6, h7}, $5;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.f32.f16 f4, h4;
            cvt.f32.f16 f5, h5;
            cvt.f32.f16 f6, h6;
            cvt.f32.f16 f7, h7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.satfinite.e4m3x2.f32 e01, f1, f0;
            cvt.rn.satfinite.e4m3x2.f32 e23, f3, f2;
            cvt.rn.satfinite.e4m3x2.f32 e45, f5, f4;
            cvt.rn.satfinite.e4m3x2.f32 e67, f7, f6;
            mov.b32 $0, {e01, e23};
            mov.b32 $1, {e45, e67};
        }
        """,
        "=r,=r,r,r,r,r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _quantize_f16x4_to_e4m3x4(
    x0: Uint32, x1: Uint32, inv_s: Float32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(x0).ir_value(loc=loc, ip=ip), Uint32(x1).ir_value(loc=loc, ip=ip),
         Float32(inv_s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1, h2, h3;
            .reg .f32 f0, f1, f2, f3;
            .reg .b16 e01, e23;
            mov.b32 {h0, h1}, $1;
            mov.b32 {h2, h3}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            mul.rn.f32 f0, f0, $3;
            mul.rn.f32 f1, f1, $3;
            mul.rn.f32 f2, f2, $3;
            mul.rn.f32 f3, f3, $3;
            cvt.rn.satfinite.e4m3x2.f32 e01, f1, f0;
            cvt.rn.satfinite.e4m3x2.f32 e23, f3, f2;
            mov.b32 $0, {e01, e23};
        }
        """,
        "=r,r,r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _prmt_5410_7632(lo: Uint32, hi: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 o0, o1; prmt.b32 o0, $2, $3, 0x5410; prmt.b32 o1, $2, $3, 0x7632; mov.b32 $0, o0; mov.b32 $1, o1; }",
        "=r,=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


def slice_blocks_for_shift(slice_shift: int, blocks: int = 48) -> int:
    """Resolve the FC2 output-block slice width for a runtime ``slice_shift``."""
    if blocks != 48:
        return int(blocks)
    if slice_shift == 0:
        return 2
    if slice_shift == 4:
        return 16
    if slice_shift in (1, 2, 3):
        return 3 << int(slice_shift)
    raise ValueError(f"unsupported EXLR8 FC2 slice_shift={slice_shift}")


def resolve_fc2_slice_geometry(
    rows: int,
    cta_limit: int,
    unit_limit: int,
    l2_bytes: int,
    *,
    top_k: int = 8,
    fc2_half: bool = True,
    blocks: int = 48,
    group_rows: int = 64,
    ctas_per_sm: int = 2,
) -> tuple[int, int]:
    """Host mirror of ``Exlr8GroupedKernel.serving``'s runtime L2 slice selector.

    Returns ``(slice_shift, slice_blocks)`` using the exact integer budget math
    emitted into the serving wrapper.
    """
    rows = int(rows)
    cta_limit = int(cta_limit)
    unit_limit = int(unit_limit)
    l2_bytes = int(l2_bytes)
    top_k = int(top_k)
    blocks = int(blocks)
    if blocks != 48:
        return 0, blocks
    fc2_cta_limit = max(1, cta_limit // int(ctas_per_sm))
    elem_bytes = 2 if fc2_half else 4
    block_fc2_bytes = rows * (top_k * 128 * elem_bytes)
    inter_bytes = (rows * top_k + 64 * int(group_rows)) * unit_limit * 256
    wave_inter_bytes = (fc2_cta_limit >> 1) * unit_limit * (int(group_rows) * 256)
    l2_budget = (l2_bytes >> 3) * 7
    full_slice_budget = l2_budget - inter_bytes
    wave_slice_budget = min((l2_bytes >> 2) * 3, l2_budget - wave_inter_bytes)
    slice_budget = (
        full_slice_budget
        if (inter_bytes < (l2_bytes >> 4) * 9 or block_fc2_bytes * 12 <= full_slice_budget)
        else wave_slice_budget
    )
    low_occ = (rows * top_k) <= ((cta_limit + 4) * 6)
    if low_occ:
        slice_shift = 1
    elif block_fc2_bytes * 24 <= slice_budget:
        slice_shift = 3
    elif block_fc2_bytes * 12 <= slice_budget:
        slice_shift = 2
    elif block_fc2_bytes * 6 <= slice_budget:
        slice_shift = 1
    else:
        slice_shift = 0
    # Minimum slices cannot retain the live intermediate; avoid launch churn.
    if slice_shift == 0 and (block_fc2_bytes * 2 > slice_budget or inter_bytes >= l2_bytes * 2):
        return 0, blocks
    return slice_shift, slice_blocks_for_shift(slice_shift, blocks=blocks)


def fc2_ws_slice_task_mapping(
    groups: int,
    step: int,
    slice_blocks: int,
    slice_shift: int,
    *,
    blocks: int = 48,
    slice_base: int = 0,
) -> tuple[int, bool, list[tuple[int, int, int]]]:
    """Mirror FC2 WS tasks as (group, b_local_base, b_pairs), including bulk/tail waves."""
    if step <= 0:
        raise ValueError(f"grouped_fc2_ws_kernel step must be positive, got {step}")
    if slice_blocks <= 0 or (slice_blocks % 2) != 0 or (blocks % slice_blocks) != 0:
        raise ValueError(
            f"grouped_fc2_ws_kernel slice_blocks={slice_blocks} must be positive, even, and divide blocks={blocks}"
        )
    b_chunk_const = 8 if (blocks % 8 == 0) else (2 if (blocks % 2 == 0) else blocks)
    chunks_per_group = blocks // b_chunk_const
    sliced = slice_blocks < blocks
    two_chunks = (slice_shift == 2) or (slice_shift == 4)
    slice_chunks = 1 if slice_shift == 0 else (2 if two_chunks else 3)
    pair_slices = slice_blocks >> 1
    tail_waves = 2 if slice_blocks > 24 else (1 if slice_blocks == 24 else 0)
    min_for_bulk = (tail_waves + 1) * step
    avail_bulk = groups - tail_waves * step if groups >= min_for_bulk else 0
    bulk_groups = (avail_bulk // step) * step
    tail_groups = groups - bulk_groups
    tail_group_tasks = tail_groups * (slice_chunks if sliced else chunks_per_group)
    use_chunks = tail_group_tasks >= step
    tail_limit_tasks = tail_group_tasks if use_chunks else (tail_groups * pair_slices)
    limit_tasks = bulk_groups + tail_limit_tasks
    b_pairs_sliced = (
        1 if slice_shift == 0
        else ((slice_blocks >> 2) if two_chunks else (1 << (slice_shift - 1)))
    )
    tail_b_pairs = (b_pairs_sliced if sliced else (b_chunk_const // 2)) if use_chunks else 1
    rev_groups = sliced and ((slice_base // slice_blocks) & 1) != 0
    tasks: list[tuple[int, int, int]] = []
    for cta in range(step):
        task = cta
        while task < limit_tasks:
            if task < bulk_groups:
                group, b_local_base, b_pairs = task, 0, pair_slices
            else:
                t_task = task - bulk_groups
                b_pairs = tail_b_pairs
                if blocks == 48:
                    if sliced:
                        if slice_shift == 0:
                            group = t_task
                            b_local_base = 0
                        elif two_chunks:
                            if use_chunks:
                                group = t_task >> 1
                                b_local_base = (t_task & 1) * (b_pairs_sliced << 1)
                            else:
                                group = (t_task >> 3) if slice_shift == 4 else ((t_task >> 1) // 3)
                                b_local_base = (t_task - group * pair_slices) << 1
                        else:
                            q = t_task if use_chunks else (t_task >> (slice_shift - 1))
                            group = q // 3
                            b_local_base = ((q - group * 3) << slice_shift) if use_chunks else ((t_task - group * pair_slices) << 1)
                    else:
                        group = (t_task // chunks_per_group) if use_chunks else (t_task // (blocks // 2))
                        b_local_base = ((t_task % chunks_per_group) * b_chunk_const) if use_chunks else ((t_task % (blocks // 2)) * 2)
                else:
                    group = (t_task // chunks_per_group) if use_chunks else (t_task // pair_slices)
                    b_local_base = ((t_task % chunks_per_group) * b_chunk_const) if use_chunks else ((t_task - group * pair_slices) << 1)
                group += bulk_groups
            if rev_groups:
                group = groups - 1 - group
            tasks.append((group, b_local_base, b_pairs))
            task += step
    return limit_tasks, use_chunks, tasks


def fc2_slice_task_mapping(
    groups: int,
    step: int,
    slice_blocks: int,
    slice_shift: int,
    *,
    blocks: int = 48,
) -> tuple[int, int, bool, list[tuple[int, int]]]:
    """Host mirror of ``grouped_fc2_kernel``'s task-to-(group, b_local_base) loop."""
    if step <= 0:
        raise ValueError(f"grouped_fc2_kernel step must be positive, got {step}")
    if slice_blocks <= 0 or (blocks % slice_blocks) != 0:
        raise ValueError(
            f"grouped_fc2_kernel slice_blocks={slice_blocks} must be positive and divide blocks={blocks}"
        )
    b_chunk_const = 8 if (blocks % 8 == 0) else (3 if (blocks % 3 == 0) else blocks)
    chunks_per_group = blocks // b_chunk_const
    sliced = slice_blocks < blocks
    two_chunks = (slice_shift == 2) or (slice_shift == 4)
    slice_chunks = 1 if slice_shift == 0 else (2 if two_chunks else 3)
    group_tasks = groups * (slice_chunks if sliced else chunks_per_group)
    use_chunks = group_tasks >= step
    limit_tasks = group_tasks if use_chunks else (groups * slice_blocks)
    b_count_sliced = 2 if slice_shift == 0 else ((slice_blocks >> 1) if two_chunks else (1 << slice_shift))
    b_count = (b_count_sliced if sliced else b_chunk_const) if use_chunks else 1
    tasks: list[tuple[int, int]] = []
    for cta in range(step):
        task = cta
        while task < limit_tasks:
            if blocks == 48:
                if sliced:
                    if slice_shift == 0:
                        group = task if use_chunks else (task >> 1)
                        b_local_base = 0 if use_chunks else (task & 1)
                    elif two_chunks:
                        if use_chunks:
                            group = task >> 1
                            b_local_base = (task & 1) * b_count_sliced
                        else:
                            group = (task >> 4) if slice_shift == 4 else ((task >> 2) // 3)
                            b_local_base = task - group * slice_blocks
                    else:
                        q = task if use_chunks else (task >> slice_shift)
                        group = q // 3
                        b_local_base = ((q - group * 3) << slice_shift) if use_chunks else (task - group * slice_blocks)
                else:
                    group = (task // chunks_per_group) if use_chunks else (task // blocks)
                    b_local_base = ((task % chunks_per_group) * b_chunk_const) if use_chunks else (task % blocks)
            else:
                group = (task // chunks_per_group) if use_chunks else (task // blocks)
                b_local_base = ((task % chunks_per_group) * b_chunk_const) if use_chunks else (task % blocks)
            tasks.append((group, b_local_base))
            task += step
    return limit_tasks, b_count, use_chunks, tasks


def reduce_slice_task_mapping(
    rows: int,
    grid: int,
    slice_blocks: int,
    slice_shift: int,
    *,
    blocks: int = 48,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Host mirror of ``reduce_kernel``'s warp-to-(row, b_out) loop."""
    if grid <= 0:
        raise ValueError(f"reduce_kernel grid must be positive, got {grid}")
    if slice_blocks <= 0 or (blocks % slice_blocks) != 0:
        raise ValueError(
            f"reduce_kernel slice_blocks={slice_blocks} must be positive and divide blocks={blocks}"
        )
    step = grid * 8
    limit_tasks = rows * slice_blocks
    sliced = slice_blocks < blocks
    tasks: list[tuple[int, int]] = []
    for cta in range(grid):
        for warp in range(8):
            task = cta * 8 + warp
            while task < limit_tasks:
                if blocks == 48:
                    if sliced:
                        if slice_shift == 0:
                            row = task >> 1
                            b_out = task & 1
                        elif slice_shift == 4:
                            row = task >> 4
                            b_out = task & 15
                        else:
                            row = (task >> slice_shift) // 3
                            b_out = task - row * slice_blocks
                    else:
                        row = task // 48
                        b_out = task % 48
                else:
                    row = task // blocks
                    b_out = task % blocks
                tasks.append((row, b_out))
                task += step
    return limit_tasks, step, tasks


def validate_fc2_slice_geometry(
    *,
    rows: int,
    cta_limit: int,
    slice_shift: int,
    slice_blocks: int,
    blocks: int = 48,
    max_groups: int = 512,
    ctas_per_sm: int = 2,
    groups_sample: tuple[int, ...] = (1, 16, 189, 376),
) -> None:
    """Assert that a resolved ``(slice_shift, slice_blocks)`` geometry is well-formed.

    Verifies that the outer ``serving`` slice loop, ``grouped_fc2_ws_kernel``,
    ``grouped_fc2_kernel``, and ``reduce_kernel`` all have positive steps,
    matching grid bounds, and bijective coverage of their task domains.
    """
    if _FC1_PRODUCER_REGS + 2 * _FC1_CONSUMER_REGS != _FC1_CTA_MAX_REGS:
        raise ValueError(
            f"FC1 warpgroup register split {_FC1_PRODUCER_REGS} + 2*{_FC1_CONSUMER_REGS} "
            f"!= {_FC1_CTA_MAX_REGS}"
        )
    if rows <= 0 or rows > 4096:
        raise ValueError(f"rows={rows} outside (0, 4096]")
    if cta_limit <= 0:
        raise ValueError(f"cta_limit={cta_limit} must be positive")
    if blocks <= 0 or (blocks % 2) != 0:
        raise ValueError(f"blocks={blocks} must be a positive even integer")
    if slice_blocks <= 0 or (slice_blocks % 2) != 0 or (blocks % slice_blocks) != 0:
        raise ValueError(
            f"slice_blocks={slice_blocks} must be positive, even, and divide blocks={blocks}"
        )
    if slice_blocks < blocks:
        expected_blocks = slice_blocks_for_shift(slice_shift, blocks=blocks)
        if slice_blocks != expected_blocks:
            raise ValueError(
                f"slice_blocks={slice_blocks} does not match slice_shift={slice_shift} "
                f"(expected {expected_blocks})"
            )

    # Outer serving slice loop: must partition [0, blocks) cleanly.
    slice_bases: list[int] = []
    slice_base = 0
    while slice_base < blocks:
        slice_bases.append(slice_base)
        slice_base += slice_blocks
    if slice_base != blocks or len(slice_bases) != blocks // slice_blocks:
        raise ValueError(
            f"serving slice loop bases={slice_bases} did not partition blocks={blocks}"
        )

    fc2_cta_limit = max(1, cta_limit // ctas_per_sm) if blocks == 48 else cta_limit
    fc2_grid = min(max_groups * blocks, fc2_cta_limit)
    if fc2_grid <= 0 or fc2_grid > max_groups * blocks:
        raise ValueError(f"invalid fc2_grid={fc2_grid} for cta_limit={cta_limit}")

    for groups in groups_sample:
        if groups < 0 or groups > max_groups:
            continue
        expected_fc2 = {(g, b) for g in range(groups) for b in range(slice_blocks)}

        limit_ws, _, ws_tasks = fc2_ws_slice_task_mapping(
            groups, fc2_grid, slice_blocks, slice_shift, blocks=blocks
        )
        if len(ws_tasks) != limit_ws:
            raise ValueError(
                f"grouped_fc2_ws_kernel visited {len(ws_tasks)} tasks != limit_tasks={limit_ws}"
            )
        covered_ws = [
            (g, b_local_base + 2 * pair_idx + j)
            for g, b_local_base, b_pairs in ws_tasks
            for pair_idx in range(b_pairs)
            for j in range(2)
        ]
        if len(covered_ws) != groups * slice_blocks or set(covered_ws) != expected_fc2:
            raise ValueError(
                f"grouped_fc2_ws_kernel coverage mismatch for groups={groups}, "
                f"slice_shift={slice_shift}, slice_blocks={slice_blocks}"
            )

        limit_f, b_count, _, f_tasks = fc2_slice_task_mapping(
            groups, fc2_grid, slice_blocks, slice_shift, blocks=blocks
        )
        if len(f_tasks) != limit_f:
            raise ValueError(
                f"grouped_fc2_kernel visited {len(f_tasks)} tasks != limit_tasks={limit_f}"
            )
        covered_f = [
            (g, b_local_base + b_idx)
            for g, b_local_base in f_tasks
            for b_idx in range(b_count)
        ]
        if len(covered_f) != groups * slice_blocks or set(covered_f) != expected_fc2:
            raise ValueError(
                f"grouped_fc2_kernel coverage mismatch for groups={groups}, "
                f"slice_shift={slice_shift}, slice_blocks={slice_blocks}"
                )

    reduce_cap = (cta_limit * 2) if blocks == 48 else cta_limit
    reduce_grid = min((rows * slice_blocks + 7) // 8, reduce_cap)
    if reduce_grid <= 0 or reduce_grid > reduce_cap:
        raise ValueError(f"invalid reduce_grid={reduce_grid} for rows={rows}, slice_blocks={slice_blocks}")
    # Verify warp loop termination and bijective coverage for a representative row count
    # (and exact rows when small enough to enumerate quickly).
    check_rows = rows if rows <= 256 else 16
    check_grid = min((check_rows * slice_blocks + 7) // 8, reduce_cap)
    limit_r, step_r, r_tasks = reduce_slice_task_mapping(
        check_rows, check_grid, slice_blocks, slice_shift, blocks=blocks
    )
    expected_r = {(r, b) for r in range(check_rows) for b in range(slice_blocks)}
    if step_r <= 0 or len(r_tasks) != limit_r or set(r_tasks) != expected_r:
        raise ValueError(
            f"reduce_kernel coverage mismatch for rows={check_rows}, "
            f"slice_shift={slice_shift}, slice_blocks={slice_blocks}"
        )


def query_optin_shared_memory() -> int | None:
    """This host's per-CTA opt-in shared memory, or None without a device.

    The CTA budget comes from the driver (cudaDevAttrMaxSharedMemoryPerBlock-
    Optin) every time a kernel is built, so no card's limit is baked into a
    layout. Offline AOT export runs with CUDA hidden from the compiler
    workers; there the caller gets None and only the universal streamed
    layout is compiled, which fits whenever FC1 does on every SM12x card.
    """
    try:
        status = cuda.cuInit(0)
        if isinstance(status, tuple):
            status = status[0]
        if status != cuda.CUresult.CUDA_SUCCESS:
            return None
        device = cuda.cuDeviceGet(0)
        if device[0] != cuda.CUresult.CUDA_SUCCESS:
            return None
        attribute = (cuda.CUdevice_attribute
                     .CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        value = cuda.cuDeviceGetAttribute(attribute, device[1])
        if value[0] != cuda.CUresult.CUDA_SUCCESS:
            return None
        limit = int(value[1])
    except Exception:
        return None
    return limit if limit > 0 else None


class Exlr8GroupedKernel(Exlr8A16Kernel):
    CODE_VERSION = 129
    A8_CODE_VERSION = 129
    FC2_WS = True
    REDUCE_MLP = True
    FC2_HALF = True
    FC2_STAGES = 3
    # Warp-specialized FC1 preserves the original per-K128 FMA chain and
    # ascending K1024 split fold; only staging and tile scheduling change.
    FC1_WS = True
    FC1_STAGES = 3
    CTAS_PER_SM = 2

    def __init__(self, hidden_size, units_per_expert, top_k, max_groups, group_rows=64,
                 activations: str = "a16", quantize_pass: bool = True, fc2_f32: bool | None = None):
        super().__init__(hidden_size, units_per_expert, top_k, max_groups, device_count=True)
        if activations not in ("a16", "a8"):
            raise ValueError("EXLR8 grouped activations must be 'a16' or 'a8'")
        self.activations = activations
        self.a8 = activations == "a8"
        self.quantize_pass = bool(quantize_pass) and self.a8
        # FC2 partial storage precision: F16 for the shared prefill instance, F32
        # when the instance must match the K1 serving kernel's numerics exactly.
        if fc2_f32 is not None:
            self.FC2_HALF = not bool(fc2_f32)
        self.group_rows = group_rows
        self.mtiles = group_rows // 16
        if group_rows not in (16, 32, 64):
            raise ValueError("EXLR8 grouped group_rows must be 16, 32 or 64")
        self.shared_memory_bytes = query_optin_shared_memory()
        required = group_rows * 256 + 8192 + max(1, self.mtiles - 1) * 8192
        if self.shared_memory_bytes is not None and required > self.shared_memory_bytes:
            raise ValueError(f"EXLR8 grouped needs {required} B shared memory, above "
                             f"the device's {self.shared_memory_bytes} B opt-in limit")
        # FC1 ring: per stage the tile's A rows (K128) and both projections'
        # compressed K128 blocks at the widest width; a tile-header ring with
        # the live-row mask and every block's scale; mbarriers.
        self.fc1_a_bytes = group_rows * 256
        self.fc1_stage_bytes = self.fc1_a_bytes + 2 * 2048 * 4
        self.fc1_hdr_words = _FC1_HEADER_WORDS + 2 * self.blocks
        self.fc1_hdr_bytes = (4 * self.fc1_hdr_words + 15) // 16 * 16
        self.fc1_hdr_off = self.FC1_STAGES * self.fc1_stage_bytes
        self.fc1_bar_off = self.fc1_hdr_off + _FC1_TILE_STAGES * self.fc1_hdr_bytes
        self.fc1_smem_bytes = self.fc1_bar_off + 8 * 2 * (self.FC1_STAGES + _FC1_TILE_STAGES)
        # The warp-specialized FC1 tile is the 64-row group (four M16 tiles).
        self.fc1_ws = bool(self.FC1_WS) and group_rows == 64
        if (self.fc1_ws and self.shared_memory_bytes is not None
                and self.fc1_smem_bytes > self.shared_memory_bytes):
            self.fc1_ws = False
        self.validate_slice_contract()

    def resolve_slice_geometry(
        self,
        rows: int,
        cta_limit: int,
        unit_limit: int,
        l2_bytes: int,
    ) -> tuple[int, int]:
        return resolve_fc2_slice_geometry(
            rows,
            cta_limit,
            unit_limit,
            l2_bytes,
            top_k=self.top_k,
            fc2_half=self.FC2_HALF,
            blocks=self.blocks,
            group_rows=self.group_rows,
            ctas_per_sm=self.CTAS_PER_SM,
        )

    def assert_valid_slice_launch(
        self,
        rows: int,
        cta_limit: int,
        unit_limit: int,
        l2_bytes: int,
    ) -> tuple[int, int]:
        slice_shift, slice_blocks = self.resolve_slice_geometry(
            rows, cta_limit, unit_limit, l2_bytes
        )
        validate_fc2_slice_geometry(
            rows=rows,
            cta_limit=cta_limit,
            slice_shift=slice_shift,
            slice_blocks=slice_blocks,
            blocks=self.blocks,
            max_groups=self.max_groups,
            ctas_per_sm=self.CTAS_PER_SM,
            groups_sample=(1, min(16, self.max_groups), min(189, self.max_groups)),
        )
        return slice_shift, slice_blocks

    def validate_slice_contract(self) -> None:
        if self.blocks == 48:
            for slice_shift in (0, 1, 2, 3, 4):
                validate_fc2_slice_geometry(
                    rows=64,
                    cta_limit=376,
                    slice_shift=slice_shift,
                    slice_blocks=slice_blocks_for_shift(slice_shift, blocks=48),
                    blocks=48,
                    max_groups=self.max_groups,
                    ctas_per_sm=self.CTAS_PER_SM,
                    groups_sample=(1, min(16, self.max_groups), min(189, self.max_groups)),
                )
        validate_fc2_slice_geometry(
            rows=64,
            cta_limit=376,
            slice_shift=0,
            slice_blocks=self.blocks,
            blocks=self.blocks,
            max_groups=self.max_groups,
            ctas_per_sm=self.CTAS_PER_SM,
            groups_sample=(1, min(16, self.max_groups)),
        )

    @property
    def __cache_key__(self):
        if self.a8:
            return ("exlr8_a8_grouped", self.A8_CODE_VERSION, self.hidden_size, self.units,
                    self.top_k, self.max_groups, self.group_rows, self.CTAS_PER_SM,
                    self.FC2_WS, self.REDUCE_MLP, self.FC2_HALF, self.FC2_STAGES,
                    self.fc1_ws, self.FC1_STAGES, self.quantize_pass)
        return ("exlr8_a16_grouped", self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.group_rows, self.CTAS_PER_SM,
                self.FC2_WS, self.REDUCE_MLP, self.FC2_HALF, self.FC2_STAGES,
                self.fc1_ws, self.FC1_STAGES)

    @cute.jit
    def quantize_a_k128_row(self, c: Int32, src_addr: Int64, valid: Int32):
        x0, x1, x2, x3 = ld_global_nc_v4_u32_pred(src_addr, valid)
        amax = _f16x8_absmax_f32(x0, x1, x2, x3)
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        s_a, inv_s = _a8_scale_and_inv(amax)
        q_lo, q_hi = _quantize_f16x8_to_e4m3x8(x0, x1, x2, x3, inv_s)
        odd = (c & Int32(1)) != Int32(0)
        send = q_lo if odd else q_hi
        keep = q_hi if odd else q_lo
        peer = cute.arch.shuffle_sync_bfly(send, offset=1)
        lo_src = peer if odd else keep
        hi_src = keep if odd else peer
        out0, out1 = _prmt_5410_7632(lo_src, hi_src)
        return out0, out1, s_a

    @cute.jit
    def quantize_stage_a_row(self, tile_smem: Int32, r_in_16: Int32, c: Int32,
                             src_addr: Int64, valid: Int32):
        out0, out1, s_a = self.quantize_a_k128_row(c, src_addr, valid)
        col8 = c >> Int32(1)
        smem_e4m3 = (
            tile_smem
            + (r_in_16 * Int32(8) + (col8 ^ (r_in_16 & Int32(7)))) * Int32(16)
            + (c & Int32(1)) * Int32(8)
        )
        st_shared_v2_u32(smem_e4m3, out0, out1)
        if (c & Int32(3)) == Int32(0):
            st_shared_f32(
                tile_smem + Int32(2048) + r_in_16 * Int32(16) + (c >> Int32(2)) * Int32(4),
                s_a)

    @cute.jit
    def stage_prequant_a_row(self, tile_smem: Int32, r_in_16: Int32, c: Int32,
                             row_src: Int64, valid: Int32):
        dst_off = (
            (r_in_16 * Int32(8) + (c ^ (r_in_16 & Int32(7)))) * Int32(16)
            if c < Int32(8) else (Int32(2048) + r_in_16 * Int32(16))
        )
        target = tile_smem + dst_off
        active_c = (c <= Int32(8)).to(Int32)
        cp_async4_shared_global_cg_zfill(target, row_src + c.to(Int64) * Int64(16), valid, active_c)

    @cute.jit
    def stage_prequant_a_row_ws(self, tile_smem: Int32, r_in_16: Int32, c: Int32,
                                row_src: Int64, valid: Int32):
        dst_off = (
            (r_in_16 * Int32(8) + (c ^ (r_in_16 & Int32(7)))) * Int32(16)
            if c < Int32(8) else (Int32(2048) + r_in_16 * Int32(16))
        )
        cp_async4_shared_global_cg_pred(
            tile_smem + dst_off, row_src + c.to(Int64) * Int64(16), valid)

    @cute.kernel
    def quantize_x_kernel(self, x: cute.Tensor, qx: cute.Tensor, rows: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        cta = Int32(cute.arch.block_idx()[0])
        grid = Int32(cute.arch.grid_dim()[0])
        c = tid & Int32(15)
        hw_odd = (tid >> Int32(4)) & Int32(1)
        warp_task_base = cta * Int32(16) + (tid >> Int32(5)) * Int32(2)
        step = grid * Int32(16)
        x_addr = x.iterator.toint()
        qx_addr = qx.iterator.toint()
        total_blocks = rows * Int32(self.blocks)
        while warp_task_base < total_blocks:
            task = warp_task_base + hw_odd
            valid = (task < total_blocks).to(Int32)
            safe_task = min(task, max(total_blocks - Int32(1), Int32(0)))
            row = safe_task // Int32(self.blocks)
            kb = safe_task % Int32(self.blocks)
            src = x_addr + (
                row.to(Int64) * Int64(self.hidden_size)
                + kb.to(Int64) * Int64(128)
                + c.to(Int64) * Int64(8)
            ) * Int64(2)
            out0, out1, s_a = self.quantize_a_k128_row(c, src, valid)
            if valid != Int32(0):
                dst = qx_addr + task.to(Int64) * Int64(256)
                st_global_v2_u32(dst + c.to(Int64) * Int64(8), out0, out1)
                if (c & Int32(3)) == Int32(0):
                    st_global_f32(dst + Int64(128) + (c >> Int32(2)).to(Int64) * Int64(4), s_a)
            warp_task_base += step

    @cute.jit
    def serving(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                group_expert: cute.Tensor, route_slots: cute.Tensor,
                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                group_count: cute.Tensor, unit_count: cute.Tensor,
                unit_map: cute.Tensor, cta_limit: Int32, unit_limit: Int32,
                l2_bytes: Int32, stream: cuda.CUstream):
        if (rows > Int32(0) and rows <= Int32(4096) and cta_limit > Int32(0)
                and unit_limit > Int32(0)):
            if cutlass.const_expr(self.quantize_pass):
                # Borrow fc2 for A8 records until FC1 finishes; same-stream FC2
                # launches overwrite it only after those readers complete.
                # cuteafd's safety gate assumes one Spark execution slot and one
                # coordinator engine stream per GPU. Re-gate for its #708
                # multi-slot pool or two decode lanes; never share this arena
                # between concurrently executing streams.
                self.quantize_x_kernel(x, fc2, rows).launch(
                    grid=(min((rows * Int32(self.blocks) + Int32(15)) // Int32(16), cta_limit), 1, 1),
                    block=(256, 1, 1), stream=stream)
            x_fc1 = fc2 if cutlass.const_expr(self.quantize_pass) else x
            if cutlass.const_expr(self.fc1_ws):
                # Persistent, one 384-thread CTA per SM: cta_limit is SMs x CTAS_PER_SM.
                self.grouped_fc1_ws_kernel(x_fc1, pointers, bits, unit_count, unit_map,
                                           group_expert, route_slots, fc1, rows,
                                           group_count, unit_limit).launch(
                    grid=(min(Int32(self.max_groups) * unit_limit,
                              max(Int32(1), cta_limit // Int32(self.CTAS_PER_SM))), 1, 1),
                    block=(_FC1_CONSUMER_THREADS + _FC1_PRODUCER_THREADS, 1, 1),
                    min_blocks_per_mp=1, stream=stream)
            else:
                self.grouped_fc1_kernel(x_fc1, pointers, bits, unit_count, unit_map,
                                        group_expert, route_slots, fc1, rows,
                                        group_count, unit_limit).launch(
                    grid=(min(Int32(self.max_groups) * unit_limit * Int32(2), cta_limit), 1, 1),
                    block=(256, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.activation_kernel(pointers, bits, unit_count, unit_map,
                                   group_expert, route_slots, fc1, intermediate,
                                   rows, group_count, unit_limit).launch(
                grid=(min(Int32(self.max_groups) * unit_limit, cta_limit * Int32(2)), 1, 1),
                block=(256, 1, 1), stream=stream)
            if cutlass.const_expr(self.blocks == 48):
                fc2_cta_limit = max(Int32(1), cta_limit // Int32(self.CTAS_PER_SM))
                elem_bytes = 2 if self.FC2_HALF else 4
                block_fc2_bytes = rows * Int32(self.top_k * 128 * elem_bytes)
                inter_bytes = (rows * Int32(self.top_k) + Int32(64 * self.group_rows)) * unit_limit * Int32(256)
                wave_inter_bytes = (fc2_cta_limit >> Int32(1)) * unit_limit * Int32(self.group_rows * 256)
                # Keep the FC2 slice partials plus live M16-tiled intermediate
                # within 7/8 (87.5%) of the runtime-queried L2 cache when
                # intermediate is under 9/16 (56.25%) of L2, and otherwise keep
                # the slice partials plus active-wave intermediate within 7/8
                # of L2 and slice partials within 3/4 (75%) of L2.
                l2_budget = (l2_bytes >> Int32(3)) * Int32(7)
                full_slice_budget = l2_budget - inter_bytes
                wave_slice_budget = min((l2_bytes >> Int32(2)) * Int32(3), l2_budget - wave_inter_bytes)
                slice_budget = (
                    full_slice_budget
                    if ((inter_bytes < (l2_bytes >> Int32(4)) * Int32(9)) | (block_fc2_bytes * Int32(12) <= full_slice_budget))
                    else wave_slice_budget
                )
                low_occ = rows * Int32(self.top_k) <= (cta_limit + Int32(4)) * Int32(6)
                slice_shift = (
                    Int32(1) if low_occ
                    else (Int32(3) if block_fc2_bytes * Int32(24) <= slice_budget
                          else (Int32(2) if block_fc2_bytes * Int32(12) <= slice_budget
                                else (Int32(1) if block_fc2_bytes * Int32(6) <= slice_budget
                                      else Int32(0))))
                )
                slice_blocks = (
                    Int32(2) if slice_shift == Int32(0)
                    else (Int32(16) if slice_shift == Int32(4)
                          else (Int32(3) << slice_shift))
                )
                slice_blocks = max(Int32(2), slice_blocks)
                if slice_shift == Int32(0) and (block_fc2_bytes * Int32(2) > slice_budget or inter_bytes >= l2_bytes * Int32(2)):
                    slice_blocks = Int32(self.blocks)
                if (slice_shift >= Int32(0) and slice_shift <= Int32(4)
                        and (Int32(48) % slice_blocks) == Int32(0)
                        and fc2_cta_limit > Int32(0)):
                    slice_base = Int32(0)
                    while slice_base < Int32(48):
                        self.launch_fc2(
                            pointers, bits, unit_count, unit_map,
                            group_expert, route_slots, intermediate, fc2,
                            rows, group_count, unit_limit, slice_base, slice_blocks, slice_shift, fc2_cta_limit, stream, True)
                        self.reduce_kernel(
                            route_group_lane, route_weights, fc2, signs_out, output,
                            rows, group_count, slice_base, slice_blocks, slice_shift
                        ).launch(
                            grid=(max(Int32(1), min((rows * slice_blocks + Int32(7)) // Int32(8), cta_limit * Int32(2))), 1, 1),
                            block=(256, 1, 1), stream=stream)
                        slice_base += slice_blocks
            else:
                self.launch_fc2(
                    pointers, bits, unit_count, unit_map,
                    group_expert, route_slots, intermediate, fc2,
                    rows, group_count, unit_limit, Int32(0), Int32(self.blocks), Int32(0), cta_limit, stream)
                self.reduce_kernel(
                    route_group_lane, route_weights, fc2, signs_out, output,
                    rows, group_count, Int32(0), Int32(self.blocks), Int32(0)
                ).launch(
                    grid=(min((rows * Int32(self.blocks) + Int32(7)) // Int32(8), cta_limit), 1, 1),
                    block=(256, 1, 1), stream=stream)

    @cute.jit
    def prepare_group_tiles(self, route_slots: cute.Tensor, group: Int32, rows: Int32,
                            tid: Int32, lane: Int32, shared_a: Int32,
                            smem_thread_off: Int32, down: cutlass.Constexpr):
        live_m = cute.make_rmem_tensor(self.mtiles, Int32)
        srow = cute.make_rmem_tensor(self.mtiles, Int32)
        live_m.fill(Int32(0))
        srow.fill(Int32(-1))
        num_live_m = Int32(0)
        max_slot = rows * Int32(self.top_k)
        group_base = group.to(Int64) * Int64(self.group_rows)
        for m in cutlass.range_constexpr(self.mtiles):
            local_check = Int32(m * 16) + (lane & Int32(15))
            slot_check = route_slots[group_base + local_check.to(Int64)]
            is_live = cute.arch.vote_ballot_sync(
                (slot_check >= Int32(0)) & (slot_check < max_slot)) != Uint32(0)
            if is_live:
                for idx in cutlass.range_constexpr(self.mtiles):
                    if num_live_m == Int32(idx):
                        live_m[idx] = Int32(m)
                num_live_m += Int32(1)
        r_s = tid >> Int32(4)
        for idx in cutlass.range_constexpr(self.mtiles):
            if Int32(idx) < num_live_m:
                m = live_m[idx]
                local_row = m * Int32(16) + r_s
                slot_s = route_slots[group_base + local_row.to(Int64)]
                if slot_s >= Int32(0) and slot_s < max_slot:
                    srow[idx] = local_row if cutlass.const_expr(down) else (slot_s // Int32(self.top_k))
                else:
                    st_shared_v4_u32(shared_a + Int32(idx * 4096) + smem_thread_off,
                                     Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if cutlass.const_expr(not down and self.mtiles == 4):
                        if num_live_m <= Int32(3) and Int32(idx) < Int32(3):
                            st_shared_v4_u32(shared_a + Int32(12288 + idx * 4096) + smem_thread_off,
                                             Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        return num_live_m, live_m, srow

    @cute.jit
    def zero_invalid_a_tiles(self, shared_a_dst: Int32, srow: cute.Tensor,
                             num_live_m: Int32, smem_thread_off: Int32):
        for idx in cutlass.range_constexpr(self.mtiles):
            if Int32(idx) < num_live_m and srow[idx] < Int32(0):
                st_shared_v4_u32(shared_a_dst + Int32(idx * 4096) + smem_thread_off,
                                 Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def stage_a_async(self, shared_a: Int32, a_address: Int64, srow: cute.Tensor,
                      num_live_m: Int32, input_block: Int32, smem_thread_off: Int32,
                      col_s: Int32, a_stride: cutlass.Constexpr,
                      max_tiles: cutlass.Constexpr):
        if cutlass.const_expr(self.a8):
            r_s = Int32(cute.arch.thread_idx()[0]) >> Int32(4)
        col_offset_bytes = (input_block.to(Int64) * Int64(128) + col_s.to(Int64) * Int64(8)) * Int64(2)
        base_a = a_address + col_offset_bytes
        for idx in cutlass.range_constexpr(max_tiles):
            if Int32(idx) < num_live_m:
                sr = srow[idx]
                valid = (sr >= Int32(0)).to(Int32)
                safe_sr = max(sr, Int32(0))
                if cutlass.const_expr(self.quantize_pass):
                    row_src = a_address + (safe_sr.to(Int64) * Int64(self.blocks) + input_block.to(Int64)) * Int64(256)
                    self.stage_prequant_a_row(shared_a + Int32(idx * 4096), r_s, col_s, row_src, valid)
                elif cutlass.const_expr(self.a8):
                    source = base_a + safe_sr.to(Int64) * Int64(a_stride * 2)
                    self.quantize_stage_a_row(shared_a + Int32(idx * 4096), r_s, col_s, source, valid)
                else:
                    source = base_a + safe_sr.to(Int64) * Int64(a_stride * 2)
                    target = shared_a + Int32(idx * 4096) + smem_thread_off
                    cp_async4_shared_global_ca_pred(target, source, valid)

    @cute.jit
    def stage_b_async_dyn(self, shared_b: Int32, b_address: Int64, tile_block: Int32,
                          tid: Int32, width: Int32):
        if width != Int32(0):
            base_b = b_address + tile_block.to(Int64) * (width.to(Int64) * Int64(2048))
            off0 = tid * Int32(16)
            cp_async4_shared_global(shared_b + off0, base_b + off0.to(Int64))
            off1 = off0 + Int32(4096)
            cp_async4_shared_global_cg_pred(
                shared_b + off1, base_b + off1.to(Int64),
                (tid < (width - Int32(2)) * Int32(128)).to(Int32))

    @cute.jit
    def get_fc2_unit_desc(self, pointers: cute.Tensor, bits: cute.Tensor,
                          unit_map: cute.Tensor, expert: Int32, k: Int32,
                          group_inter_base: Int64):
        unit = unit_map[expert.to(Int64) * Int64(self.units) + k.to(Int64)]
        desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
        width = bits[desc]
        payload = pointers[desc * Int64(3)]
        scales = pointers[desc * Int64(3) + Int64(2)]
        unit_addr = group_inter_base + k.to(Int64) * Int64(self.group_rows * 128 * 2)
        return width, payload, scales, unit_addr

    @cute.jit
    def accumulate_fixed_m(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                           lane: Int32, warp: Int32,
                           width: cutlass.Constexpr, packed_scale: Uint32,
                           m_live: cutlass.Constexpr):
        acc = cute.make_rmem_tensor((m_live, 8), Float32)
        acc.fill(Float32(0.0))
        smem_b_a = Int32(0)
        smem_b_b = Int32(0)
        shift = Int32(0)
        shift_b = Int32(0)
        if cutlass.const_expr(width == 4):
            smem_b_b = shared_b + (warp * Int32(32) + lane) * Int32(4)
        else:
            ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            smem_b_a = shared_b + (warp * Int32(8 * width) + ia) * Int32(4)
            smem_b_b = shared_b + (warp * Int32(8 * width) + ib) * Int32(4)
            shift_b = shift + Int32(4 * width)
        r_m = lane & Int32(15)
        c_half = lane >> Int32(4)
        if cutlass.const_expr(self.a8):
            sa_base = shared_a + Int32(2048) + (lane >> Int32(2)) * Int32(16)
            for k32 in cutlass.range_constexpr(4):
                e0_lo = Uint32(0)
                e1_lo = Uint32(0)
                e0_hi = Uint32(0)
                e1_hi = Uint32(0)
                for kh in cutlass.range_constexpr(2):
                    k16 = k32 * 2 + kh
                    wa = Uint32(0)
                    wb = Uint32(0)
                    if cutlass.const_expr(width == 4):
                        b = ld_shared_u32_offset(smem_b_b, k16 * 1024)
                        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                        wa = b
                        wb = (a << Uint32(16)) | (b >> Uint32(16))
                    else:
                        a = ld_shared_u32_offset(smem_b_a, k16 * 256 * width)
                        b = ld_shared_u32_offset(smem_b_b, k16 * 256 * width)
                        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                        wa = (window >> shift.to(Int64)).to(Uint32)
                        wb = (window >> shift_b.to(Int64)).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    e_lo = _e4m3x4_from_half2x2(b0, b1)
                    e_hi = _e4m3x4_from_half2x2(b2, b3)
                    if cutlass.const_expr(kh == 0):
                        e0_lo, e0_hi = e_lo, e_hi
                    else:
                        e1_lo, e1_hi = e_lo, e_hi
                col8 = Int32(k32 * 2) + c_half
                addr_k32 = shared_a + (r_m * Int32(8) + (col8 ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k32, idx * 4096)
                    sa0 = ld_ring_f32_offset(sa_base, idx * 4096 + k32 * 4)
                    sa1 = ld_ring_f32_offset(sa_base, idx * 4096 + 128 + k32 * 4)
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                        a0, a1, a2, a3, e0_lo, e1_lo)
                    acc[idx, 0] = fma_rn_f32(d0, sa0, acc[idx, 0])
                    acc[idx, 1] = fma_rn_f32(d1, sa0, acc[idx, 1])
                    acc[idx, 2] = fma_rn_f32(d2, sa1, acc[idx, 2])
                    acc[idx, 3] = fma_rn_f32(d3, sa1, acc[idx, 3])
                    d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                        a0, a1, a2, a3, e0_hi, e1_hi)
                    acc[idx, 4] = fma_rn_f32(d4, sa0, acc[idx, 4])
                    acc[idx, 5] = fma_rn_f32(d5, sa0, acc[idx, 5])
                    acc[idx, 6] = fma_rn_f32(d6, sa1, acc[idx, 6])
                    acc[idx, 7] = fma_rn_f32(d7, sa1, acc[idx, 7])
        else:
            for k16 in cutlass.range_constexpr(8):
                wa = Uint32(0)
                wb = Uint32(0)
                if cutlass.const_expr(width == 4):
                    b = ld_shared_u32_offset(smem_b_b, k16 * 1024)
                    a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                    wa = b
                    wb = (a << Uint32(16)) | (b >> Uint32(16))
                else:
                    a = ld_shared_u32_offset(smem_b_a, k16 * 256 * width)
                    b = ld_shared_u32_offset(smem_b_b, k16 * 256 * width)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift.to(Int64)).to(Uint32)
                    wb = (window >> shift_b.to(Int64)).to(Uint32)
                b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                col = Int32(k16 * 2) + c_half
                addr_k16 = shared_a + (r_m * Int32(16) + (col ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k16, idx * 4096)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[idx, 0], acc[idx, 1], acc[idx, 2], acc[idx, 3],
                        a0, a1, a2, a3, b0, b1)
                    acc[idx, 0], acc[idx, 1], acc[idx, 2], acc[idx, 3] = d0, d1, d2, d3
                    d4, d5, d6, d7 = f16_mma_m16n8k16_f32(
                        acc[idx, 4], acc[idx, 5], acc[idx, 6], acc[idx, 7],
                        a0, a1, a2, a3, b2, b3)
                    acc[idx, 4], acc[idx, 5], acc[idx, 6], acc[idx, 7] = d4, d5, d6, d7
        scale, _ = half2_to_float2_scaled(packed_scale, Float32(1.0))
        for idx in cutlass.range_constexpr(m_live):
            for v in cutlass.range_constexpr(8):
                total[idx, v] = fma_rn_f32(acc[idx, v], scale, total[idx, v])

    @cute.jit
    def accumulate(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                   lane: Int32, warp: Int32,
                   width: cutlass.Constexpr, packed_scale: Uint32, num_live_m: Int32):
        if cutlass.const_expr(self.mtiles == 1):
            if num_live_m >= Int32(1):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 1)
        elif cutlass.const_expr(self.mtiles == 2):
            if num_live_m == Int32(1):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 1)
            elif num_live_m >= Int32(2):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 2)
        else:
            if num_live_m == Int32(1):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 1)
            elif num_live_m == Int32(2):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 2)
            elif num_live_m == Int32(3):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 3)
            elif num_live_m >= Int32(4):
                self.accumulate_fixed_m(total, shared_a, shared_b, lane, warp,
                                        width, packed_scale, 4)

    @cute.jit
    def zero_invalid_fc2_a_tiles(self, shared_a_dst: Int32, out_meta: Int32,
                                 num_live_m: Int32, smem_thread_off: Int32):
        for idx in cutlass.range_constexpr(self.mtiles):
            if Int32(idx) < num_live_m and (out_meta & Int32(1 << (16 + idx))) == Int32(0):
                st_shared_v4_u32(shared_a_dst + Int32(idx * 4096) + smem_thread_off,
                                 Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def stage_a_fc2_async(self, shared_a: Int32, a_address: Int64, out_meta: Int32,
                          num_live_m: Int32, r_s: Int32, smem_thread_off: Int32,
                          col_s: Int32):
        base_a = a_address + (r_s.to(Int64) * Int64(128) + col_s.to(Int64) * Int64(8)) * Int64(2)
        for idx in cutlass.range_constexpr(self.mtiles):
            if Int32(idx) < num_live_m:
                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                valid = ((out_meta >> Int32(16 + idx)) & Int32(1)).to(Int32)
                source = base_a + m.to(Int64) * Int64(16 * 128 * 2)
                if cutlass.const_expr(self.quantize_pass):
                    self.stage_prequant_a_row(
                        shared_a + Int32(idx * 4096), r_s, col_s,
                        a_address + (m * Int32(16) + r_s).to(Int64) * Int64(256), valid)
                elif cutlass.const_expr(self.a8):
                    self.quantize_stage_a_row(shared_a + Int32(idx * 4096), r_s, col_s, source, valid)
                else:
                    target = shared_a + Int32(idx * 4096) + smem_thread_off
                    cp_async4_shared_global_cg_pred(target, source, valid)

    @cute.jit
    def accumulate_fc2_fixed_m(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                               lane: Int32, warp: Int32,
                               width: cutlass.Constexpr, packed_scale: Uint32,
                               m_live: cutlass.Constexpr):
        acc = cute.make_rmem_tensor((m_live, 8), Float32)
        acc.fill(Float32(0.0))
        smem_b_a = Int32(0)
        smem_b_b = Int32(0)
        shift = Int32(0)
        shift_b = Int32(0)
        if cutlass.const_expr(width == 4):
            smem_b_b = shared_b + (warp * Int32(32) + lane) * Int32(4)
        else:
            ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            smem_b_a = shared_b + (warp * Int32(8 * width) + ia) * Int32(4)
            smem_b_b = shared_b + (warp * Int32(8 * width) + ib) * Int32(4)
            shift_b = shift + Int32(4 * width)
        r_m = lane & Int32(15)
        c_half = lane >> Int32(4)
        if cutlass.const_expr(self.a8):
            sa_base = shared_a + Int32(2048) + (lane >> Int32(2)) * Int32(16)
            for k32 in cutlass.range_constexpr(4):
                e0_lo = Uint32(0)
                e1_lo = Uint32(0)
                e0_hi = Uint32(0)
                e1_hi = Uint32(0)
                for kh in cutlass.range_constexpr(2):
                    k16 = k32 * 2 + kh
                    wa = Uint32(0)
                    wb = Uint32(0)
                    if cutlass.const_expr(width == 4):
                        b = ld_ring_u32_offset(smem_b_b, k16 * 1024)
                        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                        wa = b
                        wb = (a << Uint32(16)) | (b >> Uint32(16))
                    else:
                        a = ld_ring_u32_offset(smem_b_a, k16 * 256 * width)
                        b = ld_ring_u32_offset(smem_b_b, k16 * 256 * width)
                        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                        wa = (window >> shift.to(Int64)).to(Uint32)
                        wb = (window >> shift_b.to(Int64)).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    e_lo = _e4m3x4_from_half2x2(b0, b1)
                    e_hi = _e4m3x4_from_half2x2(b2, b3)
                    if cutlass.const_expr(kh == 0):
                        e0_lo, e0_hi = e_lo, e_hi
                    else:
                        e1_lo, e1_hi = e_lo, e_hi
                col8 = Int32(k32 * 2) + c_half
                addr_k32 = shared_a + (r_m * Int32(8) + (col8 ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k32, idx * 4096)
                    sa0 = ld_ring_f32_offset(sa_base, idx * 4096 + k32 * 4)
                    sa1 = ld_ring_f32_offset(sa_base, idx * 4096 + 128 + k32 * 4)
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                        a0, a1, a2, a3, e0_lo, e1_lo)
                    acc[idx, 0] = fma_rn_f32(d0, sa0, acc[idx, 0])
                    acc[idx, 1] = fma_rn_f32(d1, sa0, acc[idx, 1])
                    acc[idx, 2] = fma_rn_f32(d2, sa1, acc[idx, 2])
                    acc[idx, 3] = fma_rn_f32(d3, sa1, acc[idx, 3])
                    d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                        a0, a1, a2, a3, e0_hi, e1_hi)
                    acc[idx, 4] = fma_rn_f32(d4, sa0, acc[idx, 4])
                    acc[idx, 5] = fma_rn_f32(d5, sa0, acc[idx, 5])
                    acc[idx, 6] = fma_rn_f32(d6, sa1, acc[idx, 6])
                    acc[idx, 7] = fma_rn_f32(d7, sa1, acc[idx, 7])
        else:
            for k16 in cutlass.range_constexpr(8):
                wa = Uint32(0)
                wb = Uint32(0)
                if cutlass.const_expr(width == 4):
                    b = ld_ring_u32_offset(smem_b_b, k16 * 1024)
                    a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                    wa = b
                    wb = (a << Uint32(16)) | (b >> Uint32(16))
                else:
                    a = ld_ring_u32_offset(smem_b_a, k16 * 256 * width)
                    b = ld_ring_u32_offset(smem_b_b, k16 * 256 * width)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift.to(Int64)).to(Uint32)
                    wb = (window >> shift_b.to(Int64)).to(Uint32)
                b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                col = Int32((k16 & 3) * 2) + c_half
                addr_k16 = shared_a + (r_m * Int32(16) + (col ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k16, idx * 4096 + (k16 >> 2) * 128)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[idx, 0], acc[idx, 1], acc[idx, 2], acc[idx, 3],
                        a0, a1, a2, a3, b0, b1)
                    acc[idx, 0], acc[idx, 1], acc[idx, 2], acc[idx, 3] = d0, d1, d2, d3
                    d4, d5, d6, d7 = f16_mma_m16n8k16_f32(
                        acc[idx, 4], acc[idx, 5], acc[idx, 6], acc[idx, 7],
                        a0, a1, a2, a3, b2, b3)
                    acc[idx, 4], acc[idx, 5], acc[idx, 6], acc[idx, 7] = d4, d5, d6, d7
        scale, _ = half2_to_float2_scaled(packed_scale, Float32(1.0))
        for idx in cutlass.range_constexpr(m_live):
            for v in cutlass.range_constexpr(8):
                total[idx, v] = fma_rn_f32(acc[idx, v], scale, total[idx, v])

    @cute.jit
    def accumulate_fc2(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                       lane: Int32, warp: Int32,
                       width: cutlass.Constexpr, packed_scale: Uint32, num_live_m: Int32):
        if cutlass.const_expr(self.mtiles == 1):
            if num_live_m >= Int32(1):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 1)
        elif cutlass.const_expr(self.mtiles == 2):
            if num_live_m == Int32(1):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 1)
            elif num_live_m >= Int32(2):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 2)
        else:
            if num_live_m == Int32(1):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 1)
            elif num_live_m == Int32(2):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 2)
            elif num_live_m == Int32(3):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 3)
            elif num_live_m >= Int32(4):
                self.accumulate_fc2_fixed_m(total, shared_a, shared_b, lane, warp,
                                            width, packed_scale, 4)

    @cute.jit
    def accumulate_fc2_pair_fixed_m(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                                    lane: Int32, warp: Int32,
                                    width: cutlass.Constexpr, packed_scale0: Uint32, packed_scale1: Uint32,
                                    m_live: cutlass.Constexpr):
        acc = cute.make_rmem_tensor((m_live, 2, 8), Float32)
        acc.fill(Float32(0.0))
        smem_b_a = Int32(0)
        smem_b_b = Int32(0)
        shift = Int32(0)
        shift_b = Int32(0)
        if cutlass.const_expr(width == 4):
            smem_b_b = shared_b + (warp * Int32(32) + lane) * Int32(4)
        else:
            ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            smem_b_a = shared_b + (warp * Int32(8 * width) + ia) * Int32(4)
            smem_b_b = shared_b + (warp * Int32(8 * width) + ib) * Int32(4)
            shift_b = shift + Int32(4 * width)
        r_m = lane & Int32(15)
        c_half = lane >> Int32(4)
        b1_byte_off = width * 2048
        if cutlass.const_expr(self.a8):
            sa_base = shared_a + Int32(2048) + (lane >> Int32(2)) * Int32(16)
            for k32 in cutlass.range_constexpr(4):
                frags = []
                for j in cutlass.range_constexpr(2):
                    j_off = j * b1_byte_off
                    e0_lo = Uint32(0)
                    e1_lo = Uint32(0)
                    e0_hi = Uint32(0)
                    e1_hi = Uint32(0)
                    for kh in cutlass.range_constexpr(2):
                        k16 = k32 * 2 + kh
                        wa = Uint32(0)
                        wb = Uint32(0)
                        if cutlass.const_expr(width == 4):
                            b = ld_ring_u32_offset(smem_b_b, j_off + k16 * 1024)
                            a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                            wa = b
                            wb = (a << Uint32(16)) | (b >> Uint32(16))
                        else:
                            a = ld_ring_u32_offset(smem_b_a, j_off + k16 * 256 * width)
                            b = ld_ring_u32_offset(smem_b_b, j_off + k16 * 256 * width)
                            window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                            wa = (window >> shift.to(Int64)).to(Uint32)
                            wb = (window >> shift_b.to(Int64)).to(Uint32)
                        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                        e_lo = _e4m3x4_from_half2x2(b0, b1)
                        e_hi = _e4m3x4_from_half2x2(b2, b3)
                        if cutlass.const_expr(kh == 0):
                            e0_lo, e0_hi = e_lo, e_hi
                        else:
                            e1_lo, e1_hi = e_lo, e_hi
                    frags.append((e0_lo, e1_lo, e0_hi, e1_hi))
                col8 = Int32(k32 * 2) + c_half
                addr_k32 = shared_a + (r_m * Int32(8) + (col8 ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k32, idx * 4096)
                    sa0 = ld_ring_f32_offset(sa_base, idx * 4096 + k32 * 4)
                    sa1 = ld_ring_f32_offset(sa_base, idx * 4096 + 128 + k32 * 4)
                    for j in cutlass.range_constexpr(2):
                        f = frags[j]
                        d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                            a0, a1, a2, a3, f[0], f[1])
                        acc[idx, j, 0] = fma_rn_f32(d0, sa0, acc[idx, j, 0])
                        acc[idx, j, 1] = fma_rn_f32(d1, sa0, acc[idx, j, 1])
                        acc[idx, j, 2] = fma_rn_f32(d2, sa1, acc[idx, j, 2])
                        acc[idx, j, 3] = fma_rn_f32(d3, sa1, acc[idx, j, 3])
                        d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                            a0, a1, a2, a3, f[2], f[3])
                        acc[idx, j, 4] = fma_rn_f32(d4, sa0, acc[idx, j, 4])
                        acc[idx, j, 5] = fma_rn_f32(d5, sa0, acc[idx, j, 5])
                        acc[idx, j, 6] = fma_rn_f32(d6, sa1, acc[idx, j, 6])
                        acc[idx, j, 7] = fma_rn_f32(d7, sa1, acc[idx, j, 7])
        else:
            for k16 in cutlass.range_constexpr(8):
                frags = []
                for j in cutlass.range_constexpr(2):
                    j_off = j * b1_byte_off
                    wa = Uint32(0)
                    wb = Uint32(0)
                    if cutlass.const_expr(width == 4):
                        b = ld_ring_u32_offset(smem_b_b, j_off + k16 * 1024)
                        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                        wa = b
                        wb = (a << Uint32(16)) | (b >> Uint32(16))
                    else:
                        a = ld_ring_u32_offset(smem_b_a, j_off + k16 * 256 * width)
                        b = ld_ring_u32_offset(smem_b_b, j_off + k16 * 256 * width)
                        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                        wa = (window >> shift.to(Int64)).to(Uint32)
                        wb = (window >> shift_b.to(Int64)).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    frags.append((_e4m3_round_half2(b0), _e4m3_round_half2(b1),
                                  _e4m3_round_half2(b2), _e4m3_round_half2(b3)))
                col = Int32((k16 & 3) * 2) + c_half
                addr_k16 = shared_a + (r_m * Int32(16) + (col ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(m_live):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k16, idx * 4096 + (k16 >> 2) * 128)
                    for j in cutlass.range_constexpr(2):
                        f = frags[j]
                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                            acc[idx, j, 0], acc[idx, j, 1], acc[idx, j, 2], acc[idx, j, 3],
                            a0, a1, a2, a3, f[0], f[1])
                        acc[idx, j, 0], acc[idx, j, 1], acc[idx, j, 2], acc[idx, j, 3] = d0, d1, d2, d3
                        d4, d5, d6, d7 = f16_mma_m16n8k16_f32(
                            acc[idx, j, 4], acc[idx, j, 5], acc[idx, j, 6], acc[idx, j, 7],
                            a0, a1, a2, a3, f[2], f[3])
                        acc[idx, j, 4], acc[idx, j, 5], acc[idx, j, 6], acc[idx, j, 7] = d4, d5, d6, d7
        scale0, _ = half2_to_float2_scaled(packed_scale0, Float32(1.0))
        scale1, _ = half2_to_float2_scaled(packed_scale1, Float32(1.0))
        for idx in cutlass.range_constexpr(m_live):
            for v in cutlass.range_constexpr(8):
                total[idx, 0, v] = fma_rn_f32(acc[idx, 0, v], scale0, total[idx, 0, v])
                total[idx, 1, v] = fma_rn_f32(acc[idx, 1, v], scale1, total[idx, 1, v])

    @cute.jit
    def accumulate_fc2_pair(self, total: cute.Tensor, shared_a: Int32, shared_b: Int32,
                            lane: Int32, warp: Int32,
                            width: cutlass.Constexpr, packed_scale0: Uint32, packed_scale1: Uint32,
                            num_live_m: Int32):
        if cutlass.const_expr(self.mtiles == 1):
            if num_live_m >= Int32(1):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 1)
        elif cutlass.const_expr(self.mtiles == 2):
            if num_live_m == Int32(1):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 1)
            elif num_live_m >= Int32(2):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 2)
        else:
            if num_live_m == Int32(1):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 1)
            elif num_live_m == Int32(2):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 2)
            elif num_live_m == Int32(3):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 3)
            elif num_live_m >= Int32(4):
                self.accumulate_fc2_pair_fixed_m(total, shared_a, shared_b, lane, warp,
                                                 width, packed_scale0, packed_scale1, 4)

    @cute.jit
    def fold_shared_plane(self, c_base: Int32, tid: Int32, split: Int32,
                          total: cute.Tensor, tile_idx: cutlass.Constexpr):
        c_lo = c_base + tid * Int32(16)
        c_hi = c_lo + Int32(4096)
        if split == Int32(0):
            st_shared_v4_f32(
                c_lo,
                Float32(0.0) + total[tile_idx, 0], Float32(0.0) + total[tile_idx, 1],
                Float32(0.0) + total[tile_idx, 2], Float32(0.0) + total[tile_idx, 3])
            st_shared_v4_f32(
                c_hi,
                Float32(0.0) + total[tile_idx, 4], Float32(0.0) + total[tile_idx, 5],
                Float32(0.0) + total[tile_idx, 6], Float32(0.0) + total[tile_idx, 7])
        else:
            c0, c1, c2, c3 = ld_shared_v4_f32(c_lo)
            c4, c5, c6, c7 = ld_shared_v4_f32(c_hi)
            st_shared_v4_f32(
                c_lo,
                c0 + total[tile_idx, 0], c1 + total[tile_idx, 1],
                c2 + total[tile_idx, 2], c3 + total[tile_idx, 3])
            st_shared_v4_f32(
                c_hi,
                c4 + total[tile_idx, 4], c5 + total[tile_idx, 5],
                c6 + total[tile_idx, 6], c7 + total[tile_idx, 7])

    @cute.jit
    def load_shared_plane(self, c_base: Int32, tid: Int32,
                          total: cute.Tensor, tile_idx: cutlass.Constexpr):
        c_lo = c_base + tid * Int32(16)
        c_hi = c_lo + Int32(4096)
        c0, c1, c2, c3 = ld_shared_v4_f32(c_lo)
        c4, c5, c6, c7 = ld_shared_v4_f32(c_hi)
        total[tile_idx, 0], total[tile_idx, 1], total[tile_idx, 2], total[tile_idx, 3] = c0, c1, c2, c3
        total[tile_idx, 4], total[tile_idx, 5], total[tile_idx, 6], total[tile_idx, 7] = c4, c5, c6, c7

    @cute.kernel
    def grouped_fc1_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                           unit_count: cute.Tensor, unit_map: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           fc1: cute.Tensor, rows: Int32, group_count: cute.Tensor,
                           unit_limit: Int32):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[
                    Uint32,
                    self.group_rows * 64 + 2048 + max(1, self.mtiles - 1) * 2048
                ], 1024]

        storage = smem.allocate(Storage)
        shared_a = shared_ptr_to_u32(storage.words.data_ptr())
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        r_s, col_s = tid >> Int32(4), tid & Int32(15)
        smem_thread_off = (r_s * Int32(16) + (col_s ^ (r_s & Int32(7)))) * Int32(16)
        num_up = unit_limit * Int32(2)
        grand0 = cute.make_rmem_tensor(8, Float32)
        grand1 = cute.make_rmem_tensor(8, Float32)
        total = cute.make_rmem_tensor((self.mtiles, 8), Float32)

        while task < groups * num_up:
            g_tile_base = (task // (num_up * Int32(4))) * Int32(4)
            tile_g = min(Int32(4), groups - g_tile_base)
            rem = task - g_tile_base * num_up
            group = g_tile_base + (rem % tile_g)
            up_idx = rem // tile_g
            proj = up_idx & Int32(1)
            k = up_idx >> Int32(1)
            expert = group_expert[group]
            if k < unit_count[expert]:
                unit = unit_map[expert.to(Int64) * Int64(self.units) + k.to(Int64)]
                logical = (group * unit_limit + k) * Int32(2) + proj
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
                width = bits[desc]
                num_live_m, live_m, srow = self.prepare_group_tiles(
                    route_slots, group, rows, tid, lane, shared_a, smem_thread_off, False)
                cute.arch.sync_threads()

                if width != Int32(0) and num_live_m > Int32(0):
                    grand0.fill(Float32(0.0))
                    if cutlass.const_expr(self.mtiles >= 2):
                        grand1.fill(Float32(0.0))
                    payload, scales = pointers[desc * Int64(3)], pointers[desc * Int64(3) + Int64(2)]
                    x_addr = x.iterator.toint()
                    fc1_addr = fc1.iterator.toint()
                    group_base = group.to(Int64) * Int64(self.group_rows)
                    max_slot = rows * Int32(self.top_k)
                    out_meta = Int32(0)
                    for idx in cutlass.range_constexpr(self.mtiles):
                        if Int32(idx) < num_live_m:
                            m = live_m[idx]
                            out_meta = out_meta | (m << Int32(idx * 2))
                            for v in cutlass.range_constexpr(2):
                                local_out = m * Int32(16) + Int32(v * 8) + (lane >> Int32(2))
                                slot_out = route_slots[group_base + local_out.to(Int64)]
                                if slot_out >= Int32(0) and slot_out < max_slot:
                                    out_meta = out_meta | Int32(1 << (8 + idx * 2 + v))

                    a_ping_pong = (num_live_m <= Int32(3)) if cutlass.const_expr(self.mtiles == 4) else False
                    a_stride_smem = Int32(12288) if a_ping_pong else Int32(0)
                    shared_b0 = (
                        shared_a + (Int32(24576) if a_ping_pong else Int32(16384))
                        if cutlass.const_expr(self.mtiles == 4)
                        else shared_a + Int32(self.group_rows * 256)
                    )
                    shared_c2 = shared_a + Int32(40960)
                    shared_c3 = shared_a + Int32(32768)

                    self.stage_a_async(shared_a, x_addr, srow, num_live_m, Int32(0),
                                       smem_thread_off, col_s, self.hidden_size, self.mtiles)
                    self.stage_b_async_dyn(shared_b0, payload, Int32(0), tid, width)
                    cute.arch.cp_async_commit_group()
                    next_scale = ld_global_b16(scales)

                    for split in cutlass.range(self.splits):
                        kb0 = split * Int32(8)
                        total.fill(Float32(0.0))
                        for block in cutlass.range(8):
                            kb = kb0 + block
                            if kb < Int32(self.blocks):
                                parity = kb & Int32(1)
                                next_parity = (kb + Int32(1)) & Int32(1)
                                cur_a = shared_a + parity * a_stride_smem
                                next_a = shared_a + next_parity * a_stride_smem
                                cur_b = shared_b0 + parity * Int32(8192)
                                next_b = shared_b0 + next_parity * Int32(8192)
                                has_next = (kb + Int32(1)) < Int32(self.blocks)
                                cur_scale = next_scale

                                cute.arch.cp_async_wait_group(0)
                                cute.arch.sync_threads()

                                if has_next:
                                    next_scale = ld_global_b16(scales + (kb + Int32(1)).to(Int64) * Int64(2))
                                    if cutlass.const_expr(self.mtiles == 4):
                                        if a_ping_pong:
                                            self.stage_a_async(next_a, x_addr, srow, num_live_m, kb + Int32(1),
                                                               smem_thread_off, col_s, self.hidden_size, 3)
                                    self.stage_b_async_dyn(next_b, payload, kb + Int32(1), tid, width)
                                    cute.arch.cp_async_commit_group()

                                if width == Int32(2):
                                    self.accumulate(total, cur_a, cur_b, lane, warp, 2, cur_scale, num_live_m)
                                elif width == Int32(3):
                                    self.accumulate(total, cur_a, cur_b, lane, warp, 3, cur_scale, num_live_m)
                                elif width == Int32(4):
                                    self.accumulate(total, cur_a, cur_b, lane, warp, 4, cur_scale, num_live_m)

                                if has_next and not a_ping_pong:
                                    cute.arch.sync_threads()
                                    self.stage_a_async(shared_a, x_addr, srow, num_live_m, kb + Int32(1),
                                                       smem_thread_off, col_s, self.hidden_size, self.mtiles)
                                    cute.arch.cp_async_commit_group()

                        for v in cutlass.range_constexpr(8):
                            grand0[v] = grand0[v] + total[0, v]
                        if cutlass.const_expr(self.mtiles >= 2):
                            for v in cutlass.range_constexpr(8):
                                grand1[v] = grand1[v] + total[1, v]
                        if cutlass.const_expr(self.mtiles == 4):
                            if num_live_m >= Int32(3):
                                self.fold_shared_plane(shared_c2, tid, split, total, 2)
                            if num_live_m >= Int32(4):
                                self.fold_shared_plane(shared_c3, tid, split, total, 3)

                    for v in cutlass.range_constexpr(8):
                        total[0, v] = grand0[v]
                    if cutlass.const_expr(self.mtiles >= 2):
                        for v in cutlass.range_constexpr(8):
                            total[1, v] = grand1[v]
                    if cutlass.const_expr(self.mtiles == 4):
                        if num_live_m >= Int32(3):
                            self.load_shared_plane(shared_c2, tid, total, 2)
                        if num_live_m >= Int32(4):
                            self.load_shared_plane(shared_c3, tid, total, 3)

                    out_meta_f = bind_loop_i32(out_meta, num_live_m)
                    col0 = warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                    proj_out_base = fc1_addr + (
                        logical.to(Int64) * Int64(self.group_rows * 128) + col0.to(Int64)
                    ) * Int64(4)
                    for idx in cutlass.range_constexpr(self.mtiles):
                        if Int32(idx) < num_live_m:
                            m = (out_meta_f >> Int32(idx * 2)) & Int32(3)
                            for v in cutlass.range_constexpr(2):
                                if (out_meta_f & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                    local = m * Int32(16) + Int32(v * 8) + (lane >> Int32(2))
                                    row_addr = proj_out_base + local.to(Int64) * Int64(128 * 4)
                                    st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                    st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
                cute.arch.sync_threads()
            task += step

    # ------------------------------------------------------------------
    # FC1 on K4's warp-specialized structure (mixed_trellis_ws.py), EXLR8 decode
    # ------------------------------------------------------------------

    @cute.jit
    def fc1_ws_producer(self, ptid: Int32, ring: Int32, cta: Int32, grid: Int32,
                        x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                        unit_count: cute.Tensor, unit_map: cute.Tensor,
                        group_expert: cute.Tensor, route_slots: cute.Tensor,
                        rows: Int32, groups: Int32, unit_limit: Int32):
        """Producer warpgroup: per tile, publish a header (live M16 tiles, row
        mask, widths, every K128 scale), then stream the tile's K128 blocks:
        thread 0 bulk-copies the gate and up compressed blocks, all 128
        threads cp.async the live rows' activations (16 lanes per row)."""
        lane = ptid & Int32(31)
        stages = self.FC1_STAGES
        hdr_base = ring + Int32(self.fc1_hdr_off)
        bar_full = ring + Int32(self.fc1_bar_off)
        bar_empty = bar_full + Int32(8 * stages)
        bar_tfull = bar_empty + Int32(8 * stages)
        bar_tempty = bar_tfull + Int32(8 * _FC1_TILE_STAGES)
        max_slot = rows * Int32(self.top_k)
        x_addr = x.iterator.toint()
        c = ptid & Int32(15)
        token = cute.make_rmem_tensor(8, Int32)
        seq = Int32(0)
        ti = Int32(0)
        g_base = Int32(0)
        v_base = Int32(0)
        t = cta
        while g_base < groups:
            o0 = unit_count[group_expert[g_base]]
            o1 = unit_count[group_expert[g_base + Int32(1)]] if (g_base + Int32(1)) < groups else Int32(0)
            o2 = unit_count[group_expert[g_base + Int32(2)]] if (g_base + Int32(2)) < groups else Int32(0)
            o3 = unit_count[group_expert[g_base + Int32(3)]] if (g_base + Int32(3)) < groups else Int32(0)
            quad_tiles = o0 + o1 + o2 + o3
            if v_base + quad_tiles <= t:
                v_base += quad_tiles
                g_base += Int32(4)
            else:
                rem = t - v_base
                group = g_base
                k = Int32(0)
                pos = Int32(0)
                for u in cutlass.range(unit_limit):
                    for dg in cutlass.range_constexpr(4):
                        o_dg = o0 if dg == 0 else (o1 if dg == 1 else (o2 if dg == 2 else o3))
                        if u < o_dg:
                            if pos == rem:
                                group = g_base + Int32(dg)
                                k = u
                            pos += Int32(1)
                expert = group_expert[group]
                gbase = group.to(Int64) * Int64(self.group_rows)
                unit = unit_map[expert.to(Int64) * Int64(self.units) + k.to(Int64)]
                d0 = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
                w0 = bits[d0]
                w1 = bits[d0 + Int64(1)]
                s_lo = route_slots[gbase + lane.to(Int64)]
                s_hi = route_slots[gbase + Int64(32) + lane.to(Int64)]
                v_lo = ballot_sync_u32(((s_lo >= Int32(0)) & (s_lo < max_slot)).to(Int32))
                v_hi = ballot_sync_u32(((s_hi >= Int32(0)) & (s_hi < max_slot)).to(Int32))
                num_live = Int32(0)
                live = Int32(0)
                for m in cutlass.range_constexpr(4):
                    word = v_lo if m < 2 else v_hi
                    if ((word >> Uint32(16 * (m & 1))) & Uint32(0xFFFF)) != Uint32(0):
                        live = live | (Int32(m) << (num_live * Int32(2)))
                        num_live += Int32(1)
                if num_live > Int32(0) and (w0 != Int32(0) or w1 != Int32(0)):
                    pay0 = pointers[d0 * Int64(3)]
                    pay1 = pointers[(d0 + Int64(1)) * Int64(3)]
                    last_reader = Int32(1)
                    if group > Int32(0):
                        if group_expert[group - Int32(1)] == expert:
                            last_reader = Int32(0)
                    if group + Int32(1) < groups:
                        if group_expert[group + Int32(1)] == expert:
                            last_reader = Int32(0)
                    sc0 = Uint32(0)
                    if ptid < Int32(2 * self.blocks):
                        proj_s = Int32(0) if ptid < Int32(self.blocks) else Int32(1)
                        kb_s = ptid - proj_s * Int32(self.blocks)
                        wp_s = w0 if proj_s == Int32(0) else w1
                        if wp_s != Int32(0):
                            scp = pointers[(d0 + proj_s.to(Int64)) * Int64(3) + Int64(2)]
                            sc0 = ld_global_b16(scp + kb_s.to(Int64) * Int64(2))
                    r0 = ptid >> Int32(4)
                    rr = r0 & Int32(15)
                    valid_mask = Int32(0)
                    for i in cutlass.range_constexpr(8):
                        idx = Int32(i // 2)
                        tok = Int32(-1)
                        if idx < num_live:
                            m = (live >> (idx * Int32(2))) & Int32(3)
                            src_s = s_hi if m >= Int32(2) else s_lo
                            sl = cute.arch.shuffle_sync(src_s, (m & Int32(1)) * Int32(16) + rr + Int32(8 * (i & 1)))
                            if sl >= Int32(0) and sl < max_slot:
                                tok = sl // Int32(self.top_k)
                                valid_mask = valid_mask | Int32(1 << i)
                        token[i] = max(tok, Int32(0))
                    hslot = (Uint32(ti) % Uint32(_FC1_TILE_STAGES)).to(Int32)
                    _mbar_wait(bar_tempty + hslot * Int32(8),
                               (((Uint32(ti) // Uint32(_FC1_TILE_STAGES)) & Uint32(1)) ^ Uint32(1)).to(Int32))
                    hdr = hdr_base + hslot * Int32(self.fc1_hdr_bytes)
                    if ptid == Int32(0):
                        st_shared_i32(hdr, num_live)
                        st_shared_i32(hdr + Int32(4), live)
                        st_shared_i32(hdr + Int32(8), v_lo.to(Int32))
                        st_shared_i32(hdr + Int32(12), v_hi.to(Int32))
                        st_shared_i32(hdr + Int32(16), group * unit_limit + k)
                        st_shared_i32(hdr + Int32(20), w0)
                        st_shared_i32(hdr + Int32(24), w1)
                    if ptid < Int32(2 * self.blocks):
                        st_shared_i32(hdr + Int32(4 * _FC1_HEADER_WORDS) + ptid * Int32(4), sc0.to(Int32))
                    for q in cutlass.range_constexpr(1, (2 * self.blocks + 127) // 128):
                        e = ptid + Int32(128 * q)
                        if e < Int32(2 * self.blocks):
                            proj = Int32(0) if e < Int32(self.blocks) else Int32(1)
                            kb = e - proj * Int32(self.blocks)
                            wp = w0 if proj == Int32(0) else w1
                            sc = Uint32(0)
                            if wp != Int32(0):
                                scp = pointers[(d0 + proj.to(Int64)) * Int64(3) + Int64(2)]
                                sc = ld_global_b16(scp + kb.to(Int64) * Int64(2))
                            st_shared_i32(hdr + Int32(4 * _FC1_HEADER_WORDS) + e * Int32(4), sc.to(Int32))
                    _mbar_arrive(bar_tfull + hslot * Int32(8))
                    ti += Int32(1)

                    bytes_tx = (w0 + w1) * Int32(2048)
                    smem_row0 = bind_loop_i32(r0 * Int32(256) + (c ^ (r0 & Int32(7))) * Int32(16), ti)
                    x_col_base = x_addr + c.to(Int64) * Int64(16)
                    for kb in cutlass.range(self.blocks):
                        slot = (Uint32(seq) % Uint32(stages)).to(Int32)
                        _mbar_wait(bar_empty + slot * Int32(8),
                                   (((Uint32(seq) // Uint32(stages)) & Uint32(1)) ^ Uint32(1)).to(Int32))
                        stage = ring + slot * Int32(self.fc1_stage_bytes)
                        full = bar_full + slot * Int32(8)
                        if ptid == Int32(0):
                            _mbar_arrive_expect_tx(full, bytes_tx)
                            if last_reader != Int32(0):
                                policy = create_l2_evict_first_policy()
                                if w0 != Int32(0):
                                    n0 = w0 * Int32(2048)
                                    cp_async_bulk_g2s_mbar_l2hint(
                                        stage + Int32(self.fc1_a_bytes),
                                        pay0 + kb.to(Int64) * n0.to(Int64), n0, full, policy)
                                if w1 != Int32(0):
                                    n1 = w1 * Int32(2048)
                                    cp_async_bulk_g2s_mbar_l2hint(
                                        stage + Int32(self.fc1_a_bytes + 8192),
                                        pay1 + kb.to(Int64) * n1.to(Int64), n1, full, policy)
                            else:
                                if w0 != Int32(0):
                                    n0 = w0 * Int32(2048)
                                    cp_async_bulk_g2s_mbar(stage + Int32(self.fc1_a_bytes),
                                                           pay0 + kb.to(Int64) * n0.to(Int64), n0, full)
                                if w1 != Int32(0):
                                    n1 = w1 * Int32(2048)
                                    cp_async_bulk_g2s_mbar(stage + Int32(self.fc1_a_bytes + 8192),
                                                           pay1 + kb.to(Int64) * n1.to(Int64), n1, full)
                        if cutlass.const_expr(self.quantize_pass):
                            if c <= Int32(8):
                                for i in cutlass.range_constexpr(8):
                                    if Int32(i // 2) < num_live:
                                        r = r0 + Int32(8 * i)
                                        tok = token[i]
                                        valid = (valid_mask >> Int32(i)) & Int32(1)
                                        row_src = x_addr + (
                                            tok.to(Int64) * Int64(self.blocks)
                                            + kb.to(Int64)
                                        ) * Int64(256)
                                        self.stage_prequant_a_row_ws(
                                            stage + Int32((i // 2) * 4096), r & Int32(15), c, row_src, valid)
                            _cp_async_mbar_arrive_noinc(full)
                        elif cutlass.const_expr(self.a8):
                            x_kb = x_col_base + kb.to(Int64) * Int64(256)
                            for i in cutlass.range_constexpr(8):
                                if Int32(i // 2) < num_live:
                                    r = r0 + Int32(8 * i)
                                    tok = token[i]
                                    valid = (valid_mask >> Int32(i)) & Int32(1)
                                    src = x_kb + tok.to(Int64) * Int64(self.hidden_size * 2)
                                    self.quantize_stage_a_row(
                                        stage + Int32((i // 2) * 4096), r & Int32(15), c, src, valid)
                            _mbar_arrive(full)
                        else:
                            x_kb = x_col_base + kb.to(Int64) * Int64(256)
                            stage_row0 = stage + smem_row0
                            for i in cutlass.range_constexpr(8):
                                if Int32(i // 2) < num_live:
                                    src = x_kb + token[i].to(Int64) * Int64(self.hidden_size * 2)
                                    cp_async4_shared_global_cg_pred(
                                        stage_row0 + Int32(2048 * i), src, (valid_mask >> Int32(i)) & Int32(1))
                            _cp_async_mbar_arrive_noinc(full)
                        seq += Int32(1)
                t += grid
        hslot = ti % Int32(_FC1_TILE_STAGES)
        _mbar_wait(bar_tempty + hslot * Int32(8),
                   ((ti // Int32(_FC1_TILE_STAGES)) & Int32(1)) ^ Int32(1))
        if ptid == Int32(0):
            st_shared_i32(hdr_base + hslot * Int32(self.fc1_hdr_bytes), Int32(-1))
        _mbar_arrive(bar_tfull + hslot * Int32(8))

    @cute.jit
    def fc1_ws_block(self, total: cute.Tensor, a_base: Int32, b_base: Int32,
                     lane: Int32, n16_0: Int32, width: cutlass.Constexpr,
                     scale_addr: Int32, active: cutlass.Constexpr):
        """One K128 block of this warp's N32 (two N16 trellis records): per K16
        step decode both records (exact E4M3), ldmatrix each live M16 tile once
        and run 4 x active MMAs into a fresh block accumulator, then fold it
        into ``total`` with base FC1's pinned per-block scale FMA. Per element
        this is base's accumulate_fixed_m arithmetic."""
        acc = cute.make_rmem_tensor((active, 2, 8), Float32)
        acc.fill(Float32(0.0))
        r_m = lane & Int32(15)
        c_half = lane >> Int32(4)
        rec_b0 = Int32(0)
        src_lane = (lane + Int32(31)) & Int32(31)
        shift0 = Int32(0)
        if cutlass.const_expr(width == 4):
            rec_b0 = b_base + (n16_0 * Int32(32) + lane) * Int32(4)
        else:
            _, ib, shift0, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            rec_b0 = b_base + (n16_0 * Int32(8 * width) + ib) * Int32(4)
            if cutlass.const_expr(width == 3):
                src_lane = (lane + Int32(31) - ((lane & Int32(3)) == Int32(3)).to(Int32)) & Int32(31)
        packed_scale = ld_shared_i32_relaxed(scale_addr).to(Uint32)
        if cutlass.const_expr(self.a8):
            sa_base = a_base + Int32(2048) + (lane >> Int32(2)) * Int32(16)
            for k32 in cutlass.range_constexpr(4):
                frags = []
                for j in cutlass.range_constexpr(2):
                    e0_lo = Uint32(0)
                    e1_lo = Uint32(0)
                    e0_hi = Uint32(0)
                    e1_hi = Uint32(0)
                    for kh in cutlass.range_constexpr(2):
                        k16 = k32 * 2 + kh
                        wa = Uint32(0)
                        wb = Uint32(0)
                        if cutlass.const_expr(width == 4):
                            b = ld_ring_u32_offset(rec_b0, k16 * 1024 + j * 128)
                            a = cute.arch.shuffle_sync(b, src_lane)
                            wa = b
                            wb = (a << Uint32(16)) | (b >> Uint32(16))
                        else:
                            b = ld_ring_u32_offset(rec_b0, k16 * 256 * width + j * 32 * width)
                            a = cute.arch.shuffle_sync(b, src_lane)
                            w_sh = ((a.to(Int64) << Int64(32)) | b.to(Int64)) >> shift0.to(Int64)
                            wa = w_sh.to(Uint32)
                            wb = (w_sh >> Int64(4 * width)).to(Uint32)
                        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                        e_lo = _e4m3x4_from_half2x2(b0, b1)
                        e_hi = _e4m3x4_from_half2x2(b2, b3)
                        if cutlass.const_expr(kh == 0):
                            e0_lo, e0_hi = e_lo, e_hi
                        else:
                            e1_lo, e1_hi = e_lo, e_hi
                    frags.append((e0_lo, e1_lo, e0_hi, e1_hi))
                col8 = Int32(k32 * 2) + c_half
                addr_k32 = a_base + (r_m * Int32(8) + (col8 ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(active):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k32, idx * 4096)
                    sa0 = ld_ring_f32_offset(sa_base, idx * 4096 + k32 * 4)
                    sa1 = ld_ring_f32_offset(sa_base, idx * 4096 + 128 + k32 * 4)
                    for j in cutlass.range_constexpr(2):
                        f = frags[j]
                        d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                            a0, a1, a2, a3, f[0], f[1])
                        acc[idx, j, 0] = fma_rn_f32(d0, sa0, acc[idx, j, 0])
                        acc[idx, j, 1] = fma_rn_f32(d1, sa0, acc[idx, j, 1])
                        acc[idx, j, 2] = fma_rn_f32(d2, sa1, acc[idx, j, 2])
                        acc[idx, j, 3] = fma_rn_f32(d3, sa1, acc[idx, j, 3])
                        d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                            a0, a1, a2, a3, f[2], f[3])
                        acc[idx, j, 4] = fma_rn_f32(d4, sa0, acc[idx, j, 4])
                        acc[idx, j, 5] = fma_rn_f32(d5, sa0, acc[idx, j, 5])
                        acc[idx, j, 6] = fma_rn_f32(d6, sa1, acc[idx, j, 6])
                        acc[idx, j, 7] = fma_rn_f32(d7, sa1, acc[idx, j, 7])
        else:
            shift_b5 = Int32(0)
            if cutlass.const_expr(width == 3):
                shift_b5 = shift0 + Int32(5)
            b_next0 = ld_ring_u32_offset(rec_b0, 0)
            b_next1 = ld_ring_u32_offset(rec_b0, 32 * width)
            dq_ret_ty = llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()])
            dq_tail = """
                mul.lo.u32 w0, w0, M;  lop3.b32 w0, w0, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w1, w1, M;  lop3.b32 w1, w1, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w2, w2, M;  lop3.b32 w2, w2, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w3, w3, M;  lop3.b32 w3, w3, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w4, w4, M;  lop3.b32 w4, w4, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w5, w5, M;  lop3.b32 w5, w5, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w6, w6, M;  lop3.b32 w6, w6, 0x8fff8fff, 0x3b603b60, 0x6a;
                mul.lo.u32 w7, w7, M;  lop3.b32 w7, w7, 0x8fff8fff, 0x3b603b60, 0x6a;
                shf.r.wrap.b32 lo, w0, w1, 16;  lop3.b32 hi, w0, w1, 0x0000ffff, 0xe4;  add.rn.f16x2 $0, lo, hi;
                shf.r.wrap.b32 lo, w2, w3, 16;  lop3.b32 hi, w2, w3, 0x0000ffff, 0xe4;  add.rn.f16x2 $1, lo, hi;
                shf.r.wrap.b32 lo, w4, w5, 16;  lop3.b32 hi, w4, w5, 0x0000ffff, 0xe4;  add.rn.f16x2 $2, lo, hi;
                prmt.b32 lo, w6, w7, 0x5432;    lop3.b32 hi, w6, w7, 0x0000ffff, 0xe4;  add.rn.f16x2 $3, lo, hi;
                cvt.rn.satfinite.e4m3x2.f16x2 e0, $0;  cvt.rn.f16x2.e4m3x2 $0, e0;
                cvt.rn.satfinite.e4m3x2.f16x2 e1, $1;  cvt.rn.f16x2.e4m3x2 $1, e1;
                cvt.rn.satfinite.e4m3x2.f16x2 e2, $2;  cvt.rn.f16x2.e4m3x2 $2, e2;
                cvt.rn.satfinite.e4m3x2.f16x2 e3, $3;  cvt.rn.f16x2.e4m3x2 $3, e3;
            }
            """
            for k16 in cutlass.range_constexpr(8):
                b_cur0 = b_next0
                b_cur1 = b_next1
                frags = []
                for j in cutlass.range_constexpr(2):
                    b = b_cur0 if j == 0 else b_cur1
                    a = cute.arch.shuffle_sync(b, src_lane)
                    if cutlass.const_expr(width == 4):
                        res = llvm.inline_asm(
                            dq_ret_ty,
                            [Uint32(b).ir_value(), Uint32(a).ir_value()],
                            """
                            {
                                .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi, M;
                                .reg .b16 e0, e1, e2, e3;
                                mov.b32 M, 0xCBAC1FED;
                                and.b32 w7, $4, 0xffff;
                                shr.u32 w3, $4, 16;
                                shf.r.wrap.b32 w6, $4, $5, 4;   shr.u32 w2, w6, 16;  and.b32 w6, w6, 0xffff;
                                shf.r.wrap.b32 w5, $4, $5, 8;   shr.u32 w1, w5, 16;  and.b32 w5, w5, 0xffff;
                                shf.r.wrap.b32 w4, $4, $5, 12;  shr.u32 w0, w4, 16;  and.b32 w4, w4, 0xffff;
                            """ + dq_tail,
                            "=r,=r,=r,=r,r,r",
                            has_side_effects=False, is_align_stack=False,
                            asm_dialect=llvm.AsmDialect.AD_ATT)
                    elif cutlass.const_expr(width == 3):
                        res = llvm.inline_asm(
                            dq_ret_ty,
                            [Uint32(b).ir_value(), Uint32(a).ir_value(),
                             Int32(shift0).ir_value(), Int32(shift_b5).ir_value()],
                            """
                            {
                                .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, wa, wb5, lo, hi, M;
                                .reg .b16 e0, e1, e2, e3;
                                mov.b32 M, 0xCBAC1FED;
                                shf.r.wrap.b32 wa, $4, $5, $6;
                                shf.r.wrap.b32 wb5, $4, $5, $7;
                                and.b32 w7, wa, 0xffff;
                                shr.u32 w6, wa, 3;   and.b32 w6, w6, 0xffff;
                                shr.u32 w5, wa, 6;   and.b32 w5, w5, 0xffff;
                                shr.u32 w4, wa, 9;   and.b32 w4, w4, 0xffff;
                                shr.u32 w3, wa, 12;  and.b32 w3, w3, 0xffff;
                                shr.u32 w2, wa, 15;  and.b32 w2, w2, 0xffff;
                                shr.u32 w1, wb5, 13; and.b32 w1, w1, 0xffff;
                                shr.u32 w0, wb5, 16;
                            """ + dq_tail,
                            "=r,=r,=r,=r,r,r,r,r",
                            has_side_effects=False, is_align_stack=False,
                            asm_dialect=llvm.AsmDialect.AD_ATT)
                    else:
                        res = llvm.inline_asm(
                            dq_ret_ty,
                            [Uint32(b).ir_value(), Uint32(a).ir_value(),
                             Int32(shift0).ir_value()],
                            """
                            {
                                .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, wa, lo, hi, M;
                                .reg .b16 e0, e1, e2, e3;
                                mov.b32 M, 0xCBAC1FED;
                                shf.r.wrap.b32 wa, $4, $5, $6;
                                and.b32 w7, wa, 0xffff;
                                shr.u32 w6, wa, 2;   and.b32 w6, w6, 0xffff;
                                shr.u32 w5, wa, 4;   and.b32 w5, w5, 0xffff;
                                shr.u32 w4, wa, 6;   and.b32 w4, w4, 0xffff;
                                shr.u32 w3, wa, 8;   and.b32 w3, w3, 0xffff;
                                shr.u32 w2, wa, 10;  and.b32 w2, w2, 0xffff;
                                shr.u32 w1, wa, 12;  and.b32 w1, w1, 0xffff;
                                shr.u32 w0, wa, 14;  and.b32 w0, w0, 0xffff;
                            """ + dq_tail,
                            "=r,=r,=r,=r,r,r,r",
                            has_side_effects=False, is_align_stack=False,
                            asm_dialect=llvm.AsmDialect.AD_ATT)
                    frags.append((
                        Uint32(llvm.extractvalue(T.i32(), res, [0])),
                        Uint32(llvm.extractvalue(T.i32(), res, [1])),
                        Uint32(llvm.extractvalue(T.i32(), res, [2])),
                        Uint32(llvm.extractvalue(T.i32(), res, [3])),
                    ))
                if cutlass.const_expr(k16 < 7):
                    b_next0 = ld_ring_u32_offset(rec_b0, (k16 + 1) * 256 * width)
                    b_next1 = ld_ring_u32_offset(rec_b0, (k16 + 1) * 256 * width + 32 * width)
                col = Int32((k16 & 3) * 2) + c_half
                addr_k16 = a_base + (r_m * Int32(16) + (col ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(active):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k16, idx * 4096 + (k16 >> 2) * 128)
                    for j in cutlass.range_constexpr(2):
                        f = frags[j]
                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                            acc[idx, j, 0], acc[idx, j, 1], acc[idx, j, 2], acc[idx, j, 3],
                            a0, a1, a2, a3, f[0], f[1])
                        acc[idx, j, 0], acc[idx, j, 1], acc[idx, j, 2], acc[idx, j, 3] = d0, d1, d2, d3
                        d4, d5, d6, d7 = f16_mma_m16n8k16_f32(
                            acc[idx, j, 4], acc[idx, j, 5], acc[idx, j, 6], acc[idx, j, 7],
                            a0, a1, a2, a3, f[2], f[3])
                        acc[idx, j, 4], acc[idx, j, 5], acc[idx, j, 6], acc[idx, j, 7] = d4, d5, d6, d7
        scale, _ = half2_to_float2_scaled(packed_scale, Float32(1.0))
        for idx in cutlass.range_constexpr(active):
            for j in cutlass.range_constexpr(2):
                for v in cutlass.range_constexpr(8):
                    total[idx, j, v] = fma_rn_f32(acc[idx, j, v], scale, total[idx, j, v])

    @cute.jit
    def fc1_ws_fold(self, total: cute.Tensor, fc1_addr: Int64, hdr: Int32,
                    lane: Int32, cw: Int32, split: Int32, active: cutlass.Constexpr):
        """Base FC1's split fold through the output: split 0 stores 0 + T0,
        later splits store grand + Ts. Thread-local, live rows only."""
        hdr_s = bind_loop_i32(hdr, split)
        live = ld_shared_i32_relaxed(hdr_s + Int32(4))
        v_lo = ld_shared_i32_relaxed(hdr_s + Int32(8))
        v_hi = ld_shared_i32_relaxed(hdr_s + Int32(12))
        task = ld_shared_i32_relaxed(hdr_s + Int32(16))
        cw_s = bind_loop_i32(cw, task)
        lane_s = bind_loop_i32(lane, task)
        proj = cw_s >> Int32(2)
        col = (cw_s & Int32(3)) * Int32(32) + (lane_s & Int32(3)) * Int32(2)
        out_base = fc1_addr + (
            (task * Int32(2) + proj).to(Int64) * Int64(self.group_rows * 128)
            + col.to(Int64)) * Int64(4)
        for idx in cutlass.range_constexpr(active):
            m = (live >> Int32(idx * 2)) & Int32(3)
            for v in cutlass.range_constexpr(2):
                local = m * Int32(16) + Int32(v * 8) + (lane_s >> Int32(2))
                word = v_lo if local < Int32(32) else v_hi
                if ((word >> (local & Int32(31))) & Int32(1)) != Int32(0):
                    row_addr = out_base + local.to(Int64) * Int64(128 * 4)
                    if split == Int32(0):
                        for j in cutlass.range_constexpr(2):
                            for h in cutlass.range_constexpr(2):
                                addr = row_addr + Int64(64 * j + 32 * h)
                                e0 = total[idx, j, 4 * h + 2 * v]
                                e1 = total[idx, j, 4 * h + 2 * v + 1]
                                st_global_v2_f32(addr, Float32(0.0) + e0, Float32(0.0) + e1)
                    else:
                        g00, g01 = ld_global_v2_f32(row_addr)
                        g10, g11 = ld_global_v2_f32(row_addr + Int64(32))
                        g20, g21 = ld_global_v2_f32(row_addr + Int64(64))
                        g30, g31 = ld_global_v2_f32(row_addr + Int64(96))
                        st_global_v2_f32(row_addr, g00 + total[idx, 0, 2 * v], g01 + total[idx, 0, 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(32), g10 + total[idx, 0, 4 + 2 * v], g11 + total[idx, 0, 4 + 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(64), g20 + total[idx, 1, 2 * v], g21 + total[idx, 1, 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(96), g30 + total[idx, 1, 4 + 2 * v], g31 + total[idx, 1, 4 + 2 * v + 1])

    @cute.jit
    def fc1_ws_fold_tail4(self, total: cute.Tensor, fc1_addr: Int64, hdr: Int32,
                          lane: Int32, cw: Int32, split: Int32):
        hdr_s = bind_loop_i32(hdr, split)
        v_hi = ld_shared_i32_relaxed(hdr_s + Int32(12))
        task = ld_shared_i32_relaxed(hdr_s + Int32(16))
        proj = cw >> Int32(2)
        col = (cw & Int32(3)) * Int32(32) + (lane & Int32(3)) * Int32(2)
        out_base = fc1_addr + (
            (task * Int32(2) + proj).to(Int64) * Int64(self.group_rows * 128)
            + col.to(Int64)) * Int64(4)
        r_lane = lane >> Int32(2)
        for idx in cutlass.range_constexpr(2, 4):
            for v in cutlass.range_constexpr(2):
                bit = Int32((idx - 2) * 16 + v * 8) + r_lane
                if ((v_hi >> bit) & Int32(1)) != Int32(0):
                    local = Int32(32) + bit
                    row_addr = out_base + local.to(Int64) * Int64(128 * 4)
                    if split == Int32(0):
                        for j in cutlass.range_constexpr(2):
                            for h in cutlass.range_constexpr(2):
                                addr = row_addr + Int64(64 * j + 32 * h)
                                e0 = total[idx, j, 4 * h + 2 * v]
                                e1 = total[idx, j, 4 * h + 2 * v + 1]
                                st_global_v2_f32(addr, Float32(0.0) + e0, Float32(0.0) + e1)
                    else:
                        g00, g01 = ld_global_v2_f32(row_addr)
                        g10, g11 = ld_global_v2_f32(row_addr + Int64(32))
                        g20, g21 = ld_global_v2_f32(row_addr + Int64(64))
                        g30, g31 = ld_global_v2_f32(row_addr + Int64(96))
                        st_global_v2_f32(row_addr, g00 + total[idx, 0, 2 * v], g01 + total[idx, 0, 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(32), g10 + total[idx, 0, 4 + 2 * v], g11 + total[idx, 0, 4 + 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(64), g20 + total[idx, 1, 2 * v], g21 + total[idx, 1, 2 * v + 1])
                        st_global_v2_f32(row_addr + Int64(96), g30 + total[idx, 1, 4 + 2 * v], g31 + total[idx, 1, 4 + 2 * v + 1])

    @cute.jit
    def fc1_ws_store_head4(self, grand01: cute.Tensor, fc1_addr: Int64, hdr: Int32,
                           lane: Int32, cw: Int32):
        v_lo = ld_shared_i32_relaxed(hdr + Int32(8))
        task = ld_shared_i32_relaxed(hdr + Int32(16))
        proj = cw >> Int32(2)
        col = (cw & Int32(3)) * Int32(32) + (lane & Int32(3)) * Int32(2)
        out_base = fc1_addr + (
            (task * Int32(2) + proj).to(Int64) * Int64(self.group_rows * 128)
            + col.to(Int64)) * Int64(4)
        r_lane = lane >> Int32(2)
        for idx in cutlass.range_constexpr(2):
            for v in cutlass.range_constexpr(2):
                local = Int32(idx * 16 + v * 8) + r_lane
                if ((v_lo >> local) & Int32(1)) != Int32(0):
                    row_addr = out_base + local.to(Int64) * Int64(128 * 4)
                    for j in cutlass.range_constexpr(2):
                        for h in cutlass.range_constexpr(2):
                            addr = row_addr + Int64(64 * j + 32 * h)
                            st_global_v2_f32(addr, grand01[idx, j, 4 * h + 2 * v], grand01[idx, j, 4 * h + 2 * v + 1])

    @cute.jit
    def fc1_ws_tile(self, ring: Int32, hdr: Int32, seq: Int32,
                    lane: Int32, cw: Int32, fc1_addr: Int64, width: cutlass.Constexpr,
                    active: cutlass.Constexpr):
        total = cute.make_rmem_tensor((active, 2, 8), Float32)
        stages = self.FC1_STAGES
        bar_full = ring + Int32(self.fc1_bar_off)
        bar_empty = bar_full + Int32(8 * stages)
        proj = cw >> Int32(2)
        n16_0 = (cw & Int32(3)) * Int32(2)
        scale_base = hdr + Int32(4 * _FC1_HEADER_WORDS) + proj * Int32(4 * self.blocks)
        full_grand = not (self.a8 and not self.quantize_pass)
        if cutlass.const_expr((active <= 3 or full_grand) and width != 0):
            grand = cute.make_rmem_tensor((active, 2, 8), Float32)
            grand.fill(Float32(0.0))
            for split in cutlass.range(self.splits):
                total.fill(Float32(0.0))
                for block in cutlass.range(8):
                    kb = split * Int32(8) + block
                    if cutlass.const_expr(self.blocks % 8 == 0) or kb < Int32(self.blocks):
                        slot = (Uint32(seq) % Uint32(stages)).to(Int32)
                        _mbar_wait(bar_full + slot * Int32(8), ((Uint32(seq) // Uint32(stages)) & Uint32(1)).to(Int32))
                        stage = ring + slot * Int32(self.fc1_stage_bytes)
                        self.fc1_ws_block(total, stage,
                                          stage + Int32(self.fc1_a_bytes) + proj * Int32(8192),
                                          lane, n16_0, width, scale_base + kb * Int32(4), active)
                        _mbar_arrive(bar_empty + slot * Int32(8))
                        seq += Int32(1)
                for idx in cutlass.range_constexpr(active):
                    for j in cutlass.range_constexpr(2):
                        for v in cutlass.range_constexpr(8):
                            grand[idx, j, v] = grand[idx, j, v] + total[idx, j, v]
            self.fc1_ws_fold(grand, fc1_addr, hdr, lane, cw, Int32(0), active)
        elif cutlass.const_expr(active == 4 and width != 0):
            grand01 = cute.make_rmem_tensor((2, 2, 8), Float32)
            grand01.fill(Float32(0.0))
            for split in cutlass.range(self.splits):
                total.fill(Float32(0.0))
                for block in cutlass.range(8):
                    kb = split * Int32(8) + block
                    if cutlass.const_expr(self.blocks % 8 == 0) or kb < Int32(self.blocks):
                        slot = (Uint32(seq) % Uint32(stages)).to(Int32)
                        _mbar_wait(bar_full + slot * Int32(8), ((Uint32(seq) // Uint32(stages)) & Uint32(1)).to(Int32))
                        stage = ring + slot * Int32(self.fc1_stage_bytes)
                        self.fc1_ws_block(total, stage,
                                          stage + Int32(self.fc1_a_bytes) + proj * Int32(8192),
                                          lane, n16_0, width, scale_base + kb * Int32(4), active)
                        _mbar_arrive(bar_empty + slot * Int32(8))
                        seq += Int32(1)
                for idx in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(2):
                        for v in cutlass.range_constexpr(8):
                            grand01[idx, j, v] = grand01[idx, j, v] + total[idx, j, v]
                self.fc1_ws_fold_tail4(total, fc1_addr, hdr, lane, cw, split)
            self.fc1_ws_store_head4(grand01, fc1_addr, hdr, lane, cw)
        else:
            for split in cutlass.range(self.splits):
                if cutlass.const_expr(width != 0):
                    total.fill(Float32(0.0))
                for block in cutlass.range(8):
                    kb = split * Int32(8) + block
                    if cutlass.const_expr(self.blocks % 8 == 0) or kb < Int32(self.blocks):
                        slot = (Uint32(seq) % Uint32(stages)).to(Int32)
                        _mbar_wait(bar_full + slot * Int32(8), ((Uint32(seq) // Uint32(stages)) & Uint32(1)).to(Int32))
                        if cutlass.const_expr(width != 0):
                            stage = ring + slot * Int32(self.fc1_stage_bytes)
                            self.fc1_ws_block(total, stage,
                                              stage + Int32(self.fc1_a_bytes) + proj * Int32(8192),
                                              lane, n16_0, width, scale_base + kb * Int32(4), active)
                        _mbar_arrive(bar_empty + slot * Int32(8))
                        seq += Int32(1)
                if cutlass.const_expr(width != 0):
                    self.fc1_ws_fold(total, fc1_addr, hdr, lane, cw, split, active)
        return seq

    @cute.kernel
    def grouped_fc1_ws_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                              unit_count: cute.Tensor, unit_map: cute.Tensor,
                              group_expert: cute.Tensor, route_slots: cute.Tensor,
                              fc1: cute.Tensor, rows: Int32, group_count: cute.Tensor,
                              unit_limit: Int32):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, (self.fc1_smem_bytes + 3) // 4], 1024]

        storage = smem.allocate(Storage)
        ring = shared_ptr_to_u32(storage.words.data_ptr())
        stages = self.FC1_STAGES
        bar_full = ring + Int32(self.fc1_bar_off)
        bar_empty = bar_full + Int32(8 * stages)
        bar_tfull = bar_empty + Int32(8 * stages)
        bar_tempty = bar_tfull + Int32(8 * _FC1_TILE_STAGES)
        tid = Int32(cute.arch.thread_idx()[0])
        cta = Int32(cute.arch.block_idx()[0])
        grid = Int32(cute.arch.grid_dim()[0])
        if tid == Int32(0):
            for s in cutlass.range_constexpr(stages):
                _mbar_init(bar_full + Int32(8 * s), Int32(_FC1_PRODUCER_THREADS + 1))
                _mbar_init(bar_empty + Int32(8 * s), Int32(_FC1_CONSUMER_THREADS))
            for s in cutlass.range_constexpr(_FC1_TILE_STAGES):
                _mbar_init(bar_tfull + Int32(8 * s), Int32(_FC1_PRODUCER_THREADS))
                _mbar_init(bar_tempty + Int32(8 * s), Int32(_FC1_CONSUMER_THREADS))
            _fence_mbar_init()
        cute.arch.sync_threads()
        use_full_grand = not (self.a8 and not self.quantize_pass)
        prod_regs = _FC1_FULL_GRAND_PRODUCER_REGS if use_full_grand else _FC1_PRODUCER_REGS
        cons_regs = _FC1_FULL_GRAND_CONSUMER_REGS if use_full_grand else _FC1_CONSUMER_REGS
        if tid >= Int32(_FC1_CONSUMER_THREADS):
            cute.arch.warpgroup_reg_dealloc(prod_regs)
            self.fc1_ws_producer(tid - Int32(_FC1_CONSUMER_THREADS), ring, cta, grid, x,
                                 pointers, bits, unit_count, unit_map, group_expert,
                                 route_slots, rows, groups, unit_limit)
        else:
            cute.arch.warpgroup_reg_alloc(cons_regs)
            lane, cw = tid & Int32(31), tid >> Int32(5)
            fc1_addr = fc1.iterator.toint()
            hdr_base = ring + Int32(self.fc1_hdr_off)
            seq = Int32(0)
            ti = Int32(0)
            done = Int32(0)
            while done == Int32(0):
                hslot = ti % Int32(_FC1_TILE_STAGES)
                _mbar_wait(bar_tfull + hslot * Int32(8), (ti // Int32(_FC1_TILE_STAGES)) & Int32(1))
                hdr = hdr_base + hslot * Int32(self.fc1_hdr_bytes)
                num_live = ld_shared_i32_relaxed(hdr)
                if num_live < Int32(0):
                    done = Int32(1)
                else:
                    proj = cw >> Int32(2)
                    width = ld_shared_i32_relaxed(hdr + Int32(20) + proj * Int32(4))
                    for active in cutlass.range_constexpr(1, 5):
                        if num_live == Int32(active):
                            if width == Int32(4):
                                seq = self.fc1_ws_tile(ring, hdr, seq, lane, cw, fc1_addr, 4, active)
                            elif width == Int32(3):
                                seq = self.fc1_ws_tile(ring, hdr, seq, lane, cw, fc1_addr, 3, active)
                            elif width == Int32(2):
                                seq = self.fc1_ws_tile(ring, hdr, seq, lane, cw, fc1_addr, 2, active)
                            else:
                                seq = self.fc1_ws_tile(ring, hdr, seq, lane, cw, fc1_addr, 0, active)
                    _mbar_arrive(bar_tempty + hslot * Int32(8))
                    ti += Int32(1)

    @cute.kernel
    def activation_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                          unit_count: cute.Tensor, unit_map: cute.Tensor,
                          group_expert: cute.Tensor, route_slots: cute.Tensor,
                          fc1: cute.Tensor, intermediate: cute.Tensor,
                          rows: Int32, group_count: cute.Tensor, unit_limit: Int32):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        fc1_addr = fc1.iterator.toint()
        inter_addr = intermediate.iterator.toint()
        max_slot = rows * Int32(self.top_k)
        gs_val = cute.make_rmem_tensor(4, Float32)
        us_val = cute.make_rmem_tensor(4, Float32)
        ds_val = cute.make_rmem_tensor(4, Float32)
        g = cute.make_rmem_tensor(4, Float32)

        while task < groups * unit_limit:
            group, k = task // unit_limit, task % unit_limit
            expert = group_expert[group]
            if k < unit_count[expert]:
                unit = unit_map[expert.to(Int64) * Int64(self.units) + k.to(Int64)]
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
                if bits[desc] != Int32(0):
                    gbase = group.to(Int64) * Int64(self.group_rows)
                    s_lo = route_slots[gbase + (lane & Int32(self.group_rows - 1)).to(Int64)]
                    s_hi = (
                        route_slots[gbase + Int64(32) + lane.to(Int64)]
                        if cutlass.const_expr(self.group_rows == 64) else Int32(-1)
                    )
                    v_lo = ballot_sync_u32(((s_lo >= Int32(0)) & (s_lo < max_slot)).to(Int32))
                    v_hi = (
                        ballot_sync_u32(((s_hi >= Int32(0)) & (s_hi < max_slot)).to(Int32))
                        if cutlass.const_expr(self.group_rows == 64) else Uint32(0)
                    )
                    if (v_lo | v_hi) != Uint32(0):
                        gs = pointers[desc * Int64(3) + Int64(1)]
                        us = pointers[(desc + Int64(1)) * Int64(3) + Int64(1)]
                        ds = pointers[(desc + Int64(2)) * Int64(3) + Int64(1)]
                        for v in cutlass.range_constexpr(4):
                            col = lane * Int32(4) + Int32(v)
                            gs_val[v] = _side_value(gs, col)
                            us_val[v] = _side_value(us, col)
                            ds_val[v] = _side_value(ds, col)

                        for iteration in cutlass.range(self.group_rows // 8):
                            row = warp + iteration * Int32(8)
                            v_word = v_lo if row < Int32(32) else v_hi
                            if ((v_word >> Uint32(row & Int32(31))) & Uint32(1)) != Uint32(0):
                                row_col_off = row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
                                base_elem = task.to(Int64) * Int64(2 * self.group_rows * 128) + row_col_off
                                gw0, gw1, gw2, gw3 = ld_global_nc_v4_u32(fc1_addr + base_elem * Int64(4))
                                uw0, uw1, uw2, uw3 = ld_global_nc_v4_u32(
                                    fc1_addr + (base_elem + Int64(self.group_rows * 128)) * Int64(4))
                                gh = _w4a8_had128_quad(
                                    u32_as_f32(gw0), u32_as_f32(gw1), u32_as_f32(gw2), u32_as_f32(gw3), lane)
                                uh = _w4a8_had128_quad(
                                    u32_as_f32(uw0), u32_as_f32(uw1), u32_as_f32(uw2), u32_as_f32(uw3), lane)
                                for v in cutlass.range_constexpr(4):
                                    gate = gh[v] * gs_val[v]
                                    up = uh[v] * us_val[v]
                                    sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                                    g[v] = gate * sigmoid * up * ds_val[v]
                                h0, h1, h2, h3 = _w4a8_had128_quad(g[0], g[1], g[2], g[3], lane)
                                if cutlass.const_expr(self.quantize_pass):
                                    f0 = _f32x2_to_f16x2(h0, h1)
                                    f1 = _f32x2_to_f16x2(h2, h3)
                                    amax = _f16x4_absmax_f32(f0, f1)
                                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
                                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
                                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
                                    s_a, inv_s = _a8_scale_and_inv(amax)
                                    q4 = _quantize_f16x4_to_e4m3x4(f0, f1, inv_s)
                                    peer = cute.arch.shuffle_sync_bfly(q4, offset=2)
                                    out0, out1 = _prmt_5410_7632(q4, peer)
                                    row_dst = inter_addr + (task.to(Int64) * Int64(self.group_rows) + row.to(Int64)) * Int64(256)
                                    if (lane & Int32(2)) == Int32(0):
                                        c_out = (lane >> Int32(2)) * Int32(2) + (lane & Int32(1))
                                        st_global_v2_u32(row_dst + c_out.to(Int64) * Int64(8), out0, out1)
                                    if (lane & Int32(7)) == Int32(0):
                                        st_global_f32(row_dst + Int64(128) + (lane >> Int32(3)).to(Int64) * Int64(4), s_a)
                                else:
                                    out_off = (task.to(Int64) * Int64(self.group_rows) + row.to(Int64)) * Int64(128) + lane.to(Int64) * Int64(4)
                                    st_global_v2_u32(inter_addr + out_off * Int64(2),
                                                     _f32x2_to_f16x2(h0, h1), _f32x2_to_f16x2(h2, h3))
            task += step

    @cute.jit
    def launch_fc2(self, pointers: cute.Tensor, bits: cute.Tensor,
                   unit_count: cute.Tensor, unit_map: cute.Tensor,
                   group_expert: cute.Tensor, route_slots: cute.Tensor,
                   intermediate: cute.Tensor, fc2: cute.Tensor,
                   rows: Int32, group_count: cute.Tensor, unit_limit: Int32,
                   slice_base: Int32, slice_blocks: Int32, slice_shift: Int32,
                   grid_limit: Int32, stream: cuda.CUstream,
                   evict_first: cutlass.Constexpr = False):
        if cutlass.const_expr(self.FC2_WS):
            self.grouped_fc2_ws_kernel(
                pointers, bits, unit_count, unit_map, group_expert, route_slots,
                intermediate, fc2, rows, group_count, unit_limit,
                slice_base, slice_blocks, slice_shift, evict_first).launch(
                    grid=(min(Int32(self.max_groups * self.blocks), grid_limit), 1, 1),
                    block=(384, 1, 1), min_blocks_per_mp=1, stream=stream)
        else:
            self.grouped_fc2_kernel(
                pointers, bits, unit_count, unit_map, group_expert, route_slots,
                intermediate, fc2, rows, group_count, unit_limit,
                slice_base, slice_blocks, slice_shift).launch(
                    grid=(min(Int32(self.max_groups * self.blocks), grid_limit), 1, 1),
                    block=(256, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)

    @cute.kernel
    def grouped_fc2_ws_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           unit_count: cute.Tensor, unit_map: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, group_count: cute.Tensor, unit_limit: Int32,
                           slice_base: Int32, slice_blocks: Int32, slice_shift: Int32,
                           evict_first: cutlass.Constexpr = False):
        @dsl_user_op
        def _fc2_out_store_policy(is_sliced: Int32, *, loc=None, ip=None) -> Int64:
            return Int64(
                llvm.inline_asm(
                    T.i64(),
                    [Int32(is_sliced).ir_value(loc=loc, ip=ip)],
                    "{ .reg .pred p; .reg .b64 el, ef; "
                    "createpolicy.fractional.L2::evict_last.b64 el, 1.0; "
                    "createpolicy.fractional.L2::evict_first.b64 ef, 1.0; "
                    "setp.ne.b32 p, $1, 0; "
                    "selp.b64 $0, el, ef, p; }",
                    "=l,r",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )

        @dsl_user_op
        def _st_global_l2hint_v4_u32(
            addr: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, pol: Int64, *, loc=None, ip=None
        ):
            llvm.inline_asm(
                None,
                [
                    Int64(addr).ir_value(loc=loc, ip=ip),
                    Uint32(v0).ir_value(loc=loc, ip=ip),
                    Uint32(v1).ir_value(loc=loc, ip=ip),
                    Uint32(v2).ir_value(loc=loc, ip=ip),
                    Uint32(v3).ir_value(loc=loc, ip=ip),
                    Int64(pol).ir_value(loc=loc, ip=ip),
                ],
                "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;",
                "l,r,r,r,r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )

        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[
                Uint32, self.FC2_STAGES * (self.group_rows * 64 + 4096 + 8)], 1024]

        storage = smem.allocate(Storage)
        ring = shared_ptr_to_u32(storage.words.data_ptr())
        bar_full = ring + Int32(self.FC2_STAGES * (self.group_rows * 256 + 16384))
        bar_empty = bar_full + Int32(self.FC2_STAGES * 8)
        meta = bar_empty + Int32(self.FC2_STAGES * 8)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = max(Int32(1), Int32(cute.arch.grid_dim()[0]))
        lane, warp = tid & Int32(31), tid >> Int32(5)
        inter_addr = intermediate.iterator.toint()
        fc2_addr = fc2.iterator.toint()
        if tid < Int32(self.FC2_STAGES):
            _mbar_init(bar_full + tid * Int32(8), Int32(129))
            _mbar_init(bar_empty + tid * Int32(8), Int32(256))
        _fence_mbar_init()
        cute.arch.sync_threads()
        sequence = Int32(0)

        b_chunk_const = (
            8 if (self.blocks % 8 == 0)
            else (2 if (self.blocks % 2 == 0) else self.blocks)
        )
        chunks_per_group = self.blocks // b_chunk_const
        slice_blocks = max(Int32(2), slice_blocks)
        sliced = slice_blocks < Int32(self.blocks)
        two_chunks = (slice_shift == Int32(2)) | (slice_shift == Int32(4))
        slice_chunks = (
            Int32(1) if slice_shift == Int32(0)
            else (Int32(2) if two_chunks else Int32(3))
        )
        pair_slices = max(Int32(1), slice_blocks >> Int32(1))
        tail_waves = (
            Int32(2) if slice_blocks > Int32(24)
            else (Int32(1) if slice_blocks == Int32(24) else Int32(0))
        )
        min_for_bulk = (tail_waves + Int32(1)) * step
        avail_bulk = (groups - tail_waves * step) if (groups >= min_for_bulk) else Int32(0)
        bulk_groups = (avail_bulk // step) * step
        tail_groups = groups - bulk_groups

        tail_group_tasks = tail_groups * (slice_chunks if sliced else Int32(chunks_per_group))
        use_chunks = tail_group_tasks >= step
        tail_limit_tasks = tail_group_tasks if use_chunks else (tail_groups * pair_slices)
        limit_tasks = bulk_groups + tail_limit_tasks
        b_pairs_sliced = (
            Int32(1) if slice_shift == Int32(0)
            else ((slice_blocks >> Int32(2)) if two_chunks else (Int32(1) << (slice_shift - Int32(1))))
        )
        tail_b_pairs = (
            (b_pairs_sliced if sliced else Int32(b_chunk_const // 2))
            if use_chunks else Int32(1)
        )
        max_slot = rows * Int32(self.top_k)
        rev_groups = sliced & (((slice_base // slice_blocks) & Int32(1)) != Int32(0))

        if tid >= Int32(256):
            cute.arch.warpgroup_reg_dealloc(_FC1_PRODUCER_REGS)
            producer = tid - Int32(256)
            c_prod = producer & Int32(15)
            r_prod0 = producer >> Int32(4)
            r_prod1 = r_prod0 + Int32(8)
            off_prod0 = (r_prod0 * Int32(16) + (c_prod ^ (r_prod0 & Int32(7)))) * Int32(16)
            off_prod1 = (r_prod1 * Int32(16) + (c_prod ^ (r_prod1 & Int32(7)))) * Int32(16)

            while task < limit_tasks:
                group = Int32(0)
                b_local_base = Int32(0)
                b_pairs = pair_slices
                if task < bulk_groups:
                    group = task
                    b_local_base = Int32(0)
                    b_pairs = pair_slices
                else:
                    t_task = task - bulk_groups
                    b_pairs = tail_b_pairs
                    if cutlass.const_expr(self.blocks == 48):
                        if sliced:
                            if slice_shift == Int32(0):
                                group = t_task
                                b_local_base = Int32(0)
                            elif two_chunks:
                                if use_chunks:
                                    group = t_task >> Int32(1)
                                    b_local_base = (t_task & Int32(1)) * (b_pairs_sliced << Int32(1))
                                else:
                                    group = (t_task >> Int32(3)) if slice_shift == Int32(4) else ((t_task >> Int32(1)) // Int32(3))
                                    b_local_base = (t_task - group * pair_slices) << Int32(1)
                            else:
                                q = t_task if use_chunks else (t_task >> (slice_shift - Int32(1)))
                                group = q // Int32(3)
                                b_local_base = ((q - group * Int32(3)) << slice_shift) if use_chunks else ((t_task - group * pair_slices) << Int32(1))
                        else:
                            group = (t_task // Int32(chunks_per_group)) if use_chunks else (t_task // Int32(self.blocks // 2))
                            b_local_base = ((t_task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else ((t_task % Int32(self.blocks // 2)) * Int32(2))
                    else:
                        group = (t_task // Int32(chunks_per_group)) if use_chunks else (t_task // pair_slices)
                        b_local_base = ((t_task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else ((t_task - group * pair_slices) << Int32(1))
                    group += bulk_groups
                if rev_groups:
                    group = groups - Int32(1) - group
                block_base = slice_base + b_local_base

                expert = group_expert[group]
                group_base = group.to(Int64) * Int64(self.group_rows)
                slot_lo = route_slots[group_base + (lane & Int32(self.group_rows - 1)).to(Int64)]
                slot_hi = (
                    route_slots[group_base + Int64(32) + lane.to(Int64)]
                    if cutlass.const_expr(self.mtiles == 4) else Int32(-1)
                )

                out_meta = Int32(0)
                num_live_m = Int32(0)
                for m in cutlass.range_constexpr(self.mtiles):
                    src_m = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= 2)) else slot_lo
                    slot_check = cute.arch.shuffle_sync(src_m, Int32((m & 1) * 16) + (lane & Int32(15)))
                    is_live = cute.arch.vote_ballot_sync(
                        (slot_check >= Int32(0)) & (slot_check < max_slot)) != Uint32(0)
                    if is_live:
                        out_meta = out_meta | (Int32(m) << (num_live_m * Int32(2)))
                        num_live_m += Int32(1)

                if num_live_m > Int32(0):
                    owned = unit_count[expert]
                    if owned > Int32(0):
                        group_inter_base = inter_addr + group.to(Int64) * (
                            unit_limit.to(Int64) * Int64(self.group_rows * 128 * 2))
                        valid_mask = Int32(0)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                                m_lane_base = (m & Int32(1)) * Int32(16)
                                sl0 = cute.arch.shuffle_sync(src_word, m_lane_base + r_prod0)
                                sl1 = cute.arch.shuffle_sync(src_word, m_lane_base + r_prod1)
                                if sl0 >= Int32(0) and sl0 < max_slot:
                                    valid_mask = valid_mask | Int32(1 << (idx * 2))
                                if sl1 >= Int32(0) and sl1 < max_slot:
                                    valid_mask = valid_mask | Int32(1 << (idx * 2 + 1))
                        if cutlass.const_expr(evict_first):
                            last_reader = Int32(1)
                            next_g = (group - Int32(1)) if rev_groups else (group + Int32(1))
                            if next_g >= Int32(0) and next_g < groups:
                                if group_expert[next_g] == expert:
                                    last_reader = Int32(0)
                        my_w = Int32(0)
                        my_pay = Int64(0)
                        my_sc = Int64(0)
                        if producer < owned:
                            my_unit = unit_map[expert.to(Int64) * Int64(self.units) + producer.to(Int64)]
                            my_desc = (expert.to(Int64) * Int64(self.units) + my_unit.to(Int64)) * Int64(3) + Int64(2)
                            my_w = bits[my_desc]
                            my_pay = pointers[my_desc * Int64(3)]
                            my_sc = pointers[my_desc * Int64(3) + Int64(2)]
                        if slice_blocks > Int32(2):
                            for d in cutlass.range_constexpr(self.FC2_STAGES):
                                seq_d = sequence + Int32(d)
                                slot_d = seq_d % Int32(self.FC2_STAGES)
                                phase_d = (seq_d // Int32(self.FC2_STAGES)) & Int32(1)
                                _mbar_wait(bar_empty + slot_d * Int32(8), phase_d ^ Int32(1))
                        reuse_a = (
                            ((owned <= Int32(self.FC2_STAGES)) & (slice_blocks > Int32(2)))
                            if cutlass.const_expr(not (self.a8 and not self.quantize_pass))
                            else False
                        )
                        for bp in cutlass.range(b_pairs):
                            b0 = block_base + bp * Int32(2)
                            my_s01 = Uint32(0)
                            if producer < owned and my_w != Int32(0):
                                my_s0 = ld_global_b16(my_sc + b0.to(Int64) * Int64(2))
                                my_s1 = ld_global_b16(my_sc + (b0 + Int32(1)).to(Int64) * Int64(2))
                                my_s01 = (my_s0 & Uint32(0xFFFF)) | (my_s1 << Uint32(16))
                            stage_a_now = (bp == Int32(0)) if reuse_a else True
                            for k in cutlass.range(owned):
                                slot = sequence % Int32(self.FC2_STAGES)
                                phase = (sequence // Int32(self.FC2_STAGES)) & Int32(1)
                                if (not reuse_a) | (not stage_a_now) | (slice_blocks <= Int32(2)):
                                    _mbar_wait(bar_empty + slot * Int32(8), phase ^ Int32(1))
                                stage_b = ring + slot * Int32(self.group_rows * 256 + 16384) + Int32(self.group_rows * 256)
                                if producer < Int32(32):
                                    width = cute.arch.shuffle_sync(my_w, k)
                                    payload = shfl_sync_idx_i64(my_pay, k)
                                    scale01 = cute.arch.shuffle_sync(my_s01, k)
                                    if producer == Int32(0):
                                        meta_slot = meta + slot * Int32(16)
                                        st_shared_i32(meta_slot, width)
                                        st_shared_i32(meta_slot + Int32(4), scale01.to(Int32))
                                        if width != Int32(0):
                                            nbytes = width * Int32(2048)
                                            nbytes2 = width * Int32(4096)
                                            _mbar_arrive_expect_tx(bar_full + slot * Int32(8), nbytes2)
                                            if cutlass.const_expr(evict_first):
                                                if last_reader != Int32(0):
                                                    policy = create_l2_evict_first_policy()
                                                    cp_async_bulk_g2s_mbar_l2hint(
                                                        stage_b,
                                                        payload + b0.to(Int64) * nbytes.to(Int64),
                                                        nbytes2, bar_full + slot * Int32(8), policy)
                                                else:
                                                    cp_async_bulk_g2s_mbar(
                                                        stage_b,
                                                        payload + b0.to(Int64) * nbytes.to(Int64),
                                                        nbytes2, bar_full + slot * Int32(8))
                                            else:
                                                cp_async_bulk_g2s_mbar(
                                                    stage_b,
                                                    payload + b0.to(Int64) * nbytes.to(Int64),
                                                    nbytes2, bar_full + slot * Int32(8))
                                        else:
                                            _mbar_arrive(bar_full + slot * Int32(8))
                                if stage_a_now:
                                    a_slot = k if reuse_a else slot
                                    stage_a = ring + a_slot * Int32(self.group_rows * 256 + 16384)
                                    source_a = group_inter_base + k.to(Int64) * Int64(self.group_rows * 128 * 2)
                                    if cutlass.const_expr(self.quantize_pass):
                                        if c_prod <= Int32(8):
                                            for idx in cutlass.range_constexpr(self.mtiles):
                                                if Int32(idx) < num_live_m:
                                                    m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                                    v0 = (valid_mask >> Int32(idx * 2)) & Int32(1)
                                                    v1 = (valid_mask >> Int32(idx * 2 + 1)) & Int32(1)
                                                    self.stage_prequant_a_row_ws(
                                                        stage_a + Int32(idx * 4096), r_prod0, c_prod,
                                                        source_a + (m * Int32(16) + r_prod0).to(Int64) * Int64(256), v0)
                                                    self.stage_prequant_a_row_ws(
                                                        stage_a + Int32(idx * 4096), r_prod1, c_prod,
                                                        source_a + (m * Int32(16) + r_prod1).to(Int64) * Int64(256), v1)
                                    elif cutlass.const_expr(self.a8):
                                        c_byte_off = c_prod.to(Int64) * Int64(16)
                                        for idx in cutlass.range_constexpr(self.mtiles):
                                            if Int32(idx) < num_live_m:
                                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                                v0 = (valid_mask >> Int32(idx * 2)) & Int32(1)
                                                v1 = (valid_mask >> Int32(idx * 2 + 1)) & Int32(1)
                                                m_src = source_a + m.to(Int64) * Int64(4096) + c_byte_off
                                                self.quantize_stage_a_row(
                                                    stage_a + Int32(idx * 4096), r_prod0, c_prod,
                                                    m_src + r_prod0.to(Int64) * Int64(256), v0)
                                                self.quantize_stage_a_row(
                                                    stage_a + Int32(idx * 4096), r_prod1, c_prod,
                                                    m_src + r_prod1.to(Int64) * Int64(256), v1)
                                    else:
                                        c_byte_off = c_prod.to(Int64) * Int64(16)
                                        for idx in cutlass.range_constexpr(self.mtiles):
                                            if Int32(idx) < num_live_m:
                                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                                tile_stage = stage_a + Int32(idx * 4096)
                                                m_src = source_a + m.to(Int64) * Int64(4096) + c_byte_off
                                                v0 = (valid_mask >> Int32(idx * 2)) & Int32(1)
                                                v1 = (valid_mask >> Int32(idx * 2 + 1)) & Int32(1)
                                                cp_async4_shared_global_cg_zfill(
                                                    tile_stage + off_prod0,
                                                    m_src + r_prod0.to(Int64) * Int64(256), v0, Int32(1))
                                                cp_async4_shared_global_cg_zfill(
                                                    tile_stage + off_prod1,
                                                    m_src + r_prod1.to(Int64) * Int64(256), v1, Int32(1))
                                    if cutlass.const_expr(self.a8 and not self.quantize_pass):
                                        _mbar_arrive(bar_full + slot * Int32(8))
                                    else:
                                        _cp_async_mbar_arrive_noinc(bar_full + slot * Int32(8))
                                else:
                                    _mbar_arrive(bar_full + slot * Int32(8))
                                sequence += Int32(1)
                task += step
        else:
            cute.arch.warpgroup_reg_alloc(_FC1_CONSUMER_REGS)
            total = cute.make_rmem_tensor((self.mtiles, 2, 8), Float32)
            r_out = cute.make_rmem_tensor((self.mtiles, 2), Int32)
            out_pol = _fc2_out_store_policy(Int32(1) if sliced else Int32(0))

            while task < limit_tasks:
                group = Int32(0)
                b_local_base = Int32(0)
                b_pairs = pair_slices
                if task < bulk_groups:
                    group = task
                    b_local_base = Int32(0)
                    b_pairs = pair_slices
                else:
                    t_task = task - bulk_groups
                    b_pairs = tail_b_pairs
                    if cutlass.const_expr(self.blocks == 48):
                        if sliced:
                            if slice_shift == Int32(0):
                                group = t_task
                                b_local_base = Int32(0)
                            elif two_chunks:
                                if use_chunks:
                                    group = t_task >> Int32(1)
                                    b_local_base = (t_task & Int32(1)) * (b_pairs_sliced << Int32(1))
                                else:
                                    group = (t_task >> Int32(3)) if slice_shift == Int32(4) else ((t_task >> Int32(1)) // Int32(3))
                                    b_local_base = (t_task - group * pair_slices) << Int32(1)
                            else:
                                q = t_task if use_chunks else (t_task >> (slice_shift - Int32(1)))
                                group = q // Int32(3)
                                b_local_base = ((q - group * Int32(3)) << slice_shift) if use_chunks else ((t_task - group * pair_slices) << Int32(1))
                        else:
                            group = (t_task // Int32(chunks_per_group)) if use_chunks else (t_task // Int32(self.blocks // 2))
                            b_local_base = ((t_task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else ((t_task % Int32(self.blocks // 2)) * Int32(2))
                    else:
                        group = (t_task // Int32(chunks_per_group)) if use_chunks else (t_task // pair_slices)
                        b_local_base = ((t_task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else ((t_task - group * pair_slices) << Int32(1))
                    group += bulk_groups
                if rev_groups:
                    group = groups - Int32(1) - group

                expert = group_expert[group]
                group_base = group.to(Int64) * Int64(self.group_rows)
                slot_lo = route_slots[group_base + (lane & Int32(self.group_rows - 1)).to(Int64)]
                slot_hi = (
                    route_slots[group_base + Int64(32) + lane.to(Int64)]
                    if cutlass.const_expr(self.mtiles == 4) else Int32(-1)
                )

                out_meta = Int32(0)
                num_live_m = Int32(0)
                for m in cutlass.range_constexpr(self.mtiles):
                    src_m = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= 2)) else slot_lo
                    slot_check = cute.arch.shuffle_sync(src_m, Int32((m & 1) * 16) + (lane & Int32(15)))
                    is_live = cute.arch.vote_ballot_sync(
                        (slot_check >= Int32(0)) & (slot_check < max_slot)) != Uint32(0)
                    if is_live:
                        out_meta = out_meta | (Int32(m) << (num_live_m * Int32(2)))
                        num_live_m += Int32(1)

                if num_live_m > Int32(0):
                    owned = unit_count[expert]
                    reuse_a = (
                        ((owned <= Int32(self.FC2_STAGES)) & (slice_blocks > Int32(2)))
                        if cutlass.const_expr(not (self.a8 and not self.quantize_pass))
                        else False
                    )
                    for idx in cutlass.range_constexpr(self.mtiles):
                        if Int32(idx) < num_live_m:
                            m = (out_meta >> Int32(idx * 2)) & Int32(3)
                            src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                            m_lane_base = (m & Int32(1)) * Int32(16)
                            for v in cutlass.range_constexpr(2):
                                slot_out = cute.arch.shuffle_sync(
                                    src_word, m_lane_base + Int32(v * 8) + (lane >> Int32(2)))
                                r_out[idx, v] = slot_out if sliced else (m * Int32(16) + Int32(v * 8) + (lane >> Int32(2)))
                                if slot_out >= Int32(0) and slot_out < max_slot:
                                    out_meta = out_meta | Int32(1 << (8 + idx * 2 + v))
                    for bp in cutlass.range(b_pairs):
                        b_idx0 = bp * Int32(2)
                        b_out0 = b_local_base + b_idx0
                        total.fill(Float32(0.0))
                        for k in cutlass.range(owned):
                            slot = sequence % Int32(self.FC2_STAGES)
                            phase = (sequence // Int32(self.FC2_STAGES)) & Int32(1)
                            _mbar_wait(bar_full + slot * Int32(8), phase)
                            a_slot = k if reuse_a else slot
                            cur_a = ring + a_slot * Int32(self.group_rows * 256 + 16384)
                            cur_b = ring + slot * Int32(self.group_rows * 256 + 16384) + Int32(self.group_rows * 256)
                            meta_slot = meta + slot * Int32(16)
                            w_k = ld_shared_i32_relaxed(meta_slot)
                            scale01 = ld_shared_i32_relaxed(meta_slot + Int32(4)).to(Uint32)
                            scale0 = scale01 & Uint32(0xFFFF)
                            scale1 = scale01 >> Uint32(16)
                            if cutlass.const_expr(self.mtiles == 1):
                                if num_live_m >= Int32(1):
                                    if w_k == Int32(4):
                                        self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 1)
                                    else:
                                        wk_m1 = bind_loop_i32(w_k, slot)
                                        if wk_m1 == Int32(2):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 1)
                                        elif wk_m1 == Int32(3):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 1)
                            elif cutlass.const_expr(self.mtiles == 2):
                                if num_live_m >= Int32(2):
                                    if w_k == Int32(4):
                                        self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 2)
                                    else:
                                        wk_m2 = bind_loop_i32(w_k, slot)
                                        if wk_m2 == Int32(2):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 2)
                                        elif wk_m2 == Int32(3):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 2)
                                else:
                                    nm1 = bind_loop_i32(num_live_m, w_k)
                                    if nm1 == Int32(1):
                                        if w_k == Int32(4):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 1)
                                        else:
                                            wk_m2_1 = bind_loop_i32(w_k, slot)
                                            if wk_m2_1 == Int32(2):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 1)
                                            elif wk_m2_1 == Int32(3):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 1)
                            else:
                                if num_live_m >= Int32(4):
                                    if w_k == Int32(4):
                                        self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 4)
                                    else:
                                        wk_s4 = bind_loop_i32(w_k, slot)
                                        if wk_s4 == Int32(2):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 4)
                                        elif wk_s4 == Int32(3):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 4)
                                else:
                                    nm1 = bind_loop_i32(num_live_m, w_k)
                                    if nm1 == Int32(1):
                                        if w_k == Int32(4):
                                            self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 1)
                                        else:
                                            wk_s1 = bind_loop_i32(w_k, slot)
                                            if wk_s1 == Int32(2):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 1)
                                            elif wk_s1 == Int32(3):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 1)
                                    else:
                                        nm2 = bind_loop_i32(nm1, slot)
                                        if nm2 == Int32(2):
                                            if w_k == Int32(4):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 2)
                                            else:
                                                wk_s2 = bind_loop_i32(w_k, slot)
                                                if wk_s2 == Int32(2):
                                                    self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 2)
                                                elif wk_s2 == Int32(3):
                                                    self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 2)
                                        elif nm2 == Int32(3):
                                            if w_k == Int32(4):
                                                self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 4, scale0, scale1, 3)
                                            else:
                                                wk_s3 = bind_loop_i32(w_k, slot)
                                                if wk_s3 == Int32(2):
                                                    self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 2, scale0, scale1, 3)
                                                elif wk_s3 == Int32(3):
                                                    self.accumulate_fc2_pair_fixed_m(total, cur_a, cur_b, lane, warp, 3, scale0, scale1, 3)
                            _mbar_arrive(bar_empty + slot * Int32(8))
                            sequence += Int32(1)

                        out_meta_b = bind_loop_i32(out_meta, bp)
                        col_in_block = (
                            (warp * Int32(16) + (lane & Int32(1)) * Int32(8))
                            if cutlass.const_expr(self.FC2_HALF)
                            else (warp * Int32(16) + (lane & Int32(3)) * Int32(2))
                        )
                        block_col0 = (slice_base + b_out0) * Int32(128) + col_in_block
                        elem_bytes = 2 if self.FC2_HALF else 4
                        block_out_base0 = fc2_addr + (
                            ((b_out0 * max_slot).to(Int64) * Int64(128) + col_in_block.to(Int64))
                            if sliced
                            else ((group * Int32(self.group_rows)).to(Int64) * Int64(self.hidden_size) + block_col0.to(Int64))
                        ) * Int64(elem_bytes)
                        b1_out_stride = (max_slot.to(Int64) if sliced else Int64(1)) * Int64(128 * elem_bytes)
                        row_stride_bytes = Int64((128 if sliced else self.hidden_size) * elem_bytes)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                for v in cutlass.range_constexpr(2):
                                    row_idx = r_out[idx, v]
                                    row_addr0 = block_out_base0 + row_idx.to(Int64) * row_stride_bytes
                                    row_valid = (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0)
                                    for j in cutlass.range_constexpr(2):
                                        row_addr = row_addr0 + Int64(j) * b1_out_stride
                                        if cutlass.const_expr(self.FC2_HALF):
                                            lo = _f32x2_to_f16x2(total[idx, j, 2 * v], total[idx, j, 2 * v + 1])
                                            hi = _f32x2_to_f16x2(total[idx, j, 4 + 2 * v], total[idx, j, 4 + 2 * v + 1])
                                            base_lane = lane & Int32(28)
                                            l1 = cute.arch.shuffle_sync(lo, base_lane + Int32(1))
                                            l2 = cute.arch.shuffle_sync(lo, base_lane + Int32(2))
                                            l3 = cute.arch.shuffle_sync(lo, base_lane + Int32(3))
                                            h0 = cute.arch.shuffle_sync(hi, base_lane)
                                            h2 = cute.arch.shuffle_sync(hi, base_lane + Int32(2))
                                            h3 = cute.arch.shuffle_sync(hi, base_lane + Int32(3))
                                            is_hi = (lane & Int32(1)) != Int32(0)
                                            w0 = h0 if is_hi else lo
                                            w1 = hi if is_hi else l1
                                            w2 = h2 if is_hi else l2
                                            w3 = h3 if is_hi else l3
                                            if row_valid and (lane & Int32(2)) == Int32(0):
                                                _st_global_l2hint_v4_u32(row_addr, w0, w1, w2, w3, out_pol)
                                        else:
                                            if row_valid:
                                                st_global_v2_f32(row_addr, total[idx, j, 2 * v], total[idx, j, 2 * v + 1])
                                                st_global_v2_f32(row_addr + Int64(32), total[idx, j, 4 + 2 * v], total[idx, j, 4 + 2 * v + 1])
                task += step

    @cute.kernel
    def grouped_fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           unit_count: cute.Tensor, unit_map: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, group_count: cute.Tensor, unit_limit: Int32,
                           slice_base: Int32, slice_blocks: Int32, slice_shift: Int32):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[
                    Uint32,
                    2 * self.group_rows * 64 + 4096
                ], 1024]

        storage = smem.allocate(Storage)
        shared_a = shared_ptr_to_u32(storage.words.data_ptr())
        shared_b0 = shared_a + Int32(2 * self.group_rows * 256)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = max(Int32(1), Int32(cute.arch.grid_dim()[0]))
        lane, warp = tid & Int32(31), tid >> Int32(5)
        r_s, col_s = tid >> Int32(4), tid & Int32(15)
        smem_thread_off = (r_s * Int32(16) + (col_s ^ (r_s & Int32(7)))) * Int32(16)
        inter_addr = intermediate.iterator.toint()
        fc2_addr = fc2.iterator.toint()

        b_chunk_const = (
            8 if (self.blocks % 8 == 0)
            else (3 if (self.blocks % 3 == 0) else self.blocks)
        )
        chunks_per_group = self.blocks // b_chunk_const
        slice_blocks = max(Int32(2), slice_blocks)
        sliced = slice_blocks < Int32(self.blocks)
        two_chunks = (slice_shift == Int32(2)) | (slice_shift == Int32(4))
        slice_chunks = (
            Int32(1) if slice_shift == Int32(0)
            else (Int32(2) if two_chunks else Int32(3))
        )
        group_tasks = groups * (slice_chunks if sliced else Int32(chunks_per_group))
        use_chunks = group_tasks >= step
        limit_tasks = group_tasks if use_chunks else (groups * slice_blocks)
        b_count_sliced = (
            Int32(2) if slice_shift == Int32(0)
            else ((slice_blocks >> Int32(1)) if two_chunks else (Int32(1) << slice_shift))
        )
        b_count = (
            (b_count_sliced if sliced else Int32(b_chunk_const))
            if use_chunks else Int32(1)
        )
        max_slot = rows * Int32(self.top_k)

        total = cute.make_rmem_tensor((self.mtiles, 8), Float32)

        while task < limit_tasks:
            group = Int32(0)
            b_local_base = Int32(0)
            if cutlass.const_expr(self.blocks == 48):
                if sliced:
                    if slice_shift == Int32(0):
                        group = task if use_chunks else (task >> Int32(1))
                        b_local_base = Int32(0) if use_chunks else (task & Int32(1))
                    elif two_chunks:
                        if use_chunks:
                            group = task >> Int32(1)
                            b_local_base = (task & Int32(1)) * b_count_sliced
                        else:
                            group = (task >> Int32(4)) if slice_shift == Int32(4) else ((task >> Int32(2)) // Int32(3))
                            b_local_base = task - group * slice_blocks
                    else:
                        q = task if use_chunks else (task >> slice_shift)
                        group = q // Int32(3)
                        b_local_base = ((q - group * Int32(3)) << slice_shift) if use_chunks else (task - group * slice_blocks)
                else:
                    group = (task // Int32(chunks_per_group)) if use_chunks else (task // Int32(self.blocks))
                    b_local_base = ((task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else (task % Int32(self.blocks))
            else:
                group = (task // Int32(chunks_per_group)) if use_chunks else (task // Int32(self.blocks))
                b_local_base = ((task % Int32(chunks_per_group)) * Int32(b_chunk_const)) if use_chunks else (task % Int32(self.blocks))
            block_base = slice_base + b_local_base

            expert = group_expert[group]
            group_base = group.to(Int64) * Int64(self.group_rows)
            slot_lo = route_slots[group_base + (lane & Int32(self.group_rows - 1)).to(Int64)]
            slot_hi = (
                route_slots[group_base + Int64(32) + lane.to(Int64)]
                if cutlass.const_expr(self.mtiles == 4) else Int32(-1)
            )

            out_meta = Int32(0)
            num_live_m = Int32(0)
            for m in cutlass.range_constexpr(self.mtiles):
                src_m = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= 2)) else slot_lo
                slot_check = cute.arch.shuffle_sync(src_m, Int32((m & 1) * 16) + (lane & Int32(15)))
                is_live = cute.arch.vote_ballot_sync(
                    (slot_check >= Int32(0)) & (slot_check < max_slot)) != Uint32(0)
                if is_live:
                    out_meta = out_meta | (Int32(m) << (num_live_m * Int32(2)))
                    num_live_m += Int32(1)

            if num_live_m > Int32(0):
                for idx in cutlass.range_constexpr(self.mtiles):
                    if Int32(idx) < num_live_m:
                        m = (out_meta >> Int32(idx * 2)) & Int32(3)
                        src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                        m_lane_base = (m & Int32(1)) * Int32(16)
                        slot_s = cute.arch.shuffle_sync(src_word, m_lane_base + r_s)
                        if slot_s >= Int32(0) and slot_s < max_slot:
                            out_meta = out_meta | Int32(1 << (16 + idx))
                        else:
                            st_shared_v4_u32(shared_a + Int32(idx * 4096) + smem_thread_off,
                                             Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        for v in cutlass.range_constexpr(2):
                            slot_out = cute.arch.shuffle_sync(
                                src_word, m_lane_base + Int32(v * 8) + (lane >> Int32(2)))
                            if slot_out >= Int32(0) and slot_out < max_slot:
                                out_meta = out_meta | Int32(1 << (8 + idx * 2 + v))

                group_inter_base = inter_addr + group.to(Int64) * (
                    unit_limit.to(Int64) * Int64(self.group_rows * 128 * 2))
                owned = unit_count[expert]

                if owned > Int32(0):
                    my_w = Int32(0)
                    my_pay = Int64(0)
                    my_sc = Int64(0)
                    if lane < owned:
                        my_unit = unit_map[expert.to(Int64) * Int64(self.units) + lane.to(Int64)]
                        my_desc = (expert.to(Int64) * Int64(self.units) + my_unit.to(Int64)) * Int64(3) + Int64(2)
                        my_w = bits[my_desc]
                        my_pay = pointers[my_desc * Int64(3)]
                        my_sc = pointers[my_desc * Int64(3) + Int64(2)]

                    a_slot_stride = (
                        (Int32(8192) if num_live_m <= Int32(2) else Int32(16384))
                        if cutlass.const_expr(self.mtiles == 4)
                        else Int32(self.group_rows * 256)
                    )
                    max_a_slots = (
                        (Int32(4) if num_live_m <= Int32(2) else Int32(2))
                        if cutlass.const_expr(self.mtiles == 4)
                        else Int32(2)
                    )
                    pin_all_a = (owned <= max_a_slots) & (b_count > Int32(1))

                    w0 = cute.arch.shuffle_sync(my_w, Int32(0))
                    pay0 = shfl_sync_idx_i64(my_pay, Int32(0))
                    sc0 = shfl_sync_idx_i64(my_sc, Int32(0))
                    self.stage_a_fc2_async(shared_a, group_inter_base, out_meta, num_live_m,
                                           r_s, smem_thread_off, col_s)
                    if pin_all_a:
                        for u in cutlass.range(Int32(1), owned):
                            self.zero_invalid_fc2_a_tiles(
                                shared_a + u * a_slot_stride, out_meta, num_live_m, smem_thread_off)
                            u_addr = group_inter_base + u.to(Int64) * Int64(self.group_rows * 128 * 2)
                            self.stage_a_fc2_async(shared_a + u * a_slot_stride, u_addr, out_meta,
                                                   num_live_m, r_s, smem_thread_off, col_s)
                    else:
                        self.zero_invalid_fc2_a_tiles(
                            shared_a + a_slot_stride, out_meta, num_live_m, smem_thread_off)

                    self.stage_b_async_dyn(shared_b0, pay0, block_base, tid, w0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                    packed_scale = ld_global_b16(sc0 + block_base.to(Int64) * Int64(2)) if w0 != Int32(0) else Uint32(0)
                    cur_w = w0
                    step_parity = Int32(0)

                    for b_idx in cutlass.range(b_count):
                        b_out = b_local_base + b_idx
                        total.fill(Float32(0.0))
                        for k in cutlass.range(owned):
                            cur_scale = packed_scale
                            w_k = cur_w
                            parity = step_parity & Int32(1)
                            next_parity = (step_parity + Int32(1)) & Int32(1)
                            cur_b = shared_b0 + parity * Int32(8192)
                            next_b = shared_b0 + next_parity * Int32(8192)
                            cur_a = shared_a + (k if pin_all_a else parity) * a_slot_stride

                            next_k = k + Int32(1)
                            next_b_idx = b_idx
                            if next_k >= owned:
                                next_k = Int32(0)
                                next_b_idx = b_idx + Int32(1)
                            has_next = next_b_idx < b_count

                            if has_next:
                                next_block = block_base + next_b_idx
                                nw = cute.arch.shuffle_sync(my_w, next_k)
                                npay = shfl_sync_idx_i64(my_pay, next_k)
                                nsc = shfl_sync_idx_i64(my_sc, next_k)
                                cur_w = nw
                                packed_scale = (
                                    ld_global_b16(nsc + next_block.to(Int64) * Int64(2))
                                    if nw != Int32(0) else Uint32(0)
                                )
                                self.stage_b_async_dyn(next_b, npay, next_block, tid, nw)
                                if not pin_all_a:
                                    nua = group_inter_base + next_k.to(Int64) * Int64(self.group_rows * 128 * 2)
                                    self.stage_a_fc2_async(
                                        shared_a + next_parity * a_slot_stride, nua, out_meta,
                                        num_live_m, r_s, smem_thread_off, col_s)
                                cute.arch.cp_async_commit_group()

                            if w_k == Int32(2):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 2, cur_scale, num_live_m)
                            elif w_k == Int32(3):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 3, cur_scale, num_live_m)
                            elif w_k == Int32(4):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 4, cur_scale, num_live_m)

                            if has_next:
                                cute.arch.cp_async_wait_group(0)
                                cute.arch.sync_threads()

                            step_parity += Int32(1)

                        out_meta_b = bind_loop_i32(out_meta, b_idx)
                        col_in_block = warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                        block_col = (slice_base + b_out) * Int32(128) + col_in_block
                        block_out_base = fc2_addr + (
                            ((b_out * max_slot).to(Int64) * Int64(128) + col_in_block.to(Int64))
                            if sliced
                            else ((group * Int32(self.group_rows)).to(Int64) * Int64(self.hidden_size) + block_col.to(Int64))
                        ) * Int64(4)
                        row_stride_bytes = Int64(128 * 4) if sliced else Int64(self.hidden_size * 4)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta_b >> Int32(idx * 2)) & Int32(3)
                                src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                                src_base = (m & Int32(1)) * Int32(16) + (lane >> Int32(2))
                                for v in cutlass.range_constexpr(2):
                                    slot_out = cute.arch.shuffle_sync(src_word, src_base + Int32(v * 8))
                                    row_idx = slot_out if sliced else (m * Int32(16) + Int32(v * 8) + (lane >> Int32(2)))
                                    if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                        row_addr = block_out_base + row_idx.to(Int64) * row_stride_bytes
                                        st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                        st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
                else:
                    total.fill(Float32(0.0))
                    for b_idx in cutlass.range(b_count):
                        out_meta_b = bind_loop_i32(out_meta, b_idx)
                        b_out = b_local_base + b_idx
                        col_in_block = warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                        block_col = (slice_base + b_out) * Int32(128) + col_in_block
                        block_out_base = fc2_addr + (
                            ((b_out * max_slot).to(Int64) * Int64(128) + col_in_block.to(Int64))
                            if sliced
                            else ((group * Int32(self.group_rows)).to(Int64) * Int64(self.hidden_size) + block_col.to(Int64))
                        ) * Int64(4)
                        row_stride_bytes = Int64(128 * 4) if sliced else Int64(self.hidden_size * 4)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta_b >> Int32(idx * 2)) & Int32(3)
                                src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                                src_base = (m & Int32(1)) * Int32(16) + (lane >> Int32(2))
                                for v in cutlass.range_constexpr(2):
                                    slot_out = cute.arch.shuffle_sync(src_word, src_base + Int32(v * 8))
                                    row_idx = slot_out if sliced else (m * Int32(16) + Int32(v * 8) + (lane >> Int32(2)))
                                    if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                        row_addr = block_out_base + row_idx.to(Int64) * row_stride_bytes
                                        st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                        st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
            cute.arch.sync_threads()
            task += step

    @cute.kernel
    def reduce_kernel(self, route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                      fc2: cute.Tensor, signs_out: cute.Tensor, output: cute.Tensor,
                      rows: Int32, group_count: cute.Tensor,
                      slice_base: Int32, slice_blocks: Int32, slice_shift: Int32):
        @dsl_user_op
        def _store_bf16x4_cs(address: Int64, v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                             *, loc=None, ip=None):
            llvm.inline_asm(
                None,
                [Int64(address).ir_value(loc=loc, ip=ip), Float32(v0).ir_value(loc=loc, ip=ip),
                 Float32(v1).ir_value(loc=loc, ip=ip), Float32(v2).ir_value(loc=loc, ip=ip),
                 Float32(v3).ir_value(loc=loc, ip=ip)],
                "{ .reg .b32 lo, hi; cvt.rn.bf16x2.f32 lo, $2, $1; cvt.rn.bf16x2.f32 hi, $4, $3; "
                "st.global.cs.v2.b32 [$0], {lo, hi}; }",
                "l,f,f,f,f", has_side_effects=True, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        tid = Int32(cute.arch.thread_idx()[0])
        cta = Int32(cute.arch.block_idx()[0])
        grid = max(Int32(1), Int32(cute.arch.grid_dim()[0]))
        lane, warp = tid & Int32(31), tid >> Int32(5)
        task = cta * Int32(8) + warp
        step = grid * Int32(8)
        fc2_address = fc2.iterator.toint()
        signs_address = signs_out.iterator.toint()
        output_address = output.iterator.toint()
        max_slot = rows * Int32(self.top_k)
        slice_blocks = max(Int32(2), slice_blocks)
        sliced = slice_blocks < Int32(self.blocks)

        rw_address = route_weights.iterator.toint()
        while task < rows * slice_blocks:
            row = Int32(0)
            b_out = Int32(0)
            if cutlass.const_expr(self.blocks == 48):
                if sliced:
                    if slice_shift == Int32(0):
                        row = task >> Int32(1)
                        b_out = task & Int32(1)
                    elif slice_shift == Int32(4):
                        row = task >> Int32(4)
                        b_out = task & Int32(15)
                    else:
                        row = (task >> slice_shift) // Int32(3)
                        b_out = task - row * slice_blocks
                else:
                    row = task // Int32(48)
                    b_out = task % Int32(48)
            else:
                row = task // Int32(self.blocks)
                b_out = task % Int32(self.blocks)
            block = slice_base + b_out
            col = block * Int32(128) + lane * Int32(4)
            signs = ld_global_nc_u32(signs_address + col.to(Int64))
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0.0))
            if groups > Int32(0):
                row_route_base = row.to(Int64) * Int64(self.top_k)
                if sliced:
                    row_fc2_base = fc2_address + (
                        (b_out * max_slot).to(Int64) * Int64(128)
                        + row_route_base * Int64(128)
                        + lane.to(Int64) * Int64(4)
                    ) * Int64(2 if self.FC2_HALF else 4)
                    weights = cute.make_rmem_tensor(self.top_k, Float32)
                    if cutlass.const_expr(self.top_k == 8):
                        rw_base = rw_address + row_route_base * Int64(4)
                        rw0, rw1, rw2, rw3 = ld_global_nc_v4_u32(rw_base)
                        rw4, rw5, rw6, rw7 = ld_global_nc_v4_u32(rw_base + Int64(16))
                        weights[0] = u32_as_f32(rw0)
                        weights[1] = u32_as_f32(rw1)
                        weights[2] = u32_as_f32(rw2)
                        weights[3] = u32_as_f32(rw3)
                        weights[4] = u32_as_f32(rw4)
                        weights[5] = u32_as_f32(rw5)
                        weights[6] = u32_as_f32(rw6)
                        weights[7] = u32_as_f32(rw7)
                    else:
                        for slot in cutlass.range_constexpr(self.top_k):
                            weights[slot] = route_weights[row_route_base + Int64(slot)]
                    if cutlass.const_expr(self.FC2_HALF):
                        lo_s = cute.make_rmem_tensor(self.top_k, Uint32)
                        hi_s = cute.make_rmem_tensor(self.top_k, Uint32)
                        for slot in cutlass.range_constexpr(self.top_k):
                            lo_s[slot], hi_s[slot] = ld_global_lu_v2_u32_offset(row_fc2_base, slot * 256)
                        for slot in cutlass.range_constexpr(self.top_k):
                            w0, w1 = f16x2_to_f32x2(lo_s[slot])
                            w2, w3 = f16x2_to_f32x2(hi_s[slot])
                            weight = weights[slot]
                            values[0] = values[0] + weight * w0
                            values[1] = values[1] + weight * w1
                            values[2] = values[2] + weight * w2
                            values[3] = values[3] + weight * w3
                    else:
                        routes_s = cute.make_rmem_tensor((self.top_k, 4), Float32)
                        for slot in cutlass.range_constexpr(self.top_k):
                            w0, w1, w2, w3 = ld_global_lu_v4_f32_offset(row_fc2_base, slot * 512)
                            routes_s[slot, 0], routes_s[slot, 1], routes_s[slot, 2], routes_s[slot, 3] = w0, w1, w2, w3
                        for slot in cutlass.range_constexpr(self.top_k):
                            weight = weights[slot]
                            for v in cutlass.range_constexpr(4):
                                values[v] = values[v] + weight * routes_s[slot, v]
                elif cutlass.const_expr(self.REDUCE_MLP):
                    # Load every independent route before the ordered fold. The
                    # F32 route order and multiply/add rounding stay unchanged.
                    routes = cute.make_rmem_tensor((self.top_k, 4), Float32)
                    weights = cute.make_rmem_tensor(self.top_k, Float32)
                    valid = cute.make_rmem_tensor(self.top_k, Int32)
                    for slot in cutlass.range_constexpr(self.top_k):
                        route = row_route_base + Int64(slot)
                        packed = route_group_lane[route]
                        valid[slot] = (packed >= Int32(0)).to(Int32)
                        weights[slot] = route_weights[route]
                        for v in cutlass.range_constexpr(4):
                            routes[slot, v] = Float32(0.0)
                        if packed >= Int32(0):
                            address = fc2_address + (packed.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(2 if self.FC2_HALF else 4)
                            if cutlass.const_expr(self.FC2_HALF):
                                lo, hi = ld_global_nc_v2_u32(address)
                                routes[slot, 0], routes[slot, 1] = f16x2_to_f32x2(lo)
                                routes[slot, 2], routes[slot, 3] = f16x2_to_f32x2(hi)
                            else:
                                w0, w1, w2, w3 = ld_global_nc_v4_u32(address)
                                routes[slot, 0] = u32_as_f32(w0)
                                routes[slot, 1] = u32_as_f32(w1)
                                routes[slot, 2] = u32_as_f32(w2)
                                routes[slot, 3] = u32_as_f32(w3)
                    for slot in cutlass.range_constexpr(self.top_k):
                        if valid[slot] != Int32(0):
                            for v in cutlass.range_constexpr(4):
                                values[v] = values[v] + weights[slot] * routes[slot, v]
                else:
                    for slot in cutlass.range(Int32(self.top_k)):
                        route = row_route_base + slot.to(Int64)
                        packed = route_group_lane[route]
                        if packed >= Int32(0):
                            weight = route_weights[route]
                            address = fc2_address + (packed.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4)
                            w0, w1, w2, w3 = ld_global_nc_v4_u32(address)
                            values[0] = values[0] + weight * u32_as_f32(w0)
                            values[1] = values[1] + weight * u32_as_f32(w1)
                            values[2] = values[2] + weight * u32_as_f32(w2)
                            values[3] = values[3] + weight * u32_as_f32(w3)
            h0, h1, h2, h3 = _w4a8_had128_quad(values[0], values[1], values[2], values[3], lane)
            out_ptr = output_address + (row.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(2)
            o0 = h0 * signed_byte(signs, 0)
            o1 = h1 * signed_byte(signs, 1)
            o2 = h2 * signed_byte(signs, 2)
            o3 = h3 * signed_byte(signs, 3)
            if sliced:
                _store_bf16x4_cs(out_ptr, o0, o1, o2, o3)
            else:
                store_bf16x4(out_ptr, o0, o1, o2, o3)
            task += step
