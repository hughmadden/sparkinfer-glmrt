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
    CODE_VERSION = 125
    A8_CODE_VERSION = 126
    FC2_WS = True
    REDUCE_MLP = True
    FC2_HALF = True
    FC2_STAGES = 4
    # Warp-specialized FC1 preserves the original per-K128 FMA chain and
    # ascending K1024 split fold; only staging and tile scheduling change.
    FC1_WS = True
    FC1_STAGES = 3
    CTAS_PER_SM = 2

    def __init__(self, hidden_size, units_per_expert, top_k, max_groups, group_rows=64,
                 activations: str = "a16", quantize_pass: bool = True):
        super().__init__(hidden_size, units_per_expert, top_k, max_groups, device_count=True)
        if activations not in ("a16", "a8"):
            raise ValueError("EXLR8 grouped activations must be 'a16' or 'a8'")
        self.activations = activations
        self.a8 = activations == "a8"
        self.quantize_pass = bool(quantize_pass) and self.a8
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
                stream: cuda.CUstream):
        if (rows > Int32(0) and rows <= Int32(4096) and cta_limit > Int32(0)
                and unit_limit > Int32(0)):
            if cutlass.const_expr(self.quantize_pass):
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
                grid=(min(Int32(self.max_groups) * unit_limit, cta_limit), 1, 1),
                block=(256, 1, 1), stream=stream)
            if cutlass.const_expr(self.blocks == 48):
                if rows <= Int32(1024):
                    # Keep the wide-row FC2 specialization unchanged.
                    fc2_cta_limit = max(Int32(1), cta_limit >> Int32(1))
                    slice_shift = Int32(1) if rows <= Int32(512) else Int32(0)
                    slice_blocks = Int32(3) << slice_shift
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
                            grid=(min((rows * slice_blocks + Int32(7)) // Int32(8), cta_limit), 1, 1),
                            block=(256, 1, 1), stream=stream)
                        slice_base += slice_blocks
                else:
                    self.launch_fc2(
                        pointers, bits, unit_count, unit_map,
                        group_expert, route_slots, intermediate, fc2,
                        rows, group_count, unit_limit, Int32(0), Int32(48), Int32(4), cta_limit, stream)
                    self.reduce_kernel(
                        route_group_lane, route_weights, fc2, signs_out, output,
                        rows, group_count, Int32(0), Int32(48), Int32(4)
                    ).launch(
                        grid=(min((rows * Int32(48) + Int32(7)) // Int32(8), cta_limit), 1, 1),
                        block=(256, 1, 1), stream=stream)
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
                    hslot = ti % Int32(_FC1_TILE_STAGES)
                    _mbar_wait(bar_tempty + hslot * Int32(8),
                               ((ti // Int32(_FC1_TILE_STAGES)) & Int32(1)) ^ Int32(1))
                    hdr = hdr_base + hslot * Int32(self.fc1_hdr_bytes)
                    if ptid == Int32(0):
                        st_shared_i32(hdr, num_live)
                        st_shared_i32(hdr + Int32(4), live)
                        st_shared_i32(hdr + Int32(8), v_lo.to(Int32))
                        st_shared_i32(hdr + Int32(12), v_hi.to(Int32))
                        st_shared_i32(hdr + Int32(16), group * unit_limit + k)
                        st_shared_i32(hdr + Int32(20), w0)
                        st_shared_i32(hdr + Int32(24), w1)
                    for q in cutlass.range_constexpr((2 * self.blocks + 127) // 128):
                        e = ptid + Int32(128 * q)
                        if e < Int32(2 * self.blocks):
                            proj = e // Int32(self.blocks)
                            kb = e - proj * Int32(self.blocks)
                            wp = w0 if proj == Int32(0) else w1
                            sc = Uint32(0)
                            if wp != Int32(0):
                                scp = pointers[(d0 + proj.to(Int64)) * Int64(3) + Int64(2)]
                                sc = ld_global_b16(scp + kb.to(Int64) * Int64(2))
                            st_shared_i32(hdr + Int32(4 * _FC1_HEADER_WORDS) + e * Int32(4), sc.to(Int32))
                    _mbar_arrive(bar_tfull + hslot * Int32(8))
                    ti += Int32(1)

                    for i in cutlass.range_constexpr(8):
                        r = (ptid >> Int32(4)) + Int32(8 * i)
                        idx = Int32(i // 2)
                        rr = r & Int32(15)
                        tok = Int32(-1)
                        if idx < num_live:
                            m = (live >> (idx * Int32(2))) & Int32(3)
                            sl = route_slots[gbase + (m * Int32(16) + rr).to(Int64)]
                            if sl >= Int32(0) and sl < max_slot:
                                tok = sl // Int32(self.top_k)
                        token[i] = tok
                    bytes_tx = (w0 + w1) * Int32(2048)
                    last_reader = Int32(1)
                    if group + Int32(1) < groups:
                        if group_expert[group + Int32(1)] == expert:
                            last_reader = Int32(0)
                    for kb in cutlass.range(self.blocks):
                        slot = seq % Int32(stages)
                        _mbar_wait(bar_empty + slot * Int32(8),
                                   ((seq // Int32(stages)) & Int32(1)) ^ Int32(1))
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
                                    r = (ptid >> Int32(4)) + Int32(8 * i)
                                    tok = token[i]
                                    valid = (tok >= Int32(0)).to(Int32)
                                    row_src = x_addr + (
                                        max(tok, Int32(0)).to(Int64) * Int64(self.blocks)
                                        + kb.to(Int64)
                                    ) * Int64(256)
                                    self.stage_prequant_a_row_ws(
                                        stage + Int32((i // 2) * 4096), r & Int32(15), c, row_src, valid)
                            _cp_async_mbar_arrive_noinc(full)
                        elif cutlass.const_expr(self.a8):
                            col_bytes = (kb.to(Int64) * Int64(128) + c.to(Int64) * Int64(8)) * Int64(2)
                            for i in cutlass.range_constexpr(8):
                                r = (ptid >> Int32(4)) + Int32(8 * i)
                                tok = token[i]
                                valid = (tok >= Int32(0)).to(Int32)
                                src = x_addr + max(tok, Int32(0)).to(Int64) * Int64(self.hidden_size * 2) + col_bytes
                                self.quantize_stage_a_row(
                                    stage + Int32((i // 2) * 4096), r & Int32(15), c, src, valid)
                            _mbar_arrive(full)
                        else:
                            col_bytes = (kb.to(Int64) * Int64(128) + c.to(Int64) * Int64(8)) * Int64(2)
                            for i in cutlass.range_constexpr(8):
                                r = (ptid >> Int32(4)) + Int32(8 * i)
                                tok = token[i]
                                valid = (tok >= Int32(0)).to(Int32)
                                src = x_addr + max(tok, Int32(0)).to(Int64) * Int64(self.hidden_size * 2) + col_bytes
                                cp_async4_shared_global_cg_pred(
                                    stage + r * Int32(256) + (c ^ (r & Int32(7))) * Int32(16), src, valid)
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
        rec_a0 = Int32(0)
        rec_b0 = Int32(0)
        shift0 = Int32(0)
        shift1 = Int32(0)
        if cutlass.const_expr(width == 4):
            rec_b0 = b_base + (n16_0 * Int32(32) + lane) * Int32(4)
        else:
            ia, ib, shift0, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            rec_a0 = b_base + (n16_0 * Int32(8 * width) + ia) * Int32(4)
            rec_b0 = b_base + (n16_0 * Int32(8 * width) + ib) * Int32(4)
            shift1 = shift0 + Int32(4 * width)
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
                            a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                            wa = b
                            wb = (a << Uint32(16)) | (b >> Uint32(16))
                        else:
                            a = ld_ring_u32_offset(rec_a0, k16 * 256 * width + j * 32 * width)
                            b = ld_ring_u32_offset(rec_b0, k16 * 256 * width + j * 32 * width)
                            window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                            wa = (window >> shift0.to(Int64)).to(Uint32)
                            wb = (window >> shift1.to(Int64)).to(Uint32)
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
            for k16 in cutlass.range_constexpr(8):
                frags = []
                for j in cutlass.range_constexpr(2):
                    wa = Uint32(0)
                    wb = Uint32(0)
                    if cutlass.const_expr(width == 4):
                        b = ld_ring_u32_offset(rec_b0, k16 * 1024 + j * 128)
                        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                        wa = b
                        wb = (a << Uint32(16)) | (b >> Uint32(16))
                    else:
                        a = ld_ring_u32_offset(rec_a0, k16 * 256 * width + j * 32 * width)
                        b = ld_ring_u32_offset(rec_b0, k16 * 256 * width + j * 32 * width)
                        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                        wa = (window >> shift0.to(Int64)).to(Uint32)
                        wb = (window >> shift1.to(Int64)).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    frags.append((_e4m3_round_half2(b0), _e4m3_round_half2(b1),
                                  _e4m3_round_half2(b2), _e4m3_round_half2(b3)))
                col = Int32(k16 * 2) + c_half
                addr_k16 = a_base + (r_m * Int32(16) + (col ^ (r_m & Int32(7)))) * Int32(16)
                for idx in cutlass.range_constexpr(active):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16_offset(addr_k16, idx * 4096)
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
        packed_scale = ld_shared_i32_relaxed(scale_addr).to(Uint32)
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
        proj = cw >> Int32(2)
        col = (cw & Int32(3)) * Int32(32) + (lane & Int32(3)) * Int32(2)
        out_base = fc1_addr + (
            (task * Int32(2) + proj).to(Int64) * Int64(self.group_rows * 128)
            + col.to(Int64)) * Int64(4)
        for idx in cutlass.range_constexpr(active):
            m = (live >> Int32(idx * 2)) & Int32(3)
            for v in cutlass.range_constexpr(2):
                local = m * Int32(16) + Int32(v * 8) + (lane >> Int32(2))
                word = v_lo if local < Int32(32) else v_hi
                if ((word >> (local & Int32(31))) & Int32(1)) != Int32(0):
                    row_addr = out_base + local.to(Int64) * Int64(128 * 4)
                    for j in cutlass.range_constexpr(2):
                        for h in cutlass.range_constexpr(2):
                            addr = row_addr + Int64(64 * j + 32 * h)
                            e0 = total[idx, j, 4 * h + 2 * v]
                            e1 = total[idx, j, 4 * h + 2 * v + 1]
                            if split == Int32(0):
                                st_global_v2_f32(addr, Float32(0.0) + e0, Float32(0.0) + e1)
                            else:
                                g0, g1 = ld_global_v2_f32(addr)
                                st_global_v2_f32(addr, g0 + e0, g1 + e1)

    @cute.jit
    def fc1_ws_tile(self, total: cute.Tensor, ring: Int32, hdr: Int32, seq: Int32,
                    lane: Int32, cw: Int32, fc1_addr: Int64, width: cutlass.Constexpr,
                    active: cutlass.Constexpr):
        stages = self.FC1_STAGES
        bar_full = ring + Int32(self.fc1_bar_off)
        bar_empty = bar_full + Int32(8 * stages)
        proj = cw >> Int32(2)
        n16_0 = (cw & Int32(3)) * Int32(2)
        scale_base = hdr + Int32(4 * _FC1_HEADER_WORDS) + proj * Int32(4 * self.blocks)
        for split in cutlass.range(self.splits):
            if cutlass.const_expr(width != 0):
                total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    slot = seq % Int32(stages)
                    _mbar_wait(bar_full + slot * Int32(8), (seq // Int32(stages)) & Int32(1))
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
        if tid >= Int32(_FC1_CONSUMER_THREADS):
            cute.arch.warpgroup_reg_dealloc(_FC1_PRODUCER_REGS)
            self.fc1_ws_producer(tid - Int32(_FC1_CONSUMER_THREADS), ring, cta, grid, x,
                                 pointers, bits, unit_count, unit_map, group_expert,
                                 route_slots, rows, groups, unit_limit)
        else:
            cute.arch.warpgroup_reg_alloc(_FC1_CONSUMER_REGS)
            lane, cw = tid & Int32(31), tid >> Int32(5)
            fc1_addr = fc1.iterator.toint()
            hdr_base = ring + Int32(self.fc1_hdr_off)
            total = cute.make_rmem_tensor((4, 2, 8), Float32)
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
                                seq = self.fc1_ws_tile(total, ring, hdr, seq, lane, cw, fc1_addr, 4, active)
                            elif width == Int32(3):
                                seq = self.fc1_ws_tile(total, ring, hdr, seq, lane, cw, fc1_addr, 3, active)
                            elif width == Int32(2):
                                seq = self.fc1_ws_tile(total, ring, hdr, seq, lane, cw, fc1_addr, 2, active)
                            else:
                                seq = self.fc1_ws_tile(total, ring, hdr, seq, lane, cw, fc1_addr, 0, active)
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
                        slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
                        if slot >= Int32(0) and slot < max_slot:
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
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[
                Uint32, self.FC2_STAGES * (self.group_rows * 64 + 2048 + 6)], 1024]

        storage = smem.allocate(Storage)
        ring = shared_ptr_to_u32(storage.words.data_ptr())
        bar_full = ring + Int32(self.FC2_STAGES * (self.group_rows * 256 + 8192))
        bar_empty = bar_full + Int32(self.FC2_STAGES * 8)
        meta = bar_empty + Int32(self.FC2_STAGES * 8)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        r_s = (tid & Int32(255)) >> Int32(4)
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
            else (3 if (self.blocks % 3 == 0) else self.blocks)
        )
        chunks_per_group = self.blocks // b_chunk_const
        sliced = slice_blocks < Int32(self.blocks)
        group_tasks = groups * (Int32(3) if sliced else Int32(chunks_per_group))
        use_chunks = group_tasks >= step
        limit_tasks = group_tasks if use_chunks else (groups * slice_blocks)
        b_count = (
            ((Int32(1) << slice_shift) if sliced else Int32(b_chunk_const))
            if use_chunks else Int32(1)
        )
        max_slot = rows * Int32(self.top_k)

        total = cute.make_rmem_tensor((self.mtiles, 8), Float32)

        while task < limit_tasks:
            group = Int32(0)
            b_local_base = Int32(0)
            if cutlass.const_expr(self.blocks == 48):
                if sliced:
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
                        for v in cutlass.range_constexpr(2):
                            slot_out = cute.arch.shuffle_sync(
                                src_word, m_lane_base + Int32(v * 8) + (lane >> Int32(2)))
                            if slot_out >= Int32(0) and slot_out < max_slot:
                                out_meta = out_meta | Int32(1 << (8 + idx * 2 + v))

                group_inter_base = inter_addr + group.to(Int64) * (
                    unit_limit.to(Int64) * Int64(self.group_rows * 128 * 2))
                owned = unit_count[expert]

                if tid >= Int32(256):
                    producer = tid - Int32(256)
                    if cutlass.const_expr(self.quantize_pass):
                        c_prod = producer & Int32(15)
                        r_prod0 = producer >> Int32(4)
                        r_prod1 = r_prod0 + Int32(8)
                        valid_mask = Int32(0)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                sl0 = route_slots[group_base + (m * Int32(16) + r_prod0).to(Int64)]
                                sl1 = route_slots[group_base + (m * Int32(16) + r_prod1).to(Int64)]
                                if sl0 >= Int32(0) and sl0 < max_slot:
                                    valid_mask = valid_mask | Int32(1 << (idx * 2))
                                if sl1 >= Int32(0) and sl1 < max_slot:
                                    valid_mask = valid_mask | Int32(1 << (idx * 2 + 1))
                    if cutlass.const_expr(evict_first):
                        # Evict disjoint weight bytes only after the final expert group reads them.
                        last_reader = Int32(1)
                        if group + Int32(1) < groups:
                            if group_expert[group + Int32(1)] == expert:
                                last_reader = Int32(0)
                    for b_idx in cutlass.range(b_count):
                        for k in cutlass.range(owned):
                            slot = sequence & Int32(self.FC2_STAGES - 1)
                            phase = (sequence // Int32(self.FC2_STAGES)) & Int32(1)
                            _mbar_wait(bar_empty + slot * Int32(8), phase ^ Int32(1))
                            stage = ring + slot * Int32(self.group_rows * 256 + 8192)
                            width, payload, scales, source_a = self.get_fc2_unit_desc(
                                pointers, bits, unit_map, expert, k, group_inter_base)
                            if producer == Int32(0):
                                scale = ld_global_b16(scales + (block_base + b_idx).to(Int64) * Int64(2)) if width != Int32(0) else Uint32(0)
                                st_shared_i32(meta + slot * Int32(8), width)
                                st_shared_i32(meta + slot * Int32(8) + Int32(4), scale.to(Int32))
                                if width != Int32(0):
                                    nbytes = width * Int32(2048)
                                    _mbar_arrive_expect_tx(bar_full + slot * Int32(8), nbytes)
                                    if cutlass.const_expr(evict_first):
                                        if last_reader != Int32(0):
                                            policy = create_l2_evict_first_policy()
                                            cp_async_bulk_g2s_mbar_l2hint(
                                                stage + Int32(self.group_rows * 256),
                                                payload + (block_base + b_idx).to(Int64) * nbytes.to(Int64),
                                                nbytes, bar_full + slot * Int32(8), policy)
                                        else:
                                            cp_async_bulk_g2s_mbar(
                                                stage + Int32(self.group_rows * 256),
                                                payload + (block_base + b_idx).to(Int64) * nbytes.to(Int64),
                                                nbytes, bar_full + slot * Int32(8))
                                    else:
                                        cp_async_bulk_g2s_mbar(
                                            stage + Int32(self.group_rows * 256),
                                            payload + (block_base + b_idx).to(Int64) * nbytes.to(Int64),
                                            nbytes, bar_full + slot * Int32(8))
                                else:
                                    _mbar_arrive(bar_full + slot * Int32(8))
                            if cutlass.const_expr(self.quantize_pass):
                                if c_prod <= Int32(8):
                                    for idx in cutlass.range_constexpr(self.mtiles):
                                        if Int32(idx) < num_live_m:
                                            m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                            v0 = (valid_mask >> Int32(idx * 2)) & Int32(1)
                                            v1 = (valid_mask >> Int32(idx * 2 + 1)) & Int32(1)
                                            self.stage_prequant_a_row_ws(
                                                stage + Int32(idx * 4096), r_prod0, c_prod,
                                                source_a + (m * Int32(16) + r_prod0).to(Int64) * Int64(256), v0)
                                            self.stage_prequant_a_row_ws(
                                                stage + Int32(idx * 4096), r_prod1, c_prod,
                                                source_a + (m * Int32(16) + r_prod1).to(Int64) * Int64(256), v1)
                            else:
                                for half in cutlass.range_constexpr(2):
                                    copy_tid = producer + Int32(half * 128)
                                    r = copy_tid >> Int32(4)
                                    c = copy_tid & Int32(15)
                                    if cutlass.const_expr(self.a8):
                                        for idx in cutlass.range_constexpr(self.mtiles):
                                            if Int32(idx) < num_live_m:
                                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                                raw_slot = route_slots[group_base + (m * Int32(16) + r).to(Int64)]
                                                valid = ((raw_slot >= Int32(0)) & (raw_slot < max_slot)).to(Int32)
                                                source = source_a + ((m * Int32(16) + r).to(Int64) * Int64(128) + c.to(Int64) * Int64(8)) * Int64(2)
                                                self.quantize_stage_a_row(stage + Int32(idx * 4096), r, c, source, valid)
                                    else:
                                        offset = (r * Int32(16) + (c ^ (r & Int32(7)))) * Int32(16)
                                        for idx in cutlass.range_constexpr(self.mtiles):
                                            if Int32(idx) < num_live_m:
                                                m = (out_meta >> Int32(idx * 2)) & Int32(3)
                                                raw_slot = route_slots[group_base + (m * Int32(16) + r).to(Int64)]
                                                target = stage + Int32(idx * 4096) + offset
                                                if raw_slot >= Int32(0) and raw_slot < max_slot:
                                                    source = source_a + ((m * Int32(16) + r).to(Int64) * Int64(128) + c.to(Int64) * Int64(8)) * Int64(2)
                                                    cp_async4_shared_global(target, source)
                                                else:
                                                    st_shared_v4_u32(target, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                            if cutlass.const_expr(self.a8 and not self.quantize_pass):
                                _mbar_arrive(bar_full + slot * Int32(8))
                            else:
                                _cp_async_mbar_arrive_noinc(bar_full + slot * Int32(8))
                            sequence += Int32(1)
                else:
                    for b_idx in cutlass.range(b_count):
                        b_out = b_local_base + b_idx
                        total.fill(Float32(0.0))
                        for k in cutlass.range(owned):
                            slot = sequence & Int32(self.FC2_STAGES - 1)
                            phase = (sequence // Int32(self.FC2_STAGES)) & Int32(1)
                            _mbar_wait(bar_full + slot * Int32(8), phase)
                            cur_a = ring + slot * Int32(self.group_rows * 256 + 8192)
                            cur_b = cur_a + Int32(self.group_rows * 256)
                            w_k = ld_shared_i32_relaxed(meta + slot * Int32(8))
                            scale = ld_shared_i32_relaxed(meta + slot * Int32(8) + Int32(4)).to(Uint32)
                            if w_k == Int32(2):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 2, scale, num_live_m)
                            elif w_k == Int32(3):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 3, scale, num_live_m)
                            elif w_k == Int32(4):
                                self.accumulate_fc2(total, cur_a, cur_b, lane, warp, 4, scale, num_live_m)
                            _mbar_arrive(bar_empty + slot * Int32(8))
                            sequence += Int32(1)

                        out_meta_b = bind_loop_i32(out_meta, b_idx)
                        col_in_block = warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                        block_col = (slice_base + b_out) * Int32(128) + col_in_block
                        block_out_base = fc2_addr + (
                            ((b_out * max_slot).to(Int64) * Int64(128) + col_in_block.to(Int64))
                            if sliced
                            else ((group * Int32(self.group_rows)).to(Int64) * Int64(self.hidden_size) + block_col.to(Int64))
                        ) * Int64(2 if self.FC2_HALF else 4)
                        row_stride_bytes = Int64((128 if sliced else self.hidden_size) * (2 if self.FC2_HALF else 4))
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta_b >> Int32(idx * 2)) & Int32(3)
                                src_word = slot_hi if (cutlass.const_expr(self.mtiles == 4) and (m >= Int32(2))) else slot_lo
                                src_base = (m & Int32(1)) * Int32(16) + (lane >> Int32(2))
                                for v in cutlass.range_constexpr(2):
                                    slot_out = cute.arch.shuffle_sync(src_word, src_base + Int32(v * 8))
                                    row_idx = slot_out if sliced else (m * Int32(16) + Int32(v * 8) + (lane >> Int32(2)))
                                    if cutlass.const_expr(self.FC2_HALF):
                                        lo = _f32x2_to_f16x2(total[idx, 2 * v], total[idx, 2 * v + 1])
                                        hi = _f32x2_to_f16x2(total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
                                        base_lane = lane & Int32(28)
                                        l0 = cute.arch.shuffle_sync(lo, base_lane)
                                        l1 = cute.arch.shuffle_sync(lo, base_lane + Int32(1))
                                        l2 = cute.arch.shuffle_sync(lo, base_lane + Int32(2))
                                        l3 = cute.arch.shuffle_sync(lo, base_lane + Int32(3))
                                        h0 = cute.arch.shuffle_sync(hi, base_lane)
                                        h1 = cute.arch.shuffle_sync(hi, base_lane + Int32(1))
                                        h2 = cute.arch.shuffle_sync(hi, base_lane + Int32(2))
                                        h3 = cute.arch.shuffle_sync(hi, base_lane + Int32(3))
                                        if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0) and (lane & Int32(3)) == Int32(0):
                                            row_addr = block_out_base + row_idx.to(Int64) * row_stride_bytes
                                            st_global_v4_u32(row_addr, l0, l1, l2, l3)
                                            st_global_v4_u32(row_addr + Int64(16), h0, h1, h2, h3)
                                    else:
                                        if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                            row_addr = block_out_base + row_idx.to(Int64) * row_stride_bytes
                                            st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                            st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
            cute.arch.sync_threads()
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
        step = Int32(cute.arch.grid_dim()[0])
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
        sliced = slice_blocks < Int32(self.blocks)
        group_tasks = groups * (Int32(3) if sliced else Int32(chunks_per_group))
        use_chunks = group_tasks >= step
        limit_tasks = group_tasks if use_chunks else (groups * slice_blocks)
        b_count = (
            ((Int32(1) << slice_shift) if sliced else Int32(b_chunk_const))
            if use_chunks else Int32(1)
        )
        max_slot = rows * Int32(self.top_k)

        total = cute.make_rmem_tensor((self.mtiles, 8), Float32)

        while task < limit_tasks:
            group = Int32(0)
            b_local_base = Int32(0)
            if cutlass.const_expr(self.blocks == 48):
                if sliced:
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
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        tid = Int32(cute.arch.thread_idx()[0])
        cta = Int32(cute.arch.block_idx()[0])
        grid = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        task = cta * Int32(8) + warp
        step = grid * Int32(8)
        fc2_address = fc2.iterator.toint()
        signs_address = signs_out.iterator.toint()
        output_address = output.iterator.toint()
        max_slot = rows * Int32(self.top_k)
        sliced = slice_blocks < Int32(self.blocks)

        while task < rows * slice_blocks:
            row = Int32(0)
            b_out = Int32(0)
            if cutlass.const_expr(self.blocks == 48):
                if sliced:
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
                    for slot in cutlass.range_constexpr(self.top_k):
                        route = row_route_base + Int64(slot)
                        weight = route_weights[route]
                        if cutlass.const_expr(self.FC2_HALF):
                            lo, hi = ld_global_nc_v2_u32(row_fc2_base + Int64(slot * 256))
                            w0, w1 = f16x2_to_f32x2(lo)
                            w2, w3 = f16x2_to_f32x2(hi)
                        else:
                            w0, w1, w2, w3 = ld_global_lu_v4_f32_offset(row_fc2_base, slot * 512)
                        values[0] = values[0] + weight * w0
                        values[1] = values[1] + weight * w1
                        values[2] = values[2] + weight * w2
                        values[3] = values[3] + weight * w3
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
            signs = ld_global_nc_u32(signs_address + col.to(Int64))
            store_bf16x4(output_address + (row.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(2),
                         h0 * signed_byte(signs, 0), h1 * signed_byte(signs, 1),
                         h2 * signed_byte(signs, 2), h3 * signed_byte(signs, 3))
            task += step
