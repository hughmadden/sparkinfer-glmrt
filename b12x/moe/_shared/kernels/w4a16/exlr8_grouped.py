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
    CODE_VERSION = 16
    CTAS_PER_SM = 2

    def __init__(self, hidden_size, units_per_expert, top_k, max_groups, group_rows=64):
        super().__init__(hidden_size, units_per_expert, top_k, max_groups, device_count=True)
        self.group_rows = group_rows
        self.mtiles = group_rows // 16
        if group_rows not in (16, 32, 64):
            raise ValueError("EXLR8 grouped group_rows must be 16, 32 or 64")
        self.shared_memory_bytes = query_optin_shared_memory()
        required = group_rows * 256 + 8192 + max(1, self.mtiles - 1) * 8192
        if self.shared_memory_bytes is not None and required > self.shared_memory_bytes:
            raise ValueError(f"EXLR8 grouped needs {required} B shared memory, above "
                             f"the device's {self.shared_memory_bytes} B opt-in limit")

    @property
    def __cache_key__(self):
        return ("exlr8_a16_grouped", self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.group_rows, self.CTAS_PER_SM)

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
            self.grouped_fc1_kernel(x, pointers, bits, unit_count, unit_map,
                                    group_expert, route_slots, fc1, rows,
                                    group_count, unit_limit).launch(
                grid=(min(Int32(self.max_groups) * unit_limit * Int32(2), cta_limit), 1, 1),
                block=(256, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.activation_kernel(pointers, bits, unit_count, unit_map,
                                   group_expert, route_slots, fc1, intermediate,
                                   rows, group_count, unit_limit).launch(
                grid=(min(Int32(self.max_groups) * unit_limit, cta_limit), 1, 1),
                block=(256, 1, 1), stream=stream)
            self.grouped_fc2_kernel(pointers, bits, unit_count, unit_map,
                                    group_expert, route_slots, intermediate, fc2,
                                    rows, group_count, unit_limit).launch(
                grid=(min(Int32(self.max_groups * self.blocks), cta_limit), 1, 1),
                block=(256, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                               rows, group_count).launch(
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
        col_offset_bytes = (input_block.to(Int64) * Int64(128) + col_s.to(Int64) * Int64(8)) * Int64(2)
        base_a = a_address + col_offset_bytes
        for idx in cutlass.range_constexpr(max_tiles):
            if Int32(idx) < num_live_m:
                sr = srow[idx]
                valid = (sr >= Int32(0)).to(Int32)
                safe_sr = max(sr, Int32(0))
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
                            out_off = (task.to(Int64) * Int64(self.group_rows) + row.to(Int64)) * Int64(128) + lane.to(Int64) * Int64(4)
                            st_global_v2_u32(inter_addr + out_off * Int64(2),
                                             _f32x2_to_f16x2(h0, h1), _f32x2_to_f16x2(h2, h3))
            task += step

    @cute.kernel
    def grouped_fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           unit_count: cute.Tensor, unit_map: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, group_count: cute.Tensor, unit_limit: Int32):
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

        b_chunk_const = 8 if (self.blocks % 8 == 0) else (3 if (self.blocks % 3 == 0) else self.blocks)
        chunks_per_group = self.blocks // b_chunk_const
        use_chunks = (groups * Int32(chunks_per_group)) >= step
        limit_tasks = (groups * Int32(chunks_per_group)) if use_chunks else (groups * Int32(self.blocks))
        b_count = Int32(b_chunk_const) if use_chunks else Int32(1)

        total = cute.make_rmem_tensor((self.mtiles, 8), Float32)

        while task < limit_tasks:
            group = Int32(0)
            block_base = Int32(0)
            if use_chunks:
                g_tile_base = (task // Int32(chunks_per_group * 4)) * Int32(4)
                tile_g = min(Int32(4), groups - g_tile_base)
                rem = task - g_tile_base * Int32(chunks_per_group)
                group = g_tile_base + (rem % tile_g)
                block_base = (rem // tile_g) * Int32(b_chunk_const)
            else:
                g_tile_base = (task // Int32(self.blocks * 4)) * Int32(4)
                tile_g = min(Int32(4), groups - g_tile_base)
                rem = task - g_tile_base * Int32(self.blocks)
                group = g_tile_base + (rem % tile_g)
                block_base = rem // tile_g

            expert = group_expert[group]
            num_live_m, live_m, srow = self.prepare_group_tiles(
                route_slots, group, rows, tid, lane, shared_a, smem_thread_off, True)

            if num_live_m > Int32(0):
                group_inter_base = inter_addr + group.to(Int64) * (
                    unit_limit.to(Int64) * Int64(self.group_rows * 128 * 2))
                owned = unit_count[expert]
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

                if owned > Int32(0):
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
                    pin_all_a = owned <= max_a_slots

                    w0, pay0, sc0, ua0 = self.get_fc2_unit_desc(
                        pointers, bits, unit_map, expert, Int32(0), group_inter_base)
                    self.stage_a_async(shared_a, ua0, srow, num_live_m,
                                       Int32(0), smem_thread_off, col_s, 128, self.mtiles)
                    if pin_all_a:
                        for u in cutlass.range(Int32(1), owned):
                            self.zero_invalid_a_tiles(
                                shared_a + u * a_slot_stride, srow, num_live_m, smem_thread_off)
                            _, _, _, u_addr = self.get_fc2_unit_desc(
                                pointers, bits, unit_map, expert, u, group_inter_base)
                            self.stage_a_async(shared_a + u * a_slot_stride, u_addr, srow,
                                               num_live_m, Int32(0), smem_thread_off, col_s, 128, self.mtiles)
                    else:
                        self.zero_invalid_a_tiles(
                            shared_a + a_slot_stride, srow, num_live_m, smem_thread_off)

                    self.stage_b_async_dyn(shared_b0, pay0, block_base, tid, w0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                    packed_scale = ld_global_b16(sc0 + block_base.to(Int64) * Int64(2)) if w0 != Int32(0) else Uint32(0)
                    cur_w = w0
                    step_parity = Int32(0)

                    for b_idx in cutlass.range(b_count):
                        block = block_base + b_idx
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
                                nw, npay, nsc, nua = self.get_fc2_unit_desc(
                                    pointers, bits, unit_map, expert, next_k, group_inter_base)
                                cur_w = nw
                                packed_scale = (
                                    ld_global_b16(nsc + next_block.to(Int64) * Int64(2))
                                    if nw != Int32(0) else Uint32(0)
                                )
                                self.stage_b_async_dyn(next_b, npay, next_block, tid, nw)
                                if not pin_all_a:
                                    self.stage_a_async(
                                        shared_a + next_parity * a_slot_stride, nua, srow,
                                        num_live_m, Int32(0), smem_thread_off, col_s, 128, self.mtiles)
                                cute.arch.cp_async_commit_group()

                            if w_k == Int32(2):
                                self.accumulate(total, cur_a, cur_b, lane, warp, 2, cur_scale, num_live_m)
                            elif w_k == Int32(3):
                                self.accumulate(total, cur_a, cur_b, lane, warp, 3, cur_scale, num_live_m)
                            elif w_k == Int32(4):
                                self.accumulate(total, cur_a, cur_b, lane, warp, 4, cur_scale, num_live_m)

                            if has_next:
                                cute.arch.cp_async_wait_group(0)
                                cute.arch.sync_threads()

                            step_parity += Int32(1)

                        out_meta_b = bind_loop_i32(out_meta, b_idx)
                        col0 = block * Int32(128) + warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                        block_out_base = fc2_addr + (
                            group_base * Int64(self.hidden_size) + col0.to(Int64)
                        ) * Int64(4)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta_b >> Int32(idx * 2)) & Int32(3)
                                for v in cutlass.range_constexpr(2):
                                    if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                        local = m * Int32(16) + Int32(v * 8) + (lane >> Int32(2))
                                        row_addr = block_out_base + local.to(Int64) * Int64(self.hidden_size * 4)
                                        st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                        st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
                else:
                    total.fill(Float32(0.0))
                    for b_idx in cutlass.range(b_count):
                        out_meta_b = bind_loop_i32(out_meta, b_idx)
                        block = block_base + b_idx
                        col0 = block * Int32(128) + warp * Int32(16) + (lane & Int32(3)) * Int32(2)
                        block_out_base = fc2_addr + (
                            group_base * Int64(self.hidden_size) + col0.to(Int64)
                        ) * Int64(4)
                        for idx in cutlass.range_constexpr(self.mtiles):
                            if Int32(idx) < num_live_m:
                                m = (out_meta_b >> Int32(idx * 2)) & Int32(3)
                                for v in cutlass.range_constexpr(2):
                                    if (out_meta_b & Int32(1 << (8 + idx * 2 + v))) != Int32(0):
                                        local = m * Int32(16) + Int32(v * 8) + (lane >> Int32(2))
                                        row_addr = block_out_base + local.to(Int64) * Int64(self.hidden_size * 4)
                                        st_global_v2_f32(row_addr, total[idx, 2 * v], total[idx, 2 * v + 1])
                                        st_global_v2_f32(row_addr + Int64(32), total[idx, 4 + 2 * v], total[idx, 4 + 2 * v + 1])
            cute.arch.sync_threads()
            task += step

    @cute.kernel
    def reduce_kernel(self, route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                      fc2: cute.Tensor, signs_out: cute.Tensor, output: cute.Tensor,
                      rows: Int32, group_count: cute.Tensor):
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
        max_packed = groups * Int32(self.group_rows)

        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0.0))
            row_route_base = row.to(Int64) * Int64(self.top_k)
            for slot in cutlass.range_constexpr(self.top_k):
                route = row_route_base + Int64(slot)
                packed = route_group_lane[route]
                if packed >= Int32(0) and packed < max_packed:
                    weight = route_weights[route]
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(
                        fc2_address + (packed.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4))
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
