"""Device-driven EXLR8 grouping fused with the hidden-side ingress transform.

One launch per K1 wave: CTA 0 builds the expert groups; every other CTA turns
the wave's input rows into the rotated F16 activation all local experts read,
H128(x * signs.in) once per token, never per expert. Routed Spark waves read
the standard FP8 E4M3 + UE8M0 per-32 wire rows (the bytes EXL3 receives); the
coordinator shared expert reads BF16 rows through the same transform.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program
from b12x._lib.intrinsics import (
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    st_global_v2_u32,
    u32_as_f32,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad
from b12x.moe._shared.kernels.w4a16.exlr8 import signed_byte

INPUT_FORMATS = ("fp8_k32", "bf16")
# Ingress CTAs run eight warps, one (row, 128-column block) task per warp.
INGRESS_WARPS = 8


@dsl_user_op
def _e4m3x4_scaled(packed: Uint32, scale_bits: Uint32, *, loc=None, ip=None):
    """Four E4M3 bytes times an F32 scale (exact: E4M3 -> F16 -> F32 widening
    and a power-of-two mul.rn), as the EXL3 wire decoder computes them."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip), Uint32(scale_bits).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 b01, b23, l0, l1, l2, l3;
            .reg .b32 hi32, h01, h23;
            .reg .f32 sc;
            mov.b32 sc, $5;
            cvt.u16.u32 b01, $4;
            shr.b32 hi32, $4, 16;
            cvt.u16.u32 b23, hi32;
            cvt.rn.f16x2.e4m3x2 h01, b01;
            cvt.rn.f16x2.e4m3x2 h23, b23;
            mov.b32 {l0, l1}, h01;
            mov.b32 {l2, l3}, h23;
            cvt.f32.f16 $0, l0;
            cvt.f32.f16 $1, l1;
            cvt.f32.f16 $2, l2;
            cvt.f32.f16 $3, l3;
            mul.rn.f32 $0, $0, sc;
            mul.rn.f32 $1, $1, sc;
            mul.rn.f32 $2, $2, sc;
            mul.rn.f32 $3, $3, sc;
        }
        """,
        "=f,=f,=f,=f,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@cute.jit
def _bf16x4_to_f32(lo: Uint32, hi: Uint32):
    """Four BF16 values (two packed words) widened exactly to F32."""
    return (u32_as_f32(lo << Uint32(16)), u32_as_f32(lo & Uint32(0xFFFF0000)),
            u32_as_f32(hi << Uint32(16)), u32_as_f32(hi & Uint32(0xFFFF0000)))


@dsl_user_op
def _f32x2_to_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


class _Group:
    def __init__(self, experts, top_k, max_groups, hidden, input_format, max_rows=64, group_rows=16):
        self.experts, self.top_k, self.max_groups = experts, top_k, max_groups
        self.max_rows, self.group_rows = max_rows, group_rows
        self.hidden, self.blocks, self.input_format = hidden, hidden // 128, input_format
        # Wire rows: hidden E4M3 bytes, then hidden/32 UE8M0 scale bytes.
        self.row_bytes = hidden + hidden // 32 if input_format == "fp8_k32" else hidden * 2

    @cute.jit
    def __call__(self, ids: cute.Pointer, x: cute.Pointer, signs_in: cute.Pointer,
                 group_expert: cute.Pointer, route_slots: cute.Pointer,
                 route_group_lane: cute.Pointer, group_count: cute.Pointer,
                 activation: cute.Pointer, rows: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            ingress = (rows * Int32(self.blocks) + Int32(INGRESS_WARPS - 1)) // Int32(INGRESS_WARPS)
            self.kernel(ids, x, signs_in, group_expert, route_slots, route_group_lane,
                        group_count, activation, rows).launch(
                grid=(Int32(1) + min(ingress, cta_limit), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, x: cute.Pointer, signs_in: cute.Pointer,
               group_expert: cute.Pointer, route_slots: cute.Pointer,
               route_group_lane: cute.Pointer, group_count: cute.Pointer,
               activation: cute.Pointer, rows: Int32):
        smem = utils.SmemAllocator()
        prefix = smem.allocate_tensor(Int32, cute.make_layout(256), byte_alignment=16)
        valid = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        if Int32(cute.arch.block_idx()[0]) == Int32(0):
            self.pack(ids, group_expert, route_slots, route_group_lane, group_count,
                       prefix, valid, rows)
        else:
            self.ingress(x, signs_in, activation, rows)

    @cute.jit
    def ingress(self, x: cute.Pointer, signs_in: cute.Pointer, activation: cute.Pointer,
                rows: Int32):
        """activation[row, block] = F16(H128(dequant(x[row, block]) * signs.in)).

        One warp per (row, 128-column block); each lane owns four adjacent
        columns. Per-element arithmetic depends only on that row's input, so
        it is identical for every chunk placement, row count and CTA limit.
        """
        thread = Int32(cute.arch.thread_idx()[0])
        lane, warp = thread & Int32(31), thread >> Int32(5)
        task = (Int32(cute.arch.block_idx()[0]) - Int32(1)) * Int32(INGRESS_WARPS) + warp
        step = (Int32(cute.arch.grid_dim()[0]) - Int32(1)) * Int32(INGRESS_WARPS)
        source, out = Int64(x.toint()), Int64(activation.toint())
        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            row_base = source + row.to(Int64) * Int64(self.row_bytes)
            if cutlass.const_expr(self.input_format == "fp8_k32"):
                packed = ld_global_nc_u32(row_base + col.to(Int64))
                # The block's four UE8M0 bytes; lanes 8j..8j+7 share byte j.
                scales = ld_global_nc_u32(row_base + Int64(self.hidden) + (block * Int32(4)).to(Int64))
                s = ((scales >> ((lane >> Int32(3)) * Int32(8)).to(Uint32)) & Uint32(0xFF)).to(Int32)
                # float(UE8M0) = 2^(s - 127); 0 is the F32 subnormal 2^-127, 255 NaN.
                bits = Uint32(s) << Uint32(23)
                if s == Int32(0):
                    bits = Uint32(0x00400000)
                if s == Int32(255):
                    bits = Uint32(0x7FFFFFFF)
                v0, v1, v2, v3 = _e4m3x4_scaled(packed, bits)
            else:
                lo, hi = ld_global_nc_v2_u32(row_base + col.to(Int64) * Int64(2))
                v0, v1, v2, v3 = _bf16x4_to_f32(lo, hi)
            signs = ld_global_nc_u32(Int64(signs_in.toint()) + col.to(Int64))
            h0, h1, h2, h3 = _w4a8_had128_quad(v0 * signed_byte(signs, 0), v1 * signed_byte(signs, 1),
                                               v2 * signed_byte(signs, 2), v3 * signed_byte(signs, 3), lane)
            st_global_v2_u32(out + (row.to(Int64) * Int64(self.hidden) + col.to(Int64)) * Int64(2),
                             _f32x2_to_f16x2(h0, h1), _f32x2_to_f16x2(h2, h3))
            task += step

    @cute.jit
    def pack(self, ids: cute.Pointer, group_expert: cute.Pointer,
              route_slots: cute.Pointer, route_group_lane: cute.Pointer,
              group_count: cute.Pointer, prefix: cute.Tensor, valid: cute.Tensor,
              rows: Int32):
        """Stable ascending-expert/ascending-route packing (CTA 0 only)."""
        thread = Int32(cute.arch.thread_idx()[0])
        routes = rows * Int32(self.top_k)
        if thread == Int32(0):
            valid[0] = Int32(1)
            for route in cutlass.range(routes):
                if ids[route] < Int32(0) or ids[route] >= Int32(self.experts):
                    valid[0] = Int32(0)
        for route in cutlass.range(thread, routes, 256):
            route_group_lane[route] = Int32(-1)
        count = Int32(0)
        if thread < Int32(self.experts):
            for route in cutlass.range(routes):
                if ids[route] == thread:
                    count += Int32(1)
        groups = (count + Int32(self.group_rows - 1)) // Int32(self.group_rows)
        prefix[thread] = groups
        cute.arch.sync_threads()
        # Inclusive scan of per-expert group counts; neither atomics nor route
        # arrival order can change ascending-expert/ascending-route packing.
        for step in cutlass.range_constexpr(8):
            distance = Int32(1 << step)
            addend = Int32(0)
            if thread >= distance:
                addend = prefix[thread - distance]
            cute.arch.sync_threads()
            prefix[thread] = prefix[thread] + addend
            cute.arch.sync_threads()
        total = prefix[Int32(255)]
        if thread == Int32(0):
            group_count[0] = Int32(-1)
            if valid[0] != Int32(0) and total <= Int32(self.max_groups):
                group_count[0] = total
        # Loop-carried locals are typed before the branches (helper-function scope).
        group, packed, ordinal = Int32(0), Int32(0), Int32(0)
        if valid[0] != Int32(0) and total <= Int32(self.max_groups):
            first = prefix[thread] - groups
            for local in cutlass.range(groups):
                group = first + local
                group_expert[group] = thread
                for lane in cutlass.range_constexpr(self.group_rows):
                    route_slots[group * Int32(self.group_rows) + Int32(lane)] = Int32(-1)
            ordinal = Int32(0)
            if thread < Int32(self.experts):
                for route in cutlass.range(routes):
                    if ids[route] == thread:
                        packed = first * Int32(self.group_rows) + ordinal
                        route_slots[packed] = route
                        route_group_lane[route] = packed
                        ordinal += Int32(1)


@dsl_user_op
def _warp_match_count_and_rank(val: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(val).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 mask, lane, lower;
            mov.u32 lane, %laneid;
            shl.b32 lower, 1, lane;
            sub.u32 lower, lower, 1;
            match.any.sync.b32 mask, $2, 0xffffffff;
            popc.b32 $0, mask;
            and.b32 lower, mask, lower;
            popc.b32 $1, lower;
        }
        """,
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return (Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _warp_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "bar.warp.sync 0xffffffff;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class _GroupPrefill(_Group):
    @cute.kernel
    def kernel(self, ids: cute.Pointer, x: cute.Pointer, signs_in: cute.Pointer,
               group_expert: cute.Pointer, route_slots: cute.Pointer,
               route_group_lane: cute.Pointer, group_count: cute.Pointer,
               activation: cute.Pointer, rows: Int32):
        smem = utils.SmemAllocator()
        prefix = smem.allocate_tensor(Int32, cute.make_layout(256), byte_alignment=16)
        valid = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        s_counts = smem.allocate_tensor(Int32, cute.make_layout(8 * 256), byte_alignment=16)
        if Int32(cute.arch.block_idx()[0]) == Int32(0):
            self.pack(ids, group_expert, route_slots, route_group_lane, group_count,
                      prefix, valid, s_counts, rows)
        else:
            self.ingress(x, signs_in, activation, rows)

    @cute.jit
    def pack(self, ids: cute.Pointer, group_expert: cute.Pointer,
             route_slots: cute.Pointer, route_group_lane: cute.Pointer,
             group_count: cute.Pointer, prefix: cute.Tensor, valid: cute.Tensor,
             s_counts: cute.Tensor, rows: Int32):
        """Parallel warp-interval ascending-expert/ascending-route packing (CTA 0 only)."""
        thread = Int32(cute.arch.thread_idx()[0])
        lane, warp = thread & Int32(31), thread >> Int32(5)
        routes = rows * Int32(self.top_k)
        for i in cutlass.range_constexpr(8):
            s_counts[Int32(i * 256) + thread] = Int32(0)
        if thread == Int32(0):
            valid[0] = Int32(1)
        cute.arch.sync_threads()

        vec32_total = (routes + Int32(31)) >> Int32(5)
        vec32_per_warp = (vec32_total + Int32(7)) >> Int32(3)
        r_start = warp * (vec32_per_warp << Int32(5))
        r_end = min(routes, r_start + (vec32_per_warp << Int32(5)))

        r_base = r_start
        while r_base < r_end:
            r = r_base + lane
            eid = Int32(-1)
            if r < r_end:
                eid = ids[r]
                route_group_lane[r] = Int32(-1)
                if eid < Int32(0) or eid >= Int32(self.experts):
                    valid[0] = Int32(0)
                    eid = Int32(-1)
            cnt, rank = _warp_match_count_and_rank(eid)
            if eid >= Int32(0) and rank == Int32(0):
                s_counts[warp * Int32(256) + eid] = s_counts[warp * Int32(256) + eid] + cnt
            r_base += Int32(32)
        cute.arch.sync_threads()

        c0 = s_counts[thread]
        c1 = s_counts[Int32(256) + thread]
        c2 = s_counts[Int32(512) + thread]
        c3 = s_counts[Int32(768) + thread]
        c4 = s_counts[Int32(1024) + thread]
        c5 = s_counts[Int32(1280) + thread]
        c6 = s_counts[Int32(1536) + thread]
        c7 = s_counts[Int32(1792) + thread]
        count = Int32(0)
        if thread < Int32(self.experts):
            count = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
        groups = (count + Int32(self.group_rows - 1)) // Int32(self.group_rows)
        prefix[thread] = groups
        cute.arch.sync_threads()

        for step in cutlass.range_constexpr(8):
            distance = Int32(1 << step)
            addend = Int32(0)
            if thread >= distance:
                addend = prefix[thread - distance]
            cute.arch.sync_threads()
            prefix[thread] = prefix[thread] + addend
            cute.arch.sync_threads()

        total = prefix[Int32(255)]
        if thread == Int32(0):
            group_count[0] = Int32(-1)
            if valid[0] != Int32(0) and total <= Int32(self.max_groups):
                group_count[0] = total

        if valid[0] != Int32(0) and total <= Int32(self.max_groups):
            first = prefix[thread] - groups
            base_slot = first * Int32(self.group_rows)
            s_counts[thread] = base_slot
            s_counts[Int32(256) + thread] = base_slot + c0
            s_counts[Int32(512) + thread] = base_slot + c0 + c1
            s_counts[Int32(768) + thread] = base_slot + c0 + c1 + c2
            s_counts[Int32(1024) + thread] = base_slot + c0 + c1 + c2 + c3
            s_counts[Int32(1280) + thread] = base_slot + c0 + c1 + c2 + c3 + c4
            s_counts[Int32(1536) + thread] = base_slot + c0 + c1 + c2 + c3 + c4 + c5
            s_counts[Int32(1792) + thread] = base_slot + c0 + c1 + c2 + c3 + c4 + c5 + c6
            for local in cutlass.range(groups):
                group_expert[first + local] = thread
            for idx in cutlass.range(thread, total * Int32(self.group_rows), 256):
                route_slots[idx] = Int32(-1)
            cute.arch.sync_threads()

            r_base = r_start
            while r_base < r_end:
                r = r_base + lane
                eid = Int32(-1)
                if r < r_end:
                    eid = ids[r]
                cnt, rank = _warp_match_count_and_rank(eid)
                base_val = Int32(0)
                if eid >= Int32(0):
                    base_val = s_counts[warp * Int32(256) + eid]
                _warp_sync()
                if eid >= Int32(0):
                    if rank == Int32(0):
                        s_counts[warp * Int32(256) + eid] = base_val + cnt
                    packed = base_val + rank
                    route_slots[packed] = r
                    route_group_lane[r] = packed
                _warp_sync()
                r_base += Int32(32)


def compile_exlr8_group_aot(*, experts=256, top_k=8, max_groups=512, hidden=6144,
                            input_format="fp8_k32", shared_expert=False,
                            max_rows=64, group_rows=16):
    if max_rows not in (64, 4096) or group_rows not in (16, 32, 64):
        raise ValueError("EXLR8 grouping supports planned rows 64/4096 and group rows 16/32/64")
    if max_rows == 64 and group_rows != 16:
        raise ValueError("EXLR8 decode grouping requires 16-row groups")
    if shared_expert and ((experts, top_k, max_groups) != (1, 1, 4) or input_format != "bf16"):
        raise ValueError("shared EXLR8 grouping requires experts=1, top_k=1, max_groups=4 and BF16 rows")
    if (any(type(v) is not int for v in (experts, top_k, max_groups, hidden))
            or not (1 <= experts <= 256 and 1 <= top_k <= 32
                    and 0 < max_groups <= (2**31 - 1) // group_rows)):
        raise ValueError("EXLR8 grouping requires experts1..256, top_k1..32 and positive capacity")
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unknown EXLR8 ingress format {input_format!r}")
    # 32-bit column and per-chunk element indexing: a 64-row F16 chunk's bytes fit int32.
    if hidden <= 0 or hidden % 128 or 64 * hidden * 2 > 2**31 - 1:
        raise ValueError("EXLR8 ingress requires positive hidden % 128 == 0 within int32 indexing")
    row = f"[rows,{hidden + hidden // 32}]" if input_format == "fp8_k32" else f"[rows,{hidden}]"
    x = (Operand("x", torch.uint8, row, note="E4M3 bytes then UE8M0 per-32 scales per row")
         if input_format == "fp8_k32" else Operand("x", torch.bfloat16, row))
    metadata = {"group_rows": group_rows} if max_rows > 64 else {}
    group_cls = _GroupPrefill if max_rows > 64 else _Group
    return compile_program(group_cls(experts, top_k, max_groups, hidden, input_format, max_rows, group_rows),
        name=("exlr8_grouped_group" if max_rows > 64 else
              "exlr8_shared_group" if shared_expert else "exlr8_group"),
        operands=(Operand("ids", torch.int32, f"[rows,{top_k}]"),
                  x,
                  Operand("signs_in", torch.int8, f"[{hidden}]", align=4),
                  Operand("group_expert", torch.int32, f"[{max_groups}]", "out"),
                  Operand("route_slots", torch.int32, f"[{max_groups},{group_rows}]", "out"),
                  Operand("route_group_lane", torch.int32, f"[rows,{top_k}]", "out"),
                  Operand("group_count", torch.int32, "[1]", "out"),
                  Operand("activation", torch.float16, f"[rows,{hidden}]", "out")),
        scalars=(Scalar("rows"), Scalar("cta_limit")),
        key=(experts, top_k, max_groups, hidden, input_format, shared_expert, max_rows, group_rows),
        geometry=dict(abi_version=2, experts=experts, top_k=top_k, max_groups=max_groups,
                      max_rows=max_rows, hidden=hidden, input_format=input_format, **metadata,
                      input_row_bytes=_Group(experts, top_k, max_groups, hidden, input_format).row_bytes,
                      ctas_per_sm=4, ingress_warps=INGRESS_WARPS))


def exlr8_export_programs(role="all", *, hidden=6144, units=16, top_k=8, max_groups=512):
    """Exporter tuples and role ownership; callers must not duplicate this list."""
    from .exlr8 import compile_exlr8_a16_aot
    if role not in ("all", "coordinator", "spark"):
        raise ValueError(f"unknown EXLR8 export role {role!r}")
    result = [
        ("a16_decode", "exlr8_a16", {"max_rows": 64},
         lambda: compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=top_k, max_groups=max_groups)),
        ("a16_serving", "exlr8_a16", {"max_rows": 64},
         lambda: compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=top_k,
                                      max_groups=max_groups, device_count=True)),
        ("group", "exlr8_group", {"max_rows": 64},
         lambda: compile_exlr8_group_aot(top_k=top_k, max_groups=max_groups, hidden=hidden)),
        ("shared_a16_serving", "exlr8_a16", {"max_rows": 64},
         lambda: compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=1,
                                      max_groups=4, device_count=True, shared_expert=True)),
        ("shared_group", "exlr8_group", {"max_rows": 64},
         lambda: compile_exlr8_group_aot(experts=1, top_k=1, max_groups=4, hidden=hidden,
                                         input_format="bf16", shared_expert=True)),
    ]
    grouped = [
        ("a16_grouped", "exlr8_a16", {"max_rows": 4096},
         lambda: compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=top_k,
             max_groups=(4096 * top_k + 63)//64 + 256, device_count=True, grouped=True)),
        ("grouped_group", "exlr8_group", {"max_rows": 4096},
         lambda: compile_exlr8_group_aot(hidden=hidden, top_k=top_k,
             max_groups=(4096 * top_k + 63)//64 + 256, max_rows=4096, group_rows=64)),
    ]
    if role == "spark":
        return result[1:3] + grouped
    if role == "coordinator":
        return result[3:]
    return result + grouped
