"""Dense page-table traversal and BF16/compact-MLA record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    cvt_e4m3x4_to_f32x4,
    get_ptr_as_int64,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_u32,
    st_shared_v4_u32,
)


@dsl_user_op
def _select_int32_lt(
    tok: Int32, limit: Int32, if_true: Int32, if_false: Int32, *, loc=None, ip=None
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(tok).ir_value(loc=loc, ip=ip),
                Int32(limit).ir_value(loc=loc, ip=ip),
                Int32(if_true).ir_value(loc=loc, ip=ip),
                Int32(if_false).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .pred %p;\n\t"
            "setp.lt.s32 %p, $1, $2;\n\t"
            "selp.b32 $0, $3, $4, %p;\n\t"
            "}",
            "=r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _select_int64_lt(
    tok: Int32, limit: Int32, if_true: Int64, if_false: Int64, *, loc=None, ip=None
) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Int32(tok).ir_value(loc=loc, ip=ip),
                Int32(limit).ir_value(loc=loc, ip=ip),
                Int64(if_true).ir_value(loc=loc, ip=ip),
                Int64(if_false).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .pred %p;\n\t"
            "setp.lt.s32 %p, $1, $2;\n\t"
            "selp.b64 $0, $3, $4, %p;\n\t"
            "}",
            "=l,r,r,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _anchor_prefetch_record(
    o0: Uint32,
    o1: Uint32,
    o2: Uint32,
    o3: Uint32,
    o4: Uint32,
    o5: Uint32,
    o6: Uint32,
    o7: Uint32,
    r0: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32, Uint32, Uint32, Uint32, Uint32, Uint32, Uint32]:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 9),
        [
            Uint32(o0).ir_value(loc=loc, ip=ip),
            Uint32(o1).ir_value(loc=loc, ip=ip),
            Uint32(o2).ir_value(loc=loc, ip=ip),
            Uint32(o3).ir_value(loc=loc, ip=ip),
            Uint32(o4).ir_value(loc=loc, ip=ip),
            Uint32(o5).ir_value(loc=loc, ip=ip),
            Uint32(o6).ir_value(loc=loc, ip=ip),
            Uint32(o7).ir_value(loc=loc, ip=ip),
            Uint32(r0).ir_value(loc=loc, ip=ip),
        ],
        "",
        "=r,=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,8",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(9)
    )


@cute.jit
def issue_dense_page_gather(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    record_stride_bytes_global: Int64,
    page_table_stride: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
    candidates_per_chunk: cutlass.Constexpr,
    io_threads: cutlass.Constexpr,
):
    """Stage one dense window with one bulk transaction per record.

    Every pool-scaled product is promoted before multiplication. Invalid tail
    rows copy physical record zero and are masked by the consumer; issuing a
    fixed byte count keeps the full-barrier protocol independent of live
    sequence length and therefore graph-replay stable.

    Lane zero arms the barrier with the chunk's exact byte count and each lane
    issues ``candidates_per_chunk // io_threads`` bulk copies, so the whole
    window moves with one instruction per record and the barrier completes on
    the arriving bytes rather than on a per-lane wait.  The caller therefore
    initializes the full barrier with a single arrival.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(candidates_per_chunk * record_bytes),
        )

    entry = io_lane
    for _ in cutlass.range_constexpr(candidates_per_chunk // io_threads):
        logical_token = token_begin + entry
        safe_token = logical_token
        if logical_token >= token_end:
            safe_token = Int32(0)
        logical_page = safe_token // Int32(page_size)
        in_page = safe_token - logical_page * Int32(page_size)
        physical_page = Int32(
            page_table[request.to(Int64) * page_table_stride + logical_page.to(Int64)]
        )

        # Load-bearing Int64 conversions. Neither multiplication is allowed to
        # occur in Int32, even when benchmark page ids happen to be small.
        source_offset = (
            physical_page.to(Int64) * page_stride_bytes
            + in_page.to(Int64) * record_stride_bytes_global
        )
        cp_async_bulk_g2s_mbar(
            kv_dst_addr + entry * Int32(record_stride_bytes),
            get_ptr_as_int64(cache_bytes, source_offset),
            Int32(record_bytes),
            full_mbar_addr,
        )
        entry += Int32(io_threads)


@cute.jit
def issue_ds_mla_page_gather(
    cache_bytes: cute.Tensor, page_table: cute.Tensor, kv_dst_addr: Int32,
    full_mbar_ptr, request: Int32, token_begin: Int32, token_end: Int32,
    io_lane: Int32, page_stride_bytes: Int64, record_stride_bytes_global: Int64,
    page_table_stride: Int64, *, page_size: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr, candidates_per_chunk: cutlass.Constexpr,
):
    """Read GLM's 656-byte record directly into the existing BF16 math tile.

    Only the current shared-memory window is widened; no global BF16 cache or
    query conversion is needed. Every producer lane publishes its own stores.
    """
    for entry in cutlass.range(candidates_per_chunk, unroll=1):
        token = token_begin + Int32(entry)
        if token >= token_end:
            token = Int32(0)
        page = token // Int32(page_size)
        physical = Int32(page_table[request.to(Int64) * page_table_stride + page.to(Int64)])
        source = get_ptr_as_int64(cache_bytes, physical.to(Int64) * page_stride_bytes
            + (token - page * Int32(page_size)).to(Int64) * record_stride_bytes_global)
        words = cute.make_ptr(Uint32, source, cute.AddressSpace.gmem, assumed_align=4)
        scales = cute.make_ptr(Float32, source + Int64(512), cute.AddressSpace.gmem, assumed_align=4)
        destination = kv_dst_addr + Int32(entry) * Int32(record_stride_bytes)
        for j in cutlass.range_constexpr(4):
            group = Int32(j)
            packed = words[group * Int32(32) + io_lane]
            a, b, c, d = cvt_e4m3x4_to_f32x4(packed)
            scale = Float32(scales[group])
            offset = destination + (group * Int32(128) + io_lane * Int32(4)) * Int32(2)
            st_shared_u32(offset, pack_f32x2_to_bfloat2(a * scale, b * scale))
            st_shared_u32(offset + Int32(4), pack_f32x2_to_bfloat2(c * scale, d * scale))
        st_shared_u32(destination + Int32(1024) + io_lane * Int32(4), words[Int32(132) + io_lane])
    cute.arch.mbarrier_arrive(full_mbar_ptr)


@cute.jit
def issue_ds_mla_page_gather_pipelined(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    empty_mbar_ptr,
    producer_phase: Int32,
    full_mbar_ptr,
    row_table_offset: Int64,
    zero_source: Int64,
    token_begin: Int32,
    token_end: Int32,
    io_warp: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    record_stride_bytes_global: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
    candidates_per_chunk: cutlass.Constexpr,
    io_warps: cutlass.Constexpr,
):
    """Multi-warp vectorized DS-MLA gather with register-stage prefetch before mbarrier_wait."""
    records_per_warp: cutlass.Constexpr = candidates_per_chunk // io_warps
    batches_per_warp: cutlass.Constexpr = records_per_warp // 4

    warp_entry_base = io_warp * Int32(records_per_warp)
    warp_token_begin = token_begin + warp_entry_base
    warp_dst_base = kv_dst_addr + warp_entry_base * Int32(record_stride_bytes)

    v4_lane_off = (io_lane * Int32(16)).to(Int64)
    scale_lane_off = Int64(512) + ((io_lane >> Int32(3)) * Int32(4)).to(Int64)
    rope_lane_off = Int64(528) + (io_lane * Int32(4)).to(Int64)

    smem_v4_lane_off = io_lane * Int32(32)
    smem_rope_lane_off = Int32(1024) + io_lane * Int32(4)

    prefetch_o = cute.make_rmem_tensor((records_per_warp * 8,), Uint32)
    prefetch_r = cute.make_rmem_tensor((records_per_warp,), Uint32)
    batch_w = cute.make_rmem_tensor((16,), Uint32)
    batch_s = cute.make_rmem_tensor((4,), Float32)
    batch_r = cute.make_rmem_tensor((4,), Uint32)

    warp_source_base = Int64(0)
    if cutlass.const_expr(
        page_size >= records_per_warp
        and page_size % records_per_warp == 0
        and candidates_per_chunk % records_per_warp == 0
    ):
        safe_warp_token = _select_int32_lt(
            warp_token_begin, token_end, warp_token_begin, Int32(0)
        )
        warp_page = safe_warp_token // Int32(page_size)
        warp_in_page = safe_warp_token - warp_page * Int32(page_size)
        warp_physical = Int32(page_table[row_table_offset + warp_page.to(Int64)])
        warp_source_base = get_ptr_as_int64(
            cache_bytes,
            warp_physical.to(Int64) * page_stride_bytes
            + warp_in_page.to(Int64) * record_stride_bytes_global,
        )

    for b in cutlass.range_constexpr(batches_per_warp):
        for sub in cutlass.range_constexpr(4):
            r: cutlass.Constexpr = b * 4 + sub
            rec_src = Int64(0)
            if cutlass.const_expr(
                page_size >= records_per_warp
                and page_size % records_per_warp == 0
                and candidates_per_chunk % records_per_warp == 0
            ):
                rec_src = _select_int64_lt(
                    warp_token_begin + Int32(r),
                    token_end,
                    warp_source_base + Int64(r) * record_stride_bytes_global,
                    zero_source,
                )
            else:
                tok = _select_int32_lt(
                    warp_token_begin + Int32(r),
                    token_end,
                    warp_token_begin + Int32(r),
                    Int32(0),
                )
                rec_page = tok // Int32(page_size)
                rec_in_page = tok - rec_page * Int32(page_size)
                rec_phys = Int32(page_table[row_table_offset + rec_page.to(Int64)])
                rec_src = get_ptr_as_int64(
                    cache_bytes,
                    rec_phys.to(Int64) * page_stride_bytes
                    + rec_in_page.to(Int64) * record_stride_bytes_global,
                )
            w0, w1, w2, w3 = ld_global_nc_v4_u32(rec_src + v4_lane_off)
            scale = ld_global_nc_f32(rec_src + scale_lane_off)
            r_val = ld_global_nc_u32(rec_src + rope_lane_off)
            batch_w[sub * 4 + 0] = w0
            batch_w[sub * 4 + 1] = w1
            batch_w[sub * 4 + 2] = w2
            batch_w[sub * 4 + 3] = w3
            batch_s[sub] = scale
            batch_r[sub] = r_val

        for sub in cutlass.range_constexpr(4):
            r: cutlass.Constexpr = b * 4 + sub
            scale = batch_s[sub]
            a0, b0, c0, d0 = cvt_e4m3x4_to_f32x4(batch_w[sub * 4 + 0])
            a1, b1, c1, d1 = cvt_e4m3x4_to_f32x4(batch_w[sub * 4 + 1])
            a2, b2, c2, d2 = cvt_e4m3x4_to_f32x4(batch_w[sub * 4 + 2])
            a3, b3, c3, d3 = cvt_e4m3x4_to_f32x4(batch_w[sub * 4 + 3])
            o0 = pack_f32x2_to_bfloat2(a0 * scale, b0 * scale)
            o1 = pack_f32x2_to_bfloat2(c0 * scale, d0 * scale)
            o2 = pack_f32x2_to_bfloat2(a1 * scale, b1 * scale)
            o3 = pack_f32x2_to_bfloat2(c1 * scale, d1 * scale)
            o4 = pack_f32x2_to_bfloat2(a2 * scale, b2 * scale)
            o5 = pack_f32x2_to_bfloat2(c2 * scale, d2 * scale)
            o6 = pack_f32x2_to_bfloat2(a3 * scale, b3 * scale)
            o7 = pack_f32x2_to_bfloat2(c3 * scale, d3 * scale)
            o0, o1, o2, o3, o4, o5, o6, o7, r_val = _anchor_prefetch_record(
                o0, o1, o2, o3, o4, o5, o6, o7, batch_r[sub]
            )
            prefetch_o[r * 8 + 0] = o0
            prefetch_o[r * 8 + 1] = o1
            prefetch_o[r * 8 + 2] = o2
            prefetch_o[r * 8 + 3] = o3
            prefetch_o[r * 8 + 4] = o4
            prefetch_o[r * 8 + 5] = o5
            prefetch_o[r * 8 + 6] = o6
            prefetch_o[r * 8 + 7] = o7
            prefetch_r[r] = r_val

    cute.arch.mbarrier_wait(empty_mbar_ptr, phase=producer_phase)

    for r in cutlass.range_constexpr(records_per_warp):
        dst = warp_dst_base + Int32(r * record_stride_bytes)
        st_shared_v4_u32(
            dst + smem_v4_lane_off,
            prefetch_o[r * 8 + 0],
            prefetch_o[r * 8 + 1],
            prefetch_o[r * 8 + 2],
            prefetch_o[r * 8 + 3],
        )
        st_shared_v4_u32(
            dst + smem_v4_lane_off + Int32(16),
            prefetch_o[r * 8 + 4],
            prefetch_o[r * 8 + 5],
            prefetch_o[r * 8 + 6],
            prefetch_o[r * 8 + 7],
        )
        st_shared_u32(dst + smem_rope_lane_off, prefetch_r[r])

    cute.arch.mbarrier_arrive(full_mbar_ptr)


__all__ = [
    "issue_dense_page_gather",
    "issue_ds_mla_page_gather",
    "issue_ds_mla_page_gather_pipelined",
]
