"""SM89 (Ada) K32 block-scaled FP8 projections.

The pre-Hopper build of the MXFP8 K32 contract used by DeepSeek V4.1's
coordinator projections: E4M3 activations with UE8M0 row scales per 32
values, E4M3 weights with UE8M0 scales per 32x32 block, BF16 output. Every
K32 step runs one ``mma.m16n8k32`` E4M3 MMA into a zero FP32 partial that is
multiplied by ``2^(ea + eb - 254)`` and added to the FP32 accumulator
(exact: power-of-two scales, IEEE add), so results differ from the SM120
block-scaled MMA only by FP32 accumulation order.

Scales stay in plain layouts on SM89:

* activation row scales ``[L, M, K/32]`` (live ``M``), as the row quantizer
  writes ``scale_rows``;
* weight scales ``[L, N/32, K/32]``, the checkpoint layout (no packing).

Two kernels share one pointer ABI (``_DenseGemmLaunch``'s, so native wrappers
are unchanged): ``Sm89Block32Gemv`` for decode and verify capacities (rows up
to 80, weights streamed once) and the cp.async ``DenseGemmKernel`` block-32
path for prefill. ``compile_sm89_block32_gemm_aot`` picks one per capacity.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    ld_global_nc_u32,
    ld_global_nc_u32_pred,
    mma_m16n8k32_f32_e4m3,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr, runtime_num_sm

# Decode and verify capacities stream the weight once through the GEMV.
GEMV_MAX_ROWS = 80


@dsl_user_op
def _ld_weight_v2(address, *, loc=None, ip=None):
    """8-byte weight load, read once: no L1 allocation, 128-byte L2 fetch so
    a quad's four 8-byte pieces of one row cost one DRAM burst."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.L2::128B.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _ld_act_v2_pred(address, pred, *, loc=None, ip=None):
    """8-byte activation load through L1 (re-read by every CTA); zero when
    ``pred`` is false (rows past live M)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; mov.u32 $0, 0; mov.u32 $1, 0; "
        "@p ld.global.nc.v2.u32 {$0, $1}, [$2]; }",
        "=r,=r,l,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@cute.jit
def _scale_byte(word: Uint32, step: int) -> Int32:
    return Int32((word >> Uint32(8 * step)) & Uint32(0xFF))


@cute.jit
def _pow2_biased(e: Int32) -> Float32:
    """2^(e - 127) for a biased exponent ``e`` in [1, 254]."""
    return u32_as_f32(Uint32(e) << Uint32(23))


class Sm89Block32Gemv:
    """K32 block-scaled W8A8 projection for up to ``16 * m_tiles`` rows.

    A CTA owns ``8 * col_groups`` output columns of one group (all inside one
    32-wide weight scale block) and every row tile; its warps take 128-wide K
    blocks round-robin and fold through shared memory in a fixed order.

    ``slices`` > 1 splits K across CTAs (grid z): each slice sums a contiguous
    run of 128-wide K blocks and writes FP32 partial planes [slices, M, N] for
    a fixed-order second pass (the native ``reduce_splits``); one slice
    stores BF16 directly. A row's arithmetic never depends on M or the row
    tiles, so every capacity built with the same ``(warps, slices)`` gives
    bitwise-identical rows.

    MMA k-slot permutation inside each K32 step: lane ``j`` of a quad reads
    the eight contiguous bytes ``[8j, 8j + 8)`` and feeds bytes 0-3 to slots
    ``4j..4j+3`` and bytes 4-7 to slots ``16+4j..``, for activations and
    weights alike, so every MMA sums exactly one 32-wide K block.
    """

    def __init__(self, n: int, k: int, *, groups: int = 1, max_rows: int = 16,
                 warps: int = 8, col_groups: int = 1, slices: int = 1):
        self.n, self.k, self.groups = int(n), int(k), int(groups)
        self.slices = int(slices)
        if self.slices not in (1, 2, 4) or (self.slices > 1 and self.groups != 1):
            raise ValueError("SM89 block-32 GEMV splits K 2 or 4 ways, ungrouped only")
        self.m_tiles = (int(max_rows) + 15) // 16
        self.col_groups = int(col_groups)
        self.cols = 8 * self.col_groups
        self.warps = int(warps)
        if 32 % self.cols or self.n % self.cols or self.n % 32 or self.k % 128:
            raise ValueError("SM89 block-32 GEMV needs N % 32 == 0, 8*col_groups dividing 32 "
                             "and K % 128 == 0")
        if not 1 <= self.m_tiles <= 5:
            raise ValueError("SM89 block-32 GEMV covers 1..80 rows")
        self.blocks = self.k // 128
        if self.blocks < self.slices:
            raise ValueError("SM89 block-32 GEMV needs a K block per slice")
        # Slice s covers blocks [s*B//S, (s+1)*B//S); warps take them round-robin.
        slice_blocks = -(-self.blocks // self.slices)
        self.trips = (slice_blocks + self.warps - 1) // self.warps
        self.tile_frags = self.col_groups * 4
        # Prefetch whole blocks (weights, activations, scales) while registers allow.
        self.prefetch_all = self.m_tiles <= 2

    def key(self) -> tuple:
        return ("sm89_block32_gemv", self.n, self.k, self.groups, self.m_tiles,
                self.col_groups, self.warps, self.slices)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[
                cute.struct.MemRange[Float32, self.warps * self.tile_frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, a_ptr: cute.Pointer, b_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
                 sfb_ptr: cute.Pointer, c_ptr: cute.Pointer, quant_c_values_ptr: cute.Pointer,
                 quant_c_scale_rows_ptr: cute.Pointer, quant_c_scale_mma_ptr: cute.Pointer,
                 alpha_ptr: cute.Pointer, m: Int32, current_stream: cuda.CUstream):
        self.kernel(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, m).launch(
            grid=(self.n // self.cols, self.groups, self.slices), block=(32 * self.warps, 1, 1),
            stream=current_stream)

    @cute.jit
    def _load_weights(self, dest: cute.Tensor, w_rows: cute.Tensor, block: Int32):
        k_byte = Int64(block) * Int64(128)
        for gi in cutlass.range_constexpr(self.col_groups):
            for s in cutlass.range_constexpr(4):
                lo, hi = _ld_weight_v2(w_rows[gi] + k_byte + Int64(32 * s))
                dest[(gi * 4 + s) * 2] = lo
                dest[(gi * 4 + s) * 2 + 1] = hi

    @cute.jit
    def _load_operands(self, w_dst: cute.Tensor, a_dst: cute.Tensor, s_dst: cute.Tensor,
                       w_rows: cute.Tensor, a_base: Int64, sa_base: Int64, sb_base: Int64,
                       rows_live: cute.Tensor, g: Int32, block: Int32):
        """Everything one 128-wide K block needs: weights, activation
        fragments, the weight scale word and every row's scale word."""
        self._load_weights(w_dst, w_rows, block)
        k32 = self.k // 32
        s_dst[0] = ld_global_nc_u32(sb_base + Int64(block) * Int64(4))
        k_byte = Int64(block) * Int64(128)
        for mt in cutlass.range_constexpr(self.m_tiles):
            for half in cutlass.range_constexpr(2):
                row = g + Int32(16 * mt + 8 * half)
                s_dst[1 + 2 * mt + half] = ld_global_nc_u32_pred(
                    sa_base + Int64(row) * Int64(k32) + Int64(block) * Int64(4),
                    rows_live[2 * mt + half])
            for st in cutlass.range_constexpr(4):
                at = a_base + Int64(g + Int32(16 * mt)) * Int64(self.k) + k_byte + Int64(32 * st)
                f = (st * self.m_tiles + mt) * 4
                a_dst[f], a_dst[f + 2] = _ld_act_v2_pred(at, rows_live[2 * mt])
                a_dst[f + 1], a_dst[f + 3] = _ld_act_v2_pred(at + Int64(8 * self.k),
                                                             rows_live[2 * mt + 1])

    @cute.jit
    def _compute_block(self, acc: cute.Tensor, w: cute.Tensor, act: cute.Tensor,
                       words: cute.Tensor):
        for st in cutlass.range_constexpr(4):
            e_b = _scale_byte(words[0], st) - Int32(127)
            for mt in cutlass.range_constexpr(self.m_tiles):
                f = (st * self.m_tiles + mt) * 4
                scale_lo = _pow2_biased(_scale_byte(words[1 + 2 * mt], st) + e_b)
                scale_hi = _pow2_biased(_scale_byte(words[2 + 2 * mt], st) + e_b)
                for gi in cutlass.range_constexpr(self.col_groups):
                    zero = Float32(0.0)
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        zero, zero, zero, zero, act[f], act[f + 1], act[f + 2], act[f + 3],
                        w[(gi * 4 + st) * 2], w[(gi * 4 + st) * 2 + 1])
                    o = (mt * self.col_groups + gi) * 4
                    acc[o] = acc[o] + d0 * scale_lo
                    acc[o + 1] = acc[o + 1] + d1 * scale_lo
                    acc[o + 2] = acc[o + 2] + d2 * scale_hi
                    acc[o + 3] = acc[o + 3] + d3 * scale_hi

    @cute.kernel
    def kernel(self, a_ptr: cute.Pointer, b_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
               sfb_ptr: cute.Pointer, c_ptr: cute.Pointer, m: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        cta_n, group, part = cute.arch.block_idx()
        n0 = Int64(cta_n) * Int64(self.cols)
        block_begin = Int32(part) * Int32(self.blocks) // Int32(self.slices)
        block_end = (Int32(part) + Int32(1)) * Int32(self.blocks) // Int32(self.slices)
        grp = Int64(group)
        k32 = self.k // 32
        live = Int64(m)
        a_base = Int64(a_ptr.toint()) + grp * live * Int64(self.k) + Int64(8) * Int64(j)
        sa_base = Int64(sfa_ptr.toint()) + grp * live * Int64(k32)
        sb_base = (Int64(sfb_ptr.toint()) + grp * Int64((self.n // 32) * k32)
                   + (n0 // Int64(32)) * Int64(k32))
        w_rows = cute.make_rmem_tensor(cute.make_layout((self.col_groups,), stride=(1,)), Int64)
        for gi in cutlass.range_constexpr(self.col_groups):
            w_rows[gi] = (Int64(b_ptr.toint()) + grp * Int64(self.n * self.k)
                          + (n0 + Int64(8 * gi) + Int64(g)) * Int64(self.k) + Int64(8) * Int64(j))
        rows_live = cute.make_rmem_tensor(cute.make_layout((2 * self.m_tiles,), stride=(1,)),
                                          cutlass.Boolean)
        for mt in cutlass.range_constexpr(self.m_tiles):
            for half in cutlass.range_constexpr(2):
                rows_live[2 * mt + half] = g + Int32(16 * mt + 8 * half) < m
        acc = cute.make_rmem_tensor(
            cute.make_layout((self.m_tiles * self.tile_frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.m_tiles * self.tile_frags):
            acc[i] = Float32(0.0)
        per_block = self.col_groups * 8
        cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        for i in cutlass.range_constexpr(per_block):
            cur[i] = Uint32(0)
            nxt[i] = Uint32(0)
        if const_expr(self.prefetch_all):
            # Decode: the next block's activations and scale words load with its
            # weights, so L2 latency stays off the MMA chain.
            act_words = self.m_tiles * 16
            a_cur = cute.make_rmem_tensor(cute.make_layout((act_words,), stride=(1,)), Uint32)
            a_nxt = cute.make_rmem_tensor(cute.make_layout((act_words,), stride=(1,)), Uint32)
            s_cur = cute.make_rmem_tensor(cute.make_layout((1 + 2 * self.m_tiles,), stride=(1,)), Uint32)
            s_nxt = cute.make_rmem_tensor(cute.make_layout((1 + 2 * self.m_tiles,), stride=(1,)), Uint32)
            for i in cutlass.range_constexpr(act_words):
                a_cur[i] = Uint32(0)
                a_nxt[i] = Uint32(0)
            for i in cutlass.range_constexpr(1 + 2 * self.m_tiles):
                s_cur[i] = Uint32(0)
                s_nxt[i] = Uint32(0)
            if block_begin + warp_id < block_end:
                self._load_operands(cur, a_cur, s_cur, w_rows, a_base, sa_base, sb_base,
                                    rows_live, g, block_begin + warp_id)
            for trip in cutlass.range(self.trips, unroll=1):
                block = block_begin + warp_id + Int32(trip * self.warps)
                if block < block_end:
                    if block + Int32(self.warps) < block_end:
                        self._load_operands(nxt, a_nxt, s_nxt, w_rows, a_base, sa_base, sb_base,
                                            rows_live, g, block + Int32(self.warps))
                    self._compute_block(acc, cur, a_cur, s_cur)
                    for i in cutlass.range_constexpr(per_block):
                        cur[i] = nxt[i]
                    for i in cutlass.range_constexpr(act_words):
                        a_cur[i] = a_nxt[i]
                    for i in cutlass.range_constexpr(1 + 2 * self.m_tiles):
                        s_cur[i] = s_nxt[i]
        else:
            if block_begin + warp_id < block_end:
                self._load_weights(cur, w_rows, block_begin + warp_id)
            for trip in cutlass.range(self.trips, unroll=1):
                block = block_begin + warp_id + Int32(trip * self.warps)
                if block < block_end:
                    if block + Int32(self.warps) < block_end:
                        self._load_weights(nxt, w_rows, block + Int32(self.warps))
                    k_byte = Int64(block) * Int64(128)
                    b_word = ld_global_nc_u32(sb_base + Int64(block) * Int64(4))
                    a_words = cute.make_rmem_tensor(
                        cute.make_layout((2 * self.m_tiles,), stride=(1,)), Uint32)
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = g + Int32(16 * mt + 8 * half)
                            a_words[2 * mt + half] = ld_global_nc_u32_pred(
                                sa_base + Int64(row) * Int64(k32) + Int64(block) * Int64(4),
                                rows_live[2 * mt + half])
                    for st in cutlass.range_constexpr(4):
                        e_b = _scale_byte(b_word, st) - Int32(127)
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            row_lo = Int64(g + Int32(16 * mt))
                            at = a_base + row_lo * Int64(self.k) + k_byte + Int64(32 * st)
                            a0, a2 = _ld_act_v2_pred(at, rows_live[2 * mt])
                            a1, a3 = _ld_act_v2_pred(at + Int64(8 * self.k), rows_live[2 * mt + 1])
                            scale_lo = _pow2_biased(_scale_byte(a_words[2 * mt], st) + e_b)
                            scale_hi = _pow2_biased(_scale_byte(a_words[2 * mt + 1], st) + e_b)
                            for gi in cutlass.range_constexpr(self.col_groups):
                                zero = Float32(0.0)
                                d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                                    zero, zero, zero, zero, a0, a1, a2, a3,
                                    cur[(gi * 4 + st) * 2], cur[(gi * 4 + st) * 2 + 1])
                                f = (mt * self.col_groups + gi) * 4
                                acc[f] = acc[f] + d0 * scale_lo
                                acc[f + 1] = acc[f + 1] + d1 * scale_lo
                                acc[f + 2] = acc[f + 2] + d2 * scale_hi
                                acc[f + 3] = acc[f + 3] + d3 * scale_hi
                    for i in cutlass.range_constexpr(per_block):
                        cur[i] = nxt[i]
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.tile_frags * 32,)))
        c_bytes = 4 if const_expr(self.slices > 1) else 2
        c_base = (Int64(c_ptr.toint()) + (grp + Int64(part)) * live * Int64(self.n) * Int64(c_bytes))
        # One row tile at a time through a tile-sized buffer, warps folded in order.
        for mt in cutlass.range_constexpr(self.m_tiles):
            if const_expr(mt > 0):
                cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.tile_frags):
                partial[(warp_id * Int32(self.tile_frags) + Int32(i)) * Int32(32) + lane] = (
                    acc[mt * self.tile_frags + i])
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.tile_frags):
                if Int32(i % self.warps) == warp_id:
                    total = Float32(0.0)
                    for src in cutlass.range_constexpr(self.warps):
                        total = total + partial[Int32((src * self.tile_frags + i) * 32) + lane]
                    gi = i // 4
                    row = g + Int32(16 * mt + 8 * ((i % 4) // 2))
                    col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                    if row < m:
                        address = c_base + (Int64(row) * Int64(self.n) + col) * Int64(c_bytes)
                        if const_expr(self.slices > 1):
                            target = cute.make_ptr(Float32, address, cute.AddressSpace.gmem,
                                                   assumed_align=4)
                            target[0] = total
                        else:
                            target = cute.make_ptr(cutlass.BFloat16, address,
                                                   cute.AddressSpace.gmem, assumed_align=2)
                            target[0] = total.to(cutlass.BFloat16)


class Sm89DenseBlock32Launch:
    """Pointer ABI over the cp.async ``DenseGemmKernel`` block-32 path for
    prefill capacities: A [L, M, K], SFA [L, M, K/32], B [L, N, K],
    SFB [L, N/32, K/32], C BF16 [L, M, N] (``alpha`` is not applied)."""

    def __init__(self, n: int, k: int, *, groups: int = 1, tile_mn=(128, 128)):
        self.n, self.k, self.groups = int(n), int(k), int(groups)
        self.tile_mn = tuple(int(v) for v in tile_mn)
        if self.n % 128 or self.k % 128:
            raise ValueError("SM89 block-32 GEMM needs N and K divisible by 128")

    def key(self) -> tuple:
        return ("sm89_block32_dense", self.n, self.k, self.groups, self.tile_mn)

    def make_kernel(self):
        from b12x._lib.dense_gemm import DenseGemmKernel

        return DenseGemmKernel(
            sf_vec_size=32, mma_tiler_mn=self.tile_mn, cluster_shape_mn=(1, 1), mma_k=32,
            tile_k=128, load_path="cpasync", alpha_is_one=True, target_occupancy=1,
            block32_fp8=True, block32_act_scales=True,
        )

    @cute.jit
    def __call__(self, a_ptr: cute.Pointer, b_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
                 sfb_ptr: cute.Pointer, c_ptr: cute.Pointer, quant_c_values_ptr: cute.Pointer,
                 quant_c_scale_rows_ptr: cute.Pointer, quant_c_scale_mma_ptr: cute.Pointer,
                 alpha_ptr: cute.Pointer, m: Int32, current_stream: cuda.CUstream):
        order = (1, 0, 2)
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout((m, self.k, self.groups), order=order))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout((self.n, self.k, self.groups), order=order))
        c = cute.make_tensor(c_ptr, cute.make_ordered_layout((m, self.n, self.groups), order=order))
        sfa = cute.make_tensor(
            sfa_ptr, cute.make_ordered_layout((m, self.k // 32, self.groups), order=order))
        sfb = cute.make_tensor(
            sfb_ptr, cute.make_ordered_layout((self.n // 32, self.k // 32, self.groups), order=order))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout((1,)))
        self.make_kernel()(
            a, a, alpha, alpha, b, sfa, sfb, c, alpha, alpha, alpha, alpha,
            runtime_num_sm(), current_stream,
        )


def sm89_block32_plan(*, size_m: int, size_n: int, size_k: int, num_groups: int,
                      sm_count: int) -> dict:
    """Kernel and tiling for one capacity (``size_m``) on an SM89 card."""
    if size_m <= GEMV_MAX_ROWS:
        ctas = (size_n // 8) * num_groups
        if size_m > 16:
            # Verify widths: 32 columns per CTA share each activation fragment.
            return dict(kind="gemv", col_groups=4, warps=8, slices=1)
        if size_m == 1:
            # One row: 8 columns per CTA; a deeper in-CTA K split only where
            # fewer CTAs than SMs would leave SMs idle (kv).
            return dict(kind="gemv", col_groups=1, warps=16 if ctas < sm_count else 8, slices=1)
        # Up to 16 rows: wider CTAs reuse each activation fragment across
        # column groups (activation bytes are 2x the weight bytes per CTA at 16
        # rows) while keeping 64 CTAs streaming (32 for outputs below 1024);
        # short K (<= 12 blocks) keeps one group so enough CTAs stream at once.
        # Split-K across CTAs with the fixed-order second pass measured slower
        # at every decode shape: the extra launch outweighs the added CTAs.
        col_groups = 1
        if size_k // 128 > 12:
            floor = 64 if size_n >= 1024 else 32
            col_groups = next((c for c in (4, 2) if (size_n // (8 * c)) * num_groups >= floor), 1)
        return dict(kind="gemv", col_groups=col_groups, warps=8, slices=1)
    # 128x64: the 9-warp warp-specialized CTA caps threads at 168 registers
    # on SM89 (no setmaxnreg); 128x128 with K32 activation and weight scales
    # spills (181-209 TF/s against 265-281 here, M=1024-4096).
    return dict(kind="dense", tile_mn=(128, 64))


def compile_sm89_block32_gemm_aot(*, size_m: int, size_n: int, size_k: int,
                                  num_groups: int = 1, sm_count: int):
    """Compile one runtime-M K32 block-scaled projection for native export on
    SM89; same pointer ABI as ``compile_dense_gemm_mxfp8_aot``. Returns
    ``(compiled, slices)``: split plans write FP32 [slices, M, N] planes."""
    plan = sm89_block32_plan(size_m=size_m, size_n=size_n, size_k=size_k,
                             num_groups=num_groups, sm_count=sm_count)
    if plan["kind"] == "gemv":
        launch = Sm89Block32Gemv(size_n, size_k, groups=num_groups, max_rows=size_m,
                                 warps=plan["warps"], col_groups=plan["col_groups"],
                                 slices=plan["slices"])
    else:
        launch = Sm89DenseBlock32Launch(size_n, size_k, groups=num_groups,
                                        tile_mn=plan["tile_mn"])
    key = launch.key() + (size_m,)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    gmem = cute.AddressSpace.gmem
    pointers = [
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float32 if plan.get("slices", 1) > 1 else cutlass.BFloat16, 16, gmem,
                 assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=16),
    ]
    compiled = b12x_compile(
        launch, *pointers, 1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.sm89_block32", 1, key),
    )
    return compiled, plan.get("slices", 1)
