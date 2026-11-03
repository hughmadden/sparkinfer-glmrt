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

Kernels share one pointer ABI (``_DenseGemmLaunch``'s, so native wrappers are
unchanged): ``Sm89Block32Gemv`` (weights streamed once; decode prefetches
whole K blocks in registers, verify widths stage activations through a
shared-memory ring) and the cp.async ``DenseGemmKernel`` block-32 path with
64-, 96- and 128-row tiles. ``compile_sm89_block32_gemm_aot`` builds each
capacity from ``sm89_block32_plan``: one kernel, or several picked by live
rows (``Sm89RowSplitLaunch``).
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
    ld_shared_v2_u32,
    mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr, runtime_num_sm

# Decode and verify capacities (the output reduction recipe keeps the GEMV
# arithmetic up to here).
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
def _ld_weight_v2_pred(address, pred, *, loc=None, ip=None):
    """``_ld_weight_v2`` that loads nothing (zeros) when ``pred`` is false."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; mov.u32 $0, 0; mov.u32 $1, 0; "
        "@p ld.global.nc.L1::no_allocate.L2::128B.v2.u32 {$0, $1}, [$2]; }",
        "=r,=r,l,r", has_side_effects=False, is_align_stack=False,
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


@dsl_user_op
def _cp_async16_zfill(smem_addr, gmem_addr, live, *, loc=None, ip=None):
    """16-byte ``cp.async.cg`` that zero-fills (reads nothing) when ``live``
    is false."""
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(live).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; .reg .u32 n; setp.ne.b32 p, $2, 0; selp.u32 n, 16, 0, p; "
        "cp.async.cg.shared.global [$0], [$1], 16, n; }",
        "r,l,r,~{memory}", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _warp_sync(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "bar.warp.sync -1;", "~{memory}", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


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
        if not 1 <= self.m_tiles <= 6:
            raise ValueError("SM89 block-32 GEMV covers 1..96 rows")
        self.blocks = self.k // 128
        if self.blocks < self.slices:
            raise ValueError("SM89 block-32 GEMV needs a K block per slice")
        # Slice s covers blocks [s*B//S, (s+1)*B//S); warps take them round-robin.
        slice_blocks = -(-self.blocks // self.slices)
        self.trips = (slice_blocks + self.warps - 1) // self.warps
        self.tile_frags = self.col_groups * 4
        # Decode (up to two row tiles) prefetches whole K blocks in registers;
        # verify widths stage activation half-blocks through shared memory.
        self.prefetch_all = self.m_tiles <= 2
        # Two ring slots per warp: <= 80 KiB at five row tiles. More slots
        # (or a larger carveout) shrink L1, where the warps share the scale
        # words, and measured slower.
        self.smem_slots = 2
        self.ring_bytes = 0
        if not self.prefetch_all:
            self.ring_bytes = self.warps * self.smem_slots * self.m_tiles * 1024

    def key(self) -> tuple:
        return ("sm89_block32_gemv", self.n, self.k, self.groups, self.m_tiles,
                self.col_groups, self.warps, self.slices)

    def _storage(self):
        class Storage:
            pass

        # The fold's partial tile aliases the activation ring (after the loop).
        floats = max(self.warps * self.tile_frags * 32, self.ring_bytes // 4)
        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, floats], 128],
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
    def _load_words(self, s_dst: cute.Tensor, sa_base: Int64, sb_base: Int64,
                    rows_live: cute.Tensor, g: Int32, block: Int32, tiles: int):
        """The weight scale word and every live row's scale word of a block."""
        k32 = self.k // 32
        s_dst[0] = ld_global_nc_u32(sb_base + Int64(block) * Int64(4))
        for mt in cutlass.range_constexpr(tiles):
            for half in cutlass.range_constexpr(2):
                row = g + Int32(16 * mt + 8 * half)
                s_dst[1 + 2 * mt + half] = ld_global_nc_u32_pred(
                    sa_base + Int64(row) * Int64(k32) + Int64(block) * Int64(4),
                    rows_live[2 * mt + half])

    @cute.jit
    def _load_act_step(self, a_dst: cute.Tensor, a_off: int, a_base: Int64,
                       rows_live: cute.Tensor, g: Int32, block: Int32, st: int, tiles: int):
        """Activation fragments of one K32 step for ``tiles`` row tiles."""
        k_byte = Int64(block) * Int64(128) + Int64(32 * st)
        for mt in cutlass.range_constexpr(tiles):
            at = a_base + Int64(g + Int32(16 * mt)) * Int64(self.k) + k_byte
            f = a_off + mt * 4
            a_dst[f], a_dst[f + 2] = _ld_act_v2_pred(at, rows_live[2 * mt])
            a_dst[f + 1], a_dst[f + 3] = _ld_act_v2_pred(at + Int64(8 * self.k),
                                                         rows_live[2 * mt + 1])

    @cute.jit
    def _load_words_tail(self, s_dst: cute.Tensor, sa_base: Int64, sb_base: Int64,
                         tail: cute.Tensor, any_rows, g: Int32, block: Int32, tiles: int):
        """``_load_words`` with full tiles under ``any_rows`` (see above)."""
        k32 = self.k // 32
        s_dst[0] = ld_global_nc_u32_pred(sb_base + Int64(block) * Int64(4), any_rows)
        for mt in cutlass.range_constexpr(tiles):
            for half in cutlass.range_constexpr(2):
                row = g + Int32(16 * mt + 8 * half)
                pred = any_rows
                if const_expr(mt == tiles - 1):
                    pred = tail[half]
                s_dst[1 + 2 * mt + half] = ld_global_nc_u32_pred(
                    sa_base + Int64(row) * Int64(k32) + Int64(block) * Int64(4), pred)

    @cute.jit
    def _load_operands(self, w_dst: cute.Tensor, a_dst: cute.Tensor, s_dst: cute.Tensor,
                       w_rows: cute.Tensor, a_base: Int64, sa_base: Int64, sb_base: Int64,
                       rows_live: cute.Tensor, g: Int32, block: Int32, tiles: int):
        """Everything one 128-wide K block needs: weights, activation
        fragments, the weight scale word and every row's scale word."""
        self._load_weights(w_dst, w_rows, block)
        self._load_words(s_dst, sa_base, sb_base, rows_live, g, block, tiles)
        for st in cutlass.range_constexpr(4):
            self._load_act_step(a_dst, st * tiles * 4, a_base, rows_live, g, block, st, tiles)

    @cute.jit
    def _compute_step(self, acc: cute.Tensor, w: cute.Tensor, act: cute.Tensor, a_off: int,
                      words: cute.Tensor, st: int, tiles: int):
        """One K32 step: an MMA into a zero partial per row tile and column
        group, promoted by 2^(ea + eb - 254) into the FP32 accumulator."""
        e_b = _scale_byte(words[0], st) - Int32(127)
        for mt in cutlass.range_constexpr(tiles):
            self._compute_tile(acc, w, act, words, e_b, st, mt, a_off + mt * 4)

    @cute.jit
    def _compute_tile(self, acc: cute.Tensor, w: cute.Tensor, act: cute.Tensor,
                      words: cute.Tensor, e_b: Int32, st: int, mt: int, f: int = 0):
        """Row tile ``mt`` of K32 step ``st``: activation fragment at
        ``act[f:f+4]``, every column group."""
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

    @cute.jit
    def _block_loop(self, acc: cute.Tensor, cur: cute.Tensor, nxt: cute.Tensor,
                    w_rows: cute.Tensor, a_base: Int64, sa_base: Int64, sb_base: Int64,
                    rows_live: cute.Tensor, g: Int32, warp_id: Int32, block_begin: Int32,
                    block_end: Int32, tiles: int):
        """Whole-block prefetch: the next block's weights, activations and
        scale words load together while this block computes."""
        per_block = self.col_groups * 8
        act_words = tiles * 16
        a_cur = cute.make_rmem_tensor(cute.make_layout((act_words,), stride=(1,)), Uint32)
        a_nxt = cute.make_rmem_tensor(cute.make_layout((act_words,), stride=(1,)), Uint32)
        s_cur = cute.make_rmem_tensor(cute.make_layout((1 + 2 * tiles,), stride=(1,)), Uint32)
        s_nxt = cute.make_rmem_tensor(cute.make_layout((1 + 2 * tiles,), stride=(1,)), Uint32)
        for i in cutlass.range_constexpr(act_words):
            a_cur[i] = Uint32(0)
            a_nxt[i] = Uint32(0)
        for i in cutlass.range_constexpr(1 + 2 * tiles):
            s_cur[i] = Uint32(0)
            s_nxt[i] = Uint32(0)
        if block_begin + warp_id < block_end:
            self._load_operands(cur, a_cur, s_cur, w_rows, a_base, sa_base, sb_base,
                                rows_live, g, block_begin + warp_id, tiles)
        for trip in cutlass.range(self.trips, unroll=1):
            block = block_begin + warp_id + Int32(trip * self.warps)
            if block < block_end:
                if block + Int32(self.warps) < block_end:
                    self._load_operands(nxt, a_nxt, s_nxt, w_rows, a_base, sa_base, sb_base,
                                        rows_live, g, block + Int32(self.warps), tiles)
                for st in cutlass.range_constexpr(4):
                    self._compute_step(acc, cur, a_cur, st * tiles * 4, s_cur, st, tiles)
                for i in cutlass.range_constexpr(per_block):
                    cur[i] = nxt[i]
                for i in cutlass.range_constexpr(act_words):
                    a_cur[i] = a_nxt[i]
                for i in cutlass.range_constexpr(1 + 2 * tiles):
                    s_cur[i] = s_nxt[i]

    @cute.jit
    def _smem_loop(self, acc: cute.Tensor, w_rows: cute.Tensor, a_rows: Int64, sa_base: Int64,
                   sb_base: Int64, ring_base: Int32, m: Int32, g: Int32, j: Int32,
                   warp_id: Int32, block_begin: Int32, block_end: Int32, tiles: int,
                   slots: int):
        """Activation half-blocks (all live rows x 64 bytes = two K32 steps)
        stream through a per-warp shared-memory ring with 16-byte cp.async
        (two sectors per L2 request instead of one: per-step 8-byte loads
        cap at about 2 TB/s from L2 on this card, these at about 4),
        ``slots - 1`` half-blocks ahead; MMA fragments come from shared
        memory. Each K32 step's weight registers refill with the next block's
        step once consumed (a block of weights in flight) and the next
        block's scale words load one or two steps ahead. Only the last of
        ``tiles`` live row tiles can be partial, so only its rows carry
        predicates."""
        cg = self.col_groups
        nwords = 1 + 2 * tiles
        slot_bytes = tiles * 16 * 64
        ring_base = ring_base + warp_id * Int32(slots * slot_bytes)
        words_at = 3 if const_expr(tiles >= 5) else 2
        w_ring = cute.make_rmem_tensor(cute.make_layout((4 * cg * 2,), stride=(1,)), Uint32)
        s_cur = cute.make_rmem_tensor(cute.make_layout((nwords,), stride=(1,)), Uint32)
        s_nxt = cute.make_rmem_tensor(cute.make_layout((nwords,), stride=(1,)), Uint32)
        tail = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Boolean)
        tail_next = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), cutlass.Boolean)
        frag = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
        for half in cutlass.range_constexpr(2):
            tail[half] = g + Int32(16 * (tiles - 1) + 8 * half) < m
        for i in cutlass.range_constexpr(4 * cg * 2):
            w_ring[i] = Uint32(0)
        for i in cutlass.range_constexpr(nwords):
            s_cur[i] = Uint32(0)
            s_nxt[i] = Uint32(0)
        # Rows r and r + 2 alternate 32-byte halves so a half-warp's four rows
        # of 8-byte fragment reads cover all 32 banks.
        swz = (g >> Int32(1)) & Int32(1)
        copy_off = (g * Int32(64) + (((j >> Int32(1)) ^ swz) * Int32(32))
                    + (j & Int32(1)) * Int32(16))
        read_off0 = g * Int32(64) + swz * Int32(32) + j * Int32(8)
        read_off1 = g * Int32(64) + (swz ^ Int32(1)) * Int32(32) + j * Int32(8)
        first = block_begin + warp_id
        nhalves = Int32(2) * ((block_end - first + Int32(self.warps - 1)) // Int32(self.warps))
        # Half-block q of this warp: block first + 8 * (q // 2), bytes 64 * (q % 2).
        for q in cutlass.range_constexpr(slots - 1):
            self._issue_half(a_rows, ring_base + Int32(q * slot_bytes) + copy_off,
                             cutlass.min(first + Int32(8 * (q // 2)), Int32(self.blocks - 1)),
                             q % 2, Int32(q) < nhalves, tail, tiles)
            cute.arch.cp_async_commit_group()
        if first < block_end:
            self._load_weights(w_ring, w_rows, first)
            self._load_words_tail(s_cur, sa_base, sb_base, tail, True, g, first, tiles)
        for trip in cutlass.range(self.trips, unroll=1):
            block = first + Int32(trip * self.warps)
            if block < block_end:
                after = block + Int32(self.warps)
                more = after < block_end
                nb = cutlass.min(after, Int32(self.blocks - 1))
                for half in cutlass.range_constexpr(2):
                    tail_next[half] = tail[half] & more
                k_next = Int64(nb) * Int64(128)
                for half in cutlass.range_constexpr(2):
                    q = Int32(2) * Int32(trip) + Int32(half)
                    ahead_q = q + Int32(slots - 1)
                    ahead_block = cutlass.min(first + (ahead_q >> Int32(1)) * Int32(self.warps),
                                              Int32(self.blocks - 1))
                    _warp_sync()
                    self._issue_half(
                        a_rows, ring_base + (ahead_q & Int32(slots - 1)) * Int32(slot_bytes)
                        + copy_off, ahead_block, (half + slots - 1) % 2, ahead_q < nhalves,
                        tail, tiles)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(slots - 1)
                    _warp_sync()
                    slot_base = ring_base + (q & Int32(slots - 1)) * Int32(slot_bytes)
                    for sub in cutlass.range_constexpr(2):
                        st = 2 * half + sub
                        read_off = read_off0 if const_expr(sub == 0) else read_off1
                        if const_expr(st == words_at):
                            self._load_words_tail(s_nxt, sa_base, sb_base, tail_next, more, g,
                                                  nb, tiles)
                        e_b = _scale_byte(s_cur[0], st) - Int32(127)
                        for mt in cutlass.range_constexpr(tiles):
                            row_lo = slot_base + read_off + Int32(16 * mt * 64)
                            frag[0], frag[2] = ld_shared_v2_u32(row_lo)
                            frag[1], frag[3] = ld_shared_v2_u32(row_lo + Int32(8 * 64))
                            self._compute_tile(acc, w_ring, frag, s_cur, e_b, st, mt)
                        for gi in cutlass.range_constexpr(cg):
                            lo, hi = _ld_weight_v2_pred(w_rows[gi] + k_next + Int64(32 * st),
                                                        more)
                            w_ring[(gi * 4 + st) * 2] = lo
                            w_ring[(gi * 4 + st) * 2 + 1] = hi
                for i in cutlass.range_constexpr(nwords):
                    s_cur[i] = s_nxt[i]
        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def _issue_half(self, a_rows: Int64, smem: Int32, block: Int32, half: int, valid,
                    tail: cute.Tensor, tiles: int):
        """cp.async one half-block (64 bytes of every live row) into a ring
        slot; rows past live M and halves past the warp's last block fill
        zeros without reading."""
        src = a_rows + Int64(block) * Int64(128) + Int64(64 * half)
        for rg in cutlass.range_constexpr(2 * tiles):
            live = valid
            if const_expr(rg >= 2 * tiles - 2):
                live = valid & tail[rg - (2 * tiles - 2)]
            _cp_async16_zfill(smem + Int32(rg * 8 * 64), src + Int64(rg * 8 * self.k), live)

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
        live_tiles = (m + Int32(15)) // Int32(16)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        acc = cute.make_rmem_tensor(
            cute.make_layout((self.m_tiles * self.tile_frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.m_tiles * self.tile_frags):
            acc[i] = Float32(0.0)
        if const_expr(self.prefetch_all):
            # Decode: the next block's activations and scale words load with its
            # weights, so L2 latency stays off the MMA chain.
            rows_live = cute.make_rmem_tensor(
                cute.make_layout((2 * self.m_tiles,), stride=(1,)), cutlass.Boolean)
            for mt in cutlass.range_constexpr(self.m_tiles):
                for half in cutlass.range_constexpr(2):
                    rows_live[2 * mt + half] = g + Int32(16 * mt + 8 * half) < m
            per_block = self.col_groups * 8
            cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
            nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
            for i in cutlass.range_constexpr(per_block):
                cur[i] = Uint32(0)
                nxt[i] = Uint32(0)
            self._block_loop(acc, cur, nxt, w_rows, a_base, sa_base, sb_base, rows_live, g,
                             warp_id, block_begin, block_end, self.m_tiles)
        else:
            # Verify widths: one specialized loop per live row-tile count, so
            # dead tiles cost neither MMAs nor load slots.
            a_rows = (Int64(a_ptr.toint()) + grp * live * Int64(self.k)
                      + Int64(g) * Int64(self.k) + Int64(16) * Int64(j))
            ring0 = shared_ptr_to_u32(storage.partial.get_tensor(cute.make_layout((4,))).iterator)
            for t in cutlass.range_constexpr(1, self.m_tiles + 1):
                if live_tiles == Int32(t):
                    self._smem_loop(acc, w_rows, a_rows, sa_base, sb_base, ring0, m, g, j,
                                    warp_id, block_begin, block_end, t, self.smem_slots)
            # The fold's partial tile aliases the ring.
            cute.arch.sync_threads()
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.tile_frags * 32,)))
        c_bytes = 4 if const_expr(self.slices > 1) else 2
        c_base = (Int64(c_ptr.toint()) + (grp + Int64(part)) * live * Int64(self.n) * Int64(c_bytes))
        # One live row tile at a time through a tile-sized buffer, warps folded
        # in order.
        for mt in cutlass.range_constexpr(self.m_tiles):
            if Int32(mt) < live_tiles:
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

    def __init__(self, n: int, k: int, *, groups: int = 1, tile_mn=(128, 128),
                 atom_24: bool = False):
        self.n, self.k, self.groups = int(n), int(k), int(groups)
        self.tile_mn = tuple(int(v) for v in tile_mn)
        self.atom_24 = bool(atom_24)
        if self.n % 128 or self.k % 128:
            raise ValueError("SM89 block-32 GEMM needs N and K divisible by 128")

    def key(self) -> tuple:
        return ("sm89_block32_dense", self.n, self.k, self.groups, self.tile_mn, self.atom_24)

    def make_kernel(self):
        from b12x._lib.dense_gemm import DenseGemmKernel

        return DenseGemmKernel(
            sf_vec_size=32, mma_tiler_mn=self.tile_mn, cluster_shape_mn=(1, 1), mma_k=32,
            tile_k=128, load_path="cpasync", alpha_is_one=True, target_occupancy=1,
            block32_fp8=True, block32_act_scales=True, atom_shape_24=self.atom_24,
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


def _wave_efficiency(tiles: int, sm_count: int) -> float:
    return tiles / (-(-tiles // sm_count) * sm_count)


def _dense_part(rows: int, tile_mn, atom_24: bool = False) -> dict:
    return dict(rows=rows, kind="dense", tile_mn=tuple(tile_mn), atom_24=atom_24)


def _gemv_part(rows: int, *, col_groups: int = 4, warps: int = 8, slices: int = 1) -> dict:
    return dict(rows=rows, kind="gemv", max_rows=rows, col_groups=col_groups, warps=warps,
                slices=slices)


def _verify_dense_part(rows: int, size_n: int, num_groups: int, sm_count: int) -> dict:
    """Dense tile for up to ``rows`` (<= 96) live rows: the candidate grid with
    the best wave efficiency on this card, ties to the larger tile. Up to 64
    rows one 64-row tile streams each weight once (64x32 doubles the grid);
    above, 96x64 (2x4 warps) competes with two 64-row tiles, whose second
    weight read hits L2. 32-wide tiles are ungrouped only, and over two row
    tiles only for narrow outputs (they measured slow for wide ones)."""
    tiles = (size_n // 64) * num_groups
    if rows <= 64:
        cands = [((64, 64), tiles, False)]
        if num_groups == 1:
            cands.append(((64, 32), 2 * tiles, False))
    else:
        cands = [((96, 64), tiles, True), ((64, 64), 2 * tiles, False)]
        if num_groups == 1 and size_n <= 2048:
            cands.append(((64, 32), 4 * tiles, False))
    tile, _, atom_24 = max(cands, key=lambda c: round(_wave_efficiency(c[1], sm_count), 6))
    return _dense_part(rows, tile, atom_24)


def sm89_block32_plan(*, size_m: int, size_n: int, size_k: int, num_groups: int,
                      sm_count: int, fixed_rows: bool = False) -> dict:
    """Kernels for one capacity (``size_m``) on an SM89 card: ``parts`` is a
    list of ``rows``-bounded kernels (the first whose ``rows`` holds the live
    rows runs), ``slices`` the split-K planes of a decode GEMV.

    ``fixed_rows`` (the output reduction recipe) keeps every capacity up to
    80 rows on the decode GEMV arithmetic. Measured on an RTX 4090 D (114
    SMs, CUDA graph, clean L2), these rules pick the fastest kernel of each
    row band for every DeepSeek V4.1 coordinator projection."""
    if size_m <= 16:
        ctas = (size_n // 8) * num_groups
        if size_m == 1:
            # One row: 8 columns per CTA; a deeper in-CTA K split only where
            # fewer CTAs than SMs would leave SMs idle (kv).
            return dict(parts=[_gemv_part(1, col_groups=1, warps=16 if ctas < sm_count else 8)],
                        slices=1)
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
        return dict(parts=[_gemv_part(size_m, col_groups=col_groups)], slices=1)
    if size_m <= GEMV_MAX_ROWS:
        # Verify widths. Rows <= 32: the GEMV (32 columns per CTA, whole-block
        # register prefetch) for long K, unless a narrow output still gives the
        # 64x32 dense grid half the SMs; else a 64-row dense tile, which
        # streams each weight once and computes four row tiles.
        n_total = size_n * num_groups
        long_k = size_k >= 4096
        low_gemv = fixed_rows or (long_k and (n_total >= 4096 or n_total // 32 < sm_count // 2))
        if size_m <= 32:
            part = (_gemv_part(size_m) if low_gemv
                    else _verify_dense_part(size_m, size_n, num_groups, sm_count))
            return dict(parts=[part], slices=1)
        if size_m <= 64:
            # Rows 33-64: dense, except the GEMV's shared-memory ring for the
            # fixed recipe and grouped outputs (128 64-wide tiles over 114 SMs).
            high = (_gemv_part(size_m) if fixed_rows or num_groups > 1
                    else _verify_dense_part(size_m, size_n, num_groups, sm_count))
        else:
            high = (_gemv_part(size_m) if fixed_rows
                    else _verify_dense_part(size_m, size_n, num_groups, sm_count))
            # Rows 65-80 only reach this capacity: one kernel serves it.
            return dict(parts=[high], slices=1)
        low = (_gemv_part(32) if low_gemv
               else _verify_dense_part(32, size_n, num_groups, sm_count))
        if low["kind"] == high["kind"] == "dense" and low["tile_mn"] == high["tile_mn"]:
            return dict(parts=[high], slices=1)
        return dict(parts=[low, high], slices=1)
    # Prefill. 128x64: the 9-warp warp-specialized CTA caps threads at 168
    # registers on SM89 (no setmaxnreg); 128x128 with K32 activation and weight
    # scales spills (181-209 TF/s against 265-281 here, M=1024-4096). Rows up
    # to 96 (verify beyond 80) take the verify dense tile.
    prefill = _dense_part(size_m, (128, 64))
    if size_m <= 256:
        return dict(parts=[_verify_dense_part(96, size_n, num_groups, sm_count), prefill],
                    slices=1)
    return dict(parts=[prefill], slices=1)


class Sm89RowSplitLaunch:
    """One pointer-ABI entry over several kernels picked by live rows:
    ``parts`` is ``[(max_rows, launch), ...]`` in increasing ``max_rows``; the
    last part takes every row count above the previous ones."""

    def __init__(self, parts):
        self.parts = list(parts)
        if not 1 <= len(self.parts) <= 3:
            raise ValueError("SM89 row split takes one to three kernels")

    def key(self) -> tuple:
        return ("sm89_rows",) + tuple((rows, launch.key()) for rows, launch in self.parts)

    @cute.jit
    def __call__(self, a_ptr: cute.Pointer, b_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
                 sfb_ptr: cute.Pointer, c_ptr: cute.Pointer, quant_c_values_ptr: cute.Pointer,
                 quant_c_scale_rows_ptr: cute.Pointer, quant_c_scale_mma_ptr: cute.Pointer,
                 alpha_ptr: cute.Pointer, m: Int32, current_stream: cuda.CUstream):
        args = (a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, quant_c_values_ptr,
                quant_c_scale_rows_ptr, quant_c_scale_mma_ptr, alpha_ptr, m, current_stream)
        if const_expr(len(self.parts) == 1):
            self.parts[0][1](*args)
        elif const_expr(len(self.parts) == 2):
            if m <= Int32(self.parts[0][0]):
                self.parts[0][1](*args)
            else:
                self.parts[1][1](*args)
        else:
            if m <= Int32(self.parts[0][0]):
                self.parts[0][1](*args)
            elif m <= Int32(self.parts[1][0]):
                self.parts[1][1](*args)
            else:
                self.parts[2][1](*args)


def _part_launch(part: dict, n: int, k: int, groups: int):
    if part["kind"] == "gemv":
        return Sm89Block32Gemv(n, k, groups=groups, max_rows=part["max_rows"],
                               warps=part["warps"], col_groups=part["col_groups"],
                               slices=part.get("slices", 1))
    return Sm89DenseBlock32Launch(n, k, groups=groups, tile_mn=part["tile_mn"],
                                  atom_24=part.get("atom_24", False))


def compile_sm89_block32_gemm_aot(*, size_m: int, size_n: int, size_k: int,
                                  num_groups: int = 1, sm_count: int,
                                  fixed_rows: bool = False):
    """Compile one runtime-M K32 block-scaled projection for native export on
    SM89; same pointer ABI as ``compile_dense_gemm_mxfp8_aot``. Returns
    ``(compiled, slices)``: split plans write FP32 [slices, M, N] planes.
    ``fixed_rows`` keeps every kernel on the decode GEMV arithmetic, so a
    row's bits do not depend on the capacity or the live rows."""
    plan = sm89_block32_plan(size_m=size_m, size_n=size_n, size_k=size_k,
                             num_groups=num_groups, sm_count=sm_count, fixed_rows=fixed_rows)
    parts = [(part["rows"], _part_launch(part, size_n, size_k, num_groups))
             for part in plan["parts"]]
    launch = parts[0][1] if len(parts) == 1 else Sm89RowSplitLaunch(parts)
    key = launch.key() + (size_m,)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    gmem = cute.AddressSpace.gmem
    pointers = [
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 16, gmem, assumed_align=16),
        make_ptr(cutlass.Float32 if plan["slices"] > 1 else cutlass.BFloat16, 16, gmem,
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
    return compiled, plan["slices"]
