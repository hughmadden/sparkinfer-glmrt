"""Native E4M3 dense linear on tensor cores: one K order from decode to prefill.

Every output element is one FP32 accumulator chain over all of K in ascending
64-wide chunks. Inside a chunk, m16n8k16 step ``t`` gives quad lane ``j`` the
logical ``k = 16 j + 4 t + i`` (``i`` 0..3) for both operands, so each lane
reads one contiguous 16-byte weight run and one 32-byte activation run per row.
No path splits K, and no live row count, tile shape or CTA count changes the
chain: the decode route (weights streamed global -> registers, reused by up to
four 16-row tiles) and the prefill route (cp.async-staged 128x128 tiles, the
E4M3 tile widened once per CTA tile) are bitwise one function of
``(x row, weight column)``.

Weights are widened exactly (E4M3 -> BF16) in registers; no dequantized matrix
is materialized. ``fp8_row`` applies its FP32 per-output scale to the finished
accumulator.
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
    bf16_mma_m16n8k16_f32, cp_async4_shared_global_pred, ld_global_nc_v2_u32,
    ld_shared_v4_u32, shared_ptr_to_u32, st_global_v2_f32, u32_as_f32,
)
from b12x.gemm.bf16_gemv._skinny import _ld_cached, _ld_stream

DECODE_ROWS = 64          # decode route capacity (four 16-row tiles per warp)
PREFILL_TILE = (128, 128)  # rows x columns of one prefill CTA tile
PREFILL_STAGES = 4
PREFILL_WARPS = (2, 4)    # warps along rows x columns (warp tile 64 x 32)
DECODE_WARPS = 4


@dsl_user_op
def _e4m3x4_bf16(word, *, loc=None, ip=None):
    """Exact BF16 of four packed E4M3 (byte 0 first) as two bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 l, h; mov.b32 {l, h}, $2; "
        "cvt.rn.bf16x2.e4m3x2 $0, l; cvt.rn.bf16x2.e4m3x2 $1, h; }",
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class NativeTileLinear:
    """``output[rows, N] (F32) = x[rows, K] (BF16) @ W[N, K]^T`` for E4M3 ``W``.

    ``groups``: 8-column groups a decode warp owns (its weight reuse across
    row tiles); decode CTAs take balanced contiguous column ranges so any
    ``cta_limit`` keeps per-SM bytes even."""

    def __init__(self, g, *, groups: int = 2, stages: int = PREFILL_STAGES):
        self.k, self.n = g["input_features"], g["output_features"]
        self.encoding, self.max_rows = g["encoding"], g["max_rows"]
        if self.encoding != "fp8_row":
            raise ValueError("tile core currently serves fp8_row")
        self.groups = int(groups)
        self.cols = 8 * self.groups
        self.stages = int(stages)
        self.bm, self.bn = PREFILL_TILE
        self.wm, self.wn = PREFILL_WARPS
        self.mt = self.bm // self.wm // 16           # 16-row tiles per prefill warp
        self.ng = self.bn // self.wn // 8            # 8-column groups per prefill warp
        if self.k % 128 or self.n % self.cols:
            raise ValueError("tile core needs K % 128 == 0 and N % (8 * groups) == 0")
        self.n_tiles = (self.n + self.bn - 1) // self.bn
        self.a_bytes = self.bm * 128                 # BF16 rows x 64 K
        self.w_bytes = self.bn * 64                  # E4M3 columns x 64 K
        self.stage_bytes = self.a_bytes + self.w_bytes
        self.prefill_threads = 32 * self.wm * self.wn

    # ------------------------------------------------------------------ launch
    @cute.jit
    def __call__(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                 output: cute.Tensor, rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            if rows <= Int32(DECODE_ROWS):
                self.decode(x, payload, scales, output, rows).launch(
                    grid=(min(Int32(self.n // self.cols), cta_limit), 1, 1),
                    block=(32 * DECODE_WARPS, 1, 1), stream=stream)
            else:
                tiles = ((rows + Int32(self.bm - 1)) // Int32(self.bm)) * Int32(self.n_tiles)
                self.prefill(x, payload, scales, output, rows).launch(
                    grid=(min(tiles, cta_limit), 1, 1), block=(self.prefill_threads, 1, 1),
                    stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------ shared
    @cute.jit
    def _store(self, output: cute.Tensor, scales: cute.Tensor, acc0: Float32, acc1: Float32,
               row: Int32, col: Int32, rows: Int32):
        """Scaled F32 pair at (row, col..col+1); col is even."""
        if row < rows and col < Int32(self.n):
            s0, s1 = ld_global_nc_v2_u32(scales.iterator.toint() + col.to(Int64) * Int64(4))
            st_global_v2_f32(output.iterator.toint()
                             + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4),
                             acc0 * u32_as_f32(s0), acc1 * u32_as_f32(s1))

    # ------------------------------------------------------------------ decode
    @cute.kernel
    def decode(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
               output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        g, j = lane >> Int32(2), lane & Int32(3)
        # Balanced contiguous column blocks per warp across the whole grid: a
        # column's arithmetic never depends on which warp or CTA computes it.
        blocks = Int64(self.n // self.cols)
        warps = Int64(grid) * Int64(DECODE_WARPS)
        gw = Int64(cta) * Int64(DECODE_WARPS) + warp.to(Int64)
        begin = (gw * blocks // warps).to(Int32)
        end = ((gw + Int64(1)) * blocks // warps).to(Int32)
        live = (rows + Int32(15)) >> Int32(4)
        G = self.groups
        per = 2 * G * 4
        x_base = x.iterator.toint() + Int64(32) * j.to(Int64)
        block = begin
        while block < end:
            n0 = block * Int32(self.cols)
            w_row = payload.iterator.toint() + (n0 + g).to(Int64) * Int64(self.k) + Int64(16) * j.to(Int64)
            acc = cute.make_rmem_tensor((4 * G * 4,), Float32)
            acc.fill(Float32(0.0))
            cur = cute.make_rmem_tensor((per,), Uint32)
            nxt = cute.make_rmem_tensor((per,), Uint32)
            for c in cutlass.range_constexpr(2):
                for gi in cutlass.range_constexpr(G):
                    words = _ld_stream(w_row + Int64(gi * 8 * self.k + c * 64))
                    for t in cutlass.range_constexpr(4):
                        cur[(c * G + gi) * 4 + t] = words[t]
            for kb in cutlass.range(self.k // 128, unroll=1):
                if kb + Int32(1) < Int32(self.k // 128):
                    for c in cutlass.range_constexpr(2):
                        for gi in cutlass.range_constexpr(G):
                            words = _ld_stream(w_row + Int64(gi * 8 * self.k + c * 64)
                                               + (kb + Int32(1)).to(Int64) * Int64(128))
                            for t in cutlass.range_constexpr(4):
                                nxt[(c * G + gi) * 4 + t] = words[t]
                for c in cutlass.range_constexpr(2):
                    k_off = kb.to(Int64) * Int64(128) + Int64(c * 64)
                    bw = cute.make_rmem_tensor((8 * G,), Uint32)
                    for gi in cutlass.range_constexpr(G):
                        for t in cutlass.range_constexpr(4):
                            b0, b1 = _e4m3x4_bf16(cur[(c * G + gi) * 4 + t])
                            bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                    for mt in cutlass.range_constexpr(4):
                        if Int32(mt) < live:
                            r_lo = Int32(16 * mt) + g
                            r_hi = r_lo + Int32(8)
                            xa = cute.make_rmem_tensor((16,), Uint32)
                            xa.fill(Uint32(0))
                            if r_lo < rows:
                                at = x_base + (r_lo.to(Int64) * Int64(self.k) + k_off) * Int64(2)
                                lo, hi = _ld_cached(at), _ld_cached(at + Int64(16))
                                for i in cutlass.range_constexpr(4):
                                    xa[i], xa[4 + i] = lo[i], hi[i]
                            if r_hi < rows:
                                at = x_base + (r_hi.to(Int64) * Int64(self.k) + k_off) * Int64(2)
                                lo, hi = _ld_cached(at), _ld_cached(at + Int64(16))
                                for i in cutlass.range_constexpr(4):
                                    xa[8 + i], xa[12 + i] = lo[i], hi[i]
                            for gi in cutlass.range_constexpr(G):
                                f = 4 * (mt * G + gi)
                                for t in cutlass.range_constexpr(4):
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                        bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = d0, d1, d2, d3
                for i in cutlass.range_constexpr(per):
                    cur[i] = nxt[i]
            for mt in cutlass.range_constexpr(4):
                if Int32(mt) < live:
                    for gi in cutlass.range_constexpr(G):
                        f = 4 * (mt * G + gi)
                        col = n0 + Int32(8 * gi) + Int32(2) * j
                        self._store(output, scales, acc[f], acc[f + 1], Int32(16 * mt) + g, col, rows)
                        self._store(output, scales, acc[f + 2], acc[f + 3], Int32(16 * mt + 8) + g, col, rows)
            block += Int32(1)

    # ------------------------------------------------------------------ prefill
    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.stages * self.stage_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def _issue(self, base: Int32, x: cute.Tensor, payload: cute.Tensor, rows: Int32,
               m0: Int32, n0: Int32, kc: Int32, stage: Int32, tid: Int32):
        """cp.async one 64-wide K chunk of the A (swizzled) and W tiles into ``stage``."""
        sa = base + stage * Int32(self.stage_bytes)
        sw = sa + Int32(self.a_bytes)
        for i in cutlass.range_constexpr(self.a_bytes // 16 // self.prefill_threads):
            chunk = Int32(i * self.prefill_threads) + tid
            r, c = chunk >> Int32(3), chunk & Int32(7)
            row = m0 + r
            src = x.iterator.toint() + (row.to(Int64) * Int64(self.k)
                                        + kc.to(Int64) * Int64(64) + (c * Int32(8)).to(Int64)) * Int64(2)
            pred = Int32(0)
            if row < rows:
                pred = Int32(1)
            cp_async4_shared_global_pred(sa + r * Int32(128) + ((c ^ (r & Int32(7))) << Int32(4)), src, pred)
        for i in cutlass.range_constexpr(self.w_bytes // 16 // self.prefill_threads):
            chunk = Int32(i * self.prefill_threads) + tid
            r, c = chunk >> Int32(2), chunk & Int32(3)
            src = payload.iterator.toint() + (n0 + r).to(Int64) * Int64(self.k) \
                + kc.to(Int64) * Int64(64) + (c * Int32(16)).to(Int64)
            pred = Int32(0)
            if n0 + r < Int32(self.n):
                pred = Int32(1)
            cp_async4_shared_global_pred(sw + r * Int32(64) + (c << Int32(4)), src, pred)

    @cute.kernel
    def prefill(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        g, j = lane >> Int32(2), lane & Int32(3)
        wm, wn = warp // Int32(self.wn), warp % Int32(self.wn)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        m_tiles = (rows + Int32(self.bm - 1)) // Int32(self.bm)
        tiles = m_tiles * Int32(self.n_tiles)
        kcs = Int32(self.k // 64)
        MT, NG, S = self.mt, self.ng, self.stages
        while tile < tiles:
            m0 = (tile % m_tiles) * Int32(self.bm)
            n0 = (tile // m_tiles) * Int32(self.bn)
            acc = cute.make_rmem_tensor((MT * NG * 4,), Float32)
            acc.fill(Float32(0.0))
            for s in cutlass.range_constexpr(S - 1):
                if Int32(s) < kcs:
                    self._issue(base, x, payload, rows, m0, n0, Int32(s), Int32(s), tid)
                cute.arch.cp_async_commit_group()
            for kc in cutlass.range(self.k // 64, unroll=1):
                cute.arch.cp_async_wait_group(S - 2)
                cute.arch.sync_threads()
                nk = kc + Int32(S - 1)
                if nk < kcs:
                    self._issue(base, x, payload, rows, m0, n0, nk, nk % Int32(S), tid)
                cute.arch.cp_async_commit_group()
                sa = base + (kc % Int32(S)) * Int32(self.stage_bytes)
                sw = sa + Int32(self.a_bytes)
                bw = cute.make_rmem_tensor((8 * NG,), Uint32)
                for gi in cutlass.range_constexpr(NG):
                    w = ld_shared_v4_u32(sw + (wn * Int32(8 * NG) + Int32(8 * gi) + g) * Int32(64) + (j << Int32(4)))
                    for t in cutlass.range_constexpr(4):
                        b0, b1 = _e4m3x4_bf16(w[t])
                        bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                for mt in cutlass.range_constexpr(MT):
                    r_lo = wm * Int32(16 * MT) + Int32(16 * mt) + g
                    xa = cute.make_rmem_tensor((16,), Uint32)
                    for h in cutlass.range_constexpr(2):
                        r = r_lo + Int32(8 * h)
                        row_at = sa + r * Int32(128)
                        sw7 = r & Int32(7)
                        lo = ld_shared_v4_u32(row_at + (((Int32(2) * j) ^ sw7) << Int32(4)))
                        hi = ld_shared_v4_u32(row_at + (((Int32(2) * j + Int32(1)) ^ sw7) << Int32(4)))
                        for i in cutlass.range_constexpr(4):
                            xa[8 * h + i], xa[8 * h + 4 + i] = lo[i], hi[i]
                    for gi in cutlass.range_constexpr(NG):
                        f = 4 * (mt * NG + gi)
                        for t in cutlass.range_constexpr(4):
                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = d0, d1, d2, d3
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for mt in cutlass.range_constexpr(MT):
                for gi in cutlass.range_constexpr(NG):
                    f = 4 * (mt * NG + gi)
                    row = m0 + wm * Int32(16 * MT) + Int32(16 * mt) + g
                    col = n0 + wn * Int32(8 * NG) + Int32(8 * gi) + Int32(2) * j
                    self._store(output, scales, acc[f], acc[f + 1], row, col, rows)
                    self._store(output, scales, acc[f + 2], acc[f + 3], row + Int32(8), col, rows)
            tile += grid
