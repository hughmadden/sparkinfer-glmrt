"""Opt-in streaming down GEMM over padded MXFP8 rows and checkpoint MXFP4 weights."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics import (
    cp_async4_shared_global, cp_async4_shared_global_pred, cp_async_u32_shared_global, e2m1x8_to_qmma_e2m1x8,
    ld_shared_u32, ld_shared_v4_u32, ldmatrix_m8n8x4_b16, mxfp8_mma_m16n8k32_f32_e2m1,
    pack_f32x2_to_bfloat2, shared_ptr_to_u32, st_global_v4_u32, st_shared_u32,
)
from ._fp8_moe_kernels import _i32_at
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._mxfp4_down_a8_plan import Mxfp4DownA8Plan, mxfp4_scale_row_bytes


class StreamMxfp4DownA8:
    def __init__(self, *, hidden: int, inter: int, experts: int):
        self.plan = Mxfp4DownA8Plan(int(hidden), int(inter), int(experts))
        self.hidden, self.inter, self.experts = self.plan.hidden, self.plan.inter, self.plan.experts
        self.blocks = (self.inter + self.plan.k_block - 1) // self.plan.k_block
        self.scale_row_bytes = mxfp4_scale_row_bytes(self.inter)

    def key(self) -> tuple:
        return ("stream_down_mxfp4_a8", 2, self.hidden, self.inter, self.experts)

    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.plan.stage_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, act: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(act, meta, w, s, y, self._storage()).launch(
            grid=(self.hidden // self.plan.tile_n, max_tiles, 1),
            block=(self.plan.threads, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.jit
    def _load(self, block, ring, a_rows: Int64, w_rows: Int64, s_rows: Int64, tidx):
        p = self.plan
        stage = ring + (Int32(block) % Int32(2)) * Int32(p.stage_bytes)
        for q in cutlass.range_constexpr(STREAM_TILE_M * (p.k_block // 16) // p.threads):
            c = tidx + Int32(q * p.threads)
            row = c // Int32(p.k_block // 16)
            piece = c % Int32(p.k_block // 16)
            dest = stage + row * Int32(p.a_stride) + piece * Int32(16)
            source = (a_rows + Int64(row) * Int64(p.row_bytes) + Int64(block) * Int64(p.k_block)
                      + Int64(piece) * Int64(16))
            if const_expr(self.inter % p.k_block):
                live = Int32(block) * Int32(p.k_block) + piece * Int32(16) < Int32(self.inter)
                cp_async4_shared_global_pred(dest, source, live.to(Int32))
            else:
                cp_async4_shared_global(dest, source)
        for q in cutlass.range_constexpr(p.tile_n * (p.k_block // 32) // p.threads):
            c = tidx + Int32(q * p.threads)
            row = c // Int32(p.k_block // 32)
            piece = c % Int32(p.k_block // 32)
            dest = stage + Int32(p.a_bytes) + row * Int32(p.w_stride) + piece * Int32(16)
            source = (w_rows + Int64(row) * Int64(self.inter // 2) + Int64(block) * Int64(p.k_block // 2)
                      + Int64(piece) * Int64(16))
            if const_expr(self.inter % p.k_block):
                live = Int32(block) * Int32(p.k_block) + piece * Int32(32) < Int32(self.inter)
                cp_async4_shared_global_pred(dest, source, live.to(Int32))
            else:
                cp_async4_shared_global(dest, source)
        # Four K32 scales per 128-K block; rows are padded to keep each word aligned.
        if tidx < Int32(STREAM_TILE_M):
            at = stage + Int32(p.a_bytes + p.w_bytes) + tidx * Int32(p.s_stride)
            cp_async_u32_shared_global(at, a_rows + Int64(tidx) * Int64(p.row_bytes) + Int64(self.inter)
                                       + Int64(block) * Int64(4))
            cp_async_u32_shared_global(at + Int32(p.s_bytes),
                                       s_rows + Int64(tidx) * Int64(self.scale_row_bytes) + Int64(block) * Int64(4))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _mma_stage(self, block, stage, acc: cute.Tensor, warp_m, warp_n, lane):
        p = self.plan
        sa = stage
        sw = stage + Int32(p.a_bytes)
        ssa = sw + Int32(p.w_bytes)
        ssb = ssa + Int32(p.s_bytes)
        g = lane // Int32(4)
        j = lane % Int32(4)
        shift = (Uint32(j) % Uint32(2)) * Uint32(16)
        for ks in cutlass.range_constexpr(p.k_block // 32):
            live_k = Int32(1)
            if const_expr(self.inter % p.k_block):
                live_k = (Int32(block) * Int32(p.k_block // 32) + Int32(ks) < Int32(self.inter // 32)).to(Int32)
            if live_k != Int32(0):
                a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
                for mt in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt) + lane % Int32(16)
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        sa + row * Int32(p.a_stride) + Int32(32 * ks) + lane // Int32(16) * Int32(16))
                    a[4 * mt] = a0
                    a[4 * mt + 1] = a1
                    a[4 * mt + 2] = a2
                    a[4 * mt + 3] = a3
                    srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
                    asc[mt] = ld_shared_u32(ssa + srow * Int32(p.s_stride))
                for nt in cutlass.range_constexpr(8):
                    col = warp_n * Int32(64) + Int32(8 * nt) + g
                    v0, v1, v2, v3 = ld_shared_v4_u32(sw + col * Int32(p.w_stride) + Int32(16 * ks))
                    lo, hi = v0, v2
                    if j >= Int32(2):
                        lo, hi = v1, v3
                    packed = ((lo >> shift) & Uint32(0xFFFF)) | (((hi >> shift) & Uint32(0xFFFF)) << Uint32(16))
                    b0, b1 = e2m1x8_to_qmma_e2m1x8(packed)
                    bsc = ld_shared_u32(ssb + col * Int32(p.s_stride))
                    for mt in cutlass.range_constexpr(2):
                        f = 4 * (8 * mt + nt)
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                            a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b0, b1,
                            asc[mt], bsc, bid_a=ks, bid_b=ks)
                        acc[f] = d0
                        acc[f + 1] = d1
                        acc[f + 2] = d2
                        acc[f + 3] = d3

    @cute.kernel
    def kernel(self, act: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer, y: cute.Pointer,
               Storage: cutlass.Constexpr):
        p = self.plan
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id, lane = tidx // Int32(32), tidx % Int32(32)
            ring = shared_ptr_to_u32(storage.ring.data_ptr())
            n0 = n_blk * Int32(p.tile_n)
            a_rows = Int64(act.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(p.row_bytes)
            w_rows = Int64(w.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.inter // 2)
            s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.scale_row_bytes)
            self._load(0, ring, a_rows, w_rows, s_rows, tidx)
            if const_expr(self.blocks > 1):
                self._load(1, ring, a_rows, w_rows, s_rows, tidx)
            warp_m, warp_n = warp_id // Int32(2), warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g, j = lane // Int32(4), lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)
            for block in cutlass.range(self.blocks, unroll=1):
                if block + Int32(1) < Int32(self.blocks):
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if active:
                    self._mma_stage(block, ring + (block % Int32(2)) * Int32(p.stage_bytes), acc, warp_m, warp_n, lane)
                cute.arch.sync_threads()
                if block + Int32(2) < Int32(self.blocks):
                    self._load(block + Int32(2), ring, a_rows, w_rows, s_rows, tidx)
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                            # Reuse the drained ring for the baseline's coalesced
                            # BF16 epilogue; its output fits within two stages.
                            st_shared_u32(ring + row * Int32(p.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
            cute.arch.sync_threads()
            segments = p.tile_n * 2 // 16
            for q in cutlass.range_constexpr(STREAM_TILE_M * segments // p.threads):
                c = tidx + Int32(p.threads * q)
                row, seg = c // Int32(segments), c % Int32(segments)
                if row < live:
                    v0, v1, v2, v3 = ld_shared_v4_u32(ring + row * Int32(p.out_stride) + seg * Int32(16))
                    st_global_v4_u32(Int64(y.toint()) + ((Int64(first) + Int64(row)) * Int64(self.hidden)
                                                        + Int64(n0)) * Int64(2) + Int64(seg) * Int64(16),
                                     v0, v1, v2, v3)
