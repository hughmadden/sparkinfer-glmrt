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
from b12x.moe._shared.kernels.w4a8_staging import scale_code_word


class StreamMxfp4DownA8:
    def __init__(self, *, hidden: int, inter: int, experts: int, round_output_bf16: bool = True,
                 native_resident: bool = False, native_exact: bool = False, scale_codes: bool = False):
        self.native_resident, self.native_exact = bool(native_resident), bool(native_exact)
        self.scale_codes = bool(scale_codes)
        if self.scale_codes and not self.native_resident:
            raise ValueError("compact scale codes require native resident weights")
        self.round_output_bf16 = bool(round_output_bf16)
        self.plan = Mxfp4DownA8Plan(int(hidden), int(inter), int(experts))
        self.hidden, self.inter, self.experts = self.plan.hidden, self.plan.inter, self.plan.experts
        self.blocks = (self.inter + self.plan.k_block - 1) // self.plan.k_block
        self.scale_row_bytes = mxfp4_scale_row_bytes(self.inter)

    def key(self) -> tuple:
        key = ("stream_down_mxfp4_a8", 2, self.hidden, self.inter, self.experts,
               self.native_resident, self.native_exact, self.scale_codes)
        return key if self.round_output_bf16 else key + (("f32_output",),)

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
    def _load(self, block, ring, a_rows: Int64, w_rows: Int64, s_rows: Int64, tidx, expert, n0):
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
        if const_expr(self.native_resident):
            stored = self.inter if const_expr(self.native_exact) else (self.inter + 127) // 128 * 128
            base = w_rows - (Int64(expert) * self.hidden + n0) * (self.inter // 2)
            for q in cutlass.range_constexpr(p.tile_n * (p.k_block // 32) // p.threads):
                index = tidx + Int32(q * p.threads)
                kb = index // 128
                nc = (index // 32) % 4
                lane = index % 32
                k = Int32(block) * 4 + kb
                dest = stage + Int32(p.a_bytes) + index * 16
                if k < self.inter // 32:
                    if const_expr(self.native_exact):
                        word = Int64(expert) * (self.hidden * stored // 8) + Int64(n0 * (stored // 8) + k * 512 + nc * 128 + lane * 4)
                    else:
                        n = n0 + nc * 32
                        word = Int64(expert) * (self.hidden * stored // 8) + Int64(((n // 256) * (stored // 128) + k // 4) * 4096 + (k % 4) * 1024 + (n % 256 // 32) * 128 + lane * 4)
                    cp_async4_shared_global(dest, base + word * 4)
                else:
                    for j in cutlass.range_constexpr(4):
                        st_shared_u32(dest + Int32(j * 4), Uint32(0))
        else:
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
            if const_expr(self.native_resident):
                stored = self.inter if const_expr(self.native_exact) else (self.inter + 127) // 128 * 128
                base = s_rows - (Int64(expert) * self.hidden + n0) * self.scale_row_bytes
                row = n0 + tidx
                if const_expr(self.scale_codes):
                    tiles = (self.inter // 32 + 3) // 4
                    scale_ptr = cute.make_ptr(cutlass.Uint8, base, cute.AddressSpace.gmem, assumed_align=1)
                    scales = cute.make_tensor(scale_ptr, cute.make_layout(
                        (self.experts * self.hidden * (2 * tiles + 1),)))
                    packed = scale_code_word(scales, Int64(expert), Int64(row), Int64(block),
                                             self.hidden, self.inter // 32)
                else:
                    packed = Uint32(0)
                    for j in cutlass.range_constexpr(4):
                        k = Int32(block) * 4 + Int32(j)
                        if k < self.inter // 32:
                            if const_expr(self.native_exact):
                                columns = self.inter // 32
                                count = Int32(4)
                                if const_expr(self.inter % 128 != 0):
                                    if k // 4 == self.inter // 128:
                                        count = Int32(columns % 4)
                                offset = Int64(expert * self.hidden * columns + (row // 128) * 128 * columns + (k // 4) * 512 + (row % 128) * count + k % 4)
                            else:
                                columns = stored // 32
                                offset = Int64(expert * self.hidden * columns + ((row // 256) * (columns // 4) + k // 4) * 1024 + (row % 256) * 4 + k % 4)
                            ptr = cute.make_ptr(cutlass.Uint8, base + offset, cute.AddressSpace.gmem, assumed_align=1)
                            packed |= Uint32(ptr[0]) << Uint32(j * 8)
                st_shared_u32(at + Int32(p.s_bytes), packed)
            else:
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
                    if const_expr(self.native_resident):
                        index = Int32(ks * 512) + col // 32 * 128 + col % 8 * 16 + col % 32 // 8
                        v0 = ld_shared_u32(sw + index * 4)
                        v1 = ld_shared_u32(sw + (index + 4) * 4)
                        v2 = ld_shared_u32(sw + (index + 8) * 4)
                        v3 = ld_shared_u32(sw + (index + 12) * 4)
                    else:
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
            self._load(0, ring, a_rows, w_rows, s_rows, tidx, e, n0)
            if const_expr(self.blocks > 1):
                self._load(1, ring, a_rows, w_rows, s_rows, tidx, e, n0)
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
                    self._load(block + Int32(2), ring, a_rows, w_rows, s_rows, tidx, e, n0)
            if const_expr(not self.round_output_bf16):
                if active:
                    output = cute.make_ptr(Float32, Int64(y.toint()), cute.AddressSpace.gmem, assumed_align=4)
                    for mt in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                            if row < live:
                                for nt in cutlass.range_constexpr(8):
                                    f = 4 * (8 * mt + nt) + 2 * half
                                    col = n0 + warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                                    offset = (Int64(first) + Int64(row)) * self.hidden + Int64(col)
                                    output[offset] = acc[f]
                                    output[offset + 1] = acc[f + 1]
            else:
                if active:
                    for mt in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                                # Reuse the drained ring for the coalesced BF16 epilogue.
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
