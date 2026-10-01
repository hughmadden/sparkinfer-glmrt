"""Expert-stationary streaming GEMMs over NVFP4 weights (fp8_moe ``stream``
route, ``weights="nvfp4"``): the large-row route of NVIDIA ModelOpt NVFP4
experts (W4A16).

Rows are grouped by expert without padding and cut into 128-row chunks (as
the FP8/MXFP4 stream routes). ``ChunkRows`` gathers each chunk's input rows
into a chunk-padded BF16 copy (from BF16 rows, or exactly from FP8 K32 wire
rows); ``StreamNvfp4Linear`` is ``StreamMxfp4Down`` over NVFP4 weights:
``y[p] = alpha_e * (a_p . bf16(e2m1 * e4m3)^T)`` for every chunk row, the
weight tile streamed once per chunk by TMA in whole 128-byte lines (or 64),
the CTA's E4M3 scale rows loaded once, each 32-K step widened exactly in
shared memory (one E4M3 scale per 16 values) while the previous one
multiplies (m16n8k16 BF16 MMAs, FP32 accumulation), the expert's FP32 alpha
applied before the one BF16 rounding. Gate and up are two such GEMMs over
the gathered rows (``[pairs, I]`` each, grouped rows), ``ChunkSwiGLU`` writes
the chunk-padded ``act`` rows down reads, and down is a third.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    cp_async_u32_shared_global,
    div_rn_f32,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._fp8_moe_kernels import GatherRows, _ceil, _i32_at, _ld_u8, _ld_v2_u32, _round_bf16, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from ._mxfp4_moe_kernels import e2m1x8_scaled_bf16
from ._mxfp4_moe_stream import StreamMxfp4Down
from ._nvfp4_moe_kernels import e4m3_bf16x2, nvfp4_alpha_offset

__all__ = ["ChunkRows", "ChunkSwiGLU", "StreamNvfp4Linear"]


class StreamNvfp4Linear(StreamMxfp4Down):
    """``y[p] = alpha_e * (a_p . bf16(w_e)^T)`` over NVFP4 ``w [E, N, K/2]``
    with E4M3 ``s [E, N, K/16]`` then FP32 alphas ``[E]``; ``a`` BF16 ``[max_tiles
    * 128, K]`` in chunk-padded rows, ``y`` BF16 ``[pairs, N]`` in grouped rows.
    ``hidden`` is N and ``inter`` K, as ``StreamFp8Down`` names them (down:
    N = H, K = I; gate/up: N = I, K = H). Grid ``(N / 128, max_tiles)``."""

    def __init__(self, *, n: int, k: int, experts: int):
        super().__init__(hidden=n, inter=k, experts=experts)
        self.scale_cols = self.inter // 16
        if self.scale_cols % 4:
            raise ValueError("NVFP4 stream linear needs K % 64 == 0")
        # Scales stream with their weight block (whole rows of K/16 bytes would not
        # fit SM120's shared memory at K 4096): per row, two slots of w_k/16 bytes.
        self.block_scales = self.w_k // 16
        self.sc_stride = 2 * self.block_scales

    def key(self) -> tuple:
        return ("stream_linear_nvfp4", 2, self.hidden, self.inter, self.experts, self.ring, self.w_k)

    @cute.kernel
    def kernel(self, meta: cute.Pointer, act: cute.Tensor, weight: cute.Tensor, s2: cute.Pointer, y: cute.Pointer,
               tma_a: cute.CopyAtom, tma_w: cute.CopyAtom, a_layout: cute.ComposedLayout,
               w_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            mbar = storage.mbar.data_ptr()
            s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            wbar = mbar + self.ring
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sb = shared_ptr_to_u32(storage.b.data_ptr())
            sc = shared_ptr_to_u32(storage.sc.data_ptr())
            h = self.hidden
            w_tile = e * Int32(h // self.tile_n) + n_blk
            # The expert's alpha (weight_scale_2) follows the E4M3 grid of every expert.
            alpha = _ld_f32(Int64(s2.toint()) + Int64(nvfp4_alpha_offset(self.experts, h, self.inter))
                            + Int64(e) * Int64(4))
            t_as, t_ag = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                                               cute.group_modes(cute.local_tile(act, (STREAM_TILE_M, self.k_step),
                                                                                (None, None)), 0, 2))
            t_ws, t_wg = cpasync.tma_partition(tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight, (self.tile_n, self.w_row),
                                                                                (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_a)
                cpasync.prefetch_descriptor(tma_w)
                for slot in cutlass.range_constexpr(self.ring + 2):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            # The E4M3 scales of weight blocks 0 and 1 (cp.async groups; each
            # later block's scales go out when its predecessor starts).
            blocks = self.k_steps // self.per_block
            s_rows = Int64(s2.toint()) + (Int64(e) * Int64(h) + Int64(n_blk * Int32(self.tile_n))) \
                * Int64(self.scale_cols)
            for p in cutlass.range_constexpr(min(2, self.k_steps // self.per_block)):
                self._load_scales(p, s_rows, sc, tidx)
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(min(2, blocks)):
                    self._load_w(p, tma_w, t_wg, t_ws, wbar, w_tile)
                for p in cutlass.range_constexpr(self.ring - 1):
                    if Int32(p) < Int32(self.k_steps):
                        self._load_a(p, tma_a, t_ag, t_as, mbar, tile)
            cute.arch.cp_async_wait_group(0)
            cute.arch.mbarrier_wait(wbar, 0)
            cute.arch.sync_threads()
            self._widen(0, sw, sb, sc, tidx)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g = lane // Int32(4)
            j = lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)
            for step in cutlass.range(self.k_steps, unroll=1):
                nxt = step + Int32(1)
                block = nxt // Int32(self.per_block)
                starts = (nxt < Int32(self.k_steps)) & (nxt % Int32(self.per_block) == Int32(0))
                cute.arch.mbarrier_wait(mbar + step % Int32(self.ring), (step // Int32(self.ring)) & Int32(1))
                if starts:
                    cute.arch.mbarrier_wait(wbar + block % Int32(2), (block // Int32(2)) & Int32(1))
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if starts & (block + Int32(1) < Int32(blocks)):
                    self._load_scales(block + Int32(1), s_rows, sc, tidx)
                if warp_id == Int32(0):
                    if step + Int32(self.ring - 1) < Int32(self.k_steps):
                        self._load_a(step + Int32(self.ring - 1), tma_a, t_ag, t_as, mbar, tile)
                    if starts & (block + Int32(1) < Int32(blocks)):
                        self._load_w(block + Int32(1), tma_w, t_wg, t_ws, wbar, w_tile)
                if nxt < Int32(self.k_steps):
                    self._widen(nxt, sw, sb, sc, tidx)
                if active:
                    a_base = sa + (step % Int32(self.ring)) * Int32(self.a_bytes)
                    b_base = sb + (step % Int32(2)) * Int32(self.b_bytes)
                    for kk in cutlass.range_constexpr(2):
                        a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                            chunk = Int32(2 * kk) + lane // Int32(16)
                            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                a_base + row * Int32(64) + ((chunk ^ ((row >> Int32(1)) & Int32(3))) * Int32(16)))
                            a[4 * mt] = a0
                            a[4 * mt + 1] = a1
                            a[4 * mt + 2] = a2
                            a[4 * mt + 3] = a3
                        for p in cutlass.range_constexpr(4):
                            nrow = warp_n * Int32(64) + Int32(16 * p) + (lane & Int32(7)) \
                                + (lane >> Int32(4)) * Int32(8)
                            chunk = Int32(2 * kk) + ((lane >> Int32(3)) & Int32(1))
                            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                                b_base + nrow * Int32(64) + ((chunk ^ ((nrow >> Int32(1)) & Int32(3))) * Int32(16)))
                            for mt in cutlass.range_constexpr(2):
                                for q in cutlass.range_constexpr(2):
                                    f = 4 * (8 * mt + 2 * p + q)
                                    lo = b0 if q == 0 else b2
                                    hi = b1 if q == 0 else b3
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], lo, hi)
                                    acc[f] = d0
                                    acc[f + 1] = d1
                                    acc[f + 2] = d2
                                    acc[f + 3] = d3
            cute.arch.sync_threads()
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                            st_shared_u32(sa + row * Int32(self.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f] * alpha, acc[f + 1] * alpha))
            cute.arch.sync_threads()
            n0 = n_blk * Int32(self.tile_n)
            segments = self.tile_n * 2 // 16
            for q in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
                c = tidx + Int32(self.threads * q)
                row = c // Int32(segments)
                seg = c % Int32(segments)
                if row < live:
                    v0, v1, v2, v3 = ld_shared_v4_u32(sa + row * Int32(self.out_stride) + seg * Int32(16))
                    st_global_v4_u32(Int64(y.toint()) + (Int64(first + row) * Int64(h) + Int64(n0)) * Int64(2)
                                     + Int64(seg) * Int64(16), v0, v1, v2, v3)

    @cute.jit
    def _load_scales(self, block, s_rows: Int64, sc, tidx):
        words = self.block_scales // 4
        slot = (Int32(block) % Int32(2)) * Int32(self.block_scales)
        for q in cutlass.range_constexpr((self.tile_n * words + self.threads - 1) // self.threads):
            u = tidx + Int32(q * self.threads)
            if u < Int32(self.tile_n * words):
                r = u // Int32(words)
                c = u % Int32(words)
                cp_async_u32_shared_global(
                    sc + r * Int32(self.sc_stride) + slot + c * Int32(4),
                    s_rows + Int64(r) * Int64(self.scale_cols) + Int64(block) * Int64(self.block_scales)
                    + Int64(c) * Int64(4))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _widen(self, step, sw, sb, sc, tidx):
        block = Int32(step) // Int32(self.per_block)
        piece = Int32(step) % Int32(self.per_block)
        w_stage = sw + (block % Int32(2)) * Int32(self.w_bytes)
        b_stage = sb + (Int32(step) % Int32(2)) * Int32(self.b_bytes)
        for q in cutlass.range_constexpr(self.tile_n * self.k_step // 8 // self.threads):
            c = tidx + Int32(self.threads * q)
            n = c // Int32(4)
            chunk = c % Int32(4)
            # Eight E2M1 codes (4 bytes) of the row's 16-byte unit `piece` (swizzled 128B / 64B).
            if cutlass.const_expr(self.w_row == 128):
                unit = piece ^ (n % Int32(8))
            else:
                unit = piece ^ ((n >> Int32(1)) & Int32(3))
            codes = ld_shared_u32(w_stage + n * Int32(self.w_row) + unit * Int32(16) + chunk * Int32(4))
            # One E4M3 scale per 16 K: byte 2 * piece + chunk / 2 of the block's slot.
            group = piece * Int32(2) + chunk // Int32(2)
            word = ld_shared_u32(sc + n * Int32(self.sc_stride) + (block % Int32(2)) * Int32(self.block_scales)
                                 + (group // Int32(4)) * Int32(4))
            code = (word >> (Uint32(group % Int32(4)) * Uint32(8))) & Uint32(0xFF)
            v0, v1, v2, v3 = e2m1x8_scaled_bf16(codes, e4m3_bf16x2(code))
            st_shared_v4_u32(b_stage + n * Int32(64) + ((chunk ^ ((n >> Int32(1)) & Int32(3))) * Int32(16)),
                             v0, v1, v2, v3)


class ChunkRows(GatherRows):
    """Chunk-padded BF16 input rows: row ``128 t + r`` of chunk ``t`` is the
    source row of grouped row ``first_t + r`` (``r < live_t``; others are left
    as they are and never reach a stored output). ``src`` BF16 ``[rows, K]``
    or FP8 K32 wire rows (exact). Grid: one CTA per chunk-padded row."""

    def __init__(self, *, k: int, wire: bool, experts: int):
        super().__init__(k=k, wire=wire, gather=True)
        self.experts = int(experts)

    def key(self) -> tuple:
        return ("chunk_rows", self.k, self.wire, self.experts)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, dst: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, dst).launch(grid=(max_tiles * Int32(STREAM_TILE_M), 1, 1),
                                                     block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, dst: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        tile = i // Int32(STREAM_TILE_M)
        r = i % Int32(STREAM_TILE_M)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                source = _i32_at(Int64(pair_row.toint()), first + r)
                out = Int64(dst.toint()) + Int64(i) * Int64(self.k * 2)
                vectors = self.k // 8
                for it in cutlass.range_constexpr(_ceil(vectors, self.threads)):
                    v = Int32(it * self.threads) + tidx
                    if v < Int32(vectors):
                        if const_expr(self.wire):
                            row = Int64(src.toint()) + Int64(source) * Int64(self.k + self.k // 32)
                            lo, hi = _ld_v2_u32(row + Int64(v) * Int64(8))
                            exponent = _ld_u8(row + Int64(self.k) + Int64(v // Int32(4)))
                            scale = _u32_as_f32(exponent << Uint32(23))
                            w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, scale)
                            st_global_v4_u32(out + Int64(v) * Int64(16), w0, w1, w2, w3)
                        else:
                            lo, hi = _ld_v2_u32(Int64(src.toint()) + (Int64(source) * Int64(self.k)
                                                                     + Int64(v) * Int64(8)) * Int64(2))
                            lo2, hi2 = _ld_v2_u32(Int64(src.toint()) + (Int64(source) * Int64(self.k)
                                                                       + Int64(v) * Int64(8)) * Int64(2) + Int64(8))
                            st_global_v4_u32(out + Int64(v) * Int64(16), lo, hi, lo2, hi2)


class ChunkSwiGLU:
    """``act[128 t + r] = bf16(bf16(silu(g)) * u)`` from grouped rows
    ``first_t + r`` of ``g``/``u`` ``[pairs, I]`` (clamped as ``MoeSwiGLU``
    when ``limit > 0``), chunk-padded for the down GEMM."""

    threads = 256

    def __init__(self, *, inter: int, experts: int, limit: float = 0.0):
        self.inter, self.experts, self.limit = int(inter), int(experts), float(limit)

    def key(self) -> tuple:
        return ("chunk_swiglu", self.inter, self.experts, self.limit)

    @cute.jit
    def __call__(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, act: cute.Pointer, max_tiles: Int32,
                 stream: cuda.CUstream):
        self.kernel(g, u, meta, act).launch(grid=(max_tiles * Int32(STREAM_TILE_M), _ceil(self.inter, self.threads), 1),
                                            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, act: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        col = Int32(cute.arch.block_idx()[1]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        tile = i // Int32(STREAM_TILE_M)
        r = i % Int32(STREAM_TILE_M)
        if (tile < _i32_at(Int64(meta.toint()), 1)) & (col < Int32(self.inter)):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                at = Int64(first + r) * Int64(self.inter) + Int64(col)
                gate = Float32(cute.make_tensor(g, cute.make_layout((Int64(1) << Int64(40),)))[at])
                up = Float32(cute.make_tensor(u, cute.make_layout((Int64(1) << Int64(40),)))[at])
                if const_expr(self.limit > 0.0):
                    gate = cutlass.select_(gate > Float32(self.limit), Float32(self.limit), gate)
                    up = cutlass.select_(up > Float32(self.limit), Float32(self.limit), up)
                    up = cutlass.select_(up < Float32(-self.limit), Float32(-self.limit), up)
                silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
                dst = cute.make_tensor(act, cute.make_layout((Int64(1) << Int64(40),)))
                dst[Int64(i) * Int64(self.inter) + Int64(col)] = (silu * up).to(BFloat16)
