"""W4A4 kernels of the NVFP4 stream route (``fp8_moe`` ``weights="nvfp4"``,
``activations="a4"``): the numerics NVIDIA calibrated its ModelOpt NVFP4
checkpoints for.

Activations are quantized to NVFP4 with the checkpoint's static
``input_scale`` (one per projection and expert, stored after the experts'
alphas in each scale operand): per 16 values an E4M3 block scale
``e4m3(amax * gs / 6)`` with ``gs = 1 / input_scale``, values rounded to E2M1
(nearest, ties to even) after division by ``scale / gs``
(``quantize_block_fp4``, TensorRT-LLM / ModelOpt semantics). The GEMMs run the
block-scaled FP4 MMA ``kind::mxf4nvf4.scale_vec::4X.m16n8k64`` over packed
activations and packed weights with both operands' E4M3 scales, and multiply
the FP32 accumulator by ``alpha_e * input_scale_e`` before the one BF16
rounding.

``ChunkRowsA4``       chunk-padded packed activations ``[max_tiles * 128, K/2]``
    and E4M3 scales ``[max_tiles * 128, K/16]`` from BF16 or FP8 K32 wire
    rows (each chunk with its expert's gate ``input_scale``).
``ChunkSwiGLUA4``     ``act = bf16(bf16(silu(g)) * u)`` (clamped as the BF16
    route) quantized the same way with the expert's down ``input_scale``.
``StreamNvfp4LinearA4``  ``y[p] = alpha_e * in_e * (a_p . w_e^T)`` over the
    packed operands: a CTA owns a 128-row chunk and 128 output columns (8
    warps, 32 x 64 each), streams 256-K (or 128-K) blocks of both operands
    and their scales through a two-stage ``cp.async`` ring (padded rows:
    conflict-free fragment loads) and runs 16 MMAs per warp per 64 K.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64, const_expr

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    div_rn_f32,
    ld_shared_u32,
    max_abs_16,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp4,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
)

from ._fp8_moe_kernels import _ceil, _i32_at, _ld_u8, _ld_v2_u32, _round_bf16, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._nvfp4_moe_kernels import nvfp4_alpha_offset

__all__ = ["ChunkRowsA4", "ChunkSwiGLUA4", "StreamNvfp4LinearA4", "nvfp4_input_scale_offset"]


def nvfp4_input_scale_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the FP32 input scales in an NVFP4 scale operand (after the alphas)."""
    return nvfp4_alpha_offset(experts, rows, k) + 4 * int(experts)


@cute.jit
def _quantize_row16(values: cute.Tensor, gs: Float32, packed_at: Int64, scale_at: Int64):
    packed, scale = quantize_block_fp4(values, max_abs_16(values), gs)
    st_global_u64(packed_at, packed)
    st_global_u8(scale_at, scale)


class ChunkRowsA4:
    """Chunk-padded NVFP4 activations of the chunk rows (see module docstring);
    one CTA per chunk-padded row, one thread per 16-value group."""

    threads = 128

    def __init__(self, *, k: int, wire: bool, experts: int, scale_rows: int):
        self.k, self.wire, self.experts = int(k), bool(wire), int(experts)
        # The scale operand whose input scales quantize these rows (gate: rows I, K = H).
        self.in_offset = nvfp4_input_scale_offset(experts, scale_rows, k)
        if self.k % 32:
            raise ValueError("K must be a multiple of 32")

    def key(self) -> tuple:
        return ("chunk_rows_a4", self.k, self.wire, self.experts, self.in_offset)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, scales, aq, a_s).launch(
            grid=(max_tiles * Int32(STREAM_TILE_M), 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
               aq: cute.Pointer, a_s: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        tile = i // Int32(STREAM_TILE_M)
        r = i % Int32(STREAM_TILE_M)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                source = _i32_at(Int64(pair_row.toint()), first + r)
                gs = Float32(1.0) / _ld_f32(Int64(scales.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                groups = self.k // 16
                for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
                    grp = Int32(it * self.threads) + tidx
                    if grp < Int32(groups):
                        values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                        for half in cutlass.range_constexpr(2):
                            v = grp * Int32(2) + Int32(half)
                            if const_expr(self.wire):
                                row = Int64(src.toint()) + Int64(source) * Int64(self.k + self.k // 32)
                                lo, hi = _ld_v2_u32(row + Int64(v) * Int64(8))
                                exponent = _ld_u8(row + Int64(self.k) + Int64(v // Int32(4)))
                                w = _e4m3x8_scaled_bf16(lo, hi, _u32_as_f32(exponent << Uint32(23)))
                            else:
                                at = Int64(src.toint()) + (Int64(source) * Int64(self.k) + Int64(v) * Int64(8)) * Int64(2)
                                lo, hi = _ld_v2_u32(at)
                                lo2, hi2 = _ld_v2_u32(at + Int64(8))
                                w = (lo, hi, lo2, hi2)
                            for q in cutlass.range_constexpr(4):
                                values[8 * half + 2 * q] = _bf16_lo(w[q])
                                values[8 * half + 2 * q + 1] = _bf16_hi(w[q])
                        _quantize_row16(values, gs, Int64(aq.toint()) + Int64(i) * Int64(self.k // 2) + Int64(grp) * Int64(8),
                                        Int64(a_s.toint()) + Int64(i) * Int64(groups) + Int64(grp))


class ChunkSwiGLUA4:
    """``act[128 t + r]`` of grouped row ``first_t + r``: ``bf16(bf16(silu(g)) *
    u)`` from ``g``/``u`` ``[pairs, I]``, quantized to NVFP4 with the expert's
    down ``input_scale``; one thread per 16-value group."""

    threads = 128

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0):
        self.inter, self.experts, self.limit = int(inter), int(experts), float(limit)
        self.in_offset = nvfp4_input_scale_offset(experts, hidden, inter)

    def key(self) -> tuple:
        return ("chunk_swiglu_a4", self.inter, self.experts, self.limit, self.in_offset)

    @cute.jit
    def __call__(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(g, u, meta, scales, aq, a_s).launch(
            grid=(max_tiles * Int32(STREAM_TILE_M), _ceil(self.inter // 16, self.threads), 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer, aq: cute.Pointer,
               a_s: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        grp = Int32(cute.arch.block_idx()[1]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        tile = i // Int32(STREAM_TILE_M)
        r = i % Int32(STREAM_TILE_M)
        groups = self.inter // 16
        if (tile < _i32_at(Int64(meta.toint()), 1)) & (grp < Int32(groups)):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                gs = Float32(1.0) / _ld_f32(Int64(scales.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                at = (Int64(first + r) * Int64(self.inter) + Int64(grp) * Int64(16)) * Int64(2)
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                for half in cutlass.range_constexpr(2):
                    g0, g1 = _ld_v2_u32(Int64(g.toint()) + at + Int64(16 * half))
                    g2, g3 = _ld_v2_u32(Int64(g.toint()) + at + Int64(16 * half + 8))
                    u0, u1 = _ld_v2_u32(Int64(u.toint()) + at + Int64(16 * half))
                    u2, u3 = _ld_v2_u32(Int64(u.toint()) + at + Int64(16 * half + 8))
                    gw = (g0, g1, g2, g3)
                    uw = (u0, u1, u2, u3)
                    for q in cutlass.range_constexpr(4):
                        for side in cutlass.range_constexpr(2):
                            gate = _bf16_lo(gw[q]) if side == 0 else _bf16_hi(gw[q])
                            up = _bf16_lo(uw[q]) if side == 0 else _bf16_hi(uw[q])
                            if const_expr(self.limit > 0.0):
                                gate = cutlass.select_(gate > Float32(self.limit), Float32(self.limit), gate)
                                up = cutlass.select_(up > Float32(self.limit), Float32(self.limit), up)
                                up = cutlass.select_(up < Float32(-self.limit), Float32(-self.limit), up)
                            silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
                            values[8 * half + 2 * q + side] = _round_bf16(silu * up)
                _quantize_row16(values, gs, Int64(aq.toint()) + Int64(i) * Int64(self.inter // 2) + Int64(grp) * Int64(8),
                                Int64(a_s.toint()) + Int64(i) * Int64(groups) + Int64(grp))


class StreamNvfp4LinearA4:
    """``y[p] = alpha_e * in_e * (a_p . w_e^T)`` with packed NVFP4 ``a``
    ``[max_tiles * 128, K/2]`` + E4M3 ``[max_tiles * 128, K/16]`` (chunk-padded)
    and NVFP4 ``w [E, N, K/2]`` with E4M3 ``s [E, N, K/16]``, alphas ``[E]``,
    input scales ``[E]``; ``y`` BF16 ``[pairs, N]`` in grouped rows. Grid ``(N /
    128, max_tiles)``, 256 threads (warps 4 x 2: 32 rows x 64 columns)."""

    threads = 256
    tile_n = 128

    def __init__(self, *, n: int, k: int, experts: int):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        if self.n % self.tile_n or self.k % 128:
            raise ValueError("NVFP4 A4 linear needs N % 128 == 0 and K % 128 == 0")
        self.k_block = 256 if self.k % 256 == 0 else 128
        self.blocks = self.k // self.k_block
        self.row_bytes = self.k_block // 2
        # Padded rows (bytes): fragment loads of 8 rows x 4 lanes hit distinct banks.
        self.stride = self.row_bytes + 16
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = STREAM_TILE_M * self.stride
        self.sc_bytes = STREAM_TILE_M * self.s_stride
        self.stage_bytes = 2 * self.tile_bytes + 2 * self.sc_bytes
        self.alpha_offset = nvfp4_alpha_offset(experts, self.n, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("stream_linear_nvfp4_a4", 1, self.n, self.k, self.experts, self.k_block)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.stage_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(aq, a_s, meta, w, s, y).launch(grid=(self.n // self.tile_n, max_tiles, 1),
                                                   block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.jit
    def _load(self, block, ring, a_rows: Int64, as_rows: Int64, w_rows: Int64, s_rows: Int64, tidx):
        """Stage ``block % 2``: 128 rows of A and W (16-byte cp.async) and their scales (4-byte)."""
        stage = ring + (Int32(block) % Int32(2)) * Int32(self.stage_bytes)
        chunks = self.row_bytes // 16
        k_bytes = Int64(block) * Int64(self.row_bytes)
        for q in cutlass.range_constexpr(STREAM_TILE_M * chunks // self.threads):
            c = tidx + Int32(q * self.threads)
            row = c // Int32(chunks)
            piece = c % Int32(chunks)
            dst = row * Int32(self.stride) + piece * Int32(16)
            cp_async4_shared_global(stage + dst, a_rows + Int64(row) * Int64(self.k // 2) + k_bytes + Int64(piece) * Int64(16))
            cp_async4_shared_global(stage + Int32(self.tile_bytes) + dst,
                                    w_rows + Int64(row) * Int64(self.k // 2) + k_bytes + Int64(piece) * Int64(16))
        words = self.s_bytes // 4
        s_k = Int64(block) * Int64(self.s_bytes)
        for q in cutlass.range_constexpr(_ceil(STREAM_TILE_M * words, self.threads)):
            c = tidx + Int32(q * self.threads)
            if c < Int32(STREAM_TILE_M * words):
                row = c // Int32(words)
                word = c % Int32(words)
                dst = Int32(2 * self.tile_bytes) + row * Int32(self.s_stride) + word * Int32(4)
                cp_async_u32_shared_global(stage + dst, as_rows + Int64(row) * Int64(self.k // 16) + s_k
                                           + Int64(word) * Int64(4))
                cp_async_u32_shared_global(stage + dst + Int32(self.sc_bytes),
                                           s_rows + Int64(row) * Int64(self.k // 16) + s_k + Int64(word) * Int64(4))
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
               y: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            ring = shared_ptr_to_u32(storage.ring.data_ptr())
            n0 = n_blk * Int32(self.tile_n)
            a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
            as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
            w_rows = Int64(w.toint()) + (Int64(e) * Int64(self.n) + Int64(n0)) * Int64(self.k // 2)
            s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.n) + Int64(n0)) * Int64(self.k // 16)
            factor = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                * _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
            self._load(0, ring, a_rows, as_rows, w_rows, s_rows, tidx)
            if const_expr(self.blocks > 1):
                self._load(1, ring, a_rows, as_rows, w_rows, s_rows, tidx)
            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            q = lane // Int32(4)
            c = lane % Int32(4)
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
                    stage = ring + (block % Int32(2)) * Int32(self.stage_bytes)
                    sa = stage
                    sw = stage + Int32(self.tile_bytes)
                    ssa = stage + Int32(2 * self.tile_bytes)
                    ssb = ssa + Int32(self.sc_bytes)
                    for step in cutlass.range_constexpr(self.k_block // 64):
                        k_byte = 32 * step
                        sfa = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
                        a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            r0 = warp_m * Int32(32) + Int32(16 * mt) + q
                            sfa[mt] = ld_shared_u32(ssa + (r0 + Int32(8) * (c & Int32(1))) * Int32(self.s_stride)
                                                    + Int32(4 * step))
                            for reg in cutlass.range_constexpr(4):
                                row = r0 + Int32(8 * (reg % 2))
                                a[4 * mt + reg] = ld_shared_u32(sa + row * Int32(self.stride)
                                                                + Int32(k_byte + 16 * (reg // 2)) + c * Int32(4))
                        for nt in cutlass.range_constexpr(8):
                            col = warp_n * Int32(64) + Int32(8 * nt) + q
                            sfb = ld_shared_u32(ssb + col * Int32(self.s_stride) + Int32(4 * step))
                            b0 = ld_shared_u32(sw + col * Int32(self.stride) + Int32(k_byte) + c * Int32(4))
                            b1 = ld_shared_u32(sw + col * Int32(self.stride) + Int32(k_byte + 16) + c * Int32(4))
                            for mt in cutlass.range_constexpr(2):
                                f = 4 * (8 * mt + nt)
                                d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                    a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b0, b1, sfa[mt], sfb)
                                acc[f] = d0
                                acc[f + 1] = d1
                                acc[f + 2] = d2
                                acc[f + 3] = d3
                cute.arch.sync_threads()
                if block + Int32(2) < Int32(self.blocks):
                    self._load(block + Int32(2), ring, a_rows, as_rows, w_rows, s_rows, tidx)
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + q
                        if row < live:
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = n0 + warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * c
                                st_global_u32(Int64(y.toint()) + (Int64(first + row) * Int64(self.n) + Int64(col))
                                              * Int64(2), pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
