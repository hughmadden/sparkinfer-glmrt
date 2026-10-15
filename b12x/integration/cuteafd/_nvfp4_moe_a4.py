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
    packed operands: a CTA owns either a sparse 16-row tile with 32/64/128
    output columns, or a wide 128-row tile with 128 columns. Both stream
    256-K (or 128-K) operand blocks and scales through a two-stage
    ``cp.async`` ring with conflict-free ``ldmatrix`` loads. Complementary
    live-count masks select exactly one tile per compact expert chunk.
``StreamNvfp4GateUpA4``  gate/up, SwiGLU and the down-input quantization in
    one launch: half the CTA's weight rows are gate and half are matching
    up rows, both BF16-rounded exactly as the linear's output, exchanged
    through shared memory and written as ``ChunkSwiGLUA4`` writes them
    (bit-identical ``act`` bytes, no BF16 ``[pairs, 2I]`` round trip).
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
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    max_abs_16,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp4,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
    st_shared_u32,
)

from ._fp8_moe_kernels import _ceil, _i32_at, _ld_u8, _ld_v2_u32, _round_bf16, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._nvfp4_moe_kernels import nvfp4_alpha_offset

__all__ = ["ChunkRowsA4", "ChunkSwiGLUA4", "StreamNvfp4GateUpA4", "StreamNvfp4LinearA4", "nvfp4_input_scale_offset"]


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
    one CTA per live grouped pair, one thread per 16-value group."""

    threads = 128

    def __init__(self, *, k: int, wire: bool, experts: int, scale_rows: int):
        self.k, self.wire, self.experts = int(k), bool(wire), int(experts)
        # The scale operand whose input scales quantize these rows (gate: rows I, K = H).
        self.in_offset = nvfp4_input_scale_offset(experts, scale_rows, k)
        if self.k % 32:
            raise ValueError("K must be a multiple of 32")

    def key(self) -> tuple:
        return ("chunk_rows_a4", 2, self.k, self.wire, self.experts, self.in_offset)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, pairs: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, scales, aq, a_s).launch(
            grid=(pairs, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
               aq: cute.Pointer, a_s: cute.Pointer):
        p = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        # Locate the padded destination in the sorted compact chunk table;
        # one CTA per actual pair, not 128 mostly-empty CTAs per expert.
        tile = Int32(0)
        end = _i32_at(Int64(meta.toint()), 1)
        while tile + Int32(1) < end:
            mid = (tile + end) // Int32(2)
            _, start, _ = _chunk(meta, self.experts, mid)
            if start <= p:
                tile = mid
            else:
                end = mid
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            r = p - first
            i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(r)
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
    tile_n, max_tiles)``. Sparse tiles use two 16-row warps, wide tiles use
    eight 32-row warps. Operand staging has independent A and W row counts."""

    threads = 256
    tile_n = 128

    def __init__(self, *, n: int, k: int, experts: int, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.tile_m, self.tile_n = int(tile_m), int(tile_n)
        self.live_min, self.live_max = int(live_min), int(live_max)
        if self.tile_m not in (16, 128) or self.tile_n not in (32, 64, 128):
            raise ValueError("NVFP4 A4 tile needs M in (16,128), N in (32,64,128)")
        if self.tile_m == 128 and self.tile_n != 128:
            raise ValueError("the wide NVFP4 A4 tile needs N=128")
        self.threads = 64 if self.tile_m == 16 else 256
        self.warp_rows = 16 if self.tile_m == 16 else 32
        self.m_tiles = self.warp_rows // 16
        self.warp_cols = self.tile_n // 2
        self.n_tiles = self.warp_cols // 8
        self.frags = 4 * self.m_tiles * self.n_tiles
        if self.n % self.tile_n or self.k % 128:
            raise ValueError("NVFP4 A4 linear needs N % tile_n == 0 and K % 128 == 0")
        self.k_block = 256 if self.k % 256 == 0 else 128
        self.blocks = self.k // self.k_block
        self.row_bytes = self.k_block // 2
        # Unpadded rows with a 32B-sector-preserving XOR swizzle on 16B pieces:
        # both cp.async.cg.128 and ldmatrix.x4 hit all 32 banks without splitting
        # odd rows across 32-byte L2 sectors.
        self.stride = self.row_bytes
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * self.s_stride
        self.weight_sc_bytes = self.tile_n * self.s_stride
        self.stage_bytes = self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes
        self.alpha_offset = nvfp4_alpha_offset(experts, self.n, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("stream_linear_nvfp4_a4", 5, self.n, self.k, self.experts, self.k_block,
                self.tile_m, self.tile_n, self.live_min, self.live_max)

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
    def _swizzle(self, row):
        if const_expr(self.row_bytes >= 128):
            return ((row & Int32(3)) << Int32(1)) | ((row >> Int32(2)) & Int32(1))
        return (row >> Int32(1)) & Int32(3)

    @cute.jit
    def _load(self, block, ring, a_rows: Int64, as_rows: Int64, w_rows: Int64, s_rows: Int64, live: Int32, tidx):
        stage = ring + (Int32(block) % Int32(2)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.tile_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes, Int32(self.tile_n))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_operand(self, stage, rows_at: Int64, scales_at: Int64, block, tidx,
                      count: cutlass.Constexpr, packed_offset: cutlass.Constexpr, scale_offset: cutlass.Constexpr,
                      row_limit: Int32):
        """Stage ``count`` packed rows (and their scale records) of this K block.

        ``row_limit`` bounds the activation tile to the chunk's live rows: a
        sparse tile's M is 16 rows of a chunk that may hold as few as one, and
        every row staged is fetched by every N-split CTA of the chunk. Rows past
        the limit only feed output rows the epilogue drops, so leaving them out
        of the ring is bit-exact for every output the kernel writes. Weight rows
        (and their scales) always stage in full -- every one of them is read."""
        chunks = self.row_bytes // 16
        r0 = tidx // Int32(chunks)
        piece = tidx % Int32(chunks)
        smem_piece = piece ^ self._swizzle(r0)
        row_step = self.threads // chunks
        rows = row_limit
        for q in cutlass.range_constexpr(_ceil(count * chunks, self.threads)):
            row = r0 + Int32(q * row_step)
            if row < rows:
                cp_async4_shared_global(
                    stage + Int32(packed_offset + q * row_step * self.stride)
                    + r0 * Int32(self.stride) + smem_piece * Int32(16),
                    rows_at + Int64(row) * Int64(self.k // 2) + Int64(block) * Int64(self.row_bytes)
                    + Int64(piece) * Int64(16))
        words = self.s_bytes // 4
        for q in cutlass.range_constexpr(_ceil(count * words, self.threads)):
            c = tidx + Int32(q * self.threads)
            if c < rows * Int32(words):
                row = c // Int32(words)
                word = c % Int32(words)
                cp_async_u32_shared_global(stage + Int32(scale_offset) + row * Int32(self.s_stride) + word * Int32(4),
                                           scales_at + Int64(row) * Int64(self.k // 16) + Int64(block) * Int64(self.s_bytes)
                                           + Int64(word) * Int64(4))

    @cute.jit
    def _mma_stage(self, stage, acc: cute.Tensor, warp_m, warp_n, q, c, mat, mrow):
        """One K block of the 32 x 64 warp tile from ring stage ``stage``: A and W fragments
        by ``ldmatrix`` (lane ``t`` addresses row ``t % 8`` of matrix ``t / 8``), their E4M3
        scale words, 16 ``m16n8k64`` block-scaled FP4 MMAs per 64 K."""
        sa = stage
        sw = stage + Int32(self.tile_bytes)
        ssa = stage + Int32(self.tile_bytes + self.weight_bytes)
        ssb = ssa + Int32(self.sc_bytes)
        swz = self._swizzle(mrow)
        for step in cutlass.range_constexpr(self.k_block // 64):
            apiece = ((Int32(2 * step) + (mat >> Int32(1))) ^ swz) * Int32(16)
            bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ swz) * Int32(16)
            sfa = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Uint32)
            a = cute.make_rmem_tensor(cute.make_layout((4 * self.m_tiles,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(self.m_tiles):
                r0 = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + q
                sfa[mt] = ld_shared_u32(ssa + (r0 + Int32(8) * (c & Int32(1))) * Int32(self.s_stride)
                                        + Int32(4 * step))
                # ldmatrix x4: lane t addresses row t % 8 of matrix t / 8 (rows +8, then K +16 B).
                arow = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + Int32(8) * (mat & Int32(1)) + mrow
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sa + arow * Int32(self.stride) + apiece)
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            bw = cute.make_rmem_tensor(cute.make_layout((2 * self.n_tiles,), stride=(1,)), Uint32)
            for pair in cutlass.range_constexpr(self.n_tiles // 2):
                # Matrices: b0, b1 of column block 2 * pair, then of 2 * pair + 1.
                bcol = warp_n * Int32(self.warp_cols) + Int32(16 * pair) + Int32(8) * (mat >> Int32(1)) + mrow
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(sw + bcol * Int32(self.stride) + bpiece)
                bw[4 * pair] = b0
                bw[4 * pair + 1] = b1
                bw[4 * pair + 2] = b2
                bw[4 * pair + 3] = b3
            for nt in cutlass.range_constexpr(self.n_tiles):
                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + q
                sfb = ld_shared_u32(ssb + col * Int32(self.s_stride) + Int32(4 * step))
                b0 = bw[2 * nt]
                b1 = bw[2 * nt + 1]
                for mt in cutlass.range_constexpr(self.m_tiles):
                    f = 4 * (self.n_tiles * mt + nt)
                    d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b0, b1, sfa[mt], sfb)
                    acc[f] = d0
                    acc[f + 1] = d1
                    acc[f + 2] = d2
                    acc[f + 3] = d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
               y: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
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
                self._load(0, ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                if const_expr(self.blocks > 1):
                    self._load(1, ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                warp_m = warp_id // Int32(2)
                warp_n = warp_id % Int32(2)
                active = warp_m * Int32(self.warp_rows) < live
                q = lane // Int32(4)
                c = lane % Int32(4)
                mat = lane // Int32(8)
                mrow = lane % Int32(8)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(self.frags):
                    acc[v] = Float32(0.0)
                for block in cutlass.range(self.blocks, unroll=1):
                    if block + Int32(1) < Int32(self.blocks):
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if active:
                        stage = ring + (block % Int32(2)) * Int32(self.stage_bytes)
                        self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow)
                    cute.arch.sync_threads()
                    if block + Int32(2) < Int32(self.blocks):
                        self._load(block + Int32(2), ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            if row < live:
                                for nt in cutlass.range_constexpr(self.n_tiles):
                                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                    col = n0 + warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                    st_global_u32(Int64(y.toint()) + (Int64(first + row) * Int64(self.n) + Int64(col))
                                                  * Int64(2), pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))


@cute.jit
def _swiglu16(gw, uw, values: cute.Tensor, base: cutlass.Constexpr, limit: cutlass.Constexpr):
    """``values[base + j] = bf16(bf16(silu(g)) * u)`` of eight BF16 gate/up pairs (``ChunkSwiGLUA4``'s math)."""
    for q in cutlass.range_constexpr(4):
        for side in cutlass.range_constexpr(2):
            gate = _bf16_lo(gw[q]) if side == 0 else _bf16_hi(gw[q])
            up = _bf16_lo(uw[q]) if side == 0 else _bf16_hi(uw[q])
            if const_expr(limit > 0.0):
                gate = cutlass.select_(gate > Float32(limit), Float32(limit), gate)
                up = cutlass.select_(up > Float32(limit), Float32(limit), up)
                up = cutlass.select_(up < Float32(-limit), Float32(-limit), up)
            silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
            values[base + 2 * q + side] = _round_bf16(silu * up)


class StreamNvfp4GateUpA4(StreamNvfp4LinearA4):
    """Fused W4A4 gate/up + SwiGLU + NVFP4 quantization of the down input.

    Grid ``(I / half_n, max_tiles)``. A CTA streams its chunk's packed
    activations against ``tile_n`` weight rows, ``half_n`` gate rows of ``w1``
    then matching up rows of ``w3`` (warps ``warp_n`` 0 and 1), so the
    main loop is ``StreamNvfp4LinearA4``'s. Each warp scales its accumulators by
    its projection's ``alpha_e * in_e`` and rounds them to BF16 into shared
    memory (the ring, after the last K block); then each thread takes 16
    columns of one row, computes ``bf16(bf16(silu(g)) * u)`` and writes them as
    NVFP4 with the expert's down ``input_scale`` into ``aq [max_tiles * 128,
    I/2]`` / ``a_s [max_tiles * 128, I/16]``."""

    half_n = 64
    # Exchange rows: 64 gate then 64 up BF16 columns, padded (conflict-free stores).
    x_stride = 2 * 64 * 2 + 16

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0, k_block: int | None = None,
                 stages: int = 2, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128):
        super().__init__(n=2 * int(inter), k=hidden, experts=experts, tile_m=tile_m, tile_n=tile_n,
                         live_min=live_min, live_max=live_max)
        self.half_n = self.tile_n // 2
        self.x_stride = 2 * self.tile_n + 16
        self.inter, self.limit = int(inter), float(limit)
        if k_block is None:
            k_block = self.k_block
        if self.k % k_block or k_block % 64:
            raise ValueError(f"K {self.k} is not a multiple of the {k_block}-wide K block")
        self._set_k_block(k_block)
        self.stages = int(stages)
        if self.inter % self.half_n:
            raise ValueError("the fused NVFP4 A4 gate/up needs I % 64 == 0")
        # Gate and up scale operands share the [E, I, H/16] shape.
        self.alpha_offset = nvfp4_alpha_offset(experts, self.inter, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.inter, self.k)
        self.down_in_offset = nvfp4_input_scale_offset(experts, hidden, self.inter)
        if self.tile_m * self.x_stride > self.stages * self.stage_bytes:
            raise ValueError("the gate/up exchange does not fit the ring")

    def _set_k_block(self, k_block: int):
        self.k_block = int(k_block)
        self.blocks = self.k // self.k_block
        self.row_bytes = self.k_block // 2
        self.stride = self.row_bytes
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * self.s_stride
        self.weight_sc_bytes = self.tile_n * self.s_stride
        self.stage_bytes = self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes

    def key(self) -> tuple:
        return ("stream_gate_up_nvfp4_a4", 5, self.inter, self.k, self.experts, self.k_block, self.stages,
                self.limit, self.down_in_offset, self.tile_m, self.tile_n, self.live_min, self.live_max)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.stages * self.stage_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream):
        self.fused(aq, a_s, meta, w1, s1, w3, s3, s2, act, act_s).launch(
            grid=(self.inter // self.half_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.jit
    def _load2(self, block, ring, a_rows: Int64, as_rows: Int64, w_rows: Int64, w_up: Int64, s_rows: Int64,
               s_up: Int64, live: Int32, tidx):
        stage = ring + (Int32(block) % Int32(self.stages)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.half_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes, Int32(self.half_n))
        self._load_operand(stage, w_up, s_up, block, tidx, self.half_n,
                           self.tile_bytes + self.half_n * self.stride,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes + self.half_n * self.s_stride,
                           Int32(self.half_n))
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def fused(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
              w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                ring = shared_ptr_to_u32(storage.ring.data_ptr())
                n0 = n_blk * Int32(self.half_n)
                a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
                as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
                w_off = Int64(e) * Int64(self.inter) + Int64(n0)
                w_rows = Int64(w1.toint()) + w_off * Int64(self.k // 2)
                w_up = Int64(w3.toint()) + w_off * Int64(self.k // 2)
                s_rows = Int64(s1.toint()) + w_off * Int64(self.k // 16)
                s_up = Int64(s3.toint()) + w_off * Int64(self.k // 16)
                # ``stages``-deep cp.async ring, one group per K block (empty past the end): block b
                # waits for its group, the barrier frees stage (b - 1) % stages, which takes block
                # b + stages - 1.
                for pre in cutlass.range_constexpr(self.stages - 1):
                    if const_expr(pre < self.blocks):
                        self._load2(pre, ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up, live, tidx)
                    else:
                        cute.arch.cp_async_commit_group()
                warp_m = warp_id // Int32(2)
                warp_n = warp_id % Int32(2)
                # warp_n 0: gate columns (w1 / s1); 1: up columns (w3 / s3).
                sp = cutlass.select_(warp_n == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
                factor = _ld_f32(sp + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                    * _ld_f32(sp + Int64(self.in_offset) + Int64(e) * Int64(4))
                active = warp_m * Int32(self.warp_rows) < live
                q = lane // Int32(4)
                c = lane % Int32(4)
                mat = lane // Int32(8)
                mrow = lane % Int32(8)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(self.frags):
                    acc[v] = Float32(0.0)
                for block in cutlass.range(self.blocks, unroll=1):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    if block + Int32(self.stages - 1) < Int32(self.blocks):
                        self._load2(block + Int32(self.stages - 1), ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up,
                                    live, tidx)
                    else:
                        cute.arch.cp_async_commit_group()
                    if active:
                        stage = ring + (block % Int32(self.stages)) * Int32(self.stage_bytes)
                        self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow)
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                # The ring is free (every K block consumed, last barrier passed): exchange the
                # BF16-rounded gate / up tiles, row r at ring + r * x_stride (gate 0..127 B, up 128..255 B).
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                st_shared_u32(ring + row * Int32(self.x_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
                cute.arch.sync_threads()
                gs = Float32(1.0) / _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
                groups = self.half_n // 16
                for it in cutlass.range_constexpr(_ceil(self.tile_m * groups, self.threads)):
                    gi = tidx + Int32(it * self.threads)
                    row = gi // Int32(groups)
                    grp = gi % Int32(groups)
                    if row < live:
                        values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                        at = ring + row * Int32(self.x_stride) + grp * Int32(32)
                        for half in cutlass.range_constexpr(2):
                            gw = ld_shared_v4_u32(at + Int32(16 * half))
                            uw = ld_shared_v4_u32(at + Int32(self.half_n * 2 + 16 * half))
                            _swiglu16(gw, uw, values, 8 * half, self.limit)
                        i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                        col16 = Int64(n0 // Int32(16) + grp)
                        _quantize_row16(values, gs, Int64(act.toint()) + i * Int64(self.inter // 2) + col16 * Int64(8),
                                        Int64(act_s.toint()) + i * Int64(self.inter // 16) + col16)
