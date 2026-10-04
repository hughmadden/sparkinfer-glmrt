"""Expert-stationary streaming GEMMs of the fp8_moe ``stream`` route.

Prefill on weight-bandwidth-bound devices (GB10: ~235 GB/s, ~100 BF16 /
~180 FP8 dense TFLOP/s) wants every expert weight byte read from DRAM once
per layer. Rows are grouped by expert without padding (``MoePrep`` with
``pad=1``) and cut into 128-row chunks; the tile table entry of a chunk is
``expert | chunk << 16``. A CTA owns one chunk and one slice of output
columns and streams that slice of the expert's weights through shared memory
once; the chunk's input rows come from L2 (grid x, the column slices of one
chunk, is the fastest launch dimension, so they are co-resident).

``StreamFp8GateUp``  gate/up straight from the FP8 K32 wire rows (no
    gathered BF16 copy): the chunk's rows (E4M3 + UE8M0) and the weight tile
    (E4M3) are read with 8-bit ``ldmatrix`` fragments and widened in registers
    to the rows' exact BF16 values and to ``bf16(w * s)`` (the reference's
    weight), feeding m16n8k16 BF16 MMAs under one K permutation shared by both
    operands; FP32 accumulation. A CTA owns 32 gate and the matching 32 up
    columns and writes ``act = bf16(bf16(silu(bf16(g))) * bf16(u))`` (optional
    clamp) directly: no gate|up buffer, no SwiGLU pass.
    ``qmma`` (W8A8, the default): block-scaled E4M3 x E4M3 MMAs
    (``kind::mxf8f6f4`` m16n8k32) take the wire rows' UE8M0 scales as the A
    scales and unit B scales; each 128-K step accumulates into a fresh FP32
    partial that is promoted by the weights' FP32 block scale (a CTA's 32
    gate and 32 up columns sit in one 128-column block: one scale per K
    step). This computes ``x . (w * s)`` without the reference's BF16 weight
    rounding, i.e. closer to the checkpoint's exact weights than the widening
    kernel (GLM 5.3 layer 78, 1024 rows: cosine to exact FP64 weights
    0.999995 vs 0.999986; to the ``bf16(w * s)`` reference 0.999987).
``StreamFp8Down``    down over the BF16 ``act`` rows with the E4M3 weight
    tile widened to ``bf16(w * s)`` in shared memory (the reference's weight,
    double-buffered: tile ``t + 1`` is widened while tile ``t`` multiplies),
    m16n8k16 BF16 MMAs, FP32 accumulation, BF16 ``y`` per grouped row.

Pipelines (one CTA per SM, 8 warps, one barrier per K step, one mbarrier
per TMA ring slot): the weights stream by TMA in whole 128-byte lines per row
(a 128 x 32-byte box, one sector per row, read DRAM at ~70% of the
bandwidth); gate/up rows (indexed through ``pair_row``) come by ``cp.async``
three steps ahead, down's act rows (chunk-padded, contiguous) by TMA; block
scales are loaded once per CTA. Shared tiles are XOR-swizzled (TMA 128B/64B
swizzles) and read with ``ldmatrix``. The rows' L2 latency, not DRAM
latency, set the ring depths.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    div_rn_f32,
    ld_shared_f32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v4_u32,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._fp8_moe_kernels import META_HEAD, _i32_at, _round_bf16, _u32_as_f32
from ._fp8_weights import _e4m3x4_scaled_bf16x2x2, _e4m3x8_scaled_bf16

__all__ = ["STREAM_TILE_M", "StreamFp8Down", "StreamFp8GateUp", "stream_max_tiles"]

STREAM_TILE_M = 128


def stream_max_tiles(experts: int, pairs: int) -> int:
    """Upper bound of 128-row chunks over ``pairs`` grouped rows."""
    pairs = max(int(pairs), 1)
    return -(-pairs // STREAM_TILE_M) + min(int(experts), pairs)


@cute.jit
def _chunk(meta: cute.Pointer, experts: cutlass.Constexpr, tile: Int32):
    """(expert, first grouped row, live rows) of chunk ``tile``."""
    m_at = Int64(meta.toint())
    code = _i32_at(m_at + Int64(4 * (META_HEAD + 3 * experts)), tile)
    e = code & Int32(0xFFFF)
    row0 = (code >> Int32(16)) * Int32(STREAM_TILE_M)
    live = _i32_at(m_at + Int64(4 * META_HEAD), e) - row0
    if live > Int32(STREAM_TILE_M):
        live = Int32(STREAM_TILE_M)
    first = _i32_at(m_at + Int64(4 * (META_HEAD + experts)), e) + row0
    return e, first, live


@cute.jit
def _swiglu(g: Float32, u: Float32, limit: cutlass.Constexpr) -> Float32:
    gate = _round_bf16(g)
    up = _round_bf16(u)
    if const_expr(limit > 0.0):
        gate = cutlass.select_(gate > Float32(limit), Float32(limit), gate)
        up = cutlass.select_(up > Float32(limit), Float32(limit), up)
        up = cutlass.select_(up < Float32(-limit), Float32(-limit), up)
    silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
    return silu * up


def _u8_weight(pointer: cute.Pointer, rows, cols: int) -> cute.Tensor:
    u8 = cute.make_ptr(cutlass.Uint8, Int64(pointer.toint()), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(u8, cute.make_layout((rows, cols), stride=(cols, 1)))


class StreamFp8GateUp:
    """``act[p] = swiglu(x_p . gate_e, x_p . up_e)`` for the grouped rows of
    each 128-row chunk (see module docstring). ``x`` is FP8 K32 wire rows
    ``[rows, H + H/32]`` read through ``pair_row``; ``w1``/``w3`` E4M3 ``[E, I,
    H]`` with FP32 ``[E, I/128, H/128]`` scales; ``act`` BF16 ``[max_tiles *
    128, I]`` in chunk-padded rows (chunk ``t`` at rows ``128 t``).
    Grid ``(I / 32, max_tiles)``, 256 threads: warps 4 (rows) x 2 (16 gate +
    16 up columns each); 128-column K steps (one scale block each)."""

    threads = 256
    cols = 32
    inter_alignment = 128
    k_step = 128
    # GB10, MiMo TP4 slice, 4096 rows (us): a 3 / w 5: 5689; a 4 / w 3: 5447.
    a_ring = 4
    w_ring = 3

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0, qmma: bool = False):
        self.inter, self.hidden, self.experts, self.limit = int(inter), int(hidden), int(experts), float(limit)
        # qmma: W8A8 block-scaled E4M3 x E4M3 MMAs (see the module docstring);
        # else both operands widen to BF16 (the reference's bf16(w * s)).
        self.qmma = bool(qmma)
        if self.inter % self.inter_alignment or self.hidden % 128:
            raise ValueError(f"stream gate/up needs {self.inter_alignment}-aligned I and 128-aligned H")
        self.row_bytes = self.hidden + self.hidden // 32
        self.k_steps = self.hidden // self.k_step
        self.a_bytes = STREAM_TILE_M * self.k_step
        self.half_bytes = self.cols * self.k_step
        self.s_bytes = STREAM_TILE_M * 4
        if self.k_steps < max(self.w_ring, self.a_ring):
            raise ValueError("stream gate/up needs at least as many K steps as ring slots")

    def key(self) -> tuple:
        return ("stream_gate_up", 4, self.inter, self.hidden, self.experts, self.limit, self.a_ring, self.w_ring,
                self.qmma)

    def _w_layout(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_step), cutlass.Uint8)
        return cute.tile_to_shape(atom, (self.cols, self.k_step, 2 * self.w_ring), order=(0, 1, 2))

    def _storage(self, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.a_bytes], 1024],
            "s": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.s_bytes], 16],
            "f": cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.k_steps], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, act: cute.Pointer, max_tiles: Int32,
                 stream: cuda.CUstream):
        w_layout = self._w_layout()
        rows_w = self.experts * self.inter
        box = cute.slice_(w_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.hidden), box, (self.cols, self.k_step),
            num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.hidden), box, (self.cols, self.k_step),
            num_multicast=1)
        self.kernel(x, pair_row, meta, tma_tensor_g, tma_tensor_u, s1, s3, act, tma_g, tma_u, w_layout,
                    self._storage(w_layout)).launch(
            grid=(self.inter // self.cols, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, weight_g: cute.Tensor,
               weight_u: cute.Tensor, s1: cute.Pointer, s3: cute.Pointer, act: cute.Pointer,
               tma_g: cute.CopyAtom, tma_u: cute.CopyAtom, w_layout: cute.ComposedLayout,
               Storage: cutlass.Constexpr):
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
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            ss = shared_ptr_to_u32(storage.s.data_ptr())
            sf = shared_ptr_to_u32(storage.f.data_ptr())
            h, i = self.hidden, self.inter
            n0 = n_blk * Int32(self.cols)
            w_tile = e * Int32(i // self.cols) + n_blk
            t_ws, t_wg = cpasync.tma_partition(tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight_g, (self.cols, self.k_step),
                                                                                (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                            cute.group_modes(cute.local_tile(weight_u, (self.cols, self.k_step),
                                                                             (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
                for slot in cutlass.range_constexpr(self.w_ring):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            # Loader roles: 16-byte chunk ch of rows tidx/8 + 32q, one row's
            # four UE8M0 bytes (tidx < 128), block scales (tidx < 2 K blocks).
            ch = tidx % Int32(8)
            r_base = tidx // Int32(8)
            src = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
            for q in cutlass.range_constexpr(4):
                row = r_base + Int32(32 * q)
                src[q] = Int64(-1)
                if row < live:
                    source = _i32_at(Int64(pair_row.toint()), first + row)
                    src[q] = Int64(x.toint()) + Int64(source) * Int64(self.row_bytes) + Int64(ch) * Int64(16)
            s_src = Int64(-1)
            if tidx < live:
                source = _i32_at(Int64(pair_row.toint()), first + tidx)
                s_src = Int64(x.toint()) + Int64(source) * Int64(self.row_bytes) + Int64(h)
            sc_off = (Int64(e) * Int64(i // 128) + Int64(n0 // Int32(128))) * Int64(self.k_steps * 4)
            if tidx < Int32(self.k_steps):
                cp_async_u32_shared_global(sf + tidx * Int32(4), Int64(s1.toint()) + sc_off + Int64(tidx) * Int64(4))
            elif tidx < Int32(2 * self.k_steps):
                cp_async_u32_shared_global(sf + tidx * Int32(4), Int64(s3.toint()) + sc_off
                                           + Int64(tidx - Int32(self.k_steps)) * Int64(4))

            # Prologue: rows of steps 0, 1 (cp.async groups), weights of
            # steps 0 .. w_ring-2 (TMA).
            for p in cutlass.range_constexpr(self.a_ring - 1):
                self._load_a(p, sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(self.w_ring - 1):
                    self._load_w(p, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g = lane // Int32(4)
            j = lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(32):
                acc[v] = Float32(0.0)
            for step in cutlass.range(self.k_steps, unroll=1):
                slot = step % Int32(self.w_ring)
                cute.arch.cp_async_wait_group(self.a_ring - 2)
                cute.arch.mbarrier_wait(mbar + slot, (step // Int32(self.w_ring)) & Int32(1))
                cute.arch.sync_threads()
                if step + Int32(self.a_ring - 1) < Int32(self.k_steps):
                    self._load_a(step + Int32(self.a_ring - 1), sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
                if warp_id == Int32(0):
                    if step + Int32(self.w_ring - 1) < Int32(self.k_steps):
                        self._load_w(step + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                if cutlass.const_expr(self.qmma):
                    if active:
                        self._step_qmma(step, slot, sa, sw, ss, sf, warp_m, warp_n, lane, g, acc)
                else:
                    if active:
                        stage = step % Int32(self.a_ring)
                        a_base = sa + stage * Int32(self.a_bytes)
                        w_base = sw + slot * Int32(2 * self.half_bytes)
                        s_base = ss + stage * Int32(self.s_bytes)
                        s_gate = ld_shared_f32(sf + step * Int32(4))
                        s_up = ld_shared_f32(sf + (step + Int32(self.k_steps)) * Int32(4))
                        # UE8M0 words (4 K32 groups) of this thread's rows g, g + 8 per M tile.
                        rsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            for hi in cutlass.range_constexpr(2):
                                srow = warp_m * Int32(32) + Int32(16 * mt + 8 * hi) + g
                                rsc[2 * mt + hi] = ld_shared_u32(s_base + srow * Int32(4))
                        for ks in cutlass.range_constexpr(4):
                            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                            for mt in cutlass.range_constexpr(2):
                                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                                chunk = Int32(2 * ks) + lane // Int32(16)
                                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                    a_base + row * Int32(self.k_step) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                                a[4 * mt] = a0
                                a[4 * mt + 1] = a1
                                a[4 * mt + 2] = a2
                                a[4 * mt + 3] = a3
                            b = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                            for p in cutlass.range_constexpr(2):
                                # Gate half (p 0) then up half (p 1), each a 32-row
                                # 128B-swizzled TMA box.
                                nrow = warp_n * Int32(16) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                                chunk = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
                                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                                    w_base + Int32(p * self.half_bytes) + nrow * Int32(self.k_step)
                                    + ((chunk ^ (nrow % Int32(8))) * Int32(16)))
                                b[4 * p] = b0
                                b[4 * p + 1] = b1
                                b[4 * p + 2] = b2
                                b[4 * p + 3] = b3
                            # The 8-bit fragments hold K bytes 4j..4j+3 (regs 0, 1)
                            # and 16+4j.. (regs 2, 3); widened to BF16 pairs they
                            # feed two m16n8k16 MMAs under one K permutation shared
                            # by A and B (K bytes 0..15, then 16..31).
                            wb = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                            for nt in cutlass.range_constexpr(4):
                                scale = s_gate if nt < 2 else s_up
                                for q in cutlass.range_constexpr(2):
                                    lo, hi = _e4m3x4_scaled_bf16x2x2(b[2 * nt + q], scale)
                                    wb[4 * nt + 2 * q] = lo
                                    wb[4 * nt + 2 * q + 1] = hi
                            for mt in cutlass.range_constexpr(2):
                                xa = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                                for r in cutlass.range_constexpr(4):
                                    exponent = (rsc[2 * mt + r % 2] >> Uint32(8 * ks)) & Uint32(0xFF)
                                    lo, hi = _e4m3x4_scaled_bf16x2x2(a[4 * mt + r], _u32_as_f32(exponent << Uint32(23)))
                                    xa[2 * r] = lo
                                    xa[2 * r + 1] = hi
                                for q in cutlass.range_constexpr(2):
                                    # xa: [r][lo, hi], r = (row g | g+8) + 2 * (K half q)
                                    for nt in cutlass.range_constexpr(4):
                                        f = 4 * (4 * mt + nt)
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                            xa[4 * q], xa[4 * q + 2], xa[4 * q + 1], xa[4 * q + 3],
                                            wb[4 * nt + 2 * q], wb[4 * nt + 2 * q + 1])
                                        acc[f] = d0
                                        acc[f + 1] = d1
                                        acc[f + 2] = d2
                                        acc[f + 3] = d3
            cute.arch.cp_async_wait_group(0)
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            for nt in cutlass.range_constexpr(2):
                                fg = 4 * (4 * mt + nt) + 2 * half
                                fu = 4 * (4 * mt + nt + 2) + 2 * half
                                col = n0 + warp_n * Int32(16) + Int32(8 * nt) + Int32(2) * j
                                v0 = _swiglu(acc[fg], acc[fu], self.limit)
                                v1 = _swiglu(acc[fg + 1], acc[fu + 1], self.limit)
                                st_global_u32(Int64(act.toint()) + (Int64(tile * Int32(STREAM_TILE_M) + row)
                                                                    * Int64(i) + Int64(col)) * Int64(2),
                                              pack_f32x2_to_bfloat2(v0, v1))

    @cute.jit
    def _step_qmma(self, step, slot, sa, sw, ss, sf, warp_m, warp_n, lane, g, acc):
        """One 128-K step of the W8A8 gate/up: E4M3 A (wire rows, UE8M0 per 32
        as the A scales) x E4M3 B (unit scales) into a fresh FP32 partial per
        fragment, then ``acc += partial * s_block`` (gate or up block scale)."""
        stage = step % Int32(self.a_ring)
        a_base = sa + stage * Int32(self.a_bytes)
        w_base = sw + slot * Int32(2 * self.half_bytes)
        s_base = ss + stage * Int32(self.s_bytes)
        s_gate = ld_shared_f32(sf + step * Int32(4))
        s_up = ld_shared_f32(sf + (step + Int32(self.k_steps)) * Int32(4))
        # A scales: row g (even lanes) or g + 8 (odd lanes) of each m16 tile; byte ks
        # is this step's 32-K group ks.
        asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
        for mt in cutlass.range_constexpr(2):
            srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
            asc[mt] = ld_shared_u32(s_base + srow * Int32(4))
        unit = Uint32(0x7F7F7F7F)
        part = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
        for v in cutlass.range_constexpr(32):
            part[v] = Float32(0.0)
        for ks in cutlass.range_constexpr(4):
            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(2):
                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                chunk = Int32(2 * ks) + lane // Int32(16)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    a_base + row * Int32(self.k_step) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            b = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            for p in cutlass.range_constexpr(2):
                # (b[4p], b[4p+1]): column g, K 4j.. and 16+4j..; (b[4p+2], b[4p+3]): column 8 + g.
                nrow = warp_n * Int32(16) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                chunk = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                    w_base + Int32(p * self.half_bytes) + nrow * Int32(self.k_step)
                    + ((chunk ^ (nrow % Int32(8))) * Int32(16)))
                b[4 * p] = b0
                b[4 * p + 1] = b1
                b[4 * p + 2] = b2
                b[4 * p + 3] = b3
            for mt in cutlass.range_constexpr(2):
                for nt in cutlass.range_constexpr(4):
                    f = 4 * (4 * mt + nt)
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        part[f], part[f + 1], part[f + 2], part[f + 3],
                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b[2 * nt], b[2 * nt + 1],
                        asc[mt], unit, bid_a=ks, bid_b=0)
                    part[f] = d0
                    part[f + 1] = d1
                    part[f + 2] = d2
                    part[f + 3] = d3
        for mt in cutlass.range_constexpr(2):
            for nt in cutlass.range_constexpr(4):
                scale = s_gate if nt < 2 else s_up
                for v in cutlass.range_constexpr(4):
                    f = 4 * (4 * mt + nt) + v
                    acc[f] = acc[f] + part[f] * scale

    @cute.jit
    def _load_w(self, step, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile):
        slot = Int32(step) % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * self.half_bytes)
        cute.copy(tma_g, t_wg[(None, w_tile, Int32(step))], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
        cute.copy(tma_u, t_wu[(None, w_tile, Int32(step))], t_ws[(None, slot * Int32(2) + Int32(1))],
                  tma_bar_ptr=barrier)

    @cute.jit
    def _load_a(self, step, sa, ss, src, s_src, ch, r_base, tidx):
        stage = Int32(step) % Int32(self.a_ring)
        k_off = Int64(step) * Int64(self.k_step)
        a_stage = sa + stage * Int32(self.a_bytes)
        for q in cutlass.range_constexpr(4):
            if src[q] >= Int64(0):
                row = r_base + Int32(32 * q)
                cp_async4_shared_global(a_stage + row * Int32(self.k_step) + ((ch ^ (row % Int32(8))) * Int32(16)),
                                        src[q] + k_off)
        if s_src >= Int64(0):
            cp_async_u32_shared_global(ss + stage * Int32(self.s_bytes) + tidx * Int32(4),
                                       s_src + Int64(step) * Int64(4))


class StreamFp8Down:
    """``y[p] = act_p . bf16(w2_e * s2_e)^T`` for the grouped rows of each
    128-row chunk; ``act`` BF16 ``[max_tiles * 128, I]`` in chunk-padded rows
    (chunk ``t`` at rows ``128 t``), ``w2`` E4M3 ``[E, H, I]`` with FP32 ``[E,
    H/128, I/128]`` scales, ``y`` BF16 ``[pairs, H]`` in grouped rows. Grid
    ``(H / 128, max_tiles)``, 256 threads: warps 4 (rows) x 2 (64 columns
    each). 32-column K steps: the act tiles arrive by TMA into a ``ring``-deep
    ring, the weights by TMA in 128-column blocks (whole 128-byte lines per
    row; 32-byte row pieces per step read at ~70% of DRAM bandwidth) into a
    2-deep ring, one mbarrier per slot; the widened weight tile is
    double-buffered (step t + 1 widens while step t multiplies); the output
    tile leaves through shared memory in 16-byte row segments."""

    threads = 256
    tile_n = 128
    inter_alignment = 128
    k_step = 32
    # GB10, MiMo TP4 slice, 4096 rows (us), act rows by cp.async: 4854-5068 at
    # best, and 4874 with neither MMAs nor widening (the per-thread 16-byte
    # row copies were the limit). Not kept: a CTA walking 2 or 4 column blocks
    # (6609, 5746), 256-column tiles (4827 vs 4923), two CTAs per SM (4902 vs
    # 5014), a shared-memory staged epilogue alone (5002-5222).
    ring = 5
    w_block = 128

    def __init__(self, *, hidden: int, inter: int, experts: int):
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        if self.hidden % 128 or self.inter % self.inter_alignment:
            raise ValueError(f"stream down needs 128-aligned H and {self.inter_alignment}-aligned I")
        self.k_steps = self.inter // self.k_step
        self.a_bytes = STREAM_TILE_M * self.k_step * 2
        self.w_bytes = self.tile_n * self.w_block
        self.per_block = self.w_block // self.k_step
        self.b_bytes = self.tile_n * self.k_step * 2
        self.k_blocks = self.inter // 128
        self.out_stride = self.tile_n * 2 + 16  # padded staging rows: conflict-free fragment writes
        if self.ring < 3:
            raise ValueError("stream down waits for step t + 1 before issuing step t + ring - 1: ring >= 3")
        if STREAM_TILE_M * self.out_stride > self.ring * self.a_bytes:
            raise ValueError("the output staging tile must fit the act ring")

    def key(self) -> tuple:
        return ("stream_down", 8, self.hidden, self.inter, self.experts, self.ring)

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.k_step),
            cutlass.BFloat16)
        a_layout = cute.tile_to_shape(atom, (STREAM_TILE_M, self.k_step, self.ring), order=(0, 1, 2))
        w_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.w_block), cutlass.Uint8)
        w_layout = cute.tile_to_shape(w_atom, (self.tile_n, self.w_block, 2), order=(0, 1, 2))
        return a_layout, w_layout

    def _storage(self, a_layout, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.ring + 2],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout)], 1024],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "b": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.b_bytes], 1024],
            "f": cute.struct.Align[cute.struct.MemRange[Float32, self.k_blocks], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, act: cute.Pointer, meta: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        a_layout, w_layout = self._layouts()
        act_t = cute.make_tensor(act, cute.make_layout((max_tiles * Int32(STREAM_TILE_M), self.inter),
                                                       stride=(self.inter, 1)))
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), act_t, cute.slice_(a_layout, (None, None, 0)),
            (STREAM_TILE_M, self.k_step), num_multicast=1)
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w2, self.experts * self.hidden, self.inter),
            cute.slice_(w_layout, (None, None, 0)), (self.tile_n, self.w_block), num_multicast=1)
        self.kernel(meta, tma_tensor_a, tma_tensor_w, s2, y, tma_a, tma_w, a_layout, w_layout,
                    self._storage(a_layout, w_layout)).launch(
            grid=(self.hidden // self.tile_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

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
            sf = shared_ptr_to_u32(storage.f.data_ptr())
            h = self.hidden
            w_tile = e * Int32(h // self.tile_n) + n_blk
            t_as, t_ag = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                                               cute.group_modes(cute.local_tile(act, (STREAM_TILE_M, self.k_step),
                                                                                (None, None)), 0, 2))
            t_ws, t_wg = cpasync.tma_partition(tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight, (self.tile_n, self.w_block),
                                                                                (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_a)
                cpasync.prefetch_descriptor(tma_w)
                for slot in cutlass.range_constexpr(self.ring + 2):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            # Block scales of this 128-row weight block (tidx < I/128), once.
            if tidx < Int32(self.k_blocks):
                cp_async_u32_shared_global(
                    sf + tidx * Int32(4),
                    Int64(s2.toint()) + ((Int64(e) * Int64(h // 128) + Int64(n_blk)) * Int64(self.k_blocks)
                                         + Int64(tidx)) * Int64(4))
            cute.arch.cp_async_commit_group()
            blocks = self.k_steps // self.per_block
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(min(2, blocks)):
                    self._load_w(p, tma_w, t_wg, t_ws, wbar, w_tile)
                for p in cutlass.range_constexpr(self.ring - 1):
                    if Int32(p) < Int32(self.k_steps):
                        self._load_a(p, tma_a, t_ag, t_as, mbar, tile)
            cute.arch.cp_async_wait_group(0)
            cute.arch.mbarrier_wait(wbar, 0)
            cute.arch.sync_threads()
            self._widen(0, sw, sb, sf, tidx)

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
                cute.arch.sync_threads()
                if warp_id == Int32(0):
                    if step + Int32(self.ring - 1) < Int32(self.k_steps):
                        self._load_a(step + Int32(self.ring - 1), tma_a, t_ag, t_as, mbar, tile)
                    # Block `block` starts: its predecessor's slot is free
                    # (last widened in the previous iteration).
                    if starts & (block + Int32(1) < Int32(blocks)):
                        self._load_w(block + Int32(1), tma_w, t_wg, t_ws, wbar, w_tile)
                if nxt < Int32(self.k_steps):
                    self._widen(nxt, sw, sb, sf, tidx)
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
            # Epilogue: fragments -> padded shared tile (the act ring is idle:
            # every issued copy was waited for) -> 16-byte row segments of y.
            cute.arch.sync_threads()
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                            st_shared_u32(sa + row * Int32(self.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
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
    def _load_a(self, step, tma_a, t_ag, t_as, mbar, tile):
        slot = Int32(step) % Int32(self.ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.a_bytes)
        cute.copy(tma_a, t_ag[(None, tile, Int32(step))], t_as[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _load_w(self, block, tma_w, t_wg, t_ws, wbar, w_tile):
        slot = Int32(block) % Int32(2)
        barrier = wbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.w_bytes)
        cute.copy(tma_w, t_wg[(None, w_tile, Int32(block))], t_ws[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _widen(self, step, sw, sb, sf, tidx):
        s = ld_shared_f32(sf + (Int32(step) * Int32(self.k_step) // Int32(128)) * Int32(4))
        block = Int32(step) // Int32(self.per_block)
        piece = Int32(step) % Int32(self.per_block)
        w_stage = sw + (block % Int32(2)) * Int32(self.w_bytes)
        b_stage = sb + (Int32(step) % Int32(2)) * Int32(self.b_bytes)
        for q in cutlass.range_constexpr(self.tile_n * self.k_step // 8 // self.threads):
            c = tidx + Int32(self.threads * q)
            n = c // Int32(4)
            chunk = c % Int32(4)
            # 128B-swizzled weight rows: logical 16-byte unit piece * 2 + chunk / 2.
            unit = piece * Int32(2) + chunk // Int32(2)
            lo, hi = ld_shared_v2_u32(w_stage + n * Int32(self.w_block) + ((unit ^ (n % Int32(8))) * Int32(16))
                                      + (chunk % Int32(2)) * Int32(8))
            v0, v1, v2, v3 = _e4m3x8_scaled_bf16(lo, hi, s)
            st_shared_v4_u32(b_stage + n * Int32(64) + ((chunk ^ ((n >> Int32(1)) & Int32(3))) * Int32(16)),
                             v0, v1, v2, v3)
