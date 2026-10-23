"""Native trellis linear on pre-rotated inputs; fixed-order FP32 partials.

Both native output-unit and input-unit storage are consumed without repacking.
Rotations and channel scales belong to the caller, allowing several operands to
share one input rotation. A block scale multiplies each K128 accumulator, never
the E4M3/F16 decoded fragments. Live rows never select a different arithmetic.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    f16_mma_m16n8k16_f32, ld_global_nc_u32, ld_global_nc_u32_l2hint, ld_global_nc_v4_u32,
    weight_stream_policy,
    ld_shared_u32, packed_dequant_trellis_to_half2x4,
    packed_dequant_trellis_stream_to_half2x4,
    packed_dequant_trellis_mul1_to_half2x4,
    shared_ptr_to_u32, st_shared_v4_u32, trellis_align_stream_u32x2,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a16.exlr8 import _e4m3_round_half2, _side_value


def _load_word(address, policy):
    """A trellis payload word, with ``policy`` as its L2 cache policy when given."""
    if policy is None:
        return ld_global_nc_u32(address)
    return ld_global_nc_u32_l2hint(address, policy)


@cute.jit
def decode_tile(address: Int64, tile: Int32, lane: Int32,
                bits: cutlass.Constexpr, e4m3: cutlass.Constexpr,
                mul1: cutlass.Constexpr = False, policy=None):
    """One warp's 16x16 trellis tile. ``policy``: an L2 cache policy for the
    payload words (``weight_stream_policy``), or None for plain loads."""
    base = address + tile.to(Int64) * Int64(32 * bits)

    if cutlass.const_expr(bits == 4):
        b = _load_word(base + lane.to(Int64) * Int64(4), policy)
        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
        wa, wb = b, (a << Uint32(16)) | (b >> Uint32(16))
    elif cutlass.const_expr(bits <= 4):
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = _load_word(base + ia.to(Int64) * Int64(4), policy)
        b = _load_word(base + ib.to(Int64) * Int64(4), policy)
        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
        wa = (window >> shift.to(Int64)).to(Uint32)
        wb = (window >> (shift + Int32(4 * bits)).to(Int64)).to(Uint32)
    elif cutlass.const_expr(bits == 5):
        # Four K5 windows span 31 bits; all eight may span three ring words.
        ib0, ib1, sb, _ = trellis256_lane_geom_bits(lane, 0, 4, bits)
        ia0, ia1, sa, _ = trellis256_lane_geom_bits(lane, 4, 4, bits)
        b0 = _load_word(base + ib0.to(Int64) * Int64(4), policy)
        b1 = _load_word(base + ib1.to(Int64) * Int64(4), policy)
        a0 = _load_word(base + ia0.to(Int64) * Int64(4), policy)
        a1 = _load_word(base + ia1.to(Int64) * Int64(4), policy)
        wa = (((a0.to(Int64) << Int64(32)) | a1.to(Int64)) >> sa.to(Int64)).to(Uint32)
        wb = (((b0.to(Int64) << Int64(32)) | b1.to(Int64)) >> sb.to(Int64)).to(Uint32)
    else:
        # Even four K6 windows need 34 bits. Retain the aligned 64-bit stream.
        i0, i2, shift, delta = trellis256_lane_geom_bits(lane, 0, 8, bits)
        i1 = i0 + Int32(1)
        i1 -= Int32(8 * bits) * (i1 >= Int32(8 * bits)).to(Int32)
        z0 = _load_word(base + i0.to(Int64) * Int64(4), policy)
        z1 = _load_word(base + i1.to(Int64) * Int64(4), policy)
        z2 = _load_word(base + i2.to(Int64) * Int64(4), policy)
        wa, wb = trellis_align_stream_u32x2(z0, z1, z2, shift, delta)
    if cutlass.const_expr(mul1):
        b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, bits)
    elif cutlass.const_expr(bits == 6):
        b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, bits)
    else:
        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, bits)
    if cutlass.const_expr(e4m3):
        b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
        b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
    return b0, b1, b2, b3


class NativeTrellisLinear:
    ABI_VERSION = 1
    CTAS_PER_SM = 4

    def __init__(self, g):
        self.k, self.n = g["input_features"], g["output_features"]
        self.bits, self.e4m3 = g["bits"], g["codebook"] == "e4m3"
        self.mul1 = g["codebook"] == "mul1"
        # High-rate windows need more live registers than the frozen K2--K4 path.
        self.min_blocks_per_mp = 2 if self.bits >= 5 else self.CTAS_PER_SM
        self.input_units = g["sharded_side"] == 2
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        self.splits = g["splits"]

    def _storage(self, active: int):
        class Storage:
            pass
        Storage.__annotations__ = {
            "x": cute.struct.Align[cute.struct.MemRange[Uint32, 16 * (16 * active + 1) * 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations) and cta_limit > Int32(0):
            tiles = Int32(self.n // 128 * self.splits)
            if cutlass.const_expr(self.bits < 5):
                self.multiply(inputs, payload, scales, partial, rows, rotation_id).launch(
                    grid=(min(((rows + Int32(15)) // Int32(16)) * tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            elif rows <= Int32(16):
                self.multiply(inputs, payload, scales, partial, rows, rotation_id).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            elif rows <= Int32(32):
                self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 2, 8).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            elif rows <= Int32(48):
                self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 3, 8).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            else:
                self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id, 4, 4).launch(
                    grid=(min(((rows + Int32(63)) // Int32(64)) * tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            self.reduce(partial, output, rows).launch(
                grid=(min(rows * Int32(self.n // 128), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def multiply(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor,
                 rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        x = inputs[rotation_id]
        while task < ((rows + Int32(15)) // Int32(16)) * Int32(self.n // 128 * self.splits):
            split = task % Int32(self.splits)
            nb = (task // Int32(self.splits)) % Int32(self.n // 128)
            group = task // Int32(self.splits * (self.n // 128))
            # Each 16-row group re-reads the weights; the last one evicts them first.
            policy = weight_stream_policy((group + Int32(1) == (rows + Int32(15)) // Int32(16)).to(Int32))
            row0 = group * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(2):
                            tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                            if cutlass.const_expr(self.input_units):
                                tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                            b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for n in cutlass.range_constexpr(4):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
            for n in cutlass.range_constexpr(4):
                col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                base = (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                if row0 < rows:
                    partial[base], partial[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 < rows:
                    base += Int64(8 * self.splits * self.n)
                    partial[base], partial[base + Int64(1)] = total[n, 2], total[n, 3]
            task += step

    @cute.kernel
    def multiply_wide(self, inputs: cute.Tensor, payload: cute.Tensor,
                      scales: cute.Tensor, partial: cute.Tensor,
                      rows: Int32, rotation_id: Int32,
                      active: cutlass.Constexpr, unroll_k16: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        c, r_grp = tid & Int32(15), tid >> Int32(4)
        x = inputs[rotation_id]
        M = 16 * active
        M_PAD = M + 1
        tiles = Int32(self.n // 128 * self.splits)
        limit = tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * tiles
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage(active))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)
        while task < limit:
            split = task % Int32(self.splits)
            nb = (task // Int32(self.splits)) % Int32(self.n // 128)
            group = Int32(0) if cutlass.const_expr(active < 4) else task // tiles
            # One pass over the weights below 49 rows; else the last 64-row group evicts.
            last_reader = Int32(1)
            if cutlass.const_expr(active == 4):
                last_reader = (group + Int32(1) == (rows + Int32(M - 1)) // Int32(M)).to(Int32)
            policy = weight_stream_policy(last_reader)
            row_base = (lane >> Int32(2)) if cutlass.const_expr(active < 4) else group * Int32(M) + (lane >> Int32(2))
            total = cute.make_rmem_tensor((active, 4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    k_col = kb * Int32(128) + c * Int32(8)
                    for s in cutlass.range_constexpr(2 * active):
                        row = r_grp + Int32(s * 8) if cutlass.const_expr(active < 4) else group * Int32(M) + r_grp + Int32(s * 8)
                        v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if cutlass.const_expr(active in (2, 3) and s < 2 * (active - 1)):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        else:
                            if row < rows:
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                    cute.arch.sync_threads()
                    if cutlass.const_expr(active <= 2):
                        acc = cute.make_rmem_tensor((active, 4, 4), Float32)
                        acc.fill(Float32(0.0))
                        for k16 in cutlass.range_constexpr(8):
                            a = cute.make_rmem_tensor((active, 4), Uint32)
                            for mb in cutlass.range_constexpr(active):
                                a[mb, 0] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                a[mb, 1] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                a[mb, 2] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                a[mb, 3] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                            for n16 in cutlass.range_constexpr(2):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 2 * n16, 0], acc[mb, 2 * n16, 1], acc[mb, 2 * n16, 2], acc[mb, 2 * n16, 3],
                                        a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b0, b1)
                                    acc[mb, 2 * n16, 0], acc[mb, 2 * n16, 1], acc[mb, 2 * n16, 2], acc[mb, 2 * n16, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 2 * n16 + 1, 0], acc[mb, 2 * n16 + 1, 1], acc[mb, 2 * n16 + 1, 2], acc[mb, 2 * n16 + 1, 3],
                                        a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b2, b3)
                                    acc[mb, 2 * n16 + 1, 0], acc[mb, 2 * n16 + 1, 1], acc[mb, 2 * n16 + 1, 2], acc[mb, 2 * n16 + 1, 3] = d0, d1, d2, d3
                        scale = _side_value(scales.iterator.toint(), kb)
                        for mb in cutlass.range_constexpr(active):
                            for n in cutlass.range_constexpr(4):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, n, v] = total[mb, n, v] + acc[mb, n, v] * scale
                    elif cutlass.const_expr(active == 3):
                        scale = _side_value(scales.iterator.toint(), kb)
                        for n16 in cutlass.range_constexpr(2):
                            acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                            acc.fill(Float32(0.0))
                            for k16 in cutlass.range_constexpr(8):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    a0 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                    a1 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                    a2 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                    a3 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                        a0, a1, a2, a3, b2, b3)
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                            for mb in cutlass.range_constexpr(active):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                    total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    else:
                        scale = _side_value(scales.iterator.toint(), kb)
                        for n16 in cutlass.range_constexpr(2):
                            acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                            acc.fill(Float32(0.0))
                            for ko in cutlass.range(8 // unroll_k16, unroll=1):
                                sx_k = sx_read + ko * Int32(2 * unroll_k16 * M_PAD * 16)
                                for ki in cutlass.range_constexpr(unroll_k16):
                                    k16 = ko * Int32(unroll_k16) + Int32(ki)
                                    tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + k16) * Int32(8) + warp * Int32(2) + Int32(n16)
                                    if cutlass.const_expr(self.input_units):
                                        tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + k16 * Int32(8) + warp * Int32(2) + Int32(n16)
                                    b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                    for mb in cutlass.range_constexpr(active):
                                        a0 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16) * 16))
                                        a1 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16 + 8) * 16))
                                        a2 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16) * 16))
                                        a3 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16 + 8) * 16))
                                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                            acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                            a0, a1, a2, a3, b0, b1)
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                            acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                            a0, a1, a2, a3, b2, b3)
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                            for mb in cutlass.range_constexpr(active):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                    total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    cute.arch.sync_threads()
            for mb in cutlass.range_constexpr(active):
                row0 = row_base + Int32(mb * 16)
                row1 = row0 + Int32(8)
                for n in cutlass.range_constexpr(4):
                    col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                    base = (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                    t0, t1, t2, t3 = total[mb, n, 0], total[mb, n, 1], total[mb, n, 2], total[mb, n, 3]
                    if cutlass.const_expr(active < 4 and mb < active - 1):
                        partial[base], partial[base + Int64(1)] = t0, t1
                        base += Int64(8 * self.splits * self.n)
                        partial[base], partial[base + Int64(1)] = t2, t3
                    else:
                        if row0 < rows:
                            partial[base], partial[base + Int64(1)] = t0, t1
                        if row1 < rows:
                            base += Int64(8 * self.splits * self.n)
                            partial[base], partial[base + Int64(1)] = t2, t3
            task += step

    @cute.kernel
    def reduce(self, partial: cute.Tensor, output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        while task < rows * Int32(self.n // 128):
            row, col = task // Int32(self.n // 128), (task % Int32(self.n // 128)) * Int32(128) + Int32(tid)
            total = Float32(0.0)
            for split in cutlass.range(self.splits):
                total += partial[(row.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)]
            output[row.to(Int64) * Int64(self.n) + col.to(Int64)] = total
            task += step

    @cute.jit
    def call_grouped(self, inputs: cute.Tensor,
                     payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     partial: cute.Tensor,
                     outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                     rows: Int32, rotation_id: Int32, cta_limit: Int32,
                     stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id + Int32(3) < Int32(self.rotations) and cta_limit > Int32(0):
            group_tiles = Int32(4 * (self.n // 128) * self.splits)
            if cutlass.const_expr(self.bits < 5):
                self.multiply_grouped(inputs, payloads, scales, partial, rows, rotation_id).launch(
                    grid=(min(((rows + Int32(15)) // Int32(16)) * group_tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            elif rows <= Int32(16):
                self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 1, 8).launch(
                    grid=(min(group_tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
            elif rows <= Int32(32):
                self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 2, 4).launch(
                    grid=(min(group_tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
            elif rows <= Int32(48):
                self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 3, 8 if self.mul1 else 4).launch(
                    grid=(min(group_tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=2 if self.mul1 else 3, stream=stream)
            else:
                self.multiply_wide_grouped(inputs, payloads, scales, partial, rows, rotation_id, 4, 4).launch(
                    grid=(min(((rows + Int32(63)) // Int32(64)) * group_tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp, stream=stream)
            self.reduce_grouped(partial, outputs, rows).launch(
                grid=(min(rows * Int32(4 * (self.n // 128)), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def multiply_grouped(self, inputs: cute.Tensor,
                         payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                         scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                         partial: cute.Tensor, rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        tiles = Int32(self.n // 128 * self.splits)
        group_tiles = Int32(4 * (self.n // 128) * self.splits)
        while task < ((rows + Int32(15)) // Int32(16)) * group_tiles:
            group = task // group_tiles
            policy = weight_stream_policy((group + Int32(1) == (rows + Int32(15)) // Int32(16)).to(Int32))
            rem = task % group_tiles
            proj = rem // tiles
            tile_in_proj = rem % tiles
            split = tile_in_proj % Int32(self.splits)
            nb = tile_in_proj // Int32(self.splits)
            x = inputs[rotation_id + proj]
            payload_ptr = select4(
                proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                payloads[2].iterator.toint(), payloads[3].iterator.toint(),
            )
            scales_ptr = select4(
                proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                scales[2].iterator.toint(), scales[3].iterator.toint(),
            )
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            row0 = group * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(2):
                            tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                            if cutlass.const_expr(self.input_units):
                                tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                            b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales_ptr, kb)
                    for n in cutlass.range_constexpr(4):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
            for n in cutlass.range_constexpr(4):
                col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                base = proj_partial + (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                if row0 < rows:
                    partial[base], partial[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 < rows:
                    base += Int64(8 * self.splits * self.n)
                    partial[base], partial[base + Int64(1)] = total[n, 2], total[n, 3]
            task += step

    @cute.kernel
    def multiply_wide_grouped(self, inputs: cute.Tensor,
                              payloads: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                              scales: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                              partial: cute.Tensor, rows: Int32, rotation_id: Int32,
                              active: cutlass.Constexpr, unroll_k16: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        c, r_grp = tid & Int32(15), tid >> Int32(4)
        M = 16 * active
        M_PAD = M + 1
        tiles = Int32(self.n // 128 * self.splits)
        group_tiles = Int32(4 * (self.n // 128) * self.splits)
        limit = group_tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * group_tiles
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage(active))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)
        while task < limit:
            group = Int32(0) if cutlass.const_expr(active < 4) else task // group_tiles
            last_reader = Int32(1)
            if cutlass.const_expr(active == 4):
                last_reader = (group + Int32(1) == (rows + Int32(M - 1)) // Int32(M)).to(Int32)
            policy = weight_stream_policy(last_reader)
            rem = task if cutlass.const_expr(active < 4) else task % group_tiles
            proj = rem // tiles
            tile_in_proj = rem % tiles
            split = tile_in_proj % Int32(self.splits)
            nb = tile_in_proj // Int32(self.splits)
            total = cute.make_rmem_tensor((active, 4, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    x = inputs[rotation_id + proj]
                    k_col = kb * Int32(128) + c * Int32(8)
                    for s in cutlass.range_constexpr(2 * active):
                        row = r_grp + Int32(s * 8) if cutlass.const_expr(active < 4) else group * Int32(M) + r_grp + Int32(s * 8)
                        v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if cutlass.const_expr(active in (2, 3) and s < 2 * (active - 1)):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        else:
                            if row < rows:
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2))
                        st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                    cute.arch.sync_threads()
                    scale = _side_value(select4(
                        proj, scales[0].iterator.toint(), scales[1].iterator.toint(),
                        scales[2].iterator.toint(), scales[3].iterator.toint(),
                    ), kb)
                    payload_ptr = select4(
                        proj, payloads[0].iterator.toint(), payloads[1].iterator.toint(),
                        payloads[2].iterator.toint(), payloads[3].iterator.toint(),
                    )
                    for n16 in cutlass.range_constexpr(2):
                        acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                        acc.fill(Float32(0.0))
                        if cutlass.const_expr(unroll_k16 == 8):
                            for k16 in cutlass.range_constexpr(8):
                                tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                                if cutlass.const_expr(self.input_units):
                                    tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                                b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                for mb in cutlass.range_constexpr(active):
                                    a0 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                    a1 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                    a2 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                    a3 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                        a0, a1, a2, a3, b2, b3)
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                        else:
                            for ko in cutlass.range(8 // unroll_k16, unroll=1):
                                sx_k = sx_read + ko * Int32(2 * unroll_k16 * M_PAD * 16)
                                for ki in cutlass.range_constexpr(unroll_k16):
                                    k16 = ko * Int32(unroll_k16) + Int32(ki)
                                    tile = nb * Int32(self.k // 16 * 8) + (kb * Int32(8) + k16) * Int32(8) + warp * Int32(2) + Int32(n16)
                                    if cutlass.const_expr(self.input_units):
                                        tile = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + k16 * Int32(8) + warp * Int32(2) + Int32(n16)
                                    b0, b1, b2, b3 = decode_tile(payload_ptr, tile, lane, self.bits, self.e4m3, self.mul1, policy)
                                    for mb in cutlass.range_constexpr(active):
                                        a0 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16) * 16))
                                        a1 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16 + 8) * 16))
                                        a2 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16) * 16))
                                        a3 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16 + 8) * 16))
                                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                            acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                            a0, a1, a2, a3, b0, b1)
                                        acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                        d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                            acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                            a0, a1, a2, a3, b2, b3)
                                        acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                        for mb in cutlass.range_constexpr(active):
                            for v in cutlass.range_constexpr(4):
                                total[mb, 2 * n16, v] = total[mb, 2 * n16, v] + acc[mb, 0, v] * scale
                                total[mb, 2 * n16 + 1, v] = total[mb, 2 * n16 + 1, v] + acc[mb, 1, v] * scale
                    cute.arch.sync_threads()
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            row_base = (lane >> Int32(2)) if cutlass.const_expr(active < 4) else group * Int32(M) + (lane >> Int32(2))
            for mb in cutlass.range_constexpr(active):
                row0 = row_base + Int32(mb * 16)
                row1 = row0 + Int32(8)
                for n in cutlass.range_constexpr(4):
                    col = nb * Int32(128) + warp * Int32(32) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                    base = proj_partial + (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                    if cutlass.const_expr(active < 4 and mb < active - 1):
                        partial[base], partial[base + Int64(1)] = total[mb, n, 0], total[mb, n, 1]
                        base += Int64(8 * self.splits * self.n)
                        partial[base], partial[base + Int64(1)] = total[mb, n, 2], total[mb, n, 3]
                    else:
                        if row0 < rows:
                            partial[base], partial[base + Int64(1)] = total[mb, n, 0], total[mb, n, 1]
                        if row1 < rows:
                            base += Int64(8 * self.splits * self.n)
                            partial[base], partial[base + Int64(1)] = total[mb, n, 2], total[mb, n, 3]
            task += step

    @cute.kernel
    def reduce_grouped(self, partial: cute.Tensor,
                       outputs: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
                       rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        n_blocks = Int32(self.n // 128)
        group_blocks = Int32(4 * (self.n // 128))
        while task < rows * group_blocks:
            row = task // group_blocks
            rem = task % group_blocks
            proj = rem // n_blocks
            col = (rem % n_blocks) * Int32(128) + Int32(tid)
            proj_partial = proj.to(Int64) * (rows.to(Int64) * Int64(self.splits * self.n))
            total = Float32(0.0)
            for split in cutlass.range_constexpr(self.splits):
                total += partial[proj_partial + (row.to(Int64) * Int64(self.splits) + Int64(split)) * Int64(self.n) + col.to(Int64)]
            out_idx = row.to(Int64) * Int64(self.n) + col.to(Int64)
            if proj == Int32(0):
                outputs[0][out_idx] = total
            elif proj == Int32(1):
                outputs[1][out_idx] = total
            elif proj == Int32(2):
                outputs[2][out_idx] = total
            else:
                outputs[3][out_idx] = total
            task += step


@cute.jit
def select4(index: Int32, v0: Int64, v1: Int64, v2: Int64, v3: Int64) -> Int64:
    out = v3
    if index == Int32(0):
        out = v0
    elif index == Int32(1):
        out = v1
    elif index == Int32(2):
        out = v2
    return out
