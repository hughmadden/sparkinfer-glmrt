"""Expert-major M64 EXLR8 prefill, retaining the K1 reduction tree.

One CTA owns a (group, unit) pair: 256 threads, eight warps, each warp one
N16 tile of the unit's N128. The K128 weight blocks of gate and up are staged
into a three-deep shared ring (8 KiB per projection; the native t256 layout is
contiguous per block, so the copy is width-agnostic) beside the CTA's A tile,
and every warp decodes its own tile column from shared memory and feeds all
four M16 accumulators. No weight byte is fetched or decoded twice. Gate and up
share one A tile. The six K1 splits fold inside the CTA in ascending order and
the fused activation applies the same SwiGLU and H128 pair the decode route
runs, so the fc1 scratch is never written. FC2 gives every CTA one group: when
every unit intermediate fits the device's opt-in CTA budget it stays resident while all
output blocks stream through a two-block ring, so each intermediate is read
once per group instead of once per (block, unit); otherwise (the benchmark's
four units, and the exporter's sixteen) each (block, unit) step streams one
intermediate slice and its down block through a three-deep ring, walking units
ascending per block so the FP32 total accumulates in K1's order.

Every reduction keeps K1's order: eight ascending m16n8k16 FP16 MMAs per K128
block, exactly one FP32 FMA per K128 scale, an ascending split fold, ascending
unit accumulation in FC2, and the F16 intermediate rounding.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import (
    cp_async4_shared_global, f16_mma_m16n8k16_f32, ld_shared_u32,
    ld_shared_v4_f32, ldmatrix_m8n8x4_b16, packed_dequant_trellis_to_half2x4,
    shared_ptr_to_u32, st_shared_f32, st_shared_v4_u32,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad
from .exlr8 import Exlr8A16Kernel, _e4m3_round_half2, _side_value

# Weight blocks in flight: one commit group carries the A tile and both
# projections' blocks, so the ring is 3 x (A 16 KiB + gate 8 KiB + up 8 KiB).
STAGES = 3
# One K128 weight block at width 4, and the shared slot that always holds it.
BLOCK_SLOT_WORDS = 2048
# Gate (and up) scratch for one M16 tile: 16 rows x 128 columns, F32.
A_TILE_WORDS = 16 * 128


def query_optin_shared_memory() -> int | None:
    """This host's per-CTA opt-in shared memory, or None without a device.

    The CTA budget comes from the driver (cudaDevAttrMaxSharedMemoryPerBlock-
    Optin) every time a kernel is built, so no card's limit is baked into a
    layout. Offline AOT export runs with CUDA hidden from the compiler
    workers; there the caller gets None and only the universal streamed
    layout is compiled, which fits whenever FC1 does on every SM12x card.
    """
    try:
        status = cuda.cuInit(0)
        if isinstance(status, tuple):
            status = status[0]
        if status != cuda.CUresult.CUDA_SUCCESS:
            return None
        device = cuda.cuDeviceGet(0)
        if device[0] != cuda.CUresult.CUDA_SUCCESS:
            return None
        attribute = (cuda.CUdevice_attribute
                     .CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        value = cuda.cuDeviceGetAttribute(attribute, device[1])
        if value[0] != cuda.CUresult.CUDA_SUCCESS:
            return None
        limit = int(value[1])
    except Exception:
        return None
    return limit if limit > 0 else None


@cute.jit
def decode_tile_smem(base: Int32, lane: Int32, width: cutlass.Constexpr):
    """Both N8 halves of one staged t256 tile, with the mandatory E4M3 round.

    The staged block is byte-identical to the native payload block, so the
    tile's lane geometry is unchanged; only the load moves to shared memory.
    """
    if cutlass.const_expr(width == 4):
        b = ld_shared_u32(base + lane * Int32(4))
        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
        wa = b
        wb = (a << Uint32(16)) | (b >> Uint32(16))
    else:
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
        a = ld_shared_u32(base + ia * Int32(4))
        b = ld_shared_u32(base + ib * Int32(4))
        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
        wa = (window >> shift.to(Int64)).to(Uint32)
        wb = (window >> (shift + Int32(4 * width)).to(Int64)).to(Uint32)
    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
    return (_e4m3_round_half2(b0), _e4m3_round_half2(b1),
            _e4m3_round_half2(b2), _e4m3_round_half2(b3))


class Exlr8GroupedKernel(Exlr8A16Kernel):
    CODE_VERSION = 5
    CTAS_PER_SM = 2

    def __init__(self, hidden_size, units_per_expert, top_k, max_groups, group_rows=64,
                 shared_memory_bytes=None, resident_fc2=None):
        super().__init__(hidden_size, units_per_expert, top_k, max_groups, device_count=True)
        if group_rows not in (16, 32, 64):
            raise ValueError("EXLR8 grouped group_rows must be 16, 32 or 64")
        self.group_rows = group_rows
        self.mtiles = group_rows // 16
        # The CTA budget is the device's own opt-in limit: a caller that has
        # already queried it passes it, otherwise it is queried here. It
        # decides which compiled layout this kernel object uses, and a limit
        # that fits no layout is refused instead of launching an oversized
        # request. With no device (offline export) the universal streamed
        # layout is chosen, so one artifact serves every SM12x card.
        self.shared_memory_bytes = (shared_memory_bytes if shared_memory_bytes is not None
                                    else query_optin_shared_memory())
        # Blocks staged before the loop; also the cp.async wait depth, since
        # exactly this many groups are newer than the one being consumed.
        self.prologue = min(STAGES - 1, self.blocks)
        a_words = group_rows * 64
        fc1_words = STAGES * (a_words + 2 * BLOCK_SLOT_WORDS)
        if self.shared_memory_bytes is not None and fc1_words * 4 > self.shared_memory_bytes:
            raise ValueError(
                f"EXLR8 grouped FC1 needs {fc1_words * 4} B shared memory, "
                f"above the device's {self.shared_memory_bytes} B opt-in limit")
        # FC2 keeps every unit's intermediate resident when the queried limit
        # allows it, and otherwise streams one (unit, block) at a time through
        # a three-deep ring; both walk the units ascending, so the FP32
        # accumulation is identical.
        resident_words = self.units * a_words + 2 * self.units * BLOCK_SLOT_WORDS
        if resident_fc2 is None:
            self.fc2_resident = (self.shared_memory_bytes is not None
                                 and resident_words * 4 <= self.shared_memory_bytes)
        else:
            self.fc2_resident = bool(resident_fc2)
        ring_words = 3 * (a_words + BLOCK_SLOT_WORDS)
        needed = resident_words if self.fc2_resident else ring_words
        if self.shared_memory_bytes is not None and needed * 4 > self.shared_memory_bytes:
            raise ValueError(
                f"EXLR8 grouped FC2 needs {needed * 4} B shared memory, above "
                f"the device's {self.shared_memory_bytes} B opt-in limit")
        self.shared_words = max(fc1_words, needed)
        self.fc2_prologue = min(2, self.blocks * self.units)

    @property
    def __cache_key__(self):
        return ("exlr8_a16_grouped", self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.group_rows, STAGES, self.fc2_resident)

    @cute.jit
    def serving(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                group_expert: cute.Tensor, route_slots: cute.Tensor,
                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        # Grids and launch count are independent of route data, including zero
        # groups. Each kernel bounds its tasks with the device-resident count.
        if rows > Int32(0) and rows <= Int32(4096) and cta_limit > Int32(0):
            self.grouped_fc1_kernel(x, pointers, bits, group_expert, route_slots,
                                    intermediate, rows, group_count).launch(
                grid=(min(Int32(self.max_groups * self.units), cta_limit), 1, 1),
                block=(256, 1, 1), min_blocks_per_mp=1, stream=stream)
            if self.fc2_resident:
                self.grouped_fc2_kernel(pointers, bits, group_expert, route_slots,
                                        intermediate, fc2, rows, group_count).launch(
                    grid=(min(Int32(self.max_groups), cta_limit), 1, 1),
                    block=(256, 1, 1), min_blocks_per_mp=1, stream=stream)
            else:
                self.grouped_fc2_split_kernel(pointers, bits, group_expert, route_slots,
                                              intermediate, fc2, rows, group_count).launch(
                    grid=(min(Int32(self.max_groups), cta_limit), 1, 1),
                    block=(256, 1, 1), min_blocks_per_mp=1, stream=stream)
            self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                               rows, group_count).launch(
                grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.jit
    def live_tiles(self, route_slots: cute.Tensor, group: Int32, rows: Int32,
                   lane: Int32):
        # A warp-uniform vote handles arbitrary packing, including holes and
        # reversed tiles, without changing the ordered accumulation of live rows.
        live = cute.make_rmem_tensor(self.mtiles, Int32)
        for m in cutlass.range_constexpr(self.mtiles):
            local = Int32(m * 16) + (lane & Int32(15))
            slot = route_slots[group.to(Int64) * Int64(self.group_rows) + local.to(Int64)]
            live[m] = (cute.arch.vote_ballot_sync(
                (slot >= Int32(0)) & (slot < rows * Int32(self.top_k))) != Uint32(0)).to(Int32)
        return live

    @cute.jit
    def row_ids(self, route_slots: cute.Tensor, group: Int32, rows: Int32,
                lane: Int32, down: cutlass.Constexpr):
        result = cute.make_rmem_tensor((self.mtiles, 2), Int32)
        result.fill(Int32(-1))
        for m in cutlass.range_constexpr(self.mtiles):
            for v in cutlass.range_constexpr(2):
                local = Int32(m * 16 + v * 8) + lane // Int32(4)
                slot = route_slots[group.to(Int64) * Int64(self.group_rows) + local.to(Int64)]
                if slot >= Int32(0) and slot < rows * Int32(self.top_k):
                    result[m, v] = local if cutlass.const_expr(down) else slot // Int32(self.top_k)
        return result

    @cute.jit
    def mma_block(self, b_base: Int32, a_base: Int32, lane: Int32, warp: Int32,
                  width: cutlass.Constexpr, live: cute.Tensor, acc: cute.Tensor):
        """One staged K128 block into acc[mtile, half, value]."""
        for k16 in cutlass.range_constexpr(8):
            tile = b_base + (Int32(k16 * 8) + warp) * Int32(32 * width)
            b0, b1, b2, b3 = decode_tile_smem(tile, lane, width)
            for m in cutlass.range_constexpr(self.mtiles):
                if live[m] != Int32(0):
                    row = Int32(m * 16) + (lane & Int32(15))
                    col = Int32(k16 * 2) + lane // Int32(16)
                    address = a_base + (row * Int32(16) + (col ^ (row & Int32(7)))) * Int32(16)
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(address)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[m, 0, 0], acc[m, 0, 1], acc[m, 0, 2], acc[m, 0, 3],
                        a0, a1, a2, a3, b0, b1)
                    acc[m, 0, 0], acc[m, 0, 1], acc[m, 0, 2], acc[m, 0, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[m, 1, 0], acc[m, 1, 1], acc[m, 1, 2], acc[m, 1, 3],
                        a0, a1, a2, a3, b2, b3)
                    acc[m, 1, 0], acc[m, 1, 1], acc[m, 1, 2], acc[m, 1, 3] = d0, d1, d2, d3

    @cute.jit
    def scale_into(self, total: cute.Tensor, acc: cute.Tensor, scale: Float32,
                   live: cute.Tensor):
        # Exactly one FP32 FMA per K128 block; dead M16 tiles are pure zeros.
        for m in cutlass.range_constexpr(self.mtiles):
            if live[m] != Int32(0):
                for h in cutlass.range_constexpr(2):
                    for v in cutlass.range_constexpr(4):
                        total[m, h, v] = total[m, h, v] + acc[m, h, v] * scale

    @cute.jit
    def fold_split(self, folded: cute.Tensor, total: cute.Tensor):
        # Ascending split fold, as the decode route's activation kernel runs it.
        for m in cutlass.range_constexpr(self.mtiles):
            for h in cutlass.range_constexpr(2):
                for v in cutlass.range_constexpr(4):
                    folded[m, h, v] = folded[m, h, v] + total[m, h, v]
                    total[m, h, v] = Float32(0.0)

    @cute.jit
    def stage_block(self, a_smem: Int32, b_smem: Int32, x_address: Int64,
                    payload_g: Int64, payload_u: Int64, route_slots: cute.Tensor,
                    group: Int32, rows: Int32, kb: Int32, wg: Int32, wu: Int32,
                    tid: Int32):
        """Join block kb's A tile, gate block and up block into one commit group.

        The A vectors are XOR-swizzled per row so ldmatrix's eight rows land in
        distinct banks; dead rows are zero-filled in place. A K128 block of a
        projection is 2048 * width contiguous bytes, so the copy needs no
        knowledge of the trellis layout.
        """
        for i in cutlass.range_constexpr(self.group_rows * 16 // 256):
            vector = tid + Int32(i * 256)
            row, col = vector // Int32(16), vector % Int32(16)
            slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
            target = a_smem + (row * Int32(16) + (col ^ (row & Int32(7)))) * Int32(16)
            if slot >= Int32(0) and slot < rows * Int32(self.top_k):
                source_row = slot // Int32(self.top_k)
                source = x_address + (source_row.to(Int64) * Int64(self.hidden_size)
                                      + kb.to(Int64) * Int64(128)
                                      + col.to(Int64) * Int64(8)) * Int64(2)
                cp_async4_shared_global(target, source)
            else:
                st_shared_v4_u32(target, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for i in cutlass.range_constexpr(2):
            v = tid + Int32(i * 256)
            if v < wg * Int32(128):
                cp_async4_shared_global(
                    b_smem + v * Int32(16),
                    payload_g + kb.to(Int64) * (Int64(BLOCK_SLOT_WORDS) * wg.to(Int64))
                    + v.to(Int64) * Int64(16))
            if wu != Int32(0) and v < wu * Int32(128):
                cp_async4_shared_global(
                    b_smem + Int32(BLOCK_SLOT_WORDS * 4) + v * Int32(16),
                    payload_u + kb.to(Int64) * (Int64(BLOCK_SLOT_WORDS) * wu.to(Int64))
                    + v.to(Int64) * Int64(16))

    @cute.jit
    def activate(self, act_smem: Int32, intermediate: cute.Tensor,
                 pointers: cute.Tensor, group: Int32, unit: Int32, desc: Int64,
                 route_slots: cute.Tensor, rows: Int32, lane: Int32, warp: Int32,
                 folded_g: cute.Tensor, folded_u: cute.Tensor):
        """SwiGLU and both H128 passes for each M16 tile, then the F16 store.

        Reproduces the decode route's activation kernel expression for
        expression; only the distribution of rows over warps differs.
        """
        gate_scales = pointers[desc * Int64(3) + Int64(1)]
        up_scales = pointers[(desc + Int64(1)) * Int64(3) + Int64(1)]
        down_scales = pointers[(desc + Int64(2)) * Int64(3) + Int64(1)]
        up_smem = act_smem + Int32(A_TILE_WORDS * 4)
        row0 = lane >> Int32(2)
        c0 = warp * Int32(16) + (lane & Int32(3)) * Int32(2)
        for m in cutlass.range_constexpr(self.mtiles):
            for h in cutlass.range_constexpr(2):
                col = c0 + Int32(h * 8)
                base = (row0 * Int32(128) + col) * Int32(4)
                st_shared_f32(act_smem + base, folded_g[m, h, 0])
                st_shared_f32(act_smem + base + Int32(4), folded_g[m, h, 1])
                st_shared_f32(act_smem + base + Int32(8 * 128 * 4), folded_g[m, h, 2])
                st_shared_f32(act_smem + base + Int32(8 * 128 * 4) + Int32(4), folded_g[m, h, 3])
                st_shared_f32(up_smem + base, folded_u[m, h, 0])
                st_shared_f32(up_smem + base + Int32(4), folded_u[m, h, 1])
                st_shared_f32(up_smem + base + Int32(8 * 128 * 4), folded_u[m, h, 2])
                st_shared_f32(up_smem + base + Int32(8 * 128 * 4) + Int32(4), folded_u[m, h, 3])
            cute.arch.sync_threads()
            # Eight warps cover the tile's sixteen rows two each, with the
            # decode route's four-adjacent-column lane layout.
            for pair in cutlass.range_constexpr(2):
                local = warp * Int32(2) + Int32(pair)
                absolute = Int32(m * 16) + local
                slot = route_slots[group.to(Int64) * Int64(self.group_rows) + absolute.to(Int64)]
                if slot >= Int32(0) and slot < rows * Int32(self.top_k):
                    col = lane * Int32(4)
                    g0, g1, g2, g3 = ld_shared_v4_f32(
                        act_smem + (local * Int32(128) + col) * Int32(4))
                    u0, u1, u2, u3 = ld_shared_v4_f32(
                        up_smem + (local * Int32(128) + col) * Int32(4))
                    gh = _w4a8_had128_quad(g0, g1, g2, g3, lane)
                    uh = _w4a8_had128_quad(u0, u1, u2, u3, lane)
                    r = cute.make_rmem_tensor(4, Float32)
                    for v in cutlass.range_constexpr(4):
                        column = col + Int32(v)
                        gate = gh[v] * _side_value(gate_scales, column)
                        up = uh[v] * _side_value(up_scales, column)
                        sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                        r[v] = gate * sigmoid * up * _side_value(down_scales, column)
                    r[0], r[1], r[2], r[3] = _w4a8_had128_quad(r[0], r[1], r[2], r[3], lane)
                    out_index = ((group * Int32(self.units) + unit).to(Int64) * Int64(self.group_rows)
                                 + absolute.to(Int64)) * Int64(128) + col.to(Int64)
                    for v in cutlass.range_constexpr(4):
                        intermediate[out_index + Int64(v)] = r[v].to(cutlass.Float16)
            cute.arch.sync_threads()

    @cute.kernel
    def grouped_fc1_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, rows: Int32, group_count: cute.Tensor):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()
        a_words = self.group_rows * 64

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, STAGES * (a_words + 2 * BLOCK_SLOT_WORDS)],
                1024]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        a_bytes = a_words * 4
        slot_bytes = a_bytes + 2 * BLOCK_SLOT_WORDS * 4
        # The activation scratch reuses the first ring slot: every staged
        # block has been consumed by the time it runs, and the next task's
        # prologue is issued only after its closing barrier.
        act_smem = base
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        while task < groups * Int32(self.units):
            group, unit = task // Int32(self.units), task % Int32(self.units)
            expert = group_expert[group]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
            wg, wu = bits[desc], bits[desc + Int64(1)]
            if wg != Int32(0) and wu != Int32(0):
                payload_g = pointers[desc * Int64(3)]
                payload_u = pointers[(desc + Int64(1)) * Int64(3)]
                scales_g = pointers[desc * Int64(3) + Int64(2)]
                scales_u = pointers[(desc + Int64(1)) * Int64(3) + Int64(2)]
                x_address = x.iterator.toint()
                live = self.live_tiles(route_slots, group, rows, lane)
                folded_g = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
                folded_u = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
                total_g = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
                total_u = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
                acc = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
                folded_g.fill(Float32(0.0))
                folded_u.fill(Float32(0.0))
                total_g.fill(Float32(0.0))
                total_u.fill(Float32(0.0))
                for s in cutlass.range_constexpr(self.prologue):
                    self.stage_block(base + Int32(s * slot_bytes),
                                     base + Int32(s * slot_bytes) + Int32(a_bytes),
                                     x_address, payload_g, payload_u, route_slots,
                                     group, rows, Int32(s), wg, wu, tid)
                    cute.arch.cp_async_commit_group()
                for kb in cutlass.range(self.blocks, unroll=1):
                    nxt = kb + Int32(self.prologue)
                    if nxt < Int32(self.blocks):
                        ahead = base + (nxt % Int32(STAGES)) * Int32(slot_bytes)
                        self.stage_block(ahead, ahead + Int32(a_bytes), x_address,
                                         payload_g, payload_u, route_slots, group,
                                         rows, nxt, wg, wu, tid)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(self.prologue)
                    cute.arch.sync_threads()
                    a_smem = base + (kb % Int32(STAGES)) * Int32(slot_bytes)
                    b_smem = a_smem + Int32(a_bytes)
                    acc.fill(Float32(0.0))
                    if wg == Int32(2):
                        self.mma_block(b_smem, a_smem, lane, warp, 2, live, acc)
                    elif wg == Int32(3):
                        self.mma_block(b_smem, a_smem, lane, warp, 3, live, acc)
                    elif wg == Int32(4):
                        self.mma_block(b_smem, a_smem, lane, warp, 4, live, acc)
                    self.scale_into(total_g, acc, _side_value(scales_g, kb), live)
                    acc.fill(Float32(0.0))
                    up_block = b_smem + Int32(BLOCK_SLOT_WORDS * 4)
                    if wu == Int32(2):
                        self.mma_block(up_block, a_smem, lane, warp, 2, live, acc)
                    elif wu == Int32(3):
                        self.mma_block(up_block, a_smem, lane, warp, 3, live, acc)
                    elif wu == Int32(4):
                        self.mma_block(up_block, a_smem, lane, warp, 4, live, acc)
                    self.scale_into(total_u, acc, _side_value(scales_u, kb), live)
                    if kb % Int32(8) == Int32(7):
                        self.fold_split(folded_g, total_g)
                        self.fold_split(folded_u, total_u)
                    cute.arch.sync_threads()
                if self.blocks % 8 != 0:
                    self.fold_split(folded_g, total_g)
                    self.fold_split(folded_u, total_u)
                self.activate(act_smem, intermediate, pointers, group, unit, desc,
                              route_slots, rows, lane, warp, folded_g, folded_u)
            cute.arch.sync_threads()
            task += step

    @cute.jit
    def stage_unit_a(self, a_smem: Int32, unit: Int32, unit_bytes: Int32,
                     intermediate_address: Int64, route_slots: cute.Tensor,
                     group: Int32, rows: Int32, tid: Int32):
        """Stage one unit's group_rows x 128 F16 intermediate tile, zero-filling
        dead rows."""
        for i in cutlass.range_constexpr(self.group_rows * 16 // 256):
            vector = tid + Int32(i * 256)
            row, col = vector // Int32(16), vector % Int32(16)
            slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
            target = a_smem + unit * unit_bytes + (row * Int32(16) + (col ^ (row & Int32(7)))) * Int32(16)
            if slot >= Int32(0) and slot < rows * Int32(self.top_k):
                source = intermediate_address + ((group * Int32(self.units) + unit).to(Int64)
                                                 * Int64(self.group_rows) + row.to(Int64)) * Int64(256) + col.to(Int64) * Int64(16)
                cp_async4_shared_global(target, source)
            else:
                st_shared_v4_u32(target, Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def stage_down_units(self, b_smem: Int32, pointers: cute.Tensor, bits: cute.Tensor,
                         expert: Int32, block: Int32, tid: Int32):
        """Stage every unit's down K128 block for one output block."""
        for i in cutlass.range_constexpr(2):
            v = tid + Int32(i * 256)
            for u in cutlass.range_constexpr(self.units):
                desc = (expert.to(Int64) * Int64(self.units) + Int64(u)) * Int64(3) + Int64(2)
                width = bits[desc]
                if v < width * Int32(128):
                    cp_async4_shared_global(
                        b_smem + Int32(u * BLOCK_SLOT_WORDS * 4) + v * Int32(16),
                        pointers[desc * Int64(3)]
                        + block.to(Int64) * (Int64(BLOCK_SLOT_WORDS) * width.to(Int64))
                        + v.to(Int64) * Int64(16))

    @cute.jit
    def store_fc2(self, total: cute.Tensor, rids: cute.Tensor, group: Int32,
                  block: Int32, lane: Int32, warp: Int32, fc2: cute.Tensor):
        for m in cutlass.range_constexpr(self.mtiles):
            for h in cutlass.range_constexpr(2):
                col = block * Int32(128) + warp * Int32(16) + Int32(h * 8) + (lane & Int32(3)) * Int32(2)
                if rids[m, 0] >= Int32(0):
                    index = ((group * Int32(self.group_rows) + Int32(m * 16)
                              + lane // Int32(4)).to(Int64) * Int64(self.hidden_size) + col.to(Int64))
                    fc2[index], fc2[index + Int64(1)] = total[m, h, 0], total[m, h, 1]
                if rids[m, 1] >= Int32(0):
                    index = ((group * Int32(self.group_rows) + Int32(m * 16 + 8)
                              + lane // Int32(4)).to(Int64) * Int64(self.hidden_size) + col.to(Int64))
                    fc2[index], fc2[index + Int64(1)] = total[m, h, 2], total[m, h, 3]

    @cute.kernel
    def grouped_fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, group_count: cute.Tensor):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()
        a_words = self.group_rows * 64

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, self.units * a_words + 2 * self.units * BLOCK_SLOT_WORDS],
                1024]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        unit_bytes = a_words * 4
        b_slot_bytes = self.units * BLOCK_SLOT_WORDS * 4
        b_base = base + Int32(self.units * unit_bytes)
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        while task < groups:
            group = task
            expert = group_expert[group]
            intermediate_address = intermediate.iterator.toint()
            for u in cutlass.range_constexpr(self.units):
                self.stage_unit_a(base, Int32(u), Int32(unit_bytes), intermediate_address,
                                  route_slots, group, rows, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self.stage_down_units(b_base, pointers, bits, expert, Int32(0), tid)
            cute.arch.cp_async_commit_group()
            rids = self.row_ids(route_slots, group, rows, lane, True)
            live = self.live_tiles(route_slots, group, rows, lane)
            total = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
            acc = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
            for block in cutlass.range(self.blocks, unroll=1):
                nxt = block + Int32(1)
                if nxt < Int32(self.blocks):
                    self.stage_down_units(b_base + (nxt % Int32(2)) * Int32(b_slot_bytes),
                                          pointers, bits, expert, nxt, tid)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
                b_smem = b_base + (block % Int32(2)) * Int32(b_slot_bytes)
                total.fill(Float32(0.0))
                for u in cutlass.range_constexpr(self.units):
                    desc = (expert.to(Int64) * Int64(self.units) + Int64(u)) * Int64(3) + Int64(2)
                    width = bits[desc]
                    if width != Int32(0):
                        acc.fill(Float32(0.0))
                        if width == Int32(2):
                            self.mma_block(b_smem + Int32(u * BLOCK_SLOT_WORDS * 4),
                                           base + Int32(u * unit_bytes), lane, warp, 2, live, acc)
                        elif width == Int32(3):
                            self.mma_block(b_smem + Int32(u * BLOCK_SLOT_WORDS * 4),
                                           base + Int32(u * unit_bytes), lane, warp, 3, live, acc)
                        elif width == Int32(4):
                            self.mma_block(b_smem + Int32(u * BLOCK_SLOT_WORDS * 4),
                                           base + Int32(u * unit_bytes), lane, warp, 4, live, acc)
                        self.scale_into(total, acc,
                                        _side_value(pointers[desc * Int64(3) + Int64(2)], block), live)
                cute.arch.sync_threads()
                self.store_fc2(total, rids, group, block, lane, warp, fc2)
                cute.arch.sync_threads()
            cute.arch.sync_threads()
            task += step

    @cute.jit
    def stage_fc2_step(self, slot: Int32, pointers: cute.Tensor, bits: cute.Tensor,
                       expert: Int32, block: Int32, unit: Int32,
                       intermediate_address: Int64, route_slots: cute.Tensor,
                       group: Int32, rows: Int32, tid: Int32):
        """Stage one (block, unit) step: its A tile and its down block."""
        self.stage_unit_a(slot, unit, Int32(0), intermediate_address, route_slots,
                          group, rows, tid)
        desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
        width = bits[desc]
        for i in cutlass.range_constexpr(2):
            v = tid + Int32(i * 256)
            if v < width * Int32(128):
                cp_async4_shared_global(
                    slot + Int32(self.group_rows * 64 * 4) + v * Int32(16),
                    pointers[desc * Int64(3)]
                    + block.to(Int64) * (Int64(BLOCK_SLOT_WORDS) * width.to(Int64))
                    + v.to(Int64) * Int64(16))

    @cute.kernel
    def grouped_fc2_split_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                                 group_expert: cute.Tensor, route_slots: cute.Tensor,
                                 intermediate: cute.Tensor, fc2: cute.Tensor,
                                 rows: Int32, group_count: cute.Tensor):
        """FC2 for a group whose intermediates exceed shared memory.

        One (block, unit) step streams through a three-slot ring (two steps in
        flight) in the same ascending unit order, so the FP32 accumulation is
        unchanged; only the activation tile is fetched again per output block.
        """
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()
        a_words = self.group_rows * 64

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, 3 * (a_words + BLOCK_SLOT_WORDS)], 1024]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())
        slot_bytes = (a_words + BLOCK_SLOT_WORDS) * 4
        total_steps = self.blocks * self.units
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        while task < groups:
            group = task
            expert = group_expert[group]
            intermediate_address = intermediate.iterator.toint()
            rids = self.row_ids(route_slots, group, rows, lane, True)
            live = self.live_tiles(route_slots, group, rows, lane)
            total = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
            acc = cute.make_rmem_tensor((self.mtiles, 2, 4), Float32)
            for s in cutlass.range_constexpr(self.fc2_prologue):
                self.stage_fc2_step(base + Int32(s * slot_bytes), pointers, bits, expert,
                                    Int32(s) // Int32(self.units), Int32(s) % Int32(self.units),
                                    intermediate_address, route_slots, group, rows, tid)
                cute.arch.cp_async_commit_group()
            for block in cutlass.range(self.blocks, unroll=1):
                total.fill(Float32(0.0))
                for unit in cutlass.range(self.units, unroll=1):
                    here = block * Int32(self.units) + unit
                    nxt = here + Int32(self.fc2_prologue)
                    if nxt < Int32(total_steps):
                        self.stage_fc2_step(base + (nxt % Int32(3)) * Int32(slot_bytes),
                                            pointers, bits, expert, nxt // Int32(self.units),
                                            nxt % Int32(self.units), intermediate_address,
                                            route_slots, group, rows, tid)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(self.fc2_prologue)
                    cute.arch.sync_threads()
                    slot = base + (here % Int32(3)) * Int32(slot_bytes)
                    desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                    width = bits[desc]
                    if width != Int32(0):
                        acc.fill(Float32(0.0))
                        if width == Int32(2):
                            self.mma_block(slot + Int32(a_words * 4), slot, lane, warp, 2, live, acc)
                        elif width == Int32(3):
                            self.mma_block(slot + Int32(a_words * 4), slot, lane, warp, 3, live, acc)
                        elif width == Int32(4):
                            self.mma_block(slot + Int32(a_words * 4), slot, lane, warp, 4, live, acc)
                        self.scale_into(total, acc,
                                        _side_value(pointers[desc * Int64(3) + Int64(2)], block), live)
                    cute.arch.sync_threads()
                self.store_fc2(total, rids, group, block, lane, warp, fc2)
                cute.arch.sync_threads()
            cute.arch.sync_threads()
            task += step
