"""Expert-major M64 EXLR8 prefill, retaining the K1 reduction tree.

Each warp owns N8 and four M16 tiles. Decoded B fragments feed all four M
accumulators before advancing K. A K128 accumulator is scaled with exactly
one FMA, and FC1 split partials retain the ascending fold of the decode path.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import (
    cp_async4_shared_global, f16_mma_m16n8k16_f32, ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32, st_shared_v4_u32,
)
from .exlr8 import Exlr8A16Kernel, exlr8_decode_tile, _side_value


class Exlr8GroupedKernel(Exlr8A16Kernel):
    CODE_VERSION = 3
    CTAS_PER_SM = 2

    def __init__(self, hidden_size, units_per_expert, top_k, max_groups, group_rows=64):
        super().__init__(hidden_size, units_per_expert, top_k, max_groups, device_count=True)
        self.group_rows = group_rows
        self.mtiles = group_rows // 16

    @property
    def __cache_key__(self):
        return ("exlr8_a16_grouped", self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.group_rows, self.CTAS_PER_SM)

    @cute.jit
    def serving(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                group_expert: cute.Tensor, route_slots: cute.Tensor,
                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(4096) and cta_limit > Int32(0):
            self.grouped_fc1_kernel(x, pointers, bits, group_expert, route_slots,
                                    fc1, rows, group_count).launch(
                grid=(min(Int32(self.max_groups * self.units * 2 * self.splits * 4), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.activation_kernel(pointers, bits, group_expert, route_slots,
                                   fc1, intermediate, rows, group_count).launch(
                grid=(min(Int32(self.max_groups * self.units), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)
            self.grouped_fc2_kernel(pointers, bits, group_expert, route_slots,
                                    intermediate, fc2, rows, group_count).launch(
                grid=(min(Int32(self.max_groups * self.blocks * 4), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                               rows, group_count).launch(
                grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.jit
    def stage_a(self, smem: Int32, a_address: Int64, route_slots: cute.Tensor,
                group: Int32, rows: Int32, input_block: Int32, tid: Int32,
                a_stride: cutlass.Constexpr, down: cutlass.Constexpr):
        # Sixteen-byte vectors cover K128 once per CTA. XOR the vector column
        # with the row bank group so ldmatrix's eight rows use distinct banks.
        for i in cutlass.range_constexpr(self.group_rows * 16 // 128):
            vector = tid + Int32(i * 128)
            row, col = vector // Int32(16), vector % Int32(16)
            slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
            target = smem + (row * Int32(16) + (col ^ (row & Int32(7)))) * Int32(16)
            if slot >= Int32(0) and slot < rows * Int32(self.top_k):
                source_row = row if cutlass.const_expr(down) else slot // Int32(self.top_k)
                source = a_address + (source_row.to(Int64) * Int64(a_stride)
                    + input_block.to(Int64) * Int64(128) + col.to(Int64) * Int64(8)) * Int64(2)
                cp_async4_shared_global(target, source)
            else:
                st_shared_v4_u32(target, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    @cute.jit
    def accumulate(self, total: cute.Tensor, payload: Int64, smem: Int32,
                   input_block: Int32, output_block: Int32, nquad: Int32,
                   lane: Int32, warp: Int32, width: cutlass.Constexpr,
                   down: cutlass.Constexpr, scale: Float32):
        acc = cute.make_rmem_tensor((self.mtiles, 4), Float32)
        acc.fill(Float32(0.0))
        for k16 in cutlass.range_constexpr(8):
            tile = (input_block * Int32(8) + Int32(k16)) * Int32(8) + nquad * Int32(2) + warp // Int32(2)
            if cutlass.const_expr(down):
                tile = output_block * Int32(64) + Int32(k16 * 8) + nquad * Int32(2) + warp // Int32(2)
            b0, b1, b2, b3 = exlr8_decode_tile(payload, tile, lane, width)
            if warp % Int32(2) != Int32(0):
                b0, b1 = b2, b3
            for m in cutlass.range_constexpr(self.mtiles):
                row = Int32(m * 16) + (lane & Int32(15))
                col = Int32(k16 * 2) + lane // Int32(16)
                address = smem + (row * Int32(16) + (col ^ (row & Int32(7)))) * Int32(16)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(address)
                d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                    acc[m, 0], acc[m, 1], acc[m, 2], acc[m, 3], a0, a1, a2, a3, b0, b1)
                acc[m, 0], acc[m, 1], acc[m, 2], acc[m, 3] = d0, d1, d2, d3
        for m in cutlass.range_constexpr(self.mtiles):
            for v in cutlass.range_constexpr(4):
                total[m, v] = total[m, v] + acc[m, v] * scale

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

    @cute.kernel
    def grouped_fc1_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           fc1: cute.Tensor, rows: Int32, group_count: cute.Tensor):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, self.group_rows * 64], 1024]

        storage = smem.allocate(Storage)
        shared_a = shared_ptr_to_u32(storage.words.data_ptr())
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        while task < groups * Int32(self.units * 2 * self.splits * 4):
            nquad = task % Int32(4)
            logical = task // Int32(4)
            split = logical % Int32(self.splits)
            proj = (logical // Int32(self.splits)) % Int32(2)
            unit = (logical // Int32(2 * self.splits)) % Int32(self.units)
            group = logical // Int32(self.units * 2 * self.splits)
            expert = group_expert[group]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
            width = bits[desc]
            rids = self.row_ids(route_slots, group, rows, lane, False)
            total = cute.make_rmem_tensor((self.mtiles, 4), Float32)
            total.fill(Float32(0.0))
            if width != Int32(0):
                payload, scales = pointers[desc * Int64(3)], pointers[desc * Int64(3) + Int64(2)]
                for block in cutlass.range(8):
                    kb = split * Int32(8) + block
                    if kb < Int32(self.blocks):
                        scale = _side_value(scales, kb)
                        self.stage_a(shared_a, x.iterator.toint(), route_slots, group, rows,
                                     kb, tid, self.hidden_size, False)
                        if width == Int32(2):
                            self.accumulate(total, payload, shared_a,
                                            kb, Int32(0), nquad, lane, warp, 2, False, scale)
                        elif width == Int32(3):
                            self.accumulate(total, payload, shared_a,
                                            kb, Int32(0), nquad, lane, warp, 3, False, scale)
                        elif width == Int32(4):
                            self.accumulate(total, payload, shared_a,
                                            kb, Int32(0), nquad, lane, warp, 4, False, scale)
                        cute.arch.sync_threads()
            col = nquad * Int32(32) + warp * Int32(8) + (lane & Int32(3)) * Int32(2)
            for m in cutlass.range_constexpr(self.mtiles):
                for v in cutlass.range_constexpr(2):
                    local = Int32(m * 16 + v * 8) + lane // Int32(4)
                    base = (logical.to(Int64) * Int64(self.group_rows) + local.to(Int64)) * Int64(128) + col.to(Int64)
                    if width != Int32(0) and rids[m, v] >= Int32(0):
                        fc1[base], fc1[base + Int64(1)] = total[m, 2*v], total[m, 2*v+1]
            cute.arch.sync_threads()
            task += step

    @cute.kernel
    def grouped_fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, group_count: cute.Tensor):
        groups = group_count[0]
        if groups < Int32(0) or groups > Int32(self.max_groups):
            groups = Int32(0)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[Uint32, self.group_rows * 64], 1024]

        storage = smem.allocate(Storage)
        shared_a = shared_ptr_to_u32(storage.words.data_ptr())
        tid = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        while task < groups * Int32(self.blocks * 4):
            nquad = task % Int32(4)
            logical = task // Int32(4)
            group, block = logical // Int32(self.blocks), logical % Int32(self.blocks)
            expert = group_expert[group]
            rids = self.row_ids(route_slots, group, rows, lane, True)
            total = cute.make_rmem_tensor((self.mtiles, 4), Float32)
            total.fill(Float32(0.0))
            for unit in cutlass.range(self.units):
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                width = bits[desc]
                if width != Int32(0):
                    payload = pointers[desc * Int64(3)]
                    scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                    addr = intermediate.iterator.toint() + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(self.group_rows * 128 * 2)
                    self.stage_a(shared_a, addr, route_slots, group, rows, Int32(0), tid, 128, True)
                    if width == Int32(2):
                        self.accumulate(total, payload, shared_a, Int32(0), block,
                                        nquad, lane, warp, 2, True, scale)
                    elif width == Int32(3):
                        self.accumulate(total, payload, shared_a, Int32(0), block,
                                        nquad, lane, warp, 3, True, scale)
                    elif width == Int32(4):
                        self.accumulate(total, payload, shared_a, Int32(0), block,
                                        nquad, lane, warp, 4, True, scale)
                    cute.arch.sync_threads()
            col = block * Int32(128) + nquad * Int32(32) + warp * Int32(8) + (lane & Int32(3)) * Int32(2)
            for m in cutlass.range_constexpr(self.mtiles):
                for v in cutlass.range_constexpr(2):
                    local = Int32(m * 16 + v * 8) + lane // Int32(4)
                    base = (group.to(Int64) * Int64(self.group_rows) + local.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)
                    if rids[m, v] >= Int32(0):
                        fc2[base], fc2[base + Int64(1)] = total[m, 2*v], total[m, 2*v+1]
            cute.arch.sync_threads()
            task += step
