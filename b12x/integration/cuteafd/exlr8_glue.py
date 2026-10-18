"""Device-driven EXLR8 grouping and shared hidden-domain transforms."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


class _Group:
    def __init__(self, experts, top_k, max_groups):
        self.experts, self.top_k, self.max_groups = experts, top_k, max_groups

    @cute.jit
    def __call__(self, ids: cute.Pointer, group_expert: cute.Pointer,
                 route_slots: cute.Pointer, route_group_lane: cute.Pointer,
                 group_count: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(64):
            self.kernel(ids, group_expert, route_slots, route_group_lane,
                        group_count, rows).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, group_expert: cute.Pointer,
               route_slots: cute.Pointer, route_group_lane: cute.Pointer,
               group_count: cute.Pointer, rows: Int32):
        thread = Int32(cute.arch.thread_idx()[0])
        smem = utils.SmemAllocator()
        prefix = smem.allocate_tensor(Int32, cute.make_layout(256), byte_alignment=16)
        valid = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        routes = rows * Int32(self.top_k)
        if thread == Int32(0):
            valid[0] = Int32(1)
            for route in cutlass.range(routes):
                if ids[route] < Int32(0) or ids[route] >= Int32(self.experts):
                    valid[0] = Int32(0)
        for route in cutlass.range(thread, routes, 256):
            route_group_lane[route] = Int32(-1)
        count = Int32(0)
        if thread < Int32(self.experts):
            for route in cutlass.range(routes):
                if ids[route] == thread:
                    count += Int32(1)
        groups = (count + Int32(15)) // Int32(16)
        prefix[thread] = groups
        cute.arch.sync_threads()
        # Inclusive scan of per-expert group counts; neither atomics nor route
        # arrival order can change ascending-expert/ascending-route packing.
        for step in cutlass.range_constexpr(8):
            distance = Int32(1 << step)
            addend = Int32(0)
            if thread >= distance:
                addend = prefix[thread - distance]
            cute.arch.sync_threads()
            prefix[thread] = prefix[thread] + addend
            cute.arch.sync_threads()
        total = prefix[Int32(255)]
        if thread == Int32(0):
            group_count[0] = Int32(-1)
            if valid[0] != Int32(0) and total <= Int32(self.max_groups):
                group_count[0] = total
        if valid[0] != Int32(0) and total <= Int32(self.max_groups):
            first = prefix[thread] - groups
            for local in cutlass.range(groups):
                group = first + local
                group_expert[group] = thread
                for lane in cutlass.range_constexpr(16):
                    route_slots[group * Int32(16) + Int32(lane)] = Int32(-1)
            ordinal = Int32(0)
            if thread < Int32(self.experts):
                for route in cutlass.range(routes):
                    if ids[route] == thread:
                        packed = first * Int32(16) + ordinal
                        route_slots[packed] = route
                        route_group_lane[route] = packed
                        ordinal += Int32(1)


class _Rotate:
    def __init__(self, hidden, max_rows):
        self.hidden, self.blocks, self.max_rows = hidden, hidden // 128, max_rows

    @cute.jit
    def __call__(self, x: cute.Pointer, signs: cute.Pointer, output: cute.Pointer,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            self.kernel(x, signs, output, rows).launch(
                grid=(min(rows * Int32(self.blocks), cta_limit), 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, signs: cute.Pointer, output: cute.Pointer, rows: Int32):
        lane = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        while task < rows * Int32(self.blocks):
            block = task % Int32(self.blocks)
            base = task.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
            values = cute.make_rmem_tensor(4, Float32)
            for v in cutlass.range_constexpr(4):
                values[v] = x[base + Int64(v)].to(Float32) * signs[block * Int32(128) + lane * Int32(4) + Int32(v)].to(Float32)
            a, b, c, d = _w4a8_had128_quad(values[0], values[1], values[2], values[3], lane)
            output[base] = a.to(cutlass.Float16)
            output[base + Int64(1)] = b.to(cutlass.Float16)
            output[base + Int64(2)] = c.to(cutlass.Float16)
            output[base + Int64(3)] = d.to(cutlass.Float16)
            task += step


class _Reduce:
    def __init__(self, hidden, max_ranks, max_rows):
        self.hidden, self.blocks = hidden, hidden // 128
        self.max_ranks, self.max_rows = max_ranks, max_rows

    @cute.jit
    def __call__(self, planes: cute.Pointer, shared: cute.Pointer, signs: cute.Pointer,
                 residual: cute.Pointer, output: cute.Pointer, rows: Int32,
                 ranks: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and ranks >= Int32(0) and ranks <= Int32(self.max_ranks) and cta_limit > Int32(0):
            self.kernel(planes, shared, signs, residual, output, rows, ranks).launch(
                grid=(min(rows * Int32(self.blocks), cta_limit), 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, planes: cute.Pointer, shared: cute.Pointer, signs: cute.Pointer,
               residual: cute.Pointer, output: cute.Pointer, rows: Int32, ranks: Int32):
        lane = Int32(cute.arch.thread_idx()[0])
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        while task < rows * Int32(self.blocks):
            block = task % Int32(self.blocks)
            base = task.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0))
            for rank in cutlass.range(ranks):
                plane = cute.make_ptr(Float32, planes[rank], cute.AddressSpace.gmem, assumed_align=16)
                for v in cutlass.range_constexpr(4):
                    values[v] = values[v] + plane[base + Int64(v)]
            for v in cutlass.range_constexpr(4):
                values[v] = values[v] + shared[base + Int64(v)]
            a, b, c, d = _w4a8_had128_quad(values[0], values[1], values[2], values[3], lane)
            values[0], values[1], values[2], values[3] = a, b, c, d
            for v in cutlass.range_constexpr(4):
                delta = values[v] * signs[block * Int32(128) + lane * Int32(4) + Int32(v)].to(Float32)
                output[base + Int64(v)] = (delta + residual[base + Int64(v)].to(Float32)).to(cutlass.BFloat16)
            task += step


def compile_exlr8_group_aot(*, experts=256, top_k=8, max_groups=512):
    if (any(type(v) is not int for v in (experts, top_k, max_groups))
            or not (1 <= experts <= 256 and 1 <= top_k <= 32
                    and 0 < max_groups <= (2**31 - 1) // 16)):
        raise ValueError("EXLR8 grouping requires experts1..256, top_k1..32 and positive capacity")
    return compile_program(_Group(experts, top_k, max_groups), name="exlr8_group",
        operands=(Operand("ids", torch.int32, f"[rows,{top_k}]"),
                  Operand("group_expert", torch.int32, f"[{max_groups}]", "out"),
                  Operand("route_slots", torch.int32, f"[{max_groups},16]", "out"),
                  Operand("route_group_lane", torch.int32, f"[rows,{top_k}]", "out"),
                  Operand("group_count", torch.int32, "[1]", "out")),
        scalars=(Scalar("rows"),), key=(experts, top_k, max_groups),
        geometry=dict(abi_version=1, experts=experts, top_k=top_k, max_groups=max_groups, max_rows=64))


def _check_transform_geometry(hidden, max_rows):
    if (any(type(v) is not int or v <= 0 or v > 2**31 - 1
            for v in (hidden, max_rows)) or hidden % 128
            or max_rows * (hidden // 128) > (2**31 - 1) // 2
            or max_rows * hidden * 4 > 2**63 - 1):
        raise ValueError("EXLR8 transform geometry exceeds alignment or indexing bounds")


def compile_exlr8_rotate_aot(*, hidden=6144, max_rows=4096):
    _check_transform_geometry(hidden, max_rows)
    return compile_program(_Rotate(hidden, max_rows), name="exlr8_rotate",
        operands=(Operand("x", torch.bfloat16, f"[rows,{hidden}]"),
                  Operand("signs", torch.int8, f"[{hidden}]"),
                  Operand("output", torch.float16, f"[rows,{hidden}]", "out")),
        scalars=(Scalar("rows"), Scalar("cta_limit")), key=(hidden, max_rows),
        geometry=dict(abi_version=1, hidden=hidden, max_rows=max_rows, ctas_per_sm=4))


def compile_exlr8_reduce_aot(*, hidden=6144, max_ranks=6, max_rows=4096):
    _check_transform_geometry(hidden, max_rows)
    if type(max_ranks) is not int or not 0 <= max_ranks <= 2**31 - 1:
        raise ValueError("invalid EXLR8 reduction rank capacity")
    return compile_program(_Reduce(hidden, max_ranks, max_rows), name="exlr8_reduce",
        operands=(Operand("planes", torch.int64, f"[{max_ranks}]"),
                  Operand("shared", torch.float32, f"[rows,{hidden}]"),
                  Operand("signs", torch.int8, f"[{hidden}]"),
                  Operand("residual", torch.bfloat16, f"[rows,{hidden}]"),
                  Operand("output", torch.bfloat16, f"[rows,{hidden}]", "out")),
        scalars=(Scalar("rows"), Scalar("ranks"), Scalar("cta_limit")), key=(hidden, max_ranks, max_rows),
        geometry=dict(abi_version=1, hidden=hidden, max_ranks=max_ranks, max_rows=max_rows, ctas_per_sm=4))
