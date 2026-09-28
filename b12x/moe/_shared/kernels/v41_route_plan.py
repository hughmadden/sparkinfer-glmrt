"""Allocation-free stable route grouping for the experimental V4.1 slices.

All buffers are caller-owned. Capacity and expert geometry are static; live
rows are read from a device scalar on every replay. Metadata capacity must be
capacity * topk (the worst case of one group per route). Invalid expert IDs
produce inverse=-1. Duplicate IDs are retained as separate routes.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint32, Float32, range_constexpr


class V41RoutePlan:
    def __init__(self, capacity, experts=384, topk=6):
        assert capacity > 0 and experts > 0 and topk > 0
        self.capacity = capacity
        self.experts = experts
        self.topk = topk
        self.routes = capacity * topk

    # Decode capacities plan in one CTA: every live route has its own thread,
    # and each expert/route reads the (broadcast) shared route table instead
    # of three grid launches. Grouping, order and outputs are identical.
    SMALL_THREADS = 512

    def small(self):
        return self.routes <= self.SMALL_THREADS and self.experts <= self.SMALL_THREADS

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        live_rows: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.small()):
            self.plan_small(ids, weights, live_rows, metadata, grouped_weights, inverse).launch(
                grid=(1, 1, 1), block=(self.SMALL_THREADS, 1, 1), stream=stream
            )
        else:
            self.plan_grid(ids, weights, live_rows, packed, counts, prefixes, metadata,
                grouped_weights, inverse, stream)

    @cute.kernel
    def plan_small(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        live_rows: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        route_expert = smem.allocate_tensor(Int32, cute.make_layout(self.SMALL_THREADS), byte_alignment=16)
        route_rank = smem.allocate_tensor(Int32, cute.make_layout(self.SMALL_THREADS), byte_alignment=16)
        expert_count = smem.allocate_tensor(Int32, cute.make_layout(self.SMALL_THREADS), byte_alignment=16)
        route_prefix = smem.allocate_tensor(Int32, cute.make_layout(self.SMALL_THREADS), byte_alignment=16)
        group_prefix = smem.allocate_tensor(Int32, cute.make_layout(self.SMALL_THREADS), byte_alignment=16)
        rows = max(Int32(0), min(live_rows[0], Int32(self.capacity)))
        live = rows * self.topk
        if tid < self.routes:
            metadata[tid, 1] = Int32(0)
            inverse[tid] = Int32(-1)
        expert = Int32(-1)
        if tid < live:
            candidate = Int32(ids[tid])
            if candidate >= 0 and candidate < self.experts:
                expert = candidate
        route_expert[tid] = expert
        cute.arch.sync_threads()
        # A route's rank is the number of earlier routes to its expert: the
        # stable ballot order the grid planner packs.
        rank = Int32(-1)
        if expert >= 0:
            rank = Int32(0)
            for q in range(tid):
                rank += Int32(route_expert[q] == expert)
        route_rank[tid] = rank
        cute.arch.sync_threads()
        # Per expert: its route count, the routes of smaller experts, and the
        # groups (runs of 16) that start at smaller experts. All loops cover
        # live routes only, never the 384-expert table.
        if tid < self.experts:
            count = Int32(0)
            routes = Int32(0)
            groups = Int32(0)
            for q in range(live):
                other = route_expert[q]
                if other >= 0:
                    count += Int32(other == tid)
                    if other < tid:
                        routes += 1
                        groups += Int32(route_rank[q] % 16 == 0)
            expert_count[tid] = count
            route_prefix[tid] = routes
            group_prefix[tid] = groups
        cute.arch.sync_threads()
        if expert >= 0:
            count = expert_count[expert]
            base = route_prefix[expert]
            group = group_prefix[expert] + rank // 16
            local = rank % 16
            if local == 0:
                metadata[group, 0] = expert
                metadata[group, 1] = min(Int32(16), count - rank)
                metadata[group, 2] = base + rank
                for pad in range(min(Int32(16), count - rank), 16):
                    metadata[group, 3 + pad] = Int32(-1)
            metadata[group, 3 + local] = tid // self.topk
            grouped_weights[base + rank] = weights[tid]
            inverse[tid] = base + rank

    @cute.jit
    def plan_grid(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        live_rows: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.pack(ids, live_rows, packed, counts, metadata, inverse).launch(
            grid=(self.experts, 1, 1), block=(32, 1, 1), stream=stream
        )
        self.prefix(counts, prefixes).launch(
            grid=((self.experts + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )
        self.scatter(
            weights, packed, counts, prefixes, metadata, grouped_weights, inverse
        ).launch(grid=(self.experts, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def pack(
        self,
        ids: cute.Tensor,
        live_rows: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        metadata: cute.Tensor,
        inverse: cute.Tensor,
    ):
        expert = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        for i in range(expert * 32 + lane, self.routes, self.experts * 32):
            metadata[i, 1] = Int32(0)
            inverse[i] = Int32(-1)
        rows = max(Int32(0), min(live_rows[0], Int32(self.capacity)))
        total = Int32(0)
        for base in range(0, rows * self.topk, 32):
            pair = base + lane
            match = False
            if pair < rows * self.topk:
                match = ids[pair] == expert
            ballot = cute.arch.vote_ballot_sync(match)
            offset = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
            if match:
                packed[Int64(expert) * self.routes + total + offset] = pair
            total += cute.arch.popc(ballot)
        if lane == 0:
            counts[expert] = total

    @cute.kernel
    def prefix(self, counts: cute.Tensor, prefixes: cute.Tensor):
        expert = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if expert < self.experts:
            routes = Int32(0)
            groups = Int32(0)
            for e in range(expert):
                count = counts[e]
                routes += count
                groups += (count + 16 - 1) // 16
            prefixes[expert, 0] = routes
            prefixes[expert, 1] = groups

    @cute.kernel
    def scatter(
        self,
        weights: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
    ):
        expert = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        count = counts[expert]
        base = prefixes[expert, 0]
        group_base = prefixes[expert, 1]
        for j in range(lane, ((count + 16 - 1) // 16) * 16, 32):
            group = group_base + j // 16
            local = j % 16
            if local == 0:
                metadata[group, 0] = expert
                metadata[group, 1] = min(Int32(16), count - j)
                metadata[group, 2] = base + j
            row = Int32(-1)
            if j < count:
                pair = packed[Int64(expert) * self.routes + j]
                row = pair // self.topk
                grouped_weights[base + j] = weights[pair]
                inverse[pair] = base + j
            metadata[group, 3 + local] = row


class V41SliceReduce:
    """Ordered FP32 slice sum into original route order; invalid routes are zero."""

    def __init__(self, width, capacity, topk=6, *, intermediate=576, hidden=5120):
        assert width in (64, 128, 192) and capacity > 0 and topk > 0
        self.hidden = hidden
        self.capacity = capacity
        assert intermediate > 0 and intermediate % 32 == 0
        self.slices = (intermediate + width - 1) // width
        self.routes = capacity * topk
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
        stream: cuda.CUstream,
        rows: Int32 = -1,
    ):
        launch_rows = self.capacity if rows < 0 else min(rows, self.capacity)
        self.kernel(source, dest, inverse, live_rows).launch(
            grid=(max(1, (launch_rows * self.topk * self.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
    ):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        route, col = index // self.hidden, index % self.hidden
        if route < min(live_rows[0] * self.topk, self.routes):
            grouped = inverse[route]
            value = Float32(0)
            if grouped >= 0:
                for plane in range_constexpr(self.slices):
                    value += source[plane, Int64(grouped), col]
            dest[route, col] = value
