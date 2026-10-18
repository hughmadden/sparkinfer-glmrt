"""Allocation-free byte-copy commits; the caller orders reads before publication."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint16
import torch

from ._common import Operand, Scalar, compile_program


class _RingCommit:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, staged_k: cute.Pointer, staged_v: cute.Pointer,
                 starts: cute.Pointer, base_lengths: cute.Pointer,
                 accepted: cute.Pointer, staged_counts: cute.Pointer,
                 ring_k: cute.Pointer, ring_v: cute.Pointer,
                 requests: Int32, stream: cuda.CUstream):
        self.kernel(staged_k, staged_v, starts, base_lengths, accepted,
                    staged_counts, ring_k, ring_v).launch(
            grid=(requests, 512, (self.width + 255) // 256), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, staged_k: cute.Pointer, staged_v: cute.Pointer,
               starts: cute.Pointer, base_lengths: cute.Pointer,
               accepted: cute.Pointer, staged_counts: cute.Pointer,
               ring_k: cute.Pointer, ring_v: cute.Pointer):
        request, slot, tile = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        dim = tile * 256 + thread
        old = base_lengths[request].to(Int64)
        count = accepted[request].to(Int64)
        available = staged_counts[request].to(Int64)
        end = old + count
        if (old >= 0) & (count > 0) & (count <= available):
            logical = end - 1 - ((end - 1) % 512 - slot + 512) % 512
            if (logical >= old) & (dim < self.width):
                source = (starts[request].to(Int64) + logical - old) * self.width + dim
                destination = (Int64(request) * 512 + slot) * self.width + dim
                sk = cute.recast_ptr(staged_k, dtype=Uint16)
                sv = cute.recast_ptr(staged_v, dtype=Uint16)
                rk = cute.recast_ptr(ring_k, dtype=Uint16)
                rv = cute.recast_ptr(ring_v, dtype=Uint16)
                rk[destination] = sk[source]
                rv[destination] = sv[source]


class _RegionCommit:
    def __init__(self, maximum, capacity):
        self.maximum, self.capacity = maximum, capacity

    @cute.jit
    def __call__(self, staged_k: cute.Pointer, staged_z: cute.Pointer,
                 starts: cute.Pointer, staged_summaries: cute.Pointer,
                 base_lengths: cute.Pointer, accepted: cute.Pointer,
                 staged_counts: cute.Pointer, summaries: cute.Pointer,
                 pending_k: cute.Pointer, pending_z: cute.Pointer,
                 lengths: cute.Pointer, requests: Int32, stream: cuda.CUstream):
        self.summaries(staged_summaries, base_lengths, accepted, staged_counts,
                       summaries).launch(grid=(requests, self.maximum, 1), block=(256, 1, 1), stream=stream)
        self.pending(staged_k, staged_z, starts, base_lengths, accepted,
                     staged_counts, pending_k, pending_z).launch(
            grid=(requests, 8, 1), block=(256, 1, 1), stream=stream)
        # Separate launch: every CTA has finished reading aliased base lengths.
        self.publish(base_lengths, accepted, staged_counts, lengths).launch(
            grid=(requests, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def summaries(self, staged: cute.Pointer, base_lengths: cute.Pointer,
                  accepted: cute.Pointer, staged_counts: cute.Pointer, published: cute.Pointer):
        request, region, _ = cute.arch.block_idx()
        dim, _, _ = cute.arch.thread_idx()
        old = base_lengths[request].to(Int64)
        count = accepted[request].to(Int64)
        available = staged_counts[request].to(Int64)
        if (old >= 0) & (count >= 0) & (count <= available):
            completed = (old % 8 + count) // 8
            destination_region = old // 8 + region
            if (completed <= self.maximum) & ((old + count) // 8 <= self.capacity):
                if (region < completed) & (destination_region < self.capacity):
                    source = (Int64(request) * self.maximum + region) * 256 + dim
                    destination = (Int64(request) * self.capacity + destination_region) * 256 + dim
                    published[destination] = staged[source]

    @cute.kernel
    def pending(self, staged_k: cute.Pointer, staged_z: cute.Pointer,
                starts: cute.Pointer, base_lengths: cute.Pointer,
                accepted: cute.Pointer, staged_counts: cute.Pointer,
                pending_k: cute.Pointer, pending_z: cute.Pointer):
        request, slot, _ = cute.arch.block_idx()
        dim, _, _ = cute.arch.thread_idx()
        old = base_lengths[request].to(Int64)
        count = accepted[request].to(Int64)
        available = staged_counts[request].to(Int64)
        if (old >= 0) & (count > 0) & (count <= available):
            completed = (old % 8 + count) // 8
            tail = (old % 8 + count) % 8
            if (completed <= self.maximum) & ((old + count) // 8 <= self.capacity):
                source = (starts[request].to(Int64) + completed * 8 + slot) * 256 + dim
                destination = (Int64(request) * 8 + slot) * 256 + dim
                sk = cute.recast_ptr(staged_k, dtype=Uint16)
                sz = cute.recast_ptr(staged_z, dtype=Uint16)
                pk = cute.recast_ptr(pending_k, dtype=Uint16)
                pz = cute.recast_ptr(pending_z, dtype=Uint16)
                key, z = Uint16(0), Uint16(0)
                if slot < tail:
                    key, z = Uint16(sk[source]), Uint16(sz[source])
                pk[destination], pz[destination] = key, z

    @cute.kernel
    def publish(self, base_lengths: cute.Pointer, accepted: cute.Pointer,
                staged_counts: cute.Pointer, lengths: cute.Pointer):
        request, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        if thread == 0:
            old = base_lengths[request].to(Int64)
            count = accepted[request].to(Int64)
            available = staged_counts[request].to(Int64)
            if (old >= 0) & (count > 0) & (count <= available):
                if ((old % 8 + count) // 8 <= self.maximum) & ((old + count) // 8 <= self.capacity):
                    lengths[request] = old + count


def compile_sliding_commit_aot(*, kv_heads, head_dim=192):
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("positive sliding-cache geometry required")
    width = int(kv_heads) * int(head_dim)
    return compile_program(_RingCommit(width), name="sliding_cache_commit", operands=(
        Operand("staged_k", torch.bfloat16, f"[tokens,{width}]"),
        Operand("staged_v", torch.bfloat16, f"[tokens,{width}]"),
        Operand("starts", torch.int32, "[requests]", align=4),
        Operand("base_lengths", torch.int64, "[requests]", align=8),
        Operand("accepted", torch.int32, "[requests]", align=4),
        Operand("staged_counts", torch.int32, "[requests]", align=4),
        Operand("ring_k", torch.bfloat16, f"[requests,512,{width}]", "inout"),
        Operand("ring_v", torch.bfloat16, f"[requests,512,{width}]", "inout")),
        scalars=(Scalar("requests"),), key=(width,),
        geometry=dict(kv_heads=int(kv_heads), head_dim=int(head_dim), ring_rows=512))


def compile_region_commit_aot(*, max_staged_regions, capacity):
    maximum, capacity = int(max_staged_regions), int(capacity)
    if min(maximum, capacity) <= 0:
        raise ValueError("positive staged and published summary capacities required")
    return compile_program(_RegionCommit(maximum, capacity), name="region_cache_commit", operands=(
        Operand("staged_k", torch.bfloat16, "[tokens,256]"),
        Operand("staged_z", torch.bfloat16, "[tokens,256]"),
        Operand("starts", torch.int32, "[requests]", align=4),
        Operand("staged_summaries", torch.uint8, f"[requests,{maximum},256] E4M3 bytes"),
        Operand("base_lengths", torch.int64, "[requests] (may alias lengths)", align=8),
        Operand("accepted", torch.int32, "[requests]", align=4),
        Operand("staged_counts", torch.int32, "[requests]", align=4),
        Operand("summaries", torch.uint8, f"[requests,{capacity},256] E4M3 bytes", "inout"),
        Operand("pending_k", torch.bfloat16, "[requests,8,256]", "inout"),
        Operand("pending_z", torch.bfloat16, "[requests,8,256]", "inout"),
        Operand("lengths", torch.int64, "[requests]", "inout", align=8)),
        scalars=(Scalar("requests"),), key=(maximum, capacity),
        geometry=dict(max_staged_regions=maximum, capacity=capacity, proxy_dim=256))
