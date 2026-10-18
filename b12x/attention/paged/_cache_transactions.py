"""Byte-copy commits from speculative staging; rollback discards staging.

These are metadata/packing kernels, not arithmetic reconstruction. The owner
must order all attention reads before commit on the stream and copy snapshots
only after commit completes. Snapshot restore is an ordinary exact buffer copy.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x._lib.compile_plan import launch_triton


@triton.jit
def _commit_ring(staged_k, staged_v, starts, base_lengths, accepted, staged_counts,
                 ring_k, ring_v, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    slot = tl.program_id(1)
    dim = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    old = tl.load(base_lengths + request).to(tl.int64)
    count = tl.load(accepted + request).to(tl.int64)
    available = tl.load(staged_counts + request).to(tl.int64)
    end = old + count
    # One writer per physical slot, even for accepted prefills longer than a ring.
    logical = end - 1 - ((end - 1) % 512 - slot + 512) % 512
    valid = (old >= 0) & (count > 0) & (count <= available) & (logical >= old)
    source = (tl.load(starts + request).to(tl.int64) + logical - old) * WIDTH + dim
    destination = (request.to(tl.int64) * 512 + slot) * WIDTH + dim
    mask = valid & (dim < WIDTH)
    key = tl.load(staged_k + source, mask=mask, other=0)
    value = tl.load(staged_v + source, mask=mask, other=0)
    tl.store(ring_k + destination, key, mask=mask)
    tl.store(ring_v + destination, value, mask=mask)


@triton.jit
def _commit_summaries(staged, base_lengths, accepted, staged_counts, published,
                      MAX_STAGED: tl.constexpr, CAPACITY: tl.constexpr):
    request = tl.program_id(0)
    region = tl.program_id(1)
    dim = tl.arange(0, 256)
    old = tl.load(base_lengths + request).to(tl.int64)
    count = tl.load(accepted + request).to(tl.int64)
    available = tl.load(staged_counts + request).to(tl.int64)
    completed = (old % 8 + count) // 8
    destination_region = old // 8 + region
    valid = (old >= 0) & (count >= 0) & (count <= available)
    valid = valid & (completed <= MAX_STAGED) & ((old + count) // 8 <= CAPACITY)
    valid = valid & (region < completed) & (destination_region < CAPACITY)
    source = (request.to(tl.int64) * MAX_STAGED + region) * 256 + dim
    destination = (request.to(tl.int64) * CAPACITY + destination_region) * 256 + dim
    # Preserve exact E4M3 bytes, including signed zero, without a float conversion.
    value = tl.load(staged + source, mask=valid, other=0)
    tl.store(published + destination, value, mask=valid)


@triton.jit
def _commit_pending(staged_k, staged_z, starts, base_lengths, accepted, staged_counts,
                    pending_k, pending_z, lengths, MAX_STAGED: tl.constexpr,
                    CAPACITY: tl.constexpr):
    request = tl.program_id(0)
    slot = tl.program_id(1)
    dim = tl.arange(0, 256)
    old = tl.load(base_lengths + request).to(tl.int64)
    count = tl.load(accepted + request).to(tl.int64)
    available = tl.load(staged_counts + request).to(tl.int64)
    total = old % 8 + count
    completed, tail = total // 8, total % 8
    valid = (old >= 0) & (count > 0) & (count <= available)
    valid = valid & (completed <= MAX_STAGED) & ((old + count) // 8 <= CAPACITY)
    source = (tl.load(starts + request).to(tl.int64) + completed * 8 + slot) * 256 + dim
    destination = (request.to(tl.int64) * 8 + slot) * 256 + dim
    key = tl.load(staged_k + source, mask=valid & (slot < tail), other=0)
    z = tl.load(staged_z + source, mask=valid & (slot < tail), other=0)
    tl.store(pending_k + destination, key, mask=valid)
    tl.store(pending_z + destination, z, mask=valid)


@triton.jit
def _publish_lengths(base_lengths, accepted, staged_counts, lengths,
                     MAX_STAGED: tl.constexpr, CAPACITY: tl.constexpr):
    request = tl.program_id(0)
    old = tl.load(base_lengths + request).to(tl.int64)
    count = tl.load(accepted + request).to(tl.int64)
    available = tl.load(staged_counts + request).to(tl.int64)
    valid = (old >= 0) & (count > 0) & (count <= available)
    valid = valid & ((old % 8 + count) // 8 <= MAX_STAGED) & ((old + count) // 8 <= CAPACITY)
    tl.store(lengths + request, old + count, mask=valid)


def _metadata(base_lengths, accepted, starts, staged_counts):
    n = base_lengths.numel()
    for tensor, dtype in ((base_lengths, torch.int64), (accepted, torch.int32),
                          (starts, torch.int32), (staged_counts, torch.int32)):
        if tensor.shape != (n,) or tensor.dtype != dtype:
            raise ValueError("transaction metadata requires I64 base lengths and I32 counts/starts")
    return n


def _storage(tensors):
    device = tensors[0].device
    if device.type != "cuda" or any(t.device != device or not t.is_contiguous() for t in tensors):
        raise ValueError("transaction buffers must be contiguous on one CUDA device")


def commit_sliding_ring(*, staged_k, staged_v, starts, base_lengths, accepted,
                        staged_counts, ring_k, ring_v):
    """Copy only accepted K/V into a physical 512-slot ring.

Staging contains new tokens only, with starts per request. This does not publish
lengths: the owner publishes after its layer commits. No writes occur on reject.
"""
    requests = _metadata(base_lengths, accepted, starts, staged_counts)
    if ring_k.shape != ring_v.shape or ring_k.ndim != 4 or ring_k.shape[:2] != (requests, 512):
        raise ValueError("ring K/V must be [requests,512,kv_heads,head_dim]")
    if staged_k.shape != staged_v.shape or staged_k.ndim != 3 or staged_k.shape[1:] != ring_k.shape[2:]:
        raise ValueError("staged K/V must be [tokens,kv_heads,head_dim]")
    if any(t.dtype != torch.bfloat16 for t in (staged_k, staged_v, ring_k, ring_v)):
        raise ValueError("sliding K/V must be BF16")
    _storage((staged_k, staged_v, starts, base_lengths, accepted, staged_counts, ring_k, ring_v))
    width = ring_k.shape[2] * ring_k.shape[3]
    if requests:
        launch_triton(_commit_ring, (requests, 512, triton.cdiv(width, 256)),
                      staged_k, staged_v, starts, base_lengths, accepted, staged_counts,
                      ring_k, ring_v, WIDTH=width, BLOCK=256)


def commit_region_state(*, staged_k, staged_z, starts, staged_summaries,
                        base_lengths, accepted, staged_counts, summaries,
                        pending_k, pending_z, lengths):
    """Publish accepted complete summaries and exact pending-tail bytes.

Per-request raw staging begins with the OLD pending prefix, followed by NEW
projected K/Z. Staged summaries are [requests,max_staged_regions,256] computed
from that concatenation. `staged_counts` and `accepted` count NEW tokens only.
Unused pending slots are canonically zeroed; no partial summary is published.
"""
    requests = _metadata(base_lengths, accepted, starts, staged_counts)
    if staged_k.shape != staged_z.shape or staged_k.ndim != 2 or staged_k.shape[1] != 256:
        raise ValueError("raw region staging must be [tokens,256]")
    if pending_k.shape != (requests, 8, 256) or pending_z.shape != pending_k.shape:
        raise ValueError("pending K/Z must be [requests,8,256]")
    for tensor in (staged_k, staged_z, pending_k, pending_z):
        if tensor.dtype != torch.bfloat16:
            raise ValueError("raw region tensors must be BF16")
    for tensor in (staged_summaries, summaries):
        if tensor.ndim != 3 or tensor.shape[0] != requests or tensor.shape[2] != 256 or tensor.dtype != torch.float8_e4m3fn:
            raise ValueError("summaries must be E4M3 [requests,capacity,256]")
    if lengths.shape != (requests,) or lengths.dtype != torch.int64:
        raise ValueError("committed lengths must be I64 [requests]")
    _storage((staged_k, staged_z, starts, staged_summaries, base_lengths, accepted,
              staged_counts, summaries, pending_k, pending_z, lengths))
    maximum, capacity = staged_summaries.shape[1], summaries.shape[1]
    if maximum <= 0 or capacity <= 0:
        raise ValueError("summary capacities must be positive")
    if requests:
        launch_triton(_commit_summaries, (requests, maximum), staged_summaries.view(torch.uint8),
                      base_lengths, accepted, staged_counts, summaries.view(torch.uint8),
                      MAX_STAGED=maximum, CAPACITY=capacity)
        launch_triton(_commit_pending, (requests, 8), staged_k, staged_z, starts, base_lengths,
                      accepted, staged_counts, pending_k, pending_z, lengths,
                      MAX_STAGED=maximum, CAPACITY=capacity)
        # A launch boundary makes in-place length publication safe for every CTA.
        launch_triton(_publish_lengths, (requests,), base_lengths, accepted, staged_counts,
                      lengths, MAX_STAGED=maximum, CAPACITY=capacity)
