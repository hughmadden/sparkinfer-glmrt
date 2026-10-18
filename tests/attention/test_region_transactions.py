from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("old_length", [0, 511, 512, 513, 1025])
@pytest.mark.parametrize("accepted", [0, 1, 9, 517])
def test_ring_commit_is_identical_to_serial_and_reject_is_untouched(old_length, accepted):
    from b12x.attention.paged._cache_transactions import commit_sliding_ring
    device = require_b12x()
    torch.manual_seed(274)
    width = (2, 192)
    all_k = torch.randn(old_length + 520, *width, dtype=torch.bfloat16)
    all_v = torch.randn_like(all_k)
    ring_k = torch.zeros(1, 512, *width, dtype=torch.bfloat16)
    ring_v = torch.zeros_like(ring_k)
    for token in range(old_length):
        ring_k[0, token % 512] = all_k[token]
        ring_v[0, token % 512] = all_v[token]
    expected_k, expected_v = ring_k.clone(), ring_v.clone()
    for token in range(old_length, old_length + accepted):
        expected_k[0, token % 512] = all_k[token]
        expected_v[0, token % 512] = all_v[token]
    ring_k, ring_v = ring_k.to(device), ring_v.to(device)
    snapshot = (ring_k.clone(), ring_v.clone())
    kwargs = dict(staged_k=all_k[old_length:].to(device), staged_v=all_v[old_length:].to(device),
                  starts=torch.tensor([0], dtype=torch.int32, device=device),
                  base_lengths=torch.tensor([old_length], dtype=torch.int64, device=device),
                  accepted=torch.tensor([accepted], dtype=torch.int32, device=device),
                  staged_counts=torch.tensor([520], dtype=torch.int32, device=device),
                  ring_k=ring_k, ring_v=ring_v)
    commit_sliding_ring(**kwargs)
    assert torch.equal(ring_k.cpu(), expected_k)
    assert torch.equal(ring_v.cpu(), expected_v)
    ring_k.copy_(snapshot[0])
    ring_v.copy_(snapshot[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_sliding_ring(**kwargs)
    graph.replay()
    assert torch.equal(ring_k.cpu(), expected_k)
    assert torch.equal(ring_v.cpu(), expected_v)


@pytest.mark.parametrize("accepted", [0, 1, 7, 8, 9, 17, 32])
def test_region_commit_all_tail_offsets_exact_and_snapshot_restore(accepted):
    from b12x.attention.paged._cache_transactions import commit_region_state
    from b12x.attention.paged._weighted_regions import compress_regions
    device = require_b12x()
    torch.manual_seed(275)
    requests, capacity, staged_capacity = 8, 16, 5
    keys = torch.randn(requests, 64, 256, dtype=torch.bfloat16, device=device)
    z = torch.randn_like(keys)
    old_lengths = torch.arange(16, 24, dtype=torch.int64, device=device)
    # Fixed-stride raw staging contains old pending bytes followed by 32 new rows.
    raw_k = torch.zeros(requests, 40, 256, dtype=torch.bfloat16, device=device)
    raw_z = torch.zeros_like(raw_k)
    for request in range(requests):
        raw_k[request, :request + 32] = keys[request, 16:48+request]
        raw_z[request, :request + 32] = z[request, 16:48+request]
    starts = torch.arange(0, requests * 40, 8, dtype=torch.int32, device=device)
    counts = torch.tensor([8 if region < (r+32)//8 else 0 for r in range(requests) for region in range(5)], dtype=torch.int32, device=device)
    staged = torch.zeros(requests, staged_capacity, 256, device=device).to(torch.float8_e4m3fn)
    unrounded = torch.empty(requests * staged_capacity, 256, device=device)
    compress_regions(keys=raw_k.view(-1, 256), logits=raw_z.view(-1, 256), starts=starts,
                     counts=counts, output=staged.view(-1, 256), unrounded=unrounded)
    published = torch.full((requests, capacity, 256), 2.0, device=device).to(torch.float8_e4m3fn)
    pending_k = torch.zeros(requests, 8, 256, dtype=torch.bfloat16, device=device)
    pending_z = torch.zeros_like(pending_k)
    for request in range(requests):
        pending_k[request, :request] = keys[request, 16:16+request]
        pending_z[request, :request] = z[request, 16:16+request]
    lengths = old_lengths.clone()
    snapshot = tuple(t.clone() for t in (published, pending_k, pending_z, lengths))
    kwargs = dict(staged_k=raw_k.view(-1, 256), staged_z=raw_z.view(-1, 256),
                  starts=torch.arange(requests, device=device, dtype=torch.int32)*40,
                  staged_summaries=staged, base_lengths=lengths,
                  accepted=torch.full((requests,), accepted, dtype=torch.int32, device=device),
                  staged_counts=torch.full((requests,), 32, dtype=torch.int32, device=device),
                  summaries=published, pending_k=pending_k, pending_z=pending_z, lengths=lengths)
    commit_region_state(**kwargs)
    first = tuple(t.clone() for t in (published, pending_k, pending_z, lengths))
    for request in range(requests):
        end = 16 + request + accepted
        completed = (request + accepted) // 8
        expected = snapshot[0][request].view(torch.uint8).clone()
        expected[2:2+completed] = staged[request, :completed].view(torch.uint8)
        assert torch.equal(published[request].view(torch.uint8), expected)
        for tensor, original in ((pending_k, keys), (pending_z, z)):
            expected_tail = torch.zeros(8, 256, dtype=torch.bfloat16, device=device)
            expected_tail[:end % 8] = original[request, end-end%8:end]
            assert torch.equal(tensor[request], expected_tail)
        assert int(lengths[request]) == end
    # Restoring a prefix is byte copying, never undoing the weighted reduction.
    for tensor, saved in zip((published, pending_k, pending_z, lengths), snapshot):
        tensor.copy_(saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_region_state(**kwargs)
    graph.replay()
    for tensor, expected in zip((published, pending_k, pending_z, lengths), first):
        assert torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8))
