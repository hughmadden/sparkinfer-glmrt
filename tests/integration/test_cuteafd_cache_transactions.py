"""CuTe commits equal the staged Triton oracle byte for byte, including rejects."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x
from ._cuteafd import check_export


@pytest.mark.parametrize("heads", [2, 4])
def test_sliding_commit_matches_oracle_all_wraps_and_graph(heads):
    require_b12x()
    from b12x.attention.paged._cache_transactions import commit_sliding_ring
    from b12x.integration.cuteafd.cache_transactions import compile_sliding_commit_aot

    torch.manual_seed(274)
    olds = [0, 511, 512, 513, 1025, -1, 4, 4]
    counts = [0, 1, 9, 517, 520, 1, -1, 521]
    requests = len(olds)
    staged_k = torch.randn(requests * 520, heads, 192, device="cuda", dtype=torch.bfloat16)
    staged_v = torch.randn_like(staged_k)
    starts = torch.arange(requests, device="cuda", dtype=torch.int32) * 520
    lengths = torch.tensor(olds, device="cuda", dtype=torch.int64)
    accepted = torch.tensor(counts, device="cuda", dtype=torch.int32)
    available = torch.full_like(accepted, 520)
    rings = [torch.randn(requests, 512, heads, 192, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    saved = [t.clone() for t in rings]
    expected = [t.clone() for t in rings]
    commit_sliding_ring(staged_k=staged_k, staged_v=staged_v, starts=starts,
        base_lengths=lengths, accepted=accepted, staged_counts=available,
        ring_k=expected[0], ring_v=expected[1])
    program = compile_sliding_commit_aot(kv_heads=heads)

    def launch():
        program.launch(staged_k, staged_v, starts, lengths, accepted, available,
                       *rings, scalars=[requests])

    launch()
    for actual, wanted in zip(rings, expected):
        assert torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8))
    for actual, original in zip(rings, saved):
        actual.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    for actual, wanted in zip(rings, expected):
        assert torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8))


@pytest.mark.parametrize("accepted_count", [0, 1, 7, 8, 9, 17, 32, 33, -1])
def test_region_commit_matches_oracle_tail_alias_and_reject(accepted_count):
    require_b12x()
    from b12x.attention.paged._cache_transactions import commit_region_state
    from b12x.integration.cuteafd.cache_transactions import compile_region_commit_aot

    torch.manual_seed(275)
    requests, maximum, capacity = 8, 5, 16
    k = torch.randn(requests * 40, 256, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(k)
    starts = torch.arange(requests, device="cuda", dtype=torch.int32) * 40
    staged = torch.randint(0, 256, (requests, maximum, 256), device="cuda", dtype=torch.uint8)
    accepted = torch.full((requests,), accepted_count, device="cuda", dtype=torch.int32)
    available = torch.full_like(accepted, 32)
    state = [torch.randint(0, 256, (requests, capacity, 256), device="cuda", dtype=torch.uint8),
             torch.randn(requests, 8, 256, device="cuda", dtype=torch.bfloat16),
             torch.randn(requests, 8, 256, device="cuda", dtype=torch.bfloat16),
             torch.arange(16, 24, device="cuda", dtype=torch.int64)]
    saved = [t.clone() for t in state]
    expected = [t.clone() for t in state]
    commit_region_state(staged_k=k, staged_z=z, starts=starts,
        staged_summaries=staged.view(torch.float8_e4m3fn), base_lengths=expected[3],
        accepted=accepted, staged_counts=available, summaries=expected[0].view(torch.float8_e4m3fn),
        pending_k=expected[1], pending_z=expected[2], lengths=expected[3])
    program = compile_region_commit_aot(max_staged_regions=maximum, capacity=capacity)

    def launch():
        program.launch(k, z, starts, staged, state[3], accepted, available, *state,
                       scalars=[requests])

    launch()
    for actual, wanted in zip(state, expected):
        assert torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8))
    for actual, original in zip(state, saved):
        actual.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    for actual, wanted in zip(state, expected):
        assert torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8))


@pytest.mark.parametrize("operation", ["ring", "region"])
def test_cache_commit_export_abis(tmp_path, operation):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.cache_transactions import compile_region_commit_aot, compile_sliding_commit_aot
    with exportable_compilation():
        program = (compile_sliding_commit_aot(kv_heads=4) if operation == "ring"
                   else compile_region_commit_aot(max_staged_regions=5, capacity=16))
    check_export(program, tmp_path, "cache_commit_" + operation)
