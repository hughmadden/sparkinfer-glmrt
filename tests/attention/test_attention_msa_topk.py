"""MSA selection oracle: stable ties, packed bases and caller-owned replay."""
from __future__ import annotations

import pytest
import torch

from b12x.attention.dsa_indexer._impl import msa_topk_blocks
from b12x.attention.dsa_indexer.msa_reference import msa_select_blocks_reference


def scalar_oracle(scores, positions, bases, topk):
    heads, rows, blocks = scores.shape
    result = torch.full((heads, rows, topk), -1, dtype=torch.int32)
    for head in range(heads):
        for row in range(rows):
            values = scores[head, row].tolist()
            local = int(positions[row]) // 128
            if 0 <= local < blocks:
                values[local] = float("inf")
            chosen = sorted(range(blocks), key=lambda i: (-values[i], i))[:topk]
            chosen = sorted(i - int(bases[row]) for i in chosen
                            if values[i] > -float("inf") and i >= int(bases[row]))
            result[head, row, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("blocks,topk", [(0, 16), (1, 16), (18, 16), (257, 16), (1025, 16), (18, 0)])
def test_tie_heavy_selection_matches_scalar_oracle(device, blocks, topk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    scores = -(torch.arange(4 * 3 * blocks).reshape(4, 3, blocks) % 3).float()
    positions = torch.tensor([max(blocks - 1, 0) * 128, -1, 128], dtype=torch.int32)
    bases = torch.tensor([0, 0, 1], dtype=torch.int32)
    if blocks:
        scores[1, 1].fill_(-float("inf"))
        scores[2, 0].zero_()
        scores[3, 2, 0] = -float("inf")
    expected = scalar_oracle(scores, positions, bases, topk)
    kwargs = dict(block_scores=scores.to(device), query_positions=positions.to(device),
                  block_base=bases.to(device), topk=topk)
    original = kwargs["block_scores"].clone()
    assert torch.equal(msa_select_blocks_reference(**kwargs).cpu(), expected)
    assert torch.equal(msa_topk_blocks(**kwargs).cpu(), expected)
    assert torch.equal(kwargs["block_scores"], original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_accepts_strided_caller_scratch_and_preserves_source():
    device = "cuda"
    # Three-way ties straddle scan-tile boundaries and the top-k cutoff.
    scores = -(torch.arange(4 * 2 * 513, device=device).reshape(4, 2, 513) % 3).float()
    positions = torch.tensor([512 * 128, 511 * 128], device=device, dtype=torch.int32)
    values = torch.empty((4, 2, 32), device=device)[:, :, ::2]
    indices = torch.empty((4, 2, 32), device=device, dtype=torch.int64)[:, :, ::2]
    output = torch.empty((4, 2, 16), device=device, dtype=torch.int32)
    scratch = torch.empty((4, 2, 1026), device=device)[:, :, ::2]
    original = scores.clone()
    actual = msa_topk_blocks(block_scores=scores, query_positions=positions,
        out_indices=output, score_scratch=scratch, top_values=values, top_indices=indices)
    expected = scalar_oracle(scores.cpu(), positions.cpu(), torch.zeros(2, dtype=torch.int32), 16)
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(scores, original)
    assert actual.data_ptr() == output.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_buffers_replay_live_counts_and_score_mutation():
    device = "cuda"
    capacity = (4, 3, 1025)
    scores = torch.zeros(capacity, device=device)
    positions = torch.full((3,), 1024 * 128, dtype=torch.int32, device=device)
    bases = torch.zeros(3, dtype=torch.int32, device=device)
    output = torch.empty((4, 3, 16), dtype=torch.int32, device=device)
    scratch = torch.empty_like(scores)
    values = torch.empty((4, 3, 16), dtype=torch.float32, device=device)
    indices = torch.empty((4, 3, 16), dtype=torch.int64, device=device)
    sort_values = torch.empty_like(output)
    sort_indices = torch.empty_like(indices)

    def run(heads=4, rows=3, blocks=1025, topk=16):
        return msa_topk_blocks(
            block_scores=scores[:heads, :rows, :blocks],
            query_positions=positions[:rows], block_base=bases[:rows], topk=topk,
            out_indices=output, score_scratch=scratch,
            top_values=values, top_indices=indices,
            sort_values=sort_values, sort_indices=sort_indices,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    from b12x.attention.dsa_indexer.msa_topk import _stable_msa_topk
    cache = _stable_msa_topk.device_caches[torch.cuda.current_device()][0]
    compiled = len(cache)
    for heads, rows, blocks, topk in [(1, 1, 18, 8), (3, 2, 257, 16), (4, 3, 1025, 16)]:
        actual = run(heads, rows, blocks, topk)
        expected = scalar_oracle(scores[:heads, :rows, :blocks].cpu(), positions[:rows].cpu(), bases[:rows].cpu(), topk)
        assert torch.equal(actual.cpu(), expected)
        assert actual.data_ptr() == output.data_ptr()
    assert len(cache) == compiled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    assert captured.data_ptr() == output.data_ptr()
    # Poison previous outputs, change the winning blocks, then replay twice.
    for start in [100, 500]:
        scores.fill_(-100)
        scores[:, :, start:start + 16] = 10
        output.fill_(-99)
        graph.replay()
        torch.cuda.synchronize()
        expected = scalar_oracle(scores.cpu(), positions.cpu(), bases.cpu(), 16)
        assert torch.equal(output.cpu(), expected)
    assert len(cache) == compiled
