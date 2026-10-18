from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from ..conftest import require_b12x


def compression_reference(keys, logits):
    z = logits.float()
    weights = (z - z.amax()).exp()
    return (weights * keys.float()).sum(0) / weights.sum(0).clamp_min(1e-20)


def score_reference(query, weights, summary):
    q = query.to(torch.float8_e4m3fn).float()
    dots = torch.einsum("ghd,rd->ghr", q, summary.float())
    return (dots.clamp_min(0) * weights[..., None]).sum(1)


def official_kernel():
    path = os.environ.get("STEP5_REFERENCE_KERNEL")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("step5_official_kernel", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module during the import.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cpu_signed_weights_and_two_roundings():
    q = torch.ones(1, 4, 256, dtype=torch.bfloat16)
    w = torch.tensor([[1.0, -2.0, 0.0, 0.0]])
    s = torch.ones(2, 256).to(torch.float8_e4m3fn)
    assert torch.equal(score_reference(q, w, s), torch.full((1, 2), -256.0))
    # A BF16 midpoint can change an E4M3 tie, so the intermediate cast is required.
    x = torch.tensor([1.063, 1.187])
    assert not torch.equal(x.to(torch.float8_e4m3fn).view(torch.uint8),
                           x.bfloat16().to(torch.float8_e4m3fn).view(torch.uint8))


def test_compression_matches_cpu_and_official_and_preserves_incomplete():
    from b12x.attention.paged._weighted_regions import compress_regions
    device = require_b12x()
    torch.manual_seed(274)
    keys = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    logits = torch.randn_like(keys)
    starts = torch.arange(0, 64, 8, device=device, dtype=torch.int32)
    counts = torch.tensor([8, 8, 8, 8, 8, 8, 7, 0], device=device, dtype=torch.int32)
    output = torch.full((8, 256), 3.0, device=device).to(torch.float8_e4m3fn)
    raw = torch.full((8, 256), 3.0, device=device)
    compress_regions(keys=keys, logits=logits, starts=starts, counts=counts,
                     output=output, unrounded=raw)
    expected = torch.stack([compression_reference(keys[i:i+8].cpu(), logits[i:i+8].cpu())
                            for i in range(0, 48, 8)])
    torch.testing.assert_close(raw[:6].cpu(), expected, atol=3e-7, rtol=3e-6)
    assert torch.equal(output[:6].cpu().view(torch.uint8), expected.bfloat16().to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(raw[6:], torch.full_like(raw[6:], 3))
    assert torch.equal(output[6:].float(), torch.full_like(raw[6:], 3))
    official = official_kernel()
    if official is not None:
        reference_raw, reference_bytes = official.csa_compress_regions(
            keys[:, None], logits[:, None], starts, counts)
        torch.testing.assert_close(raw[:6], reference_raw[:6, 0], atol=3e-7, rtol=3e-6)
        assert torch.equal(output[:6].view(torch.uint8), reference_bytes[:6, 0].view(torch.uint8))


def test_weighted_mma_scores_causal_signed_and_official_graph():
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._weighted_regions import score_regions
    device = require_b12x()
    torch.manual_seed(275)
    rows, groups, capacity = 4, 4, 67
    query = torch.randn(rows, groups, 4, 256, device=device, dtype=torch.bfloat16)
    weights = torch.randn(rows, groups, 4, device=device)
    summaries = torch.randn(2, capacity, 256, device=device).to(torch.float8_e4m3fn)
    requests = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.int32)
    positions = torch.tensor([0, 8, 17, 8*capacity], device=device, dtype=torch.int64)
    output = torch.empty(rows, groups, capacity, device=device)

    def launch(n):
        score_regions(query=query[:n], weights=weights[:n], summaries=summaries,
                      requests=requests[:n], positions=positions[:n], output=output[:n])

    launch(1)
    with kernel_resolution_guard("CSA score live rows"):
        for n in (1, rows):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(n)
            graph.replay()
            for row in range(n):
                visible = int(positions[row]) // 8
                expected = score_reference(query[row].cpu(), weights[row].cpu(),
                                           summaries[int(requests[row])].cpu())
                torch.testing.assert_close(output[row, :, :visible].cpu(), expected[:, :visible],
                                           atol=3e-5, rtol=3e-6)
                assert torch.isneginf(output[row, :, visible:]).all()
    official = official_kernel()
    if official is not None:
        for row in range(rows):
            expected = official.indexer_logits(query[row:row+1], weights[row:row+1],
                                               summaries[int(requests[row])][:, None].bfloat16())
            visible = int(positions[row]) // 8
            torch.testing.assert_close(output[row, :, :visible], expected[:, :visible], atol=3e-5, rtol=3e-6)


@pytest.mark.parametrize("capacity", [7, 513, 1031])
def test_region_selection_signed_ties_matches_official(capacity):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._region_selection import RegionSelectionWorkspace, select_regions
    device = require_b12x()
    rows, groups = 4, 2
    scores = torch.zeros(rows, groups, capacity, device=device)
    # Negative scores and an exact zero tie crossing the selection boundary.
    scores[:, 0, ::3] = -1.0
    scores[:, 0, 1::3] = -0.0
    scores[:, 1] = -torch.arange(capacity, device=device).flip(0).float()
    positions = torch.tensor([0, 7, 8*min(513, capacity)+3, 8*capacity+7], dtype=torch.int64, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    selected = torch.empty(rows, groups, 4104, dtype=torch.int32, device=device)
    workspace = RegionSelectionWorkspace.allocate(max_rows=rows, groups=groups, capacity=capacity, device=device)

    def launch(n):
        select_regions(scores=scores[:n], positions=positions[:n], requests=requests[:n],
                       selected=selected[:n], workspace=workspace)

    launch(1)
    with kernel_resolution_guard("signed CSA topk row reuse"):
        for n in (1, rows):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(n)
            graph.replay()
            for row in range(n):
                position = int(positions[row])
                visible = position // 8
                for group in range(groups):
                    values = scores[row, group, :visible].cpu()
                    ids = sorted(range(visible), key=lambda i: (-float(values[i]), i))[:512]
                    ids.sort()
                    expected = [token for i in ids for token in range(i*8, i*8+8)]
                    expected += list(range(visible*8, position+1))
                    actual = selected[row, group]
                    actual = actual[actual >= 0].cpu().tolist()
                    assert actual == expected
    official = official_kernel()
    if official is not None:
        lengths = (positions // 8).repeat_interleave(groups).int()
        expected = official.region_topk_ids(scores.view(rows*groups, capacity), lengths, topk=512)
        for row in range(rows*groups):
            actual = workspace.stable_ids[row].cpu().tolist()
            wanted = expected[row].cpu().tolist()
            assert sorted(i for i in actual if i >= 0) == sorted(i for i in wanted if i >= 0)


def test_sliding_selection_wrap_boundaries_and_padding():
    from b12x.attention.paged._region_selection import sliding_positions
    device = require_b12x()
    positions = torch.tensor([0, 510, 511, 512, 1023, 1024], dtype=torch.int64, device=device)
    requests = torch.tensor([0, 0, 0, 0, 0, -1], dtype=torch.int32, device=device)
    selected = torch.empty(6, 512, dtype=torch.int32, device=device)
    sliding_positions(positions=positions, requests=requests, selected=selected)
    for row, position in enumerate(positions.cpu().tolist()):
        expected = list(range(max(0, position-511), position+1)) if row < 5 else []
        assert selected[row][selected[row] >= 0].cpu().tolist() == expected


def test_sparse_gqa_d192_against_official_prefill():
    from b12x.attention.paged._region_selection import RegionSelectionWorkspace, select_regions
    from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
    official = official_kernel()
    if official is None:
        pytest.skip("STEP5_REFERENCE_KERNEL not provided")
    device = require_b12x()
    torch.manual_seed(276)
    rows, groups, capacity, page = 3, 2, 521, 16
    q = torch.randn(rows, groups*16, 192, dtype=torch.bfloat16, device=device)
    pages = (capacity*8 + 15)//16 + 1
    k = torch.randn(pages, page, groups, 192, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    scores = torch.randn(rows, groups, capacity, device=device)
    scores[:, :, ::3] = 0
    positions = torch.tensor([7, 513*8+2, capacity*8+7], dtype=torch.int64, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).flip(0)[None].contiguous()
    selected = torch.empty(rows, groups, 4104, dtype=torch.int32, device=device)
    workspace = RegionSelectionWorkspace.allocate(max_rows=rows, groups=groups, capacity=capacity, device=device)
    select_regions(scores=scores, positions=positions, requests=requests, selected=selected, workspace=workspace)
    output = torch.empty_like(q)
    launch_selected_paged_gqa_direct(
        query=q, key_cache=k, value_cache=v, k_descale=None, v_descale=None,
        block_table=table, request_ids=requests, selected_positions=selected,
        query_positions=positions, output=output, softmax_scale=192**-0.5, kv_warps=2)
    packed, counts = official.prefill_sparse_meta(
        scores.permute(1, 0, 2).contiguous().view(groups*rows, capacity),
        positions.int().repeat(groups), table, requests.repeat(groups), topk=512,
        region_size=8, regions_per_page=2)
    expected, _ = official.sparse_attention_prefill(q, k.view(-1, groups, 192), v.view(-1, groups, 192),
                                                   packed, counts, num_kv_groups=groups, block_regions=4)
    torch.testing.assert_close(output.float(), expected.float(), atol=0.006, rtol=0.006)
