"""Raw-pointer attention ABIs preserve shared kernels and live-row reuse."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import check_export


def test_region_compress_pointer_abi_preserves_incomplete():
    require_b12x()
    from b12x.attention.paged._weighted_regions import compress_regions
    from b12x.integration.cuteafd.region_attention import compile_region_compress_aot

    torch.manual_seed(274)
    keys = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn_like(keys)
    starts = torch.arange(0, 64, 8, device="cuda", dtype=torch.int32)
    counts = torch.tensor([8, 8, 8, 8, 8, 8, 7, 0], device="cuda", dtype=torch.int32)
    output = torch.full((8, 256), 3., device="cuda").to(torch.float8_e4m3fn)
    raw = torch.full((8, 256), 3., device="cuda")
    expected, expected_raw = output.clone(), raw.clone()
    compress_regions(keys=keys, logits=logits, starts=starts, counts=counts,
                     output=expected, unrounded=expected_raw)
    program = compile_region_compress_aot()
    program.launch(keys, logits, starts, counts, output, raw, scalars=[8, 64])
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(raw, expected_raw)
    assert torch.equal(raw[6:], torch.full_like(raw[6:], 3.))


@pytest.mark.parametrize("groups", [2, 4])
def test_region_score_pointer_abi_signed_visibility_and_graph(groups):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._weighted_regions import score_regions
    from b12x.integration.cuteafd.region_attention import compile_region_scores_aot

    torch.manual_seed(275)
    rows, capacity = 4, 67
    query = torch.randn(rows, groups, 4, 256, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, groups, 4, device="cuda")
    summaries = torch.randn(2, capacity, 256, device="cuda").to(torch.float8_e4m3fn)
    requests = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 8, 17, 8 * capacity], device="cuda", dtype=torch.int64)
    expected = torch.empty(rows, groups, capacity, device="cuda")
    score_regions(query=query, weights=weights, summaries=summaries, requests=requests,
                  positions=positions, output=expected)
    output = torch.full_like(expected, 123.)
    program = compile_region_scores_aot(groups=groups, capacity=capacity)
    program.launch(query, weights, summaries, requests, positions, output, scalars=[rows, 2])
    assert torch.equal(output, expected)
    with kernel_resolution_guard("region score pointer ABI graph"):
        for live in (1, rows):
            output.fill_(123.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(query, weights, summaries, requests, positions, output, scalars=[live, 2])
            graph.replay()
            assert torch.equal(output[:live], expected[:live])
            assert torch.equal(output[live:], torch.full_like(output[live:], 123.))


@pytest.mark.parametrize("heads,kv_heads", [(32, 2), (64, 4)])
@pytest.mark.parametrize("width", [512, 4104])
def test_selected_gqa_pointer_abi_d192_and_live_rows(heads, kv_heads, width):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
    from b12x.integration.cuteafd.region_attention import compile_selected_gqa_aot

    torch.manual_seed(276)
    rows, page_rows = 4, 64
    groups = 1 if width == 512 else kv_heads
    query = torch.randn(rows, heads, 192, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(3, page_rows, kv_heads, 192, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.tensor([[2, 0], [0, 1]], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    positions = torch.full((rows,), 127, device="cuda", dtype=torch.int64)
    selected = torch.full((rows, groups, width), -1, device="cuda", dtype=torch.int32)
    for group in range(groups):
        selected[:, group, :16] = torch.arange(group * 8, group * 8 + 16, device="cuda")
    selected[:, :, -1] = 127
    expected = torch.empty_like(query)
    launch_selected_paged_gqa_direct(query=query, key_cache=keys, value_cache=values,
        k_descale=None, v_descale=None, block_table=table, request_ids=requests,
        selected_positions=selected, query_positions=positions, output=expected,
        softmax_scale=192**-0.5, kv_warps=2)
    output = torch.full_like(query, 123.)
    program = compile_selected_gqa_aot(query_heads=heads, kv_heads=kv_heads,
        page_rows=page_rows, selection_width=width, selection_groups=groups)

    def launch(live):
        program.launch(query, keys, values, None, None, table, requests, selected,
            positions, None, None, output, scalars=[3, 2, 2, 192**-0.5, live, 1])

    launch(rows)
    assert torch.equal(output, expected)
    with kernel_resolution_guard("selected GQA pointer ABI graph"):
        for live in (1, rows):
            output.fill_(123.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(live)
            graph.replay()
            assert torch.equal(output[:live], expected[:live])
            assert torch.equal(output[live:], torch.full_like(output[live:], 123.))


@pytest.mark.parametrize("operation", ["compress", "scores2", "scores4", "swa32", "csa64"])
def test_region_attention_export_abis(tmp_path, operation):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import region_attention as r

    builders = {
        "compress": r.compile_region_compress_aot,
        "scores2": lambda: r.compile_region_scores_aot(groups=2, capacity=67),
        "scores4": lambda: r.compile_region_scores_aot(groups=4, capacity=67),
        "swa32": lambda: r.compile_selected_gqa_aot(query_heads=32, kv_heads=2,
            selection_width=512, selection_groups=1),
        "csa64": lambda: r.compile_selected_gqa_aot(query_heads=64, kv_heads=4),
    }
    with exportable_compilation():
        program = builders[operation]()
    check_export(program, tmp_path, "region_" + operation)
