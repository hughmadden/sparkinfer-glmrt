"""Causal CuTe selection equals the signed-score Triton metadata oracle."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x
from ._cuteafd import check_export


@pytest.mark.parametrize("capacity", [7, 513, 1031])
@pytest.mark.parametrize("groups", [2, 4])
def test_region_selection_exact_signed_ties_graph_and_live_rows(capacity, groups):
    require_b12x()
    from b12x.attention.paged._region_selection import RegionSelectionWorkspace, select_regions
    from b12x.integration.cuteafd.region_selection import compile_region_selection_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    torch.manual_seed(274)
    rows = 5
    scores = torch.randn(rows, groups, capacity, device="cuda")
    scores[:, 0] = 0.
    scores[:, 0, ::3] = -1.
    scores[:, 0, 1::3] = -0.
    scores[:, 1] = -torch.arange(capacity, device="cuda").float().flip(0)
    positions = torch.tensor([0, 7, 8 * min(513, capacity) + 3, 8 * capacity + 7, 23],
                             device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 0, -1], device="cuda", dtype=torch.int32)
    expected = torch.empty(rows, groups, 4104, device="cuda", dtype=torch.int32)
    workspace = RegionSelectionWorkspace.allocate(max_rows=rows, groups=groups, capacity=capacity, device="cuda")
    select_regions(scores=scores, positions=positions, requests=requests, selected=expected, workspace=workspace)
    output = torch.empty_like(expected)
    normalized = torch.empty_like(scores)
    lengths = torch.empty(rows * groups, device="cuda", dtype=torch.int32)
    values = torch.empty(rows * groups, 512, device="cuda")
    ids = torch.empty(rows * groups, 512, device="cuda", dtype=torch.int32)
    program = compile_region_selection_aot(groups=groups, capacity=capacity)

    def launch(live):
        program.launch(scores, positions, requests, output, normalized, lengths, values, ids, scalars=[live])

    launch(rows)
    assert torch.equal(output, expected)
    with kernel_resolution_guard("region metadata AOT graph"):
        for live in (1, rows):
            output.fill_(123)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(live)
            graph.replay()
            assert torch.equal(output[:live], expected[:live])
            assert torch.equal(output[live:], torch.full_like(output[live:], 123))


def test_sliding_positions_exact_wrap_and_padding():
    require_b12x()
    from b12x.attention.paged._region_selection import sliding_positions
    from b12x.integration.cuteafd.region_selection import compile_sliding_positions_aot
    positions = torch.tensor([0, 510, 511, 512, 1023, 1024], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 0, 0, -1], device="cuda", dtype=torch.int32)
    expected = torch.empty(6, 512, device="cuda", dtype=torch.int32)
    sliding_positions(positions=positions, requests=requests, selected=expected)
    output = torch.empty_like(expected)
    compile_sliding_positions_aot().launch(positions, requests, output, scalars=[6])
    assert torch.equal(output, expected)


@pytest.mark.parametrize("operation", ["sliding", "regions"])
def test_selection_export_abi(tmp_path, operation):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.region_selection import compile_region_selection_aot, compile_sliding_positions_aot
    with exportable_compilation():
        program = (compile_sliding_positions_aot() if operation == "sliding"
                   else compile_region_selection_aot(groups=4, capacity=1031))
    check_export(program, tmp_path, "region_selection_" + operation)
