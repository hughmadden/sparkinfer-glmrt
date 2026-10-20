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


def test_region_selection_capacity_extent_and_masked_tail():
    """1M-extent rows at capacity 131072: short/full/invalid mixes stay exact.

    The shared top-k bounds every read by ``lengths``, so the masked capacity
    tail is unreachable; poison it with NaN/+inf/777 and require the candidate
    to both select exactly what the Triton oracle selects and to leave its own
    ``normalized`` tail untouched.
    """
    require_b12x()
    from b12x.attention.paged._region_selection import RegionSelectionWorkspace, select_regions
    from b12x.integration.cuteafd.region_selection import compile_region_selection_aot

    torch.manual_seed(274)
    groups, capacity, rows = 4, 131072, 6
    # The shared top-k emits ascending logical ids only while
    # (length + 31) >> 5 <= 2 * smem_candidate_capacity: 65,536 entries for
    # topk=512's 1,024-slot candidate buffers.  Past that it keeps the atomic
    # arrival order of the selection pass; the selected set by (score, lower
    # logical id) is unchanged.  Full-extent rows compare that exact set,
    # shorter rows compare bitwise.
    canonical_order_limit = 2 * 1024 * 32
    positions = torch.tensor([0, 7, 8232, 8 * capacity - 1, 8 * capacity + 3, 1048576],
                             device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 0, 0, -1], device="cuda", dtype=torch.int32)
    visible = torch.clamp(positions // 8, min=0, max=capacity)
    visible = torch.where(requests < 0, torch.zeros_like(visible), visible)
    tail = torch.arange(capacity, device="cuda")[None, :] >= visible[:, None]
    program = compile_region_selection_aot(groups=groups, capacity=capacity)
    assert int(visible.max()) == capacity and int(visible.min()) == 0
    for poison in (float("nan"), float("inf"), 777.0):
        scores = torch.randn(rows, groups, capacity, device="cuda")
        scores[:, 0, :4096:3] = 1.0
        scores[:, 0, 1:4096:3] = -0.0
        scores[:, 1, :4096:5] = -1.0
        scores[tail.unsqueeze(1).expand_as(scores)] = poison
        expected = torch.empty(rows, groups, 4104, device="cuda", dtype=torch.int32)
        workspace = RegionSelectionWorkspace.allocate(max_rows=rows, groups=groups,
                                                     capacity=capacity, device="cuda")
        select_regions(scores=scores, positions=positions, requests=requests,
                       selected=expected, workspace=workspace)
        output = torch.empty_like(expected)
        normalized = torch.full((rows, groups, capacity), 777.0, device="cuda")
        lengths = torch.empty(rows * groups, device="cuda", dtype=torch.int32)
        values = torch.empty(rows * groups, 512, device="cuda")
        ids = torch.empty(rows * groups, 512, device="cuda", dtype=torch.int32)
        program.launch(scores, positions, requests, output, normalized, lengths, values, ids,
                       scalars=[rows])
        torch.cuda.synchronize()
        canonical = torch.where(scores == 0, torch.zeros_like(scores), scores)
        for row in range(rows):
            length = int(visible[row])
            assert torch.equal(normalized[row, :, :length].view(torch.int32),
                               canonical[row, :, :length].view(torch.int32))
            assert (normalized[row, :, length:] == 777.0).all(), (
                f"row {row} wrote past the visible prefix")
            if length <= canonical_order_limit:
                assert torch.equal(output[row], expected[row]), (
                    f"selection changed with a {poison} masked tail at row {row}")
            else:
                assert torch.equal(torch.sort(output[row], dim=-1).values,
                                   torch.sort(expected[row], dim=-1).values), (
                    f"selected token set changed with a {poison} masked tail at row {row}")
                got = torch.sort(ids[row * groups:(row + 1) * groups], dim=-1).values
                want = torch.sort(workspace.ids[row * groups:(row + 1) * groups], dim=-1).values
                assert torch.equal(got, want), (
                    f"selected id set changed with a {poison} masked tail at row {row}")
