"""CPU-only EXLR8 runtime-L2 FC2 slice geometry & task-mapping termination tests."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py"
TREE = ast.parse(SOURCE.read_text())

_CONST_NAMES = {
    "_FC1_CONSUMER_THREADS",
    "_FC1_PRODUCER_THREADS",
    "_FC1_PRODUCER_REGS",
    "_FC1_CONSUMER_REGS",
    "_FC1_CTA_MAX_REGS",
    "_FC1_FULL_GRAND_PRODUCER_REGS",
    "_FC1_FULL_GRAND_CONSUMER_REGS",
}
_FUNC_NAMES = {
    "slice_blocks_for_shift",
    "resolve_fc2_slice_geometry",
    "fc2_ws_slice_task_mapping",
    "fc2_slice_task_mapping",
    "reduce_slice_task_mapping",
    "validate_fc2_slice_geometry",
}


def _load_slice_helpers() -> dict:
    nodes: list[ast.stmt] = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in _CONST_NAMES:
                nodes.append(node)
        elif isinstance(node, ast.Assert):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _FUNC_NAMES:
            nodes.append(node)
    scope: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


HELPERS = _load_slice_helpers()
slice_blocks_for_shift = HELPERS["slice_blocks_for_shift"]
resolve_fc2_slice_geometry = HELPERS["resolve_fc2_slice_geometry"]
fc2_ws_slice_task_mapping = HELPERS["fc2_ws_slice_task_mapping"]
fc2_slice_task_mapping = HELPERS["fc2_slice_task_mapping"]
reduce_slice_task_mapping = HELPERS["reduce_slice_task_mapping"]
validate_fc2_slice_geometry = HELPERS["validate_fc2_slice_geometry"]


def test_fc1_ws_register_budget_within_ctapool():
    producer_regs = HELPERS["_FC1_PRODUCER_REGS"]
    consumer_regs = HELPERS["_FC1_CONSUMER_REGS"]
    cta_max_regs = HELPERS["_FC1_CTA_MAX_REGS"]
    assert producer_regs == 88
    assert consumer_regs == 208
    assert cta_max_regs == 3 * 168 == 504
    assert producer_regs + 2 * consumer_regs == cta_max_regs
    full_producer_regs = HELPERS["_FC1_FULL_GRAND_PRODUCER_REGS"]
    full_consumer_regs = HELPERS["_FC1_FULL_GRAND_CONSUMER_REGS"]
    assert full_producer_regs == 40
    assert full_consumer_regs == 232
    assert full_producer_regs + 2 * full_consumer_regs == cta_max_regs


@pytest.mark.parametrize(
    "rows,cta_limit,unit_limit,l2_mib,expected_shift,expected_blocks",
    [
        # Low-occupancy branch (rows * top_k <= (cta_limit + 4) * 6 => rows <= 256 on 170/188 SMs)
        (1, 376, 8, 128, 1, 6),
        (16, 376, 8, 128, 1, 6),
        (64, 376, 8, 128, 1, 6),
        (256, 376, 8, 128, 1, 6),
        (1, 340, 8, 96, 1, 6),
        (256, 340, 8, 96, 1, 6),
        # RTX PRO 6000 (188 SMs -> cta_limit=376, TP2 unit_limit=8, 128 MiB L2)
        (1024, 376, 8, 128, 3, 24),
        (2752, 376, 8, 128, 1, 6),
        (4096, 376, 8, 128, 2, 12),
        # 96 MiB L2 cap on RTX PRO 6000 (cta_limit=376) and RTX 5090 (170 SMs -> cta_limit=340)
        (1024, 376, 8, 96, 3, 24),
        (2752, 376, 8, 96, 1, 6),
        (4096, 376, 8, 96, 1, 6),
        (1024, 340, 8, 96, 3, 24),
        (2752, 340, 8, 96, 1, 6),
        (4096, 340, 8, 96, 1, 6),
        # Minimum slices cannot retain the live intermediate in a tight L2.
        (4096, 376, 8, 32, 0, 48),
        # GB10: retain the winning six-block schedule, avoid 24 minimum slices.
        (1024, 96, 8, 24, 1, 6),
        (2752, 96, 8, 24, 0, 48),
        (4096, 96, 8, 24, 0, 48),
    ],
)
def test_resolved_l2_slice_geometries(
    rows: int,
    cta_limit: int,
    unit_limit: int,
    l2_mib: int,
    expected_shift: int,
    expected_blocks: int,
):
    l2_bytes = l2_mib * 1024 * 1024
    slice_shift, slice_blocks = resolve_fc2_slice_geometry(
        rows, cta_limit, unit_limit, l2_bytes, top_k=8, fc2_half=True, blocks=48
    )
    print(
        f"geometry rows={rows:4d} cta_limit={cta_limit:3d} l2_mib={l2_mib:3d} -> "
        f"slice_shift={slice_shift} slice_blocks={slice_blocks}"
    )
    assert (slice_shift, slice_blocks) == (expected_shift, expected_blocks)
    validate_fc2_slice_geometry(
        rows=rows,
        cta_limit=cta_limit,
        slice_shift=slice_shift,
        slice_blocks=slice_blocks,
        blocks=48,
        max_groups=512,
        ctas_per_sm=2,
    )


@pytest.mark.parametrize(
    "cta_limit,l2_mib,segments",
    [
        (376, 128, [(285, 1, 6), (1664, 3, 24), (2662, 2, 12),
                    (3803, 1, 6), (4095, 0, 2), (4096, 2, 12)]),
        (376, 96, [(285, 1, 6), (1216, 3, 24), (1945, 2, 12),
                   (2779, 1, 6), (2943, 0, 2), (3072, 2, 12), (4096, 1, 6)]),
        (340, 96, [(258, 1, 6), (1216, 3, 24), (1945, 2, 12),
                   (2779, 1, 6), (2943, 0, 2), (3072, 2, 12), (4096, 1, 6)]),
    ],
)
def test_rtx_every_row_preserves_duel_winner_geometry(cta_limit, l2_mib, segments):
    # Frozen ff602159 selector boundaries, including its narrow two-block bands.
    start = 1
    for end, shift, blocks in segments:
        for rows in range(start, end + 1):
            assert resolve_fc2_slice_geometry(rows, cta_limit, 8, l2_mib * 1024**2) == (shift, blocks)
        start = end + 1
    assert start == 4097


@pytest.mark.parametrize(
    "slice_shift,expected_blocks",
    [
        (0, 2),
        (1, 6),
        (2, 12),
        (3, 24),
        (4, 16),
    ],
)
@pytest.mark.parametrize("cta_limit", [340, 376])
@pytest.mark.parametrize(
    "groups",
    [
        0,
        1,
        7,
        62,
        94,
        188,
        189,
        256,
        376,
        512,
    ],
)
def test_every_slice_shift_fc2_task_mapping_bijective_and_terminating(
    slice_shift: int,
    expected_blocks: int,
    cta_limit: int,
    groups: int,
):
    slice_blocks = slice_blocks_for_shift(slice_shift, blocks=48)
    assert slice_blocks == expected_blocks
    assert 48 % slice_blocks == 0
    step = max(1, cta_limit // 2)

    expected_domain = {(g, b) for g in range(groups) for b in range(slice_blocks)}

    # 1. Warp-specialized paired-N256 FC2 kernel (grouped_fc2_ws_kernel)
    limit_ws, b_pairs, use_chunks_ws, ws_tasks = fc2_ws_slice_task_mapping(
        groups, step, slice_blocks, slice_shift, blocks=48
    )
    assert len(ws_tasks) == limit_ws
    covered_ws = [
        (g, b_local_base + 2 * pair_idx + j)
        for g, b_local_base in ws_tasks
        for pair_idx in range(b_pairs)
        for j in range(2)
    ]
    assert len(covered_ws) == groups * slice_blocks
    assert set(covered_ws) == expected_domain
    assert all(0 <= g < groups and 0 <= b < slice_blocks for g, b in covered_ws)

    # 2. Non-WS FC2 fallback kernel (grouped_fc2_kernel)
    limit_f, b_count, use_chunks_f, f_tasks = fc2_slice_task_mapping(
        groups, step, slice_blocks, slice_shift, blocks=48
    )
    assert len(f_tasks) == limit_f
    covered_f = [
        (g, b_local_base + b_idx)
        for g, b_local_base in f_tasks
        for b_idx in range(b_count)
    ]
    assert len(covered_f) == groups * slice_blocks
    assert set(covered_f) == expected_domain


@pytest.mark.parametrize("slice_shift", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rows", [1, 16, 64, 256, 1024, 2752, 4096])
@pytest.mark.parametrize("cta_limit", [340, 376])
def test_every_slice_shift_reduce_task_mapping_bijective_and_terminating(
    slice_shift: int,
    rows: int,
    cta_limit: int,
):
    slice_blocks = slice_blocks_for_shift(slice_shift, blocks=48)
    grid = min((rows * slice_blocks + 7) // 8, cta_limit * 2)
    assert grid >= 1
    limit_r, step_r, r_tasks = reduce_slice_task_mapping(
        rows, grid, slice_blocks, slice_shift, blocks=48
    )
    assert step_r == grid * 8 > 0
    assert len(r_tasks) == rows * slice_blocks == limit_r
    assert len(set(r_tasks)) == rows * slice_blocks
    assert min(r for r, _ in r_tasks) == 0 and max(r for r, _ in r_tasks) == rows - 1
    assert min(b for _, b in r_tasks) == 0 and max(b for _, b in r_tasks) == slice_blocks - 1


def test_unsliced_geometry_and_invalid_geometries_rejected():
    # Unsliced 48-block geometry (slice_blocks == 48)
    validate_fc2_slice_geometry(
        rows=64,
        cta_limit=376,
        slice_shift=0,
        slice_blocks=48,
        blocks=48,
    )
    # Non-48 hidden block count (e.g. hidden=4096 -> blocks=32)
    shift32, blocks32 = resolve_fc2_slice_geometry(1024, 376, 8, 96 * 1024 * 1024, top_k=4, blocks=32)
    assert (shift32, blocks32) == (0, 32)
    validate_fc2_slice_geometry(
        rows=64,
        cta_limit=376,
        slice_shift=shift32,
        slice_blocks=blocks32,
        blocks=32,
    )
    # Invalid geometries fail closed via ValueError
    with pytest.raises(ValueError):
        validate_fc2_slice_geometry(rows=1024, cta_limit=0, slice_shift=1, slice_blocks=6)
    with pytest.raises(ValueError):
        validate_fc2_slice_geometry(rows=1024, cta_limit=376, slice_shift=1, slice_blocks=0)
    with pytest.raises(ValueError):
        validate_fc2_slice_geometry(rows=1024, cta_limit=376, slice_shift=2, slice_blocks=6)
    with pytest.raises(ValueError):
        slice_blocks_for_shift(5, blocks=48)
