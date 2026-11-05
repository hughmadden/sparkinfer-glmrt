"""Cross-tree bitwise oracle for selected-position paged GQA (Step-5 and Qwen).

The paged selected-attention duel changes only
``b12x/attention/paged/_selected_forward.py``. This file pins that kernel's
results for every shape the duel's bench covers, on the two real entry points:

* ``direct``  - one CTA per (row, KV head), BF16 output, no split scratch. This
  is the production Step-5 ABI (``region_attention.compile_selected_gqa_aot``)
  and Qwen's prefill route (more than 64 rows).
* ``split``   - FP32 split partials plus LSE, reduced by the merge kernel. This
  is Qwen's decode route (up to 64 rows) and the Step-5 latency census route.

Geometries are the shipped ones:

* Step-5 CSA   - 64 query / 4 KV heads of 192, 64-row pages, 4104-wide
  per-KV-head selections (``export_b12x_step5_aot.py``, region attention).
* Step-5 SWA   - 64 query / 4 KV heads of 192, 64-row pages, one shared
  512-wide sliding window.
* Qwen 3.8 Flash Next - 24 query / 2 KV heads of 256, 64-row pages,
  2112-wide indices (``Qwen4Geometry`` in ``integration/cuteafd/_common.py``).

Usage, baseline tree first, then the candidate tree with the same file::

    PAGED_SELECTED_GOLDEN=$DIR/golden.pt PAGED_SELECTED_MODE=write \
        python -m pytest -q tests/attention/test_paged_selected_oracle.py
    PAGED_SELECTED_GOLDEN=$DIR/golden.pt PAGED_SELECTED_MODE=check \
        python -m pytest -q tests/attention/test_paged_selected_oracle.py

``write`` records BF16 output, FP32 partials and FP32 LSE for every cell;
``check`` requires ``torch.equal`` on the raw bit patterns of all of them. The
golden also carries a SHA-256 of the cell specification, so an edited corpus
fails loudly instead of comparing different shapes. Inputs are generated on
the CPU from fixed seeds, so both trees see identical bytes.

Every checked cell is additionally compared against an independent FP64
reference (rows=1 cells), so a golden captured from a broken tree cannot
silently bless itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

_MODE = os.environ.get("PAGED_SELECTED_MODE", "check")
_GOLDEN = os.environ.get("PAGED_SELECTED_GOLDEN")

pytestmark = pytest.mark.skipif(
    not _GOLDEN,
    reason="set PAGED_SELECTED_GOLDEN (and PAGED_SELECTED_MODE=write|check) to run the cross-tree oracle",
)

# 8K context: the duel's screen size. Step-5 sees 8K/8 compressed regions (CSA)
# and one 512-token window (SWA); Qwen's selector sees its full 2048 budget.
_HISTORY = 8192

_GEOMETRIES = {
    "step5-csa": dict(
        q_heads=64, kv_heads=4, head_dim=192, page_rows=64,
        selection_width=4104, selection_groups=4, live=1024, layout="regions",
    ),
    "step5-swa": dict(
        q_heads=64, kv_heads=4, head_dim=192, page_rows=64,
        selection_width=512, selection_groups=1, live=512, layout="tail",
    ),
    "qwen": dict(
        q_heads=24, kv_heads=2, head_dim=256, page_rows=64,
        selection_width=2112, selection_groups=1, live=2048, layout="prefix",
    ),
}

# (family, route, rows). Rows are the real decode batch (C1/C4/C16) plus one
# 128-row unit of the prefill route; Step-5 runs direct at every row count.
_CASES = (
    ("step5-csa", "direct", (1, 4, 16, 128)),
    ("step5-csa", "split", (1, 16)),
    ("step5-swa", "direct", (1, 4, 16, 128)),
    ("step5-swa", "split", (16,)),
    ("qwen", "split", (1, 4, 16)),
    ("qwen", "direct", (128,)),
)

_CASE_LABELS = tuple(
    f"{family}-{route}-r{rows}" for family, route, rows_set in _CASES for rows in rows_set
)


def _splits(family: str, rows: int) -> int:
    """Split count each production route plans for this row count."""
    if family != "qwen":
        return 16
    if rows == 1:
        return 64
    if rows <= 4:
        return 32
    return 16


def _spec() -> dict[str, object]:
    return {
        "history": _HISTORY,
        "geometries": _GEOMETRIES,
        "cases": [[family, route, list(rows)] for family, route, rows in _CASES],
    }


_SPEC_SHA256 = hashlib.sha256(
    json.dumps(_spec(), sort_keys=True).encode("utf-8")
).hexdigest()


def _randn_cpu(shape, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    return values.to(dtype)


def _fill_positions(positions: torch.Tensor, layout: str, live: int) -> None:
    """Write one selection row in its production layout.

    ``tail`` is the SWA sliding window (the last ``live`` of 512 columns),
    ``regions`` is the CSA expansion (dense prefix plus the current region in
    the 4096..4103 tail, so the last tile always holds a selected position),
    and ``prefix`` is Qwen's ascending index list.
    """
    width = positions.shape[0]
    positions.fill_(-1)
    if layout == "prefix":
        positions[:live] = torch.arange(live, dtype=torch.int32)
    elif layout == "tail":
        positions[width - live:] = torch.arange(live, dtype=torch.int32)
    elif layout == "regions":
        tail_region = (live - 1) // 8
        dense = 8 * tail_region
        positions[:dense] = torch.arange(dense, dtype=torch.int32)
        positions[dense:4096] = -1
        positions[4096:4104] = torch.arange(dense, dense + 8, dtype=torch.int32)
    else:  # pragma: no cover - internal corpus only
        raise ValueError(layout)


def _build_case(family: str, rows: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Deterministic inputs for one (family, rows) cell, built on the CPU."""
    geometry = _GEOMETRIES[family]
    page_rows = geometry["page_rows"]
    kv_heads = geometry["kv_heads"]
    q_heads = geometry["q_heads"]
    head_dim = geometry["head_dim"]
    width = geometry["selection_width"]
    groups = geometry["selection_groups"]
    live = min(geometry["live"], _HISTORY)
    pages = max(1, (live + page_rows - 1) // page_rows)
    seed = int.from_bytes(
        hashlib.sha256(f"{family}-{rows}".encode("utf-8")).digest()[:4], "little"
    ) % (2**31)

    query = _randn_cpu((rows, q_heads, head_dim), seed + 1, torch.bfloat16)
    key = _randn_cpu((pages, page_rows, kv_heads, head_dim), seed + 2, torch.bfloat16)
    value = _randn_cpu((pages, page_rows, kv_heads, head_dim), seed + 3, torch.bfloat16)
    block_table = torch.arange(pages, dtype=torch.int32).view(1, pages)
    request_ids = torch.zeros(rows, dtype=torch.int32)
    query_positions = torch.full((rows,), _HISTORY - 1, dtype=torch.int64)
    selected = torch.empty((rows, groups, width), dtype=torch.int32)
    for row in range(rows):
        for group in range(groups):
            _fill_positions(selected[row, group], geometry["layout"], live)
    output = torch.zeros_like(query)
    splits = _splits(family, rows)
    partial_output = torch.zeros((rows, splits, q_heads, head_dim), dtype=torch.float32)
    partial_lse = torch.zeros((rows, splits, q_heads), dtype=torch.float32)
    return {
        "query": query.to(device),
        "key": key.to(device),
        "value": value.to(device),
        "block_table": block_table.to(device),
        "request_ids": request_ids.to(device),
        "query_positions": query_positions.to(device),
        "selected_positions": selected.to(device),
        "output": output.to(device),
        "partial_output": partial_output.to(device),
        "partial_lse": partial_lse.to(device),
    }


def _launch(impl, family: str, route: str, case: dict, rows: int) -> None:
    geometry = _GEOMETRIES[family]
    common = dict(
        query=case["query"],
        key_cache=case["key"],
        value_cache=case["value"],
        k_descale=None,
        v_descale=None,
        block_table=case["block_table"],
        request_ids=case["request_ids"],
        selected_positions=case["selected_positions"],
        query_positions=case["query_positions"],
        softmax_scale=float(geometry["head_dim"]) ** -0.5,
    )
    if route == "direct":
        impl.launch_selected_paged_gqa_direct(
            output=case["output"], kv_warps=2, **common
        )
        return
    splits = int(case["partial_output"].shape[1])
    impl.launch_sparse_gqa_split(
        partial_output=case["partial_output"],
        partial_lse=case["partial_lse"],
        splits=splits,
        **common,
    )
    impl.launch_sparse_gqa_merge(
        partial_output=case["partial_output"],
        partial_lse=case["partial_lse"],
        output=case["output"],
        rows=rows,
        splits=splits,
    )


def _capture(impl, family: str, route: str, rows: int, case: dict) -> dict[str, torch.Tensor]:
    """Run one cell eagerly, then through a graph replay; require both bitwise."""
    def launch() -> None:
        _launch(impl, family, route, case, rows)

    launch()
    torch.cuda.synchronize()
    first = {
        name: case[name].clone()
        for name in ("output", "partial_output", "partial_lse")
        if route == "split" or name == "output"
    }
    launch()
    torch.cuda.synchronize()
    for name, expected in first.items():
        assert torch.equal(case[name].view(torch.int16 if case[name].dtype == torch.bfloat16 else torch.int32),
                           expected.view(torch.int16 if expected.dtype == torch.bfloat16 else torch.int32)), (
            f"{family}-{route}-r{rows}: {name} changed between identical launches"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    case["output"].fill_(float("nan"))
    if route == "split":
        case["partial_output"].fill_(float("nan"))
        case["partial_lse"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed = {
        name: case[name].clone()
        for name in ("output", "partial_output", "partial_lse")
        if route == "split" or name == "output"
    }
    for name, expected in first.items():
        assert torch.equal(replayed[name].view(torch.int16 if replayed[name].dtype == torch.bfloat16 else torch.int32),
                           expected.view(torch.int16 if expected.dtype == torch.bfloat16 else torch.int32)), (
            f"{family}-{route}-r{rows}: {name} changed under graph replay"
        )
    return first


def _reference(family: str, case: dict, rows: int) -> torch.Tensor:
    """FP64 selected softmax for row 0 at one KV-head layout; all rows share it."""
    geometry = _GEOMETRIES[family]
    head_dim = geometry["head_dim"]
    kv_heads = geometry["kv_heads"]
    q_heads = geometry["q_heads"]
    groups = geometry["selection_groups"]
    page_rows = geometry["page_rows"]
    pairs = q_heads // kv_heads
    reference = torch.empty((1, q_heads, head_dim), dtype=torch.float64, device=device)
    live = int((case["selected_positions"][0, 0] >= 0).sum().item())
    for kv_head in range(kv_heads):
        group = kv_head // (kv_heads // groups)
        logical = case["selected_positions"][0, group].long()
        logical = logical[(logical >= 0) & (logical <= int(case["query_positions"][0]))]
        assert logical.numel() == live
        page = case["block_table"][0][logical // page_rows].long()
        offset = logical % page_rows
        keys = case["key"][page, offset, kv_head].double()
        values = case["value"][page, offset, kv_head].double()
        first = kv_head * pairs
        scores = case["query"][0, first:first + pairs].double() @ keys.T
        scores = scores * (head_dim ** -0.5)
        reference[0, first:first + pairs] = scores.softmax(-1) @ values
    return reference


def _load_golden() -> dict:
    path = Path(_GOLDEN)
    payload = torch.load(path, map_location="cuda", weights_only=False)
    assert payload["spec_sha256"] == _SPEC_SHA256, (
        f"golden {path} was captured for a different corpus "
        f"({payload['spec_sha256'][:12]} != {_SPEC_SHA256[:12]}); re-capture it"
    )
    return payload


def _compare(label: str, produced: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert set(produced) == set(expected), f"{label}: produced {sorted(produced)} expected {sorted(expected)}"
    for name, actual in produced.items():
        want = expected[name].to(actual.device)
        bits = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
        if not torch.equal(actual.view(bits), want.view(bits)):
            bad = (actual.view(bits) != want.view(bits)).sum().item()
            delta = (actual.float() - want.float()).abs()
            pytest.fail(
                f"{label}: {name} is not bitwise equal to the baseline "
                f"({bad} of {actual.numel()} values differ, max |delta| "
                f"{delta.nan_to_num(nan=float('inf')).max().item():.6g})"
            )


def _impl():
    from b12x.attention.paged import _selected_forward
    return _selected_forward


def test_capture_golden():
    """Baseline-tree arm: run every cell and record its outputs as the golden."""
    if _MODE != "write":
        pytest.skip("write mode only (PAGED_SELECTED_MODE=write)")
    device = require_b12x()
    impl = _impl()
    impl.clear_caches()
    cells = {}
    for family, route, rows_set in _CASES:
        for rows in rows_set:
            label = f"{family}-{route}-r{rows}"
            case = _build_case(family, rows, device)
            cells[label] = _capture(impl, family, route, rows, case)
            print(f"captured {label}", flush=True)
    impl.clear_caches()
    path = Path(_GOLDEN)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "spec_sha256": _SPEC_SHA256,
            "spec": _spec(),
            "cells": cells,
            "device": torch.cuda.get_device_name(device),
        },
        path,
    )
    print(f"golden written to {path}", flush=True)


@pytest.mark.parametrize("label", _CASE_LABELS)
def test_cell_matches_baseline(label: str):
    """Candidate-tree arm: require the baseline's exact bits for this cell."""
    if _MODE != "check":
        pytest.skip("check mode only (PAGED_SELECTED_MODE=check)")
    family, route, rows_text = label.rsplit("-", 2)
    rows = int(rows_text[1:])
    device = require_b12x()
    payload = _load_golden()
    impl = _impl()
    impl.clear_caches()
    case = _build_case(family, rows, device)
    produced = _capture(impl, family, route, rows, case)
    _compare(label, produced, payload["cells"][label])
    if rows == 1:
        reference = _reference(family, case, rows)
        assert torch.isfinite(case["output"]).all()
        delta = (case["output"][:1].double() - reference).abs().max().item()
        torch.testing.assert_close(
            case["output"][:1].double(), reference, atol=0.02, rtol=0.02,
            msg=f"{label}: kernel disagrees with the FP64 selected-softmax reference "
                f"(max |delta| {delta:.6f})",
        )
    impl.clear_caches()
