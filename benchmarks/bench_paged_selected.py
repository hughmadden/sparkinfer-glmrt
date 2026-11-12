#!/usr/bin/env python3
"""Selected-position paged GQA microbench at the real Step-5 and Qwen shapes.

Measures the one kernel the paged selected-attention duel changes
(``b12x/attention/paged/_selected_forward.py``) through the production routes:

* Step-5 (region attention, ``export_b12x_step5_aot.py``) - direct output, one
  CTA per (row, KV head). CSA: 64 query / 4 KV heads of 192, 64-row pages,
  4104-wide per-KV-head selection with 4 provider groups. SWA: same heads with
  one shared 512-wide sliding window.
* Qwen 3.8 Flash Next (``Qwen4Geometry``) - 24 query / 2 KV heads of 256,
  64-row pages, 2112-wide indices. Decode uses the split route plus the merge
  kernel; prefill uses direct output.

Context is 8K, so the live selection is 1024 (CSA), 512 (SWA) and 2048 (Qwen).
Rows are the decode widths C1/C4/C16 and one prefill unit of ``--prefill-rows``
(the production 4K prefill launch; 8K prefill is two such units). Layouts are
the production producers' (``_region_selection.py``).

Every cell is captured once and replayed, so the number is kernel time without
Python or launch overhead. The KV working set is L2-resident at 8K, matching
the serving steady state; the gathered-bytes column is the traffic a variant
would save by not re-reading it, not a DRAM bandwidth measurement.

Run the same file on the baseline and the candidate tree (separate processes;
the CuTe object cache keys on the b12x package fingerprint, so two checkouts
never collide), plus the bitwise oracle in
``tests/attention/test_paged_selected_oracle.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.common import bench_cuda_graph, capture_cuda_graph  # noqa: E402

GEOMETRIES = {
    "step5-csa": dict(
        query_heads=64, kv_heads=4, head_dim=192, page_rows=64,
        selection_width=4104, selection_groups=4, live=1024, layout="regions",
    ),
    "step5-swa": dict(
        query_heads=64, kv_heads=4, head_dim=192, page_rows=64,
        selection_width=512, selection_groups=1, live=512, layout="tail",
    ),
    "qwen": dict(
        query_heads=24, kv_heads=2, head_dim=256, page_rows=64,
        selection_width=2112, selection_groups=1, live=2048, layout="prefix",
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--context", type=int, default=8192, help="KV history (tokens)")
    parser.add_argument("--prefill-rows", type=int, default=4096,
                        help="Rows in one prefill launch (production default 4096)")
    parser.add_argument("--iters", type=int, default=30, help="Graph replays per cell")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed launches before capture")
    parser.add_argument("--only", default="", help="Substring filter on cell labels")
    parser.add_argument("--source-revision", help="Pinned fork SHA when Git metadata is not mounted")
    args = parser.parse_args(argv)
    if args.context < 512 or args.prefill_rows < 64 or args.iters <= 0 or args.warmup < 0:
        parser.error("context >= 512, prefill rows >= 64, iters > 0 and warmup >= 0 required")
    return args


def fill_positions(positions, layout, live, device):
    """Write one selection row in the requested production layout."""
    width = positions.shape[0]
    positions.fill_(-1)
    if layout == "prefix":
        positions[:live] = torch.arange(live, device=device, dtype=torch.int32)
    elif layout == "tail":
        positions[width - live:] = torch.arange(live, device=device, dtype=torch.int32)
    elif layout == "regions":
        # ``_expand_regions``: dense prefix of the top visible regions, then the
        # current partial region alone in the 4096..4103 tail.
        tail_region = (live - 1) // 8
        dense = 8 * tail_region
        positions[:dense] = torch.arange(dense, device=device, dtype=torch.int32)
        positions[dense:4096] = -1
        positions[4096:4104] = torch.arange(dense, dense + 8, device=device, dtype=torch.int32)
    else:
        raise ValueError(layout)


def splits_for(family, rows):
    """Split count the production route plans for this row count."""
    if family != "qwen" or rows > 64:
        return 16
    if rows == 1:
        return 64
    if rows <= 4:
        return 32
    return 16


def build(family, rows, context, device):
    """Allocate the cell's tensors and return (launch closure, metadata)."""
    geometry = GEOMETRIES[family]
    head_dim = geometry["head_dim"]
    kv_heads = geometry["kv_heads"]
    q_heads = geometry["query_heads"]
    width = geometry["selection_width"]
    groups = geometry["selection_groups"]
    page_rows = geometry["page_rows"]
    live = min(geometry["live"], context)
    pages = (live + page_rows - 1) // page_rows

    query = torch.randn((rows, q_heads, head_dim), device=device, dtype=torch.bfloat16)
    key = torch.randn((pages, page_rows, kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    block_table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    request_ids = torch.zeros(rows, device=device, dtype=torch.int32)
    query_positions = torch.full((rows,), context - 1, device=device, dtype=torch.int64)
    selected = torch.empty((rows, groups, width), device=device, dtype=torch.int32)
    for row in range(rows):
        for group in range(groups):
            fill_positions(selected[row, group], geometry["layout"], live, device)
    output = torch.empty_like(query)
    softmax_scale = head_dim ** -0.5

    route = "split" if family == "qwen" and rows <= 64 else "direct"
    splits = splits_for(family, rows) if route == "split" else 1
    partial_output = (
        torch.empty((rows, splits, q_heads, head_dim), device=device, dtype=torch.float32)
        if route == "split" else None
    )
    partial_lse = (
        torch.empty((rows, splits, q_heads), device=device, dtype=torch.float32)
        if route == "split" else None
    )

    from b12x.attention.paged import _selected_forward as impl

    common = dict(
        query=query, key_cache=key, value_cache=value, k_descale=None, v_descale=None,
        block_table=block_table, request_ids=request_ids, selected_positions=selected,
        query_positions=query_positions, softmax_scale=softmax_scale,
    )

    if route == "split":
        def launch():
            impl.launch_sparse_gqa_split(
                partial_output=partial_output, partial_lse=partial_lse,
                splits=splits, **common,
            )
            impl.launch_sparse_gqa_merge(
                partial_output=partial_output, partial_lse=partial_lse,
                output=output, rows=rows, splits=splits,
            )
    else:
        def launch():
            impl.launch_selected_paged_gqa_direct(output=output, kv_warps=2, **common)

    gathered_bytes = rows * kv_heads * live * head_dim * 2 * 2
    meta = dict(
        family=family, route=route, rows=rows, splits=splits, live=live,
        pages=pages, gathered_bytes=gathered_bytes,
    )
    case = dict(
        query=query, key=key, value=value, block_table=block_table,
        positions=selected, output=output,
    )
    return launch, case, meta


def reference(family, context, device):
    """FP64 selected softmax for row 0; independent check of the timed route."""
    launch, case, meta = build(family, 1, context, device)
    geometry = GEOMETRIES[family]
    head_dim = geometry["head_dim"]
    kv_heads = geometry["kv_heads"]
    q_heads = geometry["query_heads"]
    groups = geometry["selection_groups"]
    page_rows = geometry["page_rows"]
    pairs = q_heads // kv_heads
    output = case["output"]
    expected = torch.empty((1, q_heads, head_dim), device=device, dtype=torch.float64)
    for kv_head in range(kv_heads):
        group = kv_head // (kv_heads // groups)
        logical = case["positions"][0, group].long()
        logical = logical[logical >= 0]
        page = case["block_table"][0][logical // page_rows].long()
        offset = logical % page_rows
        keys = case["key"][page, offset, kv_head].double()
        values = case["value"][page, offset, kv_head].double()
        first = kv_head * pairs
        scores = (case["query"][0, first:first + pairs].double() @ keys.T) * (head_dim ** -0.5)
        expected[0, first:first + pairs] = scores.softmax(-1) @ values
    launch()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise AssertionError(f"{family}: non-finite output")
    delta = (output[0].double() - expected).abs().max().item()
    if not math.isfinite(delta) or delta > 0.05:
        raise AssertionError(f"{family}: kernel disagrees with FP64 reference (max |delta| {delta})")
    return dict(reference_max_abs_delta=delta)


def measure(label, launch, meta, args):
    graph = capture_cuda_graph(launch, warmup=args.warmup)
    timings = bench_cuda_graph(graph, replays=args.iters)
    replay = timings["replay_us"]
    gathered_gb_s = meta["gathered_bytes"] / statistics.median(replay) / 1e3
    row = dict(
        case=label, **{k: v for k, v in meta.items() if k != "gathered_bytes"},
        median_us=statistics.median(replay), min_us=min(replay),
        gathered_gb_s=gathered_gb_s,
    )
    print(json.dumps(row), flush=True)
    return row


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda")
    torch.manual_seed(274)

    revision = args.source_revision or subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    properties = torch.cuda.get_device_properties(0)

    checks = [reference(family, args.context, device) for family in GEOMETRIES]

    cells = []
    for family in GEOMETRIES:
        for rows in (1, 4, 16, args.prefill_rows):
            launch, _, meta = build(family, rows, args.context, device)
            label = f"{family}-r{rows}"
            if args.only and args.only not in label:
                continue
            cells.append(measure(label, launch, meta, args))

    payload = dict(
        profile="paged-selected-8k",
        source_revision=revision,
        device=properties.name,
        sm_count=properties.multi_processor_count,
        compute_capability=[properties.major, properties.minor],
        torch=torch.__version__,
        context=args.context,
        prefill_rows=args.prefill_rows,
        iters=args.iters,
        layout_note="L2-resident 8K selections; gathered_gb_s is saved-traffic potential",
        reference_checks=checks,
        cells=cells,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(output=args.output, cells=len(cells),
                          device=properties.name, revision=revision)), flush=True)


if __name__ == "__main__":
    main()
