#!/usr/bin/env python3
"""Single-GPU compute screen for Kimi context partials versus layers attention.

Context estimate is one owner's two h48 partial passes plus one h48 combine,
assuming balanced concurrent owners. It EXCLUDES peer exchange/projection and
is not a two-GPU engine benchmark or an auto-placement decision. Histories
are OWNED rows: defaults 512, 4K, 32K and 128K correspond to 1K, 8K,
64K and 256K logical rows. The baseline is the runtime work/SM split rule.
Run kernel oracles first. All timings replay graphs under frozen resolution.

--draft adds the h64 MLA draft Context partial screen (full and window 4096,
BF16 and compact FP8 records) with the same two-owner estimate shape, the
h64 combine measured on its own, and each row's HBM roofline at
--bandwidth-gbs. A draft partial launch includes its internal split merge.
--draft-only skips the h96 target screen.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch


DRAFT_RECORD_BYTES = {"fp8": 656, "bf16": 1152}
DRAFT_HEADS = 64
DRAFT_HEAD_TILE = 16
DRAFT_SPLIT_CAPACITY_LIMIT = 32


def timed(fn, iterations):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return dict(median_us=statistics.median(samples), samples_us=samples)


def draft_records(cache: str, pages: int):
    """Finite, exactly representable records; layout matches the draft reader."""
    if cache == "fp8":
        raw = torch.zeros((pages, 64, 656), device="cuda", dtype=torch.uint8)
        raw[..., :512] = torch.randint(0, 16, (pages, 64, 512), device="cuda", dtype=torch.uint8)
        raw[..., 512:528] = torch.ones((pages, 64, 4), device="cuda").contiguous().view(torch.uint8)
        return raw
    return (torch.randn((pages, 64, 1152), device="cuda") * .25).bfloat16()


def target_screen(args, sm_count):
    from b12x.integration.cuteafd.kimi_mla import (
        KimiMlaGeometry,
        compile_kimi_dense_mla_aot,
        compile_kimi_mla_partial_aot,
        kimi_context_heads_per_tile,
        kimi_context_split_plan,
    )
    from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    g = KimiMlaGeometry(kv_cache="fp8")
    capacity = max(args.owned_contexts)
    max_rows = max(args.rows)
    pages = 2 * capacity // 64
    # Fixed scales and exactly representable FP8 values; no invalid random bytes.
    pool = torch.zeros((pages, 64, 656), device="cuda", dtype=torch.uint8)
    pool[..., :512] = torch.randint(0, 16, (pages, 64, 512), device="cuda", dtype=torch.uint8)
    pool[..., 512:528] = torch.ones((pages, 64, 4), device="cuda").contiguous().view(torch.uint8)
    full = compile_kimi_dense_mla_aot(g, max_rows=max_rows, max_context=2*capacity)
    pairs = {
        tile: [compile_kimi_mla_partial_aot(g, max_rows=max_rows, max_context=capacity,
               head_begin=begin, heads_per_tile=tile) for begin in (0, 48)]
        for tile in (8, 16)
    }
    combine = compile_lse_combine2_aot(heads=48)
    q = (torch.randn((max_rows, 96, 576), device="cuda") * .25).bfloat16()
    out = torch.empty((max_rows, 96, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((max_rows, 96), device="cuda")
    scratch = torch.empty(full.scratch_bytes(max_rows)["scratch"], device="cuda", dtype=torch.uint8)
    po = [torch.empty((max_rows, 48, 512), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    pl = [torch.empty((max_rows, 48), device="cuda") for _ in range(2)]
    ps = torch.empty(pairs[16][0].scratch_bytes(max_rows)["scratch"],
                     device="cuda", dtype=torch.uint8)
    co = torch.empty_like(po[0])
    cu = torch.arange(max_rows+1, device="cuda", dtype=torch.int32)
    results = []
    for owned in args.owned_contexts:
        count = owned // 64
        ft = torch.arange(2*count, device="cuda", dtype=torch.int32)[None].repeat(max_rows, 1)
        pt = torch.arange(count, device="cuda", dtype=torch.int32)[None].repeat(max_rows, 1)
        fl = torch.full((max_rows,), 2*owned, device="cuda", dtype=torch.int32)
        ll = torch.full((max_rows,), owned, device="cuda", dtype=torch.int32)
        local = pool[:2*count:2].contiguous()
        for rows in args.rows:
            def layers():
                full.launch(q, pool, ft, fl, cu, out, lse, scratch,
                            scalars=(rows, rows, 2*count, full.geometry["num_splits"]))
            layers()
            torch.cuda.synchronize()
            with kernel_resolution_guard("Kimi context layers screen"):
                base = timed(layers, args.iters)
            pair = pairs[kimi_context_heads_per_tile(rows=rows)]
            planned = kimi_context_split_plan(owned_pages=count, rows=rows, sm_count=sm_count)
            for policy, splits in [("work_sm_rule", planned)] + [("fixed_diagnostic", s) for s in args.splits]:
                def context():
                    for h, p in enumerate(pair):
                        p.launch(q, local, pt, ll, cu, po[h], pl[h], ps, scalars=(rows, count, splits))
                    # Same shard twice is only a compute-cost stand-in for the
                    # received peer partial; no network or multi-GPU claim.
                    combine.launch(po[0], pl[0], po[0], pl[0], None, co, scalars=(rows,))
                context()
                torch.cuda.synchronize()
                assert torch.isfinite(co[:rows]).all() and co[:rows].abs().max() > 0
                with kernel_resolution_guard("Kimi context partial screen"):
                    candidate = timed(context, args.iters)
                row = dict(owned_context=owned, logical_context=owned*2, rows=rows, splits=splits, policy=policy,
                           layers=base, context_compute_estimate=candidate,
                           layers_over_context_compute=base["median_us"]/candidate["median_us"])
                results.append(row)
                print(json.dumps(row), flush=True)
    return results


def draft_screen(args, sm_count):
    """h64 draft Context partials: two owner passes plus the standalone h64 combine."""
    from b12x.integration.cuteafd.kimi_mla import (
        DRAFT_CONTEXT_STARVED_SPLIT_CAPACITY,
        MlaDraftGeometry,
        compile_mla_draft_partial_aot,
        draft_context_split_capacity,
        draft_context_split_plan,
    )
    from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    max_rows = max(args.draft_rows)
    max_owned = max(args.draft_owned_contexts)
    combine = compile_lse_combine2_aot(heads=DRAFT_HEADS)
    scale = float(MlaDraftGeometry().softmax_scale)
    q = (torch.randn((max_rows, DRAFT_HEADS, 576), device="cuda") * .25).bfloat16()
    po = [torch.empty((max_rows, DRAFT_HEADS, 512), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    pl = [torch.empty((max_rows, DRAFT_HEADS), device="cuda") for _ in range(2)]
    co = torch.empty_like(po[0])
    cu = torch.arange(max_rows + 1, device="cuda", dtype=torch.int32)
    results = []
    for cache in args.draft_caches:
        pages = 2 * max_owned // 64
        pool = draft_records(cache, pages)
        for window in args.draft_windows:
            window_size = None if window == 0 else window
            # One program per compiled split capacity; each bucket's rows never
            # exceed its own max_rows, so the doubled-capacity scratch stays small.
            capacities = sorted({draft_context_split_capacity(rows=r) for r in args.draft_rows})
            programs = {}
            for capacity in capacities:
                bucket_rows = max(r for r in args.draft_rows
                                  if draft_context_split_capacity(rows=r) == capacity)
                program = compile_mla_draft_partial_aot(
                    MlaDraftGeometry(kv_cache=cache, window_size=window_size),
                    max_rows=bucket_rows, max_context=max_owned, num_splits=capacity)
                programs[capacity] = (
                    program,
                    torch.empty(program.scratch_bytes(bucket_rows)["scratch"],
                                device="cuda", dtype=torch.uint8))
            for owned in args.draft_owned_contexts:
                count = owned // 64
                local = pool[:count].contiguous()
                table = torch.arange(count, device="cuda", dtype=torch.int32)[None].repeat(max_rows, 1)
                # Worst case for the shape: every owned record is visible in every row.
                intervals = torch.tensor([0, owned] * max_rows, device="cuda",
                                         dtype=torch.int32).view(max_rows, 2)
                for rows in args.draft_rows:
                    capacity = draft_context_split_capacity(rows=rows)
                    program, scratch = programs[capacity]
                    planned = draft_context_split_plan(owned_pages=count, rows=rows, sm_count=sm_count)
                    for policy, splits in ([("draft_work_sm_rule", planned)]
                                           + [("fixed_diagnostic", s) for s in args.draft_splits
                                              if s <= capacity]):
                        def partial(dst=0):
                            program.launch(q, local, table, intervals, cu, po[dst], pl[dst],
                                           scratch, scalars=(rows, count, splits, scale))
                        def merge():
                            combine.launch(po[0], pl[0], po[1], pl[1], None, co, scalars=(rows,))
                        def combined():
                            partial(0)
                            partial(1)
                            merge()
                        partial(0)
                        torch.cuda.synchronize()
                        assert torch.isfinite(pl[0][:rows]).all() and pl[0][:rows].abs().max() > 0
                        with kernel_resolution_guard("Kimi draft partial screen"):
                            single = timed(lambda: partial(0), args.iters)
                        with kernel_resolution_guard("Kimi draft partial combine screen"):
                            merged = timed(merge, args.iters)
                        with kernel_resolution_guard("Kimi draft context estimate screen"):
                            estimate = timed(combined, args.iters)
                        assert torch.isfinite(co[:rows]).all() and co[:rows].abs().max() > 0
                        value_bytes = owned * DRAFT_RECORD_BYTES[cache]
                        single_roofline = value_bytes / (args.bandwidth_gbs * 1e3)
                        ctas = rows * ((DRAFT_HEADS + DRAFT_HEAD_TILE - 1) // DRAFT_HEAD_TILE) * splits
                        row = dict(mode="draft", kv_cache=cache, window=window_size,
                                   owned_context=owned, logical_context=owned*2, rows=rows,
                                   splits=splits, policy=policy, head_tile=DRAFT_HEAD_TILE,
                                   split_capacity=capacity,
                                   ctas=ctas, record_bytes=DRAFT_RECORD_BYTES[cache],
                                   partial=dict(single, value_bytes=value_bytes,
                                                roofline_us=single_roofline,
                                                over_roofline=single["median_us"]/single_roofline),
                                   merge=merged, context_compute_estimate=estimate,
                                   # Two owner passes cover 2*64 head-halves; the h96 target's
                                   # two h48 passes cover 96, so compare per head-half.
                                   us_per_head_half=estimate["median_us"]/(2*DRAFT_HEADS),
                                   us_per_head_pass=single["median_us"]/DRAFT_HEADS,
                                   estimate_over_roofline=estimate["median_us"]/(2*single_roofline))
                        results.append(row)
                        print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-contexts", type=int, nargs="+", default=[512, 4096, 32768, 131072])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--splits", type=int, nargs="*", default=[],
                        help="Optional diagnostic active split counts; the tuned rule always runs first")
    parser.add_argument("--draft", action="store_true", help="Also screen the h64 MLA draft Context partials")
    parser.add_argument("--draft-only", action="store_true", help="Screen only the draft partials")
    parser.add_argument("--draft-owned-contexts", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--draft-rows", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--draft-caches", nargs="+", default=["fp8", "bf16"],
                        help="Draft record formats to screen (fp8, bf16)")
    parser.add_argument("--draft-windows", type=int, nargs="+", default=[0, 4096],
                        help="0 measures the full-context draft program")
    parser.add_argument("--draft-splits", type=int, nargs="*", default=[16],
                        help="Optional diagnostic draft split counts; the tuned rule always runs first")
    parser.add_argument("--bandwidth-gbs", type=float, default=1792.0,
                        help="HBM spec bandwidth used for the draft roofline")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", help="Pinned fork SHA when Git metadata is not container-mounted")
    args = parser.parse_args()
    revision = args.source_revision or subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip()
    if (min(args.owned_contexts) <= 0 or max(args.owned_contexts) > 524288
            or any(x % 64 for x in args.owned_contexts) or min(args.rows) <= 0
            or max(args.rows) > 64 or any(x < 2 or x > 16 for x in args.splits)
            or args.iters <= 0):
        parser.error("positive page-aligned owned contexts <=524288, rows <=64 and splits 2..16 required")
    if (min(args.draft_owned_contexts) <= 0 or max(args.draft_owned_contexts) > 524288
            or any(x % 64 for x in args.draft_owned_contexts) or min(args.draft_rows) <= 0
            or max(args.draft_rows) > 64 or any(x < 2 or x > DRAFT_SPLIT_CAPACITY_LIMIT for x in args.draft_splits)
            or any(x not in (0, 4096) for x in args.draft_windows)
            or any(x not in DRAFT_RECORD_BYTES for x in args.draft_caches)
            or args.bandwidth_gbs <= 0):
        parser.error("draft: page-aligned owned contexts <=524288, rows <=64, splits 2..32, "
                     "windows 0/4096, caches fp8/bf16 and a positive bandwidth required")
    torch.manual_seed(42)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    results = [] if args.draft_only else target_screen(args, sm_count)
    draft_results = draft_screen(args, sm_count) if (args.draft or args.draft_only) else []
    props = torch.cuda.get_device_properties(0)
    result = dict(scope=__doc__, source_revision=revision, device=str(props),
        sm_count=props.multi_processor_count, torch=torch.__version__, results=results,
        draft_results=draft_results)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
