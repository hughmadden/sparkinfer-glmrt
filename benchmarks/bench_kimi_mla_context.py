#!/usr/bin/env python3
"""Single-GPU compute screen for Kimi context partials versus layers attention.

Context estimate is one owner's two h48 partial passes plus one h48 combine,
assuming balanced concurrent owners. It EXCLUDES peer exchange/projection and
is not a two-GPU engine benchmark or an auto-placement decision. Histories
are OWNED rows: defaults 512, 4K, 32K and 128K correspond to 1K, 8K,
64K and 256K logical rows. The baseline is the runtime work/SM split rule.
Run kernel oracles first. All timings replay graphs under frozen resolution.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-contexts", type=int, nargs="+", default=[512, 4096, 32768, 131072])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--splits", type=int, nargs="*", default=[],
                        help="Optional diagnostic active split counts; the tuned rule always runs first")
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
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_dense_mla_aot, compile_kimi_mla_partial_aot, kimi_context_split_plan
    from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    torch.manual_seed(42)
    g = KimiMlaGeometry(kv_cache="fp8")
    capacity = max(args.owned_contexts)
    max_rows = max(args.rows)
    pages = 2 * capacity // 64
    # Fixed scales and exactly representable FP8 values; no invalid random bytes.
    pool = torch.zeros((pages, 64, 656), device="cuda", dtype=torch.uint8)
    pool[..., :512] = torch.randint(0, 16, (pages, 64, 512), device="cuda", dtype=torch.uint8)
    pool[..., 512:528] = torch.ones((pages, 64, 4), device="cuda").contiguous().view(torch.uint8)
    full = compile_kimi_dense_mla_aot(g, max_rows=max_rows, max_context=2*capacity)
    pair = [compile_kimi_mla_partial_aot(g, max_rows=max_rows, max_context=capacity,
            head_begin=begin) for begin in (0, 48)]
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    combine = compile_lse_combine2_aot(heads=48)
    q = (torch.randn((max_rows, 96, 576), device="cuda") * .25).bfloat16()
    out = torch.empty((max_rows, 96, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((max_rows, 96), device="cuda")
    scratch = torch.empty(full.scratch_bytes(max_rows)["scratch"], device="cuda", dtype=torch.uint8)
    po = [torch.empty((max_rows, 48, 512), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    pl = [torch.empty((max_rows, 48), device="cuda") for _ in range(2)]
    ps = torch.empty(pair[0].scratch_bytes(max_rows)["scratch"],
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
    props = torch.cuda.get_device_properties(0)
    result = dict(scope=__doc__, source_revision=revision, device=str(props),
        sm_count=props.multi_processor_count, torch=torch.__version__, results=results)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
