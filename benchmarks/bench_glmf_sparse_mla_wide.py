#!/usr/bin/env python3
"""Time GLM 5.3 Flash's 128-row sparse MLA decode program under its split plans.

The variants of ``glmf_sparse_mla_decode_m128`` (FP32 partials, as exported) differ only in
the 128-row bucket's split count: the split planner's own (``planner``: 33 on a 170-SM GPU,
where no count keeps 128 rows x 4 head blocks within its three waves) and
``full_launch_splits`` 1, 2 and 3. Each runs at every ``--rows`` count above 64 (127 rows x 4
blocks = 508 CTAs is the largest launch within three waves of 170 SMs), the 64-row program
at the counts up to 64. Synthetic FP8 records over ``--context`` slots, 2112 distinct
selected slots per row, all valid. Reported per variant and row count: the plan, CTAs and
waves, the median of CUDA-event timings with the L2 flushed before each launch (``cold``)
and the mean over CUDA-graph replays (``warm``); each variant's scratch. Every variant's
output is checked against the one-split variant's first. Component timings, not tok/s.

  python benchmarks/bench_glmf_sparse_mla_wide.py [--iters 50] [--rows 64,96,120,126,127,128]
      [--context 65536] [--output out.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

SLOTS = 2112
VARIANTS = {"planner": None, "splits1": 1, "splits2": 2, "splits3": 3}


def cold_us(fn, iters: int, flush: torch.Tensor) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.max()  # read-only L2 flush
        torch.cuda._sleep(3_000_000)  # covers the host-side launch of fn
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def warm_us(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rows", default="64,96,120,126,127,128")
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("B12X_MLA_SM120_NUM_SPLITS"):
        raise SystemExit("unset B12X_MLA_SM120_NUM_SPLITS: it pins every bucket's split count")

    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
    from b12x.integration.cuteafd._common import GLM53_FLASH as g
    from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot, decode_buckets

    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    h_blocks = g.heads // 16
    rows_list = [int(r) for r in args.rows.split(",")]
    gen = torch.Generator(device=dev).manual_seed(0)
    flush = torch.zeros(512 << 20, dtype=torch.uint8, device=dev)
    context = args.context // 64 * 64
    latent = torch.randn((context, 512), generator=gen, device=dev) * 0.5
    cache = pack_mla_kv_cache_reference(latent).view(context // 64, 64 * 528).contiguous()
    del latent
    widest = max(rows_list)
    q_all = (torch.randn((widest, g.heads, 512), generator=gen, device=dev) * 0.5).bfloat16()
    idx_all = torch.stack([torch.randperm(context, generator=gen, device=dev)[:SLOTS] for _ in range(widest)]).int()
    len_all = torch.full((widest,), SLOTS, dtype=torch.int32, device=dev)

    def compile_(max_rows, splits):
        return compile_glm_sparse_mla_aot(g, route="decode", max_rows=max_rows, name="glmf_sparse_mla",
                                          fp32_partials=True, full_launch_splits=splits)

    programs = {"m64": (compile_(64, None), decode_buckets(g, 64, sm_count=sms))}
    for name, splits in VARIANTS.items():
        programs[name] = (compile_(128, splits), decode_buckets(g, 128, sm_count=sms, full_launch_splits=splits))
    report = {"device": torch.cuda.get_device_name(0), "sms": sms, "context": context, "iters": args.iters,
              "variants": {name: {"buckets": [list(b) for b in buckets],
                                  "scratch_bytes": program.scratch_bytes(64 if name == "m64" else 128)["scratch"]}
                           for name, (program, buckets) in programs.items()},
              "timings": []}
    for rows in rows_list:
        q, indices, lengths = q_all[:rows].contiguous(), idx_all[:rows].contiguous(), len_all[:rows].contiguous()
        names = ["m64"] if rows <= 64 else list(VARIANTS)
        outputs = {}
        for name in names:
            program, buckets = programs[name]
            splits = next(s for cap, s, _ in buckets if rows <= cap)
            out = torch.empty((rows, g.heads, 512), dtype=torch.bfloat16, device=dev)
            scratch = torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device=dev)

            def launch(program=program, out=out, scratch=scratch):
                program.launch(q, cache, indices, lengths, out, scratch, scalars=(rows,))

            launch()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0, (name, rows)
            outputs[name] = out.clone()
            ctas = rows * h_blocks * splits
            entry = {"variant": name, "rows": rows, "splits": splits, "ctas": ctas, "waves": round(ctas / sms, 3),
                     "cold_us": round(cold_us(launch, args.iters, flush), 2),
                     "warm_us": round(warm_us(launch, args.iters), 2)}
            entry["warm_us_per_row"] = round(entry["warm_us"] / rows, 3)
            report["timings"].append(entry)
            print(json.dumps(entry), flush=True)
        if "splits1" in outputs:
            for name, out in outputs.items():
                c = cosine(out, outputs["splits1"])
                assert c >= 0.9999, (name, rows, c)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{props.name}, {sms} SMs, context {context}, {args.iters} iterations")
    for name, info in report["variants"].items():
        print(f"  {name:8s} buckets {info['buckets']} scratch {info['scratch_bytes']:,} B")
    print(f"  {'variant':8s} {'rows':>4s} {'splits':>6s} {'CTAs':>6s} {'waves':>6s} {'cold us':>9s} {'warm us':>9s}"
          f" {'us/row':>7s}")
    for e in report["timings"]:
        print(f"  {e['variant']:8s} {e['rows']:4d} {e['splits']:6d} {e['ctas']:6d} {e['waves']:6.2f} "
              f"{e['cold_us']:9.1f} {e['warm_us']:9.1f} {e['warm_us_per_row']:7.3f}")


if __name__ == "__main__":
    main()
