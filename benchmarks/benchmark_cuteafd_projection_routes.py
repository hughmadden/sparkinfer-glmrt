#!/usr/bin/env python3
"""Isolated, L2-cold cuteafd projection recipes; no model serving or timed assertions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cutlass
import torch

from benchmarks.common import make_l2_flush_fn
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program


def _program(n, k, recipe):
    from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, TmaBf16Projection, skinny_config

    op = SkinnyBf16Gemv(n, k, **skinny_config(n, k, cutlass.BFloat16)) if recipe == "skinny" else \
        TmaBf16Projection(n, k)
    operands = (Operand("x", torch.bfloat16, "x"), Operand("w", torch.bfloat16, "w"),
                Operand("out", torch.bfloat16, "out", "out"))
    return compile_program(op, name="projection_route_bench", operands=operands, scalars=(Scalar("rows"),),
                           key=(n, k, recipe))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="4096x4096")
    parser.add_argument("--rows", default="1,8,9,10,16,32,64")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shapes = [tuple(map(int, shape.split("x"))) for shape in args.shapes.split(",")]
    rows_list = [int(v) for v in args.rows.split(",")]
    if args.samples < 1 or any(v < 1 or v > 128 for v in rows_list):
        parser.error("positive samples and live rows in 1..128 required")
    recipes = ("skinny", "mma")
    torch.manual_seed(174)
    flush = make_l2_flush_fn(True)
    results = []
    for n, k in shapes:
        x = torch.randn(max(rows_list), k, device="cuda").bfloat16()
        w = (torch.randn(n, k, device="cuda") * 0.03).bfloat16()
        out = torch.empty(max(rows_list), n, device="cuda", dtype=torch.bfloat16)
        programs = {recipe: _program(n, k, recipe) for recipe in recipes}
        for rows in rows_list:
            graphs = {}
            for recipe, program in programs.items():
                operands = (x, w, out)
                program.launch(*operands, scalars=(rows,))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    program.launch(*operands, scalars=(rows,))
                graphs[recipe] = graph
                flush()
                graph.replay()
                torch.cuda.synchronize()
            times = {recipe: [] for recipe in recipes}
            for sample in range(args.samples):
                for recipe in recipes if sample % 2 == 0 else recipes[::-1]:
                    flush()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[recipe].replay()
                    end.record()
                    end.synchronize()
                    times[recipe].append(start.elapsed_time(end) * 1000)
            entry = {"n": n, "k": k, "rows": rows,
                     "median_us": {recipe: statistics.median(v) for recipe, v in times.items()},
                     "samples_us": times}
            results.append(entry)
            print(json.dumps({key: value for key, value in entry.items() if key != "samples_us"}), flush=True)
        del programs, graphs, x, w, out
    result = {"kind": "bf16", "device": torch.cuda.get_device_name(),
              "sms": torch.cuda.get_device_properties(0).multi_processor_count,
              "conditions": "one launch per recipe/M warmup; interleaved CUDA graph replays; L2 eviction before each sample",
              "results": results}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
