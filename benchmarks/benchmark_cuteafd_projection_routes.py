#!/usr/bin/env python3
"""Isolated, L2-cold cuteafd projection recipes; no model serving or timed assertions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="4096x4096", help="comma-separated output N x input K")
    parser.add_argument("--rows", default="1,8,9,10,16,32,64")
    parser.add_argument("--recipes", choices=("routes", "compensation"), default="routes",
                        help="skinny vs MMA, or identical-tile uncompensated vs compensated TMA")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        args.shapes = [tuple(map(int, shape.split("x"))) for shape in args.shapes.split(",")]
        args.rows = [int(v) for v in args.rows.split(",")]
    except ValueError:
        parser.error("shapes must be NxK and rows must be comma-separated integers")
    if any(len(shape) != 2 or any(v <= 0 or v % 8 for v in shape) for shape in args.shapes):
        parser.error("projection dimensions must be positive multiples of eight")
    row_limit = 4096 if args.recipes == "compensation" else 128
    if args.samples < 1 or any(v < 1 or v > row_limit for v in args.rows):
        parser.error(f"positive samples and live rows in 1..{row_limit} required")
    return args


def _program(n, k, recipe):
    import cutlass
    import torch
    from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, TmaBf16Projection, skinny_config
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    if recipe == "skinny":
        op = SkinnyBf16Gemv(n, k, **skinny_config(n, k, cutlass.BFloat16))
    elif recipe == "mma":
        op = TmaBf16Projection(n, k)
    else:
        # Explicitly disable shape-specific row tiling for both compensation arms.
        op = TmaBf16Projection(n, k, tiles_by_rows=(), compensated=recipe == "compensated")
    operands = (Operand("x", torch.bfloat16, "x"), Operand("w", torch.bfloat16, "w"),
                Operand("out", torch.bfloat16, "out", "out"))
    return compile_program(op, name="projection_route_bench", operands=operands, scalars=(Scalar("rows"),),
                           key=(n, k, recipe))


def _accuracy(actual, baseline, oracle):
    import torch

    actual, baseline = actual.double(), baseline.double()
    candidate = (actual - oracle).square().sum(-1)
    reference = (baseline - oracle).square().sum(-1)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(baseline).all()
                  and torch.isfinite(oracle).all())
    passed = finite and bool((candidate <= reference * 1.01 ** 2).all())
    passed = passed and bool(candidate.sum() <= reference.sum() * 1.01 ** 2)
    return {"pass": passed, "finite": finite,
            "candidate_squared_error": candidate.sum().item(),
            "reference_squared_error": reference.sum().item(),
            "rows_exceeding_bar": int((candidate > reference * 1.01 ** 2).sum().item())}


def main():
    args = parse_args()
    import torch
    from benchmarks.common import make_l2_flush_fn

    recipes = ("skinny", "mma") if args.recipes == "routes" else ("uncompensated", "compensated")
    torch.manual_seed(174)
    flush = make_l2_flush_fn(True)
    results = []
    for n, k in args.shapes:
        x = torch.randn(max(args.rows), k, device="cuda").bfloat16()
        w = (torch.randn(n, k, device="cuda") * 0.03).bfloat16()
        out = torch.empty(max(args.rows), n, device="cuda", dtype=torch.bfloat16)
        programs = {recipe: _program(n, k, recipe) for recipe in recipes}
        for rows in args.rows:
            graphs, accuracy = {}, {}
            oracle = x[:rows].double() @ w.double().T
            baseline = x[:rows] @ w.T
            for recipe, program in programs.items():
                operands = (x, w, out)
                out.fill_(float("nan"))
                program.launch(*operands, scalars=(rows,))
                torch.cuda.synchronize()
                accuracy[recipe] = _accuracy(out[:rows], baseline, oracle)
                if not accuracy[recipe]["pass"]:
                    raise RuntimeError(f"{n}x{k} rows={rows} {recipe}: FP64 gate failed: {accuracy[recipe]}")
                expected = out[:rows].clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    program.launch(*operands, scalars=(rows,))
                graphs[recipe] = graph
                flush()
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(expected, out[:rows]):
                    raise RuntimeError(f"{recipe}: graph replay differs from eager output")
            del oracle, baseline, expected
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
            medians = {recipe: statistics.median(v) for recipe, v in times.items()}
            entry = {"n": n, "k": k, "rows": rows, "accuracy": accuracy,
                     "median_us": medians, "samples_us": times,
                     "second_over_first_time": medians[recipes[1]] / medians[recipes[0]]}
            results.append(entry)
            print(json.dumps({key: value for key, value in entry.items() if key != "samples_us"}), flush=True)
        del programs, graphs, x, w, out
    properties = torch.cuda.get_device_properties(0)
    result = {"kind": "bf16", "device": torch.cuda.get_device_name(),
              "device_uuid": str(getattr(properties, "uuid", "unavailable")),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "argv": sys.argv, "sms": properties.multi_processor_count,
              "recipes": recipes, "ratio_direction": "second recipe time / first recipe time; >1 is slower",
              "conditions": "one eager and one graph warmup per recipe/M; interleaved CUDA graph replays; L2 eviction before each sample; every-row/global FP64 error <=1.01x torch BF16; byte-exact graph replay",
              "results": results}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
