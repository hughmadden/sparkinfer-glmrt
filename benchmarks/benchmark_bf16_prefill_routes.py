#!/usr/bin/env python3
"""BF16 prefill projection routes: interleaved, L2-cold, bitwise-gated.

``base`` is the serving route (``RoutedBf16Projection`` with
``max_skinny_rows=0``: the shape's ``prefill_config`` tiles, or its
``DECODE_TMA_TILES`` entry below its last row limit). ``wide`` is the same
projection with ``WIDE_PREFILL_CONFIG`` from ``WIDE_PREFILL_MIN_ROWS`` rows.
``tiles=WM,WN,TM,TN,STAGES[,MIN_ROWS]`` is one explicit tile route from
``MIN_ROWS`` rows (default ``WIDE_PREFILL_MIN_ROWS``).

Every arm retiles the one ordered-K16 uncompensated MMA accumulation, so every
arm must be bitwise identical to ``base`` at every (shape, rows) cell; a
mismatch ends the run. Rows past the live count stay poisoned, and a
row-invariance pass checks that a row's output is the same at a larger M and
from an offset input pointer.

  python benchmarks/benchmark_bf16_prefill_routes.py --shapes 6144x7168 \
    --rows 512,1024,2752,4096 --arm base --arm wide --output /tmp/routes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int32

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POISON = float("nan")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", required=True, help="comma-separated output N x input K")
    parser.add_argument("--rows", required=True, help="comma-separated live row counts")
    parser.add_argument("--arm", action="append", required=True,
                        help="base, wide, or tiles=WM,WN,TM,TN,STAGES[,MIN_ROWS]; repeatable")
    parser.add_argument("--samples", type=int, default=52, help="timed replays per cell per arm")
    parser.add_argument("--blocks", type=int, default=4, help="alternating-order blocks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.shapes = [tuple(map(int, shape.lower().split("x"))) for shape in args.shapes.split(",")]
        args.rows = [int(value) for value in args.rows.split(",")]
    except ValueError:
        parser.error("--shapes is NxK and --rows is a comma-separated integer list")
    if any(len(shape) != 2 or any(value <= 0 or value % 8 for value in shape) for shape in args.shapes):
        parser.error("projection dimensions must be positive multiples of eight")
    if any(value <= 0 or value >= 2**31 for value in args.rows):
        parser.error("row counts must be positive int32 values")
    if args.samples < 1 or args.blocks < 1:
        parser.error("--samples and --blocks must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def _arm_spec(text):
    if text in ("base", "wide"):
        return {"name": text, "kind": text}
    if text.startswith("tiles="):
        values = [int(value) for value in text.removeprefix("tiles=").split(",")]
        if len(values) not in (5, 6):
            raise SystemExit(f"{text}: tiles= takes WM,WN,TM,TN,STAGES[,MIN_ROWS]")
        spec = {"name": text, "kind": "tiles", "warp_layout": (values[0], values[1]),
                "tile_m": values[2], "tile_n": values[3], "num_stages": values[4],
                "min_rows": values[5] if len(values) == 6 else None}
        return spec
    raise SystemExit(f"unknown arm {text!r}: base, wide or tiles=WM,WN,TM,TN,STAGES[,MIN_ROWS]")


class _Routed:
    """``RoutedBf16Projection``'s dispatch with a bench-supplied prefill tile route."""

    def __init__(self, base, prefill, min_rows):
        self.base, self.prefill = base, prefill
        self.min_rows = int(min_rows)

    @cute.jit
    def __call__(self, x, w, out, rows, stream):
        if rows >= Int32(self.min_rows):
            self.prefill(x, w, out, rows, stream)
        else:
            self.base(x, w, out, rows, stream)


def _build_arm(spec, n, k, out_dtype):
    from b12x.gemm.bf16_gemv._skinny import (
        RoutedBf16Projection,
        TmaBf16Projection,
        WIDE_PREFILL_MIN_ROWS,
    )

    base = RoutedBf16Projection(n, k, out_dtype=out_dtype, max_skinny_rows=0)
    if spec["kind"] == "base":
        return base, base.key()
    if spec["kind"] == "wide":
        return (RoutedBf16Projection(n, k, out_dtype=out_dtype, max_skinny_rows=0, wide=True),
                (n, k, "wide", WIDE_PREFILL_MIN_ROWS))
    config = dict(warp_layout=tuple(spec["warp_layout"]), tile_m=int(spec["tile_m"]),
                  tile_n=int(spec["tile_n"]), num_stages=int(spec["num_stages"]), compensated=False)
    if config["tile_m"] % (16 * config["warp_layout"][0]):
        raise SystemExit(f"{spec['name']}: tile_m must be a multiple of 16 * warp_m")
    prefill = TmaBf16Projection(n, k, out_dtype=out_dtype, tiles_by_rows=(), **config)
    min_rows = spec["min_rows"] if spec["min_rows"] is not None else WIDE_PREFILL_MIN_ROWS
    return (_Routed(base, prefill, min_rows),
            (n, k, tuple(sorted(config.items())), int(min_rows)))


def _program(launch, key):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    return compile_program(launch, name="prefill_route_bench",
                           operands=(Operand("x", torch.bfloat16, "x"),
                                     Operand("w", torch.bfloat16, "w"),
                                     Operand("out", torch.bfloat16, "out", "out")),
                           scalars=(Scalar("rows"),), key=key)


def _exact(actual, expected):
    return torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _spread(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    low = ordered[len(ordered) // 10]
    high = ordered[-1 - len(ordered) // 10]
    return {"median_us": median, "p10_us": low, "p90_us": high,
            "min_us": ordered[0], "max_us": ordered[-1],
            "relative_spread": (high - low) / median if median else None,
            "even_odd_delta_us": abs(statistics.median(ordered[::2])
                                     - statistics.median(ordered[1::2])) if len(ordered) > 1 else None}


def main():
    args = parse_args()
    from benchmarks.common import bench_cuda_graph, capture_cuda_graph, make_l2_flush_fn
    from benchmarks.common import nvidia_smi_gpu_mode_snapshot

    specs = [_arm_spec(text) for text in args.arm]
    if len({spec["name"] for spec in specs}) != len(specs):
        raise SystemExit("arm names must be distinct")
    if specs[0]["kind"] != "base":
        specs.insert(0, {"name": "base", "kind": "base"})

    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    flush = make_l2_flush_fn(True)
    results = []
    for n, k in args.shapes:
        generator = torch.Generator(device="cuda").manual_seed(20261111 + n * 1009 + k)
        max_rows = max(args.rows)
        x = torch.empty((max_rows, k), device="cuda", dtype=torch.bfloat16)
        x.normal_(generator=generator)
        w = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        w.normal_(std=0.03, generator=generator)
        out = torch.empty((max_rows, n), device="cuda", dtype=torch.bfloat16)
        arms, keys = {}, {}
        for spec in specs:
            launch, key = _build_arm(spec, n, k, BFloat16)
            arms[spec["name"]], keys[spec["name"]] = _program(launch, key), key
        reference = {}
        for rows in sorted(args.rows, reverse=True):
            for name, program in arms.items():
                out.fill_(POISON)
                program.launch(x, w, out, scalars=(rows,))
                torch.cuda.synchronize()
                expected = out[:rows].clone()
                if name == "base":
                    reference[rows] = expected
                    if not bool(torch.isnan(out[rows:]).all()):
                        raise RuntimeError(f"{n}x{k} rows={rows} base: wrote past the live rows")
                elif not _exact(expected, reference[rows]):
                    bad = (expected.view(torch.int16) != reference[rows].view(torch.int16)).sum().item()
                    raise RuntimeError(f"{n}x{k} rows={rows} {name}: {bad} outputs differ from base")
        # Row invariance: a prefix of a larger-M run must equal the smaller-M run,
        # and an offset input pointer must give the same rows.
        invariance = {"prefix": True, "offset": True}
        largest = max(args.rows)
        for name, program in arms.items():
            out.fill_(POISON)
            program.launch(x, w, out, scalars=(largest,))
            torch.cuda.synchronize()
            full = out[:largest].clone()
            for rows in sorted(args.rows):
                if rows == largest:
                    continue
                out.fill_(POISON)
                program.launch(x, w, out, scalars=(rows,))
                torch.cuda.synchronize()
                if not _exact(out[:rows], full[:rows]):
                    invariance["prefix"] = False
            offset_rows = min(largest - 1, 2752)
            offset = x[1:1 + offset_rows]
            out.fill_(POISON)
            program.launch(offset, w, out, scalars=(offset_rows,))
            torch.cuda.synchronize()
            if not _exact(out[:offset_rows], full[1:1 + offset_rows]):
                invariance["offset"] = False
            if name != "base" and not all(invariance.values()):
                raise RuntimeError(f"{n}x{k} {name}: row invariance failed: {invariance}")
        del full, offset
        cells = []
        gpu_before = nvidia_smi_gpu_mode_snapshot()
        for rows in sorted(args.rows):
            graphs, expected = {}, {}
            for name, program in arms.items():
                out.fill_(POISON)
                program.launch(x, w, out, scalars=(rows,))
                torch.cuda.synchronize()
                expected[name] = out[:rows].clone()
                graph = capture_cuda_graph(
                    lambda p=program, r=rows: p.launch(x, w, out, scalars=(r,)), warmup=2)
                out.fill_(POISON)
                graph.replay()
                torch.cuda.synchronize()
                if not _exact(out[:rows], expected[name]):
                    raise RuntimeError(f"{n}x{k} rows={rows} {name}: graph replay differs from eager")
                if not bool(torch.isnan(out[rows:]).all()):
                    raise RuntimeError(f"{n}x{k} rows={rows} {name}: graph wrote past the live rows")
                graphs[name] = graph
            names = list(arms)
            samples = {name: [] for name in names}
            per_block = max(1, args.samples // args.blocks)
            for block in range(args.blocks):
                for name in (names if block % 2 == 0 else names[::-1]):
                    measured = bench_cuda_graph(graphs[name], replays=per_block, l2_flush=flush)
                    samples[name].extend(measured["replay_us"])
            cell = {"n": n, "k": k, "rows": rows,
                    "arms": {name: _spread(values) for name, values in samples.items()},
                    "samples_us": samples}
            base_us = cell["arms"]["base"]["median_us"]
            cell["ratio"] = {name: value["median_us"] / base_us
                             for name, value in cell["arms"].items()}
            cells.append(cell)
            print(json.dumps({key_: value for key_, value in cell.items() if key_ != "samples_us"}),
                  flush=True)
            for graph in graphs.values():
                graph.reset()
            del graphs, expected
        results.append({"n": n, "k": k, "weight_bytes": n * k * 2, "cells": cells,
                        "row_invariance": invariance, "telemetry_before": gpu_before,
                        "telemetry_after": nvidia_smi_gpu_mode_snapshot()})
        del arms, keys, x, w, out, reference
        torch.cuda.empty_cache()
    result = {"kind": "bf16_prefill_routes", "device": torch.cuda.get_device_name(),
              "device_uuid": str(getattr(properties, "uuid", "unavailable")),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "argv": sys.argv, "sms": properties.multi_processor_count,
              "conditions": "interleaved CUDA graph replays in alternating arm order, "
                            "2x-L2 eviction before every timed replay, bitwise gate against base "
                            "at every cell, graph-replay and poisoned-tail checks, prefix and "
                            "offset row invariance",
              "arms": [spec["name"] for spec in specs], "results": results}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
