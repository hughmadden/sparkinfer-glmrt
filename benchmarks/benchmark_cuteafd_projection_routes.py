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
    parser.add_argument("--recipes", choices=("routes", "compensation", "trellis"), default="routes",
                        help="BF16 route comparisons, or the unchanged native K6 prefill baseline")
    parser.add_argument("--trellis-config", type=Path,
                        help="JSON list of {n,k,side,groups} K6 geometries; trellis recipe only")
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
    row_limit = 4096 if args.recipes in ("compensation", "trellis") else 128
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


def _trellis_baseline(args):
    import numpy as np
    import torch
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from benchmarks.common import make_l2_flush_fn
    from tests._reference.native_trellis import Reference

    configs = (json.loads(args.trellis_config.read_text()) if args.trellis_config else
               [dict(n=n, k=k, side=1, groups=1) for n, k in args.shapes])
    flush = make_l2_flush_fn(True)
    torch.manual_seed(174)
    results = []
    properties = torch.cuda.get_device_properties(0)
    for config in configs:
        n, k, side, groups = (config[key] for key in ("n", "k", "side", "groups"))
        if n % 128 or k % 128 or side not in (1, 2) or groups not in (1, 4):
            raise ValueError(f"invalid trellis geometry: {config}")
        x = torch.randn(groups, max(args.rows), k, device="cuda", dtype=torch.float16) * .1
        inputs = torch.tensor([row.data_ptr() for row in x], device="cuda", dtype=torch.int64)
        payloads = [torch.randint(0, 256, (n*k*6//8,), device="cuda", dtype=torch.uint8)
                    for _ in range(groups)]
        scales = [torch.ones(k//128, device="cuda", dtype=torch.float16) for _ in range(groups)]
        partial = torch.empty(groups*max(args.rows)*((k+1023)//1024)*n,
                              device="cuda", dtype=torch.float32)
        outputs = [torch.empty(max(args.rows), n, device="cuda", dtype=torch.float32)
                   for _ in range(groups)]
        program = compile_trellis_linear_aot(input_features=k, output_features=n, bits=6,
            codebook="mcg", sharded_side=side, rotation_count=groups, groups=groups,
            max_rows=max(args.rows))
        operands = ((inputs, payloads[0], scales[0], partial, outputs[0]) if groups == 1 else
                    (inputs, *payloads, *scales, partial, *outputs))

        # Independently reconstruct one N128 output block, bounded in K128 chunks.
        weight = np.empty((k, 128), dtype=np.float32)
        reference = Reference("mcg")
        tile_bytes = 128*128*6//8
        for kb in range(k//128):
            offset = kb*tile_bytes if side == 1 else kb*(n//128)*tile_bytes
            raw = payloads[0][offset:offset+tile_bytes].cpu().numpy().tobytes()
            tiles = reference.decode_tiles(raw, 6)
            if side == 1:
                block = tiles.reshape(8, 8, 16, 16).transpose(0, 2, 1, 3).reshape(128, 128)
            else:
                block = tiles.reshape(1, 8, 8, 16, 16).transpose(1, 3, 0, 2, 4).reshape(128, 128)
            weight[kb*128:(kb+1)*128] = block
        sample_rows = min(8, min(args.rows))
        w = torch.tensor(weight, device="cuda")
        oracle = x[0, :sample_rows].double() @ w.double()
        baseline = x[0, :sample_rows] @ w.half()
        for rows in args.rows:
            for output in outputs:
                output.fill_(float("nan"))
            scalars = (rows, 0, properties.multi_processor_count*4)
            program.launch(*operands, scalars=scalars)
            torch.cuda.synchronize()
            accuracy = _accuracy(outputs[0][:sample_rows, :128], baseline, oracle)
            if not accuracy["pass"]:
                raise RuntimeError(f"{config} rows={rows}: bounded independent oracle failed: {accuracy}")
            expected = [output[:rows].clone() for output in outputs]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(*operands, scalars=scalars)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            if any(not torch.equal(actual[:rows].view(torch.int32), want.view(torch.int32))
                   for actual, want in zip(outputs, expected)):
                raise RuntimeError("trellis graph replay differs from eager output")
            times = []
            for _ in range(args.samples):
                flush()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end)*1000)
            entry = dict(config, rows=rows, bits=6, codebook="mcg", input_dtype="f16",
                         median_us=statistics.median(times), samples_us=times, accuracy=accuracy)
            results.append(entry)
            print(json.dumps({key: value for key, value in entry.items() if key != "samples_us"}), flush=True)
            if args.output:
                args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),
                    sms=properties.multi_processor_count, torch_version=torch.__version__,
                    cuda_version=torch.version.cuda, argv=sys.argv, kind="native_k6_baseline",
                    conditions="unchanged production core; F16 rotated inputs; one eager and graph warmup per M; L2 eviction per sample; independent N128/K-all/8-row FP64 oracle; byte-exact graph replay",
                    results=results), indent=2)+"\n")
            del graph, expected
        del program, operands, x, inputs, payloads, scales, partial, outputs, w, oracle, baseline
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.recipes == "trellis":
        return _trellis_baseline(args)
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
