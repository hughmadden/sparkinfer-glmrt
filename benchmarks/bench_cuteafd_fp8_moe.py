#!/usr/bin/env python3
"""Time cuteafd FP8/NVFP4/BF16 expert programs with L2-cold weights.

Use --graph for the serving replay path, --routing hot to put the requested
row count in each selected expert, and --profile-dir for launch/stage traces.
NVFP4 and native BF16 require --input bf16. A historical checkout may be used to measure the
old A16 ceiling; it is not a serving arithmetic override.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics

import torch


def time_samples(fn, iters: int, flush: torch.Tensor, prepare) -> list[float]:
    prepare(0)
    fn()
    torch.cuda.synchronize()
    times = []
    for it in range(iters):
        prepare(it + 1)
        flush.max()
        torch.cuda._sleep(5_000_000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return times


def time_us(fn, iters: int, flush: torch.Tensor, prepare) -> float:
    return statistics.median(time_samples(fn, iters, flush, prepare))


def exact_sweep(g, w, args):
    """Compare exact BF16 bytes, including each row's width-one execution.

    Run this same file against both revisions (separate processes/JIT caches).
    Refuse different inputs/weights before comparing actual output bytes, not
    merely rounded oracle metrics or their hashes.
    """
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    widths = [int(r) for r in args.rows.split(",")]
    if not widths or min(widths) < 1 or max(widths) > 64:
        raise ValueError("exact sweep requires widths in 1..64")
    if g.weights != "nvfp4" or g.activations != "a4" or args.input != "bf16":
        raise ValueError("exact sweep requires NVFP4 W4A4 with BF16 input")

    def raw(tensor):
        return tensor.detach().contiguous().view(torch.uint8).cpu()

    def digest(tensor):
        return hashlib.sha256(memoryview(raw(tensor).numpy())).hexdigest()

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    capacity = max(widths)
    x = torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16()
    scores = torch.rand(capacity, g.experts, device="cuda", generator=gen)
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    if args.routing == "hot":
        ids.copy_(ids[:1].expand_as(ids))
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen)
    inputs = {"x": raw(x), "ids": raw(ids), "weights": raw(weights)}
    weight_hashes = [digest(t) for t in w]
    config = {"geometry": g.name, "tp": g.tp, "rows": widths,
              "routing": args.routing, "seed": args.seed, "route": args.route}
    baseline = None
    if args.exact_baseline:
        baseline = torch.load(args.exact_baseline / "outputs.pt", weights_only=True, map_location="cpu")
        assert config == baseline["config"], "baseline configuration differs"
        assert weight_hashes == baseline["weight_sha256"], "baseline weights differ"
        for name, value in inputs.items():
            assert torch.equal(value, baseline["inputs"][name]), f"baseline input differs: {name}"

    def execute(program, source, selected, mix, rows, scratch):
        out = torch.full((rows, g.hidden), float("nan"), device="cuda", dtype=torch.bfloat16)
        # Poison staging: an unstaged dead row must not affect live outputs.
        scratch.fill_(0xA5)

        def run():
            program.launch(source, selected, mix, *w, out, scratch, scalars=(rows,))

        run()
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"nonfinite output at width {rows}"
        return raw(out)

    single = compile_fp8_moe_aot(g, route=args.route, max_rows=1, wire=False)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, args.route, 1, False), device="cuda", dtype=torch.uint8)
    alone = torch.cat([execute(single, x[r:r+1], ids[r:r+1], weights[r:r+1], 1, scratch)
                       for r in range(capacity)])
    outputs, results = {}, []
    for width in widths:
        program = compile_fp8_moe_aot(g, route=args.route, max_rows=width, wire=False)
        scratch = torch.empty(fp8_moe_scratch_bytes(g, args.route, width, False), device="cuda", dtype=torch.uint8)
        actual = execute(program, x[:width], ids[:width], weights[:width], width, scratch)
        assert torch.equal(actual, alone[:width]), f"row invariance failed at width {width}"
        if baseline is not None:
            assert torch.equal(actual, baseline["outputs"][width]), f"baseline bytes differ at width {width}"
        outputs[width] = actual
        record = {"rows": width, "sha256": digest(actual), "row_invariant": True,
                  "baseline_exact": baseline is not None}
        results.append(record)
        print(json.dumps(record), flush=True)
    args.exact_sweep.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "inputs": inputs, "weight_sha256": weight_hashes,
                "outputs": outputs, "alone": alone}, args.exact_sweep / "outputs.pt")
    (args.exact_sweep / "summary.json").write_text(json.dumps({**config, "results": results}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", default="mimo")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--activation", choices=("silu_bf16_product", "silu_post_clamp_fp32_product"),
                        help="explicit activation contract; defaults to the geometry contract")
    parser.add_argument("--rows", default="1,16,80,1024,4096")
    parser.add_argument("--route", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prefill", choices=("auto", "w8a8", "w8a16"), default="auto")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--input", choices=("wire", "bf16"), default="wire")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--routing", choices=("random", "hot"), default="random")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-sweep", type=Path, help="correctness only: dump exact outputs and check row invariance (widths 1..64)")
    parser.add_argument("--exact-baseline", type=Path, help="compare --exact-sweep byte for byte with this baseline dump")
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("iters must be positive")
    if args.exact_baseline and not args.exact_sweep:
        parser.error("--exact-baseline requires --exact-sweep")
    torch.backends.cuda.matmul.allow_tf32 = False

    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = GEOMETRIES[args.geometry].with_tp(args.tp)
    if args.activation is not None:
        g = replace(g, activation=args.activation)
    if g.weights in ("nvfp4", "bf16") and args.input != "bf16":
        parser.error("NVFP4 and native BF16 require original BF16 input")
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    h, i, e, k = g.hidden, g.slice, g.experts, g.top_k

    def fp8(shape):
        return (torch.randn(shape, device=dev, generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)

    def scales(shape):
        return torch.rand(shape, device=dev, generator=gen) * 2e-4 + 1e-4

    if g.weights == "nvfp4":
        # The integration oracle supplies nonuniform per-expert calibration and
        # independent CPU FP64 products, including exact quantizer boundaries.
        from tests.integration.test_cuteafd_nvfp4_moe_aot import _weights, reference

        w = _weights(g, i)
        per_expert = 3 * i * h * 9 // 16 + 24
    elif g.weights == "fp8":
        w = (fp8((e, i, h)), scales((e, i // 128, h // 128)), fp8((e, i, h)), scales((e, i // 128, h // 128)),
             fp8((e, h, i)), scales((e, h // 128, i // 128)))
        per_expert = 3 * i * h + 3 * (i // 128) * (h // 128) * 4
    elif g.weights == "bf16":
        unused = torch.empty(0, device=dev, dtype=torch.uint8)
        def bf16(shape):
            return (torch.randn(shape, device=dev, generator=gen) * 0.03).bfloat16()
        w = (bf16((e, i, h)), unused, bf16((e, i, h)), unused, bf16((e, h, i)), unused)
        per_expert = 3 * i * h * 2
    else:
        parser.error("this benchmark supports FP8, NVFP4 and native BF16 weights")
    if args.exact_sweep:
        exact_sweep(g, w, args)
        return
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    results = []
    for rows in [int(r) for r in args.rows.split(",")]:
        wire_in = args.input == "wire"
        program = compile_fp8_moe_aot(g, route=args.route, max_rows=rows, wire=wire_in, prefill=args.prefill)
        x = torch.randn(rows, h, device=dev, generator=gen).bfloat16()
        source, x_exact = x, x
        if wire_in:
            groups = x.float().view(rows, h // 32, 32)
            exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-127, 127)
            q = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
            source = torch.cat([q.view(rows, h).view(torch.uint8), (exponent + 127).to(torch.uint8)], 1).contiguous()
            x_exact = (q.float() * torch.exp2(exponent)[..., None]).view(rows, h).bfloat16()
        ids = torch.empty(rows, k, dtype=torch.int32, device=dev)
        weights = torch.rand(rows, k, device=dev, generator=gen).contiguous()
        out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
        scratch = torch.empty(fp8_moe_scratch_bytes(g, args.route, rows, wire_in, args.prefill), dtype=torch.uint8,
                              device=dev)
        unique = []

        def prepare(seed):
            if args.routing == "hot":
                selected = torch.randperm(e, device=dev, generator=gen)[:k].int()
                ids.copy_(selected.expand(rows, k))
            else:
                ids.copy_(torch.rand(rows, e, device=dev, generator=gen).topk(k, -1).indices.int())
            unique.append(int(ids.unique().numel()))

        def run():
            program.launch(source, ids, weights, *w, out, scratch, scalars=(rows,))

        prepare(0)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        if args.check:
            assert torch.isfinite(out).all() and out.count_nonzero() > 0
            if g.weights == "nvfp4":
                ref = reference(g, x_exact, ids, weights, *w, a4=g.activations == "a4")
            elif g.weights == "bf16":
                from tests.integration.test_cuteafd_expert_contract import bf16_reference

                ref = bf16_reference(x_exact, ids, weights, w[0], w[2], w[4], g.swiglu_limit, g.activation)
            else:
                from tests.integration.test_cuteafd_expert_contract import activation

                def dequant(wt, s):
                    return (wt.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()

                ref = torch.zeros(rows, h, device=dev)
                for expert in ids.unique().tolist():
                    r, slots = torch.where(ids == expert)
                    gate = x_exact[r] @ dequant(w[0][expert], w[1][expert]).T
                    up = x_exact[r] @ dequant(w[2][expert], w[3][expert]).T
                    y = activation(gate, up, g.swiglu_limit, g.activation) @ dequant(w[4][expert], w[5][expert]).T
                    ref.index_add_(0, r, y.float() * weights[r, slots][:, None])
                ref = ref.bfloat16()
            a, b = out.float(), ref.float()
            correctness = {"cosine": float((a * b).sum() / (a.norm() * b.norm())),
                           "relative_l2": float((a - b).norm() / b.norm()),
                           "normalized_max": float((a - b).abs().max() / b.square().mean().sqrt())}
            assert correctness["cosine"] >= 0.9999, correctness
            if g.weights == "nvfp4" and g.activations == "a4":
                assert correctness["relative_l2"] <= 0.02 and correctness["normalized_max"] <= 0.15, correctness
        else:
            correctness = {"checked": False}

        graph = None
        fn = run
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            fn = graph.replay
            fn()
            torch.cuda.synchronize()
        unique.clear()
        samples = time_samples(fn, args.iters, flush, prepare)
        us = statistics.median(samples)
        experts = statistics.mean(unique)
        record = {"geometry": g.name, "tp": g.tp, "input": args.input, "arithmetic": g.activations,
                  "activation": g.activation, "projection": g.projection, "weight_format": g.weights,
                  "route": args.route, "routing": args.routing, "graph": args.graph, "rows": rows,
                  "us": us, "samples_us": samples, "scratch_bytes": scratch.numel(),
                  "experts": experts, "gbps": experts * per_expert / us / 1e3,
                  "tflops": rows * k * 3 * 2 * i * h / us / 1e6, "correctness": correctness}
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            flush.max()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                run()
                torch.cuda.synchronize()
            trace = args.profile_dir / f"{g.name}-tp{g.tp}-{args.routing}-r{rows}.json"
            prof.export_chrome_trace(str(trace))
            kernels = [event for event in json.loads(trace.read_text())["traceEvents"]
                       if event.get("cat") == "kernel" and "dur" in event]
            record["kernels"] = [{"name": event["name"], "us": event["dur"],
                                  "grid": event.get("args", {}).get("grid"),
                                  "block": event.get("args", {}).get("block")}
                                 for event in kernels]
            record["launches"] = len(kernels)
        results.append(record)
        print(json.dumps(record) if args.json else
              " ".join(f"{key}={value}" for key, value in record.items() if key not in ("samples_us", "kernels")),
              flush=True)
        del graph, fn, program, scratch
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        args.output.write_text(json.dumps({"device": str(props), "torch": torch.__version__,
                                          "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
