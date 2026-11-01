#!/usr/bin/env python3
"""Time cuteafd FP8/NVFP4/BF16 expert programs with L2-cold weights.

Use --graph for the serving replay path, --routing hot to put the requested
row count in each selected expert, and --profile-dir for launch/stage traces.
Use --capacity for a fixed export across live --rows and --routes for
schema-1 recorded histograms or existing expert-routes JSONL logs.
NVFP4 and native BF16 require --input bf16. A historical checkout may be used to measure the
old A16 ceiling; it is not a serving arithmetic override.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
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


def time_interleaved_samples(fns: dict[str, object], iters: int, flush: torch.Tensor, prepare) -> dict[str, list[float]]:
    names = list(fns)
    if len(names) == 1:
        return {names[0]: time_samples(fns[names[0]], iters, flush, prepare)}
    prepare(0)
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    times = {name: [] for name in names}
    for it in range(iters):
        prepare(it + 1)
        order = names if it % 2 == 0 else list(reversed(names))
        for name in order:
            flush.max()
            torch.cuda._sleep(5_000_000)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fns[name]()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000.0)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", default="mimo")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--activation", choices=("silu_bf16_product", "silu_post_clamp_fp32_product"),
                        help="explicit activation contract; defaults to the geometry contract")
    parser.add_argument("--rows", default="1,16,80,1024,4096")
    parser.add_argument("--capacity", type=int, help="fixed compiled export/workspace capacity; defaults to each live row count")
    parser.add_argument("--routes", type=Path, help="schema-1 histograms or an existing expert-routes JSONL")
    parser.add_argument("--route", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="record failed oracle gates and continue; failed timings are ineligible")
    parser.add_argument("--check-prefix-rows", type=int,
                        help="compare longer cases with this identical-input prefix under the same export")
    parser.add_argument("--compare-routes-exact", action="store_true",
                        help="compare every route's raw output bytes against the first route outside timing")
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
    parser.add_argument("--mxfp4-pair", action="store_true",
                        help="interleave identical logical weights in padded and exact K32 storage")
    parser.add_argument("--widths", help="paired-screen rank widths; defaults to every T2..8 adjacent width")
    parser.add_argument("--engine-policy-root", type=Path)
    parser.add_argument("--shipped-prefill", action="store_true",
                        help="qualify ordered native STREAM against shipped prefill recipes")
    parser.add_argument("--native-slices", action="store_true",
                        help="paired native V4.1/V4 slice pipelines instead of grouped programs")
    parser.add_argument("--mxfp4-tilings", action="store_true",
                        help="screen native tiles 64/128/192 (explicit arms also allow 32) or grouped decode warp partitions")
    parser.add_argument("--mxfp4-w2-pitch", action="store_true",
                        help="development native arm: exact W13 rows with only W2 payload pitch aligned to 128 columns")
    parser.add_argument("--mxfp4-arm", action="append", default=[], metavar="NAME:LOGICAL:STORED:TILE",
                        help="explicit development native arm; repeat for each resident extent and slice tile")
    parser.add_argument("--control-repeats", type=int, default=2,
                        help="identical padded controls in the tiling screen, including its reference")
    args = parser.parse_args(argv)
    if args.shipped_prefill and (not args.native_slices or not args.mxfp4_pair
            or args.geometry not in ("v41", "dsv4f", "dsv4p") or args.engine_policy_root is None
            or args.capacity is not None or args.mxfp4_tilings or args.routes
            or any(int(row) not in (256, 1024, 4096) for row in args.rows.split(","))):
        parser.error("shipped prefill requires native V4/V4.1 pairs, policy root and prefill capacities")
    if args.iters <= 0:
        parser.error("iters must be positive")
    if args.continue_on_fail and (not args.check or args.exact_sweep):
        parser.error("--continue-on-fail requires --check and no --exact-sweep")
    if args.exact_baseline and not args.exact_sweep:
        parser.error("--exact-baseline requires --exact-sweep")
    try:
        rows = [int(r) for r in args.rows.split(",")]
    except ValueError:
        parser.error("rows must be comma-separated positive integers")
    if not rows or min(rows) < 1:
        parser.error("rows must be positive")
    if args.capacity is not None and (args.capacity < 1 or max(rows) > args.capacity):
        parser.error("capacity must be positive and cover every live row count")
    if args.check_prefix_rows is not None and (not args.check or args.exact_sweep
            or args.capacity is None or not 1 <= args.check_prefix_rows < max(rows)):
        parser.error("--check-prefix-rows requires --check, fixed --capacity, a longer case and no --exact-sweep")
    if args.routes and args.routing != "random":
        parser.error("--routes cannot be combined with --routing hot")
    if args.exact_sweep and (args.capacity is not None or args.routes):
        parser.error("--capacity/--routes are timing options, not --exact-sweep options")
    if not args.route or len(set(args.route.split(","))) != len(args.route.split(",")) or "" in args.route.split(","):
        parser.error("routes must be nonempty and unique")
    if args.compare_routes_exact and (args.exact_sweep or len(args.route.split(",")) < 2):
        parser.error("--compare-routes-exact requires at least two timing routes and no --exact-sweep")
    if args.native_slices and not args.mxfp4_pair:
        parser.error("--native-slices requires --mxfp4-pair")
    if args.mxfp4_tilings and (not args.mxfp4_pair or args.capacity is not None):
        parser.error("--mxfp4-tilings requires --mxfp4-pair without fixed capacity")
    if args.mxfp4_w2_pitch and not (args.mxfp4_tilings and args.native_slices):
        parser.error("--mxfp4-w2-pitch requires --mxfp4-tilings and --native-slices")
    if args.control_repeats < 2:
        parser.error("control-repeats must be at least two")
    if args.mxfp4_arm:
        if not (args.native_slices and args.mxfp4_tilings) or args.mxfp4_w2_pitch or args.widths:
            parser.error("explicit arms require the native tiling screen without --widths or W2 pitch arms")
        try:
            native_screen_arms(args.mxfp4_arm)
        except ValueError as error:
            parser.error(str(error))
    if args.widths:
        try:
            widths = [int(width) for width in args.widths.split(",")]
        except ValueError:
            parser.error("widths must be comma-separated K32 widths")
        if not args.mxfp4_pair or not widths or any(width <= 0 or width % 32 for width in widths):
            parser.error("--widths requires --mxfp4-pair and positive K32 widths")
    if args.routes and args.mxfp4_pair and not args.mxfp4_tilings:
        parser.error("paired recorded routes require --mxfp4-tilings")
    if args.mxfp4_pair and (args.exact_sweep or args.profile_dir or args.check_prefix_rows
                           or args.compare_routes_exact or args.input != "wire"):
        parser.error("paired MXFP4 screening requires wire input and no other sweep/profile mode")
    return args


def compare_route_outputs(outputs):
    """Compare immutable snapshots after running every route on identical inputs."""
    reference_name = next(iter(outputs))
    reference = outputs[reference_name].detach().contiguous().cpu()
    reference_bytes = reference.view(torch.uint8)
    comparisons = {}
    for name, output in outputs.items():
        actual = output.detach().contiguous().cpu()
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError("route outputs must have identical shapes and dtypes")
        actual_bytes = actual.view(torch.uint8)
        exact = torch.equal(reference_bytes, actual_bytes)
        mismatch = None
        if not exact:
            location = (reference_bytes != actual_bytes).nonzero()[0].tolist()
            element = list(location)
            element[-1] //= reference.element_size()
            mismatch = dict(byte_index=location, element_index=element,
                            reference_byte=int(reference_bytes[tuple(location)]),
                            actual_byte=int(actual_bytes[tuple(location)]))
        comparisons[name] = dict(reference_route=reference_name, bitwise_equal=exact,
                                 first_mismatch=mismatch,
                                 max_abs_diff=float((actual.float() - reference.float()).abs().max()))
    return comparisons


def check_shared_prefix(full_output, prefix_rows, execute):
    """Snapshot full output before reusing the same callable and scratch."""
    full = full_output.detach().clone()
    prefix = execute(prefix_rows)
    comparison = compare_route_outputs(dict(full=full[:prefix_rows], prefix=prefix))["prefix"]
    actual_bytes = prefix.detach().contiguous().cpu().view(torch.uint8)
    full_bytes = full[:prefix_rows].contiguous().cpu().view(torch.uint8)
    comparison["rows"] = prefix_rows
    comparison["differing_rows"] = int((actual_bytes != full_bytes).any(dim=1).sum())
    comparison["full_prefix_sha256"] = hashlib.sha256(full_bytes.numpy().tobytes()).hexdigest()
    comparison["prefix_sha256"] = hashlib.sha256(actual_bytes.numpy().tobytes()).hexdigest()
    return comparison


def oracle_metrics(output, reference, *, nvfp4_a4=False):
    """Normalize by global reference RMS, never by an individual element."""
    a, b = output.float(), reference.float()
    if a.ndim != 2 or a.shape != b.shape or not a.numel():
        raise ValueError("oracle tensors must have identical nonempty matrix shapes")
    error = (a - b).abs()
    flat_index = int(error.nan_to_num(nan=float("inf")).argmax())
    row, col = divmod(flat_index, a.shape[1])
    rms = b.square().mean().sqrt()
    threshold = 0.15 * rms
    result = dict(cosine=float((a * b).sum() / (a.norm() * b.norm())),
                  relative_l2=float((a - b).norm() / b.norm()),
                  normalized_max=float(error.max() / rms), reference_rms=float(rms),
                  max_error=dict(row=row, col=col, output=float(a[row, col]),
                                 reference=float(b[row, col]), abs_error=float(error[row, col])),
                  error_threshold_abs=float(threshold), error_threshold_rms=0.15,
                  elements_over_threshold=int((error > threshold).sum()),
                  finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                  nonzero=bool(a.count_nonzero()))
    result["passed"] = (result["finite"] and result["nonzero"] and result["cosine"] >= 0.9999
                        and (not nvfp4_a4 or (result["relative_l2"] <= 0.02
                                             and result["normalized_max"] <= 0.15)))
    return result


def enforce_oracle(result, continue_on_fail):
    if not result["passed"] and not continue_on_fail:
        raise AssertionError(result)


def recorded_waves(path, g):
    from benchmarks.benchmark_exlr8_a16 import load_route_waves

    if path.suffix == ".jsonl":
        waves = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            wave = json.loads(line)
            if wave.get("event") != "expert_routes":
                raise ValueError("not an expert-routes record")
            if "topk" in wave:
                ids = wave["topk"]
                if (not isinstance(ids, list) or len(ids) != wave.get("rows")
                        or any(not isinstance(row, list) or len(row) != g.top_k
                               or any(type(e) is not int for e in row) or len(set(row)) != len(row)
                               for row in ids)):
                    raise ValueError("invalid recorded per-row top-k")
                wave["expert_rows"] = sorted(Counter(e for row in ids for e in row).items())
            waves.append(wave)
        if not waves:
            raise ValueError("empty expert-routes log")
        # JSONLs do not attest checkpoint/family identity; preserve that limitation.
        data = {}
    else:
        data = json.loads(path.read_text())
        waves = load_route_waves(str(path))
    family = g.name.split("_")[0]
    if path.suffix != ".jsonl" and data.get("family") != family:
        # The original GLM histogram predates explicit family metadata.
        if not (family == "glm" and "family" not in data and data.get("model") == "zai-org/GLM-5.3"):
            raise ValueError("recorded route fixture family does not match geometry")
    if data.get("experts", g.experts) != g.experts or data.get("top_k", g.top_k) != g.top_k:
        raise ValueError("recorded route fixture expert/top-k metadata mismatch")
    for wave in waves:
        rows, counts = wave.get("rows", 0), wave.get("expert_rows", [])
        if (type(rows) is not int or rows < 1 or not counts
                or any(type(e) is not int or type(n) is not int or not 0 <= e < g.experts
                       or not 0 < n <= rows for e, n in counts)
                or len({e for e, _ in counts}) != len(counts)
                or sum(n for _, n in counts) != rows * g.top_k):
            raise ValueError("invalid recorded route histogram")
    return waves


def native_screen_arms(values):
    """Parse rank extents, not ownership or serving configuration."""
    result = []
    for value in values:
        try:
            name, logical, stored, tile = value.split(":")
            logical, stored, tile = int(logical), int(stored), int(tile)
        except ValueError as error:
            raise ValueError("native arm must be NAME:LOGICAL:STORED:TILE") from error
        if (not name or name.startswith(("padded", "matching", "exact-", "aligned-w2"))
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name)
                or not 0 < logical <= stored or logical % 32 or stored % 32
                or tile not in (32, 64, 128, 192)):
            raise ValueError("native arms need unique names, whole-K32 extents and tile32/64/128/192")
        if name in {arm[0] for arm in result}:
            raise ValueError("duplicate native arm name")
        result.append((name, logical, stored, tile))
    return result


def extend_native_weights(weights, scales, stored):
    """Add zero columns without changing any real K32 activation block."""
    logical = weights["w1"].shape[1]
    if stored < logical or stored % 32:
        raise ValueError("resident extent must cover logical whole-K32 columns")
    pad = torch.nn.functional.pad
    resident = {name: pad(value, (0, 0, 0, stored - logical))
                for name, value in weights.items() if name != "w2"}
    resident["w2"] = pad(weights["w2"], (0, (stored - logical) // 2))
    factors = {name: pad(value, (0, 0, 0, stored - logical))
               for name, value in scales.items() if name != "w2"}
    # Keep the real down-scale range: zero payload needs no special exponent.
    extra = stored // 32 - logical // 32
    down = scales["w2"]
    factors["w2"] = torch.cat((down, down.amin(-1, keepdim=True).expand(*down.shape[:-1], extra)), -1)
    return resident, factors


def screen_route_wave(waves, rows, seed):
    matches = [wave for wave in waves if wave["rows"] == rows]
    if not matches:
        raise ValueError(f"recorded routes contain no wave with exactly {rows} live rows")
    return matches[seed % len(matches)]


def screen_route_ids(wave, experts, topk, seed):
    if "topk" in wave:
        return torch.tensor(wave["topk"], dtype=torch.int32)
    from benchmarks.benchmark_exlr8_a16 import real_route_ids

    return real_route_ids(wave["rows"], wave, torch.Generator().manual_seed(seed), experts, topk).int()


def tiling_specs(args, width, baseline_tile, tiles):
    """Keep identical controls separate from eligible candidate layouts."""
    controls = ["padded", *(f"padded-control-{n}" for n in range(1, args.control_repeats))]
    specs = [(label, False, baseline_tile, None) for label in controls]
    if args.mxfp4_arm:
        specs += [(name, True, tile, stored) for name, logical, stored, tile
                  in native_screen_arms(args.mxfp4_arm) if logical == width]
    else:
        specs.append(("matching", True, baseline_tile, None))
        specs += [(f"exact-{tile}", True, tile, None) for tile in tiles if tile != baseline_tile]
        if args.mxfp4_w2_pitch:
            specs = specs[:len(controls) + 1] + [("aligned-w2", True, baseline_tile, None),
                                                ("aligned-w2-scales", True, baseline_tile, None)]
    return specs, controls


def tiling_timing_summary(samples, eligible, controls):
    paired = {label: statistics.median(base / value for base, value in zip(samples["padded"], times))
              for label, times in samples.items()}
    candidates = [label for label in samples if label not in controls and eligible[label]]
    winner = max(candidates, key=paired.get) if candidates else None
    control_ratios = {label: [base / value for base, value in zip(samples["padded"], samples[label])]
                      for label in controls if label != "padded"}
    return dict(medians_us={label: statistics.median(times) for label, times in samples.items()},
                paired_speedups=paired, winner=winner, speedup=paired[winner] if winner else None,
                identical_control_speedups=control_ratios)


def compiled_case(g, args, rows, route, cache):
    """Resolve once per planned capacity; live rows remain launch scalars."""
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    capacity = args.capacity if args.capacity is not None else rows
    key = (route, capacity)
    if key not in cache:
        wire = args.input == "wire"
        program = compile_fp8_moe_aot(g, route=route, max_rows=capacity, wire=wire, prefill=args.prefill)
        scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity, wire, args.prefill),
                              dtype=torch.uint8, device="cuda")
        cache[key] = program, scratch
    return cache[key]


def engine_layout_policy(root):
    if root is None:
        raise ValueError("implicit widths require --engine-policy-root; otherwise pass --widths")
    path = Path(root) / "python/tools/lib/mxfp4_layouts.py"
    spec = importlib.util.spec_from_file_location("benchmark_engine_mxfp4_layouts", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy


def paired_widths(intermediate, requested=None, policy_root=None):
    if requested:
        widths = {int(width) for width in requested.split(",")}
    else:
        policy = engine_layout_policy(policy_root)
        widths = set()
        for world in range(2, 9):
            unit = policy.ownership_unit(intermediate, world)
            units = intermediate // unit
            widths.update((unit * (units // world), unit * -(-units // world)))
    if any(width <= 0 or width % 32 or width > intermediate for width in widths):
        raise ValueError("paired widths must be whole K32 within the checkpoint intermediate")
    return sorted(widths)


def shipped_prefill_recipe(family, hidden, rank_width, capacity, policy_root):
    policy = engine_layout_policy(policy_root)
    matching = 128 if family == "dsv4f" else native_slice_width(family, rank_width)
    # Equal V4 roles pass atomic-min-capacity; unequal width exports do not.
    atomic_widths = {"dsv4f": (256, 512, 1024),
                     "dsv4p": (384, 512, 768, 1024, 1536)}
    return dict(tile=policy.native_slice_width(hidden, rank_width, 121, capacity, matching),
                exact=policy.stored_width(hidden, rank_width) == rank_width,
                atomic=capacity >= 256 and rank_width in atomic_widths.get(family, ()),
                coded=family == "v41")


def native_slice_width(family, rank_width):
    # Preserve V4.1 TP8's shipped width-192 fold across its masked final plane.
    return 192 if family == "v41" and (rank_width % 192 == 0 or rank_width == 288) else 128


def native_pair_arm(x, ids, routing, weights, scales, width, exact, atomic, coded, *, qualify=False,
                    aligned_w2=False, aligned_w2_scales=False, stored_width=None,
                    combine_output=False, fc1_k_splits=None):
    """Resolve once; keep every pipeline operand alive through graph timing."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
    from b12x.moe._shared.scale_codes import pack_scale_codes
    from tests.moe.test_v41_exact_slices import pack

    rows, hidden = x.shape
    experts, intermediate = weights["w1"].shape[:2]
    logical = intermediate
    if stored_width is not None:
        if aligned_w2 or aligned_w2_scales or atomic:
            raise ValueError("explicit extents require ordered native arms without W2 pitch overrides")
        weights, scales = extend_native_weights(weights, scales, stored_width)
        intermediate = stored_width
    topk = ids.shape[1]
    # Padded is a benchmark oracle, never an installed alternate role.
    tail = exact and (intermediate % 128 != 0 or
                      (logical + width - 1) // width * width > intermediate)
    if stored_width is not None:
        tail = intermediate % 128 != 0 or (logical + width - 1) // width * width > intermediate
    if aligned_w2 and (not tail or atomic or not coded):
        raise ValueError("aligned W2 screen requires ordered coded-scale exact native tails")
    if aligned_w2_scales and not aligned_w2:
        raise ValueError("aligned W2 scales require the aligned W2 payload arm")
    w2_pitch = (intermediate + 127) // 128 * 128 if aligned_w2 else None
    packed = pack(weights, scales, tail, w2_pitch=w2_pitch)
    if coded:
        stored = intermediate if exact else (intermediate + 127) // 128 * 128
        s13 = torch.cat(tuple(torch.nn.functional.pad(scales[name], (0, 0, 0, stored - intermediate))
                             for name in ("w3", "w1")), 1)
        packed[1] = pack_scale_codes(s13, tensor_name="w13.scale", gated_half_rows=stored).view(torch.uint32).flatten().cuda()
        down_scales = scales["w2"]
        if aligned_w2_scales:
            extra = w2_pitch // 32 - intermediate // 32
            down_scales = torch.cat((down_scales, down_scales.amin(-1, keepdim=True)
                                     .expand(*down_scales.shape[:-1], extra)), -1)
        packed[3] = pack_scale_codes(down_scales, tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
    groups = x.float().reshape(rows, -1, 32)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    wire = torch.cat(((groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
                      .view(torch.uint8).reshape(rows, hidden), (exponent + 127).to(torch.uint8)), 1)
    routes = rows * topk
    partial = torch.empty((1,) if atomic else
                          ((logical + width - 1) // width, routes, width // 4 + width // 32), device="cuda")
    output = torch.empty(rows * hidden if atomic else (routes, hidden), device="cuda")
    scratch = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in (
        ((1,), torch.int32), ((experts * routes,), torch.int32), ((experts,), torch.int32),
        ((experts, 2), torch.int32), ((routes, 19), torch.int32),
        ((routes,), torch.float32), ((routes,), torch.int32))]
    tensors = [wire[:, :hidden].view(torch.uint32), wire[:, hidden:], *packed,
               ids.flatten(), routing.flatten(), *scratch, partial, output]
    args = [from_dlpack(tensor, assumed_align=16) for tensor in tensors]
    pipeline = V41SlicePipeline(rows, width, experts=experts, topk=topk,
                                intermediate=intermediate, hidden=hidden,
                                exact_storage=tail or stored_width is not None,
                                atomic_tokens=atomic, scale_codes=coded, fc1_k_splits=fc1_k_splits)
    if stored_width is not None:
        # Physical strides come from P; the launch only visits the real S slices.
        for kernel in (pipeline.fc1, pipeline.fc2, pipeline.reduce):
            kernel.slices = (logical + width - 1) // width
        for kernel in (pipeline.fc1, pipeline.fc2):
            kernel.exact_storage = tail
    if aligned_w2:
        # Isolate W2 payload pitch; W13 and both coded scale planes stay exact.
        pipeline.fc2.w2_intermediate = w2_pitch
        if aligned_w2_scales:
            pipeline.fc2.w2_scale_columns = w2_pitch // 32
    launch = cute.compile(pipeline, *args, cutlass.Int32(rows), current_cuda_stream())
    token_output = torch.empty_like(x) if combine_output else output
    combine = None
    if combine_output:
        from b12x.moe._shared.kernels.v4_stream_pipeline import NativeTokenCombine
        source_tensor = from_dlpack(output.view(rows, hidden) if atomic else output,
                                    assumed_align=16)
        token_tensor = from_dlpack(token_output, assumed_align=16)
        combine = cute.compile(NativeTokenCombine(hidden, 1 if atomic else topk),
            source_tensor.iterator, token_tensor.iterator, cutlass.Int32(rows), current_cuda_stream())
    def execute(live):
        launch(*args, cutlass.Int32(live), current_cuda_stream())
        if combine is not None:
            combine(source_tensor.iterator, token_tensor.iterator, cutlass.Int32(live), current_cuda_stream())
    if qualify:
        if atomic and not combine_output:
            raise ValueError("atomic qualification requires combined token output")
        qualifier = qualify_atomic_pair_replay if atomic else qualify_pair_replay
        qualifier(execute, ids, token_output, [partial, *scratch, output] if combine_output else [partial, *scratch],
                  [*tensors, token_output], lambda live: token_output[:live if combine_output else live * topk])
    def run():
        execute(rows)
    return run, token_output, (tensors, args, launch, combine, execute)


def mxfp4_decode_partitions(width):
    blocks = (width + 127) // 128
    matching = 4 if blocks % 4 == 0 else max(d for d in range(1, 9) if blocks % d == 0)
    return sorted({1, 2, 4, 8, matching})


def grouped_pair_program(g, route, capacity, prefill="auto", *, down_warps=None, template=None):
    """Development-only tiling arms; no serving option or geometry override."""
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, _Fp8Moe, DECODE_MAX_ROWS
    from b12x.integration.cuteafd._mxfp4_moe_kernels import GroupedMxfp4Gemv
    from b12x.integration.cuteafd._common import compile_program

    template = template or compile_fp8_moe_aot(g, route=route, max_rows=capacity, prefill=prefill)
    if down_warps is None:
        return template
    if g.weights != "mxfp4" or route != "decode" or down_warps not in range(1, 9):
        raise ValueError("partition arms require MXFP4 decode and 1..8 warps")
    launch = _Fp8Moe(g, route, capacity, True, prefill)
    launch.decode.down = GroupedMxfp4Gemv(n=g.hidden, k=g.slice, experts=g.experts,
                                         max_rows=min(capacity, DECODE_MAX_ROWS), warps=down_warps)
    return compile_program(launch, name=template.name + f"_screen_w{down_warps}",
                           operands=template.operands, scalars=template.scalars, key=launch.key(),
                           geometry={**template.geometry, "screen_down_warps": down_warps},
                           scratch=template.scratch)


def mxfp4_f32_reference(x, ids, routing, weights, scales, limit=0.0, *, native=False):
    """F32 products with checkpoint BF16 gate/up, SiLU and activation stores.

    The down product and slot-order weighted sum remain F32, without the
    kernel's BF16 down/output rounding. TF32 must be disabled by the caller.
    """
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows
    _, qx = wire_rows(x)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=x.device)
    routes = torch.empty((*ids.shape, x.shape[1]), device=x.device, dtype=torch.float32)
    for expert in ids.unique().tolist():
        selected, slots = torch.where(ids == expert)
        def dequant(name):
            codes = weights[name][expert].to(x.device)
            values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).flatten(-2)
            factors = scales[name][expert].to(x.device).view(torch.float8_e8m0fnu).float()
            return values * factors.repeat_interleave(32, -1)
        gate = (qx[selected].float() @ dequant("w1").T).bfloat16().float()
        up = (qx[selected].float() @ dequant("w3").T).bfloat16().float()
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        if native:
            act = (torch.nn.functional.silu(gate) * up * routing[selected, slots, None]).bfloat16()
            _, act = wire_rows(act)
        else:
            act = (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16()
        routes[selected, slots] = act.float() @ dequant("w2").T
    if native:
        return routes.reshape(-1, x.shape[1])
    output = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        output += routes[:, slot] * routing[:, slot, None]
    return output


def f32_errors(output, reference):
    difference = output.float() - reference.float()
    return {"max_abs": difference.abs().max().item(), "rms": difference.square().mean().sqrt().item()}


def qualify_atomic_pair_replay(run, ids, output, scratch, buffers, prefix):
    """Atomic controls may reorder FP32 additions; do not claim byte determinism."""
    rows = ids.shape[0]
    run(rows)
    expected = output.clone()
    original = ids.clone()
    for live in sorted({1, min(17, rows), rows}):
        output.fill_(float("nan"))
        for tensor in scratch:
            tensor.fill_(255)
        run(live)
        torch.testing.assert_close(prefix(live), expected[:live], rtol=.02, atol=.03125)
    if rows > 1:
        ids[1:] = ids[:1].expand_as(ids[1:])
        run(rows)
        torch.testing.assert_close(prefix(1), expected[:1], rtol=.02, atol=.03125)
        ids.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(rows)
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=.02, atol=.03125)
    return dict(deterministic=False, graph_reuse=True, comparison="tolerance-only")


def qualify_pair_replay(run, ids, output, scratch, buffers, prefix):
    """Poison, prefix/wave-mix invariance and fixed-address graph replay."""
    rows = ids.shape[0]
    addresses = tuple(t.data_ptr() for t in buffers)
    run(rows)
    expected = output.clone()
    for _ in range(3):
        run(rows)
        assert torch.equal(output, expected), "repeat determinism"
    for live in sorted({1, min(17, rows), rows}):
        output.fill_(float("nan"))
        for tensor in scratch:
            tensor.fill_(255)
        run(live)
        assert torch.equal(prefix(live), expected[:prefix(live).shape[0]]), ("live prefix", live)
    if rows > 1:
        original = ids.clone()
        ids[1:] = ids[:1].expand_as(ids[1:])
        run(rows)
        assert torch.equal(prefix(1), expected[:prefix(1).shape[0]]), "wave-mix prefix"
        ids.copy_(original)
    run(rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(rows)
    output.fill_(float("nan"))
    for tensor in scratch:
        tensor.fill_(255)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
        assert torch.equal(output, expected), "graph replay"
    assert torch.cuda.memory_allocated() == allocated
    assert tuple(t.data_ptr() for t in buffers) == addresses
    return {"deterministic": True, "row_invariant": True, "graph_reuse": True}


def production_stream_arm(x, ids, routing, weights, scales, coded):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v4_stream_pipeline import V4StreamPipeline, NativeTokenCombine
    from b12x.moe._shared.scale_codes import pack_scale_codes
    from tests.moe.test_v41_exact_slices import pack
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows

    rows, hidden = x.shape
    experts, width = weights["w1"].shape[:2]
    topk = ids.shape[1]
    packed = pack(weights, scales, width % 128 != 0)
    if coded:
        packed[1] = pack_scale_codes(torch.cat((scales["w3"], scales["w1"]), 1),
            tensor_name="w13.scale", gated_half_rows=width).cuda().flatten()
        packed[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").cuda().flatten()
    wire, _ = wire_rows(x)
    pipeline = V4StreamPipeline(rows, hidden=hidden, intermediate=width,
                               experts=experts, topk=topk, scale_codes=coded)
    arena = torch.empty(pipeline.scratch_bytes, dtype=torch.uint8, device="cuda")
    dtype = {"u8": torch.uint8, "i32": torch.int32, "f32": torch.float32}
    scratch = [arena[spec["offset"]:spec["offset"] + spec["nbytes"]]
               .view(dtype[spec["dtype"]]).reshape(spec["shape"]) for spec in pipeline.layout]
    output = torch.empty_like(x)
    tensors = [wire, *packed, ids.flatten(), routing.flatten(), *scratch]
    views = [from_dlpack(t, assumed_align=16) for t in tensors]
    pointers = [view.iterator for view in views]
    launch = cute.compile(pipeline, *pointers, cutlass.Int32(rows), current_cuda_stream())
    token_view = from_dlpack(output, assumed_align=16)
    combine = cute.compile(NativeTokenCombine(hidden, topk), pointers[-1], token_view.iterator,
                           cutlass.Int32(rows), current_cuda_stream())
    def execute(live):
        launch(*pointers, cutlass.Int32(live), current_cuda_stream())
        combine(pointers[-1], token_view.iterator, cutlass.Int32(live), current_cuda_stream())
    qualification = qualify_pair_replay(execute, ids, output, [arena], [*tensors, output],
                                         lambda live: output[:live])
    original_ids = ids.clone()
    ids[0].fill_(-1)
    ids[-1, -1] = experts
    execute(rows)
    invalid_expected = output.clone()
    assert torch.count_nonzero(output[0]) == 0, "invalid route row must be zero"
    arena.fill_(255)
    output.fill_(float("nan"))
    execute(rows)
    assert torch.equal(output, invalid_expected), "invalid route poison/repeat"
    assert torch.isfinite(output).all(), "invalid route output"
    ids.copy_(original_ids)
    execute(rows)
    qualification["invalid_routes"] = True
    return lambda: execute(rows), output, (arena, tensors, views, launch, combine, token_view), qualification


def shipped_stream_screen(g, args):
    properties = torch.cuda.get_device_properties(0)
    sm = properties.major * 10 + properties.minor
    if g.name == "dsv4p" and sm != 121:
        raise ValueError("the V4 Pro shipped STREAM screen supports GB10 (SM121) only")
    l2_bytes = int(properties.L2_cache_size)
    flush = torch.zeros(max(256 << 20, 2 * l2_bytes), dtype=torch.uint8, device="cuda")
    records = []
    for width in paired_widths(g.intermediate, args.widths, args.engine_policy_root):
        expert_bytes = 3 * width * g.hidden // 2
        experts = max(16, (2 * l2_bytes // (16 * expert_bytes) + 1) * 16)
        generator = torch.Generator().manual_seed(args.seed + width)
        weights, scales = {}, {}
        for name, shape in (("w1", (experts, width, g.hidden // 2)),
                            ("w3", (experts, width, g.hidden // 2)),
                            ("w2", (experts, g.hidden, width // 2))):
            weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
            scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                        dtype=torch.uint8, generator=generator)
        for rows in map(int, args.rows.split(",")):
            torch.manual_seed(args.seed + rows)
            x = (torch.randn(rows, g.hidden, device="cuda") * .5).bfloat16()
            ids = torch.rand(rows, experts, device="cuda").topk(g.top_k, -1).indices.int()
            if args.routing == "hot":
                ids.copy_(ids[:1].expand_as(ids))
            routing = torch.rand(rows, g.top_k, device="cuda")
            routing /= routing.sum(-1, keepdim=True)
            if sm == 120:
                recipe = dict(tile=192 if g.name == "v41" else 128, exact=True,
                              atomic=g.name == "dsv4f", coded=g.name == "v41")
            else:
                recipe = shipped_prefill_recipe(g.name, g.hidden, width, rows, args.engine_policy_root)
            runs, outputs, keepalive, qualifications = {}, {}, [], {}
            controls = ("current", "current-2", "current-3")
            for label in controls:
                runs[label], outputs[label], keep = native_pair_arm(x, ids, routing, weights, scales,
                    recipe["tile"], recipe["exact"], recipe["atomic"], recipe["coded"],
                    qualify=True, combine_output=True)
                keepalive.append(keep)
            runs["native-stream"], outputs["native-stream"], keep, qualifications["native-stream"] = (
                production_stream_arm(x, ids, routing, weights, scales, recipe["coded"]))
            keepalive.append(keep)
            oracle_routes = mxfp4_f32_reference(x, ids, routing, weights, scales, limit=10., native=True)
            oracle = torch.zeros_like(x, dtype=torch.float32)
            for slot in range(g.top_k):
                oracle += oracle_routes.reshape(rows, g.top_k, g.hidden)[:, slot]
            errors = {}
            for label, output in outputs.items():
                error = f32_errors(output, oracle)
                error["normalized_rms"] = error["rms"] / max(oracle.square().mean().sqrt().item(), 1e-8)
                assert error["normalized_rms"] < .02, (label, error)
                errors[label] = error
            if args.graph:
                for label, run in list(runs.items()):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    keepalive.append(graph)
                    runs[label] = graph.replay
            samples = time_interleaved_samples(runs, args.iters, flush, lambda _: None)
            medians = {label: statistics.median(values) for label, values in samples.items()}
            control = [medians[label] for label in controls]
            record = dict(geometry=g.name, sm=sm, width=width, rows=rows, shipped_recipe=recipe,
                controls=controls, samples_us=samples, median_us=medians,
                speedup=min(control) / medians["native-stream"],
                control_spread=max(control) / min(control) - 1,
                regression_beyond_spread=medians["native-stream"] > max(control),
                qualifications=qualifications, matched_f32_errors=errors,
                scratch_bytes=keep[0].numel(), stream_scale_representation="codes" if recipe["coded"] else "raw")
            records.append(record)
            print(json.dumps(record), flush=True)
            del keepalive, runs, outputs, keep
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(device=str(properties), results=records), indent=2) + "\n")


def mxfp4_pair_screen(g, args):
    if args.shipped_prefill:
        return shipped_stream_screen(g, args)
    """One warmed L2-cold interleaved session with identical rank weights."""
    from dataclasses import dataclass
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows

    @dataclass(frozen=True)
    class Exact(Fp8MoeGeometry):
        rank_width: int = 32
        @property
        def slice(self):
            return self.rank_width

    if g.weights != "mxfp4":
        raise ValueError("paired screen requires an MXFP4 geometry")
    if args.mxfp4_tilings:
        mxfp4_tiling_screen(g, args, Exact)
        return
    results = []
    properties = torch.cuda.get_device_properties(0)
    l2_bytes = int(properties.L2_cache_size)
    flush = torch.zeros(max(256 << 20, 2 * l2_bytes), dtype=torch.uint8, device="cuda")
    topk = 6 if args.native_slices else 8
    for width in paired_widths(g.intermediate, args.widths, args.engine_policy_root):
        # Keep both arms' resident payloads beyond L2 on the actual card.
        expert_payload_bytes = 3 * width * g.hidden // 2
        experts = max(16, (2 * l2_bytes // (16 * expert_payload_bytes) + 1) * 16)
        padded = (width + 127) // 128 * 128
        gen = torch.Generator().manual_seed(args.seed + width)
        weights, scales = {}, {}
        for name, shape in (("w1", (experts, width, g.hidden // 2)),
                            ("w3", (experts, width, g.hidden // 2)),
                            ("w2", (experts, g.hidden, width // 2))):
            weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
            scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                        dtype=torch.uint8, generator=gen)
        weight_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in (*weights.values(), *scales.values()))).hexdigest()
        for rows in map(int, args.rows.split(",")):
            torch.manual_seed(args.seed + rows)
            x = (torch.randn(rows, g.hidden, device="cuda") * .5).bfloat16()
            ids = torch.rand(rows, experts, device="cuda").topk(topk, -1).indices.int()
            if args.routing == "hot":
                ids.copy_(ids[:1].expand_as(ids))
            routing = torch.rand(rows, topk, device="cuda")
            routing /= routing.sum(-1, keepdim=True)
            runs, outputs, keepalive = {}, {}, []
            route_names = ["native"] if args.native_slices else args.route.split(",")
            for route in route_names:
                for exact in (False, True):
                    label = route + ("-exact" if exact else "-padded")
                    if args.native_slices:
                        slice_width = native_slice_width(g.name, width)
                        run, output, owned = native_pair_arm(x, ids, routing, weights, scales,
                                                            slice_width, exact, g.name != "v41" and rows >= 256,
                                                            g.name == "v41")
                    else:
                        stored = width if exact else padded
                        geometry = Exact(g.name, g.hidden, experts, topk, g.intermediate,
                                         weights="mxfp4", rank_width=stored)
                        pad = torch.nn.functional.pad
                        s2 = pad(scales["w2"], (0, (stored // 32 + 3) // 4 * 4 - width // 32)).cuda()
                        planes = tuple(pad(t, (0, 0, 0, stored - width)).cuda()
                                       for t in (weights["w1"], scales["w1"], weights["w3"], scales["w3"]))
                        planes += (pad(weights["w2"], (0, (stored - width) // 2)).cuda(), s2)
                        source, _ = wire_rows(x)
                        program = compile_fp8_moe_aot(geometry, route=route, max_rows=rows, prefill=args.prefill)
                        scratch = torch.empty(fp8_moe_scratch_bytes(geometry, route, rows, prefill=args.prefill),
                                              dtype=torch.uint8, device="cuda")
                        output = torch.empty_like(x)
                        def bind(program, source, planes, output, scratch):
                            return lambda: program.launch(source, ids, routing, *planes, output, scratch, scalars=(rows,))
                        run = bind(program, source, planes, output, scratch)
                        owned = (program, source, planes, scratch)
                    run()
                    torch.cuda.synchronize()
                    if args.graph:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        keepalive.append((run, graph))
                        run = graph.replay
                    runs[label], outputs[label] = run, output
                    keepalive.append(owned)
            for run in runs.values():
                run()
            torch.cuda.synchronize()
            comparisons = {route: compare_route_outputs({arm: outputs[route + "-" + arm]
                           for arm in ("padded", "exact")})["exact"] for route in route_names}
            for output in outputs.values():
                assert torch.isfinite(output).all() and torch.count_nonzero(output), "invalid paired output"
            samples = time_interleaved_samples(runs, args.iters, flush, lambda _: None)
            for route in route_names:
                old, new = (statistics.median(samples[route + "-" + arm]) for arm in ("padded", "exact"))
                record = dict(geometry=g.name, hidden=g.hidden, experts=experts, topk=topk,
                              width=width, padded_width=padded, rows=rows, route=route,
                              weight_sha256=weight_hash, graph=args.graph, routing=args.routing,
                              l2_bytes=l2_bytes, resident_payload_bytes=experts * expert_payload_bytes,
                              slice_width=native_slice_width(g.name, width) if args.native_slices else None,
                              padded_us=old, exact_us=new, speedup=old/new,
                              comparison=comparisons[route], samples_us={arm: samples[route + "-" + arm]
                              for arm in ("padded", "exact")})
                results.append(record)
                print(json.dumps(record), flush=True)
            del runs, outputs, keepalive
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(device=str(torch.cuda.get_device_properties(0)),
                                               torch=torch.__version__, results=results), indent=2) + "\n")


def mxfp4_tiling_screen(g, args, geometry_type):
    """One development session: choose only quality/invariant-passing exact arms."""
    from b12x.integration.cuteafd.fp8_moe import fp8_moe_scratch_bytes
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows

    properties = torch.cuda.get_device_properties(0)
    l2_bytes = int(properties.L2_cache_size)
    flush = torch.zeros(max(256 << 20, 2 * l2_bytes), dtype=torch.uint8, device="cuda")
    explicit = native_screen_arms(args.mxfp4_arm)
    widths = sorted({arm[1] for arm in explicit}) if explicit else paired_widths(g.intermediate, args.widths, args.engine_policy_root)
    if any(width > g.intermediate for width in widths):
        raise ValueError("logical rank width exceeds the model intermediate")
    waves = recorded_waves(args.routes, g) if args.routes else None
    selected_waves = {rows: screen_route_wave(waves, rows, args.seed)
                      for rows in map(int, args.rows.split(","))} if waves else {}
    route_fixture = ({"path": str(args.routes), "sha256": hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                      "waves": len(waves), "family_attested": args.routes.suffix != ".jsonl"}
                     if waves else None)
    results = []
    for width in widths:
        expert_bytes = 3 * width * g.hidden // 2
        experts = max(16, (2 * l2_bytes // (16 * expert_bytes) + 1) * 16,
                      g.experts if waves else 0)
        padded = (width + 127) // 128 * 128
        generator = torch.Generator().manual_seed(args.seed + width)
        weights, scales = {}, {}
        for name, shape in (("w1", (experts, width, g.hidden // 2)),
                            ("w3", (experts, width, g.hidden // 2)),
                            ("w2", (experts, g.hidden, width // 2))):
            weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
            scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                        dtype=torch.uint8, generator=generator)
        weight_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in (*weights.values(), *scales.values()))).hexdigest()
        for rows in map(int, args.rows.split(",")):
            torch.manual_seed(args.seed + rows)
            x = (torch.randn(rows, g.hidden, device="cuda") * .5).bfloat16()
            topk = g.top_k
            ids = (screen_route_ids(selected_waves[rows], experts, topk, args.seed).cuda()
                   if waves else torch.rand(rows, experts, device="cuda").topk(topk, -1).indices.int())
            if args.routing == "hot":
                ids.copy_(ids[:1].expand_as(ids))
            routing = torch.rand(rows, topk, device="cuda")
            routing /= routing.sum(-1, keepdim=True)
            oracle = mxfp4_f32_reference(x, ids, routing, weights, scales,
                                         10.0 if args.native_slices else g.swiglu_limit,
                                         native=args.native_slices)
            for route in (["native"] if args.native_slices else args.route.split(",")):
                blocks = (width + 127) // 128
                matching_warps = 4 if blocks % 4 == 0 else max(d for d in range(1, 9) if blocks % d == 0)
                baseline_tile = (native_slice_width(g.name, width) if args.native_slices else
                                 matching_warps if route == "decode" else None)
                tiles = (64, 128, 192) if args.native_slices else (mxfp4_decode_partitions(width) if route == "decode" else (None,))
                specs, controls = tiling_specs(args, width, baseline_tile, tiles)
                runs, outputs, keepalive, invariants, resident_widths = {}, {}, [], {}, {}
                for label, exact, tile, stored in specs:
                    try:
                        if args.native_slices:
                            run, output, owned = native_pair_arm(x, ids, routing, weights, scales, tile,
                                                                exact, False, g.name == "v41", qualify=True,
                                                                aligned_w2=label.startswith("aligned-w2"),
                                                                aligned_w2_scales=label == "aligned-w2-scales",
                                                                stored_width=stored)
                        else:
                            stored = width if exact else padded
                            geometry = geometry_type(g.name, g.hidden, experts, topk, g.intermediate,
                                                     weights="mxfp4", rank_width=stored)
                            pad = torch.nn.functional.pad
                            s2 = pad(scales["w2"], (0, (stored // 32 + 3) // 4 * 4 - width // 32)).cuda()
                            planes = tuple(pad(t, (0, 0, 0, stored - width)).cuda()
                                           for t in (weights["w1"], scales["w1"], weights["w3"], scales["w3"]))
                            planes += (pad(weights["w2"], (0, (stored - width) // 2)).cuda(), s2)
                            source, _ = wire_rows(x)
                            program = grouped_pair_program(geometry, route, rows, args.prefill,
                                                           down_warps=tile if exact else None)
                            scratch = torch.empty(fp8_moe_scratch_bytes(geometry, route, rows, prefill=args.prefill),
                                                  dtype=torch.uint8, device="cuda")
                            output = torch.empty_like(x)
                            buffers = (source, ids, routing, *planes, output, scratch)
                            def bind(program, buffers):
                                return lambda live: program.launch(*buffers, scalars=(live,))
                            execute = bind(program, buffers)
                            qualify_pair_replay(execute, ids, output, [scratch], buffers,
                                                lambda live: output[:live])
                            def bind_run(execute, rows):
                                return lambda: execute(rows)
                            run = bind_run(execute, rows)
                            owned = (program, buffers)
                        invariants[label] = True
                    except AssertionError as error:
                        print(json.dumps(dict(geometry=g.name, width=width, rows=rows, route=route,
                                              arm=label, invariant_failure=str(error))), flush=True)
                        if label in controls:
                            raise
                        continue
                    resident_widths[label] = stored if stored is not None else (width if exact else padded)
                    run()
                    torch.cuda.synchronize()
                    if args.graph:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        keepalive.append((run, graph))
                        run = graph.replay
                    runs[label], outputs[label] = run, output
                    keepalive.append(owned)
                for run in runs.values():
                    run()
                torch.cuda.synchronize()
                errors = {label: f32_errors(output, oracle) for label, output in outputs.items()}
                baseline_error = errors["padded"]
                eligible = {label: all(error[key] <= baseline_error[key] for key in ("max_abs", "rms"))
                            for label, error in errors.items()}
                for output in outputs.values():
                    assert torch.isfinite(output).all() and torch.count_nonzero(output), "invalid tiling output"
                samples = time_interleaved_samples(runs, args.iters, flush, lambda _: None)
                comparisons = compare_route_outputs(outputs)
                assert all(comparisons[label]["bitwise_equal"] for label in controls), "identical controls differ"
                wave = selected_waves.get(rows)
                record = dict(geometry=g.name, hidden=g.hidden, width=width, padded_width=padded,
                              rows=rows, route=route, experts=experts, topk=topk, graph=args.graph,
                              routing=args.routing, weight_sha256=weight_hash, l2_bytes=l2_bytes,
                              route_fixture=route_fixture,
                              recorded_wave=({key: wave.get(key) for key in ("layer", "lane", "rows")}
                                             if wave else None),
                              route_histogram=sorted(Counter(ids.cpu().flatten().tolist()).items()),
                              resident_payload_bytes=experts * expert_bytes, samples_us=samples,
                              resident_widths=resident_widths,
                              slice_widths={label: tile for label, _, tile, _ in specs if label in runs},
                              controls=controls,
                              resident_payload_bytes_by_arm={label: experts * g.hidden *
                                  (2 * width + padded if label.startswith("aligned-w2") else
                                   3 * resident_widths[label]) // 2 for label in runs},
                              f32_errors=errors, quality_eligible=eligible,
                              within_version_invariants=invariants, comparisons=comparisons,
                              **tiling_timing_summary(samples, eligible, controls))
                results.append(record)
                print(json.dumps(record), flush=True)
                del runs, outputs, keepalive
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(device=str(properties), torch=torch.__version__, results=results), indent=2) + "\n")


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, Fp8MoeGeometry

    if args.native_slices:
        intermediate, hidden = {"v41": (2304, 5120), "dsv4f": (2048, 4096), "dsv4p": (3072, 7168)}[args.geometry]
        g = Fp8MoeGeometry(args.geometry, hidden, 256 if args.geometry == "dsv4f" else 384, 6, intermediate, weights="mxfp4")
    else:
        g = GEOMETRIES[args.geometry].with_tp(args.tp)
    if args.mxfp4_pair:
        mxfp4_pair_screen(g, args)
        return
    if args.activation is not None:
        g = replace(g, activation=args.activation)
    if g.weights in ("nvfp4", "bf16") and args.input != "bf16":
        raise ValueError("NVFP4 and native BF16 require original BF16 input")
    waves = recorded_waves(args.routes, g) if args.routes else None
    route_fixture = ({"path": str(args.routes), "sha256": hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                      "waves": len(waves), "kind": "rescaled recorded histogram; reconstructed token co-occurrence"}
                     if waves else None)
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
    elif g.weights == "mxfp4":
        from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes

        def codes(shape):
            return torch.randint(0, 256, shape, device=dev, generator=gen, dtype=torch.int32).to(torch.uint8)

        def mxfp4_scales(shape):
            return torch.randint(118, 124, shape, device=dev, generator=gen, dtype=torch.int32).to(torch.uint8)

        s2 = torch.zeros(e, h, mxfp4_scale_row_bytes(i), device=dev, dtype=torch.uint8)
        s2[..., :i // 32] = mxfp4_scales((e, h, i // 32))
        w = (codes((e, i, h // 2)), mxfp4_scales((e, i, h // 32)),
             codes((e, i, h // 2)), mxfp4_scales((e, i, h // 32)),
             codes((e, h, i // 2)), s2)
        per_expert = 2 * i * (h // 2 + h // 32) + h * (i // 2 + mxfp4_scale_row_bytes(i))
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
        raise ValueError("this benchmark supports FP8, MXFP4, NVFP4 and native BF16 weights")
    if args.exact_sweep:
        exact_sweep(g, w, args)
        return
    routes = [r for r in args.route.split(",") if r]
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    results = []
    compiled = {}
    for rows in [int(r) for r in args.rows.split(",")]:
        wire_in = args.input == "wire"
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
        unique = []
        route_ids = []
        if waves:
            from benchmarks.benchmark_exlr8_a16 import real_route_ids

            cpu_gen = torch.Generator().manual_seed(args.seed)
            route_ids = [real_route_ids(rows, wave, cpu_gen, e, k).to(device=dev, dtype=torch.int32)
                         for wave in waves]

        def prepare(seed):
            if route_ids:
                ids.copy_(route_ids[seed % len(route_ids)])
            elif args.routing == "hot":
                selected = torch.randperm(e, device=dev, generator=gen)[:k].int()
                ids.copy_(selected.expand(rows, k))
            else:
                ids.copy_(torch.rand(rows, e, device=dev, generator=gen).topk(k, -1).indices.int())
            unique.append(int(ids.unique().numel()))

        prepare(0)
        programs, outs, scratches, runs, fns, graphs, correctness_by_route = {}, {}, {}, {}, {}, {}, {}
        cached_ref = {}
        for route_name in routes:
            program, scratch = compiled_case(g, args, rows, route_name, compiled)
            out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)

            def make_run(prog, out_buf, scratch_buf):
                def _run():
                    prog.launch(source, ids, weights, *w, out_buf, scratch_buf, scalars=(rows,))
                return _run

            run = make_run(program, out, scratch)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            if args.check:
                if g.weights == "nvfp4":
                    ref = cached_ref.get(g.activations)
                    if ref is None:
                        ref = cached_ref[g.activations] = reference(g, x_exact, ids, weights, *w, a4=g.activations == "a4")
                elif g.weights == "mxfp4":
                    from b12x.integration.cuteafd.fp8_moe import MXFP4_STREAM_DOWN_A8, auto_large_rows
                    from tests.integration.test_cuteafd_mxfp4_moe_aot import reference as mxfp4_reference

                    large = auto_large_rows(wire_in, g.kind, args.prefill)
                    streamed = wire_in and (route_name == "stream" or (route_name == "auto" and rows > large))
                    a8 = streamed and MXFP4_STREAM_DOWN_A8 and args.prefill != "w8a16"
                    ref = cached_ref.get(a8)
                    if ref is None:
                        ref = cached_ref[a8] = mxfp4_reference(x_exact, ids, weights, *w[:5], w[5][..., :i // 32],
                                                               mxfp4_down_a8=a8)
                elif g.weights == "bf16":
                    from tests.integration.test_cuteafd_expert_contract import bf16_reference

                    ref = cached_ref.get("bf16")
                    if ref is None:
                        ref = cached_ref["bf16"] = bf16_reference(x_exact, ids, weights, w[0], w[2], w[4],
                                                                  g.swiglu_limit, g.activation)
                else:
                    from tests.integration.test_cuteafd_expert_contract import activation

                    def dequant(wt, s):
                        return (wt.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()

                    ref = cached_ref.get("fp8")
                    if ref is None:
                        acc = torch.zeros(rows, h, device=dev)
                        for expert in ids.unique().tolist():
                            r, slots = torch.where(ids == expert)
                            gate = x_exact[r] @ dequant(w[0][expert], w[1][expert]).T
                            up = x_exact[r] @ dequant(w[2][expert], w[3][expert]).T
                            y = activation(gate, up, g.swiglu_limit, g.activation) @ dequant(w[4][expert], w[5][expert]).T
                            acc.index_add_(0, r, y.float() * weights[r, slots][:, None])
                        ref = cached_ref["fp8"] = acc.bfloat16()
                correctness = oracle_metrics(out, ref, nvfp4_a4=g.weights == "nvfp4" and g.activations == "a4")
                enforce_oracle(correctness, args.continue_on_fail)
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
            programs[route_name], outs[route_name], scratches[route_name] = program, out, scratch
            runs[route_name], fns[route_name], graphs[route_name] = run, fn, graph
            correctness_by_route[route_name] = correctness

        comparisons = None
        if args.compare_routes_exact:
            prepare(0)
            for fn in fns.values():
                fn()
            torch.cuda.synchronize()
            comparisons = compare_route_outputs(outs)

        prefix_checks = {}
        if args.check_prefix_rows is not None and rows > args.check_prefix_rows:
            prepare(0)
            for route_name in routes:
                program, scratch = programs[route_name], scratches[route_name]
                fns[route_name]()
                torch.cuda.synchronize()

                def execute_prefix(count):
                    prefix_out = torch.full((count, h), float("nan"), dtype=torch.bfloat16, device=dev)
                    scratch.fill_(0xA5)
                    def launch():
                        program.launch(source[:count], ids[:count], weights[:count], *w,
                                       prefix_out, scratch, scalars=(count,))
                    launch()
                    if args.graph:
                        prefix_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(prefix_graph):
                            launch()
                        prefix_graph.replay()
                    torch.cuda.synchronize()
                    return prefix_out

                prefix_checks[route_name] = check_shared_prefix(outs[route_name], args.check_prefix_rows, execute_prefix)
                fns[route_name]()
                torch.cuda.synchronize()

        unique.clear()
        samples_by_route = time_interleaved_samples(fns, args.iters, flush, prepare)
        experts = statistics.mean(unique)
        for route_name in routes:
            samples = samples_by_route[route_name]
            us = statistics.median(samples)
            scratch = scratches[route_name]
            run = runs[route_name]
            record = {"geometry": g.name, "tp": g.tp, "input": args.input, "arithmetic": g.activations,
                      "activation": g.activation, "projection": g.projection, "weight_format": g.weights,
                      "route": route_name, "routing": "recorded" if waves else args.routing, "graph": args.graph, "rows": rows,
                      "capacity": args.capacity if args.capacity is not None else rows,
                      "route_fixture": route_fixture, "us_per_row": us / rows,
                      "us": us, "samples_us": samples, "scratch_bytes": scratch.numel(),
                      "experts": experts, "gbps": experts * per_expert / us / 1e3,
                      "tflops": rows * k * 3 * 2 * i * h / us / 1e6, "correctness": correctness_by_route[route_name],
                      "timing_eligible": correctness_by_route[route_name].get("passed", False)}
            if comparisons is not None:
                record["route_comparison"] = comparisons[route_name]
            if route_name in prefix_checks:
                record["prefix_comparison"] = prefix_checks[route_name]
                record["timing_eligible"] &= prefix_checks[route_name]["bitwise_equal"]
            if args.profile_dir:
                args.profile_dir.mkdir(parents=True, exist_ok=True)
                flush.max()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run()
                    torch.cuda.synchronize()
                suffix = f"-{route_name}" if len(routes) > 1 else ""
                trace = args.profile_dir / f"{g.name}-tp{g.tp}{suffix}-{args.routing}-r{rows}.json"
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
        del graphs, fns, runs, programs, outs, scratches
        if args.capacity is None:
            compiled.clear()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        args.output.write_text(json.dumps({"device": str(props), "torch": torch.__version__,
                                          "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
