"""Compare native slice widths on identical encoded inputs and weights.

Includes ordered CuTe FP32 slice reduction and GPU oracle checks before timing.
For the exact Spark TP4 workload, pass --exact-storage --live-groups --experts 384
--routing-case unique --rows 2 3 4 5 6 7 16 32 48 54 56 64. Distinct route IDs
(up to the expert count) expose sparse-wave occupancy rather than repeated experts.
This is a synchronous-kernel diagnostic, not a serving benchmark.
Run with the repository root on PYTHONPATH; the tests package supplies fixtures.
"""

import argparse
import json
import statistics
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from tests.moe.test_v41_grouped_slices import _check_grouped_slices
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes, nvidia_smi_gpu_mode_snapshot
from benchmarks.dense_autotune_common import bench_events
import cuda.bindings.driver as cuda

PRESETS = {"v41": (5120, 576, 6), "dsv4f": (4096, 512, 6)}


class OrderedReduce:
    def __init__(self, slices, hidden):
        self.slices = slices
        self.hidden = hidden

    @cute.jit
    def __call__(self, source: cute.Tensor, dest: cute.Tensor,
                 routes: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(source, dest, routes).launch(
            grid=((routes * self.hidden + 255) // 256, 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, dest: cute.Tensor, routes: cutlass.Int32):
        index = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cutlass.Int64(cute.arch.thread_idx()[0])
        row, col = index // self.hidden, index % self.hidden
        if row < routes:
            value = cutlass.Float32(0)
            for plane in cutlass.range_constexpr(self.slices):
                value += source[plane, row, col]
            dest[row, col] = value


def measure_factory(widths, iterations, replays, fc2_n_splits=(1,), *, cache="warm",
                    l2_flush_bytes=0, exact_storage=False, live_groups=False,
                    components=("compute", "reduce", "combined")):
    state = {}

    def measure(**case):
        args, meta = case["args"], case["meta_view"]
        m, routes = case["rows"], case["routes"]
        h, n, topk, capacity = (case[k] for k in ("hidden", "intermediate", "topk", "capacity"))
        launch_groups = case["group_capacity"]
        if live_groups:
            # Match the pipeline's runtime bound, not the number of nonempty groups.
            launch_groups = max(1, min(m * topk, case["experts"] +
                                      max(m * topk - case["experts"], 0) // 16))
        if not state:
            for width, splits in [(w, s) for w in widths for s in fc2_n_splits]:
                out = torch.empty(((n + width - 1) // width, capacity * topk, h), device="cuda")
                alt_args = [*args[:-1], from_dlpack(out, assumed_align=16)]
                compiled = cute.compile(
                    V41FusedSliceKernel(width, grouped=True, intermediate=n, hidden=h,
                                        fc2_n_splits=splits, exact_storage=exact_storage),
                    *alt_args, cutlass.Int32(capacity), current_cuda_stream(), meta,
                    cutlass.Int32(launch_groups))
                reduced = torch.empty((capacity * topk, h), device="cuda")
                rv = from_dlpack(reduced, assumed_align=16)
                reducer = cute.compile(OrderedReduce(out.shape[0], h), alt_args[-1], rv,
                                       cutlass.Int32(capacity * topk), current_cuda_stream())
                state[width, splits] = (out, alt_args, compiled, reduced, rv, reducer)
        graphs = {}
        baseline = {}
        for (width, splits), (out, alt_args, compiled, reduced, rv, reducer) in state.items():
            out.fill_(12345)
            compiled(*alt_args, m, current_cuda_stream(), meta, launch_groups)
            reducer(alt_args[-1], rv, routes, current_cuda_stream())
            expected_reduced = torch.zeros_like(reduced[:routes])
            for plane in range(out.shape[0]):
                expected_reduced.add_(out[plane, :routes])
            assert torch.equal(reduced[:routes], expected_reduced)
            if width in baseline:
                assert torch.equal(out[:, :routes].view(torch.int32), baseline[width].view(torch.int32)), (width, splits, "split changed FP32 planes")
            else:
                baseline[width] = out[:, :routes].clone()
            actual = torch.zeros_like(case["expected"])
            actual.index_add_(0, case["pair_gpu"][:, 0], reduced[:routes].bfloat16().float())
            rel = ((actual - case["expected"]).norm() / case["expected"].norm()).item()
            assert torch.isfinite(actual).all() and actual.norm() > 0 and rel < 0.01
            assert bool((out[:, routes:] == 12345).all())
            for component in components:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if component != "reduce":
                        compiled(*alt_args, m, current_cuda_stream(), meta, launch_groups)
                    if component != "compute":
                        reducer(alt_args[-1], rv, routes, current_cuda_stream())
                before = torch.cuda.memory_allocated()
                graph.replay()
                assert before == torch.cuda.memory_allocated()
                graphs[width, splits, component] = graph
        for _ in range(5):
            for graph in graphs.values():
                graph.replay()
        cache_modes = ("warm", "cold") if cache == "both" else (cache,)
        flush = make_l2_flush_fn("cold" in cache_modes, l2_flush_bytes)
        for cache_mode in cache_modes:
            samples = {key: [] for key in graphs}
            raw = {key: [] for key in graphs}
            gpu = [nvidia_smi_gpu_mode_snapshot()]
            for iteration in range(iterations):
                order = list(graphs)
                shift = iteration % len(order)
                order = order[shift:] + order[:shift]
                if iteration % 2:
                    order.reverse()
                for key in order:
                    if cache_mode == "cold":
                        # Evict before every replay, outside its measured event interval.
                        times = [value * 1000 for value in bench_events(
                            graphs[key].replay, warmup=2, iters=replays, l2_flush=flush)]
                        raw[key].append(times)
                        samples[key].append(statistics.median(times))
                    else:
                        for _ in range(5):
                            graphs[key].replay()
                        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        start.record()
                        for _ in range(replays):
                            graphs[key].replay()
                        end.record()
                        end.synchronize()
                        samples[key].append(start.elapsed_time(end) * 1000 / replays)
            gpu.append(nvidia_smi_gpu_mode_snapshot())
            print("TIMING " + json.dumps(dict(
                case=case["case"], rows=m, hidden=h, intermediate=n, capacity=capacity,
                groups=case["groups"], launch_group_capacity=launch_groups, routes=routes,
                cache=cache_mode, l2_flush_bytes=resolve_l2_flush_bytes(l2_flush_bytes) if cache_mode == "cold" else 0,
                gpu=gpu, graph_us={f"w{w}/n{s}/{c}": times for (w, s, c), times in samples.items()},
                replay_us={f"w{w}/n{s}/{c}": times for (w, s, c), times in raw.items()},
                median_us={f"w{w}/n{s}/{c}": statistics.median(times) for (w, s, c), times in samples.items()},
                scope="fused compute and ordered CuTe FP32 fold; excludes route planning, encoding and transport")), flush=True)
        for graph in graphs.values():
            graph.reset()

    return measure


def main():
    import hashlib
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=PRESETS, default="v41")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--exact-storage", action="store_true")
    parser.add_argument("--live-groups", action="store_true")
    parser.add_argument("--experts", type=int)
    parser.add_argument("--routing-case", choices=("shared", "unique"), default="shared")
    parser.add_argument("--components", nargs="+", choices=("compute", "reduce", "combined"),
                        default=["compute", "reduce", "combined"])
    parser.add_argument("--capacity", type=int, default=80)
    parser.add_argument("--rows", type=int, nargs="+")
    parser.add_argument("--widths", type=int, nargs="+")
    parser.add_argument("--fc2-n-splits", type=int, nargs="+", default=[1])
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cache", choices=("warm", "cold", "both"), default="warm")
    parser.add_argument("--l2-flush-bytes", type=int, default=0, help="0 uses twice the reported L2 size")
    options = parser.parse_args()
    h, n, topk = PRESETS[options.geometry]
    h, n, topk = options.hidden or h, options.intermediate or n, options.topk or topk
    widths = options.widths or ([64, 128] if options.geometry == "dsv4f" else [64, 128, 192])
    if options.capacity < 1 or options.iterations < 1 or options.replays < 1 or not widths \
            or len(set(widths)) != len(widths) or any(w not in (64, 128, 192) or -(-n // w) * w > -(-n // 128) * 128 for w in widths):
        parser.error("invalid capacity, timing counts, or slice widths")
    if any(s < 1 or (h // 128) % s for s in options.fc2_n_splits):
        parser.error("FC2 N splits must divide hidden tile count")
    if options.rows and any(m < 1 or m > options.capacity for m in options.rows):
        parser.error("rows must fit capacity")
    if options.experts is not None and options.experts < topk:
        parser.error("expert count must be at least topk")
    if options.routing_case != "shared" and not options.rows:
        parser.error("unique routing requires explicit row counts")
    if options.exact_storage and (h, n) != (5120, 576):
        parser.error("exact storage requires V4.1 TP4 geometry")
    if options.l2_flush_bytes < 0:
        parser.error("L2 flush bytes must be non-negative")
    print("SOURCE " + json.dumps(dict(
        geometry=options.geometry, hidden=h, intermediate=n, topk=topk,
        capacity=options.capacity, widths=widths, fc2_n_splits=options.fc2_n_splits,
        exact_storage=options.exact_storage, live_groups=options.live_groups,
        experts=options.experts, routing_case=options.routing_case,
        cache=options.cache, l2_flush_bytes=options.l2_flush_bytes,
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        sha256={name: hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest()
                for name, obj in [("kernel", V41FusedSliceKernel), ("fixture", _check_grouped_slices),
                                  ("benchmark", measure_factory)]})), flush=True)
    _check_grouped_slices(widths[0], measure_factory(widths, options.iterations, options.replays,
        options.fc2_n_splits, cache=options.cache, l2_flush_bytes=options.l2_flush_bytes,
        exact_storage=options.exact_storage, live_groups=options.live_groups, components=options.components),
        n=n, topk=topk, h=h, capacity=options.capacity, row_counts=options.rows,
        experts=options.experts, exact_storage=options.exact_storage, routing_case=options.routing_case)


if __name__ == "__main__":
    main()
