"""Compensated BF16 projection retiles for the Step-5 coordinator shapes.

The numerics contract is the canonical ordered compensated accumulation of
``Bf16PrefillKernel`` (``_prefill.py``): per output, every K16 tensor-core
segment is added to the FP32 accumulator with the two-term Kahan update, in
ascending K.  Every arm below is the same kernel with a different
(warp_layout, tile_m, tile_n, num_stages, unroll_k) choice; because the
K16 segment order and the compensated add are unchanged, each arm's output must
be bitwise identical to the constructor-default kernel, which is checked here
at every live-row count before anything is timed.  An arm that is not bitwise
identical, or that fails, is dropped (a result, not a crash).

Shapes are the Step-5 coordinator projections from the #703 census appendix
(K = input width, N = output features) plus the Kimi core projection as a
control.  Rows cover speculative/verify widths (1-16), the 59/64-row short
projection, 41-row prefill tails and 256-row prefill chunks.

Timing: separate CUDA-graph replays with an L2 eviction sweep before each cold
replay (the serving regime: no per-launch L2 reuse) and a warm pass; arms run
in interleaved blocks in both directions and medians are reported.

Run under the hardware fair queue, GPU0 only, short (``--quick`` for a smoke):
    python3 benchmarks/benchmark_step5_compensated_projection.py \\
        --output DIR/result.json [--quick]
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--repo", default="")
parser.add_argument("--rows", default="1,16,41,59,64,256")
parser.add_argument("--shapes", default="")
parser.add_argument("--arms", default="")
parser.add_argument("--replays", type=int, default=12)
parser.add_argument("--blocks", type=int, default=2)
parser.add_argument("--quick", action="store_true",
                    help="one shape, two arms, short replays (compile smoke)")
parser.add_argument("--compile-only", action="store_true",
                    help="build (and disk-cache) every program, no launches")
args = parser.parse_args()
if args.quick:
    args.rows, args.shapes, args.arms = "16,256", "q", "canon,serve,n64s4"
    args.replays, args.blocks = 6, 1

repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
sys.path.insert(0, str(repo))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.utils as cutlass_utils  # noqa: E402
import torch  # noqa: E402

from benchmarks.common import (  # noqa: E402
    bench_cuda_graph, capture_cuda_graph, make_l2_flush_fn)
from b12x.integration.cuteafd import _common  # noqa: E402
from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel  # noqa: E402
from b12x.gemm.bf16_gemv._skinny import (  # noqa: E402
    RowTiledBf16Projection, TmaBf16Projection)

ROWS = [int(v) for v in args.rows.split(",") if v]

# (name, K = input width, N = output features, output dtype)
SHAPES = {
    "q": (4096, 6144, "bfloat16"),
    "csa_o": (12288, 4096, "bfloat16"),
    "swa_o": (6144, 4096, "float32"),
    "kv": (4096, 384, "bfloat16"),
    "gate_up": (4096, 3072, "bfloat16"),
    "kimi_up": (7168, 24752, "bfloat16"),  # control
}
if args.shapes:
    keep = set(args.shapes.split(","))
    SHAPES = {k: v for k, v in SHAPES.items() if k in keep}

# kind: "direct" builds Bf16PrefillKernel from these exact kwargs;
#       "tma" builds TmaBf16Projection (which merges prefill_config).
# Donor geometries are the measured DECODE_TMA_TILES row-tiles from
# _skinny.py, re-expressed for the compensated kernel: a 16-row M tile with
# 4 warps splitting N and a deep pipeline keeps the three FP32 accumulator
# arrays (acc/segment/correction) small enough to hide the Kahan ALU latency.
ARMS = {
    # The constructor-default canonical kernel: the bitwise reference.
    "canon": dict(kind="direct", config={}),
    # The compensated serving tile (prefill_config merged): timing baseline.
    "serve": dict(kind="tma", config=dict(compensated=True)),
    "d16n32s12": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 4),
                                             tile_m=16, tile_n=32, num_stages=12)),
    "d16n32s16": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 4),
                                             tile_m=16, tile_n=32, num_stages=16)),
    "d16n64s9": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 4),
                                            tile_m=16, tile_n=64, num_stages=9)),
    "d16n16s8": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 2),
                                            tile_m=16, tile_n=16, num_stages=8)),
    "d16n8s12": dict(kind="tma", config=dict(compensated=True, compute_warps=1,
                                            tile_n=8, num_stages=12)),
    "n64s4": dict(kind="tma", config=dict(compensated=True, compute_warps=4,
                                         tile_n=64, num_stages=4)),
    "n32s8": dict(kind="tma", config=dict(compensated=True, compute_warps=4,
                                         tile_n=32, num_stages=8)),
    "w8n128s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                            tile_m=64, tile_n=128, num_stages=3)),
    "w16n128s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 4),
                                             tile_m=64, tile_n=128, num_stages=3)),
    "w8n64s4": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                           tile_m=64, tile_n=64, num_stages=4)),
    # Wide-N / deep-stage streams: the small-M regime is weight-bandwidth
    # bound, so these raise the CTA count and the bytes in flight per SM.
    "d16n128s5": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 4),
                                             tile_m=16, tile_n=128, num_stages=5)),
    "w8n64s5": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 8),
                                           tile_m=16, tile_n=64, num_stages=5)),
    "w8n128s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(1, 8),
                                            tile_m=16, tile_n=128, num_stages=2)),
    # Eight compute warps (two per SM partition) for the compute-bound 256-row
    # chunks: MMA issue and the compensated ALU can then overlap across warps.
    "cw8n64s3": dict(kind="tma", config=dict(compensated=True, compute_warps=8,
                                            tile_n=64, num_stages=3)),
    "w8m64n64s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                              tile_m=64, tile_n=64, num_stages=3)),
    "unroll": dict(kind="tma", config=dict(compensated=True, unroll_k=True)),
    # Round 2: the compute-bound width.  Two or four compute warps per SM
    # partition let the warp MMA issue overlap the compensated ALU of the
    # other warp instead of serializing inside one in-order warp.
    "w16n128s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 4),
                                             tile_m=64, tile_n=128, num_stages=2)),
    "w16n64s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 4),
                                            tile_m=64, tile_n=64, num_stages=3)),
    "w8m64n128s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                               tile_m=64, tile_n=128, num_stages=2)),
    "w8m32n128s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                               tile_m=32, tile_n=128, num_stages=3)),
    "w8m32n64s5": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                              tile_m=32, tile_n=64, num_stages=5)),
    # Round 3: larger tiles cut the duplicated operand traffic that limits the
    # wide-row cells (B is re-read once per tile_m rows, A once per tile_n).
    "w8m128n128s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 2),
                                               tile_m=128, tile_n=128, num_stages=3)),
    "w8m128n128s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 2),
                                               tile_m=128, tile_n=128, num_stages=2)),
    "w8m128n64s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(4, 2),
                                              tile_m=128, tile_n=64, num_stages=3)),
    "w8m64n256s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(2, 4),
                                              tile_m=64, tile_n=256, num_stages=2)),
    "w16m128n128s2": dict(kind="tma", config=dict(compensated=True, warp_layout=(8, 2),
                                                tile_m=128, tile_n=128, num_stages=2)),
    "w16m128n128s3": dict(kind="tma", config=dict(compensated=True, warp_layout=(8, 2),
                                                tile_m=128, tile_n=128, num_stages=3)),
    # The row table measured here as a single program: the small-M tiles below
    # the row-bound widths and the eight-warp tiles above them (see ROWTILES).
    "rowtile": dict(kind="rowtile", config={}),
}
if args.arms:
    keep = set(args.arms.split(","))
    ARMS = {k: v for k, v in ARMS.items() if k in keep}
assert "canon" in ARMS, "the bitwise reference arm is required"


class DirectProjection(TmaBf16Projection):
    """``Bf16PrefillKernel`` with exactly these kwargs (no prefill merge)."""

    def __init__(self, n, k, *, out_dtype=cutlass.BFloat16, **config):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.row_tiled = None
        self.tiles_by_rows = ()
        self.config = dict(config)
        self.kernel = Bf16PrefillKernel(self.n, self.k, **self.config)

    def key(self):
        return (self.n, self.k, tuple(sorted(self.config.items())))


class CompensatedRowTiled(RowTiledBf16Projection):
    """``RowTiledBf16Projection`` whose every entry stays compensated.

    The upstream row table carries ``prefill_config``'s ``compensated=False``;
    this duel may not, so the entries are rebuilt from the same table with the
    canonical compensated kernel.  The inherited ``__call__`` branches on the
    ``rows`` launch scalar, so one program covers decode and prefill widths.
    """

    def __init__(self, n, k, table, *, out_dtype=cutlass.BFloat16):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.table = tuple((int(limit), dict(config)) for limit, config in table)
        self.tiles_by_rows = tuple((int(limit),) for limit, _ in table)
        self.limits = tuple(int(limit) for limit, _ in table)
        self.floors = (0,) + self.limits[:-1]
        self.tiled = [DirectProjection(n, k, out_dtype=out_dtype,
                                       compensated=True, **config)
                      for _, config in table]
        self.default = DirectProjection(n, k, out_dtype=out_dtype,
                                        compensated=True, compute_warps=4,
                                        tile_n=64, num_stages=4)
        self.kernel = self.default.kernel
        self.row_tiled = None
        self.config = dict(compensated=True)
        for launch in [*self.tiled, self.default]:
            assert launch.kernel.compensated is True
        capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_120")
        for entry, launch in zip(self.tiles_by_rows, self.tiled):
            kernel = launch.kernel
            smem = kernel.buffer_align_bytes + (
                kernel.tile_m + kernel.tile_n) * kernel.tile_k * 2 * kernel.num_stages
            if smem > capacity:
                raise ValueError(f"rowtile entry {entry} needs {smem} B, "
                                 f"capacity {capacity}")

    def key(self):
        return (self.n, self.k, self.table)


# Per-shape row tables from the round-1 sweep: the winning tile for each live
# row band, in the same (rows, config) form RowTiledBf16Projection uses.
ROWTILES = {
    "q": [
        (16, dict(warp_layout=(1, 4), tile_m=16, tile_n=128, num_stages=5)),
        (64, dict(compute_warps=4, tile_n=64, num_stages=4)),
        (256, dict(warp_layout=(2, 4), tile_m=64, tile_n=64, num_stages=4)),
        (4096, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=2)),
    ],
    "csa_o": [
        (16, dict(warp_layout=(1, 4), tile_m=16, tile_n=32, num_stages=12)),
        (64, dict(warp_layout=(2, 4), tile_m=32, tile_n=64, num_stages=5)),
        (4096, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=2)),
    ],
    "swa_o": [
        (16, dict(warp_layout=(1, 4), tile_m=16, tile_n=32, num_stages=12)),
        (64, dict(warp_layout=(2, 4), tile_m=32, tile_n=64, num_stages=5)),
        (4096, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=2)),
    ],
    "kv": [
        (64, dict(warp_layout=(1, 2), tile_m=16, tile_n=16, num_stages=8)),
        (256, dict(warp_layout=(1, 4), tile_m=16, tile_n=64, num_stages=9)),
        (4096, dict(compute_warps=4, tile_n=64, num_stages=4)),
    ],
    "gate_up": [
        (16, dict(warp_layout=(1, 4), tile_m=16, tile_n=32, num_stages=12)),
        (64, dict(warp_layout=(2, 4), tile_m=32, tile_n=64, num_stages=5)),
        (256, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=3)),
        (4096, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=2)),
    ],
    "kimi_up": [
        (16, dict(warp_layout=(1, 4), tile_m=16, tile_n=32, num_stages=12)),
        (64, dict(compute_warps=4, tile_n=64, num_stages=4)),
        (4096, dict(warp_layout=(2, 4), tile_m=64, tile_n=128, num_stages=2)),
    ],
}


def torch_dtype(dtype):
    return torch.float32 if dtype == "float32" else torch.bfloat16


def cute_dtype(dtype):
    return cutlass.Float32 if dtype == "float32" else cutlass.BFloat16


def operations(width, outputs, dtype):
    return (
        _common.Operand("x", torch.bfloat16, f"[rows,{width}]"),
        _common.Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
        _common.Operand("out", torch_dtype(dtype), f"[rows,{outputs}]", "out"),
    ), (_common.Scalar("rows"),)


def build(arm, width, outputs, dtype, label, shape_name):
    spec = ARMS[arm]
    out_dtype = torch_dtype(dtype)
    if spec["kind"] == "rowtile":
        table = ROWTILES.get(shape_name)
        if not table:
            raise ValueError(f"{shape_name}: no row table")
        launch = CompensatedRowTiled(outputs, width, table,
                                     out_dtype=cute_dtype(dtype))
    elif spec["kind"] == "direct":
        launch = DirectProjection(outputs, width, out_dtype=cute_dtype(dtype),
                                  **spec["config"])
    else:
        launch = TmaBf16Projection(outputs, width, out_dtype=cute_dtype(dtype),
                                   tiles_by_rows=(), **spec["config"])
    kernel = launch.kernel
    assert kernel.compensated is True, f"{label}: arm must stay compensated"
    smem = kernel.buffer_align_bytes + (
        kernel.tile_m + kernel.tile_n) * kernel.tile_k * 2 * kernel.num_stages
    capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_120")
    if smem > capacity:
        raise ValueError(f"{label}: needs {smem} B smem, capacity {capacity}")
    operands, scalars = operations(width, outputs, dtype)
    program = _common.compile_program(
        launch, name=f"bf16_comp_projection_{shape_name}_{label}",
        operands=operands, scalars=scalars,
        key=(width, outputs, dtype, launch.key()),
        geometry=dict(width=width, outputs=outputs, output_dtype=dtype,
                      arm=label, config=dict(spec["config"])))
    return program, dict(
        warp_layout=list(getattr(kernel, "warp_layout", None) or ()),
        tile_m=kernel.tile_m, tile_n=kernel.tile_n, tile_k=kernel.tile_k,
        num_stages=kernel.num_stages, num_threads=kernel.num_threads,
        smem_bytes=smem, unroll_k=kernel.unroll_k,
        grid_n_tiles=(outputs + kernel.tile_n - 1) // kernel.tile_n)


def bit_view(tensor):
    return tensor.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


def same_bits(a, b):
    return torch.equal(bit_view(a), bit_view(b))


def region_bits(tensor, row):
    return bit_view(tensor[row].clone())


def roofline(width, outputs, dtype, rows):
    """Weight-streaming (one read of every weight byte) fraction of HBM peak."""
    dtype_bytes = 4 if dtype == "float32" else 2
    bytes = outputs * width * 2 + rows * width * 2 + rows * outputs * dtype_bytes
    return bytes


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(274)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    result = dict(
        device=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        sm_count=properties.multi_processor_count,
        l2_bytes=int(getattr(properties, "L2_cache_size", 0)),
        smem_capacity=cutlass_utils.get_smem_capacity_in_bytes("sm_120"),
        mode="compensated", rows=ROWS, replays=args.replays, blocks=args.blocks,
        shapes={}, errors={})
    flush = make_l2_flush_fn(True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        out_path.write_text(json.dumps(result, indent=2))

    for shape_name, (width, outputs, dtype) in SHAPES.items():
        shape = dict(width=width, outputs=outputs, output_dtype=dtype,
                     weight_bytes=outputs * width * 2, rows={}, errors={},
                     arms={})
        result["shapes"][shape_name] = shape
        programs, kernels = {}, {}
        for arm in ARMS:
            label = f"{shape_name}-{arm}"
            try:
                program, kernel = build(arm, width, outputs, dtype, arm, shape_name)
            except Exception as error:  # a rejected arm is a result
                shape["errors"][arm] = repr(error)
                print(json.dumps(dict(shape=shape_name, arm=arm,
                                      rejected=repr(error))), flush=True)
                continue
            programs[arm] = program
            kernels[arm] = kernel
        if "canon" not in programs or "serve" not in programs:
            raise SystemExit(f"{shape_name}: baseline arms did not compile")
        shape["arms"] = kernels
        dump()
        if args.compile_only:
            print(json.dumps(dict(shape=shape_name, compiled=list(programs),
                                  errors=shape["errors"])), flush=True)
            programs.clear()
            kernels.clear()
            torch.cuda.empty_cache()
            continue

        weight = torch.randn(outputs, width, dtype=torch.bfloat16, device=device)
        x_prefix = torch.randn(max(ROWS), width, dtype=torch.bfloat16, device=device)
        kept = [weight, x_prefix]
        row0 = {}
        for rows in ROWS:
            x = x_prefix[:rows]
            out = torch.empty(rows, outputs, dtype=torch_dtype(dtype), device=device)
            kept.append(out)
            expected = None
            for arm in list(programs):
                try:
                    out.zero_()
                    programs[arm].launch(x, weight, out, scalars=[rows])
                    actual = out.clone()
                    if arm == "canon":
                        expected = actual
                        row0[rows] = region_bits(actual, 0)
                    else:
                        assert same_bits(expected, actual), (
                            f"{shape_name}-{arm}-r{rows}: not bitwise identical "
                            f"to the canonical compensated kernel "
                            f"({int((bit_view(expected) != bit_view(actual)).sum())} "
                            "differing elements)")
                except Exception as error:
                    if arm in ("canon", "serve"):
                        raise  # the baselines must work
                    shape["errors"].setdefault(arm, repr(error))
                    print(json.dumps(dict(shape=shape_name, arm=arm, rows=rows,
                                          dropped=repr(error))), flush=True)
                    del programs[arm]
                finally:
                    actual = None
            graphs = {}
            for arm in programs:
                graphs[arm] = capture_cuda_graph(
                    lambda p=programs[arm]: p.launch(x, weight, out, scalars=[rows]),
                    warmup=3)
            cold = {arm: [] for arm in graphs}
            warm = {arm: [] for arm in graphs}
            active = list(graphs)
            for block in range(args.blocks):
                sequence = active if block % 2 == 0 else active[::-1]
                for arm in sequence:
                    cold[arm].extend(bench_cuda_graph(
                        graphs[arm], replays=args.replays, l2_flush=flush)["replay_us"])
                    warm[arm].extend(bench_cuda_graph(
                        graphs[arm], replays=args.replays)["replay_us"])
            row = {}
            for arm in active:
                row[arm] = dict(
                    cold_us=statistics.median(cold[arm]),
                    warm_us=statistics.median(warm[arm]),
                    cold_min_us=min(cold[arm]),
                    stream_bytes=roofline(width, outputs, dtype, rows))
            base = row["serve"]["cold_us"]
            for arm in active:
                row[arm]["ratio_vs_serve"] = base / row[arm]["cold_us"]
            shape["rows"][str(rows)] = row
            print(json.dumps(dict(shape=shape_name, rows=rows, arms=row)), flush=True)
            dump()
            del graphs
            torch.cuda.empty_cache()
        # Row invariance of the canonical kernel: one token's logits cannot
        # depend on how many other rows shared the launch.
        first = ROWS[0]
        for rows in ROWS[1:]:
            assert same_bits(row0[first], row0[rows]), (
                f"{shape_name}: canonical row 0 changed between rows={first} "
                f"and rows={rows}")
        shape["row_invariance"] = True
        del kept, weight, x_prefix
        torch.cuda.empty_cache()

    dump()
    print(json.dumps(dict(output=str(out_path), shapes=list(result["shapes"]))),
          flush=True)


if __name__ == "__main__":
    main()
