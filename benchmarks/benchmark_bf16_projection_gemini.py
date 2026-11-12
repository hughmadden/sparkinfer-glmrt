#!/usr/bin/env python3
"""Race retained native SIMT and 32x64x64 MMA BF16 projections, not dispatch.

Example (run on an explicitly isolated GPU):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_bf16_projection.py \
    --cases wo-a,compressor --rows 8,9,16,24 --output /tmp/projection.json

Use --rows all for 1/6/8/9/16/24/64/256/4096, --output-dtypes bf16,fp32,
--layouts contiguous,row-strided, and --cases vocab --vocab-width 32768 for
expanded diagnosis. Large-M SIMT and vocabulary sweeps are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as compile_native,
    run_compiled,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import current_cuda_stream
from b12x.gemm.bf16_gemv import _kernel as native
from b12x.gemm.bf16_gemv import query_from_call
from b12x.gemm.bf16_gemv._tuning import TUNING
from benchmarks.kimi_k6_dense import thermal_snapshot
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
    resolve_l2_flush_bytes,
)

CASES = {
    "wo-a": (1024, 4096),
    "compressor": (512, 5120),
    "dspark-aux": (5120, 15360),
    "router": (256, 5120),
}
KIMI_D2_SHAPES = (
    ("kda-in", 24752, 7168),
    ("gate-up", 6144, 7168),
    ("out-proj", 7168, 6144),
    ("down", 7168, 3072),
    ("latent-down", 3584, 7168),
    ("latent-up", 7168, 3584),
    ("mla-qkv", 2112, 7168),
)
KIMI_D2_EXTRA_SHAPES = (
    ("mla-qb", 9216, 1536),
    ("dense-gate-up", 33792, 7168),
    ("dense-down", 7168, 16896),
    ("kimi-router", 896, 7168),
)
KIMI_D2_ROWS = (64, 128, 511, 512, 513, 1024, 2048, 2240, 2688, 2752, 4096)
ALL_ROWS = (1, 6, 8, 9, 16, 24, 64, 256, 4096)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ARMS = ("simt", "mma")
ROUTED_ARMS = ("base", "cand")


def _parse_shapes_arg(spec: str) -> list[tuple[str, int, int]]:
    if spec == "kimi-d2":
        return list(KIMI_D2_SHAPES)
    if spec == "kimi-d2-all":
        return list(KIMI_D2_SHAPES) + list(KIMI_D2_EXTRA_SHAPES)
    named = {name: (name, n, k) for name, n, k in (*KIMI_D2_SHAPES, *KIMI_D2_EXTRA_SHAPES)}
    out: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            raise ValueError("empty shape entry")
        if item in named:
            entry = named[item]
        else:
            if ":" in item:
                label, dims = item.split(":", 1)
            else:
                label, dims = item, item
            n_str, k_str = dims.lower().split("x", 1)
            entry = (label.strip(), int(n_str), int(k_str))
        if entry[0] in seen or entry[1] < 16 or entry[2] < 64 or entry[2] % 64 != 0:
            raise ValueError(f"invalid or duplicate shape entry: {item}")
        seen.add(entry[0])
        out.append(entry)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file outside the repository",
    )
    parser.add_argument(
        "--mode",
        choices=("native", "routed"),
        default="native",
        help="native races SmallNGemv vs 32x64x64 MMA; routed races RoutedBf16Projection prefill arms",
    )
    parser.add_argument(
        "--shapes",
        default="kimi-d2",
        help="For --mode routed: kimi-d2, kimi-d2-all, or comma-separated [label:]NxK",
    )
    parser.add_argument(
        "--base-spec",
        default="c324b67f",
        help="For --mode routed: c324b67f, wide-floor, routed-wide, or JSON config dict",
    )
    parser.add_argument(
        "--cand-spec",
        default="routed-wide",
        help="For --mode routed: routed-wide, wide-floor, c324b67f, or JSON config dict",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help="Comma-separated wo-a,compressor,dspark-aux,router,vocab",
    )
    parser.add_argument(
        "--rows",
        default="1,8,9,16,24,256",
        help="Comma-separated positive M values, all, or kimi-d2",
    )
    parser.add_argument(
        "--output-dtypes", default="bf16", help="bf16, fp32, or bf16,fp32"
    )
    parser.add_argument(
        "--layouts",
        default="contiguous",
        help="contiguous,row-strided; strided rows have 16 padding elements",
    )
    parser.add_argument("--vocab-width", type=int, default=32768)
    parser.add_argument("--vocab-hidden", type=int, default=5120)
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Even number of paired blocks; alternate AB/BA",
    )
    parser.add_argument(
        "--replays", type=int, default=8, help="Graph samples per arm per paired block"
    )
    parser.add_argument(
        "--launches",
        type=int,
        default=1,
        help="Native calls per graph replay; samples normalized per call",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Calls per arm before capture"
    )
    parser.add_argument(
        "--cache",
        choices=("warm", "cold"),
        default="warm",
        help="Cold flushes L2 before each replay, outside timing",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="0 selects common helper's automatic L2 flush size",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    for name, choices in (
        ("cases", (*CASES, "vocab")),
        ("output_dtypes", DTYPES),
        ("layouts", ("contiguous", "row-strided")),
    ):
        values = getattr(args, name).split(",")
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in choices for value in values)
        ):
            parser.error(
                f"invalid or duplicate --{name.replace('_', '-')}: choose from {','.join(choices)}"
            )
        setattr(args, name, values)
    try:
        if args.rows == "all":
            args.rows = list(ALL_ROWS)
        elif args.rows == "kimi-d2":
            args.rows = list(KIMI_D2_ROWS)
        else:
            args.rows = [int(value) for value in args.rows.split(",")]
    except ValueError:
        parser.error("--rows must contain integers, all, or kimi-d2")
    if args.mode == "routed":
        try:
            args.parsed_shapes = _parse_shapes_arg(args.shapes)
        except ValueError as err:
            parser.error(str(err))
    else:
        args.parsed_shapes = []
    if (
        not args.rows
        or any(value < 1 or value >= 2**31 for value in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("--rows must be distinct positive int32 values")
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be positive and even for balanced arm ordering")
    if min(args.replays, args.launches, args.warmup) < 1 or args.l2_flush_bytes < 0:
        parser.error(
            "--replays, --launches, --warmup must be positive; flush bytes nonnegative"
        )
    if min(args.vocab_width, args.vocab_hidden) < 16:
        parser.error("vocabulary N and K must be at least 16 for the native MMA arm")
    if args.cache == "cold" and args.launches != 1:
        parser.error(
            "cold-L2 requires --launches 1 so every measured projection follows eviction"
        )
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def toolchain():
    versions = {}
    for name in (
        "torch",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "cuda-bindings",
        "triton",
        "b12x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "torch_git": torch.version.git_version,
    }


def oracle(x, weight):
    # Full FP32 contraction of the actual BF16 operands, never a timed arm.
    # Column chunking bounds temporary FP32 weights even for vocabulary widths.
    expected = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )
    x32 = x.float()
    for start in range(0, weight.shape[0], 2048):
        stop = min(start + 2048, weight.shape[0])
        expected[:, start:stop] = x32 @ weight[start:stop].float().T
    torch.cuda.synchronize()
    if (
        not torch.isfinite(expected).all().item()
        or not (expected.abs().amax(dim=1) > 0).all().item()
    ):
        raise AssertionError(
            "FP32 oracle must be finite with nonzero output in every row"
        )
    return expected


def check_output(out, expected, label):
    actual = out.float()
    finite = bool(torch.isfinite(actual).all().item())
    nonzero_rows = bool((actual.abs().amax(dim=1) > 0).all().item())
    if not finite or not nonzero_rows:
        raise AssertionError(
            f"{label}: output is nonfinite, unwritten, or has an all-zero row"
        )
    # BF16 rounding is part of the output contract, not an FP32 accumulation error.
    target = expected.to(out.dtype).float()
    rtol, atol = (0.0, 5e-5) if out.dtype == torch.bfloat16 else (2e-4, 3e-5)
    if out.dtype == torch.bfloat16:
        # Independent FP32 summation orders can straddle a BF16 midpoint.
        # Permit one representable BF16 step, plus a small cancellation floor.
        ulps = (
            out.view(torch.int16).int() - expected.to(out.dtype).view(torch.int16).int()
        ).abs()
        bad = (ulps > 1) & ((actual - target).abs() > atol)
        if bad.any().item():
            raise AssertionError(
                f"{label}: {bad.sum().item()} outputs exceed one BF16 ULP "
                f"and the {atol} cancellation floor"
            )
    else:
        torch.testing.assert_close(
            actual, target, rtol=rtol, atol=atol, msg=lambda msg: f"{label}: {msg}"
        )
    error = actual - expected
    return {
        "finite": finite,
        "every_row_nonzero": nonzero_rows,
        "all_elements_checked": out.numel(),
        "rtol": rtol,
        "atol": atol,
        "bf16_ulp_budget": 1 if out.dtype == torch.bfloat16 else None,
        "oracle": "FP32 x @ weight.T with TF32 disabled; BF16 allows one ULP plus cancellation floor",
        "max_abs_error_vs_fp32": float(error.abs().max().item()),
        "relative_rms_error_vs_fp32": float(
            (error.square().mean() / expected.square().mean()).sqrt().item()
        ),
    }


def run_case(args, name, rows, dtype_name, layout, case_index, flush):
    n, k = (args.vocab_width, args.vocab_hidden) if name == "vocab" else CASES[name]
    padding = 16 if layout == "row-strided" else 0
    # A geometry-specific seed keeps operands stable when selecting a CLI subset.
    seed = args.seed + rows * 1000003 + n * 1009 + k
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((rows, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    weight = torch.empty((n, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    x.normal_(generator=generator)
    weight.normal_(std=1 / math.sqrt(k), generator=generator)
    out = torch.empty((rows, n + padding), device="cuda", dtype=DTYPES[dtype_name])[
        :, :n
    ]
    native._validate(x, weight, out, None)
    key = (n, k, str(x.dtype), str(weight.dtype), str(out.dtype), 8)
    default = TUNING.configure(query_from_call(x, weight, out=out), device=None).default
    pointer_types = (x.dtype, weight.dtype, x.dtype, out.dtype)
    pointers = tuple(
        native.make_ptr(
            native._DTYPES[dtype],
            16,
            native.cute.AddressSpace.gmem,
            assumed_align=dtype.itemsize,
        )
        for dtype in pointer_types
    )
    # Compile the native entrypoints directly to compare both implementations.
    compiled = tuple(
        compile_native(
            kernel,
            *pointers,
            native.Int32(1),
            native.Int64(k),
            native.Int64(k),
            native.Int64(n),
            native.Int64(1),
            native.Int64(1),
            native.Int32(0),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "benchmark.bf16_projection",
                1,
                (*key, arm),
            ),
        )
        for arm, kernel in zip(
            ARMS,
            (
                native.SmallNGemvKernel(n, k, True, False),
                native.Bf16GemmKernel(n, k, False),
            ),
            strict=True,
        )
    )
    vector_loads = int(
        x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and x.stride(0) % 8 == 0
        and weight.stride(0) % 8 == 0
        and x.stride(1) == weight.stride(1) == 1
        and k % 8 == 0
    )
    launch_args = (
        native._pointer(x),
        native._pointer(weight),
        native._pointer(x),
        native._pointer(out),
        rows,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        x.stride(1),
        weight.stride(1),
        vector_loads,
    )

    def launch_kernel(kernel):
        # torch.cuda.graph uses its own capture stream; never bind that stream
        # before entering capture, even though pointers and kernels are fixed.
        run_compiled(kernel, (*launch_args, current_cuda_stream()))

    launches = {
        arm: partial(launch_kernel, kernel)
        for arm, kernel in zip(ARMS, compiled, strict=True)
    }
    expected = oracle(x, weight)
    correctness = {}
    initial_order = ARMS if case_index % 2 == 0 else ARMS[::-1]
    for arm in initial_order:
        out.fill_(float("nan"))
        for _ in range(args.warmup):
            launches[arm]()
        torch.cuda.synchronize()
        correctness[arm] = {"eager": check_output(out, expected, arm)}

    graphs = {}
    blocks = []
    samples = {arm: [] for arm in ARMS}
    with kernel_resolution_guard("BF16 projection native SIMT/MMA graph capture and paired replay"):
        for arm in initial_order:

            def repeated(launch=launches[arm]):
                for _ in range(args.launches):
                    launch()

            graphs[arm] = capture_cuda_graph(repeated, warmup=0)
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_before_timing"] = check_output(out, expected, arm)
        gpu_before = nvidia_smi_gpu_mode_snapshot()
        for repeat in range(args.repeats):
            order = initial_order if repeat % 2 == 0 else initial_order[::-1]
            block = {"repeat": repeat, "order": list(order), "replay_us": {}}
            for arm in order:
                raw = bench_cuda_graph(
                    graphs[arm], replays=args.replays, l2_flush=flush
                )["replay_us"]
                if any(not math.isfinite(value) or value <= 0 for value in raw):
                    raise RuntimeError(f"Invalid CUDA event samples for {arm}: {raw}")
                block["replay_us"][arm] = raw
                samples[arm].extend(value / args.launches for value in raw)
            block["simt_over_mma_ratio"] = statistics.median(
                block["replay_us"]["simt"]
            ) / statistics.median(block["replay_us"]["mma"])
            blocks.append(block)
        gpu_after = nvidia_smi_gpu_mode_snapshot()
        for arm in ARMS:
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_after_timing"] = check_output(out, expected, arm)
    for graph in graphs.values():
        graph.reset()
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    ratio = medians["simt"] / medians["mma"]
    return {
        "case": name,
        "M": rows,
        "N": n,
        "K": k,
        "seed": seed,
        "operand_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": dtype_name,
        "layout": layout,
        "strides": {
            "x": list(x.stride()),
            "weight": list(weight.stride()),
            "out": list(out.stride()),
        },
        "shared_operands_and_output_between_arms": True,
        "simt_vector_loads": bool(vector_loads),
        "cache_key": list(key),
        "implementations": {
            "simt": "SmallNGemvKernel",
            "mma": "Bf16GemmKernel(32,64,64)",
        },
        "production_policy_arm": default.backend,
        "production_policy_regresses_vs_simt": (
            ratio < 1 if default.backend == "mma" else
            False if default.backend == "simt" else None
        ),
        "correctness": correctness,
        "median_us_per_projection": medians,
        "simt_over_mma_ratio": ratio,
        "faster_arm": "mma" if ratio > 1 else "simt" if ratio < 1 else "tie",
        "mma_winning_blocks": sum(block["simt_over_mma_ratio"] > 1 for block in blocks),
        "samples_us_per_projection": samples,
        "blocks": blocks,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def _bits_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    raise TypeError(f"unsupported output dtype for bitwise check: {tensor.dtype}")


def _sample_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)

    def pct(q: float) -> float:
        idx = q * (count - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(ordered[lo])
        frac = idx - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    p25, p75 = pct(0.25), pct(0.75)
    return {
        "count": count,
        "median_us": float(statistics.median(ordered)),
        "mean_us": float(statistics.fmean(ordered)),
        "stdev_us": float(statistics.stdev(ordered)) if count > 1 else 0.0,
        "min_us": float(ordered[0]),
        "p10_us": pct(0.10),
        "p25_us": p25,
        "p75_us": p75,
        "p90_us": pct(0.90),
        "max_us": float(ordered[-1]),
        "iqr_us": float(p75 - p25),
    }


class _PinnedRoutedLaunch:
    def __init__(
        self,
        decode_proj,
        decode_limit: int,
        c324_proj,
        band_floors: tuple[int, ...],
        band_limits: tuple[int, ...],
        band_projs: list[object],
    ):
        self.decode_proj = decode_proj
        self.decode_limit = decode_limit
        self.c324_proj = c324_proj
        self.band_floors = band_floors
        self.band_limits = band_limits
        self.band_projs = band_projs

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.decode_proj is not None):
            if rows <= Int32(self.decode_limit):
                self.decode_proj(x, w, out, rows, stream)
        if rows > Int32(self.decode_limit):
            if cutlass.const_expr(len(self.band_projs) == 0):
                self.c324_proj(x, w, out, rows, stream)
            else:
                if rows < Int32(self.band_floors[0]):
                    self.c324_proj(x, w, out, rows, stream)
                for idx in cutlass.range_constexpr(len(self.band_projs)):
                    if rows >= Int32(self.band_floors[idx]):
                        if rows <= Int32(self.band_limits[idx]):
                            self.band_projs[idx](x, w, out, rows, stream)
                if rows > Int32(self.band_limits[-1]):
                    self.c324_proj(x, w, out, rows, stream)


def _build_routed_program(n: int, k: int, dtype_name: str, spec: str, arm_tag: str):
    from b12x.gemm.bf16_gemv._skinny import (
        DECODE_TMA_TILES,
        RoutedBf16Projection,
        RowTiledBf16Projection,
        TmaBf16Projection,
        WIDE_PREFILL_MIN_ROWS,
        wide_prefill_min_rows,
    )
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    torch_dtype = DTYPES[dtype_name]
    cutlass_dtype = (
        cutlass.BFloat16 if dtype_name == "bf16" else cutlass.Float32
    )

    wide_floor = wide_prefill_min_rows(n, k, 2**31 - 1)
    if spec == "routed-wide":
        launch = RoutedBf16Projection(
            n, k, out_dtype=cutlass_dtype, max_skinny_rows=0,
            wide=wide_floor is not None, wide_min_rows=wide_floor
        )
        desc = f"RoutedBf16Projection(wide=True, key={launch.key()})"
        prog_key = ("routed-wide", arm_tag, launch.key())
    else:
        decode_tiles = DECODE_TMA_TILES.get((n, k), ())
        decode_limit = int(decode_tiles[-1][0]) if decode_tiles else 0
        decode_proj = (
            RowTiledBf16Projection(n, k, decode_tiles, out_dtype=cutlass_dtype)
            if decode_tiles
            else None
        )
        c324_cfg = dict(
            compute_warps=4,
            tile_n=128 if n >= 1024 else 64,
            num_stages=3 if n >= 1024 else 4,
            compensated=False,
        )
        c324_proj = TmaBf16Projection(
            n, k, out_dtype=cutlass_dtype, tiles_by_rows=(), **c324_cfg
        )
        bands: list[tuple[int, int, object, dict]] = []
        if spec == "c324b67f":
            desc = f"c324b67f_pinned({c324_cfg})"
        elif spec == "wide-floor":
            if wide_floor is not None and n % 2 == 0:
                wf_cfg = dict(
                    warp_layout=(2, 2),
                    tile_m=128,
                    tile_n=128,
                    num_stages=2,
                    compensated=False,
                )
                bands.append(
                    (
                        wide_floor,
                        2**31 - 1,
                        TmaBf16Projection(
                            n, k, out_dtype=cutlass_dtype, tiles_by_rows=(), **wf_cfg
                        ),
                        wf_cfg,
                    )
                )
            desc = "wide-floor(warp_layout=(2,2),tile_m=128,tile_n=128,num_stages=2)"
        elif spec.strip().startswith("{"):
            parsed = json.loads(spec)
            if "bands" in parsed:
                for min_r, max_r, raw_cfg in parsed["bands"]:
                    cfg = dict(raw_cfg)
                    if "warp_layout" in cfg and cfg["warp_layout"] is not None:
                        cfg["warp_layout"] = tuple(int(v) for v in cfg["warp_layout"])
                    cfg.setdefault("compensated", False)
                    bands.append(
                        (
                            int(min_r),
                            int(max_r),
                            TmaBf16Projection(
                                n, k, out_dtype=cutlass_dtype, tiles_by_rows=(), **cfg
                            ),
                            cfg,
                        )
                    )
            else:
                cfg = dict(parsed)
                min_r = int(cfg.pop("min_rows", WIDE_PREFILL_MIN_ROWS))
                max_r = int(cfg.pop("max_rows", 2**31 - 1))
                if "warp_layout" in cfg and cfg["warp_layout"] is not None:
                    cfg["warp_layout"] = tuple(int(v) for v in cfg["warp_layout"])
                cfg.setdefault("compensated", False)
                bands.append(
                    (
                        min_r,
                        max_r,
                        TmaBf16Projection(
                            n, k, out_dtype=cutlass_dtype, tiles_by_rows=(), **cfg
                        ),
                        cfg,
                    )
                )
            desc = f"custom({spec})"
        else:
            raise ValueError(f"unknown routed spec: {spec}")

        band_floors = tuple(b[0] for b in bands)
        band_limits = tuple(b[1] for b in bands)
        band_projs = [b[2] for b in bands]
        launch = _PinnedRoutedLaunch(
            decode_proj, decode_limit, c324_proj, band_floors, band_limits, band_projs
        )
        prog_key = (
            "pinned",
            arm_tag,
            n,
            k,
            dtype_name,
            spec,
            tuple((b[0], b[1], b[2].key()) for b in bands),
        )

    program = compile_program(
        launch,
        name=f"bench_routed_{arm_tag}",
        operands=(
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("w", torch.bfloat16, f"[{n},{k}]"),
            Operand("out", torch_dtype, f"[rows,{n}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=prog_key,
    )
    return program, desc


def run_routed_shape_row(
    args,
    label: str,
    n: int,
    k: int,
    rows: int,
    dtype_name: str,
    programs: dict[str, object],
    descriptions: dict[str, str],
    weight: torch.Tensor,
    case_index: int,
    flush,
):
    seed = args.seed + rows * 1000003 + n * 1009 + k
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((rows, k), device="cuda", dtype=torch.bfloat16)
    x.normal_(generator=generator)
    out_dtype = DTYPES[dtype_name]
    tail_pad = 16
    out_buffers = {
        arm: torch.empty((rows + tail_pad, n), device="cuda", dtype=out_dtype)
        for arm in ROUTED_ARMS
    }
    shared_out_buf = torch.empty((rows + tail_pad, n), device="cuda", dtype=out_dtype)
    shared_out = shared_out_buf[:rows]

    expected = oracle(x, weight)
    correctness: dict[str, dict[str, object]] = {}
    initial_order = ROUTED_ARMS if case_index % 2 == 0 else ROUTED_ARMS[::-1]

    # Pre-warm both callables at this live row count and validate eager numerics + tail poison.
    for arm in initial_order:
        buf = out_buffers[arm]
        buf.fill_(float("nan"))
        view = buf[:rows]
        for _ in range(args.warmup):
            programs[arm].launch(x, weight, view, scalars=(rows,))
        torch.cuda.synchronize()
        if not bool(torch.isnan(buf[rows:]).all().item()):
            raise AssertionError(f"{label} M={rows} {arm}: tail rows past M were overwritten")
        correctness[arm] = {
            "eager": check_output(view, expected, f"{label}:{arm}:eager"),
            "tail_poison_intact_eager": True,
        }

    eager_bits_equal = bool(
        torch.equal(
            _bits_view(out_buffers["cand"][:rows]),
            _bits_view(out_buffers["base"][:rows]),
        )
    )
    if not eager_bits_equal:
        diff = int(
            (_bits_view(out_buffers["cand"][:rows]) != _bits_view(out_buffers["base"][:rows]))
            .sum()
            .item()
        )
        raise AssertionError(
            f"{label} M={rows}: cand eager output is NOT bitwise equal to base ({diff} elements differ)"
        )

    graphs = {}
    blocks = []
    samples = {arm: [] for arm in ROUTED_ARMS}
    with kernel_resolution_guard(
        "Kimi BF16 routed prefill projection graph capture, dynamic-row invariance, and paired replay"
    ):
        # Row-invariance check under frozen kernel resolution (reuses the same compiled callable!).
        probe_counts = sorted({max(1, rows - 13), *(min(rows, m) for m in
                              (1, 16, 64, 65, 511, 512, 1023, 1024, 1025, 2048, 2752, 4096))})
        row_invariance_checked: list[int] = []
        for probe_m in probe_counts:
            if probe_m <= 0 or probe_m > rows:
                continue
            probe_buf = torch.full(
                (probe_m + tail_pad, n), float("nan"), device="cuda", dtype=out_dtype
            )
            programs["cand"].launch(
                x[:probe_m], weight, probe_buf[:probe_m], scalars=(probe_m,)
            )
            torch.cuda.synchronize()
            if not bool(torch.isnan(probe_buf[probe_m:]).all().item()):
                raise AssertionError(
                    f"{label} M={rows} probe_m={probe_m}: cand overwrote tail rows"
                )
            if not bool(
                torch.equal(
                    _bits_view(probe_buf[:probe_m]),
                    _bits_view(out_buffers["base"][:probe_m]),
                )
            ):
                raise AssertionError(
                    f"{label} M={rows} probe_m={probe_m}: cand failed strict row-invariance vs base[:{probe_m}]"
                )
            row_invariance_checked.append(probe_m)

        for arm in initial_order:

            def repeated(prog=programs[arm]):
                for _ in range(args.launches):
                    prog.launch(x, weight, shared_out, scalars=(rows,))

            graphs[arm] = capture_cuda_graph(repeated, warmup=0)
            shared_out_buf.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            if not bool(torch.isnan(shared_out_buf[rows:]).all().item()):
                raise AssertionError(
                    f"{label} M={rows} {arm}: graph replay overwrote tail rows"
                )
            correctness[arm]["graph_before_timing"] = check_output(
                shared_out, expected, f"{label}:{arm}:graph_before"
            )
            if not bool(
                torch.equal(
                    _bits_view(shared_out),
                    _bits_view(out_buffers["base"][:rows]),
                )
            ):
                raise AssertionError(
                    f"{label} M={rows} {arm}: graph_before_timing is NOT bitwise equal to base"
                )

        gpu_before = nvidia_smi_gpu_mode_snapshot()
        thermal_before = thermal_snapshot()
        for repeat in range(args.repeats):
            order = initial_order if repeat % 2 == 0 else initial_order[::-1]
            block = {"repeat": repeat, "order": list(order), "replay_us": {}}
            for arm in order:
                before = thermal_snapshot()
                raw = bench_cuda_graph(
                    graphs[arm], replays=args.replays, l2_flush=flush
                )["replay_us"]
                if any(not math.isfinite(value) or value <= 0 for value in raw):
                    raise RuntimeError(f"Invalid CUDA event samples for {arm}: {raw}")
                block.setdefault("thermal", {})[arm] = {
                    "before": before, "after": thermal_snapshot()}
                block["replay_us"][arm] = raw
                samples[arm].extend(value / args.launches for value in raw)
            block["base_over_cand_ratio"] = statistics.median(
                block["replay_us"]["base"]
            ) / statistics.median(block["replay_us"]["cand"])
            blocks.append(block)
        thermal_after = thermal_snapshot()
        gpu_after = nvidia_smi_gpu_mode_snapshot()

        graph_after_outputs = {}
        for arm in ROUTED_ARMS:
            shared_out_buf.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            if not bool(torch.isnan(shared_out_buf[rows:]).all().item()):
                raise AssertionError(
                    f"{label} M={rows} {arm}: graph_after_timing overwrote tail rows"
                )
            correctness[arm]["graph_after_timing"] = check_output(
                shared_out, expected, f"{label}:{arm}:graph_after"
            )
            graph_after_outputs[arm] = shared_out.clone()

    for graph in graphs.values():
        graph.reset()

    graph_bits_equal = bool(
        torch.equal(
            _bits_view(graph_after_outputs["cand"]),
            _bits_view(graph_after_outputs["base"]),
        )
    )
    if not graph_bits_equal:
        raise AssertionError(
            f"{label} M={rows}: cand graph_after_timing is NOT bitwise equal to base"
        )

    stats = {arm: _sample_stats(values) for arm, values in samples.items()}
    medians = {arm: stats[arm]["median_us"] for arm in ROUTED_ARMS}
    base_over_cand = medians["base"] / medians["cand"]
    cand_over_base = medians["cand"] / medians["base"]
    flops = 2.0 * rows * n * k
    out_elem_bytes = 2 if dtype_name == "bf16" else 4
    logical_bytes = 2.0 * rows * k + 2.0 * n * k + float(out_elem_bytes) * rows * n
    tflops = {arm: flops / (medians[arm] * 1e6) for arm in ROUTED_ARMS}
    gbps = {arm: logical_bytes / (medians[arm] * 1e3) for arm in ROUTED_ARMS}

    return {
        "case": label,
        "M": rows,
        "N": n,
        "K": k,
        "seed": seed,
        "operand_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": dtype_name,
        "implementations": descriptions,
        "bitwise_equal_eager": eager_bits_equal,
        "bitwise_equal_graph": graph_bits_equal,
        "row_invariance_probes": row_invariance_checked,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "correctness": correctness,
        "median_us_per_projection": medians,
        "stats_us": stats,
        "effective_tflops": tflops,
        "logical_gbps": gbps,
        "base_over_cand_ratio": base_over_cand,
        "cand_over_base_ratio": cand_over_base,
        "faster_arm": (
            "cand" if base_over_cand > 1 else "base" if base_over_cand < 1 else "tie"
        ),
        "cand_winning_blocks": sum(
            block["base_over_cand_ratio"] > 1 for block in blocks
        ),
        "samples_us_per_projection": samples,
        "blocks": blocks,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def main():
    args = parse_args()
    require_sm120()
    # These options apply only to the untimed FP32 numerical oracle.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush_bytes = (
        resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache == "cold" else 0
    )
    flush = make_l2_flush_fn(args.cache == "cold", flush_bytes)
    sources = [
        Path(__file__),
        Path(native.__file__),
        ROOT / "b12x/gemm/bf16_gemv/_prefill.py",
        ROOT / "b12x/gemm/bf16_gemv/_skinny.py",
        ROOT / "benchmarks/common.py",
        ROOT / "benchmarks/kimi_k6_dense.py",
        ROOT / "b12x/_lib/compiler.py",
        ROOT / "b12x/_lib/intrinsics.py",
        ROOT / "b12x/_lib/runtime_control.py",
        ROOT / "b12x/_lib/utils.py",
    ]
    timing_scope = (
        "CUDA events around captured AOT RoutedBf16Projection replay only; compilation (once per shape), FP32 oracle, bitwise torch.equal, dynamic-row invariance probes, tail-poison checks, and cold-L2 flush excluded. Paired alternating AB/BA blocks; one shared output."
        if args.mode == "routed"
        else "CUDA events around captured native cached callable replay only; oracle, compilation, warmup, snapshots, correctness and optional L2 flush excluded. Paired alternating AB/BA blocks; one shared output; no Torch/CUBLAS measured arm."
    )
    ratio_def = (
        "base_over_cand_ratio = median base us / median cand us (>1 means cand is faster); cand_over_base_ratio = median cand us / median base us (<1 means cand is faster)."
        if args.mode == "routed"
        else "simt_over_mma_ratio = median SIMT us / median MMA us; >1 means MMA faster, <1 means MMA is slower than SIMT. No significance claim."
    )
    result = {
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "command": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "B12X_", "CUTE_", "CUTLASS_"))
        },
        "toolchain": toolchain(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "device": {
            "logical_index": torch.cuda.current_device(),
            "uuid": str(getattr(properties, "uuid", "")),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        },
        "timing_scope": timing_scope,
        "ratio_definition": ratio_def,
        "l2_flush_bytes": flush_bytes,
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.mode == "routed":
            for label, n, k in args.parsed_shapes:
                for dtype in args.output_dtypes:
                    w_gen = torch.Generator(device="cuda").manual_seed(
                        args.seed + n * 1009 + k
                    )
                    weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
                    weight.normal_(std=1 / math.sqrt(k), generator=w_gen)
                    base_prog, base_desc = _build_routed_program(
                        n, k, dtype, args.base_spec, "base"
                    )
                    cand_prog, cand_desc = _build_routed_program(
                        n, k, dtype, args.cand_spec, "cand"
                    )
                    progs = {"base": base_prog, "cand": cand_prog}
                    descs = {"base": base_desc, "cand": cand_desc}
                    for rows in args.rows:
                        record = run_routed_shape_row(
                            args,
                            label,
                            n,
                            k,
                            rows,
                            dtype,
                            progs,
                            descs,
                            weight,
                            len(result["records"]),
                            flush,
                        )
                        result["records"].append(record)
                        save()
                        print(
                            json.dumps(
                                {
                                    "case": record["case"],
                                    "M": record["M"],
                                    "N": record["N"],
                                    "K": record["K"],
                                    "base_us": round(
                                        record["median_us_per_projection"]["base"], 2
                                    ),
                                    "cand_us": round(
                                        record["median_us_per_projection"]["cand"], 2
                                    ),
                                    "base_iqr": round(
                                        record["stats_us"]["base"]["iqr_us"], 2
                                    ),
                                    "cand_iqr": round(
                                        record["stats_us"]["cand"]["iqr_us"], 2
                                    ),
                                    "speedup": round(
                                        record["base_over_cand_ratio"], 4
                                    ),
                                    "cand_tflops": round(
                                        record["effective_tflops"]["cand"], 1
                                    ),
                                    "bitwise": record["bitwise_equal_eager"]
                                    and record["bitwise_equal_graph"],
                                }
                            ),
                            flush=True,
                        )
                    del weight
        else:
            for name in args.cases:
                for rows in args.rows:
                    for dtype in args.output_dtypes:
                        for layout in args.layouts:
                            record = run_case(
                                args,
                                name,
                                rows,
                                dtype,
                                layout,
                                len(result["records"]),
                                flush,
                            )
                            result["records"].append(record)
                            save()
                            print(
                                json.dumps(
                                    {
                                        key: record[key]
                                        for key in (
                                            "case",
                                            "M",
                                            "N",
                                            "K",
                                            "output_dtype",
                                            "layout",
                                            "median_us_per_projection",
                                            "simt_over_mma_ratio",
                                            "mma_winning_blocks",
                                            "production_policy_regresses_vs_simt",
                                        )
                                    }
                                ),
                                flush=True,
                            )
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_unix_ns"] = time.time_ns()
        result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
        save()


if __name__ == "__main__":
    main()
