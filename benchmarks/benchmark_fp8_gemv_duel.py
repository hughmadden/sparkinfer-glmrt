#!/usr/bin/env python3
"""Duel harness: block-FP8 decode GEMV at checkpoint shapes, variant table.

Times the production ``MmaFp8Gemv`` (baseline) and exact-preserving variants on
the same inputs with the L2-cold graph-replay procedure of
``benchmark_weight_l2_hint.py`` (one launch per sample after a 2x-L2 eviction),
then checks each variant's output bits against the baseline's and its own
row-invariance (row 0 at M=1 equals row 0 embedded at M=16).

Only variants that keep the per-element K accumulation order are admitted:
``groups`` (columns per CTA) may change; ``warps`` may not (it reshapes the
K split and the cross-warp fold).

  python benchmarks/benchmark_fp8_gemv_duel.py --shapes qwen --rows 1 \
      --variants groups
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_weight_l2_hint import _digest  # noqa: E402

QWEN_SHAPES = ("34816x5120", "5120x17408", "16384x5120")
GLM_SHAPES = ("2624x6144", "16384x2048", "6144x16384", "4096x2048", "24576x6144",
              "6144x12288", "4096x6144", "6144x2048")
MIMO_SHAPES = ("13568x4096", "4096x8192", "32768x4096", "4096x16384")
SETS = {"qwen": QWEN_SHAPES, "glm": GLM_SHAPES, "mimo": MIMO_SHAPES}


def _time(run, samples, flush, outputs, flush_mode="zero"):
    """``benchmark_weight_l2_hint._time`` with a selectable eviction write.

    ``zero`` is the reference harness (a 2x-L2 zero fill leaves up to an L2 of
    dirty lines that the timed kernel writes back). ``read`` touches the same
    buffer read-only, so the timed kernel starts L2-cold with no writeback --
    the steady-state decode case, where nothing else dirties L2."""
    import statistics
    import torch
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for out in outputs:
        out.fill_(float("nan")) if out.is_floating_point() else out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    digest = "".join(_digest(out) for out in outputs)[:32]
    times = []
    for _ in range(samples):
        if flush_mode == "read":
            torch.sum(flush, dtype=torch.int64)
        else:
            flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times), min(times), digest


def variants_for(kind: str, n: int, k: int, rows: int) -> list[dict]:
    """Baseline first; every variant is (name, kwargs for MmaFp8Gemv)."""
    from b12x.integration.cuteafd._fp8_weights import gemv_warps
    warps = gemv_warps(k)
    out = [dict(name="baseline", warps=warps, groups=4)]
    if kind == "groups":
        for groups in (2, 8):
            if 128 % (8 * groups) == 0 and n % (8 * groups) == 0 and n // (8 * groups) >= 2:
                out.append(dict(name=f"g{groups}", warps=warps, groups=groups))
    if kind == "groups_wide":
        for groups in (1, 16):
            if 128 % (8 * groups) == 0 and n % (8 * groups) == 0 and n // (8 * groups) >= 2:
                out.append(dict(name=f"g{groups}", warps=warps, groups=groups))
    if kind == "hints":
        for hint in ("l2_256b", "l2_128b"):
            out.append(dict(name=hint, warps=warps, groups=4, load_hint=hint))
    if kind == "depth":
        for blocks, nbytes in ((4, 512), (8, 1024), (8, 2048)):
            out.append(dict(name=f"pf{blocks}x{nbytes}", warps=warps, groups=4,
                            prefetch_blocks=blocks, prefetch_bytes=nbytes))
    return out


def _compile(n: int, k: int, rows: int, cfg: dict):
    import cutlass
    import torch
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import MmaFp8Gemv

    kwargs = dict(max_rows=max(16, rows), warps=cfg["warps"], groups=cfg["groups"],
                  out_dtype=cutlass.BFloat16)
    for extra in ("load_hint", "prefetch_blocks", "prefetch_bytes"):
        if extra in cfg:
            kwargs[extra] = cfg[extra]
    op = MmaFp8Gemv(n, k, **kwargs)
    program = compile_program(op, name=f"duel_{cfg['name']}_{n}x{k}", operands=(
        Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.float8_e4m3fn, f"[{n},{k}]"),
        Operand("s", torch.float32, f"[{n // 128},{k // 128}]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{n}]", "out")), scalars=(Scalar("rows"),), key=op.key())
    return op, program


def run_shape(spec: str, rows: int, kind: str, samples: int, seed: int, flush_mode: str = "zero") -> list[dict]:
    import torch
    n, k = (int(v) for v in spec.split("x"))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(2 * l2, device="cuda", dtype=torch.uint8)
    x = torch.randn(16, k, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda", generator=gen) * 2e-4 + 1e-4
    records = []
    for cfg in variants_for(kind, n, k, rows):
        try:
            op, program = _compile(n, k, rows, cfg)
            out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            out16 = torch.empty(16, n, device="cuda", dtype=torch.bfloat16)

            def run(program=program, out=out, rows=rows):
                program.launch(x, w, s, out, scalars=(rows,))

            median, best, digest = _time(run, samples, flush, [out], flush_mode)
            median_read, best_read, digest_read = _time(run, samples, flush, [out], "read")
            assert digest_read == digest, "output bits moved with the eviction mode"
            # Row invariance inside this variant: row 0 at M=1 vs row 0 embedded at M=16.
            def run16(program=program, out=out16):
                program.launch(x, w, s, out16, scalars=(16,))

            run16()
            torch.cuda.synchronize()
            row0_a = out16.view(torch.int16)[0].clone()
            run()
            torch.cuda.synchronize()
            row0_b = out.view(torch.int16)[0].clone()
            records.append(dict(case=f"fp8gemv_{spec}", rows=rows, variant=cfg["name"],
                                warps=cfg["warps"], groups=cfg["groups"],
                                median_us=round(median, 3), min_us=round(best, 3),
                                weight_gbps=round(n * k / median / 1e3, 1),
                                median_read_us=round(median_read, 3),
                                weight_gbps_read=round(n * k / median_read / 1e3, 1),
                                digest=digest,
                                row_invariant=bool(torch.equal(row0_a, row0_b))))
            del op, program, out, out16
        except Exception as exc:  # a failed variant is a null, not a dead sweep
            records.append(dict(case=f"fp8gemv_{spec}", rows=rows, variant=cfg["name"],
                                error=repr(exc)[:400]))
        torch.cuda.empty_cache()
    base = next(rec for rec in records
                if rec["variant"] == "baseline" and "median_us" in rec)
    for rec in records:
        if "median_us" not in rec:
            continue
        rec["ratio"] = round(base["median_us"] / rec["median_us"], 4)
        rec["ratio_read"] = round(base["median_read_us"] / rec["median_read_us"], 4)
        rec["exact_vs_baseline"] = rec["digest"] == base["digest"]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="qwen", help=f"{','.join(SETS)}, or a semicolon list of NxK")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--variants", default="groups",
                        choices=("groups", "groups_wide", "hints", "depth"))
    parser.add_argument("--flush", default="zero", choices=("zero", "read"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.shapes in SETS:
        specs = SETS[args.shapes]
    else:
        specs = [s for s in args.shapes.split(";") if s]
    import torch
    print(json.dumps(dict(device=torch.cuda.get_device_name(),
                          l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
                          sms=torch.cuda.get_device_properties(0).multi_processor_count)), flush=True)
    for spec in specs:
        for rec in run_shape(spec, args.rows, args.variants, args.samples, args.seed, args.flush):
            rec["tag"] = args.tag
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
