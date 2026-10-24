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
# GLM 5.3 Flash and MiMo V2 decode projections as the engine ships them: the keys of
# ``glmf.FP8_GEMV_CONFIG``, whose (warps, groups) is already tuned per shape.
GLMF_SHAPES = ("24896x4096", "4096x8192", "4096x16384", "16384x1536", "2048x4096",
               "4096x4096", "4096x2048", "24576x4096", "4096x12288", "154880x4096")
MIMO_SHAPES = ("13568x4096", "14848x4096", "32768x4096", "152576x4096")
# Shipped per-row-scale decode projections (GLM Flash MLA in/o + FP32 head, MiMo
# qkv/o/gate_up/down + heads): same kernels, FP32 ``[N, K/128]`` scale grid.
ROWSCALE_SHAPES = ("24896x4096", "4096x8192", "4096x16384", "154880x4096",
                   "13568x4096", "14848x4096", "32768x4096", "152576x4096")
# K=6144 width family (V4-sized hidden), kept from round 1 for continuity.
GLM_SHAPES = ("2624x6144", "16384x2048", "6144x16384", "4096x2048", "24576x6144",
              "6144x12288", "4096x6144", "6144x2048")
SETS = {"qwen": QWEN_SHAPES, "glmf": GLMF_SHAPES, "glm": GLM_SHAPES, "mimo": MIMO_SHAPES,
        "rowscale": ROWSCALE_SHAPES}


def _shipped_config() -> dict:
    """The engine's per-shape decode GEMV ``(warps, groups)`` (``glmf.FP8_GEMV_CONFIG``)."""
    try:
        from b12x.integration.cuteafd.glmf import FP8_GEMV_CONFIG
        return dict(FP8_GEMV_CONFIG)
    except Exception:  # keep the benchmark runnable without the engine layer
        return {(24896, 4096): (8, 2), (4096, 8192): (8, 2), (4096, 16384): (4, 2),
                (16384, 1536): (2, 2), (2048, 4096): (8, 4), (4096, 4096): (8, 2),
                (4096, 2048): (2, 2), (24576, 4096): (8, 4), (4096, 12288): (8, 2),
                (154880, 4096): (8, 2), (13568, 4096): (2, 2), (14848, 4096): (2, 2),
                (32768, 4096): (8, 2), (152576, 4096): (8, 2)}


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
    """Baseline first; every variant is (name, kwargs for MmaFp8Gemv).

    The baseline is the config the engine ships for this shape, so a ratio is the
    gain over production. ``prod`` keeps the shipped warp count (same cross-warp
    fold, hence bit-exact) and only varies the column block size / fetch hint.
    """
    from b12x.integration.cuteafd._fp8_weights import gemv_warps
    warps = gemv_warps(k)
    out = [dict(name="baseline", warps=warps, groups=4)]
    if kind == "prod":
        warps, groups = _shipped_config().get((n, k), (gemv_warps(k), 4))
        out = [dict(name="baseline", warps=warps, groups=groups)]
        if groups != 4:
            out.append(dict(name="default_g4", warps=warps, groups=4))
        for g in (1, 2):
            if g < groups and 128 % (8 * g) == 0 and n % (8 * g) == 0 and n // (8 * g) >= 2:
                out.append(dict(name=f"g{g}", warps=warps, groups=g))
                out.append(dict(name=f"g{g}+h256", warps=warps, groups=g, load_hint="l2_256b"))
        out.append(dict(name="h256", warps=warps, groups=groups, load_hint="l2_256b"))
        return out
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
    if kind == "gem":
        # Round 4: my groups=1 rule against the opponent's vendored kernel (16-byte
        # nc loads + pipelined widen/MMA), measured in one harness so the ratio is
        # meaningful. Their rule picks groups and cta_tiles from runtime SM/L2;
        # ``gem_g1_*`` pins their load path to my column width so the two
        # improvements are separable.
        from b12x.integration.cuteafd._fp8_gemv_nc import (
            nc_gemv_cta_tiles, nc_gemv_groups)

        def legal(g):
            return 128 % (8 * g) == 0 and n % (8 * g) == 0 and n // (8 * g) >= 2

        warps, groups = _shipped_config().get((n, k), (gemv_warps(k), 4))
        out = [dict(name="baseline", warps=warps, groups=groups)]
        if groups != 1 and legal(1):
            out.append(dict(name="mine_g1", warps=warps, groups=1))
        if groups != 4 and legal(4):
            out.append(dict(name="shipped_g4", warps=warps, groups=4))
        gg = nc_gemv_groups(n, k, warps=warps, groups=groups)
        gt = nc_gemv_cta_tiles(n, k, groups=gg, warps=warps)
        out.append(dict(name=f"gem_rule_g{gg}_t{gt}", warps=warps, groups=gg,
                        cta_tiles=gt, kernel="nc"))
        # Their isolation arm (round-4 addendum): their loads/pipeline at my column
        # width and one 16-row tile per CTA.
        if legal(1):
            out.append(dict(name="gem_g1_t1", warps=warps, groups=1,
                            cta_tiles=1, kernel="nc"))
        for g in (1, 2):
            if legal(g):
                out.append(dict(name=f"gem_g{g}_t4", warps=warps, groups=g,
                                cta_tiles=4, kernel="nc"))
        return out
    if kind == "wide":
        # The 17-64-row program (``max_rows=64``): production's tile count per CTA is
        # the knob that matters here, so hold the shipped warps and sweep (groups,
        # cta_tiles) on the production kernel; the opponent's kernel stays as the
        # reference for its load path at the same tiling.
        from b12x.integration.cuteafd._fp8_gemv_nc import nc_gemv_groups

        def legal(g):
            return 128 % (8 * g) == 0 and n % (8 * g) == 0 and n // (8 * g) >= 2

        warps, groups = _shipped_config().get((n, k), (gemv_warps(k), 4))
        out = [dict(name="baseline", warps=warps, groups=groups)]
        for g, t in ((1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (4, 2)):
            if legal(g):
                out.append(dict(name=f"prod_g{g}_t{t}", warps=warps, groups=g, cta_tiles=t))
        gg = nc_gemv_groups(n, k, warps=warps, groups=groups)
        if legal(gg):
            out.append(dict(name=f"gem_rule_g{gg}_t4", warps=warps, groups=gg,
                            cta_tiles=4, kernel="nc"))
        if legal(1):
            out.append(dict(name="gem_g1_t1", warps=warps, groups=1, cta_tiles=1,
                            kernel="nc"))
        return out
    if kind == "combo":
        # Round-2 grid: the two independent winners (column count per CTA, weight
        # fetch size) and their product, plus the extreme small-CTA arm.
        for groups, hint, tag in ((2, "stream", "g2"), (4, "l2_256b", "h256"),
                                  (2, "l2_256b", "g2+h256"), (1, "l2_256b", "g1+h256"),
                                  (1, "stream", "g1")):
            if 128 % (8 * groups) == 0 and n % (8 * groups) == 0 and n // (8 * groups) >= 2:
                cfg = dict(name=tag, warps=warps, groups=groups)
                if hint != "stream":
                    cfg["load_hint"] = hint
                out.append(cfg)
    return out


def _compile(n: int, k: int, rows: int, cfg: dict, row_scales: bool = False):
    import cutlass
    import torch
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import MmaFp8Gemv

    # Production builds two programs per projection: ``max_rows=16`` for the thin
    # batch and ``max_rows=decode_rows`` (64) for a wider one. Matching that here
    # makes the ``rows=32/64`` cells the program the engine actually launches.
    cap = 16 if rows <= 16 else 64
    kwargs = dict(max_rows=cap, warps=cfg["warps"], groups=cfg["groups"],
                  out_dtype=cutlass.BFloat16, row_scales=row_scales)
    scale_shape = f"[{n},{k // 128}]" if row_scales else f"[{n // 128},{k // 128}]"
    for extra in ("load_hint", "prefetch_blocks", "prefetch_bytes", "cta_tiles"):
        if extra in cfg:
            kwargs[extra] = cfg[extra]
    cls = MmaFp8Gemv
    if cfg.get("kernel") == "nc":
        from b12x.integration.cuteafd._fp8_gemv_nc import NcMmaFp8Gemv
        cls = NcMmaFp8Gemv
    op = cls(n, k, **kwargs)
    program = compile_program(op, name=f"duel_{cfg['name']}_{n}x{k}", operands=(
        Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.float8_e4m3fn, f"[{n},{k}]"),
        Operand("s", torch.float32, scale_shape, align=4),
        Operand("out", torch.bfloat16, f"[rows,{n}]", "out")), scalars=(Scalar("rows"),), key=op.key())
    return op, program


def run_shape(spec: str, rows: int, kind: str, samples: int, seed: int, flush_mode: str = "zero",
              rows2: int = 16, row_scales: bool = False) -> list[dict]:
    import torch
    n, k = (int(v) for v in spec.split("x"))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(2 * l2, device="cuda", dtype=torch.uint8)
    x = torch.randn(max(16, rows, rows2), k, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    if row_scales:
        s = torch.rand(n, k // 128, device="cuda", generator=gen) * 2e-4 + 1e-4
    else:
        s = torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda", generator=gen) * 2e-4 + 1e-4
    records = []
    for cfg in variants_for(kind, n, k, rows):
        try:
            op, program = _compile(n, k, rows, cfg, row_scales)
            out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            wide = torch.empty(rows2, n, device="cuda", dtype=torch.bfloat16)

            def run(program=program, out=out, rows=rows):
                program.launch(x, w, s, out, scalars=(rows,))

            median, best, digest = _time(run, samples, flush, [out], flush_mode)
            median_read, best_read, digest_read = _time(run, samples, flush, [out], "read")
            assert digest_read == digest, "output bits moved with the eviction mode"

            # The same program at a wider batch (max_rows covers both): the wide
            # batch is where ``live_tiles`` and the fragment budget start to bite.
            wide_rec = {}
            if rows2 != rows:
                def run_wide(program=program, dest=wide, rows2=rows2):
                    program.launch(x, w, s, dest, scalars=(rows2,))

                w_med, w_best, w_digest = _time(run_wide, samples, flush, [wide], flush_mode)
                w_read, _, w_digest_read = _time(run_wide, samples, flush, [wide], "read")
                assert w_digest_read == w_digest, "wide output bits moved with the eviction mode"
                wide_rec = dict(median_us=round(w_med, 3), weight_gbps=round(n * k / w_med / 1e3, 1),
                                median_read_us=round(w_read, 3),
                                weight_gbps_read=round(n * k / w_read / 1e3, 1),
                                digest=w_digest)

            # Row invariance inside this variant: row 0 at the primary width vs
            # row 0 embedded at the wider batch.
            def run_wide_out(program=program):
                program.launch(x, w, s, wide, scalars=(rows2,))

            run_wide_out()
            torch.cuda.synchronize()
            row0_a = wide.view(torch.int16)[0].clone()
            run()
            torch.cuda.synchronize()
            row0_b = out.view(torch.int16)[0].clone()
            rec = dict(case=f"fp8gemv_{spec}", rows=rows, variant=cfg["name"],
                       warps=cfg["warps"], groups=cfg["groups"],
                       load_hint=cfg.get("load_hint", "stream"),
                       kernel=cfg.get("kernel", "production"),
                       cta_tiles=cfg.get("cta_tiles"),
                       median_us=round(median, 3), min_us=round(best, 3),
                       weight_gbps=round(n * k / median / 1e3, 1),
                       median_read_us=round(median_read, 3),
                       weight_gbps_read=round(n * k / median_read / 1e3, 1),
                       digest=digest,
                       row_invariant=bool(torch.equal(row0_a, row0_b)))
            if wide_rec:
                rec["wide_rows"] = rows2
                rec["wide"] = wide_rec
            records.append(rec)
            del op, program, out, wide
        except Exception as exc:  # a failed variant is a null, not a dead sweep
            records.append(dict(case=f"fp8gemv_{spec}", rows=rows, variant=cfg["name"],
                                error=repr(exc)[:400]))
        torch.cuda.empty_cache()
    base = next((rec for rec in records
                 if rec["variant"] == "baseline" and "median_us" in rec), None)
    if base is None:
        for rec in records:
            rec["error"] = rec.get("error", "no baseline") + " | no usable baseline"
        return records
    for rec in records:
        if "median_us" not in rec:
            continue
        rec["ratio"] = round(base["median_us"] / rec["median_us"], 4)
        rec["ratio_read"] = round(base["median_read_us"] / rec["median_read_us"], 4)
        rec["exact_vs_baseline"] = rec["digest"] == base["digest"]
        if "wide" in rec:
            wide_base = base["wide"]
            rec["wide"]["ratio"] = round(wide_base["median_us"] / rec["wide"]["median_us"], 4)
            rec["wide"]["ratio_read"] = round(wide_base["median_read_us"] / rec["wide"]["median_read_us"], 4)
            rec["wide"]["exact_vs_baseline"] = rec["wide"]["digest"] == wide_base["digest"]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="qwen", help=f"{','.join(SETS)}, or a semicolon list of NxK")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--variants", default="groups",
                        choices=("groups", "groups_wide", "hints", "depth", "combo", "prod", "gem", "wide"))
    parser.add_argument("--flush", default="zero", choices=("zero", "read"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--row-scales", action="store_true",
                        help="per-row FP32 scales [N, K/128] instead of the 128x128 block grid")
    parser.add_argument("--rows2", type=int, default=16,
                        help="second batch width timed on the same program (0 disables)")
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
        for rec in run_shape(spec, args.rows, args.variants, args.samples, args.seed, args.flush,
                             args.rows2 if args.rows2 > 0 else args.rows, args.row_scales):
            rec["tag"] = args.tag
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
