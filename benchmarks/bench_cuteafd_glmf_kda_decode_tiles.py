#!/usr/bin/env python3
"""Time GLM 5.3 Flash's two BF16 KDA projections alone against ``KDA_DECODE_TILES``, the TMA tiles
``compile_glmf_kda_aot(decode_tiles=True)`` picks by a step's live rows.

One program per projection and arm, at the real geometry (in-projection ``[24896, 4096]``, o_proj
``[4096, 8192]``; random BF16 weights and activations):

  default  ``glm_projection(n, k)``, as the exported decode KDA programs build it: the skinny GEMV
           to 8 rows, ``prefill_config``'s TMA tile above
  tiled    ``glm_projection(n, k, tiles_by_rows=KDA_DECODE_TILES[(n, k)])``: the same GEMV to 8
           rows, then the table's tile for the live rows

Per row count: the median of CUDA-event timings of one launch, the L2 flushed before every launch
and the GPU held busy while the host enqueues (``bench_cuteafd_glm_programs.time_us``), the arms
taken in turn within every iteration. The weights are therefore L2-cold, without the engine's L2
prefetch. Before timing, the tiled arm's outputs are compared with the default's on integer views
(outputs start as NaN; a difference, or a default output left unwritten, is marked ``!`` and fails
the run). One table per projection (us per launch; tiled / default below 1 is a gain), both
projections over the 34 KDA layers as ms a decode step and, with ``--json``, the same numbers with
each minimum. Component timings of two kernels, not tok/s.

``--candidates`` also times every tile the tables choose from, and ``prefill_config``'s, alone
(``TmaBf16Projection``) at every row count, each compared with ``prefill_config``'s bits.
``KDA_DECODE_TILES`` was tuned on an RTX 5090 (170 SMs); a table for another card is read off those
columns: per row range the fastest unmarked tile, as ``(rows up to, compute_warps, tile_n,
num_stages)`` in ascending rows, the first entry past the GEMV's 8 rows. A tile is printed as tile
rows x tile columns / stages: ``16x32/4`` is ``compute_warps`` 1, ``tile_n`` 32, 4 stages.

  python benchmarks/bench_cuteafd_glmf_kda_decode_tiles.py [--json out.json] [--iters 30] [--candidates]
      [--rows 1,8,9,12,16,24,32,48,64,96,127,128] [--projections in,out]
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

KDA_LAYERS = 34  # GLM 5.3 Flash's KDA layers: a decode step launches both projections once in each
TITLES = {"in": "in-projection", "out": "o_proj"}


def time_us(launches, iters: int, flush: torch.Tensor) -> list[tuple[float, float]]:
    """``bench_cuteafd_glm_programs.time_us`` over several launches taken in turn (one of each per
    iteration, so the arms interleave): (median, minimum) us each."""
    for launch in launches:
        launch()
    torch.cuda.synchronize()
    times = [[] for _ in launches]
    for _ in range(iters):
        for arm, launch in enumerate(launches):
            flush.max()  # read-only L2 flush: no dirty lines to write back inside the launch
            torch.cuda._sleep(3_000_000)  # covers the host-side launch
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            end.synchronize()
            times[arm].append(start.elapsed_time(end) * 1000.0)
    return [(sorted(arm)[len(arm) // 2], min(arm)) for arm in times]


def program(launch, n: int, k: int, key: tuple):
    """``launch(x, w, out, rows, stream)`` as its own program."""
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    return compile_program(
        launch, name="bench_glmf_kda_decode_tiles",
        operands=(Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.bfloat16, f"[{n},{k}]"),
                  Operand("out", torch.bfloat16, f"[rows,{n}]", "out")),
        scalars=(Scalar("rows"),), key=key)


def tile_of(config: dict) -> tuple:
    return config["compute_warps"], config["tile_n"], config["num_stages"]


def route(launch, rows: int):
    """What a routed projection launches at ``rows`` live rows: None for the skinny GEMV, else the TMA
    tile as ``(compute_warps, tile_n, num_stages)``."""
    if rows <= launch.max_skinny_rows:
        return None
    return next((entry[1:] for entry in launch.tiles_by_rows if rows <= entry[0]), tile_of(launch.large.config))


def tile_name(tile) -> str:
    return "GEMV" if tile is None else f"{16 * tile[0]}x{tile[1]}/{tile[2]}"


def run(launch, out: torch.Tensor) -> torch.Tensor:
    out.fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    return out.clone()


def same_bits(want: torch.Tensor, got: torch.Tensor) -> bool:
    """Every output of ``want`` written, and ``got`` its bit patterns (signed zeros count)."""
    return not bool(torch.isnan(want).any()) and bool(torch.equal(want.view(torch.int16), got.view(torch.int16)))


def mark(entry: dict) -> str:
    return " " if entry["same_bits"] else "!"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="also write the results here")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rows", default="1,8,9,12,16,24,32,48,64,96,127,128", help="live rows of a launch")
    parser.add_argument("--projections", default="in,out")
    parser.add_argument("--candidates", action="store_true",
                        help="also time each tile of the tables, and prefill_config's, alone")
    args = parser.parse_args()
    rows_list = [int(v) for v in args.rows.split(",")]
    names = args.projections.split(",")
    if args.iters < 1 or min(rows_list) < 1:
        parser.error("--iters and --rows take positive numbers")
    if any(name not in TITLES for name in names) or len(set(names)) != len(names):
        parser.error("--projections takes in and/or out")

    from b12x._lib.compiler import b12x_package_fingerprint
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd._glm_kernels import glm_projection
    from b12x.integration.cuteafd.glmf import KDA_DECODE_TILES

    dev = "cuda"
    shapes = {"in": (g.kda_in_width, g.hidden), "out": (g.hidden, g.kda_width)}
    gen = torch.Generator(device=dev).manual_seed(0)
    flush = torch.zeros(512 << 20, dtype=torch.uint8, device=dev)
    props = torch.cuda.get_device_properties(0)
    table_tiles = list(dict.fromkeys(entry[1:] for name in TITLES for entry in KDA_DECODE_TILES[shapes[name]]))
    # The case an arm's outputs are compared with: the default arm, or prefill_config's tile alone.
    reference = {"default": 0, "tiled": 0, "tile": 2}

    results, projections = [], {}
    for name in names:
        n, k = shapes[name]
        w = (torch.randn((n, k), generator=gen, device=dev) / 16).bfloat16()
        x = torch.randn((max(rows_list), k), generator=gen, device=dev).bfloat16()
        plain, tiled = glm_projection(n, k), glm_projection(n, k, tiles_by_rows=KDA_DECODE_TILES[(n, k)])
        routed = [("default", plain, program(plain, n, k, plain.key())),
                  ("tiled", tiled, program(tiled, n, k, tiled.key()))]
        # --candidates: prefill_config's tile first (the reference of the others), then the tables'.
        prefill_tile = tile_of(plain.large.config)
        tiles = [prefill_tile] + [tile for tile in table_tiles if tile != prefill_tile] if args.candidates else []
        alone = [("tile", tile, program(TmaBf16Projection(n, k, compute_warps=tile[0], tile_n=tile[1],
                                                          num_stages=tile[2]), n, k, ("tile", n, k, tile)))
                 for tile in tiles]
        projections[name] = {"n": n, "k": k, "skinny_rows": plain.max_skinny_rows, "prefill_tile": prefill_tile,
                             "tiles_by_rows": tiled.tiles_by_rows, "candidates": tiles}
        for rows in rows_list:
            out = torch.empty((rows, n), dtype=torch.bfloat16, device=dev)
            cases = [(arm, route(launch, rows), prog) for arm, launch, prog in routed] + alone
            launches = [partial(prog.launch, x[:rows], w, out, scalars=(rows,)) for _, _, prog in cases]
            outputs = [run(launch, out) for launch in launches]
            same = [same_bits(outputs[reference[arm]], got) for (arm, _, _), got in zip(cases, outputs)]
            timed = time_us(launches, args.iters, flush)
            for (arm, tile, _), ok, (median, low) in zip(cases, same, timed):
                results.append({"projection": name, "rows": rows, "arm": arm, "tile": tile, "us": round(median, 2),
                                "min_us": round(low, 2), "same_bits": ok})

    cells = {}
    for e in results:
        cells.setdefault((e["projection"], e["rows"]), {})[e["tile"] if e["arm"] == "tile" else e["arm"]] = e
    step = []
    if len(names) == len(TITLES):
        for rows in rows_list:
            ms = [sum(cells[name, rows][arm]["us"] for name in names) * KDA_LAYERS / 1000.0
                  for arm in ("default", "tiled")]
            step.append({"rows": rows, "default_ms": round(ms[0], 3), "tiled_ms": round(ms[1], 3)})
    status = "passed" if all(e["same_bits"] for e in results) else "mismatch"
    report = {
        "benchmark": "bench_cuteafd_glmf_kda_decode_tiles", "command": sys.argv, "device": props.name,
        "sms": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
        "b12x_fingerprint": b12x_package_fingerprint(), "iters": args.iters, "timing": "L2-cold median, CUDA events",
        "geometry": {"kda_layers": KDA_LAYERS, "projections": projections},
        "status": status, "results": results, "step": step,
    }
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{props.name}, {props.multi_processor_count} SMs, {args.iters} iterations: GLM 5.3 Flash's BF16 KDA "
          f"projections alone, us per launch (L2-cold median); tiles as rows x columns / stages")
    for name in names:
        p = projections[name]
        title = f"{TITLES[name]} [{p['n']}, {p['k']}]"
        print(f"{title}: default (GEMV to {p['skinny_rows']} rows, {tile_name(p['prefill_tile'])} above) against "
              f"tiles_by_rows {p['tiles_by_rows']}")
        print(f"  {'rows':>4s} {'default':>9s}  {'tiled':>9s}  {'tiled / default':>15s}  tiled route")
        for rows in rows_list:
            a, b = cells[name, rows]["default"], cells[name, rows]["tiled"]
            print(f"  {rows:4d} {a['us']:9.1f}{mark(a)} {b['us']:9.1f}{mark(b)} {b['us'] / a['us']:15.2f}  "
                  f"{tile_name(b['tile'])}")
        if p["candidates"]:
            print(f"{title}: each tile alone ({tile_name(p['prefill_tile'])} is prefill_config's)")
            print(f"  {'rows':>4s} " + " ".join(f"{tile_name(tile):>9s} " for tile in p["candidates"]) + "  fastest")
            for rows in rows_list:
                row = cells[name, rows]
                unmarked = [tile for tile in p["candidates"] if row[tile]["same_bits"]]
                best = tile_name(min(unmarked, key=lambda tile: row[tile]["us"])) if unmarked else "-"
                print(f"  {rows:4d} " + " ".join(f"{row[tile]['us']:9.1f}{mark(row[tile])}" for tile in p["candidates"])
                      + f"  {best}")
    if step:
        print(f"both projections x {KDA_LAYERS} KDA layers, ms a decode step")
        print(f"  {'rows':>4s} {'default':>9s}  {'tiled':>9s}  {'saved':>9s}")
        for s in step:
            print(f"  {s['rows']:4d} {s['default_ms']:9.2f}  {s['tiled_ms']:9.2f}  "
                  f"{s['default_ms'] - s['tiled_ms']:9.2f}")
    if status != "passed":
        print("MISMATCH: a cell marked '!' differs in bits from its reference (the default arm; for a tile alone, "
              "prefill_config's tile), or its reference left outputs unwritten")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
