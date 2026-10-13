#!/usr/bin/env python3
"""Time GLM 5.3 Flash's token-sequential KDA recurrence alone against ``seq_ctas``, the CTAs per head
block a step's sequences are shared among (``GlmfKdaRecurrent``: the kernel a ``glmf_kda_*`` decode
program launches once per KDA layer per step).

One program per state type and ``seq_ctas``: the kernel at the real geometry (64 heads of 128) over
the decode program's operand strides (q|k|v rows of ``3D``, the decay gate rows inside ``[rows, 2,
D]``, beta inside the ``P``-wide in-projection rows), a 128-row replay record and a 16-slot state
pool. Per step shape (sequences x rows per sequence, every sequence in its own slot), state type and
``spec``: the median of CUDA-event timings, the L2 flushed before every launch and the GPU held busy
while the host enqueues (``bench_cuteafd_glm_programs.time_us``), the ``seq_ctas`` programs taken in
turn within every iteration; a plain step (``spec`` 0) stores its state, so the pool is restored
before every measured launch, outside the timed region. Before timing, every ``seq_ctas`` > 1
program's outputs, state pool and replay record are compared with ``seq_ctas=1``'s, byte for byte
(a difference is marked ``!`` and fails the run). One table (us per launch) and, with ``--json``, the
same numbers with each minimum. Component timings of one kernel, not tok/s.

  python benchmarks/bench_cuteafd_glmf_kda_recurrent.py [--json out.json] [--iters 30]
      [--seq-ctas 1,2,4,8,16] [--shapes 1x1,1x8,1x16,4x8,8x8,16x1,16x4,16x8] [--states fp32,bf16] [--spec 0,1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

RECORD_ROWS = 128
SLOTS = 16
STATES = {"fp32": "float32", "bf16": "bfloat16"}


def time_us(fns, iters: int, flush: torch.Tensor, prepare=None) -> list[tuple[float, float]]:
    """``bench_cuteafd_glm_programs.time_us`` over several launches taken in turn (one of each per
    iteration, so the arms interleave), ``prepare`` run before every flush: (median, minimum) us each."""
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    times = [[] for _ in fns]
    for _ in range(iters):
        for arm, fn in enumerate(fns):
            if prepare is not None:
                prepare()
            flush.max()  # L2 flush, after the restore: no dirty lines to write back inside fn
            torch.cuda._sleep(3_000_000)  # covers the host-side launch of fn
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times[arm].append(start.elapsed_time(end) * 1000.0)
    return [(sorted(arm)[len(arm) // 2], min(arm)) for arm in times]


def build(state: str, seq_ctas: int):
    """``GlmfKdaRecurrent`` alone, with ``glmf_kda``'s strides, as a program of ``seq_ctas``."""
    import cutlass
    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._glmf_kernels import GlmfKdaRecurrent, kda_replay_layout

    h, d, p = g.kda_heads, g.kda_width, g.kda_in_width
    cute_type, torch_type = {"float32": (cutlass.Float32, torch.float32),
                             "bfloat16": (cutlass.BFloat16, torch.bfloat16)}[state]
    launch = GlmfKdaRecurrent(heads=h, lower_bound=g.gate_lower_bound, qkv_width=3 * d, g_stride=2 * d, b_stride=p,
                              state_dtype=cute_type, replay_rows=RECORD_ROWS, seq_ctas=seq_ctas)
    return compile_program(
        launch, name="bench_glmf_kda_recurrent",
        operands=(Operand("qkv", torch.bfloat16, f"[rows,{3 * d}]"),
                  Operand("g_raw", torch.bfloat16, f"[rows,2,{d}]"),
                  Operand("b_raw", torch.bfloat16, f"[rows,{p}]", align=2),
                  Operand("a_log", torch.float32, f"[{h}]", align=4),
                  Operand("dt_bias", torch.float32, f"[{d}]", align=4),
                  Operand("state", torch_type, f"[slots,{h},128,128]", "inout"),
                  Operand("slots", torch.int32, "[rows]", align=4),
                  Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
                  Operand("replay", torch.float32, f"[{kda_replay_layout(h, 3 * d, RECORD_ROWS)[2] // 4}]", "inout")),
        scalars=(Scalar("spec"), Scalar("rows")), key=(state, RECORD_ROWS, seq_ctas))


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="also write the results here")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seq-ctas", default="1,2,4,8,16")
    parser.add_argument("--shapes", default="1x1,1x8,1x16,4x8,8x8,16x1,16x4,16x8", help="sequences x rows each")
    parser.add_argument("--states", default="fp32,bf16")
    parser.add_argument("--spec", default="0,1")
    args = parser.parse_args()
    seq_ctas = [int(v) for v in args.seq_ctas.split(",")]
    shapes = [tuple(int(v) for v in shape.split("x")) for shape in args.shapes.split(",")]
    states = args.states.split(",")
    specs = [int(v) for v in args.spec.split(",")]
    if 1 not in seq_ctas:
        parser.error("--seq-ctas needs 1, the reference")
    if any(state not in STATES for state in states):
        parser.error("--states takes fp32 and/or bf16")
    if any(s < 1 or r < 1 or s > SLOTS or s * r > RECORD_ROWS for s, r in shapes):
        parser.error(f"--shapes: at most {SLOTS} sequences and {RECORD_ROWS} rows a step")

    from b12x._lib.compiler import b12x_package_fingerprint
    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    dev = "cuda"
    h, d, p = g.kda_heads, g.kda_width, g.kda_in_width
    gen = torch.Generator(device=dev).manual_seed(0)
    r = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device=dev) * s  # noqa: E731
    flush = torch.zeros(512 << 20, dtype=torch.uint8, device=dev)
    # The decode program's buffers: conv outputs q|k|v, f|gate rows, the in-projection rows holding beta.
    qkv = r(RECORD_ROWS, 3 * d, s=0.5).bfloat16()
    gate = r(RECORD_ROWS, 2 * d, s=2.0)
    gate[:, : d // 4] -= 12.0  # a quarter of the decay gates far below zero (multipliers within an ulp of 1)
    gate = gate.bfloat16()
    proj = r(RECORD_ROWS, p).bfloat16()
    b_raw = proj[:, 3 * d + 256:]
    a_log = torch.log(torch.rand((h,), generator=gen, device=dev) * 15 + 1).float()
    dt_bias = r(d, s=0.1).float()
    pool16 = r(SLOTS, h, 128, 128, s=0.05).bfloat16()
    out = torch.zeros((RECORD_ROWS, d), dtype=torch.bfloat16, device=dev)
    record = torch.zeros((kda_replay_layout(h, 3 * d, RECORD_ROWS)[2] // 4,), device=dev)
    props = torch.cuda.get_device_properties(0)

    results, mismatches = [], []
    for state_name in states:
        initial = pool16 if state_name == "bf16" else pool16.float()
        state = initial.clone()
        programs = {z: build(STATES[state_name], z) for z in seq_ctas}
        for spec in specs:
            for sequences, each in shapes:
                rows = sequences * each
                slots = torch.arange(sequences, dtype=torch.int32, device=dev).repeat_interleave(each).contiguous()

                def launch(z, slots=slots, rows=rows, spec=spec):
                    programs[z].launch(qkv, gate, b_raw, a_log, dt_bias, state, slots, out, record,
                                       scalars=(spec, rows))

                def restore():
                    state.copy_(initial)

                def once(z):
                    restore()
                    out.zero_()
                    record.zero_()
                    launch(z)
                    torch.cuda.synchronize()
                    return out[:rows].clone(), state.clone(), record.clone()

                want = once(1)
                same = {z: all(bool(torch.equal(bits(a), bits(b))) for a, b in zip(want, once(z))) for z in seq_ctas}
                timed = time_us([lambda z=z: launch(z) for z in seq_ctas], args.iters, flush,
                                None if spec else restore)
                for z, (median, low) in zip(seq_ctas, timed):
                    entry = {"state": state_name, "spec": spec, "sequences": sequences, "rows_per_sequence": each,
                             "rows": rows, "seq_ctas": z, "ctas": h * 4 * z, "us": round(median, 2),
                             "min_us": round(low, 2), "same_bytes_as_seq_ctas_1": same[z]}
                    results.append(entry)
                    if not same[z]:
                        mismatches.append(entry)
                restore()

    report = {
        "benchmark": "bench_cuteafd_glmf_kda_recurrent", "command": sys.argv, "device": props.name,
        "sms": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
        "b12x_fingerprint": b12x_package_fingerprint(), "iters": args.iters, "timing": "L2-cold median, CUDA events",
        "geometry": {"heads": h, "head_dim": g.kda_head_dim, "qkv_width": 3 * d, "g_stride": 2 * d, "b_stride": p,
                     "replay_rows": RECORD_ROWS, "state_slots": SLOTS},
        "status": "mismatch" if mismatches else "passed", "results": results,
    }
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{props.name}, {props.multi_processor_count} SMs, {args.iters} iterations: GlmfKdaRecurrent alone, "
          f"us per launch (L2-cold median), grid (64, 4, seq_ctas)")
    print(f"  {'state':5s} {'spec':>4s} {'seqs x rows':>11s} {'rows':>4s} "
          + " ".join(f"{'seq_ctas=' + str(z):>11s} " for z in seq_ctas))
    table = {}
    for e in results:
        table.setdefault((e["state"], e["spec"], e["sequences"], e["rows_per_sequence"]), {})[e["seq_ctas"]] = e
    for (state_name, spec, sequences, each), row in table.items():
        cells = " ".join(f"{row[z]['us']:11.1f}{' ' if row[z]['same_bytes_as_seq_ctas_1'] else '!'}" for z in seq_ctas)
        print(f"  {state_name:5s} {spec:4d} {f'{sequences} x {each}':>11s} {sequences * each:4d} {cells}")
    if mismatches:
        print(f"MISMATCH: {len(mismatches)} case(s) marked '!' differ from seq_ctas=1 in outputs, state or record")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
