"""EXLR8 <=64-row Spark serving benchmark (duel S3).

The shipped serving path is the K1 core (``exlr8_a16_serving``) plus the
64-row grouping (``exlr8_group``).  This harness measures that baseline and
scheduling-only arms on the real served geometry (TP4/TP6 rotated unit
ownership, real K2/K3/K4 tier mix ``(2, 3, 4)[(u + p) % 3]``), with a bitwise
oracle against the frozen baseline program, row invariance, graph-replay
stability and interleaved graph-replay timing.

Arms are scheduling/data-movement only; no arm edits dequantization, MMA op
order, the ascending K1024 split fold or output rounding.

  base        shipped K1 + group, cta_limit = sms * 4 (shipped policy)
  ctaN        same programs, cta_limit = sms * N (launch scalar only)
  grouped16   compact owned-unit grouped kernel at rows <= 64, group_rows 16
  grouped64   compact owned-unit warp-specialized grouped kernel, group_rows 64
  wired       the shipped dispatch: the 64-row ``exlr8_a16_grouped64`` core
              (F32 FC2 partials) plus the 64-row ``exlr8_group`` glue, both at
              the runtime CTA limits the engine derives
  k1compact   K1's serving ABI (F32 partials unchanged) with compact owned-unit
              enumeration: only owned units are enumerated, so a rank that owns
              three of sixteen units fills the grid instead of pinning each CTA
              to one (unit, proj, split) family
  k23         the S3 coalesced K2/K3 routed serving core at the shipped ABI
              and CTA policy; its output must stay bit-identical to K1

The grouped arms reuse the shipped grouped prefill kernel and grouping
verbatim; the compact ``unit_count``/``unit_map``/``unit_limit`` tables are
the ones its own ABI already carries.  Every arm's output must be bit-equal
to the frozen baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel

from benchmark_exlr8_a16 import (
    capture,
    finite_nonzero,
    load_route_waves,
    make_wire,
    rank_geometry,
    read_roofline,
    real_route_ids,
    timing,
)

HIDDEN = 6144
EXPERTS = 256
TOP_K = 8
UNITS = 16
MAX_GROUPS = 512
# Same capacity as the shipped K1 serving program: at <=64 rows a rank can
# still pack more than one group per expert, so the grouped arm must not be
# given a smaller group capacity than production.
GROUPED_MAX_GROUPS = 512
SOURCE_PATHS = ("b12x/moe/_shared/kernels/w4a16/exlr8.py",
                "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py",
                "b12x/integration/cuteafd/exlr8.py",
                "b12x/integration/cuteafd/exlr8_glue.py")
ROW_SLICE = (1, 17, 63, 64)

_SCRATCH: dict = {}


def source_digests(root):
    root = Path(root)
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in SOURCE_PATHS}


def tier_width(unit, projection):
    """The shipped checkpoint's routed tier mix for a (unit, projection) slot."""
    return (2, 3, 4)[(unit + projection) % 3]


def owned_units(world, rank, layer):
    owned, _ = rank_geometry(world, rank, layer, experts=EXPERTS, units=UNITS)
    return owned


def make_tables(owned, device="cuda"):
    """Synthetic payloads sized for each slot's real tier width."""
    pointers = torch.zeros((EXPERTS, UNITS, 3, 3), dtype=torch.int64)
    keep = []
    unit_bytes = HIDDEN * 16
    slot_bytes = {}
    # `pointers[..., 1]` is the per-H128 activation scale the activation stage
    # reads for gate/up/down; one shared unit-scale table serves every slot
    # (the harness only needs identical inputs across arms).
    act_scales = torch.ones(128, dtype=torch.float16, device=device)
    keep.append(act_scales)
    for e in range(EXPERTS):
        for u in owned[e]:
            for p in range(3):
                width = tier_width(u, p)
                payload = torch.randint(0, 256, (unit_bytes * width,), dtype=torch.uint8,
                                        device=device)
                scales = torch.full((128,), 1 / HIDDEN ** 0.5, dtype=torch.float16,
                                    device=device)
                keep.extend((payload, scales))
                pointers[e, u, p] = torch.tensor(
                    [payload.data_ptr(), act_scales.data_ptr(), scales.data_ptr()])
                slot_bytes[(e, u, p)] = unit_bytes * width
    bits = torch.zeros((EXPERTS, UNITS, 3), dtype=torch.int32)
    for e in range(EXPERTS):
        for u in owned[e]:
            for p in range(3):
                bits[e, u, p] = tier_width(u, p)
    payload_bytes = sum(slot_bytes.values())
    return dict(tables=pointers.to(device), bits=bits.to(device), keep=keep,
                slot_bytes=slot_bytes, payload_bytes=payload_bytes)


def unit_tables(owned, device="cuda"):
    counts = torch.tensor([len(u) for u in owned], dtype=torch.int32)
    table = torch.zeros((EXPERTS, UNITS), dtype=torch.int32)
    for e, units in enumerate(owned):
        table[e, :len(units)] = torch.tensor(units, dtype=torch.int32)
    return counts.to(device), table.to(device), int(counts.max())


def routed_bytes(tables, ids):
    """Weight bytes a rank with this ownership streams for the routed experts."""
    experts = ids.reshape(-1).unique().tolist()
    slots = tables["slot_bytes"]
    total = 0
    for e in experts:
        for u in range(UNITS):
            for p in range(3):
                total += slots.get((e, u, p), 0)
    return total


def alloc_scratch(program, device="cuda"):
    sizes = program.scratch_bytes(64)
    key = (program.name, tuple(sorted(sizes.items())))
    if key not in _SCRATCH:
        _SCRATCH[key] = {name: torch.empty(size, dtype=torch.uint8, device=device)
                         for name, size in sizes.items()}
    return dict(_SCRATCH[key])


def launch_program(program, values, scalars):
    tensors = [values[operand.name] for operand in program.operands]
    program.launch(*tensors, scalars=scalars)


class Arm:
    """A compiled program set, its buffers and its ordered launches.

    ``lossy`` arms change output rounding/partial precision, so they are
    measured and reported but never pass the bit-exactness gate.
    """

    def __init__(self, name, programs, buffers, launches, notes, lossy=False):
        self.name = name
        self.programs = programs
        self.buffers = buffers
        self.launches = launches
        self.notes = notes
        self.lossy = lossy

    def run(self, rows):
        for launch in self.launches:
            launch(rows)


def base_arm(cta_limit, rows_max, tables, shared, device="cuda",
             variant="baseline", arm_name="base"):
    k1 = compile_exlr8_a16_aot(hidden=HIDDEN, units=UNITS, top_k=TOP_K,
                               max_groups=MAX_GROUPS, device_count=True,
                               serving_variant=variant)
    group = compile_exlr8_group_aot(experts=EXPERTS, top_k=TOP_K,
                                    max_groups=MAX_GROUPS, hidden=HIDDEN,
                                    max_rows=64, group_rows=16)
    buffers = dict(
        group_expert=torch.full((MAX_GROUPS,), -1, dtype=torch.int32, device=device),
        route_slots=torch.full((MAX_GROUPS, 16), -1, dtype=torch.int32, device=device),
        route_group_lane=torch.full((rows_max, TOP_K), -1, dtype=torch.int32, device=device),
        group_count=torch.zeros(1, dtype=torch.int32, device=device),
        activation=torch.empty((rows_max, HIDDEN), dtype=torch.float16, device=device),
        signs_out=shared["signs_out"],
        signs_in=shared["signs_in"],
        output=torch.empty((rows_max, HIDDEN), dtype=torch.bfloat16, device=device),
        route_weights=shared["weights"],
        ids=shared["ids"],
        pointers=tables["tables"],
        bits=tables["bits"],
    )
    buffers.update(alloc_scratch(k1, device))
    group_values = dict(buffers, x=shared["wire"])
    core_values = dict(buffers, x=buffers["activation"])

    def group_launch(rows):
        launch_program(group, group_values, (rows, cta_limit))

    def core_launch(rows):
        launch_program(k1, core_values, (rows, cta_limit))

    notes = dict(kind="k1" if arm_name == "base" else arm_name,
                 serving_variant=variant, cta_limit=cta_limit, threads=128,
                 min_blocks_per_mp=k1.geometry["ctas_per_sm"],
                 group_rows=16, max_groups=MAX_GROUPS,
                 split_k_blocks=8, fc1_splits=k1.geometry["fc1_splits"],
                 program=k1.name, group_program=group.name,
                 grid_clamp="min(max_groups*units*2*splits, cta_limit) fc1; "
                            "min(max_groups*units, cta_limit) activation; "
                            "min(max_groups*blocks, cta_limit) fc2; "
                            "min((rows*blocks+3)//4, cta_limit) reduce")
    return Arm(arm_name, (k1, group), buffers, (group_launch, core_launch), notes)


def grouped_arm(name, group_rows, cta_limit, rows_max, tables, shared, owned,
                half=False, device="cuda"):
    # The shipped grouped package stores FC2 partials as F16; the <=64 serving
    # kernel keeps them F32.  F32 partials are the matching-numerics variant;
    # the shipped F16 setting is measured only as a separately labelled lossy
    # arm.
    core = compile_exlr8_a16_aot(hidden=HIDDEN, units=UNITS, top_k=TOP_K,
                                 max_groups=GROUPED_MAX_GROUPS, device_count=True,
                                 grouped=True, group_rows=group_rows,
                                 fc2_partials="f16" if half else "f32")
    group = compile_exlr8_group_aot(experts=EXPERTS, top_k=TOP_K,
                                    max_groups=GROUPED_MAX_GROUPS, hidden=HIDDEN,
                                    max_rows=4096, group_rows=group_rows)
    unit_count, unit_map, unit_limit = unit_tables(owned, device)
    from benchmarks.bench_exlr8_grouped_prefill import runtime_l2_bytes
    l2_bytes = runtime_l2_bytes()
    buffers = dict(
        group_expert=torch.full((GROUPED_MAX_GROUPS,), -1, dtype=torch.int32, device=device),
        route_slots=torch.full((GROUPED_MAX_GROUPS, group_rows), -1, dtype=torch.int32,
                               device=device),
        route_group_lane=torch.full((rows_max, TOP_K), -1, dtype=torch.int32, device=device),
        group_count=torch.zeros(1, dtype=torch.int32, device=device),
        activation=torch.empty((rows_max, HIDDEN), dtype=torch.float16, device=device),
        signs_out=shared["signs_out"],
        signs_in=shared["signs_in"],
        output=torch.empty((rows_max, HIDDEN), dtype=torch.bfloat16, device=device),
        route_weights=shared["weights"],
        ids=shared["ids"],
        unit_count=unit_count,
        unit_map=unit_map,
        pointers=tables["tables"],
        bits=tables["bits"],
    )
    buffers.update(alloc_scratch(core, device))
    group_values = dict(buffers, x=shared["wire"])
    core_values = dict(buffers, x=buffers["activation"])

    def group_launch(rows):
        launch_program(group, group_values, (rows, cta_limit))

    def core_launch(rows):
        launch_program(core, core_values, (rows, cta_limit, unit_limit, l2_bytes))

    notes = dict(kind="grouped", cta_limit=cta_limit, threads=256,
                 min_blocks_per_mp=Exlr8GroupedKernel.CTAS_PER_SM,
                 group_rows=group_rows, max_groups=GROUPED_MAX_GROUPS,
                 unit_limit=unit_limit, fc1_ws=bool(group_rows == 64),
                 fc2_partials="f16 (shipped)" if half else "f32 (matches K1)",
                 unit_enumerated="owned only (unit_count/unit_map, unit_limit=%d)"
                                 % unit_limit,
                 program=core.name, group_program=group.name)
    return Arm(name, (core, group), buffers, (group_launch, core_launch), notes,
               lossy=bool(half))


def wired_arm(name, rows_max, tables, shared, owned, device="cuda"):
    """The shipped decode dispatch exactly as the engine admits it.

    Below the K1 threshold the engine runs the 64-row grouped instance
    (``exlr8_a16_grouped64``: 16-row groups, F32 FC2 partials so output stays
    bit-identical to K1) paired with the shipped 64-row ``exlr8_group`` glue.
    Each launch uses the runtime limit the engine derives from that export's
    own ``ctas_per_sm`` hint, so this arm is the production package rather
    than a harness-only configuration.
    """
    if rows_max > 64:
        raise ValueError("the wired decode instance serves at most 64 rows")
    built = time.monotonic()
    core = compile_exlr8_a16_aot(hidden=HIDDEN, units=UNITS, top_k=TOP_K,
                                 max_groups=MAX_GROUPS, device_count=True,
                                 grouped=True, group_rows=16, grouped_rows=64,
                                 fc2_partials="f32")
    group = compile_exlr8_group_aot(experts=EXPERTS, top_k=TOP_K,
                                    max_groups=MAX_GROUPS, hidden=HIDDEN,
                                    max_rows=64, group_rows=16)
    build_seconds = time.monotonic() - built
    if core.name != "exlr8_a16_grouped64" or group.name != "exlr8_group":
        raise RuntimeError(f"wired pairing built {core.name} + {group.name}")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    core_cta = sms * Exlr8GroupedKernel.CTAS_PER_SM
    group_cta = sms * int(group.geometry["ctas_per_sm"])
    unit_count, unit_map, unit_limit = unit_tables(owned, device)
    buffers = dict(
        group_expert=torch.full((MAX_GROUPS,), -1, dtype=torch.int32, device=device),
        route_slots=torch.full((MAX_GROUPS, 16), -1, dtype=torch.int32, device=device),
        route_group_lane=torch.full((rows_max, TOP_K), -1, dtype=torch.int32, device=device),
        group_count=torch.zeros(1, dtype=torch.int32, device=device),
        activation=torch.empty((rows_max, HIDDEN), dtype=torch.float16, device=device),
        signs_out=shared["signs_out"],
        signs_in=shared["signs_in"],
        output=torch.empty((rows_max, HIDDEN), dtype=torch.bfloat16, device=device),
        route_weights=shared["weights"],
        ids=shared["ids"],
        unit_count=unit_count,
        unit_map=unit_map,
        pointers=tables["tables"],
        bits=tables["bits"],
    )
    buffers.update(alloc_scratch(core, device))
    group_values = dict(buffers, x=shared["wire"])
    core_values = dict(buffers, x=buffers["activation"])

    def group_launch(rows):
        launch_program(group, group_values, (rows, group_cta))

    def core_launch(rows):
        launch_program(core, core_values, (rows, core_cta, unit_limit))

    notes = dict(kind="wired", core_cta_limit=core_cta, group_cta_limit=group_cta,
                 threads=256, min_blocks_per_mp=Exlr8GroupedKernel.CTAS_PER_SM,
                 group_rows=16, max_groups=MAX_GROUPS, unit_limit=unit_limit,
                 fc1_ws=False, fc2_partials="f32 (bit-identical to K1)",
                 unit_enumerated="owned only (unit_count/unit_map, unit_limit=%d)"
                                 % unit_limit,
                 build_seconds=round(build_seconds, 3),
                 program=core.name, group_program=group.name)
    return Arm(name, (core, group), buffers, (group_launch, core_launch), notes)


def k1compact_arm(name, rows_max, tables, shared, owned, device="cuda"):
    """K1's serving ABI with compact owned-unit enumeration.

    Same streams, partials and accumulation order as K1, but FC1 and the
    activation enumerate only the caller's owned units, so a TP6 rank that owns
    three of sixteen units fills the whole grid instead of pinning each CTA to
    one (unit, proj, split) family. Output must stay bit-identical to K1.
    """
    if rows_max > 64:
        raise ValueError("the compact K1 instance serves at most 64 rows")
    built = time.monotonic()
    core = compile_exlr8_a16_aot(hidden=HIDDEN, units=UNITS, top_k=TOP_K,
                                 max_groups=MAX_GROUPS, device_count=True,
                                 compact_units=True)
    group = compile_exlr8_group_aot(experts=EXPERTS, top_k=TOP_K,
                                    max_groups=MAX_GROUPS, hidden=HIDDEN,
                                    max_rows=64, group_rows=16)
    build_seconds = time.monotonic() - built
    if core.name != "exlr8_a16_serving_compact" or group.name != "exlr8_group":
        raise RuntimeError(f"compact pairing built {core.name} + {group.name}")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    core_cta = sms * int(core.geometry["ctas_per_sm"])
    group_cta = sms * int(group.geometry["ctas_per_sm"])
    unit_count, unit_map, unit_limit = unit_tables(owned, device)
    buffers = dict(
        group_expert=torch.full((MAX_GROUPS,), -1, dtype=torch.int32, device=device),
        route_slots=torch.full((MAX_GROUPS, 16), -1, dtype=torch.int32, device=device),
        route_group_lane=torch.full((rows_max, TOP_K), -1, dtype=torch.int32, device=device),
        group_count=torch.zeros(1, dtype=torch.int32, device=device),
        activation=torch.empty((rows_max, HIDDEN), dtype=torch.float16, device=device),
        signs_out=shared["signs_out"],
        signs_in=shared["signs_in"],
        output=torch.empty((rows_max, HIDDEN), dtype=torch.bfloat16, device=device),
        route_weights=shared["weights"],
        ids=shared["ids"],
        unit_count=unit_count,
        unit_map=unit_map,
        pointers=tables["tables"],
        bits=tables["bits"],
    )
    buffers.update(alloc_scratch(core, device))
    group_values = dict(buffers, x=shared["wire"])
    core_values = dict(buffers, x=buffers["activation"])

    def group_launch(rows):
        launch_program(group, group_values, (rows, group_cta))

    def core_launch(rows):
        launch_program(core, core_values, (rows, core_cta, unit_limit))

    notes = dict(kind="k1compact", core_cta_limit=core_cta, group_cta_limit=group_cta,
                 threads=128, min_blocks_per_mp=core.geometry["ctas_per_sm"],
                 group_rows=16, max_groups=MAX_GROUPS, unit_limit=unit_limit,
                 fc1_ws=False, fc2_partials="f32 (K1 partials unchanged)",
                 unit_enumerated="owned only (unit_count/unit_map, unit_limit=%d)"
                                 % unit_limit,
                 build_seconds=round(build_seconds, 3),
                 grid_clamp="min(max_groups*unit_limit*2*splits, cta_limit) fc1; "
                            "min(max_groups*unit_limit, cta_limit) activation",
                 program=core.name, group_program=group.name)
    return Arm(name, (core, group), buffers, (group_launch, core_launch), notes)


def route_fixture(rows, distribution, generator, wave):
    if distribution == "random":
        ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K]
                           for _ in range(rows)])
    else:
        ids = real_route_ids(rows, wave, generator, experts=EXPERTS, top_k=TOP_K)
    weights = torch.rand((rows, TOP_K), generator=generator) / 8
    return ids.cuda().int(), weights.cuda()


STAGES = {"ingress": ("_Group",), "fc1": ("fc1_kernel", "grouped_fc1_kernel",
                                          "fc1_ws_kernel"),
          "activation": ("activation_kernel",),
          "fc2": ("fc2_kernel", "grouped_fc2_kernel", "fc2_ws_kernel"),
          "egress": ("reduce_kernel",)}

LOSSY: dict = {}


def measure(actual, expected, label):
    """Record how a lossy arm differs; never used for the exactness gate."""
    difference = actual.float() - expected.float()
    reference = expected.float()
    metrics = dict(label=label,
                   bitwise_equal=bool(torch.equal(actual.view(torch.int16),
                                                  expected.view(torch.int16))),
                   relative_l2=float(torch.linalg.vector_norm(difference)
                                     / torch.linalg.vector_norm(reference).clamp_min(1e-30)),
                   max_abs=float(difference.abs().max()))
    LOSSY[label] = metrics
    return metrics


def compare(actual, expected, label):
    if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        delta = (actual.float() - expected.float()).abs()
        raise RuntimeError(f"{label}: not bit-exact (max |delta| {float(delta.max())})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", default="1,4,8,16,32,64")
    parser.add_argument("--routes", default="random,realistic")
    parser.add_argument("--arms", default="cta2,cta8,grouped16,grouped64")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    cta = sms * 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.source_root or Path(__file__).resolve().parents[1])
    digests = source_digests(root)
    read_only = read_roofline()
    print(json.dumps({"preflight": dict(device=props.name, sms=sms, l2_bytes=l2_bytes,
                                        cta_limit=cta, read_roofline=read_only),
                      "sources": digests}), flush=True)

    owned = owned_units(args.world, args.rank, args.layer)
    tables = make_tables(owned)
    rows_values = [int(v) for v in args.rows.split(",")]
    rows_max = max(rows_values)
    print(json.dumps({"geometry": dict(world=args.world, rank=args.rank, layer=args.layer,
                                       owned_units_per_expert=sorted({len(u) for u in owned}),
                                       payload_bytes=tables["payload_bytes"],
                                       payload_gib=tables["payload_bytes"] / 2**30,
                                       tiers="(2,3,4)[(u+p)%3]")}), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    signs_in = (torch.randint(2, (HIDDEN,), generator=generator) * 2 - 1).char().cuda()
    signs_out = (torch.randint(2, (HIDDEN,), generator=generator) * 2 - 1).char().cuda()
    x = torch.randn((rows_max, HIDDEN), generator=generator).cuda() * 0.02
    wire = make_wire(x)

    wave = load_route_waves("realistic")[0] if "realistic" in args.routes else None
    distributions = [d.strip() for d in args.routes.split(",") if d.strip()]
    fixture = {d: route_fixture(rows_max, d, generator, wave) for d in distributions}

    arms = {}
    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    for distribution, (ids, weights) in fixture.items():
        shared = dict(signs_in=signs_in, signs_out=signs_out, wire=wire,
                      ids=ids, weights=weights)
        arm_set = {"base": base_arm(cta, rows_max, tables, shared)}
        for name in selected:
            if name.startswith("cta"):
                factor = int(name[3:])
                arm_set[name] = base_arm(sms * factor, rows_max, tables, shared)
            elif name == "wired":
                arm_set[name] = wired_arm(name, rows_max, tables, shared, owned)
            elif name == "k23":
                arm_set[name] = base_arm(sms * 4, rows_max, tables, shared,
                                         variant="k1_k23_coalesced", arm_name=name)
            elif name == "k1compact":
                arm_set[name] = k1compact_arm(name, rows_max, tables, shared, owned)
            elif name.startswith("grouped"):
                spec = name[len("grouped"):]
                half = spec.endswith("h")
                if half:
                    spec = spec[:-1]
                factor = 2
                if "x" in spec:
                    spec, factor = spec.split("x", 1)
                    factor = int(factor)
                group_rows = int(spec)
                arm_set[name] = grouped_arm(name, group_rows, sms * factor, rows_max,
                                            tables, shared, owned, half=half)
            else:
                raise ValueError(f"unknown arm {name}")
        arms[distribution] = arm_set

    # Frozen-baseline oracle: 64-row fixture, sliced to 1/17/63/64 rows; every
    # arm must match the baseline's own N-row output bit for bit.
    checked = 0
    for distribution in distributions:
        ids, weights = fixture[distribution]
        reference = arms[distribution]["base"]
        reference.buffers["ids"].copy_(ids)
        reference.buffers["route_weights"].copy_(weights)
        reference.run(rows_max)
        torch.cuda.synchronize()
        full = reference.buffers["output"][:rows_max].clone()
        finite_nonzero(full.float(), f"base {distribution}")
        for rows in ROW_SLICE:
            reference.run(rows)
            torch.cuda.synchronize()
            expected = reference.buffers["output"][:rows].clone()
            compare(expected, full[:rows], f"base row invariance {distribution} {rows}")
            for name, arm in arms[distribution].items():
                if arm is reference:
                    continue
                arm.run(rows)
                torch.cuda.synchronize()
                if arm.lossy:
                    measure(arm.buffers["output"][:rows], expected,
                            f"{name} {distribution} {rows}")
                else:
                    compare(arm.buffers["output"][:rows], expected,
                            f"{name} {distribution} {rows}")
            checked += 1
        print(json.dumps({"oracle": distribution, "cells": checked}), flush=True)

    records = []
    for distribution in distributions:
        ids, weights = fixture[distribution]
        cell_bytes = {rows: routed_bytes(tables, ids[:rows]) for rows in rows_values}
        for rows in rows_values:
            arm_set = arms[distribution]
            for arm in arm_set.values():
                arm.buffers["ids"][:rows].copy_(ids[:rows])
                arm.buffers["route_weights"][:rows].copy_(weights[:rows])
            graphs = {}
            for name, arm in arm_set.items():
                def step(arm=arm, rows=rows):
                    arm.run(rows)
                graph = capture(step)
                graph.replay()
                torch.cuda.synchronize()
                reference = arm_set["base"].buffers["output"][:rows].clone()
                own = arm.buffers["output"][:rows].clone()
                if arm.lossy:
                    measure(own, reference,
                            f"{name} {distribution} {rows} graph capture")
                else:
                    compare(own, reference,
                            f"{name} {distribution} {rows} graph capture")
                poisoned = arm.buffers["output"][:rows]
                poisoned.view(torch.int16).fill_(-1)
                graph.replay()
                torch.cuda.synchronize()
                # Every output byte must be rewritten by the replay, whatever
                # the arm's partial precision is.
                compare(arm.buffers["output"][:rows], own,
                        f"{name} {distribution} {rows} poisoned replay")
                graphs[name] = graph
            rounds = {name: [] for name in graphs}
            order = list(graphs)
            for repeat in range(args.rounds):
                rotated = order[repeat % len(order):] + order[:repeat % len(order)]
                if repeat % 2:
                    rotated.reverse()
                for name in rotated:
                    rounds[name].append(timing(graphs[name], args.replays))
            medians = {name: statistics.median(v) for name, v in rounds.items()}
            reference_us = medians["base"]
            payload = cell_bytes[rows]
            record = dict(distribution=distribution, rows=rows,
                          distinct_experts=int(ids[:rows].unique().numel()),
                          routed_payload_bytes=payload,
                          medians_us=medians, rounds_us=rounds,
                          ratios={name: reference_us / value
                                  for name, value in medians.items()},
                          gbps={name: payload / value / 1000
                                for name, value in medians.items()},
                          roofline_percent={name: 100 * payload / value / 1000
                                            / read_only["gbps"]
                                            for name, value in medians.items()})
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.profile:
                from benchmark_exlr8_a16 import stage_profile
                for name, graph in graphs.items():
                    stages = stage_profile(graph, args.output.parent, STAGES)
                    print(json.dumps({"stages_us": stages, "arm": name, "rows": rows,
                                      "distribution": distribution}), flush=True)
            del graphs

    result = dict(
        schema=1, kind="exlr8_spark_serving_small_rows", device=props.name,
        sm_count=sms, l2_bytes=l2_bytes,
        hidden=HIDDEN, experts=EXPERTS, top_k=TOP_K, units=UNITS,
        world=args.world, rank=args.rank, layer=args.layer,
        tiers="(2,3,4)[(u+p)%3]", ingress="fp8_k32", egress="bf16",
        sources=digests, measured_read_roofline=read_only,
        arms={distribution: {name: arm.notes for name, arm in arm_set.items()}
              for distribution, arm_set in arms.items()},
        conditions=("real served geometry (rotated unit ownership), real tier mix, "
                    "synthetic payloads sized per tier, cold L2 (%.2f GiB payload), "
                    "graph replay of grouping + expert step, one untimed warm replay"
                    % (tables["payload_bytes"] / 2**30)),
        checked_cells=checked, lossy_metrics=LOSSY, rows=records)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"wrote": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
