#!/usr/bin/env python3
"""Time the grouped EXLR8 routed-expert prefill (FC1, activation, FC2) alone.

Synthetic trellis payloads at the real GLM 5.3 expert geometry, so the arm is
the kernel's own instruction mix and nothing else: the grouping/ingress kernel
is skipped by writing the route metadata on the host. Prints ms and the
FC1+FC2 TFLOP/s, which is the number the A8 question turns on (E4M3 MMA on
SM12x has twice the F16/F32 rate).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot


def runtime_l2_bytes(cap_mib=None):
    """Query this CUDA device; a benchmark cap may only reduce its L2 budget."""
    from cuda.bindings import driver as cuda
    if cap_mib is not None and (not 0 < cap_mib < float("inf") or int(cap_mib * 2**20) < 1):
        raise ValueError("--l2-cap-mib must be finite and positive")
    status, device = cuda.cuDeviceGet(torch.cuda.current_device())
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {status}")
    status, queried = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
    if status != cuda.CUresult.CUDA_SUCCESS or not 0 < queried <= 2**31 - 1:
        raise RuntimeError(f"cuDeviceGetAttribute(L2CacheSize) failed: {status}, {queried}")
    return queried if cap_mib is None else min(queried, int(cap_mib * 2**20))


def launch_scalars(program, rows, cta_limit, unit_limit, l2_bytes):
    values = dict(rows=rows, cta_limit=cta_limit, unit_limit=unit_limit, l2_bytes=l2_bytes)
    return tuple(values[scalar.name] for scalar in program.scalars)


def compact_unit_descriptors(bits):
    host = bits.detach().cpu()
    active = host != 0
    counts = active[..., 0].sum(1).to(torch.int32)
    mapping = torch.zeros(host.shape[:2], dtype=torch.int32)
    for expert in range(host.shape[0]):
        owned = torch.nonzero(active[expert, :, 0], as_tuple=False).flatten()
        mapping[expert, :owned.numel()] = owned.to(torch.int32)
    return (counts.to(bits.device), mapping.to(bits.device), max(1, int(counts.max())))


def build_realistic_routes(rows, tk, group_rows, num_experts, wave, gen):
    counts = torch.zeros(num_experts, dtype=torch.float64)
    for expert, n in wave["expert_rows"]:
        if expert < num_experts:
            counts[expert] = n
    target = counts * (rows * tk / counts.sum())
    alloc = target.floor().clamp(max=rows)
    order = torch.argsort(target - alloc, descending=True)
    for e_id in order.tolist():
        if int(alloc.sum()) >= rows * tk:
            break
        if alloc[e_id] < rows:
            alloc[e_id] += 1
    alloc = alloc.long()
    ids = torch.full((rows, tk), -1, dtype=torch.int64)
    fill = torch.zeros(rows, dtype=torch.int64)
    for e_id in torch.argsort(alloc, descending=True).tolist():
        n = int(alloc[e_id])
        if not n:
            break
        jitter = torch.rand(rows, generator=gen)
        pick = torch.argsort(fill.double() + jitter)[:n]
        ids[pick, fill[pick]] = e_id
        fill[pick] += 1
    perm = torch.argsort(torch.rand((rows, tk), generator=gen), dim=1)
    ids = torch.gather(ids, 1, perm).flatten()

    capacity = (4096 * tk + group_rows - 1) // group_rows + num_experts
    group_expert = torch.full((capacity,), -1, dtype=torch.int32)
    route_slots = torch.full((capacity, group_rows), -1, dtype=torch.int32)
    inverse = torch.full((rows * tk,), -1, dtype=torch.int32)
    g_cursor = 0
    for e_id in range(num_experts):
        hits = torch.nonzero(ids == e_id, as_tuple=False).flatten().to(torch.int32)
        n = int(hits.numel())
        if n == 0:
            continue
        g_e = (n + group_rows - 1) // group_rows
        for g_local in range(g_e):
            group_expert[g_cursor + g_local] = e_id
        flat_base = g_cursor * group_rows
        route_slots.view(-1)[flat_base : flat_base + n] = hits
        inverse[hits.long()] = torch.arange(
            flat_base, flat_base + n, dtype=torch.int32
        )
        g_cursor += g_e
    return (
        group_expert.cuda(),
        route_slots.cuda(),
        inverse.reshape(rows, tk).cuda(),
        torch.full((1,), g_cursor, dtype=torch.int32, device="cuda"),
        capacity,
        g_cursor,
    )


def build(args, rows, wave=None):
    gen = torch.Generator(device="cuda").manual_seed(args.seed + rows)
    h, u, tk = args.hidden, args.units, args.top_k
    phys_e = args.experts
    num_e = max(phys_e, 256) if wave is not None else phys_e
    pointers = torch.zeros((num_e, u, 3, 3), dtype=torch.int64)
    bits = torch.zeros((num_e, u, 3), dtype=torch.int32)
    owners = []
    for x in range(phys_e):
        for unit in range(u):
            for p in range(3):
                width = 2 + (x + unit + p) % 3
                ts = (torch.randint(256, (h * 16 * width,), generator=gen, dtype=torch.uint8, device="cuda"),
                      torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                      torch.full((h // 128,), 0.1, dtype=torch.float16, device="cuda"))
                owners.extend(ts)
                pointers[x, unit, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[x, unit, p] = width
    for x in range(phys_e, num_e):
        pointers[x] = pointers[x % phys_e]
        bits[x] = bits[x % phys_e]
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(h, dtype=torch.int8, device="cuda") for _ in range(2)]
    activation = torch.randn((rows, h), generator=gen, device="cuda").half() * 0.02
    weights = (torch.rand((rows, tk), generator=gen, device="cuda") / tk)
    if wave is not None:
        cpu_gen = torch.Generator().manual_seed(args.seed + rows)
        group_expert, route_slots, inverse, count_t, capacity, active_groups = (
            build_realistic_routes(rows, tk, args.group_rows, num_e, wave, cpu_gen)
        )
    else:
        capacity = -(-rows * tk // args.group_rows)
        active_groups = capacity
        slots = rows * tk
        # The grouping kernel emits contiguous per-expert runs, so consecutive
        # groups share one expert's payload (it stays L2-resident, as in serving).
        group_expert = (torch.arange(capacity, device="cuda") // max(1, capacity // num_e)).clamp(max=num_e - 1).int()
        route_slots = torch.full((capacity * args.group_rows,), -1, dtype=torch.int32, device="cuda")
        route_slots[:slots] = torch.arange(slots, device="cuda", dtype=torch.int32)
        route_slots = route_slots.reshape(capacity, args.group_rows)
        inverse = torch.arange(rows * tk, device="cuda").reshape(rows, tk).int() % slots
        count_t = torch.full((1,), active_groups, dtype=torch.int32, device="cuda")
    state = dict(group_expert=group_expert, route_slots=route_slots, inverse=inverse,
                 count=count_t,
                 fc1=torch.empty((capacity, u, 2, (h + 1023) // 1024, args.group_rows, 128), device="cuda"),
                 intermediate=torch.empty((capacity, u, args.group_rows, 128), dtype=torch.float16, device="cuda"),
                 fc2=torch.empty((capacity, args.group_rows, h), device="cuda"),
                 output=torch.empty((rows, h), dtype=torch.bfloat16, device="cuda"),
                 owners=owners, active_groups=active_groups)
    state["compact"] = compact_unit_descriptors(bits)
    return activation, state, weights, pointers, bits, signs, capacity


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iters


def load_wave(routes_arg):
    if not routes_arg:
        return None
    path = (Path(__file__).resolve().parent / "fixtures" / "glm53_prefill_routes_realistic.json"
            if routes_arg == "realistic" else Path(routes_arg))
    data = json.loads(path.read_text())
    return data["waves"][0]


def audit_specs(decode):
    """Existing implementations/knobs only, without changing partial precision."""
    if decode:
        return [("current", 16, {}), ("k23", 16, {}),
                ("k1", 16, {}), ("compact", 16, {}),
                ("group16", 16, {}), ("group32", 32, {}),
                ("group64", 64, {})]
    return [("current", 64, {}), ("current_control", 64, {}),
            ("candidate", 64, {}), ("group16", 16, {}), ("group32", 32, {}),
            ("fc1_coop", 64, {"FC1_WS": False}),
            ("fc2_coop", 64, {"FC2_WS": False}),
            ("fc1_s2", 64, {"FC1_STAGES": 2}),
            ("fc1_s4", 64, {"FC1_STAGES": 4}),
            ("fc2_s2", 64, {"FC2_STAGES": 2}),
            ("cta1", 64, {"CTAS_PER_SM": 1}),
            ("cta4", 64, {"CTAS_PER_SM": 4})]


def manifest_widths(manifest, layer, rank, experts=256, units=16):
    """Retain global slots and ascending sparse TP2 ownership."""
    if rank not in (0, 1):
        raise ValueError("audit rank must be 0 or 1")
    entry = manifest["layers"][str(layer)]
    result = []
    for expert in range(experts):
        row = []
        for unit in range(units):
            widths = [entry["gate"][expert][unit], entry["up"][expert][unit],
                      entry["down"][expert]]
            if any(type(w) is not int or w not in (2, 3, 4) for w in widths):
                raise ValueError("manifest requires K2/K3/K4 integer widths")
            row.append(widths if (units * expert + unit + layer) % 2 == rank else [0, 0, 0])
        result.append(row)
    return result


def audit_order(labels, round_index):
    offset = round_index % len(labels)
    return labels[offset:] + labels[:offset]


def audit_weights(widths, hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bits = torch.tensor(widths, dtype=torch.int32)
    pointers = torch.zeros((*bits.shape, 3), dtype=torch.int64)
    owners = []
    for expert in range(bits.shape[0]):
        for unit in range(bits.shape[1]):
            for projection in range(3):
                width = int(bits[expert, unit, projection])
                if not width:
                    continue
                # Independent bytes/scales per real expert ID, never aliased payloads.
                tensors = (torch.randint(256, (hidden * 16 * width,), generator=gen,
                                         dtype=torch.uint8, device="cuda"),
                           (0.075 + 0.05 * torch.rand(128, generator=gen, device="cuda")).half(),
                           (0.075 + 0.05 * torch.rand(hidden // 128, generator=gen, device="cuda")).half())
                owners.extend(tensors)
                pointers[expert, unit, projection] = torch.tensor([t.data_ptr() for t in tensors])
    return pointers.cuda(), bits.cuda(), owners


def load_reference_core(spec):
    """Verify an immutable git blob before executing any frozen Python source."""
    path_text, separator, expected = spec.rpartition("@")
    if not separator or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--reference-core requires PATH@40-character-git-blob-SHA")
    path = Path(path_text).resolve()
    source = path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if actual != expected:
        raise ValueError("frozen reference-core git blob mismatch")
    name = "b12x.moe._shared.kernels.w4a16._exlr8_reference_" + actual
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module.__package__ = "b12x.moe._shared.kernels.w4a16"
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.Exlr8GroupedKernel, (str(path), actual, hashlib.sha256(source).hexdigest())


def audit_compile(label, group_rows, knobs, capacity, decode, reference=None):
    """Keep benchmark knob sweeps out of the production AOT cache namespace."""
    from b12x.integration.cuteafd import _common, exlr8
    from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
    root = Path(__file__).resolve().parents[1]
    paths = ("b12x/integration/cuteafd/exlr8.py",
             "b12x/moe/_shared/kernels/w4a16/exlr8.py",
             "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py")
    source = tuple((p, hashlib.sha256((root / p).read_bytes()).hexdigest()) for p in paths)
    grouped = not decode or label not in ("k23", "k1", "compact")
    saved = {key: getattr(Exlr8GroupedKernel, key) for key in knobs}
    factory = exlr8.compile_program
    def keyed(wrapper, **kwargs):
        kwargs["name"] = "exlr8_audit_" + kwargs["name"]
        arm_source = source
        if reference is not None:
            reference_type, identity = reference
            if not grouped or knobs:
                raise ValueError("frozen reference requires an unmodified grouped arm")
            wrapper.runtime_l2 = False
            wrapper.core = reference_type(6144, 16, 8, capacity, group_rows,
                                          fc2_f32=decode)
            arm_source = (identity,)
            kwargs["name"] = "exlr8_audit_frozen_reference"
        kwargs["key"] = (arm_source, tuple(sorted(knobs.items())), kwargs["key"])
        return _common.compile_program(wrapper, **kwargs)
    exlr8.compile_program = keyed
    try:
        for key, value in knobs.items():
            setattr(Exlr8GroupedKernel, key, value)
        program = exlr8.compile_exlr8_a16_aot(
            hidden=6144, units=16, top_k=8, max_groups=capacity,
            device_count=True, grouped=grouped, group_rows=group_rows,
            grouped_rows=64 if decode and group_rows == 16 else 4096,
            fc2_partials="f32" if decode else "f16",
            compact_units=label == "compact",
            serving_variant="baseline" if label == "k1" else None)
    finally:
        exlr8.compile_program = factory
        for key, value in saved.items():
            setattr(Exlr8GroupedKernel, key, value)
    return program, source


def audit_pack(ids, state, group_rows, reverse=False):
    """Host metadata is prepared outside captured/timed kernel work."""
    flat = ids.flatten()
    experts = torch.full_like(state["group_expert"], -1, device="cpu")
    slots = torch.full_like(state["route_slots"], -1, device="cpu")
    inverse = torch.full_like(ids, -1, dtype=torch.int32)
    cursor = 0
    for expert in range(256):
        hits = torch.nonzero(flat == expert, as_tuple=False).flatten().int()
        count = hits.numel()
        groups = (count + group_rows - 1) // group_rows
        experts[cursor:cursor + groups] = expert
        slots.flatten()[cursor * group_rows:cursor * group_rows + count] = hits
        cursor += groups
    if cursor > experts.numel():
        raise ValueError("route grouping exceeded planned capacity")
    if reverse:
        experts[:cursor] = experts[:cursor].flip(0)
        slots[:cursor] = slots[:cursor].flip((0, 1))
    packed = slots[:cursor].flatten()
    valid = packed >= 0
    inverse.flatten()[packed[valid].long()] = torch.arange(packed.numel(), dtype=torch.int32)[valid]
    for key, value in (("group_expert", experts), ("route_slots", slots), ("inverse", inverse)):
        state[key][:value.shape[0]].copy_(value)
    state["count"].fill_(cursor)
    return cursor


def audit_state(planned, capacity, group_rows, scratch):
    return dict(group_expert=torch.empty(capacity, dtype=torch.int32, device="cuda"),
                route_slots=torch.empty((capacity, group_rows), dtype=torch.int32, device="cuda"),
                inverse=torch.empty((planned, 8), dtype=torch.int32, device="cuda"),
                count=torch.empty(1, dtype=torch.int32, device="cuda"),
                output=torch.empty((planned + 1, 6144), dtype=torch.bfloat16, device="cuda"),
                **scratch)


def audit_main(args):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from benchmarks.benchmark_exlr8_a16 import real_route_ids, load_route_waves
    if not args.manifest or args.activations != "a16" or args.units != 16 or args.hidden != 6144:
        raise ValueError("--audit requires --manifest, A16 and global GLM geometry H6144/U16")
    if min(args.warmup, args.iters, args.repeats) < 1:
        raise ValueError("audit timing counts must be positive")
    rows_list = [int(r) for r in args.rows.split(",")]
    if any(not 1 <= r <= 4096 for r in rows_list):
        raise ValueError("audit rows must be in 1..4096")
    if args.exhaustive and (max(rows_list) != 4096 or min(rows_list) <= 64):
        raise ValueError("exhaustive mode requires prefill only, including rows4096")
    reference_core = load_reference_core(args.reference_core) if args.reference_core else None
    if reference_core and min(rows_list) <= 64:
        raise ValueError("frozen grouped reference is prefill-only")
    waves = load_route_waves(args.routes or "realistic")
    indices = [int(i) for i in args.wave_indices.split(",")]
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    route_path = (Path(__file__).resolve().parent / "fixtures/glm53_prefill_routes_realistic.json"
                  if args.routes in (None, "realistic") else Path(args.routes))
    props = torch.cuda.get_device_properties(0)
    l2_bytes = runtime_l2_bytes(args.l2_cap_mib)
    report = dict(manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  routes=str(route_path), routes_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
                  gpu=str(props), rank=args.rank, seed=args.seed,
                  l2_bytes=l2_bytes, l2_cap_mib=args.l2_cap_mib,
                  reference_core=reference_core[1] if reference_core else None,
                  payload="independent random trellis bytes, real manifest tiers and global expert IDs",
                  timing=dict(warmup=args.warmup, rounds=args.repeats, replays=args.iters), cases=[])
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    for index in indices:
        wave = waves[index]
        layer = wave["layer"]
        pointers, bits, owners = audit_weights(manifest_widths(manifest, layer, args.rank), 6144, args.seed + layer)
        compact = compact_unit_descriptors(bits)
        assert compact[2] == 8
        planned = 4096 if max(rows_list) > 64 else 64
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        activation = torch.randn((planned, 6144), generator=gen, device="cuda").half() * 0.02
        weights = torch.rand((planned, 8), generator=gen, device="cuda") / 8
        signs = torch.ones(6144, dtype=torch.int8, device="cuda")
        # Sequential arms alias one maximum-size workspace, never parallel streams.
        slots = ((planned * 8 + 63) // 64 + 256) * 64
        scratch = dict(fc1=torch.empty(slots * 16 * 2 * 6 * 128, device="cuda"),
                       intermediate=torch.empty(slots * 16 * 128, dtype=torch.float16, device="cuda"),
                       fc2=torch.empty(slots * 6144, device="cuda"))
        for rows in rows_list:
            decode = rows <= 64
            arm_capacity = 64 if decode else 4096
            ids = real_route_ids(rows, wave, torch.Generator().manual_seed(args.seed + rows))
            arms = {}
            specs = audit_specs(decode)
            known = {s[0] for s in specs}
            if args.arms and not set(args.arms.split(",")) <= known:
                raise ValueError(f"unknown arms; choose from {sorted(known)}")
            for label, group_rows, knobs in specs:
                if args.arms and label not in args.arms.split(",") and label != "current":
                    continue
                actual_label = "k23" if decode and rows < 8 and label == "current" else label
                capacity = (arm_capacity * 8 + group_rows - 1) // group_rows + 256
                if actual_label in ("k23", "k1", "compact"):
                    capacity = 512
                state = audit_state(planned, capacity, group_rows, scratch)
                audit_pack(ids, state, group_rows)
                program, source = audit_compile(actual_label, group_rows, knobs, capacity, decode,
                    reference=reference_core if label in ("current", "current_control") else None)
                limit = args.cta_limit or props.multi_processor_count * int(program.geometry["ctas_per_sm"])
                extra = compact[:2] if any(o.name == "unit_count" for o in program.operands) else ()
                def launch(live=rows, program=program, state=state, extra=extra, limit=limit):
                    scalars = launch_scalars(program, live, limit, 8, l2_bytes)
                    program.launch(activation, pointers, bits, state["group_expert"], state["route_slots"],
                        state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
                        signs, state["output"], state["count"], *extra, scalars=scalars)
                launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graph.replay()
                torch.cuda.synchronize()
                output = state["output"][:rows].clone()
                arms[label] = dict(graph=graph, launch=launch, state=state, output=output,
                                   group_rows=group_rows, program=program, limit=limit, source=source)
            reference = arms["current"]["output"]
            if not bool(torch.isfinite(reference).all()) or not bool(reference.ne(0).any()):
                raise AssertionError("current output must be finite and nonzero")
            exact = {label: torch.equal(arm["output"].view(torch.int16), reference.view(torch.int16))
                     for label, arm in arms.items()}
            # Repacking and poisoned scratch/output catch stale lanes before timing.
            with kernel_resolution_guard("EXLR8 audit frozen planned kernels"):
                for label, arm in arms.items():
                    state = arm["state"]
                    for key in ("fc1", "intermediate", "fc2", "output"):
                        state[key].fill_(float("nan"))
                    audit_pack(ids, state, arm["group_rows"], reverse=True)
                    arm["graph"].replay()
                    torch.cuda.synchronize()
                    if not torch.equal(state["output"][:rows].view(torch.int16), arm["output"].view(torch.int16)):
                        raise AssertionError(f"{label}: packing/poison changed live output")
                    if not bool(torch.isnan(state["output"][rows:]).all()):
                        raise AssertionError(f"{label}: wrote output tail")
                    audit_pack(ids, state, arm["group_rows"])
                if args.exhaustive and rows == 4096:
                    for live in range(1, 4097):
                        for label, arm in arms.items():
                            if not exact[label]:
                                continue
                            state = arm["state"]
                            for key in ("fc1", "intermediate", "fc2", "output"):
                                state[key].fill_(float("nan"))
                            audit_pack(ids[:live], state, arm["group_rows"], reverse=bool(live % 2))
                            arm["launch"](live)
                            if not torch.equal(state["output"][:live].view(torch.int16), reference[:live].view(torch.int16)):
                                raise AssertionError(f"{label}: exhaustive live={live} differs from current full prefix")
                            if not bool(torch.isnan(state["output"][live:]).all()):
                                raise AssertionError(f"{label}: exhaustive live={live} wrote tail")
                    for arm in arms.values():
                        audit_pack(ids, arm["state"], arm["group_rows"])
            samples = {label: [] for label in arms if exact[label]}
            for label in samples:
                for _ in range(args.warmup):
                    arms[label]["graph"].replay()
            torch.cuda.synchronize()
            for round_index in range(args.repeats):
                for label in audit_order(list(samples), round_index):
                    samples[label].append(timeit(arms[label]["graph"].replay, 0, args.iters))
            case = dict(rows=rows, wave_index=index, layer=layer, content=wave.get("content"),
                        bitwise=exact, samples_ms=samples,
                        median_ms={label: statistics.median(s) for label, s in samples.items()},
                        source=arms["current"]["source"], packing_poison=True,
                        exhaustive=args.exhaustive and rows == 4096, unit_limit=8)
            print(json.dumps(case), flush=True)
            report["cases"].append(case)
            if args.json_output:
                Path(args.json_output).write_text(json.dumps(report, indent=2) + "\n")
            del arms, arm, state, program, graph, launch, output, reference
            torch.cuda.empty_cache()
        del pointers, bits, owners, compact, activation, weights, signs, scratch
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--units", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--rows", default="4096")
    ap.add_argument("--group-rows", type=int, default=64)
    ap.add_argument("--activations", choices=("a16", "a8", "both", "all"), default="a16")
    ap.add_argument("--quantize-pass", action="store_true")
    ap.add_argument("--routes", default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--seed", type=int, default=408)
    ap.add_argument("--cta-limit", type=int, default=0)
    ap.add_argument("--l2-cap-mib", type=float, help="cap the queried runtime L2 budget (MiB)")
    ap.add_argument("--dump", default=None, help="save the output rows here for a bitwise arm comparison")
    ap.add_argument("--audit", action="store_true", help="graph-first real-manifest TP2 existing-kernel audit")
    ap.add_argument("--manifest", help="actual EXLR8 width manifest, never synthetic cyclic tiers")
    ap.add_argument("--reference-core", help="frozen grouped source PATH@git-blob-SHA; current/current_control use it")
    ap.add_argument("--rank", type=int, choices=(0, 1), default=0)
    ap.add_argument("--wave-indices", default="0,12,24,36,48,60")
    ap.add_argument("--arms", help="comma-separated audit arms; current is always included")
    ap.add_argument("--exhaustive", action="store_true", help="check frozen prefill kernels at every live row 1..4096")
    ap.add_argument("--json-output", help="raw audit provenance, equality and interleaved event samples")
    args = ap.parse_args()
    args.warmup = args.warmup if args.warmup is not None else (200 if args.audit else 3)
    args.iters = args.iters if args.iters is not None else (20 if args.audit else 10)
    args.repeats = args.repeats if args.repeats is not None else (7 if args.audit else 3)

    if args.audit:
        return audit_main(args)

    wave = load_wave(args.routes)
    row_list = [int(r) for r in str(args.rows).split(",")]
    cta_limit = args.cta_limit or 2 * torch.cuda.get_device_properties(0).multi_processor_count
    props = torch.cuda.get_device_properties(0)
    l2_bytes = runtime_l2_bytes(args.l2_cap_mib)

    if args.activations in ("both", "all"):
        if args.activations == "both":
            arm_specs = [("a16", "a16", True), ("a8", "a8", True)]
        else:
            arm_specs = [("a16", "a16", True), ("a8_fused", "a8", False), ("a8_qpass", "a8", True)]
        for rows in row_list:
            activation, state, weights, pointers, bits, signs, capacity = build(args, rows, wave)
            count, mapping, unit_limit = state["compact"]
            programs = {
                label: compile_exlr8_a16_aot(
                    hidden=args.hidden, units=args.units, top_k=args.top_k,
                    max_groups=capacity, device_count=True, grouped=True,
                    group_rows=args.group_rows, activations=act, quantize_pass=qp)
                for label, act, qp in arm_specs
            }
            outputs = {}
            for label, prog in programs.items():
                prog.launch(activation, pointers, bits, state["group_expert"], state["route_slots"],
                            state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
                            signs[1], state["output"], state["count"], count, mapping,
                            scalars=launch_scalars(prog, rows, cta_limit, unit_limit, l2_bytes))
                torch.cuda.synchronize()
                outputs[label] = state["output"].clone().float()
            samples = {label: [] for label, _, _ in arm_specs}
            labels = [label for label, _, _ in arm_specs]
            rounds = min(3, max(1, args.repeats))
            for r in range(rounds):
                order = labels[r % len(labels):] + labels[:r % len(labels)]
                for label in order:
                    prog = programs[label]
                    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                    start.record()
                    prog.launch(activation, pointers, bits, state["group_expert"], state["route_slots"],
                                state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
                                signs[1], state["output"], state["count"], count, mapping,
                                scalars=launch_scalars(prog, rows, cta_limit, unit_limit, l2_bytes))
                    stop.record()
                    stop.synchronize()
                    samples[label].append(start.elapsed_time(stop))
            med = {label: statistics.median(v) for label, v in samples.items()}
            ref = outputs["a16"]
            ref_norm = float(torch.linalg.vector_norm(ref).clamp_min(1e-30))
            print(f"{props.name} sm{props.major}{props.minor} sms={props.multi_processor_count} "
                  f"rows={rows} active_groups={state['active_groups']} capacity={capacity} "
                  f"routes={'realistic' if wave else 'uniform'} group_rows={args.group_rows} cta_limit={cta_limit} l2_bytes={l2_bytes}")
            for label in labels:
                diff = outputs[label] - ref
                rel_l2 = float(torch.linalg.vector_norm(diff) / ref_norm)
                cos = float(torch.nn.functional.cosine_similarity(
                    outputs[label].flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item())
                delta_pct = 100.0 * (med[label] - med["a16"]) / med["a16"]
                print(f"  {label:10s} median_ms={med[label]:.4f} ({delta_pct:+.2f}%) "
                      f"samples={[round(x, 4) for x in samples[label]]} "
                      f"rel_l2_vs_a16={rel_l2:.6f} cos_vs_a16={cos:.6f}")
            if "a8_qpass" in outputs:
                exact_v1_v2 = torch.equal(
                    outputs["a8_fused"].to(torch.bfloat16).view(torch.int16),
                    outputs["a8_qpass"].to(torch.bfloat16).view(torch.int16))
                print(f"  a8_fused_vs_a8_qpass_bitwise_equal={exact_v1_v2}")
        return

    rows = row_list[0]
    activation, state, weights, pointers, bits, signs, capacity = build(args, rows, wave)
    k1 = compile_exlr8_a16_aot(hidden=args.hidden, units=args.units, top_k=args.top_k,
                               max_groups=capacity, device_count=True, grouped=True,
                               group_rows=args.group_rows, activations=args.activations,
                               quantize_pass=args.quantize_pass)
    count, mapping, unit_limit = state["compact"]

    def run():
        k1.launch(activation, pointers, bits, state["group_expert"], state["route_slots"],
                  state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
                  signs[1], state["output"], state["count"], count, mapping,
                  scalars=launch_scalars(k1, rows, cta_limit, unit_limit, l2_bytes))

    ms = timeit(run, args.warmup, args.iters)
    if args.dump:
        torch.save(state["output"].clone(), args.dump)
    slots = rows * args.top_k
    flops = 2 * slots * (args.hidden * 2 * args.units * 128 + args.units * 128 * args.hidden)
    tf = flops / (ms * 1e9)
    print(f"{props.name} sm{props.major}{props.minor} sms={props.multi_processor_count} "
          f"groups={state['active_groups']}/{capacity} rows={rows} hidden={args.hidden} units={args.units} "
          f"top_k={args.top_k} group_rows={args.group_rows} activations={args.activations} "
          f"quantize_pass={args.quantize_pass} cta_limit={cta_limit} l2_bytes={l2_bytes}")
    print(f"ms {ms:.4f}  TFLOP/s {tf:.1f}  ({100 * tf / 508.7:.1f}% of the 508.7 F16/F32 ceiling)")
    print(f"fc1 {state['fc1'].numel() * 4 / 2**30:.2f} GiB  fc2 {state['fc2'].numel() * 4 / 2**30:.2f} GiB "
          f"inter {state['intermediate'].numel() * 2 / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
