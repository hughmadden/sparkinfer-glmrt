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
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot


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
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=408)
    ap.add_argument("--cta-limit", type=int, default=0)
    ap.add_argument("--dump", default=None, help="save the output rows here for a bitwise arm comparison")
    args = ap.parse_args()

    wave = load_wave(args.routes)
    row_list = [int(r) for r in str(args.rows).split(",")]
    cta_limit = args.cta_limit or 2 * torch.cuda.get_device_properties(0).multi_processor_count
    props = torch.cuda.get_device_properties(0)

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
                            scalars=(rows, cta_limit, unit_limit))
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
                                scalars=(rows, cta_limit, unit_limit))
                    stop.record()
                    stop.synchronize()
                    samples[label].append(start.elapsed_time(stop))
            med = {label: statistics.median(v) for label, v in samples.items()}
            ref = outputs["a16"]
            ref_norm = float(torch.linalg.vector_norm(ref).clamp_min(1e-30))
            print(f"{props.name} sm{props.major}{props.minor} sms={props.multi_processor_count} "
                  f"rows={rows} active_groups={state['active_groups']} capacity={capacity} "
                  f"routes={'realistic' if wave else 'uniform'} group_rows={args.group_rows} cta_limit={cta_limit}")
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
                  scalars=(rows, cta_limit, unit_limit))

    ms = timeit(run, args.warmup, args.iters)
    if args.dump:
        torch.save(state["output"].clone(), args.dump)
    slots = rows * args.top_k
    flops = 2 * slots * (args.hidden * 2 * args.units * 128 + args.units * 128 * args.hidden)
    tf = flops / (ms * 1e9)
    print(f"{props.name} sm{props.major}{props.minor} sms={props.multi_processor_count} "
          f"groups={state['active_groups']}/{capacity} rows={rows} hidden={args.hidden} units={args.units} "
          f"top_k={args.top_k} group_rows={args.group_rows} activations={args.activations} "
          f"quantize_pass={args.quantize_pass} cta_limit={cta_limit}")
    print(f"ms {ms:.4f}  TFLOP/s {tf:.1f}  ({100 * tf / 508.7:.1f}% of the 508.7 F16/F32 ceiling)")
    print(f"fc1 {state['fc1'].numel() * 4 / 2**30:.2f} GiB  fc2 {state['fc2'].numel() * 4 / 2**30:.2f} GiB "
          f"inter {state['intermediate'].numel() * 2 / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
