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


def build(args):
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    h, u, tk, e = args.hidden, args.units, args.top_k, args.experts
    capacity = -(-args.rows * tk // args.group_rows)
    pointers = torch.zeros((e, u, 3, 3), dtype=torch.int64)
    bits = torch.zeros((e, u, 3), dtype=torch.int32)
    owners = []
    for x in range(e):
        for unit in range(u):
            for p in range(3):
                width = 2 + (x + unit + p) % 3
                ts = (torch.randint(256, (h * 16 * width,), generator=gen, dtype=torch.uint8, device="cuda"),
                      torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                      torch.full((h // 128,), 0.1, dtype=torch.float16, device="cuda"))
                owners.extend(ts)
                pointers[x, unit, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[x, unit, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(h, dtype=torch.int8, device="cuda") for _ in range(2)]
    activation = torch.randn((args.rows, h), generator=gen, device="cuda").half() * 0.02
    slots = args.rows * tk
    # The grouping kernel emits contiguous per-expert runs, so consecutive
    # groups share one expert's payload (it stays L2-resident, as in serving).
    group_expert = (torch.arange(capacity, device="cuda") // (capacity // e)).int()
    route_slots = torch.arange(slots, device="cuda").reshape(capacity, args.group_rows).int()
    inverse = torch.arange(args.rows * tk, device="cuda").reshape(args.rows, tk).int() % slots
    weights = (torch.rand((args.rows, tk), generator=gen, device="cuda") / tk)
    state = dict(group_expert=group_expert, route_slots=route_slots, inverse=inverse,
                 count=torch.full((1,), capacity, dtype=torch.int32, device="cuda"),
                 fc1=torch.empty((capacity, u, 2, (h + 1023) // 1024, args.group_rows, 128), device="cuda"),
                 intermediate=torch.empty((capacity, u, args.group_rows, 128), dtype=torch.float16, device="cuda"),
                 fc2=torch.empty((capacity, args.group_rows, h), device="cuda"),
                 output=torch.empty((args.rows, h), dtype=torch.bfloat16, device="cuda"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--units", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--group-rows", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=408)
    ap.add_argument("--cta-limit", type=int, default=0)
    ap.add_argument("--dump", default=None, help="save the output rows here for a bitwise arm comparison")
    args = ap.parse_args()

    activation, state, weights, pointers, bits, signs, capacity = build(args)
    k1 = compile_exlr8_a16_aot(hidden=args.hidden, units=args.units, top_k=args.top_k,
                               max_groups=capacity, device_count=True, grouped=True,
                               group_rows=args.group_rows)
    cta_limit = args.cta_limit or 2 * torch.cuda.get_device_properties(0).multi_processor_count
    count, mapping, unit_limit = state["compact"]

    def run():
        k1.launch(activation, pointers, bits, state["group_expert"], state["route_slots"],
                  state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
                  signs[1], state["output"], state["count"], count, mapping,
                  scalars=(args.rows, cta_limit, unit_limit))

    ms = timeit(run, args.warmup, args.iters)
    if args.dump:
        torch.save(state["output"].clone(), args.dump)
    slots = args.rows * args.top_k
    flops = 2 * slots * (args.hidden * 2 * args.units * 128 + args.units * 128 * args.hidden)
    tf = flops / (ms * 1e9)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} sm{props.major}{props.minor} sms={props.multi_processor_count} "
          f"groups={capacity} rows={args.rows} hidden={args.hidden} units={args.units} "
          f"top_k={args.top_k} group_rows={args.group_rows} cta_limit={cta_limit}")
    print(f"ms {ms:.4f}  TFLOP/s {tf:.1f}  ({100 * tf / 508.7:.1f}% of the 508.7 F16/F32 ceiling)")
    print(f"fc1 {state['fc1'].numel() * 4 / 2**30:.2f} GiB  fc2 {state['fc2'].numel() * 4 / 2**30:.2f} GiB "
          f"inter {state['intermediate'].numel() * 2 / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
