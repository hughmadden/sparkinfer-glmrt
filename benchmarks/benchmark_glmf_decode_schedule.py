#!/usr/bin/env python3
"""GB10 decode-schedule microbenchmark for the GLM 5.3 Flash Spark EXL3 packages.

One TP4 rank's slice of GLM 5.3 Flash (H 4096, I 512 of 2048, 288 experts,
top-8; experts 284-287 decode as K3 and the rest as K4, so both tiers of the
k34 package run), at the Spark worker's decode capacities: m1 for one row and
m80 for 2-80 rows, both K64/N256, as cuteafd's GLM Flash row policy selects.

Every variant runs the same rows, routes and weights. A call is what the
worker's executor launches after its wire decode: route packing, the
cooperative core and the top-k sum. Calls are timed as CUDA-graph replays that
cycle through --layers independently generated layers, so no call finds the
previous call's weights in L2 (as in serving, where every call is another
layer). Rows are timed variant by variant, interleaved --rounds times.

Per line: the call's GPU time (minimum and median over every replay of every
round), the weight bandwidth (the distinct experts' Trellis bytes over the
minimum time) and a digest of the output, which must agree across variants:
a decode schedule changes when weight words are fetched, never a bit.

Variants: NAME, or NAME:SCHEDULE:TILE:BLOCKS_PER_SM, for example
  --variant default --variant gb10
  --variant 'gb10-n128x2:gb10:64,128,64,128:2'
  --variant 'l2only:l2=2:64,256,64,256:'
"""

from __future__ import annotations

import argparse
import hashlib
import statistics

import torch

from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    bind_mixed_trellis,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    parse_decode_schedule,
    run_bound_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

HIDDEN = 4096
INTERMEDIATE = 512
EXPERTS = 288
TOPK = 8
TIER0_IDS = tuple(range(284, 288))  # K3
TIER1_IDS = tuple(range(0, 284))  # K4
DEFAULT_TILE = (64, 256, 64, 256)


def expert_bytes(bits: int) -> int:
    """Trellis bytes of one expert slice: gate, up and down at `bits` per weight."""
    return 3 * HIDDEN * INTERMEDIATE * bits // 8


def parse_variant(text: str):
    if ":" not in text:
        name, schedule, tile, blocks = text, text, DEFAULT_TILE, None
    else:
        name, schedule, tile_text, blocks_text = text.split(":")
        tile = tuple(int(v) for v in tile_text.split(",")) if tile_text else DEFAULT_TILE
        blocks = int(blocks_text) if blocks_text else None
    parse_decode_schedule(schedule)  # fail before any GPU work
    return name, (None if schedule in ("", "default") else schedule), tile, blocks


def make_layer(seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).to(torch.float16)

    tiers = []
    for offset, (ids, bits) in enumerate(((TIER0_IDS, 3), (TIER1_IDS, 4))):
        experts = len(ids)
        tiers.append(prepare_trellis256_moe_weights(
            hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_experts=experts,
            activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + offset, params_dtype=torch.float16,
            w13_layout="trellis_t256_proj", trellis_bits=bits, codebook="mcg",
            gate_suh=scales((experts, HIDDEN)), up_suh=scales((experts, HIDDEN)),
            intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
            down_svh=scales((experts, HIDDEN)), tile_config=(64, 128, 64, 128),
        ))
    global_to_combined, descriptor = build_tiered_maps(TIER0_IDS, TIER1_IDS, device=device)
    return tiers[0], tiers[1], global_to_combined, descriptor, combine_trellis_rotations(*tiers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    variants = [parse_variant(v) for v in (args.variant or ["default", "gb10"])]
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(r < 1 or r > 80 for r in rows_list):
        raise ValueError("decode rows are 1..80")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    layers = [make_layer(1000 * (i + 1), device) for i in range(args.layers)]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(80)])
    ids = ids.to(torch.int32).to(device)
    weights = torch.rand((80, TOPK), generator=generator).to(device)
    weights = 2.5 * weights / weights.sum(dim=-1, keepdim=True)
    x = (torch.randn((80, HIDDEN), generator=generator) * 0.25).to(torch.bfloat16).to(device)
    print(f"device={props.name} sms={sms} H={HIDDEN} I={INTERMEDIATE} experts={EXPERTS} top{TOPK} "
          f"layers={args.layers} calls/graph={args.calls_per_graph} replays={args.replays} rounds={args.rounds}")

    def build(capacity, schedule, tile, blocks):
        slots = route_pack_capacity(capacity * TOPK, 8, EXPERTS, topk=TOPK)[1]
        launch = compile_mixed_trellis(
            size_m=capacity, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
            tier0_num_experts=len(TIER0_IDS), tier1_num_experts=len(TIER1_IDS), top_k=TOPK,
            route_num_experts=EXPERTS, max_m_blocks=(slots + 7) // 8, sms=sms,
            max_shared_mem=int(props.shared_memory_per_block_optin), force_tile_config=tile,
            tier0_bits=3, tier1_bits=4, trellis_codebook="mcg", swiglu_limit=10.0,
            moe_block_size=8, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
            route_ids_dtype=torch.int32, force_blocks_per_sm=blocks, decode_schedule=schedule,
        )
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
        bindings = [bind_mixed_trellis(t0, t1, g2c, desc, rot, launch) for t0, t1, g2c, desc, rot in layers]
        return launch, buffers, bindings

    for rows in rows_list:
        capacity = 1 if rows == 1 else 80
        xs, ws, ks = x[:rows].contiguous(), weights[:rows].contiguous(), ids[:rows].contiguous()
        distinct = torch.unique(ks).tolist()
        weight_bytes = sum(expert_bytes(3 if e in TIER0_IDS else 4) for e in distinct)
        state = {}
        for name, schedule, tile, blocks in variants:
            launch, buffers, bindings = build(capacity, schedule, tile, blocks)
            out = run_bound_mixed_trellis(xs, ws, ks, bindings[0], buffers)
            torch.cuda.synchronize()
            digest = hashlib.sha256(out.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]

            def calls(bindings=bindings, buffers=buffers):
                for call in range(args.calls_per_graph):
                    run_bound_mixed_trellis(xs, ws, ks, bindings[call % len(bindings)], buffers)

            for _ in range(3):
                calls()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls()
            graph.replay()
            torch.cuda.synchronize()
            state[name] = {"graph": graph, "digest": digest, "times": [],
                           "blocks": launch.blocks_per_sm, "tile": tile, "schedule": launch.decode_schedule}
        for _ in range(args.rounds):
            for name, *_ in variants:
                graph = state[name]["graph"]
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.replays)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.replays)]
                for start, end in zip(starts, ends):
                    start.record()
                    graph.replay()
                    end.record()
                torch.cuda.synchronize()
                state[name]["times"] += [s.elapsed_time(e) / args.calls_per_graph for s, e in zip(starts, ends)]
        reference = state[variants[0][0]]
        for name, *_ in variants:
            entry = state[name]
            fastest = min(entry["times"])
            print(f"rows={rows:2d} m{capacity:<2d} variant={name:<16s} experts={len(distinct):3d} "
                  f"min={fastest:.4f}ms med={statistics.median(entry['times']):.4f}ms "
                  f"GB/s={weight_bytes / fastest / 1e6:6.1f} vs_first={reference['times'] and min(reference['times']) / fastest:.3f}x "
                  f"bits={entry['digest']} {'same' if entry['digest'] == reference['digest'] else 'DIFFERENT'} "
                  f"tile={','.join(map(str, entry['tile']))} blocks/SM={entry['blocks']} schedule={entry['schedule']}",
                  flush=True)
        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
