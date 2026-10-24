#!/usr/bin/env python3
"""GB10 decode-schedule microbenchmark for native EXL3 packages.

The unchanged default is GLM Flash TP4, K3 experts 284..287 and K4 0..283,
K64/N256, m1 for one row and m80 for 2..80. Non-default geometries are read
from cuteafd's --package-policy, never duplicated here. Every arm runs identical
rows, routes and independently generated layers, cycling layers in CUDA graphs.
Variants use NAME:SCHEDULE:TILE:BLOCKS_PER_SM; default and gb10 are presets.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

import torch

from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    bind_mixed_trellis, build_compact_trellis_descriptor, build_tiered_maps,
    combine_trellis_rotations, compile_mixed_trellis, make_mixed_trellis_buffers,
    parse_decode_schedule, run_bound_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

HIDDEN = 4096
INTERMEDIATE = 512
EXPERTS = 288
TOPK = 8
TIER0_IDS = tuple(range(284, 288))
TIER1_IDS = tuple(range(284))
DEFAULT_TILE = (64, 256, 64, 256)


@dataclasses.dataclass(frozen=True)
class Geometry:
    hidden: int = HIDDEN
    intermediate: int = INTERMEDIATE
    experts: int = EXPERTS
    topk: int = TOPK
    bits: tuple[int, int] = (3, 4)
    tier0_ids: tuple[int, ...] = TIER0_IDS
    tier1_ids: tuple[int, ...] = TIER1_IDS
    activation: str = "silu"
    swiglu_limit: float | None = 10.0
    output_dtype: str = "bf16"
    full_intermediate: int = 2048
    world: int = 4
    rank: int = 0
    compact: bool = False

    def widths(self, layer: int) -> list[int]:
        if not self.compact:
            return [self.intermediate] * self.experts
        units = self.full_intermediate // 128
        return [128 * sum((units * e + u + layer) % self.world == self.rank
                          for u in range(units)) for e in range(self.experts)]


def shipping_geometry(args) -> Geometry:
    """Resolve an actual shipping profile from the engine's packaging policy."""
    if args.package_policy is None:
        if (args.geometry, args.role, args.world, args.rank, args.compact_weights) != ("glmf", "spark", 4, 0, False):
            raise ValueError("non-default geometry requires --package-policy")
        return dataclasses.replace(Geometry(), bits=tuple(args.bits or (3, 4)))
    spec = importlib.util.spec_from_file_location("exl3_shipping_policy", args.package_policy)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    hidden, full, _, _ = policy.GEOMETRIES[args.geometry]
    profiles = policy.profiles_for_role(args.role, args.geometry)
    if args.compact_weights:
        if args.role != "spark":
            raise ValueError("compact weights require a Spark profile")
        profiles = policy.compact_profiles(args.geometry, profiles)
    if args.role == "spark":
        destination = f"tp{args.world}-{'compact-' if args.compact_weights else ''}rank{args.rank}"
    else:
        destination = args.profile or f"rtx-tp{args.world}"
    matches = [p for p in profiles if destination in p[5]]
    if len(matches) != 1:
        raise ValueError(f"no unique shipping profile for {args.geometry} {destination}")
    _, width, experts, topk, dtype, _ = matches[0]
    tier0 = TIER0_IDS if args.geometry == "glmf" else tuple(range(experts // 2))
    tier1 = tuple(e for e in range(experts) if e not in tier0)
    return Geometry(hidden, width, experts, topk, tuple(args.bits or (3, 4)), tier0, tier1,
                    policy.activation(args.geometry), policy.swiglu_limit(args.geometry),
                    dtype, full, args.world, args.rank, args.compact_weights)


def expert_bytes(bits: int) -> int:
    return 3 * HIDDEN * INTERMEDIATE * bits // 8


def parse_variant(text: str):
    if ":" not in text:
        name, schedule, tile, blocks = text, text, DEFAULT_TILE, None
    else:
        name, schedule, tile_text, blocks_text = text.split(":")
        tile = tuple(int(v) for v in tile_text.split(",")) if tile_text else DEFAULT_TILE
        blocks = int(blocks_text) if blocks_text else None
    parse_decode_schedule(schedule)
    return name, (None if schedule in ("", "default") else schedule), tile, blocks


def parse_variants(texts: list[str]):
    variants = [parse_variant(text) for text in texts]
    names = [name for name, *_ in variants]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"variant names must be unique: {', '.join(repeated)} repeated")
    return variants


def output_digest(output: torch.Tensor) -> str:
    return hashlib.sha256(output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def verdict(replay: list[str], eager: list[str], reference: list[str]) -> str:
    if replay != eager:
        return "REPLAY-DIFFERS"
    return "same" if replay == reference else "DIFFERENT"


def check_policy_descriptors(sass: str) -> int:
    """Reject odd or unwritten cache policies; ignore the global tensor descriptor."""
    written, globals_, sites = set(), set(), 0
    for line in sass.splitlines():
        if "Function :" in line:
            written, globals_ = set(), set()
        if re.search(r"\b(?:LDGSTS|UBLKCP|UTMALDG)\b", line):
            for operand in re.findall(r"desc\[UR(\d+)\]", line):
                register = int(operand)
                if register % 2 or not {register, register + 1} <= written:
                    raise RuntimeError(f"invalid or unwritten cache-policy descriptor: {line.strip()}")
                if register not in globals_:
                    sites += 1
        # Predicated writes do not prove a descriptor initialized on every path.
        destination = re.search(r"/\*[0-9a-f]+\*/\s+([A-Z][A-Z0-9_.]*)\s+UR(\d+)\s*,", line)
        if destination:
            opcode, register = destination.group(1), int(destination.group(2))
            written.add(register)
            if ".64" in opcode:
                written.add(register + 1)
            # The ABI global-memory descriptor is not an eviction policy.
            if opcode == "LDCU.64" and "c[0x0][0x358]" in line:
                globals_.add(register)
    return sites


def inspect_kernel(launch, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    launch.compiled.export_to_c(str(directory), "core", "cuteafd_core")
    # CuTe embeds cubins as host data; cuobjdump cannot discover them in core.o.
    path = Path(__file__).resolve().parents[3] / "scripts/qualify/deepseek_v41/inspect-cute-object.py"
    spec = importlib.util.spec_from_file_location("schedule_cuda_images", path)
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    images = list(inspector.cuda_images((directory / "core.o").read_bytes()))
    if not images:
        raise RuntimeError("EXL3 AOT export contains no cubins")
    cubins = []
    for index, (_, blob) in enumerate(images):
        cubin = directory / f"core-{index}.cubin"
        cubin.write_bytes(blob)
        cubins.append(cubin)
    info = {"cubin_sha256": [hashlib.sha256(blob).hexdigest() for _, blob in images]}
    for option, name in (("--dump-sass", "sass"), ("--dump-resource-usage", "resources")):
        text = "\n".join(subprocess.check_output(["cuobjdump", option, str(cubin)], text=True)
                         for cubin in cubins)
        (directory / f"core.{name}").write_text(text)
        if name == "sass":
            info["policy_sites"] = check_policy_descriptors(text)
        else:
            values = re.findall(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", text)
            if not values:
                raise RuntimeError("cuobjdump did not report kernel resources")
            info["resources"] = [dict(zip(("registers", "stack", "shared", "local"), map(int, v)))
                                 for v in values]
    (directory / "inspection.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def make_layer(seed: int, device: torch.device, geometry: Geometry = Geometry(), layer: int = 3):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).to(torch.float16)

    g, tiers = geometry, []
    for offset, (ids, bits) in enumerate(zip((g.tier0_ids, g.tier1_ids), g.bits)):
        experts = len(ids)
        tiers.append(prepare_trellis256_moe_weights(
            hidden_size=g.hidden, intermediate_size=g.intermediate, num_experts=experts,
            activation=g.activation, fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + offset, params_dtype=torch.float16,
            w13_layout="trellis_t256_proj", trellis_bits=bits, codebook="mcg",
            gate_suh=scales((experts, g.hidden)), up_suh=scales((experts, g.hidden)),
            intermediate_rotations=scales((experts, 3 * g.intermediate)),
            down_svh=scales((experts, g.hidden)), tile_config=(64, 128, 64, 128),
        ))
    mapping, descriptor = build_tiered_maps(g.tier0_ids, g.tier1_ids, device=device)
    rotations = combine_trellis_rotations(*tiers)
    if g.compact:
        widths = g.widths(layer)
        for index, (tier, ids, bits) in enumerate(zip(tiers, (g.tier0_ids, g.tier1_ids), g.bits)):
            w13 = tier.w13.reshape(2, len(ids), g.hidden // 16, g.intermediate // 16, 8 * bits)
            w2 = tier.w2.reshape(len(ids), g.intermediate // 16, g.hidden // 16, 8 * bits)
            tiers[index] = dataclasses.replace(tier,
                w13=torch.cat([w13[p, e, :, :widths[source] // 16].contiguous().flatten()
                               for p in range(2) for e, source in enumerate(ids)]),
                w2=torch.cat([w2[e, :widths[source] // 16].contiguous().flatten()
                              for e, source in enumerate(ids)]))
        combined_widths = [widths[e] for e in (*g.tier0_ids, *g.tier1_ids)]
        descriptor = build_compact_trellis_descriptor(descriptor, combined_widths,
            hidden_size=g.hidden, intermediate_size=g.intermediate, bits=g.bits)
    return tiers[0], tiers[1], mapping, descriptor, rotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--capacity", type=int, choices=(1, 16, 80),
                        help="qualify an exact decode artifact instead of m1/m80 selection")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--package-policy", type=Path)
    parser.add_argument("--geometry", default="glmf")
    parser.add_argument("--role", choices=("spark", "coordinator"), default="spark")
    parser.add_argument("--profile", help="coordinator profile, including dspark")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--bits", type=int, nargs=2)
    parser.add_argument("--compact-weights", action="store_true")
    parser.add_argument("--check-bitwise", action="store_true")
    parser.add_argument("--check-row-invariance", action="store_true")
    parser.add_argument("--artifacts", type=Path, help="export actual timed cores, SASS and resources")
    args = parser.parse_args()
    try:
        variants = parse_variants(args.variant or ["default", "gb10"])
        g = shipping_geometry(args)
        if len(set(g.bits)) != 2 or any(b not in range(2, 6) for b in g.bits):
            raise ValueError("bits must be two distinct shipping K2..K5 tiers")
    except ValueError as error:
        parser.error(str(error))
    if min(args.layers, args.replays, args.rounds) < 1:
        parser.error("layers, replays and rounds must be positive")
    if args.calls_per_graph < args.layers:
        parser.error("--calls-per-graph must be at least --layers: every layer runs in the timed graph")
    rows_list = [int(v) for v in args.rows.split(",")]
    if not rows_list or any(r < 1 or r > 80 for r in rows_list):
        parser.error("decode rows are 1..80")
    if args.capacity is not None and max(rows_list) > args.capacity:
        parser.error("decode rows must not exceed --capacity")
    if args.artifacts is not None:
        os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    layers = [make_layer(1000 * (i + 1), device, g, 3 + i) for i in range(args.layers)]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.stack([torch.randperm(g.experts, generator=generator)[:g.topk] for _ in range(80)])
    ids = ids.to(torch.int32).to(device)
    weights = torch.rand((80, g.topk), generator=generator).to(device)
    weights = 2.5 * weights / weights.sum(dim=-1, keepdim=True)
    x = (torch.randn((80, g.hidden), generator=generator) * 0.25).to(torch.bfloat16).to(device)
    print(f"device={props.name} sms={sms} H={g.hidden} I={g.intermediate} experts={g.experts} top{g.topk} "
          f"layers={args.layers} calls/graph={args.calls_per_graph} replays={args.replays} rounds={args.rounds}", flush=True)

    def build(capacity, schedule, tile, blocks):
        slots = route_pack_capacity(capacity * g.topk, 8, g.experts, topk=g.topk)[1]
        if g.compact:
            tile = (64, 128, 64, 128)
        launch = compile_mixed_trellis(
            size_m=capacity, hidden_size=g.hidden, intermediate_size=g.intermediate,
            tier0_num_experts=len(g.tier0_ids), tier1_num_experts=len(g.tier1_ids), top_k=g.topk,
            route_num_experts=g.experts, max_m_blocks=(slots + 7) // 8, sms=sms,
            max_shared_mem=int(props.shared_memory_per_block_optin), force_tile_config=tile,
            tier0_bits=g.bits[0], tier1_bits=g.bits[1], trellis_codebook="mcg", swiglu_limit=g.swiglu_limit,
            activation=g.activation, compact_weights=g.compact,
            moe_block_size=8, rotation_input_dtype="bf16", full_rotation_output_dtype=g.output_dtype,
            route_ids_dtype=torch.int32, force_blocks_per_sm=blocks, decode_schedule=schedule,
        )
        key = (capacity, schedule, tile, blocks)
        if args.artifacts is not None and key not in inspected:
            tag = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
            info = inspect_kernel(launch, args.artifacts / f"m{capacity}-{tag}")
            if schedule and parse_decode_schedule(schedule).l2_evict_first_b and not info["policy_sites"]:
                raise RuntimeError("hinted core contains no policy-bearing instructions")
            if schedule is None:
                resource_baselines[(capacity, tile, blocks)] = info["resources"]
            elif (capacity, tile, blocks) in resource_baselines:
                baseline = resource_baselines[(capacity, tile, blocks)]
                if len(baseline) != len(info["resources"]):
                    raise RuntimeError("hint changed the exported kernel count")
                for before, after in zip(baseline, info["resources"]):
                    if after["registers"] > before["registers"] or after["stack"] or after["local"]:
                        raise RuntimeError(f"hint resource growth or spill: {before} -> {after}")
            print(f"resource m{capacity} schedule={schedule} tile={tile} {json.dumps(info)}", flush=True)
            inspected.add(key)
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
        bindings = [bind_mixed_trellis(t0, t1, m, d, r, launch) for t0, t1, m, d, r in layers]
        return launch, buffers, bindings

    inspected, single_outputs, resource_baselines = set(), {}, {}
    if args.check_row_invariance:
        for name, schedule, tile, blocks in variants:
            launch, buffers, bindings = build(1, schedule, tile, blocks)
            single_outputs[name] = [torch.cat([
                run_bound_mixed_trellis(x[r:r+1], weights[r:r+1], ids[r:r+1], binding, buffers).clone()
                for r in range(80)]) for binding in bindings]
    for rows in rows_list:
        capacity = args.capacity if args.capacity is not None else (1 if rows == 1 else 80)
        xs, ws, ks = x[:rows].contiguous(), weights[:rows].contiguous(), ids[:rows].contiguous()
        distinct = torch.unique(ks).tolist()
        weight_bytes = sum(3 * g.hidden * g.widths(3)[e] * g.bits[0 if e in g.tier0_ids else 1] // 8
                           for e in distinct)
        state = {}
        for name, schedule, tile, blocks in variants:
            launch, buffers, bindings = build(capacity, schedule, tile, blocks)
            layer_buffers = [dataclasses.replace(buffers, output=torch.empty_like(buffers.output))
                             for _ in bindings]
            eager = [output_digest(run_bound_mixed_trellis(xs, ws, ks, binding, layer_buffer))
                     for binding, layer_buffer in zip(bindings, layer_buffers)]
            if args.check_row_invariance:
                for layer, layer_buffer in enumerate(layer_buffers):
                    if not torch.equal(layer_buffer.output[:rows], single_outputs[name][layer][:rows]):
                        raise RuntimeError(f"row-invariance failed: {name} rows={rows} layer={layer}")

            def calls(bindings=bindings, layer_buffers=layer_buffers):
                for call in range(args.calls_per_graph):
                    layer = call % len(bindings)
                    run_bound_mixed_trellis(xs, ws, ks, bindings[layer], layer_buffers[layer])

            for _ in range(3):
                calls()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls()
            for layer_buffer in layer_buffers:
                layer_buffer.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Keep bindings and scratch alive: reusing cooperative barrier words
            # while a captured graph retains them can hang its grid barrier.
            state[name] = {"graph": graph, "eager": eager, "times": [],
                           "keep": (launch, buffers, layer_buffers, bindings, calls),
                           "blocks": launch.blocks_per_sm,
                           "tile": (launch.fc1_tile_k, launch.fc1_tile_n, launch.fc2_tile_k, launch.fc2_tile_n),
                           "schedule": launch.decode_schedule}
        for round_index in range(args.rounds):
            order = variants if round_index % 2 == 0 else list(reversed(variants))
            for name, *_ in order:
                graph = state[name]["graph"]
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.replays)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.replays)]
                for start, end in zip(starts, ends):
                    start.record()
                    graph.replay()
                    end.record()
                torch.cuda.synchronize()
                state[name]["times"] += [s.elapsed_time(e) / args.calls_per_graph for s, e in zip(starts, ends)]
        for name, *_ in variants:
            state[name]["replay"] = [output_digest(b.output[:rows]) for b in state[name]["keep"][2]]
        reference = state[variants[0][0]]
        for name, *_ in variants:
            entry = state[name]
            fastest = min(entry["times"])
            result = verdict(entry["replay"], entry["eager"], reference["replay"])
            if args.check_bitwise and result != "same":
                raise RuntimeError(f"bitwise check failed: {name} rows={rows}")
            print(f"rows={rows:2d} m{capacity:<2d} variant={name:<16s} experts={len(distinct):3d} "
                  f"min={fastest:.4f}ms med={statistics.median(entry['times']):.4f}ms "
                  f"GB/s={weight_bytes / fastest / 1e6:6.1f} vs_first={min(reference['times']) / fastest:.3f}x "
                  f"bits={'/'.join(entry['replay'])} {result} "
                  f"tile={','.join(map(str, entry['tile']))} blocks/SM={entry['blocks']} schedule={entry['schedule']}",
                  flush=True)
        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
