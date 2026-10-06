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
layer). Every call of a graph has its own routes (--routes). Rows are timed
variant by variant, interleaved --rounds times.

Routes (--routes):
  uniform             every row takes 8 distinct experts uniformly at random.
  model:SIGMA,TAU[,SEQS]
                      skewed routes as in service: the rows are split evenly
                      over SEQS sequences (default 16, the C16 batch; fewer
                      when there are fewer rows). A row's expert logits are
                      SIGMA * z_e (the layer's popularity, fixed per layer)
                      + TAU * u_se (the sequence's topic, fixed per sequence
                      and call) + Gumbel noise per row; its top 8 are its
                      routes. model:1.0,2.0 reproduces the service's distinct
                      experts a call: 175 at 57 rows in 16 sequences (service
                      C16 code: 173, layers 147-218) and 36 at 8 rows of one
                      sequence (service C1: 35).
  file:PATH           replay routes recorded by the Spark worker
                      (CUTEAFD_EXL3_ROUTE_DUMP): records of four little-endian
                      u32 (magic 0x31455452, layer, rows, top-k), then int32
                      ids [rows*top-k] and float32 weights [rows*top-k]. Calls
                      of a given row count take the file's records with that
                      row count, in order (record i on layer i % --layers).

Per line: the call's GPU time (minimum and median over every replay of every
round); the routes' mean distinct experts, 8-row route blocks (an expert with
c routes takes ceil(c / 8)), experts with more than 8 routes and the largest c;
the weight bandwidth (the distinct experts' Trellis bytes over the minimum
time); and a digest of every call's output, which must agree across variants:
a decode schedule changes when weight words are fetched, never a bit.

Variants: NAME, or NAME:SCHEDULE:TILE:BLOCKS_PER_SM[:ROUTE_BLOCK], for example
  --variant default --variant gb10
  --variant 'gb10-n128x2:gb10:64,128,64,128:2'
  --variant 'l2only:l2=2:64,256,64,256:'
  --variant 'rb16:default:::16'   (m80 with 16-row route blocks; m1 keeps 8)
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import struct

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
ROUTE_MAGIC = 0x31455452  # "RTE1" little-endian


def expert_bytes(bits: int) -> int:
    """Trellis bytes of one expert slice: gate, up and down at `bits` per weight."""
    return 3 * HIDDEN * INTERMEDIATE * bits // 8


def parse_variant(text: str):
    """NAME or NAME:SCHEDULE:TILE:BLOCKS_PER_SM[:ROUTE_BLOCK]."""
    if ":" not in text:
        name, schedule, tile, blocks, route_block = text, text, DEFAULT_TILE, None, 8
    else:
        fields = text.split(":")
        if len(fields) not in (4, 5):
            raise ValueError(f"variant {text!r}: NAME:SCHEDULE:TILE:BLOCKS[:ROUTE_BLOCK]")
        name, schedule, tile_text, blocks_text = fields[:4]
        tile = tuple(int(v) for v in tile_text.split(",")) if tile_text else DEFAULT_TILE
        blocks = int(blocks_text) if blocks_text else None
        route_block = int(fields[4]) if len(fields) == 5 and fields[4] else 8
    if route_block not in (8, 16, 32, 64):
        raise ValueError(f"variant {text!r}: route block must be 8, 16, 32 or 64")
    parse_decode_schedule(None if schedule in ("", "default") else schedule)  # fail before GPU work
    return name, (None if schedule in ("", "default") else schedule), tile, blocks, route_block


def route_stats(ids: torch.Tensor, block: int = 8) -> tuple[int, int, int, int]:
    """(distinct experts, route blocks of `block` rows, experts over `block` routes, max routes)."""
    counts = torch.bincount(ids.reshape(-1).to(torch.int64).cpu(), minlength=EXPERTS)
    counts = counts[counts > 0]
    return (int(counts.numel()), int(((counts + block - 1) // block).sum()),
            int((counts > block).sum()), int(counts.max()))


def uniform_routes(rows: int, generator: torch.Generator) -> torch.Tensor:
    return torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(rows)])


def model_routes(rows: int, sigma: float, tau: float, seqs: int, popularity: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
    """Skewed top-8 routes: layer popularity + per-sequence topic + per-row Gumbel noise."""
    seqs = max(1, min(seqs, rows))
    per = [rows // seqs + (1 if s < rows % seqs else 0) for s in range(seqs)]
    chunks = []
    for count in per:
        topic = torch.randn(EXPERTS, generator=generator, dtype=torch.float64)
        base = sigma * popularity + tau * topic
        uniform = torch.rand((count, EXPERTS), generator=generator, dtype=torch.float64)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-300)))
        chunks.append(torch.topk(base + gumbel, TOPK, dim=1).indices)
    return torch.cat(chunks)


def read_route_file(path: str) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Records (layer, ids[rows, 8], weights[rows, 8]) of a worker route dump."""
    records = []
    with open(path, "rb") as handle:
        data = handle.read()
    offset = 0
    while offset + 16 <= len(data):
        magic, layer, rows, topk = struct.unpack_from("<4I", data, offset)
        if magic != ROUTE_MAGIC or topk != TOPK or not 1 <= rows <= 4096:
            raise ValueError(f"{path}: bad route record at byte {offset}")
        offset += 16
        count = rows * topk
        ids = torch.frombuffer(bytearray(data[offset:offset + 4 * count]), dtype=torch.int32)
        offset += 4 * count
        weights = torch.frombuffer(bytearray(data[offset:offset + 4 * count]), dtype=torch.float32)
        offset += 4 * count
        records.append((layer, ids.view(rows, topk).clone(), weights.view(rows, topk).clone()))
    if offset != len(data):
        raise ValueError(f"{path}: truncated route record at byte {offset}")
    return records


class RouteSource:
    def __init__(self, spec: str, layers: int, seed: int):
        self.spec = spec
        self.layers = layers
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.records = None
        if spec == "uniform":
            self.kind = "uniform"
        elif spec.startswith("model:"):
            values = [float(v) for v in spec[len("model:"):].split(",")]
            if len(values) not in (2, 3):
                raise ValueError("--routes model:SIGMA,TAU[,SEQS]")
            self.kind = "model"
            self.sigma, self.tau = values[0], values[1]
            self.seqs = int(values[2]) if len(values) == 3 else 16
            popularity = torch.Generator(device="cpu").manual_seed(seed + 1)
            self.popularity = [torch.randn(EXPERTS, generator=popularity, dtype=torch.float64)
                               for _ in range(layers)]
        elif spec.startswith("file:"):
            self.kind = "file"
            self.records = read_route_file(spec[len("file:"):])
        else:
            raise ValueError(f"unknown --routes {spec!r}")

    def calls(self, rows: int, count: int):
        """`count` (ids, weights) route sets of `rows` rows; call i runs on layer i % layers."""
        out = []
        if self.kind == "file":
            matching = [(ids, weights) for _, ids, weights in self.records if ids.shape[0] == rows]
            if not matching:
                return []
            for i in range(count):
                ids, weights = matching[i % len(matching)]
                out.append((ids.to(torch.int32), weights.to(torch.float32)))
            return out
        for i in range(count):
            if self.kind == "uniform":
                ids = uniform_routes(rows, self.generator)
            else:
                ids = model_routes(rows, self.sigma, self.tau, self.seqs,
                                   self.popularity[i % self.layers], self.generator)
            weights = torch.rand((rows, TOPK), generator=self.generator)
            weights = 2.5 * weights / weights.sum(dim=-1, keepdim=True)
            out.append((ids.to(torch.int32), weights.to(torch.float32)))
        return out


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
    parser.add_argument("--routes", default="uniform")
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
    source = RouteSource(args.routes, args.layers, args.seed)
    layers = [make_layer(1000 * (i + 1), device) for i in range(args.layers)]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (torch.randn((80, HIDDEN), generator=generator) * 0.25).to(torch.bfloat16).to(device)
    print(f"device={props.name} sms={sms} H={HIDDEN} I={INTERMEDIATE} experts={EXPERTS} top{TOPK} "
          f"layers={args.layers} calls/graph={args.calls_per_graph} replays={args.replays} rounds={args.rounds} "
          f"routes={args.routes}", flush=True)

    def build(capacity, schedule, tile, blocks, route_block):
        slots = route_pack_capacity(capacity * TOPK, route_block, EXPERTS, topk=TOPK)[1]
        launch = compile_mixed_trellis(
            size_m=capacity, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
            tier0_num_experts=len(TIER0_IDS), tier1_num_experts=len(TIER1_IDS), top_k=TOPK,
            route_num_experts=EXPERTS, max_m_blocks=(slots + route_block - 1) // route_block, sms=sms,
            max_shared_mem=int(props.shared_memory_per_block_optin), force_tile_config=tile,
            tier0_bits=3, tier1_bits=4, trellis_codebook="mcg", swiglu_limit=10.0,
            moe_block_size=route_block, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
            route_ids_dtype=torch.int32, force_blocks_per_sm=blocks, decode_schedule=schedule,
        )
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
        bindings = [bind_mixed_trellis(t0, t1, g2c, desc, rot, launch) for t0, t1, g2c, desc, rot in layers]
        return launch, buffers, bindings

    for rows in rows_list:
        capacity = 1 if rows == 1 else 80
        routes = source.calls(rows, args.calls_per_graph)
        if not routes:
            print(f"rows={rows:2d} no recorded routes with this row count; skipped", flush=True)
            continue
        stats = [route_stats(ids) for ids, _ in routes]
        distinct = statistics.mean(s[0] for s in stats)
        blocks8 = statistics.mean(s[1] for s in stats)
        hot = statistics.mean(s[2] for s in stats)
        widest = max(s[3] for s in stats)
        weight_bytes = statistics.mean(
            sum(expert_bytes(3 if e in TIER0_IDS else 4) for e in torch.unique(ids).tolist())
            for ids, _ in routes)
        xs = x[:rows].contiguous()
        call_inputs = [(ws.to(device).contiguous(), ks.to(device).contiguous()) for ks, ws in routes]
        state = {}
        for name, schedule, tile, blocks, route_block in variants:
            block = 8 if capacity == 1 else route_block
            launch, buffers, bindings = build(capacity, schedule, tile, blocks, block)
            outputs = []
            for call, (ws, ks) in enumerate(call_inputs):
                out = run_bound_mixed_trellis(xs, ws, ks, bindings[call % len(bindings)], buffers)
                outputs.append(out[:rows].clone())
            torch.cuda.synchronize()
            digest = hashlib.sha256(b"".join(
                o.contiguous().view(torch.uint8).cpu().numpy().tobytes() for o in outputs)).hexdigest()[:16]
            del outputs

            def calls(bindings=bindings, buffers=buffers):
                for call, (ws, ks) in enumerate(call_inputs):
                    run_bound_mixed_trellis(xs, ws, ks, bindings[call % len(bindings)], buffers)

            for _ in range(3):
                calls()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls()
            graph.replay()
            torch.cuda.synchronize()
            # The graph replays these buffers and bindings: keep them alive until it is
            # done (a freed workspace reused by the next variant corrupts the
            # cooperative kernel's grid-barrier words, and its replays never finish).
            state[name] = {"graph": graph, "digest": digest, "times": [],
                           "keep": (launch, buffers, bindings, calls),
                           "blocks": launch.blocks_per_sm, "tile": tile, "schedule": launch.decode_schedule,
                           "route_block": block}
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
                state[name]["times"] += [s.elapsed_time(e) / len(call_inputs) for s, e in zip(starts, ends)]
        reference = state[variants[0][0]]
        for name, *_ in variants:
            entry = state[name]
            fastest = min(entry["times"])
            print(f"rows={rows:2d} m{capacity:<2d} variant={name:<16s} experts={distinct:5.1f} "
                  f"blocks8={blocks8:5.1f} over8={hot:4.1f} max_c={widest:3d} "
                  f"min={fastest:.4f}ms med={statistics.median(entry['times']):.4f}ms "
                  f"GB/s={weight_bytes / fastest / 1e6:6.1f} vs_first={min(reference['times']) / fastest:.3f}x "
                  f"bits={entry['digest']} {'same' if entry['digest'] == reference['digest'] else 'DIFFERENT'} "
                  f"tile={','.join(map(str, entry['tile']))} blocks/SM={entry['blocks']} "
                  f"route_block={entry['route_block']} schedule={entry['schedule']}",
                  flush=True)
        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
