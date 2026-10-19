"""Benchmark the Kimi K3 TP12 EXL3 full-rotation decode path.

The default shape is the production per-rank geometry:

* M=1, H=3584, I=3072/TP12=256
* top-k=16 over 896 global experts
* native 3-bit Trellis weights with shared H-side rotations

Only the routed experts are read, and expert count is runtime data for this
kernel.  The default therefore materializes 32 experts to keep the standalone
benchmark small while preserving the exact generated kernel and launch
geometry used by layers containing hundreds of EXL3 experts.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
import statistics

import torch

from b12x.moe import fused_moe
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    _PreparedWeightRepresentation,
    plan_b12x_fp4_moe_weights,
)


_PRODUCTION_TILES = (64, 256, 64, 256)
_TILE_CANDIDATES = (
    _PRODUCTION_TILES,
    (128, 128, 64, 256),
    (64, 256, 128, 128),
    (128, 128, 128, 128),
    (64, 128, 64, 128),
    (128, 64, 128, 64),
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experts", type=int, help="Materialized experts (default: 32, or all 256 for --compact-glm)")
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compare-tiles",
        action="store_true",
        help="benchmark every legal 128/256-thread tile candidate",
    )
    parser.add_argument("--compact-glm", action="store_true", help="Compare fixed and compact GLM TP4/TP6 expert kernels")
    parser.add_argument("--tp", type=int, nargs="+", choices=(4, 6), default=[4, 6])
    parser.add_argument("--rows", type=int, nargs="+", choices=(1, 16), default=[1, 16])
    parser.add_argument("--bits", type=int, nargs=2, default=[4, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.experts is None:
        args.experts = 256 if args.compact_glm else 32
    if args.experts < (8 if args.compact_glm else 16) or args.experts % 2:
        parser.error("experts must be even and at least top-k")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        parser.error("iterations, rounds, and warmup must be positive")
    if len(set(args.bits)) != 2 or any(bit not in range(2, 6) for bit in args.bits):
        parser.error("bits must be two distinct K2..K5 tiers")
    if args.compact_glm and args.experts > 256:
        parser.error("compact GLM has 256 route experts")
    if args.compact_glm and args.compare_tiles:
        parser.error("compact GLM uses the fixed export tile policy, not a tile sweep")
    return args


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0 / iterations)
    return samples


def _compact_glm(args, device):
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        bind_mixed_trellis, build_compact_trellis_descriptor, build_tiered_maps,
        combine_trellis_rotations, compile_mixed_trellis,
        make_mixed_trellis_buffers, run_bound_mixed_trellis,
    )
    from b12x.moe.fused_moe._impl import _projection_mixed_tile_config

    props = torch.cuda.get_device_properties(device)
    hidden, topk, experts = 6144, 8, args.experts
    count = experts // 2
    records = []
    for tp in args.tp:
        for width in ([512] if tp == 4 else [384, 256]):
            tiers = []
            for bits in args.bits:
                tiers.append(prepare_trellis256_moe_weights(
                    hidden_size=hidden, intermediate_size=width, num_experts=count,
                    activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device,
                    seed=args.seed + bits, params_dtype=torch.float16,
                    w13_layout="trellis_t256_proj", trellis_bits=bits, codebook="mcg",
                    gate_suh=torch.ones((count, hidden), dtype=torch.float16, device=device),
                    up_suh=torch.ones((count, hidden), dtype=torch.float16, device=device),
                    intermediate_rotations=torch.ones((count, 3 * width), dtype=torch.float16, device=device),
                    down_svh=torch.ones((count, hidden), dtype=torch.float16, device=device),
                    tile_config=(128, 128, 128, 128)))
            mapping, descriptor = build_tiered_maps(tuple(range(count)), tuple(range(count, experts)), device=device)
            if experts < 256:
                mapping = torch.cat((mapping, torch.full((256 - experts,), -1, dtype=torch.int32, device=device)))
            rotations = combine_trellis_rotations(*tiers)
            widths = [width] * experts
            if tp == 6 and width == 384:
                # A uniform 16-unit expert rotates ownership over six ranks:
                # four of every six experts have three units, two have two.
                widths = [128 * sum((16 * expert + unit) % 6 == 0 for unit in range(16))
                    for expert in range(experts)]
            compact = []
            for tier_id, (tier, bits) in enumerate(zip(tiers, args.bits)):
                live = widths[tier_id * count:(tier_id + 1) * count]
                w13 = tier.w13.reshape(2, count, hidden // 16, width // 16, 8 * bits)
                w2 = tier.w2.reshape(count, width // 16, hidden // 16, 8 * bits)
                compact.append(replace(tier,
                    w13=torch.cat([w13[p, e, :, :n // 16].contiguous().flatten()
                        for p in range(2) for e, n in enumerate(live)]),
                    w2=torch.cat([w2[e, :n // 16].contiguous().flatten() for e, n in enumerate(live)])))
            live_descriptor = build_compact_trellis_descriptor(descriptor, widths,
                hidden_size=hidden, intermediate_size=width, bits=tuple(args.bits))
            owned_descriptor = build_compact_trellis_descriptor(descriptor, [width] * experts,
                hidden_size=hidden, intermediate_size=width, bits=tuple(args.bits))
            masked = replace(rotations, intermediate=rotations.intermediate.clone())
            for expert, live in enumerate(widths):
                masked.intermediate.view(experts, 3 * width)[expert, live:width] = 0
            for rows in args.rows:
                fixed_tile = _projection_mixed_tile_config(None, hidden_size=hidden,
                    intermediate_size=width, token_count=rows, direct_topk_routes=False)
                compact_tile = (fixed_tile[0], 128, fixed_tile[2], 128) if min(widths) % fixed_tile[1] else fixed_tile
                slots = route_pack_capacity(rows * topk, 8, experts, topk=topk)[1]
                common = dict(size_m=rows, hidden_size=hidden, intermediate_size=width,
                    tier0_num_experts=count, tier1_num_experts=count, top_k=topk,
                    max_m_blocks=(slots + 7) // 8, sms=props.multi_processor_count,
                    max_shared_mem=props.shared_memory_per_block_optin,
                    tier0_bits=args.bits[0], tier1_bits=args.bits[1], swiglu_limit=None,
                    full_rotation_output_dtype="bf16", route_num_experts=256)
                # Uniform routes across the materialized source experts; repeat
                # the same requests for every arm, including the masked control.
                ids = torch.arange(rows * topk, device=device, dtype=torch.int32).reshape(rows, topk) % experts
                weights = torch.full((rows, topk), 1. / topk, device=device)
                x = (torch.randn(rows, hidden, device=device) * 1.e-3).to(torch.bfloat16)
                graphs, owned_output = [], None
                arms = [("fixed", tiers, descriptor, rotations, False, fixed_tile),
                    ("compact_all_owned", tiers, owned_descriptor, rotations, True, fixed_tile),
                    ("fixed_masked", tiers, descriptor, masked, False, fixed_tile),
                    ("compact", compact, live_descriptor, rotations, True, compact_tile)]
                masked_output = None
                for name, payload, desc, rot, enabled, tile in arms:
                    launch = compile_mixed_trellis(**common, force_tile_config=tile, compact_weights=enabled)
                    buffers = make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)
                    binding = bind_mixed_trellis(*payload, mapping, desc, rot, launch)
                    for _ in range(args.warmup):
                        run_bound_mixed_trellis(x, weights, ids, binding, buffers)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run_bound_mixed_trellis(x, weights, ids, binding, buffers)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    if not torch.isfinite(output).all() or output.abs().max() == 0:
                        raise RuntimeError(f"invalid {name} output")
                    if name == "fixed":
                        owned_output = output.clone()
                    if name == "compact_all_owned" and not torch.equal(output, owned_output):
                        raise RuntimeError("all-owned compact differs from fixed")
                    if name == "fixed_masked":
                        masked_output = output.clone()
                    if name == "compact":
                        relative = float((output.float() - masked_output.float()).norm() / masked_output.float().norm())
                        if relative > .004:
                            raise RuntimeError(f"compact differs from source masked control: {relative}")
                    graphs.append((name, graph, binding, buffers, tile))
                for name, graph, binding, buffers, tile in graphs:
                    samples = _time_graph(graph, iterations=args.iterations, rounds=args.rounds)
                    record = dict(tp=tp, rows=rows, hidden=hidden, width=width, arm=name,
                        bits=args.bits, experts=experts, route_experts=256, topk=topk,
                        min_width=min(widths), max_width=max(widths), tile=tile,
                        median_us=statistics.median(samples), samples_us=samples,
                        weight_bytes=sum((t.w13.numel() + t.w2.numel()) * 4 for t in (compact if name == "compact" else tiers)))
                    records.append(record)
                    print(json.dumps(record), flush=True)
                del graphs, binding, buffers, launch, output, owned_output, masked_output
                gc.collect()
            del compact, tiers, rotations, masked, live_descriptor, owned_descriptor
            gc.collect()
            torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    if args.experts < 16 and not args.compact_glm:
        raise ValueError("--experts must be at least top-k=16")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        raise ValueError("iterations, rounds, and warmup must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        raise RuntimeError(f"benchmark requires SM12x, got SM{major}{minor}")

    if args.compact_glm:
        _compact_glm(args, device)
        return

    experts = args.experts
    hidden_size = 3_584
    intermediate_size = 256
    topk = 16
    route_experts = 896
    trellis_words = 16 * 3

    w13 = torch.randint(
        -32768,
        32767,
        (
            2,
            experts,
            hidden_size // 16,
            intermediate_size // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (
            experts,
            intermediate_size // 16,
            hidden_size // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
    )
    shared_h_rotation = torch.ones((1, hidden_size), dtype=torch.float16, device=device)
    intermediate_rotations = torch.ones(
        (experts, 3 * intermediate_size), dtype=torch.float16, device=device
    )
    x = (torch.randn((1, hidden_size), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk)
    topk_weights = torch.rand((1, topk), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    expert_map = torch.full((route_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:experts] = torch.arange(experts, dtype=torch.int32, device=device)

    tile_configs = _TILE_CANDIDATES if args.compare_tiles else (_PRODUCTION_TILES,)
    reference: torch.Tensor | None = None
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"shape=M1/H{hidden_size}/I{intermediate_size}/E{route_experts}/K{topk}"
    )
    print("tiles                         median_us    min_us    rel_error    cosine")

    for tile_config in tile_configs:
        weight_plan = plan_b12x_fp4_moe_weights(
            quant_modes="w4a16",
            source_format="btx",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout="w13",
            trellis_bits=3,
            trellis_codebook="mcg",
            trellis_tile_config=tile_config,
        )
        value = prepare_trellis256_moe_weights(
            w13,
            w2,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=experts,
            activation="situ",
            fc1_tile_n=tile_config[1],
            fc2_tile_n=tile_config[3],
            params_dtype=torch.float16,
            w13_layout="trellis_t256_proj",
            trellis_bits=3,
            codebook="mcg",
            gate_suh=shared_h_rotation,
            up_suh=shared_h_rotation,
            intermediate_rotations=intermediate_rotations,
            down_svh=shared_h_rotation,
            tile_config=tile_config,
        )
        unit = torch.ones((), dtype=torch.float32, device=value.w13.device)
        weights = B12XFP4ExpertWeights(
            plan=weight_plan,
            a1_gscale=unit,
            w1_fp4=value.w13,
            w1_blockscale=value.w13_scale,
            w1_alphas=value.w13_global_scale,
            a2_gscale=unit,
            w2_fp4=value.w2,
            w2_blockscale=value.w2_scale,
            w2_alphas=value.w2_global_scale,
            representation=_PreparedWeightRepresentation(
                quant_mode="w4a16",
                layout=PreparedWeightLayout.TRELLIS_NATIVE,
                value=value,
            ),
        )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=1,
                num_topk=topk,
                route_num_experts=route_experts,
                device=device,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=8,
            )
        )
        scratch_spec = plan.scratch_specs()[0]
        scratch = torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=scratch_spec.device,
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=expert_map,
            output_expert_map=expert_map,
        )
        eager = binding.run().clone()
        torch.cuda.synchronize(device)
        if not torch.isfinite(eager).all():
            raise RuntimeError(f"non-finite output for tile config {tile_config}")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = binding.run()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(captured, eager):
            raise RuntimeError(
                f"CUDA-graph output differs from eager for tile config {tile_config}"
            )

        if reference is None:
            reference = eager.clone()
        relative_error = float(
            (eager - reference).norm() / reference.norm().clamp_min(1.0e-9)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                eager.flatten(), reference.flatten(), dim=0
            )
        )
        samples = _time_graph(
            graph,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        print(
            f"{str(tile_config):<28} "
            f"{statistics.median(samples):9.3f} "
            f"{min(samples):9.3f} "
            f"{relative_error:12.3e} "
            f"{cosine:9.6f}"
        )

        del captured, graph, binding, scratch, plan, weights, weight_plan, eager
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
