#!/usr/bin/env python3
"""Benchmark one-grid mixed K3/K4 Trellis against two serial tier launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from b12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
    route_pack_capacity,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
    bind_mixed_trellis,
    build_ordered_maps,
    build_projection_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    parse_decode_schedule,
    run_bound_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
K3_EXPERTS = 192
K4_EXPERTS = 64
TOTAL_EXPERTS = K3_EXPERTS + K4_EXPERTS
TILE_CONFIG = (128, 128, 32, 512)

KIMI_HIDDEN = 3584
KIMI_INTERMEDIATE = 512
KIMI_EXPERTS = 896
KIMI_TOPK = 16
KIMI_BITS = (2, 3)
KIMI_DEFAULT_TILE = (64, 256, 64, 256)
KIMI_DEFAULT_ROWS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 36)
KIMI_DEFAULT_VARIANTS = (
    "old:default:64,256,64,256:1",
    "n128-2cta:default:64,128,64,128:2",
)


def _prepared(bits: int, experts: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        gate_suh=scales((experts, HIDDEN)),
        up_suh=scales((experts, HIDDEN)),
        intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
        down_svh=scales((experts, HIDDEN)),
        tile_config=TILE_CONFIG,
    )


def _routing(m: int, materialized: int, device: torch.device) -> torch.Tensor:
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(20260730 + m)
    for _ in range(m):
        k3 = torch.randperm(materialized, generator=generator)[:6]
        k4 = K3_EXPERTS + torch.randperm(materialized, generator=generator)[:2]
        rows.append(torch.cat((k3, k4))[torch.randperm(TOPK, generator=generator)])
    return torch.stack(rows).to(dtype=torch.int32, device=device)


def _serial_state(m, tier, expert_map, props, device):
    buffers = make_w4a16_packed_buffers(
        tier,
        m=m,
        topk=TOPK,
        dtype=torch.float16,
        device=device,
        route_num_experts=TOTAL_EXPERTS,
        full_rotation=True,
        block_size_m=8,
    )
    launch = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=int(tier.num_experts),
        top_k=TOPK,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=int(buffers.block_expert_ids.numel()),
        element_dtype="fp16",
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=int(tier.trellis_bits),
        force_tile_config=TILE_CONFIG,
        intermediate_rotation=True,
        full_rotation=True,
        rotation_input_dtype="bf16",
    )
    topk_sum = compile_w4a16_topk_sum(
        m=m,
        topk=TOPK,
        hidden_size=HIDDEN,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=int(tier.num_experts),
        route_num_experts=TOTAL_EXPERTS,
        route_ids_dtype=torch.int32,
        use_expert_map=True,
    )
    return buffers, launch, topk_sum, expert_map


def _run_serial(state, x, weights, ids):
    tier, buffers, launch, topk_sum, expert_map = state
    return run_w4a16_moe(
        x,
        tier,
        weights,
        ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        fused_launch=launch,
        topk_sum_launch=topk_sum,
        route_block_size_m=8,
        intermediate_rotation_scales=tier.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=tier.gate_suh,
        suh_up_table=tier.up_suh,
        svh_table=tier.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


def _capture(fn, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _times(graph: torch.cuda.CUDAGraph, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def _bytes(tensors) -> int:
    storages: dict[tuple[int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def _padded_rotations(tier0, tier1, device: torch.device) -> MixedTrellisRotations:
    intermediate = torch.ones(
        (TOTAL_EXPERTS, 3 * INTERMEDIATE), dtype=torch.float16, device=device
    )
    gate_suh = torch.ones((TOTAL_EXPERTS, HIDDEN), dtype=torch.float16, device=device)
    up_suh = torch.ones_like(gate_suh)
    down_svh = torch.ones_like(gate_suh)
    k3 = int(tier0.num_experts)
    k4 = int(tier1.num_experts)
    intermediate[:k3].copy_(tier0.intermediate_rotations)
    intermediate[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.intermediate_rotations)
    gate_suh[:k3].copy_(tier0.gate_suh)
    gate_suh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.gate_suh)
    up_suh[:k3].copy_(tier0.up_suh)
    up_suh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.up_suh)
    down_svh[:k3].copy_(tier0.down_svh)
    down_svh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.down_svh)
    return MixedTrellisRotations(intermediate, gate_suh, up_suh, down_svh)


def _pad_projection_major_w13(tier, compiled_experts: int):
    """Preserve the [projection, expert, ...] plane stride used by codegen."""
    materialized = int(tier.num_experts)
    bits = int(tier.trellis_bits)
    source = tier.w13.view(
        2,
        materialized,
        HIDDEN // 16,
        INTERMEDIATE // 16,
        8 * bits,
    )
    padded = torch.zeros(
        (
            2,
            int(compiled_experts),
            HIDDEN // 16,
            INTERMEDIATE // 16,
            8 * bits,
        ),
        dtype=torch.int32,
        device=tier.w13.device,
    )
    padded[:, :materialized].copy_(source)
    return replace(tier, w13=padded.view(-1), num_experts=int(compiled_experts))


def _pad_exact_tier_storage(tier, compiled_experts: int):
    """Materialize the full expert-sized storage contract used by ABI 5."""
    materialized = int(tier.num_experts)
    bits = int(tier.trellis_bits)
    padded = _pad_projection_major_w13(tier, compiled_experts)
    w2 = torch.zeros(
        (
            int(compiled_experts),
            INTERMEDIATE // 16,
            HIDDEN // 16,
            8 * bits,
        ),
        dtype=torch.int32,
        device=tier.w2.device,
    )
    w2[:materialized].copy_(tier.w2.view(materialized, *w2.shape[1:]))

    def pad_global(source: torch.Tensor) -> torch.Tensor:
        output = torch.ones(
            int(compiled_experts), dtype=source.dtype, device=source.device
        )
        output[:materialized].copy_(source)
        return output

    return replace(
        padded,
        w2=w2.view(-1),
        w13_global_scale=pad_global(tier.w13_global_scale),
        w2_global_scale=pad_global(tier.w2_global_scale),
    )


def _parse_kimi_variant(text: str):
    if ":" not in text:
        name, schedule, tile, blocks = text, text, KIMI_DEFAULT_TILE, None
    else:
        name, schedule, tile_text, blocks_text = text.split(":")
        tile = (
            tuple(int(v) for v in tile_text.split(","))
            if tile_text
            else KIMI_DEFAULT_TILE
        )
        blocks = int(blocks_text) if blocks_text else None
    parse_decode_schedule(schedule)
    return name, (None if schedule in ("", "default") else schedule), tile, blocks


def _output_digest(output: torch.Tensor) -> str:
    return hashlib.sha256(
        output.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def _kernel_resource_info(launch, max_shared_mem: int) -> dict[str, int | float]:
    cta_threads = (int(launch.fc1_tile_k) * int(launch.fc1_tile_n)) // 64
    dyn_smem = int(launch.shared_memory_bytes)
    regs = -1
    local_bytes = -1
    static_smem = 1024
    occ_ctas = -1

    try:
        from cuda.bindings import driver, runtime

        executable = getattr(launch.compiled, "_executable", launch.compiled)
        module = getattr(executable, "jit_module", None)
        if module is not None:
            err, dev_id = runtime.cudaGetDevice()
            if int(err) == 0:
                cu_dev = driver.CUdevice(int(dev_id))
                kernels = [
                    entry.kernel for entry in getattr(module, "cuda_modules", ())
                ]
                for library in getattr(module, "cuda_library", ()):
                    err_c, count = runtime.cudaLibraryGetKernelCount(library)
                    if int(err_c) == 0:
                        err_k, enum_k = runtime.cudaLibraryEnumerateKernels(
                            count, library
                        )
                        if int(err_k) == 0:
                            kernels.extend(enum_k)
                if kernels:
                    cu_kernel = driver.CUkernel(int(kernels[0]))
                    err_r, r_val = driver.cuKernelGetAttribute(
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                        cu_kernel,
                        cu_dev,
                    )
                    if int(err_r) == 0:
                        regs = int(r_val)
                    err_l, l_val = driver.cuKernelGetAttribute(
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                        cu_kernel,
                        cu_dev,
                    )
                    if int(err_l) == 0:
                        local_bytes = int(l_val)
                    err_s, s_val = driver.cuKernelGetAttribute(
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                        cu_kernel,
                        cu_dev,
                    )
                    if int(err_s) == 0:
                        static_smem = int(s_val)
                    try:
                        err_o, o_val = (
                            runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                kernels[0], cta_threads, dyn_smem
                            )
                        )
                        if int(err_o) == 0 and int(o_val) > 0:
                            occ_ctas = int(o_val)
                    except Exception:
                        pass
                    if occ_ctas < 0:
                        try:
                            err_f, cu_func = driver.cuKernelGetFunction(cu_kernel)
                            if int(err_f) == 0:
                                err_o, o_val = (
                                    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                                        cu_func, cta_threads, dyn_smem
                                    )
                                )
                                if int(err_o) == 0 and int(o_val) > 0:
                                    occ_ctas = int(o_val)
                        except Exception:
                            pass
    except Exception:
        pass
    if regs < 0:
        try:
            import re
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory(prefix="b12x_res_") as tmpdir:
                launch.compiled.export_to_c(tmpdir, "core", "cuteafd_core")
                ru = subprocess.run(
                    ["cuobjdump", "-res-usage", f"{tmpdir}/core.o"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                m_res = re.search(
                    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)",
                    ru.stdout,
                )
                if m_res:
                    regs = int(m_res.group(1))
                    local_bytes = int(m_res.group(2)) + int(m_res.group(4))
                    static_smem = int(m_res.group(3))
        except Exception:
            pass
    if occ_ctas < 0 and regs > 0:
        alloc_regs = ((regs + 7) // 8) * 8
        reg_ctas = 65536 // (alloc_regs * cta_threads)
        smem_ctas = int(max_shared_mem) // max(dyn_smem + static_smem, 1)
        warp_ctas = 48 // (cta_threads // 32)
        occ_ctas = max(1, min(reg_ctas, smem_ctas, warp_ctas))
    warps_per_sm = occ_ctas * (cta_threads // 32) if occ_ctas > 0 else -1
    occ_pct = (100.0 * warps_per_sm / 48.0) if warps_per_sm > 0 else -1.0
    return {
        "cta_threads": cta_threads,
        "blocks_per_sm": int(launch.blocks_per_sm),
        "regs": regs,
        "spills_local_bytes": local_bytes,
        "static_smem_bytes": static_smem,
        "dyn_smem_bytes": dyn_smem,
        "occ_ctas_per_sm": occ_ctas,
        "occ_warps_per_sm": warps_per_sm,
        "occ_pct": occ_pct,
    }


def _make_kimi_layer(seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    tier0 = prepare_trellis256_moe_weights(
        hidden_size=KIMI_HIDDEN,
        intermediate_size=KIMI_INTERMEDIATE,
        num_experts=KIMI_EXPERTS,
        activation="situ",
        fc1_tile_n=128,
        fc2_tile_n=128,
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=KIMI_BITS[0],
        codebook="mcg",
        gate_suh=scales((KIMI_EXPERTS, KIMI_HIDDEN)),
        up_suh=scales((KIMI_EXPERTS, KIMI_HIDDEN)),
        intermediate_rotations=scales((KIMI_EXPERTS, 3 * KIMI_INTERMEDIATE)),
        down_svh=scales((KIMI_EXPERTS, KIMI_HIDDEN)),
        tile_config=(64, 128, 64, 128),
    )
    tier1_full = prepare_trellis256_moe_weights(
        hidden_size=KIMI_HIDDEN,
        intermediate_size=KIMI_INTERMEDIATE,
        num_experts=1,
        activation="situ",
        fc1_tile_n=128,
        fc2_tile_n=128,
        device=device,
        seed=seed + 1,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=KIMI_BITS[1],
        codebook="mcg",
        gate_suh=scales((1, KIMI_HIDDEN)),
        up_suh=scales((1, KIMI_HIDDEN)),
        intermediate_rotations=scales((1, 3 * KIMI_INTERMEDIATE)),
        down_svh=scales((1, KIMI_HIDDEN)),
        tile_config=(64, 128, 64, 128),
    )
    # Tier 1 has 0 routed experts in Kimi K2; projection-tight storage holds 1 dummy plane.
    tier1 = replace(
        tier1_full,
        w13=tier1_full.w13[: tier1_full.w13.numel() // 2].contiguous(),
    )
    global_to_combined, descriptor = build_projection_tiered_maps(
        (0,) * KIMI_EXPERTS,
        (0,) * KIMI_EXPERTS,
        (0,) * KIMI_EXPERTS,
        tier_slots=(KIMI_EXPERTS, 1),
        device=device,
    )
    rotations = combine_trellis_rotations(tier0, tier1)
    return tier0, tier1, global_to_combined, descriptor, rotations


def _run_kimi_benchmark(args, device: torch.device, props) -> None:
    variant_specs = [_parse_kimi_variant(v) for v in (args.kimi_variant or KIMI_DEFAULT_VARIANTS)]
    names = [name for name, *_ in variant_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"kimi variant names must be unique: {names}")
    rows_list = (
        [int(v) for v in args.rows.split(",")]
        if args.rows
        else list(KIMI_DEFAULT_ROWS)
    )
    if any(r < 1 or r > 80 for r in rows_list):
        raise ValueError("Kimi decode rows must be in 1..80")
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    layers = [_make_kimi_layer(1000 * (i + 1), device) for i in range(args.layers)]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    max_rows = max(rows_list)
    ids = torch.stack(
        [
            torch.randperm(KIMI_EXPERTS, generator=generator)[:KIMI_TOPK]
            for _ in range(max_rows)
        ]
    ).to(dtype=torch.int32, device=device)
    weights = torch.rand((max_rows, KIMI_TOPK), generator=generator).to(device=device)
    weights = 2.5 * weights / weights.sum(dim=-1, keepdim=True)
    x = (
        (torch.randn((max_rows, KIMI_HIDDEN), generator=generator) * 0.25)
        .to(torch.bfloat16)
        .to(device)
    )
    expert_k2_bytes = 3 * KIMI_HIDDEN * KIMI_INTERMEDIATE * KIMI_BITS[0] // 8
    print(
        f"device={props.name} sms={sms} H={KIMI_HIDDEN} I={KIMI_INTERMEDIATE} "
        f"experts={KIMI_EXPERTS}xK2 top{KIMI_TOPK} activation=situ "
        f"layers={args.layers} calls/graph={args.calls_per_graph} "
        f"replays={args.replays} rounds={args.rounds}",
        flush=True,
    )

    def build(capacity: int, schedule: str | None, tile: tuple[int, int, int, int], blocks: int | None):
        import os

        os.environ.setdefault("B12X_COMPILE_DISK_CACHE", "0")
        slots = route_pack_capacity(
            capacity * KIMI_TOPK, 8, KIMI_EXPERTS, topk=KIMI_TOPK
        )[1]
        launch = compile_mixed_trellis(
            size_m=capacity,
            hidden_size=KIMI_HIDDEN,
            intermediate_size=KIMI_INTERMEDIATE,
            tier0_num_experts=KIMI_EXPERTS,
            tier1_num_experts=1,
            top_k=KIMI_TOPK,
            route_num_experts=KIMI_EXPERTS,
            max_m_blocks=(slots + 7) // 8,
            sms=sms,
            max_shared_mem=max_shared_mem,
            force_tile_config=tile,
            tier0_bits=KIMI_BITS[0],
            tier1_bits=KIMI_BITS[1],
            trellis_codebook="mcg",
            swiglu_limit=None,
            activation="situ",
            moe_block_size=8,
            rotation_input_dtype="bf16",
            full_rotation_output_dtype="bf16",
            route_ids_dtype=torch.int32,
            force_blocks_per_sm=blocks,
            decode_schedule=schedule,
        )
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
        bindings = [
            bind_mixed_trellis(
                t0,
                t1,
                g2c,
                desc,
                rot,
                launch,
                gate_experts=(KIMI_EXPERTS, 0),
                up_experts=(KIMI_EXPERTS, 0),
            )
            for t0, t1, g2c, desc, rot in layers
        ]
        return launch, buffers, bindings

    printed_resources: set[tuple[str, int]] = set()
    for rows in rows_list:
        capacity = 1 if rows == 1 else 80
        xs = x[:rows].contiguous()
        ws = weights[:rows].contiguous()
        ks = ids[:rows].contiguous()
        distinct_experts = int(torch.unique(ks).numel())
        weight_bytes = distinct_experts * expert_k2_bytes
        state = {}
        ref_eager_tensors: list[torch.Tensor] | None = None
        for name, schedule, tile, blocks in variant_specs:
            launch, buffers, bindings = build(capacity, schedule, tile, blocks)
            res_info = _kernel_resource_info(launch, max_shared_mem)
            if (name, capacity) not in printed_resources:
                printed_resources.add((name, capacity))
                print(
                    f"resource m{capacity:<2d} variant={name:<16s} "
                    f"tile={','.join(map(str, tile))} threads={res_info['cta_threads']} "
                    f"blocks/SM_req={res_info['blocks_per_sm']} "
                    f"occ_ctas/SM={res_info['occ_ctas_per_sm']} "
                    f"occ_warps/SM={res_info['occ_warps_per_sm']} ({res_info['occ_pct']:.1f}%) "
                    f"regs={res_info['regs']} spills_B={res_info['spills_local_bytes']} "
                    f"dyn_smem_B={res_info['dyn_smem_bytes']} static_smem_B={res_info['static_smem_bytes']} "
                    f"schedule={launch.decode_schedule}",
                    flush=True,
                )
            layer_buffers = [
                replace(buffers, output=torch.empty_like(buffers.output))
                for _ in bindings
            ]
            eager_outputs = [
                run_bound_mixed_trellis(xs, ws, ks, binding, layer_buffer).clone()
                for binding, layer_buffer in zip(bindings, layer_buffers)
            ]
            if ref_eager_tensors is None:
                ref_eager_tensors = eager_outputs
            eager_digests = [_output_digest(out) for out in eager_outputs]

            def calls(bindings=bindings, layer_buffers=layer_buffers):
                for call in range(args.calls_per_graph):
                    layer = call % len(bindings)
                    run_bound_mixed_trellis(
                        xs, ws, ks, bindings[layer], layer_buffers[layer]
                    )

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
            state[name] = {
                "graph": graph,
                "eager": eager_digests,
                "eager_tensors": eager_outputs,
                "times_ms": [],
                "keep": (launch, buffers, layer_buffers, bindings, calls),
                "res": res_info,
                "tile": tile,
                "schedule": launch.decode_schedule,
            }

        for _ in range(args.rounds):
            for name, *_ in variant_specs:
                graph = state[name]["graph"]
                starts = [
                    torch.cuda.Event(enable_timing=True) for _ in range(args.replays)
                ]
                ends = [
                    torch.cuda.Event(enable_timing=True) for _ in range(args.replays)
                ]
                for start, end in zip(starts, ends, strict=True):
                    start.record()
                    graph.replay()
                    end.record()
                torch.cuda.synchronize()
                state[name]["times_ms"] += [
                    s.elapsed_time(e) / args.calls_per_graph
                    for s, e in zip(starts, ends, strict=True)
                ]

        assert ref_eager_tensors is not None
        reference = state[variant_specs[0][0]]
        ref_min_ms = min(reference["times_ms"])
        ref_med_ms = statistics.median(reference["times_ms"])
        for name, *_ in variant_specs:
            entry = state[name]
            replay_tensors = [
                lb.output[:rows].clone() for lb in entry["keep"][2]
            ]
            replay_digests = [_output_digest(t) for t in replay_tensors]
            bitwise_eager = all(
                torch.equal(r, e)
                for r, e in zip(replay_tensors, entry["eager_tensors"], strict=True)
            )
            bitwise_ref = all(
                torch.equal(r, ref)
                for r, ref in zip(replay_tensors, ref_eager_tensors, strict=True)
            )
            if not bitwise_eager:
                status = "REPLAY-DIFFERS"
            elif not bitwise_ref or replay_digests != reference["eager"]:
                status = "DIFFERENT"
            else:
                status = "BITWISE-EQUAL"
            min_ms = min(entry["times_ms"])
            med_ms = statistics.median(entry["times_ms"])
            res = entry["res"]
            print(
                f"rows={rows:2d} m{capacity:<2d} variant={name:<16s} "
                f"experts={distinct_experts:3d} "
                f"min={min_ms * 1000.0:7.1f}us ({min_ms:.4f}ms) "
                f"med={med_ms * 1000.0:7.1f}us ({med_ms:.4f}ms) "
                f"GB/s={weight_bytes / min_ms / 1e6:6.1f} "
                f"speedup_min={ref_min_ms / min_ms:6.3f}x "
                f"speedup_med={ref_med_ms / med_ms:6.3f}x "
                f"occ={res['occ_ctas_per_sm']}cta/SM regs={res['regs']} "
                f"spills={res['spills_local_bytes']}B "
                f"bits={'/'.join(replay_digests)} {status}",
                flush=True,
            )
            if status != "BITWISE-EQUAL":
                raise RuntimeError(
                    f"Kimi bitwise equality check failed for variant {name} at rows={rows}: {status}"
                )
        del state
        torch.cuda.empty_cache()


PREFILL_FAMILIES = {
    "qwen4": (2560, 512, 10, (4, 5), "silu"),
    "kimik3": (3584, 896, 16, (2, 3), "situ"),
}


def parse_prefill_arm(text):
    parts = text.split(":")
    if (len(parts) != 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", parts[0])
            or parts[0].endswith("_control") or parts[1] not in ("8", "16", "32", "64")
            or parts[2] not in ("cooperative", "ws")
            or parts[1:] == ["8", "ws"]):
        raise argparse.ArgumentTypeError("prefill arm must be NAME:BLOCK:cooperative|ws (WS block >=16)")
    return parts[0], int(parts[1]), parts[2]


def prefill_route_waves(path, family, layer):
    """Read a family-labelled histogram fixture; never silently reuse another family."""
    hidden, experts, topk, bits, activation = PREFILL_FAMILIES[family]
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != 1 or data.get("family") != family
            or data.get("experts") != experts or data.get("top_k") != topk):
        raise ValueError("route fixture family/expert/top-k metadata mismatch")
    waves = [wave for wave in data.get("waves", []) if wave.get("layer") == layer]
    if not waves:
        raise ValueError("route fixture has no waves for the requested layer")
    for wave in waves:
        rows = wave.get("rows", 0)
        counts = wave.get("expert_rows", [])
        if (not isinstance(rows, int) or rows <= 0 or not counts
                or any(not isinstance(e, int) or not isinstance(n, int)
                       or not 0 <= e < experts or not 0 < n <= rows for e, n in counts)
                or len({e for e, _ in counts}) != len(counts)
                or sum(n for _, n in counts) != rows * topk):
            raise ValueError("invalid recorded route histogram")
    return waves


def _make_prefill_layer(family, width, seed, device):
    if family == "kimik3":
        if width != KIMI_INTERMEDIATE:
            raise ValueError("Kimi prefill fixture follows the TP6 width512 package")
        return _make_kimi_layer(seed, device)
    hidden, experts, topk, bits, activation = PREFILL_FAMILIES[family]
    generator = torch.Generator(device=device).manual_seed(seed)
    tile_n = 256 if width % 256 == 0 else 128
    tiers = []
    for i, (bit, count) in enumerate(zip(bits, (3 * experts // 4, experts // 4))):
        def scales(shape):
            return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).half()
        tiers.append(prepare_trellis256_moe_weights(
            hidden_size=hidden, intermediate_size=width, num_experts=count,
            activation=activation, fc1_tile_n=tile_n, fc2_tile_n=tile_n,
            device=device, seed=seed + i, params_dtype=torch.float16,
            w13_layout="trellis_t256_proj", trellis_bits=bit, codebook="mcg",
            gate_suh=scales((count, hidden)), up_suh=scales((count, hidden)),
            intermediate_rotations=scales((count, 3 * width)),
            down_svh=scales((count, hidden)), tile_config=(64, tile_n, 64, tile_n)))
    mapping, descriptor = build_ordered_maps(*(t.num_experts for t in tiers), device=device)
    return *tiers, mapping, descriptor, combine_trellis_rotations(*tiers)


def compact_prefill_fixture(path, rank, family, width, experts):
    """Validate a planner-replayed ownership fixture, not a synthetic partition."""
    if family not in ("qwen4", "glm"):
        raise ValueError("compact prefill fixture currently requires Qwen or GLM")
    data = json.loads(Path(path).read_text())
    matches = [item for item in data["ranks"] if item["rank"] == rank]
    if len(matches) != 1:
        raise ValueError("compact fixture must contain the requested rank exactly once")
    selected = matches[0]
    entries = sorted(selected["experts"], key=lambda e: e["combined_local"])
    if (selected["compiled_max_intermediate"] != width or len(entries) != experts
            or [e["combined_local"] for e in entries] != list(range(experts))
            or sorted(e["global_expert"] for e in entries) != list(range(experts))):
        raise ValueError("compact fixture geometry or expert namespace mismatch")
    for e in entries:
        if e["width"] not in range(128, width + 1, 128):
            raise ValueError("compact fixture has invalid live width")
        if [p["projection"] for p in e["projections"]] != ["gate", "up", "down"]:
            raise ValueError("compact fixture requires ordered gate/up/down projections")
        for projection in e["projections"]:
            if projection["tier"] not in (0, 1) or projection["bits"] != 4 + projection["tier"]:
                raise ValueError("compact fixture tier/bit mismatch")
    return dict(rank=rank, entries=entries, counts=selected["projection_counts"],
                fingerprint=selected["unit_map_fingerprint"])


def _make_compact_prefill_layer(fixture, width, seed, device, mix, *,
                                hidden, experts, bits, activation):
    # Serving exports reserve E slots per tier, independently of projection counts.
    total = 2 * experts
    rows = [[-1] * total for _ in range(3)]
    widths = [-1] * total
    mapping = [-1] * experts
    for e in fixture["entries"]:
        local = e["combined_local"]
        widths[local] = e["width"]
        mapping[e["global_expert"]] = local
        for p, projection in enumerate(e["projections"]):
            rows[p][local] = (projection["tier"] << 9) | projection["tier_local"]
    canonical, payloads, counts = mix._compact_trellis_layout(
        [*rows, widths], hidden_size=hidden, intermediate_size=width, bits=bits)
    if [list(c) for c in counts] != fixture["counts"]:
        raise ValueError("compact fixture projection counts disagree")
    for e in fixture["entries"]:
        for p, projection in enumerate(e["projections"]):
            if canonical[4 + p][e["combined_local"]] != projection["payload_word_offset"]:
                raise ValueError("compact fixture payload offset is not canonical")
    generator = torch.Generator(device=device).manual_seed(seed)
    tiers = []
    for tier, bit in enumerate(bits):
        template = prepare_trellis256_moe_weights(
            hidden_size=hidden, intermediate_size=width, num_experts=1,
            activation=activation, fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + tier, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
            trellis_bits=bit, codebook="mcg", tile_config=(64, 128, 64, 128))
        def packed(size):
            return torch.randint(-(1 << 31), 1 << 31, (max(size, 4),),
                                 generator=generator, device=device, dtype=torch.int32)
        tiers.append(replace(template, num_experts=experts,
            w13=packed(payloads[tier][0]), w2=packed(payloads[tier][1]),
            w13_global_scale=template.w13_global_scale.repeat(experts),
            w2_global_scale=template.w2_global_scale.repeat(counts[tier][2])))
    def scales(size):
        return (0.875 + 0.25 * torch.rand(size, generator=generator, device=device)).half()
    rotations = mix.MixedTrellisRotations(
        scales(total * 3 * width), scales(total * hidden),
        scales(total * hidden), scales(total * hidden))
    descriptor = torch.tensor(canonical, dtype=torch.int32, device=device).flatten()
    descriptor._mt_projection_counts = (tuple(c[0] for c in counts), tuple(c[1] for c in counts))
    return (*tiers, torch.tensor(mapping, dtype=torch.int32, device=device), descriptor, rotations)


def prefill_compile_options(family, width, block, mode, props):
    hidden, experts, topk, bits, activation = PREFILL_FAMILIES[family]
    if width <= 0 or width % 128 or (family == "kimik3" and width != 512):
        raise ValueError("prefill width must follow an H128-aligned family package")
    slots = route_pack_capacity(4096 * topk, block, experts, topk=topk)[1]
    tile_n = 256 if width % 256 == 0 else 128
    counts = (384, 128) if family == "qwen4" else (896, 1)
    return dict(size_m=4096, hidden_size=hidden, intermediate_size=width,
        tier0_num_experts=counts[0], tier1_num_experts=counts[1],
        top_k=topk, route_num_experts=experts, max_m_blocks=(slots + block - 1) // block,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=(64, tile_n, 64, tile_n), tier0_bits=bits[0], tier1_bits=bits[1],
        trellis_codebook="mcg", activation=activation, swiglu_limit=None,
        moe_block_size=block, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
        warp_specialized=mode == "ws",
        token_major_rotation=family == "qwen4" and mode == "cooperative")


def _run_prefill_benchmark(args, device, props):
    from benchmarks.benchmark_exlr8_a16 import (
        materialize_k4_arm, load_k4_arm, k4_noise_summary, real_route_ids,
    )
    from b12x._lib.runtime_control import kernel_resolution_guard
    if (props.major, props.minor) != (12, 1):
        raise ValueError("prefill policy gate requires SM121")
    if args.rounds < 3 or args.layers < 2 or args.replays < 1:
        raise ValueError("prefill gate requires >=3 rounds, >=2 cold layers and positive replays")
    specs = args.prefill_arm or [parse_prefill_arm(v) for v in (
        ("coop:32:cooperative", "ws:32:ws") if args.prefill_family == "qwen4"
        else ("coop8:8:cooperative", "coop16:16:cooperative", "ws16:16:ws"))]
    names = [name for name, _, _ in specs]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("prefill needs at least two uniquely named arms")
    rows_list = list(map(int, (args.rows or "1408,2688,4096").split(",")))
    if not rows_list or any(not 1 <= rows <= 4096 for rows in rows_list):
        raise ValueError("prefill rows must be 1..4096")
    family = args.prefill_family
    hidden, experts, topk, bits, activation = PREFILL_FAMILIES[family]
    width = args.intermediate or (256 if family == "qwen4" else 512)
    waves = prefill_route_waves(args.routes, family, args.route_layer) if args.routes else None
    # No recorded family routes means correctness only, never a timing claim.
    torch.cuda.set_per_process_memory_fraction(min(
        0.25, (args.memory_cap_gib << 30) / props.total_memory), device)
    bundle = args.prefill_source_dir
    if bundle is None:
        bundle = materialize_k4_arm(args.prefill_revision,
                                   Path(tempfile.mkdtemp(prefix="family-prefill-")))
    mix, provenance = load_k4_arm(bundle)
    compact = (compact_prefill_fixture(args.compact_fixture, args.compact_rank, family, width,
                                       experts) if args.compact_fixture else None)
    layers = [(_make_compact_prefill_layer(compact, width, args.seed + 2*i, device, mix,
                                           hidden=hidden, experts=experts, bits=bits,
                                           activation=activation)
               if compact else _make_prefill_layer(family, width, args.seed + 2*i, device))
              for i in range(args.layers)]
    states = {}
    for name, block, mode in specs:
        options = prefill_compile_options(family, width, block, mode, props)
        if compact:
            options.update(tier0_num_experts=experts, tier1_num_experts=experts,
                           force_tile_config=(64, 128, 64, 128), compact_weights=True)
        launch = mix.compile_mixed_trellis(**options)
        workspace = mix.make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)
        bounds = dict(gate_experts=(896, 0), up_experts=(896, 0)) if family == "kimik3" else {}
        if compact:
            bounds = dict(gate_experts=tuple(c[0] for c in compact["counts"]),
                          up_experts=tuple(c[1] for c in compact["counts"]))
        bindings = [mix.bind_mixed_trellis(*layer, launch, **bounds) for layer in layers]
        mix.warmup_mixed_trellis_route_pack(launch, workspace, expert_map=bindings[0].global_to_combined)
        states[name] = workspace, bindings
    generator = torch.Generator().manual_seed(args.seed)
    ids = [(real_route_ids(4096, waves[i % len(waves)], generator, experts, topk)
            if waves else torch.rand((4096, experts), generator=generator).topk(topk, dim=1).indices
            ).to(device=device, dtype=torch.int32) for i in range(args.layers)]
    weights = torch.rand((4096, topk), generator=generator).to(device)
    weights /= weights.sum(-1, keepdim=True)
    x = (torch.randn((4096, hidden), generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)

    def run(name, rows, layer=0):
        workspace, bindings = states[name]
        return mix.run_bound_mixed_trellis(x[:rows], weights[:rows], ids[layer][:rows], bindings[layer], workspace)

    for name in names:
        for rows in (1, 17, 4096):
            run(name, rows)
    checked = list(range(1, 4097)) if args.prefill_exhaustive else sorted({1, 17, 63, 64, 65, 4096, *rows_list})
    records = []
    with kernel_resolution_guard("family prefill bitwise and frozen timing"):
        if compact:
            for workspace, _ in states.values():
                workspace.fc1.fill_(float("nan"))
                workspace.activated.fill_(float("nan"))
        for layer in range(args.layers if compact else 1):
            full = run(names[0], 4096, layer).clone()
            if not bool(torch.isfinite(full).all()) or not bool(full.any()):
                raise RuntimeError("reference output is not finite and nonzero")
            for rows in checked:
                expected = run(names[0], rows, layer).clone()
                if not torch.equal(expected.view(torch.int16), full[:rows].view(torch.int16)):
                    raise RuntimeError(f"reference prefix invariance failed at {rows}, layer {layer}")
                for name in names[1:]:
                    if not torch.equal(run(name, rows, layer).view(torch.int16), expected.view(torch.int16)):
                        raise RuntimeError(f"prefill arm {name} differs at {rows}, layer {layer}")
        if compact:
            original_x = x.clone()
            for rows in rows_list:
                for layer in range(args.layers):
                    outputs = {}
                    def replay_call(name):
                        outputs[name] = run(name, rows, layer)
                    graphs = {name: _capture(lambda name=name: replay_call(name), 3) for name in names}
                    for _ in range(3):
                        x.mul_(-0.75)
                        for workspace, _bindings in states.values():
                            workspace.fc1.fill_(float("nan"))
                            workspace.activated.fill_(float("nan"))
                        for graph in graphs.values():
                            graph.replay()
                        reference = outputs[names[0]].clone()
                        if not bool(torch.isfinite(reference).all()):
                            raise RuntimeError("compact poisoned reference is nonfinite")
                        for name in names[1:]:
                            if not torch.equal(reference.view(torch.int16), outputs[name].view(torch.int16)):
                                raise RuntimeError(f"compact poisoned graph differs: {name}, rows {rows}, layer {layer}")
            x.copy_(original_x)
        for rows in rows_list:
            for layer in range(args.layers):
                expected = run(names[0], rows, layer).clone()
                for name in names[1:]:
                    if not torch.equal(run(name, rows, layer).view(torch.int16), expected.view(torch.int16)):
                        raise RuntimeError(f"prefill arm {name} differs on layer {layer}, rows {rows}")
            if waves is None or args.correctness_only:
                continue
            def calls(name):
                for layer in range(args.layers):
                    run(name, rows, layer)
            graphs = {name: _capture(lambda name=name: calls(name), 3) for name in names}
            graphs[names[0] + "_control"] = _capture(lambda: calls(names[0]), 3)
            samples = {name: [] for name in graphs}
            order = list(graphs)
            for repeat in range(args.rounds):
                rotated = order[repeat % len(order):] + order[:repeat % len(order)]
                if repeat % 2:
                    rotated.reverse()
                for name in rotated:
                    samples[name].append(statistics.median(_times(graphs[name], args.replays)) / args.layers)
            record = dict(rows=rows, samples_us=samples, **k4_noise_summary(samples, names[0]))
            records.append(record)
            print(json.dumps(record), flush=True)
    result = dict(schema=1, family=family, intermediate=width, arms=specs, source=provenance,
        fixture=("synthetic weights; actual compact ownership" if compact else
                 "synthetic weights; K4/K5 expert split 384/128" if family == "qwen4"
                 else "synthetic uniform K2 with dormant K3"),
        compact_rank=compact["rank"] if compact else None,
        compact_sha256=hashlib.sha256(args.compact_fixture.read_bytes()).hexdigest() if compact else None,
        routes="recorded histograms" if waves else "uniform random; correctness only",
        route_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest() if waves else None,
        route_layer=args.route_layer, checked_rows=len(checked), exact=True, rows=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(exact=True, checked_rows=len(checked), timed=bool(records))), flush=True)
    if any(not record["noise_pass"] for record in records):
        raise RuntimeError("identical-control spread exceeds 2%; no performance verdict")


GLM_CAPACITIES = (256, 1024, 2752, 4096)
GLM_PROFILES = {"rtx-tp1": 2048, "rtx-tp2": 1024,
                "spark-tp6-width384": 384, "spark-tp6-width256": 256, "spark-tp4-width512": 512}
GLM_ROUTES = Path(__file__).resolve().parent / "fixtures/glm53_prefill_routes_realistic.json"
COORDINATOR_FAMILIES = {
    "glm": dict(hidden=6144, experts=256, top_k=8, bits=(4, 5), swiglu_limit=None,
                model="zai-org/GLM-5.3", profiles=tuple(GLM_PROFILES), routes=GLM_ROUTES),
    "glmf": dict(hidden=4096, experts=288, top_k=8, bits=(3, 4), swiglu_limit=10.0,
                 model="zai-org/GLM-5.3-Flash", profiles=("rtx-tp1", "rtx-tp2"), routes=None),
}


def parse_coordinator_arm(text):
    """NAME:key=value,...; slash-separated tile dimensions keep commas unambiguous."""
    name, separator, tail = text.partition(":")
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name == "coop":
        raise argparse.ArgumentTypeError("coordinator arm needs a unique NAME:key=value,... (coop is reserved)")
    options = {}
    try:
        for item in tail.split(","):
            key, value = item.split("=")
            if key in options:
                raise ValueError("duplicate key")
            if key in ("mode", "rotation"):
                if value not in (("cooperative", "ws") if key == "mode" else ("served", "route", "token", "fused")):
                    raise ValueError("invalid mode/rotation")
            elif key == "tile":
                value = tuple(map(int, value.split("/")))
                if len(value) != 4 or any(v < 32 or v % 16 for v in value):
                    raise ValueError("tile needs four positive, 16-aligned dimensions")
            elif key in ("route_block", "ctas", "stages", "wire", "dynamic"):
                value = int(value)
                legal = {"route_block": (8, 16, 32, 64), "ctas": (1, 2, 3, 4),
                         "stages": range(2, 7), "wire": (0, 1), "dynamic": (0, 1)}
                if value not in legal[key]:
                    raise ValueError("invalid integer override")
            else:
                raise ValueError("unknown override")
            options[key] = value
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid coordinator arm {text!r}: {exc}") from exc
    return name, options


def coordinator_compile_options(profile, capacity, arm, props, family="glm", compact=False):
    """Mirror GLM/GLMF's cooperative coordinator export, with explicit experimental overrides."""
    spec = COORDINATOR_FAMILIES.get(family)
    if spec is None or profile not in spec["profiles"] or capacity not in GLM_CAPACITIES:
        raise ValueError("GLM requires a supported RTX/Spark profile and capacity 256/1024/2752/4096")
    hidden, experts, topk = spec["hidden"], spec["experts"], spec["top_k"]
    tier0_bits, tier1_bits = spec["bits"]
    width = GLM_PROFILES[profile]
    spark = profile.startswith("spark-")
    ws = arm.get("mode", "cooperative") == "ws"
    if compact:
        # The shipped compact export is TP6 width 384 only: live widths are
        # 256/384, so FC1 keeps 128-wide N tiles (a 192 tile would straddle
        # the live edge) while FC2 may widen over the fixed hidden size.
        if family != "glm" or profile != "spark-tp6-width384":
            raise ValueError("compact coordinator requires GLM spark-tp6-width384")
        block = arm.get("route_block", 32 if ws else 64)
        rotation = arm.get("rotation", "route" if ws else "token")
        token = rotation == "token" or (rotation == "served" and capacity > 256)
        fused = rotation == "fused"
        if ((ws and (block < 16 or token or fused)) or (fused and block < 16)
                or (not ws and any(arm.get(k) for k in ("wire", "stages", "dynamic")))):
            raise ValueError("WS needs block >=16 and its own rotation; wire/stages/dynamic require WS")
        tile = arm.get("tile", (64, 128, 64, 128))
        if tile[:3] != (64, 128, 64) or tile[3] not in (128, 256):
            raise ValueError("compact coordinator fixes the H128 K64 geometry and FC2 tile 128/256")
        if width % tile[1] or hidden % tile[3]:
            raise ValueError("coordinator tile must divide both projection geometries")
    else:
        block = arm.get("route_block", 64 if ws or capacity > 256 else 8)
        rotation = arm.get("rotation", "route" if ws else "served")
        token = rotation == "token" or (rotation == "served" and capacity > 256)
        fused = rotation == "fused"
        if ((ws and (block < 16 or token or fused)) or (fused and block < 16)
                or (not ws and any(arm.get(k) for k in ("wire", "stages", "dynamic")))):
            raise ValueError("WS needs block >=16 and its own rotation; wire/stages/dynamic require WS")
        tile = arm.get("tile", (64, 128 if width == 384 else 256, 64, 128 if width == 384 else 256))
        if width % tile[1] or hidden % tile[3] or hidden % tile[0] or width % tile[2]:
            raise ValueError("coordinator tile must divide both projection geometries")
        if not ws and tile[1] != tile[3]:
            raise ValueError("cooperative FC1/FC2 tiles must have equal CTA thread counts")
    slots = route_pack_capacity(capacity * topk, block, experts, topk=topk)[1]
    return dict(size_m=capacity, hidden_size=hidden, intermediate_size=width,
        tier0_num_experts=experts, tier1_num_experts=experts, tier0_bits=tier0_bits, tier1_bits=tier1_bits,
        route_num_experts=experts, top_k=topk, max_m_blocks=(slots + block - 1) // block,
        sms=int(props.multi_processor_count), max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=tile, trellis_codebook="mcg", swiglu_limit=spec["swiglu_limit"],
        moe_block_size=block, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16" if spark else "fp32",
        direct_topk_routes=False, token_major_rotation=token, fused_input_rotation=fused,
        warp_specialized=ws, compact_weights=compact,
        input_format="e4m3_k32" if arm.get("wire") else "bf16",
        force_blocks_per_sm=arm.get("ctas"), ws_input_stages=arm.get("stages"),
        ws_dynamic_tiles=bool(arm.get("dynamic", False)))


def coordinator_route_waves(path, layer=None, family="glm"):
    spec = COORDINATOR_FAMILIES.get(family)
    if spec is None or path is None:
        raise ValueError(f"no realistic route fixture for coordinator family {family!r}")
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != 1 or data.get("model") != spec["model"]
            or data.get("family", family) != family
            or data.get("experts", spec["experts"]) != spec["experts"]
            or data.get("top_k", spec["top_k"]) != spec["top_k"]):
        raise ValueError(f"coordinator fixture must be schema1 {spec['model']} ({family}) histograms")
    waves = [w for w in data.get("waves", []) if layer is None or w.get("layer") == layer]
    if not waves:
        raise ValueError("coordinator fixture has no selected waves")
    for wave in waves:
        rows, counts = wave.get("rows", 0), wave.get("expert_rows", [])
        if (not isinstance(rows, int) or rows < 1 or not counts
                or any(not isinstance(e, int) or not isinstance(n, int)
                       or not 0 <= e < spec["experts"] or not 0 < n <= rows for e, n in counts)
                or len({e for e, _ in counts}) != len(counts)
                or sum(n for _, n in counts) != rows * spec["top_k"]):
            raise ValueError("invalid GLM coordinator route histogram")
    return waves


def coordinator_preflight(args):
    family = args.prefill_family
    spec = COORDINATOR_FAMILIES.get(family)
    if spec is None or args.output is None:
        raise ValueError("GLM requires --prefill-family glm|glmf --output")
    if (args.role == "spark") != args.profile.startswith("spark-") or args.profile not in spec["profiles"]:
        raise ValueError("GLM --role must match the RTX or Spark --profile")
    if (args.kimi or args.prefill_arm or args.prefill_source_dir
            or args.rows or args.intermediate or args.prefill_exhaustive):
        raise ValueError("coordinator uses --profile/--capacities/--coordinator-arm, not Spark/decode controls")
    compact = None
    if args.compact_fixture:
        if args.profile != "spark-tp6-width384" or args.k5_experts:
            raise ValueError("compact coordinator requires --profile spark-tp6-width384 and --k5-experts 0")
        compact = compact_prefill_fixture(args.compact_fixture, args.compact_rank, family,
                                          GLM_PROFILES[args.profile], spec["experts"])
    capacities = [int(v) for v in args.capacities.split(",")]
    if not capacities or len(capacities) != len(set(capacities)):
        raise ValueError("capacities must be nonempty and unique")
    specs = [("coop", {})]
    if args.role == "spark":
        specs.append(("served", (dict(mode="ws") if compact else
                                dict(mode="ws", wire=1, stages=3, dynamic=1, ctas=1,
                                     tile=((64, 192, 64, 256) if GLM_PROFILES[args.profile] == 384
                                           else (64, 256, 64, 256))))))
    specs.extend(args.coordinator_arm)
    if len({name for name, _ in specs}) != len(specs):
        raise ValueError("coordinator arm names must be unique")
    if min(args.layers, args.rounds, args.replays, args.warmup, args.memory_cap_gib) < 1:
        raise ValueError("layers/rounds/replays/warmup/memory cap must be positive")
    if not 0 <= args.k5_experts <= spec["experts"]:
        raise ValueError(f"upgraded projection count must be 0..{spec['experts']}")
    routes = args.route_distributions.split(",")
    if not routes or len(set(routes)) != len(routes) or set(routes) - {"random", "realistic"}:
        raise ValueError("route distributions must be random and/or realistic")
    waves = (coordinator_route_waves(args.routes or spec["routes"], args.coordinator_route_layer, family)
             if "realistic" in routes or args.routes is not None else [])
    # CPU-only placeholders validate argument plumbing, never describe a timed device.
    props = SimpleNamespace(multi_processor_count=1, shared_memory_per_block_optin=101376)
    for capacity in capacities:
        for _, arm in specs:
            coordinator_compile_options(args.profile, capacity, arm, props, family, compact is not None)
    return capacities, specs, routes, waves, compact


def _make_coordinator_layer(width, k5_experts, seed, device, family="glm"):
    """Synthetic projection-major tier payloads with the export's full expert tier slots."""
    spec = COORDINATOR_FAMILIES[family]
    hidden, experts = spec["hidden"], spec["experts"]
    tier0_bits, tier1_bits = spec["bits"]
    tiers = []
    for tier, (bits, count) in enumerate(zip((tier0_bits, tier1_bits), (experts - k5_experts, k5_experts))):
        prepared = prepare_trellis256_moe_weights(
            hidden_size=hidden, intermediate_size=width, num_experts=max(count, 1),
            activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + tier, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
            trellis_bits=bits, codebook="mcg", tile_config=(64, 128, 64, 128))
        w13 = prepared.w13 if count else prepared.w13[:prepared.w13.numel() // 2]
        scales = torch.ones(experts, dtype=torch.float32, device=device)
        scales[:max(count, 1)].copy_(prepared.w13_global_scale)
        tiers.append(replace(prepared, num_experts=experts, w13=w13, w13_global_scale=scales))
    membership = [int(e >= experts - k5_experts) for e in range(experts)]
    # Distinct projection memberships exercise the same per-projection descriptor ABI as checkpoints.
    mapping, descriptor = build_projection_tiered_maps(
        membership, membership[1:] + membership[:1], membership[2:] + membership[:2],
        tier_slots=(experts, experts), device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).half()
    rotations = MixedTrellisRotations(scales((2 * experts, 3 * width)), scales((2 * experts, hidden)),
                                     scales((2 * experts, hidden)), scales((2 * experts, hidden)))
    return *tiers, mapping, descriptor, rotations


def _run_coordinator_benchmark(args, device, props):
    from benchmarks.benchmark_exlr8_a16 import make_wire, real_route_ids
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mix
    capacities, specs, distributions, waves, compact = coordinator_preflight(args)
    spec = COORDINATOR_FAMILIES[args.prefill_family]
    hidden, experts, topk = spec["hidden"], spec["experts"], spec["top_k"]
    target = (12, 1) if args.role == "spark" else (12, 0)
    if (props.major, props.minor) != target:
        raise ValueError(f"GLM {args.role} benchmark requires SM{target[0]}{target[1]}")
    torch.cuda.set_per_process_memory_fraction(min(0.9, (args.memory_cap_gib << 30) / props.total_memory), device)
    width = GLM_PROFILES[args.profile]
    output_dtype = torch.bfloat16 if args.role == "spark" else torch.float32
    bit_dtype = torch.int16 if args.role == "spark" else torch.int32
    layers = [(_make_compact_prefill_layer(compact, width, args.seed + 2*i, device, mix,
                                           hidden=hidden, experts=experts, bits=spec["bits"],
                                           activation="silu")
               if compact else
               _make_coordinator_layer(width, args.k5_experts, args.seed + 2*i, device, args.prefill_family))
              for i in range(args.layers)]
    counts = (tuple(tuple(c[p] for c in compact["counts"]) for p in (0, 1)) if compact
              else (experts - args.k5_experts, args.k5_experts))
    bind_counts = (dict(gate_experts=counts[0], up_experts=counts[1]) if compact
                   else dict(gate_experts=counts, up_experts=counts))
    records, failures = [], []
    for capacity in capacities:
        states = {}
        for name, arm in specs:
            options = coordinator_compile_options(args.profile, capacity, arm, props, args.prefill_family,
                                                  compact is not None)
            print(json.dumps(dict(compiling=name, capacity=capacity, options=options)), flush=True)
            launch = mix.compile_mixed_trellis(**options)
            workspace = mix.make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)
            bindings = [mix.bind_mixed_trellis(*layer, launch, **bind_counts) for layer in layers]
            mix.warmup_mixed_trellis_route_pack(launch, workspace, expert_map=bindings[0].global_to_combined)
            states[name] = workspace, bindings
        generator = torch.Generator().manual_seed(args.seed + capacity)
        raw = (torch.randn((capacity, hidden), generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        # The BF16 oracle consumes exactly the decoded wire values, not the pre-quantized input.
        wire = make_wire(raw)
        exponent = wire[:, hidden:].float() - 127
        decoded = (wire[:, :hidden].contiguous().view(torch.float8_e4m3fn).float().reshape(capacity, hidden // 32, 32)
                   * torch.exp2(exponent).unsqueeze(-1)).reshape(capacity, hidden).bfloat16()
        weights = torch.rand((capacity, topk), generator=generator).to(device)
        weights /= weights.sum(-1, keepdim=True)
        for distribution in distributions:
            selected = list(enumerate(waves)) if distribution == "realistic" else [(None, None)]
            for wave_index, wave in selected:
                ids = (real_route_ids(capacity, wave, generator, experts, topk) if wave else
                       torch.rand((capacity, experts), generator=generator).topk(topk, dim=1).indices).to(device=device, dtype=torch.int32)
                for layer in range(args.layers):
                    def run(name):
                        workspace, bindings = states[name]
                        x = wire if bindings[layer].launch.input_format == "e4m3_k32" else decoded
                        return mix.run_bound_mixed_trellis(x, weights, ids, bindings[layer], workspace)
                    expected = run("coop").clone()
                    if expected.dtype != output_dtype or not bool(torch.isfinite(expected).all()) or not bool(expected.any()):
                        raise RuntimeError("cooperative oracle must use the role's sum dtype and be finite and nonzero")
                    graphs, checks = {}, {}
                    for name, _ in specs:
                        actual = run(name)
                        checks[name] = torch.equal(actual.view(bit_dtype), expected.view(bit_dtype))
                        workspace, bindings = states[name]
                        # Freeze routing outside the graph: time the real bound core + top-k sum
                        # without copying its raw ABI or charging precomputed route-pack work.
                        launch = bindings[layer].launch
                        slots = route_pack_capacity(capacity * topk, launch.moe_block_size, experts, topk=topk)[1]
                        packed = (workspace.packed_route_indices[:slots], workspace.block_expert_ids, workspace.packed_route_count)
                        with patch.object(mix, "pack_topk_routes_by_expert", return_value=packed):
                            graphs[name] = _capture(lambda name=name: run(name), args.warmup)
                        # Compact descriptors carry per-claim row state; a frozen
                        # graph must rebuild it every replay, not reuse a warm ring.
                        for _ in range(3 if compact else 1):
                            workspace.output.fill_(float("nan"))
                            if compact:
                                workspace.fc1.fill_(float("nan"))
                                workspace.activated.fill_(float("nan"))
                            graphs[name].replay()
                            checks[name] &= torch.equal(workspace.output.view(bit_dtype), expected.view(bit_dtype))
                    samples = {name: [] for name, _ in specs}
                    if all(checks.values()) and not args.correctness_only:
                        names = list(graphs)
                        with kernel_resolution_guard("SM120 coordinator frozen graph timing"):
                            for repeat in range(args.rounds):
                                order = names[repeat % len(names):] + names[:repeat % len(names)]
                                if repeat % 2:
                                    order.reverse()
                                for name in order:
                                    samples[name].append(statistics.median(_times(graphs[name], args.replays)))
                    record = dict(capacity=capacity, distribution=distribution, weight_layer=layer,
                        wave_index=wave_index, route_layer=wave.get("layer") if wave else None,
                        content=wave.get("content") if wave else None, bitwise=checks,
                        samples_us=samples, median_us={name: statistics.median(v) for name, v in samples.items() if v},
                        output_sha256=hashlib.sha256(expected.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                    records.append(record)
                    if not all(checks.values()):
                        failures.append({k: record[k] for k in ("capacity", "distribution", "wave_index", "bitwise")})
                    print(json.dumps(record), flush=True)
                    del graphs, expected, actual
        del states, workspace, bindings, launch
        torch.cuda.empty_cache()
    source = Path(__file__).resolve().parents[1]
    result = dict(schema=1, kind="glm_exl3_prefill" if args.prefill_family == "glm" else f"{args.prefill_family}_exl3_prefill",
        role=args.role, profile=args.profile,
        device=props.name, sm_count=props.multi_processor_count, hidden=hidden, intermediate=width,
        experts=experts, top_k=topk, bits=list(spec["bits"]),
        k5_experts_per_projection=None if compact else args.k5_experts,
        compact=(dict(rank=compact["rank"], projection_counts=[list(c) for c in compact["counts"]],
                      unit_map_fingerprint=compact["fingerprint"],
                      sha256=hashlib.sha256(args.compact_fixture.read_bytes()).hexdigest())
                 if compact else None),
        arms=specs,
        source_sha256={p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in (
            "benchmarks/benchmark_mixed_trellis.py", "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py",
            "b12x/moe/_shared/kernels/w4a16/mixed_trellis_ws.py")},
        route_sha256=hashlib.sha256((args.routes or spec["routes"]).read_bytes()).hexdigest() if waves else None,
        output_dtype=str(output_dtype),
        conditions="synthetic projection-major weights; role-native sums (RTX FP32, Spark BF16); common E4M3/UE8M0 K32-roundtripped BF16 input; "
                   "histogram replay preserves rescaled expert counts, not original token co-occurrence; "
                   "warm CUDA graphs, core+top-k sum only, routing and wire preparation excluded; "
                   "median of interleaved round medians, microseconds per layer"
                   + ("; compact tiers use planner-replayed rotated H128 ownership and poisoned graph replays"
                      if compact else ""),
        exact=not failures, failures=failures, rows=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} coordinator cases differ bitwise; see {args.output}")


def benchmark_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--materialized-experts", type=int, default=16)
    parser.add_argument(
        "--variant",
        choices=("both", "serial", "serial-overlap", "mixed"),
        default="both",
    )
    parser.add_argument(
        "--separate-fc2",
        action="store_true",
        help="A/B control: replace the aliased FC2 output with separate storage",
    )
    parser.add_argument(
        "--kimi",
        action="store_true",
        help="Benchmark Kimi K3 TP6 Spark geometry (E896, top-16, I=512, EXL3 K2, situ) across 1-CTA and 2-CTA variants",
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="Comma-separated decode row counts for --kimi (default: 1,2,4,6,8,12,16,24,32,36)",
    )
    parser.add_argument(
        "--kimi-variant",
        action="append",
        default=[],
        help="Variant spec NAME:SCHEDULE:TILE:BLOCKS_PER_SM for --kimi",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--prefill-family", choices=(*PREFILL_FAMILIES, *COORDINATOR_FAMILIES))
    parser.add_argument("--role", choices=("spark", "coordinator"), default="spark")
    parser.add_argument("--profile", choices=tuple(GLM_PROFILES), default="rtx-tp2")
    parser.add_argument("--capacities", default="256,1024,2752,4096", help="GLM coordinator compile capacities")
    parser.add_argument("--coordinator-arm", action="append", type=parse_coordinator_arm, default=[],
                        help="NAME:mode=cooperative|ws,rotation=served|route|token|fused,tile=64/256/64/256,"
                             "route_block=64,ctas=1,stages=3,wire=1,dynamic=1; coop baseline is automatic")
    parser.add_argument("--route-distributions", default="random,realistic", help="GLM random and/or realistic histograms")
    parser.add_argument("--coordinator-route-layer", type=int, help="GLM histogram layer filter; default all 72 waves")
    parser.add_argument("--k5-experts", type=int, default=0, help="synthetic K5 count per projection (0..256); K4 otherwise")
    parser.add_argument("--preflight-only", action="store_true", help="CPU-only GLM coordinator argument/fixture validation")
    parser.add_argument("--compact-fixture", type=Path, help="planner-replayed compact ownership JSON")
    parser.add_argument("--compact-rank", type=int, default=0)
    parser.add_argument("--prefill-arm", action="append", type=parse_prefill_arm, default=[])
    parser.add_argument("--intermediate", type=int, help="per-rank H128 width, default Qwen256/Kimi512")
    parser.add_argument("--routes", type=Path, help="family-labelled recorded histogram JSON; absent means correctness only")
    parser.add_argument("--route-layer", type=int, default=23)
    parser.add_argument("--prefill-exhaustive", action="store_true", help="check every prefix row count 1..4096")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--prefill-revision", default="HEAD", help="freeze implementation from this git revision")
    parser.add_argument("--prefill-source-dir", type=Path, help="pre-materialized K4 sibling-module bundle")
    parser.add_argument("--memory-cap-gib", type=int, default=24)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = benchmark_parser()
    args = parser.parse_args()
    if args.prefill_family and (args.kimi or args.output is None or args.memory_cap_gib <= 0):
        parser.error("--prefill-family requires --output, positive memory cap, and no --kimi")
    if args.prefill_family in COORDINATOR_FAMILIES or args.role == "coordinator" or args.preflight_only:
        try:
            capacities, specs, distributions, waves, compact = coordinator_preflight(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.preflight_only:
            print(json.dumps(dict(preflight=True, profile=args.profile, capacities=capacities,
                                  arms=specs, distributions=distributions, histogram_waves=len(waves),
                                  compact_rank=compact["rank"] if compact else None)))
            return
    elif args.coordinator_arm:
        parser.error("--coordinator-arm requires GLM coordinator mode")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if args.prefill_family in COORDINATOR_FAMILIES:
        _run_coordinator_benchmark(args, device, props)
        return
    if args.prefill_family:
        _run_prefill_benchmark(args, device, props)
        return
    if args.kimi:
        if args.replays == 100:
            args.replays = 20
        _run_kimi_benchmark(args, device, props)
        return
    torch.manual_seed(20260730)
    materialized = int(args.materialized_experts)
    if materialized < 6 or materialized > K4_EXPERTS:
        raise ValueError(f"materialized-experts must be in [6, {K4_EXPERTS}]")
    tier0 = _prepared(3, materialized, 301, device)
    tier1 = _prepared(4, materialized, 401, device)
    mixed_tier0 = mixed_tier1 = rotations = None
    if args.variant in ("both", "mixed"):
        mixed_tier0 = _pad_exact_tier_storage(tier0, K3_EXPERTS)
        mixed_tier1 = _pad_exact_tier_storage(tier1, K4_EXPERTS)
        rotations = _padded_rotations(tier0, tier1, device)
    global_map, descriptor = build_ordered_maps(K3_EXPERTS, K4_EXPERTS, device=device)
    map0 = torch.cat(
        (
            torch.arange(materialized, dtype=torch.int32, device=device),
            torch.full(
                (TOTAL_EXPERTS - materialized,), -1, dtype=torch.int32, device=device
            ),
        )
    )
    map1 = torch.cat(
        (
            torch.full((K3_EXPERTS,), -1, dtype=torch.int32, device=device),
            torch.arange(materialized, dtype=torch.int32, device=device),
            torch.full(
                (K4_EXPERTS - materialized,), -1, dtype=torch.int32, device=device
            ),
        )
    )
    print(
        f"device={props.name} H={HIDDEN} I={INTERMEDIATE} topk={TOPK} "
        f"experts={K3_EXPERTS}xK3+{K4_EXPERTS}xK4"
    )
    for m in args.m:
        ids = _routing(m, materialized, device)
        weights = torch.softmax(
            torch.randn((m, TOPK), dtype=torch.float32, device=device), dim=-1
        )
        x = (torch.randn((m, HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
        state0 = state1 = serial_output = serial_fn = None
        serial_med = serial_buffer_bytes = None
        if args.variant in ("both", "serial", "serial-overlap"):
            state0 = (tier0, *_serial_state(m, tier0, map0, props, device))
            state1 = (tier1, *_serial_state(m, tier1, map1, props, device))
            serial_output = torch.empty((m, HIDDEN), dtype=torch.float32, device=device)
            tier_streams = None
            if args.variant == "serial-overlap":
                tier_streams = (torch.cuda.Stream(), torch.cuda.Stream())

            def serial_fn(
                state0=state0,
                state1=state1,
                serial_output=serial_output,
                tier_streams=tier_streams,
                x=x,
                weights=weights,
                ids=ids,
            ):
                assert state0 is not None and state1 is not None
                assert serial_output is not None
                if tier_streams is None:
                    out0 = _run_serial(state0, x, weights, ids)
                    out1 = _run_serial(state1, x, weights, ids)
                else:
                    current = torch.cuda.current_stream()
                    tier_streams[0].wait_stream(current)
                    tier_streams[1].wait_stream(current)
                    with torch.cuda.stream(tier_streams[0]):
                        out0 = _run_serial(state0, x, weights, ids)
                    with torch.cuda.stream(tier_streams[1]):
                        out1 = _run_serial(state1, x, weights, ids)
                    current.wait_stream(tier_streams[0])
                    current.wait_stream(tier_streams[1])
                torch.add(out0, out1, out=serial_output)

            serial_fn()
            serial_graph = _capture(serial_fn, args.warmup)
            serial_med = statistics.median(_times(serial_graph, args.replays))
            serial_buffer_bytes = _bytes(vars(state0[1]).values()) + _bytes(
                vars(state1[1]).values()
            )

        launch = mixed_buffers = mixed_fn = None
        mixed_med = mixed_buffer_bytes = None
        if args.variant in ("both", "mixed"):
            route_slots = max_packed_route_slots(m * TOPK, 8, TOTAL_EXPERTS)
            launch = compile_mixed_trellis(
                size_m=m,
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                tier0_num_experts=K3_EXPERTS,
                tier1_num_experts=K4_EXPERTS,
                top_k=TOPK,
                max_m_blocks=(route_slots + 7) // 8,
                sms=int(props.multi_processor_count),
                max_shared_mem=int(props.shared_memory_per_block_optin),
                force_tile_config=TILE_CONFIG,
            )
            mixed_buffers = make_mixed_trellis_buffers(
                launch, device=device, sms=int(props.multi_processor_count)
            )
            mixed_binding = bind_mixed_trellis(
                mixed_tier0,
                mixed_tier1,
                global_map,
                descriptor,
                rotations,
                launch,
            )
            if args.separate_fc2:
                mixed_buffers.fc2 = torch.empty_like(mixed_buffers.rotation_gate)

            def mixed_fn(
                mixed_tier0=mixed_tier0,
                mixed_tier1=mixed_tier1,
                mixed_binding=mixed_binding,
                mixed_buffers=mixed_buffers,
                x=x,
                weights=weights,
                ids=ids,
            ):
                assert mixed_tier0 is not None and mixed_tier1 is not None
                assert mixed_binding is not None
                assert mixed_buffers is not None
                run_bound_mixed_trellis(
                    x,
                    weights,
                    ids,
                    mixed_binding,
                    mixed_buffers,
                )

            mixed_fn()
            mixed_graph = _capture(mixed_fn, args.warmup)
            mixed_med = statistics.median(_times(mixed_graph, args.replays))
            mixed_buffer_bytes = _bytes(vars(mixed_buffers).values())

        torch.cuda.synchronize()
        if args.variant == "both":
            assert serial_output is not None and mixed_buffers is not None
            assert serial_med is not None and mixed_med is not None
            assert serial_buffer_bytes is not None and mixed_buffer_bytes is not None
            assert launch is not None
            rel = (
                serial_output - mixed_buffers.output[:m]
            ).norm() / serial_output.norm().clamp_min(1.0e-12)
            cosine = torch.nn.functional.cosine_similarity(
                serial_output.flatten(), mixed_buffers.output[:m].flatten(), dim=0
            )
            if float(rel) > 4.0e-3 or float(cosine) < 0.99999:
                raise RuntimeError(
                    f"correctness failed: rel={float(rel)} cos={float(cosine)}"
                )
            print(
                f"m={m:4d} serial={serial_med:9.2f}us mixed={mixed_med:9.2f}us "
                f"speedup={serial_med / mixed_med:6.3f}x rel={float(rel):.3e} "
                f"buffers={mixed_buffer_bytes / 2**20:.1f}/{serial_buffer_bytes / 2**20:.1f}MiB"
            )
        elif args.variant in ("serial", "serial-overlap"):
            assert serial_med is not None and serial_buffer_bytes is not None
            print(
                f"m={m:4d} variant={args.variant} time={serial_med:9.2f}us "
                f"buffers={serial_buffer_bytes / 2**20:.1f}MiB"
            )
        else:
            assert mixed_med is not None and mixed_buffer_bytes is not None
            assert launch is not None
            print(
                f"m={m:4d} variant=mixed time={mixed_med:9.2f}us "
                f"buffers={mixed_buffer_bytes / 2**20:.1f}MiB"
            )


if __name__ == "__main__":
    main()
