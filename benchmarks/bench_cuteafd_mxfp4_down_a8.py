#!/usr/bin/env python3
"""Bounded paired MiMo Pro TP6 stream diagnostic, before any tile tuning.

Checks both real padded widths, exact BF16 intermediate quantization, and
checkpoint-format weight oracles before one CUDA profile per arm and three
interleaved graph samples. This component gate cannot qualify model quality:
incremental model KL <= .005, tool/agentic checks and warmed prefill improvement
are required separately before any default or package admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def packed_weights(g, real, gen):
    h, i, e = g.hidden, g.slice, g.experts

    def codes(shape):
        return torch.randint(0, 256, shape, generator=gen, device="cuda", dtype=torch.int32).byte()

    def scales(shape):
        return torch.randint(118, 124, shape, generator=gen, device="cuda", dtype=torch.int32).byte()

    w1, s1 = codes((e, i, h // 2)), scales((e, i, h // 32))
    w3, s3 = codes((e, i, h // 2)), scales((e, i, h // 32))
    w2, s2 = codes((e, h, i // 2)), scales((e, h, i // 32))
    for w in (w1, s1, w3, s3):
        w[:, real:] = 0
    w2[:, :, real // 2:] = 0
    s2[:, :, real // 32:] = 0
    return w1, s1, w3, s3, w2, s2


def wire_rows(x):
    from b12x._lib.intrinsics import pow2_ceil_ue8m0_torch, _ue8m0_output_scale_torch

    rows, h = x.shape
    block = x.float().reshape(rows, h // 32, 32)
    # Match the existing engine's input-wire floor. The new down quantizer has
    # no floor and is tested independently below, after exact BF16 rounding.
    scale, byte = pow2_ceil_ue8m0_torch(block.abs().amax(-1).clamp_min(1e-4) / 448.0)
    q = (block * _ue8m0_output_scale_torch(byte)[..., None]).to(torch.float8_e4m3fn)
    wire = torch.cat((q.reshape(rows, h).view(torch.uint8), byte), 1).contiguous()
    return wire, (q.float() * scale[..., None]).reshape(rows, h).bfloat16()


def dequant(w, scale):
    table = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6), device=w.device)
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2).long()
    return (table[codes] * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)).bfloat16()


def oracle(x, ids, weights, packed, a8):
    from b12x._lib.intrinsics import quant_dequant_mxfp8_torch

    out = torch.zeros_like(x, dtype=torch.float32)
    w1, s1, w3, s3, w2, s2 = packed
    for e in ids.unique().tolist():
        rows, slots = torch.where(ids == e)
        gate = x[rows] @ dequant(w1[e], s1[e]).T
        up = x[rows] @ dequant(w3[e], s3[e]).T
        act = torch.nn.functional.silu(gate) * up
        if a8:
            act = quant_dequant_mxfp8_torch(act).bfloat16()
        y = act @ dequant(w2[e], s2[e]).T
        out.index_add_(0, rows, y.float() * weights[rows, slots, None])
    return out.bfloat16()


def live_intermediates(g, scratch, rows, capacity, a8):
    """Return each real act row sorted by (expert,input row), despite routing atomics."""
    from b12x.integration.cuteafd._fp8_moe_kernels import META_HEAD, meta_words
    from b12x.integration.cuteafd._fp8_moe_stream import stream_max_tiles
    from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp8_down_row_bytes
    from b12x.integration.cuteafd.fp8_moe import _align

    meta_bytes = _align(4 * meta_words(g.experts, stream_max_tiles(g.experts, capacity * g.top_k)))
    pairs_bytes = _align(4 * rows * g.top_k)
    row_map = scratch[meta_bytes:meta_bytes + 4 * rows * g.top_k].view(torch.int32).cpu()
    meta = scratch[:meta_bytes].view(torch.int32).cpu()
    row_width = mxfp8_down_row_bytes(g.slice) if a8 else 2 * g.slice
    act_start = meta_bytes + 2 * pairs_bytes
    act = scratch[act_start:act_start + int(meta[1]) * 128 * row_width].view(-1, row_width)
    indices, keys = [], []
    for tile, code in enumerate(meta[META_HEAD + 3 * g.experts:META_HEAD + 3 * g.experts + int(meta[1])]):
        e, offset = int(code) & 65535, (int(code) >> 16) * 128
        live = min(int(meta[META_HEAD + e]) - offset, 128)
        first = int(meta[META_HEAD + g.experts + e]) + offset
        indices.extend(range(tile * 128, tile * 128 + live))
        keys.extend(e * capacity + int(r) for r in row_map[first:first + live])
    order = torch.tensor(keys).argsort()
    take = torch.tensor(indices)[order].to(scratch.device)
    selected = act[take].contiguous()
    if a8:
        q = selected[:, :g.slice].contiguous().view(torch.float8_e4m3fn).float()
        scale = selected[:, g.slice:g.slice + g.slice // 32].float()
        selected = (q * torch.exp2(scale - 127).repeat_interleave(32, -1)).bfloat16()
    else:
        selected = selected.view(torch.bfloat16)
    return torch.tensor(keys)[order], selected


def require_exact_quantization(g, baseline, candidate, rows, capacity, real):
    from b12x._lib.intrinsics import quant_dequant_mxfp8_torch

    base_keys, act = live_intermediates(g, baseline, rows, capacity, False)
    keys, actual = live_intermediates(g, candidate, rows, capacity, True)
    if not torch.equal(keys, base_keys):
        raise RuntimeError("baseline/candidate routing maps differ")
    expected = quant_dequant_mxfp8_torch(act).bfloat16()
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        mismatch = int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())
        raise RuntimeError(f"A8 differs from the exact BF16-intermediate quantizer in {mismatch} bytes")
    if torch.count_nonzero(act[:, real:]) or torch.count_nonzero(actual[:, real:]):
        raise RuntimeError("padded K32 blocks contain live values")
    return {"live_pairs": len(keys), "bf16_rounding_then_quantization": "byte_exact", "padding": "zero"}


def require_numerical_oracle(actual, expected, label, *, max_relative_rms=.01, min_cosine=.9999):
    a, b = actual.float(), expected.float()
    if not torch.isfinite(a).all() or torch.count_nonzero(a) == 0:
        raise RuntimeError(f"{label}: output is nonfinite or all zero")
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    denominator = b.square().sum()
    relative_rms = float(((a - b).square().sum() / denominator).sqrt())
    norm_ratio = float((a.square().sum() / denominator).sqrt())
    if worst < min_cosine or relative_rms > max_relative_rms or abs(norm_ratio - 1) > max_relative_rms:
        raise RuntimeError(f"{label}: worst_row={worst}, relative_rms={relative_rms}, norm_ratio={norm_ratio}")
    return {"worst_row": worst, "relative_rms": relative_rms, "norm_ratio": norm_ratio}


def require_down_oracle(g, scratch, out, ids, weights, packed, rows, capacity, a8):
    """Isolate down math from the existing gate/up's numerical floor."""
    keys, act = live_intermediates(g, scratch, rows, capacity, a8)
    keys = keys.to(out.device)
    expert_ids, source_rows = keys // capacity, keys % capacity
    expected = torch.zeros_like(out[:rows], dtype=torch.float32)
    w2, s2 = packed[-2:]
    for expert in expert_ids.unique().tolist():
        selected = expert_ids == expert
        r = source_rows[selected]
        slots = torch.where(ids[r] == expert)[1]
        y = act[selected] @ dequant(w2[expert], s2[expert]).T
        expected.index_add_(0, r, y.float() * weights[r, slots, None])
    return require_numerical_oracle(out[:rows], expected.bfloat16(), "down-only checkpoint oracle",
                                    max_relative_rms=.005, min_cosine=.99999)


def source_identity(revision=None):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted((root / "b12x").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    if revision is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"revision": revision,
            "source_sha256": digest.hexdigest(), "worktree": str(root), "command": sys.argv}


def gpu_snapshot():
    fields = "uuid,name,pstate,clocks.sm,clocks.mem,power.draw,power.limit,clocks_event_reasons.active"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv"],
                            capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--real", type=int, choices=(320, 352), default=352)
    parser.add_argument("--launches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", help="required when running an archived source outside git")
    parser.add_argument("--measurement-permit", type=Path, help="optional externally coordinated profile/timing barrier")
    args = parser.parse_args()
    if args.rows <= 640 or args.rows > 4096 or not 1 <= args.launches <= 16:
        parser.error("bounded stream diagnostic needs 641..4096 rows and 1..16 launches")
    if torch.cuda.get_device_capability() != (12, 1):
        parser.error("this gate is qualified for one SM121 Spark")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    args.output.mkdir(parents=True, exist_ok=True)
    initial_gpu = gpu_snapshot()
    g = GEOMETRIES["mimop"].with_tp(6)
    gen = torch.Generator(device="cuda").manual_seed(57)
    packed = packed_weights(g, args.real, gen)
    source, exact = wire_rows(torch.randn(args.rows, g.hidden, device="cuda", generator=gen).bfloat16())
    ids = torch.rand(args.rows, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int()
    weights = torch.rand(args.rows, g.top_k, device="cuda", generator=gen)
    weights /= weights.sum(-1, keepdim=True)
    programs, buffers, checks, graphs = {}, {}, {}, {}
    print("phase=compile_and_correctness", flush=True)
    for arm, a8 in (("bf16", False), ("a8", True)):
        program = compile_fp8_moe_aot(g, route="stream", max_rows=args.rows, mxfp4_down_a8=a8)
        out = torch.full((args.rows, g.hidden), float("nan"), device="cuda", dtype=torch.bfloat16)
        scratch = torch.full((fp8_moe_scratch_bytes(g, "stream", args.rows, mxfp4_down_a8=a8),),
                             255, device="cuda", dtype=torch.uint8)
        programs[arm], buffers[arm] = program, (source, ids, weights, *packed, out, scratch)
        program.launch(*buffers[arm], scalars=(args.rows,))
        torch.cuda.synchronize()
        ref = oracle(exact, ids, weights, packed, a8)
        checks[arm] = {**require_numerical_oracle(out, ref, f"{arm} full MoE oracle"),
                       "down_only": require_down_oracle(g, scratch, out, ids, weights, packed,
                                                         args.rows, args.rows, a8),
                       "scratch_bytes": scratch.numel(), "abi": program.abi}
    checks["activation"] = require_exact_quantization(g, buffers["bf16"][-1], buffers["a8"][-1],
                                                       args.rows, args.rows, args.real)
    print("phase=profile_ready", flush=True)
    if args.measurement_permit:
        deadline = time.monotonic() + 180
        while not args.measurement_permit.exists():
            if time.monotonic() >= deadline:
                raise SystemExit("Measurement barrier timed out; no profile or timing collected")
            time.sleep(.25)
    # Profile before any tuning or timing interpretation. Its timing is not an
    # A/B measurement; preserve the kernel breakdown and trace separately.
    profiles = {}
    with kernel_resolution_guard("prepared paired down-A8 diagnostic"):
        for arm in programs:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                programs[arm].launch(*buffers[arm], scalars=(args.rows,))
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.output / f"{arm}-profile.json"))
            profiles[arm] = [{"name": e.name, "duration_us": e.time_range.elapsed_us()}
                             for e in profile.events() if e.device_type.name == "CUDA"]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.launches):
                    programs[arm].launch(*buffers[arm], scalars=(args.rows,))
            graphs[arm] = graph
    flush = torch.zeros(256 << 20, device="cuda", dtype=torch.uint8)
    raw = []
    addresses = {arm: tuple(t.data_ptr() for t in tensors) for arm, tensors in buffers.items()}
    allocated = torch.cuda.memory_allocated()
    print("phase=timing_ready", flush=True)
    with kernel_resolution_guard("paired graph replay without resolution"):
        for order in (("bf16", "a8"), ("a8", "bf16"), ("bf16", "a8")):
            for arm in order:
                flush.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                allocation_bytes = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                raw.append({"arm": arm, "us": start.elapsed_time(end) * 1000 / args.launches})
                if addresses[arm] != tuple(t.data_ptr() for t in buffers[arm]):
                    raise RuntimeError("graph pointer changed")
                if torch.cuda.memory_stats()["allocated_bytes.all.allocated"] != allocation_bytes \
                        or torch.cuda.memory_allocated() > allocated:
                    raise RuntimeError("graph replay allocated storage")
    medians = {arm: statistics.median(x["us"] for x in raw if x["arm"] == arm) for arm in programs}
    record = {**source_identity(args.source_revision), "gpu_name": torch.cuda.get_device_name(), "capability": [12, 1],
              "rows": args.rows, "real": args.real, "profiles": profiles, "correctness": checks,
              "samples": raw, "median_us": medians, "ratio_a8_over_bf16": medians["a8"] / medians["bf16"],
              "gpu_before": initial_gpu, "gpu_after": gpu_snapshot(),
              "full_model_incremental_kl": {"maximum": .005, "measured": None,
                                            "status": "required before package/default admission"},
              "default_admissible": False}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if medians["a8"] >= medians["bf16"]:
        raise SystemExit("A8 component did not beat BF16; stop before full-model admission")


if __name__ == "__main__":
    main()
