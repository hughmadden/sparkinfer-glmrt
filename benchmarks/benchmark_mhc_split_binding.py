#!/usr/bin/env python3
"""Measure prepared exact-M mHC post/pre bindings across projection splits.

Each row owns a declarative mHC request.  Preparation resolves the selected
projection and finalizer once; timed calls only consume the bound execution and
live tensor inputs.  The diagnostic intentionally varies split count while
retaining the production numerical recipe and its FP32 oracle.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

from b12x._lib.compile_plan import program_keys
from b12x.norm import mhc
from b12x.preparation import FrozenMapping, PreparedCall, PreparationSession
from b12x.preparation.types import require_prepared

_GEOMETRIES = {
    "g0": (16, 8, 8, 1, 1, 1),
    "g1": (16, 24, 64, 3, 1, 3),
    "g2": (32, 8, 128, 4, 2, 1),
    "g3": (32, 32, 256, 2, 2, 2),
}
_EPS = 1e-6


@dataclass(frozen=True)
class Case:
    hidden: int
    tokens: int
    geometry: str
    splits: int

    @property
    def capacity(self) -> int:
        return 4 * self.hidden // 256

    @property
    def name(self) -> str:
        return f"h{self.hidden}_m{self.tokens}_{self.geometry}_ks{self.splits}"


def _domain() -> list[Case]:
    cases = [
        Case(4096, m, geometry, splits)
        for geometry in _GEOMETRIES
        for m in ((1, 4, 128, 384) if geometry in ("g0", "g1") else (128, 384))
        for splits in (1, 2, 4, 8, 16, 32)
    ]
    cases.extend(Case(7168, 128, "g1", splits) for splits in (1, 4, 7, 16, 28))
    return cases


def _config(case: Case) -> mhc.MhcConfig:
    m, n, k, stages, mw, nw = _GEOMETRIES[case.geometry]
    return mhc.MhcConfig(
        backend="tf32_tma",
        projection_tile_m=m,
        projection_tile_n=n,
        projection_tile_k=k,
        projection_num_stages=stages,
        projection_num_m_warps=mw,
        projection_num_n_warps=nw,
        projection_k_splits=case.splits,
    )


class Buffers:
    def __init__(self, case: Case, device: torch.device, seed: int):
        self.case, self.device = case, device
        generator = torch.Generator(device="cpu").manual_seed(seed + case.hidden)

        def random(shape, dtype, scale):
            return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

        h, m = case.hidden, case.tokens
        self.fn = random((24, 4 * h), torch.float32, 0.01)
        self.scale = torch.full((3,), 0.1, device=device)
        self.bias = random((24,), torch.float32, 0.1)
        self.norm_weight = random((h,), torch.bfloat16, 0.1).add_(1)
        self.x = random((m, h), torch.bfloat16, 0.1)
        self.residual = random((m, 4, h), torch.bfloat16, 0.1)
        self.prev_post = torch.full((m, 4), 0.5, device=device)
        self.prev_comb = torch.eye(4, device=device).expand(m, 4, 4).contiguous()
        self.out, self.y = torch.empty_like(self.residual), torch.empty_like(self.x)
        self.post = torch.empty((m, 4), device=device)
        self.comb = torch.empty((m, 4, 4), device=device)

    def binding_args(self):
        return dict(
            tokens=self.case.tokens,
            expected_m=self.case.tokens,
            y=self.y,
            post=self.post,
            comb=self.comb,
            out=self.out,
        )

    def run(self, binding):
        return binding.post_pre(
            self.x,
            self.residual,
            self.prev_post,
            self.prev_comb,
            self.fn,
            self.scale,
            self.bias,
            rms_eps=_EPS,
            hc_eps=_EPS,
            sinkhorn_iters=20,
            norm_weight=self.norm_weight,
            norm_eps=_EPS,
        )

    def reset(self):
        self.out.zero_()
        self.y.zero_()
        self.post.zero_()
        self.comb.zero_()


def _oracle(buffers: Buffers) -> tuple[torch.Tensor, ...]:
    residual = (
        buffers.prev_post.unsqueeze(-1) * buffers.x.unsqueeze(1).float()
        + (buffers.prev_comb.unsqueeze(-1) * buffers.residual.unsqueeze(2).float()).sum(1)
    ).to(torch.bfloat16)
    flat = residual.flatten(1).float()
    mixes = (flat @ buffers.fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + _EPS)
    pre = torch.sigmoid(mixes[:, :4] * buffers.scale[0] + buffers.bias[:4]) + _EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * buffers.scale[1] + buffers.bias[4:8])
    comb = torch.softmax(mixes[:, 8:].view(-1, 4, 4) * buffers.scale[2] + buffers.bias[8:].view(4, 4), dim=-1) + _EPS
    comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    y_raw = (pre.unsqueeze(-1) * residual.float()).sum(1)
    y = (y_raw.to(torch.bfloat16).float() * torch.rsqrt(y_raw.square().mean(-1, keepdim=True) + _EPS) * buffers.norm_weight.float()).to(torch.bfloat16)
    return residual, post, comb, y


def _check(case: Case, buffers: Buffers) -> float:
    actual, expected = (buffers.out, buffers.post, buffers.comb, buffers.y), _oracle(buffers)
    scores = []
    for value, reference in zip(actual, expected, strict=True):
        score = torch.nn.functional.cosine_similarity(value.float().flatten(), reference.float().flatten(), dim=0).item()
        if score < 0.998 or not torch.isfinite(value).all():
            raise AssertionError(f"{case.name}: cosine {score}")
        scores.append(score)
    return min(scores)


def _run_case(case: Case, args, device: torch.device) -> dict[str, object]:
    buffers = Buffers(case, device, args.seed)
    declaration = mhc.plan(
        mhc.Caps(device=device, max_tokens=case.tokens, hidden_size=case.hidden, split_k=case.capacity),
        invocation=FrozenMapping({
            "operation": "post_pre",
            "has_norm_weight": True,
            "norm_weight_dtype": "bfloat16",
            "has_fn_bf16": False,
            "output_mode": "provided",
            "rms_eps": _EPS,
            "hc_eps": _EPS,
            "sinkhorn_iters": 20,
            "norm_eps": _EPS,
        }),
        override=_config(case),
    )

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=scratch, **buffers.binding_args())
        return PreparedCall(run=lambda: buffers.run(binding), reset=buffers.reset, owners=(binding,))

    with PreparationSession(device=device, autotune=False, cache_dir=args.cache_dir) as session:
        session.prepare((declaration.request(
            name=case.name,
            prepare_call=prepare_call,
        ),))
        plan = declaration
        (spec,) = require_prepared(plan, "norm.mhc").scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = mhc.bind(plan, scratch=scratch, **buffers.binding_args())
        buffers.reset()
        buffers.run(binding)
        torch.cuda.synchronize(device)
        cosine = _check(case, buffers)
        for _ in range(args.warmup):
            buffers.reset()
            buffers.run(binding)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.samples):
            buffers.reset()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            buffers.run(binding)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        with session.capture():
            graph = torch.cuda.CUDAGraph()
            buffers.reset()
            with torch.cuda.graph(graph):
                buffers.run(binding)
        buffers.reset()
        graph.replay()
        torch.cuda.synchronize(device)
        graph_cosine = _check(case, buffers)
        if graph_cosine < 0.998:
            raise AssertionError(f"{case.name}: graph cosine {graph_cosine}")
        prepared_programs = {
            name: [program.key for program in program_keys(launcher)]
            for name, launcher in require_prepared(plan, "norm.mhc").launchers.items()
        }
        return {
            "case": case.name,
            "hidden": case.hidden,
            "tokens": case.tokens,
            "geometry": _GEOMETRIES[case.geometry],
            "splits": case.splits,
            "selection": plan.selection.config.to_dict(),
            "abi_key": repr(plan.abi_key),
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "minimum_cosine": min(cosine, graph_cosine),
            "exact_m": case.tokens,
            "graph_replay_checked": True,
            "prepared_programs": prepared_programs,
        }


def _run_partial_groups(case: Case, args, device: torch.device) -> dict[str, object]:
    """Compare the same decode arithmetic with four versus all 25 partials per CTA."""
    from b12x.integration.cuteafd import GLM53_FLASH, exportable_compilation, glmf
    from b12x.integration.cuteafd.dsv4_mhc import compile_dsv4_mhc_post_pre_aot

    geometry = glmf.mhc_geometry(GLM53_FLASH)
    buffers = Buffers(case, device, args.seed)
    reference, timings = None, {}
    for groups in (4, 25, "split"):
        options = ({"wide_partials_per_cta": 25, "wide_min_rows": 64} if groups == "split"
                   else {"partials_per_cta": groups})
        with exportable_compilation():
            program = compile_dsv4_mhc_post_pre_aot(
                geometry, max_rows=4096, route="decode", **options,
            )
        if args.export_dir is not None:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            stem = f"mhc_partial_groups_{groups}"
            program.export_to_c(str(args.export_dir), stem, "cuteafd_" + stem)
        scratch = torch.empty(program.scratch_bytes(case.tokens)["scratch"], dtype=torch.uint8, device=device)

        def run():
            program.launch(buffers.x, buffers.residual, buffers.prev_post, buffers.prev_comb,
                           buffers.fn, buffers.scale, buffers.bias, buffers.norm_weight,
                           buffers.out, buffers.post, buffers.comb, buffers.y,
                           scratch, scalars=(case.tokens,))

        buffers.reset()
        run()
        torch.cuda.synchronize(device)
        actual = (buffers.out, buffers.post, buffers.comb, buffers.y)
        if reference is None:
            reference = tuple(value.clone() for value in actual)
        for value, expected in zip(actual, reference, strict=True):
            if not torch.equal(value.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
                raise AssertionError(f"M{case.tokens} groups{groups}: raw-bit mismatch")
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for value in actual:
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        for value, expected in zip(actual, reference, strict=True):
            if not torch.equal(value.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
                raise AssertionError(f"M{case.tokens} groups{groups}: poisoned graph raw-bit mismatch")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings[groups] = start.elapsed_time(end) * 1000.0 / args.replays
        for value in actual:
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        for value, expected in zip(actual, reference, strict=True):
            if not torch.equal(value.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
                raise AssertionError(f"M{case.tokens} groups{groups}: graph raw-bit mismatch")
    return {"tokens": case.tokens, "hidden": case.hidden, "old_us": timings[4],
            "new_us": timings[25], "new_over_old": timings[25] / timings[4],
            "split_us": timings["split"], "split_over_old": timings["split"] / timings[4],
            "bitwise": True, "graph_replays": args.replays}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--hidden")
    parser.add_argument("--tokens")
    parser.add_argument("--geometries")
    parser.add_argument("--splits")
    parser.add_argument("--partial-groups", action="store_true", help="isolated GLM Flash AOT 4-vs-25 partial fusion")
    parser.add_argument("--replays", type=int, default=64)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args(argv)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if args.partial_groups:
        if not 0 <= args.device < torch.cuda.device_count():
            raise ValueError("Invalid visible CUDA device")
        if args.hidden not in (None, "4096"):
            raise ValueError("Partial-group mode uses GLM Flash hidden4096")
    elif not 0 <= args.device < len(visible) or visible[args.device] not in ("6", "7"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly; this experiment uses physical GPUs6/7")
    if args.warmup < 1 or args.samples < 1 or args.replays < 1:
        raise ValueError("Warmup, samples and replays must be positive")
    cases = ([Case(4096, m, "partial_groups", 1) for m in (1, 8, 16, 64, 128, 4096)]
             if args.partial_groups else _domain())
    for option, field, convert in (("hidden", "hidden", int), ("tokens", "tokens", int), ("geometries", "geometry", str), ("splits", "splits", int)):
        value = getattr(args, option)
        if value is not None:
            selected = {convert(item) for item in value.split(",")}
            cases = [case for case in cases if getattr(case, field) in selected]
    if not cases:
        raise ValueError("No cases selected")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"status": "running", "pairs": []}
    try:
        with torch.inference_mode():
            for case in cases:
                row = (_run_partial_groups if args.partial_groups else _run_case)(case, args, device)
                report["pairs"].append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
