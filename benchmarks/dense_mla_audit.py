"""Frozen-core controls for benchmark_dense_mla.py --engine-aot (CPU importable)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = "b12x/attention/dense_mla/_prefill_fp8.py"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-aot", action="store_true")
    parser.add_argument("--mode", choices=("extend",), default="extend")
    parser.add_argument("--dtype", choices=("fp8",), default="fp8")
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--reference-core", required=True, help="PATH@git-blob-SHA")
    parser.add_argument("--rows", default="2752,2688,2673,1,63,65")
    parser.add_argument("--prior-contexts", default="0,8192,32768,65536")
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=1048576)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--source-commit", help="host revision when worktree Git metadata is not mounted")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.rows = [int(x) for x in args.rows.split(",")]
        args.prior_contexts = [int(x) for x in args.prior_contexts.split(",")]
        if (min(args.rows) <= 0 or min(args.prior_contexts) < 0
                or max(args.rows) > args.capacity
                or max(args.rows) + max(args.prior_contexts) > args.max_context
                or min(args.warmup, args.samples, args.iterations) <= 0):
            raise ValueError("invalid capacity, context, rows or repetition count")
    except ValueError as error:
        parser.error(str(error))
    return args


def source_commit(override=None):
    if override is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", override):
            raise ValueError("--source-commit requires a full Git revision")
        return override
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def verified_core(spec):
    path, separator, expected = spec.rpartition("@")
    if not separator or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--reference-core requires PATH@40-character-git-blob-SHA")
    source = Path(path).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if actual != expected:
        raise ValueError("frozen reference-core git blob mismatch")
    return Path(path).resolve(), source, actual


def checkpoint_geometry(path):
    config = json.loads(Path(path).read_text())
    text = config.get("text_config", config)
    expected = dict(hidden_size=7168, num_attention_heads=96, q_lora_rank=1536,
                    kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)
    if any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint does not have the supported Kimi MLA geometry")
    # Page size is an engine storage choice, not a checkpoint parameter.
    return {**expected, "page_rows": 64, "absorbed_qk_dim": 576, "absorbed_v_dim": 512,
            "config_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def byte_equal(a, b):
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def source_hashes():
    paths = sorted((ROOT / "b12x/attention/dense_mla").glob("*.py"))
    paths += sorted((ROOT / "b12x/integration/cuteafd").glob("kimi_mla*.py"))
    paths += [ROOT / "b12x/integration/cuteafd/_common.py", Path(__file__).resolve(),
              ROOT / "benchmarks/benchmark_dense_mla.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def engine_program(args, core=None):
    from b12x.integration.cuteafd import _common, kimi_mla, kimi_mla_prefill_fp8
    path = ROOT / CORE
    source = path.read_bytes() if core is None else core[1]
    identity = hashlib.sha256(source).hexdigest()
    module_name = "b12x.attention.dense_mla._duel_" + identity
    spec = importlib.util.spec_from_file_location(module_name, path if core is None else core[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(source, str(spec.origin), "exec"), module.__dict__)
    saved_core = kimi_mla_prefill_fp8.DenseMlaFp8PrefillKernel
    saved_compile = kimi_mla.compile_program
    dependencies = tuple(sorted(source_hashes().items()))

    def compile_keyed(wrapper, **kwargs):
        kwargs["name"] = "kimi_mla_duel_" + ("candidate" if core is None else "reference")
        kwargs["key"] = (identity, dependencies, kwargs["key"])
        return _common.compile_program(wrapper, **kwargs)

    try:
        kimi_mla_prefill_fp8.DenseMlaFp8PrefillKernel = module.DenseMlaFp8PrefillKernel
        kimi_mla.compile_program = compile_keyed
        program = kimi_mla.compile_kimi_dense_mla_aot(kimi_mla.KimiMlaGeometry(kv_cache="fp8"),
            max_rows=args.capacity, max_context=args.max_context, mode="extend")
    finally:
        kimi_mla_prefill_fp8.DenseMlaFp8PrefillKernel = saved_core
        kimi_mla.compile_program = saved_compile
    return program, identity


def main():
    args = arguments()
    geometry = checkpoint_geometry(args.checkpoint_config)
    commit = source_commit(args.source_commit)
    core = verified_core(args.reference_core)
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        raise RuntimeError("SM120/SM121 CUDA required")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference, ref_hash = engine_program(args, core)
    candidate, cand_hash = engine_program(args)
    programs = dict(current=reference, current_control=reference, candidate=candidate)
    props = torch.cuda.get_device_properties(0)
    report = dict(command=shlex.join(sys.argv),
        commit=commit, commit_supplied_by_host=args.source_commit is not None,
        worktree=str(ROOT), source_sha256=source_hashes(),
        reference_blob=core[2], reference_sha256=ref_hash, candidate_sha256=cand_hash,
        geometry=geometry, export=f"kimik3_dense_mla_extend_h96_fp8_m{args.capacity}",
        program_geometry=dict(reference.geometry), seed=args.seed,
        device=props.name, device_uuid=str(props.uuid), sm_count=props.multi_processor_count,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        warmup=args.warmup, iterations=args.iterations, rounds=args.samples, results=[])
    for rows in args.rows:
        for prior in args.prior_contexts:
            result = audit_case(args, programs, rows, prior, pack_mla_kv_cache_reference,
                                unpack_mla_kv_cache_reference)
            report["results"].append(result)
            args.json_output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)


def audit_case(args, programs, rows, prior, pack, unpack):
    length = prior + rows
    pages = (length + 63) // 64 + 1
    table = torch.randperm(pages, device="cuda").int()[None]
    logical = torch.arange(pages * 64, device="cuda")
    slots = table[0, logical // 64].long() * 64 + logical % 64
    records = pack(torch.randn(pages * 64, 512, device="cuda") * .1,
                   (torch.randn(pages * 64, 64, device="cuda") * .1).bfloat16()).reshape(-1, 656)
    cache = torch.empty_like(records)
    cache[slots] = records
    cache = cache.reshape(pages, 64, 656)
    query = torch.full((args.capacity, 96, 576), float("nan"), device="cuda", dtype=torch.bfloat16)
    query[:rows] = (torch.randn(rows, 96, 576, device="cuda") * .1).bfloat16()
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    arms = {}
    for label, program in programs.items():
        out = torch.full((args.capacity, 96, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
        lse = torch.full((args.capacity, 96), float("nan"), device="cuda")

        def launch(p=program, o=out, l=lse):
            p.launch(query, cache, table, lengths, cu, o, l, None,
                     scalars=(rows, 1, table.shape[1], 1))

        for _ in range(args.warmup):
            launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        torch.cuda.synchronize()
        arms[label] = dict(graph=graph, out=out, lse=lse, raw_ms=[])
    expected = [arms["current"][k][:rows].clone() for k in ("out", "lse")]
    # Independent sampled-row oracle avoids a full H*Q*K score allocation.
    kv = unpack(records[:length]).squeeze(1).float()
    selected = sorted(set((0, rows - 1)))
    scores = torch.einsum("rhd,kd->hrk", query[selected].float(), kv) * (192 ** -.5)
    mask = torch.arange(length, device="cuda")[None, :] > (prior + torch.tensor(selected, device="cuda"))[:, None]
    scores.masked_fill_(mask[None], -float("inf"))
    oracle = torch.einsum("hrk,kd->rhd", scores.softmax(-1), kv[:, :512])
    relative_l2 = float((expected[0][selected].float() - oracle).norm() / oracle.norm())
    torch.testing.assert_close(expected[1][selected], scores.logsumexp(-1).T, atol=2e-3, rtol=2e-4)
    if relative_l2 >= .005:
        raise AssertionError("frozen core failed independent numerical oracle")
    for poison in (False, True):
        if poison:
            # Includes the partial final page and one entirely invalid page.
            cache.reshape(-1, 656)[slots[length:]] = 255
        for label, arm in arms.items():
            arm["out"].fill_(float("nan"))
            arm["lse"].fill_(float("nan"))
            arm["graph"].replay()
            torch.cuda.synchronize()
            for key, oracle_tensor in zip(("out", "lse"), expected):
                live = arm[key][:rows]
                if not torch.isfinite(live).all() or not byte_equal(live, oracle_tensor):
                    raise AssertionError(f"{label}: {key} mismatch, poison={poison}, rows={rows}, prior={prior}")
                if not torch.isnan(arm[key][rows:]).all():
                    raise AssertionError(f"{label}: wrote past live {key}")
    allocated = torch.cuda.memory_allocated()
    for round_index in range(args.samples):
        order = list(arms)
        order = order[round_index % 3:] + order[:round_index % 3]
        for label in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.iterations):
                arms[label]["graph"].replay()
            end.record()
            end.synchronize()
            arms[label]["raw_ms"].append(start.elapsed_time(end) / args.iterations)
    if torch.cuda.memory_allocated() != allocated:
        raise AssertionError("replay allocation changed")
    medians = {label: statistics.median(a["raw_ms"]) for label, a in arms.items()}
    pairs = rows * prior + rows * (rows + 1) // 2
    return dict(rows=rows, prior_context=prior, cache_tokens=length,
        median_ms=medians, raw_ms={k: a["raw_ms"] for k, a in arms.items()},
        control_spread=abs(medians["current"] / medians["current_control"] - 1),
        reference_over_candidate=medians["current"] / medians["candidate"],
        us_per_row=medians["current"] * 1000 / rows,
        ns_per_row_context_pair=medians["current"] * 1e6 / pairs,
        qk_pv_flops=2 * 96 * (576 + 512) * pairs,
        unique_kv_bytes=length * 656, oracle_relative_l2=relative_l2,
        output_lse_byte_exact=True, poisoned_tail=True, allocation_stable=True)
