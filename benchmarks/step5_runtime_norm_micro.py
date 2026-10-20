"""Same-session frozen/current AOT microbench for the runtime-epsilon affine norm.

Oracle: the frozen serving source of ``b12x/integration/cuteafd/pointwise.py``
(loaded under a private module name, so its compiled object is separate from the
candidate's).  Candidate: ``--repo``; ``--candidate-classes`` optionally names
``RuntimeAffineNorm`` subclasses in that module (development variants) instead of
the module default.  All arms are compiled and measured in one process so a
single queued session covers the whole comparison.

Run only under the hardware queue.  The b12x memory cache keys explicit-spec
compiles by kernel name and geometry only, so an oracle and a candidate with the
same geometry would collide and the second arm would reuse the first arm's
cubin.  This harness disables both compile caches (the disk cache is keyed on
the module path, not its content) and asserts the compiled objects differ.

Every case first proves the candidate rewrites a NaN-filled output with bits
identical to the oracle's, then times both arms through CUDA graph replays in
alternating blocks.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ORACLE_SHA256 = "209a776a3604ec27592589a8a40efe83159656c93ef50949574e77d86715e64f"

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--oracle-repo", required=True)
p.add_argument("--output", required=True)
p.add_argument("--preflight", action="store_true")
p.add_argument("--candidate-classes", default="")
p.add_argument("--rows", default="1,16,256")
p.add_argument("--replays", type=int, default=24)
p.add_argument("--blocks", type=int, default=4)
a = p.parse_args()
repo, oracle = Path(a.repo).resolve(), Path(a.oracle_repo).resolve()
frozen_path = oracle / "b12x/integration/cuteafd/pointwise.py"

if a.preflight:
    assert (repo / "b12x/integration/cuteafd/pointwise.py").is_file()
    assert (repo / "benchmarks/common.py").is_file()
    assert frozen_path.is_file()
    sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert sha == ORACLE_SHA256, f"frozen pointwise module sha {sha} != {ORACLE_SHA256}"
    sys.exit(0)

# Kill explicit-spec cache collisions before any b12x import.
os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

sys.path.insert(0, str(repo))
import torch  # noqa: E402
from benchmarks.common import bench_cuda_graph, capture_cuda_graph  # noqa: E402
from b12x.integration.cuteafd import pointwise  # noqa: E402


def frozen(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_pointwise = frozen("b12x.integration.cuteafd.step5_frozen_pointwise", frozen_path)


def compile_candidate(class_name, **kw):
    if not class_name:
        return pointwise.compile_affine_norm_aot(**kw)
    original = pointwise.RuntimeAffineNorm
    pointwise.RuntimeAffineNorm = getattr(pointwise, class_name)
    try:
        return pointwise.compile_affine_norm_aot(**kw)
    finally:
        pointwise.RuntimeAffineNorm = original


def measure(label, oracle_program, candidate_program, inputs, scalars, output_index=-1):
    def launch(program, tensors):
        program.launch(*tensors, scalars=scalars)

    tensors = list(inputs)
    launch(oracle_program, tensors)
    torch.cuda.synchronize()
    expected = tensors[output_index].clone()
    tensors[output_index].fill_(float("nan"))
    launch(candidate_program, tensors)
    torch.cuda.synchronize()
    actual = tensors[output_index]
    bits_equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if not bits_equal:
        bad = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        delta = (actual.float() - expected.float()).abs()
        max_delta = torch.nan_to_num(delta, nan=float("inf")).max().item()
        raise AssertionError(f"{label}: frozen output changed, {bad} elements, max delta {max_delta}")
    assert not actual.isnan().any(), f"{label}: candidate left elements unwritten"

    graphs = {
        "oracle": capture_cuda_graph(lambda: launch(oracle_program, tensors), warmup=3),
        "candidate": capture_cuda_graph(lambda: launch(candidate_program, tensors), warmup=3),
    }
    samples = {"oracle": [], "candidate": []}
    order = ("oracle", "candidate")
    for block in range(a.blocks):
        seq = order if block % 2 == 0 else order[::-1]
        for arm in seq:
            samples[arm].extend(bench_cuda_graph(graphs[arm], replays=a.replays)["replay_us"])
    med = {arm: statistics.median(v) for arm, v in samples.items()}
    row = dict(case=label, oracle_us=med["oracle"], candidate_us=med["candidate"],
               speedup=med["oracle"] / med["candidate"], bitwise_equal=True,
               oracle_min_us=min(samples["oracle"]),
               candidate_min_us=min(samples["candidate"]))
    print(json.dumps(row), flush=True)
    return row


assert torch.cuda.is_available()
torch.manual_seed(274)
dummy = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
kwargs = dict(width=4096, zero_centered=True, eps=None)
oracle_program = old_pointwise.compile_affine_norm_aot(**kwargs)
candidates = [name for name in a.candidate_classes.split(",") if name] or [""]
programs = {name: compile_candidate(name, **kwargs) for name in candidates}
compiled_ids = [id(oracle_program.compiled)] + [id(program.compiled) for program in programs.values()]
assert len(set(compiled_ids)) == len(compiled_ids), (
    "two arms share one compiled object (cache collision)")
assert all(program.compiled is not oracle_program.compiled for program in programs.values())

results = []
for name, program in programs.items():
    label = (name or "default").removeprefix("_RuntimeAffineNorm").lower()
    for rows in (int(value) for value in a.rows.split(",")):
        x = torch.randn(rows, 4096, device="cuda")
        weight = torch.randn(4096, device="cuda") * 0.1
        positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        output = torch.empty_like(x, dtype=torch.bfloat16)
        results.append(measure(f"{label}-{rows}x4096", oracle_program, program,
                               [x, weight, weight, dummy, dummy, positions, output],
                               [rows, 1e-5]))

Path(a.output).write_text(json.dumps(dict(
    device=torch.cuda.get_device_name(),
    multi_processor_count=torch.cuda.get_device_properties(0).multi_processor_count,
    oracle=str(oracle), repo=str(repo), candidate_classes=candidates,
    oracle_sha256=ORACLE_SHA256,
    memory_cache=os.environ["B12X_COMPILE_MEMORY_CACHE"],
    disk_cache=os.environ["B12X_COMPILE_DISK_CACHE"], results=results), indent=2))
print(json.dumps(dict(output=str(a.output), cases=len(results))), flush=True)
