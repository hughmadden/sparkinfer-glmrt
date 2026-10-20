"""Same-session frozen/current AOT microbench for Step-5 CSA region selection.

Oracle: ``--oracle-repo`` must hold the frozen serving
``b12x/integration/cuteafd/region_selection.py`` from commit e8c69f7e (sha256
pinned below), loaded under a private module name so its compiled object is
separate from the candidate's.  Candidate: ``--repo``; ``--candidate-classes``
adds development classes in that module beside the default ``_Selection``.
All arms compile and measure in one process.

The group under test is the whole fused region-selection launch set:
prepare -> shared radix top-k -> expand.  The b12x compile caches are disabled
(the memory cache keys explicit-spec compiles by kernel name and geometry only,
so the oracle and a candidate with the same geometry would collide) and the
compiled objects are asserted distinct.

Every case first proves bitwise equality against the frozen arm with the masked
capacity tail of ``scores`` poisoned (+777/NaN/+inf) and the candidate's
``normalized`` scratch prefilled with +777: a candidate that normalized or read
past ``lengths`` would change the selection.  Timing is interleaved CUDA-graph
replay, alternating blocks.

Rows whose visible length exceeds the shared top-k's canonical-order limit
(65,536 entries: it emits ascending logical ids only while
``(length + 31) >> 5 <= 2 * smem_candidate_capacity``) keep atomic arrival
order, so the frozen arm's own ``ids``/``values``/``selected`` bytes differ run
to run on identical inputs.  Those rows are gated on bitwise top-k inputs
(``lengths`` + the normalized visible prefix), the exact id/value/selected
multisets, and the untouched masked tail, and the harness records the frozen
run-to-run order variance it observed.  Every shorter row stays bitwise.

Run only under the hardware queue (scripts/hw/hwlock.py).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ORACLE_SHA256 = "bd6f505b8e8122f800771f3943550364244eebb79aae06f4147c8f08b29c3b74"
TOPK_CLOSURE = {
    "b12x/attention/dsa_indexer/tiled_topk.py": "33b5fd9fb141428fedbe49929949fe68020c3ec49cd37622a0d8927c83f2b624",
    "b12x/_lib/compile_plan.py": "9a7066143209765286ed5cb44394c938a8998a3e0ba7fa986d5bec81b4c5722b",
    "b12x/_lib/compiler.py": "b6667e1c6c27289a91a3d9f4e2c89a65836b7b65b5b369629d058bc2e9bcf7cc",
    "b12x/_lib/intrinsics.py": "4d0e1e4aaec0f1afd8553556ef33270584500460c788f9eaf3668b0b5396c728",
    "b12x/_lib/program_cache.py": "63720746570d28f74a1a45d38e20727e4497265c8151f001f0046457ab7e9b8b",
    "b12x/_lib/runtime_control.py": "0ed3446289baf12ee5a92f805f5559f38b20bb0e5fe27bb696b7bb2c8c377ea0",
    "b12x/_lib/runtime_patches.py": "53ce88b19bbb049e4d328e18f3a181e96595dfffc23279b87819bab70f852a9a",
    "b12x/_lib/utils.py": "d5af9b10906e04b87e0e99c0192e09282e2582de1bf33586c0ca92bb8f32478a",
}
POISON = 777.0

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--oracle-repo", required=True)
p.add_argument("--output", required=True)
p.add_argument("--preflight", action="store_true")
p.add_argument("--candidate-classes", default="")
p.add_argument("--cases", default="")
p.add_argument("--replays", type=int, default=16)
p.add_argument("--blocks", type=int, default=4)
a = p.parse_args()
repo, oracle = Path(a.repo).resolve(), Path(a.oracle_repo).resolve()
frozen_path = oracle / "b12x/integration/cuteafd/region_selection.py"
# The module default is always measured; dev classes are extra grid variants.
candidates = [""] + [name for name in a.candidate_classes.split(",") if name]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if a.preflight:
    assert (repo / "b12x/integration/cuteafd/region_selection.py").is_file(), "candidate module missing"
    assert (repo / "benchmarks/common.py").is_file(), "benchmarks/common.py missing"
    assert frozen_path.is_file(), f"frozen region_selection missing under {oracle}"
    assert sha256(frozen_path) == ORACLE_SHA256, "frozen region_selection sha changed"
    for rel, want in TOPK_CLOSURE.items():
        got = sha256(repo / rel)
        assert got == want, f"transitive top-k source changed: {rel} {got} != {want}"
    text = (repo / "b12x/integration/cuteafd/region_selection.py").read_text()
    for name in candidates:
        if not name:
            continue
        assert f"class {name}:" in text or f"class {name}(" in text, (
            f"candidate class {name} missing")
    print("preflight ok: oracle pin, top-k closure, candidate module", flush=True)
    sys.exit(0)

# Kill explicit-spec cache collisions before any b12x import.
os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
os.environ["B12X_COMPILE_SPEC_MEMO"] = "0"

sys.path.insert(0, str(repo))
import torch  # noqa: E402
from benchmarks.common import bench_cuda_graph, capture_cuda_graph  # noqa: E402
from b12x.integration.cuteafd import region_selection  # noqa: E402
from b12x._lib.runtime_control import kernel_resolution_guard  # noqa: E402


def frozen(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


assert sha256(frozen_path) == ORACLE_SHA256
for rel, want in TOPK_CLOSURE.items():
    assert sha256(repo / rel) == want, f"transitive top-k source changed: {rel}"
frozen_module = frozen("b12x.integration.cuteafd.step5_frozen_region_selection", frozen_path)
candidate_sha = sha256(repo / "b12x/integration/cuteafd/region_selection.py")


def compile_candidate(module, class_name, *, groups, capacity):
    original = module._Selection
    if class_name:
        module._Selection = getattr(module, class_name)
    try:
        return module.compile_region_selection_aot(groups=groups, capacity=capacity)
    finally:
        module._Selection = original


def visible_lengths(positions, requests, capacity):
    vis = torch.clamp(positions // 8, min=0, max=capacity)
    return torch.where(requests < 0, torch.zeros_like(vis), vis)


def make_scores(rows, groups, capacity, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(rows, groups, capacity, device="cuda", generator=generator)
    n = min(capacity, 4096)
    if n >= 3:
        # Exact duplicate scores, signed zeros and equal negatives in the
        # visible part: the deterministic lower-logical-id tie order must not
        # move when the masked tail stops being normalized.
        scores[:, 0, :n:3] = 1.0
        scores[:, 0, 1:n:3] = -0.0
    if groups > 1 and n >= 5:
        scores[:, 1, :n:5] = 0.0
        scores[:, 1, 2:n:5] = -1.0
    if n >= 7:
        scores[:, -1, :n:7] = -3.5
    return scores


def poisoned_suffix(scores, vis, value):
    poisoned = scores.clone()
    for row, length in enumerate(int(item) for item in vis):
        poisoned[row, :, length:] = value
    return poisoned


def make_case(label, groups, capacity, rows, positions, requests):
    return dict(label=label, groups=groups, capacity=capacity, rows=rows,
                positions=torch.tensor(positions, device="cuda", dtype=torch.int64),
                requests=torch.tensor(requests, device="cuda", dtype=torch.int32))


def mixed_positions(rows):
    edge = [0, 7, 8, 4088, 4096, 8192, 8232, 1048576]
    values = [edge[index % len(edge)] if index < len(edge) else (index * 37) % 8233
              for index in range(rows)]
    requests = [0] * rows
    if rows > 1:
        requests[-1] = -1
    if rows > 8:
        requests[7] = -1
        values[7] = 1048576
    return values, requests


def cases():
    sliced = [8232 - (64 - 1 - index) for index in range(64)]
    mixed256_v, mixed256_r = mixed_positions(256)
    mixed16_v, mixed16_r = mixed_positions(16)
    full256 = [8 * 131072] * 256
    full_req = [0] * 256
    full_req[129] = -1
    result = [
        make_case("r1-p0", 4, 131072, 1, [0], [0]),
        make_case("r1-p8k", 4, 131072, 1, [8232], [0]),
        make_case("r1-p1m", 4, 131072, 1, [1048576], [0]),
        make_case("r16-mixed", 4, 131072, 16, mixed16_v, mixed16_r),
        make_case("r64-slice8k", 4, 131072, 64, sliced, [0] * 64),
        make_case("r256-mixed", 4, 131072, 256, mixed256_v, mixed256_r),
        make_case("r256-full", 4, 131072, 256, full256, full_req),
        make_case("g2-r16-8k", 2, 131072, 16, mixed16_v, mixed16_r),
        make_case("g2-r256-8k", 2, 131072, 256, mixed256_v, mixed256_r),
        make_case("cap513-g2-r5", 2, 513, 5,
                  [0, 7, 8 * 513 + 3, 8 * 513 + 7, 23], [0, 0, 0, 0, -1]),
        make_case("cap1031-g4-r5", 4, 1031, 5,
                  [0, 7, 8 * 1031 + 3, 8 * 1031 + 7, 23], [0, 0, 0, 0, -1]),
        make_case("cap7-g4-r5", 4, 7, 5, [0, 7, 8 * 7 + 3, 8 * 7 + 7, 23], [0, 0, 0, 0, -1]),
    ]
    return [case for case in result if not a.cases or case["label"] in a.cases.split(",")]


def bits_equal(left, right):
    return torch.equal(left.view(torch.int32), right.view(torch.int32))


# The shared top-k emits selected entries in canonical (ascending logical id)
# order only while (length + 31) >> 5 <= 2 * smem_candidate_capacity -- 65,536
# entries for topk=512's 1,024-slot candidate buffers.  Longer rows keep the
# atomic arrival order of the selection pass, so the frozen arm's own output
# bytes vary run to run on identical inputs; the selected set stays exact by
# (score, lower logical id) in both the buffered-refine and overflow-fallback
# paths.  Rows within the limit are gated bitwise.
CANONICAL_ORDER_LIMIT = 2 * 1024 * 32


def sorted_rows(tensor):
    return torch.sort(tensor, dim=-1).values


def ids_with_values(ids, values):
    """Sort one row's ids ascending and carry its values along."""
    order = torch.argsort(ids, dim=-1, stable=True)
    return torch.gather(ids, -1, order), torch.gather(values, -1, order)


assert torch.cuda.is_available()
torch.manual_seed(274)
device_name = torch.cuda.get_device_name()
sms = torch.cuda.get_device_properties(0).multi_processor_count

geometry = sorted({(case["groups"], case["capacity"]) for case in cases()})
programs = {}
for class_name in candidates:
    label = class_name or "default"
    programs[label] = {}
    for groups, capacity in geometry:
        programs[label][(groups, capacity)] = compile_candidate(
            region_selection, class_name, groups=groups, capacity=capacity)
oracle_programs = {(groups, capacity): frozen_module.compile_region_selection_aot(
    groups=groups, capacity=capacity) for groups, capacity in geometry}

compiled_ids = [id(program.compiled) for arms in programs.values() for program in arms.values()]
compiled_ids += [id(program.compiled) for program in oracle_programs.values()]
assert len(set(compiled_ids)) == len(compiled_ids), "two arms share one compiled object (cache collision)"


class Arm:
    def __init__(self, program, rows, groups, capacity):
        self.program = program
        self.rows, self.groups, self.capacity = rows, groups, capacity
        self.out = torch.empty(rows, groups, 4104, dtype=torch.int32, device="cuda")
        self.norm = torch.empty(rows, groups, capacity, dtype=torch.float32, device="cuda")
        self.lens = torch.empty(rows, groups, dtype=torch.int32, device="cuda")
        self.vals = torch.empty(rows, groups, 512, dtype=torch.float32, device="cuda")
        self.ids = torch.empty(rows, groups, 512, dtype=torch.int32, device="cuda")

    def fill(self, norm_fill=POISON):
        self.out.fill_(123)
        self.lens.fill_(-99)
        self.vals.fill_(-99.0)
        self.ids.fill_(-99)
        self.norm.fill_(norm_fill)

    def launch(self, scores, positions, requests, live=None):
        self.program.launch(scores, positions, requests, self.out, self.norm, self.lens,
                            self.vals, self.ids, scalars=[self.rows if live is None else live])

    def run(self, scores, positions, requests, live=None, norm_fill=POISON):
        self.fill(norm_fill)
        self.launch(scores, positions, requests, live)
        torch.cuda.synchronize()


def snapshot(arm):
    return dict(out=arm.out.clone(), norm=arm.norm.clone(), lens=arm.lens.clone(),
                vals=arm.vals.clone(), ids=arm.ids.clone())


def mismatch(left, right, int_view):
    probe_left = left.view(torch.int32) if int_view else left
    probe_right = right.view(torch.int32) if int_view else right
    differ = probe_left != probe_right
    count = int(differ.sum())
    first = torch.nonzero(differ.view(differ.shape[0], -1), as_tuple=False)[:5].tolist() if count else []
    return count, first


def check_against(label, vis, expected, candidate_arm):
    rows = candidate_arm.rows
    if not bits_equal(candidate_arm.lens, expected["lens"]):
        count, first = mismatch(candidate_arm.lens, expected["lens"], True)
        raise AssertionError(
            f"{label}: lengths changed against the frozen arm ({count} entries, first {first})")
    norm_bits = candidate_arm.norm.view(torch.int32)
    expected_bits = expected["norm"].view(torch.int32)
    for row in range(rows):
        visible = int(vis[row])
        if visible:
            assert torch.equal(norm_bits[row, :, :visible], expected_bits[row, :, :visible]), (
                f"{label}: normalized visible prefix changed at row {row}")
        tail = candidate_arm.norm[row, :, visible:]
        assert (tail == POISON).all(), (
            f"{label}: row {row} wrote past the visible prefix (masked tail must stay untouched)")
        if visible <= CANONICAL_ORDER_LIMIT:
            for name in ("out", "vals", "ids"):
                got, want = getattr(candidate_arm, name)[row], expected[name][row]
                if not bits_equal(got, want):
                    count, first = mismatch(got, want, True)
                    raise AssertionError(
                        f"{label}: row {row} {name} changed against the frozen arm "
                        f"({count} entries, first {first})")
        else:
            got, want = sorted_rows(candidate_arm.out[row]), sorted_rows(expected["out"][row])
            if not torch.equal(got, want):
                count, first = mismatch(got, want, True)
                raise AssertionError(
                    f"{label}: row {row} selected token multiset changed against the frozen "
                    f"arm ({count} entries, first {first})")
            got_ids, got_vals = ids_with_values(candidate_arm.ids[row], candidate_arm.vals[row])
            want_ids, want_vals = ids_with_values(expected["ids"][row], expected["vals"][row])
            if not torch.equal(got_ids, want_ids) or not bits_equal(got_vals, want_vals):
                raise AssertionError(
                    f"{label}: row {row} id/value multiset changed against the frozen arm")


def check_frozen_tail_independent(label, vis, base, expected, pass_label):
    """The frozen arm must ignore the masked capacity tail beyond arrival order."""
    if not bits_equal(expected["lens"], base["lens"]):
        raise AssertionError(f"{label}: frozen lengths depend on the masked capacity tail ({pass_label})")
    expected_bits, base_bits = expected["norm"].view(torch.int32), base["norm"].view(torch.int32)
    for row in range(expected["lens"].shape[0]):
        visible = int(vis[row])
        if visible:
            assert torch.equal(expected_bits[row, :, :visible], base_bits[row, :, :visible]), (
                f"{label}: frozen normalized prefix depends on the masked tail "
                f"({pass_label} row {row})")
        if visible <= CANONICAL_ORDER_LIMIT:
            for name in ("out", "vals", "ids"):
                if not bits_equal(expected[name][row], base[name][row]):
                    raise AssertionError(
                        f"{label}: frozen arm depends on the masked capacity tail "
                        f"({pass_label}/{name} row {row})")
        else:
            if not torch.equal(sorted_rows(expected["out"][row]), sorted_rows(base["out"][row])):
                raise AssertionError(
                    f"{label}: frozen selected set depends on the masked capacity tail "
                    f"({pass_label} row {row})")
            got_ids, got_vals = ids_with_values(expected["ids"][row], expected["vals"][row])
            want_ids, want_vals = ids_with_values(base["ids"][row], base["vals"][row])
            if not torch.equal(got_ids, want_ids) or not bits_equal(got_vals, want_vals):
                raise AssertionError(
                    f"{label}: frozen id/value set depends on the masked capacity tail "
                    f"({pass_label} row {row})")


def correctness(case, arms):
    rows, groups, capacity = case["rows"], case["groups"], case["capacity"]
    vis = visible_lengths(case["positions"], case["requests"], capacity)
    observation = dict(
        noncanonical_rows=sum(1 for value in vis.tolist() if value > CANONICAL_ORDER_LIMIT),
        frozen_order_varies=False)
    scores = make_scores(rows, groups, capacity, seed=274 + rows + capacity)
    passes = [("pristine", scores)]
    passes.append(("suffix-nan", poisoned_suffix(scores, vis, float("nan"))))
    passes.append(("suffix-inf", poisoned_suffix(scores, vis, float("inf"))))
    expected_by_pass = {}
    for pass_label, pass_scores in passes:
        oracle_arm = arms["oracle"]
        oracle_arm.run(pass_scores, case["positions"], case["requests"])
        expected = snapshot(oracle_arm)
        expected_by_pass[pass_label] = expected
        if pass_label != "pristine":
            base = expected_by_pass["pristine"]
            for row in range(rows):
                if int(vis[row]) > CANONICAL_ORDER_LIMIT:
                    if (not torch.equal(expected["ids"][row], base["ids"][row])
                            or not torch.equal(expected["out"][row], base["out"][row])):
                        observation["frozen_order_varies"] = True
            check_frozen_tail_independent(case["label"], vis, base, expected, pass_label)
        for label, arm in arms.items():
            if label == "oracle":
                continue
            arm.run(pass_scores, case["positions"], case["requests"])
            check_against(f"{case['label']}/{pass_label}/{label}", vis, expected, arm)
    return vis, observation


def graph_checks(case, arms, scores, vis):
    rows = case["rows"]
    for live in sorted({1, max(1, rows // 2), rows}):
        expected = None
        for label, arm in arms.items():
            arm.fill()
            capture_cuda_graph(
                lambda arm=arm, live=live: arm.launch(scores, case["positions"], case["requests"], live),
                warmup=2)
            if live < rows:
                assert (arm.out[live:] == 123).all(), f"{case['label']}/{label}: graph wrote past live rows"
                assert (arm.lens[live:] == -99).all(), f"{case['label']}/{label}: lengths past live rows"
                assert (arm.vals[live:] == -99.0).all(), f"{case['label']}/{label}: values past live rows"
                assert (arm.ids[live:] == -99).all(), f"{case['label']}/{label}: ids past live rows"
            if label == "oracle":
                expected = snapshot(arm)
            else:
                check_against(f"{case['label']}/graph-live{live}/{label}", vis, expected, arm)


def measure(case, arms, scores):
    rows = case["rows"]
    graphs = {label: capture_cuda_graph(
        lambda arm=arm: arm.launch(scores, case["positions"], case["requests"]),
        warmup=3) for label, arm in arms.items()}
    samples = {label: [] for label in arms}
    labels = list(arms)
    for block in range(a.blocks):
        order = labels if block % 2 == 0 else labels[::-1]
        for label in order:
            samples[label].extend(bench_cuda_graph(graphs[label], replays=a.replays)["replay_us"])
    med = {label: statistics.median(values) for label, values in samples.items()}
    oracle_us = med["oracle"]
    row = dict(case=case["label"], rows=rows, groups=case["groups"],
               capacity=case["capacity"], oracle_us=oracle_us,
               oracle_min_us=min(samples["oracle"]))
    for label in labels:
        if label == "oracle":
            continue
        row[f"{label}_us"] = med[label]
        row[f"{label}_speedup"] = oracle_us / med[label]
        row[f"{label}_min_us"] = min(samples[label])
    print(json.dumps(row), flush=True)
    return row


results = []
for case in cases():
    arms = {"oracle": Arm(oracle_programs[(case["groups"], case["capacity"])],
                          case["rows"], case["groups"], case["capacity"])}
    for label, program in programs.items():
        arms[label] = Arm(program[(case["groups"], case["capacity"])],
                          case["rows"], case["groups"], case["capacity"])
    vis, observation = correctness(case, arms)
    scores = make_scores(case["rows"], case["groups"], case["capacity"],
                         seed=274 + case["rows"] + case["capacity"])
    with kernel_resolution_guard(f"region selection micro {case['label']}"):
        graph_checks(case, arms, scores, vis)
        row = measure(case, arms, scores)
        row.update(observation)
        results.append(row)
    del arms
    torch.cuda.empty_cache()

summary = []
for row in results:
    for label in programs:
        summary.append(dict(case=row["case"], arm=label, oracle_us=row["oracle_us"],
                            arm_us=row[f"{label}_us"], speedup=row[f"{label}_speedup"]))
Path(a.output).write_text(json.dumps(dict(
    device=device_name, multi_processor_count=sms, repo=str(repo),
    oracle=str(frozen_path), oracle_sha256=ORACLE_SHA256, candidate_sha256=candidate_sha,
    topk_closure=TOPK_CLOSURE, candidate_classes=candidates,
    canonical_order_limit=CANONICAL_ORDER_LIMIT,
    memory_cache=os.environ["B12X_COMPILE_MEMORY_CACHE"],
    disk_cache=os.environ["B12X_COMPILE_DISK_CACHE"],
    spec_memo=os.environ["B12X_COMPILE_SPEC_MEMO"],
    results=results, summary=summary), indent=2))
print(json.dumps(dict(output=str(a.output), cases=len(results), sms=sms)), flush=True)
