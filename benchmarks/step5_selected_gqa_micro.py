"""Same-session frozen/current AOT microbench for selected-position paged GQA.

Oracle: the frozen serving source of ``b12x/attention/paged/_selected_forward.py``
(loaded under a private module name, so its compiled object is separate from the
candidate's).  Candidate: ``--repo``.

Selection layouts modelled on the production producers in
``b12x/attention/paged/_region_selection.py``:

* ``prefix``  - valid entries are columns ``[0, live)`` (control layout).
* ``tail``    - SWA ``_sliding_positions``: the last ``live`` of 512 columns are
  valid, leading columns are ``-1`` padding.
* ``regions`` - CSA ``_expand_regions``: a dense prefix of the visible regions
  strictly before the current partial region, ``-1`` in between, plus the
  current region alone in the 4096..4103 tail, so the last tile always holds
  at least one selected position.

Run only under the hardware queue.  Compile caches: the b12x memory cache keys
explicit-spec compiles by kernel name + geometry only, so an oracle and a
candidate with identical geometry would collide and the second arm would reuse
the first arm's cubin.  This harness disables the memory cache (the disk cache
still keys on the module name) and asserts the two compiled objects differ.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ORACLE_SHA256 = "42fac73b40de013976d2a6bcd0a517eb0702383af3bc35fd4079d350a0e0bfab"

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--oracle-repo", required=True)
p.add_argument("--output", required=True)
p.add_argument("--preflight", action="store_true")
p.add_argument("--only", default="")
p.add_argument("--replays", type=int, default=24)
p.add_argument("--blocks", type=int, default=4)
a = p.parse_args()
repo, oracle = Path(a.repo).resolve(), Path(a.oracle_repo).resolve()
frozen_path = oracle / "b12x/attention/paged/_selected_forward.py"
selected_path = repo / "b12x/attention/paged/_selected_forward.py"

if a.preflight:
    assert (repo / "b12x/integration/cuteafd/region_attention.py").is_file()
    assert (repo / "benchmarks/common.py").is_file()
    assert selected_path.is_file()
    assert frozen_path.is_file()
    sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert sha == ORACLE_SHA256, f"frozen selected module sha {sha} != {ORACLE_SHA256}"
    candidate_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    print(json.dumps(dict(preflight_ok=True, candidate_sha256=candidate_sha,
                          oracle_sha256=sha)), flush=True)
    sys.exit(0)

# Kill the explicit-spec memory-cache collision before any b12x import.
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

sys.path.insert(0, str(repo))
import torch  # noqa: E402
from benchmarks.common import bench_cuda_graph, capture_cuda_graph  # noqa: E402
from b12x.integration.cuteafd import region_attention  # noqa: E402
import b12x.attention.paged._selected_forward as selected  # noqa: E402


def frozen(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_selected = frozen("b12x.attention.paged.step5_frozen_selected", frozen_path)


def compile_oracle(**kw):
    original = selected._SelectedPositionPagedForwardKernel
    selected._SelectedPositionPagedForwardKernel = (
        old_selected._SelectedPositionPagedForwardKernel
    )
    try:
        return region_attention.compile_selected_gqa_aot(**kw)
    finally:
        selected._SelectedPositionPagedForwardKernel = original


GEOMS = {
    # SWA: GQA 32 query heads over 2 KV heads, one shared 512-wide window.
    "swa": dict(query_heads=32, kv_heads=2, page_rows=64, selection_width=512,
                selection_groups=1, head_dim=192),
    # CSA: GQA 64 query heads over 4 KV heads, per-KV-head 4104-wide selection.
    "csa": dict(query_heads=64, kv_heads=4, page_rows=64, selection_width=4104,
                selection_groups=4, head_dim=192),
}


def fill_positions(positions, layout, live):
    """Write one selection row in the requested production/control layout."""
    assert positions.dim() == 1, positions.shape
    width = positions.shape[0]
    positions.fill_(-1)
    if layout == "prefix":
        positions[:live] = torch.arange(live, device="cuda", dtype=torch.int32)
    elif layout == "tail":
        positions[width - live:] = torch.arange(live, device="cuda",
                                                dtype=torch.int32)
    elif layout == "regions":
        # ``_expand_regions``: the dense prefix holds the top visible regions
        # (strictly before the current partial region); the current region
        # sits alone in the 4096..4103 tail and is never duplicated.
        tail_region = (live - 1) // 8
        dense = 8 * tail_region
        positions[:dense] = torch.arange(dense, device="cuda", dtype=torch.int32)
        positions[dense:4096] = -1
        tail = torch.arange(dense, dense + 8, device="cuda", dtype=torch.int32)
        positions[4096:4104] = tail
    else:
        raise ValueError(layout)


def make_case(geom, layout, live, rows, request=0):
    head_dim = geom["head_dim"]
    kv_heads = geom["kv_heads"]
    q_heads = geom["query_heads"]
    width = geom["selection_width"]
    groups = geom["selection_groups"]
    pages = (live + geom["page_rows"] - 1) // geom["page_rows"]
    q = torch.randn(rows, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, geom["page_rows"], kv_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    requests = torch.full((rows,), request, device="cuda", dtype=torch.int32)
    pos = torch.full((rows,), live - 1, device="cuda", dtype=torch.int64)
    positions = torch.empty((rows, groups, width), device="cuda", dtype=torch.int32)
    for row in range(rows):
        for group in range(groups):
            fill_positions(positions[row, group], layout, live)
    out = torch.empty_like(q)
    return [q, k, v, None, None, table, requests, positions, pos, None, None, out]


def make_scalars(live, rows, geom):
    pages = (live + geom["page_rows"] - 1) // geom["page_rows"]
    return [pages, 1, pages, geom["head_dim"] ** -0.5, rows, 1]


def reference(case, geom):
    q, k, v, _, _, table, requests, positions, pos, _, _, out = case
    head_dim, kv_heads, q_heads = geom["head_dim"], geom["kv_heads"], geom["query_heads"]
    groups = geom["selection_groups"]
    rows = q.shape[0]
    ours = out.clone().float()
    ref = torch.empty_like(ours)
    for row in range(rows):
        for kv_head in range(kv_heads):
            group = kv_head // (kv_heads // groups)
            logical = positions[row, group].long()
            valid = (logical >= 0) & (logical <= int(pos[row]))
            logical = logical[valid]
            page = table[0][logical // geom["page_rows"]].long()
            off = logical % geom["page_rows"]
            keys = k[page, off, kv_head].float()
            vals = v[page, off, kv_head].float()
            first = kv_head * (q_heads // kv_heads)
            qh = q[row, first:first + q_heads // kv_heads].float()
            scores = (qh @ keys.T) * (head_dim ** -0.5)
            probs = scores.softmax(dim=-1)
            ref[row, first:first + q_heads // kv_heads] = probs @ vals
    delta = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    return dict(max_abs_delta=delta, max_abs_ref=scale)


def measure(case, scalars, label, oracle_prog, cand_prog, replays, blocks):
    def launch(prog, tensors):
        prog.launch(*tensors, scalars=scalars)

    x = list(case)
    launch(oracle_prog, x)
    expected = x[-1].clone()
    x[-1].fill_(float("nan"))
    launch(cand_prog, x)
    torch.cuda.synchronize()
    actual = x[-1]
    bits_equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if not bits_equal:
        bad = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        delta = (actual.float() - expected.float()).abs()
        max_delta = torch.nan_to_num(delta, nan=float("inf")).max().item()
        raise AssertionError(f"{label}: frozen output changed, {bad} elements, max delta {max_delta}")

    graphs = {
        "oracle": capture_cuda_graph(lambda: launch(oracle_prog, x), warmup=3),
        "candidate": capture_cuda_graph(lambda: launch(cand_prog, x), warmup=3),
    }
    samples = {"oracle": [], "candidate": []}
    order = ("oracle", "candidate")
    for block in range(blocks):
        seq = order if block % 2 == 0 else order[::-1]
        for arm in seq:
            samples[arm].extend(
                bench_cuda_graph(graphs[arm], replays=replays)["replay_us"]
            )
    med = {arm: statistics.median(v) for arm, v in samples.items()}
    row = dict(case=label, oracle_us=med["oracle"], candidate_us=med["candidate"],
               speedup=med["oracle"] / med["candidate"], bitwise_equal=True,
               oracle_min_us=min(samples["oracle"]),
               candidate_min_us=min(samples["candidate"]))
    print(json.dumps(row), flush=True)
    return row


assert torch.cuda.is_available()
torch.manual_seed(274)
results = []

CASES = [
    # Control prefix layouts, comparable with the recorded baseline.
    ("swa", "prefix", 64, 1, 0), ("swa", "prefix", 256, 1, 0),
    ("swa", "prefix", 512, 1, 0), ("swa", "prefix", 512, 16, 0),
    # SWA production sliding window: the last `live` columns of 512 are valid.
    ("swa", "tail", 64, 1, 0), ("swa", "tail", 256, 1, 0),
    ("swa", "tail", 384, 1, 0), ("swa", "tail", 64, 1, -1),
    # CSA production expansion: dense region prefix plus current-region tail.
    ("csa", "prefix", 64, 1, 0), ("csa", "prefix", 256, 1, 0),
    ("csa", "prefix", 1024, 1, 0), ("csa", "prefix", 4104, 1, 0),
    ("csa", "prefix", 4104, 16, 0),
    ("csa", "regions", 64, 1, 0), ("csa", "regions", 256, 1, 0),
    ("csa", "regions", 1024, 1, 0),
]

compiled = {}
for geom_name, geom in GEOMS.items():
    compiled[geom_name] = dict(
        oracle=compile_oracle(**geom),
        candidate=region_attention.compile_selected_gqa_aot(**geom),
    )
    assert compiled[geom_name]["oracle"].compiled is not compiled[geom_name]["candidate"].compiled, (
        f"{geom_name}: oracle and candidate share one compiled object (cache collision)"
    )

for geom_name, layout, live, rows, request in CASES:
    label = f"{geom_name}-{layout}-live{live}-r{rows}"
    if request < 0:
        label += "-invalid"
    if a.only and a.only not in label:
        continue
    geom = GEOMS[geom_name]
    case = make_case(geom, layout, live, rows, request)
    scalars = make_scalars(live, rows, geom)
    row = measure(case, scalars, label, compiled[geom_name]["oracle"],
                  compiled[geom_name]["candidate"], a.replays, a.blocks)
    if label in ("swa-prefix-live64-r1", "swa-tail-live64-r1",
                 "csa-regions-live64-r1"):
        try:
            row["reference"] = reference(case, geom)
        except Exception as error:  # informational only
            row["reference"] = dict(error=repr(error))
    results.append(row)

Path(a.output).write_text(json.dumps(dict(
    device=torch.cuda.get_device_name(), oracle=str(oracle), repo=str(repo),
    memory_cache=os.environ["B12X_COMPILE_MEMORY_CACHE"], results=results), indent=2))
print(json.dumps(dict(output=str(a.output), cases=len(results))), flush=True)
