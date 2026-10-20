"""Same-session frozen/current microbench for the canonical BF16 projection.

Oracle: the frozen serving source of ``b12x/gemm/bf16_gemv/_skinny.py`` and
``b12x/gemm/bf16_gemv/_prefill.py`` from a pinned checkout, loaded together
under a private package name (``step5_frozen``), so its kernel classes and its
compiled cubins are distinct objects from the candidate's.  ``_skinny.py``'s
lazy ``from ._prefill import Bf16PrefillKernel`` is rewritten to the frozen
package name; every other byte of both files is loaded unchanged, and the
harness asserts the file hashes of the pinned commit.

Candidate: either the live serving wrapper (``--mode serving``) or explicit
retiles of the same TMA kernel (``--mode arms``).  Every retile keeps the
per-output ascending-K MMA accumulation, so outputs must be bitwise identical
to the frozen kernel at every live-row count; that is checked before timing,
and the frozen kernel's own row invariance is checked across row counts.

Timing: CUDA-graph replay with an L2 eviction sweep before every cold replay
(the serving regime: the profile shows no per-launch L2 reuse between C1 and
C16) and a warm pass for reference.  Blocks of arms are interleaved in both
directions; medians are reported.

The b12x explicit-spec memory cache keys a compile by its spec alone, so the
oracle compile and each arm compile carry distinct program names; the harness
also disables the memory cache and asserts the compiled objects differ.

Run only under the hardware fair queue.  CPU-only preflight source/hash checks:
``--preflight``.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ORACLE_COMMIT = "e88f2d091829ad06919df8c097375afe6c5ea813"
ORACLE_SHA256 = {
    "b12x/gemm/bf16_gemv/_prefill.py":
        "e888b07756cc4f2d264d3713248b5fde753233d24216023af45754ac477cf875",
    "b12x/gemm/bf16_gemv/_skinny.py":
        "bce07062dce4f01b6e67b350dde7f6f1ee25c213dcc4ea3b74b21fee751a6abb",
    "b12x/integration/cuteafd/pointwise.py":
        "209a776a3604ec27592589a8a40efe83159656c93ef50949574e77d86715e64f",
}
FROZEN_IMPORT = "from ._prefill import Bf16PrefillKernel"
FROZEN_IMPORT_REWRITTEN = "from step5_frozen._prefill import Bf16PrefillKernel"
ORACLE_PROGRAM_NAME = "bf16_projection_step5_oracle"

# Step-5 measured projection shapes from the #274 timeline inventory:
# (name, K = input width, N = output features, output dtype).
SHAPES = {
    "csa_o": (12288, 4096, "bfloat16"),
    "swa_o": (6144, 4096, "float32"),
    "q": (4096, 6144, "bfloat16"),
    "kv": (4096, 384, "bfloat16"),
    "gate_up": (4096, 3072, "bfloat16"),
}

# One kernel retile per arm; every arm is a (warp_layout, tile_m, tile_n,
# num_stages) choice over the same uncompensated m16n8k16 K-order chain.
# ``rows`` restricts the live-row counts an arm is timed at.
BIG_ARMS = [
    dict(label="d16n64s6", warp_layout=(1, 4), tile_m=16, tile_n=64,
         num_stages=6),
    dict(label="d16n64s9", warp_layout=(1, 4), tile_m=16, tile_n=64,
         num_stages=9),
    dict(label="d16n32s8", warp_layout=(1, 4), tile_m=16, tile_n=32,
         num_stages=8),
    dict(label="d16n32s12", warp_layout=(1, 4), tile_m=16, tile_n=32,
         num_stages=12),
    dict(label="d16n16s8", warp_layout=(1, 2), tile_m=16, tile_n=16,
         num_stages=8),
    dict(label="wide128s2", warp_layout=(2, 2), tile_m=128, tile_n=128,
         num_stages=2, rows=(256,)),
]
KV_ARMS = [
    dict(label="kv_d16n8s12", compute_warps=1, tile_n=8, num_stages=12),
    dict(label="kv_d16n8s16", compute_warps=1, tile_n=8, num_stages=16),
    dict(label="kv_d16n16s10", compute_warps=1, tile_n=16, num_stages=10),
    dict(label="kv_d16n16s8", warp_layout=(1, 2), tile_m=16, tile_n=16,
         num_stages=8),
    dict(label="kv_d16n32s6", warp_layout=(1, 4), tile_m=16, tile_n=32,
         num_stages=6),
]

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--oracle-repo", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--work-dir", default="")
parser.add_argument("--mode", choices=("arms", "serving"), default="arms")
parser.add_argument("--preflight", action="store_true")
parser.add_argument("--only", default="")
parser.add_argument("--rows", default="1,16,64,256")
parser.add_argument("--replays", type=int, default=24)
parser.add_argument("--blocks", type=int, default=4)
parser.add_argument("--hbm-tb-s", type=float, default=1.8,
                    help="planning HBM ceiling for the roofline fraction")
parser.add_argument("--require-parity", action="store_true",
                    help="assert the repo projection sources still equal the oracle "
                         "bytes (the pre-edit freeze check)")
args = parser.parse_args()
repo = Path(args.repo).resolve()
oracle = Path(args.oracle_repo).resolve()
ROWS = [int(v) for v in args.rows.split(",") if v]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    """CPU-only: pinned oracle files, fork parity, and the single-line rewrite."""
    import subprocess

    assert (repo / "benchmarks/common.py").is_file()
    assert (repo / "b12x/integration/cuteafd/pointwise.py").is_file()
    report = {"preflight_ok": True, "oracle_repo": str(oracle), "files": {},
              "fork_matches_oracle": True}
    for rel, expected in ORACLE_SHA256.items():
        frozen_path = oracle / rel
        live_path = repo / rel
        assert frozen_path.is_file(), f"missing oracle file {rel}"
        assert live_path.is_file(), f"missing repo file {rel}"
        frozen_sha = sha256(frozen_path)
        live_sha = sha256(live_path)
        assert frozen_sha == expected, f"{rel}: oracle sha {frozen_sha} != {expected}"
        report["files"][rel] = frozen_sha
        report["fork_matches_oracle"] &= live_sha == frozen_sha
    if args.require_parity:
        assert report["fork_matches_oracle"], (
            "fork projection sources are not the oracle bytes; "
            "freeze the oracle before touching them")
    text = (oracle / "b12x/gemm/bf16_gemv/_skinny.py").read_text()
    assert text.count(FROZEN_IMPORT) == 1, "frozen _skinny.py import rewrite is not unique"
    head = ""
    try:
        head = subprocess.run(
            ["git", "-C", str(oracle), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit_file = oracle / "COMMIT"
        if commit_file.is_file():
            head = commit_file.read_text().strip()
    report["oracle_commit"] = head or "unavailable"
    if head:
        assert head == ORACLE_COMMIT, f"oracle HEAD {head} != {ORACLE_COMMIT}"
    print(json.dumps(report), flush=True)
    sys.exit(0)


if args.preflight:
    preflight()

# Kill the explicit-spec memory-cache collision before any b12x import.
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

sys.path.insert(0, str(repo))
import cutlass  # noqa: E402
import cutlass.utils as cutlass_utils  # noqa: E402
import torch  # noqa: E402

from benchmarks.common import (  # noqa: E402
    bench_cuda_graph, capture_cuda_graph, make_l2_flush_fn)
from b12x.integration.cuteafd import _common  # noqa: E402
from b12x.integration.cuteafd import pointwise as live_pointwise  # noqa: E402
import b12x.gemm.bf16_gemv._prefill as live_prefill  # noqa: E402
import b12x.gemm.bf16_gemv._skinny as live_skinny  # noqa: E402


def load_frozen(work: Path):
    """Load the pinned projection sources as the private ``step5_frozen`` package."""
    pkg = work / "frozen_pkg" / "step5_frozen"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("_prefill.py", "_skinny.py"):
        src = oracle / "b12x/gemm/bf16_gemv" / name
        assert sha256(src) == ORACLE_SHA256[f"b12x/gemm/bf16_gemv/{name}"]
        text = src.read_text()
        if name == "_skinny.py":
            assert text.count(FROZEN_IMPORT) == 1
            text = text.replace(FROZEN_IMPORT, FROZEN_IMPORT_REWRITTEN)
        (pkg / name).write_text(text)
    root = str(pkg.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    __import__("step5_frozen._prefill")  # _skinny imports it lazily; load it eagerly
    return __import__("step5_frozen._skinny", fromlist=["*"])


def load_frozen_pointwise():
    """Oracle pointwise wrapper; its relative imports stay in the live package."""
    path = oracle / "b12x/integration/cuteafd/pointwise.py"
    assert sha256(path) == ORACLE_SHA256["b12x/integration/cuteafd/pointwise.py"]
    spec = importlib.util.spec_from_file_location(
        "b12x.integration.cuteafd.step5_frozen_pointwise", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


work = Path(args.work_dir) if args.work_dir else Path(args.output).resolve().parent / "work"
work.mkdir(parents=True, exist_ok=True)
frozen_skinny = load_frozen(work)
frozen_prefill = sys.modules["step5_frozen._prefill"]
frozen_pointwise = load_frozen_pointwise()


def prove_separate_oracle():
    """The frozen classes and their prefill module are not the live ones."""
    proof = dict(
        frozen_skinny_module=frozen_skinny.__name__,
        frozen_prefill_module=frozen_prefill.__name__,
        skinny_class_distinct=(
            frozen_skinny.RoutedBf16Projection is not live_skinny.RoutedBf16Projection),
        tma_class_distinct=(
            frozen_skinny.TmaBf16Projection is not live_skinny.TmaBf16Projection),
        prefill_class_distinct=(
            frozen_prefill.Bf16PrefillKernel is not live_prefill.Bf16PrefillKernel),
    )
    assert all(proof[key] for key in (
        "skinny_class_distinct", "tma_class_distinct", "prefill_class_distinct")), proof
    probe = frozen_skinny.TmaBf16Projection(256, 256, out_dtype=cutlass.BFloat16,
                                            tiles_by_rows=())
    proof["kernel_class_module"] = type(probe.kernel).__module__
    assert proof["kernel_class_module"] == "step5_frozen._prefill", proof
    return proof


def torch_dtype(dtype):
    return torch.float32 if dtype == "float32" else torch.bfloat16


def cute_dtype(dtype):
    return cutlass.Float32 if dtype == "float32" else cutlass.BFloat16


def operations(width, outputs, dtype):
    out_dtype = torch_dtype(dtype)
    return (
        _common.Operand("x", torch.bfloat16, f"[rows,{width}]"),
        _common.Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
        _common.Operand("out", out_dtype, f"[rows,{outputs}]", "out"),
    ), (_common.Scalar("rows"),)


def compile_oracle(width, outputs, dtype):
    """Compile through the frozen wrapper with the frozen kernel classes."""
    original_compile = frozen_pointwise.compile_program
    original_routed = live_skinny.RoutedBf16Projection
    original_tma = live_skinny.TmaBf16Projection

    def renamed(launch, *, name, **kwargs):
        assert name == "bf16_projection"
        return original_compile(launch, name=ORACLE_PROGRAM_NAME, **kwargs)

    frozen_pointwise.compile_program = renamed
    live_skinny.RoutedBf16Projection = frozen_skinny.RoutedBf16Projection
    live_skinny.TmaBf16Projection = frozen_skinny.TmaBf16Projection
    try:
        program = frozen_pointwise.compile_bf16_projection_aot(
            width=width, outputs=outputs, output_dtype=torch_dtype(dtype))
    finally:
        frozen_pointwise.compile_program = original_compile
        live_skinny.RoutedBf16Projection = original_routed
        live_skinny.TmaBf16Projection = original_tma
    assert program.name == ORACLE_PROGRAM_NAME
    return program


def compile_serving(width, outputs, dtype):
    return live_pointwise.compile_bf16_projection_aot(
        width=width, outputs=outputs, output_dtype=torch_dtype(dtype))


def compile_arm(width, outputs, dtype, config, label):
    """Compile one retile of the same kernel under its own program name."""
    kwargs = {k: v for k, v in config.items() if k not in ("rows", "label")}
    launch = live_skinny.TmaBf16Projection(
        outputs, width, out_dtype=cute_dtype(dtype), tiles_by_rows=(), **kwargs)
    kernel = launch.kernel
    assert kernel.compensated is False, f"{label}: arm must stay uncompensated"
    smem = kernel.buffer_align_bytes + (
        kernel.tile_m + kernel.tile_n) * kernel.tile_k * 2 * kernel.num_stages
    capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_120")
    assert smem <= capacity, f"{label}: needs {smem} B smem, capacity {capacity}"
    operands, scalars = operations(width, outputs, dtype)
    program = _common.compile_program(
        launch,
        name=f"bf16_projection_step5_arm_{label}",
        operands=operands,
        scalars=scalars,
        key=(width, outputs, dtype, launch.key()),
        geometry=dict(width=width, outputs=outputs, output_dtype=dtype,
                      skinny_rows=0, arm=label, config=kwargs),
    )
    return program, launch.key(), dict(
        warp_layout=getattr(kernel, "warp_layout", None),
        tile_m=kernel.tile_m, tile_n=kernel.tile_n,
        num_stages=kernel.num_stages, num_threads=kernel.num_threads,
        grid_n_tiles=(outputs + kernel.tile_n - 1) // kernel.tile_n)


def frozen_default_kernel(width, outputs, dtype):
    launch = frozen_skinny.TmaBf16Projection(
        outputs, width, out_dtype=cute_dtype(dtype), tiles_by_rows=())
    return launch.kernel


def bit_view(tensor):
    return tensor.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


def check_bitwise(label, oracle_out, candidate_out):
    if not torch.equal(bit_view(oracle_out), bit_view(candidate_out)):
        bad = (bit_view(oracle_out) != bit_view(candidate_out)).sum().item()
        delta = (oracle_out.float() - candidate_out.float()).abs().max().item()
        raise AssertionError(
            f"{label}: output is not bitwise identical to the frozen kernel "
            f"({bad} differing elements, max |delta| {delta})")


def dump(result):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


def measure_shape(name, width, outputs, dtype, device, result):
    oracle_prog = compile_oracle(width, outputs, dtype)
    default = frozen_default_kernel(width, outputs, dtype)
    programs = {"oracle": oracle_prog}
    kernels = {"oracle": dict(warp_layout=None, tile_m=default.tile_m,
                              tile_n=default.tile_n, num_stages=default.num_stages,
                              num_threads=default.num_threads,
                              grid_n_tiles=(outputs + default.tile_n - 1)
                              // default.tile_n)}
    restricted = {"oracle": None}
    errors = {}
    if args.mode == "serving":
        programs["serving"] = compile_serving(width, outputs, dtype)
        assert programs["serving"].compiled is not oracle_prog.compiled, (
            "serving wrapper reused the oracle cubin")
        kernels["serving"] = kernels["oracle"]
        restricted["serving"] = None
    else:
        arms = KV_ARMS if name == "kv" else BIG_ARMS
        for arm in arms:
            label = arm["label"]
            if args.only and args.only not in f"{name}-{label}":
                continue
            try:
                program, key, kernel = compile_arm(width, outputs, dtype, arm, label)
            except Exception as error:  # a rejected arm is a result, not a crash
                errors[label] = repr(error)
                print(json.dumps(dict(shape=name, arm=label, rejected=repr(error))),
                      flush=True)
                continue
            assert program.compiled is not oracle_prog.compiled, f"{label}: cache collision"
            programs[label] = program
            kernels[label] = kernel
            restricted[label] = arm.get("rows")

    hbm_bytes = outputs * width * 2
    shape_row = result["shapes"][name] = dict(
        width=width, outputs=outputs, output_dtype=dtype,
        weight_bytes=hbm_bytes, rows={}, errors=errors,
        arms={label: kernels[label] for label in programs})
    dump(result)

    weight = torch.randn(outputs, width, dtype=torch.bfloat16, device=device)
    # One prefix tensor per shape: each row count must see byte-identical row
    # data, otherwise the row-invariance check would only compare different
    # random draws (CUDA RNG values depend on the fill size).
    x_prefix = torch.randn(max(ROWS), width, dtype=torch.bfloat16, device=device)
    kept = [weight, x_prefix]  # tensors referenced by live CUDA graphs stay allocated
    flush = make_l2_flush_fn(True)
    cold = {label: [] for label in programs}
    warm = {label: [] for label in programs}
    frozen_row0 = {}
    failed = set()
    for rows in ROWS:
        active = [label for label in programs
                  if label not in failed
                  and (restricted[label] is None or rows in restricted[label])]
        if not active:
            continue
        x = x_prefix[:rows]
        out = torch.empty(rows, outputs, dtype=torch_dtype(dtype), device=device)
        kept.append(out)
        expected = None
        for label in list(active):
            try:
                out.zero_()
                programs[label].launch(x, weight, out, scalars=[rows])
                actual = out.clone()
                if label == "oracle":
                    expected = actual
                    frozen_row0[rows] = actual[0].clone()
                else:
                    check_bitwise(f"{name}-{label}-r{rows}", expected, actual)
            except Exception as error:
                if label == "oracle":
                    raise  # the baseline itself must work
                # A rejected arm is a result, not a crash: drop it and keep the rest.
                failed.add(label)
                errors[label] = repr(error)
                print(json.dumps(dict(shape=name, arm=label, rows=rows,
                                      rejected=repr(error))), flush=True)
            finally:
                actual = None
        active = [label for label in active if label not in failed]
        graphs = {}
        for label in active:
            try:
                graphs[label] = capture_cuda_graph(
                    lambda p=programs[label]: p.launch(x, weight, out, scalars=[rows]),
                    warmup=3)
            except Exception as error:
                if label == "oracle":
                    raise
                failed.add(label)
                errors[label] = repr(error)
                print(json.dumps(dict(shape=name, arm=label, rows=rows,
                                      capture_error=repr(error))), flush=True)
        active = [label for label in active if label in graphs]
        for block in range(args.blocks):
            sequence = active if block % 2 == 0 else active[::-1]
            for label in sequence:
                cold[label].extend(
                    bench_cuda_graph(graphs[label], replays=args.replays,
                                     l2_flush=flush)["replay_us"])
                warm[label].extend(
                    bench_cuda_graph(graphs[label], replays=args.replays)["replay_us"])
        row = {}
        for label in active:
            cold_us = statistics.median(cold[label])
            warm_us = statistics.median(warm[label])
            entry = dict(
                cold_us=cold_us, warm_us=warm_us,
                cold_min_us=min(cold[label]),
                achieved_tb_s=hbm_bytes / (cold_us * 1e-6) / 1e12,
                roofline_fraction=(hbm_bytes / (args.hbm_tb_s * 1e12))
                / (cold_us * 1e-6))
            if label not in ("oracle", "serving") and "oracle" in row:
                entry["speedup_cold"] = row["oracle"]["cold_us"] / cold_us
                entry["speedup_warm"] = row["oracle"]["warm_us"] / warm_us
            row[label] = entry
        if "serving" in row:
            row["serving"]["speedup_cold"] = (
                row["oracle"]["cold_us"] / row["serving"]["cold_us"])
            row["serving"]["speedup_warm"] = (
                row["oracle"]["warm_us"] / row["serving"]["warm_us"])
        shape_row["rows"][str(rows)] = row
        print(json.dumps(dict(shape=name, rows=rows, arms=row)), flush=True)
        dump(result)
        # Keep the per-row medians bounded; graphs go with the next row.
        del graphs
    first = ROWS[0]
    for rows in ROWS[1:]:
        if first in frozen_row0 and rows in frozen_row0:
            check_bitwise(f"{name}-oracle-row-invariance-r{rows}",
                          frozen_row0[first].unsqueeze(0),
                          frozen_row0[rows].unsqueeze(0))
    del kept, weight
    torch.cuda.empty_cache()


assert torch.cuda.is_available()
torch.manual_seed(274)
device = torch.device("cuda")
properties = torch.cuda.get_device_properties(device)
result = dict(
    device=torch.cuda.get_device_name(device),
    capability=list(torch.cuda.get_device_capability(device)),
    sm_count=properties.multi_processor_count,
    l2_bytes=getattr(properties, "L2_cache_size", None),
    oracle_commit=ORACLE_COMMIT,
    oracle_sha256=ORACLE_SHA256,
    memory_cache=os.environ["B12X_COMPILE_MEMORY_CACHE"],
    mode=args.mode,
    rows=ROWS,
    hbm_tb_s=args.hbm_tb_s,
    oracle_proof=prove_separate_oracle(),
    shapes={},
)

# Prove the frozen and live serving compiles are separate identities and agree.
probe_width, probe_outputs, probe_dtype = SHAPES["csa_o"]
frozen_probe = compile_oracle(probe_width, probe_outputs, probe_dtype)
live_probe = compile_serving(probe_width, probe_outputs, probe_dtype)
assert frozen_probe.compiled is not live_probe.compiled, (
    "frozen and live serving compiles share one cubin")
x = torch.randn(4, probe_width, dtype=torch.bfloat16, device=device)
w = torch.randn(probe_outputs, probe_width, dtype=torch.bfloat16, device=device)
out_a = torch.empty(4, probe_outputs, dtype=torch_dtype(probe_dtype), device=device)
out_b = torch.empty_like(out_a)
frozen_probe.launch(x, w, out_a, scalars=[4])
live_probe.launch(x, w, out_b, scalars=[4])
torch.cuda.synchronize()
check_bitwise("frozen-vs-live-serving-probe", out_a, out_b)
result["oracle_proof"].update(
    frozen_vs_live_compiled_distinct=True,
    frozen_vs_live_probe_bitwise=True,
    live_serving_name=live_probe.name)
del x, w, out_a, out_b, frozen_probe, live_probe
torch.cuda.empty_cache()
dump(result)

for shape_name, (width, outputs, dtype) in SHAPES.items():
    if args.only and args.only.split("-")[0] != shape_name:
        continue
    measure_shape(shape_name, width, outputs, dtype, device, result)

print(json.dumps(dict(output=str(args.output), shapes=len(result["shapes"]))),
      flush=True)
