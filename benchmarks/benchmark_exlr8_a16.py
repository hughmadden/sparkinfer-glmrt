#!/usr/bin/env python3
"""Graph-first matched-byte EXLR8 up1 versus EXL3 K4 full Spark expert step.

Both arms start from the same standard Fp8K32 wire rows (E4M3 + UE8M0 per 32)
and end with BF16 response rows. EXLR8: one fused group/ingress launch
(dequantize, signs.in, H128, F16) then K1, whose last stage sums, inverse
rotates (signs.out) and writes BF16. EXL3 K4: the worker's wire decode to BF16,
then the EXL3 kernel with its BF16 top-k sum. Cold L2: every replay streams
distinct weight copies spanning more than 2x L2 (consecutive layers).
An optional control arm is master's group + K1 (F32 rotated output).
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import types

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import benchmark_mixed_trellis as exl3
from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot


def compact_unit_descriptors(bits):
    """Build immutable ascending ownership once, before capture or timed work."""
    if bits.dtype != torch.int32 or bits.ndim != 3 or bits.shape[2] != 3:
        raise ValueError("EXLR8 bits must be int32 [E,U,3]")
    host = bits.detach().cpu()
    if not host.shape[0] or not host.shape[1]:
        raise ValueError("EXLR8 ownership must have experts and units")
    active = host != 0
    if not torch.equal(active, active[..., :1].expand_as(active)):
        raise ValueError("EXLR8 unit projections must agree on ownership")
    if not bool(((host == 0) | ((host >= 2) & (host <= 4))).all()):
        raise ValueError("EXLR8 units support K2, K3 or K4")
    counts = active[..., 0].sum(1).to(torch.int32)
    mapping = torch.zeros(host.shape[:2], dtype=torch.int32)
    for expert in range(host.shape[0]):
        owned = torch.nonzero(active[expert, :, 0], as_tuple=False).flatten()
        mapping[expert, :owned.numel()] = owned.to(torch.int32)
    return (counts.to(bits.device), mapping.to(bits.device), max(1, int(counts.max())))


def compact_program(program):
    return any(operand.name == "unit_count" for operand in program.operands)


@triton.jit
def _wire_decode(wire, out, HIDDEN: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr):
    # The worker's EXL3 wire decoder: bf16(float(e4m3) * float(ue8m0)).
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(wire + row * ROW + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    exponent = tl.load(wire + row * ROW + HIDDEN + cols // 32).to(tl.int32)
    scale = tl.where(exponent == 0, 2.0 ** -127, tl.exp2((exponent - 127).to(tl.float32)))
    tl.store(out + row * HIDDEN + cols, (value * scale).to(tl.bfloat16))


def wire_decode(wire, out, rows):
    _wire_decode[(6144 // 1024, rows)](wire, out, HIDDEN=6144, ROW=6144 + 192, BLOCK=1024)


def make_wire(x):
    rows, h = x.shape
    blocks = x.float().reshape(rows, h // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    e4m3 = (blocks / torch.exp2(exponent)).to(torch.float8_e4m3fn)
    wire = torch.empty((rows, h + h // 32), dtype=torch.uint8, device=x.device)
    wire[:, :h] = e4m3.reshape(rows, h).view(torch.uint8)
    wire[:, h:] = (exponent.reshape(rows, h // 32) + 127).to(torch.uint8)
    return wire


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def timing(graph, replays):
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000/replays)
    return statistics.median(values)


@triton.jit
def _read_stream(src, sums, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    # Keep every vectorized global load live with only one 4-byte store per CTA.
    tl.store(sums + tl.program_id(0), tl.sum(values, axis=0))


@triton.jit
def _exchange_traffic(src, dst, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    while base < n_elements:
        offsets = base + tl.arange(0, BLOCK)
        values = tl.load(src + offsets, mask=offsets < n_elements, other=0.0)
        tl.store(dst + offsets, values, mask=offsets < n_elements)
        base += step


def read_roofline():
    size = 256 * 1024 * 1024
    block = 4096
    src = torch.full((size // 4,), 17.0, dtype=torch.float32, device="cuda")
    sums = torch.empty((src.numel() // block,), device="cuda")
    graph = capture(lambda: _read_stream[(sums.numel(),)](src, sums, BLOCK=block))
    if not bool((sums == 17 * block).all()):
        raise RuntimeError("read-only stream did not consume the full buffer")
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(size * 10 / (start.elapsed_time(end) * 1e6))
    return dict(buffer_bytes=size, gbps=max(values), repeats_gbps=values,
                checksum_bytes=sums.numel() * 4,
                convention="N global-load bytes; one checksum float store per 16 KiB read")


def finite_nonzero(tensor, label):
    bad = int((~torch.isfinite(tensor)).sum())
    if bad or not bool(torch.count_nonzero(tensor)):
        raise RuntimeError(f"{label} output must be finite and nonzero "
                           f"({bad} non-finite of {tensor.numel()})")


STAGES = {"ingress": ("_Group",), "fc1": ("fc1_kernel",),
          "activation": ("activation_kernel",), "fc2": ("fc2_kernel",),
          "egress": ("reduce_kernel",)}


def stage_profile(graph, directory, stages=STAGES):
    # Profiling is a separate replay: it never supplies the performance verdict.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", dir=directory) as stream:
        prof.export_chrome_trace(stream.name)
        events = json.loads(Path(stream.name).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    result, named = {}, set()
    for stage, needles in stages.items():
        matches = [e for e in kernels if any(n in e.get("name", "") for n in needles)
                   and id(e) not in named]
        named.update(id(e) for e in matches)
        result[stage+"_us"] = sum(e["dur"] for e in matches)
    result["other_us"] = sum(e["dur"] for e in kernels if id(e) not in named)
    span = max(e["ts"]+e["dur"] for e in kernels)-min(e["ts"] for e in kernels)
    result["launch_gaps_us"] = max(0.0, span-sum(e["dur"] for e in kernels))
    result["profiled_span_us"] = span
    result["kernels"] = len(kernels)
    return result


class _ControlPointers:
    """Master's K1 serving ABI: F16 rotated input, rotated F32 output."""

    def __init__(self, core):
        self.core = core.Exlr8A16Kernel(6144, 4, 8, 512, device_count=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, pointers: cute.Pointer, bits: cute.Pointer,
                 group_expert: cute.Pointer, route_slots: cute.Pointer,
                 route_group_lane: cute.Pointer, route_weights: cute.Pointer,
                 fc1: cute.Pointer, intermediate: cute.Pointer, fc2: cute.Pointer,
                 output: cute.Pointer, group_count: cute.Pointer,
                 rows: cutlass.Int32, cta_limit: cutlass.Int32, stream: cuda.CUstream):
        self.core.serving(cute.make_tensor(x, cute.make_layout(1, stride=1)),
                          cute.make_tensor(pointers, cute.make_layout(1, stride=1)),
                          cute.make_tensor(bits, cute.make_layout(1, stride=1)),
                          cute.make_tensor(group_expert, cute.make_layout(1, stride=1)),
                          cute.make_tensor(route_slots, cute.make_layout(1, stride=1)),
                          cute.make_tensor(route_group_lane, cute.make_layout(1, stride=1)),
                          cute.make_tensor(route_weights, cute.make_layout(1, stride=1)),
                          cute.make_tensor(fc1, cute.make_layout(1, stride=1)),
                          cute.make_tensor(intermediate, cute.make_layout(1, stride=1)),
                          cute.make_tensor(fc2, cute.make_layout(1, stride=1)),
                          cute.make_tensor(output, cute.make_layout(1, stride=1)), rows,
                          cute.make_tensor(group_count, cute.make_layout(1, stride=1)),
                          cta_limit, stream)


def load_source(label, path, package=None):
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location(f"exlr8_control_{label}", path)
    module = importlib.util.module_from_spec(spec)
    if package:
        module.__package__ = package
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def compile_control(path, group_path):
    """Compile master's K1 core and its grouping under bench-only, source-keyed
    names: the K1 reference whose own speed must not drop (no transforms)."""
    from b12x.integration.cuteafd import _common
    core, digest = load_source("core", path)
    glue, glue_digest = load_source("glue", group_path, "b12x.integration.cuteafd")
    O, S = _common.Operand, _common.Scalar
    if core.Exlr8A16Kernel.ABI_VERSION == 3:
        from b12x.integration.cuteafd.exlr8 import _Exlr8ServingPointers, geometry
        g = geometry(units=4)
        wrapper = _Exlr8ServingPointers(g)
        wrapper.core = core.Exlr8A16Kernel(6144, 4, 8, 512, device_count=True)
        names = ("x", "pointers", "bits", "group_expert", "route_slots", "route_group_lane",
                 "route_weights", "fc1", "intermediate", "fc2", "signs_out", "output", "group_count")
        dtypes = (torch.float16, torch.int64, torch.int32, torch.int32, torch.int32, torch.int32,
                  torch.float32, torch.float32, torch.float16, torch.float32, torch.int8,
                  torch.bfloat16, torch.int32)
        k1 = _common.compile_program(wrapper, name="exlr8_control_k1_v3",
            operands=tuple(O(n, t, "") for n, t in zip(names, dtypes)),
            scalars=(S("rows"), S("cta_limit")), key=(digest,))
        group = _common.compile_program(glue._Group(256, 8, 512, 6144, "fp8_k32"),
            name="exlr8_control_group_v2", operands=(O("ids", torch.int32, ""),
                O("x", torch.uint8, ""), O("signs_in", torch.int8, ""),
                O("group_expert", torch.int32, ""), O("route_slots", torch.int32, ""),
                O("route_group_lane", torch.int32, ""), O("group_count", torch.int32, ""),
                O("activation", torch.float16, "")),
            scalars=(S("rows"), S("cta_limit")), key=(glue_digest,))
        return k1, group
    k1 = _common.compile_program(_ControlPointers(core), name="exlr8_control_k1", operands=(
        O("x", torch.float16, ""), O("pointers", torch.int64, ""), O("bits", torch.int32, ""),
        O("group_expert", torch.int32, ""), O("route_slots", torch.int32, ""),
        O("route_group_lane", torch.int32, ""), O("route_weights", torch.float32, ""),
        O("fc1", torch.float32, ""), O("intermediate", torch.float16, ""),
        O("fc2", torch.float32, ""), O("output", torch.float32, ""),
        O("group_count", torch.int32, "")),
        scalars=(S("rows"), S("cta_limit")), key=(digest,))
    group = _common.compile_program(glue._Group(256, 8, 512), name="exlr8_control_group",
        operands=(O("ids", torch.int32, ""), O("group_expert", torch.int32, ""),
                  O("route_slots", torch.int32, ""), O("route_group_lane", torch.int32, ""),
                  O("group_count", torch.int32, "")),
        scalars=(S("rows"),), key=(glue_digest,))
    return k1, group


L2_SPAN = 2  # weight copies span at least this many times the device L2


def parse_grouped_arm(value):
    name, separator, revision = value.partition(":")
    if (not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            or not revision or revision.startswith("-") or ":" in revision
            or name in {"chunked", "grouped", "exl3", "mixed_unreachable_k5"}):
        raise argparse.ArgumentTypeError("grouped arm must be NAME:REV with a nonreserved name")
    return name, revision


K4_ARM_MODULES = ("host", "ws_a8", "kernel", "mixed_trellis", "mixed_trellis_ws")


def materialize_k4_arm(revision, directory, repository=None):
    """Freeze WS and its sibling implementation dependencies at one commit."""
    repository = repository or Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])
    sha = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    directory = Path(directory) / sha / "k4"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"revision": sha, "sources": {}}
    for name in K4_ARM_MODULES:
        source = git("show", f"{sha}:b12x/moe/_shared/kernels/w4a16/{name}.py")
        (directory / f"{name}.py").write_bytes(source)
        manifest["sources"][name] = hashlib.sha256(source).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def load_k4_arm(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest["sources"]) != set(K4_ARM_MODULES):
        raise ValueError("incomplete frozen K4 arm")
    for name, digest in manifest["sources"].items():
        if hashlib.sha256((directory / f"{name}.py").read_bytes()).hexdigest() != digest:
            raise ValueError(f"K4 arm digest mismatch: {name}")
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    package = f"b12x.moe._shared.kernels.k4_arm_{identity}"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        module.__package__ = package
        sys.modules[package] = module
    mixed = __import__(package + ".mixed_trellis", fromlist=["mixed_trellis"])
    return mixed, manifest


def k4_noise_summary(samples, baseline):
    medians = {name: statistics.median(values) for name, values in samples.items()}
    control = medians[baseline + "_control"]
    reference = medians[baseline]
    spread = abs(control - reference) / min(control, reference)
    return dict(medians_us=medians, identical_control_spread=spread,
                noise_pass=spread < 0.02,
                speedups={name: reference / value for name, value in medians.items()})


def materialize_grouped_arm(revision, directory, repository=None):
    """Pin both sources with git before exporting a metadata-free run tree."""
    repository = repository or Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])
    sha = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    paths = {"core": "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py",
             "glue": "b12x/integration/cuteafd/exlr8_glue.py",
             "wrapper": "b12x/integration/cuteafd/exlr8.py"}
    directory = Path(directory) / sha
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"revision": sha, "sources": {}}
    for label, path in paths.items():
        source = git("show", f"{sha}:{path}")
        (directory / f"{label}.py").write_bytes(source)
        manifest["sources"][label] = hashlib.sha256(source).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def grouped_arm_bundle(revision, directory):
    if directory is None:
        return materialize_grouped_arm(revision, Path(tempfile.mkdtemp(prefix="exlr8-arms-")))
    # Staged runs have no Git database; resolve only an unambiguous SHA prefix.
    if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        raise ValueError("staged grouped arms require a commit SHA")
    matches = [p for p in Path(directory).glob(revision + "*") if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous frozen grouped arm {revision}")
    return matches[0]


def compile_grouped_arm(directory, candidate):
    """Compile each revision's own ingress and grouped core with source keys."""
    from b12x.integration.cuteafd import _common
    from b12x.integration.cuteafd.exlr8 import _Exlr8ServingPointers
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["revision"] != directory.name:
        raise ValueError("frozen grouped revision mismatch")
    modules = {}
    for label, package in (("core", "b12x.moe._shared.kernels.w4a16"),
                           ("glue", "b12x.integration.cuteafd"),
                           ("wrapper", "b12x.integration.cuteafd")):
        path = directory / f"{label}.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sources"][label]:
            raise ValueError(f"frozen grouped {label} source digest mismatch")
        modules[label], _ = load_source("grouped_" + label, path, package)
    core, glue = modules["core"], modules["glue"]
    source_key = (manifest["revision"], tuple(sorted(manifest["sources"].items())))
    g = dict(candidate.geometry)
    compact = "unit_count" in (directory / "core.py").read_text()
    g["compact_units"] = compact
    g["compact_scratch"] = compact and "unit_limit.to(Int64) * Int64(self.group_rows" in (directory / "core.py").read_text()
    wrapper_type = (getattr(modules["wrapper"], "_Exlr8CompactPointers", None)
                    or getattr(modules["wrapper"], "_Exlr8GroupedServingPointers", None)
                    or modules["wrapper"]._Exlr8ServingPointers)
    wrapper = object.__new__(wrapper_type)
    wrapper.core = core.Exlr8GroupedKernel(g["hidden"], g["units"], g["top_k"],
                                          g["max_groups"], g["group_rows"])
    program = _common.compile_program(wrapper, name="exlr8_frozen_grouped",
        operands=tuple(o for o in candidate.operands
                       if compact or o.name not in ("unit_count", "unit_map")),
        scalars=tuple(v for v in candidate.scalars
                      if v.name != "l2_bytes" and (compact or v.name != "unit_limit")),
        key=(source_key, tuple(sorted(g.items()))), geometry=g, scratch=candidate.scratch)
    # Invoke the frozen factory, which selects its own grouping class.
    compile_program = glue.compile_program
    def keyed_group(wrapper, **kwargs):
        kwargs["name"] = "exlr8_frozen_group"
        kwargs["key"] = (source_key, kwargs["key"])
        return _common.compile_program(wrapper, **kwargs)
    glue.compile_program = keyed_group
    try:
        group = glue.compile_exlr8_group_aot(hidden=g["hidden"], top_k=g["top_k"],
            max_groups=g["max_groups"], max_rows=4096, group_rows=g["group_rows"])
    finally:
        glue.compile_program = compile_program
    return program, group, manifest


def rank_geometry(world, rank, layer, experts=256, units=16):
    """Serving's rotated EXLR8 ownership and contiguous H128 EXL3 slice."""
    if not 1 <= world <= units or not 0 <= rank < world or layer < 0:
        raise ValueError("invalid world, rank or layer")
    owned = [tuple(u for u in range(units)
                   if (units * e + u + layer) % world == rank) for e in range(experts)]
    base, extra = divmod(units, world)
    return owned, 128 * (base + (rank < extra))


def serving_k4_options(width):
    # GLM Spark m4096 package policy; WS supports unequal N tiles.
    tile = ((64, 192, 64, 256) if width % 256 == 128 and width % 192 == 0
            else (64, 256, 64, 256) if width % 256 == 0
            else (64, 128, 64, 128))
    return dict(force_tile_config=tile, moe_block_size=64, warp_specialized=True,
                input_format="e4m3_k32", ws_input_stages=3, ws_dynamic_tiles=True,
                force_blocks_per_sm=1)


def prepare_uniform(bits, experts, seed, width):
    if width == 512:
        return exl3._prepared(bits, experts, seed, torch.device("cuda"))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device="cuda")).half()
    tile = serving_k4_options(width)["force_tile_config"]
    return exl3.prepare_trellis256_moe_weights(
        hidden_size=6144, intermediate_size=width, num_experts=experts,
        activation="silu", fc1_tile_n=tile[1], fc2_tile_n=tile[3],
        device=torch.device("cuda"), seed=seed, params_dtype=torch.float16,
        w13_layout="trellis_t256_proj", trellis_bits=bits,
        gate_suh=scales((experts, 6144)), up_suh=scales((experts, 6144)),
        intermediate_rotations=scales((experts, 3 * width)),
        down_svh=scales((experts, 6144)),
        tile_config=tile)



def prepare_comparison_layers(copies, width, serving):
    # Only tier 0 is routed. Its cold copies stay distinct; unused K5 aliases.
    unused_k5 = prepare_uniform(5, 256, 409, width) if serving else None
    return [(prepare_uniform(4, 256, 408 + 2*c, width),
             unused_k5 if serving else prepare_uniform(5, 1, 409 + 2*c, width))
            for c in range(copies)]


def comparison_route_ids(rows, generator):
    # Global IDs 0..255 map to tier 0; K5 occupies unreachable IDs 256..511.
    return torch.stack([torch.randperm(256, generator=generator)[:8]
                        for _ in range(rows)])


def grouped_error(actual, expected, weights, *, half_partials):
    """Check default-route fidelity without claiming FP16 partials are exact."""
    exact = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    reference = expected.float()
    difference = actual.float() - reference
    error = torch.linalg.vector_norm(difference)
    reference_norm = torch.linalg.vector_norm(reference)
    row_relative = (torch.linalg.vector_norm(difference, dim=1)
                    / torch.linalg.vector_norm(reference, dim=1).clamp_min(1e-30))
    metrics = dict(bitwise_equal=exact,
        relative_l2=float(error / reference_norm.clamp_min(1e-30)),
        max_row_relative_l2=float(row_relative.max()),
        mean_row_relative_l2=float(row_relative.mean()),
        max_abs=float(difference.abs().max()))
    if half_partials:
        # Match the serving fixture: a 1% normal-range guard plus half an
        # FP16 subnormal ULP per route element; H128 preserves the L2 norm.
        subnormal_row = (2.0 ** -25 * actual.shape[-1] ** 0.5
                         * weights.float().abs().sum(-1))
        bound = 0.01 * reference_norm + torch.linalg.vector_norm(subnormal_row)
        metrics["l2_bound"] = float(bound)
        valid = bool(torch.isfinite(actual).all() and error <= bound)
    else:
        valid = exact and bool(torch.isfinite(actual).all())
    if not valid:
        raise RuntimeError(f"grouped result differs from snapshot chunked K1: {metrics}")
    return metrics


SERVING_LANES = 3        # CUTEAFD_GLM_PREFILL_LANES default (cuteafd glm5/engine.rs)
SERVING_LANE_ROWS = 4096  # GLM prefill rows per lane; a chunk is lanes x this
SERVING_PAGE_ROWS = 64
SERVING_MIN_LANE_ROWS = 256


def serving_waves(prompt, lanes=SERVING_LANES, lane_rows=SERVING_LANE_ROWS):
    """Spark wave row counts GLM serving sends for one cold prompt.

    Mirrors the engine: chunks of lanes * lane_rows tokens; a chunk of at
    least 2 * 256 rows splits into min(lanes, rows // 256) lanes of
    ceil(rows / count) rounded up to a 64-row page. An 8046-token prompt is
    one chunk of 2688 + 2688 + 2670 rows, not 1408-row waves.
    """
    if prompt < 1 or lanes < 1:
        raise ValueError("prompt and lanes must be positive")
    waves = []
    for start in range(0, prompt, lanes * lane_rows):
        rows = min(lanes * lane_rows, prompt - start)
        if lanes > 1 and rows >= 2 * SERVING_MIN_LANE_ROWS:
            count = min(lanes, rows // SERVING_MIN_LANE_ROWS)
            per = -(-(-(-rows // count)) // SERVING_PAGE_ROWS) * SERVING_PAGE_ROWS
            waves.extend(min(per, rows - first) for first in range(0, rows, per))
        else:
            waves.append(rows)
    return waves


def real_route_ids(rows, wave, generator, experts=256, top_k=8):
    """Top-k ids with a recorded serving wave's per-expert histogram, rescaled
    to `rows` (largest-remainder rounding, at most `rows` per expert), then
    assigned so no token repeats an expert. Deterministic per generator."""
    counts = torch.zeros(experts, dtype=torch.float64)
    for expert, n in wave["expert_rows"]:
        counts[expert] = n
    target = counts * (rows * top_k / counts.sum())
    alloc = target.floor().clamp(max=rows)
    order = torch.argsort(target - alloc, descending=True)
    for e in order.tolist():
        if int(alloc.sum()) >= rows * top_k:
            break
        if alloc[e] < rows:
            alloc[e] += 1
    alloc = alloc.long()
    if int(alloc.sum()) != rows * top_k:
        raise ValueError("route histogram cannot fill the wave")
    # Fill tokens largest expert first, each expert onto the tokens with the
    # fewest assigned routes, so every token ends with top_k distinct experts.
    ids = torch.full((rows, top_k), -1, dtype=torch.int64)
    fill = torch.zeros(rows, dtype=torch.int64)
    for e in torch.argsort(alloc, descending=True).tolist():
        n = int(alloc[e])
        if not n:
            break
        jitter = torch.rand(rows, generator=generator)
        pick = torch.argsort(fill.double() + jitter)[:n]
        ids[pick, fill[pick]] = e
        fill[pick] += 1
    if bool((fill != top_k).any()):
        raise ValueError("route assignment left a token short")
    perm = torch.argsort(torch.rand((rows, top_k), generator=generator), dim=1)
    return torch.gather(ids, 1, perm)


ROUTE_FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Realistic content (code, agentic transcript, prose) is the default; filler
# (the bench's synthetic prefill prompt) is one case, not the reference.
ROUTE_SETS = {"realistic": ("glm53_prefill_routes_realistic.json",),
              "filler": ("glm53_prefill_routes_filler.json",)}


def load_route_waves(spec):
    """Route waves from a named set (realistic, filler) or a JSON path."""
    paths = ([ROUTE_FIXTURES / name for name in ROUTE_SETS[spec]] if spec in ROUTE_SETS
             else [Path(spec)])
    waves = []
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("schema") != 1 or not data.get("waves"):
            raise ValueError(f"{path}: expected schema 1 with waves")
        waves.extend(data["waves"])
    return waves


def serial_scratch(hidden, units, groups, shared=None):
    shapes = {"fc1": ((groups, units, 2, (hidden+1023)//1024, 16, 128), torch.float32),
              "intermediate": ((groups, units, 16, 128), torch.float16),
              "fc2": ((groups, 16, hidden), torch.float32)}
    if shared is None:
        return {name: torch.empty(shape, dtype=dtype, device="cuda")
                for name, (shape, dtype) in shapes.items()}
    for name, (shape, dtype) in shapes.items():
        count = 1
        for extent in shape:
            count *= extent
        tensor = shared[name]
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() < count:
            raise ValueError(f"shared {name} scratch cannot hold the serial arm")
    # Arms and graph replays run sequentially on one stream; only raw pointers
    # cross the kernel ABI, so the larger grouped shape needs no serial view.
    return {name: shared[name] for name in shapes}


def make_candidate(l2_bytes, experts=256, hidden=6144, units=4, topk=8, owned=None,
                   scratch=None, min_copies=2):
    """Synthetic K4 payloads, replicated into enough distinct layers that a
    replay over them streams every weight from DRAM (cold L2)."""
    owners = []
    unit_bytes = hidden * 16 * 4
    layer_bytes = (experts * units if owned is None else sum(map(len, owned))) * 3 * unit_bytes
    copies = max(min_copies, -(-L2_SPAN * l2_bytes // layer_bytes))
    tables = []
    for _ in range(copies):
        pointers = torch.zeros((experts, units, 3, 3), dtype=torch.int64)
        for e in range(experts):
            for u in (range(units) if owned is None else owned[e]):
                for p in range(3):
                    tensors = (torch.randint(0, 256, (unit_bytes,), dtype=torch.uint8, device="cuda"),
                               torch.ones(128, dtype=torch.float16, device="cuda"),
                               torch.full((hidden//128,), 1 / hidden**0.5, dtype=torch.float16, device="cuda"))
                    owners.extend(tensors)
                    pointers[e,u,p] = torch.tensor([t.data_ptr() for t in tensors])
        tables.append(pointers.to("cuda"))
    gen = torch.Generator().manual_seed(4080)
    signs = [(torch.randint(2, (hidden,), generator=gen) * 2 - 1).char().cuda() for _ in range(2)]
    bits = torch.zeros((experts, units, 3), dtype=torch.int32)
    for e in range(experts):
        bits[e, list(range(units) if owned is None else owned[e]), :] = 4
    groups = 64*topk
    return dict(tables=tables, owners=owners, signs_in=signs[0], signs_out=signs[1],
                bits=bits.cuda(),
                group_expert=torch.full((groups,),-1,dtype=torch.int32,device="cuda"),
                route_slots=torch.full((groups,16),-1,dtype=torch.int32,device="cuda"),
                route_group_lane=torch.full((64,topk),-1,dtype=torch.int32,device="cuda"),
                group_count=torch.zeros(1,dtype=torch.int32,device="cuda"),
                activation=torch.empty((64,hidden),dtype=torch.float16,device="cuda"),
                **serial_scratch(hidden, units, groups, scratch),
                output=torch.empty((64,hidden),dtype=torch.bfloat16,device="cuda"),
                control_output=torch.empty((64,hidden),device="cuda"),
                control_x=torch.empty((64,hidden),dtype=torch.float16,device="cuda"))


def exl3_state(rows, tier, expert_map, props):
    """EXL3 K4 at one decode capacity with the Spark worker's BF16 top-k sum."""
    buffers, launch, _, expert_map = exl3._serial_state(rows, tier, expert_map, props,
                                                         torch.device("cuda"))
    buffers = dataclasses.replace(
        buffers, output=torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda"))
    topk_sum = exl3.compile_w4a16_topk_sum(
        m=rows, topk=8, hidden_size=6144, element_dtype="fp16", full_rotation=True,
        full_rotation_output_dtype="bf16", num_experts=int(tier.num_experts),
        route_num_experts=exl3.TOTAL_EXPERTS, route_ids_dtype=torch.int32, use_expert_map=True)
    return tier, buffers, launch, topk_sum, expert_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped", action="store_true", help="matched grouped prefill gate")
    parser.add_argument("--group-rows", type=int, default=64)
    parser.add_argument("--world", type=int, default=None,
                        help="explicit serving geometry; omitted keeps legacy compact TP4 fixture")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows",default=None)
    parser.add_argument("--replays",type=int,default=20)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--control-core",type=Path,
                        help="master's K1 core source (b12x/moe/_shared/kernels/w4a16/exlr8.py)")
    parser.add_argument("--control-glue",type=Path,
                        help="master's grouping source (b12x/integration/cuteafd/exlr8_glue.py)")
    parser.add_argument("--k4-arm", action="append", type=parse_grouped_arm, default=[],
                        help="NAME:REV frozen EXL3 WS arm; K4-only comparison, first arm is reference")
    parser.add_argument("--k4-exhaustive", action="store_true",
                        help="gate every live row count 1..4096 and exact prefix row invariance before timing")
    parser.add_argument("--grouped-arm", action="append", type=parse_grouped_arm, default=[],
                        help="repeatable NAME:REV frozen grouped core and ingress")
    parser.add_argument("--approximate-arm", action="append", default=[],
                        help="named grouped arm allowed non-bitwise output; report error, still require finite output")
    parser.add_argument("--grouped-arm-source-dir", type=Path,
                        help="pre-materialized git-source bundles for metadata-free runs")
    parser.add_argument("--repeats",type=int,default=1,
                        help="interleaved timing rounds; arm order rotates each round, medians reported")
    parser.add_argument("--profile",action="store_true",
                        help="one eager launch per arm inside cudaProfilerStart/Stop (for ncu); no timing")
    parser.add_argument("--profile-arm", default="grouped",
                        help="grouped benchmark arm inside cudaProfilerStart/Stop")
    parser.add_argument("--serving-prompt", default=None,
                        help="comma list of prompt lengths; benchmark the distinct Spark wave "
                             "row counts GLM serving sends for them (overrides --rows)")
    parser.add_argument("--serving-lanes", type=int, default=SERVING_LANES)
    parser.add_argument("--memory-cap-gib", type=int, default=12,
                        help="per-process CUDA cap (also at most 25%% of device memory); more "
                             "--layer-copies need more")
    parser.add_argument("--layer-copies", type=int, default=2,
                        help="minimum cold layer copies per replay; with --routes each copy "
                             "replays a different recorded wave (6 covers two waves per content)")
    parser.add_argument("--routes", default=None,
                        help="recorded serving route histograms: 'realistic' (code, agentic, prose; "
                             "the reference), 'filler' (bench filler prompt) or a JSON path; each "
                             "timed layer copy replays the next wave")
    parser.add_argument("--serving", action="store_true",
                        help="<=64-row EXLR8 Spark serving duel benchmark (TP4/TP6 rotated ownership)")
    parser.add_argument("--tier-mix", choices=("mixed", "k4"), default="mixed",
                        help="EXLR8 tier distribution for --serving: 'mixed' ((2,3,4)[(u+p)%%3]) or 'k4'")
    parser.add_argument("--serving-variants", default="k1_k23_coalesced",
                        help="comma-separated <=64 serving variant arms to benchmark against baseline")
    args = parser.parse_args()
    if args.serving:
        if args.grouped or args.k4_arm or args.serving_prompt is not None:
            parser.error("--serving cannot be combined with --grouped, --k4-arm or --serving-prompt")
        if args.world is None:
            args.world = 6
        try:
            rank_geometry(args.world, args.rank, args.layer)
        except ValueError as exc:
            parser.error(str(exc))
        if args.rows is None:
            args.rows = "1,4,8,16,32,64"
        if not args.profile and args.output is None:
            parser.error("--output is required unless --profile")
        if torch.cuda.get_device_capability() != (12, 1):
            parser.error("this performance gate requires SM121")
        torch.cuda.set_per_process_memory_fraction(min(
            0.25, (args.memory_cap_gib << 30) / torch.cuda.get_device_properties(0).total_memory))
        return serving_main(args)
    if args.serving_prompt is not None:
        if not args.grouped:
            parser.error("--serving-prompt requires --grouped")
        rows = []
        for prompt in map(int, args.serving_prompt.split(",")):
            for wave in serving_waves(prompt, args.serving_lanes):
                if wave not in rows:
                    rows.append(wave)
        args.rows = ",".join(map(str, rows))
        print(f"serving waves for prompts {args.serving_prompt}: rows {args.rows}", flush=True)
    if args.routes is not None and not args.grouped:
        parser.error("--routes requires --grouped")
    if args.rows is None:
        args.rows = "512,1408,2048,4096" if args.grouped else "1,2,3,4,5,6,7,8,16,64,4096"
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile")
    if bool(args.control_core) != bool(args.control_glue):
        parser.error("--control-core and --control-glue go together")
    if set(args.approximate_arm) - {name for name, _ in args.grouped_arm}:
        parser.error("--approximate-arm must name a --grouped-arm")
    if args.grouped_arm and not args.grouped:
        parser.error("--grouped-arm requires --grouped")
    if len({name for name, _ in args.grouped_arm}) != len(args.grouped_arm):
        parser.error("--grouped-arm names must be unique")
    if args.profile_arm not in {"grouped", "exl3", "mixed", *(name for name, _ in args.grouped_arm)}:
        parser.error("--profile-arm must name an existing benchmark arm")
    if args.world is not None:
        if not args.grouped:
            parser.error("--world requires --grouped")
        try:
            rank_geometry(args.world, args.rank, args.layer)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.rank != 0 or args.layer != 3:
        parser.error("--rank/--layer require --world")
    if args.k4_arm:
        names = [name for name, _ in args.k4_arm]
        if (not args.grouped or args.world is None or args.grouped_arm or args.profile
                or len(names) < 2 or len(set(names)) != len(names)
                or any(name.endswith("_control") for name in names)):
            parser.error("--k4-arm requires --grouped --world, >=2 unique arms, no *_control names or grouped/profile arms")
        if args.repeats < 3:
            parser.error("K4 timing requires at least three interleaved repeats")
    elif args.k4_exhaustive:
        parser.error("--k4-exhaustive requires --k4-arm")
    if torch.cuda.get_device_capability() != (12,1):
        parser.error("this performance gate requires SM121")
    torch.cuda.set_per_process_memory_fraction(min(
        0.25, (args.memory_cap_gib << 30) / torch.cuda.get_device_properties(0).total_memory))
    if args.k4_arm:
        return k4_main(args)
    if args.grouped:
        return grouped_main(args)
    props = torch.cuda.get_device_properties(0)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 24 << 20)
    if not args.profile:
        args.output.parent.mkdir(parents=True,exist_ok=True)
    read_only = None if args.profile else read_roofline()
    t = make_candidate(l2_bytes)
    copies = len(t["tables"])
    group = compile_exlr8_group_aot(top_k=8, max_groups=512, hidden=6144, input_format="fp8_k32")
    k1 = compile_exlr8_a16_aot(units=4, device_count=True)
    control = compile_control(args.control_core, args.control_glue) if args.control_core else None
    # This is the existing EXL3 uniform-K4 implementation, not a proxy GEMM.
    # Each EXL3 K4 layer holds 256 experts x 3 x 6144 x 512 x 4 bits (1.2 GB),
    # far above 2x L2; two copies alternate like the EXLR8 tables.
    exl3_copies = [exl3._prepared(4,256,408+c,torch.device("cuda")) for c in range(copies)]
    tier = exl3_copies[0]
    # The legacy helper predates the compiler's codebook default change.
    exl3.compile_w4a16_fused_moe = functools.partial(
        exl3.compile_w4a16_fused_moe, trellis_codebook=tier.trellis_codebook)
    expert_map = torch.arange(256,dtype=torch.int32,device="cuda")
    cta_limit = props.multi_processor_count*k1.geometry["ctas_per_sm"]
    records = []
    for total in map(int,args.rows.split(",")):
        gen = torch.Generator().manual_seed(408+total)
        chunks = [min(64, total-first) for first in range(0, total, 64)]
        ids = comparison_route_ids(total, gen).cuda().int()
        weights = torch.rand((total,8),generator=gen).cuda()/8
        # Master's amplitude: EXL3's FP16 intermediates overflow on random K4
        # payloads at unit-scale activations.
        x = torch.randn((total,6144),generator=gen).cuda()*0.02
        wire = make_wire(x)
        # One graph per arm; chunks replay serially through 64-row owned
        # buffers, layers rotate through distinct weight copies (cold L2).
        wire_slot = torch.empty((64,wire.shape[1]),dtype=torch.uint8,device="cuda")
        ids_slot = torch.empty((64,8),dtype=torch.int32,device="cuda")
        weights_slot = torch.empty((64,8),device="cuda")
        response = torch.empty((total,6144),dtype=torch.bfloat16,device="cuda")
        layers = max(1, min(copies, -(-2*copies // len(chunks))))

        def candidate_chunk(first, rows, layer):
            wire_slot[:rows].copy_(wire[first:first+rows])
            ids_slot[:rows].copy_(ids[first:first+rows])
            weights_slot[:rows].copy_(weights[first:first+rows])
            group.launch(ids_slot, wire_slot, t["signs_in"], t["group_expert"], t["route_slots"],
                         t["route_group_lane"], t["group_count"], t["activation"],
                         scalars=(rows, cta_limit))
            k1.launch(t["activation"], t["tables"][layer], t["bits"], t["group_expert"],
                      t["route_slots"], t["route_group_lane"], weights_slot, t["fc1"],
                      t["intermediate"], t["fc2"], t["signs_out"], t["output"],
                      t["group_count"], scalars=(rows, cta_limit))
            response[first:first+rows].copy_(t["output"][:rows])

        def candidate():
            for layer in range(layers):
                first = 0
                for rows in chunks:
                    candidate_chunk(first, rows, (layer*len(chunks)+first//64) % copies)
                    first += rows

        def control_chunk(first, rows, layer):
            ids_slot[:rows].copy_(ids[first:first+rows])
            weights_slot[:rows].copy_(weights[first:first+rows])
            k1c, groupc = control
            if len(k1c.operands) == 13:
                wire_slot[:rows].copy_(wire[first:first+rows])
                groupc.launch(ids_slot, wire_slot, t["signs_in"], t["group_expert"], t["route_slots"],
                              t["route_group_lane"], t["group_count"], t["activation"],
                              scalars=(rows, cta_limit))
                k1c.launch(t["activation"], t["tables"][layer], t["bits"], t["group_expert"],
                           t["route_slots"], t["route_group_lane"], weights_slot, t["fc1"],
                           t["intermediate"], t["fc2"], t["signs_out"], t["output"],
                           t["group_count"], scalars=(rows, cta_limit))
                response[first:first+rows].copy_(t["output"][:rows])
            else:
                t["control_x"][:rows].copy_(x[first:first+rows])
                groupc.launch(ids_slot, t["group_expert"], t["route_slots"], t["route_group_lane"],
                              t["group_count"], scalars=(rows,))
                k1c.launch(t["control_x"], t["tables"][layer], t["bits"], t["group_expert"],
                           t["route_slots"], t["route_group_lane"], weights_slot, t["fc1"],
                           t["intermediate"], t["fc2"], t["control_output"], t["group_count"],
                           scalars=(rows, cta_limit))

        def control_run():
            for layer in range(layers):
                first = 0
                for rows in chunks:
                    control_chunk(first, rows, (layer*len(chunks)+first//64) % copies)
                    first += rows

        # EXL3 K4 at the worker's real decode capacities: one launch per chunk.
        # EXL3 runs its m4096 warp-specialized export for prefill; only decode
        # capacities (<= 64 rows) are compared against it.
        states = ({rows: [exl3_state(rows, c, expert_map, props) for c in exl3_copies]
                   for rows in set(chunks)} if total <= 64 else {})
        decoded = torch.empty((64,6144),dtype=torch.bfloat16,device="cuda")

        def baseline():
            for layer in range(layers):
                first = 0
                for rows in chunks:
                    state = states[rows][(layer*len(chunks)+first//64) % len(exl3_copies)]
                    wire_slot[:rows].copy_(wire[first:first+rows])
                    ids_slot[:rows].copy_(ids[first:first+rows])
                    weights_slot[:rows].copy_(weights[first:first+rows])
                    wire_decode(wire_slot, decoded, rows)
                    out = exl3._run_serial(state, decoded[:rows], weights_slot[:rows], ids_slot[:rows])
                    response[first:first+rows].copy_(out)
                    first += rows

        if args.profile:
            for _ in range(3):
                baseline(); candidate()
                if control: control_run()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            baseline(); candidate()
            if control: control_run()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled rows={total}", flush=True)
            continue
        candidate_graph = capture(candidate)
        candidate_graph.replay()
        finite_nonzero(response.float(), "EXLR8")
        graphs = {"candidate": candidate_graph}
        if states:
            graphs["baseline"] = capture(baseline)
            graphs["baseline"].replay()
            finite_nonzero(response.float(), "EXL3")
        if control:
            graphs["control"] = capture(control_run)
        names = list(graphs)
        rounds = {n: [] for n in names}
        for r in range(args.repeats):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                rounds[n].append(timing(graphs[n], args.replays)/layers)
        med = {n: statistics.median(v) for n, v in rounds.items()}
        baseline_us = med.get("baseline")
        candidate_us = med["candidate"]
        unique = sum(int(ids[f:f+r].unique().numel()) for f, r in zip(range(0,total,64), chunks))
        payload_bytes = unique*3*6144*512*4//8
        stages = stage_profile(candidate_graph,args.output.parent)
        stages = {k: (v/layers if k.endswith("_us") else v) for k, v in stages.items()}
        record = dict(rows=total, chunks=len(chunks), layer_copies=copies, payload_bytes=payload_bytes,
                      exlr8_step_us=candidate_us, exlr8_gbps=payload_bytes/candidate_us/1000,
                      read_roofline_percent=100*payload_bytes/candidate_us/1000/read_only["gbps"],
                      stages=stages,
                      transform_share_percent=100*(stages["ingress_us"]+stages["egress_us"])/candidate_us)
        if baseline_us is not None:
            record.update(exl3_k4_us=baseline_us, speedup=baseline_us/candidate_us,
                          exl3_gbps=payload_bytes/baseline_us/1000,
                          exl3_read_roofline_percent=100*payload_bytes/baseline_us/1000/read_only["gbps"],
                          target=0.97, target_met=baseline_us/candidate_us>=0.97)
        record["rounds_us"] = rounds
        if control:
            control_graph = graphs["control"]
            control_us = med["control"]
            cstages = stage_profile(control_graph,args.output.parent)
            record.update(control_us=control_us, candidate_over_control=control_us/candidate_us,
                          control_stages={k: (v/layers if k.endswith("_us") else v) for k, v in cstages.items()})
        records.append(record)
        print(json.dumps(record), flush=True)
        del states
    if args.profile:
        return
    result = dict(schema=2,kind="exlr8_spark_step_vs_exl3_k4",device=props.name,
                  sm_count=props.multi_processor_count,l2_bytes=l2_bytes,
                  hidden=6144,units_per_rank=4,top_k=8,experts=256,ingress="fp8_k32",egress="bf16",
                  measured_read_roofline=read_only,
                  conditions=("TP4 synthetic K4 payloads, cold L2 (distinct weight copies >= 2x L2), "
                              "graph replay, one untimed warm replay; serial 64-row chunks; profiler separate"),
                  rows=records,target_met=all(r.get("target_met", True) for r in records))
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2)


def grouped_mixed_state(rows, tiers, props, width=512, serving=False, workspace=None, mix=None):
    if mix is None:
        from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mix
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
    tile = _projection_mixed_tile_config(None, hidden_size=6144,
        intermediate_size=width, token_count=4096, direct_topk_routes=False)
    options = (serving_k4_options(width) if serving else
               dict(force_tile_config=tile, moe_block_size=32, warp_specialized=True,
                    input_format="e4m3_k32", ws_input_stages=3, ws_dynamic_tiles=True))
    block = options["moe_block_size"]
    tier1_experts = 256 if serving else 1
    launch = mix.compile_mixed_trellis(
        size_m=4096, hidden_size=6144, intermediate_size=width,
        tier0_num_experts=256, tier1_num_experts=tier1_experts, tier0_bits=4, tier1_bits=5,
        route_num_experts=256,
        top_k=8, max_m_blocks=(max_packed_route_slots(4096*8, block, 256)+block-1)//block,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        trellis_codebook=tiers[0].trellis_codebook, **options,
        rotation_input_dtype="bf16", full_rotation_output_dtype="bf16")
    bufs = (mix.make_mixed_trellis_buffers(launch, device=torch.device("cuda"),
                                         sms=props.multi_processor_count)
            if workspace is None else workspace)
    # Every route is K4. Explicit geometry retains the production K5 tier
    # capacity even though no request in this synthetic K4 arm reaches it.
    mapping, desc = mix.build_ordered_maps(256, tier1_experts, device=torch.device("cuda"))
    mapping = mapping[:256]
    rotations = mix.MixedTrellisRotations(
        torch.cat([t.intermediate_rotations for t in tiers]),
        torch.cat([t.gate_suh for t in tiers]), torch.cat([t.up_suh for t in tiers]),
        torch.cat([t.down_svh for t in tiers]))
    binding = mix.bind_mixed_trellis(*tiers, mapping, desc, rotations, launch)
    return binding, bufs



def k4_main(args):
    """Same-session frozen WS comparison; synthetic K4, recorded GLM routes."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    props = torch.cuda.get_device_properties(0)
    _, width = rank_geometry(args.world, args.rank, args.layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layers = prepare_comparison_layers(args.layer_copies, width, True)
    modules, states, provenance = {}, {}, {}
    for name, revision in args.k4_arm:
        directory = (materialize_k4_arm(revision, Path(tempfile.mkdtemp(prefix="k4-arms-")))
                     if args.grouped_arm_source_dir is None else
                     grouped_arm_bundle(revision, args.grouped_arm_source_dir) / "k4")
        modules[name], provenance[name] = load_k4_arm(directory)
        states[name] = []
        workspace = None
        for tiers in layers:
            state = grouped_mixed_state(4096, tiers, props, width, serving=True,
                                        workspace=workspace, mix=modules[name])
            states[name].append(state)
            workspace = state[1]
    torch.cuda.empty_cache()
    waves = load_route_waves(args.routes or "realistic")
    generator = torch.Generator().manual_seed(408)
    ids = [real_route_ids(4096, waves[i % len(waves)], generator).cuda().int()
           for i in range(len(layers))]
    weights = torch.rand((4096, 8), generator=generator).cuda() / 8
    wire = make_wire(torch.randn((4096, 6144), generator=generator).cuda() * 0.02)
    names = list(modules)
    reference = names[0]

    def run(name, rows, layer=0):
        binding, workspace = states[name][layer]
        return modules[name].run_bound_mixed_trellis(
            wire[:rows], weights[:rows], ids[layer][:rows], binding, workspace)

    # Resolve route packing before the frozen-resolution gate and any capture.
    for name in names:
        binding, workspace = states[name][0]
        modules[name].warmup_mixed_trellis_route_pack(
            binding.launch, workspace, expert_map=binding.global_to_combined)
        for rows in (1, 17, 4096):
            run(name, rows)
    checked = range(1, 4097) if args.k4_exhaustive else sorted(
        {1, 17, 63, 64, 65, 4096, *map(int, args.rows.split(","))})
    with kernel_resolution_guard("EXL3 WS frozen-arm row invariance"):
        full = run(reference, 4096).clone()
        finite_nonzero(full, reference)
        for rows in checked:
            expected = run(reference, rows).clone()
            if not torch.equal(expected.view(torch.int16), full[:rows].view(torch.int16)):
                raise RuntimeError(f"reference row invariance failed at {rows}")
            for name in names[1:]:
                actual = run(name, rows)
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError(f"K4 arm {name} differs at {rows} rows")
    records = []
    for rows in map(int, args.rows.split(",")):
        if not 1 <= rows <= 4096:
            raise ValueError("K4 rows must be 1..4096")
        def layer_step(name):
            for layer in range(len(layers)):
                run(name, rows, layer)
        # Check every timed cold copy too, not only the exhaustive fixture.
        for layer in range(len(layers)):
            expected = run(reference, rows, layer).clone()
            for name in names[1:]:
                if not torch.equal(run(name, rows, layer).view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError(f"K4 arm {name} differs on timed layer {layer}, rows {rows}")
        graphs = {name: capture(functools.partial(layer_step, name)) for name in names}
        graphs[reference + "_control"] = capture(functools.partial(layer_step, reference))
        samples = {name: [] for name in graphs}
        order = list(graphs)
        for repeat in range(args.repeats):
            rotated = order[repeat % len(order):] + order[:repeat % len(order)]
            if repeat % 2:
                rotated.reverse()
            for name in rotated:
                samples[name].append(timing(graphs[name], args.replays) / len(layers))
        record = dict(rows=rows, samples_us=samples, **k4_noise_summary(samples, reference))
        records.append(record)
        print(json.dumps(record), flush=True)
    result = dict(schema=1, kind="exl3_ws_revision_comparison", arms=provenance,
                  fixture="synthetic K4 weights, recorded GLM layer routes",
                  world=args.world, rank=args.rank, layer=args.layer, width=width,
                  layer_copies=len(layers), device=props.name,
                  exhaustive=args.k4_exhaustive, checked_rows=len(checked), rows=records)
    args.output.write_text(json.dumps(result, indent=2))
    if not all(record["noise_pass"] for record in records):
        raise RuntimeError("K4 identical-control timing spread exceeds 2%; no performance verdict")


def serving_k4_states(rows, layers, props, width):
    # Serial cold layers share one stream workspace, as in serving. Bindings
    # retain distinct K4 pointers; captured work finishes before scratch reuse.
    states = []
    workspace = None
    for tiers in layers:
        state = grouped_mixed_state(rows, tiers, props, width, serving=True,
                                    workspace=workspace)
        states.append(state)
        workspace = state[1]
    return states


def grouped_uniform_state(rows, tier, props, width=512):
    from b12x.moe._shared.kernels.w4a16.host import (
        make_w4a16_packed_buffers, select_route_block_size_m)
    # Native tiles do not depend on launch K/N geometry. Share their storage,
    # but retain the uniform serving route's historical K64/N256 policy.
    tile = (64, 256 if width % 256 == 0 else 128, 64, 256)
    tier = dataclasses.replace(tier, fc1_tile_n=tile[1], fc2_tile_n=256,
                              trellis=dataclasses.replace(tier.trellis, tile_config=tile))
    block = select_route_block_size_m(rows, 8, 256)
    bufs = make_w4a16_packed_buffers(tier, m=rows, topk=8, dtype=torch.float16,
        device=torch.device("cuda"), full_rotation=True, block_size_m=block)
    bufs = dataclasses.replace(bufs,
        output=torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda"))
    launch = exl3.compile_w4a16_fused_moe(
        size_m=rows, hidden_size=6144, intermediate_size=width, num_experts=256,
        top_k=8, activation="silu", apply_router_weight_on_input=False,
        zero_fc2_output=False, moe_block_size=block,
        max_m_blocks=int(bufs.block_expert_ids.numel()), element_dtype="fp16",
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        weight_layout="trellis_t256", scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj", trellis_bits=4,
        trellis_codebook=tier.trellis_codebook, force_tile_config=tile,
        intermediate_rotation=True, full_rotation=True, rotation_input_dtype="bf16")
    topk_sum = exl3.compile_w4a16_topk_sum(m=rows, topk=8, hidden_size=6144,
        element_dtype="fp16", full_rotation=True, full_rotation_output_dtype="bf16",
        num_experts=256, route_ids_dtype=torch.int32)
    return tier, bufs, launch, topk_sum, block


def run_grouped_uniform(state, decoded, weights, ids):
    tier, bufs, launch, topk_sum, block = state
    return exl3.run_w4a16_moe(decoded, tier, weights, ids, activation="silu",
        intermediate_cache13=bufs.intermediate_cache13, intermediate_cache2=bufs.intermediate_cache2,
        output=bufs.output, fc1_c_tmp=bufs.fc1_c_tmp, fc2_c_tmp=bufs.fc2_c_tmp,
        packed_route_indices=bufs.packed_route_indices, block_expert_ids=bufs.block_expert_ids,
        packed_route_count=bufs.packed_route_count, expert_offsets=bufs.expert_offsets,
        expert_counts=bufs.expert_counts, fused_launch=launch, topk_sum_launch=topk_sum,
        route_block_size_m=block, route_mode="packed", full_rotation=True,
        intermediate_rotation_scales=tier.intermediate_rotations,
        suh_gate_table=tier.gate_suh, suh_up_table=tier.up_suh, svh_table=tier.down_svh,
        rotation_a_gate=bufs.rotation_a_gate, rotation_a_up=bufs.rotation_a_up)


def grouped_main(args):
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import run_bound_mixed_trellis
    from tests.moe.test_exlr8_grouped import buffers
    props = torch.cuda.get_device_properties(0)
    l2 = int(props.L2_cache_size)
    if l2 <= 0:
        raise RuntimeError("runtime device L2 size is required")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    roof = None if args.profile else read_roofline()
    owned, width = (rank_geometry(args.world, args.rank, args.layer)
                    if args.world is not None else (None, 512))
    units = 16 if owned is not None else 4
    group_rows = args.group_rows
    capacity = (4096*8 + group_rows-1)//group_rows + 256
    t = make_candidate(l2, units=units, owned=owned, min_copies=args.layer_copies)
    control = ((compile_exlr8_a16_aot(units=units, device_count=True),
                compile_exlr8_group_aot(hidden=6144, top_k=8, max_groups=512))
               if owned is not None else compile_control(args.control_core, args.control_glue))
    limit = int(props.multi_processor_count) * 4
    group = compile_exlr8_group_aot(hidden=6144, top_k=8, max_groups=capacity,
                                    max_rows=4096, group_rows=group_rows)
    grouped = compile_exlr8_a16_aot(units=units, top_k=8, max_groups=capacity,
                                    device_count=True, grouped=True, group_rows=group_rows)
    serving_limit = int(props.multi_processor_count) * int(grouped.geometry.get("ctas_per_sm", 2))
    frozen = {name: compile_grouped_arm(grouped_arm_bundle(revision,
                  args.grouped_arm_source_dir), grouped)
              for name, revision in args.grouped_arm}
    # Legacy keeps the historical uniform baseline. Explicit geometry uses
    # the sealed GLM serving package ABI, with every synthetic route in K4.
    exl_layers = prepare_comparison_layers(len(t["tables"]), width, owned is not None)
    # Weight preparation has large transient integer tensors. Allocate grouped
    # arenas only afterward, replacing rather than retaining serial scratch.
    for name in ("fc1", "intermediate", "fc2"):
        del t[name]
    torch.cuda.empty_cache()
    state = buffers(4096, 6144, units, 8, capacity, group_rows)
    t.update(serial_scratch(6144, units, 512, state))
    compact = compact_unit_descriptors(t["bits"])
    mixed_states = ([grouped_mixed_state(4096, ts, props) for ts in exl_layers]
                    if owned is None else [])
    exchange_stream = torch.cuda.Stream(priority=-1)
    exch_pulses = 12
    exch_slice_floats = max(65536, l2 // (4 * exch_pulses))
    exch_ring_src = torch.full((exch_pulses, exch_slice_floats), 1.0,
                               dtype=torch.float32, device="cuda")
    exch_ring_dst = torch.empty_like(exch_ring_src)
    fork_events = [torch.cuda.Event() for _ in t["tables"]]
    join_events = [torch.cuda.Event() for _ in t["tables"]]
    route_waves = load_route_waves(args.routes) if args.routes is not None else None
    records = []
    for total in map(int, args.rows.split(",")):
        if not 1 <= total <= 4096:
            raise ValueError("grouped benchmark rows must be 1..4096")
        gen = torch.Generator().manual_seed(408+total)
        copies = len(t["tables"])
        if route_waves:
            # Each cold layer copy replays a different recorded serving wave,
            # spread across the fixture's layers, rescaled to this row count.
            picks = [route_waves[(i * len(route_waves)) // copies] for i in range(copies)]
            ids_list = [real_route_ids(total, wave, gen).cuda().int() for wave in picks]
            route_meta = [dict(layer=w["layer"], lane=w["lane"], recorded_rows=w["rows"],
                               max_expert_share=max(n for _, n in w["expert_rows"]) / w["rows"])
                          for w in picks]
        else:
            ids_list = [torch.stack([torch.randperm(256,generator=gen)[:8]
                                     for _ in range(total)]).cuda().int()] * copies
            route_meta = None
        ids = ids_list[-1]
        weights = torch.rand((total,8),generator=gen).cuda()/8
        wire = make_wire(torch.randn((total,6144),generator=gen).cuda()*0.02)
        decoded = torch.empty((total,6144),dtype=torch.bfloat16,device="cuda")
        uniform_states = ([grouped_uniform_state(total, ts[0], props, width) for ts in exl_layers]
                          if owned is None else
                          serving_k4_states(total, exl_layers, props, width))
        chunk_response = torch.empty((total,6144),dtype=torch.bfloat16,device="cuda")
        chunk_wire = torch.empty((64,6336),dtype=torch.uint8,device="cuda")
        chunk_ids = torch.empty((64,8),dtype=torch.int32,device="cuda")
        chunk_weights = torch.empty((64,8),device="cuda")
        exch_wire = torch.empty_like(wire)
        exch_ids = torch.empty_like(ids)
        exch_weights = torch.empty_like(weights)
        exch_response = torch.empty((total, 6144), dtype=torch.bfloat16, device="cuda")

        def chunked():
            k1c, gc = control
            for idx, table in enumerate(t["tables"]):
                for first in range(0,total,64):
                    live = min(64,total-first)
                    chunk_wire[:live].copy_(wire[first:first+live])
                    chunk_ids[:live].copy_(ids_list[idx][first:first+live])
                    chunk_weights[:live].copy_(weights[first:first+live])
                    gc.launch(chunk_ids,chunk_wire,t["signs_in"],t["group_expert"],t["route_slots"],
                              t["route_group_lane"],t["group_count"],t["activation"],scalars=(live,limit))
                    k1c.launch(t["activation"],table,t["bits"],t["group_expert"],t["route_slots"],
                               t["route_group_lane"],chunk_weights,t["fc1"],t["intermediate"],t["fc2"],
                               t["signs_out"],t["output"],t["group_count"],scalars=(live,limit))
                    chunk_response[first:first+live].copy_(t["output"][:live])

        def candidate(program=grouped, ingress=group, cta_lim=limit):
            for idx, table in enumerate(t["tables"]):
                # Serving launches ingress at its own SM x 4 limit; only K1 takes cta_lim.
                ingress.launch(ids_list[idx],wire,t["signs_in"],state["group_expert"],state["route_slots"],
                             state["inverse"],state["count"],state["activation"],scalars=(total,limit))
                program.launch(state["activation"],table,t["bits"],state["group_expert"],state["route_slots"],
                               state["inverse"],weights,state["fc1"],state["intermediate"],state["fc2"],
                               t["signs_out"],state["output"],state["count"],
                               *(compact[:2] if compact_program(program) else ()),
                               scalars=((total,cta_lim,compact[2]) if compact_program(program)
                                        else (total,cta_lim))
                               + ((l2,) if any(s.name == "l2_bytes" for s in program.scalars) else ()))

        def candidate_exchange(program=grouped, ingress=group, cta_lim=limit):
            main_stream = torch.cuda.current_stream()
            for idx, table in enumerate(t["tables"]):
                exch_wire.copy_(wire)
                exch_ids.copy_(ids_list[idx])
                exch_weights.copy_(weights)
                fork_events[idx].record(main_stream)
                with torch.cuda.stream(exchange_stream):
                    exchange_stream.wait_event(fork_events[idx])
                    for p in range(exch_pulses):
                        _exchange_traffic[(8,)](exch_ring_src[p], exch_ring_dst[p],
                                                exch_slice_floats, BLOCK=1024)
                    join_events[idx].record(exchange_stream)
                ingress.launch(exch_ids,exch_wire,t["signs_in"],state["group_expert"],state["route_slots"],
                               state["inverse"],state["count"],state["activation"],scalars=(total,limit))
                program.launch(state["activation"],table,t["bits"],state["group_expert"],state["route_slots"],
                               state["inverse"],exch_weights,state["fc1"],state["intermediate"],state["fc2"],
                               t["signs_out"],state["output"],state["count"],
                               *(compact[:2] if compact_program(program) else ()),
                               scalars=((total,cta_lim,compact[2]) if compact_program(program)
                                        else (total,cta_lim))
                               + ((l2,) if any(s.name == "l2_bytes" for s in program.scalars) else ()))
                exch_response.copy_(state["output"][:total])
                main_stream.wait_event(join_events[idx])

        def baseline():
            for idx, uniform in enumerate(uniform_states):
                if owned is None:
                    wire_decode(wire, decoded, total)
                    run_grouped_uniform(uniform, decoded, weights, ids_list[idx])
                else:
                    run_bound_mixed_trellis(wire, weights, ids_list[idx], *uniform)

        def baseline_exchange():
            main_stream = torch.cuda.current_stream()
            for idx, uniform in enumerate(uniform_states):
                exch_wire.copy_(wire)
                exch_ids.copy_(ids_list[idx])
                exch_weights.copy_(weights)
                fork_events[idx].record(main_stream)
                with torch.cuda.stream(exchange_stream):
                    exchange_stream.wait_event(fork_events[idx])
                    for p in range(exch_pulses):
                        _exchange_traffic[(8,)](exch_ring_src[p], exch_ring_dst[p],
                                                exch_slice_floats, BLOCK=1024)
                    join_events[idx].record(exchange_stream)
                if owned is None:
                    wire_decode(exch_wire, decoded, total)
                    run_grouped_uniform(uniform, decoded, exch_weights, exch_ids)
                else:
                    run_bound_mixed_trellis(exch_wire, exch_weights, exch_ids, *uniform)
                exch_response.copy_(uniform[1].output[:total])
                main_stream.wait_event(join_events[idx])

        def mixed():
            for binding, bufs in mixed_states:
                run_bound_mixed_trellis(wire,weights,ids,binding,bufs)

        chunked(); candidate(); baseline(); mixed()
        torch.cuda.synchronize()
        from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
        default_error = grouped_error(state["output"][:total], chunk_response, weights,
                                      half_partials=Exlr8GroupedKernel.FC2_HALF)
        finite_nonzero(state["output"][:total].float(), "grouped EXLR8")
        finite_nonzero(uniform_states[-1][1].output[:total].float(), "uniform grouped EXL3")
        if mixed_states:
            finite_nonzero(mixed_states[-1][1].output[:total].float(), "mixed grouped EXL3")
        errors = {}
        for name, (program, ingress, _) in frozen.items():
            candidate(program, ingress)
            torch.cuda.synchronize()
            actual = state["output"][:total].float()
            finite_nonzero(actual, name)
            exact = torch.equal(chunk_response.view(torch.int16), state["output"][:total].view(torch.int16))
            if name not in args.approximate_arm and not exact:
                raise RuntimeError(f"frozen grouped arm {name} differs from snapshot chunked K1")
            reference = chunk_response.float()
            difference = (actual - reference).abs()
            row_relative = torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(reference, dim=1).clamp_min(1e-30)
            errors[name] = dict(bitwise_equal=exact,
                relative_l2=float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-30)),
                max_row_relative_l2=float(row_relative.max()), mean_row_relative_l2=float(row_relative.mean()),
                max_abs=float(difference.max()), fixture="synthetic weights and activations; recorded route distribution")
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            profile_arms = {"grouped": candidate, "exl3": baseline, "mixed": mixed}
            profile_arms.update({name: functools.partial(candidate, program, ingress)
                                 for name, (program, ingress, _) in frozen.items()})
            profile_arms[args.profile_arm]()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled arm={args.profile_arm} rows={total}", flush=True)
            continue
        graphs = {"chunked":capture(chunked),"grouped":capture(candidate),
                  "exl3":capture(baseline)}
        if mixed_states:
            graphs["mixed_unreachable_k5"] = capture(mixed)
        for name, (program, ingress, _) in frozen.items():
            graphs[name] = capture(functools.partial(candidate, program, ingress))
        rounds = {n:[] for n in graphs}
        names = list(graphs)
        for r in range(min(3,args.repeats)):
            for n in names[r%len(names):]+names[:r%len(names)]:
                rounds[n].append(timing(graphs[n],args.replays)/copies)
        med = {n:statistics.median(ts) for n,ts in rounds.items()}

        exch_graphs = {"grouped": capture(candidate_exchange),
                       "exl3": capture(baseline_exchange)}
        for name, (program, ingress, _) in frozen.items():
            exch_graphs[name] = capture(functools.partial(candidate_exchange, program, ingress))
        exch_rounds = {n: [] for n in exch_graphs}
        exch_names = list(exch_graphs)
        for r in range(min(3, args.repeats)):
            for n in exch_names[r % len(exch_names):] + exch_names[:r % len(exch_names)]:
                exch_rounds[n].append(timing(exch_graphs[n], args.replays) / copies)
        exch_med = {n: statistics.median(ts) for n, ts in exch_rounds.items()}

        cta96_graphs = {"grouped": capture(functools.partial(candidate, grouped, group, serving_limit))}
        for name, (program, ingress, _) in frozen.items():
            cta96_graphs[name] = capture(functools.partial(candidate, program, ingress, serving_limit))
        cta96_rounds = {n: [] for n in cta96_graphs}
        cta96_names = list(cta96_graphs)
        for r in range(min(3, args.repeats)):
            for n in cta96_names[r % len(cta96_names):] + cta96_names[:r % len(cta96_names)]:
                cta96_rounds[n].append(timing(cta96_graphs[n], args.replays) / copies)
        cta96_med = {n: statistics.median(ts) for n, ts in cta96_rounds.items()}

        exch_cta96_graphs = {"grouped": capture(functools.partial(candidate_exchange, grouped, group, serving_limit)),
                             "exl3": exch_graphs["exl3"]}
        for name, (program, ingress, _) in frozen.items():
            exch_cta96_graphs[name] = capture(functools.partial(candidate_exchange, program, ingress, serving_limit))
        exch_cta96_rounds = {n: [] for n in exch_cta96_graphs}
        exch_cta96_names = list(exch_cta96_graphs)
        for r in range(min(3, args.repeats)):
            for n in exch_cta96_names[r % len(exch_cta96_names):] + exch_cta96_names[:r % len(exch_cta96_names)]:
                exch_cta96_rounds[n].append(timing(exch_cta96_graphs[n], args.replays) / copies)
        exch_cta96_med = {n: statistics.median(ts) for n, ts in exch_cta96_rounds.items()}

        unique = (int(ids.unique().numel()) * 4 if owned is None else
                  sum(len(owned[e]) for e in ids.unique().tolist())) * 3 * 6144 * 128 * 4//8
        grouped_stage_map = {"ingress":("_Group",),"fc1":("grouped_fc1_kernel",),
                             "activation":("activation_kernel",),"fc2":("grouped_fc2_kernel", "grouped_fc2_ws_kernel"),
                             "egress":("reduce_kernel",),"exchange":("_exchange_traffic",)}
        stages = stage_profile(graphs["grouped"], args.output.parent, grouped_stage_map)
        arm_stages = {
            name: {k: v / copies if k.endswith("_us") else v
                   for k, v in stage_profile(graphs[name], args.output.parent, grouped_stage_map).items()}
            for name in frozen
        }
        exch_stages = {
            k: v / copies if k.endswith("_us") else v
            for k, v in stage_profile(exch_graphs["grouped"], args.output.parent, grouped_stage_map).items()
        }
        exch_arm_stages = {
            name: {k: v / copies if k.endswith("_us") else v
                   for k, v in stage_profile(exch_graphs[name], args.output.parent, grouped_stage_map).items()}
            for name in frozen
        }
        cta96_stages = {
            k: v / copies if k.endswith("_us") else v
            for k, v in stage_profile(cta96_graphs["grouped"], args.output.parent, grouped_stage_map).items()
        }
        cta96_arm_stages = {
            name: {k: v / copies if k.endswith("_us") else v
                   for k, v in stage_profile(cta96_graphs[name], args.output.parent, grouped_stage_map).items()}
            for name in frozen
        }
        exch_cta96_stages = {
            k: v / copies if k.endswith("_us") else v
            for k, v in stage_profile(exch_cta96_graphs["grouped"], args.output.parent, grouped_stage_map).items()
        }
        exch_cta96_arm_stages = {
            name: {k: v / copies if k.endswith("_us") else v
                   for k, v in stage_profile(exch_cta96_graphs[name], args.output.parent, grouped_stage_map).items()}
            for name in frozen
        }
        k4_stages = stage_profile(graphs["exl3"], args.output.parent,
            {"wire_decode": ("_wire_decode",), "fc1_fc2_fused": ("W4A16FusedMoeKernel",),
             "fc1": ("ws_fc1_kernel",), "activation": ("ws_act_kernel",),
             "fc2": ("ws_fc2_kernel",), "egress": ("W4A16TopKSumKernel",)})
        rec = dict(rows=total, routes=route_meta, **{n+"_us":v for n,v in med.items()}, rounds_us=rounds,
                   **{n+"_exchange_us": v for n, v in exch_med.items()},
                   exchange_rounds_us=exch_rounds,
                   exl3_exchange_speedup=exch_med["exl3"] / exch_med["grouped"],
                   **{n+"_cta96_us": v for n, v in cta96_med.items()},
                   cta96_rounds_us=cta96_rounds,
                   **{n+"_exchange_cta96_us": v for n, v in exch_cta96_med.items()},
                   exchange_cta96_rounds_us=exch_cta96_rounds,
                   exl3_exchange_cta96_speedup=exch_cta96_med["exl3"] / exch_cta96_med["grouped"],
                   exl3_stages={k:v/copies if k.endswith("_us") else v for k,v in k4_stages.items()},
                   chunked_speedup=med["chunked"]/med["grouped"],
                   exl3_speedup=med["exl3"]/med["grouped"], unique_weight_bytes=unique,
                   unique_read_roofline_percent=100*unique/med["grouped"]/1000/roof["gbps"],
                   stages={k:v/copies if k.endswith("_us") else v for k,v in stages.items()},
                   arm_stages=arm_stages,
                   exchange_stages=exch_stages,
                   exchange_arm_stages=exch_arm_stages,
                   cta96_stages=cta96_stages,
                   cta96_arm_stages=cta96_arm_stages,
                   exchange_cta96_stages=exch_cta96_stages,
                   exchange_cta96_arm_stages=exch_cta96_arm_stages,
                   groups=int(state["count"].item()), bitwise_equal=default_error["bitwise_equal"],
                   grouped_error=default_error, arm_errors=errors,
                   uniform_route_block=uniform_states[0][-1] if owned is None else 64,
                   uniform_input_conversion=("fp8_k32_to_bf16_included" if owned is None
                                             else "serving_ws_direct_fp8_k32"),
                   k4_options=None if owned is None else serving_k4_options(width))
        records.append(rec)
        print(json.dumps(rec),flush=True)
        del graphs, exch_graphs, cta96_graphs, exch_cta96_graphs, uniform_states
    if not args.profile:
        args.output.write_text(json.dumps(dict(schema=3, device=props.name, sm_count=props.multi_processor_count,
            l2_bytes=l2, measured_read_roofline=roof, group_rows=group_rows, rows=records,
            serving_prompts=args.serving_prompt, serving_lanes=args.serving_lanes,
            routes=str(args.routes) if args.routes else None,
            geometry=dict(world=args.world or 4, rank=args.rank, layer=args.layer,
                descriptor_units=units, owned_units_per_expert=([4]*256 if owned is None else list(map(len, owned))),
                unit_ids=owned, exl3_width=width,
                mode="legacy_compact" if owned is None else "serving_sparse"),
            primary_baseline=("uniform K4 full_rotation; wire decode included" if owned is None else
                              "serving GLM K4/K5 WS package geometry; all routes K4"),
            secondary_baseline="mixed K4/K5; K5 descriptor unreachable",
            frozen_grouped_sources={name: manifest for name, (_, _, manifest) in frozen.items()},
            target_met=all(r["exl3_speedup"]>=1.0 for r in records)),indent=2))


def serving_tier_bits(experts=256, units=16, owned=None, tier_mix="mixed"):
    """Build [E, U, 3] int32 bit-width descriptor table for rotated ownership."""
    if tier_mix not in ("mixed", "k4"):
        raise ValueError(f"unsupported tier_mix: {tier_mix!r}")
    bits = torch.zeros((experts, units, 3), dtype=torch.int32)
    for e in range(experts):
        for u in (range(units) if owned is None else owned[e]):
            for p in range(3):
                bits[e, u, p] = 4 if tier_mix == "k4" else (2, 3, 4)[(u + p) % 3]
    return bits


def make_serving_candidate(l2_bytes, *, world, rank, layer, tier_mix="mixed",
                           experts=256, hidden=6144, units=16, topk=8, min_copies=2):
    """Allocate cold-L2 rotated-ownership EXLR8 tables and 64-row serving scratch."""
    owned, _ = rank_geometry(world, rank, layer, experts=experts, units=units)
    bits_cpu = serving_tier_bits(experts=experts, units=units, owned=owned, tier_mix=tier_mix)
    tiles_per_unit = (hidden // 16) * (128 // 16)  # 3072 tiles of 256 elements per projection
    layer_bytes = int(bits_cpu.sum().item()) * tiles_per_unit * 32
    copies = max(min_copies, -(-L2_SPAN * l2_bytes // max(1, layer_bytes)))
    owners = []
    tables = []
    gen = torch.Generator(device="cuda").manual_seed(4080 + world * 100 + rank * 10 + layer)
    for _ in range(copies):
        pointers = torch.zeros((experts, units, 3, 3), dtype=torch.int64)
        for e in range(experts):
            for u in owned[e]:
                for p in range(3):
                    w = int(bits_cpu[e, u, p].item())
                    payload_bytes = tiles_per_unit * 32 * w
                    tensors = (
                        torch.randint(0, 256, (payload_bytes,), generator=gen, dtype=torch.uint8, device="cuda"),
                        (0.875 + 0.25 * torch.rand((128,), generator=gen, device="cuda")).half(),
                        torch.full((hidden // 128,), 1.0 / (hidden ** 0.5), dtype=torch.float16, device="cuda"),
                    )
                    owners.extend(tensors)
                    pointers[e, u, p] = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)
        tables.append(pointers.to("cuda"))
    host_gen = torch.Generator().manual_seed(4080)
    signs = [(torch.randint(2, (hidden,), generator=host_gen) * 2 - 1).char().cuda() for _ in range(2)]
    groups = 64 * topk
    return dict(
        tables=tables,
        owners=owners,
        owned=owned,
        bits_cpu=bits_cpu,
        bits=bits_cpu.cuda(),
        layer_bytes=layer_bytes,
        signs_in=signs[0],
        signs_out=signs[1],
        group_expert=torch.full((groups,), -1, dtype=torch.int32, device="cuda"),
        route_slots=torch.full((groups, 16), -1, dtype=torch.int32, device="cuda"),
        route_group_lane=torch.full((64, topk), -1, dtype=torch.int32, device="cuda"),
        group_count=torch.zeros(1, dtype=torch.int32, device="cuda"),
        activation=torch.empty((64, hidden), dtype=torch.float16, device="cuda"),
        **serial_scratch(hidden, units, groups),
        output=torch.empty((65, hidden), dtype=torch.bfloat16, device="cuda"),
    )


def serving_route_payload_bytes(ids, bits_cpu):
    """Exact unique weight payload bytes touched by the unique experts in `ids`."""
    unique_experts = ids.unique().cpu().tolist()
    if not unique_experts:
        return 0
    total_bits = int(bits_cpu[unique_experts].sum().item())
    return total_bits * (6144 // 16) * (128 // 16) * 32


def serving_main(args):
    from b12x._lib.runtime_control import kernel_resolution_guard
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 0) or (24 << 20))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    roof = None if args.profile else read_roofline()
    t = make_serving_candidate(
        l2, world=args.world, rank=args.rank, layer=args.layer,
        tier_mix=args.tier_mix, min_copies=args.layer_copies,
    )
    copies = len(t["tables"])
    variant_names = [v.strip() for v in args.serving_variants.split(",") if v.strip()]
    if not variant_names or len(variant_names) > 3 or len(set(variant_names)) != len(variant_names):
        raise ValueError("--serving-variants must specify 1..3 unique variant arms")
    if "baseline" in variant_names or any(v.endswith("_control") for v in variant_names):
        raise ValueError("--serving-variants must not include 'baseline' or '*_control'")

    group = compile_exlr8_group_aot(
        experts=256, top_k=8, max_groups=512, hidden=6144,
        input_format="fp8_k32", max_rows=64, group_rows=16,
    )
    arm_order = ["baseline", *variant_names]
    programs = {
        name: compile_exlr8_a16_aot(
            hidden=6144, units=16, top_k=8, max_groups=512,
            device_count=True, serving_variant=name,
        )
        for name in arm_order
    }
    cta_limit = int(props.multi_processor_count) * 4
    arm_metadata = {
        name: dict(
            program_name=prog.name,
            serving_variant=name,
            ctas_per_sm=int(prog.geometry["ctas_per_sm"]),
            cta_limit=cta_limit,
            max_groups=int(prog.geometry["max_groups"]),
            group_rows=int(prog.geometry.get("group_rows", 16)),
        )
        for name, prog in programs.items()
    }

    route_waves = load_route_waves(args.routes) if args.routes is not None else None
    row_list = [int(r) for r in args.rows.split(",")]
    if any(not 1 <= r <= 64 for r in row_list):
        raise ValueError("--serving rows must be in 1..64")

    def run_single(arm_name, wire_buf, ids_buf, weights_buf, rows, layer_idx=0):
        group.launch(
            ids_buf, wire_buf, t["signs_in"], t["group_expert"], t["route_slots"],
            t["route_group_lane"], t["group_count"], t["activation"],
            scalars=(rows, cta_limit),
        )
        programs[arm_name].launch(
            t["activation"], t["tables"][layer_idx], t["bits"], t["group_expert"],
            t["route_slots"], t["route_group_lane"], weights_buf, t["fc1"],
            t["intermediate"], t["fc2"], t["signs_out"], t["output"],
            t["group_count"], scalars=(rows, cta_limit),
        )

    # Warm up and resolve all arms at boundary row counts before the resolution guard.
    gen_warm = torch.Generator().manual_seed(4080)
    warm_ids = comparison_route_ids(64, gen_warm).cuda().int()
    warm_weights = (torch.rand((64, 8), generator=gen_warm).cuda() + 0.05) / 8
    warm_wire = make_wire(torch.randn((64, 6144), generator=gen_warm).cuda() * 0.05)
    for arm_name in arm_order:
        for r_warm in (1, 17, 63, 64):
            run_single(arm_name, warm_wire, warm_ids, warm_weights, r_warm, 0)
    torch.cuda.synchronize()

    invariance_rows = sorted({1, 4, 8, 16, 17, 32, 63, 64, *row_list})
    with kernel_resolution_guard("EXLR8 <=64 serving row invariance and bitwise oracle"):
        # Compute solo row-by-row reference with baseline at cta_limit=48.
        solo = torch.empty((64, 6144), dtype=torch.bfloat16, device="cuda")
        for r_idx in range(64):
            t["output"].fill_(float("nan"))
            run_single(
                "baseline",
                warm_wire[r_idx : r_idx + 1],
                warm_ids[r_idx : r_idx + 1],
                warm_weights[r_idx : r_idx + 1],
                1,
                0,
            )
            solo[r_idx : r_idx + 1].copy_(t["output"][:1])
        finite_nonzero(solo.float(), "baseline_solo")
        for r_check in invariance_rows:
            for arm_name in arm_order:
                t["output"].fill_(float("nan"))
                run_single(arm_name, warm_wire, warm_ids, warm_weights, r_check, 0)
                torch.cuda.synchronize()
                actual = t["output"][:r_check]
                if not torch.equal(actual.view(torch.int16), solo[:r_check].view(torch.int16)):
                    raise RuntimeError(f"row invariance / bitwise oracle failed for {arm_name} at rows={r_check}")
                if not bool(torch.isnan(t["output"][r_check:]).all()):
                    raise RuntimeError(f"out-of-bounds write past rows={r_check} for {arm_name}")

    records = []
    for rows in row_list:
        gen = torch.Generator().manual_seed(408 + rows)
        if route_waves:
            picks = [route_waves[(i * len(route_waves)) // copies] for i in range(copies)]
            ids_list = [real_route_ids(rows, wave, gen).cuda().int() for wave in picks]
            route_meta = [
                dict(
                    layer=w["layer"],
                    lane=w["lane"],
                    recorded_rows=w["rows"],
                    max_expert_share=max(n for _, n in w["expert_rows"]) / w["rows"],
                )
                for w in picks
            ]
        else:
            ids_list = [comparison_route_ids(rows, gen).cuda().int() for _ in range(copies)]
            route_meta = None
        weights = (torch.rand((rows, 8), generator=gen).cuda() + 0.05) / 8
        wire = make_wire(torch.randn((rows, 6144), generator=gen).cuda() * 0.05)

        # Pre-capture bitwise verification across all cold layer copies.
        expected_layers = []
        for layer_idx in range(copies):
            t["output"].fill_(float("nan"))
            run_single("baseline", wire, ids_list[layer_idx], weights, rows, layer_idx)
            torch.cuda.synchronize()
            exp = t["output"][:rows].clone()
            finite_nonzero(exp.float(), f"baseline_rows{rows}_layer{layer_idx}")
            expected_layers.append(exp)
            for arm_name in variant_names:
                t["output"].fill_(float("nan"))
                run_single(arm_name, wire, ids_list[layer_idx], weights, rows, layer_idx)
                torch.cuda.synchronize()
                act = t["output"][:rows]
                if not torch.equal(act.view(torch.int16), exp.view(torch.int16)):
                    raise RuntimeError(
                        f"pre-capture bitwise mismatch for {arm_name} at rows={rows}, layer={layer_idx}"
                    )

        def make_arm_step(arm_name):
            def step():
                for layer_idx in range(copies):
                    run_single(arm_name, wire, ids_list[layer_idx], weights, rows, layer_idx)
            return step

        graphs = {arm_name: capture(make_arm_step(arm_name)) for arm_name in arm_order}
        graphs["baseline_control"] = capture(make_arm_step("baseline"))

        # Post-capture & poisoned-output (-1 int16) double replay check.
        expected_last = expected_layers[-1]
        bitwise_checks = {}
        for arm_name in arm_order:
            for replay_idx in range(2):
                t["output"].view(torch.int16).fill_(-1)
                graphs[arm_name].replay()
                torch.cuda.synchronize()
                act = t["output"][:rows]
                if not torch.equal(act.view(torch.int16), expected_last.view(torch.int16)):
                    raise RuntimeError(
                        f"graph replay {replay_idx} bitwise mismatch for {arm_name} at rows={rows}"
                    )
                if not bool((t["output"][rows:].view(torch.int16) == -1).all()):
                    raise RuntimeError(
                        f"graph replay {replay_idx} wrote past rows={rows} for {arm_name}"
                    )
            bitwise_checks[arm_name] = True

        samples = {name: [] for name in graphs}
        order = list(graphs)
        repeats = max(3, int(args.repeats))
        for r in range(repeats):
            rotated = order[r % len(order) :] + order[: r % len(order)]
            if r % 2:
                rotated.reverse()
            for name in rotated:
                samples[name].append(timing(graphs[name], args.replays) / copies)
        noise = k4_noise_summary(samples, "baseline")
        med = noise["medians_us"]

        arm_stages = {
            arm_name: {
                k: (v / copies if k.endswith("_us") else v)
                for k, v in stage_profile(graphs[arm_name], args.output.parent).items()
            }
            for arm_name in arm_order
        }
        payload_bytes = sum(serving_route_payload_bytes(ids_l, t["bits_cpu"]) for ids_l in ids_list) / copies
        arm_gbps = {arm_name: payload_bytes / med[arm_name] / 1000.0 for arm_name in arm_order}
        arm_roofline_pct = {
            arm_name: 100.0 * arm_gbps[arm_name] / roof["gbps"] for arm_name in arm_order
        }
        rec = dict(
            rows=rows,
            routes=route_meta,
            payload_bytes=payload_bytes,
            medians_us=med,
            medians_ms={k: v / 1000.0 for k, v in med.items()},
            speedups=noise["speedups"],
            ratios={k: v / med["baseline"] for k, v in med.items()},
            identical_control_spread=noise["identical_control_spread"],
            noise_pass=noise["noise_pass"],
            samples_us=samples,
            gbps=arm_gbps,
            read_roofline_percent=arm_roofline_pct,
            arm_stages=arm_stages,
            bitwise_exact=bitwise_checks,
        )
        records.append(rec)
        print(json.dumps(rec), flush=True)
        del graphs

    result = dict(
        schema=1,
        kind="exlr8_routed_serving_duel_s3",
        device=props.name,
        sm_count=int(props.multi_processor_count),
        l2_bytes=l2,
        measured_read_roofline=roof,
        geometry=dict(
            world=args.world,
            rank=args.rank,
            layer=args.layer,
            hidden=6144,
            experts=256,
            units=16,
            top_k=8,
            owned_units_per_expert=[len(u) for u in t["owned"]],
            tier_mix=args.tier_mix,
            layer_bytes=t["layer_bytes"],
            layer_copies=copies,
        ),
        routes=str(args.routes) if args.routes else "random",
        arms=arm_metadata,
        checked_invariance_rows=invariance_rows,
        rows=records,
    )
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
