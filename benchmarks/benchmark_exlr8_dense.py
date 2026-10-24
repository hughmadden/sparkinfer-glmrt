#!/usr/bin/env python3
"""Native dense operand timing at matched GLM projection shapes, not model quality.

Payload throughput counts stored weight and scale bytes once per projection.
Inputs are already rotated for trellis; caller-owned rotation/output transforms
are excluded. The FP8 control uses the serving projection primitive at the same
shape and rows, with synthetic block-FP8 weights (not equivalent quantization).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import re
import sys
import contextlib

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_exlr8_a16 import capture, timing, read_roofline, finite_nonzero
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program, exportable_compilation
from b12x.integration.cuteafd import trellis_linear, scaled_linear, fp8_row_embedding
from b12x.integration.cuteafd._fp8_weights import Fp8Projection
from benchmarks._exlr8_dense_arms import parse_arms, parse_references, materialize, verify, program_key, cold_copies, validate_timing, timing_order, timing_summary

# Unique native formats/shapes from the GLM 5.3 bit325 load plan.
SHAPES = (
    ("trellis", "q_b", 2048, 16384, 1),
    ("trellis", "o", 16384, 6144, 2),
    ("trellis", "gate_up", 6144, 12288, 1),
    ("trellis", "down", 12288, 6144, 2),
    ("mxfp8", "q_a", 6144, 2048, 1),
    ("mxfp8", "kv_a", 6144, 576, 1),
    ("mxfp8", "index_q", 2048, 4096, 1),
    ("mxfp8", "index_k", 6144, 128, 1),
    ("int4_g32", "q_b", 2048, 16384, 1),
    ("int4_g32", "o", 16384, 6144, 2),
    ("fp8_row", "lm_head", 6144, 154880, 1),
    ("embedding", "embedding", 6144, 154880, 1),
)


class BlockFp8:
    """Serving block-FP8 projection. ``fp8_rows`` 255 selects W8A8 prefill, 0 W8A16
    (TmaFp8Gemm: BF16 activations, the A16 comparator for native operands)."""

    def __init__(self, k, n, prefill, fp8_rows=255):
        self.projection = Fp8Projection(n, k, prefill_rows=4096 if prefill else None)
        self.fp8_rows = fp8_rows

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scales: cute.Pointer,
                 scratch: cute.Pointer, output: cute.Pointer, rows: cutlass.Int32,
                 stream: cuda.CUstream):
        self.projection(x, w, scales, output, rows, cutlass.Int32(self.fp8_rows),
                        cutlass.Int64(scratch.toint()), stream)


def copies_for(weight_bytes, cold_l2):
    """Consecutive-layer copies span twice this device's queried L2 capacity."""
    if not cold_l2:
        return 1
    status, device = cuda.cuDeviceGet(torch.cuda.current_device())
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {status}")
    status, l2_bytes = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGetAttribute(L2CacheSize) failed: {status}")
    return cold_copies(weight_bytes, l2_bytes)


def rotated(launches):
    """One graph launching each copy in turn; ``timing`` divides by ``len(launches)``."""
    def run():
        for launch in launches:
            launch()
    return run


def baseline(k, n, rows, a16=False, cold_l2=False, graph_repeats=1):
    # Decode programs keep their GEMV to 64 rows either way; prefill W8A16 on fp8_rows 0.
    core = BlockFp8(k, n, rows > 64, 0 if a16 and rows > 64 else 255)
    ops = (Operand("x", torch.bfloat16, "[rows,K]"),
           Operand("w", torch.float8_e4m3fn, "[N,K]"),
           Operand("scales", torch.float32, "[ceil(N/128),K/128]"),
           Operand("scratch", torch.uint8, "[bytes]", "scratch"),
           Operand("output", torch.bfloat16, "[rows,N]", "out"))
    program = compile_program(core, name="bench_dense_block_fp8", operands=ops,
                              scalars=(Scalar("rows"),), key=(k, n, rows > 64, core.fp8_rows))
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16) * .1
    weight = torch.randint(-16, 17, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full(((n+127)//128, k//128), .01, device="cuda")
    extra = [(weight.clone(), scales.clone()) for _ in range(copies_for(n*k, cold_l2 and rows <= 64)-1)]
    scratch = torch.empty(max(1024, core.projection.prefill_scratch_bytes(rows)),
                          dtype=torch.uint8, device="cuda")
    output = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    tensors = (x, weight, scales, scratch, output)
    launches = [lambda: program.launch(*tensors, scalars=(rows,))]
    launches += [lambda w=w, s=s: program.launch(x, w, s, scratch, output, scalars=(rows,)) for w, s in extra]
    graph = capture(rotated(launches * graph_repeats))
    finite_nonzero(output, "block-FP8")
    return graph, (tensors, extra), len(launches)


def compile_native(fmt, k, n, side, input_dtype="f16"):
    if fmt == "trellis":
        return trellis_linear.compile_trellis_linear_aot(input_features=k, output_features=n,
                                                       sharded_side=side, input_dtype=input_dtype)
    if fmt == "embedding":
        return fp8_row_embedding.compile_fp8_row_embedding_aot(hidden=k, vocabulary=n)
    return scaled_linear.compile_scaled_linear_aot(input_features=k, output_features=n,
                                                  encoding=fmt, sharded_side=side)


@contextlib.contextmanager
def arm_source(name, arm, bundle, fmt="mxfp8"):
    """Freeze the implementation package and restore every imported sibling."""
    import b12x.gemm.native_tile_linear as current
    from benchmarks._exlr8_dense_arms import MODULE, TRELLIS
    suffix = name + "_" + arm["sha256"]
    integration = trellis_linear if fmt == "trellis" else scaled_linear
    original_compile = integration.compile_program
    saved = {}
    parent = None
    if fmt == "trellis":
        import b12x.gemm as parent
        package = "b12x.gemm.trellis_linear"
        saved = {k: v for k, v in sys.modules.items() if k == package or k.startswith(package + ".")}
        old_parent = getattr(parent, "trellis_linear", None)
        for key in saved:
            del sys.modules[key]
        entry = arm["files"][TRELLIS + "/__init__.py"]
        path = bundle / entry["path"]
        spec = importlib.util.spec_from_file_location(package, path, submodule_search_locations=[str(path.parent)])
    else:
        entry = arm.get("files", {}).get(MODULE, arm)
        path = bundle / entry["path"]
        spec = importlib.util.spec_from_file_location("dense_arm_" + suffix, path)
    module = importlib.util.module_from_spec(spec)

    def compile_distinct(*args, **kwargs):
        kwargs["name"] += "_" + suffix[:len(name)+17]
        kwargs["key"] = program_key(arm, kwargs["key"])
        return original_compile(*args, **kwargs)

    original_class = current.NativeTileLinear
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if fmt != "trellis":
            current.NativeTileLinear = module.NativeTileLinear
        integration.compile_program = compile_distinct
        yield
    finally:
        integration.compile_program = original_compile
        current.NativeTileLinear = original_class
        if fmt == "trellis":
            for key in list(sys.modules):
                if key == package or key.startswith(package + "."):
                    del sys.modules[key]
            sys.modules.update(saved)
            if old_parent is None:
                if hasattr(parent, "trellis_linear"):
                    delattr(parent, "trellis_linear")
            else:
                parent.trellis_linear = old_parent


def trellis_reference(k, n, columns, side, payload, scales):
    """Independent MSB-first CPU oracle, restricted to checked output units."""
    import numpy as np
    from tests._reference.native_trellis import Reference
    ref = Reference("e4m3")
    if side == 1:
        raw = payload.flatten()[:columns*k//2].cpu().numpy()
        weight = np.concatenate([ref.decode_unit(u.tobytes(), 4, "gate", k)
                                 for u in raw.reshape(columns//128, -1)], axis=1)
    else:
        raw = payload.reshape(k//128, n//128, 8192)[:, :columns//128].contiguous().cpu().numpy()
        weight = np.concatenate([ref.decode_unit(u.tobytes(), 4, "down", columns)
                                 for u in raw], axis=0)
    return (torch.from_numpy(weight).to(device="cuda") * scales.float().repeat_interleave(128)[:, None]).T.contiguous()


def reference_weight(fmt, k, n, side, payload, scales):
    q = payload.reshape(n, -1) if side == 1 else payload.reshape(k//128, n, -1).permute(1, 0, 2).reshape(n, -1)
    if fmt == "int4_g32":
        values = torch.stack((q & 15, q >> 4), -1).reshape(n, k).float() - 8
    else:
        values = q.contiguous().view(torch.float8_e4m3fn).float()
    if fmt == "fp8_row":
        return values * scales[:, None]
    scale = scales.reshape(n, k//32) if side == 1 else scales.reshape(k//128, n, 4).permute(1, 0, 2).reshape(n, k//32)
    if fmt == "mxfp8":
        scale = torch.exp2(scale.float() - 127)
    return values * scale.float().repeat_interleave(32, 1)


def native_tensors(fmt, k, n, rows, splits=None):
    if fmt == "embedding":
        ids = torch.arange(rows, dtype=torch.int32, device="cuda") % n
        payload = torch.randint(-16, 17, (n, k), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.full((n,), .01, device="cuda")
        output = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
        return (ids, payload, scales, output), rows*k+rows*4
    dtype = torch.float16 if fmt == "trellis" else torch.bfloat16
    x = torch.randn((rows, k), dtype=dtype, device="cuda") * .1
    if fmt == "trellis":
        inputs = torch.tensor([x.data_ptr()]*4, dtype=torch.int64, device="cuda")
        payload = torch.randint(0, 256, (n*k//2,), dtype=torch.uint8, device="cuda")
        scales = torch.ones((k//128,), dtype=torch.float16, device="cuda")
    else:
        inputs = x
        if fmt == "int4_g32":
            payload = torch.randint(0, 256, (n*k//2,), dtype=torch.uint8, device="cuda")
            scales = torch.full((n*k//32,), .01, dtype=torch.float16, device="cuda")
        else:
            payload = torch.randint(-16, 17, (n, k), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
            scales = (torch.full((n*k//32,), 120, dtype=torch.uint8, device="cuda") if fmt == "mxfp8"
                      else torch.full((n,), .01, device="cuda"))
    # The tile core takes no split-K partials (splits 0: a 4-byte placeholder).
    splits = (k+1023)//1024 if splits is None else splits
    partial = torch.empty((rows, splits, n) if splits else (1,), dtype=torch.float32, device="cuda")
    output = torch.empty((rows, n), dtype=torch.float32, device="cuda")
    # Keep x alive even when the ABI receives only the pointer table.
    return (inputs, payload, scales, partial, output, x), payload.numel()+scales.numel()*scales.element_size()


def spill_check(program, directory):
    """Inspect the exact exported program's embedded cubins, not a profiled run."""
    directory.mkdir(parents=True, exist_ok=True)
    program.export_to_c(str(directory), "native", "bench_native")
    root = Path("/workspace/cuteafd")
    path = root / "scripts/qualify/deepseek_v41/inspect-cute-object.py"
    spec = importlib.util.spec_from_file_location("dense_cuda_images", path)
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    images = list(inspector.cuda_images((directory / "native.o").read_bytes()))
    if not images:
        raise RuntimeError("native AOT export contains no cubins")
    records = []
    for index, (_, blob) in enumerate(images):
        cubin = directory / f"native-{index}.cubin"
        cubin.write_bytes(blob)
        resources = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump",
                                             "--dump-resource-usage", str(cubin)], text=True)
        local = re.findall(r"LOCAL:(\d+)", resources)
        if not local:
            raise RuntimeError("cuobjdump returned no local-memory records")
        sass = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin)], text=True)
        records.append(dict(sha256=hashlib.sha256(blob).hexdigest(), resources=resources,
                            local_bytes=list(map(int, local)),
                            stack_bytes=list(map(int, re.findall(r"STACK:(\d+)", resources))),
                            local_load_store_instructions=len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))))
    return records


def source_revision():
    source = Path(__file__).resolve().parents[1]
    if (source / ".git").exists():
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    # Release compiler sources are verified, metadata-free copies.
    return json.loads((source.parent / "sparkinfer.lock.json").read_text())["revision"]


def benchmark_unpacked(args):
    """Sanctioned trellis-only FP8 unpack arm, sharing the existing graph timer."""
    from b12x.integration.cuteafd.trellis_unpack import compile_trellis_unpack_e4m3_aot
    from b12x.integration.cuteafd.native_dense import compile_trellis_fp8_prefill_aot
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(r < 1 or r > 4096 for r in rows_list) or args.replays < 1 or args.output is None:
        raise ValueError("unpacked arm requires output, rows in 1..4096 and positive replays")
    shapes = [s for s in SHAPES if s[0] == "trellis"]
    shapes += [(fmt, op + "_tp2", k if side == 1 else k//2,
                n//2 if side == 1 else n, side) for fmt, op, k, n, side in shapes]
    if args.shapes:
        names = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[1] in names]
    if not shapes:
        raise ValueError("empty trellis shape selection")
    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    report = dict(device=props.name, sms=props.multi_processor_count,
        revision=source_revision(),
        scope="synthetic native K4/e4m3; already-rotated input; one-time unpack reported separately",
        measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for _, op, k, n, side in shapes:
        capacity = max(rows_list)
        storage, read_bytes = native_tensors("trellis", k, n, capacity)
        inputs, payload, scales, partial, native, x16 = storage
        x = x16.bfloat16()
        if args.trellis_input_dtype == "bf16":
            inputs = torch.tensor([x.data_ptr()]*4, dtype=torch.int64, device="cuda")
        else:
            x16.copy_(x)
        weight = torch.empty((n, k), dtype=torch.uint8, device="cuda")
        unit_grid = torch.ones((n//128, k//128), device="cuda")
        output = torch.empty_like(native)
        unpack = compile_trellis_unpack_e4m3_aot(input_features=k, output_features=n, sharded_side=side)
        gemm = compile_trellis_fp8_prefill_aot(input_features=k, output_features=n)
        core = compile_native("trellis", k, n, side, input_dtype=args.trellis_input_dtype)
        unpack_graph = capture(lambda: unpack.launch(payload, weight, scalars=()))
        finite_nonzero(weight.view(torch.float8_e4m3fn).float(), "unpacked E4M3")
        unpack_us = timing(unpack_graph, args.replays)
        native_resources = spill_check(core, args.output.parent / "resources" / op)
        spill_free = all(not any(r["local_bytes"]) and not any(r["stack_bytes"])
                         and r["local_load_store_instructions"] == 0 for r in native_resources)
        for rows in rows_list:
            scalars = (rows, 0, props.multi_processor_count*4)
            native_graph = capture(lambda: core.launch(inputs, payload, scales, partial, native, scalars=scalars))
            gemm_graph = capture(lambda: gemm.launch(x, weight, unit_grid, output, scalars=(rows,)))
            finite_nonzero(native[:rows], "native trellis")
            finite_nonzero(output[:rows], "unpacked FP8 GEMM")
            # The unit and synthetic CPU-oracle tests gate every unpacked byte.
            # Here check the actual full-shape GEMM against native and an FP64 slice.
            columns = min(n, 128)
            target = x[:rows].double() @ weight[:columns].view(torch.float8_e4m3fn).double().T
            bf16 = (x[:rows] @ weight[:columns].view(torch.float8_e4m3fn).bfloat16().T).double()
            bound = 1.01*torch.linalg.vector_norm(bf16-target)
            native_error = torch.linalg.vector_norm(native[:rows, :columns].double()-target)
            gemm_error = torch.linalg.vector_norm(output[:rows, :columns].double()-target)
            passed = bool(native_error <= bound and gemm_error <= bound and spill_free)
            native_us, gemm_us = timing(native_graph, args.replays), timing(gemm_graph, args.replays)
            base_graph, keep, count = baseline(k, n, rows, a16=True)
            base_us = timing(base_graph, args.replays)/count
            total = unpack_us + gemm_us
            delta = output[:rows].double()-native[:rows].double()
            relative_l2 = (torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(native[:rows].double())).item()
            record = dict(projection=op, k=k, n=n, side=side, rows=rows,
                unpack_us=unpack_us, gemm_us=gemm_us, unpack_plus_gemm_us=total,
                native_us=native_us, w8a16_us=base_us, fp64_pass=passed,
                input_dtype=args.trellis_input_dtype, spill_free=spill_free, native_resources=native_resources,
                native_speedup_vs_w8a16=base_us/native_us,
                native_fp64_error=native_error.item(), gemm_fp64_error=gemm_error.item(),
                fp64_bound=bound.item(), relative_l2=relative_l2,
                speedup_vs_native=native_us/total, speedup_vs_w8a16=base_us/total,
                gemm_speedup_vs_w8a16=base_us/gemm_us, read_bytes=read_bytes)
            report["measurements"].append(record)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(record), flush=True)
            del native_graph, gemm_graph, base_graph, keep, target, bf16, delta
            if not passed:
                raise RuntimeError(f"{op} rows={rows}: FP64 gate failed")
        del unpack_graph, storage, inputs, payload, scales, partial, native, x16, x, weight, unit_grid, output
    return False


def benchmark_native_bf16(args):
    """Full-output A8 vs current A16, including the serving companions.

    The existing dense harness has no checkpoint record loader; use synthetic
    trellis payloads and deterministic BF16 inputs, with identical signs/scales.
    """
    import statistics
    from b12x.integration.cuteafd.dense_mxfp8 import (
        compile_dense_mxfp8_aot, compile_dense_mxfp8_quant_aot, record_bytes)
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot)
    if args.formats != "trellis" or args.rows != "4096" or args.output is None:
        raise ValueError("--compare-native-bf16 requires --formats trellis --rows 4096 --output")
    names = set(args.shapes.split(",")) if args.shapes else {"q_b", "o", "gate_up", "down"}
    shapes = [s for s in SHAPES if s[0] == "trellis" and s[1] in names]
    if not shapes or names - {s[1] for s in shapes}:
        raise ValueError("--compare-native-bf16 requires full trellis projection names")
    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    rows, limit = 4096, props.multi_processor_count * 4
    report = dict(device=props.name, sms=props.multi_processor_count,
                  scope="synthetic trellis K4/e4m3; BF16 input; input/output companions included; A8 quant included",
                  seed=408, measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for _, op, k, n, side in shapes:
        x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16) * .1
        payload = torch.randint(0, 256, (n*k//2,), dtype=torch.uint8, device="cuda")
        suh = torch.ones(k, dtype=torch.float16, device="cuda")
        svh = torch.ones(n, dtype=torch.float16, device="cuda")
        scales = torch.empty(k//128, dtype=torch.float16, device="cuda")
        records = torch.empty(record_bytes(rows, k), dtype=torch.uint8, device="cuda")
        partial = torch.empty(256, dtype=torch.float32, device="cuda")
        acc = torch.empty((rows, n), dtype=torch.float32, device="cuda")
        outputs = {arm: torch.empty((rows, n), dtype=torch.bfloat16, device="cuda") for arm in ("a16", "a8")}
        rotated_inputs = {arm: torch.empty_like(x, dtype=dtype) for arm, dtype in
                          (("a16", torch.bfloat16), ("a8", torch.float16))}
        tables = {arm: torch.tensor([value.data_ptr()]*4, device="cuda", dtype=torch.int64)
                  for arm, value in rotated_inputs.items()}
        transforms = {arm: compile_trellis_input_aot(features=k, output_dtype=dtype)
                      for arm, dtype in (("a16", "bf16"), ("a8", "f16"))}
        output = compile_trellis_output_aot(features=n)
        a16 = compile_native("trellis", k, n, side, input_dtype="bf16")
        a8 = compile_dense_mxfp8_aot(input_features=k, output_features=n, sharded_side=side)
        quant = compile_dense_mxfp8_quant_aot(input_features=k)
        def run(arm):
            transforms[arm].launch(x, suh, rotated_inputs[arm], scales, scalars=(rows, limit))
            if arm == "a8":
                quant.launch(tables[arm], records, scalars=(rows, 0, limit))
                a8.launch(tables[arm], payload, scales, records, acc, scalars=(rows, 0, limit))
            else:
                a16.launch(tables[arm], payload, scales, partial, acc, scalars=(rows, 0, limit))
            output.launch(acc, svh, outputs[arm], scalars=(rows, limit))
        # Capture warms both arms before correctness and timing.
        graphs = {arm: capture(lambda arm=arm: run(arm)) for arm in ("a16", "a8")}
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        for arm in outputs:
            finite_nonzero(outputs[arm].float(), arm)
        delta = outputs["a8"].double() - outputs["a16"].double()
        relative = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(outputs["a16"].double())).item()
        max_abs = delta.abs().max().item()
        samples = {arm: [] for arm in graphs}
        for iteration in range(3):
            for arm in timing_order(graphs, iteration):
                samples[arm].append(timing(graphs[arm], args.replays))
        us = {arm: statistics.median(values) for arm, values in samples.items()}
        entry = dict(projection=op, k=k, n=n, rows=rows, relative_frobenius=relative,
                     max_abs_error=max_abs, samples_us=samples,
                     a16_ms=us["a16"]/1000, a8_ms=us["a8"]/1000,
                     a16_tflops=2*rows*k*n/(us["a16"]*1e6),
                     a8_tflops=2*rows*k*n/(us["a8"]*1e6), speedup=us["a16"]/us["a8"])
        report["measurements"].append(entry)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(entry), flush=True)
        del graphs, delta, outputs, rotated_inputs, tables, payload, records, acc, x
    return False


def benchmark_mx8(args):
    """Shared-record native prefill, quantized-A oracle and frozen row checks."""
    from b12x.integration.cuteafd.dense_mxfp8 import (
        compile_dense_mxfp8_aot, compile_dense_mxfp8_quant_aot, record_bytes)
    from b12x.integration.cuteafd.trellis_unpack import compile_trellis_unpack_e4m3_aot
    from b12x.integration.cuteafd.native_dense import compile_trellis_fp8_prefill_aot
    rows_list = [int(r) for r in args.rows.split(",")]
    if not rows_list or any(not 0 < r <= 4096 for r in rows_list) or args.output is None:
        raise ValueError("MX8 requires output and rows in 1..4096")
    shapes = [s for s in SHAPES if s[0] in {"trellis", "mxfp8"}]
    shapes += [(f, op+"_tp2", k if side == 1 else k//2,
                n//2 if side == 1 else n, side) for f, op, k, n, side in shapes]
    names = set(args.shapes.split(",")) if args.shapes else {s[1] for s in shapes}
    formats = set(args.formats.split(","))
    shapes = [s for s in shapes if s[1] in names and s[0] in formats]
    if not shapes:
        raise ValueError("empty MX8 shape selection")
    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    limit, capacity = props.multi_processor_count*4, max(rows_list)
    report = dict(device=props.name, sms=props.multi_processor_count, measurements=[],
                  scope="native weights, shared MXFP8 records, synthetic operands; rotation excluded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fmt, op, k, n, side in shapes:
        storage, _ = native_tensors(fmt, k, n, capacity)
        inputs, payload, scales, partial, output, x = storage
        # Quantization range regressions: zero, inverse scales beyond F16,
        # and finite near-maximum F16 inputs. Keep these in the FP64 checks.
        pattern = torch.linspace(-1, 1, k, device="cuda")
        for row, amplitude in enumerate((0.0, 1e-3, 1e-5, 60000.0)):
            if row < capacity:
                x[row].copy_(pattern * amplitude)
        if fmt == "mxfp8":
            scales.random_(118,124)
        else:
            scales.copy_(torch.exp2(torch.randint(-2,3,scales.shape,device="cuda").float()))
        table = (inputs if fmt == "trellis" else
                 torch.tensor([x.data_ptr()]*4, dtype=torch.int64, device="cuda"))
        dtype = "f16" if fmt == "trellis" else "bf16"
        quant = compile_dense_mxfp8_quant_aot(input_features=k, input_dtype=dtype)
        with exportable_compilation():
            core = compile_dense_mxfp8_aot(input_features=k, output_features=n,
                       encoding=fmt, sharded_side=side, input_dtype=dtype)
        spills = spill_check(core, args.output.parent/"cubins"/f"{fmt}-{op}")
        spill_free = all(not any(r["local_bytes"]+r["stack_bytes"])
                         and not r["local_load_store_instructions"] for r in spills)
        records = torch.empty(record_bytes(capacity, k), dtype=torch.uint8, device="cuda")
        native = compile_native(fmt, k, n, side)
        if fmt == "trellis":
            weight_bytes = torch.empty((n,k), dtype=torch.uint8, device="cuda")
            unpack = compile_trellis_unpack_e4m3_aot(input_features=k, output_features=n, sharded_side=side)
            unpack_graph = capture(lambda: unpack.launch(payload, weight_bytes, scalars=()))
            unpack_us = timing(unpack_graph, args.replays)
            weight = weight_bytes.view(torch.float8_e4m3fn).float()
            weight *= scales.float().repeat_interleave(128)[None,:]
            gemm = compile_trellis_fp8_prefill_aot(input_features=k, output_features=n)
            unit_grid = scales.float().expand(n//128, -1).contiguous()
            xb = x.bfloat16()
        else:
            weight = reference_weight(fmt,k,n,side,payload,scales)
        def run(rows):
            quant.launch(table, records, scalars=(rows,0,limit))
            core.launch(table,payload,scales,records,output,scalars=(rows,0,limit))
        run(capacity)
        torch.cuda.synchronize()
        frozen = output[:min(capacity,257)].clone()
        # Decode the actual MMA-order records independently into row-major A.
        rec = records.view(-1,k//32,528)
        packed = rec[:,:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(-1,k//32,32,16)
        aq = torch.empty((rec.shape[0]*16,k), dtype=torch.float32, device="cuda")
        for lane in range(32):
            r, c = lane//4, 4*(lane%4)
            for row_off, byte_off in ((0,0),(8,4)):
                scale = torch.exp2(rec[:,:,512+r+row_off].float()-127)
                for ko, bo in ((0,byte_off),(16,byte_off+8)):
                    values = packed[:,:,lane,bo:bo+4]*scale[:,:,None]
                    aq.reshape(-1,16,k)[:,r+row_off,:].reshape(-1,k//32,32)[:,:,c+ko:c+ko+4] = values
        amax = x.float().reshape(capacity,k//32,32).abs().amax(-1)
        exponent = torch.ceil(torch.log2(amax/448)).clamp(-127,127)
        sf = torch.exp2(exponent)
        scaled_a = torch.where(amax[:,:,None] == 0, 0,
                               x.float().reshape(capacity,k//32,32)/sf[:,:,None])
        expected_a = (scaled_a.to(torch.float8_e4m3fn).float()*sf[:,:,None]).reshape(capacity,k)
        quant_equal = torch.equal(aq[:capacity],expected_a)
        columns, checks = min(n,64), min(capacity,32)
        target = aq[:checks].double() @ weight[:columns].double().T
        exact = x[:checks].double() @ weight[:columns].double().T
        # Standard forward FP32 accumulation bound, applied elementwise.
        absolute_products = aq[:checks].double().abs() @ weight[:columns].double().abs().T
        gamma = (k*2**-24)/(1-k*2**-24)
        error = (output[:checks,:columns].double()-target).abs()
        fp64_pass = bool((error <= gamma*absolute_products+1e-7).all())
        for count in sorted({r for r in (1,15,16,17,127,128,129,255,256,257) if r<=capacity}):
            output.fill_(float("nan"))
            run(count)
            torch.cuda.synchronize()
            if not torch.equal(output[:count].view(torch.int32),frozen[:count].view(torch.int32)):
                raise RuntimeError(f"{fmt}/{op}: row invariance failed at {count}")
            if not bool(torch.isnan(output[count:]).all()):
                raise RuntimeError(f"{fmt}/{op}: output tail overwritten")
        for rows in rows_list:
            graph = capture(lambda: run(rows))
            total_us = timing(graph,args.replays)
            qgraph = capture(lambda: quant.launch(table,records,scalars=(rows,0,limit)))
            quant_us = timing(qgraph,args.replays)
            ggraph = capture(lambda: core.launch(table,payload,scales,records,output,scalars=(rows,0,limit)))
            gemm_us = timing(ggraph,args.replays)
            ns = (rows,0,limit) if fmt == "trellis" else (rows,limit)
            ng = capture(lambda: native.launch(*storage[:5],scalars=ns))
            native_us = timing(ng,1)
            bars = {}
            for label,a16 in (("w8a16",True),("w8a8",False)):
                try:
                    bg,keep,count = baseline(k,n,rows,a16=a16)
                    bars[label+"_us"] = timing(bg,args.replays)/count
                    del bg,keep
                except (TypeError,ValueError) as exc:
                    bars[label+"_unsupported"] = str(exc)
            unpack_total = None
            if fmt == "trellis":
                ug = capture(lambda: gemm.launch(xb,weight_bytes,unit_grid,output,scalars=(rows,)))
                unpack_total = unpack_us+timing(ug,args.replays)
            record = dict(format=fmt,projection=op,k=k,n=n,side=side,rows=rows,
                mx8_us=total_us,quant_us=quant_us,gemm_us=gemm_us,native_us=native_us,
                unpack_us=unpack_total,quant_equal=quant_equal,row_invariant=True,
                fp64_pass=fp64_pass,fp64_max_error=error.max().item(),
                a8_vs_exact_relative_l2=(torch.linalg.vector_norm(target[4:]-exact[4:])/torch.linalg.vector_norm(exact[4:])).item(),
                quant_range_rows=[0.0,1e-3,1e-5,60000.0],
                spill_free=spill_free,cubin_spill_check=spills,**bars)
            report["measurements"].append(record)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(record),flush=True)
        if not (quant_equal and fp64_pass and spill_free):
            raise RuntimeError(f"{fmt}/{op}: MX8 correctness/resource gate failed")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mxfp8-prefill", action="store_true", help="shared MXFP8 records at whole and TP2 native dense shapes")
    parser.add_argument("--compare-native-bf16", action="store_true", help="full-output A8 vs serving A16 with companions; requires --mxfp8-prefill")
    parser.add_argument("--trellis-input-dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--unpacked-trellis", action="store_true", help="measure exact E4M3 unpack plus F32 W8A16 at full/TP2 trellis shapes")
    parser.add_argument("--arm", action="append", help="NAME=git-rev; repeatable")
    parser.add_argument("--arm-bundle", type=Path)
    parser.add_argument("--materialize-only", action="store_true", help="host-side git show, no CUDA")
    parser.add_argument("--reference")
    parser.add_argument("--op-reference", action="append", default=[], help="FORMAT/OP=ARM frozen route")
    parser.add_argument("--accuracy-reference", help="arm whose FP64 error defines the 1.01x ceiling")
    parser.add_argument("--require-reference-arm", action="append", default=[], help="arm required to match the selected frozen op reference")
    parser.add_argument("--formats", default="mxfp8,int4_g32,fp8_row")
    parser.add_argument("--shapes", help="comma-separated projection names")
    parser.add_argument("--rows", default="1,4,16,64,4096")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--interleave-rounds", type=int, default=0, help="alternate arms and duplicate reference control; replays are captured inside each graph")
    parser.add_argument("--profile-arm")
    parser.add_argument("--profile-op", help="FORMAT/OP")
    parser.add_argument("--profile-rows", type=int)
    args = parser.parse_args()
    try:
        validate_timing(args.interleave_rounds, args.replays)
    except ValueError as error:
        parser.error(str(error))
    if args.compare_native_bf16:
        if not args.mxfp8_prefill:
            parser.error("--compare-native-bf16 requires --mxfp8-prefill")
        return benchmark_native_bf16(args)
    if args.mxfp8_prefill:
        return benchmark_mx8(args)
    if args.unpacked_trellis:
        return benchmark_unpacked(args)
    if not args.arm or args.arm_bundle is None or args.reference is None:
        parser.error("multi-arm scaled dense requires --arm, --arm-bundle and --reference")
    arms = parse_arms(args.arm)
    try:
        op_references = parse_references(args.reference, args.op_reference, arms)
    except ValueError as error:
        parser.error(str(error))
    accuracy_reference = args.accuracy_reference or args.reference
    if accuracy_reference not in arms or set(args.require_reference_arm) - set(arms):
        parser.error("accuracy and required-reference arms must name an arm")
    if args.materialize_only:
        materialize(Path(__file__).resolve().parents[1], args.arm_bundle, arms, args.formats.split(","))
        return
    arms = verify(args.arm_bundle, arms, args.formats.split(","))
    if args.output is None:
        parser.error("--output is required for measurement")
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(r < 1 or r > 4096 for r in rows_list) or args.replays < 1:
        parser.error("rows must be in 1..4096 and replays positive")
    formats = set(args.formats.split(","))
    if formats - {"trellis", "mxfp8", "int4_g32", "fp8_row"}:
        parser.error("multi-arm dense core supports trellis, mxfp8, int4_g32 and fp8_row")
    names = set(args.shapes.split(",")) if args.shapes else {s[1] for s in SHAPES}
    shapes = [s for s in SHAPES if s[0] in formats and s[1] in names]
    profile = any((args.profile_arm, args.profile_op, args.profile_rows))
    if profile:
        if args.profile_arm not in arms or not args.profile_op or args.profile_rows not in rows_list:
            parser.error("profile requires a named arm, FORMAT/OP and one selected row count")
        shapes = [s for s in shapes if f"{s[0]}/{s[1]}" == args.profile_op]
        rows_list = [args.profile_rows]
    if not shapes:
        parser.error("empty shape selection")
    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    limit = props.multi_processor_count * 4
    report = dict(device=props.name, sms=props.multi_processor_count, arms=arms,
                  read_ceiling=read_roofline(), measurements=[],
                  scope="synthetic scaled dense operands; stored bytes, not measured DRAM traffic",
                  reference=args.reference, cold_l2=args.cold_l2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fmt, op, k, n, side in shapes:
        programs, spills = {}, {}
        for name, arm in arms.items():
            with arm_source(name, arm, args.arm_bundle, fmt), exportable_compilation():
                programs[name] = compile_native(fmt, k, n, side)
            spills[name] = spill_check(programs[name], args.output.parent / "cubins" / name / f"{fmt}-{op}")
        # All widths use exactly the same input prefix and weights. Each arm's
        # own prefill freezes its K order, independently of other arms' ordering.
        capacity = 64 if fmt == "trellis" and max(rows_list) <= 64 else 4096
        storage, read_bytes = native_tensors(fmt, k, n, capacity, None if fmt == "trellis" else 0)
        tensors = storage[:5]
        columns = min(n, 256)
        if fmt == "trellis":
            weight = trellis_reference(k, n, columns, side, tensors[1], tensors[2])
        elif side == 1:
            payload = tensors[1].flatten()[:columns*k//(2 if fmt == "int4_g32" else 1)]
            scales = tensors[2][:columns if fmt == "fp8_row" else columns*k//32]
        else:
            payload = tensors[1].reshape(k//128, n, -1)[:, :columns].contiguous().flatten()
            scales = tensors[2].reshape(k//128, n, 4)[:, :columns].contiguous().flatten()
        if fmt != "trellis":
            weight = reference_weight(fmt, k, columns, side, payload, scales)
        x = storage[-1]
        oracle = x.double() @ weight.double().T
        bf16 = (x.bfloat16() @ weight.bfloat16().T).double()
        prefixes, global_errors = {}, {}
        def launch_scalars(rows):
            return (rows, 0, limit) if fmt == "trellis" else (rows, limit)

        invariance_all = {}
        for name, program in programs.items():
            program.launch(*tensors, scalars=launch_scalars(capacity))
            torch.cuda.synchronize()
            prefixes[name] = tensors[-1][:64].clone()
            global_errors[name] = torch.linalg.vector_norm(tensors[-1][:, :columns].double()-oracle).item()
            invariant = True
            for count in range(1, 65):
                tensors[-1].fill_(float("nan"))
                program.launch(*tensors, scalars=launch_scalars(count))
                torch.cuda.synchronize()
                invariant &= torch.equal(prefixes[name][:count].view(torch.int32), tensors[-1][:count].view(torch.int32))
                invariant &= bool(torch.isnan(tensors[-1][count:]).all())
            invariance_all[name] = invariant
        global_bound = torch.linalg.vector_norm(bf16-oracle).item()
        global_pass = {name: error <= 1.01*global_errors[accuracy_reference] and error <= 1.01*global_bound
                       for name, error in global_errors.items()}
        selected_reference = op_references.get(f"{fmt}/{op}", args.reference)
        for rows in rows_list:
            scalars = launch_scalars(rows)
            reference = programs[selected_reference]
            reference.launch(*tensors, scalars=scalars)
            torch.cuda.synchronize()
            expected = tensors[-1][:rows].clone()
            programs[accuracy_reference].launch(*tensors, scalars=scalars)
            torch.cuda.synchronize()
            reference_error = torch.linalg.vector_norm(tensors[-1][:rows, :columns].double()-oracle[:rows]).item()
            bound = torch.linalg.vector_norm(bf16[:rows]-oracle[:rows]).item()
            copies = copies_for(read_bytes, args.cold_l2 and rows <= 64)
            extra = [(tensors[1].clone(), tensors[2].clone()) for _ in range(copies-1)]
            base_us, base_error = None, None
            timed, keeps, records = {}, [], []
            repeats = args.replays if args.interleave_rounds and not profile else 1
            if not profile:
                try:
                    graph, keep, count = baseline(k, n, rows, a16=True, cold_l2=args.cold_l2, graph_repeats=repeats)
                except TypeError as error:
                    if "dense_gemm launch is unsupported" not in str(error):
                        raise
                    base_error = str(error)
                else:
                    if args.interleave_rounds:
                        timed["block_fp8"] = (graph, count * repeats)
                        keeps.append(keep)
                    else:
                        base_us = timing(graph, 1 if rows > 64 else args.replays)/count
                    del graph, keep
            for name, program in programs.items():
                tensors[-1].fill_(float("nan"))
                program.launch(*tensors, scalars=scalars)
                torch.cuda.synchronize()
                equal = torch.equal(expected.view(torch.int32), tensors[-1][:rows].view(torch.int32))
                prefix_rows = min(rows, 64)
                invariant = torch.equal(prefixes[name][:prefix_rows].view(torch.int32),
                                        tensors[-1][:prefix_rows].view(torch.int32))
                tail_untouched = torch.isnan(tensors[-1][rows:]).all().item()
                error = torch.linalg.vector_norm(tensors[-1][:rows, :columns].double()-oracle[:rows]).item()
                record = dict(arm=name, revision=arms[name]["revision"], format=fmt, projection=op,
                              rows=rows, bitwise_equal=equal, row_invariant=invariant and invariance_all[name],
                              row_invariance_checked=list(range(1, 65)),
                              output_tail_untouched=tail_untouched, frozen_reference=selected_reference,
                              fp64_error=error, bf16_error=bound, accuracy_reference_error=reference_error,
                              global_fp64_error=global_errors[name],
                              global_accuracy_reference_error=global_errors[accuracy_reference],
                              fp64_pass=global_pass[name] and error <= 1.01*bound and error <= 1.01*reference_error,
                              w8a16_us=base_us, w8a16_error=base_error, cubin_spill_check=spills[name])
                # Different K-order groups may be eligible; required arms must
                # additionally agree with the per-op frozen group reference.
                record["spill_free"] = all(not any(r["local_bytes"] + r["stack_bytes"]) and not r["local_load_store_instructions"] for r in spills[name])
                record["eligible"] = record["row_invariant"] and tail_untouched and record["spill_free"] and record["fp64_pass"] and (equal or name not in args.require_reference_arm)
                if record["eligible"]:
                    launches = [lambda p=program: p.launch(*tensors, scalars=scalars)]
                    launches += [lambda w=w, s=s, p=program: p.launch(tensors[0], w, s, *tensors[3:], scalars=scalars) for w, s in extra]
                    graph = capture(rotated(launches * repeats))
                    if profile:
                        if name == args.profile_arm:
                            torch.cuda.cudart().cudaProfilerStart()
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStop()
                    elif args.interleave_rounds:
                        timed[name] = (graph, copies * repeats)
                        if name == selected_reference:
                            timed["reference_control"] = (capture(rotated(launches * repeats)), copies * repeats)
                        record.update(weight_copies=copies, read_bytes=read_bytes)
                    else:
                        us = timing(graph, 1 if rows > 64 else args.replays)/copies
                        record.update(native_us=us, speedup_vs_w8a16=base_us/us if base_us else None,
                                      weight_copies=copies, read_bytes=read_bytes)
                    del graph
                records.append(record)
                report["measurements"].append(record)
                args.output.write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps(record), flush=True)
            if args.interleave_rounds and not profile:
                samples = {name: [] for name in timed}
                # Warm every graph before alternating. One replay contains all
                # repetitions, so host/WSL submission gaps are not multiplied.
                for graph, _ in timed.values():
                    graph.replay()
                torch.cuda.synchronize()
                order = list(timed)
                for iteration in range(args.interleave_rounds):
                    for name in timing_order(order, iteration):
                        graph, count = timed[name]
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / count)
                medians, spread, noise_pass = timing_summary(samples, selected_reference)
                reference_us = medians.get(selected_reference)
                for record in records:
                    name = record["arm"]
                    if name in medians:
                        us = medians[name]
                        record.update(native_us=us, native_samples_us=samples[name],
                                      w8a16_us=medians.get("block_fp8"),
                                      speedup_vs_w8a16=medians.get("block_fp8", 0)/us or None,
                                      speedup_vs_reference=reference_us/us if reference_us else None,
                                      identical_control_spread=spread,
                                      timing_noise_pass=noise_pass,
                                      interleaved_samples_us=samples, graph_repeats=repeats)
                        print(json.dumps(record), flush=True)
                args.output.write_text(json.dumps(report, indent=2)+"\n")
            del expected, extra, timed, keeps
        del storage, tensors, prefixes, weight, x, oracle, bf16
    failed = [m for m in report["measurements"] if not m["eligible"]]
    print(f"Wrote {len(report['measurements'])} measurements; {len(failed)} failed gates", flush=True)
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
