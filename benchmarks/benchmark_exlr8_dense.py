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
import ctypes

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_exlr8_a16 import capture, timing, read_roofline, finite_nonzero
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program, exportable_compilation
from b12x.integration.cuteafd import trellis_linear, scaled_linear, fp8_row_embedding
from b12x.integration.cuteafd._fp8_weights import Fp8Projection
from benchmarks._exlr8_dense_arms import parse_arms, materialize, verify, program_key, cold_copies

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
    runtime = ctypes.CDLL("libcudart.so.12")
    l2_bytes = ctypes.c_int()
    # cudaDevAttrL2CacheSize = 38; no probing another card or assumed SM count.
    status = runtime.cudaDeviceGetAttribute(ctypes.byref(l2_bytes), 38, torch.cuda.current_device())
    if status:
        raise RuntimeError(f"cudaDeviceGetAttribute(L2CacheSize) failed: {status}")
    return cold_copies(weight_bytes, l2_bytes.value)


def rotated(launches):
    """One graph launching each copy in turn; ``timing`` divides by ``len(launches)``."""
    def run():
        for launch in launches:
            launch()
    return run


def baseline(k, n, rows, a16=False, cold_l2=False):
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
    graph = capture(rotated(launches))
    finite_nonzero(output, "block-FP8")
    return graph, (tensors, extra), len(launches)


def compile_native(fmt, k, n, side):
    if fmt == "trellis":
        return trellis_linear.compile_trellis_linear_aot(input_features=k, output_features=n,
                                                       sharded_side=side)
    if fmt == "embedding":
        return fp8_row_embedding.compile_fp8_row_embedding_aot(hidden=k, vocabulary=n)
    return scaled_linear.compile_scaled_linear_aot(input_features=k, output_features=n,
                                                  encoding=fmt, sharded_side=side)


@contextlib.contextmanager
def arm_source(name, arm, bundle):
    """Patch only the imported dense class; compiled symbols and keys include its source."""
    import b12x.gemm.native_tile_linear as current
    path = bundle / arm["path"]
    spec = importlib.util.spec_from_file_location("dense_arm_" + name + "_" + arm["sha256"], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_class, original_compile = current.NativeTileLinear, scaled_linear.compile_program

    def compile_distinct(*args, **kwargs):
        kwargs["name"] += "_" + name + "_" + arm["sha256"][:16]
        kwargs["key"] = program_key(arm, kwargs["key"])
        return original_compile(*args, **kwargs)

    current.NativeTileLinear = module.NativeTileLinear
    scaled_linear.compile_program = compile_distinct
    try:
        yield
    finally:
        current.NativeTileLinear, scaled_linear.compile_program = original_class, original_compile


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
                            local_load_store_instructions=len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", action="append", required=True, help="NAME=git-rev; repeatable")
    parser.add_argument("--arm-bundle", type=Path, required=True)
    parser.add_argument("--materialize-only", action="store_true", help="host-side git show, no CUDA")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--formats", default="mxfp8,int4_g32,fp8_row")
    parser.add_argument("--shapes", help="comma-separated projection names")
    parser.add_argument("--rows", default="1,4,16,64,4096")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--profile-arm")
    parser.add_argument("--profile-op", help="FORMAT/OP")
    parser.add_argument("--profile-rows", type=int)
    args = parser.parse_args()
    arms = parse_arms(args.arm)
    if args.reference not in arms:
        parser.error("--reference must name an arm")
    if args.materialize_only:
        materialize(Path(__file__).resolve().parents[1], args.arm_bundle, arms)
        return
    arms = verify(args.arm_bundle, arms)
    if args.output is None:
        parser.error("--output is required for measurement")
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(r < 1 or r > 4096 for r in rows_list) or args.replays < 1:
        parser.error("rows must be in 1..4096 and replays positive")
    formats = set(args.formats.split(","))
    if formats - {"mxfp8", "int4_g32", "fp8_row"}:
        parser.error("multi-arm dense core supports mxfp8, int4_g32 and fp8_row")
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
            with arm_source(name, arm, args.arm_bundle), exportable_compilation():
                programs[name] = compile_native(fmt, k, n, side)
            spills[name] = spill_check(programs[name], args.output.parent / "cubins" / name / f"{fmt}-{op}")
        # One fixture per shape/row, shared by every arm and the frozen reference.
        for rows in rows_list:
            storage, read_bytes = native_tensors(fmt, k, n, rows, 0)
            tensors = storage[:5]
            scalars = (rows, limit)
            reference = programs[args.reference]
            reference.launch(*tensors, scalars=scalars)
            torch.cuda.synchronize()
            expected = tensors[-1].clone()
            # Bound FP64 work at the largest vocabulary shape, retaining full K.
            columns = min(n, 256)
            if side == 1:
                payload, scales = tensors[1][:columns*k//(2 if fmt == "int4_g32" else 1)], tensors[2][:columns if fmt == "fp8_row" else columns*k//32]
            else:
                payload = tensors[1].reshape(k//128, n, -1)[:, :columns].contiguous().flatten()
                scales = tensors[2].reshape(k//128, n, 4)[:, :columns].contiguous().flatten()
            weight = reference_weight(fmt, k, columns, side, payload, scales)
            x = storage[-1][:min(rows, 64)]
            oracle = x.double() @ weight.double().T
            bf16 = (x @ weight.bfloat16().T).double()
            bound = torch.linalg.vector_norm(bf16 - oracle).item()
            copies = copies_for(read_bytes, args.cold_l2 and rows <= 64)
            extra = [(tensors[1].clone(), tensors[2].clone()) for _ in range(copies-1)]
            base_us, base_error = None, None
            if not profile:
                try:
                    graph, keep, count = baseline(k, n, rows, a16=True, cold_l2=args.cold_l2)
                except TypeError as error:
                    if "dense_gemm launch is unsupported" not in str(error):
                        raise
                    base_error = str(error)
                else:
                    base_us = timing(graph, 1 if rows > 64 else args.replays)/count
                    del graph, keep
            for name, program in programs.items():
                tensors[-1].fill_(float("nan"))
                program.launch(*tensors, scalars=scalars)
                torch.cuda.synchronize()
                equal = torch.equal(expected.view(torch.int32), tensors[-1].view(torch.int32))
                error = torch.linalg.vector_norm(tensors[-1][:len(x), :columns].double()-oracle).item()
                record = dict(arm=name, revision=arms[name]["revision"], format=fmt, projection=op,
                              rows=rows, bitwise_equal=equal, fp64_error=error, bf16_error=bound,
                              fp64_pass=error <= 1.01*bound, w8a16_us=base_us,
                              w8a16_error=base_error, cubin_spill_check=spills[name])
                # A failed arm is reported but never receives a performance result.
                if equal and record["fp64_pass"]:
                    launches = [lambda p=program: p.launch(*tensors, scalars=scalars)]
                    launches += [lambda w=w, s=s, p=program: p.launch(tensors[0], w, s, *tensors[3:], scalars=scalars) for w, s in extra]
                    graph = capture(rotated(launches))
                    if profile:
                        if name == args.profile_arm:
                            torch.cuda.cudart().cudaProfilerStart()
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStop()
                    else:
                        us = timing(graph, 1 if rows > 64 else args.replays)/copies
                        record.update(native_us=us, speedup_vs_w8a16=base_us/us if base_us else None,
                                      weight_copies=copies, read_bytes=read_bytes)
                    del graph
                report["measurements"].append(record)
                args.output.write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps(record), flush=True)
            del storage, tensors, expected, extra, weight, x, oracle, bf16
    failed = [m for m in report["measurements"] if not m["bitwise_equal"] or not m["fp64_pass"]]
    print(f"Wrote {len(report['measurements'])} measurements; {len(failed)} failed gates", flush=True)
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
