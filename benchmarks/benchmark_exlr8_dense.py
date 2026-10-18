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

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_exlr8_a16 import capture, timing, read_roofline, finite_nonzero
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program, exportable_compilation
from b12x.integration.cuteafd import trellis_linear, scaled_linear, fp8_row_embedding
from b12x.integration.cuteafd._fp8_weights import Fp8Projection

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
    def __init__(self, k, n, prefill):
        self.projection = Fp8Projection(n, k, prefill_rows=4096 if prefill else None)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scales: cute.Pointer,
                 scratch: cute.Pointer, output: cute.Pointer, rows: cutlass.Int32,
                 stream: cuda.CUstream):
        self.projection(x, w, scales, output, rows, cutlass.Int32(255),
                        cutlass.Int64(scratch.toint()), stream)


def baseline(k, n, rows):
    core = BlockFp8(k, n, rows > 64)
    ops = (Operand("x", torch.bfloat16, "[rows,K]"),
           Operand("w", torch.float8_e4m3fn, "[N,K]"),
           Operand("scales", torch.float32, "[ceil(N/128),K/128]"),
           Operand("scratch", torch.uint8, "[bytes]", "scratch"),
           Operand("output", torch.bfloat16, "[rows,N]", "out"))
    program = compile_program(core, name="bench_dense_block_fp8", operands=ops,
                              scalars=(Scalar("rows"),), key=(k, n, rows > 64))
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16) * .1
    weight = torch.randint(-16, 17, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full(((n+127)//128, k//128), .01, device="cuda")
    scratch = torch.empty(max(1024, core.projection.prefill_scratch_bytes(rows)),
                          dtype=torch.uint8, device="cuda")
    output = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    tensors = (x, weight, scales, scratch, output)
    graph = capture(lambda: program.launch(*tensors, scalars=(rows,)))
    finite_nonzero(output, "block-FP8")
    return graph, tensors


def compile_native(fmt, k, n, side, control=False):
    module = {"trellis": trellis_linear, "embedding": fp8_row_embedding}.get(fmt, scaled_linear)
    original = module.compile_program

    def distinct(*args, **kwargs):
        kwargs["name"] += "_bench_control"
        return original(*args, **kwargs)

    try:
        if control:
            module.compile_program = distinct
        if fmt == "trellis":
            return module.compile_trellis_linear_aot(input_features=k, output_features=n,
                                                     sharded_side=side)
        if fmt == "embedding":
            return module.compile_fp8_row_embedding_aot(hidden=k, vocabulary=n)
        return module.compile_scaled_linear_aot(input_features=k, output_features=n,
                                                encoding=fmt, sharded_side=side)
    finally:
        module.compile_program = original


def native_tensors(fmt, k, n, rows):
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
    partial = torch.empty((rows, (k+1023)//1024, n), dtype=torch.float32, device="cuda")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formats", default="trellis,mxfp8,int4_g32,fp8_row,embedding")
    parser.add_argument("--shapes", help="comma-separated projection names")
    parser.add_argument("--rows", default="1,4,16,64,4096")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="retain completed measurements and run only missing rows")
    args = parser.parse_args()
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(r < 1 or r > 4096 for r in rows_list) or args.replays < 1:
        parser.error("rows must be in 1..4096 and replays positive")
    formats = set(args.formats.split(","))
    if formats - {s[0] for s in SHAPES}:
        parser.error("unknown format")
    names = set(args.shapes.split(",")) if args.shapes else {s[1] for s in SHAPES}
    shapes = [s for s in SHAPES if s[0] in formats and s[1] in names]
    if not shapes:
        parser.error("empty shape selection")
    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    cta_limit = props.multi_processor_count * 4
    report = dict(device=props.name, sms=props.multi_processor_count,
                  read_ceiling=read_roofline(), measurements=[],
                  scope="synthetic native operand only; rotations/output transforms excluded",
                  throughput="stored weight+scale bytes read once; not DRAM traffic counters",
                  fp8_comparison="same shape/rows, synthetic FP8 weights, not quality equivalence",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    session = {key: value for key, value in report.items() if key != "measurements"}
    if args.resume:
        previous = json.loads(args.output.read_text())
        if (previous["device"], previous["sms"]) != (report["device"], report["sms"]):
            raise RuntimeError("resume requires the same GPU model and SM count")
        previous.setdefault("sessions", [{key: value for key, value in previous.items()
                                          if key not in ("measurements", "sessions")}])
        report = previous
    else:
        report["sessions"] = []
    session_id = len(report["sessions"])
    report["sessions"].append(session)
    ceiling = session["read_ceiling"]["gbps"]
    complete = {(m["format"], m["projection"], m["rows"]) for m in report["measurements"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fmt, name, k, n, side in shapes:
        missing = [rows for rows in rows_list if (fmt, name, rows) not in complete]
        if not missing:
            continue
        with exportable_compilation():
            program = compile_native(fmt, k, n, side)
        spill = spill_check(program, args.output.parent / "cubins" / f"{fmt}-{name}")
        control = compile_native(fmt, k, n, side, control=True)
        for rows in missing:
            storage, read_bytes = native_tensors(fmt, k, n, rows)
            tensors = storage if fmt == "embedding" else storage[:5]
            scalars = (rows, 2, cta_limit) if fmt == "trellis" else (rows, cta_limit)
            graph = capture(lambda: program.launch(*tensors, scalars=scalars))
            finite_nonzero(tensors[-1], fmt)
            expected = tensors[-1].clone()
            control_graph = capture(lambda: control.launch(*tensors, scalars=scalars))
            for tensor in tensors[3:]:
                tensor.fill_(float("nan"))
            control_graph.replay()
            if not torch.equal(expected.view(torch.int16), tensors[-1].view(torch.int16)):
                raise RuntimeError(f"{fmt}/{name}/rows{rows}: same-session control mismatch")
            replays = 1 if rows > 64 else args.replays
            native_us = timing(graph, replays)
            control_us = timing(control_graph, replays)
            fp8_us = None
            fp8_status = "not_applicable" if fmt == "embedding" else "ok"
            fp8_error = None
            if fmt != "embedding":
                try:
                    fp8_graph, fp8_storage = baseline(k, n, rows)
                except TypeError as error:
                    if "dense_gemm launch is unsupported" not in str(error):
                        raise
                    fp8_status, fp8_error = "unsupported", str(error)
                else:
                    fp8_us = timing(fp8_graph, replays)
                    del fp8_graph, fp8_storage
            gbps = read_bytes/native_us/1000
            row = dict(format=fmt, projection=name, k=k, n=n, sharded_side=side,
                       rows=rows, read_bytes=read_bytes, native_us=native_us,
                       gbps=gbps, read_ceiling_percent=100*gbps/ceiling,
                       target_met=gbps >= .95*ceiling,
                       session_id=session_id, read_ceiling_gbps=ceiling,
                       control_us=control_us, control_bitwise_equal=True, cubin_spill_check=spill,
                       block_fp8_status=fp8_status, block_fp8_error=fp8_error,
                       block_fp8_us=fp8_us, speedup_vs_block_fp8=fp8_us/native_us if fp8_us else None)
            report["measurements"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(row), flush=True)
            del graph, control_graph, expected, tensors, storage
    print(f"Wrote {len(report['measurements'])} measurements", flush=True)


if __name__ == "__main__":
    main()
