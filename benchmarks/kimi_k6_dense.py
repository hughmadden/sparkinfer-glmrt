"""Same-process exported shared-ABI K6 versus canonical engine BF16 projections.

This deliberately does not call the general trellis GEMM. Random native payloads
measure the projection; --transforms adds the exported H128 ingress and egress.
Decode samples evict L2 outside the timed region. Each arm runs through its
exported C wrapper, captured once before timing; sampling order alternates each
round. Poisoned-output replay guards reject empty or wrong-stream captures.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


def exported(program, directory, stem):
    from b12x.integration.cuteafd import validate_exported_header
    program.export_to_c(str(directory), stem, stem)
    validate_exported_header(program, directory / f"{stem}.h", stem)
    pointers = [f"void *p{i}" for i in range(len(program.operands))]
    scalars = [f"int32_t s{i}" for i in range(len(program.scalars))]
    arguments = [f"p{i}" for i in range(len(pointers))] + [f"s{i}" for i in range(len(scalars))]
    source = directory / f"{stem}_bridge.cc"
    source.write_text(f'#include "{stem}.h"\n'
        f'static {stem}_Kernel_Module_t module;\n'
        f'extern "C" void load() {{ {stem}_Kernel_Module_Load(&module); }}\n'
        f'extern "C" int run({", ".join(pointers + scalars + ["cudaStream_t stream"])}) {{\n'
        f'  return cute_dsl_{stem}_wrapper(&module, {", ".join(arguments)}, stream);\n}}\n')
    runtime = Path(subprocess.check_output(
        [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True).strip())
    library = directory / f"{stem}.so"
    subprocess.run(["g++", "-shared", "-fPIC", "-std=c++17", "-I/usr/local/cuda/include",
        str(source), str(directory / f"{stem}.o"), "-L/usr/local/cuda/lib64", "-lcudart",
        f"-L{runtime}", "-lcute_dsl_runtime", f"-Wl,-rpath,{runtime}", "-Wl,-z,defs",
        "-o", str(library)], check=True)
    lib = C.CDLL(str(library))
    lib.load.argtypes, lib.load.restype = [], None
    lib.run.argtypes = [C.c_void_p] * len(pointers) + [C.c_int32] * len(scalars) + [C.c_void_p]
    lib.run.restype = C.c_int32
    lib.load()
    return lib


# Unique stored geometries from Core-K6 headers and the engine's TP2 ownership.
# MLA kv_a is logically 576 outputs, but both routes must reconstruct all 640.
PREFILL_TP2_SHAPES = (
    (7168, 6144, 1, 4), (6144, 7168, 2, 1),
    (7168, 1536, 1, 1), (1536, 9216, 1, 1), (7168, 640, 1, 1),
    (7168, 3072, 1, 1), (3072, 7168, 2, 1),
    (7168, 3584, 1, 1), (3584, 7168, 1, 1),
)


def benchmark_prefill_route(args):
    """Exact exported prefill routes; every replay uses preallocated storage."""
    import torch
    import cutlass.cute as cute
    from cutlass import Int32
    from cuda.bindings import driver as cuda
    from b12x.gemm.trellis_linear import _native
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from b12x.integration.cuteafd.trellis_dequant import compile_trellis_dequant_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot,
    )
    from benchmarks.benchmark_exlr8_dense import spill_check

    class BaseOracle(_native.NativeTrellisLinear):
        @cute.jit
        def __call__(self, inputs: cute.Tensor, payload: cute.Tensor,
                     scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                     rows: Int32, rotation_id: Int32, cta_limit: Int32,
                     stream: cuda.CUstream):
            if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations) and cta_limit > Int32(0):
                tiles = Int32(self.n // 128 * self.splits)
                self.multiply(inputs, payload, scales, partial, rows, rotation_id).launch(
                    grid=(min(((rows + Int32(15)) // Int32(16)) * tiles, cta_limit), 1, 1),
                    block=(128, 1, 1), min_blocks_per_mp=self.min_blocks_per_mp,
                    stream=stream)
                self.reduce(partial, output, rows).launch(
                    grid=(min(rows * Int32(self.n // 128), cta_limit), 1, 1),
                    block=(128, 1, 1), stream=stream)

    def prefill_resources(program, directory):
        records = spill_check(program, directory)
        selected = []
        # One export also contains the unchanged narrow-row decode entry points.
        # Gate the candidate prefill entry, retaining the full census as evidence.
        for index, record in enumerate(records):
            sass = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-sass",
                str(directory / f"native-{index}.cubin")], text=True)
            for section in re.split(r"(?=\s*Function :)", sass):
                if "multiply_prefill" in section and re.search(r"\b(?:LDL|STL)(?:\.|\s)", section):
                    raise RuntimeError("T2 prefill has local load/store instructions")
            for name, fields in re.findall(r"Function\s+([^:]+):\s*([^\n]+)", record["resources"]):
                if "multiply_prefill" in name:
                    usage = {key: int(value) for key, value in re.findall(r"(REG|STACK|LOCAL):(\d+)", fields)}
                    if usage.get("STACK", -1) != 0 or usage.get("LOCAL", -1) != 0:
                        raise RuntimeError(f"T2 prefill spills: {name}: {fields}")
                    selected.append(dict(name=name, **usage))
        if not selected:
            raise RuntimeError("missing prefill resource census")
        return dict(prefill=selected, full_census=records)

    rows_list = [int(r) for r in args.rows.split(",")]
    if args.samples < 1 or not rows_list or min(rows_list) < 1:
        raise ValueError("positive rows and sample count required")
    shapes = PREFILL_TP2_SHAPES if args.shapes is None else [
        tuple(int(v) for v in shape.split(":")) for shape in args.shapes.split(",")]
    capacity = max(rows_list)
    torch.manual_seed(273)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    limit = props.multi_processor_count * 4
    report = dict(metadata=dict(device=props.name, sms=props.multi_processor_count,
        codebook="mcg", bits=6, rows=rows_list, samples=args.samples,
        transforms_included=args.transforms, gemm_output="f32",
        baseline="chunk_dequant_bf16 + canonical BF16 GEMM + chunk stitch",
        ratio="route / t2; greater than one is faster", argv=sys.argv,
        candidate="Gemini T2 8f3e4c9f", records_scope="synthetic served TP2 geometries"),
        records=[])
    for shape in shapes:
        k, n, side = shape[:3]
        grouped = shape[3] == 4 if len(shape) > 3 else (k, n, side) == (7168, 6144, 1)
        directory = args.output_dir / f"k{k}_n{n}_s{side}"
        directory.mkdir(parents=True, exist_ok=True)
        print(f"compiling prefill route and T2: {k}:{n}:{side}", flush=True)
        columns = n if k * n * 2 <= 256 * 1024 * 1024 else 128
        with exportable_compilation():
            t2_program = compile_trellis_linear_aot(input_features=k, output_features=n,
                bits=6, codebook="mcg", sharded_side=side, rotation_count=1, max_rows=capacity)
            dequant = compile_trellis_dequant_aot(input_features=k, output_features=n,
                chunk_columns=columns, codebook="mcg", sharded_side=side)
            gemm = compile_bf16_projection_aot(width=k, outputs=columns, output_dtype=torch.float32)
        t2 = exported(t2_program, directory, "t2")
        dequant = exported(dequant, directory, "dequant")
        gemm = exported(gemm, directory, "gemm_f32")
        resources = [prefill_resources(t2_program, directory / "resources")]
        original = _native.NativeTrellisLinear
        _native.NativeTrellisLinear = BaseOracle
        try:
            with exportable_compilation():
                oracle = compile_trellis_linear_aot(input_features=k, output_features=n,
                    bits=6, codebook="mcg", sharded_side=side, rotation_count=1, max_rows=capacity)
        finally:
            _native.NativeTrellisLinear = original
        oracle = exported(oracle, directory, "base_oracle")
        members = 4 if grouped else 1
        source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
        xs = [source.half() for _ in range(members)]
        wide_xs = [source.clone() for _ in range(members)]
        tables = [torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64) for x in xs]
        payloads = [torch.randint(256, (k*n*6//8,), device="cuda", dtype=torch.uint8) for _ in xs]
        scales = [torch.ones(k//128, device="cuda", dtype=torch.float16) for _ in xs]
        suhs = [torch.randn(k, device="cuda", dtype=torch.float16) * .2 for _ in xs]
        svhs = [torch.randn(n, device="cuda", dtype=torch.float16) * .2 for _ in xs]
        outputs = [torch.empty(capacity, n, device="cuda", dtype=torch.float32) for _ in xs]
        route_outputs = [torch.empty_like(out) for out in outputs]
        final_outputs = [torch.empty_like(out) for out in outputs]
        route_finals = [torch.empty_like(out) for out in outputs]
        partial = torch.empty(members * capacity * ((k+1023)//1024) * n, device="cuda")
        weights = torch.empty(columns, k, device="cuda", dtype=torch.bfloat16)
        chunk_out = torch.empty(capacity, columns, device="cuda", dtype=torch.float32)
        oracle_out = torch.empty_like(outputs[0])
        ingress = wide_ingress = egress = None
        if args.transforms:
            with exportable_compilation():
                ingress = compile_trellis_input_aot(features=k, max_rows=capacity)
                wide_ingress = compile_trellis_input_aot(features=k, max_rows=capacity, output_dtype="bf16")
                egress = compile_trellis_output_aot(features=n, max_rows=capacity, output_dtype="f32")
            ingress = exported(ingress, directory, "ingress_f16")
            wide_ingress = exported(wide_ingress, directory, "ingress_bf16")
            egress = exported(egress, directory, "egress_f32")
        if grouped:
            with exportable_compilation():
                grouped_program = compile_trellis_linear_aot(input_features=k, output_features=n,
                    bits=6, codebook="mcg", sharded_side=side, rotation_count=4, max_rows=capacity, groups=4)
            grouped_t2 = exported(grouped_program, directory, "t2_qkvg")
            resources.append(prefill_resources(grouped_program, directory / "resources_grouped"))
            grouped_table = torch.tensor([x.data_ptr() for x in xs], device="cuda", dtype=torch.int64)
            if args.transforms:
                with exportable_compilation():
                    grouped_ingress = compile_trellis_input_aot(features=k, max_rows=capacity, groups=4)
                    grouped_egress = compile_trellis_output_aot(features=n, max_rows=capacity, output_dtype="f32", groups=4)
                grouped_ingress = exported(grouped_ingress, directory, "ingress_grouped")
                grouped_egress = exported(grouped_egress, directory, "egress_grouped")

        def call(lib, *values):
            result = lib.run(*values, torch.cuda.current_stream().cuda_stream)
            if result:
                raise RuntimeError(f"exported wrapper returned {result}")

        # One bounded BF16 scratch allocation, reused after every serialized GEMM.
        def route_core(rows, i, component="route"):
            for first in range(0, n, columns):
                if component != "gemm":
                    call(dequant, payloads[i].data_ptr(), scales[i].data_ptr(), weights.data_ptr(), first)
                if component != "dequant":
                    call(gemm, wide_xs[i].data_ptr(), weights.data_ptr(), chunk_out.data_ptr(), rows)
                    route_outputs[i][:rows, first:first+columns].copy_(chunk_out[:rows])

        def single(rows, i, native):
            if args.transforms:
                call(ingress if native else wide_ingress, source.data_ptr(), suhs[i].data_ptr(),
                    (xs if native else wide_xs)[i].data_ptr(), scales[i].data_ptr(), rows, limit)
            if native:
                call(t2, tables[i].data_ptr(), payloads[i].data_ptr(), scales[i].data_ptr(),
                    partial.data_ptr(), outputs[i].data_ptr(), rows, 0, limit)
            else:
                route_core(rows, i)
            if args.transforms:
                call(egress, (outputs if native else route_outputs)[i].data_ptr(), svhs[i].data_ptr(),
                    (final_outputs if native else route_finals)[i].data_ptr(), rows, limit)

        def run_grouped(rows):
            if args.transforms:
                call(grouped_ingress, source.data_ptr(), *(s.data_ptr() for s in suhs),
                    *(x.data_ptr() for x in xs), *(s.data_ptr() for s in scales), rows, limit)
            call(grouped_t2, grouped_table.data_ptr(), *(p.data_ptr() for p in payloads),
                *(s.data_ptr() for s in scales), partial.data_ptr(), *(o.data_ptr() for o in outputs), rows, 0, limit)
            if args.transforms:
                call(grouped_egress, *(o.data_ptr() for o in outputs), *(s.data_ptr() for s in svhs),
                    *(o.data_ptr() for o in final_outputs), rows, limit)

        for rows in rows_list:
            single(rows, 0, True)
            expected = outputs[0][:rows].clone()
            call(oracle, tables[0].data_ptr(), payloads[0].data_ptr(), scales[0].data_ptr(),
                partial.data_ptr(), oracle_out.data_ptr(), rows, 0, limit)
            if not torch.equal(expected.view(torch.int32), oracle_out[:rows].view(torch.int32)):
                raise RuntimeError(f"T2 is not bitwise equal to base native: {shape}, rows={rows}")
            ragged = max(1, rows-13)
            call(t2, tables[0].data_ptr(), payloads[0].data_ptr(), scales[0].data_ptr(),
                partial.data_ptr(), oracle_out.data_ptr(), ragged, 0, limit)
            if not torch.equal(expected[:ragged].view(torch.int32), oracle_out[:ragged].view(torch.int32)):
                raise RuntimeError("T2 is not exactly row invariant")
            # Cross the old narrow/wide and the new prefill dispatch boundaries.
            for prefix in (1, 16, 32, 48, 64, 128):
                if prefix <= rows:
                    call(t2, tables[0].data_ptr(), payloads[0].data_ptr(), scales[0].data_ptr(),
                        partial.data_ptr(), oracle_out.data_ptr(), prefix, 0, limit)
                    if not torch.equal(expected[:prefix].view(torch.int32), oracle_out[:prefix].view(torch.int32)):
                        raise RuntimeError(f"T2 row-invariance failure at prefix {prefix}")
            runners = dict(t2=lambda: single(rows, 0, True), route=lambda: single(rows, 0, False),
                dequant=lambda: route_core(rows, 0, "dequant"), gemm=lambda: route_core(rows, 0, "gemm"))
            targets = dict(t2=[(final_outputs if args.transforms else outputs)[0][:rows]],
                route=[(route_finals if args.transforms else route_outputs)[0][:rows]],
                dequant=[weights], gemm=[route_outputs[0][:rows]])
            if grouped:
                def four_routes():
                    for i in range(4):
                        single(rows, i, False)
                def four_native():
                    for i in range(4):
                        single(rows, i, True)
                four_native()
                expected_group = [o[:rows].clone() for o in outputs]
                run_grouped(rows)
                for got, ref in zip(outputs, expected_group):
                    if not torch.equal(got[:rows].view(torch.int32), ref.view(torch.int32)):
                        raise RuntimeError("grouped T2 differs bitwise from four single kernels")
                runners.update(t2_qkvg=lambda: run_grouped(rows), route_4x=four_routes)
                targets.update(t2_qkvg=[o[:rows] for o in (final_outputs if args.transforms else outputs)],
                    route_4x=[o[:rows] for o in (route_finals if args.transforms else route_outputs)])
            graphs = {}
            for name, run in runners.items():
                run()  # Untimed first-use warmup for this shape and concurrency.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for target in targets[name]:
                    target.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not all(torch.isfinite(t).all() and torch.count_nonzero(t) for t in targets[name]):
                    raise RuntimeError(f"{name} graph failed poisoned/nonzero-output replay")
                graphs[name] = graph
            samples = {name: [] for name in graphs}
            for repeat in range(args.samples):
                order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            record = dict(k=k, n=n, side=side, rows=rows, us=medians, samples_us=samples,
                speedup=medians["route"]/medians["t2"], component_sum_us=medians["dequant"]+medians["gemm"],
                bitwise_base=True, row_invariant=True, resources=resources)
            if grouped:
                record["grouped_speedup"] = medians["route_4x"]/medians["t2_qkvg"]
                record["grouped_bitwise"] = True
            report["records"].append(record)
            print(json.dumps({key: value for key, value in record.items() if key not in ("samples_us", "resources")}), flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        # Drop graph closures before their caller-owned tensor storage.
        del graphs, runners, targets, single, route_core, run_grouped
        del xs, wide_xs, payloads, outputs, route_outputs, final_outputs, route_finals
        del partial, oracle_out, expected, weights, chunk_out, source, tables, scales, suhs, svhs
        if grouped:
            del expected_group, grouped_table, four_routes, four_native
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", help="K:N:side[:groups] list; --prefill-route defaults to audited TP2 shapes")
    parser.add_argument("--rows", default="1,64,256")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--transforms", action="store_true", help="include actual exported H128 input/output transforms")
    parser.add_argument("--prefill-route", action="store_true", help="MCG T2 versus dequant + F32-output BF16 GEMM, including grouped KDA")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prefill_route:
        benchmark_prefill_route(args)
        return
    if args.shapes is None:
        args.shapes = "7168:6144:1,7168:163840:1"
    import torch
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    torch.manual_seed(273)
    device = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
    if int(status):
        raise RuntimeError(status)
    flush = torch.empty(2 * l2, device="cuda", dtype=torch.uint8)
    rows_list = [int(x) for x in args.rows.split(",")]
    records = []
    metadata = dict(device=torch.cuda.get_device_name(), sms=sms, l2_bytes=l2,
                    samples=args.samples, kernel_path="exported shared trellis ABI v1",
                    transforms_included=args.transforms, decode_cache="cold, 2x runtime L2 bytes")
    for shape in args.shapes.split(","):
        parts = [int(x) for x in shape.split(":")]
        k, n, side = parts[:3]
        grouped = (parts[3] == 4) if len(parts) > 3 else (k == 7168 and n == 6144 and side == 1)
        directory = args.output_dir / f"k{k}_n{n}_s{side}"
        directory.mkdir(exist_ok=True)
        capacity = max(rows_list)
        splits = (k + 1023) // 1024
        x_bf16 = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
        x = x_bf16.half()
        inputs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
        payload = torch.randint(256, (k*n*6//8,), device="cuda", dtype=torch.uint8)
        scales = torch.ones(k//128, device="cuda", dtype=torch.float16)
        partial = torch.empty(capacity*splits*n, device="cuda", dtype=torch.float32)
        output = torch.empty(capacity*n, device="cuda", dtype=torch.float32)
        bf16_output = torch.empty(capacity*n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        arms = {}
        suh=torch.ones(k,device="cuda",dtype=torch.float16)
        svh=torch.ones(n,device="cuda",dtype=torch.float16)
        out_dtype = torch.float32 if n==163840 else torch.bfloat16
        transformed_output=torch.empty(capacity*n,device="cuda",dtype=out_dtype)
        if args.transforms:
            with exportable_compilation():
                ingress=compile_trellis_input_aot(features=k,max_rows=capacity)
                egress=compile_trellis_output_aot(features=n,output_dtype="f32" if n==163840 else "bf16",max_rows=capacity)
            ingress=exported(ingress,directory,"input_transform")
            egress=exported(egress,directory,"output_transform")
        for codebook in ("mcg", "mul1", "bf16"):
            print(f"compiling {shape} {codebook}", flush=True)
            with exportable_compilation():
                program = compile_bf16_projection_aot(width=k, outputs=n) if codebook == "bf16" else \
                    compile_trellis_linear_aot(input_features=k, output_features=n, bits=6,
                        codebook=codebook, sharded_side=side, rotation_count=1, max_rows=capacity)
            arms[codebook] = exported(program, directory, codebook)
        if grouped:
            qkvg_suh = [torch.randn(k, device="cuda", dtype=torch.float16) * 0.2 for _ in range(4)]
            qkvg_svh = [torch.randn(n, device="cuda", dtype=torch.float16) * 0.2 for _ in range(4)]
            qkvg_x_4x = [(x_bf16.float() * s.float()).half() for s in qkvg_suh]
            qkvg_x_grp = [t.clone() for t in qkvg_x_4x]
            qkvg_inputs_4x = [torch.tensor([t.data_ptr()], device="cuda", dtype=torch.int64) for t in qkvg_x_4x]
            qkvg_inputs_grp = torch.tensor([t.data_ptr() for t in qkvg_x_grp], device="cuda", dtype=torch.int64)
            qkvg_payloads = [torch.randint(256, (k*n*6//8,), device="cuda", dtype=torch.uint8) for _ in range(4)]
            qkvg_scales_4x = [torch.ones(k//128, device="cuda", dtype=torch.float16) * (0.75 + 0.25 * i) for i in range(4)]
            qkvg_scales_grp = [t.clone() for t in qkvg_scales_4x]
            qkvg_partial = torch.empty(4*capacity*splits*n, device="cuda", dtype=torch.float32)
            qkvg_out_4x = [torch.empty(capacity*n, device="cuda", dtype=torch.float32) for _ in range(4)]
            qkvg_out_grp = [torch.empty(capacity*n, device="cuda", dtype=torch.float32) for _ in range(4)]
            qkvg_trans_4x = [torch.empty(capacity*n, device="cuda", dtype=out_dtype) for _ in range(4)]
            qkvg_trans_grp = [torch.empty(capacity*n, device="cuda", dtype=out_dtype) for _ in range(4)]
            if args.transforms:
                with exportable_compilation():
                    grp_in = compile_trellis_input_aot(features=k, max_rows=capacity, groups=4)
                    grp_out = compile_trellis_output_aot(features=n, output_dtype="f32" if n==163840 else "bf16", max_rows=capacity, groups=4)
                grp_ingress = exported(grp_in, directory, "input_transform_qkvg")
                grp_egress = exported(grp_out, directory, "output_transform_qkvg")
            grp_libs = {}
            for codebook in ("mcg", "mul1"):
                print(f"compiling {shape} {codebook}_qkvg", flush=True)
                with exportable_compilation():
                    program = compile_trellis_linear_aot(
                        input_features=k, output_features=n, bits=6, codebook=codebook,
                        sharded_side=side, rotation_count=4, max_rows=capacity, groups=4)
                grp_libs[codebook] = exported(program, directory, f"{codebook}_qkvg")
        for rows in rows_list:
            runners = {}
            targets = {}
            for name, lib in arms.items():
                values = [x_bf16.data_ptr(), weight.data_ptr(), bf16_output.data_ptr(), rows] if name == "bf16" else \
                    [t.data_ptr() for t in (inputs, payload, scales, partial, output)] + [rows, 0, sms*4]
                def run(lib=lib, values=values, name=name):
                    stream=torch.cuda.current_stream().cuda_stream
                    if args.transforms and name != "bf16":
                        result=ingress.run(x_bf16.data_ptr(),suh.data_ptr(),x.data_ptr(),scales.data_ptr(),rows,sms*4,stream)
                        if result:
                            raise RuntimeError(f"input transform returned {result}")
                    result = lib.run(*values, stream)
                    if result:
                        raise RuntimeError(f"exported wrapper returned {result}")
                    if args.transforms and name != "bf16":
                        result=egress.run(output.data_ptr(),svh.data_ptr(),transformed_output.data_ptr(),rows,sms*4,stream)
                        if result:
                            raise RuntimeError(f"output transform returned {result}")
                runners[name] = run
                targets[name] = [bf16_output[:rows*n] if name=="bf16" else transformed_output[:rows*n] if args.transforms else output[:rows*n]]
            if grouped:
                for codebook in ("mcg", "mul1"):
                    lib = arms[codebook]
                    grp_lib = grp_libs[codebook]
                    def run_4x(lib=lib):
                        stream = torch.cuda.current_stream().cuda_stream
                        for i in range(4):
                            if args.transforms:
                                res = ingress.run(x_bf16.data_ptr(), qkvg_suh[i].data_ptr(), qkvg_x_4x[i].data_ptr(), qkvg_scales_4x[i].data_ptr(), rows, sms*4, stream)
                                if res:
                                    raise RuntimeError(f"input transform 4x[{i}] returned {res}")
                            res = lib.run(qkvg_inputs_4x[i].data_ptr(), qkvg_payloads[i].data_ptr(), qkvg_scales_4x[i].data_ptr(), partial.data_ptr(), qkvg_out_4x[i].data_ptr(), rows, 0, sms*4, stream)
                            if res:
                                raise RuntimeError(f"exported wrapper 4x[{i}] returned {res}")
                            if args.transforms:
                                res = egress.run(qkvg_out_4x[i].data_ptr(), qkvg_svh[i].data_ptr(), qkvg_trans_4x[i].data_ptr(), rows, sms*4, stream)
                                if res:
                                    raise RuntimeError(f"output transform 4x[{i}] returned {res}")
                    def run_qkvg(grp_lib=grp_lib):
                        stream = torch.cuda.current_stream().cuda_stream
                        if args.transforms:
                            res = grp_ingress.run(x_bf16.data_ptr(), *(s.data_ptr() for s in qkvg_suh), *(r.data_ptr() for r in qkvg_x_grp), *(s.data_ptr() for s in qkvg_scales_grp), rows, sms*4, stream)
                            if res:
                                raise RuntimeError(f"grouped input transform returned {res}")
                        res = grp_lib.run(qkvg_inputs_grp.data_ptr(), *(p.data_ptr() for p in qkvg_payloads), *(s.data_ptr() for s in qkvg_scales_grp), qkvg_partial.data_ptr(), *(o.data_ptr() for o in qkvg_out_grp), rows, 0, sms*4, stream)
                        if res:
                            raise RuntimeError(f"grouped exported wrapper returned {res}")
                        if args.transforms:
                            res = grp_egress.run(*(o.data_ptr() for o in qkvg_out_grp), *(s.data_ptr() for s in qkvg_svh), *(t.data_ptr() for t in qkvg_trans_grp), rows, sms*4, stream)
                            if res:
                                raise RuntimeError(f"grouped output transform returned {res}")
                    runners[f"{codebook}_4x"] = run_4x
                    runners[f"{codebook}_qkvg"] = run_qkvg
                    targets[f"{codebook}_4x"] = [t[:rows*n] for t in (qkvg_trans_4x if args.transforms else qkvg_out_4x)]
                    targets[f"{codebook}_qkvg"] = [t[:rows*n] for t in (qkvg_trans_grp if args.transforms else qkvg_out_grp)]
            graphs = {}
            saved_outputs = {}
            for name, run in runners.items():
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for target in targets[name]:
                    target.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for target in targets[name]:
                    if not torch.isfinite(target).all().item():
                        raise RuntimeError(f"{name} graph did not overwrite poisoned output")
                if name.endswith(("_4x", "_qkvg")):
                    saved_outputs[name] = [t.clone() for t in targets[name]]
                graphs[name] = graph
            if grouped:
                view_dtype = torch.int16 if (args.transforms and out_dtype == torch.bfloat16) else torch.int32
                for codebook in ("mcg", "mul1"):
                    for i in range(4):
                        if not torch.equal(saved_outputs[f"{codebook}_qkvg"][i].view(view_dtype),
                                           saved_outputs[f"{codebook}_4x"][i].view(view_dtype)):
                            raise RuntimeError(f"{codebook} projection {i} bitwise mismatch at rows={rows}")
            torch.cuda.synchronize()
            samples = {name: [] for name in graphs}
            for repeat in range(args.samples):
                order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
                for name in order:
                    if rows <= 64:
                        flush.zero_()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end)*1000)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            record = dict(k=k, n=n, side=side, rows=rows, us=medians, samples_us=samples,
                          speedup={name: medians["bf16"]/medians[name] for name in ("mcg", "mul1")})
            if grouped:
                record["saved_us"] = {name: medians[f"{name}_4x"] - medians[f"{name}_qkvg"] for name in ("mcg", "mul1")}
                record["saved_vs_4x_single_us"] = {name: 4 * medians[name] - medians[f"{name}_qkvg"] for name in ("mcg", "mul1")}
                record["bitwise_equal"] = True
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "samples_us"}), flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(dict(metadata=metadata, records=records), indent=2)+"\n")
        del graphs, runners, arms, x_bf16, x, inputs, payload, scales, partial, output, bf16_output, weight
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
