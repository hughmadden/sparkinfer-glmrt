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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", default="7168:6144:1,7168:163840:1", help="K:N:side list")
    parser.add_argument("--rows", default="1,64,256")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--transforms", action="store_true", help="include actual exported H128 input/output transforms")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        k, n, side = map(int, shape.split(":"))
        directory = args.output_dir / f"k{k}_n{n}_s{side}"
        directory.mkdir(exist_ok=True)
        capacity = max(rows_list)
        x_bf16 = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
        x = x_bf16.half()
        inputs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
        payload = torch.randint(256, (k*n*6//8,), device="cuda", dtype=torch.uint8)
        scales = torch.ones(k//128, device="cuda", dtype=torch.float16)
        partial = torch.empty(capacity*((k+1023)//1024)*n, device="cuda", dtype=torch.float32)
        output = torch.empty(capacity*n, device="cuda", dtype=torch.float32)
        bf16_output = torch.empty(capacity*n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        arms = {}
        suh=torch.ones(k,device="cuda",dtype=torch.float16)
        svh=torch.ones(n,device="cuda",dtype=torch.float16)
        transformed_output=torch.empty(capacity*n,device="cuda",dtype=torch.float32 if n==163840 else torch.bfloat16)
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
        for rows in rows_list:
            graphs = {}
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
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                target=bf16_output[:rows*n] if name=="bf16" else transformed_output[:rows*n] if args.transforms else output[:rows*n]
                target.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(target).all().item():
                    raise RuntimeError(f"{name} graph did not overwrite poisoned output")
                graphs[name] = graph
            torch.cuda.synchronize()
            samples = {name: [] for name in arms}
            for repeat in range(args.samples):
                order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
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
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "samples_us"}), flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(dict(metadata=metadata, records=records), indent=2)+"\n")
        del graphs, arms, x_bf16, x, inputs, payload, scales, partial, output, bf16_output, weight
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
