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


def exported(program, directory, stem, *, load_module=True):
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
    if not load_module:
        return None
    lib = C.CDLL(str(library))
    lib.load.argtypes, lib.load.restype = [], None
    lib.run.argtypes = [C.c_void_p] * len(pointers) + [C.c_int32] * len(scalars) + [C.c_void_p]
    lib.run.restype = C.c_int32
    lib.load()
    return lib


# Unique stored geometries from Core-K6 headers and TP2/owned-MLA exports.
# MLA kv_a is logically 576 outputs, but both routes must reconstruct all 640.
PREFILL_TP2_SHAPES = (
    (7168, 6144, 1, 4), (6144, 7168, 2, 1),
    (7168, 1536, 1, 1), (1536, 9216, 1, 1), (7168, 640, 1, 1),
    (7168, 3072, 1, 1), (3072, 7168, 2, 1),
    (7168, 3584, 1, 1), (3584, 7168, 1, 1),
    (1536, 18432, 1, 1), (7168, 12288, 1, 1), (12288, 7168, 1, 1),
)


# Per GPU, one target and one draft head; rotations remain separate and unchanged.
GLMF_VERIFY_MIX = {
    (4096, 4096, 1): 34 * 3, (4096, 4096, 2): 34,
    (4096, 1024, 1): 42 * 2, (1024, 4096, 2): 42,
    (4096, 6144, 1): 3 * 2, (6144, 4096, 2): 3,
    (4096, 1536, 1): 11, (4096, 512, 1): 11,
    (1536, 8192, 1): 11, (8192, 4096, 2): 11,
    (4096, 154880, 1): 2,
}


# Per-GPU target projection counts for the qualified TP2 placements. Kimi has
# 69 KDA blocks, 24 MLA blocks split by owner, and 92 shared blocks; the
# routed latent projections alternate owners (46 blocks per GPU).
KIMI_VERIFY_MIX = {
    (7168, 6144, 1): 69 * 4, (6144, 7168, 2): 69,
    (7168, 3072, 1): 92 * 2, (3072, 7168, 2): 92,
    (7168, 3584, 1): 46, (3584, 7168, 1): 46,
    (7168, 1536, 1): 12, (7168, 640, 1): 12,
    (1536, 18432, 1): 12, (7168, 12288, 1): 12,
    (12288, 7168, 1): 12, (7168, 163840, 1): 1,
}
GLM5_VERIFY_MIX = {
    (6144, 2048, 1): 78, (6144, 640, 1): 78,
    (2048, 8192, 1): 78, (8192, 6144, 2): 78,
    (6144, 6144, 1): 3 * 2, (6144, 6144, 2): 3,
    (6144, 1024, 1): 75 * 2, (1024, 6144, 2): 75,
    (6144, 154880, 1): 1,
}
FAMILY_VERIFY_MIXES = {"glmf": GLMF_VERIFY_MIX, "kimik3": KIMI_VERIFY_MIX,
                       "glm5": GLM5_VERIFY_MIX}
# Include both owned and split variants, not only the weighted placement above.
FAMILY_SHAPES = {
    "kimik3": tuple(KIMI_VERIFY_MIX) + ((1536, 9216, 1),),
    "glm5": tuple(GLM5_VERIFY_MIX) + (
        (2048, 16384, 1), (16384, 6144, 1), (6144, 12288, 1),
        (12288, 6144, 1), (2048, 6144, 1)),
    "glmf": tuple(GLMF_VERIFY_MIX),
}


def streaming_read_bandwidth(l2_bytes, samples):
    """Read beyond L2 and reduce to a tiny sink; report measured DRAM GB/s."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def read_kernel(source, sink, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(source + offsets, offsets < N, 0)
        tl.store(sink + tl.program_id(0), tl.sum(values, 0))

    count = max(256 * 1024 * 1024, 4 * l2_bytes) // 4
    source = torch.ones(count, device="cuda", dtype=torch.float32)
    sink = torch.empty(triton.cdiv(count, 4096), device="cuda", dtype=torch.float32)
    def run():
        read_kernel[(sink.numel(),)](source, sink, count, 4096)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(sink.sum(), torch.tensor(float(count), device="cuda")):
        raise RuntimeError("streaming read checksum mismatch")
    times = []
    for _ in range(samples):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1000)
    us = statistics.median(times)
    return dict(bytes=count * 4, us=us, gbs=count * 4 / (us * 1000), samples_us=times)


def projection_aggregate(records, mix):
    aggregate = []
    for rows in sorted({r["rows"] for r in records}):
        cells = {(r["k"], r["n"], r["side"]): r for r in records
                 if r["rows"] == rows and "baseline_us" in r}
        if not mix.keys() <= cells.keys():
            continue
        base = sum(cells[s]["baseline_us"] * count for s, count in mix.items())
        cand = sum(cells[s]["candidate_us"] * count for s, count in mix.items())
        aggregate.append(dict(rows=rows, baseline_us=base, candidate_us=cand, ratio=base/cand))
    return aggregate


def benchmark_layer_replay(args):
    """Back-to-back projections with distinct weights, no timed cache flush."""
    import torch
    from cuda.bindings import driver as cuda
    props = torch.cuda.get_device_properties(0)
    status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0)
    if int(status):
        raise RuntimeError(status)
    flush = torch.empty(2*l2, device="cuda", dtype=torch.uint8)
    max_rows = max(129, max(map(int, args.rows.split(","))))
    mixes = {
        "KDA": [(4096,4096,1)]*3 + [(4096,4096,2)],
        "MLA": [(4096,1536,1), (4096,512,1), (1536,8192,1), (8192,4096,2)],
    }
    results = []
    for layer, shapes in mixes.items():
        shapes = shapes + [(4096,1024,1)]*2 + [(1024,4096,2)]
        owned = []
        runners = []
        def load(directory, stem, pointers, scalars):
            lib = C.CDLL(str((directory / f"{stem}.so").resolve()))
            lib.load.argtypes, lib.load.restype = [], None
            lib.run.argtypes = [C.c_void_p]*pointers + [C.c_int32]*scalars + [C.c_void_p]
            lib.run.restype = C.c_int32
            lib.load()
            return lib
        for k,n,side in shapes:
            directory = args.output_dir / f"k{k}_n{n}_s{side}_b{args.bits}_{args.codebook}_{args.input_dtype}_mr{max_rows}"
            libs = {name: load(directory,name,5,3) for name in ("baseline","candidate")}
            ingress,egress = load(directory,"ingress",4,2),load(directory,"egress",3,2)
            source = torch.randn(8,k,device="cuda",dtype=torch.bfloat16)
            x = torch.empty(8,k,device="cuda",dtype=torch.bfloat16 if args.input_dtype=="bf16" else torch.float16)
            inputs = torch.tensor([x.data_ptr()],device="cuda",dtype=torch.int64)
            payload = torch.randint(256,(k*n*args.bits//8,),device="cuda",dtype=torch.uint8)
            scales = torch.ones(k//128,device="cuda",dtype=torch.float16)
            suh,svh = torch.rand(k,device="cuda",dtype=torch.float16),torch.rand(n,device="cuda",dtype=torch.float16)
            parts = {a:torch.empty(8*((k+1023)//1024)*n,device="cuda") for a in libs}
            outs = {a:torch.empty(8*n,device="cuda") for a in libs}
            finals = {a:torch.empty(8*n,device="cuda",dtype=torch.bfloat16) for a in libs}
            owned.append((source,x,inputs,payload,scales,suh,svh,parts,outs,finals,libs,ingress,egress))
            def call(arm,rows, state=owned[-1]):
                source,x,inputs,payload,scales,suh,svh,parts,outs,finals,libs,ingress,egress=state
                stream=torch.cuda.current_stream().cuda_stream
                codes=[ingress.run(source.data_ptr(),suh.data_ptr(),x.data_ptr(),scales.data_ptr(),rows,(args.cta_limit or props.multi_processor_count*4),stream),
                       libs[arm].run(inputs.data_ptr(),payload.data_ptr(),scales.data_ptr(),parts[arm].data_ptr(),outs[arm].data_ptr(),rows,0,(args.cta_limit or props.multi_processor_count*4),stream)]
                codes.append(egress.run(outs[arm].data_ptr(),svh.data_ptr(),finals[arm].data_ptr(),rows,(args.cta_limit or props.multi_processor_count*4),stream))
                if any(codes):
                    raise RuntimeError(codes)
            runners.append(call)
        for rows in (4,8):
            graphs={}
            for arm in ("baseline","candidate"):
                for call in runners:
                    call(arm,rows)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for call in runners:
                        call(arm,rows)
                graphs[arm]=graph
            torch.cuda.synchronize()
            for state in owned:
                finals=state[9]
                if not torch.equal(finals["baseline"].view(torch.int16),finals["candidate"].view(torch.int16)):
                    # Only the live rows are initialized for M4.
                    n=finals["baseline"].numel()//8
                    if not torch.equal(finals["baseline"][:rows*n].view(torch.int16),finals["candidate"][:rows*n].view(torch.int16)):
                        raise RuntimeError("layer replay byte mismatch")
            for mode in ("warm", "cold"):
                samples={arm:[] for arm in graphs}
                for repeat in range(args.samples):
                    for arm in (list(graphs) if repeat%2==0 else list(reversed(graphs))):
                        if mode=="cold":
                            flush.zero_()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record();graphs[arm].replay();end.record();end.synchronize()
                        samples[arm].append(begin.elapsed_time(end)*1000)
                base,cand=(statistics.median(samples[a]) for a in ("baseline","candidate"))
                results.append(dict(layer=layer,rows=rows,cache=mode,baseline_us=base,candidate_us=cand,ratio=base/cand,exact=True,samples_us=samples))
        del graphs,runners,owned
    return results


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


THRESHOLD_DEFAULTS = dict(narrow=8, decode=16, wide2=32, wide3=48,
                          folded=16, prefill=130)
THRESHOLD_CONSTANTS = dict(narrow="_K6_NARROW_ROWS", decode="_K6_DECODE_ROWS",
                           wide2="_K6_WIDE2_ROWS", wide3="_K6_WIDE3_ROWS",
                           folded="_FOLDED_DECODE_ROWS", prefill="_PREFILL_MIN_ROWS")


def threshold_config(text):
    """Only lower a tile's coverage boundary; never overrun its row capacity."""
    values = dict(THRESHOLD_DEFAULTS)
    if text:
        for assignment in text.split(","):
            name, separator, value = assignment.partition("=")
            if not separator or name not in values:
                raise ValueError(f"unknown threshold assignment {assignment!r}")
            values[name] = int(value)
    for name in ("narrow", "decode", "wide2", "wide3", "folded"):
        if not 0 <= values[name] <= THRESHOLD_DEFAULTS[name]:
            raise ValueError(f"{name} must be 0..{THRESHOLD_DEFAULTS[name]}")
    if not values["narrow"] <= values["decode"] <= values["wide2"] <= values["wide3"]:
        raise ValueError("decode thresholds must be ordered")
    if not 1 <= values["prefill"] <= 130:
        raise ValueError("prefill must be 1..130 (fixed partial capacity)")
    return values


def threshold_variant(source, destination, values):
    """Materialize a benchmark-only module; production sources stay untouched."""
    text = source.read_text()
    for name, constant in THRESHOLD_CONSTANTS.items():
        text, count = re.subn(rf"^{constant} = [0-9]+$",
                              f"{constant} = {values[name]}", text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"expected one threshold definition for {constant}")
    # Generic high-rate and grouped decode use the same 16/32/48-row tiles,
    # but do not currently expose named cutovers. K6 narrow is K6-only.
    for method in ("_legacy", "_generic_grouped"):
        start = text.index(f"    def {method}(")
        end = text.index("\n    @cute.", start)
        section = text[start:end]
        for boundary, name in ((16, "decode"), (32, "wide2"), (48, "wide3")):
            section = section.replace(f"rows <= Int32({boundary})",
                                      f"rows <= Int32({values[name]})")
        text = text[:start] + section + text[end:]
    destination.write_text(text)
    return destination


def thermal_snapshot():
    """Cumulative driver counters cover short phases that periodic polling misses."""
    import csv
    import time
    fields = ("uuid", "utilization.gpu", "clocks.sm", "power.draw", "power.limit",
              "temperature.gpu", "temperature.gpu.tlimit",
              "clocks_event_reasons_counters.sw_power_cap",
              "clocks_event_reasons_counters.sw_thermal_slowdown",
              "clocks_event_reasons_counters.hw_thermal_slowdown")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                      "--format=csv,noheader,nounits"], text=True)
    return dict(unix_ns=time.time_ns(), gpus=[dict(zip(fields, row, strict=True))
                for row in csv.reader(output.splitlines(), skipinitialspace=True)])


def benchmark_native_comparison(args):
    """Compare a previous native implementation through the same exported ABI."""
    import importlib.util
    from b12x.gemm.trellis_linear import _native
    def native_module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    baseline_module = native_module(args.baseline_native, "trellis_native_baseline")
    baseline = baseline_module.NativeTrellisLinear
    values = threshold_config(args.thresholds)
    if args.thresholds:
        path = threshold_variant(Path(_native.__file__), args.output_dir / "threshold_native.py", values)
        candidate = native_module(path, "trellis_native_threshold").NativeTrellisLinear
    else:
        candidate = _native.NativeTrellisLinear
    import hashlib
    import torch
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot

    if args.compile_only:
        import multiprocessing
        from b12x._lib.compile_pool import _initialize_worker
        _initialize_worker(0, (12, 0), "native-byte-comparison", "native-byte-comparison",
                           args.export_sms, 101376, 102400,
                           multiprocessing.Array("i", [0, 0]))

    def snapshot():
        if args.compile_only:
            return "CPU-only export"
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader"], text=True).strip()

    def boost_warmup():
        warm_a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        warm_b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(128):
            torch.mm(warm_a, warm_b)
        torch.cuda.synchronize()
        del warm_a, warm_b

    def load(directory, stem, pointers=5, scalars=3):
        lib = C.CDLL(str((directory / f"{stem}.so").resolve()))
        lib.load.argtypes, lib.load.restype = [], None
        lib.run.argtypes = [C.c_void_p] * pointers + [C.c_int32] * scalars + [C.c_void_p]
        lib.run.restype = C.c_int32
        lib.load()
        return lib

    default_shapes = "4096:4096:1,4096:4096:2,4096:1024:1,1024:4096:2,16384:4096:1,4096:154880:1,4096:6144:1,6144:4096:2"
    family = "glmf" if args.glmf_verify else args.family_verify
    shapes = list(FAMILY_SHAPES[family]) if family else [tuple(map(int, s.split(":"))) for s in (args.shapes or default_shapes).split(",")]
    rows_list = list(map(int, args.rows.split(",")))
    if not rows_list or min(rows_list) < 1 or max(rows_list) > 4096 or args.samples < 1:
        raise ValueError("native comparison requires rows 1..4096 and positive samples")
    if args.thresholds and not args.check_only and args.samples < 50:
        raise ValueError("threshold measurements require at least 50 samples")
    max_rows = max(129, max(rows_list))
    arms = (("baseline", baseline), ("candidate", candidate))
    if args.compile_only:
        props = None
        l2 = 0
    else:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not args.compile_only:
        status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0)
        if int(status):
            raise RuntimeError(status)
    snapshot_idle = snapshot()
    if not args.compile_only:
        boost_warmup()
        flush = torch.empty(2 * l2, device="cuda", dtype=torch.uint8)
        flush.zero_()
        torch.cuda.synchronize()
    report = dict(metadata=dict(device=props.name if props else "CPU-only", sms=props.multi_processor_count if props else None,
        l2_bytes=l2, argv=sys.argv, snapshot_idle=snapshot_idle, snapshot_before=snapshot(), bits=args.bits,
        input_dtype=args.input_dtype, codebook=args.codebook,
        thresholds=values, source_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(args.baseline_native.read_bytes()).hexdigest(),
        ratio="baseline/candidate", cache="32 in-graph cold units (decode) / 4 in-graph cold units (prefill), 2x runtime L2 eviction outside each timed interval"), records=[])
    if family and not args.compile_only:
        report["metadata"]["streaming_read"] = streaming_read_bandwidth(l2, args.samples)
        report["metadata"]["mix"] = {":".join(map(str, s)): count for s, count in FAMILY_VERIFY_MIXES[family].items()}
        operations = "ingress + multiply/reduce + native egress" if args.transforms else "native multiply/reduce only"
        heads = "one target + one draft head" if family == "glmf" else "target projections and target head"
        report["metadata"]["aggregate_scope"] = operations + "; " + heads
    torch.manual_seed(273)
    for k, n, side in shapes:
        directory = args.output_dir / f"k{k}_n{n}_s{side}_b{args.bits}_{args.codebook}_{args.input_dtype}_mr{max_rows}"
        directory.mkdir(parents=True, exist_ok=True)
        contract = dict(bits=args.bits, k=k, n=n, side=side, max_rows=max_rows,
                        input_dtype=args.input_dtype, codebook=args.codebook, thresholds=values,
                        source_sha256=report["metadata"]["source_sha256"],
                        baseline_sha256=report["metadata"]["baseline_sha256"])
        manifest = directory / "benchmark_contract.json"
        if args.compile_only:
            manifest.write_text(json.dumps(contract, sort_keys=True) + "\n")
        elif json.loads(manifest.read_text()) != contract:
            raise ValueError(f"stale exported benchmark contract: {directory}")
        libs = {}
        for stem, core in arms:
            if args.compile_only:
                print(f"compile {k}:{n}:{side} {stem} max_rows={max_rows}", flush=True)
                _native.NativeTrellisLinear = core
                try:
                    with exportable_compilation():
                        program = compile_trellis_linear_aot(input_features=k, output_features=n,
                            bits=args.bits, codebook=args.codebook, sharded_side=side, rotation_count=1,
                            max_rows=max_rows, input_dtype=args.input_dtype)
                finally:
                    _native.NativeTrellisLinear = candidate
                exported(program, directory, stem, load_module=False)
                (directory / f"{stem}.json").write_text(json.dumps(program.geometry))
            else:
                libs[stem] = load(directory, stem, 5)
        ingress = egress = None
        if args.transforms:
            from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
            if args.compile_only:
                with exportable_compilation():
                    ip = compile_trellis_input_aot(features=k, max_rows=max_rows, output_dtype=args.input_dtype)
                    op = compile_trellis_output_aot(features=n, max_rows=max_rows,
                                                    output_dtype="f32" if n > 65536 else "bf16")
                exported(ip, directory, "ingress", load_module=False)
                exported(op, directory, "egress", load_module=False)
            else:
                ingress, egress = load(directory, "ingress", 4, 2), load(directory, "egress", 3, 2)
        if args.compile_only:
            continue
        capacity = max(rows_list)
        decode_cap = min(capacity, 129)
        splits = (k + 1023) // 1024
        core = baseline(dict(input_features=k, output_features=n, bits=args.bits,
            codebook=args.codebook, sharded_side=side, rotation_count=1,
            max_rows=max_rows, splits=splits, input_dtype=args.input_dtype))
        candidate_core = candidate(dict(input_features=k, output_features=n, bits=args.bits,
            codebook=args.codebook, sharded_side=side, rotation_count=1,
            max_rows=max_rows, splits=splits, input_dtype=args.input_dtype))
        def publishes_partials(rows):
            if rows >= min(core.exact_row_cap + 1, candidate_core.exact_row_cap + 1):
                return False
            # Fused K6 decode and folded K4 write output without publishing partials.
            if args.bits == 6 and rows <= 16:
                return False
            return not (args.bits == 4 and args.codebook == "e4m3"
                        and splits <= 2 and rows <= 16)
        x = torch.randn(capacity, k, device="cuda", dtype=getattr(torch, {"bf16":"bfloat16", "f16":"float16"}[args.input_dtype]))
        inputs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
        payload = torch.randint(256, (k*n*args.bits//8,), device="cuda", dtype=torch.uint8)
        scales = torch.rand(k//128, device="cuda", dtype=torch.float16) + .125
        partials = {name: torch.empty(decode_cap*splits*n, device="cuda") for name in libs}
        outputs = {name: torch.empty(capacity*n, device="cuda") for name in libs}
        if args.transforms:
            source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
            suh = torch.rand(k, device="cuda", dtype=torch.float16) + .125
            svh = torch.rand(n, device="cuda", dtype=torch.float16) + .125
            final_dtype = torch.float32 if n > 65536 else torch.bfloat16
            finals = {name: torch.empty(capacity*n, device="cuda", dtype=final_dtype) for name in libs}
        # Untimed boost warm-up before row 1 of each shape
        boost_warmup()
        for rows in rows_list:
            graphs = {}
            hashes = {}
            events = {}
            def call(name, count=rows):
                stream = torch.cuda.current_stream().cuda_stream
                if args.transforms:
                    result = ingress.run(source.data_ptr(), suh.data_ptr(), x.data_ptr(), scales.data_ptr(),
                                         count, (args.cta_limit or props.multi_processor_count*4), stream)
                    if result:
                        raise RuntimeError(f"ingress returned {result}")
                result = libs[name].run(inputs.data_ptr(), payload.data_ptr(), scales.data_ptr(),
                    partials[name].data_ptr(), outputs[name].data_ptr(), count, 0,
                    (args.cta_limit or props.multi_processor_count*4), stream)
                if result:
                    raise RuntimeError(f"{name} returned {result}")
                if args.transforms:
                    result = egress.run(outputs[name].data_ptr(), svh.data_ptr(), finals[name].data_ptr(),
                                        count, (args.cta_limit or props.multi_processor_count*4), stream)
                    if result:
                        raise RuntimeError(f"egress returned {result}")
            for name in libs:
                call(name)
                graph = torch.cuda.CUDAGraph()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    start.record()
                    call(name)
                    end.record()
                events[name] = (start, end)
                outputs[name][:rows*n].fill_(float("nan"))
                if args.transforms:
                    finals[name][:rows*n].fill_(float("nan"))
                if publishes_partials(rows):
                    partials[name][:rows*splits*n].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(outputs[name][:rows*n]).all().item():
                    raise RuntimeError(f"{name}: poisoned output not overwritten")
                if args.transforms and not torch.isfinite(finals[name][:rows*n]).all().item():
                    raise RuntimeError(f"{name}: poisoned transformed output not overwritten")
                if publishes_partials(rows) and not torch.isfinite(partials[name][:rows*splits*n]).all().item():
                    raise RuntimeError(f"{name}: poisoned partial not overwritten")
                graphs[name] = graph
                hashes[name] = hashlib.sha256(outputs[name][:rows*n].cpu().numpy().tobytes()).hexdigest()
            exact = {}
            partials_exact = {}
            for name in libs:
                out_eq = torch.equal(outputs[name][:rows*n].view(torch.int32), outputs["baseline"][:rows*n].view(torch.int32))
                part_eq = (torch.equal(partials[name][:rows*splits*n].view(torch.int32),
                                       partials["baseline"][:rows*splits*n].view(torch.int32))
                           if publishes_partials(rows) else None)
                if args.transforms:
                    out_eq = out_eq and torch.equal(finals[name][:rows*n].view(torch.int32),
                                                   finals["baseline"][:rows*n].view(torch.int32))
                exact[name] = out_eq and part_eq is not False
                partials_exact[name] = part_eq
                if not exact[name]:
                    raise RuntimeError(f"non-bit-exact {k}:{n}:{side} rows={rows} {name} (out_eq={out_eq}, part_eq={part_eq})")
                saved = outputs[name][:n].clone()
                outputs[name][:n].fill_(float("nan"))
                call(name, 1)
                if not torch.equal(saved.view(torch.int32), outputs[name][:n].view(torch.int32)):
                    raise RuntimeError(f"row invariance failure: {name} rows={rows}")
                graphs[name].replay()
            if args.check_only or (args.timing_rows and rows not in set(map(int, args.timing_rows.split(",")))):
                record = dict(k=k, n=n, side=side, splits=splits, rows=rows,
                    bits=args.bits, codebook=args.codebook, input_dtype=args.input_dtype,
                    exact=True, partials_exact=partials_exact["candidate"],
                    partials_checked=publishes_partials(rows), exact_row_cap=core.exact_row_cap,
                    sha256=hashes["candidate"])
                report["records"].append(record)
                continue
            intervals = 32 if rows <= (64 if args.codebook == "e4m3" else 129) else (1 if n > 65536 else 4)
            timing_graphs = {}
            for name in libs:
                pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True)) for _ in range(intervals)]
                timed = torch.cuda.CUDAGraph()
                with torch.cuda.graph(timed):
                    for start, end in pairs:
                        flush.zero_()
                        start.record()
                        call(name)
                        end.record()
                timing_graphs[name] = timed
                events[name] = pairs
                timed.replay()
            torch.cuda.synchronize()
            thermal_before = thermal_snapshot()
            samples = {name: [] for name in libs}
            for repeat in range(args.samples):
                order = list(libs) if repeat % 2 == 0 else list(reversed(libs))
                for name in order:
                    timing_graphs[name].replay()
                    events[name][-1][1].synchronize()
                    samples[name].append(statistics.mean(start.elapsed_time(end)*1000 for start, end in events[name]))
            times = {name: statistics.median(v) for name, v in samples.items()}
            name = "candidate"
            record = dict(k=k, n=n, side=side, splits=splits, rows=rows, arm=name,
                baseline_us=times["baseline"], candidate_us=times[name], ratio=times["baseline"]/times[name],
                exact=exact[name], sha256=hashes[name], partials_exact=partials_exact[name],
                partials_checked=publishes_partials(rows), exact_row_cap=core.exact_row_cap,
                weight_gbs=(k*n*args.bits/8)/(times[name]*1000),
                baseline_samples_us=samples["baseline"], samples_us=samples[name], snapshot=snapshot(),
                thermal_before=thermal_before, thermal_after=thermal_snapshot())
            if family:
                record["bandwidth_fraction"] = record["weight_gbs"] / report["metadata"]["streaming_read"]["gbs"]
            report["records"].append(record)
            if family:
                report["aggregate"] = projection_aggregate(report["records"], FAMILY_VERIFY_MIXES[family])
            print(json.dumps({key: value for key, value in record.items() if key not in ("samples_us", "baseline_samples_us")}), flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        del graphs, partials, outputs, x, inputs, payload, scales
        if "timing_graphs" in locals():
            del timing_graphs
        torch.cuda.empty_cache()
    if args.layer_replay and not args.compile_only:
        report["layer_replay"] = benchmark_layer_replay(args)
    report["metadata"]["snapshot_after"] = snapshot()
    if not args.compile_only:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")



def reference_encoder(source, directory):
    """Build only the unchanged GPTQModel quantize/pack operators, not its GEMMs."""
    from torch.utils.cpp_extension import load
    source = source.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binding = directory / "encoder_bindings.cpp"
    binding.write_text('''#include <torch/library.h>
#include "quant/quantize.cuh"
void pack_trellis(at::Tensor packed, at::Tensor unpacked, int K);
void quantize(at::Tensor input, at::Tensor output, at::Tensor indices,
              at::Tensor costs, at::Tensor edges, int64_t K, bool mcg, bool mul1) {
  TORCH_CHECK(K >= 1 && K <= 8, "K must be 1..8");
  quantize_tiles(input, output, indices, costs, edges, static_cast<int>(K), mcg, mul1);
}
void pack(at::Tensor packed, at::Tensor unpacked, int64_t K) {
  TORCH_CHECK(K >= 1 && K <= 8, "K must be 1..8");
  pack_trellis(packed, unpacked, static_cast<int>(K));
}
TORCH_LIBRARY(trellis_bench_encoder, m) {
  m.def("quantize_tiles(Tensor input, Tensor(a!) output, Tensor(b!) indices, Tensor(c!) costs, Tensor(d!) edges, int K, bool mcg, bool mul1) -> ()", &quantize);
  m.def("pack_trellis(Tensor(a!) packed, Tensor unpacked, int K) -> ()", &pack);
}
''')
    sources = [binding, *(source / name for name in (
        "quant/quantize.cu", "quant/pack.cu", "quant/exl3_devctx.cu"))]
    load(name="trellis_bench_encoder", sources=[str(path) for path in sources],
         extra_include_paths=[str(source)], extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_120,code=sm_120"],
         build_directory=str(directory), is_python_module=False, verbose=True)
    import hashlib
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*")) if path.is_file()
            and path.suffix in (".cu", ".cuh", ".h", ".cpp")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", help="K:N:side[:groups] list; --prefill-route defaults to audited TP2 shapes")
    parser.add_argument("--rows", default="1,64,256")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--transforms", action="store_true", help="include actual exported H128 input/output transforms")
    parser.add_argument("--cta-limit", type=int, help="override runtime grid budget for exactness checks")
    parser.add_argument("--layer-replay", action="store_true", help="time real seven-projection KDA/MLA graph sequences")
    parser.add_argument("--family-verify", choices=tuple(FAMILY_VERIFY_MIXES), help="family TP2 projection shapes and weighted per-round aggregates")
    parser.add_argument("--glmf-verify", action="store_true", help="GLMF verify mix, measured streaming bandwidth and round aggregate")
    parser.add_argument("--prefill-route", action="store_true", help="MCG T2 versus dequant + F32-output BF16 GEMM, including grouped KDA")
    parser.add_argument("--baseline-native", type=Path, help="compare output and partial bytes against a previous _native.py")
    parser.add_argument("--compile-only", action="store_true", help="CPU-only export of both comparison arms")
    parser.add_argument("--check-only", action="store_true", help="compare every row without collecting timings")
    parser.add_argument("--timing-rows", help="time only these rows while checking all requested rows")
    parser.add_argument("--input-dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--codebook", choices=("mcg", "mul1", "f16", "e4m3"), default="mcg")
    parser.add_argument("--bits", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--thresholds", help="benchmark-only narrow/decode/wide2/wide3/folded/prefill overrides")
    parser.add_argument("--export-sms", type=int, choices=(170, 188), default=188,
                        help="offline SM120 compiler geometry; runtime CTA budget stays device-derived")
    parser.add_argument("--encoder-source", type=Path, help="official GPTQModel gptqmodel_ext/exllamav3 source")
    parser.add_argument("--encoder-build-only", action="store_true", help="CPU compile official quantize/pack operators")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.encoder_build_only:
        if args.encoder_source is None:
            parser.error("--encoder-build-only requires --encoder-source")
        hashes = reference_encoder(args.encoder_source, args.output_dir / "encoder")
        (args.output_dir / "encoder_sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
        return
    if args.baseline_native:
        benchmark_native_comparison(args)
        return
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
