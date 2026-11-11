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


def threshold_config(text, groups=1):
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
    if groups == 4 and (values["decode"] < 16 or values["wide2"] < 32):
        raise ValueError("grouped wide2/wide3 lower cutovers overlap the next projection's partial plane")
    return values


def threshold_variant(source, destination, values, groups=1):
    """Materialize a benchmark-only module; production sources stay untouched."""
    threshold_config(",".join(f"{name}={value}" for name, value in values.items()), groups)
    text = source.read_text()
    if values != threshold_config(None) and groups == 1:
        text = text.replace("self.decode_rows = decode_rows(g)", f"self.decode_rows = {values['decode']}")
    if groups == 4:
        text = text.replace("_GROUPED_DECODE_ROWS = 16", f"_GROUPED_DECODE_ROWS = {values['decode']}")
        text = text.replace("_GROUPED_WIDE2_ROWS = 32", f"_GROUPED_WIDE2_ROWS = {values['wide2']}")
    for name, constant in THRESHOLD_CONSTANTS.items():
        text, count = re.subn(rf"^{constant} = [0-9]+$",
                              f"{constant} = {values[name]}", text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"expected one threshold definition for {constant}")
    # Grouped tiles require full earlier row blocks; standalone cutovers do not apply.
    for method in (("_legacy", "_generic_grouped") if groups == 4 else ("_legacy",)):
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
    values = threshold_config(args.thresholds, args.groups)
    if args.thresholds:
        path = threshold_variant(Path(_native.__file__), args.output_dir / "threshold_native.py", values, args.groups)
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

    if args.groups != 1 and args.transforms:
        raise ValueError("grouped threshold comparison measures the native multiply/reduce only")
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
        input_dtype=args.input_dtype, codebook=args.codebook, groups=args.groups,
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
                        input_dtype=args.input_dtype, codebook=args.codebook, thresholds=values, groups=args.groups,
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
                            bits=args.bits, codebook=args.codebook, sharded_side=side, rotation_count=args.groups,
                            max_rows=max_rows, input_dtype=args.input_dtype, groups=args.groups)
                finally:
                    _native.NativeTrellisLinear = candidate
                exported(program, directory, stem, load_module=False)
                (directory / f"{stem}.json").write_text(json.dumps(program.geometry))
            else:
                libs[stem] = load(directory, stem, 5 if args.groups == 1 else 14)
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
            if args.groups == 1 and args.bits == 6 and rows <= 16:
                return False
            return not (args.bits == 4 and args.codebook == "e4m3"
                        and splits <= 2 and rows <= 16)
        x = torch.randn(capacity, k, device="cuda", dtype=getattr(torch, {"bf16":"bfloat16", "f16":"float16"}[args.input_dtype]))
        inputs = torch.tensor([x.data_ptr()] * args.groups, device="cuda", dtype=torch.int64)
        payload = torch.randint(256, (args.groups, k*n*args.bits//8), device="cuda", dtype=torch.uint8)
        scales = torch.rand(args.groups, k//128, device="cuda", dtype=torch.float16) + .125
        partials = {name: torch.empty(decode_cap*splits*n*args.groups, device="cuda") for name in libs}
        outputs = {name: torch.empty(capacity*n*args.groups, device="cuda") for name in libs}
        if args.transforms:
            source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
            suh = torch.rand(k, device="cuda", dtype=torch.float16) + .125
            svh = torch.rand(n, device="cuda", dtype=torch.float16) + .125
            final_dtype = torch.float32 if n > 65536 else torch.bfloat16
            finals = {name: torch.empty(capacity*n, device="cuda", dtype=final_dtype) for name in libs}
        def live_output(name, count):
            return outputs[name].view(args.groups, capacity, n)[:, :count]

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
                pointers = ([inputs.data_ptr(), payload.data_ptr(), scales.data_ptr(),
                             partials[name].data_ptr(), outputs[name].data_ptr()] if args.groups == 1 else
                            [inputs.data_ptr(), *(p.data_ptr() for p in payload),
                             *(s.data_ptr() for s in scales), partials[name].data_ptr(),
                             *(outputs[name][i*capacity*n:].data_ptr() for i in range(args.groups))])
                result = libs[name].run(*pointers, count, 0,
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
                live_output(name, rows).fill_(float("nan"))
                if args.transforms:
                    finals[name][:rows*n].fill_(float("nan"))
                if publishes_partials(rows):
                    partials[name][:rows*splits*n*args.groups].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(live_output(name, rows)).all().item():
                    raise RuntimeError(f"{name}: poisoned output not overwritten")
                if args.transforms and not torch.isfinite(finals[name][:rows*n]).all().item():
                    raise RuntimeError(f"{name}: poisoned transformed output not overwritten")
                if publishes_partials(rows) and not torch.isfinite(partials[name][:rows*splits*n*args.groups]).all().item():
                    raise RuntimeError(f"{name}: poisoned partial not overwritten")
                graphs[name] = graph
                hashes[name] = hashlib.sha256(live_output(name, rows).cpu().numpy().tobytes()).hexdigest()
            exact = {}
            partials_exact = {}
            for name in libs:
                out_eq = torch.equal(live_output(name, rows).view(torch.int32), live_output("baseline", rows).view(torch.int32))
                part_eq = (torch.equal(partials[name][:rows*splits*n*args.groups].view(torch.int32),
                                       partials["baseline"][:rows*splits*n*args.groups].view(torch.int32))
                           if publishes_partials(rows) else None)
                if args.transforms:
                    out_eq = out_eq and torch.equal(finals[name][:rows*n].view(torch.int32),
                                                   finals["baseline"][:rows*n].view(torch.int32))
                exact[name] = out_eq and part_eq is not False
                partials_exact[name] = part_eq
                if not exact[name]:
                    raise RuntimeError(f"non-bit-exact {k}:{n}:{side} rows={rows} {name} (out_eq={out_eq}, part_eq={part_eq})")
                saved = live_output(name, 1).clone()
                live_output(name, 1).fill_(float("nan"))
                call(name, 1)
                if not torch.equal(saved.view(torch.int32), live_output(name, 1).view(torch.int32)):
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
                weight_gbs=(args.groups*k*n*args.bits/8)/(times[name]*1000),
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



def kimi_full_core_operations(config):
    """TP2 head-split baseline: physical calls per verify step, not per token."""
    c = config.get("text_config", config)
    h, layers = c["hidden_size"], c["num_hidden_layers"]
    linear = c["linear_attn_config"]
    kda, mla = len(linear["kda_layers"]), len(linear["full_attn_layers"])
    if kda + mla != layers or h != 7168 or c["num_attention_heads"] != 96:
        raise ValueError("full-core Kimi benchmark requires the catalog's TP2 Kimi geometry")
    dense = sum(not (i >= c.get("first_k_dense_replace", 1)
                        and i % c.get("moe_layer_freq", 1) == 0) for i in range(layers))
    sparse = layers - dense
    if sparse % 2:
        raise ValueError("whole-owner Kimi projections need explicit uneven rank ownership")
    d = linear["num_heads"] * linear["head_dim"] // 2
    shared = c["num_shared_experts"] * c["moe_intermediate_size"] // 2
    def op(role, k, n, count, *, groups=1, side=1, native=False, packed_n=None,
           output="bf16", batch=1, head=False):
        return dict(role=role, k=k, n=n, count=[count, count] if not head else [0, 1],
                    groups=groups, side=side, native=native, packed_n=packed_n or n*groups,
                    output=output, batch=batch, head=head)
    return [
        op("kda", h, d, kda, groups=4, packed_n=4*d+176),
        op("gate_up", h, shared, sparse, groups=2),
        op("attn_out", d, h, layers, side=2),
        op("shared_down", shared, h, sparse, side=2),
        op("latent_down", h, c["routed_expert_hidden_size"], sparse//2),
        op("latent_up", c["routed_expert_hidden_size"], h, sparse//2),
        op("mla_rank", h, c["q_lora_rank"], mla, groups=2,
           packed_n=c["q_lora_rank"]+c["kv_lora_rank"]+c["qk_rope_head_dim"]),
        op("mla_qb", c["q_lora_rank"], c["num_attention_heads"]//2 *
           (c["qk_nope_head_dim"]+c["qk_rope_head_dim"]), mla),
        op("mla_gate", h, d, mla),
        op("router", h, c["num_experts"], sparse//2, native=True, output="f32"),
        op("layer0_gate_up", h, c["intermediate_size"], dense, native=True),
        op("layer0_down", c["intermediate_size"]//2, h, dense, side=2, native=True),
        op("kda_fb", 128, d, kda, native=True),
        op("mla_uk", 128, 512, mla, native=True, batch=48),
        op("mla_uv", 512, 128, mla, native=True, batch=48),
        op("head", h, c["vocab_size"], 1, output="f32", head=True),
    ]


def full_core_fixture_shapes(operations):
    """No tensor concatenation: rank q_a and padded kv_a have separate records."""
    shapes = []
    for op in operations:
        if op["native"]:
            continue
        for member in range(op["groups"]):
            n = 640 if op["role"] == "mla_rank" and member == 1 else op["n"]
            shapes.append(dict(role=op["role"], member=member, k=op["k"], n=n, side=op["side"]))
    return shapes


def validate_kimi_full_core_catalog(operations, catalog, inventory):
    """Check config-derived call multiplicities against every dense catalog entry."""
    subset = kimi_profile_accounting(catalog, inventory)
    entries = [e for e in catalog if e.get("lane") == "target"]
    steps = None
    checked = []
    for op in operations:
        for gpu in (0, 1):
            if op["head"] and gpu == 0:
                continue
            matches = [e for e in entries if e.get("device") == gpu and
                       tuple((e.get("geometry") or {}).get(k) for k in ("n", "k", "batch")) ==
                       (op["packed_n"], op["k"], op["batch"])]
            if len(matches) != 1:
                raise ValueError(f"{op['role']}/GPU{gpu}: missing or ambiguous target catalog geometry")
            phase = matches[0]["phases"]["decode"]
            count = phase["count"]
            # The h48 MLA gate and packed shared gate/up have identical N/K.
            per_step = op["count"][gpu]
            if op["role"] in ("gate_up", "mla_gate"):
                per_step = sum(o["count"][gpu] for o in operations
                               if o["role"] in ("gate_up", "mla_gate"))
            if (phase.get("unknown_rows", 0) or sum(phase["rows"].values()) != count
                    or phase["unit_count"] != subset["emitted_tokens"] or count % per_step):
                raise ValueError(f"{op['role']}/GPU{gpu}: incomplete catalog counts")
            derived = count // per_step
            if steps is not None and derived != steps:
                raise ValueError(f"{op['role']}/GPU{gpu}: config multiplicity differs from catalog")
            steps = derived
            checked.append(dict(role=op["role"], gpu=gpu, count=count, per_verify_step=per_step))
    return dict(verify_steps=steps, emitted_tokens=subset["emitted_tokens"],
                verify_steps_per_emitted_token=steps/subset["emitted_tokens"],
                checked=checked, subset=subset,
                limitation="local dense work only; excludes peer communication and attention recurrence")


def encode_chunked_fixture(args, index, shape, costs, edges, oracle):
    """Bound encoder transient storage for the 163840x7168 vocabulary matrix."""
    import hashlib
    import numpy as np
    import torch

    k, n, bits, side = shape["k"], shape["n"], args.bits, shape["side"]
    packed = torch.empty((k//16, n//16, 16*bits), dtype=torch.int16)
    weight = torch.empty((n, k), dtype=torch.bfloat16)
    perm = []
    for lane in range(32):
        r, c = 2*(lane % 4), lane//4
        perm.extend((r*16+c, (r+1)*16+c, (r+8)*16+c, (r+9)*16+c,
                     r*16+c+8, (r+1)*16+c+8, (r+8)*16+c+8, (r+9)*16+c+8))
    inverse = torch.argsort(torch.tensor(perm, device="cuda"))
    oracle_tiles = 0
    # K128 chunks cap tile/quant/index storage below 256 MiB for every head.
    for begin in range(0, k, 128):
        width = min(128, k-begin)
        tiles = torch.randn((width*n//256, 256), device="cuda", dtype=torch.float32)
        quantized = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        torch.ops.trellis_bench_encoder.quantize_tiles(
            tiles, quantized, indices, costs, edges, bits,
            args.codebook == "mcg", args.codebook == "mul1")
        encoded = torch.empty((width//16, n//16, 16*bits), device="cuda", dtype=torch.int16)
        torch.ops.trellis_bench_encoder.pack_trellis(encoded, indices.view(width//16, n//16, 256), bits)
        if begin in (0, k-128):
            decoded = oracle.Reference(args.codebook).decode_tiles(
                encoded.view(-1, 16*bits)[:8].cpu().numpy().tobytes(), bits)
            got = decoded.reshape(8, 256)[:, perm]
            expected = quantized[:8].cpu().numpy()
            if not np.array_equal(got.view(np.uint32), expected.view(np.uint32)):
                raise RuntimeError("chunked official encoder differs from independent decoder")
            oracle_tiles += 8
        packed[begin//16:(begin+width)//16].copy_(encoded.cpu())
        weight[:, begin:begin+width].copy_(tiles[:, inverse].view(width//16, n//16, 16, 16)
            .permute(0, 2, 1, 3).reshape(width, n).T.to(torch.bfloat16).cpu())
        del tiles, quantized, indices, encoded
    if side == 1:
        payload = packed.view(k//16, n//128, 8, 16*bits).permute(1, 0, 2, 3).contiguous()
    else:
        payload = packed.view(k//128, 8, n//128, 8, 16*bits).permute(0, 2, 1, 3, 4).contiguous()
    payload = payload.view(torch.uint8).flatten()
    path = args.output_dir / f"projection{index}_k{bits}.pt"
    torch.save(dict(payload=payload, weight=weight, bits=bits, codebook=args.codebook,
                    source_seed=273, projection=index, **shape), path)
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                result.update(block)
        return result.hexdigest()
    return dict(path=path.name, bits=bits, projection=index, **shape,
                sha256=digest(path), payload_sha256=hashlib.sha256(memoryview(payload.numpy())).hexdigest(),
                oracle_tiles=oracle_tiles, oracle_exact=True, encoding_chunk_k=128)


def encode_reference_fixtures(args):
    """Encode each projection independently with unchanged official CUDA operators."""
    import hashlib
    import importlib.util
    import torch
    import numpy as np

    torch.ops.load_library(str(args.encoder_library.resolve()))
    spec = importlib.util.spec_from_file_location(
        "fixture_trellis_oracle", Path(__file__).resolve().parents[1] / "tests/_reference/native_trellis.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    bits = args.bits
    torch.manual_seed(273)
    costs = torch.empty((256, 2, 65536 >> bits), device="cuda", dtype=torch.float16)
    edges = torch.empty((256, 256, 65536 >> bits), device="cuda", dtype=torch.int16)
    records = []
    if args.full_core:
        operations = kimi_full_core_operations(json.loads(args.checkpoint_config.read_text()))
        shapes = full_core_fixture_shapes(operations)
    else:
        shapes = [dict(k=7168, n=n, side=1) for n in [6144]*4 + [3072]*2]
    for index, shape in enumerate(shapes):
        k, n, side = shape["k"], shape["n"], shape["side"]
        if args.full_core:
            record = encode_chunked_fixture(args, index, shape, costs, edges, oracle)
            records.append(record)
            (args.output_dir / f"fixtures_k{bits}.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            continue
        # These are full tensor-core-order random tiles, not random packed bytes.
        tiles = torch.randn((k*n//256, 256), device="cuda", dtype=torch.float32)
        quantized = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        torch.ops.trellis_bench_encoder.quantize_tiles(
            tiles, quantized, indices, costs, edges, bits,
            args.codebook == "mcg", args.codebook == "mul1")
        packed = torch.empty((k//16, n//16, 16*bits), device="cuda", dtype=torch.int16)
        torch.ops.trellis_bench_encoder.pack_trellis(packed, indices.view(k//16, n//16, 256), bits)
        torch.cuda.synchronize()
        sample = packed.view(-1, 16*bits)[:8].cpu().numpy().tobytes()
        decoded = oracle.Reference(args.codebook).decode_tiles(sample, bits)
        perm = []
        for lane in range(32):
            r, c = 2*(lane % 4), lane//4
            perm.extend((r*16+c, (r+1)*16+c, (r+8)*16+c, (r+9)*16+c,
                         r*16+c+8, (r+1)*16+c+8, (r+8)*16+c+8, (r+9)*16+c+8))
        expected = quantized[:8].cpu().numpy()
        got = decoded.reshape(8, 256)[:, perm]
        if not np.array_equal(got.view(np.uint32), expected.view(np.uint32)):
            raise RuntimeError("official encoder reconstruction differs from independent packed-byte decoder")
        # Side1 native tiles are [N128,K16,N16-within128], never concatenated calibrations.
        payload = packed.view(k//16, n//128, 8, 16*bits).permute(1, 0, 2, 3).contiguous().view(torch.uint8).flatten().cpu()
        inverse = torch.argsort(torch.tensor(perm, device="cuda"))
        weight = tiles[:, inverse].view(k//16, n//16, 16, 16).permute(0, 2, 1, 3).reshape(k, n).T.to(torch.bfloat16).contiguous().cpu()
        path = args.output_dir / f"projection{index}_k{bits}.pt"
        torch.save(dict(payload=payload, weight=weight, k=k, n=n, bits=bits,
                        codebook=args.codebook, source_seed=273, projection=index), path)
        record = dict(path=path.name, k=k, n=n, bits=bits, projection=index,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      payload_sha256=hashlib.sha256(payload.numpy().tobytes()).hexdigest(),
                      oracle_tiles=8, oracle_exact=True)
        records.append(record)
        (args.output_dir / f"fixtures_k{bits}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        del tiles, quantized, indices, packed, payload, weight
    torch.cuda.synchronize()


def kimi_profile_accounting(catalog, inventory):
    """Disambiguate packed shared gate/up from the same-shaped single MLA gate."""
    units = inventory.get("unit_count")
    if inventory.get("phase") != "decode" or inventory.get("unit") != "emitted-token" or type(units) is not int or units <= 0:
        raise ValueError("decode inventory must count emitted tokens")

    def entry(n, k):
        matches = [e for e in catalog if e.get("device") == 0 and e.get("lane") == "target"
                   and (e.get("geometry") or {}).get("n") == n
                   and e["geometry"].get("k") == k]
        if len(matches) != 1:
            raise ValueError(f"expected one GPU0 target profile entry for N{n}/K{k}")
        phase = matches[0]["phases"]["decode"]
        if phase["unit_count"] != units or phase.get("unknown_rows", 0):
            raise ValueError("profile token counts or known-row coverage disagree")
        count = phase["count"]
        if type(count) is not int or count <= 0 or sum(phase["rows"].values()) != count:
            raise ValueError("profile counts must equal the complete row histogram")
        return phase

    kda, mixed, down, qb = (entry(n, k) for n, k in
                           ((24752, 7168), (6144, 7168), (7168, 3072), (9216, 1536)))
    symbols = {}
    for kernel in inventory["kernels"]:
        if kernel.get("device") != 0 or kernel["name"] not in mixed["symbols"]:
            continue
        symbol = symbols.setdefault(kernel["name"], dict(count=0, rows={}))
        if kernel.get("row_unknown", 0):
            raise ValueError("mixed projection symbol has unknown rows")
        symbol["count"] += kernel["count"]
        for row, count in kernel["rows"].items():
            symbol["rows"][row] = symbol["rows"].get(row, 0) + count
    if len(symbols) != 2:
        raise ValueError("mixed projection requires two independently identified symbols")
    roles = {}
    for role, reference in (("packed shared gate/up", down), ("single MLA h48 gate", qb)):
        matches = [name for name, value in symbols.items()
                   if value["count"] == reference["count"] and value["rows"] == reference["rows"]]
        if len(matches) != 1:
            raise ValueError(f"cannot disambiguate {role} from corroborating projection counts")
        roles[role] = matches[0]
    combined = {row: down["rows"].get(row, 0) + qb["rows"].get(row, 0)
                for row in down["rows"].keys() | qb["rows"].keys()}
    if (len(set(roles.values())) != 2 or mixed["count"] != down["count"] + qb["count"]
            or mixed["rows"] != combined):
        raise ValueError("mixed profile does not decompose into shared gate/up plus MLA gate")
    return dict(calls_per_emitted_token=dict(kda=kda["count"] / units, gate_up=down["count"] / units),
                emitted_tokens=units, mixed_count=mixed["count"],
                shared_gate_up_count=down["count"], mla_gate_count=qb["count"], symbols=roles,
                corroboration="shared down N7168/K3072; MLA q_b N9216/K1536; not names of the mixed weights")


def aggregate_kimi_full_core(operations, records, accounting, rows_list):
    """Separate simultaneous GPU work; head prefill rows follow the real trace."""
    lookup = {(r["role"], r["rows"]): r["total_call_us"] for r in records}
    def cost(op, rows, arm):
        times = lookup[(op["role"], rows)]
        return times["native" if op["native"] else arm]
    decode = []
    for rows in rows_list:
        if rows > 64:
            continue
        if any((o["role"], rows) not in lookup for o in operations):
            continue
        sums = {arm: [sum(cost(o, rows, arm)*o["count"][gpu] for o in operations)
                       for gpu in (0, 1)] for arm in ("k6", "k8", "bf16")}
        decode.append(dict(rows=rows, local_dense_us_per_verify_step=sums,
            local_dense_us_per_emitted_token={a:[v*accounting["verify_steps_per_emitted_token"] for v in values]
                                             for a,values in sums.items()}))
    prefill = []
    for tokens, histogram, heads in ((8192, {2752:2,2688:1}, 1), (65536,{2752:23,2240:1},8)):
        if any((o["role"], row) not in lookup for o in operations if not o["head"] for row in histogram):
            continue
        if ("head", 1) not in lookup:
            continue
        sums = {arm:[sum((heads*cost(o,1,arm) if o["head"] else
                           sum(count*cost(o,row,arm) for row,count in histogram.items()))*o["count"][gpu]
                          for o in operations) for gpu in (0,1)] for arm in ("k6","k8","bf16")}
        prefill.append(dict(tokens=tokens, body_row_histogram=histogram, head_rows=1,
                            head_calls=heads, local_dense_us=sums))
    return dict(decode=decode,prefill=prefill,
        scope="per-GPU local launch sums, not summed-across-GPU latency; communication and nondense work excluded")


def benchmark_kimi_full_core(args):
    """Complete local dense work, including each format's actual output boundary."""
    import hashlib
    import torch
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    from b12x.gemm.bf16_gemv._skinny import wide_prefill_min_rows

    rows_list = sorted(set(map(int, args.rows.split(","))))
    if not rows_list or min(rows_list) < 1 or max(rows_list) > 4096 or args.samples < 50:
        raise ValueError("full-core rows must be 1..4096, samples >=50")
    config = json.loads(args.checkpoint_config.read_text())
    operations = kimi_full_core_operations(config)
    contract = dict(capacity=4096, export_sms=args.export_sms,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] /
                        "b12x/gemm/trellis_linear/_native.py").read_bytes()).hexdigest(), operations=operations)
    if args.compile_only:
        import multiprocessing
        from b12x._lib.compile_pool import _initialize_worker
        _initialize_worker(0, (12, 0), "full-core", "full-core", args.export_sms,
                           101376, 102400, multiprocessing.Array("i", [0, 0]))
    else:
        props = torch.cuda.get_device_properties(0)
        status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0)
        if int(status):
            raise RuntimeError(status)
        limit = props.multi_processor_count*4
        flush = torch.empty(2*l2, device="cuda", dtype=torch.uint8)
        native = C.CDLL(str(args.native_library.resolve()))
        native_info = dict(path=str(args.native_library.resolve()),
            sha256=hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
            provenance=json.loads(args.native_library.with_name("COMPILER_PROVENANCE.json").read_text()))
        signatures = {
            "cuteafd_load_narrow_f32_bf16": [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32, C.c_void_p],
            "cuteafd_bf16_f32_add_async": [C.c_void_p]*3+[C.c_size_t, C.c_int32, C.c_void_p],
            "cuteafd_cuda_residual_add_f32_async": [C.c_void_p]*3+[C.c_size_t, C.c_void_p],
            "cuteafd_add_tp2_shared_async": [C.c_void_p]*3+[C.c_size_t, C.c_void_p],
            "cuteafd_vocabulary_head_create_vocab": [C.c_void_p, C.c_uint64]+[C.c_int32]*4+[C.POINTER(C.c_void_p)],
            "cuteafd_vocabulary_head_launch_tensor_invariant": [C.c_void_p]*4+[C.c_int32, C.c_void_p],
            "cuteafd_v41_markov_destroy": [C.c_void_p],
        }
        for name, signature in signatures.items():
            function = getattr(native, name)
            function.argtypes, function.restype = signature, C.c_int32
        manifests = {b: json.loads((args.fixture_dir/f"fixtures_k{b}.json").read_text()) for b in (6, 8)}
        expected = full_core_fixture_shapes(operations)
        for records in manifests.values():
            if len(records) != len(expected) or any(not r["oracle_exact"] or
                    any(r[k] != s[k] for k in s) for r, s in zip(records, expected)):
                raise ValueError("full-core fixture geometry or oracle coverage differs")
        accounting = validate_kimi_full_core_catalog(operations, json.loads(args.profile_catalog.read_text()),
                                                     json.loads(args.profile_inventory.read_text()))
        report = dict(metadata=dict(contract=contract, samples=args.samples, native_library=native_info,
            device=props.name, sms=props.multi_processor_count, l2_bytes=l2,
            accounting=accounting, fixture_manifests=manifests,
            scope="local dense launch sums, not end-to-end latency; no peer communication, recurrence, norms or activations",
            cache="2x runtime L2 eviction outside each complete-operation interval",
            numerics="reference encoded random tiles and unit H128 scales; no quality claim; equality within formats only"),
            records=[])
        torch.manual_seed(811)
        warm = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(128):
            torch.mm(warm, warm)
        torch.cuda.synchronize()
        del warm

    def program(directory, name, builder):
        if args.compile_only:
            print("compile", directory.name, name, flush=True)
            with exportable_compilation():
                result = builder()
            exported(result, directory, name, load_module=False)
            (directory/f"{name}.json").write_text(json.dumps(dict(pointers=len(result.operands), scalars=len(result.scalars))))
            return None
        abi = json.loads((directory/f"{name}.json").read_text())
        lib = C.CDLL(str((directory/f"{name}.so").resolve()))
        lib.load.argtypes, lib.load.restype = [], None
        lib.run.argtypes = [C.c_void_p]*abi["pointers"]+[C.c_int32]*abi["scalars"]+[C.c_void_p]
        lib.run.restype = C.c_int32
        lib.load()
        return lib

    def batched(op):
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glm_kernels import BatchedBf16Gemm
        uk, fb = op["role"] == "mla_uk", op["role"] == "kda_fb"
        a_row, a_batch = (24752, 128) if fb else (48*192, 192) if uk else (48*512, 512)
        o_row, o_batch = (6144, 6144) if fb else (48*576, 576) if uk else (48*128, 128)
        batch = 1 if fb else 48
        launch = BatchedBf16Gemm(n=op["n"], k=op["k"], batch=batch,
            a_row=a_row, a_batch=a_batch, o_row=o_row, o_batch=o_batch,
            compute_warps=op.get("compute_warps", 4))
        return compile_program(launch, name="dense_kept_batched", operands=(
            Operand("a", torch.bfloat16, f"[rows,{a_row}]"),
            Operand("w", torch.bfloat16, f"[{batch},{op['n']},{op['k']}]"),
            Operand("out", torch.bfloat16, f"[rows,{o_row}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key(), geometry=op)

    for op in operations:
        role, k, n, members = op["role"], op["k"], op["n"], op["groups"]
        directory = args.output_dir/role
        directory.mkdir(parents=True, exist_ok=True)
        if args.compile_only:
            (directory/"contract.json").write_text(json.dumps(contract, sort_keys=True))
        elif json.loads((directory/"contract.json").read_text()) != contract:
            raise ValueError("stale full-core exports")
        capacity = 64 if op["head"] else 4096
        sizes = [640 if role == "mla_rank" and i == 1 else n for i in range(members)]
        partial_boundary = op["side"] == 2
        output_dtype = torch.float32 if op["output"] == "f32" else torch.bfloat16
        use_batched = op["batch"] > 1 or role == "kda_fb"
        baseline = None if op["head"] else program(directory, "baseline", lambda: batched(op) if use_batched else
            compile_bf16_projection_aot(width=k, outputs=op["packed_n"], output_dtype=output_dtype,
                wide=role not in ("layer0_gate_up", "layer0_down", "router") and
                     wide_prefill_min_rows(op["packed_n"], k, capacity) is not None))
        baseline_prefill = (program(directory, "baseline_prefill", lambda: batched(dict(op,compute_warps=8)))
                            if role == "kda_fb" else None)
        remainder = (program(directory, "remainder", lambda: compile_bf16_projection_aot(width=k, outputs=176))
                     if role == "kda" else None)
        ingress, egress, cores = [], [], {}
        if not op["native"]:
            grouped = members == 4
            for i in range(1 if grouped else members):
                ingress.append(program(directory, f"ingress{i}", lambda: compile_trellis_input_aot(
                    features=k, max_rows=capacity, output_dtype="bf16", groups=4 if grouped else 1)))
                egress.append(program(directory, f"egress{i}", lambda i=i: compile_trellis_output_aot(
                    features=sizes[i], max_rows=capacity, output_dtype=op["output"], groups=4 if grouped else 1)))
            for bits in (6, 8):
                cores[bits] = [program(directory, f"core{index}_k{bits}", lambda index=index, bits=bits:
                    compile_trellis_linear_aot(input_features=k, output_features=sizes[index], bits=bits,
                        codebook="mcg", sharded_side=op["side"], rotation_count=4 if grouped else 1,
                        groups=4 if grouped else 1, max_rows=capacity, input_dtype="bf16"))
                    for index in range(1 if grouped else members)]
        if args.compile_only:
            continue
        fixtures = {}
        for bits in (() if op["native"] else (6, 8)):
            fixtures[bits] = []
            for record in (r for r in manifests[bits] if r["role"] == role):
                path = args.fixture_dir/record["path"]
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b""):
                        digest.update(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("fixture checksum differs")
                fixtures[bits].append(torch.load(path, weights_only=True, map_location="cpu"))
        if not op["native"] and not all(torch.equal(a["weight"], b["weight"]) for a,b in zip(fixtures[6],fixtures[8])):
            raise ValueError("K6/K8 random source weights differ")
        weights = ([f["weight"].cuda() for f in fixtures[6]] if not op["native"] else
                   [torch.randn(op["batch"]*op["packed_n"], k, device="cuda", dtype=torch.bfloat16)])
        remainder_weight = torch.randn(176, k, device="cuda", dtype=torch.bfloat16) if role == "kda" else None
        if role == "mla_rank":
            packed_weight = torch.cat((weights[0], weights[1][:576])).contiguous()
        else:
            packed_weight = torch.cat(weights+([remainder_weight] if role == "kda" else []))
        input_width = 24752 if role == "kda_fb" else 48*192 if role == "mla_uk" else 48*512 if role == "mla_uv" else k
        output_width = 48*576 if role == "mla_uk" else 48*128 if role == "mla_uv" else op["packed_n"]
        x = torch.randn(capacity, input_width, device="cuda", dtype=torch.bfloat16)
        baseline_out = torch.empty(capacity, output_width, device="cuda", dtype=output_dtype)
        reduced = torch.empty(capacity, n, device="cuda", dtype=torch.float32) if partial_boundary else None
        baseline_final = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16) if partial_boundary else baseline_out
        peer = torch.randn(capacity, n, device="cuda", dtype=torch.float32) if partial_boundary else None
        peer_bf16 = peer.bfloat16() if partial_boundary else None
        raw = [torch.empty(capacity, sn, device="cuda", dtype=torch.float32) for sn in sizes]
        finals = {f"k{bits}": [torch.empty(capacity, sn, device="cuda",
                  dtype=torch.float32 if op["output"] == "f32" else torch.bfloat16) for sn in sizes] for bits in (6,8)}
        rotated = [torch.empty(capacity, k, device="cuda", dtype=torch.bfloat16) for _ in sizes]
        suhs = [torch.ones(k, device="cuda", dtype=torch.float16) for _ in sizes]
        scales = [torch.ones(k//128, device="cuda", dtype=torch.float16) for _ in sizes]
        svhs = [torch.ones(sn, device="cuda", dtype=torch.float16) for sn in sizes]
        tables = [torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64) for x in rotated]
        grouped_table = torch.tensor([x.data_ptr() for x in rotated], device="cuda", dtype=torch.int64)
        payloads = {bits:[f["payload"].cuda() for f in fs] for bits,fs in fixtures.items()}
        partial = torch.empty(129*((k+1023)//1024)*max(sizes)*members, device="cuda", dtype=torch.float32)
        retained = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16) if partial_boundary else None
        remainder_out = torch.empty(capacity,176,device="cuda",dtype=torch.bfloat16) if role == "kda" else None
        workspace, handle = None, C.c_void_p()
        if op["head"]:
            workspace = torch.empty(4<<20,device="cuda",dtype=torch.uint8)
            status = native.cuteafd_vocabulary_head_create_vocab(workspace.data_ptr(), workspace.numel(),
                k, capacity, n, 1, C.byref(handle))
            if status:
                raise RuntimeError(f"canonical head create failed {status}")
        def checked(status):
            if status:
                raise RuntimeError(f"full-core native launch returned {status}")
        def call(lib, *values):
            checked(lib.run(*values, torch.cuda.current_stream().cuda_stream))
        def run(arm, rows):
            stream = torch.cuda.current_stream().cuda_stream
            if op["native"] or arm == "bf16":
                if op["head"]:
                    checked(native.cuteafd_vocabulary_head_launch_tensor_invariant(handle,x.data_ptr(),
                        packed_weight.data_ptr(),baseline_out.data_ptr(),rows,stream))
                else:
                    selected = baseline_prefill if baseline_prefill is not None and rows > 64 else baseline
                    call(selected,x.data_ptr(),packed_weight.data_ptr(),baseline_out.data_ptr(),rows)
                if partial_boundary:
                    checked(native.cuteafd_add_tp2_shared_async(baseline_out.data_ptr(),peer_bf16.data_ptr(),
                        baseline_final.data_ptr(),rows*n,stream))
                return
            bits = int(arm[1:])
            if members == 4:
                call(ingress[0],x.data_ptr(),*(s.data_ptr() for s in suhs),*(t.data_ptr() for t in rotated),
                     *(s.data_ptr() for s in scales),rows,limit)
                call(cores[bits][0],grouped_table.data_ptr(),*(p.data_ptr() for p in payloads[bits]),
                     *(s.data_ptr() for s in scales),partial.data_ptr(),*(o.data_ptr() for o in raw),rows,0,limit)
                call(egress[0],*(o.data_ptr() for o in raw),*(s.data_ptr() for s in svhs),
                     *(o.data_ptr() for o in finals[arm]),rows,limit)
            else:
                for i in range(members):
                    call(ingress[i],x.data_ptr(),suhs[i].data_ptr(),rotated[i].data_ptr(),scales[i].data_ptr(),rows,limit)
                    call(cores[bits][i],tables[i].data_ptr(),payloads[bits][i].data_ptr(),scales[i].data_ptr(),
                         partial.data_ptr(),raw[i].data_ptr(),rows,0,limit)
                    if partial_boundary:
                        checked(native.cuteafd_load_narrow_f32_bf16(raw[i].data_ptr(),retained.data_ptr(),rows*n,
                            props.multi_processor_count,stream))
                        checked(native.cuteafd_bf16_f32_add_async(retained.data_ptr(),peer_bf16.data_ptr(),
                            reduced.data_ptr(),rows*n,props.multi_processor_count,stream))
                    call(egress[i],(reduced if partial_boundary else raw[i]).data_ptr(),svhs[i].data_ptr(),
                         finals[arm][i].data_ptr(),rows,limit)
            if remainder:
                call(remainder,x.data_ptr(),remainder_weight.data_ptr(),remainder_out.data_ptr(),rows)
        arms = ("native",) if op["native"] else ("k6","k8","bf16")
        live_rows = sorted(set([1]+[r for r in rows_list if r<=64])) if op["head"] else rows_list
        for rows in live_rows:
            graphs, events = {}, {}
            for arm in arms:
                targets = [baseline_final] if arm in ("native","bf16") else finals[arm]+([remainder_out] if remainder else [])
                if role == "mla_uk":
                    # UK writes the 512 latent columns, not each head's 64-column key tail.
                    targets = [baseline_final.view(capacity,48,576)[:,:,:512]]
                run(arm,rows)
                expected = [t[:rows].clone() for t in targets]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(arm,rows)
                for target in targets:
                    target[:rows].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not all(torch.equal(t[:rows].view(torch.uint8),e.view(torch.uint8)) and
                           torch.isfinite(t[:rows]).all().item() for t,e in zip(targets,expected)):
                    raise RuntimeError(f"{role}/{arm}/{rows}: poisoned replay differs")
                pairs = [(torch.cuda.Event(enable_timing=True,external=True),
                          torch.cuda.Event(enable_timing=True,external=True)) for _ in range(16 if rows<=64 else 4)]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for start,end in pairs:
                        flush.zero_()
                        start.record()
                        run(arm,rows)
                        end.record()
                graphs[arm],events[arm] = graph,pairs
                graph.replay()
            torch.cuda.synchronize()
            before = thermal_snapshot()
            samples = {arm:[] for arm in arms}
            for repeat in range(args.samples):
                for arm in (arms if repeat%2==0 else tuple(reversed(arms))):
                    graphs[arm].replay()
                    events[arm][-1][1].synchronize()
                    samples[arm].append(statistics.mean(a.elapsed_time(b)*1000 for a,b in events[arm]))
            times = {a:statistics.median(s) for a,s in samples.items()}
            logical = {"bf16":op["batch"]*op["packed_n"]*k*2}
            logical.update({f"k{b}":sum(k*sn*b//8+2*(k+sn) for sn in sizes)+(176*k*2 if remainder else 0) for b in (6,8)})
            if op["native"]:
                logical = {"native":logical["bf16"]}
            record = dict(role=role,rows=rows,per_verify_step_calls=op["count"],total_call_us=times,
                samples_us=samples,logical_weight_bytes=logical,
                logical_weight_gbs={a:logical[a]/(times[a]*1000) for a in arms},
                replay_exact=True,poisoned_outputs_overwritten=True,
                thermal_before=before,thermal_after=thermal_snapshot())
            report["records"].append(record)
            print(json.dumps({k:v for k,v in record.items() if k!="samples_us"}),flush=True)
            report["aggregate"] = aggregate_kimi_full_core(operations,report["records"],accounting,rows_list)
            (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        del graphs,events,expected,targets,graph,fixtures,payloads,weights,packed_weight,x,raw,finals
        del rotated,suhs,scales,svhs,tables,grouped_table,partial,baseline_out,baseline_final
        del retained,reduced,peer,peer_bf16,remainder_out,remainder_weight
        if handle.value:
            checked(native.cuteafd_v41_markov_destroy(handle))
        del workspace
        torch.cuda.empty_cache()


def benchmark_kimi_decision(args):
    """Full independent-projection launch sums, not concatenated trellis tensors."""
    import hashlib
    import torch
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot,
    )

    rows_list = list(map(int, args.rows.split(",")))
    if not rows_list or min(rows_list) < 1 or max(rows_list) > 4096 or args.samples < 50:
        raise ValueError("Kimi decision requires rows 1..4096 and at least 50 samples")
    if args.compile_only:
        import multiprocessing
        from b12x._lib.compile_pool import _initialize_worker
        _initialize_worker(0, (12, 0), "kimi-decision", "kimi-decision",
                           args.export_sms, 101376, 102400, multiprocessing.Array("i", [0, 0]))
    source = Path(__file__).resolve().parents[1] / "b12x/gemm/trellis_linear/_native.py"
    contract = dict(native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    capacity=4096, input_dtype="bf16", export_sms=args.export_sms)

    def program(directory, name, builder):
        if args.compile_only:
            print("compile", directory.name, name, flush=True)
            with exportable_compilation():
                result = builder()
            exported(result, directory, name, load_module=False)
            (directory / f"{name}.json").write_text(json.dumps(dict(
                pointers=len(result.operands), scalars=len(result.scalars))))
            return None
        abi = json.loads((directory / f"{name}.json").read_text())
        lib = C.CDLL(str((directory / f"{name}.so").resolve()))
        lib.load.argtypes, lib.load.restype = [], None
        lib.run.argtypes = [C.c_void_p]*abi["pointers"] + [C.c_int32]*abi["scalars"] + [C.c_void_p]
        lib.run.restype = C.c_int32
        lib.load()
        return lib

    report = dict(metadata=dict(contract=contract, samples=args.samples,
        cache="2x runtime L2 eviction outside each timed full-operation interval",
        scope="grouped KDA + BF16 N176 remainder; two independent gate/up projections; H128 included",
        equality="replay versus eager within each format; formats deliberately have different numerics",
        weights="official reference-encoded random tiles, no checkpoint calibration or quality claim",
        reference_bf16="ordered K16 DECODE_TMA_TILES plus measured wide-prefill route"), records=[])
    if not args.compile_only:
        props = torch.cuda.get_device_properties(0)
        status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0)
        if int(status):
            raise RuntimeError(status)
        limit = props.multi_processor_count * 4
        flush = torch.empty(2*l2, device="cuda", dtype=torch.uint8)
        report["metadata"].update(device=props.name, sms=props.multi_processor_count,
                                  l2_bytes=l2, thermal_initial=thermal_snapshot())
        fixture_records = {bits: json.loads((args.fixture_dir / f"fixtures_k{bits}.json").read_text())
                           for bits in (6, 8)}
        for records in fixture_records.values():
            if len(records) != 6 or not all(record["oracle_exact"] for record in records):
                raise ValueError("six independent oracle-qualified fixtures required")
        report["metadata"]["fixture_manifests"] = fixture_records
        catalog = json.loads(args.profile_catalog.read_text())
        accounting = kimi_profile_accounting(catalog, json.loads(args.profile_inventory.read_text()))
        calls_per_token = accounting["calls_per_emitted_token"]
        report["metadata"]["profile_accounting"] = accounting
        report["metadata"]["profile_calls_per_emitted_token"] = calls_per_token
        report["metadata"]["projection_scope"] = "KDA and shared gate/up only, not a full-core speed estimate"
        torch.manual_seed(811)
        warm = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(128):
            torch.mm(warm, warm)
        torch.cuda.synchronize()
        del warm
    for role, n, groups, indices, packed_n in (
        ("kda", 6144, 4, range(4), 24752),
        ("gate_up", 3072, 1, range(4, 6), 6144),
    ):
        directory = args.output_dir / role
        directory.mkdir(parents=True, exist_ok=True)
        if args.compile_only:
            (directory / "contract.json").write_text(json.dumps(contract, sort_keys=True))
        elif json.loads((directory / "contract.json").read_text()) != contract:
            raise ValueError("stale Kimi decision exports")
        bf16 = program(directory, "bf16", lambda: compile_bf16_projection_aot(
            width=7168, outputs=packed_n, wide=True))
        remainder = (program(directory, "remainder", lambda: compile_bf16_projection_aot(
            width=7168, outputs=176)) if role == "kda" else None)
        ingress = program(directory, "ingress", lambda: compile_trellis_input_aot(
            features=7168, max_rows=4096, output_dtype="bf16", groups=groups))
        egress = program(directory, "egress", lambda: compile_trellis_output_aot(
            features=n, max_rows=4096, groups=groups))
        cores = {bits: program(directory, f"core_k{bits}", lambda bits=bits: compile_trellis_linear_aot(
            input_features=7168, output_features=n, bits=bits, codebook="mcg",
            sharded_side=1, rotation_count=groups, max_rows=4096, groups=groups,
            input_dtype="bf16")) for bits in (6, 8)}
        if args.compile_only:
            continue
        fixtures = {}
        for bits in (6, 8):
            fixtures[bits] = []
            for index in indices:
                record = fixture_records[bits][index]
                path = args.fixture_dir / record["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"fixture checksum mismatch: {path}")
                fixtures[bits].append(torch.load(path, map_location="cpu", weights_only=True))
        if not all(torch.equal(a["weight"], b["weight"]) for a, b in zip(fixtures[6], fixtures[8])):
            raise ValueError("K6 and K8 must encode the same random source weights")
        weights = [f["weight"].cuda() for f in fixtures[6]]
        remainder_weight = torch.randn(176, 7168, device="cuda", dtype=torch.bfloat16) if remainder else None
        packed_weight = torch.cat(weights + ([remainder_weight] if remainder else []))
        x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
        rotated = [torch.empty_like(x) for _ in indices]
        suhs = [torch.ones(7168, device="cuda", dtype=torch.float16) for _ in indices]
        svhs = [torch.ones(n, device="cuda", dtype=torch.float16) for _ in indices]
        scales = [torch.ones(7168//128, device="cuda", dtype=torch.float16) for _ in indices]
        tables = [torch.tensor([t.data_ptr()], device="cuda", dtype=torch.int64) for t in rotated]
        grouped_table = torch.tensor([t.data_ptr() for t in rotated], device="cuda", dtype=torch.int64)
        payloads = {bits: [f["payload"].cuda() for f in fixtures[bits]] for bits in (6, 8)}
        partial = torch.empty(129*7*n*len(weights), device="cuda", dtype=torch.float32)
        raw = [torch.empty(4096, n, device="cuda", dtype=torch.float32) for _ in indices]
        finals = {f"k{bits}": [torch.empty(4096, n, device="cuda", dtype=torch.bfloat16) for _ in indices]
                  for bits in (6, 8)}
        packed_out = torch.empty(4096, packed_n, device="cuda", dtype=torch.bfloat16)
        remainder_out = torch.empty(4096, 176, device="cuda", dtype=torch.bfloat16) if remainder else None

        def call(lib, *values):
            result = lib.run(*values, torch.cuda.current_stream().cuda_stream)
            if result:
                raise RuntimeError(f"Kimi decision wrapper returned {result}")

        def run(arm, rows):
            if arm == "bf16":
                call(bf16, x.data_ptr(), packed_weight.data_ptr(), packed_out.data_ptr(), rows)
                return
            bits = int(arm[1:])
            if groups == 4:
                call(ingress, x.data_ptr(), *(s.data_ptr() for s in suhs),
                     *(t.data_ptr() for t in rotated), *(s.data_ptr() for s in scales), rows, limit)
                call(cores[bits], grouped_table.data_ptr(), *(p.data_ptr() for p in payloads[bits]),
                     *(s.data_ptr() for s in scales), partial.data_ptr(), *(o.data_ptr() for o in raw), rows, 0, limit)
                call(egress, *(o.data_ptr() for o in raw), *(s.data_ptr() for s in svhs),
                     *(o.data_ptr() for o in finals[arm]), rows, limit)
            else:
                for i in range(2):
                    call(ingress, x.data_ptr(), suhs[i].data_ptr(), rotated[i].data_ptr(), scales[i].data_ptr(), rows, limit)
                    call(cores[bits], tables[i].data_ptr(), payloads[bits][i].data_ptr(), scales[i].data_ptr(),
                         partial.data_ptr(), raw[i].data_ptr(), rows, 0, limit)
                    call(egress, raw[i].data_ptr(), svhs[i].data_ptr(), finals[arm][i].data_ptr(), rows, limit)
            if remainder:
                call(remainder, x.data_ptr(), remainder_weight.data_ptr(), remainder_out.data_ptr(), rows)

        arms = ("k6", "k8", "bf16")
        for rows in rows_list:
            graphs, events = {}, {}
            for arm in arms:
                targets = [packed_out] if arm == "bf16" else finals[arm] + ([remainder_out] if remainder else [])
                run(arm, rows)
                expected = [t[:rows].clone() for t in targets]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(arm, rows)
                for t in targets:
                    t[:rows].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not all(torch.equal(t[:rows].view(torch.int16), ref.view(torch.int16))
                           and torch.isfinite(t[:rows]).all().item() for t, ref in zip(targets, expected)):
                    raise RuntimeError(f"{role}/{arm}/{rows}: poisoned replay differs from eager")
                intervals = 16 if rows <= 64 else 4
                pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True)) for _ in range(intervals)]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for start, end in pairs:
                        flush.zero_()
                        start.record()
                        run(arm, rows)
                        end.record()
                graphs[arm], events[arm] = graph, pairs
                graph.replay()
            torch.cuda.synchronize()
            before = thermal_snapshot()
            samples = {arm: [] for arm in arms}
            for repeat in range(args.samples):
                order = arms if repeat % 2 == 0 else tuple(reversed(arms))
                for arm in order:
                    graphs[arm].replay()
                    events[arm][-1][1].synchronize()
                    samples[arm].append(statistics.mean(a.elapsed_time(b)*1000 for a, b in events[arm]))
            times = {arm: statistics.median(values) for arm, values in samples.items()}
            logical = {f"k{bits}": len(weights)*(7168*n*bits//8 + 2*(7168+n)) + (176*7168*2 if remainder else 0)
                       for bits in (6, 8)}
            logical["bf16"] = packed_n*7168*2
            record = dict(role=role, rows=rows, total_call_us=times, samples_us=samples,
                profile_calls_per_emitted_token=calls_per_token[role],
                projected_us_per_emitted_token={a: times[a]*calls_per_token[role] for a in arms},
                logical_weight_bytes=logical, logical_weight_gbs={a: logical[a]/(times[a]*1000) for a in arms},
                replay_exact=True, poisoned_outputs_overwritten=True,
                thermal_before=before, thermal_after=thermal_snapshot())
            report["records"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "samples_us"}), flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        del graphs, events, expected, targets, fixtures, payloads, weights, packed_weight, x, rotated
        del suhs, svhs, scales, tables, grouped_table, partial, raw, finals, packed_out, remainder_out, remainder_weight


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
    parser.add_argument("--groups", type=int, choices=(1, 4), default=1,
                        help="standalone or independent grouped KDA threshold comparison")
    parser.add_argument("--export-sms", type=int, choices=(170, 188), default=188,
                        help="offline SM120 compiler geometry; runtime CTA budget stays device-derived")
    parser.add_argument("--encoder-source", type=Path, help="official GPTQModel gptqmodel_ext/exllamav3 source")
    parser.add_argument("--encoder-build-only", action="store_true", help="CPU compile official quantize/pack operators")
    parser.add_argument("--encoder-library", type=Path, help="prebuilt official quantize/pack operators for fixture generation")
    parser.add_argument("--encode-fixtures", action="store_true", help="encode independent full Kimi KDA and gate/up fixtures")
    parser.add_argument("--kimi-decision", action="store_true", help="full K6/K8/BF16 KDA and gate/up launch sums")
    parser.add_argument("--full-core", action="store_true", help="extend Kimi decision and fixtures to all local dense operations")
    parser.add_argument("--checkpoint-config", type=Path, help="actual checkpoint config.json for full-core multiplicities")
    parser.add_argument("--native-library", type=Path, help="shipping native DSO plus adjacent COMPILER_PROVENANCE.json")
    parser.add_argument("--fixture-dir", type=Path, help="independently reference-encoded K6/K8 fixture directory")
    parser.add_argument("--profile-catalog", type=Path, help="baseline kernel-catalog.json for per-emitted-token call counts")
    parser.add_argument("--profile-inventory", type=Path, help="decode kernel-inventory.json disambiguating shared gate/up and MLA gate symbols")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.encoder_build_only:
        if args.encoder_source is None:
            parser.error("--encoder-build-only requires --encoder-source")
        hashes = reference_encoder(args.encoder_source, args.output_dir / "encoder")
        (args.output_dir / "encoder_sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
        return
    if args.full_core and (args.checkpoint_config is None or not (args.encode_fixtures or args.kimi_decision)):
        parser.error("--full-core requires --checkpoint-config and --encode-fixtures or --kimi-decision")
    if args.encode_fixtures:
        if args.encoder_library is None or args.bits not in (6, 8) or args.codebook not in ("mcg", "mul1"):
            parser.error("--encode-fixtures requires --encoder-library, K6/K8 and MCG/MUL1")
        encode_reference_fixtures(args)
        return
    if args.kimi_decision:
        if not args.compile_only and (args.fixture_dir is None or args.profile_catalog is None
                                      or args.profile_inventory is None):
            parser.error("--kimi-decision requires --fixture-dir, --profile-catalog and --profile-inventory")
        if args.full_core:
            if not args.compile_only and args.native_library is None:
                parser.error("--full-core measurement requires --native-library")
            benchmark_kimi_full_core(args)
        else:
            benchmark_kimi_decision(args)
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
