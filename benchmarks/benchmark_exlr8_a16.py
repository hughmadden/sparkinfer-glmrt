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
from pathlib import Path
import statistics
import sys
import tempfile

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


def compile_grouped_control(path, candidate):
    """Freeze the previous grouped core while sharing the unchanged serving ABI."""
    from b12x.integration.cuteafd import _common
    from b12x.integration.cuteafd.exlr8 import _Exlr8ServingPointers
    core, digest = load_source("grouped_control", path,
                               "b12x.moe._shared.kernels.w4a16")
    g = candidate.geometry
    wrapper = object.__new__(_Exlr8ServingPointers)
    wrapper.core = core.Exlr8GroupedKernel(g["hidden"], g["units"], g["top_k"],
                                          g["max_groups"], g["group_rows"])
    program = _common.compile_program(wrapper, name="exlr8_grouped_control",
        operands=candidate.operands, scalars=candidate.scalars,
        key=(digest, tuple(sorted(g.items()))), geometry=g, scratch=candidate.scratch)
    return program, digest


def make_candidate(l2_bytes, experts=256, hidden=6144, units=4, topk=8):
    """Synthetic K4 payloads, replicated into enough distinct layers that a
    replay over them streams every weight from DRAM (cold L2)."""
    owners = []
    unit_bytes = hidden * 16 * 4
    layer_bytes = experts * units * 3 * unit_bytes
    copies = max(2, -(-L2_SPAN * l2_bytes // layer_bytes))
    tables = []
    for _ in range(copies):
        pointers = torch.empty((experts, units, 3, 3), dtype=torch.int64)
        for e in range(experts):
            for u in range(units):
                for p in range(3):
                    tensors = (torch.randint(0, 256, (unit_bytes,), dtype=torch.uint8, device="cuda"),
                               torch.ones(128, dtype=torch.float16, device="cuda"),
                               torch.full((hidden//128,), 1 / hidden**0.5, dtype=torch.float16, device="cuda"))
                    owners.extend(tensors)
                    pointers[e,u,p] = torch.tensor([t.data_ptr() for t in tensors])
        tables.append(pointers.to("cuda"))
    gen = torch.Generator().manual_seed(4080)
    signs = [(torch.randint(2, (hidden,), generator=gen) * 2 - 1).char().cuda() for _ in range(2)]
    groups = 64*topk
    return dict(tables=tables, owners=owners, signs_in=signs[0], signs_out=signs[1],
                bits=torch.full((experts,units,3),4,dtype=torch.int32,device="cuda"),
                group_expert=torch.full((groups,),-1,dtype=torch.int32,device="cuda"),
                route_slots=torch.full((groups,16),-1,dtype=torch.int32,device="cuda"),
                route_group_lane=torch.full((64,topk),-1,dtype=torch.int32,device="cuda"),
                group_count=torch.zeros(1,dtype=torch.int32,device="cuda"),
                activation=torch.empty((64,hidden),dtype=torch.float16,device="cuda"),
                fc1=torch.empty((groups,units,2,(hidden+1023)//1024,16,128),device="cuda"),
                intermediate=torch.empty((groups,units,16,128),dtype=torch.float16,device="cuda"),
                fc2=torch.empty((groups,16,hidden),device="cuda"),
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
    parser.add_argument("--rows",default="1,2,3,4,5,6,7,8,16,64,4096")
    parser.add_argument("--replays",type=int,default=20)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--control-core",type=Path,
                        help="master's K1 core source (b12x/moe/_shared/kernels/w4a16/exlr8.py)")
    parser.add_argument("--control-glue",type=Path,
                        help="master's grouping source (b12x/integration/cuteafd/exlr8_glue.py)")
    parser.add_argument("--control-grouped-core", type=Path,
                        help="frozen previous grouped core for a same-session fourth arm")
    parser.add_argument("--repeats",type=int,default=1,
                        help="interleaved timing rounds; arm order rotates each round, medians reported")
    parser.add_argument("--profile",action="store_true",
                        help="one eager launch per arm inside cudaProfilerStart/Stop (for ncu); no timing")
    parser.add_argument("--profile-arm", choices=("grouped", "exl3", "mixed"), default="grouped",
                        help="grouped benchmark arm inside cudaProfilerStart/Stop")
    args = parser.parse_args()
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile")
    if bool(args.control_core) != bool(args.control_glue):
        parser.error("--control-core and --control-glue go together")
    if args.control_grouped_core and not args.grouped:
        parser.error("--control-grouped-core requires --grouped")
    if torch.cuda.get_device_capability() != (12,1):
        parser.error("this performance gate requires SM121")
    torch.cuda.set_per_process_memory_fraction(min(
        0.25, (12 << 30) / torch.cuda.get_device_properties(0).total_memory))
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
        ids = torch.stack([torch.randperm(256,generator=gen)[:8] for _ in range(total)]).cuda().int()
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
            t["control_x"][:rows].copy_(x[first:first+rows])
            k1c, groupc = control
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


def grouped_mixed_state(rows, tiers, props):
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mix
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
    tile = _projection_mixed_tile_config(None, hidden_size=6144,
        intermediate_size=512, token_count=4096, direct_topk_routes=False)
    launch = mix.compile_mixed_trellis(
        size_m=4096, hidden_size=6144, intermediate_size=512,
        tier0_num_experts=256, tier1_num_experts=1, tier0_bits=4, tier1_bits=5,
        route_num_experts=256,
        top_k=8, max_m_blocks=(max_packed_route_slots(4096*8, 32, 256)+31)//32,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=tile, trellis_codebook=tiers[0].trellis_codebook,
        moe_block_size=32, warp_specialized=True, input_format="e4m3_k32",
        ws_input_stages=3, ws_dynamic_tiles=True,
        rotation_input_dtype="bf16", full_rotation_output_dtype="bf16")
    bufs = mix.make_mixed_trellis_buffers(launch, device=torch.device("cuda"), sms=props.multi_processor_count)
    # The production mixed ABI requires distinct tiers. Every routed expert
    # is K4; one unreachable K5 slot keeps that ABI without changing its kernels.
    mapping, desc = mix.build_ordered_maps(256, 1, device=torch.device("cuda"))
    mapping = mapping[:256]
    rotations = mix.MixedTrellisRotations(
        torch.cat([t.intermediate_rotations for t in tiers]),
        torch.cat([t.gate_suh for t in tiers]), torch.cat([t.up_suh for t in tiers]),
        torch.cat([t.down_svh for t in tiers]))
    binding = mix.bind_mixed_trellis(*tiers, mapping, desc, rotations, launch)
    return binding, bufs


def grouped_uniform_state(rows, tier, props):
    from b12x.moe._shared.kernels.w4a16.host import (
        make_w4a16_packed_buffers, select_route_block_size_m)
    # Native tiles do not depend on launch K/N geometry. Share their storage,
    # but retain the uniform serving route's historical K64/N256 policy.
    tile = (64, 256, 64, 256)
    tier = dataclasses.replace(tier, fc1_tile_n=256, fc2_tile_n=256,
                              trellis=dataclasses.replace(tier.trellis, tile_config=tile))
    block = select_route_block_size_m(rows, 8, 256)
    bufs = make_w4a16_packed_buffers(tier, m=rows, topk=8, dtype=torch.float16,
        device=torch.device("cuda"), full_rotation=True, block_size_m=block)
    bufs = dataclasses.replace(bufs,
        output=torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda"))
    launch = exl3.compile_w4a16_fused_moe(
        size_m=rows, hidden_size=6144, intermediate_size=512, num_experts=256,
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
    t = make_candidate(l2)
    control = compile_control(args.control_core, args.control_glue)
    limit = int(props.multi_processor_count) * 4
    group_rows = args.group_rows
    capacity = (4096*8 + group_rows-1)//group_rows + 256
    group = compile_exlr8_group_aot(hidden=6144, top_k=8, max_groups=capacity,
                                    max_rows=4096, group_rows=group_rows)
    grouped = compile_exlr8_a16_aot(units=4, top_k=8, max_groups=capacity,
                                    device_count=True, grouped=True, group_rows=group_rows)
    previous, previous_digest = (compile_grouped_control(args.control_grouped_core, grouped)
                                if args.control_grouped_core else (None, None))
    # Secondary only: mixed K4/K5 with an unreachable K5 slot is not the
    # uniform K4 serving route used for the primary speed bar.
    exl_layers = [(exl3._prepared(4,256,408+2*c,torch.device("cuda")),
                   exl3._prepared(5,1,409+2*c,torch.device("cuda")))
                  for c in range(len(t["tables"]))]
    state = buffers(4096, 6144, 4, 8, capacity, group_rows)
    mixed_states = [grouped_mixed_state(4096, ts, props) for ts in exl_layers]
    records = []
    for total in map(int, args.rows.split(",")):
        if not 1 <= total <= 4096:
            raise ValueError("grouped benchmark rows must be 1..4096")
        gen = torch.Generator().manual_seed(408+total)
        ids = torch.stack([torch.randperm(256,generator=gen)[:8] for _ in range(total)]).cuda().int()
        weights = torch.rand((total,8),generator=gen).cuda()/8
        wire = make_wire(torch.randn((total,6144),generator=gen).cuda()*0.02)
        decoded = torch.empty((total,6144),dtype=torch.bfloat16,device="cuda")
        uniform_states = [grouped_uniform_state(total, ts[0], props) for ts in exl_layers]
        chunk_response = torch.empty((total,6144),dtype=torch.bfloat16,device="cuda")
        chunk_wire = torch.empty((64,6336),dtype=torch.uint8,device="cuda")
        chunk_ids = torch.empty((64,8),dtype=torch.int32,device="cuda")
        chunk_weights = torch.empty((64,8),device="cuda")
        copies = len(t["tables"])

        def chunked():
            k1c, gc = control
            for table in t["tables"]:
                for first in range(0,total,64):
                    live = min(64,total-first)
                    chunk_wire[:live].copy_(wire[first:first+live])
                    chunk_ids[:live].copy_(ids[first:first+live])
                    chunk_weights[:live].copy_(weights[first:first+live])
                    gc.launch(chunk_ids,chunk_wire,t["signs_in"],t["group_expert"],t["route_slots"],
                              t["route_group_lane"],t["group_count"],t["activation"],scalars=(live,limit))
                    k1c.launch(t["activation"],table,t["bits"],t["group_expert"],t["route_slots"],
                               t["route_group_lane"],chunk_weights,t["fc1"],t["intermediate"],t["fc2"],
                               t["signs_out"],t["output"],t["group_count"],scalars=(live,limit))
                    chunk_response[first:first+live].copy_(t["output"][:live])

        def candidate(program=grouped):
            for table in t["tables"]:
                group.launch(ids,wire,t["signs_in"],state["group_expert"],state["route_slots"],
                             state["inverse"],state["count"],state["activation"],scalars=(total,limit))
                program.launch(state["activation"],table,t["bits"],state["group_expert"],state["route_slots"],
                               state["inverse"],weights,state["fc1"],state["intermediate"],state["fc2"],
                               t["signs_out"],state["output"],state["count"],scalars=(total,limit))

        def baseline():
            for uniform in uniform_states:
                # Uniform full_rotation accepts BF16/FP16, not FP8 wire rows.
                # Charge the worker's FP8-K32 -> BF16 conversion in this arm.
                wire_decode(wire, decoded, total)
                run_grouped_uniform(uniform, decoded, weights, ids)

        def mixed():
            for binding, bufs in mixed_states:
                run_bound_mixed_trellis(wire,weights,ids,binding,bufs)

        chunked(); candidate(); baseline(); mixed()
        torch.cuda.synchronize()
        if not torch.equal(chunk_response.view(torch.int16),state["output"][:total].view(torch.int16)):
            n = int((chunk_response.view(torch.int16) != state["output"][:total].view(torch.int16)).sum())
            raise RuntimeError(f"grouped result differs from snapshot chunked K1: {n} BF16 words")
        finite_nonzero(state["output"][:total].float(), "grouped EXLR8")
        finite_nonzero(uniform_states[-1][1].output.float(), "uniform grouped EXL3")
        finite_nonzero(mixed_states[-1][1].output[:total].float(), "mixed grouped EXL3")
        if previous is not None:
            candidate(previous)
            torch.cuda.synchronize()
            if not torch.equal(chunk_response.view(torch.int16), state["output"][:total].view(torch.int16)):
                raise RuntimeError("frozen grouped control differs from snapshot chunked K1")
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            {"grouped":candidate, "exl3":baseline, "mixed":mixed}[args.profile_arm]()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled arm={args.profile_arm} rows={total}", flush=True)
            continue
        graphs = {"chunked":capture(chunked),"grouped":capture(candidate),
                  "exl3":capture(baseline),"mixed_unreachable_k5":capture(mixed)}
        if previous is not None:
            graphs["previous_grouped"] = capture(lambda: candidate(previous))
        rounds = {n:[] for n in graphs}
        names = list(graphs)
        for r in range(min(3,args.repeats)):
            for n in names[r%len(names):]+names[:r%len(names)]:
                rounds[n].append(timing(graphs[n],args.replays)/copies)
        med = {n:statistics.median(ts) for n,ts in rounds.items()}
        unique = int(ids.unique().numel()) * 3 * 6144 * 512 * 4//8
        stages = stage_profile(graphs["grouped"], args.output.parent,
            {"ingress":("_Group",),"fc1":("grouped_fc1_kernel",),"activation":("activation_kernel",),
             "fc2":("grouped_fc2_kernel",),"egress":("reduce_kernel",)})
        rec = dict(rows=total, **{n+"_us":v for n,v in med.items()}, rounds_us=rounds,
                   chunked_speedup=med["chunked"]/med["grouped"],
                   exl3_speedup=med["exl3"]/med["grouped"], unique_weight_bytes=unique,
                   unique_read_roofline_percent=100*unique/med["grouped"]/1000/roof["gbps"],
                   stages={k:v/copies if k.endswith("_us") else v for k,v in stages.items()},
                   groups=int(state["count"].item()), bitwise_equal=True,
                   uniform_route_block=uniform_states[0][-1],
                   uniform_input_conversion="fp8_k32_to_bf16_included")
        records.append(rec)
        print(json.dumps(rec),flush=True)
        del graphs, uniform_states
    if not args.profile:
        args.output.write_text(json.dumps(dict(schema=3, device=props.name, sm_count=props.multi_processor_count,
            l2_bytes=l2, measured_read_roofline=roof, group_rows=group_rows, rows=records,
            primary_baseline="uniform K4 full_rotation; wire decode included",
            secondary_baseline="mixed K4/K5; K5 descriptor unreachable",
            previous_grouped_source_sha256=previous_digest,
            target_met=all(r["exl3_speedup"]>=1.0 for r in records)),indent=2))


if __name__ == "__main__":
    main()
