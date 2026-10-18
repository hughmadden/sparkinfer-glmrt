#!/usr/bin/env python3
"""Graph-first matched-byte EXLR8 up1 versus EXL3 K4 Spark-step gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import benchmark_mixed_trellis as exl3
from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot


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


def stage_profile(graph, directory):
    # Profiling is a separate replay: it never supplies the performance verdict.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", dir=directory) as stream:
        prof.export_chrome_trace(stream.name)
        events = json.loads(Path(stream.name).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    stages = {}
    for stage in ("fc1", "activation", "fc2", "reduce"):
        matches = [e for e in kernels if stage+"_kernel" in e.get("name", "")]
        if not matches:
            raise RuntimeError(f"profiler did not identify {stage} kernel")
        stages[stage+"_us"] = sum(e["dur"] for e in matches)
    span = max(e["ts"]+e["dur"] for e in kernels)-min(e["ts"] for e in kernels)
    stages["launch_gaps_us"] = max(0.0, span-sum(e["dur"] for e in kernels))
    stages["profiled_span_us"] = span
    return stages


def make_candidate(experts=256, hidden=6144, units=4, topk=8):
    owners = []
    pointers = torch.empty((experts, units, 3, 3), dtype=torch.int64)
    for e in range(experts):
        for u in range(units):
            for p in range(3):
                tensors = (torch.randint(0, 256, (hidden*16*4,), dtype=torch.uint8, device="cuda"),
                           torch.ones(128, dtype=torch.float16, device="cuda"),
                           torch.ones(hidden//128, dtype=torch.float16, device="cuda"))
                owners.extend(tensors)
                pointers[e,u,p] = torch.tensor([t.data_ptr() for t in tensors])
    groups = 64*topk
    tensors = [torch.randn((64,hidden), dtype=torch.float16, device="cuda")*.02,
               pointers.to("cuda"), torch.full((experts,units,3),4,dtype=torch.int32,device="cuda"),
               torch.full((groups,),-1,dtype=torch.int32,device="cuda"),
               torch.full((groups,16),-1,dtype=torch.int32,device="cuda"),
               torch.full((64,topk),-1,dtype=torch.int32,device="cuda"),
               torch.full((64,topk),1/topk,device="cuda"),
               torch.empty((groups,units,2,(hidden+1023)//1024,16,128),device="cuda"),
               torch.empty((groups,units,16,128),dtype=torch.float16,device="cuda"),
               torch.empty((groups,16,hidden),device="cuda"),
               torch.empty((64,hidden),device="cuda")]
    return tensors, owners


def pack(tensors, ids):
    slots = ids.cpu().numpy()
    expert, routes = [], []
    inverse = np.full(slots.shape,-1,dtype=np.int32)
    for e in sorted(set(slots.flat)):
        positions = np.flatnonzero(slots.ravel()==e)
        for start in range(0,len(positions),16):
            subset = positions[start:start+16]
            g = len(expert)
            expert.append(int(e))
            routes.append(list(subset)+[-1]*(16-len(subset)))
            inverse.flat[subset] = g*16+np.arange(len(subset))
    tensors[3][:len(expert)] = torch.tensor(expert,dtype=torch.int32,device="cuda")
    tensors[4][:len(expert)] = torch.tensor(routes,dtype=torch.int32,device="cuda")
    tensors[5][:len(ids)] = torch.from_numpy(inverse).cuda()
    return len(expert)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows",default="1,2,3,4,5,6,7,8,16,64")
    parser.add_argument("--replays",type=int,default=100)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--copy-roofline-gbps",type=float,required=True,
                        help="measured GB10 streaming-copy bytes/s ceiling, not a datasheet rate")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12,1):
        parser.error("this performance gate requires SM121")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    tensors, owners = make_candidate()
    candidate = compile_exlr8_a16_aot(units=4)
    # This is the existing EXL3 uniform-K4 implementation, not a proxy GEMM.
    tier = exl3._prepared(4,256,408,torch.device("cuda"))
    expert_map = torch.arange(256,dtype=torch.int32,device="cuda")
    records = []
    for rows in map(int,args.rows.split(",")):
        gen = torch.Generator().manual_seed(408+rows)
        ids = torch.stack([torch.randperm(256,generator=gen)[:8] for _ in range(rows)]).cuda().int()
        groups = pack(tensors,ids)
        cta_limit = props.multi_processor_count*2
        # L2 is recorded, never compiled in. Launch limiting changes assignment only.
        launch = lambda: candidate.launch(*(t.flatten() for t in tensors),scalars=(rows,groups,cta_limit))
        state = (tier,*exl3._serial_state(rows,tier,expert_map,props,torch.device("cuda")))
        baseline_x = tensors[0][:rows].to(torch.bfloat16)
        baseline = lambda: exl3._run_serial(state,baseline_x,tensors[6][:rows],ids)
        baseline_graph, candidate_graph = capture(baseline), capture(launch)
        baseline_us, candidate_us = timing(baseline_graph,args.replays), timing(candidate_graph,args.replays)
        payload_bytes = int(ids.unique().numel())*3*6144*512*4//8
        ratio = baseline_us/candidate_us
        records.append(dict(rows=rows,groups=groups,payload_bytes=payload_bytes,
                            exl3_k4_us=baseline_us,exlr8_up1_us=candidate_us,speedup=ratio,
                            exl3_gbps=payload_bytes/baseline_us/1000,
                            exlr8_gbps=payload_bytes/candidate_us/1000,
                            roofline_percent=100*payload_bytes/candidate_us/1000/args.copy_roofline_gbps,
                            pass_bar=0.97,verdict="PASS" if ratio>=0.97 else "FAIL",
                            **stage_profile(candidate_graph,args.output.parent)))
    result = dict(schema=1,kind="exlr8_up1_vs_exl3_k4",device=props.name,
                  sm_count=props.multi_processor_count,l2_bytes=props.L2_cache_size,
                  hidden=6144,units_per_rank=4,top_k=8,experts=256,
                  measured_copy_roofline_gbps=args.copy_roofline_gbps,
                  conditions="TP4 synthetic K4 payloads, graph replay, one untimed warm batch per rows; profiler separate",
                  rows=records,verdict="PASS" if all(r["verdict"]=="PASS" for r in records) else "FAIL")
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
