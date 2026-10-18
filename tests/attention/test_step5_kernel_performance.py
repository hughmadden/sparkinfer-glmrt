"""Opt-in bounded latency census, not an end-to-end speedup benchmark.

Run with -m step5_perf, STEP5_RUN_PERF=1 and STEP5_PEAK_BW_GBS set to the
card's published memory bandwidth. Byte floors assume perfect within-request
KV/summary reuse; cache-resident cases may exceed external-memory bandwidth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics

import pytest
import torch

from ..conftest import require_b12x

pytestmark = [pytest.mark.step5_perf,
              pytest.mark.skipif(os.environ.get("STEP5_RUN_PERF") != "1", reason="opt-in kernel latency census")]


def _median_graph_us(function, *, samples=9, replays=16):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    timings = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / replays)
    return statistics.median(timings)


def test_step5_kernel_bandwidth_census():
    from b12x.attention.paged._selected_forward import launch_sparse_gqa_split, launch_sparse_gqa_merge
    from b12x.attention.paged._weighted_regions import compress_regions, score_regions
    from b12x.attention.paged._region_selection import RegionSelectionWorkspace, select_regions, sliding_positions
    device = require_b12x()
    bandwidth = float(os.environ["STEP5_PEAK_BW_GBS"])
    assert bandwidth > 0
    records = []
    identity = dict(device=torch.cuda.get_device_name(device),
                    capability=list(torch.cuda.get_device_capability(device)),
                    peak_memory_gbs=bandwidth)
    torch.manual_seed(274)
    # One launch per architecture; decode concurrency and single-request verify.
    for context in (8192, 65536, 262144):
        for mode, requests, verify in (("decode", 1, 1), ("decode", 16, 1),
                                       ("verify", 1, 4), ("verify", 1, 8), ("verify", 1, 16)):
            rows = requests * verify
            groups, heads, dim, page, splits = 4, 64, 192, 16, 16
            capacity = context // 8
            req_ids = torch.arange(requests, dtype=torch.int32, device=device).repeat_interleave(verify)
            positions = (torch.arange(context-verify, context, dtype=torch.int64, device=device)
                         .repeat(requests))
            q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
            index_q = torch.randn(rows, groups, 4, 256, dtype=torch.bfloat16, device=device)
            weights = torch.randn(rows, groups, 4, device=device)
            summaries = torch.randn(requests, capacity, 256, device=device).to(torch.float8_e4m3fn)
            scores = torch.empty(rows, groups, capacity, device=device)
            pages = context // page
            k = torch.empty(requests*pages, page, groups, dim, dtype=torch.bfloat16, device=device)
            v = torch.empty_like(k)
            # Touch the whole live pool once, outside timing. Constant KV keeps
            # correctness checks cheap without changing address/traffic geometry.
            k.fill_(0.25)
            v.fill_(0.5)
            table = torch.arange(requests*pages, device=device, dtype=torch.int32).view(requests, pages)
            selected = torch.empty(rows, groups, 4104, dtype=torch.int32, device=device)
            swa = torch.empty(rows, 512, dtype=torch.int32, device=device)
            ws = RegionSelectionWorkspace.allocate(max_rows=rows, groups=groups, capacity=capacity, device=device)
            output = torch.empty_like(q)
            partial = torch.empty(rows, splits, heads, dim, device=device)
            lse = torch.empty(rows, splits, heads, device=device)

            def score():
                score_regions(query=index_q, weights=weights, summaries=summaries,
                              requests=req_ids, positions=positions, output=scores)

            def select():
                select_regions(scores=scores, positions=positions, requests=req_ids,
                               selected=selected, workspace=ws)

            def attend(selection):
                launch_sparse_gqa_split(query=q, key_cache=k, value_cache=v,
                    k_descale=None, v_descale=None, block_table=table, request_ids=req_ids,
                    selected_positions=selection, query_positions=positions,
                    partial_output=partial, partial_lse=lse,
                    softmax_scale=dim**-0.5, splits=splits)
                launch_sparse_gqa_merge(partial_output=partial, partial_lse=lse,
                                        output=output, rows=rows, splits=splits)

            # Prepare real provider-specific selections before attention timing.
            score()
            select()
            sliding_positions(positions=positions, requests=req_ids, selected=swa)
            # Ceiling counts compulsory bytes, not internal repeats or split scratch.
            score_bytes = requests*capacity*256 + rows*groups*(4*256*2+4*4+capacity*4)
            io_bytes = rows*heads*dim*2*2
            csa_bytes = requests*4104*groups*dim*2*2 + io_bytes + rows*groups*4104*4
            swa_bytes = requests*512*groups*dim*2*2 + io_bytes + rows*512*4
            operations = [("score", score, score_bytes),
                          ("selection", select, rows*groups*capacity*4 + selected.numel()*4),
                          ("selected_csa", lambda: attend(selected), csa_bytes),
                          ("selected_swa", lambda: attend(swa), swa_bytes)]
            for name, function, byte_floor in operations:
                lower_bound_us = byte_floor / (bandwidth*1000)
                print(json.dumps(dict(phase="ceiling", kernel=name, context=context,
                                      mode=mode, requests=requests, verify=verify,
                                      bytes=byte_floor, ideal_memory_us=lower_bound_us)), flush=True)
                elapsed = _median_graph_us(function)
                record = dict(kernel=name, context=context, mode=mode, requests=requests,
                              verify=verify, rows=rows, median_us=elapsed,
                              compulsory_bytes=byte_floor, ideal_memory_us=lower_bound_us,
                              bandwidth_fraction=lower_bound_us/elapsed)
                print(json.dumps(record), flush=True)
                records.append(record)
                if name.startswith("selected"):
                    torch.testing.assert_close(output, torch.full_like(output, 0.5), atol=0.004, rtol=0)
            # Bound the compressor at one completed region per live query.
            raw_k = torch.randn(rows*8, 256, dtype=torch.bfloat16, device=device)
            raw_z = torch.randn_like(raw_k)
            starts = torch.arange(rows, dtype=torch.int32, device=device)*8
            counts = torch.full((rows,), 8, dtype=torch.int32, device=device)
            compressed = torch.empty(rows, 256, dtype=torch.float8_e4m3fn, device=device)
            unrounded = torch.empty(rows, 256, device=device)
            def compress():
                compress_regions(keys=raw_k, logits=raw_z, starts=starts, counts=counts,
                                 output=compressed, unrounded=unrounded)
            byte_floor = rows*(8*256*2*2 + 256*5)
            lower_bound_us = byte_floor/(bandwidth*1000)
            print(json.dumps(dict(phase="ceiling", kernel="compress", context=context,
                                  mode=mode, requests=requests, verify=verify,
                                  bytes=byte_floor, ideal_memory_us=lower_bound_us)), flush=True)
            elapsed = _median_graph_us(compress)
            record = dict(kernel="compress", context=context, mode=mode, requests=requests,
                          verify=verify, rows=rows, median_us=elapsed,
                          compulsory_bytes=byte_floor, ideal_memory_us=lower_bound_us,
                          bandwidth_fraction=lower_bound_us/elapsed)
            records.append(record)
            print(json.dumps(record), flush=True)
            del k, v, summaries, scores, ws, selected, swa, partial, output, lse
    path = os.environ.get("STEP5_PERF_OUTPUT")
    if path:
        Path(path).write_text(json.dumps(dict(**identity, samples=9, graph_replays=16,
                                            records=records), indent=2) + "\n")
