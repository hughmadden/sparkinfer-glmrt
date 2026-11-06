#!/usr/bin/env python3
"""Paired HQA TP2 q_b exact launch-partition screen (no run orchestration)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=31)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('new output and positive samples required')
    if args.preflight:
        import ast
        ast.parse(Path(__file__).read_text())
        print('HQA round-1 preflight passed')
        return
    import torch
    from benchmarks.common import make_l2_flush_fn
    from benchmarks.benchmark_hqa_projection_ceiling import snapshot
    from b12x.integration.cuteafd._fp8_weights import FP8_GEMV_CONFIG, FP8_GEMV_WIDE
    from b12x.integration.cuteafd.glmf_split import compile_glmf_fp8_linear_aot
    from tests.integration.test_cuteafd_glmf_native_stages import test_flash_standalone_block_fp8_matches_fused

    torch.manual_seed(836)
    n, k = 8192, 1536
    candidate_config, candidate_wide = FP8_GEMV_CONFIG.pop((n,k)), FP8_GEMV_WIDE.pop((n,k))
    try:
        base = compile_glmf_fp8_linear_aot(input_features=k, output_features=n, max_rows=64)
    finally:
        FP8_GEMV_CONFIG[n,k], FP8_GEMV_WIDE[n,k] = candidate_config, candidate_wide
    candidate = compile_glmf_fp8_linear_aot(input_features=k, output_features=n, max_rows=64)
    assert base.geometry == candidate.geometry
    x = (torch.randn(64,k,device='cuda')*.1).bfloat16()
    w = (torch.randn(n,k,device='cuda')*.1).to(torch.float8_e4m3fn)
    s = torch.rand(n//128,k//128,device='cuda')*.3+.05
    out = {a: torch.empty(64,n,device='cuda',dtype=torch.bfloat16) for a in ('base','candidate')}
    scratch = torch.empty(256,device='cuda',dtype=torch.uint8)
    programs = {'base':base, 'candidate':candidate}
    flush = make_l2_flush_fn(True)
    result = {'shape':[n,k], 'hardware_before':snapshot(), 'results':[],
              'conditions':'Same-process alternating L2-cold graph single launches; fixed K-splitting warps=4; candidate groups=1 for M<=16; wide tile unchanged groups=4/cta_tiles=4.'}
    row_zero = None
    for m in (1,4,16,17,64):
        graphs={}
        for arm,program in programs.items():
            out[arm].fill_(float('nan'))
            program.launch(x,w,s,out[arm],scratch,scalars=(m,64))
            torch.cuda.synchronize()
            expected=out[arm][:m].clone()
            assert torch.isfinite(expected).all()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(x,w,s,out[arm],scratch,scalars=(m,64))
            out[arm].fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(expected.view(torch.uint8),out[arm][:m].view(torch.uint8))
            assert torch.isnan(out[arm][m:]).all()
            graphs[arm]=graph
        assert torch.equal(out['base'][:m].view(torch.uint8),out['candidate'][:m].view(torch.uint8))
        if row_zero is None:
            row_zero=out['base'][0].clone()
        assert torch.equal(row_zero.view(torch.uint8),out['candidate'][0].view(torch.uint8))
        samples={a:[] for a in programs}
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in range(args.samples):
            for arm in ('base','candidate') if i%2==0 else ('candidate','base'):
                flush(); start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                samples[arm].append(start.elapsed_time(end)*1000)
        medians={a:statistics.median(times) for a,times in samples.items()}
        item={'m':m,'median_us':medians,'speedup':medians['base']/medians['candidate'],
              'paired_speedup':statistics.median(a/b for a,b in zip(samples['base'],samples['candidate'])),
              'samples_us':samples,'byte_exact':True,'row_invariant':True,'tail_guard':True}
        result['results'].append(item)
        print(json.dumps({k:v for k,v in item.items() if k!='samples_us'}),flush=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    result['fused_oracles']=[]
    for m in (1,4,16,17,64):
        test_flash_standalone_block_fp8_matches_fused(32,'decode',m,64)
        result['fused_oracles'].append(m)
    result['hardware_after']=snapshot()
    result['complete']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('HQA round-1 complete; five old/new byte+row+tail checks and five fused oracles passed',flush=True)


if __name__=='__main__':
    main()
