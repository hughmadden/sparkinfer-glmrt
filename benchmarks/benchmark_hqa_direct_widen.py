#!/usr/bin/env python3
"""Paired HQA W8A16 direct-widen screen, preserving every production tile."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def direct_program(n, k, decode):
    import torch
    from b12x.integration.cuteafd.glmf_split import _Fp8Linear
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_gemv_nc import (
        _e4m3x4_scaled_bf16x2x2_fast, _e4m3x8_scaled_bf16_fast)
    launch = _Fp8Linear(n, k, 64 if decode else 4096, decode)
    for op in (launch.projection.gemv, launch.projection.gemv_wide):
        if op is not None:
            op.direct_widen = True
            op.widen = _e4m3x4_scaled_bf16x2x2_fast
    launch.projection.w8a16.direct_widen = True
    launch.projection.w8a16.widen = _e4m3x8_scaled_bf16_fast
    return compile_program(launch, name='hqa_direct_widen', operands=(
        Operand('x',torch.bfloat16,f'[rows,{k}]'),
        Operand('weight',torch.uint8,f'[{n},{k}]'),
        Operand('scales',torch.float32,f'[{n//128},{k//128}]'),
        Operand('out',torch.bfloat16,f'[rows,{n}]','out'),
        Operand('scratch',torch.uint8,'[scratch]','scratch')),
        scalars=(Scalar('rows'),Scalar('fp8_rows')),
        key=(n,k,decode,launch.projection.key()),
        scratch={'scratch':lambda rows:max(256,launch.projection.prefill_scratch_bytes(rows))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=25)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or args.samples<1:
        raise ValueError('new output and positive sample count required')
    if args.preflight:
        import ast
        ast.parse(Path(__file__).read_text())
        print('HQA direct-widen preflight passed')
        return
    import torch
    from benchmarks.benchmark_hqa_projection_ceiling import SHAPES,snapshot
    from benchmarks.common import make_l2_flush_fn
    from b12x.integration.cuteafd.glmf_split import compile_glmf_fp8_linear_aot
    torch.manual_seed(836)
    flush=make_l2_flush_fn(True)
    result={'hardware_before':snapshot(),'results':[],
            'conditions':'Same-process paired alternating L2-cold graphs. Production tiles/memory/MMA/K-fold unchanged; only E4M3->FP32 widening changes from FP16 detour to direct BF16 bits. FP8-A16 only; A8 unchanged.'}
    for shape,(n,k) in SHAPES.items():
        x=(torch.randn(4096,k,device='cuda')*.1).bfloat16()
        w=(torch.randn(n,k,device='cuda')*.1).to(torch.float8_e4m3fn)
        s=torch.rand(n//128,k//128,device='cuda')*.3+.05
        out={a:torch.empty(4096,n,device='cuda',dtype=torch.bfloat16) for a in ('base','candidate')}
        programs={decode:{'base':compile_glmf_fp8_linear_aot(input_features=k,output_features=n,
                            max_rows=64 if decode else 4096,decode=decode),
                          'candidate':direct_program(n,k,decode)} for decode in (True,False)}
        scratch=torch.empty(programs[False]['base'].scratch_bytes(4096)['scratch'],device='cuda',dtype=torch.uint8)
        first_row=None
        for m in (1,4,16,64,2752,4096):
            decode=m<=64
            flag=64 if decode else 0
            graphs={}
            for arm,program in programs[decode].items():
                out[arm].fill_(float('nan'))
                program.launch(x,w,s,out[arm],scratch,scalars=(m,flag))
                torch.cuda.synchronize()
                expected=out[arm][:m].clone()
                assert torch.isfinite(expected).all()
                assert torch.isnan(out[arm][m:]).all()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    program.launch(x,w,s,out[arm],scratch,scalars=(m,flag))
                out[arm].fill_(float('nan'))
                graph.replay(); torch.cuda.synchronize()
                assert torch.equal(expected.view(torch.uint8),out[arm][:m].view(torch.uint8))
                assert torch.isnan(out[arm][m:]).all()
                graphs[arm]=graph
            assert torch.equal(out['base'][:m].view(torch.uint8),out['candidate'][:m].view(torch.uint8)),(shape,m)
            if m==1:
                first_row=out['candidate'][0].clone()
            if decode:
                assert torch.equal(first_row.view(torch.uint8),out['candidate'][0].view(torch.uint8))
            samples={a:[] for a in out}
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for i in range(args.samples):
                for arm in ('base','candidate') if i%2==0 else ('candidate','base'):
                    flush(); start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                    samples[arm].append(start.elapsed_time(end)*1000)
            medians={a:statistics.median(v) for a,v in samples.items()}
            entry={'shape':shape,'m':m,'median_us':medians,
                   'paired_speedup':statistics.median(a/b for a,b in zip(samples['base'],samples['candidate'])),
                   'byte_exact':True,'tail_guard':True,'samples_us':samples}
            result['results'].append(entry)
            print(json.dumps({k:v for k,v in entry.items() if k!='samples_us'}),flush=True)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
        del programs,graphs,x,w,s,out,scratch,expected
        torch.cuda.empty_cache()
    result['hardware_after']=snapshot()
    result['complete']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('HQA direct-widen complete: all 24 byte/graph/tail checks passed',flush=True)


if __name__=='__main__':
    main()
