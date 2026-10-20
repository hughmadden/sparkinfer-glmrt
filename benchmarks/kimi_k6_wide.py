"""Diagnostic existing tiled MCG and bounded chunk-dequant K6 wide-row routes.

Not an engine route or serving policy. Tiled MCG uses its prepared public API
with identity rotation callbacks (copies included); MUL1 is explicitly
unsupported there. Chunk dequant and BF16 GEMM run exported C wrappers, with
at most 256 MiB of dequantized weight scratch and no persistent dense copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.kimi_k6_dense import exported


def checked_run(lib, *args):
    result = lib.run(*args)
    if result:
        raise RuntimeError(f"exported wrapper returned {result}")


def tiled(x, raw, k, n, output, scratch):
    from b12x.gemm import trellis_linear as tl
    from b12x.preparation import PreparationSession, PreparedCall
    import torch
    signs_k = torch.ones(k, device="cuda", dtype=torch.float16)
    signs_n = torch.ones(n, device="cuda", dtype=torch.float16)
    weight = tl.prepare_weight(raw.view(torch.int16).reshape(k//16,n//16,96), signs_k,
                               signs_n, codebook="mcg", params_dtype=torch.float16)
    gemm_output = torch.empty_like(output)
    rotated = torch.empty_like(x)
    kwargs = dict(output=output,gemm_output=gemm_output,rotated_f16=rotated)
    query = tl.query_from_weight(weight,max_rows=x.shape[0],input_dtype=x.dtype,
        output_mode="provided",c_tmp_mode="provided",gemm_output_provided=True,rotated_f16_provided=True,
        hadamard_128=lambda source,dest,*_: dest.copy_(source))
    plan = tl.plan(query,weight=weight,c_tmp=scratch,hadamard_128=lambda source,dest,*_: dest.copy_(source))
    request=plan.request(name="tiled",prepare_call=lambda state: PreparedCall(run=lambda: state.run(x,**kwargs)))
    session=PreparationSession(device=x.device,autotune=False,compile_workers=1)
    session.__enter__()
    try:
        session.prepare((request,))
        return lambda: tl.run(x,plan=plan,**kwargs), session, (weight,gemm_output,rotated)
    except BaseException:
        session.__exit__(*sys.exc_info())
        raise


def native_raw(raw, k, n, side):
    # Shared side 1 [N128,K16,N16in128]; side 2 [K128,N128,K16in128,N16in128].
    if side == 1:
        return raw.reshape(n//128,k//16,8,192).permute(1,0,2,3).contiguous().reshape(-1)
    return raw.reshape(k//128,n//128,8,8,192).permute(0,2,1,3,4).contiguous().reshape(-1)


def correctness(directory):
    import torch
    import numpy as np
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.trellis_dequant import compile_trellis_dequant_aot
    from tests._reference.native_trellis import Reference
    records=[]
    k,n=256,256
    for side in (1,2):
        raw=torch.randint(256,(k*n*6//8,),device="cuda",dtype=torch.uint8)
        for codebook in ("mcg","mul1"):
            ref=Reference(codebook)
            data=raw.cpu().numpy()
            decoded=np.concatenate([ref.decode_unit(chunk.tobytes(),6,"gate" if side==1 else "down",k if side==1 else n)
                for chunk in data.reshape((n if side==1 else k)//128,-1)],axis=1 if side==1 else 0)
            expected=torch.tensor(decoded.T.copy(),device="cuda").bfloat16()
            scales=torch.ones(k//128,device="cuda",dtype=torch.float16)
            out=torch.empty_like(expected)
            with exportable_compilation():
                program=compile_trellis_dequant_aot(input_features=k,output_features=n,chunk_columns=128,codebook=codebook,sharded_side=side)
            lib=exported(program,directory,f"check_s{side}_{codebook}")
            for first in (0,128):
                checked_run(lib,raw.data_ptr(),scales.data_ptr(),out[first:].data_ptr(),first,torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            if not torch.equal(out,expected):
                raise RuntimeError((side,codebook,"dequant mismatch"))
            record=dict(side=side,codebook=codebook,dequant_exact=True)
            if codebook=="mcg":
                x=torch.randn(64,k,device="cuda",dtype=torch.float16)*.1
                result=torch.empty(64,n,device="cuda",dtype=torch.float16)
                scratch=torch.empty(1<<22,device="cuda",dtype=torch.float32)
                try:
                    run,session,owners=tiled(x,native_raw(raw,k,n,side),k,n,result,scratch)
                    run();torch.cuda.synchronize()
                    reference=x.float()@torch.tensor(decoded,device="cuda")
                    relative=(result.float()-reference).norm()/reference.norm()
                    if not torch.isfinite(relative).item() or relative >= .01:
                        raise RuntimeError((side,"tiled error",relative.item()))
                    record["tiled_relative_rmse"]=relative.item()
                    session.__exit__(None,None,None)
                except Exception as exc:
                    record["tiled_error"]=str(exc)
                    print(f"tiled correctness unavailable: {exc}",flush=True)
            else:
                record["tiled_error"]="unsupported MUL1 codebook in existing public preparation"
            records.append(record)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--shapes",default="7168:6144:1,6144:7168:2,7168:163840:1")
    parser.add_argument("--rows",default="64,256,1024,4096")
    parser.add_argument("--samples",type=int,default=15)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    import torch
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.trellis_dequant import compile_trellis_dequant_aot
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    torch.manual_seed(273)
    checks=correctness(args.output_dir)
    allow_tiled=all("tiled_relative_rmse" in r for r in checks if r["codebook"]=="mcg")
    report=dict(correctness=checks,records=[],tiled_includes_identity_rotation_copies=True,transforms_included=False)
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    for shape in args.shapes.split(","):
        k,n,side=map(int,shape.split(":"));directory=args.output_dir/f"k{k}_n{n}_s{side}";directory.mkdir(exist_ok=True)
        rows_list=list(map(int,args.rows.split(",")));capacity=max(rows_list)
        c=min(n,(256*1024*1024//(2*k)//128)*128)
        while n%c:c-=128
        scratch=torch.empty(c,k,device="cuda",dtype=torch.bfloat16)
        raw=torch.randint(256,(k*n*6//8,),device="cuda",dtype=torch.uint8)
        natural=native_raw(raw,k,n,side) if allow_tiled else None
        scales=torch.ones(k//128,device="cuda",dtype=torch.float16)
        x=torch.randn(capacity,k,device="cuda",dtype=torch.bfloat16)
        x16=x.half();weight=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)
        out=torch.empty(capacity,n,device="cuda",dtype=torch.bfloat16)
        chunk_out=torch.empty(capacity,c,device="cuda",dtype=torch.bfloat16)
        tiled_out=torch.empty(capacity,n,device="cuda",dtype=torch.float16) if allow_tiled else None
        tiled_scratch=torch.empty(1<<25,device="cuda",dtype=torch.float32) if allow_tiled else None
        inputs=torch.tensor([x16.data_ptr()],device="cuda",dtype=torch.int64)
        partial=torch.empty(capacity*((k+1023)//1024)*n,device="cuda",dtype=torch.float32)
        native_out=torch.empty(capacity,n,device="cuda",dtype=torch.float32)
        sms=torch.cuda.get_device_properties(0).multi_processor_count
        with exportable_compilation():
            native={book:compile_trellis_linear_aot(input_features=k,output_features=n,bits=6,
                codebook=book,sharded_side=side,rotation_count=1,max_rows=capacity) for book in ("mcg","mul1")}
            bf=compile_bf16_projection_aot(width=k,outputs=n)
            cbf=compile_bf16_projection_aot(width=k,outputs=c)
            ds={book:compile_trellis_dequant_aot(input_features=k,output_features=n,chunk_columns=c,codebook=book,sharded_side=side) for book in ("mcg","mul1")}
        bf=exported(bf,directory,"bf16");cbf=exported(cbf,directory,"chunk_bf16")
        ds={book:exported(p,directory,"dequant_"+book) for book,p in ds.items()}
        native={book:exported(p,directory,"native_"+book) for book,p in native.items()}
        for rows in rows_list:
            def baseline():
                checked_run(bf,x.data_ptr(),weight.data_ptr(),out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
            arms={"bf16":baseline}
            for book,lib in native.items():
                def core(lib=lib):
                    checked_run(lib,inputs.data_ptr(),raw.data_ptr(),scales.data_ptr(),partial.data_ptr(),
                        native_out.data_ptr(),rows,0,sms*4,torch.cuda.current_stream().cuda_stream)
                arms["native_"+book]=core
            for book,lib in ds.items():
                def chunk(lib=lib):
                    for first in range(0,n,c):
                        checked_run(lib,raw.data_ptr(),scales.data_ptr(),scratch.data_ptr(),first,torch.cuda.current_stream().cuda_stream)
                        checked_run(cbf,x.data_ptr(),scratch.data_ptr(),chunk_out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
                        out[:rows,first:first+c].copy_(chunk_out[:rows])
                arms["chunk_"+book]=chunk
            session=None
            if allow_tiled:
                try:
                    run,session,owners=tiled(x16[:rows],natural,k,n,tiled_out[:rows],tiled_scratch)
                    arms["tiled_mcg"]=run
                except Exception as exc:
                    report.setdefault("tiled_errors",[]).append(dict(k=k,n=n,rows=rows,error=str(exc)))
            graphs={}
            for name,run in arms.items():
                run();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                if name=="tiled_mcg":
                    with session.capture(),torch.cuda.graph(graph):run()
                else:
                    with torch.cuda.graph(graph):run()
                target=tiled_out[:rows] if name=="tiled_mcg" else native_out[:rows] if name.startswith("native_") else out[:rows]
                target.fill_(float("nan"))
                graph.replay();torch.cuda.synchronize()
                if not torch.isfinite(target).all().item():
                    raise RuntimeError(f"{name} graph did not overwrite poisoned output")
                graphs[name]=graph
            torch.cuda.synchronize();samples={name:[] for name in graphs}
            for repeat in range(args.samples):
                for name in (list(graphs) if repeat%2==0 else list(reversed(graphs))):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[name].replay();end.record();end.synchronize()
                    samples[name].append(start.elapsed_time(end)*1000)
            medians={name:statistics.median(v) for name,v in samples.items()}
            record=dict(k=k,n=n,side=side,rows=rows,chunk_columns=c,weight_scratch_bytes=c*k*2,
                chunk_output_scratch_bytes=capacity*c*2,
                native_partial_scratch_bytes=capacity*((k+1023)//1024)*n*4,
                tiled_scratch_bytes=(1<<27)+rows*(k+n)*2 if allow_tiled else None,
                us=medians,speedup={name:medians['bf16']/us for name,us in medians.items() if name!='bf16'})
            report["records"].append(record);print(json.dumps(record),flush=True)
            (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            if session:session.__exit__(None,None,None)
        del raw,natural,scratch,scales,x,x16,weight,out,chunk_out,tiled_out,tiled_scratch,graphs,arms
        torch.cuda.empty_cache()


if __name__=="__main__":main()
