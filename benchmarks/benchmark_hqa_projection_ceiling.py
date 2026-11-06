#!/usr/bin/env python3
"""L2-cold HQA baseline ceiling; native cuBLAS and standalone Flash block FP8.

Run through cuteafd's hwlock and coordinator development container. This is a
microbenchmark, not a hardware/container driver. Results live outside the repo.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPES = {
    "qb-full": (16384, 1536), "qb-tp2": (8192, 1536),
    "o-full": (4096, 16384), "o-tp2": (4096, 8192),
}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-lib", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", default="1,4,16,64,2752,4096")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--bandwidth-tb-s", type=float, default=1.792)
    p.add_argument("--bf16-tflops", type=float, default=503.8)
    p.add_argument("--fp8-tflops", type=float, default=1007.6)
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,power.limit,clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main():
    args = args_parser()
    rows_list = [int(x) for x in args.rows.split(",")]
    if not args.native_lib.is_file() or args.samples < 1 or any(m < 1 for m in rows_list):
        raise ValueError("native library, positive rows and samples required")
    if args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        raise ValueError("output must be a new file outside the fork")
    if args.preflight:
        import ast
        ast.parse(Path(__file__).read_text())
        print("HQA ceiling preflight passed: native library and benchmark syntax")
        return

    import torch
    from benchmarks.common import make_l2_flush_fn
    from b12x.integration.cuteafd.glmf_split import compile_glmf_fp8_linear_aot
    from tests.integration.test_cuteafd_glmf_native_stages import test_flash_standalone_block_fp8_matches_fused

    prop = torch.cuda.get_device_properties(0)
    if "RTX PRO 6000" not in prop.name or (prop.major, prop.minor) != (12, 0):
        raise RuntimeError(f"ceiling requires PRO 6000 SM120, got {prop.name}")
    torch.manual_seed(836)
    lib = C.CDLL(str(args.native_lib))
    native = lib.cuteafd_cuda_linear_bf16_cublas_async
    native.argtypes = [C.c_void_p] * 4 + [C.c_size_t] * 3 + [C.c_void_p]
    native.restype = C.c_int32
    flush = make_l2_flush_fn(True)
    result = {
        "device": prop.name, "sms": prop.multi_processor_count,
        "native_lib": str(args.native_lib),
        "native_lib_sha256": hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "hardware_before": snapshot(),
        "roofline": {"bandwidth_tb_s": args.bandwidth_tb_s,
                     "bf16_dense_tflops": args.bf16_tflops,
                     "fp8_dense_tflops": args.fp8_tflops},
        "conditions": "One warm launch; per-shape eager+graph warmup; alternating L2-cold graph samples. Decode: compulsory weight/input/output bytes divided by GDDR7 peak. Prefill: useful 2MNK divided by dense precision peak (no sparsity); W8A8 includes activation quantization. This is theoretical peak, not a measured sustainable ceiling.",
        "results": [], "fused_oracles": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for label, (n, k) in SHAPES.items():
        x = (torch.randn(max(rows_list), k, device="cuda") * .1).bfloat16()
        w = (torch.randn(n, k, device="cuda") * .02).bfloat16()
        wf = (torch.randn(n, k, device="cuda") * .1).to(torch.float8_e4m3fn)
        scales = torch.rand(n // 128, k // 128, device="cuda") * .3 + .05
        outputs = {name: torch.empty(max(rows_list), n, device="cuda", dtype=torch.bfloat16)
                   for name in ("bf16", "fp8-a16", "fp8-a8")}
        programs = {decode: compile_glmf_fp8_linear_aot(input_features=k,
            output_features=n, max_rows=64 if decode else max(rows_list), decode=decode)
            for decode in (True, False)}
        scratches = {decode: torch.empty(max(256, program.scratch_bytes(64 if decode else max(rows_list))["scratch"]),
            device="cuda", dtype=torch.uint8) for decode, program in programs.items()}
        for m in rows_list:
            decode = m <= 64
            program, scratch = programs[decode], scratches[decode]
            def bf16():
                status = native(x.data_ptr(), w.data_ptr(), None, outputs["bf16"].data_ptr(),
                                m, k, n, torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"native cuBLAS status {status}")
            calls = {"bf16": bf16}
            for arm, flag in (("fp8-a16", 64 if decode else 0), ("fp8-a8", 1)):
                if decode and arm == "fp8-a8":
                    continue
                calls[arm] = lambda arm=arm, flag=flag: program.launch(
                    x, wf, scales, outputs[arm], scratch, scalars=(m, flag))
            graphs = {}
            for arm, call in calls.items():
                call()
                torch.cuda.synchronize()
                expected = outputs[arm][:m].clone()
                if not torch.isfinite(expected).all() or not torch.count_nonzero(expected):
                    raise AssertionError(f"invalid output {label}/{m}/{arm}")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(expected.view(torch.uint8), outputs[arm][:m].view(torch.uint8))
                graphs[arm] = graph
            samples = {arm: [] for arm in calls}
            order = list(calls)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for sample in range(args.samples):
                for arm in order if sample % 2 == 0 else order[::-1]:
                    flush()
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(start.elapsed_time(end) * 1000)
            for arm in calls:
                us = statistics.median(samples[arm])
                byte_count = (2 if arm == "bf16" else 1) * n*k + 2*m*(n+k)
                if arm != "bf16":
                    byte_count += 4*(n//128)*(k//128)
                peak = args.fp8_tflops if arm == "fp8-a8" else args.bf16_tflops
                ideal_us = byte_count / (args.bandwidth_tb_s * 1e6) if decode else 2*m*n*k/(peak*1e6)
                entry = dict(shape=label, n=n, k=k, m=m, arm=arm, median_us=us,
                             ideal_us=ideal_us, roofline_fraction=ideal_us/us,
                             samples_us=samples[arm], graph_byte_exact=True)
                result["results"].append(entry)
                print(json.dumps({k:v for k,v in entry.items() if k != "samples_us"}), flush=True)
            save()
        del programs, scratches, graphs, calls, x, w, wf, scales, outputs
        torch.cuda.empty_cache()

    # The authoritative fused-vs-standalone oracle includes all requested M,
    # both head counts and both prefill arithmetic modes; timing above excludes it.
    for heads in (32, 64):
        for m in rows_list:
            mode = "decode" if m <= 64 else "prefill"
            for flag in ((64,) if mode == "decode" else (0, 1)):
                test_flash_standalone_block_fp8_matches_fused(heads, mode, m, flag)
                result["fused_oracles"].append(dict(heads=heads, m=m, mode=mode, flag=flag, passed=True))
                print(f"fused byte oracle passed heads={heads} M={m} mode={mode} flag={flag}", flush=True)
                save()
    result["hardware_after"] = snapshot()
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
