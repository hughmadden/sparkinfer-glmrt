#!/usr/bin/env python3
"""L2-cold decode timing of the register-loaded weight-streaming kernels.

One process times one source tree; hint on/off is two trees (the fork before and
after the ``weight_stream_policy`` change) run alternately by the caller, so the
kernels compare exactly as they ship. Each case compiles its exported program,
fills fixed-seed inputs, and replays a CUDA graph of one launch with a 2x-L2
eviction before every sample; it prints the median and the sha256 of the output
bytes (the trees must agree bit for bit).

Cases (``--cases``):
  trellis   Kimi K6 dense trellis linear (mcg), sharded side 1, e.g. 7168x6144
  moe       routed-expert decode route, BF16 input (FP8: glmf; MXFP4: mimop)
  fp8gemv   dense block-FP8 MmaFp8Gemv (16-row tile, multi-tile above 16)
  skinny    BF16 SkinnyBf16Gemv (bf16 projection's small-row route)
  scaled    NativeTileLinear decode (fp8_row / mxfp8 / int4_g32)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(-1).view(__import__("torch").uint8)
                          .cpu().numpy().tobytes()).hexdigest()[:16]


def _time(run, samples, flush, outputs):
    import torch
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for out in outputs:
        out.fill_(float("nan")) if out.is_floating_point() else out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    digest = "".join(_digest(out) for out in outputs)[:32]
    times = []
    for _ in range(samples):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times), min(times), digest


COPIES_L2 = 0  # >0: enough distinct weight copies per sample to stream this many bytes (set in main)


def _copies(weight_bytes):
    """Distinct weight copies one sample walks: small operands are timed over several cold
    copies (each still L2-cold) so a sample is long enough to resolve."""
    if not COPIES_L2:
        return 1
    return max(1, min(16, -(-COPIES_L2 // int(weight_bytes))))


def _scaled_payload(encoding, size, gen):
    import torch
    if encoding == "int4_g32":
        return torch.randint(256, (size,), device="cuda", dtype=torch.uint8, generator=gen)
    return (torch.randn(size, device="cuda", generator=gen) * 50).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def trellis_cases(spec, rows_list, gen):
    import torch
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    k, n = (int(v) for v in spec.split("x"))
    capacity = max(rows_list)
    program = compile_trellis_linear_aot(input_features=k, output_features=n, bits=6, codebook="mcg",
                                         sharded_side=1, rotation_count=1, max_rows=capacity)
    splits = (k + 1023) // 1024
    x = torch.randn(capacity, k, device="cuda", generator=gen).half()
    inputs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
    payloads = [torch.randint(256, (k * n * 6 // 8,), device="cuda", dtype=torch.uint8, generator=gen)
                for _ in range(_copies(k * n * 6 // 8))]
    scales = torch.ones(k // 128, device="cuda", dtype=torch.float16)
    partial = torch.empty(capacity * splits * n, device="cuda", dtype=torch.float32)
    output = torch.empty(capacity * n, device="cuda", dtype=torch.float32)
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def run(rows):
        for payload in payloads:
            program.launch(inputs, payload, scales, partial, output, scalars=(rows, 0, sms * 4))
    for rows in rows_list:
        yield (f"trellis_k6_{spec}", rows, n * k * 6 // 8, len(payloads),
               lambda rows=rows: run(rows), [output[:rows * n]])


def moe_cases(name, rows_list, gen):
    import torch
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    tp = 4
    g = GEOMETRIES[name].with_tp(tp)
    h, i, e, k = g.hidden, g.slice, g.experts, g.top_k
    if g.weights == "fp8":
        def fp8(shape):
            return (torch.randn(shape, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
        def sc(shape):
            return torch.rand(shape, device="cuda", generator=gen) * 2e-4 + 1e-4
        w = (fp8((e, i, h)), sc((e, i // 128, h // 128)), fp8((e, i, h)), sc((e, i // 128, h // 128)),
             fp8((e, h, i)), sc((e, h // 128, i // 128)))
        per_expert = 3 * i * h
    elif g.weights == "mxfp4":
        from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes
        def packed(shape):
            return torch.randint(256, shape, device="cuda", dtype=torch.uint8, generator=gen)
        def ue8m0(shape):
            return torch.randint(118, 130, shape, device="cuda", dtype=torch.uint8, generator=gen)
        w = (packed((e, i, h // 2)), ue8m0((e, i, h // 32)), packed((e, i, h // 2)), ue8m0((e, i, h // 32)),
             packed((e, h, i // 2)), ue8m0((e, h, mxfp4_scale_row_bytes(i))))
        per_expert = 3 * i * h // 2
    else:
        raise ValueError(f"unsupported MoE weights {g.weights}")
    for rows in rows_list:
        program = compile_fp8_moe_aot(g, route="decode", max_rows=rows, wire=False)
        x = torch.randn(rows, h, device="cuda", generator=gen).bfloat16()
        ids = torch.rand(rows, e, device="cuda", generator=gen).topk(k, -1).indices.int().contiguous()
        mix = torch.rand(rows, k, device="cuda", generator=gen).contiguous()
        out = torch.empty(rows, h, dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(fp8_moe_scratch_bytes(g, "decode", rows, False), dtype=torch.uint8, device="cuda")
        unique = int(ids.unique().numel())
        yield (f"moe_{name}_tp{tp}", rows, unique * per_expert, 1,
               lambda program=program, x=x, ids=ids, mix=mix, out=out, scratch=scratch, rows=rows:
               program.launch(x, ids, mix, *w, out, scratch, scalars=(rows,)),
               [out])


def fp8gemv_cases(spec, rows_list, gen):
    import cutlass
    import torch
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import MmaFp8Gemv, gemv_warps
    n, k = (int(v) for v in spec.split("x"))
    x = torch.randn(max(rows_list), k, device="cuda", generator=gen).bfloat16()
    ws = [(torch.randn(n, k, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
          for _ in range(_copies(n * k))]
    s = torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 2e-4 + 1e-4
    out = torch.empty(max(rows_list), n, device="cuda", dtype=torch.bfloat16)
    programs = {}
    for cap in sorted({16 if r <= 16 else 64 for r in rows_list}):
        op = MmaFp8Gemv(n, k, max_rows=cap, warps=gemv_warps(k), groups=4, out_dtype=cutlass.BFloat16)
        programs[cap] = compile_program(op, name="fp8_gemv_l2_bench", operands=(
            Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.float8_e4m3fn, f"[{n},{k}]"),
            Operand("s", torch.float32, f"[{n // 128},{k // 128}]", align=4),
            Operand("out", torch.bfloat16, f"[rows,{n}]", "out")), scalars=(Scalar("rows"),), key=op.key())
    for rows in rows_list:
        program = programs[16 if rows <= 16 else 64]
        def run(program=program, rows=rows):
            for w in ws:
                program.launch(x, w, s, out, scalars=(rows,))
        yield (f"fp8gemv_{spec}", rows, n * k, len(ws), run, [out[:rows]])


def skinny_cases(spec, rows_list, gen):
    import cutlass
    import torch
    from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, skinny_config, skinny_max_rows
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    n, k = (int(v) for v in spec.split("x"))
    op = SkinnyBf16Gemv(n, k, **skinny_config(n, k, cutlass.BFloat16))
    program = compile_program(op, name="skinny_l2_bench", operands=(
        Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.bfloat16, f"[{n},{k}]"),
        Operand("out", torch.bfloat16, f"[rows,{n}]", "out")), scalars=(Scalar("rows"),), key=(n, k, "skinny"))
    x = torch.randn(max(rows_list), k, device="cuda", generator=gen).bfloat16()
    ws = [(torch.randn(n, k, device="cuda", generator=gen) * 0.03).bfloat16() for _ in range(_copies(n * k * 2))]
    out = torch.empty(max(rows_list), n, device="cuda", dtype=torch.bfloat16)
    limit = skinny_max_rows(n, k)
    for rows in rows_list:
        if rows > limit:
            continue
        def run(rows=rows):
            for w in ws:
                program.launch(x, w, out, scalars=(rows,))
        yield (f"skinny_{spec}", rows, n * k * 2, len(ws), run, [out[:rows]])


def scaled_cases(spec, rows_list, gen):
    import torch
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
    encoding, shape = spec.split(":")
    n, k = (int(v) for v in shape.split("x"))
    capacity = max(rows_list)
    program = compile_scaled_linear_aot(input_features=k, output_features=n, encoding=encoding,
                                        sharded_side=1, max_rows=capacity)
    payload_size = k * n // (2 if encoding == "int4_g32" else 1)
    copies = _copies(payload_size)
    payloads = []
    for _ in range(copies):
        payloads.append(_scaled_payload(encoding, payload_size, gen))
    if encoding == "fp8_row":
        payload = (torch.randn(payload_size, device="cuda", generator=gen) * 50).clamp(-448, 448) \
            .to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.rand(n, device="cuda", generator=gen) * 1e-3 + 1e-4
    elif encoding == "mxfp8":
        payload = (torch.randn(payload_size, device="cuda", generator=gen) * 50).clamp(-448, 448) \
            .to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(118, 128, (n * k // 32,), device="cuda", dtype=torch.uint8, generator=gen)
    else:
        payload = torch.randint(256, (payload_size,), device="cuda", dtype=torch.uint8, generator=gen)
        scales = (torch.rand(n * k // 32, device="cuda", generator=gen) * 1e-2 + 1e-3).half()
    x = torch.randn(capacity, k, device="cuda", generator=gen).bfloat16()
    partial = torch.empty(16, device="cuda", dtype=torch.float32)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.float32)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for rows in rows_list:
        def run(rows=rows):
            for p in [payload] + payloads[1:]:
                program.launch(x, p, scales, partial, output, scalars=(rows, sms * 4))
        yield (f"scaled_{spec}", rows, payload_size, copies, run, [output[:rows]])


CASES = {"trellis": trellis_cases, "moe": moe_cases, "fp8gemv": fp8gemv_cases,
         "skinny": skinny_cases, "scaled": scaled_cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True,
                        help="semicolon list of CASE=SPEC[,SPEC], e.g. 'trellis=7168x6144;moe=glmf,mimop'")
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stream-l2", type=float, default=0.0,
                        help="time each sample over distinct weight copies totalling this many L2 sizes "
                             "(at most 16 copies), so short kernels resolve; 0: one copy")
    args = parser.parse_args()
    import torch
    rows_list = [int(r) for r in args.rows.split(",")]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(2 * l2, device="cuda", dtype=torch.uint8)
    global COPIES_L2
    COPIES_L2 = int(args.stream_l2 * l2)
    records = []
    for item in args.cases.split(";"):
        case, _, specs = item.partition("=")
        for spec in specs.split(","):
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            for name, rows, weight_bytes, copies, run, outputs in CASES[case](spec, rows_list, gen):
                median, best, digest = _time(run, args.samples, flush, outputs)
                median, best = median / copies, best / copies
                record = dict(tag=args.tag, case=name, rows=rows, copies=copies, median_us=round(median, 3),
                              min_us=round(best, 3), weight_gbps=round(weight_bytes / median / 1e3, 1),
                              digest=digest)
                records.append(record)
                print(json.dumps(record), flush=True)
            torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), l2_bytes=l2,
                                               samples=args.samples, records=records), indent=1) + "\n")


if __name__ == "__main__":
    main()
