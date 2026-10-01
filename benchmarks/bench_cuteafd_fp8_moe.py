#!/usr/bin/env python3
"""Time the cuteafd exact FP8 routed-expert programs (``fp8_moe``).

One program per (geometry, TP slice, route, capacity = rows); random E4M3
weights with FP32 128x128 block scales of the real shapes, fresh random
routes every launch (expert weights L2-cold, as in serving), L2 flushed
before every launch, the GPU held busy while the host enqueues (device
time). Prints the median per-layer time, the unique expert weight bytes
streamed per launch and the implied bandwidth, and (``--check``) the cosine
against the torch reference of the checkpoint semantics.

  python benchmarks/bench_cuteafd_fp8_moe.py --geometry mimo --tp 4 --rows 1,16,80,1024,4096 \\
      [--route auto] [--iters 20] [--check]
"""

from __future__ import annotations

import argparse
import json

import torch


def time_us(fn, iters: int, flush: torch.Tensor, prepare) -> float:
    prepare(0)
    fn()
    torch.cuda.synchronize()
    times = []
    for it in range(iters):
        prepare(it + 1)
        flush.max()
        torch.cuda._sleep(5_000_000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", default="mimo")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rows", default="1,16,80,1024,4096")
    parser.add_argument("--route", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prefill", choices=("auto", "w8a8", "w8a16"), default="auto",
                        help="large-row form of FP8 programs (auto: W8A8 for wire input, W8A16 for BF16 rows)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--input", choices=("wire", "bf16"), default="wire",
                        help="expert input rows (bf16: the coordinator packages' input)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = GEOMETRIES[args.geometry].with_tp(args.tp)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    h, i, e, k = g.hidden, g.slice, g.experts, g.top_k

    def fp8(shape):
        return (torch.randn(shape, device=dev, generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)

    def scales(shape):
        return torch.rand(shape, device=dev, generator=gen) * 2e-4 + 1e-4

    w = (fp8((e, i, h)), scales((e, i // 128, h // 128)), fp8((e, i, h)), scales((e, i // 128, h // 128)),
         fp8((e, h, i)), scales((e, h // 128, i // 128)))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    per_expert = 3 * i * h + (2 * (i // 128) * (h // 128) + (h // 128) * (i // 128)) * 4
    results = []
    for rows in [int(r) for r in args.rows.split(",")]:
        wire_in = args.input == "wire"
        program = compile_fp8_moe_aot(g, route=args.route, max_rows=rows, wire=wire_in, prefill=args.prefill)
        x = torch.randn(rows, h, device=dev, generator=gen).bfloat16()
        groups = x.float().view(rows, h // 32, 32)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-127, 127)
        q = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
        wire = torch.cat([q.view(rows, h).view(torch.uint8), (exponent + 127).to(torch.uint8)], 1).contiguous()
        x_exact = (q.float() * torch.exp2(exponent)[..., None]).view(rows, h).bfloat16()
        if not wire_in:
            wire, x_exact = x, x
        ids = torch.empty(rows, k, dtype=torch.int32, device=dev)
        weights = torch.rand(rows, k, device=dev, generator=gen).contiguous()
        out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
        scratch = torch.empty(fp8_moe_scratch_bytes(g, args.route, rows, wire_in, args.prefill), dtype=torch.uint8,
                              device=dev)
        unique = []

        def prepare(seed):
            ids.copy_(torch.rand(rows, e, device=dev, generator=gen).topk(k, -1).indices.int())
            unique.append(int(ids.unique().numel()))

        def run():
            program.launch(wire, ids, weights, *w, out, scratch, scalars=(rows,))

        us = time_us(run, args.iters, flush, prepare)
        experts = sum(unique) / len(unique)
        record = {"geometry": g.name, "tp": g.tp, "input": args.input,
                  "route": args.route + ("" if args.prefill == "auto" else "/" + args.prefill),
                  "rows": rows, "us": round(us, 1),
                  "experts": round(experts, 1), "gbps": round(experts * per_expert / us / 1e3, 1),
                  "tflops": round(rows * k * 3 * 2 * i * h / us / 1e6, 1)}
        if args.check:
            prepare(0)
            run()
            torch.cuda.synchronize()

            def dequant(wt, s):
                return (wt.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()

            ref = torch.zeros(rows, h, device=dev)
            for expert in ids.unique().tolist():
                r, slots = torch.where(ids == expert)
                gate = x_exact[r] @ dequant(w[0][expert], w[1][expert]).T
                up = x_exact[r] @ dequant(w[2][expert], w[3][expert]).T
                if g.swiglu_limit > 0:
                    gate, up = gate.clamp(max=g.swiglu_limit), up.clamp(-g.swiglu_limit, g.swiglu_limit)
                y = (torch.nn.functional.silu(gate) * up) @ dequant(w[4][expert], w[5][expert]).T
                ref.index_add_(0, r, y.float() * weights[r, slots][:, None])
            a, b = out.float(), ref.bfloat16().float()
            record["cosine"] = round(float((a * b).sum() / (a.norm() * b.norm())), 8)
            record["worst_row"] = round(float(torch.nn.functional.cosine_similarity(a, b, dim=1).min()), 6)
        results.append(record)
        print(json.dumps(record) if args.json else
              " ".join(f"{key}={value}" for key, value in record.items()), flush=True)
        del program, scratch


if __name__ == "__main__":
    main()
