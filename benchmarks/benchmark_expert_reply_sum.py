"""Byte-exact GLM Spark sum/reply duel; run inside the SM121 dev image.

A candidate MODULE:FACTORY receives the sum compiler's static geometry plus
reply_format=1|2. It returns launch(fc2, out, weights, ids, expert_map, svh,
rows, stream). Arguments are Torch buffers, live row count, and CUstream;
all compilation/allocation must happen in the factory, never in launch.
The output is packed uint8, with the existing ExpertReply row layout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import torch

from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum
from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot


def pointer(dtype, tensor, alignment=16):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=alignment)


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def cu_stream():
    # Capture switches PyTorch's current stream; never retain the setup stream.
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def capture(launch):
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    return graph


def elapsed(graph, iterations):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("requires a Blackwell GPU inside its architecture's dev image")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise RuntimeError("the Spark reply duel must run on SM121")
    torch.manual_seed(args.seed)
    width, topk, experts, capacity = 6144, 8, 256, max(args.rows)
    options = dict(m=capacity, topk=topk, hidden_size=width,
                   element_dtype="fp16", full_rotation=True,
                   full_rotation_output_dtype="bf16", num_experts=experts,
                   route_num_experts=experts, route_ids_dtype=torch.int32,
                   use_expert_map=True, broadcast_svh=False)
    baseline = compile_w4a16_topk_sum(**options)
    codec = compile_expert_reply_aot(quantize=True)
    factory = None
    if args.candidate:
        module, sep, symbol = args.candidate.partition(":")
        if not sep:
            raise ValueError("candidate must be MODULE:FACTORY")
        factory = getattr(importlib.import_module(module), symbol)
    candidates = {fmt: factory(**options, reply_format=fmt) for fmt in args.formats} if factory else {}
    # Random top-k without repeats across all 256 real experts, not hot routes.
    ids = torch.rand(capacity, experts, device="cuda").argsort(-1)[:, :topk].int().contiguous()
    expert_map = torch.randperm(experts, device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(capacity, topk, device="cuda"), -1)
    fc2 = (torch.randn(capacity * topk, width, device="cuda") * .25).half()
    svh = (torch.randn(experts, width, device="cuda") * .1 + 1).half()
    # Preserve all-zero, tiny, large and cancellation-sensitive routes in the oracle.
    fc2[:topk].zero_()
    fc2[topk:2 * topk].mul_(2 ** -12)
    fc2[2 * topk:3 * topk].mul_(128)
    bf16 = torch.empty(capacity, width, device="cuda", dtype=torch.bfloat16)

    def sum_only(rows):
        baseline.compiled(pointer(cutlass.Float16, fc2), pointer(cutlass.BFloat16, bf16),
                          pointer(cutlass.Float32, weights, 4), pointer(cutlass.Int32, ids, 4),
                          pointer(cutlass.Int32, expert_map, 4), pointer(cutlass.Float16, svh),
                          cutlass.Int32(experts), cutlass.Int32(experts), rows, cu_stream())

    records = []
    for rows in args.rows:
        sum_only(rows)
        torch.cuda.synchronize()
        expected_bf16_hash = digest(bf16[:rows].view(torch.uint8))
        sum_graph = capture(lambda: sum_only(rows))
        bf16[:rows].zero_()
        sum_graph.replay()
        torch.cuda.synchronize()
        assert digest(bf16[:rows].view(torch.uint8)) == expected_bf16_hash, "sum capture did not replay"
        for _ in range(args.warmup):
            sum_graph.replay()
        sum_ms = elapsed(sum_graph, args.iterations)
        if rows >= 2663 and sum_ms < 0.3:
            raise RuntimeError(f"invalid sum capture timing: {sum_ms:.6f} ms at {rows}x{width}; expected >=0.3 ms")
        for fmt in args.formats:
            stride = width + width // 32 if fmt == 1 else width // 2 + width // 16
            expected = torch.full((rows, stride), 0xA5, device="cuda", dtype=torch.uint8)
            guarded = torch.full((rows * stride + 512,), 0x5A, device="cuda", dtype=torch.uint8)
            actual = guarded[256:-256].view(rows, stride)

            def unfused():
                sum_only(rows)
                codec.launch(bf16, expected, None, scalars=(rows, width, fmt), stream=cu_stream())

            unfused()
            torch.cuda.synchronize()
            oracle_hash = digest(expected)
            graph_base = capture(unfused)
            expected.fill_(0xD3)
            graph_base.replay()
            torch.cuda.synchronize()
            assert digest(expected) == oracle_hash, "baseline capture did not replay"
            graphs = {"baseline": graph_base}
            exact = None
            if factory:
                launch = candidates[fmt]

                def fused():
                    launch(fc2, actual, weights, ids, expert_map, svh, rows, cu_stream())

                fused()
                torch.cuda.synchronize()
                if not torch.equal(actual, expected):
                    changed = (actual != expected).nonzero()
                    raise AssertionError(f"rows={rows} format={fmt}: {len(changed)} byte mismatches, first={changed[:8].tolist()}")
                graph_candidate = capture(fused)
                # Replay must write new data, not retain an oracle-sized first output.
                actual.fill_(0xD3)
                graph_candidate.replay()
                torch.cuda.synchronize()
                assert torch.equal(actual, expected), "captured candidate differs"
                assert bool((guarded[:256] == 0x5A).all() and (guarded[-256:] == 0x5A).all()), "candidate overwrote canary"
                fc2[3 * topk:4 * topk].neg_()
                graph_base.replay()
                graph_candidate.replay()
                torch.cuda.synchronize()
                assert torch.equal(actual, expected), "candidate ignores changed route values"
                fc2[3 * topk:4 * topk].neg_()
                graph_base.replay()
                graph_candidate.replay()
                torch.cuda.synchronize()
                assert digest(expected) == oracle_hash, "oracle restore differs"
                exact = digest(actual) == oracle_hash
                assert exact
                graphs["candidate"] = graph_candidate
            for graph in graphs.values():
                for _ in range(args.warmup):
                    graph.replay()
            samples = {name: [] for name in graphs}
            for round_index in range(args.rounds):
                order = list(graphs.items())
                if round_index % 2:
                    order.reverse()
                for name, graph in order:
                    samples[name].append(elapsed(graph, args.iterations))
            sum_only(rows)
            torch.cuda.synchronize()
            assert digest(bf16[:rows].view(torch.uint8)) == expected_bf16_hash, "BF16 baseline changed"
            record = dict(rows=rows, width=width, topk=topk, experts=experts,
                          reply_format=fmt, exact=exact, oracle_sha256=oracle_hash,
                          bf16_sha256=expected_bf16_hash, bf16_sum_ms=sum_ms,
                          samples_ms=samples,
                          medians_ms={k: statistics.median(v) for k, v in samples.items()})
            records.append(record)
            print(json.dumps(record), flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    report = dict(schema="expert-reply-sum-duel/1", revision=revision, dirty=dirty,
                  candidate=args.candidate, gpu=props.name, sms=props.multi_processor_count,
                  torch=torch.__version__, cuda=torch.version.cuda, seed=args.seed,
                  iterations=args.iterations, rounds=args.rounds, records=records)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="MODULE:FACTORY; omit to measure the unchanged oracle")
    parser.add_argument("--rows", nargs="+", type=int, default=[2752, 2663])
    parser.add_argument("--formats", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--seed", type=int, default=1166)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows + [args.warmup, args.iterations, args.rounds]) < 1 or min(args.rows) < 4:
        parser.error("positive counts and at least four rows required")
    run(args)


if __name__ == "__main__":
    main()
