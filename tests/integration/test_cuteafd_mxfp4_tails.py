"""Exact MXFP4 K32 tails against identical, zero-padded weights on SM12x."""
from dataclasses import dataclass
import json
import statistics

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_fp8_moe_aot import wire_rows


# Each production geometry gets every adjacent width from worlds 2..8.
GEOMETRIES = ((2048, 4096), (2048, 6144), (2304, 5120), (3072, 7168))
WIDTH_CASES = [(intermediate, hidden, width)
               for intermediate, hidden in GEOMETRIES
               for width in sorted({32 * (intermediate // 32 // world + extra)
                                    for world in range(2, 9)
                                    for extra in (0, int(intermediate // 32 % world != 0))})]
WIDTH_CASES += [(2048, 6144, width) for width in (32, 64, 96)]


@pytest.mark.parametrize("intermediate,hidden,width", WIDTH_CASES)
def test_exact_decode_retains_padded_warp_partition(intermediate, hidden, width, monkeypatch):
    from b12x.integration.cuteafd import fp8_moe
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, _Route

    # The GB10 four-warp exceptions have a separate real-dispatch numerical gate.
    monkeypatch.setattr(fp8_moe, "_sm121", lambda: False)

    @dataclass(frozen=True)
    class Exact(Fp8MoeGeometry):
        width: int = 32
        @property
        def slice(self):
            return self.width

    exact = _Route(Exact("tail", hidden, 16, 8, intermediate, weights="mxfp4", width=width), "decode", 17, True)
    padded = _Route(Exact("tail", hidden, 16, 8, intermediate, weights="mxfp4",
                          width=(width + 127) // 128 * 128), "decode", 17, True)
    assert exact.down.warps == padded.down.warps
    assert exact.down.k_per_warp == padded.down.k_per_warp
    assert exact.down.blocks == padded.down.blocks
    assert exact.down.row_bytes == width // 2


TAIL_CASES = [(intermediate, hidden, width, route, a8, 4096)
              for intermediate, hidden, width in WIDTH_CASES
              for route, a8 in (("decode", False), ("stream", False), ("stream", True))]
TAIL_CASES += [(2048, 6144, width, "decode", False, capacity)
               for width in (672, 704) for capacity in (1, 16, 80)]


@pytest.mark.parametrize("intermediate,hidden,width,route,a8,capacity", TAIL_CASES)
def test_exact_tail_matches_padding_with_poison_and_graph_replay(
        intermediate, hidden, width, route, a8, capacity, monkeypatch):
    require_b12x()
    changed_fold = (torch.cuda.get_device_capability() == (12, 1) and hidden == 6144
                    and width in (672, 704) and route == "decode")
    if changed_fold:
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    observations = []
    seeds = ((weight, activation) for weight in range(92800, 92804)
             for activation in range(8)) if changed_fold and capacity <= 80 else [(97, 97)]
    for weight_seed, activation_seed in seeds:
        observations.extend(_exact_tail_case(
            intermediate, hidden, width, route, a8, capacity, weight_seed, activation_seed))
    # Large-capacity tail checks retain invariance coverage; the independent
    # C1/C16/C80 cases adjudicate the changed fold across quantization boundaries.
    if changed_fold and capacity <= 80:
        for rows in sorted({item["rows"] for item in observations}):
            samples = [item for item in observations if item["rows"] == rows]
            assert len(samples) == 32
            summary = _seed_oracle_summary(samples)
            print("mxfp4_dispatch_seed_summary " + json.dumps({
                "width": width, "capacity": capacity, "rows": rows,
                "observations": len(samples), "metrics": summary,
            }), flush=True)
            assert all(item["candidate_mean"] <= item["padded_mean"] + item["padded_stdev"]
                       for item in summary.values()), (width, capacity, rows, summary)


def _seed_oracle_summary(samples):
    summary = {}
    for metric in ("max_abs", "rms"):
        candidate = [item["candidate"][metric] for item in samples]
        padded = [item["padded"][metric] for item in samples]
        summary[metric] = {
            "candidate_mean": statistics.fmean(candidate),
            "padded_mean": statistics.fmean(padded),
            "padded_stdev": statistics.stdev(padded),
            "paired_mean_delta": statistics.fmean(a - b for a, b in zip(candidate, padded)),
            "paired_median_delta": statistics.median(a - b for a, b in zip(candidate, padded)),
            "worst_paired_delta": max(a - b for a, b in zip(candidate, padded)),
            "candidate_higher_count": sum(a > b for a, b in zip(candidate, padded)),
        }
    return summary


@pytest.mark.parametrize("delta,passes", [(0.01, True), (10.0, False)])
def test_seed_oracle_mean_bound(delta, passes):
    samples = [{"candidate": {metric: value + delta for metric in ("max_abs", "rms")},
                "padded": {metric: value for metric in ("max_abs", "rms")}}
               for value in range(1, 33)]
    summary = _seed_oracle_summary(samples)
    assert all(item["candidate_mean"] <= item["padded_mean"] + item["padded_stdev"]
               for item in summary.values()) is passes
    assert all(item["candidate_higher_count"] == 32 for item in summary.values())


def _exact_tail_case(intermediate, hidden, width, route, a8, capacity, weight_seed, activation_seed):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    @dataclass(frozen=True)
    class Exact(Fp8MoeGeometry):
        width: int = 352

        @property
        def slice(self):
            return self.width

    # Hidden width meets the production stream's weight ring contract.
    g = Exact("tail", hidden=hidden, experts=16, top_k=8, intermediate=intermediate,
              weights="mxfp4", width=width)
    padded_width = (width + 127) // 128 * 128
    padded = Exact("tail", hidden=hidden, experts=16, top_k=8, intermediate=intermediate,
                   weights="mxfp4", width=padded_width)
    gen = torch.Generator(device="cuda").manual_seed(activation_seed)
    x = torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16()
    source, _ = wire_rows(x)
    changed_fold = (torch.cuda.get_device_capability() == (12, 1) and hidden == 6144
                    and width in (672, 704) and route == "decode")
    if changed_fold:
        from b12x.integration.cuteafd.fp8_moe import _Route
        assert _Route(g, route, capacity, True).down.warps == 4
        assert _Route(padded, route, capacity, True).down.warps == 6
    ids = torch.rand(capacity, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen).contiguous()
    if changed_fold:
        gen = torch.Generator(device="cuda").manual_seed(weight_seed)
    codes = lambda shape: torch.randint(0, 256, shape, device="cuda", generator=gen).byte()
    scales = lambda shape: torch.randint(118, 124, shape, device="cuda", generator=gen).byte()
    e, h = g.experts, g.hidden
    w1, s1 = codes((e, width, h // 2)), scales((e, width, h // 32))
    w3, s3 = codes((e, width, h // 2)), scales((e, width, h // 32))
    w2 = codes((e, h, width // 2))
    stride = (width // 32 + 3) // 4 * 4
    s2 = torch.zeros(e, h, stride, device="cuda", dtype=torch.uint8)
    s2[..., :width // 32] = scales((e, h, width // 32))
    packed = (w1, s1, w3, s3, w2, s2)
    pad = torch.nn.functional.pad
    old = tuple(pad(t, (0, 0, 0, padded_width - width)) for t in packed[:4]) + (
        pad(w2, (0, (padded_width - width) // 2)), s2.clone())
    candidate = compile_fp8_moe_aot(g, route=route, max_rows=capacity, mxfp4_down_a8=a8)
    baseline = compile_fp8_moe_aot(padded, route=route, max_rows=capacity, mxfp4_down_a8=a8)
    out = torch.empty(capacity, h, device="cuda", dtype=torch.bfloat16)
    ref = torch.empty_like(out)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity, mxfp4_down_a8=a8), device="cuda", dtype=torch.uint8)
    old_scratch = torch.empty(fp8_moe_scratch_bytes(padded, route, capacity, mxfp4_down_a8=a8), device="cuda", dtype=torch.uint8)
    buffers = (source, ids, weights, *packed, out, scratch)
    addresses = tuple(t.data_ptr() for t in buffers)
    warm_rows = min(17, capacity)
    candidate.launch(*buffers, scalars=(warm_rows,))
    baseline.launch(source, ids, weights, *old, ref, old_scratch, scalars=(warm_rows,))
    torch.cuda.synchronize()
    oracle = None
    if changed_fold:
        from benchmarks.bench_cuteafd_fp8_moe import f32_errors, mxfp4_f32_reference
        oracle = mxfp4_f32_reference(x, ids, weights,
            {"w1": w1, "w3": w3, "w2": w2},
            {"w1": s1, "w3": s3, "w2": s2[..., :width // 32]})
    observations = []
    live_rows = sorted({min(rows, capacity) for rows in (1, 16, 17, 80, 127, 128, 129, 641, capacity)})
    with kernel_resolution_guard("exact K32 tails across live rows and graph replay"):
        for rows in live_rows:
            scratch.fill_(0xFF)
            out.fill_(float("nan"))
            candidate.launch(*buffers, scalars=(rows,))
            baseline.launch(source, ids, weights, *old, ref, old_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.isfinite(out[:rows]).all() and torch.count_nonzero(out[:rows]) > 0
            cosine = torch.nn.functional.cosine_similarity(out[:rows].float().flatten(), ref[:rows].float().flatten(), dim=0)
            assert float(cosine) >= 0.999998
            # Ordered integer coordinates measure BF16 ULPs across sign/zero.
            def ordered_bits(value):
                bits = value.view(torch.int16).to(torch.int32) & 65535
                return torch.where(bits & 32768 != 0, 65535 - bits, bits + 32768)
            ulps = (ordered_bits(out[:rows]) - ordered_bits(ref[:rows])).abs()
            print("mxfp4_tail_ulp " + json.dumps({
                "hidden": hidden, "width": width, "padded": padded_width,
                "route": route, "a8": a8, "rows": rows,
                "changed": int(torch.count_nonzero(ulps)), "maximum": int(ulps.max()),
                "histogram": {str(bound): int((ulps == bound).sum()) for bound in (0, 1, 2, 3, 4)},
                "over4": int((ulps > 4).sum()), "cosine": float(cosine),
            }), flush=True)
            if changed_fold:
                actual_error = f32_errors(out[:rows], oracle[:rows])
                padded_error = f32_errors(ref[:rows], oracle[:rows])
                observation = {
                    "hidden": hidden, "width": width, "capacity": capacity, "rows": rows,
                    "weight_seed": weight_seed, "activation_seed": activation_seed,
                    "down_warps": 4, "padded_down_warps": 6,
                    "candidate": actual_error, "padded": padded_error,
                }
                print("mxfp4_dispatch_oracle " + json.dumps(observation), flush=True)
                observations.append(observation)
            else:
                assert torch.equal(out[:rows], ref[:rows]), (hidden, width, route, a8, rows, int(ulps.max()))
            result = out[:rows].clone()
            scratch.zero_()
            # Padding bytes must be ignored by all three down kernels.
            s2[..., width // 32:] = 255
            candidate.launch(*buffers, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.equal(out[:rows], result)
            assert tuple(t.data_ptr() for t in buffers) == addresses
        expected = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.launch(*buffers, scalars=(capacity,))
        for _ in range(3):
            out.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocated
            assert torch.equal(out, expected)
            assert tuple(t.data_ptr() for t in buffers) == addresses
    return observations
