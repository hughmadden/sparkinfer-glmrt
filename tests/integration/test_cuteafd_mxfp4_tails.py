"""Exact MXFP4 K32 tails against identical, zero-padded weights on SM12x."""
from dataclasses import dataclass
import json

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
@pytest.mark.parametrize("route,a8", [("decode", False), ("stream", False), ("stream", True)])
def test_exact_tail_matches_padding_with_poison_and_graph_replay(intermediate, hidden, width, route, a8):
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
    gen = torch.Generator(device="cuda").manual_seed(97)
    capacity = 4096
    source, _ = wire_rows(torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16())
    ids = torch.rand(capacity, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen).contiguous()
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
    candidate.launch(*buffers, scalars=(17,))
    baseline.launch(source, ids, weights, *old, ref, old_scratch, scalars=(17,))
    torch.cuda.synchronize()
    with kernel_resolution_guard("exact K32 tails across live rows and graph replay"):
        for rows in (1, 17, 127, 128, 129, 641, 4096):
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
            if width == padded_width:
                assert torch.equal(out[:rows], ref[:rows])
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
