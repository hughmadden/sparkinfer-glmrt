"""GPU gates for the real opt-in MXFP4 stream path, its ABI and graph lifetime."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_fp8_moe_aot import wire_rows
from .test_cuteafd_mxfp4_moe_aot import _geometry, _weights, reference


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


@pytest.mark.parametrize("real", [320, 352])
def test_quantized_oracle_padding_poison_and_fixed_resolution(real):
    from benchmarks.bench_cuteafd_mxfp4_down_a8 import (
        require_down_oracle, require_exact_quantization, require_numerical_oracle,
    )
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g, capacity = _geometry(6), 1024
    gen = torch.Generator(device="cuda").manual_seed(19)
    w = _weights(g, real)
    x = torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16()
    source, exact = wire_rows(x)
    ids = torch.rand(capacity, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen).contiguous()
    program = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, mxfp4_down_a8=True)
    baseline = compile_fp8_moe_aot(g, route="stream", max_rows=capacity)
    assert len(program.operands) == 11 and len(program.scalars) == 1
    size = fp8_moe_scratch_bytes(g, "stream", capacity, mxfp4_down_a8=True)
    assert size == program.scratch_bytes(capacity)["scratch"]
    out = torch.empty(capacity, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
    base_out = torch.empty_like(out)
    base_scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity), dtype=torch.uint8, device="cuda")
    buffers = (source, ids, weights, *w, out, scratch)
    addresses = tuple(t.data_ptr() for t in buffers)
    program.launch(*buffers, scalars=(641,))
    torch.cuda.synchronize()
    saved = None
    with kernel_resolution_guard("one prepared A8 program across live tails"):
        for rows in (1, 17, 127, 128, 129, 641, 1000):
            # Unwritten/dead chunk rows must never affect a live row.
            scratch.fill_(0xFF)
            out.fill_(float("nan"))
            program.launch(*buffers, scalars=(rows,))
            baseline.launch(source, ids, weights, *w, base_out, base_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            require_exact_quantization(g, base_scratch, scratch, rows, capacity, real)
            assert tuple(t.data_ptr() for t in buffers) == addresses
            assert torch.isfinite(out[:rows]).all() and torch.count_nonzero(out[:rows]) > 0
            expected = reference(exact[:rows], ids[:rows], weights[:rows], *w, mxfp4_down_a8=True)
            require_numerical_oracle(out[:rows], expected, "A8 full MoE oracle")
            require_down_oracle(g, scratch, out, ids[:rows], weights[:rows], w, rows, capacity, True)
            if rows == 641:
                saved = out[:641].clone()
            if rows == 1000:
                assert torch.equal(out[:641], saved), "future rows changed an earlier row"
            # Different poison must produce exactly the same live outputs.
            result = out[:rows].clone()
            scratch.fill_(0x00)
            program.launch(*buffers, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.equal(out[:rows], result)
        # A zero input row must change its result, without changing any other
        # row. This also rules out accidentally ignoring the input payload.
        original = out[:1000].clone()
        source[0].zero_()
        program.launch(*buffers, scalars=(1000,))
        torch.cuda.synchronize()
        assert torch.count_nonzero(out[0]) == 0
        assert torch.count_nonzero(original[0]) > 0
        assert torch.equal(out[1:1000], original[1:1000])


def test_auto_small_rows_are_byte_exact_and_graph_replay_keeps_storage():
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g, capacity = _geometry(6), 1024
    w = _weights(g, 352)
    gen = torch.Generator(device="cuda").manual_seed(29)
    x = torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16()
    source, _ = wire_rows(x)
    ids = torch.rand(capacity, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen).contiguous()
    base = compile_fp8_moe_aot(g, route="auto", max_rows=capacity)
    candidate = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, mxfp4_down_a8=True)
    out = torch.empty(capacity, g.hidden, dtype=torch.bfloat16, device="cuda")
    old = torch.empty_like(out)
    scratch = torch.empty(max(fp8_moe_scratch_bytes(g, "auto", capacity),
                              fp8_moe_scratch_bytes(g, "auto", capacity, mxfp4_down_a8=True)),
                          dtype=torch.uint8, device="cuda")
    for rows in (1, 64, 639, 640):
        base.launch(source, ids, weights, *w, old, scratch, scalars=(rows,))
        candidate.launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(old[:rows], out[:rows])
    buffers = (source, ids, weights, *w, out, scratch)
    candidate.launch(*buffers, scalars=(641,))
    torch.cuda.synchronize()
    expected = out[:641].clone()
    addresses = tuple(t.data_ptr() for t in buffers)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("A8 down graph capture/replay"):
        with torch.cuda.graph(graph):
            candidate.launch(*buffers, scalars=(641,))
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            out.fill_(float("nan"))
            allocation_bytes = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocation_bytes
            # PyTorch may release a deferred capture temporary. The replay
            # must allocate nothing and must not grow its live storage.
            assert torch.cuda.memory_allocated() <= allocated
            assert torch.equal(out[:641], expected)
            assert tuple(t.data_ptr() for t in buffers) == addresses


def test_a8_down_rejects_non_mxfp4_or_bf16_input_before_compile():
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    for g, wire in ((GEOMETRIES["mimo"].with_tp(4), True), (_geometry(6), False)):
        with pytest.raises(ValueError, match="MXFP4 weights and wire input"):
            fp8_moe_scratch_bytes(g, "auto", 1024, wire=wire, mxfp4_down_a8=True)
        with pytest.raises(ValueError, match="MXFP4 weights and wire input"):
            compile_fp8_moe_aot(g, route="auto", max_rows=1024, wire=wire, mxfp4_down_a8=True)
