"""Shared activation contracts and native BF16 experts, CPU oracle first."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

PRE = "silu_bf16_product"
POST = "silu_post_clamp_fp32_product"


def activation(gate, up, limit, contract):
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        if contract == PRE:
            gate = gate.clamp(max=limit)
    silu = torch.nn.functional.silu(gate)
    if contract == PRE:
        silu = silu.bfloat16().float()
    elif contract == POST:
        if limit > 0:
            silu = silu.clamp(max=limit)
    else:
        raise ValueError(contract)
    return (silu * up).bfloat16()


def test_cpu_activation_separates_order_and_rounding():
    gate = torch.tensor([10.0], dtype=torch.bfloat16)
    up = torch.tensor([1.0], dtype=torch.bfloat16)
    assert activation(gate, up, 2.0, POST).item() == 2.0
    assert activation(gate, up, 2.0, PRE).item() < 1.8
    gate = torch.linspace(-6, 6, 4096).bfloat16()
    up = torch.linspace(0.125, 6.5, 4096).bfloat16()
    pre = activation(gate, up, 7.0, PRE)
    post = activation(gate, up, 7.0, POST)
    assert torch.count_nonzero(pre != post) > 100
    assert torch.equal(post, (torch.nn.functional.silu(gate.float()).clamp(max=7) * up.float()).bfloat16())
    assert torch.equal(pre, torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize("tp,width", [(4, 384), (6, 256)])
def test_cpu_geometry_contract(tp, width):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, Fp8MoeGeometry, fp8_moe_scratch_bytes

    for name in ("step5", "step5_bf16", "step5_nvfp4"):
        g = GEOMETRIES[name].with_tp(tp)
        assert (g.hidden, g.experts, g.top_k, g.slice) == (4096, 352, 8, width)
        assert (g.activation, g.projection, g.swiglu_limit) == (POST, "bf16", 7.0)
    g = GEOMETRIES["step5_bf16"].with_tp(tp)
    from b12x.integration.cuteafd._fp8_moe_kernels import GroupedFp8Gemm

    grouped = GroupedFp8Gemm(n=width, k=g.hidden, experts=g.experts, native_bf16=True)
    staging_bytes = (grouped.tile_m * 2 + grouped.tile_n * 2) * grouped.tile_k * grouped.num_stages
    expanded_bytes = grouped.tile_n * grouped.tile_k * 2
    assert staging_bytes + expanded_bytes + 1024 <= 99 * 1024
    assert fp8_moe_scratch_bytes(g, "decode", 16, False) > 0
    assert fp8_moe_scratch_bytes(g, "prefill", 129, False) > fp8_moe_scratch_bytes(g, "decode", 16, False)
    with pytest.raises(ValueError, match="BF16 input"):
        fp8_moe_scratch_bytes(g, "decode", 16, True)
    with pytest.raises(ValueError, match="activation contract"):
        Fp8MoeGeometry("bad", 4096, 352, 8, 1536, activation="unknown")


def bf16_reference(x, ids, route_weights, w1, w3, w2, limit, contract):
    # Sum in slot order, as MoeCombine does (including duplicate expert ids).
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        selected = ids[:, slot]
        for expert in selected.unique().tolist():
            if expert < 0 or expert >= w1.shape[0]:
                continue
            rows = torch.where(selected == expert)[0]
            gate = (x[rows].float() @ w1[expert].float().T).bfloat16()
            up = (x[rows].float() @ w3[expert].float().T).bfloat16()
            act = activation(gate, up, limit, contract)
            down = (act.float() @ w2[expert].float().T).bfloat16()
            result[rows] += down.float() * route_weights[rows, slot, None]
    return result.bfloat16()


def test_cpu_bf16_reference_empty_duplicate_and_inactive_routes():
    gen = torch.Generator().manual_seed(23)
    x = torch.randn(3, 16, generator=gen).bfloat16()
    w1, w3 = [torch.randn(4, 8, 16, generator=gen).bfloat16() for _ in range(2)]
    w2 = torch.randn(4, 16, 8, generator=gen).bfloat16()
    ids = torch.tensor([[0, 0], [2, -1], [-1, -1]])
    weights = torch.tensor([[0.25, 0.75], [1.0, 0], [0, 0]])
    actual = bf16_reference(x, ids, weights, w1, w3, w2, 7, POST)
    expected = bf16_reference(x, torch.tensor([[0], [2], [-1]]), torch.ones(3, 1), w1, w3, w2, 7, POST)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[2]) == 0
    assert torch.count_nonzero(actual[:2]) > 0


@pytest.mark.parametrize("contract,limit", [(PRE, 2.0), (PRE, 7.0), (POST, 2.0), (POST, 7.0)])
def test_gpu_activation(contract, limit):
    require_b12x()
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd._fp8_moe_kernels import MoeSwiGLU

    gen = torch.Generator(device="cuda").manual_seed(19)
    gu = (torch.randn(7, 512, device="cuda", generator=gen) * 4).bfloat16()
    meta = torch.tensor([0, 0, 7, 0], device="cuda", dtype=torch.int32)
    out = torch.empty(7, 256, device="cuda", dtype=torch.bfloat16)
    args = [from_dlpack(t).iterator for t in (gu, meta, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = MoeSwiGLU(inter=256, limit=limit, activation=contract)
    compiled = cute.compile(kernel, *args, cutlass.Int32(7), stream)
    compiled(*args, 7, stream)
    expected = activation(gu[:, :256], gu[:, 256:], limit, contract)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [4, 6])
@pytest.mark.parametrize("route,capacity,rows", [("decode", 16, 1), ("decode", 16, 11), ("prefill", 128, 79)])
def test_gpu_native_bf16(tp, route, capacity, rows):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    torch.backends.cuda.matmul.allow_tf32 = False
    g = GEOMETRIES["step5_bf16"].with_tp(tp)
    gen = torch.Generator(device="cuda").manual_seed(29)
    # Full installed geometry; only a few experts are live, leaving empty groups.
    w1, w3 = [(torch.randn(g.experts, g.slice, g.hidden, device="cuda", generator=gen) * 0.03).bfloat16()
              for _ in range(2)]
    w2 = (torch.randn(g.experts, g.hidden, g.slice, device="cuda", generator=gen) * 0.03).bfloat16()
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    ids = torch.randint(0, 5, (rows, g.top_k), device="cuda", generator=gen, dtype=torch.int32)
    ids[:, 0] = 0  # skew + duplicate routes + multi-tile occupancy
    if rows > 1:
        ids[-1] = -1
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen) / g.top_k
    out = torch.empty_like(x)
    unused = torch.empty(0, device="cuda", dtype=torch.uint8)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity, False), device="cuda", dtype=torch.uint8)
    program = compile_fp8_moe_aot(g, route=route, max_rows=capacity, wire=False)
    program.launch(x, ids, weights, w1, unused, w3, unused, w2, unused, out, scratch, scalars=(rows,))
    expected = bf16_reference(x, ids, weights, w1, w3, w2, 7, POST)
    torch.testing.assert_close(out, expected, rtol=0.025, atol=0.015)
    assert torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
    assert torch.count_nonzero(out) > 0
    if rows > 1:
        assert torch.count_nonzero(out[-1]) == 0
    from b12x._lib.runtime_control import kernel_resolution_guard

    tensors = (x, ids, weights, w1, unused, w3, unused, w2, unused, out, scratch)
    pointers = tuple(t.data_ptr() for t in tensors)
    with kernel_resolution_guard("native BF16 live-count reuse and graph replay"):
        # Reuse installed capacity and storage after shrinking the live prefix.
        live = max(1, rows // 2)
        program.launch(*tensors, scalars=(live,))
        torch.testing.assert_close(out[:live], expected[:live], rtol=0.025, atol=0.015)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*tensors, scalars=(rows,))
        graph.replay()
        torch.cuda.synchronize()
    assert tuple(t.data_ptr() for t in tensors) == pointers
    torch.testing.assert_close(out, expected, rtol=0.025, atol=0.015)


@pytest.mark.parametrize("fused", ["0", "1"])
@pytest.mark.parametrize("limit", [2.0, 7.0])
def test_gpu_nvfp4_post_silu(fused, limit, monkeypatch):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from .test_cuteafd_nvfp4_moe_aot import _weights, reference

    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", fused)
    g = Fp8MoeGeometry("post_nvfp4_test", 256, 8, 2, 256, swiglu_limit=limit,
                      weights="nvfp4", activations="a4", activation=POST)
    w = _weights(g, g.slice)
    gen = torch.Generator(device="cuda").manual_seed(73)
    x = (torch.randn(41, g.hidden, generator=gen, device="cuda") * 2).bfloat16()
    ids = torch.zeros(41, 2, dtype=torch.int32, device="cuda")
    ids[:, 1] = torch.arange(41, device="cuda", dtype=torch.int32) % g.experts
    weights = torch.full((41, 2), 0.5, device="cuda")
    out = torch.empty_like(x)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", 64, False), device="cuda", dtype=torch.uint8)
    program = compile_fp8_moe_aot(g, route="auto", max_rows=64, wire=False)
    program.launch(x, ids, weights, *w, out, scratch, scalars=(41,))
    expected = reference(g, x, ids, weights, *w, a4=True)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    assert torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0) > 0.9999


@pytest.mark.parametrize("route", ["decode", "prefill", "stream"])
def test_gpu_fp8_post_silu(route):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from .test_cuteafd_fp8_moe_aot import wire_rows

    torch.backends.cuda.matmul.allow_tf32 = False
    g = Fp8MoeGeometry("post_fp8_test", 512, 8, 2, 256, swiglu_limit=7.0, activation=POST)
    gen = torch.Generator(device="cuda").manual_seed(77)
    x = torch.randn(39, g.hidden, device="cuda", generator=gen).bfloat16()
    ids = torch.randint(0, 8, (39, 2), device="cuda", generator=gen, dtype=torch.int32)
    weights = torch.rand(39, 2, device="cuda", generator=gen) / 2
    shapes = [(8, g.slice, g.hidden), (8, g.slice, g.hidden), (8, g.hidden, g.slice)]
    w = [(torch.randn(shape, device="cuda", generator=gen) * 0.1).to(torch.float8_e4m3fn) for shape in shapes]
    scales = [torch.ones(e, n // 128, k // 128, device="cuda") for e, n, k in shapes]
    out = torch.empty_like(x)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, 64, False), device="cuda", dtype=torch.uint8)
    program = compile_fp8_moe_aot(g, route=route, max_rows=64, wire=False)
    program.launch(x, ids, weights, w[0], scales[0], w[1], scales[1], w[2], scales[2], out, scratch, scalars=(39,))
    source = wire_rows(x)[1] if route == "stream" else x
    expected = bf16_reference(source, ids, weights, *(v.bfloat16() for v in w), 7.0, POST)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.025)
    assert torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
