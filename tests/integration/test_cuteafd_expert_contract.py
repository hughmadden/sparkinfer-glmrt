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
    # Capture once with fixed pointers/scratch; replay must neither resolve nor allocate kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(x, ids, weights, w1, unused, w3, unused, w2, unused, out, scratch, scalars=(rows,))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0.025, atol=0.015)
