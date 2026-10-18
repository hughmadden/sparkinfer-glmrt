"""Shared pointwise operand rounding, row invariance, and native D192 RoPE."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch
from ..conftest import require_b12x


def _native():
    path = os.environ.get("CUTEAFD_STEP5_REFERENCE_KERNEL")
    if not path:
        pytest.skip("set CUTEAFD_STEP5_REFERENCE_KERNEL to official inference/kernel.py")
    spec = importlib.util.spec_from_file_location("step5_reference_kernel", Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _placeholder():
    return torch.zeros(16, dtype=torch.bfloat16, device="cuda")


@pytest.mark.parametrize("rows", [1, 17, 64])
@pytest.mark.parametrize("delta_dtype", [torch.bfloat16, torch.float32])
def test_f32_residual_and_zero_centered_norm(rows, delta_dtype):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot, compile_residual_add_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    residual = torch.randn(rows, 4096, generator=gen, device="cuda")
    delta = torch.randn(residual.shape, generator=gen, device="cuda").to(delta_dtype)
    out = torch.empty_like(residual)
    add = compile_residual_add_aot(width=4096, delta_dtype=delta_dtype)
    add.launch(residual, delta, out, scalars=[rows])
    assert torch.equal(out, residual + delta.float())
    weight = torch.randn(4096, generator=gen, device="cuda") * 0.1
    normed = torch.empty_like(out, dtype=torch.bfloat16)
    norm = compile_affine_norm_aot(width=4096, zero_centered=True)
    dummy = _placeholder()
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    norm.launch(out, weight, weight, dummy, dummy, positions, normed, scalars=[rows])
    ref = (out * torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-5) * (1 + weight)).bfloat16()
    torch.testing.assert_close(normed, ref, atol=0.015625, rtol=0.008)
    repeated = torch.empty_like(normed)
    norm.launch(out[:1].expand_as(out).contiguous(), weight, weight, dummy, dummy, positions,
        repeated, scalars=[rows])
    assert torch.equal(repeated, normed[:1].expand_as(normed))


@pytest.mark.parametrize("heads,rotary", [(64, 192), (32, 192), (4, 64), (2, 64)])
@pytest.mark.parametrize("rows", [1, 17])
def test_qk192_matches_official_fixed_reduction(heads, rotary, rows):
    require_b12x()
    native = _native()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    x = torch.randn(rows, heads, 192, generator=gen, device="cuda").bfloat16()
    w = torch.randn(192, generator=gen, device="cuda") * 0.1
    cos = torch.randn(128, rotary // 2, generator=gen, device="cuda").bfloat16()
    sin = torch.randn(cos.shape, generator=gen, device="cuda").bfloat16()
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") * 3 + 5
    program = compile_affine_norm_aot(width=192, heads=heads, rotary=rotary,
        zero_centered=True, input_dtype=torch.bfloat16)
    out = torch.empty_like(x)
    program.launch(x, w, w, cos, sin, positions, out, scalars=[rows])
    # Add a separate K/V pair to the official packed ABI; the Q result uses
    # exactly the same per-head reduction and F32 norm weight as this export.
    qkv = torch.cat((x.flatten(1), x[:, :1].flatten(1), x[:, :1].flatten(1)), dim=-1)
    reference, _, _ = native.fused_qknorm_rope(qkv, w, w, cos, sin, positions,
        head_dim=192, num_q_heads=heads, num_kv_heads=1, rotary_pairs=rotary // 2)
    assert torch.equal(out.flatten(1), reference)


@pytest.mark.parametrize("subtract_mean", [False, True])
def test_indexer_norm_has_bf16_parameters_and_unchanged_tail(subtract_mean):
    require_b12x()
    native = _native()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    rows, heads, dim, rotary = 17, 1 if subtract_mean else 8, 256, 32
    x = torch.randn(rows, heads, dim, generator=gen, device="cuda").bfloat16()
    w = (torch.randn(dim, generator=gen, device="cuda") * 0.1).bfloat16()
    bias = (torch.randn(dim, generator=gen, device="cuda") * 0.1).bfloat16()
    cos = torch.randn(128, rotary // 2, generator=gen, device="cuda").bfloat16()
    sin = torch.randn(cos.shape, generator=gen, device="cuda").bfloat16()
    pos = torch.arange(rows, dtype=torch.int64, device="cuda")
    out = torch.empty_like(x)
    program = compile_affine_norm_aot(width=dim, heads=heads, rotary=rotary, eps=1e-6,
        zero_centered=not subtract_mean, subtract_mean=subtract_mean, bias=subtract_mean,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16)
    program.launch(x, w, bias, cos, sin, pos, out, scalars=[rows])
    ref = native._norm_rope(x.flatten(1), w, bias if subtract_mean else None,
        cos, sin, pos, num_heads=heads, head_dim=dim, rotary_dim=rotary // 2,
        eps=1e-6, weight_bias=0.0 if subtract_mean else 1.0, subtract_mean=subtract_mean)
    torch.testing.assert_close(out.flatten(1), ref, atol=0.015625, rtol=0.008)
    assert torch.equal(out[:, :, rotary:], ref.view_as(out)[:, :, rotary:])


@pytest.mark.parametrize("rows", [1, 17])
def test_native_head_gate_and_post_silu(rows):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_head_gate_aot, compile_post_silu_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    x = torch.randn(rows, 32, 192, generator=gen, device="cuda").bfloat16()
    logits = torch.randn(rows, 32, generator=gen, device="cuda").bfloat16()
    out = torch.empty_like(x)
    compile_head_gate_aot(heads=32, dim=192).launch(x, logits, out, scalars=[rows])
    assert torch.equal(out, (x.float() * logits.float().sigmoid().bfloat16().float().unsqueeze(-1)).bfloat16())
    gu = (torch.randn(rows, 3072, generator=gen, device="cuda") * 9).bfloat16()
    product = torch.empty(rows, 1536, device="cuda", dtype=torch.bfloat16)
    compile_post_silu_aot(width=1536, limit=7.0).launch(gu, product, scalars=[rows])
    g, u = gu.float().chunk(2, dim=-1)
    ref = (torch.nn.functional.silu(g).clamp(max=7) * u.clamp(-7, 7)).bfloat16()
    assert torch.equal(product, ref)


def test_router_weights_keep_subtract_bias_rounding():
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_route_weights_aot
    logits = torch.tensor([[-1., 1., 0.]], device="cuda").repeat(17, 1)
    bias = torch.tensor([16777216., 0., -1.], device="cuda")
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32).repeat(17, 1)
    out = torch.empty(17, 2, device="cuda")
    compile_route_weights_aot(experts=3, topk=2).launch(logits, bias, ids, out, scalars=[17])
    scores = logits.sigmoid() + bias
    weights = scores.gather(-1, ids.long()) - bias[ids.long()]
    ref = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 3
    assert torch.equal(out, ref)
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))


@pytest.mark.parametrize("outputs", [8, 16])
def test_indexer_weight_projection_is_bf16_then_f32_prescale(outputs):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_bf16_scaled_projection_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    x = torch.randn(17, 4096, generator=gen, device="cuda").bfloat16()
    w = torch.randn(outputs, 4096, generator=gen, device="cuda").bfloat16()
    out = torch.empty(17, outputs, device="cuda")
    scratch = torch.empty(17, outputs, device="cuda", dtype=torch.bfloat16)
    program = compile_bf16_scaled_projection_aot(width=4096, outputs=outputs, prescale=0.5)
    program.launch(x, w, out, scratch, scalars=[17])
    assert torch.equal(out, scratch.float() * 0.5)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w).float() * 0.5, atol=0.5, rtol=0.008)


@pytest.mark.parametrize("outputs,width", [(352, 4096), (768, 4096), (4096, 1536)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_canonical_projection_dynamic_rows_and_graph(outputs, width, dtype):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    x = torch.randn(64, width, generator=gen, device="cuda").bfloat16()
    # Repeat a row to prove serving-width invariance independently of torch GEMM.
    x[1:] = x[:1]
    weight = (torch.randn(outputs, width, generator=gen, device="cuda") * 0.01).bfloat16()
    out = torch.empty(64, outputs, dtype=dtype, device="cuda")
    program = compile_bf16_projection_aot(width=width, outputs=outputs, output_dtype=dtype)
    program.launch(x, weight, out, scalars=[64])
    torch.cuda.synchronize()
    expected = out.clone()
    reference = (x[:1].double() @ weight.double().t()).to(dtype)
    relative = torch.linalg.vector_norm(expected[:1].double() - reference.double()) / torch.linalg.vector_norm(reference.double())
    assert float(relative) < (0.004 if dtype == torch.bfloat16 else 2e-5)
    assert bool(torch.isfinite(expected).all()) and bool(expected.abs().max() > 0)
    assert torch.equal(expected, expected[:1].expand_as(expected))
    for live in [1, 16, 17, 64]:
        out.fill_(float("nan"))
        program.launch(x, weight, out, scalars=[live])
        torch.cuda.synchronize()
        assert torch.equal(out[:live], expected[:live])
        assert bool(out[live:].isnan().all())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(x, weight, out, scalars=[64])
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


@pytest.mark.parametrize("rows", [1, 17, 64])
def test_canonical_router_matches_widened_fp32_logits_and_topk(rows):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot, compile_route_weights_aot
    gen = torch.Generator(device="cuda").manual_seed(274)
    hidden = torch.randn(rows, 4096, generator=gen, device="cuda").bfloat16()
    weight = (torch.randn(352, 4096, generator=gen, device="cuda") * 0.01).bfloat16()
    bias = torch.randn(352, generator=gen, device="cuda") * 0.1
    logits = torch.empty(rows, 352, device="cuda")
    program = compile_bf16_projection_aot(width=4096, outputs=352, output_dtype=torch.float32)
    program.launch(hidden, weight, logits, scalars=[rows])
    # The official router widens these native BF16 operands before its FP32 dot.
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        reference = torch.nn.functional.linear(hidden.float(), weight.float())
    finally:
        torch.set_float32_matmul_precision(previous)
    # Different FP32 reduction orders can disagree near zero. Check both
    # implementations against an independent FP64 dot, not a relative error
    # per almost-zero element, using the canonical projection's existing bar.
    oracle = hidden.double() @ weight.double().t()
    norm = torch.linalg.vector_norm(oracle)
    for actual in (logits, reference):
        relative = torch.linalg.vector_norm(actual.double() - oracle) / norm
        assert float(relative) < 2e-5
    delta = (logits - reference).abs().max()
    print(f"router rows={rows}: max_logit_delta={float(delta):.9g}")
    scores = logits.sigmoid() + bias
    reference_scores = reference.sigmoid() + bias
    ids = scores.topk(8, dim=-1).indices
    reference_ids = reference_scores.topk(8, dim=-1).indices
    assert torch.equal(ids, reference_ids)
    # Rule out a vacuous agreement from a tied eighth/ninth selection boundary.
    boundary = reference_scores.topk(9, dim=-1).values
    assert bool(((boundary[:, 7] - boundary[:, 8]) > 1e-6).all())
    out = torch.empty(rows, 8, device="cuda")
    compile_route_weights_aot(experts=352, topk=8).launch(
        logits, bias, ids.to(torch.int32), out, scalars=[rows])
    selected = reference_scores.gather(-1, reference_ids) - bias[reference_ids]
    expected = selected / (selected.sum(-1, keepdim=True) + 1e-20) * 3
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_canonical_projection_export_abi(tmp_path, dtype):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    from ._cuteafd import check_export
    with exportable_compilation():
        program = compile_bf16_projection_aot(width=4096, outputs=352, output_dtype=dtype)
    check_export(program, tmp_path, "bf16_projection_" + str(dtype).split(".")[-1])


def test_pointwise_export_abi(tmp_path):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.pointwise import compile_residual_add_aot
    from ._cuteafd import check_export
    with exportable_compilation():
        program = compile_residual_add_aot(width=4096)
    check_export(program, tmp_path, "residual_add_f32")


@pytest.mark.parametrize("operation", ["norm", "qk", "gate", "route", "silu", "projection"])
def test_all_pointwise_export_abis(tmp_path, operation):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import pointwise as p
    from ._cuteafd import check_export
    builders = {
        "norm": lambda: p.compile_affine_norm_aot(width=4096, zero_centered=True),
        "qk": lambda: p.compile_affine_norm_aot(width=192, heads=32, rotary=192,
            zero_centered=True, input_dtype=torch.bfloat16),
        "gate": lambda: p.compile_head_gate_aot(heads=32, dim=192),
        "route": lambda: p.compile_route_weights_aot(experts=352, topk=8),
        "silu": lambda: p.compile_post_silu_aot(width=1536, limit=7.0),
        "projection": lambda: p.compile_bf16_scaled_projection_aot(width=4096, outputs=8, prescale=0.5),
    }
    with exportable_compilation():
        program = builders[operation]()
    check_export(program, tmp_path, "pointwise_" + operation)


def test_graph_replay_and_dynamic_live_rows():
    require_b12x()
    from b12x.integration.cuteafd import pointwise as p
    rows = 17
    gen = torch.Generator(device="cuda").manual_seed(274)
    residual = torch.randn(rows, 4096, generator=gen, device="cuda")
    delta = torch.randn(rows, 4096, generator=gen, device="cuda").bfloat16()
    weight = torch.randn(4096, generator=gen, device="cuda") * 0.1
    projection = torch.randn(8, 4096, generator=gen, device="cuda").bfloat16()
    summed = torch.empty_like(residual)
    normed = torch.empty_like(delta)
    scratch = torch.empty(rows, 8, dtype=torch.bfloat16, device="cuda")
    scores = torch.empty(rows, 8, device="cuda")
    gu = torch.randn(rows, 3072, generator=gen, device="cuda").bfloat16()
    product = torch.empty(rows, 1536, dtype=torch.bfloat16, device="cuda")
    attn = torch.randn(rows, 32, 192, generator=gen, device="cuda").bfloat16()
    gates = torch.randn(rows, 32, generator=gen, device="cuda").bfloat16()
    gated = torch.empty_like(attn)
    route_logits = torch.randn(rows, 352, generator=gen, device="cuda")
    route_bias = torch.randn(352, generator=gen, device="cuda")
    route_ids = torch.arange(8, dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
    route_out = torch.empty(rows, 8, device="cuda")
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    dummy = _placeholder()
    add = p.compile_residual_add_aot(width=4096)
    norm = p.compile_affine_norm_aot(width=4096, zero_centered=True)
    project = p.compile_bf16_scaled_projection_aot(width=4096, outputs=8, prescale=0.5)
    silu = p.compile_post_silu_aot(width=1536, limit=7.0)
    gate = p.compile_head_gate_aot(heads=32, dim=192)
    route = p.compile_route_weights_aot(experts=352, topk=8)

    def launch(live):
        add.launch(residual, delta, summed, scalars=[live])
        norm.launch(summed, weight, weight, dummy, dummy, positions, normed, scalars=[live])
        project.launch(normed, projection, scores, scratch, scalars=[live])
        silu.launch(gu, product, scalars=[live])
        gate.launch(attn, gates, gated, scalars=[live])
        route.launch(route_logits, route_bias, route_ids, route_out, scalars=[live])

    launch(rows)
    torch.cuda.synchronize()
    outputs = [summed, normed, scores, product, gated, route_out]
    expected = [out.clone() for out in outputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        launch(rows)
    torch.cuda.current_stream().wait_stream(stream)
    for out in outputs:
        out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(outputs, expected):
        assert torch.equal(actual, reference)
    for live in (1, 3, 16):
        for out in outputs:
            out.fill_(float("nan"))
        launch(live)
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected):
            assert torch.equal(actual[:live], reference[:live])
            assert bool(actual[live:].isnan().all())
