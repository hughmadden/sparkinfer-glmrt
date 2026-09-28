"""cuteafd DSV4 shared expert, router scores and expert input quantizer AOT programs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, cosine, has_checkpoint, prepare, u8

LAYER = 2


def _block_fp8(rows, cols, gen, device):
    weight = (torch.randn((rows, cols), generator=gen) / 16).to(torch.float8_e4m3fn).to(device)
    scale = torch.randint(118, 124, (rows // 128, cols // 128), generator=gen, dtype=torch.uint8).to(device)
    return weight, scale


def _shared_weights(geometry, device):
    """Packed exactly like b12x_model.load_layer_weights: w13 = cat(w1 (gate), w3 (up))."""
    from b12x.gemm import block_fp8_linear

    if geometry.name == "flash" and has_checkpoint():
        g = lambda n: checkpoint_tensor(f"layers.{LAYER}.ffn.shared_experts.{n}", device)  # noqa: E731
        w1, w3, w2 = g("w1.weight"), g("w3.weight"), g("w2.weight")
        s1, s3, s2 = u8(g("w1.scale")), u8(g("w3.scale")), u8(g("w2.scale"))
    else:
        gen = torch.Generator(device="cpu").manual_seed(geometry.hidden)
        h, i = geometry.hidden, geometry.moe_inter
        (w1, s1), (w3, s3), (w2, s2) = (_block_fp8(i, h, gen, device), _block_fp8(i, h, gen, device),
                                        _block_fp8(h, i, gen, device))
    w13 = block_fp8_linear.pack_weight(
        torch.cat((w1.view(torch.uint8), w3.view(torch.uint8))).view(torch.float8_e4m3fn), torch.cat((s1, s3)))
    return w13, block_fp8_linear.pack_weight(w2, s2)


def _fp8_linear(x, weight, max_rows):
    """Prototype OpPlans.fp8_linear (prepared block_fp8_linear, K128 activations)."""
    from b12x.gemm import block_fp8_linear

    plan = prepare(block_fp8_linear.plan(block_fp8_linear.Caps(
        device=x.device, max_tokens=max_rows, in_features=weight.in_features,
        out_features=weight.out_features, activation_block_size=128)), "test.shared_ffn")
    spec = plan.scratch_specs()[0]
    out = torch.empty((x.shape[0], weight.out_features, 1), dtype=torch.bfloat16, device=x.device)
    binding = block_fp8_linear.bind(plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=x.device),
                                    source=x, packed_weight=weight, output=out, activation_block_size=128)
    block_fp8_linear.run(binding=binding)
    return out[:, :, 0]


def _prototype_shared(x, w13, w2, geometry, max_rows):
    gate_up = _fp8_linear(x, w13, max_rows)
    gate, up = gate_up.float().split(geometry.moe_inter, dim=-1)
    up = up.clamp(-geometry.swiglu_limit, geometry.swiglu_limit)
    gate = gate.clamp(max=geometry.swiglu_limit)
    hidden = (F.silu(gate) * up).to(torch.bfloat16).contiguous()
    return _fp8_linear(hidden, w2, max_rows), gate_up, hidden


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd import dsv4_ffn

    with exportable_compilation():
        return {
            ("ffn", "flash", 16): dsv4_ffn.compile_dsv4_shared_ffn_aot(FLASH, max_rows=16),
            ("ffn", "flash", 512): dsv4_ffn.compile_dsv4_shared_ffn_aot(FLASH, max_rows=512),
            ("ffn", "flash", 1): dsv4_ffn.compile_dsv4_shared_ffn_aot(FLASH, max_rows=1),
            ("ffn", "flash", 8): dsv4_ffn.compile_dsv4_shared_ffn_aot(FLASH, max_rows=8),
            ("ffn", "pro", 8): dsv4_ffn.compile_dsv4_shared_ffn_aot(PRO, max_rows=8),
            ("ffn", "pro", 300): dsv4_ffn.compile_dsv4_shared_ffn_aot(PRO, max_rows=300),
            ("router", "flash"): dsv4_ffn.compile_dsv4_router_scores_aot(FLASH),
            ("router", "pro"): dsv4_ffn.compile_dsv4_router_scores_aot(PRO),
            ("quant", "flash"): dsv4_ffn.compile_dsv4_expert_input_quant_aot(FLASH),
            ("quant", "pro"): dsv4_ffn.compile_dsv4_expert_input_quant_aot(PRO),
        }


def _geometry(name):
    from b12x.integration.cuteafd import FLASH, PRO

    return FLASH if name == "flash" else PRO


def _x(rows, hidden, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((rows, hidden), generator=gen).bfloat16().to(device)


def _run_ffn(program, x, w13, w2, max_rows):
    out = torch.full((x.shape[0], x.shape[1]), float("nan"), dtype=torch.bfloat16, device=x.device)
    scratch = torch.full((program.scratch_bytes(max_rows)["scratch"],), 0xFF, dtype=torch.uint8, device=x.device)
    program.launch(x, w13.weight.values, w13.weight.scale_mma, w2.weight.values, w2.weight.scale_mma,
                   out, scratch, scalars=(x.shape[0],))
    return out, scratch


@pytest.mark.parametrize(("name", "max_rows", "rows"), [
    ("flash", 16, 1), ("flash", 16, 9), ("flash", 16, 16), ("flash", 512, 300), ("flash", 512, 512),
    ("pro", 300, 257),
])
def test_shared_ffn_matches_prototype_bitwise(programs, name, max_rows, rows):
    device = require_b12x()
    geometry = _geometry(name)
    w13, w2 = _shared_weights(geometry, device)
    x = _x(rows, geometry.hidden, rows, device)
    expected, _, _ = _prototype_shared(x, w13, w2, geometry, max_rows)
    out, _ = _run_ffn(programs[("ffn", name, max_rows)], x, w13, w2, max_rows)
    torch.cuda.synchronize()
    assert_bitwise(out, expected, "out")


@pytest.mark.parametrize(("name", "max_rows", "rows"), [
    ("flash", 1, 1), ("flash", 8, 5), ("pro", 8, 3), ("pro", 8, 8)])
def test_shared_ffn_split_k_capacities(programs, name, max_rows, rows):
    """Capacities whose prepared lowering splits K with BF16 atomics.

    [w1;w3] splits at Flash capacity 1 (2 slices) and 8 (4 slices) and Pro
    capacities 1-8 (2 slices); w2 never splits.

    The AOT stage reduces FP32 partial planes once (deterministic), so the
    result must be at least as close to an unsplit FP32 GEMM of the same
    quantized operands as the prepared path is, and close to it.
    """
    device = require_b12x()
    geometry = _geometry(name)
    w13, w2 = _shared_weights(geometry, device)
    x = _x(rows, geometry.hidden, 50 + rows, device)
    prepared, _, _ = _prototype_shared(x, w13, w2, geometry, max_rows)
    unsplit, _, _ = _prototype_shared(x, w13, w2, geometry, 64)  # no split-K at 64 rows
    out, _ = _run_ffn(programs[("ffn", name, max_rows)], x, w13, w2, max_rows)
    torch.cuda.synchronize()
    err = lambda a: float((a.float() - unsplit.float()).abs().max())  # noqa: E731
    assert err(out) <= err(prepared)
    assert cosine(out, unsplit) > 0.99995


def test_shared_ffn_swiglu_matches_torch_bitwise(programs):
    """The CuTe clamped SwiGLU equals torch's FP32 F.silu path on real activations."""
    device = require_b12x()
    geometry = _geometry("flash")
    w13, w2 = _shared_weights(geometry, device)
    rows, max_rows = 64, 512
    program = programs[("ffn", "flash", max_rows)]
    x = _x(rows, geometry.hidden, 7, device) * 4  # push some gates/ups past the +-10 clamp
    _, scratch = _run_ffn(program, x, w13, w2, max_rows)
    torch.cuda.synchronize()
    i = geometry.moe_inter
    gate_up = scratch[: rows * 2 * i * 2].view(torch.bfloat16).view(rows, 2 * i)
    offset = (rows * 4 * i + 1023) // 1024 * 1024
    hidden = scratch[offset: offset + rows * i * 2].view(torch.bfloat16).view(rows, i)
    gate, up = gate_up.float().split(i, dim=-1)
    assert bool((gate.abs() > 10).any()) and bool((up.abs() > 10).any())
    expected = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(torch.bfloat16)
    assert_bitwise(hidden, expected, "swiglu")


def test_shared_ffn_replays_in_cuda_graph(programs):
    device = require_b12x()
    geometry = _geometry("flash")
    w13, w2 = _shared_weights(geometry, device)
    program, rows = programs[("ffn", "flash", 16)], 5
    x = _x(rows, geometry.hidden, 3, device)
    out = torch.empty_like(x)
    scratch = torch.empty((program.scratch_bytes(16)["scratch"],), dtype=torch.uint8, device=device)
    args = (x, w13.weight.values, w13.weight.scale_mma, w2.weight.values, w2.weight.scale_mma, out, scratch)
    program.launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    eager = out.clone()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(*args, scalars=(rows,))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(out, eager, "graph out")


@pytest.mark.parametrize(("name", "rows"), [("flash", 1), ("flash", 13), ("flash", 32), ("flash", 33), ("flash", 300), ("pro", 7), ("pro", 32), ("pro", 129)])
def test_router_scores_match_fp32_gate(programs, name, rows):
    """Prototype router: logits = x.float() @ gate.float().t() (FP32)."""
    device = require_b12x()
    geometry = _geometry(name)
    e, h = geometry.routed_experts, geometry.hidden
    if name == "flash" and has_checkpoint():
        gate = checkpoint_tensor(f"layers.{LAYER}.ffn.gate.weight", device).contiguous()
    else:
        gate = (torch.randn((e, h), generator=torch.Generator().manual_seed(e)) / 64).bfloat16().to(device)
    x = _x(rows, h, 900 + rows, device)
    logits = torch.full((rows, e), float("nan"), device=device)
    programs[("router", name)].launch(x, gate, logits, scalars=(rows,))
    expected = x.float() @ gate.float().t()
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
    # Same top-6 experts as the FP32 reference except exact near-ties.
    top = logits.topk(6, dim=-1).indices.sort().values
    ref = expected.topk(6, dim=-1).indices.sort().values
    assert float((top == ref).all(dim=-1).float().mean()) > 0.99


def _wire_reference(x, hidden):
    rows = x.shape[0]
    blocks = x.float().reshape(rows, -1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    payload = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return payload.reshape(rows, hidden), (exponent + 127).to(torch.uint8)


@pytest.mark.parametrize(("name", "rows"), [("flash", 1), ("flash", 80), ("flash", 4096), ("pro", 6), ("pro", 300)])
def test_expert_input_quant_wire_rows(programs, name, rows):
    from b12x.integration.cuteafd.dsv4_ffn import expert_input_quant_grid

    device = require_b12x()
    geometry = _geometry(name)
    h = geometry.hidden
    stride = h + h // 32
    x = _x(rows, h, rows, device)
    x[0].zero_()
    if rows > 2:
        x[1].mul_(1e-7)
        x[2].mul_(300)
    wire = torch.full((rows + 1, stride), 255, dtype=torch.uint8, device=device)
    unused = torch.full((16,), 93, dtype=torch.uint8, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    program = programs[("quant", name)]
    for grid in (expert_input_quant_grid(h, rows, sms), 1):  # any grid >= 1 is correct
        wire.fill_(255)
        program.launch(x, wire, wire[0, h:], unused, scalars=(rows, grid))
        torch.cuda.synchronize()
        payload, scales = _wire_reference(x, h)
        assert_bitwise(wire[:rows, :h], payload, "payload")
        assert_bitwise(wire[:rows, h:], scales, "scales")
        assert bool((wire[rows:] == 255).all()) and bool((unused == 93).all())


@pytest.mark.parametrize("key", [("ffn", "flash", 16), ("ffn", "pro", 8), ("router", "flash"), ("router", "pro"),
                                 ("quant", "flash"), ("quant", "pro")])
def test_export_to_c_signature(programs, key, tmp_path):
    stem = "dsv4_" + "_".join(map(str, key))
    info = check_export(programs[key], tmp_path, stem)
    assert info["argument_count"] == len(programs[key].operands) + len(programs[key].scalars) + 1
