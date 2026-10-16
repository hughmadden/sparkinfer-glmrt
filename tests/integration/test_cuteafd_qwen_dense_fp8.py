"""Native dense checkpoint formats; gates remain BF16 even with FP8 matrices."""
import pytest
import torch

from ..conftest import require_b12x


def _weight(rows, cols):
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).bfloat16()
    blocks = weight.float().reshape(rows // 128, 128, cols // 128, 128)
    scale = (blocks.abs().amax((1, 3)) / 448).bfloat16().float()
    quantized = (blocks / scale[:, None, :, None]).reshape(rows, cols).to(torch.float8_e4m3fn)
    restored = (quantized.float().reshape_as(blocks) * scale[:, None, :, None]).reshape(rows, cols).bfloat16()
    return quantized, scale, restored


@pytest.mark.parametrize("rows", [3, 65])
def test_native_dense_fp8_gdn_preserves_bf16_gates(rows):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_gdn_aot
    from b12x.integration.cuteafd.qwen_dense_fp8 import compile_qwen_dense_gdn_fp8_aot
    g = QwenDenseGeometry(hidden=256, intermediate=256, gdn_key_heads=1, gdn_value_heads=2)
    capacity = 64 if rows <= 64 else 128
    native = compile_qwen_dense_gdn_fp8_aot(g, max_rows=capacity)
    reference = compile_qwen_dense_gdn_aot(g, max_rows=capacity, norm_dtype=torch.bfloat16)
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    qkvz, in_scale, in_bf16 = _weight(g.gdn_conv_width + g.gdn_value_width, g.hidden)
    # Deliberately different scales and signs for b/a: quantizing or swapping them
    # changes the recurrence, rather than only a nearly-zero projection result.
    ba = torch.randn(2 * g.gdn_value_heads, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.4
    out_weight, out_scale, out_bf16 = _weight(g.hidden, g.gdn_value_width)
    packed = torch.cat((in_bf16, ba))
    conv_w = torch.randn(g.gdn_conv_width, 4, device="cuda") * 0.1
    a_log = torch.randn(g.gdn_value_heads, device="cuda") * 0.1
    bias = torch.randn_like(a_log) * 0.1
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(rows, device="cuda", dtype=torch.int32)
    first = torch.zeros_like(slots)
    outputs = []
    states = []
    tails = []
    for program, weights in ((native, (qkvz, in_scale, ba)), (reference, (packed,))):
        state = torch.zeros(1, g.gdn_value_heads, 128, 128, device="cuda")
        conv = torch.zeros(1, 3, g.gdn_conv_width, device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x)
        scratch = torch.empty(program.scratch_bytes(capacity)["scratch"], device="cuda", dtype=torch.uint8)
        out_weights = (out_weight, out_scale) if program is native else (out_bf16,)
        args = (x, *weights, conv_w, a_log, bias, norm, *out_weights, conv, state, slots, first, output)
        if rows <= 64:
            program.launch(*args, None, scratch, scalars=[rows, 0])
        else:
            program.launch(*args, scratch, scalars=[rows])
        outputs.append(output)
        states.append(state)
        tails.append(conv)
        projected = scratch[:rows * g.gdn_in_width * 2].view(torch.bfloat16).reshape(rows, g.gdn_in_width)
        expected_ba = torch.nn.functional.linear(x, ba)
        torch.testing.assert_close(projected[:, -2 * g.gdn_value_heads:], expected_ba, atol=0.004, rtol=0.02)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.001, rtol=0.08)
    torch.testing.assert_close(states[0], states[1], atol=0.001, rtol=0.08)
    torch.testing.assert_close(tails[0], tails[1], atol=0.004, rtol=0.08)
    names = [op.name for op in native.operands]
    assert "w_ba" in names and "w_qkvz_fp8" in names
    assert "w_in" not in names and "w_out" not in names
    assert native.geometry["gate_weights"] == "bf16"
    assert native.geometry["gate_order"] == ["b", "a"]


@pytest.mark.parametrize("rows", [3, 65])
def test_native_dense_fp8_ffn(rows):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense_fp8 import compile_qwen_dense_ffn_fp8_aot
    g = QwenDenseGeometry(hidden=256, intermediate=256)
    program = compile_qwen_dense_ffn_fp8_aot(g, max_rows=64 if rows <= 64 else 128)
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    up, up_scale, restored_up = _weight(2 * g.intermediate, g.hidden)
    down, down_scale, restored_down = _weight(g.hidden, g.intermediate)
    out = torch.empty_like(x)
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    program.launch(x, up, up_scale, down, down_scale, out, scratch, scalars=[rows])
    gate, value = torch.nn.functional.linear(x, restored_up).chunk(2, -1)
    expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * value, restored_down)
    torch.testing.assert_close(out, expected, atol=0.0002, rtol=0.08)
    assert program.geometry["weight_block"] == [128, 128]
