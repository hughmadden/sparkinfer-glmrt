"""Shared GDN output gate: Flash Next sigmoid and dense Qwen SiLU rounding."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("heads,weight_dtype", [(32, torch.float32), (48, torch.bfloat16)])
@pytest.mark.parametrize("rows", [1, 17, 64])
def test_qwen_gdn_gated_norm(activation, heads, weight_dtype, rows):
    require_b12x()
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._qwen4_gdn_kernels import Qwen4GdnGatedNorm

    width, stride = heads * 128, heads * 128 + 256
    kernel = Qwen4GdnGatedNorm(heads=heads, eps=1e-6, gate_stride=stride, activation=activation)
    program = compile_program(kernel, name="qwen_gated_norm_test", operands=(
        Operand("x", torch.bfloat16, "[rows,width]"), Operand("z", torch.bfloat16, "[rows,stride]"),
        Operand("w", weight_dtype, "[128]"), Operand("out", torch.bfloat16, "[rows,width]")),
        scalars=(Scalar("rows"),), key=(activation, heads, str(weight_dtype), stride))
    gen = torch.Generator(device="cuda").manual_seed(276)
    x = torch.randn(rows, heads, 128, device="cuda", generator=gen).bfloat16()
    z = (4 * torch.randn(rows, stride, device="cuda", generator=gen)).bfloat16()
    w = (1 + 0.1 * torch.randn(128, device="cuda", generator=gen)).to(weight_dtype)
    out = torch.empty_like(x)
    program.launch(x, z, w, out, scalars=[rows])
    norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    weighted = w * norm
    gate = z[:, :width].reshape(rows, heads, 128).float()
    gate = torch.nn.functional.silu(gate) if activation == "silu" else gate.sigmoid()
    reference = (weighted.float() * gate).bfloat16()
    torch.testing.assert_close(out, reference, atol=0.03125, rtol=0.008)
    # Repeating a row at another position must not change its output.
    repeated = torch.empty_like(out)
    program.launch(x[:1].expand_as(x).contiguous(), z[:1].expand_as(z).contiguous(), w, repeated, scalars=[rows])
    assert torch.equal(repeated, out[:1].expand_as(out))


def test_qwen_gdn_rejects_unknown_gate():
    from b12x.integration.cuteafd._qwen4_gdn_kernels import Qwen4GdnGatedNorm
    with pytest.raises(ValueError, match="sigmoid or silu"):
        Qwen4GdnGatedNorm(heads=32, eps=1e-6, gate_stride=4096, activation="swish")
