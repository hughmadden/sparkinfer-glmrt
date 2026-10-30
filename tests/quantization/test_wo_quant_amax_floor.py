"""Explicit K32 floor on the grouped WO-A quantizer (V4.1 act_quant before wo_a)."""
import pytest
import torch
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm, quantize_wo_a_input_mxfp8
from b12x.gemm.wo_projection._quant_cute import _get_compiled_wo_quant
from tests.conftest import require_b12x


def _grouped_quant(k, group_width, floor):
    return _get_compiled_wo_quant("grouped", k, group_width, torch.bfloat16, False, 0, 0, 0,
                                  torch.int64, torch.bfloat16, False, floor)


@pytest.mark.parametrize("floor", [0.0, 1e-4])
def test_amax_floor_values_scales_and_replay(floor):
    require_b12x()
    rows, groups, width = 16, 2, 256
    k = groups * width
    factors = torch.tensor([0, 1e-7, 1e-5, 1e-4, 1, 4, 128, .002], device="cuda").repeat(k // 32 // 8)
    x = (torch.linspace(-1, 1, 32, device="cuda")[None, None, :] *
         factors[None, :, None]).expand(rows, -1, -1).reshape(rows, k).bfloat16().contiguous()
    out = empty_mxfp8_rows_for_dense_gemm(rows, width, num_groups=groups, device="cuda")
    values, scales, mma = (out.values.view(torch.uint8), out.scale_rows.view(torch.uint8),
                           out.scale_mma.view(torch.uint8))
    run = _grouped_quant(k, width, floor)

    def check():
        blocks = x.float().reshape(rows, groups, width // 32, 32)
        maximum = blocks.abs().amax(-1).clamp_min(floor)
        expected_scales = torch.exp2(torch.ceil(torch.log2(maximum/448)))
        expected_scales = torch.where(maximum == 0, 1.0, expected_scales)
        expected = (blocks/expected_scales[..., None]).to(torch.float8_e4m3fn).reshape(rows, groups, width)
        # Logical [M,K,L] values, [L,M,K/32] row scales.
        assert torch.equal(values, expected.permute(0, 2, 1).view(torch.uint8))
        assert torch.equal(out.scale_rows.view(torch.float8_e8m0fnu).float(), expected_scales.permute(1, 0, 2))
        # Every live hardware MMA scale slot ([32,4,M/128,4,K/128,L]) repeats its row scale.
        r = torch.arange(rows, device="cuda")[:, None, None]
        c = torch.arange(width // 32, device="cuda")[None, :, None]
        g = torch.arange(groups, device="cuda")[None, None, :]
        assert torch.equal(mma[r % 32, r // 32 % 4, r // 128, c % 4, c // 4, g], scales.permute(1, 2, 0))
        if floor == 0.0:
            # Without a floor the kernel stays bit-exact with the routed Triton quantizer.
            reference = quantize_wo_a_input_mxfp8(x.reshape(rows, groups, width))
            assert torch.equal(values, reference.values.view(torch.uint8))
            assert torch.equal(scales, reference.scale_rows.view(torch.uint8))
            assert torch.equal(mma, reference.scale_mma.view(torch.uint8))

    run(x, x, x, values, scales, mma, rows)
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run(x, x, x, values, scales, mma, rows)
    x.mul_(-.25)
    graph.replay()
    check()
    torch.cuda.synchronize()
    graph.reset()


@pytest.mark.parametrize("floor", [-1.0, float("nan"), float("inf"), 2.0])
def test_invalid_floor(floor):
    with pytest.raises(ValueError, match="amax_floor"):
        _grouped_quant(512, 256, floor)
