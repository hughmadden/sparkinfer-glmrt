"""Combined replies match an independent block-codec oracle byte for byte."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def reference(x, format):
    rows, width = x.shape
    block = 32 if format == 1 else 16
    values = x.float().reshape(rows, -1, block)
    amax = values.abs().amax(-1)
    if format == 1:
        exponent = torch.ceil(torch.log2(amax.double() / 448)).clamp(-127, 127)
        exponent = torch.where(amax == 0, -127, exponent)
        scale = torch.exp2(exponent).float()
        codes = (exponent + 127).to(torch.uint8)
        packed = (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
        packed = torch.where((amax == 0).unsqueeze(-1), 0, packed).flatten(1)
        restored = packed.view(torch.float8_e4m3fn).float() * scale.repeat_interleave(32, -1)
    else:
        codes = (amax / 6).clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8)
        codes = torch.where(amax == 0, 0, codes.clamp_min(1))
        scale = codes.view(torch.float8_e4m3fn).float()
        normalized = values / torch.where(scale == 0, 1, scale).unsqueeze(-1)
        magnitude = normalized.abs()
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
        distance = (magnitude.unsqueeze(-1) - table).abs()
        # E2M1 halfway values round to the even code, including saturation.
        minimum = distance.amin(-1, keepdim=True)
        candidates = torch.arange(8, device=x.device).expand_as(distance)
        priority = candidates + (candidates % 2) * 8
        priority = torch.where(distance == minimum, priority, 32)
        code = priority.argmin(-1).to(torch.uint8)
        code |= normalized.signbit().to(torch.uint8) * 8
        code = torch.where((amax == 0).unsqueeze(-1), 0, code).flatten(1)
        packed = code[:, ::2] | (code[:, 1::2] << 4)
        restored = table[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)
        restored *= scale.repeat_interleave(16, -1)
    return torch.cat((packed, codes), -1), restored.bfloat16()


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot
    return (compile_expert_reply_aot(quantize=True),
            compile_expert_reply_aot(stage=True), compile_expert_reply_aot())


@pytest.mark.parametrize("width", [32, 128, 4096, 5120, 7168])
@pytest.mark.parametrize("format", [1, 2])
def test_reply_pair_exact_and_graph_dynamic_mode(programs, width, format):
    quantize, stage, decode = programs
    torch.manual_seed(width + format)
    for rows in [1, 65, 129]:
        x = (torch.randn(rows, width, device="cuda") * .3).bfloat16()
        x[0] = 0
        if rows > 1:
            x[1] *= 1e-5
        expected_wire, expected = reference(x, format)
        packed = torch.full_like(expected_wire, 0xA5)
        scratch = torch.full_like(expected_wire, 0xA5)
        out = torch.empty_like(x)
        mode = torch.tensor([format], device="cuda", dtype=torch.uint32)
        quantize.launch(x, packed, None, scalars=(rows, width, format))
        stage.launch(packed, scratch, mode, scalars=(rows, width, -1))
        decode.launch(scratch, out, mode, scalars=(rows, width, -1))
        torch.cuda.synchronize()
        assert torch.equal(packed, expected_wire)
        assert torch.equal(scratch, expected_wire)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        # The actual consumer expands into the BF16-sized landing plane only
        # after a separate launch staged the complete compressed source.
        landing = torch.full((rows * width * 2 + 512,), 0xA5, device="cuda", dtype=torch.uint8)
        plane = landing[256:-256]
        plane[:packed.numel()].copy_(packed.flatten())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage.launch(plane, scratch, mode, scalars=(rows, width, -1))
            decode.launch(scratch, plane, mode, scalars=(rows, width, -1))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(plane.view(torch.bfloat16).view_as(x).view(torch.int16), expected.view(torch.int16))
        assert (landing[:256] == 0xA5).all() and (landing[-256:] == 0xA5).all()
        # A mixed/decode wave switches the same captured programs to no-op.
        mode.zero_()
        plane.copy_(x.view(torch.uint8).flatten())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(plane, x.view(torch.uint8).flatten())
