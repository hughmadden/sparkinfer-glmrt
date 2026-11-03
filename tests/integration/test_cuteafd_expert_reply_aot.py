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
        # 448 = .875 * 2**9. frexp is exact at power-of-two boundaries;
        # CUDA log2 can round just above an integer and make ceil add a bit.
        mantissa, binary_exponent = torch.frexp(amax.double())
        exponent = (binary_exponent - 9 + (mantissa > .875)).clamp(-127, 127)
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


def mismatch_details(x, actual, expected, format):
    width = x.shape[1]
    block = 32 if format == 1 else 16
    value_bytes = width if format == 1 else width // 2
    changed_values = (actual[:, :value_bytes] != expected[:, :value_bytes]).reshape(
        x.shape[0], -1, block if format == 1 else block // 2).sum(-1)
    changed_scales = actual[:, value_bytes:] != expected[:, value_bytes:]
    blocks = ((changed_values != 0) | changed_scales).nonzero()
    details = []
    for row, index in blocks[:16].tolist():
        amax = float(x[row, index * block:(index + 1) * block].float().abs().max())
        import math
        details.append(dict(row=row, block=index, changed_values=int(changed_values[row, index]),
            actual_scale=int(actual[row, value_bytes + index]),
            expected_scale=int(expected[row, value_bytes + index]),
            amax_hex=amax.hex(), ratio_448_hex=(amax / 448).hex(),
            fp32_subnormal=0 < amax < 2.0 ** -126, zero_amax=amax == 0,
            cpu_exponent=max(-127, min(127, math.ceil(math.log2(amax / 448)))) if amax else -127))
    result = dict(changed_blocks=len(blocks), changed_scales=int(changed_scales.sum()), blocks=details)
    if len(blocks):
        import json
        print("reply_codec_mismatch=" + json.dumps(result, sort_keys=True))
    return result


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot
    return (compile_expert_reply_aot(quantize=True),
            compile_expert_reply_aot(stage=True), compile_expert_reply_aot())


def test_fp8_reference_exact_power_of_two_boundary(programs):
    quantize, stage, decode = programs
    boundary = torch.tensor(float.fromhex("0x1.cp-18"), device="cuda", dtype=torch.bfloat16)
    above = torch.nextafter(boundary, torch.full_like(boundary, float("inf")))
    x = torch.stack((boundary, above, torch.zeros_like(boundary))).unsqueeze(1).expand(-1, 32).contiguous()
    wire, restored = reference(x, 1)
    assert wire[:, -1].tolist() == [101, 102, 0]
    packed, scratch, output = torch.empty_like(wire), torch.empty_like(wire), torch.empty_like(x)
    mode = torch.tensor([1], device="cuda", dtype=torch.uint32)
    quantize.launch(x, packed, None, scalars=(3, 32, 1))
    stage.launch(packed, scratch, mode, scalars=(3, 32, -1))
    decode.launch(scratch, output, mode, scalars=(3, 32, -1))
    torch.cuda.synchronize()
    assert torch.equal(packed, wire), mismatch_details(x, packed, wire, 1)
    assert torch.equal(output.view(torch.int16), restored.view(torch.int16))


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
        mismatch = (packed != expected_wire).nonzero()
        samples = [(r, c, int(packed[r, c]), int(expected_wire[r, c]),
                    float(x[r, min(c, width - 1)])) for r, c in mismatch[:16].tolist()]
        assert not mismatch.numel(), (rows, width, format, samples,
                                      mismatch_details(x, packed, expected_wire, format))
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


@pytest.mark.parametrize("format", [1, 2])
def test_reply_slot_scratch_is_independent_across_consumer_streams(programs, format):
    quantize, stage, decode = programs
    rows, width = 129, 4096
    torch.manual_seed(100 + format)
    inputs = [(torch.randn(rows, width, device="cuda") * scale).bfloat16()
              for scale in [.03, 3.]]
    expected = [reference(x, format) for x in inputs]
    packed = [torch.empty_like(wire) for wire, _ in expected]
    outputs = [torch.empty_like(x) for x in inputs]
    scratch = torch.empty((2, *packed[0].shape), device="cuda", dtype=torch.uint8)
    mode = torch.tensor([format], device="cuda", dtype=torch.uint32)
    for x, wire in zip(inputs, packed):
        quantize.launch(x, wire, None, scalars=(rows, width, format))
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(32):
        index = iteration % 2
        stream = streams[index]
        with torch.cuda.stream(stream):
            stage.launch(packed[index], scratch[index], mode, scalars=(rows, width, -1))
            decode.launch(scratch[index], outputs[index], mode, scalars=(rows, width, -1))
    torch.cuda.synchronize()
    for actual, (_, restored) in zip(outputs, expected):
        assert torch.equal(actual.view(torch.int16), restored.view(torch.int16))


@pytest.mark.parametrize("format", [1, 2])
def test_reply_chunking_is_byte_identical_above_arena_capacity(programs, format):
    quantize, _, _ = programs
    rows, width, capacity = 4161, 128, 4096
    torch.manual_seed(format)
    x = (torch.randn(rows, width, device="cuda") * .3).bfloat16()
    # Distinct scale regimes immediately before and after the arena boundary.
    x[capacity - 1] *= 1e-5
    x[capacity] = 0
    x[capacity + 1] *= 128
    expected, _ = reference(x, format)
    whole = torch.full_like(expected, 0xA5)
    chunked = torch.full_like(expected, 0x5A)
    quantize.launch(x, whole, None, scalars=(rows, width, format))
    for first in range(0, rows, capacity):
        count = min(capacity, rows - first)
        quantize.launch(x[first:first + count], chunked[first:first + count], None,
                        scalars=(count, width, format))
    torch.cuda.synchronize()
    assert torch.equal(chunked, whole)
    assert torch.equal(whole, expected), mismatch_details(x, whole, expected, format)
