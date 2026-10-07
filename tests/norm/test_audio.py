"""Static contracts and numerical oracles for the FP32 audio support path."""
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x


def test_static_query_registration_and_live_counts_rejected():
    from b12x.norm._audio_preparation import AudioQuery, TUNING, plan
    from b12x.preparation.catalog import get_tuning_contract
    assert get_tuning_contract("norm.audio") is TUNING
    assert {field.name for field in fields(AudioQuery)} == {"operation", "width", "kernel", "stride"}
    query = AudioQuery(operation="im2col", width=384, kernel=3)
    assert TUNING.encode_query(query) == {"operation": "im2col", "width": 384, "kernel": 3, "stride": 1}
    with pytest.raises(ValueError, match="runtime inputs"):
        plan(query, device="cuda:0", invocation={"rows": 1})
    with pytest.raises(ValueError, match="CUDA"):
        plan(query, device="cpu")


@pytest.mark.parametrize("kwargs", [
    {"operation": "unknown", "width": 1024},
    {"operation": "layer_norm", "width": 0},
    {"operation": "layer_norm", "width": True},
    {"operation": "layer_norm", "width": 128},
    {"operation": "rope_pack", "width": 64},
    {"operation": "frame", "width": 1024},
    {"operation": "magnitude", "width": 960},
    {"operation": "gelu", "width": 4096},
    {"operation": "im2col", "width": 384, "stride": 2},
    {"operation": "bias", "width": 1024, "kernel": 3},
])
def test_unsupported_static_geometry_fails_closed(kwargs):
    from b12x.norm._audio_preparation import AudioQuery
    with pytest.raises(ValueError):
        AudioQuery(**kwargs)


def launcher(operation, width, device, kernel=1, stride=1):
    from b12x.norm.audio import _compile
    from b12x._lib.compiler import run_compiled
    from b12x._lib.utils import current_cuda_stream, make_ptr
    import cutlass
    import cutlass.cute as cute
    compiled, types = _compile(operation, width, kernel, stride, device.index)

    def run(x, out, rows, *, weight=None, bias=None, aux=None, codes=None,
            length=1, offset=0, parameter=1.0):
        buffers = (x, weight, bias, out, aux, codes)
        pointers = tuple(make_ptr(dtype, (value if value is not None else out).data_ptr(),
            cute.AddressSpace.gmem, assumed_align=4) for dtype, value in zip(types, buffers, strict=True))
        run_compiled(compiled, (*pointers, cutlass.Int32(rows), cutlass.Int32(length),
            cutlass.Int32(offset), cutlass.Float32(parameter), current_cuda_stream()))
    return run


def test_norm_live_counts_poison_and_graph_replay():
    device = require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    x = torch.randn(7, 1024, device=device)
    weight, bias = torch.randn(2, 1024, device=device).unbind()
    out = torch.empty_like(x)
    run = launcher("layer_norm", 1024, device)
    run(x, out, 7, weight=weight, bias=bias, parameter=1e-5)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("audio support live rows reuse one callable"):
        for rows in (1, 3, 7):
            out.fill_(float("nan"))
            run(x, out, rows, weight=weight, bias=bias, parameter=1e-5)
            torch.testing.assert_close(out[:rows], F.layer_norm(x[:rows], (1024,), weight, bias), atol=2e-6, rtol=3e-6)
            assert torch.isnan(out[rows:]).all()
        with torch.cuda.graph(graph):
            run(x, out, 3, weight=weight, bias=bias, parameter=1e-5)
        x.mul_(2)
        graph.replay()
        torch.testing.assert_close(out[:3], F.layer_norm(x[:3], (1024,), weight, bias), atol=2e-6, rtol=3e-6)


@pytest.mark.parametrize("window", [-1, 0, 2])
def test_group_attention_masks_and_dynamic_lengths(window):
    device = require_b12x()
    run = launcher("softmax", 1024, device)
    for length in (1, 4, 17):
        x = torch.randn(2 * length, length, device=device)
        out = torch.empty_like(x)
        run(x, out, 2 * length, length=length, offset=window, parameter=0.125)
        pos = torch.arange(length, device=device)
        mask = torch.ones(length, length, device=device, dtype=torch.bool)
        if window >= 0:
            mask = pos[:, None] >= pos[None, :]
            if window:
                mask &= pos[:, None] - pos[None, :] <= window
        oracle = (x.reshape(2, length, length) * 0.125).masked_fill(~mask, -float("inf")).softmax(-1)
        torch.testing.assert_close(out.reshape_as(oracle), oracle, atol=2e-7, rtol=2e-6)


def test_channel_major_convolution_odd_zero_padding():
    device = require_b12x()
    for channels, kernel, stride, pad in ((128, 3, 1, 1), (1024, 3, 2, 1), (1024, 2, 2, 0)):
        x = torch.arange(5 * channels, device=device, dtype=torch.float32).reshape(5, channels)
        source = F.pad(x.T[None, :, None], (pad, pad + (1 if kernel == 2 else 0)))
        oracle = F.unfold(source, (1, kernel), stride=(1, stride))[0].T.contiguous()
        out = torch.empty_like(oracle)
        run = launcher("im2col", channels * kernel, device, kernel, stride)
        run(x, out, out.shape[0], length=5, offset=pad)
        torch.testing.assert_close(out, oracle, atol=0, rtol=0)


def test_first_tie_rvq_fp32_residual_and_repeat_last_speech():
    device = require_b12x()
    dot = torch.zeros(3, 4, device=device)
    books = torch.randn(4, 1024, device=device)
    books[1].copy_(books[0])
    norms = torch.ones(4, device=device)
    square = torch.ones(3, device=device)
    residual = torch.randn(3, 1024, device=device)
    expected = residual - books[0]
    codes = torch.full((3, 20), -1, device=device, dtype=torch.int32)
    run = launcher("rvq_select", 1024, device)
    run(dot, residual, 3, weight=books, bias=square, aux=norms, codes=codes, length=4, offset=19)
    torch.testing.assert_close(residual, expected, atol=0, rtol=0)
    assert (codes[:, 19] == 0).all()
    assert (codes[:, :19] == -1).all()
    codes[:, 19] = torch.arange(3, device=device, dtype=torch.int32)
    out = torch.zeros(4, 1024, device=device)
    run = launcher("speech_add", 1024, device)
    run(None, out, 4, weight=books, codes=codes, length=3, offset=19)
    torch.testing.assert_close(out, books[torch.tensor([0, 1, 2, 2], device=device)], atol=0, rtol=0)


def test_reflect_frames_zero_capacity_and_head_layouts():
    device = require_b12x()
    x = torch.randn(481, device=device)
    hann = torch.hann_window(960, device=device)
    out = torch.full((5, 960), float("nan"), device=device)
    run = launcher("frame", 960, device)
    run(x, out, 5, weight=hann, length=x.numel())
    oracle = F.pad(x[None], (480, 480), mode="reflect").unfold(-1, 960, 240)[0] * hann
    torch.testing.assert_close(out[:3], oracle, atol=0, rtol=0)
    assert (out[3:] == 0).all()
    x = torch.randn(8, 1024, device=device)
    packed, restored = torch.empty_like(x), torch.empty_like(x)
    run = launcher("pack_heads", 1024, device)
    run(x, packed, 8, length=4)
    run = launcher("unpack_heads", 1024, device)
    run(packed, restored, 8, length=4)
    torch.testing.assert_close(restored, x, atol=0, rtol=0)
    angles = torch.randn(4, 32, device=device)
    rotary = torch.cat((angles.cos(), angles.sin()), -1)
    run = launcher("rope_pack", 1024, device)
    run(x, packed, 8, weight=rotary, length=4)
    halves = x.reshape(2, 4, 16, 64).chunk(2, -1)
    cosine, sine = angles.cos()[None, :, None], angles.sin()[None, :, None]
    expected = torch.cat((halves[0] * cosine - halves[1] * sine,
                          halves[1] * cosine + halves[0] * sine), -1).transpose(1, 2).contiguous()
    torch.testing.assert_close(packed.reshape_as(expected), expected, atol=3e-7, rtol=3e-6)
