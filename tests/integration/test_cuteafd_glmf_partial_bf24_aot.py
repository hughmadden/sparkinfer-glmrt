"""BF24 peer codec: raw-bit rounding, exceptional values and graph replay."""

import pytest
import torch

from ..conftest import require_b12x


def _reference_bits(x):
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = ((bits + 127 + ((bits >> 8) & 1)) >> 8) & 0xFFFFFF
    special = (bits & 0x7F800000) == 0x7F800000
    truncated = bits >> 8
    lost_nan = ((bits & 0x007FFFFF) != 0) & ((truncated & 0x007FFF) == 0)
    return torch.where(special, truncated | lost_nan.to(torch.int64), rounded)


def _reference_pack(x):
    bits = _reference_bits(x)
    return torch.stack([((bits >> shift) & 255).to(torch.uint8) for shift in (0, 8, 16)], dim=-1)


def _reference_decode(x):
    return (_reference_bits(x) << 8).to(torch.int32).view(torch.float32)


def _guarded(shape, dtype):
    elements = 1
    for size in shape:
        elements *= size
    nbytes = elements * torch.empty((), dtype=dtype).element_size()
    storage = torch.full((nbytes + 2048,), 0xA5, device="cuda", dtype=torch.uint8)
    return storage[1024:-1024].view(dtype).reshape(shape), storage


def _check_guard(storage):
    assert (storage[:1024] == 0xA5).all()
    assert (storage[-1024:] == 0xA5).all()


def _assert_sum(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(actual[finite].view(torch.int16), expected[finite].view(torch.int16))


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd._glmf_partial_bf24 import (
        compile_glmf_pack_bf24_aot, compile_glmf_add_bf24_aot,
    )
    return compile_glmf_pack_bf24_aot(), compile_glmf_add_bf24_aot()


@pytest.mark.parametrize("rows", (1, 22, 64, 512, 4096))
def test_bf24_bits_sum_and_graph_replay(programs, rows):
    pack, add = programs
    shape = (rows, 4096)
    a, guard_a = _guarded(shape, torch.float32)
    b, guard_b = _guarded(shape, torch.float32)
    pa, guard_pa = _guarded((*shape, 3), torch.uint8)
    pb, guard_pb = _guarded((*shape, 3), torch.uint8)
    out, guard_out = _guarded(shape, torch.bfloat16)
    # Real-range partials, cancellation and rounding ties at both signs.
    a.copy_(torch.randn(shape, device="cuda") * torch.exp2(torch.randint(-20, 20, shape, device="cuda")))
    b.copy_(torch.randn(shape, device="cuda") * torch.exp2(torch.randint(-20, 20, shape, device="cuda")))
    b.flatten()[128:256].copy_(-a.flatten()[128:256])
    special_bits = torch.tensor([
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x0000007F, 0x00000080, 0x00000180, 0x007FFFFF,
        0x00800000, 0x807FFFFF, 0x3F80007F, 0x3F800080,
        0x3F800081, 0x3F800180, 0xBF800080, 0xBF800180,
        0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
        0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001,
        0x7F800080, 0x7F800100, 0x7FFFFFFF, 0xFFFFFFFF,
    ], device="cuda", dtype=torch.int64).to(torch.int32).view(torch.float32)
    a.flatten()[:len(special_bits)].copy_(special_bits)
    b.flatten()[:len(special_bits)].copy_(special_bits.roll(7))

    pack.launch(a, pa, scalars=(rows,))
    pack.launch(b, pb, scalars=(rows,))
    add.launch(pa, pb, out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(pa, _reference_pack(a))
    assert torch.equal(pb, _reference_pack(b))
    expected = (_reference_decode(a) + _reference_decode(b)).bfloat16()
    _assert_sum(out, expected)
    original = out.clone()
    add.launch(pb, pa, out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), original.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack.launch(a, pa, scalars=(rows,))
        pack.launch(b, pb, scalars=(rows,))
        add.launch(pa, pb, out, scalars=(rows,))
    a.mul_(1.25)
    b.mul_(-0.75)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(pa, _reference_pack(a))
    assert torch.equal(pb, _reference_pack(b))
    _assert_sum(out, (_reference_decode(a) + _reference_decode(b)).bfloat16())
    for storage in (guard_a, guard_b, guard_pa, guard_pb, guard_out):
        _check_guard(storage)
