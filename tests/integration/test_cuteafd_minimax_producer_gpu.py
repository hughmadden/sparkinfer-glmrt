"""MiniMax producer block oracles; native helpers, live-size reuse and replay."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def _cos_sin(device):
    frequency = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    phase = torch.arange(257, device=device).float()[:, None] * frequency
    return torch.cat((phase.cos(), phase.sin()), -1).contiguous()


def _reference(x, weight, positions, cs):
    y = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
         * (1 + weight.float())).bfloat16().float()
    c, s = cs[positions, :32, None].transpose(1, 2), cs[positions, 32:, None].transpose(1, 2)
    a, b = y[..., :32], y[..., 32:64]
    return torch.cat(((a * c - b * s).bfloat16(), (b * c + a * s).bfloat16(),
                      y[..., 64:].bfloat16()), -1)


def _quantize(value):
    value = value.float()
    maximum = value.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 448.0, 1.0)
    inverse = 1.0 / scale
    return (value * inverse).clamp(-448, 448).to(torch.float8_e4m3fn), scale.squeeze(-1)


@pytest.mark.parametrize("heads", [1, 4, 64])
def test_norm_rope_matches_reference_and_reuses_live_counts(heads):
    require_b12x()
    from b12x.integration.cuteafd.minimax_producer import compile_minimax_norm_rope_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_minimax_norm_rope_aot(heads=heads)
    gen = torch.Generator(device="cuda").manual_seed(811)
    x = torch.randn((17, heads, 128), generator=gen, device="cuda").bfloat16()
    x[0].zero_()
    weight = (torch.randn(128, generator=gen, device="cuda") * 0.1).bfloat16()
    positions = torch.arange(17, device="cuda", dtype=torch.int64) * 13
    cs = _cos_sin("cuda")
    out, scratch = torch.empty_like(x), torch.empty_like(x)
    expected = _reference(x, weight, positions, cs)
    tensors = (x, weight, positions, cs, out, scratch)
    pointers = [t.data_ptr() for t in tensors]
    program.launch(*tensors, scalars=(17,))
    with kernel_resolution_guard("MiniMax producer live rows"):
        for rows in (1, 7, 17):
            out.fill_(float("nan"))
            program.launch(*tensors, scalars=(rows,))
            torch.testing.assert_close(out[:rows], expected[:rows], rtol=0.02, atol=0.016)
            assert torch.isnan(out[rows:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*tensors, scalars=(17,))
        x.neg_()
        graph.replay()
        torch.testing.assert_close(out, _reference(x, weight, positions, cs), rtol=0.02, atol=0.016)
    assert pointers == [t.data_ptr() for t in tensors]
    assert torch.count_nonzero(out[1:]) > 0
    assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("big_page", [False, True])
def test_index_producer_plain_scales_zero_rows_negative_slots_and_replay(big_page):
    require_b12x()
    from b12x.integration.cuteafd.minimax_producer import compile_minimax_index_producer_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_minimax_index_producer_aot()
    gen = torch.Generator(device="cuda").manual_seed(241)
    iq = torch.randn((7, 4, 128), generator=gen, device="cuda").bfloat16()
    ik = torch.randn((7, 1, 128), generator=gen, device="cuda").bfloat16()
    iq[0].zero_()
    ik[0].zero_()
    qn, kn = [(torch.randn(128, generator=gen, device="cuda") * 0.1).bfloat16() for _ in range(2)]
    positions = torch.arange(7, device="cuda", dtype=torch.int64) * 31
    cs = _cos_sin("cuda")
    page = 2**31 // 8448 + 3 if big_page else 3
    # Only the tail page is touched, despite the large virtual pool extent.
    cache = torch.empty((page + 1, 8448), device="cuda", dtype=torch.uint8)
    cache[page].fill_(173)
    slots = torch.tensor([page * 64 + i if i != 3 else -1 for i in range(7)],
                         device="cuda", dtype=torch.int64)
    qout = torch.empty_like(iq, dtype=torch.float8_e4m3fn)
    kout = torch.empty((7, 128), device="cuda", dtype=torch.float8_e4m3fn)
    qs = torch.empty((7, 4), device="cuda")
    ks = torch.empty(7, device="cuda")
    scratch = torch.empty(program.scratch_bytes(7)["scratch"] // 2, device="cuda", dtype=torch.bfloat16)
    tensors = (iq, ik, qn, kn, positions, slots, cs, qout, qs, kout, ks, cache, scratch)
    pointers = [t.data_ptr() for t in tensors]

    def check(rows):
        qref, qscale = _quantize(_reference(iq, qn, positions, cs))
        kref, kscale = _quantize(_reference(ik, kn, positions, cs).squeeze(1))
        # Norm reduction and FP32 RoPE can move BF16 halfway cases by one ULP.
        torch.testing.assert_close(qs[:rows], qscale[:rows], rtol=0.02, atol=1e-6)
        torch.testing.assert_close(ks[:rows], kscale[:rows], rtol=0.02, atol=1e-6)
        torch.testing.assert_close(qout[:rows].float() * qs[:rows, :, None],
            qref[:rows].float() * qscale[:rows, :, None], rtol=0.15, atol=0.04)
        torch.testing.assert_close(kout[:rows].float() * ks[:rows, None],
            kref[:rows].float() * kscale[:rows, None], rtol=0.15, atol=0.04)
        page_keys = cache[page, :8192].view(torch.float8_e4m3fn).reshape(64, 128)
        page_scales = cache[page, 8192:].view(torch.float32)
        for row in range(rows):
            if row != 3:
                assert torch.equal(page_keys[row].view(torch.uint8), kout[row].view(torch.uint8))
                assert torch.equal(page_scales[row], ks[row])
        assert (cache[page, 3 * 128:4 * 128] == 173).all()
        assert (cache[page, 8192 + 3 * 4:8192 + 4 * 4] == 173).all()
        assert (qs[0] == 1).all() and ks[0] == 1
        assert torch.count_nonzero(qout[0].view(torch.uint8)) == 0
        assert torch.count_nonzero(kout[0].view(torch.uint8)) == 0
        assert torch.count_nonzero(qout[1:rows].view(torch.uint8)) > 0

    program.launch(*tensors, scalars=(7,))
    check(7)
    with kernel_resolution_guard("MiniMax index producer live rows and replay"):
        for rows in (2, 7):
            program.launch(*tensors, scalars=(rows,))
            check(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*tensors, scalars=(7,))
        iq.neg_()
        ik.neg_()
        graph.replay()
        check(7)
    assert pointers == [t.data_ptr() for t in tensors]


@pytest.mark.parametrize("width", [3072, 12288])
def test_split_oai_uses_shared_native_activation(width):
    require_b12x()
    from b12x.integration.cuteafd.minimax_producer import compile_minimax_oai_split_aot

    program = compile_minimax_oai_split_aot(width=width)
    gate = torch.linspace(-12, 12, width * 3, device="cuda").reshape(3, width).bfloat16()
    up = torch.linspace(12, -12, width * 3, device="cuda").reshape(3, width).bfloat16()
    out = torch.empty_like(gate)
    program.launch(gate, up, out, scalars=(3,))
    g, u = gate.float().clamp(max=7), up.float().clamp(-7, 7)
    expected = (g * torch.sigmoid(1.702 * g) * (u + 1)).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.004)
    assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(gate, up, out, scalars=(3,))
    up.zero_()
    graph.replay()
    torch.testing.assert_close(out, (g * torch.sigmoid(1.702 * g)).bfloat16(), rtol=0.008, atol=0.004)
