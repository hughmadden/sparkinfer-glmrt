"""Shared MLA draft ABI and numerical visibility/rotary contracts."""
from __future__ import annotations

import inspect

import pytest
import torch

from b12x.integration.cuteafd import kimi_mla as mla


def geometry(**kwargs):
    # Shared loader supplies the checkpoint's mscale_all_dim-adjusted value.
    return mla.MlaDraftGeometry(attention_scale=192 ** -0.5 * 1.3465735902799727 ** 2, **kwargs)


def test_draft_geometry_is_explicit_and_target_unchanged():
    target = mla.KimiMlaGeometry()
    assert (target.heads, target.norm_eps, target.softmax_scale) == (96, 1e-6, 192 ** -0.5)
    with pytest.raises(TypeError):
        mla.MlaDraftGeometry()
    for args in ({"heads": 48}, {"norm_eps": 1e-6}, {"window_size": 2048}, {"attention_scale": float("nan")}):
        with pytest.raises(ValueError):
            mla.MlaDraftGeometry(**{"attention_scale": 0.1, **args})
    assert geometry().record_bytes == target.record_bytes == 1152


def test_draft_abi_and_capacity_not_live_counts(monkeypatch):
    monkeypatch.setattr(mla, "compile_program", lambda launch, **kwargs: (launch, kwargs))
    g = geometry()
    producer, p = mla.compile_mla_draft_producer_aot(g, max_rows=64)
    output, o = mla.compile_mla_draft_output_aot(g, max_rows=64)
    attention, a = mla.compile_mla_draft_attention_aot(g, max_rows=64, max_context=8192)
    assert p["geometry"]["rotary_style"] == "interleaved"
    assert [v.name for v in p["scalars"]] == ["rotary_factor", "rows"]
    assert [v.name for v in p["operands"]][2:4] == ["positions", "inv_freq"]
    assert not any(v.name in ("x", "w_g") for v in o["operands"])
    assert not hasattr(output, "gate_proj")
    assert o["geometry"]["output_gate"] is False
    assert a["geometry"]["noncausal_block"] is True
    assert attention.forward.noncausal_block is True
    assert attention.forward.window_size is None
    assert o["scratch"]["scratch"](9) == mla._align(9 * 64 * 128 * 2)
    for compile_fn in (mla.compile_mla_draft_producer_aot, mla.compile_mla_draft_output_aot,
                       mla.compile_mla_draft_attention_aot):
        assert "rows" not in inspect.signature(compile_fn).parameters
        with pytest.raises(TypeError):
            compile_fn(mla.KimiMlaGeometry())


@pytest.mark.parametrize("window", [None, 4096])
def test_draft_attention_gpu_ragged_noncausal_window_and_graph(window):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    g = geometry(window_size=window)
    p = mla.compile_mla_draft_attention_aot(g, max_rows=16, max_context=4352)
    counts, lengths = [8, 3], [4101, 65]
    width = 68
    generator = torch.Generator(device="cuda").manual_seed(273)
    cache = (torch.randn(2 * width, 64, 576, generator=generator, device="cuda") * 0.1).bfloat16()
    q = (torch.randn(11, 64, 576, generator=generator, device="cuda") * 0.1).bfloat16()
    table = torch.arange(2 * width - 1, -1, -1, dtype=torch.int32, device="cuda").reshape(width, 2).T.contiguous()
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 8, 11], dtype=torch.int32, device="cuda")
    out = torch.empty(11, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(11, 64, device="cuda")
    scratch = torch.empty(p.scratch_bytes(16)["scratch"], dtype=torch.uint8, device="cuda")

    def run():
        p.launch(q, cache, table, lens, cu, out, lse, scratch,
                 scalars=(11, 2, width, p.geometry["num_splits"]))

    run()
    torch.cuda.synchronize()
    for index, (count, length, begin) in enumerate(zip(counts, lengths, [0, 8])):
        records = cache[table[index].long()].flatten(0, 1)[:length].float()
        scores = torch.einsum("thd,sd->hts", q[begin:begin + count].float(), records) * g.softmax_scale
        if window:
            positions = torch.arange(length, device="cuda")
            low = length - count + torch.arange(count, device="cuda") - window + 1
            scores.masked_fill_(positions[None, None] < low[None, :, None], -torch.inf)
        expected = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
        torch.testing.assert_close(out[begin:begin + count].float(), expected, atol=0.003, rtol=0.03)
        torch.testing.assert_close(lse[begin:begin + count], scores.logsumexp(-1).T, atol=2e-4, rtol=2e-5)
    assert torch.isfinite(out).all() and out.abs().sum() > 0
    saved = out.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    for actual, expected in zip((out, lse), saved):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_rotary_gpu_interleaved_table_and_negative_slots():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    kernel = mla._DraftRotary(64)
    p = compile_program(kernel, name="test_mla_draft_rotary", operands=(
        Operand("query", torch.bfloat16, "[rows,64,576]", "inout"),
        Operand("cache", torch.bfloat16, "[pages,64,576]", "inout"),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("inv_freq", torch.float32, "[32]", align=4)),
        scalars=(Scalar("factor", "float32"), Scalar("rows")), key=(64,))
    q = torch.randn(4, 64, 576, device="cuda").bfloat16()
    cache = torch.randn(64, 576, device="cuda").bfloat16()
    slots = torch.tensor([0, 63, -1, 11], device="cuda", dtype=torch.int64)
    positions = torch.tensor([0, 32767, 32768, 1048575], device="cuda", dtype=torch.int64)
    # Deliberately non-default frequencies ensure the kernel consumes the loader table.
    inv = (1.0 / (50000.0 ** (torch.arange(32, device="cuda").float() / 32))) / 7
    before_q, before_cache = q.clone(), cache.clone()
    factor = 1.125
    p.launch(q, cache, slots, positions, inv, scalars=(factor, 4))
    phase = positions.float()[:, None] * inv[None]
    c, s = (phase.cos() * factor).bfloat16(), (phase.sin() * factor).bfloat16()

    def rotate(x, cos, sin):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x1 * cos + x0 * sin
        return out

    torch.testing.assert_close(q[..., 512:], rotate(before_q[..., 512:], c[:, None], s[:, None]), atol=0.015625, rtol=0.01)
    assert torch.equal(q[..., :512], before_q[..., :512])
    expected_cache = before_cache.clone()
    for row, slot in enumerate([0, 63, -1, 11]):
        if slot >= 0:
            expected_cache[slot, 512:] = rotate(before_cache[slot, 512:], c[row], s[row])
    torch.testing.assert_close(cache, expected_cache, atol=0.015625, rtol=0.01)
