"""MiMo V2 qkv producer, output projection and FFN over FP8-only weights (``fp8_only``)
against the BF16 (+ FP8) programs fed the dequantized weights: decode bitwise at
every row count (16-row GEMV, two-tile GEMV to 32, W8A16 TMA above), prefill W8A16
(``fp8_rows`` 0) bitwise above the BF16 skinny GEMV rows, W8A8 close."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._mimo import cos_sin

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _geometry(geo):
    from dataclasses import replace
    from b12x.integration.cuteafd import MIMO_V2_FLASH, MIMO_V26_PRO

    return {"flash": MIMO_V2_FLASH, "pro": MIMO_V26_PRO,
            "pro_tp2": replace(MIMO_V26_PRO, heads=MIMO_V26_PRO.heads // 2)}[geo]


def P(program, geo, **kw):
    key = (program, geo, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import mimo_attention, mimo_ffn

        fn = {"producer": mimo_attention.compile_mimo_producer_aot, "ffn": mimo_ffn.compile_mimo_ffn_aot,
              "o": mimo_attention.compile_mimo_o_aot}[program]
        _PROGRAMS[key] = fn(_geometry(geo), **kw)
    return _PROGRAMS[key]


def _w(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    s = torch.rand((n, k // 128), generator=gen, device="cuda") * 0.004 + 5e-4
    return w8, s, (w8.float() * s.repeat_interleave(128, 1)).bfloat16()


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


CASES = [("decode", 64, r, 32) for r in (1, 16, 17, 32, 33, 64)] + \
        [("prefill", 4096, r, f) for r in (64, 600) for f in (0, 1)]


def _cmp(new, old, mode, fp8_rows):
    if mode == "prefill" and fp8_rows:
        c = torch.nn.functional.cosine_similarity(new.float().flatten(), old.float().flatten(), dim=0).item()
        print(f"W8A8 cosine {c:.7f}")
        assert c > 0.999
    else:
        assert torch.equal(new, old)


@pytest.mark.parametrize("geo,kind", [("flash", "full"), ("flash", "swa"), ("pro", "full")])
@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_mimo_producer_w8(geo, kind, mode, cap, rows, fp8_rows):
    g = _geometry(geo)
    h, wq = g.hidden, g.qkv_width(kind)
    w8, s, wd = _w(wq, h, rows)
    x = torch.randn((rows, h), device="cuda").bfloat16()
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") + 7
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    table = cos_sin(int(positions.max()) + 1, kind)
    r = g.record_elems(kind)

    def run(program, *weights, scalars):
        cache = torch.zeros((rows, r), dtype=torch.bfloat16, device="cuda")
        query = torch.empty((rows, g.heads, g.qk_head_dim), dtype=torch.bfloat16, device="cuda")
        program.launch(x, positions, slots, table, *weights, cache, query, _scratch(program, rows), scalars=scalars)
        torch.cuda.synchronize()
        return torch.cat([query.view(rows, -1), cache], 1)

    if mode == "decode":
        old = run(P("producer", geo, kind=kind, max_rows=cap, fp8=True), wd, w8, s, scalars=(rows, fp8_rows))
        new = run(P("producer", geo, kind=kind, max_rows=cap, fp8_only="decode"), w8, s, scalars=(rows, fp8_rows))
    else:
        old = run(P("producer", geo, kind=kind, max_rows=cap), wd, scalars=(rows,))
        new = run(P("producer", geo, kind=kind, max_rows=cap, fp8_only="prefill"), w8, s.t().contiguous(),
                  scalars=(rows, fp8_rows))
    _cmp(new, old, mode, fp8_rows)


@pytest.mark.parametrize("geo", ["flash", "pro", "pro_tp2"])
@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_mimo_output_w8(geo, mode, cap, rows, fp8_rows):
    g = _geometry(geo)
    h, w = g.hidden, g.heads * g.v_head_dim
    w8, scale, wd = _w(h, w, rows)
    x = torch.randn((rows, w), device="cuda").bfloat16()
    old = torch.empty((rows, h), device="cuda", dtype=torch.bfloat16)
    new = torch.empty_like(old)
    op = P("o", geo, max_rows=cap, fp8=mode == "decode")
    np_ = P("o", geo, max_rows=cap, fp8_only=mode)
    assert "w_o" not in [operand.name for operand in np_.operands]
    if mode == "decode":
        op.launch(x, wd, w8, scale, old, scalars=(rows, fp8_rows))
    else:
        op.launch(x, wd, old, scalars=(rows,))
    np_.launch(x, w8, scale if mode == "decode" else scale.t().contiguous(), new,
               _scratch(np_, cap), scalars=(rows, fp8_rows))
    torch.cuda.synchronize()
    assert torch.isfinite(new).all() and torch.count_nonzero(new) > 0
    _cmp(new, old, mode, fp8_rows)


@pytest.mark.parametrize("geo", ["pro", "pro_tp2"])
@pytest.mark.parametrize("mode,cap,counts", [("decode", 64, (1, 16, 17, 32, 33, 64)),
                                          ("prefill", 4096, (64, 65, 1024, 4096))])
def test_mimo_output_w8_frozen_graph(geo, mode, cap, counts):
    """One compiled capacity handles live tails and replay with changed inputs.

    This checks the FP8 kernel against dequantized weights; it does not qualify
    quantizing the checkpoint's original BF16 output projections.
    """
    from b12x._lib.runtime_control import kernel_resolution_guard

    g = _geometry(geo)
    h, w = g.hidden, g.heads * g.v_head_dim
    w8, scale, wd = _w(h, w, 927)
    scales = scale if mode == "decode" else scale.t().contiguous()
    x = torch.empty((cap, w), device="cuda", dtype=torch.bfloat16)
    new = torch.empty((cap + 1, h), device="cuda", dtype=torch.bfloat16)
    old = torch.empty_like(new)
    np_ = P("o", geo, max_rows=cap, fp8_only=mode)
    op = P("o", geo, max_rows=cap, fp8=mode == "decode")
    scratch = _scratch(np_, cap)
    fp8_rows = 32 if mode == "decode" else 0
    ptrs = tuple(t.data_ptr() for t in (x, w8, scales, new, scratch))
    with kernel_resolution_guard("MiMo output projection live rows reuse one FP8-only capacity"):
        for rows in counts:
            x.normal_()
            x[rows:].fill_(float("nan"))
            new.fill_(123)
            np_.launch(x, w8, scales, new, scratch, scalars=(rows, fp8_rows))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                np_.launch(x, w8, scales, new, scratch, scalars=(rows, fp8_rows))
            for _ in range(2):
                x[:rows].normal_()
                new.fill_(123)
                graph.replay()
                if mode == "decode":
                    op.launch(x, wd, w8, scale, old, scalars=(rows, fp8_rows))
                else:
                    op.launch(x, wd, old, scalars=(rows,))
                torch.cuda.synchronize()
                assert torch.isfinite(new[:rows]).all() and torch.count_nonzero(new[:rows]) > 0
                assert torch.equal(new[:rows], old[:rows])
                assert torch.all(new[rows:] == 123)
                assert ptrs == tuple(t.data_ptr() for t in (x, w8, scales, new, scratch))


@pytest.mark.parametrize("geo", ["flash", "pro"])
@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_mimo_ffn_w8(geo, mode, cap, rows, fp8_rows):
    g = _geometry(geo)
    h, i = g.hidden, g.dense_inter
    gu8, gus, gud = _w(2 * i, h, 1)
    dn8, dns, dnd = _w(h, i, 2)
    x = torch.randn((rows, h), device="cuda").bfloat16()
    old, new = torch.empty_like(x), torch.empty_like(x)
    if mode == "decode":
        op = P("ffn", geo, max_rows=cap, fp8=True)
        op.launch(x, gud, gu8, gus, dnd, dn8, dns, old, _scratch(op, rows), scalars=(rows, fp8_rows))
        np_ = P("ffn", geo, max_rows=cap, fp8_only="decode")
        np_.launch(x, gu8, gus, dn8, dns, new, _scratch(np_, rows), scalars=(rows, fp8_rows))
    else:
        op = P("ffn", geo, max_rows=cap)
        op.launch(x, gud, dnd, old, _scratch(op, rows), scalars=(rows,))
        np_ = P("ffn", geo, max_rows=cap, fp8_only="prefill")
        np_.launch(x, gu8, gus.t().contiguous(), dn8, dns.t().contiguous(), new, _scratch(np_, rows),
                   scalars=(rows, fp8_rows))
    torch.cuda.synchronize()
    _cmp(new, old, mode, fp8_rows)
