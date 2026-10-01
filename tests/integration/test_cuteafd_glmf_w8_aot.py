"""GLM 5.3 Flash MLA / o / FFN programs over FP8-only weights (``fp8_only``) against
the BF16 + FP8 programs fed the dequantized weights as their BF16 copy: decode
programs bitwise at every row count, prefill programs bitwise in both routes
(W8A8 ``fp8_rows`` 1, W8A16 ``fp8_rows`` 0) above the BF16 skinny GEMV rows."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def P(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53_FLASH, glmf

        fn = {"producer": glmf.compile_glmf_mla_producer_aot, "o": glmf.compile_glmf_o_aot,
              "ffn": glmf.compile_glmf_ffn_aot}[kind]
        _PROGRAMS[key] = fn(GLM53_FLASH, **kw)
    return _PROGRAMS[key]


def _w(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    s = torch.rand((n // 128, k // 128), generator=gen, device="cuda") * 0.004 + 5e-4
    wd = (w8.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
    return w8, s, wd


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


CASES = [("decode", 64, r, 16) for r in (1, 16, 17, 64)] + \
        [("prefill", 4096, r, f) for r in (64, 600) for f in (0, 1)]


@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_glmf_ffn_w8(mode, cap, rows, fp8_rows):
    from b12x.integration.cuteafd import GLM53_FLASH as g

    i, h = 12288, g.hidden
    gu8, gus, gud = _w(2 * i, h, 1)
    dn8, dns, dnd = _w(h, i, 2)
    x = (torch.randn((rows, h), device="cuda")).bfloat16()
    old_p = P("ffn", inter=i, max_rows=cap, fp8=True if mode == "decode" else "prefill")
    new_p = P("ffn", inter=i, max_rows=cap, fp8_only=mode)
    old, new = torch.empty_like(x), torch.empty_like(x)
    old_p.launch(x, gud, gu8, gus, dnd, dn8, dns, old, _scratch(old_p, rows), scalars=(rows, fp8_rows))
    new_p.launch(x, gu8, gus, dn8, dns, new, _scratch(new_p, rows), scalars=(rows, fp8_rows))
    torch.cuda.synchronize()
    assert torch.equal(old, new)


@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_glmf_o_w8(mode, cap, rows, fp8_rows):
    from b12x.integration.cuteafd import GLM53_FLASH as g

    n, v, h = g.heads, g.v_head_dim, g.hidden
    o8, os_, od = _w(h, n * v, 3)
    w_uv = (torch.randn((n, v, g.kv_lora_rank), device="cuda") * 0.02).bfloat16()
    attn = (torch.randn((rows, n, g.kv_lora_rank), device="cuda") * 0.3).bfloat16()
    old_p = P("o", max_rows=cap, fp8=True if mode == "decode" else "prefill")
    new_p = P("o", max_rows=cap, fp8_only=mode)
    old = torch.empty((rows, h), dtype=torch.bfloat16, device="cuda")
    new = torch.empty_like(old)
    old_p.launch(attn, w_uv, od, o8, os_, old, _scratch(old_p, rows), scalars=(rows, fp8_rows))
    new_p.launch(attn, w_uv, o8, os_, new, _scratch(new_p, rows), scalars=(rows, fp8_rows))
    torch.cuda.synchronize()
    assert torch.equal(old, new)


@pytest.mark.parametrize("mode,cap,rows,fp8_rows", CASES)
def test_glmf_mla_producer_w8(mode, cap, rows, fp8_rows):
    from b12x.integration.cuteafd import GLM53_FLASH as g

    h, q, n = g.hidden, g.q_lora_rank, g.heads
    a8, as_, ad = _w(g.qkv_a_width, h, 4)
    b8, bs, bd = _w(n * g.qk_head_dim, q, 5)
    norms = [torch.rand(q, device="cuda").bfloat16() + 0.5, torch.rand(g.kv_lora_rank, device="cuda").bfloat16() + 0.5]
    w_uk = (torch.randn((n, g.kv_lora_rank, g.qk_nope_dim), device="cuda") * 0.02).bfloat16()
    x = torch.randn((rows, h), device="cuda").bfloat16()
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    pages = (rows + 63) // 64 + 1
    outs = []
    for kw, weights in ((dict(fp8=True if mode == "decode" else "prefill"), [ad, a8, as_, *norms, bd, b8, bs]),
                        (dict(fp8_only=mode), [a8, as_, *norms, b8, bs])):
        program = P("producer", max_rows=cap, **kw)
        cache = torch.zeros((pages, g.kv_page_bytes), dtype=torch.uint8, device="cuda")
        query = torch.empty((rows, n, g.latent_dim), dtype=torch.bfloat16, device="cuda")
        q_resid = torch.empty((rows, q), dtype=torch.bfloat16, device="cuda")
        program.launch(x, slots, *weights, w_uk, cache, query, q_resid, _scratch(program, rows),
                       scalars=(rows, fp8_rows))
        outs.append((query, q_resid, cache))
    torch.cuda.synchronize()
    for a, b in zip(*outs):
        assert torch.equal(a, b)
