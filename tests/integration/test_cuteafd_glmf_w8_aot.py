"""GLM 5.3 Flash MLA / o / FFN programs over FP8-only weights (``fp8_only``) against
the BF16 + FP8 programs fed the dequantized weights as their BF16 copy: decode
programs bitwise at every row count, prefill programs bitwise in both routes
(W8A8 ``fp8_rows`` 1, W8A16 ``fp8_rows`` 0) above the activation-quant row threshold."""

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


@pytest.mark.parametrize("n,threshold", [(4096, 8), (2048, 8), (1024, 64), (256, 160)])
def test_activation_quant_admission_is_independent_of_bf16_route(n, threshold):
    import cutlass
    import cutlass.cute as cute

    from b12x.gemm.bf16_gemv._skinny import skinny_max_rows
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._glmf_fp8 import quant_scratch_bytes
    from b12x.integration.cuteafd.glmf import _Fp8Switch

    k, cap = 4096, threshold + 1
    switch = _Fp8Switch(n, k, fp8=True, prefill_rows=cap, prefill_mask=2)
    assert switch.bf16.max_skinny_rows == 0
    assert switch.activation_quant_row_threshold == threshold == (8 if n >= 2048 else skinny_max_rows(n, k))
    key = switch.key()
    switch.activation_quant_row_threshold += 1
    assert switch.key() != key
    switch.activation_quant_row_threshold = threshold

    class Launch:
        @cute.jit
        def __call__(self, x, w, w8, scale, out, scratch, rows, mask, stream):
            switch.run(x, w, w8, scale, out, rows, mask, cutlass.Int64(scratch.toint()), stream)

    operands = (Operand("x", torch.bfloat16, "x"), Operand("w", torch.bfloat16, "w"),
                Operand("w8", torch.float8_e4m3fn, "w8"), Operand("scale", torch.float32, "scale"),
                Operand("out", torch.bfloat16, "out", "out"), Operand("scratch", torch.uint8, "scratch", "scratch"))
    program = compile_program(Launch(), name="activation_quant_admission", operands=operands,
                              scalars=(Scalar("rows"), Scalar("mask")), key=key)
    canonical = compile_program(switch.bf16, name="activation_quant_bf16", operands=operands[:2] + operands[4:5],
                                scalars=(Scalar("rows"),), key=switch.bf16.key())
    w8, scale, weight = _w(n, k, n)
    x = torch.randn(cap, k, device="cuda").bfloat16()
    scratch = torch.empty(quant_scratch_bytes(k, cap), device="cuda", dtype=torch.uint8)
    out = torch.empty(cap, n, device="cuda", dtype=torch.bfloat16)
    want = torch.empty_like(out)
    for rows in (1, threshold, cap):
        canonical.launch(x, weight, want, scalars=(rows,))
        for mask in (0, 1, 2, 3):
            scratch.fill_(165)
            program.launch(x, weight, w8, scale, out, scratch, scalars=(rows, mask))
            torch.cuda.synchronize()
            quantized = rows > threshold and bool(mask & 2)
            assert bool((scratch != 165).any()) == quantized, f"N={n}, M={rows}, mask={mask}"
            if not quantized:
                assert torch.equal(out[:rows].view(torch.int16), want[:rows].view(torch.int16))


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
