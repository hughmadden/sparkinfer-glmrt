"""GLM 5.3 Flash KDA over FP8-only ``w_in``/``w_o`` (``fp8_only``: E4M3 + per-row x 128-K
scales K-block major, no BF16 copy) and the single-copy FP8 LM head in 16-row spans.

KDA, against the dual-copy programs (``fp8=True`` / ``fp8="prefill"``) given the same E4M3
copy and, as their BF16 weight, either the dequantized FP8 weight (the W8A16 reference) or
the original BF16 weight (what the dual-copy mode ran above 16 rows):

- decode rows <= 16: bitwise equal to the dual-copy FP8 GEMV path (output, recurrent and
  conv state, replay record), whatever the BF16 copy holds;
- decode rows 17-64 and prefill: within BF16 rounding of the W8A16 reference (bitwise where
  both run the same GEMM: prefill W8A8 above the skinny rows), and the deviation from the
  original BF16 weights is reported.

Head: ``glmf_head_fp8`` over any row count in 16-row spans is bitwise one launch per span
and per row, and close to the FP32 reference over the dequantized weight."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def P(geometry=None, **kw):
    key = (repr(geometry), tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53_FLASH, glmf

        _PROGRAMS[key] = glmf.compile_glmf_kda_aot(GLM53_FLASH if geometry is None else geometry, **kw)
    return _PROGRAMS[key]


def _quant_rows(w: torch.Tensor):
    """Per-row x 128-K E4M3 quantization (amax / 448), the loader's ``Row128`` layout."""
    n, k = w.shape
    blocks = w.float().view(n, k // 128, 128)
    s = (blocks.abs().amax(-1) / 448.0).clamp_min(1e-12)
    w8 = (blocks / s[..., None]).to(torch.float8_e4m3fn).view(n, k)
    wd = (w8.float().view(n, k // 128, 128) * s[..., None]).view(n, k).bfloat16()
    return w8, s.contiguous(), s.t().contiguous(), wd


_WEIGHTS: dict = {}


def _weights(geometry=None):
    from b12x.integration.cuteafd import GLM53_FLASH
    g = GLM53_FLASH if geometry is None else geometry
    if g.kda_heads in _WEIGHTS:
        return _WEIGHTS[g.kda_heads]

    gen = torch.Generator(device="cuda").manual_seed(7)
    h, d, p = g.hidden, g.kda_width, g.kda_in_width
    r = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device="cuda") * s  # noqa: E731
    w_in = r(p, h, s=0.02).bfloat16()
    w_o = r(h, d, s=0.01).bfloat16()
    weights = dict(
        w_in=w_in, w_o=w_o, in_q=_quant_rows(w_in), o_q=_quant_rows(w_o),
        w_fg=r(2, d, g.kda_head_dim, s=0.05).bfloat16(), conv_w=r(3 * d, 4, s=0.3).float(),
        a_log=torch.log(torch.rand((g.kda_heads,), generator=gen, device="cuda") * 15 + 1).float(),
        dt_bias=r(d, s=0.1).float(), o_norm=(1 + r(g.kda_head_dim, s=0.1)).bfloat16())
    _WEIGHTS[g.kda_heads] = weights
    return weights


def _state(rows: int, per_seq: int, geometry=None):
    from b12x.integration.cuteafd import GLM53_FLASH
    g = GLM53_FLASH if geometry is None else geometry
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + per_seq)
    d = g.kda_width
    seqs = -(-rows // per_seq)
    slots = (torch.arange(rows, device="cuda", dtype=torch.int32) // per_seq).contiguous()
    first = (slots * per_seq).to(torch.int32).contiguous()
    state = torch.randn((seqs, g.kda_heads, 128, 128), generator=gen, device="cuda") * 0.01
    conv = (torch.randn((seqs, 3, 3 * d), generator=gen, device="cuda") * 0.1).bfloat16()
    replay = torch.zeros((kda_replay_layout(g.kda_heads, 3 * d)[2] // 4,), device="cuda")
    x = (torch.randn((rows, g.hidden), generator=gen, device="cuda")).bfloat16()
    return x, slots, first, state, conv, replay


def _padded(rows: int, pad: int):
    """``_state(rows, rows)`` (one sequence in slot 0) followed by ``pad - rows`` rows of a
    second sequence in slot 1: per-row projections and slot 0's recurrence are unchanged."""
    x, slots, first, state, conv, replay = _state(rows, rows)
    ex, _, _, es, ec, _ = _state(pad - rows, pad - rows)
    extra = torch.ones((pad - rows,), dtype=torch.int32, device="cuda")
    return (torch.cat([x, ex]).contiguous(), torch.cat([slots, extra]).contiguous(),
            torch.cat([first, extra * rows]).contiguous(), torch.cat([state, es]).contiguous(),
            torch.cat([conv, ec]).contiguous(), replay)


def _run(mode: str, cap: int, rows: int, fp8_rows: int, spec: int, bf16_in, bf16_o, per_seq: int = 4,
         old_pad: int = 0):
    """(old, new) outputs: out, state, conv_state (+ replay for decode). ``old_pad``: the old
    program runs ``old_pad`` rows (see ``_padded``; ``per_seq`` must be ``rows``) and returns the
    first sequence's, so a prefill reference past the skinny-GEMV rows covers fewer rows."""
    w = _weights()
    in8, in_s, in_k, _ = w["in_q"]
    o8, o_s, o_k, _ = w["o_q"]
    decode = mode == "decode"
    results = []
    for which in ("old", "new"):
        pad = old_pad if which == "old" and old_pad else 0
        x, slots, first, state, conv, replay = _padded(rows, pad) if pad else _state(rows, per_seq)
        n = pad or rows
        out = torch.empty((n, x.shape[1]), dtype=torch.bfloat16, device="cuda")
        prog = P(max_rows=cap, fp8=True if decode else "prefill") if which == "old" else \
            P(max_rows=cap, fp8_only=mode)
        scratch = torch.empty(prog.scratch_bytes(n)["scratch"], dtype=torch.uint8, device="cuda")
        shared = (w["w_fg"], w["conv_w"], w["a_log"], w["dt_bias"], w["o_norm"])
        if which == "old":
            ins = (bf16_in, in8, in_s if decode else in_k)
            outs = (bf16_o, o8, o_s if decode else o_k)
        else:
            ins = (in8, in_k)
            outs = (o8, o_k)
        tail = (conv, state, slots, first, out) + ((replay,) if decode else ()) + (scratch,)
        scalars = (n, fp8_rows, spec) if decode else (n, fp8_rows)
        prog.launch(x, *ins, *shared, *outs, *tail, scalars=scalars)
        torch.cuda.synchronize()
        if pad:
            out, state, conv = out[:rows], state[:1], conv[:1]
        results.append((out, state, conv) + ((replay,) if decode else ()))
    return results


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize("fp8_rows", (32, 0))
@pytest.mark.parametrize("heads,dtype,kind", ((64, "bfloat16", "projection"),
    (32, "bfloat16", "projection"), (32, "float32", "projection"), (32, "bfloat16", "norm")))
def test_glmf_kda_w8_bf16_recipe_cross_width_bits(fp8_rows, heads, dtype, kind):
    """Compare the new W8/S16 recipe with itself, not today's FP32 state.

    The canonical option fixes W8A16 at every width even when the runtime
    supplies the legacy GEMV threshold of 16. Every row has its own recurrent slot so widening
    a batch cannot legitimately alter its projection, conv or recurrent state.
    """
    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    from dataclasses import replace
    g = replace(g, kda_heads=heads)
    output_width = g.kda_width if kind == "norm" else g.hidden
    width = 128
    x, slots, first, state, conv, _ = _state(width, 1, g)
    state = state.bfloat16()
    weights = _weights(g)
    shared = (weights["w_fg"], weights["conv_w"], weights["a_log"],
              weights["dt_bias"], weights["o_norm"])
    ins, outs = weights["in_q"], weights["o_q"]

    def run(cap, rows, start=0, spec=0):
        prog = P(g, max_rows=cap, fp8_only="decode", state_dtype="bfloat16",
                 decode_recipe="w8a16", output_dtype=dtype, output_kind=kind)
        ss, cc = state[start:start + rows].clone(), conv[start:start + rows].clone()
        local_slots = torch.arange(rows, device="cuda", dtype=torch.int32)
        out = torch.full((rows, output_width), float("nan"), device="cuda", dtype=getattr(torch, dtype))
        replay_rows = max(64, cap)
        replay = torch.zeros(kda_replay_layout(g.kda_heads, 3 * g.kda_width, replay_rows)[2] // 4,
                             device="cuda")
        size = prog.scratch_bytes(rows)["scratch"]
        storage = torch.full((size + 1024,), 0xA5, device="cuda", dtype=torch.uint8)
        prog.launch(x[start:start + rows], ins[0], ins[2], *shared, outs[0], outs[2],
                    cc, ss, local_slots, local_slots, out, replay, storage[:size],
                    scalars=(rows, fp8_rows, spec))
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all())
        assert bool((storage[size:] == 0xA5).all())
        if spec:
            assert torch.equal(ss.view(torch.int16), state[start:start + rows].view(torch.int16))
            assert torch.equal(cc.view(torch.int16), conv[start:start + rows].view(torch.int16))
        return out, ss, cc

    serial = [run(64, 1, start=i) for i in range(width)]
    reference = tuple(torch.cat([row[k] for row in serial]) for k in range(3))
    failures = []
    for cap in (64, 128):
        for rows in (1, 2, 7, 8, 16, 17, 32, 33, 63, 64, 65, 96, 127, 128):
            if rows > cap:
                continue
            got = run(cap, rows)
            for name, actual, expected in zip(("out", "state", "conv"), got, reference):
                bad = int((actual.view(torch.int16) != expected[:rows].view(torch.int16)).sum())
                if bad:
                    failures.append(f"cap={cap} M={rows} {name}: {bad} differing BF16 words")
            verify = run(cap, rows, spec=1)[0]
            if not torch.equal(verify.view(torch.int16), got[0].view(torch.int16)):
                failures.append(f"cap={cap} M={rows}: verify output differs from plain step")
    print(f"W8/S16 fp8_rows={fp8_rows}: " + ("; ".join(failures) if failures else "all widths byte-identical"))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("heads,dtype,kind", ((64, "bfloat16", "projection"),
    (32, "bfloat16", "projection"), (32, "float32", "projection"), (32, "bfloat16", "norm")))
def test_glmf_kda_w8_bf16_canonical_sequence_commit_and_graph_bits(heads, dtype, kind):
    from b12x.integration.cuteafd import GLM53_FLASH as g, glmf
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    from dataclasses import replace
    g = replace(g, kda_heads=heads)
    output_width = g.kda_width if kind == "norm" else g.hidden
    rows, per_seq = 128, 8
    x, slots, first, initial, conv, _ = _state(rows, per_seq, g)
    initial = initial.bfloat16()
    weights = _weights(g)
    ins, outs = weights["in_q"], weights["o_q"]
    shared = (weights["w_fg"], weights["conv_w"], weights["a_log"],
              weights["dt_bias"], weights["o_norm"])
    prog = P(g, max_rows=128, fp8_only="decode", state_dtype="bfloat16",
             decode_recipe="w8a16", output_dtype=dtype, output_kind=kind)
    seqs = rows // per_seq

    def buffers():
        size = prog.scratch_bytes(rows)["scratch"]
        return (initial.clone(), conv.clone(),
                torch.full((rows, output_width), float("nan"), device="cuda", dtype=getattr(torch, dtype)),
                torch.zeros(kda_replay_layout(g.kda_heads, 3 * g.kda_width, rows)[2] // 4, device="cuda"),
                torch.full((size + 1024,), 0xA5, device="cuda", dtype=torch.uint8))

    def launch(activation, buffers, n=rows, at=0, spec=0, serial=False):
        ss, cc, out, replay, storage = buffers
        local_first = first[at:at + n] if not serial else torch.zeros_like(first[at:at + n])
        prog.launch(activation[at:at + n], ins[0], ins[2], *shared, outs[0], outs[2],
                    cc, ss, slots[at:at + n], local_first, out[at:at + n], replay, storage[:-1024],
                    scalars=(n, 0, spec))

    def same(actual, expected, name):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), name

    serial = buffers()
    after = []
    for r in range(rows):
        launch(x, serial, n=1, at=r, serial=True)
        after.append((serial[0][r // per_seq].clone(), serial[1][r // per_seq].clone()))
    torch.cuda.synchronize()
    wide = buffers()
    launch(x, wide)
    torch.cuda.synchronize()
    for name, a, b in zip(("state", "conv", "output"), wide, serial):
        same(a, b, name)
    assert bool(torch.isfinite(wide[2]).all())
    assert bool((wide[4][-1024:] == 0xA5).all())

    verify = buffers()
    launch(x, verify, spec=1)
    torch.cuda.synchronize()
    same(verify[0], initial, "verify state mutated")
    same(verify[1], conv, "verify conv mutated")
    same(verify[2], serial[2], "verify output")
    commit = glmf.compile_glmf_kda_commit_aot(g, state_dtype="bfloat16", replay_rows=rows)
    for keep in (1, 7, 8):
        ss, cc = initial.clone(), conv.clone()
        tables = torch.stack((torch.arange(seqs, device="cuda", dtype=torch.int32),
                              torch.arange(seqs, device="cuda", dtype=torch.int32) * per_seq,
                              torch.full((seqs,), keep, device="cuda", dtype=torch.int32)))
        commit.launch(ss, cc, verify[3], tables, scalars=(seqs, 1, seqs))
        torch.cuda.synchronize()
        for seq in range(seqs):
            same(ss[seq], after[seq * per_seq + keep - 1][0], f"commit state {seq}/{keep}")
            same(cc[seq], after[seq * per_seq + keep - 1][1], f"commit conv {seq}/{keep}")

    captured = buffers()
    activation = torch.zeros_like(x)
    launch(activation, captured)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(activation, captured)
    for fresh in (x, -x):
        eager = buffers()
        launch(fresh, eager)
        activation.copy_(fresh)
        captured[0].copy_(initial)
        captured[1].copy_(conv)
        captured[2].fill_(float("nan"))
        captured[4].fill_(0xA5)
        graph.replay()
        torch.cuda.synchronize()
        for name, a, b in zip(("state", "conv", "output"), captured, eager):
            same(a, b, "graph " + name)
        assert bool(torch.isfinite(captured[2]).all())
        assert bool((captured[4][-1024:] == 0xA5).all())


DECODE_ROWS = (1, 5, 16)


@pytest.mark.parametrize("rows", DECODE_ROWS)
@pytest.mark.parametrize("spec", (0, 1))
def test_glmf_kda_w8_decode_gemv_rows_bitwise(rows, spec):
    """Rows <= 16: the FP8 GEMV route of the dual-copy program, bitwise (its BF16 copy unread)."""
    w = _weights()
    old, new = _run("decode", 64, rows, 16, spec, w["w_in"], w["w_o"])
    for a, b in zip(old, new):
        assert torch.equal(a, b)


@pytest.mark.parametrize("rows", (17, 24, 33, 64))
@pytest.mark.parametrize("spec", (0, 1))
def test_glmf_kda_w8_decode_wide_rows(rows, spec):
    """Rows 17-64: W8A16 over the one FP8 copy against the W8A16 reference (the dual-copy
    program over the dequantized weight); reports the deviation from the original BF16."""
    w = _weights()
    ref, new = _run("decode", 64, rows, 16, spec, w["in_q"][3], w["o_q"][3])
    orig, _ = _run("decode", 64, rows, 16, spec, w["w_in"], w["w_o"])
    same = torch.equal(ref[0], new[0])
    print(f"decode rows={rows} spec={spec}: vs W8A16 ref out {_rel(new[0], ref[0]):.2e} "
          f"state {_rel(new[1], ref[1]):.2e} bitwise={same}; vs original BF16 out {_rel(new[0], orig[0]):.2e} "
          f"state {_rel(new[1], orig[1]):.2e}")
    assert _rel(new[0], ref[0]) < 1e-2 and _rel(new[1], ref[1]) < 1e-3
    assert _rel(new[0], orig[0]) < 8e-2


@pytest.mark.parametrize("rows", (8, 64, 600))
@pytest.mark.parametrize("fp8_rows", (0, 1, 2, 3))
def test_glmf_kda_w8_prefill(rows, fp8_rows):
    """Prefill: W8A8 on bit 0 / bit 1, else W8A16, bitwise against the dual-copy program over
    the dequantized weight running the same route. Skinny prefills (fewer rows than the BF16
    skinny GEMV takes, where the dual-copy program never ran FP8) compare with that program on
    a 64-row step whose first sequence is the same rows."""
    w = _weights()
    pad = 64 if rows < 64 else 0
    ref, new = _run("prefill", 4096, rows, fp8_rows, 0, w["in_q"][3], w["o_q"][3], per_seq=rows, old_pad=pad)
    orig, _ = _run("prefill", 4096, rows, fp8_rows, 0, w["w_in"], w["w_o"], per_seq=rows)
    same = all(torch.equal(a, b) for a, b in zip(ref, new))
    print(f"prefill rows={rows} fp8_rows={fp8_rows}: vs ref out {_rel(new[0], ref[0]):.2e} "
          f"state {_rel(new[1], ref[1]):.2e} bitwise={same}; vs original BF16 (skinny rows: BF16 GEMV) "
          f"out {_rel(new[0], orig[0]):.2e}")
    assert same
    assert _rel(new[0], orig[0]) < 8e-2


@pytest.mark.parametrize("rows", (1, 16, 17, 40, 64))
def test_glmf_head_fp8_spans(rows):
    """The single FP8 head over any row count: 16-row spans equal per-row launches bitwise and
    track the FP32 reference over the dequantized head."""
    from b12x.integration.cuteafd import GLM53_FLASH as g, glmf

    key = ("head_fp8",)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = glmf.compile_glmf_head_fp8_aot(g)
    prog = _PROGRAMS[key]
    vocab, h = 154880, g.hidden
    gen = torch.Generator(device="cuda").manual_seed(11)
    w8, s, _, wd = _quant_rows((torch.randn((vocab, h), generator=gen, device="cuda") * 0.02).bfloat16())
    x = torch.randn((rows, h), generator=gen, device="cuda").bfloat16()
    spans = torch.empty((rows, vocab), device="cuda")
    for first in range(0, rows, 16):
        n = min(16, rows - first)
        prog.launch(x[first:first + n], w8, s, spans[first:first + n], scalars=(n,))
    single = torch.empty_like(spans)
    for r in range(rows):
        prog.launch(x[r:r + 1], w8, s, single[r:r + 1], scalars=(1,))
    torch.cuda.synchronize()
    ref = x.float() @ wd.float().t()
    print(f"head rows={rows}: spans vs per-row bitwise={torch.equal(spans, single)}, vs FP32 ref "
          f"{_rel(spans, ref):.2e}, argmax agreement "
          f"{float((spans.argmax(-1) == ref.argmax(-1)).float().mean()):.3f}")
    assert torch.equal(spans, single)
    assert _rel(spans, ref) < 1e-2
