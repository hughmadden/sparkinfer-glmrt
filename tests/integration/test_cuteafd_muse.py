"""Muse coordinator rounding boundaries and live-row invariance."""
from __future__ import annotations

import ctypes
import functools

import numpy as np
import pytest
import torch
from ..conftest import require_b12x

ROWS = (1, 16, 80, 2752, 4096)


def _canonical_norm(x, eps=1e-5):
    """Independent CPU transcription of reference.rs sum_squares and inv_rms."""
    width = x.shape[-1]
    work = x.float().reshape(-1, width)
    lanes = torch.zeros(work.shape[0], 8, 32)
    for slab in range((width + 127) // 128):
        for j in range(4):
            at = slab * 128 + torch.arange(32) * 4 + j
            valid = at < width
            lanes[:, slab % 8, valid] += work[:, at[valid]].square()
    for stage in range(5):
        lanes = lanes + lanes[:, :, torch.arange(32) ^ (1 << stage)]
    total = torch.zeros(work.shape[0])
    for warp in range(8):
        total = total + lanes[:, warp, 0]
    inv = 1.0 / ((total / width) + eps).sqrt()
    return (work * inv[:, None]).reshape_as(x)


def _bf16_ulp_close(actual, expected, count=1):
    a, e = actual.float(), expected.float()
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0 ** -126))) - 7)
    assert ((a - e).abs() <= count * ulp).all(), ((a - e).abs().max(), ((a - e).abs() / ulp).max())


def _attention_error_bound(actual, query, keys, values, *, row, expected=None):
    """FP64 softmax/PV after the checkpoint's explicit BF16 score boundaries."""
    q, k, v = query.cpu().double(), keys.cpu().double(), values.cpu().double()
    score = torch.einsum("hd,khd->hk", q, k).bfloat16()
    score = (score * .125).bfloat16().double()
    probability = score.softmax(-1)
    reference = torch.einsum("hk,khd->hd", probability, v)
    terms = torch.einsum("hk,khd->hd", probability, v.abs())
    ulp = torch.exp2(torch.floor(torch.log2(reference.abs().clamp_min(2.0 ** -126))) - 7)
    bound = terms * (2.0 ** -7) + ulp
    eager = (torch.einsum("hk,khd->hd", probability.bfloat16().double(), v).bfloat16().double()
             if expected is None else expected.cpu().double())
    ratios = []
    for name, candidate in (("kernel", actual.cpu().double()), ("eager", eager)):
        ratio = (candidate - reference).abs() / bound
        worst = int(ratio.argmax())
        index = (worst // ratio.shape[-1], worst % ratio.shape[-1])
        assert ratio.max() <= 1, (name, (row, *index), ratio.max(), candidate[index], reference[index], bound[index])
        ratios.append(float(ratio.max()))
    return ratios


@pytest.mark.parametrize("rows", ROWS)
def test_residual_gate(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_residual_gate_aot
    gen = torch.Generator().manual_seed(31)
    residual = torch.randn(rows, 8192, generator=gen).bfloat16()
    x = torch.randn(rows, 8192, generator=gen).bfloat16()
    gate = (torch.randn(8192, generator=gen) * 0.2).bfloat16()
    w = gate.float() / 0.3
    beta = 1 / (1 + torch.exp(-w))
    alpha = ((1 / (1 + torch.exp(w))) * (1 + beta)).clamp_min(1e-3).sqrt()
    expected = (alpha * residual.float() + beta * x.float()).bfloat16()
    r, h, a, b = (t.cuda() for t in (residual, x, alpha, beta))
    out = torch.empty_like(r)
    program = compile_residual_gate_aot()
    program.launch(r, h, a, b, out, scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), expected), ((out.cpu().float() - expected.float()).abs().max(),
        torch.nonzero(out.cpu() != expected)[:10], out.cpu()[out.cpu() != expected][:10], expected[out.cpu() != expected][:10])
    one = torch.empty_like(out[:1])
    program.launch(r[:1], h[:1], a, b, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_route_select(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_route_select_aot
    gen = torch.Generator().manual_seed(32)
    logits = torch.randn(rows, 256, generator=gen)
    bias = torch.randn(256, generator=gen) * 0.01
    logits[0] = 0
    bias[:16] = 0.25
    # Rust f32::exp uses scalar libm expf, unlike Torch CPU's vector exp.
    expf = ctypes.CDLL("libm.so.6").expf
    expf.argtypes, expf.restype = (ctypes.c_float,), ctypes.c_float
    scaled = (logits * 0.1767766952966369).numpy()
    exponent = np.fromiter((expf(float(-v)) for v in scaled.flat), dtype=np.float32).reshape(scaled.shape)
    scores = torch.from_numpy(np.float32(1.0) / (np.float32(1.0) + exponent))
    ids = torch.argsort(scores + bias, descending=True, stable=True)[:, :8].int()
    selected = scores.gather(1, ids.long())
    total = torch.zeros(rows)
    for j in range(8):
        total = total + selected[:, j]
    expected = selected / (total[:, None] + 1e-15)
    l, b = logits.cuda(), bias.cuda()
    out_ids = torch.empty_like(ids, device="cuda")
    weights = torch.empty(rows, 8, device="cuda")
    program = compile_route_select_aot()
    program.launch(l, b, out_ids, weights, scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(out_ids.cpu(), ids)
    actual = weights.cpu()
    ulps = (actual.view(torch.int32) - expected.view(torch.int32)).abs()
    assert ulps.max() <= 2, (ulps.max(), (actual - expected).abs().max())
    # Every row must retain its exact weights outside its original live batch.
    for start in range(0, rows, 16):
        count = min(16, rows - start)
        small_ids, small_weights = torch.empty_like(out_ids[:count]), torch.empty_like(weights[:count])
        program.launch(l[start:start + count], b, small_ids, small_weights, scalars=[count])
        assert torch.equal(small_ids, out_ids[start:start + count])
        assert torch.equal(small_weights, weights[start:start + count])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ("swa", "full"))
@pytest.mark.parametrize("heads", (64, 128))
def test_qk_norm_rope(rows, kind, heads):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_qk_norm_rope_aot
    gen = torch.Generator().manual_seed(33)
    kv_heads = heads // 8
    q = torch.randn(rows, heads, 64, generator=gen).bfloat16()
    k = torch.randn(rows, kv_heads, 64, generator=gen).bfloat16()
    # Include signed zeros, NaNs and subnormals: V is a bitwise copy, not math.
    v = torch.randint(-32768, 32768, (rows, kv_heads, 64), generator=gen,
                      dtype=torch.int16).view(torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)
    freq = positions.float()[:, None] / (500000 ** (torch.arange(0, 64, 2).float() / 64))
    cos, sin = freq.cos().bfloat16(), freq.sin().bfloat16()

    def norm(t):
        return _canonical_norm(t).bfloat16()

    expected_q, expected_k = (norm(q) * 5.4730064863875).bfloat16(), norm(k)
    if kind == "swa":
        def rope(t):
            half = torch.cat((-t[..., 32:], t[..., :32]), -1)
            return t * torch.cat((cos, cos), -1)[:, None] + half * torch.cat((sin, sin), -1)[:, None]
        expected_q, expected_k = rope(expected_q), rope(expected_k)
    operands = [t.cuda() for t in (q, k, v, positions, cos, sin)]
    # Discontiguous page-like records, including an untouched hole after each row.
    slots = (torch.randperm(rows, generator=gen).long() * 2 + 1).cuda()
    qo = torch.empty_like(operands[0])
    records = torch.full((rows * 2 + 2, kv_heads * 128), -17., dtype=torch.bfloat16, device="cuda")
    expected_records = records.cpu()
    expected_records[slots.cpu()] = torch.cat((expected_k.flatten(1), v.flatten(1)), -1)
    program = compile_qk_norm_rope_aot(heads=heads, kind=kind)
    program.launch(*operands, slots, qo, records, scalars=[rows])
    assert torch.equal(qo.cpu(), expected_q)
    assert torch.equal(records.cpu().view(torch.int16), expected_records.view(torch.int16))
    # Fresh singleton prefills must produce identical complete record bytes.
    one_q = torch.empty_like(qo[:1])
    one_record = torch.empty_like(records[:1])
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    for row in sorted({0, rows // 2, rows - 1}):
        single = [t[row:row + 1] for t in operands[:4]] + operands[4:]
        program.launch(*single, zero, one_q, one_record, scalars=[1])
        assert torch.equal(one_q, qo[row:row + 1])
        assert torch.equal(one_record.view(torch.int16), records[slots[row:row + 1]].view(torch.int16))
    # Negative slots must leave every destination byte untouched, but still write Q.
    before = records.clone()
    skipped = torch.full_like(slots, -1)
    program.launch(*operands, skipped, qo, records, scalars=[rows])
    assert torch.equal(qo.cpu(), expected_q)
    assert torch.equal(records.view(torch.int16), before.view(torch.int16))
    from b12x._lib.runtime_control import kernel_resolution_guard
    with kernel_resolution_guard("KV producers are precompiled for all live row counts"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*operands, slots, qo, records, scalars=[rows])
        records.fill_(-17.)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(records.cpu().view(torch.int16), expected_records.view(torch.int16))
        # Zero rows must not launch an invalid zero grid or touch either output.
        saved_q = qo.clone()
        program.launch(*operands, slots, qo, records, scalars=[0])
        assert torch.equal(qo, saved_q)
        assert torch.equal(records.cpu().view(torch.int16), expected_records.view(torch.int16))


@pytest.mark.parametrize("heads", (64, 128))
def test_qk_half_table_matches_full_above_million(heads):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_qk_norm_rope_aot
    gen = torch.Generator().manual_seed(133)
    position, kv_heads = (1 << 20) + 137, heads // 8
    q = torch.randn(1, heads, 64, generator=gen).bfloat16()
    k = torch.randn(1, kv_heads, 64, generator=gen).bfloat16()
    v = torch.randn_like(k)
    freq = torch.tensor(position, dtype=torch.float32) / (
        500000 ** (torch.arange(0, 64, 2).float() / 64))
    c, s = freq.cos().bfloat16(), freq.sin().bfloat16()
    # The previous ABI's full-width row repeats each half-width table row.
    full_cos, full_sin = torch.cat((c, c)), torch.cat((s, s))

    def old_full_table(t, scale=1.0):
        t = (_canonical_norm(t).bfloat16() * scale).bfloat16()
        rotated = torch.cat((-t[..., 32:], t[..., :32]), -1)
        return t * full_cos + rotated * full_sin

    cos = torch.empty(position + 1, 32, dtype=torch.bfloat16, device="cuda")
    sin = torch.empty_like(cos)
    cos[position], sin[position] = c.cuda(), s.cuda()
    qo = torch.empty_like(q, device="cuda")
    record = torch.empty(1, kv_heads * 128, dtype=torch.bfloat16, device="cuda")
    program = compile_qk_norm_rope_aot(heads=heads, kind="swa")
    program.launch(q.cuda(), k.cuda(), v.cuda(),
                   torch.tensor([position], dtype=torch.int64, device="cuda"), cos, sin,
                   torch.zeros(1, dtype=torch.int64, device="cuda"), qo, record, scalars=[1])
    assert torch.equal(qo.cpu(), old_full_table(q, 5.4730064863875))
    expected_record = torch.cat((old_full_table(k).flatten(1), v.flatten(1)), -1)
    assert torch.equal(record.cpu().view(torch.int16), expected_record.view(torch.int16))


@pytest.mark.parametrize("rows", ROWS)
def test_attention_output(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_attention_output_aot
    gen = torch.Generator().manual_seed(34)
    x = torch.randn(rows, 128, 64, generator=gen).bfloat16()
    gate = torch.randn(rows, 128, 64, generator=gen).bfloat16()
    norm = _canonical_norm(x).bfloat16()
    expected = norm * gate.sigmoid()
    xd, gd = x.cuda(), gate.cuda()
    out = torch.empty_like(xd)
    program = compile_attention_output_aot()
    program.launch(xd, gd, out, scalars=[rows])
    _bf16_ulp_close(out.cpu(), expected)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], gd[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("op", ("embed_norm", "input_norm", "pre_ffn_norm", "pre_expert_norm", "post_attn_norm", "post_ffn_norm", "final_norm"))
def test_row_norms(rows, op):
    require_b12x()
    from b12x.integration.cuteafd.muse import NORMS, compile_norm_aot
    width, mode, eps = NORMS[op]
    gen = torch.Generator().manual_seed(36)
    x = torch.randn(rows, width, generator=gen).bfloat16()
    weight = (torch.randn(width, generator=gen) * 0.1).bfloat16()
    scale = (1 + weight).float()
    source = (x.float() * scale).bfloat16() if mode == "input_scaled" else x
    expected = _canonical_norm(source, eps)
    if mode == "centered":
        expected = expected * scale
    elif mode == "scaled":
        expected = expected * weight.float()
    expected = expected.bfloat16()
    xd, wd = x.cuda(), weight.cuda()
    out = torch.empty_like(xd)
    args = (xd,) if mode == "weightless" else (xd, wd)
    program = compile_norm_aot(op=op)
    program.launch(*args, out, scalars=[rows])
    assert torch.equal(out.cpu(), expected), ((out.cpu() != expected).sum(), (out.cpu().float() - expected.float()).abs().max())
    one = torch.empty_like(out[:1])
    args = (xd[:1],) if mode == "weightless" else (xd[:1], wd)
    program.launch(*args, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_softcap(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_softcap_aot
    # Include every finite BF16 code across rows; a compact head exercises tail masks.
    vocab = 257
    codes = torch.arange(rows * vocab, dtype=torch.int32)
    values = (codes.remainder(0x7f80) | ((codes % 2) << 15)).to(torch.int16)
    x = values.view(torch.bfloat16).reshape(rows, vocab)
    scaled = (x * 0.17677669529663687) / 20
    expected = (scaled.tanh() * 20).float()
    xd = x.float().cuda()
    out = torch.empty_like(xd, dtype=torch.float32)
    program = compile_softcap_aot(vocab=vocab)
    program.launch(xd, out, scalars=[rows])
    _bf16_ulp_close(out.cpu(), expected)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_native_f32_router(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_router_aot
    gen = torch.Generator().manual_seed(37)
    x = torch.randn(rows, 8192, generator=gen).bfloat16()
    weight = torch.randn(256, 8192, generator=gen) * 0.01
    # The checkpoint's low F32 bits must influence the result.
    expected = x.float() @ weight.T
    xd, wd = x.cuda(), weight.cuda()
    out = torch.empty(rows, 256, device="cuda")
    program = compile_router_aot()
    program.launch(xd, wd, out, scalars=[rows])
    torch.testing.assert_close(out.cpu(), expected, atol=3e-6, rtol=3e-4)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], wd, one, scalars=[1])
    assert torch.equal(one, out[:1])


# Every width the router sees: decode/verify widths, partial tiles, the scalar
# route cap and its neighbour, and the three prefill unit widths.
ROUTER_ROWS = (1, 7, 8, 16, 64, 80, 256, 257, 1024, 2048, 2721, 2752, 4096)


@functools.lru_cache(maxsize=None)
def _router_program(scalar_max_rows, rows_per_warp=8, experts_per_warp=8,
                    block_warps=8, row_groups=4):
    from b12x.integration.cuteafd.muse import compile_router_aot
    return compile_router_aot(
        scalar_max_rows=scalar_max_rows, rows_per_warp=rows_per_warp,
        experts_per_warp=experts_per_warp, block_warps=block_warps, row_groups=row_groups)


def _router_bits(out):
    return out.detach().view(torch.int32).cpu()


def _router_inputs(rows, seed=37):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, 8192, generator=gen).bfloat16().cuda(),
            (torch.randn(256, 8192, generator=gen) * 0.01).cuda())


@pytest.mark.parametrize("rows", ROUTER_ROWS)
def test_router_tile_bit_exact(rows):
    """Forcing either route stores identical bits at every live width."""
    require_b12x()
    x, weight = _router_inputs(rows)
    scalar = torch.empty(rows, 256, device="cuda")
    tiled = torch.empty_like(scalar)
    _router_program(2**31 - 1).launch(x, weight, scalar, scalars=[rows])
    _router_program(0).launch(x, weight, tiled, scalars=[rows])
    assert torch.equal(_router_bits(tiled), _router_bits(scalar))
    replay = torch.empty_like(tiled)
    _router_program(0).launch(x, weight, replay, scalars=[rows])
    assert torch.equal(_router_bits(replay), _router_bits(tiled))


@pytest.mark.parametrize("rows", (64, 2721, 2752))
@pytest.mark.parametrize("tile", ((8, 8, 8, 4), (8, 4, 8, 4), (4, 8, 8, 4)))
def test_router_tile_geometry_bit_exact(rows, tile):
    """Every screened tile geometry stores the scalar route's bits."""
    require_b12x()
    x, weight = _router_inputs(rows, seed=53)
    scalar = torch.empty(rows, 256, device="cuda")
    tiled = torch.empty_like(scalar)
    _router_program(2**31 - 1).launch(x, weight, scalar, scalars=[rows])
    _router_program(0, *tile).launch(x, weight, tiled, scalars=[rows])
    assert torch.equal(_router_bits(tiled), _router_bits(scalar))


@pytest.mark.parametrize("rows", (256, 257, 512, 2752, 2753))
def test_router_route_invariance(rows):
    """The row-count route cannot move a result: every row is bit-identical to
    the scalar route's row, and a row launched alone matches its batch row."""
    require_b12x()
    from b12x.integration.cuteafd.muse import ROUTER_SCALAR_MAX_ROWS
    x, weight = _router_inputs(rows, seed=41)
    dispatch = torch.empty(rows, 256, device="cuda")
    scalar_route = torch.empty_like(dispatch)
    program = _router_program(ROUTER_SCALAR_MAX_ROWS)
    program.launch(x, weight, dispatch, scalars=[rows])
    _router_program(2**31 - 1).launch(x, weight, scalar_route, scalars=[rows])
    assert torch.equal(_router_bits(dispatch), _router_bits(scalar_route))
    for at in (0, rows // 2, rows - 1):
        one = torch.empty(1, 256, device="cuda")
        program.launch(x[at:at + 1], weight, one, scalars=[1])
        assert torch.equal(_router_bits(one), _router_bits(dispatch[at:at + 1]))


def test_router_nonfinite_and_tie_policy():
    """Both routes share the IEEE rounding policy: infinities, NaN, zeros and
    exact weight ties produce the same bits, and no row poisons another."""
    require_b12x()
    rows = 40
    x, weight = _router_inputs(rows, seed=11)
    x, weight = x.cpu(), weight.cpu()
    x[3] = float("inf")
    x[4, 0] = float("nan")
    x[5] = 0.0
    x[6, 7] = -0.0
    x[7] = torch.finfo(torch.bfloat16).tiny
    weight[0] = 0.0
    weight[1] = 1.0
    weight[2] = weight[1]
    weight[3, 0] = float("inf")
    weight[4, 0] = float("nan")
    weight[5, 0] = -0.0
    # Exact cancellation: alternating +-1 inputs against an all-ones row sums to
    # zero in both routes (every partial is an exact small integer).
    weight[6] = 1.0
    x[8] = 0.0
    x[8, ::2] = 1.0
    x[8, 1::2] = -1.0
    xd, wd = x.cuda(), weight.cuda()
    scalar = torch.empty(rows, 256, device="cuda")
    tiled = torch.empty_like(scalar)
    _router_program(2**31 - 1).launch(xd, wd, scalar, scalars=[rows])
    _router_program(0).launch(xd, wd, tiled, scalars=[rows])
    assert torch.equal(_router_bits(tiled), _router_bits(scalar))
    # A NaN input row stays NaN for every expert; unrelated rows and experts
    # stay finite, so neither route poisons data it does not own.
    assert torch.isnan(scalar[4]).all()
    finite_experts = torch.ones(256, dtype=torch.bool)
    finite_experts[[3, 4]] = False
    assert torch.isfinite(scalar[0][finite_experts]).all()
    assert torch.isfinite(scalar[9][finite_experts]).all()
    # Identical checkpoint rows keep identical logits, ties included, and the
    # cancellation row is exactly +0.0 (positive zero, not only == 0). NaN rows
    # are compared by classification: torch.equal rejects NaN != NaN.
    tie = _router_bits(scalar)
    finite_rows = torch.isfinite(scalar[:, 1])
    tie_finite = finite_rows.cpu()
    assert tie_finite.any() and not tie_finite.all()
    assert torch.equal(tie[tie_finite, 1], tie[tie_finite, 2])
    nan_rows = torch.isnan(scalar[:, 1])
    assert nan_rows.any()
    assert torch.isnan(scalar[nan_rows, 2]).all()
    assert int(_router_bits(scalar)[8, 6]) == 0


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("history", (2050, 6140))
@pytest.mark.parametrize("kind", ("full", "swa"))
def test_attention_oracle_and_unit_invariance(kind, history, heads):
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot, compile_swa_commit_aot
    gen = torch.Generator().manual_seed(35)
    rows, kv_heads, ring_rows = 17, heads // 8, 6144
    end = history + rows
    query = torch.randn(rows, heads, 64, generator=gen).bfloat16().cuda()
    keys = torch.randn(end, kv_heads, 64, generator=gen).bfloat16()
    values = torch.randn(end, kv_heads, 64, generator=gen).bfloat16()
    records = torch.cat((keys.flatten(1), values.flatten(1)), -1).cuda()
    positions = torch.arange(history, end, dtype=torch.int64, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    prefill = compile_attention_aot(heads=heads, kind=kind, route="prefill")
    decode = compile_attention_aot(heads=heads, kind=kind, route="decode")
    out = torch.empty_like(query)
    singles = torch.empty_like(query)
    if kind == "full":
        cache = torch.zeros(((end + 63) // 64 * 64, kv_heads * 128), dtype=torch.bfloat16, device="cuda")
        cache[:end].copy_(records)
        table = torch.arange(cache.shape[0] // 64, dtype=torch.int32, device="cuda")
        prefill.launch(query, cache, positions, table, out, scratch, scalars=[rows, 0])
        for row in range(rows):
            decode.launch(query[row:row + 1], cache, positions[row:row + 1], table,
                          singles[row:row + 1], scratch, scalars=[1, 0, 1])
    else:
        ring = torch.zeros(ring_rows, kv_heads * 128, dtype=torch.bfloat16, device="cuda")
        prior = torch.arange(max(0, history - 2048), history, device="cuda")
        ring[prior % ring_rows] = records[prior]
        slots = positions % ring_rows
        seq_first = torch.zeros(rows, dtype=torch.int32, device="cuda")
        step = records[history:].contiguous()
        prefill.launch(query, step, ring, positions, slots, seq_first, out, scratch, scalars=[rows])
        commit = compile_swa_commit_aot(heads=heads)
        for row in range(rows):
            decode.launch(query[row:row + 1], step[row:row + 1], ring, positions[row:row + 1], slots[row:row + 1],
                          seq_first[:1], singles[row:row + 1], scratch, scalars=[1])
            commit.launch(step[row:row + 1], ring, slots[row:row + 1], seq_first[:1], scalars=[1])
    assert torch.equal(out, singles), (out.float() - singles.float()).abs().max()
    from b12x._lib.runtime_control import kernel_resolution_guard
    expected_graph = out.clone()
    args = ((query, cache, positions, table, out, scratch) if kind == "full" else
            (query, step, ring, positions, slots, seq_first, out, scratch))
    scalars = [rows, 0] if kind == "full" else [rows]
    with kernel_resolution_guard("Muse attention reuses precompiled program on graph replay"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prefill.launch(*args, scalars=scalars)
        for _ in range(2):
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected_graph)
    k, v = keys.repeat_interleave(8, dim=1), values.repeat_interleave(8, dim=1)
    ratios = []
    for row in range(rows):
        position = history + row
        first = max(0, position - 2047) if kind == "swa" else 0
        ratios.append(_attention_error_bound(out[row], query[row],
                        k[first:position + 1], v[first:position + 1], row=row))
    print(f"random {kind} heads={heads} history={history}: kernel/eager max err/bound "
          f"{np.max(ratios, axis=0).tolist()}")


@pytest.mark.parametrize("layer,kind", ((0, "swa"), (1, "full")))
def test_engine_attention_exchange(layer, kind):
    import os
    from pathlib import Path
    source = os.environ.get("CUTEAFD_MUSE_ATTENTION_FIXTURE")
    if not source:
        pytest.skip("Muse engine attention exchange fixture not configured")
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot
    root = Path(source)
    def read(leaf, shape):
        bits = np.fromfile(root / f"layer{layer}.{leaf}.bin", dtype="<u2").copy()
        return torch.from_numpy(bits).view(torch.bfloat16).reshape(shape).cuda()
    q = read("q", (9, 8, 64))
    k, v = (read(leaf, (9, 2, 64)) for leaf in ("k", "v"))
    expected = read("out", (9, 8, 64))
    records = torch.cat((k.flatten(1), v.flatten(1)), -1)
    pos = torch.arange(9, dtype=torch.int64, device="cuda")
    out, scratch = torch.empty_like(q), torch.empty(1024, dtype=torch.uint8, device="cuda")
    program = compile_attention_aot(heads=8, kv_heads=2, window=5, kind=kind,
                                    route="prefill", ring_rows=64)
    if kind == "full":
        cache = torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda")
        cache[:9] = records
        table = torch.zeros(1, dtype=torch.int32, device="cuda")
        program.launch(q, cache, pos, table, out, scratch, scalars=[9, 0])
    else:
        ring = torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda")
        first = torch.zeros(9, dtype=torch.int32, device="cuda")
        program.launch(q, records, ring, pos, pos, first, out, scratch, scalars=[9])
    destination = Path(os.environ["CUTEAFD_MUSE_ATTENTION_OUTPUT"])
    destination.mkdir(parents=True, exist_ok=True)
    out.cpu().view(torch.uint16).numpy().tofile(destination / f"layer{layer}.out.bin")
    ratios = []
    keys, values = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    for row in range(9):
        first_key = max(0, row - 4) if kind == "swa" else 0
        ratios.append(_attention_error_bound(out[row], q[row],
                      keys[first_key:row + 1], values[first_key:row + 1], row=row, expected=expected[row]))
    print(f"fixture {kind}: kernel/eager max err/bound {np.max(ratios, axis=0).tolist()}")

@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ("full", "swa"))
def test_attention_live_row_ladder(rows, kind):
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot, compile_swa_commit_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    heads, kv, ring_rows = 64, 8, 6144
    gen = torch.Generator(device="cuda").manual_seed(39)
    q = torch.randn(rows, heads, 64, device="cuda", generator=gen).bfloat16()
    records = torch.randn(rows, kv * 128, device="cuda", generator=gen).bfloat16()
    pos = torch.arange(rows, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    prefill = compile_attention_aot(heads=heads, kind=kind, route="prefill")
    decode = compile_attention_aot(heads=heads, kind=kind, route="decode")
    if kind == "full":
        cache = torch.zeros((rows + 63) // 64 * 64, kv * 128, device="cuda", dtype=torch.bfloat16)
        cache[:rows] = records
        table = torch.arange(cache.shape[0] // 64, device="cuda", dtype=torch.int32)
        prefill.launch(q, cache, pos, table, out, scratch, scalars=[rows, 0])
    else:
        ring = torch.zeros(ring_rows, kv * 128, device="cuda", dtype=torch.bfloat16)
        first = torch.zeros(rows, device="cuda", dtype=torch.int32)
        prefill.launch(q, records, ring, pos, pos, first, out, scratch, scalars=[rows])
        commit = compile_swa_commit_aot(heads=heads)
        commit.launch(records, ring, pos, first, scalars=[rows])
    one = torch.empty_like(out[:1])
    with kernel_resolution_guard("Muse attention live rows reuse fixed geometry"):
        for row in sorted({0, rows // 2, rows - 1}):
            if kind == "full":
                decode.launch(q[row:row + 1], cache, pos[row:row + 1], table,
                              one, scratch, scalars=[1, 0, 1])
            else:
                decode.launch(q[row:row + 1], records[row:row + 1], ring, pos[row:row + 1],
                              pos[row:row + 1], first[:1], one, scratch, scalars=[1])
            assert torch.equal(one[0], out[row])


def test_full_attention_large_page_id_and_mixed_sequences():
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot
    # The second sequence starts beyond a signed 32-bit byte offset.
    page = 16385
    cache = torch.empty((page + 1) * 64, 1024, device="cuda", dtype=torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(40)
    first = torch.randn(2, 1024, device="cuda", generator=gen).bfloat16()
    second = torch.randn(2, 1024, device="cuda", generator=gen).bfloat16()
    cache[:2], cache[page * 64:page * 64 + 2] = first, second
    table = torch.tensor([0, page], dtype=torch.int32, device="cuda")
    pos = torch.tensor([1, 1], dtype=torch.int64, device="cuda")
    q = torch.randn(2, 64, 64, device="cuda", generator=gen).bfloat16()
    out = torch.empty_like(q)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    program = compile_attention_aot(heads=64, kind="full", route="decode")
    program.launch(q, cache, pos, table, out, scratch, scalars=[2, 1, 1])
    one = torch.empty_like(out[:1])
    for row in range(2):
        program.launch(q[row:row + 1], cache, pos[row:row + 1], table[row:row + 1],
                       one, scratch, scalars=[1, 1, 1])
        assert torch.equal(one[0], out[row])
        compact = torch.zeros(64, 1024, device="cuda", dtype=torch.bfloat16)
        compact[:2] = (first, second)[row]
        zero_page = torch.zeros(1, device="cuda", dtype=torch.int32)
        program.launch(q[row:row + 1], compact, pos[row:row + 1], zero_page,
                       one, scratch, scalars=[1, 0, 1])
        assert torch.equal(one[0], out[row])
