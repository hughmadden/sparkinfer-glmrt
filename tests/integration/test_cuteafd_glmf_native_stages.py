"""Bitwise native projection seams against GLM Flash's existing fused stages."""
from dataclasses import replace

import pytest
import torch

from .test_cuteafd_kimi_split import rand, empty, scratch, same, projection
from ..conftest import require_b12x


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    require_b12x()
    torch.manual_seed(836)


@pytest.mark.parametrize("rows,decode", [(1, True), (16, True), (64, True), (129, False)])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("mode", ["core", "fully_projected", "caller_rows"])
def test_flash_kda_core_bitwise(rows, decode, heads, mode):
    """``mode``: the staged core (aux projection + pack), the native-weight seam, or the projected
    core over in-projection rows the caller writes into its scratch (as KX7 engines do)."""
    fully_projected = mode == "fully_projected"
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_kda_aot, compile_glmf_kda_commit_aot
    from b12x.integration.cuteafd.glmf_split import (compile_glmf_kda_core_aot,
                                                   compile_glmf_kda_core_projected_aot,
                                                   compile_glmf_projected_kda_aot)
    from b12x.integration.cuteafd._glm_kernels import BatchedBf16Gemm
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout
    g = replace(GLM53_FLASH, kda_heads=heads)
    cap, d, h, p = max(64, rows), g.kda_width, g.hidden, g.kda_in_width
    fused = compile_glmf_kda_aot(g, max_rows=cap, decode=decode, output_kind="norm")
    split = {"core": compile_glmf_kda_core_aot, "fully_projected": compile_glmf_projected_kda_aot,
             "caller_rows": compile_glmf_kda_core_projected_aot}[mode](g, max_rows=cap, decode=decode)
    x, w = rand(rows, h), rand(p, h)
    wf, cw = rand(2, d, 128), rand(3*d, 4, dtype=torch.float32)
    al, dt, norm = rand(heads, dtype=torch.float32), rand(d, dtype=torch.float32), rand(128, scale=1)
    proj = empty(rows, p)
    projection(p, h).launch(x, w, proj, scalars=(rows,))
    q, k, v = [proj[:, i*d:(i+1)*d].contiguous() for i in range(3)]
    fab = w[3*d:].contiguous()
    if fully_projected:
        # Use the identical BF16 second GEMMs, isolating the new pack/state seam.
        aux = proj[:, 3*d:].contiguous()
        fg = empty(rows, 2*d)
        fg_program = compile_program(BatchedBf16Gemm(n=d, k=128, batch=2,
            a_row=256+heads, a_batch=128, o_row=2*d, o_batch=d),
            name="flash_projected_test_fg",
            operands=tuple(Operand(name, torch.bfloat16, "[]", role)
                for name, role in (("a", "in"), ("w", "in"), ("out", "out"))),
            scalars=(Scalar("rows"),), key=(g,))
        fg_program.launch(aux, wf, fg, scalars=(rows,))
        decay, gate = fg[:, :d].contiguous(), fg[:, d:].contiguous()
        beta = aux[:, 256:].contiguous()
    wo = empty(h, d)
    state0, conv0 = rand(2, heads, 128, 128, dtype=torch.float32), rand(2, 3, 3*d)
    slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first = torch.zeros_like(slots)
    if decode and rows > 1:
        slots[rows//2:] = 1
        first[rows//2:] = rows//2
        slots[-1], first[-1] = -1, rows-1
    out, ref = empty(rows, d), empty(rows, d)
    state, conv, sr, cr = state0.clone(), conv0.clone(), state0.clone(), conv0.clone()
    replay = torch.zeros(kda_replay_layout(heads, 3*d, 64)[2]//4, device="cuda")
    rr = replay.clone()
    fs, ss = scratch(fused, cap), scratch(split, cap)
    def run(spec):
        state.copy_(state0); conv.copy_(conv0); sr.copy_(state0); cr.copy_(conv0)
        replay.zero_(); rr.zero_()
        args = (x, w, wf, cw, al, dt, norm, wo, cr, sr, slots, first, ref)
        if decode:
            fused.launch(*args, rr, fs, scalars=(rows, spec))
        else:
            fused.launch(*args, fs, scalars=(rows,))
        if fully_projected:
            split.launch(q, k, v, decay, gate, beta, cw, al, dt, norm, conv, state,
                slots, first, out, replay, ss, scalars=(rows, spec))
        elif mode == "caller_rows":
            # The whole in-projection rows (q | k | v | f_a, g_a, beta) at the base of scratch.
            ss[:rows * p * 2].view(torch.bfloat16).copy_(proj.reshape(-1))
            split.launch(x, q, k, v, q, fab, wf, cw, al, dt, norm, conv, state,
                slots, first, out, replay, ss, scalars=(rows, spec))
        else:
            split.launch(x, q, k, v, q, fab, wf, cw, al, dt, norm, conv, state,
                slots, first, out, replay, ss, scalars=(rows, spec))
    for spec in ((0, 1) if decode else (0,)):
        run(spec)
        same(out, ref); same(state, sr); same(conv, cr)
        if spec:
            same(replay, rr)
            same(state, state0); same(conv, conv0)
            commit = compile_glmf_kda_commit_aot(g)
            tables = torch.tensor([[0], [0], [max(1, rows//3)]], device="cuda", dtype=torch.int32)
            commit.launch(state, conv, replay, tables, scalars=(1, 1, 1))
            commit.launch(sr, cr, rr, tables, scalars=(1, 1, 1))
            same(state, sr); same(conv, cr)
    pointers = [t.data_ptr() for t in (state, conv, replay, out, ss)]
    with kernel_resolution_guard("Flash KDA seams are precompiled"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(0)
        graph.replay(); graph.replay()
    same(out, ref); same(state, sr); same(conv, cr)
    assert [t.data_ptr() for t in (state, conv, replay, out, ss)] == pointers
    assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0


@pytest.mark.parametrize("rows", [1, 16, 64, 129])
@pytest.mark.parametrize("heads", [32, 64])
def test_flash_mla_stages_bitwise(rows, heads):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_mla_producer_aot, compile_glmf_o_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_stage_aot
    g = replace(GLM53_FLASH, heads=heads)
    cap, h, q = max(rows, 64), g.hidden, g.q_lora_rank
    fused = compile_glmf_mla_producer_aot(g, max_rows=cap)
    norm = compile_glmf_stage_aot(g, stage="rank_norm", max_rows=cap)
    absorb = compile_glmf_stage_aot(g, stage="absorb", max_rows=cap)
    values = compile_glmf_stage_aot(g, stage="values", max_rows=cap)
    output = compile_glmf_o_aot(g, max_rows=cap)
    x, wa, wb, uk = rand(rows, h), rand(q+512, h), rand(heads*256, q), rand(heads, 512, 256)
    qn, kn = rand(q, scale=1), rand(512, scale=1)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots[-1] = -1
    cache = torch.full((rows, 528), 173, device="cuda", dtype=torch.uint8)
    cref = cache.clone()
    packed = empty(rows, q+512)
    projection(q+512, h).launch(x, wa, packed, scalars=(rows,))
    qa, ka = packed[:, :q].contiguous(), packed[:, q:].contiguous()
    query, ref = empty(rows, heads, 512), empty(rows, heads, 512)
    qr, resid, qb = empty(rows, q), empty(rows, q), empty(rows, heads*256)
    bp = projection(heads*256, q)
    fs, ns = scratch(fused, cap), scratch(norm, cap)
    attn, uv, wo = rand(rows, heads, 512), rand(heads, 256, 512), rand(h, heads*256)
    vs, out, oref = empty(rows, heads*256), empty(rows, h), empty(rows, h)
    os, op = scratch(output, cap), projection(h, heads*256)
    def run():
        fused.launch(x, slots, wa, qn, kn, wb, uk, cref, ref, resid, fs, scalars=(rows,))
        norm.launch(qa, ka, qn, kn, slots, qr, cache, ns, scalars=(rows,))
        bp.launch(qr, wb, qb, scalars=(rows,))
        absorb.launch(qb, uk, query, scalars=(rows,))
        output.launch(attn, uv, wo, oref, os, scalars=(rows,))
        values.launch(attn, uv, vs, scalars=(rows,))
        op.launch(vs, wo, out, scalars=(rows,))
    run()
    same(qr, resid); same(cache, cref); same(query, ref); same(out, oref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay(); graph.replay()
    same(cache, cref); same(query, ref); same(out, oref)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("stage", ["absorb", "values"])
@pytest.mark.parametrize("mode,rows", [("decode", 1), ("decode", 17), ("decode", 64),
                                       ("prefill", 129)])
def test_flash_native_fp8_mla_oracle(heads, stage, mode, rows):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf_split import (compile_glmf_fp8_stage_aot,
                                                   compile_glmf_stage_aot)
    from b12x._lib.runtime_control import kernel_resolution_guard

    g = replace(GLM53_FLASH, heads=heads)
    cap = max(64, rows)
    n, k = (512, 256) if stage == "absorb" else (256, 512)
    native = compile_glmf_fp8_stage_aot(g, stage=stage, mode=mode, max_rows=cap)
    bf16 = compile_glmf_stage_aot(g, stage=stage, max_rows=cap)
    a = rand(cap, heads, k, scale=.2)
    packed = rand(heads, n, k, scale=.2).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-3, 2, (heads, n, k//64), device="cuda").float())
    weight = (packed.float()*scales.repeat_interleave(64, dim=-1)).bfloat16()
    guarded = torch.full((cap*heads*n+64,), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = guarded[32:-32].view(cap, heads, n)
    expected = empty(cap, heads, n)
    immutable = [t.view(torch.uint8).clone() for t in (a, packed, scales)]
    pointers = [t.data_ptr() for t in (a, packed, scales, out)]

    def run(live):
        native.launch(a, packed, scales, out, scalars=(live,))

    bf16.launch(a, weight, expected, scalars=(rows,))
    run(rows)
    same(out[:rows], expected[:rows])
    oracle = torch.einsum("rhk,hnk->rhn", a[:rows].double(), weight.double())
    torch.testing.assert_close(out[:rows].double(), oracle, rtol=.005, atol=.005)
    assert torch.isfinite(out[:rows]).all() and torch.count_nonzero(out[:rows]) > 0
    saved = out[:rows].clone()
    torch.cuda.synchronize()
    with kernel_resolution_guard("Flash native FP8 MLA is precompiled"):
        for live in sorted({1, min(rows, 16), min(rows, 17), rows}):
            guarded.fill_(float("nan"))
            run(live)
            same(out[:live], saved[:live])
            assert torch.isnan(out[live:]).all()
            assert torch.isnan(guarded[:32]).all() and torch.isnan(guarded[-32:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(rows)
        for _ in range(3):
            guarded.fill_(float("nan"))
            graph.replay()
            same(out[:rows], saved)
            assert torch.isnan(out[rows:]).all()
            assert torch.isnan(guarded[:32]).all() and torch.isnan(guarded[-32:]).all()
    for value, before in zip((a, packed, scales), immutable):
        assert torch.equal(value.view(torch.uint8), before)
    assert [t.data_ptr() for t in (a, packed, scales, out)] == pointers


@pytest.mark.parametrize("capacity,rows", [(64, 1), (64, 17), (128, 128), (4096, 257)])
def test_flash_projected_token_keys_bitwise(capacity, rows):
    from b12x.integration.cuteafd._common import GLM53_FLASH as g
    from b12x.integration.cuteafd.glmf import compile_glmf_index_producer_aot, _align
    from b12x.integration.cuteafd.glmf_split import compile_glmf_projected_index_keys_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    fused = compile_glmf_index_producer_aot(g, max_rows=capacity)
    projected = compile_glmf_projected_index_keys_aot(g, max_rows=capacity)
    x, qr = rand(rows, g.hidden), rand(rows, g.q_lora_rank)
    wiq, wik = rand(32*128, g.q_lora_rank, scale=.02), rand(288, g.hidden, scale=.02)
    norm, bias, ape = rand(128, scale=1), rand(128), rand(4, 128)
    # Four-record pools remain contiguous, but successive pools jump between pages.
    groups = (rows + 3)//4
    record = [64*(groups - r//4) + r%4 for r in range(rows)]
    pool = [64*(groups - r//4) + 7 if r%4 == 3 else -1 for r in range(rows)]
    if rows % 4:
        record[-1], pool[-1] = -1, -1
    slots = torch.tensor(record, device="cuda", dtype=torch.int64)
    pools = torch.tensor(pool, device="cuda", dtype=torch.int64)
    initial_keys = rand((groups+2)*64, 256)
    initial_cache = torch.full((groups+2, 8448), 173, device="cuda", dtype=torch.uint8)
    keys, ref_keys = initial_keys.clone(), initial_keys.clone()
    cache, ref_cache = initial_cache.clone(), initial_cache.clone()
    query, ref_query = [torch.empty((rows, 32, 128), device="cuda", dtype=torch.float8_e4m3fn)
                        for _ in range(2)]
    heads, ref_heads = empty(rows, 32, dtype=torch.float32), empty(rows, 32, dtype=torch.float32)
    fs = scratch(fused, capacity)
    iq = fs[:rows*32*128*2].view(torch.bfloat16)
    kw_at = _align(rows*32*128*2)
    kw = fs[kw_at:kw_at+rows*288*2].view(torch.bfloat16)
    saved_pointers = [t.data_ptr() for t in (keys, cache, query, heads, fs)]

    def run():
        fused.launch(x, qr, slots, pools, wiq, wik, norm, bias, ape, ref_keys, ref_cache,
                     ref_query, ref_heads, fs, scalars=(rows,))
        projected.launch(iq, kw, slots, pools, norm, bias, ape, keys, cache, query, heads,
                         scalars=(rows,))

    def check():
        for a, b in ((keys, ref_keys), (cache, ref_cache), (query, ref_query), (heads, ref_heads)):
            same(a, b)
        written_keys = torch.zeros(keys.shape[0], device="cuda", dtype=torch.bool)
        written_keys[slots[slots >= 0]] = True
        same(keys[~written_keys], initial_keys[~written_keys])
        assert torch.count_nonzero(query.view(torch.uint8)) > 0
        assert torch.isfinite(heads).all()

    run()
    check()
    with kernel_resolution_guard("projected token-key stages are precompiled"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            keys.copy_(initial_keys); ref_keys.copy_(initial_keys)
            cache.copy_(initial_cache); ref_cache.copy_(initial_cache)
            graph.replay()
            check()
    assert [t.data_ptr() for t in (keys, cache, query, heads, fs)] == saved_pointers


@pytest.mark.parametrize("inter", [1024, 2048, 6144, 12288])
def test_flash_ffn_clamped_stage_bitwise(inter):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_ffn_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_stage_aot
    rows, h = 16, GLM53_FLASH.hidden
    fused = compile_glmf_ffn_aot(inter=inter, max_rows=64)
    stage = compile_glmf_stage_aot(stage="swiglu", inter=inter, max_rows=64)
    # Large projections exercise the activation's +/-10 clamp.
    x, w, wd = rand(rows, h, scale=.4), rand(2*inter, h, scale=.4), rand(h, inter)
    gu, hidden, out, ref = empty(rows, 2*inter), empty(rows, inter), empty(rows, h), empty(rows, h)
    projection(2*inter, h).launch(x, w, gu, scalars=(rows,))
    gate, up = gu[:, :inter].contiguous(), gu[:, inter:].contiguous()
    fs, ss, down = scratch(fused, 64), scratch(stage, 64), projection(h, inter)
    fused.launch(x, w, wd, ref, fs, scalars=(rows,))
    stage.launch(gate, up, hidden, ss, scalars=(rows,))
    down.launch(hidden, wd, out, scalars=(rows,))
    same(out, ref)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("mode,rows,flag", [("decode", 1, 64), ("decode", 17, 64),
    ("decode", 64, 64), ("prefill", 129, 0), ("prefill", 129, 1)])
def test_flash_standalone_block_fp8_matches_fused(heads, mode, rows, flag):
    """Use the fused producers' own projected rows, not a different GEMM oracle."""
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_mla_producer_aot, compile_glmf_o_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_fp8_linear_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    g = replace(GLM53_FLASH, heads=heads)
    cap = 64 if mode == "decode" else 4096
    producer = compile_glmf_mla_producer_aot(g, max_rows=cap, fp8_only=mode)
    output = compile_glmf_o_aot(g, max_rows=cap, fp8_only=mode)
    qb = compile_glmf_fp8_linear_aot(input_features=g.q_lora_rank,
        output_features=heads*g.qk_nope_dim, max_rows=cap, decode=mode == "decode")
    op = compile_glmf_fp8_linear_aot(input_features=heads*g.v_head_dim,
        output_features=g.hidden, max_rows=cap, decode=mode == "decode")
    def weight(n, k):
        q = rand(n, k, scale=.1).to(torch.float8_e4m3fn)
        s = torch.rand(n//128, k//128, device="cuda")*.3 + .05
        return q, s
    wa, sa = weight(g.qkv_a_width, g.hidden)
    wb, sb = weight(heads*g.qk_nope_dim, g.q_lora_rank)
    wo, so = weight(g.hidden, heads*g.v_head_dim)
    x, uk = rand(rows, g.hidden), rand(heads, 512, 256)
    qn, kn = rand(g.q_lora_rank, scale=1), rand(512, scale=1)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    cache = torch.zeros(rows, 528, device="cuda", dtype=torch.uint8)
    query, resid = empty(rows, heads, 512), empty(rows, g.q_lora_rank)
    ps, os = scratch(producer, cap), scratch(output, cap)
    bs, ops = scratch(qb, cap), scratch(op, cap)
    attn, uv = rand(rows, heads, 512), rand(heads, 256, 512)
    expected, actual = empty(rows, g.hidden), empty(rows, g.hidden)
    projected = empty(rows, heads*g.qk_nope_dim)
    def run():
        producer.launch(x, slots, wa, sa, qn, kn, wb, sb, uk, cache, query,
                        resid, ps, scalars=(rows, flag))
        qb.launch(resid, wb, sb, projected, bs, scalars=(rows, flag))
        # Fused scratch starts with qkv_a followed by exactly these q_b rows.
        offset = ((rows*g.qkv_a_width*2 + 1023)//1024)*1024
        fused_q = ps[offset:offset+projected.numel()*2].view(torch.bfloat16).view_as(projected)
        output.launch(attn, uv, wo, so, expected, os, scalars=(rows, flag))
        fused_values = os[:rows*heads*g.v_head_dim*2].view(torch.bfloat16)
        op.launch(fused_values, wo, so, actual, ops, scalars=(rows, flag))
        return fused_q
    reference = run()
    same(projected, reference); same(actual, expected)
    torch.cuda.synchronize()
    with kernel_resolution_guard("standalone Flash FP8 precompiled"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay(); graph.replay()
        same(projected, reference); same(actual, expected)
