"""Bitwise projection seams against the unchanged fused BF16 Kimi programs."""
import pytest
import torch

from ..conftest import require_b12x


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    require_b12x()
    torch.manual_seed(273)


def rand(*shape, scale=0.025, dtype=torch.bfloat16):
    return (torch.randn(shape, device="cuda") * scale).to(dtype)


def empty(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, device="cuda", dtype=dtype)


def scratch(p, rows):
    return empty(max(1, p.scratch_bytes(rows)["scratch"]), dtype=torch.uint8)


def same(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def projection(n, k):
    from b12x.integration.cuteafd._glm_kernels import glm_projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    return compile_program(glm_projection(n, k), name="split_test_projection",
        operands=tuple(Operand(name, torch.bfloat16, "[]", mode)
                       for name, mode in (("x", "in"), ("w", "in"), ("out", "out"))),
        scalars=(Scalar("rows"),), key=(n, k))


@pytest.mark.parametrize("rows,decode", [(1, True), (16, True), (64, True), (129, False), (4096, False)])
@pytest.mark.parametrize("heads", [48, 96])
def test_kda_split_bitwise(rows, decode, heads):
    from b12x.integration.cuteafd.glmf import compile_glmf_kda_aot, compile_glmf_kda_commit_aot
    from b12x.integration.cuteafd.kimi_split import compile_kimi_kda_core_aot
    from b12x.integration.cuteafd._kda_contract import KDAGeometry
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout
    g = KDAGeometry(hidden=7168, kda_heads=heads)
    cap, d, h, p = max(64, rows), g.kda_width, g.hidden, g.kda_in_width
    fused = compile_glmf_kda_aot(g, max_rows=cap, decode=decode, output_kind="norm")
    split = compile_kimi_kda_core_aot(g, max_rows=cap, decode=decode)
    x, w = rand(rows, h), rand(p, h)
    wf, cw = rand(1, d, 128), rand(3*d, 4, dtype=torch.float32)
    al, dt, norm = rand(heads, dtype=torch.float32), rand(d, dtype=torch.float32), rand(128, scale=1, dtype=torch.float32)
    proj = empty(rows, p)
    projection(p, h).launch(x, w, proj, scalars=(rows,))
    q, k, v = [proj[:, i*d:(i+1)*d].contiguous() for i in range(3)]
    gate = proj[:, 3*d+128:4*d+128].contiguous()
    fab = torch.cat((w[3*d:3*d+128], w[4*d+128:])).contiguous()
    wo = empty(h, d)
    state0, conv0 = rand(2, heads, 128, 128, dtype=torch.float32), rand(2, 3, 3*d)
    slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first = torch.zeros_like(slots)
    if decode and rows > 1:
        slots[rows//2:] = 1
        first[rows//2:] = rows//2
        slots[-1] = -1
        first[-1] = rows-1
    out, ref = empty(rows, d), empty(rows, d)
    state, conv = state0.clone(), conv0.clone()
    sr, cr = state0.clone(), conv0.clone()
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
        split.launch(x, q, k, v, gate, fab, wf, cw, al, dt, norm, conv, state,
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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(0)
    graph.replay(); graph.replay()
    same(out, ref); same(state, sr); same(conv, cr)
    assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0


@pytest.mark.parametrize("rows", [1, 16, 64, 4096])
@pytest.mark.parametrize("heads", [48, 96])
@pytest.mark.parametrize("kv_stride", [576, 640])
@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_mla_producer_split_bitwise(rows, heads, kv_stride, kv_cache):
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_mla_producer_aot
    from b12x.integration.cuteafd.kimi_split import compile_kimi_mla_rank_norm_aot, compile_kimi_mla_query_aot
    g = KimiMlaGeometry(heads=heads, kv_cache=kv_cache)
    fused = compile_kimi_mla_producer_aot(g, max_rows=4096)
    norm = compile_kimi_mla_rank_norm_aot(g, max_rows=4096, kv_stride=kv_stride)
    query = compile_kimi_mla_query_aot(g, max_rows=4096)
    x, w, wb = rand(rows, 7168), rand(2112, 7168), rand(heads*192, 1536)
    uk, qn, kn = rand(heads, 512, 128), rand(1536, scale=1), rand(512, scale=1)
    slots = torch.arange(rows-1, -1, -1, dtype=torch.int64, device="cuda")
    slots[-1] = -1
    cache = (torch.full((max(rows, 1), 656), 173, device="cuda", dtype=torch.uint8)
             if kv_cache == "fp8" else rand(max(rows, 1), 576))
    cref = cache.clone()
    qa, ka = empty(rows, 1536), empty(rows, kv_stride)
    packed = empty(rows, 2112)
    projection(2112, 7168).launch(x, w, packed, scalars=(rows,))
    qa.copy_(packed[:, :1536]); ka[:, :576].copy_(packed[:, 1536:])
    ka[:, 576:] = float("nan")
    q, qr, qb = empty(rows, heads, 576), empty(rows, 1536), empty(rows, heads*192)
    ref, resid = torch.empty_like(q), torch.empty_like(qr)
    fs, ns = scratch(fused, rows), scratch(norm, rows)
    bp = projection(heads*192, 1536)
    def run():
        fused.launch(x, slots, w, qn, kn, wb, uk, cref, ref, resid, fs, scalars=(rows,))
        norm.launch(qa, ka, qn, kn, slots, qr, cache, ns, scalars=(rows,))
        bp.launch(qr, wb, qb, scalars=(rows,))
        query.launch(qb, uk, q, scalars=(rows,))
    run()
    same(qr, resid); same(cache, cref); same(q, ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay(); graph.replay()
    same(qr, resid); same(cache, cref); same(q, ref)
    assert torch.isfinite(q).all() and torch.count_nonzero(q) > 0


@pytest.mark.parametrize("rows", [1, 16, 64, 4096])
@pytest.mark.parametrize("heads", [48, 96])
def test_mla_output_split_bitwise(rows, heads):
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_mla_output_aot
    from b12x.integration.cuteafd.kimi_split import compile_kimi_mla_gate_output_aot
    g = KimiMlaGeometry(heads=heads)
    fused = compile_kimi_mla_output_aot(g, max_rows=4096)
    split = compile_kimi_mla_gate_output_aot(g, max_rows=4096)
    x, attn = rand(rows, 7168), rand(rows, heads, 512)
    uv, wg, wo = rand(heads, 128, 512), rand(heads*128, 7168), rand(7168, heads*128)
    gates, values, out, ref = empty(rows, heads*128), empty(rows, heads*128), empty(rows, 7168), empty(rows, 7168)
    projection(heads*128, 7168).launch(x, wg, gates, scalars=(rows,))
    fs, op = scratch(fused, rows), projection(7168, heads*128)
    def run():
        fused.launch(attn, x, uv, wg, wo, ref, fs, scalars=(rows,))
        split.launch(attn, uv, gates, values, scalars=(rows,))
        op.launch(values, wo, out, scalars=(rows,))
    run()
    same(values, fs[:rows*heads*128*2].view(torch.bfloat16).reshape_as(values))
    same(out, ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay(); graph.replay()
    same(out, ref)
    assert torch.isfinite(values).all() and torch.count_nonzero(values) > 0
