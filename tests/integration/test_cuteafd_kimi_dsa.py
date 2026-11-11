"""K3.1 index post against the pinned non-MX vLLM QDQ arithmetic."""
import pytest
import torch

from ..conftest import require_b12x


def qdq(x, floor):
    scale = torch.exp2(torch.ceil(torch.log2(x.float().abs().amax(-1, keepdim=True).clamp_min(floor) / 448)))
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("rows", [1, 17, 64, 129])
def test_kimi_index_post_qdq_graph_and_untouched_slots(rows):
    require_b12x()
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_index_post_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    torch.manual_seed(731 + rows)
    program = compile_kimi_index_post_aot(max_rows=max(64, rows))
    iq = torch.randn(rows, 32, 128, device="cuda").bfloat16()
    kw = torch.randn(rows, 160, device="cuda").bfloat16()
    # Zero/sub-floor queries, as well as ordinary amax values, exercise both floors.
    iq[0, 0].zero_()
    iq[0, 1].fill_(1e-12)
    norm = torch.randn(128, device="cuda").bfloat16()
    bias = torch.randn(128, device="cuda").bfloat16()
    positions = torch.zeros(rows, device="cuda", dtype=torch.int64)
    slots = torch.randperm((rows + 63) // 64 * 64, device="cuda")[:rows].long()
    slots[-1] = -1
    pages = (rows + 63) // 64
    cache = torch.full((pages, 8448), 123, device="cuda", dtype=torch.uint8)
    q8 = torch.empty_like(iq, dtype=torch.float8_e4m3fn)
    weights = torch.empty(rows, 32, device="cuda")
    cs = torch.zeros(64, device="cuda")
    args = (iq, kw, positions, slots, cs, norm, bias, q8, weights, cache)
    program.launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    q_ref, q_scale = qdq(iq, 1e-10)
    raw = kw[:, :128].float()
    mean = raw.mean(-1, keepdim=True)
    key = ((raw - mean) * torch.rsqrt((raw - mean).square().mean(-1, keepdim=True) + 1e-5)
           * norm.float() + bias.float()).bfloat16()
    k_ref, k_scale = qdq(key, 1e-4)
    folded = kw[:, 128:].float() * (32 ** -0.5 * 128 ** -0.5) * q_scale.squeeze(-1)
    assert torch.equal(q8.view(torch.uint8), q_ref.view(torch.uint8))
    torch.testing.assert_close(weights, folded, rtol=1e-6, atol=1e-12)
    expected = torch.full_like(cache, 123)
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, 64)
        expected[page, offset * 128:(offset + 1) * 128] = k_ref[row].view(torch.uint8)
        expected[page, 8192 + offset * 4:8196 + offset * 4] = k_scale[row].view(torch.uint8)
    assert torch.equal(cache, expected)
    # The serving cache's scales are powers of two stored as full FP32 slots.
    live = slots >= 0
    s = k_scale[live]
    assert torch.equal(torch.log2(s), torch.log2(s).round())
    before = cache.clone()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("K3.1 index post is precompiled"):
        with torch.cuda.graph(graph):
            program.launch(*args, scalars=(rows,))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache, before)
    assert torch.equal(q8.view(torch.uint8), q_ref.view(torch.uint8))


def test_kimi_index_key_floor_and_large_physical_page():
    require_b12x()
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_index_post_aot

    # Only the final page is touched: the >2GiB offset must remain Int64.
    pages = (2 ** 31 // 8448) + 2
    cache = torch.empty((pages, 8448), device="cuda", dtype=torch.uint8)
    cache[-1].fill_(123)
    p = compile_kimi_index_post_aot(max_rows=64)
    iq = torch.zeros(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    kw = torch.zeros(1, 160, device="cuda", dtype=torch.bfloat16)
    pos = torch.zeros(1, device="cuda", dtype=torch.int64)
    slots = torch.tensor([(pages - 1) * 64 + 63], device="cuda", dtype=torch.int64)
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros_like(norm)
    q = torch.empty_like(iq, dtype=torch.float8_e4m3fn)
    weights = torch.empty(1, 32, device="cuda")
    p.launch(iq, kw, pos, slots, torch.zeros(64, device="cuda"), norm, bias, q, weights, cache, scalars=(1,))
    torch.cuda.synchronize()
    expected = torch.full((8448,), 123, device="cuda", dtype=torch.uint8)
    expected[63 * 128:64 * 128] = 0
    scale = torch.tensor([2.0 ** -22], device="cuda", dtype=torch.float32)
    expected[8192 + 63 * 4:] = scale.view(torch.uint8)
    assert torch.equal(cache[-1], expected)
    assert torch.count_nonzero(q.float()) == 0
    assert torch.count_nonzero(weights) == 0


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("ties", [False, True])
def test_kimi_index_topk_matches_qdq_selection(mode, ties):
    require_b12x()
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_index_topk_aot

    torch.manual_seed(7319)
    lengths = torch.tensor([1, 777, 2048, 2049, 3073], device="cuda", dtype=torch.int32)
    rows, width, pool = len(lengths), 49, 53
    table = torch.randperm(pool, device="cuda")[:width].int()
    stride = width + 3 if mode == "decode" else 0
    if mode == "decode":
        page_table = torch.full((rows, stride), -1, device="cuda", dtype=torch.int32)
        page_table[:, :width] = table
    else:
        page_table = table
    raw_keys = torch.randn(pool, 64, 128, device="cuda").bfloat16()
    raw_queries = torch.randn(rows, 32, 128, device="cuda").bfloat16()
    if ties:
        raw_keys.zero_()
    keys, scales = qdq(raw_keys, 1e-4)
    queries, query_scales = qdq(raw_queries, 1e-10)
    weights = torch.randn(rows, 32, device="cuda") * query_scales.squeeze(-1) / 64
    cache = torch.empty(pool, 8448, device="cuda", dtype=torch.uint8)
    cache[:, :8192] = keys.view(torch.uint8).reshape(pool, 8192)
    cache[:, 8192:] = scales.view(torch.uint8).reshape(pool, 256)
    program = compile_kimi_index_topk_aot(max_rows=64, max_pages=width, mode=mode)
    output = torch.full((rows, 2048), -7, device="cuda", dtype=torch.int32)
    scratch = torch.zeros(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    program.launch(queries, weights, cache, page_table, lengths, output, scratch,
                   scalars=(rows, width, stride))
    torch.cuda.synchronize()
    for row, length in enumerate(lengths.tolist()):
        logical = torch.arange(length, device="cuda")
        physical = table[logical // 64].long() * 64 + logical % 64
        key = keys.reshape(-1, 128)[physical].float()
        scale = scales.reshape(-1)[physical]
        scores = (torch.einsum("hd,jd->hj", queries[row].float(), key).relu()
                  * weights[row, :, None]).sum(0) * scale
        selected = torch.argsort(scores, descending=True, stable=True)[:2048]
        # Canonical order is ascending logical token, mapped to physical slots.
        expected = physical[torch.sort(selected).values].int()
        valid = output[row, :len(expected)]
        assert torch.equal(valid, expected), (mode, ties, length, program.geometry)
        assert torch.all(output[row, len(expected):] == -1)


@pytest.mark.parametrize("rows", [1, 17, 64])
def test_kimi_native_index_projections_and_post(rows):
    require_b12x()
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_index_producer_aot, compile_kimi_index_post_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    torch.manual_seed(7331)
    p = compile_kimi_index_producer_aot(max_rows=64)
    x = torch.randn(rows, 7168, device="cuda").bfloat16()
    q_resid = (torch.randn(rows, 1536, device="cuda") * 0.25).bfloat16()
    wq = (torch.randn(4096, 1536, device="cuda") / 40).bfloat16()
    wk = (torch.randn(160, 7168, device="cuda") / 85).bfloat16()
    norm = torch.randn(128, device="cuda").bfloat16()
    bias = torch.randn(128, device="cuda").bfloat16()
    positions = torch.zeros(rows, device="cuda", dtype=torch.int64)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    cs = torch.zeros(64, device="cuda")
    cache = torch.full((1, 8448), 123, device="cuda", dtype=torch.uint8)
    query = torch.empty(rows, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.empty(rows, 32, device="cuda")
    scratch = torch.empty(p.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    args = (x, q_resid, positions, slots, cs, wq, wk, norm, bias, cache, query, weights, scratch)
    p.launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    # Read the shared producer's native projected rows, before exact QDQ.
    iq_bytes = rows * 4096 * 2
    kw_start = (iq_bytes + 1023) // 1024 * 1024
    iq = scratch[:iq_bytes].view(torch.bfloat16).reshape(rows, 32, 128)
    kw = scratch[kw_start:kw_start + rows * 160 * 2].view(torch.bfloat16).reshape(rows, 160)
    torch.testing.assert_close(iq.float().flatten(1), q_resid.float() @ wq.float().T, rtol=0.01, atol=0.004)
    torch.testing.assert_close(kw.float(), x.float() @ wk.float().T, rtol=0.01, atol=0.016)
    expected_cache = torch.full_like(cache, 123)
    expected_query = torch.empty_like(query)
    expected_weights = torch.empty_like(weights)
    post = compile_kimi_index_post_aot(max_rows=64)
    post.launch(iq, kw, positions, slots, cs, norm, bias, expected_query, expected_weights,
                expected_cache, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(cache, expected_cache)
    assert torch.equal(query.view(torch.uint8), expected_query.view(torch.uint8))
    assert torch.equal(weights, expected_weights)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("native Kimi indexer precompiled"), torch.cuda.graph(graph):
        p.launch(*args, scalars=(rows,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache, expected_cache)
    assert torch.equal(query.view(torch.uint8), expected_query.view(torch.uint8))
    assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize("cache_format", ["bf16", "fp8"])
@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("heads", [48, 96])
def test_kimi_sparse_mla_selected_tokens(cache_format, rows, heads):
    require_b12x()
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_sparse_mla_aot
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry
    from b12x._lib.runtime_control import kernel_resolution_guard

    torch.manual_seed(7321)
    g = KimiMlaGeometry(name="kimi_k31_mla", heads=heads, norm_eps=1e-5, kv_cache=cache_format)
    p = compile_kimi_sparse_mla_aot(g, max_rows=64)
    tokens = 4096
    records = torch.randn(tokens, 576, device="cuda").bfloat16()
    if cache_format == "bf16":
        cache = records.view(torch.uint8)
    else:
        # Native compact records: four arbitrary FP32 group scales and BF16 tail.
        group = records[:, :512].float().reshape(tokens, 4, 128)
        scale = group.abs().amax(-1).clamp_min(1e-6) / 448
        quant = (group / scale[..., None]).to(torch.float8_e4m3fn)
        cache = torch.zeros(tokens, 656, device="cuda", dtype=torch.uint8)
        cache[:, :512] = quant.view(torch.uint8).reshape(tokens, 512)
        cache[:, 512:528] = scale.view(torch.uint8)
        cache[:, 528:] = records[:, 512:].contiguous().view(torch.uint8)
        records[:, :512] = (quant.float() * scale[..., None]).reshape(tokens, 512).bfloat16()
    lengths = torch.tensor(([1, 63, 777, 2048, 2047][:rows]), device="cuda", dtype=torch.int32)
    indices = torch.full((rows, 2048), -1, device="cuda", dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        indices[row, :length] = torch.sort(torch.randperm(tokens - 1, device="cuda")[:length] + 1).values.int()
    query = torch.randn(rows, heads, 576, device="cuda").bfloat16()
    offsets = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(rows, heads, device="cuda")
    scratch = torch.empty(p.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    args = (query, cache, indices, lengths, offsets, output, lse, scratch)
    scalars = (rows, rows, 2048, p.geometry["num_splits"])
    p.launch(*args, scalars=scalars)
    torch.cuda.synchronize()
    for row, length in enumerate(lengths.tolist()):
        kv = records[indices[row, :length].long()].float()
        scores = query[row].float() @ kv.T * (192 ** -0.5)
        expected = scores.softmax(-1) @ kv[:, :512]
        torch.testing.assert_close(output[row].float(), expected, rtol=0.035, atol=0.012)
        torch.testing.assert_close(lse[row], scores.logsumexp(-1), rtol=3e-4, atol=3e-4)
    before = output.clone()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("selected Kimi MLA precompiled"), torch.cuda.graph(graph):
        p.launch(*args, scalars=scalars)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, before)
