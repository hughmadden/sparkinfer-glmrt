"""MiniMax decode index scoring on the GPU: balanced and per-row page schedules
select exactly the same blocks using the current shared scalar packed-page loads.

Both programs score every live page whole in one CTA, so the per-page head maxima, the
block scores and the top-16 selections must be the same bits for single rows, long rows
and mixed batches, with page tables that scatter rows over a pool wider than the table.
(No postponed annotations: tensors are built at run time only.)
"""
import pytest

from ..conftest import require_b12x

torch = pytest.importorskip("torch")

MAX_ROWS, MAX_PAGES, POOL = 64, 1024, 1100
CASES = {
    "one_token": [1],
    "one_page_and_a_bit": [65],
    "short": [700],
    "long": [15403],
    "full_table": [MAX_PAGES * 64 - 3],
    "mixed": [5, 300, 64, 129, 4096, 20000, 1],
    "c16": [700 + 37 * i for i in range(16)],
    "c64": [1 + (i * 977) % 30000 for i in range(64)],
}


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot

    new = compile_minimax_index_aot(max_rows=MAX_ROWS, max_pages=MAX_PAGES, mode="decode")
    reference = compile_minimax_index_aot(max_rows=MAX_ROWS, max_pages=MAX_PAGES, mode="decode", reference=True)
    return new, reference


def _cache(generator):
    keys = (torch.rand((POOL, 64, 128), generator=generator, device="cuda") * 4 - 2).to(torch.float8_e4m3fn)
    scales = torch.rand((POOL, 64), generator=generator, device="cuda") + 0.5
    cache = torch.empty((POOL, 8448), dtype=torch.uint8, device="cuda")
    cache[:, :8192] = keys.view(torch.uint8).reshape(POOL, 8192)
    cache[:, 8192:] = scales.contiguous().view(torch.uint8).reshape(POOL, 256)
    return cache


@pytest.mark.parametrize("case", list(CASES))
def test_balanced_scoring_selects_exactly_what_the_row_schedule_did(programs, case):
    from b12x.integration.cuteafd.minimax_indexer import _layout

    lengths = CASES[case]
    rows = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(rows * 7919 + lengths[0])
    cache = _cache(generator)
    q = (torch.rand((rows, 4, 128), generator=generator, device="cuda") * 4 - 2).to(torch.float8_e4m3fn)
    q_scale = torch.rand((rows, 4), generator=generator, device="cuda") + 0.5
    widths = [(n + 63) // 64 for n in lengths]
    stride = max(widths) + 3  # a table pitch wider than its live width
    table = torch.full((rows, stride), -1, dtype=torch.int32)
    picker = torch.Generator().manual_seed(len(lengths))
    for row, width in enumerate(widths):
        table[row, :width] = torch.randperm(POOL, generator=picker)[:width].to(torch.int32)
    table = table.cuda().contiguous()
    cache_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _, offsets, nbytes = _layout(MAX_ROWS, MAX_PAGES, "decode")
    results = []
    for program in programs:
        q2k = torch.full((4, MAX_ROWS, 16), -7, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        program.launch(q.view(torch.uint8), q_scale, cache, table, cache_lengths, q2k, scratch,
                       scalars=(rows, max(widths), stride, POOL))
        torch.cuda.synchronize()
        pages = scratch[offsets["pages"]:offsets["pages"] + 4 * rows * MAX_PAGES * 4]
        scores = scratch[offsets["scores"]:offsets["scores"] + 4 * rows * ((MAX_PAGES + 1) // 2) * 4]
        results.append((q2k[:, :rows].clone(), pages.clone(), scores.clone()))
    (new_q2k, new_pages, new_scores), (ref_q2k, ref_pages, ref_scores) = results
    assert torch.equal(new_pages, ref_pages), "per-page head maxima differ"
    assert torch.equal(new_scores, ref_scores), "block scores differ"
    assert torch.equal(new_q2k, ref_q2k), "selections differ"
    for row, length in enumerate(lengths):
        own = (length - 1) // 128
        assert (new_q2k[:, row] == own).any(dim=-1).all(), f"row {row} lost its own block"


@pytest.mark.parametrize("tied", [False, True])
def test_signed_heads_and_low_id_ties_match_independent_oracle(programs, tied):
    program = programs[0]
    blocks, length = 21, 20 * 128 + 17
    pages = (length + 63) // 64
    cache = torch.zeros((pages, 8448), dtype=torch.uint8, device="cuda")
    block_values = torch.ones(blocks, device="cuda") if tied else torch.arange(
        -10, 11, dtype=torch.float32, device="cuda")
    keys = block_values.repeat_interleave(128)[:pages * 64, None].expand(-1, 128)
    # The final physical page is padded, but only its first 17 keys are causal.
    keys = torch.nn.functional.pad(keys, (0, 0, 0, pages * 64 - keys.shape[0]))
    cache[:, :8192] = keys.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).reshape(pages, 8192)
    cache[:, 8192:] = torch.ones((pages, 64), device="cuda").view(torch.uint8).reshape(pages, 256)
    signs = torch.tensor([1., -1., 1., -1.], device="cuda")
    q = signs[:, None].expand(4, 128).contiguous().to(torch.float8_e4m3fn)[None]
    scales = torch.ones((1, 4), device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    output = torch.full((4, MAX_ROWS, 16), -7, dtype=torch.int32, device="cuda")
    scratch = torch.empty(program.scratch_bytes(MAX_ROWS)["scratch"], dtype=torch.uint8, device="cuda")

    def launch():
        program.launch(q.view(torch.uint8), scales, cache, table, lengths, output, scratch,
                       scalars=(1, pages, pages, pages))

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Mutate input under stable pointers: replay must compute, not reuse old scores.
    for multiplier in [1., -1.]:
        q.copy_((signs[:, None] * multiplier).expand(4, 128).to(torch.float8_e4m3fn)[None])
        graph.replay()
        actual = output[:, 0].cpu().tolist()
        for head, sign in enumerate([1., -1., 1., -1.]):
            scores = [(float(block_values[b]) * sign * multiplier, b) for b in range(blocks - 1)]
            picked = [b for _, b in sorted(scores, key=lambda item: (-item[0], item[1]))[:15]]
            expected = sorted(picked + [blocks - 1])
            assert actual[head] == expected
        assert (output[:, 1:] == -7).all(), "inactive capacity rows were overwritten"


@pytest.mark.parametrize("tied", [False, True])
def test_contiguous_prefill_matches_independent_signed_block_oracle(tied):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot

    capacity, max_pages = 16, 96
    program = compile_minimax_index_aot(max_rows=capacity, max_pages=max_pages, mode="prefill")
    extent = max_pages * 64
    generator = torch.Generator(device="cuda").manual_seed(20261112)
    keys = torch.randn((extent, 128), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    if tied:
        keys.fill_(1)
    key_scales = torch.rand(extent, generator=generator, device="cuda") + .5
    if tied:
        key_scales.fill_(1)
    query = torch.empty((capacity, 4, 128), dtype=torch.float8_e4m3fn, device="cuda")
    query_scales = torch.rand((capacity, 4), generator=generator, device="cuda") + .5
    starts = torch.empty(capacity, dtype=torch.int32, device="cuda")
    ends = torch.empty_like(starts)
    output = torch.full((4, capacity, 16), -7, dtype=torch.int32, device="cuda")
    scratch = torch.empty(program.scratch_bytes(capacity)["scratch"], dtype=torch.uint8, device="cuda")
    cases = [([0], [65]), ([0, 2816, 2816, 0], [2700, 2817, 2816 + 129, 257])]
    signs = torch.tensor([1., -1., 1., -1.], device="cuda")
    for first, last in cases:
        rows, k_rows = len(first), max(last)
        query.copy_(signs[None, :, None].expand(capacity, 4, 128).to(torch.float8_e4m3fn))
        starts[:rows] = torch.tensor(first, dtype=torch.int32, device="cuda")
        ends[:rows] = torch.tensor(last, dtype=torch.int32, device="cuda")
        output.fill_(-7)

        def launch():
            program.launch(query, query_scales, keys, key_scales, starts, ends, output, scratch,
                           scalars=(rows, k_rows))

        with kernel_resolution_guard("MiniMax prefill index live rows and graph replay"):
            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            for _ in range(2):
                query.copy_((-query.float()).to(torch.float8_e4m3fn))
                graph.replay()
                torch.cuda.synchronize()
                actual = output[:, :rows].cpu().tolist()
                for row, (begin, end) in enumerate(zip(first, last)):
                    k = keys[begin:end].float() * key_scales[begin:end, None]
                    scores = (query[row].float() @ k.T) * query_scales[row, :, None] * 128 ** -.5
                    block_scores = torch.stack([scores[:, p:p + 128].amax(-1)
                                                for p in range(0, end - begin, 128)], dim=-1).cpu().tolist()
                    for head in range(4):
                        own = len(block_scores[head]) - 1
                        history = sorted(range(own), key=lambda b: (-block_scores[head][b], b))[:15]
                        picked = sorted(history + [own])
                        expected = picked + [-1] * (16 - len(picked))
                        assert actual[head][row] == expected, (row, head, actual[head][row], expected)
                assert (output[:, rows:] == -7).all(), "inactive index rows were overwritten"
