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
