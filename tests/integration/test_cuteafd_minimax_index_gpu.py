"""MiniMax decode index scoring on the GPU: the balanced page schedule with 16-byte page
loads selects exactly what the former per-row schedule with scalar byte loads selected.

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
def test_balanced_vector_scoring_selects_exactly_what_the_row_schedule_did(programs, case):
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
