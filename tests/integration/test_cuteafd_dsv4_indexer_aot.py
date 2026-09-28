"""cuteafd DSV4 C4 index top-k AOT programs vs the prepared attention.dsa_indexer path."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import check_export

PAGE_BYTES = 8448
# First physical page whose byte offset exceeds 2^31 (AGENTS.md big-pid rule).
HIGH_PAGE = (2**31) // PAGE_BYTES + 5


def _pack_pages(num_pages: int, gen: torch.Generator, device) -> torch.Tensor:
    from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference

    rows = torch.randn((num_pages * 64, 128), generator=gen)
    rows *= torch.rand((num_pages * 64, 1), generator=gen) * 2 + 0.25
    return pack_index_k_cache_reference(rows.to(device))


def _query(rows: int, gen: torch.Generator, device):
    q = (torch.randn((rows, 64, 128), generator=gen) * 1.5).to(torch.float8_e4m3fn).to(device)
    w = (torch.randn((rows, 64), generator=gen) * 0.02).bfloat16().float().to(device)
    return q.contiguous(), w.contiguous()


class _Pool:
    """A paged index cache with chosen physical page ids (optionally past 2^31 bytes)."""

    def __init__(self, pool_pages: int, device):
        self.cache = torch.zeros((pool_pages, PAGE_BYTES), dtype=torch.uint8, device=device)

    def place(self, page_ids: torch.Tensor, gen: torch.Generator):
        packed = _pack_pages(int(page_ids.numel()), gen, self.cache.device)
        self.cache[page_ids.long()] = packed


def _prepared(mode, topk, q, w, cache, table, lengths, table_width):
    """The prototype's prepared plan: capacity = these live tensors."""
    from b12x.attention import dsa_indexer

    from ._cuteafd import prepare

    rows = q.shape[0]
    max_rows, max_pages = rows, table_width
    output = torch.full((rows, topk), -7, dtype=torch.int32, device=q.device)
    operands = dict(q_fp8=q, query_weights=w, index_k_cache=cache, page_table=table,
                    cache_lengths=lengths,
                    active_width=torch.tensor([table_width * 64], dtype=torch.int32, device=q.device),
                    output_indices=output)
    caps = dsa_indexer.Caps(device=q.device, num_q_heads=64, max_q_rows=max_rows,
                            max_page_table_width=max_pages, topk=topk, mode=mode,
                            output_index_space="physical")
    plan = prepare(dsa_indexer.plan(caps, invocation=dsa_indexer.invocation_from_tensors(caps, **operands)),
                   f"test.dsa.{mode}.{max_rows}.{max_pages}.{rows}.{table_width}")
    spec = plan.scratch_specs()[0]
    binding = dsa_indexer.bind(plan, scratch=torch.zeros(spec.shape, dtype=spec.dtype, device=q.device),
                               **operands)
    dsa_indexer.run(binding)
    return output


def _aot(program, q, w, cache, table_ptr, lengths, table_width, table_stride, scratch=None):
    rows = q.shape[0]
    topk = program.geometry["topk"]
    output = torch.full((rows, topk), -7, dtype=torch.int32, device=q.device)
    if scratch is None:
        scratch = torch.zeros((program.scratch_bytes(rows)["scratch"],), dtype=torch.uint8, device=q.device)
    program.launch(q, w, cache, table_ptr, lengths, output, scratch,
                   scalars=(rows, table_width, table_stride))
    return output


def _sorted(x):
    return torch.sort(x, dim=1).values


def _assert_same_selection(actual, expected, lengths, topk):
    torch.cuda.synchronize()
    a = torch.sort(actual, dim=1).values
    e = torch.sort(expected, dim=1).values
    mismatch = (a != e).any(dim=1)
    assert not bool(mismatch.any()), (
        f"{int(mismatch.sum())} rows differ; first {int(mismatch.nonzero()[0])}")
    # -1 padding exactly where fewer than topk index rows are visible.
    pads = (actual == -1).sum(dim=1)
    assert torch.equal(pads, (topk - lengths.clamp(max=topk)).to(pads.dtype))


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd.dsv4_indexer import compile_dsv4_index_topk_aot as build

    with exportable_compilation():
        return {
            "prefill": build(FLASH, max_rows=512, max_pages=40, mode="prefill"),
            "prefill_chunks": build(FLASH, max_rows=64, max_pages=1100, mode="prefill"),
            "decode_fused": build(FLASH, max_rows=16, max_pages=300, mode="decode"),
            "decode_tiled": build(FLASH, max_rows=48, max_pages=300, mode="decode"),
            "pro_prefill": build(PRO, max_rows=256, max_pages=40, mode="prefill"),
            "pro_decode": build(PRO, max_rows=4, max_pages=300, mode="decode"),
        }


def test_routes_match_the_prepared_plan(programs):
    assert programs["prefill"].geometry["route"] == "packed_contiguous"
    assert programs["prefill_chunks"].geometry["supertile_tokens"] < 1100 * 64
    assert programs["decode_fused"].geometry["route"] == "paged_fused"
    assert programs["decode_tiled"].geometry["route"] == "paged_tiled"
    assert programs["pro_decode"].geometry["route"] == "paged_fused"


@pytest.fixture(scope="module")
def pool():
    require_b12x()
    return _Pool(HIGH_PAGE + 2400, torch.device("cuda"))


def _prefill_case(pool, rows, width, seed, *, high=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    device = pool.cache.device
    base = HIGH_PAGE if high else 0
    table = (torch.randperm(2 * width, generator=gen)[:width] + base).to(torch.int32).to(device)
    pool.place(table, gen)
    # model.py causal visibility of completed C4 groups: row t sees (t+1)//4 index rows;
    # rows are the tail of a long prompt so they see most of the table.
    tail = torch.arange(rows) + max(0, width * 64 * 4 - rows)
    lengths = ((tail + 1) // 4).clamp(max=width * 64).to(torch.int32).to(device)
    q, w = _query(rows, gen, device)
    return table, lengths, q, w


@pytest.mark.parametrize(("key", "rows", "width", "high"), [
    ("prefill", 1, 40, False), ("prefill", 300, 40, False), ("prefill", 512, 23, True),
    ("prefill_chunks", 64, 1100, False), ("prefill_chunks", 7, 600, True),
    ("pro_prefill", 256, 40, False),
])
def test_prefill_matches_prepared(programs, pool, key, rows, width, high):
    program = programs[key]
    table, lengths, q, w = _prefill_case(pool, rows, width, 17 + rows + width, high=high)
    topk = program.geometry["topk"]
    expected = _prepared("prefill", topk, q, w, pool.cache, table[None, :].expand(rows, width), lengths, width)
    actual = _aot(program, q, w, pool.cache, table, lengths, width, 0)
    _assert_same_selection(actual, expected, lengths, topk)


def _decode_case(pool, rows, width, seed, *, high=False, stride=None):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    device = pool.cache.device
    base = HIGH_PAGE if high else 0
    pages = torch.randperm(width + 600, generator=gen)[: width + 64].to(torch.int32) + base
    pool.place(pages.to(device), gen)
    table = torch.stack([pages[torch.randperm(pages.numel(), generator=gen)[:width]] for _ in range(rows)])
    lengths = torch.randint(1, width * 64 + 1, (rows,), generator=gen, dtype=torch.int32)
    lengths[0] = min(300, width * 64)  # a row with fewer than topk visible rows
    stride = width if stride is None else stride
    padded = torch.full((rows, stride), -1, dtype=torch.int32)
    padded[:, :width] = table
    q, w = _query(rows, gen, device)
    return table.to(device).contiguous(), padded.to(device).contiguous(), lengths.to(device), q, w


@pytest.mark.parametrize(("key", "rows", "width", "high", "stride"), [
    ("decode_fused", 1, 300, False, None), ("decode_fused", 5, 120, True, 301),
    ("decode_fused", 16, 300, False, None),
    ("decode_tiled", 17, 300, False, None), ("decode_tiled", 48, 77, True, 128),
    ("decode_tiled", 3, 300, False, None),
    ("pro_decode", 4, 300, False, None),
])
def test_decode_matches_prepared(programs, pool, key, rows, width, high, stride):
    program = programs[key]
    table, padded, lengths, q, w = _decode_case(pool, rows, width, 91 + rows, high=high, stride=stride)
    topk = program.geometry["topk"]
    expected = _prepared("decode", topk, q, w, pool.cache, table, lengths, width)
    actual = _aot(program, q, w, pool.cache, padded, lengths, width, padded.shape[1])
    _assert_same_selection(actual, expected, lengths, topk)


def test_decode_matches_torch_reference(programs, pool):
    """Independent oracle: dense logits reference + torch top-k (ties excluded)."""
    from b12x.attention.dsa_indexer.reference import paged_decode_logits_reference

    rows, width = 6, 64
    table, padded, lengths, q, w = _decode_case(pool, rows, width, 5)
    actual = _aot(programs["decode_fused"], q, w, pool.cache, padded, lengths, width, width)
    logits = paged_decode_logits_reference(
        q_fp8=q, weights=w, index_k_cache=pool.cache, real_page_table=table,
        query_row_to_batch=torch.arange(rows, device=q.device, dtype=torch.int32),
        seqlens_per_query=lengths)
    torch.cuda.synchronize()
    for row in range(rows):
        n = int(lengths[row])
        k = min(512, n)
        values, logical = torch.topk(logits[row, :n], k)
        # Skip candidates tied with the k-th score (either may be selected).
        keep = values > values[-1] + 1e-3 * values.abs().max()
        physical = table[row, logical // 64].long() * 64 + logical % 64
        chosen = set(actual[row].tolist())
        assert set(physical[keep].tolist()) <= chosen
        assert len([s for s in chosen if s >= 0]) == k


def test_live_rows_reuse_one_program_and_replay(programs, pool):
    program = programs["prefill"]
    table, lengths, q, w = _prefill_case(pool, 200, 40, 3)
    scratch = torch.zeros((program.scratch_bytes(200)["scratch"],), dtype=torch.uint8, device=q.device)
    eager = _aot(program, q, w, pool.cache, table, lengths, 40, 0, scratch=scratch)
    output = torch.full_like(eager, -7)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(q, w, pool.cache, table, lengths, output, scratch, scalars=(200, 40, 0))
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        output.fill_(-7)
        graph.replay()
    torch.cuda.synchronize()
    # Winner slots are reserved atomically: the selected set is exact, its order is not.
    assert torch.equal(_sorted(output), _sorted(eager))
    # Same compiled program, fewer rows and a narrower table.
    small = _aot(program, q[:9].contiguous(), w[:9].contiguous(), pool.cache, table[:11].contiguous(),
                 lengths[:9].clamp(max=11 * 64).contiguous(), 11, 0, scratch=scratch)
    expected = _prepared("prefill", 512, q[:9].contiguous(), w[:9].contiguous(), pool.cache,
                         table[None, :11].expand(9, 11), lengths[:9].clamp(max=11 * 64).contiguous(), 11)
    _assert_same_selection(small, expected, lengths[:9].clamp(max=11 * 64), 512)


def test_fused_decode_replays_in_cuda_graph(programs, pool):
    program = programs["decode_fused"]
    table, padded, lengths, q, w = _decode_case(pool, 8, 300, 44)
    scratch = torch.zeros((program.scratch_bytes(8)["scratch"],), dtype=torch.uint8, device=q.device)
    eager = _aot(program, q, w, pool.cache, padded, lengths, 300, 300, scratch=scratch)
    output = torch.full_like(eager, -7)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(q, w, pool.cache, padded, lengths, output, scratch, scalars=(8, 300, 300))
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        output.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_sorted(output), _sorted(eager))


@pytest.mark.parametrize("key", ["prefill", "decode_fused", "decode_tiled"])
def test_export_to_c_signature(programs, key, tmp_path):
    info = check_export(programs[key], tmp_path, f"dsv4_index_topk_{key}")
    assert info["argument_count"] == len(programs[key].operands) + 4
