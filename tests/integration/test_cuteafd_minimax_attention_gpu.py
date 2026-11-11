"""MiniMax decode attention program on the GPU against dense FP32 attention.

Decode rows are independent one-token requests over a paged FP8 cache.
Extend packs multi-token requests and unions their per-head selections.
Lengths include a full earlier page followed by a partial page whose valid
rows leave whole 16-key warp slices empty.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

ROWS = 64
PAGE = 64
GROUPS = {
    "one_page": [1, 17, 33, 64],
    "second_page": [65, 72, 80, 96, 97, 113, 128],
    "later_pages": [129, 140, 160, 192, 204, 255, 256, 257],
    "long": [700, 2048],
    # A long last row reads its page table after every other row's LSE store.
    "mixed": [128, 129, 140, 160, 192, 204, 1900],
}


@pytest.fixture(scope="module")
def program():
    require_b12x()
    from b12x.integration.cuteafd.minimax_attention import compile_minimax_attention_aot

    return compile_minimax_attention_aot(max_rows=ROWS, max_batch=ROWS, mode="decode")


def _cache(generator, pages):
    values = torch.randn((pages, PAGE, 4, 128), generator=generator, device="cuda") * 2
    return values.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("group", list(GROUPS))
def test_decode_rows_match_dense_attention_across_page_fills(program, group):
    lengths = GROUPS[group]
    generator = torch.Generator(device="cuda").manual_seed(20261024)
    batch = len(lengths)
    widths = [(length + PAGE - 1) // PAGE for length in lengths]
    pool = sum(widths) + 3
    order = torch.randperm(pool, generator=torch.Generator().manual_seed(7)).tolist()
    table = torch.full((batch, max(widths)), -1, dtype=torch.int32)
    cursor = 0
    for row, width in enumerate(widths):
        table[row, :width] = torch.tensor(order[cursor:cursor + width], dtype=torch.int32)
        cursor += width
    key, value = _cache(generator, pool), _cache(generator, pool)
    k_descale = (torch.rand((batch, 4), generator=generator, device="cuda") * .5 + .25).contiguous()
    v_descale = (torch.rand((batch, 4), generator=generator, device="cuda") * .5 + .25).contiguous()
    query = (torch.randn((batch, 64, 128), generator=generator, device="cuda") * .5).bfloat16()
    q2k = torch.full((4, ROWS, 16), -1, dtype=torch.int32)
    for row, length in enumerate(lengths):
        blocks = (length + 127) // 128
        assert blocks <= 16
        q2k[:, row, :blocks] = torch.arange(blocks, dtype=torch.int32)
    output = torch.zeros((batch, 64, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros((batch, 64), dtype=torch.float32, device="cuda")
    scratch = torch.zeros(program.scratch_bytes(ROWS)["scratch"], dtype=torch.uint8, device="cuda")
    cache_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    program.launch(query, key.view(torch.uint8), value.view(torch.uint8), k_descale, v_descale,
                   table.cuda().contiguous(), cache_lengths, cu, q2k.cuda().contiguous(), output, lse, scratch,
                   scalars=(batch, batch, table.shape[1], pool))
    torch.cuda.synchronize()

    failures = []
    for row, length in enumerate(lengths):
        pages = table[row, :widths[row]].tolist()
        k = key[pages].float().reshape(-1, 4, 128)[:length] * k_descale[row][None, :, None]
        v = value[pages].float().reshape(-1, 4, 128)[:length] * v_descale[row][None, :, None]
        q = query[row].float()
        kv_head = torch.arange(64, device="cuda") // 16
        scores = torch.einsum("hd,thd->ht", q, k[:, kv_head]) * 128 ** -0.5
        expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v[:, kv_head])
        expected_lse = torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0))
        if (lse[row] - expected_lse).abs().max().item() > .02:
            failures.append((length, "lse"))
        actual = output[row].float()
        relative = ((actual - expected).norm() / expected.norm()).item()
        if not torch.isfinite(actual).all() or relative > .03:
            failures.append((length, round(relative, 4)))
    assert not failures, f"rows (length, relative L2) differ from dense attention: {failures}"


def _assert_selected_attention(query, key, value, kd, vd, table, lengths, cu,
                               selections, output, lse):
    kv_head = torch.arange(64, device="cuda") // 16
    for request, length in enumerate(lengths):
        pages = table[request, :(length + PAGE - 1) // PAGE].tolist()
        k = key[pages].float().reshape(-1, 4, 128)[:length] * kd[request][None, :, None]
        v = value[pages].float().reshape(-1, 4, 128)[:length] * vd[request][None, :, None]
        positions = torch.arange(length, device="cuda")
        start, end = cu[request:request + 2]
        for row in range(start, end):
            visible = length - (end - start) + row - start + 1
            selected = selections[:, row]
            block_mask = (positions[None, :, None] // 128 == selected[:, None, :]).any(-1)
            mask = block_mask[kv_head] & (positions[None] < visible)
            scores = torch.einsum("hd,thd->ht", query[row].float(), k[:, kv_head]) * 128 ** -.5
            scores.masked_fill_(~mask, -torch.inf)
            expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v[:, kv_head])
            actual = output[row].float()
            relative = ((actual - expected).norm() / expected.norm()).item()
            assert torch.isfinite(actual).all() and relative <= .03, (request, row, relative)
            expected_lse = torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0))
            assert (lse[row] - expected_lse).abs().max().item() <= .02, (request, row)


@pytest.mark.parametrize("boundary", [2**31, 2**32])
def test_recycled_physical_pages_cross_byte_offset_boundary(boundary):
    require_b12x()
    from b12x.integration.cuteafd.minimax_attention import compile_minimax_attention_aot

    page_bytes = PAGE * 4 * 128
    high = boundary // page_bytes + 13
    pool = high + 2
    program = compile_minimax_attention_aot(max_rows=1, max_batch=1, mode="decode")
    generator = torch.Generator(device="cuda").manual_seed(650 + high)
    # Only initialize live tail pages; the untouched prefix models recycled IDs.
    key = torch.empty((pool, PAGE, 4, 128), dtype=torch.float8_e4m3fn, device="cuda")
    value = torch.empty_like(key)
    key[high:].copy_(_cache(generator, 2))
    value[high:].copy_(_cache(generator, 2))
    query = (torch.randn((1, 64, 128), generator=generator, device="cuda") * .5).bfloat16()
    descale = torch.ones((1, 4), device="cuda")
    table = torch.tensor([[high, high + 1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([65], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    selected = torch.full((4, 1, 16), -1, dtype=torch.int32, device="cuda")
    selected[:, :, 0] = 0
    output = torch.empty_like(query)
    lse = torch.empty((1, 64), device="cuda")
    scratch = torch.empty(program.scratch_bytes(1)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(query, key.view(torch.uint8), value.view(torch.uint8), descale, descale,
                   table, lengths, cu, selected, output, lse, scratch,
                   scalars=(1, 1, 2, pool))
    torch.cuda.synchronize()
    _assert_selected_attention(query, key, value, descale, descale, table, [65], [0, 1],
                               selected, output, lse)


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_sparse_selections_reuse_program_and_graph_with_live_shapes(mode):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.minimax_attention import compile_minimax_attention_aot

    program = compile_minimax_attention_aot(max_rows=ROWS, max_batch=16, mode=mode)
    generator = torch.Generator(device="cuda").manual_seed(20261111)
    pool, stride = 96, 44
    key, value = _cache(generator, pool), _cache(generator, pool)
    query = torch.empty((ROWS, 64, 128), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    lse = torch.empty((ROWS, 64), device="cuda")
    kd = (torch.rand((16, 4), generator=generator, device="cuda") * .5 + .25).contiguous()
    vd = (torch.rand((16, 4), generator=generator, device="cuda") * .5 + .25).contiguous()
    table = torch.full((16, stride), -1, dtype=torch.int32, device="cuda")
    cache_lengths = torch.zeros(16, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(17, dtype=torch.int32, device="cuda")
    selected = torch.full((4, ROWS, 16), -1, dtype=torch.int32, device="cuda")
    scratch = torch.empty(program.scratch_bytes(ROWS)["scratch"], dtype=torch.uint8, device="cuda")
    cases = ([([1], [65]), ([1] * 3, [129, 257, 2700]),
              ([1] * 16, [65 + 137 * i for i in range(16)])] if mode == "decode" else
             [([1], [65]), ([1, 9], [129, 257]), ([3, 9, 13], [65, 257, 2700])])
    for query_lengths, lengths in cases:
        rows, batch = sum(query_lengths), len(lengths)
        cu = [0]
        for count in query_lengths:
            cu.append(cu[-1] + count)
        query.copy_((torch.randn(query.shape, generator=generator, device="cuda") * .5).bfloat16())
        output.fill_(-7)
        lse.fill_(-7)
        selected.fill_(-1)
        cache_lengths[:batch] = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        offsets[:batch + 1] = torch.tensor(cu, dtype=torch.int32, device="cuda")
        for request, length in enumerate(lengths):
            width = (length + PAGE - 1) // PAGE
            table[request, :width] = torch.randperm(pool, generator=generator, device="cuda")[:width].int()
            start, end = cu[request:request + 2]
            for row in range(start, end):
                current = (length - (end - start) + row - start) // 128
                for head in range(4):
                    # Different sparse histories per row/head exercise union masks,
                    # not merely the all-blocks/dense special case.
                    history = sorted(range(current), key=lambda b: ((b + head + row) % 3, b))[:15]
                    blocks = sorted(history + [current])
                    selected[head, row, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device="cuda")

        def launch():
            program.launch(query, key.view(torch.uint8), value.view(torch.uint8), kd, vd,
                           table, cache_lengths, offsets, selected, output, lse, scratch,
                           scalars=(rows, batch, stride, pool))

        with kernel_resolution_guard("MiniMax attention live shapes and graph replay"):
            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            for _ in range(2):
                query.neg_()
                graph.replay()
                torch.cuda.synchronize()
                _assert_selected_attention(query, key, value, kd, vd, table, lengths, cu,
                                           selected, output, lse)
                assert (output[rows:] == -7).all(), "inactive output rows were overwritten"
                assert (lse[rows:] == -7).all(), "inactive LSE rows were overwritten"
