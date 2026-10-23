"""MiniMax decode attention program on the GPU against dense FP32 attention.

Each row is an independent one-token request over a paged FP8 cache, as the
engine issues both decode and its row-wise prefill. Lengths cover every
last-page fill class, including a full earlier page followed by a last page
whose valid rows leave whole 16-key warp slices empty.
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
        actual = output[row].float()
        relative = ((actual - expected).norm() / expected.norm()).item()
        if not torch.isfinite(actual).all() or relative > .03:
            failures.append((length, round(relative, 4)))
    assert not failures, f"rows (length, relative L2) differ from dense attention: {failures}"
