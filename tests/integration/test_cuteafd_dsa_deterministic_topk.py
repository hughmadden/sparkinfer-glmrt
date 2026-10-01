"""DSA index top-k determinism (tiled radix select): at equal scores the lower logical index
wins, and the selection is emitted in ascending logical order with -1 padding last, whatever
the atomic arrival order. Index keys come from a few shared vectors, so whole groups of
tokens score exactly alike and the top-k boundary falls inside a tied group (with few groups
the tied candidates overflow the select's buffer and take the exact fallback)."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_dsv4_indexer_aot import HIGH_PAGE, PAGE_BYTES, _aot, _query

KEY_DIM = 128


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd._common import GLM53
    from b12x.integration.cuteafd.dsv4_indexer import compile_dsv4_index_topk_aot as dsv4
    from b12x.integration.cuteafd.glm_indexer import compile_glm_index_topk_aot as glm

    with exportable_compilation():
        return {
            "flash_prefill": (dsv4(FLASH, max_rows=64, max_pages=1100, mode="prefill"), 64),
            "flash_decode": (dsv4(FLASH, max_rows=64, max_pages=1100, mode="decode"), 64),
            "pro_prefill": (dsv4(PRO, max_rows=64, max_pages=1100, mode="prefill"), 64),
            "glm_prefill": (glm(GLM53, max_rows=64, max_pages=1100, mode="prefill"), 32),
            "glm_decode": (glm(GLM53, max_rows=64, max_pages=1100, mode="decode"), 32),
        }


def _grouped_cache(pages: torch.Tensor, groups: int, gen: torch.Generator, pool_pages: int, device):
    """Index keys drawn from `groups` vectors: every token of a group scores the same."""
    from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference

    basis = torch.randn((groups, KEY_DIM), generator=gen) * (torch.rand((groups, 1), generator=gen) * 2 + 0.25)
    group = torch.randint(0, groups, (pages.numel() * 64,), generator=gen)
    cache = torch.zeros((pool_pages, PAGE_BYTES), dtype=torch.uint8, device=device)
    cache[pages.long().to(device)] = pack_index_k_cache_reference(basis[group].to(device))
    return cache, group


def _check_row(actual, scores, group, pages, n, topk):
    """`actual` (physical slots) against the top-k by (group score desc, logical index asc) in
    logical order with -1 last. Returns False when another group scores within rounding of the
    boundary group (the kernel's order between those groups may differ from the reference's),
    after checking order and padding only."""
    col = {int(p): c for c, p in enumerate(pages.tolist())}
    logical = [col[s // 64] * 64 + s % 64 if s >= 0 else -1 for s in actual.tolist()]
    valid = [x for x in logical if x >= 0]
    assert logical[: len(valid)] == valid and all(x == -1 for x in logical[len(valid):]), "padding not last"
    assert valid == sorted(valid) and len(set(valid)) == len(valid), "not in ascending logical order"
    assert len(valid) == min(n, topk)
    if n <= topk:
        assert valid == list(range(n))
        return True
    key = scores[group[:n]]
    order = sorted(range(n), key=lambda i: (-float(key[i]), i))
    boundary = float(key[order[topk - 1]])
    present = set(group[:n].tolist())
    close = [g for g in present if abs(float(scores[g]) - boundary) <= 1e-4 * max(1.0, abs(boundary))]
    if len(close) > 1:
        return False
    assert valid == sorted(order[:topk]), "selection differs from the (score, index) top-k"
    return True


@pytest.mark.parametrize(("key", "groups", "rows", "width", "mode"), [
    ("flash_prefill", 7, 9, 1100, "prefill"), ("flash_prefill", 200, 9, 1100, "prefill"),
    ("flash_prefill", 40, 5, 9, "prefill"),
    ("flash_decode", 7, 6, 1100, "decode"), ("flash_decode", 300, 6, 600, "decode"),
    ("pro_prefill", 11, 5, 1100, "prefill"),
    ("glm_prefill", 9, 5, 1100, "prefill"), ("glm_prefill", 500, 5, 700, "prefill"),
    ("glm_decode", 9, 5, 1100, "decode"),
])
def test_ties_resolve_by_index_and_emit_in_order(programs, key, groups, rows, width, mode):
    from b12x.attention.dsa_indexer.reference import paged_decode_logits_reference

    program, heads = programs[key]
    topk = program.geometry["topk"]
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(groups * 1000 + rows + width)
    pages = (torch.randperm(2 * width, generator=gen)[:width] + HIGH_PAGE).to(torch.int32)
    cache, group = _grouped_cache(pages, groups, gen, HIGH_PAGE + 2 * width + 1, device)
    lengths = torch.randint(width * 64 // 2, width * 64 + 1, (rows,), generator=gen, dtype=torch.int32)
    lengths[0] = min(topk // 2, width * 64)  # fewer visible rows than topk: -1 padding
    q, _ = _query(rows, gen, device)
    q = q[:, :heads].contiguous()
    w = (torch.randn((rows, heads), generator=gen) * 0.02).bfloat16().float().to(device)
    table = pages[None, :].expand(rows, width).contiguous().to(device)
    logits = paged_decode_logits_reference(
        q_fp8=q, weights=w, index_k_cache=cache, real_page_table=table,
        query_row_to_batch=torch.arange(rows, device=device, dtype=torch.int32),
        seqlens_per_query=torch.full((rows,), width * 64, dtype=torch.int32, device=device)).cpu()
    first_of = {}
    for i, g in enumerate(group.tolist()):
        first_of.setdefault(g, i)
    scores = [torch.tensor([float(logits[row, first_of[g]]) if g in first_of else -1e30 for g in range(groups)])
              for row in range(rows)]
    pointer, stride = (pages.to(device), 0) if mode == "prefill" else (table, width)
    lengths = lengths.to(device)
    first = _aot(program, q, w, cache, pointer, lengths, width, stride).cpu()
    second = _aot(program, q, w, cache, pointer, lengths, width, stride).cpu()
    assert torch.equal(first, second), "the top-k differs between two identical launches"
    checked = sum(_check_row(first[row], scores[row], group, pages, int(lengths[row]), topk) for row in range(rows))
    assert checked >= rows // 2, f"only {checked} of {rows} rows had an unambiguous boundary group"
