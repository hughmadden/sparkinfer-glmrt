"""GLM 5.3 Flash's 128-row sparse MLA decode program (``glmf_sparse_mla_decode_m128``).

128-row verify steps run the 128-row decode programs for 65..128 live rows. On a
170-SM GPU the split planner finds no split count for the program's 128-row bucket
within its three waves (128 rows x 4 head blocks is 512 CTAs) and splits it
maximally: 33 splits of one 64-slot chunk, FP32 partials of every chunk in a
554,729,472-byte scratch. ``full_launch_splits`` plans such a bucket with the given
splits instead. The contract tests take explicit SM counts and need no device; the
program tests compare the wide program with the FP8-record reference and, row for
row, with the 64-row program wherever both run the same plan.
"""

from __future__ import annotations

import pytest
import torch

from b12x.integration.cuteafd._common import GLM53_FLASH
from b12x.integration.cuteafd.glm_sparse_mla import (
    compile_glm_sparse_mla_aot,
    decode_buckets,
    sparse_mla_scratch_bytes,
)

from ..conftest import require_b12x

G = GLM53_FLASH
SLOTS = 2112


def _scratch(buckets, rows: int) -> int:
    return sparse_mla_scratch_bytes(G, route="decode", rows=rows, buckets=buckets, fp32_partials=True)


def _plan(buckets, rows: int) -> tuple[int, int]:
    """(splits, chunks per split) of the bucket ``rows`` live rows run."""
    return next((splits, per_split) for cap, splits, per_split in buckets if rows <= cap)


def test_a_full_launch_bucket_takes_the_given_splits():
    # 170 SMs: 1 and 8 rows keep the planner's plan; 128 rows x 4 blocks exceed its 3 x 170.
    assert decode_buckets(G, 128, sm_count=170) == ((1, 33, 1), (8, 5, 7), (128, 33, 1))
    assert decode_buckets(G, 128, sm_count=170, full_launch_splits=1) == ((1, 33, 1), (8, 5, 7), (128, 1, 33))
    assert decode_buckets(G, 128, sm_count=170, full_launch_splits=2) == ((1, 33, 1), (8, 5, 7), (128, 2, 17))


@pytest.mark.parametrize("sms,max_rows", [(170, 64), (170, 127), (188, 128), (188, 64)])
def test_buckets_within_the_waves_keep_the_planner(sms, max_rows):
    # Up to 127 rows on 170 SMs (508 CTAs) and 128 on 188: the option changes nothing.
    planner = decode_buckets(G, max_rows, sm_count=sms)
    assert decode_buckets(G, max_rows, sm_count=sms, full_launch_splits=1) == planner
    assert _plan(planner, max_rows)[0] <= 2


def test_the_wide_scratch_holds_the_unsplit_partials():
    planner = decode_buckets(G, 128, sm_count=170)
    one = decode_buckets(G, 128, sm_count=170, full_launch_splits=1)
    assert _scratch(planner, 128) == 554_729_472
    # 128 rows x 64 heads x 512 FP32 partials, 1024-aligned, then the LSE: the 1- and
    # 8-row buckets' 33 and 8 x 5 partial rows fit within.
    assert _scratch(one, 128) == 16_809_984 == 128 * 64 * 512 * 4 + 128 * 64 * 4
    for rows in range(1, 129):
        splits, _ = _plan(one, rows)
        partial_bytes = rows * G.heads * splits * 512 * 4
        used = (partial_bytes + 1023) // 1024 * 1024 + rows * G.heads * splits * 4
        assert used <= _scratch(one, rows) <= _scratch(one, 128)


def test_full_launch_splits_plan_decode_buckets_only():
    with pytest.raises(ValueError, match="full_launch_splits"):
        compile_glm_sparse_mla_aot(G, route="prefill", max_rows=4096, full_launch_splits=1)
    with pytest.raises(ValueError, match="full_launch_splits"):
        compile_glm_sparse_mla_aot(G, route="decode", max_rows=128, fp32_partials=True, full_launch_splits=0)


_PROGRAMS: dict = {}


def _program(max_rows: int, full_launch_splits: int | None = None):
    key = (max_rows, full_launch_splits)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_glm_sparse_mla_aot(G, route="decode", max_rows=max_rows, name="glmf_sparse_mla",
                                                    fp32_partials=True, full_launch_splits=full_launch_splits)
    return _PROGRAMS[key]


def _inputs(rows: int, seed: int, context: int = 8192):
    """Queries, a paged cache of FP8 records, 2112-slot selections (every third row a shorter one)."""
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference

    gen = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn((context, 512), generator=gen) * 0.5
    records = pack_mla_kv_cache_reference(latent.cuda()).view(context, 528)
    cache = records.reshape(context // 64, 64 * 528).contiguous()
    q = (torch.randn((rows, G.heads, 512), generator=gen) * 0.5).bfloat16().cuda()
    indices = torch.full((rows, SLOTS), -1, dtype=torch.int32)
    lengths = torch.empty(rows, dtype=torch.int32)
    for i in range(rows):
        n = SLOTS if i % 3 else int(torch.randint(1, SLOTS, (1,), generator=gen))
        indices[i, :n] = torch.randperm(context, generator=gen)[:n].int()
        lengths[i] = n
    return q, cache, records, indices.cuda(), lengths.cuda()


def _run(program, q, cache, indices, lengths):
    rows = q.shape[0]
    out = torch.empty((rows, G.heads, 512), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(q, cache, indices, lengths, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("rows", [65, 96, 127, 128])
def test_the_wide_program_is_the_reference_and_the_64_row_program(rows):
    require_b12x()
    from b12x.attention._shared.mla.reference import sparse_mla_reference

    wide, narrow = _program(128, 1), _program(64)
    q, cache, records, indices, lengths = _inputs(rows, rows)
    out = _run(wide, q, cache, indices, lengths)
    assert torch.isfinite(out).all()
    expected = sparse_mla_reference(q_all=q, kv_cache=records.view(-1, 1, 528), page_table_1=indices,
                                    active_token_counts=lengths, sm_scale=G.softmax_scale, v_head_dim=512)
    assert _cosine(out, expected) >= 0.9995
    # The 64-row program over the first 64 rows and the rest: rows independent, so every row
    # whose two plans agree is bit-identical (and the rest equal to rounding).
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    wide_plan = _plan(decode_buckets(G, 128, sm_count=sms, full_launch_splits=1), rows)
    narrow_buckets = decode_buckets(G, 64, sm_count=sms)
    for start in (0, 64):
        part = slice(start, min(start + 64, rows))
        got = _run(narrow, q[part].contiguous(), cache, indices[part].contiguous(), lengths[part].contiguous())
        if _plan(narrow_buckets, part.stop - part.start) == wide_plan:
            assert torch.equal(out[part], got), (rows, start)
        else:
            assert _cosine(out[part], got) >= 0.9999, (rows, start)
