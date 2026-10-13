"""GLM 5.3 Flash's 128-row sparse MLA decode program (``glmf_sparse_mla_decode_m128``).

128-row verify steps run wide execution with the same 64-row numerical domain
as serial decode. The startup hint fixes split grouping for every live width.
Contract tests take explicit SM counts without a device; program tests compare
nonuniform FP8 records against the reference, narrow chunks and one-row decode.
"""

from __future__ import annotations

import logging

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


def _scratch(buckets, rows: int, plan_rows=None) -> int:
    return sparse_mla_scratch_bytes(G, route="decode", rows=rows, buckets=buckets, fp32_partials=True,
                                    plan_rows=plan_rows)


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


@pytest.mark.parametrize("sms", [170, 188])
def test_explicit_card_plan_reaches_compiler_and_scratch(monkeypatch, sms):
    from types import SimpleNamespace
    from b12x.integration.cuteafd import glm_sparse_mla as module
    buckets = decode_buckets(G, 64, sm_count=sms)
    seen = []
    def decode(g, rows, fp32, full, sm_count, plan_capacity_rows):
        seen.append((g, rows, fp32, full, sm_count, plan_capacity_rows))
        return SimpleNamespace(buckets=buckets, key=("decode", buckets))
    monkeypatch.setattr(module, "_Decode", decode)
    monkeypatch.setattr(module, "compile_program", lambda *_, **kwargs: kwargs)
    program = compile_glm_sparse_mla_aot(G, route="decode", max_rows=128,
                                       fp32_partials=True, sm_count=sms, plan_capacity_rows=64)
    assert seen == [(G, 128, True, None, sms, 64)]
    assert program["geometry"]["decode_buckets"] == [list(b) for b in buckets]
    assert program["geometry"]["attention_plan_capacity_rows"] == 64
    assert program["geometry"]["attention_plan_scratch"] == [
        {"rows_cap": cap, "scratch": _scratch(buckets, 128, cap)} for cap, _, _ in buckets]
    assert [scalar.name for scalar in program["scalars"]] == ["rows", "plan_rows"]
    assert program["scratch"]["scratch"](128) == _scratch(buckets, 128)


@pytest.mark.parametrize("sms", [0, -1, True, 170.0])
def test_explicit_card_plan_rejects_invalid_counts(sms):
    with pytest.raises(ValueError, match="sm_count must be a positive integer"):
        compile_glm_sparse_mla_aot(G, route="decode", sm_count=sms)


def test_the_wide_scratch_holds_the_unsplit_partials():
    planner = decode_buckets(G, 128, sm_count=170)
    one = decode_buckets(G, 128, sm_count=170, full_launch_splits=1)
    assert _scratch(planner, 128) == 554_729_472
    # The safe fallback holds all recipes at full capacity, not just live buckets.
    assert _scratch(one, 128) == 554_729_472
    assert _scratch(one, 128, 128) == 16_809_984 == 128 * 64 * 512 * 4 + 128 * 64 * 4
    for hint in (1, 8, 128):
        splits, _ = _plan(one, hint)
        for rows in range(1, 129):
            partial_bytes = rows * G.heads * splits * 512 * 4
            used = (partial_bytes + 1023) // 1024 * 1024 + rows * G.heads * splits * 4
            assert used == _scratch(one, rows, hint) <= _scratch(one, 128, hint)


def test_the_split_count_override_wins_over_full_launch_splits_and_says_so(monkeypatch, caplog):
    # B12X_MLA_SM120_NUM_SPLITS pins every bucket's split count, ahead of full_launch_splits: at
    # 170 SMs 33 keeps the 128-row bucket's 33 splits and its 554,729,472-byte scratch.
    logger = "b12x.integration.cuteafd.glm_sparse_mla"
    monkeypatch.setenv("B12X_MLA_SM120_NUM_SPLITS", "33")
    with caplog.at_level(logging.WARNING, logger=logger):
        buckets = decode_buckets(G, 128, sm_count=170, full_launch_splits=1)
    assert buckets == ((1, 33, 1), (8, 33, 1), (128, 33, 1))
    assert _scratch(buckets, 128) == 554_729_472
    messages = [record.getMessage() for record in caplog.records if record.name == logger]
    assert len(messages) == 1, messages
    assert "B12X_MLA_SM120_NUM_SPLITS=33 overrides full_launch_splits=1" in messages[0]
    assert "the 128-row bucket takes 33 splits" in messages[0]
    caplog.clear()
    # No warning where nothing is overridden: the same count, a bucket within the waves (188 SMs),
    # or no full_launch_splits; nor without the variable.
    with caplog.at_level(logging.WARNING, logger=logger):
        decode_buckets(G, 128, sm_count=188, full_launch_splits=1)
        decode_buckets(G, 128, sm_count=170)
        monkeypatch.setenv("B12X_MLA_SM120_NUM_SPLITS", "1")
        assert decode_buckets(G, 128, sm_count=170, full_launch_splits=1)[-1] == (128, 1, 33)
        monkeypatch.delenv("B12X_MLA_SM120_NUM_SPLITS")
        assert decode_buckets(G, 128, sm_count=170, full_launch_splits=1) == ((1, 33, 1), (8, 5, 7), (128, 1, 33))
    assert not [record for record in caplog.records if record.name == logger]


def test_the_wide_benchmark_needs_every_selected_slot_in_its_context(monkeypatch, capsys):
    # Every row selects 2112 distinct slots: a --context that rounds down (to whole 64-slot pages)
    # below that stops before any device work.
    from benchmarks import bench_glmf_sparse_mla_wide as bench

    monkeypatch.delenv("B12X_MLA_SM120_NUM_SPLITS", raising=False)
    for context in ("0", "2048", "2111"):
        monkeypatch.setattr("sys.argv", ["bench", "--context", context])
        with pytest.raises(SystemExit) as stop:
            bench.main()
        assert stop.value.code == 2 and "needs at least 2112" in capsys.readouterr().err, context


@pytest.mark.parametrize("domain", [0, -1, 129, True, 64.0])
def test_numerical_domain_rejects_invalid_geometry(domain):
    with pytest.raises(ValueError, match="plan_capacity_rows"):
        compile_glm_sparse_mla_aot(G, route="decode", max_rows=128, plan_capacity_rows=domain)


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
                                                    fp32_partials=True, full_launch_splits=full_launch_splits,
                                                    plan_capacity_rows=64)
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


def _run(program, q, cache, indices, lengths, plan_rows=8):
    rows = q.shape[0]
    out = torch.empty((rows, G.heads, 512), dtype=torch.bfloat16, device="cuda")
    scratch_bytes = sparse_mla_scratch_bytes(
        G, route="decode", rows=program.geometry["max_rows"],
        buckets=decode_buckets(G, 64), fp32_partials=True, plan_rows=plan_rows)
    storage = torch.full((scratch_bytes + 1024,), 0xA5, dtype=torch.uint8, device="cuda")
    scratch = storage[:scratch_bytes]
    program.launch(q, cache, indices, lengths, out, scratch, scalars=(rows, plan_rows))
    torch.cuda.synchronize()
    assert torch.all(storage[scratch_bytes:] == 0xA5), "selected MLA scratch overrun"
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
    # Different execution capacities share one startup numerical recipe, byte-exactly.
    for start in (0, 64):
        part = slice(start, min(start + 64, rows))
        got = _run(narrow, q[part].contiguous(), cache, indices[part].contiguous(), lengths[part].contiguous())
        assert torch.equal(out[part], got), (rows, start)
    serial = _run(narrow, q[:1].contiguous(), cache, indices[:1].contiguous(), lengths[:1].contiguous())
    assert torch.equal(out[:1], serial)
