"""cuteafd DSV4 producer / index-producer AOT programs vs b12x.attention.dsv4_producer."""

from __future__ import annotations

import math

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, cosine, has_checkpoint, u8

LAYER = 2
PAGE_BYTES = 149_760
EPS = 1.0e-6
# A physical page past the 2^31-byte line (AGENTS.md big-pid rule).
HIGH_PAGE = (2**31) // PAGE_BYTES + 3


def _block_fp8(rows, cols, gen, device):
    weight = (torch.randn((rows, cols), generator=gen) / 16).to(torch.float8_e4m3fn).to(device)
    scale = torch.randint(118, 124, (rows // 128, cols // 128), generator=gen, dtype=torch.uint8).to(device)
    return weight, scale


def _producer_weights(geometry, device):
    from b12x.attention import dsv4_producer

    if geometry.name == "flash" and has_checkpoint():
        g = lambda n: checkpoint_tensor(f"layers.{LAYER}.attn.{n}", device)  # noqa: E731
        return dsv4_producer.pack_weights(
            g("wq_a.weight"), u8(g("wq_a.scale")), g("wq_b.weight"), u8(g("wq_b.scale")),
            g("wkv.weight"), u8(g("wkv.scale")), g("q_norm.weight").contiguous(),
            g("kv_norm.weight").contiguous())
    gen = torch.Generator(device="cpu").manual_seed(geometry.hidden)
    h, q, n = geometry.hidden, geometry.q_lora_rank, geometry.heads
    wq_a, wq_a_s = _block_fp8(q, h, gen, device)
    wq_b, wq_b_s = _block_fp8(n * 512, q, gen, device)
    wkv, wkv_s = _block_fp8(512, h, gen, device)
    q_norm = (1 + 0.1 * torch.randn((q,), generator=gen)).bfloat16().to(device)
    kv_norm = (1 + 0.1 * torch.randn((512,), generator=gen)).bfloat16().to(device)
    return dsv4_producer.pack_weights(wq_a, wq_a_s, wq_b, wq_b_s, wkv, wkv_s, q_norm, kv_norm)


def _cos_sin(positions: int, device) -> torch.Tensor:
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous().to(device)


def _inputs(geometry, rows, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn((rows, geometry.hidden), generator=gen).bfloat16().to(device)
    positions = (torch.randperm(4096, generator=gen)[:rows]).to(torch.int64).to(device)
    # Mix low pages with one row parked on a page past the 2^31-byte line.
    slots = torch.randperm(8 * 256, generator=gen)[:rows].to(torch.int64)
    slots[-1] = HIGH_PAGE * 256 + 77
    return hidden, positions, slots.to(device), _cos_sin(4096, device)


@pytest.fixture(scope="module")
def cache():
    require_b12x()
    return torch.zeros((HIGH_PAGE + 1, PAGE_BYTES), dtype=torch.uint8, device="cuda")


def _prepared_producer(geometry, max_rows, weights, hidden, positions, slots, cos_sin, cache):
    from b12x.attention import dsv4_producer

    rows = hidden.shape[0]
    plan = dsv4_producer.plan(dsv4_producer.Caps(
        device=hidden.device, max_tokens=max_rows, hidden=geometry.hidden,
        q_lora_rank=geometry.q_lora_rank, heads=geometry.heads, cache_format="fp8"))
    spec = plan.scratch_specs()[0]
    query = torch.empty((rows, geometry.heads, 512), dtype=torch.bfloat16, device=hidden.device)
    binding = dsv4_producer.bind(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=hidden.device),
        hidden_states=hidden, positions=positions, main_slots=slots, cos_sin_cache=cos_sin,
        main_kv_cache=cache, query=query, weights=weights, eps=EPS, expected_m=rows)
    dsv4_producer.run(binding=binding)
    return query, binding.q_rank.clone()


def _aot_producer(program, weights, hidden, positions, slots, cos_sin, cache, geometry):
    rows = hidden.shape[0]
    query = torch.full((rows, geometry.heads, 512), float("nan"), dtype=torch.bfloat16, device=hidden.device)
    q_rank = torch.full((rows, geometry.q_lora_rank), float("nan"), dtype=torch.bfloat16, device=hidden.device)
    scratch = torch.full((program.scratch_bytes(program.geometry["max_rows"])["scratch"],), 0xFF,
                         dtype=torch.uint8, device=hidden.device)
    program.launch(
        hidden, positions, slots, cos_sin,
        weights.qkv_rank.weight.values, weights.qkv_rank.weight.scale_mma,
        weights.q.weight.values, weights.q.weight.scale_mma,
        weights.q_norm, weights.kv_norm, cache, query, q_rank, scratch, scalars=(rows,))
    return query, q_rank


def _records(cache, slots):
    """(payload 576 B, scale 8 B) per slot."""
    out = []
    for slot in slots.tolist():
        page, row = divmod(slot, 256)
        out.append(torch.cat((cache[page, row * 576:(row + 1) * 576],
                              cache[page, 256 * 576 + row * 8:256 * 576 + row * 8 + 8])))
    return torch.stack(out)


def _assert_close_bf16(a, e, name, *, max_flip_fraction=1e-3):
    """Equal except rare one-ulp BF16 flips from FP32 reduction order."""
    flips = (a != e)
    frac = float(flips.float().mean())
    assert frac <= max_flip_fraction, f"{name}: {frac:.2e} of values differ"
    if bool(flips.any()):
        torch.testing.assert_close(a.float(), e.float(), rtol=2.0 ** -7, atol=2.0 ** -8)


def _assert_records(actual, expected):
    # Scale bytes and RoPE BF16 lanes: allow rare ulp flips; FP8 payload may
    # differ by one E4M3 code where the BF16 input flipped.
    assert_bitwise(actual[:, 576 + 7], expected[:, 576 + 7], "pad byte")
    scale_diff = (actual[:, 576:583].int() - expected[:, 576:583].int()).abs()
    assert int(scale_diff.max()) <= 1 and float((scale_diff != 0).float().mean()) < 0.01
    fp8_a = actual[:, :448].view(torch.float8_e4m3fn).float()
    fp8_e = expected[:, :448].view(torch.float8_e4m3fn).float()
    assert float((fp8_a != fp8_e).float().mean()) < 2e-3
    rope_a = actual[:, 448:576].contiguous().view(torch.bfloat16)
    rope_e = expected[:, 448:576].contiguous().view(torch.bfloat16)
    _assert_close_bf16(rope_a, rope_e, "rope", max_flip_fraction=5e-3)


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd.dsv4_producer import (
        compile_dsv4_index_producer_aot,
        compile_dsv4_producer_aot,
    )

    with exportable_compilation():
        return {
            ("flash", 16): compile_dsv4_producer_aot(FLASH, max_rows=16),
            ("flash", 512): compile_dsv4_producer_aot(FLASH, max_rows=512),
            ("pro", 4): compile_dsv4_producer_aot(PRO, max_rows=4),
            ("index", 16): compile_dsv4_index_producer_aot(FLASH, max_rows=16),
            ("index", 512): compile_dsv4_index_producer_aot(FLASH, max_rows=512),
        }


@pytest.mark.parametrize(("geometry_name", "max_rows", "rows"), [
    ("flash", 16, 1), ("flash", 16, 5), ("flash", 16, 16), ("flash", 512, 300), ("pro", 4, 3),
])
def test_producer_matches_prepared(programs, cache, geometry_name, max_rows, rows):
    from b12x.integration.cuteafd import FLASH, PRO

    device = require_b12x()
    geometry = FLASH if geometry_name == "flash" else PRO
    weights = _producer_weights(geometry, device)
    hidden, positions, slots, cos_sin = _inputs(geometry, rows, 1000 + rows, device)
    cache.zero_()
    ref_query, ref_q_rank = _prepared_producer(geometry, max_rows, weights, hidden, positions,
                                               slots, cos_sin, cache)
    ref_records = _records(cache, slots)
    cache.zero_()
    query, q_rank = _aot_producer(programs[(geometry_name, max_rows)], weights, hidden, positions,
                                  slots, cos_sin, cache, geometry)
    torch.cuda.synchronize()
    _assert_close_bf16(q_rank, ref_q_rank, "q_rank")
    _assert_close_bf16(query, ref_query, "query", max_flip_fraction=2e-3)
    assert cosine(query, ref_query) > 0.99999
    _assert_records(_records(cache, slots), ref_records)
    # Nothing outside the addressed rows is written.
    touched = sum(int((cache[page] != 0).any()) for page in {s // 256 for s in slots.tolist()})
    assert int((cache != 0).any(dim=1).sum()) == touched


def _index_weights(geometry, device):
    from b12x.attention import dsv4_producer

    if geometry.name == "flash" and has_checkpoint():
        g = lambda n: checkpoint_tensor(f"layers.{LAYER}.attn.indexer.{n}", device)  # noqa: E731
        return dsv4_producer.pack_indexer_weights(g("wq_b.weight"), u8(g("wq_b.scale")),
                                                  g("weights_proj.weight").contiguous())
    gen = torch.Generator(device="cpu").manual_seed(7)
    wq_b, wq_b_s = _block_fp8(8192, geometry.q_lora_rank, gen, device)
    proj = (torch.randn((64, geometry.hidden), generator=gen) / 64).bfloat16().to(device)
    return dsv4_producer.pack_indexer_weights(wq_b, wq_b_s, proj)


@pytest.mark.parametrize(("max_rows", "rows"), [(16, 1), (16, 11), (512, 257)])
def test_index_producer_matches_prepared(programs, max_rows, rows):
    from b12x.attention import dsv4_producer
    from b12x.integration.cuteafd import FLASH

    device = require_b12x()
    geometry = FLASH
    weights = _index_weights(geometry, device)
    gen = torch.Generator(device="cpu").manual_seed(rows)
    q_rank = (torch.randn((rows, geometry.q_lora_rank), generator=gen)).bfloat16().to(device)
    hidden = torch.randn((rows, geometry.hidden), generator=gen).bfloat16().to(device)
    positions = torch.randperm(4096, generator=gen)[:rows].to(torch.int64).to(device)
    cos_sin = _cos_sin(4096, device)
    plan = dsv4_producer.plan_indexer(dsv4_producer.IndexerCaps(
        device=device, max_tokens=max_rows, hidden=geometry.hidden, q_lora_rank=geometry.q_lora_rank))
    spec = plan.scratch_specs()[0]
    ref_query = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    ref_weights = torch.empty((rows, 64), dtype=torch.float32, device=device)
    binding = dsv4_producer.bind_indexer(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), q_rank=q_rank,
        hidden_states=hidden, positions=positions, cos_sin_cache=cos_sin, query=ref_query,
        head_weights=ref_weights, weights=weights, expected_m=rows)
    dsv4_producer.run_indexer(binding=binding)
    program = programs[("index", max_rows)]
    query = torch.empty_like(ref_query)
    head_weights = torch.full_like(ref_weights, float("nan"))
    scratch = torch.empty((program.scratch_bytes(max_rows)["scratch"],), dtype=torch.uint8, device=device)
    program.launch(q_rank, hidden, positions, cos_sin, weights.q.weight.values,
                   weights.q.weight.scale_mma, weights.weights_projection, query, head_weights,
                   scratch, scalars=(rows,))
    torch.cuda.synchronize()
    # The head-weight projection accumulates in a different order than
    # cuBLAS; outputs are BF16-rounded twice, so allow one-ulp differences.
    torch.testing.assert_close(head_weights, ref_weights, rtol=2.0 ** -6, atol=1e-6)
    assert float((head_weights != ref_weights).float().mean()) < 0.05
    q, e = query.float(), ref_query.float()
    assert float((q != e).float().mean()) < 2e-3
    assert cosine(q, e) > 0.9995


@pytest.mark.parametrize("key", [("flash", 16), ("pro", 4), ("index", 16)])
def test_export_to_c_signature(programs, key, tmp_path):
    stem = "dsv4_" + "_".join(map(str, key))
    info = check_export(programs[key], tmp_path, stem)
    assert info["argument_count"] == len(programs[key].operands) + 2
