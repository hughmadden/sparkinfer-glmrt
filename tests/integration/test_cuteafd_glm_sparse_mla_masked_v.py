"""Masked slots of the GLM sparse MLA programs over a poisoned record slot 0
(``zero_masked_v``).

A masked slot (``-1``, or past its row's length) stages record slot 0, page 0's
first record, and is weighted by zero. 0 x NaN is NaN, so the default programs
need finite bytes there; ``zero_masked_v`` zeroes the staged V and its inline
FP32 scales first, as the UE8M0 kernels always do. No row here selects slot 0,
so what it holds is only ever the stand-in, and:

- with NaN bytes there, the default programs' outputs are not finite and the
  opt-in programs' are, bit for bit the default programs' over a zeroed slot 0;
- with finite bytes there, the opt-in programs equal the default programs on
  every row with a valid slot (decode bit for bit: the split merge sums from
  +0, so even a zero's sign agrees; prefill up to a zero's sign), and give
  zeros on a row whose every slot is masked, where the default programs return
  an average of the stand-in.

Rows take every way a slot is masked: the tail of a short row's last chunk,
holes, wholly masked leading chunks (a masked prefix inside a split, and whole
splits of the one-row decode bucket), one valid slot, every slot masked, no
slot at all, a masked last chunk, and none.
"""

from __future__ import annotations

import pytest
import torch

from b12x.integration.cuteafd._common import GLM53, GLM53_FLASH, GLMFGeometry

from ..conftest import require_b12x

CONTEXT = 4096
PATTERNS = ("short", "holes", "masked_lead", "one_valid", "all_masked", "empty", "masked_tail", "full")
_PROGRAMS: dict = {}

# (route, max_rows, full_launch_splits, rows): the decode buckets of one row (a
# 64-slot chunk per split), up to eight rows and up to max_rows, GLM 5.3 Flash's
# 128-row program as exported (one split), and the prefill program.
CASES = [
    pytest.param("decode", 64, None, 1, id="decode-1"),
    pytest.param("decode", 64, None, 8, id="decode-8"),
    pytest.param("decode", 64, None, 40, id="decode-40"),
    pytest.param("decode", 128, 1, 100, id="decode-m128-100"),
    pytest.param("prefill", 64, None, 40, id="prefill-40"),
]
GEOMETRIES = [pytest.param(GLM53_FLASH, id="glm53-flash"), pytest.param(GLM53, id="glm53")]


def _program(g, route: str, max_rows: int, splits, zero_masked_v: bool):
    from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot

    key = (g.name, route, max_rows, splits, zero_masked_v)
    if key not in _PROGRAMS:
        flash = isinstance(g, GLMFGeometry)
        _PROGRAMS[key] = compile_glm_sparse_mla_aot(
            g, route=route, max_rows=max_rows, name="glmf_sparse_mla" if flash else "glm_sparse_mla",
            fp32_partials=flash and route == "decode", full_launch_splits=splits, zero_masked_v=zero_masked_v)
    return _PROGRAMS[key]


def _slots(g) -> int:
    return int(getattr(g, "sparse_topk", g.index_topk))


def _cache(g, seed: int) -> torch.Tensor:
    """CONTEXT FP8 records (RoPE lanes for GLM 5.x) on 64-record pages."""
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference

    gen = torch.Generator(device="cpu").manual_seed(seed)
    latent = (torch.randn((CONTEXT, 512), generator=gen) * 0.5).cuda()
    rope = None
    if g.record_bytes == 656:
        rope = (torch.randn((CONTEXT, 64), generator=gen) * 0.5).bfloat16().cuda()
    records = pack_mla_kv_cache_reference(latent, rope).view(CONTEXT, g.record_bytes)
    return records.reshape(CONTEXT // 64, 64 * g.record_bytes).contiguous()


def _with_stand_in(cache: torch.Tensor, g, kind: str) -> torch.Tensor:
    """``cache`` with record slot 0 zeroed, all 0xFF (NaN as E4M3 and as FP32), or
    finite: E4M3 bytes other than NaN, FP32 scales of either sign from 1e-3 to
    1e3, finite BF16 RoPE."""
    out = cache.clone()
    width = g.record_bytes
    if kind == "zero":
        out[0, :width] = 0
    elif kind == "nan":
        out[0, :width] = 0xFF
    else:
        gen = torch.Generator(device="cpu").manual_seed(11)
        data = torch.randint(0, 256, (512,), generator=gen, dtype=torch.uint8)
        data[(data & 0x7F) == 0x7F] = 0x3C
        magnitude = 10.0 ** (torch.rand(4, generator=gen) * 6 - 3)
        sign = torch.tensor([1.0, -1.0, 1.0, -1.0])
        parts = [data, (magnitude * sign).float().view(torch.uint8)]
        if width == 656:
            parts.append(torch.randn(64, generator=gen).bfloat16().view(torch.uint8))
        out[0, :width] = torch.cat(parts).cuda()
    return out


def _selection(g, rows: int, seed: int):
    """Index rows that never select slot 0 (row i takes PATTERNS[i % 8]) and lengths."""
    slots = _slots(g)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.full((rows, slots), -1, dtype=torch.int32)
    lengths = torch.zeros(rows, dtype=torch.int32)
    for i in range(rows):
        pick = (torch.randperm(CONTEXT - 1, generator=gen)[:slots] + 1).int()
        pattern = PATTERNS[i % len(PATTERNS)]
        row = pick.clone()
        n = slots
        if pattern == "short":
            n = 1000
        elif pattern == "holes":
            row[::7] = -1
        elif pattern == "masked_lead":
            row[:192] = -1
        elif pattern == "one_valid":
            n = 200
            row[:n] = -1
            row[137] = pick[0]
        elif pattern == "all_masked":
            n = 300
            row[:] = -1
        elif pattern == "empty":
            n = 0
        elif pattern == "masked_tail":
            row[slots - 64:] = -1
        indices[i, :n] = row[:n]
        lengths[i] = n
    return indices.cuda(), lengths.cuda()


def _rows(rows: int, *patterns: str) -> list[int]:
    return [i for i in range(rows) if PATTERNS[i % len(PATTERNS)] in patterns]


def _valid_rows(rows: int) -> list[int]:
    return [i for i in range(rows) if PATTERNS[i % len(PATTERNS)] not in ("all_masked", "empty")]


def _query(g, rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn((rows, g.heads, g.latent_dim), generator=gen) * 0.5).bfloat16().cuda()


def _run(program, q, cache, indices, lengths) -> torch.Tensor:
    rows = q.shape[0]
    out = torch.zeros((rows, q.shape[1], 512), dtype=torch.bfloat16, device="cuda")
    scratch = torch.zeros(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(q, cache, indices, lengths, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor, *, zero_sign: bool) -> bool:
    """Bit for bit, or with ``zero_sign`` up to the sign of a zero."""
    differ = _bits(a) != _bits(b)
    if zero_sign:
        differ &= ~((a == 0) & (b == 0))
    return not bool(differ.any())


def _skip_unless_exported(g, splits):
    if splits is not None and not isinstance(g, GLMFGeometry):
        pytest.skip("only GLM 5.3 Flash exports the one-split 128-row decode program")


@pytest.mark.parametrize("g", GEOMETRIES)
@pytest.mark.parametrize("route,max_rows,splits,rows", CASES)
def test_a_nan_stand_in_reaches_only_the_default_programs(g, route, max_rows, splits, rows):
    require_b12x()
    _skip_unless_exported(g, splits)
    cache, q = _cache(g, 1), _query(g, rows, 2)
    indices, lengths = _selection(g, rows, 3)
    nan, zero = _with_stand_in(cache, g, "nan"), _with_stand_in(cache, g, "zero")
    default = _program(g, route, max_rows, splits, False)
    masked = _program(g, route, max_rows, splits, True)
    # The hazard: masked slots weight the stand-in by zero, and 0 x NaN is NaN.
    assert not bool(torch.isfinite(_run(default, q, nan, indices, lengths)).all())
    got = _run(masked, q, nan, indices, lengths)
    assert bool(torch.isfinite(got).all())
    # Zeroed in shared memory, the stand-in is the zeroed slot 0 the default programs need.
    assert _same(got, _run(masked, q, zero, indices, lengths), zero_sign=False)
    assert _same(got, _run(default, q, zero, indices, lengths), zero_sign=False)


@pytest.mark.parametrize("g", GEOMETRIES)
@pytest.mark.parametrize("route,max_rows,splits,rows", CASES)
def test_finite_stand_in_bytes_reach_no_row_with_a_valid_slot(g, route, max_rows, splits, rows):
    require_b12x()
    _skip_unless_exported(g, splits)
    cache, q = _cache(g, 4), _query(g, rows, 5)
    indices, lengths = _selection(g, rows, 6)
    finite, zero = _with_stand_in(cache, g, "finite"), _with_stand_in(cache, g, "zero")
    default = _program(g, route, max_rows, splits, False)
    masked = _program(g, route, max_rows, splits, True)
    a = _run(default, q, finite, indices, lengths)
    b = _run(masked, q, finite, indices, lengths)
    clean = _run(default, q, zero, indices, lengths)
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    valid = _valid_rows(rows)
    sign = route == "prefill"
    assert _same(a[valid], b[valid], zero_sign=sign)
    assert _same(a[valid], clean[valid], zero_sign=sign)
    if all_masked := _rows(rows, "all_masked"):
        # Every slot masked: zeros, where the default programs average the stand-in.
        assert not bool(_bits(b[all_masked]).any())
        assert bool((a[all_masked] != 0).any())
    if empty := _rows(rows, "empty"):
        assert _same(a[empty], b[empty], zero_sign=False)


@pytest.mark.parametrize("g", GEOMETRIES)
def test_zero_masked_v_is_opt_in_and_keyed(g):
    require_b12x()
    from b12x.integration.cuteafd.glm_sparse_mla import _Decode, _Prefill

    plain, opted = _Decode(g, 64), _Decode(g, 64, zero_masked_v=True)
    assert opted.key == plain.key + ("zero_masked_v",)
    assert not any(k.zero_masked_v for k in plain.kernels) and all(k.zero_masked_v for k in opted.kernels)
    plain, opted = _Prefill(g), _Prefill(g, zero_masked_v=True)
    assert opted.key == plain.key + ("zero_masked_v",)
    assert not any(k.zero_masked_v for k in plain.kernels) and all(k.zero_masked_v for k in opted.kernels)
