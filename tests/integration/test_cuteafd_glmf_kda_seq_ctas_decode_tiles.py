"""GLM 5.3 Flash decode KDA programs built with ``seq_ctas`` and ``decode_tiles`` together
(``compile_glmf_kda_aot(fp8=True, seq_ctas=8, decode_tiles=True)``, the options of a serving A/B build): the bytes
of the default programs.

The 64- and 128-row decode programs, FP32 and BF16 state, over plain (``spec`` 0) and speculative (``spec`` 1)
steps shaped as cuteafd's: 16 sequences of 8, 4 and 1 rows, 15 of 8 rows and one of 7 (a 127-row verify), one
sequence of 8 rows (the skinny GEMV), 9 single-row sequences padded with masked rows to a 16-row bucket, and
unequal sequences padded to 32 rows. Against the default build: the same ABI, and byte-identical outputs,
recurrent state, conv state and replay record.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_bf16_state import _geometry
from .test_cuteafd_glmf_kda_seq_ctas import POOL, _pool, _run_kda, _same

OPTIONS = {"seq_ctas": 8, "decode_tiles": True}
_PROGRAMS: dict = {}

#: (program rows, step): a step is stretches of rows in step order, n > 0 a sequence of n rows, n < 0 -n masked rows.
STEPS = {
    "16x8": (128, [8] * 16),
    "15x8+7": (128, [8] * 15 + [7]),
    "16x4": (64, [4] * 16),
    "16x1": (64, [1] * 16),
    "1x8": (64, [8]),
    "9x1_padded": (64, [1] * 9 + [-7]),
    "unequal_padded": (64, [3, 1, 8, 2, 5, -13]),
}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _kda(max_rows: int, state_dtype: str, optioned: bool):
    """The decode KDA program (``fp8=True``, run with ``fp8_rows`` 0: the BF16 projections)."""
    key = ("kda", max_rows, state_dtype, optioned)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        _PROGRAMS[key] = glmf.compile_glmf_kda_aot(_geometry(), max_rows=max_rows, fp8=True, state_dtype=state_dtype,
                                                   **(OPTIONS if optioned else {}))
    return _PROGRAMS[key]


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("spec", [0, 1])
@pytest.mark.parametrize("name", list(STEPS))
def test_kda_program_with_seq_ctas_and_decode_tiles_is_the_default_program(name, spec, state_dtype):
    from b12x.integration.cuteafd import glmf

    g = _geometry()
    max_rows, step = STEPS[name]
    base, both = _kda(max_rows, state_dtype, False), _kda(max_rows, state_dtype, True)
    assert both.name == base.name and both.abi == base.abi
    assert both.scratch_bytes(max_rows) == base.scratch_bytes(max_rows)
    tables = {"in": glmf.KDA_DECODE_TILES[(g.kda_in_width, g.hidden)],
              "out": glmf.KDA_DECODE_TILES[(g.hidden, g.kda_width)]}
    added = {"seq_ctas": OPTIONS["seq_ctas"], "decode_tiles": tables}
    assert not set(added) & set(base.geometry)
    assert dict(both.geometry) == {**base.geometry, **added}
    rows = sum(abs(n) for n in step)
    gen = torch.Generator(device="cuda").manual_seed(7000 + rows + spec)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    conv = (torch.randn((POOL, 3, 3 * g.kda_width), generator=gen, device="cuda") * 0.1).bfloat16()
    want = _run_kda(base, x, step, conv, _pool(state_dtype), spec, max(max_rows, 64))
    got = _run_kda(both, x, step, conv, _pool(state_dtype), spec, max(max_rows, 64))
    assert bool(want[0].float().abs().sum() > 0)
    for what, a, b in zip(("outputs", "state pool", "conv state pool", "replay record"), want, got):
        assert _same(a, b), f"{what}: seq_ctas and decode_tiles against the default, step {name}"
