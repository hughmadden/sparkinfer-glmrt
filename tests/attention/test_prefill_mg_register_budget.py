"""CPU-only guard against an exact-fill setmaxnreg split in the MG prefill kernel."""

from __future__ import annotations

import pytest

from b12x.attention._shared.mla.prefill_mg import _check_prefill_register_budget


def test_exact_fill_split_is_rejected() -> None:
    # 240 * 256 math threads + 32 * 128 IO threads == 65,536: an exact fill.
    with pytest.raises(AssertionError, match="exactly"):
        _check_prefill_register_budget(240, 32, 256)
    # 240/24 leaves 1,024 registers of slack and is the shipped H32 split.
    _check_prefill_register_budget(240, 24, 256)
    # The default split keeps slack too.
    _check_prefill_register_budget(232, 32, 256)
