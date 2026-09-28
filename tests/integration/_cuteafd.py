"""Shared helpers for the cuteafd DeepSeek V4 AOT program tests."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

SNAPSHOT = Path(os.environ.get(
    "CUTEAFD_DSV4_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
))


@lru_cache(maxsize=1)
def _weight_map() -> dict[str, str] | None:
    index = SNAPSHOT / "model.safetensors.index.json"
    if not index.is_file():
        return None
    return json.loads(index.read_text())["weight_map"]


def has_checkpoint() -> bool:
    return _weight_map() is not None


def checkpoint_tensor(name: str, device="cuda") -> torch.Tensor:
    """Read one real V4 Flash tensor (skips the test when unavailable)."""
    weights = _weight_map()
    if weights is None:
        pytest.skip(f"DeepSeek V4 Flash snapshot not mounted at {SNAPSHOT}")
    from safetensors import safe_open

    with safe_open(str(SNAPSHOT / weights[name]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device)


def u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def prepare(plan, name: str):
    """Prepare one declaration with its default configuration (no timing)."""
    from b12x.preparation import PreparedCall, prepare_default

    if plan.prepared is None:
        prepare_default(plan.request(name=name, prepare_call=lambda state: PreparedCall(run=lambda: None)))
    return plan


def check_export(program, tmp_path: Path, stem: str) -> dict:
    """export_to_c the program and validate the generated wrapper ABI."""
    from b12x.integration.cuteafd import validate_exported_header

    symbol = "cuteafd_" + stem
    program.export_to_c(str(tmp_path), stem, symbol)
    header = tmp_path / f"{stem}.h"
    assert header.is_file() and (tmp_path / f"{stem}.o").is_file()
    return validate_exported_header(program, header, symbol)


def assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    a, e = actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    mismatch = int((a != e).sum())
    assert mismatch == 0, f"{name}: {mismatch} of {a.numel()} bytes differ"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))
