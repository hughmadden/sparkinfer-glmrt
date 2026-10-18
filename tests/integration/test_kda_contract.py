"""CPU-only KDA layout contracts against independent NumPy projections."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Geometry contracts deliberately do not import CUDA, Torch or CuTe.
_spec = importlib.util.spec_from_file_location("_test_kda_geometry", Path(__file__).parents[2] /
                                             "b12x/integration/cuteafd/_kda_contract.py")
_contract = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _contract
_spec.loader.exec_module(_contract)
KDAGeometry, kda_layout = _contract.KDAGeometry, _contract.kda_layout


def test_kimi_geometry_head_split_and_native_roles():
    g = KDAGeometry(hidden=7168, kda_heads=96)
    half = g.head_split(2)
    assert half.kda_heads == 48 and half.hidden == 7168
    assert half.kda_in_width == 4 * 6144 + 128 + 48
    layout = kda_layout(half)
    assert layout["fg_batch"] == 1 and layout["fg_stride"] == 6144
    assert layout["gate_offset"] == 3 * 6144 + 128
    assert layout["beta_offset"] == 4 * 6144 + 128
    assert layout["gate_stride"] == half.kda_in_width
    assert layout["o_norm_dtype"] == "float32"
    assert not layout["round_recurrent_beta"]
    assert g.kda_state_bytes == 96 * 128 * 128 * 4
    with pytest.raises(ValueError):
        g.head_split(5)
    with pytest.raises(ValueError):
        g.head_split(16)  # six heads cannot form an eight-warp norm block


def test_glmf_layout_retains_original_offsets_and_beta_rounding():
    g = SimpleNamespace(kda_width=8192, kda_head_dim=128, kda_in_width=3 * 8192 + 256 + 64)
    assert kda_layout(g) == {"full_rank_gate": False, "fg_batch": 2, "fg_stride": 16384,
        "gate_offset": 8192, "gate_stride": 16384, "beta_offset": 3 * 8192 + 256,
        "o_norm_dtype": "bfloat16", "round_recurrent_beta": True}


@pytest.mark.parametrize("full", [False, True])
def test_fused_projection_offsets_match_unfused_numpy_oracle(full):
    g = KDAGeometry(hidden=128, kda_heads=16, output_gate="full_rank" if full else "low_rank")
    rng = np.random.default_rng(273)
    d, hd = g.kda_width, g.kda_head_dim
    x = rng.standard_normal((3, g.hidden), dtype=np.float32) * 0.01
    roles = [rng.standard_normal((n, g.hidden), dtype=np.float32) * 0.01
             for n in (d, d, d, hd, d if full else hd, g.kda_heads)]
    fb = rng.standard_normal((d, hd), dtype=np.float32) * 0.01
    gb = rng.standard_normal((d, hd), dtype=np.float32) * 0.01
    direct = [x @ w.T for w in roles]
    packed = x @ np.concatenate(roles).T
    layout = kda_layout(g)
    np.testing.assert_allclose(packed[:, layout["beta_offset"]:], direct[-1], rtol=1e-5, atol=1e-8)
    fg = packed[:, 3 * d:3 * d + hd] @ fb.T
    np.testing.assert_allclose(fg, direct[3] @ fb.T, rtol=1e-5, atol=1e-8)
    gate = packed[:, layout["gate_offset"]:layout["gate_offset"] + d] if full else direct[4] @ gb.T
    np.testing.assert_allclose(gate, direct[4] if full else direct[4] @ gb.T, rtol=1e-5, atol=1e-8)


def test_numpy_recurrence_commutes_with_head_split():
    """Safe gate, F32 beta and state [value,key] have no cross-head coupling."""
    rng = np.random.default_rng(6)
    q, k, v, raw = [rng.standard_normal((8, 16, 128), dtype=np.float32) for _ in range(4)]
    beta = rng.standard_normal((8, 16), dtype=np.float32)
    a = rng.standard_normal(16, dtype=np.float32)
    bias = rng.standard_normal((16, 128), dtype=np.float32)

    def run(heads):
        state = np.zeros((len(a[heads]), 128, 128), dtype=np.float32)
        out = []
        for t in range(8):
            qt, kt = q[t, heads], k[t, heads]
            qt = qt / np.sqrt(np.sum(qt * qt, axis=-1, keepdims=True) + 1e-6) * (128 ** -0.5)
            kt = kt / np.sqrt(np.sum(kt * kt, axis=-1, keepdims=True) + 1e-6)
            gate = -5 / (1 + np.exp(-np.exp(a[heads, None]) * (raw[t, heads] + bias[heads])))
            state *= np.exp(gate)[:, None, :]
            delta = (v[t, heads] - np.einsum("hvk,hk->hv", state, kt)) / (1 + np.exp(-beta[t, heads, None]))
            state += delta[:, :, None] * kt[:, None, :]
            out.append(np.einsum("hvk,hk->hv", state, qt))
        return np.stack(out), state

    all_out, all_state = run(slice(None))
    halves = [run(slice(rank * 8, (rank + 1) * 8)) for rank in (0, 1)]
    np.testing.assert_array_equal(all_out, np.concatenate([h[0] for h in halves], axis=1))
    np.testing.assert_array_equal(all_state, np.concatenate([h[1] for h in halves]))
