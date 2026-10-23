"""CPU-only MiniMax NVFP4 geometry and the real A4 activation arithmetic."""
from __future__ import annotations

import ast
import __future__
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "b12x/integration/cuteafd"
OAI = "swigluoai_uninterleave"


def definitions(path, names, env):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if
                 isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == "property"]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), env)
    return env


def bf16(value):
    bits = np.float32(value).view(np.uint32).item()
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return np.uint32(rounded).view(np.float32)


def kernels():
    env = dict(Float32=np.float32, const_expr=bool,
               cutlass=SimpleNamespace(range_constexpr=range, select_=lambda c, a, b: a if c else b),
               cute=SimpleNamespace(math=SimpleNamespace(exp=lambda x, **kw: np.exp(x))),
               div_rn_f32=lambda a, b: np.float32(a / b), _round_bf16=bf16,
               _bf16_lo=lambda word: bf16(word[0]), _bf16_hi=lambda word: bf16(word[1]))
    definitions(INTEGRATION / "_fp8_moe_kernels.py", {"_post_silu_product"}, env)
    definitions(INTEGRATION / "_nvfp4_moe_kernels.py", {"nvfp4_alpha_offset"}, env)
    return definitions(INTEGRATION / "_nvfp4_moe_a4.py", {
        "_swiglu16", "nvfp4_input_scale_offset", "ChunkSwiGLUA4", "StreamNvfp4LinearA4", "StreamNvfp4GateUpA4"}, env)


def activation(gate, up, *, oai=True, post=False):
    out = np.full(16, np.nan, dtype=np.float32)
    kernels()["_swiglu16"](list(zip(gate[::2], gate[1::2])), list(zip(up[::2], up[1::2])),
                           out, 8, 7.0, post, oai)
    assert np.isnan(out[:8]).all()
    return out[8:]


def test_swigluoai_matches_shared_reference_and_rounds_product_once():
    gate = np.array([bf16(v) for v in [-12, -7, -1.234, 0, 0.234, 1.234, 7, 12]])
    up = np.array([bf16(v) for v in [-12, 12, 0.234, 3, -1, 1.234, 0, 0]])
    env = dict(SWIGLUOAI_UNINTERLEAVE=OAI, torch=SimpleNamespace(
        clamp=lambda x, min=None, max=None: np.clip(x, min, max),
        sigmoid=lambda x: np.float32(1) / (np.float32(1) + np.exp(-x))))
    reference = definitions(ROOT / "b12x/moe/_shared/kernels/reference.py", {"_apply_gated_activation"}, env)
    expected = reference["_apply_gated_activation"](gate, up, activation=OAI,
        swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    actual = activation(gate, up)
    np.testing.assert_array_equal(actual, [bf16(v) for v in expected])
    assert actual[3] == actual[4] == 0
    assert actual[6] == actual[7] == 7
    assert actual[0] != bf16(-7 / (1 + np.exp(np.float32(1.702 * 7))) * -6)
    clipped = np.minimum(gate, 7)
    twice = np.array([bf16(v) for v in clipped / (1 + np.exp(-np.float32(1.702) * clipped))])
    twice = np.array([bf16(v) for v in twice * (np.clip(up, -7, 7) + 1)])
    assert np.any(actual != twice)


@pytest.mark.parametrize("post", [False, True])
def test_existing_silu_arithmetic_is_unchanged(post):
    gate = np.array([bf16(v) for v in [-12, -7, -1.234, 0, 0.234, 1.234, 7, 12]])
    up = np.array([bf16(v) for v in [-12, 12, 0.234, 3, -1, 1.234, 0, 0.234]])
    g = gate if post else np.minimum(gate, 7)
    silu = g / (1 + np.exp(-g))
    silu = np.minimum(silu, 7) if post else np.array([bf16(v) for v in silu])
    np.testing.assert_array_equal(activation(gate, up, oai=False, post=post),
                                  [bf16(v) for v in silu * np.clip(up, -7, 7)])


@pytest.mark.parametrize("tp,width", [(1, 3072), (2, 1536), (4, 768), (6, 512)])
def test_minimax_geometry_native_w4a4_and_tp(tp, width):
    env = definitions(INTEGRATION / "fp8_moe.py", {"Fp8MoeGeometry", "GEOMETRIES"}, {"dataclass": dataclass})
    g = env["GEOMETRIES"]["minimax_nvfp4"].with_tp(tp)
    assert (g.hidden, g.experts, g.top_k, g.intermediate, g.slice) == (6144, 128, 4, 3072, width)
    assert (g.weights, g.activations, g.kind, g.activation, g.swiglu_limit) == ("nvfp4", "a4", "nvfp4a4", OAI, 7.0)
    with pytest.raises(ValueError, match="require native NVFP4 A4"):
        env["Fp8MoeGeometry"]("unsupported", 6144, 128, 4, 3072, activation=OAI)


@pytest.mark.parametrize("name", ["ChunkSwiGLUA4", "StreamNvfp4GateUpA4"])
def test_activation_is_in_each_a4_cache_key(name):
    cls = kernels()[name]
    args = dict(inter=768, hidden=6144, experts=128, limit=7.0)
    variants = [cls(**args, activation=v) for v in ("silu_bf16_product", "silu_post_clamp_fp32_product", OAI)]
    assert len({v.key() for v in variants}) == 3
    assert variants[-1].swigluoai and not variants[-1].post_silu
    assert variants[-1].key()[-1] == OAI
