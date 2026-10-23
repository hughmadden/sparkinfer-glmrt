"""The dense GEMM's TMA L2 evict-first policy streams the weights, never the activations.

``dense_gemm(lhs, rhs)`` always binds rhs (weights, ``[N, K, L]``) to ``mB_nkl`` and
``tma_atom_b``; ``swap_ab`` only swaps which shared-memory tile feeds MMA-A. So every
TMA copy carrying a ``cache_policy`` must use ``tma_atom_b``.
"""
from __future__ import annotations

import ast
from pathlib import Path

DENSE = Path(__file__).resolve().parents[2] / "b12x" / "_lib" / "dense_gemm.py"


def _policy_copies():
    tree = ast.parse(DENSE.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy" and any(k.arg == "cache_policy" for k in node.keywords)):
            yield node


def test_every_dense_tma_cache_policy_is_on_the_weight_atom():
    copies = list(_policy_copies())
    assert copies, "dense_gemm no longer issues an L2 cache policy; update this test"
    atoms = {ast.unparse(call.args[0]) for call in copies}
    assert atoms == {"tma_atom_b"}, atoms


def test_swap_ab_does_not_rebind_the_weight_tensor():
    source = DENSE.read_text()
    # The host binds b (rhs) to the B TMA atom unconditionally; swap_ab swaps fragments only.
    assert "tCsA = thr_mma.partition_A(sB)" in source
    assert "b_launch_torch = b_torch" in source
    assert "a_torch, b_torch = b_torch" not in source and "b_torch, a_torch = a_torch" not in source
