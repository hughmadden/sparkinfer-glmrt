"""CPU contracts for shared coordinator route planning and live-row dispatch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd/fp8_moe.py"


def policy(sm):
    tree = ast.parse(SOURCE.read_text())
    names = {"_w8a8", "_streams", "auto_large_rows", "_large_route", "_resolve"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and all(isinstance(t, ast.Name) for t in n.targets)
             and isinstance(n.value, (ast.Constant, ast.Tuple))]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"_sm121": lambda: sm == 121}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    return tree, env


@pytest.mark.parametrize("sm", [120, 121])
@pytest.mark.parametrize("weights", ["fp8", "mxfp4", "nvfp4a4"])
def test_coordinator_auto_covers_decode_and_prefill(sm, weights):
    tree, env = policy(sm)
    threshold = env["auto_large_rows"](False, weights)
    resolved = env["_resolve"]("auto", 4096, False, weights)
    if weights == "nvfp4a4":
        assert resolved == "stream"  # W4A4 for every row, never an A16 GEMV.
        assert threshold == 0
    else:
        assert resolved == "auto"
        assert env["_resolve"]("auto", threshold, False, weights) == "decode"
        assert env["_resolve"]("auto", threshold + 1, False, weights) == "auto"
    expected_large = "prefill" if weights == "fp8" else "stream"
    assert env["_large_route"](False, weights) == expected_large

    # Exercise the actual DSL entry's scalar branch without CUDA imports.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8Moe")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    run.decorator_list = []
    run.returns = None
    for arg in run.args.args:
        arg.annotation = None
    calls = []
    env.update(Int32=int, cutlass=SimpleNamespace(const_expr=bool))
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(SOURCE), "exec"), env)
    plan = SimpleNamespace(route=resolved, threshold=threshold,
                           decode=lambda *args: calls.append("decode"),
                           prefill=lambda *args: calls.append(expected_large))
    for rows in [1, 16, 80, threshold or 128, max(threshold + 1, 256), 4096]:
        env["__call__"](plan, *([None] * 11), rows, None)
        assert calls[-1] == ("decode" if resolved == "auto" and rows <= threshold else expected_large)


def test_mxfp4_stream_preserves_native_bf16():
    _, env = policy(120)
    assert env["_streams"](False, "mxfp4")
    assert env["_resolve"]("stream", 4096, False, "mxfp4") == "stream"
    with pytest.raises(ValueError, match="use stream"):
        env["_resolve"]("prefill", 4096, False, "mxfp4")
