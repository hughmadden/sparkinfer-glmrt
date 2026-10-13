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


def grouping_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8Moe")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "weight_read_grouping")
    env = {"_Route": type("Route", (), {}), "_StreamRoute": type("Stream", (), {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
    return env


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
@pytest.mark.parametrize("capacity,tiles", [(1, 1), (16, 1), (17, 2), (40, 3), (80, 4), (4096, 4)])
def test_weight_read_groups_use_actual_gemv_launch_tiles(weights, capacity, tiles):
    env = grouping_method()
    route = env["_Route"]()
    route.route = "decode"
    route.gate_up = SimpleNamespace(m_tiles=tiles)
    route.down = SimpleNamespace(m_tiles=tiles)
    launch = SimpleNamespace(route="decode", decode=route)
    assert env["weight_read_grouping"](launch, weights, capacity) == {
        "contract": "exact_grouping", "segments": [{"end_rows": capacity, "group_rows": 16 * tiles}]}


@pytest.mark.parametrize("stream,large", [(False, 64), (True, 128)])
def test_weight_read_groups_follow_actual_auto_cutoff_and_large_route(stream, large):
    env = grouping_method()
    decode = env["_Route"]()
    decode.route = "decode"
    decode.gate_up = decode.down = SimpleNamespace(m_tiles=4)
    prefill = env["_StreamRoute" if stream else "_Route"]()
    if stream:
        prefill.prep = SimpleNamespace(tile_rows=large)
    else:
        prefill.route = "prefill"
        prefill.gate_up = prefill.down = SimpleNamespace(tile_m=large)
    launch = SimpleNamespace(route="auto", threshold=80, decode=decode, prefill=prefill)
    expected = ([{"end_rows": 80, "group_rows": 64}, {"end_rows": 4096, "group_rows": large}]
                if stream else [{"end_rows": 4096, "group_rows": 64}])
    assert env["weight_read_grouping"](launch, "mxfp4" if stream else "fp8", 4096) == {
        "contract": "exact_grouping", "segments": expected}


def test_nvfp4_reports_a_bound_not_a_fabricated_installed_tile():
    env = grouping_method()
    # It must not inspect the occupancy-dependent sparse/wide launch objects.
    assert env["weight_read_grouping"](None, "nvfp4", 4096) == {
        "contract": "conservative_route_upper_bound",
        "segments": [{"end_rows": 4096, "group_rows": 1}]}


def test_weight_read_groups_reject_different_gate_and_down_tiles():
    env = grouping_method()
    route = env["_Route"]()
    route.route = "decode"
    route.gate_up = SimpleNamespace(m_tiles=1)
    route.down = SimpleNamespace(m_tiles=2)
    with pytest.raises(ValueError, match="groups differ"):
        env["weight_read_grouping"](SimpleNamespace(route="decode", decode=route), "fp8", 16)


def test_mxfp4_stream_preserves_native_bf16():
    _, env = policy(120)
    assert env["_streams"](False, "mxfp4")
    assert env["_resolve"]("stream", 4096, False, "mxfp4") == "stream"
    with pytest.raises(ValueError, match="use stream"):
        env["_resolve"]("prefill", 4096, False, "mxfp4")
