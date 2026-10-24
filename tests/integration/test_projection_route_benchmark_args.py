"""CPU-only CLI contracts; importing the benchmark must not import CUDA libraries."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_cuteafd_projection_routes.py"
_SPEC = importlib.util.spec_from_file_location("projection_route_benchmark", _PATH)
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def test_compensation_pair_accepts_production_shapes_and_wide_rows():
    args = _BENCH.parse_args(["--recipes", "compensation", "--shapes", "7168x6144,6144x6144",
                              "--rows", "1,16,4096"])
    assert args.recipes == "compensation"
    assert args.shapes == [(7168, 6144), (6144, 6144)]
    assert args.rows == [1, 16, 4096]


def test_existing_route_defaults_are_preserved():
    args = _BENCH.parse_args([])
    assert args.recipes == "routes"
    assert args.shapes == [(4096, 4096)]
    assert args.rows == [1, 8, 9, 10, 16, 32, 64]
    assert args.samples == 20


def test_compensation_arms_use_identical_tiles_and_runtime_rows(monkeypatch):
    calls = []

    def tma(n, k, **kwargs):
        calls.append((n, k, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(BFloat16="bf16"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        SkinnyBf16Gemv=None, TmaBf16Projection=tma, skinny_config=None))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", SimpleNamespace(
        Operand=lambda *a: a, Scalar=lambda *a: a,
        compile_program=lambda op, **kwargs: kwargs))
    direct = _BENCH._program(7168, 6144, "uncompensated")
    compensated = _BENCH._program(7168, 6144, "compensated")
    assert calls == [(7168, 6144, {"tiles_by_rows": (), "compensated": False}),
                     (7168, 6144, {"tiles_by_rows": (), "compensated": True})]
    assert direct["scalars"] == compensated["scalars"] == (("rows",),)
    assert direct["key"] != compensated["key"]


def test_projection_compensation_is_opt_in_and_default_key_is_unchanged(monkeypatch):
    source = _PATH.parents[1] / "b12x/integration/cuteafd/pointwise.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_projection_aot")
    calls = []

    class Projection:
        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS={}, RoutedBf16Projection=Projection, TmaBf16Projection=Projection))
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32"),
                     BFloat16="bf16", Float32="f32", Operand=lambda *a: a,
                     Scalar=lambda *a: a, compile_program=lambda op, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    build = namespace["compile_bf16_projection_aot"]
    default = build(width=6144, outputs=7168)
    explicit = build(width=6144, outputs=7168, compensated=False)
    compensated = build(width=6144, outputs=7168, compensated=True)
    assert default == explicit
    assert calls[:2] == [(7168, 6144, {"out_dtype": "bf16", "max_skinny_rows": 0})] * 2
    assert calls[2] == (7168, 6144, {"out_dtype": "bf16", "tiles_by_rows": (), "compensated": True})
    assert default["geometry"] == dict(width=6144, outputs=7168, output_dtype="bf16", skinny_rows=0)
    assert compensated["geometry"]["compensated"] is True
    assert default["key"] != compensated["key"]
    with pytest.raises(ValueError, match="boolean"):
        build(width=6144, outputs=7168, compensated="yes")


def _compensated_build(monkeypatch, table):
    """``compile_bf16_projection_aot`` with stub projections; returns (build, calls)."""
    source = _PATH.parents[1] / "b12x/integration/cuteafd/pointwise.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_projection_aot")
    calls = []

    class Projection:
        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS=table, RoutedBf16Projection=Projection, TmaBf16Projection=Projection))
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32"),
                     BFloat16="bf16", Float32="f32", Operand=lambda *a: a,
                     Scalar=lambda *a: a, compile_program=lambda op, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["compile_bf16_projection_aot"], calls


def test_compensated_route_selects_its_measured_row_tiles(monkeypatch):
    """A shape with a ``COMPENSATED_TILES_BY_ROWS`` entry carries it; others stay fixed."""
    table = ((16, 1, 4, 16, 32, 12), (4096, 2, 4, 64, 128, 3))
    build, calls = _compensated_build(monkeypatch, {(6144, 4096): table})
    program = build(width=4096, outputs=6144, compensated=True)
    assert calls == [(6144, 4096, {"out_dtype": "bf16", "tiles_by_rows": table, "compensated": True})]
    assert program["geometry"]["compensated"] is True
    assert program["geometry"]["tiles_by_rows"] == table
    plain = build(width=6144, outputs=7168, compensated=True)
    assert calls[-1] == (7168, 6144, {"out_dtype": "bf16", "tiles_by_rows": (), "compensated": True})
    assert "tiles_by_rows" not in plain["geometry"]


@pytest.mark.parametrize("argv", [
    ["--rows", "4096"],
    ["--recipes", "compensation", "--rows", "4097"],
    ["--recipes", "compensation", "--rows", "0"],
    ["--rows", ""],
    ["--samples", "0"],
    ["--shapes", "7168"],
    ["--shapes", "7168x6144x8"],
    ["--shapes", "7168x6145"],
    ["--shapes", "0x6144"],
    ["--shapes", "nonsense"],
])
def test_invalid_geometry_is_rejected_before_cuda_import(argv):
    with pytest.raises(SystemExit) as error:
        _BENCH.parse_args(argv)
    assert error.value.code == 2
