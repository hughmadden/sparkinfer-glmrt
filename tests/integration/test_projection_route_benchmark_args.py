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
        exact_row_cap = 2**31 - 1

        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS={}, RoutedBf16Projection=Projection,
        TmaBf16Projection=Projection, wide_prefill_min_rows=lambda *args: 4097))
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32"),
                     BFloat16="bf16", Float32="f32", Operand=lambda *a: a,
                     Scalar=lambda *a: a, compile_program=lambda op, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    build = namespace["compile_bf16_projection_aot"]
    default = build(width=6144, outputs=7168)
    explicit = build(width=6144, outputs=7168, compensated=False)
    compensated = build(width=6144, outputs=7168, compensated=True)
    assert default == explicit
    assert calls[:2] == [(7168, 6144, {"out_dtype": "bf16", "max_skinny_rows": 0,
                                     "wide": False, "wide_min_rows": None})] * 2
    assert calls[2] == (7168, 6144, {"out_dtype": "bf16", "tiles_by_rows": (), "compensated": True})
    assert default["geometry"] == dict(width=6144, outputs=7168, output_dtype="bf16", skinny_rows=0,
                                       exact_row_cap=2**31 - 1)
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
        exact_row_cap = 2**31 - 1

        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS=table, RoutedBf16Projection=Projection,
        TmaBf16Projection=Projection, wide_prefill_min_rows=lambda *args: 4097))
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


def test_trellis_accepts_config_and_continuous_pairing_without_cuda():
    args = _BENCH.parse_args(["--recipes", "trellis", "--shapes", "6144x16384",
                              "--rows", "2752,4096", "--trellis-config", "shapes.json",
                              "--trellis-paired", "--trellis-steady-ms", "200"])
    assert args.shapes == [(6144, 16384)]
    assert args.rows == [2752, 4096]
    assert args.trellis_config == Path("shapes.json")
    assert args.trellis_paired
    assert args.trellis_steady_ms == 200


@pytest.mark.parametrize("samples,flagged", [
    ([100, 100, 100, 100], False),
    ([100, 101, 100, 101], False),
    ([100, 120, 100, 120, 100], True),
    ([100], True),
])
def test_trellis_order_audit_flags_bias_or_missing_reverse_order(samples, flagged):
    audit = _BENCH._alternating_order_audit(samples)
    assert audit["flagged"] is flagged
    assert audit["mad_us"] >= 0
    assert (audit["delta_us"] is None) == (len(samples) == 1)


@pytest.mark.parametrize("argv", [
    ["--recipes", "trellis", "--rows", "4097"],
    ["--recipes", "trellis", "--trellis-steady-ms", "199"],
    ["--recipes", "trellis", "--trellis-steady-ms", "nan"],
    ["--recipes", "trellis", "--trellis-steady-ms", "inf"],
    ["--recipes", "trellis", "--trellis-arms", "ring3"],
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


def _native_threshold_bench():
    path = _PATH.with_name("kimi_k6_dense.py")
    spec = importlib.util.spec_from_file_location("native_threshold_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text", ["narrow=9", "decode=17", "wide2=33", "wide3=49",
                                  "folded=17", "prefill=131", "prefill=0",
                                  "narrow=8,decode=4", "splits=4", "stages=3"])
def test_native_thresholds_refuse_unsafe_knobs(text):
    with pytest.raises(ValueError):
        _native_threshold_bench().threshold_config(text)


def test_reference_encoder_build_requires_explicit_official_source(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--encoder-build-only"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


@pytest.mark.parametrize("text", ["decode=8", "wide2=24"])
def test_grouped_threshold_builder_refuses_partial_plane_overlap(text, tmp_path):
    bench = _native_threshold_bench()
    with pytest.raises(ValueError, match="next projection's partial plane"):
        bench.threshold_config(text, groups=4)
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    with pytest.raises(ValueError, match="next projection's partial plane"):
        bench.threshold_variant(source, tmp_path / "bad.py", bench.threshold_config(text), groups=4)


@pytest.mark.parametrize("decode,wide2", [(8, 32), (16, 24)])
def test_native_grouped_constructor_refuses_partial_plane_overlap(decode, wide2):
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    tree = ast.parse(source.read_text())
    check = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "validate_grouped_route_thresholds")
    core = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "NativeTrellisLinear")
    init = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    namespace = {"_GROUPED_DECODE_ROWS": decode, "_GROUPED_WIDE2_ROWS": wide2}
    exec(compile(ast.Module(body=[check, init], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(AssertionError, match="next projection's partial plane"):
        namespace["__init__"](SimpleNamespace(), {"groups": 4})


def test_standalone_threshold_variant_keeps_grouped_routes_frozen(tmp_path):
    bench = _native_threshold_bench()
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    variant = bench.threshold_variant(source, tmp_path / "standalone.py", bench.threshold_config("decode=8"))
    def grouped(text):
        core = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                    and n.name == "NativeTrellisLinear")
        return ast.dump(next(n for n in core.body if isinstance(n, ast.FunctionDef)
                             and n.name == "_generic_grouped"))
    assert grouped(variant.read_text()) == grouped(source.read_text())
    assert "_GROUPED_DECODE_ROWS = 16" in variant.read_text()


def test_reference_fixtures_require_prebuilt_official_encoder(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--encode-fixtures"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_kimi_decision_requires_encoded_fixtures_and_profile_before_cuda(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--kimi-decision"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_kimi_decision_keeps_independent_projection_and_native_remainder_shapes():
    source = _PATH.with_name("kimi_k6_dense.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "benchmark_kimi_decision")
    assert '("kda", 6144, 4, range(4), 24752)' in ast.get_source_segment(source, node)
    assert '("gate_up", 3072, 1, range(4, 6), 6144)' in ast.get_source_segment(source, node)
    assert 'outputs=176' in ast.get_source_segment(source, node)
    assert '"KDA and shared gate/up only, not a full-core speed estimate"' in source


def _kimi_profile_fixture():
    def entry(n, k, count, symbols):
        return dict(device=0, lane="target", geometry=dict(n=n, k=k), phases=dict(decode=dict(
            count=count, unit_count=255, unknown_rows=0, rows={"5": count}, symbols=symbols)))
    catalog = [entry(24752, 7168, 5037, ["kda"]),
               entry(6144, 7168, 8468, ["shared", "mla"]),
               entry(7168, 3072, 6716, ["down"]), entry(9216, 1536, 1752, ["qb"])]
    inventory = dict(phase="decode", unit="emitted-token", unit_count=255, kernels=[
        dict(device=0, name="shared", count=3000, rows={"5": 3000}),
        dict(device=0, name="shared", count=3716, rows={"5": 3716}),
        dict(device=0, name="mla", count=1752, rows={"5": 1752}),
        dict(device=1, name="shared", count=999, rows={"5": 999})])
    return catalog, inventory


def test_kimi_profile_accounting_separates_shared_gate_up_from_single_mla_gate():
    catalog, inventory = _kimi_profile_fixture()
    accounting = _native_threshold_bench().kimi_profile_accounting(catalog, inventory)
    assert accounting["calls_per_emitted_token"] == dict(kda=5037 / 255, gate_up=6716 / 255)
    assert accounting["symbols"] == {"packed shared gate/up": "shared", "single MLA h48 gate": "mla"}
    assert accounting["shared_gate_up_count"] + accounting["mla_gate_count"] == accounting["mixed_count"]


@pytest.mark.parametrize("bad", ["units", "missing_symbol", "count", "histogram", "unknown", "duplicate"])
def test_kimi_profile_accounting_refuses_ambiguous_or_incomplete_data(bad):
    catalog, inventory = _kimi_profile_fixture()
    if bad == "units":
        inventory["unit_count"] = 256
    elif bad == "missing_symbol":
        inventory["kernels"] = inventory["kernels"][:2]
    elif bad == "count":
        inventory["kernels"][0]["count"] -= 1
    elif bad == "histogram":
        inventory["kernels"][0]["rows"] = {"4": 3000}
    elif bad == "unknown":
        inventory["kernels"][0]["row_unknown"] = 1
    elif bad == "duplicate":
        catalog.append(catalog[0])
    with pytest.raises(ValueError):
        _native_threshold_bench().kimi_profile_accounting(catalog, inventory)


def test_kimi_decision_requires_symbol_inventory_before_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--kimi-decision",
                                     "--fixture-dir", str(tmp_path), "--profile-catalog", "catalog.json"])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def _native_route_policy():
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                (t.id.startswith("_K6_") or t.id.startswith("_GROUPED_") or
                 t.id in ("DECODE_ROWS_BY_SHAPE", "_FOLDED_DECODE_ROWS", "_GEMINI_NARROW_ROWS"))
                for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("decode_rows", "decode_route_thresholds"):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_measured_decode_rows_are_scoped_to_qualified_format_and_storage_sides():
    policy = _native_route_policy()
    build = policy["decode_rows"]
    for (bits, n, k), (edge, sides) in policy["DECODE_ROWS_BY_SHAPE"].items():
        for side in (1, 2):
            g = dict(bits=bits, output_features=n, input_features=k, sharded_side=side,
                     codebook="mcg", input_dtype="bf16", groups=1)
            assert build(g) == (edge if side in sides else 16)
            for key, value in (("codebook", "mul1"), ("input_dtype", "f16"), ("groups", 4)):
                assert build(dict(g, **{key: value})) == 16
    assert build(dict(bits=6, output_features=3072, input_features=7168,
                      sharded_side=1, codebook="mcg", input_dtype="bf16")) == 16


@pytest.mark.parametrize("bits,n,k,groups,expected", [
    (6, 6144, 7168, 1, (8, 32, 48)),
    (6, 6144, 7168, 4, (16, 32, 48)),
    (6, 3072, 7168, 1, (16, 32, 48)),
    (6, 163840, 7168, 1, (16, 32, 48)),
    (8, 6144, 7168, 1, (8, 32, 48)),
    (8, 6144, 7168, 4, (16, 32, 48)),
    (8, 7168, 3072, 1, (16, 32, 48)),
    (4, 6144, 7168, 1, ()),
])
def test_native_route_metadata_describes_each_actual_program(bits, n, k, groups, expected):
    policy = _native_route_policy()
    g = dict(bits=bits, output_features=n, input_features=k, splits=(k + 1023)//1024,
             sharded_side=1, codebook="mcg", input_dtype="bf16", groups=groups, max_rows=4096)
    assert policy["decode_route_thresholds"](g) == expected
    g["max_rows"] = 8
    assert policy["decode_route_thresholds"](g) == ()


def test_native_threshold_variants_only_change_legal_dispatch(tmp_path):
    bench = _native_threshold_bench()
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    original = source.read_text()
    unchanged = bench.threshold_variant(source, tmp_path / "defaults.py", bench.threshold_config(None))
    assert unchanged.read_text() == original
    values = bench.threshold_config("narrow=4,decode=8,wide2=24,wide3=32,prefill=64")
    variant = bench.threshold_variant(source, tmp_path / "variant.py", values).read_text()
    assert "_K6_NARROW_ROWS = 4" in variant
    assert "_PREFILL_MIN_ROWS = 64" in variant
    before, after = ast.parse(original), ast.parse(variant)
    base = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "NativeTrellisLinear")
    changed = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == "NativeTrellisLinear")
    methods = {n.name: n for n in changed.body if isinstance(n, ast.FunctionDef)}
    for method in (n for n in base.body if isinstance(n, ast.FunctionDef)):
        if method.name not in ("__init__", "_legacy", "_generic_grouped"):
            assert ast.dump(method) == ast.dump(methods[method.name]), method.name
    assert source.read_text() == original
