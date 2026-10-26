"""CPU-only controls for fixed-export row sweeps and recorded route replay."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PATH = Path(__file__).resolve().parents[2] / "benchmarks/bench_cuteafd_fp8_moe.py"
_SPEC = importlib.util.spec_from_file_location("fp8_moe_benchmark", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_capacity_is_independent_of_live_rows():
    args = bench.parse_args(["--capacity", "4096", "--rows", "512,2048,2752,4096"])
    assert args.capacity == 4096
    assert args.rows == "512,2048,2752,4096"
    assert bench.parse_args([]).capacity is None


@pytest.mark.parametrize("argv", [
    ["--capacity", "0"], ["--capacity", "2048", "--rows", "2752"],
    ["--rows", "0"], ["--rows", "-1"], ["--rows", ""], ["--rows", "1,broken"],
    ["--routes", "routes.json", "--routing", "hot"],
    ["--capacity", "4096", "--exact-sweep", "out"],
    ["--routes", "routes.json", "--exact-sweep", "out"],
    ["--route", "auto,auto"], ["--route", "auto,"],
    ["--compare-routes-exact"],
    ["--compare-routes-exact", "--route", "decode,auto", "--exact-sweep", "out"],
])
def test_invalid_cases_fail_before_cuda(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


def test_fixed_capacity_reuses_compiled_callable_and_workspace(monkeypatch):
    from b12x.integration.cuteafd import fp8_moe

    calls = []
    def compile_program(g, **kwargs):
        calls.append(("compile", kwargs))
        return object()
    def scratch_bytes(g, route, capacity, wire, prefill):
        calls.append(("scratch", capacity))
        return capacity
    monkeypatch.setattr(fp8_moe, "compile_fp8_moe_aot", compile_program)
    monkeypatch.setattr(fp8_moe, "fp8_moe_scratch_bytes", scratch_bytes)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"}))
    args = bench.parse_args(["--capacity", "4096", "--rows", "2048,2752"])
    cache = {}
    first = bench.compiled_case(None, args, 2048, "auto", cache)
    second = bench.compiled_case(None, args, 2752, "auto", cache)
    assert first is second
    assert first[1].numel() == 4096
    assert calls == [("compile", dict(route="auto", max_rows=4096, wire=True, prefill="auto")),
                     ("scratch", 4096)]
    args.capacity = None
    bench.compiled_case(None, args, 2048, "auto", cache)
    assert calls[-2][1]["max_rows"] == 2048


def test_exact_comparison_reports_raw_bytes_and_first_element():
    reference = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    changed = reference.clone()
    changed[1, 0] = 3.5
    result = bench.compare_route_outputs(dict(decode=reference, auto=reference.clone(), prefill=changed))
    assert result["auto"] == dict(reference_route="decode", bitwise_equal=True,
                                  first_mismatch=None, max_abs_diff=0.)
    assert not result["prefill"]["bitwise_equal"]
    assert result["prefill"]["first_mismatch"]["element_index"] == [1, 0]
    assert result["prefill"]["max_abs_diff"] == .5
    assert bench.parse_args(["--compare-routes-exact", "--route", "decode,auto"]).compare_routes_exact


def test_exact_comparison_distinguishes_signed_zero():
    result = bench.compare_route_outputs(dict(a=torch.tensor([[0.]], dtype=torch.bfloat16),
                                             b=torch.tensor([[-0.]], dtype=torch.bfloat16)))
    assert not result["b"]["bitwise_equal"]
    assert result["b"]["max_abs_diff"] == 0.


@pytest.mark.parametrize("actual", [torch.ones(1, 3), torch.ones(2, 2, dtype=torch.bfloat16)])
def test_exact_comparison_rejects_output_contract_mismatch(actual):
    with pytest.raises(ValueError, match="shapes and dtypes"):
        bench.compare_route_outputs(dict(a=torch.ones(2, 2), b=actual))


def fixture(tmp_path, **updates):
    data = dict(schema=1, family="glm", model="zai-org/GLM-5.3", experts=4, top_k=2,
                waves=[dict(layer=7, lane=0, rows=4, expert_rows=[[0, 4], [1, 2], [2, 2]])])
    data.update(updates)
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(data))
    return path


def test_reuses_exlr8_loader_and_histogram_reconstruction(tmp_path):
    from benchmarks.benchmark_exlr8_a16 import real_route_ids

    geometry = SimpleNamespace(name="glm_nvfp4", experts=4, top_k=2)
    waves = bench.recorded_waves(fixture(tmp_path), geometry)
    ids = real_route_ids(4, waves[0], torch.Generator().manual_seed(5), 4, 2)
    assert torch.bincount(ids.flatten(), minlength=4).tolist() == [4, 2, 2, 0]
    assert bool((ids[:, 0] != ids[:, 1]).all())
    replay = real_route_ids(4, waves[0], torch.Generator().manual_seed(5), 4, 2)
    assert torch.equal(ids, replay)


@pytest.mark.parametrize("updates", [
    dict(family="step5"), dict(experts=5), dict(top_k=3), dict(schema=2), dict(waves=[]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4], [0, 4]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 5], [1, 3]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4], [4, 4]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4]])]),
])
def test_wrong_family_or_malformed_histograms_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        bench.recorded_waves(fixture(tmp_path, **updates), SimpleNamespace(name="glm", experts=4, top_k=2))


def test_legacy_glm_fixture_is_not_relabelled_for_other_families():
    path = _PATH.parent / "fixtures/glm53_prefill_routes_realistic.json"
    assert bench.recorded_waves(path, SimpleNamespace(name="glm", experts=256, top_k=8))
    with pytest.raises(ValueError, match="family"):
        bench.recorded_waves(path, SimpleNamespace(name="mimo", experts=256, top_k=8))
