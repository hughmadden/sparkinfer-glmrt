"""CPU-only controls for fixed-export row sweeps and recorded route replay."""
from __future__ import annotations

import ast
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


def test_native_oai_geometry_and_cache_keys():
    from dataclasses import replace
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, GEOMETRIES, _StreamNvfp4A4Route

    base = Fp8MoeGeometry("oai_contract", hidden=256, experts=128, top_k=4,
                          intermediate=256, weights="nvfp4", activations="a4")
    oai = replace(base, activation="swigluoai", swiglu_limit=7.0,
                  swiglu_alpha=1.702, swiglu_beta=1.0)
    assert (oai.with_tp(2).swiglu_alpha, oai.with_tp(2).swiglu_beta) == (1.702, 1.0)
    key = _StreamNvfp4A4Route(oai, 129, False).key()
    assert key != _StreamNvfp4A4Route(base, 129, False).key()
    for field, value in [("swiglu_alpha", 1.5), ("swiglu_beta", 0.5), ("swiglu_limit", 6.0)]:
        assert key != _StreamNvfp4A4Route(replace(oai, **{field: value}), 129, False).key()
    for changes in [dict(swiglu_alpha=None), dict(swiglu_alpha=float("nan")),
                    dict(swiglu_alpha=0.0), dict(swiglu_beta=None),
                    dict(swiglu_beta=float("inf")), dict(swiglu_limit=0.0),
                    dict(weights="fp8", activations="a16")]:
        with pytest.raises(ValueError):
            replace(oai, **changes)
    with pytest.raises(ValueError, match="alpha/beta"):
        replace(base, swiglu_alpha=1.702)
    assert all(g.activation != "swigluoai" and g.swiglu_alpha is None
               and g.swiglu_beta is None for g in GEOMETRIES.values())


def test_v41_fc1_split_is_explicit_and_defaults_to_one():
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/v41_slice_pipeline.py"
    tree = ast.parse(path.read_text())
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "V41SlicePipeline")
    init = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    defaults = dict(zip((arg.arg for arg in init.args.kwonlyargs), init.args.kw_defaults))
    assert ast.literal_eval(defaults["fc1_k_splits"]) == 1
    assert "get_sm_version" not in path.read_text()
    assert "FC1_DECODE_K_SPLITS" not in path.read_text()


def test_seed_oracle_control_does_not_inherit_default_split():
    path = _PATH.parents[1] / "tests/moe/test_v41_exact_slices.py"
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_synthetic_oracle_seeds")
    arms = [node for node in ast.walk(helper) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "native_pair_arm"]
    control, candidate = arms
    assert isinstance(control.args[2], ast.Name) and control.args[2].id == "exact_control"
    assert next(kw.value.value for kw in control.keywords if kw.arg == "fc1_k_splits") == 1
    assert next(kw.value.id for kw in candidate.keywords if kw.arg == "fc1_k_splits") == "k_split"


def test_mxfp4_pair_widths_cover_all_worlds_without_padding(monkeypatch):
    monkeypatch.setattr(bench, "engine_layout_policy", lambda root: SimpleNamespace(
        ownership_unit=lambda intermediate, world: 64 if (intermediate, world) == (2304, 5) else 32))
    assert bench.paired_widths(2304) == [288, 320, 352, 384, 448, 512, 576, 768, 1152]

    assert bench.paired_widths(2048) == [256, 288, 320, 352, 384, 416, 512, 672, 704, 1024]
    assert bench.paired_widths(3072) == [384, 416, 448, 512, 608, 640, 768, 1024, 1536]
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--geometry", "v41", "--widths", "448,480"])
    assert args.native_slices and args.widths == "448,480"
    with pytest.raises(ValueError):
        bench.paired_widths(2048, "2080")


@pytest.mark.parametrize("family,rank_width,expected", [
    ("v41", 288, 192), ("v41", 576, 192), ("v41", 768, 192),
    ("v41", 448, 128), ("dsv4f", 384, 128), ("dsv4p", 768, 128),
])
def test_native_pair_retains_shipped_slice_recipe(family, rank_width, expected):
    assert bench.native_slice_width(family, rank_width) == expected


@pytest.mark.parametrize("argv", [["--native-slices"], ["--widths", "448"],
    ["--mxfp4-pair", "--widths", "33"], ["--mxfp4-pair", "--widths", "oops"],
    ["--mxfp4-pair", "--input", "bf16"], ["--mxfp4-pair", "--profile-dir", "out"]])
def test_invalid_pair_modes_fail_before_cuda(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


@pytest.mark.parametrize("width,expected", [(288, [1, 2, 3, 4, 8]), (576, [1, 2, 4, 5, 8]),
                                          (672, [1, 2, 4, 6, 8]), (608, [1, 2, 4, 5, 8])])
def test_tiling_screen_includes_old_four_warps_and_matching_partition(width, expected):
    assert bench.mxfp4_decode_partitions(width) == expected


def test_tiling_controls_are_development_pair_only():
    assert bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings"]).mxfp4_tilings
    for argv in (["--mxfp4-tilings"], ["--mxfp4-pair", "--mxfp4-tilings", "--capacity", "4096"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


@pytest.mark.parametrize("width", [288, 576])
def test_aligned_w2_changes_only_payload_tile_pitch(monkeypatch, width):
    from tests.moe.test_v41_exact_slices import pack
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    experts, hidden = 2, 256
    weights, scales = {}, {}
    for name, shape in (("w1", (experts, width, hidden // 2)),
                        ("w3", (experts, width, hidden // 2)),
                        ("w2", (experts, hidden, width // 2))):
        weights[name] = (torch.arange(shape[0] * shape[1] * shape[2]) % 251 + 1).byte().reshape(shape)
        scales[name] = torch.full((*shape[:-1], shape[-1] // 16), 122, dtype=torch.uint8)
    pitch = (width + 127) // 128 * 128
    exact = pack(weights, scales, True)
    aligned = pack(weights, scales, True, w2_pitch=pitch)
    for plane in (0, 1, 3):
        assert torch.equal(exact[plane], aligned[plane])
    expected = exact[2].reshape(experts, hidden // 128, width // 32, 512)
    actual = aligned[2].reshape(experts, hidden // 128, pitch // 32, 512)
    assert torch.equal(actual[:, :, :width // 32], expected)
    assert torch.count_nonzero(actual[:, :, width // 32:].view(torch.int32)) == 0


@pytest.mark.parametrize("width", [288, 576])
def test_w2_coded_pitch_is_already_aligned_and_extension_preserves_bytes(width):
    from b12x.moe._shared.scale_codes import pack_scale_codes
    columns = width // 32
    source = (torch.arange(256 * columns).reshape(1, 256, columns) % 3 + 121).byte()
    aligned_columns = (columns + 3) // 4 * 4
    extension = source.amin(-1, keepdim=True).expand(1, 256, aligned_columns - columns)
    exact = pack_scale_codes(source, tensor_name="w2.scale")
    aligned = pack_scale_codes(torch.cat((source, extension), -1), tensor_name="w2.scale")
    assert exact.shape == (1, 256 * (2 * ((columns + 3) // 4) + 1))
    assert torch.equal(exact, aligned)


def test_w2_pitch_arm_is_native_development_only():
    assert bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--native-slices",
                             "--mxfp4-w2-pitch"]).mxfp4_w2_pitch
    for argv in (["--mxfp4-w2-pitch"], ["--mxfp4-pair", "--mxfp4-tilings", "--mxfp4-w2-pitch"],
                 ["--mxfp4-pair", "--native-slices", "--mxfp4-w2-pitch"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


def test_f32_errors_measure_unrounded_reference():
    error = bench.f32_errors(torch.tensor([1., 3.]), torch.tensor([2., 5.]))
    assert error == pytest.approx(dict(max_abs=2., rms=(2.5 ** .5)))


@pytest.mark.parametrize("native", [False, True])
def test_mxfp4_f32_oracle_has_explicit_quantization_boundaries(native):
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows
    x = torch.full((1, 32), .25, dtype=torch.bfloat16)
    ids, routing = torch.zeros((1, 1), dtype=torch.int32), torch.full((1, 1), .5)
    weights = {name: torch.full((1, 32, 16), 0x22, dtype=torch.uint8) for name in ("w1", "w3", "w2")}
    scales = {name: torch.full((1, 32, 1), 126, dtype=torch.uint8) for name in weights}
    output = bench.mxfp4_f32_reference(x, ids, routing, weights, scales, 10. if native else 0., native=native)
    gate = torch.full((1, 32), 4.)
    if native:
        act = (torch.nn.functional.silu(gate) * gate * .5).bfloat16()
        _, act = wire_rows(act)
        expected = act.float().sum() * .5
    else:
        act = (torch.nn.functional.silu(gate).bfloat16().float() * gate).bfloat16()
        expected = act.float().sum() * .5 * .5
    assert torch.equal(output, torch.full((1, 32), expected.item()))


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
    ["--continue-on-fail"],
    ["--check-prefix-rows", "2048"],
    ["--check", "--check-prefix-rows", "0", "--capacity", "4096"],
    ["--check", "--check-prefix-rows", "4096", "--capacity", "4096"],
    ["--continue-on-fail", "--check", "--exact-sweep", "out"],
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


@pytest.mark.parametrize("changed", [False, True])
def test_shared_prefix_compares_snapshot_before_scratch_reuse(changed):
    full = torch.arange(12).reshape(4, 3).to(torch.bfloat16)
    prefix = full[:3].clone()
    if changed:
        prefix[1, 2] += 1
    def execute(rows):
        assert rows == 3
        full.fill_(99)
        return prefix
    result = bench.check_shared_prefix(full, 3, execute)
    assert result["bitwise_equal"] == (not changed)
    assert result["differing_rows"] == int(changed)
    assert (result["prefix_sha256"] == result["full_prefix_sha256"]) == (not changed)
    if changed:
        assert result["first_mismatch"]["element_index"] == [1, 2]
    assert bench.parse_args(["--check", "--capacity", "4096", "--check-prefix-rows", "2048"]).check_prefix_rows == 2048


def test_oracle_reports_global_rms_and_outlier_location():
    reference = torch.tensor([[0., 1.], [1., 1.]])
    output = reference.clone()
    output[0, 0] = .2
    result = bench.oracle_metrics(output, reference, nvfp4_a4=True)
    assert result["reference_rms"] == pytest.approx(3 ** .5 / 2)
    assert result["max_error"] == pytest.approx(dict(row=0, col=0, output=.2, reference=0., abs_error=.2))
    assert result["normalized_max"] == pytest.approx(.2 / (3 ** .5 / 2))
    assert result["elements_over_threshold"] == 1
    assert not result["passed"]
    with pytest.raises(AssertionError):
        bench.enforce_oracle(result, False)
    bench.enforce_oracle(result, True)
    assert bench.parse_args(["--check", "--continue-on-fail"]).continue_on_fail


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_oracle_continuation_never_turns_invalid_outputs_green(value):
    result = bench.oracle_metrics(torch.full((2, 2), value), torch.ones(2, 2), nvfp4_a4=True)
    assert not result["passed"]
    bench.enforce_oracle(result, True)


def test_oracle_equal_nonzero_output_passes():
    result = bench.oracle_metrics(torch.ones(2, 2), torch.ones(2, 2), nvfp4_a4=True)
    assert result["passed"] and result["elements_over_threshold"] == 0
    bench.enforce_oracle(result, False)


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


@pytest.mark.parametrize("tile", [32, 64, 128, 192])
@pytest.mark.parametrize("logical", [288, 416, 576, 608])
@pytest.mark.parametrize("atomic", [False, True])
def test_native_partial_allocation_has_exact_fc1_activation_extent(tile, logical, atomic):
    function = next(node for node in ast.parse(_PATH.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "native_pair_arm")
    assignment, = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "partial" for target in node.targets)]
    expression = compile(ast.Expression(assignment.value), str(_PATH), "eval")
    # Run the real allocation expression on CPU; CUDA dataflow is GPU-qualified separately.
    allocator = SimpleNamespace(empty=lambda shape, **kwargs: torch.empty(shape))
    partial = eval(expression, {"__builtins__": {}, "torch": allocator},
                   dict(logical=logical, width=tile, routes=96, atomic=atomic))
    assert partial.numel() * 4 == (4 if atomic else
        ((logical + tile - 1) // tile) * 96 * (tile + tile // 8))


def test_explicit_rank_extents_and_identical_controls():
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings",
                             "--mxfp4-arm", "compact:288:288:128",
                             "--mxfp4-arm", "pad64:288:320:64", "--control-repeats", "3"])
    specs, controls = bench.tiling_specs(args, 288, 192, (64, 128, 192))
    assert controls == ["padded", "padded-control-1", "padded-control-2"]
    assert specs == [(name, False, 192, None, 1) for name in controls] + [
        ("compact", True, 128, 288, 1), ("pad64", True, 64, 320, 1)]
    samples = {name: [1., 2., 3.] for name in controls}
    samples.update(compact=[2., 4., 6.], pad64=[4., 8., 12.])
    result = bench.tiling_timing_summary(samples, {name: True for name in samples}, controls)
    assert result["winner"] == "compact" and result["speedup"] == .5
    assert result["identical_control_speedups"] == {name: [1., 1., 1.] for name in controls[1:]}
    assert bench.tiling_timing_summary(samples, {name: False for name in samples}, controls)["winner"] is None


@pytest.mark.parametrize("value", ["bad", "a:0:128:64", "a:33:128:64", "a:128:96:64",
                                   "a:96:97:64", "a:96:96:16", "padded-x:96:96:64",
                                   "exact-64:96:96:64", "aligned-w2-x:96:96:64", "a/b:96:96:64"])
def test_invalid_explicit_extents_fail_before_cuda(value):
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings", "--mxfp4-arm", value])


def test_explicit_tile32_is_benchmark_only():
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings",
                             "--mxfp4-arm", "tile32:288:288:32"])
    specs, _ = bench.tiling_specs(args, 288, 192, (64, 128, 192))
    assert specs[-1] == ("tile32", True, 32, 288, 1)


def test_explicit_names_and_mode_are_unambiguous():
    with pytest.raises(ValueError, match="duplicate"):
        bench.native_screen_arms(["a:96:96:64", "a:128:128:64"])
    for argv in (["--mxfp4-arm", "a:96:96:64"], ["--control-repeats", "1"],
                 ["--mxfp4-pair", "--native-slices", "--mxfp4-tilings", "--widths", "96",
                  "--mxfp4-arm", "a:96:96:64"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


@pytest.mark.parametrize("stored", [96, 128, 160])
def test_explicit_padding_preserves_real_payload_and_scale_blocks(stored):
    logical, hidden, experts = 96, 128, 2
    shapes = {"w1": (experts, logical, hidden // 2), "w3": (experts, logical, hidden // 2),
              "w2": (experts, hidden, logical // 2)}
    weights = {name: torch.full(shape, 0x32, dtype=torch.uint8) for name, shape in shapes.items()}
    scales = {name: torch.full((*shape[:-1], shape[-1] // 16), 122, dtype=torch.uint8)
              for name, shape in shapes.items()}
    resident, factors = bench.extend_native_weights(weights, scales, stored)
    assert resident["w1"].shape == (experts, stored, hidden // 2)
    assert resident["w2"].shape == (experts, hidden, stored // 2)
    for name in ("w1", "w3"):
        assert torch.equal(resident[name][:, :logical], weights[name])
        assert torch.equal(factors[name][:, :logical], scales[name])
        assert not bool(resident[name][:, logical:].count_nonzero())
    assert torch.equal(resident["w2"][..., :logical // 2], weights["w2"])
    assert not bool(resident["w2"][..., logical // 2:].count_nonzero())
    assert torch.equal(factors["w2"][..., :logical // 32], scales["w2"])
    assert bool((factors["w2"] == 122).all())


def jsonl_routes(tmp_path, records):
    path = tmp_path / "routes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_existing_jsonl_replays_exact_narrow_rows_and_wide_histograms(tmp_path):
    narrow = dict(event="expert_routes", layer=7, lane=0, rows=3, topk=[[3, 0], [0, 1], [3, 0]])
    wide = dict(event="expert_routes", layer=8, lane=1, rows=80, expert_rows=[[0, 80], [1, 40], [3, 40]])
    waves = bench.recorded_waves(jsonl_routes(tmp_path, [narrow, wide]),
                                 SimpleNamespace(name="mimo", experts=4, top_k=2))
    assert waves[0]["expert_rows"] == [(0, 3), (1, 1), (3, 2)]
    assert bench.screen_route_wave(waves, 3, 0) is waves[0]
    assert bench.screen_route_wave(waves + [waves[0]], 3, 1) is waves[0]
    assert bench.screen_route_ids(waves[0], 4, 2, 1).tolist() == narrow["topk"]
    ids = bench.screen_route_ids(waves[1], 4, 2, 5)
    assert torch.bincount(ids.flatten().long(), minlength=4).tolist() == [80, 40, 0, 40]
    assert bool((ids[:, 0] != ids[:, 1]).all())
    assert torch.equal(ids, bench.screen_route_ids(waves[1], 4, 2, 5))
    with pytest.raises(ValueError, match="exactly 16"):
        bench.screen_route_wave(waves, 16, 0)


@pytest.mark.parametrize("record", [dict(event="other", rows=1, topk=[[0, 1]]),
    dict(event="expert_routes", rows=1, topk=[[0, 0]]),
    dict(event="expert_routes", rows=1, topk=[[0, 4]]),
    dict(event="expert_routes", rows=1, topk=[[0]]),
    dict(event="expert_routes", rows=2, topk=[[0, 1]]),
    dict(event="expert_routes", rows=1, expert_rows=[[0, 1]])])
def test_malformed_jsonl_routes_fail_closed(tmp_path, record):
    with pytest.raises(ValueError):
        bench.recorded_waves(jsonl_routes(tmp_path, [record]), SimpleNamespace(name="v41", experts=4, top_k=2))


@pytest.mark.parametrize("family", ["v41", "dsv4f", "dsv4p"])
def test_native_stream_screen_requires_shared_policy_and_prefill(family):
    args = ["--geometry", family, "--mxfp4-pair", "--native-slices", "--shipped-prefill",
            "--engine-policy-root", "engine", "--rows", "256,1024,4096"]
    assert bench.parse_args(args).shipped_prefill
    for extra in (["--rows", "80"], ["--capacity", "2752"], ["--mxfp4-tilings"]):
        with pytest.raises(SystemExit):
            bench.parse_args(args + extra)


@pytest.mark.parametrize("capacity", [80, 256, 1024, 4096])
@pytest.mark.parametrize("width", [384, 416, 448, 512, 608, 640, 768, 1024, 1536])
def test_pro_stream_baseline_matches_shipped_equal_and_unequal_roles(monkeypatch, capacity, width):
    calls = []
    def tile(hidden, rank_width, sm, rows, matching):
        calls.append((hidden, rank_width, sm, rows, matching))
        return 128
    monkeypatch.setattr(bench, "engine_layout_policy", lambda root: SimpleNamespace(
        native_slice_width=tile, stored_width=lambda hidden, rank_width: rank_width))
    recipe = bench.shipped_prefill_recipe("dsv4p", 7168, width, capacity, "engine")
    assert calls == [(7168, width, 121, capacity, 128)]
    assert recipe == dict(tile=128, exact=True, coded=False,
                         atomic=capacity >= 256 and width in (384, 512, 768, 1024, 1536))


@pytest.mark.parametrize("capacity", [80, 256, 1024, 4096])
@pytest.mark.parametrize("width", [256, 288, 352, 512, 704, 1024])
def test_flash_stream_baseline_retains_ordered_output(monkeypatch, capacity, width):
    monkeypatch.setattr(bench, "engine_layout_policy", lambda root: SimpleNamespace(
        native_slice_width=lambda hidden, rank_width, sm, rows, matching: matching,
        stored_width=lambda hidden, rank_width: rank_width))
    assert bench.shipped_prefill_recipe("dsv4f", 4096, width, capacity, "engine") == dict(
        tile=128, exact=True, coded=False, atomic=False)
    screen = next(node for node in ast.parse(_PATH.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "shipped_stream_screen")
    rtx = next(node for node in ast.walk(screen) if isinstance(node, ast.If)
               and ast.unparse(node.test) == "sm == 120")
    recipe = next(node.value for node in rtx.body if isinstance(node, ast.Assign))
    assert next(keyword.value.value for keyword in recipe.keywords if keyword.arg == "atomic") is False


def test_pro_stream_screen_rejects_unqualified_rtx_before_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(major=12, minor=0))
    with pytest.raises(ValueError, match="GB10"):
        bench.shipped_stream_screen(SimpleNamespace(name="dsv4p"), SimpleNamespace())


@pytest.mark.parametrize("capacity", [256, 1024, 4096])
@pytest.mark.parametrize("width", [288, 352, 512, 576, 1152, 2304])
def test_native_stream_scratch_matches_every_static_tensor(capacity, width):
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/v4_stream_pipeline.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(stream_max_tiles=lambda e, p: (p + 127) // 128 + min(e, p),
                     meta_words=lambda e, t: 4 + 3 * e + t,
                     mxfp8_down_row_bytes=lambda i: (i + i // 32 + 15) // 16 * 16)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(path), "exec"), namespace)
    layout, size = namespace["stream_scratch_layout"](capacity, 5120, width, 384, 6)
    assert [row["name"] for row in layout] == ["meta", "pair_row", "pair_pos", "act",
                                               "grouped_routing", "grouped_route", "output"]
    end = 0
    for row in layout:
        assert row["offset"] == (end + 15) // 16 * 16
        expected = 1 if row["dtype"] == "u8" else 4
        for extent in row["shape"]:
            expected *= extent
        assert row["nbytes"] == expected
        end = row["offset"] + expected
    assert size == end
    assert layout[-1]["shape"] == (capacity * 6, 5120)
    assert layout[-2]["shape"] == (capacity * 6,)


def test_native_stream_export_to_c_signature_is_supported_without_cuda():
    import inspect

    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass.base_dsl.common import DSLRuntimeError
    from cutlass.base_dsl.jit_executor import ExecutionArgs
    from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator
    from cutlass.cute.runtime import make_ptr

    from b12x.moe._shared.kernels.v4_stream_pipeline import V4StreamPipeline

    signature = inspect.signature(V4StreamPipeline.__call__)
    signature = signature.replace(parameters=list(signature.parameters.values())[1:])
    pointer_names = [name for name in signature.parameters if name not in ("rows", "stream")]
    pointer_types = (cutlass.Uint8, cutlass.Uint32, cutlass.Uint8, cutlass.Uint32,
                     cutlass.Uint8, cutlass.Int32, cutlass.Float32, cutlass.Int32,
                     cutlass.Int32, cutlass.Int32, cutlass.Uint8, cutlass.Float32,
                     cutlass.Int32, cutlass.Float32)
    assert len(pointer_names) == len(pointer_types) == 14
    args = tuple(make_ptr(dtype, 0, assumed_align=16) for dtype in pointer_types)
    args += (cutlass.Int32(256), cuda.CUstream(0))
    generator = CuteCHeaderGenerator()
    # This is the checker called by export_to_c, not a static annotation check.
    arguments, packed, declarations = generator._generate_arguments(
        "v4_stream", ExecutionArgs(signature, "V4StreamPipeline"), args, {})
    assert arguments == [*(f"void *{name}" for name in pointer_names),
                         "int32_t rows", "cudaStream_t stream"]
    assert packed == [f"&{name}" for name in signature.parameters]
    assert declarations == []

    untyped = signature.replace(parameters=[
        param.replace(annotation=inspect.Parameter.empty) if name == "stream" else param
        for name, param in signature.parameters.items()])
    with pytest.raises(DSLRuntimeError, match="Unsupported argument.*stream"):
        generator._generate_arguments(
            "v4_stream", ExecutionArgs(untyped, "V4StreamPipeline"), args, {})
    assert not torch.cuda.is_initialized()


def test_native_stream_has_ordered_float_output_and_no_atomic_reduction():
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/v4_stream_pipeline.py"
    source = path.read_text()
    assert "red_add" not in source and "atomic" not in source.split('"""', 2)[-1]
    assert "value = fadd_rn_f32" in source
    assert "ordered_combine" not in source
    assert "grouped_route[position] = index" in source
    assert "original_routes=True" in source
    down = (_PATH.parents[1] / "b12x/integration/cuteafd/_mxfp4_moe_a8.py").read_text()
    assert "route = Int64(grouped_route[route])" in down
    assert '"f32_original_routes"' in down
    assert "scale_codes=scale_codes" in source
    qualifier = ast.unparse(next(node for node in ast.parse(_PATH.read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name == "qualify_pair_replay"))
    assert "repeat determinism" in qualifier and "torch.equal(output, expected)" in qualifier


def test_stream_fixed_capacity_and_resident_experts():
    base = ["--geometry", "v41", "--mxfp4-pair", "--native-slices", "--shipped-prefill",
            "--engine-policy-root", "engine", "--rows", "256,1024,2752,4096", "--capacity", "4096"]
    args = bench.parse_args(base + ["--experts", "384", "--routes", "routes.jsonl"])
    assert args.capacity == 4096 and args.experts == 384
    for extra in (["--experts", "0"], ["--capacity", "1024"], ["--routing", "hot", "--routes", "routes.jsonl"]):
        with pytest.raises(SystemExit):
            bench.parse_args(base + extra)
    with pytest.raises(SystemExit):
        bench.parse_args(["--experts", "384"])


def test_stream_capacity_padding_preserves_live_inputs():
    x = torch.randn(3, 8)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
    mix = torch.rand(3, 2)
    padded = bench.screen_capacity_inputs(x, ids, mix, 8)
    for original, actual in zip((x, ids, mix), padded):
        assert actual.shape[0] == 8
        assert torch.equal(actual[:3], original)
    assert bool((padded[1][3:] == -1).all())
    assert not bool(padded[0][3:].count_nonzero())
    assert not bool(padded[2][3:].count_nonzero())
    assert bench.screen_capacity_inputs(x, ids, mix, None)[0] is x
    with pytest.raises(ValueError, match="cover live"):
        bench.screen_capacity_inputs(x, ids, mix, 2)


@pytest.mark.parametrize("family,experts", [("v41", 384), ("dsv4f", 256), ("dsv4p", 384)])
def test_stream_default_uses_checkpoint_population_and_hot_is_diagnostic(family, experts):
    args = bench.parse_args(["--geometry", family, "--mxfp4-pair", "--native-slices",
                             "--shipped-prefill", "--engine-policy-root", "engine", "--rows", "256"])
    geometry = SimpleNamespace(experts=experts, top_k=6)
    assert args.routing == "random"
    assert bench.stream_screen_population(geometry, args) == (experts, True)
    args.routing = "hot"
    assert bench.stream_screen_population(geometry, args) == (experts, False)
    args.routing, args.experts = "random", 16
    assert bench.stream_screen_population(geometry, args) == (16, False)


@pytest.mark.parametrize("tilings", [False, True])
@pytest.mark.parametrize("native,experts,topk", [(True, 384, 6), (False, 384, 8)])
def test_pair_population_defaults_to_checkpoint_and_labels_diagnostics(tilings, native, experts, topk, capsys):
    argv = ["--mxfp4-pair"] + (["--mxfp4-tilings"] if tilings else [])
    argv += ["--native-slices", "--geometry", "v41"] if native else ["--geometry", "mimop"]
    geometry = SimpleNamespace(experts=experts, top_k=topk)
    args = bench.parse_args(argv)
    assert args.routing == "random"
    assert bench.stream_screen_population(geometry, args) == (experts, True)
    assert not capsys.readouterr().out
    for flags in (["--routing", "hot"], ["--experts", "16"]):
        diagnostic = bench.parse_args(argv + flags)
        assert not bench.stream_screen_population(geometry, diagnostic)[1]
        assert "DIAGNOSTIC ONLY" in capsys.readouterr().out
    same = bench.parse_args(argv + ["--experts", str(experts)])
    assert bench.stream_screen_population(geometry, same) == (experts, True)
    with pytest.raises(ValueError, match="top-k"):
        bench.stream_screen_population(geometry, bench.parse_args(argv + ["--experts", str(topk - 1)]))


def test_recorded_pair_population_cannot_be_custom():
    args = bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--routes", "routes.jsonl", "--experts", "16"])
    with pytest.raises(ValueError, match="checkpoint expert count"):
        bench.stream_screen_population(SimpleNamespace(experts=384, top_k=8), args)


@pytest.mark.parametrize("function", ["mxfp4_pair_screen", "mxfp4_tiling_screen"])
def test_pair_paths_use_shared_population_not_l2_derived_experts(function):
    tree = ast.parse(_PATH.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    population = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                  and isinstance(call.func, ast.Name) and call.func.id == "stream_screen_population"]
    assert len(population) == 1
    assert "route_qualification_eligible" in ast.unparse(node)
    assert "max(16," not in ast.unparse(node)
    assert "topk = g.top_k" in ast.unparse(node)


def test_explicit_split_defaults_and_shipped_identical_controls():
    args = bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--native-slices",
                            "--mxfp4-arm", "old:384:384:192",
                            "--mxfp4-arm", "shipped:384:384:128:2",
                            "--mxfp4-control", "shipped", "--control-repeats", "3"])
    specs, controls = bench.tiling_specs(args, 384, 192, (64, 128, 192))
    assert ("old", True, 192, 384, 1) in specs
    assert controls == ["shipped", "shipped-control-1", "shipped-control-2"]
    for name in controls:
        assert (name, True, 128, 384, 2) in specs
    samples = {"padded": [20., 20.], "old": [20., 20.], **{name: [10., 10.] for name in controls}}
    summary = bench.tiling_timing_summary(samples, {name: True for name in samples}, controls)
    assert summary["paired_speedups"]["old"] == .5
    assert summary["identical_control_speedups"] == {name: [1., 1.] for name in controls[1:]}


@pytest.mark.parametrize("arm", ["x:384:384:128:0", "x:384:384:128:3", "x:384:384:128:2:8"])
def test_invalid_native_split_fails_before_cuda(arm):
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--native-slices", "--mxfp4-arm", arm])


def test_explicit_control_requires_matching_arm():
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--mxfp4-control", "absent"])
