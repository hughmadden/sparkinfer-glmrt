"""CPU-only serving geometry contracts; no CUDA imports required."""
import ast
import importlib.util
import os
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_exlr8_a16.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "rank_geometry")
namespace = {}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), namespace)
rank_geometry = namespace["rank_geometry"]


@pytest.mark.parametrize("world", range(1, 7))
@pytest.mark.parametrize("layer", [0, 3, 40, 77])
def test_rotated_ownership_partitions_each_expert(world, layer):
    all_owned = [rank_geometry(world, rank, layer) for rank in range(world)]
    assert sum(width for _, width in all_owned) == 2048
    for expert in range(256):
        assert sorted(u for owned, _ in all_owned for u in owned[expert]) == list(range(16))
        for rank, (owned, width) in enumerate(all_owned):
            assert width == 128 * len(range(rank, 16, world))
            assert all((16 * expert + u + layer) % world == rank for u in owned[expert])


def test_tp6_keeps_global_slots_and_rotating_heavy_experts():
    r0, w0 = rank_geometry(6, 0, 3)
    r4, w4 = rank_geometry(6, 4, 3)
    assert (w0, w4) == (384, 256)
    assert r0[:3] == [(3, 9, 15), (5, 11), (1, 7, 13)]
    assert r4[:3] == [(1, 7, 13), (3, 9, 15), (5, 11)]
    assert {len(u) for u in r0 + r4} == {2, 3}
    text = SOURCE.read_text()
    assert "units = 16 if owned is not None else 4" in text
    assert "bits[e, list(range(units) if owned is None else owned[e]), :] = 4" in text
    assert "grouped = compile_exlr8_a16_aot(units=units" in text
    assert "state = buffers(4096, 6144, units" in text


@pytest.mark.parametrize("args", [(0, 0, 3), (17, 0, 3), (6, -1, 3), (6, 6, 3), (6, 0, -1)])
def test_invalid_geometry(args):
    with pytest.raises(ValueError):
        rank_geometry(*args)


def test_against_engine_tp_placement():
    path = os.environ.get("CUTEAFD_TP_PLACEMENT")
    if not path:
        pytest.skip("set CUTEAFD_TP_PLACEMENT to the engine's scripts/bench/tp-placement.py")
    spec = importlib.util.spec_from_file_location("engine_tp_placement", path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    geometry = engine.Geometry(experts=256)
    for world in range(1, 7):
        for layer in (0, 3, 40, 77):
            for rank in range(world):
                owned, width = rank_geometry(world, rank, layer)
                expected = engine.rank_units("rotated", geometry, layer, world, rank)
                assert owned == [expected.units[e] for e in range(256)]
                contiguous = engine.rank_units("contiguous", geometry, layer, world, rank)
                assert width == 128 * len(contiguous.units[0])


@pytest.mark.parametrize("width,tile", [(384, (64, 192, 64, 256)),
                                       (256, (64, 256, 64, 256))])
def test_serving_k4_ws_tile_contract(width, tile):
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serving_k4_options")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    options = scope["serving_k4_options"](width)
    assert options["force_tile_config"] == tile
    assert width % tile[1] == 0
    assert 6144 % tile[3] == 0
    assert options == dict(force_tile_config=tile, moe_block_size=64,
                          warp_specialized=True, input_format="e4m3_k32",
                          ws_input_stages=3, ws_dynamic_tiles=True, force_blocks_per_sm=1)
    # Unequal N is a WS feature, not a cooperative launch with unequal CTA threads.
    text = SOURCE.read_text()
    assert 'serving_k4_states(total, exl_layers, props, width)' in text
    assert 'if owned is None else' in text
    assert 'finite_nonzero(uniform_states[-1][1].output[:total].float()' in text


def test_serving_k4_against_sealed_package():
    import json
    root = os.environ.get("CUTEAFD_EXL3_PACKAGE")
    if not root:
        pytest.skip("set CUTEAFD_EXL3_PACKAGE to sealed GLM K4/K5 package")
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serving_k4_options")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    for rank in (0, 4):
        meta = json.loads((Path(root) / f"tp6-rank{rank}/m4096/v41_exl3.json").read_text())
        options = scope["serving_k4_options"](meta["intermediate"])
        assert list(options.pop("force_tile_config")) == meta["tile"]
        assert options.pop("moe_block_size") == meta["route_block"]
        assert options.pop("force_blocks_per_sm") == meta["blocks_per_sm"]
        assert all(value == meta[key] for key, value in options.items())
        assert (meta["capacity"], meta["experts"], meta["bits"]) == (4096, 256, [4, 5])


def test_only_unrouted_k5_aliases_across_cold_copies():
    calls = []
    def prepared(bits, experts, seed, width):
        result = object()
        calls.append((bits, experts, seed, width, result))
        return result
    functions = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("prepare_comparison_layers", "comparison_route_ids")]
    scope = {"prepare_uniform": prepared}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), scope)
    layers = scope["prepare_comparison_layers"](3, 384, True)
    assert len({id(k4) for k4, _ in layers}) == 3
    assert len({id(k5) for _, k5 in layers}) == 1
    assert [(b, e, s) for b, e, s, _, _ in calls] == [
        (5, 256, 409), (4, 256, 408), (4, 256, 410), (4, 256, 412)]
    import numpy as np
    class FakeTorch:
        @staticmethod
        def randperm(count, generator):
            assert count == 256
            return generator.permutation(count)
        stack = staticmethod(np.stack)
    scope["torch"] = FakeTorch
    routes = scope["comparison_route_ids"](4096, np.random.default_rng(408))
    assert routes.shape == (4096, 8)
    assert routes.min() == 0 and routes.max() == 255
    assert all(len(set(row)) == 8 for row in routes)
    text = SOURCE.read_text()
    assert "mapping = mapping[:256]" in text
    assert "ids = comparison_route_ids(total, gen).cuda().int()" in text
    calls.clear()
    legacy = scope["prepare_comparison_layers"](3, 512, False)
    assert len({id(k5) for _, k5 in legacy}) == 3
    assert [e for b, e, _, _, _ in calls if b == 5] == [1, 1, 1]


def test_serial_k4_cold_layers_reuse_one_workspace():
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serving_k4_states")
    allocated = []
    def state(rows, tiers, props, width, serving=False, workspace=None):
        assert serving
        if workspace is None:
            workspace = object()
            allocated.append(workspace)
        return tiers, workspace
    scope = {"grouped_mixed_state": state}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    layers = [object(), object(), object()]
    states = scope["serving_k4_states"](4096, layers, None, 384)
    assert len(allocated) == 1
    assert all(result[0] is original for result, original in zip(states, layers))
    assert all(result[1] is allocated[0] for result in states)


def test_sequential_exlr8_arms_alias_grouped_scratch():
    from types import SimpleNamespace
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serial_scratch")
    allocations = []
    class Tensor:
        def __init__(self, shape, dtype, contiguous=True):
            self.shape, self.dtype, self.contiguous = shape, dtype, contiguous
        def numel(self):
            import math
            return math.prod(self.shape)
        def is_contiguous(self):
            return self.contiguous
    def empty(shape, dtype, device):
        assert device == "cuda"
        result = Tensor(shape, dtype)
        allocations.append(result)
        return result
    scope = {"torch": SimpleNamespace(float32="f32", float16="f16", empty=empty)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    scratch = scope["serial_scratch"]
    serial = scratch(6144, 16, 512)
    assert len(allocations) == 3
    grouped = {"fc1": Tensor((768, 16, 2, 6, 64, 128), "f32"),
               "intermediate": Tensor((768, 16, 64, 128), "f16"),
               "fc2": Tensor((768, 64, 6144), "f32")}
    alias = scratch(6144, 16, 512, grouped)
    assert len(allocations) == 3
    assert all(alias[name] is grouped[name] for name in serial)
    assert sum(t.numel() * (2 if t.dtype == "f16" else 4)
               for t in serial.values()) == 1040187392
    for invalid in (Tensor((1,), "f32"), Tensor(grouped["fc1"].shape, "f16"),
                    Tensor(grouped["fc1"].shape, "f32", contiguous=False)):
        with pytest.raises(ValueError, match="shared fc1 scratch"):
            scratch(6144, 16, 512, dict(grouped, fc1=invalid))
    text = SOURCE.read_text()
    assert 't.update(serial_scratch(6144, units, 512, state))' in text
    grouped_source = text[text.index('def grouped_main('):]
    assert grouped_source.index('exl_layers = prepare_comparison_layers') < grouped_source.index('state = buffers(')
    assert grouped_source.index('del t[name]') < grouped_source.index('state = buffers(')


def test_compact_ownership_cpu():
    torch = pytest.importorskip("torch")
    path = SOURCE
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "compact_unit_descriptors")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    build = scope["compact_unit_descriptors"]
    owned, _ = rank_geometry(6, 0, 3)
    bits = torch.zeros((256, 16, 3), dtype=torch.int32)
    for expert, units in enumerate(owned):
        bits[expert, list(units)] = torch.tensor([2, 3, 4], dtype=torch.int32)
    counts, mapping, limit = build(bits)
    assert limit == 3 and int(counts.sum()) == sum(map(len, owned))
    for expert, units in enumerate(owned):
        assert mapping[expert, :counts[expert]].tolist() == list(units)
        assert not bool(mapping[expert, counts[expert]:].any())
    zero = build(torch.zeros_like(bits))
    assert zero[2] == 1 and not bool(zero[0].any())
    bits[0, 0, 0] = 2
    with pytest.raises(ValueError, match="agree"):
        build(bits)


def test_dense_mx8_geometry_has_distinct_record_abi():
    source = SOURCE.parents[1] / "b12x/integration/cuteafd/dense_mxfp8.py"
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("geometry", "record_bytes")]
    scope = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    geometry = scope["geometry"](2048, 16384)
    assert geometry["abi_version"] == 2
    assert geometry["input_dtype"] == "f16" and geometry["rotation_count"] == 4
    assert scope["record_bytes"](4096, 2048) == 256 * 64 * 528


def audit_helpers(*names, **scope):
    source = SOURCE.with_name("bench_exlr8_grouped_prefill.py")
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return scope


def test_audit_frozen_reference_rejects_edits_and_ignores_candidate(tmp_path):
    import hashlib
    import importlib.util
    import re
    candidate = tmp_path / "candidate.py"
    frozen = tmp_path / "reference.py"
    source = b"class Exlr8GroupedKernel:\n    VALUE = 'frozen'\n"
    candidate.write_bytes(source)
    frozen.write_bytes(source)
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    load = audit_helpers("load_reference_core", Path=Path, hashlib=hashlib,
                         importlib=importlib, re=re, sys=sys)["load_reference_core"]
    first, identity = load(str(frozen) + "@" + blob)
    candidate.write_text("raise RuntimeError('candidate edited')\n")
    second, second_identity = load(str(frozen) + "@" + blob)
    assert first.VALUE == second.VALUE == "frozen" and identity == second_identity
    frozen.write_text("raise RuntimeError('must not execute altered reference')\n")
    with pytest.raises(ValueError, match="blob mismatch"):
        load(str(frozen) + "@" + blob)
    with pytest.raises(ValueError, match="PATH@"):
        load(str(frozen))


@pytest.mark.parametrize("layer", [7, 47])
def test_audit_manifest_tp2_keeps_exact_tiers_and_global_ids(layer):
    entry = dict(gate=[[2, 3, 4, 2] for _ in range(3)],
                 up=[[4, 3, 2, 4] for _ in range(3)], down=[2, 3, 4])
    manifest = {"layers": {str(layer): entry}}
    build = audit_helpers("manifest_widths")["manifest_widths"]
    halves = [build(manifest, layer, rank, experts=3, units=4) for rank in (0, 1)]
    for expert in range(3):
        for unit in range(4):
            owner = (4 * expert + unit + layer) % 2
            assert halves[owner][expert][unit] == [entry["gate"][expert][unit],
                entry["up"][expert][unit], entry["down"][expert]]
            assert halves[1-owner][expert][unit] == [0, 0, 0]
    entry["down"][0] = 5
    with pytest.raises(ValueError, match="K2/K3/K4"):
        build(manifest, layer, 0, experts=3, units=4)
    with pytest.raises(ValueError, match="rank"):
        build(manifest, layer, 2, experts=3, units=4)


def test_audit_existing_arm_inventory_and_balanced_order():
    scope = audit_helpers("audit_specs", "audit_order")
    prefill = scope["audit_specs"](False)
    assert prefill[:2] == [("current", 64, {}), ("current_control", 64, {})]
    assert {rows for _, rows, _ in prefill} == {16, 32, 64}
    assert all("FC2_HALF" not in knobs and "REDUCE_MLP" not in knobs for _, _, knobs in prefill)
    # FC2 ring indexes with sequence & (stages - 1), unlike FC1's modulo.
    assert all((knobs.get("FC2_STAGES", 4) & (knobs.get("FC2_STAGES", 4) - 1)) == 0
               for _, _, knobs in prefill)
    decode = scope["audit_specs"](True)
    assert {label for label, _, _ in decode} >= {"k1", "k23", "compact", "current"}
    labels = [label for label, _, _ in prefill]
    orders = [scope["audit_order"](labels, r) for r in range(len(labels))]
    assert all(sorted(order) == sorted(labels) for order in orders)
    assert all(len({order[i] for order in orders}) == len(labels) for i in range(len(labels)))


@pytest.mark.parametrize("group_rows", [16, 32, 64])
@pytest.mark.parametrize("reverse", [False, True])
def test_audit_packing_uses_global_ids_and_invalid_tails(group_rows, reverse):
    torch = pytest.importorskip("torch")
    pack = audit_helpers("audit_pack", torch=torch)["audit_pack"]
    ids = torch.tensor([[255, 1], [1, 128], [255, 128]])
    capacity = 256
    state = dict(group_expert=torch.empty(capacity, dtype=torch.int32),
                 route_slots=torch.empty((capacity, group_rows), dtype=torch.int32),
                 inverse=torch.empty((3, 2), dtype=torch.int32), count=torch.empty(1, dtype=torch.int32))
    active = pack(ids, state, group_rows, reverse)
    assert active == int(state["count"][0]) == 3
    assert set(state["group_expert"][:active].tolist()) == {1, 128, 255}
    flat = state["route_slots"].flatten()
    for slot, inverse in enumerate(state["inverse"].flatten().tolist()):
        assert int(flat[inverse]) == slot
        group = inverse // group_rows
        assert int(state["group_expert"][group]) == int(ids.flatten()[slot])
    assert bool((state["route_slots"][active:] == -1).all())
    assert bool((state["group_expert"][active:] == -1).all())


def test_dense_bf16_comparison_argument_plumbing():
    import argparse
    from types import SimpleNamespace
    source = SOURCE.with_name("benchmark_exlr8_dense.py")
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = []
    for node in main.body:
        body.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.mxfp8_prefill":
            break
    main.body = body
    seen = []
    class Parser(argparse.ArgumentParser):
        def parse_args(self):
            return super().parse_args(["--mxfp8-prefill", "--compare-native-bf16",
                "--formats", "trellis", "--shapes", "q_b,o,gate_up,down", "--rows", "4096",
                "--output", "/unused/results.json"])
    scope = dict(argparse=SimpleNamespace(ArgumentParser=Parser), Path=Path, __doc__="test",
                 validate_timing=lambda *a: None,
                 benchmark_native_bf16=lambda args: seen.append(args) or 17,
                 benchmark_mx8=lambda args: pytest.fail("legacy route selected"))
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), scope)
    assert scope["main"]() == 17
    args, = seen
    assert args.rows == "4096" and args.formats == "trellis"
    assert args.shapes == "q_b,o,gate_up,down"


@pytest.mark.parametrize("queried,cap,expected", [(128, None, 128), (128, 96, 96), (96, 128, 96)])
def test_grouped_runtime_l2_query_and_cap(monkeypatch, queried, cap, expected):
    from types import SimpleNamespace
    import types
    calls = []
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=38),
        cuDeviceGet=lambda ordinal: (0, ordinal + 10),
        cuDeviceGetAttribute=lambda attr, device: calls.append((attr, device)) or (0, queried * 2**20))
    module = types.ModuleType("cuda.bindings")
    module.driver = driver
    monkeypatch.setitem(sys.modules, "cuda.bindings", module)
    function = audit_helpers("runtime_l2_bytes", torch=SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: 2)))["runtime_l2_bytes"]
    assert function(cap) == expected * 2**20
    assert calls == [(38, 12)]
    for invalid in (0, -1, float("nan"), float("inf"), 1e-10):
        with pytest.raises(ValueError, match="finite and positive"):
            function(invalid)
    driver.cuDeviceGetAttribute = lambda *a: (1, 0)
    with pytest.raises(RuntimeError, match="L2CacheSize"):
        function()
    driver.cuDeviceGet = lambda *a: (1, 0)
    with pytest.raises(RuntimeError, match="cuDeviceGet failed"):
        function()


def test_grouped_launch_scalars_preserve_non_grouped_abi():
    from types import SimpleNamespace
    launch = audit_helpers("launch_scalars")["launch_scalars"]
    for names, expected in [(("rows", "cta_limit"), (17, 376)),
                            (("rows", "cta_limit", "unit_limit"), (17, 376, 8)),
                            (("rows", "cta_limit", "unit_limit", "l2_bytes"),
                             (17, 376, 8, 96 * 2**20))]:
        program = SimpleNamespace(scalars=[SimpleNamespace(name=n) for n in names])
        assert launch(program, 17, 376, 8, 96 * 2**20) == expected


def test_grouped_frozen_audit_disables_l2_forwarding():
    source = SOURCE.with_name("bench_exlr8_grouped_prefill.py").read_text()
    tree = ast.parse(source)
    compile_arm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_compile")
    keyed = next(n for n in compile_arm.body if isinstance(n, ast.FunctionDef) and n.name == "keyed")
    branch = next(n for n in keyed.body if isinstance(n, ast.If))
    assert "wrapper.runtime_l2 = False" in ast.unparse(branch)
    assert "wrapper.core = reference_type" in ast.unparse(branch)
