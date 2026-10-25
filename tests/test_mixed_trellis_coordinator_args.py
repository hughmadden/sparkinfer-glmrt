"""CPU-only coordinator argument contracts (no CUDA/DSL imports)."""
import argparse
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_mixed_trellis.py"
TREE = ast.parse(SOURCE.read_text())
HOST = SOURCE.parents[1] / "b12x/moe/_shared/kernels/w4a16/host.py"


def benchmark():
    names = {"parse_coordinator_arm", "coordinator_compile_options", "coordinator_route_waves",
             "coordinator_preflight", "benchmark_parser", "parse_prefill_arm", "compact_prefill_fixture"}
    tables = {"GLM_CAPACITIES", "GLM_PROFILES", "GLM_ROUTES", "COORDINATOR_FAMILIES",
              "COORDINATOR_PROFILES", "SERVED_ROUTE_BLOCKS", "SERVED_TOKEN_ROTATION",
              "PREFILL_FAMILIES"}
    glm_profiles = {"rtx-tp1": 2048, "rtx-tp2": 1024, "spark-tp6-width384": 384,
                    "spark-tp6-width256": 256, "spark-tp4-width512": 512}
    glm_routes = SOURCE.parent / "fixtures/glm53_prefill_routes_realistic.json"
    scope = dict(argparse=argparse, json=json, Path=Path, re=re, SimpleNamespace=SimpleNamespace,
                 __file__=str(SOURCE),
                 GLM_PROFILES=glm_profiles, GLM_ROUTES=glm_routes)
    host_names = {"route_pack_capacity", "route_pack_numel_capacity", "route_pack_token_capacity",
                  "max_packed_route_slots"}
    # Exec the source's own family tables, so the contracts below run against
    # the shipped table rather than a copy that can drift from it.
    table_nodes = [n for n in TREE.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in tables for t in n.targets)]
    exec(compile(ast.Module(body=table_nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    for tree, selected in ((ast.parse(HOST.read_text()), host_names), (TREE, names)):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**scope)


def arguments(b, *extra):
    return b.benchmark_parser().parse_args([
        "--prefill-family", "glm", "--role", "coordinator", "--output", "result.json", *extra])


@pytest.mark.parametrize("profile,width", [("rtx-tp1", 2048), ("rtx-tp2", 1024)])
@pytest.mark.parametrize("capacity", [256, 1024, 2048, 2752, 4096])
def test_cooperative_matches_export_contract(profile, width, capacity):
    b = benchmark()
    for sms in (170, 188):
        props = SimpleNamespace(multi_processor_count=sms, shared_memory_per_block_optin=101376)
        opts = b.coordinator_compile_options(profile, capacity, {}, props)
        assert (opts["hidden_size"], opts["intermediate_size"], opts["top_k"], opts["route_num_experts"]) == (6144, width, 8, 256)
        assert opts["tier0_num_experts"] == opts["tier1_num_experts"] == 256
        assert (opts["tier0_bits"], opts["tier1_bits"]) == (4, 5)
        assert opts["sms"] == sms
        assert opts["full_rotation_output_dtype"] == "fp32"
        assert opts["moe_block_size"] == (8 if capacity == 256 else 64)
        assert opts["token_major_rotation"] == (capacity > 256)
        assert not opts["warp_specialized"] and not opts["fused_input_rotation"]
        assert opts["max_m_blocks"] >= (capacity * 8 + 256 * (opts["moe_block_size"] - 1) + opts["moe_block_size"] - 1) // opts["moe_block_size"]


def test_variant_plumbing_and_preflight():
    b = benchmark()
    args = arguments(b, "--coordinator-arm", "wire:mode=ws,wire=1,stages=3,dynamic=1,ctas=1,route_block=32,tile=64/128/64/128")
    capacities, specs, routes, waves, compact = b.coordinator_preflight(args)
    assert compact is None
    assert capacities == [256, 1024, 2048, 2752, 4096] and routes == ["random", "realistic"]
    assert len(waves) == 72 and specs[0] == ("coop", {})
    opts = b.coordinator_compile_options(args.profile, 2752, specs[1][1], SimpleNamespace(multi_processor_count=170, shared_memory_per_block_optin=101376))
    assert opts["warp_specialized"] and opts["input_format"] == "e4m3_k32"
    assert opts["force_tile_config"] == (64, 128, 64, 128)
    assert opts["moe_block_size"] == 32 and opts["force_blocks_per_sm"] == 1
    assert opts["ws_input_stages"] == 3 and opts["ws_dynamic_tiles"]
    assert not opts["token_major_rotation"] and not opts["fused_input_rotation"]


@pytest.mark.parametrize("text", ["bad", "coop:mode=ws", "a:wire=2", "a:stages=1", "a:route_block=17",
                                 "a:mode=other", "a:tile=64/256", "a:unknown=1", "a:wire=1,wire=0"])
def test_bad_arm(text):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark().parse_coordinator_arm(text)


@pytest.mark.parametrize("text", ["a:mode=ws,rotation=token", "a:mode=ws,route_block=8", "a:wire=1",
                                 "a:stages=3", "a:dynamic=1", "a:rotation=fused,route_block=8",
                                 "a:tile=64/128/64/256", "a:tile=64/192/64/256"])
def test_invalid_combinations_fail_before_hardware(text):
    b = benchmark()
    with pytest.raises(ValueError):
        b.coordinator_preflight(arguments(b, "--coordinator-arm", text))


@pytest.mark.parametrize("extra", [("--capacities", "256,80"), ("--capacities", "256,256"),
                                   ("--route-distributions", "random,bad"), ("--rounds", "0"),
                                   ("--k5-experts", "257"), ("--intermediate", "1024")])
def test_invalid_arguments(extra):
    b = benchmark()
    with pytest.raises(ValueError):
        b.coordinator_preflight(arguments(b, *extra))


@pytest.mark.parametrize("profile,width,tile", [("spark-tp6-width384", 384, (64, 192, 64, 256)),
                                               ("spark-tp6-width256", 256, (64, 256, 64, 256)),
                                               ("spark-tp4-width512", 512, (64, 256, 64, 256))])
def test_glm_spark_served_exports(profile, width, tile):
    b = benchmark()
    args = arguments(b, "--role", "spark", "--profile", profile, "--capacities", "1024,2752,4096")
    capacities, specs, _, _, compact = b.coordinator_preflight(args)
    assert compact is None
    assert specs[0] == ("coop", {}) and specs[1][0] == "served"
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    for capacity in capacities:
        opts = b.coordinator_compile_options(profile, capacity, specs[1][1], props)
        assert opts["intermediate_size"] == width and opts["force_tile_config"] == tile
        assert opts["full_rotation_output_dtype"] == "bf16" and opts["sms"] == 48
        assert opts["moe_block_size"] == 64 and opts["warp_specialized"]
        assert opts["input_format"] == "e4m3_k32" and opts["ws_input_stages"] == 3
        assert opts["ws_dynamic_tiles"] and opts["force_blocks_per_sm"] == 1


def test_spark_defaults_unchanged():
    args = benchmark().benchmark_parser().parse_args(["--prefill-family", "qwen4", "--output", "result.json"])
    assert args.role == "spark" and args.route_layer == 23
    assert args.prefill_arm == [] and args.rounds == 5 and args.replays == 100


def test_routes_filter_and_reject_malformed(tmp_path):
    b = benchmark()
    assert len(b.coordinator_route_waves(b.GLM_ROUTES, 7)) == 9
    path = tmp_path / "routes.json"
    data = dict(schema=1, model="zai-org/GLM-5.3", waves=[dict(layer=7, rows=256, expert_rows=[[e, 256] for e in range(8)])])
    path.write_text(json.dumps(data))
    assert len(b.coordinator_route_waves(path)) == 1
    data["waves"][0]["expert_rows"][0][0] = 256
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="histogram"):
        b.coordinator_route_waves(path)


@pytest.mark.parametrize("profile,width", [("rtx-tp1", 2048), ("rtx-tp2", 1024)])
@pytest.mark.parametrize("capacity", [256, 1024, 2048, 2752, 4096])
def test_glmf_cooperative_matches_export_contract(profile, width, capacity):
    b = benchmark()
    for sms in (170, 188):
        props = SimpleNamespace(multi_processor_count=sms, shared_memory_per_block_optin=101376)
        opts = b.coordinator_compile_options(profile, capacity, {}, props, "glmf")
        assert (opts["hidden_size"], opts["intermediate_size"], opts["top_k"], opts["route_num_experts"]) == (4096, width, 8, 288)
        assert opts["tier0_num_experts"] == opts["tier1_num_experts"] == 288
        assert (opts["tier0_bits"], opts["tier1_bits"]) == (3, 4)
        assert opts["swiglu_limit"] == 10.0
        assert opts["sms"] == sms
        assert opts["full_rotation_output_dtype"] == "fp32"
        assert opts["moe_block_size"] == (8 if capacity == 256 else 64)
        assert opts["token_major_rotation"] == (capacity > 256)
        assert not opts["warp_specialized"] and not opts["fused_input_rotation"]
        assert opts["max_m_blocks"] >= (capacity * 8 + 288 * (opts["moe_block_size"] - 1) + opts["moe_block_size"] - 1) // opts["moe_block_size"]


def test_glmf_preflight_and_foreign_fixture_refusal():
    b = benchmark()
    args = b.benchmark_parser().parse_args([
        "--prefill-family", "glmf", "--role", "coordinator", "--route-distributions", "random",
        "--k5-experts", "72", "--output", "result.json",
        "--coordinator-arm", "wire:mode=ws,wire=1,stages=3,dynamic=1,ctas=1,route_block=32,tile=64/128/64/128",
    ])
    capacities, specs, routes, waves, compact = b.coordinator_preflight(args)
    assert compact is None
    assert capacities == [256, 1024, 2048, 2752, 4096] and routes == ["random"] and waves == []
    opts = b.coordinator_compile_options(args.profile, 2752, specs[1][1],
                                         SimpleNamespace(multi_processor_count=188, shared_memory_per_block_optin=101376), "glmf")
    assert opts["hidden_size"] == 4096 and opts["route_num_experts"] == 288
    assert (opts["tier0_bits"], opts["tier1_bits"]) == (3, 4) and opts["swiglu_limit"] == 10.0
    assert opts["warp_specialized"] and opts["input_format"] == "e4m3_k32"
    assert opts["force_tile_config"] == (64, 128, 64, 128) and opts["moe_block_size"] == 32
    with pytest.raises(ValueError, match="no realistic route fixture"):
        b.coordinator_preflight(b.benchmark_parser().parse_args([
            "--prefill-family", "glmf", "--role", "coordinator", "--output", "result.json"]))
    with pytest.raises(ValueError, match="zai-org/GLM-5.3-Flash"):
        b.coordinator_route_waves(b.GLM_ROUTES, family="glmf")
    with pytest.raises(ValueError, match="zai-org/GLM-5.3-Flash"):
        b.coordinator_preflight(b.benchmark_parser().parse_args([
            "--prefill-family", "glmf", "--role", "coordinator", "--route-distributions", "random",
            "--routes", str(b.GLM_ROUTES), "--output", "result.json"]))
    with pytest.raises(ValueError):
        b.coordinator_preflight(b.benchmark_parser().parse_args([
            "--prefill-family", "glmf", "--role", "spark", "--profile", "spark-tp4-width512",
            "--route-distributions", "random", "--output", "result.json"]))
    with pytest.raises(ValueError):
        b.coordinator_preflight(b.benchmark_parser().parse_args([
            "--prefill-family", "glmf", "--role", "coordinator", "--route-distributions", "random",
            "--k5-experts", "289", "--output", "result.json"]))


# Shipped cooperative coordinator policy per family: (capacity, block, token).
SERVED = {
    ("qwen4", 256): (32, False), ("qwen4", 1024): (32, True), ("qwen4", 4096): (32, True),
    ("v41", 256): (8, False), ("v41", 1024): (8, False), ("v41", 4096): (8, False),
    ("dsv4p", 256): (8, False), ("dsv4p", 1024): (64, True), ("dsv4p", 2752): (64, True),
    ("kimik3", 256): (8, False), ("kimik3", 1024): (64, True), ("kimik3", 4096): (8, False),
    ("kimik3", 2048): (8, False), ("kimik3", 2752): (8, False),
}


FAMILY_PROFILE = {"qwen4": "rtx-tp1", "v41": "rtx-tp2", "dsv4p": "rtx-tp2", "kimik3": "rtx-tp2"}


@pytest.mark.parametrize("family,capacity", sorted(SERVED))
def test_family_served_defaults_match_exporter(family, capacity):
    b = benchmark()
    block, token = SERVED[(family, capacity)]
    options = b.coordinator_compile_options(FAMILY_PROFILE[family], capacity, {}, SimpleNamespace(
        multi_processor_count=188, shared_memory_per_block_optin=101376), family)
    assert options["moe_block_size"] == block
    assert options["token_major_rotation"] == token
    assert not options["warp_specialized"]
    assert options["full_rotation_output_dtype"] == "fp32"


@pytest.mark.parametrize("family,profile,capacity,width", [
    ("qwen4", "rtx-tp1", 4096, 640), ("qwen4", "rtx-tp2-rank0", 2752, 384),
    ("qwen4", "rtx-tp2-rank1", 1024, 256), ("v41", "rtx-tp1", 4096, 2304),
    ("v41", "rtx-tp2", 1024, 1152), ("dsv4p", "rtx-tp1", 4096, 3072),
    ("dsv4p", "rtx-tp2", 2752, 1536), ("kimik3", "rtx-tp1", 1024, 3072),
    ("kimik3", "rtx-tp2", 4096, 1536)])
def test_family_compact_ws_arm_plumbing(family, profile, capacity, width):
    b = benchmark()
    args = b.benchmark_parser().parse_args([
        "--prefill-family", family, "--role", "coordinator", "--profile", profile,
        "--capacities", str(capacity), "--route-distributions", "random",
        "--output", "result.json",
        "--coordinator-arm", "s14:mode=ws,route_block=64,dynamic=1,tile=64/128/64/256"])
    capacities, specs, routes, waves, compact = b.coordinator_preflight(args)
    assert capacities == [capacity] and specs[1][0] == "s14" and compact is None
    options = b.coordinator_compile_options(profile, capacity, specs[1][1], SimpleNamespace(
        multi_processor_count=188, shared_memory_per_block_optin=101376), family)
    assert options["intermediate_size"] == width and options["hidden_size"] == b.COORDINATOR_FAMILIES[family]["hidden"]
    assert options["warp_specialized"] and options["force_tile_config"] == (64, 128, 64, 256)
    assert options["ws_dynamic_tiles"] and options["input_format"] == "bf16"
    assert options["activation"] == b.COORDINATOR_FAMILIES[family]["activation"]
    assert not options["token_major_rotation"]


FAMILY_CAPACITIES = {"qwen4": [256, 1024, 2048, 2752, 4096], "v41": [256, 1024, 2048, 2752, 4096],
                     "dsv4p": [256, 1024, 2048, 2752, 4096], "kimik3": [256, 1024, 2048, 2752, 4096]}


@pytest.mark.parametrize("family", sorted(FAMILY_CAPACITIES))
def test_family_capacity_contract(family):
    b = benchmark()
    args = b.benchmark_parser().parse_args([
        "--prefill-family", family, "--role", "coordinator", "--profile", FAMILY_PROFILE[family],
        "--route-distributions", "random", "--output", "result.json"])
    capacities, _, _, _, _ = b.coordinator_preflight(args)
    assert capacities == FAMILY_CAPACITIES[family]
    with pytest.raises(ValueError, match="no measured export"):
        b.coordinator_preflight(b.benchmark_parser().parse_args([
            "--prefill-family", family, "--role", "coordinator", "--profile", FAMILY_PROFILE[family],
            "--capacities", "2049",
            "--route-distributions", "random", "--output", "result.json"]))


@pytest.mark.parametrize("family", sorted(FAMILY_CAPACITIES))
@pytest.mark.parametrize("capacity", [2048, 2752])
def test_adaptive_capacities_and_exhaustive_preflight(family, capacity):
    b = benchmark()
    args = b.benchmark_parser().parse_args([
        "--prefill-family", family, "--role", "coordinator", "--profile", FAMILY_PROFILE[family],
        "--capacities", str(capacity), "--prefill-exhaustive",
        "--route-distributions", "random", "--output", "result.json"])
    capacities, _, _, _, _ = b.coordinator_preflight(args)
    assert capacities == [capacity]


def test_control_name_is_reserved():
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark().parse_coordinator_arm("coop_control:mode=ws")


def test_n192_ws_requires_whole_projection_tiles():
    b = benchmark()
    props = SimpleNamespace(multi_processor_count=188, shared_memory_per_block_optin=101376)
    options = b.coordinator_compile_options("rtx-tp2", 2048,
        dict(mode="ws", tile=(64, 192, 64, 256)), props, "v41")
    assert options["force_tile_config"] == (64, 192, 64, 256)
    with pytest.raises(ValueError):
        b.coordinator_compile_options("rtx-tp2-rank1", 2048,
            dict(mode="ws", tile=(64, 192, 64, 256)), props, "qwen4")


def test_prefill_capacity_controls_allocation():
    b = benchmark()
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "prefill_compile_options"]
    tables = [n for n in TREE.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in ("PREFILL_TIER_COUNTS", "PREFILL_SWIGLU_LIMITS")
                      for t in n.targets)]
    scope = dict(vars(b))
    exec(compile(ast.Module(body=tables + nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    for capacity in (1024, 2048, 2752, 4096):
        options = scope["prefill_compile_options"]("glm", 384, 64, "ws+t384", props,
                                                  compact=True, capacity=capacity)
        assert options["size_m"] == capacity
        slots = b.route_pack_capacity(capacity * 8, 64, 256, topk=8)[1]
        assert options["max_m_blocks"] == (slots + 63) // 64


@pytest.mark.parametrize("family,profile,width", [
    ("glm", "spark-compact-width384", 384), ("glm", "spark-compact-width768", 768),
    ("glmf", "spark-compact-width384", 384), ("glmf", "spark-compact-width768", 768),
    ("qwen4", "spark-compact-width384", 384), ("qwen4", "spark-compact-width256", 256),
    ("v41", "spark-compact-width640", 640),
    ("dsv4p", "spark-tp4-width768", 768), ("kimik3", "spark-tp6-width512", 512),
])
def test_spark_tuning_profile_options(family, profile, width):
    b = benchmark()
    compact = "compact" in profile
    options = b.coordinator_compile_options(profile, 2048,
        dict(mode="ws", route_block=64, dynamic=1, tile=(64, 128, 64, 256)),
        SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376),
        family, compact=compact)
    assert options["size_m"] == 2048 and options["intermediate_size"] == width
    assert options["compact_weights"] == compact
    assert options["full_rotation_output_dtype"] == "bf16"
    if compact and width >= 384:
        options = b.coordinator_compile_options(profile, 2752,
            dict(mode="ws", tile=(64, 192, 64, 256)),
            SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376),
            family, compact=True)
        assert options["force_tile_config"] == (64, 192, 64, 256)
