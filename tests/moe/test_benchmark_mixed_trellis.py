from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")


def _load_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "benchmark_mixed_trellis.py"
    spec = importlib.util.spec_from_file_location("benchmark_mixed_trellis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Tier:
    num_experts: int
    trellis_bits: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


def test_benchmark_materializes_full_abi5_tier_storage(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "HIDDEN", 32)
    monkeypatch.setattr(benchmark, "INTERMEDIATE", 16)

    materialized = 2
    compiled = 3
    bits = 3
    w13 = torch.arange(2 * materialized * 2 * 1 * 8 * bits, dtype=torch.int32)
    w2 = torch.arange(materialized * 1 * 2 * 8 * bits, dtype=torch.int32)
    tier = _Tier(
        num_experts=materialized,
        trellis_bits=bits,
        w13=w13,
        w2=w2,
        w13_global_scale=torch.tensor([2.0, 3.0]),
        w2_global_scale=torch.tensor([4.0, 5.0]),
    )

    padded = benchmark._pad_exact_tier_storage(tier, compiled)

    assert padded.num_experts == compiled
    padded_w13 = padded.w13.view(2, compiled, 2, 1, 8 * bits)
    assert torch.equal(
        padded_w13[:, :materialized], w13.view(2, materialized, 2, 1, 8 * bits)
    )
    assert torch.count_nonzero(padded_w13[:, materialized:]) == 0
    padded_w2 = padded.w2.view(compiled, 1, 2, 8 * bits)
    assert torch.equal(padded_w2[:materialized], w2.view(materialized, 1, 2, 8 * bits))
    assert torch.count_nonzero(padded_w2[materialized:]) == 0
    assert torch.equal(padded.w13_global_scale, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(padded.w2_global_scale, torch.tensor([4.0, 5.0, 1.0]))


def test_benchmark_kimi_variant_parsing_and_geometry() -> None:
    benchmark = _load_benchmark_module()
    assert (
        benchmark.KIMI_HIDDEN,
        benchmark.KIMI_INTERMEDIATE,
        benchmark.KIMI_EXPERTS,
        benchmark.KIMI_TOPK,
        benchmark.KIMI_BITS,
    ) == (3584, 512, 896, 16, (2, 3))
    parsed = [
        benchmark._parse_kimi_variant(v) for v in benchmark.KIMI_DEFAULT_VARIANTS
    ]
    assert parsed == [
        ("old", None, (64, 256, 64, 256), 1),
        ("n128-2cta", None, (64, 128, 64, 128), 2),
    ]


@pytest.mark.parametrize("value", ["ws:32:ws", "coop8:8:cooperative", "coop16:16:cooperative"])
def test_prefill_arm(value):
    bench = _load_benchmark_module()
    name, block, mode = value.split(":")
    assert bench.parse_prefill_arm(value) == (name, int(block), mode)


@pytest.mark.parametrize("value", ["bad", "ws:8:ws", "a:17:ws", "a_control:32:ws", ":32:ws", "a:32:other"])
def test_prefill_arm_rejects_invalid(value):
    bench = _load_benchmark_module()
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench.parse_prefill_arm(value)


def test_prefill_policy_separates_route_block_from_ws():
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    qwen_coop = bench.prefill_compile_options("qwen4", 256, 32, "cooperative", props)
    qwen_ws = bench.prefill_compile_options("qwen4", 256, 32, "ws", props)
    assert qwen_coop["moe_block_size"] == qwen_ws["moe_block_size"] == 32
    assert qwen_coop["token_major_rotation"] and not qwen_ws["token_major_rotation"]
    assert qwen_ws["warp_specialized"] and not qwen_coop["warp_specialized"]
    assert qwen_ws["tier0_bits"] == 4 and qwen_ws["tier1_bits"] == 5
    assert qwen_ws["route_num_experts"] == 512 and qwen_ws["top_k"] == 10
    coop = bench.prefill_compile_options("kimik3", 512, 16, "cooperative", props)
    ws = bench.prefill_compile_options("kimik3", 512, 16, "ws", props)
    assert {k for k in coop if coop[k] != ws[k]} == {"warp_specialized"}
    assert ws["activation"] == "situ" and ws["tier1_num_experts"] == 1
    assert bench.prefill_compile_options("kimik3", 512, 8, "cooperative", props)["moe_block_size"] == 8
    for width in (128, 256, 384, 640):
        opts = bench.prefill_compile_options("qwen4", width, 32, "ws", props)
        assert width % opts["force_tile_config"][1] == 0
    with pytest.raises(ValueError):
        bench.prefill_compile_options("kimik3", 384, 16, "ws", props)


def test_prefill_route_fixture_is_family_specific(tmp_path):
    bench = _load_benchmark_module()
    path = tmp_path / "routes.json"
    wave = dict(layer=23, rows=4096, expert_rows=[[e, 4096] for e in range(10)])
    data = dict(schema=1, family="qwen4", experts=512, top_k=10, waves=[wave])
    path.write_text(bench.json.dumps(data))
    assert bench.prefill_route_waves(path, "qwen4", 23) == [wave]
    with pytest.raises(ValueError, match="metadata"):
        bench.prefill_route_waves(path, "kimik3", 23)
    with pytest.raises(ValueError, match="no waves"):
        bench.prefill_route_waves(path, "qwen4", 0)
    data["waves"][0]["expert_rows"][0][0] = 512
    path.write_text(bench.json.dumps(data))
    with pytest.raises(ValueError, match="histogram"):
        bench.prefill_route_waves(path, "qwen4", 23)


def test_prefill_kimi_routes_preserve_high_expert_ids(tmp_path):
    bench = _load_benchmark_module()
    wave = dict(layer=23, rows=4096, expert_rows=[[e, 4096] for e in range(880, 896)])
    path = tmp_path / "routes.json"
    path.write_text(bench.json.dumps(dict(schema=1, family="kimik3", experts=896, top_k=16, waves=[wave])))
    assert bench.prefill_route_waves(path, "kimik3", 23) == [wave]


def test_compact_prefill_fixture_requires_complete_ownership(tmp_path):
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=128,
                    projections=[dict(projection=p, tier=0, bits=4,
                                      tier_local=e, payload_word_offset=0)
                                 for p in ("gate", "up", "down")]) for e in range(512)]
    rank = dict(rank=0, compiled_max_intermediate=256, experts=entries,
                projection_counts=[[512, 512, 512], [0, 0, 0]], unit_map_fingerprint="test")
    path = tmp_path / "ownership.json"
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    assert len(bench.compact_prefill_fixture(path, 0, "qwen4", 256, 512)["entries"]) == 512
    with pytest.raises(ValueError, match="requested rank"):
        bench.compact_prefill_fixture(path, 1, "qwen4", 256, 512)
    with pytest.raises(ValueError, match="requires Qwen"):
        bench.compact_prefill_fixture(path, 0, "kimik3", 256, 512)
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 128, 512)
    entries[0]["width"] = 129
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    with pytest.raises(ValueError, match="live width"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 256, 512)
    entries[0]["width"] = 128
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 256, 256)


def test_compact_glm_fixture_uses_live_width_ownership(tmp_path):
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=384 if e < 85 else 256,
                    projections=[dict(projection=p, tier=e & 1, bits=4 + (e & 1),
                                      tier_local=e, payload_word_offset=0)
                                 for p in ("gate", "up", "down")]) for e in range(256)]
    rank = dict(rank=2, compiled_max_intermediate=384, experts=entries,
                projection_counts=[[82, 80, 64], [86, 88, 96]], unit_map_fingerprint="test")
    path = tmp_path / "glm-ownership.json"
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    fixture = bench.compact_prefill_fixture(path, 2, "glm", 384, 256)
    assert sorted({e["width"] for e in fixture["entries"]}) == [256, 384]
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 2, "glm", 384, 512)
    # Only the TP6 Spark profile ships compact live widths.
    from types import SimpleNamespace
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    with pytest.raises(ValueError, match="spark-tp6"):
        bench.coordinator_compile_options("spark-tp4-width512", 4096, {}, props, "glm", compact=True)
    options = bench.coordinator_compile_options("spark-tp6-width384", 4096, {"mode": "ws"}, props, "glm", compact=True)
    assert options["compact_weights"] and options["warp_specialized"]
    assert options["force_tile_config"] == (64, 128, 64, 128)
    assert options["moe_block_size"] == 32 and not options["token_major_rotation"]
    baseline = bench.coordinator_compile_options("spark-tp6-width384", 4096, {}, props, "glm", compact=True)
    assert baseline["moe_block_size"] == 64 and baseline["token_major_rotation"]
    assert not baseline["warp_specialized"] and baseline["compact_weights"]
    widened = bench.coordinator_compile_options("spark-tp6-width384", 4096,
                                                {"mode": "ws", "tile": (64, 128, 64, 256)},
                                                props, "glm", compact=True)
    assert widened["force_tile_config"] == (64, 128, 64, 256)
