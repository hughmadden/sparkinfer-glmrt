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
    t384 = bench.coordinator_compile_options("spark-tp6-width384", 4096,
                                             {"mode": "ws", "tile": (64, 192, 64, 256)},
                                             props, "glm", compact=True)
    assert t384["force_tile_config"] == (64, 192, 64, 256)


def test_compact_prefill_compile_options_match_served_exports(tmp_path):
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)

    # Qwen4 compact (TP4/TP3 w256, TP2 w384): cooperative must force (64,128,64,128) and no token rotation.
    for width in (256, 384):
        qwen_coop = bench.prefill_compile_options("qwen4", width, 32, "cooperative", props, compact=True)
        assert qwen_coop["compact_weights"] is True
        assert qwen_coop["warp_specialized"] is False
        assert qwen_coop["force_tile_config"] == (64, 128, 64, 128)
        assert qwen_coop["token_major_rotation"] is False
        assert qwen_coop["ws_dynamic_tiles"] is False
        assert qwen_coop["swiglu_limit"] is None
        assert (qwen_coop["tier0_num_experts"], qwen_coop["tier1_num_experts"]) == (512, 512)

        qwen_ws = bench.prefill_compile_options("qwen4", width, 64, "ws", props, compact=True)
        assert qwen_ws["compact_weights"] is True
        assert qwen_ws["warp_specialized"] is True
        assert qwen_ws["force_tile_config"] == (64, 128, 64, 256)
        assert qwen_ws["token_major_rotation"] is False
        assert qwen_ws["ws_dynamic_tiles"] is True
        assert qwen_ws["swiglu_limit"] is None

    qwen_t384 = bench.prefill_compile_options("qwen4", 384, 64, "ws+t384", props, compact=True)
    assert qwen_t384["force_tile_config"] == (64, 192, 64, 256) and qwen_t384["ws_dynamic_tiles"] is True
    with pytest.raises(ValueError, match="t384"):
        bench.prefill_compile_options("qwen4", 256, 64, "ws+t384", props, compact=True)

    # GLM 5.3 (glm, bits=(4,5), swiglu_limit=None), GLM Flash (glmf, bits=(3,4), swiglu_limit=10.0),
    # V4 Flash (dsv4f, bits=(4,5), swiglu_limit=10.0), and V4.1 (v41, bits=(3,4), swiglu_limit=10.0)
    for family, experts, topk, bits, block, clamp, widths in (
        ("glm", 256, 8, (4, 5), 64, None, (384, 768)),
        ("glmf", 288, 8, (3, 4), 64, 10.0, (384, 768)),
        ("dsv4f", 256, 6, (4, 5), 8, 10.0, (384, 768)),
        ("v41", 384, 6, (3, 4), 8, 10.0, (640,)),
    ):
        for width in widths:
            coop = bench.prefill_compile_options(family, width, block, "cooperative", props, compact=True)
            assert coop["compact_weights"] is True and not coop["warp_specialized"]
            assert coop["force_tile_config"] == (64, 128, 64, 128)
            assert not coop["token_major_rotation"] and not coop["ws_dynamic_tiles"]
            assert coop["swiglu_limit"] == clamp
            assert (coop["tier0_bits"], coop["tier1_bits"]) == bits
            assert (coop["tier0_num_experts"], coop["tier1_num_experts"], coop["route_num_experts"]) == (experts, experts, experts)
            assert coop["top_k"] == topk and coop["moe_block_size"] == block

            ws = bench.prefill_compile_options(family, width, 64, "ws", props, compact=True)
            assert ws["compact_weights"] is True and ws["warp_specialized"]
            assert ws["force_tile_config"] == (64, 128, 64, 256)
            assert ws["ws_dynamic_tiles"] is True and ws["swiglu_limit"] == clamp

            ws_t384 = bench.prefill_compile_options(family, width, 64, "ws+t384", props, compact=True)
            assert ws_t384["force_tile_config"] == (64, 192, 64, 256) and ws_t384["ws_dynamic_tiles"] is True

    # Validate compact_prefill_fixture for glmf (bits 3/4), dsv4f (bits 4/5), and v41 (bits 3/4, width 640)
    glmf_entries = [dict(global_expert=e, combined_local=e, width=384 if e < 96 else 256,
                         projections=[dict(projection=p, tier=e & 1, bits=3 + (e & 1),
                                           tier_local=e // 2, payload_word_offset=0)
                                      for p in ("gate", "up", "down")]) for e in range(288)]
    glmf_rank = dict(rank=0, compiled_max_intermediate=384, experts=glmf_entries,
                     projection_counts=[[144, 144, 144], [144, 144, 144]], unit_map_fingerprint="glmf-test")
    glmf_path = tmp_path / "glmf-ownership.json"
    glmf_path.write_text(bench.json.dumps(dict(synthetic=True, source="synthetic", ranks=[glmf_rank])))
    glmf_fix = bench.compact_prefill_fixture(glmf_path, 0, "glmf", 384)
    assert glmf_fix["synthetic"] is True and len(glmf_fix["entries"]) == 288

    # V4.1 compact TP4 w640: (3,4) bits, swiglu_limit 10.0, and a non-divisible
    # 192-wide FC1 tile arm (640 = 2 x 320-row halves -> 2 x 192-wide tiles).
    v41 = bench.prefill_compile_options("v41", 640, 64, "ws", props, compact=True)
    assert v41["force_tile_config"] == (64, 128, 64, 256)
    assert (v41["tier0_bits"], v41["tier1_bits"]) == (3, 4)
    assert (v41["tier0_num_experts"], v41["route_num_experts"], v41["top_k"]) == (384, 384, 6)
    assert v41["swiglu_limit"] == 10.0
    v41_t384 = bench.prefill_compile_options("v41", 640, 64, "ws+t384", props, compact=True)
    assert v41_t384["force_tile_config"] == (64, 192, 64, 256) and v41_t384["ws_dynamic_tiles"] is True
    with pytest.raises(ValueError, match="t384"):
        bench.prefill_compile_options("v41", 256, 64, "ws+t384", props, compact=True)

    v41_entries = [dict(global_expert=e, combined_local=e, width=640 if e < 192 else 512,
                        projections=[dict(projection=p, tier=e & 1, bits=3 + (e & 1),
                                          tier_local=e // 2, payload_word_offset=0)
                                     for p in ("gate", "up", "down")]) for e in range(384)]
    v41_rank = dict(rank=0, compiled_max_intermediate=640, experts=v41_entries,
                    projection_counts=[[192, 192, 192], [192, 192, 192]], unit_map_fingerprint="v41-test")
    v41_path = tmp_path / "v41-ownership.json"
    v41_path.write_text(bench.json.dumps(dict(synthetic=True, source="synthetic", ranks=[v41_rank])))
    v41_fix = bench.compact_prefill_fixture(v41_path, 0, "v41", 640)
    assert v41_fix["synthetic"] is True and len(v41_fix["entries"]) == 384
