"""The GLM 5.3 Flash decode-schedule benchmark's CPU-checkable parts: its
variant list and how it certifies the timed replays' outputs."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")

from benchmarks import benchmark_glmf_decode_schedule as bench


def test_variant_names_must_be_unique() -> None:
    variants = bench.parse_variants(["default", "gb10", "gb10-n128x2:gb10:64,128,64,128:2"])
    assert [name for name, *_ in variants] == ["default", "gb10", "gb10-n128x2"]
    # A repeated name would key one graph and its timings for two schedules.
    with pytest.raises(ValueError, match="gb10 repeated"):
        bench.parse_variants(["default", "gb10", "gb10:l2=2:64,256,64,256:"])


def test_a_repeated_name_stops_the_run_before_any_gpu_work(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["bench", "--variant", "default", "--variant", "default"])
    with pytest.raises(SystemExit) as stop:
        bench.main()
    assert stop.value.code == 2 and "default repeated" in capsys.readouterr().err


def test_every_layer_must_run_in_the_timed_graph(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["bench", "--layers", "3", "--calls-per-graph", "2"])
    with pytest.raises(SystemExit) as stop:
        bench.main()
    assert stop.value.code == 2 and "--calls-per-graph" in capsys.readouterr().err


def test_replay_results_certify_every_layer() -> None:
    first = torch.arange(16, dtype=torch.float32).to(torch.bfloat16).reshape(2, 8)
    digests = [bench.output_digest(first), bench.output_digest(first.flip(0))]
    assert digests[0] != digests[1] and all(len(d) == 16 for d in digests)
    assert bench.output_digest(first.clone()) == digests[0]
    assert bench.verdict(digests, digests, digests) == "same"
    # A layer whose replay result differs from its eager run, or from the
    # reference variant's, is not certified.
    assert bench.verdict([digests[0], digests[0]], digests, digests) == "REPLAY-DIFFERS"
    assert bench.verdict(digests, digests, [digests[0], digests[0]]) == "DIFFERENT"


def test_flash_default_stays_unchanged():
    g = bench.Geometry()
    assert (g.hidden, g.intermediate, g.experts, g.topk, g.bits) == (4096, 512, 288, 8, (3, 4))
    assert g.tier0_ids == tuple(range(284, 288))
    assert g.tier1_ids == tuple(range(284))
    assert bench.parse_variant("default") == ("default", None, (64, 256, 64, 256), None)


def test_compact_widths_use_balanced_rotated_ownership():
    g = dataclasses.replace(bench.Geometry(), full_intermediate=2048,
                            intermediate=384, experts=256, world=6, compact=True)
    widths = [dataclasses.replace(g, rank=rank).widths(3) for rank in range(6)]
    assert [w[0] for w in widths] == [384, 256, 256, 384, 384, 384]
    assert all(sum(w[e] for w in widths) == 2048 for e in range(g.experts))
    assert set(g.widths(3)) == {256, 384}


@pytest.mark.parametrize("geometry,world,compact,expected", [
    ("glm", 6, True, (6144, 384, 256, 8, "silu", None)),
    ("glmf", 4, False, (4096, 512, 288, 8, "silu", 10.0)),
    ("kimik3", 6, False, (3584, 512, 896, 16, "situ", None)),
    ("qwen4", 4, True, (2560, 256, 512, 10, "silu", None)),
])
def test_shipping_geometry_comes_from_engine_policy(geometry, world, compact, expected):
    root = Path(__file__).resolve().parents[4]
    policy = root / "python/tools/aot/package_exl3_aot.py"
    if not policy.is_file():
        pytest.skip("engine packaging policy is not available in a standalone fork")
    args = argparse.Namespace(package_policy=policy, geometry=geometry, role="spark",
                              world=world, rank=0, compact_weights=compact, bits=None, profile=None)
    g = bench.shipping_geometry(args)
    assert (g.hidden, g.intermediate, g.experts, g.topk, g.activation, g.swiglu_limit) == expected


def test_policy_descriptors_must_be_even_and_written_before_use():
    good = """Function : test
/*0000*/ UMOV UR4, URZ;
/*0010*/ USHF.L.U32 UR5, URZ, 0x14, UR6;
/*0020*/ LDGSTS.E desc[UR4], [R2], [R4.64];
"""
    assert bench.check_policy_descriptors(good) == 1
    assert bench.check_policy_descriptors("/*0000*/ UTMALDG desc[URZ], [R2];") == 0
    for bad in (good.replace("desc[UR4]", "desc[UR1]"),
                good.replace("/*0010*/ USHF.L.U32 UR5, URZ, 0x14, UR6;", ""),
                good.replace("/*0000*/ UMOV", "/*0000*/ @UP0 UMOV"),
                good.replace("/*0020*/", "Function : next\n/*0020*/")):
        with pytest.raises(RuntimeError, match="descriptor"):
            bench.check_policy_descriptors(bad)


def test_inspection_extracts_embedded_cubins_before_cuobjdump(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class Program:
        def export_to_c(self, directory, stem, symbol):
            (Path(directory) / f"{stem}.o").write_bytes(b"host-object")

    inspector = SimpleNamespace(cuda_images=lambda data: [(12, b"device-elf")])
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(bench.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(bench.importlib.util, "module_from_spec", lambda spec: inspector)
    calls = []

    def dump(command, **kwargs):
        calls.append(command)
        assert Path(command[-1]).read_bytes() == b"device-elf"
        if command[1] == "--dump-sass":
            return "/*0000*/ UMOV UR4, URZ;\n/*0010*/ UMOV UR5, URZ;\n/*0020*/ LDGSTS.E desc[UR4], [R2], [R4.64];"
        return "REG:128 STACK:0 SHARED:1024 LOCAL:0"

    monkeypatch.setattr(bench.subprocess, "check_output", dump)
    info = bench.inspect_kernel(SimpleNamespace(compiled=Program()), tmp_path)
    assert len(calls) == 2 and info["policy_sites"] == 1
    assert info["resources"] == [dict(registers=128, stack=0, shared=1024, local=0)]
    assert len(info["cubin_sha256"]) == 1
