"""The GLM 5.3 Flash decode-schedule benchmark's CPU-checkable parts: its
variant list and how it certifies the timed replays' outputs."""

from __future__ import annotations

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
