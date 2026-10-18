"""Byte-exact rotary tables against the pinned official checkpoint source."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x


def _official():
    root = os.environ.get("CUTEAFD_SOURCE_ROOT")
    snapshot = os.environ.get("STEP5_REFERENCE_SNAPSHOT")
    if not root or not snapshot:
        pytest.skip("set CUTEAFD_SOURCE_ROOT and STEP5_REFERENCE_SNAPSHOT for official RoPE gate")
    path = Path(root) / "python/reference/families/step5/golden.py"
    spec = importlib.util.spec_from_file_location("step5_rope_golden", path)
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    release = golden.load_release(Path(snapshot))[0]
    config = json.loads((Path(snapshot) / "config.json").read_text())["text_config"]
    return release, release.Step5Config(**config)


@pytest.mark.parametrize("kind,span", [("sliding_attention", 192), ("full_attention", 64),
                                       ("indexer", 32)])
def test_rope_full_extent_official_bytes(kind, span):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.rope import compile_rope_tables_aot
    release, config = _official()
    layer_kind = "full_attention" if kind == "indexer" else kind
    layer = next(i for i in range(config.num_hidden_layers) if config.layer_type(i) == layer_kind)
    scaling = config.rope_scaling_for(layer)
    theta = config.rope_theta[layer]
    extent = config.max_position_embeddings
    assert extent == 1048576
    program = compile_rope_tables_aot(rotary_span=span, llama3=scaling is not None)
    values = scaling or {}
    scalars = [extent, theta, values.get("factor", 1.0),
               float(values.get("original_max_position_embeddings", extent)),
               values.get("low_freq_factor", 1.0), values.get("high_freq_factor", 32.0)]
    expected = release.build_rope_cache(rotary_span=span, theta=theta, max_position=extent,
                                        device=torch.device("cuda"), dtype=torch.bfloat16,
                                        scaling=scaling)
    outputs = [torch.empty_like(x) for x in expected]
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with kernel_resolution_guard():
        start.record()
        program.launch(*outputs, scalars=scalars)
        stop.record()
    stop.synchronize()
    print(f"{kind}: table build {start.elapsed_time(stop):.3f} ms")
    for name, actual, oracle in zip(("cos", "sin"), outputs, expected):
        different = actual.view(torch.int16) != oracle.view(torch.int16)
        mismatches = int(different.sum())
        if mismatches:
            first = different.nonzero()[0].tolist()
            pytest.fail(f"{kind}/{name}: {mismatches} byte mismatches, first {first}, "
                        f"actual={actual[tuple(first)].item()} expected={oracle[tuple(first)].item()}")
    with kernel_resolution_guard():
        for live in (1, 7, 257):
            for output in outputs:
                output.fill_(float("nan"))
            program.launch(*outputs, scalars=[live, *scalars[1:]])
            for actual, oracle in zip(outputs, expected):
                assert torch.equal(actual[:live].view(torch.int16), oracle[:live].view(torch.int16))
                assert torch.isnan(actual[live:]).all()
    with kernel_resolution_guard(), torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        program.launch(*outputs, scalars=scalars)
    for _ in range(3):
        graph.replay()
    for actual, oracle in zip(outputs, expected):
        assert torch.equal(actual.view(torch.int16), oracle.view(torch.int16))


@pytest.mark.parametrize("llama3", [False, True])
def test_rope_runtime_recipe_and_export(llama3, tmp_path):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.rope import compile_rope_tables_aot
    from ._cuteafd import check_export
    release, _ = _official()
    with exportable_compilation():
        program = compile_rope_tables_aot(rotary_span=64, llama3=llama3)
    check_export(program, tmp_path, "rope_tables")
    assert [(s.name, s.ctype) for s in program.scalars] == [
        ("positions", "int32"), ("theta", "float32"), ("factor", "float32"),
        ("original_max", "float32"), ("low_factor", "float32"), ("high_factor", "float32")]
    for theta, factor in ((10000.0, 1.0), (10000000.0, 8.0), (500000.0, 2.5)):
        scaling = dict(rope_type="llama3", factor=factor,
                       original_max_position_embeddings=8192,
                       low_freq_factor=1.0, high_freq_factor=4.0) if llama3 else None
        expected = release.build_rope_cache(rotary_span=64, theta=theta, max_position=8193,
                                            device=torch.device("cuda"), dtype=torch.bfloat16,
                                            scaling=scaling)
        outputs = [torch.empty_like(x) for x in expected]
        with kernel_resolution_guard():
            program.launch(*outputs, scalars=[8193, theta, factor, 8192.0, 1.0, 4.0])
        for actual, oracle in zip(outputs, expected):
            assert torch.equal(actual.view(torch.int16), oracle.view(torch.int16))
