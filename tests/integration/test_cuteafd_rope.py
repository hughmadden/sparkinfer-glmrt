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
