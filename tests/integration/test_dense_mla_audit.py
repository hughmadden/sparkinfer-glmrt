"""CPU controls for the frozen Kimi MLA duel harness."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dense_mla_audit", ROOT / "benchmarks/dense_mla_audit.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def base():
    return ["--checkpoint-config", "config.json", "--reference-core", "frozen.py@" + "a" * 40,
            "--json-output", "result.json"]


def test_host_revision_does_not_require_container_git(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Git metadata is not mounted")
    monkeypatch.setattr(bench.subprocess, "check_output", unavailable)
    assert bench.source_commit("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="full Git revision"):
        bench.source_commit("short")


def test_defaults_are_real_fixed_capacity():
    args = bench.arguments(base())
    assert args.capacity == 4096
    assert args.rows == [2752, 2688, 2673, 1, 63, 65]
    assert args.prior_contexts == [0, 8192, 32768, 65536]


@pytest.mark.parametrize("flags", [["--rows", "0"], ["--rows", "4097"], ["--rows", ""],
    ["--prior-contexts", "-1"], ["--max-context", "100"], ["--samples", "0"],
    ["--iterations", "0"], ["--warmup", "0"], ["--dtype", "bf16"]])
def test_invalid_args(flags):
    with pytest.raises(SystemExit):
        bench.arguments(base() + flags)


def test_frozen_bytes_checked_before_execution(tmp_path):
    path = tmp_path / "frozen.py"
    source = b"raise RuntimeError('must not execute')\n"
    path.write_bytes(source)
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    assert bench.verified_core(f"{path}@{blob}")[1] == source
    path.write_bytes(source + b"#changed\n")
    with pytest.raises(ValueError, match="mismatch"):
        bench.verified_core(f"{path}@{blob}")
    with pytest.raises(ValueError, match="requires"):
        bench.verified_core(str(path))


def test_byte_comparison_includes_lse_signed_zero():
    assert not bench.byte_equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert bench.byte_equal(torch.tensor([1.], dtype=torch.bfloat16), torch.tensor([1.], dtype=torch.bfloat16))


def test_checkpoint_geometry(tmp_path):
    path = tmp_path / "config.json"
    geometry = dict(hidden_size=7168, num_attention_heads=96, q_lora_rank=1536,
                    kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)
    path.write_text(json.dumps(dict(text_config=geometry)))
    assert bench.checkpoint_geometry(path)["page_rows"] == 64
    geometry["num_attention_heads"] = 48
    path.write_text(json.dumps(dict(text_config=geometry)))
    with pytest.raises(ValueError, match="geometry"):
        bench.checkpoint_geometry(path)


def test_engine_compiler_fixed_capacity_and_restored_symbols(monkeypatch):
    from b12x.integration.cuteafd import _common, kimi_mla, kimi_mla_prefill_fp8
    saved = kimi_mla_prefill_fp8.DenseMlaFp8PrefillKernel
    calls = []
    def compile_program(wrapper, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(geometry=kwargs["geometry"])
    monkeypatch.setattr(_common, "compile_program", compile_program)
    args = SimpleNamespace(capacity=4096, max_context=1048576)
    core = bench.verified_core(f"{ROOT / bench.CORE}@722a492f1d6143ed722fdefb20e981b453341169")
    reference, _ = bench.engine_program(args, core)
    candidate, _ = bench.engine_program(args)
    assert reference.geometry["native_fp8_prefill"] == (64, 1, 2, 0, 1)
    assert reference.geometry == candidate.geometry
    assert calls[0]["name"] != calls[1]["name"]
    assert kimi_mla_prefill_fp8.DenseMlaFp8PrefillKernel is saved
