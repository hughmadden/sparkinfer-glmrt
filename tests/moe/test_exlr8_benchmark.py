"""CPU regressions for the EXLR8 Spark-step microbench helpers."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_exlr8_a16 as bench
from .exlr8_oracle import make_wire


def test_bench_wire_matches_oracle_wire():
    generator = torch.Generator().manual_seed(408)
    x = torch.randn((5, 6144), generator=generator) * 3
    x[1, :32] = 0
    x[2, 64:96] *= 1e4
    wire = bench.make_wire(x)
    expected, dequant = make_wire(x)
    assert torch.equal(wire, expected)
    assert wire.shape == (5, 6144 + 192)
    # The E4M3 payload times its UE8M0 scale reproduces the oracle dequant.
    values = wire[:, :6144].view(torch.float8_e4m3fn).double().reshape(5, 192, 32)
    scales = torch.exp2(wire[:, 6144:].double() - 127).unsqueeze(-1)
    assert torch.equal((values * scales).reshape(5, 6144), dequant)


@pytest.mark.parametrize("value", ["master:6d67f5eb", "gemini:e8717e05"])
def test_grouped_arm_argument(value):
    assert bench.parse_grouped_arm(value) == tuple(value.split(":"))


@pytest.mark.parametrize("value", ["missing", ":HEAD", "a:", "exl3:HEAD", "a:--help", "a:b:c"])
def test_grouped_arm_bad_argument(value):
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench.parse_grouped_arm(value)


def test_grouped_arm_uses_own_ingress_and_source_keys(monkeypatch, tmp_path):
    from b12x.integration.cuteafd import _common
    directory = tmp_path / ("a" * 40)
    directory.mkdir()
    (directory / "core.py").write_text("class Exlr8GroupedKernel:\n"
                    "    def __init__(self, *args): self.geometry = args\n")
    glue = directory / "glue.py"
    glue.write_text("from ._common import compile_program\n"
                   "def compile_exlr8_group_aot(**kwargs):\n"
                   "    return compile_program('frozen ingress', key=(64,), **kwargs)\n")
    (directory / "wrapper.py").write_text("class _Exlr8ServingPointers:\n"
                                          "    frozen = True\n")
    def manifest():
        result = dict(revision=directory.name, sources={label:
            bench.hashlib.sha256((directory / (label + '.py')).read_bytes()).hexdigest()
            for label in ('core', 'glue', 'wrapper')})
        (directory / "manifest.json").write_text(bench.json.dumps(result))
        return result
    manifest()
    geometry = dict(hidden=6144, units=4, top_k=8, max_groups=768, group_rows=64)
    candidate = SimpleNamespace(geometry=geometry, operands=(SimpleNamespace(name="x"),),
                                scalars=(SimpleNamespace(name="rows"),), scratch={})
    calls = []
    def compile_program(wrapper, **kwargs):
        calls.append((wrapper, kwargs))
        return wrapper
    monkeypatch.setattr(_common, "compile_program", compile_program)
    program, ingress, provenance = bench.compile_grouped_arm(directory, candidate)
    assert program.core.geometry == (6144, 4, 8, 768, 64)
    assert program.frozen
    assert ingress == "frozen ingress"
    assert calls[0][1]["key"][0] == calls[1][1]["key"][0]
    old_key = calls[0][1]["key"]
    glue.write_text(glue.read_text() + "\n# distinct ingress\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        bench.compile_grouped_arm(directory, candidate)
    manifest()
    bench.compile_grouped_arm(directory, candidate)
    assert calls[-2][1]["key"] != old_key
    assert bench.grouped_arm_bundle('aaaaaaa', tmp_path) == directory


def test_grouped_arm_materialization_uses_git_revision(monkeypatch, tmp_path):
    calls = []
    def git(args):
        calls.append(args[3:])
        return (b'b' * 40 + b'\n') if args[3] == 'rev-parse' else args[-1].encode()
    monkeypatch.setattr(bench.subprocess, 'check_output', git)
    directory = bench.materialize_grouped_arm('HEAD', tmp_path)
    assert directory.name == 'b' * 40
    assert calls[0] == ['rev-parse', '--verify', 'HEAD^{commit}']
    assert all(call[1].startswith('b' * 40 + ':') for call in calls[1:])


def test_stage_needles_cover_every_step_kernel():
    assert set(bench.STAGES) == {"ingress", "fc1", "activation", "fc2", "egress"}


@pytest.mark.parametrize("rows,block", [(512, 32), (1024, 48), (4096, 64)])
def test_grouped_uniform_uses_serving_policy(monkeypatch, rows, block):
    from b12x.moe._shared.kernels.w4a16 import host

    @dataclass(frozen=True)
    class Trellis:
        tile_config: tuple = (128, 128, 32, 512)

    @dataclass(frozen=True)
    class Tier:
        fc1_tile_n: int = 128
        fc2_tile_n: int = 512
        trellis: Trellis = Trellis()
        trellis_codebook: str = "mcg"

    @dataclass(frozen=True)
    class Buffers:
        block_expert_ids: object
        output: object = None

    calls = {}
    def allocate(tier, **kwargs):
        calls["buffers"] = kwargs
        return Buffers(torch.empty(1))

    def compile_fused(**kwargs):
        calls["fused"] = kwargs
        return object()

    def compile_sum(**kwargs):
        calls["sum"] = kwargs
        return object()

    monkeypatch.setattr(host, "make_w4a16_packed_buffers", allocate)
    monkeypatch.setattr(bench.exl3, "compile_w4a16_fused_moe", compile_fused)
    monkeypatch.setattr(bench.exl3, "compile_w4a16_topk_sum", compile_sum)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "device":"cpu"}))
    state = bench.grouped_uniform_state(rows, Tier(),
        SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376))
    assert state[-1] == block
    assert state[0].trellis.tile_config == (64, 256, 64, 256)
    assert calls["fused"]["moe_block_size"] == block
    assert calls["fused"]["trellis_bits"] == 4
    assert calls["fused"]["full_rotation"]
    assert calls["fused"]["rotation_input_dtype"] == "bf16"
    assert calls["sum"]["full_rotation_output_dtype"] == "bf16"
