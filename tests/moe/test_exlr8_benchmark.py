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


def test_default_grouped_half_partials_report_real_error():
    expected = torch.full((2, 128), 1e-7, dtype=torch.bfloat16)
    actual = (expected.float() + 2.0 ** -27).bfloat16()
    weights = torch.ones((2, 2)) / 2
    metrics = bench.grouped_error(actual, expected, weights, half_partials=True)
    assert not metrics["bitwise_equal"]
    assert metrics["relative_l2"] > 0.01
    assert metrics["max_abs"] == float((actual.float() - expected.float()).abs().max())
    with pytest.raises(RuntimeError, match="differs from snapshot chunked K1"):
        bench.grouped_error(actual, expected, weights, half_partials=False)
    exact = bench.grouped_error(expected, expected, weights, half_partials=False)
    assert exact["bitwise_equal"] and exact["relative_l2"] == exact["max_abs"] == 0


@pytest.mark.parametrize("value", [1.0, float("nan"), float("inf")])
def test_default_grouped_half_partials_reject_bad_output(value):
    expected = torch.full((2, 128), 1e-7, dtype=torch.bfloat16)
    actual = torch.full_like(expected, value)
    with pytest.raises(RuntimeError, match="differs from snapshot chunked K1"):
        bench.grouped_error(actual, expected, torch.ones((2, 2)) / 2, half_partials=True)


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


def test_k4_arm_materializes_all_sibling_dependencies(monkeypatch, tmp_path):
    calls = []
    def git(args):
        calls.append(args[3:])
        return (b'c' * 40 + b'\n') if args[3] == 'rev-parse' else args[-1].encode()
    monkeypatch.setattr(bench.subprocess, 'check_output', git)
    directory = bench.materialize_k4_arm('HEAD', tmp_path)
    assert directory == tmp_path / ('c' * 40) / 'k4'
    manifest = bench.json.loads((directory / 'manifest.json').read_text())
    assert set(manifest['sources']) == set(bench.K4_ARM_MODULES)
    assert all(call[1].startswith('c' * 40 + ':') for call in calls[1:])
    for name, digest in manifest['sources'].items():
        assert digest == bench.hashlib.sha256((directory / f'{name}.py').read_bytes()).hexdigest()
    (directory / 'mixed_trellis_ws.py').write_text('tampered')
    with pytest.raises(ValueError, match='digest mismatch'):
        bench.load_k4_arm(directory)


def test_k4_arm_isolates_revision_module_names(tmp_path):
    packages = []
    for revision in ('a' * 40, 'b' * 40):
        directory = tmp_path / revision / 'k4'
        directory.mkdir(parents=True)
        manifest = dict(revision=revision, sources={})
        for name in bench.K4_ARM_MODULES:
            source = f'REVISION = {revision!r}\n'
            (directory / f'{name}.py').write_text(source)
            manifest['sources'][name] = bench.hashlib.sha256(source.encode()).hexdigest()
        (directory / 'manifest.json').write_text(bench.json.dumps(manifest))
        module, _ = bench.load_k4_arm(directory)
        assert module.REVISION == revision
        packages.append(module.__package__)
    assert packages[0] != packages[1]


@pytest.mark.parametrize('control,passes', [(100.5, True), (103.0, False)])
def test_k4_identical_control_noise_gate(control, passes):
    result = bench.k4_noise_summary(dict(base=[100, 100, 100],
        base_control=[control] * 3, candidate=[90, 90, 90]), 'base')
    assert result['noise_pass'] == passes
    assert result['speedups']['candidate'] == pytest.approx(100 / 90)


def test_serving_tier_bits_and_payload_bytes():
    owned, _ = bench.rank_geometry(6, 0, 3)
    bits_mixed = bench.serving_tier_bits(owned=owned, tier_mix="mixed")
    bits_k4 = bench.serving_tier_bits(owned=owned, tier_mix="k4")
    assert set(bits_mixed.unique().tolist()) == {0, 2, 3, 4}
    assert set(bits_k4.unique().tolist()) == {0, 4}
    for e in range(256):
        for u in range(16):
            if u in owned[e]:
                assert [int(bits_mixed[e, u, p]) for p in range(3)] == [(2, 3, 4)[(u + p) % 3] for p in range(3)]
            else:
                assert int(bits_mixed[e, u].abs().sum()) == 0
    ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    expected_bits = int(bits_mixed[[0, 1, 2]].sum().item())
    assert bench.serving_route_payload_bytes(ids, bits_mixed) == expected_bits * (6144 // 16) * (128 // 16) * 32


def test_routed_serving_variant_keys_keep_unrelated_programs_unchanged(monkeypatch):
    from b12x.integration.cuteafd import exlr8 as exlr8_mod
    from b12x.integration.cuteafd.exlr8 import DEFAULT_SERVING_VARIANT, compile_exlr8_a16_aot
    monkeypatch.setattr(exlr8_mod, "compile_program", lambda wrapper, **kwargs: SimpleNamespace(wrapper=wrapper, **kwargs))
    decode = compile_exlr8_a16_aot(device_count=False)
    shared = compile_exlr8_a16_aot(top_k=1, max_groups=4, device_count=True, shared_expert=True)
    grouped = compile_exlr8_a16_aot(device_count=True, grouped=True, group_rows=64)
    default_srv = compile_exlr8_a16_aot(device_count=True)
    base_srv = compile_exlr8_a16_aot(device_count=True, serving_variant="baseline")
    v4 = compile_exlr8_a16_aot(device_count=True, serving_variant="k1_k23_coalesced")
    assert DEFAULT_SERVING_VARIANT == "k1_k23_coalesced"
    assert decode.key[-1] == "decode_l2_fc2_prefetch_v9"
    assert shared.key[-1] == "decode_l2_fc2_prefetch_v9"
    assert grouped.key[-1] == "grouped_compact_units"
    assert default_srv.key == v4.key
    assert default_srv.geometry["serving_variant"] == "k1_k23_coalesced"
    assert base_srv.key[-1] == "baseline"
    assert base_srv.key != v4.key
    from b12x.moe._shared.kernels.w4a16.exlr8 import SERVING_VARIANTS, Exlr8RoutedServingKernel
    assert SERVING_VARIANTS == (DEFAULT_SERVING_VARIANT,)
    for variant in ("k1_compact_prefetch", "k1_paired_fc1", "k1_reg_pair_b2"):
        with pytest.raises(ValueError):
            compile_exlr8_a16_aot(device_count=True, serving_variant=variant)
        with pytest.raises(ValueError):
            Exlr8RoutedServingKernel(6144, 16, 8, 512, variant=variant)

