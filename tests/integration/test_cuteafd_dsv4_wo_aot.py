"""cuteafd DSV4 inverse-RoPE WO projection AOT program vs b12x.gemm.wo_projection."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, cosine, has_checkpoint, prepare, u8

LAYER = 2


def _weights(geometry, device):
    from b12x.gemm import wo_projection

    g, w, r, d = geometry.o_groups, geometry.o_group_width, geometry.o_lora_rank, geometry.hidden
    if geometry.name == "flash" and has_checkpoint():
        get = lambda n: checkpoint_tensor(f"layers.{LAYER}.attn.{n}", device)  # noqa: E731
        wo_a, wo_a_s, wo_b, wo_b_s = get("wo_a.weight"), get("wo_a.scale"), get("wo_b.weight"), get("wo_b.scale")
    else:
        gen = torch.Generator(device="cpu").manual_seed(d)
        wo_a = (torch.randn((g * r, w), generator=gen) / 16).to(torch.float8_e4m3fn).to(device)
        wo_b = (torch.randn((d, g * r), generator=gen) / 16).to(torch.float8_e4m3fn).to(device)
        wo_a_s = torch.randint(118, 124, (g * r // 128, w // 128), generator=gen, dtype=torch.uint8).to(device)
        wo_b_s = torch.randint(118, 124, (d // 128, g * r // 128), generator=gen, dtype=torch.uint8).to(device)
    return wo_projection.pack_weights(wo_a, u8(wo_a_s), wo_b, u8(wo_b_s), groups=g, group_width=w,
                                      rank=r, hidden=d)


def _cos_sin(positions: int, device) -> torch.Tensor:
    inv = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous().to(device)


def _inputs(geometry, rows, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    o = (torch.randn((rows, geometry.heads, 512), generator=gen) * 0.5).bfloat16().to(device)
    positions = torch.randperm(4096, generator=gen)[:rows].to(torch.int64).to(device)
    return o, positions, _cos_sin(4096, device)


def _prepared(geometry, weights, o, positions, cos_sin, *, max_tokens, dynamic):
    """The prototype's OpPlans.wo / output_projection call (optionally dynamic_tokens)."""
    from b12x.gemm import wo_projection
    from b12x.preparation import FrozenMapping

    caps = wo_projection.Caps(device=o.device, max_tokens=max_tokens, groups=geometry.o_groups,
                              group_width=geometry.o_group_width, rank=geometry.o_lora_rank,
                              hidden=geometry.hidden)
    options = dict(operation="inv_rope", heads_per_group=geometry.heads // geometry.o_groups,
                   nope_dim=448, rope_dim=64, positions_dtype="int64", cos_sin_dtype="float32")
    if dynamic:
        options["dynamic_tokens"] = True
    plan = prepare(wo_projection.plan(caps, invocation=FrozenMapping(options)),
                   f"test.wo.{geometry.name}.{max_tokens}.{dynamic}")
    spec = plan.scratch_specs()[0]
    binding = wo_projection.bind_inv_rope(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=o.device), o=o,
        positions=positions, cos_sin_cache=cos_sin, weights=weights,
        heads_per_group=geometry.heads // geometry.o_groups, nope_dim=448, rope_dim=64,
        expected_m=o.shape[0])
    return wo_projection.run_inv_rope(binding=binding, plan=plan).clone()


def _aot(program, weights, o, positions, cos_sin, geometry):
    rows = o.shape[0]
    out = torch.full((rows, geometry.hidden), float("nan"), dtype=torch.bfloat16, device=o.device)
    scratch = torch.full((program.scratch_bytes(program.geometry["max_rows"])["scratch"],), 0xFF,
                         dtype=torch.uint8, device=o.device)
    program.launch(o, positions, cos_sin, weights.wo_a.values, weights.wo_a.scale_mma,
                   weights.wo_b.values, weights.wo_b.scale_mma, out, scratch, scalars=(rows,))
    return out, scratch


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd.dsv4_wo import compile_dsv4_wo_projection_aot

    with exportable_compilation():
        return {
            ("flash", 16): compile_dsv4_wo_projection_aot(FLASH, max_rows=16),
            ("flash", 8): compile_dsv4_wo_projection_aot(FLASH, max_rows=8),
            ("flash", 512): compile_dsv4_wo_projection_aot(FLASH, max_rows=512),
            ("pro", 4): compile_dsv4_wo_projection_aot(PRO, max_rows=4),
        }


def _geometry(name):
    from b12x.integration.cuteafd import FLASH, PRO

    return FLASH if name == "flash" else PRO


@pytest.mark.parametrize(("name", "max_rows", "rows"), [
    ("flash", 16, 1), ("flash", 16, 5), ("flash", 16, 16), ("flash", 8, 3),
    ("flash", 512, 300), ("flash", 512, 512), ("pro", 4, 3),
])
def test_matches_prepared_dynamic_plan(programs, name, max_rows, rows):
    device = require_b12x()
    geometry = _geometry(name)
    weights = _weights(geometry, device)
    o, positions, cos_sin = _inputs(geometry, rows, 50 + rows, device)
    expected = _prepared(geometry, weights, o, positions, cos_sin, max_tokens=max_rows, dynamic=True)
    out, _ = _aot(programs[(name, max_rows)], weights, o, positions, cos_sin, geometry)
    torch.cuda.synchronize()
    if max_rows > 8:
        assert_bitwise(out, expected, "out")
    else:
        _assert_split_close(out, expected)
        # The FP32 split reduction is at least as close to the unsplit
        # (single-pass) GEMM as the prepared BF16-atomic split.
        unsplit = _prepared(geometry, weights, o, positions, cos_sin, max_tokens=16, dynamic=True)
        err_aot = float((out.float() - unsplit.float()).abs().max())
        err_prepared = float((expected.float() - unsplit.float()).abs().max())
        assert err_aot <= err_prepared


def _assert_split_close(out, expected):
    """Prepared wo_b at <= 8 rows adds its K halves with BF16 atomics (two
    roundings); the AOT program sums FP32 partials (one rounding)."""
    rows_max = expected.float().abs().amax(dim=-1, keepdim=True)
    assert bool(((out.float() - expected.float()).abs() <= rows_max * 2.0 ** -6).all())
    assert cosine(out, expected) > 0.99999


@pytest.mark.parametrize(("max_rows", "rows"), [(16, 16), (8, 3), (512, 300)])
def test_matches_prototype_exact_token_plan(programs, max_rows, rows):
    """The prototype declares exact-T plans (fused wo_b at <= 8 rows)."""
    device = require_b12x()
    geometry = _geometry("flash")
    weights = _weights(geometry, device)
    o, positions, cos_sin = _inputs(geometry, rows, 70 + rows, device)
    expected = _prepared(geometry, weights, o, positions, cos_sin, max_tokens=rows, dynamic=False)
    out, _ = _aot(programs[("flash", max_rows)], weights, o, positions, cos_sin, geometry)
    torch.cuda.synchronize()
    if rows > 8:
        assert_bitwise(out, expected, "out")
    else:
        _assert_split_close(out, expected)


def test_replays_in_cuda_graph(programs):
    device = require_b12x()
    geometry = _geometry("flash")
    weights = _weights(geometry, device)
    program = programs[("flash", 16)]
    rows = 7
    o, positions, cos_sin = _inputs(geometry, rows, 90, device)
    out = torch.empty((rows, geometry.hidden), dtype=torch.bfloat16, device=device)
    scratch = torch.empty((program.scratch_bytes(16)["scratch"],), dtype=torch.uint8, device=device)
    args = (o, positions, cos_sin, weights.wo_a.values, weights.wo_a.scale_mma,
            weights.wo_b.values, weights.wo_b.scale_mma, out, scratch)
    program.launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(*args, scalars=(rows,))
    out.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(out, eager, "graph out")


def test_scratch_formula_is_monotone_and_capacity_sized(programs):
    program = programs[("flash", 512)]
    sizes = [program.scratch_bytes(rows)["scratch"] for rows in (1, 2, 127, 128, 129, 300, 512)]
    assert sizes == sorted(sizes)
    small = programs[("flash", 8)]
    # wo_b splits K below 16 rows: the FP32 partial planes are part of scratch.
    assert small.geometry["split_k_slices"][1] == 2
    assert small.scratch_bytes(8)["scratch"] >= 2 * 8 * 4096 * 4


@pytest.mark.parametrize("key", [("flash", 16), ("flash", 8), ("pro", 4)])
def test_export_to_c_signature(programs, key, tmp_path):
    stem = "dsv4_wo_" + "_".join(map(str, key))
    info = check_export(programs[key], tmp_path, stem)
    assert info["argument_count"] == len(programs[key].operands) + 2
