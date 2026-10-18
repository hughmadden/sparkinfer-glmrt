"""Real Kimi K3 KDA parity; opt in with CUTEAFD_KIMI_KDA_FIXTURE.

Build the fixture with fixtures/kimi_kda_official.py in the pinned official
runtime. The candidate needs only coordinator-dev, not FLA or official code.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

pytestmark = pytest.mark.skipif(not os.environ.get("CUTEAFD_KIMI_KDA_FIXTURE"),
                                reason="requires the pinned official Kimi KDA fixture")


def _metrics(name, actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-20))
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"{name}: relative_l2={rel:.8g} cosine={cosine:.8g}", flush=True)
    # BF16 projections and independent chunk summation orders need not be bitwise.
    assert rel < 0.01 and cosine > 0.99995, name


@pytest.fixture(scope="module")
def fixture():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.load(Path(os.environ["CUTEAFD_KIMI_KDA_FIXTURE"]), weights_only=True, map_location="cuda")


def _packed(weights, rank):
    d, hd, heads = 12288, 128, 96
    sl = slice(rank * d // 2, (rank + 1) * d // 2)
    hs = slice(rank * heads // 2, (rank + 1) * heads // 2)
    parts = [weights[n + ".weight"][sl] for n in ("q_proj", "k_proj", "v_proj")]
    parts += [weights["f_a_proj.weight"], weights["g_proj.weight"][sl], weights["b_proj.weight"][hs]]
    return [torch.cat(parts).contiguous(), weights["f_b_proj.weight"][sl].contiguous(),
            torch.cat([weights[n + "_conv1d.weight"][sl].reshape(-1, 4)
                       for n in ("q", "k", "v")]).float().contiguous(),
            weights["A_log"][hs].contiguous(), weights["dt_bias"][sl].contiguous(),
            weights["o_norm.weight"], weights["o_proj.weight"][:, sl].contiguous()]


def _run(program, packed, x, state, conv, *, spec=0):
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout
    rows = x.shape[0]
    g = program[1]
    program = program[0]
    out = torch.empty((rows, g.kda_width), dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first = torch.zeros_like(slots)
    scratch = torch.empty(program.scratch_bytes(rows=rows)["scratch"], dtype=torch.uint8, device="cuda")
    if program.geometry.get("bf16_decode"):
        replay = torch.zeros(kda_replay_layout(48, 3 * g.kda_width, 64)[2] // 4, device="cuda")
        program.launch(x, *packed, conv, state, slots, first, out, replay, scratch, scalars=(rows, spec))
    else:
        replay = None
        program.launch(x, *packed, conv, state, slots, first, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    return out, replay


@pytest.mark.parametrize("rows", [1, 8, 129])
def test_kimi_kda_tp2_matches_official_bf16(fixture, rows):
    from b12x.integration.cuteafd import KDAGeometry, glmf
    g = KDAGeometry(hidden=7168, kda_heads=96, norm_eps=fixture["eps"]).head_split(2)
    program = glmf.compile_glmf_kda_aot(g, max_rows=max(rows, 64), decode=rows <= 64, output_kind="norm")
    x = fixture["x"][0, :rows].contiguous()
    case = fixture["cases"][str(rows)]
    outputs = []
    for rank in (0, 1):
        state = torch.zeros((1, 48, 128, 128), device="cuda")
        conv = torch.zeros((1, 3, 3 * g.kda_width), dtype=torch.bfloat16, device="cuda")
        packed = _packed(fixture["weights"], rank)
        out, replay = _run((program, g), packed, x, state, conv)
        _metrics(f"rows={rows} rank={rank} norm", out, case["norm"][0, :, rank * 48:(rank + 1) * 48].flatten(1))
        _metrics(f"rows={rows} rank={rank} state", state, case["state"][:, rank * 48:(rank + 1) * 48])
        outputs.append(out)
        # The pool holds only the previous three raw projection rows, oldest first.
        ref_conv = torch.cat([c[:, rank * 6144:(rank + 1) * 6144, -3:].transpose(1, 2)
                              for c in case["conv"]], dim=-1)
        _metrics(f"rows={rows} rank={rank} conv", conv, ref_conv)
        if rows <= 64:
            verify_state, verify_conv = torch.zeros_like(state), torch.zeros_like(conv)
            verify, record = _run((program, g), packed, x, verify_state, verify_conv, spec=1)
            assert torch.equal(verify, out)
            assert torch.count_nonzero(verify_state) == 0
            assert torch.count_nonzero(verify_conv) == 0
            commit = glmf.compile_glmf_kda_commit_aot(g)
            tables = torch.tensor([[0], [0], [rows]], dtype=torch.int32, device="cuda")
            commit.launch(verify_state, verify_conv, record, tables, scalars=(1, 1, 1))
            torch.cuda.synchronize()
            assert torch.equal(verify_state, state)
            assert torch.equal(verify_conv, conv)
    full = torch.cat(outputs, dim=1)
    _metrics(f"rows={rows} output", torch.nn.functional.linear(full, fixture["weights"]["o_proj.weight"]), case["out"][0])
