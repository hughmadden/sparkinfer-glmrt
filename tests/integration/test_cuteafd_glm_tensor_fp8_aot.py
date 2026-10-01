"""cuteafd GLM 5.3 dense MLP over ModelOpt per-tensor FP8 weights (static W8A8).

nvidia/GLM-5.3-NVFP4 stores layers 0-2's MLP as E4M3 with one FP32
``weight_scale`` and a calibrated ``input_scale`` per projection. The
``tensor_scales`` prefill program must reproduce the static W8A8 numerics
(activations ``satfinite(rn(x / input_scale))``, ``alpha = input_scale *
weight_scale``) and stay close to the BF16 program over the dequantized weights.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x
from ._glm import cosine, golden_rows

SNAPSHOT = Path(os.environ.get(
    "CUTEAFD_GLM_NVFP4_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--nvidia--GLM-5.3-NVFP4/snapshots/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
))

_PROGRAMS: dict = {}


def P(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_ffn

        _PROGRAMS[key] = glm_ffn.compile_glm_ffn_aot(GLM53, **kw)
    return _PROGRAMS[key]


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


@lru_cache(maxsize=1)
def _weight_map():
    index = SNAPSHOT / "model.safetensors.index.json"
    if not index.is_file():
        pytest.skip(f"nvidia GLM 5.3 NVFP4 snapshot not mounted at {SNAPSHOT}")
    return json.loads(index.read_text())["weight_map"]


def _raw(name):
    from safetensors import safe_open

    with safe_open(str(SNAPSHOT / _weight_map()[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _proj(prefix):
    w = _raw(prefix + ".weight").cuda()
    assert w.dtype == torch.float8_e4m3fn
    return w, float(_raw(prefix + ".weight_scale")), float(_raw(prefix + ".input_scale"))


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


def _static_reference(x, gate, up, down):
    """Static W8A8 in torch: FP32 dot products of the E4M3 operands, alpha, one BF16 rounding."""
    def linear(a, proj):
        w, ws, s = proj
        q = (a.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return ((q @ w.float().t()) * torch.tensor(s * ws, dtype=torch.float32)).bfloat16()

    g, u = linear(x, gate).float(), linear(x, up).float()
    hidden = (torch.nn.functional.silu(g).bfloat16().float() * u).bfloat16()
    return linear(hidden, down)


def _tscale(input_scale, *weight_scales):
    alphas = [float(torch.tensor(input_scale, dtype=torch.float32) * torch.tensor(w, dtype=torch.float32))
              for w in weight_scales]
    out = torch.zeros(12, dtype=torch.float32)
    out[0] = input_scale
    for i, a in enumerate(alphas):
        out[4 * (i + 1)] = a
    return out.cuda()


@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("rows", [1, 77, 1024, 4096])
def test_glm_ffn_tensor_fp8_prefill(layer, rows):
    m = f"model.layers.{layer}.mlp."
    gate, up, down = _proj(m + "gate_proj"), _proj(m + "up_proj"), _proj(m + "down_proj")
    assert gate[2] == up[2], "gate and up share the calibrated input scale"
    # Post-attention-norm-like rows: golden residual rows through the layer's norm.
    h = golden_rows(layer, rows).float()
    norm = _raw(f"model.layers.{layer}.post_attention_layernorm.weight").cuda()
    x = norm * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()

    program = P(inter=12288, max_rows=4096, fp8_only="prefill", tensor_scales=True)
    out = torch.empty_like(x)
    program.launch(x, torch.cat([gate[0], up[0]]).contiguous(), _tscale(gate[2], gate[1], up[1]), down[0],
                   _tscale(down[2], down[1]), out, _scratch(program, rows), scalars=(rows,))

    bf16 = P(inter=12288, max_rows=4096)
    deq = lambda p: (p[0].float() * p[1]).bfloat16()  # noqa: E731
    ref = torch.empty_like(x)
    bf16.launch(x, torch.cat([deq(gate), deq(up)]).contiguous(), deq(down), ref, _scratch(bf16, rows),
                scalars=(rows,))
    # The former route: the same bytes under a uniform 128x128 grid, dynamic per-1x128 activations.
    block = P(inter=12288, max_rows=4096, fp8_only="prefill")
    grid = lambda p, n, k: torch.full((n // 128, k // 128), p[1], dtype=torch.float32, device="cuda")  # noqa: E731
    old = torch.empty_like(x)
    block.launch(x, torch.cat([gate[0], up[0]]).contiguous(),
                 torch.cat([grid(gate, 12288, 6144), grid(up, 12288, 6144)]).contiguous(), down[0],
                 grid(down, 6144, 12288), old, _scratch(block, rows), scalars=(rows, 1))
    torch.cuda.synchronize()
    static = _static_reference(x, gate, up, down)
    c_static, c_bf16 = cosine(out, static), cosine(out, ref)
    print(f"layer {layer} rows={rows}: cosine vs static W8A8 {c_static:.7f}, vs BF16 weights {c_bf16:.6f} "
          f"(block W8A8 vs BF16 weights {cosine(old, ref):.6f})")
    assert c_static >= 0.9999
    assert c_bf16 >= 0.995
