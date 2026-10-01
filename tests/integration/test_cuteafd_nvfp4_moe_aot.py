"""cuteafd NVFP4 routed-expert programs (``fp8_moe`` with ``weights="nvfp4"``) vs torch.

Random ModelOpt NVFP4 experts (packed E2M1 bytes, E4M3 per-16 scales, an FP32
``weight_scale_2`` per expert and projection) at the GLM 5.3 Flash (H 4096,
288 experts, top-8, SwiGLU clamp 10) and Qwen 3.8 Flash Next (H 2560, 512
experts, top-10, intermediate 640) geometries: full width (TP1, BF16 input:
the RTX local layers) and zero-padded Spark slices (FP8 K32 wire input). The
reference dequantizes in torch: ``e2m1 * e4m3`` (E4M3 through
``torch.float8_e4m3fn``) in FP32, ``F.linear`` in FP32 times alpha, one BF16
rounding per projection, ``bf16(silu(g)) * u`` with the config's clamp, FP32
weighted route sum.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_fp8_moe_aot import wire_rows

COS = 0.9999
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_WEIGHTS: dict = {}
_PROGRAMS: dict = {}


def dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """U8 ``[N, K/2]`` + E4M3 bytes ``[N, K/16]`` -> FP32 ``[N, K]`` (``e2m1 * e4m3``, no alpha)."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).flatten(-2).long()
    return table[codes] * scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)


def split_scales(s: torch.Tensor, experts: int, rows: int, k: int):
    """An NVFP4 scale operand -> (E4M3 bytes ``[E, rows, K/16]``, FP32 alphas ``[E]``)."""
    grid = experts * rows * (k // 16)
    return s[:grid].view(experts, rows, k // 16), s[grid:].view(torch.float32)


def _geometry(name, tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES[name].with_tp(tp)


def _weights(g, real: int):
    key = (g.name, g.tp, real)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        gen = torch.Generator(device="cuda").manual_seed(0)
        h, i, e = g.hidden, g.slice, g.experts

        def codes(shape):
            return torch.randint(0, 256, shape, device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)

        def scales(shape):
            # Positive E4M3 codes 2^-3 .. 2^3 (exponent field 4 .. 10), some subnormals.
            exponent = torch.randint(4, 11, shape, device="cuda", generator=gen, dtype=torch.int32)
            mantissa = torch.randint(0, 8, shape, device="cuda", generator=gen, dtype=torch.int32)
            sub = torch.rand(shape, device="cuda", generator=gen) < 0.02
            exponent = torch.where(sub, torch.zeros_like(exponent), exponent)
            mantissa = torch.where(sub, mantissa.clamp_min(1), mantissa)
            return (exponent * 8 + mantissa).to(torch.uint8)

        def operand(rows, k, real_rows=None, real_k=None):
            w, s = codes((e, rows, k // 2)), scales((e, rows, k // 16))
            if real_rows is not None:
                w[:, real_rows:] = 0
                s[:, real_rows:] = 0
            if real_k is not None:
                w[:, :, real_k // 2:] = 0
                s[:, :, real_k // 16:] = 0
            alpha = (torch.rand(e, device="cuda", generator=gen) * 4e-3 + 1e-3).float()
            return w, torch.cat([s.flatten(), alpha.view(torch.uint8)]).contiguous()

        w1, s1 = operand(i, h, real_rows=real)
        w3, s3 = operand(i, h, real_rows=real)
        w2, s2 = operand(h, i, real_k=real)
        _WEIGHTS[key] = (w1, s1, w3, s3, w2, s2)
    return _WEIGHTS[key]


def reference(g, x, ids, weights, w1, s1, w3, s3, w2, s2):
    h, i, e = g.hidden, g.slice, g.experts
    (q1, a1), (q3, a3), (q2, a2) = (split_scales(s1, e, i, h), split_scales(s3, e, i, h),
                                    split_scales(s2, e, h, i))
    limit = g.swiglu_limit
    out = torch.zeros(x.shape, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        xr = x[rows].float()
        gate = (xr @ dequant(w1[expert], q1[expert]).T * a1[expert]).bfloat16().float()
        up = (xr @ dequant(w3[expert], q3[expert]).T * a3[expert]).bfloat16().float()
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        act = (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16().float()
        y = (act @ dequant(w2[expert], q2[expert]).T * a2[expert]).bfloat16()
        out.index_add_(0, rows, y.float() * weights[rows, slots][:, None])
    return out.bfloat16()


def _run(g, real, capacity, rows, wire=True, seed=1, hot=0, scale=1.0):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = _weights(g, real)
    x = (torch.randn(rows, g.hidden, device="cuda", generator=gen) * scale).bfloat16()
    source, x_exact = wire_rows(x) if wire else (x, x)
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    key = (g, capacity, wire)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=wire)
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, wire), dtype=torch.uint8, device="cuda")
    _PROGRAMS[key].launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    expected = reference(g, x_exact, ids, weights, *w)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    return c, worst


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (geometry, tp, real rows of the slice, capacity, rows, wire, hot, input scale)
    ("glmf_nvfp4", 1, 2048, 1, 1, False, 0, 1.0), ("glmf_nvfp4", 1, 2048, 16, 16, False, 0, 1.0),
    ("glmf_nvfp4", 1, 2048, 1024, 700, False, 8, 1.0), ("glmf_nvfp4", 1, 2048, 4096, 4096, False, 0, 1.0),
    # Large inputs drive gate/up past the clamp.
    ("glmf_nvfp4", 1, 2048, 16, 16, False, 0, 40.0),
    ("glmf_nvfp4", 4, 512, 1, 1, True, 0, 1.0), ("glmf_nvfp4", 4, 512, 256, 256, True, 8, 1.0),
    # TP6 of 2048 in 16-blocks: 22 blocks (352) or 21 (336) stored 384 wide.
    ("glmf_nvfp4", 6, 352, 16, 16, True, 0, 1.0), ("glmf_nvfp4", 6, 336, 1024, 1000, True, 0, 1.0),
    ("glmf_nvfp4", 2, 1024, 80, 80, True, 0, 1.0),
    ("qwen4_nvfp4", 1, 640, 1, 1, False, 0, 1.0), ("qwen4_nvfp4", 1, 640, 80, 80, False, 0, 1.0),
    ("qwen4_nvfp4", 1, 640, 4096, 4096, False, 0, 1.0),
    # Qwen TP2 / TP3 of 640: 20 blocks (320) in 384, 14 blocks (224) in 256.
    ("qwen4_nvfp4", 2, 320, 16, 16, True, 0, 1.0), ("qwen4_nvfp4", 3, 224, 256, 200, True, 0, 1.0),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale", CASES)
def test_nvfp4_moe(name, tp, real, capacity, rows, wire, hot, scale):
    g = _geometry(name, tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot, scale=scale)
    print(f"nvfp4_moe {name} tp{tp} (slice {g.slice}, {real} real) m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'} x{scale}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


def test_nvfp4_geometry():
    assert [_geometry("glmf_nvfp4", tp).slice for tp in (1, 2, 3, 4, 6)] == [2048, 1024, 768, 512, 384]
    assert [_geometry("qwen4_nvfp4", tp).slice for tp in (1, 2, 3, 4, 6)] == [640, 384, 256, 256, 128]


def test_e4m3_scale_is_exact():
    """Every E2M1 x E4M3 product is a BF16 number: the widening is exact."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1))
    scales = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    scales = scales[torch.isfinite(scales)]
    products = table[:, None] * scales[None, :]
    assert torch.equal(products.bfloat16().float(), products)
