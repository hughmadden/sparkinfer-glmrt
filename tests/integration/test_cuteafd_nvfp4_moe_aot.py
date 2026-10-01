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
    return s[:grid].view(experts, rows, k // 16), s[grid:grid + 4 * experts].view(torch.float32)


def input_scales(s: torch.Tensor, experts: int, rows: int, k: int) -> torch.Tensor:
    """The FP32 input scales ``[E]`` after an NVFP4 scale operand's alphas."""
    grid = experts * rows * (k // 16)
    return s[grid + 4 * experts:].view(torch.float32)


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
            # One static input scale per layer and projection, as the ModelOpt releases store it
            # (amax / (6 * 448) of activations ~N(0, 1) and of the SwiGLU outputs).
            inscale = torch.full((e,), float(k_scale[0]), device="cuda")
            return w, torch.cat([s.flatten(), alpha.view(torch.uint8), inscale.view(torch.uint8)]).contiguous()

        k_scale = [5.0 / (6 * 448)]
        w1, s1 = operand(i, h, real_rows=real)
        w3, s3 = operand(i, h, real_rows=real)
        k_scale[0] = 100.0 / (6 * 448)
        w2, s2 = operand(h, i, real_k=real)
        _WEIGHTS[key] = (w1, s1, w3, s3, w2, s2)
    return _WEIGHTS[key]


_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4(x: torch.Tensor, inscale: float) -> torch.Tensor:
    """FP32 ``[rows, K]`` -> its NVFP4 value (``quantize_block_fp4``: E4M3 scale of
    ``amax * gs / 6`` per 16, E2M1 nearest-even), dequantized with ``input_scale``."""
    gs = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(inscale, dtype=torch.float32)
    gs = gs.to(x.device)
    blocks = x.float().reshape(x.shape[0], -1, 16)
    amax = blocks.abs().amax(-1, keepdim=True)
    sf = (amax * gs / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    vs = sf / gs
    mag = blocks.abs()
    t = [vs * v for v in _THRESHOLDS]
    code = torch.zeros_like(mag)
    code = torch.where((mag > t[0]) & (mag < t[1]), 0.5, code)
    code = torch.where((mag >= t[1]) & (mag <= t[2]), 1.0, code)
    code = torch.where((mag > t[2]) & (mag < t[3]), 1.5, code)
    code = torch.where((mag >= t[3]) & (mag <= t[4]), 2.0, code)
    code = torch.where((mag > t[4]) & (mag < t[5]), 3.0, code)
    code = torch.where((mag >= t[5]) & (mag <= t[6]), 4.0, code)
    code = torch.where(mag > t[6], 6.0, code)
    code = torch.where(sf == 0, 0.0, code)
    return (torch.sign(blocks) * code * sf * inscale).reshape(x.shape)


def reference(g, x, ids, weights, w1, s1, w3, s3, w2, s2, a4=False):
    h, i, e = g.hidden, g.slice, g.experts
    (q1, a1), (q3, a3), (q2, a2) = (split_scales(s1, e, i, h), split_scales(s3, e, i, h),
                                    split_scales(s2, e, h, i))
    limit = g.swiglu_limit
    in1, in2 = input_scales(s1, e, i, h), input_scales(s2, e, h, i)
    out = torch.zeros(x.shape, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        xr = x[rows].float()
        if a4:
            xr = quantize_nvfp4(xr, float(in1[expert]))
        gate = (xr @ dequant(w1[expert], q1[expert]).T * a1[expert]).bfloat16().float()
        up = (xr @ dequant(w3[expert], q3[expert]).T * a3[expert]).bfloat16().float()
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        act = (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16().float()
        if a4:
            act = quantize_nvfp4(act, float(in2[expert]))
        y = (act @ dequant(w2[expert], q2[expert]).T * a2[expert]).bfloat16()
        out.index_add_(0, rows, y.float() * weights[rows, slots][:, None])
    return out.bfloat16()


def _run(g, real, capacity, rows, wire=True, seed=1, hot=0, scale=1.0, route="auto"):
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
    key = (g, capacity, wire, route)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route=route, max_rows=capacity, wire=wire)
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity, wire), dtype=torch.uint8, device="cuda")
    _PROGRAMS[key].launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    from b12x.integration.cuteafd.fp8_moe import auto_large_rows

    a4 = g.activations == "a4" and (route == "stream" or (route == "auto" and rows > auto_large_rows(wire, g.kind)))
    expected = reference(g, x_exact, ids, weights, *w, a4=a4)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    if a4:
        exact = reference(g, x_exact, ids, weights, *w).float()
        print(f"  W4A4 vs the W4A16 reference: cosine {float((a * exact).sum() / (a.norm() * exact.norm())):.6f}")
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
    # GLM 5.3 Flash's dense MLP as one expert (ids 0).
    ("glmfdense_nvfp4", 1, 12288, 1, 1, False, 0, 1.0), ("glmfdense_nvfp4", 1, 12288, 16, 16, False, 0, 1.0),
    ("glmfdense_nvfp4", 1, 12288, 4096, 3000, False, 0, 1.0),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale", CASES)
def test_nvfp4_moe(name, tp, real, capacity, rows, wire, hot, scale):
    g = _geometry(name, tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot, scale=scale)
    print(f"nvfp4_moe {name} tp{tp} (slice {g.slice}, {real} real) m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'} x{scale}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


STREAM_CASES = [
    # (geometry, tp, real rows, capacity, rows, wire, hot, scale, route): the stream route, forced and via auto.
    ("glmf_nvfp4", 1, 2048, 4096, 4096, False, 0, 1.0, "stream"), ("glmf_nvfp4", 1, 2048, 4096, 3000, False, 8, 1.0, "auto"),
    ("glmf_nvfp4", 1, 2048, 256, 17, False, 0, 40.0, "stream"),
    ("glmf_nvfp4", 4, 512, 4096, 4096, True, 0, 1.0, "stream"), ("glmf_nvfp4", 6, 336, 1024, 1000, True, 8, 1.0, "stream"),
    ("glmf_nvfp4", 3, 688, 4096, 2100, True, 0, 1.0, "auto"),
    ("qwen4_nvfp4", 1, 640, 4096, 4096, False, 0, 1.0, "stream"), ("qwen4_nvfp4", 1, 640, 4096, 4000, False, 0, 1.0, "auto"),
    ("qwen4_nvfp4", 2, 320, 1024, 1024, True, 0, 1.0, "stream"), ("qwen4_nvfp4", 6, 112, 4096, 4096, True, 0, 1.0, "stream"),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale,route", STREAM_CASES)
def test_nvfp4_moe_stream(name, tp, real, capacity, rows, wire, hot, scale, route):
    g = _geometry(name, tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot, scale=scale, route=route)
    print(f"nvfp4_moe {name} tp{tp} (slice {g.slice}, {real} real) {route} m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'} x{scale}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= 0.99999 and worst >= 0.9999


A4_CASES = [
    # (geometry, tp, real rows, capacity, rows, wire, hot, scale, route): the W4A4 stream route,
    # forced and via auto (decode rows stay W4A16).
    ("glmf_nvfp4a4", 1, 2048, 4096, 4096, False, 0, 1.0, "stream"), ("glmf_nvfp4a4", 1, 2048, 256, 17, False, 0, 40.0, "stream"),
    ("glmf_nvfp4a4", 4, 512, 4096, 3000, True, 8, 1.0, "auto"), ("glmf_nvfp4a4", 6, 336, 1024, 1000, True, 0, 1.0, "stream"),
    ("qwen4_nvfp4a4", 1, 640, 4096, 4096, False, 0, 1.0, "auto"), ("qwen4_nvfp4a4", 3, 224, 2048, 2000, True, 0, 1.0, "stream"),
    ("qwen4_nvfp4a4", 1, 640, 4096, 300, False, 0, 1.0, "auto"),
    ("glmfdense_nvfp4a4", 1, 12288, 4096, 1500, False, 0, 1.0, "auto"),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale,route", A4_CASES)
def test_nvfp4_moe_a4(name, tp, real, capacity, rows, wire, hot, scale, route):
    g = _geometry(name, tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot, scale=scale, route=route)
    print(f"nvfp4_moe {name} tp{tp} (slice {g.slice}, {real} real) a4 m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'} x{scale}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= 0.9999 and worst >= 0.999


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
