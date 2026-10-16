"""NVFP4 expert programs vs independent CPU FP64 W4A4 products.

All local and Spark slices quantize original BF16 activations once with each
expert's calibrated input_scale, E4M3 K16 scales and nearest-even E2M1 values.
Packed weights dequantize exactly; nonlinear/quantizer boundaries remain FP32
and projection stores BF16. Exercise decode, verification and large prefill.
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
            # Nonuniform expert calibration catches accidentally shared FC1 scales.
            inscale = (float(k_scale[0]) * torch.pow(2.0,
                torch.arange(e, device="cuda", dtype=torch.float32).remainder(7) - 3)).contiguous()
            return w, torch.cat([s.flatten(), alpha.view(torch.uint8), inscale.view(torch.uint8)]).contiguous()

        k_scale = [5.0 / (6 * 448)]
        w1, s1 = operand(i, h, real_rows=real)
        w3, s3 = operand(i, h, real_rows=real)
        k_scale[0] = 100.0 / (6 * 448)
        w2, s2 = operand(h, i, real_k=real)
        _WEIGHTS[key] = (w1, s1, w3, s3, w2, s2)
    return _WEIGHTS[key]


_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4(x: torch.Tensor, inscale: float, *, include_input_scale: bool = True) -> torch.Tensor:
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
    value = torch.sign(blocks) * code * sf
    return (value * inscale if include_input_scale else value).reshape(x.shape)


def reference(g, x, ids, weights, w1, s1, w3, s3, w2, s2, a4=False):
    h, i, e = g.hidden, g.slice, g.experts
    (q1, a1), (q3, a3), (q2, a2) = (split_scales(s1, e, i, h), split_scales(s3, e, i, h),
                                    split_scales(s2, e, h, i))
    limit = g.swiglu_limit
    in1, in2 = input_scales(s1, e, i, h), input_scales(s2, e, h, i)
    device = x.device
    if a4:
        # Independent CPU FP64 products; activation scale formation intentionally
        # remains FP32, matching the declared E4M3/E2M1 quantizer's rounding.
        x, ids, weights = x.cpu(), ids.cpu(), weights.cpu()
        a1, a3, a2, in1, in2 = (s.cpu() for s in (a1, a3, a2, in1, in2))
    dtype = torch.float64 if a4 else torch.float32
    out = torch.zeros(x.shape, device=x.device, dtype=dtype)
    routes = torch.zeros((*ids.shape, h), device=x.device, dtype=torch.float32) if a4 else None
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        xr = x[rows].float()
        if a4:
            xr = quantize_nvfp4(xr, float(in1[expert]), include_input_scale=False)
        xr = xr.to(dtype)
        operand = lambda w, q: dequant(w[expert].cpu() if a4 else w[expert],
                                      q[expert].cpu() if a4 else q[expert]).to(dtype)
        # Independent FP64 products, then the kernel's FP32 accumulator/factor
        # boundary before BF16 projection stores. Do not pre-round dequantized
        # activations by folding the global input scale into every element.
        project = lambda a, w, alpha, inscale: ((a @ w.T).float() *
            (alpha * inscale if a4 else alpha)).bfloat16().to(dtype)
        gate = project(xr, operand(w1, q1), a1[expert], in1[expert])
        up = project(xr, operand(w3, q3), a3[expert], in1[expert])
        if g.activation == "silu_post_clamp_fp32_product":
            silu = torch.nn.functional.silu(gate.float())
            up32 = up.float()
            if limit > 0:
                silu, up32 = silu.clamp(max=limit), up32.clamp(-limit, limit)
            act = (silu * up32).bfloat16().to(dtype)
        else:
            if limit > 0:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            act = (torch.nn.functional.silu(gate.float()).bfloat16().to(dtype) * up).bfloat16().to(dtype)
        if a4:
            act = quantize_nvfp4(act, float(in2[expert]), include_input_scale=False).to(dtype)
        y = project(act, operand(w2, q2), a2[expert], in2[expert]).bfloat16()
        if a4:
            routes[rows, slots] = y.float()
        else:
            out.index_add_(0, rows, y.to(dtype) * weights[rows, slots][:, None].to(dtype))
    if a4:
        # Serving combines BF16 routes in router-slot order using FP32, not an
        # expert-ordered FP64 index_add. Products remain independently FP64.
        out = torch.zeros(x.shape, dtype=torch.float32)
        for slot in range(g.top_k):
            out += routes[:, slot] * weights[:, slot, None]
    return out.bfloat16().to(device)


def _run(g, real, capacity, rows, wire=False, seed=1, hot=0, scale=1.0, route="auto"):
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
    a4 = g.activations == "a4"
    expected = reference(g, x_exact, ids, weights, *w, a4=a4)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    if a4:
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        relative_l2 = float((a - b).norm() / b.norm().clamp_min(1e-12))
        row_l2 = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
        normalized_max = float((a - b).abs().max() / b.square().mean().sqrt().clamp_min(1e-12))
        print(f"  CPU FP64 W4A4: rel_l2={relative_l2:.7f} worst_row={float(row_l2.max()):.7f} "
              f"normalized_max={normalized_max:.7f}")
        assert relative_l2 <= 0.02 and float(row_l2.max()) <= 0.04
        assert normalized_max <= 0.15
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
    ("glmf_nvfp4", 4, 512, 1, 1, False, 0, 1.0), ("glmf_nvfp4", 4, 512, 256, 256, False, 8, 1.0),
    # TP6 of 2048 in 16-blocks: 22 blocks (352) or 21 (336) stored 384 wide.
    ("glmf_nvfp4", 6, 352, 16, 16, False, 0, 1.0), ("glmf_nvfp4", 6, 336, 1024, 1000, False, 0, 1.0),
    ("glmf_nvfp4", 2, 1024, 80, 80, False, 0, 1.0),
    ("qwen4_nvfp4", 1, 640, 1, 1, False, 0, 1.0), ("qwen4_nvfp4", 1, 640, 80, 80, False, 0, 1.0),
    ("qwen4_nvfp4", 1, 640, 4096, 4096, False, 0, 1.0),
    # Qwen TP2 / TP3 of 640: 20 blocks (320) in 384, 14 blocks (224) in 256.
    ("qwen4_nvfp4", 2, 320, 16, 16, False, 0, 1.0), ("qwen4_nvfp4", 3, 224, 256, 200, False, 0, 1.0),
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
    ("glmf_nvfp4", 4, 512, 4096, 4096, False, 0, 1.0, "stream"), ("glmf_nvfp4", 6, 336, 1024, 1000, False, 8, 1.0, "stream"),
    ("glmf_nvfp4", 3, 688, 4096, 2100, False, 0, 1.0, "auto"),
    ("qwen4_nvfp4", 1, 640, 4096, 4096, False, 0, 1.0, "stream"), ("qwen4_nvfp4", 1, 640, 4096, 4000, False, 0, 1.0, "auto"),
    ("qwen4_nvfp4", 2, 320, 1024, 1024, False, 0, 1.0, "stream"), ("qwen4_nvfp4", 6, 112, 4096, 4096, False, 0, 1.0, "stream"),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale,route", STREAM_CASES)
def test_nvfp4_moe_stream(name, tp, real, capacity, rows, wire, hot, scale, route):
    g = _geometry(name, tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot, scale=scale, route=route)
    print(f"nvfp4_moe {name} tp{tp} (slice {g.slice}, {real} real) {route} m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'} x{scale}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= 0.99999 and worst >= 0.9999


A4_CASES = [
    ("glm_nvfp4a4", 4, 512, 1, 1, False, 0, 1.0, "decode"),
    ("glm_nvfp4a4", 4, 512, 2048, 1500, False, 8, 1.0, "prefill"),
    ("glmf_nvfp4a4", 4, 512, 16, 16, False, 0, 1.0, "auto"),
    ("qwen4_nvfp4a4", 5, 128, 80, 17, False, 0, 1.0, "auto"),
    # (geometry, tp, real rows, capacity, rows, wire, hot, scale, route): the W4A4 stream route,
    # forced and via auto, including decode and verification.
    ("glmf_nvfp4a4", 1, 2048, 4096, 4096, False, 0, 1.0, "stream"), ("glmf_nvfp4a4", 1, 2048, 256, 17, False, 0, 40.0, "stream"),
    ("glmf_nvfp4a4", 4, 512, 4096, 3000, False, 8, 1.0, "auto"), ("glmf_nvfp4a4", 6, 336, 1024, 1000, False, 0, 1.0, "stream"),
    ("qwen4_nvfp4a4", 1, 640, 4096, 4096, False, 0, 1.0, "auto"), ("qwen4_nvfp4a4", 3, 224, 2048, 2000, False, 0, 1.0, "stream"),
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


FUSED_CASES = [
    # (geometry, tp, real rows, capacity, rows, wire, hot, scale): fused vs separate W4A4 gate/up.
    ("glmf_nvfp4a4", 1, 2048, 4096, 4096, False, 0, 1.0), ("glmf_nvfp4a4", 1, 2048, 256, 17, False, 0, 40.0),
    ("glmf_nvfp4a4", 6, 336, 1024, 1000, False, 8, 1.0), ("qwen4_nvfp4a4", 1, 640, 4096, 3000, False, 0, 1.0),
    ("qwen4_nvfp4a4", 6, 112, 1024, 1000, False, 0, 1.0), ("glmfdense_nvfp4a4", 1, 12288, 2048, 1500, False, 0, 1.0),
]


@pytest.mark.parametrize("name,tp,real,capacity,rows,wire,hot,scale", FUSED_CASES)
def test_nvfp4_moe_a4_fused_exact(name, tp, real, capacity, rows, wire, hot, scale, monkeypatch):
    """The fused gate/up + SwiGLU + NVFP4 quantization keeps every rounding point: the
    stream route's output is bit-identical to the separate GEMMs + ``ChunkSwiGLUA4``."""
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = _geometry(name, tp)
    w = _weights(g, real)
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(rows, g.hidden, device="cuda", generator=gen) * scale).bfloat16()
    source, _ = wire_rows(x) if wire else (x, x)
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    outs = []
    for fused in ("1", "0"):
        monkeypatch.setenv("B12X_NVFP4_A4_FUSED", fused)
        program = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=wire)
        out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity, wire), dtype=torch.uint8, device="cuda")
        program.launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
        torch.cuda.synchronize()
        outs.append(out)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_nvfp4_sparse_tiles_reuse_and_graph(monkeypatch):
    """One capacity program covers sparse/dense chunks under frozen resolution."""
    from b12x.integration.cuteafd import fp8_moe

    g = fp8_moe.Fp8MoeGeometry("nvfp4_sparse_test", hidden=256, experts=32, top_k=2,
                              intermediate=256, weights="nvfp4", activations="a4")
    capacity = 256
    w = _weights(g, g.slice)
    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", "1")
    program = fp8_moe.compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=False)
    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", "0")
    separate = fp8_moe.compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=False)
    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", "1")
    gen = torch.Generator(device="cuda").manual_seed(91)
    x = torch.randn(capacity, g.hidden, device="cuda", generator=gen).bfloat16()
    ids = torch.empty(capacity, g.top_k, device="cuda", dtype=torch.int32)
    weights = torch.rand(capacity, g.top_k, device="cuda", generator=gen)
    out = torch.empty_like(x)
    scratch = torch.empty(fp8_moe.fp8_moe_scratch_bytes(g, "stream", capacity, False),
                          device="cuda", dtype=torch.uint8)
    separate_out = torch.empty_like(out)
    separate_scratch = torch.empty(fp8_moe.fp8_moe_scratch_bytes(g, "stream", capacity, False),
                                   device="cuda", dtype=torch.uint8)
    pointers = tuple(t.data_ptr() for t in (x, ids, weights, *w, out, scratch))

    def no_compile(*args, **kwargs):
        pytest.fail("live rows must reuse the precompiled capacity program")

    monkeypatch.setattr(fp8_moe, "compile_program", no_compile)
    cases = [(rows, False) for rows in (1, 8, 16, 17, 31, 64, 129, 256)] + [(8, True), (16, True)]
    for rows, duplicate in cases:
        # Hot e0/e1 cross the 16/128-row boundaries; remaining routes are sparse.
        ids.copy_(torch.rand(capacity, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int())
        ids[:max(1, rows * 3 // 4)].copy_(torch.arange(g.top_k, device="cuda", dtype=torch.int32))
        if duplicate:
            ids[:rows].zero_()
        out.fill_(float("nan"))
        scratch.fill_(0xA5)

        def run():
            program.launch(x, ids, weights, *w, out, scratch, scalars=(rows,))

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        assert pointers == tuple(t.data_ptr() for t in (x, ids, weights, *w, out, scratch))
        separate_out.fill_(float("nan"))
        separate_scratch.fill_(0xA5)
        separate.launch(x, ids, weights, *w, separate_out, separate_scratch, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out[:rows].view(torch.int16), separate_out[:rows].view(torch.int16))
        actual = out[:rows].float()
        expected = reference(g, x[:rows], ids[:rows], weights[:rows], *w, a4=True).float()
        assert torch.isfinite(actual).all() and actual.count_nonzero() > 0
        assert float((actual * expected).sum() / (actual.norm() * expected.norm())) >= COS
        assert float((actual - expected).norm() / expected.norm()) <= 0.02
        assert float((actual - expected).abs().max() / expected.square().mean().sqrt()) <= 0.15
        assert torch.isnan(out[rows:]).all()
        del graph


def test_nvfp4_geometry():
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, Fp8MoeGeometry, _resolve, auto_large_rows

    for g in GEOMETRIES.values():
        if g.weights != "nvfp4":
            continue
        assert g.activations == "a4" and g.kind == "nvfp4a4"
        for rows in (1, 16, 80, 1024, 4096):
            for route in ("decode", "prefill", "auto", "stream"):
                assert _resolve(route, rows, False, g.kind) == "stream"
                with pytest.raises(ValueError, match="original BF16"):
                    _resolve(route, rows, True, g.kind)
        assert auto_large_rows(False, g.kind) == 0
    with pytest.raises(ValueError, match="W4A16 is not a serving mode"):
        Fp8MoeGeometry("bad", hidden=128, experts=1, top_k=1, intermediate=128, weights="nvfp4")
    assert [_geometry("glmf_nvfp4", tp).slice for tp in (1, 2, 3, 4, 6)] == [2048, 1024, 768, 512, 384]
    assert [_geometry("qwen4_nvfp4", tp).slice for tp in (1, 2, 3, 4, 6)] == [640, 384, 256, 256, 128]


def test_e4m3_scale_is_exact():
    """Every E2M1 x E4M3 product is a BF16 number: the widening is exact."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1))
    scales = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    scales = scales[torch.isfinite(scales)]
    products = table[:, None] * scales[None, :]
    assert torch.equal(products.bfloat16().float(), products)
