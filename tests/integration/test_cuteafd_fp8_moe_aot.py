"""cuteafd exact FP8 routed-expert programs (``fp8_moe``) vs a torch reference.

Random E4M3 expert weights with FP32 128x128 block scales at the MiMo V2
Flash (H 4096) and GLM 5.3 (H 6144) geometries, full width (TP1, the RTX
local/MTP layers) and the TP4 Spark slice (I 512); FP8 K32 wire or BF16
input rows; decode (grouped GEMV), prefill (grouped TMA GEMM), stream
(expert-stationary streaming GEMMs, FP8 wire input) and auto routes. The
reference is the checkpoint semantics: ``F.linear`` of BF16 activations
with ``bf16(w * s)`` weights, BF16 gate/up, ``bf16(silu(g)) *
u``, BF16 down output, FP32 weighted route sum.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

COS = 0.9999
_WEIGHTS: dict = {}
_PROGRAMS: dict = {}


def _geometry(name, tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES[name].with_tp(tp)


def _weights(g):
    key = (g.name, g.tp)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        gen = torch.Generator(device="cuda").manual_seed(0)
        h, i, e = g.hidden, g.slice, g.experts

        def fp8(shape):
            return (torch.randn(shape, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)

        def scales(shape):
            return torch.rand(shape, device="cuda", generator=gen) * 2e-4 + 1e-4

        _WEIGHTS[key] = (fp8((e, i, h)), scales((e, i // 128, h // 128)), fp8((e, i, h)),
                         scales((e, i // 128, h // 128)), fp8((e, h, i)), scales((e, h // 128, i // 128)))
    return _WEIGHTS[key]


def _program(g, route, capacity, wire):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot

    key = (g, route, capacity, wire)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route=route, max_rows=capacity, wire=wire)
    return _PROGRAMS[key]


def wire_rows(x: torch.Tensor):
    """FP8 K32 wire rows of BF16 ``x`` and their exact BF16 value."""
    rows, h = x.shape
    groups = x.float().view(rows, h // 32, 32)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-127, 127)
    q = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
    wire = torch.cat([q.view(rows, h).view(torch.uint8), (exponent + 127).to(torch.uint8)], 1).contiguous()
    return wire, (q.float() * torch.exp2(exponent)[..., None]).view(rows, h).bfloat16()


def reference(x, ids, weights, w1, s1, w3, s3, w2, s2, limit=0.0):
    def dequant(w, s):
        return (w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()

    out = torch.zeros(x.shape, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = x[rows] @ dequant(w1[expert], s1[expert]).T
        up = x[rows] @ dequant(w3[expert], s3[expert]).T
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        y = (torch.nn.functional.silu(gate) * up) @ dequant(w2[expert], s2[expert]).T
        out.index_add_(0, rows, y.float() * weights[rows, slots][:, None])
    return out.bfloat16()


def _run(g, route, capacity, rows, wire=True, seed=1, hot=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = _weights(g)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    source, x_exact = wire_rows(x) if wire else (x, x)
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0  # every row routes to the first `hot` experts (multi-chunk groups)
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    from b12x.integration.cuteafd.fp8_moe import fp8_moe_scratch_bytes

    program = _program(g, route, capacity, wire)
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity), dtype=torch.uint8, device="cuda")
    program.launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    expected = reference(x_exact, ids, weights, *w, limit=g.swiglu_limit)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    return c, worst


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (geometry, tp, route, capacity, rows)
    ("mimo", 4, "decode", 1, 1), ("mimo", 4, "decode", 16, 16), ("mimo", 4, "decode", 80, 80),
    ("mimo", 4, "auto", 1024, 1024), ("mimo", 4, "auto", 4096, 4096), ("mimo", 4, "prefill", 4096, 80),
    ("mimo", 1, "decode", 1, 1), ("mimo", 1, "decode", 16, 16), ("mimo", 1, "auto", 80, 80),
    ("mimo", 1, "auto", 1024, 1024), ("mimo", 1, "auto", 4096, 4096),
    ("glm", 1, "decode", 1, 1), ("glm", 1, "auto", 80, 80), ("glm", 1, "auto", 4096, 4096),
    ("glm", 4, "auto", 256, 256),
    ("mimo", 4, "stream", 16, 5), ("mimo", 4, "stream", 256, 200), ("mimo", 4, "stream", 4096, 4096),
    ("mimo", 2, "stream", 1024, 1024), ("glm", 4, "stream", 4096, 3000), ("glm", 2, "stream", 256, 256),
    # TP6 of 2048: 3-block (384) slices, the stored width of every rank.
    ("mimo", 6, "decode", 1, 1), ("glm", 6, "decode", 16, 16), ("glmf", 6, "auto", 80, 80),
    ("mimo", 6, "stream", 1024, 1024), ("glm", 6, "stream", 4096, 4096),
]


@pytest.mark.parametrize("name,tp,route,capacity,rows", CASES)
def test_fp8_moe(name, tp, route, capacity, rows):
    g = _geometry(name, tp)
    c, worst = _run(g, route, capacity, rows)
    print(f"fp8_moe {name} tp{tp} {route} m{capacity} rows={rows}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


def test_fp8_moe_bf16_input():
    g = _geometry("mimo", 4)
    c, worst = _run(g, "auto", 80, 33, wire=False)
    print(f"fp8_moe mimo tp4 bf16 input rows=33: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


def test_fp8_moe_clamp():
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    g = Fp8MoeGeometry("clamped", hidden=4096, experts=256, top_k=8, intermediate=2048, tp=4, swiglu_limit=0.5)
    c, worst = _run(g, "auto", 16, 16)
    print(f"fp8_moe clamp 0.5 rows=16: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


def test_fp8_moe_stream_hot_experts():
    g = _geometry("mimo", 4)
    c, worst = _run(g, "stream", 1024, 1000, hot=6)
    print(f"fp8_moe mimo tp4 stream hot experts rows=1000: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


def test_fp8_moe_stream_clamp():
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    g = Fp8MoeGeometry("clamped", hidden=4096, experts=256, top_k=8, intermediate=2048, tp=4, swiglu_limit=0.5)
    c, worst = _run(g, "stream", 256, 256)
    print(f"fp8_moe clamp 0.5 stream rows=256: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


@pytest.mark.parametrize("route,capacity,rows", [("decode", 1, 1), ("decode", 80, 80), ("stream", 1024, 1024)])
def test_fp8_moe_tp6_zero_padding_is_exact(route, capacity, rows):
    """A TP6 rank owning two of its three stored 128-blocks (ranks 4 and 5 of
    2048) keeps zero gate/up rows and zero down columns in the third: its
    output equals the reference over the unpadded 256-wide slice."""
    g = _geometry("glm", 6)
    w1, s1, w3, s3, w2, s2 = (t.clone() for t in _weights(g))
    for w in (w1, w3):
        w[:, 256:].zero_()
    w2[:, :, 256:].zero_()
    padded = (w1, s1, w3, s3, w2, s2)
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    source, x_exact = wire_rows(x)
    ids = torch.rand(rows, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    from b12x.integration.cuteafd.fp8_moe import fp8_moe_scratch_bytes

    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity), dtype=torch.uint8, device="cuda")
    _program(g, route, capacity, True).launch(source, ids, weights, *padded, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    narrow = (w1[:, :256].contiguous(), s1[:, :2].contiguous(), w3[:, :256].contiguous(), s3[:, :2].contiguous(),
              w2[:, :, :256].contiguous(), s2[:, :, :2].contiguous())
    expected = reference(x_exact, ids, weights, *narrow, limit=g.swiglu_limit)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    print(f"fp8_moe glm tp6 padded {route} rows={rows}: cosine {c:.7f}")
    assert c >= COS
