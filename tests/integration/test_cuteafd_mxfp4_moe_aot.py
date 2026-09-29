"""cuteafd MXFP4 routed-expert programs (``fp8_moe`` with ``weights="mxfp4"``) vs torch.

Random MXFP4 experts (packed E2M1 bytes, UE8M0 per-32 scales) at the MiMo
V2.6 Pro geometry (H 6144, 384 experts, top-8): full width (TP1, the RTX
local layers), the TP2 slice (1024) and the zero-padded TP6 slice (352 or 320
real rows stored 384 wide); FP8 K32 wire or BF16 input rows. The reference is
the checkpoint semantics: ``F.linear`` of BF16 activations with the exactly
widened ``bf16(e2m1 * 2^(s-127))`` weights, BF16 gate/up, ``bf16(silu(g)) *
u``, BF16 down output, FP32 weighted route sum.
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
    """U8 ``[..., N, K/2]`` + U8 ``[..., N, K/32]`` -> BF16 ``[..., N, K]``."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).flatten(-2).long()
    power = torch.exp2(scale.float() - 127.0).repeat_interleave(32, -1)
    return (table[codes] * power).bfloat16()


def _geometry(tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["mimop"].with_tp(tp)


def _weights(g, real: int):
    key = (g.tp, real)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        gen = torch.Generator(device="cuda").manual_seed(0)
        h, i, e = g.hidden, g.slice, g.experts

        def codes(shape):
            return torch.randint(0, 256, shape, device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)

        def scales(shape):
            return torch.randint(118, 124, shape, device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)

        w1, s1, w3, s3 = codes((e, i, h // 2)), scales((e, i, h // 32)), codes((e, i, h // 2)), scales((e, i, h // 32))
        w2, s2 = codes((e, h, i // 2)), scales((e, h, i // 32))
        # Zero padding past the rank's real rows, as the loader stores it.
        for w in (w1, s1, w3, s3):
            w[:, real:] = 0
        w2[:, :, real // 2:] = 0
        s2[:, :, real // 32:] = 0
        _WEIGHTS[key] = (w1, s1, w3, s3, w2, s2)
    return _WEIGHTS[key]


def reference(x, ids, weights, w1, s1, w3, s3, w2, s2):
    out = torch.zeros(x.shape, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = x[rows] @ dequant(w1[expert], s1[expert]).T
        up = x[rows] @ dequant(w3[expert], s3[expert]).T
        y = (torch.nn.functional.silu(gate) * up) @ dequant(w2[expert], s2[expert]).T
        out.index_add_(0, rows, y.float() * weights[rows, slots][:, None])
    return out.bfloat16()


def _run(g, real, capacity, rows, wire=True, seed=1, hot=0, route="auto"):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = _weights(g, real)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
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
    expected = reference(x_exact, ids, weights, *w)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    return c, worst


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (tp, real rows of the slice, capacity, rows, wire, hot)
    (6, 352, 1, 1, True, 0), (6, 352, 16, 16, True, 0), (6, 320, 80, 80, True, 0),
    (6, 352, 256, 256, True, 8), (6, 352, 1024, 1000, True, 0),
    (2, 1024, 16, 7, True, 0), (2, 1024, 256, 256, True, 0),
    (1, 2048, 1, 1, False, 0), (1, 2048, 80, 80, False, 0), (1, 2048, 1024, 700, False, 0),
]


@pytest.mark.parametrize("tp,real,capacity,rows,wire,hot", CASES)
def test_mxfp4_moe(tp, real, capacity, rows, wire, hot):
    g = _geometry(tp)
    c, worst = _run(g, real, capacity, rows, wire=wire, hot=hot)
    print(f"mxfp4_moe mimop tp{tp} (slice {g.slice}, {real} real) m{capacity} rows={rows} "
          f"{'wire' if wire else 'bf16'}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


STREAM_CASES = [
    # (tp, real rows, capacity, rows, hot): the streaming route (wire input), forced and via auto.
    (6, 352, 4096, 1024, 0, "stream"), (6, 320, 4096, 4096, 0, "stream"), (6, 352, 256, 200, 8, "stream"),
    (6, 352, 4096, 4096, 0, "auto"), (2, 1024, 4096, 4096, 0, "stream"),
    (1, 2048, 1024, 1024, 0, "stream"), (1, 2048, 4096, 4096, 0, "stream"), (1, 2048, 4096, 3000, 0, "auto"),
]


@pytest.mark.parametrize("tp,real,capacity,rows,hot,route", STREAM_CASES)
def test_mxfp4_moe_stream(tp, real, capacity, rows, hot, route):
    g = _geometry(tp)
    c, worst = _run(g, real, capacity, rows, wire=True, hot=hot, route=route)
    print(f"mxfp4_moe mimop tp{tp} (slice {g.slice}, {real} real) {route} m{capacity} rows={rows}: "
          f"cosine {c:.7f} worst row {worst:.6f}")
    assert c >= 0.99999 and worst >= 0.9999


def test_mxfp4_geometry():
    assert [_geometry(tp).slice for tp in (1, 2, 4, 6)] == [2048, 1024, 512, 384]
