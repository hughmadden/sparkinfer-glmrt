"""Weight-variant row steering: bit-exact fixed order, base rows untouched, batch invariance."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import require_b12x

THREADS, LANES, VECTOR = 512, 32, 8


def _dot(b, x, h):
    """The kernel's fixed reduction order (see steer_rows module docstring)."""
    vectors = h // VECTOR
    partial = np.zeros(THREADS, np.float32)
    for t in range(THREADS):
        p = np.float32(0)
        for q in (t, t + THREADS):
            if q < vectors:
                for e in range(q * VECTOR, (q + 1) * VECTOR):
                    p = _fma(b[e], x[e], p)
        partial[t] = p
    total = np.float32(0)
    for w in range(THREADS // LANES):
        lanes = partial[w * LANES:(w + 1) * LANES].copy()
        for offset in (1, 2, 4, 8, 16):
            lanes = (lanes + lanes[np.arange(LANES) ^ offset]).astype(np.float32)
        total = np.float32(total + lanes[0])
    return total


def _fma(a, b, c):
    # Exact a*b in float64 (24+24 bits), one rounding of the sum to float32
    # matches fma.rn.f32 except in double-rounding corner cases; the products
    # here are far from those.
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def _reference(x, vidx, a, b, k, k_max, h):
    out = x.clone()
    xf = x.float().numpy()
    a, b = a.numpy(), b.numpy()
    for r, v in enumerate(vidx.tolist()):
        kv = min(int(k[v]), k_max) if v > 0 else 0
        if kv <= 0:
            continue
        row = xf[r, :h]
        t = [_dot(b[v, j], row, h) for j in range(kv)]
        s = np.zeros(h, np.float32)
        for j in range(kv):
            s = np.array([_fma(a[v, j, e], t[j], s[e]) for e in range(h)], np.float32)
        out[r, :h] = torch.from_numpy((row - s).astype(np.float32)).bfloat16()
    return out


def _case(rows, h, ld, kk, seed):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.rand(rows, ld, generator=gen) * 6 - 3).bfloat16()
    vidx = torch.randint(0, 4, (rows,), generator=gen, dtype=torch.int32)
    a = (torch.rand(4, kk, h, generator=gen) - 0.5)
    b = (torch.rand(4, kk, h, generator=gen) * 2 - 1) * (2 / h ** 0.5)
    k = torch.tensor([0, kk, 0, (kk + 1) // 2], dtype=torch.int32)
    return x, vidx, a, b, k


@pytest.mark.parametrize("form,kk", [("one", 1), ("general", 1), ("general", 3), ("general", 19)])
@pytest.mark.parametrize("h,rows", [(4096, 3), (5120, 2), (6144, 2), (384, 9)])
def test_steer_rows_bit_exact(form, kk, h, rows):
    require_b12x()
    from b12x.integration.cuteafd.steer_rows import compile_steer_rows_aot
    if form == "one" and kk != 1:
        pytest.skip("the one form is k_max == 1")
    ld = h + 8
    x, vidx, a, b, k = _case(rows, h, ld, kk, seed=h + kk + rows)
    vidx[0] = 1
    want = _reference(x, vidx, a, b, k, kk, h)
    dx = x.cuda()
    program = compile_steer_rows_aot(form)
    program.launch(dx, vidx.cuda(), a.cuda(), b.cuda(), k.cuda(), scalars=[ld, rows, h, kk])
    torch.cuda.synchronize()
    got = dx.cpu()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    base = (vidx == 0) | (vidx == 2)
    assert torch.equal(got[base].view(torch.int16), x[base].view(torch.int16))


def test_steer_rows_batch_invariant():
    require_b12x()
    from b12x.integration.cuteafd.steer_rows import compile_steer_rows_aot
    h, kk = 6144, 19
    x, vidx, a, b, k = _case(300, h, h, kk, seed=5)
    vidx[:] = 1
    program = compile_steer_rows_aot("general")
    da, db, dk = a.cuda(), b.cuda(), k.cuda()
    full = x.cuda()
    program.launch(full, vidx.cuda(), da, db, dk, scalars=[h, 300, h, kk])
    lone = x[257:258].clone().cuda()
    program.launch(lone, vidx[:1].cuda(), da, db, dk, scalars=[h, 1, h, kk])
    torch.cuda.synchronize()
    assert torch.equal(full[257].view(torch.int16), lone[0].view(torch.int16))
