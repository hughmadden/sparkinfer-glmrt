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
        kv = min(int(k[v]), k_max) if 0 < v < len(k) else 0
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
    program.launch(dx, vidx.cuda(), a.cuda(), b.cuda(), k.cuda(), scalars=[ld, rows, h, kk, 3])
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
    program.launch(full, vidx.cuda(), da, db, dk, scalars=[h, 300, h, kk, 3])
    lone = x[257:258].clone().cuda()
    program.launch(lone, vidx[:1].cuda(), da, db, dk, scalars=[h, 1, h, kk, 3])
    torch.cuda.synchronize()
    assert torch.equal(full[257].view(torch.int16), lone[0].view(torch.int16))


@pytest.mark.parametrize("form", ["f32", "bf16"])
@pytest.mark.parametrize("h,rows", [(4096, 5), (6144, 3), (8, 2)])
def test_add_rows_bit_exact(form, h, rows):
    require_b12x()
    from b12x.integration.cuteafd.steer_rows import compile_add_rows_aot
    gen = torch.Generator().manual_seed(h + rows)
    ld_y, ld_d = h + 8, h + 8
    y = (torch.rand(rows, ld_y, generator=gen) * 6 - 3).bfloat16()
    d = (torch.rand(rows, ld_d, generator=gen) - 0.5) * 0.1
    if form == "bf16":
        d = d.bfloat16()
    vidx = torch.tensor([1, 2, 0, -1, 3][:rows], dtype=torch.int32)
    on = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    want = y.clone()
    for r, v in enumerate(vidx.tolist()):
        if v > 0 and on[v]:
            want[r, :h] = (y[r, :h].float() + d[r, :h].float()).bfloat16()
    dy = y.cuda()
    compile_add_rows_aot(form).launch(dy, d.cuda(), vidx.cuda(), on.cuda(), scalars=[ld_y, ld_d, rows, h, 3])
    torch.cuda.synchronize()
    assert torch.equal(dy.cpu().view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("form,kk", [("one", 1), ("general", 1), ("general", 3), ("general", 32)])
@pytest.mark.parametrize("variants", [1, 3, 7])
@pytest.mark.parametrize("padded", [False, True])
def test_steer_rows_bounds_and_replay(form, kk, variants, padded):
    require_b12x()
    from b12x.integration.cuteafd.steer_rows import compile_steer_rows_aot
    h, ld = 128, 136
    masks = torch.tensor([0, -1, -(2**31), variants + 1, 63, 2**31 - 1, 1, variants], dtype=torch.int32)
    rows = len(masks)
    slots = 64 if padded else variants + 1
    gen = torch.Generator().manual_seed(91)
    x = torch.rand(rows, ld, generator=gen).bfloat16()
    # Payload-preserving copies expose accidental reads/rounding on base rows.
    x.view(torch.int16)[:6, :h] = 0x7f81
    a = torch.rand(slots, kk, h, generator=gen) * 0.125
    b = torch.rand(slots, kk, h, generator=gen) * 0.125
    k = torch.full((slots,), kk, dtype=torch.int32)
    k[0] = 2**31 - 1
    k[1] = 2**31 - 1  # clamp rank without changing the table stride
    dx, dv, da, db, dk = x.cuda(), masks.cuda(), a.cuda(), b.cuda(), k.cuda()
    program = compile_steer_rows_aot(form)
    scalars = [ld, rows, h, kk, variants]
    program.launch(dx, dv, da, db, dk, scalars=scalars)  # compile and warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(dx, dv, da, db, dk, scalars=scalars)
    # Avoid NaN arithmetic in active rows: original invalid rows retain payloads
    # only on the first replay, then all input rows are finite.
    for mask in (masks, masks.roll(2), torch.zeros_like(masks)):
        source = x.clone()
        if not torch.equal(mask, masks):
            source[:6] = 1
        dx.copy_(source)
        dv.copy_(mask)
        graph.replay()
        torch.cuda.synchronize()
        want = _reference(source, mask, a, b, k[:variants + 1], kk, h)
        assert torch.equal(dx.cpu().view(torch.int16), want.view(torch.int16))
    assert torch.equal(da.cpu(), a) and torch.equal(db.cpu(), b) and torch.equal(dk.cpu(), k)


@pytest.mark.parametrize("form", ["f32", "bf16"])
@pytest.mark.parametrize("variants", [1, 3, 7])
@pytest.mark.parametrize("padded", [False, True])
def test_add_rows_bounds_and_replay(form, variants, padded):
    require_b12x()
    from b12x.integration.cuteafd.steer_rows import compile_add_rows_aot
    h, ld = 128, 136
    masks = torch.tensor([0, -1, -(2**31), variants + 1, 63, 2**31 - 1, 1, variants], dtype=torch.int32)
    rows = len(masks)
    y = torch.ones(rows, ld, dtype=torch.bfloat16)
    y.view(torch.int16)[:6, :h] = 0x7f81
    d = torch.full((rows, ld), 0.25, dtype=torch.float32 if form == "f32" else torch.bfloat16)
    on = torch.ones(64 if padded else variants + 1, dtype=torch.int32)
    on[0] = 99
    if variants > 1:
        on[variants] = 0
    dy, dd, dv, do = y.cuda(), d.cuda(), masks.cuda(), on.cuda()
    program = compile_add_rows_aot(form)
    scalars = [ld, ld, rows, h, variants]
    program.launch(dy, dd, dv, do, scalars=scalars)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(dy, dd, dv, do, scalars=scalars)
    for mask in (masks, masks.roll(2), torch.zeros_like(masks)):
        source = y.clone()
        if not torch.equal(mask, masks):
            source[:6] = 1
        dy.copy_(source)
        dv.copy_(mask)
        graph.replay()
        torch.cuda.synchronize()
        want = source.clone()
        for r, v in enumerate(mask.tolist()):
            if 0 < v <= variants and on[v]:
                want[r, :h] = (source[r, :h].float() + d[r, :h].float()).bfloat16()
        assert torch.equal(dy.cpu().view(torch.int16), want.view(torch.int16))
    assert torch.equal(do.cpu(), on)
