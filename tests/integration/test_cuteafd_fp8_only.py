"""FP8-only weight kernels for the cuteafd coordinator programs (no BF16 copies).

* ``TmaFp8Gemm`` with per-row scales (``row`` / ``row_kmajor``) equals the BF16
  TMA GEMM over ``bf16(w * s)`` bitwise, as the block-scale form does.
* ``BatchedFp8Gemm`` (per-row x 64-K scales) equals ``BatchedBf16Gemm`` over
  the dequantized weights bitwise (GLM ``kv_b_proj`` absorb / up-projection).
* ``Fp8Projection``: decode rows equal ``RoutedFp8Projection`` with the BF16
  copy bitwise (GEMV up to 16 rows, W8A16 TMA above); prefill rows run W8A8
  (E4M3 activations per row and 128-K block, FP32 scales) against a float
  reference of the same quantization, or W8A16 bitwise with ``fp8_rows`` 0.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _program(key, make):
    if key not in _PROGRAMS:
        _PROGRAMS[key] = make()
    return _PROGRAMS[key]


def _weights(n, k, seed, row_scales=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    if row_scales:
        s = torch.rand((n, k // 128), generator=gen, device="cuda") * 0.01 + 1e-3
        grown = s.repeat_interleave(128, 1)
    else:
        s = torch.rand(((n + 127) // 128, k // 128), generator=gen, device="cuda") * 0.01 + 1e-3
        grown = s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w8, s, (w8.float() * grown).bfloat16()


@pytest.mark.parametrize("tile_n", [128, 64, 32])
@pytest.mark.parametrize("scales", ["block", "row", "row_kmajor"])
@pytest.mark.parametrize("n,k,rows", [(13568, 4096, 40), (2624, 6144, 300), (1000, 1024, 17)])
def test_tma_fp8_gemm_row_scales_bitwise(scales, n, k, rows, tile_n):
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import TmaFp8Gemm

    w8, s, wd = _weights(n, k, n + rows, row_scales=scales != "block")
    x = torch.randn((rows, k), device="cuda").bfloat16()
    outs = []
    for fp8 in (True, False):
        ops = [Operand("x", torch.bfloat16, "x"), Operand("w", torch.float8_e4m3fn if fp8 else torch.bfloat16, "w")]
        ops += [Operand("s", torch.float32, "s", align=4)] if fp8 else []
        ops.append(Operand("o", torch.bfloat16, "o", "out"))
        program = _program(("tma", scales, n, k, fp8, tile_n if fp8 else 0), lambda: compile_program(
            TmaFp8Gemm(n, k, scales=scales, tile_n=tile_n) if fp8 else TmaBf16Projection(n, k),
            name=f"t{int(fp8)}_{n}_{k}", operands=ops, scalars=(Scalar("rows"),),
            key=(n, k, fp8, scales, tile_n if fp8 else 0)))
        out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
        scale = s.t().contiguous() if scales == "row_kmajor" else s
        program.launch(x, *([w8, scale] if fp8 else [wd]), out, scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("kind", ["uk", "uv"])
@pytest.mark.parametrize("rows,warps", [(1, 4), (64, 4), (300, 8)])
def test_batched_fp8_gemm_bitwise(kind, rows, warps):
    """GLM 5.3 kv_b [64*448, 512] FP8 with its 128x128 grid, split per head."""
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._glm_kernels import BatchedBf16Gemm, BatchedFp8Gemm

    heads, d, v, c = 64, 192, 256, 512
    w8, s, wd = _weights(heads * (d + v), c, rows)
    grid = s.repeat_interleave(128, 0)  # [rows of kv_b, 4]
    k8 = w8.view(heads, d + v, c)
    kd = wd.view(heads, d + v, c)
    srow = grid[: heads * (d + v)].view(heads, d + v, 4)
    if kind == "uk":  # w_uk[h, c, d] = kv_b[h*448 + d, c]; scale per (h, c, d/64)
        n, k, a_row, a_batch, o_row, o_batch = c, d, heads * 256, 256, heads * 576, 576
        w_fp8 = k8[:, :d, :].transpose(1, 2).contiguous()
        w_bf16 = kd[:, :d, :].transpose(1, 2).contiguous()
        scale = srow[:, :d, :].repeat_interleave(128, 2)[:, :, :c]  # [h, d, c]
        scale = scale.transpose(1, 2)[:, :, ::64].contiguous()  # [h, c, d/64]
        a = torch.randn((rows, heads * 256), device="cuda").bfloat16()
        out_shape = (rows, heads * 576)
    else:  # w_uv[h, v, c] = kv_b[h*448 + 192 + v, c]; scale per (h, v, c/64)
        n, k, a_row, a_batch, o_row, o_batch = v, c, heads * c, c, heads * v, v
        w_fp8 = k8[:, d:, :].contiguous()
        w_bf16 = kd[:, d:, :].contiguous()
        scale = srow[:, d:, :].repeat_interleave(2, 2).contiguous()  # [h, v, c/64]
        a = torch.randn((rows, heads * c), device="cuda").bfloat16()
        out_shape = (rows, heads * v)
    geo = dict(n=n, k=k, batch=heads, a_row=a_row, a_batch=a_batch, o_row=o_row, o_batch=o_batch,
               compute_warps=warps)
    outs = []
    for fp8 in (True, False):
        ops = [Operand("a", torch.bfloat16, "a"), Operand("w", torch.float8_e4m3fn if fp8 else torch.bfloat16, "w")]
        ops += [Operand("s", torch.float32, "s", align=4)] if fp8 else []
        ops.append(Operand("o", torch.bfloat16, "o", "out"))
        program = _program(("batched", kind, warps, fp8), lambda: compile_program(
            (BatchedFp8Gemm if fp8 else BatchedBf16Gemm)(**geo), name=f"b{int(fp8)}_{kind}_{warps}",
            operands=ops, scalars=(Scalar("rows"),), key=(kind, warps, fp8)))
        out = torch.zeros(out_shape, dtype=torch.bfloat16, device="cuda")
        program.launch(a, *([w_fp8, scale] if fp8 else [w_bf16]), out, scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


def _w8a8_reference(x, w8, s, row_scales):
    """bf16(sum over K blocks of (q_x . q_w) * sx * sw) with x quantized per row and 128-K block."""
    rows, k = x.shape
    xb = x.float().view(rows, k // 128, 128)
    amax = xb.abs().amax(-1, keepdim=True)
    sx = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    qx = (xb / sx).to(torch.float8_e4m3fn).float() * sx
    n = w8.shape[0]
    grown = s.repeat_interleave(128, 1) if row_scales else s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return (qx.view(rows, k) @ (w8.float() * grown).t()).bfloat16()


@pytest.mark.parametrize("row_scales", [False, True])
@pytest.mark.parametrize("n,k", [(2624, 6144), (16384, 2048), (6144, 16384)])
@pytest.mark.parametrize("rows", [1, 7, 130, 4096])
def test_fp8_projection_prefill(row_scales, n, k, rows):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import Fp8Projection

    if n % 128 and not row_scales:
        pytest.skip("the block-FP8 GEMM takes 128x128 grids for N % 128 == 0 only (GLM qkv_a: row scales)")

    class Launch:
        def __init__(self):
            self.p = Fp8Projection(n, k, prefill_rows=4096, row_scales=row_scales)

        def key(self):
            return self.p.key()

        @__import__("cutlass").cute.jit
        def __call__(self, x, w, s, o, scratch, rows, fp8_rows, stream):
            from cutlass import Int64
            self.p(x, w, s, o, rows, fp8_rows, Int64(scratch.toint()), stream)

    program = _program(("proj_prefill", n, k, row_scales), lambda: compile_program(
        Launch(), name=f"pp_{n}_{k}_{int(row_scales)}",
        operands=(Operand("x", torch.bfloat16, "x"), Operand("w", torch.float8_e4m3fn, "w"),
                  Operand("s", torch.float32, "s", align=16), Operand("o", torch.bfloat16, "o", "out"),
                  Operand("scratch", torch.uint8, "scratch", "scratch")),
        scalars=(Scalar("rows"), Scalar("fp8_rows")), key=(n, k, row_scales),
        scratch={"scratch": lambda r: Launch().p.prefill_scratch_bytes(r)}))
    w8, s, wd = _weights(n, k, n + k + rows, row_scales=row_scales)
    x = (torch.randn((rows, k), device="cuda") * 2).bfloat16()
    scale = s.t().contiguous() if row_scales else s
    scratch = torch.empty(Launch().p.prefill_scratch_bytes(rows), dtype=torch.uint8, device="cuda")
    out8 = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    out16 = torch.empty_like(out8)
    program.launch(x, w8, scale, out8, scratch, scalars=(rows, 1))
    program.launch(x, w8, scale, out16, scratch, scalars=(rows, 0))
    torch.cuda.synchronize()
    ref8 = _w8a8_reference(x, w8, s, row_scales)
    ref16 = (x.float() @ wd.float().t())
    cos = torch.nn.functional.cosine_similarity(out8.float().flatten(), ref8.float().flatten(), dim=0).item()
    err = (out8.float() - ref8.float()).abs().max().item() / ref8.float().abs().max().item()
    print(f"W8A8 n={n} k={k} rows={rows} row_scales={row_scales}: cosine {cos:.7f} max rel err {err:.2e}")
    assert cos > 0.99998 and err < 2e-2
    cos16 = torch.nn.functional.cosine_similarity(out16.float().flatten(), ref16.flatten(), dim=0).item()
    assert cos16 > 0.99999


@pytest.mark.parametrize("rows", [1, 16, 17, 40, 64])
def test_fp8_projection_decode_matches_routed(rows):
    """Decode rows: GEMV up to 16 then W8A16 TMA, bitwise the BF16-copy program."""
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import Fp8Projection, RoutedFp8Projection

    n, k = 16384, 2048

    class New:
        def __init__(self):
            self.p = Fp8Projection(n, k)

        def key(self):
            return self.p.key()

        @__import__("cutlass").cute.jit
        def __call__(self, x, w, s, o, rows, stream):
            from cutlass import Int32, Int64
            self.p(x, w, s, o, rows, Int32(16), Int64(0), stream)

    new = _program(("dec_new",), lambda: compile_program(
        New(), name="dn", operands=(Operand("x", torch.bfloat16, "x"), Operand("w", torch.float8_e4m3fn, "w"),
                                    Operand("s", torch.float32, "s", align=4), Operand("o", torch.bfloat16, "o", "out")),
        scalars=(Scalar("rows"),), key=("dn",)))
    old = _program(("dec_old",), lambda: compile_program(
        RoutedFp8Projection(n, k), name="do",
        operands=(Operand("x", torch.bfloat16, "x"), Operand("wb", torch.bfloat16, "wb"),
                  Operand("w", torch.float8_e4m3fn, "w"), Operand("s", torch.float32, "s", align=4),
                  Operand("o", torch.bfloat16, "o", "out")),
        scalars=(Scalar("rows"),), key=("do",)))
    w8, s, wd = _weights(n, k, rows)
    x = torch.randn((rows, k), device="cuda").bfloat16()
    a = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    new.launch(x, w8, s, a, scalars=(rows,))
    old.launch(x, wd, w8, s, b, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(a, b)
