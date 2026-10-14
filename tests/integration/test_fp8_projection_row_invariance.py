"""Bitwise row invariance of every decode projection that switches GEMV and GEMM.

Drop-in location (this copy is not a host-suite test; ``pytest scripts/tests``
does not collect ``scripts/qualify``)::

    third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py

Needs an SM120 or SM121 GPU. Not run with the #210 write-up.

Contract, on the pinned fork (``91b104db``) and after the fix: row 0 of a
one-row launch equals row 0 of the same weights and the same row embedded in
a launch of M = 16, 17, 32 and 64. Comparison is the BF16 integer bits.
Other input rows are a finite poison, and the output buffer is filled with
the ``int16`` sentinel ``0x7FFF`` so an unwritten row cannot pass.

On this pin the contract is expected to fail where the live ``rows`` scalar
crosses the GEMV cap (do not xfail; a red row is the defect, a green row is
the fix):

* ``s16`` — GEMV through 16, ``TmaFp8Gemm`` above. M=16 should pass. M=17, 32
  and 64 should fail. This is the #173 KDA W8/S16 break, shared by every
  full-head GLM Flash projection, GLM 5.3 W8 decode, and Qwen W8 decode.
* ``s32`` — narrow GEMV through 16 and a wide GEMV through 32 with the same
  ``(warps, groups)``, TMA above 32. M=16, 17 and 32 should pass. M=64 should
  fail. Half-head KDA, the KDA output-row/shard projections, and MiMo W8.
* ``copy`` — the dual-copy switch (``RoutedFp8Projection`` or ``_Fp8Switch``):
  GEMV through 16, a BF16 TMA GEMM above. Same expected colours as ``s16``.
  Production FP8 checkpoints use the ``fp8_only`` path; these two shapes are
  the unmeasured equivalent, not a second copy of every BF16 projection.

The structural check does not freeze the 16 or 32 cap. A compiled GEMV must
use more than one warp, and a wide GEMV must keep that tile's
``(warps, groups)``. A same-order GEMV through 64 and TMA at every width
both satisfy the bit contract. The colours above are the prediction for
this pin.

This file passes ``fp8_rows`` from the projection's own cap
(``max_gemv_rows``, or ``max_fp8_rows`` for the switch). It does not execute
the Rust scalar, the baked ``Int32(FP8_GEMV_ROWS)``, or the output-row
wrappers. Those stay on TMA at M=32 until they move with the cap; a
kernel-only change goes green here while serving still switches.

Serving pads 17 up to a bucket (GLM Flash 32, GLM short-context 20, Qwen
speculative 24). The kernel still branches on the live ``rows`` scalar, so
the bucket above 16 is the other kernel. M=17 is the first integer past the
switch that #173 measured. The file stops at 64. The ``_m128`` program is a
separate compile and is not launched.
"""

from __future__ import annotations

import pytest

try:
    from ..conftest import require_b12x
except ImportError:  # collected from scripts/qualify, not the fork package
    def require_b12x():
        pytest.skip("drop-in for third_party/sparkinfer/tests/integration/; needs SM120/SM121")


_PROGRAMS: dict = {}
_WIDTHS = (16, 17, 32, 64)
_SENTINEL = 0x7FFF


# (id, tag, kind, N, K). kind selects the production constructor:
#   w8       glmf._w8, block scales, no wide GEMV
#   w8row    glmf._w8, per-row K-major scales, no wide GEMV
#   w8row32  glmf._w8, per-row K-major scales, wide_rows=32 (half-head o)
#   half_in  _KdaW8's replaced in-projection: (8, 2) on both GEMV tiles
#   rows     GlmfOutputRows.proj (wide 32, route cap 32)
#   shard    GlmfOutputShard.proj (wide 32, fp8_rows baked at 32)
#   plain    Fp8Projection defaults (GLM 5.3 and Qwen W8 decode)
#   mimo     mimo_w8 (per-row scales, wide_rows=32, same warps on both tiles)
#   routed   RoutedFp8Projection (BF16 TMA above 16)
#   switch   glmf._Fp8Switch decode (BF16 glm_projection above 16)
_CASES = (
    # GLM 5.3 Flash W8, full heads. fp8_rows 16. Certain at 17.
    ("glmf_kda_in", "s16", "w8row", 24896, 4096),
    ("glmf_kda_o", "s16", "w8row", 4096, 8192),
    ("glmf_mla_qkv_a", "s16", "w8", 2048, 4096),
    ("glmf_mla_q_b", "s16", "w8", 16384, 1536),
    ("glmf_mla_o", "s16", "w8", 4096, 16384),
    ("glmf_dense_gu", "s16", "w8", 24576, 4096),
    ("glmf_dense_down", "s16", "w8", 4096, 12288),
    ("glmf_shared_gu", "s16", "w8", 4096, 4096),
    ("glmf_shared_down", "s16", "w8", 4096, 2048),
    # Head-split (2× RTX). q_b / o / dense / shared are halved; fp8_rows stays 16.
    ("glmf_mla_q_b_half", "s16", "w8", 8192, 1536),
    ("glmf_mla_o_half", "s16", "w8", 4096, 8192),
    ("glmf_mla_output_rows", "s16", "w8", 4096, 16384),
    ("glmf_dense_gu_half", "s16", "w8", 12288, 4096),
    ("glmf_dense_down_half", "s16", "w8", 4096, 6144),
    ("glmf_shared_gu_half", "s16", "w8", 2048, 4096),
    ("glmf_shared_down_half", "s16", "w8", 4096, 1024),
    # Half-head KDA and the output projections: GEMV through 32, TMA above.
    ("glmf_kda_in_half", "s32", "half_in", 12576, 4096),
    ("glmf_kda_o_half", "s32", "w8row32", 4096, 4096),
    ("glmf_output_rows", "s32", "rows", 4096, 8192),
    ("glmf_output_shard", "s32", "shard", 2048, 8192),
    # GLM 5.3 W8 decode. Block scales, fp8_rows baked to 16. Short buckets include 20.
    ("glm_qkv_a", "s16", "plain", 2624, 6144),
    ("glm_q_b", "s16", "plain", 16384, 2048),
    ("glm_o", "s16", "plain", 6144, 16384),
    ("glm_index_wq", "s16", "plain", 4096, 2048),
    ("glm_dense_gu", "s16", "plain", 24576, 6144),
    ("glm_dense_down", "s16", "plain", 6144, 12288),
    ("glm_shared_gu", "s16", "plain", 4096, 6144),
    ("glm_shared_down", "s16", "plain", 6144, 2048),
    # Qwen 3.8 W8 decode. Plain buckets stop at 16; speculative buckets are 24/32/64.
    ("qwen_gdn_in", "s16", "plain", 16480, 2560),
    ("qwen_gdn_o", "s16", "plain", 2560, 6144),
    ("qwen_attn_in", "s16", "plain", 13952, 2560),
    ("qwen_attn_o", "s16", "plain", 2560, 6144),
    # MiMo V2 Flash and V2.6 Pro. fp8_rows 32, verify captures 1..=64.
    ("mimo_qkv_full", "s32", "mimo", 13568, 4096),
    ("mimo_qkv_swa", "s32", "mimo", 14848, 4096),
    ("mimo_o", "s32", "mimo", 4096, 8192),
    ("mimo_dense_gu", "s32", "mimo", 32768, 4096),
    ("mimo_dense_down", "s32", "mimo", 4096, 16384),
    ("mimo_pro_qkv", "s32", "mimo", 27648, 6144),
    ("mimo_pro_o", "s32", "mimo", 6144, 16384),
    ("mimo_pro_dense_gu", "s32", "mimo", 32768, 6144),
    ("mimo_pro_dense_down", "s32", "mimo", 6144, 16384),
    # Dual-copy equivalents. Large path is BF16 TMA, not TmaFp8Gemm.
    ("routed_glm_q_b", "copy", "routed", 16384, 2048),
    ("switch_glmf_kda_in", "copy", "switch", 24896, 4096),
)


def _ids():
    seen = set()
    for case_id, tag, kind, n, k in _CASES:
        assert case_id not in seen, case_id
        seen.add(case_id)
        assert tag in ("s16", "s32", "copy")
        assert n > 0 and k > 0 and k % 128 == 0
        # GEMV legality is checked against the built kernel. Reject shapes that
        # cannot be a 4-warp, 4-group tile either, so a typo fails at import.
        assert n % 8 == 0, (case_id, n)


_ids()


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _program(key, make):
    if key not in _PROGRAMS:
        _PROGRAMS[key] = make()
    return _PROGRAMS[key]


def _build(kind, n, k):
    """The production decode projection, not a retuned stand-in."""
    if kind == "w8":
        from b12x.integration.cuteafd.glmf import _w8
        return _w8(n, k, None)
    if kind == "w8row":
        from b12x.integration.cuteafd.glmf import _w8
        return _w8(n, k, None, row_scales=True)
    if kind == "w8row32":
        from b12x.integration.cuteafd.glmf import _w8
        return _w8(n, k, None, row_scales=True, wide_rows=32)
    if kind == "half_in":
        # _KdaW8 replaces _w8 for the half-head in-projection so both tiles
        # keep the full-head (8, 2) K split. glmf.py _KdaW8.__init__.
        from b12x.integration.cuteafd._fp8_weights import Fp8Projection
        return Fp8Projection(n, k, row_scales=True, kmajor=True, warps=8, groups=2,
                             wide_rows=32, wide_warps=8, wide_groups=2)
    if kind == "rows":
        from b12x.integration.cuteafd._glmf_output_rows import GlmfOutputRows
        return GlmfOutputRows(n, k, 64, False, False).proj
    if kind == "shard":
        from b12x.integration.cuteafd._glmf_output_shard import GlmfOutputShard
        return GlmfOutputShard(n, k, 64, False, False).proj
    if kind == "plain":
        from b12x.integration.cuteafd._fp8_weights import Fp8Projection
        return Fp8Projection(n, k)
    if kind == "mimo":
        from b12x.integration.cuteafd.mimo_attention import mimo_w8
        return mimo_w8(n, k, None)
    if kind == "routed":
        from b12x.integration.cuteafd._fp8_weights import RoutedFp8Projection
        return RoutedFp8Projection(n, k)
    if kind == "switch":
        from b12x.integration.cuteafd.glmf import _Fp8Switch
        return _Fp8Switch(n, k, fp8=True, row_scales=True)
    raise AssertionError(kind)


def _scale_kind(proj, kind):
    if kind == "routed":
        return "block"
    if kind == "switch":
        return "row"
    if not proj.row_scales:
        return "block"
    return "kmajor" if proj.kmajor else "row"


def _cap(proj, kind):
    """The projection's own cap, passed as ``fp8_rows``.

    On this pin that equals the baked decode scalar at the four widths.
    The test does not execute the Rust scalar or the output-row wrappers.
    """
    if kind == "routed":
        return None
    if kind == "switch":
        return int(proj.max_fp8_rows)
    return int(proj.max_gemv_rows)


def _check_shape(proj, kind):
    """A wide GEMV must split K the same way as the thin tile.

    The integer-bit compare is the contract. Today's 16/32 cap is not
    frozen, so a same-order GEMV through the verify range and a single
    TMA route both stay green. A mixed switch stays red.
    """
    if kind in ("routed", "switch"):
        gemv = getattr(proj, "fp8", None)
        wide = getattr(proj, "fp8_wide", None) if kind == "switch" else None
    else:
        gemv = getattr(proj, "gemv", None)
        wide = getattr(proj, "gemv_wide", None)
    if gemv is not None:
        assert gemv.warps > 1
    if wide is not None:
        assert gemv is not None
        assert (wide.warps, wide.groups) == (gemv.warps, gemv.groups)


def _compile(case_id, kind, n, k, proj):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    scale = _scale_kind(proj, kind)

    class Launch:
        def __init__(self):
            self.p = proj

        def key(self):
            return (kind, n, k, scale, self.p.key())

        @__import__("cutlass").cute.jit
        def __call__(self, x, w, s, o, rows, fp8_rows, stream):
            from cutlass import Int64
            self.p(x, w, s, o, rows, fp8_rows, Int64(0), stream)

    class Routed:
        def __init__(self):
            self.p = proj

        def key(self):
            return ("routed", n, k, self.p.key())

        @__import__("cutlass").cute.jit
        def __call__(self, x, wb, w, s, o, rows, stream):
            self.p(x, wb, w, s, o, rows, stream)

    class Switch:
        def __init__(self):
            self.p = proj

        def key(self):
            return ("switch", n, k, self.p.key())

        @__import__("cutlass").cute.jit
        def __call__(self, x, wb, w, s, o, rows, fp8_rows, stream):
            self.p(x, wb, w, s, o, rows, fp8_rows, stream)

    if kind == "routed":
        body, ops, scalars = Routed(), (
            Operand("x", __import__("torch").bfloat16, "x"),
            Operand("wb", __import__("torch").bfloat16, "wb"),
            Operand("w", __import__("torch").float8_e4m3fn, "w"),
            Operand("s", __import__("torch").float32, "s", align=16),
            Operand("o", __import__("torch").bfloat16, "o", "out"),
        ), (Scalar("rows"),)
    elif kind == "switch":
        torch = __import__("torch")
        body, ops, scalars = Switch(), (
            Operand("x", torch.bfloat16, "x"),
            Operand("wb", torch.bfloat16, "wb"),
            Operand("w", torch.float8_e4m3fn, "w"),
            Operand("s", torch.float32, "s", align=16),
            Operand("o", torch.bfloat16, "o", "out"),
        ), (Scalar("rows"), Scalar("fp8_rows"))
    else:
        torch = __import__("torch")
        body, ops, scalars = Launch(), (
            Operand("x", torch.bfloat16, "x"),
            Operand("w", torch.float8_e4m3fn, "w"),
            Operand("s", torch.float32, "s", align=16),
            Operand("o", torch.bfloat16, "o", "out"),
        ), (Scalar("rows"), Scalar("fp8_rows"))
    return _program(body.key(), lambda: compile_program(
        body, name=f"inv_{case_id}", operands=ops, scalars=scalars, key=body.key()))


def _weights(n, k, scale_kind, seed, dequant):
    """E4M3 weight and the scale layout the kernel reads.

    ``dequant`` builds the BF16 copy the dual-copy large path multiplies, so
    both of its routes are the same product. FP8-only cases never read it.
    """
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kb = k // 128
    if scale_kind == "block":
        scale = torch.rand(((n + 127) // 128, kb), generator=gen, device="cuda") * 0.01 + 1e-3
        grown = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1) if dequant else None
    else:
        rows = torch.rand((n, kb), generator=gen, device="cuda") * 0.01 + 1e-3
        grown = rows.repeat_interleave(128, 1) if dequant else None
        scale = rows.t().contiguous() if scale_kind == "kmajor" else rows
    bf16 = (w8.float() * grown).bfloat16() if dequant else None
    return w8, scale, bf16


def _rows(k, rows, seed):
    """Row 0 is stable across M. Every other row is a finite poison."""
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    row0 = torch.randn((k,), generator=gen, device="cuda").bfloat16()
    x = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
    if rows == 1:
        x[0] = row0
        return x, row0
    poison = torch.randn((rows, k), generator=torch.Generator(device="cuda").manual_seed(seed + 1),
                         device="cuda").bfloat16()
    poison.add_(1)
    x.copy_(poison)
    x[0] = row0
    return x, row0


def _launch(program, kind, x, w8, scale, bf16, out, rows, cap):
    if kind == "routed":
        program.launch(x, bf16, w8, scale, out, scalars=(rows,))
    elif kind == "switch":
        program.launch(x, bf16, w8, scale, out, scalars=(rows, cap))
    else:
        program.launch(x, w8, scale, out, scalars=(rows, cap))


def _fresh(rows, n):
    import torch
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    out.view(torch.int16).fill_(_SENTINEL)
    return out


@pytest.mark.parametrize("rows", _WIDTHS)
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_fp8_projection_row_invariance(case, rows):
    """Row 0 at M=1 matches the same row embedded at ``rows``, as integer bits."""
    import torch
    case_id, tag, kind, n, k = case
    proj = _build(kind, n, k)
    _check_shape(proj, kind)
    program = _compile(case_id, kind, n, k, proj)
    scale_kind = _scale_kind(proj, kind)
    cap = _cap(proj, kind)
    seed = 20_000 + (n * 131 + k * 17 + sum(ord(ch) for ch in case_id)) % 50_000
    w8, scale, bf16 = _weights(n, k, scale_kind, seed, dequant=kind in ("routed", "switch"))
    lone, row0 = _rows(k, 1, seed + 3)
    wide, _ = _rows(k, rows, seed + 3)
    assert torch.equal(wide[0], row0)
    if rows > 1:
        assert not torch.equal(wide[1], wide[0])
    one = _fresh(1, n)
    many = _fresh(rows, n)
    _launch(program, kind, lone, w8, scale, bf16, one, 1, cap)
    _launch(program, kind, wide, w8, scale, bf16, many, rows, cap)
    torch.cuda.synchronize()
    bits_one = one.view(torch.int16)[0]
    bits_row0 = many.view(torch.int16)[0]
    assert (bits_one != _SENTINEL).all(), f"{case_id}: M=1 left the sentinel in place"
    assert (bits_row0 != _SENTINEL).all(), f"{case_id}: M={rows} row 0 left the sentinel in place"
    if rows > 1:
        written = (many.view(torch.int16)[1:] != _SENTINEL).any(dim=1)
        assert bool(written.all()), f"{case_id}: M={rows} left a poison row unwritten"
    assert torch.equal(bits_one, bits_row0), (
        f"{case_id} [{tag}] M=1 vs embedded M={rows}: row 0 BF16 bits differ "
        f"({int((bits_one != bits_row0).sum())} of {n} columns)"
    )
