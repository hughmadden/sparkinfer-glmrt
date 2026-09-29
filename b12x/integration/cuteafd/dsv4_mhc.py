"""Native AOT mHC programs for DeepSeek V4 (non-lagged hyper-connections).

Four programs, all with ``rows`` as a live launch argument (``H`` = hidden,
``S`` = ``geometry.mhc_split_k`` = 2H/128, i.e. 64 for Flash, 112 for Pro):

``compile_dsv4_mhc_pre_aot`` (``b12x.norm.mhc.run_pre`` with an expanded
residual, fused attn/ffn RMSNorm)
    residual  bf16 [rows,4,H]  in   (the returned ``residual_out`` of the
                                     Python API is a bitwise copy; pass this
                                     same buffer to post/post_pre)
    fn        f32  [24,4H]     in   hc_*_fn
    scale     f32  [3]         in   hc_*_scale
    base      f32  [24]        in   hc_*_base
    norm      bf16 [H]         in   attn_norm / ffn_norm weight
    post      f32  [rows,4]    out
    comb      f32  [rows,4,4]  out
    y         bf16 [rows,H]    out  normalized layer input
    scratch   f32  [rows,S,25] scratch   bytes = rows * S * 25 * 4

``compile_dsv4_mhc_post_pre_aot`` (``run_post_pre``: stream' = post*x +
comb^T residual, then pre on stream')
    x         bf16 [rows,H]    in   layer delta (attention or FFN output)
    residual  bf16 [rows,4,H]  in   previous stream
    prev_post f32  [rows,4]    in
    prev_comb f32  [rows,4,4]  in
    fn, scale, base, norm      in   as for pre (next hc/norm weights)
    residual_out bf16 [rows,4,H] out  new stream (feed to the next post/post_pre)
    post, comb, y              out  as for pre
    scratch   f32  [rows,S,25] scratch   bytes = rows * S * 25 * 4

``compile_dsv4_mhc_post_aot`` (``run_post``)
    x bf16 [rows,H], residual bf16 [rows,4,H], prev_post f32 [rows,4],
    prev_comb f32 [rows,4,4] in; out bf16 [rows,4,H] out. No scratch.

``compile_dsv4_mhc_head_aot`` (``run_head``: sigmoid hc_head collapse +
final RMSNorm)
    residual bf16 [rows,4,H], fn f32 [4,4H] (hc_head_fn), scale f32 [1],
    base f32 [4], norm bf16 [H] in; collapsed bf16 [rows,H] out (written only
    when compiled with ``store_collapsed=True``, otherwise ignored and may be
    any address); out bf16 [rows,H] out. No scratch.

Epsilons (rms/hc/norm) and Sinkhorn iterations come from the geometry
(V4: 1e-6/1e-6/1e-6, 20). All outputs, scratch and inputs must be disjoint.
``post_pre`` compiles the same kernels the prepared plan selects for the
declared ``max_rows`` capacity on SM120 (decode route below 96 rows, block-M
route otherwise); the prepared TF32 tensor-core route (>= 384 rows) is not
exported, the block-M route is used there (FP32-equivalent, not bitwise).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from b12x.norm.mhc._head_cute import MhcHeadKernel
from b12x.norm.mhc._kernels import (
    MHCFinalizeGramKernel,
    MHCPostPrePartialKernel,
    MHCPostPrePrefillBlockMPartialKernel,
    MHCPostPrePrefillGramKernel,
    MHCPrefillTf32ProjectTmaKernel,
    _PREFILL_BLOCK_M,
    _PREFILL_BLOCK_TILE_N,
)

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program

__all__ = [
    "compile_dsv4_mhc_head_aot",
    "compile_dsv4_mhc_post_aot",
    "compile_dsv4_mhc_post_pre_aot",
    "compile_dsv4_mhc_pre_aot",
    "mhc_post_pre_route",
    "mhc_scratch_bytes",
]

_PARTIALS = 25
_MIXES = 24
_PREFILL_MIN_ROWS = 96  # B12X_MHC_PREFILL_MIN_TOKENS default


def mhc_scratch_bytes(geometry: DSV4Geometry, rows: int) -> int:
    """FP32 partials [rows, 2H/128, 25] used by pre and post_pre."""
    return int(rows) * geometry.mhc_split_k * _PARTIALS * 4


def mhc_post_pre_route(max_rows: int) -> str:
    """Route the prepared SM120 plan uses at this capacity (TF32 -> block_m)."""
    return "decode" if int(max_rows) < _PREFILL_MIN_ROWS else "block_m"


def _finalize(geometry: DSV4Geometry, *, compact: bool, projection_splits: int = 1) -> MHCFinalizeGramKernel:
    return MHCFinalizeGramKernel(
        hidden_size=geometry.hidden, split_k=geometry.mhc_split_k,
        rms_eps=geometry.norm_eps, hc_eps=geometry.hc_eps,
        sinkhorn_iters=geometry.hc_sinkhorn_iters, norm_eps=geometry.norm_eps,
        fuse_norm=True, compact_partials=compact, compact_projection_splits=projection_splits,
    )


# The "tf32" post_pre route (prefill capacities): the post and Gram rows on
# CUDA cores, the 24 fn mixes on TF32 tensor cores with fn split into two
# TF32 terms (the BF16 stream values are exact in TF32), K split 8 ways.
# Tiles of the torch route's 4096-hidden, >= 3584-token configuration. Below
# _TF32_MIN_ROWS live rows the program runs the block_m route instead.
_TF32_PROJECTION = dict(tile_m=192, tile_n=24, tile_k=64, num_stages=2, num_m_warps=12, num_n_warps=1,
                        k_splits=8)
_TF32_MIN_ROWS = 384


class _MhcPre:
    def __init__(self, geometry: DSV4Geometry, partials_per_cta: int):
        self.h, self.s = geometry.hidden, geometry.mhc_split_k
        self.partial = MHCPostPrePartialKernel(
            hidden_size=self.h, split_k=self.s, compute_gram=True, pre_only=True,
            materialize_pre=False, partials_per_cta=partials_per_cta,
        )
        self.finalize = _finalize(geometry, compact=False)

    @cute.jit
    def __call__(self, residual: cute.Pointer, fn: cute.Pointer, scale: cute.Pointer,
                 base: cute.Pointer, norm: cute.Pointer, post: cute.Pointer,
                 comb: cute.Pointer, y: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        h, s = self.h, self.s
        m = Int64(rows)
        r = cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        w = cute.make_tensor(fn, cute.make_layout((_MIXES, 4 * h), stride=(4 * h, 1)))
        p = cute.make_tensor(scratch, cute.make_layout((m, s, _PARTIALS), stride=(s * _PARTIALS, _PARTIALS, 1)))
        out_y = cute.make_tensor(y, cute.make_layout((m, h), stride=(h, 1)))
        sc = cute.make_tensor(scale, cute.make_layout((3,)))
        bs = cute.make_tensor(base, cute.make_layout((_MIXES,)))
        # materialize_pre=False: the partial kernel never writes ``out``; the
        # finalizer collapses the unmodified residual directly.
        self.partial(r, r, p, p, w, p, r, p, out_y, rows, stream)
        self.finalize(
            r, p, sc, bs, out_y,
            cute.make_tensor(post, cute.make_layout((m, 4), stride=(4, 1))),
            cute.make_tensor(comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1))),
            cute.make_tensor(norm, cute.make_layout((h,))),
            sc, bs, rows, stream,
        )


class _MhcPostPre:
    def __init__(self, geometry: DSV4Geometry, route: str, partials_per_cta: int, block_m: int | None = None,
                 tile_n: int | None = None):
        self.h, self.s = geometry.hidden, geometry.mhc_split_k
        self.route = route
        if route == "decode":
            self.partial = MHCPostPrePartialKernel(
                hidden_size=self.h, split_k=self.s, compute_gram=True,
                partials_per_cta=partials_per_cta,
            )
        elif route == "tf32":
            self.gram = MHCPostPrePrefillGramKernel(hidden_size=self.h, split_k=self.s)
            self.project = MHCPrefillTf32ProjectTmaKernel(hidden_size=self.h, split_k=self.s, split_fp32_fn=True,
                                                          **_TF32_PROJECTION)
            self.tf32_finalize = _finalize(geometry, compact=True, projection_splits=_TF32_PROJECTION["k_splits"])
            self.partial = MHCPostPrePrefillBlockMPartialKernel(
                hidden_size=self.h, split_k=self.s, block_m=_PREFILL_BLOCK_M,
                tile_n=12 if self.h == 7168 else _PREFILL_BLOCK_TILE_N, compute_gram=True,
            )
        elif route == "block_m":
            tuned = block_m is not None or tile_n is not None
            self.partial = MHCPostPrePrefillBlockMPartialKernel(
                hidden_size=self.h, split_k=self.s, block_m=_PREFILL_BLOCK_M if block_m is None else block_m,
                tile_n=tile_n if tile_n is not None else 12 if self.h == 7168 else _PREFILL_BLOCK_TILE_N,
                compute_gram=True, n_fastest=tuned,
            )
        else:
            raise ValueError(f"unknown mHC post_pre route {route!r}")
        self.finalize = _finalize(geometry, compact=route != "decode")

    @cute.jit
    def __call__(self, x: cute.Pointer, residual: cute.Pointer, prev_post: cute.Pointer,
                 prev_comb: cute.Pointer, fn: cute.Pointer, scale: cute.Pointer,
                 base: cute.Pointer, norm: cute.Pointer, residual_out: cute.Pointer,
                 post: cute.Pointer, comb: cute.Pointer, y: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        h, s = self.h, self.s
        m = Int64(rows)
        xt = cute.make_tensor(x, cute.make_layout((m, h), stride=(h, 1)))
        r = cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        pp = cute.make_tensor(prev_post, cute.make_layout((m, 4), stride=(4, 1)))
        pc = cute.make_tensor(prev_comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1)))
        w = cute.make_tensor(fn, cute.make_layout((_MIXES, 4 * h), stride=(4 * h, 1)))
        p = cute.make_tensor(scratch, cute.make_layout((m, s, _PARTIALS), stride=(s * _PARTIALS, _PARTIALS, 1)))
        out = cute.make_tensor(residual_out, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        out_y = cute.make_tensor(y, cute.make_layout((m, h), stride=(h, 1)))
        sc = cute.make_tensor(scale, cute.make_layout((3,)))
        bs = cute.make_tensor(base, cute.make_layout((_MIXES,)))
        if const_expr(self.route == "decode"):
            self.partial(xt, r, pp, pc, w, p, out, pp, out, rows, stream)
        elif const_expr(self.route == "tf32"):
            if rows >= Int32(_TF32_MIN_ROWS):
                self.gram(xt, r, pp, pc, p, out, rows, stream)
                self.project(cute.make_tensor(residual_out, cute.make_layout((m, 4 * h), stride=(4 * h, 1))), w, p,
                             rows, stream)
                self.tf32_finalize(
                    out, p, sc, bs, out_y,
                    cute.make_tensor(post, cute.make_layout((m, 4), stride=(4, 1))),
                    cute.make_tensor(comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1))),
                    cute.make_tensor(norm, cute.make_layout((h,))),
                    sc, bs, rows, stream,
                )
            else:
                self.partial(xt, r, pp, pc, w, p, out, rows, stream)
        else:
            self.partial(xt, r, pp, pc, w, p, out, rows, stream)
        if const_expr(self.route == "tf32"):
            if rows < Int32(_TF32_MIN_ROWS):
                self.finalize(
                    out, p, sc, bs, out_y,
                    cute.make_tensor(post, cute.make_layout((m, 4), stride=(4, 1))),
                    cute.make_tensor(comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1))),
                    cute.make_tensor(norm, cute.make_layout((h,))),
                    sc, bs, rows, stream,
                )
            return
        self.finalize(
            out, p, sc, bs, out_y,
            cute.make_tensor(post, cute.make_layout((m, 4), stride=(4, 1))),
            cute.make_tensor(comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1))),
            cute.make_tensor(norm, cute.make_layout((h,))),
            sc, bs, rows, stream,
        )


class _MhcPost:
    def __init__(self, geometry: DSV4Geometry):
        self.h, self.s = geometry.hidden, geometry.mhc_split_k
        self.post = MHCPostPrePartialKernel(hidden_size=self.h, split_k=self.s, post_only=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, residual: cute.Pointer, prev_post: cute.Pointer,
                 prev_comb: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        h = self.h
        m = Int64(rows)
        xt = cute.make_tensor(x, cute.make_layout((m, h), stride=(h, 1)))
        r = cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        pp = cute.make_tensor(prev_post, cute.make_layout((m, 4), stride=(4, 1)))
        pc = cute.make_tensor(prev_comb, cute.make_layout((m, 4, 4), stride=(16, 4, 1)))
        o = cute.make_tensor(out, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        # Post-only ignores fn/partials; pass typed placeholders as the
        # prepared launcher does.
        self.post(xt, r, pp, pc, pc, pp, o, pp, o, rows, stream)


def _stream_ops(geometry: DSV4Geometry):
    h = geometry.hidden
    return {
        "fn": Operand("fn", torch.float32, f"[24,{4 * h}]"),
        "scale": Operand("scale", torch.float32, "[3]", align=4),
        "base": Operand("base", torch.float32, "[24]", align=4),
        "norm": Operand("norm", torch.bfloat16, f"[{h}]"),
        "post": Operand("post", torch.float32, "[rows,4]", "out"),
        "comb": Operand("comb", torch.float32, "[rows,4,4]", "out"),
        "y": Operand("y", torch.bfloat16, f"[rows,{h}]", "out"),
        "scratch": Operand("scratch", torch.float32, f"[rows,{geometry.mhc_split_k},25]", "scratch"),
    }


def _geometry_key(geometry: DSV4Geometry) -> tuple:
    return (geometry.hidden, geometry.mhc_split_k, geometry.norm_eps, geometry.hc_eps,
            geometry.hc_sinkhorn_iters)


def compile_dsv4_mhc_pre_aot(geometry: DSV4Geometry = FLASH, *, partials_per_cta: int = 4):
    """mHC pre with fused RMSNorm; see module docstring for the ABI."""
    ops = _stream_ops(geometry)
    h = geometry.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        ops["fn"], ops["scale"], ops["base"], ops["norm"],
        ops["post"], ops["comb"], ops["y"], ops["scratch"],
    )
    return compile_program(
        _MhcPre(geometry, partials_per_cta), name="dsv4_mhc_pre", operands=operands,
        scalars=(Scalar("rows"),), key=_geometry_key(geometry) + (partials_per_cta,),
        geometry={"hidden": h, "split_k": geometry.mhc_split_k, "partials_per_cta": partials_per_cta,
                  "eps": geometry.norm_eps},
        scratch={"scratch": lambda rows: mhc_scratch_bytes(geometry, rows)},
        doc=__doc__,
    )


def compile_dsv4_mhc_post_pre_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int = 1,
                                  route: str | None = None, partials_per_cta: int = 4,
                                  block_m: int | None = None, tile_n: int | None = None):
    """Fused mHC post + next pre; ``max_rows`` selects the prepared route (``route="tf32"``:
    TF32 tensor-core mixes with split FP32 fn from 384 live rows, the block_m route below). ``block_m`` /
    ``tile_n`` retile the block_m route (tokens and mixes per CTA; the mix tiles of a
    token block then launch adjacently) without changing its arithmetic."""
    route = mhc_post_pre_route(max_rows) if route is None else route
    ops = _stream_ops(geometry)
    h = geometry.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("prev_comb", torch.float32, "[rows,4,4]", align=4),
        ops["fn"], ops["scale"], ops["base"], ops["norm"],
        Operand("residual_out", torch.bfloat16, f"[rows,4,{h}]", "out"),
        ops["post"], ops["comb"], ops["y"], ops["scratch"],
    )
    return compile_program(
        _MhcPostPre(geometry, route, partials_per_cta, block_m, tile_n), name=f"dsv4_mhc_post_pre_{route}",
        operands=operands, scalars=(Scalar("rows"),),
        key=_geometry_key(geometry) + (route, partials_per_cta)
        + (() if block_m is None and tile_n is None else (block_m, tile_n)),
        geometry={"hidden": h, "split_k": geometry.mhc_split_k, "route": route,
                  "eps": geometry.norm_eps},
        scratch={"scratch": lambda rows: mhc_scratch_bytes(geometry, rows)},
        doc=__doc__,
    )


def compile_dsv4_mhc_post_aot(geometry: DSV4Geometry = FLASH):
    """mHC post (mix back): out = post * x + comb^T residual."""
    h = geometry.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("prev_comb", torch.float32, "[rows,4,4]", align=4),
        Operand("out", torch.bfloat16, f"[rows,4,{h}]", "out"),
    )
    return compile_program(
        _MhcPost(geometry), name="dsv4_mhc_post", operands=operands,
        scalars=(Scalar("rows"),), key=(h,), geometry={"hidden": h}, doc=__doc__,
    )


def compile_dsv4_mhc_head_aot(geometry: DSV4Geometry = FLASH, *, store_collapsed: bool = False):
    """Terminal hc_head collapse + final RMSNorm (CuTe port of run_head)."""
    h = geometry.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("fn", torch.float32, f"[4,{4 * h}]"),
        Operand("scale", torch.float32, "[1]", align=4),
        Operand("base", torch.float32, "[4]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("collapsed", torch.bfloat16, f"[rows,{h}]", "out",
                note="written only when compiled with store_collapsed=True"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    kernel = MhcHeadKernel(hidden_size=h, rms_eps=geometry.norm_eps, hc_eps=geometry.hc_eps,
                           norm_eps=geometry.norm_eps, store_collapsed=store_collapsed)
    return compile_program(
        kernel, name="dsv4_mhc_head", operands=operands, scalars=(Scalar("rows"),),
        key=(h, geometry.norm_eps, geometry.hc_eps, store_collapsed),
        geometry={"hidden": h, "eps": geometry.norm_eps, "store_collapsed": store_collapsed},
        doc=__doc__,
    )
