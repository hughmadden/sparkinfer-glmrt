"""Native AOT GLM 5.x sparse MLA over FP8 656-byte latent records.

``compile_glm_sparse_mla_aot(g, route=..., max_rows=R)`` is the b12x
``attention.sparse_mla`` GLM_NSA path (``ModelType.GLM_NSA``, FP8 records
with inline FP32 group scales, 64 records per page): the absorbed 576-wide
query attends the selected latent records,
``out = softmax(q . k * 256^-0.5) . v`` with ``k`` the dequantized 576-wide
record and ``v`` its 512 latent dims; no sink. BF16 output, the 512-wide
latent per head (``glm_o`` applies W_UV).

Routes (both take the live row count as a launch scalar):

``route="decode"``  split-KV decode kernel (normalized BF16 partials + LSE)
    then the split merge. The split count is planned per live-row bucket
    (``rows == 1``, ``rows <= 8``, ``rows <= max_rows``) with the prepared
    plan's wave-balanced planner, and the program branches on ``rows``.
``route="prefill"`` the single-pass multi-head-group (MG) kernel (FP8 QK),
    one CTA per (row, 32-head group); scratch holds the base-2 LSE.

ABI (``N`` = 64 heads, ``K`` = 2048 selected slots per row)::

    q         bf16 [rows,N,576]     in   glm_producer query
    kv_cache  u8   [pages,41984]    in   latent records (64 x 656 bytes per page)
    indices   i32  [rows,K]         in   physical slots page*64+row; -1 = masked
    lengths   i32  [rows]           in   valid leading entries of each indices row
    out       bf16 [rows,N,512]     out
    scratch   u8   sparse_mla_scratch_bytes(...)
    rows      int32

Scratch (size with ``rows = max_rows``):
    prefill: lse f32 [rows,N]
    decode:  partials bf16 [rows,N,S,512] then (1024-aligned) lse f32
             [rows,N,S] for the live bucket's split count ``S``; sized for
             the largest ``rows * S`` over the buckets.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x.attention._shared.cute.ops import LOG2_E

from ._common import GLM53, GLMGeometry, Operand, Scalar, compile_program

__all__ = ["compile_glm_sparse_mla_aot", "decode_buckets"]

_DV = 512
_CAND = 64
_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _qk(g) -> int:
    """Absorbed query / record width: 576 (GLM 5.x, RoPE) or 512 (GLM 5.3 Flash, no RoPE)."""
    return int(g.latent_dim)


def _topk(g) -> int:
    """Selected slots per row (GLM 5.3 Flash pads index_topk + its open tail pool to 2112)."""
    return int(getattr(g, "sparse_topk", g.index_topk))


def _traits(g=GLM53):
    from b12x.attention._shared.mla.traits import ModelType, resolve_unplanned_traits

    if _qk(g) == 512:
        return resolve_unplanned_traits(512, torch.uint8, g.record_bytes, model_type=ModelType.GLM_NEXT)
    return resolve_unplanned_traits(_QK, torch.uint8, 656)


_QK = 576


def decode_buckets(g: GLMGeometry, max_rows: int) -> tuple[tuple[int, int, int], ...]:
    """``(rows_cap, num_splits, chunks_per_split)`` per live-row bucket."""
    from b12x.attention._shared.mla.kernel import plan_unified_decode_splits

    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    h_blocks = g.heads // 16
    max_chunks = -(-_topk(g) // _CAND)
    caps = sorted({c for c in (1, 8, int(max_rows)) if c <= int(max_rows)})
    out = []
    for cap in caps:
        _, splits, per_split = plan_unified_decode_splits(
            topk=_topk(g), max_chunks=max_chunks, num_tokens=cap, h_blocks=h_blocks, sm_count=sms)
        out.append((cap, int(splits), int(per_split)))
    return tuple(out)


def sparse_mla_scratch_bytes(g: GLMGeometry, *, route: str, rows: int, buckets=()) -> int:
    rows = max(int(rows), 1)
    if route == "prefill":
        return rows * g.heads * 4
    units = max(min(rows, cap) * splits for cap, splits, _ in buckets)
    return _align(units * g.heads * _DV * 2) + units * g.heads * 4


class _Prefill:
    def __init__(self, g: GLMGeometry):
        from dataclasses import replace

        from b12x.attention._shared.mla.prefill import _mg_head_partitions
        from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
        from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg
        from b12x.attention._shared.mla.traits import ComputeMode

        self.g = g
        self.heads, self.topk, self.qk = g.heads, _topk(g), _qk(g)
        traits = replace(_traits(g), compute_mode=ComputeMode.FP8)
        tiles = -(-self.topk // _CAND)
        self.kernels = []
        for mg_n_hg, active, offset in _mg_head_partitions(self.heads, int(traits.hpb)):
            layout = make_smem_layout_mg(traits, int(mg_n_hg))
            per_cta = int(layout.heads_per_cta)
            if active % per_cta == 0:
                valid_hpb, replicate = int(traits.hpb), active // per_cta
            else:
                valid_hpb, replicate = active, 1
            self.kernels.append(UnifiedPrefillMGKernel(
                traits, layout, g.page_rows, tiles, replicate_h=replicate, num_heads=self.heads,
                q_stride=(self.heads * self.qk, self.qk, 1), indices_stride0=self.topk,
                output_stride=(self.heads * _DV, _DV, 1), out_lse_stride=(self.heads, 1),
                has_sink=False, topk=self.topk, has_extra=False, pbs_extra=1, num_main_tiles=0,
                extra_topk=0, extra_indices_stride0=self.topk, row_xor=False, head_offset=offset,
                valid_hpb=valid_hpb, pack_hilo_rows=False,
            ))
        self.key = ("prefill", tiles, tuple(_mg_head_partitions(self.heads, int(traits.hpb))))

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, indices: cute.Pointer,
                 lengths: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        n = self.heads
        qt = cute.make_tensor(q, cute.make_layout((m, n, self.qk), stride=(n * self.qk, self.qk, 1)))
        kv = cute.make_tensor(kv_cache, cute.make_layout((1,)))
        idx = cute.make_tensor(indices, cute.make_layout((m, self.topk), stride=(self.topk, 1)))
        lens = cute.make_tensor(lengths, cute.make_layout((m,)))
        # has_sink=False elides every read of the sink operand.
        sink = cute.make_tensor(cute.make_ptr(Float32, Int64(lengths.toint()), cute.AddressSpace.gmem,
                                              assumed_align=4), cute.make_layout((1,)))
        ot = cute.make_tensor(out, cute.make_layout((m, n, _DV), stride=(n * _DV, _DV, 1)))
        lse = cute.make_tensor(
            cute.make_ptr(Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n), stride=(n, 1)))
        scale = Float32(float(self.g.softmax_scale) * LOG2_E)
        for k in cutlass.range_constexpr(len(self.kernels)):
            self.kernels[k](qt, kv, idx, lens, sink, ot, lse, scale, Float32(1.0),
                            Int64(self.g.kv_page_bytes), rows, stream)


class _Decode:
    def __init__(self, g: GLMGeometry, max_rows: int):
        from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
        from b12x.attention._shared.mla.merge import SparseMLASplitDecodeMergeKernel
        from b12x.attention._shared.mla.smem import make_smem_layout

        self.g = g
        self.heads, self.topk, self.qk = g.heads, _topk(g), _qk(g)
        traits = _traits(g)
        layout = make_smem_layout(traits)
        hpb = int(traits.hpb)
        if self.heads % hpb:
            raise ValueError("GLM decode needs heads divisible by 16")
        self.buckets = decode_buckets(g, max_rows)
        n = self.heads
        self.kernels, self.merges = [], []
        for _, splits, per_split in self.buckets:
            self.kernels.append(UnifiedDecodeKernel(
                traits, layout, g.page_rows, per_split, h_blocks=n // hpb, num_splits=splits,
                num_heads=n, q_head_dim=self.qk, topk=self.topk, extra_topk=0,
                q_stride=(n * self.qk, self.qk, 1), swa_indices_stride0=self.topk,
                extra_indices_stride0=self.topk,
                mid_out_stride=(n * splits * _DV, splits * _DV, _DV, 1),
                mid_lse_stride=(n * splits, splits, 1), has_extra=False, pbs_extra=1,
                valid_hpb=hpb, head_block_offset=0, per_token_len=True, native_glm_h8=False,
                native_dsv4_h8=False, native_dsv4_h16=False, native_dsv41_fp8=False, vector_q=True,
            ))
            self.merges.append(SparseMLASplitDecodeMergeKernel(static_num_chunks=splits))
        self.key = ("decode", self.buckets)

    @cute.jit
    def _run(self, b: cutlass.Constexpr, q: cute.Pointer, kv_cache: cute.Pointer,
             indices: cute.Pointer, lengths: cute.Pointer, out: cute.Pointer, base: Int64,
             rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        n = self.heads
        s = self.buckets[b][1]
        partials = cute.make_tensor(
            cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n, s, _DV), stride=(n * s * _DV, s * _DV, _DV, 1)))
        lse_off = (m * Int64(n * s * _DV * 2) + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        partial_lse = cute.make_tensor(
            cute.make_ptr(Float32, base + lse_off, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n, s), stride=(n * s, s, 1)))
        qt = cute.make_tensor(q, cute.make_layout((m, n, self.qk), stride=(n * self.qk, self.qk, 1)))
        kv = cute.make_tensor(kv_cache, cute.make_layout((1,)))
        idx = cute.make_tensor(indices, cute.make_layout((m, self.topk), stride=(self.topk, 1)))
        lens = cute.make_tensor(lengths, cute.make_layout((m,)))
        scale = Float32(float(self.g.softmax_scale) * LOG2_E)
        self.kernels[b].call_pertok(qt, kv, idx, partials, partial_lse, scale, Float32(1.0), lens,
                                    Int64(self.g.kv_page_bytes), rows, stream)
        # Static split count: the merge never reads its count operand.
        self.merges[b](partials, partial_lse, lens,
                       cute.make_tensor(out, cute.make_layout((m, n, _DV), stride=(n * _DV, _DV, 1))),
                       stream)

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, indices: cute.Pointer,
                 lengths: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        base = Int64(scratch.toint())
        if cutlass.const_expr(len(self.buckets) == 1):
            self._run(0, q, kv_cache, indices, lengths, out, base, rows, stream)
        elif cutlass.const_expr(len(self.buckets) == 2):
            if rows <= Int32(self.buckets[0][0]):
                self._run(0, q, kv_cache, indices, lengths, out, base, rows, stream)
            else:
                self._run(1, q, kv_cache, indices, lengths, out, base, rows, stream)
        else:
            if rows <= Int32(self.buckets[0][0]):
                self._run(0, q, kv_cache, indices, lengths, out, base, rows, stream)
            elif rows <= Int32(self.buckets[1][0]):
                self._run(1, q, kv_cache, indices, lengths, out, base, rows, stream)
            else:
                self._run(2, q, kv_cache, indices, lengths, out, base, rows, stream)


def compile_glm_sparse_mla_aot(g: GLMGeometry = GLM53, *, route: str = "prefill", max_rows: int = 1,
                               name: str = "glm_sparse_mla"):
    """GLM latent sparse MLA; see the module docstring for the ABI. A GLM 5.3
    Flash geometry (``GLMFGeometry``) selects the 512-wide query, 528-byte
    records (``ModelType.GLM_NEXT``) and its 2112-slot index rows."""
    if route == "prefill":
        launch = _Prefill(g)
        buckets = ()
    elif route == "decode":
        launch = _Decode(g, int(max_rows))
        buckets = launch.buckets
    else:
        raise ValueError("route must be 'prefill' or 'decode'")
    n, k = g.heads, _topk(g)
    operands = (
        Operand("q", torch.bfloat16, f"[rows,{n},{_qk(g)}]"),
        Operand("kv_cache", torch.uint8, f"[pages,{g.kv_page_bytes}]"),
        Operand("indices", torch.int32, f"[rows,{k}]", align=4),
        Operand("lengths", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{n},{_DV}]", "out"),
        Operand("scratch", torch.uint8, "[sparse_mla_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"{name}_{route}", operands=operands, scalars=(Scalar("rows"),),
        key=(n, k, int(max_rows), launch.key),
        geometry={"heads": n, "route": route, "max_rows": int(max_rows), "topk": k,
                  "softmax_scale": g.softmax_scale, "decode_buckets": [list(b) for b in buckets]},
        scratch={"scratch": lambda rows: sparse_mla_scratch_bytes(g, route=route, rows=rows, buckets=buckets)},
        doc=__doc__,
    )
