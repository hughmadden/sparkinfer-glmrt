"""Native AOT compressed sparse MLA for DeepSeek V4 FP8 584-byte records.

``compile_dsv4_sparse_mla_aot(geometry, *, route, max_rows, indexed_width,
indexed_page_rows)`` returns one program equal to
``b12x.attention.compressed_sparse_mla`` (``cache_format="deepseek_v4"``) as
the prototype runs it: softmax(q.k * 512^-0.5 ; per-head sink) . v over the
union of the window slots (main cache, 256-token pages, width 128) and an
optional indexed compressed cache (C4: 64 rows/page, width = index_topk; C128:
2 rows/page, width given at compile), BF16 output, not de-rotated.

Routes (both take the live row count as a launch scalar):

``route="prefill"``  the prepared plan's ``mode="extend"`` route (what the
    prototype prefill uses): the single-pass multi-head-group kernel, one CTA
    per (row, 32-head group). Scratch holds the base-2 LSE only.
``route="decode"``   the prepared ``mode="decode"`` SM120 route: split-KV
    decode kernel writing normalized partials + LSE, then the sink merge. The
    split count is planned from ``max_rows`` exactly as the prepared plan of
    that capacity does (capacity is a compile input; rows <= max_rows).

ABI (``N`` heads, ``W`` indexed width, ``B`` indexed page bytes = 37440 for
C4 / 1728 for C128; all index values are physical slots, ``-1`` = masked;
positions past a row's length are masked)::

    q               bf16 [rows,N,512]      in   produced query (RoPE applied)
    swa_cache       u8   [pages,149760]    in   main/window FP8 cache
    swa_indices     i32  [rows,128]        in   window slots (page*256+row)
    swa_lengths     i32  [rows]            in   valid window entries per row
    indexed_cache   u8   [pages_c,B]       in   compressed FP8 cache (ignored when W == 0)
    indexed_indices i32  [rows,W]          in   compressed slots (page*R+row) (ignored when W == 0)
    indexed_lengths i32  [rows]            in   valid compressed entries (ignored when W == 0)
    attn_sink       f32  [N]               in   attn.attn_sink
    out             bf16 [rows,N,512]      out
    scratch         u8   sparse_mla_scratch_bytes(...)  scratch
    rows            int32

Scratch (live-row layout; size with ``rows = max_rows``):
    prefill: lse f32 [rows,N]                                   = rows*N*4
    decode:  partials bf16 [rows,N,S,512] then (1024-aligned)
             partial_lse f32 [rows,N,S]            = align1024(rows*N*S*1024) + rows*N*S*4
    with ``S = program.geometry["num_splits"]``.

All cache addressing is 64-bit (page * page_bytes); the unused indexed
pointers of a window-only program may be any address. The cache, index,
q/out and scratch bases must be 16-byte aligned (256 recommended).
"""

from __future__ import annotations

import math
from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x.attention._shared.cute.ops import LOG2_E

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program

__all__ = ["compile_dsv4_sparse_mla_aot", "sparse_mla_scratch_bytes"]

_HEAD = 512
_SWA_WIDTH = 128
_SWA_PAGE = 256
_SWA_PAGE_BYTES = 149_760
_CAND = 64
_ALIGN = 1024


def _indexed_page_bytes(page_rows: int) -> int:
    return (page_rows * 584 + 575) // 576 * 576


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def sparse_mla_scratch_bytes(*, route: str, heads: int, rows: int, num_splits: int = 0) -> int:
    rows = max(int(rows), 1)
    if route == "prefill":
        return rows * heads * 4
    return _align(rows * heads * num_splits * _HEAD * 2) + rows * heads * num_splits * 4


def _traits():
    from b12x.attention._shared.mla.traits import resolve_unplanned_traits

    # Exactly what the prepared deepseek_v4 route resolves for a uint8 cache.
    return resolve_unplanned_traits(_HEAD, torch.uint8, _SWA_PAGE_BYTES)


def _prepared_query(geometry: DSV4Geometry, mode: str, max_rows: int, width: int, page_rows: int):
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._preparation import _query
    from b12x.preparation import FrozenMapping

    heads = geometry.heads
    total = _SWA_WIDTH + width
    caps = mla.Caps(
        device=torch.device("cuda", torch.cuda.current_device()), num_q_heads=heads,
        max_q_rows=max_rows, max_width=total, max_page_table_width=total, max_batch=max_rows,
        max_kv_rows=max_rows * total, mode=mode, swa_width=_SWA_WIDTH, indexed_width=width,
        swa_page_size=_SWA_PAGE, indexed_page_size=page_rows if width else _SWA_PAGE,
    )

    def desc(shape):
        stride = tuple(int(math.prod(shape[i + 1:])) for i in range(len(shape)))
        return FrozenMapping({"shape": tuple(shape), "stride": stride, "alignment": 16,
                              "dtype": "bfloat16" if len(shape) == 3 else "uint8"})

    invocation = mla.invocation_from_descriptors(
        q=desc((max_rows, heads, _HEAD)), swa_cache=desc((1, _SWA_PAGE_BYTES)),
        indexed_cache=desc((1, _indexed_page_bytes(page_rows))) if width else None,
        attn_sink_present=True, output_mode="provided",
    )
    return _query(caps, invocation)


class _Prefill:
    """Single-pass MG route (``mode="extend"``), mirroring run_unified_prefill."""

    def __init__(self, geometry: DSV4Geometry, width: int, page_rows: int):
        from b12x.attention._shared.mla.prefill import _mg_head_partitions
        from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
        from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg
        from b12x.attention._shared.mla.traits import ComputeMode, make_unified_traits

        self.heads = geometry.heads
        self.width = int(width)
        self.has_extra = self.width > 0
        base = _traits()
        # The 128-wide window routes to the BF16-QK MG specialization, single
        # or dual cache alike (run_unified_prefill).
        traits = make_unified_traits(
            int(base.model_type), int(ComputeMode.BF16), int(base.scale_format),
            fp8_rope=bool(base.fp8_rope), latent_scale_per_token=bool(base.latent_scale_per_token),
        )
        traits = replace(traits, fp8_internal=False)
        num_main_tiles = (_SWA_WIDTH + _CAND - 1) // _CAND
        num_tiles = num_main_tiles + ((self.width + _CAND - 1) // _CAND if self.has_extra else 0)
        self.page_rows = int(page_rows) if self.has_extra else 1
        self.extra_stride = _indexed_page_bytes(page_rows) if self.has_extra else 0
        self.kernels = []
        for mg_n_hg, active_heads, head_offset in _mg_head_partitions(self.heads, 16):
            layout = make_smem_layout_mg(traits, int(mg_n_hg))
            heads_per_cta = int(layout.heads_per_cta)
            if active_heads % heads_per_cta == 0:
                valid_hpb, replicate_h = int(traits.hpb), active_heads // heads_per_cta
            else:
                valid_hpb, replicate_h = active_heads, 1
            self.kernels.append(UnifiedPrefillMGKernel(
                traits, layout, _SWA_PAGE, num_tiles if self.has_extra else num_main_tiles,
                replicate_h=replicate_h, num_heads=self.heads,
                q_stride=(self.heads * _HEAD, _HEAD, 1), indices_stride0=_SWA_WIDTH,
                output_stride=(self.heads * _HEAD, _HEAD, 1), out_lse_stride=(self.heads, 1),
                has_sink=True, topk=_SWA_WIDTH, has_extra=self.has_extra,
                pbs_extra=self.page_rows,
                num_main_tiles=num_main_tiles if self.has_extra else 0,
                extra_topk=self.width if self.has_extra else 0,
                extra_indices_stride0=self.width if self.has_extra else _SWA_WIDTH,
                row_xor=self.has_extra and self.page_rows == 2, head_offset=head_offset,
                valid_hpb=valid_hpb, pack_hilo_rows=False,
            ))
        self.key = ("prefill", len(self.kernels), num_tiles, self.page_rows)

    @cute.jit
    def __call__(self, q: cute.Pointer, swa_cache: cute.Pointer, swa_indices: cute.Pointer,
                 swa_lengths: cute.Pointer, indexed_cache: cute.Pointer,
                 indexed_indices: cute.Pointer, indexed_lengths: cute.Pointer,
                 attn_sink: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        n = self.heads
        qt = cute.make_tensor(q, cute.make_layout((m, n, _HEAD), stride=(n * _HEAD, _HEAD, 1)))
        kv = cute.make_tensor(swa_cache, cute.make_layout((1,)))
        idx = cute.make_tensor(swa_indices, cute.make_layout((m, _SWA_WIDTH), stride=(_SWA_WIDTH, 1)))
        lens = cute.make_tensor(swa_lengths, cute.make_layout((m,)))
        sink = cute.make_tensor(attn_sink, cute.make_layout((n,)))
        ot = cute.make_tensor(out, cute.make_layout((m, n, _HEAD), stride=(n * _HEAD, _HEAD, 1)))
        lse = cute.make_tensor(
            cute.make_ptr(Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n), stride=(n, 1)))
        scale = Float32(float(_HEAD ** -0.5) * LOG2_E)
        for kernel in cutlass.range_constexpr(len(self.kernels)):
            if cutlass.const_expr(self.has_extra):
                width = self.width
                self.kernels[kernel].call_dual(
                    qt, kv, idx, lens, sink, ot, lse, scale, Float32(1.0),
                    Int64(_SWA_PAGE_BYTES),
                    cute.make_tensor(indexed_cache, cute.make_layout((1,))),
                    cute.make_tensor(indexed_indices, cute.make_layout((m, width), stride=(width, 1))),
                    cute.make_tensor(indexed_lengths, cute.make_layout((m,))),
                    Int64(self.extra_stride), rows, stream,
                )
            else:
                self.kernels[kernel](qt, kv, idx, lens, sink, ot, lse, scale, Float32(1.0),
                                     Int64(_SWA_PAGE_BYTES), rows, stream)


class _Decode:
    """Split-KV decode + sink merge (``mode="decode"`` on SM120)."""

    def __init__(self, geometry: DSV4Geometry, max_rows: int, width: int, page_rows: int):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from b12x.attention._shared.mla.kernel import (
            UnifiedDecodeKernel,
            prepare_unified_decode_launch,
        )
        from b12x.attention._shared.mla.merge import SparseMLASplitDecodeSinkMergeKernel
        from b12x.attention._shared.mla.smem import make_smem_layout
        from b12x.attention._shared.mla.traits import ModelType, ScaleFormat, make_unified_traits
        from b12x.attention.compressed_sparse_mla._preparation import _fake, _fake_binding
        from b12x.attention.compressed_sparse_mla._tuning import TUNING
        from b12x.preparation import detect_device

        self.heads = geometry.heads
        self.width = int(width)
        self.has_extra = self.width > 0
        query = _prepared_query(geometry, "decode", max_rows, self.width, page_rows)
        device = detect_device(torch.device("cuda", torch.cuda.current_device()))
        configuration = TUNING.configure(query, device=device.identity)
        config = configuration.default if configuration.pinned is None else configuration.pinned
        if config.single_pass:
            raise ValueError("this device plans the single-pass decode; use route='prefill'")
        ordinal = device.ordinal
        with torch.cuda.device(ordinal), FakeTensorMode():
            _dev, binding, swa, indexed = _fake_binding(query, config, ordinal)
            plan = prepare_unified_decode_launch(
                q_all=binding.q, q_alignment=16, swa_k_cache=swa, swa_indices=binding.swa_indices,
                swa_page_size=_SWA_PAGE, workspace=binding.scratch,
                sm_count=torch.cuda.get_device_properties(ordinal).multi_processor_count,
                indexed_k_cache=indexed, indexed_indices=binding.indexed_indices,
                indexed_page_size=page_rows if self.has_extra else None,
                attn_sink=_fake((self.heads,), dtype=torch.float32, device=_dev),
            )
        self.plan = plan
        self.splits = int(plan.num_splits)
        traits = plan.traits
        native_h8 = plan.native_dsv4_h8
        if native_h8:
            traits = replace(traits, nt_per_warp_xv=int(traits.nt_per_warp_xv) * 2,
                             math_threads=128, block_threads=160)
        elif plan.native_dsv4_h16:
            traits = replace(traits, nt_per_warp_xv=int(traits.nt_per_warp_xv) * 2)
        assert int(traits.model_type) == int(ModelType.DSV4)
        assert int(traits.scale_format) != int(ScaleFormat.NVFP4_E4M3)
        layout = make_smem_layout(traits)
        n, s = self.heads, self.splits
        self.num_main_chunks = (_SWA_WIDTH + _CAND - 1) // _CAND
        self.page_rows = int(page_rows) if self.has_extra else 1
        self.extra_stride = _indexed_page_bytes(page_rows) if self.has_extra else 0
        grids = []
        if plan.h_blocks_full:
            grids.append((plan.h_blocks_full, plan.hpb, 0))
        if plan.rem_heads:
            grids.append((1, plan.rem_heads, plan.h_blocks_full))
        self.kernels = [
            UnifiedDecodeKernel(
                traits, layout, _SWA_PAGE, int(plan.chunks_per_split), h_blocks=h_blocks,
                num_splits=s, num_heads=n, q_head_dim=_HEAD, topk=_SWA_WIDTH,
                extra_topk=self.width, q_stride=(n * _HEAD, _HEAD, 1),
                swa_indices_stride0=_SWA_WIDTH,
                extra_indices_stride0=self.width if self.has_extra else _SWA_WIDTH,
                mid_out_stride=(n * s * _HEAD, s * _HEAD, _HEAD, 1), mid_lse_stride=(n * s, s, 1),
                has_extra=self.has_extra, pbs_extra=self.page_rows, valid_hpb=valid_hpb,
                head_block_offset=offset, per_token_len=True, native_glm_h8=False,
                native_dsv4_h8=plan.native_dsv4_h8, native_dsv4_h16=plan.native_dsv4_h16,
                native_dsv41_fp8=False, vector_q=plan.vector_q,
            )
            for h_blocks, valid_hpb, offset in grids
        ]
        self.merge = SparseMLASplitDecodeSinkMergeKernel(static_num_chunks=s)
        self.key = ("decode", s, int(plan.chunks_per_split), plan.native_dsv4_h8,
                    plan.native_dsv4_h16, plan.vector_q, tuple(grids), self.page_rows)

    @cute.jit
    def __call__(self, q: cute.Pointer, swa_cache: cute.Pointer, swa_indices: cute.Pointer,
                 swa_lengths: cute.Pointer, indexed_cache: cute.Pointer,
                 indexed_indices: cute.Pointer, indexed_lengths: cute.Pointer,
                 attn_sink: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        n, s = self.heads, self.splits
        base = Int64(scratch.toint())
        partials = cute.make_tensor(
            cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n, s, _HEAD), stride=(n * s * _HEAD, s * _HEAD, _HEAD, 1)))
        lse_off = (m * Int64(n * s * _HEAD * 2) + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        partial_lse = cute.make_tensor(
            cute.make_ptr(Float32, base + lse_off, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n, s), stride=(n * s, s, 1)))
        qt = cute.make_tensor(q, cute.make_layout((m, n, _HEAD), stride=(n * _HEAD, _HEAD, 1)))
        kv = cute.make_tensor(swa_cache, cute.make_layout((1,)))
        idx = cute.make_tensor(swa_indices, cute.make_layout((m, _SWA_WIDTH), stride=(_SWA_WIDTH, 1)))
        lens = cute.make_tensor(swa_lengths, cute.make_layout((m,)))
        scale = Float32(float(_HEAD ** -0.5) * LOG2_E)
        for kernel in cutlass.range_constexpr(len(self.kernels)):
            if cutlass.const_expr(self.has_extra):
                width = self.width
                self.kernels[kernel].call_extra_pertok(
                    qt, kv, idx, partials, partial_lse, scale, Float32(1.0), lens,
                    Int64(_SWA_PAGE_BYTES),
                    cute.make_tensor(indexed_cache, cute.make_layout((1,))),
                    cute.make_tensor(indexed_indices, cute.make_layout((m, width), stride=(width, 1))),
                    cute.make_tensor(indexed_lengths, cute.make_layout((m,))),
                    Int32(self.num_main_chunks), Int64(self.extra_stride), rows, stream,
                )
            else:
                self.kernels[kernel].call_pertok(
                    qt, kv, idx, partials, partial_lse, scale, Float32(1.0), lens,
                    Int64(_SWA_PAGE_BYTES), rows, stream,
                )
        self.merge(
            partials, partial_lse, lens, cute.make_tensor(attn_sink, cute.make_layout((n,))),
            cute.make_tensor(out, cute.make_layout((m, n, _HEAD), stride=(n * _HEAD, _HEAD, 1))),
            stream,
        )


def compile_dsv4_sparse_mla_aot(geometry: DSV4Geometry = FLASH, *, route: str = "prefill",
                                max_rows: int = 1, indexed_width: int = 0,
                                indexed_page_rows: int = 64):
    """Compressed sparse MLA with sink; see the module docstring for the ABI.

    ``indexed_width`` 0 compiles the window-only program (compress ratio 0
    layers); C4 layers use ``indexed_width=geometry.index_topk`` and
    ``indexed_page_rows=64``; C128 layers ``indexed_page_rows=2`` and a width
    that is a multiple of 64 covering the longest compressed prefix.
    """
    width = int(indexed_width)
    if width < 0 or (width and width % _CAND):
        raise ValueError("indexed_width must be zero or a positive multiple of 64")
    if width and indexed_page_rows not in (2, 64):
        raise ValueError("indexed_page_rows must be 64 (C4) or 2 (C128)")
    if route == "prefill":
        launch = _Prefill(geometry, width, indexed_page_rows)
        splits = 0
    elif route == "decode":
        launch = _Decode(geometry, int(max_rows), width, indexed_page_rows)
        splits = launch.splits
    else:
        raise ValueError("route must be 'prefill' or 'decode'")
    n = geometry.heads
    pb = _indexed_page_bytes(indexed_page_rows) if width else 0
    operands = (
        Operand("q", torch.bfloat16, f"[rows,{n},512]"),
        Operand("swa_cache", torch.uint8, "[pages,149760]"),
        Operand("swa_indices", torch.int32, "[rows,128]", align=4),
        Operand("swa_lengths", torch.int32, "[rows]", align=4),
        Operand("indexed_cache", torch.uint8, f"[pages_c,{pb}]"),
        Operand("indexed_indices", torch.int32, f"[rows,{width}]", align=4),
        Operand("indexed_lengths", torch.int32, "[rows]", align=4),
        Operand("attn_sink", torch.float32, f"[{n}]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{n},512]", "out"),
        Operand("scratch", torch.uint8, "[sparse_mla_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name=f"dsv4_sparse_mla_{route}", operands=operands, scalars=(Scalar("rows"),),
        key=(n, width, int(indexed_page_rows) if width else 0, launch.key),
        geometry={"heads": n, "route": route, "max_rows": int(max_rows), "indexed_width": width,
                  "indexed_page_rows": int(indexed_page_rows) if width else 0,
                  "indexed_page_bytes": pb, "num_splits": splits},
        scratch={"scratch": lambda rows: sparse_mla_scratch_bytes(
            route=route, heads=n, rows=rows, num_splits=splits)},
        doc=__doc__,
    )
