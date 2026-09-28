"""Native AOT DeepSeek V4 output projection: inverse RoPE + grouped wo_a + wo_b.

``compile_dsv4_wo_projection_aot(geometry, max_rows=...)`` is one program equal
to the prepared ``gemm.wo_projection`` plan with ``operation="inv_rope"`` and
``dynamic_tokens=True`` (``bind_inv_rope`` / ``run_inv_rope``):

1. inverse RoPE on ``o[..., 448:]`` (FP32 rotation, sin negated) fused into
   the grouped MXFP8 activation quantizer (K32 UE8M0 scales, per group);
2. grouped MXFP8 GEMM ``wo_a``: ``o_groups`` x ``[o_lora_rank x group_width]``
   into a BF16 group-major intermediate ``[groups, rows, rank]``;
3. group-major MXFP8 quantization of that intermediate as ``[rows, groups*rank]``;
4. MXFP8 GEMM ``wo_b`` (``[hidden x groups*rank]``) into ``out``. Split-K
   lowerings (wo_b below 16 rows) write FP32 partial planes reduced before one
   BF16 rounding; where the prepared lowering uses BF16 atomics into a zeroed
   output instead (order-dependent double rounding), this program keeps the
   same tiles and split but reduces deterministically.

The quantizers are the CuTe ports in ``b12x.gemm.wo_projection._quant_cute``
(bit-identical to the prepared Triton quantizers); the GEMMs are the default
lowerings the prepared plan compiles for capacity ``max_rows``.

ABI (``G`` o_groups, ``W`` group_width = heads/G * 512, ``R`` o_lora_rank,
``D`` hidden; ``rows <= max_rows`` live tokens; ``P`` RoPE table rows)::

    o          bf16 [rows,heads,512]   in   attention output (not de-rotated)
    positions  i64  [rows]             in   RoPE position per row
    cos_sin    f32  [P,64]             in   cos(32) | sin(32), same table as the producer
    wo_a       fp8  [G,R,W]            in   pack_weights(...).wo_a.values storage
                                            (logical view [R,W,G])
    wo_a_scale u8   G*ceil(R/128)*ceil(W/128)*512 bytes   .wo_a.scale_mma storage
    wo_b       fp8  [D,G*R]            in   .wo_b.values
    wo_b_scale u8   ceil(D/128)*ceil(G*R/128)*512 bytes   .wo_b.scale_mma storage
    out        bf16 [rows,D]           out
    scratch    u8   wo_scratch_bytes(geometry, rows, max_rows)
    rows       int32

Scratch sub-regions, each 1024-byte aligned and laid out from the live row
count (size the buffer with ``rows = max_rows``)::

    a_values      rows*G*W            FP8 [G,rows,W]
    a_scale_rows  rows*G*W/32         UE8M0 [G,rows,W/32] (quantizer byproduct)
    a_scale_mma   G*ceil(rows/128)*(W/128)*512
    tmp           rows*G*R*2          BF16 [G,rows,R]
    b_values      rows*G*R            FP8 [rows,G*R]
    b_scale_rows  rows*G*R/32
    b_scale_mma   ceil(rows/128)*(G*R/128)*512
    alpha         16 (FP32 1.0 written by the program; split-K scales by it)
    a_split       slices_a*rows*G*R*4 FP32 partials when wo_a splits K (never
                  for the V4 geometries; 0 bytes otherwise)
    b_split       slices_b*rows*D*4 FP32 partials when wo_b splits K
                  (slices_b = 2 for max_rows <= 8, else 0 bytes)

The prepared plan declared with the exact token count (``dynamic_tokens``
False, what the prototype uses) runs a fused quantize-A wo_b GEMM at <= 8
rows; that route is not exported: at those capacities this program runs the
dynamic-token route (group-major quantizer + split-K GEMM), which agrees to
BF16 rounding. Outputs, scratch and inputs must be disjoint; bases 256-byte
aligned. Flash (G=8) and Pro (G=16, heads 128, hidden 7168) are supported.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from b12x._lib.dense_gemm import _DenseSplitKReduce, dense_gemm_launch_from_lowering
from b12x.gemm.wo_projection._quant_cute import _GRID_CTAS_PER_SM, _THREADS, _WOQuantCuTeLaunch

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program
from ._linear import StoreOne

__all__ = ["compile_dsv4_wo_projection_aot", "wo_lowerings", "wo_scratch_bytes"]

_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def wo_lowerings(geometry: DSV4Geometry, max_rows: int, device=None):
    """Default (wo_a, wo_b) dense lowerings of the prepared plan at this capacity."""
    from b12x.gemm._preparation import _default_lowering
    from b12x.gemm._tuning import DenseGemmQuery
    from b12x.preparation import detect_device

    identity = detect_device(device if device is not None else torch.device(
        "cuda", torch.cuda.current_device())).identity
    g, w, r, d = geometry.o_groups, geometry.o_group_width, geometry.o_lora_rank, geometry.hidden
    common = dict(recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",
                  output_dtype="bfloat16", max_rows=int(max_rows), output_mode="provided",
                  alpha_mode="unit", expected_m=int(max_rows))
    a = DenseGemmQuery(batch=g, in_features=w, out_features=r, **common)
    b = DenseGemmQuery(batch=1, in_features=r * g, out_features=d, **common)
    return _default_lowering(a, identity), _default_lowering(b, identity)


def _layout(geometry: DSV4Geometry, rows: int, slices_a: int, slices_b: int) -> dict[str, int]:
    g, w, r, d = geometry.o_groups, geometry.o_group_width, geometry.o_lora_rank, geometry.hidden
    m_tiles = (rows + 127) // 128
    sizes = (
        ("a_values", rows * g * w),
        ("a_scale_rows", rows * g * w // 32),
        ("a_scale_mma", g * m_tiles * (w // 128) * 512),
        ("tmp", rows * g * r * 2),
        ("b_values", rows * g * r),
        ("b_scale_rows", rows * g * r // 32),
        ("b_scale_mma", m_tiles * (g * r // 128) * 512),
        ("alpha", 16),
        ("a_split", slices_a * rows * g * r * 4 if slices_a > 1 else 0),
        ("b_split", slices_b * rows * d * 4 if slices_b > 1 else 0),
    )
    offsets, cursor = {}, 0
    for name, size in sizes:
        offsets[name] = cursor
        cursor += _align(size)
    offsets["total"] = cursor
    return offsets


def wo_scratch_bytes(geometry: DSV4Geometry, rows: int, max_rows: int) -> int:
    """Exact scratch bytes for ``rows`` live rows of a ``max_rows`` program."""
    low_a, low_b = wo_lowerings(geometry, max_rows)
    return _layout(geometry, max(int(rows), 1), int(low_a.policy.split_k_slices),
                   int(low_b.policy.split_k_slices))["total"]


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class _WoProjection:
    def __init__(self, geometry: DSV4Geometry, max_rows: int):
        self.g, self.w = geometry.o_groups, geometry.o_group_width
        self.r, self.d = geometry.o_lora_rank, geometry.hidden
        if self.w != (geometry.heads // self.g) * 512:
            raise ValueError("wo_projection requires group_width = heads/groups * 512")
        low_a, low_b = wo_lowerings(geometry, max_rows)
        self.lowerings = (low_a, low_b)
        self.gemm_a, self.slices_a = dense_gemm_launch_from_lowering(low_a, atomic_split="partials")
        self.gemm_b, self.slices_b = dense_gemm_launch_from_lowering(low_b, atomic_split="partials")
        self.reduce_a = _DenseSplitKReduce(self.r * self.g, self.slices_a) if self.slices_a > 1 else None
        self.reduce_b = _DenseSplitKReduce(self.d, self.slices_b) if self.slices_b > 1 else None
        if self.slices_a > 1 and self.g > 1:
            raise ValueError("grouped wo_a split-K lowering is not supported")
        self.quant_a = _WOQuantCuTeLaunch(
            "grouped", self.g * self.w, self.w, cutlass.BFloat16, True, 512, 448, 64,
            cutlass.Int64, cutlass.Float32, _THREADS, False)
        self.quant_b = _WOQuantCuTeLaunch(
            "group_major", self.g * self.r, self.r, cutlass.BFloat16, False, 0, 0, 0,
            cutlass.Int64, cutlass.BFloat16, _THREADS, False)
        self.store_one = StoreOne()
        self.grid_cap = int(low_a.sm_count) * _GRID_CTAS_PER_SM
        self.warps = _THREADS // 32

    def key(self) -> tuple:
        return tuple(
            (low.mma_tiler_mn, low.tile_k, low.policy, low.sm_count, low.load_path,
             low.sfb_k_reuse, low.b_tile_major, low.direct_sfa_live16, low.target_occupancy_override)
            for low in self.lowerings)

    @cute.jit
    def _grid(self, rows: Int32, total_k: cutlass.Constexpr):
        grid = (rows * Int32(total_k // 128) + Int32(self.warps - 1)) // Int32(self.warps)
        if grid > Int32(self.grid_cap):
            grid = Int32(self.grid_cap)
        return grid

    @cute.jit
    def __call__(self, o: cute.Pointer, positions: cute.Pointer, cos_sin: cute.Pointer,
                 wo_a: cute.Pointer, wo_a_scale: cute.Pointer, wo_b: cute.Pointer,
                 wo_b_scale: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        g, w, r, d = self.g, self.w, self.r, self.d
        m = Int64(rows)
        m_tiles = (m + Int64(127)) // Int64(128)
        a_values = Int64(scratch.toint())
        a_scale_rows = a_values + _align_i64(m * Int64(g * w))
        a_scale_mma = a_scale_rows + _align_i64(m * Int64(g * w // 32))
        tmp = a_scale_mma + _align_i64(m_tiles * Int64(g * (w // 128) * 512))
        b_values = tmp + _align_i64(m * Int64(g * r * 2))
        b_scale_rows = b_values + _align_i64(m * Int64(g * r))
        b_scale_mma = b_scale_rows + _align_i64(m * Int64(g * r // 32))
        alpha_off = b_scale_mma + _align_i64(m_tiles * Int64((g * r // 128) * 512))
        a_split = alpha_off + Int64(_ALIGN)
        b_split = a_split + _align_i64(m * Int64(self.slices_a * g * r * 4 if self.slices_a > 1 else 0))

        gmem = cute.AddressSpace.gmem
        alpha = cute.make_ptr(cutlass.Float32, alpha_off, gmem, assumed_align=16)
        if const_expr(self.slices_a > 1 or self.slices_b > 1):
            self.store_one(alpha, stream)
        # 1. inverse RoPE + grouped quantization of o -> [G, rows, W] MXFP8.
        self.quant_a(
            o, positions, cos_sin,
            cute.make_ptr(cutlass.Uint32, a_values, gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, a_scale_rows, gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, a_scale_mma, gmem, assumed_align=16),
            rows, Int32(0x7FFFFFFF), self._grid(rows, g * w), stream,
        )
        # 2. grouped wo_a -> tmp BF16 [G, rows, R].
        a_fp8 = cute.make_ptr(cutlass.Float8E4M3FN, a_values, gmem, assumed_align=16)
        a_sf = cute.make_ptr(cutlass.Float8E8M0FNU, a_scale_mma, gmem, assumed_align=16)
        a_rows_sf = cute.make_ptr(cutlass.Float8E8M0FNU, a_scale_rows, gmem, assumed_align=16)
        wa = cute.make_ptr(cutlass.Float8E4M3FN, Int64(wo_a.toint()), gmem, assumed_align=16)
        wa_sf = cute.make_ptr(cutlass.Float8E8M0FNU, Int64(wo_a_scale.toint()), gmem, assumed_align=16)
        tmp_ptr = cute.make_ptr(cutlass.BFloat16, tmp, gmem, assumed_align=16)
        if const_expr(self.slices_a > 1):
            partials = cute.make_ptr(cutlass.Float32, a_split, gmem, assumed_align=16)
            self.gemm_a(a_fp8, wa, a_sf, wa_sf, partials, a_fp8, a_rows_sf, a_sf, alpha, rows, stream)
            self.reduce_a(partials, tmp_ptr, rows, stream)
        else:
            self.gemm_a(a_fp8, wa, a_sf, wa_sf, tmp_ptr, a_fp8, a_rows_sf, a_sf, alpha, rows, stream)
        # 3. group-major quantization of tmp -> [rows, G*R] MXFP8.
        self.quant_b(
            tmp_ptr, positions, tmp_ptr,
            cute.make_ptr(cutlass.Uint32, b_values, gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, b_scale_rows, gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, b_scale_mma, gmem, assumed_align=16),
            rows, Int32(1), self._grid(rows, g * r), stream,
        )
        # 4. wo_b -> out BF16 [rows, D].
        b_fp8 = cute.make_ptr(cutlass.Float8E4M3FN, b_values, gmem, assumed_align=16)
        b_sf = cute.make_ptr(cutlass.Float8E8M0FNU, b_scale_mma, gmem, assumed_align=16)
        b_rows_sf = cute.make_ptr(cutlass.Float8E8M0FNU, b_scale_rows, gmem, assumed_align=16)
        wb = cute.make_ptr(cutlass.Float8E4M3FN, Int64(wo_b.toint()), gmem, assumed_align=16)
        wb_sf = cute.make_ptr(cutlass.Float8E8M0FNU, Int64(wo_b_scale.toint()), gmem, assumed_align=16)
        if const_expr(self.slices_b > 1):
            partials = cute.make_ptr(cutlass.Float32, b_split, gmem, assumed_align=16)
            self.gemm_b(b_fp8, wb, b_sf, wb_sf, partials, b_fp8, b_rows_sf, b_sf, alpha, rows, stream)
            self.reduce_b(partials, out, rows, stream)
        else:
            self.gemm_b(b_fp8, wb, b_sf, wb_sf, out, b_fp8, b_rows_sf, b_sf, alpha, rows, stream)


def compile_dsv4_wo_projection_aot(geometry: DSV4Geometry = FLASH, *, max_rows: int):
    """Inverse-RoPE grouped WO projection for ``rows <= max_rows`` (see module docstring)."""
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    launch = _WoProjection(geometry, max_rows)
    g, w, r, d, h = geometry.o_groups, geometry.o_group_width, geometry.o_lora_rank, geometry.hidden, geometry.heads
    operands = (
        Operand("o", torch.bfloat16, f"[rows,{h},512]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("wo_a", torch.float8_e4m3fn, f"[{g},{r},{w}]"),
        Operand("wo_a_scale", torch.uint8, f"[{g * ((r + 127) // 128) * (w // 128) * 512}]"),
        Operand("wo_b", torch.float8_e4m3fn, f"[{d},{g * r}]"),
        Operand("wo_b_scale", torch.uint8, f"[{((d + 127) // 128) * (g * r // 128) * 512}]"),
        Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
        Operand("scratch", torch.uint8, "[wo_scratch_bytes]", "scratch"),
    )
    slices = (launch.slices_a, launch.slices_b)
    return compile_program(
        launch, name="dsv4_wo_projection", operands=operands, scalars=(Scalar("rows"),),
        key=(g, w, r, d, launch.key()),
        geometry={"groups": g, "group_width": w, "rank": r, "hidden": d, "heads": h,
                  "max_rows": max_rows, "split_k_slices": slices},
        scratch={"scratch": lambda rows: _layout(geometry, max(int(rows), 1), *slices)["total"]},
        doc=__doc__,
    )
