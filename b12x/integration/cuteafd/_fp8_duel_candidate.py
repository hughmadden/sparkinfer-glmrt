"""Duel A round 4: the Gemini entrant's FP8 decode GEMV, vendored verbatim.

Copied unchanged from fork ``duel/gem-fp8dense`` @ ``635098db`` (kernel commit
``87f7e91b``) so both entrants' kernels can be timed in one harness (a
cross-harness ratio is not comparable: the two harnesses evict differently).
Kept as a benchmark-side module: the production kernel stays
``_fp8_weights.MmaFp8Gemv``. The class subclasses ``MmaFp8Gemv`` and is
bit-exact against it.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v2_f32,
    st_shared_v4_u32,
)

from ._fp8_weights import MmaFp8Gemv, gemv_warps

__all__ = ["CandidateMmaFp8Gemv", "candidate_gemv_cta_tiles", "candidate_gemv_groups"]


_DEVICE_CAPS_CACHE: dict[int, tuple[int, int]] = {}


def _query_device_caps() -> tuple[int, int]:
    """Query ``(sm_count, l2_bytes)`` of the active CUDA device at runtime (never bake 188 or L2)."""
    err, ctx_dev = cuda.cuCtxGetDevice()
    dev_id = int(ctx_dev) if err == cuda.CUresult.CUDA_SUCCESS else 0
    cached = _DEVICE_CAPS_CACHE.get(dev_id)
    if cached is not None:
        return cached
    err_sm, sm_count = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_id
    )
    err_l2, l2_bytes = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev_id
    )
    sm = int(sm_count) if err_sm == cuda.CUresult.CUDA_SUCCESS and int(sm_count) > 0 else 170
    l2 = int(l2_bytes) if err_l2 == cuda.CUresult.CUDA_SUCCESS and int(l2_bytes) > 0 else (64 << 20)
    if err == cuda.CUresult.CUDA_SUCCESS and err_sm == cuda.CUresult.CUDA_SUCCESS:
        _DEVICE_CAPS_CACHE[dev_id] = (sm, l2)
    return sm, l2


def candidate_gemv_groups(n: int, k: int, *, warps: int = 4, groups: int = 4, row_scales: bool = False) -> int:
    """Select column groups per CTA using runtime SM count while preserving exact K reduction."""
    n = int(n)
    sm_count, _ = _query_device_caps()
    # When groups=2 (16 cols/CTA) yields fewer than 2.5 CTAs/SM, drop to groups=1 (8 cols/CTA)
    # so every SM has multiple resident CTAs (e.g. N <= 6144 on 170-SM 5090 and 188-SM PRO 6000).
    if n // 16 < (5 * sm_count) // 2 and n % 8 == 0:
        return 1
    if n % 16 == 0:
        return 2
    return 1 if n % 8 == 0 else int(groups)


def candidate_gemv_cta_tiles(n: int, k: int, *, groups: int, warps: int) -> int:
    """Select 16-row M-tiles per CTA for wide decode GEMV using runtime L2 capacity."""
    n, k = int(n), int(k)
    _, l2_bytes = _query_device_caps()
    # When the weight matrix fits within half of L2 and N <= 16480, 2 M-tiles per CTA halves
    # accumulator register pressure at 32 rows (grid_m=1) and doubles CTA parallelism at 64 rows
    # while keeping weight reuse in L2. Larger weights keep 4 tiles so 64 rows streams DRAM once.
    if n * k <= l2_bytes // 2 and n <= 16480:
        return 2
    return 4


@dsl_user_op
def _ld_stream_v4(address, *, loc=None, ip=None):
    """16-byte streaming weight load with side effects to preserve software-pipelined schedule."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::evict_first.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_cached_v4(address, *, loc=None, ip=None):
    """16-byte L1-cached activation load with side effects to preserve prefetch schedule."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::evict_last.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_f32_se(address, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _st_shared_v2_f32(smem_addr, v0, v1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)


@dsl_user_op
def _ld_shared_v2_f32(smem_addr, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _pack_bf16x2_rn(x0, x1, *, loc=None, ip=None) -> Uint32:
    """Pack two FP32 values into bf16x2 with exact round-to-nearest-even (no satfinite)."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _e4m3x8_scaled_bf16_fast(lo, hi, s, *, loc=None, ip=None):
    """Bit-identical bf16(e4m3 * s) for eight E4M3 bytes using SM120 cvt.rn.bf16x2.e4m3x2."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e0, e1, e2, e3;
            .reg .b32 p0, p1, p2, p3, b0, b1, b2, b3, b4, b5, b6, b7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            mov.b32 {e0, e1}, $4;
            mov.b32 {e2, e3}, $5;
            cvt.rn.bf16x2.e4m3x2 p0, e0;
            cvt.rn.bf16x2.e4m3x2 p1, e1;
            cvt.rn.bf16x2.e4m3x2 p2, e2;
            cvt.rn.bf16x2.e4m3x2 p3, e3;
            shl.b32 b0, p0, 16;
            and.b32 b1, p0, 0xffff0000;
            shl.b32 b2, p1, 16;
            and.b32 b3, p1, 0xffff0000;
            shl.b32 b4, p2, 16;
            and.b32 b5, p2, 0xffff0000;
            shl.b32 b6, p3, 16;
            and.b32 b7, p3, 0xffff0000;
            mov.b32 f0, b0;
            mov.b32 f1, b1;
            mov.b32 f2, b2;
            mov.b32 f3, b3;
            mov.b32 f4, b4;
            mov.b32 f5, b5;
            mov.b32 f6, b6;
            mov.b32 f7, b7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _e4m3x4_scaled_bf16x2x2_fast(word, s, *, loc=None, ip=None):
    """Bit-identical bf16(e4m3 * s) for four E4M3 bytes using SM120 cvt.rn.bf16x2.e4m3x2."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip), Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23;
            .reg .b32 p01, p23, b0, b1, b2, b3;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.bf16x2.e4m3x2 p01, e01;
            cvt.rn.bf16x2.e4m3x2 p23, e23;
            shl.b32 b0, p01, 16;
            and.b32 b1, p01, 0xffff0000;
            shl.b32 b2, p23, 16;
            and.b32 b3, p23, 0xffff0000;
            mov.b32 f0, b0;
            mov.b32 f1, b1;
            mov.b32 f2, b2;
            mov.b32 f3, b3;
            mul.rn.f32 f0, f0, $3;
            mul.rn.f32 f1, f1, $3;
            mul.rn.f32 f2, f2, $3;
            mul.rn.f32 f3, f3, $3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class CandidateMmaFp8Gemv(MmaFp8Gemv):
    """Bit-identical ``MmaFp8Gemv`` with 4-chunk rotating ping-pong pipeline (zero MOV copies,
    3 chunks continuously in flight), immediate per-group weight reload, SM120 direct
    ``cvt.rn.bf16x2.e4m3x2`` dequant, clamped/prefetched activation loads, and vectorized
    ``v2.f32`` shared fold + packed ``bf16x2`` stores."""

    def __init__(self, n: int, k: int, *, max_rows: int = 64, warps: int = 4, groups: int = 4,
                 out_dtype=cutlass.BFloat16, row_scales: bool = False, kmajor: bool = False,
                 cta_tiles: int | None = None):
        super().__init__(n, k, max_rows=max_rows, warps=warps, groups=groups, out_dtype=out_dtype,
                         row_scales=row_scales, kmajor=kmajor)
        target_tiles = (candidate_gemv_cta_tiles(self.n, self.k, groups=self.groups, warps=self.warps)
                        if cta_tiles is None else int(cta_tiles))
        self.m_tiles = min(self.total_m_tiles, target_tiles)
        self.multi_m_grid = self.total_m_tiles > self.m_tiles
        self.frags = self.m_tiles * self.tile_frags
        # Use single-barrier fold whenever total partial SMEM is <= 16 KiB.
        self.smem_frags = self.frags if (self.warps * self.frags * 32 * 4 <= 16384) else self.tile_frags

    def key(self) -> tuple:
        return ("candidate_mma_fp8_gemv", super().key(), self.smem_frags)

    @cute.jit
    def _load_chunk_v4(self, dest: cute.Tensor, w_row: Int64, k_off: Int64):
        for gi in cutlass.range_constexpr(self.groups):
            w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + k_off)
            dest[gi * 4 + 0] = w0
            dest[gi * 4 + 1] = w1
            dest[gi * 4 + 2] = w2
            dest[gi * 4 + 3] = w3

    @cute.jit
    def _load_scale_se(self, s_dest: cute.Tensor, s_row: Int64, block_idx: Int64, s_gi: Int64, s_kb: Int64):
        s_rows = self.groups if self.row_scales else 1
        base = s_row + block_idx * s_kb
        for si in cutlass.range_constexpr(s_rows):
            s_dest[si] = _ld_f32_se(base + Int64(si) * s_gi)

    @cute.jit
    def _consume_chunk(self, acc: cute.Tensor, c_buf: cute.Tensor, s_buf: cute.Tensor,
                       x_lo_base: cute.Tensor, x_hi_base: cute.Tensor,
                       w_row: Int64, s_row: Int64, s_gi: Int64, s_kb: Int64,
                       m0: Int32, rows: Int32, live_tiles: Int32,
                       k_off: Int64, next_k_off: Int64, next_block_idx: Int64,
                       reload_w: cutlass.Constexpr, reload_s: cutlass.Constexpr):
        x_byte_off = k_off * Int64(2)
        if const_expr(self.m_tiles == 1):
            lo0, lo1, lo2, lo3 = _ld_cached_v4(x_lo_base[0] + x_byte_off)
            lo4, lo5, lo6, lo7 = _ld_cached_v4(x_lo_base[0] + x_byte_off + Int64(16))
            hi0, hi1, hi2, hi3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            hi4, hi5, hi6, hi7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if m0 + Int32(8) < rows:
                hi0, hi1, hi2, hi3 = _ld_cached_v4(x_hi_base[0] + x_byte_off)
                hi4, hi5, hi6, hi7 = _ld_cached_v4(x_hi_base[0] + x_byte_off + Int64(16))
            xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
            xa[0], xa[1], xa[2], xa[3] = lo0, lo1, lo2, lo3
            xa[4], xa[5], xa[6], xa[7] = lo4, lo5, lo6, lo7
            xa[8], xa[9], xa[10], xa[11] = hi0, hi1, hi2, hi3
            xa[12], xa[13], xa[14], xa[15] = hi4, hi5, hi6, hi7
            for gi in cutlass.range_constexpr(self.groups):
                s_val = s_buf[gi if self.row_scales else 0]
                f = 4 * gi
                for t in cutlass.range_constexpr(4):
                    b0, b1 = _e4m3x4_scaled_bf16x2x2_fast(c_buf[gi * 4 + t], s_val)
                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                        xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                        b0, b1)
                    acc[f] = d0
                    acc[f + 1] = d1
                    acc[f + 2] = d2
                    acc[f + 3] = d3
                if const_expr(reload_w):
                    w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + next_k_off)
                    c_buf[gi * 4 + 0] = w0
                    c_buf[gi * 4 + 1] = w1
                    c_buf[gi * 4 + 2] = w2
                    c_buf[gi * 4 + 3] = w3
            if const_expr(reload_s):
                self._load_scale_se(s_buf, s_row, next_block_idx, s_gi, s_kb)
        else:
            bw = cute.make_rmem_tensor(cute.make_layout((8 * self.groups,), stride=(1,)), Uint32)
            for gi in cutlass.range_constexpr(self.groups):
                s_val = s_buf[gi if self.row_scales else 0]
                for t in cutlass.range_constexpr(4):
                    b0, b1 = _e4m3x4_scaled_bf16x2x2_fast(c_buf[gi * 4 + t], s_val)
                    bw[8 * gi + 2 * t] = b0
                    bw[8 * gi + 2 * t + 1] = b1
                if const_expr(reload_w):
                    w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + next_k_off)
                    c_buf[gi * 4 + 0] = w0
                    c_buf[gi * 4 + 1] = w1
                    c_buf[gi * 4 + 2] = w2
                    c_buf[gi * 4 + 3] = w3
            if const_expr(reload_s):
                self._load_scale_se(s_buf, s_row, next_block_idx, s_gi, s_kb)
            for mt in cutlass.range_constexpr(self.m_tiles):
                if Int32(mt) < live_tiles:
                    lo0, lo1, lo2, lo3 = _ld_cached_v4(x_lo_base[mt] + x_byte_off)
                    lo4, lo5, lo6, lo7 = _ld_cached_v4(x_lo_base[mt] + x_byte_off + Int64(16))
                    hi0, hi1, hi2, hi3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    hi4, hi5, hi6, hi7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if m0 + Int32(16 * mt + 8) < rows:
                        hi0, hi1, hi2, hi3 = _ld_cached_v4(x_hi_base[mt] + x_byte_off)
                        hi4, hi5, hi6, hi7 = _ld_cached_v4(x_hi_base[mt] + x_byte_off + Int64(16))
                    xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                    xa[0], xa[1], xa[2], xa[3] = lo0, lo1, lo2, lo3
                    xa[4], xa[5], xa[6], xa[7] = lo4, lo5, lo6, lo7
                    xa[8], xa[9], xa[10], xa[11] = hi0, hi1, hi2, hi3
                    xa[12], xa[13], xa[14], xa[15] = hi4, hi5, hi6, hi7
                    for gi in cutlass.range_constexpr(self.groups):
                        f = 4 * (mt * self.groups + gi)
                        for t in cutlass.range_constexpr(4):
                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                            acc[f] = d0
                            acc[f + 1] = d1
                            acc[f + 2] = d2
                            acc[f + 3] = d3

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        m0 = (Int32(cute.arch.block_idx()[1]) * Int32(16 * self.m_tiles)
              if const_expr(self.multi_m_grid) else Int32(0))
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        smem_base = shared_ptr_to_u32(storage.partial.data_ptr())
        acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.frags):
            acc[i] = Float32(0.0)
        k_begin = Int64(warp_id) * Int64(self.k_per_warp)
        w_row = Int64(w.toint()) + (n0 + Int64(g)) * Int64(self.k) + k_begin + Int64(16) * Int64(j)
        x_base = Int64(x.toint()) + (k_begin + Int64(16) * Int64(j)) * Int64(2)
        s_rows = self.groups if self.row_scales else 1
        s_first = (n0 + Int64(g)) if self.row_scales else (n0 // Int64(128))
        s_kb = Int64(self.n * 4) if self.kmajor else Int64(4)
        s_gi = Int64(8 * 4) if self.kmajor else Int64(8 * self.k_blocks * 4)
        s_row = Int64(scale.toint()) + s_first * (Int64(4) if self.kmajor else Int64(self.k_blocks * 4)) \
            + (k_begin // Int64(128)) * s_kb
        live_tiles = ((rows - m0 + Int32(15)) // Int32(16)
                      if const_expr(self.multi_m_grid) else (rows + Int32(15)) // Int32(16))
        max_row = rows - Int32(1)
        if max_row < Int32(0):
            max_row = Int32(0)
        x_lo_base = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int64)
        x_hi_base = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int64)
        for mt in cutlass.range_constexpr(self.m_tiles):
            r_lo = m0 + Int32(16 * mt) + g
            if r_lo > max_row:
                r_lo = max_row
            r_hi = m0 + Int32(16 * mt) + g + Int32(8)
            if r_hi > max_row:
                r_hi = max_row
            x_lo_base[mt] = x_base + Int64(r_lo) * Int64(self.k * 2)
            x_hi_base[mt] = x_base + Int64(r_hi) * Int64(self.k * 2)

        chunk_words = 4 * self.groups
        b0_c0 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b0_c1 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b1_c0 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b1_c1 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        s0 = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        s1 = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)

        if const_expr(self.blocks == 1):
            self._load_chunk_v4(b0_c0, w_row, Int64(0))
            self._load_chunk_v4(b0_c1, w_row, Int64(64))
            self._load_scale_se(s0, s_row, Int64(0), s_gi, s_kb)
            self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                m0, rows, live_tiles, Int64(0), Int64(0), Int64(0), False, False)
            self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                m0, rows, live_tiles, Int64(64), Int64(0), Int64(0), False, False)
        else:
            self._load_chunk_v4(b0_c0, w_row, Int64(0))
            self._load_chunk_v4(b0_c1, w_row, Int64(64))
            self._load_scale_se(s0, s_row, Int64(0), s_gi, s_kb)
            self._load_chunk_v4(b1_c0, w_row, Int64(128))
            self._load_chunk_v4(b1_c1, w_row, Int64(192))
            self._load_scale_se(s1, s_row, Int64(1), s_gi, s_kb)
            tail_blocks = 2 if (self.blocks % 2 == 0) else 3
            main_pairs = (self.blocks - tail_blocks) // 2
            if const_expr(main_pairs > 0):
                for pair in cutlass.range(main_pairs, unroll=1):
                    b = Int64(pair) * Int64(2)
                    self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, b * Int64(128), (b + Int64(2)) * Int64(128),
                                        b + Int64(2), True, False)
                    self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, b * Int64(128) + Int64(64),
                                        (b + Int64(2)) * Int64(128) + Int64(64), b + Int64(2), True, True)
                    self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, (b + Int64(1)) * Int64(128),
                                        (b + Int64(3)) * Int64(128), b + Int64(3), True, False)
                    self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, (b + Int64(1)) * Int64(128) + Int64(64),
                                        (b + Int64(3)) * Int64(128) + Int64(64), b + Int64(3), True, True)
            b_tail = Int64(2 * main_pairs)
            if const_expr(tail_blocks == 2):
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
            else:
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128), (b_tail + Int64(2)) * Int64(128),
                                    b_tail + Int64(2), True, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128) + Int64(64),
                                    (b_tail + Int64(2)) * Int64(128) + Int64(64), b_tail + Int64(2), True, True)
                self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(2)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(2)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)

        if const_expr(self.smem_frags == self.frags):
            pairs = self.frags // 2
            for p in cutlass.range_constexpr(pairs):
                addr = smem_base + ((warp_id * Int32(pairs) + Int32(p)) * Int32(32) + lane) * Int32(8)
                _st_shared_v2_f32(addr, acc[2 * p], acc[2 * p + 1])
            cute.arch.sync_threads()
            for p in cutlass.range_constexpr(pairs):
                if Int32(p % self.warps) == warp_id:
                    total0 = Float32(0.0)
                    total1 = Float32(0.0)
                    for src in cutlass.range_constexpr(self.warps):
                        addr = smem_base + (Int32(src * pairs + p) * Int32(32) + lane) * Int32(8)
                        v0, v1 = _ld_shared_v2_f32(addr)
                        total0 = total0 + v0
                        total1 = total1 + v1
                    mt = p // (2 * self.groups)
                    gi = (p // 2) % self.groups
                    row = m0 + Int32(16 * mt) + g + Int32(8 * (p % 2))
                    col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                    if row < rows:
                        if const_expr(self.out_dtype == cutlass.BFloat16):
                            address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(2)
                            st_global_u32(address, _pack_bf16x2_rn(total0, total1))
                        else:
                            address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(4)
                            st_global_v2_f32(address, total0, total1)
        else:
            tile_pairs = self.tile_frags // 2
            for mt in cutlass.range_constexpr(self.m_tiles):
                if Int32(mt) < live_tiles:
                    if const_expr(mt > 0):
                        cute.arch.sync_threads()
                    for p in cutlass.range_constexpr(tile_pairs):
                        addr = smem_base + ((warp_id * Int32(tile_pairs) + Int32(p)) * Int32(32) + lane) * Int32(8)
                        base_idx = mt * self.tile_frags + 2 * p
                        _st_shared_v2_f32(addr, acc[base_idx], acc[base_idx + 1])
                    cute.arch.sync_threads()
                    for p in cutlass.range_constexpr(tile_pairs):
                        if Int32(p % self.warps) == warp_id:
                            total0 = Float32(0.0)
                            total1 = Float32(0.0)
                            for src in cutlass.range_constexpr(self.warps):
                                addr = smem_base + (Int32(src * tile_pairs + p) * Int32(32) + lane) * Int32(8)
                                v0, v1 = _ld_shared_v2_f32(addr)
                                total0 = total0 + v0
                                total1 = total1 + v1
                            gi = p // 2
                            row = m0 + Int32(16 * mt) + g + Int32(8 * (p % 2))
                            col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                            if row < rows:
                                if const_expr(self.out_dtype == cutlass.BFloat16):
                                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(2)
                                    st_global_u32(address, _pack_bf16x2_rn(total0, total1))
                                else:
                                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(4)
                                    st_global_v2_f32(address, total0, total1)


