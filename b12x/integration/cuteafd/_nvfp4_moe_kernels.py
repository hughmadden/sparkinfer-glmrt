"""NVFP4 weights for the exact routed-expert program (``fp8_moe``, ``weights="nvfp4"``).

NVIDIA ModelOpt NVFP4 checkpoints store routed experts as ``weight`` U8 ``[N,
K/2]`` (two E2M1 codes per byte, the even element in the low nibble),
``weight_scale`` E4M3 ``[N, K/16]`` (one scale per 16 values along K, linear
layout) and a per-tensor FP32 ``weight_scale_2`` (``alpha``): ``w = e2m1 *
e4m3 * alpha``. Every E2M1 value times an E4M3 scale is a BF16 number (at most
6 significant bits, exponents within BF16's range), so the kernels widen the
weights to ``bf16(e2m1 * e4m3)`` exactly (``cvt.rn.bf16x2.e2m1x2`` then one
``mul.bf16x2`` by the scale), run BF16 tensor-core MMAs with FP32 accumulation
and multiply each output by its expert's ``alpha`` in FP32 before the one BF16
rounding: ``bf16(alpha * (x . bf16(e2m1 * e4m3)))``, the W4A16 reading of
NVIDIA's weights (no activation quantization; the checkpoint's static
``input_scale`` belongs to its W4A4 recipe and is not used).

Scale operands carry the per-expert scalars: ``s`` is E4M3 ``[E, N, K/16]``
followed by FP32 ``[E]`` (the experts' ``weight_scale_2``) at byte ``E * N *
K / 16`` (a multiple of 16 for every 128-aligned K), then FP32 ``[E]`` (their
``input_scale``, read only by the W4A4 route).

``GroupedNvfp4Gemv``  every row count: ``GroupedMxfp4Gemv``'s grouped
    weight-streaming GEMV with each lane reading 32 consecutive K values (16
    bytes, two scale bytes) per 128-wide K block.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import bf16_mma_m16n8k16_f32
from b12x.gemm.bf16_gemv._skinny import _ld_cached, _ld_stream

from ._fp8_moe_kernels import META_HEAD, _ceil, _i32_at
from ._mxfp4_moe_kernels import e2m1x8_scaled_bf16

__all__ = ["GroupedNvfp4Gemv", "e4m3_bf16x2", "nvfp4_alpha_offset"]


def nvfp4_alpha_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the FP32 alphas in an NVFP4 scale operand ``[E, rows, K/16]``."""
    return int(experts) * int(rows) * (int(k) // 16)


@dsl_user_op
def _ld_u16(address, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.u16 $0, [$1];", "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_f32(address, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def e4m3_bf16x2(code, *, loc=None, ip=None):
    """One E4M3 code (low byte) as a BF16 pair (exact: E4M3 is a subset of BF16)."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Uint32(code).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e, lo, hi;
            .reg .b32 p, b;
            .reg .f32 f;
            and.b32 b, $1, 255;
            cvt.u16.u32 e, b;
            cvt.rn.f16x2.e4m3x2 p, e;
            mov.b32 {lo, hi}, p;
            cvt.f32.f16 f, lo;
            cvt.rn.bf16x2.f32 $0, f, f;
        }
        """,
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


class GroupedNvfp4Gemv:
    """Per active expert ``e`` (grid y), ``out[off_e + r] = alpha_e * (a_r @
    bf16(w_e)^T)`` over NVFP4 weights for its grouped rows (``max_rows`` per
    chunk; larger groups stream the weights again per chunk). Columns ``[0,
    split)`` use ``w_a``/``s_a`` and ``[split, N)`` ``w_b``/``s_b`` (gate then
    up), each U8 ``[E, part, K/2]`` with E4M3 ``[E, part, K/16]`` scales then
    FP32 ``[E]`` alphas; ``out`` BF16 ``[grouped, N]``. A CTA owns ``8 *
    groups`` columns, its warps split K in 128-wide blocks; lane ``j`` of a
    column quad reads K values ``[32j, 32j + 32)`` of each block (two 16-value
    scale groups; the activation fragment uses the same K order), the next
    block is register double-buffered, m16n8k16 BF16 MMAs accumulate in FP32,
    the warps' partials are summed in shared memory and scaled by the
    expert's alpha before the BF16 store."""

    def __init__(self, *, n: int, k: int, experts: int, split: int | None = None, max_rows: int = 64,
                 warps: int = 4, groups: int = 4, gather: bool = False):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.split = self.n if split is None else int(split)
        self.warps, self.groups = int(warps), int(groups)
        self.cols = 8 * self.groups
        self.m_tiles = _ceil(max_rows, 16)
        self.gather = bool(gather)
        if self.n % self.cols or self.split % self.cols or self.k % (128 * self.warps):
            raise ValueError("grouped NVFP4 GEMV needs N and the gate/up split % (8*groups) == 0 and "
                             "K % (128*warps) == 0")
        self.k_per_warp = self.k // self.warps
        self.blocks = self.k_per_warp // 128
        self.row_bytes = self.k // 2
        self.scale_bytes = self.k // 16
        self.frags = self.m_tiles * self.groups * 4

    def key(self) -> tuple:
        return ("nvfp4", 1, self.n, self.k, self.experts, self.split, self.warps, self.groups, self.m_tiles,
                self.gather)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
                 s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer, max_groups: Int32,
                 stream: cuda.CUstream):
        self.kernel(a, pair_row, meta, w_a, s_a, w_b, s_b, out).launch(
            grid=(self.n // self.cols, max_groups, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load_block(self, dest: cute.Tensor, factors: cute.Tensor, w_row: Int64, s_row: Int64, block):
        for gi in cutlass.range_constexpr(self.groups):
            words = _ld_stream(w_row + Int64(gi * 8 * self.row_bytes) + Int64(block) * Int64(64))
            for t in cutlass.range_constexpr(4):
                dest[gi * 4 + t] = words[t]
            factors[gi] = _ld_u16(s_row + Int64(gi * 8 * self.scale_bytes) + Int64(block) * Int64(8))

    @cute.jit
    def _row_address(self, a: cute.Pointer, pair_row: cute.Pointer, grouped: Int32) -> Int64:
        source = grouped
        if const_expr(self.gather):
            source = _i32_at(Int64(pair_row.toint()), grouped)
        return Int64(a.toint()) + Int64(source) * Int64(self.k * 2)

    @cute.kernel
    def kernel(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
               s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer):
        group = Int32(cute.arch.block_idx()[1])
        m_at = Int64(meta.toint())
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.frags * 32,)))
        if group < _i32_at(m_at, 0):
            e = _i32_at(m_at + Int64(4 * (META_HEAD + 2 * self.experts)), group)
            n_e = _i32_at(m_at + Int64(4 * META_HEAD), e)
            first = _i32_at(m_at + Int64(4 * (META_HEAD + self.experts)), e)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            g = lane // Int32(4)
            j = lane % Int32(4)
            n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
            w_base = Int64(w_a.toint())
            s_base = Int64(s_a.toint())
            part = Int64(self.split)
            local = n0
            if n0 >= Int64(self.split):
                w_base = Int64(w_b.toint())
                s_base = Int64(s_b.toint())
                part = Int64(self.n - self.split)
                local = n0 - Int64(self.split)
            # The expert's alpha follows the region's E4M3 scales.
            alpha = _ld_f32(s_base + Int64(self.experts) * part * Int64(self.scale_bytes) + Int64(e) * Int64(4))
            w_base = w_base + Int64(e) * part * Int64(self.row_bytes)
            s_base = s_base + Int64(e) * part * Int64(self.scale_bytes)
            k_begin = Int64(warp_id) * Int64(self.k_per_warp)
            w_row = w_base + (local + Int64(g)) * Int64(self.row_bytes) + k_begin // Int64(2) + Int64(16) * Int64(j)
            s_row = s_base + (local + Int64(g)) * Int64(self.scale_bytes) + k_begin // Int64(16) + Int64(2) * Int64(j)
            x_off = (k_begin + Int64(32) * Int64(j)) * Int64(2)
            chunks = (n_e + Int32(16 * self.m_tiles - 1)) // Int32(16 * self.m_tiles)
            for c in cutlass.range(chunks, unroll=1):
                row0 = first + c * Int32(16 * self.m_tiles)
                n_c = n_e - c * Int32(16 * self.m_tiles)
                if n_c > Int32(16 * self.m_tiles):
                    n_c = Int32(16 * self.m_tiles)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for i in cutlass.range_constexpr(self.frags):
                    acc[i] = Float32(0.0)
                live_tiles = (n_c + Int32(15)) // Int32(16)
                per_block = self.groups * 4
                cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                f_cur = cute.make_rmem_tensor(cute.make_layout((self.groups,), stride=(1,)), Uint32)
                f_nxt = cute.make_rmem_tensor(cute.make_layout((self.groups,), stride=(1,)), Uint32)
                self._load_block(cur, f_cur, w_row, s_row, Int32(0))
                for block in cutlass.range(self.blocks, unroll=1):
                    if block + 1 < self.blocks:
                        self._load_block(nxt, f_nxt, w_row, s_row, block + 1)
                    # Widen the block: 16 bf16x2 words per group (K 32j .. 32j + 32); words
                    # 0-1 take the first 16-value scale, 2-3 the second.
                    wide = cute.make_rmem_tensor(cute.make_layout((16 * self.groups,), stride=(1,)), Uint32)
                    for gi in cutlass.range_constexpr(self.groups):
                        f_lo = e4m3_bf16x2(f_cur[gi])
                        f_hi = e4m3_bf16x2(f_cur[gi] >> Uint32(8))
                        for t in cutlass.range_constexpr(4):
                            factor = f_lo if t < 2 else f_hi
                            v0, v1, v2, v3 = e2m1x8_scaled_bf16(cur[gi * 4 + t], factor)
                            wide[16 * gi + 4 * t] = v0
                            wide[16 * gi + 4 * t + 1] = v1
                            wide[16 * gi + 4 * t + 2] = v2
                            wide[16 * gi + 4 * t + 3] = v3
                    for chunk in cutlass.range_constexpr(2):
                        k_off = (Int64(block) * Int64(128) + Int64(chunk * 16)) * Int64(2)
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            if Int32(mt) < live_tiles:
                                r_lo = Int32(16 * mt) + g
                                r_hi = r_lo + Int32(8)
                                xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                                for i in cutlass.range_constexpr(16):
                                    xa[i] = Uint32(0)
                                if r_lo < n_c:
                                    at = self._row_address(a, pair_row, row0 + r_lo) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[i] = lo[i]
                                        xa[4 + i] = hi[i]
                                if r_hi < n_c:
                                    at = self._row_address(a, pair_row, row0 + r_hi) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[8 + i] = lo[i]
                                        xa[12 + i] = hi[i]
                                for gi in cutlass.range_constexpr(self.groups):
                                    f = 4 * (mt * self.groups + gi)
                                    for t in cutlass.range_constexpr(4):
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                            xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                            wide[16 * gi + 8 * chunk + 2 * t], wide[16 * gi + 8 * chunk + 2 * t + 1])
                                        acc[f] = d0
                                        acc[f + 1] = d1
                                        acc[f + 2] = d2
                                        acc[f + 3] = d3
                    for i in cutlass.range_constexpr(per_block):
                        cur[i] = nxt[i]
                    for i in cutlass.range_constexpr(self.groups):
                        f_cur[i] = f_nxt[i]
                for i in cutlass.range_constexpr(self.frags):
                    partial[(warp_id * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(self.frags):
                    if Int32(i % self.warps) == warp_id:
                        total = Float32(0.0)
                        for src in cutlass.range_constexpr(self.warps):
                            total = total + partial[(Int32(src * self.frags + i)) * Int32(32) + lane]
                        mt = i // (4 * self.groups)
                        gi = (i // 4) % self.groups
                        row = Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                        col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                        if row < n_c:
                            address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.n) + col) * Int64(2)
                            target = cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)
                            target[0] = (total * alpha).to(BFloat16)
                cute.arch.sync_threads()
