"""Expert-stationary streaming GEMMs over MXFP4 weights (fp8_moe ``stream``
route, ``weights="mxfp4"``, MiMo V2.6 Pro).

The ``_fp8_moe_stream`` design with packed E2M1 weights (``[E, N, K/2]`` U8,
the even element in the low nibble) and UE8M0 scales per 32 values along K
(``[E, N, K/32]`` U8). Every weight byte is read from DRAM once per layer per
128-row chunk group, by TMA in whole 128-byte lines per row (256 K values; a
64-byte box when the down projection's K slice is not a multiple of 256, as
TP6's 384). Each expert's scale rows for the CTA's columns are loaded once
into shared memory. Weights are widened exactly: ``cvt.rn.bf16x2.e2m1x2``
then one ``mul.bf16x2`` by ``2^(s - 127)`` (every product is a BF16 number),
the reference's dequantized weight, feeding m16n8k16 BF16 MMAs with FP32
accumulation.

``StreamMxfp4GateUp``  gate/up from the FP8 K32 wire rows exactly as
    ``StreamFp8GateUp`` (A by 8-bit ``ldmatrix`` under its K permutation);
    each thread widens the B fragment of that permutation straight from the
    swizzled E2M1 tile (two bytes per 16-K MMA half), SwiGLU fused.
``StreamMxfp4Down``    down over the BF16 act rows as ``StreamFp8Down``, the
    E2M1 tile widened to BF16 in shared memory (double-buffered).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    e2m1x8_to_qmma_e2m1x8,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    max_abs_32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v4_u32,
    st_global_u8,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._fp8_moe_kernels import _i32_at, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, StreamFp8Down, StreamFp8GateUp, _chunk, _swiglu, _u8_weight
from ._fp8_weights import _e4m3x4_scaled_bf16x2x2
from ._mxfp4_moe_kernels import e2m1x8_scaled_bf16, ue8m0_bf16x2
from ._mxfp4_down_a8_plan import mxfp8_down_row_bytes
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

__all__ = ["StreamMxfp4Down", "StreamMxfp4GateUp"]


@dsl_user_op
def e2m1x4_scaled_bf16(codes, factor, *, loc=None, ip=None):
    """Four E2M1 codes (low 16 bits, byte 0 low nibble first) times a BF16
    pair ``factor``: two bf16x2 words (exact)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(codes).ir_value(loc=loc, ip=ip), Uint32(factor).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 q0, q1, q2, q3;
            .reg .b32 v0, v1;
            mov.b32 {q0, q1, q2, q3}, $2;
            cvt.rn.bf16x2.e2m1x2 v0, q0;
            cvt.rn.bf16x2.e2m1x2 v1, q1;
            mul.rn.bf16x2 $0, v0, $3;
            mul.rn.bf16x2 $1, v1, $3;
        }
        """,
        "=r,=r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class StreamMxfp4GateUp(StreamFp8GateUp):
    """``act[p] = swiglu(x_p . gate_e, x_p . up_e)`` over MXFP4 ``w1``/``w3``
    ``[E, I, H/2]`` with UE8M0 ``s1``/``s3`` ``[E, I, H/32]``; otherwise
    ``StreamFp8GateUp`` (grid ``(I / 32, max_tiles)``, 256 threads, 128-K A
    steps). The weight ring holds 256-K blocks (128-byte lines): one block
    per two A steps."""

    # Shared memory: A 3 x 16 KiB, weights 3 x 8 KiB, scales 64 x (H/32 + 16) B.
    a_ring = 3
    w_ring = 3
    w_k = 256

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0, qmma: bool = False,
                 act_mxfp8: bool = False):
        super().__init__(inter=inter, hidden=hidden, experts=experts, limit=limit)
        # qmma: block-scaled E4M3 x E2M1 MMAs (m16n8k32, UE8M0 per 32 on both
        # operands) read the wire rows and the packed weights as they are.
        self.qmma = bool(qmma)
        self.act_mxfp8 = bool(act_mxfp8)
        self.act_row_bytes = mxfp8_down_row_bytes(self.inter) if self.act_mxfp8 else 2 * self.inter
        # Once its async reads drain, reuse A shared memory for exact BF16
        # SwiGLU rounding before the row-local K32 quantization.
        self.act_stride = self.cols * 2 + 16
        if self.hidden % 512:
            raise ValueError("MXFP4 stream gate/up needs H % 512 == 0 (16-byte scale rows)")
        self.w_blocks = self.hidden // self.w_k
        self.scale_cols = self.hidden // 32
        # Padded scale rows: the 8 column rows of a fragment hit distinct banks.
        self.sc_stride = self.scale_cols + 16
        if self.w_blocks < self.w_ring:
            raise ValueError("MXFP4 stream gate/up needs at least as many weight blocks as ring slots")

    def key(self) -> tuple:
        key = ("stream_gate_up_mxfp4", 1, self.inter, self.hidden, self.experts, self.limit, self.a_ring,
               self.w_ring, self.qmma)
        return key + (("act_mxfp8", self.act_row_bytes),) if self.act_mxfp8 else key

    def _storage(self, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.a_bytes], 1024],
            "s": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.s_bytes], 16],
            "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.cols * self.sc_stride], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, act: cute.Pointer, max_tiles: Int32,
                 stream: cuda.CUstream):
        w_layout = self._w_layout()
        rows_w = self.experts * self.inter
        box = cute.slice_(w_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.hidden // 2), box,
            (self.cols, self.k_step), num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.hidden // 2), box,
            (self.cols, self.k_step), num_multicast=1)
        self.kernel(x, pair_row, meta, tma_tensor_g, tma_tensor_u, s1, s3, act, tma_g, tma_u, w_layout,
                    self._storage(w_layout)).launch(
            grid=(self.inter // self.cols, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, weight_g: cute.Tensor,
               weight_u: cute.Tensor, s1: cute.Pointer, s3: cute.Pointer, act: cute.Pointer,
               tma_g: cute.CopyAtom, tma_u: cute.CopyAtom, w_layout: cute.ComposedLayout,
               Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            mbar = storage.mbar.data_ptr()
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            ss = shared_ptr_to_u32(storage.s.data_ptr())
            sc = shared_ptr_to_u32(storage.sc.data_ptr())
            h, i = self.hidden, self.inter
            n0 = n_blk * Int32(self.cols)
            w_tile = e * Int32(i // self.cols) + n_blk
            t_ws, t_wg = cpasync.tma_partition(tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight_g, (self.cols, self.k_step),
                                                                                (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                            cute.group_modes(cute.local_tile(weight_u, (self.cols, self.k_step),
                                                                             (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
                for slot in cutlass.range_constexpr(self.w_ring):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            ch = tidx % Int32(8)
            r_base = tidx // Int32(8)
            src = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
            for q in cutlass.range_constexpr(4):
                row = r_base + Int32(32 * q)
                src[q] = Int64(-1)
                if row < live:
                    source = _i32_at(Int64(pair_row.toint()), first + row)
                    src[q] = Int64(x.toint()) + Int64(source) * Int64(self.row_bytes) + Int64(ch) * Int64(16)
            s_src = Int64(-1)
            if tidx < live:
                source = _i32_at(Int64(pair_row.toint()), first + tidx)
                s_src = Int64(x.toint()) + Int64(source) * Int64(self.row_bytes) + Int64(h)
            # The CTA's scale rows (32 gate then 32 up columns, H/32 bytes each), once;
            # they join the first cp.async group.
            units = self.scale_cols // 16
            for q in cutlass.range_constexpr((2 * self.cols * units + self.threads - 1) // self.threads):
                u = tidx + Int32(q * self.threads)
                if u < Int32(2 * self.cols * units):
                    r = u // Int32(units)
                    c = u % Int32(units)
                    base = Int64(s1.toint())
                    col = r
                    if r >= Int32(self.cols):
                        base = Int64(s3.toint())
                        col = r - Int32(self.cols)
                    cp_async4_shared_global(
                        sc + r * Int32(self.sc_stride) + c * Int32(16),
                        base + (Int64(e) * Int64(i) + Int64(n0 + col)) * Int64(self.scale_cols) + Int64(c) * Int64(16))

            for p in cutlass.range_constexpr(self.a_ring - 1):
                self._load_a(p, sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(self.w_ring - 1):
                    self._load_w(p, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g = lane // Int32(4)
            j = lane % Int32(4)
            shift = (Uint32(j) % Uint32(2)) * Uint32(16)
            acc = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(32):
                acc[v] = Float32(0.0)
            for step in cutlass.range(self.k_steps, unroll=1):
                block = step // Int32(2)
                slot = block % Int32(self.w_ring)
                cute.arch.cp_async_wait_group(self.a_ring - 2)
                cute.arch.mbarrier_wait(mbar + slot, (block // Int32(self.w_ring)) & Int32(1))
                cute.arch.sync_threads()
                if step + Int32(self.a_ring - 1) < Int32(self.k_steps):
                    self._load_a(step + Int32(self.a_ring - 1), sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
                if warp_id == Int32(0):
                    # Block `block` starts: every thread is past block - 1, whose slot takes
                    # block + w_ring - 1.
                    if (step % Int32(2) == Int32(0)) & (block + Int32(self.w_ring - 1) < Int32(self.w_blocks)):
                        self._load_w(block + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                if cutlass.const_expr(self.qmma):
                    if active:
                        stage = step % Int32(self.a_ring)
                        a_base = sa + stage * Int32(self.a_bytes)
                        w_base = sw + slot * Int32(2 * self.half_bytes)
                        s_base = ss + stage * Int32(self.s_bytes)
                        half = step % Int32(2)
                        # A scales: row g (lanes with even j) or g + 8 (odd j) of each m16 tile,
                        # this step's four UE8M0 bytes (one per 32-K block, byte ks).
                        asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
                            asc[mt] = ld_shared_u32(s_base + srow * Int32(4))
                        wsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                        for nt in cutlass.range_constexpr(4):
                            col = Int32((nt // 2) * self.cols + (nt % 2) * 8) + warp_n * Int32(16) + g
                            wsc[nt] = ld_shared_u32(sc + col * Int32(self.sc_stride) + step * Int32(4))
                        for ks in cutlass.range_constexpr(4):
                            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                            for mt in cutlass.range_constexpr(2):
                                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                                chunk = Int32(2 * ks) + lane // Int32(16)
                                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                    a_base + row * Int32(self.k_step) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                                a[4 * mt] = a0
                                a[4 * mt + 1] = a1
                                a[4 * mt + 2] = a2
                                a[4 * mt + 3] = a3
                            for nt in cutlass.range_constexpr(4):
                                # Column g's K 4j..4j+3 (bytes 2j, 2j+1) and 16+4j.. (bytes 8+2j, 9+2j)
                                # of the 32-K chunk's 16-byte unit.
                                nrow = warp_n * Int32(16) + Int32((nt % 2) * 8) + g
                                unit = half * Int32(4) + Int32(ks)
                                v0, v1, v2, v3 = ld_shared_v4_u32(
                                    w_base + Int32((nt // 2) * self.half_bytes) + nrow * Int32(self.k_step)
                                    + ((unit ^ (nrow % Int32(8))) * Int32(16)))
                                lo_word = v0
                                hi_word = v2
                                if j >= Int32(2):
                                    lo_word = v1
                                    hi_word = v3
                                packed = ((lo_word >> shift) & Uint32(0xFFFF)) | (((hi_word >> shift) & Uint32(0xFFFF))
                                                                                 << Uint32(16))
                                b0, b1 = e2m1x8_to_qmma_e2m1x8(packed)
                                for mt in cutlass.range_constexpr(2):
                                    f = 4 * (4 * mt + nt)
                                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b0, b1,
                                        asc[mt], wsc[nt], bid_a=ks, bid_b=ks)
                                    acc[f] = d0
                                    acc[f + 1] = d1
                                    acc[f + 2] = d2
                                    acc[f + 3] = d3
                else:
                    if active:
                        stage = step % Int32(self.a_ring)
                        a_base = sa + stage * Int32(self.a_bytes)
                        w_base = sw + slot * Int32(2 * self.half_bytes)
                        s_base = ss + stage * Int32(self.s_bytes)
                        half = step % Int32(2)
                        rsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            for hi in cutlass.range_constexpr(2):
                                srow = warp_m * Int32(32) + Int32(16 * mt + 8 * hi) + g
                                rsc[2 * mt + hi] = ld_shared_u32(s_base + srow * Int32(4))
                        # This step's four UE8M0 weight scales of each column tile nt.
                        wsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                        for nt in cutlass.range_constexpr(4):
                            col = Int32((nt // 2) * self.cols + (nt % 2) * 8) + warp_n * Int32(16) + g
                            wsc[nt] = ld_shared_u32(sc + col * Int32(self.sc_stride) + step * Int32(4))
                        for ks in cutlass.range_constexpr(4):
                            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                            for mt in cutlass.range_constexpr(2):
                                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                                chunk = Int32(2 * ks) + lane // Int32(16)
                                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                    a_base + row * Int32(self.k_step) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                                a[4 * mt] = a0
                                a[4 * mt + 1] = a1
                                a[4 * mt + 2] = a2
                                a[4 * mt + 3] = a3
                            # B of the A permutation: MMA half q of this 32-K chunk takes
                            # physical K 16q + 4j .. +3 of column g, E2M1 bytes 8q + 2j, +1
                            # of the chunk's 16-byte unit (half * 4 + ks of the 128-byte row).
                            wb = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                            for nt in cutlass.range_constexpr(4):
                                nrow = warp_n * Int32(16) + Int32((nt % 2) * 8) + g
                                unit = half * Int32(4) + Int32(ks)
                                v0, v1, v2, v3 = ld_shared_v4_u32(
                                    w_base + Int32((nt // 2) * self.half_bytes) + nrow * Int32(self.k_step)
                                    + ((unit ^ (nrow % Int32(8))) * Int32(16)))
                                lo_word = v0
                                hi_word = v2
                                if j >= Int32(2):
                                    lo_word = v1
                                    hi_word = v3
                                factor = ue8m0_bf16x2((wsc[nt] >> Uint32(8 * ks)) & Uint32(0xFF))
                                b0, b1 = e2m1x4_scaled_bf16((lo_word >> shift) & Uint32(0xFFFF), factor)
                                b2, b3 = e2m1x4_scaled_bf16((hi_word >> shift) & Uint32(0xFFFF), factor)
                                wb[4 * nt] = b0
                                wb[4 * nt + 1] = b1
                                wb[4 * nt + 2] = b2
                                wb[4 * nt + 3] = b3
                            for mt in cutlass.range_constexpr(2):
                                xa = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                                for r in cutlass.range_constexpr(4):
                                    exponent = (rsc[2 * mt + r % 2] >> Uint32(8 * ks)) & Uint32(0xFF)
                                    lo, hi = _e4m3x4_scaled_bf16x2x2(a[4 * mt + r], _u32_as_f32(exponent << Uint32(23)))
                                    xa[2 * r] = lo
                                    xa[2 * r + 1] = hi
                                for q in cutlass.range_constexpr(2):
                                    for nt in cutlass.range_constexpr(4):
                                        f = 4 * (4 * mt + nt)
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                            xa[4 * q], xa[4 * q + 2], xa[4 * q + 1], xa[4 * q + 3],
                                            wb[4 * nt + 2 * q], wb[4 * nt + 2 * q + 1])
                                        acc[f] = d0
                                        acc[f + 1] = d1
                                        acc[f + 2] = d2
                                        acc[f + 3] = d3
            cute.arch.cp_async_wait_group(0)
            if cutlass.const_expr(self.act_mxfp8):
                cute.arch.sync_threads()
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            for nt in cutlass.range_constexpr(2):
                                fg = 4 * (4 * mt + nt) + 2 * half
                                fu = 4 * (4 * mt + nt + 2) + 2 * half
                                col = n0 + warp_n * Int32(16) + Int32(8 * nt) + Int32(2) * j
                                v0 = _swiglu(acc[fg], acc[fu], self.limit)
                                v1 = _swiglu(acc[fg + 1], acc[fu + 1], self.limit)
                                if cutlass.const_expr(self.act_mxfp8):
                                    st_shared_u32(sa + row * Int32(self.act_stride) + (col - n0) * Int32(2),
                                                  pack_f32x2_to_bfloat2(v0, v1))
                                else:
                                    st_global_u32(Int64(act.toint()) + (Int64(tile * Int32(STREAM_TILE_M) + row)
                                                                        * Int64(i) + Int64(col)) * Int64(2),
                                                  pack_f32x2_to_bfloat2(v0, v1))
            if cutlass.const_expr(self.act_mxfp8):
                cute.arch.sync_threads()
                if tidx < live:
                    # A CTA owns exactly 32 intermediate columns, including
                    # the TP6 slice's whole K32 zero-padding blocks.
                    values = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
                    for q in cutlass.range_constexpr(4):
                        words = ld_shared_v4_u32(sa + tidx * Int32(self.act_stride) + Int32(16 * q))
                        for word in cutlass.range_constexpr(4):
                            values[8 * q + 2 * word] = _bf16_lo(words[word])
                            values[8 * q + 2 * word + 1] = _bf16_hi(words[word])
                    payload, scale = quantize_block_fp8_mx(values, max_abs_32(values))
                    at = Int64(act.toint()) + (Int64(tile) * Int64(STREAM_TILE_M) + Int64(tidx)) \
                        * Int64(self.act_row_bytes)
                    for half in cutlass.range_constexpr(2):
                        st_global_v4_u32(at + Int64(n0) + Int64(16 * half),
                                         payload[4 * half], payload[4 * half + 1],
                                         payload[4 * half + 2], payload[4 * half + 3])
                    st_global_u8(at + Int64(i) + Int64(n0 // Int32(32)), cutlass.Uint8(scale))


class StreamMxfp4Down(StreamFp8Down):
    """``y[p] = act_p . bf16(w2_e)^T`` over MXFP4 ``w2 [E, H, I/2]`` with
    UE8M0 ``s2 [E, H, I/32]``; otherwise ``StreamFp8Down``. Weight blocks are
    256 K (128-byte lines) when ``I % 256 == 0``, else 128 K (64 bytes: TP6's
    384); the CTA's 128 scale rows are loaded once."""

    def __init__(self, *, hidden: int, inter: int, experts: int):
        super().__init__(hidden=hidden, inter=inter, experts=experts)
        self.w_k = 256 if self.inter % 256 == 0 else 128
        self.w_row = self.w_k // 2
        self.w_bytes = self.tile_n * self.w_row
        self.per_block = self.w_k // self.k_step
        self.scale_cols = self.inter // 32
        if self.scale_cols % 4:
            raise ValueError("MXFP4 stream down needs I % 128 == 0")
        self.sc_stride = self.scale_cols + 4

    def key(self) -> tuple:
        return ("stream_down_mxfp4", 1, self.hidden, self.inter, self.experts, self.ring, self.w_k)

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.k_step),
            cutlass.BFloat16)
        a_layout = cute.tile_to_shape(atom, (STREAM_TILE_M, self.k_step, self.ring), order=(0, 1, 2))
        w_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.w_row), cutlass.Uint8)
        w_layout = cute.tile_to_shape(w_atom, (self.tile_n, self.w_row, 2), order=(0, 1, 2))
        return a_layout, w_layout

    def _storage(self, a_layout, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.ring + 2],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout)], 1024],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "b": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.b_bytes], 1024],
            "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.tile_n * self.sc_stride], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, act: cute.Pointer, meta: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        a_layout, w_layout = self._layouts()
        act_t = cute.make_tensor(act, cute.make_layout((max_tiles * Int32(STREAM_TILE_M), self.inter),
                                                       stride=(self.inter, 1)))
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), act_t, cute.slice_(a_layout, (None, None, 0)),
            (STREAM_TILE_M, self.k_step), num_multicast=1)
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w2, self.experts * self.hidden, self.inter // 2),
            cute.slice_(w_layout, (None, None, 0)), (self.tile_n, self.w_row), num_multicast=1)
        self.kernel(meta, tma_tensor_a, tma_tensor_w, s2, y, tma_a, tma_w, a_layout, w_layout,
                    self._storage(a_layout, w_layout)).launch(
            grid=(self.hidden // self.tile_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, meta: cute.Pointer, act: cute.Tensor, weight: cute.Tensor, s2: cute.Pointer, y: cute.Pointer,
               tma_a: cute.CopyAtom, tma_w: cute.CopyAtom, a_layout: cute.ComposedLayout,
               w_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            mbar = storage.mbar.data_ptr()
            s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            wbar = mbar + self.ring
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sb = shared_ptr_to_u32(storage.b.data_ptr())
            sc = shared_ptr_to_u32(storage.sc.data_ptr())
            h = self.hidden
            w_tile = e * Int32(h // self.tile_n) + n_blk
            t_as, t_ag = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                                               cute.group_modes(cute.local_tile(act, (STREAM_TILE_M, self.k_step),
                                                                                (None, None)), 0, 2))
            t_ws, t_wg = cpasync.tma_partition(tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight, (self.tile_n, self.w_row),
                                                                                (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_a)
                cpasync.prefetch_descriptor(tma_w)
                for slot in cutlass.range_constexpr(self.ring + 2):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            # This block's 128 scale rows (I/32 bytes each), once.
            words = self.scale_cols // 4
            for q in cutlass.range_constexpr((self.tile_n * words + self.threads - 1) // self.threads):
                u = tidx + Int32(q * self.threads)
                if u < Int32(self.tile_n * words):
                    r = u // Int32(words)
                    c = u % Int32(words)
                    cp_async_u32_shared_global(
                        sc + r * Int32(self.sc_stride) + c * Int32(4),
                        Int64(s2.toint()) + (Int64(e) * Int64(h) + Int64(n_blk * Int32(self.tile_n) + r))
                        * Int64(self.scale_cols) + Int64(c) * Int64(4))
            cute.arch.cp_async_commit_group()
            blocks = self.k_steps // self.per_block
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(min(2, blocks)):
                    self._load_w(p, tma_w, t_wg, t_ws, wbar, w_tile)
                for p in cutlass.range_constexpr(self.ring - 1):
                    if Int32(p) < Int32(self.k_steps):
                        self._load_a(p, tma_a, t_ag, t_as, mbar, tile)
            cute.arch.cp_async_wait_group(0)
            cute.arch.mbarrier_wait(wbar, 0)
            cute.arch.sync_threads()
            self._widen(0, sw, sb, sc, tidx)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g = lane // Int32(4)
            j = lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)
            for step in cutlass.range(self.k_steps, unroll=1):
                nxt = step + Int32(1)
                block = nxt // Int32(self.per_block)
                starts = (nxt < Int32(self.k_steps)) & (nxt % Int32(self.per_block) == Int32(0))
                cute.arch.mbarrier_wait(mbar + step % Int32(self.ring), (step // Int32(self.ring)) & Int32(1))
                if starts:
                    cute.arch.mbarrier_wait(wbar + block % Int32(2), (block // Int32(2)) & Int32(1))
                cute.arch.sync_threads()
                if warp_id == Int32(0):
                    if step + Int32(self.ring - 1) < Int32(self.k_steps):
                        self._load_a(step + Int32(self.ring - 1), tma_a, t_ag, t_as, mbar, tile)
                    if starts & (block + Int32(1) < Int32(blocks)):
                        self._load_w(block + Int32(1), tma_w, t_wg, t_ws, wbar, w_tile)
                if nxt < Int32(self.k_steps):
                    self._widen(nxt, sw, sb, sc, tidx)
                if active:
                    a_base = sa + (step % Int32(self.ring)) * Int32(self.a_bytes)
                    b_base = sb + (step % Int32(2)) * Int32(self.b_bytes)
                    for kk in cutlass.range_constexpr(2):
                        a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                        for mt in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                            chunk = Int32(2 * kk) + lane // Int32(16)
                            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                a_base + row * Int32(64) + ((chunk ^ ((row >> Int32(1)) & Int32(3))) * Int32(16)))
                            a[4 * mt] = a0
                            a[4 * mt + 1] = a1
                            a[4 * mt + 2] = a2
                            a[4 * mt + 3] = a3
                        for p in cutlass.range_constexpr(4):
                            nrow = warp_n * Int32(64) + Int32(16 * p) + (lane & Int32(7)) \
                                + (lane >> Int32(4)) * Int32(8)
                            chunk = Int32(2 * kk) + ((lane >> Int32(3)) & Int32(1))
                            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                                b_base + nrow * Int32(64) + ((chunk ^ ((nrow >> Int32(1)) & Int32(3))) * Int32(16)))
                            for mt in cutlass.range_constexpr(2):
                                for q in cutlass.range_constexpr(2):
                                    f = 4 * (8 * mt + 2 * p + q)
                                    lo = b0 if q == 0 else b2
                                    hi = b1 if q == 0 else b3
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], lo, hi)
                                    acc[f] = d0
                                    acc[f + 1] = d1
                                    acc[f + 2] = d2
                                    acc[f + 3] = d3
            cute.arch.sync_threads()
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                            st_shared_u32(sa + row * Int32(self.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
            cute.arch.sync_threads()
            n0 = n_blk * Int32(self.tile_n)
            segments = self.tile_n * 2 // 16
            for q in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
                c = tidx + Int32(self.threads * q)
                row = c // Int32(segments)
                seg = c % Int32(segments)
                if row < live:
                    v0, v1, v2, v3 = ld_shared_v4_u32(sa + row * Int32(self.out_stride) + seg * Int32(16))
                    st_global_v4_u32(Int64(y.toint()) + (Int64(first + row) * Int64(h) + Int64(n0)) * Int64(2)
                                     + Int64(seg) * Int64(16), v0, v1, v2, v3)

    @cute.jit
    def _widen(self, step, sw, sb, sc, tidx):
        block = Int32(step) // Int32(self.per_block)
        piece = Int32(step) % Int32(self.per_block)
        w_stage = sw + (block % Int32(2)) * Int32(self.w_bytes)
        b_stage = sb + (Int32(step) % Int32(2)) * Int32(self.b_bytes)
        # One UE8M0 scale per 32-K step and row: byte `step` of the row.
        word_at = (Int32(step) // Int32(4)) * Int32(4)
        byte_shift = Uint32(Int32(step) % Int32(4)) * Uint32(8)
        for q in cutlass.range_constexpr(self.tile_n * self.k_step // 8 // self.threads):
            c = tidx + Int32(self.threads * q)
            n = c // Int32(4)
            chunk = c % Int32(4)
            # Eight E2M1 codes (4 bytes) of the row's 16-byte unit `piece` (swizzled 128B / 64B).
            if cutlass.const_expr(self.w_row == 128):
                unit = piece ^ (n % Int32(8))
            else:
                unit = piece ^ ((n >> Int32(1)) & Int32(3))
            codes = ld_shared_u32(w_stage + n * Int32(self.w_row) + unit * Int32(16) + chunk * Int32(4))
            exponent = (ld_shared_u32(sc + n * Int32(self.sc_stride) + word_at) >> byte_shift) & Uint32(0xFF)
            v0, v1, v2, v3 = e2m1x8_scaled_bf16(codes, ue8m0_bf16x2(exponent))
            st_shared_v4_u32(b_stage + n * Int32(64) + ((chunk ^ ((n >> Int32(1)) & Int32(3))) * Int32(16)),
                             v0, v1, v2, v3)
