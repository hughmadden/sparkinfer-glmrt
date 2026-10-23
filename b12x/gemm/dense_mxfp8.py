"""Shared MXFP8 activation records and native dense prefill weight loaders."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from b12x._lib.intrinsics import (
    broadcast_f32_to_half2,
    bfloat2_habs2, bfloat2_hmax2, bfloat2_hmax_to_f32,
    bfloat2_to_float2_scaled, cvt_f32x4_to_e4m3x4,
    habs2,
    half2_mul,
    hmax2,
    hmax_to_f32,
    ld_global_nc_u32,
    ld_global_nc_u8,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ldmatrix_m8n8x4_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_dequant_trellis_to_half2x4,
    pow2_ceil_ue8m0,
    shared_ptr_to_u32,
    st_global_u8,
    st_global_v2_f32,
    st_global_v4_u32,
    st_shared_u16,
    st_shared_u32,
    ue8m0_to_output_scale,
)
from b12x.moe._shared.kernels.w4a16.exlr8 import _side_value

M_CTA = 128
NWARP = 8
K16_PER_BLOCK = 8
E4M3_MAX = 448.0


def _e4m3x4_from_half2x2(lo: Uint32, hi: Uint32) -> Uint32:
    """Round two half2 fragment registers to E4M3 and pack four bytes.

    ``lo`` supplies bytes 0-1 and ``hi`` bytes 2-3, exactly the order
    ``fp8x4_e4m3_to_half2x2`` reads back. Rounding is what the native kernel
    already does (``_e4m3_round_half2``); a decoded trellis value is an exact
    E4M3 number, so this is a no-op on it and bitwise for the F16 consumer.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(), Uint32(hi).ir_value()],
            """
            {
                .reg .b16 e0, e1;
                cvt.rn.satfinite.e4m3x2.f16x2 e0, $1;
                cvt.rn.satfinite.e4m3x2.f16x2 e1, $2;
                mov.b32 $0, {e0, e1};
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _load_words4(address: Int64, tile: Int32, lane: Int32):
    """One K4 payload tile (128 B): the lane word and the rotated neighbour."""
    base = address + tile.to(Int64) * Int64(128)
    b = ld_global_nc_u32(base + lane.to(Int64) * Int64(4))
    a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
    return b, (a << Uint32(16)) | (b >> Uint32(16))


# ---------------------------------------------------------------- MXFP8 records
# One record per (m16 row block, k32): 32 lanes x 16 fragment bytes plus the 16
# per-row E8M0 scale bytes. The 16 fragment bytes at lane ``l`` are the A
# operand of ``mxfp8_mma_m16n8k32_f32_e4m3`` in register order: rows l/4 and
# l/4+8, k = 4*(l&3)..+3 and 16+4*(l&3)..+3. A record is read back with one
# 128-bit lane load plus one scale byte, so no activation ever touches smem and
# every column CTA shares the same quantize pass.
MX8_REC = 528


class TrellisMxfp8Quant:
    """Quantize F16 activations into MXFP8 records (one pass, all column CTAs).

    Per-32-k block scales are E8M0 (``pow2_ceil_ue8m0`` of the block amax over
    448, the exact FlashInfer/OCP rule), so the MMA's ``scale_vec::1X`` operand
    carries them with no in-kernel per-block fold.
    """

    def __init__(self, g, *, dbg: bool = False):
        self.k = g["input_features"]
        if self.k % 128:
            raise ValueError("MXFP8 records need K a multiple of 128")
        self.nk32 = self.k // 32
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        self.dbg = dbg
        self.bf16 = g.get("input_dtype", "f16") == "bf16"

    @cute.kernel
    def quantize(self, inputs: cute.Tensor, records: cute.Tensor,
                 rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        x = inputs[rotation_id]
        rbase = records.iterator.toint()
        n_rb = (rows + Int32(15)) // Int32(16)
        total = n_rb * Int32(self.nk32)
        task = task * Int32(NWARP) + warp
        stride = step * Int32(NWARP)
        row_last = Float32(rows - Int32(1))
        while task < total:
            rb = task // Int32(self.nk32)
            s = task % Int32(self.nk32)
            r0 = rb * Int32(16) + (lane >> Int32(2))
            c = (lane & Int32(3)) * Int32(4)
            klo = s * Int32(32) + c
            khi = klo + Int32(16)
            r0c = Int32(cute.arch.fmin(Float32(r0), row_last))
            r1c = Int32(cute.arch.fmin(Float32(r0 + Int32(8)), row_last))
            x0 = x + r0c.to(Int64) * Int64(2 * self.k)
            x1 = x + r1c.to(Int64) * Int64(2 * self.k)
            # Two rows x two 4-value groups each: a lane holds 16 values.
            f00, f01 = ld_global_nc_v2_u32(x0 + klo.to(Int64) * Int64(2))
            f10, f11 = ld_global_nc_v2_u32(x1 + klo.to(Int64) * Int64(2))
            f20, f21 = ld_global_nc_v2_u32(x0 + khi.to(Int64) * Int64(2))
            f30, f31 = ld_global_nc_v2_u32(x1 + khi.to(Int64) * Int64(2))
            if cutlass.const_expr(self.bf16):
                m0 = bfloat2_hmax_to_f32(bfloat2_hmax2(
                    bfloat2_hmax2(bfloat2_habs2(f00), bfloat2_habs2(f01)),
                    bfloat2_hmax2(bfloat2_habs2(f20), bfloat2_habs2(f21))))
                m1 = bfloat2_hmax_to_f32(bfloat2_hmax2(
                    bfloat2_hmax2(bfloat2_habs2(f10), bfloat2_habs2(f11)),
                    bfloat2_hmax2(bfloat2_habs2(f30), bfloat2_habs2(f31))))
            else:
                m0 = hmax_to_f32(hmax2(hmax2(habs2(f00), habs2(f01)),
                                       hmax2(habs2(f20), habs2(f21))))
                m1 = hmax_to_f32(hmax2(hmax2(habs2(f10), habs2(f11)),
                                       hmax2(habs2(f30), habs2(f31))))
            # The 32-k block lives in the four lanes of this quad.
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=1))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=1))
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=2))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=2))
            _, code0 = pow2_ceil_ue8m0(m0 * Float32(1.0 / E4M3_MAX))
            _, code1 = pow2_ceil_ue8m0(m1 * Float32(1.0 / E4M3_MAX))
            if cutlass.const_expr(self.bf16):
                v = cute.make_rmem_tensor((4,), Uint32)
                for i in cutlass.range_constexpr(4):
                    lo, hi = f00, f01
                    code = code0
                    if cutlass.const_expr(i == 1):
                        lo, hi, code = f10, f11, code1
                    if cutlass.const_expr(i == 2):
                        lo, hi = f20, f21
                    if cutlass.const_expr(i == 3):
                        lo, hi, code = f30, f31, code1
                    inv = ue8m0_to_output_scale(code)
                    a, b = bfloat2_to_float2_scaled(lo, inv)
                    c0, d = bfloat2_to_float2_scaled(hi, inv)
                    v[i] = cvt_f32x4_to_e4m3x4(a, b, c0, d)
                w0, w1, w2, w3 = v[0], v[1], v[2], v[3]
            else:
                h0 = broadcast_f32_to_half2(ue8m0_to_output_scale(code0))
                h1 = broadcast_f32_to_half2(ue8m0_to_output_scale(code1))
                w0 = _e4m3x4_from_half2x2(half2_mul(f00, h0), half2_mul(f01, h0))
                w1 = _e4m3x4_from_half2x2(half2_mul(f10, h1), half2_mul(f11, h1))
                w2 = _e4m3x4_from_half2x2(half2_mul(f20, h0), half2_mul(f21, h0))
                w3 = _e4m3x4_from_half2x2(half2_mul(f30, h1), half2_mul(f31, h1))
            rec = rbase + (rb * Int32(self.nk32) + s).to(Int64) * Int64(MX8_REC)
            st_global_v4_u32(rec + lane.to(Int64) * Int64(16), w0, w1, w2, w3)
            row_lo = (lane >> Int32(2)).to(Int64)
            st_global_u8(rec + Int64(512) + row_lo, Uint8(code0))
            st_global_u8(rec + Int64(520) + row_lo, Uint8(code1))
            if cutlass.const_expr(self.dbg):
                if (lane & Int32(3)) == Int32(0):
                    dbase = (rbase + total.to(Int64) * Int64(MX8_REC)
                             + (rb * Int32(self.nk32) + s).to(Int64) * Int64(64))
                    st_global_v2_f32(dbase + row_lo * Int64(8), m0, m1)
            task += stride

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, records: cute.Pointer,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations)
                and cta_limit > Int32(0)):
            layout = cute.make_layout(1, stride=1)
            tasks = ((rows + Int32(15)) // Int32(16)) * Int32(self.nk32)
            warps = (tasks + Int32(NWARP - 1)) // Int32(NWARP)
            self.quantize(cute.make_tensor(rotated_inputs, layout),
                          cute.make_tensor(records, layout),
                          rows, rotation_id).launch(
                grid=(min(warps, cta_limit), 1, 1), block=(32 * NWARP, 1, 1),
                stream=stream)


class TrellisPrefillMxfp8:
    """Many-row trellis dense linear on MXFP8 records.

    ``TrellisMxfp8Quant`` writes the records into the ABI's ``partial`` scratch
    (``rows*K`` bytes of the 4-byte-per-element split-K scratch, so no new
    allocation); this kernel then runs the decode-once E4M3 tile in smem in
    ``ldmatrix`` order -- ``[n][k128]`` bytes with the k16 group XOR-swizzled by
    ``n & 7`` -- and feeds ``mma.kind::mxf8f6f4.block_scale.scale_vec::1X``
    m16n8k32 with the activation's E8M0 scale byte and B at E8M0 127.

    ``part`` selects the pass to launch: ``both`` (the whole route), ``quant``
    or ``gemm`` (the two halves, for separate timing).
    """

    ABI_VERSION = 1

    def __init__(self, g, *, n_cta: int = 64, stages: int = 2,
                 part: str = "gemm", dbg: bool = False, mpw: int = 2):
        self.k, self.n = g["input_features"], g["output_features"]
        self.encoding = g.get("encoding", "trellis")
        self.bits = g.get("bits", 4)
        if self.bits != 4:
            raise ValueError("the T1 prefill kernel covers K4 trellis only")
        if g.get("codebook", "e4m3") != "e4m3":
            raise ValueError("the T1 E4M3 tile covers the e4m3 codebook only")
        if n_cta != 64 or self.n % 32:
            raise ValueError("MX8 prefill requires N32 alignment and n_cta=64")
        if n_cta % 16 or (n_cta // 8) % 2:
            raise ValueError("n_cta must cover whole pairs of n8 tiles")
        if part not in ("both", "quant", "gemm"):
            raise ValueError("part must be both, quant or gemm")
        self.input_units = g["sharded_side"] == 2
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        if mpw not in (1, 2):
            raise ValueError("mpw must be 1 or 2")
        self.mpw, self.m_cta = mpw, M_CTA * mpw
        self.scratch_bytes = ((self.max_rows + 15) // 16) * (self.k // 32) * MX8_REC
        self.n_cta, self.stages, self.part = n_cta, stages, part
        self.nk32 = self.k // 32
        self.n16 = n_cta // 16
        self.n8w = n_cta // 8
        self.quant = TrellisMxfp8Quant(g, dbg=dbg)

    def _storage(self):
        # E4M3 B tile: stages x n_cta rows of 128 k-bytes.
        words = self.stages * 32 * self.n_cta

        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[Uint32, words], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def _tile_index(self, nb: Int32, kb: Int32, k16: Int32, n16_local: Int32) -> Int32:
        n16g = nb * Int32(self.n_cta // 16) + n16_local
        group = n16g // Int32(8)
        slot = n16g % Int32(8)
        if cutlass.const_expr(self.input_units):
            return (kb * Int32(self.n // 16 * 8) + group * Int32(64)
                    + k16 * Int32(8) + slot)
        return (group * Int32(self.k // 16 * 8)
                + (kb * Int32(8) + k16) * Int32(8) + slot)

    @cute.jit
    def _stage(self, sbase: Int32, stage: Int32, k16: Int32, n16: Int32,
               lane: Int32, wa: Uint32, wb: Uint32):
        """Decode one K16 x N16 tile into the ldmatrix-order byte tile.

        Byte address of (n, k) inside a stage: ``n*128 + ((k>>4) ^ (n&7))*16 +
        (k&15)``. Every lane of an ``ldmatrix.x4`` reads one 16-byte group, and
        the 8 rows it reads (n = lane&7) land on 8 different bank quads.
        """
        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, self.bits)
        n0 = n16 * Int32(16) + (lane >> Int32(2))
        kb_off = stage * Int32(128 * self.n_cta) + (lane & Int32(3)) * Int32(2)
        g = k16 ^ ((lane >> Int32(2)) & Int32(7))
        for s in cutlass.range_constexpr(2):
            if cutlass.const_expr(s == 0):
                e = _e4m3x4_from_half2x2(b0, b1)
                n = n0
            else:
                e = _e4m3x4_from_half2x2(b2, b3)
                n = n0 + Int32(8)
            base = sbase + n * Int32(128) + (g << 4) + kb_off
            st_shared_u16(base, e)
            st_shared_u16(base + Int32(8), e >> 16)

    @cute.jit
    def _native_stage(self, sbase: Int32, stage: Int32, nb: Int32, kb: Int32,
                      warp: Int32, lane: Int32, payload: Int64):
        for j in cutlass.range_constexpr(self.n16):
            n = Int32(16 * j) + (lane >> Int32(1))
            ng = min(nb * Int32(self.n_cta) + n, Int32(self.n-1))
            k = warp * Int32(16) + (lane & Int32(1)) * Int32(8)
            offset = ng.to(Int64) * Int64(self.k) + kb.to(Int64) * Int64(128) + k.to(Int64)
            if cutlass.const_expr(self.input_units):
                offset = ((kb.to(Int64) * Int64(self.n) + ng.to(Int64)) * Int64(128)
                          + k.to(Int64))
            if cutlass.const_expr(self.encoding == "int4_g32"):
                q = ld_global_nc_u32(payload + offset // Int64(2))
                e = cute.make_rmem_tensor((2,), Uint32)
                for h in cutlass.range_constexpr(2):
                    v = cute.make_rmem_tensor((4,), Float32)
                    for t in cutlass.range_constexpr(4):
                        v[t] = ((q >> Uint32(16*h + 4*t)) & Uint32(15)).to(Float32) - Float32(8)
                    e[h] = cvt_f32x4_to_e4m3x4(v[0], v[1], v[2], v[3])
                lo, hi = e[0], e[1]
            else:
                lo, hi = ld_global_nc_v2_u32(payload + offset)
            at = (sbase + stage * Int32(128*self.n_cta) + n * Int32(128)
                  + ((warp ^ (n & Int32(7))) << 4) + (lane & Int32(1))*Int32(8))
            st_shared_u32(at, lo)
            st_shared_u32(at + Int32(4), hi)

    @cute.jit
    def _scale_index(self, nb: Int32, kb: Int32, col: Int32, k32: Int32):
        n = min(nb * Int32(self.n_cta) + col, Int32(self.n-1))
        index = n.to(Int64) * Int64(self.nk32) + kb.to(Int64)*Int64(4) + k32.to(Int64)
        if cutlass.const_expr(self.input_units):
            index = (kb.to(Int64)*Int64(self.n) + n.to(Int64))*Int64(4) + k32.to(Int64)
        return index

    @cute.kernel
    def prefill(self, inputs: cute.Tensor, payload: cute.Tensor,
                scales: cute.Tensor, records: cute.Tensor, output: cute.Tensor,
                rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        wbase = payload.iterator.toint()
        abase = records.iterator.toint()
        obase = output.iterator.toint()
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sbase = shared_ptr_to_u32(storage.w.data_ptr())
        n_tiles = Int32((self.n + self.n_cta-1) // self.n_cta)
        nkb = self.k // 128
        total_tasks = ((rows + Int32(self.m_cta - 1)) // Int32(self.m_cta)) * n_tiles
        row_last = Float32(rows - Int32(1))
        # B-fragment ldmatrix rows: matrix m = lane>>3 selects (n8 of the pair,
        # k16 half); every lane addresses column n = (m>>1)*8 + (lane&7).
        blk = Int32((lane >> Int32(3)) & Int32(1))
        bcol = (lane >> Int32(4)) * Int32(8) + (lane & Int32(7))
        b_row_base = sbase + bcol * Int32(128)
        # sfa byte: row l/4 for even lanes, row l/4+8 for odd lanes.
        srow = (lane >> Int32(2)) + (lane & Int32(1)) * Int32(8)
        while task < total_tasks:
            nb = task % n_tiles
            m_base = (task // n_tiles) * Int32(self.m_cta)
            row = m_base + warp * Int32(16 * self.mpw) + (lane >> Int32(2))
            rc = Int32(cute.arch.fmin(Float32(row), row_last))
            total = cute.make_rmem_tensor((self.mpw, self.n8w, 4), Float32)
            total.fill(Float32(0.0))
            mb = ((task // n_tiles) * Int32(NWARP) + warp) * Int32(self.mpw)
            krec = Int64(4 * MX8_REC)
            words = cute.make_rmem_tensor((2 * self.n16,), Uint32)
            if cutlass.const_expr(self.encoding == "trellis"):
                for j in cutlass.range_constexpr(self.n16):
                    tile = self._tile_index(nb, Int32(0), warp, Int32(j))
                    wa, wb = _load_words4(wbase, tile, lane)
                    self._stage(sbase, Int32(0), warp, Int32(j), lane, wa, wb)
            else:
                self._native_stage(sbase, Int32(0), nb, Int32(0), warp, lane, wbase)
            cute.arch.sync_threads()
            for kb in cutlass.range(nkb, unroll=1):
                if cutlass.const_expr(self.encoding == "trellis"):
                    if kb + Int32(1) < Int32(nkb):
                        for j in cutlass.range_constexpr(self.n16):
                            tile = self._tile_index(nb, kb + Int32(1), warp, Int32(j))
                            wa, wb = _load_words4(wbase, tile, lane)
                            words[2 * j], words[2 * j + 1] = wa, wb
                acc = cute.make_rmem_tensor((self.mpw, self.n8w, 4), Float32)
                acc.fill(Float32(0.0))
                stage = (kb % Int32(self.stages)) * Int32(128 * self.n_cta)
                for k32 in cutlass.range_constexpr(4):
                    af = cute.make_rmem_tensor((self.mpw, 5), Uint32)
                    for m in cutlass.range_constexpr(self.mpw):
                        # A partial CTA may have inactive M16 consumers. Reuse the
                        # final record for those consumers; their stores are masked.
                        live_mb = min(mb + Int32(m), (rows - Int32(1)) // Int32(16))
                        rec = (abase + (live_mb * Int32(self.nk32)).to(Int64) * Int64(MX8_REC)
                               + kb.to(Int64) * krec + Int64(k32 * MX8_REC))
                        a0, a1, a2, a3 = ld_global_nc_v4_u32(rec + lane.to(Int64) * Int64(16))
                        af[m, 0], af[m, 1], af[m, 2], af[m, 3] = a0, a1, a2, a3
                        af[m, 4] = ld_global_nc_u8(rec + Int64(512) + srow.to(Int64))
                    for p in cutlass.range_constexpr(self.n8w // 2):
                        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                            b_row_base + stage + Int32(2048 * p)
                            + (((Int32(2 * k32) + blk) ^ (lane & Int32(7))) << 4))
                        # Both M16 blocks consume this B fragment before it dies.
                        for m in cutlass.range_constexpr(self.mpw):
                            i = 2 * p
                            sb0, sb1 = Uint32(0x7F), Uint32(0x7F)
                            if cutlass.const_expr(self.encoding == "mxfp8"):
                                sb0 = ld_global_nc_u8(scales.iterator.toint() + self._scale_index(
                                    nb, kb, Int32(16*p) + (lane >> Int32(2)), Int32(k32)))
                                sb1 = ld_global_nc_u8(scales.iterator.toint() + self._scale_index(
                                    nb, kb, Int32(16*p+8) + (lane >> Int32(2)), Int32(k32)))
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                acc[m, i, 0], acc[m, i, 1], acc[m, i, 2], acc[m, i, 3],
                                af[m, 0], af[m, 1], af[m, 2], af[m, 3], b0, b1, af[m, 4], sb0)
                            acc[m, i, 0], acc[m, i, 1], acc[m, i, 2], acc[m, i, 3] = d0, d1, d2, d3
                            i = 2 * p + 1
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                acc[m, i, 0], acc[m, i, 1], acc[m, i, 2], acc[m, i, 3],
                                af[m, 0], af[m, 1], af[m, 2], af[m, 3], b2, b3, af[m, 4], sb1)
                            acc[m, i, 0], acc[m, i, 1], acc[m, i, 2], acc[m, i, 3] = d0, d1, d2, d3
                    if cutlass.const_expr(self.encoding == "int4_g32"):
                        for m in cutlass.range_constexpr(self.mpw):
                            for r in cutlass.range_constexpr(self.n8w):
                                col = Int32(8*r) + (lane & Int32(3))*Int32(2)
                                idx = self._scale_index(nb, kb, col, Int32(k32))
                                s0 = _side_value(scales.iterator.toint(), idx)
                                s1 = _side_value(scales.iterator.toint(), idx + Int64(4 if self.input_units else self.nk32))
                                for v in cutlass.range_constexpr(4):
                                    factor = s0 if v % 2 == 0 else s1
                                    total[m, r, v] = total[m, r, v] + acc[m, r, v]*factor
                        acc.fill(Float32(0))
                if cutlass.const_expr(self.encoding != "int4_g32"):
                    scale = Float32(1)
                    if cutlass.const_expr(self.encoding == "trellis"):
                        scale = _side_value(scales.iterator.toint(), kb)
                    for m in cutlass.range_constexpr(self.mpw):
                        for r in cutlass.range_constexpr(self.n8w):
                            for v in cutlass.range_constexpr(4):
                                total[m, r, v] = total[m, r, v] + acc[m, r, v]*scale
                if kb + Int32(1) < Int32(nkb):
                    if cutlass.const_expr(self.encoding == "trellis"):
                        for j in cutlass.range_constexpr(self.n16):
                            self._stage(sbase, (kb + Int32(1)) % Int32(self.stages), warp,
                                        Int32(j), lane, words[2*j], words[2*j+1])
                    else:
                        self._native_stage(sbase, (kb + Int32(1)) % Int32(self.stages),
                                           nb, kb + Int32(1), warp, lane, wbase)
                cute.arch.sync_threads()
            for m in cutlass.range_constexpr(self.mpw):
                mr = row + Int32(m * 16)
                for i in cutlass.range_constexpr(self.n8w):
                    col = nb * Int32(self.n_cta) + i * Int32(8) + (lane & Int32(3)) * Int32(2)
                    addr = obase + (mr.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                    if col < Int32(self.n):
                        if mr < rows:
                            st_global_v2_f32(addr, total[m, i, 0], total[m, i, 1])
                        if mr + Int32(8) < rows:
                            st_global_v2_f32(addr + Int64(8 * self.n * 4), total[m, i, 2], total[m, i, 3])
            task += step

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 input_block_scales: cute.Pointer, partial: cute.Pointer,
                 output: cute.Pointer, rows: Int32, rotation_id: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        # ``partial`` (the ABI's split-K scratch, 4 B/element) carries the MXFP8
        # records (K*rows bytes) for every column CTA: it is at least
        # ceil(rows/16)*(K/32)*528 B whenever N >= 264.
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations)
                and cta_limit > Int32(0)):
            if self.part in ("both", "quant"):
                self.quant(rotated_inputs, partial, rows, rotation_id,
                           cta_limit, stream)
            if self.part in ("both", "gemm"):
                layout = cute.make_layout(1, stride=1)
                tasks = ((rows + Int32(self.m_cta - 1)) // Int32(self.m_cta)) * Int32((self.n + self.n_cta-1) // self.n_cta)
                self.prefill(cute.make_tensor(rotated_inputs, layout),
                             cute.make_tensor(payload, layout),
                             cute.make_tensor(input_block_scales, layout),
                             cute.make_tensor(partial, layout),
                             cute.make_tensor(output, layout),
                             rows, rotation_id).launch(
                    grid=(min(tasks, cta_limit), 1, 1), block=(32 * NWARP, 1, 1),
                    stream=stream)
