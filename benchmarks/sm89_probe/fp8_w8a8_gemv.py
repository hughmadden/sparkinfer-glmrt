"""Probe kernel: W8A8 E4M3 decode GEMV on the plain m16n8k32 e4m3 MMA (SM89+).

Rows 1..16 of E4M3 activations against an ``[N, K]`` E4M3 weight. A CTA owns
eight output columns, its warps split K and fold through shared memory.

* ``block32=False``: plain per-tensor FP8, the MMA accumulates in place and
  one FP32 ``alpha`` scales the result.
* ``block32=True``: 32x32 UE8M0 weight blocks (scale bytes ``[N/32, K/32]``,
  K contiguous). Every K32 MMA runs into a zero partial that is multiplied by
  ``2^(e - 127)`` and added to the FP32 accumulator (exact: power-of-two
  scale, IEEE FP32 add), then ``alpha`` (the activation scale) at the end.

K-slot permutation inside a K32 step: lane ``j`` of a quad reads the eight
contiguous bytes ``[8j, 8j + 8)`` of its row and places bytes 0-3 in MMA
slots ``4j..4j+3`` and bytes 4-7 in slots ``16+4j..``, for activations and
weights alike, so each MMA still sums exactly one 32-wide K block.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import ld_global_nc_u32, mma_m16n8k32_f32_e4m3, u32_as_f32


_W_LOAD = {
    "stream": "ld.global.nc.L1::no_allocate.v2.u32 {$0, $1}, [$2];",
    "l2_128b": "ld.global.nc.L1::no_allocate.L2::128B.v2.u32 {$0, $1}, [$2];",
    "l2_256b": "ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {$0, $1}, [$2];",
}


def _ld_w_stream_v2(address, hint="stream"):
    return _ld_w_v2(address, _W_LOAD[hint])


@dsl_user_op
def _ld_w_v2(address, asm, *, loc=None, ip=None):
    """8-byte weight load that does not allocate in L1 (read once)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,=r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _ld_x_cached_v2(address, *, loc=None, ip=None):
    """8-byte activation load through L1 (every CTA re-reads the rows)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class Fp8W8A8Gemv:
    def __init__(self, n: int, k: int, *, warps: int = 4, block32: bool = False, hint: str = "stream"):
        self.n, self.k, self.warps, self.block32 = int(n), int(k), int(warps), bool(block32)
        self.hint = hint
        if self.n % 32 or self.k % (128 * self.warps):
            raise ValueError("needs N % 32 == 0 and K a multiple of 128 * warps")
        self.k_per_warp = self.k // self.warps
        self.blocks = self.k_per_warp // 128  # four K32 steps per loop trip

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * 4 * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, alpha: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, w, scale, alpha, out, rows).launch(
            grid=(self.n // 8, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, alpha: cute.Pointer,
               out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        n0 = Int64(cute.arch.block_idx()[0]) * Int64(8)
        k_begin = Int64(warp_id) * Int64(self.k_per_warp)
        lane_k = k_begin + Int64(8) * Int64(j)
        w_row = Int64(w.toint()) + (n0 + Int64(g)) * Int64(self.k) + lane_k
        x_lo = Int64(x.toint()) + Int64(g) * Int64(self.k) + lane_k
        x_hi = x_lo + Int64(8 * self.k)
        lo_live = g < rows
        hi_live = g + Int32(8) < rows
        # UE8M0 bytes of this CTA's 32-row block; one 32-bit word = 4 K32 steps.
        s_row = Int64(scale.toint()) + (n0 // Int64(32)) * Int64(self.k // 32) + k_begin // Int64(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * 4 * 32,)))
        acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(4):
            acc[i] = Float32(0.0)
        cur = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
        nxt = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
        for s in cutlass.range_constexpr(4):
            cur[2 * s], cur[2 * s + 1] = _ld_w_stream_v2(w_row + Int64(32 * s), self.hint)
        s_cur = Uint32(0)
        s_nxt = Uint32(0)
        if const_expr(self.block32):
            s_cur = ld_global_nc_u32(s_row)
        for block in cutlass.range(self.blocks, unroll=1):
            if block + 1 < self.blocks:
                for s in cutlass.range_constexpr(4):
                    nxt[2 * s], nxt[2 * s + 1] = _ld_w_stream_v2(
                        w_row + Int64(block + 1) * Int64(128) + Int64(32 * s), self.hint)
                if const_expr(self.block32):
                    s_nxt = ld_global_nc_u32(s_row + Int64(block + 1) * Int64(4))
            for s in cutlass.range_constexpr(4):
                k_off = Int64(block) * Int64(128) + Int64(32 * s)
                a0 = Uint32(0)
                a2 = Uint32(0)
                a1 = Uint32(0)
                a3 = Uint32(0)
                if lo_live:
                    a0, a2 = _ld_x_cached_v2(x_lo + k_off)
                if hi_live:
                    a1, a3 = _ld_x_cached_v2(x_hi + k_off)
                if const_expr(self.block32):
                    zero = Float32(0.0)
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        zero, zero, zero, zero, a0, a1, a2, a3, cur[2 * s], cur[2 * s + 1])
                    sc = u32_as_f32(((s_cur >> Uint32(8 * s)) & Uint32(0xFF)) << Uint32(23))
                    acc[0] = acc[0] + d0 * sc
                    acc[1] = acc[1] + d1 * sc
                    acc[2] = acc[2] + d2 * sc
                    acc[3] = acc[3] + d3 * sc
                else:
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, cur[2 * s], cur[2 * s + 1])
                    acc[0] = d0
                    acc[1] = d1
                    acc[2] = d2
                    acc[3] = d3
            for i in cutlass.range_constexpr(8):
                cur[i] = nxt[i]
            if const_expr(self.block32):
                s_cur = s_nxt
        for i in cutlass.range_constexpr(4):
            partial[(warp_id * Int32(4) + Int32(i)) * Int32(32) + lane] = acc[i]
        cute.arch.sync_threads()
        if warp_id == Int32(0):
            a_scale = cute.make_ptr(Float32, alpha.toint(), cute.AddressSpace.gmem, assumed_align=4)[0]
            for i in cutlass.range_constexpr(4):
                total = Float32(0.0)
                for src in cutlass.range_constexpr(self.warps):
                    total = total + partial[Int32((src * 4 + i) * 32) + lane]
                row = g + Int32(8 * (i // 2))
                col = n0 + Int64(2) * Int64(j) + Int64(i % 2)
                if row < rows:
                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(2)
                    target = cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)
                    target[0] = (total * a_scale).to(cutlass.BFloat16)
