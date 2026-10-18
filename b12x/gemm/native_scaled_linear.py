"""Native MXFP8, int4-g32 and row-FP8 linear, with FP32 scale application.

Only unscaled E4M3 or integer weights widen in registers (exactly) for BF16 MMA.
No dequantized matrix is materialized. Fixed K32 group accumulation preserves
native group scales and row-invariant arithmetic from decode through prefill.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, cvt_f32_to_bf16_bits,
    ld_global_b16, ld_global_nc_u32, u32_as_f32,
)


@dsl_user_op
def _fp8_pair(packed: Uint32, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [packed.ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 q; cvt.u16.u32 q, $1; cvt.rn.bf16x2.e4m3x2 $0, q; }",
        "=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def _int4_pair(packed: Uint32):
    lo = ((packed & Uint32(15)).to(Int32) - Int32(8)).to(Float32)
    hi = (((packed >> Uint32(4)) & Uint32(15)).to(Int32) - Int32(8)).to(Float32)
    return cvt_f32_to_bf16_bits(lo) | (cvt_f32_to_bf16_bits(hi) << Uint32(16))


@cute.jit
def _e8m0(byte: Uint32):
    value = u32_as_f32(byte << Uint32(23))
    if byte == Uint32(0):
        value = u32_as_f32(Uint32(0x00400000))
    if byte == Uint32(255):
        value = u32_as_f32(Uint32(0x7fffffff))
    return value


class NativeScaledLinear:
    def __init__(self, g):
        self.k, self.n = g["input_features"], g["output_features"]
        self.encoding, self.input_units = g["encoding"], g["sharded_side"] == 2
        self.splits, self.max_rows = g["splits"], g["max_rows"]

    @cute.jit
    def __call__(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                 partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            self.multiply(x, payload, scales, partial, rows).launch(
                grid=(min(((rows+Int32(15))//Int32(16))*Int32(self.n//32*self.splits), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=4, stream=stream)
            self.reduce(partial, scales, output, rows).launch(
                grid=(min(rows*Int32((self.n+127)//128), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.jit
    def _weights(self, payload: cute.Tensor, col: Int32, k: Int32):
        index = col.to(Int64)*Int64(self.k) + k.to(Int64)
        if cutlass.const_expr(self.input_units):
            index = ((k//Int32(128)).to(Int64)*Int64(self.n) + col.to(Int64))*Int64(128) + (k%Int32(128)).to(Int64)
        if cutlass.const_expr(self.encoding == "int4_g32"):
            result = _int4_pair(payload[index//Int64(2)].to(Uint32))
        else:
            result = _fp8_pair(ld_global_b16(payload.iterator.toint() + index))
        return result

    @cute.jit
    def _scale(self, scales: cute.Tensor, col: Int32, group: Int32):
        scale = Float32(1.0)
        if cutlass.const_expr(self.encoding != "fp8_row"):
            index = col.to(Int64)*Int64(self.k//32) + group.to(Int64)
            if cutlass.const_expr(self.input_units):
                index = ((group//Int32(4)).to(Int64)*Int64(self.n) + col.to(Int64))*Int64(4) + (group%Int32(4)).to(Int64)
            if cutlass.const_expr(self.encoding == "mxfp8"):
                scale = _e8m0(scales[index].to(Uint32))
            else:
                scale = scales[index].to(Float32)
        return scale

    @cute.kernel
    def multiply(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                 partial: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid)&Int32(31), Int32(tid)>>Int32(5)
        while task < ((rows+Int32(15))//Int32(16))*Int32(self.n//32*self.splits):
            split = task%Int32(self.splits)
            nb = (task//Int32(self.splits))%Int32(self.n//32)
            group = task//Int32(self.splits*(self.n//32))
            row0, row1 = group*Int32(16)+lane//Int32(4), group*Int32(16)+lane//Int32(4)+Int32(8)
            col = nb*Int32(32)+warp*Int32(8)+(lane&Int32(3))*Int32(2)
            bcol = nb*Int32(32)+warp*Int32(8)+lane//Int32(4)
            total = cute.make_rmem_tensor((4,), Float32)
            total.fill(Float32(0.0))
            for kg in cutlass.range(32):
                kg32 = split*Int32(32)+kg
                if kg32 < Int32(self.k//32):
                    acc = cute.make_rmem_tensor((4,), Float32)
                    acc.fill(Float32(0.0))
                    for sub in cutlass.range_constexpr(2):
                        k = kg32*Int32(32)+Int32(sub*16)+(lane&Int32(3))*Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x.iterator.toint()+(row0.to(Int64)*Int64(self.k)+k.to(Int64))*Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr+Int64(16))
                        if row1 < rows:
                            addr = x.iterator.toint()+(row1.to(Int64)*Int64(self.k)+k.to(Int64))*Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr+Int64(16))
                        b0 = self._weights(payload, bcol, k)
                        b1 = self._weights(payload, bcol, k+Int32(8))
                        d0,d1,d2,d3 = bf16_mma_m16n8k16_f32(acc[0],acc[1],acc[2],acc[3],a0,a1,a2,a3,b0,b1)
                        acc[0],acc[1],acc[2],acc[3] = d0,d1,d2,d3
                    s0, s1 = self._scale(scales,col,kg32), self._scale(scales,col+Int32(1),kg32)
                    total[0],total[1] = total[0]+acc[0]*s0,total[1]+acc[1]*s1
                    total[2],total[3] = total[2]+acc[2]*s0,total[3]+acc[3]*s1
            base = (row0.to(Int64)*Int64(self.splits)+split.to(Int64))*Int64(self.n)+col.to(Int64)
            if row0 < rows:
                partial[base],partial[base+Int64(1)] = total[0],total[1]
            if row1 < rows:
                base += Int64(8*self.splits*self.n)
                partial[base],partial[base+Int64(1)] = total[2],total[3]
            task += step

    @cute.kernel
    def reduce(self, partial: cute.Tensor, scales: cute.Tensor, output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        while task < rows*Int32((self.n+127)//128):
            row = task//Int32((self.n+127)//128)
            col = (task%Int32((self.n+127)//128))*Int32(128)+Int32(tid)
            if col < Int32(self.n):
                value = Float32(0.0)
                for split in cutlass.range(self.splits):
                    value += partial[(row.to(Int64)*Int64(self.splits)+split.to(Int64))*Int64(self.n)+col.to(Int64)]
                if cutlass.const_expr(self.encoding == "fp8_row"):
                    value *= scales[col]
                output[row.to(Int64)*Int64(self.n)+col.to(Int64)] = value
            task += step
