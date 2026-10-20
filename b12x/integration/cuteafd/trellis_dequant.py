"""Bounded N-chunk trellis dequantization into caller-owned BF16 scratch.

Diagnostic wide-row alternative: never cache a second resident weight matrix.
Payload order is shared native ABI v1. Output is [chunk_columns,K] for the
canonical BF16 projection. Input block scales are absorbed before BF16 rounding.
"""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cuda.bindings import driver as cuda
from b12x._lib.intrinsics import half2_to_float2_scaled
from b12x.gemm.trellis_linear._native import decode_tile
from ._common import Operand, Scalar, compile_program


class _Dequant:
    def __init__(self, k, n, columns, codebook, side):
        self.k, self.n, self.columns = k, n, columns
        self.mul1, self.side = codebook == "mul1", side

    @cute.jit
    def __call__(self, payload: cute.Pointer, scales: cute.Pointer, output: cute.Pointer,
                 first_column: Int32, stream: cuda.CUstream):
        self.kernel(payload, scales, output, first_column).launch(
            grid=(self.k // 16 * self.columns // 16 // 4, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, payload: cute.Pointer, scales: cute.Pointer, output: cute.Pointer, first: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        tile = Int32(cute.arch.block_idx()[0])*Int32(4) + tid//Int32(32)
        lane = tid & Int32(31)
        kt, nt = tile // Int32(self.columns//16), tile % Int32(self.columns//16)
        global_nt = nt + first//Int32(16)
        if cutlass.const_expr(self.side == 1):
            native_tile = global_nt//Int32(8)*Int32(self.k//16*8) + kt*Int32(8) + global_nt%Int32(8)
        else:
            native_tile = kt//Int32(8)*Int32(self.n//16*8) + global_nt//Int32(8)*Int32(64) + kt%Int32(8)*Int32(8) + global_nt%Int32(8)
        b0,b1,b2,b3 = decode_tile(payload.toint(), native_tile, lane, 6, False, self.mul1)
        s = cute.make_tensor(scales, cute.make_layout(self.k//128))
        out = cute.make_tensor(output, cute.make_layout(self.columns*self.k))
        factor = s[kt//Int32(8)].to(Float32)
        for j in cutlass.range_constexpr(4):
            word = b0
            if cutlass.const_expr(j == 1):
                word = b1
            elif cutlass.const_expr(j == 2):
                word = b2
            elif cutlass.const_expr(j == 3):
                word = b3
            lo, hi = half2_to_float2_scaled(word, factor)
            row = kt*Int32(16) + (lane%Int32(4))*Int32(2) + Int32((j%2)*8)
            col = nt*Int32(16) + lane//Int32(4) + Int32((j//2)*8)
            offset = col.to(Int64)*Int64(self.k) + row.to(Int64)
            out[offset] = lo.to(cutlass.BFloat16)
            out[offset+Int64(1)] = hi.to(cutlass.BFloat16)


def compile_trellis_dequant_aot(*, input_features, output_features, chunk_columns,
                               codebook="mcg", sharded_side=1):
    k,n,c = input_features,output_features,chunk_columns
    if min(k,n,c) <= 0 or k%128 or n%128 or c%128 or n%c or codebook not in ("mcg","mul1") or sharded_side not in (1,2):
        raise ValueError("dequant requires H128 shapes, a dividing N-chunk, mcg/mul1 and side 1/2")
    g=dict(input_features=k,output_features=n,chunk_columns=c,codebook=codebook,sharded_side=sharded_side)
    return compile_program(_Dequant(k,n,c,codebook,sharded_side),name="trellis_dequant",
        operands=(Operand("payload",torch.uint8,f"[{k*n*6//8}]"),
                  Operand("scales",torch.float16,f"[{k//128}]"),
                  Operand("output",torch.bfloat16,f"[{c},{k}]","out")),
        scalars=(Scalar("first_column"),),key=tuple(g.values()),geometry=g)
