"""One-time native ABI v1 K4/e4m3 unpack into exact E4M3 [N,K] bytes.

No rotations or scales are applied. The caller retains only the resulting FP8
weight for wide-row projection; H128/suh/svh remain in the usual transforms.
"""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import half2_to_float2_scaled
from b12x.gemm.trellis_linear._native import decode_tile
from ._common import Operand, compile_program


class _Unpack:
    def __init__(self, k, n, side):
        self.k, self.n, self.side = k, n, side

    @cute.jit
    def __call__(self, payload: cute.Pointer, output: cute.Pointer, stream: cuda.CUstream):
        self.kernel(payload, output).launch(
            grid=(self.k // 16 * (self.n // 16) // 4, 1, 1),
            block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, payload: cute.Pointer, output: cute.Pointer):
        tid = Int32(cute.arch.thread_idx()[0])
        tile = Int32(cute.arch.block_idx()[0]) * Int32(4) + tid // Int32(32)
        lane = tid & Int32(31)
        kt, nt = tile // Int32(self.n // 16), tile % Int32(self.n // 16)
        if cutlass.const_expr(self.side == 1):
            native_tile = nt // Int32(8) * Int32(self.k // 16 * 8) + kt * Int32(8) + nt % Int32(8)
        else:
            native_tile = (kt // Int32(8) * Int32(self.n // 16 * 8)
                           + nt // Int32(8) * Int32(64)
                           + kt % Int32(8) * Int32(8) + nt % Int32(8))
        b0, b1, b2, b3 = decode_tile(payload.toint(), native_tile, lane, 4, True, False)
        fp8 = cute.make_ptr(cutlass.Float8E4M3FN, output.toint(), cute.AddressSpace.gmem, assumed_align=16)
        out = cute.make_tensor(fp8, cute.make_layout(self.n * self.k))
        for j in cutlass.range_constexpr(4):
            word = b0
            if cutlass.const_expr(j == 1):
                word = b1
            elif cutlass.const_expr(j == 2):
                word = b2
            elif cutlass.const_expr(j == 3):
                word = b3
            lo, hi = half2_to_float2_scaled(word, Float32(1.0))
            row = kt * Int32(16) + (lane % Int32(4)) * Int32(2) + Int32((j % 2) * 8)
            col = nt * Int32(16) + lane // Int32(4) + Int32((j // 2) * 8)
            offset = col.to(Int64) * Int64(self.k) + row.to(Int64)
            out[offset] = lo.to(cutlass.Float8E4M3FN)
            out[offset + Int64(1)] = hi.to(cutlass.Float8E4M3FN)


def compile_trellis_unpack_e4m3_aot(*, input_features, output_features, sharded_side=1):
    k, n = input_features, output_features
    if (any(type(v) is not int or v <= 0 for v in (k, n)) or k % 128 or n % 128
            or sharded_side not in (1, 2) or k * n // 256 > 2**31 - 1):
        raise ValueError("unpack requires H128-aligned K/N, side 1/2 and int32 tile indexing")
    g = dict(input_features=k, output_features=n, sharded_side=sharded_side,
             bits=4, codebook="e4m3", abi_version=1)
    return compile_program(_Unpack(k, n, sharded_side), name="trellis_unpack_e4m3",
        operands=(Operand("payload", torch.uint8, f"[{k*n//2}]"),
                  Operand("output", torch.uint8, f"[{n},{k}]", "out")),
        scalars=(), key=tuple(g.values()), geometry=g, doc=__doc__)
