"""Native row-scaled FP8 embedding gather ABI v1; invalid IDs write zero."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from ._common import Operand, Scalar, compile_program
from b12x._lib.intrinsics import ld_global_b16, u32_as_f32
from b12x.gemm.native_scaled_linear import _fp8_pair


class _Gather:
    def __init__(self, hidden, vocabulary, max_rows):
        self.hidden,self.vocabulary,self.max_rows = hidden,vocabulary,max_rows

    @cute.jit
    def __call__(self, ids: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer,
                 output: cute.Pointer, rows: cutlass.Int32, cta_limit: cutlass.Int32,
                 stream: cuda.CUstream):
        if rows > cutlass.Int32(0) and rows <= cutlass.Int32(self.max_rows) and cta_limit > cutlass.Int32(0):
            self.kernel(cute.make_tensor(ids,cute.make_layout(1,stride=1)),
                        cute.make_tensor(payload,cute.make_layout(1,stride=1)),
                        cute.make_tensor(scales,cute.make_layout(1,stride=1)),
                        cute.make_tensor(output,cute.make_layout(1,stride=1)),rows).launch(
                grid=(min(rows*cutlass.Int32((self.hidden+255)//256),cta_limit),1,1),
                block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
               output: cute.Tensor, rows: cutlass.Int32):
        tid,_,_ = cute.arch.thread_idx()
        task,_,_ = cute.arch.block_idx()
        step,_,_ = cute.arch.grid_dim()
        while task < rows*cutlass.Int32((self.hidden+255)//256):
            row = task//cutlass.Int32((self.hidden+255)//256)
            col = (task%cutlass.Int32((self.hidden+255)//256))*cutlass.Int32(256)+cutlass.Int32(tid)*cutlass.Int32(2)
            token = ids[row]
            if col < cutlass.Int32(self.hidden):
                lo,hi = cutlass.Float32(0),cutlass.Float32(0)
                if token >= cutlass.Int32(0) and token < cutlass.Int32(self.vocabulary):
                    index = token.to(cutlass.Int64)*cutlass.Int64(self.hidden)+col.to(cutlass.Int64)
                    pair = _fp8_pair(ld_global_b16(payload.iterator.toint()+index))
                    scale = scales[token]
                    lo = u32_as_f32(pair<<cutlass.Uint32(16))*scale
                    hi = u32_as_f32(pair&cutlass.Uint32(0xffff0000))*scale
                index = row.to(cutlass.Int64)*cutlass.Int64(self.hidden)+col.to(cutlass.Int64)
                output[index] = lo.to(cutlass.BFloat16)
                output[index+cutlass.Int64(1)] = hi.to(cutlass.BFloat16)
            task += step


def compile_fp8_row_embedding_aot(*, hidden, vocabulary, max_rows=4096):
    if any(type(v) is not int or not 0 < v <= 2**31-1 for v in (hidden,vocabulary,max_rows)) or hidden%2:
        raise ValueError("embedding requires positive int32 geometry and even hidden")
    if max_rows*((hidden+255)//256) > (2**31-1)//2:
        raise ValueError("embedding grid exceeds int32")
    g = dict(abi_version=1,hidden=hidden,vocabulary=vocabulary,max_rows=max_rows,ctas_per_sm=4)
    operands = (Operand("ids",torch.int32,"[rows]"),
                Operand("payload",torch.uint8,f"[{vocabulary},{hidden}]"),
                Operand("scales",torch.float32,f"[{vocabulary}]"),
                Operand("output",torch.bfloat16,f"[rows,{hidden}]","out"))
    return compile_program(_Gather(hidden,vocabulary,max_rows),name="fp8_row_embedding",
                           operands=operands,scalars=(Scalar("rows"),Scalar("cta_limit")),
                           key=tuple(g.values()),geometry=g,doc=__doc__)
