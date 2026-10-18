"""Shared native MXFP8/int4-g32/row-FP8 linear ABI v1.

Pointer order: x BF16[rows,K], payload U8, scales, partial (unused since the
tile core: no scratch, may be any pointer), output F32[rows,N]. Scalars rows/cta_limit I32, then stream. Payload is native
[out,in] (int4 low nibble first, biased by 8), or side2 [in/128,out,128]
(int4 last dimension 64 bytes). Group scales are [out,in/32], or
[in/128,out,4]. MXFP8 uses UE8M0 bytes; int4 uses F16. Row-FP8 uses F32[out].
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from ._common import Operand, Scalar, compile_program

ABI_VERSION = 1
POINTERS = ("x", "payload", "scales", "partial", "output")
SCALARS = ("rows", "cta_limit")


def geometry(input_features, output_features, *, encoding="mxfp8", sharded_side=1, max_rows=4096):
    if any(type(v) is not int or not 0 < v <= 2**31-1 for v in (input_features, output_features, max_rows)):
        raise ValueError("native linear dimensions must be positive int32")
    if input_features%128 or output_features%32:
        raise ValueError("native linear requires K multiple128 and N multiple32")
    if encoding not in ("mxfp8", "int4_g32", "fp8_row") or sharded_side not in (1, 2):
        raise ValueError("unsupported native encoding or shard side")
    if encoding == "fp8_row" and sharded_side != 1:
        raise ValueError("row-FP8 uses output-major storage")
    splits = (input_features+1023)//1024
    tasks = (max_rows+15)//16*(output_features//32)*splits
    if max(tasks, max_rows*((output_features+127)//128)) > (2**31-1)//2:
        raise ValueError("native linear grid indexing exceeds int32")
    if max_rows*splits*output_features*4 > 2**63-1:
        raise ValueError("native linear scratch exceeds int64")
    # Tile core (every encoding): fixed per-shape K segments folded in order, no split-K
    # partials; ``partial`` stays in the pointer list for ABI v1 and is never touched.
    # Segments come from the weight shape only (never the device): enough parallel
    # decode chains for small-N projections, each a power of two dividing K/128.
    groups = 2 if encoding == "fp8_row" or output_features >= 8192 else 1
    segments = 1
    while (output_features//(8*groups)*segments < 2048 and segments < 16
           and (input_features//128) % (segments*2) == 0):
        segments *= 2
    return dict(abi_version=ABI_VERSION,input_features=input_features,output_features=output_features,
                encoding=encoding,sharded_side=sharded_side,max_rows=max_rows,
                splits=1,split_k=input_features,segments=segments,ctas_per_sm=4,core="tile")


class _Pointers:
    def __init__(self, g):
        from b12x.gemm.native_tile_linear import NativeTileLinear
        self.core = NativeTileLinear(g)

    @cute.jit
    def __call__(self, x: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer,
                 partial: cute.Pointer, output: cute.Pointer, rows: cutlass.Int32,
                 cta_limit: cutlass.Int32, stream: cuda.CUstream):
        self.core(cute.make_tensor(x,cute.make_layout(1,stride=1)),
                  cute.make_tensor(payload,cute.make_layout(1,stride=1)),
                  cute.make_tensor(scales,cute.make_layout(1,stride=1)),
                  cute.make_tensor(output,cute.make_layout(1,stride=1)),rows,cta_limit,stream)


def compile_scaled_linear_aot(*, input_features, output_features, encoding="mxfp8", sharded_side=1, max_rows=4096):
    g = geometry(input_features,output_features,encoding=encoding,sharded_side=sharded_side,max_rows=max_rows)
    k,n,s = input_features,output_features,g["splits"]
    scale_dtype = {"mxfp8":torch.uint8,"int4_g32":torch.float16,"fp8_row":torch.float32}[encoding]
    payload_size = k*n//(2 if encoding == "int4_g32" else 1)
    scale_size = n if encoding == "fp8_row" else n*k//32
    operands = (Operand("x",torch.bfloat16,f"[rows,{k}]"),
                Operand("payload",torch.uint8,f"[{payload_size}]"),
                Operand("scales",scale_dtype,f"[{scale_size}]"),
                Operand("partial",torch.float32,"[0]","scratch"),
                Operand("output",torch.float32,f"[rows,{n}]","out"))
    return compile_program(_Pointers(g),name="scaled_linear",operands=operands,
                           scalars=tuple(Scalar(n) for n in SCALARS),key=tuple(g.values()),geometry=g,
                           scratch={"partial":lambda rows:0},doc=__doc__)
