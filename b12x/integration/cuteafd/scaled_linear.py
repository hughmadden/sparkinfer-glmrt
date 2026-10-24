"""Shared scaled-linear pointer ABIs.

Both versions take x BF16[rows,K], payload U8, scales, partial, output;
then rows and cta_limit I32, then stream.

ABI v1 (default): output F32[rows,N]; partial is unused (zero scratch bytes).
Payload is native [out,in] (int4 low nibble first, biased by8), or side2
[in/128,out,128] (int4 last dimension64 bytes). Group scales are [out,in/32],
or [in/128,out,4]. MXFP8 uses UE8M0 bytes; int4 uses F16; row-FP8 uses F32[out].

ABI v2 (activation_encoding="mxfp8"): dynamic K32 MXFP8 activations and native
hardware MXFP8 MMA, output BF16[rows,N]. Payload stays E4M3 [N,K]; scales are
UE8M0 in MMA tile order, prepared once from checkpoint [N,K/32] scales.
partial is U8 scratch sized by stage_scratch_bytes(K,N,rows). The fixed
(64,128) tile uses K128 stages and one K slice. Launch grids query the runtime
SM count; cta_limit is unused. No device count enters the compile key.
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
    # Prefill tiles (128 rows x 64 columns at the finest) and column indices stay int32;
    # weights stay far inside 64-bit byte addressing.
    if (max_rows+127)//128*((output_features+63)//64) > (2**31-1)//2 or output_features+128 > 2**31-1:
        raise ValueError("native linear grid indexing exceeds int32")
    if input_features*output_features > 2**40:
        raise ValueError("native linear weight exceeds 2^40 elements")
    # Tile core (every encoding): fixed per-shape K segments folded in order, no split-K
    # partials; ``partial`` stays in the pointer list for ABI v1 and is never touched.
    # Segments come from the weight shape only (never the device): enough parallel
    # decode chains for small-N projections, each a power of two dividing K/128.
    groups = 2 if encoding == "fp8_row" else 1
    segments = 1
    while (output_features//(8*groups)*segments < 2048 and segments < 8
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


def compile_scaled_linear_aot(*, input_features, output_features, encoding="mxfp8", sharded_side=1,
                              max_rows=4096, activation_encoding=None, splits=1):
    g = geometry(input_features,output_features,encoding=encoding,sharded_side=sharded_side,max_rows=max_rows)
    k,n = input_features,output_features
    if activation_encoding is not None:
        if activation_encoding != "mxfp8" or encoding != "mxfp8" or sharded_side != 1 or n % 128:
            raise ValueError("native W8A8 requires MXFP8, output-major weights and N multiple128")
        from ._mxfp8_linear import Mxfp8Linear
        launch = Mxfp8Linear(k, n, splits)
        g = dict(abi_version=2, input_features=k, output_features=n, max_rows=max_rows,
                 encoding="mxfp8", activation_encoding="mxfp8", sharded_side=1,
                 core="mxfp8_mma", scale_layout="mma", scale_block_size=32,
                 output_dtype="bfloat16", splits=splits, split_k=k // splits,
                 tile_m=64, tile_n=128, tile_k=128, mma_k=32,
                 load_path="tma", target_occupancy=1, grid="runtime_num_sm")
        return compile_program(launch, name="scaled_linear_mxfp8", operands=(
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("payload", torch.uint8, f"[{n},{k}] (native E4M3)"),
            Operand("scales", torch.uint8, f"[{n*k//32}] (MMA-packed UE8M0)"),
            Operand("partial", torch.uint8, "[scratch_bytes(rows)]", "scratch"),
            Operand("output", torch.bfloat16, f"[rows,{n}]", "out")),
            scalars=(Scalar("rows"), Scalar("cta_limit", note="unused; runtime SM count sets the grid")),
            key=(k, n, "mxfp8") + ((splits,) if splits > 1 else ()), geometry=g,
            scratch={"partial": launch.stage.scratch_bytes}, doc=__doc__, version=2)
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
