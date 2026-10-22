"""Shared native trellis ABI v1, independent of model family and checkpoint.

All views are contiguous. ``rotated_inputs`` is an I64 device pointer table;
entry ``rotation_id`` addresses an F16 [rows,K] matrix with K-element row stride.
The caller owns H128/sign rotations, including private id 0, and output H128
and channel scales. ``input_block_scales`` multiplies each K128 FP32 accumulator.
Output and partials stay F32. No allocation, atomics or live-row specialization.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program

ABI_VERSION = 1
CTAS_PER_SM = 4
POINTERS = ("rotated_inputs", "payload", "input_block_scales", "partial", "output")
SCALARS = ("rows", "rotation_id", "cta_limit")


def geometry(input_features, output_features, *, bits=4, codebook="e4m3",
             sharded_side=1, rotation_count=4, max_rows=4096):
    values = (input_features, output_features, rotation_count, max_rows)
    if any(type(v) is not int or not 0 < v <= 2**31-1 for v in values):
        raise ValueError("trellis geometry must contain positive int32 dimensions")
    if input_features % 128 or output_features % 128:
        raise ValueError("trellis input/output dimensions must be multiples of 128")
    if type(bits) is not int or bits not in (2, 3, 4, 5, 6) or codebook not in ("e4m3", "f16", "mcg", "mul1") or sharded_side not in (1, 2):
        raise ValueError("trellis requires K2--K6, mcg/mul1/e4m3/f16 and side 1/2")
    splits = (input_features + 1023) // 1024
    tasks = (max_rows + 15) // 16 * (output_features // 128) * splits
    if max(tasks, max_rows * (output_features // 128), input_features * output_features // 256) > (2**31-1)//2:
        raise ValueError("trellis task/tile indexing exceeds signed int32")
    partial = max_rows * splits * output_features * 4
    if partial > 2**63-1:
        raise ValueError("trellis partials exceed signed int64 addressing")
    return dict(abi_version=ABI_VERSION, input_features=input_features,
                output_features=output_features, bits=bits, codebook=codebook,
                sharded_side=sharded_side, rotation_count=rotation_count,
                max_rows=max_rows, split_k_blocks=8, splits=splits,
                ctas_per_sm=CTAS_PER_SM)


class _Pointers:
    def __init__(self, g):
        from b12x.gemm.trellis_linear._native import NativeTrellisLinear
        self.core = NativeTrellisLinear(g)

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 input_block_scales: cute.Pointer, partial: cute.Pointer,
                 output: cute.Pointer, rows: cutlass.Int32,
                 rotation_id: cutlass.Int32, cta_limit: cutlass.Int32,
                 stream: cuda.CUstream):
        self.core(cute.make_tensor(rotated_inputs, cute.make_layout(1, stride=1)),
                  cute.make_tensor(payload, cute.make_layout(1, stride=1)),
                  cute.make_tensor(input_block_scales, cute.make_layout(1, stride=1)),
                  cute.make_tensor(partial, cute.make_layout(1, stride=1)),
                  cute.make_tensor(output, cute.make_layout(1, stride=1)),
                  rows, rotation_id, cta_limit, stream)


def compile_trellis_linear_aot(*, input_features, output_features, bits=4,
                               codebook="e4m3", sharded_side=1,
                               rotation_count=4, max_rows=4096):
    g = geometry(input_features, output_features, bits=bits, codebook=codebook,
                 sharded_side=sharded_side, rotation_count=rotation_count, max_rows=max_rows)
    k, n, s = g["input_features"], g["output_features"], g["splits"]
    operands = (
        Operand("rotated_inputs", torch.int64, f"[{rotation_count}]"),
        Operand("payload", torch.uint8, f"[{k*n*bits//8}]"),
        Operand("input_block_scales", torch.float16, f"[{k//128}]"),
        Operand("partial", torch.float32, f"[rows,{s},{n}]", "scratch"),
        Operand("output", torch.float32, f"[rows,{n}]", "out"),
    )
    pointers = _Pointers(g)
    return compile_program(pointers, name="trellis_linear", operands=operands,
                           scalars=tuple(Scalar(name) for name in SCALARS),
                           key=(*g.values(), pointers.core.wide), geometry=g,
                           scratch={"partial": lambda rows: rows*s*n*4}, doc=__doc__)
