"""Format-neutral dense glue and exact-unpacked E4M3 W8A16 prefill."""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._fp8_weights import TmaFp8Gemm


class _SwiGLUSplit:
    def __init__(self, inter):
        self.inter = inter

    @cute.jit
    def __call__(self, gate: cute.Pointer, up: cute.Pointer, hidden: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(gate, up, hidden).launch(
            grid=(rows, (self.inter + 255) // 256, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate: cute.Pointer, up: cute.Pointer, hidden: cute.Pointer):
        col = Int64(cute.arch.block_idx()[1]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.inter):
            offset = Int64(cute.arch.block_idx()[0]) * Int64(self.inter) + col
            g, u = Float32(gate[offset]), Float32(up[offset])
            silu = div_rn_f32(g, Float32(1.0) + cute.math.exp(-g, fastmath=False)).to(BFloat16).to(Float32)
            hidden[offset] = (silu * u).to(BFloat16)


class _NarrowPack:
    def __init__(self, width, pitch):
        self.width, self.pitch = width, pitch

    @cute.jit
    def __call__(self, input: cute.Pointer, output: cute.Pointer,
                 rows: Int32, c0: Int32, stream: cuda.CUstream):
        if c0 >= Int32(0) and c0 + Int32(self.width) <= Int32(self.pitch):
            self.kernel(input, output, c0).launch(
                grid=(rows, (self.width + 255) // 256, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, input: cute.Pointer, output: cute.Pointer, c0: Int32):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.width):
            output[row * Int64(self.pitch) + Int64(c0) + col] = input[row * Int64(self.width) + col].to(BFloat16)


def _positive(*values):
    if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in values):
        raise ValueError("dense geometry requires positive int32 dimensions")


def compile_swiglu_split_aot(*, inter):
    _positive(inter)
    return compile_program(_SwiGLUSplit(inter), name="swiglu_split", operands=(
        Operand("gate", torch.bfloat16, f"[rows,{inter}]"),
        Operand("up", torch.bfloat16, f"[rows,{inter}]"),
        Operand("hidden", torch.bfloat16, f"[rows,{inter}]", "out")),
        scalars=(Scalar("rows"),), key=(inter,), geometry=dict(inter=inter))


def compile_f32_narrow_pack_aot(*, width, pitch):
    _positive(width, pitch)
    if width > pitch:
        raise ValueError("pack width exceeds destination pitch")
    return compile_program(_NarrowPack(width, pitch), name="f32_narrow_pack", operands=(
        Operand("input", torch.float32, f"[rows,{width}]"),
        Operand("output", torch.bfloat16, f"[rows,{pitch}]", "out")),
        scalars=(Scalar("rows"), Scalar("c0")), key=(width, pitch),
        geometry=dict(width=width, pitch=pitch))


def compile_trellis_fp8_prefill_aot(*, input_features, output_features, max_rows=4096):
    k, n = input_features, output_features
    _positive(k, n, max_rows)
    if k % 128 or n % 128:
        raise ValueError("unpacked trellis FP8 prefill requires H128-aligned K/N")
    launch = TmaFp8Gemm(n, k, scales="block", out_dtype=Float32,
                        compute_warps=8, num_stages=3)
    g = dict(input_features=k, output_features=n, max_rows=max_rows,
             scales="block", output_dtype="f32", compute_warps=8, num_stages=3)
    return compile_program(launch, name="trellis_fp8_prefill", operands=(
        Operand("x", torch.bfloat16, f"[rows,{k}]"),
        Operand("w", torch.uint8, f"[{n},{k}]"),
        Operand("scale", torch.float32, f"[{n//128},{k//128}]"),
        Operand("out", torch.float32, f"[rows,{n}]", "out")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()), geometry=g)
