"""Allocation-free H128 companions for shared native trellis ABI v1.

Input: rotated = F16(H128(F32(x) * F32(suh))). The full per-channel suh is
absorbed before H128, so the shared ABI's K128 block scales are exactly one.
Output: BF16/F32(H128(accumulator) * F32(svh)); select output_dtype="f32"
for logits or for TP reduction without a BF16 narrowing. All H128s are normalized.
Input ABI: input BF16/F32 [rows,K], suh F16 [K], rotated F16 [rows,K],
input_block_scales F16 [K/128], rows i32, cta_limit i32, CUDA stream.
Output ABI: input F32 [rows,N], svh F16 [N], output BF16/F32 [rows,N],
rows i32, cta_limit i32, CUDA stream. All arrays are contiguous row-major;
row strides are K and N elements respectively. Capacity is fixed at compile.

N-split shards replicate full suh and slice svh; K-split shards slice suh and
retain full svh. H128 boundaries must align with the shard boundaries. The
output input is the core's split-K-reduced output, not its partial workspace.
For TP2 K-split projections, the caller normally reduces F32 core outputs
before this transform; alternatively transform each rank then reduce BF16
outputs (with that extra rounding). Apply svh exactly once either way.

Features are stored extents. For kv_a compile N640, transform all 640 channels,
and let the caller trim to logical N576 afterwards, before norm/rope. This is
not equivalent to padding a logical H576 rotation: the quantizer must calibrate
its stored H640 layout. No trimming, TP communication, allocation or pointer
registration occurs here. Input/output transforms may not alias their inputs.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float16, Float32, Int32, Int64
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program
from .pointwise import _mul
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


class _Transform:
    def __init__(self, features, max_rows, ingress):
        self.features, self.max_rows, self.ingress = features, max_rows, ingress
        self.blocks = features // 128

    @cute.jit
    def __call__(self, input: cute.Pointer, suh: cute.Pointer,
                 rotated: cute.Pointer, input_block_scales: cute.Pointer,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            self.kernel(input, suh, rotated, input_block_scales, rows).launch(
                grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, scales: cute.Pointer,
               output: cute.Pointer, block_scales: cute.Pointer, rows: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        task = Int32(cute.arch.block_idx()[0]) * Int32(4) + warp
        step = Int32(cute.arch.grid_dim()[0]) * Int32(4)
        x = cute.make_tensor(source, cute.make_layout(self.max_rows * self.features))
        s = cute.make_tensor(scales, cute.make_layout(self.features))
        y = cute.make_tensor(output, cute.make_layout(self.max_rows * self.features))
        b = cute.make_tensor(block_scales, cute.make_layout(self.blocks))
        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            offset = row.to(Int64) * Int64(self.features) + col.to(Int64)
            v0, v1 = x[offset].to(Float32), x[offset + Int64(1)].to(Float32)
            v2, v3 = x[offset + Int64(2)].to(Float32), x[offset + Int64(3)].to(Float32)
            s0, s1 = s[col].to(Float32), s[col + Int32(1)].to(Float32)
            s2, s3 = s[col + Int32(2)].to(Float32), s[col + Int32(3)].to(Float32)
            if cutlass.const_expr(self.ingress):
                # Preserve the F32 scale boundary before the first H4 additions.
                v0, v1, v2, v3 = _mul(v0,s0), _mul(v1,s1), _mul(v2,s2), _mul(v3,s3)
            h0, h1, h2, h3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)
            if cutlass.const_expr(not self.ingress):
                h0, h1, h2, h3 = h0*s0, h1*s1, h2*s2, h3*s3
            y[offset], y[offset + Int64(1)] = h0.to(y.element_type), h1.to(y.element_type)
            y[offset + Int64(2)], y[offset + Int64(3)] = h2.to(y.element_type), h3.to(y.element_type)
            if cutlass.const_expr(self.ingress):
                if row == Int32(0) and lane == Int32(0):
                    b[block] = Float16(1.)
            task += step


GROUP_NAMES = ("q", "k", "v", "g")


class _Output:
    def __init__(self, features, max_rows):
        self.core = _Transform(features, max_rows, False)

    @cute.jit
    def __call__(self, input: cute.Pointer, svh: cute.Pointer, output: cute.Pointer,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        # The core's fourth operand is constexpr-unused on output.
        self.core(input, svh, output, svh, rows, cta_limit, stream)


class _GroupedTransform:
    def __init__(self, features, max_rows, ingress):
        self.features, self.max_rows, self.ingress = features, max_rows, ingress
        self.blocks = features // 128

    @cute.jit
    def __call__(self, sources: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                 scales: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                 outputs: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                 block_scales: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            self.kernel(sources, scales, outputs, block_scales, rows).launch(
                grid=(min(rows * Int32(self.blocks), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, sources: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
               scales: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
               outputs: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
               block_scales: tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
               rows: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        lane, proj = tid & Int32(31), tid >> Int32(5)
        task = Int32(cute.arch.block_idx()[0])
        step = Int32(cute.arch.grid_dim()[0])
        x0 = cute.make_tensor(sources[0], cute.make_layout(self.max_rows * self.features))
        x1 = cute.make_tensor(sources[1], cute.make_layout(self.max_rows * self.features))
        x2 = cute.make_tensor(sources[2], cute.make_layout(self.max_rows * self.features))
        x3 = cute.make_tensor(sources[3], cute.make_layout(self.max_rows * self.features))
        s_0 = cute.make_tensor(scales[0], cute.make_layout(self.features))
        s_1 = cute.make_tensor(scales[1], cute.make_layout(self.features))
        s_2 = cute.make_tensor(scales[2], cute.make_layout(self.features))
        s_3 = cute.make_tensor(scales[3], cute.make_layout(self.features))
        y0 = cute.make_tensor(outputs[0], cute.make_layout(self.max_rows * self.features))
        y1 = cute.make_tensor(outputs[1], cute.make_layout(self.max_rows * self.features))
        y2 = cute.make_tensor(outputs[2], cute.make_layout(self.max_rows * self.features))
        y3 = cute.make_tensor(outputs[3], cute.make_layout(self.max_rows * self.features))
        b0 = cute.make_tensor(block_scales[0], cute.make_layout(self.blocks))
        b1 = cute.make_tensor(block_scales[1], cute.make_layout(self.blocks))
        b2 = cute.make_tensor(block_scales[2], cute.make_layout(self.blocks))
        b3 = cute.make_tensor(block_scales[3], cute.make_layout(self.blocks))
        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            offset = row.to(Int64) * Int64(self.features) + col.to(Int64)
            v0, v1, v2, v3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if cutlass.const_expr(self.ingress):
                v0, v1 = x0[offset].to(Float32), x0[offset + Int64(1)].to(Float32)
                v2, v3 = x0[offset + Int64(2)].to(Float32), x0[offset + Int64(3)].to(Float32)
            elif proj == Int32(0):
                v0, v1 = x0[offset].to(Float32), x0[offset + Int64(1)].to(Float32)
                v2, v3 = x0[offset + Int64(2)].to(Float32), x0[offset + Int64(3)].to(Float32)
            elif proj == Int32(1):
                v0, v1 = x1[offset].to(Float32), x1[offset + Int64(1)].to(Float32)
                v2, v3 = x1[offset + Int64(2)].to(Float32), x1[offset + Int64(3)].to(Float32)
            elif proj == Int32(2):
                v0, v1 = x2[offset].to(Float32), x2[offset + Int64(1)].to(Float32)
                v2, v3 = x2[offset + Int64(2)].to(Float32), x2[offset + Int64(3)].to(Float32)
            else:
                v0, v1 = x3[offset].to(Float32), x3[offset + Int64(1)].to(Float32)
                v2, v3 = x3[offset + Int64(2)].to(Float32), x3[offset + Int64(3)].to(Float32)
            s0, s1, s2, s3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if proj == Int32(0):
                s0, s1 = s_0[col].to(Float32), s_0[col + Int32(1)].to(Float32)
                s2, s3 = s_0[col + Int32(2)].to(Float32), s_0[col + Int32(3)].to(Float32)
            elif proj == Int32(1):
                s0, s1 = s_1[col].to(Float32), s_1[col + Int32(1)].to(Float32)
                s2, s3 = s_1[col + Int32(2)].to(Float32), s_1[col + Int32(3)].to(Float32)
            elif proj == Int32(2):
                s0, s1 = s_2[col].to(Float32), s_2[col + Int32(1)].to(Float32)
                s2, s3 = s_2[col + Int32(2)].to(Float32), s_2[col + Int32(3)].to(Float32)
            else:
                s0, s1 = s_3[col].to(Float32), s_3[col + Int32(1)].to(Float32)
                s2, s3 = s_3[col + Int32(2)].to(Float32), s_3[col + Int32(3)].to(Float32)
            if cutlass.const_expr(self.ingress):
                v0, v1, v2, v3 = _mul(v0,s0), _mul(v1,s1), _mul(v2,s2), _mul(v3,s3)
            h0, h1, h2, h3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)
            if cutlass.const_expr(not self.ingress):
                h0, h1, h2, h3 = h0*s0, h1*s1, h2*s2, h3*s3
            e0, e1 = h0.to(y0.element_type), h1.to(y0.element_type)
            e2, e3 = h2.to(y0.element_type), h3.to(y0.element_type)
            if proj == Int32(0):
                y0[offset], y0[offset + Int64(1)], y0[offset + Int64(2)], y0[offset + Int64(3)] = e0, e1, e2, e3
            elif proj == Int32(1):
                y1[offset], y1[offset + Int64(1)], y1[offset + Int64(2)], y1[offset + Int64(3)] = e0, e1, e2, e3
            elif proj == Int32(2):
                y2[offset], y2[offset + Int64(1)], y2[offset + Int64(2)], y2[offset + Int64(3)] = e0, e1, e2, e3
            else:
                y3[offset], y3[offset + Int64(1)], y3[offset + Int64(2)], y3[offset + Int64(3)] = e0, e1, e2, e3
            if cutlass.const_expr(self.ingress):
                if row == Int32(0) and lane == Int32(0):
                    if proj == Int32(0):
                        b0[block] = Float16(1.)
                    elif proj == Int32(1):
                        b1[block] = Float16(1.)
                    elif proj == Int32(2):
                        b2[block] = Float16(1.)
                    else:
                        b3[block] = Float16(1.)
            task += step


class _GroupedInput:
    def __init__(self, features, max_rows):
        self.core = _GroupedTransform(features, max_rows, True)

    @cute.jit
    def __call__(self, input: cute.Pointer,
                 q_suh: cute.Pointer, k_suh: cute.Pointer,
                 v_suh: cute.Pointer, g_suh: cute.Pointer,
                 q_rotated: cute.Pointer, k_rotated: cute.Pointer,
                 v_rotated: cute.Pointer, g_rotated: cute.Pointer,
                 q_input_block_scales: cute.Pointer, k_input_block_scales: cute.Pointer,
                 v_input_block_scales: cute.Pointer, g_input_block_scales: cute.Pointer,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        self.core((input, input, input, input), (q_suh, k_suh, v_suh, g_suh),
                  (q_rotated, k_rotated, v_rotated, g_rotated),
                  (q_input_block_scales, k_input_block_scales,
                   v_input_block_scales, g_input_block_scales),
                  rows, cta_limit, stream)


class _GroupedOutput:
    def __init__(self, features, max_rows):
        self.core = _GroupedTransform(features, max_rows, False)

    @cute.jit
    def __call__(self, q_input: cute.Pointer, k_input: cute.Pointer,
                 v_input: cute.Pointer, g_input: cute.Pointer,
                 q_svh: cute.Pointer, k_svh: cute.Pointer,
                 v_svh: cute.Pointer, g_svh: cute.Pointer,
                 q_output: cute.Pointer, k_output: cute.Pointer,
                 v_output: cute.Pointer, g_output: cute.Pointer,
                 rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        svh = (q_svh, k_svh, v_svh, g_svh)
        self.core((q_input, k_input, v_input, g_input), svh,
                  (q_output, k_output, v_output, g_output), svh,
                  rows, cta_limit, stream)


def _geometry(features, max_rows, groups=1):
    if any(type(v) is not int or v <= 0 for v in (features, max_rows)) or features % 128:
        raise ValueError("trellis transforms require positive H128-aligned features and rows")
    if type(groups) is not int or groups not in (1, 4):
        raise ValueError("trellis transform groups must be 1 or 4")
    if max_rows * (features // 128) * groups > 2**31 - 1:
        raise ValueError("trellis transform task indexing exceeds signed int32")


def compile_trellis_input_aot(*, features, input_dtype="bf16", max_rows=4096, output_dtype="f16", groups=1):
    _geometry(features, max_rows, groups)
    if input_dtype not in ("bf16", "f32"):
        raise ValueError("trellis input transform requires bf16/f32")
    if output_dtype not in ("f16", "bf16"):
        raise ValueError("trellis rotated output requires f16/bf16")
    g = dict(features=features, input_dtype=input_dtype, max_rows=max_rows)
    if output_dtype != "f16":
        g["output_dtype"] = output_dtype
    if groups != 1:
        g["groups"] = groups
    in_dt = torch.bfloat16 if input_dtype == "bf16" else torch.float32
    out_dt = torch.float16 if output_dtype == "f16" else torch.bfloat16
    if groups == 1:
        launch = _Transform(features, max_rows, True)
        operands = (Operand("input", in_dt, f"[rows,{features}]"),
                    Operand("suh", torch.float16, f"[{features}]"),
                    Operand("rotated", out_dt, f"[rows,{features}]", "out"),
                    Operand("input_block_scales", torch.float16, f"[{features//128}]", "out"))
    else:
        launch = _GroupedInput(features, max_rows)
        operands = (Operand("input", in_dt, f"[rows,{features}]"),
                    *(Operand(f"{p}_suh", torch.float16, f"[{features}]") for p in GROUP_NAMES),
                    *(Operand(f"{p}_rotated", out_dt, f"[rows,{features}]", "out") for p in GROUP_NAMES),
                    *(Operand(f"{p}_input_block_scales", torch.float16, f"[{features//128}]", "out") for p in GROUP_NAMES))
    return compile_program(launch, name="trellis_input", operands=operands,
        scalars=(Scalar("rows"), Scalar("cta_limit")), key=tuple(g.values()), geometry=g, doc=__doc__)


def compile_trellis_output_aot(*, features, output_dtype="bf16", max_rows=4096, groups=1):
    _geometry(features, max_rows, groups)
    if output_dtype not in ("bf16", "f32"):
        raise ValueError("trellis output transform requires bf16/f32")
    g = dict(features=features, output_dtype=output_dtype, max_rows=max_rows)
    if groups != 1:
        g["groups"] = groups
    out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    if groups == 1:
        launch = _Output(features, max_rows)
        operands = (Operand("input", torch.float32, f"[rows,{features}]"),
                    Operand("svh", torch.float16, f"[{features}]"),
                    Operand("output", out_dt, f"[rows,{features}]", "out"))
    else:
        launch = _GroupedOutput(features, max_rows)
        operands = (*(Operand(f"{p}_input", torch.float32, f"[rows,{features}]") for p in GROUP_NAMES),
                    *(Operand(f"{p}_svh", torch.float16, f"[{features}]") for p in GROUP_NAMES),
                    *(Operand(f"{p}_output", out_dt, f"[rows,{features}]", "out") for p in GROUP_NAMES))
    return compile_program(launch, name="trellis_output", operands=operands,
        scalars=(Scalar("rows"), Scalar("cta_limit")), key=tuple(g.values()), geometry=g, doc=__doc__)
