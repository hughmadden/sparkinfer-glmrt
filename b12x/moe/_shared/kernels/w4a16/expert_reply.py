"""Combined BF16 expert replies using Muse's partial-wire block layouts.

Width is a runtime extent: one export serves every supported model and card.
The quantizer shares Muse's exact block scale/conversion helpers.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from b12x._lib.intrinsics import cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16, div_rn_f32
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.pointwise import _view
from .expert_partial import _fp4_pair, fp8_scale, decode_fp4
from b12x.integration.cuteafd.muse_tp import _ue8m0_scale
from b12x.integration.cuteafd.pointwise import _mul


class ExpertReply:
    def __init__(self, quantize, stage=False):
        self.quantize, self.stage = quantize, stage

    @cute.jit
    def __call__(self, source: cute.Pointer, out: cute.Pointer, mode: cute.Pointer,
                 rows: Int32, width: Int32, format: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            extent = width
            if cutlass.const_expr(self.stage):
                extent = width + width // Int32(32)
            self.kernel(source, out, mode, rows, width, format).launch(
                grid=(rows, (extent + Int32(127)) // Int32(128), 1),
                block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, out: cute.Pointer, mode: cute.Pointer,
               rows: Int32, width: Int32, format: Int32):
        source, out, mode = (_view(p) for p in (source, out, mode))
        row, slab, _ = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        selected = format
        if format < Int32(0):
            selected = Int32(mode[0])
        col = Int64(slab) * Int64(128) + Int64(lane) * Int64(4)
        h = Int64(width)
        if selected == Int32(0):
            # BF16 is deliberately not touched on the in-place consumer path.
            pass
        elif cutlass.const_expr(self.stage):
            stride = h + h // Int64(32)
            if selected == Int32(2):
                stride = h // Int64(2) + h // Int64(16)
            # Copy the compressed plane before expansion. Separate launches
            # provide the global ordering an in-place multi-CTA expansion lacks.
            first = Int64(slab) * Int64(128) + Int64(lane)
            for j in cutlass.range_constexpr(4):
                c = first + Int64(j * 32)
                if c < stride:
                    out[Int64(row) * stride + c] = source[Int64(row) * stride + c]
        elif cutlass.const_expr(self.quantize):
            values = cute.make_rmem_tensor((4,), Float32)
            values.fill(0.0)
            for j in cutlass.range_constexpr(4):
                if col + Int64(j) < h:
                    values[j] = Float32(source[Int64(row) * h + col + Int64(j)])
            amax = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                amax = cutlass.max(amax, cute.math.abs(values[j]))
            for shift in cutlass.range_constexpr(2):
                amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift))
            if selected == Int32(1):
                amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
            scale, code = Float32(1.0), Uint32(0)
            if selected == Int32(2):
                if amax > Float32(0.0):
                    code = cvt_f32_to_e4m3(div_rn_f32(amax, Float32(6.0)))
                    scale = cvt_e4m3_to_f32_via_f16(code)
                    if scale < Float32(2.0 ** -9):
                        scale, code = Float32(2.0 ** -9), Uint32(1)
                base = Int64(row) * (h // Int64(2) + h // Int64(16))
                for j in cutlass.range_constexpr(2):
                    if col + Int64(j * 2) < h:
                        packed = _fp4_pair(div_rn_f32(values[j * 2], scale),
                                           div_rn_f32(values[j * 2 + 1], scale))
                        if amax == Float32(0.0):
                            packed = Uint32(0)
                        out[base + col // Int64(2) + Int64(j)] = Uint8(packed)
                if lane % Int32(4) == Int32(0) and col < h:
                    out[base + h // Int64(2) + col // Int64(16)] = Uint8(code)
            else:
                if amax > Float32(0.0):
                    scale, code = fp8_scale(amax)
                base = Int64(row) * (h + h // Int64(32))
                for j in cutlass.range_constexpr(4):
                    if col + Int64(j) < h:
                        value = cvt_f32_to_e4m3(div_rn_f32(values[j], scale))
                        if amax == Float32(0.0):
                            value = Uint32(0)
                        out[base + col + Int64(j)] = Uint8(value)
                if lane % Int32(8) == Int32(0) and col < h:
                    out[base + h + col // Int64(32)] = Uint8(code)
        else:
            for j in cutlass.range_constexpr(4):
                c = col + Int64(j)
                if c < h:
                    value = Float32(0.0)
                    if selected == Int32(2):
                        base = Int64(row) * (h // Int64(2) + h // Int64(16))
                        packed = Uint32(source[base + c // Int64(2)])
                        code = (packed >> (Uint32(c % Int64(2)) * Uint32(4))) & Uint32(15)
                        scale = cvt_e4m3_to_f32_via_f16(Uint32(source[base + h // Int64(2) + c // Int64(16)]))
                        value = _mul(decode_fp4(code), scale)
                    else:
                        base = Int64(row) * (h + h // Int64(32))
                        code = Uint32(source[base + h + c // Int64(32)])
                        # UE8M0 code zero is 2^-127, not IEEE zero.
                        scale = _ue8m0_scale(code)
                        value = _mul(cvt_e4m3_to_f32_via_f16(Uint32(source[base + c])), scale)
                    out[Int64(row) * h + c] = BFloat16(value)


def compile_expert_reply_aot(*, quantize=False, stage=False):
    name = "expert_reply_stage" if stage else ("expert_reply_quantize" if quantize else "expert_reply_dequantize")
    return compile_program(ExpertReply(quantize, stage), name=name,
        operands=(Operand("source", torch.bfloat16 if quantize else torch.uint8, "[rows,wire]"),
                  Operand("out", torch.uint8 if quantize or stage else torch.bfloat16, "[rows,wire]", "out"),
                  Operand("mode", torch.uint32, "[1]", align=4)),
        scalars=tuple(Scalar(n) for n in ("rows", "width", "format")),
        key=(quantize, stage), geometry=dict(quantize=quantize, stage=stage, fp8_block=32, nvfp4_block=16))
