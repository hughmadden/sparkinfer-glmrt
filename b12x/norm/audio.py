"""FP32 MiMo audio support stages for native AOT integration.

GEMM and FFT belong to the resident encoder's admitted library plans. These
stages take caller-owned buffers; live row/sequence counts never specialize a
compiled object. This is a qualification path, not a mixed-BF16 default.
"""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.intrinsics import block_reduce, warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


def _add(a: Float32, b: Float32) -> Float32:
    return a + b


def _max(a: Float32, b: Float32) -> Float32:
    return cute.math.max(a, b)


class AudioSupport:
    def __init__(self, operation, width, kernel=1, stride=1):
        self.operation, self.width = operation, width
        self.kernel_size, self.stride = kernel, stride

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 out: cute.Pointer, aux: cute.Pointer, codes: cute.Pointer,
                 rows: Int32, length: Int32, offset: Int32, parameter: Float32,
                 stream: cuda.CUstream):
        self.run(x, weight, bias, out, aux, codes, rows, length, offset, parameter).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def run(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
            out: cute.Pointer, aux: cute.Pointer, codes: cute.Pointer,
            rows: Int32, length: Int32, offset: Int32, parameter: Float32):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        base = Int64(row) * Int64(self.width)
        if cutlass.const_expr(self.operation in {"layer_norm", "rms_norm", "sum_square"}):
            allocator = cutlass.utils.SmemAllocator()
            reduction = allocator.allocate_tensor(Float32, cute.make_layout((1, 8)), byte_alignment=16)
            statistics = allocator.allocate_tensor(Float32, cute.make_layout((2,)), byte_alignment=8)
            total = Float32(0.0)
            for item in cutlass.range_constexpr((self.width + 255) // 256):
                col = tid + item * 256
                if col < self.width:
                    value = Float32(x[base + Int64(col)])
                    if cutlass.const_expr(self.operation == "layer_norm"):
                        total += value
                    else:
                        total += value * value
            reduced = block_reduce(warp_reduce(total, _add), _add, reduction, Float32(0.0))
            if cutlass.const_expr(self.operation == "sum_square"):
                if tid == 0:
                    out[row] = reduced
            else:
                if tid == 0:
                    statistics[0] = reduced / Float32(self.width) if cutlass.const_expr(self.operation == "layer_norm") else Float32(0.0)
                cute.arch.sync_threads()
                mean = Float32(statistics[0])
                if cutlass.const_expr(self.operation == "layer_norm"):
                    total = Float32(0.0)
                    for item in cutlass.range_constexpr((self.width + 255) // 256):
                        col = tid + item * 256
                        if col < self.width:
                            value = Float32(x[base + Int64(col)]) - mean
                            total += value * value
                    cute.arch.sync_threads()
                    reduced = block_reduce(warp_reduce(total, _add), _add, reduction, Float32(0.0))
                if tid == 0:
                    statistics[1] = cute.math.rsqrt(reduced / Float32(self.width) + parameter)
                cute.arch.sync_threads()
                inverse = Float32(statistics[1])
                for item in cutlass.range_constexpr((self.width + 255) // 256):
                    col = tid + item * 256
                    if col < self.width:
                        value = (Float32(x[base + Int64(col)]) - mean) * inverse * Float32(weight[col])
                        if cutlass.const_expr(self.operation == "layer_norm"):
                            value += Float32(bias[col])
                        out[base + Int64(col)] = value
        elif cutlass.const_expr(self.operation == "softmax"):
            allocator = cutlass.utils.SmemAllocator()
            reduction = allocator.allocate_tensor(Float32, cute.make_layout((1, 8)), byte_alignment=16)
            statistics = allocator.allocate_tensor(Float32, cute.make_layout((2,)), byte_alignment=8)
            # rows = heads*sequence; length = sequence. offset=-1 is full
            # bidirectional group attention; 0 is causal, positive is left window.
            position = row % length
            maximum = Float32(float("-inf"))
            for col in cutlass.range(tid, length, 256):
                allowed = offset < 0 or (col <= position and (offset == 0 or position - col <= offset))
                if allowed:
                    maximum = cute.math.max(maximum, Float32(x[Int64(row) * Int64(length) + Int64(col)]) * parameter)
            maximum = block_reduce(warp_reduce(maximum, _max), _max, reduction, Float32(float("-inf")))
            if tid == 0:
                statistics[0] = maximum
            cute.arch.sync_threads()
            total = Float32(0.0)
            for col in cutlass.range(tid, length, 256):
                allowed = offset < 0 or (col <= position and (offset == 0 or position - col <= offset))
                value = Float32(0.0)
                if allowed:
                    value = cute.math.exp(Float32(x[Int64(row) * Int64(length) + Int64(col)]) * parameter - Float32(statistics[0]))
                out[Int64(row) * Int64(length) + Int64(col)] = value
                total += value
            cute.arch.sync_threads()
            total = block_reduce(warp_reduce(total, _add), _add, reduction, Float32(0.0))
            if tid == 0:
                statistics[1] = total
            cute.arch.sync_threads()
            for col in cutlass.range(tid, length, 256):
                index = Int64(row) * Int64(length) + Int64(col)
                out[index] = Float32(out[index]) / Float32(statistics[1])
        elif cutlass.const_expr(self.operation == "rvq_select"):
            # One lane chooses first-index ties with the official FP32 distance
            # association. Residual subtraction stays FP32 on all lanes.
            allocator = cutlass.utils.SmemAllocator()
            chosen = allocator.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
            if tid == 0:
                best, index = Float32(float("-inf")), Int32(0)
                for candidate in cutlass.range(length):
                    distance = -(Float32(bias[row]) - Float32(2.0) * Float32(x[Int64(row) * Int64(length) + Int64(candidate)]) + Float32(aux[candidate]))
                    if distance > best:
                        best, index = distance, candidate
                chosen[0] = index
                codes[Int64(row) * Int64(20) + Int64(offset)] = index
            cute.arch.sync_threads()
            index = Int32(chosen[0])
            for col in cutlass.range(tid, self.width, 256):
                out[base + Int64(col)] = Float32(out[base + Int64(col)]) - Float32(weight[Int64(index) * Int64(self.width) + Int64(col)])
        else:
            for item in cutlass.range_constexpr((self.width + 255) // 256):
                col = tid + item * 256
                if col < self.width:
                    dst = base + Int64(col)
                    if cutlass.const_expr(self.operation == "frame"):
                        source = (row + offset) * 240 + col - 480
                        if source < 0:
                            source = -source
                        if source >= length:
                            source = 2 * length - 2 - source
                        # Unused capacity rows are zeroed for a fixed FFT plan.
                        value = Float32(0.0)
                        if row + offset <= length // 240:
                            value = Float32(x[source]) * Float32(weight[col])
                        out[dst] = value
                    elif cutlass.const_expr(self.operation == "magnitude"):
                        real, imag = Float32(x[2 * dst]), Float32(x[2 * dst + 1])
                        out[dst] = cute.math.sqrt(real * real + imag * imag)
                    elif cutlass.const_expr(self.operation == "log"):
                        out[dst] = cute.math.log(cute.math.max(Float32(x[dst]), Float32(1e-7)))
                    elif cutlass.const_expr(self.operation == "im2col"):
                        channels = self.width // self.kernel_size
                        channel, tap = col // self.kernel_size, col % self.kernel_size
                        source = row * self.stride + tap - offset
                        value = Float32(0.0)
                        if source >= 0 and source < length:
                            value = Float32(x[Int64(source) * Int64(channels) + Int64(channel)])
                        out[dst] = value
                    elif cutlass.const_expr(self.operation in {"bias", "gelu", "bias_gelu", "silu_product", "add"}):
                        value = Float32(x[dst])
                        if cutlass.const_expr(self.operation in {"bias", "bias_gelu"}):
                            value += Float32(bias[col])
                        if cutlass.const_expr(self.operation in {"gelu", "bias_gelu"}):
                            value = Float32(0.5) * value * (Float32(1.0) + cute.math.erf(value * Float32(0.7071067811865476)))
                        elif cutlass.const_expr(self.operation == "silu_product"):
                            value = value / (Float32(1.0) + cute.math.exp(-value)) * Float32(aux[dst])
                        elif cutlass.const_expr(self.operation == "add"):
                            value += Float32(aux[dst])
                        out[dst] = value
                    elif cutlass.const_expr(self.operation == "rope_pack"):
                        # Width1024, heads16, half-split RoPE. length is each
                        # independently attended sequence (tokenizer or group4).
                        head, dim = col // 64, col % 64
                        half = dim % 32
                        partner = col + 32 if dim < 32 else col - 32
                        position = row % length
                        cosine = Float32(weight[Int64(position) * Int64(64) + Int64(half)])
                        sine = Float32(weight[Int64(position) * Int64(64) + Int64(32 + half)])
                        sign = Float32(-1.0) if dim < 32 else Float32(1.0)
                        value = Float32(x[dst]) * cosine + sign * Float32(x[base + Int64(partner)]) * sine
                        destination = ((Int64(row // length) * Int64(16) + Int64(head)) * Int64(length) + Int64(position)) * Int64(64) + Int64(dim)
                        out[destination] = value
                    elif cutlass.const_expr(self.operation in {"pack_heads", "unpack_heads"}):
                        head, dim = col // 64, col % 64
                        packed = ((Int64(row // length) * Int64(16) + Int64(head)) * Int64(length) + Int64(row % length)) * Int64(64) + Int64(dim)
                        if cutlass.const_expr(self.operation == "pack_heads"):
                            out[packed] = x[dst]
                        else:
                            out[dst] = x[packed]
                    elif cutlass.const_expr(self.operation == "speech_add"):
                        frame = cute.math.min(row, length - 1)
                        code = Int32(codes[Int64(frame) * Int64(20) + Int64(offset)])
                        out[dst] = Float32(out[dst]) + Float32(weight[Int64(code) * Int64(1024) + Int64(col)])


@cache
def _compile(operation, width, kernel, stride, device):
    key = (operation, width, kernel, stride, device)
    entry = AudioSupport(operation, width, kernel, stride)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = (Float32, Float32, Float32, Float32, Float32, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int32(1), Int32(0),
            Float32(1e-5), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("norm.audio." + operation, 1, key))
    return compiled, types


from ._audio_preparation import AudioQuery, AudioState, TUNING, plan

__all__ = ["AudioSupport", "AudioQuery", "AudioState", "TUNING", "plan"]
