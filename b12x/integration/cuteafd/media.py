"""Shared BF16 media boundaries; extents and table contents stay runtime data."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _mul, _view


class AxialRotary:
    def __init__(self, heads, dim, bf16_products):
        self.heads, self.dim = heads, dim
        self.bf16_products = bf16_products

    @cute.jit
    def __call__(self, q: cute.Pointer, k: cute.Pointer, cos: cute.Pointer,
                 sin: cute.Pointer, out_q: cute.Pointer, out_k: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.heads * self.dim)
        self.kernel(q, k, cos, sin, out_q, out_k, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Pointer, k: cute.Pointer, cos: cute.Pointer,
               sin: cute.Pointer, out_q: cute.Pointer, out_k: cute.Pointer, count: Int64):
        q, k, cos, sin = _view(q), _view(k), _view(cos), _view(sin)
        out_q, out_k = _view(out_q), _view(out_k)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            dim = at % Int64(self.dim)
            pair = at - dim + (dim + Int64(self.dim // 2)) % Int64(self.dim)
            table = (at // Int64(self.heads * self.dim)) * Int64(self.dim) + dim
            c, s = Float32(cos[table]), Float32(sin[table])
            sign = Float32(1.0)
            if dim < Int64(self.dim // 2):
                sign = Float32(-1.0)
            for x, out in ((q, out_q), (k, out_k)):
                a = _mul(Float32(x[at]), c)
                b = _mul(_mul(sign, Float32(x[pair])), s)
                if cutlass.const_expr(self.bf16_products):
                    a, b = Float32(a.to(BFloat16)), Float32(b.to(BFloat16))
                out[at] = _add(a, b).to(BFloat16)


class PositionGather:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, table: cute.Pointer, indices: cute.Pointer, weights: cute.Pointer,
                 patch: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.width)
        self.kernel(table, indices, weights, patch, out, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, table: cute.Pointer, indices: cute.Pointer, weights: cute.Pointer,
               patch: cute.Pointer, out: cute.Pointer, count: Int64):
        table, indices, weights = _view(table), _view(indices), _view(weights)
        patch, out = _view(patch), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            row, col = at // Int64(self.width), at % Int64(self.width)
            value = Float32(0.0)
            for tap in cutlass.range_constexpr(4):
                ix = Int64(indices[row * Int64(4) + Int64(tap)])
                term = _mul(Float32(table[ix * Int64(self.width) + col]),
                            Float32(weights[row * Int64(4) + Int64(tap)]))
                if cutlass.const_expr(tap == 0):
                    value = term
                else:
                    value = _add(value, term)
            position = Float32(value.to(BFloat16))
            out[at] = _add(Float32(patch[at]), position).to(BFloat16)


class CausalDepthwiseConv7:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 history: cute.Pointer, out: cute.Pointer, next_history: cute.Pointer,
                 batch: Int32, length: Int32, stream: cuda.CUstream):
        count = Int64(batch) * Int64(length) * Int64(self.width)
        self.convolve(x, weight, bias, history, out, length, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)
        # Separate launch permits history == next_history: all convolution reads
        # finish first; update reads each channel's old taps before any writes.
        channels = Int64(batch) * Int64(self.width)
        self.update(x, history, next_history, length, channels).launch(
            grid=((channels + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def convolve(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 history: cute.Pointer, out: cute.Pointer, length: Int32, count: Int64):
        x, weight, bias, history, out = _view(x), _view(weight), _view(bias), _view(history), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            col = at % Int64(self.width)
            row = at // Int64(self.width)
            batch, time = row // Int64(length), row % Int64(length)
            total = Float32(0.0)
            for tap in cutlass.range_constexpr(7):
                pos = time - Int64(6) + Int64(tap)
                value = Float32(0.0)
                if pos < Int64(0):
                    value = Float32(history[(batch * Int64(self.width) + col) * Int64(6) + pos + Int64(6)])
                else:
                    value = Float32(x[(batch * Int64(length) + pos) * Int64(self.width) + col])
                total = _add(total, _mul(value, Float32(weight[col * Int64(7) + Int64(tap)])))
            out[at] = _add(total, Float32(bias[col])).to(BFloat16)

    @cute.kernel
    def update(self, x: cute.Pointer, history: cute.Pointer, next_history: cute.Pointer,
               length: Int32, channels: Int64):
        x, history, next_history = _view(x), _view(history), _view(next_history)
        channel = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if channel < channels:
            batch, col = channel // Int64(self.width), channel % Int64(self.width)
            held = []
            for tap in cutlass.range_constexpr(6):
                pos = Int64(length) - Int64(6) + Int64(tap)
                value = BFloat16(0.0)
                if pos < Int64(0):
                    value = history[channel * Int64(6) + pos + Int64(6)]
                else:
                    value = x[(batch * Int64(length) + pos) * Int64(self.width) + col]
                held.append(value)
            for tap in cutlass.range_constexpr(6):
                next_history[channel * Int64(6) + Int64(tap)] = held[tap]


def compile_axial_rotary_aot(*, heads=16, dim, bf16_products=False):
    """Full repeated BF16 cos/sin rows encode arbitrary axial recipes/permutations.

    Outputs must not alias Q/K: a paired component can be read by another CTA.
    """
    if heads <= 0 or dim <= 0 or dim % 2:
        raise ValueError("rotary requires positive heads and an even positive dimension")
    return compile_program(AxialRotary(heads, dim, bool(bf16_products)), name="media_axial_rotary",
        operands=(Operand("q", torch.bfloat16, f"[rows,{heads},{dim}]"),
                  Operand("k", torch.bfloat16, f"[rows,{heads},{dim}]"),
                  Operand("cos", torch.bfloat16, f"[rows,{dim}]"),
                  Operand("sin", torch.bfloat16, f"[rows,{dim}]"),
                  Operand("out_q", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
                  Operand("out_k", torch.bfloat16, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"),), key=(heads, dim, bool(bf16_products)),
        geometry=dict(heads=heads, dim=dim, bf16_products=bool(bf16_products)))


def compile_position_gather_aot(*, width=1536):
    """Indices must address valid table rows, including zero-weight taps."""
    if width <= 0:
        raise ValueError("position gather width must be positive")
    return compile_program(PositionGather(width), name="media_position_gather",
        operands=(Operand("table", torch.bfloat16, f"[table_rows,{width}]"),
                  Operand("indices", torch.int32, "[rows,4]"),
                  Operand("weights", torch.float32, "[rows,4]"),
                  Operand("patch", torch.bfloat16, f"[rows,{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,), geometry=dict(width=width, taps=4))


def compile_causal_depthwise_conv7_aot(*, width):
    """Positive batch/length; x/out and x/history must be disjoint.

    Old/new history may alias. The caller drains this stream before releasing
    any buffer; history stores raw input BF16 bits, never convolution output.
    """
    if width <= 0:
        raise ValueError("convolution width must be positive")
    return compile_program(CausalDepthwiseConv7(width), name="media_causal_depthwise_conv7",
        operands=(Operand("x", torch.bfloat16, f"[batch,length,{width}]"),
                  Operand("weight", torch.bfloat16, f"[{width},7]"),
                  Operand("bias", torch.bfloat16, f"[{width}]"),
                  Operand("history", torch.bfloat16, f"[batch,{width},6]"),
                  Operand("out", torch.bfloat16, f"[batch,length,{width}]", "out"),
                  Operand("next_history", torch.bfloat16, f"[batch,{width},6]", "out")),
        scalars=(Scalar("batch"), Scalar("length")), key=(width,),
        geometry=dict(width=width, kernel=7, history=6, layout="BTH"))


class BiasedProjection:
    def __init__(self, width, outputs):
        from b12x.gemm.bf16_gemv._kernel import Bf16GemmKernel
        self.width, self.outputs = width, outputs
        self.projection = Bf16GemmKernel(outputs, width, True)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.projection(x, weight, bias, out, rows, Int64(self.width), Int64(self.width),
                        Int64(self.outputs), Int64(1), Int64(1), Int32(0), stream)


def compile_biased_projection_aot(*, width, outputs):
    """Existing library MMA epilogue: F32 accumulator + BF16 bias, one BF16 store.

    No intermediate BF16 matmul result; the output row extent remains runtime.
    """
    if min(width, outputs) <= 0:
        raise ValueError("projection dimensions must be positive")
    return compile_program(BiasedProjection(width, outputs), name="media_biased_projection",
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("weight", torch.bfloat16, f"[{outputs},{width}]"),
                  Operand("bias", torch.bfloat16, f"[{outputs}]"),
                  Operand("out", torch.bfloat16, f"[rows,{outputs}]", "out")),
        scalars=(Scalar("rows"),), key=(width, outputs),
        geometry=dict(width=width, outputs=outputs, bias_before_cast=True))


class Gelu:
    def __init__(self, width):
        from b12x.norm.vision import _Vision
        self.launch = _Vision("gelu", width, 1, 1)

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.launch(x, out, out, out, out, out, rows, Int32(1), Int32(1), stream)


def compile_gelu_aot(*, width):
    """ABI adapter to the existing BF16 erf GELU, not a second implementation."""
    if width <= 0:
        raise ValueError("GELU width must be positive")
    return compile_program(Gelu(width), name="media_gelu",
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,), geometry=dict(width=width))


def compile_bf16_residual_add_aot(*, width):
    """Reuse the vectorized BF16 add, preserving its single BF16 store."""
    from ._qwen4_kernels import Qwen4Add
    if width <= 0 or width % 256 or width > 8192:
        raise ValueError("BF16 residual width must be a multiple of 256 in 256..8192")
    return compile_program(Qwen4Add(width), name="media_residual_add_bf16", operands=(
        Operand("a", torch.bfloat16, f"[rows,{width}]"),
        Operand("b", torch.bfloat16, f"[rows,{width}]"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,), geometry=dict(width=width))


class SpatialMerge:
    def __init__(self, width, ratio):
        from b12x.norm.vision import _Vision
        self.launch = _Vision("merge", width, 1, ratio)

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32,
                 height: Int32, width: Int32, stream: cuda.CUstream):
        self.launch(x, out, out, out, out, out, rows, height, width, stream)


def compile_spatial_merge_aot(*, width, ratio):
    """Existing channel-major spatial gather; rows is the merged output count.

    Input raster is [height,width,channels], output [ceil(H/r),ceil(W/r),C,r,r].
    Input and output must be disjoint. Edge padding retains the library zeros.
    """
    if min(width, ratio) <= 0:
        raise ValueError("spatial merge requires positive channel width and ratio")
    return compile_program(SpatialMerge(width, ratio), name="media_spatial_merge",
        operands=(Operand("x", torch.bfloat16, f"[height,width,{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width * ratio * ratio}]", "out")),
        scalars=(Scalar("rows"), Scalar("height"), Scalar("width")), key=(width, ratio),
        geometry=dict(width=width, ratio=ratio, layout="channel_spatial"))


def vision_transformer_programs(*, hidden, intermediate, patch_width, merge,
                                adapter_width, output_width):
    """Native BF16 affine/GELU vision stack, expressed by checkpoint geometry.

    Reuses existing arithmetic kernels and byte-copy gather. Live image rows,
    permutation indices and attention/window tables remain launch inputs.
    """
    from .pointwise import compile_affine_norm_aot, compile_bf16_projection_aot
    from .context_split import compile_paged_staging_gather_aot
    if min(hidden, intermediate, patch_width, merge, adapter_width, output_width) <= 0:
        raise ValueError("vision transformer geometry must be positive")
    result = {
        f"layer_norm_h{hidden}": lambda: compile_affine_norm_aot(width=hidden,
            subtract_mean=True, bias=True, input_dtype=torch.bfloat16,
            parameter_dtype=torch.bfloat16),
        f"residual_h{hidden}": lambda: compile_bf16_residual_add_aot(width=hidden),
        f"row_gather_h{hidden}": lambda: compile_paged_staging_gather_aot(
            row_bytes=hidden * 2, page_rows=1),
        f"spatial_merge_h{hidden}_r{merge}": lambda: compile_spatial_merge_aot(width=hidden, ratio=merge),
    }
    for width, outputs in ((hidden, 3 * hidden), (hidden, hidden),
                           (hidden, intermediate), (intermediate, hidden)):
        result[f"biased_projection_h{width}_o{outputs}"] = (
            lambda w=width, o=outputs: compile_biased_projection_aot(width=w, outputs=o))
    for width, outputs in ((patch_width, hidden), (hidden * merge * merge, adapter_width),
                           (adapter_width, adapter_width), (adapter_width, output_width)):
        result[f"projection_h{width}_o{outputs}"] = (
            lambda w=width, o=outputs: compile_bf16_projection_aot(width=w, outputs=o))
    for width in dict.fromkeys((intermediate, adapter_width)):
        result[f"gelu_h{width}"] = lambda w=width: compile_gelu_aot(width=w)
    return result


def vision_programs(*, dim, hidden, bf16_products=False,
                    learned_position=False, bf16_before_gamma=False):
    """Geometry-only declarations; callers derive these values from tower config."""
    from .pointwise import compile_affine_norm_aot
    result = {
        f"axial_rotary_d{dim}_{'bf16' if bf16_products else 'f32'}":
            lambda: compile_axial_rotary_aot(dim=dim, bf16_products=bf16_products),
    }
    if learned_position:
        result[f"position_gather_h{hidden}"] = lambda: compile_position_gather_aot(width=hidden)
    if bf16_before_gamma:
        result[f"rms_bf16_gamma_h{hidden}"] = lambda: compile_affine_norm_aot(width=hidden,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16,
            round_normalized_before_weight=True)
    return result


def audio_programs(*, widths=(256, 512, 1024, 2048), output_width=8192):
    """Causal PCM support using config stage widths, never request extents."""
    from .pointwise import compile_affine_norm_aot
    result = {}
    for width in widths:
        result[f"causal_conv7_h{width}"] = lambda w=width: compile_causal_depthwise_conv7_aot(width=w)
        result[f"rms_h{width}"] = lambda w=width: compile_affine_norm_aot(width=w, eps=1e-6,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16)
        result[f"gelu_h{width * 4}"] = lambda w=width: compile_gelu_aot(width=w * 4)
        result[f"residual_h{width}"] = lambda w=width: compile_bf16_residual_add_aot(width=w)
    result[f"rms_h{output_width}"] = lambda: compile_affine_norm_aot(width=output_width, eps=1e-6,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16)
    return result


class VisionAttention:
    def __init__(self, dim, capacity, segments, max_segment_rows):
        from b12x.attention._shared.contiguous.api import _VarlenAttentionForwardLaunch
        self.launch = _VarlenAttentionForwardLaunch(
            q_shape=(capacity, 16, dim), k_shape=(capacity, 16, dim),
            v_shape=(capacity, 16, dim), cu_seqlens_q_shape=(segments + 1,),
            cu_seqlens_k_shape=(segments + 1,), dtype=torch.bfloat16,
            causal=False, window_size_left=-1, window_size_right=-1,
            has_attention_sink_bias=False, max_seqlen_q=max_segment_rows,
            max_seqlen_k=max_segment_rows, tile_m=128, tile_n=64)

    @cute.jit
    def __call__(self, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 out: cute.Pointer, lse: cute.Pointer, cu: cute.Pointer,
                 scale: Float32, stream: cuda.CUstream):
        self.launch(q, k, v, out, lse, cu, cu, lse, scale, stream)


def compile_vision_attention_aot(*, dim, windowed=False):
    """Fixed admission capacities, never compile from live image/window counts.

    Windowed input is already permuted into independent spatial32x32 segments;
    pad cu with the last live offset through the fixed 512-segment capacity.
    Full attention uses a single segment; multi-image callers launch per image.
    Empty segments must not write O/LSE. Q/K/V/O storage capacity is 16384 rows.
    """
    if dim not in (96, 128) or (windowed and dim != 96):
        raise ValueError("vision attention requires d96/d128; windows require d96")
    capacity, segments = 16384, 512 if windowed else 1
    limit = 1024 if windowed else capacity
    return compile_program(VisionAttention(dim, capacity, segments, limit),
        name="media_vision_attention", operands=(
            Operand("q", torch.bfloat16, f"[{capacity},16,{dim}]"),
            Operand("k", torch.bfloat16, f"[{capacity},16,{dim}]"),
            Operand("v", torch.bfloat16, f"[{capacity},16,{dim}]"),
            Operand("out", torch.bfloat16, f"[{capacity},16,{dim}]", "out"),
            Operand("lse", torch.float32, f"[16,{capacity}]", "scratch"),
            Operand("cu", torch.int32, f"[{segments + 1}]")),
        scalars=(Scalar("scale", "float32"),), key=(dim, bool(windowed)),
        geometry=dict(dim=dim, heads=16, capacity=capacity, segments=segments,
                      max_segment_rows=limit, windowed=bool(windowed)))
