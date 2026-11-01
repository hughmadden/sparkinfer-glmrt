"""Ordered native MXFP4 V4/V4.1 prefill over caller-owned scratch.

Weights and scales retain their decode residency. FP32 token output is consumed
by the native engine's existing BF16 token compactor; no floating-point atomics
or resident repacking are used.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from b12x.integration.cuteafd._fp8_moe_kernels import MoePrep, meta_words
from b12x.integration.cuteafd._fp8_moe_stream import stream_max_tiles
from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp8_down_row_bytes
from b12x.integration.cuteafd._mxfp4_moe_a8 import StreamMxfp4DownA8
from b12x.integration.cuteafd._mxfp4_moe_stream import StreamMxfp4GateUp
from b12x.moe._shared.kernels.w4a8_v41_slice import fadd_rn_f32


def stream_scratch_specs(capacity, hidden, intermediate, experts, topk):
    """Static tensor specs, shared by the exporter and scratch-size query."""
    if (capacity not in (256, 1024, 4096) or hidden <= 0 or hidden % 512
            or intermediate <= 0 or intermediate % 32 or experts <= 0 or topk <= 0):
        raise ValueError("invalid native V4 prefill geometry")
    pairs = capacity * topk
    tiles = stream_max_tiles(experts, pairs)
    return (("meta", "i32", (meta_words(experts, tiles),)),
            ("pair_row", "i32", (pairs,)), ("pair_pos", "i32", (pairs,)),
            ("act", "u8", (tiles * 128 * mxfp8_down_row_bytes(intermediate),)),
            ("down", "f32", (pairs, hidden)), ("grouped_routing", "f32", (pairs,)),
            ("output", "f32", (capacity, hidden)))


def stream_scratch_layout(capacity, hidden, intermediate, experts, topk):
    offset, layout = 0, []
    for name, dtype, shape in stream_scratch_specs(capacity, hidden, intermediate, experts, topk):
        offset = (offset + 15) // 16 * 16
        size = 1 if dtype == "u8" else 4
        for extent in shape:
            size *= extent
        layout.append(dict(name=name, dtype=dtype, shape=shape, offset=offset, nbytes=size))
        offset += size
    return tuple(layout), offset


class NativeTokenCombine:
    """Native FP32 routes/tokens to BF16, with an explicit ordered slot fold."""

    def __init__(self, hidden, topk):
        self.hidden, self.topk = hidden, topk

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, rows: Int32, stream):
        self.kernel(source, output, rows).launch(
            grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, output: cute.Pointer, rows: Int32):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if index < Int64(rows) * self.hidden:
            row, col = index // self.hidden, index % self.hidden
            value = Float32(0)
            for slot in cutlass.range_constexpr(self.topk):
                value = fadd_rn_f32(value, Float32(source[(row * self.topk + slot) * self.hidden + col]))
            output[index] = value.to(cutlass.BFloat16)


class V4StreamPipeline:
    """One ordered A8 prefill recipe; native slice decode remains separate."""

    def __init__(self, capacity, *, hidden, intermediate, experts, topk=6, scale_codes=False):
        self.layout, self.scratch_bytes = stream_scratch_layout(
            capacity, hidden, intermediate, experts, topk)
        self.capacity, self.hidden, self.intermediate = capacity, hidden, intermediate
        self.experts, self.topk, self.scale_codes = experts, topk, bool(scale_codes)
        self.max_tiles = stream_max_tiles(experts, capacity * topk)
        self.prep = MoePrep(experts=experts, top_k=topk, pad=1, max_tiles=self.max_tiles,
                            tile_rows=128, chunked=True)
        self.gate_up = StreamMxfp4GateUp(hidden=hidden, inter=intermediate, experts=experts,
            limit=10., qmma=True, wire=True, act_mxfp8=True, silu_bf16_product=False,
            route_weight_before_fc2=True, native_resident=True,
            native_exact=intermediate % 128 != 0, scale_codes=scale_codes)
        self.down = StreamMxfp4DownA8(hidden=hidden, inter=intermediate, experts=experts,
            round_output_bf16=False, native_resident=True,
            native_exact=intermediate % 128 != 0, scale_codes=scale_codes)

    def key(self):
        return ("v4_native_stream", 1, self.capacity, self.hidden, self.intermediate,
                self.experts, self.topk, self.scale_codes, self.gate_up.key(), self.down.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w13: cute.Pointer, s13: cute.Pointer,
                 w2: cute.Pointer, s2: cute.Pointer, ids: cute.Pointer, routing: cute.Pointer,
                 meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer,
                 act: cute.Pointer, down: cute.Pointer, grouped_routing: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream):
        pairs = rows * self.topk
        tiles = (pairs + 127) // 128 + min(pairs, self.experts)
        self.prep(ids, meta, pair_row, pair_pos, rows, pairs, stream)
        self.group_weights(routing, pair_pos, grouped_routing, pairs).launch(
            grid=(max(1, (pairs + 255) // 256), 1, 1), block=(256, 1, 1), stream=stream)
        self.gate_up(x, pair_row, meta, w13, s13, w13, s13, act, tiles, stream, grouped_routing)
        self.down(act, meta, w2, s2, down, tiles, stream)
        self.ordered_combine(down, pair_pos, output, rows).launch(
            grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def group_weights(self, routing: cute.Pointer, positions: cute.Pointer,
                      grouped: cute.Pointer, pairs: Int32):
        index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if index < pairs:
            position = positions[index]
            if position >= 0:
                grouped[position] = routing[index]

    @cute.kernel
    def ordered_combine(self, down: cute.Pointer, positions: cute.Pointer,
                        output: cute.Pointer, rows: Int32):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if index < Int64(rows) * self.hidden:
            row, col = index // self.hidden, index % self.hidden
            value = Float32(0)
            for slot in cutlass.range_constexpr(self.topk):
                position = positions[row * self.topk + slot]
                if position >= 0:
                    value = fadd_rn_f32(value, Float32(down[Int64(position) * self.hidden + col]))
            output[index] = value
