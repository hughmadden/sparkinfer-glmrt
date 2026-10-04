"""Native Spark/TP2 compact pipeline consuming the existing FP8 K32 wire rows.

The signature matches V41SlicePipeline so the native expert ABI is unchanged.
Only intermediate, projections and FP32 route output require scratch. Native
outer scales are compile-time ones: capacity variants share and overwrite the
same arena, so no persistent initialized constant may be read from scratch.
"""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Uint8, Uint32, BFloat16, Float32
from b12x.moe._shared.kernels.w4a8_compact_micro import _DirectW4A8CompactLaunch


class V41CompactPipeline:
    def __init__(self, capacity, sm_count, *, kernel_intermediate=1152):
        self.sm_count = int(sm_count)
        self.compact = _DirectW4A8CompactLaunch(
            max_tokens=capacity, num_topk=6, k=5120, n=kernel_intermediate, experts=384,
            input_scale_count=384, down_scale_count=384, swiglu_limit=10,
            fast_math=False, native_v41=True, wire_rows=True, unit_scales=True)

    @cute.jit
    def __call__(
        self, x: cute.Tensor, xs: cute.Tensor,
        w13: cute.Tensor, s13: cute.Tensor, w2: cute.Tensor, s2: cute.Tensor,
        ids: cute.Tensor, routing: cute.Tensor, live: cute.Tensor,
        packed: cute.Tensor, counts: cute.Tensor, prefixes: cute.Tensor,
        metadata: cute.Tensor, grouped: cute.Tensor, inverse: cute.Tensor,
        partial: cute.Tensor, output: cute.Tensor,
        rows: Int32, stream: cuda.CUstream,
    ):
        self.compact(
            cute.recast_ptr(x.iterator, dtype=Uint8), xs.iterator,
            w13.iterator, s13.iterator, cute.recast_ptr(packed.iterator, dtype=Uint32),
            cute.recast_ptr(partial.iterator, dtype=BFloat16),
            routing.iterator, ids.iterator, cute.recast_ptr(counts.iterator, dtype=Float32), cute.recast_ptr(counts.iterator, dtype=Float32),
            w2.iterator, s2.iterator, output.iterator, cute.recast_ptr(counts.iterator, dtype=Float32),
            cute.recast_ptr(counts.iterator, dtype=Float32), rows * Int32(6), Int32(self.sm_count), stream)


class V41HybridPipeline:
    """Choose compact or grouped execution from live rows under one capacity.

    Both paths share the grouped pipeline's caller-owned scratch. Compact
    materialization fits in packed routing storage, and BF16 projections fit
    in the FP32 slice partial arena. Neither path retains state between calls.
    """

    def __init__(self, capacity, width, sm_count, *, intermediate, cutoff, exact_storage=False):
        from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
        if intermediate not in (576, 1152) or not 1 <= cutoff <= capacity:
            raise ValueError("invalid native hybrid geometry or cutoff")
        self.cutoff = int(cutoff)
        self.compact = V41CompactPipeline(capacity, sm_count,
            kernel_intermediate=intermediate if exact_storage else (intermediate + 127) // 128 * 128)
        self.grouped = V41SlicePipeline(capacity, width, intermediate=intermediate, exact_storage=exact_storage)

    @cute.jit
    def __call__(
        self, x: cute.Tensor, xs: cute.Tensor,
        w13: cute.Tensor, s13: cute.Tensor, w2: cute.Tensor, s2: cute.Tensor,
        ids: cute.Tensor, routing: cute.Tensor, live: cute.Tensor,
        packed: cute.Tensor, counts: cute.Tensor, prefixes: cute.Tensor,
        metadata: cute.Tensor, grouped: cute.Tensor, inverse: cute.Tensor,
        partial: cute.Tensor, output: cute.Tensor,
        rows: Int32, stream: cuda.CUstream,
    ):
        if rows <= Int32(self.cutoff):
            self.compact(x, xs, w13, s13, w2, s2, ids, routing, live,
                packed, counts, prefixes, metadata, grouped, inverse,
                partial, output, rows, stream)
        else:
            self.grouped(x, xs, w13, s13, w2, s2, ids, routing, live,
                packed, counts, prefixes, metadata, grouped, inverse,
                partial, output, rows, stream)
