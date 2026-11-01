"""Exportable V4.1 Spark slice pipeline with caller-owned persistent scratch.

The ordered default writes FP32 route planes. Opt-in atomic_tokens clears and
writes a flat FP32 token output without materializing route/slice intermediates.
Both paths keep live rows out of compile keys and bound launches by live work.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Float32
from b12x._lib.utils import runtime_num_sm
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan, V41SliceReduce
from b12x.moe._shared.kernels.w4a8_v41_slice import (
    V41FusedSliceKernel,
    V41SliceFc1Kernel,
    V41SliceFc2Kernel,
)


# GB10 ordered decode winner; keep every decode capacity on one accumulation order.
FC1_DECODE_K_SPLITS = {(121, 5120, 384, 128, False, True): 2}


def fc1_decode_k_splits(sm, hidden, intermediate, width, exact_storage, scale_codes):
    return FC1_DECODE_K_SPLITS.get((sm, hidden, intermediate, width, exact_storage, scale_codes), 1)


class V41SlicePipeline:
    def __init__(self, capacity, width, atomic_tokens=False, *, experts=384, topk=6, intermediate=576,
                 hidden=5120, exact_storage=False, fc2_n_splits=1, scale_codes=False,
                 fc1_k_splits=None):
        self.capacity = capacity
        self.hidden = hidden
        self.atomic_tokens = atomic_tokens
        self.experts = experts
        self.topk = topk
        if fc1_k_splits is None:
            fc1_k_splits = 1
            if capacity <= 80 and not atomic_tokens:
                from b12x._lib.intrinsics import get_sm_version
                fc1_k_splits = fc1_decode_k_splits(get_sm_version(), hidden, intermediate,
                                                  width, exact_storage, scale_codes)
        if atomic_tokens and fc1_k_splits != 1:
            raise ValueError("FC1 K splits require ordered route output")
        self.plan = V41RoutePlan(capacity, experts, topk, route_metadata=not atomic_tokens)
        if atomic_tokens:
            self.compute = V41FusedSliceKernel(width, grouped=True, atomic_tokens=True,
                                               intermediate=intermediate, hidden=hidden, exact_storage=exact_storage,
                                               fc2_n_splits=fc2_n_splits, scale_codes=scale_codes)
            self.fc1 = None
            self.fc2 = None
        else:
            self.compute = None
            self.fc1 = V41SliceFc1Kernel(width, capacity, topk=topk, intermediate=intermediate,
                                         hidden=hidden, exact_storage=exact_storage, scale_codes=scale_codes,
                                         fc1_k_splits=fc1_k_splits)
            self.fc2 = V41SliceFc2Kernel(width, capacity, topk=topk, intermediate=intermediate,
                                         hidden=hidden, exact_storage=exact_storage, fc2_n_splits=fc2_n_splits,
                                         scale_codes=scale_codes)
        self.reduce = V41SliceReduce(width, capacity, topk, intermediate=intermediate, hidden=hidden)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        ids: cute.Tensor,
        routing: cute.Tensor,
        live: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        inverse: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.publish(live, rows).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        self.plan(
            ids,
            routing,
            live,
            packed,
            counts,
            prefixes,
            metadata,
            grouped,
            inverse,
            stream,
        )
        groups = max(1, min(rows * self.topk, self.experts + max(rows * self.topk - self.experts, 0) // 16))
        if cutlass.const_expr(self.atomic_tokens):
            self.clear_output(output, rows).launch(
                grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
                block=(256, 1, 1), stream=stream,
            )
            self.compute(
                x,
                xs,
                w13,
                s13,
                w2,
                s2,
                grouped,
                output,
                rows,
                stream,
                metadata,
                groups,
            )
        else:
            self.fc1(x, xs, w13, s13, grouped, partial, metadata, groups, stream)
            self.fc2(w2, s2, partial, output, metadata, inverse, live, groups, stream)

    @cute.kernel
    def publish(self, live: cute.Tensor, rows: Int32):
        if cute.arch.thread_idx()[0] == 0:
            live[0] = rows

    @cute.kernel
    def clear_output(self, output: cute.Tensor, rows: Int32):
        i = Int64(cute.arch.block_idx()[0]) * 256 + Int64(cute.arch.thread_idx()[0])
        if i < Int64(rows) * self.hidden:
            output[i] = Float32(0)


class V41DraftSlicePipeline:
    """Local dSpark BF16 input through FP8 quantization and ordered route output.

    All temporaries belong to the caller. The quantizer uses the same K32,
    subgroup-four, 1e-4-floor contract as the native input quantizer.
    """

    # Qualified model recipe; explicit widths remain available for comparison.
    DEFAULT_WIDTH = 192

    def __init__(self, capacity, width, *, scale_codes=False):
        from b12x._lib.quant.mxfp8_rows import _MXFP8RowsQuantLaunch
        self.quant = _MXFP8RowsQuantLaunch(
            5120, cutlass.BFloat16, 4, 256, 32, False, 1e-4, wire_rows=True
        )
        self.pipeline = V41SlicePipeline(
            capacity, width, experts=128, topk=3, intermediate=2304,
            scale_codes=scale_codes
        )

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        ids: cute.Tensor,
        routing: cute.Tensor,
        live: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        inverse: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        # 160 K32 groups / row, 8 groups / warp, 8 warps / CTA.
        grid = min(max(1, (rows * 20 + 7) // 8), runtime_num_sm() * Int32(4))
        self.quant(source.iterator, x.iterator, xs.iterator, xs.iterator,
                   rows, Int32(5120), grid, stream)
        self.pipeline(x, xs, w13, s13, w2, s2, ids, routing, live, packed,
                      counts, prefixes, metadata, grouped, inverse, partial,
                      output, rows, stream)
