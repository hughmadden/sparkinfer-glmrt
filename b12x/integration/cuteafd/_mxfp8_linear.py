"""Native MXFP8 W8A8 composition with checkpoint row-scale preparation."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.dense_gemm import _DenseGemmLaunch, _DenseGemmPolicy, _DenseSplitKReduce
from b12x._lib.quant.mxfp8_rows import _MXFP8RowsQuantLaunch
from ._linear import Fp8LinearStage, StoreOne, stage_scratch_bytes


class Mxfp8LinearStage(Fp8LinearStage):
    """K32 E4M3/UE8M0 activations and weights, hardware MXFP8 MMA.

    ``slices`` > 1 splits K into FP32 partial planes summed once (one BF16
    rounding): decode-sized rows fill one M tile, so only K can widen the
    grid of a weight-streaming projection past its N/128 tiles.
    """

    def __init__(self, k, n, slices=1):
        self.k, self.n, self.slices = int(k), int(n), int(slices)
        if self.slices < 1 or (self.k // 128) % self.slices:
            raise ValueError("MXFP8 split-K slices must divide the K/128 tiles")
        self.quant = _MXFP8RowsQuantLaunch(
            self.k, cutlass.BFloat16, 4, 256, 32, False, 0.0, False)
        self.quant_warps = 8
        self.quant_tasks_per_row = (self.k // 32 + 7) // 8
        self.reduce = _DenseSplitKReduce(self.n, self.slices) if self.slices > 1 else None
        self.store_one = StoreOne()
        self.gemm = _DenseGemmLaunch(
            n=self.n, k=self.k, l=1, c_l=self.slices, a_major="k", b_major="k", c_major="n",
            ab_dtype=cutlass.Float8E4M3FN, sf_dtype=cutlass.Float8E8M0FNU,
            c_dtype=cutlass.Float32 if self.slices > 1 else cutlass.BFloat16,
            alpha_dtype=cutlass.Float32, sf_vec_size=32,
            mma_k=32, tile_k=128, mma_tiler_mn=(64, 128), cluster_shape_mn=(1, 1),
            policy=_DenseGemmPolicy(False, False, False, self.slices, False, False),
            # The single-cluster launch queries runtime_num_sm; this constructor
            # field is unused by generated code and cannot specialize by card.
            sm_count=1, sm_version="sm_120", load_path="tma", swap_ab=False,
            sfb_k_reuse=False, alpha_is_one=True)

    def scratch_bytes(self, rows):
        return stage_scratch_bytes(self.k, self.n, rows, self.slices)


class Mxfp8Linear:
    def __init__(self, k, n, slices=1):
        self.stage = Mxfp8LinearStage(k, n, slices)

    @cute.jit
    def __call__(self, x: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer,
                 partial: cute.Pointer, output: cute.Pointer, rows: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        self.stage(x, payload, scales, output, Int64(partial.toint()), rows, stream)


class Mxfp8ScalePrep:
    @cute.jit
    def __call__(self, scale: cute.Pointer, scale_mma: cute.Pointer, outputs: Int32,
                 input_groups: Int32, stream: cuda.CUstream):
        count = Int64(outputs) * Int64(input_groups)
        self.kernel(scale, scale_mma, input_groups, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, scale: cute.Pointer, scale_mma: cute.Pointer,
               input_groups: Int32, count: Int64):
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            row, group = i // Int64(input_groups), i % Int64(input_groups)
            offset = ((row // Int64(128)) * (Int64(input_groups) // Int64(4))
                      + group // Int64(4)) * Int64(512)
            offset += (row % Int64(32)) * Int64(16) + (row % Int64(128) // Int64(32)) * Int64(4)
            offset += group % Int64(4)
            scale_mma[offset] = scale[i]
