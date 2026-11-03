"""V4.1 BF16 router projection with FP32 logits.

Uses the TMA/warp-MMA pipeline from sequence.mtp_feedback._cute_prefill,
with runtime descriptor bounds and without BF16 output rounding.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from b12x._lib.utils import make_ptr
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.sm80_pipeline import Sm80CpAsyncPipeline, target_is_pre_hopper
from b12x._lib.utils import current_cuda_stream


_TILE_N = 16
_TILE_K = 64
_STAGES = 3
_BUFFER_ALIGN_BYTES = 1_024
def _convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    column_major = cute.make_layout(acc_layout.shape)
    shape = (
        (column_major.shape[0][1], column_major.shape[1]),
        (
            column_major.shape[0][0],
            *column_major.shape[0][2:],
            column_major.shape[2],
        ),
        *column_major.shape[3:],
    )
    stride = (
        (column_major.stride[0][1], column_major.stride[1]),
        (
            column_major.stride[0][0],
            *column_major.stride[0][2:],
            column_major.stride[2],
        ),
        *column_major.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_mn(acc.layout))


@cute.jit
def _warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    register_a: cute.Tensor,
    register_b: cute.Tensor,
    shared_a: cute.Tensor,
    shared_b: cute.Tensor,
    copy_a: cute.TiledCopy,
    copy_b: cute.TiledCopy,
):
    register_a_copy = copy_a.retile(register_a)
    register_b_copy = copy_b.retile(register_b)
    cute.copy(copy_a, shared_a[None, None, 0], register_a_copy[None, None, 0])
    cute.copy(copy_b, shared_b[None, None, 0], register_b_copy[None, None, 0])
    for k_step in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
        if k_step < cute.size(shared_a.shape[2]) - 1:
            cute.copy(
                copy_a,
                shared_a[None, None, k_step + 1],
                register_a_copy[None, None, k_step + 1],
            )
            cute.copy(
                copy_b,
                shared_b[None, None, k_step + 1],
                register_b_copy[None, None, k_step + 1],
            )
        cute.gemm(
            tiled_mma,
            accumulator,
            register_a[None, None, k_step],
            register_b[None, None, k_step],
            accumulator,
        )


class _V41RouterScores:
    """TMA-fed BF16 gate projection, with runtime rows and FP32 output."""

    def __init__(self, experts: int, tile_m: int = 64, hidden: int = 5120,
                 tile_n: int = _TILE_N, stages: int = _STAGES,
                 warps_m: int | None = None, warps_n: int = 1):
        assert hidden % _TILE_K == 0 and experts % tile_n == 0
        self.tile_n = tile_n
        self.stages = stages
        # MMA warp grid; by default one 16-row warp per M atom.
        self.warps_m = tile_m // 16 if warps_m is None else warps_m
        self.warps_n = warps_n
        assert tile_m % (16 * self.warps_m) == 0 and tile_n % (8 * warps_n) == 0
        self.output_columns = experts
        self.reduction_width = hidden
        self.tile_m = tile_m
        self.reduction_tiles = self.reduction_width // _TILE_K
        self.compute_warps = self.warps_m * self.warps_n
        self.producer_warp = self.compute_warps
        self.threads = (self.compute_warps + 1) * 32

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.warps_m, self.warps_n, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _shared_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                _TILE_K,
            ),
            cutlass.BFloat16,
        )
        layout_a = cute.tile_to_shape(
            layout_atom,
            (self.tile_m, _TILE_K, self.stages),
            order=(0, 1, 2),
        )
        layout_b = cute.tile_to_shape(
            layout_atom,
            (self.tile_n, _TILE_K, self.stages),
            order=(0, 1, 2),
        )
        return layout_a, layout_b

    def _shared_storage(
        self,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "barriers": cute.struct.MemRange[cutlass.Int64, self.stages * 2],
            "a": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(layout_a)],
                _BUFFER_ALIGN_BYTES,
            ],
            "b": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(layout_b)],
                _BUFFER_ALIGN_BYTES,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        logits: cute.Pointer,
        live_rows: Int32,
        stream: cuda.CUstream,
    ):
        # The descriptor bounds match the actual allocation, including short tails.
        inputs = cute.make_tensor(x, cute.make_layout(
            (live_rows, self.reduction_width), stride=(self.reduction_width, 1)))
        weight = cute.make_tensor(w, cute.make_layout(
            (self.output_columns, self.reduction_width), stride=(self.reduction_width, 1)))
        output = cute.make_tensor(logits, cute.make_layout(
            live_rows.to(Int64) * Int64(self.output_columns)))
        layout_a, layout_b = self._shared_layouts()
        tiled_mma = self._tiled_mma()
        SharedStorage = self._shared_storage(layout_a, layout_b)
        tma_layout_a = cute.slice_(layout_a, (None, None, 0))
        tma_layout_b = cute.slice_(layout_b, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            inputs,
            tma_layout_a,
            (self.tile_m, _TILE_K),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            weight,
            tma_layout_b,
            (self.tile_n, _TILE_K),
            num_multicast=1,
        )
        self.kernel(
            tma_tensor_a,
            tma_tensor_b,
            output,
            live_rows,
            tma_atom_a,
            tma_atom_b,
            layout_a,
            layout_b,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=(
                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),
                self.output_columns // self.tile_n,
                1,
            ),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        inputs: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        live_rows: Int32,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_index == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        allocator = cutlass_utils.SmemAllocator()
        storage = allocator.allocate(SharedStorage)
        shared_a = storage.a.get_tensor(layout_a.outer, swizzle=layout_a.inner)
        shared_b = storage.b.get_tensor(layout_b.outer, swizzle=layout_b.inner)
        copy_bytes = (
            (self.tile_m + self.tile_n) * _TILE_K * cutlass.BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.compute_warps,
            ),
            tx_count=copy_bytes,
            barrier_storage=storage.barriers.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        global_a = cute.local_tile(
            inputs,
            (self.tile_m, _TILE_K),
            (None, None),
        )
        global_b = cute.local_tile(
            weight,
            (self.tile_n, _TILE_K),
            (None, None),
        )
        cta_layout = cute.make_layout(1)
        partition_shared_a, partition_global_a = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(shared_a, 0, 2),
            cute.group_modes(global_a, 0, 2),
        )
        partition_shared_b, partition_global_b = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(shared_b, 0, 2),
            cute.group_modes(global_b, 0, 2),
        )

        if warp_index < Int32(self.compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.stages,
            )
            thread_mma = tiled_mma.get_slice(tidx)
            thread_shared_a = thread_mma.partition_A(shared_a)
            thread_shared_b = thread_mma.partition_B(shared_b)
            register_a = thread_mma.make_fragment_A(
                thread_shared_a[None, None, None, 0]
            )
            register_b = thread_mma.make_fragment_B(
                thread_shared_b[None, None, None, 0]
            )
            accumulator_shape = thread_mma.partition_shape_C((self.tile_m, self.tile_n))
            accumulator = cute.make_rmem_tensor(accumulator_shape, Float32)
            total_accumulator = cute.make_rmem_tensor(accumulator_shape, Float32)
            total_accumulator.fill(0.0)
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma).get_slice(tidx)
            copy_source_a = copy_a.partition_S(shared_a)
            copy_source_b = copy_b.partition_S(shared_b)

            for _ in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.consumer_wait(consumer_state)
                # Bound the MMA accumulation chain to K64. BF16 products are
                # accumulated into FP32 partials before adding to the full dot.
                accumulator.fill(0.0)
                _warp_mma_gemm(
                    tiled_mma,
                    accumulator,
                    register_a,
                    register_b,
                    copy_source_a[None, None, None, consumer_state.index],
                    copy_source_b[None, None, None, consumer_state.index],
                    copy_a,
                    copy_b,
                )
                total_accumulator.store(total_accumulator.load() + accumulator.load())
                cute.arch.fence_view_async_shared()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            accumulator_mn = _reshape_acc_to_mn(total_accumulator)
            coordinate_mn = _reshape_acc_to_mn(
                thread_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for accumulator_m in cutlass.range_constexpr(
                cute.size(accumulator_mn.shape[0])
            ):
                for accumulator_n in cutlass.range_constexpr(
                    cute.size(accumulator_mn.shape[1])
                ):
                    coordinate = coordinate_mn[accumulator_m, accumulator_n]
                    row = m_tile * Int32(self.tile_m) + coordinate[0]
                    column = n_tile * Int32(self.tile_n) + coordinate[1]
                    value = accumulator_mn[accumulator_m, accumulator_n]
                    if row < live_rows:
                        output_offset = (
                            row.to(Int64) * Int64(self.output_columns)
                            + column.to(Int64)
                        )
                        output[output_offset] = Float32(value)

        elif warp_index == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.stages,
            )
            for reduction_tile in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    partition_global_a[(None, m_tile, reduction_tile)],
                    partition_shared_a[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    partition_global_b[(None, n_tile, reduction_tile)],
                    partition_shared_b[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class _V41RouterScoresCpAsync(_V41RouterScores):
    """Pre-Hopper (SM80/86/89) form of ``_V41RouterScores``.

    SM89 has no TMA, mbarrier expect-tx or async-proxy fences: the producer
    warp issues 16-byte cp.async copies (rows past ``live_rows`` skipped) and
    commits each stage with ``cp.async.mbarrier.arrive.noinc`` on the
    inline-PTX ring. The consumer warps, shared layouts and K64-bounded
    partial order are the TMA kernel's, so every output keeps its reduction
    order whatever the tile.
    """

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        logits: cute.Pointer,
        live_rows: Int32,
        stream: cuda.CUstream,
    ):
        inputs = cute.make_tensor(x, cute.make_layout(
            (live_rows, self.reduction_width), stride=(self.reduction_width, 1)))
        weight = cute.make_tensor(w, cute.make_layout(
            (self.output_columns, self.reduction_width), stride=(self.reduction_width, 1)))
        output = cute.make_tensor(logits, cute.make_layout(
            live_rows.to(Int64) * Int64(self.output_columns)))
        layout_a, layout_b = self._shared_layouts()
        tiled_mma = self._tiled_mma()
        SharedStorage = self._shared_storage(layout_a, layout_b)
        self.kernel(
            inputs,
            weight,
            output,
            live_rows,
            layout_a,
            layout_b,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=(
                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),
                self.output_columns // self.tile_n,
                1,
            ),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        inputs: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        live_rows: Int32,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        allocator = cutlass_utils.SmemAllocator()
        storage = allocator.allocate(SharedStorage)
        shared_a = storage.a.get_tensor(layout_a.outer, swizzle=layout_a.inner)
        shared_b = storage.b.get_tensor(layout_b.outer, swizzle=layout_b.inner)
        load_pipeline = Sm80CpAsyncPipeline.create(
            num_stages=self.stages,
            producer_threads=32,
            consumer_warps=self.compute_warps,
            barrier_storage=storage.barriers.data_ptr(),
            tidx=tidx,
        )

        if warp_index < Int32(self.compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.stages,
            )
            thread_mma = tiled_mma.get_slice(tidx)
            thread_shared_a = thread_mma.partition_A(shared_a)
            thread_shared_b = thread_mma.partition_B(shared_b)
            register_a = thread_mma.make_fragment_A(
                thread_shared_a[None, None, None, 0]
            )
            register_b = thread_mma.make_fragment_B(
                thread_shared_b[None, None, None, 0]
            )
            accumulator_shape = thread_mma.partition_shape_C((self.tile_m, self.tile_n))
            accumulator = cute.make_rmem_tensor(accumulator_shape, Float32)
            total_accumulator = cute.make_rmem_tensor(accumulator_shape, Float32)
            total_accumulator.fill(0.0)
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma).get_slice(tidx)
            copy_source_a = copy_a.partition_S(shared_a)
            copy_source_b = copy_b.partition_S(shared_b)

            for _ in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.consumer_wait(consumer_state)
                accumulator.fill(0.0)
                _warp_mma_gemm(
                    tiled_mma,
                    accumulator,
                    register_a,
                    register_b,
                    copy_source_a[None, None, None, consumer_state.index],
                    copy_source_b[None, None, None, consumer_state.index],
                    copy_a,
                    copy_b,
                )
                total_accumulator.store(total_accumulator.load() + accumulator.load())
                # cp.async writes and ldmatrix reads are both generic-proxy:
                # the release arrive alone orders this stage's reads.
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            accumulator_mn = _reshape_acc_to_mn(total_accumulator)
            coordinate_mn = _reshape_acc_to_mn(
                thread_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for accumulator_m in cutlass.range_constexpr(
                cute.size(accumulator_mn.shape[0])
            ):
                for accumulator_n in cutlass.range_constexpr(
                    cute.size(accumulator_mn.shape[1])
                ):
                    coordinate = coordinate_mn[accumulator_m, accumulator_n]
                    row = m_tile * Int32(self.tile_m) + coordinate[0]
                    column = n_tile * Int32(self.tile_n) + coordinate[1]
                    value = accumulator_mn[accumulator_m, accumulator_n]
                    if row < live_rows:
                        output_offset = (
                            row.to(Int64) * Int64(self.output_columns)
                            + column.to(Int64)
                        )
                        output[output_offset] = Float32(value)

        elif warp_index == Int32(self.producer_warp):
            # Eight lanes cover one 128-byte K row; four rows per instruction.
            tiled_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                ),
                cute.make_ordered_layout((4, _TILE_K // 8), order=(1, 0)),
                cute.make_layout((1, 8)),
            )
            thread_copy = tiled_copy.get_slice(tidx - Int32(self.producer_warp * 32))
            source_a = thread_copy.partition_S(
                cute.local_tile(inputs, (self.tile_m, _TILE_K), (None, None))
            )
            rows_a = thread_copy.partition_S(
                cute.local_tile(
                    cute.make_identity_tensor((live_rows, self.reduction_width)),
                    (self.tile_m, _TILE_K),
                    (None, None),
                )
            )
            target_a = thread_copy.partition_D(shared_a)
            source_b = thread_copy.partition_S(
                cute.local_tile(weight, (self.tile_n, _TILE_K), (None, None))
            )
            target_b = thread_copy.partition_D(shared_b)
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.stages,
            )
            for reduction_tile in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.producer_acquire(producer_state)
                stage = producer_state.index
                for copy_m in cutlass.range_constexpr(cute.size(target_a.shape[1])):
                    if rows_a[0, copy_m, 0, m_tile, reduction_tile][0] < live_rows:
                        cute.copy(
                            tiled_copy,
                            source_a[None, copy_m, None, m_tile, reduction_tile],
                            target_a[None, copy_m, None, stage],
                        )
                cute.copy(
                    tiled_copy,
                    source_b[None, None, None, n_tile, reduction_tile],
                    target_b[None, None, None, stage],
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


# Pre-Hopper MMA tiles by live rows, per gate shape [experts, hidden]:
# (rows, warps_m, warps_n, tile_m, tile_n, stages) in ascending rows, the last
# entry serving every wider launch. Each entry retiles one per-output order
# (K64 MMA partials added in K order), so every entry gives the first entry's
# bits. RTX 4090 D, CUDA-graph replay over rotated copies (cold L2), us at
# 65/256/512/1024/2048/3072/4096 rows:
#   384 64x16  25.8  29.0  44.0 103.6 176.7 242.8 312.9
#       64x64  33.7  32.6  34.7  40.9  70.9 137.8 142.9  (2x2 warps)
#       64x128 46.4  46.4  46.5  43.6  55.6 110.4 122.4  (2x2 warps)
#      128x128 77.6  70.9  70.9  70.7  72.3  79.3  90.5  (4x2 warps)
#   128 64x16  24.6  25.1  26.7  32.4  56.3 111.7 120.5
#       64x32  28.5  27.6  28.1  27.9  35.1  49.5  86.4
#       64x64  32.3  33.4  32.4  33.5  37.3  45.8  61.4  (2x2 warps)
_PRE_HOPPER_TILES = {
    (384, 5120): ((256, 4, 1, 64, 16, 3), (1024, 2, 2, 64, 64, 3),
                  (2048, 2, 2, 64, 128, 3), (4096, 4, 2, 128, 128, 3)),
    (128, 5120): ((512, 4, 1, 64, 16, 3), (2048, 4, 1, 64, 32, 3),
                  (4096, 2, 2, 64, 64, 3)),
}


class _RowTiledRouterScores:
    """Pre-Hopper MMA route with its tile picked by the ``live_rows`` launch
    scalar (no host sync; one program serves every width)."""

    def __init__(self, experts: int, hidden: int, tiles_by_rows):
        self.tiles_by_rows = tuple(tuple(int(v) for v in entry) for entry in tiles_by_rows)
        self.floors = (0,) + tuple(entry[0] for entry in self.tiles_by_rows[:-1])
        self.tiled = [
            _V41RouterScoresCpAsync(experts, tile_m=tile_m, hidden=hidden, tile_n=tile_n,
                                    stages=stages, warps_m=warps_m, warps_n=warps_n)
            for _, warps_m, warps_n, tile_m, tile_n, stages in self.tiles_by_rows
        ]

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, logits: cute.Pointer,
                 live_rows: Int32, stream: cuda.CUstream):
        for i in cutlass.range_constexpr(len(self.tiled)):
            if live_rows > Int32(self.floors[i]):
                if cutlass.const_expr(i + 1 == len(self.tiled)):
                    self.tiled[i](x, w, logits, live_rows, stream)
                else:
                    if live_rows <= Int32(self.tiles_by_rows[i][0]):
                        self.tiled[i](x, w, logits, live_rows, stream)


def _pre_hopper_tiles(experts: int, hidden: int):
    return _PRE_HOPPER_TILES.get((experts, hidden), ((4096, 4, 1, 64, _TILE_N, _STAGES),))


def _router_mma(experts: int, hidden: int):
    """MMA route: TMA-fed on SM90+, cp.async-fed and row-tiled before SM90."""
    if not target_is_pre_hopper():
        return _V41RouterScores(experts, hidden=hidden)
    return _RowTiledRouterScores(experts, hidden, _pre_hopper_tiles(experts, hidden))


def v41_router_mma_tiles(experts: int, hidden: int = 5120) -> tuple:
    """MMA tiles for the current target as (rows, warps_m, warps_n, tile_m,
    tile_n, stages) by live rows; one entry on SM90+."""
    if not target_is_pre_hopper():
        return ((4096, 4, 1, 64, _TILE_N, _STAGES),)
    return tuple(tuple(entry) for entry in _pre_hopper_tiles(experts, hidden))


# One canonical reduction order for every V4.1 decode/verify width (#736).
# Keep the kernel choice here so a measured row-invariant winner can replace it.
V41_ROUTER_SKINNY_MAX_ROWS = 64


from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, router_skinny


class _V41ActivatedSkinnyScores(SkinnyBf16Gemv):
    @cute.jit
    def _epilogue(self, value: Float32):
        softplus = value
        if value <= Float32(20.0):
            softplus = cute.math.log1p(cute.math.exp(value, fastmath=False), fastmath=False)
        return cute.math.sqrt(softplus, fastmath=False)


class _V41DecodeRouterScores:
    def __init__(self, experts: int, hidden: int, activated: bool = False):

        skinny = _V41ActivatedSkinnyScores if activated else SkinnyBf16Gemv
        self.skinny = router_skinny(experts, hidden, kernel_type=skinny)
        self.mma = _router_mma(experts, hidden)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, logits: cute.Pointer,
                 live_rows: Int32, stream: cuda.CUstream):
        if live_rows <= Int32(V41_ROUTER_SKINNY_MAX_ROWS):
            self.skinny(x, w, logits, live_rows, stream)
        else:
            self.mma(x, w, logits, live_rows, stream)


def compile_v41_router_scores_aot(*, experts: int, hidden: int = 5120, activated: bool = False):
    """Compile BF16 [rows,hidden] @ gate.T -> FP32 [rows,experts].

    Supports the V4.1 384-expert backbone and 128-expert dSpark gates (hidden
    5120) and the DeepSeek V4 gates (Flash: 256 x 4096, Pro: 384 x 7168).
    With activated=True, only the skinny path writes sqrtsoftplus scores.
    Caller owns all buffers; no scratch, packing, allocation, or live-row cache
    keys. Score transform, bias and top-k selection are separate operations.
    """
    if experts not in (128, 256, 384) or hidden not in (4096, 5120, 7168):
        raise ValueError("router scores support 128/256/384 experts over hidden 4096/5120/7168")
    if hidden == 5120 and experts in (128, 384):
        kernel = _V41DecodeRouterScores(experts, hidden, activated)
        s = kernel.skinny
        route_key = (V41_ROUTER_SKINNY_MAX_ROWS, s.tiles_by_rows, activated)
    else:
        kernel = _router_mma(experts, hidden)
        route_key = ()
    key = (experts, hidden, 64, route_key, torch.cuda.current_device())
    mma = kernel.mma if isinstance(kernel, _V41DecodeRouterScores) else kernel
    if isinstance(mma, _RowTiledRouterScores):
        key += (("cp_async", mma.tiles_by_rows),)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    return b12x_compile(
        kernel,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("moe.v41_router_scores", 1, key),
    )
