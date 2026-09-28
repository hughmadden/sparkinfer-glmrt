"""Block-FP8 linear stage composed into cuteafd AOT programs.

``Fp8LinearStage`` reproduces one prepared ``gemm.block_fp8_linear`` call
(DeepSeek checkpoint 128x128 UE8M0 weights, per-token K128 activation
quantization) inside a larger ``@cute.jit`` program:

1. the same CuTe MXFP8 row quantizer the prepared plan compiles
   (``_MXFP8RowsQuantLaunch``, K128 blocks, subgroup/threads by capacity);
2. the dense MXFP8 GEMM launch of the plan's default lowering for the
   declared capacity (``dense_gemm_launch_from_lowering``);
3. for split-K lowerings, FP32 partial planes and one reduction (BF16-atomic
   split lowerings are converted to partial planes: deterministic, one BF16
   rounding instead of the prepared path's order-dependent atomics).

Weights are the packed ``block_fp8_linear.pack_weight`` operands: ``values``
FP8 E4M3 ``[N, K]`` and ``scale_mma``, the UE8M0 scales in dense-GEMM MMA tile
order (``ceil(N/128) * ceil(K/128) * 512`` bytes; pass the storage pointer of
``weight.scale_mma``).

Activation scratch for ``rows`` live rows (all offsets 1024-byte aligned,
computed from the live row count at launch):

    values      rows * K                         FP8 E4M3
    scale_rows  rows * K / 32                     UE8M0 (quantizer byproduct)
    scale_mma   ceil(rows/128) * ceil(K/128)*512  UE8M0 MMA layout
    alpha       16 bytes                          FP32 1.0 (written when K is split)
    split-K     slices * rows * N * 4             FP32 partials (slices > 1)

``stage_scratch_bytes(K, N, rows, slices)`` is the exact total; it is
monotone in ``rows`` so sizing for the capacity covers every live count.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from b12x._lib.dense_gemm import _DenseSplitKReduce, dense_gemm_launch_from_lowering
from b12x._lib.quant.mxfp8_rows import _MXFP8RowsQuantLaunch, _GRID_CTAS_PER_SM

__all__ = ["Fp8LinearStage", "StoreOne", "stage_scratch_bytes", "block_fp8_lowering"]

_ALIGN = 1024


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def stage_scratch_bytes(k: int, n: int, rows: int, slices: int = 1) -> int:
    """Bytes of one Fp8LinearStage scratch region for ``rows`` rows."""
    rows = max(int(rows), 1)
    values = _align(rows * k)
    scale_rows = _align(rows * (k // 32))
    scale_mma = _align(((rows + 127) // 128) * ((k + 127) // 128) * 512)
    alpha = _align(16)
    split = rows * n * 4 * slices if slices > 1 else 0
    return values + scale_rows + scale_mma + alpha + split


def block_fp8_lowering(*, max_rows: int, in_features: int, out_features: int, device=None):
    """Default lowering the prepared block_fp8_linear plan selects (K128 activations)."""
    from b12x.gemm.block_fp8_linear._preparation import _dense_lowering
    from b12x.gemm.block_fp8_linear._tuning import TUNING, BlockFp8LinearQuery
    from b12x.preparation import detect_device

    identity = detect_device(device if device is not None else torch.device(
        "cuda", torch.cuda.current_device())).identity
    query = BlockFp8LinearQuery(
        max_tokens=int(max_rows), in_features=int(in_features), out_features=int(out_features),
        source_dtype="bfloat16", output_dtype="bfloat16", output_mode="provided",
        weight_block_size=128, activation_block_size=128,
    )
    configuration = TUNING.configure(query, device=identity)
    config = configuration.default if configuration.pinned is None else configuration.pinned
    return _dense_lowering(query, config, identity)


class StoreOne:
    """Write FP32 1.0 to a dense GEMM alpha slot.

    The unit-alpha dense GEMM still multiplies its FP32 split-K partial planes
    by ``alpha[0]``, so the slot must hold 1.0 when a lowering splits K
    (scratch is otherwise uninitialized).
    """

    @cute.jit
    def __call__(self, alpha: cute.Pointer, stream: cuda.CUstream):
        self.kernel(alpha).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, alpha: cute.Pointer):
        if cute.arch.thread_idx()[0] == 0:
            alpha[0] = cutlass.Float32(1.0)


@cute.jit
def _align_i64(value: Int64) -> Int64:
    return (value + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)


class Fp8LinearStage:
    """Quantize ``source [rows, K]`` BF16 and write ``out [rows, N]`` BF16."""

    def __init__(self, *, in_features: int, out_features: int, max_rows: int, device=None):
        self.k = int(in_features)
        self.n = int(out_features)
        if self.k % 128:
            raise ValueError("block-FP8 stage requires K divisible by 128")
        self.max_rows = int(max_rows)
        self.lowering = block_fp8_lowering(max_rows=max_rows, in_features=self.k,
                                           out_features=self.n, device=device)
        # BF16-atomic split lowerings become FP32 partial planes + one
        # deterministic reduction (single BF16 rounding).
        self.gemm, self.slices = dense_gemm_launch_from_lowering(self.lowering, atomic_split="partials")
        self.sm_count = int(self.lowering.sm_count)
        subgroup_width, threads = (8, 128) if self.max_rows <= 8 else (4, 256)
        self.quant = _MXFP8RowsQuantLaunch(
            self.k, cutlass.BFloat16, subgroup_width, threads, 128, False, 0.0, False,
        )
        self.quant_warps = threads // 32
        self.quant_tasks_per_row = (self.k // 32 + (32 // subgroup_width) - 1) // (32 // subgroup_width)
        self.quant_grid_cap = self.sm_count * _GRID_CTAS_PER_SM
        self.reduce = _DenseSplitKReduce(self.n, self.slices) if self.slices > 1 else None
        self.store_one = StoreOne()

    def scratch_bytes(self, rows: int) -> int:
        return stage_scratch_bytes(self.k, self.n, rows, self.slices)

    def key(self) -> tuple:
        low = self.lowering
        return (self.k, self.n, low.mma_tiler_mn, low.tile_k, low.policy, low.sm_count,
                low.load_path, low.sfb_k_reuse, low.target_occupancy_override, self.max_rows <= 8)

    @cute.jit
    def __call__(self, source: cute.Pointer, w_values: cute.Pointer, w_scales: cute.Pointer,
                 out: cute.Pointer, scratch: Int64, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        values_off = Int64(0)
        scale_rows_off = values_off + _align_i64(m * Int64(self.k))
        scale_mma_off = scale_rows_off + _align_i64(m * Int64(self.k // 32))
        alpha_off = scale_mma_off + _align_i64(
            (m + Int64(127)) // Int64(128) * Int64(((self.k + 127) // 128) * 512))
        split_off = alpha_off + Int64(_ALIGN)

        grid = (rows * Int32(self.quant_tasks_per_row) + Int32(self.quant_warps - 1)) // Int32(self.quant_warps)
        if grid > Int32(self.quant_grid_cap):
            grid = Int32(self.quant_grid_cap)
        self.quant(
            source,
            cute.make_ptr(cutlass.Uint32, scratch + values_off, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, scratch + scale_rows_off, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_ptr(cutlass.Uint8, scratch + scale_mma_off, cute.AddressSpace.gmem, assumed_align=16),
            rows, Int32(self.k), grid, stream,
        )
        a = cute.make_ptr(cutlass.Float8E4M3FN, scratch + values_off, cute.AddressSpace.gmem, assumed_align=16)
        sfa = cute.make_ptr(cutlass.Float8E8M0FNU, scratch + scale_mma_off, cute.AddressSpace.gmem, assumed_align=16)
        b = cute.make_ptr(cutlass.Float8E4M3FN, Int64(w_values.toint()), cute.AddressSpace.gmem, assumed_align=16)
        sfb = cute.make_ptr(cutlass.Float8E8M0FNU, Int64(w_scales.toint()), cute.AddressSpace.gmem, assumed_align=16)
        alpha = cute.make_ptr(cutlass.Float32, scratch + alpha_off, cute.AddressSpace.gmem, assumed_align=16)
        qc_values = cute.make_ptr(cutlass.Float8E4M3FN, scratch + values_off, cute.AddressSpace.gmem, assumed_align=16)
        qc_rows = cute.make_ptr(cutlass.Float8E8M0FNU, scratch + scale_rows_off, cute.AddressSpace.gmem, assumed_align=16)
        if const_expr(self.slices > 1):
            self.store_one(alpha, stream)
            partials = cute.make_ptr(cutlass.Float32, scratch + split_off, cute.AddressSpace.gmem, assumed_align=16)
            self.gemm(a, b, sfa, sfb, partials, qc_values, qc_rows, sfa, alpha, rows, stream)
            self.reduce(partials, out, rows, stream)
        else:
            self.gemm(a, b, sfa, sfb, out, qc_values, qc_rows, sfa, alpha, rows, stream)
