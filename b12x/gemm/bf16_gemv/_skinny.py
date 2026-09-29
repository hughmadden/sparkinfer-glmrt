"""Bandwidth-bound BF16 projection for a few live rows (decode GEMV).

``out[m, n] = sum_k x[m, k] * w[n, k]``, FP32 accumulation, BF16 or FP32
output, runtime ``rows``. A CTA owns ``cols`` output columns and a tile of
``rows_per_tile`` rows (grid ``(N/cols, ceil(rows/rows_per_tile))``); its
threads split K into 16-byte vectors (thread ``t`` takes vectors
``t, t+threads, ...``). Each weight vector is loaded once (L1 no-allocate)
and converted to FP32 once and reused from registers by every row of the tile; activation vectors are
reused from registers by every column. Per output, each thread accumulates
its vectors in K order with FMA, then a 32-lane butterfly and an in-order
sum over warps: within the rounding of one FP32-accumulated dot product.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["RoutedBf16Projection", "SkinnyBf16Gemv", "TmaBf16Projection", "prefill_config", "skinny_config", "skinny_max_rows"]


@dsl_user_op
def _ld_v4(address: Int64, asm: str, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)], asm, "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


def _ld_stream(address):
    return _ld_v4(address, "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];")


def _ld_cached(address):
    return _ld_v4(address, "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];")


@dsl_user_op
def _bf16_lo(word, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(word).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 16;", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _bf16_hi(word, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(word).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


class SkinnyBf16Gemv:
    """Pointer ABI ``(x, w, out, rows, stream)``; x [rows,K], w [N,K], out [rows,N] (dense)."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, cols: int = 4,
                 rows_per_tile: int = 8, threads: int = 256):
        self.n, self.k = int(n), int(k)
        self.cols, self.rows_per_tile, self.threads = int(cols), int(rows_per_tile), int(threads)
        self.vectors = self.k // 8
        if self.k % 8 or self.vectors % self.threads:
            raise ValueError("K/8 must be a multiple of the CTA thread count")
        if self.n % self.cols:
            raise ValueError("N must be a multiple of cols")
        self.steps = self.vectors // self.threads
        self.warps = self.threads // 32
        self.out_dtype = out_dtype
        self.outputs = self.rows_per_tile * self.cols
        if self.outputs > self.threads:
            raise ValueError("rows_per_tile * cols must not exceed the CTA thread count")

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.outputs], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        tiles = (rows + Int32(self.rows_per_tile - 1)) // Int32(self.rows_per_tile)
        self.kernel(x, w, out, rows).launch(
            grid=(self.n // self.cols, tiles, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        col0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        row0 = Int32(cute.arch.block_idx()[1]) * Int32(self.rows_per_tile)
        live = rows - row0
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(
            cute.make_layout((self.warps, self.outputs), stride=(self.outputs, 1)))
        acc = cute.make_rmem_tensor(cute.make_layout((self.outputs,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.outputs):
            acc[i] = Float32(0.0)
        w_base = Int64(w.toint()) + col0 * Int64(self.k * 2) + Int64(tidx) * Int64(16)
        x_base = Int64(x.toint()) + Int64(row0) * Int64(self.k * 2) + Int64(tidx) * Int64(16)
        for step in cutlass.range_constexpr(self.steps):
            offset = step * self.threads * 16
            wv = cute.make_rmem_tensor(cute.make_layout((self.cols * 8,), stride=(1,)), Float32)
            for c in cutlass.range_constexpr(self.cols):
                words = _ld_stream(w_base + Int64(c * self.k * 2 + offset))
                for j in cutlass.range_constexpr(4):
                    wv[8 * c + 2 * j] = _bf16_lo(words[j])
                    wv[8 * c + 2 * j + 1] = _bf16_hi(words[j])
            for m in cutlass.range_constexpr(self.rows_per_tile):
                if Int32(m) < live:
                    self._accumulate(acc, wv, x_base + Int64(m * self.k * 2 + offset), m)
        for i in cutlass.range_constexpr(self.outputs):
            value = acc[i]
            for shift in cutlass.range_constexpr(5):
                value = value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
            acc[i] = value
        if lane == Int32(0):
            for i in cutlass.range_constexpr(self.outputs):
                partial[warp, i] = acc[i]
        cute.arch.sync_threads()
        if tidx < Int32(self.outputs):
            self._store(partial, out, col0, row0, live, tidx)

    @cute.jit
    def _accumulate(self, acc: cute.Tensor, wv: cute.Tensor, x_address: Int64,
                    m: cutlass.Constexpr):
        words = _ld_cached(x_address)
        xs = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        for j in cutlass.range_constexpr(4):
            xs[2 * j] = _bf16_lo(words[j])
            xs[2 * j + 1] = _bf16_hi(words[j])
        for c in cutlass.range_constexpr(self.cols):
            a = acc[m * self.cols + c]
            for j in cutlass.range_constexpr(8):
                a = a + wv[8 * c + j] * xs[j]
            acc[m * self.cols + c] = a

    @cute.jit
    def _store(self, partial: cute.Tensor, out: cute.Pointer, col0: Int64, row0: Int32,
               live: Int32, index: Int32):
        out_bytes = 2 if const_expr(self.out_dtype == cutlass.BFloat16) else 4
        m = index // Int32(self.cols)
        c = index % Int32(self.cols)
        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = total + partial[src, index]
        address = Int64(out.toint()) + ((Int64(row0) + Int64(m)) * Int64(self.n) + col0 + Int64(c)) * Int64(out_bytes)
        target = cute.make_ptr(self.out_dtype, address, cute.AddressSpace.gmem, assumed_align=out_bytes)
        converted = total.to(self.out_dtype)
        if m < live:
            target[0] = converted


def skinny_config(n: int, k: int, out_dtype) -> dict:
    """Measured SM120 choices (L2-cold weights): cols, rows_per_tile, threads."""
    if (n, k) == (256, 4096):
        return dict(cols=4, rows_per_tile=2, threads=256)
    if (n, k) == (384, 7168):
        return dict(cols=4, rows_per_tile=2, threads=224)
    if (n, k) == (64, 4096):
        return dict(cols=1, rows_per_tile=2, threads=256)
    if (n, k) == (64, 7168):
        return dict(cols=2, rows_per_tile=2, threads=448)
    vectors = int(k) // 8
    if vectors % 128:
        # K/8 must split evenly over the CTA (GLM 5.3 Flash q_lora 1536: 192).
        for threads in (192, 96, 64, 32):
            if vectors % threads == 0:
                return dict(cols=1, rows_per_tile=8, threads=threads)
    return dict(cols=1, rows_per_tile=8, threads=128)


def skinny_max_rows(n: int, k: int) -> int:
    """Largest live row count routed to the skinny GEMV (measured crossover
    against the TMA tensor-core route on SM120, L2-cold weights)."""
    if n >= 2048:
        return 24
    if n >= 512:
        return 64
    return 160


def prefill_config(n: int, k: int) -> dict:
    """Measured SM120 choices for the TMA tensor-core route (uncompensated)."""
    if n >= 1024:
        return dict(compute_warps=4, tile_n=128, num_stages=3, compensated=False)
    return dict(compute_warps=4, tile_n=64, num_stages=4, compensated=False)


class TmaBf16Projection:
    """Pointer ABI over ``Bf16PrefillKernel`` (TMA pipeline, FP32 MMA accumulation)."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, **config):
        from ._prefill import Bf16PrefillKernel

        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.config = dict(prefill_config(n, k), **config)
        self.kernel = Bf16PrefillKernel(self.n, self.k, **self.config)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        xt = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        wt = cute.make_tensor(w, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        ot = cute.make_tensor(out, cute.make_layout((rows, self.n), stride=(self.n, 1)))
        self.kernel(xt, wt, ot, rows, stream)


class RoutedBf16Projection:
    """``out = x @ w^T``: skinny GEMV for few live rows, TMA tensor-core GEMM above.

    The branch is on the ``rows`` launch scalar inside the program (no host
    sync), so one compiled program serves decode and prefill and stays
    CUDA-graph capturable. ``x``/``w``/``out`` are dense and 16-byte aligned;
    ``out`` is ``[rows, N]`` BF16 or FP32. Both routes accumulate in FP32.
    """

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, max_skinny_rows: int | None = None):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.max_skinny_rows = int(skinny_max_rows(n, k) if max_skinny_rows is None else max_skinny_rows)
        self.skinny = SkinnyBf16Gemv(self.n, self.k, out_dtype=out_dtype, **skinny_config(n, k, out_dtype))
        self.large = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype)

    def key(self) -> tuple:
        s = self.skinny
        return (self.n, self.k, str(self.out_dtype), self.max_skinny_rows, s.cols, s.rows_per_tile,
                s.threads, tuple(sorted(self.large.config.items())))

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        if rows <= Int32(self.max_skinny_rows):
            self.skinny(x, w, out, rows, stream)
        else:
            self.large(x, w, out, rows, stream)
