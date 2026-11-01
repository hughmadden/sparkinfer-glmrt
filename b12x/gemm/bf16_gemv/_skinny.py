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

__all__ = [
    "COMPENSATED_TILES_BY_ROWS",
    "DECODE_TMA_TILES",
    "RoutedBf16Projection",
    "RowTiledBf16Projection",
    "RowTiledSkinnyBf16Gemv",
    "SKINNY_TILES_BY_ROWS",
    "SkinnyBf16Gemv",
    "TmaBf16Projection",
    "prefill_config",
    "skinny_config",
    "skinny_max_rows",
]


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
        if live >= Int32(self.rows_per_tile):
            self._loop_fixed(acc, w_base, x_base, self.rows_per_tile)
        else:
            handled = Int32(0)
            if cutlass.const_expr(self.rows_per_tile > 4):
                if live == Int32(4):
                    self._loop_fixed(acc, w_base, x_base, 4)
                    handled = Int32(1)
            if cutlass.const_expr(self.rows_per_tile > 2):
                if live == Int32(2):
                    self._loop_fixed(acc, w_base, x_base, 2)
                    handled = Int32(1)
            if cutlass.const_expr(self.rows_per_tile > 1):
                if live == Int32(1):
                    self._loop_fixed(acc, w_base, x_base, 1)
                    handled = Int32(1)
            if handled == Int32(0):
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
    def _loop_fixed(self, acc: cute.Tensor, w_base: Int64, x_base: Int64,
                    live_rows: cutlass.Constexpr):
        for step in cutlass.range_constexpr(self.steps):
            offset = step * self.threads * 16
            wv = cute.make_rmem_tensor(cute.make_layout((self.cols * 8,), stride=(1,)), Float32)
            for c in cutlass.range_constexpr(self.cols):
                words = _ld_stream(w_base + Int64(c * self.k * 2 + offset))
                for j in cutlass.range_constexpr(4):
                    wv[8 * c + 2 * j] = _bf16_lo(words[j])
                    wv[8 * c + 2 * j + 1] = _bf16_hi(words[j])
            for m in cutlass.range_constexpr(live_rows):
                self._accumulate(acc, wv, x_base + Int64(m * self.k * 2 + offset), m)

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
        converted = self._epilogue(total).to(self.out_dtype)
        if m < live:
            target[0] = converted

    @cute.jit
    def _epilogue(self, value: Float32):
        return value


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


# Measured SM120 skinny GEMV tiles by live rows, per weight shape [N, K]:
# (rows, cols, rows_per_tile, threads) in ascending rows. Every entry for a
# shape shares the same CTA thread count, so each thread sums the same K vectors
# in the same order and the warp/CTA fold is bitwise identical across row counts.
SKINNY_TILES_BY_ROWS = {
    (324, 10240): ((4, 1, 4, 128), (16, 4, 8, 128), (160, 4, 16, 128)),
    (320, 10240): ((4, 1, 4, 128), (16, 4, 8, 128), (160, 4, 16, 128)),
    (1296, 2560): ((4, 1, 8, 64), (64, 2, 8, 64)),
    (512, 2560): ((4, 1, 4, 64), (64, 4, 8, 64)),
}


# Router-only tiles preserve each shape's original K partition and fold order.
ROUTER_TILES_BY_ROWS = {
    (384, 5120): ((16, 2, 4, 128), (64, 8, 4, 128)),
    (128, 5120): ((64, 2, 4, 128),),
    (256, 6144): ((16, 2, 4, 128), (64, 8, 4, 128)),
    (256, 4096): ((16, 4, 2, 256), (64, 8, 4, 256)),
    (384, 7168): ((16, 4, 2, 224), (64, 4, 4, 224)),
}


def router_skinny(n: int, k: int, *, kernel_type=SkinnyBf16Gemv):
    tiles = ROUTER_TILES_BY_ROWS.get((n, k))
    if tiles is None:
        cfg = skinny_config(n, k, Float32)
        tiles = ((64, cfg['cols'], cfg['rows_per_tile'], cfg['threads']),)
    return RowTiledSkinnyBf16Gemv(n, k, tiles, out_dtype=Float32, kernel_type=kernel_type)


class RowTiledSkinnyBf16Gemv:
    """``SkinnyBf16Gemv`` with ``(cols, rows_per_tile)`` picked by live rows.

    All entries share one ``threads`` count, so the K reduction order is
    invariant across row counts and matches a single-row launch bit for bit.
    """

    def __init__(self, n: int, k: int, tiles_by_rows, *, out_dtype=cutlass.BFloat16,
                 kernel_type=SkinnyBf16Gemv):
        self.n, self.k = int(n), int(k)
        self.tiles_by_rows = tuple(tuple(int(v) for v in entry) for entry in tiles_by_rows)
        if not self.tiles_by_rows or any(len(entry) != 4 for entry in self.tiles_by_rows):
            raise ValueError("skinny_tiles_by_rows entries are (rows, cols, rows_per_tile, threads)")
        if len({entry[3] for entry in self.tiles_by_rows}) != 1:
            raise ValueError("skinny_tiles_by_rows entries must share one CTA thread count")
        self.limits = tuple(entry[0] for entry in self.tiles_by_rows)
        self.floors = (0,) + self.limits[:-1]
        if any(limit <= floor for floor, limit in zip(self.floors, self.limits)):
            raise ValueError("skinny_tiles_by_rows rows must be positive and ascending")
        self.cols, self.rows_per_tile, self.threads = self.tiles_by_rows[0][1:]
        self.tiled = [kernel_type(self.n, self.k, out_dtype=out_dtype, cols=c, rows_per_tile=r,
                                     threads=t) for _, c, r, t in self.tiles_by_rows]
        self.default = self.tiled[-1]

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        for i in cutlass.range_constexpr(len(self.tiled)):
            if rows > Int32(self.floors[i]):
                if rows <= Int32(self.limits[i]):
                    self.tiled[i](x, w, out, rows, stream)
        if rows > Int32(self.limits[-1]):
            self.default(x, w, out, rows, stream)


# 2 x 2 MMA warps on 128 x 128 tiles, two stages: the same per-element K
# order as prefill_config's, 17-21% faster from 512 rows on SM120 (4096 rows:
# 4096x8192 0.83 -> 0.69 ms, 16384x1536 0.69 -> 0.55, 24576x4096 2.55 -> 2.05),
# slower on narrow outputs below that (4096x8192 at 100 rows 0.056 -> 0.101 ms).
WIDE_PREFILL_CONFIG = dict(warp_layout=(2, 2), tile_m=128, tile_n=128, num_stages=2, compensated=False)
WIDE_PREFILL_MIN_ROWS = 512

# Measured SM120 TMA tensor-core decode tiles by live rows, per weight shape [N, K]:
# (rows, warp_m, warp_n, tile_n, num_stages) in ascending rows. Each entry retiles
# the same uncompensated m16n8k16 MMA K-reduction as prefill_config and produces
# bitwise-identical outputs at every decode/verify width.
DECODE_TMA_TILES = {
    (2560, 6144): ((32, 1, 2, 32, 7), (64, 1, 4, 64, 6)),
    (16480, 2560): ((32, 1, 4, 64, 6),),
    (13952, 2560): ((16, 1, 4, 128, 4), (32, 1, 4, 64, 5)),
    (2560, 640): ((32, 1, 4, 64, 5), (64, 2, 1, 32, 5)),
    (10240, 320): ((16, 1, 4, 64, 5), (32, 1, 4, 128, 5), (64, 4, 1, 64, 5)),
    # Step-5 heads 64/32, 1..256 live rows (L2-flushed cold medians, SM120 RTX
    # PRO 6000, 24 replays x 4 interleaved blocks): 12288x4096 88.8-90.1 ->
    # 67.6-79.1 us (96 MiB of weights), 6144x4096 FP32 out 47.1 -> 34.9-40.3,
    # 4096x384 18.4 -> 8.2-8.5 (3 MiB, launch-bound), 4096x3072 30.7 ->
    # 18.5-22.9. The 4096x6144 q projection measured 1.06x cold (80% of the
    # roofline) and stays on prefill_config.
    (4096, 12288): ((256, 1, 4, 64, 9),),
    (4096, 6144): ((64, 1, 4, 32, 12), (256, 1, 4, 64, 9)),
    (384, 4096): ((256, 1, 2, 16, 8),),
    (3072, 4096): ((64, 1, 4, 32, 8), (256, 1, 4, 32, 12)),
}


# Measured SM120 compensated tiles by live rows, keyed by the weight shape
# ``(n, k)`` exactly as ``TmaBf16Projection(n, k)`` receives it -- N = output
# features, K = input width, the same key ``DECODE_TMA_TILES`` uses, in
# ascending rows: (rows, warp_m, warp_n, tile_m, tile_n, num_stages),
# (rows, warp_m, warp_n, tile_n, num_stages) or
# (rows, compute_warps, tile_n, num_stages). Each entry retiles the K16
# compensated accumulation of ``Bf16PrefillKernel``: the ordered K16 segments
# and the four-FP32-op Kahan update per segment are unchanged, so every entry
# is expected to give the canonical kernel's bits at every row count (it does,
# see tests/integration/test_cuteafd_step5_compensated_tiles.py).
#
# Where the compensated route spends its time (issue at 8K, #703): small live
# widths stream weights and want many small CTAs and deep stages; the 64-256
# row chunks are balanced between the tensor pipe and the compensated FADD
# stream and want 8 warps; wider rows re-read the weight once per tile_m rows,
# so they want 128-row or 8-warp tiles. L2-flushed cold medians, SM120 RTX PRO
# 6000, 12-16 replays x 2-3 interleaved blocks, ratio against
# ``prefill_config``'s compensated tile (64x128x64, 3 stages, 4 warps):
#
#   q      (6144, 4096)   1-16 rows 49.15 -> 34.82 us, 41-64 49.15 -> 36.86,
#                     256 124.93 -> 104.45, 1024 381.49 -> 356.35
#   csa_o  (4096, 12288)  1-16 184.29 -> 67.58, 41-64 184.32 -> 73.73,
#                     256 221.18 -> 165.92, 1024 820.24 -> 741.92,
#                     2048 1465.87 -> 1278.96
#   swa_o  (4096, 6144) FP32 out  1-16 71.68 -> 34.85, 41-64 71.68 -> 38.91,
#                     256 90.14 -> 86.02, 1024 378.91 -> 362.53,
#                     2048 778.22 -> 741.90
#   kv     (384, 4096)    1 26.62 -> 8.19, 16-64 26.62 -> 10.24,
#                     256 26.62 -> 14.34, 2048 69.63 -> 45.06
#   gate_up (3072, 4096) 1-16 49.15 -> 18.43, 41-64 49.15 -> 20.48,
#                     256 51.20 -> 45.06, 1024 200.72 -> 180.24
#   kimi   (24752, 7168)  1 319.49 -> 227.33, 16 317.44 -> 229.38,
#                     41-64 319.49 -> 261.10, 256 702.98 -> 571.41,
#                     1024 2462.72 -> 2094.11, 2048 4869.12 -> 4134.94
#
# Rows a table does not cover keep ``prefill_config``'s compensated tile. That
# is deliberate at 2048 rows for q and gate_up: the 8-warp tiles there
# measured 1.14x and 1.08x in one session and 0.94x and 1.06x in the next
# (two arms with identical tiles also landed 3-6% apart inside one run), so
# the row band is withheld until the cell stops moving.
#
# The (6144, 4096) 16-row band is split at 8 rows. Both Duel B entrants
# measured the narrower, deeper 16x64x6 tile faster at 16 rows (11.17 vs
# 15.53 us, 1.39x, and it keeps 1.386x transplanted into
# ``RowTiledBf16Projection``), and a same-session rows 4-8 check puts the
# crossing at 8: rows 4-7 tie (L2-flushed cold 34.82 us for both arms, warm
# 18.08 vs 18.05-18.08) while row 8 warm goes 18.08 -> 13.95 us at an
# identical 34.82 us cold. The cold medians tie across 4-8, so the split costs
# nothing in the flushed regime and wins the L2-resident one.
COMPENSATED_TILES_BY_ROWS = {
    (6144, 4096): ((7, 1, 4, 16, 128, 5), (16, 1, 4, 16, 64, 6), (64, 4, 64, 4),
                   (256, 2, 4, 64, 64, 4), (1024, 2, 4, 64, 128, 2)),
    (4096, 12288): ((16, 1, 4, 16, 32, 12), (64, 2, 4, 32, 64, 5),
                    (4096, 2, 4, 64, 128, 2)),
    (4096, 6144): ((16, 1, 4, 16, 32, 12), (64, 2, 4, 32, 64, 5),
                   (4096, 2, 4, 64, 128, 3)),
    (384, 4096): ((64, 1, 2, 16, 16, 8), (256, 1, 4, 16, 64, 9), (1024, 4, 64, 4),
                  (4096, 4, 2, 128, 64, 3)),
    (3072, 4096): ((16, 1, 4, 16, 32, 12), (64, 2, 4, 32, 64, 5),
                   (4096, 2, 4, 64, 128, 3)),
    (24752, 7168): ((16, 1, 4, 16, 32, 12), (64, 4, 64, 4), (4096, 2, 4, 64, 128, 2)),
}


def prefill_config(n: int, k: int) -> dict:
    """Measured SM120 choices for the TMA tensor-core route (uncompensated)."""
    if n >= 1024:
        return dict(compute_warps=4, tile_n=128, num_stages=3, compensated=False)
    return dict(compute_warps=4, tile_n=64, num_stages=4, compensated=False)


class TmaBf16Projection:
    """Pointer ABI over ``Bf16PrefillKernel`` (TMA pipeline, FP32 MMA accumulation)."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, tiles_by_rows: tuple | None = None,
                 **config):
        from ._prefill import Bf16PrefillKernel

        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        if tiles_by_rows is None:
            tiles_by_rows = DECODE_TMA_TILES.get((self.n, self.k), ()) if not config else ()
        if tiles_by_rows:
            self.row_tiled = RowTiledBf16Projection(self.n, self.k, tiles_by_rows, out_dtype=out_dtype, **config)
            self.tiles_by_rows = self.row_tiled.tiles_by_rows
            self.config = self.row_tiled.config
            self.kernel = self.row_tiled.default.kernel
        else:
            self.row_tiled = None
            self.tiles_by_rows = ()
            self.config = dict(prefill_config(n, k), **config)
            self.kernel = Bf16PrefillKernel(self.n, self.k, **self.config)

    def key(self) -> tuple:
        return (self.n, self.k, tuple(sorted(self.config.items()))) \
            + ((("tiles_by_rows", self.tiles_by_rows),) if self.tiles_by_rows else ())

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        if cutlass.const_expr(self.row_tiled is not None):
            self.row_tiled(x, w, out, rows, stream)
        else:
            xt = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
            wt = cute.make_tensor(w, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
            ot = cute.make_tensor(out, cute.make_layout((rows, self.n), stride=(self.n, 1)))
            self.kernel(xt, wt, ot, rows, stream)


class RowTiledBf16Projection:
    """``TmaBf16Projection`` with its tiles picked by the live rows of a launch.

    ``tiles_by_rows`` entries are ``(rows, compute_warps, tile_n, num_stages)``,
    ``(rows, warp_m, warp_n, tile_n, num_stages)`` or
    ``(rows, warp_m, warp_n, tile_m, tile_n, num_stages)`` in
    ascending ``rows``: a launch runs the first entry whose ``rows`` it does
    not exceed (``16 * compute_warps`` x ``tile_n`` tiles, or ``warp_m`` x
    ``warp_n`` warps over a ``16 * warp_m`` or the explicit ``tile_m`` row
    tile, ``prefill_config``'s other settings; ``ceil(rows / tile_m)`` row
    tiles) and ``prefill_config``'s kernel past the last entry. The entries
    retile one MMA accumulation: each output keeps its K order, so every
    entry is expected to give the bits of ``prefill_config``'s kernel
    (tests/integration/test_cuteafd_glmf_kda_decode_tiles.py). The branch is
    on the ``rows`` launch scalar, as ``RoutedBf16Projection``'s.

    Keyword arguments override the ``prefill_config`` settings the entries do
    not set, for every entry and for the past-the-table default. The
    compensated route needs that: ``prefill_config`` selects
    ``compensated=False`` and every entry of ``COMPENSATED_TILES_BY_ROWS`` is
    a retile of the compensated K16 accumulation, so ``compensated=True``
    must reach all of them (it enters the key, so the two routes never share
    a compiled program).
    """

    def __init__(self, n: int, k: int, tiles_by_rows, *, out_dtype=cutlass.BFloat16, **base):
        self.n, self.k = int(n), int(k)
        self.tiles_by_rows = tuple(tuple(int(v) for v in entry) for entry in tiles_by_rows)
        if not self.tiles_by_rows or any(len(entry) not in (4, 5, 6) for entry in self.tiles_by_rows):
            raise ValueError(
                "tiles_by_rows entries are (rows, compute_warps, tile_n, num_stages), "
                "(rows, warp_m, warp_n, tile_n, num_stages) or "
                "(rows, warp_m, warp_n, tile_m, tile_n, num_stages)")
        self.limits = tuple(entry[0] for entry in self.tiles_by_rows)
        # Entry i serves floors[i] < rows <= limits[i].
        self.floors = (0,) + self.limits[:-1]
        if any(limit <= floor for floor, limit in zip(self.floors, self.limits)):
            raise ValueError("tiles_by_rows rows must be positive and ascending")
        self.base = dict(base)
        self.default = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=(), **self.base)
        self.config = self.default.config
        self.tiled = []
        for entry in self.tiles_by_rows:
            if len(entry) == 4:
                _, warps, tile_n, stages = entry
                self.tiled.append(TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=(),
                                                    compute_warps=warps, tile_n=tile_n, num_stages=stages,
                                                    **self.base))
            else:
                if len(entry) == 5:
                    _, warp_m, warp_n, tile_n, stages = entry
                    tile_m = 16 * warp_m
                else:
                    _, warp_m, warp_n, tile_m, tile_n, stages = entry
                if tile_m % (16 * warp_m):
                    raise ValueError(f"tiles_by_rows entry {entry}: tile_m must be a multiple of 16 * warp_m")
                self.tiled.append(TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=(),
                                                    warp_layout=(warp_m, warp_n), tile_m=tile_m,
                                                    tile_n=tile_n, num_stages=stages, **self.base))
        capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_120")
        for entry, launch in zip(self.tiles_by_rows, self.tiled):
            kernel = launch.kernel
            # Bf16PrefillKernel's shared storage: the barriers' block, then the A and B stages.
            smem = kernel.buffer_align_bytes + (kernel.tile_m + kernel.tile_n) * kernel.tile_k * 2 * kernel.num_stages
            if smem > capacity:
                raise ValueError(f"tiles_by_rows entry {entry} needs {smem} B of shared memory a CTA; SM120 has {capacity}")

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        for i in cutlass.range_constexpr(len(self.tiled)):
            if rows > Int32(self.floors[i]):
                if rows <= Int32(self.limits[i]):
                    self.tiled[i](x, w, out, rows, stream)
        if rows > Int32(self.limits[-1]):
            self.default(x, w, out, rows, stream)


class RoutedBf16Projection:
    """``out = x @ w^T``: skinny GEMV for few live rows, TMA tensor-core GEMM above.

    The branch is on the ``rows`` launch scalar inside the program (no host
    sync), so one compiled program serves decode and prefill and stays
    CUDA-graph capturable. ``x``/``w``/``out`` are dense and 16-byte aligned;
    ``out`` is ``[rows, N]`` BF16 or FP32. Both routes accumulate in FP32.
    """

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, max_skinny_rows: int | None = None,
                 wide: bool = False, tiles_by_rows: tuple = (), skinny_tiles_by_rows: tuple = ()):
        """``wide`` adds the ``WIDE_PREFILL_CONFIG`` TMA route from
        ``WIDE_PREFILL_MIN_ROWS`` live rows (prefill capacities).

        ``tiles_by_rows`` (default none) picks the TMA route's tiles by live rows
        below that (``RowTiledBf16Projection``; its first entry's rows exceed
        the skinny GEMV's)."""
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.max_skinny_rows = int(skinny_max_rows(n, k) if max_skinny_rows is None else max_skinny_rows)
        if self.max_skinny_rows > 0 and not skinny_tiles_by_rows:
            skinny_tiles_by_rows = SKINNY_TILES_BY_ROWS.get((self.n, self.k), ())
        if skinny_tiles_by_rows and self.max_skinny_rows > 0:
            self.skinny = RowTiledSkinnyBf16Gemv(self.n, self.k, skinny_tiles_by_rows, out_dtype=out_dtype)
            self.skinny_tiles_by_rows = self.skinny.tiles_by_rows
        else:
            self.skinny = SkinnyBf16Gemv(self.n, self.k, out_dtype=out_dtype, **skinny_config(n, k, out_dtype))
            self.skinny_tiles_by_rows = ()
        if not tiles_by_rows and self.max_skinny_rows == 0:
            tiles_by_rows = DECODE_TMA_TILES.get((self.n, self.k), ())
        if tiles_by_rows:
            self.large = RowTiledBf16Projection(self.n, self.k, tiles_by_rows, out_dtype=out_dtype)
            if self.large.limits[0] <= self.max_skinny_rows:
                raise ValueError("tiles_by_rows serves rows past max_skinny_rows")
        else:
            self.large = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=())
        self.tiles_by_rows = self.large.tiles_by_rows if tiles_by_rows else ()
        self.wide = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=(), **WIDE_PREFILL_CONFIG) \
            if wide and self.n % 2 == 0 else None

    @property
    def exact_row_cap(self) -> int:
        # Skinny row tiles keep the K partition; TMA row/wide tiles also keep
        # their MMA fold. Only crossing SIMT -> MMA changes row arithmetic.
        return self.max_skinny_rows if self.max_skinny_rows > 0 else 2**31 - 1

    def key(self) -> tuple:
        s = self.skinny
        # tiles_by_rows extends the key; projections without it keep theirs.
        return (self.n, self.k, str(self.out_dtype), self.max_skinny_rows, s.cols, s.rows_per_tile,
                s.threads, tuple(sorted(self.large.config.items())), self.wide is not None) \
            + ((("tiles_by_rows", self.tiles_by_rows),) if self.tiles_by_rows else ()) \
            + ((("skinny_tiles_by_rows", self.skinny_tiles_by_rows),) if self.skinny_tiles_by_rows else ())

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        if rows <= Int32(self.max_skinny_rows):
            self.skinny(x, w, out, rows, stream)
        else:
            if cutlass.const_expr(self.wide is not None):
                if rows >= Int32(WIDE_PREFILL_MIN_ROWS):
                    self.wide(x, w, out, rows, stream)
                else:
                    self.large(x, w, out, rows, stream)
            else:
                self.large(x, w, out, rows, stream)
