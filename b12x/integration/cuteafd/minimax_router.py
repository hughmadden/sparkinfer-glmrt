"""MiniMax FP32-weight router logits (BF16 activations, exact FP32 weights).

``out[m, n] = sum_k x[m, k] * w[n, k]`` with BF16 ``x``, FP32 ``w`` and FP32
``out``.  The router weight is consumed at the checkpoint's full FP32
precision: each thread loads four FP32 weight values from every 16-byte
vector and never rounds them to BF16 or TF32.

Derived from the donated router GEMV shape (arithmetic and layout reference:
Hugh Madden, hughmadden/cuteafd) and from
``b12x.gemm.bf16_gemv._skinny.SkinnyBf16Gemv``: same CTA/thread structure,
with the weight lane reinterpreted as four FP32 values instead of eight BF16
values and the activation lane as four BF16 values instead of eight.

A CTA owns ``cols`` router columns and a tile of ``rows_per_tile`` live rows;
its ``threads`` threads split K into interleaved four-element vectors, so
thread ``t`` sums k-vectors ``t, t + threads, ...`` in ascending order for
every output of the tile.  Each weight vector is loaded once per CTA and
reused by the tile's rows; each activation vector is loaded once per row and
reused by the tile's columns.  A 32-lane butterfly and an in-order sum over
warp partials finish each logit, so a logit is within the rounding of one
FP32-accumulated dot product.

Every tile shape of a shape shares one CTA thread count and one K partition,
so a logit is bitwise identical at every live row count: the tile shape
changes only which CTA and which thread index store it, never how its
K values are summed or folded.  A partial tail tile keeps that property by
sliding its window back to a full tile and running the full-tile address
pattern; it stores only the rows its own (unshifted) window owns, so the
earlier rows are written once, by the previous tile.  Only a launch with
fewer live rows than the tile (a single-tile launch) still pairs dead slots
with row 0 and drops their accumulators.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import u32_as_f32

__all__ = [
    "MINIMAX_ROUTER_TILES",
    "RouterF32Gemv",
    "RowTiledRouterF32",
    "minimax_router_scores",
]


@dsl_user_op
def _ld_v2(address: Int64, asm: str, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)], asm, "=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _ld_v4(address: Int64, asm: str, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)], asm, "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


def _ld_stream(address):
    """One 16-byte FP32 weight vector, streamed past L1 (no reuse in-CTA)."""
    return _ld_v4(address, "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];")


def _ld_cached(address):
    """One 8-byte BF16 activation vector, reused by every column of the tile."""
    return _ld_v2(address, "ld.global.nc.v2.u32 {$0, $1}, [$2];")


@dsl_user_op
def _bf16_lo(word, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(word).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 16;", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _bf16_hi(word, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(word).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


class RouterF32Gemv:
    """Pointer ABI ``(x, w, out, rows, stream)``; x [rows,K] BF16, w [N,K] F32.

    One CTA per ``cols`` router columns and ``rows_per_tile`` live rows.
    """

    def __init__(self, n: int, k: int, *, cols: int, rows_per_tile: int,
                 threads: int = 128):
        self.n, self.k = int(n), int(k)
        self.cols, self.rows_per_tile, self.threads = (
            int(cols), int(rows_per_tile), int(threads))
        if self.k % 4:
            raise ValueError("K must divide into four-element FP32 vectors")
        self.vectors = self.k // 4
        if self.vectors % self.threads:
            raise ValueError("K/4 must split evenly over the CTA thread count")
        if self.n % self.cols:
            raise ValueError("N must be a multiple of cols")
        if self.threads % 32:
            raise ValueError("threads must be a multiple of 32")
        self.steps = self.vectors // self.threads
        self.warps = self.threads // 32
        self.outputs = self.rows_per_tile * self.cols
        if self.outputs > self.threads:
            raise ValueError("rows_per_tile * cols must not exceed the CTA thread count")

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[
                cute.struct.MemRange[Float32, self.warps * self.outputs], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        tiles = (rows + Int32(self.rows_per_tile - 1)) // Int32(self.rows_per_tile)
        self.kernel(x, w, out, rows).launch(
            grid=(self.n // self.cols, tiles, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
               rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        col0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        tile0 = Int32(cute.arch.block_idx()[1]) * Int32(self.rows_per_tile)
        live = rows - tile0
        row0 = tile0
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(
            cute.make_layout((self.warps, self.outputs), stride=(self.outputs, 1)))
        acc = cute.make_rmem_tensor(cute.make_layout((self.outputs,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.outputs):
            acc[i] = Float32(0.0)
        w_base = (Int64(w.toint()) + col0 * Int64(self.k * 4)
                  + Int64(tidx) * Int64(16))
        x_base = (Int64(x.toint()) + Int64(tile0) * Int64(self.k * 2)
                  + Int64(tidx) * Int64(8))
        if live >= Int32(self.rows_per_tile):
            self._loop(acc, w_base, x_base, self.rows_per_tile)
        else:
            handled = Int32(0)
            if cutlass.const_expr(self.rows_per_tile > 8):
                if live == Int32(8):
                    self._loop(acc, w_base, x_base, 8)
                    handled = Int32(1)
            if cutlass.const_expr(self.rows_per_tile > 4):
                if live == Int32(4):
                    self._loop(acc, w_base, x_base, 4)
                    handled = Int32(1)
            if cutlass.const_expr(self.rows_per_tile > 2):
                if live == Int32(2):
                    self._loop(acc, w_base, x_base, 2)
                    handled = Int32(1)
            if cutlass.const_expr(self.rows_per_tile > 1):
                if live == Int32(1):
                    self._loop(acc, w_base, x_base, 1)
                    handled = Int32(1)
            if handled == Int32(0):
                if rows >= Int32(self.rows_per_tile):
                    # Slide the partial tail back to a full tile: same K
                    # partition and fold, full-tile address pattern instead of a
                    # per-row clamped one.  The store keeps the unshifted window
                    # as its lower bound, so the rows the previous tile owns are
                    # written once and the shift adds no duplicate writes.
                    row0 = rows - Int32(self.rows_per_tile)
                    shifted_base = (Int64(x.toint()) + Int64(row0) * Int64(self.k * 2)
                                    + Int64(tidx) * Int64(8))
                    self._loop(acc, w_base, shifted_base, self.rows_per_tile)
                else:
                    self._loop_partial(acc, w_base, x_base, live)
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
            self._store(partial, out, col0, row0, tile0, rows, tidx)

    @cute.jit
    def _loop(self, acc: cute.Tensor, w_base: Int64, x_base: Int64,
              live_rows: cutlass.Constexpr):
        for step in cutlass.range_constexpr(self.steps):
            w_offset = step * self.threads * 16
            x_offset = step * self.threads * 8
            wv = cute.make_rmem_tensor(
                cute.make_layout((self.cols * 4,), stride=(1,)), Float32)
            for c in cutlass.range_constexpr(self.cols):
                words = _ld_stream(w_base + Int64(c * self.k * 4 + w_offset))
                for j in cutlass.range_constexpr(4):
                    wv[4 * c + j] = u32_as_f32(words[j])
            for m in cutlass.range_constexpr(live_rows):
                self._accumulate(acc, wv, x_base + Int64(m * self.k * 2 + x_offset), m)

    @cute.jit
    def _loop_partial(self, acc: cute.Tensor, w_base: Int64, x_base: Int64,
                      live: Int32):
        """Single-tile launch: dead slots re-read row 0 and are dropped by the store.

        Only reached when the live count is smaller than the tile, where a full
        window cannot slide back; clamping the row offset keeps the path
        branch-free and cannot change a live row's accumulation.
        """
        for step in cutlass.range_constexpr(self.steps):
            w_offset = step * self.threads * 16
            x_offset = step * self.threads * 8
            wv = cute.make_rmem_tensor(
                cute.make_layout((self.cols * 4,), stride=(1,)), Float32)
            for c in cutlass.range_constexpr(self.cols):
                words = _ld_stream(w_base + Int64(c * self.k * 4 + w_offset))
                for j in cutlass.range_constexpr(4):
                    wv[4 * c + j] = u32_as_f32(words[j])
            for m in cutlass.range_constexpr(self.rows_per_tile):
                row_bytes = Int64(m) * Int64(self.k * 2)
                if Int32(m) >= live:
                    row_bytes = Int64(0)
                self._accumulate(acc, wv, x_base + row_bytes + Int64(x_offset), m)

    @cute.jit
    def _accumulate(self, acc: cute.Tensor, wv: cute.Tensor, x_address: Int64,
                    m: cutlass.Constexpr):
        words = _ld_cached(x_address)
        xs = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        xs[0] = _bf16_lo(words[0])
        xs[1] = _bf16_hi(words[0])
        xs[2] = _bf16_lo(words[1])
        xs[3] = _bf16_hi(words[1])
        for c in cutlass.range_constexpr(self.cols):
            a = acc[m * self.cols + c]
            for j in cutlass.range_constexpr(4):
                a = a + wv[4 * c + j] * xs[j]
            acc[m * self.cols + c] = a

    @cute.jit
    def _store(self, partial: cute.Tensor, out: cute.Pointer, col0: Int64,
               row0: Int32, tile0: Int32, rows: Int32, index: Int32):
        m = index // Int32(self.cols)
        c = index % Int32(self.cols)
        row = row0 + m
        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = total + partial[src, index]
        address = (Int64(out.toint())
                   + (Int64(row) * Int64(self.n) + col0 + Int64(c)) * Int64(4))
        target = cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)
        if row >= tile0:
            if row < rows:
                target[0] = total


# Router tile shapes for MiniMax's [128, 6144] FP32 weight, ascending live rows:
# (rows up to, cols, rows_per_tile). Every entry keeps threads=128 and one
# K partition, so all shapes produce bitwise identical logits. Narrow tiles
# keep the CTA count high while rows are few; wide tiles cut activation
# re-reads as rows grow (each column group reads the whole tile of x). The
# 4x16 shape was dropped: at 256 rows it measured 21.7-24.5 us against
# 19.2-19.4 us for 8x16 at 257-258 rows, so the wider shape serves 65+ rows.
MINIMAX_ROUTER_TILES = ((16, 2, 8), (64, 4, 8), (4096, 8, 16))


class RowTiledRouterF32:
    """``RouterF32Gemv`` tile shapes picked by live rows, one K partition."""

    def __init__(self, n: int, k: int, tiles, *, threads: int = 128):
        self.n, self.k = int(n), int(k)
        self.tiles = tuple(tuple(int(v) for v in entry) for entry in tiles)
        if not self.tiles or any(len(entry) != 3 for entry in self.tiles):
            raise ValueError("router tiles entries are (rows up to, cols, rows_per_tile)")
        self.limits = tuple(entry[0] for entry in self.tiles)
        self.floors = (0,) + self.limits[:-1]
        if any(limit <= floor for floor, limit in zip(self.floors, self.limits)):
            raise ValueError("router tile rows must be positive and ascending")
        self.threads = int(threads)
        self.kernels = [RouterF32Gemv(n, k, cols=c, rows_per_tile=r, threads=threads)
                        for _, c, r in self.tiles]
        self.default = self.kernels[-1]

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        for i in cutlass.range_constexpr(len(self.kernels)):
            if rows > Int32(self.floors[i]):
                if rows <= Int32(self.limits[i]):
                    self.kernels[i](x, w, out, rows, stream)
        if rows > Int32(self.limits[-1]):
            self.default(x, w, out, rows, stream)


def minimax_router_scores(n: int = 128, k: int = 6144) -> RowTiledRouterF32:
    """MiniMax router: BF16 activations, the checkpoint's FP32 weight, FP32 logits."""
    return RowTiledRouterF32(n, k, MINIMAX_ROUTER_TILES)
