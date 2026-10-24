"""Per-row low-rank steering ``y -= A (B^T y)`` on BF16 rows, in place (ABI v2).

Weight-variant rows only: ``v = vidx[r]``; a row outside ``0 < v <= variants``
is rejected before accessing any table. A row with ``k[v] <= 0``
is never read or written, so base rows stay bit-identical. For a steered row,
with ``kv = min(k[v], k_max)``:

    t_j  = sum_h b[v][j][h] * x[h]                 (FP32, fixed order below)
    s_h  = sum_j a[v][j][h] * t_j                  (FP32, j ascending)
    x[h] = bf16_rne(float(x[h]) - s_h)             (one rounding, no saturation)

Tables ``a``/``b`` are FP32 ``[V+1][k_max][h]`` (slot 0 unused), ``k`` is
int32 ``[V+1]``, ``vidx`` int32 ``[rows]``, ``x`` BF16 ``[rows, ld]`` (first
``h`` columns). ``h`` is any multiple of 128 up to 8192; ``h``, ``ld``,
``rows``, ``k_max`` and ``variants`` (the actual V in 1..7) are launch scalars, so one compiled program serves
every model width and no live count enters a compile key.

Fixed reduction order (``steer_rows_ref`` in cuteafd copies it bit for bit):
one 512-thread CTA per row; thread ``t`` owns the 8-element vectors
``q = t`` and ``q = t + 512`` where ``8q < h``. Its partial is an
``fma.rn`` chain from 0 over its elements in ascending order; each warp sums
partials with a butterfly (offsets 1, 2, 4, 8, 16); ``t_j`` is
``((0 + w0) + w1) + ... + w15``. ``s`` starts at 0 and takes
``s = fma.rn(a_j, t_j, s)`` for ``j = 0 .. kv-1``. A row's bits depend only
on its own inputs, never on ``rows`` or other rows (batch-invariant).

Two programs: ``one`` (``k_max == 1``; every load issues before the single
reduction) and ``general`` (``k_max <= 32``; ``j`` in tiles of four). Both
give identical bits at ``k == 1``. No allocation and no host sync: graph
capture records the pointers and scalars. Keep tables and layout metadata
immutable until graph retirement; only row assignments change between replays.
Allocation padding never increases V or the initialized table extent.

``AddRows`` (``add_rows_f32`` / ``add_rows_bf16``) is the stored-delta site:
``y = bf16_rne(y + d)`` on rows whose variant has ``on[v] != 0``.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import ld_global_v4_u32, st_global_v4_u32, u32_as_f32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo, _ld_cached

from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _sub

THREADS = 512
WARPS = THREADS // 32
VECTORS = 2           # 8-element BF16 vectors per thread at h = 8192
MAX_HIDDEN = THREADS * 8 * VECTORS
MAX_K = 32
MAX_VARIANTS = 7
ABI_VERSION = 2
J_TILE = 4


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """RNE pack without ``satfinite``: overflow rounds to infinity as bf16_rne does."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = _add(value, Float32(cute.arch.shuffle_sync_bfly(value, offset=1 << shift)))
    return value


def _storage(slots: int):
    class Storage:
        pass

    Storage.__annotations__ = {
        "sums": cute.struct.Align[cute.struct.MemRange[Float32, slots * WARPS], 16],
    }
    return cute.struct(Storage)


class SteerRows:
    """Launch ``(x, vidx, a, b, k, ld, rows, h, k_max, variants, stream)``; one CTA per row."""

    def __init__(self, one: bool):
        self.one = bool(one)
        self.slots = 1 if self.one else MAX_K

    @cute.jit
    def __call__(self, x: cute.Pointer, vidx: cute.Pointer, a: cute.Pointer, b: cute.Pointer,
                 k: cute.Pointer, ld: Int64, rows: Int32, h: Int32, k_max: Int32, variants: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(x, vidx, a, b, k, ld, h, k_max, variants).launch(
                grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.jit
    def _load_f32x8(self, address: Int64, out: cute.Tensor, base: cutlass.Constexpr):
        lo = _ld_cached(address)
        hi = _ld_cached(address + Int64(16))
        for i in cutlass.range_constexpr(4):
            out[base + i] = u32_as_f32(lo[i])
            out[base + 4 + i] = u32_as_f32(hi[i])

    @cute.kernel
    def kernel(self, x: cute.Pointer, vidx: cute.Pointer, a: cute.Pointer, b: cute.Pointer,
               k: cute.Pointer, ld: Int64, h: Int32, k_max: Int32, variants: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = tid % Int32(32)
        warp = tid // Int32(32)
        row = Int64(cute.arch.block_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_storage(self.slots))
        sums = storage.sums.get_tensor(cute.make_layout((self.slots, WARPS), stride=(WARPS, 1)))
        v = cute.make_tensor(vidx, cute.make_layout(1, stride=1))[row]
        kv = Int32(0)
        if (variants > Int32(0)) & (variants <= Int32(MAX_VARIANTS)):
            if (v > Int32(0)) & (v <= variants):
                kv = cute.make_tensor(k, cute.make_layout(1, stride=1))[v]
        if cutlass.const_expr(self.one):
            kv = cutlass.min(kv, Int32(1))
        else:
            kv = cutlass.min(kv, k_max)
        if kv > Int32(0):
            vectors = h // Int32(8)
            x_row = Int64(x.toint()) + row * ld * Int64(2)
            # Table row (v, j) starts at ((v * k_max + j) * h) floats.
            table = Int64(v) * Int64(k_max) * Int64(h) * Int64(4)
            a_base = Int64(a.toint()) + table
            b_base = Int64(b.toint()) + table
            values = cute.make_rmem_tensor(cute.make_layout((8 * VECTORS,), stride=(1,)), Float32)
            for i in cutlass.range_constexpr(VECTORS):
                q = Int32(i * THREADS) + tid
                if q < vectors:
                    words = ld_global_v4_u32(x_row + Int64(q) * Int64(16))
                    for w in cutlass.range_constexpr(4):
                        values[8 * i + 2 * w] = _bf16_lo(words[w])
                        values[8 * i + 2 * w + 1] = _bf16_hi(words[w])
            steer = cute.make_rmem_tensor(cute.make_layout((8 * VECTORS,), stride=(1,)), Float32)
            for e in cutlass.range_constexpr(8 * VECTORS):
                steer[e] = Float32(0.0)
            if cutlass.const_expr(self.one):
                self._one(values, steer, a_base, b_base, sums, tid, lane, warp, vectors)
            else:
                self._general(values, steer, a_base, b_base, sums, tid, lane, warp, vectors,
                              kv, h)
            for i in cutlass.range_constexpr(VECTORS):
                q = Int32(i * THREADS) + tid
                if q < vectors:
                    o = [_sub(values[8 * i + e], steer[8 * i + e]) for e in range(8)]
                    st_global_v4_u32(x_row + Int64(q) * Int64(16),
                                     _pack_bf16x2(o[0], o[1]), _pack_bf16x2(o[2], o[3]),
                                     _pack_bf16x2(o[4], o[5]), _pack_bf16x2(o[6], o[7]))

    @cute.jit
    def _one(self, values: cute.Tensor, steer: cute.Tensor, a_base: Int64, b_base: Int64,
             sums: cute.Tensor, tid: Int32, lane: Int32, warp: Int32, vectors: Int32):
        bv = cute.make_rmem_tensor(cute.make_layout((8 * VECTORS,), stride=(1,)), Float32)
        av = cute.make_rmem_tensor(cute.make_layout((8 * VECTORS,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(VECTORS):
            q = Int32(i * THREADS) + tid
            if q < vectors:
                self._load_f32x8(b_base + Int64(q) * Int64(32), bv, 8 * i)
                self._load_f32x8(a_base + Int64(q) * Int64(32), av, 8 * i)
        partial = Float32(0.0)
        for i in cutlass.range_constexpr(VECTORS):
            q = Int32(i * THREADS) + tid
            if q < vectors:
                for e in cutlass.range_constexpr(8):
                    partial = _fma(bv[8 * i + e], values[8 * i + e], partial)
        partial = _warp_sum(partial)
        if lane == Int32(0):
            sums[0, warp] = partial
        cute.arch.sync_threads()
        t = Float32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            t = _add(t, sums[0, w])
        for i in cutlass.range_constexpr(VECTORS):
            q = Int32(i * THREADS) + tid
            if q < vectors:
                for e in cutlass.range_constexpr(8):
                    steer[8 * i + e] = _fma(av[8 * i + e], t, steer[8 * i + e])

    @cute.jit
    def _general(self, values: cute.Tensor, steer: cute.Tensor, a_base: Int64, b_base: Int64,
                 sums: cute.Tensor, tid: Int32, lane: Int32, warp: Int32, vectors: Int32,
                 kv: Int32, h: Int32):
        stride = Int64(h) * Int64(4)
        tile = cute.make_rmem_tensor(cute.make_layout((J_TILE * 8 * VECTORS,), stride=(1,)), Float32)
        partials = cute.make_rmem_tensor(cute.make_layout((J_TILE,), stride=(1,)), Float32)
        tiles = (kv + Int32(J_TILE - 1)) // Int32(J_TILE)
        for jt in cutlass.range(tiles, unroll=1):
            j0 = jt * Int32(J_TILE)
            for u in cutlass.range_constexpr(J_TILE):
                partials[u] = Float32(0.0)
                if j0 + Int32(u) < kv:
                    row_b = b_base + Int64(j0 + Int32(u)) * stride
                    for i in cutlass.range_constexpr(VECTORS):
                        q = Int32(i * THREADS) + tid
                        if q < vectors:
                            self._load_f32x8(row_b + Int64(q) * Int64(32), tile, (u * VECTORS + i) * 8)
            for u in cutlass.range_constexpr(J_TILE):
                if j0 + Int32(u) < kv:
                    p = Float32(0.0)
                    for i in cutlass.range_constexpr(VECTORS):
                        q = Int32(i * THREADS) + tid
                        if q < vectors:
                            for e in cutlass.range_constexpr(8):
                                p = _fma(tile[(u * VECTORS + i) * 8 + e], values[8 * i + e], p)
                    partials[u] = p
            # Shuffles run on every lane; the tile predicate is CTA-uniform.
            for u in cutlass.range_constexpr(J_TILE):
                if j0 + Int32(u) < kv:
                    total = _warp_sum(partials[u])
                    if lane == Int32(0):
                        sums[j0 + Int32(u), warp] = total
        cute.arch.sync_threads()
        for jt in cutlass.range(tiles, unroll=1):
            j0 = jt * Int32(J_TILE)
            for u in cutlass.range_constexpr(J_TILE):
                if j0 + Int32(u) < kv:
                    row_a = a_base + Int64(j0 + Int32(u)) * stride
                    for i in cutlass.range_constexpr(VECTORS):
                        q = Int32(i * THREADS) + tid
                        if q < vectors:
                            self._load_f32x8(row_a + Int64(q) * Int64(32), tile, (u * VECTORS + i) * 8)
            for u in cutlass.range_constexpr(J_TILE):
                if j0 + Int32(u) < kv:
                    t = Float32(0.0)
                    for w in cutlass.range_constexpr(WARPS):
                        t = _add(t, sums[j0 + Int32(u), w])
                    for i in cutlass.range_constexpr(VECTORS):
                        q = Int32(i * THREADS) + tid
                        if q < vectors:
                            for e in cutlass.range_constexpr(8):
                                steer[8 * i + e] = _fma(tile[(u * VECTORS + i) * 8 + e], t,
                                                        steer[8 * i + e])


OPERANDS = (
    Operand("x", torch.bfloat16, "[rows,ld]", "inout", note="first h columns, in place"),
    Operand("vidx", torch.int32, "[rows]", align=4),
    Operand("a", torch.float32, "[V+1,k_max,h]"),
    Operand("b", torch.float32, "[V+1,k_max,h]"),
    Operand("k", torch.int32, "[V+1]", align=4),
)
SCALARS = (Scalar("ld", "int64"), Scalar("rows"), Scalar("h"), Scalar("k_max"), Scalar("variants"))
FORMS = {"one": True, "general": False}


def steer_rows_launch(form: str) -> SteerRows:
    return SteerRows(FORMS[form])


def compile_steer_rows_aot(form: str = "general"):
    """``form`` ``one`` (k_max == 1) or ``general`` (k_max <= 32); no geometry key."""
    if form not in FORMS:
        raise ValueError("steer_rows form must be 'one' or 'general'")
    geometry = dict(abi_version=ABI_VERSION, form=form, threads=THREADS, max_hidden=MAX_HIDDEN,
                    max_k=1 if FORMS[form] else MAX_K, hidden_multiple=128)
    return compile_program(steer_rows_launch(form), name=f"steer_rows_{form}", operands=OPERANDS,
                           scalars=SCALARS, key=tuple(geometry.values()), geometry=geometry,
                           doc=__doc__)


# ---------------------------------------------------------------------------
# Delta rows: y += d on variant rows (stored-delta sites)
# ---------------------------------------------------------------------------

ADD_THREADS = 256


class AddRows:
    """``y[r][i] = bf16_rne(float(y[r][i]) + float(d[r][i]))`` when ``v = vidx[r] > 0`` and
    ``on[v] != 0``; every other row is never read or written. One FP32 add and one
    rounding per element, so the result is exact and batch-invariant.

    Launch ``(y, d, vidx, on, ld_y, ld_d, rows, h, variants, stream)``;
    reject indices outside ``0 < v <= variants <= 7`` before accessing ``on`` or rows.
    ``h`` and both strides
    are multiples of 8 elements, ``y`` and ``d`` 16-byte aligned. ``d`` is FP32
    (``delta_bf16=False``) or BF16.
    """

    def __init__(self, delta_bf16: bool):
        self.delta_bf16 = bool(delta_bf16)

    @cute.jit
    def __call__(self, y: cute.Pointer, d: cute.Pointer, vidx: cute.Pointer, on: cute.Pointer,
                 ld_y: Int64, ld_d: Int64, rows: Int32, h: Int32, variants: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(y, d, vidx, on, ld_y, ld_d, h, variants).launch(
                grid=(rows, 1, 1), block=(ADD_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, y: cute.Pointer, d: cute.Pointer, vidx: cute.Pointer, on: cute.Pointer,
               ld_y: Int64, ld_d: Int64, h: Int32, variants: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        row = Int64(cute.arch.block_idx()[0])
        v = cute.make_tensor(vidx, cute.make_layout(1, stride=1))[row]
        enabled = Int32(0)
        if (variants > Int32(0)) & (variants <= Int32(MAX_VARIANTS)):
            if (v > Int32(0)) & (v <= variants):
                enabled = cute.make_tensor(on, cute.make_layout(1, stride=1))[v]
        if enabled != Int32(0):
            y_row = Int64(y.toint()) + row * ld_y * Int64(2)
            element = 2 if self.delta_bf16 else 4
            d_row = Int64(d.toint()) + row * ld_d * Int64(element)
            for q in cutlass.range(tid, h // Int32(8), ADD_THREADS):
                words = ld_global_v4_u32(y_row + Int64(q) * Int64(16))
                delta = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
                if cutlass.const_expr(self.delta_bf16):
                    dw = _ld_cached(d_row + Int64(q) * Int64(16))
                    for w in cutlass.range_constexpr(4):
                        delta[2 * w] = _bf16_lo(dw[w])
                        delta[2 * w + 1] = _bf16_hi(dw[w])
                else:
                    lo = _ld_cached(d_row + Int64(q) * Int64(32))
                    hi = _ld_cached(d_row + Int64(q) * Int64(32) + Int64(16))
                    for w in cutlass.range_constexpr(4):
                        delta[w] = u32_as_f32(lo[w])
                        delta[4 + w] = u32_as_f32(hi[w])
                st_global_v4_u32(y_row + Int64(q) * Int64(16),
                                 _pack_bf16x2(_add(_bf16_lo(words[0]), delta[0]), _add(_bf16_hi(words[0]), delta[1])),
                                 _pack_bf16x2(_add(_bf16_lo(words[1]), delta[2]), _add(_bf16_hi(words[1]), delta[3])),
                                 _pack_bf16x2(_add(_bf16_lo(words[2]), delta[4]), _add(_bf16_hi(words[2]), delta[5])),
                                 _pack_bf16x2(_add(_bf16_lo(words[3]), delta[6]), _add(_bf16_hi(words[3]), delta[7])))


ADD_FORMS = {"f32": False, "bf16": True}


def add_rows_operands(form: str):
    return (
        Operand("y", torch.bfloat16, "[rows,ld_y]", "inout", note="first h columns, in place"),
        Operand("d", torch.bfloat16 if ADD_FORMS[form] else torch.float32, "[rows,ld_d]"),
        Operand("vidx", torch.int32, "[rows]", align=4),
        Operand("on", torch.int32, "[V+1]", align=4),
    )


ADD_SCALARS = (Scalar("ld_y", "int64"), Scalar("ld_d", "int64"), Scalar("rows"), Scalar("h"), Scalar("variants"))


def add_rows_launch(form: str) -> AddRows:
    return AddRows(ADD_FORMS[form])


def compile_add_rows_aot(form: str = "f32"):
    """``form`` is the delta dtype, ``f32`` or ``bf16``; no geometry key."""
    if form not in ADD_FORMS:
        raise ValueError("add_rows form must be 'f32' or 'bf16'")
    geometry = dict(abi_version=ABI_VERSION, form=form, threads=ADD_THREADS, hidden_multiple=8)
    return compile_program(add_rows_launch(form), name=f"add_rows_{form}",
                           operands=add_rows_operands(form), scalars=ADD_SCALARS,
                           key=tuple(geometry.values()), geometry=geometry, doc=AddRows.__doc__)
