"""CuTe DSL kernels of the exact FP8 routed-expert program (``fp8_moe``).

Checkpoint experts are E4M3 ``[N, K]`` with one FP32 scale per 128x128
block (``weight_scale_inv``). The GEMMs widen every weight to ``bf16(w * s)``
in registers or shared memory (the reference's dequantized BF16 weight) and
multiply BF16 activations on tensor cores (m16n8k16, FP32 accumulation), so
results match ``F.linear(x, dequant(w))`` up to FP32 summation order.
Nothing is re-quantized.

Pipeline over ``rows`` input rows with ``top_k`` routes each (``P = rows *
top_k`` (row, slot) pairs):

``MoePrep``        one CTA: per-expert pair counts, group offsets (padded to
                   ``pad`` rows for the tiled GEMM), ``pair_row`` (source row
                   of every grouped position, -1 for padding), ``pair_pos``
                   (grouped position of every (row, slot), -1 for invalid
                   expert ids), the active-expert list and the tile table.
``GatherRows``     BF16 activation rows: input rows (BF16, or FP8 K32 wire
                   rows ``[K E4M3 | K/32 UE8M0]``, converted exactly) in
                   grouped order (prefill) or in row order (decode).
``GroupedFp8Gemv`` decode: per active expert, its few grouped rows against the
                   streamed expert weight (``MmaFp8Gemv`` per expert).
``GroupedFp8Gemm`` prefill: per 64-row tile of one expert's group, the TMA
                   warp-MMA GEMM with E4M3 weight tiles widened in shared
                   memory (``TmaFp8Gemm`` per tile).
``MoeSwiGLU``      ``act = bf16(bf16(silu(g)) * u)``, optional DeepSeek clamp
                   (``u`` to [-L, L], ``g`` to at most L).
``MoeCombine``     ``out[r] = bf16(sum_k w[r, k] * y[pair_pos[r, k]])``, FP32
                   sum in slot order.

Metadata (int32, ``meta_words(E, max_tiles)`` words): ``[0]`` active expert
count, ``[1]`` tile count, ``[2]`` grouped rows used; then from word
``META_HEAD``: counts ``[E]``, offsets ``[E]``, active list ``[E]``, tile
experts ``[max_tiles]``. Grouped results are invariant to the order of rows
inside a group (every GEMM row depends only on its own input row), so the
atomic placement is deterministic in value.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    div_rn_f32,
    ld_global_v4_u32,
    ld_shared_v2_u32,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo, _ld_cached, _ld_stream

from ._fp8_weights import _e4m3x4_scaled_bf16x2x2, _e4m3x8_scaled_bf16, _ld_f32

META_HEAD = 4


def meta_words(experts: int, max_tiles: int) -> int:
    return META_HEAD + 3 * int(experts) + int(max_tiles)


def _ceil(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


@cute.jit
def _i32_at(base: Int64, index) -> Int32:
    return Int32(cute.make_ptr(Int32, base + Int64(index) * Int64(4), cute.AddressSpace.gmem, assumed_align=4)[0])


@cute.jit
def _set_i32(base: Int64, index, value):
    p = cute.make_ptr(Int32, base + Int64(index) * Int64(4), cute.AddressSpace.gmem, assumed_align=4)
    p[0] = Int32(value)


@dsl_user_op
def _u32_as_f32(bits, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(bits).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_u8(address, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.u8 $0, [$1];", "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_v2_u32(address, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]), [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];", "=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _atomic_add_shared_i32(address, value, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "atom.shared.add.u32 $0, [$1], $2;", "=r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# ---------------------------------------------------------------------------
# Route grouping
# ---------------------------------------------------------------------------


class MoePrep:
    """Group the (row, slot) pairs by expert in one CTA (see module docstring).
    ``grouped_cap`` is the number of grouped rows the scratch holds (every
    one is reset to padding first)."""

    threads = 1024

    def __init__(self, *, experts: int, top_k: int, pad: int, max_tiles: int):
        self.experts, self.top_k, self.pad, self.max_tiles = int(experts), int(top_k), int(pad), int(max_tiles)
        if self.experts > self.threads:
            raise ValueError("MoePrep handles at most 1024 experts")

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "counts": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "cursor": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "flags": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, ids: cute.Pointer, meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer,
                 rows: Int32, grouped_cap: Int32, stream: cuda.CUstream):
        self.kernel(ids, meta, pair_row, pair_pos, rows, grouped_cap).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer,
               rows: Int32, grouped_cap: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        counts = storage.counts.get_tensor(cute.make_layout((self.experts,)))
        cursor = storage.cursor.get_tensor(cute.make_layout((self.experts,)))
        scan_flags = storage.flags.get_tensor(cute.make_layout((self.experts,)))
        counts_smem = shared_ptr_to_u32(storage.counts.data_ptr())
        cursor_smem = shared_ptr_to_u32(storage.cursor.data_ptr())
        e_count = Int32(self.experts)
        m = Int64(meta.toint())
        counts_at = m + Int64(4 * META_HEAD)
        offsets_at = counts_at + Int64(4 * self.experts)
        active_at = offsets_at + Int64(4 * self.experts)
        tiles_at = active_at + Int64(4 * self.experts)
        ids_at = Int64(ids.toint())
        rows_at = Int64(pair_row.toint())
        pos_at = Int64(pair_pos.toint())
        pairs = rows * Int32(self.top_k)
        if tidx < e_count:
            counts[tidx] = Int32(0)
        g = tidx
        while g < grouped_cap:
            _set_i32(rows_at, g, Int32(-1))
            g = g + Int32(self.threads)
        cute.arch.sync_threads()
        p = tidx
        while p < pairs:
            e = _i32_at(ids_at, p)
            if (e >= Int32(0)) & (e < e_count):
                _atomic_add_shared_i32(counts_smem + e * Int32(4), Int32(1))
            p = p + Int32(self.threads)
        cute.arch.sync_threads()
        # Exclusive scans over experts (thread e owns expert e): padded rows
        # and active flags, Hillis-Steele in shared memory.
        n = Int32(0)
        rows_e = Int32(0)
        flag = Int32(0)
        if tidx < e_count:
            n = Int32(counts[tidx])
            rows_e = (n + Int32(self.pad - 1)) // Int32(self.pad) * Int32(self.pad)
            flag = cutlass.select_(n > Int32(0), Int32(1), Int32(0))
            cursor[tidx] = rows_e
            scan_flags[tidx] = flag
        cute.arch.sync_threads()
        step = 1
        while step < self.experts:
            a = Int32(0)
            b = Int32(0)
            if (tidx < e_count) & (tidx >= Int32(step)):
                a = Int32(cursor[tidx - Int32(step)])
                b = Int32(scan_flags[tidx - Int32(step)])
            cute.arch.sync_threads()
            if (tidx < e_count) & (tidx >= Int32(step)):
                cursor[tidx] = Int32(cursor[tidx]) + a
                scan_flags[tidx] = Int32(scan_flags[tidx]) + b
            cute.arch.sync_threads()
            step = step * 2
        if tidx < e_count:
            offset = Int32(cursor[tidx]) - rows_e
            active = Int32(scan_flags[tidx]) - flag
            _set_i32(counts_at, tidx, n)
            _set_i32(offsets_at, tidx, offset)
            if n > Int32(0):
                _set_i32(active_at, active, tidx)
                first_tile = offset // Int32(self.pad)
                for t in cutlass.range(rows_e // Int32(self.pad), unroll=1):
                    if first_tile + Int32(t) < Int32(self.max_tiles):
                        _set_i32(tiles_at, first_tile + Int32(t), tidx)
            if tidx == e_count - Int32(1):
                _set_i32(m, 0, active + flag)
                _set_i32(m, 1, (offset + rows_e) // Int32(self.pad))
                _set_i32(m, 2, offset + rows_e)
        cute.arch.sync_threads()
        if tidx < e_count:
            cursor[tidx] = Int32(cursor[tidx]) - rows_e
        cute.arch.sync_threads()
        p = tidx
        while p < pairs:
            e = _i32_at(ids_at, p)
            position = Int32(-1)
            if (e >= Int32(0)) & (e < e_count):
                position = _atomic_add_shared_i32(cursor_smem + e * Int32(4), Int32(1))
                _set_i32(rows_at, position, p // Int32(self.top_k))
            _set_i32(pos_at, p, position)
            p = p + Int32(self.threads)


# ---------------------------------------------------------------------------
# Activation rows
# ---------------------------------------------------------------------------


class GatherRows:
    """BF16 activation rows ``dst[i]``: ``src[pair_row[i]]`` for grouped rows
    ``i < meta[2]`` (``gather``; padding skips) or ``src[i]``. ``src`` is BF16
    ``[rows, K]`` or FP8 K32 wire rows (``wire``: ``K`` E4M3 then ``K/32``
    UE8M0 per row; ``e4m3 * 2^(s-127)`` is exact in BF16). One CTA per row."""

    threads = 128

    def __init__(self, *, k: int, wire: bool, gather: bool):
        self.k, self.wire, self.gather = int(k), bool(wire), bool(gather)
        if self.k % 32:
            raise ValueError("K must be a multiple of 32")

    def key(self) -> tuple:
        return (self.k, self.wire, self.gather)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, dst: cute.Pointer,
                 dst_rows: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, dst).launch(grid=(dst_rows, 1, 1), block=(self.threads, 1, 1),
                                                     stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, dst: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        source = i
        if const_expr(self.gather):
            source = Int32(-1)
            if i < _i32_at(Int64(meta.toint()), 2):
                source = _i32_at(Int64(pair_row.toint()), i)
        if source >= Int32(0):
            out = Int64(dst.toint()) + Int64(i) * Int64(self.k * 2)
            vectors = self.k // 8
            for it in cutlass.range_constexpr(_ceil(vectors, self.threads)):
                v = Int32(it * self.threads) + tidx
                if v < Int32(vectors):
                    if const_expr(self.wire):
                        row = Int64(src.toint()) + Int64(source) * Int64(self.k + self.k // 32)
                        lo, hi = _ld_v2_u32(row + Int64(v) * Int64(8))
                        exponent = _ld_u8(row + Int64(self.k) + Int64(v // Int32(4)))
                        scale = _u32_as_f32(exponent << Uint32(23))
                        w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, scale)
                        st_global_v4_u32(out + Int64(v) * Int64(16), w0, w1, w2, w3)
                    else:
                        w = ld_global_v4_u32(Int64(src.toint()) + (Int64(source) * Int64(self.k) + Int64(v) * Int64(8))
                                             * Int64(2))
                        st_global_v4_u32(out + Int64(v) * Int64(16), w[0], w[1], w[2], w[3])


# ---------------------------------------------------------------------------
# SwiGLU and the weighted route sum
# ---------------------------------------------------------------------------


class MoeSwiGLU:
    """``act[i] = bf16(bf16(silu(g)) * u)`` over ``gu [i, 2I]`` (gate first)
    for grouped rows ``i < meta[2]``; ``limit > 0`` clamps as DeepSeek V4
    (``u`` to [-limit, limit], ``g`` to at most limit) before the product."""

    threads = 256

    def __init__(self, *, inter: int, limit: float = 0.0):
        self.inter, self.limit = int(inter), float(limit)

    def key(self) -> tuple:
        return (self.inter, self.limit)

    @cute.jit
    def __call__(self, gu: cute.Pointer, meta: cute.Pointer, act: cute.Pointer, max_rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(gu, meta, act).launch(grid=(max_rows, _ceil(self.inter, self.threads), 1),
                                          block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gu: cute.Pointer, meta: cute.Pointer, act: cute.Pointer):
        row = Int32(cute.arch.block_idx()[0])
        col = Int32(cute.arch.block_idx()[1]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        if (row < _i32_at(Int64(meta.toint()), 2)) & (col < Int32(self.inter)):
            src = cute.make_tensor(gu, cute.make_layout((Int64(1) << Int64(40),)))
            base = Int64(row) * Int64(2 * self.inter)
            gate = Float32(src[base + Int64(col)])
            up = Float32(src[base + Int64(self.inter) + Int64(col)])
            if const_expr(self.limit > 0.0):
                gate = cutlass.select_(gate > Float32(self.limit), Float32(self.limit), gate)
                up = cutlass.select_(up > Float32(self.limit), Float32(self.limit), up)
                up = cutlass.select_(up < Float32(-self.limit), Float32(-self.limit), up)
            silu = Float32(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)).to(BFloat16))
            dst = cute.make_tensor(act, cute.make_layout((Int64(1) << Int64(40),)))
            dst[Int64(row) * Int64(self.inter) + Int64(col)] = (silu * up).to(BFloat16)


class MoeCombine:
    """``out[r] = bf16(sum_k weights[r, k] * y[pair_pos[r, k]])`` (FP32, slot
    order; invalid routes skip). One CTA per row, 8 BF16 per thread-step."""

    def __init__(self, *, hidden: int, top_k: int):
        self.hidden, self.top_k = int(hidden), int(top_k)
        if self.hidden % 8:
            raise ValueError("hidden must be a multiple of 8")
        self.threads = min(512, self.hidden // 8)

    def key(self) -> tuple:
        return (self.hidden, self.top_k)

    @cute.jit
    def __call__(self, y: cute.Pointer, pair_pos: cute.Pointer, weights: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(y, pair_pos, weights, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, y: cute.Pointer, pair_pos: cute.Pointer, weights: cute.Pointer, out: cute.Pointer):
        row = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        vectors = self.hidden // 8
        for it in cutlass.range_constexpr(_ceil(vectors, self.threads)):
            v = Int32(it * self.threads) + tidx
            if v < Int32(vectors):
                acc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
                for j in cutlass.range_constexpr(8):
                    acc[j] = Float32(0.0)
                for k in cutlass.range_constexpr(self.top_k):
                    p = _i32_at(Int64(pos_base(pair_pos)), row * Int32(self.top_k) + Int32(k))
                    if p >= Int32(0):
                        w = _ld_f32(Int64(weights.toint()) + (Int64(row) * Int64(self.top_k) + Int64(k)) * Int64(4))
                        words = ld_global_v4_u32(Int64(y.toint()) + (Int64(p) * Int64(self.hidden)
                                                                    + Int64(v) * Int64(8)) * Int64(2))
                        for j in cutlass.range_constexpr(4):
                            acc[2 * j] = acc[2 * j] + w * _bf16_lo(words[j])
                            acc[2 * j + 1] = acc[2 * j + 1] + w * _bf16_hi(words[j])
                st_global_v4_u32(Int64(out.toint()) + (Int64(row) * Int64(self.hidden) + Int64(v) * Int64(8))
                                 * Int64(2),
                                 pack_f32x2_to_bfloat2(acc[0], acc[1]), pack_f32x2_to_bfloat2(acc[2], acc[3]),
                                 pack_f32x2_to_bfloat2(acc[4], acc[5]), pack_f32x2_to_bfloat2(acc[6], acc[7]))


@cute.jit
def pos_base(pointer: cute.Pointer) -> Int64:
    return Int64(pointer.toint())


# ---------------------------------------------------------------------------
# Decode: grouped weight-streaming GEMV
# ---------------------------------------------------------------------------


class GroupedFp8Gemv:
    """Per active expert ``e`` (grid y), ``out[off_e + r] = a_r @ bf16(w_e *
    s_e)^T`` for its ``n_e <= max_rows`` grouped rows; ``a_r`` is
    ``a[pair_row[off_e + r]]`` (``gather``) or ``a[off_e + r]``; groups larger
    than ``max_rows`` run in ``max_rows`` chunks. Output
    columns ``[0, split)`` use ``w_a``/``s_a`` and ``[split, N)`` use
    ``w_b``/``s_b`` (gate then up), each ``[E, part, K]`` E4M3 with ``[E,
    part/128, K/128]`` FP32 scales; ``out`` is BF16 ``[grouped, N]``. The
    ``MmaFp8Gemv`` kernel per expert: a CTA owns ``8 * groups`` columns, its
    warps split K, the streamed weight is register double-buffered, m16n8k16
    BF16 MMAs with FP32 accumulation."""

    def __init__(self, *, n: int, k: int, experts: int, split: int | None = None, max_rows: int = 64,
                 warps: int = 4, groups: int = 4, gather: bool = False):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.split = self.n if split is None else int(split)
        self.warps, self.groups = int(warps), int(groups)
        self.cols = 8 * self.groups
        self.m_tiles = _ceil(max_rows, 16)
        self.gather = bool(gather)
        if self.n % self.cols or 128 % self.cols or self.k % (128 * self.warps) or self.split % 128:
            raise ValueError("grouped FP8 GEMV needs N % (8*groups) == 0, a 128-aligned split and "
                             "K % (128*warps) == 0")
        self.k_per_warp = self.k // self.warps
        self.blocks = self.k_per_warp // 128
        self.k_blocks = self.k // 128
        self.frags = self.m_tiles * self.groups * 4

    def key(self) -> tuple:
        return (self.n, self.k, self.experts, self.split, self.warps, self.groups, self.m_tiles, self.gather)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
                 s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer, max_groups: Int32,
                 stream: cuda.CUstream):
        self.kernel(a, pair_row, meta, w_a, s_a, w_b, s_b, out).launch(
            grid=(self.n // self.cols, max_groups, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load_block(self, dest: cute.Tensor, w_row: Int64, k_off: Int64):
        for chunk in cutlass.range_constexpr(2):
            for gi in cutlass.range_constexpr(self.groups):
                words = _ld_stream(w_row + Int64(gi * 8 * self.k) + k_off + Int64(chunk * 64))
                for t in cutlass.range_constexpr(4):
                    dest[(chunk * self.groups + gi) * 4 + t] = words[t]

    @cute.jit
    def _row_address(self, a: cute.Pointer, pair_row: cute.Pointer, grouped: Int32) -> Int64:
        source = grouped
        if const_expr(self.gather):
            source = _i32_at(Int64(pair_row.toint()), grouped)
        return Int64(a.toint()) + Int64(source) * Int64(self.k * 2)

    @cute.kernel
    def kernel(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
               s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer):
        group = Int32(cute.arch.block_idx()[1])
        m_at = Int64(meta.toint())
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.frags * 32,)))
        if group < _i32_at(m_at, 0):
            e = _i32_at(m_at + Int64(4 * (META_HEAD + 2 * self.experts)), group)
            n_e = _i32_at(m_at + Int64(4 * META_HEAD), e)
            first = _i32_at(m_at + Int64(4 * (META_HEAD + self.experts)), e)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            g = lane // Int32(4)
            j = lane % Int32(4)
            n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
            # Gate (w_a) or up (w_b) half; expert e's slab and its scale grid.
            w_base = Int64(w_a.toint())
            s_base = Int64(s_a.toint())
            part = Int64(self.split)
            local = n0
            if n0 >= Int64(self.split):
                w_base = Int64(w_b.toint())
                s_base = Int64(s_b.toint())
                part = Int64(self.n - self.split)
                local = n0 - Int64(self.split)
            w_base = w_base + Int64(e) * part * Int64(self.k)
            s_base = s_base + Int64(e) * (part // Int64(128)) * Int64(self.k_blocks * 4)
            k_begin = Int64(warp_id) * Int64(self.k_per_warp)
            w_row = w_base + (local + Int64(g)) * Int64(self.k) + k_begin + Int64(16) * Int64(j)
            s_row = s_base + (local // Int64(128)) * Int64(self.k_blocks * 4) + (k_begin // Int64(128)) * Int64(4)
            x_off = (k_begin + Int64(16) * Int64(j)) * Int64(2)
            # Groups past the compiled M tiles run in chunks (the weights stream again).
            chunks = (n_e + Int32(16 * self.m_tiles - 1)) // Int32(16 * self.m_tiles)
            for c in cutlass.range(chunks, unroll=1):
                row0 = first + c * Int32(16 * self.m_tiles)
                n_c = n_e - c * Int32(16 * self.m_tiles)
                if n_c > Int32(16 * self.m_tiles):
                    n_c = Int32(16 * self.m_tiles)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for i in cutlass.range_constexpr(self.frags):
                    acc[i] = Float32(0.0)
                live_tiles = (n_c + Int32(15)) // Int32(16)
                per_block = 2 * self.groups * 4
                cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                self._load_block(cur, w_row, Int64(0))
                s_cur = _ld_f32(s_row)
                for block in cutlass.range(self.blocks, unroll=1):
                    s_nxt = s_cur
                    if block + 1 < self.blocks:
                        self._load_block(nxt, w_row, Int64(block + 1) * Int64(128))
                        s_nxt = _ld_f32(s_row + Int64(block + 1) * Int64(4))
                    for chunk in cutlass.range_constexpr(2):
                        k_off = (Int64(block) * Int64(128) + Int64(chunk * 64)) * Int64(2)
                        bw = cute.make_rmem_tensor(cute.make_layout((8 * self.groups,), stride=(1,)), Uint32)
                        for gi in cutlass.range_constexpr(self.groups):
                            for t in cutlass.range_constexpr(4):
                                b0, b1 = _e4m3x4_scaled_bf16x2x2(cur[(chunk * self.groups + gi) * 4 + t], s_cur)
                                bw[8 * gi + 2 * t] = b0
                                bw[8 * gi + 2 * t + 1] = b1
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            if Int32(mt) < live_tiles:
                                r_lo = Int32(16 * mt) + g
                                r_hi = r_lo + Int32(8)
                                xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                                for i in cutlass.range_constexpr(16):
                                    xa[i] = Uint32(0)
                                if r_lo < n_c:
                                    at = self._row_address(a, pair_row, row0 + r_lo) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[i] = lo[i]
                                        xa[4 + i] = hi[i]
                                if r_hi < n_c:
                                    at = self._row_address(a, pair_row, row0 + r_hi) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[8 + i] = lo[i]
                                        xa[12 + i] = hi[i]
                                for gi in cutlass.range_constexpr(self.groups):
                                    f = 4 * (mt * self.groups + gi)
                                    for t in cutlass.range_constexpr(4):
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                            xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                            bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                                        acc[f] = d0
                                        acc[f + 1] = d1
                                        acc[f + 2] = d2
                                        acc[f + 3] = d3
                    for i in cutlass.range_constexpr(per_block):
                        cur[i] = nxt[i]
                    s_cur = s_nxt
                for i in cutlass.range_constexpr(self.frags):
                    partial[(warp_id * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(self.frags):
                    if Int32(i % self.warps) == warp_id:
                        total = Float32(0.0)
                        for src in cutlass.range_constexpr(self.warps):
                            total = total + partial[(Int32(src * self.frags + i)) * Int32(32) + lane]
                        mt = i // (4 * self.groups)
                        gi = (i // 4) % self.groups
                        row = Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                        col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                        if row < n_c:
                            address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.n) + col) * Int64(2)
                            target = cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)
                            target[0] = total.to(BFloat16)
                cute.arch.sync_threads()


# ---------------------------------------------------------------------------
# Prefill: grouped TMA tensor-core GEMM over FP8 weights
# ---------------------------------------------------------------------------


class GroupedFp8Gemm:
    """``out[t*64 + i, n] = a[t*64 + i] @ bf16(w_e * s_e)^T`` for every 64-row
    tile ``t < meta[1]`` of expert ``e = tile_expert[t]`` (rows past the
    group's end are not stored). ``a`` is BF16 ``[grouped, K]`` in grouped
    order (groups padded to 64 rows). Grid ``(max_tiles, N_part/128, halves)``:
    half 0 uses ``w_a``/``s_a`` into output columns ``[0, N_part)``, half 1
    ``w_b``/``s_b`` into ``[N_part, 2 N_part)`` (gate and up); ``out`` has
    ``halves * N_part`` columns. The ``TmaFp8Gemm`` pipeline: a TMA producer
    warp, E4M3 weight tiles widened to ``bf16(w * s)`` in shared memory by the
    compute warps, m16n8k16 MMAs, FP32 accumulation."""

    tile_k = 64
    tile_n = 128
    buffer_align_bytes = 1024

    def __init__(self, *, n: int, k: int, experts: int, halves: int = 1, compute_warps: int = 4,
                 num_stages: int = 4):
        self.n, self.k, self.experts, self.halves = int(n), int(k), int(experts), int(halves)
        self.num_compute_warps = int(compute_warps)
        self.compute_threads = 32 * self.num_compute_warps
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.num_compute_warps
        self.num_stages = int(num_stages)
        if self.k % 128 or self.n % 128:
            raise ValueError("grouped FP8 GEMM needs N and K multiples of 128")
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = self.n // self.tile_n
        self.k_blocks = self.k // 128
        self.chunks = self.tile_n * self.tile_k // 8 // self.compute_threads
        self.out_cols = self.n * self.halves

    def key(self) -> tuple:
        return (self.n, self.k, self.experts, self.halves, self.num_compute_warps, self.num_stages)

    def _tiled_mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.tile_k),
            cutlass.BFloat16)
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k), order=(0, 1))
        s_w = cute.make_layout((self.tile_n, self.tile_k, self.num_stages),
                               stride=(self.tile_k, 1, self.tile_n * self.tile_k))
        return s_a, s_b, s_w

    def _storage(self, s_a, s_b, s_w):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_a)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_b)], self.buffer_align_bytes],
            "sW": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(s_w)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(self, a: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer, s_a: cute.Pointer,
                 w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer, max_tiles: Int32,
                 stream: cuda.CUstream):
        a_rows = max_tiles * Int32(self.tile_m)
        a_t = cute.make_tensor(a, cute.make_layout((a_rows, self.k), stride=(self.k, 1)))
        rows_w = self.experts * self.n
        wa8 = cute.make_ptr(cutlass.Uint8, Int64(w_a.toint()), cute.AddressSpace.gmem, assumed_align=16)
        wb8 = cute.make_ptr(cutlass.Uint8, Int64(w_b.toint()), cute.AddressSpace.gmem, assumed_align=16)
        wa_t = cute.make_tensor(wa8, cute.make_layout((rows_w, self.k), stride=(self.k, 1)))
        wb_t = cute.make_tensor(wb8, cute.make_layout((rows_w, self.k), stride=(self.k, 1)))
        s_a_l, s_b_l, s_w_l = self._layouts()
        storage = self._storage(s_a_l, s_b_l, s_w_l)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a_l, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1)
        tma_wa, tma_tensor_wa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), wa_t, cute.slice_(s_w_l, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        tma_wb, tma_tensor_wb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), wb_t, cute.slice_(s_w_l, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        self.kernel(tma_tensor_a, tma_tensor_wa, tma_tensor_wb, meta, s_a, s_b, out, tma_a, tma_wa, tma_wb,
                    s_a_l, s_b_l, s_w_l, self._tiled_mma(), storage).launch(
            grid=(max_tiles, self.n_tiles, self.halves), block=[self.num_threads, 1, 1], stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight_a: cute.Tensor, weight_b: cute.Tensor, meta: cute.Pointer,
               scale_a: cute.Pointer, scale_b: cute.Pointer, out: cute.Pointer, tma_atom_a: cute.CopyAtom,
               tma_atom_wa: cute.CopyAtom, tma_atom_wb: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
               s_b_layout: cute.ComposedLayout, s_w_layout: cute.Layout, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, half = cute.arch.block_idx()
        m_at = Int64(meta.toint())
        if Int32(m_tile) < _i32_at(m_at, 1):
            e = _i32_at(m_at + Int64(4 * (META_HEAD + 3 * self.experts)), m_tile)
            group_end = _i32_at(m_at + Int64(4 * (META_HEAD + self.experts)), e) \
                + _i32_at(m_at + Int64(4 * META_HEAD), e)
            w_tile = Int32(e) * Int32(self.n_tiles) + Int32(n_tile)
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_wa)
                cpasync.prefetch_descriptor(tma_atom_wb)
            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
            s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
            s_w = storage.sW.get_tensor(s_w_layout)
            tma_bytes = (self.tile_m * 2 + self.tile_n) * self.tile_k
            load_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.num_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
                tx_count=tma_bytes,
                barrier_storage=storage.mbar_ptr.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            cute.arch.sync_threads()
            g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
            g_wa = cute.local_tile(weight_a, (self.tile_n, self.tile_k), (None, None))
            g_wb = cute.local_tile(weight_b, (self.tile_n, self.tile_k), (None, None))
            cta_layout = cute.make_layout(1)
            t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                               cute.group_modes(g_a, 0, 2))
            t_was, t_wag = cpasync.tma_partition(tma_atom_wa, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                                 cute.group_modes(g_wa, 0, 2))
            t_wbs, t_wbg = cpasync.tma_partition(tma_atom_wb, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                                 cute.group_modes(g_wb, 0, 2))
            if warp_idx < Int32(self.num_compute_warps):
                consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
                thr_mma = tiled_mma.get_slice(tidx)
                t_csa = thr_mma.partition_A(s_a)
                t_csb = thr_mma.partition_B(s_b)
                t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
                t_crb = thr_mma.make_fragment_B(t_csb)
                acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
                acc.fill(0.0)
                copy_a = cute.make_tiled_copy_A(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                    tiled_mma).get_slice(tidx)
                copy_b = cute.make_tiled_copy_B(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                    tiled_mma).get_slice(tidx)
                t_ssa = copy_a.partition_S(s_a)
                t_ssb = copy_b.partition_S(s_b)
                w_base = shared_ptr_to_u32(storage.sW.data_ptr())
                b_base = shared_ptr_to_u32(storage.sB.data_ptr())
                scale = Int64(scale_a.toint())
                if Int32(half) == Int32(1):
                    scale = Int64(scale_b.toint())
                scale_row = scale + Int64(w_tile) * Int64(self.k_blocks * 4)
                for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                    s = _ld_f32(scale_row + Int64(k_tile // Int32(2)) * Int64(4))
                    load_pipeline.consumer_wait(consumer_state)
                    stage = w_base + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
                    for i in cutlass.range_constexpr(self.chunks):
                        chunk = Int32(i * self.compute_threads) + Int32(tidx)
                        n = chunk // Int32(self.tile_k // 8)
                        c = chunk % Int32(self.tile_k // 8)
                        lo, hi = ld_shared_v2_u32(stage + n * Int32(self.tile_k) + c * Int32(8))
                        v0, v1, v2, v3 = _e4m3x8_scaled_bf16(lo, hi, s)
                        st_shared_v4_u32(b_base + n * Int32(128) + ((c ^ (n % Int32(8))) * Int32(16)), v0, v1, v2, v3)
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
                    shared_a = t_ssa[None, None, None, consumer_state.index]
                    target_a = copy_a.retile(t_cra)
                    target_b = copy_b.retile(t_crb)
                    cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                    cute.copy(copy_b, t_ssb[None, None, 0], target_b[None, None, 0])
                    for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                        if kk < cute.size(shared_a.shape[2]) - 1:
                            cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                            cute.copy(copy_b, t_ssb[None, None, kk + 1], target_b[None, None, kk + 1])
                        cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                    load_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
                coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
                col0 = Int64(half) * Int64(self.n) + Int64(n_tile) * Int64(self.tile_n)
                for index in cutlass.range_constexpr(cute.size(acc)):
                    coord = coordinates[index]
                    token = Int32(m_tile) * Int32(self.tile_m) + coord[0]
                    if token < group_end:
                        address = Int64(out.toint()) + (Int64(token) * Int64(self.out_cols) + col0
                                                        + Int64(coord[1])) * Int64(2)
                        target = cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)
                        target[0] = acc[index].to(BFloat16)
            elif warp_idx == Int32(self.producer_warp):
                producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
                for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                    load_pipeline.producer_acquire(producer_state)
                    cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile)], t_as[(None, producer_state.index)],
                              tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                    if Int32(half) == Int32(0):
                        cute.copy(tma_atom_wa, t_wag[(None, w_tile, k_tile)], t_was[(None, producer_state.index)],
                                  tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                    else:
                        cute.copy(tma_atom_wb, t_wbg[(None, w_tile, k_tile)], t_wbs[(None, producer_state.index)],
                                  tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                    load_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                load_pipeline.producer_tail(producer_state)
