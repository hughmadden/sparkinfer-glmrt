"""CuTe DSL ports of the DSV4 compressor's Triton kernels (FP8 cache format).

One pooling kernel (:class:`DSV4CompressorPool`) covers the three serving
modes of ``_impl.py`` for either head:

* ``decode``        ``_update_pool_pack_{main,index}_kernel``: one row per
  sequence, writes the row into the rolling state, pools when it completes a
  group;
* ``prefill``       ``_prefill_pool_pack_{main,index}_kernel``: one CTA per
  planned group, sources every row from the joint projection;
* ``continuation``  ``_continuation_pool_pack_{main,index}_kernel``: rows at
  or after the chunk start come from the projection, earlier rows from state.

:class:`DSV4CompressorFinalize` ports ``_prefill_finalize_state_kernel`` and
``_continuation_finalize_state_kernel`` (one CTA per (sequence, state row)).

The main head (512 dims) pools with softmax over the group's rows (C4 adds
the previous group's first half, "overlap"), RMSNorm * norm, partial RoPE at
the group's RoPE position and the FP8 584-byte record. The C4 index head (128
dims) additionally applies the 128-point Hadamard, E2M1 QAT and FP8 with a
per-row FP32 scale (8448-byte pages of 64 rows). Arithmetic mirrors Triton's
lowering (``ex2.approx(x*log2e)`` for ``exp``, FMA accumulation,
``div.full.f32``, ``rsqrt.approx.ftz``); only the RMS reduction order differs.
Every thread owns four consecutive head dimensions for both the state write
and the pooling, so no thread reads another thread's global writes.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4,
    div_full_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    pow2_ceil_ue8m0,
    rsqrt_approx_ftz_f32,
)
from b12x.attention.dsv4_producer._cute import _fma_rn, rope_pair

MAIN_HEAD = 512
MAIN_NOPE = 448
INDEX_HEAD = 128
INDEX_NOPE = 64
INDEX_PAGE_ROWS = 64
INDEX_PAGE_BYTES = 8448
SOURCE_PAGE = 256
PAYLOAD_BYTES = 576
SCALE_BYTES = 8
FP8_MAX = 448.0
_LOG2E = 1.4426950408889634
_NEG_INF = float("-inf")
_FP4_TINY = 1.1754943508222875e-38


def main_page_bytes(ratio: int) -> int:
    rows = SOURCE_PAGE // ratio
    return (rows * (PAYLOAD_BYTES + SCALE_BYTES) + PAYLOAD_BYTES - 1) // PAYLOAD_BYTES * PAYLOAD_BYTES


@dsl_user_op
def _ex2_approx(x, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)], "ex2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _exp(x: Float32) -> Float32:
    """Triton's ``tl.exp``: ex2.approx.f32(x * log2(e))."""
    return _ex2_approx(x * Float32(_LOG2E))


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _fp4_magnitude(m: Float32) -> Float32:
    result = Float32(6.0)
    if m < Float32(5.0):
        result = Float32(4.0)
    if m < Float32(3.5):
        result = Float32(3.0)
    if m < Float32(2.5):
        result = Float32(2.0)
    if m < Float32(1.75):
        result = Float32(1.5)
    if m < Float32(1.25):
        result = Float32(1.0)
    if m < Float32(0.75):
        result = Float32(0.5)
    if m < Float32(0.25):
        result = Float32(0.0)
    return result


class _Head:
    """Static head geometry shared by pooling and finalize kernels."""

    def __init__(self, *, ratio: int, index: bool):
        self.ratio = int(ratio)
        if self.ratio not in (4, 128):
            raise ValueError("DSV4 compressor ratio must be 4 or 128")
        self.index = bool(index)
        if self.index and self.ratio != 4:
            raise ValueError("only C4 layers carry the index compressor")
        self.overlap = self.ratio == 4
        self.coefficient = 2 if self.overlap else 1
        self.head = INDEX_HEAD if self.index else MAIN_HEAD
        self.nope = INDEX_NOPE if self.index else MAIN_NOPE
        self.width = self.coefficient * self.head           # projected width
        main_width = (2 if self.overlap else 1) * MAIN_HEAD
        self.offset = 2 * main_width if self.index else 0    # column in the joint projection
        self.joint_width = 2 * (main_width + (2 * INDEX_HEAD if self.ratio == 4 else 0))
        self.state_rows = 2 * self.coefficient * self.ratio
        self.threads = self.head // 4
        self.warps = self.threads // 32
        self.page_rows = INDEX_PAGE_ROWS if self.index else SOURCE_PAGE // self.ratio
        self.page_bytes = INDEX_PAGE_BYTES if self.index else main_page_bytes(self.ratio)


def _storage(warps: int):
    class Storage:
        pass

    Storage.__annotations__ = {
        "sums": cute.struct.Align[cute.struct.MemRange[Float32, max(warps, 1)], 16],
    }
    return cute.struct(Storage)


class DSV4CompressorPool(_Head):
    """Pool + normalize + RoPE (+ index QAT) + page pack for one head.

    ``mode`` is ``decode``, ``prefill`` or ``continuation``; see the module
    docstring. Pointers not used by a mode are ignored.
    """

    def __init__(self, *, ratio: int, index: bool, mode: str, eps: float):
        super().__init__(ratio=ratio, index=index)
        if mode not in ("decode", "prefill", "continuation"):
            raise ValueError(f"unknown compressor mode {mode!r}")
        self.mode = mode
        self.eps = float(eps)

    # ------------------------------------------------------------------ launch
    @cute.jit
    def __call__(self, projection: cute.Pointer, cos_sin: cute.Pointer, ape: cute.Pointer,
                 norm: cute.Pointer, cache: cute.Pointer, kv_state: cute.Pointer,
                 score_state: cute.Pointer, a0: cute.Pointer, a1: cute.Pointer,
                 a2: cute.Pointer, a3: cute.Pointer, a4: cute.Pointer, a5: cute.Pointer,
                 a6: cute.Pointer, a7: cute.Pointer, units: Int32, stream: cuda.CUstream):
        """int32 metadata pointers ``a0..a7`` by mode (unused ones are ignored):

        * decode:       positions, sequence_ids, compressed_slots
        * prefill:      active_groups, group_source_starts, group_rope_positions,
                        compressed_slots
        * continuation: active_groups, group_sequence_slots, group_source_positions,
                        group_rope_positions, compressed_slots, sequence_offsets,
                        sequence_start_positions, state_sequence_ids

        ``units`` is rows (decode) or the group capacity (prefill/continuation).
        """
        self.kernel(projection, cos_sin, ape, norm, cache, kv_state, score_state,
                    a0, a1, a2, a3, a4, a5, a6, a7).launch(
            grid=(units, 1, 1), block=(self.threads, 1, 1), stream=stream)

    # ------------------------------------------------------------------ loads
    @cute.jit
    def _proj(self, projection: cute.Pointer, row: Int64, column: Int32) -> Float32:
        address = Int64(projection.toint()) + (row * Int64(self.joint_width)
                                               + Int64(self.offset) + Int64(column)) * Int64(2)
        return Float32(cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)[0])

    @cute.jit
    def _state(self, state: cute.Pointer, sequence: Int64, position: Int32, column: Int32) -> Float32:
        # Floor modulo: the first decode group's absent previous rows map to
        # the -inf rows the prefill finalizer leaves (Triton's C remainder
        # would address the preceding sequence's rows instead).
        row = Int64((position % Int32(self.state_rows) + Int32(self.state_rows)) % Int32(self.state_rows))
        address = Int64(state.toint()) + ((sequence * Int64(self.state_rows) + row)
                                          * Int64(self.width) + Int64(column)) * Int64(4)
        return Float32(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)[0])

    @cute.jit
    def _state_ptr(self, state: cute.Pointer, sequence: Int64, position: Int32, column: Int32):
        row = Int64(position % Int32(self.state_rows))
        address = Int64(state.toint()) + ((sequence * Int64(self.state_rows) + row)
                                          * Int64(self.width) + Int64(column)) * Int64(4)
        return cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)

    @cute.jit
    def _ape(self, ape: cute.Pointer, lane: Int32, column: Int32) -> Float32:
        address = Int64(ape.toint()) + (Int64(lane) * Int64(self.width) + Int64(column)) * Int64(4)
        return Float32(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)[0])

    @cute.jit
    def _source(self, projection: cute.Pointer, state: cute.Pointer, ape: cute.Pointer,
                score: cutlass.Constexpr, position: Int32, column: Int32, sequence: Int64,
                chunk_offset: Int32, chunk_start: Int32, row: Int32, lane: Int32) -> Float32:
        """Value (``score`` False) or score+ape (True) of logical ``position``."""
        result = Float32(0.0)
        if const_expr(self.mode == "decode"):
            result = self._state(state, sequence, position, column)
        elif const_expr(self.mode == "prefill"):
            # ``row`` is the projection row; the ape lane is the in-group index.
            if const_expr(score):
                result = self._proj(projection, Int64(row), Int32(self.width) + column) \
                    + self._ape(ape, lane, column)
            else:
                result = self._proj(projection, Int64(row), column)
        else:
            if position >= chunk_start:
                prow = Int64(chunk_offset) + Int64(position - chunk_start)
                if const_expr(score):
                    result = self._proj(projection, prow, Int32(self.width) + column) \
                        + self._ape(ape, position % Int32(self.ratio), column)
                else:
                    result = self._proj(projection, prow, column)
            else:
                result = self._state(state, sequence, position, column)
        return result

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(self, projection: cute.Pointer, cos_sin: cute.Pointer, ape: cute.Pointer,
               norm: cute.Pointer, cache: cute.Pointer, kv_state: cute.Pointer,
               score_state: cute.Pointer, a0: cute.Pointer, a1: cute.Pointer,
               a2: cute.Pointer, a3: cute.Pointer, a4: cute.Pointer, a5: cute.Pointer,
               a6: cute.Pointer, a7: cute.Pointer):
        unit = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_storage(self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((max(self.warps, 1),), stride=(1,)))

        active = Int32(1)
        emits = Int32(1)
        sequence = Int64(0)
        group_start = Int32(0)
        rope_position = Int32(0)
        slot = Int64(0)
        chunk_offset = Int32(0)
        chunk_start = Int32(0)
        source_start = Int32(0)
        prev_valid = Int32(1)
        if const_expr(self.mode == "decode"):
            position = _ld(a0, unit)
            sequence = Int64(_ld(a1, unit))
            slot = Int64(_ld(a2, unit))
            lane = position % Int32(self.ratio)
            # Record this token (value and score + ape) in the rolling state.
            for e in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(self.coefficient):
                    column = Int32(half * self.head) + Int32(4) * tidx + Int32(e)
                    value = self._proj(projection, Int64(unit), column)
                    score = self._proj(projection, Int64(unit), Int32(self.width) + column) \
                        + self._ape(ape, lane, column)
                    self._state_ptr(kv_state, sequence, position, column)[0] = value
                    self._state_ptr(score_state, sequence, position, column)[0] = score
            emits = Int32(lane == Int32(self.ratio - 1))
            group_start = position + Int32(1 - self.ratio)
            rope_position = group_start
        else:
            active = Int32(unit < _ld(a0, Int32(0)))
            if active != Int32(0):
                if const_expr(self.mode == "prefill"):
                    source_start = _ld(a1, unit)
                    rope_position = _ld(a2, unit)
                    slot = Int64(_ld(a3, unit))
                    group_start = rope_position
                    prev_valid = Int32(rope_position > Int32(0))
                else:
                    seq_slot = _ld(a1, unit)
                    group_start = _ld(a2, unit)
                    rope_position = _ld(a3, unit)
                    slot = Int64(_ld(a4, unit))
                    chunk_offset = _ld(a5, seq_slot)
                    chunk_start = _ld(a6, seq_slot)
                    sequence = Int64(_ld(a7, seq_slot))
                    prev_valid = Int32(group_start > Int32(0))
        if (active != Int32(0)) & (emits != Int32(0)):
            self._emit(projection, cos_sin, ape, norm, cache, kv_state, score_state, sums, tidx,
                       sequence, group_start, rope_position, slot, chunk_offset, chunk_start,
                       source_start, prev_valid)

    @cute.jit
    def _emit(self, projection: cute.Pointer, cos_sin: cute.Pointer, ape: cute.Pointer,
              norm: cute.Pointer, cache: cute.Pointer, kv_state: cute.Pointer,
              score_state: cute.Pointer, sums: cute.Tensor, tidx: Int32, sequence: Int64,
              group_start: Int32, rope_position: Int32, slot: Int64, chunk_offset: Int32,
              chunk_start: Int32, source_start: Int32, prev_valid: Int32):
        pooled = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            d = Int32(4) * tidx + Int32(e)
            current_col = Int32(self.head) + d if const_expr(self.overlap) else d
            row_max = Float32(_NEG_INF)
            for r in cutlass.range(self.ratio, unroll=4 if self.ratio == 4 else 1):
                cur_pos = group_start + r
                prev_pos = group_start - Int32(self.ratio) + r
                cur_row = source_start + r
                prev_row = source_start - Int32(self.ratio) + r
                cur = self._source(projection, score_state, ape, True, cur_pos, current_col,
                                   sequence, chunk_offset, chunk_start, cur_row, r)
                if const_expr(self.overlap):
                    if const_expr(self.mode == "decode"):
                        prev = self._source(projection, score_state, ape, True, prev_pos, d,
                                            sequence, chunk_offset, chunk_start, prev_row, r)
                        row_max = fmax_f32(row_max, prev)
                        row_max = fmax_f32(row_max, cur)
                    else:
                        row_max = fmax_f32(row_max, cur)
                        if prev_valid != Int32(0):
                            prev = self._source(projection, score_state, ape, True, prev_pos, d,
                                                sequence, chunk_offset, chunk_start, prev_row, r)
                            row_max = fmax_f32(row_max, prev)
                else:
                    row_max = fmax_f32(row_max, cur)
            numerator = Float32(0.0)
            denominator = Float32(0.0)
            for r in cutlass.range(self.ratio, unroll=4 if self.ratio == 4 else 1):
                cur_pos = group_start + r
                prev_pos = group_start - Int32(self.ratio) + r
                cur_row = source_start + r
                prev_row = source_start - Int32(self.ratio) + r
                cur_s = self._source(projection, score_state, ape, True, cur_pos, current_col,
                                     sequence, chunk_offset, chunk_start, cur_row, r)
                cur_v = self._source(projection, kv_state, ape, False, cur_pos, current_col,
                                     sequence, chunk_offset, chunk_start, cur_row, r)
                cur_w = _exp(cur_s - row_max)
                if const_expr(self.overlap and self.mode == "decode"):
                    prev_s = self._source(projection, score_state, ape, True, prev_pos, d,
                                          sequence, chunk_offset, chunk_start, prev_row, r)
                    prev_v = self._source(projection, kv_state, ape, False, prev_pos, d,
                                          sequence, chunk_offset, chunk_start, prev_row, r)
                    prev_w = _exp(prev_s - row_max)
                    numerator = _fma_rn(prev_w, prev_v, numerator)
                    numerator = _fma_rn(cur_w, cur_v, numerator)
                    denominator = denominator + (prev_w + cur_w)
                else:
                    numerator = _fma_rn(cur_w, cur_v, numerator)
                    denominator = denominator + cur_w
                    if const_expr(self.overlap):
                        if prev_valid != Int32(0):
                            prev_s = self._source(projection, score_state, ape, True, prev_pos, d,
                                                  sequence, chunk_offset, chunk_start, prev_row, r)
                            prev_v = self._source(projection, kv_state, ape, False, prev_pos, d,
                                                  sequence, chunk_offset, chunk_start, prev_row, r)
                            prev_w = _exp(prev_s - row_max)
                            numerator = _fma_rn(prev_w, prev_v, numerator)
                            denominator = denominator + prev_w
            pooled[e] = Float32(div_full_f32(numerator, denominator).to(BFloat16))

        square = pooled[0] * pooled[0] + pooled[1] * pooled[1] + pooled[2] * pooled[2] + pooled[3] * pooled[3]
        total = _warp_sum(square)
        if const_expr(self.warps > 1):
            if tidx % Int32(32) == Int32(0):
                sums[tidx // Int32(32)] = total
            cute.arch.sync_threads()
            total = Float32(0.0)
            for w in cutlass.range_constexpr(self.warps):
                total = total + sums[w]
        inv = rsqrt_approx_ftz_f32(div_full_f32(total, Float32(self.head)) + Float32(self.eps))
        g = cute.make_tensor(norm, cute.make_layout((self.head,)))
        out = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            d = Int32(4) * tidx + Int32(e)
            out[e] = Float32((pooled[e] * inv * Float32(g[d])).to(BFloat16))
        if tidx >= Int32(self.nope // 4):
            rope_local = Int32(4) * tidx - Int32(self.nope)
            cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + Int64(rope_position) * Int64(256),
                               cute.AddressSpace.gmem, assumed_align=4)
            for pair in cutlass.range_constexpr(2):
                index = rope_local // Int32(2) + Int32(pair)
                even, odd = rope_pair(out[2 * pair], out[2 * pair + 1], Float32(cs[index]),
                                      Float32(cs[Int32(32) + index]))
                out[2 * pair] = even
                out[2 * pair + 1] = odd
        page = slot // Int64(self.page_rows)
        page_row = slot - page * Int64(self.page_rows)
        page_base = Int64(cache.toint()) + page * Int64(self.page_bytes)
        if const_expr(self.index):
            self._store_index(out, tidx, page_base, page_row)
        else:
            self._store_main(out, tidx, page_base, page_row)

    @cute.jit
    def _store_main(self, out: cute.Tensor, tidx: Int32, page_base: Int64, page_row: Int64):
        data_base = page_base + page_row * Int64(PAYLOAD_BYTES)
        scale_base = page_base + Int64(self.page_rows * PAYLOAD_BYTES) + page_row * Int64(SCALE_BYTES)
        local = fmax_f32(fmax_f32(fabs_f32(out[0]), fabs_f32(out[1])),
                         fmax_f32(fabs_f32(out[2]), fabs_f32(out[3])))
        for shift in cutlass.range_constexpr(4):
            local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1 << shift)))
        if tidx < Int32(MAIN_NOPE // 4):
            max_abs = fmax_f32(local, Float32(1.0e-4))
            scale, scale_byte = pow2_ceil_ue8m0(div_full_f32(max_abs, Float32(FP8_MAX)))
            q = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            for e in cutlass.range_constexpr(4):
                q[e] = fmin_f32(fmax_f32(div_full_f32(out[e], scale), Float32(-FP8_MAX)), Float32(FP8_MAX))
            cute.make_ptr(Uint32, data_base + Int64(4) * Int64(tidx), cute.AddressSpace.gmem,
                          assumed_align=4)[0] = cvt_f32x4_to_e4m3x4(q[0], q[1], q[2], q[3])
            if tidx % Int32(16) == Int32(0):
                cute.make_ptr(Uint8, scale_base + Int64(tidx // Int32(16)), cute.AddressSpace.gmem,
                              assumed_align=1)[0] = Uint8(scale_byte)
        else:
            rope = cute.make_ptr(BFloat16, data_base + Int64(2) * Int64(Int32(4) * tidx) - Int64(MAIN_NOPE),
                                 cute.AddressSpace.gmem, assumed_align=2)
            for e in cutlass.range_constexpr(4):
                rope[e] = out[e].to(BFloat16)
            if tidx == Int32(MAIN_NOPE // 4):
                cute.make_ptr(Uint8, scale_base + Int64(7), cute.AddressSpace.gmem, assumed_align=1)[0] = Uint8(0)

    @cute.jit
    def _store_index(self, out: cute.Tensor, lane: Int32, page_base: Int64, page_row: Int64):
        v = out
        for pair in cutlass.range_constexpr(2):
            a, b = v[2 * pair], v[2 * pair + 1]
            v[2 * pair] = a + b
            v[2 * pair + 1] = a - b
        for e in cutlass.range_constexpr(2):
            a, b = v[e], v[e + 2]
            v[e] = a + b
            v[e + 2] = a - b
        for stage in cutlass.range_constexpr(5):
            offset = 1 << stage
            high = (lane & Int32(offset)) != Int32(0)
            for e in cutlass.range_constexpr(4):
                partner = Float32(cute.arch.shuffle_sync_bfly(v[e], offset=offset))
                if high:
                    v[e] = partner - v[e]
                else:
                    v[e] = v[e] + partner
        for e in cutlass.range_constexpr(4):
            v[e] = Float32((v[e] * Float32(0.08838834764831845)).to(BFloat16))
        block_max = fmax_f32(fmax_f32(fabs_f32(v[0]), fabs_f32(v[1])), fmax_f32(fabs_f32(v[2]), fabs_f32(v[3])))
        for shift in cutlass.range_constexpr(3):
            block_max = fmax_f32(block_max, Float32(cute.arch.shuffle_sync_bfly(block_max, offset=1 << shift)))
        fp4_scale, _ = pow2_ceil_ue8m0(div_full_f32(fmax_f32(block_max, Float32(6.0 * _FP4_TINY)), Float32(6.0)))
        fp4 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            q = _fp4_magnitude(fmin_f32(div_full_f32(fabs_f32(v[e]), fp4_scale), Float32(6.0)))
            if v[e] < Float32(0.0):
                q = Float32(0.0) - q  # Triton lowers unary minus to 0 - x (+0 for +0)
            fp4[e] = Float32((q * fp4_scale).to(BFloat16))
        row_max = fmax_f32(fmax_f32(fabs_f32(fp4[0]), fabs_f32(fp4[1])), fmax_f32(fabs_f32(fp4[2]), fabs_f32(fp4[3])))
        for shift in cutlass.range_constexpr(5):
            row_max = fmax_f32(row_max, Float32(cute.arch.shuffle_sync_bfly(row_max, offset=1 << shift)))
        cache_scale = div_full_f32(row_max, Float32(FP8_MAX))
        if cache_scale <= Float32(0.0):
            cache_scale = Float32(1.0)
        q8 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            q8[e] = fmin_f32(fmax_f32(div_full_f32(fp4[e], cache_scale), Float32(-FP8_MAX)), Float32(FP8_MAX))
        cute.make_ptr(Uint32, page_base + page_row * Int64(INDEX_HEAD) + Int64(4) * Int64(lane),
                      cute.AddressSpace.gmem, assumed_align=4)[0] = cvt_f32x4_to_e4m3x4(q8[0], q8[1], q8[2], q8[3])
        if lane == Int32(0):
            cute.make_ptr(Float32, page_base + Int64(INDEX_PAGE_ROWS * INDEX_HEAD) + Int64(4) * page_row,
                          cute.AddressSpace.gmem, assumed_align=4)[0] = cache_scale


@cute.jit
def _ld(array: cute.Pointer, index: Int32) -> Int32:
    """Element ``index`` of an int32 device array."""
    return Int32(cute.make_ptr(Int32, Int64(array.toint()) + Int64(4) * Int64(index),
                               cute.AddressSpace.gmem, assumed_align=4)[0])


class DSV4CompressorFinalize(_Head):
    """Write the terminal rolling state for prefill / continuation chunks.

    Grid (sequence capacity, state_rows); CTA (s, r) rewrites state row ``r``
    of ``state_sequence_ids[s]`` with the last source position congruent to
    ``r`` (value = projected value, score = projected score + ape), or
    (0, -inf) when no such position exists yet.
    """

    def __init__(self, *, ratio: int, index: bool, mode: str):
        super().__init__(ratio=ratio, index=index)
        if mode not in ("prefill", "continuation"):
            raise ValueError("finalize runs for prefill and continuation only")
        self.mode = mode
        self.finalize_threads = self.width // 4

    @cute.jit
    def __call__(self, projection: cute.Pointer, ape: cute.Pointer, kv_state: cute.Pointer,
                 score_state: cute.Pointer, active_sequences: cute.Pointer,
                 sequence_offsets: cute.Pointer, state_sequence_ids: cute.Pointer,
                 sequence_start_positions: cute.Pointer, sequences: Int32,
                 stream: cuda.CUstream):
        """``sequence_start_positions`` is ignored for prefill (chunks start at 0)."""
        self.kernel(projection, ape, kv_state, score_state, active_sequences, sequence_offsets,
                    state_sequence_ids, sequence_start_positions).launch(
            grid=(sequences, self.state_rows, 1), block=(self.finalize_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, projection: cute.Pointer, ape: cute.Pointer, kv_state: cute.Pointer,
               score_state: cute.Pointer, active_sequences: cute.Pointer,
               sequence_offsets: cute.Pointer, state_sequence_ids: cute.Pointer,
               sequence_start_positions: cute.Pointer):
        seq_slot = Int32(cute.arch.block_idx()[0])
        state_row = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        if seq_slot < _ld(active_sequences, Int32(0)):
            chunk_offset = _ld(sequence_offsets, seq_slot)
            chunk_end = _ld(sequence_offsets, seq_slot + Int32(1))
            sequence = Int64(_ld(state_sequence_ids, seq_slot))
            chunk_start = Int32(0)
            if const_expr(self.mode == "continuation"):
                chunk_start = _ld(sequence_start_positions, seq_slot)
            last = chunk_start + chunk_end - chunk_offset - Int32(1)
            cycles = (last - state_row) // Int32(self.state_rows)
            if last < state_row:
                cycles = Int32(0)
            position = state_row + cycles * Int32(self.state_rows)
            fill = position <= last
            lane = position % Int32(self.ratio)
            for e in cutlass.range_constexpr(4):
                column = Int32(4) * tidx + Int32(e)
                value = Float32(0.0)
                score = Float32(_NEG_INF)
                if fill:
                    if position >= chunk_start:
                        prow = Int64(chunk_offset) + Int64(position - chunk_start)
                        value = _proj_value(projection, prow, self.joint_width, self.offset, column)
                        score = _proj_value(projection, prow, self.joint_width, self.offset,
                                            Int32(self.width) + column) + _ape_value(ape, lane, self.width, column)
                    else:
                        value = _state_value(kv_state, sequence, position, self.state_rows, self.width, column)
                        score = _state_value(score_state, sequence, position, self.state_rows, self.width, column)
                _state_store(kv_state, sequence, state_row, self.state_rows, self.width, column, value)
                _state_store(score_state, sequence, state_row, self.state_rows, self.width, column, score)


@cute.jit
def _proj_value(projection: cute.Pointer, row: Int64, joint_width: cutlass.Constexpr,
                offset: cutlass.Constexpr, column: Int32) -> Float32:
    address = Int64(projection.toint()) + (row * Int64(joint_width) + Int64(offset) + Int64(column)) * Int64(2)
    return Float32(cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)[0])


@cute.jit
def _ape_value(ape: cute.Pointer, lane: Int32, width: cutlass.Constexpr, column: Int32) -> Float32:
    address = Int64(ape.toint()) + (Int64(lane) * Int64(width) + Int64(column)) * Int64(4)
    return Float32(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)[0])


@cute.jit
def _state_value(state: cute.Pointer, sequence: Int64, position: Int32, rows: cutlass.Constexpr,
                 width: cutlass.Constexpr, column: Int32) -> Float32:
    row = Int64(position % Int32(rows))
    address = Int64(state.toint()) + ((sequence * Int64(rows) + row) * Int64(width) + Int64(column)) * Int64(4)
    return Float32(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)[0])


@cute.jit
def _state_store(state: cute.Pointer, sequence: Int64, row: Int32, rows: cutlass.Constexpr,
                 width: cutlass.Constexpr, column: Int32, value: Float32):
    address = Int64(state.toint()) + ((sequence * Int64(rows) + Int64(row)) * Int64(width) + Int64(column)) * Int64(4)
    cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=4)[0] = value


__all__ = ["DSV4CompressorFinalize", "DSV4CompressorPool", "main_page_bytes"]
