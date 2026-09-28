"""CuTe DSL ports of the DSV4 producer's Triton epilogue kernels.

These kernels reproduce ``_normalize_rank_pack_kv_kernel`` (FP8 584-byte
cache format) and ``_normalize_query_rope_kernel`` from ``_impl.py`` with raw
pointer ABIs and runtime row counts so they can be linked into native AOT
programs. Arithmetic mirrors the Triton lowering (``div.full.f32`` for ``/``,
``rsqrt.approx.ftz.f32``, RN-satfinite E4M3); only the FP32 reduction order
differs, so outputs agree bitwise except for rare last-ulp BF16 flips.

Cache record (fp8, 256 rows per 149760-byte page): row ``r`` of page ``p``
stores 448 E4M3 NoPE values then 64 BF16 rotated RoPE values at
``p*149760 + r*576``, and 7 UE8M0 group scales plus one zero pad byte at
``p*149760 + 256*576 + r*8``. All page offsets are computed in 64-bit.
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

DSV4_PAGE_SIZE = 256
DSV4_PAYLOAD_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_PAGE_BYTES = 149_760
DSV4_NOPE = 448
DSV4_HEAD = 512
FP8_MAX = 448.0

_THREADS = 128  # 4 KV dims per thread; one warp covers two 64-dim FP8 groups


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _max16(value: Float32) -> Float32:
    """Max over aligned groups of 16 lanes (one 64-dim FP8 group)."""
    for shift in cutlass.range_constexpr(4):
        peer = Float32(cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
        value = fmax_f32(value, peer)
    return value


def _reduction_storage(warps: int):
    class Storage:
        pass

    Storage.__annotations__ = {
        "sums": cute.struct.Align[cute.struct.MemRange[Float32, 2 * warps], 16],
    }
    return cute.struct(Storage)


@cute.jit
def _block_sum(value: Float32, sums: cute.Tensor, slot: Int32, warps: cutlass.Constexpr):
    lane = Int32(cute.arch.thread_idx()[0]) % Int32(32)
    warp = Int32(cute.arch.thread_idx()[0]) // Int32(32)
    value = _warp_sum(value)
    if lane == Int32(0):
        sums[slot, warp] = value
    cute.arch.sync_threads()
    total = Float32(0.0)
    for src in cutlass.range_constexpr(warps):
        total = total + sums[slot, src]
    return total


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def rope_pair(a: Float32, b: Float32, cos_v: Float32, sin_v: Float32):
    """Rotate (a, b) exactly as Triton contracts ``x*cos -/+ partner*sin``."""
    return _fma_rn(a, cos_v, -(b * sin_v)), _fma_rn(b, cos_v, a * sin_v)


class DSV4RankNormPackKV:
    """Q-rank RMSNorm and KV RMSNorm + RoPE + FP8 584-byte page pack.

    ``qkv`` is the BF16 [rows, q_rank + 512] joint projection; writes
    ``q_rank_out`` BF16 [rows, q_rank] and the cache row at ``slots[row]``.
    ``positions`` and ``slots`` are int64 [rows]; ``cos_sin`` is FP32
    [positions, 64] (cos 0:32 | sin 32:64).
    """

    def __init__(self, *, q_rank: int, eps: float):
        self.q_rank = int(q_rank)
        if self.q_rank % _THREADS:
            raise ValueError("q_rank must divide by the producer CTA width")
        self.q_per_thread = self.q_rank // _THREADS
        self.eps = float(eps)
        self.warps = _THREADS // 32

    @cute.jit
    def __call__(self, qkv: cute.Pointer, q_norm: cute.Pointer, kv_norm: cute.Pointer,
                 positions: cute.Pointer, slots: cute.Pointer, cos_sin: cute.Pointer,
                 q_rank_out: cute.Pointer, cache: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        width = self.q_rank + DSV4_HEAD
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(qkv, cute.make_layout((m, width), stride=(width, 1))),
            cute.make_tensor(q_norm, cute.make_layout((self.q_rank,))),
            cute.make_tensor(kv_norm, cute.make_layout((DSV4_HEAD,))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cute.make_tensor(slots, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(q_rank_out, cute.make_layout((m, self.q_rank), stride=(self.q_rank, 1))),
            cache,
        ).launch(grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, q_norm: cute.Tensor, kv_norm: cute.Tensor,
               positions: cute.Tensor, slots: cute.Tensor, cos_sin: cute.Pointer,
               q_rank_out: cute.Tensor, cache: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((2, self.warps), stride=(self.warps, 1)))

        # Q-rank RMSNorm (weighted).
        qv = cute.make_rmem_tensor(cute.make_layout((self.q_per_thread,), stride=(1,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(self.q_per_thread):
            d = Int64(j * _THREADS) + Int64(tidx)
            qv[j] = Float32(qkv[token, d])
            square = square + qv[j] * qv[j]
        # KV RMSNorm statistics: thread owns dims 4*tidx .. 4*tidx+3.
        kv = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        kv_square = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            kv[e] = Float32(qkv[token, Int64(self.q_rank) + Int64(4) * Int64(tidx) + Int64(e)])
            kv_square = kv_square + kv[e] * kv[e]
        q_total = _block_sum(square, sums, Int32(0), self.warps)
        kv_total = _block_sum(kv_square, sums, Int32(1), self.warps)
        q_inv = rsqrt_approx_ftz_f32(div_full_f32(q_total, Float32(self.q_rank)) + Float32(self.eps))
        k_inv = rsqrt_approx_ftz_f32(div_full_f32(kv_total, Float32(DSV4_HEAD)) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.q_per_thread):
            d = Int64(j * _THREADS) + Int64(tidx)
            q_rank_out[token, d] = (qv[j] * q_inv * Float32(q_norm[d])).to(BFloat16)

        normalized = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            d = Int32(4) * tidx + Int32(e)
            normalized[e] = Float32((kv[e] * k_inv * Float32(kv_norm[d])).to(BFloat16))

        slot = Int64(slots[token])
        page = slot // Int64(DSV4_PAGE_SIZE)
        row = slot - page * Int64(DSV4_PAGE_SIZE)
        page_base = Int64(cache.toint()) + page * Int64(DSV4_PAGE_BYTES)
        data_base = page_base + row * Int64(DSV4_PAYLOAD_BYTES)
        scale_base = page_base + Int64(DSV4_PAGE_SIZE * DSV4_PAYLOAD_BYTES) + row * Int64(DSV4_SCALE_BYTES)

        # FP8 NoPE groups: 16 threads per 64-dim group (tidx < 112).
        local = fmax_f32(
            fmax_f32(fmax_f32(fabs_f32(normalized[0]), fabs_f32(normalized[1])), fabs_f32(normalized[2])),
            fabs_f32(normalized[3]))
        group_max = _max16(local)
        if tidx < Int32(DSV4_NOPE // 4):
            max_abs = fmax_f32(group_max, Float32(1.0e-4))
            scale, scale_byte = pow2_ceil_ue8m0(div_full_f32(max_abs, Float32(FP8_MAX)))
            quant = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            for e in cutlass.range_constexpr(4):
                value = div_full_f32(normalized[e], scale)
                value = fmin_f32(fmax_f32(value, Float32(-FP8_MAX)), Float32(FP8_MAX))
                quant[e] = value
            packed = cvt_f32x4_to_e4m3x4(quant[0], quant[1], quant[2], quant[3])
            words = cute.make_ptr(Uint32, data_base + Int64(4) * Int64(tidx),
                                  cute.AddressSpace.gmem, assumed_align=4)
            words[0] = packed
            if tidx % Int32(16) == Int32(0):
                scale_ptr = cute.make_ptr(Uint8, scale_base + Int64(tidx // Int32(16)),
                                          cute.AddressSpace.gmem, assumed_align=1)
                scale_ptr[0] = Uint8(scale_byte)
        else:
            rope_local = Int32(4) * tidx - Int32(DSV4_NOPE)  # 0, 4, ..., 60
            position = Int64(positions[token])
            cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                               cute.AddressSpace.gmem, assumed_align=4)
            out = cute.make_ptr(BFloat16, data_base + Int64(DSV4_NOPE) + Int64(2) * Int64(rope_local),
                                cute.AddressSpace.gmem, assumed_align=2)
            for pair in cutlass.range_constexpr(2):
                index = rope_local // Int32(2) + Int32(pair)
                cos_v = Float32(cs[index])
                sin_v = Float32(cs[Int32(32) + index])
                even, odd = rope_pair(normalized[2 * pair], normalized[2 * pair + 1], cos_v, sin_v)
                out[2 * pair] = even.to(BFloat16)
                out[2 * pair + 1] = odd.to(BFloat16)
            if tidx == Int32(DSV4_NOPE // 4):
                pad = cute.make_ptr(Uint8, scale_base + Int64(7), cute.AddressSpace.gmem, assumed_align=1)
                pad[0] = Uint8(0)


class DSV4QueryNormRope:
    """Per-head RMSNorm (no weight) + partial RoPE in place on query [rows, heads, 512]."""

    def __init__(self, *, heads: int, eps: float):
        self.heads = int(heads)
        self.eps = float(eps)
        self.warps = _THREADS // 32

    @cute.jit
    def __call__(self, query: cute.Pointer, positions: cute.Pointer, cos_sin: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(query, cute.make_layout(
                (m, self.heads, DSV4_HEAD), stride=(self.heads * DSV4_HEAD, DSV4_HEAD, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cos_sin,
        ).launch(grid=(rows, self.heads, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, query: cute.Tensor, positions: cute.Tensor, cos_sin: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((2, self.warps), stride=(self.warps, 1)))
        values = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        square = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            values[e] = Float32(query[token, head, Int64(4) * Int64(tidx) + Int64(e)])
            square = square + values[e] * values[e]
        total = _block_sum(square, sums, Int32(0), self.warps)
        inv = rsqrt_approx_ftz_f32(div_full_f32(total, Float32(DSV4_HEAD)) + Float32(self.eps))
        normalized = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            normalized[e] = Float32((values[e] * inv).to(BFloat16))
        if tidx >= Int32(DSV4_NOPE // 4):
            rope_local = Int32(4) * tidx - Int32(DSV4_NOPE)
            position = Int64(positions[token])
            cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                               cute.AddressSpace.gmem, assumed_align=4)
            for pair in cutlass.range_constexpr(2):
                index = rope_local // Int32(2) + Int32(pair)
                cos_v = Float32(cs[index])
                sin_v = Float32(cs[Int32(32) + index])
                even, odd = rope_pair(normalized[2 * pair], normalized[2 * pair + 1], cos_v, sin_v)
                normalized[2 * pair] = even
                normalized[2 * pair + 1] = odd
        for e in cutlass.range_constexpr(4):
            query[token, head, Int64(4) * Int64(tidx) + Int64(e)] = normalized[e].to(BFloat16)


_INDEX_HEAD_DIM = 128
_INDEX_NOPE = 64
_FP4_TINY = 1.1754943508222875e-38


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


class DSV4IndexerQueryPost:
    """C4 index query post-processing (port of ``_indexer_query_post_kernel``).

    Per (row, head): partial RoPE on dims 64:128 of the raw BF16 projection,
    BF16 round, 128-point Walsh-Hadamard * 128^-0.5 (BF16), E2M1 QAT with a
    power-of-two scale per 32 values (BF16 dequantized), stored as FP8 E4M3
    ``query [rows, 64, 128]``; learned head weight ``bf16(raw_w * scale)`` as
    FP32 ``head_weights [rows, 64]``. One warp per (row, head), four dims per
    lane.
    """

    def __init__(self, *, heads: int = 64, weight_scale: float):
        self.heads = int(heads)
        self.weight_scale = float(weight_scale)

    @cute.jit
    def __call__(self, raw_query: cute.Pointer, raw_weights: cute.Pointer,
                 positions: cute.Pointer, cos_sin: cute.Pointer, query: cute.Pointer,
                 head_weights: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        h, d = self.heads, _INDEX_HEAD_DIM
        self.kernel(
            cute.make_tensor(raw_query, cute.make_layout((m, h, d), stride=(h * d, d, 1))),
            cute.make_tensor(raw_weights, cute.make_layout((m, h), stride=(h, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cos_sin,
            query,
            cute.make_tensor(head_weights, cute.make_layout((m, h), stride=(h, 1))),
        ).launch(grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, raw_query: cute.Tensor, raw_weights: cute.Tensor, positions: cute.Tensor,
               cos_sin: cute.Pointer, query: cute.Pointer, head_weights: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        v = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            v[e] = Float32(raw_query[token, head, Int64(4) * Int64(lane) + Int64(e)])
        if lane >= Int32(_INDEX_NOPE // 4):
            rope_local = Int32(4) * lane - Int32(_INDEX_NOPE)
            position = Int64(positions[token])
            cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                               cute.AddressSpace.gmem, assumed_align=4)
            for pair in cutlass.range_constexpr(2):
                index = rope_local // Int32(2) + Int32(pair)
                cos_v = Float32(cs[index])
                sin_v = Float32(cs[Int32(32) + index])
                even, odd = rope_pair(v[2 * pair], v[2 * pair + 1], cos_v, sin_v)
                v[2 * pair] = even
                v[2 * pair + 1] = odd
        for e in cutlass.range_constexpr(4):
            v[e] = Float32(v[e].to(BFloat16))
        # Walsh-Hadamard butterflies: low = a + b, high = a - b at stride W.
        for pair in cutlass.range_constexpr(2):
            a, b = v[2 * pair], v[2 * pair + 1]
            v[2 * pair] = a + b
            v[2 * pair + 1] = a - b
        for e in cutlass.range_constexpr(2):
            a, b = v[e], v[e + 2]
            v[e] = a + b
            v[e + 2] = a - b
        for stage in cutlass.range_constexpr(5):
            offset = 1 << stage  # lane stride for W = 4 << stage
            high = (lane & Int32(offset)) != Int32(0)
            for e in cutlass.range_constexpr(4):
                partner = Float32(cute.arch.shuffle_sync_bfly(v[e], offset=offset))
                if high:
                    v[e] = partner - v[e]
                else:
                    v[e] = v[e] + partner
        for e in cutlass.range_constexpr(4):
            v[e] = Float32((v[e] * Float32(0.08838834764831845)).to(BFloat16))
        block_max = fmax_f32(fmax_f32(fabs_f32(v[0]), fabs_f32(v[1])),
                             fmax_f32(fabs_f32(v[2]), fabs_f32(v[3])))
        for shift in cutlass.range_constexpr(3):
            block_max = fmax_f32(block_max, Float32(cute.arch.shuffle_sync_bfly(block_max, offset=1 << shift)))
        raw_scale = div_full_f32(fmax_f32(block_max, Float32(6.0 * _FP4_TINY)), Float32(6.0))
        fp4_scale, _ = pow2_ceil_ue8m0(raw_scale)
        out = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for e in cutlass.range_constexpr(4):
            magnitude = fmin_f32(div_full_f32(fabs_f32(v[e]), fp4_scale), Float32(6.0))
            q = _fp4_magnitude(magnitude)
            if v[e] < Float32(0.0):
                q = -q
            out[e] = Float32((q * fp4_scale).to(BFloat16))
        packed = cvt_f32x4_to_e4m3x4(out[0], out[1], out[2], out[3])
        words = cute.make_ptr(
            Uint32,
            Int64(query.toint()) + (token * Int64(self.heads) + head) * Int64(_INDEX_HEAD_DIM)
            + Int64(4) * Int64(lane),
            cute.AddressSpace.gmem, assumed_align=4)
        words[0] = packed
        if lane == Int32(0):
            raw_weight = Float32(raw_weights[token, head])
            head_weights[token, head] = Float32((raw_weight * Float32(self.weight_scale)).to(BFloat16))


__all__ = ["DSV4IndexerQueryPost", "DSV4QueryNormRope", "DSV4RankNormPackKV"]
