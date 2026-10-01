"""CuTe DSL kernels composed into the GLM 5.x (``glm_*``) AOT programs.

Every kernel takes raw pointers and a runtime ``rows`` so it can be linked
into a native program. Arithmetic follows the transformers reference module
(``modeling_glm_moe_dsa``) rounding points: RMSNorm rounds the normalized row
to BF16 before the BF16 weight product, SwiGLU rounds ``silu(gate)`` to BF16
before the ``up`` product, and every BF16 tensor the reference materializes
is rounded here too. RoPE is computed in FP32 from an FP32 ``cos_sin`` table
(the reference rounds its cos/sin to BF16) and rounded to BF16 once.

Interleaved RoPE (GLM's ``apply_rotary_pos_emb_interleave``) rotates the
pairs ``(x[2i], x[2i+1])`` by frequency ``i`` and writes the result
de-interleaved: ``out[i] = x[2i] cos - x[2i+1] sin``, ``out[32+i] =
x[2i+1] cos + x[2i] sin``. Queries and keys use the same order, so dot
products equal the reference's. ``cos_sin`` is FP32 ``[P, 64]``: cos of the
32 frequencies then their sin (the DeepSeek V4 table layout).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    ld_global_v4_u32,
    ld_shared_v2_u32,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo, _ld_cached

from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32

FP8_MAX = 448.0
ROPE_PAIRS = 32
# Skinny GEMV -> TMA crossover for the GLM BF16 projections with N >= 2048:
# the GEMV is flat to 8 rows then grows linearly, the TMA route is flat
# (SM120, L2-cold, us at M=1/8/16/24: q_b 45/47/59/102 vs 49/49/49/49;
# o_proj 127/133/174/231 vs 135/133/133/133; gate_up(12288) 186/190/248/383
# vs 225/223/223/225).
WIDE_SKINNY_MAX_ROWS = 8


def glm_projection(n: int, k: int, wide: bool = False):
    """``RoutedBf16Projection`` with the GLM crossover (narrow outputs keep the default);
    ``wide`` adds its 128 x 128-tile route for prefill capacities."""
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

    return RoutedBf16Projection(n, k, max_skinny_rows=WIDE_SKINNY_MAX_ROWS if n >= 2048 else None, wide=wide)


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _warp_max(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = fmax_f32(value, Float32(cute.arch.shuffle_sync_bfly(value, offset=1 << shift)))
    return value


def _reduction_storage(slots: int, warps: int):
    class Storage:
        pass

    Storage.__annotations__ = {
        "sums": cute.struct.Align[cute.struct.MemRange[Float32, slots * warps], 16],
    }
    return cute.struct(Storage)


@cute.jit
def _block_sum(value: Float32, sums: cute.Tensor, slot: Int32, warps: cutlass.Constexpr):
    lane = Int32(cute.arch.thread_idx()[0]) % Int32(32)
    warp_id = Int32(cute.arch.thread_idx()[0]) // Int32(32)
    value = _warp_sum(value)
    if lane == Int32(0):
        sums[slot, warp_id] = value
    cute.arch.sync_threads()
    total = Float32(0.0)
    for src in cutlass.range_constexpr(warps):
        total = total + sums[slot, src]
    return total


@cute.jit
def _rsqrt(value: Float32) -> Float32:
    return div_rn_f32(Float32(1.0), cute.math.sqrt(value, fastmath=False))


@cute.jit
def _bf16(value: Float32) -> Float32:
    """Round to BF16 (RNE) through the packing instruction: ``Float32(x.to(BFloat16))``
    lowers to a truncf/extf pair the compiler may fold away."""
    return _bf16_lo(pack_f32x2_to_bfloat2(value, value))


@cute.jit
def _fp8_scale(amax: Float32) -> Float32:
    """``amax / 448`` (1.0 for an all-zero block), as ``pack_mla_kv_cache_reference``."""
    scale = Float32(1.0)
    if amax > Float32(0.0):
        scale = div_rn_f32(amax, Float32(FP8_MAX))
    return scale


# ---------------------------------------------------------------------------
# Residual add + RMSNorm
# ---------------------------------------------------------------------------


class GlmAddRmsNorm:
    """``residual += delta`` (optional) then ``out = w * RMSNorm(residual)``.

    ``deltas`` (runtime): 0 normalizes ``residual`` as is; 1 adds ``delta0``;
    2 adds ``bf16(delta0 + delta1)`` (routed + shared expert outputs, the
    reference MoE order). The updated residual is written back in BF16.
    One CTA per row; each thread owns 16-byte vectors ``t, t+256, ...``, all
    loads issue before any store.
    """

    threads = 256

    def __init__(self, width: int, eps: float):
        self.width = int(width)
        if self.width % (8 * self.threads):
            raise ValueError("RMSNorm width must be a multiple of 8 * CTA width")
        self.vectors = self.width // (8 * self.threads)
        self.eps = float(eps)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, residual: cute.Pointer, delta0: cute.Pointer, delta1: cute.Pointer,
                 weight: cute.Pointer, out: cute.Pointer, rows: Int32, deltas: Int32,
                 stream: cuda.CUstream):
        self.kernel(residual, delta0, delta1, weight, out, deltas).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _load(self, address: Int64, values: cute.Tensor, cached: cutlass.Constexpr):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            at = address + (Int64(j * self.threads) + tidx) * Int64(16)
            if cutlass.const_expr(cached):
                words = _ld_cached(at)
            else:
                words = ld_global_v4_u32(at)
            for i in cutlass.range_constexpr(4):
                values[8 * j + 2 * i] = _bf16_lo(words[i])
                values[8 * j + 2 * i + 1] = _bf16_hi(words[i])

    @cute.jit
    def _store(self, address: Int64, values: cute.Tensor):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            at = address + (Int64(j * self.threads) + tidx) * Int64(16)
            st_global_v4_u32(at, pack_f32x2_to_bfloat2(values[8 * j], values[8 * j + 1]),
                             pack_f32x2_to_bfloat2(values[8 * j + 2], values[8 * j + 3]),
                             pack_f32x2_to_bfloat2(values[8 * j + 4], values[8 * j + 5]),
                             pack_f32x2_to_bfloat2(values[8 * j + 6], values[8 * j + 7]))

    @cute.kernel
    def kernel(self, residual: cute.Pointer, delta0: cute.Pointer, delta1: cute.Pointer,
               weight: cute.Pointer, out: cute.Pointer, deltas: Int32):
        row = Int64(cute.arch.block_idx()[0]) * Int64(self.width * 2)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(1, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((1, self.warps), stride=(self.warps, 1)))
        count = 8 * self.vectors
        values = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        self._load(Int64(residual.toint()) + row, values, False)
        if deltas > Int32(0):
            add = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
            self._load(Int64(delta0.toint()) + row, add, True)
            if deltas > Int32(1):
                other = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
                self._load(Int64(delta1.toint()) + row, other, True)
                for k in cutlass.range_constexpr(count):
                    add[k] = _bf16(add[k] + other[k])
            for k in cutlass.range_constexpr(count):
                values[k] = _bf16(values[k] + add[k])
            self._store(Int64(residual.toint()) + row, values)
        square = Float32(0.0)
        for k in cutlass.range_constexpr(count):
            square = square + values[k] * values[k]
        total = _block_sum(square, sums, Int32(0), self.warps)
        inv = _rsqrt(total / Float32(self.width) + Float32(self.eps))
        w = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        self._load(Int64(weight.toint()), w, True)
        for k in cutlass.range_constexpr(count):
            values[k] = w[k] * _bf16(values[k] * inv)
        self._store(Int64(out.toint()) + row, values)


# ---------------------------------------------------------------------------
# Attention producer epilogues
# ---------------------------------------------------------------------------


class GlmRankNormPackKV:
    """q_a/kv_a layer norms, key RoPE and the FP8 656-byte latent record.

    ``qkv`` is BF16 ``[rows, Q + 576]`` (``[q_a | kv latent 512 | k_rot 64]``).
    Writes ``q_resid = q_a_layernorm(q_a)`` BF16 ``[rows, Q]`` and, at
    ``slots[row]`` (``page * 64 + row``; negative skips), the record
    ``[512 E4M3 | 4 FP32 group scales | 64 BF16 RoPE]`` of
    ``kv_a_layernorm(latent)`` with group scale ``amax / 448``.
    """

    threads = 256

    def __init__(self, *, q_rank: int, eps: float, page_rows: int = 64, record_bytes: int = 656,
                 rope: int = 2 * ROPE_PAIRS):
        self.rope = int(rope)
        if self.rope not in (0, 2 * ROPE_PAIRS):
            raise ValueError("the latent record carries 64 RoPE dims or none (GLM 5.3 Flash)")
        self.q_rank = int(q_rank)
        if self.q_rank % self.threads:
            raise ValueError("q_lora_rank must divide by the CTA width")
        self.q_per_thread = self.q_rank // self.threads
        self.eps = float(eps)
        self.warps = self.threads // 32
        self.page_rows = int(page_rows)
        self.record_bytes = int(record_bytes)

    @cute.jit
    def __call__(self, qkv: cute.Pointer, q_norm: cute.Pointer, kv_norm: cute.Pointer,
                 positions: cute.Pointer, slots: cute.Pointer, cos_sin: cute.Pointer,
                 q_resid: cute.Pointer, cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        width = self.q_rank + 512 + self.rope
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(qkv, cute.make_layout((m, width), stride=(width, 1))),
            cute.make_tensor(q_norm, cute.make_layout((self.q_rank,))),
            cute.make_tensor(kv_norm, cute.make_layout((512,))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cute.make_tensor(slots, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(q_resid, cute.make_layout((m, self.q_rank), stride=(self.q_rank, 1))),
            cache,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, q_norm: cute.Tensor, kv_norm: cute.Tensor,
               positions: cute.Tensor, slots: cute.Tensor, cos_sin: cute.Pointer,
               q_resid: cute.Tensor, cache: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(2, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((2, self.warps), stride=(self.warps, 1)))

        qv = cute.make_rmem_tensor(cute.make_layout((self.q_per_thread,), stride=(1,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(self.q_per_thread):
            d = Int64(j * self.threads) + Int64(tidx)
            qv[j] = Float32(qkv[token, d])
            square = square + qv[j] * qv[j]
        # Latent: threads 0..127 own four dims each (warp w = FP8 group w).
        kv = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        kv_square = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            kv[e] = Float32(0.0)
        if tidx < Int32(128):
            for e in cutlass.range_constexpr(4):
                kv[e] = Float32(qkv[token, Int64(self.q_rank) + Int64(4) * Int64(tidx) + Int64(e)])
                kv_square = kv_square + kv[e] * kv[e]
        q_total = _block_sum(square, sums, Int32(0), self.warps)
        kv_total = _block_sum(kv_square, sums, Int32(1), self.warps)
        q_inv = _rsqrt(q_total / Float32(self.q_rank) + Float32(self.eps))
        k_inv = _rsqrt(kv_total / Float32(512.0) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.q_per_thread):
            d = Int64(j * self.threads) + Int64(tidx)
            q_resid[token, d] = (Float32(q_norm[d]) * _bf16(qv[j] * q_inv)).to(BFloat16)

        slot = Int64(slots[token])
        if slot >= Int64(0):
            page = slot // Int64(self.page_rows)
            row = slot - page * Int64(self.page_rows)
            record = Int64(cache.toint()) + page * Int64(self.page_rows * self.record_bytes) \
                + row * Int64(self.record_bytes)
            if tidx < Int32(128):
                normalized = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
                local = Float32(0.0)
                for e in cutlass.range_constexpr(4):
                    dk = Int32(4) * tidx + Int32(e)
                    normalized[e] = _bf16(Float32(kv_norm[dk]) * _bf16(kv[e] * k_inv))
                    local = fmax_f32(local, fabs_f32(normalized[e]))
                scale = _fp8_scale(_warp_max(local))
                packed = cvt_f32x4_to_e4m3x4(
                    div_rn_f32(normalized[0], scale), div_rn_f32(normalized[1], scale),
                    div_rn_f32(normalized[2], scale), div_rn_f32(normalized[3], scale))
                words = cute.make_ptr(Uint32, record + Int64(4) * Int64(tidx),
                                      cute.AddressSpace.gmem, assumed_align=4)
                words[0] = packed
                if lane == Int32(0):
                    scale_ptr = cute.make_ptr(Float32, record + Int64(512) + Int64(4) * Int64(tidx // Int32(32)),
                                              cute.AddressSpace.gmem, assumed_align=4)
                    scale_ptr[0] = scale
            elif cutlass.const_expr(self.rope > 0) and tidx < Int32(128 + ROPE_PAIRS):
                pair = tidx - Int32(128)
                position = Int64(positions[token])
                cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                                   cute.AddressSpace.gmem, assumed_align=4)
                cos_v = Float32(cs[pair])
                sin_v = Float32(cs[Int32(ROPE_PAIRS) + pair])
                base = Int64(self.q_rank + 512) + Int64(2) * Int64(pair)
                even = Float32(qkv[token, base])
                odd = Float32(qkv[token, base + Int64(1)])
                rope = cute.make_ptr(BFloat16, record + Int64(528), cute.AddressSpace.gmem, assumed_align=2)
                rope[pair] = (even * cos_v - odd * sin_v).to(BFloat16)
                rope[Int32(ROPE_PAIRS) + pair] = (odd * cos_v + even * sin_v).to(BFloat16)


class GlmQueryRope:
    """``query[:, h, 512:576] = rope(q[:, h, nope:nope+64])`` (BF16, de-interleaved)."""

    def __init__(self, *, heads: int, qk_nope: int, qk_head: int, latent: int = 576):
        self.heads, self.nope, self.head, self.latent = int(heads), int(qk_nope), int(qk_head), int(latent)

    @cute.jit
    def __call__(self, q: cute.Pointer, positions: cute.Pointer, cos_sin: cute.Pointer,
                 query: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        h = self.heads
        self.kernel(
            cute.make_tensor(q, cute.make_layout((m, h, self.head), stride=(h * self.head, self.head, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(query, cute.make_layout((m, h, self.latent), stride=(h * self.latent, self.latent, 1))),
        ).launch(grid=(rows, 1, 1), block=(ROPE_PAIRS * 8, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, positions: cute.Tensor, cos_sin: cute.Pointer, query: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        pair = tidx % Int32(ROPE_PAIRS)
        position = Int64(positions[token])
        cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                           cute.AddressSpace.gmem, assumed_align=4)
        cos_v = Float32(cs[pair])
        sin_v = Float32(cs[Int32(ROPE_PAIRS) + pair])
        head0 = Int64(tidx // Int32(ROPE_PAIRS))
        per = self.heads // 8
        src = Int64(self.nope) + Int64(2) * Int64(pair)
        values = cute.make_rmem_tensor(cute.make_layout((2 * per,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(per):  # every load before any store
            values[2 * i] = Float32(q[token, head0 + Int64(8 * i), src])
            values[2 * i + 1] = Float32(q[token, head0 + Int64(8 * i), src + Int64(1)])
        for i in cutlass.range_constexpr(per):
            even, odd = values[2 * i], values[2 * i + 1]
            head = head0 + Int64(8 * i)
            query[token, head, Int64(512) + Int64(pair)] = (even * cos_v - odd * sin_v).to(BFloat16)
            query[token, head, Int64(512 + ROPE_PAIRS) + Int64(pair)] = (odd * cos_v + even * sin_v).to(BFloat16)


@cute.jit
def _source_index(lane: Int32, e: cutlass.Constexpr) -> Int32:
    """Input dim ``e`` (0..7) of ``lane``: RoPE lanes (< 16) read the pairs of
    frequencies ``4*(lane%8) .. +3``; pass-through lanes read ``4*lane + e``
    (entries 4..7 repeat entries 0..3 there, unused)."""
    index = Int32(4) * lane + Int32(e % 4)
    if lane < Int32(16):
        index = Int32(8) * (lane % Int32(8)) + Int32(e)
    return index


class GlmIndexPost:
    """DSA indexer query/key epilogue.

    ``iq`` BF16 ``[rows, H*128]`` = ``wq_b(q_resid)``; ``kw`` BF16
    ``[rows, 128 + H]`` = ``[wk(x) | weights_proj(x)]``. Per (row, head): RoPE
    on dims 0:64, BF16 round, E4M3 with per-(row, head) scale ``amax/448``
    into ``q_fp8 [rows, H, 128]``; ``head_weights = bf16(proj) * H^-0.5 *
    128^-0.5 * q_scale`` (the query scale folds out of ``relu``). Key:
    ``LayerNorm(wk(x))`` (weight, bias, eps), RoPE, BF16 round, E4M3 with a
    per-token FP32 scale into the index cache page (64 x 128 E4M3 rows then
    64 FP32 scales) at ``slots[row]``.

    One warp per (row, head) plus one warp for the key; lane ``l`` owns the
    four outputs ``4l..4l+3``.
    """

    def __init__(self, *, heads: int, eps: float, weight_scale: float, page_rows: int = 64):
        self.heads = int(heads)
        self.eps = float(eps)
        self.weight_scale = float(weight_scale)
        self.page_rows = int(page_rows)
        self.page_bytes = self.page_rows * (128 + 4)

    @cute.jit
    def __call__(self, iq: cute.Pointer, kw: cute.Pointer, positions: cute.Pointer,
                 slots: cute.Pointer, cos_sin: cute.Pointer, k_weight: cute.Pointer,
                 k_bias: cute.Pointer, q_fp8: cute.Pointer, head_weights: cute.Pointer,
                 cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        h = self.heads
        self.kernel(
            cute.make_tensor(iq, cute.make_layout((m, h, 128), stride=(h * 128, 128, 1))),
            cute.make_tensor(kw, cute.make_layout((m, 128 + h), stride=(128 + h, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cute.make_tensor(slots, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(k_weight, cute.make_layout((128,))),
            cute.make_tensor(k_bias, cute.make_layout((128,))),
            q_fp8,
            cute.make_tensor(head_weights, cute.make_layout((m, h), stride=(h, 1))),
            cache,
        ).launch(grid=(rows, h + 1, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def _rotate(self, lane: Int32, src: cute.Tensor, cs: cute.Pointer, out: cute.Tensor):
        """Four outputs of ``lane`` from the eight inputs ``_source_index`` names."""
        if lane < Int32(16):
            for e in cutlass.range_constexpr(4):
                pair = Int32(4) * (lane % Int32(8)) + Int32(e)
                cos_v = Float32(cs[pair])
                sin_v = Float32(cs[Int32(ROPE_PAIRS) + pair])
                even = src[2 * e]
                odd = src[2 * e + 1]
                if lane < Int32(8):
                    out[e] = _bf16(even * cos_v - odd * sin_v)
                else:
                    out[e] = _bf16(odd * cos_v + even * sin_v)
        else:
            for e in cutlass.range_constexpr(4):
                out[e] = src[e]

    @cute.kernel
    def kernel(self, iq: cute.Tensor, kw: cute.Tensor, positions: cute.Tensor, slots: cute.Tensor,
               cos_sin: cute.Pointer, k_weight: cute.Tensor, k_bias: cute.Tensor,
               q_fp8: cute.Pointer, head_weights: cute.Tensor, cache: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        head = Int32(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        position = Int64(positions[token])
        cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                           cute.AddressSpace.gmem, assumed_align=4)
        out = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        if head < Int32(self.heads):
            h64 = Int64(head)
            src = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
            for e in cutlass.range_constexpr(8):
                src[e] = Float32(iq[token, h64, Int64(_source_index(lane, e))])
            self._rotate(lane, src, cs, out)
            local = fmax_f32(fmax_f32(fabs_f32(out[0]), fabs_f32(out[1])),
                             fmax_f32(fabs_f32(out[2]), fabs_f32(out[3])))
            scale = _fp8_scale(_warp_max(local))
            packed = cvt_f32x4_to_e4m3x4(div_rn_f32(out[0], scale), div_rn_f32(out[1], scale),
                                         div_rn_f32(out[2], scale), div_rn_f32(out[3], scale))
            words = cute.make_ptr(
                Uint32, Int64(q_fp8.toint()) + (token * Int64(self.heads) + h64) * Int64(128)
                + Int64(4) * Int64(lane), cute.AddressSpace.gmem, assumed_align=4)
            words[0] = packed
            if lane == Int32(0):
                raw = Float32(kw[token, Int64(128) + h64])
                head_weights[token, h64] = raw * Float32(self.weight_scale) * scale
        else:
            # LayerNorm statistics over the 128 raw key values.
            raw = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            total = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                raw[e] = Float32(kw[token, Int64(4) * Int64(lane) + Int64(e)])
                total = total + raw[e]
            mean = _warp_sum(total) / Float32(128.0)
            var = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                diff = raw[e] - mean
                var = var + diff * diff
            rstd = _rsqrt(_warp_sum(var) / Float32(128.0) + Float32(self.eps))

            src = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
            for e in cutlass.range_constexpr(8):
                dk = _source_index(lane, e)
                xk = Float32(kw[token, Int64(dk)])
                src[e] = _bf16((xk - mean) * rstd * Float32(k_weight[dk]) + Float32(k_bias[dk]))
            self._rotate(lane, src, cs, out)
            local = fmax_f32(fmax_f32(fabs_f32(out[0]), fabs_f32(out[1])),
                             fmax_f32(fabs_f32(out[2]), fabs_f32(out[3])))
            scale = _fp8_scale(_warp_max(local))
            slot = Int64(slots[token])
            if slot >= Int64(0):
                page = slot // Int64(self.page_rows)
                row = slot - page * Int64(self.page_rows)
                page_base = Int64(cache.toint()) + page * Int64(self.page_bytes)
                packed = cvt_f32x4_to_e4m3x4(div_rn_f32(out[0], scale), div_rn_f32(out[1], scale),
                                             div_rn_f32(out[2], scale), div_rn_f32(out[3], scale))
                words = cute.make_ptr(Uint32, page_base + row * Int64(128) + Int64(4) * Int64(lane),
                                      cute.AddressSpace.gmem, assumed_align=4)
                words[0] = packed
                if lane == Int32(0):
                    scale_ptr = cute.make_ptr(Float32, page_base + Int64(self.page_rows * 128) + row * Int64(4),
                                              cute.AddressSpace.gmem, assumed_align=4)
                    scale_ptr[0] = scale


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


class GlmSwiGLU:
    """``hidden = bf16(bf16(silu(gate)) * up)`` from ``gate_up [rows, 2I]`` (gate first).

    With ``limit`` (GLM 5.3 Flash's ``swiglu_limit``) the BF16 gate is clamped
    to ``<= limit`` and up to ``[-limit, limit]`` first, as the reference.
    """

    threads = 256

    def __init__(self, inter: int, limit: float | None = None):
        self.inter = int(inter)
        self.limit = None if limit is None else float(limit)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, hidden: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        i = self.inter
        self.kernel(cute.make_tensor(gate_up, cute.make_layout((m, 2 * i), stride=(2 * i, 1))),
                    cute.make_tensor(hidden, cute.make_layout((m, i), stride=(i, 1)))).launch(
            grid=(rows, (i + self.threads - 1) // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Tensor, hidden: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.inter):
            gate = Float32(gate_up[row, col])
            up = Float32(gate_up[row, Int64(self.inter) + col])
            if cutlass.const_expr(self.limit is not None):
                gate = fmin_f32(gate, Float32(self.limit))
                up = fmax_f32(fmin_f32(up, Float32(self.limit)), Float32(-self.limit))
            silu = _bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
            hidden[row, col] = (silu * up).to(BFloat16)


# ---------------------------------------------------------------------------
# Batched (per-head) BF16 GEMM: TMA pipeline, FP32 MMA accumulation
# ---------------------------------------------------------------------------


class BatchedBf16Gemm:
    """``out[m, n, l] = sum_k a[m, k, l] * w[l, n, k]`` for ``l < batch``.

    ``a`` rows (stride ``a_row``) hold ``batch`` K-slices at stride
    ``a_batch`` (e.g. per-head slices of a projected query); ``w`` is dense
    ``[batch, N, K]``; ``out`` rows (stride ``o_row``) take ``batch``
    N-slices at stride ``o_batch``. Grid ``(ceil(rows/tile_m), N/tile_n,
    batch)``; the Bf16PrefillKernel pipeline (warp MMA m16n8k16, FP32
    accumulators, TMA producer warp) with an L mode.
    """

    tile_k = 64
    buffer_align_bytes = 1024

    def __init__(self, *, n: int, k: int, batch: int, a_row: int, a_batch: int, o_row: int,
                 o_batch: int, compute_warps: int = 4, tile_n: int = 128, num_stages: int = 3):
        self.n, self.k, self.batch = int(n), int(k), int(batch)
        self.a_row, self.a_batch, self.o_row, self.o_batch = int(a_row), int(a_batch), int(o_row), int(o_batch)
        self.num_compute_warps = int(compute_warps)
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.num_compute_warps
        self.tile_n = int(tile_n)
        self.num_stages = int(num_stages)
        if self.k % self.tile_k or self.n % self.tile_n:
            raise ValueError("batched BF16 GEMM needs K % 64 == 0 and N % tile_n == 0")
        if (self.a_row * 2) % 16 or (self.a_batch * 2) % 16:
            raise ValueError("TMA source strides must be 16-byte multiples")
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = self.n // self.tile_n

    def key(self) -> tuple:
        return (self.n, self.k, self.batch, self.a_row, self.a_batch, self.o_row, self.o_batch,
                self.num_compute_warps, self.tile_n, self.num_stages)

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    def _smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.tile_k),
            cutlass.BFloat16,
        )
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k, self.num_stages), order=(0, 1, 2))
        return s_a, s_b

    def _storage(self, s_a, s_b):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_a)],
                                    self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_b)],
                                    self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(self, a: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        a_t = cute.make_tensor(a, cute.make_layout((rows, self.k, self.batch),
                                                    stride=(self.a_row, 1, self.a_batch)))
        w_t = cute.make_tensor(w, cute.make_layout((self.n, self.k, self.batch),
                                                    stride=(self.k, 1, self.n * self.k)))
        o_t = cute.make_tensor(out, cute.make_layout((rows, self.n, self.batch),
                                                      stride=(self.o_row, 1, self.o_batch)))
        s_a, s_b = self._smem_layouts()
        tiled_mma = self._tiled_mma()
        storage = self._storage(s_a, s_b)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1)
        tma_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_b, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(tma_tensor_a, tma_tensor_b, o_t, tma_a, tma_b, s_a, s_b, tiled_mma, storage,
                    rows).launch(grid=(grid_m, self.n_tiles, self.batch),
                                 block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, output: cute.Tensor,
               tma_atom_a: cute.CopyAtom, tma_atom_b: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
               s_b_layout: cute.ComposedLayout, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        tma_bytes = (self.tile_m + self.tile_n) * self.tile_k * cutlass.BFloat16.width // 8
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None, None))
        g_b = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                           cute.group_modes(g_a, 0, 2))
        t_bs, t_bg = cpasync.tma_partition(tma_atom_b, 0, cta_layout, cute.group_modes(s_b, 0, 2),
                                           cute.group_modes(g_b, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb[None, None, None, 0])
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
            for _k in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.consumer_wait(consumer_state)
                shared_a = t_ssa[None, None, None, consumer_state.index]
                shared_b = t_ssb[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, shared_b[None, None, 0], target_b[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                    if kk < cute.size(shared_a.shape[2]) - 1:
                        cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                        cute.copy(copy_b, shared_b[None, None, kk + 1], target_b[None, None, kk + 1])
                    cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            # Accumulator pairs (2i, 2i + 1) are adjacent columns of one row: one
            # packed BF16x2 store each (the same round-to-nearest conversion).
            base = Int64(output.iterator.toint())
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                coord = coordinates[2 * pair]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens:
                    at = (Int64(token) * Int64(self.o_row) + Int64(batch) * Int64(self.o_batch) + Int64(column)) * Int64(2)
                    st_global_u32(base + at, pack_f32x2_to_bfloat2(acc[2 * pair], acc[2 * pair + 1]))
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile, batch)], t_as[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_b, t_bg[(None, n_tile, k_tile, batch)], t_bs[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class BatchedFp8Gemm(BatchedBf16Gemm):
    """``BatchedBf16Gemm`` over E4M3 weights ``w [batch, N, K]`` with one FP32 scale per weight
    row and 64-wide K tile (``scale [batch, N, K/64]``): each stage's weight tile lands as
    E4M3 (half the bytes), the compute warps widen it to ``bf16(w * s)`` in a swizzled BF16
    tile, then run the same MMAs in the same K order, so results equal ``BatchedBf16Gemm``
    over the dequantized weights bitwise. Per-row scales carry block grids whose 128-row
    blocks do not align with the batch slices (GLM ``kv_b_proj``: 448 rows per head)."""

    def key(self) -> tuple:
        return ("fp8",) + super().key()

    def _fp8_layouts(self):
        s_a, _ = self._smem_layouts()
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.tile_k),
            cutlass.BFloat16)
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k), order=(0, 1))
        s_w = cute.make_layout((self.tile_n, self.tile_k, self.num_stages),
                               stride=(self.tile_k, 1, self.tile_n * self.tile_k))
        return s_a, s_b, s_w

    def _fp8_storage(self, s_a, s_b, s_w):
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
    def __call__(self, a: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        a_t = cute.make_tensor(a, cute.make_layout((rows, self.k, self.batch),
                                                    stride=(self.a_row, 1, self.a_batch)))
        w8 = cute.make_ptr(cutlass.Uint8, Int64(w.toint()), cute.AddressSpace.gmem, assumed_align=16)
        w_t = cute.make_tensor(w8, cute.make_layout((self.n, self.k, self.batch),
                                                     stride=(self.k, 1, self.n * self.k)))
        o_t = cute.make_tensor(out, cute.make_layout((rows, self.n, self.batch),
                                                      stride=(self.o_row, 1, self.o_batch)))
        s_a, s_b, s_w = self._fp8_layouts()
        storage = self._fp8_storage(s_a, s_b, s_w)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1)
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_w, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.fp8_kernel(tma_tensor_a, tma_tensor_w, o_t, scale, tma_a, tma_w, s_a, s_b, s_w, self._tiled_mma(),
                        storage, rows).launch(grid=(grid_m, self.n_tiles, self.batch),
                                              block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def fp8_kernel(self, source: cute.Tensor, weight: cute.Tensor, output: cute.Tensor, scale: cute.Pointer,
                   tma_atom_a: cute.CopyAtom, tma_atom_w: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
                   s_b_layout: cute.ComposedLayout, s_w_layout: cute.Layout, tiled_mma: cute.TiledMma,
                   SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_w)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        s_w = storage.sW.get_tensor(s_w_layout)
        compute_threads = 32 * self.num_compute_warps
        chunks = self.tile_n * self.tile_k // 8 // compute_threads
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
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None, None))
        g_w = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                           cute.group_modes(g_a, 0, 2))
        t_ws, t_wg = cpasync.tma_partition(tma_atom_w, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                           cute.group_modes(g_w, 0, 2))
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
            # Scales of this CTA's weight rows: [batch, N, K/64].
            s_rows = Int64(scale.toint()) + (Int64(batch) * Int64(self.n) + Int64(n_tile) * Int64(self.tile_n)) \
                * Int64(self.k_tiles * 4)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.consumer_wait(consumer_state)
                stage = w_base + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
                for i in cutlass.range_constexpr(chunks):
                    chunk = Int32(i * compute_threads) + Int32(tidx)
                    n = chunk // Int32(self.tile_k // 8)
                    c = chunk % Int32(self.tile_k // 8)
                    s = _ld_f32(s_rows + (Int64(n) * Int64(self.k_tiles) + Int64(k_tile)) * Int64(4))
                    lo, hi = ld_shared_v2_u32(stage + n * Int32(self.tile_k) + c * Int32(8))
                    v0, v1, v2, v3 = _e4m3x8_scaled_bf16(lo, hi, s)
                    # 128-byte swizzle: 16-byte chunk c of row n sits at chunk c ^ (n % 8).
                    st_shared_v4_u32(b_base + n * Int32(128) + ((c ^ (n % Int32(8))) * Int32(16)), v0, v1, v2, v3)
                cute.arch.barrier(barrier_id=1, number_of_threads=compute_threads)
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
                cute.arch.barrier(barrier_id=1, number_of_threads=compute_threads)
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            base = Int64(output.iterator.toint())
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                coord = coordinates[2 * pair]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens:
                    at = (Int64(token) * Int64(self.o_row) + Int64(batch) * Int64(self.o_batch) + Int64(column)) * Int64(2)
                    st_global_u32(base + at, pack_f32x2_to_bfloat2(acc[2 * pair], acc[2 * pair + 1]))
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile, batch)], t_as[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_w, t_wg[(None, n_tile, k_tile, batch)], t_ws[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


__all__ = [
    "WIDE_SKINNY_MAX_ROWS",
    "glm_projection",
    "BatchedBf16Gemm",
    "BatchedFp8Gemm",
    "GlmAddRmsNorm",
    "GlmIndexPost",
    "GlmQueryRope",
    "GlmRankNormPackKV",
    "GlmSwiGLU",
]
