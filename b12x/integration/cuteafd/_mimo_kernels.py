"""CuTe DSL kernels composed into the MiMo V2 (``mimo_*``) AOT programs.

Every kernel takes raw pointers and a runtime ``rows``. Arithmetic follows
the transformers ``modeling_mimo_v2_flash`` rounding points where they are
cheap to keep (every BF16 tensor the reference materializes is rounded here
too); RoPE is FP32 from an FP32 ``cos_sin`` table and rounded once, and the
attention softmax is FP32 with BF16 probabilities for the PV product.

RoPE is NeoX style on the first 64 dims of every 192-wide query/key head
(``partial_rotary_factor`` 0.334): ``out[i] = x[i] cos_i - x[32+i] sin_i``,
``out[32+i] = x[32+i] cos_i + x[i] sin_i`` for ``i < 32``; ``cos_sin`` is FP32
``[P, 64]``, the cos of the 32 frequencies ``theta^(-2i/64)`` then their sin
(one table per theta: full layers 5e6, SWA layers 1e4).

KV record (one per token, BF16): every KV head's 192-wide key (RoPE applied)
then every KV head's 128-wide value times ``v_scale`` (``record_elems =
kv_heads * 320``).

8-bit KV record (``kv8``, :func:`fp8_record_bytes`): the same keys and values
rounded to BF16 as above, then quantized with one FP32 scale per ``kv_group``
dims of each head's key and value (64 or 32; 0: one per key, one per value):
``"s8"`` signed bytes ``clamp(rne(x / s), -127, 127)`` with ``s = amax / 127``,
``"e4m3"`` E4M3 ``rn(x / s)`` with ``s = amax / 448`` (``s = 1`` for an all-zero
group). Layout: ``[G, 192]`` keys, ``[G, 128]`` values, ``[G, 192 / kv_group]``
FP32 key scales, ``[G, 128 / kv_group]`` FP32 value scales, zero padding to 16
bytes. Attention widens each key/value to ``bf16(q * s)`` in shared memory and
runs the BF16 MMAs unchanged. On the MiMo goldens E4M3 keys are too coarse
(3-bit mantissa: KL(golden||engine) +0.235 on V2 Flash, +0.012 on V2.6 Pro over
BF16 records) while int8 per 32 dims stays within noise (-0.0002, +0.0015).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    div_rn_f32, fabs_f32, fmax_f32, ld_global_nc_v2_u32, ld_global_nc_v4_u32, ld_global_v4_u32, shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
)
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention.paged._selected_forward import _cp_async_load_128b_zfill

from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from ._glm_kernels import _bf16, _warp_max

ROPE_HALF = 32
LOG2_E = 1.4426950408889634
MAX_SPLITS = 64


def _ceil(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


KV8_FORMATS = ("e4m3", "s8")
KV8_MAX = {"e4m3": 448.0, "s8": 127.0}


def fp8_scale_counts(kv_group: int, head: int = 192, v_head: int = 128) -> tuple[int, int]:
    """FP32 scales per KV head of an 8-bit record: (key, value)."""
    if int(kv_group) == 0:
        return 1, 1
    if int(kv_group) not in (32, 64):
        raise ValueError("kv_group is 32, 64 or 0 (per head)")
    return head // int(kv_group), v_head // int(kv_group)


@dsl_user_op
def _s8x8_scaled_bf16(lo, hi, s, *, loc=None, ip=None):
    """bf16(int8 * s) for eight packed signed bytes, as four bf16x2 words. Each byte, biased by
    128, becomes the low mantissa byte of 2^23 (``prmt``), so ``float - (2^23 + 128)`` is the
    integer exactly without an I2F conversion."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 x0, x1, m, u0, u1, u2, u3, u4, u5, u6, u7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            xor.b32 x0, $4, 0x80808080;
            xor.b32 x1, $5, 0x80808080;
            mov.b32 m, 0x4B000000;
            prmt.b32 u0, x0, m, 0x7440;
            prmt.b32 u1, x0, m, 0x7441;
            prmt.b32 u2, x0, m, 0x7442;
            prmt.b32 u3, x0, m, 0x7443;
            prmt.b32 u4, x1, m, 0x7440;
            prmt.b32 u5, x1, m, 0x7441;
            prmt.b32 u6, x1, m, 0x7442;
            prmt.b32 u7, x1, m, 0x7443;
            mov.b32 f0, u0;
            mov.b32 f1, u1;
            mov.b32 f2, u2;
            mov.b32 f3, u3;
            mov.b32 f4, u4;
            mov.b32 f5, u5;
            mov.b32 f6, u6;
            mov.b32 f7, u7;
            sub.f32 f0, f0, 0f4B000080;
            sub.f32 f1, f1, 0f4B000080;
            sub.f32 f2, f2, 0f4B000080;
            sub.f32 f3, f3, 0f4B000080;
            sub.f32 f4, f4, 0f4B000080;
            sub.f32 f5, f5, 0f4B000080;
            sub.f32 f6, f6, 0f4B000080;
            sub.f32 f7, f7, 0f4B000080;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _f32_to_s8(x, *, loc=None, ip=None):
    """``x`` rounded to the nearest integer (ties to even), clamped to [-127, 127]."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "{ .reg .s32 t; cvt.rni.s32.f32 t, $1; max.s32 t, t, -127; min.s32 $0, t, 127; }", "=r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def fp8_record_bytes(kv_heads: int, kv_group: int = 64, head: int = 192, v_head: int = 128) -> int:
    """Bytes of one token's FP8 KV record (E4M3 keys, values, FP32 scales, padded to 16)."""
    sk, sv = fp8_scale_counts(kv_group, head, v_head)
    raw = int(kv_heads) * (head + v_head) + 4 * int(kv_heads) * (sk + sv)
    return (raw + 15) // 16 * 16


# ---------------------------------------------------------------------------
# Producer epilogue: partial RoPE, value scale, KV record
# ---------------------------------------------------------------------------


class MimoQkvRope:
    """``qkv [rows, N*192 + R]`` (``[q_proj | k_proj | v_proj]``) -> ``query
    [rows, N, 192]`` with RoPE on dims 0:64 of every head, and the token's KV
    record at ``cache + slots[row] * R`` (negative slots skip): keys with
    RoPE, values ``bf16(v * v_scale)``. One CTA per row."""

    threads = 256

    def __init__(self, *, heads: int, kv_heads: int, v_scale: float, head: int = 192, v_head: int = 128,
                 k_stride: int | None = None, kv8: str | None = None, kv_group: int = 64):
        self.heads, self.kv_heads = int(heads), int(kv_heads)
        self.head, self.v_head = int(head), int(v_head)
        if kv8 is not None and kv8 not in KV8_FORMATS:
            raise ValueError(f"kv8 is one of {KV8_FORMATS}")
        self.kv8, self.kv_fp8, self.kv_group = kv8, kv8 is not None, int(kv_group)
        self.k_scales, self.v_scales = fp8_scale_counts(self.kv_group, self.head, self.v_head)
        self.record_bytes = fp8_record_bytes(self.kv_heads, self.kv_group, self.head, self.v_head) if self.kv_fp8 \
            else self.kv_heads * (self.head + self.v_head) * 2
        # Key heads are k_stride apart in qkv (zero padding past each 192-wide key).
        self.k_stride = self.head if k_stride is None else int(k_stride)
        self.v_scale = float(v_scale)
        self.q_width = self.heads * self.head
        self.record = self.kv_heads * (self.head + self.v_head)
        self.width = self.q_width + self.kv_heads * (self.k_stride + self.v_head)

    @cute.jit
    def __call__(self, qkv: cute.Pointer, positions: cute.Pointer, cos_sin: cute.Pointer,
                 slots: cute.Pointer, query: cute.Pointer, cache: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(qkv, cute.make_layout((m, self.width), stride=(self.width, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(query, cute.make_layout((m, self.q_width), stride=(self.q_width, 1))),
            cache,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _rope(self, src: cute.Tensor, token: Int64, src_base: Int64, dst: cute.Tensor, dst_row: Int64,
              dst_base: Int64, cos_v: Float32, sin_v: Float32, i: Int64):
        x1 = Float32(src[token, src_base + i])
        x2 = Float32(src[token, src_base + Int64(ROPE_HALF) + i])
        dst[dst_row, dst_base + i] = (x1 * cos_v - x2 * sin_v).to(BFloat16)
        dst[dst_row, dst_base + Int64(ROPE_HALF) + i] = (x2 * cos_v + x1 * sin_v).to(BFloat16)

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, positions: cute.Tensor, cos_sin: cute.Pointer, slots: cute.Tensor,
               query: cute.Tensor, cache: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        position = Int64(positions[token])
        cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(64 * 4),
                           cute.AddressSpace.gmem, assumed_align=4)
        pass_width = self.head - 2 * ROPE_HALF
        # Query: RoPE pairs, then the pass-through dims.
        for it in cutlass.range_constexpr(_ceil(self.heads * ROPE_HALF, self.threads)):
            idx = Int32(it * self.threads) + tidx
            if idx < Int32(self.heads * ROPE_HALF):
                h = Int64(idx // Int32(ROPE_HALF))
                i = Int32(idx % Int32(ROPE_HALF))
                cos_v = Float32(cs[i])
                sin_v = Float32(cs[Int32(ROPE_HALF) + i])
                self._rope(qkv, token, h * Int64(self.head), query, token, h * Int64(self.head), cos_v, sin_v,
                           Int64(i))
        for it in cutlass.range_constexpr(_ceil(self.heads * pass_width, self.threads)):
            idx = Int32(it * self.threads) + tidx
            if idx < Int32(self.heads * pass_width):
                h = Int64(idx // Int32(pass_width))
                d = Int64(2 * ROPE_HALF) + Int64(idx % Int32(pass_width))
                query[token, h * Int64(self.head) + d] = qkv[token, h * Int64(self.head) + d]
        slot = Int64(slots[token])
        if const_expr(self.kv_fp8):
            if slot >= Int64(0):
                self._record_fp8(qkv, token, Int64(cache.toint()) + slot * Int64(self.record_bytes), cs, tidx)
        elif slot >= Int64(0):
            record = cute.make_tensor(
                cute.make_ptr(BFloat16, Int64(cache.toint()) + slot * Int64(self.record * 2),
                              cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((1, self.record), stride=(self.record, 1)))
            k0 = Int64(self.q_width)
            for it in cutlass.range_constexpr(_ceil(self.kv_heads * ROPE_HALF, self.threads)):
                idx = Int32(it * self.threads) + tidx
                if idx < Int32(self.kv_heads * ROPE_HALF):
                    h = Int64(idx // Int32(ROPE_HALF))
                    i = Int32(idx % Int32(ROPE_HALF))
                    cos_v = Float32(cs[i])
                    sin_v = Float32(cs[Int32(ROPE_HALF) + i])
                    self._rope(qkv, token, k0 + h * Int64(self.k_stride), record, Int64(0), h * Int64(self.head),
                               cos_v, sin_v, Int64(i))
            for it in cutlass.range_constexpr(_ceil(self.kv_heads * pass_width, self.threads)):
                idx = Int32(it * self.threads) + tidx
                if idx < Int32(self.kv_heads * pass_width):
                    h = Int64(idx // Int32(pass_width))
                    d = Int64(2 * ROPE_HALF) + Int64(idx % Int32(pass_width))
                    record[Int64(0), h * Int64(self.head) + d] = qkv[token, k0 + h * Int64(self.k_stride) + d]
            v0 = Int64(self.kv_heads * self.head)
            v_src = k0 + Int64(self.kv_heads * self.k_stride)
            for it in cutlass.range_constexpr(_ceil(self.kv_heads * self.v_head, self.threads)):
                idx = Int32(it * self.threads) + tidx
                if idx < Int32(self.kv_heads * self.v_head):
                    value = Float32(qkv[token, v_src + Int64(idx)])
                    record[Int64(0), v0 + Int64(idx)] = (value * Float32(self.v_scale)).to(BFloat16)


    @cute.jit
    def _record_fp8(self, qkv: cute.Tensor, token: Int64, record: Int64, cs: cute.Pointer, tidx: Int32):
        """The token's FP8 record: one warp per (KV head, key | value); lane ``l`` holds dims
        ``l + 32 j`` (RoPE pairs stay in one lane), BF16-rounded, then E4M3 per scale group."""
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        k0 = Int64(self.q_width)
        v_src = k0 + Int64(self.kv_heads * self.k_stride)
        k_scales = record + Int64(self.kv_heads * (self.head + self.v_head))
        v_scales = k_scales + Int64(4 * self.kv_heads * self.k_scales)
        k_per = self.head // 32
        v_per = self.v_head // 32
        cos_v = Float32(cs[lane])
        sin_v = Float32(cs[Int32(ROPE_HALF) + lane])
        units = 2 * self.kv_heads
        for it in cutlass.range_constexpr(_ceil(units, self.threads // 32)):
            unit = Int32(it * (self.threads // 32)) + warp_id
            if unit < Int32(self.kv_heads):
                # Key of KV head `unit`: RoPE on dims 0:64 (lane, 32 + lane), the rest passed through.
                h = Int64(unit)
                src = k0 + h * Int64(self.k_stride)
                vals = cute.make_rmem_tensor(cute.make_layout((k_per,)), Float32)
                x1 = Float32(qkv[token, src + Int64(lane)])
                x2 = Float32(qkv[token, src + Int64(ROPE_HALF) + Int64(lane)])
                vals[0] = _bf16(x1 * cos_v - x2 * sin_v)
                vals[1] = _bf16(x2 * cos_v + x1 * sin_v)
                for j in cutlass.range_constexpr(2, k_per):
                    vals[j] = Float32(qkv[token, src + Int64(32 * j) + Int64(lane)])
                self._store_fp8(vals, k_per, record + h * Int64(self.head), k_scales + h * Int64(4 * self.k_scales),
                                lane)
            elif unit < Int32(units):
                h = Int64(unit - Int32(self.kv_heads))
                vals = cute.make_rmem_tensor(cute.make_layout((v_per,)), Float32)
                for j in cutlass.range_constexpr(v_per):
                    vals[j] = _bf16(Float32(qkv[token, v_src + h * Int64(self.v_head) + Int64(32 * j) + Int64(lane)])
                                    * Float32(self.v_scale))
                self._store_fp8(vals, v_per, record + Int64(self.kv_heads * self.head) + h * Int64(self.v_head),
                                v_scales + h * Int64(4 * self.v_scales), lane)

    @cute.jit
    def _store_fp8(self, vals: cute.Tensor, per: cutlass.Constexpr, dst: Int64, scale_dst: Int64, lane: Int32):
        """8-bit ``vals`` (lane's dims ``l + 32 j``) at ``dst`` with one scale ``amax / max`` per
        ``kv_group`` dims (E4M3: RN to E4M3; s8: round to nearest even, clamped to +-127)."""
        scales = cute.make_tensor(cute.make_ptr(Float32, scale_dst, cute.AddressSpace.gmem, assumed_align=4),
                                  cute.make_layout((per,)))
        group = per if self.kv_group == 0 else self.kv_group // 32
        for g0 in cutlass.range_constexpr(per // group):
            local = Float32(0.0)
            for j in cutlass.range_constexpr(group):
                local = fmax_f32(local, fabs_f32(vals[g0 * group + j]))
            amax = _warp_max(local)
            scale = Float32(1.0)
            if amax > Float32(0.0):
                scale = div_rn_f32(amax, Float32(KV8_MAX[self.kv8]))
            for j in cutlass.range_constexpr(group):
                q = div_rn_f32(vals[g0 * group + j], scale)
                if const_expr(self.kv8 == "e4m3"):
                    out = cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, dst, cute.AddressSpace.gmem,
                                                         assumed_align=1), cute.make_layout((per * 32,)))
                    out[Int32(32 * (g0 * group + j)) + lane] = q.to(cutlass.Float8E4M3FN)
                else:
                    out = cute.make_tensor(cute.make_ptr(cutlass.Int8, dst, cute.AddressSpace.gmem, assumed_align=1),
                                           cute.make_layout((per * 32,)))
                    out[Int32(32 * (g0 * group + j)) + lane] = _f32_to_s8(q).to(cutlass.Int8)
            if lane == Int32(0):
                scales[g0] = scale


# ---------------------------------------------------------------------------
# GQA attention over KV records
# ---------------------------------------------------------------------------


class MimoGqaAttention:
    """Causal (``window == 0``) or sliding-window GQA over BF16 KV records.

    One CTA per (``tokens`` consecutive rows, KV head, split). Its M tile is
    ``tokens * group`` query rows (token-major, the group's heads inside;
    padded to 16 per QK warp); a multi-token CTA's rows must belong to one
    sequence with consecutive positions (the prefill route runs one sequence).
    Keys stream in ``tile_n`` tiles over the CTA's key range ``[lo, hi]``
    (``hi`` = last row's position; ``lo`` = 0, or first position - window + 1),
    tile ``t`` going to split ``t % splits``; each (row, key) pair is masked
    to ``k <= p`` (and ``k > p - window``).

    ``paged``: keys live in a paged cache (``cache + (page_table[row *
    table_stride + k / page_rows] * page_rows + k % page_rows) * R``), or with
    ``contiguous`` at ``cache + k * R`` (a :class:`MimoKvWiden` copy).
    Otherwise (SWA): keys at or past the row's sequence start ``s =
    positions[seq_first[row]]`` come from this step's records (``kv_step``
    row ``seq_first[row] + k - s``), older keys from the sequence's ring
    (slot ``ring_slots[row] - p % ring + k % ring``).

    ``direct``: BF16 output ``[rows, N, 128]`` with the sink (when ``sink``)
    in the softmax denominator. Otherwise normalized FP32 partials
    ``[rows, splits, N, 128]`` and natural-log LSE ``[rows, splits, N]``
    (empty splits: LSE -inf, zero partial) for :class:`MimoSplitMerge`.

    FP32 scores (MMA m16n8k16, FP32 accumulation), online softmax in base 2,
    BF16 probabilities into the PV MMA. Loads are cp.async per tile (not yet
    double-buffered).
    """

    def __init__(self, *, heads: int, kv_heads: int, tokens: int, window: int, paged: bool, sink: bool,
                 direct: bool, softmax_scale: float, page_rows: int = 64, ring_rows: int = 256,
                 head: int = 192, v_head: int = 128, kv_warps: int = 4, kv8: str | None = None,
                 kv_group: int = 64, contiguous: bool = False):
        self.heads, self.kv_heads = int(heads), int(kv_heads)
        self.contiguous = bool(contiguous)
        if kv8 is not None and kv8 not in KV8_FORMATS:
            raise ValueError(f"kv8 is one of {KV8_FORMATS}")
        self.kv8, self.kv_fp8, self.kv_group = kv8, kv8 is not None, int(kv_group)
        self.group = self.heads // self.kv_heads
        self.tokens = int(tokens)
        self.window = int(window)
        self.paged, self.sink, self.direct = bool(paged), bool(sink), bool(direct)
        self.softmax_scale = float(softmax_scale)
        self.page_rows, self.ring_rows = int(page_rows), int(ring_rows)
        self.head, self.v_head = int(head), int(v_head)
        self.record = self.kv_heads * (self.head + self.v_head)
        self.k_scales, self.v_scales = fp8_scale_counts(self.kv_group, self.head, self.v_head)
        self.record_bytes = fp8_record_bytes(self.kv_heads, self.kv_group, self.head, self.v_head) if self.kv_fp8 \
            else self.record * 2
        self.tile_n = 16 * int(kv_warps)
        self.qk_warps = _ceil(self.tokens * self.group, 16)
        self.tile_m = 16 * self.qk_warps
        self.pv_warps = 4 * self.qk_warps
        self.threads = 32 * self.pv_warps
        # 8-bit records: each thread loads whole 16-byte pieces (8-byte when 16 do not divide
        # across the CTA) of the next tile into registers while the current one computes.
        self.chunk = 16
        if (self.tile_n * self.head // 16) % self.threads or (self.tile_n * self.v_head // 16) % self.threads:
            self.chunk = 8
        self.k_iters = self.tile_n * self.head // self.chunk // self.threads
        self.v_iters = self.tile_n * self.v_head // self.chunk // self.threads
        if self.head % 64 or self.v_head % 64:
            raise ValueError("head dims must be multiples of 64")
        if (self.tile_n * self.head // 8) % self.threads or (self.tile_n * self.v_head // 8) % self.threads:
            raise ValueError("K/V tile vectors must divide across the CTA")
        if self.tile_n > self.threads:
            raise ValueError("tile_n must not exceed the CTA width")

    def key(self) -> tuple:
        return (self.heads, self.kv_heads, self.tokens, self.window, self.paged, self.sink, self.direct,
                self.softmax_scale, self.page_rows, self.ring_rows, self.head, self.v_head, self.tile_n,
                self.kv8, self.kv_group, self.contiguous)

    def _tiled_mma_qk(self):
        return cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (self.qk_warps, 1, 1),
                                   permutation_mnk=(self.tile_m, self.tile_n, 16))

    def _tiled_mma_pv(self):
        return cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (self.qk_warps, 4, 1),
                                   permutation_mnk=(self.tile_m, self.v_head, 16))

    def _layouts(self):
        def atom(width):
            return warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, width), BFloat16)

        return (cute.tile_to_shape(atom(self.head), (self.tile_m, self.head), order=(0, 1)),
                cute.tile_to_shape(atom(self.head), (self.tile_n, self.head), order=(0, 1)),
                cute.tile_to_shape(atom(self.v_head), (self.tile_n, self.v_head), order=(0, 1)),
                cute.tile_to_shape(atom(self.tile_n), (self.tile_m, self.tile_n), order=(0, 1)))

    @cute.jit
    def __call__(self, q: cute.Pointer, cache: cute.Pointer, kv_step: cute.Pointer, positions: cute.Pointer,
                 page_table: cute.Pointer, ring_slots: cute.Pointer, seq_first: cute.Pointer,
                 sinks: cute.Pointer, out: cute.Pointer, partial_out: cute.Pointer, partial_lse: cute.Pointer,
                 rows: Int32, table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        q_layout, k_layout, v_layout, p_layout = self._layouts()
        self.kernel(q, cache, kv_step, positions, page_table, ring_slots, seq_first, sinks, out, partial_out,
                    partial_lse, rows, table_stride, splits, q_layout, k_layout, v_layout, p_layout,
                    self._tiled_mma_qk(), self._tiled_mma_pv()).launch(
            grid=((rows + Int32(self.tokens - 1)) // Int32(self.tokens), self.kv_heads, splits),
            block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _resolve(self, tile: Int32, buf: Int32, lo: Int64, hi: Int64, cache: cute.Pointer, kv_step: cute.Pointer,
                 page_table: cute.Pointer, table_base: Int64, seq_start: Int64, step_first: Int64, ring_base: Int64,
                 key_addr: cute.Tensor, key_pos: cute.Tensor, thread: Int32):
        """Tile ``tile``'s keys into slot ``buf``: absolute record address (or -1) and position."""
        if thread < Int32(self.tile_n):
            k = lo + Int64(tile) * Int64(self.tile_n) + Int64(thread)
            address = Int64(-1)
            if k <= hi:
                record_bytes = Int64(self.record_bytes)
                if const_expr(self.paged and self.contiguous):
                    address = Int64(cache.toint()) + k * record_bytes
                elif const_expr(self.paged):
                    page = Int64(cute.make_ptr(Int32, Int64(page_table.toint()) + (table_base + k // Int64(
                        self.page_rows)) * Int64(4), cute.AddressSpace.gmem, assumed_align=4)[0])
                    address = Int64(cache.toint()) + (page * Int64(self.page_rows) + k % Int64(self.page_rows)) \
                        * record_bytes
                else:
                    if k >= seq_start:
                        address = Int64(kv_step.toint()) + (step_first + k - seq_start) * record_bytes
                    else:
                        address = Int64(cache.toint()) + (ring_base + k % Int64(self.ring_rows)) * record_bytes
            key_addr[buf, thread] = address
            key_pos[buf, thread] = cutlass.select_(address >= Int64(0), k, Int64(-1))

    @cute.jit
    def _tile(self, tile: Int32, next_tile: Int32, has_next, buf: Int32, lo: Int64, hi: Int64, row0: Int64,
              kv_head: Int32, cache: cute.Pointer,
              kv_step: cute.Pointer, page_table: cute.Pointer, table_base: Int64, seq_start: Int64,
              step_first: Int64, ring_base: Int64, token_pos: cute.Tensor, key_addr: cute.Tensor,
              key_pos: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor, s_p: cute.Tensor, s_scale: cute.Tensor,
              thread: Int32, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, r_q: cute.Tensor,
              r_k: cute.Tensor, r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy,
              c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor, cs_k: cute.Tensor, cs_p: cute.Tensor,
              cs_v: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax, raw: cute.Tensor, raw_s: cute.Tensor,
              is_first: cutlass.Constexpr):
        if const_expr(self.kv_fp8):
            # `raw` holds this tile's bytes (slot `buf`); widen them, then start the next tile's loads.
            self._widen(raw, raw_s, s_k, s_v, thread)
            if has_next:
                self._resolve(next_tile, Int32(1) - buf, lo, hi, cache, kv_step, page_table, table_base, seq_start,
                              step_first, ring_base, key_addr, key_pos, thread)
            cute.arch.sync_threads()
            if has_next:
                self._load_raw(key_addr, Int32(1) - buf, kv_head, cache, raw, raw_s, thread)
        else:
            self._resolve(tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                          ring_base, key_addr, key_pos, thread)
            cute.arch.sync_threads()
            self._load_bf16(key_addr, kv_head, cache, s_k, s_v)
            cute.arch.sync_threads()

        if thread < Int32(self.qk_warps * 32):
            acc_s = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc_s.fill(0.0)
            warp_mma_gemm(tiled_mma_qk, acc_s, r_q, r_k, cs_q, cs_k, c_q, c_k, A_in_regs=not is_first)
            s_mn = layout_utils.reshape_acc_to_mn(acc_s)
            coords = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                t_local = coords[m, 0][0] // Int32(self.group)
                p = Int64(-1)
                if t_local < Int32(self.tokens):
                    p = Int64(token_pos[t_local])
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    kp = Int64(key_pos[buf, coords[m, n][1]])
                    masked = (p < Int64(0)) | (kp < Int64(0)) | (kp > p)
                    if const_expr(self.window > 0):
                        masked = masked | (kp <= p - Int64(self.window))
                    if masked:
                        s_mn[m, n] = -Float32.inf
            row_scale = softmax.online_softmax(acc_s, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    c = coords[m, n]
                    s_p[c[0], c[1]] = BFloat16(s_mn[m, n])
                if coords[m, 0][1] == Int32(0):
                    s_scale[coords[m, 0][0]] = row_scale[m]
        cute.arch.sync_threads()

        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
                scale = Float32(s_scale[o_coords[m, 0][0]])
                o_mn[m, None].store(o_mn[m, None].load() * scale)
        warp_mma_gemm(tiled_mma_pv, acc_o, r_p, r_v, cs_p, cs_v, c_p, c_v)
        cute.arch.sync_threads()

    @cute.jit
    def _load_bf16(self, key_addr: cute.Tensor, kv_head: Int32, cache: cute.Pointer, s_k: cute.Tensor,
                   s_v: cute.Tensor):
        """BF16 records: each key's and value's 16-byte vectors by cp.async."""
        thread = Int32(cute.arch.thread_idx()[0])
        k_vectors = self.head // 8
        for it in cutlass.range_constexpr(self.tile_n * k_vectors // self.threads):
            linear = thread + Int32(it * self.threads)
            token = linear // Int32(k_vectors)
            dim = (linear % Int32(k_vectors)) * Int32(8)
            base = Int64(key_addr[0, token])
            valid = base >= Int64(0)
            source = cutlass.select_(valid, base + (Int64(kv_head) * Int64(self.head) + Int64(dim)) * Int64(2),
                                     Int64(cache.toint()))
            _cp_async_load_128b_zfill(shared_ptr_to_u32(s_k.iterator + cute.crd2idx((token, dim), s_k.layout)),
                                      source, cutlass.select_(valid, Int32(16), Int32(0)))
        v_vectors = self.v_head // 8
        v0 = self.kv_heads * self.head
        for it in cutlass.range_constexpr(self.tile_n * v_vectors // self.threads):
            linear = thread + Int32(it * self.threads)
            token = linear // Int32(v_vectors)
            dim = (linear % Int32(v_vectors)) * Int32(8)
            base = Int64(key_addr[0, token])
            valid = base >= Int64(0)
            source = cutlass.select_(
                valid, base + (Int64(v0) + Int64(kv_head) * Int64(self.v_head) + Int64(dim)) * Int64(2),
                Int64(cache.toint()))
            _cp_async_load_128b_zfill(shared_ptr_to_u32(s_v.iterator + cute.crd2idx((token, dim), s_v.layout)),
                                      source, cutlass.select_(valid, Int32(16), Int32(0)))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def _load_raw(self, key_addr: cute.Tensor, buf: Int32, kv_head: Int32, cache: cute.Pointer, raw: cute.Tensor,
                  raw_s: cute.Tensor, thread: Int32):
        """8-bit records of slot ``buf``'s keys: this thread's ``chunk``-byte pieces of every key and
        value and their scales into registers (missing keys: zeros), all loads in flight together."""
        k_vectors = self.head // self.chunk
        v_vectors = self.v_head // self.chunk
        words = self.chunk // 4
        k_scale_off = self.kv_heads * (self.head + self.v_head)
        v_scale_off = k_scale_off + 4 * self.kv_heads * self.k_scales
        k_group = self.head if self.kv_group == 0 else self.kv_group
        v_group = self.v_head if self.kv_group == 0 else self.kv_group
        fallback = Int64(cache.toint())
        for it in cutlass.range_constexpr(self.k_iters + self.v_iters):
            is_k = it < self.k_iters
            vectors = k_vectors if is_k else v_vectors
            linear = thread + Int32((it if is_k else it - self.k_iters) * self.threads)
            token = linear // Int32(vectors)
            dim = (linear % Int32(vectors)) * Int32(self.chunk)
            base = Int64(key_addr[buf, token])
            valid = base >= Int64(0)
            if const_expr(is_k):
                offset = Int64(kv_head) * Int64(self.head) + Int64(dim)
                scale_off = Int64(k_scale_off) + Int64(4) * (Int64(kv_head) * Int64(self.k_scales)
                                                             + Int64(dim // Int32(k_group)))
            else:
                offset = Int64(self.kv_heads * self.head) + Int64(kv_head) * Int64(self.v_head) + Int64(dim)
                scale_off = Int64(v_scale_off) + Int64(4) * (Int64(kv_head) * Int64(self.v_scales)
                                                             + Int64(dim // Int32(v_group)))
            src = cutlass.select_(valid, base + offset, fallback)
            if const_expr(words == 4):
                w0, w1, w2, w3 = ld_global_nc_v4_u32(src)
                raw[it * 4 + 0] = cutlass.select_(valid, w0, Uint32(0))
                raw[it * 4 + 1] = cutlass.select_(valid, w1, Uint32(0))
                raw[it * 4 + 2] = cutlass.select_(valid, w2, Uint32(0))
                raw[it * 4 + 3] = cutlass.select_(valid, w3, Uint32(0))
            else:
                w0, w1 = ld_global_nc_v2_u32(src)
                raw[it * 2 + 0] = cutlass.select_(valid, w0, Uint32(0))
                raw[it * 2 + 1] = cutlass.select_(valid, w1, Uint32(0))
            raw_s[it] = cutlass.select_(valid, _ld_f32(cutlass.select_(valid, base + scale_off, fallback)),
                                        Float32(0.0))

    @cute.jit
    def _widen(self, raw: cute.Tensor, raw_s: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor, thread: Int32):
        """The registers of :meth:`_load_raw` as ``bf16(q * scale)`` into the swizzled BF16 tiles."""
        k_vectors = self.head // self.chunk
        v_vectors = self.v_head // self.chunk
        words = self.chunk // 4
        widen = _e4m3x8_scaled_bf16 if self.kv8 == "e4m3" else _s8x8_scaled_bf16
        for it in cutlass.range_constexpr(self.k_iters + self.v_iters):
            is_k = it < self.k_iters
            vectors = k_vectors if is_k else v_vectors
            linear = thread + Int32((it if is_k else it - self.k_iters) * self.threads)
            token = linear // Int32(vectors)
            dim = (linear % Int32(vectors)) * Int32(self.chunk)
            for half in cutlass.range_constexpr(words // 2):
                w0, w1, w2, w3 = widen(raw[it * words + 2 * half], raw[it * words + 2 * half + 1], raw_s[it])
                tile = s_k if is_k else s_v
                st_shared_v4_u32(shared_ptr_to_u32(tile.iterator + cute.crd2idx((token, dim + Int32(8 * half)),
                                                                                tile.layout)), w0, w1, w2, w3)

    @cute.kernel
    def kernel(self, q: cute.Pointer, cache: cute.Pointer, kv_step: cute.Pointer, positions: cute.Pointer,
               page_table: cute.Pointer, ring_slots: cute.Pointer, seq_first: cute.Pointer, sinks: cute.Pointer,
               out: cute.Pointer, partial_out: cute.Pointer, partial_lse: cute.Pointer, rows: Int32,
               table_stride: Int32, splits: Int32, q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, tiled_mma_qk: cute.TiledMma,
               tiled_mma_pv: cute.TiledMma):
        block, kv_head_idx, split_idx = cute.arch.block_idx()
        thread = Int32(cute.arch.thread_idx()[0])
        kv_head = Int32(kv_head_idx)
        split = Int32(split_idx)
        row0 = Int64(block) * Int64(self.tokens)
        live = Int32(rows) - Int32(row0)
        if live > Int32(self.tokens):
            live = Int32(self.tokens)

        allocator = cutlass.utils.SmemAllocator()
        s_q = allocator.allocate_tensor(element_type=BFloat16, layout=q_layout, byte_alignment=1024)
        s_k = allocator.allocate_tensor(element_type=BFloat16, layout=k_layout, byte_alignment=1024)
        s_v = allocator.allocate_tensor(element_type=BFloat16, layout=v_layout, byte_alignment=1024)
        s_p = allocator.allocate_tensor(element_type=BFloat16, layout=p_layout, byte_alignment=1024)
        s_scale = allocator.allocate_tensor(element_type=Float32, layout=cute.make_layout((self.tile_m,)),
                                            byte_alignment=16)
        # Two slots: an 8-bit-record CTA resolves the next tile's keys while this one computes.
        key_addr = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout(
            (2, self.tile_n), stride=(self.tile_n, 1)), byte_alignment=16)
        key_pos = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout(
            (2, self.tile_n), stride=(self.tile_n, 1)), byte_alignment=16)
        token_pos = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout((self.tokens,)),
                                              byte_alignment=16)

        pos_t = cute.make_tensor(positions, cute.make_layout((Int64(rows),)))
        if thread < Int32(self.tokens):
            p = Int64(-1)
            if thread < live:
                p = Int64(pos_t[row0 + Int64(thread)])
            token_pos[thread] = p
        # Query tile: M row m = (token m / group, head kv_head * group + m % group).
        q_t = cute.make_tensor(q, cute.make_layout((Int64(rows) * Int64(self.heads * self.head),)))
        for it in cutlass.range_constexpr(_ceil(self.tile_m * self.head // 8, self.threads)):
            linear = thread + Int32(it * self.threads)
            if linear < Int32(self.tile_m * self.head // 8):
                m = linear // Int32(self.head // 8)
                dim = (linear % Int32(self.head // 8)) * Int32(8)
                t_local = m // Int32(self.group)
                valid = (t_local < live) & (m < Int32(self.tokens * self.group))
                source = Int64(q.toint()) + (((row0 + Int64(t_local)) * Int64(self.heads) + Int64(
                    kv_head * Int32(self.group) + m % Int32(self.group))) * Int64(self.head) + Int64(dim)) * Int64(2)
                source = cutlass.select_(valid, source, Int64(q.toint()))
                _cp_async_load_128b_zfill(shared_ptr_to_u32(s_q.iterator + cute.crd2idx((m, dim), s_q.layout)),
                                          source, cutlass.select_(valid, Int32(16), Int32(0)))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Key range and addressing bases of this CTA's sequence.
        p_first = Int64(token_pos[0])
        p_last = Int64(token_pos[0])
        for t in cutlass.range_constexpr(self.tokens):
            if Int32(t) < live:
                p_last = Int64(token_pos[t])
        hi = p_last
        lo = Int64(0)
        if const_expr(self.window > 0):
            lo = p_first - Int64(self.window - 1)
            if lo < Int64(0):
                lo = Int64(0)
        table_base = Int64(0)
        seq_start = Int64(0)
        step_first = Int64(0)
        ring_base = Int64(0)
        if const_expr(self.paged):
            table_base = row0 * Int64(table_stride)
        else:
            first_t = cute.make_tensor(seq_first, cute.make_layout((Int64(rows),)))
            ring_t = cute.make_tensor(ring_slots, cute.make_layout((Int64(rows),)))
            step_first = Int64(first_t[row0])
            seq_start = Int64(pos_t[step_first])
            ring_base = Int64(ring_t[row0]) - p_first % Int64(self.ring_rows)

        qk_thread = thread % Int32(self.qk_warps * 32)
        thr_qk = tiled_mma_qk.get_slice(qk_thread)
        thr_pv = tiled_mma_pv.get_slice(thread)
        r_q = thr_qk.make_fragment_A(thr_qk.partition_A(s_q))
        r_k = thr_qk.make_fragment_B(thr_qk.partition_B(s_k))
        s_vt = layout_utils.transpose_view(s_v)
        r_p = thr_pv.make_fragment_A(thr_pv.partition_A(s_p))
        r_v = thr_pv.make_fragment_B(thr_pv.partition_B(s_vt))
        atom_qk = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
        atom_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
        c_q = cute.make_tiled_copy_A(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_k = cute.make_tiled_copy_B(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_p = cute.make_tiled_copy_A(atom_qk, tiled_mma_pv).get_slice(thread)
        c_v = cute.make_tiled_copy_B(atom_v, tiled_mma_pv).get_slice(thread)
        cs_q = c_q.partition_S(s_q)
        cs_k = c_k.partition_S(s_k)
        cs_p = c_p.partition_S(s_p)
        cs_v = c_v.partition_S(s_vt)
        acc_o = cute.make_rmem_tensor(thr_pv.partition_shape_C((self.tile_m, self.v_head)), Float32)
        acc_o.fill(0.0)
        s_layout = layout_utils.convert_layout_acc_mn(
            thr_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))).layout)
        softmax = Softmax.create(Float32(self.softmax_scale * LOG2_E), cute.size(s_layout.shape[0]), arch=120)
        softmax.reset()

        tile_count = Int32((hi - lo + Int64(self.tile_n)) // Int64(self.tile_n))
        if live <= Int32(0):
            tile_count = Int32(0)
        my_tiles = (tile_count - split + splits - Int32(1)) // splits
        raw_words = max((self.k_iters + self.v_iters) * self.chunk // 4, 1) if self.kv_fp8 else 1
        raw = cute.make_rmem_tensor(cute.make_layout((raw_words,)), Uint32)
        raw_s = cute.make_rmem_tensor(cute.make_layout((max(self.k_iters + self.v_iters, 1),)), Float32)
        if my_tiles > Int32(0):
            if const_expr(self.kv_fp8):
                self._resolve(split, Int32(0), lo, hi, cache, kv_step, page_table, table_base, seq_start,
                              step_first, ring_base, key_addr, key_pos, thread)
                cute.arch.sync_threads()
                self._load_raw(key_addr, Int32(0), kv_head, cache, raw, raw_s, thread)
            self._tile(split, split + splits, my_tiles > Int32(1), Int32(0), lo, hi, row0, kv_head, cache, kv_step,
                       page_table, table_base, seq_start,
                       step_first, ring_base, token_pos, key_addr, key_pos, s_k, s_v, s_p, s_scale, thread,
                       tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q, cs_k, cs_p,
                       cs_v, acc_o, softmax, raw, raw_s, is_first=True)
            for local in cutlass.range(my_tiles - Int32(1), unroll=1):
                tile = split + (Int32(local) + Int32(1)) * splits
                buf = (Int32(local) + Int32(1)) % Int32(2)
                if const_expr(not self.kv_fp8):
                    buf = Int32(0)
                self._tile(tile, tile + splits, Int32(local) + Int32(2) < my_tiles, buf, lo, hi, row0, kv_head,
                           cache, kv_step, page_table, table_base, seq_start,
                           step_first, ring_base, token_pos, key_addr, key_pos, s_k, s_v, s_p, s_scale, thread,
                           tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q, cs_k,
                           cs_p, cs_v, acc_o, softmax, raw, raw_s, is_first=False)

        # Final scale per M row (and LSE for split partials).
        if thread < Int32(self.qk_warps * 32):
            coords = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(qk_thread).partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            if const_expr(self.direct and self.sink):
                sink_vals = cute.make_rmem_tensor(cute.size(s_layout.shape[0]), Float32)
                for m in cutlass.range_constexpr(cute.size(s_layout.shape[0])):
                    sink_head = kv_head * Int32(self.group) + coords[m, 0][0] % Int32(self.group)
                    sink_vals[m] = Float32(cute.make_tensor(sinks, cute.make_layout((self.heads,)))[sink_head])
                scale = softmax.finalize(sink_val=sink_vals)
            else:
                scale = softmax.finalize()
            if my_tiles <= Int32(0):
                for m in cutlass.range_constexpr(cute.size(s_layout.shape[0])):
                    scale[m] = Float32(0.0)
                    softmax.row_sum[m] = -Float32.inf
            if coords[0, 0][1] == Int32(0):
                for m in cutlass.range_constexpr(cute.size(softmax.row_sum)):
                    m_row = coords[m, 0][0]
                    s_scale[m_row] = scale[m]
                    if const_expr(not self.direct):
                        lse_token = m_row // Int32(self.group)
                        if (lse_token < live) & (m_row < Int32(self.tokens * self.group)):
                            lse_head = kv_head * Int32(self.group) + m_row % Int32(self.group)
                            lse_t = cute.make_tensor(partial_lse, cute.make_layout((Int64(rows) * Int64(
                                MAX_SPLITS * self.heads),)))
                            lse_t[((row0 + Int64(lse_token)) * Int64(splits) + Int64(split)) * Int64(self.heads)
                                  + Int64(lse_head)] = Float32(softmax.row_sum[m])
        cute.arch.sync_threads()

        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
            o_row = o_coords[m, 0][0]
            o_scale = Float32(s_scale[o_row])
            o_token = o_row // Int32(self.group)
            if (o_token < live) & (o_row < Int32(self.tokens * self.group)):
                o_head = Int64(kv_head * Int32(self.group) + o_row % Int32(self.group))
                row = row0 + Int64(o_token)
                for n in cutlass.range_constexpr(cute.size(o_mn.shape[1])):
                    dim = Int64(o_coords[m, n][1])
                    value = Float32(o_mn[m, n]) * o_scale
                    if const_expr(self.direct):
                        o_t = cute.make_tensor(out, cute.make_layout((Int64(rows) * Int64(self.heads * self.v_head),)))
                        o_t[(row * Int64(self.heads) + o_head) * Int64(self.v_head) + dim] = BFloat16(value)
                    else:
                        po = cute.make_tensor(partial_out, cute.make_layout((Int64(rows) * Int64(
                            MAX_SPLITS * self.heads * self.v_head),)))
                        po[((row * Int64(splits) + Int64(split)) * Int64(self.heads) + o_head) * Int64(self.v_head)
                           + dim] = value


class MimoKvWiden:
    """BF16 copy of one sequence's 8-bit full-attention records: key ``k < keys`` (paged at
    ``page_table[k / page_rows] * page_rows + k % page_rows``) to ``wide + k * R`` as
    ``bf16(q * scale)``, the values :class:`MimoGqaAttention` widens in shared memory. A
    prefill CTA re-reads every earlier key, so widening once per step and running the BF16
    attention over the copy costs one pass instead of one per CTA. ``keys_per_cta`` keys per
    CTA; each thread loads all its 8-byte pieces before widening any."""

    threads = 256

    def __init__(self, *, kv_heads: int, kv8: str, kv_group: int, page_rows: int = 64, head: int = 192,
                 v_head: int = 128, keys_per_cta: int = 8):
        if kv8 not in KV8_FORMATS:
            raise ValueError(f"kv8 is one of {KV8_FORMATS}")
        self.kv_heads, self.kv8, self.kv_group = int(kv_heads), kv8, int(kv_group)
        self.page_rows, self.head, self.v_head = int(page_rows), int(head), int(v_head)
        self.k_scales, self.v_scales = fp8_scale_counts(self.kv_group, self.head, self.v_head)
        self.record_bytes = fp8_record_bytes(self.kv_heads, self.kv_group, self.head, self.v_head)
        self.k_chunks = self.kv_heads * self.head // 8
        self.chunks = self.kv_heads * (self.head + self.v_head) // 8
        self.keys_per_cta = int(keys_per_cta)
        if page_rows % self.keys_per_cta:
            raise ValueError("keys_per_cta must divide page_rows")
        self.iters = _ceil(self.chunks * self.keys_per_cta, self.threads)

    @cute.jit
    def __call__(self, cache: cute.Pointer, page_table: cute.Pointer, wide: cute.Pointer, keys: Int32,
                 stream: cuda.CUstream):
        self.kernel(cache, page_table, wide, keys).launch(
            grid=((keys + Int32(self.keys_per_cta - 1)) // Int32(self.keys_per_cta), 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, cache: cute.Pointer, page_table: cute.Pointer, wide: cute.Pointer, keys: Int32):
        k0 = Int64(cute.arch.block_idx()[0]) * Int64(self.keys_per_cta)
        tidx = Int32(cute.arch.thread_idx()[0])
        # The CTA's keys share one page (keys_per_cta divides page_rows).
        page = Int64(cute.make_ptr(Int32, Int64(page_table.toint()) + (k0 // Int64(self.page_rows)) * Int64(4),
                                   cute.AddressSpace.gmem, assumed_align=4)[0])
        base = Int64(cache.toint()) + (page * Int64(self.page_rows) + k0 % Int64(self.page_rows)) \
            * Int64(self.record_bytes)
        wide_row = Int64(self.kv_heads * (self.head + self.v_head) * 2)
        k_group = self.head if self.kv_group == 0 else self.kv_group
        v_group = self.v_head if self.kv_group == 0 else self.kv_group
        widen = _e4m3x8_scaled_bf16 if self.kv8 == "e4m3" else _s8x8_scaled_bf16
        lo = cute.make_rmem_tensor(cute.make_layout((self.iters,)), Uint32)
        hi = cute.make_rmem_tensor(cute.make_layout((self.iters,)), Uint32)
        sc = cute.make_rmem_tensor(cute.make_layout((self.iters,)), Float32)
        for it in cutlass.range_constexpr(self.iters):
            linear = Int32(it * self.threads) + tidx
            key = linear // Int32(self.chunks)
            c = linear % Int32(self.chunks)
            live = (key < Int32(self.keys_per_cta)) & (k0 + Int64(key) < Int64(keys))
            src = base + Int64(key) * Int64(self.record_bytes)
            k_scales = src + Int64(self.kv_heads * (self.head + self.v_head))
            v_scales = k_scales + Int64(4 * self.kv_heads * self.k_scales)
            scale_at = Int64(0)
            if c < Int32(self.k_chunks):
                h = c // Int32(self.head // 8)
                d = (c % Int32(self.head // 8)) * Int32(8)
                scale_at = k_scales + Int64(4) * Int64(h * Int32(self.k_scales) + d // Int32(k_group))
            else:
                v = c - Int32(self.k_chunks)
                h = v // Int32(self.v_head // 8)
                d = (v % Int32(self.v_head // 8)) * Int32(8)
                scale_at = v_scales + Int64(4) * Int64(h * Int32(self.v_scales) + d // Int32(v_group))
            lo[it] = Uint32(0)
            hi[it] = Uint32(0)
            sc[it] = Float32(0.0)
            if live:
                a, b = ld_global_nc_v2_u32(src + Int64(c) * Int64(8))
                lo[it] = a
                hi[it] = b
                sc[it] = _ld_f32(scale_at)
        for it in cutlass.range_constexpr(self.iters):
            linear = Int32(it * self.threads) + tidx
            key = linear // Int32(self.chunks)
            c = linear % Int32(self.chunks)
            if (key < Int32(self.keys_per_cta)) & (k0 + Int64(key) < Int64(keys)):
                w0, w1, w2, w3 = widen(lo[it], hi[it], sc[it])
                st_global_v4_u32(Int64(wide.toint()) + (k0 + Int64(key)) * wide_row + Int64(c) * Int64(16),
                                 w0, w1, w2, w3)


class MimoSplitMerge:
    """Merge split partials: ``out = sum_s w_s o_s / (sum_s w_s + [exp(sink - M)])``
    with ``w_s = exp(lse_s - M)``, ``M`` the max over the splits (and the
    sink). One CTA per (head, row), one thread per value dim."""

    def __init__(self, *, heads: int, v_head: int = 128, sink: bool = False):
        self.heads, self.v_head, self.sink = int(heads), int(v_head), bool(sink)

    @cute.jit
    def __call__(self, partial_out: cute.Pointer, partial_lse: cute.Pointer, sinks: cute.Pointer,
                 out: cute.Pointer, rows: Int32, splits: Int32, stream: cuda.CUstream):
        self.kernel(partial_out, partial_lse, sinks, out, splits).launch(
            grid=(self.heads, rows, 1), block=(self.v_head, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, partial_out: cute.Pointer, partial_lse: cute.Pointer, sinks: cute.Pointer,
               out: cute.Pointer, splits: Int32):
        head_i, row_i, _ = cute.arch.block_idx()
        dim = Int64(cute.arch.thread_idx()[0])
        head, row = Int64(head_i), Int64(row_i)
        lse = cute.make_tensor(partial_lse, cute.make_layout((Int64(1) << Int64(40),)))
        po = cute.make_tensor(partial_out, cute.make_layout((Int64(1) << Int64(40),)))
        top = Float32(-Float32.inf)
        sink = Float32(0.0)
        if const_expr(self.sink):
            sink = Float32(cute.make_tensor(sinks, cute.make_layout((self.heads,)))[head])
            top = sink
        for s in cutlass.range(splits, unroll=1):
            value = Float32(lse[(row * Int64(splits) + Int64(s)) * Int64(self.heads) + head])
            if value > top:
                top = value
        total = Float32(0.0)
        acc = Float32(0.0)
        for s in cutlass.range(splits, unroll=1):
            value = Float32(lse[(row * Int64(splits) + Int64(s)) * Int64(self.heads) + head])
            if value > -Float32.inf:
                w = cute.math.exp2((value - top) * Float32(LOG2_E), fastmath=False)
                total = total + w
                acc = acc + w * Float32(po[((row * Int64(splits) + Int64(s)) * Int64(self.heads) + head)
                                           * Int64(self.v_head) + dim])
        if const_expr(self.sink):
            total = total + cute.math.exp2((sink - top) * Float32(LOG2_E), fastmath=False)
        o = cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(40),)))
        o[(row * Int64(self.heads) + head) * Int64(self.v_head) + dim] = BFloat16(acc / total)


class MimoRingCommit:
    """Copy each row's step record into its sequence's ring (``ring_slots[row]``)
    unless a later row of the same step (``seq_first`` equal) lands on the
    same slot (``row + ring_rows``). One CTA per row, 16-byte vectors."""

    threads = 128

    def __init__(self, *, record: int, ring_rows: int, record_bytes: int | None = None):
        self.record, self.ring_rows = int(record), int(ring_rows)
        self.record_bytes = self.record * 2 if record_bytes is None else int(record_bytes)
        if self.record_bytes % 16:
            raise ValueError("records must be whole 16-byte vectors")
        self.vectors = self.record_bytes // 16

    @cute.jit
    def __call__(self, kv_step: cute.Pointer, ring: cute.Pointer, ring_slots: cute.Pointer,
                 seq_first: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(kv_step, ring, ring_slots, seq_first, rows).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, kv_step: cute.Pointer, ring: cute.Pointer, ring_slots: cute.Pointer,
               seq_first: cute.Pointer, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        first = cute.make_tensor(seq_first, cute.make_layout((Int64(rows),)))
        slots = cute.make_tensor(ring_slots, cute.make_layout((Int64(rows),)))
        later = row + Int64(self.ring_rows)
        later_first = Int64(-1)
        if later < Int64(rows):
            later_first = Int64(first[later])
        slot = Int64(slots[row])
        if (later_first != Int64(first[row])) & (slot >= Int64(0)):
            src = Int64(kv_step.toint()) + row * Int64(self.record_bytes)
            dst = Int64(ring.toint()) + slot * Int64(self.record_bytes)
            for it in cutlass.range_constexpr(_ceil(self.vectors, self.threads)):
                v = Int64(it * self.threads) + tidx
                if v < Int64(self.vectors):
                    w = ld_global_v4_u32(src + v * Int64(16))
                    st_global_v4_u32(dst + v * Int64(16), w[0], w[1], w[2], w[3])


class RouterHiLoAdd:
    """``logits[r, e] = hilo[r, e] + hilo[r, E + e]``: the FP32 router weight
    split into BF16 high and low parts (``w_hi = bf16(w)``, ``w_lo = bf16(w -
    w_hi)``), each product accumulated in FP32."""

    threads = 256

    def __init__(self, experts: int):
        self.experts = int(experts)

    @cute.jit
    def __call__(self, hilo: cute.Pointer, logits: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        e = self.experts
        self.kernel(cute.make_tensor(hilo, cute.make_layout((m, 2 * e), stride=(2 * e, 1))),
                    cute.make_tensor(logits, cute.make_layout((m, e), stride=(e, 1)))).launch(
            grid=(rows, _ceil(e, self.threads), 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, hilo: cute.Tensor, logits: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.experts):
            logits[row, col] = Float32(hilo[row, col]) + Float32(hilo[row, Int64(self.experts) + col])
