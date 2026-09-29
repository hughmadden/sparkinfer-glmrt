"""CuTe DSL kernels composed into the Qwen 3.8 Flash Next (``qwen4_*``) attention programs.

Arithmetic follows transformers ``modeling_qwen4_exp`` rounding points: RMS
norms in FP32 times ``1 + w`` rounded once to BF16; RoPE on BF16 operands with
BF16 ``cos``/``sin`` (``bf16(bf16(x * c) + bf16(+-x' * s))``, NeoX halves of
the first ``rope_dim`` dims), ``cos``/``sin`` of ``inv_freq * position`` in
FP32 with the exact ``inv_freq`` table the reference builds; the QSA pooled
key is the FP32 mean of the 4 raw BF16 keys rounded to BF16, then its norm,
then RoPE at the block's first position.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    div_rn_f32,
    ld_global_v4_u32,
    pack_f32x2_to_bfloat2,
    st_global_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._glm_kernels import _rsqrt, _warp_sum

TOPK_THREADS = 512


def rope_inv_freq(rope_dim: int, theta: float) -> list[float]:
    """The reference's FP32 ``inv_freq`` (computed with torch on the CPU, as transformers does)."""
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).to(dtype=torch.float) / rope_dim))
    return [float(v) for v in inv.float().tolist()]


@cute.jit
def _bf16(value: Float32) -> Float32:
    """Round to BF16 (RNE) through the packing instruction: a plain truncf/extf pair may be folded away."""
    return _bf16_lo(pack_f32x2_to_bfloat2(value, value))


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return div_rn_f32(Float32(1.0), Float32(1.0) + cute.math.exp(-x, fastmath=False))


class _Rope:
    """``cos``/``sin`` (BF16-rounded) of frequency ``i`` at ``position``."""

    def __init__(self, rope_dim: int, theta: float):
        self.half = int(rope_dim) // 2
        self.inv = rope_inv_freq(rope_dim, theta)

    @cute.jit
    def cos_sin(self, i: Int32, position: Int64):
        inv = Float32(0.0)
        for k in cutlass.range_constexpr(self.half):
            if i == Int32(k):
                inv = Float32(self.inv[k])
        freq = inv * Float32(position)
        return _bf16(cute.math.cos(freq, fastmath=False)), _bf16(cute.math.sin(freq, fastmath=False))

    @cute.jit
    def apply(self, y: Float32, partner: Float32, dim: Int32, position: Int64) -> Float32:
        """RoPE of dim ``dim`` (``< 2 * half``) whose NeoX partner holds ``partner``."""
        c, s = self.cos_sin(dim % Int32(self.half), position)
        rotated = partner * s
        if dim < Int32(self.half):
            rotated = -rotated
        return _bf16(_bf16(y * c) + _bf16(rotated))


class Qwen4AttnPost:
    """Everything after the in-projection of one full-attention layer.

    ``proj`` BF16 ``[rows, P]`` = ``[q/gate per head (24 x 512); k (2 x 256); v
    (2 x 256); index q (4 x 128); raw index key (128)]``. One warp per
    (row, item): items ``0..23`` query heads (``q_norm`` + RoPE -> ``query``,
    the gate half copied to ``gate``), ``24..25`` KV heads (``k_norm`` + RoPE
    -> K, V copied; record ``[K 2x256 | V 2x256]`` BF16 at ``kv_slots[row]``),
    ``26..29`` index query heads (``iq_norm`` + RoPE -> ``index_q``), ``30``
    the raw index key -> ``token_keys[kv_slots[row]]``. Negative slots skip
    the cache writes.
    """

    def __init__(self, g):
        self.g = g
        self.heads, self.kv_heads, self.d = g.heads, g.kv_heads, g.head_dim
        self.ih, self.id = g.index_heads, g.index_head_dim
        self.width = g.attn_in_width
        self.k_off = 2 * self.heads * self.d
        self.v_off = self.k_off + self.kv_heads * self.d
        self.iq_off = self.v_off + self.kv_heads * self.d
        self.ik_off = self.iq_off + self.ih * self.id
        self.items = self.heads + self.kv_heads + self.ih + 1
        self.eps = float(g.norm_eps)
        self.rope = _Rope(g.rope_dim, g.rope_theta)
        if self.d != 256 or self.id != 128 or g.rope_dim != 64:
            raise ValueError("Qwen4AttnPost is built for 256-wide heads, 128-wide index heads and 64 RoPE dims")

    @cute.jit
    def __call__(self, proj: cute.Pointer, q_norm: cute.Pointer, k_norm: cute.Pointer, iq_norm: cute.Pointer,
                 ik_norm: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, kv_cache: cute.Pointer,
                 token_keys: cute.Pointer, query: cute.Pointer, gate: cute.Pointer, index_q: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(proj, q_norm, k_norm, iq_norm, ik_norm, positions, kv_slots, kv_cache, token_keys, query,
                    gate, index_q).launch(grid=(rows, self.items, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def _load8(self, address: Int64, values: cute.Tensor):
        words = ld_global_v4_u32(address)
        for i in cutlass.range_constexpr(4):
            values[2 * i] = _bf16_lo(words[i])
            values[2 * i + 1] = _bf16_hi(words[i])

    @cute.jit
    def _store8(self, address: Int64, values: cute.Tensor):
        st_global_v4_u32(address, pack_f32x2_to_bfloat2(values[0], values[1]),
                         pack_f32x2_to_bfloat2(values[2], values[3]),
                         pack_f32x2_to_bfloat2(values[4], values[5]),
                         pack_f32x2_to_bfloat2(values[6], values[7]))

    @cute.jit
    def _head256(self, src: Int64, weight: cute.Pointer, position: Int64, lane: Int32, out: Int64):
        """Norm (1 + w) + RoPE of one 256-wide head at ``src``; lane owns dims 8l..8l+7."""
        x = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        self._load8(src + Int64(lane) * Int64(16), x)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(8):
            square = square + x[j] * x[j]
        inv = _rsqrt(_warp_sum(square) / Float32(256.0) + Float32(self.eps))
        w = cute.make_tensor(weight, cute.make_layout((256,)))
        for j in cutlass.range_constexpr(8):
            x[j] = _bf16((x[j] * inv) * (Float32(1.0) + Float32(w[Int32(8) * lane + Int32(j)])))
        for j in cutlass.range_constexpr(8):
            partner = cute.arch.shuffle_sync_bfly(x[j], offset=4)
            dim = Int32(8) * lane + Int32(j)
            if lane < Int32(8):
                x[j] = self.rope.apply(x[j], partner, dim, position)
        self._store8(out + Int64(lane) * Int64(16), x)

    @cute.jit
    def _head128(self, src: cute.Tensor, weight: cute.Pointer, position: Int64, lane: Int32, out: cute.Tensor,
                 rope: cutlass.Constexpr):
        """Norm (1 + w) (+ RoPE) of one 128-wide head; lane owns dims 4l..4l+3."""
        x = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for j in cutlass.range_constexpr(4):
            x[j] = Float32(src[Int32(4) * lane + Int32(j)])
        square = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            square = square + x[j] * x[j]
        inv = _rsqrt(_warp_sum(square) / Float32(128.0) + Float32(self.eps))
        w = cute.make_tensor(weight, cute.make_layout((128,)))
        for j in cutlass.range_constexpr(4):
            x[j] = _bf16((x[j] * inv) * (Float32(1.0) + Float32(w[Int32(4) * lane + Int32(j)])))
        if cutlass.const_expr(rope):
            for j in cutlass.range_constexpr(4):
                partner = cute.arch.shuffle_sync_bfly(x[j], offset=8)
                dim = Int32(4) * lane + Int32(j)
                if lane < Int32(16):
                    x[j] = self.rope.apply(x[j], partner, dim, position)
        for j in cutlass.range_constexpr(4):
            out[Int32(4) * lane + Int32(j)] = x[j].to(BFloat16)

    @cute.kernel
    def kernel(self, proj: cute.Pointer, q_norm: cute.Pointer, k_norm: cute.Pointer, iq_norm: cute.Pointer,
               ik_norm: cute.Pointer, positions: cute.Pointer, kv_slots: cute.Pointer, kv_cache: cute.Pointer,
               token_keys: cute.Pointer, query: cute.Pointer, gate: cute.Pointer, index_q: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        item = Int32(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        pos = cute.make_tensor(positions, cute.make_layout((row + Int64(1),)))
        slots = cute.make_tensor(kv_slots, cute.make_layout((row + Int64(1),)))
        position = Int64(pos[row])
        slot = Int64(slots[row])
        base = Int64(proj.toint()) + row * Int64(self.width * 2)
        if item < Int32(self.heads):
            h = Int64(item)
            self._head256(base + h * Int64(2 * self.d * 2), q_norm, position, lane,
                          Int64(query.toint()) + (row * Int64(self.heads) + h) * Int64(self.d * 2))
            g = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
            self._load8(base + h * Int64(2 * self.d * 2) + Int64(self.d * 2) + Int64(lane) * Int64(16), g)
            self._store8(Int64(gate.toint()) + (row * Int64(self.heads) + h) * Int64(self.d * 2)
                         + Int64(lane) * Int64(16), g)
        elif item < Int32(self.heads + self.kv_heads):
            kh = Int64(item - Int32(self.heads))
            if slot >= Int64(0):
                record = Int64(kv_cache.toint()) + slot * Int64(self.g.record_bytes)
                self._head256(base + Int64(self.k_off * 2) + kh * Int64(self.d * 2), k_norm, position, lane,
                              record + kh * Int64(self.d * 2))
                v = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
                self._load8(base + Int64(self.v_off * 2) + kh * Int64(self.d * 2) + Int64(lane) * Int64(16), v)
                self._store8(record + Int64(self.kv_heads * self.d * 2) + kh * Int64(self.d * 2)
                             + Int64(lane) * Int64(16), v)
        elif item < Int32(self.heads + self.kv_heads + self.ih):
            ih = Int64(item - Int32(self.heads + self.kv_heads))
            src = cute.make_tensor(
                cute.make_ptr(BFloat16, base + Int64(self.iq_off * 2) + ih * Int64(self.id * 2),
                              cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.id,)))
            out = cute.make_tensor(
                cute.make_ptr(BFloat16, Int64(index_q.toint()) + (row * Int64(self.ih) + ih) * Int64(self.id * 2),
                              cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.id,)))
            self._head128(src, iq_norm, position, lane, out, True)
        else:
            if slot >= Int64(0):
                src = cute.make_tensor(
                    cute.make_ptr(BFloat16, base + Int64(self.ik_off * 2), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((self.id,)))
                out = cute.make_tensor(
                    cute.make_ptr(BFloat16, Int64(token_keys.toint()) + slot * Int64(self.id * 2),
                                  cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.id,)))
                for j in cutlass.range_constexpr(4):
                    out[Int32(4) * lane + Int32(j)] = src[Int32(4) * lane + Int32(j)]


class Qwen4PoolKeys:
    """QSA block key of every 4-token block a row completes (``pool_slots[row] >= 0``).

    Raw keys of the block's tokens sit at ``token_keys[kv_slots[row] - 3 ..
    kv_slots[row]]`` (a block never crosses a 64-row page). ``key =
    rope(ik_norm(bf16(mean_fp32(raw))), position - 3)`` into
    ``index_cache[pool_slots[row]]``. One warp per row.
    """

    def __init__(self, g):
        self.g = g
        self.eps = float(g.norm_eps)
        self.rope = _Rope(g.rope_dim, g.rope_theta)
        self.post = Qwen4AttnPost(g)

    @cute.jit
    def __call__(self, positions: cute.Pointer, kv_slots: cute.Pointer, pool_slots: cute.Pointer,
                 ik_norm: cute.Pointer, token_keys: cute.Pointer, index_cache: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(positions, kv_slots, pool_slots, ik_norm, token_keys, index_cache).launch(
            grid=(rows, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, positions: cute.Pointer, kv_slots: cute.Pointer, pool_slots: cute.Pointer,
               ik_norm: cute.Pointer, token_keys: cute.Pointer, index_cache: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        lane = Int32(cute.arch.thread_idx()[0])
        pool = Int64(cute.make_tensor(pool_slots, cute.make_layout((row + Int64(1),)))[row])
        if pool >= Int64(0):
            slot = Int64(cute.make_tensor(kv_slots, cute.make_layout((row + Int64(1),)))[row])
            position = Int64(cute.make_tensor(positions, cute.make_layout((row + Int64(1),)))[row])
            keys = cute.make_tensor(
                cute.make_ptr(BFloat16, Int64(token_keys.toint()) + (slot - Int64(3)) * Int64(256),
                              cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((4, 128), stride=(128, 1)))
            mean = cute.make_tensor(
                cute.make_ptr(BFloat16, Int64(index_cache.toint()) + pool * Int64(256), cute.AddressSpace.gmem,
                              assumed_align=16), cute.make_layout((128,)))
            # Mean into the destination row first (BF16), then norm + RoPE in place (same warp).
            for j in cutlass.range_constexpr(4):
                d = Int32(4) * lane + Int32(j)
                total = (Float32(keys[0, d]) + Float32(keys[1, d])) + (Float32(keys[2, d]) + Float32(keys[3, d]))
                mean[d] = (total * Float32(0.25)).to(BFloat16)
            cute.arch.sync_warp()
            self.post._head128(mean, ik_norm, position - Int64(3), lane, mean, True)


class Qwen4BlockTopK:
    """Top ``k`` of ``n = min((positions[row] + 1) // 4, max_groups)`` FP32 block scores per row.

    ``scores`` FP32 ``[rows, score_stride]`` (non-negative for eligible
    blocks); ``out`` i32 ``[rows, k]``: the selected block ids in ascending
    order, ``-1`` padded (every block when ``n <= k``). Exact radix select on
    the score bits; ties at the threshold keep the lowest block ids. One CTA of
    512 threads per row.
    """

    threads = TOPK_THREADS

    def __init__(self, *, k: int, block: int, max_groups: int):
        self.k, self.block, self.max_groups = int(k), int(block), int(max_groups)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, positions: cute.Pointer, scores: cute.Pointer, out: cute.Pointer, rows: Int32,
                 score_stride: Int64, stream: cuda.CUstream):
        self.kernel(positions, scores, out, score_stride).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, positions: cute.Pointer, scores: cute.Pointer, out: cute.Pointer, score_stride: Int64):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        lane = tid % Int32(32)
        warp = tid // Int32(32)
        smem = cutlass_utils.SmemAllocator()
        hist = smem.allocate_tensor(Int32, cute.make_layout((256,)), byte_alignment=16)
        warp_eq = smem.allocate_tensor(Int32, cute.make_layout((self.warps,)), byte_alignment=16)
        warp_sel = smem.allocate_tensor(Int32, cute.make_layout((self.warps,)), byte_alignment=16)
        state = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        position = Int64(cute.make_tensor(positions, cute.make_layout((row + Int64(1),)))[row])
        n = Int32(cutlass.min((position + Int64(1)) // Int64(self.block), Int64(self.max_groups)))
        s = cute.make_tensor(cute.make_ptr(Uint32, Int64(scores.toint()) + row * score_stride * Int64(4),
                                           cute.AddressSpace.gmem, assumed_align=4),
                             cute.make_layout((self.max_groups,)))
        o = cute.make_tensor(cute.make_ptr(Int32, Int64(out.toint()) + row * Int64(self.k * 4),
                                           cute.AddressSpace.gmem, assumed_align=4), cute.make_layout((self.k,)))
        if n <= Int32(self.k):
            for it in cutlass.range_constexpr((self.k + self.threads - 1) // self.threads):
                i = Int32(it * self.threads) + tid
                if i < Int32(self.k):
                    value = Int32(-1)
                    if i < n:
                        value = i
                    o[i] = value
        else:
            prefix = Uint32(0)
            mask = Uint32(0)
            remaining = Int32(self.k)
            for p in cutlass.range_constexpr(4):
                shift = 24 - 8 * p
                if tid < Int32(256):
                    hist[tid] = Int32(0)
                cute.arch.sync_threads()
                i = tid
                while i < n:
                    key = Uint32(s[i])
                    if (key & mask) == prefix:
                        cute.arch.atomic_add(hist.iterator + Int32((key >> Uint32(shift)) & Uint32(255)), Int32(1))
                    i += Int32(self.threads)
                cute.arch.sync_threads()
                if tid == Int32(0):
                    above = Int32(0)
                    chosen = Int32(0)
                    found = Int32(0)
                    b = Int32(255)
                    while b >= Int32(0):
                        c = hist[b]
                        if found == Int32(0):
                            if above + c >= remaining:
                                chosen = b
                                found = Int32(1)
                            else:
                                above = above + c
                        b = b - Int32(1)
                    state[0] = chosen
                    state[1] = remaining - above
                cute.arch.sync_threads()
                prefix = prefix | (Uint32(state[0]) << Uint32(shift))
                mask = mask | (Uint32(255) << Uint32(shift))
                remaining = state[1]
                cute.arch.sync_threads()
            # Emit in ascending id order: keys above the threshold, then the first `remaining` ties.
            emitted = Int32(0)
            ties = Int32(0)
            start = Int32(0)
            while start < n:
                i = start + tid
                key = Uint32(0)
                valid = i < n
                if valid:
                    key = Uint32(s[i])
                is_eq = valid & (key == prefix)
                is_gt = valid & (key > prefix)
                eq_ballot = Uint32(cute.arch.vote_ballot_sync(is_eq))
                lower = (Uint32(1) << Uint32(lane)) - Uint32(1)
                eq_before_lane = Int32(cute.arch.popc(eq_ballot & lower))
                if lane == Int32(0):
                    warp_eq[warp] = Int32(cute.arch.popc(eq_ballot))
                cute.arch.sync_threads()
                eq_before = ties
                eq_total = Int32(0)
                for w in cutlass.range_constexpr(self.warps):
                    c = warp_eq[w]
                    if Int32(w) < warp:
                        eq_before = eq_before + c
                    eq_total = eq_total + c
                take = is_gt | (is_eq & ((eq_before + eq_before_lane) < remaining))
                sel_ballot = Uint32(cute.arch.vote_ballot_sync(take))
                sel_before_lane = Int32(cute.arch.popc(sel_ballot & lower))
                if lane == Int32(0):
                    warp_sel[warp] = Int32(cute.arch.popc(sel_ballot))
                cute.arch.sync_threads()
                sel_before = emitted
                sel_total = Int32(0)
                for w in cutlass.range_constexpr(self.warps):
                    c = warp_sel[w]
                    if Int32(w) < warp:
                        sel_before = sel_before + c
                    sel_total = sel_total + c
                if take:
                    o[sel_before + sel_before_lane] = i
                emitted = emitted + sel_total
                ties = ties + eq_total
                start = start + Int32(self.threads)
                cute.arch.sync_threads()


class Qwen4IndexExpand:
    """Selected logical token positions per row (``width`` wide, ``-1`` padded) and their count.

    Rows with ``p + 1 <= dense_limit`` select every token ``0..p``; longer
    rows expand their top-k blocks (``blocks [rows, pools]``, ascending ids)
    to 4 tokens each, then append the open tail ``(p + 1) // 4 * 4 .. p``.
    One CTA of 256 threads per row.
    """

    threads = 256

    def __init__(self, *, pools: int, width: int, dense_limit: int, kpool: int = 4):
        self.pools, self.width, self.dense_limit, self.kpool = int(pools), int(width), int(dense_limit), int(kpool)

    @cute.jit
    def __call__(self, positions: cute.Pointer, blocks: cute.Pointer, indices: cute.Pointer, lengths: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(positions, cute.make_layout((m,))),
            cute.make_tensor(blocks, cute.make_layout((m, self.pools), stride=(self.pools, 1))),
            cute.make_tensor(indices, cute.make_layout((m, self.width), stride=(self.width, 1))),
            cute.make_tensor(lengths, cute.make_layout((m,))),
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, positions: cute.Tensor, blocks: cute.Tensor, indices: cute.Tensor, lengths: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        p = Int64(positions[row])
        count = p + Int64(1)
        if count <= Int64(self.dense_limit):
            for it in cutlass.range_constexpr((self.width + self.threads - 1) // self.threads):
                i = Int64(it * self.threads) + tidx
                if i < Int64(self.width):
                    value = Int32(-1)
                    if i < count:
                        value = Int32(i)
                    indices[row, i] = value
            if tidx == Int64(0):
                lengths[row] = Int32(count)
        else:
            for it in cutlass.range_constexpr((self.pools + self.threads - 1) // self.threads):
                j = Int64(it * self.threads) + tidx
                if j < Int64(self.pools):
                    b = Int64(blocks[row, j])
                    for e in cutlass.range_constexpr(4):
                        value = Int32(-1)
                        if b >= Int64(0):
                            value = Int32(b * Int64(self.kpool) + Int64(e))
                        indices[row, j * Int64(4) + Int64(e)] = value
            tail_start = count // Int64(self.kpool) * Int64(self.kpool)
            tail = count - tail_start
            filled = Int64(self.pools * self.kpool)
            if tidx < Int64(self.width) - filled:
                value = Int32(-1)
                if tidx < tail:
                    value = Int32(tail_start + tidx)
                indices[row, filled + tidx] = value
            if tidx == Int64(0):
                lengths[row] = Int32(filled + tail)


class Qwen4RowTables:
    """Per-row request ids (``row``, the page-table row; ``stride == 0`` rows all read row 0's
    table through a zero stride) and token counts ``positions + 1`` for the b12x kernels."""

    threads = 128

    @cute.jit
    def __call__(self, positions: cute.Pointer, requests: cute.Pointer, lengths: cute.Pointer, rows: Int32,
                 shared: Int32, stream: cuda.CUstream):
        self.kernel(positions, requests, lengths, rows, shared).launch(
            grid=((rows + Int32(self.threads - 1)) // Int32(self.threads), 1, 1), block=(self.threads, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, positions: cute.Pointer, requests: cute.Pointer, lengths: cute.Pointer, rows: Int32,
               shared: Int32):
        i = Int32(cute.arch.block_idx()[0]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        if i < rows:
            pos = cute.make_tensor(positions, cute.make_layout((rows,)))
            req = cute.make_tensor(requests, cute.make_layout((rows,)))
            lens = cute.make_tensor(lengths, cute.make_layout((rows,)))
            value = i
            if shared != Int32(0):
                value = Int32(0)
            req[i] = value
            lens[i] = Int32(Int64(pos[i]) + Int64(1))


class Qwen4GateMul:
    """``out = bf16(attn * bf16(sigmoid(gate)))`` over BF16 ``[rows, width]``."""

    threads = 256

    def __init__(self, width: int):
        self.width = int(width)
        if self.width % (8 * self.threads):
            raise ValueError("width must be a multiple of 2048")
        self.vectors = self.width // (8 * self.threads)

    @cute.jit
    def __call__(self, attn: cute.Pointer, gate: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(attn, gate, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, attn: cute.Pointer, gate: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        g = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        for v in cutlass.range_constexpr(self.vectors):
            offset = row * Int64(self.width * 2) + (Int64(v * self.threads) + tidx) * Int64(16)
            wa = ld_global_v4_u32(Int64(attn.toint()) + offset)
            wg = ld_global_v4_u32(Int64(gate.toint()) + offset)
            for i in cutlass.range_constexpr(4):
                a[2 * i] = _bf16_lo(wa[i])
                a[2 * i + 1] = _bf16_hi(wa[i])
                g[2 * i] = _bf16_lo(wg[i])
                g[2 * i + 1] = _bf16_hi(wg[i])
            for i in cutlass.range_constexpr(8):
                a[i] = _bf16(a[i] * _bf16(_sigmoid(g[i])))
            st_global_v4_u32(Int64(out.toint()) + offset, pack_f32x2_to_bfloat2(a[0], a[1]),
                             pack_f32x2_to_bfloat2(a[2], a[3]), pack_f32x2_to_bfloat2(a[4], a[5]),
                             pack_f32x2_to_bfloat2(a[6], a[7]))


__all__ = [
    "Qwen4AttnPost",
    "Qwen4BlockTopK",
    "Qwen4GateMul",
    "Qwen4IndexExpand",
    "Qwen4PoolKeys",
    "Qwen4RowTables",
    "rope_inv_freq",
]
