"""K3 dense-MLA prefill that reads the compact GLM-format FP8 latent record.

Record (656 bytes, the shared GLM/DeepSeek MLA latent ABI):
    [  0, 512)  E4M3 latent, natural dimension order, four 128-wide groups
    [512, 528)  four FP32 group scales (dequantized value = e4m3 * scale)
    [528, 656)  64 BF16 raw key tail

Arithmetic stays BF16 tensor-core math with exact widening: E4M3 widens to
BF16 without rounding, the per-(candidate, group) FP32 scale is applied after
each group's QK MMA (GLM style), and the PV side folds the scale into the BF16
probability operand (each value warp owns one 128-wide group).  Heads are the
MMA M dimension (sixteen per tile), so the value accumulators and the online
softmax statistics of a head live in the same lanes and the A operand of the
value MMA is loaded once per chunk and reused across all value columns.

The KV ring holds whole records exactly as stored; a chunk never crosses a
64-token page, so a full chunk is a single bulk copy.  A chunk that reaches
past ``cache_length`` falls back to per-record copies that substitute record
zero for the invalid tail (masked by the consumer, never NaN-poisoning PV).

Serves the single-request, non-windowed causal extend route only (one split).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    fmax_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_v4_f32,
    ldmatrix_m16n16x1_trans_b8,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    prefetch_bulk_global_l2,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._math import exp2_approx, log2_approx

RECORD_BYTES = 656
LATENT_DIM = 512
SCALE_OFFSET = 512
TAIL_OFFSET = 528
QK_DIM = 576
VALUE_DIM = 512
GROUP = 128
HEADS_PER_TILE = 16
MATH_WARPS = 4
# BF16 query row in shared memory: 576 dims + 8 elements so successive head
# rows rotate across bank groups for ldmatrix (1168 = 9*128 + 16).
Q_ROW_BYTES = QK_DIM * 2 + 16
_MASK = -1.0e30


def smem_bytes(*, chunk: int, stages: int, query_tile: int) -> int:
    kv = stages * chunk * RECORD_BYTES
    q = query_tile * HEADS_PER_TILE * Q_ROW_BYTES
    p = query_tile * MATH_WARPS * HEADS_PER_TILE * (chunk + 8) * 2
    reduce = query_tile * 2 * MATH_WARPS * HEADS_PER_TILE * 4
    return kv + q + p + reduce + 2 * stages * 8 + 4 * 128


class DenseMlaFp8PrefillKernel:
    def __init__(
        self,
        *,
        num_heads: int,
        page_size: int = 64,
        chunk: int = 64,
        stages: int = 1,
        query_tile: int = 1,
        prefetch: int = 0,
        min_blocks: int = 1,
    ):
        if num_heads % HEADS_PER_TILE:
            raise ValueError("FP8 MLA prefill needs a multiple of sixteen heads")
        if chunk not in (32, 64) or page_size % chunk:
            raise ValueError("chunk must be 32 or 64 and divide the page")
        if stages not in (1, 2, 3) or query_tile not in (1, 2):
            raise ValueError("unsupported FP8 MLA prefill ring")
        self.num_heads = int(num_heads)
        self.page_size = int(page_size)
        self.chunk = int(chunk)
        self.stages = int(stages)
        self.query_tile = int(query_tile)
        self.prefetch = int(prefetch)
        self.min_blocks = int(min_blocks)
        self.head_tiles = self.num_heads // HEADS_PER_TILE
        # Candidates per warp in the score MMA: n8 tiles of eight records.
        self.warp_cands = self.chunk // MATH_WARPS
        self.ntiles = self.warp_cands // 8
        self.math_threads = 32 * MATH_WARPS * self.query_tile
        self.block_threads = self.math_threads + 32
        self.p_stride = (self.chunk + 8) * 2
        self.p_group_bytes = HEADS_PER_TILE * self.p_stride
        self.kv_stage_bytes = self.chunk * RECORD_BYTES
        total = smem_bytes(chunk=chunk, stages=stages, query_tile=query_tile)
        if total > 101376:
            raise ValueError(f"FP8 MLA prefill tile needs {total} bytes of SMEM")

    def _storage(self):
        qt = self.query_tile

        class Storage:
            pass

        Storage.__annotations__ = {
            "kv": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.stages * self.kv_stage_bytes], 128
            ],
            "q": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, qt * HEADS_PER_TILE * Q_ROW_BYTES], 128
            ],
            "p": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, qt * MATH_WARPS * self.p_group_bytes], 128
            ],
            "reduce": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, qt * 2 * MATH_WARPS * HEADS_PER_TILE], 16
            ],
            "mbar": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint64, 2 * self.stages], 16
            ],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        page_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        query_tiles = (total_q + Int32(self.query_tile - 1)) // Int32(self.query_tile)
        if self.query_tile > 1:
            query_tiles = query_tiles + batch
        self.kernel(
            q_bytes, cache_bytes, page_table, cache_seqlens, cu_seqlens_q,
            output, final_lse, sm_scale_log2, q_stride_row_bytes,
            page_stride_bytes, page_table_stride, total_q, batch,
        ).launch(
            grid=(query_tiles, self.head_tiles, 1),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    # ------------------------------------------------------------------ math
    @cute.jit
    def _stage_query(
        self, q_bytes: cute.Tensor, q_addr: Int32, query_row: Int32, head_base: Int32,
        total_q: Int32, math_tid: Int32, q_stride_row_bytes: Int64,
    ):
        """Stage sixteen BF16 query heads, permuting each latent 16-dim block.

        Within a latent block the score MMA's k-slots (2t, 2t+1, 2t+8, 2t+9)
        address physical dimensions (4t .. 4t+3), the byte order a
        non-transposed ldmatrix delivers for E4M3 keys.  As 32-bit words the
        permutation is [w0, w2, w4, w6, w1, w3, w5, w7].  The tail is natural.
        """
        group = math_tid // Int32(32 * MATH_WARPS)
        tid = math_tid - group * Int32(32 * MATH_WARPS)
        row = query_row + group
        safe_row = row
        if row >= total_q:
            safe_row = Int32(0)
        units = HEADS_PER_TILE * (QK_DIM // 16)
        unit = tid
        while unit < Int32(units):
            head = unit // Int32(QK_DIM // 16)
            block = unit - head * Int32(QK_DIM // 16)
            src = get_ptr_as_int64(
                q_bytes,
                safe_row.to(Int64) * q_stride_row_bytes
                + (head_base + head).to(Int64) * Int64(QK_DIM * 2)
                + (block * Int32(32)).to(Int64),
            )
            w0, w1, w2, w3 = ld_global_nc_v4_u32(src)
            w4, w5, w6, w7 = ld_global_nc_v4_u32(src + Int64(16))
            dst = (
                q_addr
                + group * Int32(HEADS_PER_TILE * Q_ROW_BYTES)
                + head * Int32(Q_ROW_BYTES)
                + block * Int32(32)
            )
            if block < Int32(LATENT_DIM // 16):
                st_shared_v4_u32(dst, w0, w2, w4, w6)
                st_shared_v4_u32(dst + Int32(16), w1, w3, w5, w7)
            else:
                st_shared_v4_u32(dst, w0, w1, w2, w3)
                st_shared_v4_u32(dst + Int32(16), w4, w5, w6, w7)
            unit += Int32(32 * MATH_WARPS)

    @cute.jit
    def _scores(self, q_addr: Int32, kv_addr: Int32, warp: Int32, lane: Int32):
        """Sixteen heads x this warp's candidates; returns 4*ntiles scores."""
        gid = lane >> Int32(2)
        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col_bytes = (lane >> Int32(4)) * Int32(16)
        q_lane = q_addr + a_row * Int32(Q_ROW_BYTES) + a_col_bytes
        b_row = lane & Int32(7)
        b_col = (lane >> Int32(3)) * Int32(16)
        first = warp * Int32(self.warp_cands)
        k_lane = kv_addr + (first + b_row) * Int32(RECORD_BYTES) + b_col
        tid = lane & Int32(3)
        qk = [Float32(0.0) for _ in range(4 * self.ntiles)]
        for g in cutlass.range_constexpr(LATENT_DIM // GROUP):
            part = [Float32(0.0) for _ in range(4 * self.ntiles)]
            for slab in cutlass.range_constexpr(GROUP // 64):
                base = g * GROUP + slab * 64
                kregs = []
                for nt in cutlass.range_constexpr(self.ntiles):
                    k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(
                        k_lane + Int32(nt * 8 * RECORD_BYTES + base)
                    )
                    kregs.append((k0, k1, k2, k3))
                for blk in cutlass.range_constexpr(4):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        q_lane + Int32((base + blk * 16) * 2)
                    )
                    for nt in cutlass.range_constexpr(self.ntiles):
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(kregs[nt][blk])
                        (
                            part[nt * 4], part[nt * 4 + 1],
                            part[nt * 4 + 2], part[nt * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            part[nt * 4], part[nt * 4 + 1],
                            part[nt * 4 + 2], part[nt * 4 + 3],
                            a0, a1, a2, a3, b0, b1,
                        )
            for nt in cutlass.range_constexpr(self.ntiles):
                c0 = first + Int32(nt * 8) + tid * Int32(2)
                s0 = ld_shared_f32(
                    kv_addr + c0 * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET + 4 * g)
                )
                s1 = ld_shared_f32(
                    kv_addr + (c0 + Int32(1)) * Int32(RECORD_BYTES)
                    + Int32(SCALE_OFFSET + 4 * g)
                )
                qk[nt * 4] = qk[nt * 4] + part[nt * 4] * s0
                qk[nt * 4 + 1] = qk[nt * 4 + 1] + part[nt * 4 + 1] * s1
                qk[nt * 4 + 2] = qk[nt * 4 + 2] + part[nt * 4 + 2] * s0
                qk[nt * 4 + 3] = qk[nt * 4 + 3] + part[nt * 4 + 3] * s1
        # BF16 raw tail, natural order, unscaled.
        for slab in cutlass.range_constexpr(2):
            kregs = []
            for nt in cutlass.range_constexpr(self.ntiles):
                k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(
                    k_lane + Int32(nt * 8 * RECORD_BYTES + TAIL_OFFSET + slab * 64)
                )
                kregs.append((k0, k1, k2, k3))
            for half in cutlass.range_constexpr(2):
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    q_lane + Int32((LATENT_DIM + slab * 32 + half * 16) * 2)
                )
                for nt in cutlass.range_constexpr(self.ntiles):
                    (
                        qk[nt * 4], qk[nt * 4 + 1], qk[nt * 4 + 2], qk[nt * 4 + 3],
                    ) = mma_m16n8k16_f32_bf16(
                        qk[nt * 4], qk[nt * 4 + 1], qk[nt * 4 + 2], qk[nt * 4 + 3],
                        a0, a1, a2, a3,
                        kregs[nt][half * 2], kregs[nt][half * 2 + 1],
                    )
        return qk

    @cute.jit
    def _run_group(
        self, q_addr: Int32, kv_base: Int32, p_base: Int32, red_addr: Int32,
        mbar_base, active_chunks: Int32, first_chunk: Int32, visible_end: Int32,
        query_row: Int32, query_valid: Int32, head_base: Int32,
        warp: Int32, lane: Int32, scale_log2: Float32,
        output: cute.Tensor, final_lse: cute.Tensor,
        *, barrier_id: cutlass.Constexpr,
    ):
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        group_threads = 32 * MATH_WARPS
        acc_frag = cute.make_rmem_tensor(64, Float32)
        for i in cutlass.range_constexpr(64):
            acc_frag[i] = Float32(0.0)
        stat = cute.make_rmem_tensor(4, Float32)  # m(gid), m(gid+8), l(gid), l(gid+8)
        stat[0] = Float32(_MASK)
        stat[1] = Float32(_MASK)
        stat[2] = Float32(0.0)
        stat[3] = Float32(0.0)
        red_max = red_addr
        # V operand: lane l (mod 16) supplies record row pi(l) so that the
        # transposed byte ldmatrix lands in the m16n8k16 B layout.
        l16 = lane & Int32(15)
        v_row = (
            (l16 & Int32(1))
            | ((l16 & Int32(4)) >> Int32(1))
            | ((l16 & Int32(8)) >> Int32(1))
            | ((l16 & Int32(2)) << Int32(2))
        )
        p_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        p_col = (lane >> Int32(4)) * Int32(16)
        phase = Int32(0)
        stage = Int32(0)
        for local in cutlass.range(active_chunks, unroll=1):
            chunk = first_chunk + Int32(local)
            chunk_begin = chunk * Int32(self.chunk)
            kv_addr = kv_base + stage * Int32(self.kv_stage_bytes)
            cute.arch.mbarrier_wait(mbar_base + stage, phase=phase)
            if query_valid != Int32(0) and chunk_begin < visible_end:
                qk = self._scores(q_addr, kv_addr, warp, lane)
                # causal mask + base-2 scale; local max per head row
                mx0 = Float32(_MASK)
                mx1 = Float32(_MASK)
                for nt in cutlass.range_constexpr(self.ntiles):
                    c0 = (
                        chunk_begin + warp * Int32(self.warp_cands)
                        + Int32(nt * 8) + tid * Int32(2)
                    )
                    if c0 >= visible_end:
                        qk[nt * 4] = Float32(_MASK)
                        qk[nt * 4 + 2] = Float32(_MASK)
                    if c0 + Int32(1) >= visible_end:
                        qk[nt * 4 + 1] = Float32(_MASK)
                        qk[nt * 4 + 3] = Float32(_MASK)
                    for r in cutlass.range_constexpr(4):
                        qk[nt * 4 + r] = qk[nt * 4 + r] * scale_log2
                    mx0 = fmax_f32(mx0, fmax_f32(qk[nt * 4], qk[nt * 4 + 1]))
                    mx1 = fmax_f32(mx1, fmax_f32(qk[nt * 4 + 2], qk[nt * 4 + 3]))
                for off in (1, 2):
                    mx0 = fmax_f32(mx0, cute.arch.shuffle_sync_bfly(mx0, offset=off))
                    mx1 = fmax_f32(mx1, cute.arch.shuffle_sync_bfly(mx1, offset=off))
                if tid == Int32(0):
                    st_shared_f32(red_max + (warp * Int32(16) + gid) * Int32(4), mx0)
                    st_shared_f32(red_max + (warp * Int32(16) + gid + Int32(8)) * Int32(4), mx1)
                cute.arch.barrier(barrier_id=barrier_id, number_of_threads=group_threads)
                bm0 = stat[0]
                bm1 = stat[1]
                for w in cutlass.range_constexpr(MATH_WARPS):
                    bm0 = fmax_f32(bm0, ld_shared_f32(red_max + Int32((w * 16) * 4) + gid * Int32(4)))
                    bm1 = fmax_f32(bm1, ld_shared_f32(red_max + Int32((w * 16 + 8) * 4) + gid * Int32(4)))
                alpha0 = exp2_approx(stat[0] - bm0)
                alpha1 = exp2_approx(stat[1] - bm1)
                stat[0] = bm0
                stat[1] = bm1
                sum0 = Float32(0.0)
                sum1 = Float32(0.0)
                prob = [Float32(0.0) for _ in range(4 * self.ntiles)]
                for nt in cutlass.range_constexpr(self.ntiles):
                    prob[nt * 4] = exp2_approx(qk[nt * 4] - bm0)
                    prob[nt * 4 + 1] = exp2_approx(qk[nt * 4 + 1] - bm0)
                    prob[nt * 4 + 2] = exp2_approx(qk[nt * 4 + 2] - bm1)
                    prob[nt * 4 + 3] = exp2_approx(qk[nt * 4 + 3] - bm1)
                    sum0 = sum0 + prob[nt * 4] + prob[nt * 4 + 1]
                    sum1 = sum1 + prob[nt * 4 + 2] + prob[nt * 4 + 3]
                stat[2] = stat[2] * alpha0 + sum0
                stat[3] = stat[3] * alpha1 + sum1
                # Scaled probability copies: P_g[h][c] = p[h][c] * scale[c][g].
                for nt in cutlass.range_constexpr(self.ntiles):
                    lc = warp * Int32(self.warp_cands) + Int32(nt * 8) + tid * Int32(2)
                    sa0, sa1, sa2, sa3 = ld_shared_v4_f32(
                        kv_addr + lc * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET)
                    )
                    sb0, sb1, sb2, sb3 = ld_shared_v4_f32(
                        kv_addr + (lc + Int32(1)) * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET)
                    )
                    sa = (sa0, sa1, sa2, sa3)
                    sb = (sb0, sb1, sb2, sb3)
                    for g in cutlass.range_constexpr(MATH_WARPS):
                        row0 = (
                            p_base + Int32(g * self.p_group_bytes)
                            + gid * Int32(self.p_stride) + lc * Int32(2)
                        )
                        st_shared_u32(
                            row0,
                            pack_f32x2_to_bfloat2(prob[nt * 4] * sa[g], prob[nt * 4 + 1] * sb[g]),
                        )
                        st_shared_u32(
                            row0 + Int32(8 * self.p_stride),
                            pack_f32x2_to_bfloat2(prob[nt * 4 + 2] * sa[g], prob[nt * 4 + 3] * sb[g]),
                        )
                acc = [acc_frag[i] for i in range(64)]
                for j in cutlass.range_constexpr(16):
                    acc[j * 4] = acc[j * 4] * alpha0
                    acc[j * 4 + 1] = acc[j * 4 + 1] * alpha0
                    acc[j * 4 + 2] = acc[j * 4 + 2] * alpha1
                    acc[j * 4 + 3] = acc[j * 4 + 3] * alpha1
                cute.arch.barrier(barrier_id=barrier_id, number_of_threads=group_threads)
                # Value MMA: this warp owns latent group ``warp`` (128 dims).
                p_lane = p_base + warp * Int32(self.p_group_bytes) + p_row * Int32(self.p_stride) + p_col
                for ks in cutlass.range_constexpr(self.chunk // 16):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(p_lane + Int32(ks * 32))
                    v_lane = (
                        kv_addr + (Int32(ks * 16) + v_row) * Int32(RECORD_BYTES)
                        + warp * Int32(GROUP)
                    )
                    for blk in cutlass.range_constexpr(GROUP // 16):
                        f0, f1 = ldmatrix_m16n16x1_trans_b8(v_lane + Int32(blk * 16))
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                        b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                        n0 = blk * 2
                        (
                            acc[n0 * 4], acc[n0 * 4 + 1], acc[n0 * 4 + 2], acc[n0 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc[n0 * 4], acc[n0 * 4 + 1], acc[n0 * 4 + 2], acc[n0 * 4 + 3],
                            a0, a1, a2, a3, b0, b1,
                        )
                        n1 = blk * 2 + 1
                        (
                            acc[n1 * 4], acc[n1 * 4 + 1], acc[n1 * 4 + 2], acc[n1 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc[n1 * 4], acc[n1 * 4 + 1], acc[n1 * 4 + 2], acc[n1 * 4 + 3],
                            a0, a1, a2, a3, b2, b3,
                        )
                for i in cutlass.range_constexpr(64):
                    acc_frag[i] = acc[i]
            # Every warp releases the stage once its last KV read is done.
            cute.arch.sync_warp()
            if lane == Int32(0):
                cute.arch.mbarrier_arrive(mbar_base + Int32(self.stages) + stage)
            stage += Int32(1)
            if stage == Int32(self.stages):
                stage = Int32(0)
                phase ^= Int32(1)

        # Epilogue: combine per-lane partial row sums, normalize, store.
        l0 = stat[2]
        l1 = stat[3]
        for off in (1, 2):
            l0 = l0 + cute.arch.shuffle_sync_bfly(l0, offset=off)
            l1 = l1 + cute.arch.shuffle_sync_bfly(l1, offset=off)
        red_sum = red_addr + Int32(MATH_WARPS * 16 * 4)
        if tid == Int32(0):
            st_shared_f32(red_sum + (warp * Int32(16) + gid) * Int32(4), l0)
            st_shared_f32(red_sum + (warp * Int32(16) + gid + Int32(8)) * Int32(4), l1)
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=group_threads)
        t0 = Float32(0.0)
        t1 = Float32(0.0)
        for w in cutlass.range_constexpr(MATH_WARPS):
            t0 = t0 + ld_shared_f32(red_sum + Int32((w * 16) * 4) + gid * Int32(4))
            t1 = t1 + ld_shared_f32(red_sum + Int32((w * 16 + 8) * 4) + gid * Int32(4))
        inv0 = Float32(0.0)
        inv1 = Float32(0.0)
        if t0 > Float32(0.0):
            inv0 = Float32(1.0) / t0
        if t1 > Float32(0.0):
            inv1 = Float32(1.0) / t1
        if query_valid != Int32(0):
            h0 = head_base + gid
            h1 = h0 + Int32(8)
            out0 = get_ptr_as_int64(
                output,
                (query_row.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                + (warp * Int32(GROUP) + tid * Int32(2)).to(Int64),
            )
            out1 = out0 + Int64(8 * VALUE_DIM * 2)
            for j in cutlass.range_constexpr(16):
                st_global_u32(
                    out0 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc_frag[j * 4] * inv0, acc_frag[j * 4 + 1] * inv0),
                )
                st_global_u32(
                    out1 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc_frag[j * 4 + 2] * inv1, acc_frag[j * 4 + 3] * inv1),
                )
            if warp == Int32(0) and tid == Int32(0):
                lse0 = Float32(-Float32.inf)
                lse1 = Float32(-Float32.inf)
                if t0 > Float32(0.0):
                    lse0 = (log2_approx(t0) + stat[0]) * Float32(0.6931471805599453)
                if t1 > Float32(0.0):
                    lse1 = (log2_approx(t1) + stat[1]) * Float32(0.6931471805599453)
                final_lse[query_row, h0] = lse0
                final_lse[query_row, h1] = lse1

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        page_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)
        tile_index, head_tile, _ = cute.arch.block_idx()
        tile_index = Int32(tile_index)
        head_base = Int32(head_tile) * Int32(HEADS_PER_TILE)

        request = Int32(0)
        query_start = Int32(0)
        tile_valid = Int32(1)
        if cutlass.const_expr(self.query_tile == 1):
            query_start = tile_index
            lower = Int32(0)
            upper = batch
            while lower + Int32(1) < upper:
                middle = (lower + upper) // Int32(2)
                if query_start >= Int32(cu_seqlens_q[middle]):
                    lower = middle
                else:
                    upper = middle
            request = lower
        else:
            remaining = tile_index
            scan = Int32(0)
            found = Int32(0)
            tile_offset = Int32(0)
            while scan < batch:
                if found == Int32(0):
                    scan_begin = Int32(cu_seqlens_q[scan])
                    scan_end = Int32(cu_seqlens_q[scan + Int32(1)])
                    scan_tiles = (scan_end - scan_begin + Int32(self.query_tile - 1)) // Int32(
                        self.query_tile
                    )
                    if remaining < scan_tiles:
                        request = scan
                        tile_offset = remaining
                        found = Int32(1)
                    else:
                        remaining -= scan_tiles
                scan += Int32(1)
            tile_valid = found
            query_start = Int32(cu_seqlens_q[request]) + tile_offset * Int32(self.query_tile)
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
        query_length = query_end - query_begin
        cache_length = Int32(cache_seqlens[request])
        causal_end = cache_length - query_length + (query_start - query_begin) + Int32(
            self.query_tile
        )
        if causal_end < Int32(0):
            causal_end = Int32(0)
        if causal_end > cache_length:
            causal_end = cache_length
        active_chunks = (causal_end + Int32(self.chunk - 1)) // Int32(self.chunk)
        if tile_valid == Int32(0):
            active_chunks = Int32(0)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        kv_addr = shared_ptr_to_u32(storage.kv.data_ptr())
        q_addr = shared_ptr_to_u32(storage.q.data_ptr())
        p_addr = shared_ptr_to_u32(storage.p.data_ptr())
        red_addr = shared_ptr_to_u32(storage.reduce.data_ptr())
        mbar_base = storage.mbar.data_ptr()
        if tid == Int32(0):
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))
                cute.arch.mbarrier_init(
                    mbar_base + Int32(self.stages + s), Int32(MATH_WARPS * self.query_tile)
                )
        cute.arch.barrier()

        if warp_id == Int32(MATH_WARPS * self.query_tile):
            producer_phase = Int32(1)
            stage = Int32(0)
            table_base = request.to(Int64) * page_table_stride
            for local in cutlass.range(active_chunks, unroll=1):
                token_begin = Int32(local) * Int32(self.chunk)
                cute.arch.mbarrier_wait(
                    mbar_base + Int32(self.stages) + stage, phase=producer_phase
                )
                full = mbar_base + stage
                dst = kv_addr + stage * Int32(self.kv_stage_bytes)
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full, Int32(self.chunk * RECORD_BYTES)
                    )
                if token_begin + Int32(self.chunk) <= cache_length:
                    if lane == Int32(0):
                        page = token_begin // Int32(self.page_size)
                        physical = Int32(page_table[table_base + page.to(Int64)])
                        src = get_ptr_as_int64(
                            cache_bytes,
                            physical.to(Int64) * page_stride_bytes
                            + (token_begin - page * Int32(self.page_size)).to(Int64)
                            * Int64(RECORD_BYTES),
                        )
                        cp_async_bulk_g2s_mbar(
                            dst, src, Int32(self.chunk * RECORD_BYTES),
                            shared_ptr_to_u32(full),
                        )
                        if cutlass.const_expr(self.prefetch > 0):
                            ahead = token_begin + Int32(self.prefetch * self.chunk)
                            if ahead + Int32(self.chunk) <= cache_length and ahead < causal_end:
                                apage = ahead // Int32(self.page_size)
                                aphys = Int32(page_table[table_base + apage.to(Int64)])
                                prefetch_bulk_global_l2(
                                    get_ptr_as_int64(
                                        cache_bytes,
                                        aphys.to(Int64) * page_stride_bytes
                                        + (ahead - apage * Int32(self.page_size)).to(Int64)
                                        * Int64(RECORD_BYTES),
                                    ),
                                    Int32(self.chunk * RECORD_BYTES),
                                )
                else:
                    entry = lane
                    for _ in cutlass.range_constexpr(self.chunk // 32):
                        token = token_begin + entry
                        if token >= cache_length:
                            token = Int32(0)
                        page = token // Int32(self.page_size)
                        physical = Int32(page_table[table_base + page.to(Int64)])
                        src = get_ptr_as_int64(
                            cache_bytes,
                            physical.to(Int64) * page_stride_bytes
                            + (token - page * Int32(self.page_size)).to(Int64)
                            * Int64(RECORD_BYTES),
                        )
                        cp_async_bulk_g2s_mbar(
                            dst + entry * Int32(RECORD_BYTES), src, Int32(RECORD_BYTES),
                            shared_ptr_to_u32(full),
                        )
                        entry += Int32(32)
                stage += Int32(1)
                if stage == Int32(self.stages):
                    stage = Int32(0)
                    producer_phase ^= Int32(1)
        else:
            self._stage_query(
                q_bytes, q_addr, query_start, head_base, total_q, tid, q_stride_row_bytes
            )
            cute.arch.barrier(barrier_id=1, number_of_threads=self.math_threads)
            group = warp_id // Int32(MATH_WARPS)
            warp = warp_id - group * Int32(MATH_WARPS)
            query_row = query_start + group
            query_valid = Int32(1)
            if query_row >= total_q or query_row >= query_end:
                query_valid = Int32(0)
            if tile_valid == Int32(0):
                query_valid = Int32(0)
            visible_end = cache_length - query_length + (query_row - query_begin) + Int32(1)
            if visible_end > cache_length:
                visible_end = cache_length
            if visible_end < Int32(0):
                visible_end = Int32(0)
            gq = q_addr + group * Int32(HEADS_PER_TILE * Q_ROW_BYTES)
            gp = p_addr + group * Int32(MATH_WARPS * self.p_group_bytes)
            gr = red_addr + group * Int32(2 * MATH_WARPS * HEADS_PER_TILE * 4)
            if group == Int32(0):
                self._run_group(
                    gq, kv_addr, gp, gr, mbar_base, active_chunks, Int32(0), visible_end,
                    query_row, query_valid, head_base, warp, lane, sm_scale_log2,
                    output, final_lse, barrier_id=2,
                )
            else:
                self._run_group(
                    gq, kv_addr, gp, gr, mbar_base, active_chunks, Int32(0), visible_end,
                    query_row, query_valid, head_base, warp, lane, sm_scale_log2,
                    output, final_lse, barrier_id=3,
                )


__all__ = ["DenseMlaFp8PrefillKernel", "smem_bytes"]
