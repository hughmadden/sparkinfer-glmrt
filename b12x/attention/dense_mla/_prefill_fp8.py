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
    ld_shared_v4_u32,
    ldmatrix_m16n16x1_trans_b8,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    prefetch_bulk_global_l2,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4_f32,
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
        self.unified_pair = (self.chunk == 64 and self.query_tile == 2 and self.stages == 1)
        self.eff_query_tile = 4 if self.unified_pair else self.query_tile
        self.math_threads = 32 * MATH_WARPS * self.query_tile
        self.block_threads = 256 if self.unified_pair else (self.math_threads + 32)
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

        if self.unified_pair:
            Storage.__annotations__ = {
                "q": cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, 4 * HEADS_PER_TILE * Q_ROW_BYTES], 128
                ],
                "kv": cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, 2 * 16 * RECORD_BYTES], 128
                ],
                "reduce": cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, 640], 16
                ],
                "mbar": cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint64, 2], 16
                ],
            }
            return cute.struct(Storage)

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

    def _storage_pair(self):
        class StoragePair:
            pass

        StoragePair.__annotations__ = {
            "kv": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, 3 * 32 * RECORD_BYTES], 128
            ],
            "q": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, 2 * HEADS_PER_TILE * Q_ROW_BYTES], 128
            ],
            "reduce": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 240], 16
            ],
            "mbar": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint64, 6], 16
            ],
        }
        return cute.struct(StoragePair)

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
        if cutlass.const_expr(self.unified_pair):
            if total_q <= Int32(self.page_size):
                query_tiles_pair = (total_q + Int32(1)) // Int32(2) + (batch // Int32(2))
                self.kernel_pair(
                    q_bytes, cache_bytes, page_table, cache_seqlens, cu_seqlens_q,
                    output, final_lse, sm_scale_log2, q_stride_row_bytes,
                    page_stride_bytes, page_table_stride, total_q, batch,
                ).launch(
                    grid=(self.head_tiles, query_tiles_pair, 1),
                    block=[288, 1, 1],
                    min_blocks_per_mp=self.min_blocks,
                    stream=stream,
                )
            else:
                query_tiles_quad = (total_q + batch * Int32(3)) // Int32(4)
                self.kernel(
                    q_bytes, cache_bytes, page_table, cache_seqlens, cu_seqlens_q,
                    output, final_lse, sm_scale_log2, q_stride_row_bytes,
                    page_stride_bytes, page_table_stride, total_q, batch,
                ).launch(
                    grid=(self.head_tiles, query_tiles_quad, 1),
                    block=[256, 1, 1],
                    min_blocks_per_mp=self.min_blocks,
                    stream=stream,
                )
        else:
            query_tiles = (total_q + Int32(self.query_tile - 1)) // Int32(self.query_tile)
            if self.query_tile > 1:
                query_tiles = query_tiles + (batch // Int32(2))
            self.kernel(
                q_bytes, cache_bytes, page_table, cache_seqlens, cu_seqlens_q,
                output, final_lse, sm_scale_log2, q_stride_row_bytes,
                page_stride_bytes, page_table_stride, total_q, batch,
            ).launch(
                grid=(self.head_tiles, query_tiles, 1),
                block=[self.block_threads, 1, 1],
                min_blocks_per_mp=self.min_blocks,
                stream=stream,
            )

    # ------------------------------------------------------------------ math
    @cute.jit
    def _stage_query(
        self, q_bytes: cute.Tensor, q_addr: Int32, query_row: Int32, head_base: Int32,
        total_q: Int32, math_tid: Int32, q_stride_row_bytes: Int64,
        *, use_pair: cutlass.Constexpr = True,
    ):
        """Stage sixteen BF16 query heads, permuting each latent 16-dim block.

        Within a latent block the score MMA's k-slots (2t, 2t+1, 2t+8, 2t+9)
        address physical dimensions (4t .. 4t+3), the byte order a
        non-transposed ldmatrix delivers for E4M3 keys.  As 32-bit words the
        permutation is [w0, w2, w4, w6, w1, w3, w5, w7].  The tail is natural.
        """
        group_threads = 32 * MATH_WARPS
        if cutlass.const_expr(self.unified_pair and not use_pair):
            group_threads = 64
        group = math_tid // Int32(group_threads)
        tid = math_tid - group * Int32(group_threads)
        row = query_row + group
        safe_row = row
        if row >= total_q:
            safe_row = Int32(0)
        if cutlass.const_expr(not use_pair) or row < total_q:
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
                unit += Int32(group_threads)

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

    @cute.jit
    def _scores_unified_pair(
        self, q_addr: Int32, s_kv: Int32, first_in_half: Int32, lane: Int32,
        *, single_query: cutlass.Constexpr = False,
    ):
        """Sixteen heads x 8 candidates for both queries q0 and q1."""
        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col_bytes = (lane >> Int32(4)) * Int32(16)
        q0_lane = q_addr + a_row * Int32(Q_ROW_BYTES) + a_col_bytes
        q1_lane = q0_lane + Int32(HEADS_PER_TILE * Q_ROW_BYTES)
        b_row = lane & Int32(7)
        b_col = (lane >> Int32(3)) * Int32(16)
        k_lane = s_kv + (first_in_half + b_row) * Int32(RECORD_BYTES) + b_col
        tid = lane & Int32(3)
        c0 = first_in_half + tid * Int32(2)
        s_lane = s_kv + c0 * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET)
        sa0, sa1, sa2, sa3 = ld_shared_v4_f32(s_lane)
        sb0, sb1, sb2, sb3 = ld_shared_v4_f32(s_lane + Int32(RECORD_BYTES))
        sa = (sa0, sa1, sa2, sa3)
        sb = (sb0, sb1, sb2, sb3)
        qk0 = [Float32(0.0) for _ in range(4)]
        qk1 = [Float32(0.0) for _ in range(4)]
        for g in cutlass.range_constexpr(LATENT_DIM // GROUP):
            part0 = [Float32(0.0) for _ in range(4)]
            part1 = [Float32(0.0) for _ in range(4)]
            for slab in cutlass.range_constexpr(GROUP // 64):
                base = g * GROUP + slab * 64
                k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(k_lane + Int32(base))
                k_slab = (k0, k1, k2, k3)
                for blk in cutlass.range_constexpr(4):
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(k_slab[blk])
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        q0_lane + Int32((base + blk * 16) * 2)
                    )
                    part0[0], part0[1], part0[2], part0[3] = mma_m16n8k16_f32_bf16(
                        part0[0], part0[1], part0[2], part0[3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    if cutlass.const_expr(not single_query):
                        a0_1, a1_1, a2_1, a3_1 = ldmatrix_m8n8x4_b16(
                            q1_lane + Int32((base + blk * 16) * 2)
                        )
                        part1[0], part1[1], part1[2], part1[3] = mma_m16n8k16_f32_bf16(
                            part1[0], part1[1], part1[2], part1[3],
                            a0_1, a1_1, a2_1, a3_1, b0, b1,
                        )
            s0 = sa[g]
            s1 = sb[g]
            qk0[0] = qk0[0] + part0[0] * s0
            qk0[1] = qk0[1] + part0[1] * s1
            qk0[2] = qk0[2] + part0[2] * s0
            qk0[3] = qk0[3] + part0[3] * s1
            if cutlass.const_expr(not single_query):
                qk1[0] = qk1[0] + part1[0] * s0
                qk1[1] = qk1[1] + part1[1] * s1
                qk1[2] = qk1[2] + part1[2] * s0
                qk1[3] = qk1[3] + part1[3] * s1
        for slab in cutlass.range_constexpr(2):
            k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(
                k_lane + Int32(TAIL_OFFSET + slab * 64)
            )
            k_slab = (k0, k1, k2, k3)
            for half in cutlass.range_constexpr(2):
                b0 = k_slab[half * 2]
                b1 = k_slab[half * 2 + 1]
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    q0_lane + Int32((LATENT_DIM + slab * 32 + half * 16) * 2)
                )
                qk0[0], qk0[1], qk0[2], qk0[3] = mma_m16n8k16_f32_bf16(
                    qk0[0], qk0[1], qk0[2], qk0[3],
                    a0, a1, a2, a3, b0, b1,
                )
                if cutlass.const_expr(not single_query):
                    a0_1, a1_1, a2_1, a3_1 = ldmatrix_m8n8x4_b16(
                        q1_lane + Int32((LATENT_DIM + slab * 32 + half * 16) * 2)
                    )
                    qk1[0], qk1[1], qk1[2], qk1[3] = mma_m16n8k16_f32_bf16(
                        qk1[0], qk1[1], qk1[2], qk1[3],
                        a0_1, a1_1, a2_1, a3_1, b0, b1,
                    )
        return qk0, qk1, sa, sb

    @cute.jit
    def _issue_kv_half_chunk(
        self,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        table_base: Int64,
        page_stride_bytes: Int64,
        cache_length: Int32,
        token_begin: Int32,
        dst: Int32,
        full_mbar,
        lane: Int32,
    ):
        if lane == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar, Int32(32 * RECORD_BYTES)
            )
        if token_begin + Int32(32) <= cache_length:
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
                    dst, src, Int32(32 * RECORD_BYTES),
                    shared_ptr_to_u32(full_mbar),
                )
        else:
            token = token_begin + lane
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
                dst + lane * Int32(RECORD_BYTES), src, Int32(RECORD_BYTES),
                shared_ptr_to_u32(full_mbar),
            )

    @cute.jit
    def _run_unified_quad(
        self,
        q_addr: Int32,
        kv_base: Int32,
        red_addr: Int32,
        mbar_base,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        table_base: Int64,
        page_stride_bytes: Int64,
        cache_length: Int32,
        active_chunks: Int32,
        visible_end_0: Int32,
        visible_end_1: Int32,
        visible_end_2: Int32,
        visible_end_3: Int32,
        query_row_0: Int32,
        query_valid_3: Int32,
        head_base: Int32,
        warp_id: Int32,
        lane: Int32,
        scale_log2: Float32,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        *,
        single_query: cutlass.Constexpr = False,
    ):
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        bw = warp_id >> Int32(1)
        sub = warp_id & Int32(1)
        q_pair = warp_id >> Int32(2)
        w_sub = warp_id & Int32(3)
        first_in_half_h0 = w_sub * Int32(8)
        first_in_half_h1 = (w_sub ^ Int32(1)) * Int32(8)
        s_my = w_sub >> Int32(1)
        q_pair_addr = q_addr
        if cutlass.const_expr(not single_query):
            q_pair_addr = q_addr + q_pair * Int32(2 * HEADS_PER_TILE * Q_ROW_BYTES)

        vis_a = visible_end_0
        vis_b = visible_end_1
        if cutlass.const_expr(not single_query):
            if q_pair != Int32(0):
                vis_a = visible_end_2
                vis_b = visible_end_3

        acc0_frag = cute.make_rmem_tensor(32, Float32)
        acc1_frag = cute.make_rmem_tensor(32, Float32)
        acc2_frag = cute.make_rmem_tensor(32, Float32)
        acc3_frag = cute.make_rmem_tensor(32, Float32)
        for i in cutlass.range_constexpr(32):
            acc0_frag[i] = Float32(0.0)
            if cutlass.const_expr(not single_query):
                acc1_frag[i] = Float32(0.0)
                acc2_frag[i] = Float32(0.0)
                acc3_frag[i] = Float32(0.0)

        stat_m0_q0 = Float32(_MASK)
        stat_m1_q0 = Float32(_MASK)
        stat_m0_q1 = Float32(_MASK)
        stat_m1_q1 = Float32(_MASK)
        stat_m0_q2 = Float32(_MASK)
        stat_m1_q2 = Float32(_MASK)
        stat_m0_q3 = Float32(_MASK)
        stat_m1_q3 = Float32(_MASK)

        stat_l0_a = Float32(0.0)
        stat_l1_a = Float32(0.0)
        stat_l0_b = Float32(0.0)
        stat_l1_b = Float32(0.0)

        l16 = lane & Int32(15)
        v_row = (
            (l16 & Int32(1))
            | ((l16 & Int32(4)) >> Int32(1))
            | ((l16 & Int32(8)) >> Int32(1))
            | ((l16 & Int32(2)) << Int32(2))
        )
        v_col_off = bw * Int32(GROUP) + sub * Int32(64)
        v_lane0 = kv_base + v_row * Int32(RECORD_BYTES) + v_col_off
        v_lane1 = v_lane0 + Int32(16 * RECORD_BYTES)

        phase = Int32(0)
        for local in cutlass.range(active_chunks, unroll=1):
            chunk_begin = Int32(local) * Int32(64)

            # --- Half-Chunk 0 (records 0..31) ---
            cute.arch.mbarrier_wait(mbar_base + Int32(0), phase=phase)
            qk_h0_a, qk_h0_b, _, _ = self._scores_unified_pair(
                q_pair_addr, kv_base, first_in_half_h0, lane, single_query=single_query
            )
            v0_regs = []
            v1_regs = []
            for b in cutlass.range_constexpr(4):
                v0_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane0 + Int32(b * 16)))
                v1_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))
            if warp_id == Int32(0):
                s_src = kv_base + lane * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET)
                sc0, sc1, sc2, sc3 = ld_shared_v4_f32(s_src)
                st_shared_v4_f32(red_addr + Int32(1024) + lane * Int32(16), sc0, sc1, sc2, sc3)

            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            if warp_id == Int32(0):
                self._issue_kv_half_chunk(
                    cache_bytes, page_table, table_base, page_stride_bytes,
                    cache_length, chunk_begin + Int32(32), kv_base, mbar_base + Int32(1), lane,
                )

            c0 = chunk_begin + first_in_half_h0 + tid * Int32(2)
            if c0 >= vis_a:
                qk_h0_a[0] = Float32(_MASK)
                qk_h0_a[2] = Float32(_MASK)
            if c0 + Int32(1) >= vis_a:
                qk_h0_a[1] = Float32(_MASK)
                qk_h0_a[3] = Float32(_MASK)
            for r in cutlass.range_constexpr(4):
                qk_h0_a[r] = qk_h0_a[r] * scale_log2
            if cutlass.const_expr(not single_query):
                if c0 >= vis_b:
                    qk_h0_b[0] = Float32(_MASK)
                    qk_h0_b[2] = Float32(_MASK)
                if c0 + Int32(1) >= vis_b:
                    qk_h0_b[1] = Float32(_MASK)
                    qk_h0_b[3] = Float32(_MASK)
                for r in cutlass.range_constexpr(4):
                    qk_h0_b[r] = qk_h0_b[r] * scale_log2

            # --- Half-Chunk 1 (records 32..63) ---
            cute.arch.mbarrier_wait(mbar_base + Int32(1), phase=phase)
            phase ^= Int32(1)
            qk_h1_a, qk_h1_b, sa_h1, sb_h1 = self._scores_unified_pair(
                q_pair_addr, kv_base, first_in_half_h1, lane, single_query=single_query
            )
            v2_regs = []
            v3_regs = []
            for b in cutlass.range_constexpr(4):
                v2_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane0 + Int32(b * 16)))
                v3_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))

            c1 = chunk_begin + Int32(32) + first_in_half_h1 + tid * Int32(2)
            if c1 >= vis_a:
                qk_h1_a[0] = Float32(_MASK)
                qk_h1_a[2] = Float32(_MASK)
            if c1 + Int32(1) >= vis_a:
                qk_h1_a[1] = Float32(_MASK)
                qk_h1_a[3] = Float32(_MASK)
            for r in cutlass.range_constexpr(4):
                qk_h1_a[r] = qk_h1_a[r] * scale_log2
            mx0_a = fmax_f32(fmax_f32(qk_h0_a[0], qk_h0_a[1]), fmax_f32(qk_h1_a[0], qk_h1_a[1]))
            mx1_a = fmax_f32(fmax_f32(qk_h0_a[2], qk_h0_a[3]), fmax_f32(qk_h1_a[2], qk_h1_a[3]))
            for off in (1, 2):
                mx0_a = fmax_f32(mx0_a, cute.arch.shuffle_sync_bfly(mx0_a, offset=off))
                mx1_a = fmax_f32(mx1_a, cute.arch.shuffle_sync_bfly(mx1_a, offset=off))
            if cutlass.const_expr(not single_query):
                if c1 >= vis_b:
                    qk_h1_b[0] = Float32(_MASK)
                    qk_h1_b[2] = Float32(_MASK)
                if c1 + Int32(1) >= vis_b:
                    qk_h1_b[1] = Float32(_MASK)
                    qk_h1_b[3] = Float32(_MASK)
                for r in cutlass.range_constexpr(4):
                    qk_h1_b[r] = qk_h1_b[r] * scale_log2
                mx0_b = fmax_f32(fmax_f32(qk_h0_b[0], qk_h0_b[1]), fmax_f32(qk_h1_b[0], qk_h1_b[1]))
                mx1_b = fmax_f32(fmax_f32(qk_h0_b[2], qk_h0_b[3]), fmax_f32(qk_h1_b[2], qk_h1_b[3]))
                for off in (1, 2):
                    mx0_b = fmax_f32(mx0_b, cute.arch.shuffle_sync_bfly(mx0_b, offset=off))
                    mx1_b = fmax_f32(mx1_b, cute.arch.shuffle_sync_bfly(mx1_b, offset=off))
            else:
                mx0_b = Float32(_MASK)
                mx1_b = Float32(_MASK)

            if tid == Int32(0):
                dst_red = red_addr + (q_pair * Int32(32) + w_sub * Int32(8) + gid) * Int32(16)
                st_shared_v4_f32(dst_red, mx0_a, mx1_a, mx0_b, mx1_b)

            # Barrier A: Half-Chunk 1 KV reads and row-max writes complete!
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            bm0_q0 = stat_m0_q0
            bm1_q0 = stat_m1_q0
            bm0_q1 = stat_m0_q1
            bm1_q1 = stat_m1_q1
            bm0_q2 = stat_m0_q2
            bm1_q2 = stat_m1_q2
            bm0_q3 = stat_m0_q3
            bm1_q3 = stat_m1_q3
            base_r = red_addr + gid * Int32(16)
            for w in cutlass.range_constexpr(4):
                m0_0, m1_0, m0_1, m1_1 = ld_shared_v4_f32(base_r + Int32(w * 128))
                bm0_q0 = fmax_f32(bm0_q0, m0_0)
                bm1_q0 = fmax_f32(bm1_q0, m1_0)
                if cutlass.const_expr(not single_query):
                    m0_2, m1_2, m0_3, m1_3 = ld_shared_v4_f32(base_r + Int32(512 + w * 128))
                    bm0_q1 = fmax_f32(bm0_q1, m0_1)
                    bm1_q1 = fmax_f32(bm1_q1, m1_1)
                    bm0_q2 = fmax_f32(bm0_q2, m0_2)
                    bm1_q2 = fmax_f32(bm1_q2, m1_2)
                    bm0_q3 = fmax_f32(bm0_q3, m0_3)
                    bm1_q3 = fmax_f32(bm1_q3, m1_3)

            if chunk_begin >= visible_end_0:
                bm0_q0 = stat_m0_q0
                bm1_q0 = stat_m1_q0
            alpha0_q0 = exp2_approx(stat_m0_q0 - bm0_q0)
            alpha1_q0 = exp2_approx(stat_m1_q0 - bm1_q0)
            stat_m0_q0 = bm0_q0
            stat_m1_q0 = bm1_q0

            if cutlass.const_expr(not single_query):
                if chunk_begin >= visible_end_1:
                    bm0_q1 = stat_m0_q1
                    bm1_q1 = stat_m1_q1
                if chunk_begin >= visible_end_2:
                    bm0_q2 = stat_m0_q2
                    bm1_q2 = stat_m1_q2
                if chunk_begin >= visible_end_3:
                    bm0_q3 = stat_m0_q3
                    bm1_q3 = stat_m1_q3

                alpha0_q1 = exp2_approx(stat_m0_q1 - bm0_q1)
                alpha1_q1 = exp2_approx(stat_m1_q1 - bm1_q1)
                alpha0_q2 = exp2_approx(stat_m0_q2 - bm0_q2)
                alpha1_q2 = exp2_approx(stat_m1_q2 - bm1_q2)
                alpha0_q3 = exp2_approx(stat_m0_q3 - bm0_q3)
                alpha1_q3 = exp2_approx(stat_m1_q3 - bm1_q3)

                stat_m0_q1 = bm0_q1
                stat_m1_q1 = bm1_q1
                stat_m0_q2 = bm0_q2
                stat_m1_q2 = bm1_q2
                stat_m0_q3 = bm0_q3
                stat_m1_q3 = bm1_q3
            else:
                alpha0_q1 = Float32(1.0)
                alpha1_q1 = Float32(1.0)
                alpha0_q2 = Float32(1.0)
                alpha1_q2 = Float32(1.0)
                alpha0_q3 = Float32(1.0)
                alpha1_q3 = Float32(1.0)

            bm0_a = bm0_q0
            bm1_a = bm1_q0
            bm0_b = bm0_q1
            bm1_b = bm1_q1
            alpha0_a = alpha0_q0
            alpha1_a = alpha1_q0
            alpha0_b = alpha0_q1
            alpha1_b = alpha1_q1
            if cutlass.const_expr(not single_query):
                if q_pair != Int32(0):
                    bm0_a = bm0_q2
                    bm1_a = bm1_q2
                    bm0_b = bm0_q3
                    bm1_b = bm1_q3
                    alpha0_a = alpha0_q2
                    alpha1_a = alpha1_q2
                    alpha0_b = alpha0_q3
                    alpha1_b = alpha1_q3

            prob_h0_a = [
                exp2_approx(qk_h0_a[0] - bm0_a),
                exp2_approx(qk_h0_a[1] - bm0_a),
                exp2_approx(qk_h0_a[2] - bm1_a),
                exp2_approx(qk_h0_a[3] - bm1_a),
            ]
            prob_h1_a = [
                exp2_approx(qk_h1_a[0] - bm0_a),
                exp2_approx(qk_h1_a[1] - bm0_a),
                exp2_approx(qk_h1_a[2] - bm1_a),
                exp2_approx(qk_h1_a[3] - bm1_a),
            ]
            if chunk_begin >= vis_a:
                for r in cutlass.range_constexpr(4):
                    prob_h0_a[r] = Float32(0.0)
                    prob_h1_a[r] = Float32(0.0)

            if cutlass.const_expr(not single_query):
                prob_h0_b = [
                    exp2_approx(qk_h0_b[0] - bm0_b),
                    exp2_approx(qk_h0_b[1] - bm0_b),
                    exp2_approx(qk_h0_b[2] - bm1_b),
                    exp2_approx(qk_h0_b[3] - bm1_b),
                ]
                prob_h1_b = [
                    exp2_approx(qk_h1_b[0] - bm0_b),
                    exp2_approx(qk_h1_b[1] - bm0_b),
                    exp2_approx(qk_h1_b[2] - bm1_b),
                    exp2_approx(qk_h1_b[3] - bm1_b),
                ]
                if chunk_begin >= vis_b:
                    for r in cutlass.range_constexpr(4):
                        prob_h0_b[r] = Float32(0.0)
                        prob_h1_b[r] = Float32(0.0)
            else:
                prob_h0_b = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                prob_h1_b = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]

            acc0 = [acc0_frag[i] for i in range(32)]
            acc1 = [acc1_frag[i] for i in range(32)]
            acc2 = [acc2_frag[i] for i in range(32)]
            acc3 = [acc3_frag[i] for i in range(32)]
            for j in cutlass.range_constexpr(8):
                acc0[j * 4] = acc0[j * 4] * alpha0_q0
                acc0[j * 4 + 1] = acc0[j * 4 + 1] * alpha0_q0
                acc0[j * 4 + 2] = acc0[j * 4 + 2] * alpha1_q0
                acc0[j * 4 + 3] = acc0[j * 4 + 3] * alpha1_q0
                if cutlass.const_expr(not single_query):
                    acc1[j * 4] = acc1[j * 4] * alpha0_q1
                    acc1[j * 4 + 1] = acc1[j * 4 + 1] * alpha0_q1
                    acc1[j * 4 + 2] = acc1[j * 4 + 2] * alpha1_q1
                    acc1[j * 4 + 3] = acc1[j * 4 + 3] * alpha1_q1
                    acc2[j * 4] = acc2[j * 4] * alpha0_q2
                    acc2[j * 4 + 1] = acc2[j * 4 + 1] * alpha0_q2
                    acc2[j * 4 + 2] = acc2[j * 4 + 2] * alpha1_q2
                    acc2[j * 4 + 3] = acc2[j * 4 + 3] * alpha1_q2
                    acc3[j * 4] = acc3[j * 4] * alpha0_q3
                    acc3[j * 4 + 1] = acc3[j * 4 + 1] * alpha0_q3
                    acc3[j * 4 + 2] = acc3[j * 4 + 2] * alpha1_q3
                    acc3[j * 4 + 3] = acc3[j * 4 + 3] * alpha1_q3

            c0_scale = red_addr + Int32(1024) + (first_in_half_h0 + tid * Int32(2)) * Int32(16)
            sa_h0_0, sa_h0_1, sa_h0_2, sa_h0_3 = ld_shared_v4_f32(c0_scale)
            sb_h0_0, sb_h0_1, sb_h0_2, sb_h0_3 = ld_shared_v4_f32(c0_scale + Int32(16))
            sa_h0 = (sa_h0_0, sa_h0_1, sa_h0_2, sa_h0_3)
            sb_h0 = (sb_h0_0, sb_h0_1, sb_h0_2, sb_h0_3)

            # Stage P(ks=0, 1) (16,384 B) into kv_base[0..16384) and sum_nt0 (4,096 B) into kv_base[16384..20480)!
            dst_p_h0 = (
                kv_base
                + s_my * Int32(8192)
                + q_pair * Int32(1024)
                + lane * Int32(16)
                + sub * Int32(8)
            )
            for g_idx in cutlass.range_constexpr(4):
                base_g = dst_p_h0 + Int32(g_idx * 2048)
                st_shared_u32(
                    base_g,
                    pack_f32x2_to_bfloat2(prob_h0_a[0] * sa_h0[g_idx], prob_h0_a[1] * sb_h0[g_idx]),
                )
                st_shared_u32(
                    base_g + Int32(4),
                    pack_f32x2_to_bfloat2(prob_h0_a[2] * sa_h0[g_idx], prob_h0_a[3] * sb_h0[g_idx]),
                )
                if cutlass.const_expr(not single_query):
                    st_shared_u32(
                        base_g + Int32(512),
                        pack_f32x2_to_bfloat2(prob_h0_b[0] * sa_h0[g_idx], prob_h0_b[1] * sb_h0[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(516),
                        pack_f32x2_to_bfloat2(prob_h0_b[2] * sa_h0[g_idx], prob_h0_b[3] * sb_h0[g_idx]),
                    )

            dst_sum = (
                kv_base
                + Int32(16384)
                + sub * Int32(2048)
                + (q_pair * Int32(2) + s_my) * Int32(512)
                + lane * Int32(16)
            )
            if sub == Int32(0):
                st_shared_v4_f32(
                    dst_sum,
                    prob_h0_a[0] + prob_h0_a[1],
                    prob_h0_a[2] + prob_h0_a[3],
                    prob_h0_b[0] + prob_h0_b[1],
                    prob_h0_b[2] + prob_h0_b[3],
                )
            else:
                st_shared_v4_f32(
                    dst_sum,
                    prob_h1_a[0] + prob_h1_a[1],
                    prob_h1_a[2] + prob_h1_a[3],
                    prob_h1_b[0] + prob_h1_b[1],
                    prob_h1_b[2] + prob_h1_b[3],
                )

            # Barrier B1: P(ks=0, 1) and sum_nt0 ready!
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            src_sum = (
                kv_base
                + Int32(16384)
                + (sub ^ Int32(1)) * Int32(2048)
                + (q_pair * Int32(2) + s_my) * Int32(512)
                + lane * Int32(16)
            )
            s0_a, s1_a, s0_b, s1_b = ld_shared_v4_f32(src_sum)
            if sub == Int32(1):
                stat_l0_a = stat_l0_a * alpha0_a + ((s0_a + prob_h0_a[0]) + prob_h0_a[1])
                stat_l1_a = stat_l1_a * alpha1_a + ((s1_a + prob_h0_a[2]) + prob_h0_a[3])
                if cutlass.const_expr(not single_query):
                    stat_l0_b = stat_l0_b * alpha0_b + ((s0_b + prob_h0_b[0]) + prob_h0_b[1])
                    stat_l1_b = stat_l1_b * alpha1_b + ((s1_b + prob_h0_b[2]) + prob_h0_b[3])
            else:
                stat_l0_a = stat_l0_a * alpha0_a + ((s0_a + prob_h1_a[0]) + prob_h1_a[1])
                stat_l1_a = stat_l1_a * alpha1_a + ((s1_a + prob_h1_a[2]) + prob_h1_a[3])
                if cutlass.const_expr(not single_query):
                    stat_l0_b = stat_l0_b * alpha0_b + ((s0_b + prob_h1_b[0]) + prob_h1_b[1])
                    stat_l1_b = stat_l1_b * alpha1_b + ((s1_b + prob_h1_b[2]) + prob_h1_b[3])

            src_p01 = kv_base + bw * Int32(2048) + lane * Int32(16)
            p0_0_q0, p0_1_q0, p0_2_q0, p0_3_q0 = ld_shared_v4_u32(src_p01)
            if cutlass.const_expr(not single_query):
                p0_0_q1, p0_1_q1, p0_2_q1, p0_3_q1 = ld_shared_v4_u32(src_p01 + Int32(512))
                p0_0_q2, p0_1_q2, p0_2_q2, p0_3_q2 = ld_shared_v4_u32(src_p01 + Int32(1024))
                p0_0_q3, p0_1_q3, p0_2_q3, p0_3_q3 = ld_shared_v4_u32(src_p01 + Int32(1536))

            # PV ks = 0
            for b in cutlass.range_constexpr(4):
                f0, f1 = v0_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p0_0_q0, p0_1_q0, p0_2_q0, p0_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p0_0_q0, p0_1_q0, p0_2_q0, p0_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p0_0_q1, p0_1_q1, p0_2_q1, p0_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p0_0_q1, p0_1_q1, p0_2_q1, p0_3_q1, b2, b3,
                    )
                    acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3],
                        p0_0_q2, p0_1_q2, p0_2_q2, p0_3_q2, b0, b1,
                    )
                    acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3],
                        p0_0_q2, p0_1_q2, p0_2_q2, p0_3_q2, b2, b3,
                    )
                    acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3],
                        p0_0_q3, p0_1_q3, p0_2_q3, p0_3_q3, b0, b1,
                    )
                    acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3],
                        p0_0_q3, p0_1_q3, p0_2_q3, p0_3_q3, b2, b3,
                    )

            p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0 = ld_shared_v4_u32(src_p01 + Int32(8192))
            if cutlass.const_expr(not single_query):
                p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1 = ld_shared_v4_u32(src_p01 + Int32(8704))
                p1_0_q2, p1_1_q2, p1_2_q2, p1_3_q2 = ld_shared_v4_u32(src_p01 + Int32(9216))
                p1_0_q3, p1_1_q3, p1_2_q3, p1_3_q3 = ld_shared_v4_u32(src_p01 + Int32(9728))

            # PV ks = 1
            for b in cutlass.range_constexpr(4):
                f0, f1 = v1_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b2, b3,
                    )
                    acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3],
                        p1_0_q2, p1_1_q2, p1_2_q2, p1_3_q2, b0, b1,
                    )
                    acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3],
                        p1_0_q2, p1_1_q2, p1_2_q2, p1_3_q2, b2, b3,
                    )
                    acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3],
                        p1_0_q3, p1_1_q3, p1_2_q3, p1_3_q3, b0, b1,
                    )
                    acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3],
                        p1_0_q3, p1_1_q3, p1_2_q3, p1_3_q3, b2, b3,
                    )

            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            dst_p_h1 = dst_p_h0 ^ Int32(8)
            for g_idx in cutlass.range_constexpr(4):
                base_g = dst_p_h1 + Int32(g_idx * 2048)
                st_shared_u32(
                    base_g,
                    pack_f32x2_to_bfloat2(prob_h1_a[0] * sa_h1[g_idx], prob_h1_a[1] * sb_h1[g_idx]),
                )
                st_shared_u32(
                    base_g + Int32(4),
                    pack_f32x2_to_bfloat2(prob_h1_a[2] * sa_h1[g_idx], prob_h1_a[3] * sb_h1[g_idx]),
                )
                if cutlass.const_expr(not single_query):
                    st_shared_u32(
                        base_g + Int32(512),
                        pack_f32x2_to_bfloat2(prob_h1_b[0] * sa_h1[g_idx], prob_h1_b[1] * sb_h1[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(516),
                        pack_f32x2_to_bfloat2(prob_h1_b[2] * sa_h1[g_idx], prob_h1_b[3] * sb_h1[g_idx]),
                    )
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0 = ld_shared_v4_u32(src_p01)
            p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0 = ld_shared_v4_u32(src_p01 + Int32(8192))
            if cutlass.const_expr(not single_query):
                p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1 = ld_shared_v4_u32(src_p01 + Int32(512))
                p2_0_q2, p2_1_q2, p2_2_q2, p2_3_q2 = ld_shared_v4_u32(src_p01 + Int32(1024))
                p2_0_q3, p2_1_q3, p2_2_q3, p2_3_q3 = ld_shared_v4_u32(src_p01 + Int32(1536))

                p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1 = ld_shared_v4_u32(src_p01 + Int32(8704))
                p3_0_q2, p3_1_q2, p3_2_q2, p3_3_q2 = ld_shared_v4_u32(src_p01 + Int32(9216))
                p3_0_q3, p3_1_q3, p3_2_q3, p3_3_q3 = ld_shared_v4_u32(src_p01 + Int32(9728))

            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            if local + Int32(1) < active_chunks and warp_id == Int32(0):
                self._issue_kv_half_chunk(
                    cache_bytes, page_table, table_base, page_stride_bytes,
                    cache_length, chunk_begin + Int32(64), kv_base, mbar_base + Int32(0), lane,
                )

            # PV ks = 2
            for b in cutlass.range_constexpr(4):
                f0, f1 = v2_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1, b2, b3,
                    )
                    acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3],
                        p2_0_q2, p2_1_q2, p2_2_q2, p2_3_q2, b0, b1,
                    )
                    acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3],
                        p2_0_q2, p2_1_q2, p2_2_q2, p2_3_q2, b2, b3,
                    )
                    acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3],
                        p2_0_q3, p2_1_q3, p2_2_q3, p2_3_q3, b0, b1,
                    )
                    acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3],
                        p2_0_q3, p2_1_q3, p2_2_q3, p2_3_q3, b2, b3,
                    )

            # PV ks = 3
            for b in cutlass.range_constexpr(4):
                f0, f1 = v3_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b2, b3,
                    )
                    acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n0 * 4], acc2[n0 * 4 + 1], acc2[n0 * 4 + 2], acc2[n0 * 4 + 3],
                        p3_0_q2, p3_1_q2, p3_2_q2, p3_3_q2, b0, b1,
                    )
                    acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc2[n1 * 4], acc2[n1 * 4 + 1], acc2[n1 * 4 + 2], acc2[n1 * 4 + 3],
                        p3_0_q2, p3_1_q2, p3_2_q2, p3_3_q2, b2, b3,
                    )
                    acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n0 * 4], acc3[n0 * 4 + 1], acc3[n0 * 4 + 2], acc3[n0 * 4 + 3],
                        p3_0_q3, p3_1_q3, p3_2_q3, p3_3_q3, b0, b1,
                    )
                    acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc3[n1 * 4], acc3[n1 * 4 + 1], acc3[n1 * 4 + 2], acc3[n1 * 4 + 3],
                        p3_0_q3, p3_1_q3, p3_2_q3, p3_3_q3, b2, b3,
                    )

            for i in cutlass.range_constexpr(32):
                acc0_frag[i] = acc0[i]
                if cutlass.const_expr(not single_query):
                    acc1_frag[i] = acc1[i]
                    acc2_frag[i] = acc2[i]
                    acc3_frag[i] = acc3[i]

        # Epilogue: combine per-lane partial row sums in exact w_ref = 0, 1, 2, 3 order.
        for off in (1, 2):
            stat_l0_a = stat_l0_a + cute.arch.shuffle_sync_bfly(stat_l0_a, offset=off)
            stat_l1_a = stat_l1_a + cute.arch.shuffle_sync_bfly(stat_l1_a, offset=off)
            if cutlass.const_expr(not single_query):
                stat_l0_b = stat_l0_b + cute.arch.shuffle_sync_bfly(stat_l0_b, offset=off)
                stat_l1_b = stat_l1_b + cute.arch.shuffle_sync_bfly(stat_l1_b, offset=off)
        w_ref = (w_sub >> Int32(1)) + (sub ^ Int32(1)) * Int32(2)
        if tid == Int32(0):
            dst_l = red_addr + (q_pair * Int32(32) + w_ref * Int32(8) + gid) * Int32(16)
            st_shared_v4_f32(dst_l, stat_l0_a, stat_l1_a, stat_l0_b, stat_l1_b)
        cute.arch.barrier(barrier_id=2, number_of_threads=256)

        t0_q0 = Float32(0.0)
        t1_q0 = Float32(0.0)
        t0_q1 = Float32(0.0)
        t1_q1 = Float32(0.0)
        t0_q2 = Float32(0.0)
        t1_q2 = Float32(0.0)
        t0_q3 = Float32(0.0)
        t1_q3 = Float32(0.0)
        base_l = red_addr + gid * Int32(16)
        for w in cutlass.range_constexpr(4):
            l0_0, l1_0, l0_1, l1_1 = ld_shared_v4_f32(base_l + Int32(w * 128))
            t0_q0 = t0_q0 + l0_0
            t1_q0 = t1_q0 + l1_0
            if cutlass.const_expr(not single_query):
                l0_2, l1_2, l0_3, l1_3 = ld_shared_v4_f32(base_l + Int32(512 + w * 128))
                t0_q1 = t0_q1 + l0_1
                t1_q1 = t1_q1 + l1_1
                t0_q2 = t0_q2 + l0_2
                t1_q2 = t1_q2 + l1_2
                t0_q3 = t0_q3 + l0_3
                t1_q3 = t1_q3 + l1_3

        h0 = head_base + gid
        h1 = h0 + Int32(8)
        col_off = (v_col_off + tid * Int32(2)).to(Int64)
        query_row_1 = query_row_0 + Int32(1)
        query_row_2 = query_row_0 + Int32(2)
        query_row_3 = query_row_0 + Int32(3)

        inv0_q0 = Float32(0.0)
        inv1_q0 = Float32(0.0)
        if t0_q0 > Float32(0.0):
            inv0_q0 = Float32(1.0) / t0_q0
        if t1_q0 > Float32(0.0):
            inv1_q0 = Float32(1.0) / t1_q0
        out0_q0 = get_ptr_as_int64(
            output,
            (query_row_0.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
            + col_off,
        )
        out1_q0 = out0_q0 + Int64(8 * VALUE_DIM * 2)
        for j in cutlass.range_constexpr(8):
            st_global_u32(
                out0_q0 + Int64(j * 16),
                pack_f32x2_to_bfloat2(acc0_frag[j * 4] * inv0_q0, acc0_frag[j * 4 + 1] * inv0_q0),
            )
            st_global_u32(
                out1_q0 + Int64(j * 16),
                pack_f32x2_to_bfloat2(acc0_frag[j * 4 + 2] * inv1_q0, acc0_frag[j * 4 + 3] * inv1_q0),
            )
        if warp_id == Int32(0) and tid == Int32(0):
            lse0_q0 = Float32(-Float32.inf)
            lse1_q0 = Float32(-Float32.inf)
            if t0_q0 > Float32(0.0):
                lse0_q0 = (log2_approx(t0_q0) + stat_m0_q0) * Float32(0.6931471805599453)
            if t1_q0 > Float32(0.0):
                lse1_q0 = (log2_approx(t1_q0) + stat_m1_q0) * Float32(0.6931471805599453)
            final_lse[query_row_0, h0] = lse0_q0
            final_lse[query_row_0, h1] = lse1_q0

        if cutlass.const_expr(not single_query):
            inv0_q1 = Float32(0.0)
            inv1_q1 = Float32(0.0)
            if t0_q1 > Float32(0.0):
                inv0_q1 = Float32(1.0) / t0_q1
            if t1_q1 > Float32(0.0):
                inv1_q1 = Float32(1.0) / t1_q1
            out0_q1 = get_ptr_as_int64(
                output,
                (query_row_1.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                + col_off,
            )
            out1_q1 = out0_q1 + Int64(8 * VALUE_DIM * 2)
            for j in cutlass.range_constexpr(8):
                st_global_u32(
                    out0_q1 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc1_frag[j * 4] * inv0_q1, acc1_frag[j * 4 + 1] * inv0_q1),
                )
                st_global_u32(
                    out1_q1 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc1_frag[j * 4 + 2] * inv1_q1, acc1_frag[j * 4 + 3] * inv1_q1),
                )
            if warp_id == Int32(1) and tid == Int32(0):
                lse0_q1 = Float32(-Float32.inf)
                lse1_q1 = Float32(-Float32.inf)
                if t0_q1 > Float32(0.0):
                    lse0_q1 = (log2_approx(t0_q1) + stat_m0_q1) * Float32(0.6931471805599453)
                if t1_q1 > Float32(0.0):
                    lse1_q1 = (log2_approx(t1_q1) + stat_m1_q1) * Float32(0.6931471805599453)
                final_lse[query_row_1, h0] = lse0_q1
                final_lse[query_row_1, h1] = lse1_q1

            inv0_q2 = Float32(0.0)
            inv1_q2 = Float32(0.0)
            if t0_q2 > Float32(0.0):
                inv0_q2 = Float32(1.0) / t0_q2
            if t1_q2 > Float32(0.0):
                inv1_q2 = Float32(1.0) / t1_q2
            out0_q2 = get_ptr_as_int64(
                output,
                (query_row_2.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                + col_off,
            )
            out1_q2 = out0_q2 + Int64(8 * VALUE_DIM * 2)
            for j in cutlass.range_constexpr(8):
                st_global_u32(
                    out0_q2 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc2_frag[j * 4] * inv0_q2, acc2_frag[j * 4 + 1] * inv0_q2),
                )
                st_global_u32(
                    out1_q2 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc2_frag[j * 4 + 2] * inv1_q2, acc2_frag[j * 4 + 3] * inv1_q2),
                )
            if warp_id == Int32(2) and tid == Int32(0):
                lse0_q2 = Float32(-Float32.inf)
                lse1_q2 = Float32(-Float32.inf)
                if t0_q2 > Float32(0.0):
                    lse0_q2 = (log2_approx(t0_q2) + stat_m0_q2) * Float32(0.6931471805599453)
                if t1_q2 > Float32(0.0):
                    lse1_q2 = (log2_approx(t1_q2) + stat_m1_q2) * Float32(0.6931471805599453)
                final_lse[query_row_2, h0] = lse0_q2
                final_lse[query_row_2, h1] = lse1_q2

            if query_valid_3 != Int32(0):
                inv0_q3 = Float32(0.0)
                inv1_q3 = Float32(0.0)
                if t0_q3 > Float32(0.0):
                    inv0_q3 = Float32(1.0) / t0_q3
                if t1_q3 > Float32(0.0):
                    inv1_q3 = Float32(1.0) / t1_q3
                out0_q3 = get_ptr_as_int64(
                    output,
                    (query_row_3.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                    + col_off,
                )
                out1_q3 = out0_q3 + Int64(8 * VALUE_DIM * 2)
                for j in cutlass.range_constexpr(8):
                    st_global_u32(
                        out0_q3 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc3_frag[j * 4] * inv0_q3, acc3_frag[j * 4 + 1] * inv0_q3),
                    )
                    st_global_u32(
                        out1_q3 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc3_frag[j * 4 + 2] * inv1_q3, acc3_frag[j * 4 + 3] * inv1_q3),
                    )
                if warp_id == Int32(3) and tid == Int32(0):
                    lse0_q3 = Float32(-Float32.inf)
                    lse1_q3 = Float32(-Float32.inf)
                    if t0_q3 > Float32(0.0):
                        lse0_q3 = (log2_approx(t0_q3) + stat_m0_q3) * Float32(0.6931471805599453)
                    if t1_q3 > Float32(0.0):
                        lse1_q3 = (log2_approx(t1_q3) + stat_m1_q3) * Float32(0.6931471805599453)
                    final_lse[query_row_3, h0] = lse0_q3
                    final_lse[query_row_3, h1] = lse1_q3

    @cute.jit
    def _run_unified_pair(
        self,
        q_addr: Int32,
        kv_addr0: Int32,
        red_addr: Int32,
        mbar_base,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        table_base: Int64,
        page_stride_bytes: Int64,
        cache_length: Int32,
        active_chunks: Int32,
        visible_end_0: Int32,
        visible_end_1: Int32,
        query_row_0: Int32,
        query_row_1: Int32,
        query_valid_0: Int32,
        query_valid_1: Int32,
        head_base: Int32,
        warp_id: Int32,
        lane: Int32,
        scale_log2: Float32,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        *,
        single_query: cutlass.Constexpr = False,
    ):
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        bw = warp_id >> Int32(1)
        sub = warp_id & Int32(1)
        first_in_half = (warp_id & Int32(3)) * Int32(8)

        kv_addr1 = q_addr + Int32(2 * HEADS_PER_TILE * Q_ROW_BYTES)
        p_addr = q_addr + Int32(2 * HEADS_PER_TILE * Q_ROW_BYTES + 32 * RECORD_BYTES)
        sum_addr = red_addr + Int32(512)

        acc0_frag = cute.make_rmem_tensor(32, Float32)
        acc1_frag = cute.make_rmem_tensor(32, Float32)
        for i in cutlass.range_constexpr(32):
            acc0_frag[i] = Float32(0.0)
            if cutlass.const_expr(not single_query):
                acc1_frag[i] = Float32(0.0)

        stat_m0_q0 = Float32(_MASK)
        stat_m1_q0 = Float32(_MASK)
        stat_m0_q1 = Float32(_MASK)
        stat_m1_q1 = Float32(_MASK)
        stat_l0_q0 = Float32(0.0)
        stat_l1_q0 = Float32(0.0)
        stat_l0_q1 = Float32(0.0)
        stat_l1_q1 = Float32(0.0)

        l16 = lane & Int32(15)
        v_row = (
            (l16 & Int32(1))
            | ((l16 & Int32(4)) >> Int32(1))
            | ((l16 & Int32(8)) >> Int32(1))
            | ((l16 & Int32(2)) << Int32(2))
        )
        v_col_off = bw * Int32(GROUP) + sub * Int32(64)
        v_lane0 = kv_addr0 + v_row * Int32(RECORD_BYTES) + v_col_off
        v_lane1 = v_lane0 + Int32(16 * RECORD_BYTES)
        v_lane2 = kv_addr1 + v_row * Int32(RECORD_BYTES) + v_col_off
        v_lane3 = v_lane2 + Int32(16 * RECORD_BYTES)

        q_pad_lane = (
            q_addr
            + (gid >> Int32(1)) * Int32(Q_ROW_BYTES)
            + Int32(QK_DIM * 2)
            + (gid & Int32(1)) * Int32(8)
        )

        phase = Int32(0)
        for local in cutlass.range(active_chunks, unroll=1):
            chunk_begin = Int32(local) * Int32(64)
            s_wait = Int32(0)
            s_other = Int32(1)
            s_kv_score = kv_addr0
            if warp_id >= Int32(4):
                s_wait = Int32(1)
                s_other = Int32(0)
                s_kv_score = kv_addr1
            cute.arch.mbarrier_wait(mbar_base + s_wait, phase=phase)

            qk0, qk1, sa, sb = self._scores_unified_pair(
                q_addr, s_kv_score, first_in_half, lane, single_query=single_query
            )

            cute.arch.mbarrier_wait(mbar_base + s_other, phase=phase)
            phase ^= Int32(1)

            v0_regs = []
            v1_regs = []
            v2_regs = []
            v3_regs = []
            for b in cutlass.range_constexpr(4):
                v0_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane0 + Int32(b * 16)))
                v1_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))
                v2_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane2 + Int32(b * 16)))
                v3_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane3 + Int32(b * 16)))

            c0 = chunk_begin + warp_id * Int32(8) + tid * Int32(2)
            if c0 >= visible_end_0:
                qk0[0] = Float32(_MASK)
                qk0[2] = Float32(_MASK)
            if c0 + Int32(1) >= visible_end_0:
                qk0[1] = Float32(_MASK)
                qk0[3] = Float32(_MASK)
            for r in cutlass.range_constexpr(4):
                qk0[r] = qk0[r] * scale_log2
            mx0_q0 = fmax_f32(qk0[0], qk0[1])
            mx1_q0 = fmax_f32(qk0[2], qk0[3])
            for off in (1, 2):
                mx0_q0 = fmax_f32(mx0_q0, cute.arch.shuffle_sync_bfly(mx0_q0, offset=off))
                mx1_q0 = fmax_f32(mx1_q0, cute.arch.shuffle_sync_bfly(mx1_q0, offset=off))

            if cutlass.const_expr(not single_query):
                if c0 >= visible_end_1:
                    qk1[0] = Float32(_MASK)
                    qk1[2] = Float32(_MASK)
                if c0 + Int32(1) >= visible_end_1:
                    qk1[1] = Float32(_MASK)
                    qk1[3] = Float32(_MASK)
                for r in cutlass.range_constexpr(4):
                    qk1[r] = qk1[r] * scale_log2
                mx0_q1 = fmax_f32(qk1[0], qk1[1])
                mx1_q1 = fmax_f32(qk1[2], qk1[3])
                for off in (1, 2):
                    mx0_q1 = fmax_f32(mx0_q1, cute.arch.shuffle_sync_bfly(mx0_q1, offset=off))
                    mx1_q1 = fmax_f32(mx1_q1, cute.arch.shuffle_sync_bfly(mx1_q1, offset=off))

            if tid == Int32(0):
                st_shared_f32(red_addr + (warp_id * Int32(16) + gid) * Int32(4), mx0_q0)
                st_shared_f32(red_addr + (warp_id * Int32(16) + gid + Int32(8)) * Int32(4), mx1_q0)
                if cutlass.const_expr(not single_query):
                    q_pad_w = q_pad_lane + warp_id * Int32(4 * Q_ROW_BYTES)
                    st_shared_f32(q_pad_w, mx0_q1)
                    st_shared_f32(q_pad_w + Int32(4), mx1_q1)

            # Barrier A: _scores and all V(ks=0..3) reads complete across all 8 warps!
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            if local + Int32(1) < active_chunks:
                if warp_id == Int32(0):
                    self._issue_kv_half_chunk(
                        cache_bytes, page_table, table_base, page_stride_bytes,
                        cache_length, chunk_begin + Int32(64), kv_addr0, mbar_base + Int32(0), lane,
                    )
                if warp_id == Int32(1):
                    self._issue_kv_half_chunk(
                        cache_bytes, page_table, table_base, page_stride_bytes,
                        cache_length, chunk_begin + Int32(96), kv_addr1, mbar_base + Int32(1), lane,
                    )

            bm0_q0 = stat_m0_q0
            bm1_q0 = stat_m1_q0
            bm0_q1 = stat_m0_q1
            bm1_q1 = stat_m1_q1
            for w in cutlass.range_constexpr(8):
                bm0_q0 = fmax_f32(bm0_q0, ld_shared_f32(red_addr + Int32((w * 16) * 4) + gid * Int32(4)))
                bm1_q0 = fmax_f32(bm1_q0, ld_shared_f32(red_addr + Int32((w * 16 + 8) * 4) + gid * Int32(4)))
                if cutlass.const_expr(not single_query):
                    bm0_q1 = fmax_f32(bm0_q1, ld_shared_f32(q_pad_lane + Int32(w * 4 * Q_ROW_BYTES)))
                    bm1_q1 = fmax_f32(bm1_q1, ld_shared_f32(q_pad_lane + Int32(w * 4 * Q_ROW_BYTES + 4)))

            if cutlass.const_expr(not single_query):
                if chunk_begin >= visible_end_0:
                    bm0_q0 = stat_m0_q0
                    bm1_q0 = stat_m1_q0

            alpha0_q0 = exp2_approx(stat_m0_q0 - bm0_q0)
            alpha1_q0 = exp2_approx(stat_m1_q0 - bm1_q0)
            stat_m0_q0 = bm0_q0
            stat_m1_q0 = bm1_q0

            prob0 = [
                exp2_approx(qk0[0] - bm0_q0),
                exp2_approx(qk0[1] - bm0_q0),
                exp2_approx(qk0[2] - bm1_q0),
                exp2_approx(qk0[3] - bm1_q0),
            ]
            if cutlass.const_expr(not single_query):
                if chunk_begin >= visible_end_0:
                    prob0[0] = Float32(0.0)
                    prob0[1] = Float32(0.0)
                    prob0[2] = Float32(0.0)
                    prob0[3] = Float32(0.0)
                alpha0_q1 = exp2_approx(stat_m0_q1 - bm0_q1)
                alpha1_q1 = exp2_approx(stat_m1_q1 - bm1_q1)
                stat_m0_q1 = bm0_q1
                stat_m1_q1 = bm1_q1
                prob1 = [
                    exp2_approx(qk1[0] - bm0_q1),
                    exp2_approx(qk1[1] - bm0_q1),
                    exp2_approx(qk1[2] - bm1_q1),
                    exp2_approx(qk1[3] - bm1_q1),
                ]
            else:
                alpha0_q1 = Float32(1.0)
                alpha1_q1 = Float32(1.0)
                prob1 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]

            acc0 = [acc0_frag[i] for i in range(32)]
            acc1 = [acc1_frag[i] for i in range(32)]
            for j in cutlass.range_constexpr(8):
                acc0[j * 4] = acc0[j * 4] * alpha0_q0
                acc0[j * 4 + 1] = acc0[j * 4 + 1] * alpha0_q0
                acc0[j * 4 + 2] = acc0[j * 4 + 2] * alpha1_q0
                acc0[j * 4 + 3] = acc0[j * 4 + 3] * alpha1_q0
                if cutlass.const_expr(not single_query):
                    acc1[j * 4] = acc1[j * 4] * alpha0_q1
                    acc1[j * 4 + 1] = acc1[j * 4 + 1] * alpha0_q1
                    acc1[j * 4 + 2] = acc1[j * 4 + 2] * alpha1_q1
                    acc1[j * 4 + 3] = acc1[j * 4 + 3] * alpha1_q1

            # Stage ALL P(ks=0..3) (16,384 B) into p_addr and sum_nt0 (2,048 B) into sum_addr!
            dst_p = p_addr + bw * Int32(4096) + lane * Int32(16) + sub * Int32(8)
            for g_idx in cutlass.range_constexpr(4):
                base_g = dst_p + Int32(g_idx * 1024)
                st_shared_u32(
                    base_g,
                    pack_f32x2_to_bfloat2(prob0[0] * sa[g_idx], prob0[1] * sb[g_idx]),
                )
                st_shared_u32(
                    base_g + Int32(4),
                    pack_f32x2_to_bfloat2(prob0[2] * sa[g_idx], prob0[3] * sb[g_idx]),
                )
                if cutlass.const_expr(not single_query):
                    st_shared_u32(
                        base_g + Int32(512),
                        pack_f32x2_to_bfloat2(prob1[0] * sa[g_idx], prob1[1] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(516),
                        pack_f32x2_to_bfloat2(prob1[2] * sa[g_idx], prob1[3] * sb[g_idx]),
                    )

            if sub == Int32(0):
                dst_sum = sum_addr + bw * Int32(512) + lane * Int32(16)
                st_shared_v4_f32(
                    dst_sum,
                    prob0[0] + prob0[1],
                    prob0[2] + prob0[3],
                    prob1[0] + prob1[1],
                    prob1[2] + prob1[3],
                )

            # Barrier B (ONLY 2nd barrier of chunk!): P(ks=0..3) and sum_nt0 ready!
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            if sub == Int32(1):
                src_sum = sum_addr + bw * Int32(512) + lane * Int32(16)
                s0_q0, s1_q0, s0_q1, s1_q1 = ld_shared_v4_f32(src_sum)
                stat_l0_q0 = stat_l0_q0 * alpha0_q0 + ((s0_q0 + prob0[0]) + prob0[1])
                stat_l1_q0 = stat_l1_q0 * alpha1_q0 + ((s1_q0 + prob0[2]) + prob0[3])
                if cutlass.const_expr(not single_query):
                    stat_l0_q1 = stat_l0_q1 * alpha0_q1 + ((s0_q1 + prob1[0]) + prob1[1])
                    stat_l1_q1 = stat_l1_q1 * alpha1_q1 + ((s1_q1 + prob1[2]) + prob1[3])

            src_p = p_addr + bw * Int32(1024) + lane * Int32(16)
            a0_q0, a1_q0, a2_q0, a3_q0 = ld_shared_v4_u32(src_p)
            if cutlass.const_expr(not single_query):
                a0_q1, a1_q1, a2_q1, a3_q1 = ld_shared_v4_u32(src_p + Int32(512))
            p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0 = ld_shared_v4_u32(src_p + Int32(4096))
            if cutlass.const_expr(not single_query):
                p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1 = ld_shared_v4_u32(src_p + Int32(4608))
            p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0 = ld_shared_v4_u32(src_p + Int32(8192))
            if cutlass.const_expr(not single_query):
                p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1 = ld_shared_v4_u32(src_p + Int32(8704))
            p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0 = ld_shared_v4_u32(src_p + Int32(12288))
            if cutlass.const_expr(not single_query):
                p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1 = ld_shared_v4_u32(src_p + Int32(12800))

            # PV ks = 0 (from preloaded v0_regs).
            for b in cutlass.range_constexpr(4):
                f0, f1 = v0_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    a0_q0, a1_q0, a2_q0, a3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    a0_q0, a1_q0, a2_q0, a3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b2, b3,
                    )

            # PV ks = 1 (from preloaded v1_regs and p1_*).
            for b in cutlass.range_constexpr(4):
                f0, f1 = v1_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b2, b3,
                    )

            # PV ks = 2 (from preloaded v2_regs and p2_*).
            for b in cutlass.range_constexpr(4):
                f0, f1 = v2_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p2_0_q1, p2_1_q1, p2_2_q1, p2_3_q1, b2, b3,
                    )

            # PV ks = 3 (from preloaded v3_regs).
            for b in cutlass.range_constexpr(4):
                f0, f1 = v3_regs[b]
                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                n0 = b * 2
                n1 = b * 2 + 1
                acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                    p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b0, b1,
                )
                acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                    p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b2, b3,
                )
                if cutlass.const_expr(not single_query):
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b2, b3,
                    )

            for i in cutlass.range_constexpr(32):
                acc0_frag[i] = acc0[i]
                if cutlass.const_expr(not single_query):
                    acc1_frag[i] = acc1[i]

        # Epilogue: combine per-lane partial row sums from odd warps (sub == 1), normalize, store.
        l0_q0 = stat_l0_q0
        l1_q0 = stat_l1_q0
        l0_q1 = stat_l0_q1
        l1_q1 = stat_l1_q1
        for off in (1, 2):
            l0_q0 = l0_q0 + cute.arch.shuffle_sync_bfly(l0_q0, offset=off)
            l1_q0 = l1_q0 + cute.arch.shuffle_sync_bfly(l1_q0, offset=off)
            if cutlass.const_expr(not single_query):
                l0_q1 = l0_q1 + cute.arch.shuffle_sync_bfly(l0_q1, offset=off)
                l1_q1 = l1_q1 + cute.arch.shuffle_sync_bfly(l1_q1, offset=off)
        if sub == Int32(1) and tid == Int32(0):
            st_shared_f32(red_addr + (bw * Int32(16) + gid) * Int32(4), l0_q0)
            st_shared_f32(red_addr + (bw * Int32(16) + gid + Int32(8)) * Int32(4), l1_q0)
            if cutlass.const_expr(not single_query):
                st_shared_f32(red_addr + Int32(64 * 4) + (bw * Int32(16) + gid) * Int32(4), l0_q1)
                st_shared_f32(red_addr + Int32(64 * 4) + (bw * Int32(16) + gid + Int32(8)) * Int32(4), l1_q1)
        cute.arch.barrier(barrier_id=2, number_of_threads=256)

        t0_q0 = Float32(0.0)
        t1_q0 = Float32(0.0)
        t0_q1 = Float32(0.0)
        t1_q1 = Float32(0.0)
        for w in cutlass.range_constexpr(MATH_WARPS):
            t0_q0 = t0_q0 + ld_shared_f32(red_addr + Int32((w * 16) * 4) + gid * Int32(4))
            t1_q0 = t1_q0 + ld_shared_f32(red_addr + Int32((w * 16 + 8) * 4) + gid * Int32(4))
            if cutlass.const_expr(not single_query):
                t0_q1 = t0_q1 + ld_shared_f32(red_addr + Int32((64 + w * 16) * 4) + gid * Int32(4))
                t1_q1 = t1_q1 + ld_shared_f32(red_addr + Int32((64 + w * 16 + 8) * 4) + gid * Int32(4))

        h0 = head_base + gid
        h1 = h0 + Int32(8)
        col_off = (v_col_off + tid * Int32(2)).to(Int64)

        if query_valid_0 != Int32(0):
            inv0_q0 = Float32(0.0)
            inv1_q0 = Float32(0.0)
            if t0_q0 > Float32(0.0):
                inv0_q0 = Float32(1.0) / t0_q0
            if t1_q0 > Float32(0.0):
                inv1_q0 = Float32(1.0) / t1_q0
            out0_q0 = get_ptr_as_int64(
                output,
                (query_row_0.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                + col_off,
            )
            out1_q0 = out0_q0 + Int64(8 * VALUE_DIM * 2)
            for j in cutlass.range_constexpr(8):
                st_global_u32(
                    out0_q0 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc0_frag[j * 4] * inv0_q0, acc0_frag[j * 4 + 1] * inv0_q0),
                )
                st_global_u32(
                    out1_q0 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc0_frag[j * 4 + 2] * inv1_q0, acc0_frag[j * 4 + 3] * inv1_q0),
                )
            if warp_id == Int32(0) and tid == Int32(0):
                lse0_q0 = Float32(-Float32.inf)
                lse1_q0 = Float32(-Float32.inf)
                if t0_q0 > Float32(0.0):
                    lse0_q0 = (log2_approx(t0_q0) + stat_m0_q0) * Float32(0.6931471805599453)
                if t1_q0 > Float32(0.0):
                    lse1_q0 = (log2_approx(t1_q0) + stat_m1_q0) * Float32(0.6931471805599453)
                final_lse[query_row_0, h0] = lse0_q0
                final_lse[query_row_0, h1] = lse1_q0

        if cutlass.const_expr(not single_query):
            if query_valid_1 != Int32(0):
                inv0_q1 = Float32(0.0)
                inv1_q1 = Float32(0.0)
                if t0_q1 > Float32(0.0):
                    inv0_q1 = Float32(1.0) / t0_q1
                if t1_q1 > Float32(0.0):
                    inv1_q1 = Float32(1.0) / t1_q1
                out0_q1 = get_ptr_as_int64(
                    output,
                    (query_row_1.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                    + col_off,
                )
                out1_q1 = out0_q1 + Int64(8 * VALUE_DIM * 2)
                for j in cutlass.range_constexpr(8):
                    st_global_u32(
                        out0_q1 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc1_frag[j * 4] * inv0_q1, acc1_frag[j * 4 + 1] * inv0_q1),
                    )
                    st_global_u32(
                        out1_q1 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc1_frag[j * 4 + 2] * inv1_q1, acc1_frag[j * 4 + 3] * inv1_q1),
                    )
                if warp_id == Int32(1) and tid == Int32(0):
                    lse0_q1 = Float32(-Float32.inf)
                    lse1_q1 = Float32(-Float32.inf)
                    if t0_q1 > Float32(0.0):
                        lse0_q1 = (log2_approx(t0_q1) + stat_m0_q1) * Float32(0.6931471805599453)
                    if t1_q1 > Float32(0.0):
                        lse1_q1 = (log2_approx(t1_q1) + stat_m1_q1) * Float32(0.6931471805599453)
                    final_lse[query_row_1, h0] = lse0_q1
                    final_lse[query_row_1, h1] = lse1_q1

    @cute.jit
    def _run_pair_3stage(
        self, q_addr: Int32, kv_base: Int32, red_addr: Int32,
        mbar_base, active_chunks: Int32,
        visible_end_0: Int32, visible_end_1: Int32,
        query_row_0: Int32, query_row_1: Int32,
        query_valid_0: Int32, query_valid_1: Int32,
        head_base: Int32, warp_id: Int32, lane: Int32, scale_log2: Float32,
        output: cute.Tensor, final_lse: cute.Tensor,
        *, single_query: cutlass.Constexpr = False,
    ):
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        bw = warp_id >> Int32(1)
        sub = warp_id & Int32(1)
        first_in_half = (warp_id & Int32(3)) * Int32(8)

        acc0_frag = cute.make_rmem_tensor(32, Float32)
        acc1_frag = cute.make_rmem_tensor(32, Float32)
        for i in cutlass.range_constexpr(32):
            acc0_frag[i] = Float32(0.0)
            acc1_frag[i] = Float32(0.0)

        stat_m0_q0 = Float32(_MASK)
        stat_m1_q0 = Float32(_MASK)
        stat_m0_q1 = Float32(_MASK)
        stat_m1_q1 = Float32(_MASK)
        stat_l0_q0 = Float32(0.0)
        stat_l1_q0 = Float32(0.0)
        stat_l0_q1 = Float32(0.0)
        stat_l1_q1 = Float32(0.0)

        l16 = lane & Int32(15)
        v_row = (
            (l16 & Int32(1))
            | ((l16 & Int32(4)) >> Int32(1))
            | ((l16 & Int32(8)) >> Int32(1))
            | ((l16 & Int32(2)) << Int32(2))
        )
        v_col_off = bw * Int32(GROUP) + sub * Int32(64)
        q_pad_lane = (
            q_addr
            + (gid >> Int32(1)) * Int32(Q_ROW_BYTES)
            + Int32(QK_DIM * 2)
            + (gid & Int32(1)) * Int32(8)
        )

        stage = Int32(0)
        phase = Int32(0)
        if cutlass.const_expr(single_query):
            if visible_end_0 <= Int32(32) and active_chunks == Int32(1):
                cute.arch.mbarrier_wait(mbar_base + Int32(0), phase=Int32(0))
                qk0, _, sa, sb = self._scores_unified_pair(
                    q_addr, kv_base, first_in_half, lane, single_query=True
                )
                v_lane0 = kv_base + v_row * Int32(RECORD_BYTES) + v_col_off
                v_lane1 = v_lane0 + Int32(16 * RECORD_BYTES)
                v0_regs = []
                v1_regs = []
                for b in cutlass.range_constexpr(4):
                    v0_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane0 + Int32(b * 16)))
                    v1_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))

                c0 = warp_id * Int32(8) + tid * Int32(2)
                if c0 >= visible_end_0:
                    qk0[0] = Float32(_MASK)
                    qk0[2] = Float32(_MASK)
                if c0 + Int32(1) >= visible_end_0:
                    qk0[1] = Float32(_MASK)
                    qk0[3] = Float32(_MASK)
                for r in cutlass.range_constexpr(4):
                    qk0[r] = qk0[r] * scale_log2
                mx0_q0 = fmax_f32(qk0[0], qk0[1])
                mx1_q0 = fmax_f32(qk0[2], qk0[3])
                for off in (1, 2):
                    mx0_q0 = fmax_f32(mx0_q0, cute.arch.shuffle_sync_bfly(mx0_q0, offset=off))
                    mx1_q0 = fmax_f32(mx1_q0, cute.arch.shuffle_sync_bfly(mx1_q0, offset=off))
                if tid == Int32(0):
                    st_shared_f32(red_addr + (warp_id * Int32(16) + gid) * Int32(4), mx0_q0)
                    st_shared_f32(red_addr + (warp_id * Int32(16) + gid + Int32(8)) * Int32(4), mx1_q0)

                cute.arch.barrier(barrier_id=2, number_of_threads=256)

                bm0_q0 = stat_m0_q0
                bm1_q0 = stat_m1_q0
                for w in cutlass.range_constexpr(8):
                    bm0_q0 = fmax_f32(bm0_q0, ld_shared_f32(red_addr + Int32((w * 16) * 4) + gid * Int32(4)))
                    bm1_q0 = fmax_f32(bm1_q0, ld_shared_f32(red_addr + Int32((w * 16 + 8) * 4) + gid * Int32(4)))
                stat_m0_q0 = bm0_q0
                stat_m1_q0 = bm1_q0

                prob0 = [
                    exp2_approx(qk0[0] - bm0_q0),
                    exp2_approx(qk0[1] - bm0_q0),
                    exp2_approx(qk0[2] - bm1_q0),
                    exp2_approx(qk0[3] - bm1_q0),
                ]
                kv_addr1 = kv_base + Int32(32 * RECORD_BYTES)
                dst_p = kv_addr1 + bw * Int32(4096) + lane * Int32(16) + sub * Int32(8)
                for g_idx in cutlass.range_constexpr(4):
                    base_g = dst_p + Int32(g_idx * 1024)
                    st_shared_u32(
                        base_g,
                        pack_f32x2_to_bfloat2(prob0[0] * sa[g_idx], prob0[1] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(4),
                        pack_f32x2_to_bfloat2(prob0[2] * sa[g_idx], prob0[3] * sb[g_idx]),
                    )
                if sub == Int32(0):
                    dst_sum = kv_addr1 + Int32(16384) + bw * Int32(512) + lane * Int32(16)
                    st_shared_v4_f32(
                        dst_sum,
                        prob0[0] + prob0[1],
                        prob0[2] + prob0[3],
                        Float32(0.0),
                        Float32(0.0),
                    )
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                if sub == Int32(1):
                    src_sum = kv_addr1 + Int32(16384) + bw * Int32(512) + lane * Int32(16)
                    s0_q0, s1_q0, _, _ = ld_shared_v4_f32(src_sum)
                    stat_l0_q0 = (s0_q0 + prob0[0]) + prob0[1]
                    stat_l1_q0 = (s1_q0 + prob0[2]) + prob0[3]
                src_p = kv_addr1 + bw * Int32(1024) + lane * Int32(16)
                a0_q0, a1_q0, a2_q0, a3_q0 = ld_shared_v4_u32(src_p)
                p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0 = ld_shared_v4_u32(src_p + Int32(4096))
                acc0 = [acc0_frag[i] for i in range(32)]
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v0_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b2, b3,
                    )
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v1_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b2, b3,
                    )
                for i in cutlass.range_constexpr(32):
                    acc0_frag[i] = acc0[i]
                active_chunks = Int32(0)
        for local in cutlass.range(active_chunks, unroll=1):
            chunk_begin = Int32(local) * Int32(64)
            s0 = stage
            p0 = phase
            stage += Int32(1)
            if stage == Int32(3):
                stage = Int32(0)
                phase ^= Int32(1)
            s1 = stage
            p1 = phase
            stage += Int32(1)
            if stage == Int32(3):
                stage = Int32(0)
                phase ^= Int32(1)

            kv_addr0 = kv_base + s0 * Int32(32 * RECORD_BYTES)
            kv_addr1 = kv_base + s1 * Int32(32 * RECORD_BYTES)
            s_wait = s0
            p_wait = p0
            s_other = s1
            p_other = p1
            s_kv_score = kv_addr0
            if warp_id >= Int32(4):
                s_wait = s1
                p_wait = p1
                s_other = s0
                p_other = p0
                s_kv_score = kv_addr1
            cute.arch.mbarrier_wait(mbar_base + s_wait, phase=p_wait)

            qk0, qk1, sa, sb = self._scores_unified_pair(
                q_addr, s_kv_score, first_in_half, lane, single_query=single_query
            )

            cute.arch.mbarrier_wait(mbar_base + s_other, phase=p_other)
            v_lane0 = kv_addr0 + v_row * Int32(RECORD_BYTES) + v_col_off
            v_lane2 = kv_addr1 + v_row * Int32(RECORD_BYTES) + v_col_off
            v0_regs = []
            v2_regs = []
            for b in cutlass.range_constexpr(4):
                v0_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane0 + Int32(b * 16)))
                v2_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane2 + Int32(b * 16)))
            if cutlass.const_expr(single_query):
                v_lane1 = v_lane0 + Int32(16 * RECORD_BYTES)
                v_lane3 = v_lane2 + Int32(16 * RECORD_BYTES)
                v1_regs = []
                v3_regs = []
                for b in cutlass.range_constexpr(4):
                    v1_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))
                    v3_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane3 + Int32(b * 16)))
                cute.arch.sync_warp()
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(3) + s0)

            c0 = chunk_begin + warp_id * Int32(8) + tid * Int32(2)
            if c0 >= visible_end_0:
                qk0[0] = Float32(_MASK)
                qk0[2] = Float32(_MASK)
            if c0 + Int32(1) >= visible_end_0:
                qk0[1] = Float32(_MASK)
                qk0[3] = Float32(_MASK)
            for r in cutlass.range_constexpr(4):
                qk0[r] = qk0[r] * scale_log2
            mx0_q0 = fmax_f32(qk0[0], qk0[1])
            mx1_q0 = fmax_f32(qk0[2], qk0[3])
            for off in (1, 2):
                mx0_q0 = fmax_f32(mx0_q0, cute.arch.shuffle_sync_bfly(mx0_q0, offset=off))
                mx1_q0 = fmax_f32(mx1_q0, cute.arch.shuffle_sync_bfly(mx1_q0, offset=off))

            if cutlass.const_expr(not single_query):
                if c0 >= visible_end_1:
                    qk1[0] = Float32(_MASK)
                    qk1[2] = Float32(_MASK)
                if c0 + Int32(1) >= visible_end_1:
                    qk1[1] = Float32(_MASK)
                    qk1[3] = Float32(_MASK)
                for r in cutlass.range_constexpr(4):
                    qk1[r] = qk1[r] * scale_log2
                mx0_q1 = fmax_f32(qk1[0], qk1[1])
                mx1_q1 = fmax_f32(qk1[2], qk1[3])
                for off in (1, 2):
                    mx0_q1 = fmax_f32(mx0_q1, cute.arch.shuffle_sync_bfly(mx0_q1, offset=off))
                    mx1_q1 = fmax_f32(mx1_q1, cute.arch.shuffle_sync_bfly(mx1_q1, offset=off))

            if tid == Int32(0):
                st_shared_f32(red_addr + (warp_id * Int32(16) + gid) * Int32(4), mx0_q0)
                st_shared_f32(red_addr + (warp_id * Int32(16) + gid + Int32(8)) * Int32(4), mx1_q0)
                if cutlass.const_expr(not single_query):
                    q_pad_w = q_pad_lane + warp_id * Int32(4 * Q_ROW_BYTES)
                    st_shared_f32(q_pad_w, mx0_q1)
                    st_shared_f32(q_pad_w + Int32(4), mx1_q1)

            # Barrier A: _scores and V(ks=0, 2) reads complete across all 8 warps!
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            if cutlass.const_expr(not single_query):
                # Preload V(ks=1) from s0[16..31] before Barrier B.
                v_lane1 = v_lane0 + Int32(16 * RECORD_BYTES)
                v1_regs = []
                for b in cutlass.range_constexpr(4):
                    v1_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane1 + Int32(b * 16)))

            bm0_q0 = stat_m0_q0
            bm1_q0 = stat_m1_q0
            bm0_q1 = stat_m0_q1
            bm1_q1 = stat_m1_q1
            for w in cutlass.range_constexpr(8):
                bm0_q0 = fmax_f32(bm0_q0, ld_shared_f32(red_addr + Int32((w * 16) * 4) + gid * Int32(4)))
                bm1_q0 = fmax_f32(bm1_q0, ld_shared_f32(red_addr + Int32((w * 16 + 8) * 4) + gid * Int32(4)))
                if cutlass.const_expr(not single_query):
                    bm0_q1 = fmax_f32(bm0_q1, ld_shared_f32(q_pad_lane + Int32(w * 4 * Q_ROW_BYTES)))
                    bm1_q1 = fmax_f32(bm1_q1, ld_shared_f32(q_pad_lane + Int32(w * 4 * Q_ROW_BYTES + 4)))

            if cutlass.const_expr(not single_query):
                if chunk_begin >= visible_end_0:
                    bm0_q0 = stat_m0_q0
                    bm1_q0 = stat_m1_q0

            alpha0_q0 = exp2_approx(stat_m0_q0 - bm0_q0)
            alpha1_q0 = exp2_approx(stat_m1_q0 - bm1_q0)
            stat_m0_q0 = bm0_q0
            stat_m1_q0 = bm1_q0

            prob0 = [
                exp2_approx(qk0[0] - bm0_q0),
                exp2_approx(qk0[1] - bm0_q0),
                exp2_approx(qk0[2] - bm1_q0),
                exp2_approx(qk0[3] - bm1_q0),
            ]
            if cutlass.const_expr(not single_query):
                if chunk_begin >= visible_end_0:
                    prob0[0] = Float32(0.0)
                    prob0[1] = Float32(0.0)
                    prob0[2] = Float32(0.0)
                    prob0[3] = Float32(0.0)
                alpha0_q1 = exp2_approx(stat_m0_q1 - bm0_q1)
                alpha1_q1 = exp2_approx(stat_m1_q1 - bm1_q1)
                stat_m0_q1 = bm0_q1
                stat_m1_q1 = bm1_q1
                prob1 = [
                    exp2_approx(qk1[0] - bm0_q1),
                    exp2_approx(qk1[1] - bm0_q1),
                    exp2_approx(qk1[2] - bm1_q1),
                    exp2_approx(qk1[3] - bm1_q1),
                ]
            else:
                alpha0_q1 = Float32(1.0)
                alpha1_q1 = Float32(1.0)
                prob1 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]

            acc0 = [acc0_frag[i] for i in range(32)]
            acc1 = [acc1_frag[i] for i in range(32)]
            for j in cutlass.range_constexpr(8):
                acc0[j * 4] = acc0[j * 4] * alpha0_q0
                acc0[j * 4 + 1] = acc0[j * 4 + 1] * alpha0_q0
                acc0[j * 4 + 2] = acc0[j * 4 + 2] * alpha1_q0
                acc0[j * 4 + 3] = acc0[j * 4 + 3] * alpha1_q0
                if cutlass.const_expr(not single_query):
                    acc1[j * 4] = acc1[j * 4] * alpha0_q1
                    acc1[j * 4 + 1] = acc1[j * 4 + 1] * alpha0_q1
                    acc1[j * 4 + 2] = acc1[j * 4 + 2] * alpha1_q1
                    acc1[j * 4 + 3] = acc1[j * 4 + 3] * alpha1_q1

            if cutlass.const_expr(single_query):
                dst_p = kv_addr1 + bw * Int32(4096) + lane * Int32(16) + sub * Int32(8)
                for g_idx in cutlass.range_constexpr(4):
                    base_g = dst_p + Int32(g_idx * 1024)
                    st_shared_u32(
                        base_g,
                        pack_f32x2_to_bfloat2(prob0[0] * sa[g_idx], prob0[1] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(4),
                        pack_f32x2_to_bfloat2(prob0[2] * sa[g_idx], prob0[3] * sb[g_idx]),
                    )
                if sub == Int32(0):
                    dst_sum = kv_addr1 + Int32(16384) + bw * Int32(512) + lane * Int32(16)
                    st_shared_v4_f32(
                        dst_sum,
                        prob0[0] + prob0[1],
                        prob0[2] + prob0[3],
                        Float32(0.0),
                        Float32(0.0),
                    )
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                if sub == Int32(1):
                    src_sum = kv_addr1 + Int32(16384) + bw * Int32(512) + lane * Int32(16)
                    s0_q0, s1_q0, _, _ = ld_shared_v4_f32(src_sum)
                    stat_l0_q0 = stat_l0_q0 * alpha0_q0 + ((s0_q0 + prob0[0]) + prob0[1])
                    stat_l1_q0 = stat_l1_q0 * alpha1_q0 + ((s1_q0 + prob0[2]) + prob0[3])
                src_p = kv_addr1 + bw * Int32(1024) + lane * Int32(16)
                a0_q0, a1_q0, a2_q0, a3_q0 = ld_shared_v4_u32(src_p)
                p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0 = ld_shared_v4_u32(src_p + Int32(4096))
                p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0 = ld_shared_v4_u32(src_p + Int32(8192))
                p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0 = ld_shared_v4_u32(src_p + Int32(12288))
                cute.arch.sync_warp()
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(3) + s1)
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v0_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b2, b3,
                    )
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v1_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b2, b3,
                    )
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v2_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p2_0_q0, p2_1_q0, p2_2_q0, p2_3_q0, b2, b3,
                    )
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v3_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b2, b3,
                    )
                for i in cutlass.range_constexpr(32):
                    acc0_frag[i] = acc0[i]
            else:
                # Stage ALL P(ks=0, 1) into dead s0[0..8192) and P(ks=2, 3) into dead s1[0..8192)!
                kv_p_dst = kv_addr0
                if bw >= Int32(2):
                    kv_p_dst = kv_addr1
                dst_p = kv_p_dst + (bw & Int32(1)) * Int32(4096) + lane * Int32(16) + sub * Int32(8)
                for g_idx in cutlass.range_constexpr(4):
                    base_g = dst_p + Int32(g_idx * 1024)
                    st_shared_u32(
                        base_g,
                        pack_f32x2_to_bfloat2(prob0[0] * sa[g_idx], prob0[1] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(4),
                        pack_f32x2_to_bfloat2(prob0[2] * sa[g_idx], prob0[3] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(512),
                        pack_f32x2_to_bfloat2(prob1[0] * sa[g_idx], prob1[1] * sb[g_idx]),
                    )
                    st_shared_u32(
                        base_g + Int32(516),
                        pack_f32x2_to_bfloat2(prob1[2] * sa[g_idx], prob1[3] * sb[g_idx]),
                    )

                # Stage sum_nt0 for all 4 bw groups into dead s0[8192..10240) (2,048 B)!
                if sub == Int32(0):
                    dst_sum = kv_addr0 + Int32(8192) + bw * Int32(512) + lane * Int32(16)
                    st_shared_v4_f32(
                        dst_sum,
                        prob0[0] + prob0[1],
                        prob0[2] + prob0[3],
                        prob1[0] + prob1[1],
                        prob1[2] + prob1[3],
                    )

                # Barrier B (ONLY 2nd barrier of chunk!): P(ks=0..3) and sum_nt0 ready!
                cute.arch.barrier(barrier_id=2, number_of_threads=256)

                if sub == Int32(1):
                    src_sum = kv_addr0 + Int32(8192) + bw * Int32(512) + lane * Int32(16)
                    s0_q0, s1_q0, s0_q1, s1_q1 = ld_shared_v4_f32(src_sum)
                    stat_l0_q0 = stat_l0_q0 * alpha0_q0 + ((s0_q0 + prob0[0]) + prob0[1])
                    stat_l1_q0 = stat_l1_q0 * alpha1_q0 + ((s1_q0 + prob0[2]) + prob0[3])
                    stat_l0_q1 = stat_l0_q1 * alpha0_q1 + ((s0_q1 + prob1[0]) + prob1[1])
                    stat_l1_q1 = stat_l1_q1 * alpha1_q1 + ((s1_q1 + prob1[2]) + prob1[3])

                src_p01 = kv_addr0 + bw * Int32(1024) + lane * Int32(16)
                a0_q0, a1_q0, a2_q0, a3_q0 = ld_shared_v4_u32(src_p01)
                a0_q1, a1_q1, a2_q1, a3_q1 = ld_shared_v4_u32(src_p01 + Int32(512))
                p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0 = ld_shared_v4_u32(src_p01 + Int32(4096))
                p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1 = ld_shared_v4_u32(src_p01 + Int32(4608))

                # Release s0 IMMEDIATELY at Barrier B before PV(ks=0..3) runs!
                cute.arch.sync_warp()
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(3) + s0)

                # PV ks = 0 (from preloaded v0_regs).
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v0_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b2, b3,
                    )
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b2, b3,
                    )

                # PV ks = 1 (from preloaded v1_regs and p1_*).
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v1_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p1_0_q0, p1_1_q0, p1_2_q0, p1_3_q0, b2, b3,
                    )
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p1_0_q1, p1_1_q1, p1_2_q1, p1_3_q1, b2, b3,
                    )

                # Load P(ks=2, 3) and V(ks=3) from s1, then release s1 BEFORE PV(ks=2, 3) runs!
                src_p23 = kv_addr1 + bw * Int32(1024) + lane * Int32(16)
                a0_q0, a1_q0, a2_q0, a3_q0 = ld_shared_v4_u32(src_p23)
                a0_q1, a1_q1, a2_q1, a3_q1 = ld_shared_v4_u32(src_p23 + Int32(512))
                p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0 = ld_shared_v4_u32(src_p23 + Int32(4096))
                p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1 = ld_shared_v4_u32(src_p23 + Int32(4608))

                v_lane3 = v_lane2 + Int32(16 * RECORD_BYTES)
                v3_regs = []
                for b in cutlass.range_constexpr(4):
                    v3_regs.append(ldmatrix_m16n16x1_trans_b8(v_lane3 + Int32(b * 16)))

                cute.arch.sync_warp()
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(3) + s1)

                # PV ks = 2 (from preloaded v2_regs).
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v2_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        a0_q0, a1_q0, a2_q0, a3_q0, b2, b3,
                    )
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        a0_q1, a1_q1, a2_q1, a3_q1, b2, b3,
                    )

                # PV ks = 3 (from preloaded v3_regs).
                for b in cutlass.range_constexpr(4):
                    f0, f1 = v3_regs[b]
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                    n0 = b * 2
                    n1 = b * 2 + 1
                    acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b0, b1,
                    )
                    acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        p3_0_q0, p3_1_q0, p3_2_q0, p3_3_q0, b2, b3,
                    )
                    acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b0, b1,
                    )
                    acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                        acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        p3_0_q1, p3_1_q1, p3_2_q1, p3_3_q1, b2, b3,
                    )

                for i in cutlass.range_constexpr(32):
                    acc0_frag[i] = acc0[i]
                    acc1_frag[i] = acc1[i]

        # Epilogue: combine per-lane partial row sums from odd warps (sub == 1), normalize, store.
        l0_q0 = stat_l0_q0
        l1_q0 = stat_l1_q0
        l0_q1 = stat_l0_q1
        l1_q1 = stat_l1_q1
        for off in (1, 2):
            l0_q0 = l0_q0 + cute.arch.shuffle_sync_bfly(l0_q0, offset=off)
            l1_q0 = l1_q0 + cute.arch.shuffle_sync_bfly(l1_q0, offset=off)
            if cutlass.const_expr(not single_query):
                l0_q1 = l0_q1 + cute.arch.shuffle_sync_bfly(l0_q1, offset=off)
                l1_q1 = l1_q1 + cute.arch.shuffle_sync_bfly(l1_q1, offset=off)
        if sub == Int32(1) and tid == Int32(0):
            st_shared_f32(red_addr + (bw * Int32(16) + gid) * Int32(4), l0_q0)
            st_shared_f32(red_addr + (bw * Int32(16) + gid + Int32(8)) * Int32(4), l1_q0)
            if cutlass.const_expr(not single_query):
                st_shared_f32(red_addr + Int32(64 * 4) + (bw * Int32(16) + gid) * Int32(4), l0_q1)
                st_shared_f32(red_addr + Int32(64 * 4) + (bw * Int32(16) + gid + Int32(8)) * Int32(4), l1_q1)
        cute.arch.barrier(barrier_id=2, number_of_threads=256)

        t0_q0 = Float32(0.0)
        t1_q0 = Float32(0.0)
        t0_q1 = Float32(0.0)
        t1_q1 = Float32(0.0)
        for w in cutlass.range_constexpr(MATH_WARPS):
            t0_q0 = t0_q0 + ld_shared_f32(red_addr + Int32((w * 16) * 4) + gid * Int32(4))
            t1_q0 = t1_q0 + ld_shared_f32(red_addr + Int32((w * 16 + 8) * 4) + gid * Int32(4))
            if cutlass.const_expr(not single_query):
                t0_q1 = t0_q1 + ld_shared_f32(red_addr + Int32((64 + w * 16) * 4) + gid * Int32(4))
                t1_q1 = t1_q1 + ld_shared_f32(red_addr + Int32((64 + w * 16 + 8) * 4) + gid * Int32(4))

        h0 = head_base + gid
        h1 = h0 + Int32(8)
        col_off = (v_col_off + tid * Int32(2)).to(Int64)

        if query_valid_0 != Int32(0):
            inv0_q0 = Float32(0.0)
            inv1_q0 = Float32(0.0)
            if t0_q0 > Float32(0.0):
                inv0_q0 = Float32(1.0) / t0_q0
            if t1_q0 > Float32(0.0):
                inv1_q0 = Float32(1.0) / t1_q0
            out0_q0 = get_ptr_as_int64(
                output,
                (query_row_0.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                + col_off,
            )
            out1_q0 = out0_q0 + Int64(8 * VALUE_DIM * 2)
            for j in cutlass.range_constexpr(8):
                st_global_u32(
                    out0_q0 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc0_frag[j * 4] * inv0_q0, acc0_frag[j * 4 + 1] * inv0_q0),
                )
                st_global_u32(
                    out1_q0 + Int64(j * 16),
                    pack_f32x2_to_bfloat2(acc0_frag[j * 4 + 2] * inv1_q0, acc0_frag[j * 4 + 3] * inv1_q0),
                )
            if warp_id == Int32(0) and tid == Int32(0):
                lse0_q0 = Float32(-Float32.inf)
                lse1_q0 = Float32(-Float32.inf)
                if t0_q0 > Float32(0.0):
                    lse0_q0 = (log2_approx(t0_q0) + stat_m0_q0) * Float32(0.6931471805599453)
                if t1_q0 > Float32(0.0):
                    lse1_q0 = (log2_approx(t1_q0) + stat_m1_q0) * Float32(0.6931471805599453)
                final_lse[query_row_0, h0] = lse0_q0
                final_lse[query_row_0, h1] = lse1_q0

        if cutlass.const_expr(not single_query):
            if query_valid_1 != Int32(0):
                inv0_q1 = Float32(0.0)
                inv1_q1 = Float32(0.0)
                if t0_q1 > Float32(0.0):
                    inv0_q1 = Float32(1.0) / t0_q1
                if t1_q1 > Float32(0.0):
                    inv1_q1 = Float32(1.0) / t1_q1
                out0_q1 = get_ptr_as_int64(
                    output,
                    (query_row_1.to(Int64) * Int64(self.num_heads) + h0.to(Int64)) * Int64(VALUE_DIM)
                    + col_off,
                )
                out1_q1 = out0_q1 + Int64(8 * VALUE_DIM * 2)
                for j in cutlass.range_constexpr(8):
                    st_global_u32(
                        out0_q1 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc1_frag[j * 4] * inv0_q1, acc1_frag[j * 4 + 1] * inv0_q1),
                    )
                    st_global_u32(
                        out1_q1 + Int64(j * 16),
                        pack_f32x2_to_bfloat2(acc1_frag[j * 4 + 2] * inv1_q1, acc1_frag[j * 4 + 3] * inv1_q1),
                    )
                if warp_id == Int32(1) and tid == Int32(0):
                    lse0_q1 = Float32(-Float32.inf)
                    lse1_q1 = Float32(-Float32.inf)
                    if t0_q1 > Float32(0.0):
                        lse0_q1 = (log2_approx(t0_q1) + stat_m0_q1) * Float32(0.6931471805599453)
                    if t1_q1 > Float32(0.0):
                        lse1_q1 = (log2_approx(t1_q1) + stat_m1_q1) * Float32(0.6931471805599453)
                    final_lse[query_row_1, h0] = lse0_q1
                    final_lse[query_row_1, h1] = lse1_q1

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel_pair(
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
        head_tile, raw_tile, _ = cute.arch.block_idx()
        raw_tile = Int32(raw_tile)
        tile_index = raw_tile
        if batch == Int32(1):
            full_two_row_tiles = total_q // Int32(2)
            if raw_tile < full_two_row_tiles:
                tile_index = (full_two_row_tiles - Int32(1)) - raw_tile
        head_base = Int32(head_tile) * Int32(HEADS_PER_TILE)

        request = Int32(0)
        query_start = Int32(0)
        tile_valid = Int32(1)
        remaining = tile_index
        scan = Int32(0)
        found = Int32(0)
        tile_offset = Int32(0)
        while scan < batch:
            if found == Int32(0):
                scan_begin = Int32(cu_seqlens_q[scan])
                scan_end = Int32(cu_seqlens_q[scan + Int32(1)])
                scan_tiles = (scan_end - scan_begin + Int32(1)) // Int32(2)
                if remaining < scan_tiles:
                    request = scan
                    tile_offset = remaining
                    found = Int32(1)
                else:
                    remaining -= scan_tiles
            scan += Int32(1)
        tile_valid = found
        query_start = Int32(cu_seqlens_q[request]) + tile_offset * Int32(2)
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
        query_length = query_end - query_begin
        cache_length = Int32(cache_seqlens[request])
        causal_end = cache_length - query_length + (query_start - query_begin) + Int32(2)
        if causal_end < Int32(0):
            causal_end = Int32(0)
        if causal_end > cache_length:
            causal_end = cache_length
        active_chunks = (causal_end + Int32(self.chunk - 1)) // Int32(self.chunk)
        if tile_valid == Int32(0):
            active_chunks = Int32(0)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage_pair())
        kv_addr = shared_ptr_to_u32(storage.kv.data_ptr())
        q_addr = shared_ptr_to_u32(storage.q.data_ptr())
        red_addr = shared_ptr_to_u32(storage.reduce.data_ptr())
        mbar_base = storage.mbar.data_ptr()

        if tid == Int32(0):
            for s in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))
                cute.arch.mbarrier_init(mbar_base + Int32(3 + s), Int32(8))
        cute.arch.barrier()

        if warp_id == Int32(8):
            producer_phase = Int32(1)
            stage = Int32(0)
            table_base = request.to(Int64) * page_table_stride
            active_half_chunks = active_chunks * Int32(2)
            if query_start + Int32(1) >= total_q and cache_length <= Int32(32) and active_chunks == Int32(1):
                active_half_chunks = Int32(1)
            for h in cutlass.range(active_half_chunks, unroll=1):
                token_begin = Int32(h) * Int32(32)
                cute.arch.mbarrier_wait(
                    mbar_base + Int32(3) + stage, phase=producer_phase
                )
                full = mbar_base + stage
                dst = kv_addr + stage * Int32(32 * RECORD_BYTES)
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full, Int32(32 * RECORD_BYTES)
                    )
                if token_begin + Int32(32) <= cache_length:
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
                            dst, src, Int32(32 * RECORD_BYTES),
                            shared_ptr_to_u32(full),
                        )
                else:
                    valid_count = cache_length - token_begin
                    if valid_count > Int32(0) and lane == Int32(0):
                        page = token_begin // Int32(self.page_size)
                        physical = Int32(page_table[table_base + page.to(Int64)])
                        src = get_ptr_as_int64(
                            cache_bytes,
                            physical.to(Int64) * page_stride_bytes
                            + (token_begin - page * Int32(self.page_size)).to(Int64)
                            * Int64(RECORD_BYTES),
                        )
                        cp_async_bulk_g2s_mbar(
                            dst, src, valid_count * Int32(RECORD_BYTES),
                            shared_ptr_to_u32(full),
                        )
                    if lane >= valid_count:
                        physical0 = Int32(page_table[table_base])
                        src0 = get_ptr_as_int64(
                            cache_bytes,
                            physical0.to(Int64) * page_stride_bytes,
                        )
                        cp_async_bulk_g2s_mbar(
                            dst + lane * Int32(RECORD_BYTES), src0, Int32(RECORD_BYTES),
                            shared_ptr_to_u32(full),
                        )
                stage += Int32(1)
                if stage == Int32(3):
                    stage = Int32(0)
                    producer_phase ^= Int32(1)
        else:
            self._stage_query(
                q_bytes, q_addr, query_start, head_base, total_q, tid,
                q_stride_row_bytes, use_pair=True,
            )
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            query_row_0 = query_start
            query_row_1 = query_start + Int32(1)
            query_valid_0 = Int32(1)
            if query_row_0 >= total_q or query_row_0 >= query_end or tile_valid == Int32(0):
                query_valid_0 = Int32(0)
            query_valid_1 = Int32(1)
            if query_row_1 >= total_q or query_row_1 >= query_end or tile_valid == Int32(0):
                query_valid_1 = Int32(0)
            visible_end_0 = cache_length - query_length + (query_row_0 - query_begin) + Int32(1)
            if visible_end_0 > cache_length:
                visible_end_0 = cache_length
            if visible_end_0 < Int32(0):
                visible_end_0 = Int32(0)
            visible_end_1 = cache_length - query_length + (query_row_1 - query_begin) + Int32(1)
            if visible_end_1 > cache_length:
                visible_end_1 = cache_length
            if visible_end_1 < Int32(0):
                visible_end_1 = Int32(0)
            if query_valid_1 != Int32(0):
                self._run_pair_3stage(
                    q_addr, kv_addr, red_addr, mbar_base, active_chunks,
                    visible_end_0, visible_end_1,
                    query_row_0, query_row_1,
                    query_valid_0, query_valid_1,
                    head_base, warp_id, lane, sm_scale_log2,
                    output, final_lse,
                    single_query=False,
                )
            else:
                self._run_pair_3stage(
                    q_addr, kv_addr, red_addr, mbar_base, active_chunks,
                    visible_end_0, visible_end_1,
                    query_row_0, query_row_1,
                    query_valid_0, query_valid_1,
                    head_base, warp_id, lane, sm_scale_log2,
                    output, final_lse,
                    single_query=True,
                )

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
        head_tile, raw_tile, _ = cute.arch.block_idx()
        raw_tile = Int32(raw_tile)
        tile_index = raw_tile
        if cutlass.const_expr(self.eff_query_tile > 1):
            if batch == Int32(1):
                num_tiles = (total_q + Int32(self.eff_query_tile - 1)) // Int32(
                    self.eff_query_tile
                )
                if raw_tile < num_tiles:
                    tile_index = (num_tiles - Int32(1)) - raw_tile
        head_base = Int32(head_tile) * Int32(HEADS_PER_TILE)

        request = Int32(0)
        query_start = Int32(0)
        tile_valid = Int32(1)
        if cutlass.const_expr(self.eff_query_tile == 1):
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
                    scan_tiles = (
                        scan_end - scan_begin + Int32(self.eff_query_tile - 1)
                    ) // Int32(self.eff_query_tile)
                    if remaining < scan_tiles:
                        request = scan
                        tile_offset = remaining
                        found = Int32(1)
                    else:
                        remaining -= scan_tiles
                scan += Int32(1)
            tile_valid = found
            query_start = Int32(cu_seqlens_q[request]) + tile_offset * Int32(
                self.eff_query_tile
            )
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
        query_length = query_end - query_begin
        cache_length = Int32(cache_seqlens[request])
        causal_end = cache_length - query_length + (query_start - query_begin) + Int32(
            self.eff_query_tile
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
        red_addr = shared_ptr_to_u32(storage.reduce.data_ptr())
        mbar_base = storage.mbar.data_ptr()

        if cutlass.const_expr(self.unified_pair):
            if tid == Int32(0):
                cute.arch.mbarrier_init(mbar_base + Int32(0), Int32(1))
                cute.arch.mbarrier_init(mbar_base + Int32(1), Int32(1))
            cute.arch.barrier()

            query_row_0 = query_start
            query_row_1 = query_start + Int32(1)
            query_row_2 = query_start + Int32(2)
            query_row_3 = query_start + Int32(3)
            query_valid_0 = Int32(1)
            if query_row_0 >= total_q or query_row_0 >= query_end or tile_valid == Int32(0):
                query_valid_0 = Int32(0)
            query_valid_1 = Int32(1)
            if query_row_1 >= total_q or query_row_1 >= query_end or tile_valid == Int32(0):
                query_valid_1 = Int32(0)
            query_valid_2 = Int32(1)
            if query_row_2 >= total_q or query_row_2 >= query_end or tile_valid == Int32(0):
                query_valid_2 = Int32(0)
            query_valid_3 = Int32(1)
            if query_row_3 >= total_q or query_row_3 >= query_end or tile_valid == Int32(0):
                query_valid_3 = Int32(0)
            use_pair = query_valid_2 == Int32(0)

            table_base = request.to(Int64) * page_table_stride
            if active_chunks > Int32(0):
                if warp_id == Int32(0):
                    self._issue_kv_half_chunk(
                        cache_bytes, page_table, table_base, page_stride_bytes,
                        cache_length, Int32(0), kv_addr, mbar_base + Int32(0), lane,
                    )
                if use_pair and warp_id == Int32(1):
                    self._issue_kv_half_chunk(
                        cache_bytes, page_table, table_base, page_stride_bytes,
                        cache_length, Int32(32),
                        q_addr + Int32(2 * HEADS_PER_TILE * Q_ROW_BYTES),
                        mbar_base + Int32(1), lane,
                    )

            if use_pair:
                self._stage_query(
                    q_bytes, q_addr, query_start, head_base, total_q, tid,
                    q_stride_row_bytes, use_pair=True,
                )
            else:
                self._stage_query(
                    q_bytes, q_addr, query_start, head_base, total_q, tid,
                    q_stride_row_bytes, use_pair=False,
                )
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            visible_end_0 = cache_length - query_length + (query_row_0 - query_begin) + Int32(1)
            if visible_end_0 > cache_length:
                visible_end_0 = cache_length
            if visible_end_0 < Int32(0):
                visible_end_0 = Int32(0)
            visible_end_1 = cache_length - query_length + (query_row_1 - query_begin) + Int32(1)
            if visible_end_1 > cache_length:
                visible_end_1 = cache_length
            if visible_end_1 < Int32(0):
                visible_end_1 = Int32(0)
            visible_end_2 = cache_length - query_length + (query_row_2 - query_begin) + Int32(1)
            if visible_end_2 > cache_length:
                visible_end_2 = cache_length
            if visible_end_2 < Int32(0):
                visible_end_2 = Int32(0)
            visible_end_3 = cache_length - query_length + (query_row_3 - query_begin) + Int32(1)
            if visible_end_3 > cache_length:
                visible_end_3 = cache_length
            if visible_end_3 < Int32(0):
                visible_end_3 = Int32(0)
            if not use_pair:
                self._run_unified_quad(
                    q_addr, kv_addr, red_addr, mbar_base,
                    cache_bytes, page_table, table_base, page_stride_bytes, cache_length,
                    active_chunks,
                    visible_end_0, visible_end_1, visible_end_2, visible_end_3,
                    query_row_0, query_valid_3,
                    head_base, warp_id, lane, sm_scale_log2,
                    output, final_lse,
                    single_query=False,
                )
            elif query_valid_1 != Int32(0):
                self._run_unified_pair(
                    q_addr, kv_addr, red_addr, mbar_base,
                    cache_bytes, page_table, table_base, page_stride_bytes, cache_length,
                    active_chunks,
                    visible_end_0, visible_end_1,
                    query_row_0, query_row_1,
                    query_valid_0, query_valid_1,
                    head_base, warp_id, lane, sm_scale_log2,
                    output, final_lse,
                    single_query=False,
                )
            else:
                self._run_unified_pair(
                    q_addr, kv_addr, red_addr, mbar_base,
                    cache_bytes, page_table, table_base, page_stride_bytes, cache_length,
                    active_chunks,
                    visible_end_0, visible_end_1,
                    query_row_0, query_row_1,
                    query_valid_0, query_valid_1,
                    head_base, warp_id, lane, sm_scale_log2,
                    output, final_lse,
                    single_query=True,
                )
        else:
            p_addr = shared_ptr_to_u32(storage.p.data_ptr())
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
                    q_bytes, q_addr, query_start, head_base, total_q, tid,
                    q_stride_row_bytes, use_pair=True,
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
