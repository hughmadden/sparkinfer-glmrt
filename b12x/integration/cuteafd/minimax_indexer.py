"""Native MSA index-score and top-16 pointer ABIs.

Decode reads 64-row FP8 index pages (8192 key bytes then 256 scale bytes).
pool_pages must fit signed Int32; physical page offsets use Int64.
Prefill reads contiguous FP8 rows and FP32 scales: request starts must be
128-token aligned, and the K allocation must be padded to 256 rows. k_end
is the exclusive causal end for each query. max_pages bounds the whole
contiguous staging extent in 64-row units, including request alignment.
Selections are ascending request-local blocks, -1 padded, with the causal
block forced. The output head stride is max_rows, independent of live rows.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr

from b12x.attention._shared.cute import ops as attention_ops
from b12x.attention.dsa_indexer.kernel import PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT as _SPLIT

from ._common import Operand, Scalar, compile_program


def _layout(max_rows, max_pages, mode):
    if mode not in ("decode", "prefill") or min(max_rows, max_pages) <= 0:
        raise ValueError("positive capacities and decode/prefill mode required")
    blocks = (max_pages + 1) // 2
    if 4 * max_rows * blocks >= 2**31:
        raise ValueError("MSA score scratch exceeds the row selector's signed indexing extent")
    sizes = dict(weights=4 * max_rows * 4, scores=4 * max_rows * blocks * 4,
                 pages=4 * max_rows * max_pages * 4 if mode == "decode" else 0,
                 starts=4 * max_rows * 4, lengths=4 * max_rows * 4,
                 indices=4 * max_rows * 16 * 4, values=4 * max_rows * 16 * 4,
                 schedule=(max_rows + 1) * 2 * 4, controls=256)
    offsets, cursor = {}, 0
    for key, size in sizes.items():
        offsets[key] = cursor
        cursor += (size + 255) // 256 * 256
    return blocks, offsets, cursor


def minimax_index_scratch_bytes(*, max_rows, max_pages, mode="decode"):
    return _layout(int(max_rows), int(max_pages), mode)[2]


class _Index:
    def __init__(self, max_rows, max_pages, mode):
        from b12x.attention.dsa_indexer.kernel import (
            DSAScheduledMultiRowLogitsKernel, IndexerOutputMode, IndexerScoreMode)
        from b12x.attention.dsa_indexer.contiguous_kernel import DSAContiguousLogitsPrefillKernelBlockScores
        from b12x.attention.dsa_indexer.tiled_topk import DSATiledTopkKernel

        self.rows, self.pages, self.mode = max_rows, max_pages, mode
        self.blocks, self.offsets, self.nbytes = _layout(max_rows, max_pages, mode)
        self.scorer = (DSAScheduledMultiRowLogitsKernel(4, 4,
            score_mode=IndexerScoreMode.MSA_BILINEAR, output_mode=IndexerOutputMode.PAGE_HEAD_MAX)
            if mode == "decode" else DSAContiguousLogitsPrefillKernelBlockScores(
                score_mode=IndexerScoreMode.MSA_BILINEAR, block_score_output=True))
        self.selector = DSATiledTopkKernel(is_tiled=False, topk=16, zero_row_start=False,
            smem_candidate_capacity=256, write_values=False, deterministic=True)

    @cute.jit
    def _ptr(self, scratch, offset: cutlass.Constexpr, dtype: cutlass.Constexpr):
        return cute.recast_ptr(scratch + offset, dtype=dtype)

    @cute.jit
    def _tensor(self, ptr, shape, strides):
        return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))

    @cute.kernel
    def init(self, scale, weights, schedule, controls, lengths, rows: Int32, width: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        idx = bx * 128 + tx
        if idx < rows * 4:
            weights[idx] = scale[idx] * Float32(128 ** -0.5)
        if const_expr(self.mode == "decode"):
            # The decode scorer's max_rows intervals split the rows' live pages
            # evenly in 4-page segments (the DSA paged-MQA schedule), instead of
            # one interval per row whose 4 CTAs walk every page of a long row.
            # Each page's score is computed whole by one CTA either way.
            if idx <= self.rows:
                total = Int32(0)
                for r in cutlass.range(rows):
                    live = cutlass.min(lengths[r], width)
                    total += ((live + Int32(63)) // Int32(64) + Int32(_SPLIT - 1)) // Int32(_SPLIT)
                share = total // Int32(self.rows)
                start = idx * share + cutlass.min(idx, total - share * Int32(self.rows))
                row, before = Int32(0), Int32(0)
                searching = Int32(1)
                while row < rows and searching != 0:
                    live = cutlass.min(lengths[row], width)
                    segments = ((live + Int32(63)) // Int32(64) + Int32(_SPLIT - 1)) // Int32(_SPLIT)
                    if before + segments <= start:
                        before += segments
                        row += 1
                    else:
                        searching = Int32(0)
                schedule[Int64(idx) * 2] = row
                schedule[Int64(idx) * 2 + 1] = start - before
        else:
            if idx <= rows:
                schedule[Int64(idx) * 2] = idx
                schedule[Int64(idx) * 2 + 1] = Int32(0)
        if idx == 0:
            controls[0] = width
            controls[1] = Int32(1)  # Scalar byte loads preserve the packed page stride.
            controls[2] = Int32(0)
            controls[3] = Int32(0)

    @cute.kernel
    def finish_scores(self, source, scores, starts, lengths, k_start, k_end,
                      rows: Int32, width: Int32):
        tx, _, _ = cute.arch.thread_idx()
        rh, _, _ = cute.arch.block_idx()
        row = rh % rows
        start = Int32(0)
        if const_expr(self.mode == "prefill"):
            start = k_start[row] // 128
        end = (k_end[row] + 127) // 128
        own = (k_end[row] - 1) // 128
        if tx == 0:
            starts[rh] = start
            lengths[rh] = end - start
        block = tx
        while block < self.blocks:
            value = Float32(-Float32.inf)
            if block >= start and block < end:
                if const_expr(self.mode == "decode"):
                    page = block * 2
                    if page < width:
                        value = source[Int64(rh) * self.pages + page]
                    if page + 1 < width:
                        other = source[Int64(rh) * self.pages + page + 1]
                        value = attention_ops.fmax(value, other)
                else:
                    value = scores[Int64(rh) * self.blocks + block]
                if block == own:
                    value = Float32.inf
            scores[Int64(rh) * self.blocks + block] = value
            block += 128

    @cute.kernel
    def publish(self, indices, starts, output, rows: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        idx = bx * 128 + tx
        if idx < rows * 4 * 16:
            rh, slot = idx // 16, idx % 16
            head, row = rh // rows, rh % rows
            value = indices[idx]
            rank = Int32(0)
            for other_slot in cutlass.range_constexpr(16):
                other = indices[Int64(rh) * 16 + other_slot]
                if (other >= 0 and (value < 0 or other < value)) or (other == value and other_slot < slot):
                    rank += 1
            if value >= 0:
                value -= starts[rh]
            output[(Int64(head) * self.rows + row) * 16 + rank] = value

    @cute.jit
    def select(self, source, scores, starts, lengths, k_start, k_end, indices, values,
               output, rows: Int32, width: Int32, stream: cuda.CUstream):
        self.finish_scores(source, scores, starts, lengths, k_start, k_end, rows, width).launch(
            grid=(rows * 4, 1, 1), block=(128, 1, 1), stream=stream)
        n = rows * 4
        score_t = self._tensor(scores, (n * self.blocks,), (1,))
        start_t = self._tensor(starts, (n,), (1,))
        len_t = self._tensor(lengths, (n,), (1,))
        index_t = self._tensor(indices, (n * 16,), (1,))
        value_t = self._tensor(values, (n * 16,), (1,))
        self.selector(score_t, start_t, len_t, value_t, index_t, value_t, index_t, start_t,
            Int32(0), n, Int32(self.blocks), Int32(1), Int32(0), Int32(1), Int32(1),
            Int32(16), Int32(0), Int32(self.blocks), Int32(0), Int32(1), Int32(1), Int32(0), stream)
        self.publish(indices, starts, output, rows).launch(
            grid=((rows * 4 * 16 + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)


class _Decode(_Index):
    @cute.jit
    def __call__(self, q_fp8: cute.Pointer, q_scale: cute.Pointer, index_k_cache: cute.Pointer,
                 page_table: cute.Pointer, cache_lengths: cute.Pointer, q2k_indices: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, table_width: Int32, table_stride: Int32,
                 pool_pages: Int64, stream: cuda.CUstream):
        weights = self._ptr(scratch, self.offsets["weights"], Float32)
        schedule = self._ptr(scratch, self.offsets["schedule"], Int32)
        controls = self._ptr(scratch, self.offsets["controls"], Int32)
        scores = self._ptr(scratch, self.offsets["scores"], Float32)
        pages = self._ptr(scratch, self.offsets["pages"], Float32)
        # Threads for every row's head weights and every schedule boundary.
        self.init(q_scale, weights, schedule, controls, cache_lengths, rows, table_width * 64).launch(
            grid=((rows * 4 + 127) // 128 + (self.rows + 128) // 128, 1, 1), block=(128, 1, 1), stream=stream)
        # The scoring kernel visits only live pages; initialize the padded page scores.
        self.clear_pages(pages, rows).launch(grid=((rows * 4 * self.pages + 127) // 128, 1, 1),
                                           block=(128, 1, 1), stream=stream)
        cache_bytes = cute.recast_ptr(index_k_cache, dtype=cutlass.Uint8)
        self.scorer(
            self._tensor(cute.recast_ptr(q_fp8, dtype=cutlass.Uint8), (rows, 4, 128), (512, 128, 1)),
            self._tensor(weights, (rows, 4), (4, 1)),
            self._tensor(cache_bytes, (Int32(pool_pages), 64, 128), (8448, 128, 1)),
            self._tensor(cute.recast_ptr(controls + 2, dtype=Int64), (1,), (1,)),
            self._tensor(controls + 1, (1,), (1,)),
            self._tensor(cute.recast_ptr(cache_bytes + 8192, dtype=Float32),
                         (Int32(pool_pages), 64), (2112, 1)),
            self._tensor(page_table, (rows, table_width), (Int64(table_stride), 1)),
            self._tensor(cache_lengths, (rows,), (1,)),
            self._tensor(schedule, (self.rows + 1, 2), (2, 1)),
            self._tensor(controls, (1,), (1,)),
            self._tensor(pages, (4, rows, self.pages), (Int64(rows) * self.pages, self.pages, 1)), stream)
        self.select(pages, scores, self._ptr(scratch, self.offsets["starts"], Int32),
            self._ptr(scratch, self.offsets["lengths"], Int32), cache_lengths, cache_lengths,
            self._ptr(scratch, self.offsets["indices"], Int32),
            self._ptr(scratch, self.offsets["values"], Float32), q2k_indices, rows, table_width, stream)

    @cute.kernel
    def clear_pages(self, pages, rows: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        idx = Int64(bx) * 128 + tx
        if idx < Int64(rows) * 4 * self.pages:
            pages[idx] = Float32(-Float32.inf)


class _Prefill(_Index):
    @cute.jit
    def __call__(self, q_fp8: cute.Pointer, q_scale: cute.Pointer, k_fp8: cute.Pointer,
                 k_scale: cute.Pointer, k_start: cute.Pointer, k_end: cute.Pointer,
                 q2k_indices: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, k_rows: Int32, stream: cuda.CUstream):
        weights = self._ptr(scratch, self.offsets["weights"], Float32)
        scores = self._ptr(scratch, self.offsets["scores"], Float32)
        controls = self._ptr(scratch, self.offsets["controls"], Int32)
        self.init(q_scale, weights, self._ptr(scratch, self.offsets["schedule"], Int32),
                  controls, k_end, rows, k_rows).launch(
            grid=((rows * 4 + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)
        padded_k_rows = (k_rows + 255) // 256 * 256
        dummy = self._tensor(cute.recast_ptr(controls, dtype=Float32), (1, 1), (1, 1))
        self.scorer(
            self._tensor(cute.recast_ptr(q_fp8, dtype=cutlass.Uint32), (rows, 4, 32), (128, 32, 1)),
            self._tensor(weights, (rows, 4), (4, 1)),
            self._tensor(cute.recast_ptr(k_fp8, dtype=cutlass.Uint8), (padded_k_rows, 128), (128, 1)),
            self._tensor(cute.recast_ptr(controls + 2, dtype=Int64), (1,), (1,)),
            self._tensor(k_scale, (padded_k_rows,), (1,)),
            self._tensor(k_start, (rows,), (1,)), self._tensor(k_end, (rows,), (1,)),
            dummy, dummy,
            self._tensor(scores, (4, rows, self.blocks), (Int64(rows) * self.blocks, self.blocks, 1)),
            rows, k_rows, Int32(self.blocks), Int32(0), (k_rows + 255) // 256, Int32(0), stream)
        self.select(scores, scores, self._ptr(scratch, self.offsets["starts"], Int32),
            self._ptr(scratch, self.offsets["lengths"], Int32), k_start, k_end,
            self._ptr(scratch, self.offsets["indices"], Int32),
            self._ptr(scratch, self.offsets["values"], Float32), q2k_indices, rows, k_rows, stream)


def compile_minimax_index_aot(*, max_rows, max_pages, mode="decode"):
    _layout(int(max_rows), int(max_pages), mode)
    launch = (_Decode if mode == "decode" else _Prefill)(int(max_rows), int(max_pages), mode)
    operands = [Operand("q_fp8", torch.float8_e4m3fn, "[rows,4,128]"),
                Operand("q_scale", torch.float32, "[rows,4]")]
    if mode == "decode":
        operands += [Operand("index_k_cache", torch.uint8, "[pool_pages,8448]"),
            Operand("page_table", torch.int32, "[rows,table_stride]", align=4),
            Operand("cache_lengths", torch.int32, "[rows]", align=4)]
        scalars = (Scalar("rows"), Scalar("table_width"), Scalar("table_stride"), Scalar("pool_pages", "int64", "must fit signed int32 page extent"))
    else:
        operands += [Operand("k_fp8", torch.float8_e4m3fn, "[ceil(k_rows/256)*256,128]"),
            Operand("k_scale", torch.float32, "[ceil(k_rows/256)*256]"),
            Operand("k_start", torch.int32, "[rows] request base, multiple of 128", align=4),
            Operand("k_end", torch.int32, "[rows] exclusive causal end", align=4)]
        scalars = (Scalar("rows"), Scalar("k_rows"))
    operands += [Operand("q2k_indices", torch.int32, f"[4,{max_rows},16]", "out", align=4),
                 Operand("scratch", torch.uint8, f"[{launch.nbytes}]", "scratch")]
    return compile_program(launch, name=f"minimax_index_{mode}", operands=tuple(operands),
        scalars=scalars, key=(int(max_rows), int(max_pages), mode),
        geometry=dict(max_rows=int(max_rows), max_pages=int(max_pages), heads=4,
                      head_dim=128, block_tokens=128, topk=16, mode=mode),
        scratch={"scratch": lambda rows: launch.nbytes}, doc=__doc__)
