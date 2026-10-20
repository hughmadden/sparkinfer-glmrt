"""CuTe causal metadata with the shared deterministic, lower-logical-id top-k."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
import torch

from ._common import Operand, Scalar, compile_program


class _Selection:
    """Prepare the visible causal prefix only, over a bounded persistent grid.

    The shared radix top-k bounds every read with ``lengths``: it never reads
    ``normalized[visible:]``.  The frozen serving prepare normalized the whole
    capacity extent anyway -- 131,072 entries per row for a checkpoint whose
    runtime extent is 1M tokens -- so a short prompt paid ~128x dead traffic.
    Normalizing ``[0, visible)`` leaves the top-k bit-identical (the masked
    tail is unreachable and its bytes are ignored), and the persistent y grid
    keeps a short prefix from dispatching up to 512 empty tiles per row while
    a full extent still spreads across the SMs.

    ``TARGET_BLOCKS`` sizes the launch: ``ceil(TARGET_BLOCKS / count)`` parts
    per row, capped by the capacity tiles.  A block whose columns start past
    the visible prefix loads one scalar and exits, so the y extent has to stay
    small for short prompts; it still has to expose enough blocks to fill the
    card when every row is at full extent.
    """

    # Total blocks the prepare grid aims for.  At the captured serving shape
    # (64 rows x 4 groups, 8K prompt) 1,024 blocks cover the ~263K visible
    # entries once each; the full 131,072-entry extent still gets 1,024 blocks
    # of 256 threads over count=1,024 rows, enough to saturate memory.
    TARGET_BLOCKS = 1024

    def __init__(self, groups, capacity):
        from b12x.attention.dsa_indexer.tiled_topk import _build_row_topk_kernel, _resolve_smem_candidate_capacity
        self.groups, self.capacity = groups, capacity
        self.tiles = (capacity + 255) // 256
        self.topk = _build_row_topk_kernel(512, False,
            _resolve_smem_candidate_capacity(topk=512), True, True)

    @cute.jit
    def __call__(self, scores: cute.Pointer, positions: cute.Pointer,
                 requests: cute.Pointer, selected: cute.Pointer,
                 normalized: cute.Pointer, lengths: cute.Pointer,
                 values: cute.Pointer, ids: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        count = rows * self.groups
        if rows > Int32(0):
            # Runtime row count, runtime grid: ceil(TARGET_BLOCKS/count) parts
            # per row, never more capacity tiles than the extent has.
            parts = min((Int32(self.TARGET_BLOCKS) + count - Int32(1)) // count,
                        Int32(self.tiles))
            self.prepare(scores, positions, requests, normalized, lengths).launch(
                grid=(count, parts, 1), block=(256, 1, 1), stream=stream)
            score_tensor = cute.make_tensor(normalized, cute.make_layout((Int64(count) * self.capacity,)))
            length_tensor = cute.make_tensor(lengths, cute.make_layout((count,)))
            value_tensor = cute.make_tensor(values, cute.make_layout((Int64(count) * 512,)))
            id_tensor = cute.make_tensor(ids, cute.make_layout((Int64(count) * 512,)))
            self.topk(score_tensor, length_tensor, length_tensor, value_tensor, id_tensor,
                      value_tensor, id_tensor, length_tensor, Int32(0), count,
                      Int32(self.capacity), Int32(0), Int32(0), Int32(1), Int32(1),
                      Int32(512), Int32(0), Int32(self.capacity), Int32(0), Int32(1),
                      Int32(1), Int32(0), stream)
            self.expand(ids, positions, requests, selected).launch(
                grid=(count, 17, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def prepare(self, scores: cute.Pointer, positions: cute.Pointer,
                requests: cute.Pointer, normalized: cute.Pointer, lengths: cute.Pointer):
        row, part, _ = cute.arch.block_idx()
        _, parts, _ = cute.arch.grid_dim()
        thread, _, _ = cute.arch.thread_idx()
        query = row // self.groups
        visible = min(max(positions[query].to(Int64) // 8, Int64(0)), Int64(self.capacity))
        if requests[query] < 0:
            visible = Int64(0)
        if (part == 0) & (thread == 0):
            lengths[row] = Int32(visible)
        base = Int64(row) * self.capacity
        column = part * 256 + thread
        stride = parts * 256
        visible32 = Int32(visible)
        while column < visible32:
            offset = base + Int64(column)
            score = Float32(scores[offset])
            # IEEE +/-0 compare equal in the oracle. Radix ordering must see one
            # representation so its exact lower-id tie order remains identical.
            if score == Float32(0):
                score = Float32(0)
            normalized[offset] = score
            column = column + stride

    @cute.kernel
    def expand(self, ids: cute.Pointer, positions: cute.Pointer,
               requests: cute.Pointer, selected: cute.Pointer):
        row, tile, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        column = tile * 256 + thread
        query = row // self.groups
        if column < 4104:
            position = positions[query].to(Int64)
            region = position // 8
            if column < 4096:
                region = ids[Int64(row) * 512 + column // 8].to(Int64)
            token = region * 8 + column % 8
            result = Int32(-1)
            if (requests[query] >= 0) & (region >= 0) & (token <= position):
                result = Int32(token)
            selected[Int64(row) * 4104 + column] = result


class _Sliding:
    @cute.jit
    def __call__(self, positions: cute.Pointer, requests: cute.Pointer,
                 selected: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(positions, requests, selected).launch(
            grid=(rows, 2, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, positions: cute.Pointer, requests: cute.Pointer, selected: cute.Pointer):
        row, tile, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        column = tile * 256 + thread
        token = positions[row].to(Int64) - 511 + column
        result = Int32(-1)
        if (requests[row] >= 0) & (token >= 0):
            result = Int32(token)
        selected[Int64(row) * 512 + column] = result


def compile_region_selection_aot(*, groups, capacity):
    groups, capacity = int(groups), int(capacity)
    if min(groups, capacity) <= 0:
        raise ValueError("positive provider groups and summary capacity required")
    program = compile_program(_Selection(groups, capacity), name="region_selection", operands=(
        Operand("scores", torch.float32, f"[rows,{groups},{capacity}]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("requests", torch.int32, "[rows]", align=4),
        Operand("selected", torch.int32, f"[rows,{groups},4104]", "out", align=4),
        Operand("normalized", torch.float32, f"[rows,{groups},{capacity}]", "scratch"),
        Operand("lengths", torch.int32, f"[rows,{groups}]", "scratch", align=4),
        Operand("values", torch.float32, f"[rows,{groups},512]", "scratch"),
        Operand("ids", torch.int32, f"[rows,{groups},512]", "scratch", align=4)),
        scalars=(Scalar("rows"),), key=(groups, capacity),
        geometry=dict(groups=groups, capacity=capacity, topk=512, region_tokens=8),
        scratch={"normalized": lambda rows: rows * groups * capacity * 4,
                 "lengths": lambda rows: rows * groups * 4,
                 "values": lambda rows: rows * groups * 512 * 4,
                 "ids": lambda rows: rows * groups * 512 * 4})
    return program


def compile_sliding_positions_aot():
    return compile_program(_Sliding(), name="sliding_positions", operands=(
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("requests", torch.int32, "[rows]", align=4),
        Operand("selected", torch.int32, "[rows,512]", "out", align=4)),
        scalars=(Scalar("rows"),), key=(512,), geometry=dict(window=512))
