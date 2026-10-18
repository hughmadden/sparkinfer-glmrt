"""Graph-stable metadata for weighted regions and causal sliding windows.

CuTe row-topk finds the score threshold. Existing stable-tie metadata rescans
scores to pick lower logical ids, including negative scores and signed zeros.
Unlike QSA's nonnegative-score ordering, the final set stays in logical order.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x._lib.compile_plan import launch_triton
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
from b12x.attention.qsa._kernels import (
    _stable_topk_threshold_kernel, _count_stable_topk_candidates_kernel,
    _emit_stable_topk_kernel,
)


@triton.jit
def _visibility(positions, requests, lengths, GROUPS: tl.constexpr,
                CAPACITY: tl.constexpr):
    row = tl.program_id(0)
    query = row // GROUPS
    position = tl.load(positions + query)
    request = tl.load(requests + query)
    visible = tl.minimum(tl.maximum(position // 8, 0), CAPACITY)
    tl.store(lengths + row, tl.where(request >= 0, visible, 0).to(tl.int32))


@triton.jit
def _expand_regions(ids, positions, requests, selected,
                    GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    query = row // GROUPS
    position = tl.load(positions + query)
    request = tl.load(requests + query)
    region = tl.load(ids + row.to(tl.int64) * 512 + column // 8,
                     mask=column < 4096, other=-1).to(tl.int64)
    region = tl.where(column >= 4096, position // 8, region)
    token = region * 8 + column % 8
    valid = (request >= 0) & (region >= 0) & (token <= position)
    tl.store(selected + row.to(tl.int64) * 4104 + column,
             tl.where(valid, token, -1).to(tl.int32), mask=column < 4104)


@triton.jit
def _sliding_positions(positions, requests, selected, WINDOW: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    position = tl.load(positions + row)
    request = tl.load(requests + row)
    token = position - WINDOW + 1 + column
    valid = (request >= 0) & (token >= 0)
    tl.store(selected + row.to(tl.int64) * WINDOW + column,
             tl.where(valid, token, -1).to(tl.int32), mask=column < WINDOW)


@dataclass
class RegionSelectionWorkspace:
    """Caller-owned fixed-capacity buffers; allocate once before capture."""
    capacity: int
    lengths: torch.Tensor
    values: torch.Tensor
    ids: torch.Tensor
    thresholds: torch.Tensor
    greater_totals: torch.Tensor
    tie_counts: torch.Tensor
    greater_counts: torch.Tensor
    stable_values: torch.Tensor
    stable_ids: torch.Tensor

    @classmethod
    def allocate(cls, *, max_rows: int, groups: int, capacity: int, device):
        if min(max_rows, groups, capacity) <= 0:
            raise ValueError("selection capacities and groups must be positive")
        rows = max_rows * groups
        blocks = triton.cdiv(capacity, 512)
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        return cls(capacity, empty((rows,), torch.int32),
                   empty((rows, 512), torch.float32), empty((rows, 512), torch.int32),
                   empty((rows,), torch.float32), empty((rows,), torch.int32),
                   empty((rows, blocks), torch.int32), empty((rows, blocks), torch.int32),
                   empty((rows, 512), torch.float32), empty((rows, 512), torch.int32))


def select_regions(*, scores, positions, requests, selected, workspace):
    """Select 512 historical regions plus the causal current-region prefix.

``scores`` is F32 [rows,groups,capacity], ``selected`` I32 [rows,groups,4104].
Positions must fit I32 logical token ids; pool-address multiplication is I64.
The caller must provision summary capacity for every visible history region.
"""
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be F32 [rows,groups,capacity]")
    rows, groups, capacity = scores.shape
    count = rows * groups
    if capacity != workspace.capacity or count > workspace.lengths.numel():
        raise ValueError("selection exceeds planned workspace")
    if selected.shape != (rows, groups, 4104) or selected.dtype != torch.int32:
        raise ValueError("selection must be I32 [rows,groups,4104]")
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("positions must be I64 [rows]")
    if requests.shape != (rows,) or requests.dtype != torch.int32:
        raise ValueError("requests must be I32 [rows]")
    for tensor in (scores, positions, requests, selected, workspace.lengths):
        if tensor.device != scores.device or not tensor.is_contiguous():
            raise ValueError("selection buffers must be contiguous on one device")
    if rows == 0:
        return
    lengths = workspace.lengths[:count]
    values, ids = workspace.values[:count], workspace.ids[:count]
    stable_values, stable_ids = workspace.stable_values[:count], workspace.stable_ids[:count]
    thresholds, totals = workspace.thresholds[:count], workspace.greater_totals[:count]
    ties, greater = workspace.tie_counts[:count], workspace.greater_counts[:count]
    blocks = ties.shape[1]
    launch_triton(_visibility, (count,), positions, requests, lengths,
                  GROUPS=groups, CAPACITY=capacity)
    flat_scores = scores.view(count, capacity)
    run_row_topk(row_logits=flat_scores, lengths=lengths, topk=512,
                 output_values=values, output_indices=ids)
    launch_triton(_stable_topk_threshold_kernel, (count,), values, lengths,
                  thresholds, totals, stable_values, stable_ids,
                  GROUP_BUDGET=512, BLOCK_K=512, num_warps=8)
    launch_triton(_count_stable_topk_candidates_kernel, (count, blocks),
                  flat_scores, lengths, thresholds, ties, greater, capacity,
                  NUM_BLOCKS=blocks, BLOCK_C=512, num_warps=8)
    # At offset zero there is no carried selection; ids is an unread placeholder.
    launch_triton(_emit_stable_topk_kernel, (count, blocks), flat_scores, lengths,
                  ids, lengths, thresholds, totals, ties, greater, stable_values,
                  stable_ids, capacity, GROUP_OFFSET=0, GROUP_BUDGET=512,
                  NUM_BLOCKS=blocks, BLOCK_COUNTS=triton.next_power_of_2(blocks),
                  BLOCK_C=512, num_warps=8)
    launch_triton(_expand_regions, (count,), stable_ids, positions, requests, selected,
                  GROUPS=groups, BLOCK=8192)


def sliding_positions(*, positions, requests, selected):
    """Populate chronological logical positions; cache ownership stays separate."""
    rows = positions.numel()
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("positions must be I64 [rows]")
    if requests.shape != (rows,) or requests.dtype != torch.int32:
        raise ValueError("requests must be I32 [rows]")
    if selected.shape != (rows, 512) or selected.dtype != torch.int32:
        raise ValueError("sliding selection must be I32 [rows,512]")
    if any(t.device != positions.device or not t.is_contiguous() for t in (positions, requests, selected)):
        raise ValueError("sliding buffers must be contiguous on one device")
    if rows:
        launch_triton(_sliding_positions, (rows,), positions, requests, selected,
                      WINDOW=512, BLOCK=512)
