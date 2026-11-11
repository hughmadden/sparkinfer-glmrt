"""Stable low-ID selection of MSA block metadata into caller-owned buffers."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=[
    "rows", "blocks", "topk", "score_head_stride", "score_row_stride",
    "value_head_stride", "value_row_stride", "index_head_stride", "index_row_stride",
])
def _stable_msa_topk(
    scores, values, indices, rows, blocks, topk,
    score_head_stride, score_row_stride,
    value_head_stride, value_row_stride,
    index_head_stride, index_row_stride,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    score_row = scores + head * score_head_stride + row * score_row_stride
    value_row = values + head * value_head_stride + row * value_row_stride
    index_row = indices + head * index_head_stride + row * index_row_stride
    offsets = tl.arange(0, BLOCK)
    for slot in range(topk):
        best_value = float("-inf")
        best_id = 2147483647
        for start in range(tl.cdiv(blocks, BLOCK)):
            ids = start * BLOCK + offsets
            tile = tl.load(score_row + ids.to(tl.int64), ids < blocks, other=float("-inf"))
            tile_value = tl.max(tile, axis=0)
            tile_id = tl.min(tl.where((ids < blocks) & (tile == tile_value), ids, 2147483647), axis=0)
            better = (tile_value > best_value) | ((tile_value == best_value) & (tile_id < best_id))
            best_id = tl.where(better, tile_id, best_id)
            best_value = tl.maximum(best_value, tile_value)
        tl.store(value_row + slot, best_value)
        tl.store(index_row + slot, best_id.to(tl.int64))
        # One CTA owns a score row. Retire its winner before the next scan.
        tl.store(score_row + best_id.to(tl.int64), float("-inf"), best_id < blocks)
        tl.debug_barrier()


def stable_msa_topk(
    scores: torch.Tensor, *, topk: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume the mutable score scratch; finite ties prefer the lowest ID.

    The scan tile is fixed. Live head/row/block/top-k counts and strides stay
    runtime arguments, so replay across live shapes does not compile kernels.
    """
    if scores.device.type != "cuda":
        values, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
        values, indices = values[..., :topk], indices[..., :topk]
        if out is not None:
            out[0].copy_(values)
            out[1].copy_(indices)
            return out
        return values, indices
    if out is None:
        shape = (*scores.shape[:2], topk)
        out = (
            torch.empty(shape, dtype=torch.float32, device=scores.device),
            torch.empty(shape, dtype=torch.int64, device=scores.device),
        )
    _stable_msa_topk[(scores.shape[0], scores.shape[1])](
        scores, out[0], out[1], scores.shape[1], scores.shape[2], topk,
        scores.stride(0), scores.stride(1),
        out[0].stride(0), out[0].stride(1),
        out[1].stride(0), out[1].stride(1),
        BLOCK=256, num_warps=4,
    )
    return out
