"""Shared-memory geometry for the standalone K3 dense MLA kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

K3_QK_DIM = 576
K3_VALUE_DIM = 512
HEADS_PER_TILE = 8
CANDIDATES_PER_CHUNK = 64
KV_STAGES = 2
MATH_WARPS_PER_QUERY = 4


@dataclass(frozen=True)
class SmemLayout:
    query_tile: int
    fp8: bool
    kv_stages: int
    heads_per_tile: int
    candidates_per_chunk: int
    math_warps_per_query: int
    record_bytes: int
    record_stride_bytes: int
    q_stage_bytes: int
    kv_stage_bytes: int
    reduce_group_bytes: int
    reduce_bytes: int
    weight_group_bytes: int
    weight_bytes: int
    weight_scale_group_bytes: int
    weight_scale_bytes: int
    total_bytes: int


def make_smem_layout(
    *,
    query_tile: int,
    fp8: bool,
    qk_dim: int = K3_QK_DIM,
    value_dim: int = K3_VALUE_DIM,
    heads_per_tile: int = HEADS_PER_TILE,
    candidates_per_chunk: int = CANDIDATES_PER_CHUNK,
    kv_stages: int | None = None,
    math_warps_per_query: int = MATH_WARPS_PER_QUERY,
) -> SmemLayout:
    """Describe one dense MLA specialization.

    ``query_tile`` independent query rows share a single KV stream; each row is
    served by ``math_warps_per_query`` warps.  ``heads_per_tile`` sets the MMA
    N width (and the PV M width), ``candidates_per_chunk`` the per-stage KV
    window, and ``kv_stages`` the producer ring depth.
    """
    query_tile = int(query_tile)
    heads_per_tile = int(heads_per_tile)
    candidates_per_chunk = int(candidates_per_chunk)
    math_warps_per_query = int(math_warps_per_query)
    if query_tile not in (1, 2, 3, 4):
        raise ValueError("dense MLA query_tile must be 1, 2, 3, or 4")
    if heads_per_tile not in (8, 16):
        raise ValueError("dense MLA heads_per_tile must be 8 or 16")
    if candidates_per_chunk not in (32, 64):
        # Each producer lane stages whole records, so the chunk must tile the
        # producer warp exactly.
        raise ValueError("dense MLA candidates_per_chunk must be 32 or 64")
    if math_warps_per_query not in (1, 2, 4):
        raise ValueError("dense MLA math_warps_per_query must be 1, 2 or 4")
    if candidates_per_chunk != 16 * math_warps_per_query:
        # Each math warp owns sixteen consecutive cache records.
        raise ValueError(
            "dense MLA candidates_per_chunk must equal 16 * math_warps_per_query"
        )
    value_dim = int(value_dim)
    if value_dim <= 0 or value_dim % (16 * math_warps_per_query):
        raise ValueError("dense MLA value_dim must be a multiple of 16 * mw")
    # The persistent value accumulators are the register budget: every lane
    # holds four accumulators per value tile and head tile.
    accumulators_per_lane = (
        4 * (value_dim // (16 * math_warps_per_query)) * (heads_per_tile // 8)
    )
    if accumulators_per_lane > 64:
        raise ValueError(
            "dense MLA tile needs too many value accumulators per lane"
        )

    element_bytes = 1 if fp8 else 2
    qk_dim = int(qk_dim)
    if qk_dim <= 0 or qk_dim % 32:
        raise ValueError("dense MLA qk_dim must be a positive multiple of 32")
    if kv_stages is None:
        # Two BF16 64-record stages cannot physically fit in the 101,376-byte
        # opt-in SMEM limit; callers targeting BF16 prefill pass kv_stages
        # explicitly together with a smaller KV window.
        kv_stages = KV_STAGES if fp8 and qk_dim <= K3_QK_DIM else 1
    kv_stages = int(kv_stages)
    if kv_stages not in (1, 2, 3):
        raise ValueError("dense MLA kv_stages must be 1, 2 or 3")
    record_bytes = qk_dim * element_bytes
    # The extra 16 bytes rotate successive rows across banks while making
    # every row a legal 16-byte cp.async destination.
    record_stride_bytes = record_bytes + 16

    q_stage_bytes = query_tile * heads_per_tile * record_stride_bytes
    kv_stage_bytes = kv_stages * candidates_per_chunk * record_stride_bytes

    # Per query: max and sum for each warp x head pair.
    reduce_group_bytes = 2 * math_warps_per_query * heads_per_tile * 4
    reduce_bytes = query_tile * reduce_group_bytes

    if fp8:
        # HIGH/LOW E4M3 probability rows, (chunk + 16 pad) bytes each.
        weight_group_bytes = 16 * (candidates_per_chunk + 16)
        # Eight scales plus eight cached reciprocals.
        weight_scale_group_bytes = 16 * 4
    else:
        # BF16 probability rows, padded by eight elements: a 16-byte row pad
        # rotates consecutive head rows across the bank groups, so the
        # A-operand ldmatrix of the value MMA reads each bank once instead of
        # colliding eight ways on an unpadded power-of-two stride.
        weight_group_bytes = 16 * (candidates_per_chunk + 8) * 2
        weight_scale_group_bytes = 4
    weight_bytes = query_tile * weight_group_bytes
    weight_scale_bytes = query_tile * weight_scale_group_bytes

    # Typed structs apply the declared alignments; instantiate below and use
    # its exact result as the launch resource contract.
    layout = SmemLayout(
        query_tile=query_tile,
        fp8=bool(fp8),
        kv_stages=kv_stages,
        heads_per_tile=heads_per_tile,
        candidates_per_chunk=candidates_per_chunk,
        math_warps_per_query=math_warps_per_query,
        record_bytes=record_bytes,
        record_stride_bytes=record_stride_bytes,
        q_stage_bytes=q_stage_bytes,
        kv_stage_bytes=kv_stage_bytes,
        reduce_group_bytes=reduce_group_bytes,
        reduce_bytes=reduce_bytes,
        weight_group_bytes=weight_group_bytes,
        weight_bytes=weight_bytes,
        weight_scale_group_bytes=weight_scale_group_bytes,
        weight_scale_bytes=weight_scale_bytes,
        total_bytes=0,
    )
    storage = get_shared_storage_cls(layout)
    return SmemLayout(
        **{
            **layout.__dict__,
            "total_bytes": int(storage.size_in_bytes()),
        }
    )


# Prefill tile geometry: (query_tile, heads_per_tile, candidates_per_chunk,
# math_warps_per_query) with a single KV stage.  Every query row of the tile
# shares one gathered KV window, so the cache is streamed once per tile rather
# than once per row.
PREFILL_TILE = (1, 16, 64, 4)
PREFILL_KV_STAGES = 1


def choose_layout(
    *,
    mode: str,
    fp8: bool,
    qk_dim: int = K3_QK_DIM,
    value_dim: int = K3_VALUE_DIM,
    single_request: bool = True,
    windowed: bool = False,
    query_rows: int = 1,
) -> SmemLayout:
    """Pick the SMEM specialization for one serving role.

    Decode and verify keep one query row per block: a single row already fills
    the machine and its gather is latency bound, not bandwidth bound.  A
    multi-row prefill tile amortizes the gathered KV window across query rows.
    """
    if mode == "extend" and single_request and not windowed:
        if not fp8:
            query_tile, heads, candidates, mw = PREFILL_TILE
            return make_smem_layout(
                query_tile=query_tile,
                fp8=False,
                qk_dim=qk_dim,
                value_dim=value_dim,
                heads_per_tile=heads,
                candidates_per_chunk=candidates,
                kv_stages=PREFILL_KV_STAGES,
                math_warps_per_query=mw,
            )
        return make_smem_layout(
            query_tile=4 if int(query_rows) >= 3 else 2,
            fp8=True,
            qk_dim=qk_dim,
            value_dim=value_dim,
        )
    return make_smem_layout(query_tile=1, fp8=fp8, qk_dim=qk_dim, value_dim=value_dim)


def get_shared_storage_cls(layout: SmemLayout):
    """Return the typed storage for one dtype/tile specialization."""

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.q_stage_bytes],
            128,
        ],
        "kv": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.kv_stage_bytes],
            128,
        ],
        # full[kv_stages], empty[kv_stages].
        "mbar": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint64, 2 * layout.kv_stages],
            16,
        ],
        "reduce": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, layout.reduce_bytes // 4],
            16,
        ],
        "weight_scale": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8,
                max(layout.weight_scale_bytes, 1),
            ],
            16,
        ],
        "weight": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.weight_bytes],
            128,
        ],
    }
    return cute.struct(SharedStorage)


__all__ = [
    "CANDIDATES_PER_CHUNK",
    "SmemLayout",
    "HEADS_PER_TILE",
    "K3_QK_DIM",
    "K3_VALUE_DIM",
    "KV_STAGES",
    "MATH_WARPS_PER_QUERY",
    "PREFILL_KV_STAGES",
    "PREFILL_TILE",
    "get_shared_storage_cls",
    "choose_layout",
    "make_smem_layout",
]
