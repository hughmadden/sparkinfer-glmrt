"""Native AOT DeepSeek V4 KV compressor (C4 with index compressor, C128).

Three compile functions mirror ``b12x.attention.dsv4_compressor``:

* ``compile_dsv4_compressor_decode_aot``  = ``run_decode`` (one row per
  sequence, ``rows_are_sequence_unique``);
* ``compile_dsv4_compressor_prefill_aot`` = ``run_prefill``
  (``initial_prefill``: every sequence starts at logical position 0);
* ``compile_dsv4_compressor_continuation_aot`` = ``run_continuation``
  (one ordered chunk per sequence continuing its persistent state).

Each program computes the joint BF16 projection ``hidden @ joint^T`` (FP32
accumulate, BF16 out, in place of ``torch.mm``): a bandwidth-bound skinny
GEMV while the live ``rows`` <= 48 (C4, W=2560) / 128 (C128), else the CuTe
warp-MMA GEMM; the branch is on the ``rows`` scalar inside the program. It pools/normalizes/RoPEs
each completed group into the FP8 compressed cache (C4 also pools the
128-dim index key: Hadamard, E2M1 QAT, FP8 + FP32 row scale into the index
cache) and maintains the FP32 rolling state. Only the fp8 cache format is
exported.

Common pointers (``R`` = ratio, ``W`` = joint width = 2560 for C4, 1024 for
C128; ``pw`` = 1024 (C4) / 512 (C128); ``SR`` = state rows = 16 (C4) / 256
(C128); ``S`` = state sequences; page bytes 37440 (C4, 64 rows) / 1728
(C128, 2 rows))::

    hidden            bf16 [rows,H]      in
    cos_sin           f32  [P,64]        in   compressed RoPE table (cos | sin)
    joint_projection  bf16 [W,H]         in   pack_weights(...).joint_projection
    main_ape          f32  [R,pw]        in
    main_norm         bf16 [512]         in
    compressed_cache  u8   [pages,page]  inout
    main_kv_state     f32  [S,SR,pw]     inout
    main_score_state  f32  [S,SR,pw]     inout
    (C4 only)
    index_ape         f32  [4,256]       in
    index_norm        bf16 [128]         in
    index_cache       u8   [pages,8448]  inout (same page ids as compressed_cache)
    index_kv_state    f32  [S,16,256]    inout
    index_score_state f32  [S,16,256]    inout
    scratch           u8   rows * W * 2 bytes (the BF16 joint projection)

Mode metadata (all int32, contiguous; ``G`` group capacity, ``C`` sequence
capacity)::

    decode:        positions [rows], sequence_ids [rows], compressed_slots [rows]
                   (slot of the group a row completes; read only when
                   position % R == R-1)
    prefill:       active_groups [1], group_source_starts [G],
                   group_rope_positions [G], compressed_slots [G],
                   active_sequences [1], sequence_offsets [C+1],
                   state_sequence_ids [C]
    continuation:  active_groups [1], group_sequence_slots [G],
                   group_source_positions [G], group_rope_positions [G],
                   compressed_slots [G], active_sequences [1],
                   sequence_offsets [C+1], sequence_start_positions [C],
                   state_sequence_ids [C]

Scalars: ``rows`` (live hidden rows), and for prefill/continuation
``groups`` (= G, grid bound, >= 1) and ``sequences`` (= C, grid bound, >= 1);
the kernels read the live counts from ``active_groups`` /
``active_sequences``. Compressed/index slot offsets are 64-bit.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from b12x.attention.dsv4_compressor._cute import (
    DSV4CompressorFinalize,
    DSV4CompressorPool,
    main_page_bytes,
)
from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

from ._common import FLASH, DSV4Geometry, Operand, Scalar, compile_program

__all__ = [
    "compile_dsv4_compressor_continuation_aot",
    "compile_dsv4_compressor_decode_aot",
    "compile_dsv4_compressor_prefill_aot",
    "compressor_scratch_bytes",
    "joint_width",
]


def joint_width(ratio: int) -> int:
    return 2 * (1024 + 256) if int(ratio) == 4 else 1024


def compressor_scratch_bytes(ratio: int, rows: int) -> int:
    """BF16 joint projection [rows, W]."""
    return int(rows) * joint_width(ratio) * 2


class _Base:
    def __init__(self, geometry: DSV4Geometry, ratio: int, mode: str):
        self.ratio = int(ratio)
        if self.ratio not in (4, 128):
            raise ValueError("DSV4 compressor ratio must be 4 or 128")
        self.mode = mode
        self.h = geometry.hidden
        self.w = joint_width(self.ratio)
        self.gemm = RoutedBf16Projection(self.w, self.h)
        eps = geometry.norm_eps
        self.main = DSV4CompressorPool(ratio=self.ratio, index=False, mode=mode, eps=eps)
        self.index = (DSV4CompressorPool(ratio=4, index=True, mode=mode, eps=eps)
                      if self.ratio == 4 else None)
        if mode != "decode":
            self.main_final = DSV4CompressorFinalize(ratio=self.ratio, index=False, mode=mode)
            self.index_final = (DSV4CompressorFinalize(ratio=4, index=True, mode=mode)
                                if self.ratio == 4 else None)

    @cute.jit
    def _project(self, hidden: cute.Pointer, joint: cute.Pointer, projection: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.gemm(hidden, joint, projection, rows, stream)


def _bf16(pointer: cute.Pointer):
    import cutlass

    return cute.make_ptr(cutlass.BFloat16, Int64(pointer.toint()), cute.AddressSpace.gmem,
                         assumed_align=16)


class _DecodeC4(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, positions: cute.Pointer, sequence_ids: cute.Pointer,
                 compressed_slots: cute.Pointer, cos_sin: cute.Pointer,
                 joint_projection: cute.Pointer, main_ape: cute.Pointer, main_norm: cute.Pointer,
                 compressed_cache: cute.Pointer, main_kv_state: cute.Pointer,
                 main_score_state: cute.Pointer, index_ape: cute.Pointer,
                 index_norm: cute.Pointer, index_cache: cute.Pointer,
                 index_kv_state: cute.Pointer, index_score_state: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        m = positions
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, positions, sequence_ids, compressed_slots, m, m, m, m, m,
                  rows, stream)
        self.index(projection, cos_sin, index_ape, index_norm, index_cache, index_kv_state,
                   index_score_state, positions, sequence_ids, compressed_slots, m, m, m, m, m,
                   rows, stream)


class _DecodeC128(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, positions: cute.Pointer, sequence_ids: cute.Pointer,
                 compressed_slots: cute.Pointer, cos_sin: cute.Pointer,
                 joint_projection: cute.Pointer, main_ape: cute.Pointer, main_norm: cute.Pointer,
                 compressed_cache: cute.Pointer, main_kv_state: cute.Pointer,
                 main_score_state: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        m = positions
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, positions, sequence_ids, compressed_slots, m, m, m, m, m,
                  rows, stream)


class _PrefillC4(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, active_groups: cute.Pointer,
                 group_source_starts: cute.Pointer, group_rope_positions: cute.Pointer,
                 compressed_slots: cute.Pointer, active_sequences: cute.Pointer,
                 sequence_offsets: cute.Pointer, state_sequence_ids: cute.Pointer,
                 cos_sin: cute.Pointer, joint_projection: cute.Pointer, main_ape: cute.Pointer,
                 main_norm: cute.Pointer, compressed_cache: cute.Pointer,
                 main_kv_state: cute.Pointer, main_score_state: cute.Pointer,
                 index_ape: cute.Pointer, index_norm: cute.Pointer, index_cache: cute.Pointer,
                 index_kv_state: cute.Pointer, index_score_state: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, groups: Int32, sequences: Int32,
                 stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        g = active_groups
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, active_groups, group_source_starts, group_rope_positions,
                  compressed_slots, g, g, g, g, groups, stream)
        self.main_final(projection, main_ape, main_kv_state, main_score_state, active_sequences,
                        sequence_offsets, state_sequence_ids, sequence_offsets, sequences, stream)
        self.index(projection, cos_sin, index_ape, index_norm, index_cache, index_kv_state,
                   index_score_state, active_groups, group_source_starts, group_rope_positions,
                   compressed_slots, g, g, g, g, groups, stream)
        self.index_final(projection, index_ape, index_kv_state, index_score_state,
                         active_sequences, sequence_offsets, state_sequence_ids,
                         sequence_offsets, sequences, stream)


class _PrefillC128(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, active_groups: cute.Pointer,
                 group_source_starts: cute.Pointer, group_rope_positions: cute.Pointer,
                 compressed_slots: cute.Pointer, active_sequences: cute.Pointer,
                 sequence_offsets: cute.Pointer, state_sequence_ids: cute.Pointer,
                 cos_sin: cute.Pointer, joint_projection: cute.Pointer, main_ape: cute.Pointer,
                 main_norm: cute.Pointer, compressed_cache: cute.Pointer,
                 main_kv_state: cute.Pointer, main_score_state: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, groups: Int32, sequences: Int32,
                 stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        g = active_groups
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, active_groups, group_source_starts, group_rope_positions,
                  compressed_slots, g, g, g, g, groups, stream)
        self.main_final(projection, main_ape, main_kv_state, main_score_state, active_sequences,
                        sequence_offsets, state_sequence_ids, sequence_offsets, sequences, stream)


class _ContinuationC4(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, active_groups: cute.Pointer,
                 group_sequence_slots: cute.Pointer, group_source_positions: cute.Pointer,
                 group_rope_positions: cute.Pointer, compressed_slots: cute.Pointer,
                 active_sequences: cute.Pointer, sequence_offsets: cute.Pointer,
                 sequence_start_positions: cute.Pointer, state_sequence_ids: cute.Pointer,
                 cos_sin: cute.Pointer, joint_projection: cute.Pointer, main_ape: cute.Pointer,
                 main_norm: cute.Pointer, compressed_cache: cute.Pointer,
                 main_kv_state: cute.Pointer, main_score_state: cute.Pointer,
                 index_ape: cute.Pointer, index_norm: cute.Pointer, index_cache: cute.Pointer,
                 index_kv_state: cute.Pointer, index_score_state: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, groups: Int32, sequences: Int32,
                 stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        meta = (active_groups, group_sequence_slots, group_source_positions,
                group_rope_positions, compressed_slots, sequence_offsets,
                sequence_start_positions, state_sequence_ids)
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, *meta, groups, stream)
        self.index(projection, cos_sin, index_ape, index_norm, index_cache, index_kv_state,
                   index_score_state, *meta, groups, stream)
        self.main_final(projection, main_ape, main_kv_state, main_score_state, active_sequences,
                        sequence_offsets, state_sequence_ids, sequence_start_positions,
                        sequences, stream)
        self.index_final(projection, index_ape, index_kv_state, index_score_state,
                         active_sequences, sequence_offsets, state_sequence_ids,
                         sequence_start_positions, sequences, stream)


class _ContinuationC128(_Base):
    @cute.jit
    def __call__(self, hidden: cute.Pointer, active_groups: cute.Pointer,
                 group_sequence_slots: cute.Pointer, group_source_positions: cute.Pointer,
                 group_rope_positions: cute.Pointer, compressed_slots: cute.Pointer,
                 active_sequences: cute.Pointer, sequence_offsets: cute.Pointer,
                 sequence_start_positions: cute.Pointer, state_sequence_ids: cute.Pointer,
                 cos_sin: cute.Pointer, joint_projection: cute.Pointer, main_ape: cute.Pointer,
                 main_norm: cute.Pointer, compressed_cache: cute.Pointer,
                 main_kv_state: cute.Pointer, main_score_state: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, groups: Int32, sequences: Int32,
                 stream: cuda.CUstream):
        projection = _bf16(scratch)
        self._project(hidden, joint_projection, projection, rows, stream)
        meta = (active_groups, group_sequence_slots, group_source_positions,
                group_rope_positions, compressed_slots, sequence_offsets,
                sequence_start_positions, state_sequence_ids)
        self.main(projection, cos_sin, main_ape, main_norm, compressed_cache, main_kv_state,
                  main_score_state, *meta, groups, stream)
        self.main_final(projection, main_ape, main_kv_state, main_score_state, active_sequences,
                        sequence_offsets, state_sequence_ids, sequence_start_positions,
                        sequences, stream)


def _i32(name: str, shape: str) -> Operand:
    return Operand(name, torch.int32, shape, align=4)


def _weights_and_state(geometry: DSV4Geometry, ratio: int) -> tuple[Operand, ...]:
    h = geometry.hidden
    pw = 1024 if ratio == 4 else 512
    sr = 16 if ratio == 4 else 256
    ops = [
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("joint_projection", torch.bfloat16, f"[{joint_width(ratio)},{h}]"),
        Operand("main_ape", torch.float32, f"[{ratio},{pw}]", align=4),
        Operand("main_norm", torch.bfloat16, "[512]"),
        Operand("compressed_cache", torch.uint8, f"[pages,{main_page_bytes(ratio)}]", "inout"),
        Operand("main_kv_state", torch.float32, f"[S,{sr},{pw}]", "inout", align=4),
        Operand("main_score_state", torch.float32, f"[S,{sr},{pw}]", "inout", align=4),
    ]
    if ratio == 4:
        ops += [
            Operand("index_ape", torch.float32, "[4,256]", align=4),
            Operand("index_norm", torch.bfloat16, "[128]"),
            Operand("index_cache", torch.uint8, "[pages,8448]", "inout"),
            Operand("index_kv_state", torch.float32, "[S,16,256]", "inout", align=4),
            Operand("index_score_state", torch.float32, "[S,16,256]", "inout", align=4),
        ]
    ops.append(Operand("scratch", torch.uint8, f"[rows*{joint_width(ratio) * 2}]", "scratch"))
    return tuple(ops)


def _compile(launch, *, geometry, ratio, mode, metadata, scalars):
    return compile_program(
        launch, name=f"dsv4_compressor_{mode}_c{ratio}",
        operands=(Operand("hidden", torch.bfloat16, f"[rows,{geometry.hidden}]"),) + metadata
        + _weights_and_state(geometry, ratio),
        scalars=scalars, key=(geometry.hidden, ratio, mode, geometry.norm_eps, launch.gemm.key()),
        geometry={"hidden": geometry.hidden, "ratio": ratio, "mode": mode,
                  "joint_width": joint_width(ratio), "cache_format": "fp8"},
        scratch={"scratch": lambda rows: compressor_scratch_bytes(ratio, rows)},
        doc=__doc__,
    )


def compile_dsv4_compressor_decode_aot(geometry: DSV4Geometry = FLASH, *, ratio: int):
    """One sequence-unique row per launch row; see module docstring."""
    launch = (_DecodeC4 if ratio == 4 else _DecodeC128)(geometry, ratio, "decode")
    metadata = (_i32("positions", "[rows]"), _i32("sequence_ids", "[rows]"),
                _i32("compressed_slots", "[rows]"))
    return _compile(launch, geometry=geometry, ratio=ratio, mode="decode", metadata=metadata,
                    scalars=(Scalar("rows"),))


def compile_dsv4_compressor_prefill_aot(geometry: DSV4Geometry = FLASH, *, ratio: int):
    """Initial prefill (every sequence from position 0); see module docstring."""
    launch = (_PrefillC4 if ratio == 4 else _PrefillC128)(geometry, ratio, "prefill")
    metadata = (_i32("active_groups", "[1]"), _i32("group_source_starts", "[G]"),
                _i32("group_rope_positions", "[G]"), _i32("compressed_slots", "[G]"),
                _i32("active_sequences", "[1]"), _i32("sequence_offsets", "[C+1]"),
                _i32("state_sequence_ids", "[C]"))
    return _compile(launch, geometry=geometry, ratio=ratio, mode="prefill", metadata=metadata,
                    scalars=(Scalar("rows"), Scalar("groups"), Scalar("sequences")))


def compile_dsv4_compressor_continuation_aot(geometry: DSV4Geometry = FLASH, *, ratio: int):
    """Ordered continuation chunk per sequence; see module docstring."""
    launch = (_ContinuationC4 if ratio == 4 else _ContinuationC128)(geometry, ratio, "continuation")
    metadata = (_i32("active_groups", "[1]"), _i32("group_sequence_slots", "[G]"),
                _i32("group_source_positions", "[G]"), _i32("group_rope_positions", "[G]"),
                _i32("compressed_slots", "[G]"), _i32("active_sequences", "[1]"),
                _i32("sequence_offsets", "[C+1]"), _i32("sequence_start_positions", "[C]"),
                _i32("state_sequence_ids", "[C]"))
    return _compile(launch, geometry=geometry, ratio=ratio, mode="continuation",
                    metadata=metadata, scalars=(Scalar("rows"), Scalar("groups"), Scalar("sequences")))
