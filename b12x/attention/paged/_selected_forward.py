"""Selected-position specialization of the CuTe paged-forward engine.

The engine supports direct BF16 output for large QSA prefills and FP32 split
partials for latency-sensitive small launches. Cache strides specialize the
compiled callable so both entries consume zero-copy per-layer views from an
interleaved BLHNC allocation without carrying layout arithmetic as runtime
state. Every page-scaled cache offset is widened to Int64 before multiplication.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import (
    BFloat16,
    Float32,
    Float8E4M3FN,
    Int32,
    Int64,
    Uint32,
    const_expr,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import (
    warp_mma_gemm,
)
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention._shared.cute.ops import fmax
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.compile_plan import compile_only_launches_enabled, program_keys, record_program
from b12x._lib.intrinsics import (
    atomic_add_shared_i32,
    bfloat2_to_float2_scaled,
    pack_f32x2_to_bfloat2,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._selected_forward_config import HEAD_DIM as _HEAD_DIM
from ._selected_forward_config import MAX_SPLITS as _NUM_SPLITS
from ._selected_forward_config import SELECTION_WIDTH as _SELECTION_WIDTH


_THREADS = 128
_WARPS = _THREADS // 32
_LOG2_E = 1.4426950408889634
_LOCK = RLock()
_KERNEL_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}
_MERGE_CACHE: dict[tuple[int, int, int, str], Callable[..., None]] = {}
_MERGE_WARMED: dict[tuple[int, int, int, str], Callable[..., None]] = {}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _exp2_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pointer_as_int64(pointer, offset, *, loc=None, ip=None) -> Int64:
    element_pointer = pointer + offset
    return Int64(llvm.ptrtoint(T.i64(), element_pointer.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _cp_async_load_128b_zfill(
    smem_addr: Int32,
    gmem_addr: Int64,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _vote_ballot_sync(predicate: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(predicate).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc_u32(value: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _SelectedPositionPagedForwardKernel:
    """Paged-forward entry over caller-selected logical K/V positions."""

    def __init__(
        self,
        *,
        q_heads: int,
        kv_heads: int,
        kv_is_fp8: bool,
        direct_output: bool,
        kv_warps: int,
        page_size: int,
        key_strides: tuple[int, int, int],
        value_strides: tuple[int, int, int],
        selection_width: int = _SELECTION_WIDTH,
        head_dim: int = _HEAD_DIM,
        selection_groups: int = 1,
        inline_scale_offsets: tuple[int, int] | None = None,
    ) -> None:
        self.q_heads = int(q_heads)
        self.selection_width = int(selection_width)
        self.head_dim = int(head_dim)
        self.selection_groups = int(selection_groups)
        if self.head_dim not in (192, 256):
            raise ValueError("selected-position head_dim must be 192 or 256")
        if self.selection_width <= 0:
            raise ValueError("selected-position capacity must be positive")
        self.kv_heads = int(kv_heads)
        if self.kv_heads <= 0 or self.q_heads <= 0 or self.q_heads % self.kv_heads:
            raise ValueError("query heads must be a positive multiple of KV heads")
        if self.selection_groups <= 0 or self.kv_heads % self.selection_groups:
            raise ValueError("selection groups must divide KV heads")
        self.heads_per_kv = self.q_heads // self.kv_heads
        self.kv_is_fp8 = bool(kv_is_fp8)
        self.inline_scale_offsets = inline_scale_offsets
        if inline_scale_offsets is not None and not self.kv_is_fp8:
            raise ValueError("inline descales require FP8 KV")
        self.direct_output = bool(direct_output)
        self.kv_warps = int(kv_warps)
        if self.kv_warps not in (1, 2, 4):
            raise ValueError(
                "selected-position paged forward requires 1, 2, or 4 KV warps"
            )
        self.tile_n = 16 * self.kv_warps
        self.tile_count = (self.selection_width + self.tile_n - 1) // self.tile_n
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise ValueError(
                "selected-position paged forward page_size must be positive"
            )
        self.key_page_stride, self.key_token_stride, self.key_head_stride = map(
            int, key_strides
        )
        (
            self.value_page_stride,
            self.value_token_stride,
            self.value_head_stride,
        ) = map(int, value_strides)
        self.shared_cache_strides = tuple(map(int, key_strides)) == tuple(
            map(int, value_strides)
        )
        self.slot_m = (
            8
            if self.heads_per_kv <= 8
            else (16 if self.heads_per_kv <= 16 else 32)
        )
        self.qk_warps = 2
        self.pv_warps = self.qk_warps * 4
        self.tile_m = self.qk_warps * 16
        self.rows_per_cta = self.tile_m // self.slot_m
        self.threads = self.pv_warps * 32

    def _tiled_mma_qk(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.qk_warps, 1, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _tiled_mma_pv(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.qk_warps, 4, 1),
            permutation_mnk=(self.tile_m, self.head_dim, 16),
        )

    @cute.jit
    def _descale_pair(self, pair: Uint32, scale: Float32) -> Uint32:
        low, high = bfloat2_to_float2_scaled(pair, scale)
        return pack_f32x2_to_bfloat2(low, high)

    @cute.jit
    def _load_fp8_vector_to_bf16_shared(
        self,
        source: cute.Pointer,
        source_base: Int64,
        destination: cute.Tensor,
        token: Int32,
        dimension_base: Int32,
        head: Int32,
        head_stride: cutlass.Constexpr,
        scale_offset: cutlass.Constexpr,
    ) -> None:
        x0 = Uint32(0)
        x1 = Uint32(0)
        x2 = Uint32(0)
        x3 = Uint32(0)
        if source_base >= Int64(0):
            x0, x1, x2, x3 = ld_global_nc_v4_u32(
                _pointer_as_int64(source, source_base + dimension_base.to(Int64))
            )
        y00, y01 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x0)
        y10, y11 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x1)
        y20, y21 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x2)
        y30, y31 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x3)
        if const_expr(self.inline_scale_offsets is not None):
            scale = Float32(1.0)
            if source_base >= Int64(0):
                address = (_pointer_as_int64(source, source_base)
                           - head.to(Int64) * Int64(head_stride)
                           + Int64(scale_offset) + head.to(Int64) * Int64(4))
                scale = cute.make_tensor(cute.make_ptr(Float32, address, cute.AddressSpace.gmem,
                                                      assumed_align=4), cute.make_layout((1,)))[0]
            y00 = self._descale_pair(y00, scale)
            y01 = self._descale_pair(y01, scale)
            y10 = self._descale_pair(y10, scale)
            y11 = self._descale_pair(y11, scale)
            y20 = self._descale_pair(y20, scale)
            y21 = self._descale_pair(y21, scale)
            y30 = self._descale_pair(y30, scale)
            y31 = self._descale_pair(y31, scale)
        destination0 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx((token, dimension_base), destination.layout)
        )
        destination1 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx(
                (token, dimension_base + Int32(8)),
                destination.layout,
            )
        )
        st_shared_v4_u32(destination0, y00, y01, y10, y11)
        st_shared_v4_u32(destination1, y20, y21, y30, y31)

    def _shared_layouts(
        self,
    ) -> tuple[
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
    ]:
        head_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.head_dim,
            ),
            BFloat16,
        )
        probability_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.tile_n,
            ),
            BFloat16,
        )
        query_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_m, self.head_dim),
            order=(0, 1),
        )
        key_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, self.head_dim),
            order=(0, 1),
        )
        value_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, self.head_dim),
            order=(0, 1),
        )
        probability_layout = cute.tile_to_shape(
            probability_layout_atom,
            (self.tile_m, self.tile_n),
            order=(0, 1),
        )
        return query_layout, key_layout, value_layout, probability_layout

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        query_layout, key_layout, value_layout, probability_layout = (
            self._shared_layouts()
        )
        tiled_mma_qk = self._tiled_mma_qk()
        tiled_mma_pv = self._tiled_mma_pv()
        row_groups = (rows + Int32(self.rows_per_cta - 1)) // Int32(
            self.rows_per_cta
        )
        self.kernel(
            query,
            key_cache,
            value_cache,
            k_descale,
            v_descale,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            partial_output,
            partial_lse,
            output,
            num_cache_pages,
            table_batch,
            table_width,
            softmax_scale,
            rows,
            splits,
            query_layout,
            key_layout,
            value_layout,
            probability_layout,
            tiled_mma_qk,
            tiled_mma_pv,
        ).launch(
            grid=(row_groups, self.kv_heads, splits),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _resolve_selected_bases_warp(
        self,
        tile: Int32,
        stage: Int32,
        kv_head: Int32,
        position_base: Int64,
        table_row_base: Int64,
        request_valid,
        query_position: Int64,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        key_bases: cute.Tensor,
        value_bases: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
        lane: Int32,
    ) -> None:
        key_head_offset = kv_head.to(Int64) * Int64(self.key_head_stride)
        value_head_offset = kv_head.to(Int64) * Int64(self.value_head_stride)
        for col_iter in cutlass.range_constexpr((self.tile_n + 31) // 32):
            column = lane + Int32(col_iter * 32)
            if column < Int32(self.tile_n):
                selected_column = tile * Int32(self.tile_n) + column
                valid = request_valid & (selected_column < Int32(self.selection_width))
                logical_position = Int32(-1)
                if valid:
                    logical_position = Int32(
                        selected_positions[position_base + selected_column.to(Int64)]
                    )
                    valid = (logical_position >= Int32(0)) & (
                        logical_position.to(Int64) <= query_position
                    )

                physical_page = Int64(-1)
                page_offset = Int64(0)
                if valid:
                    logical_page = logical_position // Int32(self.page_size)
                    valid = logical_page.to(Int64) < table_width
                    if valid:
                        physical_page = block_table[
                            table_row_base + logical_page.to(Int64)
                        ].to(Int64)
                        valid = (physical_page >= Int64(0)) & (
                            physical_page < num_cache_pages
                        )
                        page_offset = (
                            logical_position - logical_page * Int32(self.page_size)
                        ).to(Int64)

                key_base = Int64(-1)
                value_base = Int64(-1)
                if valid:
                    key_base = (
                        physical_page * Int64(self.key_page_stride)
                        + page_offset * Int64(self.key_token_stride)
                        + key_head_offset
                    )
                    if const_expr(self.shared_cache_strides):
                        value_base = key_base
                    else:
                        value_base = (
                            physical_page * Int64(self.value_page_stride)
                            + page_offset * Int64(self.value_token_stride)
                            + value_head_offset
                        )
                key_bases[stage, column] = key_base
                if const_expr(not self.shared_cache_strides):
                    value_bases[stage, column] = value_base

    @cute.jit
    def _issue_key_load(
        self,
        stage: Int32,
        kv_head: Int32,
        key_cache: cute.Pointer,
        key_bases: cute.Tensor,
        shared_key: cute.Tensor,
        thread: Int32,
    ) -> None:
        if const_expr(not self.kv_is_fp8):
            tokens_per_iter = self.threads // 8
            sub_token = thread // Int32(8)
            dimension_col = (thread % Int32(8)) * Int32(8)
            for tok_iter in cutlass.range_constexpr(
                (self.tile_n + tokens_per_iter - 1) // tokens_per_iter
            ):
                token = sub_token + Int32(tok_iter * tokens_per_iter)
                if const_expr(self.tile_n % tokens_per_iter == 0):
                    key_base = Int64(key_bases[stage, token])
                    key_valid = key_base >= Int64(0)
                    safe_key_base = Int64(
                        cutlass.select_(key_valid, key_base, Int64(0))
                    )
                    src_bytes = cutlass.select_(key_valid, Int32(16), Int32(0))
                    for chunk in cutlass.range_constexpr(self.head_dim // 64):
                        dimension_base = dimension_col + Int32(chunk * 64)
                        key_destination = shared_ptr_to_u32(
                            shared_key.iterator
                            + cute.crd2idx(
                                (token, dimension_base),
                                shared_key.layout,
                            )
                        )
                        _cp_async_load_128b_zfill(
                            key_destination,
                            _pointer_as_int64(
                                key_cache,
                                safe_key_base + dimension_base.to(Int64),
                            ),
                            src_bytes,
                        )
                else:
                    if token < Int32(self.tile_n):
                        key_base = Int64(key_bases[stage, token])
                        key_valid = key_base >= Int64(0)
                        safe_key_base = Int64(
                            cutlass.select_(key_valid, key_base, Int64(0))
                        )
                        src_bytes = cutlass.select_(key_valid, Int32(16), Int32(0))
                        for chunk in cutlass.range_constexpr(self.head_dim // 64):
                            dimension_base = dimension_col + Int32(chunk * 64)
                            key_destination = shared_ptr_to_u32(
                                shared_key.iterator
                                + cute.crd2idx(
                                    (token, dimension_base),
                                    shared_key.layout,
                                )
                            )
                            _cp_async_load_128b_zfill(
                                key_destination,
                                _pointer_as_int64(
                                    key_cache,
                                    safe_key_base + dimension_base.to(Int64),
                                ),
                                src_bytes,
                            )
            cute.arch.cp_async_commit_group()
        else:
            vectors_per_row = self.head_dim // 16
            for vector_iter in cutlass.range_constexpr(
                (self.tile_n * vectors_per_row + self.threads - 1) // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                if linear_vector < Int32(self.tile_n * vectors_per_row):
                    token = linear_vector // Int32(vectors_per_row)
                    dimension_base = (
                        linear_vector % Int32(vectors_per_row)
                    ) * Int32(16)
                    key_base = Int64(key_bases[stage, token])
                    self._load_fp8_vector_to_bf16_shared(
                        key_cache,
                        key_base,
                        shared_key,
                        token,
                        dimension_base,
                        kv_head,
                        self.key_head_stride,
                        self.inline_scale_offsets[0]
                        if self.inline_scale_offsets is not None
                        else 0,
                    )

    @cute.jit
    def _issue_value_load(
        self,
        stage: Int32,
        kv_head: Int32,
        value_cache: cute.Pointer,
        key_bases: cute.Tensor,
        value_bases: cute.Tensor,
        shared_value: cute.Tensor,
        thread: Int32,
    ) -> None:
        if const_expr(not self.kv_is_fp8):
            tokens_per_iter = self.threads // 8
            sub_token = thread // Int32(8)
            dimension_col = (thread % Int32(8)) * Int32(8)
            for tok_iter in cutlass.range_constexpr(
                (self.tile_n + tokens_per_iter - 1) // tokens_per_iter
            ):
                token = sub_token + Int32(tok_iter * tokens_per_iter)
                if const_expr(self.tile_n % tokens_per_iter == 0):
                    value_base = Int64(key_bases[stage, token])
                    if const_expr(not self.shared_cache_strides):
                        value_base = Int64(value_bases[stage, token])
                    value_valid = value_base >= Int64(0)
                    safe_value_base = Int64(
                        cutlass.select_(value_valid, value_base, Int64(0))
                    )
                    src_bytes = cutlass.select_(value_valid, Int32(16), Int32(0))
                    for chunk in cutlass.range_constexpr(self.head_dim // 64):
                        dimension_base = dimension_col + Int32(chunk * 64)
                        value_destination = shared_ptr_to_u32(
                            shared_value.iterator
                            + cute.crd2idx(
                                (token, dimension_base),
                                shared_value.layout,
                            )
                        )
                        _cp_async_load_128b_zfill(
                            value_destination,
                            _pointer_as_int64(
                                value_cache,
                                safe_value_base + dimension_base.to(Int64),
                            ),
                            src_bytes,
                        )
                else:
                    if token < Int32(self.tile_n):
                        value_base = Int64(key_bases[stage, token])
                        if const_expr(not self.shared_cache_strides):
                            value_base = Int64(value_bases[stage, token])
                        value_valid = value_base >= Int64(0)
                        safe_value_base = Int64(
                            cutlass.select_(value_valid, value_base, Int64(0))
                        )
                        src_bytes = cutlass.select_(value_valid, Int32(16), Int32(0))
                        for chunk in cutlass.range_constexpr(self.head_dim // 64):
                            dimension_base = dimension_col + Int32(chunk * 64)
                            value_destination = shared_ptr_to_u32(
                                shared_value.iterator
                                + cute.crd2idx(
                                    (token, dimension_base),
                                    shared_value.layout,
                                )
                            )
                            _cp_async_load_128b_zfill(
                                value_destination,
                                _pointer_as_int64(
                                    value_cache,
                                    safe_value_base + dimension_base.to(Int64),
                                ),
                                src_bytes,
                            )
            cute.arch.cp_async_commit_group()
        else:
            vectors_per_row = self.head_dim // 16
            for vector_iter in cutlass.range_constexpr(
                (self.tile_n * vectors_per_row + self.threads - 1) // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                if linear_vector < Int32(self.tile_n * vectors_per_row):
                    token = linear_vector // Int32(vectors_per_row)
                    dimension_base = (
                        linear_vector % Int32(vectors_per_row)
                    ) * Int32(16)
                    value_base = Int64(key_bases[stage, token])
                    if const_expr(not self.shared_cache_strides):
                        value_base = Int64(value_bases[stage, token])
                    self._load_fp8_vector_to_bf16_shared(
                        value_cache,
                        value_base,
                        shared_value,
                        token,
                        dimension_base,
                        kv_head,
                        self.value_head_stride,
                        self.inline_scale_offsets[1]
                        if self.inline_scale_offsets is not None
                        else 0,
                    )

    @cute.jit
    def _selected_qk_softmax(
        self,
        stage: Int32,
        slots_per_pass: Int32,
        key_bases: cute.Tensor,
        shared_probability: cute.Tensor,
        shared_row_scale: cute.Tensor,
        thread: Int32,
        tiled_mma_qk: cute.TiledMma,
        register_query: cute.Tensor,
        register_key: cute.Tensor,
        copy_query: cute.TiledCopy,
        copy_key: cute.TiledCopy,
        copy_source_query: cute.Tensor,
        copy_source_key: cute.Tensor,
        softmax: Softmax,
        is_first: cutlass.Constexpr,
    ) -> None:
        if thread < Int32(self.qk_warps * 32):
            accumulator_score = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C(
                    (self.tile_m, self.tile_n)
                ),
                Float32,
            )
            accumulator_score.fill(0.0)
            warp_mma_gemm(
                tiled_mma_qk,
                accumulator_score,
                register_query,
                register_key,
                copy_source_query,
                copy_source_key,
                copy_query,
                copy_key,
                A_in_regs=not is_first,
            )
            score_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    m_row = coordinate[0]
                    slot = m_row // Int32(self.slot_m)
                    local_head = m_row - slot * Int32(self.slot_m)
                    if (
                        (slot >= slots_per_pass)
                        | (local_head >= Int32(self.heads_per_kv))
                        | (Int64(key_bases[stage, coordinate[1]]) < Int64(0))
                    ):
                        score_mn[m, n] = -Float32.inf

            row_scale = softmax.online_softmax(
                accumulator_score,
                is_first=is_first,
                check_inf=True,
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    shared_probability[coordinate[0], coordinate[1]] = BFloat16(
                        score_mn[m, n]
                    )
                if score_coordinates[m, 0][1] == Int32(0):
                    shared_row_scale[score_coordinates[m, 0][0]] = row_scale[m]

    @cute.jit
    def _selected_pv_mma(
        self,
        shared_row_scale: cute.Tensor,
        thread: Int32,
        tiled_mma_pv: cute.TiledMma,
        register_probability: cute.Tensor,
        register_value: cute.Tensor,
        copy_probability: cute.TiledCopy,
        copy_value: cute.TiledCopy,
        copy_source_probability: cute.Tensor,
        copy_source_value: cute.Tensor,
        accumulator_output: cute.Tensor,
        is_first: cutlass.Constexpr,
    ) -> None:
        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, self.head_dim))
            )
        )
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
        warp_mma_gemm(
            tiled_mma_pv,
            accumulator_output,
            register_probability,
            register_value,
            copy_source_probability,
            copy_source_value,
            copy_probability,
            copy_value,
        )

    @cute.jit
    def _scan_selected_flags(
        self,
        split: Int32,
        position_base: Int64,
        request_valid,
        query_position: Int64,
        selected_positions: cute.Pointer,
        tile_flags: cute.Tensor,
        tiles_per_split: Int32,
        splits: Int32,
        thread: Int32,
    ) -> None:
        """Flag every extra tile (local_tile >= 1) that can hold a selected position."""
        extra_tiles = tiles_per_split - Int32(1)
        if const_expr(self.selection_width % 4 == 0):
            groups_per_tile = self.tile_n // 4
            scan_total = extra_tiles * Int32(groups_per_tile)
            for scan_iter in cutlass.range(
                (scan_total + Int32(self.threads) - 1) // Int32(self.threads),
                unroll=1,
            ):
                linear = thread + Int32(scan_iter * self.threads)
                if linear < scan_total:
                    local_tile = Int32(1) + linear // Int32(groups_per_tile)
                    column_group = linear % Int32(groups_per_tile)
                    tile = split + local_tile * splits
                    selected_column = (
                        tile * Int32(self.tile_n) + column_group * Int32(4)
                    )
                    if request_valid & (
                        selected_column < Int32(self.selection_width)
                    ):
                        word0, word1, word2, word3 = ld_global_nc_v4_u32(
                            _pointer_as_int64(
                                selected_positions,
                                position_base + selected_column.to(Int64),
                            )
                        )
                        selected0 = word0.to(Int32).to(Int64)
                        selected1 = word1.to(Int32).to(Int64)
                        selected2 = word2.to(Int32).to(Int64)
                        selected3 = word3.to(Int32).to(Int64)
                        valid = (
                            (
                                (selected0 >= Int64(0))
                                & (selected0 <= query_position)
                            )
                            | (
                                (selected1 >= Int64(0))
                                & (selected1 <= query_position)
                            )
                            | (
                                (selected2 >= Int64(0))
                                & (selected2 <= query_position)
                            )
                            | (
                                (selected3 >= Int64(0))
                                & (selected3 <= query_position)
                            )
                        )
                        if valid:
                            tile_flags[local_tile] = Int32(1)
        else:
            scan_total = extra_tiles * Int32(self.tile_n)
            for scan_iter in cutlass.range(
                (scan_total + Int32(self.threads) - 1) // Int32(self.threads),
                unroll=1,
            ):
                linear = thread + Int32(scan_iter * self.threads)
                if linear < scan_total:
                    local_tile = Int32(1) + linear // Int32(self.tile_n)
                    column = linear % Int32(self.tile_n)
                    tile = split + local_tile * splits
                    selected_column = tile * Int32(self.tile_n) + column
                    valid = request_valid & (
                        selected_column < Int32(self.selection_width)
                    )
                    if valid:
                        logical_position = selected_positions[
                            position_base + selected_column.to(Int64)
                        ].to(Int64)
                        valid = (logical_position >= Int64(0)) & (
                            logical_position <= query_position
                        )
                    if valid:
                        tile_flags[local_tile] = Int32(1)

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        rows: Int32,
        splits: Int32,
        query_layout: cute.ComposedLayout,
        key_layout: cute.ComposedLayout,
        value_layout: cute.ComposedLayout,
        probability_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
    ) -> None:
        row_group_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        thread = Int32(thread_idx)
        split = Int32(split_idx)
        kv_head = Int32(kv_head_idx)
        row_base = Int32(row_group_idx) * Int32(self.rows_per_cta)
        remaining_rows = rows - row_base
        active_rows = cutlass.select_(
            remaining_rows < Int32(self.rows_per_cta),
            remaining_rows,
            Int32(self.rows_per_cta),
        )
        selection_group = kv_head // Int32(self.kv_heads // self.selection_groups)

        k_scale = Float32(1.0)
        v_scale = Float32(1.0)
        if const_expr(self.kv_is_fp8 and self.inline_scale_offsets is None):
            k_scale = Float32(k_descale[0])
            v_scale = Float32(v_descale[0])

        allocator = cutlass.utils.SmemAllocator()
        shared_query = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=query_layout,
            byte_alignment=1024,
        )
        shared_key = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=key_layout,
            byte_alignment=1024,
        )
        shared_value = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=value_layout,
            byte_alignment=1024,
        )
        shared_probability = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=probability_layout,
            byte_alignment=1024,
        )
        shared_row_scale = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((self.tile_m,), stride=(1,)),
            byte_alignment=16,
        )
        key_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((2, self.tile_n), stride=(self.tile_n, 1)),
            byte_alignment=16,
        )
        value_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((2, self.tile_n), stride=(self.tile_n, 1)),
            byte_alignment=16,
        )
        tile_flags = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.tile_count,), stride=(1,)),
            byte_alignment=4,
        )
        active_tiles = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.tile_count,), stride=(1,)),
            byte_alignment=4,
        )
        shared_num_extra = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((1,), stride=(1,)),
            byte_alignment=4,
        )
        shared_warp_mismatch = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.pv_warps,), stride=(1,)),
            byte_alignment=16,
        )

        tile_count = Int32(self.tile_count)
        tiles_per_split = (tile_count - split + splits - Int32(1)) // splits
        pass_count = cutlass.select_(
            active_rows > Int32(0), Int32(1), Int32(0)
        )
        slots_per_pass = active_rows
        if const_expr(self.rows_per_cta > 1):
            if active_rows > Int32(1):
                req0 = request_ids[row_base].to(Int64)
                qpos0 = query_positions[row_base].to(Int64)
                pos_base0 = (
                    row_base.to(Int64) * Int64(self.selection_groups)
                    + selection_group.to(Int64)
                ) * Int64(self.selection_width)
                local_mismatch = Int32(0)
                for s_idx in cutlass.range_constexpr(self.rows_per_cta - 1):
                    s = s_idx + 1
                    if Int32(s) < active_rows:
                        row_s = row_base + Int32(s)
                        if (request_ids[row_s].to(Int64) != req0) | (
                            query_positions[row_s].to(Int64) != qpos0
                        ):
                            local_mismatch = Int32(1)
                if local_mismatch == Int32(0):
                    if const_expr(self.selection_width % 4 == 0):
                        groups_per_tile = self.tile_n // 4
                        check_total = tiles_per_split * Int32(groups_per_tile)
                        for check_iter in cutlass.range(
                            (check_total + Int32(self.threads) - 1)
                            // Int32(self.threads),
                            unroll=1,
                        ):
                            linear = thread + Int32(check_iter * self.threads)
                            if linear < check_total:
                                local_tile = linear // Int32(groups_per_tile)
                                column_group = linear % Int32(groups_per_tile)
                                tile = split + local_tile * splits
                                selected_column = (
                                    tile * Int32(self.tile_n)
                                    + column_group * Int32(4)
                                )
                                if selected_column < Int32(self.selection_width):
                                    w0, w1, w2, w3 = ld_global_nc_v4_u32(
                                        _pointer_as_int64(
                                            selected_positions,
                                            pos_base0 + selected_column.to(Int64),
                                        )
                                    )
                                    for s_idx in cutlass.range_constexpr(
                                        self.rows_per_cta - 1
                                    ):
                                        s = s_idx + 1
                                        if Int32(s) < active_rows:
                                            pos_base_s = (
                                                pos_base0
                                                + Int64(s)
                                                * Int64(self.selection_groups)
                                                * Int64(self.selection_width)
                                            )
                                            u0, u1, u2, u3 = ld_global_nc_v4_u32(
                                                _pointer_as_int64(
                                                    selected_positions,
                                                    pos_base_s
                                                    + selected_column.to(Int64),
                                                )
                                            )
                                            if (
                                                (u0.to(Int32) != w0.to(Int32))
                                                | (u1.to(Int32) != w1.to(Int32))
                                                | (u2.to(Int32) != w2.to(Int32))
                                                | (u3.to(Int32) != w3.to(Int32))
                                            ):
                                                local_mismatch = Int32(1)
                    else:
                        check_total = tiles_per_split * Int32(self.tile_n)
                        for check_iter in cutlass.range(
                            (check_total + Int32(self.threads) - 1)
                            // Int32(self.threads),
                            unroll=1,
                        ):
                            linear = thread + Int32(check_iter * self.threads)
                            if linear < check_total:
                                local_tile = linear // Int32(self.tile_n)
                                column = linear % Int32(self.tile_n)
                                tile = split + local_tile * splits
                                selected_column = (
                                    tile * Int32(self.tile_n) + column
                                )
                                if selected_column < Int32(self.selection_width):
                                    val0 = Int32(
                                        selected_positions[
                                            pos_base0 + selected_column.to(Int64)
                                        ]
                                    )
                                    for s_idx in cutlass.range_constexpr(
                                        self.rows_per_cta - 1
                                    ):
                                        s = s_idx + 1
                                        if Int32(s) < active_rows:
                                            pos_base_s = (
                                                pos_base0
                                                + Int64(s)
                                                * Int64(self.selection_groups)
                                                * Int64(self.selection_width)
                                            )
                                            vals = Int32(
                                                selected_positions[
                                                    pos_base_s
                                                    + selected_column.to(Int64)
                                                ]
                                            )
                                            if vals != val0:
                                                local_mismatch = Int32(1)
                warp_id = thread // Int32(32)
                lane = thread % Int32(32)
                mismatch_ballot = _vote_ballot_sync(local_mismatch)
                if lane == Int32(0):
                    shared_warp_mismatch[warp_id] = _popc_u32(mismatch_ballot)
                cute.arch.sync_threads()
                any_mismatch = Int32(0)
                for w in cutlass.range_constexpr(self.pv_warps):
                    any_mismatch = any_mismatch | Int32(shared_warp_mismatch[w])
                if any_mismatch != Int32(0):
                    pass_count = active_rows
                    slots_per_pass = Int32(1)

        qk_thread = thread % Int32(self.qk_warps * 32)
        thread_mma_qk = tiled_mma_qk.get_slice(qk_thread)
        thread_mma_pv = tiled_mma_pv.get_slice(thread)
        register_query = thread_mma_qk.make_fragment_A(
            thread_mma_qk.partition_A(shared_query)
        )
        register_key = thread_mma_qk.make_fragment_B(
            thread_mma_qk.partition_B(shared_key)
        )
        shared_value_transposed = layout_utils.transpose_view(shared_value)
        register_probability = thread_mma_pv.make_fragment_A(
            thread_mma_pv.partition_A(shared_probability)
        )
        register_value = thread_mma_pv.make_fragment_B(
            thread_mma_pv.partition_B(shared_value_transposed)
        )
        copy_atom_query_key = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            BFloat16,
        )
        copy_atom_value = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            BFloat16,
        )
        copy_query = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_qk
        ).get_slice(qk_thread)
        copy_key = cute.make_tiled_copy_B(copy_atom_query_key, tiled_mma_qk).get_slice(
            qk_thread
        )
        copy_probability = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_pv
        ).get_slice(thread)
        copy_value = cute.make_tiled_copy_B(copy_atom_value, tiled_mma_pv).get_slice(
            thread
        )
        copy_source_query = copy_query.partition_S(shared_query)
        copy_source_key = copy_key.partition_S(shared_key)
        copy_source_probability = copy_probability.partition_S(shared_probability)
        copy_source_value = copy_value.partition_S(shared_value_transposed)
        accumulator_output = cute.make_rmem_tensor(
            thread_mma_pv.partition_shape_C((self.tile_m, self.head_dim)),
            Float32,
        )
        score_layout = thread_mma_qk.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n))
        ).layout
        score_mn_layout = layout_utils.convert_layout_acc_mn(score_layout)
        score_scale = Float32(softmax_scale * Float32(_LOG2_E))
        if const_expr(self.kv_is_fp8):
            score_scale *= k_scale
        softmax = Softmax.create(
            score_scale,
            cute.size(score_mn_layout.shape[0]),
            arch=120,
        )
        first_query_head = kv_head * Int32(self.heads_per_kv)

        for pass_idx in cutlass.range(pass_count, unroll=1):
            if pass_idx > Int32(0):
                cute.arch.sync_threads()
            pass_row = row_base + pass_idx
            request_id = request_ids[pass_row].to(Int64)
            query_position = query_positions[pass_row].to(Int64)
            request_valid = (request_id >= Int64(0)) & (request_id < table_batch)
            position_base = (
                pass_row.to(Int64) * Int64(self.selection_groups)
                + selection_group.to(Int64)
            ) * Int64(self.selection_width)
            table_row_base = request_id * table_width

            flag_element = thread
            while flag_element < tiles_per_split:
                tile_flags[flag_element] = Int32(0)
                flag_element += Int32(self.threads)
            if thread == Int32(0):
                shared_num_extra[0] = Int32(0)

            query_element = thread
            while query_element < Int32(self.tile_m * self.head_dim):
                m_row = query_element // Int32(self.head_dim)
                dimension = query_element - m_row * Int32(self.head_dim)
                slot = m_row // Int32(self.slot_m)
                local_head = m_row - slot * Int32(self.slot_m)
                query_value = Float32(0.0)
                if (slot < slots_per_pass) & (
                    local_head < Int32(self.heads_per_kv)
                ):
                    query_row = pass_row + slot
                    query_head = first_query_head + local_head
                    query_offset = (
                        query_row.to(Int64) * Int64(self.q_heads)
                        + query_head.to(Int64)
                    ) * Int64(self.head_dim) + dimension.to(Int64)
                    query_value = Float32(query[query_offset])
                shared_query[m_row, dimension] = BFloat16(query_value)
                query_element += Int32(self.threads)

            if thread < Int32(32):
                self._resolve_selected_bases_warp(
                    split,
                    Int32(0),
                    kv_head,
                    position_base,
                    table_row_base,
                    request_valid,
                    query_position,
                    block_table,
                    selected_positions,
                    key_bases,
                    value_bases,
                    num_cache_pages,
                    table_width,
                    thread,
                )
            cute.arch.sync_threads()

            accumulator_output.fill(0.0)
            softmax.reset()

            self._issue_key_load(
                Int32(0),
                kv_head,
                key_cache,
                key_bases,
                shared_key,
                thread,
            )
            self._issue_value_load(
                Int32(0),
                kv_head,
                value_cache,
                key_bases,
                value_bases,
                shared_value,
                thread,
            )
            if tiles_per_split > Int32(1):
                self._scan_selected_flags(
                    split,
                    position_base,
                    request_valid,
                    query_position,
                    selected_positions,
                    tile_flags,
                    tiles_per_split,
                    splits,
                    thread,
                )
            if const_expr(not self.kv_is_fp8):
                cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            self._selected_qk_softmax(
                Int32(0),
                slots_per_pass,
                key_bases,
                shared_probability,
                shared_row_scale,
                thread,
                tiled_mma_qk,
                register_query,
                register_key,
                copy_query,
                copy_key,
                copy_source_query,
                copy_source_key,
                softmax,
                is_first=True,
            )
            if (thread >= Int32(self.qk_warps * 32)) & (
                thread < Int32((self.qk_warps + 1) * 32)
            ):
                if tiles_per_split > Int32(1):
                    resolver_lane = thread - Int32(self.qk_warps * 32)
                    num_extra = Int32(0)
                    lane_mask = (Uint32(1) << Uint32(resolver_lane)) - Uint32(1)
                    chunk_count = (tiles_per_split + Int32(30)) // Int32(32)
                    for chunk in cutlass.range(chunk_count, unroll=1):
                        local_tile = Int32(1) + chunk * Int32(32) + resolver_lane
                        hit = Int32(0)
                        if local_tile < tiles_per_split:
                            if tile_flags[local_tile] != Int32(0):
                                hit = Int32(1)
                        ballot = _vote_ballot_sync(hit)
                        if hit != Int32(0):
                            active_tiles[
                                num_extra + _popc_u32(ballot & lane_mask)
                            ] = split + local_tile * splits
                        num_extra += _popc_u32(ballot)
                    if resolver_lane == Int32(0):
                        shared_num_extra[0] = num_extra
                    if num_extra > Int32(0):
                        cute.arch.sync_warp()
                        self._resolve_selected_bases_warp(
                            Int32(active_tiles[0]),
                            Int32(1),
                            kv_head,
                            position_base,
                            table_row_base,
                            request_valid,
                            query_position,
                            block_table,
                            selected_positions,
                            key_bases,
                            value_bases,
                            num_cache_pages,
                            table_width,
                            resolver_lane,
                        )
            if const_expr(not self.kv_is_fp8):
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            num_extra = Int32(shared_num_extra[0])
            if num_extra > Int32(0):
                self._issue_key_load(
                    Int32(1),
                    kv_head,
                    key_cache,
                    key_bases,
                    shared_key,
                    thread,
                )
            self._selected_pv_mma(
                shared_row_scale,
                thread,
                tiled_mma_pv,
                register_probability,
                register_value,
                copy_probability,
                copy_value,
                copy_source_probability,
                copy_source_value,
                accumulator_output,
                is_first=True,
            )

            for extra_idx in cutlass.range(num_extra, unroll=1):
                stage = (extra_idx + Int32(1)) & Int32(1)
                next_stage = extra_idx & Int32(1)
                cute.arch.sync_threads()
                self._issue_value_load(
                    stage,
                    kv_head,
                    value_cache,
                    key_bases,
                    value_bases,
                    shared_value,
                    thread,
                )
                if const_expr(not self.kv_is_fp8):
                    cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

                self._selected_qk_softmax(
                    stage,
                    slots_per_pass,
                    key_bases,
                    shared_probability,
                    shared_row_scale,
                    thread,
                    tiled_mma_qk,
                    register_query,
                    register_key,
                    copy_query,
                    copy_key,
                    copy_source_query,
                    copy_source_key,
                    softmax,
                    is_first=False,
                )
                if (thread >= Int32(self.qk_warps * 32)) & (
                    thread < Int32((self.qk_warps + 1) * 32)
                ):
                    if extra_idx + Int32(1) < num_extra:
                        resolver_lane = thread - Int32(self.qk_warps * 32)
                        self._resolve_selected_bases_warp(
                            Int32(active_tiles[extra_idx + Int32(1)]),
                            next_stage,
                            kv_head,
                            position_base,
                            table_row_base,
                            request_valid,
                            query_position,
                            block_table,
                            selected_positions,
                            key_bases,
                            value_bases,
                            num_cache_pages,
                            table_width,
                            resolver_lane,
                        )
                if const_expr(not self.kv_is_fp8):
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                if extra_idx + Int32(1) < num_extra:
                    self._issue_key_load(
                        next_stage,
                        kv_head,
                        key_cache,
                        key_bases,
                        shared_key,
                        thread,
                    )
                self._selected_pv_mma(
                    shared_row_scale,
                    thread,
                    tiled_mma_pv,
                    register_probability,
                    register_value,
                    copy_probability,
                    copy_value,
                    copy_source_probability,
                    copy_source_value,
                    accumulator_output,
                    is_first=False,
                )
            cute.arch.sync_threads()

            if thread < Int32(self.qk_warps * 32):
                output_scale = softmax.finalize()
                score_coordinates = layout_utils.reshape_acc_to_mn(
                    tiled_mma_qk.get_slice(qk_thread).partition_C(
                        cute.make_identity_tensor((self.tile_m, self.tile_n))
                    )
                )
                if score_coordinates[0, 0][1] == Int32(0):
                    for m in cutlass.range_constexpr(cute.size(softmax.row_sum)):
                        m_row = score_coordinates[m, 0][0]
                        slot = m_row // Int32(self.slot_m)
                        local_head = m_row - slot * Int32(self.slot_m)
                        scale = output_scale[m]
                        if const_expr(self.kv_is_fp8):
                            scale *= v_scale
                        shared_row_scale[m_row] = scale
                        if const_expr(not self.direct_output) and (
                            (slot < slots_per_pass)
                            & (local_head < Int32(self.heads_per_kv))
                        ):
                            query_row = pass_row + slot
                            query_head = first_query_head + local_head
                            lse_offset = (
                                query_row.to(Int64) * splits.to(Int64)
                                + split.to(Int64)
                            ) * Int64(self.q_heads) + query_head.to(Int64)
                            partial_lse[lse_offset] = Float32(softmax.row_sum[m])
            cute.arch.sync_threads()

            output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
            output_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_pv.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.head_dim))
                )
            )
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                m_row = output_coordinates[m, 0][0]
                slot = m_row // Int32(self.slot_m)
                local_head = m_row - slot * Int32(self.slot_m)
                scale = Float32(shared_row_scale[m_row])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
                if (slot < slots_per_pass) & (
                    local_head < Int32(self.heads_per_kv)
                ):
                    query_row = pass_row + slot
                    query_head = first_query_head + local_head
                    output_base = (
                        query_row.to(Int64) * Int64(self.q_heads)
                        if const_expr(self.direct_output)
                        else (
                            query_row.to(Int64) * splits.to(Int64)
                            + split.to(Int64)
                        )
                        * Int64(self.q_heads)
                    ) * Int64(self.head_dim) + query_head.to(Int64) * Int64(
                        self.head_dim
                    )
                    for n in cutlass.range_constexpr(cute.size(output_mn.shape[1])):
                        coordinate = output_coordinates[m, n]
                        offset = output_base + coordinate[1].to(Int64)
                        if const_expr(self.direct_output):
                            output[offset] = BFloat16(output_mn[m, n])
                        else:
                            partial_output[offset] = Float32(output_mn[m, n])


class _SparseGqaMergeKernel:
    """Merge FP32 split-softmax partials into caller-owned BF16 output."""

    def __init__(self, *, q_heads: int, head_dim: int = _HEAD_DIM) -> None:
        self.q_heads = int(q_heads)
        self.head_dim = int(head_dim)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, splits).launch(
            grid=(self.q_heads, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        row_i = Int64(row)
        head_i = Int64(query_head)
        lane = thread_i % Int32(32)
        warp_id = thread_i // Int32(32)
        weights = cute.make_rmem_tensor((_NUM_SPLITS // 32,), Float32)
        local_maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            local_maximum = fmax(local_maximum, weight)
        maximum = warp_reduce(local_maximum, fmax)

        local_denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            local_denominator += weight
        denominator = warp_reduce(local_denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((_NUM_SPLITS,), stride=(1,)),
            byte_alignment=16,
        )
        if warp_id == Int32(0):
            for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        # Adjacent lanes load adjacent head dimensions. Preserve the original
        # split-pair accumulation and ascending butterfly reduction order.
        for tile in cutlass.range_constexpr((self.head_dim + _THREADS - 1) // _THREADS):
            dimension = thread_i + Int32(tile * _THREADS)
            partials = cute.make_rmem_tensor((32,), Float32)
            for split_lane in cutlass.range_constexpr(32):
                total = Float32(0.0)
                for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                    split = Int32(split_lane + slot * 32)
                    if (split < splits) & (dimension < Int32(self.head_dim)):
                        offset = (
                            (row_i * splits.to(Int64) + split.to(Int64))
                            * Int64(self.q_heads)
                            + head_i
                        ) * Int64(self.head_dim) + dimension.to(Int64)
                        total += Float32(partial_output[offset]) * Float32(
                            shared_weights[split]
                        )
                partials[split_lane] = total
            for stage in cutlass.range_constexpr(5):
                for group in cutlass.range_constexpr(32 // (2 << stage)):
                    left = group * (2 << stage)
                    right = left + (1 << stage)
                    partials[left] = Float32(partials[left]) + Float32(partials[right])
            output_offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                self.head_dim
            ) + dimension.to(Int64)
            if dimension < Int32(self.head_dim):
                output[output_offset] = BFloat16(Float32(partials[0]) * inverse)


class _CooperativeSparseGqaMergeKernel(_SparseGqaMergeKernel):
    """Distribute each ordered split reduction across four warps and dimension tiles."""

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, splits).launch(
            grid=(self.q_heads, rows, self.head_dim // 32),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, tile = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        row_i, head_i = Int64(row), Int64(query_head)
        weights = cute.make_rmem_tensor((2,), Float32)
        maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(2):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            maximum = fmax(maximum, weight)
        maximum = warp_reduce(maximum, fmax)
        denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(2):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            denominator += weight
        denominator = warp_reduce(denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((64,), stride=(1,)),
            byte_alignment=16,
        )
        shared_partials = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((4, 32), stride=(32, 1)),
            byte_alignment=16,
        )
        if warp == Int32(0):
            for slot in cutlass.range_constexpr(2):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        dimension = Int64(tile) * Int64(32) + lane.to(Int64)
        partials = cute.make_rmem_tensor((8,), Float32)
        for local in cutlass.range_constexpr(8):
            total = Float32(0.0)
            for slot in cutlass.range_constexpr(2):
                split = warp * Int32(8) + Int32(local + slot * 32)
                if split < splits:
                    offset = (
                        (row_i * splits.to(Int64) + split.to(Int64))
                        * Int64(self.q_heads)
                        + head_i
                    ) * Int64(self.head_dim) + dimension
                    total += Float32(partial_output[offset]) * Float32(
                        shared_weights[split]
                    )
            partials[local] = total
        # Pair s with s+32; ascending butterfly stages 0..2 stay warp-local.
        for stage in cutlass.range_constexpr(3):
            for group in cutlass.range_constexpr(8 // (2 << stage)):
                left = group * (2 << stage)
                partials[left] = Float32(partials[left]) + Float32(
                    partials[left + (1 << stage)]
                )
        shared_partials[warp, lane] = partials[0]
        cute.arch.sync_threads()
        if warp == Int32(0):
            # Stages 3 and 4 join the four warp-owned contiguous groups.
            column = lane
            low = Float32(shared_partials[0, column]) + Float32(
                shared_partials[1, column]
            )
            high = Float32(shared_partials[2, column]) + Float32(
                shared_partials[3, column]
            )
            dimension = Int64(tile) * Int64(32) + column.to(Int64)
            offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                self.head_dim
            ) + dimension
            output[offset] = BFloat16((low + high) * inverse)


class _ShapeAdaptiveSparseGqaMergeKernel:
    """Trade split parallelism for independent query rows in one compiled callable."""

    def __init__(self, *, q_heads: int, head_dim: int = _HEAD_DIM) -> None:
        self.q_heads = q_heads
        self.head_dim = int(head_dim)
        self.serial = _SparseGqaMergeKernel(q_heads=q_heads, head_dim=head_dim)
        self.cooperative = _CooperativeSparseGqaMergeKernel(q_heads=q_heads, head_dim=head_dim)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # Independent query rows supply parallelism without multiplying CTAs
        # across head dimensions. Both branches preserve the same reduction tree.
        if rows > splits:
            self.serial.kernel(partial_output, partial_lse, output, splits).launch(
                grid=(self.q_heads, rows, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
            )
        else:
            self.cooperative.kernel(partial_output, partial_lse, output, splits).launch(
                grid=(self.q_heads, rows, self.head_dim // 32),
                block=(_THREADS, 1, 1),
                stream=stream,
            )


_MERGE_KERNEL = _ShapeAdaptiveSparseGqaMergeKernel


def _cache_key(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    direct_output: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
    selection_groups: int = 1,
) -> tuple[object, ...]:
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(request_ids.element_size() * 8),
        key_cache.dtype,
        int(key_cache.shape[1]),
        tuple(map(int, key_cache.stride()[:3])),
        tuple(map(int, value_cache.stride()[:3])),
        bool(direct_output),
        int(kv_warps),
        int(selection_width),
        int(query.shape[2]),
        int(selection_groups),
    )


def _pointer(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        16,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _compile(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    direct_output: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
    selection_groups: int = 1,
) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=direct_output,
        kv_warps=kv_warps,
        selection_width=selection_width,
        selection_groups=selection_groups,
    )
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return key, cached
        (
            _,
            q_heads,
            kv_heads,
            request_id_bits,
            kv_dtype,
            page_size,
            key_strides,
            value_strides,
            direct_output,
            kv_warps,
            selection_width,
            head_dim,
            selection_groups,
        ) = key
        request_id_type = Int32 if request_id_bits == 32 else Int64
        kv_type = Float8E4M3FN if kv_dtype == torch.float8_e4m3fn else BFloat16
        from .forward_extend_generic import PagedForwardKernel

        kernel = PagedForwardKernel.selected_positions(
            q_heads=q_heads,
            kv_heads=kv_heads,
            kv_is_fp8=kv_dtype == torch.float8_e4m3fn,
            direct_output=direct_output,
            kv_warps=kv_warps,
            page_size=page_size,
            key_strides=key_strides,
            value_strides=value_strides,
            selection_width=selection_width,
            head_dim=head_dim,
            selection_groups=selection_groups,
        )
        with torch.cuda.device(key[0]):
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(BFloat16),
                _fake_pointer(kv_type),
                _fake_pointer(kv_type),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(Int32),
                _fake_pointer(request_id_type),
                _fake_pointer(Int32),
                _fake_pointer(Int64),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                Int64(1),
                Int64(1),
                Int64(1),
                Float32(1.0),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.paged.selected_forward",
                    5,
                    (
                        q_heads,
                        kv_heads,
                        request_id_bits,
                        str(kv_dtype),
                        page_size,
                        key_strides,
                        value_strides,
                        direct_output,
                        kv_warps,
                        selection_width,
                        head_dim,
                        selection_groups,
                    ),
                    labels=(
                        "q_heads",
                        "kv_heads",
                        "request_id_bits",
                        "kv_dtype",
                        "page_size",
                        "key_strides",
                        "value_strides",
                        "direct_output",
                        "kv_warps",
                        "selection_width",
                        "head_dim",
                        "selection_groups",
                    ),
                ),
            )
        _KERNEL_CACHE[key] = raw
        return key, raw


def precompile_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    selection_width: int = _SELECTION_WIDTH,
    selection_groups: int = 1,
) -> None:
    """Compile the caller's planned selection capacity without storage writes."""
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe sparse GQA compilation is forbidden during capture"
            )
        _compile(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            request_ids=request_ids,
            direct_output=False,
            kv_warps=2,
            selection_width=selection_width,
            selection_groups=selection_groups,
        )


def launch_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
    splits: int,
) -> None:
    """Launch the specialized split core without allocation or fallback."""
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=False,
        kv_warps=2,
        selection_width=int(selected_positions.shape[-1]),
        selection_groups=(int(selected_positions.shape[1]) if selected_positions.ndim == 3 else 1),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA must be compiled and warm-run before CUDA graph "
                "capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=False,
                kv_warps=2,
                selection_width=int(selected_positions.shape[-1]),
                selection_groups=(int(selected_positions.shape[1]) if selected_positions.ndim == 3 else 1),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _fake_pointer(BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_selected_paged_gqa_direct(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    kv_warps: int,
) -> None:
    """Launch direct selected-position paged attention without split scratch."""
    kv_warps = int(kv_warps)
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=True,
        kv_warps=kv_warps,
        selection_width=int(selected_positions.shape[-1]),
        selection_groups=(int(selected_positions.shape[1]) if selected_positions.ndim == 3 else 1),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "selected-position paged attention must be compiled and warm-run "
                "before CUDA graph capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=True,
                kv_warps=kv_warps,
                selection_width=int(selected_positions.shape[-1]),
                selection_groups=(int(selected_positions.shape[1]) if selected_positions.ndim == 3 else 1),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _pointer(output, BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                1,
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_sparse_gqa_merge(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    splits: int,
) -> None:
    """Merge the Qwen split layout without allocation or fallback."""
    device_index = partial_output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    q_heads = int(partial_output.shape[2])
    head_dim = int(partial_output.shape[3])
    if head_dim not in (192, 256):
        raise ValueError("selected-position merge head_dim must be 192 or 256")
    key = (int(device_index), q_heads, head_dim, _MERGE_KERNEL.__name__)
    with torch.cuda.device(device_index):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _MERGE_CACHE.get(key)
            warmed = raw is not None and _MERGE_WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA merge must be compiled and warm-run before "
                "CUDA graph capture"
            )
        if raw is None:
            kernel = _MERGE_KERNEL(q_heads=q_heads, head_dim=head_dim)
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.sparse_gqa_merge",
                    10,
                    key[1:],
                ),
            )
            with _LOCK:
                _MERGE_CACHE[key] = raw
        if compile_only_launches_enabled():
            for program in program_keys(raw):
                record_program(program)
            return raw
        run_compiled(
            raw,
            (
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _pointer(output, BFloat16),
                int(rows),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _MERGE_CACHE.get(key) is raw:
                _MERGE_WARMED[key] = raw


def clear_caches() -> None:
    """Clear process-local compiled and warm-launch state for focused tests."""
    with _LOCK:
        _KERNEL_CACHE.clear()
        _WARMED.clear()
        _MERGE_CACHE.clear()
        _MERGE_WARMED.clear()


__all__ = [
    "clear_caches",
    "launch_selected_paged_gqa_direct",
    "launch_sparse_gqa_split",
    "launch_sparse_gqa_merge",
    "precompile_sparse_gqa_split",
]
