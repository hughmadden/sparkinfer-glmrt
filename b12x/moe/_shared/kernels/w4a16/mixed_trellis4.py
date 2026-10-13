"""Four-tier native mixed Trellis grid with an independent fixed launch ABI.

Shares the existing decoder and cooperative scheduler. This module is separate
so adding a fourth descriptor tier does not change two/three-tier launch ABIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import get_ptr_as_int64, shared_ptr_to_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.trellis_codebooks import (
    normalize_codebook,
    validate_codebook_bits,
)

from .kernel import (
    _SQG_XOR_CHEB_T12_LUT_ENTRIES,
    _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
    W4A16FusedMoeKernel,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _trellis256_execution_lut,
    compile_w4a16_topk_sum,
)
from .mixed_trellis import (
    MixedTrellis3CompileResult, W4A16MixedTrellisKernel,
    _MAX_TIER_EXPERTS, _TIER_DESCRIPTOR_BITS, _TIER_DESCRIPTOR_MASK,
    _select_mixed_fc2_kernel,
)

@dataclass(frozen=True)
class MixedTrellis4CompileResult(MixedTrellis3CompileResult):
    tier3_num_experts: int
    tier3_bits: int


def _normalize_mixed_trellis_format(
    codebook: str | int, bits: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    normalized = normalize_codebook(codebook)
    tiers = tuple(int(value) for value in bits)
    if len(tiers) != 4 or len(set(tiers)) != 4:
        raise ValueError("four-tier mixed Trellis requires four distinct bitrates")
    for value in tiers:
        validate_codebook_bits(normalized, value)
    return normalized, tiers


class W4A16MixedTrellis4Kernel(W4A16MixedTrellisKernel):
    """One cooperative grid over four native Trellis bitrates."""

    # The four-tier ABI has an independent persistent compile namespace.
    ABI_VERSION = 2

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        tier2: W4A16FusedMoeKernel,
        tier3: W4A16FusedMoeKernel,
    ):
        kernels = (driver, tier0, tier1, tier2, tier3)
        for name, moe in zip(
            ("driver", "tier0", "tier1", "tier2", "tier3"), kernels, strict=True
        ):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.direct_topk_routes or moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} requires route packing")
            if moe.weight_layout != "trellis_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "broadcast_suh",
            "trellis_codebook",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in kernels)
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in kernels)
            geometry = tuple(
                (
                    gemm.n_tiles,
                    gemm.k_tiles,
                    gemm.tile_n,
                    gemm.tile_k,
                    gemm.cta_threads,
                    gemm.moe_block_size,
                    gemm.schedule_route_block_factor,
                )
                for gemm in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        tiers = (tier0, tier1, tier2, tier3)
        if any(tier.num_experts > _MAX_TIER_EXPERTS for tier in tiers):
            raise ValueError("tier-local expert ids must fit in nine bits")
        if driver.num_experts != sum(tier.num_experts for tier in tiers):
            raise ValueError("driver expert count must equal the sum of all tiers")
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.tier2 = tier2
        self.tier3 = tier3
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.blocks_per_sm = min(tier.blocks_per_sm for tier in kernels)
        self.shared_words = max(tier.shared_words for tier in kernels)
        self.sqg_xor_cheb_t12_smem_off = max(
            tier.sqg_xor_cheb_t12_smem_off for tier in kernels
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis4",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.tier2.__cache_key__,
            self.tier3.__cache_key__,
            self.blocks_per_sm,
            self.shared_words,
        )

    @cute.jit
    def _emit_tier_tile4(
        self,
        is_fc1: cutlass.Constexpr,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_scales_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_scales_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        t2_b_flat: cute.Tensor,
        t2_scales_flat: cute.Tensor,
        t2_global_scale: cute.Tensor,
        t3_b_flat: cute.Tensor,
        t3_scales_flat: cute.Tensor,
        t3_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier2_num_experts: Int32,
        tier3_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier2_fc2_experts: Int32,
        tier3_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier2_gate_experts: Int32,
        tier3_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        tier2_up_experts: Int32,
        tier3_up_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts + tier3_num_experts
        descriptor_row = Int32(2)
        if cutlass.const_expr(is_fc1):
            fc1_half_tiles = Int32(self.driver.fc1.n_tiles // 2)
            descriptor_row = Int32(0)
            if output_n_tile >= fc1_half_tiles:
                descriptor_row = Int32(1)
        if combined_expert >= Int32(0) and combined_expert < total_experts:
            descriptor = descriptor_map[
                descriptor_row * total_experts + combined_expert
            ].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(_TIER_DESCRIPTOR_BITS)
                local_expert = descriptor & Int32(_TIER_DESCRIPTOR_MASK)

                if cutlass.const_expr(is_fc1):
                    t0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t0_in_bounds = local_expert < tier0_up_experts
                else:
                    t0_in_bounds = local_expert < tier0_fc2_experts
                if tier == Int32(0) and t0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t1_in_bounds = local_expert < tier1_up_experts
                else:
                    t1_in_bounds = local_expert < tier1_fc2_experts
                if tier == Int32(1) and t1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t1_b_flat,
                        c_flat,
                        t1_scales_flat,
                        t1_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t2_in_bounds = local_expert < tier2_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t2_in_bounds = local_expert < tier2_up_experts
                else:
                    t2_in_bounds = local_expert < tier2_fc2_experts
                if tier == Int32(2) and t2_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier2.fc1
                    else:
                        gemm = self.tier2.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t2_b_flat,
                        c_flat,
                        t2_scales_flat,
                        t2_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )
                if cutlass.const_expr(is_fc1):
                    t3_in_bounds = local_expert < tier3_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t3_in_bounds = local_expert < tier3_up_experts
                else:
                    t3_in_bounds = local_expert < tier3_fc2_experts
                if tier == Int32(3) and t3_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier3.fc1
                    else:
                        gemm = self.tier3.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t3_b_flat,
                        c_flat,
                        t3_scales_flat,
                        t3_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

    @cute.jit
    def __call__(
        self,
        rotation_input_ptr: cute.Pointer,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13_ptr: cute.Pointer,
        t0_w2_ptr: cute.Pointer,
        t0_w13_scales_ptr: cute.Pointer,
        t0_w2_scales_ptr: cute.Pointer,
        t0_w13_global_ptr: cute.Pointer,
        t0_w2_global_ptr: cute.Pointer,
        t1_w13_ptr: cute.Pointer,
        t1_w2_ptr: cute.Pointer,
        t1_w13_scales_ptr: cute.Pointer,
        t1_w2_scales_ptr: cute.Pointer,
        t1_w13_global_ptr: cute.Pointer,
        t1_w2_global_ptr: cute.Pointer,
        t2_w13_ptr: cute.Pointer,
        t2_w2_ptr: cute.Pointer,
        t2_w13_scales_ptr: cute.Pointer,
        t2_w2_scales_ptr: cute.Pointer,
        t2_w13_global_ptr: cute.Pointer,
        t2_w2_global_ptr: cute.Pointer,
        t3_w13_ptr: cute.Pointer,
        t3_w2_ptr: cute.Pointer,
        t3_w13_scales_ptr: cute.Pointer,
        t3_w2_scales_ptr: cute.Pointer,
        t3_w13_global_ptr: cute.Pointer,
        t3_w2_global_ptr: cute.Pointer,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        trellis_lut_ptr: cute.Pointer,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier2_num_experts: cutlass.Int32,
        tier3_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        tier3_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier3_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
        tier3_up_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        tier2_experts = cutlass.Int64(tier2_num_experts)
        tier3_experts = cutlass.Int64(tier3_num_experts)
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        tier2_fc2 = cutlass.Int64(tier2_fc2_experts)
        tier3_fc2 = cutlass.Int64(tier3_fc2_experts)
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        tier2_gate = cutlass.Int64(tier2_gate_experts)
        tier3_gate = cutlass.Int64(tier3_gate_experts)
        total_experts = tier0_experts + tier1_experts + tier2_experts + tier3_experts

        def weight_tensor(ptr, elements):
            return cute.make_tensor(
                ptr,
                layout=cute.make_layout((elements,), stride=(1,)),
            )

        t0_w13 = weight_tensor(
            t0_w13_ptr,
            tier0_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t0_w2 = weight_tensor(
            t0_w2_ptr,
            tier0_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t1_w13 = weight_tensor(
            t1_w13_ptr,
            tier1_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t1_w2 = weight_tensor(
            t1_w2_ptr,
            tier1_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t2_w13 = weight_tensor(
            t2_w13_ptr,
            tier2_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )
        t3_w13 = weight_tensor(
            t3_w13_ptr,
            tier3_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier3.trellis_bits),
        )
        t2_w2 = weight_tensor(
            t2_w2_ptr,
            tier2_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )
        t3_w2 = weight_tensor(
            t3_w2_ptr,
            tier3_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier3.trellis_bits),
        )

        t0_w13_scales = weight_tensor(
            t0_w13_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc1.scale_k_groups)
            * cutlass.Int64(self.tier0.fc1.scale_size_n // 4),
        )
        t0_w2_scales = weight_tensor(
            t0_w2_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc2.scale_k_groups)
            * cutlass.Int64(self.tier0.fc2.scale_size_n // 4),
        )
        t1_w13_scales = weight_tensor(
            t1_w13_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc1.scale_k_groups)
            * cutlass.Int64(self.tier1.fc1.scale_size_n // 4),
        )
        t1_w2_scales = weight_tensor(
            t1_w2_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc2.scale_k_groups)
            * cutlass.Int64(self.tier1.fc2.scale_size_n // 4),
        )
        t2_w13_scales = weight_tensor(
            t2_w13_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc1.scale_k_groups)
            * cutlass.Int64(self.tier2.fc1.scale_size_n // 4),
        )
        t3_w13_scales = weight_tensor(
            t3_w13_scales_ptr,
            tier3_experts
            * cutlass.Int64(self.tier3.fc1.scale_k_groups)
            * cutlass.Int64(self.tier3.fc1.scale_size_n // 4),
        )
        t2_w2_scales = weight_tensor(
            t2_w2_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc2.scale_k_groups)
            * cutlass.Int64(self.tier2.fc2.scale_size_n // 4),
        )
        t3_w2_scales = weight_tensor(
            t3_w2_scales_ptr,
            tier3_experts
            * cutlass.Int64(self.tier3.fc2.scale_k_groups)
            * cutlass.Int64(self.tier3.fc2.scale_size_n // 4),
        )
        t0_w13_global = weight_tensor(t0_w13_global_ptr, tier0_experts)
        t0_w2_global = weight_tensor(t0_w2_global_ptr, tier0_fc2)
        t1_w13_global = weight_tensor(t1_w13_global_ptr, tier1_experts)
        t1_w2_global = weight_tensor(t1_w2_global_ptr, tier1_fc2)
        t2_w13_global = weight_tensor(t2_w13_global_ptr, tier2_experts)
        t2_w2_global = weight_tensor(t2_w2_global_ptr, tier2_fc2)
        t3_w13_global = weight_tensor(t3_w13_global_ptr, tier3_experts)
        t3_w2_global = weight_tensor(t3_w2_global_ptr, tier3_fc2)
        descriptor_map = weight_tensor(
            descriptor_map_ptr, cutlass.Int64(3) * total_experts
        )
        intermediate_rotations = weight_tensor(
            intermediate_rotations_ptr,
            total_experts * cutlass.Int64(3 * self.intermediate_size),
        )
        suh_rows = total_experts
        if cutlass.const_expr(self.driver.broadcast_suh):
            suh_rows = cutlass.Int64(1)
        gate_suh = weight_tensor(
            gate_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size)
        )
        up_suh = weight_tensor(up_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size))
        trellis_lut = weight_tensor(
            trellis_lut_ptr, Int64(_SQG_XOR_CHEB_T12_LUT_ENTRIES)
        )
        trellis_lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        rotation_input = weight_tensor(
            rotation_input_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),
        )
        topk_weights = weight_tensor(
            topk_weights_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),
        )
        self.kernel4(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            t2_w13,
            t2_w2,
            t2_w13_scales,
            t2_w2_scales,
            t2_w13_global,
            t2_w2_global,
            t3_w13,
            t3_w2,
            t3_w13_scales,
            t3_w2_scales,
            t3_w13_global,
            t3_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            trellis_lut_addr,
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier3_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier3_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier3_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
            tier3_up_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel4(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        t2_w13: cute.Tensor,
        t2_w2: cute.Tensor,
        t2_w13_scales: cute.Tensor,
        t2_w2_scales: cute.Tensor,
        t2_w13_global: cute.Tensor,
        t2_w2_global: cute.Tensor,
        t3_w13: cute.Tensor,
        t3_w2: cute.Tensor,
        t3_w13_scales: cute.Tensor,
        t3_w2_scales: cute.Tensor,
        t3_w13_global: cute.Tensor,
        t3_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut_addr: Int64,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier2_num_experts: cutlass.Int32,
        tier3_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        tier3_fc2_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier3_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
        tier3_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        phase_lut_addr = trellis_lut_addr
        if cutlass.const_expr(self.driver.sqg_xor_cheb_t12_smem):
            self.driver._sqg_smem_copy(
                trellis_lut_addr,
                smem_base + Int32(self.sqg_xor_cheb_t12_smem_off),
                _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
                tid,
            )
            cute.arch.sync_threads()
            phase_lut_addr = Int64(smem_base + Int32(self.sqg_xor_cheb_t12_smem_off))
        common = (
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
        )
        counts = (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier3_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier3_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier3_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
            tier3_up_experts,
        )
        fc1_emit = partial(
            self._emit_tier_tile4,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            t2_w13,
            t2_w13_scales,
            t2_w13_global,
            t3_w13,
            t3_w13_scales,
            t3_w13_global,
            fc1,
            *common,
            fc1_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m,
            *counts,
        )
        fc2_emit = partial(
            self._emit_tier_tile4,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            t2_w2,
            t2_w2_scales,
            t2_w2_global,
            t3_w2,
            t3_w2_scales,
            t3_w2_global,
            fc2,
            *common,
            fc2_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
            *counts,
        )
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts + tier3_num_experts
        self.driver._moe_body(
            rotation_gate,
            rotation_up,
            rotation_input,
            t0_w13,
            t0_w2,
            fc1,
            activated,
            fc2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            t0_w13_global,
            Int32(0),
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            descriptor_map,
            trellis_lut_addr,
            trellis_lut_addr,
            total_experts,
            total_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit,
            fc2_emit,
        )


_CACHE4: dict[tuple[object, ...], MixedTrellis4CompileResult] = {}


def compile_mixed_trellis4(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    tier2_num_experts: int,
    tier3_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 2,
    tier1_bits: int = 3,
    tier2_bits: int = 4,
    tier3_bits: int = 5,
    trellis_codebook: str = "mcg",
    swiglu_limit: float | None = None,
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    full_rotation_output_dtype: str = "fp32",
    route_ids_dtype: torch.dtype = torch.int32,
    broadcast_suh: bool = False,
    broadcast_svh: bool = False,
    route_num_experts: int | None = None,
    workspace_sms: int | None = None,
) -> MixedTrellis4CompileResult:
    """Compile the dedicated four-bitrate cooperative Trellis grid."""

    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    trellis_codebook, bits = _normalize_mixed_trellis_format(
        trellis_codebook,
        (tier0_bits, tier1_bits, tier2_bits, tier3_bits),
    )
    counts = tuple(
        int(value)
        for value in (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier3_num_experts,
        )
    )
    if any(value <= 0 or value > _MAX_TIER_EXPERTS for value in counts):
        raise ValueError(
            "four-tier mixed Trellis requires each tier to contain 1..512 slots"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    # The whole-tile scheduler gives one CTA the complete K reduction, so K64
    # does not use the retired cross-CTA partial-reduction path.
    total_experts = sum(counts)
    if route_num_experts is None:
        route_num_experts = total_experts
    route_num_experts = int(route_num_experts)
    if route_num_experts <= 0:
        raise ValueError("mixed Trellis route_num_experts must be positive")
    def make_kernel(
        num_experts: int,
        trellis_bits: int,
        *,
        grouped_m8_fc2: bool,
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation="silu",
            swiglu_limit=swiglu_limit,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            intermediate_rotation=True,
            full_rotation=True,
            workspace_sms=workspace_sms,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=broadcast_suh,
            schedule_whole_tiles=True,
        )

    def build_kernel(grouped_m8_fc2: bool) -> W4A16MixedTrellis4Kernel:
        common = {"grouped_m8_fc2": grouped_m8_fc2}
        return W4A16MixedTrellis4Kernel(
            driver=make_kernel(total_experts, bits[0], **common),
            tier0=make_kernel(counts[0], bits[0], **common),
            tier1=make_kernel(counts[1], bits[1], **common),
            tier2=make_kernel(counts[2], bits[2], **common),
            tier3=make_kernel(counts[3], bits[3], **common),
        )

    kernel = _select_mixed_fc2_kernel(
        build_kernel,
        moe_block_size=moe_block_size,
        max_shared_mem=max_shared_mem,
    )
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis4",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
    )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        full_rotation_output_dtype=full_rotation_output_dtype,
        num_experts=total_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
        broadcast_svh=broadcast_svh,
    )
    cached = _CACHE4.get(cache_key)
    if cached is not None:
        return replace(
            cached,
            topk_sum=topk_sum,
            tier0_num_experts=counts[0],
            tier1_num_experts=counts[1],
            tier2_num_experts=counts[2],
            tier3_num_experts=counts[3],
            sms=int(sms),
            broadcast_suh=bool(broadcast_suh),
            broadcast_svh=bool(broadcast_svh),
        )

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = _cutlass_element_dtype(rotation_input_dtype)

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args():
        return (
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(),
        *tier_args(),
        *tier_args(),
        *tier_args(),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, moe_block_size),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[3]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[3]),
        1,
        1,
        current_cuda_stream(),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[3]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[3]),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis4",
            W4A16MixedTrellis4Kernel.ABI_VERSION,
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = MixedTrellis4CompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        trellis_lut=_trellis256_execution_lut(
            torch.device("cuda", device), trellis_codebook
        ),
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=counts[0],
        tier1_num_experts=counts[1],
        tier2_num_experts=counts[2],
        tier3_num_experts=counts[3],
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        tier2_bits=bits[2],
        tier3_bits=bits[3],
        trellis_codebook=trellis_codebook,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
        direct_topk_routes=False,
        broadcast_suh=bool(broadcast_suh),
        broadcast_svh=bool(broadcast_svh),
        workspace_sms=workspace_sms,
    )
    _CACHE4[cache_key] = result
    return result
