"""Warp-specialized mixed-Trellis prefill (SM12x, packed routes).

The cooperative mixed-Trellis kernel runs every stage of a tile in the same
eight warps: cp.async staging, the FC1 input rotation, Trellis decode and the
MMAs, separated by CTA-wide barriers. On GB10 (one 256-thread CTA per SM at
~250 registers) that leaves the SM waiting on memory with the tensor pipe
mostly idle.

This variant splits each CTA into roles (one CTA per SM, persistent):

``consumer`` warps (``tile_n / 32``: two warpgroups for 256-wide tiles)
    own a 32-column slice of the ``route_block x tile_n`` output tile. Per
    128-wide K block they wait once for the block's input rows and the two
    compressed 64-wide weight units, then run eight straight-line K16 steps:
    window loads, t256 decode of their two N16 fragments, ``ldmatrix`` of the
    rows, ``mma.m16n8k16``. They hold the tile's FP32 accumulators and store
    the FP16 result directly.
``producer`` warpgroup (4 warps, 88 registers)
    streams the compressed weights with bulk copies (thread 0) into an up to
    8-deep ring, gathers each block's input rows with ``cp.async``, applies
    the FC1 input rotation (``x * suh``, H128) one block ahead of the
    consumers, and publishes each tile's metadata (valid rows, route ids,
    tier, global scale) through a small shared ring so consumers never wait
    on dependent global loads between tiles.

Rings are mbarrier-synchronized and cross tile boundaries. FC1, the
SwiGLU/intermediate-rotation pass and FC2 are three launches of one compiled
entry point with the cooperative kernel's ABI; the prefill capacities spend
milliseconds per layer, so its grid barriers buy nothing here.

Decoding in dedicated producer warps (weights decoded to FP16 MMA fragments in
a shared ring) was measured first and lost on GB10: one decode warp per SMSP
is latency bound, and even two leave the consumers idle while the decode
instructions still issue on the same schedulers.

Arithmetic is bit-identical to the cooperative kernel with (64, 256, 64, 256)
tiles: the same decoded fragments, rotated rows and MMA instructions, and each
output element accumulates its K16 steps in the same two chains (K16 index
mod 4 in {0, 1} and {2, 3}: the cooperative kernel's K-split warp rows),
summed once before the global scale and FP16 rounding.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    cp_async4_shared_global_pred,
    cp_async_bulk_g2s_mbar,
    cp_async_bulk_g2s_mbar_l2hint,
    create_l2_evict_first_policy,
    cvt_bf16x2_to_f16x2,
    f16_mma_m16n8k16_f32,
    f16x2_to_f32x2,
    get_ptr_as_int64,
    half2_mul,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    pack_f32x2_to_f16x2,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_f32,
    st_shared_i32,
    st_shared_v4_u32,
    trellis_align_stream_u32x2,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits

from .kernel import _SQG_XOR_CHEB_T12_LUT_ENTRIES
from .mixed_trellis import (
    W4A16MixedTrellisKernel,
    _TIER_DESCRIPTOR_BITS,
    _TIER_DESCRIPTOR_MASK,
)

_PRODUCER_WARPS = 4
_PRODUCER_THREADS = 32 * _PRODUCER_WARPS
# setmaxnreg split for the 384-thread layout (two consumer warpgroups):
# 128 * 88 + 256 * 208 = 64512 <= 65536 registers.
_PRODUCER_REGS = 88
_CONSUMER_REGS = 208
_MAX_BC_STAGES = 8
_TILE_STAGES = 4
_TILE_HEADER_WORDS = 8
_BARRIER_BYTES = 8


# --------------------------------------------------------------------------
# mbarrier primitives on 32-bit shared addresses
# --------------------------------------------------------------------------


@dsl_user_op
def _mbar_init(addr: Int32, count: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.mbarrier_init.release.cluster;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive(addr: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive_expect_tx(addr: Int32, nbytes: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_wait(addr: Int32, parity: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; WS_WAIT: "
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; "
        "@!p bra WS_WAIT; }",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_mbar_arrive_noinc(addr: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class W4A16MixedTrellisWSKernel(W4A16MixedTrellisKernel):
    """Warp-specialized two-tier mixed Trellis for packed-route prefill."""

    WS_ABI_VERSION = 2

    def __init__(self, *, driver, tier0, tier1, max_shared_mem: int):
        super().__init__(driver=driver, tier0=tier0, tier1=tier1)
        d = driver
        if d.direct_topk_routes:
            raise ValueError("warp-specialized mixed Trellis requires packed routes")
        if d.broadcast_suh or d.coupled_hadamard or d.rotation_input_dtype != "bf16":
            raise ValueError(
                "warp-specialized mixed Trellis requires per-expert SUH, "
                "H128 rotation and BF16 input rows"
            )
        if d.sqg_xor_cheb_t12_smem:
            raise ValueError("warp-specialized mixed Trellis reads the T12 table from global memory")
        block = int(d.moe_block_size)
        if block not in (16, 32, 64):
            raise ValueError("warp-specialized mixed Trellis needs 16, 32 or 64-row route blocks")
        if int(d.fc2.moe_block_size) != block or int(d.fc2.schedule_route_block_factor) != 1:
            raise ValueError("warp-specialized FC2 shares the FC1 route block")
        tile_n = int(d.fc1.tile_n)
        if tile_n not in (128, 256) or int(d.fc2.tile_n) != tile_n:
            raise ValueError("warp-specialized mixed Trellis needs equal 128/256-wide N tiles")
        if int(d.fc1.tile_k) != 64 or int(d.fc2.tile_k) != 64:
            # The accumulation chains reproduce the cooperative K64 tile's
            # two K-split warp rows.
            raise ValueError("warp-specialized mixed Trellis reproduces K64 tiles only")
        for moe in (tier0, tier1):
            if moe.fc1.weight_layout_trellis256_pair or moe.fc2.weight_layout_trellis256_pair:
                raise ValueError("warp-specialized mixed Trellis does not decode QSRT pairs")
            if not moe.fc1.weight_layout_trellis256_proj:
                raise ValueError("warp-specialized FC1 requires projection-major W13")
        hidden = int(self.hidden_size)
        inter = int(self.intermediate_size)
        if hidden % 128 or inter % 128:
            raise ValueError("warp-specialized mixed Trellis needs H128-aligned K")
        if inter % tile_n or hidden % tile_n:
            raise ValueError("gate/up halves and FC2 must hold whole N tiles")
        self.ws_m = block
        self.ws_mb = block // 16
        self.ws_n = tile_n
        self.ws_n16 = tile_n // 16
        self.ws_consumer_warps = tile_n // 32
        self.ws_consumer_threads = tile_n
        self.ws_threads = tile_n + _PRODUCER_THREADS
        self.ws_setmaxnreg = self.ws_threads > 256
        self.ws_bits = (int(tier0.trellis_bits), int(tier1.trellis_bits))
        self.ws_max_bits = max(self.ws_bits)
        # Shared layout (bytes): A ring (block rows x 128 K, rows at 256-byte
        # stride), compressed-weight ring (four K16 rows x N at the widest
        # tier), tile-metadata ring, mbarriers.
        self.ws_a_slot = block * 256
        self.ws_bc_slot = 4 * self.ws_n16 * 32 * self.ws_max_bits
        self.ws_tile_slot = 4 * (_TILE_HEADER_WORDS + block)
        fixed = _TILE_STAGES * (self.ws_tile_slot + 2 * _BARRIER_BYTES)
        budget = int(max_shared_mem) - 1024 - fixed
        a_stages = 2
        bc_stages = 0
        for candidate_a in (3, 2):
            rest = budget - candidate_a * (self.ws_a_slot + 3 * _BARRIER_BYTES)
            candidate_bc = (
                min(_MAX_BC_STAGES, rest // (self.ws_bc_slot + 2 * _BARRIER_BYTES))
                if rest > 0
                else 0
            )
            if candidate_bc >= 4 or (candidate_a == 2 and candidate_bc >= 2):
                a_stages, bc_stages = candidate_a, candidate_bc
                break
        if bc_stages < 2:
            raise ValueError(
                "warp-specialized mixed Trellis rings exceed shared memory: "
                f"block={block} tile_n={tile_n} bits={self.ws_bits}"
            )
        self.ws_a_stages = a_stages
        self.ws_bc_stages = bc_stages
        self.ws_a_off = 0
        self.ws_bc_off = self.ws_a_off + a_stages * self.ws_a_slot
        self.ws_tile_off = self.ws_bc_off + bc_stages * self.ws_bc_slot
        bar = self.ws_tile_off + _TILE_STAGES * self.ws_tile_slot
        self.ws_bar_a_raw = bar
        self.ws_bar_a_full = self.ws_bar_a_raw + a_stages * _BARRIER_BYTES
        self.ws_bar_a_empty = self.ws_bar_a_full + a_stages * _BARRIER_BYTES
        self.ws_bar_bc_full = self.ws_bar_a_empty + a_stages * _BARRIER_BYTES
        self.ws_bar_bc_empty = self.ws_bar_bc_full + bc_stages * _BARRIER_BYTES
        self.ws_bar_t_full = self.ws_bar_bc_empty + bc_stages * _BARRIER_BYTES
        self.ws_bar_t_empty = self.ws_bar_t_full + _TILE_STAGES * _BARRIER_BYTES
        self.ws_smem_bytes = self.ws_bar_t_empty + _TILE_STAGES * _BARRIER_BYTES
        if self.ws_smem_bytes > int(max_shared_mem):
            raise ValueError("warp-specialized mixed Trellis shared layout exceeds the device limit")
        self.blocks_per_sm = 1
        self.shared_words = (self.ws_smem_bytes + 3) // 4
        self.ws_act_ctas_per_sm = 8
        # Timing probes only (wrong numerics): nodecode, nomma, norot.
        self.ws_exp = frozenset(filter(None, os.environ.get("B12X_WS_EXP", "").split(",")))

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis_ws",
            self.WS_ABI_VERSION,
            super().__cache_key__,
            self.ws_a_stages,
            self.ws_bc_stages,
            self.ws_threads,
            tuple(sorted(self.ws_exp)),
        )

    # ------------------------------------------------------------------
    # Host entry: same ABI as the cooperative mixed kernel.
    # ------------------------------------------------------------------

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
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
        global_to_combined_ptr: cute.Pointer,
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
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        total_experts = tier0_experts + tier1_experts
        h16 = cutlass.Int64(self.hidden_size // 16)
        fc1_n16 = cutlass.Int64(self.driver.fc1_cols // 16)
        i16 = cutlass.Int64(self.intermediate_size // 16)
        b0 = cutlass.Int64(8 * self.tier0.trellis_bits)
        b1 = cutlass.Int64(8 * self.tier1.trellis_bits)

        def flat(ptr, n):
            return cute.make_tensor(ptr, layout=cute.make_layout((n,), stride=(1,)))

        # Weight extents match the cooperative kernel's views: the W13 plane
        # is sized by the gate count so the up plane starts at its end.
        t0_w13 = flat(t0_w13_ptr, tier0_gate * h16 * fc1_n16 * b0)
        t1_w13 = flat(t1_w13_ptr, tier1_gate * h16 * fc1_n16 * b1)
        t0_w2 = flat(t0_w2_ptr, tier0_fc2 * i16 * h16 * b0)
        t1_w2 = flat(t1_w2_ptr, tier1_fc2 * i16 * h16 * b1)
        t0_w13_global = flat(t0_w13_global_ptr, tier0_experts)
        t1_w13_global = flat(t1_w13_global_ptr, tier1_experts)
        t0_w2_global = flat(t0_w2_global_ptr, tier0_fc2)
        t1_w2_global = flat(t1_w2_global_ptr, tier1_fc2)
        descriptor_map = flat(descriptor_map_ptr, cutlass.Int64(3) * total_experts)
        intermediate_rotations = flat(
            intermediate_rotations_ptr,
            total_experts * cutlass.Int64(3 * self.intermediate_size),
        )
        gate_suh = flat(gate_suh_ptr, total_experts * cutlass.Int64(self.hidden_size))
        up_suh = flat(up_suh_ptr, total_experts * cutlass.Int64(self.hidden_size))
        trellis_lut = flat(trellis_lut_ptr, cutlass.Int64(_SQG_XOR_CHEB_T12_LUT_ENTRIES))
        rotation_input = flat(
            rotation_input_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),
        )

        self.ws_fc1_kernel(
            rotation_input,
            t0_w13,
            t1_w13,
            t0_w13_global,
            t1_w13_global,
            fc1,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            gate_suh,
            up_suh,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.ws_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )
        self.ws_act_kernel(
            fc1,
            activated,
            intermediate_rotations,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            active_m,
        ).launch(
            grid=(grid_x * Int32(self.ws_act_ctas_per_sm), 1, 1),
            block=[self.driver.cta_threads, 1, 1],
            stream=stream,
        )
        self.ws_fc2_kernel(
            activated,
            t0_w2,
            t1_w2,
            t0_w2_global,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.ws_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------

    @cute.kernel
    def ws_act_kernel(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        active_m: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        total = tier0_num_experts + tier1_num_experts
        # The cooperative kernel's activation phase, unchanged (it strides by
        # the driver's 256-thread CTA).
        self.driver._run_activation_compact(
            fc1,
            activated,
            intermediate_rotations,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            total,
            total,
            Int32(tidx),
            Int32(bidx),
            Int32(gdim),
            active_m,
        )

    @cute.kernel
    def ws_fc1_kernel(
        self,
        a_src: cute.Tensor,
        t0_w: cute.Tensor,
        t1_w: cute.Tensor,
        t0_global: cute.Tensor,
        t1_global: cute.Tensor,
        c_out: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        active_m: Int32,
    ):
        self._ws_gemm(
            True,
            a_src,
            t0_w,
            t1_w,
            t0_global,
            t1_global,
            c_out,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            gate_suh,
            up_suh,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
        )

    @cute.kernel
    def ws_fc2_kernel(
        self,
        a_src: cute.Tensor,
        t0_w: cute.Tensor,
        t1_w: cute.Tensor,
        t0_global: cute.Tensor,
        t1_global: cute.Tensor,
        c_out: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        trellis_lut: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        active_m: Int32,
    ):
        self._ws_gemm(
            False,
            a_src,
            t0_w,
            t1_w,
            t0_global,
            t1_global,
            c_out,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            a_src,
            a_src,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            active_m,
        )

    # ------------------------------------------------------------------
    # Shared GEMM body
    # ------------------------------------------------------------------

    @cute.jit
    def _ws_gemm(
        self,
        is_fc1: cutlass.Constexpr,
        a_src: cute.Tensor,
        t0_w: cute.Tensor,
        t1_w: cute.Tensor,
        t0_global: cute.Tensor,
        t1_global: cute.Tensor,
        c_out: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        # FC1: gate/up slot counts; FC2: the down counts twice.
        tier0_lo_experts: Int32,
        tier1_lo_experts: Int32,
        tier0_hi_experts: Int32,
        tier1_hi_experts: Int32,
        active_m: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        base = shared_ptr_to_u32(storage.words.data_ptr())

        if tid == Int32(0):
            for s in cutlass.range_constexpr(self.ws_a_stages):
                _mbar_init(base + Int32(self.ws_bar_a_raw + 8 * s), Int32(_PRODUCER_THREADS))
                _mbar_init(base + Int32(self.ws_bar_a_full + 8 * s), Int32(_PRODUCER_THREADS))
                _mbar_init(
                    base + Int32(self.ws_bar_a_empty + 8 * s), Int32(self.ws_consumer_threads)
                )
            for s in cutlass.range_constexpr(self.ws_bc_stages):
                _mbar_init(base + Int32(self.ws_bar_bc_full + 8 * s), Int32(1))
                _mbar_init(
                    base + Int32(self.ws_bar_bc_empty + 8 * s), Int32(self.ws_consumer_threads)
                )
            for s in cutlass.range_constexpr(_TILE_STAGES):
                _mbar_init(base + Int32(self.ws_bar_t_full + 8 * s), Int32(_PRODUCER_THREADS))
                _mbar_init(
                    base + Int32(self.ws_bar_t_empty + 8 * s), Int32(self.ws_consumer_threads)
                )
            _fence_mbar_init()
        cute.arch.sync_threads()

        if cutlass.const_expr(is_fc1):
            k_size = self.hidden_size
            n_total = self.driver.fc1_cols
        else:
            k_size = self.intermediate_size
            n_total = self.hidden_size
        lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        if tid >= Int32(self.ws_consumer_threads):
            if cutlass.const_expr(self.ws_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(_PRODUCER_REGS)
            self._ws_producer(
                is_fc1,
                k_size,
                n_total,
                base,
                tid - Int32(self.ws_consumer_threads),
                Int32(bidx),
                Int32(gdim),
                active_m,
                a_src,
                t0_w,
                t1_w,
                t0_global,
                t1_global,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                descriptor_map,
                gate_suh,
                up_suh,
                tier0_num_experts,
                tier1_num_experts,
                tier0_lo_experts,
                tier1_lo_experts,
                tier0_hi_experts,
                tier1_hi_experts,
            )
        else:
            if cutlass.const_expr(self.ws_setmaxnreg):
                cute.arch.warpgroup_reg_alloc(_CONSUMER_REGS)
            self._ws_consumer(is_fc1, k_size, n_total, base, lut_addr, tid, c_out)

    # ------------------------------------------------------------------
    # Tile schedule (producer side)
    # ------------------------------------------------------------------

    @cute.jit
    def _ws_tile_meta(
        self,
        is_fc1: cutlass.Constexpr,
        t: Int32,
        n_tiles: Int32,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_lo_experts: Int32,
        tier1_lo_experts: Int32,
        tier0_hi_experts: Int32,
        tier1_hi_experts: Int32,
    ):
        """(route block, N tile, combined expert, tier or -1, tier-local expert):
        the cooperative kernel's _emit_tier_tile resolution."""
        rb = t // n_tiles
        nt = t - rb * n_tiles
        combined = block_expert_ids[rb].to(Int32)
        total = tier0_num_experts + tier1_num_experts
        tier = Int32(-1)
        local = Int32(0)
        row = Int32(2)
        bound0 = tier0_lo_experts
        bound1 = tier1_lo_experts
        if cutlass.const_expr(is_fc1):
            row = Int32(0)
            if nt >= n_tiles // Int32(2):
                row = Int32(1)
                bound0 = tier0_hi_experts
                bound1 = tier1_hi_experts
        if combined >= Int32(0) and combined < total:
            descriptor = descriptor_map[row * total + combined].to(Int32)
            if descriptor >= Int32(0):
                d_tier = descriptor >> Int32(_TIER_DESCRIPTOR_BITS)
                d_local = descriptor & Int32(_TIER_DESCRIPTOR_MASK)
                if d_tier == Int32(0) and d_local < bound0:
                    tier = Int32(0)
                    local = d_local
                if d_tier == Int32(1) and d_local < bound1:
                    tier = Int32(1)
                    local = d_local
        return rb, nt, combined, tier, local

    @cute.jit
    def _ws_next_tile(
        self,
        is_fc1: cutlass.Constexpr,
        t: Int32,
        grid: Int32,
        total_tiles: Int32,
        n_tiles: Int32,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_lo_experts: Int32,
        tier1_lo_experts: Int32,
        tier0_hi_experts: Int32,
        tier1_hi_experts: Int32,
    ):
        """First tile >= t (stride grid) that has weights; total_tiles if none."""
        found = Int32(0)
        rb = Int32(0)
        nt = Int32(0)
        combined = Int32(0)
        tier = Int32(-1)
        local = Int32(0)
        while t < total_tiles and found == Int32(0):
            rb, nt, combined, tier, local = self._ws_tile_meta(
                is_fc1,
                t,
                n_tiles,
                block_expert_ids,
                descriptor_map,
                tier0_num_experts,
                tier1_num_experts,
                tier0_lo_experts,
                tier1_lo_experts,
                tier0_hi_experts,
                tier1_hi_experts,
            )
            if tier >= Int32(0):
                found = Int32(1)
            else:
                t += grid
        if found == Int32(0):
            t = total_tiles
        return t, rb, nt, combined, tier, local

    @cute.jit
    def _ws_valid_rows(self, packed_route_indices: cute.Tensor, rb: Int32, live: Int32, lane: Int32):
        count = Int32(0)
        for i in cutlass.range_constexpr((self.ws_m + 31) // 32):
            r = lane + Int32(32 * i)
            if r < Int32(self.ws_m):
                idx = packed_route_indices[rb * Int32(self.ws_m) + r].to(Int32)
                if idx < live:
                    count += Int32(1)
        return cute.arch.warp_redux_sync(count, "add")

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    @cute.jit
    def _ws_weight_row(
        self,
        is_fc1: cutlass.Constexpr,
        n_total: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        nt: Int32,
        n_tiles: Int32,
        local: Int32,
        tier: Int32,
        t0_w: cute.Tensor,
        t1_w: cute.Tensor,
        tier0_lo_experts: Int32,
        tier1_lo_experts: Int32,
    ):
        """(byte address of the tile's first K16 weight row, K16 row stride in
        bytes, bytes of one tile row). The tile's N16 records of one K16 row
        are contiguous in both Trellis layouts."""
        bits = Int32(self.ws_bits[0])
        wbase = get_ptr_as_int64(t0_w, Int32(0))
        gate_count = tier0_lo_experts
        if tier == Int32(1):
            bits = Int32(self.ws_bits[1])
            wbase = get_ptr_as_int64(t1_w, Int32(0))
            gate_count = tier1_lo_experts
        tile_u32 = Int64(8) * bits.to(Int64)
        row_bytes = Int32(self.ws_n16 * 32) * bits
        if cutlass.const_expr(is_fc1):
            # Projection-major W13: [gate plane | up plane], each [E, K16, N16/2].
            half_n16 = Int64(n_total // 32)
            proj = Int64(0)
            if nt >= n_tiles // Int32(2):
                proj = Int64(1)
            local_n16 = Int64(nt) * Int64(self.ws_n16) - proj * half_n16
            proj_expert_u32 = Int64(k_size // 16) * half_n16 * tile_u32
            plane_u32 = gate_count.to(Int64) * proj_expert_u32
            first_u32 = (
                proj * plane_u32
                + Int64(local) * proj_expert_u32
                + local_n16 * tile_u32
            )
            stride_u32 = half_n16 * tile_u32
        else:
            n16_total = Int64(n_total // 16)
            first_u32 = (
                Int64(local) * Int64(k_size // 16) * n16_total * tile_u32
                + Int64(nt) * Int64(self.ws_n16) * tile_u32
            )
            stride_u32 = n16_total * tile_u32
        return wbase + first_u32 * Int64(4), stride_u32 * Int64(4), row_bytes

    @cute.jit
    def _ws_issue_bc(
        self,
        base: Int32,
        slot: Int32,
        unit: Int32,
        wrow0: Int64,
        wstride: Int64,
        row_bytes: Int32,
        last_reader: Int32,
    ):
        """One thread: four K16 rows (one 64-wide K unit) of a tile's weights.
        The last route block of an expert streams them L2 evict-first: no
        later block rereads them, and the gathered input rows stay cached."""
        bar = base + Int32(self.ws_bar_bc_full) + slot * Int32(_BARRIER_BYTES)
        dst = base + Int32(self.ws_bc_off) + slot * Int32(self.ws_bc_slot)
        _mbar_arrive_expect_tx(bar, row_bytes * Int32(4))
        if last_reader != Int32(0):
            policy = create_l2_evict_first_policy()
            for q in cutlass.range_constexpr(4):
                k16 = unit * Int32(4) + Int32(q)
                cp_async_bulk_g2s_mbar_l2hint(
                    dst + Int32(q) * row_bytes,
                    wrow0 + Int64(k16) * wstride,
                    row_bytes,
                    bar,
                    policy,
                )
        else:
            for q in cutlass.range_constexpr(4):
                k16 = unit * Int32(4) + Int32(q)
                cp_async_bulk_g2s_mbar(
                    dst + Int32(q) * row_bytes,
                    wrow0 + Int64(k16) * wstride,
                    row_bytes,
                    bar,
                )

    @cute.jit
    def _ws_last_block(self, block_expert_ids: cute.Tensor, rb: Int32, route_blocks: Int32):
        """1 when route block rb is its expert's last (no later block rereads
        the expert's weights)."""
        last = Int32(1)
        if rb + Int32(1) < route_blocks:
            if block_expert_ids[rb + Int32(1)].to(Int32) == block_expert_ids[rb].to(Int32):
                last = Int32(0)
        return last

    @cute.jit
    def _ws_load_rows(
        self,
        is_fc1: cutlass.Constexpr,
        rows: cute.Tensor,
        packed_route_indices: cute.Tensor,
        rb: Int32,
        ptid: Int32,
        live: Int32,
    ):
        """Source row ids of the rows this producer thread copies (rows
        ptid/16 + 8i): tokens for FC1, routes for FC2, -1 for padding."""
        for i in cutlass.range_constexpr(self.ws_m // 8):
            r = (ptid >> Int32(4)) + Int32(8 * i)
            idx = packed_route_indices[rb * Int32(self.ws_m) + r].to(Int32)
            row_id = Int32(-1)
            if idx < live:
                row_id = idx
                if cutlass.const_expr(is_fc1):
                    row_id = idx // Int32(self.top_k)
            rows[i] = row_id

    @cute.jit
    def _ws_issue_a(
        self,
        k_size: cutlass.Constexpr,
        a_src: cute.Tensor,
        rows: cute.Tensor,
        base: Int32,
        slot: Int32,
        kblock: Int32,
        ptid: Int32,
        bar: Int32,
    ):
        """cp.async one 128-wide K block of the tile's rows into an A slot
        (row r at r * 256 bytes; 16-byte chunk c at (c & 8) | ((c ^ r) & 7)),
        then arrive on ``bar`` when this thread's copies land."""
        c = ptid & Int32(15)
        a_slot = base + Int32(self.ws_a_off) + slot * Int32(self.ws_a_slot)
        for i in cutlass.range_constexpr(self.ws_m // 8):
            r = (ptid >> Int32(4)) + Int32(8 * i)
            row_id = rows[i]
            pos = (c & Int32(8)) | ((c & Int32(7)) ^ (r & Int32(7)))
            dst = a_slot + r * Int32(256) + pos * Int32(16)
            src_elem = (
                Int64(row_id) * Int64(k_size)
                + Int64(kblock) * Int64(128)
                + Int64(c) * Int64(8)
            )
            if row_id < Int32(0):
                src_elem = Int64(0)
            cp_async4_shared_global_pred(
                dst,
                get_ptr_as_int64(a_src, src_elem),
                (row_id >= Int32(0)).to(Int32),
            )
        _cp_async_mbar_arrive_noinc(bar)

    @cute.jit
    def _ws_rotate_block(self, base: Int32, slot: Int32, valid: Int32, suh_w, ptid: Int32):
        """fp16(had128(fp16(fp16(x) * suh))) in place with the cooperative
        kernel's _rotate_a_pair arithmetic: a half-warp per row, lane l holding
        the row's elements 8l..8l+7 (one 16-byte chunk); three butterfly
        stages in registers, four across the half-warp."""
        lane = ptid & Int32(31)
        l16 = lane & Int32(15)
        half = lane >> Int32(4)
        warp = ptid >> Int32(5)
        pos_hi = l16 & Int32(8)
        pos_lo = l16 & Int32(7)
        a_slot = base + Int32(self.ws_a_off) + slot * Int32(self.ws_a_slot)
        iters = self.ws_m // (2 * _PRODUCER_WARPS)
        group = min(iters, 2)
        for g in cutlass.range_constexpr(iters // group):
            first_row = Int32(2) * (warp + Int32(_PRODUCER_WARPS * g * group))
            if first_row < valid:
                addrs = []
                loads = []
                for j in cutlass.range_constexpr(group):
                    row = Int32(2) * (warp + Int32(_PRODUCER_WARPS * (g * group + j))) + half
                    addr = (
                        a_slot
                        + row * Int32(256)
                        + (pos_hi | (pos_lo ^ (row & Int32(7)))) * Int32(16)
                    )
                    addrs.append(addr)
                    loads.append(ld_shared_v4_u32(addr))
                for j in cutlass.range_constexpr(group):
                    v = []
                    for w in cutlass.range_constexpr(4):
                        scaled = half2_mul(cvt_bf16x2_to_f16x2(loads[j][w]), suh_w[w])
                        lo, hi = f16x2_to_f32x2(scaled)
                        v.append(lo)
                        v.append(hi)
                    for bi in cutlass.range_constexpr(3):
                        b = 1 << bi
                        for e in cutlass.range_constexpr(8):
                            if cutlass.const_expr(e & b == 0):
                                lo = v[e]
                                hi = v[e | b]
                                v[e] = lo + hi
                                v[e | b] = lo - hi
                    for si in cutlass.range_constexpr(4):
                        st = 1 << si
                        sign = cutlass.Float32(1.0) - cutlass.Float32(2.0) * (
                            (l16 >> Int32(si)) & Int32(1)
                        ).to(cutlass.Float32)
                        for e in cutlass.range_constexpr(8):
                            p = cute.arch.shuffle_sync_bfly(v[e], offset=st)
                            v[e] = p + sign * v[e]
                    rs = cutlass.Float32(0.088388347648)
                    st_shared_v4_u32(
                        addrs[j],
                        pack_f32x2_to_f16x2(v[0] * rs, v[1] * rs),
                        pack_f32x2_to_f16x2(v[2] * rs, v[3] * rs),
                        pack_f32x2_to_f16x2(v[4] * rs, v[5] * rs),
                        pack_f32x2_to_f16x2(v[6] * rs, v[7] * rs),
                    )

    @cute.jit
    def _ws_publish_tile(
        self,
        base: Int32,
        ti: Int32,
        valid: Int32,
        rb: Int32,
        nt: Int32,
        tier: Int32,
        scale: cutlass.Float32,
        packed_route_indices: cute.Tensor,
        live: Int32,
        ptid: Int32,
    ):
        """Write tile ``ti``'s header (valid rows or -1 for the end, route
        block, N tile, tier, global scale) and route ids into the tile ring."""
        slot = ti % Int32(_TILE_STAGES)
        _mbar_wait(
            base + Int32(self.ws_bar_t_empty) + slot * Int32(_BARRIER_BYTES),
            ((ti // Int32(_TILE_STAGES)) & Int32(1)) ^ Int32(1),
        )
        hdr = base + Int32(self.ws_tile_off) + slot * Int32(self.ws_tile_slot)
        if ptid == Int32(0):
            st_shared_i32(hdr, valid)
            st_shared_i32(hdr + Int32(4), rb)
            st_shared_i32(hdr + Int32(8), nt)
            st_shared_i32(hdr + Int32(12), tier)
            st_shared_f32(hdr + Int32(16), scale)
        if valid >= Int32(0) and ptid < Int32(self.ws_m):
            idx = packed_route_indices[rb * Int32(self.ws_m) + ptid].to(Int32)
            if idx >= live:
                idx = Int32(-1)
            st_shared_i32(hdr + Int32(4 * _TILE_HEADER_WORDS) + ptid * Int32(4), idx)
        _mbar_arrive(base + Int32(self.ws_bar_t_full) + slot * Int32(_BARRIER_BYTES))

    @cute.jit
    def _ws_producer(
        self,
        is_fc1: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        n_total: cutlass.Constexpr,
        base: Int32,
        ptid: Int32,
        cta: Int32,
        grid: Int32,
        active_m: Int32,
        a_src: cute.Tensor,
        t0_w: cute.Tensor,
        t1_w: cute.Tensor,
        t0_global: cute.Tensor,
        t1_global: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_lo_experts: Int32,
        tier1_lo_experts: Int32,
        tier0_hi_experts: Int32,
        tier1_hi_experts: Int32,
    ):
        """Loads and FC1 rotation. Three cursors walk the CTA's tiles (tile
        cta + i * grid, whole K per tile, the cooperative whole-tile order):
        compressed weights (units of 64 K), input rows (blocks of 128 K) and
        the block cursor. Block-cursor iteration j publishes the tile header
        at a tile's first block, rotates block j, then refills the A slot and
        the two weight units the consumers released when they finished block
        j - 1, so loads run up to (A stages - 1) blocks and (weight stages)
        units ahead of the consumers."""
        lane = ptid & Int32(31)
        l16 = lane & Int32(15)
        n_tiles = Int32(n_total // self.ws_n)
        route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(self.ws_m)
        total_tiles = route_blocks * n_tiles
        live = active_m * Int32(self.top_k)
        units_per_tile = Int32(k_size // 64)
        blocks_per_tile = Int32(k_size // 128)
        na = self.ws_a_stages
        nc = self.ws_bc_stages
        meta = (
            n_tiles,
            block_expert_ids,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_lo_experts,
            tier1_lo_experts,
            tier0_hi_experts,
            tier1_hi_experts,
        )
        a_bar_base = Int32(self.ws_bar_a_full)
        if cutlass.const_expr(is_fc1):
            a_bar_base = Int32(self.ws_bar_a_raw)

        # Weight cursor and prologue.
        ct, _crb, cnt, _ccomb, ctier, clocal = self._ws_next_tile(
            is_fc1, cta, grid, total_tiles, *meta
        )
        cu = Int32(0)
        crow0, cstride, crow_bytes = self._ws_weight_row(
            is_fc1, n_total, k_size, cnt, n_tiles, clocal, ctier, t0_w, t1_w,
            tier0_lo_experts, tier1_lo_experts,
        )
        clast = Int32(0)
        if ct < total_tiles:
            clast = self._ws_last_block(block_expert_ids, _crb, route_blocks)
        bc_issued = Int32(0)
        for i in cutlass.range_constexpr(nc):
            if ct < total_tiles:
                if ptid == Int32(0):
                    self._ws_issue_bc(base, Int32(i), cu, crow0, cstride, crow_bytes, clast)
                bc_issued += Int32(1)
                cu += Int32(1)
                if cu == units_per_tile:
                    cu = Int32(0)
                    ct, _crb, cnt, _ccomb, ctier, clocal = self._ws_next_tile(
                        is_fc1, ct + grid, grid, total_tiles, *meta
                    )
                    if ct < total_tiles:
                        crow0, cstride, crow_bytes = self._ws_weight_row(
                            is_fc1, n_total, k_size, cnt, n_tiles, clocal, ctier, t0_w, t1_w,
                            tier0_lo_experts, tier1_lo_experts,
                        )
                        clast = self._ws_last_block(block_expert_ids, _crb, route_blocks)

        # Input-row cursor and prologue.
        rows = cute.make_rmem_tensor((self.ws_m // 8,), Int32)
        at, arb, _ant, _acomb, _atier, _alocal = self._ws_next_tile(
            is_fc1, cta, grid, total_tiles, *meta
        )
        ab = Int32(0)
        if at < total_tiles:
            self._ws_load_rows(is_fc1, rows, packed_route_indices, arb, ptid, live)
        a_issued = Int32(0)
        for i in cutlass.range_constexpr(na):
            if at < total_tiles:
                self._ws_issue_a(
                    k_size, a_src, rows, base, Int32(i), ab, ptid,
                    base + a_bar_base + Int32(i * _BARRIER_BYTES),
                )
                a_issued += Int32(1)
                ab += Int32(1)
                if ab == blocks_per_tile:
                    ab = Int32(0)
                    at, arb, _ant, _acomb, _atier, _alocal = self._ws_next_tile(
                        is_fc1, at + grid, grid, total_tiles, *meta
                    )
                    if at < total_tiles:
                        self._ws_load_rows(is_fc1, rows, packed_route_indices, arb, ptid, live)

        # Block cursor.
        dt, drb, dnt, dcomb, dtier, dlocal = self._ws_next_tile(
            is_fc1, cta, grid, total_tiles, *meta
        )
        blk = Int32(0)
        ti = Int32(0)
        suh_next = cute.make_rmem_tensor((4,), Uint32)
        while dt < total_tiles:
            dvalid = self._ws_valid_rows(packed_route_indices, drb, live, lane)
            scale = cutlass.Float32(0.0)
            if dtier == Int32(1):
                scale = t1_global[dlocal].to(cutlass.Float32)
            else:
                scale = t0_global[dlocal].to(cutlass.Float32)
            self._ws_publish_tile(
                base, ti, dvalid, drb, dnt, dtier, scale, packed_route_indices, live, ptid
            )
            ti += Int32(1)
            suh_addr = Int64(0)
            if cutlass.const_expr(is_fc1):
                suh_off = dcomb * Int32(k_size) + l16 * Int32(8)
                suh_addr = get_ptr_as_int64(gate_suh, suh_off)
                if dnt >= n_tiles // Int32(2):
                    suh_addr = get_ptr_as_int64(up_suh, suh_off)
                w0, w1, w2, w3 = ld_global_nc_v4_u32(suh_addr)
                suh_next[0] = w0
                suh_next[1] = w1
                suh_next[2] = w2
                suh_next[3] = w3
            b = Int32(0)
            while b < blocks_per_tile:
                if cutlass.const_expr(is_fc1):
                    suh_w = (suh_next[0], suh_next[1], suh_next[2], suh_next[3])
                    if b + Int32(1) < blocks_per_tile:
                        # The next block's SUH words load under this block's
                        # wait and rotation.
                        w0, w1, w2, w3 = ld_global_nc_v4_u32(
                            suh_addr + Int64(b + Int32(1)) * Int64(256)
                        )
                        suh_next[0] = w0
                        suh_next[1] = w1
                        suh_next[2] = w2
                        suh_next[3] = w3
                    a_slot = blk % Int32(na)
                    _mbar_wait(
                        base + Int32(self.ws_bar_a_raw) + a_slot * Int32(_BARRIER_BYTES),
                        (blk // Int32(na)) & Int32(1),
                    )
                    if cutlass.const_expr("norot" not in self.ws_exp):
                        self._ws_rotate_block(base, a_slot, dvalid, suh_w, ptid)
                    _mbar_arrive(base + Int32(self.ws_bar_a_full) + a_slot * Int32(_BARRIER_BYTES))
                if blk >= Int32(1):
                    # Consumers finished block blk - 1: its A slot and its two
                    # weight units are free.
                    if at < total_tiles:
                        fill = a_issued % Int32(na)
                        _mbar_wait(
                            base + Int32(self.ws_bar_a_empty) + fill * Int32(_BARRIER_BYTES),
                            ((a_issued // Int32(na)) & Int32(1)) ^ Int32(1),
                        )
                        self._ws_issue_a(
                            k_size, a_src, rows, base, fill, ab, ptid,
                            base + a_bar_base + fill * Int32(_BARRIER_BYTES),
                        )
                        a_issued += Int32(1)
                        ab += Int32(1)
                        if ab == blocks_per_tile:
                            ab = Int32(0)
                            at, arb, _ant, _acomb, _atier, _alocal = self._ws_next_tile(
                                is_fc1, at + grid, grid, total_tiles, *meta
                            )
                            if at < total_tiles:
                                self._ws_load_rows(
                                    is_fc1, rows, packed_route_indices, arb, ptid, live
                                )
                    for _u in cutlass.range_constexpr(2):
                        if ct < total_tiles:
                            fill = bc_issued % Int32(nc)
                            if ptid == Int32(0):
                                _mbar_wait(
                                    base + Int32(self.ws_bar_bc_empty) + fill * Int32(_BARRIER_BYTES),
                                    ((bc_issued // Int32(nc)) & Int32(1)) ^ Int32(1),
                                )
                                self._ws_issue_bc(base, fill, cu, crow0, cstride, crow_bytes, clast)
                            bc_issued += Int32(1)
                            cu += Int32(1)
                            if cu == units_per_tile:
                                cu = Int32(0)
                                ct, _crb, cnt, _ccomb, ctier, clocal = self._ws_next_tile(
                                    is_fc1, ct + grid, grid, total_tiles, *meta
                                )
                                if ct < total_tiles:
                                    crow0, cstride, crow_bytes = self._ws_weight_row(
                                        is_fc1, n_total, k_size, cnt, n_tiles, clocal, ctier,
                                        t0_w, t1_w, tier0_lo_experts, tier1_lo_experts,
                                    )
                                    clast = self._ws_last_block(
                                        block_expert_ids, _crb, route_blocks
                                    )
                blk += Int32(1)
                b += Int32(1)
            dt, drb, dnt, dcomb, dtier, dlocal = self._ws_next_tile(
                is_fc1, dt + grid, grid, total_tiles, *meta
            )
        # End of the CTA's tiles.
        self._ws_publish_tile(
            base, ti, Int32(-1), Int32(0), Int32(0), Int32(0), cutlass.Float32(0.0),
            packed_route_indices, live, ptid,
        )

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    @cute.jit
    def _ws_windows(self, gemm, row_addr: Int32, n16: Int32, lane: Int32, bits: cutlass.Constexpr):
        """The cooperative kernel's per-lane t256 funnel windows of one N16
        record (_load_b_registers_trellis256 for a single tile)."""
        tile_u32 = 8 * bits
        tbase = n16 * Int32(tile_u32)
        wa = Uint32(0)
        wb = Uint32(0)
        if cutlass.const_expr(bits == 4):
            previous_lane = (lane + Int32(31)) & Int32(31)
            bw = ld_shared_u32(row_addr + (tbase + lane) * Int32(4))
            aw = cute.arch.shuffle_sync(bw, previous_lane)
            wa = bw
            wb = (aw << Uint32(16)) | (bw >> Uint32(16))
        elif cutlass.const_expr(bits < 4):
            ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
            aw = ld_shared_u32(row_addr + (tbase + ia) * Int32(4))
            bw = ld_shared_u32(row_addr + (tbase + ib) * Int32(4))
            wa = gemm._trellis_funnel256(aw, bw, s2)
            wb = gemm._trellis_funnel256(aw, bw, s2 + Int32(4 * bits))
        elif cutlass.const_expr(bits == 5):
            ib0, ib1, sb, _ = trellis256_lane_geom_bits(lane, 0, 4, bits)
            ia0, ia1, sa, _ = trellis256_lane_geom_bits(lane, 4, 4, bits)
            b0 = ld_shared_u32(row_addr + (tbase + ib0) * Int32(4))
            b1 = ld_shared_u32(row_addr + (tbase + ib1) * Int32(4))
            a0 = ld_shared_u32(row_addr + (tbase + ia0) * Int32(4))
            a1 = ld_shared_u32(row_addr + (tbase + ia1) * Int32(4))
            wb = gemm._trellis_funnel256(b0, b1, sb)
            wa = gemm._trellis_funnel256(a0, a1, sa)
        else:
            i0, i2, s2, delta = trellis256_lane_geom_bits(lane, 0, 8, bits)
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)
            z0 = ld_shared_u32(row_addr + (tbase + i0) * Int32(4))
            z1 = ld_shared_u32(row_addr + (tbase + i1) * Int32(4))
            z2 = ld_shared_u32(row_addr + (tbase + i2) * Int32(4))
            wa, wb = trellis_align_stream_u32x2(z0, z1, z2, s2, delta)
        return wa, wb

    @cute.jit
    def _ws_kloop(
        self,
        is_fc1: cutlass.Constexpr,
        tier_idx: cutlass.Constexpr,
        active: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        acc,
        base: Int32,
        lut_addr: Int64,
        tid: Int32,
        step: Int32,
        blk: Int32,
    ):
        """One tile's K loop. Each 128-wide block waits once for its A slot
        and both compressed units, then runs eight straight-line K16 steps
        (window loads, t256 decode of this warp's two N16 records, ldmatrix of
        the ``active`` occupied M16 fragments, MMAs into the step's chain) and
        releases the slots."""
        bits = self.ws_bits[tier_idx]
        lane = tid & Int32(31)
        cw = tid >> Int32(5)
        na = self.ws_a_stages
        nc = self.ws_bc_stages
        mb_count = self.ws_mb
        blocks = Int32(k_size // 128)
        row_bytes = self.ws_n16 * 32 * bits
        lrow = lane & Int32(15)
        lchunk = lane >> Int32(4)
        b = Int32(0)
        while b < blocks:
            a_slot = blk % Int32(na)
            unit0 = step >> Int32(2)
            slot0 = unit0 % Int32(nc)
            slot1 = (unit0 + Int32(1)) % Int32(nc)
            _mbar_wait(
                base + Int32(self.ws_bar_bc_full) + slot0 * Int32(_BARRIER_BYTES),
                (unit0 // Int32(nc)) & Int32(1),
            )
            _mbar_wait(
                base + Int32(self.ws_bar_bc_full) + slot1 * Int32(_BARRIER_BYTES),
                ((unit0 + Int32(1)) // Int32(nc)) & Int32(1),
            )
            _mbar_wait(
                base + Int32(self.ws_bar_a_full) + a_slot * Int32(_BARRIER_BYTES),
                (blk // Int32(na)) & Int32(1),
            )
            a_base = base + Int32(self.ws_a_off) + a_slot * Int32(self.ws_a_slot)
            bc0 = base + Int32(self.ws_bc_off) + slot0 * Int32(self.ws_bc_slot)
            bc1 = base + Int32(self.ws_bc_off) + slot1 * Int32(self.ws_bc_slot)
            for q in cutlass.range_constexpr(8):
                chain = (q % 4) // 2
                row_addr = bc0 + Int32((q % 4) * row_bytes)
                if cutlass.const_expr(q >= 4):
                    row_addr = bc1 + Int32((q % 4) * row_bytes)
                gemm = (self.tier0, self.tier1)[tier_idx].fc2
                if cutlass.const_expr(is_fc1):
                    gemm = (self.tier0, self.tier1)[tier_idx].fc1
                b_regs = []
                for jj in cutlass.range_constexpr(2):
                    wa, wb = self._ws_windows(
                        gemm, row_addr, cw * Int32(2) + Int32(jj), lane, bits
                    )
                    frag = cute.make_rmem_tensor((2, 2), Uint32)
                    if cutlass.const_expr("nodecode" in self.ws_exp):
                        frag[0, 0] = wa
                        frag[0, 1] = wb
                        frag[1, 0] = wa
                        frag[1, 1] = wb
                    else:
                        gemm._scaled_dequant_b_fragment_trellis256_bits(
                            frag, wa, wb, lut_addr, bits
                        )
                    b_regs.append((frag[0, 0], frag[0, 1], frag[1, 0], frag[1, 1]))
                a_regs = []
                for mb in cutlass.range_constexpr(active):
                    row = Int32(16 * mb) + lrow
                    c = Int32(2 * q) + lchunk
                    pos = (c & Int32(8)) | ((c & Int32(7)) ^ (row & Int32(7)))
                    a_regs.append(ldmatrix_m8n8x4_b16(a_base + row * Int32(256) + pos * Int32(16)))
                for jj in cutlass.range_constexpr(0 if "nomma" in self.ws_exp else 2):
                    for mb in cutlass.range_constexpr(active):
                        for h in cutlass.range_constexpr(2):
                            o = (((chain * mb_count + mb) * 2 + jj) * 2 + h) * 4
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[o][0],
                                acc[o + 1][0],
                                acc[o + 2][0],
                                acc[o + 3][0],
                                a_regs[mb][0],
                                a_regs[mb][1],
                                a_regs[mb][2],
                                a_regs[mb][3],
                                b_regs[jj][2 * h],
                                b_regs[jj][2 * h + 1],
                            )
                            acc[o][0] = d0
                            acc[o + 1][0] = d1
                            acc[o + 2][0] = d2
                            acc[o + 3][0] = d3
            _mbar_arrive(base + Int32(self.ws_bar_bc_empty) + slot0 * Int32(_BARRIER_BYTES))
            _mbar_arrive(base + Int32(self.ws_bar_bc_empty) + slot1 * Int32(_BARRIER_BYTES))
            _mbar_arrive(base + Int32(self.ws_bar_a_empty) + a_slot * Int32(_BARRIER_BYTES))
            step += Int32(8)
            blk += Int32(1)
            b += Int32(1)
        return step, blk

    @cute.jit
    def _ws_store(
        self,
        active: cutlass.Constexpr,
        n_total: cutlass.Constexpr,
        acc,
        tid: Int32,
        hdr: Int32,
        nt: Int32,
        valid: Int32,
        scale: cutlass.Float32,
        c_out: cute.Tensor,
    ):
        """fp16((chain0 + chain1) * scale) for the warp's 32 columns of every
        occupied row (the cooperative kernel's fold and _write_bf16x2_shared
        rounding), stored straight to the route's output row."""
        lane = tid & Int32(31)
        cw = tid >> Int32(5)
        mb_count = self.ws_mb
        col_base = nt * Int32(self.ws_n) + cw * Int32(32) + Int32(2) * (lane & Int32(3))
        for mb in cutlass.range_constexpr(active):
            for rh in cutlass.range_constexpr(2):
                row = Int32(16 * mb + 8 * rh) + (lane >> Int32(2))
                if row < valid:
                    route = ld_shared_i32_relaxed(
                        hdr + Int32(4 * _TILE_HEADER_WORDS) + row * Int32(4)
                    )
                    row_base = Int64(route) * Int64(n_total)
                    for jj in cutlass.range_constexpr(2):
                        for h in cutlass.range_constexpr(2):
                            o0 = (((0 * mb_count + mb) * 2 + jj) * 2 + h) * 4 + 2 * rh
                            o1 = (((1 * mb_count + mb) * 2 + jj) * 2 + h) * 4 + 2 * rh
                            v0 = acc[o0][0] + acc[o1][0]
                            v1 = acc[o0 + 1][0] + acc[o1 + 1][0]
                            col = col_base + Int32(jj * 16 + h * 8)
                            st_global_u32(
                                get_ptr_as_int64(c_out, row_base + Int64(col)),
                                pack_f32x2_to_f16x2(v0 * scale, v1 * scale),
                            )

    @cute.jit
    def _ws_consumer(
        self,
        is_fc1: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        n_total: cutlass.Constexpr,
        base: Int32,
        lut_addr: Int64,
        tid: Int32,
        c_out: cute.Tensor,
    ):
        n_acc = 2 * self.ws_mb * 2 * 2 * 4
        acc = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(n_acc)]
        step = Int32(0)
        blk = Int32(0)
        ti = Int32(0)
        done = Int32(0)
        while done == Int32(0):
            slot = ti % Int32(_TILE_STAGES)
            _mbar_wait(
                base + Int32(self.ws_bar_t_full) + slot * Int32(_BARRIER_BYTES),
                (ti // Int32(_TILE_STAGES)) & Int32(1),
            )
            hdr = base + Int32(self.ws_tile_off) + slot * Int32(self.ws_tile_slot)
            valid = ld_shared_i32_relaxed(hdr)
            if valid < Int32(0):
                done = Int32(1)
            else:
                nt = ld_shared_i32_relaxed(hdr + Int32(8))
                tier = ld_shared_i32_relaxed(hdr + Int32(12))
                scale = ld_shared_f32(hdr + Int32(16))
                for i in cutlass.range_constexpr(n_acc):
                    acc[i][0] = cutlass.Float32(0.0)
                occupied = (valid + Int32(15)) // Int32(16)
                if occupied < Int32(1):
                    occupied = Int32(1)
                if occupied > Int32(self.ws_mb):
                    occupied = Int32(self.ws_mb)
                for active in cutlass.range_constexpr(1, self.ws_mb + 1):
                    if occupied == Int32(active):
                        if tier == Int32(0):
                            step, blk = self._ws_kloop(
                                is_fc1, 0, active, k_size, acc, base, lut_addr, tid, step, blk
                            )
                        else:
                            step, blk = self._ws_kloop(
                                is_fc1, 1, active, k_size, acc, base, lut_addr, tid, step, blk
                            )
                        self._ws_store(
                            active, n_total, acc, tid, hdr, nt, valid, scale, c_out
                        )
                _mbar_arrive(base + Int32(self.ws_bar_t_empty) + slot * Int32(_BARRIER_BYTES))
                ti += Int32(1)
