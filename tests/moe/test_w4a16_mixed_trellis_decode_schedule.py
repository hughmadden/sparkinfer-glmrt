"""The L2 decode schedule of the cooperative mixed-Trellis kernel.

A decode schedule changes only when, and with which L2 policy, weight words
are fetched: every output bit must equal the default schedule's. The GPU
tests use the GLM 5.3 Flash TP4 rank geometry (H 4096, a 512-wide intermediate
slice, 288 experts, top-8, K3/K4 tiers) at the decode capacities the Spark
worker runs (m1 for one row, m80 for 2-80 rows, both K64), and also check the
FR-G.7(a) property: a row alone, in 8 and in 64 gets the same bits.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel, W4A16GemmKernel
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    DECODE_SCHEDULE_PRESETS,
    MixedTrellisDecodeSchedule,
    bind_mixed_trellis,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    parse_decode_schedule,
    run_bound_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

HIDDEN = 4096
INTERMEDIATE = 512  # one TP4 rank's slice of GLM 5.3 Flash's 2048
EXPERTS = 288
TOPK = 8
K64_N256 = (64, 256, 64, 256)
# Experts 284-287 decode as K3 (tier 0), the rest as K4 (tier 1): both tiers run.
TIER0_IDS = tuple(range(284, 288))
TIER1_IDS = tuple(range(0, 284))


def test_decode_schedule_presets_parse_and_round_trip() -> None:
    gb10 = parse_decode_schedule("gb10")
    assert gb10 is not None
    assert gb10.canonical() == DECODE_SCHEDULE_PRESETS["gb10"]
    assert parse_decode_schedule(gb10.canonical()) == gb10
    # Measured on GB10: evict-first weights only.
    assert gb10 == MixedTrellisDecodeSchedule(l2_evict_first_b=True)
    for default in (None, "", "default", "l2=1,pf1=0,pf2=0,pdl=1", "pf2=0"):
        assert parse_decode_schedule(default) is None
    custom = parse_decode_schedule("pf2=8,l2=2")
    assert custom == MixedTrellisDecodeSchedule(
        l2_evict_first_b=True, fc2_prefetch_k_tiles=8
    )
    assert custom.kernel_options() == {
        "l2_evict_first_b": True,
        "fc1_l2_prefetch_k_tiles": 0,
        "fc2_l2_prefetch_k_tiles": 8,
        "phase_l2_prefetch": False,
    }


@pytest.mark.parametrize(
    "spec",
    ["gb11", "l2=3", "pdl=0", "pf1=-1", "pf2=65", "l2=2,l2=2", "pdl=2,pf1=4", "pf1", "pf1=x"],
)
def test_decode_schedule_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_decode_schedule(spec)


@pytest.mark.parametrize(
    "options",
    [
        {"warp_specialized": True, "moe_block_size": 64},
        {"direct_topk_routes": True},
        {"paired_boundary": "first"},
    ],
)
def test_decode_schedule_rejects_unsupported_kernels_before_cuda(options) -> None:
    kwargs = dict(
        size_m=80, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        tier0_num_experts=4, tier1_num_experts=284, top_k=TOPK,
        max_m_blocks=96, sms=48, max_shared_mem=101376,
        force_tile_config=K64_N256, decode_schedule="gb10",
    )
    kwargs.update(options)
    with pytest.raises(ValueError, match="decode schedule"):
        compile_mixed_trellis(**kwargs)


def _fc2_gemm(**options) -> W4A16GemmKernel:
    return W4A16GemmKernel(
        size_m=80 * TOPK, size_n=HIDDEN, size_k=INTERMEDIATE, num_experts=EXPERTS,
        top_k=1, mul_topk_weights=False, tile_n=256, tile_k=64, moe_block_size=8,
        max_m_blocks=96, element_dtype="fp16", weight_layout="trellis_t256",
        scale_format="e4m3_k32", w13_layout="packed", trellis_bits=4,
        trellis_codebook="mcg", schedule_whole_tiles=True, dynamic_num_experts=True,
        **options,
    )


def test_memory_schedule_extends_kernel_keys_only_when_set() -> None:
    default = _fc2_gemm()
    explicit_default = _fc2_gemm(l2_evict_first_b=False, l2_prefetch_k_tiles=0)
    scheduled = _fc2_gemm(l2_evict_first_b=True, l2_prefetch_k_tiles=8)
    assert explicit_default.__cache_key__ == default.__cache_key__
    assert scheduled.__cache_key__[:-1] == default.__cache_key__
    assert scheduled.__cache_key__[-1] == ("l2_schedule", True, 8)
    # One bulk prefetch per K16 row: 16 Trellis tiles of 4 bits, 2 KiB.
    assert scheduled.l2_prefetch_row_bytes == 2048
    # A depth beyond the tile's K range prefetches the whole tile.
    assert _fc2_gemm(l2_prefetch_k_tiles=20).l2_prefetch_k_tiles == default.k_tiles == 8


def test_memory_schedule_requires_whole_tile_trellis_routes() -> None:
    with pytest.raises(ValueError, match="L2 weight schedule"):
        W4A16GemmKernel(
            size_m=80, size_n=HIDDEN, size_k=INTERMEDIATE, num_experts=EXPERTS,
            top_k=1, mul_topk_weights=False, tile_n=256, tile_k=64, moe_block_size=8,
            max_m_blocks=96, element_dtype="fp16", weight_layout="packed",
            l2_evict_first_b=True,
        )
    with pytest.raises(ValueError, match="L2 weight schedule"):
        _fc2_gemm(direct_topk_routes=True, l2_prefetch_k_tiles=2)


def _fused(**options) -> W4A16FusedMoeKernel:
    return W4A16FusedMoeKernel(
        size_m=80, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS, top_k=TOPK, activation="silu", swiglu_limit=10.0,
        apply_router_weight_on_input=False, zero_fc2_output=False,
        fc1_tile_n=256, fc1_tile_k=64, fc2_tile_n=256, fc2_tile_k=64,
        moe_block_size=8, max_m_blocks=96, element_dtype="fp16",
        weight_layout="trellis_t256", scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj", trellis_bits=4, trellis_codebook="mcg",
        intermediate_rotation=True, full_rotation=True, rotation_input_dtype="bf16",
        schedule_whole_tiles=True, **options,
    )


def test_phase_prefetch_needs_both_tile_prefetches() -> None:
    with pytest.raises(ValueError, match="phase L2 prefetch"):
        _fused(fc1_l2_prefetch_k_tiles=4, phase_l2_prefetch=True)
    default = _fused()
    scheduled = _fused(**parse_decode_schedule("l2=2,pf1=4,pf2=8,pdl=2").kernel_options())
    assert scheduled.phase_l2_prefetch and not default.phase_l2_prefetch
    assert scheduled.__cache_key__[-1] == "phase_l2_prefetch"
    assert "phase_l2_prefetch" not in default.__cache_key__
    assert (scheduled.fc1.l2_prefetch_k_tiles, scheduled.fc2.l2_prefetch_k_tiles) == (4, 8)


def test_entries_without_prefetch_hooks_refuse_the_tile_prefetch() -> None:
    """Only the cooperative mixed-Trellis kernel wires the next-tile and phase
    prefetches (its _prefetch_tier_tile hooks). The standalone GEMM entry and
    the single-tier fused entry refuse them instead of compiling a kernel that
    silently skips them; evict-first staging needs no hook and stays."""
    for kernel in (_fc2_gemm(), _fc2_gemm(l2_evict_first_b=True), _fused(), _fused(l2_evict_first_b=True),
                   _fused(**parse_decode_schedule("gb10").kernel_options())):
        kernel._require_wired_l2_prefetch()
    for kernel in (_fc2_gemm(l2_prefetch_k_tiles=8), _fused(fc1_l2_prefetch_k_tiles=4),
                   _fused(fc2_l2_prefetch_k_tiles=8),
                   _fused(**parse_decode_schedule("l2=2,pf1=4,pf2=8,pdl=2").kernel_options())):
        with pytest.raises(ValueError, match="mixed-Trellis"):
            kernel._require_wired_l2_prefetch()


# ---------------------------------------------------------------------------
# GPU: bit identity and batch invariance (SM120/SM121).


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


requires_sm12x = pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")


def _tier(experts: int, bits: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).to(torch.float16)

    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_experts=experts,
        activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device, seed=seed,
        params_dtype=torch.float16, w13_layout="trellis_t256_proj", trellis_bits=bits,
        codebook="mcg", gate_suh=scales((experts, HIDDEN)), up_suh=scales((experts, HIDDEN)),
        intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
        down_svh=scales((experts, HIDDEN)), tile_config=(64, 128, 64, 128),
    )


@pytest.fixture(scope="module")
def glmf_layer():
    if not _sm12x_available():
        pytest.skip("requires an SM120/SM121 GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    tier0 = _tier(len(TIER0_IDS), 3, 301, device)
    tier1 = _tier(len(TIER1_IDS), 4, 401, device)
    global_to_combined, descriptor = build_tiered_maps(TIER0_IDS, TIER1_IDS, device=device)
    rotations = combine_trellis_rotations(tier0, tier1)
    generator = torch.Generator(device="cpu").manual_seed(20261006)
    ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(80)])
    ids = ids.to(torch.int32).to(device)
    weights = torch.rand((80, TOPK), generator=generator).to(device)
    weights = 2.5 * weights / weights.sum(dim=-1, keepdim=True)
    x = (torch.randn((80, HIDDEN), generator=generator) * 0.25).to(torch.bfloat16).to(device)
    return device, tier0, tier1, global_to_combined, descriptor, rotations, x, weights, ids


_RUNNERS: dict[tuple, object] = {}


def _runner(layer, capacity: int, schedule: str | None, tile, blocks_per_sm):
    key = (capacity, schedule, tile, blocks_per_sm)
    if key not in _RUNNERS:
        device, tier0, tier1, global_to_combined, descriptor, rotations, *_ = layer
        props = torch.cuda.get_device_properties(device)
        slots = route_pack_capacity(capacity * TOPK, 8, EXPERTS, topk=TOPK)[1]
        launch = compile_mixed_trellis(
            size_m=capacity, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
            tier0_num_experts=len(TIER0_IDS), tier1_num_experts=len(TIER1_IDS),
            top_k=TOPK, route_num_experts=EXPERTS, max_m_blocks=(slots + 7) // 8,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=tile, tier0_bits=3, tier1_bits=4, trellis_codebook="mcg",
            swiglu_limit=10.0, moe_block_size=8, rotation_input_dtype="bf16",
            full_rotation_output_dtype="bf16", route_ids_dtype=torch.int32,
            force_blocks_per_sm=blocks_per_sm, decode_schedule=schedule,
        )
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=int(props.multi_processor_count))
        binding = bind_mixed_trellis(tier0, tier1, global_to_combined, descriptor, rotations, launch)
        _RUNNERS[key] = (launch, buffers, binding)
    return _RUNNERS[key]


def _run(layer, rows: slice, capacity: int, schedule=None, tile=K64_N256, blocks_per_sm=None):
    *_, x, weights, ids = layer
    launch, buffers, binding = _runner(layer, capacity, schedule, tile, blocks_per_sm)
    out = run_bound_mixed_trellis(
        x[rows].contiguous(), weights[rows].contiguous(), ids[rows].contiguous(), binding, buffers
    )
    torch.cuda.synchronize()
    return out.clone()


# (schedule, tile, blocks per SM). The N128 two-CTA rows check that the N
# tiling and residency are bit-neutral too; they are GB10 sweep candidates.
VARIANTS = [
    ("gb10", K64_N256, None),
    ("l2=2,pf1=0,pf2=0,pdl=1", K64_N256, None),
    ("l2=1,pf1=8,pf2=8,pdl=2", K64_N256, None),
    (None, (64, 128, 64, 128), 2),
    ("gb10", (64, 128, 64, 128), 2),
]


@requires_sm12x
@pytest.mark.parametrize("rows", [1, 8, 64, 80])
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"{v[0]}|{v[1][1]}|{v[2]}")
def test_glmf_decode_schedule_is_bit_identical(glmf_layer, rows, variant) -> None:
    schedule, tile, blocks_per_sm = variant
    capacity = 1 if rows == 1 else 80
    window = slice(0, rows)
    reference = _run(glmf_layer, window, capacity)
    candidate = _run(glmf_layer, window, capacity, schedule, tile, blocks_per_sm)
    assert torch.isfinite(reference.float()).all()
    assert float(reference.float().abs().max()) > 0.0
    assert torch.equal(candidate, reference)


@requires_sm12x
@pytest.mark.parametrize("variant", [(None, K64_N256, None), *VARIANTS],
                         ids=lambda v: f"{v[0]}|{v[1][1]}|{v[2]}")
def test_glmf_decode_row_alone_in_8_and_in_64_has_one_result(glmf_layer, variant) -> None:
    schedule, tile, blocks_per_sm = variant
    row = 5
    alone = _run(glmf_layer, slice(row, row + 1), 1, schedule, tile, blocks_per_sm)
    in_8 = _run(glmf_layer, slice(0, 8), 80, schedule, tile, blocks_per_sm)[row : row + 1]
    in_64 = _run(glmf_layer, slice(0, 64), 80, schedule, tile, blocks_per_sm)[row : row + 1]
    assert torch.equal(in_8, alone)
    assert torch.equal(in_64, alone)
    # ...and the same bits as the default schedule's row alone.
    assert torch.equal(alone, _run(glmf_layer, slice(row, row + 1), 1))


@requires_sm12x
def test_glmf_gb10_package_shape_keeps_one_result_per_row(glmf_layer) -> None:
    """The gb10 package's decode exports: m1 keeps the 64x256 tile, m80 runs 64x128 tiles at
    two CTAs per SM, both evict-first. A row alone (m1), in 8 and in 64 (m80) still gets
    the default schedule's bits."""
    row = 5
    alone = _run(glmf_layer, slice(row, row + 1), 1, "gb10", K64_N256, None)
    in_8 = _run(glmf_layer, slice(0, 8), 80, "gb10", (64, 128, 64, 128), 2)[row : row + 1]
    in_64 = _run(glmf_layer, slice(0, 64), 80, "gb10", (64, 128, 64, 128), 2)[row : row + 1]
    default_alone = _run(glmf_layer, slice(row, row + 1), 1)
    assert torch.equal(alone, default_alone)
    assert torch.equal(in_8, default_alone)
    assert torch.equal(in_64, default_alone)
