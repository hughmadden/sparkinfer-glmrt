from __future__ import annotations

import torch

from b12x.attention import dsa_indexer as nsa_indexer
from b12x.attention import dsv4_producer
from b12x.attention._shared.mla.compressed_reference import (
    pack_compressed_sparse_mla_kv_cache_reference as pack_compressed_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.kernel import run_unified_decode
from b12x.attention._shared.mla.traits import ScaleFormat
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps as B12XCompressedMLAScratchCaps,
    _compressed_sparse_mla_scratch_layout as _compressed_mla_scratch_layout,
    _materialize_compressed_sparse_mla_scratch as _materialize_compressed_mla_scratch,
)
from b12x.attention.dsv4_producer._impl import (
    DSV4_INDEX_WEIGHT_SCALE,
    DSV4_KV_PAGE_BYTES,
    DSV4_NVFP4_PAGE_BYTES,
    _run_indexer_query_post,
    _run_normalize_query_rope,
    _run_normalize_rank_pack_kv,
)
from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)

from ..conftest import require_b12x
from .._reference.helpers import (
    dequantize_nvfp4_mla_nope,
    pack_dsv4_nvfp4_record_reference,
)


def _rope_forward(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    nope_dim: int,
) -> torch.Tensor:
    output = values.clone()
    rope = values[..., nope_dim:].float()
    pairs = rope.unflatten(-1, (-1, 2))
    selected = cos_sin[positions.long()]
    cos_v, sin_v = selected.chunk(2, dim=-1)
    even = pairs[..., 0] * cos_v - pairs[..., 1] * sin_v
    odd = pairs[..., 1] * cos_v + pairs[..., 0] * sin_v
    output[..., nope_dim:] = (
        torch.stack((even, odd), dim=-1).flatten(-2).to(torch.bfloat16)
    )
    return output


def _rmsnorm(
    values: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + eps
    )
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(torch.bfloat16)


def _block_fp8_weight(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) / 16).to(
        torch.float8_e4m3fn
    )
    scale = torch.full(
        (rows // 128, cols // 128),
        127,
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    return weight, scale


def _fwht_128(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    output = values.float().reshape(-1, 128)
    width = 1
    while width < 128:
        groups = output.view(-1, 2, width)
        first = groups[:, 0].clone()
        second = groups[:, 1].clone()
        output = torch.stack((first + second, first - second), dim=1).reshape(-1, 128)
        width *= 2
    return (output * (128**-0.5)).reshape(shape)


def _fp4_qat_reference(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    blocks = values.float().reshape(-1, 32)
    scales = blocks.abs().amax(dim=1).clamp_min(6 * torch.finfo(torch.float32).tiny) / 6
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    magnitude = (blocks.abs() / scales[:, None]).clamp_max(6)
    fp4 = torch.where(
        magnitude < 0.25,
        0.0,
        torch.where(
            magnitude < 0.75,
            0.5,
            torch.where(
                magnitude < 1.25,
                1.0,
                torch.where(
                    magnitude < 1.75,
                    1.5,
                    torch.where(
                        magnitude < 2.5,
                        2.0,
                        torch.where(
                            magnitude < 3.5,
                            3.0,
                            torch.where(magnitude < 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return (
        (torch.copysign(fp4, blocks) * scales[:, None])
        .reshape(shape)
        .to(torch.bfloat16)
    )


def test_plan_pins_flash_and_pro_caller_owned_workspace() -> None:
    flash = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=2048,
            hidden=4096,
            q_lora_rank=1024,
            heads=64,
        )
    )
    pro = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=2048,
            hidden=7168,
            q_lora_rank=1536,
            heads=128,
        )
    )

    assert flash.scratch_specs()[0].name == "dsv4_producer.scratch"
    assert flash.scratch_specs()[0].nbytes == 21_626_880
    assert pro.scratch_specs()[0].nbytes == 33_619_968
    assert flash.layout.qkv_linear_offset == 0
    assert flash.layout.q_linear_offset % 1024 == 0
    assert flash.layout.qkv_output_offset % 1024 == 0
    assert flash.layout.q_rank_offset % 1024 == 0
    assert flash.caps.main_page_bytes == DSV4_KV_PAGE_BYTES
    assert (
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=1,
            hidden=4096,
            q_lora_rank=1024,
            heads=64,
            cache_format="nvfp4",
        ).main_page_bytes
        == DSV4_NVFP4_PAGE_BYTES
    )

    flash_kv = dsv4_producer.plan_kv(flash.caps)
    pro_kv = dsv4_producer.plan_kv(pro.caps)
    assert flash_kv.scratch_specs()[0].name == "dsv4_kv_producer.scratch"
    assert flash_kv.scratch_specs()[0].nbytes == 11_010_048
    assert pro_kv.scratch_specs()[0].nbytes == 17_694_720
    assert flash_kv.layout.kv_linear_offset == 0
    assert flash_kv.layout.kv_output_offset % 1_024 == 0
    assert flash_kv.layout.kv_output_bytes == 2_048 * 512 * 2
    assert flash_kv.scratch_specs()[0].nbytes < flash.scratch_specs()[0].nbytes
    for entry_point in (
        "KVPlan",
        "KVBinding",
        "KVWeights",
        "plan_kv",
        "bind_kv",
        "pack_kv_weights",
        "run_kv",
    ):
        assert entry_point in dsv4_producer.META.entry_points


def test_caps_reject_absorbed_mla_and_non_native_page_geometry() -> None:
    for kwargs in (
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 8},
        {"hidden": 4096, "q_lora_rank": 512, "heads": 64},
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 64, "head_dim": 576},
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 64, "page_size": 64},
    ):
        try:
            dsv4_producer.Caps(device="cpu", max_tokens=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid DSV4 producer geometry was accepted: {kwargs}"
            )


def test_indexer_plan_pins_flash_and_pro_query_producer_arena() -> None:
    flash = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device="cpu", max_tokens=2_048, hidden=4_096, q_lora_rank=1_024
        )
    )
    pro = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device="cpu", max_tokens=2_048, hidden=7_168, q_lora_rank=1_536
        )
    )

    assert flash.scratch_specs()[0].name == "dsv4_indexer_producer.scratch"
    assert flash.scratch_specs()[0].nbytes == 36_044_800
    assert pro.scratch_specs()[0].nbytes == 37_158_912
    assert flash.layout.q_output_bytes == 2_048 * 64 * 128 * 2
    assert flash.layout.weights_output_bytes == 2_048 * 64 * 2
    assert flash.layout.q_output_offset % 1_024 == 0
    assert flash.layout.weights_output_offset % 1_024 == 0


def test_indexer_caps_reject_cross_variant_or_non_native_geometry() -> None:
    for kwargs in (
        {"hidden": 4_096, "q_lora_rank": 1_536},
        {"hidden": 7_168, "q_lora_rank": 1_024},
        {"hidden": 4_096, "q_lora_rank": 1_024, "heads": 128},
        {"hidden": 4_096, "q_lora_rank": 1_024, "head_dim": 64},
    ):
        try:
            dsv4_producer.IndexerCaps(device="cpu", max_tokens=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid DSV4 indexer producer geometry was accepted: {kwargs}"
            )


def test_native_dsv4_nvfp4_writer_matches_432_byte_record_and_replays() -> None:
    require_b12x()
    torch.manual_seed(20260811)
    tokens, q_rank, eps = 2, 1_024, 1.0e-6
    qkv = torch.randn((tokens, q_rank + 512), device="cuda", dtype=torch.bfloat16)
    q_norm = torch.randn((q_rank,), device="cuda", dtype=torch.bfloat16)
    kv_norm = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 257], device="cuda", dtype=torch.int32)
    cos_sin = torch.randn((2, 64), device="cuda", dtype=torch.float32)
    q_rank_out = torch.empty((tokens, q_rank), device="cuda", dtype=torch.bfloat16)
    cache = torch.full(
        (2, DSV4_NVFP4_PAGE_BYTES), 0xA5, device="cuda", dtype=torch.uint8
    )

    def run() -> None:
        _run_normalize_rank_pack_kv(
            qkv,
            q_norm,
            kv_norm,
            positions,
            slots,
            cos_sin,
            q_rank_out,
            cache,
            eps=eps,
            cache_format="nvfp4",
        )

    run()
    torch.cuda.synchronize()
    eager_cache = cache.clone()
    eager_q_rank = q_rank_out.clone()

    kv = qkv[:, q_rank:].float()
    normalized = (
        kv * torch.rsqrt(kv.square().mean(dim=-1, keepdim=True) + eps) * kv_norm.float()
    ).to(torch.bfloat16)
    expected = normalized.clone()
    pairs = normalized[:, 448:].float().reshape(tokens, 32, 2)
    selected = cos_sin[positions.long()]
    cos_v, sin_v = selected.chunk(2, dim=-1)
    expected[:, 448:] = (
        torch.stack(
            (
                pairs[..., 0] * cos_v - pairs[..., 1] * sin_v,
                pairs[..., 1] * cos_v + pairs[..., 0] * sin_v,
            ),
            dim=-1,
        )
        .reshape(tokens, 64)
        .to(torch.bfloat16)
    )
    expected_records = pack_dsv4_nvfp4_record_reference(expected)
    actual_records = torch.stack((cache[0, :432], cache[1, 432:864]))
    assert torch.equal(actual_records, expected_records)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache, eager_cache)
    assert torch.equal(q_rank_out, eager_q_rank)

    # Consume those exact producer records through the native DSV4 decode arm.
    records = torch.stack((eager_cache[0, :432], eager_cache[1, 432:864]))
    dequantized, _ = dequantize_nvfp4_mla_nope(records)
    rope = records[:, 304:432].contiguous().view(torch.bfloat16).float()
    keys = torch.cat((dequantized[:, :448], rope), dim=1)
    query = torch.randn((1, 8, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([[0, 257]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2], device="cuda", dtype=torch.int32)
    sm_scale = 512**-0.5
    scores = torch.matmul(query[0].float(), keys.t()) * sm_scale
    expected_output = torch.matmul(scores.softmax(dim=-1), dequantized).unsqueeze(0)
    caps = B12XCompressedMLAScratchCaps(
        device=torch.device("cuda"),
        num_q_heads=8,
        max_q_rows=1,
        max_width=2,
        head_dim=512,
        v_head_dim=512,
        max_chunks_per_row=8,
        page_size=256,
    )
    layout = _compressed_mla_scratch_layout(caps)
    workspace = _materialize_compressed_mla_scratch(
        caps,
        torch.zeros(layout.nbytes, device="cuda", dtype=torch.uint8),
        layout,
        None,
    )
    output = torch.empty_like(query)

    def decode() -> None:
        run_unified_decode(
            q_all=query,
            swa_k_cache=eager_cache,
            swa_indices=indices,
            swa_topk_lengths=lengths,
            workspace=workspace,
            sm_scale=sm_scale,
            swa_page_size=256,
            forced_num_splits=1,
            out=output,
            scale_format_override=ScaleFormat.NVFP4_E4M3,
        )

    decode()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.float(), expected_output, atol=3.0e-2, rtol=3.0e-2
    )
    decode_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode_graph):
        decode()
    output.fill_(float("nan"))
    decode_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.float(), expected_output, atol=3.0e-2, rtol=3.0e-2
    )


def test_native_dsv4_nvfp4_writer_addresses_pages_past_int32_byte_range() -> None:
    """Keep the production pool offset arithmetic wider than 32 bits."""

    require_b12x()
    torch.manual_seed(20260812)
    q_rank, eps = 1_024, 1.0e-6
    high_page = (2**31) // DSV4_NVFP4_PAGE_BYTES + 1
    high_row = 17
    qkv = torch.randn((1, q_rank + 512), device="cuda", dtype=torch.bfloat16)
    q_norm = torch.ones((q_rank,), device="cuda", dtype=torch.bfloat16)
    kv_norm = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros((1,), device="cuda", dtype=torch.int32)
    slots = torch.tensor([high_page * 256 + high_row], device="cuda", dtype=torch.int64)
    cos_sin = torch.zeros((1, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1
    q_rank_out = torch.empty((1, q_rank), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(
        (high_page + 1, DSV4_NVFP4_PAGE_BYTES),
        device="cuda",
        dtype=torch.uint8,
    )
    try:
        cache[high_page].fill_(0xA5)
        _run_normalize_rank_pack_kv(
            qkv,
            q_norm,
            kv_norm,
            positions,
            slots,
            cos_sin,
            q_rank_out,
            cache,
            eps=eps,
            cache_format="nvfp4",
        )
        torch.cuda.synchronize()

        start = high_row * 432
        record = cache[high_page, start : start + 432]
        assert torch.count_nonzero(record[:288]) > 0
        assert torch.count_nonzero(record[288:304]) == 0
        assert torch.all(cache[high_page, start - 432 : start] == 0xA5)
        assert torch.all(cache[high_page, start + 432 : start + 864] == 0xA5)
    finally:
        del cache
        torch.cuda.empty_cache()


def test_rank_norm_and_main_kv_pack_match_checkpoint_reference() -> None:
    require_b12x()
    torch.manual_seed(20260804)
    tokens, q_rank, eps = 2, 1024, 1.0e-6
    qkv = (
        torch.randn((tokens, q_rank + 512), device="cuda", dtype=torch.bfloat16) / 3
    ).contiguous()
    q_norm = (
        torch.randn((q_rank,), device="cuda", dtype=torch.bfloat16) / 8 + 1
    ).contiguous()
    kv_norm = (
        torch.randn((512,), device="cuda", dtype=torch.bfloat16) / 8 + 1
    ).contiguous()
    positions = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    angles = torch.randn((4, 32), device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    slots = torch.tensor([0, 256], device="cuda", dtype=torch.int64)
    q_rank_out = torch.empty((tokens, q_rank), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((2, DSV4_KV_PAGE_BYTES), device="cuda", dtype=torch.uint8)

    _run_normalize_rank_pack_kv(
        qkv,
        q_norm,
        kv_norm,
        positions,
        slots,
        cos_sin,
        q_rank_out,
        cache,
        eps=eps,
    )
    torch.cuda.synchronize()

    expected_q_rank = _rmsnorm(qkv[:, :q_rank], q_norm, eps)
    expected_kv = _rmsnorm(qkv[:, q_rank:], kv_norm, eps)
    expected_kv = _rope_forward(expected_kv, positions, cos_sin, nope_dim=448)
    torch.testing.assert_close(q_rank_out, expected_q_rank, rtol=0, atol=0)
    for token in range(tokens):
        expected_page = pack_compressed_mla_kv_cache_reference(
            expected_kv[token : token + 1, :448],
            expected_kv[token : token + 1, 448:],
            page_size=256,
            num_pages=1,
        )
        torch.testing.assert_close(cache[token], expected_page[0], rtol=0, atol=0)


def test_query_post_matches_per_head_rmsnorm_and_partial_rope() -> None:
    require_b12x()
    torch.manual_seed(20260805)
    tokens, heads, eps = 2, 64, 1.0e-6
    query = (
        torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    original = query.clone()
    positions = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    angles = torch.randn((4, 32), device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()

    _run_normalize_query_rope(query, positions, cos_sin, eps=eps)
    torch.cuda.synchronize()

    expected = _rmsnorm(original, None, eps)
    expanded_positions = positions[:, None].expand(tokens, heads).reshape(-1)
    expected = _rope_forward(
        expected.reshape(tokens * heads, 512),
        expanded_positions,
        cos_sin,
        nope_dim=448,
    ).reshape(tokens, heads, 512)
    torch.testing.assert_close(query, expected, rtol=0, atol=0)


def test_indexer_query_post_matches_checkpoint_rope_hadamard_and_fp4_qat() -> None:
    require_b12x()
    torch.manual_seed(20260807)
    tokens, heads = 2, 64
    raw_query = (
        torch.randn((tokens, heads, 128), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    raw_weights = (
        torch.randn((tokens, heads), device="cuda", dtype=torch.bfloat16) / 3
    ).contiguous()
    positions = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
    angles = torch.randn((5, 32), device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    query = torch.empty((tokens, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
    head_weights = torch.empty((tokens, heads), device="cuda", dtype=torch.float32)

    _run_indexer_query_post(
        raw_query, raw_weights, positions, cos_sin, query, head_weights
    )
    torch.cuda.synchronize()

    expanded_positions = positions[:, None].expand(tokens, heads).reshape(-1)
    expected = _rope_forward(
        raw_query.reshape(-1, 128),
        expanded_positions,
        cos_sin,
        nope_dim=64,
    ).reshape_as(raw_query)
    expected = _fp4_qat_reference(_fwht_128(expected)).to(torch.float8_e4m3fn)
    error = (query.float() - expected.float()).abs()
    # Register reduction order may move a very small number of lanes across an
    # E2M1 discontinuity; all other lanes retain the exact FP8 encoding.
    assert int((error > 0).sum()) <= tokens * heads
    assert float(error.max()) <= 0.25
    expected_weights = (raw_weights * DSV4_INDEX_WEIGHT_SCALE).float()
    torch.testing.assert_close(head_weights, expected_weights, rtol=0, atol=0)


def test_indexer_query_feeds_physical_slot_topk_without_remap_allocation() -> None:
    require_b12x()
    torch.manual_seed(20260808)
    device = torch.device("cuda")
    rows, heads, pages, topk = 2, 64, 12, 512
    raw_query = (
        torch.randn((rows, heads, 128), device=device, dtype=torch.bfloat16) / 3
    ).contiguous()
    raw_weights = (
        torch.randn((rows, heads), device=device, dtype=torch.bfloat16) / 4
    ).contiguous()
    positions = torch.tensor([15, 19], device=device, dtype=torch.int32)
    angles = torch.randn((20, 32), device=device, dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    query = torch.empty((rows, heads, 128), device=device, dtype=torch.float8_e4m3fn)
    head_weights = torch.empty((rows, heads), device=device, dtype=torch.float32)
    _run_indexer_query_post(
        raw_query, raw_weights, positions, cos_sin, query, head_weights
    )

    index_rows = torch.randn((pages * 64, 128), device=device, dtype=torch.float32)
    index_cache = pack_index_k_cache_reference(index_rows)
    first = torch.tensor(
        [5, 2, 10, 1, 8, 0, 11, 4, 7, 3, 9, 6],
        device=device,
        dtype=torch.int32,
    )
    second = torch.tensor(
        [9, 0, 6, 11, 3, 5, 1, 8, 4, 10, 2, 7],
        device=device,
        dtype=torch.int32,
    )
    page_table = torch.stack((first, second)).contiguous()
    seqlens = torch.tensor([700, 641], device=device, dtype=torch.int32)
    output = torch.empty((rows, topk), device=device, dtype=torch.int32)
    operands = dict(
        q_fp8=query,
        query_weights=head_weights,
        index_k_cache=index_cache,
        page_table=page_table,
        cache_lengths=seqlens,
        active_width=torch.tensor([pages * 64], device=device, dtype=torch.int32),
        output_indices=output,
    )
    selection_caps = nsa_indexer.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=pages,
        topk=topk,
        mode="decode",
        output_index_space="physical",
    )
    selection_plan = nsa_indexer.plan(
        selection_caps,
        invocation=nsa_indexer.invocation_from_tensors(selection_caps, **operands),
    )
    (selection_spec,) = selection_plan.scratch_specs()
    selection_scratch = torch.empty(
        selection_spec.shape, dtype=selection_spec.dtype, device=device
    )
    selection_binding = nsa_indexer.bind(
        selection_plan, scratch=selection_scratch, **operands
    )
    nsa_indexer.run(selection_binding)
    torch.cuda.synchronize()

    logits = paged_decode_logits_reference(
        q_fp8=query,
        weights=head_weights,
        index_k_cache=index_cache,
        real_page_table=page_table,
        query_row_to_batch=torch.arange(rows, device=device, dtype=torch.int32),
        seqlens_per_query=seqlens,
    )
    logical = torch.topk(logits, topk, dim=1).indices
    physical = torch.gather(page_table, 1, logical // 64) * 64 + logical % 64
    for row in range(rows):
        assert set(output[row].tolist()) == set(physical[row].to(torch.int32).tolist())


def test_full_indexer_producer_replays_without_serving_allocations() -> None:
    require_b12x()
    torch.manual_seed(20260809)
    hidden, q_rank, heads, tokens = 4_096, 1_024, 64, 1
    wq_b, wq_b_scale = _block_fp8_weight(heads * 128, q_rank)
    weights_projection = (
        torch.randn((heads, hidden), device="cuda", dtype=torch.bfloat16) / 64
    ).contiguous()
    weights = dsv4_producer.pack_indexer_weights(wq_b, wq_b_scale, weights_projection)
    plan = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device="cuda",
            max_tokens=tokens,
            hidden=hidden,
            q_lora_rank=q_rank,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q_rank_values = (
        torch.randn((tokens, q_rank), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    hidden_states = (
        torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    cos_sin = torch.zeros((1, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1
    query = torch.empty((tokens, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
    head_weights = torch.empty((tokens, heads), device="cuda", dtype=torch.float32)
    binding = dsv4_producer.bind_indexer(
        plan,
        scratch=scratch,
        q_rank=q_rank_values,
        hidden_states=hidden_states,
        positions=positions,
        cos_sin_cache=cos_sin,
        query=query,
        head_weights=head_weights,
        weights=weights,
        expected_m=tokens,
    )

    dsv4_producer.run_indexer(binding=binding)
    torch.cuda.synchronize()
    eager_query = query.clone()
    eager_weights = head_weights.clone()
    assert bool(torch.isfinite(query.float()).all())
    assert bool(torch.isfinite(head_weights).all())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_producer.run_indexer(binding=binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(query, eager_query)
    assert torch.equal(head_weights, eager_weights)


def test_full_flash_plan_replays_without_serving_allocations() -> None:
    require_b12x()
    torch.manual_seed(20260806)
    hidden, q_rank, heads, tokens = 4096, 1024, 64, 1
    wq_a, wq_a_scale = _block_fp8_weight(q_rank, hidden)
    wq_b, wq_b_scale = _block_fp8_weight(heads * 512, q_rank)
    wkv, wkv_scale = _block_fp8_weight(512, hidden)
    q_norm = torch.ones((q_rank,), device="cuda", dtype=torch.bfloat16)
    kv_norm = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
    weights = dsv4_producer.pack_weights(
        wq_a,
        wq_a_scale,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        q_norm,
        kv_norm,
    )
    kv_weights = dsv4_producer.pack_kv_weights(wkv, wkv_scale, kv_norm)
    plan = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cuda",
            max_tokens=tokens,
            hidden=hidden,
            q_lora_rank=q_rank,
            heads=heads,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    hidden_states = (
        torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    slots = torch.zeros((tokens,), device="cuda", dtype=torch.int32)
    cos_sin = torch.zeros((1, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1
    cache = torch.zeros((1, DSV4_KV_PAGE_BYTES), device="cuda", dtype=torch.uint8)
    query = torch.empty((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    binding = dsv4_producer.bind(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=slots,
        cos_sin_cache=cos_sin,
        main_kv_cache=cache,
        query=query,
        weights=weights,
        expected_m=tokens,
    )
    kv_plan = dsv4_producer.plan_kv(plan.caps)
    (kv_spec,) = kv_plan.scratch_specs()
    kv_scratch = torch.empty(kv_spec.shape, dtype=kv_spec.dtype, device=kv_spec.device)
    kv_cache = torch.zeros_like(cache)
    kv_binding = dsv4_producer.bind_kv(
        kv_plan,
        scratch=kv_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=slots,
        cos_sin_cache=cos_sin,
        main_kv_cache=kv_cache,
        weights=kv_weights,
        expected_m=tokens,
    )

    dsv4_producer.run(binding=binding)
    dsv4_producer.run_kv(binding=kv_binding)
    torch.cuda.synchronize()
    eager_query = query.clone()
    eager_cache = cache.clone()
    assert bool(torch.isfinite(eager_query).all())
    torch.testing.assert_close(kv_cache, eager_cache, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_producer.run(binding=binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(query, eager_query, rtol=0, atol=0)
    torch.testing.assert_close(cache, eager_cache, rtol=0, atol=0)

    kv_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(kv_graph):
        dsv4_producer.run_kv(binding=kv_binding)
    for _ in range(3):
        kv_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(kv_cache, eager_cache, rtol=0, atol=0)
