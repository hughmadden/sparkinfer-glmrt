"""Raw-pointer attention ABIs preserve shared kernels and live-row reuse."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import check_export


def test_region_compress_pointer_abi_preserves_incomplete():
    require_b12x()
    from b12x.attention.paged._weighted_regions import compress_regions
    from b12x.integration.cuteafd.region_attention import compile_region_compress_aot

    torch.manual_seed(274)
    keys = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn_like(keys)
    starts = torch.arange(0, 64, 8, device="cuda", dtype=torch.int32)
    counts = torch.tensor([8, 8, 8, 8, 8, 8, 7, 0], device="cuda", dtype=torch.int32)
    output = torch.full((8, 256), 3., device="cuda").to(torch.float8_e4m3fn)
    raw = torch.full((8, 256), 3., device="cuda")
    expected, expected_raw = output.clone(), raw.clone()
    compress_regions(keys=keys, logits=logits, starts=starts, counts=counts,
                     output=expected, unrounded=expected_raw)
    program = compile_region_compress_aot()
    program.launch(keys, logits, starts, counts, output, raw, scalars=[8, 64])
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(raw, expected_raw)
    assert torch.equal(raw[6:], torch.full_like(raw[6:], 3.))


@pytest.mark.parametrize("groups", [2, 4])
def test_region_score_pointer_abi_signed_visibility_and_graph(groups):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._weighted_regions import score_regions
    from b12x.integration.cuteafd.region_attention import compile_region_scores_aot

    torch.manual_seed(275)
    rows, capacity = 4, 67
    query = torch.randn(rows, groups, 4, 256, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, groups, 4, device="cuda")
    summaries = torch.randn(2, capacity, 256, device="cuda").to(torch.float8_e4m3fn)
    requests = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 8, 17, 8 * capacity], device="cuda", dtype=torch.int64)
    expected = torch.empty(rows, groups, capacity, device="cuda")
    score_regions(query=query, weights=weights, summaries=summaries, requests=requests,
                  positions=positions, output=expected)
    output = torch.full_like(expected, 123.)
    program = compile_region_scores_aot(groups=groups, capacity=capacity)
    program.launch(query, weights, summaries, requests, positions, output, scalars=[rows, 2])
    assert torch.equal(output, expected)
    with kernel_resolution_guard("region score pointer ABI graph"):
        for live in (1, rows):
            output.fill_(123.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(query, weights, summaries, requests, positions, output, scalars=[live, 2])
            graph.replay()
            assert torch.equal(output[:live], expected[:live])
            assert torch.equal(output[live:], torch.full_like(output[live:], 123.))


def _pack_reference(values):
    # CPU division is deliberate: the GPU reciprocal-multiply oracle can move
    # a quantization midpoint relative to the native div.rn scale contract.
    from b12x.integration.cuteafd.pointwise import fp8_kv_plane_bytes
    source = values.cpu().float()
    heads, dim = source.shape[-2:]
    scale = source.abs().amax(dim=-1) / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (source * (1.0 / scale).unsqueeze(-1)).to(torch.float8_e4m3fn)
    record = torch.zeros((*source.shape[:-2], fp8_kv_plane_bytes(heads, dim)), dtype=torch.uint8)
    record[..., :heads * dim] = codes.view(torch.uint8).flatten(-2)
    record[..., heads * dim:heads * (dim + 4)] = scale.contiguous().view(torch.uint8)
    dequantized = (codes.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
    return record.to(values.device), dequantized.to(values.device)


@pytest.mark.parametrize("heads,dim", [(2, 192), (4, 192), (2, 256)])
def test_planar_fp8_writer_exact_bytes_zero_tail_and_graph(heads, dim):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.pointwise import compile_pack_fp8_kv_plane_aot

    torch.manual_seed(277)
    source = torch.randn(5, heads, dim, device="cuda", dtype=torch.bfloat16)
    source[0].zero_()
    source[1] *= 16
    source[2].zero_()
    source[2, :, 0] = 3.0625
    source[2, :, 1] = 0.73828125
    source[2, :, 2] = -0.73828125
    expected, _ = _pack_reference(source)
    out = torch.full_like(expected, 123)
    program = compile_pack_fp8_kv_plane_aot(kv_heads=heads, head_dim=dim)
    program.launch(source, out, scalars=[5])
    assert torch.equal(out, expected)
    with kernel_resolution_guard("planar FP8 writer graph"):
        for live in (1, 5):
            out.fill_(123)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(source, out, scalars=[live])
            graph.replay()
            assert torch.equal(out[:live], expected[:live])
            assert torch.equal(out[live:], torch.full_like(out[live:], 123))


@pytest.mark.parametrize("heads,kv_heads", [(32, 2), (64, 4)])
@pytest.mark.parametrize("width", [512, 4104])
@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
@pytest.mark.parametrize("amplitude", [1.0, 16.0])
def test_selected_gqa_pointer_abi_d192_and_live_rows(heads, kv_heads, width, kv_format, amplitude):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
    from b12x.integration.cuteafd.region_attention import compile_selected_gqa_aot

    torch.manual_seed(276)
    rows, page_rows = 4, 64
    groups = 1 if width == 512 else kv_heads
    query = torch.randn(rows, heads, 192, device="cuda", dtype=torch.bfloat16) * amplitude
    keys = torch.randn(3, page_rows, kv_heads, 192, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    stored_keys, stored_values = keys, values
    if kv_format == "fp8":
        stored_keys, keys = _pack_reference(keys)
        stored_values, values = _pack_reference(values)
    table = torch.tensor([[2, 0], [0, 1]], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    positions = torch.full((rows,), 127, device="cuda", dtype=torch.int64)
    selected = torch.full((rows, groups, width), -1, device="cuda", dtype=torch.int32)
    for group in range(groups):
        selected[:, group, :16] = torch.arange(group * 8, group * 8 + 16, device="cuda")
    selected[:, :, -1] = 127
    expected = torch.empty_like(query)
    launch_selected_paged_gqa_direct(query=query, key_cache=keys, value_cache=values,
        k_descale=None, v_descale=None, block_table=table, request_ids=requests,
        selected_positions=selected, query_positions=positions, output=expected,
        softmax_scale=192**-0.5, kv_warps=2)
    output = torch.full_like(query, 123.)
    program = compile_selected_gqa_aot(query_heads=heads, kv_heads=kv_heads,
        page_rows=page_rows, selection_width=width, selection_groups=groups, kv_format=kv_format)

    def launch(live):
        program.launch(query, stored_keys, stored_values, None, None, table, requests, selected,
            positions, None, None, output, scalars=[3, 2, 2, 192**-0.5, live, 1])

    launch(rows)
    assert torch.equal(output, expected)
    with kernel_resolution_guard("selected GQA pointer ABI graph"):
        for live in (1, rows):
            output.fill_(123.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(live)
            graph.replay()
            assert torch.equal(output[:live], expected[:live])
            assert torch.equal(output[live:], torch.full_like(output[live:], 123.))


def test_planar_fp8_selected_cache_offsets_past_i32():
    require_b12x()
    from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
    from b12x.integration.cuteafd.pointwise import fp8_kv_plane_bytes
    from b12x.integration.cuteafd.region_attention import compile_selected_gqa_aot

    heads, kv_heads, dim, page_rows = 32, 2, 192, 64
    stride = fp8_kv_plane_bytes(kv_heads, dim)
    tail = (1 << 31) // (page_rows * stride) + 1
    # Leave the large pools uninitialized; only the addressed tail page is live.
    keys = torch.randn(1, page_rows, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    packed_k, unpacked_k = _pack_reference(keys)
    packed_v, unpacked_v = _pack_reference(values)
    pool_k = torch.empty((tail + 1, page_rows, stride), device="cuda", dtype=torch.uint8)
    pool_v = torch.empty_like(pool_k)
    pool_k[tail].copy_(packed_k[0])
    pool_v[tail].copy_(packed_v[0])
    query = torch.randn(1, heads, dim, device="cuda", dtype=torch.bfloat16) * 16
    requests = torch.zeros(1, device="cuda", dtype=torch.int32)
    positions = torch.tensor([63], device="cuda", dtype=torch.int64)
    selected = torch.full((1, 1, 512), -1, device="cuda", dtype=torch.int32)
    selected[0, 0, :page_rows] = torch.arange(page_rows, device="cuda")
    expected = torch.empty_like(query)
    launch_selected_paged_gqa_direct(query=query, key_cache=unpacked_k, value_cache=unpacked_v,
        k_descale=None, v_descale=None,
        block_table=torch.zeros((1, 1), device="cuda", dtype=torch.int32), request_ids=requests,
        selected_positions=selected, query_positions=positions, output=expected,
        softmax_scale=dim**-0.5, kv_warps=2)
    output = torch.empty_like(query)
    program = compile_selected_gqa_aot(query_heads=heads, kv_heads=kv_heads,
        selection_width=512, selection_groups=1, kv_format="fp8")
    table = torch.tensor([[tail]], device="cuda", dtype=torch.int32)
    program.launch(query, pool_k, pool_v, None, None, table, requests, selected,
        positions, None, None, output, scalars=[tail + 1, 1, 1, dim**-0.5, 1, 1])
    assert torch.equal(output, expected)


@pytest.mark.parametrize("operation", ["compress", "scores2", "scores4", "swa32", "csa64",
    "pack2", "pack4", "swa32_fp8", "csa64_fp8"])
def test_region_attention_export_abis(tmp_path, operation):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import region_attention as r
    from b12x.integration.cuteafd.pointwise import compile_pack_fp8_kv_plane_aot

    builders = {
        "compress": r.compile_region_compress_aot,
        "scores2": lambda: r.compile_region_scores_aot(groups=2, capacity=67),
        "scores4": lambda: r.compile_region_scores_aot(groups=4, capacity=67),
        "swa32": lambda: r.compile_selected_gqa_aot(query_heads=32, kv_heads=2,
            selection_width=512, selection_groups=1),
        "csa64": lambda: r.compile_selected_gqa_aot(query_heads=64, kv_heads=4),
        "pack2": lambda: compile_pack_fp8_kv_plane_aot(kv_heads=2),
        "pack4": lambda: compile_pack_fp8_kv_plane_aot(kv_heads=4),
        "swa32_fp8": lambda: r.compile_selected_gqa_aot(query_heads=32, kv_heads=2,
            selection_width=512, selection_groups=1, kv_format="fp8"),
        "csa64_fp8": lambda: r.compile_selected_gqa_aot(query_heads=64, kv_heads=4, kv_format="fp8"),
    }
    with exportable_compilation():
        program = builders[operation]()
    check_export(program, tmp_path, "region_" + operation)
