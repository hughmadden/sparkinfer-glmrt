"""Shared selected-position GQA: native D192 and independent provider groups."""
from __future__ import annotations

import math

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("head_dim", [192, 256])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("width", [512, 4104])
@pytest.mark.parametrize("heads_per_kv", [16, 32])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_selected_geometry_graph_matches_reference(head_dim, groups, width, heads_per_kv, kv_dtype):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged import _selected_forward as impl

    device = require_b12x()
    torch.manual_seed(274)
    rows, hq, hkv, splits = 4, heads_per_kv * 2, 2, 4
    q = torch.randn(rows, hq, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(3, 16, hkv, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k).to(kv_dtype)
    k = k.to(kv_dtype)
    descale = torch.ones(hkv, device=device) if kv_dtype == torch.float8_e4m3fn else None
    table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 31, dtype=torch.int64, device=device)
    selected = torch.full((rows, groups, width), -1, dtype=torch.int32, device=device)
    for group in range(groups):
        selected[:, group, :16] = torch.arange(group * 8, group * 8 + 16, device=device)
    # The last slot must participate, even with >4096 selected capacity.
    selected[:, :, -1] = 31
    partial = torch.empty(rows, splits, hq, head_dim, dtype=torch.float32, device=device)
    lse = torch.empty(rows, splits, hq, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    direct = torch.empty_like(q)

    def launch(n):
        impl.launch_sparse_gqa_split(
            query=q[:n], key_cache=k, value_cache=v, k_descale=descale, v_descale=descale,
            block_table=table, request_ids=requests[:n], selected_positions=selected[:n],
            query_positions=positions[:n], partial_output=partial[:n], partial_lse=lse[:n],
            softmax_scale=1 / math.sqrt(head_dim), splits=splits,
        )
        impl.launch_sparse_gqa_merge(
            partial_output=partial[:n], partial_lse=lse[:n], output=out[:n], rows=n, splits=splits,
        )
        impl.launch_selected_paged_gqa_direct(
            query=q[:n], key_cache=k, value_cache=v, k_descale=descale, v_descale=descale,
            block_table=table, request_ids=requests[:n], selected_positions=selected[:n],
            query_positions=positions[:n], output=direct[:n],
            softmax_scale=1 / math.sqrt(head_dim), kv_warps=1,
        )

    impl.clear_caches()
    launch(1)
    with kernel_resolution_guard("selected GQA live-row reuse"):
        for n in (1, rows):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(n)
            q.normal_()
            graph.replay()
            expected = torch.empty_like(out[:n], dtype=torch.float64)
            for head in range(hq):
                kv_head = head // (hq // hkv)
                group = kv_head // (hkv // groups)
                logical = selected[0, group]
                logical = logical[logical >= 0].long()
                physical = table[0, logical // 16].long()
                keys = k[physical, logical % 16, kv_head].double()
                values = v[physical, logical % 16, kv_head].double()
                scores = q[:n, head].double() @ keys.T / math.sqrt(head_dim)
                expected[:, head] = scores.softmax(-1) @ values
            torch.testing.assert_close(out[:n].double(), expected, atol=0.006, rtol=0.006)
            torch.testing.assert_close(direct[:n].double(), expected, atol=0.006, rtol=0.006)
            assert torch.isfinite(out[:n]).all()
            assert torch.count_nonzero(out[:n, :, 128:]) > 0
    impl.clear_caches()


def test_d192_selected_pool_offsets_above_int32():
    from b12x.attention.paged import _selected_forward as impl
    device = require_b12x()
    page_size, heads, dim = 16, 2, 192
    page_stride = page_size * heads * dim
    physical = (1 << 31) // page_stride + 1
    # Only the last page is initialized; address products exceed 2^31 elements.
    keys = torch.empty(physical + 1, page_size, heads, dim, dtype=torch.bfloat16, device=device)
    values = torch.empty_like(keys)
    keys[physical].fill_(0.25)
    values[physical, :, 0].fill_(0.5)
    values[physical, :, 1].fill_(-0.5)
    query = torch.ones(1, 32, dim, dtype=torch.bfloat16, device=device)
    table = torch.tensor([[physical]], dtype=torch.int32, device=device)
    selected = torch.full((1, 2, 512), -1, dtype=torch.int32, device=device)
    selected[:, :, :16] = torch.arange(16, device=device)
    output = torch.empty_like(query)
    impl.launch_selected_paged_gqa_direct(
        query=query, key_cache=keys, value_cache=values, k_descale=None, v_descale=None,
        block_table=table, request_ids=torch.zeros(1, dtype=torch.int32, device=device),
        selected_positions=selected, query_positions=torch.tensor([15], dtype=torch.int64, device=device),
        output=output, softmax_scale=dim**-0.5, kv_warps=2,
    )
    torch.testing.assert_close(output[:, :16], torch.full_like(output[:, :16], 0.5), atol=0, rtol=0)
    torch.testing.assert_close(output[:, 16:], torch.full_like(output[:, 16:], -0.5), atol=0, rtol=0)
