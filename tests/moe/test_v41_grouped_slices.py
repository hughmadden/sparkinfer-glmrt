"""One compiled fused slice kernel across live expert counts and route groups."""

import json
import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from b12x.moe.fused_moe._impl import (
    _logical_weight_to_w4a8_rp_inplace,
    _e8m0_scale_to_w4a8_sfb_inplace,
)
from tests.moe.test_v41_expert_numerics import reference


def _routing_cases():
    return [
        ("one", torch.arange(6).reshape(1, 6)),
        ("shared2", torch.arange(6).expand(2, 6).clone()),
        (
            "mixed6",
            torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 6, 7, 8, 9],
                    [0, 2, 6, 10, 11, 12],
                    [0, 3, 7, 13, 14, 15],
                    [1, 4, 8, 16, 17, 18],
                    [2, 5, 9, 19, 20, 21],
                ]
            ),
        ),
        ("shared16", torch.arange(6).expand(16, 6).clone()),
        ("shared80", torch.arange(6).expand(80, 6).clone()),
        ("return_one", torch.arange(6).reshape(1, 6)),
    ]


def _metadata(ids):
    tasks = []
    pairs = []
    for expert in ids.unique(sorted=True).tolist():
        locations = (ids == expert).nonzero().tolist()
        for begin in range(0, len(locations), 16):
            chunk = locations[begin : begin + 16]
            tasks.append(
                [expert, len(chunk), len(pairs)]
                + [row for row, _ in chunk]
                + [-1] * (16 - len(chunk))
            )
            pairs.extend(chunk)
    return torch.tensor(tasks, dtype=torch.int32), torch.tensor(pairs, dtype=torch.long)


@pytest.mark.parametrize("h,n,topk", [(5120, 576, 6), (5120, 2304, 3), (4096, 512, 6)])
@pytest.mark.parametrize("width", [64, 128, 192])
def test_grouped_slices(width, h, n, topk):
    _check_grouped_slices(width, n=n, topk=topk, h=h)


def _check_grouped_slices(width, after_case=None, *, n=576, topk=6, h=5120,
                          capacity=80, row_counts=None, experts=None):
    if -(-n // width) * width > (n + 127) // 128 * 128:
        pytest.skip("slice width does not tile this intermediate within its packed storage")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")
    torch.manual_seed(4164)
    experts = experts or (128 if n == 2304 else 32)
    assert capacity > 0 and 0 < topk <= experts
    weights = {}
    scales = {}
    for name, shape in [
        ("w1", (experts, n, h // 2)),
        ("w3", (experts, n, h // 2)),
        ("w2", (experts, h, n // 2)),
    ]:
        weights[name] = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
        scales[name] = torch.randint(
            121, 124, (*shape[:-1], shape[-1] // 16), device="cuda", dtype=torch.uint8
        )
    w13 = _logical_weight_to_w4a8_rp_inplace(
        torch.cat([weights["w3"], weights["w1"]], 1),
        size_k=h,
        size_n=n * 2,
        gated_half_rows=n,
    )
    s13 = _e8m0_scale_to_w4a8_sfb_inplace(
        torch.cat([scales["w3"], scales["w1"]], 1),
        weight_E=experts,
        rows=n * 2,
        k_dim=h,
        gated_half_rows=n,
    )
    w2 = _logical_weight_to_w4a8_rp_inplace(weights["w2"].clone(), size_k=n, size_n=h)
    s2 = _e8m0_scale_to_w4a8_sfb_inplace(
        scales["w2"].clone(), weight_E=experts, rows=h, k_dim=n
    )
    packed = [t.view(torch.uint32).flatten() for t in [w13, s13, w2, s2]]
    x = torch.randn(capacity, h, device="cuda").mul_(0.5).bfloat16()
    wire = torch.empty((capacity, h + h // 32), device="cuda", dtype=torch.uint8)
    qa = wire[:, :h].view(torch.uint32)
    qs = wire[:, h:]
    group_capacity = min(capacity * topk, experts + max(capacity * topk - experts, 0) // 16)
    metadata = torch.full((group_capacity, 19), -1, device="cuda", dtype=torch.int32)
    routing = torch.empty(capacity * topk, device="cuda")
    out = torch.empty(((n + width - 1) // width, capacity * topk, h), device="cuda")
    args = [from_dlpack(t, assumed_align=16) for t in [qa, qs, *packed, routing, out]]
    meta_view = from_dlpack(metadata, assumed_align=16)
    compiled = cute.compile(
        V41FusedSliceKernel(width, grouped=True, intermediate=n, hidden=h),
        *args,
        cutlass.Int32(capacity),
        current_cuda_stream(),
        meta_view,
        cutlass.Int32(group_capacity),
    )
    compiled(*args, capacity, current_cuda_stream(), meta_view, group_capacity)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, capacity, current_cuda_stream(), meta_view, group_capacity)
    cases = (_routing_cases() if row_counts is None else
             [(f"shared{m}", torch.arange(topk).expand(m, topk).clone())
              for m in row_counts])
    cases = [(name, ids) for name, ids in cases if ids.shape[0] <= capacity]
    if n == 2304 and row_counts is None:
        cases += [(f"draft{m}", torch.arange(m * topk).reshape(m, topk) % experts)
                  for m in (5, 15, 40)]
    for case_index, (case, ids) in enumerate(cases):
        ids = ids[:, :topk]
        ids = experts - 1 - ids if case_index % 2 else ids
        m = ids.shape[0]
        tasks, pairs = _metadata(ids)
        groups = len(tasks)
        routes = len(pairs)
        metadata.fill_(-1)
        metadata[:groups].copy_(tasks)
        route_weights = torch.rand(m, topk, device="cuda")
        route_weights.mul_(1.5 / route_weights.sum(-1, keepdim=True))
        pair_gpu = pairs.cuda()
        routing.fill_(float("nan"))
        routing[:routes].copy_(route_weights[pair_gpu[:, 0], pair_gpu[:, 1]])
        x.mul_(-0.75)
        blocks = x.float().reshape(capacity, h // 32, 32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
        qa.copy_(
            (blocks / torch.exp2(exponent)[..., None])
            .to(torch.float8_e4m3fn)
            .view(torch.uint32)
            .reshape(capacity, h // 4)
        )
        qs.copy_((exponent + 127).to(torch.uint8))
        out.fill_(12345)
        # Change metadata and route weights inside a fixed captured launch.
        # Reverse each expert-local row run, preserving disjoint output spans.
        changed = tasks.clone()
        changed[:, 0] = experts - 1 - changed[:, 0]
        ids = experts - 1 - ids
        new_pairs = pairs.clone()
        for task in changed:
            count, base = int(task[1]), int(task[2])
            task[3 : 3 + count] = task[3 : 3 + count].flip(0)
            new_pairs[base : base + count] = pairs[base : base + count].flip(0)
        metadata[:groups].copy_(changed)
        pair_gpu.copy_(new_pairs)
        route_weights.mul_(0.875)
        routing[:routes].copy_(route_weights[pair_gpu[:, 0], pair_gpu[:, 1]])
        out.fill_(12345)
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == before
        partial = out[:, :routes].sum(0).bfloat16().float()
        actual = torch.zeros(m, h, device="cuda")
        actual.index_add_(0, pair_gpu[:, 0], partial)
        expected = reference(x[:m], ids.cuda().int(), route_weights, weights, scales)
        rel = ((actual - expected).norm() / expected.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        assert rel < 0.01 and cosine > 0.9999, (case, width, rel, cosine)
        assert bool((out[:, routes:] == 12345).all())
        print(
            json.dumps(
                dict(
                    case=case,
                    width=width,
                    intermediate=n,
                    topk=topk,
                    rows=m,
                    experts=int(ids.unique().numel()),
                    groups=groups,
                    routes=routes,
                    rel_l2=rel,
                    cosine=cosine,
                )
            ),
            flush=True,
        )
        if after_case is not None:
            after_case(
                case=case,
                native_inputs=(packed, wire, ids, route_weights),
                args=args,
                meta_view=meta_view,
                out=out,
                expected=expected,
                pair_gpu=pair_gpu,
                rows=m,
                hidden=h,
                intermediate=n,
                topk=topk,
                capacity=capacity,
                experts=experts,
                group_capacity=group_capacity,
                groups=groups,
                routes=routes,
                baseline_graph=graph,
                baseline_compiled=compiled,
            )
    graph.reset()


@pytest.mark.parametrize("h,n", [(4096, 512), (7168, 768)])
def test_dsv4_canonical_fc2_split_row_invariance(h, n):
    """Frozen input rows retain every FP32 route bit across capacity boundaries."""
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline

    def check(**case):
        packed, wire, ids, route_weights = case["native_inputs"]
        topk, experts = 6, case["experts"]
        wire = wire[:1].repeat(256, 1)
        ids = ids[:1].cuda().int().repeat(256, 1).flatten()
        routing = route_weights[:1].repeat(256, 1).flatten()
        qa, qs = wire[:, :h].view(torch.uint32), wire[:, h:]
        expected = None
        for capacity, rows in [(1, (1,)), (16, (1, 2, 7, 8, 15, 16)),
                               (80, (1, 16, 17, 64, 80)), (256, (1, 80, 81, 255, 256))]:
            for splits in (1, 2):
                routes = capacity * topk
                partial = torch.empty((n // 128, routes, h), device="cuda")
                output = torch.empty((routes, h), device="cuda")
                buffers = [qa, qs, *packed, ids, routing,
                           torch.empty(1, device="cuda", dtype=torch.int32),
                           torch.empty(experts * routes, device="cuda", dtype=torch.int32),
                           torch.empty(experts, device="cuda", dtype=torch.int32),
                           torch.empty((experts, 2), device="cuda", dtype=torch.int32),
                           torch.empty((routes, 19), device="cuda", dtype=torch.int32),
                           torch.empty(routes, device="cuda"),
                           torch.empty(routes, device="cuda", dtype=torch.int32), partial, output]
                views = [from_dlpack(t, assumed_align=16) for t in buffers]
                compiled = cute.compile(V41SlicePipeline(capacity, 128, experts=experts,
                    hidden=h, intermediate=n, topk=topk, fc2_n_splits=splits),
                    *views, cutlass.Int32(capacity), current_cuda_stream())
                for m in rows:
                    output.fill_(12345)
                    compiled(*views, m, current_cuda_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        compiled(*views, m, current_cuda_stream())
                    before = torch.cuda.memory_allocated()
                    graph.replay()
                    assert before == torch.cuda.memory_allocated()
                    live = output[:m * topk].reshape(m, topk, h)
                    assert torch.isfinite(live).all() and live.norm() > 0
                    if expected is None:
                        expected = live[0].clone()
                    assert torch.equal(live.view(torch.int32),
                                       expected.view(torch.int32).expand_as(live)), (h, n, capacity, splits, m)
                    assert bool((output[m * topk:] == 12345).all())
                    graph.reset()
                del compiled, views, buffers

    _check_grouped_slices(128, check, n=n, h=h, capacity=1, row_counts=[1])
