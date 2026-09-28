"""Device-side live count, stable grouping and stale-capacity replay checks."""

import pytest
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan


@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3), (256, 6)])
@pytest.mark.parametrize("capacity", [1, 16, 80, 4096])
def test_route_plan(capacity, experts, topk):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    r = capacity * topk
    specs = [
        (r, torch.int32),
        (r, torch.float32),
        (1, torch.int32),
        (experts * r, torch.int32),
        (experts, torch.int32),
        ((experts, 2), torch.int32),
        ((r, 19), torch.int32),
        (r, torch.float32),
        (r, torch.int32),
    ]
    tensors = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs]
    ids, weights, live, packed, counts, prefixes, meta, grouped, inverse = tensors
    ids.zero_()
    weights.fill_(1)
    live.fill_(capacity)
    args = [from_dlpack(t) for t in tensors]
    compiled = cute.compile(V41RoutePlan(capacity, experts, topk), *args, current_cuda_stream())
    compiled(*args, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, current_cuda_stream())
    pointers = [t.data_ptr() for t in tensors]
    generator = torch.Generator().manual_seed(41)
    for case, rows in enumerate(
        [
            capacity,
            1,
            0,
            min(2, capacity),
            min(6, capacity),
            min(16, capacity),
            capacity,
            1,
        ]
    ):
        source = torch.randint(-2, experts + 2, (r,), generator=generator, dtype=torch.int32)
        if case == 0:
            source.fill_(experts - 1)  # duplicates and long runs, even within one row
        if case == 6:
            source.fill_(0)
            source[:min(r, experts)] = torch.arange(min(r, experts))
        values = torch.randn(r, generator=generator)
        ids.copy_(source)
        weights.copy_(values)
        live.fill_(rows)
        meta.fill_(9876)
        inverse.fill_(9876)
        grouped.fill_(float("nan"))
        graph.replay()
        tasks, permutation = [], []
        for expert in range(experts):
            locations = (source[: rows * topk] == expert).nonzero().flatten().tolist()
            for start in range(0, len(locations), 16):
                chunk = locations[start : start + 16]
                tasks.append(
                    [expert, len(chunk), len(permutation)]
                    + [p // topk for p in chunk]
                    + [-1] * (16 - len(chunk))
                )
                permutation.extend(chunk)
        # The pipeline launches this tight worst-case group bound from live rows.
        bound = min(rows * topk, experts + max(rows * topk - experts, 0) // 16)
        assert len(tasks) <= bound
        n = len(permutation)
        expected_inverse = torch.full((r,), -1, dtype=torch.int32)
        expected_inverse[permutation] = torch.arange(n, dtype=torch.int32)
        assert torch.equal(inverse.cpu(), expected_inverse)
        assert torch.equal(grouped[:n].cpu(), values[permutation])
        if tasks:
            assert torch.equal(
                meta[: len(tasks)].cpu(), torch.tensor(tasks, dtype=torch.int32)
            )
        assert (meta[len(tasks) :, 1] == 0).all()
        assert [t.data_ptr() for t in tensors] == pointers


@pytest.mark.parametrize("h,n,topk", [(5120, 576, 6), (5120, 2304, 3), (4096, 512, 6)])
@pytest.mark.parametrize("width", [64, 128, 192])
def test_inverse_reduce(width, h, n, topk):
    from b12x.moe._shared.kernels.v41_route_plan import V41SliceReduce

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capacity = 16
    routes = capacity * topk
    planes = (n + width - 1) // width
    source = torch.randn(planes, routes, h, device="cuda")
    dest = torch.empty(routes, h, device="cuda")
    inverse = torch.empty(routes, dtype=torch.int32, device="cuda")
    live = torch.empty(1, dtype=torch.int32, device="cuda")
    args = [from_dlpack(t) for t in [source, dest, inverse, live]]
    fn = cute.compile(V41SliceReduce(width, capacity, topk, intermediate=n, hidden=h), *args, current_cuda_stream())
    live.zero_()
    fn(*args, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args, current_cuda_stream())
    for rows in [16, 1, 0, 6, 2, 16, 1]:
        live.fill_(rows)
        mapping = torch.randperm(routes, device="cuda").int()
        mapping[::7] = -1
        inverse.copy_(mapping)
        source.mul_(-0.75)
        dest.fill_(12345)
        expected = torch.zeros(rows * topk, h, device="cuda")
        valid = mapping[: rows * topk] >= 0
        for plane in range(planes):
            expected[valid] += source[plane, mapping[: rows * topk][valid].long()]
        graph.replay()
        assert torch.equal(dest[: rows * topk], expected)
        assert (dest[rows * topk :] == 12345).all()
