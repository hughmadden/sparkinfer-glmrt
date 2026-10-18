"""EXLR8 device grouping and hidden-domain transforms, without model weights."""
import pytest
import torch

from .exlr8_oracle import hadamard


def test_grouping_stable_packing_and_graph():
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot

    experts, topk, capacity, rows = 7, 3, 64, 64
    program = compile_exlr8_group_aot(experts=experts, top_k=topk, max_groups=capacity)
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    groups = torch.full((capacity,), -99, dtype=torch.int32, device="cuda")
    slots = torch.full((capacity, 16), -99, dtype=torch.int32, device="cuda")
    inverse = torch.full_like(ids, -99)
    count = torch.empty(1, dtype=torch.int32, device="cuda")

    def launch(live=rows):
        program.launch(ids, groups, slots, inverse, count, scalars=(live,))

    def check(host, live=rows):
        expected_groups, expected_slots = [], []
        expected_inverse = torch.full((live * topk,), -1, dtype=torch.int32)
        for expert in range(experts):
            routes = (host[:live].flatten() == expert).nonzero().flatten().tolist()
            for start in range(0, len(routes), 16):
                chunk = routes[start:start + 16]
                group = len(expected_groups)
                expected_groups.append(expert)
                expected_slots.append(chunk + [-1] * (16 - len(chunk)))
                for lane, route in enumerate(chunk):
                    expected_inverse[route] = group * 16 + lane
        n = len(expected_groups)
        assert count.item() == n
        assert torch.equal(groups[:n].cpu(), torch.tensor(expected_groups, dtype=torch.int32))
        assert torch.equal(slots[:n].cpu(), torch.tensor(expected_slots, dtype=torch.int32))
        assert torch.equal(inverse[:live].cpu().flatten(), expected_inverse)

    host = torch.arange(rows * topk, dtype=torch.int32).reshape(rows, topk) % experts
    ids.copy_(host)
    for live in (1, 2, 16, 63, 64):
        inverse.fill_(-99)
        launch(live)
        check(host, live)
        assert (inverse[live:] == -99).all()
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for mode in range(3):
        host = (torch.arange(rows * topk).reshape(rows, topk) * mode + mode).remainder(experts).int()
        ids.copy_(host)
        graph.replay()
        check(host)
    for invalid in (-1, experts):
        ids.fill_(0)
        ids[0, 0] = invalid
        groups.fill_(-99)
        slots.fill_(-99)
        graph.replay()
        assert count.item() == -1
        assert (inverse == -1).all()
        assert (groups == -99).all() and (slots == -99).all()


def test_grouping_overflow_does_not_publish_tables():
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot

    program = compile_exlr8_group_aot(experts=2, top_k=1, max_groups=1)
    ids = torch.arange(64, dtype=torch.int32, device="cuda") % 2
    groups = torch.full((1,), -99, dtype=torch.int32, device="cuda")
    slots = torch.full((16,), -99, dtype=torch.int32, device="cuda")
    inverse = torch.full_like(ids, -99)
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    program.launch(ids, groups, slots, inverse, count, scalars=(64,))
    assert count.item() == -1
    assert (inverse == -1).all()
    assert (groups == -99).all() and (slots == -99).all()


@pytest.mark.parametrize("hidden", [128, 384, 6144])
def test_rotation_reference_rows_ctas_and_graph(hidden):
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_rotate_aot

    rows = 65
    program = compile_exlr8_rotate_aot(hidden=hidden, max_rows=rows)
    generator = torch.Generator().manual_seed(408)
    x = torch.randn(rows, hidden, generator=generator).bfloat16().cuda()
    signs = (torch.randint(2, (hidden,), generator=generator) * 2 - 1).char().cuda()
    output = torch.empty((rows + 1, hidden), dtype=torch.float16, device="cuda")
    reference = hadamard((x.cpu().double() * signs.cpu()).reshape(rows, -1, 128)).reshape(rows, hidden)
    for live in (1, 16, 64, 65):
        for limit in (1, 7, 188):
            output.fill_(float("nan"))
            program.launch(x, signs, output, scalars=(live, limit))
            torch.testing.assert_close(output[:live].cpu().double(), reference[:live], atol=0.002, rtol=0.001)
            assert torch.isnan(output[live:]).all()
            if live == rows and limit == 1:
                invariant = output[:rows].clone()
            if live == rows:
                assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(x, signs, output, scalars=(rows, 7))
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    for invalid in (0, -1, rows + 1):
        output.fill_(float("nan"))
        program.launch(x, signs, output, scalars=(invalid, 7))
        assert torch.isnan(output).all()


@pytest.mark.parametrize("hidden", [128, 384, 6144])
def test_reduce_reference_rows_ctas_and_graph(hidden):
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_reduce_aot

    rows, ranks = 65, 6
    program = compile_exlr8_reduce_aot(hidden=hidden, max_ranks=ranks, max_rows=rows)
    generator = torch.Generator().manual_seed(409)
    planes = [torch.randn(rows, hidden, generator=generator).cuda() for _ in range(ranks)]
    addresses = torch.tensor([p.data_ptr() for p in planes], dtype=torch.int64, device="cuda")
    shared = torch.randn(rows, hidden, generator=generator).cuda()
    signs = (torch.randint(2, (hidden,), generator=generator) * 2 - 1).char().cuda()
    residual = torch.randn(rows, hidden, generator=generator).bfloat16().cuda()
    output = torch.empty((rows + 1, hidden), dtype=torch.bfloat16, device="cuda")
    for n in (0, 1, ranks):
        summed = torch.zeros_like(shared, device="cpu")
        for plane in planes[:n]:
            summed += plane.cpu()
        summed += shared.cpu()
        reference = hadamard(summed.double().reshape(rows, -1, 128)).reshape(rows, hidden)
        reference = reference * signs.cpu() + residual.cpu().double()
        for live in (1, 16, 64, rows):
            for limit in (1, 7, 188):
                output.fill_(float("nan"))
                program.launch(addresses, shared, signs, residual, output, scalars=(live, n, limit))
                actual = output[:live].cpu().double()
                torch.testing.assert_close(actual, reference[:live], atol=0.04, rtol=0.005)
                assert torch.isnan(output[live:]).all()
                if live == rows and limit == 1:
                    invariant = output[:rows].clone()
                if live == rows:
                    assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(addresses, shared, signs, residual, output, scalars=(rows, ranks, 7))
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    for live, n in ((rows + 1, ranks), (-1, ranks), (rows, -1), (rows, ranks + 1)):
        output.fill_(float("nan"))
        program.launch(addresses, shared, signs, residual, output, scalars=(live, n, 7))
        assert torch.isnan(output).all()


@pytest.mark.parametrize("kind,kwargs", [
    ("group", {"experts": 257}), ("group", {"top_k": 0}),
    ("group", {"max_groups": 2**31}), ("group", {"experts": True}),
    ("rotate", {"hidden": 129}), ("rotate", {"max_rows": 0}),
    ("rotate", {"hidden": 2**30, "max_rows": 4096}),
    ("reduce", {"max_ranks": -1}), ("reduce", {"max_rows": 2**31}),
])
def test_invalid_geometry_rejected_before_compilation(kind, kwargs):
    from b12x.integration.cuteafd import exlr8_glue

    with pytest.raises(ValueError):
        getattr(exlr8_glue, f"compile_exlr8_{kind}_aot")(**kwargs)
