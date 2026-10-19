"""EXLR8 device grouping fused with the ingress transform, without model weights."""
import pytest
import torch

from .exlr8_oracle import hadamard, make_wire


def ingress_buffers(rows, hidden, fmt, seed=408):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, generator=generator)
    signs = (torch.randint(2, (hidden,), generator=generator) * 2 - 1).char().cuda()
    if fmt == "fp8_k32":
        wire, dequant = make_wire(x.cuda())
        return wire, dequant.cpu(), signs
    x = x.bfloat16()
    return x.cuda(), x.double(), signs


def test_grouping_stable_packing_and_graph():
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot

    experts, topk, capacity, rows = 7, 3, 64, 64
    program = compile_exlr8_group_aot(experts=experts, top_k=topk, max_groups=capacity, hidden=128)
    x, _, signs = ingress_buffers(rows, 128, "fp8_k32")
    activation = torch.empty((rows, 128), dtype=torch.float16, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    groups = torch.full((capacity,), -99, dtype=torch.int32, device="cuda")
    slots = torch.full((capacity, 16), -99, dtype=torch.int32, device="cuda")
    inverse = torch.full_like(ids, -99)
    count = torch.empty(1, dtype=torch.int32, device="cuda")

    def launch(live=rows):
        program.launch(ids, x, signs, groups, slots, inverse, count, activation, scalars=(live, 7))

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

    program = compile_exlr8_group_aot(experts=2, top_k=1, max_groups=1, hidden=128)
    x, _, signs = ingress_buffers(64, 128, "fp8_k32")
    activation = torch.empty((64, 128), dtype=torch.float16, device="cuda")
    ids = torch.arange(64, dtype=torch.int32, device="cuda") % 2
    groups = torch.full((1,), -99, dtype=torch.int32, device="cuda")
    slots = torch.full((16,), -99, dtype=torch.int32, device="cuda")
    inverse = torch.full_like(ids, -99)
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    program.launch(ids, x, signs, groups, slots, inverse, count, activation, scalars=(64, 7))
    assert count.item() == -1
    assert (inverse == -1).all()
    assert (groups == -99).all() and (slots == -99).all()


@pytest.mark.parametrize("fmt", ["fp8_k32", "bf16"])
@pytest.mark.parametrize("hidden", [128, 384, 6144])
def test_ingress_reference_rows_ctas_and_graph(hidden, fmt):
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot

    rows = 64
    program = compile_exlr8_group_aot(experts=3, top_k=2, max_groups=128, hidden=hidden,
                                      input_format=fmt)
    x, exact, signs = ingress_buffers(rows, hidden, fmt)
    reference = hadamard((exact * signs.cpu()).reshape(rows, -1, 128)).reshape(rows, hidden)
    ids = (torch.arange(rows * 2, dtype=torch.int32, device="cuda") % 3).reshape(rows, 2)
    tables = [torch.empty(128, dtype=torch.int32, device="cuda"),
              torch.empty((128, 16), dtype=torch.int32, device="cuda"),
              torch.empty_like(ids), torch.empty(1, dtype=torch.int32, device="cuda")]
    output = torch.empty((rows + 1, hidden), dtype=torch.float16, device="cuda")
    for live in (1, 16, 63, 64):
        for limit in (1, 7, 188):
            output.fill_(float("nan"))
            program.launch(ids, x, signs, *tables, output, scalars=(live, limit))
            torch.testing.assert_close(output[:live].cpu().double(), reference[:live].half().double(),
                                       rtol=1e-3, atol=1e-7)
            assert torch.isnan(output[live:]).all()
            routes = ids[:live].flatten().tolist()
            assert tables[3].item() == sum(-(-routes.count(e) // 16) for e in set(routes))
            if live == rows and limit == 1:
                invariant = output[:rows].clone()
            if live == rows:
                assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    # Each row's transform is independent of the rows around it.
    for first in (0, 17, 40):
        output.fill_(float("nan"))
        # A fresh allocation keeps the shifted rows 16-byte aligned.
        program.launch(ids, x[first:].clone(), signs, *tables, output, scalars=(rows - first, 7))
        assert torch.equal(output[:rows - first].view(torch.int16), invariant[first:].view(torch.int16))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(ids, x, signs, *tables, output, scalars=(rows, 7))
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:rows].view(torch.int16), invariant.view(torch.int16))
    for invalid in (0, -1, rows + 1):
        output.fill_(float("nan"))
        program.launch(ids, x, signs, *tables, output, scalars=(invalid, 7))
        assert torch.isnan(output).all()


@pytest.mark.parametrize("kind,kwargs", [
    ("group", {"experts": 257}), ("group", {"top_k": 0}),
    ("group", {"max_groups": 2**31}), ("group", {"experts": True}),
    ("group", {"hidden": 129}), ("group", {"hidden": 2**30}),
    ("group", {"input_format": "e4m3"}), ("group", {"shared_expert": True}),
])
def test_invalid_geometry_rejected_before_compilation(kind, kwargs):
    from b12x.integration.cuteafd import exlr8_glue

    with pytest.raises(ValueError):
        getattr(exlr8_glue, f"compile_exlr8_{kind}_aot")(**kwargs)
