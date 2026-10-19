"""Bitwise K1 contract for expert-major EXLR8 prefill."""
import os

import pytest
import torch

from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from .exlr8_oracle import Case, make_wire


def buffers(rows, hidden, units, topk, capacity, group_rows):
    return dict(group_expert=torch.empty(capacity, dtype=torch.int32, device="cuda"),
                route_slots=torch.empty((capacity, group_rows), dtype=torch.int32, device="cuda"),
                inverse=torch.empty((rows, topk), dtype=torch.int32, device="cuda"),
                count=torch.empty(1, dtype=torch.int32, device="cuda"),
                activation=torch.empty((rows, hidden), dtype=torch.float16, device="cuda"),
                fc1=torch.empty((capacity, units, 2, (hidden+1023)//1024, group_rows, 128), device="cuda"),
                intermediate=torch.empty((capacity, units, group_rows, 128), dtype=torch.float16, device="cuda"),
                fc2=torch.empty((capacity, group_rows, hidden), device="cuda"),
                output=torch.empty((rows+1, hidden), dtype=torch.bfloat16, device="cuda"))


def programs(hidden, units, topk, capacity, group_rows, grouped):
    return (compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=capacity,
                                   hidden=hidden, max_rows=4096 if grouped else 64,
                                   group_rows=group_rows),
            compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=topk,
                                  max_groups=capacity, device_count=True,
                                  grouped=grouped, group_rows=group_rows))


def run(program, state, wire, ids, weights, pointers, bits, signs, rows, limit):
    group, k1 = program
    group.launch(ids, wire, signs[0], state["group_expert"], state["route_slots"],
                 state["inverse"], state["count"], state["activation"], scalars=(rows, limit))
    k1.launch(state["activation"], pointers, bits, state["group_expert"], state["route_slots"],
              state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
              signs[1], state["output"], state["count"], scalars=(rows, limit))


@pytest.mark.parametrize("group_rows", [16, 32, 64])
@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="checkpoint fixture required")
def test_checkpoint_grouped_matches_chunks(group_rows):
    torch.cuda.set_per_process_memory_fraction(0.25)
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=1152, units=3)
    rows, hidden, units, topk = 129, case.h, case.u, case.topk
    gen = torch.Generator().manual_seed(408)
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen).cuda() * 0.02)
    ids = (torch.arange(rows * topk, device="cuda").reshape(rows, topk) % 4).int()
    weights = torch.rand((rows, topk), generator=gen).cuda() / topk
    signs = (case.signs_in, case.signs_out)
    base = buffers(64, hidden, units, topk, 128, 16)
    dec = programs(hidden, units, topk, 128, 16, False)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        live = min(rows-first, 64)
        run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
        expected[first:first+live].copy_(base["output"][:live])
    state = buffers(rows, hidden, units, topk, 128, group_rows)
    grouped = programs(hidden, units, topk, 128, group_rows, True)
    for live in (1, 8, 16, 17, 32, 63, 64, 65, 127, 128, 129):
        for limit in (1, 48, 188):
            state["output"].fill_(float("nan"))
            run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, live, limit)
            assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16))
            assert torch.isnan(state["output"][live:]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
    for _ in range(2):
        weights.mul_(0.5)
        graph.replay()
        actual = state["output"][:rows].clone()
        for first in range(0, rows, 64):
            live = min(rows-first, 64)
            run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
            assert torch.equal(actual[first:first+live].view(torch.int16), base["output"][:live].view(torch.int16))


def test_every_live_row_count_matches_decode():
    torch.cuda.set_per_process_memory_fraction(0.25)
    hidden, units, topk, rows = 128, 1, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.empty((4, units, 3, 3), dtype=torch.int64)
    bits = torch.empty((4, units, 3), dtype=torch.int32)
    for e in range(4):
        for p in range(3):
            width = 2 + (e+p) % 3
            ts = (torch.randint(256, (hidden*16*width,), generator=gen, dtype=torch.uint8, device="cuda"),
                  torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                  torch.full((hidden//128,), 0.1, dtype=torch.float16, device="cuda"))
            owners.extend(ts)
            pointers[e, 0, p] = torch.tensor([t.data_ptr() for t in ts])
            bits[e, 0, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows*topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk
    base, state = buffers(64, hidden, units, topk, 128, 16), buffers(rows, hidden, units, topk, 132, 64)
    dec, grouped = programs(hidden, units, topk, 128, 16, False), programs(hidden, units, topk, 132, 64, True)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        run(dec, base, wire[first:], ids[first:], weights[first:], pointers, bits, signs, 64, 48)
        expected[first:first+64].copy_(base["output"][:64])
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    live_counts = range(1, rows+1) if os.environ.get("EXLR8_EXHAUSTIVE") else (1, 16, 64, 65, 512, 1024, 4096)
    for live in live_counts:
        run(grouped, state, wire, ids, weights, pointers, bits, signs, live, 48)
        assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16)), live
