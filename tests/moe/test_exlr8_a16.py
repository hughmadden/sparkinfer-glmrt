"""Real-checkpoint K1 component gates; opt in with EXLR8 snapshot/reference paths."""
import os

import numpy as np
import pytest
import torch

from .exlr8_oracle import Case

pytestmark = pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"),
                                reason="EXLR8 component gate requires checkpoint and E1 reference")


@pytest.fixture(scope="module", params=[3, 40, 77])
def case(request):
    return Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=request.param)


def program(case, device_count=False):
    from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
    return compile_exlr8_a16_aot(hidden=case.h, units=case.u, top_k=case.topk,
                                 max_groups=case.max_groups, device_count=device_count)


def launch(case, compiled, rows, cta_limit=96):
    compiled.launch(*(t.flatten() for t in case.tensors()),
                    scalars=(rows, case.groups, cta_limit))


def test_serving_device_count_rows_and_graph_invariance(case):
    compiled = program(case, device_count=True)
    baseline = program(case)
    count = torch.empty(1, dtype=torch.int32, device=case.device)
    case.routing(64)
    launch(case, baseline, 64)
    expected = case.output.clone()

    def serving(rows, limit=96):
        compiled.launch(*(t.flatten() for t in case.tensors()), count,
                        scalars=(rows, limit))

    for rows in (1, 2, 8, 16, 32, 64):
        case.routing(rows, chunk=1 if rows % 3 else 7, reverse=bool(rows % 2))
        count.fill_(case.groups)
        for limit in (1, 96, 188):
            case.output.fill_(float("nan"))
            serving(rows, limit)
            assert torch.equal(case.output[:rows].view(torch.int32), expected[:rows].view(torch.int32))
            assert torch.isnan(case.output[rows:]).all()

    case.routing(16)
    count.fill_(case.groups)
    serving(16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        serving(16)
    # Same graph, changed live count and packing, including zero and malformed
    # counts. Invalid counts must not read poisoned descriptor/scratch memory.
    for chunk in (1, 7, 16):
        case.routing(16, chunk=chunk, reverse=bool(chunk % 2))
        count.fill_(case.groups)
        case.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(case.output[:16].view(torch.int32), expected[:16].view(torch.int32))
    for invalid in (0, -1, case.max_groups + 1):
        count.fill_(invalid)
        case.group_expert.fill_(-1)
        case.fc2.fill_(float("nan"))
        graph.replay()
        assert torch.equal(case.output[:16], torch.zeros_like(case.output[:16]))


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual.double()-expected.double()) /
                 torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))


def test_reference_a16_fp64_accuracy(case):
    # This gate is usable before the kernel exists and isolates the F16 down input.
    a16 = case.oracle(1)
    fp64 = case.oracle(1, fp64=True, round_intermediate=False)
    assert relative(a16, fp64) <= 1e-3


@pytest.mark.parametrize("amplitude", [1.0, 0.02])
def test_kernel_accuracy(case, amplitude):
    original = case.x
    try:
        case.x = (original.float()*amplitude).half()
        case.routing(8)
        compiled = program(case)
        case.fc1.fill_(float("nan"))
        case.intermediate.fill_(float("nan"))
        case.fc2.fill_(float("nan"))
        launch(case, compiled, 8)
        reference = case.oracle(8)
        assert relative(case.output[:8], reference) <= 1e-3
    finally:
        case.x = original


def test_rows_groups_packing_cta_invariance(case):
    compiled = program(case)
    case.routing(64)
    launch(case, compiled, 64)
    expected = case.output.clone()
    for rows in range(1, 65):
        # Both live group count and each row's group/lane assignment change.
        # Single-route groups cross the 24-group wide-FC2 threshold.
        chunk = 1 if rows % 3 == 0 else (7 if rows % 2 else 16)
        case.routing(rows, chunk=chunk, reverse=bool(rows % 3))
        for limit in (1, 48, 188):
            case.fc1.fill_(float("nan"))
            case.intermediate.fill_(float("nan"))
            case.fc2.fill_(float("nan"))
            case.output.fill_(float("nan"))
            launch(case, compiled, rows, limit)
            assert torch.equal(case.output[:rows].view(torch.int32),
                               expected[:rows].view(torch.int32)), (rows, limit)
            assert torch.isnan(case.output[rows:]).all(), "writes past live rows"


def test_graph_replay_exact(case):
    compiled = program(case)
    case.routing(16)
    launch(case, compiled, 16)
    expected = case.output[:16].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(case, compiled, 16)
    for _ in range(3):
        case.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(case.output[:16].view(torch.int32), expected.view(torch.int32))


def test_exact_trellis_decode(case):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32
    from cutlass.cute.runtime import from_dlpack
    from b12x.moe._shared.kernels.w4a16.exlr8 import exlr8_decode_tile

    @cute.kernel
    def decode(payload: cute.Tensor, output: cute.Tensor, width: cutlass.Constexpr):
        lane, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        values = exlr8_decode_tile(Int64(payload.iterator.toint()), Int32(tile), Int32(lane), width)
        for j in cutlass.range_constexpr(4):
            output[tile*128+lane*4+j] = values[j]

    @cute.jit
    def probe(payload: cute.Tensor, output: cute.Tensor, width: cutlass.Constexpr, stream):
        decode(payload, output, width).launch(grid=(4, 1, 1), block=(32, 1, 1), stream=stream)

    from cuda.bindings import driver as cuda
    for (e, u, p), (weight, _, _) in case.decoded.items():
        width = int(case.host_bits[e, u, p])
        # owners are retained in projection insertion order, independent of arenas.
        address = int(case.pointers[e, u, p, 0].cpu())
        payload = next(t for t in case.owners if t.data_ptr() == address)
        output = torch.empty(4*128, dtype=torch.uint32, device=case.device)
        pview, oview = from_dlpack(payload), from_dlpack(output)
        compiled = cute.compile(probe, pview, oview, width, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        compiled(pview, oview, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        reference = case.ref.decode_tiles(payload[:4*32*width].cpu().numpy().tobytes(), width)
        pairs = output.view(torch.float16).cpu().numpy().reshape(4, 32, 8)
        for lane in range(32):
            r, c = 2*(lane % 4), lane//4
            positions = ((r,c),(r+1,c),(r+8,c),(r+9,c),
                         (r,c+8),(r+1,c+8),(r+8,c+8),(r+9,c+8))
            for j, (r, c) in enumerate(positions):
                np.testing.assert_array_equal(pairs[:, lane, j], reference[:, r, c])


@pytest.mark.parametrize("units,hidden", [(5, 6144), (7, 6016)])
def test_fc2_chunk_tail_and_odd_blocks(units, hidden):
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], units=units, hidden=hidden,
                nonresident=((1, 1), (1, 3), (1, 4)))
    compiled = program(case)
    case.routing(16)
    launch(case, compiled, 16, 1)
    expected = case.output[:16].clone()
    assert relative(expected[:1], case.oracle(1)) <= 1e-3

    for groups in (23, 24, 25):
        case.routing(16, chunk=1)
        # Merge same-expert singleton groups to hit the dispatch boundary
        # exactly without changing a route or its per-element reduction.
        while case.groups > groups:
            experts = case.group_expert[:case.groups].cpu().tolist()
            pair = next((i, i + 1) for i in range(case.groups - 1)
                        if experts[i] == experts[i + 1])
            i, j = pair
            joined = torch.cat((case.route_slots[i][case.route_slots[i] >= 0],
                                case.route_slots[j][case.route_slots[j] >= 0]))
            assert joined.numel() <= 16
            case.route_slots[i].fill_(-1)
            case.route_slots[i, :joined.numel()] = joined
            case.route_slots[j:case.groups - 1] = case.route_slots[j + 1:case.groups].clone()
            case.group_expert[j:case.groups - 1] = case.group_expert[j + 1:case.groups].clone()
            case.groups -= 1
        case.inverse.fill_(-1)
        for group in range(case.groups):
            for lane, slot in enumerate(case.route_slots[group].cpu().tolist()):
                if slot >= 0:
                    case.inverse.view(-1)[slot] = group * 16 + lane
        for limit in (1, 48, 188):
            case.fc1.fill_(float("nan"))
            case.intermediate.fill_(float("nan"))
            case.fc2.fill_(float("nan"))
            case.output.fill_(float("nan"))
            launch(case, compiled, 16, limit)
            assert torch.equal(case.output[:16].view(torch.int32),
                               expected.view(torch.int32)), (units, hidden, groups, limit)
            assert torch.isnan(case.output[16:]).all()


def test_empty_routes_do_not_read_poison(case):
    compiled = program(case)
    case.routing(8)
    case.groups = 0
    case.inverse.fill_(-1)
    case.fc1.fill_(float("nan"))
    case.intermediate.fill_(float("nan"))
    case.fc2.fill_(float("nan"))
    case.output.fill_(float("nan"))
    launch(case, compiled, 8, 1)
    assert torch.equal(case.output[:8], torch.zeros_like(case.output[:8]))
    assert torch.isnan(case.output[8:]).all()
