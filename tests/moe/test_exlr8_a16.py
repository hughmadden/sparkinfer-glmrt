"""Real-checkpoint K1 component gates; opt in with EXLR8 snapshot/reference paths."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from exlr8_oracle import Case

pytestmark = pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"),
                                reason="EXLR8 component gate requires checkpoint and E1 reference")


@pytest.fixture(scope="module", params=[3, 40, 77])
def case(request):
    return Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=request.param)


def program(case):
    from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
    return compile_exlr8_a16_aot(hidden=case.h, units=case.u, top_k=case.topk,
                                 max_groups=case.max_groups)


def launch(case, compiled, rows, cta_limit=96):
    compiled.launch(*(t.flatten() for t in case.tensors()),
                    scalars=(rows, case.groups, cta_limit))


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual.double()-expected.double()) /
                 torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))


def test_reference_a16_fp64_accuracy(case):
    # This gate is usable before the kernel exists and isolates the F16 down input.
    a16 = case.oracle(1)
    fp64 = case.oracle(1, fp64=True, round_intermediate=False)
    assert relative(a16, fp64) <= 1e-3


def test_kernel_accuracy(case):
    case.routing(8)
    compiled = program(case)
    launch(case, compiled, 8)
    reference = case.oracle(8)
    assert relative(case.output[:8], reference) <= 1e-3


def test_rows_groups_packing_cta_invariance(case):
    compiled = program(case)
    case.routing(64)
    launch(case, compiled, 64)
    expected = case.output.clone()
    for rows in range(1, 65):
        # Both live group count and each row's group/lane assignment change.
        case.routing(rows, chunk=7 if rows % 2 else 16, reverse=bool(rows % 3))
        for limit in (1, 48, 188):
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
