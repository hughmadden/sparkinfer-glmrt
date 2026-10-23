"""Real-checkpoint K1 component gates; opt in with EXLR8 snapshot/reference paths.

K1 consumes the rotated F16 activation and writes the BF16 hidden-domain
partial (local sum, inverse H128 and signs.out). The rotated-domain oracle
therefore compares through ``case.egress``.
"""
import os

import numpy as np
import pytest
import torch

from .exlr8_oracle import Case, make_wire

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
        # One BF16 rounding of the response bounds the extra relative error.
        reference = case.egress(case.oracle(8))
        assert relative(case.output[:8], reference) <= 2**-9 + 1e-3
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
    from b12x._lib.intrinsics import create_l2_evict_first_policy
    from b12x.moe._shared.kernels.w4a16.exlr8 import exlr8_decode_tile

    @cute.kernel
    def decode(payload: cute.Tensor, output: cute.Tensor, width: cutlass.Constexpr):
        lane, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        policy = create_l2_evict_first_policy()
        values = exlr8_decode_tile(Int64(payload.iterator.toint()), Int32(tile), Int32(lane), width, policy)
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
    assert relative(expected[:1], case.egress(case.oracle(1))) <= 2**-9 + 1e-3

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


def shared_programs(case):
    from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot

    shared = compile_exlr8_a16_aot(hidden=case.h, units=case.u, top_k=1,
                                  max_groups=4, device_count=True, shared_expert=True)
    group = compile_exlr8_group_aot(experts=1, top_k=1, max_groups=4, hidden=case.h,
                                    input_format="bf16", shared_expert=True)
    return shared, group


class SharedRun:
    """Coordinator shared expert: BF16 rows -> shared group ingress -> K1 half."""

    def __init__(self, case, owned=None):
        self.shared, self.group = shared_programs(case)
        d = case.device
        self.ids = torch.zeros((64, 1), dtype=torch.int32, device=d)
        self.experts = torch.empty(4, dtype=torch.int32, device=d)
        self.slots = torch.empty((4, 16), dtype=torch.int32, device=d)
        self.inverse = torch.empty_like(self.ids)
        self.count = torch.empty(1, dtype=torch.int32, device=d)
        self.gates = torch.ones((64, 1), device=d)
        self.activation = torch.empty((64, case.h), dtype=torch.float16, device=d)
        self.scratch = [torch.empty_like(case.fc1[:4]), torch.empty_like(case.intermediate[:4]),
                        torch.empty_like(case.fc2[:4])]
        self.output = torch.empty((64, case.h), dtype=torch.bfloat16, device=d)
        # Expert slices need dedicated aligned tables, retained across graph replays.
        self.pointers, self.bits = case.pointers[3:4].clone(), case.bits[3:4].clone()
        if owned is not None:
            # TP2 split: unowned intermediate units are nonresident on this GPU.
            for u in range(case.u):
                if u not in owned:
                    self.bits[0, u] = 0
        self.case = case

    def __call__(self, x, rows, limit=96):
        c = self.case
        self.group.launch(self.ids, x, c.signs_in, self.experts, self.slots, self.inverse,
                          self.count, self.activation, scalars=(rows, limit))
        self.shared.launch(self.activation, self.pointers, self.bits, self.experts, self.slots,
                           self.inverse, self.gates, *self.scratch, c.signs_out, self.output,
                           self.count, scalars=(rows, limit))


def test_shared_expert_grouping_serving_reference_and_graph(case):
    run = SharedRun(case)
    generator = torch.Generator().manual_seed(4081)
    x = torch.randn((64, case.h), generator=generator).bfloat16().to(case.device)
    ids = torch.full((64, 1), 3, dtype=torch.int32)
    ones = torch.ones((64, 1), dtype=torch.float64)
    fp64 = case.hidden_reference(x[:4].double(), ids[:4], ones[:4], official=False)
    official = case.hidden_reference(x[:4].double(), ids[:4], ones[:4], official=True)
    run(x, 64)
    expected = run.output.clone()
    native_error = relative(expected[:4], fp64)
    assert native_error <= 2**-9 + 1e-3
    assert native_error <= relative(official, fp64)
    for rows in (1, 2, 16, 32, 63, 64):
        for limit in (1, 7, 188):
            run.output.fill_(float("nan"))
            run.activation.fill_(float("nan"))
            run(x, rows, limit)
            assert torch.equal(run.output[:rows].view(torch.int16), expected[:rows].view(torch.int16))
            assert torch.isnan(run.output[rows:]).all()
            assert run.count.item() == (rows + 15) // 16
    run(x, 64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(x, 64)
    for scale in (0.25, 1.0, 0.5):
        run.gates.fill_(scale)
        graph.replay()
        assert torch.equal(run.output.view(torch.int16), (expected * scale).view(torch.int16))
    run.ids[0, 0] = 1
    graph.replay()
    assert run.count.item() == -1
    assert torch.equal(run.output, torch.zeros_like(run.output))


def test_shared_expert_tp2_halves_sum_to_whole(case):
    """Each GPU rotates its BF16 input, runs its interleaved half of the
    intermediate units and inverse-rotates its half; the BF16 halves add up."""
    generator = torch.Generator().manual_seed(4082)
    x = torch.randn((64, case.h), generator=generator).bfloat16().to(case.device)
    halves = []
    for rank in (0, 1):
        run = SharedRun(case, owned={u for u in range(case.u) if u % 2 == rank})
        run(x, 64)
        halves.append(run.output.float())
    ids = torch.full((64, 1), 3, dtype=torch.int32)
    ones = torch.ones((64, 1), dtype=torch.float64)
    fp64 = case.hidden_reference(x[:4].double(), ids[:4], ones[:4], official=False)
    assert relative(halves[0][:4] + halves[1][:4], fp64) <= 4e-3
    whole = SharedRun(case)
    whole(x, 64)
    assert relative(halves[0] + halves[1], whole.output.float()) <= 4e-3


class SparkStep:
    """Spark worker step for one <=64-row chunk: Fp8K32 wire rows -> fused
    group/ingress -> K1 serving -> BF16 response rows."""

    def __init__(self, case):
        from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
        self.case = case
        self.group = compile_exlr8_group_aot(experts=4, top_k=case.topk, max_groups=case.max_groups,
                                             hidden=case.h, input_format="fp8_k32")
        self.k1 = program(case, device_count=True)
        d = case.device
        self.ids = torch.empty((64, case.topk), dtype=torch.int32, device=d)
        self.weights = torch.empty((64, case.topk), device=d)
        self.wire = torch.empty((64, case.h + case.h // 32), dtype=torch.uint8, device=d)
        self.count = torch.empty(1, dtype=torch.int32, device=d)
        self.activation = torch.empty((64, case.h), dtype=torch.float16, device=d)
        self.group_expert = torch.empty((case.max_groups,), dtype=torch.int32, device=d)
        self.route_slots = torch.empty((case.max_groups, 16), dtype=torch.int32, device=d)
        self.inverse = torch.empty((64, case.topk), dtype=torch.int32, device=d)
        self.output = torch.empty((64, case.h), dtype=torch.bfloat16, device=d)

    def launch(self, rows, limit=96):
        c = self.case
        self.group.launch(self.ids, self.wire, c.signs_in, self.group_expert, self.route_slots,
                          self.inverse, self.count, self.activation, scalars=(rows, limit))
        self.k1.launch(self.activation, c.pointers, c.bits, self.group_expert, self.route_slots,
                       self.inverse, self.weights, c.fc1, c.intermediate, c.fc2, c.signs_out,
                       self.output, self.count, scalars=(rows, limit))

    def wave(self, wire, ids, weights, chunks, limit=96):
        """Run a whole wave in serial chunks, as the worker does; returns BF16 rows."""
        out = torch.empty((wire.shape[0], self.case.h), dtype=torch.bfloat16, device=self.case.device)
        first = 0
        for rows in chunks:
            self.wire[:rows].copy_(wire[first:first + rows])
            self.ids[:rows].copy_(ids[first:first + rows])
            self.weights[:rows].copy_(weights[first:first + rows])
            self.output.fill_(float("nan"))
            self.activation.fill_(float("nan"))
            self.launch(rows, limit)
            out[first:first + rows].copy_(self.output[:rows])
            assert torch.isnan(self.output[rows:]).all(), "writes past live rows"
            first += rows
        assert first == wire.shape[0]
        return out


def wave_inputs(case, rows, seed=4083):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, case.h), generator=generator) * 0.5
    # Some rows carry outliers so per-32 scales span several binades.
    x[::7, ::97] *= 40.0
    wire, dequant = make_wire(x.to(case.device))
    ids = torch.randint(0, 4, (rows, case.topk), generator=generator).int().to(case.device)
    weights = torch.rand((rows, case.topk), generator=generator).add(0.05).to(case.device)
    return wire, dequant, ids, weights


def test_fp8_ingress_matches_reference(case):
    from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
    group = compile_exlr8_group_aot(experts=4, top_k=case.topk, max_groups=case.max_groups,
                                    hidden=case.h, input_format="fp8_k32")
    wire, dequant, ids, _ = wave_inputs(case, 64)
    # UE8M0 edge cases: the F32 subnormal 2^-127 scale byte 0.
    wire[5, case.h:case.h + 3] = 0
    dequant[5, :96] = wire[5, :96].view(torch.float8_e4m3fn).double() * 2.0 ** -127
    rotated = (dequant * case.signs_in.double()).reshape(64, -1, 128)
    from .exlr8_oracle import hadamard
    reference = hadamard(rotated).reshape(64, case.h)
    activation = torch.empty((64, case.h), dtype=torch.float16, device=case.device)
    tables = [torch.empty((case.max_groups,), dtype=torch.int32, device=case.device),
              torch.empty((case.max_groups, 16), dtype=torch.int32, device=case.device),
              torch.empty_like(ids), torch.empty(1, dtype=torch.int32, device=case.device)]
    for limit in (1, 7, 188):
        activation.fill_(float("nan"))
        group.launch(ids, wire, case.signs_in, *tables, activation, scalars=(64, limit))
        torch.testing.assert_close(activation.double(), reference.half().double(), rtol=1e-3, atol=1e-7)
        if limit == 1:
            invariant = activation.clone()
        assert torch.equal(activation.view(torch.int16), invariant.view(torch.int16))
        assert tables[3].item() > 0


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_fp8_spark_step_same_quant_error_ratio(case, rows):
    """Same-quant gate: identical FP8 activations into the native step, the
    checkpoint's BF16 official arithmetic and FP64; native error <= 1.01x
    official error against FP64, globally and per row."""
    step = SparkStep(case)
    wire, dequant, ids, weights = wave_inputs(case, rows, seed=4084 + rows)
    native = step.wave(wire, ids, weights, [rows]).double()
    fp64 = case.hidden_reference(dequant, ids, weights, official=False)
    official = case.hidden_reference(dequant, ids, weights, official=True).double()
    assert relative(native, fp64) <= 1.01 * relative(official, fp64)
    for r in range(rows):
        assert relative(native[r], fp64[r]) <= 1.01 * relative(official[r], fp64[r]), r


def test_fp8_spark_step_chunk_and_cta_invariance(case):
    """64-row chunking is invisible: each row's BF16 response is bitwise the
    row computed alone, for waves of 1/8/64/65/128 rows, any chunk placement,
    packing and CTA limit, and through graph replay of a chunk."""
    step = SparkStep(case)
    wire, _, ids, weights = wave_inputs(case, 128, seed=4090)
    solo = torch.cat([step.wave(wire[r:r + 1], ids[r:r + 1], weights[r:r + 1], [1], limit=1)
                      for r in range(128)])
    for chunks in ([1], [8], [64], [64, 1], [64, 64], [1, 63, 64], [17, 47, 64]):
        total = sum(chunks)
        for limit in (1, 48, 188):
            out = step.wave(wire[:total], ids[:total], weights[:total], chunks, limit)
            assert torch.equal(out.view(torch.int16), solo[:total].view(torch.int16)), (chunks, limit)
    # Graph replay with new chunk contents in the captured buffers.
    step.wire[:64].copy_(wire[:64])
    step.ids[:64].copy_(ids[:64])
    step.weights[:64].copy_(weights[:64])
    step.launch(64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step.launch(64)
    for first in (64, 0, 64):
        step.wire[:64].copy_(wire[first:first + 64])
        step.ids[:64].copy_(ids[first:first + 64])
        step.weights[:64].copy_(weights[first:first + 64])
        step.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(step.output.view(torch.int16), solo[first:first + 64].view(torch.int16))
