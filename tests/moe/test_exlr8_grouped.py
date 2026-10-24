"""Decode fidelity and bitwise row/packing invariance for grouped EXLR8."""
import os
from pathlib import Path

import pytest
import torch

from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from benchmarks.benchmark_exlr8_a16 import compact_unit_descriptors, compact_program
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from b12x._lib.runtime_control import kernel_resolution_guard
from .exlr8_oracle import Case, make_wire


def assert_matches_decode(actual, expected, weights):
    from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
    if Exlr8GroupedKernel.FC2_HALF and not os.environ.get("EXLR8_FROZEN_GROUPED_ARM"):
        # V3 rounds FC2 route partials to FP16; frozen exact experiment arms
        # retain the original bitwise contract.
        diff = actual.float() - expected.float()
        # A rounded FP16 subnormal contributes at most 2^-25 per column.
        # Normalized H128 preserves L2; route weights bound each row's
        # aggregate error. Keep the 1% normal-range bound as a separate term.
        subnormal_row = 2.0 ** -25 * actual.shape[-1] ** 0.5 * weights.float().abs().sum(-1)
        bound = 0.01 * torch.linalg.vector_norm(expected.float()) + torch.linalg.vector_norm(subnormal_row)
        error = torch.linalg.vector_norm(diff)
        assert torch.isfinite(actual).all() and error <= bound, (float(error), float(bound))
    else:
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


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


def programs(hidden, units, topk, capacity, group_rows, grouped,
             activations="a16", quantize_pass=False):
    candidate = compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=topk,
                                      max_groups=capacity, device_count=True,
                                      grouped=grouped, group_rows=group_rows,
                                      activations=activations,
                                      quantize_pass=quantize_pass)
    frozen = os.environ.get("EXLR8_FROZEN_GROUPED_ARM")
    if grouped and frozen and activations == "a16":
        from benchmarks.benchmark_exlr8_a16 import compile_grouped_arm
        kernel, group, _ = compile_grouped_arm(Path(frozen), candidate)
        return group, kernel
    return (compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=capacity,
                                   hidden=hidden, max_rows=4096 if grouped else 64,
                                   group_rows=group_rows), candidate)


def quantize_k32_e4m3_reference(x_f16):
    blocks = x_f16.float().reshape(*x_f16.shape[:-1], x_f16.shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    inv_448 = torch.tensor(float.fromhex("0x1.24924ap-9"), dtype=torch.float32, device=x_f16.device)
    c_448 = torch.tensor(448.0, dtype=torch.float32, device=x_f16.device)
    s_a = amax * inv_448
    inv_s = torch.where(amax > 0.0, c_448 / amax, torch.zeros_like(amax))
    q = (blocks * inv_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return q, s_a.squeeze(-1)


def reference_a8_grouped(case, activation_f16, ids, weights):
    from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
    from .exlr8_oracle import hadamard
    rows, h = activation_f16.shape
    topk = ids.shape[1]
    q_x, s_x = quantize_k32_e4m3_reference(activation_f16)
    output = torch.zeros((rows, h), dtype=torch.float32, device=activation_f16.device)
    for r in range(rows):
        for slot in range(topk):
            e = int(ids[r, slot].item())
            route = torch.zeros(h, dtype=torch.float32, device=activation_f16.device)
            for u in range(case.u):
                if not case.host_bits[e, u, 0]:
                    continue
                projections = []
                for p in (0, 1):
                    w, c, b = case.decoded[e, u, p]
                    w_e4m3 = w.half().to(torch.float8_e4m3fn).float()
                    splits = []
                    for start in range(0, h // 128, 8):
                        total = torch.zeros(128, dtype=torch.float32, device=activation_f16.device)
                        for block in range(start, min(start + 8, h // 128)):
                            acc_blk = torch.zeros(128, dtype=torch.float32, device=activation_f16.device)
                            for k32 in range(4):
                                k_idx = block * 4 + k32
                                w_slice = w_e4m3[k_idx * 32 : (k_idx + 1) * 32]
                                acc_blk = acc_blk + (q_x[r, k_idx] @ w_slice) * s_x[r, k_idx]
                            total = total + acc_blk * b[block].float()
                        splits.append(total)
                    total = splits[0]
                    for partial in splits[1:]:
                        total = total + partial
                    projections.append(hadamard(total) * c.float())
                gate, up = projections
                w_d, c_d, b_d = case.decoded[e, u, 2]
                w_d_e4m3 = w_d.half().to(torch.float8_e4m3fn).float()
                mid = hadamard((gate * torch.sigmoid(gate) * up) * c_d.float()).half()
                q_mid, s_mid = quantize_k32_e4m3_reference(mid)
                partial = torch.zeros(h, dtype=torch.float32, device=activation_f16.device)
                for k32 in range(4):
                    w_slice = w_d_e4m3[k32 * 32 : (k32 + 1) * 32]
                    partial = partial + (q_mid[k32] @ w_slice) * s_mid[k32]
                route = route + partial * b_d.float().repeat_interleave(128)
            if Exlr8GroupedKernel.FC2_HALF:
                route = route.half().float()
            output[r] = output[r] + route * weights[r, slot].float()
    return case.egress(output).to(torch.bfloat16)


def compute(k1, state, weights, pointers, bits, signs, rows, limit):
    extra = state["compact"][:2] if compact_program(k1) else ()
    scalars = (rows, limit, state["compact"][2]) if extra else (rows, limit)
    k1.launch(state["activation"], pointers, bits, state["group_expert"], state["route_slots"],
              state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
              signs[1], state["output"], state["count"], *extra, scalars=scalars)


def run(program, state, wire, ids, weights, pointers, bits, signs, rows, limit):
    group, k1 = program
    group.launch(ids, wire, signs[0], state["group_expert"], state["route_slots"],
                 state["inverse"], state["count"], state["activation"], scalars=(rows, limit))
    compute(k1, state, weights, pointers, bits, signs, rows, limit)


def reverse_packing(state, rows):
    count = int(state["count"].item())
    state["group_expert"][:count].copy_(state["group_expert"][:count].flip(0))
    state["route_slots"][:count].copy_(state["route_slots"][:count].flip((0, 1)))
    slots = state["route_slots"][:count].flatten().long()
    packed = torch.arange(slots.numel(), device=slots.device, dtype=torch.int32)
    valid = slots >= 0
    state["inverse"][:rows].flatten()[slots[valid]] = packed[valid]


# (group_rows, units): the units=8 case exceeds the shared-memory budget for
# resident intermediates, so it exercises the streamed FC2 ring.
@pytest.mark.parametrize("group_rows, units", [(16, 3), (32, 3), (64, 3), (64, 8)])
@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="checkpoint fixture required")
def test_checkpoint_grouped_matches_chunks(group_rows, units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=1152, units=units)
    rows, hidden, topk = 129, case.h, case.topk
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
    state["compact"] = compact_unit_descriptors(case.bits)
    run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
    grouped_expected = state["output"][:rows].clone()
    assert_matches_decode(grouped_expected, expected, weights)
    with kernel_resolution_guard("EXLR8 grouped runtime rows and packing"):
        for live in (1, 8, 16, 17, 32, 63, 64, 65, 127, 128, 129):
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16))
                reverse_packing(state, live)
                compute(grouped[1], state, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16))
                assert torch.isnan(state["output"][live:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
        for _ in range(2):
            weights.mul_(0.5)
            wire.copy_(wire.roll(1, 0))
            ids.copy_(ids.roll(1, 0))
            graph.replay()
            actual = state["output"][:rows].clone()
            for first in range(0, rows, 64):
                live = min(rows-first, 64)
                run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
                assert_matches_decode(actual[first:first+live], base["output"][:live], weights[first:first+live])


@pytest.mark.parametrize("units", [1, 16])
def test_every_live_row_count_matches_decode(units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    # Two K128 blocks: block 1's pinned FMA uses its own scale load and the
    # partial plane is written after both blocks accumulate. Cross-split
    # read+add is covered by the hidden-1152 checkpoint cases (9 blocks,
    # 2 splits).
    hidden, topk, rows = 256, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.zeros((4, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((4, units, 3), dtype=torch.int32)
    for e in range(4):
        # Include an empty expert and rotated, unequal noncontiguous unit sets.
        owned = range(units) if units == 1 else (() if e == 3 else
                (u for u in range(units) if (units * e + u + 3) % 6 == 0))
        for u in owned:
            for p in range(3):
                width = 2 + (e+u+p) % 3
                ts = (torch.randint(256, (hidden*16*width,), generator=gen, dtype=torch.uint8, device="cuda"),
                      torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                      torch.full((hidden//128,), 0.1, dtype=torch.float16, device="cuda"))
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows*topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk
    base, state = buffers(64, hidden, units, topk, 128, 16), buffers(rows, hidden, units, topk, 132, 64)
    dec, grouped = programs(hidden, units, topk, 128, 16, False), programs(hidden, units, topk, 132, 64, True)
    state["compact"] = compact_unit_descriptors(bits)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        run(dec, base, wire[first:], ids[first:], weights[first:], pointers, bits, signs, 64, 48)
        expected[first:first+64].copy_(base["output"][:64])
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    run(grouped, state, wire, ids, weights, pointers, bits, signs, rows, 48)
    grouped_expected = state["output"][:rows].clone()
    assert_matches_decode(grouped_expected, expected, weights)
    live_counts = range(1, rows+1) if os.environ.get("EXLR8_EXHAUSTIVE") else (1, 16, 64, 65, 512, 1024, 4096)
    with kernel_resolution_guard("EXLR8 grouped exhaustive row invariance"):
        for live in live_counts:
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped, state, wire, ids, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16)), (live, limit)
                reverse_packing(state, live)
                compute(grouped[1], state, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16)), (live, limit, "reverse")
                assert torch.isnan(state["output"][live:]).all()


@pytest.mark.parametrize("group_rows, units", [(16, 3), (32, 3), (64, 3), (64, 8)])
@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="checkpoint fixture required")
def test_checkpoint_grouped_a8_oracle_and_invariance(group_rows, units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=1152, units=units)
    rows, hidden, topk = 129, case.h, case.topk
    gen = torch.Generator().manual_seed(408)
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen).cuda() * 0.02)
    ids = (torch.arange(rows * topk, device="cuda").reshape(rows, topk) % 4).int()
    weights = torch.rand((rows, topk), generator=gen).cuda() / topk
    signs = (case.signs_in, case.signs_out)

    state = buffers(rows, hidden, units, topk, 128, group_rows)
    state["compact"] = compact_unit_descriptors(case.bits)
    grouped_a16 = programs(hidden, units, topk, 128, group_rows, True, activations="a16")
    grouped_a8 = programs(hidden, units, topk, 128, group_rows, True, activations="a8", quantize_pass=False)
    grouped_a8_qp = programs(hidden, units, topk, 128, group_rows, True, activations="a8", quantize_pass=True)

    run(grouped_a16, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
    expected_a16 = state["output"][:rows].clone()

    run(grouped_a8, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
    actual_a8 = state["output"][:rows].clone()

    run(grouped_a8_qp, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
    actual_a8_qp = state["output"][:rows].clone()
    assert torch.equal(actual_a8.view(torch.int16), actual_a8_qp.view(torch.int16))

    # Gate 1: Exactness vs reference A8 computation using identical activation quantization rule
    ref_a8 = reference_a8_grouped(case, state["activation"][:rows].clone(), ids, weights)
    diff_ref = actual_a8.float() - ref_a8.float()
    rel_l2_ref = float(torch.linalg.vector_norm(diff_ref) / torch.linalg.vector_norm(ref_a8.float()).clamp_min(1e-30))
    cos_ref = float(torch.nn.functional.cosine_similarity(
        actual_a8.float().flatten().unsqueeze(0), ref_a8.float().flatten().unsqueeze(0)).item())
    assert torch.isfinite(actual_a8).all() and rel_l2_ref <= 1e-3 and cos_ref >= 0.999999, (rel_l2_ref, cos_ref)

    # Gate 3: Informational error vs A16 on the same inputs
    diff_a16 = actual_a8.float() - expected_a16.float()
    rel_l2_a16 = float(torch.linalg.vector_norm(diff_a16) / torch.linalg.vector_norm(expected_a16.float()).clamp_min(1e-30))
    cos_a16 = float(torch.nn.functional.cosine_similarity(
        actual_a8.float().flatten().unsqueeze(0), expected_a16.float().flatten().unsqueeze(0)).item())
    print(f"\n[A8 info group_rows={group_rows} units={units}] vs_ref: rel_l2={rel_l2_ref:.6e} cos={cos_ref:.8f} | "
          f"vs_a16: rel_l2={rel_l2_a16:.6f} cos={cos_a16:.6f}")

    # Gate 2: Row/packing/CTA invariance and graph replay under frozen kernel resolution
    with kernel_resolution_guard("EXLR8 A8 grouped runtime rows and packing"):
        for live in (1, 8, 16, 17, 32, 63, 64, 65, 127, 128, 129):
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped_a8, state, wire, ids, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), actual_a8[:live].view(torch.int16))
                reverse_packing(state, live)
                compute(grouped_a8[1], state, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), actual_a8[:live].view(torch.int16))
                assert torch.isnan(state["output"][live:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(grouped_a8, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
        for _ in range(2):
            weights.mul_(0.5)
            wire.copy_(wire.roll(1, 0))
            ids.copy_(ids.roll(1, 0))
            graph.replay()
            replayed = state["output"][:rows].clone()
            run(grouped_a8, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
            assert torch.equal(replayed.view(torch.int16), state["output"][:rows].view(torch.int16))


@pytest.mark.parametrize("units", [1, 16])
def test_every_live_row_count_a8_invariance(units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    hidden, topk, rows = 256, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.zeros((4, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((4, units, 3), dtype=torch.int32)
    for e in range(4):
        owned = range(units) if units == 1 else (() if e == 3 else
                (u for u in range(units) if (units * e + u + 3) % 6 == 0))
        for u in owned:
            for p in range(3):
                width = 2 + (e + u + p) % 3
                ts = (torch.randint(256, (hidden * 16 * width,), generator=gen, dtype=torch.uint8, device="cuda"),
                      torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                      torch.full((hidden // 128,), 0.1, dtype=torch.float16, device="cuda"))
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows * topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk
    state = buffers(rows, hidden, units, topk, 132, 64)
    state["compact"] = compact_unit_descriptors(bits)
    grouped_a16 = programs(hidden, units, topk, 132, 64, True, activations="a16")
    grouped_a8 = programs(hidden, units, topk, 132, 64, True, activations="a8", quantize_pass=False)
    grouped_a8_qp = programs(hidden, units, topk, 132, 64, True, activations="a8", quantize_pass=True)

    run(grouped_a16, state, wire, ids, weights, pointers, bits, signs, rows, 48)
    expected_a16 = state["output"][:rows].clone()
    assert torch.isfinite(expected_a16).all() and torch.count_nonzero(expected_a16)

    run(grouped_a8, state, wire, ids, weights, pointers, bits, signs, rows, 48)
    grouped_expected = state["output"][:rows].clone()
    assert torch.isfinite(grouped_expected).all() and torch.count_nonzero(grouped_expected)

    run(grouped_a8_qp, state, wire, ids, weights, pointers, bits, signs, rows, 48)
    assert torch.equal(state["output"][:rows].view(torch.int16), grouped_expected.view(torch.int16))

    diff_a16 = grouped_expected.float() - expected_a16.float()
    rel_l2_a16 = float(torch.linalg.vector_norm(diff_a16) / torch.linalg.vector_norm(expected_a16.float()).clamp_min(1e-30))
    cos_a16 = float(torch.nn.functional.cosine_similarity(
        grouped_expected.float().flatten().unsqueeze(0), expected_a16.float().flatten().unsqueeze(0)).item())
    print(f"\n[A8 synthetic units={units}] vs_a16: rel_l2={rel_l2_a16:.6f} cos={cos_a16:.6f}")

    live_counts = range(1, rows + 1) if os.environ.get("EXLR8_EXHAUSTIVE") else (1, 16, 64, 65, 512, 1024, 4096)
    with kernel_resolution_guard("EXLR8 A8 grouped exhaustive row invariance"):
        for live in live_counts:
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped_a8, state, wire, ids, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16)), (live, limit)
                reverse_packing(state, live)
                compute(grouped_a8[1], state, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), grouped_expected[:live].view(torch.int16)), (live, limit, "reverse")
                assert torch.isnan(state["output"][live:]).all()
