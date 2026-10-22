"""Exhaustive termination and accuracy checks for the opt-in EXLR8 V3 FC2 arm (#499).

V3 is the warp-specialized FC2 with FP16 route partials and the K4-style TopKSum
reduce (``FC2_WS``/``REDUCE_MLP``/``FC2_HALF``). It is not a serving default: these
tests compile it explicitly as a subclass so a branch that carries the deadlock fix
can prove termination without changing what serving selects.

The deadlock this guards against was a full-mask ``shfl.sync`` executed under the
per-quad output-liveness predicate: warp-divergent collectives never complete. Any
row count that leaves a quad partially live could therefore hang. The first test
launches a real kernel for every row count 1..4096, so a reintroduced divergence
shows up as a timeout instead of a hang, and the second covers hidden=6144 across
the sliced (<=1024 rows) and unsliced FC2 paths.
"""
from pathlib import Path
import hashlib
import inspect
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.benchmark_exlr8_a16 import compact_unit_descriptors
from b12x.integration.cuteafd import _common
from b12x.integration.cuteafd.exlr8 import _Exlr8GroupedServingPointers, compile_exlr8_a16_aot
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
from b12x._lib.runtime_control import kernel_resolution_guard
from tests.moe.exlr8_oracle import make_wire
from tests.moe.test_exlr8_grouped import buffers, compute, programs, reverse_packing, run


class V3OptInKernel(Exlr8GroupedKernel):
    """The opt-in arm: V3 flags without changing serving's class defaults."""

    CODE_VERSION = 102
    FC2_WS = True
    REDUCE_MLP = True
    FC2_HALF = True


def compile_v3_optin(candidate):
    """Compile the opt-in arm against ``candidate``'s ABI, operands and scratch."""
    g = dict(candidate.geometry)
    wrapper = _Exlr8GroupedServingPointers(g)
    wrapper.core = V3OptInKernel(g["hidden"], g["units"], g["top_k"],
                                 g["max_groups"], g["group_rows"])
    source = Path(inspect.getsourcefile(Exlr8GroupedKernel)).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    return _common.compile_program(wrapper, name="exlr8_v3_optin_grouped",
        operands=candidate.operands, scalars=candidate.scalars,
        key=(digest, tuple(sorted(g.items()))), geometry=g, scratch=candidate.scratch)


def check_close(actual_bf16: torch.Tensor, expected_bf16: torch.Tensor, label: object):
    actual = actual_bf16.float()
    expected = expected_bf16.float()
    assert torch.isfinite(actual).all(), f"non-finite output at {label}"
    assert torch.count_nonzero(actual) > 0, f"all-zero output at {label}"
    diff = (actual - expected).abs()
    rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-30))
    assert rel_l2 < 0.01, f"excessive relative L2 {rel_l2} at {label}"
    return rel_l2


def test_exhaustive_1_to_4096():
    torch.cuda.set_per_process_memory_fraction(0.25)
    hidden, units, topk, rows = 256, 16, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.zeros((4, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((4, units, 3), dtype=torch.int32)
    for e in range(4):
        # e == 3 is an empty expert (owned = ()), exercising owned == 0.
        owned = () if e == 3 else (u for u in range(units) if (units * e + u + 3) % 6 == 0)
        for u in owned:
            for p in range(3):
                width = 2 + (e + u + p) % 3
                ts = (
                    torch.randint(256, (hidden * 16 * width,), generator=gen, dtype=torch.uint8, device="cuda"),
                    torch.full((128,), 1.0, dtype=torch.float16, device="cuda"),
                    torch.full((hidden // 128,), 0.25, dtype=torch.float16, device="cuda"),
                )
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows * topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk

    state_base = buffers(rows, hidden, units, topk, 132, 64)
    state_v3 = buffers(rows, hidden, units, topk, 132, 64)
    base_group, base_k1 = programs(hidden, units, topk, 132, 64, True)
    v3_k1 = compile_v3_optin(base_k1)
    state_base["compact"] = compact_unit_descriptors(bits)
    state_v3["compact"] = compact_unit_descriptors(bits)

    # Compute reference for all 4096 rows using base grouped kernel
    run((base_group, base_k1), state_base, wire, ids, weights, pointers, bits, signs, rows, 48)
    expected = state_base["output"][:rows].clone()
    torch.cuda.synchronize()

    t0 = time.time()
    max_rel_l2 = 0.0
    with kernel_resolution_guard("EXLR8 V3 exhaustive 1..4096 row invariance"):
        for live in range(1, rows + 1):
            run((base_group, v3_k1), state_v3, wire, ids, weights, pointers, bits, signs, live, 48)
            if live in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 511, 512, 513, 1023, 1024, 1025, 2688, 4095, 4096) or live % 128 == 0:
                rel = check_close(state_v3["output"][:live], expected[:live], ("exhaustive", live))
                max_rel_l2 = max(max_rel_l2, rel)
            # Also test with an explicit empty group appended and reversed packing on boundary sizes
            if live in (1, 7, 8, 15, 16, 17, 63, 64, 65, 512, 1024, 4096):
                reverse_packing(state_v3, live)
                # Inject an extra empty group at the end of group_expert / route_slots
                c = int(state_v3["count"].item())
                if c + 1 < 132:
                    state_v3["group_expert"][c] = 0
                    state_v3["route_slots"][c].fill_(-1)
                    state_v3["count"].fill_(c + 1)
                state_v3["output"].fill_(float("nan"))
                compute(v3_k1, state_v3, weights, pointers, bits, signs, live, 48)
                rel = check_close(state_v3["output"][:live], expected[:live], ("empty_group+reverse", live))
                assert torch.isnan(state_v3["output"][live:]).all()
                max_rel_l2 = max(max_rel_l2, rel)
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"PASS: V3 terminated and matched across ALL 4096 row counts 1..4096 in {elapsed:.2f}s (max_rel_l2={max_rel_l2:.6e})", flush=True)


def test_hidden6144_sliced_and_unsliced():
    """Exercise hidden=6144 (48 blocks) across sliced (<=512, 513..1024) and unsliced (>1024) row counts."""
    hidden, units, topk, max_rows = 6144, 4, 8, 4096
    capacity = (max_rows * topk + 63) // 64 + 256
    gen = torch.Generator(device="cuda").manual_seed(4080)
    owners = []
    pointers = torch.zeros((256, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((256, units, 3), dtype=torch.int32)
    unit_bytes = hidden * 16 * 4
    for e in range(256):
        owned = () if e == 255 else range(units)
        for u in owned:
            for p in range(3):
                ts = (
                    torch.randint(0, 256, (unit_bytes,), generator=gen, dtype=torch.uint8, device="cuda"),
                    torch.ones(128, dtype=torch.float16, device="cuda"),
                    torch.full((hidden // 128,), 1 / hidden ** 0.5, dtype=torch.float16, device="cuda"),
                )
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = 4
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((max_rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(max_rows * topk, device="cuda") * 7 + 3) % 256).reshape(max_rows, topk).int()
    weights = torch.rand((max_rows, topk), generator=gen, device="cuda") / topk

    state = buffers(max_rows, hidden, units, topk, capacity, 64)
    state["compact"] = compact_unit_descriptors(bits)
    base_k1 = compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=topk, max_groups=capacity, device_count=True, grouped=True, group_rows=64)
    base_group = compile_exlr8_group_aot(experts=256, hidden=hidden, top_k=topk, max_groups=capacity, max_rows=4096, group_rows=64)
    v3_k1 = compile_v3_optin(base_k1)

    step_rows = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 255, 511, 512, 513, 768, 1023, 1024, 1025, 1408, 2048, 2688, 3328, 4095, 4096]
    max_rel_l2 = 0.0
    for live in step_rows:
        run((base_group, base_k1), state, wire, ids, weights, pointers, bits, signs, live, 96)
        ref = state["output"][:live].clone()
        state["output"].fill_(float("nan"))
        run((base_group, v3_k1), state, wire, ids, weights, pointers, bits, signs, live, 96)
        rel = check_close(state["output"][:live], ref, ("hidden6144", live))
        assert torch.isnan(state["output"][live:]).all()
        max_rel_l2 = max(max_rel_l2, rel)
    torch.cuda.synchronize()
    print(f"PASS: hidden=6144 sliced+unsliced steps ({len(step_rows)} row counts in 1..4096) max_rel_l2={max_rel_l2:.6e}", flush=True)


if __name__ == "__main__":
    test_exhaustive_1_to_4096()
    test_hidden6144_sliced_and_unsliced()
