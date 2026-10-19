"""CPU regressions for the EXLR8 Spark-step microbench helpers."""
import torch

from benchmarks import benchmark_exlr8_a16 as bench
from .exlr8_oracle import make_wire


def test_bench_wire_matches_oracle_wire():
    generator = torch.Generator().manual_seed(408)
    x = torch.randn((5, 6144), generator=generator) * 3
    x[1, :32] = 0
    x[2, 64:96] *= 1e4
    wire = bench.make_wire(x)
    expected, dequant = make_wire(x)
    assert torch.equal(wire, expected)
    assert wire.shape == (5, 6144 + 192)
    # The E4M3 payload times its UE8M0 scale reproduces the oracle dequant.
    values = wire[:, :6144].view(torch.float8_e4m3fn).double().reshape(5, 192, 32)
    scales = torch.exp2(wire[:, 6144:].double() - 127).unsqueeze(-1)
    assert torch.equal((values * scales).reshape(5, 6144), dequant)


def test_stage_needles_cover_every_step_kernel():
    assert set(bench.STAGES) == {"ingress", "fc1", "activation", "fc2", "egress"}
