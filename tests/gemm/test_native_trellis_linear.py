"""Shared operand gates against the independent E1 trellis oracle."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch


def reference(codebook="e4m3"):
    path = Path(os.environ.get("CUTEAFD_EXLR8_REFERENCE", "/workspace/cuteafd/python/reference/exlr8.py"))
    spec = importlib.util.spec_from_file_location("e1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if codebook == "f16":
        # E1 already rounds the MCG half sum; omit only the E4M3 conversion.
        module.e4m3_round = lambda values: np.asarray(values, dtype=np.float32)
    return module


class Case:
    def __init__(self, k=1152, n=256, side=1, bits=4, rows=4096, codebook="e4m3"):
        from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
        rng = np.random.default_rng(408)
        raw = rng.integers(0, 256, k*n*bits//8, dtype=np.uint8)
        ref = reference(codebook)
        if side == 1:
            weight = np.concatenate([ref.decode_unit(chunk.tobytes(), bits, "gate", k)
                                     for chunk in raw.reshape(n//128, -1)], axis=1)
        else:
            weight = np.concatenate([ref.decode_unit(chunk.tobytes(), bits, "down", n)
                                     for chunk in raw.reshape(k//128, -1)], axis=0)
        self.x = torch.tensor(rng.normal(0, .1, (rows, k)), device="cuda", dtype=torch.float16)
        self.other_x = torch.tensor(rng.normal(0, .1, (rows, k)), device="cuda", dtype=torch.float16)
        self.inputs = torch.tensor([self.other_x.data_ptr(), self.x.data_ptr()], dtype=torch.int64, device="cuda")
        self.payload = torch.tensor(raw, dtype=torch.uint8, device="cuda")
        self.scales = torch.tensor(rng.uniform(.5, 2, k//128), dtype=torch.float16, device="cuda")
        self.weight = torch.tensor(weight, device="cuda") * self.scales.float().repeat_interleave(128)[:, None]
        self.partial = torch.empty((rows, (k+1023)//1024, n), dtype=torch.float32, device="cuda")
        self.output = torch.empty((rows, n), dtype=torch.float32, device="cuda")
        self.program = compile_trellis_linear_aot(input_features=k, output_features=n,
                           sharded_side=side, bits=bits, codebook=codebook, rotation_count=2, max_rows=rows)

    def launch(self, rows, limit=188*4, rotation=1):
        self.program.launch(self.inputs, self.payload, self.scales,
                            self.partial.flatten(), self.output.flatten(),
                            scalars=(rows, rotation, limit))


@pytest.mark.parametrize("side,bits,codebook", [(1, 4, "e4m3"), (2, 4, "e4m3"), (1, 2, "e4m3"), (2, 3, "e4m3"), (1, 4, "f16"), (2, 3, "f16")])
def test_accuracy_rows_and_cta_invariance(side, bits, codebook):
    c = Case(side=side, bits=bits, codebook=codebook)
    c.launch(64, rotation=0)
    other_expected = c.output[:64].clone()
    other_ref = c.other_x[:64].double() @ c.weight.double()
    other_baseline = (c.other_x[:64].bfloat16() @ c.weight.bfloat16()).double()
    assert torch.linalg.vector_norm(other_expected.double()-other_ref) <= 1.01*torch.linalg.vector_norm(other_baseline-other_ref)
    c.launch(64)
    expected = c.output[:64].clone()
    assert not torch.equal(other_expected, expected)
    ref = c.x[:64].double() @ c.weight.double()
    baseline = (c.x[:64].bfloat16() @ c.weight.bfloat16()).double()
    error = torch.linalg.vector_norm(expected.double()-ref)
    bound = torch.linalg.vector_norm(baseline-ref)
    assert error <= 1.01 * bound, (error.item(), bound.item())
    for rows in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64):
        for limit in (1, 48, 170*4, 188*4):
            c.partial.fill_(float("nan"))
            c.output.fill_(float("nan"))
            c.launch(rows, limit, rotation=rows % 2)
            selected = expected if rows % 2 else other_expected
            assert torch.equal(c.output[:rows].view(torch.int32), selected[:rows].view(torch.int32))
            assert torch.isnan(c.output[rows:]).all()
    c.launch(4096)
    ref = c.x.double() @ c.weight.double()
    baseline = (c.x.bfloat16() @ c.weight.bfloat16()).double()
    assert torch.linalg.vector_norm(c.output.double()-ref) <= 1.01*torch.linalg.vector_norm(baseline-ref)
    assert torch.equal(c.output[:64].view(torch.int32), expected.view(torch.int32))


def test_graph_replay_and_invalid_launch():
    c = Case(k=128, n=128, rows=64)
    c.launch(17)
    expected = c.output[:17].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.launch(17)
    for _ in range(3):
        c.partial.fill_(float("nan"))
        c.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(c.output[:17].view(torch.int32), expected.view(torch.int32))
        assert torch.isnan(c.output[17:]).all()
    for rows, rotation, limit in ((0, 0, 1), (65, 0, 1), (1, -1, 1), (1, 2, 1), (1, 0, 0)):
        c.output.fill_(float("nan"))
        c.launch(rows, limit, rotation)
        assert torch.isnan(c.output).all()


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("codebook", ["e4m3", "f16"])
def test_bit_exact_decode(bits, codebook):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.gemm.trellis_linear._native import decode_tile

    @cute.kernel
    def decode(payload: cute.Tensor, output: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        values = decode_tile(Int64(payload.iterator.toint()), Int32(tile), Int32(lane), bits, codebook == "e4m3")
        for j in cutlass.range_constexpr(4):
            output[tile*128+lane*4+j] = values[j]

    @cute.jit
    def probe(payload: cute.Tensor, output: cute.Tensor, stream):
        decode(payload, output).launch(grid=(32, 1, 1), block=(32, 1, 1), stream=stream)

    raw = np.random.default_rng(408).integers(0, 256, 32*32*bits, dtype=np.uint8)
    p = torch.tensor(raw, device="cuda")
    out = torch.empty(32*128, dtype=torch.uint32, device="cuda")
    pv, ov = from_dlpack(p), from_dlpack(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    program = cute.compile(probe, pv, ov, stream)
    program(pv, ov, stream)
    expected = reference(codebook).decode_tiles(raw.tobytes(), bits)
    values = out.view(torch.float16).cpu().numpy().reshape(32, 32, 8)
    for lane in range(32):
        r, c = 2*(lane % 4), lane//4
        positions = ((r,c),(r+1,c),(r+8,c),(r+9,c),(r,c+8),(r+1,c+8),(r+8,c+8),(r+9,c+8))
        for j, (r, c) in enumerate(positions):
            np.testing.assert_array_equal(values[:, lane, j], expected[:, r, c])
