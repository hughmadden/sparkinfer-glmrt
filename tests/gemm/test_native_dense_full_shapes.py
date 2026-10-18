"""Full checkpoint-shape arithmetic gates, with bounded FP64 reference rows."""
import importlib.util
from pathlib import Path

import pytest
import torch


def case_module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("encoding,side,k,n,rows", [
    ("mxfp8", 1, 6144, 2048, 16),
    ("int4_g32", 2, 16384, 6144, 16),
    ("fp8_row", 1, 6144, 154880, 1),
])
def test_scaled_full_checkpoint_shape(encoding, side, k, n, rows):
    c = case_module("test_native_scaled_linear").Case(encoding, side, k=k, n=n, rows=rows)
    c.partial.fill_(float("nan"))
    c.output.fill_(float("nan"))
    c.launch(rows, torch.cuda.get_device_properties(0).multi_processor_count*4)
    reference = c.x.double() @ c.weight.double().T
    baseline = (c.x @ c.weight.bfloat16().T).double()
    error = torch.linalg.vector_norm(c.output.double()-reference)
    bound = torch.linalg.vector_norm(baseline-reference)
    assert error <= 1.01*bound, (error.item(), bound.item())
    if rows > 1:
        expected = c.output.clone()
        for width in (1, 4, 16):
            c.partial.fill_(float("nan"))
            c.output.fill_(float("nan"))
            c.launch(width, 48)
            assert torch.equal(c.output[:width].view(torch.int32), expected[:width].view(torch.int32))
            assert torch.isnan(c.output[width:]).all()


def test_trellis_full_checkpoint_shape():
    c = case_module("test_native_trellis_linear").Case(k=2048, n=16384, rows=16)
    c.partial.fill_(float("nan"))
    c.output.fill_(float("nan"))
    c.launch(16, torch.cuda.get_device_properties(0).multi_processor_count*4)
    expected = c.output.clone()
    reference = c.x.double() @ c.weight.double()
    baseline = (c.x.bfloat16() @ c.weight.bfloat16()).double()
    assert torch.linalg.vector_norm(c.output.double()-reference) <= 1.01*torch.linalg.vector_norm(baseline-reference)
    for width in (1, 4, 16):
        c.partial.fill_(float("nan"))
        c.output.fill_(float("nan"))
        c.launch(width, 48)
        assert torch.equal(c.output[:width].view(torch.int32), expected[:width].view(torch.int32))
        assert torch.isnan(c.output[width:]).all()


def test_embedding_full_checkpoint_shape():
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    k, n = 6144, 154880
    payload = torch.randint(-32, 33, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.linspace(.01, .2, n, device="cuda")
    ids = torch.tensor([0, n-1, n//2, 1], dtype=torch.int32, device="cuda")
    output = torch.empty((4, k), dtype=torch.bfloat16, device="cuda")
    program = compile_fp8_row_embedding_aot(hidden=k, vocabulary=n)
    program.launch(ids, payload.view(torch.uint8), scales, output, scalars=(4, 48))
    selected = payload.view(torch.uint8)[ids.long()].view(torch.float8_e4m3fn)
    expected = (selected.float()*scales[ids.long(), None]).bfloat16()
    assert torch.equal(output, expected)
