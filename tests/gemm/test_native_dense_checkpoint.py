"""Opt-in real dense trellis units, checksums and distinct rotation pointers."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"),
                                reason="requires real EXLR8 checkpoint")


def reference():
    path = Path(os.environ.get("CUTEAFD_EXLR8_REFERENCE", "/workspace/cuteafd/python/reference/exlr8.py"))
    spec = importlib.util.spec_from_file_location("dense_e1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stem", ["self_attn.q_b_proj", "self_attn.o_proj",
                                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
def test_real_dense_unit(stem):
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    ref = reference()
    snapshot = os.environ["CUTEAFD_EXLR8_SNAPSHOT"]
    tensors = ref.Tensors(Path(snapshot) / "dense/exlr8-dense-001.safetensors")
    # Tensor stems in the dense shard are the checkpoint's short projection names.
    if stem + ".record" not in tensors.meta:
        stem = "model.layers.1." + stem
    record = tensors.array(stem + ".record")
    _, out, inp, bits, side, _, rotation, _ = map(int, record)
    total_units = (inp if side == 2 else out)//128
    unit = total_units-1
    _, core, _checksum = ref.dense_trellis_unit(snapshot, 1, stem, unit)
    k, n = core.shape
    length = tensors.meta[stem + ".trellis"]["shape"][1]
    raw = tensors.raw(stem + ".trellis", unit*length, length)
    # Caller-derived block factors are independent of the packed trellis core.
    scales = np.linspace(.5, 1.5, k//128, dtype=np.float16)
    torch.manual_seed(408)
    x = torch.randn((4, 64, k), dtype=torch.float16, device="cuda") * .1
    ptrs = torch.tensor([v.data_ptr() for v in x], dtype=torch.int64, device="cuda")
    payload = torch.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), device="cuda")
    scale = torch.tensor(scales.copy(), dtype=torch.float16, device="cuda")
    weight = torch.tensor(core, device="cuda")*scale.float().repeat_interleave(128)[:, None]
    partial = torch.empty((64, (k+1023)//1024, n), device="cuda")
    output = torch.empty((64, n), device="cuda")
    program = compile_trellis_linear_aot(input_features=k, output_features=n,
                                        bits=bits, sharded_side=side, max_rows=64)
    for selected in range(4):
        partial.fill_(float("nan"))
        output.fill_(float("nan"))
        program.launch(ptrs, payload, scale, partial, output, scalars=(64, selected, 752))
        expected = output.clone()
        target = x[selected].double() @ weight.double()
        baseline = (x[selected].bfloat16() @ weight.bfloat16()).double()
        assert torch.linalg.vector_norm(output.double()-target) <= 1.01*torch.linalg.vector_norm(baseline-target)
        for rows in (1, 4, 16, 64):
            partial.fill_(float("nan"))
            output.fill_(float("nan"))
            program.launch(ptrs, payload, scale, partial, output, scalars=(rows, selected, 48))
            assert torch.equal(output[:rows].view(torch.int32), expected[:rows].view(torch.int32))
            assert torch.isnan(output[rows:]).all()
