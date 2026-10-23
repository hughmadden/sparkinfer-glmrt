"""Exact E4M3 unpack and F32 W8A16 prefill against independent CPU decode."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._reference.native_trellis import Reference
from tests.gemm.test_native_dense_checkpoint import reference


def qualify(raw, core, side, *, rows=65):
    from b12x.integration.cuteafd.trellis_unpack import compile_trellis_unpack_e4m3_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.native_dense import compile_trellis_fp8_prefill_aot
    k, n = core.shape
    payload = torch.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), device="cuda")
    unpacked = torch.empty(n, k, device="cuda", dtype=torch.uint8)
    unpack = compile_trellis_unpack_e4m3_aot(input_features=k, output_features=n, sharded_side=side)
    unpack.launch(payload, unpacked)
    expected = torch.tensor(core.T.copy(), device="cuda").to(torch.float8_e4m3fn)
    assert torch.equal(unpacked, expected.view(torch.uint8))
    assert torch.equal(unpacked.view(torch.float8_e4m3fn).half(), expected.half())
    # BF16-representable rotated input is identical on the F16 native path.
    torch.manual_seed(408)
    x = (torch.randn(rows, k, device="cuda") * .03).bfloat16()
    x16 = x.half()
    ptrs = torch.tensor([x16.data_ptr()], device="cuda", dtype=torch.int64)
    scales = torch.ones(k//128, device="cuda", dtype=torch.float16)
    grid = torch.ones(n//128, k//128, device="cuda", dtype=torch.float32)
    partial = torch.empty(rows, (k+1023)//1024, n, device="cuda")
    native = torch.empty(rows, n, device="cuda")
    output = torch.empty_like(native)
    linear = compile_trellis_linear_aot(input_features=k, output_features=n,
        sharded_side=side, rotation_count=1, max_rows=rows)
    gemm = compile_trellis_fp8_prefill_aot(input_features=k, output_features=n, max_rows=rows)
    limit = torch.cuda.get_device_properties(0).multi_processor_count * 4
    linear.launch(ptrs, payload, scales, partial, native, scalars=(rows, 0, limit))
    target = x.double() @ expected.double().T
    baseline = (x @ expected.bfloat16().T).double()
    bound = 1.01 * torch.linalg.vector_norm(baseline-target)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack.launch(payload, unpacked)
        gemm.launch(x, unpacked, grid, output, scalars=(rows,))
    unpacked.fill_(255)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(unpacked, expected.view(torch.uint8))
    assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
    assert torch.linalg.vector_norm(output.double()-target) <= bound
    assert torch.linalg.vector_norm(native.double()-target) <= bound
    rel = (torch.linalg.vector_norm(output.double()-native.double()) /
           torch.linalg.vector_norm(target)).item()
    max_rel = ((output.double()-native.double()).abs() / target.abs().clamp_min(1e-6)).max().item()
    print(f"unpack k={k} n={n} side={side}: relative_L2={rel:.9g}, max_relative={max_rel:.9g}")
    for live in (1, 16, 64, rows):
        output.fill_(float("nan"))
        gemm.launch(x, unpacked, grid, output, scalars=(live,))
        assert torch.linalg.vector_norm(output[:live].double()-target[:live]) <= bound
        assert torch.isnan(output[live:]).all()


@pytest.mark.parametrize("side", [1, 2])
def test_synthetic_unpack(side):
    k, n = 2048, 256
    rng = np.random.default_rng(408)
    raw = rng.integers(256, size=k*n//2, dtype=np.uint8).tobytes()
    units = np.frombuffer(raw, dtype=np.uint8).reshape((n if side == 1 else k)//128, -1)
    ref = Reference("e4m3")
    core = np.concatenate([ref.decode_unit(unit.tobytes(), 4, "gate" if side == 1 else "down",
        k if side == 1 else n) for unit in units], axis=1 if side == 1 else 0)
    qualify(raw, core, side)


@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="requires real checkpoint")
@pytest.mark.parametrize("stem", ["self_attn.q_b_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
def test_real_unpack(stem):
    ref = reference()
    snapshot = os.environ["CUTEAFD_EXLR8_SNAPSHOT"]
    tensors = ref.Tensors(Path(snapshot) / "dense/exlr8-dense-001.safetensors")
    if stem + ".record" not in tensors.meta:
        stem = "model.layers.1." + stem
    _, n, k, bits, side, _, _, _ = map(int, tensors.array(stem + ".record"))
    assert bits == 4
    unit = (k if side == 2 else n)//128 - 1
    _, core, _ = ref.dense_trellis_unit(snapshot, 1, stem, unit)
    length = tensors.meta[stem + ".trellis"]["shape"][1]
    raw = tensors.raw(stem + ".trellis", unit*length, length)
    # dense_trellis_unit uses python/reference/exlr8.py decode_tiles.
    qualify(raw, core, side)
