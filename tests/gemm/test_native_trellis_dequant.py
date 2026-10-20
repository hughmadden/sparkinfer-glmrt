"""Bounded native K6 -> BF16 scratch -> F32 projection -> H128 egress."""
import numpy as np
import pytest
import torch

from tests._reference.native_trellis import Reference
from tests.gemm.test_native_trellis_transform import hadamard


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
def test_chunked_composed_forward(side, codebook):
    from b12x.integration.cuteafd.trellis_dequant import compile_trellis_dequant_aot
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    k, n, columns, capacity = 2048, 256, 128, 256
    torch.manual_seed(273)
    raw = torch.randint(256, (k*n*6//8,), device="cuda", dtype=torch.uint8)
    ref = Reference(codebook)
    data = raw.cpu().numpy().reshape((n if side == 1 else k)//128, -1)
    decoded = np.concatenate([ref.decode_unit(unit.tobytes(), 6, "gate" if side == 1 else "down",
        k if side == 1 else n) for unit in data], axis=1 if side == 1 else 0)
    weight = torch.tensor(decoded.T.copy(), device="cuda").bfloat16()
    x = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)*.03
    suh = torch.randn(k, device="cuda", dtype=torch.float16)*.2
    svh = torch.randn(n, device="cuda", dtype=torch.float16)*.2
    rotated = torch.empty_like(x)
    scales = torch.empty(k//128, device="cuda", dtype=torch.float16)
    scratch = torch.empty(columns, k, device="cuda", dtype=torch.bfloat16)
    chunk = torch.empty(capacity, columns, device="cuda", dtype=torch.float32)
    accumulator = torch.empty(capacity, n, device="cuda", dtype=torch.float32)
    output = torch.empty_like(accumulator, dtype=torch.bfloat16)
    inp = compile_trellis_input_aot(features=k, max_rows=capacity, output_dtype="bf16")
    dequant = compile_trellis_dequant_aot(input_features=k, output_features=n,
        chunk_columns=columns, codebook=codebook, sharded_side=side)
    gemm = compile_bf16_projection_aot(width=k, outputs=columns, output_dtype=torch.float32)
    egress = compile_trellis_output_aot(features=n, max_rows=capacity)
    limit = torch.cuda.get_device_properties(0).multi_processor_count*4
    expected_rotated = hadamard(x.float()*suh.float()).bfloat16()
    expected_acc = expected_rotated.float() @ weight.float().T
    expected = (hadamard(expected_acc)*svh.float()).bfloat16()
    for rows in (1, 64, 256):
        def run():
            inp.launch(x, suh, rotated, scales, scalars=(rows, limit))
            for first in range(0, n, columns):
                dequant.launch(raw, scales, scratch, scalars=(first,))
                gemm.launch(rotated, scratch, chunk, scalars=(rows,))
                accumulator[:rows, first:first+columns].copy_(chunk[:rows])
            egress.launch(accumulator, svh, output, scalars=(rows, limit))
        run()
        torch.testing.assert_close(scratch, weight[-columns:], rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(rotated[:rows], expected_rotated[:rows], rtol=0, atol=0)
        torch.testing.assert_close(accumulator[:rows], expected_acc[:rows], rtol=.002, atol=.00001)
        torch.testing.assert_close(output[:rows], expected[:rows], rtol=.02, atol=.00001)
