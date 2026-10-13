"""Canonical single-row FP8 arithmetic at wider admitted row counts."""
import pytest
import torch

from b12x.gemm._shared.block_fp8 import pack_block_fp8_linear_weight_mxfp8
from tests.conftest import require_b12x


@pytest.mark.parametrize("capacity,rows", [(16, 2), (16, 16), (80, 2), (80, 16),
                                          (80, 17), (80, 33), (80, 64), (80, 80)])
@pytest.mark.parametrize("n,slices", [(5120, 2), (2560, 1)])
def test_m1_recipe_all_m_matches_serial_and_old_grid_drops_rows(capacity, rows, n, slices, monkeypatch):
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import make_ptr
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot
    from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
    from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm

    device = require_b12x()
    k = 8192
    torch.manual_seed(41000 + rows + n)
    x = torch.randn((rows, k), device=device, dtype=torch.bfloat16)
    w = (torch.randn((n, k), device=device) / 16).to(torch.float8_e4m3fn)
    scale = torch.full((n // 32, k // 32), 127, device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weight = pack_block_fp8_linear_weight_mxfp8(w, scale, block_size=(32, 32)).weight
    alpha = torch.ones(1, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    def pointer(dtype, tensor):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    def packed(source):
        value = empty_mxfp8_rows_for_dense_gemm(source.shape[0], k, num_groups=1, device=device)
        quantize_mxfp8_rows_cute(source, value.values, value.scale_rows, value.scale_mma,
                               scale_block_size=32, amax_floor=1e-4, expected_m=1)
        return value
    def compile(all_m, admitted=capacity):
        kernel, actual_slices = compile_dense_gemm_mxfp8_aot(
            size_m=admitted, size_n=n, size_k=k, expected_m=1, sm_count=188,
            return_split_k_metadata=True, split_k_all_m=all_m)
        assert actual_slices == slices
        return kernel
    fixed, old, serial = compile(True), compile(False), compile(False, 1)
    def arguments(value, partial):
        return (pointer(cutlass.Float8E4M3FN, value.values),
                pointer(cutlass.Float8E4M3FN, weight.values),
                pointer(cutlass.Float8E8M0FNU, value.scale_mma),
                pointer(cutlass.Float8E8M0FNU, weight.scale_mma),
                pointer(cutlass.Float32 if slices > 1 else cutlass.BFloat16, partial),
                pointer(cutlass.Float8E4M3FN, value.values),
                pointer(cutlass.Float8E8M0FNU, value.scale_rows),
                pointer(cutlass.Float8E8M0FNU, value.scale_mma), pointer(cutlass.Float32, alpha))
    def folded(partial):
        # Preserve ordered plane fold, not a parallel sum reduction.
        return (partial[0] + partial[1]).bfloat16() if slices > 1 else partial[0]
    def launch(kernel, source):
        value = packed(source)
        partial = torch.full((slices, source.shape[0], n), float("nan"), device=device,
                             dtype=torch.float32 if slices > 1 else torch.bfloat16)
        kernel(*arguments(value, partial), source.shape[0], stream)
        return partial, folded(partial)
    partial, actual = launch(fixed, x)
    expected = torch.cat([launch(serial, x[i:i+1])[1] for i in range(rows)])
    assert torch.isfinite(partial).all()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    dropped, dropped_output = launch(old, x)
    # Split planes already write a full first tile; unsplit direct C writes row zero.
    first_missing = 16 if slices > 1 else 1
    assert torch.isfinite(dropped[:, :min(rows, first_missing)]).all()
    if rows > first_missing:
        assert torch.isnan(dropped[:, first_missing:]).all()

    value = packed(x)
    replay_partial = torch.empty_like(partial)
    replay_args = arguments(value, replay_partial)
    fixed(*replay_args, rows, stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fixed(*replay_args, rows, cuda.CUstream(capture_stream.cuda_stream))
    x.mul_(-0.5)
    quantize_mxfp8_rows_cute(x, value.values, value.scale_rows, value.scale_mma,
                           scale_block_size=32, amax_floor=1e-4, expected_m=1)
    replay_expected = torch.cat([launch(serial, x[i:i+1])[1] for i in range(rows)])

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen replay must not resolve or compile a kernel")
    monkeypatch.setattr("b12x._lib.dense_gemm._get_compiled_dense_gemm", forbidden)
    # Drain reference temporaries before comparing allocator state across replay.
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    for replay_index in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.equal(folded(replay_partial).view(torch.uint8), replay_expected.view(torch.uint8))
