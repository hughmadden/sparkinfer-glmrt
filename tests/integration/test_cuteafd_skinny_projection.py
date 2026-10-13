"""Routed BF16 projection (skinny GEMV for few rows, TMA tensor-core GEMM above)
used by the compressor projection, the index head-weight projection and (skinny
only) the router scores.

Accuracy contract: every output is within the rounding of one
FP32-accumulated dot product of the FP64 reference,
``|out - ref| <= K * 2^-24 * sum_k |x_k w_k| (+ half an ulp of the output
dtype)``. A separate FP32 reference obeys the same FP64 bound.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def _compile(n, k, out_dtype, max_skinny_rows=None):
    import cutlass
    import cutlass.cute as cute

    from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
    from b12x._lib.utils import current_cuda_stream, make_ptr
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection

    cdt = cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float32
    launch = RoutedBf16Projection(n, k, out_dtype=cdt, max_skinny_rows=max_skinny_rows)
    ptr = lambda dt: make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
    compiled = compile_cute(launch, ptr(cutlass.BFloat16), ptr(cutlass.BFloat16), ptr(cdt),
                            cutlass.Int32(1), current_cuda_stream(),
                            compile_spec=KernelCompileSpec.from_key("test.skinny_route", 1, launch.key()))

    def run(x, w, out):
        p = lambda t, dt: make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
        compiled(p(x, cutlass.BFloat16), p(w, cutlass.BFloat16), p(out, cdt), cutlass.Int32(x.shape[0]),
                 current_cuda_stream())

    return run, launch.max_skinny_rows


def _bound(out, x, w):
    ref = x.double() @ w.double().t()
    bound = x.shape[1] * 2.0 ** -24 * (x.double().abs() @ w.double().abs().t())
    if out.dtype == torch.bfloat16:
        bound = bound + ref.abs() * 2.0 ** -8 + 1e-30
    return ref, bound


def _check(out, x, w):
    ref, bound = _bound(out, x, w)
    err = (out.double() - ref).abs()
    assert bool((err <= bound).all()), f"max excess {float((err - bound).max())}"


@pytest.mark.parametrize(("n", "k", "dtype"), [
    (2560, 4096, torch.bfloat16), (1024, 4096, torch.bfloat16), (2560, 7168, torch.bfloat16),
    (1024, 7168, torch.bfloat16), (64, 4096, torch.bfloat16), (64, 7168, torch.bfloat16),
    (256, 4096, torch.float32), (384, 7168, torch.float32)])
def test_routed_projection_accuracy_across_route_boundary(n, k, dtype):
    device = require_b12x()
    run, threshold = _compile(n, k, dtype)
    gen = torch.Generator().manual_seed(n + k)
    w = (torch.randn((n, k), generator=gen) / 16).bfloat16().to(device)
    for rows in sorted({1, 2, 7, 8, 9, 16, threshold, threshold + 1, 200, 1000}):
        x = torch.randn((rows, k), generator=gen).bfloat16().to(device)
        out = torch.full((rows, n), float("nan"), dtype=dtype, device=device)
        run(x, w, out)
        torch.cuda.synchronize()
        _check(out, x, w)
        # FP64 bounds define the contract, not cuBLAS's device-dependent
        # BF16 reduction order or its rate of last-bit flips.
        # TF32 is not an FP32-dot-product reference.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            reference = x.float() @ w.float().t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        _check(reference, x, w)
        _, bound = _bound(out, x, w)
        assert bool(((out.double() - reference.double()).abs() <= 2 * bound).all())


def test_routed_projection_replays_in_cuda_graph():
    device = require_b12x()
    run, _ = _compile(2560, 4096, torch.bfloat16)
    w = torch.randn((2560, 4096), device=device).bfloat16()
    x = torch.randn((5, 4096), device=device).bfloat16()
    out = torch.empty((5, 2560), device=device, dtype=torch.bfloat16)
    run(x, w, out)
    torch.cuda.synchronize()
    eager = out.clone()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        run(x, w, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
