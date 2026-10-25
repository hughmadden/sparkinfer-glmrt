"""Router score GEMM for the V4.1 and DeepSeek V4 gate shapes."""

import cutlass
import pytest
import torch
import cutlass.cute as cute
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.v41_router import compile_v41_router_scores_aot


@pytest.mark.parametrize("experts,hidden", [(384, 5120), (128, 5120), (256, 4096), (384, 7168)])
def test_router_scores(experts, hidden):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")
    torch.manual_seed(experts + hidden)
    compiled = compile_v41_router_scores_aot(experts=experts, hidden=hidden)
    w = (torch.randn(experts, hidden, device="cuda") * 0.02).bfloat16()
    for rows in (1, 7, 64, 300):
        x = torch.randn(rows, hidden, device="cuda").bfloat16()
        out = torch.full((rows, experts), float("nan"), device="cuda")
        ptr = lambda t, dtype: make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        import cutlass
        compiled(ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16), ptr(out, cutlass.Float32),
                 rows, current_cuda_stream())
        torch.cuda.synchronize()
        ref = x.float() @ w.float().T
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4), (rows, (out - ref).abs().max().item())


@pytest.mark.parametrize("experts", [384, 128])
def test_router_scores_rows_match_single_row_decode(experts):
    """The canonical skinny M=1 logits must survive every verify width (#736)."""
    from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, skinny_config
    from b12x._lib.compiler import compile as compile_kernel

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")
    hidden = 5120
    torch.manual_seed(experts + hidden)
    w = (torch.randn(experts, hidden, device="cuda") / 64).bfloat16()
    x = torch.randn(64, hidden, device="cuda").bfloat16()
    ptr = lambda t, dtype: make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    scratch = torch.empty(64, experts, device="cuda")
    args = (ptr(x, cutlass.BFloat16), ptr(w, cutlass.BFloat16),
            ptr(scratch, cutlass.Float32))
    skinny = compile_kernel(
        SkinnyBf16Gemv(experts, hidden, out_dtype=cutlass.Float32,
                       **skinny_config(experts, hidden, cutlass.Float32)),
        *args, cutlass.Int32(1), current_cuda_stream())
    compiled = compile_v41_router_scores_aot(experts=experts)

    def run(kernel, row_x):
        out = torch.full((len(row_x), experts), float("nan"), device="cuda")
        kernel(ptr(row_x, cutlass.BFloat16), ptr(w, cutlass.BFloat16),
               ptr(out, cutlass.Float32), cutlass.Int32(len(row_x)), current_cuda_stream())
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all())
        return out

    # Pin the accepted canonical M=1 route too: the old MMA-only export must fail.
    expected = torch.cat([run(skinny, x[r:r + 1]) for r in range(64)])
    single = torch.cat([run(compiled, x[r:r + 1]) for r in range(64)])
    assert torch.equal(single.view(torch.int32), expected.view(torch.int32))
    for rows in range(1, 65):
        actual = run(compiled, x[:rows])
        assert torch.equal(actual.view(torch.int32), single[:rows].view(torch.int32)), rows
