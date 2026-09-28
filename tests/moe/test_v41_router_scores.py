"""Router score GEMM for the V4.1 and DeepSeek V4 gate shapes."""

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
