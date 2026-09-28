"""Fused N64/N128/N192 V4.1 expert math and runtime-M graph replay."""

import json, pytest, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe.fused_moe._impl import (
    _logical_weight_to_w4a8_rp_inplace,
    _e8m0_scale_to_w4a8_sfb_inplace,
)
from tests.moe.test_v41_expert_numerics import reference
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel


# V4.1 TP4 and dSpark widths, plus the DeepSeek V4 Flash TP4 shard (hidden 4096).
@pytest.mark.parametrize("h,n", [(5120, 576), (5120, 2304), (4096, 512)])
@pytest.mark.parametrize("width", [64, 128, 192])
def test_v41_fused_slice(width, h, n):
    if -(-n // width) * width > (n + 127) // 128 * 128:
        pytest.skip("slice width does not tile this intermediate within its packed storage")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")
    torch.manual_seed(4106576)
    weights = {}
    scales = {}
    for name, shape in [
        ("w1", (1, n, h // 2)),
        ("w3", (1, n, h // 2)),
        ("w2", (1, h, n // 2)),
    ]:
        weights[name] = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
        scales[name] = torch.randint(
            121, 124, (*shape[:-1], shape[-1] // 16), device="cuda", dtype=torch.uint8
        )
    w13 = _logical_weight_to_w4a8_rp_inplace(
        torch.cat([weights["w3"], weights["w1"]], 1),
        size_k=h,
        size_n=n * 2,
        gated_half_rows=n,
    )
    s13 = _e8m0_scale_to_w4a8_sfb_inplace(
        torch.cat([scales["w3"], scales["w1"]], 1),
        weight_E=1,
        rows=n * 2,
        k_dim=h,
        gated_half_rows=n,
    )
    w2 = _logical_weight_to_w4a8_rp_inplace(weights["w2"].clone(), size_k=n, size_n=h)
    s2 = _e8m0_scale_to_w4a8_sfb_inplace(
        scales["w2"].clone(), weight_E=1, rows=h, k_dim=n
    )
    packed = [t.view(torch.uint32).flatten() for t in [w13, s13, w2, s2]]
    x = torch.randn(16, h, device="cuda").mul_(0.5).bfloat16()
    routing = torch.rand(16, device="cuda").mul_(1.5)
    qa = torch.empty((16, h // 4), device="cuda", dtype=torch.uint32)
    qs = torch.empty((16, h // 32), device="cuda", dtype=torch.uint8)

    def encode():
        blocks = x.float().reshape(16, h // 32, 32)
        exp = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
        qa.copy_(
            (blocks / torch.exp2(exp)[..., None])
            .to(torch.float8_e4m3fn)
            .view(torch.uint32)
            .reshape(16, h // 4)
        )
        qs.copy_((exp + 127).to(torch.uint8))

    encode()
    out = torch.empty(
        ((n + width - 1) // width, 16, h), device="cuda", dtype=torch.float32
    )
    args = [from_dlpack(t, assumed_align=16) for t in [qa, qs, *packed, routing, out]]
    compiled = cute.compile(
        V41FusedSliceKernel(width, intermediate=n, hidden=h), *args, cutlass.Int32(16), current_cuda_stream()
    )
    results = []
    for m in [1, 2, 6, 16, 1]:
        compiled(*args, m, current_cuda_stream())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(*args, m, current_cuda_stream())
        x.mul_(-0.75)
        routing.mul_(0.9)
        encode()
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == before
        actual = out[:, :m].sum(0).bfloat16().float()
        wanted = reference(
            x[:m],
            torch.zeros((m, 1), device="cuda", dtype=torch.int32),
            routing[:m, None],
            weights,
            scales,
        )
        rel = ((actual - wanted).norm() / wanted.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            actual.flatten(), wanted.flatten(), dim=0
        ).item()
        print(
            json.dumps(
                dict(
                    width=width,
                    intermediate=n,
                    rows=m,
                    rel_l2=rel,
                    cosine=cos,
                    nonzero=int(torch.count_nonzero(actual)),
                    invalid_row_nonzero=int(torch.count_nonzero(out[:, m:])),
                )
            ),
            flush=True,
        )
        assert rel < 0.01 and cos > 0.9999
        assert torch.count_nonzero(out[:, m:]) == 0
        graph.reset()
    compiled(*args, 0, current_cuda_stream())
    assert torch.count_nonzero(out) == 0
    x.zero_()
    encode()
    compiled(*args, 16, current_cuda_stream())
    assert torch.count_nonzero(out) == 0
