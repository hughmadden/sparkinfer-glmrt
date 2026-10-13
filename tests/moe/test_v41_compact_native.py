"""Compact V4.1 routing, FP32 output, tails and frozen-capacity replay."""

import pytest
import torch

from tests._reference.helpers import prepare_tp_moe_fp4_experts
from tests.moe.test_v41_expert_numerics import quantized_rows


@pytest.mark.parametrize('n', [576, 1152])
def test_native_compact_boundaries_and_live_rows(n):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a8_compact_micro import (
        _layout, launch_w4a8_compact_micro, micro_scratch_nbytes,
    )

    h, e, cap, topk = 5120, 8, 4, 6
    x = torch.full((cap, h), .5, device='cuda', dtype=torch.bfloat16)
    ids = torch.arange(topk, device='cuda', dtype=torch.int32).repeat(cap, 1)
    routing = torch.ones(cap, topk, device='cuda')
    ones = torch.ones(e, device='cuda')
    w13 = torch.full((e, 2*n, h//2), 0x22, device='cuda', dtype=torch.uint8)
    s13 = torch.full((e, 2*n, h//32), 121, device='cuda', dtype=torch.uint8)
    w2 = torch.full((e, h, n//2), 0x22, device='cuda', dtype=torch.uint8)
    s2 = torch.full((e, h, n//32), 119, device='cuda', dtype=torch.uint8)
    experts = prepare_tp_moe_fp4_experts(
        a=x, a1_gscale=ones, w1_fp4=w13, w1_blockscale=s13,
        w1_alphas=ones, a2_gscale=ones, w2_fp4=w2,
        w2_blockscale=s2, w2_alphas=ones, activation='silu',
        quant_mode='w4a8_mx', source_format='fp4_e8m0_k32', swiglu_limit=10)
    runtime = experts._impl.representation_for('w4a8_mx')
    scratch = torch.empty(micro_scratch_nbytes(cap, h, n, topk, True),
                          device='cuda', dtype=torch.uint8)
    offset, size = _layout(cap, topk, h, n, True)['route_output']
    all_routes = scratch[offset:offset+size].view(torch.float32).view(cap*topk, h)

    def run(rows):
        return launch_w4a8_compact_micro(
            scratch=scratch, a=x[:rows], topk_ids=ids[:rows],
            topk_weights=routing[:rows], w13=runtime.w13_rp,
            w13_scales=runtime.w13_sfb, w2=runtime.w2_rp,
            w2_scales=runtime.w2_sfb, alpha1=ones, alpha2=ones,
            input_scale=ones, down_scale=ones, max_tokens=cap,
            num_topk=topk, swiglu_limit=10, fast_math=False, native_v41=True)

    run(cap)
    torch.cuda.synchronize()
    with kernel_resolution_guard('compact native capacity must cover live rows'):
        for rows in (1, 3, 4):
            run(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run(rows)
            for value, weight, invalid in [(.5, .25, False), (.01, .125, False),
                                            (1e-10, 1., False), (.5, 1e-12, False),
                                            (.5, 1., True)]:
                x.fill_(value); routing.fill_(weight)
                ids.fill_(-1 if invalid else 2)
                # Independent constant-weight oracle, including native FP8 floor.
                gate = (quantized_rows(x[:rows]).sum(-1) / 64).bfloat16().float().clamp_max(10)
                up = (quantized_rows(x[:rows]).sum(-1) / 64).bfloat16().float().clamp(-10, 10)
                mid = (torch.nn.functional.silu(gate) * up * weight).bfloat16()
                expected = quantized_rows(mid[:, None].expand(rows, n).contiguous()).sum(-1) / 256
                if invalid: expected.zero_()
                all_routes.fill_(float('nan'))
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual.view(rows, topk, h),
                    expected[:, None, None].expand(rows, topk, h), rtol=1e-6, atol=0)
                assert torch.isnan(all_routes[rows*topk:]).all()
            graph.reset()


@pytest.mark.parametrize('n,hybrid', [(1152,False),(1152,True),(576,True)])
def test_compact_wire_pipeline_live_rows(n, hybrid):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_compact_pipeline import V41CompactPipeline, V41HybridPipeline

    cap, h, e, topk = 16 if hybrid else 4, 5120, 384, 6
    ones = torch.ones(e, device='cuda')
    # Only six experts are addressed; unused pool contents cannot affect output.
    w13 = torch.empty((e, 2*n, h//2), device='cuda', dtype=torch.uint8)
    s13 = torch.empty((e, 2*n, h//32), device='cuda', dtype=torch.uint8)
    w2 = torch.empty((e, h, n//2), device='cuda', dtype=torch.uint8)
    s2 = torch.empty((e, h, n//32), device='cuda', dtype=torch.uint8)
    w13[:6].fill_(0x22); s13[:6].fill_(121)
    w2[:6].fill_(0x22); s2[:6].fill_(119)
    experts = prepare_tp_moe_fp4_experts(
        a=torch.empty(cap,h,device='cuda',dtype=torch.bfloat16),
        a1_gscale=ones, w1_fp4=w13, w1_blockscale=s13, w1_alphas=ones,
        a2_gscale=ones, w2_fp4=w2, w2_blockscale=s2, w2_alphas=ones,
        activation='silu_v41' if hybrid else 'silu', quant_mode='w4a8_mx', source_format='fp4_e8m0_k32', swiglu_limit=10)
    rt = experts._impl.representation_for('w4a8_mx')
    padded_n = (n+127)//128*128
    assert rt.w13_rp.numel()*rt.w13_rp.element_size() == e*padded_n*h
    assert rt.w2_rp.numel()*rt.w2_rp.element_size() == e*padded_n*h//2
    wire = torch.empty(cap,5280,device='cuda',dtype=torch.uint8)
    ids = torch.arange(6,device='cuda',dtype=torch.int32).repeat(cap)
    routing = torch.full((cap*topk,),.25,device='cuda')
    dummy = torch.empty(1,device='cuda',dtype=torch.int32)
    mid = torch.empty(cap*topk*(n+(n//128)*4)//4,device='cuda',dtype=torch.uint32)
    projected = torch.empty(cap*topk,2*n,device='cuda',dtype=torch.bfloat16)
    output = torch.empty(cap*topk,h,device='cuda')
    tensors = [wire[:,:h].view(torch.uint32),wire[:,h:],
        *[t.view(torch.uint32).flatten() for t in (rt.w13_rp,rt.w13_sfb,rt.w2_rp,rt.w2_sfb)],
        ids,routing,dummy,mid,torch.zeros_like(ones),dummy,dummy,dummy,dummy,projected,output]
    if hybrid:
        planes = (n+191)//192
        tensors[9] = torch.empty(e*cap*topk, device='cuda', dtype=torch.int32)
        tensors[10] = torch.empty(e, device='cuda', dtype=torch.int32)
        tensors[11] = torch.empty(e,2, device='cuda', dtype=torch.int32)
        tensors[12] = torch.empty(cap*topk,19, device='cuda', dtype=torch.int32)
        tensors[13] = torch.empty(cap*topk, device='cuda')
        tensors[14] = torch.empty(cap*topk, device='cuda', dtype=torch.int32)
        tensors[15] = torch.empty(planes,cap*topk,h, device='cuda')
    args = [from_dlpack(t,assumed_align=16) for t in tensors]
    pipeline = (V41HybridPipeline(cap,192,intermediate=n,cutoff=8 if n==1152 else 2)
                if hybrid else V41CompactPipeline(cap))
    compiled = cute.compile(pipeline,
                            *args,cutlass.Int32(1),current_cuda_stream())
    for rows in ((1,2,4,8,16,1,16,2) if hybrid else (1,3,4)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): compiled(*args,rows,current_cuda_stream())
        for value, invalid in [(256.,False),(128.,False),(0.,False),(256.,True)]:
            code = torch.tensor(value,device='cuda',dtype=torch.float8_e4m3fn).view(torch.uint8)
            wire[:,:h].fill_(int(code.item()));wire[:,h:].fill_(118)
            ids.fill_(-1 if invalid else 2)
            tensors[8].fill_(777)
            output.fill_(float('nan'))
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay();torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated']==allocations
            grouped_expected = hybrid and rows > (8 if n==1152 else 2)
            assert int(tensors[8][0]) == (rows if grouped_expected else 777)
            gate = min(value*2**-9*h/64,10.)
            activated = torch.nn.functional.silu(torch.tensor(gate,device='cuda'))*gate*.25
            mid_ref = activated.bfloat16().expand(rows,n).contiguous()
            expected = quantized_rows(mid_ref).sum(-1)/256
            if invalid: expected.zero_()
            torch.testing.assert_close(output[:rows*topk].view(rows,topk,h),
                expected[:,None,None].expand(rows,topk,h),rtol=1e-6,atol=0)
            assert torch.isnan(output[rows*topk:]).all()
        graph.reset()
