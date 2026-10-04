"""Half-head W8 KDA: precise partials, transient prefill expansion and graph replay."""
from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_w8_aot import _quant_rows


@pytest.fixture(scope="module")
def case():
    require_b12x()
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    g = replace(GLM53_FLASH, kda_heads=32)
    torch.manual_seed(123)
    rand = lambda *shape: torch.randn(shape, device="cuda")
    iq = _quant_rows((rand(g.kda_in_width, g.hidden) * .02).bfloat16())
    oq = _quant_rows((rand(g.hidden, g.kda_width) * .01).bfloat16())
    common = [
        (rand(2, g.kda_width, 128) * .05).bfloat16(), rand(3 * g.kda_width, 4) * .3,
        torch.log(torch.rand(g.kda_heads, device="cuda") * 15 + 1), rand(g.kda_width) * .1,
        (1 + rand(128) * .1).bfloat16(),
    ]
    programs = {(mode, dtype, expanded): glmf.compile_glmf_kda_aot(
        g, max_rows=64 if mode == "decode" else 4096, fp8_only=mode,
        output_dtype=dtype, prefill_expanded=expanded)
        for mode in ("decode", "prefill") for dtype in ("bfloat16", "float32")
        for expanded in ((False, True) if mode == "prefill" else (False,))}
    replay_size = kda_replay_layout(32, 3 * g.kda_width)[2] // 4
    return g, iq, oq, common, programs, replay_size


@pytest.mark.parametrize("mode,rows,bits", [("decode", 1, 16), ("decode", 24, 32),
                                           ("prefill", 64, 0), ("prefill", 512, 0),
                                           ("prefill", 64, 1)])
def test_partial_dtype_keeps_dot_product_and_state(case, mode, rows, bits):
    g, iq, oq, common, programs, replay_size = case
    torch.manual_seed(rows)
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(rows, device="cuda", dtype=torch.int32)
    initial_state = torch.randn(1, 32, 128, 128, device="cuda") * .01
    initial_conv = (torch.randn(1, 3, 3 * g.kda_width, device="cuda") * .1).bfloat16()
    results = {}
    for dtype in ("bfloat16", "float32"):
        program = programs[mode, dtype, False]
        out = torch.full((rows, g.hidden), float("nan"), device="cuda",
                         dtype=getattr(torch, dtype))
        state, conv = initial_state.clone(), initial_conv.clone()
        replay = torch.zeros(replay_size, device="cuda")
        scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
        args = [x, iq[0], iq[2], *common, oq[0], oq[2], conv, state, slots, slots, out]
        if mode == "decode":
            args.append(replay)
        args.append(scratch)
        scalars = (rows, bits, 1) if mode == "decode" else (rows, bits)
        program.launch(*args, scalars=scalars)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all() and out.abs().max() > 0
        results[dtype] = (out, state, conv, replay)
    for dtype in ("bfloat16",):
        assert torch.equal(results[dtype][0], results['float32'][0].to(getattr(torch, dtype)))
        for old, new in zip(results[dtype][1:], results['float32'][1:]):
            assert torch.equal(old, new)
    assert (results['float32'][0] != results['float32'][0].bfloat16().float()).any()


@pytest.mark.parametrize('dtype', ['float32'])
def test_partial_sum_is_one_rounding_with_graph_replay(dtype):
    require_b12x()
    from b12x.integration.cuteafd import glmf

    program = glmf.compile_glmf_add_fp32_aot()
    for rows in (1, 4, 24, 64):
        a, b = [torch.randn(rows, 4096, device="cuda").to(getattr(torch, dtype)) for _ in range(2)]
        out = torch.full_like(a, float("nan"), dtype=torch.bfloat16)
        program.launch(a, b, out, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out, (a.float() + b.float()).bfloat16())
        assert not torch.equal(out, (a.bfloat16() + b.bfloat16()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(a, b, out, scalars=(rows,))
        a.mul_(2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, (a.float() + b.float()).bfloat16())


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('rows,bits', [(64, 0), (512, 0), (512, 1)])
def test_expanded_prefill_is_byte_exact_and_fits_full_workspace(case, dtype, rows, bits):
    g, iq, oq, common, programs, replay_size = case
    torch.manual_seed(rows)
    x = torch.randn(rows, g.hidden, device='cuda', dtype=torch.bfloat16)
    slots = torch.zeros(rows, device='cuda', dtype=torch.int32)
    initial_state = torch.randn(1, 32, 128, 128, device='cuda') * .01
    initial_conv = (torch.randn(1, 3, 3 * g.kda_width, device='cuda') * .1).bfloat16()
    results=[]
    for expanded in (False, True):
        program=programs['prefill', dtype, expanded]
        out=torch.full((rows,g.hidden), float('nan'), device='cuda', dtype=getattr(torch,dtype))
        state,conv=initial_state.clone(),initial_conv.clone()
        scratch=torch.empty(program.scratch_bytes(rows)['scratch'] + 1024, device='cuda', dtype=torch.uint8)
        scratch[-1024:].fill_(0xA5)
        program.launch(x,iq[0],iq[2],*common,oq[0],oq[2],conv,state,slots,slots,out,scratch,
                       scalars=(rows,bits))
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert (scratch[-1024:] == 0xA5).all()
        results.append((out,state,conv))
    for old,new in zip(*results):
        assert torch.equal(old,new)
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    full=glmf.compile_glmf_kda_aot(GLM53_FLASH,max_rows=4096,fp8_only='prefill')
    assert programs['prefill',dtype,True].scratch_bytes(4096)['scratch'] <= full.scratch_bytes(4096)['scratch']
