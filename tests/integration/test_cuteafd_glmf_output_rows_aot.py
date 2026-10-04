"""Owned-token output rows preserve full-head math and global route thresholds."""
from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_w8_aot import _quant_rows
from .test_cuteafd_glmf_output_shard_aot import _ProjectionOracle, _guarded, _guard


@pytest.fixture(scope='module')
def case():
    require_b12x()
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    g = replace(GLM53_FLASH, kda_heads=32)
    torch.manual_seed(4573)
    w8, _, scale, _ = _quant_rows(torch.randn(4096, 8192, device='cuda').bfloat16() * .01)
    broad = torch.exp2(torch.randint(-20, 21, scale.shape, device='cuda').float())
    broad *= .75 + torch.rand_like(broad) * .5
    programs, oracles = {}, {}
    for mode in ['decode', 'prefill']:
        for expanded in ([False] if mode == 'decode' else [False, True]):
            programs[mode, expanded] = glmf.compile_glmf_kda_output_rows_aot(g,
                max_rows=64 if mode == 'decode' else 4096, fp8_only=mode, prefill_expanded=expanded)
        fn = glmf._w8(4096, 8192, 4096 if mode == 'prefill' else None,
                      row_scales=True, prefill_mask=0, wide_rows=32)
        oracles[mode] = compile_program(_ProjectionOracle(fn), name='glmf_output_rows_full_oracle',
            operands=(Operand('x', torch.bfloat16, '[rows,8192]'),
                      Operand('w', torch.float8_e4m3fn, '[4096,8192]'),
                      Operand('s', torch.float32, '[64,4096]'),
                      Operand('out', torch.bfloat16, '[rows,4096]', 'out'),
                      Operand('scratch', torch.uint8, '[scratch_bytes]', 'scratch')),
            scalars=(Scalar('rows'),), key=(mode,), geometry={})
    return w8, {'real': scale, 'broad': broad}, programs, oracles


CASES = [('decode', 1, False), ('decode', 22, False), ('decode', 63, False), ('decode', 64, False),
         ('prefill', 512, False), ('prefill', 513, False), ('prefill', 4096, False),
         ('prefill', 512, True), ('prefill', 513, True), ('prefill', 4096, True)]


@pytest.mark.parametrize('mode,total,expanded', CASES)
@pytest.mark.parametrize('recipe', ['real', 'broad'])
def test_output_rows_match_full_head_and_changed_input_graph(case, mode, total, expanded, recipe):
    w, scales, programs, oracles = case
    s, p, oracle = scales[recipe], programs[mode, expanded], oracles[mode]
    x = torch.randn(total, 8192, device='cuda', dtype=torch.bfloat16)
    expected = torch.empty(total, 4096, device='cuda', dtype=torch.bfloat16)
    split = (total + 1) // 2
    for start, owned in [(0, split), (split, total - split)]:
        selected = slice(start, start + owned)
        owned_x = x[selected] if owned else x[:1]
        out, guard = _guarded(max(1, owned), 4096)
        scratch_bytes = p.scratch_bytes(total)['scratch']
        if expanded:
            assert scratch_bytes == 64 << 20
        else:
            assert scratch_bytes == 0
        scratch = torch.full((scratch_bytes + 2048,), 0xA5, device='cuda', dtype=torch.uint8)
        workspace = scratch[1024:-1024] if scratch_bytes else scratch[1024:1280]
        p.launch(owned_x, w, s, out, workspace, scalars=(owned, total))
        oracle.launch(x, w, s, expected, None, scalars=(total,))
        torch.cuda.synchronize()
        if owned:
            assert torch.equal(out, expected[selected])
            assert torch.isfinite(out).all() and out.abs().max() > 0
        else:
            assert (guard == 0xA5).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p.launch(owned_x, w, s, out, workspace, scalars=(owned, total))
        x.mul_(1.125)
        oracle.launch(x, w, s, expected, None, scalars=(total,))
        graph.replay()
        torch.cuda.synchronize()
        if owned:
            assert torch.equal(out, expected[selected])
        else:
            assert (guard == 0xA5).all()
        _guard(guard)
        _guard(scratch)


@pytest.mark.parametrize('mode,total,expanded', [('decode', 64, False), ('prefill', 512, False),
                                               ('prefill', 513, True), ('prefill', 4096, True)])
def test_zero_owned_projection_preserves_output_and_scratch(case, mode, total, expanded):
    w, scales, programs, _ = case
    p = programs[mode, expanded]
    x = torch.empty(1, 8192, device='cuda', dtype=torch.bfloat16)
    out, guard = _guarded(1, 4096)
    scratch = torch.full((p.scratch_bytes(total)['scratch'] + 2048,), 0xA5,
                         device='cuda', dtype=torch.uint8)
    workspace = scratch[1024:-1024] if scratch.numel() > 2048 else scratch[1024:1280]
    p.launch(x, w, scales['real'], out, workspace, scalars=(0, total))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(x, w, scales['real'], out, workspace, scalars=(0, total))
    graph.replay()
    torch.cuda.synchronize()
    assert (guard == 0xA5).all() and (scratch == 0xA5).all()


@pytest.mark.parametrize('a_rows,b_rows', [(0, 0), (1, 0), (0, 1), (11, 11), (32, 31),
                                         (32, 32), (256, 256), (257, 256), (2048, 2048)])
def test_join_rows_preserves_bits_canaries_and_changed_input_graph(a_rows, b_rows):
    require_b12x()
    from b12x.integration.cuteafd import glmf
    p = glmf.compile_glmf_join_rows_aot(4096)
    a = torch.randint(-32768, 32767, (max(1, a_rows), 4096), device='cuda', dtype=torch.int16).view(torch.bfloat16)
    b = torch.randint(-32768, 32767, (max(1, b_rows), 4096), device='cuda', dtype=torch.int16).view(torch.bfloat16)
    total = a_rows + b_rows
    out, guard = _guarded(max(1, total), 4096)
    p.launch(a, b, out, scalars=(a_rows, b_rows))
    torch.cuda.synchronize()
    if total:
        assert torch.equal(out.view(torch.int16), torch.cat([a[:a_rows], b[:b_rows]]).view(torch.int16))
    else:
        assert (guard == 0xA5).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(a, b, out, scalars=(a_rows, b_rows))
    a.view(torch.int16).bitwise_xor_(713)
    graph.replay()
    torch.cuda.synchronize()
    if total:
        assert torch.equal(out.view(torch.int16), torch.cat([a[:a_rows], b[:b_rows]]).view(torch.int16))
    else:
        assert (guard == 0xA5).all()
    _guard(guard)
