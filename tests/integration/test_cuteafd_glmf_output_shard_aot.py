"""KDA normalized-head output, full-K output shards, and bit-exact row joins."""
from dataclasses import replace

import pytest
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_w8_aot import _quant_rows
from .test_cuteafd_glmf_split_w8_aot import case


class _ProjectionOracle:
    def __init__(self, projection):
        self.projection = projection

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, s: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.projection(x, w, s, out, rows, Int32(32), Int64(scratch.toint()), stream)


def _guarded(rows, width):
    storage = torch.full((rows * width * 2 + 2048,), 0xA5, device='cuda', dtype=torch.uint8)
    return storage[1024:-1024].view(torch.bfloat16).reshape(rows, width), storage


def _guard(storage):
    assert (storage[:1024] == 0xA5).all() and (storage[-1024:] == 0xA5).all()


@pytest.mark.parametrize('rows', [1, 22, 64, 512, 4096])
@pytest.mark.parametrize('width', [2048, 4096])
def test_join_bits_canaries_and_graph(rows, width):
    require_b12x()
    from b12x.integration.cuteafd import glmf
    p = glmf.compile_glmf_join_aot(width)
    a, b = [torch.randint(-32768, 32767, (rows, width), device='cuda', dtype=torch.int16).view(torch.bfloat16)
            for _ in range(2)]
    out, storage = _guarded(rows, 2 * width)
    p.launch(a, b, out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), torch.cat([a, b], dim=1).view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(a, b, out, scalars=(rows,))
    a.view(torch.int16).bitwise_xor_(123)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), torch.cat([a, b], dim=1).view(torch.int16))
    _guard(storage)


@pytest.mark.parametrize('mode,rows,expanded', [('decode', 1, False), ('decode', 22, False),
    ('decode', 64, False), ('prefill', 512, False), ('prefill', 4096, False),
    ('prefill', 512, True), ('prefill', 4096, True)])
def test_norm_is_original_scratch_and_state(case, mode, rows, expanded):
    from b12x.integration.cuteafd import glmf
    g, iq, oq, common, original, replay_size = case
    norm = glmf.compile_glmf_kda_aot(g, max_rows=64 if mode == 'decode' else 4096,
        fp8_only=mode, prefill_expanded=expanded, output_kind='norm')
    torch.manual_seed(rows)
    x = torch.randn(rows, g.hidden, device='cuda', dtype=torch.bfloat16)
    slots = torch.zeros(rows, device='cuda', dtype=torch.int32)
    initial_state = torch.randn(1, 32, 128, 128, device='cuda') * .01
    initial_conv = (torch.randn(1, 3, 3 * g.kda_width, device='cuda') * .1).bfloat16()
    results = []
    for p in [original[mode, 'bfloat16', expanded], norm]:
        out, guard = _guarded(rows, g.kda_width)
        state, conv, replay = initial_state.clone(), initial_conv.clone(), torch.zeros(replay_size, device='cuda')
        scratch = torch.empty(p.scratch_bytes(rows)['scratch'], device='cuda', dtype=torch.uint8)
        args = [x, iq[0], iq[2], *common, oq[0], oq[2], conv, state, slots, slots, out]
        if mode == 'decode':
            args.append(replay)
        args.append(scratch)
        scalars = (rows, 32, 1) if mode == 'decode' else (rows, 0)
        p.launch(*args, scalars=scalars)
        torch.cuda.synchronize()
        y_at = sum(glmf._align(rows * size * 2) for size in [g.kda_in_width, 2 * g.kda_width,
                                                          3 * g.kda_width, g.kda_width])
        y = scratch[y_at:y_at + rows * g.kda_width * 2].view(torch.bfloat16).reshape(rows, g.kda_width).clone()
        results.append((out.clone(), state.clone(), conv.clone(), replay.clone(), y))
        if p is norm:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                p.launch(*args, scalars=scalars)
            state.copy_(initial_state)
            conv.copy_(initial_conv)
            replay.zero_()
            graph.replay()
            torch.cuda.synchronize()
            for actual, expected in zip([out, state, conv, replay], results[-1][:4]):
                assert torch.equal(actual, expected)
        _guard(guard)
    assert torch.equal(results[1][0], results[0][4])
    assert torch.isfinite(results[1][0]).all() and results[1][0].abs().max() > 0
    for a, b in zip(results[0][1:4], results[1][1:4]):
        assert torch.equal(a, b)


@pytest.fixture(scope='module')
def projections():
    require_b12x()
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    g = replace(GLM53_FLASH, kda_heads=32)
    w8, _, scale, _ = _quant_rows(torch.randn(4096, 8192, device='cuda').bfloat16() * .01)
    halves = [(w8[:2048].contiguous(), scale[:, :2048].contiguous()),
              (w8[2048:].contiguous(), scale[:, 2048:].contiguous())]
    programs = {}
    for mode in ['decode', 'prefill']:
        for expanded in ([False] if mode == 'decode' else [False, True]):
            shard = glmf.compile_glmf_kda_output_shard_aot(g, max_rows=64 if mode == 'decode' else 4096,
                fp8_only=mode, prefill_expanded=expanded)
            # Same full-K GEMV accumulation grouping at both decode widths;
            # current one-GPU serving only exposes the <=16 route.
            fn = glmf._w8(4096, 8192, 4096 if mode == 'prefill' else None,
                          row_scales=True, prefill_mask=0, wide_rows=32)
            full = compile_program(_ProjectionOracle(fn), name='glmf_output_full_oracle',
                operands=(Operand('x', torch.bfloat16, '[rows,8192]'),
                          Operand('w', torch.float8_e4m3fn, '[4096,8192]'),
                          Operand('s', torch.float32, '[64,4096]'),
                          Operand('out', torch.bfloat16, '[rows,4096]', 'out'),
                          Operand('scratch', torch.uint8, '[scratch_bytes]', 'scratch')),
                scalars=(Scalar('rows'),), key=(mode,), geometry={}, scratch={})
            programs[mode, expanded] = shard, full
    return g, w8, scale, halves, programs


@pytest.mark.parametrize('mode,rows,expanded', [('decode', 1, False), ('decode', 22, False),
    ('decode', 64, False), ('prefill', 512, False), ('prefill', 4096, False),
    ('prefill', 512, True), ('prefill', 4096, True)])
def test_output_rows_match_full_k_and_graph(projections, mode, rows, expanded):
    _, w8, scale, halves, programs = projections
    shard, full = programs[mode, expanded]
    x = torch.randn(rows, 8192, device='cuda', dtype=torch.bfloat16)
    expected = torch.empty(rows, 4096, device='cuda', dtype=torch.bfloat16)
    scratch = torch.empty(max(256, shard.scratch_bytes(rows)['scratch']), device='cuda', dtype=torch.uint8)
    full.launch(x, w8, scale, expected, scratch, scalars=(rows,))
    for rank, (w, s) in enumerate(halves):
        out, storage = _guarded(rows, 2048)
        shard.launch(x, w, s, out, scratch, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out, expected[:, rank * 2048:(rank + 1) * 2048])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            shard.launch(x, w, s, out, scratch, scalars=(rows,))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected[:, rank * 2048:(rank + 1) * 2048])
        _guard(storage)


def test_prefix1_input_matches_full_k_grouping(case):
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import Fp8Projection
    g, iq, _, _, _, _ = case
    d, p = g.kda_width, g.kda_in_width
    # Mirror each head segment, keeping the shared f_a/g_a rows single.
    index = torch.cat([torch.arange(start, start + d, device='cuda').repeat(2) for start in [0, d, 2*d]] +
                      [torch.arange(3*d, 3*d+256, device='cuda'),
                       torch.arange(3*d+256, p, device='cuda').repeat(2)])
    full_w, full_s = iq[0][index].contiguous(), iq[2][:, index].contiguous()
    select = torch.cat([torch.arange(start, start+d, device='cuda') for start in [0, 2*d, 4*d]] +
                      [torch.arange(6*d, 6*d+256, device='cuda'),
                       torch.arange(6*d+256, 6*d+256+32, device='cuda')])
    x = torch.randn(1, 4096, device='cuda', dtype=torch.bfloat16)
    scratch = torch.empty(256, device='cuda', dtype=torch.uint8)
    outputs, elapsed = {}, {}
    for name, n, w, s, warps, groups in [('full8/2', GLM53_FLASH.kda_in_width, full_w, full_s, 8, 2),
                                      ('half4/4', p, iq[0], iq[2], 4, 4),
                                      ('half8/2', p, iq[0], iq[2], 8, 2)]:
        fn = Fp8Projection(n, 4096, row_scales=True, kmajor=True, warps=warps, groups=groups)
        prog = compile_program(_ProjectionOracle(fn), name='glmf_input_prefix_oracle',
            operands=(Operand('x', torch.bfloat16, '[rows,4096]'),
                      Operand('w', torch.float8_e4m3fn, f'[{n},4096]'),
                      Operand('s', torch.float32, f'[32,{n}]'),
                      Operand('out', torch.bfloat16, f'[rows,{n}]', 'out'),
                      Operand('scratch', torch.uint8, '[scratch_bytes]', 'scratch')),
            scalars=(Scalar('rows'),), key=(n, warps, groups), geometry={}, scratch={})
        out = torch.empty(1, n, device='cuda', dtype=torch.bfloat16)
        prog.launch(x, w, s, out, scratch, scalars=(1,))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prog.launch(x, w, s, out, scratch, scalars=(1,))
        graph.replay()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100):
            graph.replay()
        b.record(); b.synchronize()
        elapsed[name] = a.elapsed_time(b)*1000/100
        outputs[name] = out
    expected = outputs['full8/2'][:, select]
    assert torch.equal(outputs['half8/2'], expected)
    print('prefix1 full input selected rows: half4/4 differences=',
          (outputs['half4/4'] != expected).sum().item(), 'us=', elapsed, flush=True)


@pytest.mark.parametrize('spec', [0, 1])
@pytest.mark.parametrize('output_kind', ['projection', 'norm'])
def test_prefix1_state_and_replay_match_full_heads(case, spec, output_kind):
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout
    g, iq, oq, common, _, _ = case
    d = g.kda_width
    qkv_index = torch.cat([torch.arange(start, start + d, device='cuda').repeat(2)
                           for start in [0, d, 2 * d]])
    in_index = torch.cat([qkv_index, torch.arange(3*d, 3*d+256, device='cuda'),
                         torch.arange(3*d+256, g.kda_in_width, device='cuda').repeat(2)])
    full_common = [common[0].repeat(1, 2, 1), common[1][qkv_index].contiguous(),
                   common[2].repeat(2), common[3].repeat(2), common[4]]
    torch.manual_seed(953 + spec)
    x = torch.randn(1, g.hidden, device='cuda', dtype=torch.bfloat16)
    slots = torch.zeros(1, device='cuda', dtype=torch.int32)
    state = torch.randn(1, g.kda_heads, 128, 128, device='cuda') * .01
    conv = (torch.randn(1, 3, 3 * d, device='cuda') * .1).bfloat16()
    results = []
    for geometry, w_in, in_scale, weights, w_o, o_scale, initial_state, initial_conv in [
        (g, iq[0], iq[2], common, oq[0], oq[2], state, conv),
        (GLM53_FLASH, iq[0][in_index].contiguous(), iq[2][:, in_index].contiguous(),
         full_common, oq[0].repeat(1, 2), oq[2].repeat(2, 1),
         state.repeat(1, 2, 1, 1), conv[:, :, qkv_index].contiguous())]:
        p = glmf.compile_glmf_kda_aot(geometry, max_rows=64, fp8_only='decode', output_kind=output_kind)
        actual_state, actual_conv = initial_state.clone(), initial_conv.clone()
        out_width = geometry.kda_width if output_kind == 'norm' else geometry.hidden
        out = torch.empty(1, out_width, device='cuda', dtype=torch.bfloat16)
        beta_at, proj_at, total = kda_replay_layout(geometry.kda_heads, 3 * geometry.kda_width)
        replay = torch.zeros(total, device='cuda', dtype=torch.uint8)
        scratch = torch.empty(p.scratch_bytes(1)['scratch'], device='cuda', dtype=torch.uint8)
        p.launch(x, w_in, in_scale, *weights, w_o, o_scale, actual_conv, actual_state,
                 slots, slots, out, replay, scratch, scalars=(1, 16, spec))
        torch.cuda.synchronize()
        records = replay[:beta_at].view(torch.float32).reshape(64, geometry.kda_heads, 3, 128)
        beta = replay[beta_at:proj_at].view(torch.float32).reshape(64, geometry.kda_heads)
        projection = replay[proj_at:].view(torch.bfloat16).reshape(64, 3 * geometry.kda_width)
        results.append((out, actual_state, actual_conv, records, beta, projection))
    half, full = results
    selected_qkv = torch.cat([torch.arange(start, start+d, device='cuda')
                              for start in [0, 2*d, 4*d]])
    expected = (full[0][:, :d], full[1][:, :32], full[2][:, :, selected_qkv],
                full[3][:, :32], full[4][:, :32], full[5][:, selected_qkv])
    if output_kind == 'norm':
        assert torch.equal(half[0], expected[0])
    for actual, reference in zip(half[1:], expected[1:]):
        assert torch.equal(actual, reference)
