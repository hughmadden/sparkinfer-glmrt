#!/usr/bin/env python3
"""Five-shape BF16 projection screen; run under cuteafd hwlock in its dev image."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPES = ((1536, 8192), (1536, 16384), (4096, 4096), (8192, 4096), (16384, 4096))
ROWS = (1, 2, 4, 8, 16, 32, 64)
GATE_ROWS = (1, 4, 8, 16, 64, 128, 140, 2048)
OFFSETS = (0, 1, 17, 63, 64, 65, 127, 139, 511, 1023, 2047)
BASE_TILES = ((16, 1, 4, 16, 64, 6),)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=31)
    p.add_argument('--tiles', type=Path, help='JSON mapping K,N to candidate tiles_by_rows')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if not args.native_lib.is_file() or args.output.exists() or args.samples < 1:
        raise ValueError('existing native library, new output and positive samples required')
    tiles = json.loads(args.tiles.read_text()) if args.tiles else None
    if tiles is not None and set(tiles) != {f'{k},{n}' for k, n in SHAPES}:
        raise ValueError('candidate must explicitly cover all five shapes')
    if args.preflight:
        import ast
        ast.parse(Path(__file__).read_text())
        print('BF16 row-invariant projection preflight passed')
        return

    import torch
    from benchmarks.common import make_l2_flush_fn
    from benchmarks.benchmark_hqa_projection_ceiling import snapshot
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    warnings.filterwarnings('error', message='The CUDA Graph is empty.*')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1045)
    prop = torch.cuda.get_device_properties(0)
    lib = C.CDLL(str(args.native_lib))
    native = lib.cuteafd_cuda_linear_bf16_cublas_async
    native.argtypes = [C.c_void_p] * 4 + [C.c_size_t] * 3 + [C.c_void_p]
    native.restype = C.c_int32
    flush = make_l2_flush_fn(True)
    result = dict(device=prop.name, sms=prop.multi_processor_count, hardware_before=snapshot(),
        torch=torch.__version__, cuda=torch.version.cuda, bandwidth_tb_s=1.8,
        native_lib=str(args.native_lib), native_lib_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        baseline_tiles=BASE_TILES, candidate_tiles=tiles, results=[], gates=[],
        conditions='Alternating arm order; warm graph replay then timed replay; cold flush then timed replay. 1.8 TB/s theoretical compulsory weight/input/output floor. No TF32 oracle. Nonempty poisoned graphs and 11-offset row invariance.')

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    for k, n in SHAPES:
        x = (torch.randn(2048, k, device='cuda') * .1).bfloat16()
        for offset in OFFSETS:
            x[offset].copy_(x[0].clone())
        w = (torch.randn(n, k, device='cuda') * .02).bfloat16()
        reference = x.float() @ w.float().T
        programs = {}
        for arm, config in [('tma', BASE_TILES)] + ([('candidate', tiles[f'{k},{n}'])] if tiles else []):
            kernel = TmaBf16Projection(n, k, tiles_by_rows=config)
            programs[arm] = compile_program(kernel, name='hqa_bf16_rowinv_' + arm, operands=(
                Operand('x', torch.bfloat16, f'[rows,{k}]'),
                Operand('w', torch.bfloat16, f'[{n},{k}]'),
                Operand('out', torch.bfloat16, f'[rows,{n}]', 'out')),
                scalars=(Scalar('rows'),), key=kernel.key())
        outputs = {arm: torch.empty(2048, n, device='cuda', dtype=torch.bfloat16)
                   for arm in ('cublas', *programs)}

        def call(arm, m):
            if arm == 'cublas':
                status = native(x.data_ptr(), w.data_ptr(), None, outputs[arm].data_ptr(),
                                m, k, n, torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f'cuBLAS failed: {status}')
            else:
                programs[arm].launch(x, w, outputs[arm], scalars=(m,))

        graphs = {}
        row_zero = {}
        invariance = {arm: True for arm in outputs}
        for m in sorted(set(ROWS + GATE_ROWS)):
            for arm, out in outputs.items():
                out.fill_(float('nan'))
                call(arm, m)
                torch.cuda.synchronize()
                expected = out[:m].clone()
                assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
                torch.testing.assert_close(expected.float(), reference[:m], rtol=1e-2, atol=1e-2)
                if m == 1:
                    row_zero[arm] = expected[0].clone()
                for offset in OFFSETS:
                    if offset < m:
                        invariance[arm] &= torch.equal(row_zero[arm].view(torch.int16), expected[offset].view(torch.int16))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call(arm, m)
                out.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out[:m].view(torch.int16), expected.view(torch.int16)), (k, n, m, arm, 'graph')
                assert torch.isnan(out[m:]).all(), (k, n, m, arm, 'tail')
                graphs[arm, m] = graph
        for arm in outputs:
            if arm != 'cublas':
                assert invariance[arm], (k, n, arm, 'row invariance')
            result['gates'].append(dict(k=k, n=n, arm=arm, row_invariant=invariance[arm],
                rows=GATE_ROWS, offsets=OFFSETS, graph_poison=True, fp32_oracle=True, tail_guard=True))
        for m in ROWS:
            samples = {mode: {arm: [] for arm in outputs} for mode in ('warm', 'cold')}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            order = list(outputs)
            for i in range(args.samples):
                for arm in order if i % 2 == 0 else order[::-1]:
                    for mode in ('warm', 'cold') if i % 2 == 0 else ('cold', 'warm'):
                        if mode == 'cold':
                            flush()
                        else:
                            graphs[arm, m].replay()
                        start.record()
                        graphs[arm, m].replay()
                        end.record()
                        end.synchronize()
                        samples[mode][arm].append(start.elapsed_time(end) * 1000)
            floor = (2*n*k + 2*m*(n+k)) / 1.8e6
            medians = {mode: {arm: statistics.median(values) for arm, values in arms.items()}
                       for mode, arms in samples.items()}
            cell = dict(k=k, n=n, m=m, floor_us=floor, median_us=medians,
                floor_fraction={mode: {arm: floor/us for arm, us in arms.items()} for mode, arms in medians.items()},
                tma_gap_us=medians['cold']['tma']-min(floor, medians['cold']['cublas']),
                skip_duel=medians['cold']['tma'] <= 1.05*medians['cold']['cublas'], samples_us=samples)
            result['results'].append(cell)
            print(json.dumps({key: value for key, value in cell.items() if key != 'samples_us'}), flush=True)
            save()
        del graphs, programs, outputs, x, w, reference
        torch.cuda.empty_cache()
    result['hardware_after'] = snapshot()
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
