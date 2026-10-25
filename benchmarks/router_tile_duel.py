"""Cold-L2 router tile screen; fixed K order and exhaustive width correctness."""
import fcntl
import json
import os
import statistics
import subprocess

import torch
import cutlass
from b12x.gemm.bf16_gemv._skinny import SkinnyBf16Gemv, skinny_config
from b12x.integration.cuteafd._common import compile_program, Operand, Scalar

SHAPES = ((384, 5120), (128, 5120), (256, 6144), (256, 4096), (384, 7168))
ROWS = (1, 2, 4, 8, 16, 32, 48, 64)


def build(e, h, c, r, t):
    # Serialize only compilation, never timing or correctness launches.
    with open(os.environ['CUTEAFD_LOCK_DIR'] + '/build.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return compile_program(SkinnyBf16Gemv(e, h, out_dtype=cutlass.Float32,
                               cols=c, rows_per_tile=r, threads=t),
            name=f'router_{e}_{h}_{c}_{r}_{t}',
            operands=(Operand('x', torch.bfloat16, f'[rows,{h}]'),
                      Operand('w', torch.bfloat16, f'[{e},{h}]'),
                      Operand('out', torch.float32, f'[rows,{e}]', 'out')),
            scalars=(Scalar('rows'),), key=(e, h, c, r, t))


def telemetry():
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu', '--format=csv'], text=True), flush=True)


def main():
    torch.manual_seed(736)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device='cuda')
    results = []
    for e, h in SHAPES:
        cfg = skinny_config(e, h, cutlass.Float32)
        t = cfg['threads']
        base = (cfg['cols'], cfg['rows_per_tile'])
        tiles = list(dict.fromkeys([base, (1, 4), (2, 4), (2, 8), (4, 4), (4, 8)]))
        arms = {f'c{c}r{r}': build(e, h, c, r, t) for c, r in tiles}
        baseline = f'c{base[0]}r{base[1]}'
        good = set(arms)
        for mode in ('random', 'cancellation', 'dynamic'):
            x = torch.randn(64, h, device='cuda').bfloat16()
            w = torch.randn(e, h, device='cuda').bfloat16()
            if mode == 'cancellation':
                w[:, 1::2] = -w[:, ::2]
                x[:, 1::2] = x[:, ::2]
                x[:, ::257] *= 1.0078125
            if mode == 'dynamic':
                scale = torch.pow(2., (torch.arange(h, device='cuda') % 41 - 20).float())
                x = (x.float() * scale).bfloat16()
                w = (w.float() / scale).bfloat16()
                x[0].zero_()
            def run(p, xx):
                m = xx.shape[0]
                out = torch.full((m, e), float('nan'), device='cuda')
                p.launch(xx.contiguous(), w, out, scalars=(m,))
                return out
            ref = torch.cat([run(arms[baseline], x[i:i+1]) for i in range(64)])
            for name, p in arms.items():
                for m in range(1, 65):
                    actual = run(p, x[:m])
                    if not torch.equal(actual.view(torch.int32), ref[:m].view(torch.int32)):
                        good.discard(name)
                        print('FAIL', e, h, name, mode, m, flush=True)
                        break
            print('GATE', e, h, mode, sorted(good), flush=True)
        telemetry()
        for m in ROWS:
            out = torch.empty(m, e, device='cuda')
            active = [(name, p) for name, p in arms.items() if name in good]
            for name, p in active:
                p.launch(x[:m], w, out, scalars=(m,))
            torch.cuda.synchronize()
            samples = {name: [] for name, _ in active}
            # Rotate and reverse arm order; flush before every timed launch.
            for rep in range(30):
                order = active[rep % len(active):] + active[:rep % len(active)]
                if rep % 2:
                    order = order[::-1]
                for name, p in order:
                    flush.zero_()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    p.launch(x[:m], w, out, scalars=(m,))
                    b.record()
                    b.synchronize()
                    samples[name].append(a.elapsed_time(b) * 1000)
            stats = {}
            for name, ss in samples.items():
                ordered = sorted(ss)
                stats[name] = dict(median=statistics.median(ss), p10=ordered[3], p90=ordered[26],
                                   blocks=[statistics.median(ss[i:i+10]) for i in (0, 10, 20)])
            record = dict(e=e, h=h, rows=m, baseline=baseline, stats=stats, exact=sorted(good))
            results.append(record)
            print(json.dumps(record), flush=True)
        telemetry()
        with open(os.environ['RESULT_PATH'], 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
