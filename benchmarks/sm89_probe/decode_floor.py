"""SM89 decode calibration: what one graph-replayed launch can stream.

Per-launch floor (an empty kernel) and a plain read-reduce of each small
decode weight size, timed like the GEMV (CUDA graph of back-to-back launches
over copies exceeding L2). Bounds what a decode projection of that size can
reach before any GEMV-specific cost.
"""

import math
import sys

import torch
import triton
import triton.language as tl

from _bench import time_graph


@triton.jit
def _empty(dst):
    pass


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _read(src, dst, n_words, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * ITERS
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc ^= tl.load(src + offs, mask=offs < n_words, other=0, cache_modifier=".cg")
    tl.atomic_xor(dst, tl.reduce(acc, 0, _xor))


def main() -> int:
    dev = torch.device("cuda", 0)
    dst = torch.zeros(1, device=dev, dtype=torch.int32)
    us = time_graph([lambda h: _empty[(1,)](dst)], rounds=200)
    print(f"empty kernel per launch (graph): {us:.2f} us", flush=True)
    for name, n, k in (("kv", 512, 5120), ("ffn_tp2_up", 1152, 5120), ("ffn_tp2_down", 5120, 1152),
                       ("q_a", 1280, 5120), ("index_q", 4096, 1280), ("ffn_up", 2304, 5120),
                       ("o_b_tp2", 2560, 8192), ("q_b_tp2", 16384, 1280)):
        nbytes = n * k
        copies = max(2, math.ceil(160e6 / nbytes))
        bufs = [torch.randint(0, 255, (nbytes,), device=dev, dtype=torch.uint8).view(torch.int32)
                for _ in range(copies)]
        results = []
        for block, iters, warps in ((1024, 2, 4), (2048, 2, 4), (1024, 4, 4), (4096, 1, 4),
                                    (2048, 1, 8), (4096, 2, 8), (8192, 1, 8), (512, 2, 4), (512, 4, 2)):
            grid = (triton.cdiv(bufs[0].numel(), block * iters),)
            calls = [(lambda h, b=b, grid=grid, block=block, iters=iters, warps=warps:
                      _read[grid](b, dst, b.numel(), BLOCK=block, ITERS=iters, num_warps=warps))
                     for b in bufs]
            results.append((time_graph(calls, rounds=max(2, 64 // copies)), block, iters, grid[0]))
        t, block, iters, ctas = min(results)
        print(f"read {name:12s} {nbytes / 1e6:5.1f} MB: {t:6.2f} us {nbytes / (t * 1e-6) / 1e9:5.0f} GB/s "
              f"({100 * nbytes / (t * 1e-6) / 1e9 / 978:.0f}%) block={block} iters={iters} ctas={ctas}",
              flush=True)
        del bufs
    return 0


if __name__ == "__main__":
    sys.exit(main())
