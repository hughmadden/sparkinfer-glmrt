"""SM89 probe 3 calibration: what one launch can stream from a 50 MB weight.

L2-cold (``do_bench`` clears 256 MB between calls). A plain read-reduce of
the same bytes, cuBLASLt FP8 at 16 rows and cuBLAS BF16 GEMV over a weight of
equal byte size bound what a decode GEMV of this size can reach.
"""

import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

from _bench import time_rotating

COPIES = 6  # 6 x 50.3 MB rotate through the 72 MB L2


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _read_sum(src, dst, n_words, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * ITERS
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc ^= tl.load(src + offs, mask=offs < n_words, other=0, cache_modifier=".cg")
    tl.atomic_xor(dst, tl.reduce(acc, 0, _xor))


def main() -> int:
    dev = torch.device("cuda", 0)
    n, k = 12288, 4096
    w = torch.randint(0, 255, (n, k), device=dev, dtype=torch.uint8)
    words = w.view(torch.int32).view(-1)
    dst = torch.zeros(1, device=dev, dtype=torch.int32)
    nbytes = w.numel()
    copies = [w.view(torch.int32).view(-1)] + [
        torch.randint(0, 255, (n, k), device=dev, dtype=torch.uint8).view(torch.int32).view(-1)
        for _ in range(COPIES - 1)]
    for block, iters in ((2048, 4), (4096, 4), (2048, 16), (4096, 8)):
        grid = (triton.cdiv(words.numel(), block * iters),)
        fn = lambda: _read_sum[grid](words, dst, words.numel(), BLOCK=block, ITERS=iters, num_warps=8)
        us = do_bench(fn) * 1e3
        rot = time_rotating([
            (lambda c=c: _read_sum[grid](c, dst, c.numel(), BLOCK=block, ITERS=iters, num_warps=8))
            for c in copies])
        print(f"read_reduce 50.3 MB block={block} iters={iters} ctas={grid[0]}: "
              f"do_bench(dirty-L2 flush) {us:.1f} us {nbytes / (us * 1e-6) / 1e9:.0f} GB/s | "
              f"rotating {COPIES} copies {rot:.1f} us {nbytes / (rot * 1e-6) / 1e9:.0f} GB/s", flush=True)
    big = torch.empty(1 << 31, device=dev, dtype=torch.uint8).view(torch.int32)
    grid = (triton.cdiv(big.numel(), 4096 * 8),)
    us = do_bench(lambda: _read_sum[grid](big, dst, big.numel(), BLOCK=4096, ITERS=8, num_warps=8)) * 1e3
    print(f"read_reduce 2 GiB: {us:.1f} us {big.numel() * 4 / (us * 1e-6) / 1e9:.0f} GB/s", flush=True)
    del big
    w8s = [c.view(torch.uint8).view(n, k).view(torch.float8_e4m3fn) for c in copies]
    one = torch.ones((), device=dev)
    for rows in (16,):
        x8 = torch.randn(rows, k, device=dev).to(torch.float8_e4m3fn)
        rot = time_rotating([
            (lambda w8=w8: torch._scaled_mm(x8, w8.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            for w8 in w8s])
        print(f"cublaslt_fp8 rows={rows} N={n} K={k}: rotating {rot:.1f} us "
              f"{nbytes / (rot * 1e-6) / 1e9:.0f} GB/s", flush=True)
    del w8s
    wbs = [c.view(torch.bfloat16).view(n, k // 2) for c in copies]
    for rows in (1, 16):
        x = torch.randn(rows, k // 2, device=dev).to(torch.bfloat16)
        rot = time_rotating([(lambda wb=wb: x @ wb.T) for wb in wbs])
        print(f"cublas_bf16 rows={rows} N={n} K={k // 2} (same 50.3 MB): rotating {rot:.1f} us "
              f"{nbytes / (rot * 1e-6) / 1e9:.0f} GB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
