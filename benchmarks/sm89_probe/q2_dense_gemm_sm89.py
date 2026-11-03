"""SM89 probe 2: b12x ``DenseGemmKernel`` on the cp.async path at prefill shapes.

Plain BF16 (m16n8k16), plain per-tensor FP8 (e4m3 m16n8k32, FP32
accumulate, one FP32 alpha) and FP8 with 32x32 UE8M0 weight blocks (``fp8b32``:
each K32 MMA into a zero partial, times the block scale, added to the
accumulator) through the warp-specialized kernel: one cp.async DMA warp
feeding eight MMA warps through an mbarrier ring, persistent tile scheduler,
register-to-global epilogue on pre-Hopper parts. Timed back-to-back over two
weight copies (clean L2, launch gaps included).

Usage: q2_dense_gemm_sm89.py [--modes bf16,fp8,fp8b32] [--tile 128x128]
       [--tile-k-bf16 64] [--tile-k-fp8 128] [--m 4096]
"""

import argparse
import sys
import time

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from _bench import time_rotating
from b12x._lib.dense_gemm import DenseGemmKernel
from b12x._lib.utils import make_ptr

SHAPES = ((12288, 4096), (4096, 12288))  # (N, K)
PEAK_TFLOPS = {"bf16": 301.0, "fp8": 597.0, "fp8b32": 597.0}


class PlainDenseLaunch:
    """Pointer ABI around ``DenseGemmKernel`` for the plain (no scale factor) modes."""

    def __init__(self, mode: str, n: int, k: int, tile_mn, tile_k: int, sm_count: int,
                 mma_warps: int = 8):
        self.mode, self.n, self.k = mode, n, k
        self.tile_mn, self.tile_k, self.sm_count = tuple(tile_mn), tile_k, sm_count
        self.mma_warps = mma_warps

    def make_kernel(self) -> DenseGemmKernel:
        fp8 = self.mode in ("fp8", "fp8b32")
        return DenseGemmKernel(
            sf_vec_size=32,
            mma_tiler_mn=self.tile_mn,
            cluster_shape_mn=(1, 1),
            mma_k=32 if fp8 else 16,
            tile_k=self.tile_k,
            load_path="cpasync",
            alpha_is_one=False,
            target_occupancy=1,
            plain_fp8=fp8,
            plain_bf16=not fp8,
            block32_fp8=self.mode == "fp8b32",
            four_mma_warps=self.mma_warps == 4,
        )

    @cute.jit
    def __call__(self, a_ptr: cute.Pointer, b_ptr: cute.Pointer, c_ptr: cute.Pointer,
                 alpha_ptr: cute.Pointer, sf_ptr: cute.Pointer, m: Int32, stream: cuda.CUstream):
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout((m, self.k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout((self.n, self.k, 1), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, cute.make_ordered_layout((m, self.n, 1), order=(1, 0, 2)))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout((1,)))
        sf = cute.make_tensor(sf_ptr, cute.make_layout((1,)))
        # UE8M0 weight block scales [N/32, K/32], K contiguous (fp8b32 only).
        sfb = cute.make_tensor(
            sf_ptr, cute.make_ordered_layout((self.n // 32, self.k // 32, 1), order=(1, 0, 2)))
        kernel = self.make_kernel()
        self.traced_kernel = kernel  # stage depth etc. for the report
        kernel(
            a, a, alpha, alpha, b, sf, sfb, c, alpha, alpha, alpha, alpha,
            Int32(self.sm_count), stream,
        )


def run_case(mode, m, n, k, tile_mn, tile_k, sm_count, dev, mma_warps=8):
    torch.manual_seed(0)
    a32 = torch.randn(m, k, device=dev)
    b32 = torch.randn(n, k, device=dev) * 0.05
    scales = torch.zeros(1, device=dev, dtype=torch.uint8)
    if mode == "fp8":
        a = a32.to(torch.float8_e4m3fn)
        b = (b32 * 20).to(torch.float8_e4m3fn)
        alpha = torch.full((1,), 0.05, device=dev)
        ab_dtype = cutlass.Float8E4M3FN
        ref = (a.float() @ b.float().T) * 0.05
    elif mode == "fp8b32":
        a = a32.to(torch.float8_e4m3fn)
        b = (b32 * 20).to(torch.float8_e4m3fn)
        alpha = torch.full((1,), 0.05, device=dev)
        ab_dtype = cutlass.Float8E4M3FN
        scales = torch.randint(121, 134, (n // 32, k // 32), device=dev, dtype=torch.uint8)
        w_scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, 0).repeat_interleave(32, 1)
        ref = (a.float() @ (b.float() * w_scale).T) * 0.05
    else:
        a = a32.to(torch.bfloat16)
        b = b32.to(torch.bfloat16)
        alpha = torch.ones(1, device=dev)
        ab_dtype = cutlass.BFloat16
        ref = a.float() @ b.float().T
    c = torch.zeros(m, n, device=dev, dtype=torch.bfloat16)
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(ab_dtype, a.data_ptr(), gmem, assumed_align=16),
        make_ptr(ab_dtype, b.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, c.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float32, alpha.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, (scales if mode == "fp8b32" else alpha).data_ptr(), gmem,
                 assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    launch = PlainDenseLaunch(mode, n, k, tile_mn, tile_k, sm_count, mma_warps)
    t0 = time.time()
    compiled = cute.compile(launch, *ptrs, Int32(m), stream)
    compile_s = time.time() - t0
    compiled(*ptrs, Int32(m), stream)
    torch.cuda.synchronize()
    diff = (c.float() - ref).abs()
    rel = (diff.max() / ref.abs().max()).item()
    nz = (c != 0).float().mean().item()
    ok = rel < 1e-2 and nz > 0.99
    b2 = b.clone()
    b2_ptr = make_ptr(ab_dtype, b2.data_ptr(), gmem, assumed_align=16)
    ms = time_rotating([
        (lambda bp=bp: compiled(ptrs[0], bp, *ptrs[2:], Int32(m), stream)) for bp in (ptrs[1], b2_ptr)
    ], rounds=10) * 1e-3
    tf = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    print(f"dense_gemm cpasync {mode:6s} M={m} N={n} K={k} tile={tile_mn[0]}x{tile_mn[1]}x{tile_k} "
          f"mma_warps={mma_warps} stages={kern_stages(launch)}: {ms:.3f} ms {tf:.1f} TF/s "
          f"({100 * tf / PEAK_TFLOPS[mode]:.0f}% of {PEAK_TFLOPS[mode]:.0f}) "
          f"maxrel={rel:.2e} nonzero={nz:.3f} ok={ok} compile={compile_s:.1f}s", flush=True)
    return ok


def kern_stages(launch):
    kernel = getattr(launch, "traced_kernel", None)
    return getattr(kernel, "ab_stage", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="bf16,fp8,fp8b32")
    ap.add_argument("--tile", default="128x128")
    ap.add_argument("--tile-k-bf16", type=int, default=64)
    ap.add_argument("--tile-k-fp8", type=int, default=128)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--shapes", default="12288x4096,4096x12288")
    ap.add_argument("--mma-warps", type=int, default=8, choices=(4, 8))
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"device={torch.cuda.get_device_name(dev)} sms={sm_count} cutlass-dsl={cutlass.__version__}")
    tile_mn = tuple(int(x) for x in args.tile.split("x"))
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    ok = True
    for mode in args.modes.split(","):
        tile_k = args.tile_k_bf16 if mode == "bf16" else args.tile_k_fp8
        for n, k in shapes:
            ok &= run_case(mode, args.m, n, k, tile_mn, tile_k, sm_count, dev, args.mma_warps)
    print("Q2", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
