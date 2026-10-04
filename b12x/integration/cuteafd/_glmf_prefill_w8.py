"""W8A16 prefill with one transient BF16 expansion per projection launch."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.intrinsics import ld_global_nc_v2_u32, st_global_v4_u32
from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection, WIDE_PREFILL_CONFIG
from ._fp8_weights import _e4m3x8_scaled_bf16


class ExpandRow128Weight:
    """Expand row-major E4M3 with ``scale[K/128, N]`` into BF16 scratch."""

    threads = 256
    elements = 2048

    def __init__(self, n: int, k: int):
        self.n, self.k = int(n), int(k)
        if self.k % 128:
            raise ValueError("row128 expansion requires K % 128 == 0")

    @cute.jit
    def __call__(self, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer, stream: cuda.CUstream):
        self.kernel(weight, scales, out).launch(
            grid=((self.n * self.k + self.elements - 1) // self.elements, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer):
        at = Int64(cute.arch.block_idx()[0]) * Int64(self.elements) + Int64(cute.arch.thread_idx()[0]) * Int64(8)
        if at < Int64(self.n * self.k):
            row, kb = at // Int64(self.k), (at % Int64(self.k)) // Int64(128)
            scale = scales[kb * Int64(self.n) + row]
            lo, hi = ld_global_nc_v2_u32(Int64(weight.toint()) + at)
            words = _e4m3x8_scaled_bf16(lo, hi, scale)
            st_global_v4_u32(Int64(out.toint()) + at * Int64(2), *words)


class ExpandedW8Prefill:
    """Expand once in graph-owned scratch, then reuse across all row tiles."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16):
        self.n, self.k = int(n), int(k)
        self.expand = ExpandRow128Weight(n, k)
        self.projection = TmaBf16Projection(n, k, out_dtype=out_dtype, **WIDE_PREFILL_CONFIG)

    def key(self):
        return (self.n, self.k, str(self.projection.out_dtype), tuple(sorted(self.projection.config.items())))

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.expand(weight, scales, scratch, stream)
        self.projection(x, scratch, out, rows, stream)
