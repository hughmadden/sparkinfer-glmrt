"""Full-K KDA output projections for owned token rows, with global row routing."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from b12x._lib.intrinsics import ld_global_v4_u32, st_global_v4_u32
from ._common import GLMFGeometry, Operand, Scalar, compile_program
from ._fp8_weights import Fp8Projection
from .glmf import _check_rows, _ptr


class GlmfOutputRows:
    def __init__(self, n: int, k: int, max_rows: int, prefill: bool, expanded: bool):
        self.prefill = bool(prefill)
        self.proj = Fp8Projection(n, k, prefill_rows=max_rows if prefill else None,
            row_scales=True, kmajor=True, warps=8, groups=2, wide_rows=32,
            wide_warps=8, wide_groups=2, prefill_mask=0)
        self.expanded = None
        if expanded:
            from ._glmf_prefill_w8 import ExpandedW8Prefill
            self.expanded = ExpandedW8Prefill(n, k)

    def key(self):
        return (self.proj.key(), None if self.expanded is None else self.expanded.key())

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, w_kscale: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, owned_rows: Int32,
                 total_rows: Int32, stream: cuda.CUstream):
        if owned_rows > Int32(0):
            if cutlass.const_expr(self.expanded is not None):
                if total_rows >= Int32(512):
                    self.expanded(x, w_fp8, w_kscale, out,
                        _ptr(cutlass.BFloat16, Int64(scratch.toint())), owned_rows, stream)
                else:
                    self.proj(x, w_fp8, w_kscale, out, owned_rows, Int32(0), Int64(scratch.toint()), stream)
            else:
                fp8_rows = Int32(0)
                if cutlass.const_expr(not self.prefill):
                    if total_rows <= Int32(32):
                        fp8_rows = Int32(32)
                self.proj(x, w_fp8, w_kscale, out, owned_rows, fp8_rows, Int64(scratch.toint()), stream)


class GlmfJoinRows:
    threads = 256

    def __init__(self, width: int):
        self.width = int(width)
        if self.width <= 0 or self.width % 8:
            raise ValueError("GLMF row join requires a positive width divisible by eight")

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer,
                 rows_a: Int32, rows_b: Int32, stream: cuda.CUstream):
        groups = (Int64(rows_a) + Int64(rows_b)) * Int64(self.width // 8)
        split = Int64(rows_a) * Int64(self.width // 8)
        if groups > Int64(0):
            self.kernel(a, b, out, groups, split).launch(
                grid=((groups + Int64(self.threads - 1)) // Int64(self.threads), 1, 1),
                block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, groups: Int64, split: Int64):
        at = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if at < groups:
            source = Int64(a.toint()) + at * Int64(16)
            if at >= split:
                source = Int64(b.toint()) + (at - split) * Int64(16)
            st_global_v4_u32(Int64(out.toint()) + at * Int64(16), *ld_global_v4_u32(source))


def compile_glmf_join_rows_aot(width: int):
    """Bit-exact BF16 row concatenation ``out = [a[rows_a,W]; b[rows_b,W]]``.

    Either row count may be zero. Both zero skips the kernel launch.
    """
    fn = GlmfJoinRows(width)
    w = fn.width
    return compile_program(fn, name="glmf_join_rows",
        operands=(Operand("a", torch.bfloat16, f"[rows_a,{w}]"),
                  Operand("b", torch.bfloat16, f"[rows_b,{w}]"),
                  Operand("out", torch.bfloat16, f"[rows_a+rows_b,{w}]", "out")),
        scalars=(Scalar("rows_a"), Scalar("rows_b")), key=(w,), geometry={"width": w},
        doc=compile_glmf_join_rows_aot.__doc__)


def compile_glmf_kda_output_rows_aot(g_half: GLMFGeometry, *, max_rows: int,
                                   fp8_only: str = "decode", prefill_expanded: bool = False):
    """Full-N/full-K BF16 output for owned token rows over FP8 row128 weights.

    Decode uses the 8-warp/2-group GEMV when global ``total_rows <=32``,
    otherwise W8A16 TMA, even if ``owned_rows <=32``. Prefill always uses
    W8A16; optional expansion starts at global ``total_rows >=512``.
    Zero owned rows skips projection. Query scratch with global total rows.
    """
    max_rows = _check_rows(max_rows)
    if fp8_only not in ("decode", "prefill"):
        raise ValueError("output rows fp8_only must be decode or prefill")
    if (g_half.hidden, g_half.kda_heads, g_half.kda_head_dim) != (4096, 32, 128):
        raise ValueError("output rows requires GLM53 Flash half-head geometry")
    prefill = fp8_only == "prefill"
    if prefill_expanded and not prefill:
        raise ValueError("expanded output rows requires prefill")
    n, k = g_half.hidden, 2 * g_half.kda_width
    fn = GlmfOutputRows(n, k, max_rows, prefill, prefill_expanded)
    return compile_program(fn, name="glmf_kda_output_rows",
        operands=(Operand("x", torch.bfloat16, f"[owned_rows,{k}]"),
                  Operand("w_fp8", torch.float8_e4m3fn, f"[{n},{k}]"),
                  Operand("w_kscale", torch.float32, f"[{k // 128},{n}]"),
                  Operand("out", torch.bfloat16, f"[owned_rows,{n}]", "out"),
                  Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch")),
        scalars=(Scalar("owned_rows"), Scalar("total_rows")), key=(max_rows, fp8_only, fn.key()),
        geometry={"n": n, "k": k, "max_rows": max_rows, "fp8_only": fp8_only,
                  "prefill_expanded": prefill_expanded, "decode_gemv_rows": 32,
                  "route_rows": "total_rows", "scratch_rows": "total_rows"},
        scratch={"scratch": lambda total_rows: n * k * 2 if prefill_expanded and total_rows >= 512 else 0},
        doc=compile_glmf_kda_output_rows_aot.__doc__)
