"""GLM Flash KDA output rows sharded after joining normalized head activations."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from b12x._lib.intrinsics import ld_global_v4_u32, st_global_v4_u32
from ._common import GLM53_FLASH, GLMFGeometry, Operand, Scalar, compile_program
from ._fp8_weights import Fp8Projection
from .glmf import _check_rows, _ptr


class GlmfJoin:
    """Concatenate rank0/rank1 BF16 rows without floating-point conversion."""

    threads = 256

    def __init__(self, half_width: int):
        self.width = int(half_width)
        if self.width <= 0 or self.width % 8:
            raise ValueError("GLMF join requires a positive half width divisible by eight")

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        groups = Int64(rows) * Int64(self.width // 8)
        self.kernel(a, b, out, groups).launch(
            grid=((groups + Int64(self.threads - 1)) // Int64(self.threads), 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, groups: Int64):
        at = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if at < groups:
            row, col = at // Int64(self.width // 8), at % Int64(self.width // 8)
            address = Int64(out.toint()) + (row * Int64(2 * self.width) + col * Int64(8)) * Int64(2)
            st_global_v4_u32(address, *ld_global_v4_u32(Int64(a.toint()) + at * Int64(16)))
            st_global_v4_u32(address + Int64(self.width * 2),
                             *ld_global_v4_u32(Int64(b.toint()) + at * Int64(16)))


class GlmfOutputShard:
    def __init__(self, n: int, k: int, max_rows: int, prefill: bool, expanded: bool):
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
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.expanded is not None):
            if rows >= Int32(512):
                self.expanded(x, w_fp8, w_kscale, out, _ptr(cutlass.BFloat16, Int64(scratch.toint())), rows, stream)
            else:
                self.proj(x, w_fp8, w_kscale, out, rows, Int32(32), Int64(scratch.toint()), stream)
        else:
            self.proj(x, w_fp8, w_kscale, out, rows, Int32(32), Int64(scratch.toint()), stream)


def compile_glmf_join_aot(half_width: int):
    """Bit-exact row concat ``out[rows,2W] = [a[rows,W], b[rows,W]]``."""
    fn = GlmfJoin(half_width)
    w = fn.width
    return compile_program(fn, name="glmf_join",
        operands=(Operand("a", torch.bfloat16, f"[rows,{w}]"),
                  Operand("b", torch.bfloat16, f"[rows,{w}]"),
                  Operand("out", torch.bfloat16, f"[rows,{2 * w}]", "out")),
        scalars=(Scalar("rows"),), key=(w,), geometry={"half_width": w},
        doc=compile_glmf_join_aot.__doc__)


def compile_glmf_kda_output_shard_aot(g_half: GLMFGeometry = GLM53_FLASH, *, max_rows: int,
                                    fp8_only: str = "decode", prefill_expanded: bool = False):
    """Full K reduction for half the output columns, over FP8 row128 weights."""
    max_rows = _check_rows(max_rows)
    if fp8_only not in ("decode", "prefill"):
        raise ValueError("output shard fp8_only must be decode or prefill")
    if (g_half.hidden, g_half.kda_heads, g_half.kda_head_dim) != (4096, 32, 128):
        raise ValueError("output shard requires GLM53 Flash half-head geometry")
    prefill = fp8_only == "prefill"
    if prefill_expanded and not prefill:
        raise ValueError("expanded output shard requires prefill")
    n, k = g_half.hidden // 2, 2 * g_half.kda_width
    fn = GlmfOutputShard(n, k, max_rows, prefill, prefill_expanded)
    return compile_program(fn, name="glmf_kda_output_shard",
        operands=(Operand("x", torch.bfloat16, f"[rows,{k}]"),
                  Operand("w_fp8", torch.float8_e4m3fn, f"[{n},{k}]"),
                  Operand("w_kscale", torch.float32, f"[{k // 128},{n}]"),
                  Operand("out", torch.bfloat16, f"[rows,{n}]", "out"),
                  Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, fp8_only, fn.key()),
        geometry={"n": n, "k": k, "max_rows": max_rows, "fp8_only": fp8_only,
                  "prefill_expanded": prefill_expanded, "decode_gemv_rows": 32},
        scratch={"scratch": lambda rows: n * k * 2 if prefill_expanded and rows >= 512 else 0},
        doc=compile_glmf_kda_output_shard_aot.__doc__)
