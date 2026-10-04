"""MLA value heads and full-K output projections for owned token rows."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import GLMFGeometry, Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm
from ._glmf_fp8 import quant_scratch_bytes
from .glmf import FP8_ROWS, _batched_warps, _check_rows, _w8


def _check_geometry(g: GLMFGeometry):
    if (g.hidden, g.heads, g.v_head_dim, g.kv_lora_rank) != (4096, 32, 256, 512):
        raise ValueError("MLA output rows requires GLM53 Flash half-head geometry")


def compile_glmf_mla_values_aot(g_half: GLMFGeometry, *, max_rows: int):
    """The original per-head BF16 W_UV expansion, without the o_proj.

    Output is BF16 ``[rows,32*256]``; input is BF16 ``[rows,32,512]``.
    The global row count and planned capacity retain the full consumer's
    per-head math. No scratch or output-projection weights are consumed.
    """
    _check_geometry(g_half)
    max_rows = _check_rows(max_rows)
    heads, v, latent = g_half.heads, g_half.v_head_dim, g_half.kv_lora_rank
    uv = BatchedBf16Gemm(n=v, k=latent, batch=heads, a_row=heads * latent,
        a_batch=latent, o_row=heads * v, o_batch=v, compute_warps=_batched_warps(max_rows))
    return compile_program(uv, name="glmf_mla_values",
        operands=(Operand("attn", torch.bfloat16, f"[rows,{heads},{latent}]"),
                  Operand("w_uv", torch.bfloat16, f"[{heads},{v},{latent}]"),
                  Operand("out", torch.bfloat16, f"[rows,{heads * v}]", "out")),
        scalars=(Scalar("rows"),), key=(max_rows, uv.key()),
        geometry={"heads": heads, "v_head_dim": v, "kv_lora_rank": latent, "max_rows": max_rows},
        doc=compile_glmf_mla_values_aot.__doc__)


class GlmfMlaOutputRows:
    def __init__(self, n: int, k: int, max_rows: int, prefill: bool):
        self.prefill = bool(prefill)
        # Match the original full-head projection's tuning and native block scales.
        self.proj = _w8(n, k, max_rows if prefill else None)

    def key(self):
        return self.proj.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, w_scale: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, owned_rows: Int32,
                 total_rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        if owned_rows > Int32(0):
            route = fp8_rows
            if cutlass.const_expr(not self.prefill):
                route = Int32(0)
                if total_rows <= Int32(FP8_ROWS):
                    route = Int32(FP8_ROWS)
            self.proj(x, w_fp8, w_scale, out, owned_rows, route, Int64(scratch.toint()), stream)


def compile_glmf_mla_output_rows_aot(g_half: GLMFGeometry, *, max_rows: int,
                                   fp8_only: str = "decode"):
    """Full-N/full-K MLA projection for owned token rows over native block FP8.

    Decode uses the original full-head GEMV only when global ``total_rows<=16``;
    wider global batches use W8A16 TMA even for skinny owned rows. Prefill
    retains the original ``fp8_rows`` A8/W8A16 switch. Zero owned rows skips
    projection. Query scratch with global total rows, not owned rows.
    """
    _check_geometry(g_half)
    max_rows = _check_rows(max_rows)
    if fp8_only not in ("decode", "prefill"):
        raise ValueError("MLA output rows fp8_only must be decode or prefill")
    prefill = fp8_only == "prefill"
    n, k = g_half.hidden, 2 * g_half.heads * g_half.v_head_dim
    fn = GlmfMlaOutputRows(n, k, max_rows, prefill)
    return compile_program(fn, name="glmf_mla_output_rows",
        operands=(Operand("x", torch.bfloat16, f"[owned_rows,{k}]"),
                  Operand("w_fp8", torch.float8_e4m3fn, f"[{n},{k}]"),
                  Operand("w_scale", torch.float32, f"[{n // 128},{k // 128}]", align=16 if prefill else 4),
                  Operand("out", torch.bfloat16, f"[owned_rows,{n}]", "out"),
                  Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch")),
        scalars=(Scalar("owned_rows"), Scalar("total_rows"), Scalar("fp8_rows")),
        key=(max_rows, fp8_only, fn.key()),
        geometry={"n": n, "k": k, "max_rows": max_rows, "fp8_only": fp8_only,
                  "decode_gemv_rows": FP8_ROWS, "route_rows": "total_rows", "scratch_rows": "total_rows"},
        scratch={"scratch": lambda total_rows: quant_scratch_bytes(k, (int(total_rows) + 1) // 2)
                 if prefill else 0}, doc=compile_glmf_mla_output_rows_aot.__doc__)


def compile_glmf_join_mla_heads_aot(g_half: GLMFGeometry):
    """Bit-exact concatenation of rank 0 and rank 1's BF16 MLA value heads."""
    from ._glmf_output_shard import compile_glmf_join_aot
    _check_geometry(g_half)
    return compile_glmf_join_aot(g_half.heads * g_half.v_head_dim)
