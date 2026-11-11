"""Kimi K3.1 NoPE DSA seams over the shared GLM producer and selector.

The native indexer operands remain BF16. Index queries and cache keys use
E4M3 with power-of-two scales, not GLM's unrounded amax/448 scales. Cache
pages retain the shared 8448-byte ABI (8192 key bytes, 64 FP32 scale slots).
This is the non-MX ABI in vLLM 4414f472fb1dd5722b6468c2057943a8c45f83dc:
models/kimi_k3/nvidia/dsa.py:180-182 and ops/dsa_indexer.py:24-26,68,72-79,199-205.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._common import Operand, Scalar, compile_program
from ._glm_kernels import GlmIndexPost
from .glm_indexer import compile_glm_index_topk_aot


@dataclass(frozen=True)
class KimiDsaGeometry:
    hidden: int = 7168
    q_lora_rank: int = 1536
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_norm_eps: float = 1e-5
    page_rows: int = 64

    def __post_init__(self):
        if (self.hidden, self.q_lora_rank, self.index_heads, self.index_head_dim,
            self.index_topk, self.page_rows) != (7168, 1536, 32, 128, 2048, 64):
            raise ValueError("unsupported Kimi K3.1 indexer geometry")
        if not math.isfinite(self.index_norm_eps) or self.index_norm_eps <= 0:
            raise ValueError("indexer epsilon must be finite and positive")

    @property
    def index_page_bytes(self):
        return self.page_rows * (self.index_head_dim + 4)


def compile_kimi_index_post_aot(g=KimiDsaGeometry(), *, max_rows=64):
    """Projected BF16 iq/kw -> E4M3 queries, folded weights, paged keys/scales.

    kw is the contiguous concatenation [wk(x), weights_proj(x)]. Slots are
    physical cache rows; -1 suppresses a key write. NoPE ignores cos_sin.
    """
    if not 1 <= int(max_rows) <= 4096:
        raise ValueError("indexer row capacity must be in 1..4096")
    launch = GlmIndexPost(heads=g.index_heads, eps=g.index_norm_eps,
        weight_scale=g.index_heads ** -0.5 * g.index_head_dim ** -0.5,
        page_rows=g.page_rows, rope=False, scale_format="ue8m0",
        query_amax_floor=1e-10, key_amax_floor=1e-4)
    operands = (
        Operand("iq", torch.bfloat16, "[rows,32,128]"),
        Operand("kw", torch.bfloat16, "[rows,160]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("k_weight", torch.bfloat16, "[128]"),
        Operand("k_bias", torch.bfloat16, "[128]"),
        Operand("q_fp8", torch.float8_e4m3fn, "[rows,32,128]", "out"),
        Operand("head_weights", torch.float32, "[rows,32]", "out"),
        Operand("cache", torch.uint8, "[pages,8448]", "inout"),
    )
    return compile_program(launch, name="kimi_index_post", operands=operands,
        scalars=(Scalar("rows"),), key=(g, int(max_rows), "ue8m0", 1e-10, 1e-4),
        geometry={**vars(g), "max_rows": int(max_rows), "rope": False,
                  "scale_format": "ue8m0", "scale_storage": "float32",
                  "query_amax_floor": 1e-10, "key_amax_floor": 1e-4},
        doc=compile_kimi_index_post_aot.__doc__)


def compile_kimi_index_topk_aot(g=KimiDsaGeometry(), *, max_rows, max_pages, mode="decode"):
    """Shared 32-head weighted-ReLU selector; physical slots, ascending, -1 padded."""
    return compile_glm_index_topk_aot(g, max_rows=max_rows, max_pages=max_pages, mode=mode)
