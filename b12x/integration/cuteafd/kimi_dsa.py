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
from .glm_attention import _IndexProducer, index_producer_scratch_bytes


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


def _post(g):
    return GlmIndexPost(heads=g.index_heads, eps=g.index_norm_eps,
        weight_scale=g.index_heads ** -0.5 * g.index_head_dim ** -0.5,
        page_rows=g.page_rows, rope=False, scale_format="ue8m0",
        query_amax_floor=1e-10, key_amax_floor=1e-4)


def compile_kimi_index_post_aot(g=KimiDsaGeometry(), *, max_rows=64):
    """Projected BF16 iq/kw -> E4M3 queries, folded weights, paged keys/scales.

    kw is the contiguous concatenation [wk(x), weights_proj(x)]. Slots are
    physical cache rows; -1 suppresses a key write. NoPE ignores cos_sin.
    """
    if not 1 <= int(max_rows) <= 4096:
        raise ValueError("indexer row capacity must be in 1..4096")
    launch = _post(g)
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


class _KimiIndexProducer(_IndexProducer):
    def __init__(self, g):
        super().__init__(g)
        self.post = _post(g)


def compile_kimi_index_producer_aot(g=KimiDsaGeometry(), *, max_rows=64):
    """Native BF16 projections and NoPE QDQ over the shared index producer.

    w_ik concatenates native wk and weights_proj rows. q_resid is the MLA
    producer's normalized q-LoRA output; no extra query normalization occurs.
    """
    if not 1 <= int(max_rows) <= 4096:
        raise ValueError("indexer row capacity must be in 1..4096")
    launch = _KimiIndexProducer(g)
    operands = (
        Operand("x", torch.bfloat16, "[rows,7168]"),
        Operand("q_resid", torch.bfloat16, "[rows,1536]"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("index_slots", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("w_iq", torch.bfloat16, "[4096,1536]"),
        Operand("w_ik", torch.bfloat16, "[160,7168]"),
        Operand("k_norm_w", torch.bfloat16, "[128]"),
        Operand("k_norm_b", torch.bfloat16, "[128]"),
        Operand("index_cache", torch.uint8, "[pages,8448]", "inout"),
        Operand("q_fp8", torch.float8_e4m3fn, "[rows,32,128]", "out"),
        Operand("head_weights", torch.float32, "[rows,32]", "out"),
        Operand("scratch", torch.uint8, "[index_producer_scratch_bytes]", "scratch"),
    )
    return compile_program(launch, name="kimi_index_producer", operands=operands,
        scalars=(Scalar("rows"),), key=(g, int(max_rows), launch.key(), "ue8m0"),
        geometry={**vars(g), "max_rows": int(max_rows), "rope": False,
                  "scale_format": "ue8m0", "scale_storage": "float32",
                  "query_amax_floor": 1e-10, "key_amax_floor": 1e-4},
        scratch={"scratch": lambda rows: index_producer_scratch_bytes(g, rows)},
        doc=compile_kimi_index_producer_aot.__doc__)


def compile_kimi_index_topk_aot(g=KimiDsaGeometry(), *, max_rows, max_pages, mode="decode"):
    """Shared selector; logical ascending order mapped to physical slots, -1 padded."""
    return compile_glm_index_topk_aot(g, max_rows=max_rows, max_pages=max_pages, mode=mode)


def compile_kimi_context_index_topk_aot(g=KimiDsaGeometry(), *, max_rows, max_pages, mode="decode"):
    """Score the full replicated index, returning global logical candidates.

    Use page_stride=1 and page_offset=0, never rank-local top-k. Pass these
    candidates as the first run and an empty (-inf,-1) second run to the
    shared dsa_candidate_merge. Its parity filter maps the globally selected
    logical IDs through each rank's independent compact MLA page table.
    """
    from .context_split import compile_scored_index_topk_aot

    return compile_scored_index_topk_aot(g, max_rows=max_rows, max_pages=max_pages, mode=mode)


def compile_kimi_sparse_mla_partial_aot(g=None, *, max_rows=64, head_begin=0):
    """Parity-local selected-token MLA partial with base2 LSE for peer combine.

    Global index selection and shared candidate parity compaction precede this
    stage. Indices are local physical token slots; lengths may be zero. Query
    rows retain the h96 stride, while each output contains one h48 range.
    """
    from .kimi_mla import KimiMlaGeometry, _ContextPartial

    g = g or KimiMlaGeometry(name="kimi_k31_mla", norm_eps=1e-5)
    if type(g) is not KimiMlaGeometry or g.heads != 96 or head_begin not in (0, 48):
        raise ValueError("Kimi sparse Context partial requires an aligned h48 range of h96")
    if not 1 <= int(max_rows) <= 4096:
        raise ValueError("sparse MLA row capacity must be in 1..4096")
    launch = _ContextPartial(g, int(max_rows), 2048, head_begin, 48, 16, page_size=1)
    operands = (
        Operand("query", torch.bfloat16, "[rows,96,576]"),
        Operand("kv_cache", torch.uint8, f"[pool_tokens,{g.record_bytes}]"),
        Operand("page_table", torch.int32, "[rows,2048]", align=4),
        Operand("cache_seqlens", torch.int32, "[rows]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[rows+1]", align=4),
        Operand("attn", torch.bfloat16, "[rows,48,512]", "out"),
        Operand("lse", torch.float32, "[rows,48]", "out", align=4),
        Operand("scratch", torch.uint8, "[sparse_mla_scratch_bytes]", "scratch"),
    )
    return compile_program(launch, name="kimi_sparse_mla_partial", operands=operands,
        scalars=(Scalar("rows"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        key=(g, int(max_rows), head_begin, "selected-token-gather", "base2"),
        geometry={**vars(g), "max_rows": int(max_rows), "topk": 2048,
                  "head_begin": head_begin, "head_count": 48, "num_splits": 16,
                  "page_size": 1, "lse_base": 2},
        scratch={"scratch": launch.scratch_bytes}, doc=compile_kimi_sparse_mla_partial_aot.__doc__)


def compile_kimi_sparse_mla_aot(g=None, *, max_rows=64):
    """Selected-token MLA using the existing Kimi BF16 attention pipeline.

    Each query is one request. Its table contains physical token slots, so
    page_size=1 changes only gather addressing; no global gather/repack occurs.
    lengths count the valid leading indices (1..2048). They are already causal.
    The ordinary Kimi UV/sigmoid gate/output stage consumes out unchanged.
    """
    from .kimi_mla import KimiMlaGeometry, _Attention

    g = g or KimiMlaGeometry(name="kimi_k31_mla", norm_eps=1e-5)
    if not 1 <= int(max_rows) <= 4096:
        raise ValueError("sparse MLA row capacity must be in 1..4096")
    launch = _Attention(g, int(max_rows), 2048, "decode", page_size=1)
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand("kv_cache", torch.uint8, f"[pool_tokens,{g.record_bytes}]"),
        Operand("page_table", torch.int32, "[rows,2048]", align=4),
        Operand("cache_seqlens", torch.int32, "[rows]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[rows+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out", align=4),
        Operand("scratch", torch.uint8, "[sparse_mla_scratch_bytes]", "scratch"),
    )
    return compile_program(launch, name="kimi_sparse_mla", operands=operands,
        scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        key=(g, int(max_rows), "selected-token-gather"),
        geometry={**vars(g), **launch.layout, "max_rows": int(max_rows), "topk": 2048,
                  "page_size": 1, "lse_base": "e"},
        scratch={"scratch": lambda rows: launch.layout["scratch_bytes"]},
        doc=compile_kimi_sparse_mla_aot.__doc__)
