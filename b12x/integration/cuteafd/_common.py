"""Shared contract for the cuteafd DeepSeek V4 native AOT programs.

Every compile function in this package returns an :class:`AotProgram`: one
CuTe DSL program whose host ABI is ``(void *ptr..., <scalars>..., stream)``.
The ABI is recorded on the program (:attr:`AotProgram.abi`) so exporters can
validate the generated header against it, and so tests can launch the program
from torch tensors through the same pointer order the native engine uses.

Conventions shared by all programs:

* Pointers are raw device addresses; the caller owns every buffer, including
  scratch. Shapes in :class:`Operand` use the symbolic live row count ``rows``
  (the ``rows`` scalar) and the static geometry the program was compiled for.
  All buffers are dense row-major with the stated shape unless the operand
  says otherwise.
* Row and page counts are runtime scalars; no live request quantity enters a
  compile key. Pool-scaled offsets are computed in 64-bit arithmetic.
* Programs perform no allocation, host synchronization, or compilation at
  launch. Compile (and export) every program before CUDA graph capture.
* Exporting requires the retained IR, so compile under
  :func:`exportable_compilation` (it disables the b12x object caches for the
  duration of the call) before calling ``export_to_c``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

__all__ = [
    "AotProgram",
    "DSV4Geometry",
    "FLASH",
    "GLM53",
    "GLM53_FLASH",
    "GLMFGeometry",
    "GLMGeometry",
    "MIMO_V2_FLASH",
    "MIMO_V26_PRO",
    "MiMoGeometry",
    "Operand",
    "PRO",
    "Scalar",
    "exportable_compilation",
    "validate_exported_header",
]


@dataclass(frozen=True)
class DSV4Geometry:
    """Static DeepSeek V4 model geometry baked into the exported programs.

    Only model constants live here; capacities are separate compile
    parameters and live counts are launch scalars.
    """

    name: str = "flash"
    hidden: int = 4096
    heads: int = 64
    q_lora_rank: int = 1024
    head_dim: int = 512
    nope_dim: int = 448
    rope_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    window: int = 128
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    norm_eps: float = 1.0e-6
    moe_inter: int = 2048
    routed_experts: int = 256
    swiglu_limit: float = 10.0

    def __post_init__(self) -> None:
        if (self.head_dim, self.nope_dim, self.rope_dim) != (512, 448, 64):
            raise ValueError("DSV4 requires head/nope/rope dims 512/448/64")
        if self.hc_mult != 4:
            raise ValueError("DSV4 mHC requires hc_mult=4")
        if self.hidden not in (4096, 7168):
            raise ValueError("DSV4 hidden must be 4096 (Flash) or 7168 (Pro)")
        if (self.index_heads, self.index_head_dim) != (64, 128):
            raise ValueError("DSV4 indexer requires 64 heads of 128 dims")
        if self.heads % self.o_groups:
            raise ValueError("DSV4 heads must divide into o_groups")

    @property
    def mhc_split_k(self) -> int:
        """Partial rows per token of the mHC fn projection (2 * hidden / 128)."""
        return 2 * self.hidden // 128

    @property
    def o_group_width(self) -> int:
        return self.heads * self.head_dim // self.o_groups


FLASH = DSV4Geometry()
PRO = DSV4Geometry(
    name="pro", hidden=7168, heads=128, q_lora_rank=1536, o_groups=16,
    o_lora_rank=1024, index_topk=1024, moe_inter=3072, routed_experts=384,
)


@dataclass(frozen=True)
class GLMGeometry:
    """Static GLM 5.x (``glm_moe_dsa``) geometry baked into the ``glm_*`` programs.

    MLA with a 512-wide latent (q_lora 2048, nope 192, rope 64, v 256), the
    DSA indexer (32 heads x 128, top-2048, LayerNorm k), SwiGLU dense layers
    and a top-8 sigmoid router over 256 experts plus one shared expert.
    """

    name: str = "glm53"
    hidden: int = 6144
    heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_dim: int = 192
    qk_rope_dim: int = 64
    v_head_dim: int = 256
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    dense_inter: int = 12288
    moe_inter: int = 2048
    routed_experts: int = 256
    norm_eps: float = 1.0e-5
    index_norm_eps: float = 1.0e-6
    page_rows: int = 64

    def __post_init__(self) -> None:
        if (self.kv_lora_rank, self.qk_rope_dim) != (512, 64):
            raise ValueError("GLM programs require the 512 latent + 64 RoPE record")
        if (self.index_head_dim, self.page_rows) != (128, 64):
            raise ValueError("GLM index cache requires 128-dim keys on 64-row pages")
        if self.qk_nope_dim % 64 or self.v_head_dim % 64 or self.hidden % 1024:
            raise ValueError("GLM projection widths must be multiples of 64 (hidden of 1024)")

    @property
    def qkv_a_width(self) -> int:
        """Joint ``[q_a_proj; kv_a_proj_with_mqa]`` output width."""
        return self.q_lora_rank + self.kv_lora_rank + self.qk_rope_dim

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def latent_dim(self) -> int:
        """Absorbed query / cache record width (latent + RoPE)."""
        return self.kv_lora_rank + self.qk_rope_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @property
    def record_bytes(self) -> int:
        """FP8 latent record: 512 E4M3, 4 FP32 group scales, 64 BF16 RoPE."""
        return self.kv_lora_rank + 4 * (self.kv_lora_rank // 128) + 2 * self.qk_rope_dim

    @property
    def kv_page_bytes(self) -> int:
        return self.page_rows * self.record_bytes

    @property
    def index_page_bytes(self) -> int:
        return self.page_rows * (self.index_head_dim + 4)


GLM53 = GLMGeometry()


@dataclass(frozen=True)
class MiMoGeometry:
    """Static MiMo V2 (``mimo_v2_flash``) geometry baked into the ``mimo_*`` programs.

    Hybrid GQA: ``full`` layers (64 query / 4 KV heads, no sink, RoPE theta
    5e6) and 128-token sliding-window ``swa`` layers (64 / 8 heads, learned
    per-head sink, theta 1e4); QK head 192 with RoPE on its first 64 dims
    (NeoX halves), V head 128 scaled by ``v_scale``; SwiGLU dense layer 0 and
    a top-8 sigmoid router over 256 experts, no shared expert.

    KV records are BF16, one per token: every KV head's 192-wide key, then
    every head's 128-wide value (``record_elems``). Full layers keep records
    in a paged cache (``page_rows`` per page); SWA layers keep a per-sequence
    ring of ``ring_rows`` records (``ring_rows >= window + max verify rows -
    1`` so a rejected speculative suffix never overwrites a key the next step
    still needs).
    """

    name: str = "mimo_v2_flash"
    hidden: int = 4096
    heads: int = 64
    full_kv_heads: int = 4
    swa_kv_heads: int = 8
    qk_head_dim: int = 192
    v_head_dim: int = 128
    rope_dim: int = 64
    window: int = 128
    ring_rows: int = 256
    page_rows: int = 64
    v_scale: float = 0.707
    full_rope_theta: float = 5.0e6
    swa_rope_theta: float = 1.0e4
    dense_inter: int = 16384
    moe_inter: int = 2048
    routed_experts: int = 256
    top_k: int = 8
    norm_eps: float = 1.0e-5
    # The router weight is stored FP32 (Flash: split into BF16 hi + lo) or
    # BF16 (V2.6 Pro: one BF16 product, FP32 accumulation).
    router_fp32: bool = True
    # Rows of each KV head's key in the qkv projection output: 192, or 256
    # when every key is padded to whole 128-row blocks (V2.6 Pro: its fused
    # qkv_proj grid restarts per head, so the padded layout keeps the
    # checkpoint's 128x128 FP8 blocks for the FP8 decode projection).
    qkv_k_stride: int = 192
    # Export decode producers that read the qkv weight as the checkpoint's FP8.
    fp8_qkv: bool = False

    def __post_init__(self) -> None:
        if (self.qk_head_dim, self.v_head_dim, self.rope_dim) != (192, 128, 64):
            raise ValueError("MiMo programs require QK/V/RoPE head dims 192/128/64")
        if self.heads % self.full_kv_heads or self.heads % self.swa_kv_heads:
            raise ValueError("query heads must divide into KV groups")
        if self.qkv_k_stride < self.qk_head_dim:
            raise ValueError("qkv_k_stride must hold a key head")
        if self.fp8_qkv and (self.heads * self.qk_head_dim) % 128 or self.fp8_qkv and self.qkv_k_stride % 128:
            raise ValueError("FP8 qkv needs 128-row blocks per query width and padded key")
        if self.ring_rows < self.window or self.ring_rows & (self.ring_rows - 1):
            raise ValueError("ring_rows must be a power of two covering the window")

    def kv_heads(self, kind: str) -> int:
        return {"full": self.full_kv_heads, "swa": self.swa_kv_heads}[kind]

    def record_elems(self, kind: str) -> int:
        """BF16 elements of one token's KV record: K heads then V heads."""
        return self.kv_heads(kind) * (self.qk_head_dim + self.v_head_dim)

    def qkv_width(self, kind: str) -> int:
        """Joint ``[q_proj; k_proj; v_proj]`` output width (keys ``qkv_k_stride`` apart)."""
        return self.heads * self.qk_head_dim + self.kv_heads(kind) * (self.qkv_k_stride + self.v_head_dim)

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    def rope_theta(self, kind: str) -> float:
        return {"full": self.full_rope_theta, "swa": self.swa_rope_theta}[kind]


MIMO_V2_FLASH = MiMoGeometry()
# MiMo V2.6 Pro (``mimo_v2``, MiMoV2ForCausalLM): 128 query / 8 KV heads on
# both layer kinds, hidden 6144, 384 experts, BF16 router weight.
MIMO_V26_PRO = MiMoGeometry(
    name="mimo_v2_pro", hidden=6144, heads=128, full_kv_heads=8, swa_kv_heads=8, v_scale=0.612,
    full_rope_theta=1.0e7, swa_rope_theta=1.0e4, dense_inter=16384, routed_experts=384, router_fp32=False,
    qkv_k_stride=256, fp8_qkv=True,
)


@dataclass(frozen=True)
class GLMFGeometry:
    """Static GLM 5.3 Flash (``glm5_next``) geometry baked into the ``glmf_*`` programs.

    Hybrid attention: Kimi Delta Attention layers (``kda_heads`` x 128,
    short convolution ``conv_kernel`` over q/k/v, per-key decay gate bounded
    below by ``gate_lower_bound``, FP32 recurrent state) and MLA layers
    without RoPE (``mla_use_nope``: q_lora 1536, a 512 latent, qk/v heads of
    256; the absorbed query is 512 wide and the FP8 latent record is 528
    bytes: 512 E4M3 then 4 FP32 group scales) with a DSA indexer over 4-token
    key pools. Four mHC streams (``hc_mult``) around every sublayer, an
    unweighted stream mean before the final norm; SwiGLU clamped at
    ``swiglu_limit`` in dense, shared and routed experts; a top-8 sigmoid
    router over 288 experts plus one shared expert.
    """

    name: str = "glm53_flash"
    hidden: int = 4096
    heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_dim: int = 256
    v_head_dim: int = 256
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    kda_heads: int = 64
    kda_head_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0
    dense_inter: int = 12288
    moe_inter: int = 2048
    routed_experts: int = 288
    top_k: int = 8
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    norm_eps: float = 1.0e-5
    index_norm_eps: float = 1.0e-6
    page_rows: int = 64

    def __post_init__(self) -> None:
        if (self.kv_lora_rank, self.kda_head_dim, self.conv_kernel) != (512, 128, 4):
            raise ValueError("GLM Flash programs require the 512 latent, 128-wide KDA heads and a 4-tap conv")
        if self.hc_mult != 4 or self.hidden % 2048:
            raise ValueError("GLM Flash programs require 4 mHC streams and hidden % 2048 == 0")

    # MLA (the GLMGeometry names the shared GLM programs read).
    qk_rope_dim: int = 0

    @property
    def qkv_a_width(self) -> int:
        """Joint ``[q_a_proj; kv_a_proj_with_mqa]`` output width (no RoPE key)."""
        return self.q_lora_rank + self.kv_lora_rank

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @property
    def record_bytes(self) -> int:
        """FP8 latent record: 512 E4M3 then 4 FP32 group scales."""
        return self.kv_lora_rank + 4 * (self.kv_lora_rank // 128)

    @property
    def kv_page_bytes(self) -> int:
        return self.page_rows * self.record_bytes

    @property
    def sparse_topk(self) -> int:
        """Selected slots per row: ``index_topk`` plus the open tail pool, padded to 64."""
        return -(-(self.index_topk + self.index_kpool - 1) // 64) * 64

    # KDA
    @property
    def kda_width(self) -> int:
        """Per-projection KDA width (q, k, v, decay gate, output gate)."""
        return self.kda_heads * self.kda_head_dim

    @property
    def kda_in_width(self) -> int:
        """``[q; k; v; f_a; g_a; b]`` in-projection rows."""
        return 3 * self.kda_width + 2 * self.kda_head_dim + self.kda_heads

    @property
    def kda_state_bytes(self) -> int:
        """FP32 recurrent state of one sequence in one layer, ``[heads, v, k]``."""
        return self.kda_heads * self.kda_head_dim * self.kda_head_dim * 4

    @property
    def conv_state_bytes(self) -> int:
        """BF16 short-conv state of one sequence in one layer: the last 3 q/k/v inputs."""
        return (self.conv_kernel - 1) * 3 * self.kda_width * 2


GLM53_FLASH = GLMFGeometry()


@dataclass(frozen=True)
class Qwen4Geometry:
    """Static Qwen 3.8 Flash Next (``qwen4_exp``) geometry baked into the ``qwen4_*`` programs.

    48 layers: Gated DeltaNet (GDN) linear attention with every fourth layer
    full attention. GDN: ``gdn_key_heads`` x 128 query/key heads shared by
    ``gdn_value_heads`` x 128 value heads (value head ``h`` reads key head
    ``h // 3``), a 4-tap causal conv + SiLU over ``[q; k; v]``, L2-normalized
    q/k, per-head decay ``-exp(A_log) * softplus(a + dt_bias)`` and write
    strength ``sigmoid(b)``, FP32 recurrent state, a gated RMSNorm with a
    *sigmoid* gate ``z`` (weight used as ``w``, not ``1 + w``). Full
    attention: GQA 24 query heads / 2 KV heads of 256, per-head
    ``[q | gate]`` rows in q_proj, q/k RMSNorm (``1 + w``), NeoX RoPE on the
    first ``rope_dim`` (64) dims (theta 1e7; the interleaved mRoPE sections
    reduce to plain RoPE for text), a sigmoid output gate, and the QSA
    indexer: ``index_heads`` query heads and one key of 128, 4-token key
    blocks (mean of raw keys, then ``1 + w`` RMSNorm, then RoPE at the
    block's first position), score ``sum_h relu(q_h . k) / sqrt(128)``, the
    top ``index_budget / index_block`` blocks plus the open tail (every token
    up to ``index_budget + index_block - 1``). Low-rank gated
    hyper-connections (``hc_count`` BF16 streams of ``hidden``, rank
    ``hc_lowrank``) around every sublayer, no final norm (the stream mixer
    feeds lm_head). MoE: softmax top-10 of 512 experts (renormalized), one
    shared expert with a sigmoid gate, SiLU without a clamp. PLE (layer 1):
    16 hashed n-gram rows of 160 per token, key/value projections gating each
    stream, and a dilated (``ngram_size``) depthwise conv of ``ple_conv``
    taps.
    """

    name: str = "qwen38_flash_next"
    hidden: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rope_dim: int = 64
    rope_theta: float = 10_000_000.0
    index_heads: int = 4
    index_head_dim: int = 128
    index_budget: int = 2048
    index_block: int = 4
    gdn_key_heads: int = 16
    gdn_value_heads: int = 48
    gdn_head_dim: int = 128
    conv_kernel: int = 4
    moe_inter: int = 640
    shared_inter: int = 640
    routed_experts: int = 512
    top_k: int = 10
    vocab: int = 248320
    norm_eps: float = 1.0e-6
    ple_dim: int = 2560
    ple_rows: int = 16
    ple_row_dim: int = 160
    ple_conv: int = 4
    ngram_size: int = 3
    page_rows: int = 64

    def __post_init__(self) -> None:
        if (self.gdn_head_dim, self.conv_kernel, self.hc_count) != (128, 4, 4):
            raise ValueError("Qwen4 programs require 128-wide GDN heads, a 4-tap conv and 4 streams")
        if self.gdn_value_heads % self.gdn_key_heads or self.heads % self.kv_heads:
            raise ValueError("GDN value heads and attention heads must group evenly")

    @property
    def hc_width(self) -> int:
        """Features of all streams of one row: ``hc_count * hidden``."""
        return self.hc_count * self.hidden

    # GDN
    @property
    def gdn_key_width(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_conv_width(self) -> int:
        """Short-conv channels ``[q; k; v]``."""
        return 2 * self.gdn_key_width + self.gdn_value_width

    @property
    def gdn_in_width(self) -> int:
        """``[in_proj_qkv; in_proj_z; in_proj_b; in_proj_a]`` in-projection rows."""
        return self.gdn_conv_width + self.gdn_value_width + 2 * self.gdn_value_heads

    @property
    def gdn_state_bytes(self) -> int:
        """FP32 recurrent state of one sequence in one layer."""
        return self.gdn_value_heads * self.gdn_head_dim * self.gdn_head_dim * 4

    @property
    def gdn_conv_state_bytes(self) -> int:
        """BF16 short-conv state of one sequence in one layer: the last 3 q/k/v inputs."""
        return (self.conv_kernel - 1) * self.gdn_conv_width * 2

    # Full attention
    @property
    def attn_in_width(self) -> int:
        """``[q_proj (q|gate per head); k_proj; v_proj; indexer.index_qk_proj]`` rows."""
        return (2 * self.heads * self.head_dim + 2 * self.kv_heads * self.head_dim
                + (self.index_heads + 1) * self.index_head_dim)

    @property
    def record_bytes(self) -> int:
        """BF16 KV record of one token: K ``[kv_heads, head_dim]`` then V."""
        return 2 * self.kv_heads * self.head_dim * 2

    @property
    def kv_page_bytes(self) -> int:
        return self.page_rows * self.record_bytes

    @property
    def index_blocks(self) -> int:
        """Selected key blocks per row past the dense limit."""
        return self.index_budget // self.index_block

    @property
    def dense_limit(self) -> int:
        """Visible tokens up to which every token is selected."""
        return self.index_budget + self.index_block - 1

    @property
    def sparse_topk(self) -> int:
        """Selected slots per row: the budget plus the open tail, padded to 64."""
        return -(-self.dense_limit // 64) * 64

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    # PLE
    @property
    def ple_state_rows(self) -> int:
        """Rows of the dilated PLE conv state: ``(ple_conv - 1) * ngram_size``."""
        return (self.ple_conv - 1) * self.ngram_size


QWEN38_FLASH_NEXT = Qwen4Geometry()


_TORCH_TO_CUTE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
    torch.uint32: cutlass.Uint32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


@dataclass(frozen=True)
class Operand:
    """One pointer argument of an exported program.

    ``dtype`` is the element type the kernel reads or writes through the
    pointer, ``shape`` a human/machine readable shape using ``rows`` for the
    live row count, ``role`` one of ``in``, ``out``, ``inout`` or ``scratch``,
    and ``align`` the byte alignment the compiled code assumes.
    """

    name: str
    dtype: torch.dtype
    shape: str
    role: str = "in"
    align: int = 16
    note: str = ""

    @property
    def cute_dtype(self):
        return _TORCH_TO_CUTE[self.dtype]


@dataclass(frozen=True)
class Scalar:
    """One scalar launch argument (``int32``/``int64``/``float32``)."""

    name: str
    ctype: str = "int32"
    note: str = ""

    def placeholder(self):
        return {"int32": cutlass.Int32(1), "int64": cutlass.Int64(1),
                "float32": cutlass.Float32(1.0)}[self.ctype]

    def value(self, value):
        return {"int32": cutlass.Int32, "int64": cutlass.Int64,
                "float32": cutlass.Float32}[self.ctype](value)


@dataclass(frozen=True)
class AotProgram:
    """A compiled raw-pointer program plus its documented ABI.

    ``compiled`` is the CuTe DSL JIT executor (it provides ``export_to_c``).
    ``scratch`` maps each scratch operand name to a byte-size function of the
    live/capacity row count; ``geometry`` records the static compile inputs.
    """

    name: str
    compiled: object
    operands: tuple[Operand, ...]
    scalars: tuple[Scalar, ...]
    geometry: Mapping[str, object] = field(default_factory=dict)
    scratch: Mapping[str, Callable[[int], int]] = field(default_factory=dict)
    doc: str = ""

    @property
    def abi(self) -> dict[str, object]:
        return {
            "pointers": [(o.name, str(o.dtype).removeprefix("torch."), o.shape, o.role)
                         for o in self.operands],
            "scalars": [(s.name, s.ctype) for s in self.scalars],
            "stream": "stream",
        }

    def scratch_bytes(self, rows: int) -> dict[str, int]:
        """Byte size of every scratch pointer for ``rows`` (capacity) rows."""
        return {name: int(fn(int(rows))) for name, fn in self.scratch.items()}

    def export_to_c(self, file_path: str, file_name: str, function_prefix: str):
        return self.compiled.export_to_c(str(file_path), file_name, function_prefix)

    def launch(self, *tensors: torch.Tensor, scalars: Sequence[object], stream=None):
        """Launch from torch tensors in ABI order (test/diagnostic helper).

        ``tensors`` may contain ``None`` only for operands documented as
        ignorable; a 16-byte-aligned dummy address is passed in that case.
        """
        if len(tensors) != len(self.operands):
            raise ValueError(f"{self.name}: expected {len(self.operands)} pointers, got {len(tensors)}")
        if len(scalars) != len(self.scalars):
            raise ValueError(f"{self.name}: expected {len(self.scalars)} scalars")
        pointers = []
        for operand, tensor in zip(self.operands, tensors):
            address = 16 if tensor is None else int(tensor.data_ptr())
            if tensor is not None and address % operand.align:
                raise ValueError(f"{self.name}.{operand.name} is not {operand.align}-byte aligned")
            pointers.append(make_ptr(operand.cute_dtype, address, cute.AddressSpace.gmem,
                                     assumed_align=operand.align))
        values = [s.value(v) for s, v in zip(self.scalars, scalars)]
        self.compiled(*pointers, *values, current_cuda_stream() if stream is None else stream)


def compile_program(
    launch: object,
    *,
    name: str,
    operands: Sequence[Operand],
    scalars: Sequence[Scalar],
    key: tuple,
    geometry: Mapping[str, object] | None = None,
    scratch: Mapping[str, Callable[[int], int]] | None = None,
    doc: str = "",
    version: int = 1,
) -> AotProgram:
    """Compile ``launch(*pointers, *scalars, stream)`` into an AotProgram."""
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    full_key = (name, tuple(key), capability, device)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=full_key)
    pointers = [make_ptr(o.cute_dtype, 16, cute.AddressSpace.gmem, assumed_align=o.align)
                for o in operands]
    compiled = compile_cute(
        launch, *pointers, *(s.placeholder() for s in scalars), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(f"cuteafd.{name}", version, full_key),
    )
    return AotProgram(
        name=name, compiled=compiled, operands=tuple(operands), scalars=tuple(scalars),
        geometry=dict(geometry or {}), scratch=dict(scratch or {}), doc=doc,
    )


@contextmanager
def exportable_compilation():
    """Compile with the b12x object caches disabled so export_to_c has IR."""
    names = ("B12X_COMPILE_DISK_CACHE", "B12X_COMPILE_MEMORY_CACHE")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "0"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_C_TYPES = {"int32": "int32_t", "int64": "int64_t", "float32": "float"}


def validate_exported_header(program: AotProgram, header: str | Path, symbol: str) -> dict:
    """Check the generated wrapper signature and argument order against the ABI.

    Returns ``{"symbol": <_mlir entry>, "argument_count": n}`` for manifests.
    """
    text = Path(header).read_text() if not isinstance(header, str) or "\n" not in header else header
    expected = [f"{symbol}_Kernel_Module_t *module"]
    expected += [f"void *{o.name}" for o in program.operands]
    expected += [f"{_C_TYPES[s.ctype]} {s.name}" for s in program.scalars]
    expected += ["cudaStream_t stream"]
    signature = re.search(
        r"static inline int32_t cute_dsl_" + re.escape(symbol) + r"_wrapper\(([^)]*)\)", text)
    if signature is None:
        raise ValueError(f"{program.name}: wrapper for {symbol} not found in header")
    if re.sub(r"\s+", "", signature[1]) != re.sub(r"\s+", "", ",".join(expected)):
        raise ValueError(f"{program.name}: unexpected wrapper signature {signature[1]!r}")
    names = [o.name for o in program.operands] + [s.name for s in program.scalars] + ["stream"]
    count = len(names)
    arguments = re.search(r"void \*args\[" + str(count + 1) + r"\] = \{([^}]*)\}", text)
    if arguments is None or re.sub(r"\s+", "", arguments[1]) != ",".join("&" + n for n in names + ["ret"]):
        raise ValueError(f"{program.name}: unexpected generated argument order")
    entries = re.findall(r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\);", text)
    if len(entries) != 1:
        raise ValueError(f"{program.name}: expected one generated entry point, got {entries}")
    return {"symbol": entries[0], "argument_count": count}
