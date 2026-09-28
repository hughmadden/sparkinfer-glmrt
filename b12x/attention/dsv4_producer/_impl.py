from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ...gemm._shared.block_fp8 import (
    BlockFP8LinearBinding,
    BlockFP8LinearScratchCaps,
    BlockFP8LinearWeight,
    _scratch_plan as _block_fp8_scratch_plan,
    block_fp8_linear_mxfp8,
    pack_block_fp8_linear_weight_mxfp8,
)


DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_KV_PAGE_SIZE = 256
DSV4_KV_PAYLOAD_BYTES = 576
DSV4_KV_SCALE_BYTES = 8
DSV4_KV_PAGE_BYTES = 149_760
DSV4_NVFP4_RECORD_BYTES = 432
DSV4_NVFP4_PAGE_BYTES = DSV4_KV_PAGE_SIZE * DSV4_NVFP4_RECORD_BYTES
DSV4_FP8_MAX = 448.0
DSV4_INDEX_HEADS = 64
DSV4_INDEX_HEAD_DIM = 128
DSV4_INDEX_NOPE_DIM = 64
DSV4_INDEX_WEIGHT_SCALE = (DSV4_INDEX_HEAD_DIM * DSV4_INDEX_HEADS) ** -0.5
_SCRATCH_ALIGNMENT = 1024


def _align_up(value: int, alignment: int = _SCRATCH_ALIGNMENT) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


# The checkpoint quantizes every projection input per token over 128-wide K
# blocks (UE8M0 scale replicated across four hardware K32 slots).
_ACTIVATION_BLOCK_SIZE = 128


@dataclass(frozen=True)
class _ProducerLinear:
    """One ``gemm.block_fp8_linear`` declaration owned by a producer plan.

    CUDA declarations are prepared with their default configuration when the
    producer is planned, so the retained scratch size includes any split-K
    workspace of the selected configuration. A PreparationSession may prepare
    ``plan`` first (for example to autotune); the producer then sizes its arena
    from that prepared selection. CPU declarations only describe the arena and
    reserve the activation staging (the non-split layout).
    """

    caps: BlockFP8LinearScratchCaps
    plan: object

    @property
    def scratch_nbytes(self) -> int:
        if self.plan is None:
            return int(
                _block_fp8_scratch_plan(self.caps, (0, 0)).scratch_specs()[0].nbytes
            )
        return int(self.plan.scratch_specs()[0].nbytes)

    def bind(
        self,
        *,
        scratch: torch.Tensor,
        source: torch.Tensor,
        packed_weight: BlockFP8LinearWeight,
        output: torch.Tensor,
        expected_m: int | None = None,
    ) -> BlockFP8LinearBinding:
        if self.plan is None:
            raise ValueError("a CPU DSV4 producer declaration cannot be bound")
        from ...gemm import block_fp8_linear

        return block_fp8_linear.bind(
            self.plan,
            scratch=scratch,
            source=source,
            packed_weight=packed_weight,
            output=output,
            expected_m=expected_m,
            activation_block_size=_ACTIVATION_BLOCK_SIZE,
        )


def _plan_linear(
    *, device: torch.device, max_tokens: int, in_features: int, out_features: int,
    name: str,
) -> _ProducerLinear:
    caps = BlockFP8LinearScratchCaps(
        device=device,
        max_tokens=max_tokens,
        in_features=in_features,
        out_features=out_features,
        output_dtype=torch.bfloat16,
        activation_block_size=_ACTIVATION_BLOCK_SIZE,
    )
    if caps.device.type != "cuda":
        return _ProducerLinear(caps=caps, plan=None)
    from ...gemm import block_fp8_linear
    from ...preparation import PreparedCall, prepare_default

    plan = block_fp8_linear.plan(caps)
    if plan.prepared is None:
        prepare_default(plan.request(
            name=f"{name}.m{caps.max_tokens}",
            prepare_call=lambda state: PreparedCall(run=lambda: None),
        ))
    return _ProducerLinear(caps=caps, plan=plan)


@dataclass(frozen=True)
class DSV4ProducerWeights:
    qkv_rank: BlockFP8LinearWeight
    kv: BlockFP8LinearWeight
    q: BlockFP8LinearWeight
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    hidden: int
    q_lora_rank: int
    heads: int
    head_dim: int = DSV4_HEAD_DIM


@dataclass(frozen=True)
class DSV4KVProducerWeights:
    kv: BlockFP8LinearWeight
    kv_norm: torch.Tensor
    hidden: int
    head_dim: int = DSV4_HEAD_DIM


@dataclass(frozen=True)
class DSV4IndexerProducerWeights:
    q: BlockFP8LinearWeight
    weights_projection: torch.Tensor
    weights_projection_t: torch.Tensor
    hidden: int
    q_lora_rank: int
    heads: int = DSV4_INDEX_HEADS
    head_dim: int = DSV4_INDEX_HEAD_DIM


@dataclass(frozen=True, kw_only=True)
class DSV4ProducerCaps:
    device: torch.device | str
    max_tokens: int
    hidden: int
    q_lora_rank: int
    heads: int
    head_dim: int = DSV4_HEAD_DIM
    nope_dim: int = DSV4_NOPE_DIM
    rope_dim: int = DSV4_ROPE_DIM
    page_size: int = DSV4_KV_PAGE_SIZE
    cache_format: str = "fp8"
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "hidden", int(self.hidden))
        object.__setattr__(self, "q_lora_rank", int(self.q_lora_rank))
        object.__setattr__(self, "heads", int(self.heads))
        object.__setattr__(self, "cache_format", str(self.cache_format).strip().lower())
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.hidden not in (4096, 7168):
            raise ValueError(
                f"DSV4 producer hidden must be 4096 or 7168, got {self.hidden}"
            )
        if self.q_lora_rank not in (1024, 1536):
            raise ValueError(
                f"DSV4 producer q_lora_rank must be 1024 or 1536, got {self.q_lora_rank}"
            )
        if self.heads not in (64, 128):
            raise ValueError(f"DSV4 producer heads must be 64 or 128, got {self.heads}")
        if (self.head_dim, self.nope_dim, self.rope_dim) != (512, 448, 64):
            raise ValueError(
                "DSV4 producer requires head/nope/rope 512/448/64, got "
                f"{self.head_dim}/{self.nope_dim}/{self.rope_dim}"
            )
        if self.page_size != DSV4_KV_PAGE_SIZE:
            raise ValueError(
                f"DSV4 producer requires {DSV4_KV_PAGE_SIZE}-token pages, got {self.page_size}"
            )
        if self.cache_format not in ("fp8", "nvfp4"):
            raise ValueError(
                "DSV4 producer cache_format must be fp8 or nvfp4, got "
                f"{self.cache_format!r}"
            )
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"DSV4 producer requires BF16 activations, got {self.dtype}"
            )

    @property
    def main_page_bytes(self) -> int:
        return (
            DSV4_NVFP4_PAGE_BYTES
            if self.cache_format == "nvfp4"
            else DSV4_KV_PAGE_BYTES
        )


@dataclass(frozen=True)
class _DSV4ProducerScratchLayout:
    nbytes: int
    qkv_linear_offset: int
    qkv_linear_bytes: int
    q_linear_offset: int
    q_linear_bytes: int
    qkv_output_offset: int
    q_rank_offset: int


@dataclass(frozen=True)
class _DSV4KVProducerScratchLayout:
    nbytes: int
    kv_linear_offset: int
    kv_linear_bytes: int
    kv_output_offset: int
    kv_output_bytes: int


@dataclass(frozen=True)
class _DSV4IndexerProducerScratchLayout:
    nbytes: int
    q_linear_offset: int
    q_linear_bytes: int
    q_output_offset: int
    q_output_bytes: int
    weights_output_offset: int
    weights_output_bytes: int


@dataclass(frozen=True, kw_only=True)
class DSV4ProducerBinding:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    main_slots: torch.Tensor
    cos_sin_cache: torch.Tensor
    main_kv_cache: torch.Tensor
    query: torch.Tensor
    weights: DSV4ProducerWeights
    qkv_linear: BlockFP8LinearBinding
    q_linear: BlockFP8LinearBinding
    qkv_output: torch.Tensor
    q_rank: torch.Tensor
    cache_format: str
    eps: float

    def run(self) -> torch.Tensor:
        return run_dsv4_producer(binding=self)


@dataclass(frozen=True, kw_only=True)
class DSV4KVProducerBinding:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    main_slots: torch.Tensor
    cos_sin_cache: torch.Tensor
    main_kv_cache: torch.Tensor
    weights: DSV4KVProducerWeights | DSV4ProducerWeights
    kv_linear: BlockFP8LinearBinding
    kv_output: torch.Tensor
    cache_format: str
    eps: float

    def run(self) -> torch.Tensor:
        return run_dsv4_kv_producer(binding=self)


@dataclass(frozen=True, kw_only=True)
class DSV4IndexerProducerCaps:
    device: torch.device | str
    max_tokens: int
    hidden: int
    q_lora_rank: int
    heads: int = DSV4_INDEX_HEADS
    head_dim: int = DSV4_INDEX_HEAD_DIM
    rope_dim: int = DSV4_ROPE_DIM
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "hidden", int(self.hidden))
        object.__setattr__(self, "q_lora_rank", int(self.q_lora_rank))
        object.__setattr__(self, "heads", int(self.heads))
        object.__setattr__(self, "head_dim", int(self.head_dim))
        object.__setattr__(self, "rope_dim", int(self.rope_dim))
        if self.max_tokens <= 0:
            raise ValueError("DSV4 indexer producer max_tokens must be positive")
        expected_q_rank = {4_096: 1_024, 7_168: 1_536}.get(self.hidden)
        if expected_q_rank is None or self.q_lora_rank != expected_q_rank:
            raise ValueError(
                "DSV4 indexer producer requires Flash 4096/1024 or Pro "
                f"7168/1536 hidden/Q-rank geometry, got {self.hidden}/{self.q_lora_rank}"
            )
        if (self.heads, self.head_dim, self.rope_dim) != (
            DSV4_INDEX_HEADS,
            DSV4_INDEX_HEAD_DIM,
            DSV4_ROPE_DIM,
        ):
            raise ValueError(
                "DSV4 indexer producer requires heads/head/rope 64/128/64, got "
                f"{self.heads}/{self.head_dim}/{self.rope_dim}"
            )
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"DSV4 indexer producer requires BF16 activations, got {self.dtype}"
            )


@dataclass(frozen=True, kw_only=True)
class DSV4IndexerProducerBinding:
    q_rank: torch.Tensor
    hidden_states: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    query: torch.Tensor
    head_weights: torch.Tensor
    weights: DSV4IndexerProducerWeights
    q_linear: BlockFP8LinearBinding
    q_output: torch.Tensor
    weights_output: torch.Tensor

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return run_dsv4_indexer_producer(binding=self)


@dataclass(frozen=True)
class DSV4IndexerProducerPlan:
    caps: DSV4IndexerProducerCaps
    layout: _DSV4IndexerProducerScratchLayout
    q_linear_plan: _ProducerLinear
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q_rank: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        query: torch.Tensor,
        head_weights: torch.Tensor,
        weights: DSV4IndexerProducerWeights,
        expected_m: int | None = None,
    ) -> DSV4IndexerProducerBinding:
        tokens = _validate_indexer_runtime_tensors(
            self.caps,
            q_rank=q_rank,
            hidden_states=hidden_states,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            query=query,
            head_weights=head_weights,
        )
        _validate_indexer_weights(self.caps, weights)
        arena = scratch_tensor(
            scratch, self._scratch_specs, owner="DSV4 indexer producer"
        )
        q_linear_scratch = arena.narrow(
            0, self.layout.q_linear_offset, self.layout.q_linear_bytes
        )
        q_output = (
            arena.narrow(0, self.layout.q_output_offset, self.layout.q_output_bytes)
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.heads * self.caps.head_dim, 1)
            .narrow(0, 0, tokens)
        )
        weights_output = (
            arena.narrow(
                0, self.layout.weights_output_offset, self.layout.weights_output_bytes
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.heads)
            .narrow(0, 0, tokens)
        )
        q_linear = self.q_linear_plan.bind(
            scratch=q_linear_scratch,
            source=q_rank,
            packed_weight=weights.q,
            output=q_output,
            expected_m=expected_m,
        )
        return DSV4IndexerProducerBinding(
            q_rank=q_rank,
            hidden_states=hidden_states,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            query=query,
            head_weights=head_weights,
            weights=weights,
            q_linear=q_linear,
            q_output=q_output,
            weights_output=weights_output,
        )


@dataclass(frozen=True)
class DSV4ProducerPlan:
    caps: DSV4ProducerCaps
    layout: _DSV4ProducerScratchLayout
    qkv_linear_plan: _ProducerLinear
    q_linear_plan: _ProducerLinear
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        main_slots: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        main_kv_cache: torch.Tensor,
        query: torch.Tensor,
        weights: DSV4ProducerWeights,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
    ) -> DSV4ProducerBinding:
        tokens = _validate_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            query=query,
            eps=eps,
        )
        _validate_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 producer")

        qkv_scratch = arena.narrow(
            0, self.layout.qkv_linear_offset, self.layout.qkv_linear_bytes
        )
        q_scratch = arena.narrow(
            0, self.layout.q_linear_offset, self.layout.q_linear_bytes
        )
        qkv_output = (
            arena.narrow(
                0,
                self.layout.qkv_output_offset,
                self.caps.max_tokens
                * (self.caps.q_lora_rank + self.caps.head_dim)
                * torch.empty((), dtype=torch.bfloat16).element_size(),
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.q_lora_rank + self.caps.head_dim, 1)
            .narrow(0, 0, tokens)
        )
        q_rank = (
            arena.narrow(
                0,
                self.layout.q_rank_offset,
                self.caps.max_tokens
                * self.caps.q_lora_rank
                * torch.empty((), dtype=torch.bfloat16).element_size(),
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.q_lora_rank)
            .narrow(0, 0, tokens)
        )
        qkv_linear = self.qkv_linear_plan.bind(
            scratch=qkv_scratch,
            source=hidden_states,
            packed_weight=weights.qkv_rank,
            output=qkv_output,
            expected_m=expected_m,
        )
        q_linear = self.q_linear_plan.bind(
            scratch=q_scratch,
            source=q_rank,
            packed_weight=weights.q,
            output=query.view(tokens, self.caps.heads * self.caps.head_dim, 1),
            expected_m=expected_m,
        )
        return DSV4ProducerBinding(
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            query=query,
            weights=weights,
            qkv_linear=qkv_linear,
            q_linear=q_linear,
            qkv_output=qkv_output,
            q_rank=q_rank,
            cache_format=self.caps.cache_format,
            eps=float(eps),
        )


@dataclass(frozen=True)
class DSV4KVProducerPlan:
    caps: DSV4ProducerCaps
    layout: _DSV4KVProducerScratchLayout
    kv_linear_plan: _ProducerLinear
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        main_slots: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        main_kv_cache: torch.Tensor,
        weights: DSV4KVProducerWeights | DSV4ProducerWeights,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
    ) -> DSV4KVProducerBinding:
        tokens = _validate_kv_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            eps=eps,
        )
        _validate_kv_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 KV producer")
        kv_scratch = arena.narrow(
            0, self.layout.kv_linear_offset, self.layout.kv_linear_bytes
        )
        kv_output = (
            arena.narrow(0, self.layout.kv_output_offset, self.layout.kv_output_bytes)
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.head_dim, 1)
            .narrow(0, 0, tokens)
        )
        kv_linear = self.kv_linear_plan.bind(
            scratch=kv_scratch,
            source=hidden_states,
            packed_weight=weights.kv,
            output=kv_output,
            expected_m=expected_m,
        )
        return DSV4KVProducerBinding(
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            weights=weights,
            kv_linear=kv_linear,
            kv_output=kv_output,
            cache_format=self.caps.cache_format,
            eps=float(eps),
        )


def pack_dsv4_producer_weights(
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
) -> DSV4ProducerWeights:
    """Pack checkpoint tensors and concatenate the shared-input projections."""

    for name, tensor in (
        ("wq_a", wq_a),
        ("wq_a_scale", wq_a_scale),
        ("wq_b", wq_b),
        ("wq_b_scale", wq_b_scale),
        ("wkv", wkv),
        ("wkv_scale", wkv_scale),
        ("q_norm", q_norm),
        ("kv_norm", kv_norm),
    ):
        _check_cuda_tensor(name, tensor)
    if wq_a.ndim != 2 or wq_b.ndim != 2 or wkv.ndim != 2:
        raise ValueError("DSV4 producer projection weights must be rank-2")
    q_lora_rank, hidden = map(int, wq_a.shape)
    head_dim, kv_hidden = map(int, wkv.shape)
    q_width, q_input = map(int, wq_b.shape)
    if hidden != kv_hidden or q_input != q_lora_rank:
        raise ValueError(
            "DSV4 producer projection dimensions disagree: "
            f"wq_a={tuple(wq_a.shape)} wq_b={tuple(wq_b.shape)} wkv={tuple(wkv.shape)}"
        )
    if head_dim != DSV4_HEAD_DIM or q_width % head_dim:
        raise ValueError(
            f"DSV4 producer requires wkv N=512 and wq_b N divisible by 512, got {head_dim}/{q_width}"
        )
    heads = q_width // head_dim
    caps = DSV4ProducerCaps(
        device=wq_a.device,
        max_tokens=1,
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
    )
    del caps
    if q_norm.shape != (q_lora_rank,) or kv_norm.shape != (head_dim,):
        raise ValueError(
            f"DSV4 norm shapes must be {(q_lora_rank,)} and {(head_dim,)}, got "
            f"{tuple(q_norm.shape)} and {tuple(kv_norm.shape)}"
        )
    if q_norm.dtype != torch.bfloat16 or kv_norm.dtype != torch.bfloat16:
        raise ValueError("DSV4 producer norm weights must be BF16")
    if not q_norm.is_contiguous() or not kv_norm.is_contiguous():
        raise ValueError("DSV4 producer norm weights must be contiguous")
    if any(tensor.device != wq_a.device for tensor in (wq_b, wkv, q_norm, kv_norm)):
        raise ValueError("DSV4 producer weights must share one CUDA device")

    # Both projections consume the identical checkpoint activation-quantized
    # hidden row. Concatenating their N dimension is exact because q_lora_rank
    # and head_dim are independently aligned to the checkpoint's N128 blocks.
    qkv_weight = torch.cat((wq_a, wkv), dim=0)
    qkv_scale = torch.cat((wq_a_scale, wkv_scale), dim=0)
    return DSV4ProducerWeights(
        qkv_rank=pack_block_fp8_linear_weight_mxfp8(qkv_weight, qkv_scale),
        kv=pack_block_fp8_linear_weight_mxfp8(wkv, wkv_scale),
        q=pack_block_fp8_linear_weight_mxfp8(wq_b, wq_b_scale),
        q_norm=q_norm.detach(),
        kv_norm=kv_norm.detach(),
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
        head_dim=head_dim,
    )


def pack_dsv4_kv_producer_weights(
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm: torch.Tensor,
) -> DSV4KVProducerWeights:
    """Pack only the weights needed by integrated dSpark prompt priming."""

    for name, tensor in (
        ("wkv", wkv),
        ("wkv_scale", wkv_scale),
        ("kv_norm", kv_norm),
    ):
        _check_cuda_tensor(name, tensor)
    if wkv.ndim != 2:
        raise ValueError("DSV4 KV-only projection weight must be rank-2")
    head_dim, hidden = map(int, wkv.shape)
    if head_dim != DSV4_HEAD_DIM or hidden not in (4_096, 7_168):
        raise ValueError(
            "DSV4 KV-only weights require [512,4096] or [512,7168], got "
            f"{tuple(wkv.shape)}"
        )
    if wkv_scale.shape != (head_dim // 128, hidden // 128):
        raise ValueError(
            "DSV4 KV-only scale shape must match 128x128 weight blocks, got "
            f"{tuple(wkv_scale.shape)}"
        )
    if kv_norm.shape != (head_dim,):
        raise ValueError(
            f"DSV4 KV-only norm shape must be {(head_dim,)}, got {tuple(kv_norm.shape)}"
        )
    if kv_norm.dtype != torch.bfloat16 or not kv_norm.is_contiguous():
        raise ValueError("DSV4 KV-only norm weight must be contiguous BF16")
    if wkv_scale.device != wkv.device or kv_norm.device != wkv.device:
        raise ValueError("DSV4 KV-only weights must share one CUDA device")
    return DSV4KVProducerWeights(
        kv=pack_block_fp8_linear_weight_mxfp8(wkv, wkv_scale),
        kv_norm=kv_norm.detach(),
        hidden=hidden,
        head_dim=head_dim,
    )


def pack_dsv4_indexer_producer_weights(
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    weights_projection: torch.Tensor,
) -> DSV4IndexerProducerWeights:
    """Pack the checkpoint's separate learned-index query and head weights."""

    for name, tensor in (
        ("wq_b", wq_b),
        ("wq_b_scale", wq_b_scale),
        ("weights_projection", weights_projection),
    ):
        _check_cuda_tensor(name, tensor)
    if wq_b.ndim != 2 or weights_projection.ndim != 2:
        raise ValueError("DSV4 indexer projection weights must be rank-2")
    query_width, q_lora_rank = map(int, wq_b.shape)
    heads, hidden = map(int, weights_projection.shape)
    if query_width != DSV4_INDEX_HEADS * DSV4_INDEX_HEAD_DIM:
        raise ValueError(
            "DSV4 indexer wq_b must have output width "
            f"{DSV4_INDEX_HEADS * DSV4_INDEX_HEAD_DIM}, got {query_width}"
        )
    caps = DSV4IndexerProducerCaps(
        device=wq_b.device,
        max_tokens=1,
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
    )
    del caps
    if weights_projection.dtype != torch.bfloat16:
        raise ValueError("DSV4 indexer weights projection must be BF16")
    if not weights_projection.is_contiguous():
        raise ValueError("DSV4 indexer weights projection must be contiguous")
    if weights_projection.device != wq_b.device:
        raise ValueError("DSV4 indexer weights must share one CUDA device")
    return DSV4IndexerProducerWeights(
        q=pack_block_fp8_linear_weight_mxfp8(wq_b, wq_b_scale),
        weights_projection=weights_projection.detach(),
        weights_projection_t=weights_projection.detach().t(),
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
    )


def plan_dsv4_producer(caps: DSV4ProducerCaps) -> DSV4ProducerPlan:
    qkv_linear_plan = _plan_linear(
        device=caps.device,
        max_tokens=caps.max_tokens,
        in_features=caps.hidden,
        out_features=caps.q_lora_rank + caps.head_dim,
        name="dsv4_producer.qkv",
    )
    q_linear_plan = _plan_linear(
        device=caps.device,
        max_tokens=caps.max_tokens,
        in_features=caps.q_lora_rank,
        out_features=caps.heads * caps.head_dim,
        name="dsv4_producer.q",
    )
    cursor = 0
    qkv_linear_offset = _align_up(cursor)
    qkv_linear_bytes = qkv_linear_plan.scratch_nbytes
    cursor = qkv_linear_offset + qkv_linear_bytes
    q_linear_offset = _align_up(cursor)
    q_linear_bytes = q_linear_plan.scratch_nbytes
    cursor = q_linear_offset + q_linear_bytes
    qkv_output_offset = _align_up(cursor)
    cursor = (
        qkv_output_offset + caps.max_tokens * (caps.q_lora_rank + caps.head_dim) * 2
    )
    q_rank_offset = _align_up(cursor)
    cursor = q_rank_offset + caps.max_tokens * caps.q_lora_rank * 2
    layout = _DSV4ProducerScratchLayout(
        nbytes=_align_up(cursor),
        qkv_linear_offset=qkv_linear_offset,
        qkv_linear_bytes=qkv_linear_bytes,
        q_linear_offset=q_linear_offset,
        q_linear_bytes=q_linear_bytes,
        qkv_output_offset=qkv_output_offset,
        q_rank_offset=q_rank_offset,
    )
    return DSV4ProducerPlan(
        caps=caps,
        layout=layout,
        qkv_linear_plan=qkv_linear_plan,
        q_linear_plan=q_linear_plan,
        _scratch_specs=(
            scratch_buffer_spec(
                "dsv4_producer.scratch", nbytes=layout.nbytes, device=caps.device
            ),
        ),
    )


def plan_dsv4_kv_producer(caps: DSV4ProducerCaps) -> DSV4KVProducerPlan:
    """Plan the dSpark target-main path without unused query scratch."""

    kv_linear_plan = _plan_linear(
        device=caps.device,
        max_tokens=caps.max_tokens,
        in_features=caps.hidden,
        out_features=caps.head_dim,
        name="dsv4_kv_producer.kv",
    )
    cursor = 0
    kv_linear_offset = _align_up(cursor)
    kv_linear_bytes = kv_linear_plan.scratch_nbytes
    cursor = kv_linear_offset + kv_linear_bytes
    kv_output_offset = _align_up(cursor)
    kv_output_bytes = caps.max_tokens * caps.head_dim * 2
    cursor = kv_output_offset + kv_output_bytes
    layout = _DSV4KVProducerScratchLayout(
        nbytes=_align_up(cursor),
        kv_linear_offset=kv_linear_offset,
        kv_linear_bytes=kv_linear_bytes,
        kv_output_offset=kv_output_offset,
        kv_output_bytes=kv_output_bytes,
    )
    return DSV4KVProducerPlan(
        caps=caps,
        layout=layout,
        kv_linear_plan=kv_linear_plan,
        _scratch_specs=(
            scratch_buffer_spec(
                "dsv4_kv_producer.scratch", nbytes=layout.nbytes, device=caps.device
            ),
        ),
    )


def plan_dsv4_indexer_producer(
    caps: DSV4IndexerProducerCaps,
) -> DSV4IndexerProducerPlan:
    q_linear_plan = _plan_linear(
        device=caps.device,
        max_tokens=caps.max_tokens,
        in_features=caps.q_lora_rank,
        out_features=caps.heads * caps.head_dim,
        name="dsv4_indexer_producer.q",
    )
    cursor = 0
    q_linear_offset = _align_up(cursor)
    q_linear_bytes = q_linear_plan.scratch_nbytes
    cursor = q_linear_offset + q_linear_bytes
    q_output_offset = _align_up(cursor)
    q_output_bytes = caps.max_tokens * caps.heads * caps.head_dim * 2
    cursor = q_output_offset + q_output_bytes
    weights_output_offset = _align_up(cursor)
    weights_output_bytes = caps.max_tokens * caps.heads * 2
    cursor = weights_output_offset + weights_output_bytes
    layout = _DSV4IndexerProducerScratchLayout(
        nbytes=_align_up(cursor),
        q_linear_offset=q_linear_offset,
        q_linear_bytes=q_linear_bytes,
        q_output_offset=q_output_offset,
        q_output_bytes=q_output_bytes,
        weights_output_offset=weights_output_offset,
        weights_output_bytes=weights_output_bytes,
    )
    return DSV4IndexerProducerPlan(
        caps=caps,
        layout=layout,
        q_linear_plan=q_linear_plan,
        _scratch_specs=(
            scratch_buffer_spec(
                "dsv4_indexer_producer.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


def _validate_weights(caps: DSV4ProducerCaps, weights: DSV4ProducerWeights) -> None:
    expected = (caps.hidden, caps.q_lora_rank, caps.heads, caps.head_dim)
    actual = (weights.hidden, weights.q_lora_rank, weights.heads, weights.head_dim)
    if actual != expected:
        raise ValueError(f"DSV4 producer weights {actual} do not match caps {expected}")
    if weights.qkv_rank.in_features != caps.hidden or weights.qkv_rank.out_features != (
        caps.q_lora_rank + caps.head_dim
    ):
        raise ValueError("DSV4 joint Q-rank/KV packed weight has drifted geometry")
    if (
        weights.kv.in_features != caps.hidden
        or weights.kv.out_features != caps.head_dim
    ):
        raise ValueError("DSV4 KV-only packed weight has drifted geometry")
    if weights.q.in_features != caps.q_lora_rank or weights.q.out_features != (
        caps.heads * caps.head_dim
    ):
        raise ValueError("DSV4 Q-B packed weight has drifted geometry")


def _validate_kv_weights(
    caps: DSV4ProducerCaps,
    weights: DSV4KVProducerWeights | DSV4ProducerWeights,
) -> None:
    if not isinstance(weights, (DSV4KVProducerWeights, DSV4ProducerWeights)):
        raise TypeError("DSV4 KV producer requires KVWeights or full Weights")
    if (weights.hidden, weights.head_dim) != (caps.hidden, caps.head_dim):
        raise ValueError(
            "DSV4 KV producer weight geometry "
            f"{(weights.hidden, weights.head_dim)} does not match "
            f"{(caps.hidden, caps.head_dim)}"
        )
    if (
        weights.kv.in_features != caps.hidden
        or weights.kv.out_features != caps.head_dim
    ):
        raise ValueError("DSV4 KV-only packed weight has drifted geometry")
    if weights.kv_norm.shape != (caps.head_dim,):
        raise ValueError("DSV4 KV-only norm weight has drifted geometry")


def _validate_indexer_weights(
    caps: DSV4IndexerProducerCaps, weights: DSV4IndexerProducerWeights
) -> None:
    expected = (
        caps.hidden,
        caps.q_lora_rank,
        caps.heads,
        caps.head_dim,
    )
    actual = (
        weights.hidden,
        weights.q_lora_rank,
        weights.heads,
        weights.head_dim,
    )
    if actual != expected:
        raise ValueError(
            f"DSV4 indexer producer weights {actual} do not match caps {expected}"
        )
    if (
        weights.q.in_features != caps.q_lora_rank
        or weights.q.out_features != caps.heads * caps.head_dim
    ):
        raise ValueError("DSV4 indexer packed Q weight has drifted geometry")
    if weights.weights_projection.shape != (caps.heads, caps.hidden):
        raise ValueError("DSV4 indexer head-weight projection has drifted geometry")


def _validate_indexer_runtime_tensors(
    caps: DSV4IndexerProducerCaps,
    *,
    q_rank: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    query: torch.Tensor,
    head_weights: torch.Tensor,
) -> int:
    for name, tensor in (
        ("q_rank", q_rank),
        ("hidden_states", hidden_states),
        ("positions", positions),
        ("cos_sin_cache", cos_sin_cache),
        ("query", query),
        ("head_weights", head_weights),
    ):
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {caps.device}"
            )
    if q_rank.ndim != 2 or q_rank.shape[1] != caps.q_lora_rank:
        raise ValueError(
            f"q_rank must have shape [tokens,{caps.q_lora_rank}], got {tuple(q_rank.shape)}"
        )
    tokens = int(q_rank.shape[0])
    if tokens <= 0 or tokens > caps.max_tokens:
        raise ValueError(f"tokens must be in [1,{caps.max_tokens}], got {tokens}")
    if q_rank.dtype != torch.bfloat16 or not q_rank.is_contiguous():
        raise ValueError("q_rank must be contiguous BF16")
    if hidden_states.shape != (tokens, caps.hidden):
        raise ValueError(
            f"hidden_states must have shape {(tokens, caps.hidden)}, got "
            f"{tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if positions.shape != (tokens,) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"positions must be int32/int64 [{tokens}]")
    if not positions.is_contiguous():
        raise ValueError("positions must be contiguous")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != caps.rope_dim:
        raise ValueError(
            f"cos_sin_cache must have shape [positions,{caps.rope_dim}], got "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous FP32")
    if query.shape != (tokens, caps.heads, caps.head_dim):
        raise ValueError(
            f"query must have shape {(tokens, caps.heads, caps.head_dim)}, got "
            f"{tuple(query.shape)}"
        )
    if query.dtype != torch.float8_e4m3fn or not query.is_contiguous():
        raise ValueError("query must be contiguous FP8 E4M3")
    if head_weights.shape != (tokens, caps.heads):
        raise ValueError(
            f"head_weights must have shape {(tokens, caps.heads)}, got "
            f"{tuple(head_weights.shape)}"
        )
    if head_weights.dtype != torch.float32 or not head_weights.is_contiguous():
        raise ValueError("head_weights must be contiguous FP32")
    return tokens


def _validate_runtime_tensors(
    caps: DSV4ProducerCaps,
    *,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    main_slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    main_kv_cache: torch.Tensor,
    query: torch.Tensor,
    eps: float,
) -> int:
    tokens = _validate_kv_runtime_tensors(
        caps,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        eps=eps,
    )
    _check_cuda_tensor("query", query)
    if query.device != caps.device:
        raise ValueError(f"query device {query.device} does not match {caps.device}")
    if query.shape != (tokens, caps.heads, caps.head_dim):
        raise ValueError(
            f"query must have shape {(tokens, caps.heads, caps.head_dim)}, got {tuple(query.shape)}"
        )
    if query.dtype != torch.bfloat16 or not query.is_contiguous():
        raise ValueError("query must be contiguous BF16")
    return tokens


def _validate_kv_runtime_tensors(
    caps: DSV4ProducerCaps,
    *,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    main_slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    main_kv_cache: torch.Tensor,
    eps: float,
) -> int:
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("positions", positions),
        ("main_slots", main_slots),
        ("cos_sin_cache", cos_sin_cache),
        ("main_kv_cache", main_kv_cache),
    ):
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {caps.device}"
            )
    if hidden_states.ndim != 2 or hidden_states.shape[1] != caps.hidden:
        raise ValueError(
            f"hidden_states must have shape [tokens,{caps.hidden}], got {tuple(hidden_states.shape)}"
        )
    tokens = int(hidden_states.shape[0])
    if tokens <= 0 or tokens > caps.max_tokens:
        raise ValueError(f"tokens must be in [1,{caps.max_tokens}], got {tokens}")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if positions.shape != (tokens,) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"positions must be int32/int64 [{tokens}]")
    if main_slots.shape != (tokens,) or main_slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"main_slots must be int32/int64 [{tokens}]")
    if not positions.is_contiguous() or not main_slots.is_contiguous():
        raise ValueError("positions and main_slots must be contiguous")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != caps.rope_dim:
        raise ValueError(
            f"cos_sin_cache must have shape [positions,{caps.rope_dim}], got "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous FP32")
    if (
        main_kv_cache.ndim != 2
        or main_kv_cache.shape[1] != caps.main_page_bytes
        or main_kv_cache.dtype != torch.uint8
        or not main_kv_cache.is_contiguous()
    ):
        raise ValueError(
            "main_kv_cache must be contiguous uint8 "
            f"[pages,{caps.main_page_bytes}] for {caps.cache_format}"
        )
    if int(main_kv_cache.shape[0]) <= 0:
        raise ValueError("main_kv_cache must contain at least one physical page")
    if not math.isfinite(float(eps)) or not float(eps) > 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    return tokens


@triton.jit
def _float_to_e2m1(value):
    magnitude = tl.abs(value)
    code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude < 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude < 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(
                            magnitude < 3.5,
                            5,
                            tl.where(magnitude <= 5.0, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    return code.to(tl.uint8) | ((value < 0).to(tl.uint8) << 3)


@triton.jit
def _normalize_rank_pack_kv_kernel(
    qkv,
    q_norm,
    kv_norm,
    positions,
    slots,
    cos_sin,
    q_rank_out,
    cache_fp8,
    cache_bf16,
    cache_u8,
    qkv_stride_t,
    q_rank_stride_t,
    cos_sin_stride_pos,
    eps,
    Q_RANK: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NVFP4: tl.constexpr,
):
    token = tl.program_id(0)

    if Q_RANK > 0:
        qd = tl.arange(0, Q_BLOCK)
        qmask = qd < Q_RANK
        qv = tl.load(qkv + token * qkv_stride_t + qd, mask=qmask, other=0.0).to(
            tl.float32
        )
        q_inv = tl.rsqrt(tl.sum(qv * qv, axis=0) / Q_RANK + eps)
        qw = tl.load(q_norm + qd, mask=qmask, other=0.0).to(tl.float32)
        tl.store(
            q_rank_out + token * q_rank_stride_t + qd,
            (qv * q_inv * qw).to(tl.bfloat16),
            mask=qmask,
        )

    kd = tl.arange(0, HEAD_DIM)
    kv = tl.load(qkv + token * qkv_stride_t + Q_RANK + kd).to(tl.float32)
    k_inv = tl.rsqrt(tl.sum(kv * kv, axis=0) / HEAD_DIM + eps)

    slot = tl.load(slots + token).to(tl.int64)
    page = slot // PAGE_SIZE
    row = slot - page * PAGE_SIZE
    page_base = page * PAGE_BYTES

    if NVFP4:
        # Quantize the complete rotated latent because V consumes all 512
        # dimensions; K reads 0:448 plus the separately stored BF16 RoPE tail.
        dim = tl.arange(0, HEAD_DIM)
        raw = tl.load(qkv + token * qkv_stride_t + Q_RANK + dim).to(tl.float32)
        weight = tl.load(kv_norm + dim).to(tl.float32)
        normalized = (raw * k_inv * weight).to(tl.bfloat16).to(tl.float32)
        rope_mask = dim >= NOPE_DIM
        rope_dim = dim - NOPE_DIM
        partner_dim = NOPE_DIM + (rope_dim ^ 1)
        partner_raw = tl.load(
            qkv + token * qkv_stride_t + Q_RANK + partner_dim,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(kv_norm + partner_dim, mask=rope_mask, other=0.0).to(
            tl.float32
        )
        partner = (partner_raw * k_inv * partner_weight).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(positions + token)
        pair = tl.maximum(rope_dim >> 1, 0)
        cs = cos_sin + pos * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            (rope_dim & 1) == 0,
            normalized * cos_v - partner * sin_v,
            normalized * cos_v + partner * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized).to(tl.bfloat16).to(tl.float32)
        data_base = page_base + row * 432
        quant_3d = tl.reshape(output, (32, 8, 2))
        lo, hi = tl.split(quant_3d)
        max_abs = tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi), axis=1))
        raw_scale = tl.maximum(max_abs * (1.0 / 6.0), 1.1754943508222875e-38)
        scale_fp8 = raw_scale.to(tl.float8e4nv)
        decoded_scale = scale_fp8.to(tl.float32)
        inv_scale = tl.reshape(1.0 / decoded_scale, (32, 1))
        packed = _float_to_e2m1(lo * inv_scale) | (_float_to_e2m1(hi * inv_scale) << 4)
        tl.store(cache_u8 + data_base + tl.arange(0, 256), tl.reshape(packed, (256,)))
        tl.store(
            cache_u8 + data_base + 256 + tl.arange(0, 32),
            scale_fp8.to(tl.uint8, bitcast=True),
        )
        tl.store(cache_u8 + data_base + 288 + tl.arange(0, 16), 0)
        rope_local = dim - NOPE_DIM
        tl.store(
            cache_bf16 + (data_base + 304) // 2 + rope_local,
            output.to(tl.bfloat16),
            mask=rope_mask,
        )
    else:
        data_base = page_base + row * PAYLOAD_BYTES
        scale_base = page_base + PAGE_SIZE * PAYLOAD_BYTES + row * SCALE_BYTES
        # Store seven independently scaled 64-value NoPE groups. The integer
        # exponent path implements pow2-ceil without an approximate log2 boundary.
        gd = tl.arange(0, 64)
        for group in range(NOPE_DIM // 64):
            dim = group * 64 + gd
            values = tl.load(qkv + token * qkv_stride_t + Q_RANK + dim).to(tl.float32)
            weights = tl.load(kv_norm + dim).to(tl.float32)
            values = (values * k_inv * weights).to(tl.bfloat16).to(tl.float32)
            max_abs = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
            raw_scale = max_abs / FP8_MAX
            bits = raw_scale.to(tl.uint32, bitcast=True)
            mantissa = bits & 0x007FFFFF
            rounded = (bits + 0x00800000) & 0x7F800000
            scale_bits = tl.where(mantissa != 0, rounded, bits & 0x7F800000)
            scale = scale_bits.to(tl.float32, bitcast=True)
            quant = tl.maximum(tl.minimum(values / scale, FP8_MAX), -FP8_MAX)
            tl.store(cache_fp8 + data_base + dim, quant.to(tl.float8e4nv))
            tl.store(cache_u8 + scale_base + group, (scale_bits >> 23).to(tl.uint8))

        # Store the rotated BF16 lane directly behind the NoPE payload.
        rd = tl.arange(0, ROPE_DIM)
        rope_raw = tl.load(qkv + token * qkv_stride_t + Q_RANK + NOPE_DIM + rd).to(
            tl.float32
        )
        rope_weight = tl.load(kv_norm + NOPE_DIM + rd).to(tl.float32)
        rope = (rope_raw * k_inv * rope_weight).to(tl.bfloat16).to(tl.float32)
        partner_d = rd ^ 1
        partner_raw = tl.load(
            qkv + token * qkv_stride_t + Q_RANK + NOPE_DIM + partner_d
        ).to(tl.float32)
        partner_weight = tl.load(kv_norm + NOPE_DIM + partner_d).to(tl.float32)
        partner = (partner_raw * k_inv * partner_weight).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(positions + token)
        pair = rd >> 1
        cs = cos_sin + pos * cos_sin_stride_pos
        cos_v = tl.load(cs + pair)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair)
        rotated = tl.where(
            (rd & 1) == 0,
            rope * cos_v - partner * sin_v,
            rope * cos_v + partner * sin_v,
        )
        rope_bf16_offset = (data_base + NOPE_DIM) // 2
        tl.store(
            cache_bf16 + rope_bf16_offset + rd,
            rotated.to(tl.bfloat16),
        )
        tl.store(cache_u8 + scale_base + 7, 0)


@triton.jit
def _normalize_query_rope_kernel(
    query,
    positions,
    cos_sin,
    query_stride_t,
    query_stride_h,
    cos_sin_stride_pos,
    eps,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    base = query + token * query_stride_t + head * query_stride_h
    values = tl.load(base + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + eps)
    normalized = (values * inv).to(tl.bfloat16).to(tl.float32)
    rd = d - NOPE_DIM
    rope_mask = d >= NOPE_DIM
    partner_d = NOPE_DIM + (rd ^ 1)
    partner = tl.load(base + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
    partner = (partner * inv).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(positions + token)
    pair = tl.maximum(rd >> 1, 0)
    cs = cos_sin + pos * cos_sin_stride_pos
    cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
    sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
    rotated = tl.where(
        (rd & 1) == 0,
        normalized * cos_v - partner * sin_v,
        normalized * cos_v + partner * sin_v,
    )
    output = tl.where(rope_mask, rotated, normalized)
    tl.store(base + d, output.to(tl.bfloat16))


def _run_normalize_rank_pack_kv(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_rank_out: torch.Tensor,
    main_kv_cache: torch.Tensor,
    *,
    eps: float,
    cache_format: str = "fp8",
) -> None:
    tokens = int(qkv.shape[0])
    q_rank = int(q_rank_out.shape[1])
    _normalize_rank_pack_kv_kernel[(tokens,)](
        qkv,
        q_norm,
        kv_norm,
        positions,
        slots,
        cos_sin_cache,
        q_rank_out,
        main_kv_cache.view(torch.float8_e4m3fn),
        main_kv_cache.view(torch.bfloat16),
        main_kv_cache,
        qkv.stride(0),
        q_rank_out.stride(0),
        cos_sin_cache.stride(0),
        float(eps),
        Q_RANK=q_rank,
        Q_BLOCK=triton.next_power_of_2(q_rank),
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_SIZE=DSV4_KV_PAGE_SIZE,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=(
            DSV4_NVFP4_PAGE_BYTES if cache_format == "nvfp4" else DSV4_KV_PAGE_BYTES
        ),
        FP8_MAX=DSV4_FP8_MAX,
        NVFP4=cache_format == "nvfp4",
        num_warps=8,
    )


def _run_normalize_pack_kv(
    kv: torch.Tensor,
    kv_norm: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    main_kv_cache: torch.Tensor,
    *,
    eps: float,
    cache_format: str = "fp8",
) -> None:
    tokens = int(kv.shape[0])
    _normalize_rank_pack_kv_kernel[(tokens,)](
        kv,
        kv_norm,
        kv_norm,
        positions,
        slots,
        cos_sin_cache,
        kv,
        main_kv_cache.view(torch.float8_e4m3fn),
        main_kv_cache.view(torch.bfloat16),
        main_kv_cache,
        kv.stride(0),
        kv.stride(0),
        cos_sin_cache.stride(0),
        float(eps),
        Q_RANK=0,
        Q_BLOCK=1,
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_SIZE=DSV4_KV_PAGE_SIZE,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=(
            DSV4_NVFP4_PAGE_BYTES if cache_format == "nvfp4" else DSV4_KV_PAGE_BYTES
        ),
        FP8_MAX=DSV4_FP8_MAX,
        NVFP4=cache_format == "nvfp4",
        num_warps=8,
    )


def _run_normalize_query_rope(
    query: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float,
) -> None:
    tokens, heads, _ = map(int, query.shape)
    _normalize_query_rope_kernel[(tokens, heads)](
        query,
        positions,
        cos_sin_cache,
        query.stride(0),
        query.stride(1),
        cos_sin_cache.stride(0),
        float(eps),
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        num_warps=8,
    )


@triton.jit
def _index_fwht_stage(values, d, WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr):
    shaped = tl.reshape(values, (HEAD_DIM // (2 * WIDTH), 2, WIDTH))
    partner = tl.reshape(tl.flip(shaped, 1), (HEAD_DIM,))
    return tl.where((d & WIDTH) == 0, values + partner, partner - values)


@triton.jit
def _indexer_query_post_kernel(
    raw_query,
    raw_weights,
    positions,
    cos_sin,
    query_fp8,
    head_weights,
    raw_query_stride_t,
    raw_query_stride_h,
    raw_weights_stride_t,
    cos_sin_stride_pos,
    query_stride_t,
    query_stride_h,
    weights_stride_t,
    weight_scale,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    source = raw_query + token * raw_query_stride_t + head * raw_query_stride_h
    values = tl.load(source + d).to(tl.float32)

    rope_mask = d >= NOPE_DIM
    rope_d = d - NOPE_DIM
    partner_d = NOPE_DIM + (rope_d ^ 1)
    partner = tl.load(source + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
    position = tl.load(positions + token)
    pair = tl.maximum(rope_d >> 1, 0)
    cs = cos_sin + position * cos_sin_stride_pos
    cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
    sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
    rotated = tl.where(
        (rope_d & 1) == 0,
        values * cos_v - partner * sin_v,
        values * cos_v + partner * sin_v,
    )
    # apply_rotary_emb writes BF16 before the checkpoint's Hadamard transform.
    hadamard = tl.where(rope_mask, rotated, values).to(tl.bfloat16).to(tl.float32)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=1, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=2, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=4, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=8, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=16, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=32, HEAD_DIM=HEAD_DIM)
    hadamard = _index_fwht_stage(hadamard, d, WIDTH=64, HEAD_DIM=HEAD_DIM)
    hadamard = (hadamard * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)

    # Match fp4_act_quant(..., block_size=32, inplace=True): power-of-two
    # E2M1 scale, nearest representable value, and BF16 dequantized output.
    blocks = tl.reshape(tl.abs(hadamard), (HEAD_DIM // 32, 32))
    block_max = tl.max(blocks, axis=1)
    raw_scale = tl.maximum(block_max, 6.0 * 1.1754943508222875e-38) / 6.0
    scale_bits = raw_scale.to(tl.uint32, bitcast=True)
    mantissa = scale_bits & 0x007FFFFF
    rounded_bits = (scale_bits + 0x00800000) & 0x7F800000
    fp4_scale = tl.where(mantissa != 0, rounded_bits, scale_bits & 0x7F800000).to(
        tl.float32, bitcast=True
    )
    fp4_scale = tl.reshape(
        tl.broadcast_to(tl.expand_dims(fp4_scale, 1), (HEAD_DIM // 32, 32)),
        (HEAD_DIM,),
    )
    magnitude = tl.minimum(tl.abs(hadamard) / fp4_scale, 6.0)
    fp4 = tl.where(
        magnitude < 0.25,
        0.0,
        tl.where(
            magnitude < 0.75,
            0.5,
            tl.where(
                magnitude < 1.25,
                1.0,
                tl.where(
                    magnitude < 1.75,
                    1.5,
                    tl.where(
                        magnitude < 2.5,
                        2.0,
                        tl.where(
                            magnitude < 3.5,
                            3.0,
                            tl.where(magnitude < 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    fp4 = tl.where(hadamard < 0.0, -fp4, fp4) * fp4_scale
    fp4 = fp4.to(tl.bfloat16).to(tl.float32)
    target = query_fp8 + token * query_stride_t + head * query_stride_h
    tl.store(target + d, fp4.to(tl.float8e4nv))

    raw_weight = tl.load(raw_weights + token * raw_weights_stride_t + head).to(
        tl.float32
    )
    # PyTorch's BF16 linear output multiplied by the Python scale remains
    # BF16 in the checkpoint reference; expose that rounded value as FP32 to
    # the production scorer.
    scaled_weight = (raw_weight * weight_scale).to(tl.bfloat16).to(tl.float32)
    tl.store(head_weights + token * weights_stride_t + head, scaled_weight)


def _run_indexer_query_post(
    raw_query: torch.Tensor,
    raw_weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    query: torch.Tensor,
    head_weights: torch.Tensor,
) -> None:
    tokens, heads, _ = map(int, query.shape)
    _indexer_query_post_kernel[(tokens, heads)](
        raw_query,
        raw_weights,
        positions,
        cos_sin_cache,
        query,
        head_weights,
        raw_query.stride(0),
        raw_query.stride(1),
        raw_weights.stride(0),
        cos_sin_cache.stride(0),
        query.stride(0),
        query.stride(1),
        head_weights.stride(0),
        DSV4_INDEX_WEIGHT_SCALE,
        HEAD_DIM=DSV4_INDEX_HEAD_DIM,
        NOPE_DIM=DSV4_INDEX_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        num_warps=4,
    )


def run_dsv4_producer(*, binding: DSV4ProducerBinding) -> torch.Tensor:
    """Run the allocation-free DSV4 query/main-KV producer on the current stream."""

    if not isinstance(binding, DSV4ProducerBinding):
        raise TypeError("run_dsv4_producer requires a DSV4ProducerBinding")
    block_fp8_linear_mxfp8(binding=binding.qkv_linear)
    _run_normalize_rank_pack_kv(
        binding.qkv_output[:, :, 0],
        binding.weights.q_norm,
        binding.weights.kv_norm,
        binding.positions,
        binding.main_slots,
        binding.cos_sin_cache,
        binding.q_rank,
        binding.main_kv_cache,
        eps=binding.eps,
        cache_format=binding.cache_format,
    )
    block_fp8_linear_mxfp8(binding=binding.q_linear)
    _run_normalize_query_rope(
        binding.query,
        binding.positions,
        binding.cos_sin_cache,
        eps=binding.eps,
    )
    return binding.query


def run_dsv4_kv_producer(*, binding: DSV4KVProducerBinding) -> torch.Tensor:
    """Project and pack dSpark target-main KV without producing a query."""

    if not isinstance(binding, DSV4KVProducerBinding):
        raise TypeError("run_dsv4_kv_producer requires a DSV4KVProducerBinding")
    block_fp8_linear_mxfp8(binding=binding.kv_linear)
    kv = binding.kv_output[:, :, 0]
    _run_normalize_pack_kv(
        kv,
        binding.weights.kv_norm,
        binding.positions,
        binding.main_slots,
        binding.cos_sin_cache,
        binding.main_kv_cache,
        eps=binding.eps,
        cache_format=binding.cache_format,
    )
    return kv


def run_dsv4_indexer_producer(
    *, binding: DSV4IndexerProducerBinding
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce checkpoint-native C4 index queries and learned head weights."""

    if not isinstance(binding, DSV4IndexerProducerBinding):
        raise TypeError(
            "run_dsv4_indexer_producer requires a DSV4IndexerProducerBinding"
        )
    block_fp8_linear_mxfp8(binding=binding.q_linear)
    torch.mm(
        binding.hidden_states,
        binding.weights.weights_projection_t,
        out=binding.weights_output,
    )
    _run_indexer_query_post(
        binding.q_output[:, :, 0].view(
            int(binding.q_output.shape[0]), DSV4_INDEX_HEADS, DSV4_INDEX_HEAD_DIM
        ),
        binding.weights_output,
        binding.positions,
        binding.cos_sin_cache,
        binding.query,
        binding.head_weights,
    )
    return binding.query, binding.head_weights


__all__ = [
    "DSV4IndexerProducerBinding",
    "DSV4IndexerProducerCaps",
    "DSV4IndexerProducerPlan",
    "DSV4IndexerProducerWeights",
    "DSV4ProducerBinding",
    "DSV4ProducerCaps",
    "DSV4KVProducerBinding",
    "DSV4KVProducerWeights",
    "DSV4KVProducerPlan",
    "DSV4ProducerPlan",
    "DSV4ProducerWeights",
    "pack_dsv4_indexer_producer_weights",
    "pack_dsv4_producer_weights",
    "pack_dsv4_kv_producer_weights",
    "plan_dsv4_indexer_producer",
    "plan_dsv4_kv_producer",
    "plan_dsv4_producer",
    "run_dsv4_indexer_producer",
    "run_dsv4_kv_producer",
    "run_dsv4_producer",
]
