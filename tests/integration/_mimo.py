"""Shared helpers for the cuteafd MiMo V2.6 Flash AOT program tests.

Weights come from the MiMo-V2.6-Flash-MOPD checkpoint (FP8 blocks times their
FP32 scales, BF16, as the programs take them). Its attention is one fused
``qkv_proj`` stored as TP-interleaved row shards, each with its own 128x128
block grid over the whole shard; ``fused_qkv`` dequantizes every shard before
splitting it into q/k/v, so the k/v boundary does not restart the grid.
References are the transformers ``modeling_mimo_v2_flash`` modules.
Activations are the golden harness outputs when ``CUTEAFD_MIMO_GOLDEN`` points
at them (layerNN.bin BF16 [T, 4096]), otherwise Gaussian rows.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
GOLDEN = Path(os.environ.get("CUTEAFD_MIMO_GOLDEN", "/golden"))
HIDDEN = 4096


@lru_cache(maxsize=1)
def snapshot() -> Path:
    """The checkpoint snapshot, resolved from the Hugging Face cache (``HF_HOME``)."""
    override = os.environ.get("CUTEAFD_MIMO_SNAPSHOT")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, local_files_only=True))


@lru_cache(maxsize=1)
def _index() -> dict | None:
    try:
        index = snapshot() / "model.safetensors.index.json"
    except Exception:
        return None
    if not index.is_file():
        return None
    return json.loads(index.read_text())


@lru_cache(maxsize=1)
def _weight_map() -> dict[str, str] | None:
    index = _index()
    return None if index is None else index["weight_map"]


@lru_cache(maxsize=1)
def checkpoint_tp() -> int:
    """The checkpoint's tensor-parallel degree: its ``qkv_proj`` row shards."""
    return int((_index() or {}).get("metadata", {}).get("tp_size", 1))


def checkpoint_has(name: str) -> bool:
    """Whether the mounted checkpoint stores ``name``."""
    return name in (_weight_map() or {})


_HANDLES: dict[str, object] = {}


def raw_tensor(name: str) -> torch.Tensor:
    weights = _weight_map()
    if weights is None:
        pytest.skip(f"{REPO} snapshot not mounted under HF_HOME ({os.environ.get('HF_HOME', 'unset')})")
    from safetensors import safe_open

    shard = weights[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(str(snapshot() / shard), framework="pt", device="cpu")
    return _HANDLES[shard].get_tensor(name)


def scale_row_of(rows: int, scale_rows: int, device="cuda", head_dim: int = 192) -> torch.Tensor:
    """Scale-grid row of every weight row (128-row blocks, or per 192-row head)."""
    r = torch.arange(rows, device=device)
    if -(-rows // 128) == scale_rows:
        return r // 128
    per_head = -(-head_dim // 128)
    assert rows % head_dim == 0 and scale_rows == rows // head_dim * per_head, (rows, scale_rows)
    return r // head_dim * per_head + r % head_dim // 128


def fp8_blocks_to_bf16(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[R, C]`` times FP32 ``[ceil(R/128), ceil(C/128)]`` block scales."""
    rows, cols = value.shape
    assert scale.shape == (-(-rows // 128), -(-cols // 128)), (value.shape, scale.shape)
    grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (value.float() * grown).bfloat16()


@lru_cache(maxsize=None)
def fused_qkv(name: str, heads: int, kv_heads: int, head_dim: int, v_head_dim: int, device="cuda"
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """De-interleaved ``(q, k, v)`` BF16 rows of the fused ``qkv_proj``.

    Each checkpoint TP shard stores ``[q | k | v]`` with its own block grid over
    the whole shard: a full-attention shard's grid row 25 spans its k rows
    128-191 and v rows 0-63, so dequantize the shard before splitting it.
    """
    tp = checkpoint_tp()
    value = raw_tensor(name).to(device)
    scale = raw_tensor(name + "_scale_inv").to(device).float()
    q = heads // tp * head_dim
    k = max(1, kv_heads // tp) * head_dim
    v = max(1, kv_heads // tp) * v_head_dim
    rows, blocks = q + k + v, -(-(q + k + v) // 128)
    assert value.shape[0] == rows * tp and scale.shape[0] == blocks * tp, (name, value.shape, scale.shape)
    parts: list[list[torch.Tensor]] = [[], [], []]
    for shard in range(tp):
        dense = fp8_blocks_to_bf16(value[shard * rows:(shard + 1) * rows],
                                   scale[shard * blocks:(shard + 1) * blocks])
        first = 0
        for i, size in enumerate((q, k, v)):
            parts[i].append(dense[first:first + size])
            first += size
    return tuple(torch.cat(part, 0).contiguous() for part in parts)


def fused_qkv_part(prefix: str, key: str, module, device="cuda") -> torch.Tensor | None:
    """The q/k/v slice of a fused checkpoint ``qkv_proj``, when it is stored fused."""
    for index, part in enumerate(("q_proj", "k_proj", "v_proj")):
        if key != f"{part}.weight":
            continue
        name = prefix + "qkv_proj.weight"
        if not checkpoint_has(name):
            return None
        heads = module.q_proj.out_features // module.head_dim
        kv_heads = module.k_proj.out_features // module.head_dim
        return fused_qkv(name, heads, kv_heads, module.head_dim, module.v_head_dim, device)[index]
    return None


def tensor(name: str, device="cuda") -> torch.Tensor:
    """A checkpoint tensor as the programs take it: FP8 blocks dequantized to BF16.

    The fused ``qkv_proj`` has a whole-shard grid; take its q/k/v rows from
    ``fused_qkv`` instead.
    """
    if name.endswith(".qkv_proj.weight"):
        raise ValueError(f"{name}: fused qkv_proj has a per-shard grid; use fused_qkv()")
    value = raw_tensor(name).to(device)
    if value.dtype != torch.float8_e4m3fn:
        return value
    scale = raw_tensor(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
    return fp8_blocks_to_bf16(value, scale)


@lru_cache(maxsize=1)
def checkpoint_config() -> dict:
    """The checkpoint's own ``config.json``."""
    if _weight_map() is None:
        pytest.skip(f"{REPO} snapshot not mounted under HF_HOME ({os.environ.get('HF_HOME', 'unset')})")
    return json.loads((snapshot() / "config.json").read_text())


@lru_cache(maxsize=1)
def config():
    from transformers.models.mimo_v2_flash import MiMoV2FlashConfig

    raw = checkpoint_config()
    # The pinned reference modules are the Flash ones: SWA doubles the KV heads
    # and shares the QK/V head dims. V2.6 stores both explicitly; check they agree.
    assert raw["attention_projection_layout"] == "fused_qkv", raw["attention_projection_layout"]
    assert raw["swa_num_attention_heads"] == raw["num_attention_heads"]
    assert (raw["swa_head_dim"], raw["swa_v_head_dim"]) == (raw["head_dim"], raw["v_head_dim"])
    assert raw["swa_num_key_value_heads"] == 2 * raw["num_key_value_heads"]
    pattern, mlp = raw["hybrid_layer_pattern"], raw["moe_layer_freq"]
    assert len(pattern) == len(mlp) == raw["num_hidden_layers"] and set(pattern) <= {0, 1}
    cfg = MiMoV2FlashConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        v_head_dim=raw["v_head_dim"],
        sliding_window=raw["sliding_window"],
        attention_value_scale=raw["attention_value_scale"],
        rms_norm_eps=raw["layernorm_epsilon"],
        max_position_embeddings=raw["max_position_embeddings"],
        attention_bias=raw["attention_bias"],
        attention_dropout=raw["attention_dropout"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        n_routed_experts=raw["n_routed_experts"],
        routed_scaling_factor=raw["routed_scaling_factor"],
        n_group=raw["n_group"],
        topk_group=raw["topk_group"],
        norm_topk_prob=raw["norm_topk_prob"],
        hidden_act=raw["hidden_act"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        layer_types=["full_attention" if dense == 0 else "sliding_attention" for dense in pattern],
        mlp_layer_types=["dense" if freq == 0 else "sparse" for freq in mlp],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": raw["rope_theta"],
                               "partial_rotary_factor": raw["partial_rotary_factor"]},
            "sliding_attention": {"rope_type": "default", "rope_theta": raw["swa_rope_theta"],
                                  "partial_rotary_factor": raw["partial_rotary_factor"]},
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def _checkpoint_name(key: str) -> str:
    return {"sinks": "attention_sink_bias"}.get(key, key)


def reference_module(cls_name: str, *args, prefix: str, device="cuda", fp32: tuple[str, ...] = ()):
    """Build a transformers MiMo module in BF16 and load its checkpoint tensors
    (parameters named in ``fp32`` stay FP32, as the checkpoint stores them).
    A fused ``qkv_proj`` feeds the module's q/k/v projections through
    ``fused_qkv``."""
    from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash as ref

    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            module = getattr(ref, cls_name)(*args)
    finally:
        torch.set_default_dtype(torch.float32)
    module = module.to_empty(device=device)
    for key in fp32:
        owner, _, leaf = key.rpartition(".")
        target = module.get_submodule(owner) if owner else module
        current = getattr(target, leaf)
        if isinstance(current, torch.nn.Parameter):
            setattr(target, leaf, torch.nn.Parameter(current.float(), requires_grad=False))
        else:
            target.register_buffer(leaf, current.float())
    with torch.no_grad():
        for key, param in module.named_parameters():
            if ".experts." in f".{key}":
                continue
            value = fused_qkv_part(prefix, key, module, device)
            if value is None:
                value = tensor(prefix + _checkpoint_name(key), device)
            param.copy_(value.to(param.dtype))
        for key, buffer in module.named_buffers():
            name = prefix + key
            if checkpoint_has(name):
                buffer.copy_(tensor(name, device).to(buffer.dtype))
    return module.eval()


def layer_kind(layer: int) -> str:
    return "swa" if config().layer_types[layer] == "sliding_attention" else "full"


def cos_sin(positions: int, kind: str, device="cuda", theta: float | None = None) -> torch.Tensor:
    """FP32 [positions, 64]: cos then sin of the 32 NeoX frequencies theta^(-2i/64)."""
    theta = (5.0e6 if kind == "full" else 1.0e4) if theta is None else theta
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.int64).float() / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous().to(device)


def reference_rope(positions: torch.Tensor, kind: str, dtype=torch.bfloat16):
    """(cos, sin) exactly as MiMoV2FlashRotaryEmbedding returns them, [1, T, 64]."""
    from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash as ref

    rotary = ref.MiMoV2FlashRotaryEmbedding(config=config()).to(positions.device)
    probe = torch.empty((1,), dtype=dtype, device=positions.device)
    layer_type = "full_attention" if kind == "full" else "sliding_attention"
    return rotary(probe, positions[None], layer_type=layer_type)


def golden_rows(layer: int, rows: int, device="cuda", seed: int = 0) -> torch.Tensor:
    """``rows`` BF16 hidden rows: the golden output of ``layer`` (tiled with a
    small perturbation past T), or Gaussian rows."""
    path = GOLDEN / f"layer{layer:02d}.bin"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if path.is_file():
        data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).view(-1, HIDDEN)
        reps = -(-rows // data.shape[0])
        base = data.repeat(reps, 1)[:rows].float()
        if rows > data.shape[0]:
            base[data.shape[0]:] += 0.01 * base[data.shape[0]:].std() * torch.randn(
                (rows - data.shape[0], HIDDEN), generator=gen)
        return base.bfloat16().to(device)
    return (torch.randn((rows, HIDDEN), generator=gen) * 0.5).bfloat16().to(device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """transformers MiMoV2FlashRMSNorm (``eps`` defaults to the checkpoint's)."""
    if eps is None:
        eps = config().rms_norm_eps
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def masks(t: int, window: int = 128, device="cuda") -> dict[str, torch.Tensor]:
    """Additive BF16 [1, 1, T, T] masks for the eager reference."""
    q = torch.arange(t, device=device)[:, None]
    k = torch.arange(t, device=device)[None, :]
    low = torch.finfo(torch.bfloat16).min
    causal = k <= q
    make = lambda keep: torch.where(keep, 0.0, low).to(torch.bfloat16)[None, None]  # noqa: E731
    return {"full": make(causal), "swa": make(causal & (k > q - window))}
