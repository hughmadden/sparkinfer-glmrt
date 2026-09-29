"""Shared helpers for the cuteafd MiMo V2 Flash AOT program tests.

Weights come from the MiMo-V2-Flash checkpoint (FP8 blocks times their FP32
scales, BF16, as the programs take them; the full-attention ``k_proj`` scale
grid is per KV head: a 128-row then a 64-row block per 192-row head).
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

SNAPSHOT = Path(os.environ.get(
    "CUTEAFD_MIMO_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--XiaomiMiMo--MiMo-V2-Flash/snapshots/1afd314a2406c282e0956375c34a676501c78649",
))
GOLDEN = Path(os.environ.get("CUTEAFD_MIMO_GOLDEN", "/golden"))
HIDDEN = 4096


@lru_cache(maxsize=1)
def _weight_map() -> dict[str, str] | None:
    index = SNAPSHOT / "model.safetensors.index.json"
    if not index.is_file():
        return None
    return json.loads(index.read_text())["weight_map"]


_HANDLES: dict[str, object] = {}


def raw_tensor(name: str) -> torch.Tensor:
    weights = _weight_map()
    if weights is None:
        pytest.skip(f"MiMo-V2-Flash snapshot not mounted at {SNAPSHOT}")
    from safetensors import safe_open

    shard = weights[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(str(SNAPSHOT / shard), framework="pt", device="cpu")
    return _HANDLES[shard].get_tensor(name)


def scale_row_of(rows: int, scale_rows: int, device="cuda", head_dim: int = 192) -> torch.Tensor:
    """Scale-grid row of every weight row (128-row blocks, or per 192-row head)."""
    r = torch.arange(rows, device=device)
    if -(-rows // 128) == scale_rows:
        return r // 128
    per_head = -(-head_dim // 128)
    assert rows % head_dim == 0 and scale_rows == rows // head_dim * per_head, (rows, scale_rows)
    return r // head_dim * per_head + r % head_dim // 128


def tensor(name: str, device="cuda") -> torch.Tensor:
    """A checkpoint tensor as the programs take it: FP8 blocks dequantized to BF16."""
    value = raw_tensor(name).to(device)
    if value.dtype != torch.float8_e4m3fn:
        return value
    scale = raw_tensor(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
    rows, cols = value.shape
    grown = scale[scale_row_of(rows, scale.shape[0], device)].repeat_interleave(128, 1)[:, :cols]
    return (value.float() * grown).bfloat16()


@lru_cache(maxsize=1)
def config():
    if _weight_map() is None:
        pytest.skip(f"MiMo-V2-Flash snapshot not mounted at {SNAPSHOT}")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SNAPSHOT)
    cfg._attn_implementation = "eager"
    return cfg


def _checkpoint_name(key: str) -> str:
    return {"sinks": "attention_sink_bias"}.get(key, key)


def reference_module(cls_name: str, *args, prefix: str, device="cuda", fp32: tuple[str, ...] = ()):
    """Build a transformers MiMo module in BF16 and load its checkpoint tensors
    (parameters named in ``fp32`` stay FP32, as the checkpoint stores them)."""
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
            param.copy_(tensor(prefix + _checkpoint_name(key), device).to(param.dtype))
        for key, buffer in module.named_buffers():
            name = prefix + key
            if name in (_weight_map() or {}):
                buffer.copy_(tensor(name, device).to(buffer.dtype))
    return module.eval()


def layer_kind(layer: int) -> str:
    return "swa" if config().layer_types[layer] == "sliding_attention" else "full"


def cos_sin(positions: int, kind: str, device="cuda") -> torch.Tensor:
    """FP32 [positions, 64]: cos then sin of the 32 NeoX frequencies theta^(-2i/64)."""
    theta = 5.0e6 if kind == "full" else 1.0e4
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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """transformers MiMoV2FlashRMSNorm."""
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
