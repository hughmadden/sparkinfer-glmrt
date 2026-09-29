"""Shared helpers for the cuteafd GLM 5.x AOT program tests.

Weights come from the GLM 5.3 checkpoint (FP8 128x128 blocks times their
FP32 scales, BF16, as the programs take them); references are the
transformers ``modeling_glm_moe_dsa`` modules. Activations are the golden
harness outputs when ``CUTEAFD_GLM_GOLDEN`` points at them (layerNN.bin
BF16 [T, 6144]), otherwise normalized Gaussian rows.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

SNAPSHOT = Path(os.environ.get(
    "CUTEAFD_GLM_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--zai-org--GLM-5.3/snapshots/935644c05e76fc198714f4cca449fd8b970ff6d7",
))
GOLDEN = Path(os.environ.get("CUTEAFD_GLM_GOLDEN", "/golden"))


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
        pytest.skip(f"GLM 5.3 snapshot not mounted at {SNAPSHOT}")
    from safetensors import safe_open

    shard = weights[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(str(SNAPSHOT / shard), framework="pt", device="cpu")
    return _HANDLES[shard].get_tensor(name)


def tensor(name: str, device="cuda") -> torch.Tensor:
    """A checkpoint tensor as the programs take it: FP8 blocks dequantized to BF16."""
    value = raw_tensor(name).to(device)
    if value.dtype != torch.float8_e4m3fn:
        return value
    scale = raw_tensor(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
    rows, cols = value.shape
    grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (value.float() * grown).bfloat16()


@lru_cache(maxsize=1)
def config():
    weights = _weight_map()
    if weights is None:
        pytest.skip(f"GLM 5.3 snapshot not mounted at {SNAPSHOT}")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SNAPSHOT)
    cfg._attn_implementation = "eager"
    return cfg


def reference_module(cls_name: str, *args, prefix: str, device="cuda"):
    """Build a transformers GLM module in BF16 and load its checkpoint tensors."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            module = getattr(ref, cls_name)(*args)
    finally:
        torch.set_default_dtype(torch.float32)
    module = module.to_empty(device=device)
    with torch.no_grad():
        for key, param in module.named_parameters():
            if ".experts." in f".{key}":
                continue
            param.copy_(tensor(prefix + key, device).to(param.dtype))
        for key, buffer in module.named_buffers():
            name = prefix + key
            if name in (_weight_map() or {}):
                buffer.copy_(tensor(name, device).to(buffer.dtype))
    return module.eval()


def cos_sin(positions: int, device="cuda", theta: float | None = None) -> torch.Tensor:
    """FP32 [positions, 64]: cos then sin of the 32 interleaved-RoPE frequencies."""
    theta = float(config().rope_parameters["rope_theta"]) if theta is None else theta
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous().to(device)


def reference_rope(positions: torch.Tensor, dtype=torch.bfloat16):
    """(cos, sin) exactly as GlmMoeDsaRotaryEmbedding returns them, [1, T, 64]."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

    rotary = ref.GlmMoeDsaRotaryEmbedding(config=config()).to(positions.device)
    probe = torch.empty((1,), dtype=dtype, device=positions.device)
    return rotary(probe, position_ids=positions[None])


def golden_rows(layer: int, rows: int, device="cuda", seed: int = 0) -> torch.Tensor:
    """``rows`` BF16 hidden rows: golden layer outputs (tiled with a small
    perturbation past T) or unit Gaussian rows scaled like a residual."""
    path = GOLDEN / f"layer{layer:02d}.bin"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if path.is_file():
        data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).view(-1, 6144)
        reps = -(-rows // data.shape[0])
        base = data.repeat(reps, 1)[:rows].float()
        if rows > data.shape[0]:
            base[data.shape[0]:] += 0.01 * base[data.shape[0]:].std() * torch.randn(
                (rows - data.shape[0], 6144), generator=gen)
        return base.bfloat16().to(device)
    return (torch.randn((rows, 6144), generator=gen) * 0.5).bfloat16().to(device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """transformers GlmMoeDsaRMSNorm."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def unpack_latent_records(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Dequantized FP32 [len(slots), 576] from 656-byte GLM latent records."""
    from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference

    records = cache.view(-1, 656)[slots.long()]
    return unpack_mla_kv_cache_reference(records.unsqueeze(1)).squeeze(1)


def fp8_rows(names: list[str], device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Row-concatenated checkpoint E4M3 weights and their FP32 block-scale grids."""
    weights, scales = [], []
    for name in names:
        weights.append(raw_tensor(name).to(device))
        scales.append(raw_tensor(name.removesuffix("weight") + "weight_scale_inv").to(device).float())
        assert weights[-1].dtype == torch.float8_e4m3fn
    for w in weights[:-1]:
        assert w.shape[0] % 128 == 0, "only the last concatenated weight may end mid-block"
    return torch.cat(weights).contiguous(), torch.cat(scales).contiguous()
