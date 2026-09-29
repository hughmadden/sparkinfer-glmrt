"""cuteafd GLM 5.x sparse MLA AOT programs vs the latent-record reference and
the transformers (non-absorbed, BF16) attention.

The cache and queries come from ``glm_producer`` over 4096 layer-0 rows
(the golden prompt, then perturbed repeats). Each query row attends a
2048-wide selection: every causal row while the position is below 2048,
otherwise a random 2048-subset of its causal prefix (a top-k stand-in).
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import config, cos_sin, cosine, golden_rows, reference_module, reference_rope, rms_norm, tensor
from .test_cuteafd_glm_dense_aot import producer_weights, rope_interleaved

T = 4096
_PROGRAMS: dict = {}


def _program(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_attention, glm_sparse_mla

        fn = {"producer": glm_attention.compile_glm_producer_aot,
              "mla": glm_sparse_mla.compile_glm_sparse_mla_aot}[kind]
        _PROGRAMS[key] = fn(GLM53, **kw)
    return _PROGRAMS[key]


@pytest.fixture(scope="module")
def world():
    """Layer-0 attention module, x, query, cache and slot map for T rows."""
    require_b12x()
    attn = reference_module("GlmMoeDsaAttention", config(), 0, prefix="model.layers.0.self_attn.")
    w = producer_weights(attn)
    x = rms_norm(golden_rows(0, T), tensor("model.layers.0.input_layernorm.weight"))
    positions = torch.arange(T, dtype=torch.int64, device="cuda")
    gen = torch.Generator(device="cpu").manual_seed(5)
    pages = T // 64 + 7
    page_of = torch.randperm(pages, generator=gen)[:T // 64].cuda()
    slots = page_of.repeat_interleave(64) * 64 + torch.arange(T, device="cuda") % 64
    cache = torch.zeros((pages, 64 * 656), dtype=torch.uint8, device="cuda")
    query = torch.empty((T, 64, 576), dtype=torch.bfloat16, device="cuda")
    q_resid = torch.empty((T, 2048), dtype=torch.bfloat16, device="cuda")
    producer = _program("producer", max_rows=T)
    scratch = torch.empty(producer.scratch_bytes(T)["scratch"], dtype=torch.uint8, device="cuda")
    producer.launch(x, positions, slots, cos_sin(T), w["w_qkv_a"], w["q_a_norm"], w["kv_a_norm"],
                    w["w_q_b"], w["w_uk"], cache, query, q_resid, scratch, scalars=(T,))
    torch.cuda.synchronize()
    return dict(attn=attn, w=w, x=x, positions=positions, slots=slots, cache=cache, query=query)


def selections(positions: torch.Tensor, slots: torch.Tensor, seed: int):
    """(indices [rows, 2048] physical slots -1 padded, lengths [rows], logical [rows, 2048])."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = positions.numel()
    logical = torch.full((rows, 2048), -1, dtype=torch.int64)
    lengths = torch.empty(rows, dtype=torch.int32)
    for i, p in enumerate(positions.tolist()):
        n = p + 1
        if n <= 2048:
            logical[i, :n] = torch.arange(n)
            lengths[i] = n
        else:
            logical[i] = torch.randperm(n, generator=gen)[:2048].sort().values
            lengths[i] = 2048
    logical = logical.cuda()
    indices = torch.where(logical >= 0, slots[logical.clamp_min(0)], -1).to(torch.int32)
    return indices.contiguous(), lengths.cuda(), logical


def transformers_attention(world, rows_pos: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
    """Eager GLM attention (BF16 un-absorbed q/k/v) over each row's selection,
    returned as the per-head value output [rows, 64, 256] (before o_proj)."""
    attn, x = world["attn"], world["x"]
    with torch.no_grad():
        cos, sin = reference_rope(torch.arange(T, device="cuda"))
        q_resid = attn.q_a_layernorm(attn.q_a_proj(x[rows_pos]))
        q = attn.q_b_proj(q_resid).view(-1, 64, 256)
        q_rot = rope_interleaved(q[..., 192:].transpose(0, 1), cos[0, rows_pos], sin[0, rows_pos]).transpose(0, 1)
        query = torch.cat([q[..., :192], q_rot], -1)
        kv = attn.kv_a_proj_with_mqa(x)
        kvb = attn.kv_b_proj(attn.kv_a_layernorm(kv[:, :512])).view(T, 64, 448)
        k_rot = rope_interleaved(kv[:, 512:], cos[0], sin[0])
        keys = torch.cat([kvb[..., :192], k_rot[:, None, :].expand(T, 64, 64)], -1)
        values = kvb[..., 192:]
        out = torch.empty((len(rows_pos), 64, 256), dtype=torch.bfloat16, device="cuda")
        for i in range(len(rows_pos)):
            sel = logical[i][logical[i] >= 0]
            k, v = keys[sel].transpose(0, 1), values[sel].transpose(0, 1)  # [64, n, d]
            scores = torch.matmul(query[i][:, None, :], k.transpose(1, 2)) * attn.scaling
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            out[i] = torch.matmul(probs, v)[:, 0]
    return out


def _check(world, route, capacity, rows_pos, seed):
    from b12x.attention._shared.mla.reference import sparse_mla_reference

    program = _program("mla", route=route, max_rows=capacity)
    rows = rows_pos.numel()
    indices, lengths, logical = selections(rows_pos, world["slots"], seed)
    q = world["query"][rows_pos].contiguous()
    out = torch.empty((rows, 64, 512), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(q, world["cache"], indices, lengths, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    records = world["cache"].view(-1, 1, 656)
    expected = sparse_mla_reference(q_all=q, kv_cache=records, page_table_1=indices,
                                    active_token_counts=lengths, sm_scale=256 ** -0.5, v_head_dim=512)
    c_contract = cosine(out, expected)
    w_uv = world["w"]["w_uv"].float()
    values = torch.einsum("thc,hvc->thv", out.float(), w_uv)
    ref_values = transformers_attention(world, rows_pos, logical)
    c_model = cosine(values, ref_values)
    print(f"glm_sparse_mla {route} m{capacity} rows={rows}: vs FP8-record reference {c_contract:.7f}; "
          f"after W_UV vs transformers BF16 attention {c_model:.6f}")
    assert c_contract >= 0.9999
    assert c_model >= 0.999
    return program


@pytest.mark.parametrize("rows", [1, 3, 16, 64])
def test_glm_sparse_mla_decode(world, rows):
    gen = torch.Generator(device="cpu").manual_seed(rows)
    rows_pos = torch.randint(0, T, (rows,), generator=gen)
    rows_pos[0] = T - 1
    _check(world, "decode", 64, rows_pos.cuda(), rows)


def test_glm_sparse_mla_prefill(world):
    _check(world, "prefill", T, torch.arange(T, device="cuda"), 99)
