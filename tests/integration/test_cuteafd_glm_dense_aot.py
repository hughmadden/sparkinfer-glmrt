"""cuteafd GLM 5.x norm / FFN / attention-projection AOT programs vs transformers.

Real GLM 5.3 weights (layer 0 dense, layer 3 shared expert and router),
golden activations when mounted; decode programs (m64) at 1/16/64 live rows,
prefill programs (m4096) at 4096 rows. Every output is compared with the
transformers ``modeling_glm_moe_dsa`` module it replaces at cosine >= 0.9999.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import (
    config, cos_sin, cosine, golden_rows, reference_module, reference_rope, rms_norm, tensor,
    unpack_latent_records,
)

DECODE, PREFILL = 64, 4096
ROWS = [(DECODE, 1), (DECODE, 16), (DECODE, 64), (PREFILL, 4096)]
COS = 0.9999


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


@pytest.fixture(scope="module")
def g():
    require_b12x()
    from b12x.integration.cuteafd import GLM53

    return GLM53


_PROGRAMS: dict = {}


def _program(kind, *args, **kw):
    key = (kind, args, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_attention, glm_ffn

        fn = {"norm": glm_ffn.compile_glm_norm_aot, "ffn": glm_ffn.compile_glm_ffn_aot,
              "router": glm_ffn.compile_glm_router_scores_aot,
              "producer": glm_attention.compile_glm_producer_aot,
              "index_producer": glm_attention.compile_glm_index_producer_aot,
              "o": glm_attention.compile_glm_o_aot}[kind]
        _PROGRAMS[key] = fn(GLM53, *args, **kw)
    return _PROGRAMS[key]


@pytest.mark.parametrize("deltas", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 64, 4096])
def test_glm_norm(g, rows, deltas):
    program = _program("norm")
    weight = tensor("model.layers.0.post_attention_layernorm.weight")
    residual = golden_rows(1, rows)
    d0, d1 = golden_rows(2, rows, seed=1) * 0.3, golden_rows(3, rows, seed=2) * 0.2
    expected_residual = residual
    if deltas == 1:
        expected_residual = residual + d0
    elif deltas == 2:
        expected_residual = residual + (d0 + d1)
    expected = rms_norm(expected_residual, weight)
    out = torch.empty_like(residual)
    program.launch(residual, d0, d1, weight, out, scalars=(rows, deltas))
    torch.cuda.synchronize()
    assert cosine(residual, expected_residual) >= COS
    assert torch.equal(residual, expected_residual) or deltas == 0 or \
        (residual != expected_residual).float().mean() < 1e-3
    c = cosine(out, expected)
    mismatch = float((out != expected).float().mean())
    print(f"glm_norm rows={rows} deltas={deltas}: cosine {c:.7f} bf16 mismatch {mismatch:.2e}")
    assert c >= COS and mismatch < 1e-2


@pytest.mark.parametrize("layer,inter", [(0, 12288), (3, 2048)])
@pytest.mark.parametrize("capacity,rows", ROWS)
def test_glm_ffn(g, layer, inter, capacity, rows):
    program = _program("ffn", inter=inter, max_rows=capacity)
    cfg = config()
    if inter == 12288:
        mlp = reference_module("GlmMoeDsaMLP", cfg, prefix=f"model.layers.{layer}.mlp.")
    else:
        mlp = reference_module("GlmMoeDsaMLP", cfg, inter, prefix=f"model.layers.{layer}.mlp.shared_experts.")
    x = rms_norm(golden_rows(layer, rows), tensor(f"model.layers.{layer}.post_attention_layernorm.weight"))
    w_gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
    out = torch.empty_like(x)
    program.launch(x, w_gate_up, mlp.down_proj.weight, out, _scratch(program, rows), scalars=(rows,))
    with torch.no_grad():
        expected = mlp(x)
    torch.cuda.synchronize()
    c = cosine(out, expected)
    print(f"glm_ffn I={inter} m{capacity} rows={rows}: cosine {c:.7f}")
    assert c >= COS


@pytest.mark.parametrize("rows", [1, 32, 64, 4096])
def test_glm_router_scores(g, rows):
    program = _program("router")
    gate = reference_module("GlmMoeDsaTopkRouter", config(), prefix="model.layers.3.mlp.gate.")
    x = rms_norm(golden_rows(3, rows), tensor("model.layers.3.post_attention_layernorm.weight"))
    logits = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
    program.launch(x, gate.weight, logits, scalars=(rows,))
    with torch.no_grad():
        expected, _, _ = gate(x)
    torch.cuda.synchronize()
    c = cosine(logits, expected)
    print(f"glm_router_scores rows={rows}: cosine {c:.9f} max abs {float((logits - expected).abs().max()):.2e}")
    assert c >= 0.999999


def _attention(layer):
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref  # noqa: F401

    return reference_module("GlmMoeDsaAttention", config(), layer, prefix=f"model.layers.{layer}.self_attn.")


_ATTENTION: dict = {}


def _attn(layer):
    if layer not in _ATTENTION:
        _ATTENTION[layer] = _attention(layer)
    return _ATTENTION[layer]


def producer_weights(attn):
    """Load-time packing of the producer's weights from the reference module."""
    n = 64
    kv_b = attn.kv_b_proj.weight.view(n, 448, 512)
    return dict(
        w_qkv_a=torch.cat([attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight], 0).contiguous(),
        q_a_norm=attn.q_a_layernorm.weight.contiguous(),
        kv_a_norm=attn.kv_a_layernorm.weight.contiguous(),
        w_q_b=attn.q_b_proj.weight.contiguous(),
        w_uk=kv_b[:, :192, :].transpose(1, 2).contiguous(),
        w_uv=kv_b[:, 192:, :].contiguous(),
    )


def rope_interleaved(x, cos, sin):
    """apply_rotary_pos_emb_interleave on one tensor [.., T, 64] with [T, 64] cos/sin."""
    c = cos[..., :32]
    s = sin[..., :32]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


@pytest.mark.parametrize("capacity,rows", ROWS)
def test_glm_producer(g, capacity, rows):
    program = _program("producer", max_rows=capacity)
    attn = _attn(0)
    w = producer_weights(attn)
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    gen = torch.Generator(device="cpu").manual_seed(rows)
    positions = torch.randperm(8192, generator=gen)[:rows].to(torch.int64).cuda()
    pages = (rows + 63) // 64 + 4
    slots = torch.randperm(pages * 64, generator=gen)[:rows].to(torch.int64).cuda()
    cache = torch.zeros((pages, 64 * 656), dtype=torch.uint8, device="cuda")
    query = torch.empty((rows, 64, 576), dtype=torch.bfloat16, device="cuda")
    q_resid = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    program.launch(x, positions, slots, cos_sin(8192), w["w_qkv_a"], w["q_a_norm"], w["kv_a_norm"],
                   w["w_q_b"], w["w_uk"], cache, query, q_resid, _scratch(program, rows),
                   scalars=(rows,))
    with torch.no_grad():
        cos, sin = reference_rope(positions)
        ref_resid = attn.q_a_layernorm(attn.q_a_proj(x))
        q_states = attn.q_b_proj(ref_resid).view(rows, 64, 256)
        q_pass, q_rot = q_states[..., :192], q_states[..., 192:]
        q_rot = rope_interleaved(q_rot.transpose(0, 1), cos[0], sin[0]).transpose(0, 1)
        latent_query = torch.einsum("thd,hcd->thc", q_pass.float(), w["w_uk"].float())
        ref_query = torch.cat([latent_query, q_rot.float()], -1)
        kv = attn.kv_a_proj_with_mqa(x)
        latent = attn.kv_a_layernorm(kv[:, :512])
        k_rot = rope_interleaved(kv[:, 512:], cos[0], sin[0])
        from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference

        ref_records = unpack_mla_kv_cache_reference(pack_mla_kv_cache_reference(latent, k_rot)).squeeze(1)
    torch.cuda.synchronize()
    records = unpack_latent_records(cache, slots)
    c_resid = cosine(q_resid, ref_resid)
    c_query = cosine(query, ref_query)
    c_rope = cosine(query[..., 512:], q_rot)
    c_rec = cosine(records, ref_records)
    c_lat = cosine(records[:, :512], latent)
    c_krope = cosine(records[:, 512:], k_rot)
    print(f"glm_producer m{capacity} rows={rows}: q_resid {c_resid:.7f} query {c_query:.7f} "
          f"q_rope {c_rope:.7f} record-vs-packed-ref {c_rec:.7f} latent-vs-bf16 {c_lat:.6f} k_rope {c_krope:.7f}")
    assert min(c_resid, c_query, c_rope, c_rec, c_krope) >= COS
    assert c_lat >= 0.999  # FP8 E4M3 group-128 latent vs the BF16 reference


@pytest.mark.parametrize("capacity,rows", ROWS)
def test_glm_o(g, capacity, rows):
    program = _program("o", max_rows=capacity)
    attn = _attn(0)
    w = producer_weights(attn)
    gen = torch.Generator(device="cpu").manual_seed(7)
    latent_out = (torch.randn((rows, 64, 512), generator=gen) * 0.3).bfloat16().cuda()
    out = torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda")
    program.launch(latent_out, w["w_uv"], attn.o_proj.weight, out, _scratch(program, rows), scalars=(rows,))
    with torch.no_grad():
        values = torch.einsum("thc,hvc->thv", latent_out.float(), w["w_uv"].float()).bfloat16()
        expected = attn.o_proj(values.reshape(rows, -1))
    torch.cuda.synchronize()
    c = cosine(out, expected)
    print(f"glm_o m{capacity} rows={rows}: cosine {c:.7f}")
    assert c >= COS
