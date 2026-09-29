"""A whole GLM 5.3 decoder layer chained from the cuteafd AOT programs,
against the golden harness (transformers reference) layer outputs.

Layer 0 (dense MLP, full indexer) and layer 3 (MoE, shared indexer: the
top-k of layer 2, recomputed from layer01.bin with layer 2's programs).
Routed experts of layer 3 run in torch from the checkpoint (BF16, the
reference ``GlmMoeDsaExperts``) for this test only. Two schedules:

* ``prefill``: the whole prompt as one m4096 chunk;
* ``mixed``:   a 1024-row m4096 prefill chunk, then 64-row m64 decode steps
               (per-row page tables, split-KV decode attention).

Needs the golden directory (``CUTEAFD_GLM_GOLDEN``, default ``/golden``).
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import GOLDEN, config, cos_sin, cosine, tensor

DECODE, PREFILL = 64, 4096
MAX_PAGES = 131072 // 64
_PROGRAMS: dict = {}


def P(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_attention, glm_ffn, glm_indexer, glm_sparse_mla

        fn = {"norm": glm_ffn.compile_glm_norm_aot, "ffn": glm_ffn.compile_glm_ffn_aot,
              "router": glm_ffn.compile_glm_router_scores_aot,
              "producer": glm_attention.compile_glm_producer_aot,
              "index_producer": glm_attention.compile_glm_index_producer_aot,
              "o": glm_attention.compile_glm_o_aot,
              "topk": glm_indexer.compile_glm_index_topk_aot,
              "mla": glm_sparse_mla.compile_glm_sparse_mla_aot}[kind]
        _PROGRAMS[key] = fn(GLM53, **kw)
    return _PROGRAMS[key]


def golden(name: str) -> torch.Tensor:
    path = GOLDEN / name
    if not path.is_file():
        pytest.skip(f"golden output {path} not mounted")
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).view(-1, 6144).cuda()


def scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


class LayerWeights:
    def __init__(self, layer: int):
        p = f"model.layers.{layer}."
        a = p + "self_attn."
        t = lambda n: tensor(n).contiguous()  # noqa: E731
        kv_b = t(a + "kv_b_proj.weight").view(64, 448, 512)
        self.input_norm = t(p + "input_layernorm.weight")
        self.post_norm = t(p + "post_attention_layernorm.weight")
        self.w_qkv_a = torch.cat([t(a + "q_a_proj.weight"), t(a + "kv_a_proj_with_mqa.weight")], 0).contiguous()
        self.q_a_norm = t(a + "q_a_layernorm.weight")
        self.kv_a_norm = t(a + "kv_a_layernorm.weight")
        self.w_q_b = t(a + "q_b_proj.weight")
        self.w_uk = kv_b[:, :192, :].transpose(1, 2).contiguous()
        self.w_uv = kv_b[:, 192:, :].contiguous()
        self.w_o = t(a + "o_proj.weight")
        self.full = config().indexer_types[layer] == "full"
        if self.full:
            ix = a + "indexer."
            self.w_iq = t(ix + "wq_b.weight")
            self.w_ik = torch.cat([t(ix + "wk.weight"), t(ix + "weights_proj.weight").bfloat16()], 0).contiguous()
            self.k_norm_w = t(ix + "k_norm.weight")
            self.k_norm_b = t(ix + "k_norm.bias")
        self.dense = config().mlp_layer_types[layer] == "dense"
        m = p + "mlp."
        if self.dense:
            self.w_gate_up = torch.cat([t(m + "gate_proj.weight"), t(m + "up_proj.weight")], 0).contiguous()
            self.w_down = t(m + "down_proj.weight")
        else:
            s = m + "shared_experts."
            self.w_gate_up = torch.cat([t(s + "gate_proj.weight"), t(s + "up_proj.weight")], 0).contiguous()
            self.w_down = t(s + "down_proj.weight")
            self.w_router = t(m + "gate.weight").bfloat16()
            self.router_bias = t(m + "gate.e_score_correction_bias").float()


class Sequence:
    """One sequence's page map; latent and index caches share page ids."""

    def __init__(self, tokens: int, seed: int = 3):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.pages = -(-tokens // 64)
        pool = self.pages + 9
        self.page_of = torch.randperm(pool, generator=gen)[:self.pages].to(torch.int32).cuda()
        self.kv_cache = torch.zeros((pool, 64 * 656), dtype=torch.uint8, device="cuda")
        self.index_cache = torch.zeros((pool, 8448), dtype=torch.uint8, device="cuda")

    def slots(self, positions: torch.Tensor) -> torch.Tensor:
        return (self.page_of[positions // 64].long() * 64 + positions % 64).contiguous()


def run_indexer(w: LayerWeights, seq: Sequence, x, q_resid, positions, cs, route, capacity):
    rows = x.shape[0]
    ip = P("index_producer", max_rows=capacity)
    q_fp8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
    head_weights = torch.empty((rows, 32), dtype=torch.float32, device="cuda")
    ip.launch(x, q_resid, positions, seq.slots(positions), cs, w.w_iq, w.w_ik, w.k_norm_w, w.k_norm_b,
              seq.index_cache, q_fp8, head_weights, scratch(ip, rows), scalars=(rows,))
    topk = P("topk", max_rows=capacity, max_pages=MAX_PAGES, mode=route)
    lengths = (positions + 1).to(torch.int32)
    width = int(positions.max()) // 64 + 1
    indices = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
    sc = torch.zeros(topk.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    if route == "prefill":
        table, stride = seq.page_of[:width].contiguous(), 0
    else:
        stride = seq.pages
        table = seq.page_of[None].expand(rows, -1).contiguous()
    topk.launch(q_fp8, head_weights, seq.index_cache, table, lengths, indices, sc, scalars=(rows, width, stride))
    return indices


def run_attention(w: LayerWeights, seq: Sequence, h, positions, cs, route, capacity, indices=None,
                  index_weights=None):
    """Returns (x_attn_in, attention output, indices used)."""
    rows = h.shape[0]
    norm = P("norm")
    x = torch.empty_like(h)
    norm.launch(h.clone(), h, h, w.input_norm, x, scalars=(rows, 0))
    producer = P("producer", max_rows=capacity)
    query = torch.empty((rows, 64, 576), dtype=torch.bfloat16, device="cuda")
    q_resid = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    producer.launch(x, positions, seq.slots(positions), cs, w.w_qkv_a, w.q_a_norm, w.kv_a_norm, w.w_q_b,
                    w.w_uk, seq.kv_cache, query, q_resid, scratch(producer, rows), scalars=(rows,))
    if indices is None:
        indices = run_indexer(w, seq, x, q_resid, positions, cs, route, capacity)
    mla = P("mla", route=route, max_rows=capacity)
    lengths = torch.full((rows,), 2048, dtype=torch.int32, device="cuda")
    latent = torch.empty((rows, 64, 512), dtype=torch.bfloat16, device="cuda")
    mla.launch(query, seq.kv_cache, indices, lengths, latent, scratch(mla, rows), scalars=(rows,))
    o = P("o", max_rows=capacity)
    out = torch.empty_like(h)
    o.launch(latent, w.w_uv, w.w_o, out, scratch(o, rows), scalars=(rows,))
    return out, indices


_EXPERTS: dict = {}


def routed_experts(layer: int, x: torch.Tensor, logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference router selection from FP32 logits + GlmMoeDsaExperts (torch, BF16)."""
    cfg = config()
    scores = logits.sigmoid()
    choice = scores + bias
    idx = torch.topk(choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    weights = scores.gather(1, idx)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
    if layer not in _EXPERTS:
        from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                experts = ref.GlmMoeDsaExperts(cfg)
        finally:
            torch.set_default_dtype(torch.float32)
        experts = experts.to_empty(device="cuda").eval()
        base = f"model.layers.{layer}.mlp.experts."
        with torch.no_grad():
            for e in range(experts.gate_up_proj.shape[0]):
                experts.gate_up_proj[e].copy_(torch.cat([tensor(f"{base}{e}.gate_proj.weight"),
                                                         tensor(f"{base}{e}.up_proj.weight")], 0))
                experts.down_proj[e].copy_(tensor(f"{base}{e}.down_proj.weight"))
        _EXPERTS.clear()
        _EXPERTS[layer] = experts
    with torch.no_grad():
        return _EXPERTS[layer](x, idx, weights.to(x.dtype))


def run_layer(layer: int, h_in: torch.Tensor, schedule: str, next_norm: torch.Tensor,
              shared_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    w = LayerWeights(layer)
    T = h_in.shape[0]
    seq = Sequence(T, seed=layer + 1)
    cs = cos_sin(T)
    chunks = [(0, T, "prefill", PREFILL)] if schedule == "prefill" else \
        [(0, 1024, "prefill", PREFILL)] + [(s, min(s + 64, T), "decode", DECODE) for s in range(1024, T, 64)]
    out = torch.empty_like(h_in)
    stats = {"selected_all_causal": True}
    for start, end, route, capacity in chunks:
        rows = end - start
        positions = torch.arange(start, end, dtype=torch.int64, device="cuda")
        h = h_in[start:end].clone()
        indices = None if w.full else shared_indices[start:end].contiguous()
        attn_out, indices = run_attention(w, seq, h, positions, cs, route, capacity, indices)
        # T <= 2048: DSA selects every causal row.
        valid = torch.sort(torch.where(indices >= 0, indices.long(), -1), dim=1).values
        for i in (0, rows // 2, rows - 1):
            expected = torch.sort(seq.slots(torch.arange(int(positions[i]) + 1, device="cuda"))).values
            got = valid[i][valid[i] >= 0]
            stats["selected_all_causal"] &= bool(torch.equal(got, expected))
        norm = P("norm")
        x2 = torch.empty_like(h)
        norm.launch(h, attn_out, attn_out, w.post_norm, x2, scalars=(rows, 1))
        ffn = P("ffn", inter=12288 if w.dense else 2048, max_rows=capacity)
        mlp = torch.empty_like(h)
        ffn.launch(x2, w.w_gate_up, w.w_down, mlp, scratch(ffn, rows), scalars=(rows,))
        deltas = 1
        routed = mlp
        if not w.dense:
            router = P("router")
            logits = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
            router.launch(x2, w.w_router, logits, scalars=(rows,))
            routed = routed_experts(layer, x2, logits, w.router_bias)
            deltas = 2
        next_x = torch.empty_like(h)
        norm.launch(h, routed, mlp, next_norm, next_x, scalars=(rows, deltas))
        out[start:end] = h
    torch.cuda.synchronize()
    return out, stats


def shared_topk_from(layer: int, h_in: torch.Tensor) -> torch.Tensor:
    """Layer ``layer``'s (full) indexer top-k over the whole prompt, via its programs."""
    w = LayerWeights(layer)
    T = h_in.shape[0]
    seq = Sequence(T, seed=layer + 1)
    cs = cos_sin(T)
    positions = torch.arange(T, dtype=torch.int64, device="cuda")
    x = torch.empty_like(h_in)
    P("norm").launch(h_in.clone(), h_in, h_in, w.input_norm, x, scalars=(T, 0))
    producer = P("producer", max_rows=PREFILL)
    query = torch.empty((T, 64, 576), dtype=torch.bfloat16, device="cuda")
    q_resid = torch.empty((T, 2048), dtype=torch.bfloat16, device="cuda")
    producer.launch(x, positions, seq.slots(positions), cs, w.w_qkv_a, w.q_a_norm, w.kv_a_norm, w.w_q_b,
                    w.w_uk, seq.kv_cache, query, q_resid, scratch(producer, T), scalars=(T,))
    indices = run_indexer(w, seq, x, q_resid, positions, cs, "prefill", PREFILL)
    # Physical slots of layer ``layer``'s own page map -> logical positions, then
    # re-mapped by the consumer's page map (the engine shares one page map).
    return indices, seq


@pytest.mark.parametrize("schedule", ["prefill", "mixed"])
@pytest.mark.parametrize("layer", [0, 3])
def test_glm_layer_chain(layer, schedule):
    require_b12x()
    h_in = golden(f"layer{layer - 1:02d}.bin") if layer else None
    if layer == 0:
        from tokenizers import Tokenizer  # noqa: F401

        ids = torch.frombuffer(bytearray((GOLDEN / "tokens.bin").read_bytes()), dtype=torch.int32).long()
        embed = tensor("model.embed_tokens.weight")
        h_in = torch.nn.functional.embedding(ids.cuda(), embed)
    expected = golden(f"layer{layer:02d}.bin")
    next_norm = tensor(f"model.layers.{layer + 1}.input_layernorm.weight")
    shared = None
    if config().indexer_types[layer] == "shared":
        source = layer - 1
        while config().indexer_types[source] != "full":
            source -= 1
        phys, src_seq = shared_topk_from(source, golden(f"layer{source - 1:02d}.bin") if source else None)
        # Translate the source layer's physical slots into this layer's page map.
        page_rank = torch.empty(int(src_seq.page_of.max()) + 1, dtype=torch.long, device="cuda")
        page_rank[src_seq.page_of.long()] = torch.arange(src_seq.pages, device="cuda")
        logical = torch.where(phys >= 0, page_rank[(phys // 64).long().clamp_min(0)] * 64 + phys % 64, -1)
        dst = Sequence(h_in.shape[0], seed=layer + 1)
        shared = torch.where(logical >= 0, dst.slots(logical.clamp_min(0)), -1).to(torch.int32)
    out, stats = run_layer(layer, h_in, schedule, next_norm, shared)
    c = cosine(out, expected)
    c_delta = cosine(out.float() - h_in.float(), expected.float() - h_in.float())
    rel = float((out.float() - expected.float()).norm() / expected.float().norm())
    print(f"glm layer {layer} ({schedule}, T={h_in.shape[0]}): output cosine {c:.7f} "
          f"rel_l2 {rel:.2e}; layer delta cosine {c_delta:.6f}; DSA selects all causal rows: "
          f"{stats['selected_all_causal']}")
    assert stats["selected_all_causal"]
    assert c >= 0.9999 and c_delta >= 0.999
