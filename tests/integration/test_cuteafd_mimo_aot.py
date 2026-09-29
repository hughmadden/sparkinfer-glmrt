"""cuteafd MiMo V2 Flash AOT programs vs transformers ``modeling_mimo_v2_flash``.

Real MiMo-V2-Flash weights (layer 0: dense MLP and full attention; layer 1:
SWA attention with sinks and the FP32 router), golden activations when
mounted. Decode programs (m64) at 1/16/64 live rows, prefill programs (m4096)
up to 4096 rows. Every output is compared with the module it replaces at
cosine >= 0.9999; attention runs as the engine chains it (producer ->
attention -> o_proj) over prefill chunks and decode steps against the
reference module over the whole sequence.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._mimo import (
    config, cos_sin, cosine, golden_rows, layer_kind, masks, reference_module, reference_rope, rms_norm, tensor,
)

DECODE, PREFILL = 64, 4096
ROWS = [(DECODE, 1), (DECODE, 16), (DECODE, 64), (PREFILL, 4096)]
COS = 0.9999


@pytest.fixture(scope="module")
def g():
    require_b12x()
    from b12x.integration.cuteafd import MIMO_V2_FLASH

    # The NVIDIA PyTorch images default FP32 matmuls to TF32; the router
    # reference must be FP32 like the checkpoint's F.linear.
    torch.backends.cuda.matmul.allow_tf32 = False

    return MIMO_V2_FLASH


_PROGRAMS: dict = {}


def _program(program_kind, **kw):
    key = (program_kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import MIMO_V2_FLASH, mimo_attention, mimo_ffn

        fn = {"norm": mimo_ffn.compile_mimo_norm_aot, "ffn": mimo_ffn.compile_mimo_ffn_aot,
              "router": mimo_ffn.compile_mimo_router_scores_aot, "o": mimo_attention.compile_mimo_o_aot,
              "producer": mimo_attention.compile_mimo_producer_aot,
              "attention": mimo_attention.compile_mimo_attention_aot}[program_kind]
        _PROGRAMS[key] = fn(MIMO_V2_FLASH, **kw)
    return _PROGRAMS[key]


def _scratch(program, rows):
    size = program.scratch_bytes(rows).get("scratch", 0)
    return torch.empty(max(size, 1024), dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("deltas", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 64, 4096])
def test_mimo_norm(g, rows, deltas):
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
    c = cosine(out, expected)
    mismatch = float((out != expected).float().mean())
    print(f"mimo_norm rows={rows} deltas={deltas}: cosine {c:.7f} bf16 mismatch {mismatch:.2e}")
    assert cosine(residual, expected_residual) >= COS and c >= COS and mismatch < 1e-2


@pytest.mark.parametrize("capacity,rows", ROWS)
def test_mimo_ffn(g, capacity, rows):
    program = _program("ffn", max_rows=capacity)
    mlp = reference_module("MiMoV2FlashMLP", config(), prefix="model.layers.0.mlp.")
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.post_attention_layernorm.weight"))
    w_gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
    out = torch.empty_like(x)
    program.launch(x, w_gate_up, mlp.down_proj.weight, out, _scratch(program, rows), scalars=(rows,))
    with torch.no_grad():
        expected = mlp(x)
    torch.cuda.synchronize()
    c = cosine(out, expected)
    print(f"mimo_ffn I=16384 m{capacity} rows={rows}: cosine {c:.7f}")
    assert c >= COS


@pytest.mark.parametrize("rows", [1, 32, 64, 4096])
def test_mimo_router_scores(g, rows):
    from b12x.integration.cuteafd.mimo_ffn import split_router_weight

    program = _program("router")
    gate = reference_module("MiMoV2FlashTopkRouter", config(), prefix="model.layers.1.mlp.gate.",
                            fp32=("weight", "e_score_correction_bias"))
    x = rms_norm(golden_rows(1, rows), tensor("model.layers.1.post_attention_layernorm.weight"))
    logits = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
    program.launch(x, split_router_weight(gate.weight), logits, _scratch(program, rows), scalars=(rows,))
    with torch.no_grad():
        expected, _, expected_ids = gate(x)
    torch.cuda.synchronize()
    c = cosine(logits, expected)
    choice = logits.sigmoid() + gate.e_score_correction_bias
    ids = choice.topk(8, dim=-1)[1]
    same = float((ids.sort(-1)[0] == expected_ids.sort(-1)[0]).all(-1).float().mean())
    err = float((logits - expected).abs().max())
    print(f"mimo_router_scores rows={rows}: cosine {c:.9f} max abs {err:.2e} identical top-8 {same:.4f}")
    assert c >= 0.999999 and err < 1e-3 and same >= 0.999


@pytest.mark.parametrize("capacity,rows", ROWS)
def test_mimo_o(g, capacity, rows):
    program = _program("o", max_rows=capacity)
    w_o = tensor("model.layers.1.self_attn.o_proj.weight")
    attn = (golden_rows(2, rows * 2, seed=3).view(rows, 8192) * 0.2).contiguous()
    out = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    program.launch(attn, w_o, out, scalars=(rows,))
    expected = torch.nn.functional.linear(attn, w_o)
    torch.cuda.synchronize()
    c = cosine(out, expected)
    print(f"mimo_o m{capacity} rows={rows}: cosine {c:.7f}")
    assert c >= COS


_ATTENTION: dict = {}


def _attention_module(layer):
    if layer not in _ATTENTION:
        _ATTENTION[layer] = reference_module("MiMoV2FlashAttention", config(), layer,
                                             prefix=f"model.layers.{layer}.self_attn.")
    return _ATTENTION[layer]


def _qkv(attn):
    return torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0).contiguous()


def _layer_input(layer, rows, seed=0):
    return rms_norm(golden_rows(max(layer - 1, 0), rows, seed=seed),
                    tensor(f"model.layers.{layer}.input_layernorm.weight"))


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("capacity,rows", ROWS)
def test_mimo_producer(g, layer, capacity, rows):
    from transformers.models.glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb

    kind = layer_kind(layer)
    program = _program("producer", kind=kind, max_rows=capacity)
    attn = _attention_module(layer)
    kv_heads, r = g.kv_heads(kind), g.record_elems(kind)
    x = _layer_input(layer, rows)
    positions = torch.arange(rows, device="cuda") + 7
    slots = torch.randperm(rows + 8, device="cuda")[:rows]
    cache = torch.zeros((rows + 8, r), dtype=torch.bfloat16, device="cuda")
    query = torch.empty((rows, 64, 192), dtype=torch.bfloat16, device="cuda")
    table = cos_sin(int(positions.max()) + 1, kind)
    program.launch(x, positions, slots, table, _qkv(attn), cache, query, _scratch(program, rows), scalars=(rows,))
    with torch.no_grad():
        q = attn.q_proj(x).view(1, rows, 64, 192).transpose(1, 2)
        k = attn.k_proj(x).view(1, rows, kv_heads, 192).transpose(1, 2)
        v = attn.v_proj(x).view(rows, kv_heads * 128) * attn.v_scale
        cos, sin = reference_rope(positions, kind)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    torch.cuda.synchronize()
    records = cache[slots]
    cq = cosine(query, q[0].transpose(0, 1))
    ck = cosine(records[:, :kv_heads * 192], k[0].transpose(0, 1).reshape(rows, -1))
    cv = cosine(records[:, kv_heads * 192:], v)
    print(f"mimo_{kind}_producer m{capacity} rows={rows}: query {cq:.7f} key {ck:.7f} value {cv:.7f}")
    assert min(cq, ck, cv) >= COS


class _Sequence:
    """One sequence's engine state: paged records (full) or its ring (swa)."""

    def __init__(self, g, kind, capacity_tokens, ring_id=0, rings=1, pages_total=None, pages=None, cache=None,
                 ring=None):
        self.kind, self.len = kind, 0
        r = g.record_elems(kind)
        if kind == "full":
            count = -(-capacity_tokens // 64)
            self.pages = pages if pages is not None else torch.randperm(count, device="cuda").int()
            self.cache = cache if cache is not None else torch.zeros(((pages_total or count) * 64, r),
                                                                     dtype=torch.bfloat16, device="cuda")
        else:
            self.ring_id = ring_id
            self.cache = ring if ring is not None else torch.zeros((rings * g.ring_rows, r), dtype=torch.bfloat16,
                                                                   device="cuda")

    def slots(self, positions):
        if self.kind == "full":
            return self.pages.long()[positions // 64] * 64 + positions % 64
        return self.ring_id * 256 + positions % 256


def _step(g, kind, route, sequences, x_rows, attn_weights, splits=8):
    """One engine step: rows of each (sequence, n) in order; returns o_proj rows."""
    capacity = DECODE if route == "decode" else PREFILL
    rows = sum(n for _, n in sequences)
    positions = torch.cat([torch.arange(s.len, s.len + n, device="cuda") for s, n in sequences])
    producer = _program("producer", kind=kind, max_rows=capacity)
    attention = _program("attention", kind=kind, route=route, max_rows=capacity)
    o = _program("o", max_rows=capacity)
    table = cos_sin(int(positions.max()) + 1, kind)
    query = torch.empty((rows, 64, 192), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, 64, 128), dtype=torch.bfloat16, device="cuda")
    if kind == "full":
        slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, n in sequences])
        cache = sequences[0][0].cache
        producer.launch(x_rows, positions, slots, table, attn_weights["qkv"], cache, query,
                        _scratch(producer, rows), scalars=(rows,))
        stride = max(len(s.pages) for s, _ in sequences)
        if route == "prefill":
            page_table, table_stride = sequences[0][0].pages.contiguous(), 0
        else:
            page_table = torch.zeros((rows, stride), dtype=torch.int32, device="cuda")
            r0 = 0
            for s, n in sequences:
                page_table[r0:r0 + n, :len(s.pages)] = s.pages
                r0 += n
            table_stride = stride
        scalars = (rows, table_stride) + ((splits,) if route == "decode" else ())
        attention.launch(query, cache, positions, page_table, out, _scratch(attention, rows), scalars=scalars)
    else:
        kv_step = torch.empty((rows, g.record_elems(kind)), dtype=torch.bfloat16, device="cuda")
        producer.launch(x_rows, positions, torch.arange(rows, device="cuda"), table, attn_weights["qkv"], kv_step,
                        query, _scratch(producer, rows), scalars=(rows,))
        ring_slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, n in sequences])
        firsts, r0 = [], 0
        for _, n in sequences:
            firsts.append(torch.full((n,), r0, dtype=torch.int32, device="cuda"))
            r0 += n
        attention.launch(query, kv_step, sequences[0][0].cache, positions, ring_slots, torch.cat(firsts),
                         attn_weights["sinks"], out, _scratch(attention, rows), scalars=(rows,))
    result = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    o.launch(out.view(rows, -1), attn_weights["w_o"], result, scalars=(rows,))
    for s, n in sequences:
        s.len += n
    return result


def _weights(layer):
    attn = _attention_module(layer)
    w = {"qkv": _qkv(attn), "w_o": attn.o_proj.weight.contiguous()}
    if attn.sinks is not None:
        w["sinks"] = attn.sinks.detach().contiguous()
    return w


def _reference(layer, x):
    attn = _attention_module(layer)
    kind = layer_kind(layer)
    t = x.shape[0]
    positions = torch.arange(t, device="cuda")
    with torch.no_grad():
        out, _ = attn(x[None], position_embeddings=reference_rope(positions, kind), attention_mask=masks(t)[kind])
    return out[0]


# (schedule of (route, rows) steps over one sequence)
SCHEDULES = {
    "prefill": [("prefill", 1571)],
    "prefill_4096": [("prefill", 4096)],
    "chunked": [("prefill", 700), ("prefill", 300), ("prefill", 571)],
    "decode": [("prefill", 1400)] + [("decode", 1)] * 8 + [("decode", 5)] * 4 + [("decode", 64), ("decode", 79)],
}


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_mimo_attention_chain(g, layer, schedule):
    kind = layer_kind(layer)
    steps = SCHEDULES[schedule]
    total = sum(n for _, n in steps)
    x = _layer_input(layer, total)
    expected = _reference(layer, x)
    weights = _weights(layer)
    seq = _Sequence(g, kind, total)
    worst = 1.0
    for route, n in steps:
        while n > 0:
            take = min(n, DECODE if route == "decode" else PREFILL)
            start = seq.len
            got = _step(g, kind, route, [(seq, take)], x[start:start + take], weights)
            torch.cuda.synchronize()
            c = cosine(got, expected[start:start + take])
            worst = min(worst, c)
            if c < COS:
                print(f"  step {route} rows {start}..{start + take}: cosine {c:.7f}")
            n -= take
    print(f"mimo_{kind}_attention chain {schedule} ({total} rows): worst step cosine {worst:.7f}")
    assert worst >= COS


@pytest.mark.parametrize("layer", [0, 1])
def test_mimo_attention_two_sequences(g, layer):
    """One decode step mixing two sequences of different lengths."""
    kind = layer_kind(layer)
    a_len, b_len, a_new, b_new = 900, 300, 3, 5
    xa = _layer_input(layer, a_len + a_new)
    xb = _layer_input(layer, b_len + b_new, seed=5)[torch.arange(b_len + b_new).flip(0)].contiguous()
    ea, eb = _reference(layer, xa), _reference(layer, xb)
    weights = _weights(layer)
    if kind == "full":
        pages = torch.randperm(32, device="cuda").int()
        a = _Sequence(g, kind, a_len + a_new, pages=pages[:15], pages_total=32)
        b = _Sequence(g, kind, b_len + b_new, pages=pages[15:21], cache=a.cache)
    else:
        a = _Sequence(g, kind, 0, ring_id=1, rings=3)
        b = _Sequence(g, kind, 0, ring_id=2, ring=a.cache)
    _step(g, kind, "prefill", [(a, a_len)], xa[:a_len], weights)
    _step(g, kind, "prefill", [(b, b_len)], xb[:b_len], weights)
    got = _step(g, kind, "decode", [(a, a_new), (b, b_new)], torch.cat([xa[a_len:], xb[b_len:]]), weights)
    torch.cuda.synchronize()
    ca, cb = cosine(got[:a_new], ea[a_len:]), cosine(got[a_new:], eb[b_len:])
    print(f"mimo_{kind}_attention two sequences: a {ca:.7f} b {cb:.7f}")
    assert min(ca, cb) >= COS
