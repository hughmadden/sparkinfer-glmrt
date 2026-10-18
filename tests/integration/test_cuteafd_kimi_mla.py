"""Kimi NoPE MLA vs the checkpoint's official class, with real layer-3 weights.

Range reads load only attention tensors, never an entire expert shard. The
AST extraction executes the original RMSNorm/MLA/eager definitions without
requiring the unrelated KDA extension. No replacement forward is injected.
"""
from __future__ import annotations

import ast
import __future__
from functools import lru_cache
import json
import os
from pathlib import Path
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

SNAPSHOT = Path(os.environ.get("CUTEAFD_KIMI_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06"))


@lru_cache(maxsize=1)
def official():
    source = SNAPSHOT / "modeling_kimi_linear.py"
    if not source.is_file():
        pytest.skip("official Kimi snapshot is not mounted")
    names = {"KimiRMSNorm", "KimiMLAAttention", "repeat_kv", "eager_attention_forward"}
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in tree.body} == names
    ns = dict(torch=torch, nn=nn, F=F, ALL_LAYERNORM_LAYERS=[])
    exec(compile(tree, str(source), "exec", flags=__future__.annotations.compiler_flag), ns)
    return ns


@lru_cache(maxsize=12)
def weight(name):
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    key = "language_model.model.layers.3.self_attn." + name
    path = SNAPSHOT / index[key]
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))[key]
        start, end = header["data_offsets"]
        handle.seek(8 + size + start)
        data = bytearray(handle.read(end - start))
    dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[header["dtype"]]
    return torch.frombuffer(data, dtype=dtype).reshape(header["shape"]).clone()


def module(device, *, heads=48, rank=0):
    cfg = json.loads((SNAPSHOT / "config.json").read_text())["text_config"]
    cfg.update(num_attention_heads=heads, num_key_value_heads=heads, _attn_implementation="eager")
    with torch.device("meta"):
        model = official()["KimiMLAAttention"](types.SimpleNamespace(**cfg), 3)
    model.to_empty(device=device)
    h = slice(rank * heads, (rank + 1) * heads)
    with torch.no_grad():
        for name, param in model.named_parameters():
            value = weight(name)
            if name == "q_b_proj.weight":
                value = value.reshape(96, 192, 1536)[h].flatten(0, 1)
            elif name == "kv_b_proj.weight":
                value = value.reshape(96, 256, 512)[h].flatten(0, 1)
            elif name == "g_proj.weight":
                value = value.reshape(96, 128, 7168)[h].flatten(0, 1)
            elif name == "o_proj.weight":
                value = value.reshape(7168, 96, 128)[:, h].flatten(1)
            param.copy_(value.float())
    return model.eval()


def inputs(rows, device):
    gen = torch.Generator().manual_seed(273)
    return (torch.randn(rows, 7168, generator=gen) * 0.2).bfloat16().to(device)


def mask(rows, total, device, dtype):
    q = torch.arange(total - rows, total, device=device)
    k = torch.arange(total, device=device)
    return torch.zeros(rows, total, device=device, dtype=dtype).masked_fill(k[None] > q[:, None], -torch.inf)[None, None]


def metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-20))
    cos = float(F.cosine_similarity(a, b, dim=0))
    print(f"relative_l2={rel:.8g} cosine={cos:.9g}", flush=True)
    assert rel < 0.025 and cos > 0.9997


def test_official_norm_and_fp64_absorption_cpu():
    torch.set_num_threads(min(16, torch.get_num_threads()))
    model = module("cpu")
    assert model.q_a_layernorm.variance_epsilon == 1e-6
    assert model.kv_a_layernorm.variance_epsilon == 1e-6
    assert model.config.rms_norm_eps == 1e-5
    assert model.rotary_emb is None
    x = inputs(3, "cpu").float()
    with torch.no_grad():
        qa = model.q_a_layernorm(model.q_a_proj(x))
        q = model.q_b_proj(qa).reshape(3, 48, 192).double()
        compressed = model.kv_a_proj_with_mqa(x)
        latent = model.kv_a_layernorm(compressed[:, :512]).double()
        raw = compressed[:, 512:].double()
        uk, uv = model.kv_b_proj.weight.double().reshape(48, 256, 512).split(128, dim=1)
        qabs = torch.cat((torch.einsum("thd,hdk->thk", q[:, :, :128], uk), q[:, :, 128:]), dim=-1)
        records = torch.cat((latent, raw), dim=-1)
        score = torch.einsum("thd,sd->hts", qabs, records) * (192 ** -0.5)
        score += mask(3, 3, "cpu", torch.float64)[0]
        prob = score.softmax(-1)
        value = torch.einsum("hts,sk->thk", prob, latent)
        value = torch.einsum("thk,hvk->thv", value, uv).flatten(1)
        gate = model.g_proj(x).double().sigmoid()
        expected = F.linear(value * gate, model.o_proj.weight.double())
        actual = model(x[None], attention_mask=mask(3, 3, "cpu", torch.float32))[0]
        torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-4)
        # Moving/removing the output gate is not an equivalent operator.
        assert not torch.allclose(expected, F.linear(value, model.o_proj.weight.double()))


class History:
    def __init__(self):
        self.key = self.value = None

    def update(self, key, value, layer):
        if self.key is not None:
            key = torch.cat((self.key, key), dim=2)
            value = torch.cat((self.value, value), dim=2)
        self.key, self.value = key, value
        return key, value


@pytest.mark.parametrize("heads,rank", [(48, 0), (48, 1), (96, 0)])
@pytest.mark.parametrize("mode,rows", [(mode, rows) for mode in ("decode", "verify", "extend")
                                     for rows in (1, 16, 64)] + [("extend", 4096)])
def test_real_mla_gpu_and_graph_replay(heads, rank, mode, rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd.kimi_mla import (KimiMlaGeometry, compile_kimi_mla_producer_aot,
        compile_kimi_dense_mla_aot, compile_kimi_mla_output_aot)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    g = KimiMlaGeometry(heads=heads)
    model = module("cuda", heads=heads, rank=rank)
    packed = {name: param.detach().bfloat16().contiguous() for name, param in model.named_parameters()}
    w = torch.cat((packed["q_a_proj.weight"], packed["kv_a_proj_with_mqa.weight"])).contiguous()
    uk, uv = packed["kv_b_proj.weight"].reshape(heads, 256, 512).split(128, dim=1)
    uk = uk.transpose(1, 2).contiguous()
    uv = uv.contiguous()
    capacity = 4096 if rows == 4096 else 64
    context = 4352 if rows == 4096 else 256
    producer = compile_kimi_mla_producer_aot(g, max_rows=capacity)
    attention = compile_kimi_dense_mla_aot(g, max_rows=capacity, max_context=context, mode=mode)
    output = compile_kimi_mla_output_aot(g, max_rows=capacity)
    cache = torch.full(((context + 63) // 64, 64, 576), float("nan"), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(cache.shape[0] - 1, -1, -1, dtype=torch.int32, device="cuda")[None]
    def physical_slots(count, begin=0):
        logical = torch.arange(begin, begin + count, device="cuda")
        return table[0, logical // 64].long() * 64 + logical % 64
    history = History()
    prefix = 65
    x = inputs(rows + prefix, "cuda")
    q = torch.empty(rows, heads, 576, dtype=torch.bfloat16, device="cuda")
    qr = torch.empty(rows, 1536, dtype=torch.bfloat16, device="cuda")
    a = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(rows, heads, device="cuda")
    y = torch.empty(rows, 7168, dtype=torch.bfloat16, device="cuda")
    scratch = [torch.empty(max(1, p.scratch_bytes(rows)["scratch"]), dtype=torch.uint8, device="cuda")
               for p in (producer, attention, output)]
    if prefix:
        px = x[:prefix].contiguous()
        pq, pr = torch.empty(prefix, heads, 576, dtype=torch.bfloat16, device="cuda"), torch.empty(prefix, 1536, dtype=torch.bfloat16, device="cuda")
        ps = torch.empty(producer.scratch_bytes(prefix)["scratch"], dtype=torch.uint8, device="cuda")
        producer.launch(px, physical_slots(prefix), w, packed["q_a_layernorm.weight"],
            packed["kv_a_layernorm.weight"], packed["q_b_proj.weight"], uk, cache, pq, pr, ps, scalars=(prefix,))
        with torch.no_grad():
            model(px.float()[None], attention_mask=mask(prefix, prefix, "cuda", torch.float32), past_key_values=history)
    live = x[prefix:].contiguous()
    saved_prefix = cache.clone()
    slots = physical_slots(rows, prefix)
    lengths = torch.tensor([prefix + rows], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    splits = attention.geometry["num_splits"]

    def run():
        producer.launch(live, slots, w, packed["q_a_layernorm.weight"], packed["kv_a_layernorm.weight"],
                        packed["q_b_proj.weight"], uk, cache, q, qr, scratch[0], scalars=(rows,))
        attention.launch(q, cache, table, lengths, cu, a, lse, scratch[1],
                         scalars=(rows, 1, table.shape[1], splits))
        output.launch(a, live, uv, packed["g_proj.weight"], packed["o_proj.weight"], y, scratch[2], scalars=(rows,))

    run()
    torch.cuda.synchronize()
    with torch.no_grad():
        expected = model(live.float()[None], attention_mask=mask(rows, prefix + rows, "cuda", torch.float32), past_key_values=history)[0]
    metrics(y, expected)
    assert torch.isfinite(lse).all()
    baseline = [t.clone() for t in (y, q, cache, lse)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip((y, q, cache, lse), baseline):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    # Restore the exact frontier, including the partial page. Replaying the
    # continuation must regenerate every cache byte and output identically.
    cache.copy_(saved_prefix)
    run()
    for actual, expected in zip((y, q, cache, lse), baseline):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    # Padding rows still compute queries but must not overwrite a cache slot.
    slots.fill_(-1)
    before = cache.clone()
    producer.launch(live, slots, w, packed["q_a_layernorm.weight"], packed["kv_a_layernorm.weight"],
                    packed["q_b_proj.weight"], uk, cache, q, qr, scratch[0], scalars=(rows,))
    assert torch.equal(cache.view(torch.uint8), before.view(torch.uint8))
