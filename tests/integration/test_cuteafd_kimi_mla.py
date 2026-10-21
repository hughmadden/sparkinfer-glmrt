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

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
SNAPSHOT = Path(os.environ.get("CUTEAFD_KIMI_SNAPSHOT", HF_HOME / "hub" /
    "models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06"))


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
    definitions = official()
    cfg = json.loads((SNAPSHOT / "config.json").read_text())["text_config"]
    cfg.update(num_attention_heads=heads, num_key_value_heads=heads, _attn_implementation="eager")
    with torch.device("meta"):
        model = definitions["KimiMLAAttention"](types.SimpleNamespace(**cfg), 3)
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


@pytest.mark.parametrize("heads", [48, 96])
@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_mla_cache_geometry_and_abi_cpu(monkeypatch, heads, kv_cache):
    from b12x.integration.cuteafd import kimi_mla as mla, kimi_split
    monkeypatch.setattr(mla, "compile_program", lambda launch, **kw: (launch, kw))
    monkeypatch.setattr(kimi_split, "compile_program", lambda launch, **kw: (launch, kw))
    g = mla.KimiMlaGeometry(heads=heads, kv_cache=kv_cache)
    record = 656 if kv_cache == "fp8" else 1152
    dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    assert g.record_bytes == record
    for compile_fn in (mla.compile_kimi_mla_producer_aot, kimi_split.compile_kimi_mla_rank_norm_aot):
        _, abi = compile_fn(g, max_rows=64)
        cache = next(v for v in abi["operands"] if v.name == "kv_cache")
        assert cache.dtype == dtype
        assert cache.shape == ("[pages,64,656]" if kv_cache == "fp8" else "[pages,64,576]")
        assert abi["geometry"]["kv_cache"] == kv_cache
    for mode in ("decode", "verify", "extend"):
        kernel, abi = mla.compile_kimi_dense_mla_aot(g, max_rows=64, max_context=524288, mode=mode)
        if kv_cache == "fp8" and mode == "extend":
            assert abi["geometry"]["native_fp8_prefill"] and abi["geometry"]["num_splits"] == 1
        else:
            assert kernel.forward.ds_mla_record == (kv_cache == "fp8")
        assert abi["geometry"]["max_context"] == 524288
        assert next(v for v in abi["operands"] if v.name == "kv_cache").dtype == dtype


def test_mla_compact_reference_scale_rounding_cpu():
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
    # Native GLM uses div.rn, not a CUDA scalar divide's reciprocal multiply.
    maxima = torch.tensor([0.25390625, 0.0, 448.0, 1.0], dtype=torch.bfloat16)
    latent = maxima.repeat_interleave(128)[None]
    tail = torch.arange(64, dtype=torch.bfloat16)[None]
    record = pack_mla_kv_cache_reference(latent, tail).squeeze(1)
    scales = record[:, 512:528].contiguous().view(torch.float32)
    expected = torch.tensor([[float(v) / 448.0 if v else 1.0 for v in maxima]])
    assert torch.equal(scales.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(record[:, 528:], tail.view(torch.uint8))
    # The observed value/scale is exactly 108, an E4M3 rounding midpoint.
    # A one-ULP larger reciprocal-derived scale wrongly flips its code.
    latent.fill_(3.0625)
    latent[0, 0] = 0.73828125
    record = pack_mla_kv_cache_reference(latent, tail).squeeze(1)
    exact_scale = record[0, 512:516].contiguous().view(torch.float32)[0]
    reciprocal_scale = torch.tensor(3.0625) * torch.tensor(1.0 / 448.0)
    assert exact_scale == 0.0068359375 and reciprocal_scale > exact_scale
    assert latent[0, 0].float() / exact_scale == 108
    reciprocal_code = (latent[0, 0].float() / reciprocal_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    assert record[0, 0] == 110 and reciprocal_code == 109


@pytest.mark.parametrize("heads", [48, 96])
def test_fp8_rank_norm_record_matches_shared_glm_format(heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, _RankNormPack
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
    programs = []
    for fmt in ("bf16", "fp8"):
        g = KimiMlaGeometry(heads=heads, kv_cache=fmt)
        programs.append(compile_program(_RankNormPack(g), name="kimi_norm_record_probe", operands=(
            Operand("qkv", torch.bfloat16, "[rows,2112]"),
            Operand("qn", torch.bfloat16, "[1536]"), Operand("kn", torch.bfloat16, "[512]"),
            Operand("slots", torch.int64, "[rows]", align=8),
            Operand("qr", torch.bfloat16, "[rows,1536]", "out"),
            Operand("cache", torch.uint8 if fmt == "fp8" else torch.bfloat16,
                    "[slots,656]" if fmt == "fp8" else "[slots,576]", "inout")),
            scalars=(Scalar("rows"),), key=(g,)))
    torch.manual_seed(273)
    rows = 16
    qkv = torch.randn(rows, 2112, device="cuda", dtype=torch.bfloat16)
    qkv[0].zero_()
    qn = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    kn = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int64)
    slots[-1] = -1
    bf16 = torch.zeros(rows, 576, device="cuda", dtype=torch.bfloat16)
    fp8 = torch.full((rows, 656), 173, device="cuda", dtype=torch.uint8)
    residuals = [torch.empty(rows, 1536, device="cuda", dtype=torch.bfloat16) for _ in programs]
    for p, cache, qr in zip(programs, (bf16, fp8), residuals):
        p.launch(qkv, qn, kn, slots, qr, cache, scalars=(rows,))
    assert torch.equal(residuals[0], residuals[1])
    # CUDA scalar /448 can multiply by a rounded reciprocal. CPU packing
    # retains GLM's div.rn scale contract, including quantization midpoints.
    host = bf16.cpu()
    expected = pack_mla_kv_cache_reference(host[:, :512], host[:, 512:]).squeeze(1).to(fp8.device)
    mismatch = (fp8[1:] != expected[1:]).nonzero()
    assert torch.equal(fp8[1:], expected[1:]), [
        (int(r + 1), int(c), int(fp8[r + 1, c]), int(expected[r + 1, c]))
        for r, c in mismatch[:32]]
    assert (fp8[0] == 173).all(), "negative slots must not write a record"


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


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("heads,rank", [(48, 0), (48, 1), (96, 0)])
@pytest.mark.parametrize("mode,rows", [(mode, rows) for mode in ("decode", "verify", "extend")
                                     for rows in (1, 16, 64)] + [("extend", 4096)])
def test_real_mla_gpu_and_graph_replay(heads, rank, mode, rows, kv_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd.kimi_mla import (KimiMlaGeometry, compile_kimi_mla_producer_aot,
        compile_kimi_dense_mla_aot, compile_kimi_mla_output_aot)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    g = KimiMlaGeometry(heads=heads, kv_cache=kv_cache)
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
    cache = (torch.full(((context + 63) // 64, 64, 656), 173, dtype=torch.uint8, device="cuda")
        if kv_cache == "fp8" else
        torch.full(((context + 63) // 64, 64, 576), float("nan"), dtype=torch.bfloat16, device="cuda"))
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
    external_expected = expected
    if kv_cache == "fp8":
        from dataclasses import replace
        from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference
        widened = unpack_mla_kv_cache_reference(cache.flatten(0, 1)).bfloat16().reshape(-1, 64, 576)
        bf16_attention = compile_kimi_dense_mla_aot(replace(g, kv_cache="bf16"),
            max_rows=capacity, max_context=context, mode=mode)
        reference_a, reference_lse = torch.empty_like(a), torch.empty_like(lse)
        bf16_attention.launch(q, widened, table, lengths, cu, reference_a, reference_lse, scratch[1],
            scalars=(rows, 1, table.shape[1], splits))
        if attention.geometry.get("native_fp8_prefill"):
            # The native extend route widens E4M3 exactly and applies the FP32
            # group scale after the MMA, so it is checked against a float32
            # oracle over the dequantized records, not the BF16-rounded reader.
            records = unpack_mla_kv_cache_reference(cache.flatten(0, 1)).squeeze(1).float()
            records = records[physical_slots(prefix + rows)]
            scores = torch.einsum("thd,sd->hts", q.float(), records) * g.softmax_scale
            scores += mask(rows, prefix + rows, "cuda", torch.float32)[0]
            oracle = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
            rel = float((a.float() - oracle).norm() / oracle.norm())
            print(f"native FP8 extend vs dequantized oracle relative_l2={rel:.6g}", flush=True)
            assert rel < 0.005
            torch.testing.assert_close(lse, scores.logsumexp(-1).T, atol=2e-3, rtol=2e-4)
        else:
            assert torch.equal(a.view(torch.uint8), reference_a.view(torch.uint8))
            assert torch.equal(lse.view(torch.uint8), reference_lse.view(torch.uint8))
            print("compact reader equals BF16 reader of widened records byte-for-byte", flush=True)
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
    if kv_cache == "bf16":
        metrics(y, external_expected)
    else:
        rel = float((y.float() - external_expected.float()).norm() / external_expected.float().norm())
        cos = float(F.cosine_similarity(y.float().flatten(), external_expected.float().flatten(), dim=0))
        print(f"FP8 storage delta relative_l2={rel:.8g} cosine={cos:.9g} (model fidelity gates)", flush=True)


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("amplitude", [0.1, 8.0])
@pytest.mark.parametrize("heads", [48, 96])
@pytest.mark.parametrize("context", [64, 256])
def test_ragged_mla_gpu_attention_and_merged_lse(heads, context, amplitude, kv_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_dense_mla_aot
    g = KimiMlaGeometry(heads=heads, kv_cache=kv_cache)
    program = compile_kimi_dense_mla_aot(g, max_rows=16, max_context=context, mode="verify")
    counts = [1, 3, 12]
    lengths = [17, 33, 63] if context == 64 else [65, 129, 200]
    width = (context + 63) // 64
    gen = torch.Generator(device="cuda").manual_seed(3273)
    cache = (torch.randn(3 * width, 64, 576, generator=gen, device="cuda") * 0.1).bfloat16()
    if kv_cache == "fp8":
        from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
        packed = pack_mla_kv_cache_reference(cache.flatten(0, 1)[:, :512], cache.flatten(0, 1)[:, 512:])
        oracle_cache = unpack_mla_kv_cache_reference(packed).bfloat16().reshape_as(cache)
        cache = packed.reshape(3 * width, 64, 656)
    else:
        oracle_cache = cache
    q = (torch.randn(sum(counts), heads, 576, generator=gen, device="cuda") * amplitude).bfloat16()
    # Each request's physical pages are interleaved, with reversed allocation.
    table = torch.arange(3 * width - 1, -1, -1, dtype=torch.int32, device="cuda").reshape(width, 3).T.contiguous()
    offsets = [0, 1, 4, 16]
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out = torch.empty(16, heads, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(16, heads, device="cuda")
    scratch = torch.empty(max(1, program.scratch_bytes(16)["scratch"]), dtype=torch.uint8, device="cuda")
    program.launch(q, cache, table, lens, cu, out, lse, scratch,
                   scalars=(16, 3, width, program.geometry["num_splits"]))
    for request, (count, length) in enumerate(zip(counts, lengths)):
        query = q[offsets[request]:offsets[request + 1]].float()
        records = oracle_cache[table[request].long()].flatten(0, 1)[:length].float()
        scores = torch.einsum("thd,sd->hts", query, records) * g.softmax_scale
        scores += mask(count, length, "cuda", torch.float32)[0]
        expected_lse = scores.logsumexp(-1).T
        expected = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
        metrics(out[offsets[request]:offsets[request + 1]], expected)
        torch.testing.assert_close(lse[offsets[request]:offsets[request + 1]], expected_lse, atol=2e-4, rtol=2e-5)
