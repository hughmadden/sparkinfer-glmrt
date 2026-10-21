"""Shared MLA draft ABI and numerical visibility/rotary contracts."""
from __future__ import annotations

import ast
import __future__
import inspect
import json
import math
import os
from pathlib import Path
import types

import pytest
import torch

from b12x.integration.cuteafd import kimi_mla as mla


SCORE_SCALE = 192 ** -0.5 * 1.3465735902799727 ** 2


def geometry(**kwargs):
    return mla.MlaDraftGeometry(**kwargs)


def test_draft_geometry_is_explicit_and_target_unchanged():
    target = mla.KimiMlaGeometry()
    assert (target.heads, target.norm_eps, target.softmax_scale) == (96, 1e-6, 192 ** -0.5)
    for args in ({"heads": 48}, {"norm_eps": 1e-6}, {"window_size": 2048}):
        with pytest.raises(ValueError):
            mla.MlaDraftGeometry(**args)
    assert geometry().record_bytes == target.record_bytes == 1152
    assert geometry(kv_cache="fp8").record_bytes == mla.KimiMlaGeometry(kv_cache="fp8").record_bytes == 656
    for cls in (mla.KimiMlaGeometry, mla.MlaDraftGeometry):
        with pytest.raises(ValueError, match="cache must be"):
            cls(kv_cache="int8")


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_draft_abi_and_capacity_not_live_counts(monkeypatch, kv_cache):
    monkeypatch.setattr(mla, "compile_program", lambda launch, **kwargs: (launch, kwargs))
    g = geometry(kv_cache=kv_cache)
    producer, p = mla.compile_mla_draft_producer_aot(g, max_rows=64)
    output, o = mla.compile_mla_draft_output_aot(g, max_rows=64)
    attention, a = mla.compile_mla_draft_attention_aot(g, max_rows=64, max_context=8192)
    assert p["geometry"]["rotary_style"] == "interleaved"
    assert p["geometry"]["kv_cache"] == a["geometry"]["kv_cache"] == kv_cache
    assert attention.forward.ds_mla_record == (kv_cache == "fp8")
    cache_operand = next(v for v in p["operands"] if v.name == "kv_cache")
    assert cache_operand.dtype == (torch.uint8 if kv_cache == "fp8" else torch.bfloat16)
    assert [v.name for v in p["scalars"]] == ["rotary_factor", "rows"]
    assert [v.name for v in p["operands"]][2:4] == ["positions", "inv_freq"]
    assert not any(v.name in ("x", "w_g") for v in o["operands"])
    assert not hasattr(output, "gate_proj")
    assert o["geometry"]["output_gate"] is False
    assert a["geometry"]["noncausal_block"] is True
    assert [v.name for v in a["scalars"]][-1] == "attention_scale"
    assert "attention_scale" not in a["geometry"]
    assert attention.forward.noncausal_block is True
    assert attention.forward.window_size is None
    assert o["scratch"]["scratch"](9) == mla._align(9 * 64 * 128 * 2)
    for compile_fn in (mla.compile_mla_draft_producer_aot, mla.compile_mla_draft_output_aot,
                       mla.compile_mla_draft_attention_aot):
        assert "rows" not in inspect.signature(compile_fn).parameters
        with pytest.raises(TypeError):
            compile_fn(mla.KimiMlaGeometry())


@pytest.mark.parametrize("window", [None, 4096])
@pytest.mark.parametrize("high_pages", [False, True])
@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("query_amplitude", [0.1, 8.0])
def test_draft_attention_gpu_ragged_noncausal_window_and_graph(window, high_pages, kv_cache, query_amplitude):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    g = geometry(window_size=window, kv_cache=kv_cache)
    p = mla.compile_mla_draft_attention_aot(g, max_rows=16, max_context=4352)
    counts, lengths = [8, 3], [4101, 65]
    width = 68
    generator = torch.Generator(device="cuda").manual_seed(273)
    cache = (torch.randn(2 * width, 64, 576, generator=generator, device="cuda") * 0.1).bfloat16()
    if kv_cache == "fp8":
        from b12x.attention._shared.mla.reference import (
            pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference)
        packed = pack_mla_kv_cache_reference(cache.flatten(0, 1)[:, :512], cache.flatten(0, 1)[:, 512:])
        oracle_cache = unpack_mla_kv_cache_reference(packed).bfloat16().reshape_as(cache)
        cache = packed.reshape(2 * width, 64, 656)
    else:
        oracle_cache = cache
    q = (torch.randn(11, 64, 576, generator=generator, device="cuda") * query_amplitude).bfloat16()
    if high_pages:
        # Tail pages beyond the signed-32-bit byte-offset boundary. Allocate
        # uninitialized prefix storage, never initialize the whole large pool.
        first_page = 2 ** 31 // (64 * g.record_bytes) + 1
        pool = torch.empty(first_page + 2 * width, *cache.shape[1:], device="cuda", dtype=cache.dtype)
        pool[first_page:].copy_(cache)
        cache = pool
    else:
        first_page = 0
    table = (torch.arange(2 * width - 1, -1, -1, dtype=torch.int32, device="cuda") + first_page).reshape(width, 2).T.contiguous()
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 8, 11], dtype=torch.int32, device="cuda")
    out = torch.empty(11, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(11, 64, device="cuda")
    scratch = torch.empty(p.scratch_bytes(16)["scratch"], dtype=torch.uint8, device="cuda")

    def run():
        p.launch(q, cache, table, lens, cu, out, lse, scratch,
                 scalars=(11, 2, width, p.geometry["num_splits"], SCORE_SCALE))

    run()
    torch.cuda.synchronize()
    for index, (count, length, begin) in enumerate(zip(counts, lengths, [0, 8])):
        records = oracle_cache[(table[index] - first_page).long()].flatten(0, 1)[:length].float()
        scores = torch.einsum("thd,sd->hts", q[begin:begin + count].float(), records) * SCORE_SCALE
        if window:
            positions = torch.arange(length, device="cuda")
            low = length - count + torch.arange(count, device="cuda") - window + 1
            scores.masked_fill_(positions[None, None] < low[None, :, None], -torch.inf)
        expected = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
        torch.testing.assert_close(out[begin:begin + count].float(), expected, atol=0.003, rtol=0.03)
        torch.testing.assert_close(lse[begin:begin + count], scores.logsumexp(-1).T, atol=2e-4, rtol=2e-5)
    assert torch.isfinite(out).all() and out.abs().sum() > 0
    saved = out.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    for actual, expected in zip((out, lse), saved):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_draft_real_weights_gpu_hf_eager_absorption_and_output(kv_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snapshot = hf / "hub/models--lightseekorg--kimi-k3-dflash2/snapshots/e77935fb4804e17eb55085bffd045eae1d779769"
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("DFlash2 checkpoint not mounted")
    # Execute only the pinned, reviewed HF eager attention definition, not
    # checkpoint Python. Keeping the oracle expanded detects absorption errors.
    hf_source = Path(os.environ.get("CUTEAFD_HF_MLA_REFERENCE",
        "/workspace/cuteafd/third_party/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py"))
    tree = ast.parse(hf_source.read_text())
    names = {"eager_attention_forward", "repeat_kv"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in tree.body} == names
    ns = {"torch": torch, "nn": torch.nn}
    exec(compile(tree, str(hf_source), "exec", flags=__future__.annotations.compiler_flag), ns)
    config = json.loads((snapshot / "config.json").read_text())
    rope = config["rope_parameters"]
    pos_freq = rope["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64)
    correction = lambda beta: 64 * math.log(rope["original_max_position_embeddings"] / (beta * 2 * math.pi)) / (2 * math.log(rope["rope_theta"]))
    low, high = max(0, math.floor(correction(rope["beta_fast"]))), min(63, math.ceil(correction(rope["beta_slow"])))
    ramp = 1 - ((torch.arange(32).float() - low) / (high - low)).clamp(0, 1)
    inv = ((1 / (rope["factor"] * pos_freq)) * (1 - ramp) + (1 / pos_freq) * ramp).cuda()
    mscale = lambda m: 1 + 0.1 * m * math.log(rope["factor"])
    factor = mscale(rope["mscale"]) / mscale(rope["mscale_all_dim"])
    scale = 192 ** -0.5 * mscale(rope["mscale_all_dim"]) ** 2
    g = mla.MlaDraftGeometry(kv_cache=kv_cache)
    from safetensors import safe_open
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as tensors:
        w = {name: tensors.get_tensor("layers.0.self_attn." + name).cuda() for name in (
            "q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_a_layernorm.weight",
            "kv_a_layernorm.weight", "q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight")}
    uk, uv = w["kv_b_proj.weight"].reshape(64, 256, 512).split(128, dim=1)
    uk, uv = uk.transpose(1, 2).contiguous(), uv.contiguous()
    combined = torch.cat((w["q_a_proj.weight"], w["kv_a_proj_with_mqa.weight"])).contiguous()
    producer = mla.compile_mla_draft_producer_aot(g, max_rows=64)
    attention = mla.compile_mla_draft_attention_aot(g, max_rows=64, max_context=128)
    output = mla.compile_mla_draft_output_aot(g, max_rows=64)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = (torch.randn(16, 7168, device="cuda") * 0.2).bfloat16()
    slots = torch.arange(16, device="cuda", dtype=torch.int64)
    positions = slots + 32760
    cache = (torch.full((2, 64, 656), 173, device="cuda", dtype=torch.uint8)
             if kv_cache == "fp8" else
             torch.full((2, 64, 576), float("nan"), device="cuda", dtype=torch.bfloat16))
    query = torch.empty(16, 64, 576, device="cuda", dtype=torch.bfloat16)
    residual = torch.empty(16, 1536, device="cuda", dtype=torch.bfloat16)
    attn = torch.empty(16, 64, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(16, 64, device="cuda")
    out = torch.empty(16, 7168, device="cuda", dtype=torch.bfloat16)
    scratch = [torch.empty(p.scratch_bytes(16)["scratch"], device="cuda", dtype=torch.uint8) for p in (producer, attention, output)]
    lengths = torch.tensor([16], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)

    def run():
        producer.launch(x, slots, positions, inv, combined, w["q_a_layernorm.weight"], w["kv_a_layernorm.weight"],
                        w["q_b_proj.weight"], uk, cache, query, residual, scratch[0], scalars=(factor, 16))
        attention.launch(query, cache, table, lengths, cu, attn, lse, scratch[1],
                         scalars=(16, 1, 2, attention.geometry["num_splits"], scale))
        output.launch(attn, uv, w["o_proj.weight"], out, scratch[2], scalars=(16,))

    run()
    norm = lambda value, weight: weight * (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16()
    qa = norm(torch.nn.functional.linear(x, w["q_a_proj.weight"]), w["q_a_layernorm.weight"])
    q = torch.nn.functional.linear(qa, w["q_b_proj.weight"]).reshape(16, 64, 192)
    kv = torch.nn.functional.linear(x, w["kv_a_proj_with_mqa.weight"])
    latent = norm(kv[:, :512], w["kv_a_layernorm.weight"])
    phase = positions.float()[:, None] * inv[None]
    c, s = (phase.cos() * factor).bfloat16(), (phase.sin() * factor).bfloat16()

    def rotate(value):
        a, b = value[..., 0::2], value[..., 1::2]
        rotated = torch.empty_like(value)
        rotated[..., 0::2] = a * c[:, None] - b * s[:, None]
        rotated[..., 1::2] = b * c[:, None] + a * s[:, None]
        return rotated

    q = torch.cat((q[..., :128], rotate(q[..., 128:])), -1)
    keys = torch.nn.functional.linear(latent, w["kv_b_proj.weight"]).reshape(16, 64, 256)
    k = torch.cat((keys[..., :128], rotate(kv[:, None, 512:]).expand(-1, 64, -1)), -1)
    expected, _ = ns["eager_attention_forward"](types.SimpleNamespace(num_key_value_groups=1, training=False),
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], keys[..., 128:].transpose(0, 1)[None], None, scale)
    expected = torch.nn.functional.linear(expected.reshape(16, 8192), w["o_proj.weight"])
    relative = (out.float() - expected.float()).norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0)
    if kv_cache == "bf16":
        assert relative < 0.03 and cosine > 0.9995
    else:
        from dataclasses import replace
        from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference
        widened = unpack_mla_kv_cache_reference(cache.flatten(0, 1)).bfloat16().reshape(-1, 64, 576)
        bf16_attention = mla.compile_mla_draft_attention_aot(replace(g, kv_cache="bf16"),
            max_rows=64, max_context=128)
        reference_attn, reference_lse = torch.empty_like(attn), torch.empty_like(lse)
        bf16_attention.launch(query, widened, table, lengths, cu, reference_attn, reference_lse, scratch[1],
            scalars=(16, 1, 2, attention.geometry["num_splits"], scale))
        assert torch.equal(attn.view(torch.uint8), reference_attn.view(torch.uint8))
        assert torch.equal(lse.view(torch.uint8), reference_lse.view(torch.uint8))
        print(f"draft compact/BF16 widened-record parity exact; FP8 storage delta "
              f"relative_l2={float(relative):.8g} cosine={float(cosine):.9g} (model fidelity gates)", flush=True)
    assert torch.isfinite(out).all() and out.abs().sum() > 0
    saved = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert torch.equal(out.view(torch.uint8), saved.view(torch.uint8))
    # Restore raw records across a partial-page frontier, then replay a
    # continuation against the saved native cache, query and output bytes.
    frontier = cache.clone()
    slots.add_(16)
    lengths.fill_(32)
    positions.add_(16)
    x.mul_(0.5)
    run()
    assert torch.equal(cache.flatten(0, 1)[:16].view(torch.uint8),
                       frontier.flatten(0, 1)[:16].view(torch.uint8))
    continued = [v.clone() for v in (cache, query, residual, out, lse)]
    cache.copy_(frontier)
    run()
    for actual, expected in zip((cache, query, residual, out, lse), continued):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_rotary_gpu_interleaved_table_and_negative_slots(kv_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    g = geometry(kv_cache=kv_cache)
    kernel = mla._DraftRotary(g)
    p = compile_program(kernel, name="test_mla_draft_rotary", operands=(
        Operand("query", torch.bfloat16, "[rows,64,576]", "inout"),
        Operand("cache", torch.uint8 if kv_cache == "fp8" else torch.bfloat16,
                "[pages,64,656]" if kv_cache == "fp8" else "[pages,64,576]", "inout"),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("inv_freq", torch.float32, "[32]", align=4)),
        scalars=(Scalar("factor", "float32"), Scalar("rows")), key=(g,))
    q = torch.randn(4, 64, 576, device="cuda").bfloat16()
    cache = torch.randn(64, 576, device="cuda").bfloat16()
    if kv_cache == "fp8":
        from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
        cache = pack_mla_kv_cache_reference(cache[:, :512], cache[:, 512:]).squeeze(1)
    slots = torch.tensor([0, 63, -1, 11], device="cuda", dtype=torch.int64)
    positions = torch.tensor([0, 32767, 32768, 1048575], device="cuda", dtype=torch.int64)
    # Deliberately non-default frequencies ensure the kernel consumes the loader table.
    inv = (1.0 / (50000.0 ** (torch.arange(32, device="cuda").float() / 32))) / 7
    before_q, before_cache = q.clone(), cache.clone()
    factor = 1.125
    p.launch(q, cache, slots, positions, inv, scalars=(factor, 4))
    phase = positions.float()[:, None] * inv[None]
    c, s = (phase.cos() * factor).bfloat16(), (phase.sin() * factor).bfloat16()

    def rotate(x, cos, sin):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x1 * cos + x0 * sin
        return out

    torch.testing.assert_close(q[..., 512:], rotate(before_q[..., 512:], c[:, None], s[:, None]), atol=0.015625, rtol=0.01)
    assert torch.equal(q[..., :512], before_q[..., :512])
    if kv_cache == "fp8":
        assert torch.equal(cache[:, :528], before_cache[:, :528])
        tail = cache[:, 528:].contiguous().view(torch.bfloat16)
        before_tail = before_cache[:, 528:].contiguous().view(torch.bfloat16)
    else:
        tail, before_tail = cache[:, 512:], before_cache[:, 512:]
    expected_tail = before_tail.clone()
    for row, slot in enumerate([0, 63, -1, 11]):
        if slot >= 0:
            expected_tail[slot] = rotate(before_tail[slot], c[row], s[row])
    torch.testing.assert_close(tail, expected_tail, atol=0.015625, rtol=0.01)
