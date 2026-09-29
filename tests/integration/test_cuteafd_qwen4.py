"""Qualify the qwen4 hyper-connection, head, shared-expert, router and PLE programs
against transformers' modeling_qwen4_exp modules with real weights.

Inputs are real: tokenized text, embedded, repeated into the four streams
(the layer-0 input); sublayer outputs are Gaussian at the scale of a real
attention output. PLE runs one prefill and then single-row steps carrying
its conv state, compared with the one-shot reference rows.

  PYTHONPATH=.:<transformers>/src USE_HUB_KERNELS=0 python3 tests/integration/test_cuteafd_qwen4.py
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch

SNAPSHOT = Path(os.environ.get(
    "CUTEAFD_QWEN4_SNAPSHOT",
    "/mnt/sparknest/hf-home/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/"
    "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
))
PREFIX = "model.language_model."
TEXT = Path(os.environ.get("CUTEAFD_QWEN4_TEXT", "/home/tj/Developer/cuteafd-glmf/runs/glmf-golden/prompt.txt"))
SHARD_ROWS = 2_500_012


@lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    path = SNAPSHOT / "model.safetensors.index.json"
    if not path.is_file():
        pytest.skip(f"Qwen 3.8 Flash Next snapshot not mounted at {SNAPSHOT}")
    return json.loads(path.read_text())["weight_map"]


def tensor(name: str, device="cuda") -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(SNAPSHOT / _index()[name]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def report(name: str, ours: torch.Tensor, ref: torch.Tensor) -> float:
    c = cosine(ours, ref)
    exact = float((ours.float() == ref.float()).float().mean())
    print(f"  {name:34s} cosine {c:.7f} rel_l2 {rel(ours, ref):.2e} bit-equal {exact:.4f}", flush=True)
    return c


@lru_cache(maxsize=1)
def config():
    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(SNAPSHOT).text_config
    text._attn_implementation = "eager"
    return text


@lru_cache(maxsize=1)
def tokens(limit: int = 1024) -> tuple[int, ...]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    return tuple(tok.encode(TEXT.read_text(), add_special_tokens=False).ids[:limit])


def streams_for(t: int) -> torch.Tensor:
    ids = torch.tensor(tokens()[:t], device="cuda")
    embed = torch.nn.functional.embedding(ids, tensor(PREFIX + "embed_tokens.weight"))
    return embed.repeat(1, 4).contiguous()  # [t, 4H]


def gated_residual(site: str, layer: int | None, use_combine: bool = True):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    torch.set_default_dtype(torch.bfloat16)
    module = Qwen4ExpTextGatedResidual(config(), use_combine=use_combine).cuda()
    torch.set_default_dtype(torch.float32)
    prefix = f"{PREFIX}{site}." if layer is None else f"{PREFIX}layers.{layer}.{site}."
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(tensor(prefix + name))
    return module, prefix


def hc_operands(prefix: str, inject: bool = True):
    down = tensor(prefix + "input_mix_weight_down.weight")
    w_di = torch.cat([down, tensor(prefix + "block_inject_weight.weight")]) if inject else down
    return tensor(prefix + "hc_norm.weight"), w_di.contiguous(), tensor(prefix + "input_mix_weight_up.weight")


@lru_cache(maxsize=None)
def programs():
    from b12x.integration.cuteafd import qwen4

    return {
        "pre": qwen4.compile_qwen4_hc_pre_aot(),
        "post_pre": qwen4.compile_qwen4_hc_post_pre_aot(),
        "post": qwen4.compile_qwen4_hc_post_aot(),
        "head": qwen4.compile_qwen4_head_aot(),
        "shared": qwen4.compile_qwen4_shared_aot(),
        "router": qwen4.compile_qwen4_router_scores_aot(),
        "add": qwen4.compile_qwen4_add_aot(),
    }


def scratch(program, rows: int) -> torch.Tensor:
    size = max(program.scratch_bytes(rows).get("scratch", 0), 256)
    return torch.empty(size, dtype=torch.uint8, device="cuda")


ROWS = (1, 7, 64, 1024)


def test_hc_pre_post_head():
    p = programs()
    h = config().hidden_size
    worst = 1.0
    for rows in ROWS:
        print(f"rows {rows}")
        s = streams_for(rows)
        attn, prefix = gated_residual("attn_hyper_connection", 0)
        norm, w_di, w_up = hc_operands(prefix)
        y = torch.empty(rows, h, dtype=torch.bfloat16, device="cuda")
        inject = torch.empty(rows, 4, dtype=torch.bfloat16, device="cuda")
        p["pre"].launch(s, norm, w_di, w_up, y, inject, scratch(p["pre"], rows), scalars=[rows])
        with torch.no_grad():
            ref_y, _, ref_inj = attn(s[None])
        worst = min(worst, report("hc_pre y", y, ref_y[0]), report("hc_pre inject", inject, ref_inj[0]))
        # A sublayer output, posted, then layer 0's MLP site.
        out = (torch.randn(rows, h, device="cuda") * 0.05).bfloat16()
        mlp, prefix = gated_residual("mlp_hyper_connection", 0)
        norm, w_di, w_up = hc_operands(prefix)
        residual_out = torch.empty_like(s)
        y2 = torch.empty_like(y)
        inj2 = inject.clone()
        p["post_pre"].launch(out, s, inj2, norm, w_di, w_up, residual_out, y2, scratch(p["post_pre"], rows),
                             scalars=[rows])
        with torch.no_grad():
            ref_s = s + (out.unsqueeze(-2) * ref_inj[0].unsqueeze(-1)).flatten(-2)
            ref_y2, _, ref_inj2 = mlp(ref_s[None])
        worst = min(worst, report("hc_post streams", residual_out, ref_s),
                    report("hc_post_pre y", y2, ref_y2[0]), report("hc_post_pre inject", inj2, ref_inj2[0]))
        final = torch.empty_like(s)
        p["post"].launch(out, s, inject, final, scalars=[rows])
        worst = min(worst, report("hc_post (last)", final, ref_s))
        mixer, prefix = gated_residual("hyper_connection_mixer", None, use_combine=False)
        norm, w_down, w_up = hc_operands(prefix, inject=False)
        head = torch.empty_like(y)
        p["head"].launch(s, norm, w_down, w_up, head, scratch(p["head"], rows), scalars=[rows])
        with torch.no_grad():
            ref_head = mixer(s[None])
        worst = min(worst, report("head (mixer)", head, ref_head[0]))
    assert worst > 0.9999


def test_shared_router():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock

    p = programs()
    h = config().hidden_size
    layer = 5
    prefix = f"{PREFIX}layers.{layer}.mlp."
    gate = tensor(prefix + "shared_expert.gate_proj.weight")
    up = tensor(prefix + "shared_expert.up_proj.weight")
    sg = tensor(prefix + "shared_expert_gate.weight")
    w_gate_up = torch.cat([gate, up, sg, torch.zeros(1296 - 1281, h, dtype=torch.bfloat16, device="cuda")])
    w_down = tensor(prefix + "shared_expert.down_proj.weight")
    router = tensor(prefix + "gate.weight")
    worst = 1.0
    for rows in ROWS:
        print(f"rows {rows}")
        x = (torch.randn(rows, h, device="cuda") * 0.5).bfloat16()
        out = torch.empty(rows, h, dtype=torch.bfloat16, device="cuda")
        p["shared"].launch(x, w_gate_up.contiguous(), w_down, out, scratch(p["shared"], rows), scalars=[rows])
        mlp = torch.nn.functional.linear
        with torch.no_grad():
            hidden = torch.nn.functional.silu(mlp(x, gate)) * mlp(x, up)
            ref = torch.sigmoid(mlp(x, sg)) * mlp(hidden, w_down)
        worst = min(worst, report("shared expert", out, ref))
        logits = torch.empty(rows, 512, dtype=torch.float32, device="cuda")
        p["router"].launch(x, router, logits, scalars=[rows])
        ref_logits = x.float() @ router.float().T
        worst = min(worst, report("router logits (FP32)", logits, ref_logits))
        total = torch.empty_like(out)
        p["add"].launch(out, x, total, scalars=[rows])
        worst = min(worst, report("add", total, out + x))
    del Qwen4ExpTextSparseMoeBlock
    assert worst > 0.9999


# ---------------------------------------------------------------------------
# PLE
# ---------------------------------------------------------------------------


def ple_module():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    cfg = config()
    layer = cfg.ple_layer_ids[0] - 1
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        module = Qwen4ExpTextPLELayer(cfg, layer, 0)
        # The ~95 GiB n-gram table is never materialized: rows are gathered per test.
        module.ple_embedding.ngram_embedding = torch.nn.Embedding(1, cfg.ple_embed_dim // 16)
    torch.set_default_dtype(torch.float32)
    module = module.to_empty(device="cuda")
    prefix = f"{PREFIX}layers.{layer}.ple."
    emb = module.ple_embedding
    with torch.no_grad():
        for name, param in list(module.named_parameters()) + list(module.named_buffers()):
            if name.startswith("ple_embedding.ngram_embedding"):
                continue
            param.copy_(tensor(prefix + name).reshape(param.shape))
    return module, prefix, emb


def ngram_ids(emb, ids: torch.Tensor) -> torch.Tensor:
    """[T, 16] global table rows, the module's own hashing (no cache, one sequence)."""
    history = torch.cat([ids.new_full((1, emb.context_len), emb.eos_token_id), ids[None]], dim=-1)
    shifted = [emb._shift_right_ignore_eos(history, s) for s in range(emb.ngram_size)]
    blocks = []
    for ngram in range(2, emb.ngram_size + 1):
        start = (ngram - 2) * emb.heads_per_ngram
        mixed = shifted[0] * emb.layer_multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * emb.layer_multipliers[position])
        sizes = emb.ngram_heads_vocab_sizes[start:start + emb.heads_per_ngram]
        offsets = emb.ngram_heads_offsets[start:start + emb.heads_per_ngram]
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[0, -ids.shape[0]:]


def table_rows(prefix: str, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The distinct table rows ``rows`` needs as a compact table plus the remapped ids."""
    from safetensors import safe_open

    unique, inverse = torch.unique(rows.flatten(), return_inverse=True)
    out = []
    fp8 = None
    for row in unique.tolist():
        shard, local = divmod(row, SHARD_ROWS)
        name = f"{prefix}ple_embedding.ngram_embedding.shard_{shard}.weight"
        with safe_open(str(SNAPSHOT / _index()[name]), framework="pt", device="cpu") as handle:
            out.append(handle.get_slice(name)[local:local + 1])
    table = torch.cat(out).cuda()
    scale_name = f"{prefix}ple_embedding.ngram_embedding.weight_scale"
    if scale_name in _index():
        fp8 = tensor(scale_name)
    return table, fp8, inverse.reshape(rows.shape)


def test_ple():
    p_bf16 = None
    module, prefix, emb = ple_module()
    t = 300
    ids = torch.tensor(tokens()[:t], device="cuda")
    rows = ngram_ids(emb, ids)
    table, scale, local = table_rows(prefix, rows)
    from b12x.integration.cuteafd import qwen4

    fp8 = table.dtype == torch.float8_e4m3fn
    program = qwen4.compile_qwen4_ple_aot(fp8=fp8)
    dense = (table.to(torch.bfloat16) * scale).to(torch.bfloat16) if fp8 else table
    # Reference: the module with its embedding patched to the compact table.
    ref_emb = dense[local].flatten(-2)
    emb.forward = lambda input_ids, past_key_values: ref_emb[None]
    s = streams_for(t) + (torch.randn(t, 4 * 2560, device="cuda") * 0.02).bfloat16()
    with torch.no_grad():
        ref = s[None] + module(s[None], ids[None], None)
    c = 4 * 2560
    w_kv = torch.cat([tensor(prefix + "key_proj.weight"), tensor(prefix + "value_proj.weight")]).contiguous()
    conv_w = tensor(prefix + "conv1d.weight").float().reshape(c, 4).contiguous()
    norms = [tensor(prefix + f"{n}.weight") for n in ("norm_key", "norm_query", "norm_conv")]
    scale_t = (scale.float() if fp8 else torch.ones(1, device="cuda")).reshape(1)
    state = torch.zeros(2, 9, c, dtype=torch.bfloat16, device="cuda")
    ours = s.clone()
    # Prefill the first `split` rows, then single-row steps (slot 1), carrying the conv state.
    split = 200
    def run(first: int, n: int):
        slots = torch.full((n,), 1, dtype=torch.int32, device="cuda")
        seq_first = torch.zeros(n, dtype=torch.int32, device="cuda")
        chunk = ours[first:first + n].clone()
        program.launch(chunk, local[first:first + n].contiguous(), table, scale_t, w_kv, *norms, conv_w, state,
                       slots, seq_first, scratch(program, n), scalars=[n])
        ours[first:first + n] = chunk
    t0 = time.perf_counter()
    run(0, split)
    for r in range(split, t):
        run(r, 1)
    torch.cuda.synchronize()
    print(f"PLE ({'fp8' if fp8 else 'bf16'} table, {table.shape[0]} distinct rows) "
          f"{(time.perf_counter() - t0) * 1e3:.1f} ms for 1 prefill + {t - split} steps")
    c1 = report("PLE prefill rows", ours[:split], ref[0, :split])
    c2 = report("PLE decode rows", ours[split:], ref[0, split:])
    del p_bf16
    assert min(c1, c2) > 0.9999


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    test_hc_pre_post_head()
    test_shared_router()
    test_ple()
