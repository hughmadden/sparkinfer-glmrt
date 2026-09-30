"""Qualify the qwen4_gdn AOT program against transformers' Qwen4ExpTextGatedDeltaNet.

Real layer weights (the GDN tensors are BF16 in every Qwen 3.8 Flash Next
checkpoint) and real inputs: tokenized text, embedded, through the layer's own
attn_hyper_connection. Cases: one-step prefills (token-sequential and
chunked), prefill then single-row decode steps, a multi-sequence decode step
and a 4-row verify step, comparing outputs and final recurrent state.

  PYTHONPATH=.:<transformers>/src USE_HUB_KERNELS=0 python3 tests/integration/test_cuteafd_qwen4_gdn.py
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


class Layer:
    """One GDN layer: the transformers reference module and the program operands."""

    def __init__(self, layer: int):
        from transformers import AutoConfig
        from transformers.models.qwen4_exp import modeling_qwen4_exp as ref

        self.ref = ref
        config = AutoConfig.from_pretrained(SNAPSHOT).text_config
        config._attn_implementation = "eager"
        assert config.layer_types[layer] == "linear_attention"
        p = f"{PREFIX}layers.{layer}."
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            self.hc = ref.Qwen4ExpTextGatedResidual(config)
            self.gdn = ref.Qwen4ExpTextGatedDeltaNet(config, layer)
        torch.set_default_dtype(torch.float32)
        with torch.no_grad():
            for name, param in self.hc.named_parameters():
                param.copy_(tensor(p + "attn_hyper_connection." + name))
            for name, param in self.gdn.named_parameters():
                param.copy_(tensor(p + "linear_attn." + name).reshape(param.shape))
        a = lambda n: tensor(p + "linear_attn." + n)  # noqa: E731
        self.ops = {
            "w_in": torch.cat([a("in_proj_qkv.weight"), a("in_proj_z.weight"), a("in_proj_b.weight"),
                               a("in_proj_a.weight")], 0).contiguous(),
            "conv_w": a("conv1d.weight").float().reshape(-1, 4).contiguous(),
            "a_log": a("A_log").float().contiguous(),
            "dt_bias": a("dt_bias").float().contiguous(),
            "norm_w": a("norm.weight").contiguous(),
            "w_out": a("out_proj.weight").contiguous(),
        }
        self.config = config

    def inputs(self, tokens: list[int]) -> torch.Tensor:
        """x for these tokens: embedding streams through the layer's attn_hyper_connection."""
        ids = torch.tensor(tokens, device="cuda")
        embed = torch.nn.functional.embedding(ids, tensor(PREFIX + "embed_tokens.weight"))
        with torch.inference_mode():
            x, _, _ = self.hc(embed.repeat(1, self.config.hc_count)[None])
        return x[0].contiguous()

    def reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output [T,H] and final recurrent state [heads, v, k] of one cache-free prefill."""
        ref = self.ref
        captured = {}
        original = ref.torch_chunk_gated_delta_rule

        def capture(*args, **kwargs):
            kwargs["output_final_state"] = True
            out, state = original(*args, **kwargs)
            captured["state"] = state
            return out, state

        ref.torch_chunk_gated_delta_rule = capture
        try:
            with torch.inference_mode():
                out = self.gdn(x[None])[0]
        finally:
            ref.torch_chunk_gated_delta_rule = original
        # transformers keeps [B, heads, k, v]; the program keeps [heads, v, k].
        return out, captured["state"][0].transpose(-1, -2).contiguous()


@lru_cache(maxsize=4)
def program(max_rows: int):
    from b12x.integration.cuteafd.qwen4_gdn import compile_qwen4_gdn_aot

    return compile_qwen4_gdn_aot(max_rows=max_rows)


class Pools:
    def __init__(self, slots: int, rows: int, g=None):
        from b12x.integration.cuteafd import QWEN38_FLASH_NEXT

        g = g or QWEN38_FLASH_NEXT
        self.g = g
        self.conv = torch.zeros(slots, 3, g.gdn_conv_width, dtype=torch.bfloat16, device="cuda")
        self.state = torch.zeros(slots, g.gdn_value_heads, 128, 128, dtype=torch.float32, device="cuda")


def run(prog, layer: Layer, pools: Pools, x: torch.Tensor, slots: list[int], seq_first: list[int]) -> torch.Tensor:
    rows = x.shape[0]
    out = torch.empty(rows, pools.g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    o = layer.ops
    # Decode capacities carry the speculative replay record (unused with spec 0).
    spec = [op.name for op in prog.operands].count("replay")
    prog.launch(x, o["w_in"], o["conv_w"], o["a_log"], o["dt_bias"], o["norm_w"], o["w_out"], pools.conv,
                pools.state, torch.tensor(slots, dtype=torch.int32, device="cuda"),
                torch.tensor(seq_first, dtype=torch.int32, device="cuda"), out, *([None] * spec), scratch,
                scalars=[rows] + [0] * spec)
    return out


def tokens(count: int) -> list[int]:
    from tokenizers import Tokenizer

    text = TEXT.read_text() if TEXT.is_file() else "The quick brown fox jumps over the lazy dog. " * 400
    ids = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    while len(ids) < count:
        ids = ids + ids
    return ids[:count]


def qualify(layer_id: int, report: list) -> None:
    layer = Layer(layer_id)
    xs = layer.inputs(tokens(4096))
    big, small = program(4096), program(64)

    def row(case, ours, theirs, state=None, theirs_state=None):
        entry = (layer_id, case, cosine(ours, theirs), rel(ours, theirs),
                 None if state is None else cosine(state, theirs_state),
                 None if state is None else rel(state, theirs_state))
        report.append(entry)
        return entry

    # (a) one-step prefills.
    for t in (1, 17, 64, 65, 300, 1500, 4096):
        x = xs[:t].contiguous()
        ref_out, ref_state = layer.reference(x)
        pools = Pools(2, t)
        prog = big if t > 64 else small
        out = run(prog, layer, pools, x, [1] * t, [0] * t)
        row(f"prefill T={t}{' (chunked)' if t > 64 else ''}", out, ref_out, pools.state[1], ref_state)
    # (b) prefill N (chunked) then single-row decode steps.
    n, steps = 1000, 24
    ref_out, ref_state = layer.reference(xs[:n + steps].contiguous())
    pools = Pools(1, n)
    run(big, layer, pools, xs[:n].contiguous(), [0] * n, [0] * n)
    outs = [run(small, layer, pools, xs[n + i:n + i + 1].contiguous(), [0], [0]) for i in range(steps)]
    row(f"prefill {n} + {steps} decode rows", torch.cat(outs), ref_out[n:], pools.state[0], ref_state)
    # (c) multi-sequence decode step: sequence A (slot 0, prefix 40 of xs) and B (slot 1, prefix 70 of xs[2000:]).
    pools = Pools(2, 64)
    a_x, b_x = xs[:41].contiguous(), xs[2000:2071].contiguous()
    run(small, layer, pools, a_x[:40].contiguous(), [0] * 40, [0] * 40)
    run(big, layer, pools, b_x[:70].contiguous(), [1] * 70, [0] * 70)
    step = torch.cat([a_x[40:41], b_x[70:71]]).contiguous()
    out = run(small, layer, pools, step, [0, 1], [0, 1])
    ra, sa = layer.reference(a_x)
    rb, sb = layer.reference(b_x)
    row("2-sequence decode step", out, torch.cat([ra[-1:], rb[-1:]]),
        torch.stack([pools.state[0], pools.state[1]]), torch.stack([sa, sb]))
    # (d) 4-row verify step after a 200-row prefill; a third sequence's 2 rows ride along.
    pools = Pools(3, 64)
    run(big, layer, pools, xs[:200].contiguous(), [2] * 200, [0] * 200)
    run(small, layer, pools, xs[3000:3010].contiguous(), [0] * 10, [0] * 10)
    step = torch.cat([xs[200:204], xs[3010:3012]]).contiguous()
    out = run(small, layer, pools, step, [2, 2, 2, 2, 0, 0], [0, 0, 0, 0, 4, 4])
    r1, s1 = layer.reference(xs[:204].contiguous())
    r2, s2 = layer.reference(xs[3000:3012].contiguous())
    row("4-row verify + 2-row seq", out, torch.cat([r1[-4:], r2[-2:]]),
        torch.stack([pools.state[2], pools.state[0]]), torch.stack([s1, s2]))


def timing(layer_id: int = 0) -> list:
    """Median launch time (CUDA events, 20 launches) per live row count."""
    layer = Layer(layer_id)
    xs = layer.inputs(tokens(4096))
    o = layer.ops
    results = []
    for rows, prog in ((1, program(64)), (4, program(64)), (64, program(64)), (4096, program(4096))):
        pools = Pools(1, rows)
        x = xs[:rows].contiguous()
        out = torch.empty(rows, pools.g.hidden, dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
        slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
        first = torch.zeros(rows, dtype=torch.int32, device="cuda")

        spec = [op.name for op in prog.operands].count("replay")

        def launch():
            prog.launch(x, o["w_in"], o["conv_w"], o["a_log"], o["dt_bias"], o["norm_w"], o["w_out"], pools.conv,
                        pools.state, slots, first, out, *([None] * spec), scratch, scalars=[rows] + [0] * spec)

        for _ in range(3):
            launch()
        times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        results.append((rows, times[len(times) // 2]))
    return results


@pytest.mark.parametrize("layer_id", [0, 44])
def test_qwen4_gdn_matches_transformers(layer_id):
    report: list = []
    qualify(layer_id, report)
    for entry in report:
        assert entry[2] >= 0.9999, entry
        if entry[4] is not None:
            assert entry[4] >= 0.9999, entry


def test_qwen4_gdn_exports(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.qwen4_gdn import compile_qwen4_gdn_aot

    for rows in (64, 4096):
        with exportable_compilation():
            prog = compile_qwen4_gdn_aot(max_rows=rows)
        stem = f"qwen4_gdn_m{rows}"
        prog.export_to_c(str(tmp_path), stem, "cuteafd_" + stem)
        validate_exported_header(prog, tmp_path / f"{stem}.h", "cuteafd_" + stem)


if __name__ == "__main__":
    import sys

    torch.backends.cuda.matmul.allow_tf32 = False
    report: list = []
    for layer_id in [int(v) for v in sys.argv[1:]] or [0, 44]:
        qualify(layer_id, report)
    print(f"{'layer':>5} {'case':34s} {'out cos':>10} {'out rel':>9} {'state cos':>10} {'state rel':>9}")
    for layer_id, case, c, r, sc, sr in report:
        s = "" if sc is None else f"{sc:10.6f} {sr:9.2e}"
        print(f"{layer_id:5d} {case:34s} {c:10.6f} {r:9.2e} {s}")
    for rows, ms in timing():
        print(f"timing rows={rows}: {ms:.3f} ms")
