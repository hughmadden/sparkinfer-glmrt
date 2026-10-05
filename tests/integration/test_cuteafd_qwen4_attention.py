"""cuteafd Qwen 3.8 Flash Next full-attention programs versus transformers ``modeling_qwen4_exp``.

Real layer weights from a Qwen3.8-Flash-Next snapshot (``CUTEAFD_QWEN4_SNAPSHOT``;
the attention tensors are BF16 in every published checkpoint), real text
embedded and mixed by the layer's own ``attn_hyper_connection``, then the
reference ``Qwen4ExpTextAttention`` (eager, QSA indexer) against the qwen4
programs: producer -> [index top-k] -> expand -> sparse GQA -> gated o_proj.
Needs the transformers tree that carries ``qwen4_exp`` on ``PYTHONPATH``.

Run as a script for the qualification table::

    python3 tests/integration/test_cuteafd_qwen4_attention.py [--layers 3 47] [--full]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
PAGE = 64


@lru_cache(maxsize=1)
def _weight_map():
    index = SNAPSHOT / "model.safetensors.index.json"
    if not index.is_file():
        return None
    return json.loads(index.read_text())["weight_map"]


def tensor(name: str, device="cuda") -> torch.Tensor:
    weights = _weight_map()
    if weights is None:
        pytest.skip(f"Qwen3.8-Flash-Next snapshot not mounted at {SNAPSHOT}")
    from safetensors import safe_open

    with safe_open(str(SNAPSHOT / weights[name]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _modeling():
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    except ImportError:
        pytest.skip("transformers with qwen4_exp is not importable")
    return ref


@lru_cache(maxsize=1)
def config():
    ref = _modeling()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SNAPSHOT).text_config
    cfg._attn_implementation = "eager"
    del ref
    return cfg


@lru_cache(maxsize=4)
def reference_layer(layer: int):
    """(attention module, hyper-connection module) of ``layer`` with real weights."""
    ref = _modeling()
    cfg = config()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = ref.Qwen4ExpTextAttention(cfg, layer).cuda().eval()
        hc = ref.Qwen4ExpTextGatedResidual(cfg).cuda().eval()
    finally:
        torch.set_default_dtype(torch.float32)
    p = f"{PREFIX}layers.{layer}."
    with torch.no_grad():
        for name, param in attn.named_parameters():
            param.copy_(tensor(p + "self_attn." + name))
        for name, param in hc.named_parameters():
            param.copy_(tensor(p + "attn_hyper_connection." + name))
    return attn, hc


@lru_cache(maxsize=1)
def text_tokens() -> list[int]:
    from tokenizers import Tokenizer

    root = Path(__file__).resolve().parents[2]
    parts = []
    for path in sorted((root / "b12x" / "integration" / "cuteafd").glob("*.py"))[:12]:
        parts.append(path.read_text())
    tok = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    return tok.encode("\n\n".join(parts), add_special_tokens=False).ids


def layer_input(layer: int, tokens: list[int]) -> torch.Tensor:
    """x = attn_hyper_connection(embed repeated over the 4 streams) for ``tokens``."""
    _, hc = reference_layer(layer)
    embed = embedding()
    ids = torch.tensor(tokens, device="cuda")
    with torch.no_grad():
        e = torch.nn.functional.embedding(ids, embed)[None]
        x, _, _ = hc(e.repeat(1, 1, 4))
    return x[0].contiguous()


@lru_cache(maxsize=1)
def embedding() -> torch.Tensor:
    return tensor(PREFIX + "embed_tokens.weight")


def reference(layer: int, x: torch.Tensor):
    """One-shot reference output rows [T, H] and the selected-token mask [T, T] (bool)."""
    ref = _modeling()
    cfg = config()
    attn, _ = reference_layer(layer)
    t = x.shape[0]
    rotary = ref.Qwen4ExpTextRotaryEmbedding(cfg).cuda()
    positions = torch.arange(t, device="cuda").view(1, 1, -1).expand(3, 1, -1)
    with torch.no_grad():
        cos, sin = rotary(x[None], positions)
        mask = torch.full((t, t), torch.finfo(torch.bfloat16).min, device="cuda", dtype=torch.bfloat16)
        mask = torch.triu(mask, 1)[None, None]
        selected = attn.indexer(x[None], (cos, sin), mask, None)
        out, _ = attn(x[None], (cos, sin), mask, None)
    return out[0], (selected[0, 0] == 0)


# ---------------------------------------------------------------------------
# The programs
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def program(name: str, rows: int, **kwargs):
    from b12x.integration.cuteafd import qwen4_attention as qa

    make = {"producer": qa.compile_qwen4_attn_producer_aot, "topk": qa.compile_qwen4_index_topk_aot,
            "sparse_gqa": qa.compile_qwen4_sparse_gqa_aot, "o": qa.compile_qwen4_attn_o_aot}
    if name == "expand":
        return qa.compile_qwen4_index_expand_aot()
    return make[name](max_rows=rows, **kwargs)


def scratch(prog, rows):
    nbytes = max(prog.scratch_bytes(rows).values(), default=256)
    return torch.empty(max(nbytes, 256), dtype=torch.uint8, device="cuda")


class Weights:
    def __init__(self, layer: int):
        p = f"{PREFIX}layers.{layer}.self_attn."
        self.w_in = torch.cat([tensor(p + "q_proj.weight"), tensor(p + "k_proj.weight"), tensor(p + "v_proj.weight"),
                               tensor(p + "indexer.index_qk_proj.weight")], 0).contiguous()
        self.q_norm = tensor(p + "q_norm.weight")
        self.k_norm = tensor(p + "k_norm.weight")
        self.iq_norm = tensor(p + "indexer.q_layernorm.weight")
        self.ik_norm = tensor(p + "indexer.k_layernorm.weight")
        self.w_o = tensor(p + "o_proj.weight")


class Sequence:
    """One sequence's KV pages and pool pages (shuffled physical ids)."""

    def __init__(self, pages: list[int], pool_pages: list[int]):
        self.pages, self.pool_pages, self.len = pages, pool_pages, 0

    def slot(self, p):
        return self.pages[p // PAGE] * PAGE + p % PAGE

    def pool_slot(self, p):
        if p % 4 != 3:
            return -1
        b = p // 4
        return self.pool_pages[b // PAGE] * PAGE + b % PAGE


class Caches:
    def __init__(self, g, pages: int, seed: int = 0):
        self.g = g
        self.pages = pages
        self.pool_pages = (pages + 3) // 4
        self.kv = torch.zeros(pages, PAGE * g.record_bytes, dtype=torch.uint8, device="cuda")
        self.token_keys = torch.zeros(pages * PAGE, 128, dtype=torch.bfloat16, device="cuda")
        self.index = torch.zeros(self.pool_pages * PAGE, 128, dtype=torch.bfloat16, device="cuda")
        gen = torch.Generator().manual_seed(seed)
        self.free = torch.randperm(pages, generator=gen).tolist()
        self.free_pool = torch.randperm(self.pool_pages, generator=gen).tolist()

    def sequence(self, capacity: int) -> Sequence:
        n, m = -(-capacity // PAGE), -(-capacity // (4 * PAGE))
        seq = Sequence(self.free[:n], self.free_pool[:m])
        self.free, self.free_pool = self.free[n:], self.free_pool[m:]
        return seq


def run_step(g, w: Weights, caches: Caches, rows_of: list[tuple[Sequence, int]], x: torch.Tensor, cap: int,
             force_topk: bool = False, timing: dict | None = None):
    """One step: rows_of = [(sequence, count)] (rows contiguous per sequence). Returns attention out and
    (indices, lengths)."""
    t = x.shape[0]
    positions, kv_slots, pool_slots, tables, pools = [], [], [], [], []
    prefill = cap > 64
    for seq, count in rows_of:
        for i in range(count):
            p = seq.len + i
            positions.append(p)
            kv_slots.append(seq.slot(p))
            pool_slots.append(seq.pool_slot(p))
            tables.append(seq.pages)
            pools.append(seq.pool_pages)
    long = any(p + 1 > g.dense_limit for p in positions)
    dev = lambda v, dt: torch.tensor(v, dtype=dt, device="cuda")  # noqa: E731
    pos = dev(positions, torch.int64)
    if prefill:
        assert len(rows_of) == 1
        page_table = dev(rows_of[0][0].pages, torch.int32)
        pool_table = dev(rows_of[0][0].pool_pages, torch.int32)
        stride, pstride, width = 0, 0, len(rows_of[0][0].pages)
    else:
        stride = max(len(p) for p in tables)
        pstride = max(len(p) for p in pools)
        page_table = dev([p + [0] * (stride - len(p)) for p in tables], torch.int32)
        pool_table = dev([p + [0] * (pstride - len(p)) for p in pools], torch.int32)
        width = stride
    n, d = g.heads, g.head_dim
    query = torch.empty(t, n, d, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(t, n * d, dtype=torch.bfloat16, device="cuda")
    index_q = torch.empty(t, g.index_heads, 128, dtype=torch.bfloat16, device="cuda")
    attn = torch.empty(t, n * d, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(t, g.hidden, dtype=torch.bfloat16, device="cuda")
    blocks = torch.full((t, g.index_blocks), -1, dtype=torch.int32, device="cuda")
    indices = torch.empty(t, g.sparse_topk, dtype=torch.int32, device="cuda")
    lengths = torch.empty(t, dtype=torch.int32, device="cuda")
    producer, sparse, o = program("producer", cap), program("sparse_gqa", cap), program("o", cap)
    topk, expand = program("topk", cap), program("expand", 0)
    sp, ss, so, st = scratch(producer, cap), scratch(sparse, cap), scratch(o, cap), scratch(topk, cap)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record()
    rope_pos = pos.to(torch.int32)[:, None].expand(-1, 3).contiguous()
    block_rope_pos = (pos - pos % 4).to(torch.int32)[:, None].expand(-1, 3).contiguous()
    producer.launch(x, w.w_in, w.q_norm, w.k_norm, w.iq_norm, w.ik_norm, pos, rope_pos, block_rope_pos,
                    dev(kv_slots, torch.int64),
                    dev(pool_slots, torch.int64), caches.kv, caches.token_keys, caches.index, query, gate, index_q,
                    sp, scalars=[t])
    events[1].record()
    if long or force_topk:
        topk.launch(index_q, pos, caches.index, pool_table, blocks, st, scalars=[t, pstride])
    events[2].record()
    expand.launch(pos, blocks, indices, lengths, scalars=[t])
    events[3].record()
    sparse.launch(query, caches.kv, pos, page_table, indices, attn, ss, scalars=[t, width, stride])
    events[4].record()
    o.launch(attn, gate, w.w_o, out, so, scalars=[t])
    events[5].record()
    torch.cuda.synchronize()
    if timing is not None:
        for k, name in enumerate(("producer", "topk", "expand", "sparse_gqa", "o")):
            timing[name] = events[k].elapsed_time(events[k + 1])
    for seq, count in rows_of:
        seq.len += count
    return out, indices, lengths


def selection_agreement(indices: torch.Tensor, ref_mask: torch.Tensor, first: int) -> float:
    """Fraction of rows whose selected token set equals the reference's."""
    same = 0
    for r in range(indices.shape[0]):
        ours = indices[r][indices[r] >= 0].long()
        mask = torch.zeros(ref_mask.shape[1], dtype=torch.bool, device=indices.device)
        mask[ours] = True
        same += int(torch.equal(mask, ref_mask[first + r]))
    return same / max(indices.shape[0], 1)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _g():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT

    return QWEN38_FLASH_NEXT


def case_prefill(layer: int, t: int, report: list):
    g = _g()
    tokens = text_tokens()
    assert len(tokens) >= t, f"need {t} tokens of text, have {len(tokens)}"
    x = layer_input(layer, tokens[:t])
    ref_out, ref_sel = reference(layer, x)
    w = Weights(layer)
    caches = Caches(g, pages=-(-t // PAGE) + 8, seed=t)
    seq = caches.sequence(t)
    out, indices, _ = run_step(g, w, caches, [(seq, t)], x, 4096)
    agree = selection_agreement(indices, ref_sel, 0) if t > g.dense_limit - 1 else 1.0
    report.append((f"L{layer} prefill T={t}", cosine(out, ref_out), rel_l2(out, ref_out), agree))
    return out, ref_out


def case_decode(layer: int, n: int, steps: int, report: list):
    """Prefill n rows, then `steps` single-row decode steps, then one 4-row verify step (m64)."""
    g = _g()
    t = n + steps + 4
    tokens = text_tokens()[:t]
    x = layer_input(layer, tokens)
    ref_out, ref_sel = reference(layer, x)
    w = Weights(layer)
    caches = Caches(g, pages=-(-t // PAGE) + 8, seed=n)
    seq = caches.sequence(t)
    run_step(g, w, caches, [(seq, n)], x[:n], 4096)
    outs, agree = [], []
    for i in range(steps):
        out, indices, _ = run_step(g, w, caches, [(seq, 1)], x[n + i:n + i + 1], 64)
        outs.append(out)
        agree.append(selection_agreement(indices, ref_sel, n + i))
    ours = torch.cat(outs)
    ref = ref_out[n:n + steps]
    report.append((f"L{layer} decode {steps}x1 after {n}", cosine(ours, ref), rel_l2(ours, ref),
                   sum(agree) / len(agree)))
    out, indices, _ = run_step(g, w, caches, [(seq, 4)], x[n + steps:], 64)
    ref = ref_out[n + steps:]
    report.append((f"L{layer} verify 4 rows at {n + steps}", cosine(out, ref), rel_l2(out, ref),
                   selection_agreement(indices, ref_sel, n + steps)))


def case_chunked(layer: int, n: int, m: int, report: list):
    """Prefill n rows, then a second prefill chunk of m rows on the same sequence (m4096)."""
    g = _g()
    t = n + m
    x = layer_input(layer, text_tokens()[:t])
    ref_out, ref_sel = reference(layer, x)
    w = Weights(layer)
    caches = Caches(g, pages=-(-t // PAGE) + 8, seed=m)
    seq = caches.sequence(t)
    run_step(g, w, caches, [(seq, n)], x[:n], 4096)
    out, indices, _ = run_step(g, w, caches, [(seq, m)], x[n:], 4096)
    report.append((f"L{layer} prefill {m} rows after {n}", cosine(out, ref_out[n:]), rel_l2(out, ref_out[n:]),
                   selection_agreement(indices, ref_sel, n)))


def case_two_sequences(layer: int, n1: int, n2: int, report: list):
    g = _g()
    tokens = text_tokens()
    xa = layer_input(layer, tokens[:n1 + 2])
    xb = layer_input(layer, tokens[500:500 + n2 + 1])
    ra, _ = reference(layer, xa)
    rb, _ = reference(layer, xb)
    w = Weights(layer)
    caches = Caches(g, pages=-(-(n1 + n2) // PAGE) + 16, seed=7)
    a, b = caches.sequence(n1 + 2), caches.sequence(n2 + 1)
    run_step(g, w, caches, [(a, n1)], xa[:n1], 4096)
    run_step(g, w, caches, [(b, n2)], xb[:n2], 4096)
    out, _, _ = run_step(g, w, caches, [(a, 2), (b, 1)], torch.cat([xa[n1:], xb[n2:]]), 64)
    ref = torch.cat([ra[n1:], rb[n2:]])
    report.append((f"L{layer} 2 seqs (2 rows @{n1}, 1 row @{n2})", cosine(out, ref), rel_l2(out, ref), 1.0))


# pytest entry points (short cases; the script runs the full table).

@pytest.fixture(autouse=True)
def _exact_reference():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_prefill_short():
    report = []
    case_prefill(3, 64, report)
    assert report[-1][1] > 0.9999, report


def test_prefill_topk():
    report = []
    case_prefill(3, 2600, report)
    assert report[-1][1] > 0.999 and report[-1][3] > 0.95, report


def test_decode_and_verify():
    report = []
    case_decode(3, 2100, 4, report)
    assert all(r[1] > 0.999 for r in report), report


def test_chunked_prefill():
    report = []
    case_chunked(3, 1501, 1100, report)
    assert report[-1][1] > 0.9999 and report[-1][3] > 0.95, report


def test_two_sequences():
    report = []
    case_two_sequences(3, 300, 700, report)
    assert report[-1][1] > 0.9999, report


def bench(report_lines: list):
    """Decode rows=1 at contexts 1K/8K/32K; 4096-row prefill at context 0 and 4096 (random caches)."""
    g = _g()
    w = Weights(3)
    for ctx in (1024, 8192, 32768):
        caches = Caches(g, pages=-(-ctx // PAGE) + 64, seed=1)
        seq = caches.sequence(ctx + 64)
        caches.kv.random_(0, 64)  # small finite BF16 bit patterns are fine for timing
        caches.index.normal_()
        caches.token_keys.normal_()
        seq.len = ctx
        x = torch.randn(1, g.hidden, dtype=torch.bfloat16, device="cuda")
        timing = {}
        for _ in range(3):
            seq.len = ctx
            run_step(g, w, caches, [(seq, 1)], x, 64, timing=timing)
        report_lines.append(f"decode rows=1 ctx {ctx}: " + ", ".join(f"{k} {v:.3f}" for k, v in timing.items())
                            + f" ms (total {sum(timing.values()):.3f})")
    for ctx in (0, 4096):
        caches = Caches(g, pages=-(-(ctx + 4096) // PAGE) + 64, seed=2)
        seq = caches.sequence(ctx + 4096)
        caches.kv.random_(0, 64)
        caches.index.normal_()
        caches.token_keys.normal_()
        x = torch.randn(4096, g.hidden, dtype=torch.bfloat16, device="cuda") * 0.1
        timing = {}
        for _ in range(3):
            seq.len = ctx
            run_step(g, w, caches, [(seq, 4096)], x, 4096, timing=timing)
        report_lines.append(f"prefill 4096 rows at ctx {ctx}: " + ", ".join(f"{k} {v:.2f}" for k, v in timing.items())
                            + f" ms (total {sum(timing.values()):.2f})")


def export_all(out_dir: Path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd import qwen4_attention as qa

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    with exportable_compilation():
        programs = [("qwen4_index_expand", qa.compile_qwen4_index_expand_aot())]
        for rows in (64, 4096):
            programs += [(f"qwen4_attn_producer_m{rows}", qa.compile_qwen4_attn_producer_aot(max_rows=rows)),
                         (f"qwen4_index_topk_m{rows}", qa.compile_qwen4_index_topk_aot(max_rows=rows)),
                         (f"qwen4_sparse_gqa_m{rows}", qa.compile_qwen4_sparse_gqa_aot(max_rows=rows)),
                         (f"qwen4_attn_o_m{rows}", qa.compile_qwen4_attn_o_aot(max_rows=rows))]
        for stem, prog in programs:
            prog.export_to_c(out_dir, stem, "cuteafd_" + stem)
            info = validate_exported_header(prog, out_dir / f"{stem}.h", "cuteafd_" + stem)
            made.append((stem, info["argument_count"], prog.scratch_bytes(64 if "m64" in stem else 4096)))
    return made


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="+", default=[3])
    p.add_argument("--full", action="store_true", help="all prefill sizes, decode, verify, two sequences")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--export", type=Path)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    report = []
    sizes = [1, 5, 64, 1000, 2051, 2052, 3000, 4096] if a.full else [5, 64, 2600]
    for layer in a.layers:
        for t in sizes:
            start = time.time()
            case_prefill(layer, t, report)
            print(f"{report[-1]}  ({time.time() - start:.1f}s)", flush=True)
            torch.cuda.empty_cache()
        if a.full:
            case_decode(layer, 1500, 3, report)
            print(report[-2:], flush=True)
            case_decode(layer, 2300, 6, report)
            print(report[-2:], flush=True)
            case_two_sequences(layer, 300, 2200, report)
            print(report[-1], flush=True)
            case_chunked(layer, 1501, 1100, report)
            print(report[-1], flush=True)
    print("\ncase                                        cosine    rel_l2    selection==ref")
    for name, c, r, agree in report:
        print(f"{name:42s} {c:.6f}  {r:.2e}  {agree:.4f}")
    if a.bench:
        lines = []
        bench(lines)
        print("\n".join(lines))
    if a.export:
        for stem, argc, sb in export_all(a.export):
            print(f"exported {stem}: {argc} args, scratch {sb}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
