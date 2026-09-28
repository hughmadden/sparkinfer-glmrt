"""cuteafd DSV4 compressed sparse MLA AOT programs vs b12x.attention.compressed_sparse_mla."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, cosine, has_checkpoint

SWA_PAGE_BYTES = 149_760
# A physical main-cache page past the 2^31-byte line (AGENTS.md big-pid rule).
HIGH_PAGE = (2**31) // SWA_PAGE_BYTES + 3
LAYER = 2


def _page_bytes(rows: int) -> int:
    return (rows * 584 + 575) // 576 * 576


def _pack(tokens: int, page_rows: int, pages: int, gen, device):
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_sparse_mla_kv_cache_reference as pack,
    )

    nope = (torch.randn((tokens, 448), generator=gen) * 1.5).to(device)
    rope = torch.randn((tokens, 64), generator=gen).bfloat16().to(device)
    return pack(nope, rope, page_size=page_rows, num_pages=pages)


def _sink(heads, device):
    if heads == 64 and has_checkpoint():
        return checkpoint_tensor(f"layers.{LAYER}.attn.attn_sink").float().contiguous()
    gen = torch.Generator(device="cpu").manual_seed(heads)
    return torch.randn((heads,), generator=gen).to(device)


class _Case:
    """One prefill chunk of ``rows`` queries at positions start..start+rows-1."""

    def __init__(self, heads, rows, start, ratio, width, seed, pool):
        device = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        context = start + rows
        pages = (context + 255) // 256
        packed = _pack(context, 256, pages, gen, device)
        # Park logical page 0 past the 2^31-byte line; the rest stay low.
        physical = torch.arange(pages, dtype=torch.int64) + 1
        physical[0] = HIGH_PAGE
        pool.zero_()
        for logical, page in enumerate(physical.tolist()):
            pool[page].copy_(packed[logical])
        self.swa_cache = pool
        positions = torch.arange(start, context, dtype=torch.int64)
        window = positions[:, None] - 127 + torch.arange(128)[None, :]
        valid = window >= 0
        slots = physical[window.clamp_min(0) // 256] * 256 + window.clamp_min(0) % 256
        self.swa_indices = torch.where(valid, slots, -1).to(torch.int32).to(device)
        self.swa_lengths = valid.sum(dim=1).to(torch.int32).to(device)
        # Left-pack valid window slots like model.py get_window_topk_idxs.
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
        self.swa_indices = torch.gather(self.swa_indices.cpu(), 1, order).to(device)
        self.q = (torch.randn((rows, heads, 512), generator=gen)).bfloat16().to(device)
        self.sink = _sink(heads, device)
        self.width = width
        self.page_rows = {4: 64, 128: 2}.get(ratio, 0)
        if width:
            groups = max(context // ratio, 1)
            cpages = (groups + self.page_rows - 1) // self.page_rows
            self.indexed_cache = _pack(groups, self.page_rows, cpages + 1, gen, device)
            visible = ((positions + 1) // ratio).to(torch.int32)
            if ratio == 4:
                keys = torch.rand((rows, groups), generator=gen)
                keys[torch.arange(groups)[None, :] >= visible[:, None]] = -1.0
                top = torch.topk(keys, min(width, groups), dim=1).indices.to(torch.int32)
                lengths = torch.clamp(visible, max=width)
                idx = torch.full((rows, width), -1, dtype=torch.int32)
                cols = torch.arange(width)[None, :]
                idx[:, : top.shape[1]] = torch.where(cols[:, : top.shape[1]] < lengths[:, None], top, -1)
            else:
                cols = torch.arange(width, dtype=torch.int32)[None, :]
                idx = torch.where(cols < visible[:, None], cols, -1).to(torch.int32)
                lengths = torch.clamp(visible, max=width)
            self.indexed_indices = idx.contiguous().to(device)
            self.indexed_lengths = lengths.to(torch.int32).to(device)
        else:
            self.indexed_cache = self.indexed_indices = self.indexed_lengths = None


def _prepared(case, mode, max_rows):
    from b12x.attention import compressed_sparse_mla as mla
    from ._cuteafd import prepare

    rows, heads, _ = case.q.shape
    out = torch.empty_like(case.q)
    width = 128 + case.width
    caps = mla.Caps(
        device=case.q.device, num_q_heads=heads, max_q_rows=max_rows, max_width=width,
        max_page_table_width=width, max_batch=max_rows, max_kv_rows=max_rows * width, mode=mode,
        swa_width=128, indexed_width=case.width, swa_page_size=256,
        indexed_page_size=case.page_rows or 256,
    )
    q_cap = torch.empty((max_rows, heads, 512), dtype=torch.bfloat16, device=case.q.device)
    q_cap[:rows] = case.q
    invocation = mla.invocation_from_tensors(q=q_cap, swa_k_cache=case.swa_cache,
                                             indexed_k_cache=case.indexed_cache,
                                             attn_sink=case.sink, out=out)
    plan = prepare(mla.plan(caps, invocation=invocation), f"test.mla.{mode}.{max_rows}.{case.width}")
    spec = plan.scratch_specs()[0]
    binding = mla.bind(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=case.q.device),
        q=q_cap[:rows], swa_indices=case.swa_indices, swa_lengths=case.swa_lengths,
        indexed_indices=case.indexed_indices, indexed_lengths=case.indexed_lengths)
    mla.run(plan=plan, binding=binding, swa_k_cache=case.swa_cache,
            indexed_k_cache=case.indexed_cache, attn_sink=case.sink, sm_scale=512 ** -0.5, out=out)
    return out


def _aot(program, case, *, stream=None):
    rows = case.q.shape[0]
    out = torch.full_like(case.q, float("nan"))
    scratch = torch.full((program.scratch_bytes(max(rows, program.geometry["max_rows"]))["scratch"],),
                         0xFF, dtype=torch.uint8, device=case.q.device)
    program.launch(case.q, case.swa_cache, case.swa_indices, case.swa_lengths,
                   case.indexed_cache, case.indexed_indices, case.indexed_lengths, case.sink,
                   out, scratch, scalars=(rows,), stream=stream)
    return out, scratch


@pytest.fixture(scope="module")
def pool():
    require_b12x()
    return torch.zeros((HIGH_PAGE + 1, SWA_PAGE_BYTES), dtype=torch.uint8, device="cuda")


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd.dsv4_sparse_mla import compile_dsv4_sparse_mla_aot as c

    with exportable_compilation():
        return {
            ("prefill", 0): c(FLASH, route="prefill"),
            ("prefill", 4): c(FLASH, route="prefill", indexed_width=512, indexed_page_rows=64),
            ("prefill", 128): c(FLASH, route="prefill", indexed_width=64, indexed_page_rows=2),
            ("decode", 0): c(FLASH, route="decode", max_rows=8),
            ("decode", 4): c(FLASH, route="decode", max_rows=8, indexed_width=512, indexed_page_rows=64),
            ("decode", 128): c(FLASH, route="decode", max_rows=8, indexed_width=64, indexed_page_rows=2),
            ("pro_prefill", 4): c(PRO, route="prefill", indexed_width=1024, indexed_page_rows=64),
            ("pro_decode", 4): c(PRO, route="decode", max_rows=4, indexed_width=1024, indexed_page_rows=64),
        }


CASES = [
    # (route, ratio, width, rows, start)
    ("prefill", 0, 0, 1, 0), ("prefill", 0, 0, 37, 300), ("prefill", 0, 0, 200, 0),
    ("prefill", 4, 512, 5, 2100), ("prefill", 4, 512, 150, 1900),
    ("prefill", 128, 64, 3, 4000), ("prefill", 128, 64, 90, 5000),
    ("decode", 0, 0, 1, 700), ("decode", 0, 0, 8, 10),
    ("decode", 4, 512, 1, 3000), ("decode", 4, 512, 6, 800),
    ("decode", 128, 64, 1, 7000), ("decode", 128, 64, 8, 300),
]


@pytest.mark.parametrize(("route", "ratio", "width", "rows", "start"), CASES)
def test_matches_prepared_compressed_sparse_mla(programs, pool, route, ratio, width, rows, start):
    require_b12x()
    case = _Case(64, rows, start, ratio, width, seed=rows * 31 + start, pool=pool)
    program = programs[(route, ratio)]
    mode = "extend" if route == "prefill" else "decode"
    max_rows = rows if route == "prefill" else program.geometry["max_rows"]
    expected = _prepared(case, mode, max_rows)
    actual, _ = _aot(program, case)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual.float()).all())
    assert_bitwise(actual, expected, f"{route} C{ratio} out")


@pytest.mark.parametrize(("key", "route", "rows"), [(("pro_prefill", 4), "prefill", 33),
                                                   (("pro_decode", 4), "decode", 3)])
def test_pro_heads(programs, pool, key, route, rows):
    require_b12x()
    case = _Case(128, rows, 4500, 4, 1024, seed=77 + rows, pool=pool)
    program = programs[key]
    max_rows = rows if route == "prefill" else program.geometry["max_rows"]
    expected = _prepared(case, "extend" if route == "prefill" else "decode", max_rows)
    actual, _ = _aot(program, case)
    torch.cuda.synchronize()
    assert_bitwise(actual, expected, f"pro {route} out")


def test_graph_replay_and_live_rows(programs, pool):
    require_b12x()
    program = programs[("decode", 4)]
    case = _Case(64, 8, 2500, 4, 512, seed=5, pool=pool)
    out = torch.empty_like(case.q)
    scratch = torch.empty((program.scratch_bytes(8)["scratch"],), dtype=torch.uint8, device="cuda")
    args = (case.q, case.swa_cache, case.swa_indices, case.swa_lengths, case.indexed_cache,
            case.indexed_indices, case.indexed_lengths, case.sink, out, scratch)
    program.launch(*args, scalars=(8,))
    torch.cuda.synchronize()
    eager = out.clone()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(*args, scalars=(8,))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(out, eager, "graph out")
    # Fewer live rows under the same compiled program touch only those rows.
    out.fill_(float("nan"))
    program.launch(*args, scalars=(3,))
    torch.cuda.synchronize()
    assert_bitwise(out[:3], eager[:3], "live-row prefix")
    assert bool(torch.isnan(out[3:].float()).all())


@pytest.mark.parametrize("key", [("prefill", 0), ("prefill", 4), ("decode", 128)])
def test_export_to_c_signature(programs, key, tmp_path):
    stem = "dsv4_sparse_mla_" + "_".join(map(str, key))
    info = check_export(programs[key], tmp_path, stem)
    assert info["argument_count"] == len(programs[key].operands) + 2


@pytest.mark.parametrize("route", ["prefill", "decode"])
def test_matches_fp32_dequantized_reference(programs, pool, route):
    """Oracle sanity: union softmax with sink over dequantized FP8 records."""
    from b12x.attention._shared.mla.compressed_reference import (
        gather_compressed_sparse_mla_kv_cache_reference as gather,
    )

    require_b12x()
    case = _Case(64, 6, 2500, 4, 512, seed=3, pool=pool)
    actual, _ = _aot(programs[(route, 4)], case)
    expected = torch.empty_like(actual, dtype=torch.float32)
    for r in range(6):
        k1, v1 = gather(pool, case.swa_indices[r, : int(case.swa_lengths[r])], page_size=256)
        k2, v2 = gather(case.indexed_cache, case.indexed_indices[r, : int(case.indexed_lengths[r])],
                        page_size=64)
        keys, values = torch.cat((k1, k2)), torch.cat((v1, v2))
        scores = (case.q[r].float() @ keys.t()) * 512 ** -0.5
        probs = torch.softmax(torch.cat((scores, case.sink[:, None]), 1), 1)[:, :-1]
        expected[r] = probs @ values
    # Decode's FP8 QK is coarser than the prefill route's BF16 QK.
    assert cosine(actual, expected) > (0.9995 if route == "prefill" else 0.998)
