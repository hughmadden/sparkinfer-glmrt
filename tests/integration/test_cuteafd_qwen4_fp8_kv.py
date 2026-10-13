"""Immutable token/head E4M3 KV: producer, FP64 direct/split oracles and graph replay."""
from functools import lru_cache

import pytest
import torch

from ..conftest import require_b12x


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


@lru_cache(None)
def _post(kv_format):
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._qwen4_attention_kernels import Qwen4AttnPost
    bf = torch.bfloat16
    specs = (("proj", bf, "[rows,P]", 16), ("q_norm", bf, "[256]", 16),
             ("k_norm", bf, "[256]", 16), ("iq_norm", bf, "[128]", 16),
             ("ik_norm", bf, "[128]", 16), ("rope_positions", torch.int32, "[rows,3]", 4),
             ("kv_slots", torch.int64, "[rows]", 8), ("kv_cache", torch.uint8, "[records,R]", 16),
             ("token_keys", bf, "[records,128]", 16), ("query", bf, "[rows,24,256]", 16),
             ("gate", bf, "[rows,6144]", 16), ("index_q", bf, "[rows,4,128]", 16))
    return compile_program(Qwen4AttnPost(g, kv_format), name="qwen_fp8_kv_post_test",
                           operands=tuple(Operand(n, dt, shape, align=a) for n, dt, shape, a in specs),
                           scalars=(Scalar("rows"),), key=(kv_format,))


def _pack(kv):
    # Match div.rn.f32, not Torch's multiply-by-rounded-reciprocal constant division.
    scales = (kv.double().abs().amax(-1, keepdim=True) / 448).float()
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    inverse = scales.double().reciprocal().float()
    payload = (kv.float() * inverse).to(torch.float8_e4m3fn).view(torch.uint8).flatten(1)
    return torch.cat((payload, scales.flatten(1).contiguous().view(torch.uint8)), 1), scales


def _unpack(records):
    payload = records[:, :1024].contiguous().view(torch.float8_e4m3fn).float().view(-1, 2, 2, 256)
    scales = records[:, 1024:].contiguous().view(torch.float32).view(-1, 2, 2, 1)
    return (payload * scales).bfloat16()


@pytest.mark.parametrize("large", [False, True])
def test_fp8_producer_exact_and_append(large):
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    torch.manual_seed(51)
    rows = 5
    proj = torch.randn(rows, g.attn_in_width, device="cuda").bfloat16()
    proj[-1].zero_()
    norms = [(torch.randn(d, device="cuda") * .1).bfloat16() for d in (256, 256, 128, 128)]
    positions = torch.arange(rows, device="cuda", dtype=torch.int32)[:, None].expand(-1, 3).contiguous()
    base = (1 << 31) // 1040 + 1 if large else 0
    slots = torch.tensor([base + 64, base + 65, -1, base + 67, base + 68], device="cuda", dtype=torch.int64)
    results = []
    for fmt in ("bf16", "fp8"):
        cache = torch.empty(base + 128, g.kv_record_bytes(fmt), device="cuda", dtype=torch.uint8)
        cache[base:].zero_()
        outputs = [torch.empty(base + 128, 128, device="cuda", dtype=torch.bfloat16),
                   torch.empty(rows, 24, 256, device="cuda", dtype=torch.bfloat16),
                   torch.empty(rows, 6144, device="cuda", dtype=torch.bfloat16),
                   torch.empty(rows, 4, 128, device="cuda", dtype=torch.bfloat16)]
        _post(fmt).launch(proj, *norms, positions, slots, cache, *outputs, scalars=[rows])
        results.append((cache, outputs))
    torch.cuda.synchronize()
    touched = slots[slots >= 0]
    bf = results[0][0][touched].contiguous().view(torch.bfloat16).view(-1, 2, 2, 256)
    expected, _ = _pack(bf)
    actual = results[1][0][touched]
    mismatch = (actual != expected).nonzero()
    assert torch.equal(actual, expected), {
        "mismatch": mismatch.cpu().tolist(),
        "actual": actual[mismatch[:, 0], mismatch[:, 1]].cpu().tolist(),
        "expected": expected[mismatch[:, 0], mismatch[:, 1]].cpu().tolist(),
        "scales_equal": torch.equal(actual[:, 1024:], expected[:, 1024:]),
        "bf16": bf.flatten(1)[mismatch[:, 0], mismatch[:, 1].clamp_max(1023)].cpu().tolist(),
    }
    assert torch.count_nonzero(results[1][0][base:base + 64]) == 0
    for a, b in zip(results[0][1][1:], results[1][1][1:]):
        assert torch.equal(a, b)
    before = results[1][0][base:base + 69].clone()
    # A later append must never change previously committed payload or scales.
    slots.fill_(base + 69)
    slots[1:] = -1
    _post("fp8").launch(proj, *norms, positions, slots, results[1][0], *results[1][1], scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(before, results[1][0][base:base + 69])


@lru_cache(None)
def _sparse():
    from b12x.integration.cuteafd.qwen4_attention import compile_qwen4_sparse_gqa_aot
    return compile_qwen4_sparse_gqa_aot(max_rows=128, kv_format="fp8")


@pytest.mark.parametrize("rows,large", [(1, False), (4, False), (65, False), (1, True)])
def test_fp8_sparse_fp64_and_replay(rows, large):
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    torch.manual_seed(321)
    # Cross the 2^31-byte boundary with only tail pages initialized.
    page_base = (1 << 31) // (64 * 1040) + 1 if large else 3
    pages = page_base + 4
    cache = torch.empty(pages * 64, 1040, device="cuda", dtype=torch.uint8)
    kv = torch.randn(256, 2, 2, 256, device="cuda").bfloat16()
    kv *= torch.logspace(-2, 1, 256, device="cuda")[:, None, None, None]
    packed, _ = _pack(kv)
    cache[page_base * 64:pages * 64].copy_(packed)
    query = torch.randn(rows, 24, 256, device="cuda").bfloat16()
    positions = torch.full((rows,), 255, device="cuda", dtype=torch.int64)
    tables = torch.arange(page_base, pages, device="cuda", dtype=torch.int32)
    indices = torch.full((rows, g.sparse_topk), -1, device="cuda", dtype=torch.int32)
    selected = torch.randperm(256, device="cuda")[:131].sort().values
    indices[:, :131] = selected
    out = torch.empty(rows, 24 * 256, device="cuda", dtype=torch.bfloat16)
    program = _sparse()
    scratch = torch.empty(program.scratch_bytes(128)["scratch"], device="cuda", dtype=torch.uint8)
    def run():
        program.launch(query, cache, positions, tables, indices, out, scratch, scalars=[rows, 4, 0, 4])
    run()
    torch.cuda.synchronize()
    actual = out.clone()
    decoded = _unpack(packed)[selected].double()
    keys = decoded[:, 0].repeat_interleave(12, dim=1)
    values = decoded[:, 1].repeat_interleave(12, dim=1)
    scores = torch.einsum("rhd,thd->rht", query.double(), keys) / 16
    expected = torch.einsum("rht,thd->rhd", scores.softmax(-1), values).flatten(1)
    relative = (actual.double() - expected).norm() / expected.norm()
    assert relative < .01, relative.item()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, actual)
    indices.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
