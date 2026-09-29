"""cuteafd GLM 5.x DSA indexer AOT programs vs the transformers GlmMoeDsaIndexer.

``glm_index_producer``: the FP8 query/key and folded head weights against the
reference BF16 tensors (and a torch emulation of the same quantization), and
the resulting index scores against the reference scores. ``glm_index_topk``:
selection against ``torch.topk`` of the scores of the same FP8 cache, with
causal lengths below and above the 2048 budget.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import config, cos_sin, cosine, golden_rows, reference_module, reference_rope, rms_norm, tensor
from .test_cuteafd_glm_dense_aot import rope_interleaved

DECODE, PREFILL = 64, 4096
MAX_CONTEXT = 131072
MAX_PAGES = MAX_CONTEXT // 64
_PROGRAMS: dict = {}


def _program(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_attention, glm_indexer

        fn = {"index_producer": glm_attention.compile_glm_index_producer_aot,
              "topk": glm_indexer.compile_glm_index_topk_aot}[kind]
        _PROGRAMS[key] = fn(GLM53, **kw)
    return _PROGRAMS[key]


@pytest.fixture(scope="module")
def attn0():
    require_b12x()
    return reference_module("GlmMoeDsaAttention", config(), 0, prefix="model.layers.0.self_attn.")


def index_weights(attn):
    ix = attn.indexer
    return dict(w_iq=ix.wq_b.weight.contiguous(),
                w_ik=torch.cat([ix.wk.weight, ix.weights_proj.weight.to(torch.bfloat16)], 0).contiguous(),
                k_norm_w=ix.k_norm.weight.contiguous(), k_norm_b=ix.k_norm.bias.contiguous())


def _fp8(x: torch.Tensor):
    """Torch emulation of the program's quantizer: per last-dim amax/448 scale."""
    amax = x.float().abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def split_index_cache(cache: torch.Tensor, slots: torch.Tensor):
    """(E4M3 keys as FP32 [n,128], FP32 scales [n]) at physical slots."""
    pages, rows = slots // 64, slots % 64
    keys = cache[:, :8192].view(-1, 64, 128)[pages, rows].view(torch.float8_e4m3fn).float()
    scales = cache[:, 8192:].contiguous().view(torch.float32).view(-1, 64)[pages, rows]
    return keys, scales


@pytest.mark.parametrize("capacity,rows", [(DECODE, 1), (DECODE, 16), (DECODE, 64), (PREFILL, 4096)])
def test_glm_index_producer(attn0, capacity, rows):
    program = _program("index_producer", max_rows=capacity)
    w = index_weights(attn0)
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    gen = torch.Generator(device="cpu").manual_seed(rows + 1)
    positions = torch.randperm(8192, generator=gen)[:rows].to(torch.int64).cuda()
    pages = (rows + 63) // 64 + 3
    slots = torch.randperm(pages * 64, generator=gen)[:rows].to(torch.int64).cuda()
    cache = torch.zeros((pages, 8448), dtype=torch.uint8, device="cuda")
    q_fp8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
    head_weights = torch.empty((rows, 32), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        q_resid = attn0.q_a_layernorm(attn0.q_a_proj(x))
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(x, q_resid, positions, slots, cos_sin(8192), w["w_iq"], w["w_ik"], w["k_norm_w"],
                   w["k_norm_b"], cache, q_fp8, head_weights, scratch, scalars=(rows,))
    ix = attn0.indexer
    with torch.no_grad():
        cos, sin = reference_rope(positions)
        q = ix.wq_b(q_resid).view(rows, 32, 128)
        q_ref = torch.cat([rope_interleaved(q[..., :64].transpose(0, 1), cos[0], sin[0]).transpose(0, 1),
                           q[..., 64:]], -1)
        k = ix.k_norm(ix.wk(x))
        k_ref = torch.cat([rope_interleaved(k[:, :64], cos[0], sin[0]), k[:, 64:]], -1)
        w_ref = ix.weights_proj(x.to(ix.weights_proj.weight.dtype)).float() * 32 ** -0.5
    torch.cuda.synchronize()
    keys, k_scales = split_index_cache(cache, slots)
    q8_emul, q_scale_emul = _fp8(q_ref)
    k8_emul, k_scale_emul = _fp8(k_ref)
    # Query side as the scorer consumes it: q8 * folded weight.
    a_prog = q_fp8.float() * head_weights[..., None]
    a_ref = q_ref.float() * (w_ref * 128 ** -0.5)[..., None]
    a_emul = q8_emul.float() * (w_ref * 128 ** -0.5)[..., None] * q_scale_emul
    k_prog = keys * k_scales[:, None]
    c_q, c_q_emul = cosine(a_prog, a_ref), cosine(a_prog, a_emul)
    c_k, c_k_emul = cosine(k_prog, k_ref), cosine(k_prog, k8_emul.float() * k_scale_emul)
    q_bytes = float((q_fp8.view(torch.uint8) != q8_emul.view(torch.uint8)).float().mean())
    k_bytes = float((keys.to(torch.float8_e4m3fn).view(torch.uint8) != k8_emul.view(torch.uint8)).float().mean())
    # Index scores over the batch's own keys (all visible).
    n = min(rows, 1024)
    s_prog = torch.einsum("ihd,jd->ihj", q_fp8[:n].float(), keys[:n]).relu()
    s_prog = (s_prog * k_scales[None, None, :n] * head_weights[:n, :, None]).sum(1)
    s_ref = (torch.einsum("ihd,jd->ihj", q_ref[:n].float(), k_ref[:n].float()) * 128 ** -0.5).relu()
    s_ref = (s_ref * w_ref[:n, :, None]).sum(1)
    c_s = cosine(s_prog, s_ref)
    print(f"glm_index_producer m{capacity} rows={rows}: q*w {c_q:.6f} (vs fp8 emulation {c_q_emul:.7f}, "
          f"bytes differ {q_bytes:.1e}) k {c_k:.6f} (emulation {c_k_emul:.7f}, bytes {k_bytes:.1e}) "
          f"scores {c_s:.6f}")
    assert c_q_emul >= 0.9999 and c_k_emul >= 0.9999
    assert c_q >= 0.999 and c_k >= 0.999 and c_s >= 0.999


def _index_scores(q8, weights, keys, scales):
    """Scores [rows, n] from dequantized FP8 operands (the top-k contract)."""
    out = []
    for start in range(0, q8.shape[0], 256):
        s = torch.einsum("ihd,jd->ihj", q8[start:start + 256].float(), keys).relu()
        out.append((s * weights[start:start + 256, :, None]).sum(1) * scales[None])
    return torch.cat(out)


def _random_cache(pages, gen):
    cache = torch.zeros((pages, 8448), dtype=torch.uint8)
    keys = (torch.randn((pages, 64, 128), generator=gen) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
    cache[:, :8192] = keys.view(torch.uint8).view(pages, 8192)
    scales = torch.rand((pages, 64), generator=gen) * 0.02 + 0.01
    cache[:, 8192:] = scales.view(torch.uint8).view(pages, 256)
    return cache.cuda()


@pytest.mark.parametrize("mode,capacity,rows", [("decode", DECODE, 1), ("decode", DECODE, 16),
                                                ("decode", DECODE, 64), ("prefill", PREFILL, 4096)])
def test_glm_index_topk(mode, capacity, rows):
    require_b12x()
    program = _program("topk", max_rows=capacity, max_pages=MAX_PAGES, mode=mode)
    gen = torch.Generator(device="cpu").manual_seed(rows + 11)
    q8 = (torch.randn((rows, 32, 128), generator=gen) * 3).to(torch.float8_e4m3fn).cuda()
    weights = (torch.randn((rows, 32), generator=gen) * 0.05).cuda()
    if mode == "prefill":
        context = 4096  # the first chunk: causal lengths 1..4096
        width = context // 64
        pool = width + 5
        table = torch.randperm(pool, generator=gen)[:width].to(torch.int32).cuda()
        lengths = (torch.arange(rows, dtype=torch.int32) + context - rows + 1).cuda()
        page_table = table
        stride = 0
        tables = table[None].expand(rows, -1)
    else:
        lengths = torch.randint(1, 12000, (rows,), generator=gen, dtype=torch.int32)
        lengths[0] = 2048  # exactly the budget
        if rows > 2:
            lengths[1] = 777  # fewer visible rows than the budget
            lengths[2] = 2049
        width = int(lengths.max()) // 64 + 1
        pool = rows * width + 3
        stride = width + 3
        perm = torch.randperm(pool, generator=gen)[:rows * width].to(torch.int32).view(rows, width)
        page_table = torch.full((rows, stride), -1, dtype=torch.int32)
        page_table[:, :width] = perm
        page_table = page_table.cuda()
        lengths = lengths.cuda()
        tables = page_table[:, :width]
    cache = _random_cache(pool, gen)
    out = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(q8, weights, cache, page_table, lengths, out, scratch,
                   scalars=(rows, width, stride))
    torch.cuda.synchronize()
    keys_all = cache[:, :8192].view(pool, 64, 128).view(torch.float8_e4m3fn).float()
    scales_all = cache[:, 8192:].contiguous().view(torch.float32).view(pool, 64)
    recalls, exact_all = [], True
    for i in range(rows):
        n = int(lengths[i])
        logical = torch.arange(n, device="cuda")
        phys = tables[i, logical // 64].long() * 64 + logical % 64
        keys = keys_all.view(-1, 128)[phys]
        scales = scales_all.view(-1)[phys]
        s = _index_scores(q8[i:i + 1], weights[i:i + 1], keys, scales)[0]
        got = out[i]
        valid = got[got >= 0]
        if n <= 2048:
            exact_all &= bool(torch.equal(torch.sort(valid.long()).values, torch.sort(phys).values))
            exact_all &= int((got < 0).sum()) == 2048 - n
            continue
        top = torch.topk(s, 2048).indices
        expected = set(phys[top].tolist())
        recalls.append(len(expected & set(valid.tolist())) / 2048)
        assert valid.numel() == 2048
    recall = sum(recalls) / len(recalls) if recalls else 1.0
    print(f"glm_index_topk {mode} m{capacity} rows={rows}: route {program.geometry['route']} "
          f"recall {recall:.6f} over {len(recalls)} long rows; short rows exact {exact_all}")
    assert exact_all and recall >= 0.999
