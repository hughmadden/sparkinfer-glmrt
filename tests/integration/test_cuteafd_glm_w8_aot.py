"""cuteafd GLM 5.3 programs over FP8-only weights (``fp8_only``: no BF16 copies).

Decode programs (``fp8_only="decode"``) must equal the BF16+FP8 decode
programs bitwise at every row count (GEMV up to 16 rows, the W8A16 TMA GEMMs
and batched kv_b GEMMs above equal the BF16 ones over dequantized weights).
Prefill programs (``fp8_only="prefill"``) with ``fp8_rows`` 0 (W8A16) equal
the BF16 prefill programs bitwise above the BF16 skinny GEMV's 8 rows; with
``fp8_rows`` 1 (W8A8) they stay close to them.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import cos_sin, cosine, fp8_rows, golden_rows, raw_tensor, rms_norm, tensor

_PROGRAMS: dict = {}


def P(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53, glm_attention, glm_ffn

        fn = {"producer": glm_attention.compile_glm_producer_aot,
              "index_producer": glm_attention.compile_glm_index_producer_aot,
              "o": glm_attention.compile_glm_o_aot, "ffn": glm_ffn.compile_glm_ffn_aot}[kind]
        _PROGRAMS[key] = fn(GLM53, **kw)
    return _PROGRAMS[key]


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


def _kv_b_fp8(prefix):
    """w_uk / w_uv E4M3 slices of kv_b and their per-row x 64-K scales (the loader's layout)."""
    w8 = raw_tensor(prefix + "kv_b_proj.weight").cuda().view(64, 448, 512)
    grid = raw_tensor(prefix + "kv_b_proj.weight_scale_inv").cuda().float()
    rows = grid.repeat_interleave(128, 0)[: 64 * 448].view(64, 448, 4)
    uk = w8[:, :192, :].transpose(1, 2).contiguous()
    uk_s = rows[:, :192, :].repeat_interleave(128, 2).transpose(1, 2)[:, :, ::64].contiguous()
    uv = w8[:, 192:, :].contiguous()
    uv_s = rows[:, 192:, :].repeat_interleave(2, 2).contiguous()
    return uk, uk_s, uv, uv_s


def _kscale(names):
    """Per-row x 128-K scales, K-block major, of the row-concatenated 128x128 grids."""
    _, grid = fp8_rows(names)
    n = sum(raw_tensor(nm).shape[0] for nm in names)
    return grid.repeat_interleave(128, 0)[:n].t().contiguous()


def _compare(name, rows, new, old, bitwise):
    c = cosine(new, old)
    same = torch.equal(new, old)
    print(f"{name} rows={rows}: cosine {c:.7f}{' (bitwise equal)' if same else ''}")
    if bitwise:
        assert same
    else:
        assert c >= 0.999


DECODE_ROWS = [1, 16, 17, 64]
PREFILL = [(64, 0), (300, 0), (5, 1), (300, 1)]


def _producer_inputs(rows):
    a = "model.layers.0.self_attn."
    names = [a + "q_a_proj.weight", a + "kv_a_proj_with_mqa.weight"]
    qkv_fp8, qkv_scale = fp8_rows(names)
    q_b_fp8, q_b_scale = fp8_rows([a + "q_b_proj.weight"])
    norms = [tensor(a + "q_a_layernorm.weight"), tensor(a + "kv_a_layernorm.weight")]
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    return a, names, qkv_fp8, qkv_scale, q_b_fp8, q_b_scale, norms, x


def _run_producer(program, rows, x, weights):
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") + 100
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    pages = (rows + 63) // 64 + 1
    cache = torch.zeros((pages, 64 * 656), dtype=torch.uint8, device="cuda")
    query = torch.empty((rows, 64, 576), dtype=torch.bfloat16, device="cuda")
    q_resid = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    scalars = weights.pop()
    program.launch(x, positions, slots, cos_sin(4096), *weights, cache, query, q_resid, _scratch(program, rows),
                   scalars=scalars)
    return query, q_resid, cache


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_glm_producer_w8_decode(rows):
    a, names, qkv_fp8, qkv_scale, q_b_fp8, q_b_scale, norms, x = _producer_inputs(rows)
    w_qkv_a = torch.cat([tensor(n) for n in names]).contiguous()
    w_q_b = tensor(a + "q_b_proj.weight")
    kv_b = tensor(a + "kv_b_proj.weight").view(64, 448, 512)
    w_uk = kv_b[:, :192, :].transpose(1, 2).contiguous()
    uk, uk_s, _, _ = _kv_b_fp8(a)
    old = _run_producer(P("producer", max_rows=64, fp8=True), rows, x,
                        [w_qkv_a, qkv_fp8, qkv_scale, *norms, w_q_b, q_b_fp8, q_b_scale, w_uk, (rows,)])
    new = _run_producer(P("producer", max_rows=64, fp8_only="decode"), rows, x,
                        [qkv_fp8, qkv_scale, *norms, q_b_fp8, q_b_scale, uk, uk_s, (rows,)])
    torch.cuda.synchronize()
    for i, label in enumerate(("query", "q_resid", "cache")):
        _compare(f"glm_producer {label}", rows, new[i].view(torch.uint8) if i == 2 else new[i],
                 old[i].view(torch.uint8) if i == 2 else old[i], True)


@pytest.mark.parametrize("rows,fp8", PREFILL)
def test_glm_producer_w8_prefill(rows, fp8):
    a, names, qkv_fp8, _, q_b_fp8, q_b_scale, norms, x = _producer_inputs(rows)
    w_qkv_a = torch.cat([tensor(n) for n in names]).contiguous()
    w_q_b = tensor(a + "q_b_proj.weight")
    kv_b = tensor(a + "kv_b_proj.weight").view(64, 448, 512)
    w_uk = kv_b[:, :192, :].transpose(1, 2).contiguous()
    uk, uk_s, _, _ = _kv_b_fp8(a)
    old = _run_producer(P("producer", max_rows=4096), rows, x, [w_qkv_a, *norms, w_q_b, w_uk, (rows,)])
    new = _run_producer(P("producer", max_rows=4096, fp8_only="prefill"), rows, x,
                        [qkv_fp8, _kscale(names), *norms, q_b_fp8, q_b_scale, uk, uk_s, (rows, fp8)])
    torch.cuda.synchronize()
    _compare(f"glm_producer prefill fp8={fp8} query", rows, new[0], old[0], fp8 == 0)
    _compare(f"glm_producer prefill fp8={fp8} q_resid", rows, new[1], old[1], fp8 == 0)


def _index_inputs(rows):
    ix = "model.layers.0.self_attn.indexer."
    w_ik = torch.cat([tensor(ix + "wk.weight"), tensor(ix + "weights_proj.weight").bfloat16()]).contiguous()
    k_norm = [tensor(ix + "k_norm.weight"), tensor(ix + "k_norm.bias")]
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    q_resid = rms_norm(golden_rows(1, rows)[:, :2048], tensor("model.layers.0.self_attn.q_a_layernorm.weight"))
    return ix, w_ik, k_norm, x, q_resid.contiguous()


def _run_index(program, rows, x, q_resid, weights, scalars):
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    cache = torch.zeros(((rows + 63) // 64 + 1, 8448), dtype=torch.uint8, device="cuda")
    q8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
    hw = torch.empty((rows, 32), dtype=torch.float32, device="cuda")
    program.launch(x, q_resid, positions, positions, cos_sin(4096), *weights, cache, q8, hw,
                   _scratch(program, rows), scalars=scalars)
    return q8.float() * hw[..., None]


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_glm_index_producer_w8_decode(rows):
    ix, w_ik, k_norm, x, q_resid = _index_inputs(rows)
    iq_fp8, iq_scale = fp8_rows([ix + "wq_b.weight"])
    old = _run_index(P("index_producer", max_rows=64, fp8=True), rows, x, q_resid,
                     [tensor(ix + "wq_b.weight"), iq_fp8, iq_scale, w_ik, *k_norm], (rows,))
    new = _run_index(P("index_producer", max_rows=64, fp8_only="decode"), rows, x, q_resid,
                     [iq_fp8, iq_scale, w_ik, *k_norm], (rows,))
    torch.cuda.synchronize()
    _compare("glm_index_producer q*w", rows, new, old, True)


@pytest.mark.parametrize("rows,fp8", PREFILL)
def test_glm_index_producer_w8_prefill(rows, fp8):
    ix, w_ik, k_norm, x, q_resid = _index_inputs(rows)
    iq_fp8, iq_scale = fp8_rows([ix + "wq_b.weight"])
    old = _run_index(P("index_producer", max_rows=4096), rows, x, q_resid,
                     [tensor(ix + "wq_b.weight"), w_ik, *k_norm], (rows,))
    new = _run_index(P("index_producer", max_rows=4096, fp8_only="prefill"), rows, x, q_resid,
                     [iq_fp8, iq_scale, w_ik, *k_norm], (rows, fp8))
    torch.cuda.synchronize()
    _compare(f"glm_index_producer prefill fp8={fp8}", rows, new, old, fp8 == 0)


def _run_o(program, rows, attn, weights, scalars):
    out = torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda")
    program.launch(attn, *weights, out, _scratch(program, rows), scalars=scalars)
    return out


def _o_inputs(rows):
    a = "model.layers.0.self_attn."
    gen = torch.Generator(device="cpu").manual_seed(rows)
    attn = (torch.randn((rows, 64, 512), generator=gen) * 0.3).bfloat16().cuda()
    kv_b = tensor(a + "kv_b_proj.weight").view(64, 448, 512)
    return a, attn, kv_b[:, 192:, :].contiguous()


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_glm_o_w8_decode(rows):
    a, attn, w_uv = _o_inputs(rows)
    o_fp8, o_scale = fp8_rows([a + "o_proj.weight"])
    _, _, uv, uv_s = _kv_b_fp8(a)
    old = _run_o(P("o", max_rows=64, fp8=True), rows, attn, [w_uv, tensor(a + "o_proj.weight"), o_fp8, o_scale],
                 (rows,))
    new = _run_o(P("o", max_rows=64, fp8_only="decode"), rows, attn, [uv, uv_s, o_fp8, o_scale], (rows,))
    torch.cuda.synchronize()
    _compare("glm_o", rows, new, old, True)


@pytest.mark.parametrize("rows,fp8", PREFILL)
def test_glm_o_w8_prefill(rows, fp8):
    a, attn, w_uv = _o_inputs(rows)
    o_fp8, o_scale = fp8_rows([a + "o_proj.weight"])
    _, _, uv, uv_s = _kv_b_fp8(a)
    old = _run_o(P("o", max_rows=4096), rows, attn, [w_uv, tensor(a + "o_proj.weight")], (rows,))
    new = _run_o(P("o", max_rows=4096, fp8_only="prefill"), rows, attn, [uv, uv_s, o_fp8, o_scale], (rows, fp8))
    torch.cuda.synchronize()
    _compare(f"glm_o prefill fp8={fp8}", rows, new, old, fp8 == 0)


def _ffn_inputs(layer, inter, rows):
    m = f"model.layers.{layer}.mlp." + ("" if inter == 12288 else "shared_experts.")
    gu = [m + "gate_proj.weight", m + "up_proj.weight"]
    x = rms_norm(golden_rows(layer, rows), tensor(f"model.layers.{layer}.post_attention_layernorm.weight"))
    return m, gu, x


@pytest.mark.parametrize("layer,inter", [(0, 12288), (3, 2048)])
@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_glm_ffn_w8_decode(layer, inter, rows):
    m, gu, x = _ffn_inputs(layer, inter, rows)
    gu_fp8, gu_scale = fp8_rows(gu)
    down_fp8, down_scale = fp8_rows([m + "down_proj.weight"])
    w_gu = torch.cat([tensor(n) for n in gu]).contiguous()
    outs = []
    for kw, weights in (({"fp8": True}, [w_gu, gu_fp8, gu_scale, tensor(m + "down_proj.weight"), down_fp8,
                                         down_scale]),
                        ({"fp8_only": "decode"}, [gu_fp8, gu_scale, down_fp8, down_scale])):
        program = P("ffn", inter=inter, max_rows=64, **kw)
        out = torch.empty_like(x)
        program.launch(x, *weights, out, _scratch(program, rows), scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    _compare(f"glm_ffn_i{inter}", rows, outs[1], outs[0], True)


@pytest.mark.parametrize("layer,inter", [(0, 12288), (3, 2048)])
@pytest.mark.parametrize("rows,fp8", PREFILL)
def test_glm_ffn_w8_prefill(layer, inter, rows, fp8):
    m, gu, x = _ffn_inputs(layer, inter, rows)
    gu_fp8, gu_scale = fp8_rows(gu)
    down_fp8, down_scale = fp8_rows([m + "down_proj.weight"])
    w_gu = torch.cat([tensor(n) for n in gu]).contiguous()
    old_p = P("ffn", inter=inter, max_rows=4096)
    old = torch.empty_like(x)
    old_p.launch(x, w_gu, tensor(m + "down_proj.weight"), old, _scratch(old_p, rows), scalars=(rows,))
    new_p = P("ffn", inter=inter, max_rows=4096, fp8_only="prefill")
    new = torch.empty_like(x)
    new_p.launch(x, gu_fp8, gu_scale, down_fp8, down_scale, new, _scratch(new_p, rows), scalars=(rows, fp8))
    torch.cuda.synchronize()
    _compare(f"glm_ffn_i{inter} prefill fp8={fp8}", rows, new, old, fp8 == 0)
