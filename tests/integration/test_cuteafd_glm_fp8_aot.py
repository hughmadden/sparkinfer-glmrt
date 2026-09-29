"""cuteafd GLM decode programs with FP8 weight operands (E4M3 + FP32 128x128
block scales) against the BF16-weight programs they replace.

Real GLM 5.3 weights and golden activations; the m64 FP8 programs run the
FP8 tensor-core GEMV up to 16 rows and the BF16 GEMM (their BF16 operand)
above, so 1..16 rows compare FP8 against BF16 math and 17+ rows must match
the BF16 programs bitwise.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._glm import cos_sin, cosine, fp8_rows, golden_rows, rms_norm, tensor

ROWS = [1, 6, 16, 17, 64]
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


def _scratch(program, rows):
    return torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


def _check(name, rows, fp8_out, bf16_out):
    c = cosine(fp8_out, bf16_out)
    same = torch.equal(fp8_out, bf16_out)
    print(f"{name} rows={rows}: cosine {c:.8f}{' (bitwise equal)' if same else ''}")
    assert c >= 0.99999
    if rows > 16:
        assert same, "above 16 rows the FP8 program runs the BF16 GEMM"


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


@pytest.mark.parametrize("rows", ROWS)
def test_glm_producer_fp8(rows):
    a = "model.layers.0.self_attn."
    w_qkv_a = torch.cat([tensor(a + "q_a_proj.weight"), tensor(a + "kv_a_proj_with_mqa.weight")]).contiguous()
    qkv_fp8, qkv_scale = fp8_rows([a + "q_a_proj.weight", a + "kv_a_proj_with_mqa.weight"])
    w_q_b = tensor(a + "q_b_proj.weight")
    q_b_fp8, q_b_scale = fp8_rows([a + "q_b_proj.weight"])
    kv_b = tensor(a + "kv_b_proj.weight").view(64, 448, 512)
    w_uk = kv_b[:, :192, :].transpose(1, 2).contiguous()
    norms = [tensor(a + "q_a_layernorm.weight"), tensor(a + "kv_a_layernorm.weight")]
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") + 100
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    outs = []
    for fp8 in (False, True):
        program = P("producer", max_rows=64, fp8=fp8)
        cache = torch.zeros((2, 64 * 656), dtype=torch.uint8, device="cuda")
        query = torch.empty((rows, 64, 576), dtype=torch.bfloat16, device="cuda")
        q_resid = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
        weights = ([w_qkv_a, qkv_fp8, qkv_scale, *norms, w_q_b, q_b_fp8, q_b_scale] if fp8
                   else [w_qkv_a, *norms, w_q_b])
        program.launch(x, positions, slots, cos_sin(256), *weights, w_uk, cache, query, q_resid,
                       _scratch(program, rows), scalars=(rows,))
        outs.append((query, q_resid, cache))
    torch.cuda.synchronize()
    _check("glm_producer query", rows, outs[1][0], outs[0][0])
    _check("glm_producer q_resid", rows, outs[1][1], outs[0][1])


@pytest.mark.parametrize("rows", ROWS)
def test_glm_index_producer_fp8(rows):
    ix = "model.layers.0.self_attn.indexer."
    w_iq = tensor(ix + "wq_b.weight")
    iq_fp8, iq_scale = fp8_rows([ix + "wq_b.weight"])
    w_ik = torch.cat([tensor(ix + "wk.weight"), tensor(ix + "weights_proj.weight").bfloat16()]).contiguous()
    k_norm = [tensor(ix + "k_norm.weight"), tensor(ix + "k_norm.bias")]
    x = rms_norm(golden_rows(0, rows), tensor("model.layers.0.input_layernorm.weight"))
    q_resid = rms_norm(golden_rows(1, rows)[:, :2048], tensor("model.layers.0.self_attn.q_a_layernorm.weight"))
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    outs = []
    for fp8 in (False, True):
        program = P("index_producer", max_rows=64, fp8=fp8)
        cache = torch.zeros((2, 8448), dtype=torch.uint8, device="cuda")
        q8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
        hw = torch.empty((rows, 32), dtype=torch.float32, device="cuda")
        weights = [w_iq, iq_fp8, iq_scale] if fp8 else [w_iq]
        program.launch(x, q_resid.contiguous(), positions, positions, cos_sin(256), *weights, w_ik, *k_norm,
                       cache, q8, hw, _scratch(program, rows), scalars=(rows,))
        outs.append(q8.float() * hw[..., None])
    torch.cuda.synchronize()
    _check("glm_index_producer q*w", rows, outs[1], outs[0])


@pytest.mark.parametrize("rows", ROWS)
def test_glm_o_fp8(rows):
    a = "model.layers.0.self_attn."
    kv_b = tensor(a + "kv_b_proj.weight").view(64, 448, 512)
    w_uv = kv_b[:, 192:, :].contiguous()
    w_o = tensor(a + "o_proj.weight")
    o_fp8, o_scale = fp8_rows([a + "o_proj.weight"])
    gen = torch.Generator(device="cpu").manual_seed(rows)
    attn = (torch.randn((rows, 64, 512), generator=gen) * 0.3).bfloat16().cuda()
    outs = []
    for fp8 in (False, True):
        program = P("o", max_rows=64, fp8=fp8)
        out = torch.empty((rows, 6144), dtype=torch.bfloat16, device="cuda")
        weights = [w_o, o_fp8, o_scale] if fp8 else [w_o]
        program.launch(attn, w_uv, *weights, out, _scratch(program, rows), scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    _check("glm_o", rows, outs[1], outs[0])


@pytest.mark.parametrize("layer,inter", [(0, 12288), (3, 2048)])
@pytest.mark.parametrize("rows", ROWS)
def test_glm_ffn_fp8(layer, inter, rows):
    m = f"model.layers.{layer}.mlp." + ("" if inter == 12288 else "shared_experts.")
    w_gu = torch.cat([tensor(m + "gate_proj.weight"), tensor(m + "up_proj.weight")]).contiguous()
    gu_fp8, gu_scale = fp8_rows([m + "gate_proj.weight", m + "up_proj.weight"])
    w_down = tensor(m + "down_proj.weight")
    down_fp8, down_scale = fp8_rows([m + "down_proj.weight"])
    x = rms_norm(golden_rows(layer, rows), tensor(f"model.layers.{layer}.post_attention_layernorm.weight"))
    outs = []
    for fp8 in (False, True):
        program = P("ffn", inter=inter, max_rows=64, fp8=fp8)
        out = torch.empty_like(x)
        weights = [w_gu, gu_fp8, gu_scale, w_down, down_fp8, down_scale] if fp8 else [w_gu, w_down]
        program.launch(x, *weights, out, _scratch(program, rows), scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    _check(f"glm_ffn_i{inter}", rows, outs[1], outs[0])


@pytest.mark.parametrize("n,k", [(2624, 6144), (6144, 16384)])
@pytest.mark.parametrize("rows", [1, 64, 300])
def test_tma_fp8_gemm_matches_bf16_gemm(n, k, rows):
    """The FP8-tile TMA GEMM equals the BF16 TMA GEMM over bf16(w * s) bitwise."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import TmaFp8Gemm

    gen = torch.Generator(device="cuda").manual_seed(n + rows)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    s = torch.rand(((n + 127) // 128, k // 128), generator=gen, device="cuda") * 0.01 + 1e-3
    wd = (w8.float() * s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)).bfloat16()
    x = torch.randn((rows, k), generator=gen, device="cuda").bfloat16()
    outs = []
    for fp8 in (True, False):
        ops = [Operand("x", torch.bfloat16, "x"), Operand("w", torch.float8_e4m3fn if fp8 else torch.bfloat16, "w")]
        ops += [Operand("s", torch.float32, "s", align=4)] if fp8 else []
        ops.append(Operand("o", torch.bfloat16, "o", "out"))
        launch = TmaFp8Gemm(n, k) if fp8 else TmaBf16Projection(n, k)
        program = compile_program(launch, name=f"t{int(fp8)}_{n}_{k}", operands=ops, scalars=(Scalar("rows"),),
                                  key=(n, k, fp8))
        out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
        program.launch(x, *([w8, s] if fp8 else [wd]), out, scalars=(rows,))
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
