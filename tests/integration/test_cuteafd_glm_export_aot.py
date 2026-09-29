"""Every cuteafd GLM 5.x program exports with the documented ABI, and the
routed-expert input quantizer writes K32 wire rows at the GLM width."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import check_export


def _programs():
    from b12x.integration.cuteafd import GLM53 as g
    from b12x.integration.cuteafd import glm_attention as attn
    from b12x.integration.cuteafd import glm_ffn as ffn
    from b12x.integration.cuteafd import glm_indexer as idx
    from b12x.integration.cuteafd import glm_sparse_mla as mla

    return {
        "glm_norm": lambda: ffn.compile_glm_norm_aot(g),
        "glm_router_scores": lambda: ffn.compile_glm_router_scores_aot(g),
        "glm_expert_input_quant": lambda: ffn.compile_glm_expert_input_quant_aot(g),
        "glm_producer_m64": lambda: attn.compile_glm_producer_aot(g, max_rows=64),
        "glm_index_producer_m64": lambda: attn.compile_glm_index_producer_aot(g, max_rows=64),
        "glm_o_m64": lambda: attn.compile_glm_o_aot(g, max_rows=64),
        "glm_ffn_i12288_m64": lambda: ffn.compile_glm_ffn_aot(g, inter=12288, max_rows=64),
        "glm_index_topk_decode_m64": lambda: idx.compile_glm_index_topk_aot(
            g, max_rows=64, max_pages=2048, mode="decode"),
        "glm_index_topk_prefill_m4096": lambda: idx.compile_glm_index_topk_aot(
            g, max_rows=4096, max_pages=2048, mode="prefill"),
        "glm_sparse_mla_decode_m64": lambda: mla.compile_glm_sparse_mla_aot(g, route="decode", max_rows=64),
        "glm_sparse_mla_prefill_m4096": lambda: mla.compile_glm_sparse_mla_aot(g, route="prefill", max_rows=4096),
    }


@pytest.mark.parametrize("stem", list(_programs()))
def test_glm_program_exports(stem, tmp_path):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation

    with exportable_compilation():
        program = _programs()[stem]()
    checked = check_export(program, tmp_path, stem)
    assert checked["argument_count"] == len(program.operands) + len(program.scalars) + 1


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_glm_expert_input_quant_wire_rows(rows):
    device = require_b12x()
    from b12x.integration.cuteafd import GLM53, glm_ffn
    from b12x.integration.cuteafd.dsv4_ffn import expert_input_quant_grid

    program = glm_ffn.compile_glm_expert_input_quant_aot(GLM53)
    h = GLM53.hidden
    stride = h + h // 32
    gen = torch.Generator(device="cpu").manual_seed(rows)
    x = torch.randn((rows, h), generator=gen).bfloat16().to(device)
    wire = torch.full((rows, stride), 255, dtype=torch.uint8, device=device)
    unused = torch.zeros(16, dtype=torch.uint8, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    program.launch(x, wire, wire[:, h:], unused, scalars=(rows, expert_input_quant_grid(h, rows, sms)))
    torch.cuda.synchronize()
    values = wire[:, :h].contiguous().view(torch.float8_e4m3fn).float().view(rows, h // 32, 32)
    exponents = wire[:, h:].contiguous().to(torch.int32) - 127
    decoded = (values * torch.exp2(exponents.float())[..., None]).view(rows, h)
    amax = x.float().view(rows, h // 32, 32).abs().amax(-1).clamp_min(1e-4)
    expected_exp = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32)
    assert torch.equal(exponents, expected_exp)
    rel = float((decoded - x.float()).norm() / x.float().norm())
    assert rel < 0.05, rel
