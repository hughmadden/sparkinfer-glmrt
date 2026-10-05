"""MLA token-row outputs against the unchanged full-head consumer and its UV scratch."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_output_shard_aot import _guarded, _guard


@pytest.fixture(scope="module")
def case():
    require_b12x()
    from b12x.integration.cuteafd import GLM53_FLASH, glmf
    torch.manual_seed(8731)
    snapshot = os.environ.get("GLMF_MLA_NATIVE_SNAPSHOT")
    if snapshot:
        from safetensors import safe_open
        path = Path(snapshot)
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.language_model.layers.3.self_attn."
        def read(name):
            with safe_open(str(path / index[prefix + name]), framework="pt", device="cpu") as f:
                return f.get_tensor(prefix + name).to("cuda")
        w = read("o_proj.weight")
        scales = read("o_proj.weight_scale_inv").float()
        kv_b = read("kv_b_proj.weight")
        assert w.dtype == torch.float8_e4m3fn and scales.shape == (32, 128)
        assert kv_b.dtype == torch.bfloat16 and kv_b.shape == (32768, 512)
        uv = kv_b.view(64, 512, 512)[:, 256:].contiguous()
        print("MLA oracle weights: native layer3 block FP8 and BF16 W_UV")
    else:
        w = (torch.randn(4096, 16384, device="cuda") * 16).to(torch.float8_e4m3fn)
        scales = torch.rand(32, 128, device="cuda") * .001 + .0005
        uv = (torch.randn(64, 256, 512, device="cuda") * .01).bfloat16()
        print("MLA oracle weights: synthetic block FP8")
    broad = torch.exp2(torch.randint(-20, 21, scales.shape, device="cuda").float())
    broad *= .75 + torch.rand_like(broad) * .5
    half = replace(GLM53_FLASH, heads=32)
    values, projections, oracles = {}, {}, {}
    for mode in ["decode", "prefill"]:
        capacity = 64 if mode == "decode" else 4096
        values[mode] = glmf.compile_glmf_mla_values_aot(half, max_rows=capacity)
        projections[mode] = glmf.compile_glmf_mla_output_rows_aot(half, max_rows=capacity, fp8_only=mode)
        oracles[mode] = glmf.compile_glmf_o_aot(GLM53_FLASH, max_rows=capacity, fp8_only=mode)
    join = glmf.compile_glmf_join_mla_heads_aot(half)
    return w, {"native": scales, "broad": broad}, uv, values, projections, oracles, join


@pytest.mark.parametrize("max_rows", [64, 4096])
def test_mla_values_exported_header_matches_abi(tmp_path, max_rows):
    require_b12x()
    from b12x.integration.cuteafd import GLM53_FLASH, exportable_compilation, glmf, validate_exported_header
    with exportable_compilation():
        program = glmf.compile_glmf_mla_values_aot(replace(GLM53_FLASH, heads=32), max_rows=max_rows)
    stem = f"glmf2_mla_values_m{max_rows}"
    program.export_to_c(str(tmp_path), stem, "cuteafd_" + stem)
    checked = validate_exported_header(program, tmp_path / (stem + ".h"), "cuteafd_" + stem)
    assert [operand.name for operand in program.operands] == ["a", "w", "out"]
    assert checked["argument_count"] == 5


CASES = [("decode", n, 16) for n in [1, 22, 63, 64]] + [
    ("prefill", n, a8) for n in [1, 22, 63, 64, 512, 513, 4096] for a8 in [0, 1]]


@pytest.mark.parametrize("mode,total,fp8", CASES)
@pytest.mark.parametrize("recipe", ["native", "broad"])
def test_owned_rows_and_uv_match_original_full_consumer_with_changed_input_graph(case, mode, total, fp8, recipe):
    w, scale_recipes, uv, values, projections, oracles, join = case
    scale = scale_recipes[recipe]
    value_program, projection, oracle = values[mode], projections[mode], oracles[mode]
    attn = torch.randn(total, 64, 512, device="cuda", dtype=torch.bfloat16) * .1
    halves = [attn[:, :32].contiguous(), attn[:, 32:].contiguous()]
    value_out = [_guarded(total, 8192) for _ in range(2)]
    expected, expected_guard = _guarded(total, 4096)
    oracle_bytes = oracle.scratch_bytes(total)["scratch"]
    oracle_storage = torch.full((oracle_bytes + 2048,), 0xA5, device="cuda", dtype=torch.uint8)
    oracle_scratch = oracle_storage[1024:-1024]

    def reference():
        oracle.launch(attn, uv, w, scale, expected, oracle_scratch, scalars=(total, fp8))

    def expand_values():
        for rank in [0, 1]:
            value_program.launch(halves[rank], uv[rank * 32:(rank + 1) * 32],
                value_out[rank][0], scalars=(total,))

    reference()
    expand_values()
    torch.cuda.synchronize()
    expected_values = oracle_scratch[:total * 16384 * 2].view(torch.bfloat16).reshape(total, 16384)
    assert torch.equal(torch.cat([a[0] for a in value_out], dim=1), expected_values)
    assert torch.isfinite(expected_values).all() and expected_values.abs().max() > 0

    split = (total + 1) // 2
    for first, owned in [(0, split), (split, total - split)]:
        full, full_guard = _guarded(max(1, owned), 16384)
        out, out_guard = _guarded(max(1, owned), 4096)
        scratch_bytes = projection.scratch_bytes(total)["scratch"]
        assert scratch_bytes == (0 if mode == "decode" else
            ((split * 16384 + 1023) // 1024) * 1024 + ((split * 512 + 1023) // 1024) * 1024)
        storage = torch.full((scratch_bytes + 2048,), 0xA5, device="cuda", dtype=torch.uint8)
        scratch = storage[1024:-1024] if scratch_bytes else storage[1024:1280]

        def run():
            expand_values()
            if owned:
                join.launch(value_out[0][0][first:first + owned], value_out[1][0][first:first + owned],
                    full, scalars=(owned,))
            projection.launch(full, w, scale, out, scratch, scalars=(owned, total, fp8))

        run()
        torch.cuda.synchronize()
        if owned:
            assert torch.equal(out, expected[first:first + owned])
            assert torch.isfinite(out).all() and out.abs().max() > 0
        else:
            assert (out_guard == 0xA5).all() and (storage == 0xA5).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        attn.mul_(1.125)
        halves[0].copy_(attn[:, :32])
        halves[1].copy_(attn[:, 32:])
        reference()
        graph.replay()
        torch.cuda.synchronize()
        if owned:
            assert torch.equal(out, expected[first:first + owned])
        else:
            assert (out_guard == 0xA5).all() and (storage == 0xA5).all()
        expected_values = oracle_scratch[:total * 16384 * 2].view(torch.bfloat16).reshape(total, 16384)
        assert torch.equal(torch.cat([a[0] for a in value_out], dim=1), expected_values)
        for guard in [full_guard, out_guard, storage, expected_guard, oracle_storage, *[a[1] for a in value_out]]:
            _guard(guard)


@pytest.mark.parametrize("mode,total,fp8", [("decode", 1, 16), ("decode", 64, 16),
    ("prefill", 1, 0), ("prefill", 513, 1), ("prefill", 4096, 1)])
def test_zero_owned_output_preserves_storage_and_graph(case, mode, total, fp8):
    w, scales, _, _, programs, _, _ = case
    p = programs[mode]
    x = torch.empty(1, 16384, device="cuda", dtype=torch.bfloat16)
    out, guard = _guarded(1, 4096)
    storage = torch.full((p.scratch_bytes(total)["scratch"] + 2048,), 0xA5, device="cuda", dtype=torch.uint8)
    scratch = storage[1024:-1024] if storage.numel() > 2048 else storage[1024:1280]
    p.launch(x, w, scales["native"], out, scratch, scalars=(0, total, fp8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(x, w, scales["native"], out, scratch, scalars=(0, total, fp8))
    graph.replay()
    torch.cuda.synchronize()
    assert (guard == 0xA5).all() and (storage == 0xA5).all()
