"""CPU-only compile contracts; numerical GPU qualification is engine-owned."""
import inspect

import pytest
import torch

from b12x.integration.cuteafd import kimi_residual as module


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(module, "compile_program", lambda launch, **kw: (launch, kw))


@pytest.mark.parametrize("candidates", range(1, 10))
@pytest.mark.parametrize("prefix_mode", ["keep", "add", "replace"])
def test_residual_abi(capture, candidates, prefix_mode):
    kernel, info = module.compile_kimi_attn_res_rmsnorm_aot(
        candidates=candidates, prefix_mode=prefix_mode, has_mix=candidates != 1)
    assert [(x.name, x.dtype, x.role) for x in info["operands"]] == [
        ("arena", torch.bfloat16, "inout"), ("delta", torch.bfloat16, "in"),
        ("score_weight", torch.float32, "in"), ("norm_weight", torch.bfloat16, "in"),
        ("out", torch.bfloat16, "out")]
    assert [(x.name, x.ctype) for x in info["scalars"]] == [
        ("rows", "int32"), ("arena_rows", "int32"), ("eps", "float32")]
    assert info["operands"][0].shape == "[9,arena_rows,7168]"
    assert info["key"] == (7168, candidates, prefix_mode, False, candidates != 1)
    assert "scratch" not in info
    assert kernel.elements * kernel.threads == 7168


@pytest.mark.parametrize("candidates", range(1, 9))
def test_append_has_a_free_block_slot(capture, candidates):
    kernel, _ = module.compile_kimi_attn_res_rmsnorm_aot(
        candidates=candidates, append=True, has_mix=candidates != 1)
    assert kernel.append


@pytest.mark.parametrize("kwargs", [dict(hidden=6144), dict(candidates=0),
    dict(candidates=10), dict(candidates=9, append=True), dict(prefix_mode="reset"),
    dict(candidates=2, has_mix=False)])
def test_bad_geometry_fails_before_compilation(capture, kwargs):
    with pytest.raises(ValueError):
        module.compile_kimi_attn_res_rmsnorm_aot(**kwargs)


@pytest.mark.parametrize("intermediate", [3072, 6144, 16896, 33792])
def test_situ_abi(capture, intermediate):
    _, info = module.compile_kimi_situ_aot(intermediate)
    assert [(x.name, x.shape, x.role) for x in info["operands"]] == [
        ("gate_up", f"[rows,{2 * intermediate}]", "in"),
        ("out", f"[rows,{intermediate}]", "out")]
    assert [(x.name, x.ctype) for x in info["scalars"]] == [("rows", "int32")]
    assert info["key"] == (intermediate,)


def test_no_live_count_compile_keys(capture):
    for compile_fn in (module.compile_kimi_attn_res_rmsnorm_aot, module.compile_kimi_situ_aot):
        params = inspect.signature(compile_fn).parameters
        assert "rows" not in params and "arena_rows" not in params
    with pytest.raises(ValueError):
        module.compile_kimi_situ_aot(4096)
