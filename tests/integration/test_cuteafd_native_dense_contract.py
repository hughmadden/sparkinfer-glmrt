"""CPU-only native dense ABI contracts, no CUDA context required."""
import pytest
import torch

from b12x.integration.cuteafd import native_dense, trellis_unpack


@pytest.mark.parametrize("side", [1, 2])
def test_unpack_abi(monkeypatch, side):
    monkeypatch.setattr(trellis_unpack, "compile_program", lambda launch, **kw: kw)
    info = trellis_unpack.compile_trellis_unpack_e4m3_aot(
        input_features=2048, output_features=6144, sharded_side=side)
    assert [(p.name, p.dtype, p.shape) for p in info["operands"]] == [
        ("payload", torch.uint8, "[6291456]"), ("output", torch.uint8, "[6144,2048]")]
    assert info["scalars"] == ()
    assert info["geometry"]["bits"] == 4
    assert info["geometry"]["codebook"] == "e4m3"


def test_prefill_and_glue_abi(monkeypatch):
    monkeypatch.setattr(native_dense, "compile_program", lambda launch, **kw: (launch, kw))
    launch, info = native_dense.compile_trellis_fp8_prefill_aot(
        input_features=2048, output_features=8192)
    assert (launch.num_compute_warps, launch.num_stages, launch.scales) == (8, 3, "block")
    assert [(p.name, p.dtype, p.shape) for p in info["operands"]] == [
        ("x", torch.bfloat16, "[rows,2048]"), ("weight", torch.uint8, "[8192,2048]"),
        ("scales", torch.float32, "[64,16]"), ("output", torch.float32, "[rows,8192]")]
    assert [(s.name, s.ctype) for s in info["scalars"]] == [("rows", "int32")]
    _, info = native_dense.compile_swiglu_split_aot(inter=6144)
    assert [p.name for p in info["operands"]] == ["gate", "up", "hidden"]
    assert [(s.name, s.ctype) for s in info["scalars"]] == [("rows", "int32")]
    _, info = native_dense.compile_f32_narrow_pack_aot(width=576, pitch=2624)
    assert [(p.dtype, p.shape) for p in info["operands"]] == [
        (torch.float32, "[rows,576]"), (torch.bfloat16, "[rows,2624]")]
    assert [(s.name, s.ctype) for s in info["scalars"]] == [("rows", "int32"), ("c0", "int32")]


@pytest.mark.parametrize("kwargs", [dict(input_features=127, output_features=128),
    dict(input_features=128, output_features=0), dict(input_features=128, output_features=128, sharded_side=0)])
def test_unpack_bad_geometry(kwargs):
    with pytest.raises(ValueError):
        trellis_unpack.compile_trellis_unpack_e4m3_aot(**kwargs)


@pytest.mark.parametrize("compiler,kwargs", [
    (native_dense.compile_trellis_fp8_prefill_aot, dict(input_features=129, output_features=128)),
    (native_dense.compile_trellis_fp8_prefill_aot, dict(input_features=128, output_features=128, max_rows=0)),
    (native_dense.compile_swiglu_split_aot, dict(inter=0)),
    (native_dense.compile_f32_narrow_pack_aot, dict(width=577, pitch=576)),
])
def test_dense_bad_geometry(compiler, kwargs):
    with pytest.raises(ValueError):
        compiler(**kwargs)
