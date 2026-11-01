"""CPU-only alignment and storage contracts used by the real A8 down kernel."""
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import runpy

import pytest


PLAN = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                         "b12x/integration/cuteafd/_mxfp4_down_a8_plan.py"))
Mxfp4DownA8Plan = PLAN["Mxfp4DownA8Plan"]


@pytest.mark.parametrize("inter, stride", [(32, 48), (64, 80), (96, 112), (320, 336), (352, 368), (128, 144), (384, 400), (512, 528), (1024, 1056), (2048, 2112)])
def test_every_row_and_copy_stays_aligned_without_crossing_payload_or_scales(inter, stride):
    p = Mxfp4DownA8Plan(6144, inter, 384)
    assert p.row_bytes == stride
    for row in range(128):
        base = row * p.row_bytes
        for block in range((inter + p.k_block - 1) // p.k_block):
            for piece in range(p.k_block // 16):
                at = base + block * p.k_block + 16 * piece
                if block * p.k_block + 16 * piece < inter:
                    assert at % 16 == 0 and base <= at and at + 16 <= base + inter
            scale = base + inter + 4 * block
            assert scale % 4 == 0 and scale + 4 <= base + inter + PLAN["mxfp4_scale_row_bytes"](inter)
        assert base + inter + inter // 32 <= base + p.row_bytes
    assert p.row_bytes < 2 * inter


@pytest.mark.parametrize("inter", [32, 64, 96, 320, 352, 128, 384, 1024, 2048])
def test_shared_stages_do_not_overlap_and_copy_offsets_are_aligned(inter):
    p = Mxfp4DownA8Plan(6144, inter, 384)
    for stage in (0, 1):
        base = stage * p.stage_bytes
        for row in range(128):
            for piece in range(p.k_block // 16):
                at = base + row * p.a_stride + 16 * piece
                assert at % 16 == 0 and at + 16 <= base + p.a_bytes
            for piece in range(p.k_block // 32):
                at = base + p.a_bytes + row * p.w_stride + 16 * piece
                assert at % 16 == 0 and at + 16 <= base + p.a_bytes + p.w_bytes
            for plane in (0, 1):
                at = base + p.a_bytes + p.w_bytes + plane * p.s_bytes + row * p.s_stride
                assert at % 4 == 0 and at + 4 <= base + p.a_bytes + p.w_bytes + (plane + 1) * p.s_bytes
    assert 2 * p.stage_bytes == 61440
    assert p.out_bytes <= 2 * p.stage_bytes
    for row in range(128):
        for segment in range(p.tile_n * 2 // 16):
            at = row * p.out_stride + 16 * segment
            assert at % 16 == 0 and at + 16 <= p.out_bytes


def test_plan_contains_only_static_geometry_and_cannot_change_after_preparation():
    p = Mxfp4DownA8Plan(6144, 384, 384)
    assert [f.name for f in fields(p)] == ["hidden", "inter", "experts"]
    with pytest.raises(FrozenInstanceError):
        p.inter = 1024


@pytest.mark.parametrize("hidden, inter, experts", [(0, 384, 384), (6000, 384, 384), (6144, 0, 384),
                                                    (6144, 351, 384), (6144, 384, 0)])
def test_unrepresentable_geometry_fails_before_allocating(hidden, inter, experts):
    with pytest.raises(ValueError):
        Mxfp4DownA8Plan(hidden, inter, experts)


@pytest.mark.parametrize("k, stride", [(32, 4), (64, 4), (96, 4), (128, 4), (320, 12), (352, 12), (384, 12), (6144, 192)])
def test_weight_scale_rows_are_u32_aligned(k, stride):
    assert PLAN["mxfp4_scale_row_bytes"](k) == stride
    assert stride >= k // 32 and stride % 4 == 0


@pytest.mark.parametrize("rows", [1, 16, 80])
@pytest.mark.parametrize("spark", [False, True])
@pytest.mark.parametrize("hidden,width,matching", [(6144, 672, 6), (6144, 704, 6),
    (4096, 672, 6), (6144, 320, 3), (6144, 352, 3), (6144, 384, 3), (6144, 512, 4)])
def test_measured_decode_partition_is_fixed_across_capacities_and_hardware(rows, spark, hidden, width, matching):
    import ast
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd/fp8_moe.py"
    tree = ast.parse(source.read_text())
    route = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Route")
    init = next(node for node in route.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    primitive = lambda **kw: SimpleNamespace(**kw)
    env = dict(Fp8MoeGeometry=SimpleNamespace, DECODE_MAX_ROWS=80, MoePrep=primitive,
        GatherRows=primitive, GroupedMxfp4Gemv=primitive, GroupedFp8Gemv=primitive,
        MoeSwiGLU=primitive, MoeCombine=primitive, _gate_up_groups=lambda _: 4,
        _sm121=lambda: spark)
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(source), "exec"), env)
    geometry = SimpleNamespace(hidden=hidden, slice=width, experts=16, top_k=8,
        weights="mxfp4", swiglu_limit=0., activation="silu_bf16_product")
    selected = SimpleNamespace()
    env["__init__"](selected, geometry, "decode", rows, True)
    expected = 4 if spark and hidden == 6144 and width in (672, 704) else matching
    assert selected.down.warps == expected
    assert selected.down.max_rows == rows
