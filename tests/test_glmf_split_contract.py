"""CPU-only ABI contracts for Flash's projection-free primitive exports."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Operand:
    name: str
    dtype: str
    shape: str
    role: str = "in"
    align: int = 16


class Geometry(SimpleNamespace):
    pass


class Primitive:
    def __init__(self, *args, **kwargs):
        self.pack = self.absorb = self.uv = self


G = Geometry(hidden=4096, heads=64, q_lora_rank=1536, kv_lora_rank=512,
    qk_head_dim=256, qk_nope_dim=256, v_head_dim=256, latent_dim=512,
    qkv_a_width=2048, kv_page_bytes=33792)


def compiler():
    source = ROOT / "b12x/integration/cuteafd/glmf_split.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compile_glmf_stage_aot"]
    ns = dict(GLM53_FLASH=G, GLMFGeometry=Geometry, Operand=Operand,
        Scalar=lambda x: x, _RankNorm=Primitive, _MlaProducer=Primitive,
        _GlmfOutput=Primitive, _SwiGLU=Primitive, _check_rows=int,
        _align=lambda n: (n + 1023)//1024*1024,
        torch=SimpleNamespace(bfloat16="bf16", uint8="u8", int64="i64"),
        compile_program=lambda launch, **kw: dict(launch=launch, **kw))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns["compile_glmf_stage_aot"]


@pytest.mark.parametrize("stage,names", [
    ("rank_norm", ["q_a", "kv_a", "q_a_norm", "kv_a_norm", "kv_slots", "q_resid", "kv_cache", "scratch"]),
    ("absorb", ["a", "w", "out"]), ("values", ["a", "w", "out"]),
    ("swiglu", ["gate", "up", "hidden", "scratch"]),
])
def test_stage_abi(stage, names):
    p = compiler()(stage=stage, max_rows=4096)
    assert [o.name for o in p["operands"]] == names
    assert p["scalars"] == ("rows",)
    if stage == "rank_norm":
        assert p["operands"][1].shape == "[rows,512]"
        assert p["operands"][6].shape == "[pages,33792]"
        assert p["scratch"]["scratch"](17) == 17*2048*2
    if stage == "swiglu":
        assert p["scratch"]["scratch"](17) == 17*2048*4


@pytest.mark.parametrize("capacity,rows,replay", [(64, 1, None), (64, 17, None), (4096, 257, 128)])
def test_projected_index_contract(capacity, rows, replay):
    g = Geometry(index_heads=32, index_kpool=4, page_rows=64, index_norm_eps=1e-6)
    ns = extracted(ROOT / "b12x/integration/cuteafd/glmf_split.py",
        {"compile_glmf_projected_index_c_aot"}, dict(
            GLM53_FLASH=g, GLMFGeometry=Geometry, REPLAY_ROWS=64, TAIL_BYTES=1552,
            _check_rows=int, _ProjectedIndexCompact=Primitive, Operand=Operand, Scalar=lambda x: x,
            _align=lambda n: (n + 1023)//1024*1024,
            torch=SimpleNamespace(bfloat16="bf16", uint8="u8", int64="i64", int32="i32",
                                  float8_e4m3fn="fp8", float32="f32"),
            compile_program=lambda launch, **kw: dict(launch=launch, **kw)))
    compile_index = ns["compile_glmf_projected_index_c_aot"]
    p = compile_index(max_rows=capacity, verify_rows=replay)
    ops = {o.name: o for o in p["operands"]}
    assert list(ops) == ["iq", "kw", "pool_slots", "positions", "kda_slots", "seq_first",
        "k_norm_w", "k_norm_b", "ape", "tails", "replay", "index_cache", "q_fp8", "head_weights", "scratch"]
    assert ops["iq"].shape == "[rows,32,128]"
    assert ops["kw"].shape == "[rows,288]"
    assert ops["replay"].shape == f"[{64 if replay is None else replay},256]"
    assert ops["tails"].role == ops["index_cache"].role == "inout"
    assert p["scalars"] == ("rows", "spec")
    assert p["key"] == (g, capacity, 64 if replay is None else replay)
    assert p["scratch"]["scratch"](rows) == (rows*512 + 1023)//1024*1024
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="verify_rows"):
            compile_index(verify_rows=invalid)
    with pytest.raises(ValueError, match="GLMFGeometry"):
        compile_index(object())


def test_projected_index_reuses_original_post_and_pool():
    tree = ast.parse((ROOT / "b12x/integration/cuteafd/glmf_split.py").read_text())
    stage = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ProjectedIndexCompact")
    calls = [ast.unparse(n.func) for n in ast.walk(stage) if isinstance(n, ast.Call)]
    assert "GlmfIndexPostRows" in calls and "GlmfPoolKeysTail" in calls
    assert "self.post" in calls and "self.pool" in calls
    assert not any("projection" in name for name in calls)


def test_kda_reuses_original_geometry_and_clamped_activation():
    split = (ROOT / "b12x/integration/cuteafd/glmf_split.py").read_text()
    shared = (ROOT / "b12x/integration/cuteafd/kimi_split.py").read_text()
    assert "compile_kda_core_aot(g," in split
    assert "_GlmfFfn(g, inter, False, max_rows).swiglu" in split
    assert "self.core = _Kda(g," in shared
    assert 'self.full_rank_gate = kda_layout(g)["full_rank_gate"]' in shared
    assert "self.core.in_proj = _NoProjection()" in shared


def extracted(source, names, ns):
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns


def native_compilers():
    def delegated(g, **kw):
        return dict(g=g, **kw)
    return extracted(ROOT / "b12x/integration/cuteafd/glmf_split.py",
        {"compile_glmf_projected_kda_aot", "compile_glmf_fp8_stage_aot"},
        dict(GLM53_FLASH=G, GLMFGeometry=Geometry,
             compile_projected_kda_core_aot=delegated,
             compile_glm_attention_stage_aot=delegated))


@pytest.mark.parametrize("stage", ["absorb", "values"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_native_fp8_preserves_flash_geometry(stage, mode):
    p = native_compilers()["compile_glmf_fp8_stage_aot"](
        stage=stage, mode=mode, max_rows=4096, query_row=64*512)
    assert p == dict(g=G, stage=stage, fp8_only=mode, max_rows=4096, query_row=64*512)
    assert p["g"] is G


@pytest.mark.parametrize("mode", [None, "bf16", "", "unknown"])
def test_native_fp8_never_falls_back_to_bf16(mode):
    with pytest.raises(ValueError, match="mode"):
        native_compilers()["compile_glmf_fp8_stage_aot"](stage="absorb", mode=mode)


def test_native_entries_reject_other_geometry_and_stages():
    ns = native_compilers()
    for name in ("compile_glmf_projected_kda_aot", "compile_glmf_fp8_stage_aot"):
        kw = {"stage": "absorb"} if name.endswith("stage_aot") else {}
        with pytest.raises(ValueError, match="GLMFGeometry"):
            ns[name](object(), **kw)
    for stage in ("rank_norm", "swiglu", "unknown"):
        with pytest.raises(ValueError, match="stage"):
            ns["compile_glmf_fp8_stage_aot"](stage=stage)


def test_projected_flash_preserves_rounding_geometry():
    p = native_compilers()["compile_glmf_projected_kda_aot"](
        max_rows=4096, decode=False, state_dtype="bfloat16", state_rounding="window")
    assert p == dict(g=G, max_rows=4096, decode=False,
                     state_dtype="bfloat16", state_rounding="window")
    assert p["g"] is G


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("decode", [True, False])
def test_fully_projected_kda_abi(heads, decode):
    g = Geometry(kda_width=heads*128, kda_heads=heads, kda_in_width=3*heads*128+256+heads)
    def launch(*args):
        return SimpleNamespace(core=SimpleNamespace(replay_rows=64, chunked_bytes=1024))
    ns = extracted(ROOT / "b12x/integration/cuteafd/kimi_split.py",
        {"compile_projected_kda_core_aot"}, dict(
            KDAGeometry=type("KimiGeometry", (), {}), GLMFGeometry=Geometry,
            _check_rows=int, _kda_state=lambda *a: (None, "f32"),
            _ProjectedKdaCore=launch, Operand=Operand, Scalar=lambda x: x,
            torch=SimpleNamespace(bfloat16="bf16", float32="f32", int32="i32", uint8="u8"),
            kda_layout=lambda g: {"o_norm_dtype": "bfloat16"},
            kda_replay_layout=lambda h, d, r: (0, 0, h*d*r*4),
            kda_scratch_bytes=lambda g, rows, chunks: rows*g.kda_in_width*2+chunks,
            _state_geometry=lambda *a: {}, _replay_geometry=lambda *a: {},
            compile_program=lambda launch, **kw: dict(launch=launch, **kw)))
    p = ns["compile_projected_kda_core_aot"](g, max_rows=4096, decode=decode)
    operands = {o.name: o for o in p["operands"]}
    assert list(operands) == ["q", "k", "v", "decay", "gate", "beta", "conv_w", "a_log",
        "dt_bias", "o_norm", "conv_state", "state", "slots", "seq_first", "out", "replay", "scratch"]
    for name in ("q", "k", "v", "decay", "gate"):
        assert operands[name].shape == f"[rows,{heads*128}]"
        assert operands[name].dtype == "bf16"
    assert operands["beta"].shape == f"[rows,{heads}]"
    assert operands["state"].shape == f"[slots,{heads},128,128]"
    assert operands["conv_state"].shape == f"[slots,3,{3*heads*128}]"
    assert operands["state"].role == operands["conv_state"].role == "inout"
    assert operands["out"].role == "out"
    assert p["scalars"] == ("rows", "spec")
    assert p["key"] == (g, 4096, decode, "float32", "window")
    assert p["geometry"]["projected_inputs"] is True
    assert p["scratch"]["scratch"](17) == 17*g.kda_in_width*2+1024
    with pytest.raises(ValueError, match="requires"):
        ns["compile_projected_kda_core_aot"](object())
