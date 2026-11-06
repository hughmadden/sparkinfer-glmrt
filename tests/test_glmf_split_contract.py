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


@pytest.mark.parametrize("capacity", [64, 128, 4096])
def test_projected_token_keys_contract_and_composition(capacity):
    g = Geometry(index_heads=32, index_kpool=4, page_rows=64, index_norm_eps=1e-6)
    source = ROOT / "b12x/integration/cuteafd/glmf_split.py"
    ns = extracted(source, {"compile_glmf_projected_index_keys_aot"}, dict(
        GLM53_FLASH=g, GLMFGeometry=Geometry, _check_rows=int,
        _ProjectedIndexKeys=Primitive, Operand=Operand, Scalar=lambda x: x,
        torch=SimpleNamespace(bfloat16="bf16", uint8="u8", int64="i64",
                              float8_e4m3fn="fp8", float32="f32"),
        compile_program=lambda launch, **kw: dict(launch=launch, **kw)))
    p = ns["compile_glmf_projected_index_keys_aot"](max_rows=capacity)
    ops = {o.name: o for o in p["operands"]}
    assert list(ops) == ["iq", "kw", "slots", "pool_slots", "k_norm_w", "k_norm_b",
                         "ape", "token_keys", "index_cache", "q_fp8", "head_weights"]
    assert ops["iq"].shape == "[rows,32,128]"
    assert ops["kw"].shape == "[rows,288]"
    assert ops["slots"].align == ops["pool_slots"].align == 8
    assert ops["token_keys"].shape == "[record_slots,256]"
    assert ops["token_keys"].role == ops["index_cache"].role == "inout"
    assert p["scalars"] == ("rows",) and "scratch" not in p
    assert p["geometry"]["token_key_row_bytes"] == 512
    assert p["key"] == (g, capacity)
    with pytest.raises(ValueError, match="GLMFGeometry"):
        ns["compile_glmf_projected_index_keys_aot"](object())

    calls = []
    def primitive(kind):
        def create(**kw):
            return lambda *args: calls.append((kind, kw, args))
        return create
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ProjectedIndexKeys")
    env = dict(cute=SimpleNamespace(jit=lambda f: f, Pointer=object), Int32=int,
        cuda=SimpleNamespace(CUstream=object), GlmfIndexPost=primitive("post"), GlmfPoolKeys=primitive("pool"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
    stage = env["_ProjectedIndexKeys"](g)
    stage(*range(13))
    assert calls == [
        ("post", dict(heads=32, eps=1e-6, weight_scale=32**-.5 * 128**-.5),
         (0, 1, 2, 4, 5, 9, 10, 7, 11, 12)),
        ("pool", dict(kpool=4, page_rows=64), (2, 3, 6, 7, 8, 11, 12))]


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


@pytest.mark.parametrize("decode,capacity", [(True, 64), (True, 128), (False, 4096)])
def test_block_fp8_projection_abi(decode, capacity):
    class Projection:
        def key(self):
            return ("original-fp8-projection",)

        def prefill_scratch_bytes(self, rows):
            return rows * 256 if not decode else 0

    ns = extracted(ROOT / "b12x/integration/cuteafd/glmf_split.py",
        {"compile_glmf_fp8_linear_aot"}, dict(
            _Fp8Linear=lambda n, k, cap, dec: SimpleNamespace(projection=Projection()),
            _check_rows=int, Operand=Operand, Scalar=lambda name: name,
            torch=SimpleNamespace(bfloat16="bf16", uint8="u8", float32="f32"),
            compile_program=lambda launch, **kw: dict(launch=launch, **kw)))
    compile_projection = ns["compile_glmf_fp8_linear_aot"]
    p = compile_projection(input_features=1536, output_features=8192,
                           max_rows=capacity, decode=decode)
    assert [(o.name, o.dtype, o.shape) for o in p["operands"][:4]] == [
        ("x", "bf16", "[rows,1536]"), ("weight", "u8", "[8192,1536]"),
        ("scales", "f32", "[64,12]"), ("out", "bf16", "[rows,8192]")]
    assert p["scalars"] == ("rows", "fp8_rows")
    assert p["scratch"]["scratch"](17) == (256 if decode else 17*256)
    assert p["geometry"]["storage"] == "row_major_e4m3"
    assert p["geometry"]["scale_block"] == 128
    for bad in (0, -128, 129, True):
        with pytest.raises(ValueError, match="H128"):
            compile_projection(input_features=bad, output_features=8192)


@pytest.mark.parametrize("n,k,warps,groups,rows,cap,row_scales,dtype,selected", [
    (4096, 16384, 4, 1, 16, 64, False, "bf16", True),
    (4096, 8192, 8, 1, 16, 64, False, "bf16", True),
    (8192, 1536, 4, 1, 16, 64, False, "bf16", False),
    (4096, 16384, 8, 1, 16, 64, False, "bf16", False),
    (4096, 16384, 4, 2, 16, 64, False, "bf16", False),
    (4096, 16384, 4, 1, 32, 64, False, "bf16", False),
    (4096, 16384, 4, 1, 16, 16, False, "bf16", False),
    (4096, 16384, 4, 1, 16, 64, True, "bf16", False),
    (4096, 16384, 4, 1, 16, 64, False, "fp32", False),
])
def test_hqa_direct_widen_only_measured_recipes(n, k, warps, groups, rows, cap,
                                               row_scales, dtype, selected):
    path = ROOT / "b12x/integration/cuteafd/_fp8_weights.py"
    tree = ast.parse(path.read_text())
    projection = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Fp8Projection")
    init = next(node for node in projection.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    condition = next(node.test for node in ast.walk(init)
                     if isinstance(node, ast.If) and any(
                         isinstance(child, ast.Attribute) and child.attr == "gemv_m16"
                         for statement in node.body for child in ast.walk(statement))
                     and "self.n" in ast.unparse(node.test))
    actual = eval(compile(ast.Expression(condition), str(path), "eval"), dict(
        self=SimpleNamespace(n=n, k=k, row_scales=row_scales), warps=warps,
        groups=groups, gemv_rows=rows, cap=cap, out_dtype=dtype,
        cutlass=SimpleNamespace(BFloat16="bf16")))
    assert bool(actual) == selected
    call = next(node for node in projection.body
                if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "rows == Int32(16)"
               for node in ast.walk(call))


@pytest.mark.parametrize("n,k,warps,groups,rows,cap,row_scales,dtype,wide_groups,selected", [
    (8192, 1536, 4, 1, 16, 64, False, "bf16", None, True),
    (8192, 1536, 4, 1, 16, 128, False, "bf16", None, True),
    (16384, 1536, 4, 1, 16, 64, False, "bf16", None, False),
    (8192, 1536, 8, 1, 16, 64, False, "bf16", None, False),
    (8192, 1536, 4, 4, 16, 64, False, "bf16", None, False),
    (8192, 1536, 4, 1, 32, 64, False, "bf16", None, False),
    (8192, 1536, 4, 1, 16, 63, False, "bf16", None, False),
    (8192, 1536, 4, 1, 16, 64, True, "bf16", None, False),
    (8192, 1536, 4, 1, 16, 64, False, "fp32", None, False),
    (8192, 1536, 4, 1, 16, 64, False, "bf16", 4, False),
])
def test_hqa_m64_only_measured_recipe(n, k, warps, groups, rows, cap,
                                    row_scales, dtype, wide_groups, selected):
    path = ROOT / "b12x/integration/cuteafd/_fp8_weights.py"
    tree = ast.parse(path.read_text())
    projection = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Fp8Projection")
    init = next(node for node in projection.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    condition = next(node.test for node in ast.walk(init)
                     if isinstance(node, ast.If) and any(
                         isinstance(child, ast.Attribute) and child.attr == "gemv_m64"
                         for statement in node.body for child in ast.walk(statement))
                     and "self.n" in ast.unparse(node.test))
    actual = eval(compile(ast.Expression(condition), str(path), "eval"), dict(
        self=SimpleNamespace(n=n, k=k, row_scales=row_scales), warps=warps,
        groups=groups, gemv_rows=rows, cap=cap, out_dtype=dtype,
        wide_groups=wide_groups, cutlass=SimpleNamespace(BFloat16="bf16")))
    assert bool(actual) == selected
    call = next(node for node in projection.body
                if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "rows == Int32(64)"
               for node in ast.walk(call))
