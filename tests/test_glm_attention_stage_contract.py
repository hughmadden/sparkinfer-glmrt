"""CPU contracts for format-neutral GLM primitive boundaries, without CUDA imports."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "b12x/integration/cuteafd/glm_attention.py"
TREE = ast.parse(SOURCE.read_text())


@dataclass
class Operand:
    name: str
    dtype: str
    shape: str
    role: str = "in"
    align: int = 16
    note: str = ""


class Primitive:
    def __init__(self, **kwargs):
        self.config = kwargs


class Bf16(Primitive):
    pass


class Fp8(Primitive):
    pass


G = SimpleNamespace(heads=64, index_heads=32, q_lora_rank=2048, qkv_a_width=2624,
                    kv_lora_rank=512, qk_nope_dim=192, qk_head_dim=256, v_head_dim=256,
                    latent_dim=576, norm_eps=1e-5, index_norm_eps=1e-5, page_rows=64,
                    record_bytes=656, kv_page_bytes=41984, index_page_bytes=8448)


def contract():
    names = {"_check_rows", "_rank_pack", "_query_rope", "_index_post", "_kv_projection",
             "kv_b_fp8_operands", "_w8_batched_warps", "compile_glm_attention_stage_aot"}
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(GLM53=G, GLMGeometry=SimpleNamespace, Operand=Operand,
                     Scalar=lambda name: name, GlmRankNormPackKV=Primitive,
                     GlmQueryRope=Primitive, GlmIndexPost=Primitive,
                     BatchedBf16Gemm=Bf16, BatchedFp8Gemm=Fp8,
                     torch=SimpleNamespace(bfloat16="bf16", int64="i64", float32="f32",
                                           uint8="u8", float8_e4m3fn="e4m3"),
                     compile_program=lambda launch, **kwargs: dict(launch=launch, **kwargs))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["compile_glm_attention_stage_aot"]


@pytest.mark.parametrize("stage,pointers", [
    ("rank_pack", ["qkv", "q_norm", "kv_norm", "positions", "slots", "cos_sin",
                   "q_resid", "cache"]),
    ("query_rope", ["q", "positions", "cos_sin", "query"]),
    ("index_post", ["iq", "kw", "positions", "slots", "cos_sin", "k_weight", "k_bias",
                    "q_fp8", "head_weights", "cache"]),
    ("absorb", ["a", "w", "out"]),
    ("values", ["a", "w", "out"]),
])
def test_stage_operand_order(stage, pointers):
    result = contract()(stage=stage, max_rows=4096)
    assert [o.name for o in result["operands"]] == pointers
    assert result["scalars"] == ("rows",)
    assert result.get("scratch", {}) == {}
    assert result["geometry"]["max_rows"] == 4096
    for op in result["operands"]:
        if op.name in ("positions", "kv_slots", "index_slots"):
            assert op.align == 8 and op.dtype == "i64"
        if op.name == "cos_sin":
            assert op.align == 4 and op.dtype == "f32"


@pytest.mark.parametrize("stage", ["absorb", "values"])
@pytest.mark.parametrize("mode,cap,warps", [(None, 64, 4), ("decode", 64, 4),
                                           ("prefill", 4096, 8)])
def test_batched_stage_matches_fused_geometry(stage, mode, cap, warps):
    result = contract()(stage=stage, max_rows=cap, fp8_only=mode)
    core = result["launch"]
    assert type(core) is (Bf16 if mode is None else Fp8)
    assert core.config == dict(n=512 if stage == "absorb" else 256,
                               k=192 if stage == "absorb" else 512, batch=64,
                               a_row=64 * (256 if stage == "absorb" else 512),
                               a_batch=256 if stage == "absorb" else 512,
                               o_row=64 * (576 if stage == "absorb" else 256),
                               o_batch=576 if stage == "absorb" else 256,
                               compute_warps=warps)
    if mode is not None:
        weights = result["operands"][1:3]
        assert [o.dtype for o in weights] == ["e4m3", "f32"]
        assert weights[1].shape == ("[64,512,3]" if stage == "absorb" else "[64,256,8]")


@pytest.mark.parametrize("stage", ["absorb", "query_rope"])
def test_context_split_pitch(stage):
    result = contract()(stage=stage, max_rows=64, query_row=2 * 64 * 576)
    assert result["geometry"]["query_row"] == 2 * 64 * 576
    output = result["operands"][-1]
    assert "rows 73728 apart" in output.shape
    config = result["launch"].config
    assert config["o_row" if stage == "absorb" else "query_row"] == 2 * 64 * 576


@pytest.mark.parametrize("kwargs", [
    {"stage": "unknown"}, {"stage": "rank_pack", "fp8_only": "decode"},
    {"stage": "absorb", "fp8_only": "unknown"}, {"stage": "rank_pack", "query_row": 73728},
    {"stage": "query_rope", "query_row": 36863}, {"stage": "values", "max_rows": 0},
])
def test_invalid_stage_contract(kwargs):
    with pytest.raises(ValueError):
        contract()(**dict({"max_rows": 64}, **kwargs))


@pytest.mark.parametrize("name,calls", [
    ("_Producer", ["qkv_a", "pack", "q_b", "absorb", "rope"]),
    ("_ProducerW8", ["qkv_a", "pack", "q_b", "absorb", "rope"]),
    ("_IndexProducer", ["wq", "wk", "post"]),
    ("_IndexProducerW8", ["wq", "wk", "post"]),
    ("_Output", ["uv", "o"]), ("_OutputW8", ["uv", "o"]),
])
def test_fused_launch_order_is_preserved(name, calls):
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == name)
    body = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "body")
    launches = [n.func.attr for n in ast.walk(body) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"]
    assert launches == calls


def test_batched_weight_policy_is_explicit_and_preserves_default_key():
    source = SOURCE.with_name("_glm_kernels.py")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchedBf16Gemm")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "key")]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Core", bases=[], keywords=[],
        body=methods, decorator_list=[])], type_ignores=[])), str(source), "exec"), namespace)
    namespace["Core"].tile_k = 64
    kwargs = dict(n=512, k=256, batch=64, a_row=16384, a_batch=256,
                  o_row=32768, o_batch=512)
    base = namespace["Core"](**kwargs)
    hinted = namespace["Core"](**kwargs, weight_evict_first=True)
    assert base.weight_evict_first is False
    assert base.key() == (512, 256, 64, 16384, 256, 32768, 512, 4, 128, 3)
    assert hinted.key() == base.key() + ("weight_last_m_l2",)
    kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    gates = [n for n in ast.walk(kernel) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "cutlass.const_expr(self.weight_evict_first)"]
    assert len(gates) == 1
    assert "cache_policy" not in ast.unparse(gates[0].orelse[0])


def test_production_batched_callers_do_not_implicitly_enable_weight_policy():
    root = SOURCE.parent
    enabled = []
    for path in root.glob("*.py"):
        if path.name == "_glm_kernels.py":
            continue
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "BatchedBf16Gemm":
                for keyword in call.keywords:
                    if keyword.arg == "weight_evict_first":
                        assert path.name == "glmf.py"
                        assert ast.unparse(keyword.value) == "_mla_decode_weight_evict_first(g, max_rows, prefill)"
                        enabled.append(call)
    assert len(enabled) == 2


@pytest.mark.parametrize("capacity", [1, 16, 64, 128, 1408, 2048, 4096])
@pytest.mark.parametrize("prefill", [False, True])
def test_flash_mla_weight_policy_only_measured_decode(capacity, prefill):
    source = SOURCE.with_name("glmf.py")
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_mla_decode_weight_evict_first")
    namespace = dict(GLMFGeometry=SimpleNamespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    policy = namespace[function.name]
    geometry = dict(heads=64, kv_lora_rank=512, qk_nope_dim=256, qk_head_dim=256,
                    latent_dim=512, v_head_dim=256)
    assert policy(SimpleNamespace(**geometry), capacity, prefill) == (capacity <= 64 and not prefill)
    for field in geometry:
        changed = dict(geometry)
        changed[field] //= 2
        assert not policy(SimpleNamespace(**changed), capacity, prefill)
