"""CPU export contracts; GPU QDQ and binary identity are separate gates."""
import ast
from dataclasses import dataclass
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "b12x/integration/cuteafd"


def constructor():
    tree = ast.parse((ROOT / "_glm_kernels.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlmIndexPost")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    wrapper = ast.ClassDef(name="Core", bases=[], keywords=[], body=[init], decorator_list=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "index", "exec"), ns)
    return ns["Core"]


def test_index_scale_policy_defaults_preserve_glm():
    core = constructor()(heads=32, eps=1e-5, weight_scale=1.0)
    assert (core.scale_format, core.query_amax_floor, core.key_amax_floor) == ("fp32", 0.0, 0.0)
    assert core.page_bytes == 8448 and core.rope
    tree = ast.parse((ROOT / "_glm_kernels.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlmIndexPost")
    gates = [n for n in ast.walk(cls) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "cutlass.const_expr(self.scale_format == 'ue8m0')"]
    assert len(gates) == 2
    for gate in gates:
        assert len(gate.orelse) == 1
        assert ast.unparse(gate.orelse[0]) == "scale = _fp8_scale(_warp_max(local))"


@pytest.mark.parametrize("kwargs", [dict(scale_format="other"), dict(scale_format="ue8m0"),
    dict(query_amax_floor=1e-10), dict(key_amax_floor=-1), dict(key_amax_floor=float("nan"))])
def test_invalid_scale_policy(kwargs):
    with pytest.raises(ValueError):
        constructor()(heads=32, eps=1e-5, weight_scale=1.0, **kwargs)


def test_kimi_index_post_explicit_qdq_abi():
    tree = ast.parse((ROOT / "kimi_dsa.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(dataclass=dataclass, math=math, GlmIndexPost=constructor(), _IndexProducer=object,
        torch=SimpleNamespace(bfloat16="bf16", int64="i64", float32="f32", uint8="u8", float8_e4m3fn="e4m3"),
        Operand=lambda *args, **kwargs: (args, kwargs), Scalar=lambda name: name,
        compile_program=lambda launch, **kwargs: dict(launch=launch, **kwargs))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "kimi", "exec"), ns)
    result = ns["compile_kimi_index_post_aot"](max_rows=4096)
    core = result["launch"]
    assert (core.scale_format, core.query_amax_floor, core.key_amax_floor) == ("ue8m0", 1e-10, 1e-4)
    assert core.page_bytes == 8448 and not core.rope
    assert result["scalars"] == ("rows",)
    assert result["geometry"]["scale_storage"] == "float32"
    assert [op[0][0] for op in result["operands"]] == ["iq", "kw", "positions", "slots", "cos_sin",
        "k_weight", "k_bias", "q_fp8", "head_weights", "cache"]
    for eps in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ns["KimiDsaGeometry"](index_norm_eps=eps)


def test_kimi_sparse_context_reuses_partial_and_global_selection():
    tree = ast.parse((ROOT / "kimi_dsa.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    partial = ast.unparse(functions["compile_kimi_sparse_mla_partial_aot"])
    assert "_ContextPartial(g, int(max_rows), 2048, head_begin, 48, 16, page_size=1)" in partial
    assert "'lse_base': 2" in partial
    assert "head_begin not in (0, 48)" in partial
    ordinary = ast.unparse(functions["compile_kimi_sparse_mla_aot"])
    assert "'lse_base': 'e'" in ordinary
    context = ast.unparse(functions["compile_kimi_context_index_topk_aot"])
    assert "compile_scored_index_topk_aot(g," in context
    mla = ast.parse((ROOT / "kimi_mla.py").read_text())
    cls = next(n for n in mla.body if isinstance(n, ast.ClassDef) and n.name == "_ContextPartial")
    assert "page_size=64" in ast.unparse(cls)
    assert "Int64(self.page_size * g.record_bytes)" in ast.unparse(cls)


@pytest.mark.parametrize("partial", [False, True], ids=["ordinary", "context-partial"])
def test_kimi_sparse_mla_descriptor_matches_generated_header(partial):
    tree = ast.parse((ROOT / "kimi_dsa.py").read_text())
    name = "compile_kimi_sparse_mla" + ("_partial" if partial else "") + "_aot"
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]

    @dataclass
    class Geometry:
        name: str = "kimi_k31_mla"
        norm_eps: float = 1e-5
        heads: int = 96
        record_bytes: int = 1152

    launch = SimpleNamespace(scratch_bytes=lambda rows: 0, layout={"scratch_bytes": 0})
    ns = dict(KimiMlaGeometry=Geometry, _Attention=lambda *a, **k: launch,
        _ContextPartial=lambda *a, **k: launch,
        torch=SimpleNamespace(bfloat16="bf16", uint8="u8", int32="i32", float32="f32"),
        Operand=lambda name, *a, **k: SimpleNamespace(name=name),
        Scalar=lambda name, ctype="int32": SimpleNamespace(name=name, ctype=ctype),
        compile_program=lambda launch, **kwargs: SimpleNamespace(**kwargs))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "kimi", "exec"), ns)
    program = ns[name]()

    # Derive the simulated generated header from the reused launcher's signature,
    # independently of the descriptor names the exporter passes to its validator.
    mla = ast.parse((ROOT / "kimi_mla.py").read_text())
    cls = next(n for n in mla.body if isinstance(n, ast.ClassDef)
               and n.name == ("_ContextPartial" if partial else "_Attention"))
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    types = {"cute.Pointer": "void *", "Int32": "int32_t ", "Int64": "int64_t ",
             "cuda.CUstream": "cudaStream_t "}
    arguments = call.args.args[1:]
    assert all(a.annotation is not None for a in arguments), "exported launcher requires explicit ABI types"
    params = [types[ast.unparse(a.annotation)] + a.arg for a in arguments]
    symbol = "cuteafd_" + program.name
    header = (f"static inline int32_t cute_dsl_{symbol}_wrapper("
              f"{symbol}_Kernel_Module_t *module, " + ", ".join(params) + ") {\n"
              f"void *args[{len(arguments) + 1}] = {{"
              + ",".join("&" + a.arg for a in arguments) + ",&ret};\n}\n"
              "void _mlir_test(void **args, int32_t num_args);\n")
    common = ast.parse((ROOT / "_common.py").read_text())
    validator = next(n for n in common.body if isinstance(n, ast.FunctionDef)
                     and n.name == "validate_exported_header")
    scope = dict(AotProgram=object, Path=Path, re=re,
                 _C_TYPES={"int32": "int32_t", "int64": "int64_t", "float32": "float"})
    exec(compile(ast.Module(body=[validator], type_ignores=[]), "header", "exec"), scope)
    checked = scope["validate_exported_header"](program, header, symbol)
    assert checked == {"symbol": "_mlir_test", "argument_count": len(arguments)}
    program.operands[2].name = "indices"
    with pytest.raises(ValueError, match="unexpected wrapper signature"):
        scope["validate_exported_header"](program, header, symbol)


def test_kimi_rank_norm_accepts_checkpoint_epsilon_without_changing_k3_default():
    tree = ast.parse((ROOT / "kimi_mla.py").read_text())
    geometry = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KimiMlaGeometry")
    ns = dict(dataclass=dataclass, math=math)
    exec(compile(ast.Module(body=[geometry], type_ignores=[]), "mla", "exec"), ns)
    cls = ns["KimiMlaGeometry"]
    assert cls().norm_eps == 1e-6
    assert cls(norm_eps=1e-5).norm_eps == 1e-5
    for eps in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            cls(norm_eps=eps)
