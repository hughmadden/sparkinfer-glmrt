"""CPU-only ABI and scalar-kernel contracts; no CUDA import or device probe."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd"


def load_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    for node in ast.walk(ast.Module(body=selected, type_ignores=[])):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = []
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), filename, "exec"), namespace)
    return namespace


@dataclass
class Operand:
    name: str
    dtype: str
    shape: str
    role: str = "in"


@dataclass
class Scalar:
    name: str
    ctype: str = "int32"
    note: str = ""


def capture(*args, **kwargs):
    return kwargs


@pytest.fixture
def module():
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32", int32="i32", int64="i64", uint8="u8"),
        Operand=Operand, Scalar=Scalar, compile_program=capture, compile_affine_norm_aot=capture,
        compile_residual_add_aot=capture, compile_rope_tables_aot=capture, compile_scaled_linear_aot=capture)
    for name in ("Mxfp8ScalePrep", "SwigluOai", "RouterScores", "Route", "IndexQuant", "KvPack", "KvSplit"):
        namespace[name] = lambda *a, **kw: (a, kw)
    names = {node.name for node in ast.parse((ROOT / "minimax.py").read_text()).body if isinstance(node, ast.FunctionDef)}
    return load_definitions("minimax.py", names | {"LINEAR_SHAPES"}, namespace)


@pytest.mark.parametrize("name,k,n", [("q",6144,8192),("kv",6144,1024),("o",8192,6144),
    ("index_q",6144,512),("index_k",6144,128),("shared_gate_up",6144,6144),
    ("shared_down",3072,6144),("dense_gate_up",6144,24576),("dense_down",12288,6144)])
def test_native_projection_shapes(module, name, k, n):
    kw = module["compile_minimax_linear_aot"](name, max_rows=129)
    assert kw == dict(input_features=k, output_features=n, encoding="mxfp8", activation_encoding="mxfp8", max_rows=129)


def test_norm_and_partial_rope_boundaries(module):
    assert module["compile_minimax_norm_aot"]() == dict(width=6144, eps=1e-6, zero_centered=True,
        input_dtype="f32", parameter_dtype="bf16")
    for heads in (1, 4, 64):
        assert module["compile_minimax_qk_norm_rope_aot"](heads=heads) == dict(width=128, heads=heads,
            eps=1e-6, zero_centered=True, rotary=64, input_dtype="bf16", parameter_dtype="bf16")
    assert module["compile_minimax_rope_tables_aot"]() == dict(rotary_span=64)


@pytest.mark.parametrize("name,kwargs", [("linear",dict(projection="bad")),
    ("qk_norm_rope",dict(heads=8)),("swiglu",dict(width=2048)),
    ("index_quant",dict(heads=4,paged=True)),("index_quant",dict(heads=2))])
def test_invalid_geometry_fails_before_compile(module, name, kwargs):
    with pytest.raises(ValueError):
        module[f"compile_minimax_{name}_aot"](**kwargs)


def test_router_preserves_f32_weights_and_unbiased_output_contract(module):
    scores = module["compile_minimax_router_scores_aot"]()
    assert [(p.name,p.dtype,p.shape) for p in scores["operands"]] == [
        ("x","bf16","[rows,6144]"),("weight","f32","[128,6144] (checkpoint FP32)"),
        ("logits","f32","[rows,128]")]
    route = module["compile_minimax_route_aot"]()
    assert [(p.name,p.dtype,p.role) for p in route["operands"]] == [
        ("logits","f32","in"),("bias","f32","in"),("ids","i32","out"),("weights","f32","out")]
    assert route["geometry"] == dict(experts=128,topk=4,factor=2.0)
    assert [s.name for s in route["scalars"]] == ["rows"]


def f32(value):
    return struct.unpack("f",struct.pack("f",value))[0]


def bf16(value):
    bits = struct.unpack("I",struct.pack("f",value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("f",struct.pack("I",bits))[0]


@pytest.mark.parametrize("gate,up", [(0.,3.),(20.,0.),(7.,9.),(-20.,2.),(-9.,-20.),(2.,.5),(.37,-.19)])
def test_swiglu_kernel_scalar_math_and_single_bf16_round(gate, up):
    namespace = dict(Float32=f32, cutlass=SimpleNamespace(min=min,max=max),
        _mul=lambda a,b:f32(a*b), _add=lambda a,b:f32(a+b),
        _sigmoid=lambda x:f32(1/(1+math.exp(-x))))
    actual = load_definitions("_minimax_kernels.py", {"_swiglu_oai"}, namespace)["_swiglu_oai"](gate,up)
    g,u = min(gate,7),max(-7,min(up,7))
    expected = g/(1+math.exp(-1.702*g))*(u+1)
    assert bf16(actual) == bf16(expected)
    if gate == -9:
        assert bf16(actual) != bf16(-7/(1+math.exp(1.702*7))*(u+1))


@pytest.mark.parametrize("n,groups", [(128,4),(256,8),(512,192)])
def test_scale_preparation_is_a_bijective_byte_permutation(n, groups):
    arch = SimpleNamespace(block_idx=lambda:(0,0,0),thread_idx=lambda:(0,0,0))
    namespace = load_definitions("_mxfp8_linear.py", {"Mxfp8ScalePrep"},
        dict(Int64=int,cute=SimpleNamespace(arch=arch)))
    kernel = namespace["Mxfp8ScalePrep"]()
    source,out = list(range(n*groups)), [None]*(n*groups)
    for i in range(n*groups):
        arch.block_idx = lambda i=i:(i//256,0,0)
        arch.thread_idx = lambda i=i:(i%256,0,0)
        kernel.kernel(source,out,groups,len(source))
    assert sorted(out) == source
    for row in range(n):
        for group in range(groups):
            offset = (row//128)*(groups//4)*512 + (group//4)*512 + (row%32)*16 + (row//32%4)*4 + group%4
            assert out[offset] == source[row*groups+group]


def test_native_stage_uses_k32_mxfp8_mma_and_runtime_card_grid():
    launch = lambda *args,**kwargs: SimpleNamespace(args=args,kwargs=kwargs)
    namespace = dict(Fp8LinearStage=object, cutlass=SimpleNamespace(BFloat16="bf16",Float32="f32",
        Float8E4M3FN="e4m3",Float8E8M0FNU="ue8m0"), _MXFP8RowsQuantLaunch=launch,
        _DenseGemmLaunch=launch, _DenseGemmPolicy=launch)
    stage = load_definitions("_mxfp8_linear.py", {"Mxfp8LinearStage"}, namespace)["Mxfp8LinearStage"](6144,128)
    assert stage.quant.args == (6144,"bf16",4,256,32,False,0.,False)
    assert stage.gemm.kwargs["ab_dtype"] == "e4m3"
    assert stage.gemm.kwargs["sf_dtype"] == "ue8m0"
    assert stage.gemm.kwargs["sf_vec_size"] == 32
    assert stage.gemm.kwargs["cluster_shape_mn"] == (1,1)
    assert stage.gemm.kwargs["alpha_is_one"]
    assert "weight_only" not in stage.gemm.kwargs


@pytest.mark.parametrize("heads,paged", [(4,False),(1,False),(1,True)])
def test_index_cache_layout(module, heads, paged):
    info = module["compile_minimax_index_quant_aot"](heads=heads,paged=paged)
    assert info["operands"][2].shape == ("[pool_pages,8448]" if paged else f"[rows,{heads},128]")
    assert [p.name for p in info["operands"]] == ["x","slots","values","scales"]
    assert info["key"] == (heads,paged)
    assert [s.name for s in info["scalars"]] == ["rows"]


def test_main_kv_cache_uses_persistent_request_head_descales(module):
    info = module["compile_minimax_kv_pack_aot"]()
    assert [p.name for p in info["operands"]] == ["k","v","slots","batch_ids","k_descale","v_descale","key_cache","value_cache"]
    assert [p.shape for p in info["operands"][-4:]] == ["[batch,4]","[batch,4]","[pool_pages,64,4,128]","[pool_pages,64,4,128]"]
    assert not info.get("scratch")


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_fused_kv_splits_by_row_without_crossing_head_groups(rows):
    arch = SimpleNamespace(block_idx=lambda:(0,0,0),thread_idx=lambda:(0,0,0))
    namespace = load_definitions("_minimax_kernels.py", {"KvSplit"},
        dict(Int64=int,cute=SimpleNamespace(arch=arch),_view=lambda x:x))
    kernel = namespace["KvSplit"]()
    source,k,v = list(range(rows*1024)), [None]*(rows*512), [None]*(rows*512)
    for i in range(rows*512):
        arch.block_idx = lambda i=i:(i//256,0,0)
        arch.thread_idx = lambda i=i:(i%256,0,0)
        kernel.kernel(source,k,v,rows*512)
    for row in range(rows):
        assert k[row*512:(row+1)*512] == source[row*1024:row*1024+512]
        assert v[row*512:(row+1)*512] == source[row*1024+512:(row+1)*1024]
