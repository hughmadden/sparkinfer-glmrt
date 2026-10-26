"""CPU-only EXLR8 onboarding/export contracts; no torch, CUDA or DSL imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "b12x/integration/cuteafd"
POINTERS = ("x", "pointers", "bits", "group_expert", "route_slots",
            "route_group_lane", "route_weights", "fc1", "intermediate", "fc2",
            "signs_out", "output", "group_count")
ROUTED = ("a16_serving", "group", "a16_grouped", "a8_grouped", "grouped_group",
          "a16_grouped64")
SHARED = ("shared_a16_serving", "shared_group")


class _WithoutImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None


@pytest.fixture
def contract():
    # Execute the real inventory and descriptor builders, replacing only the
    # CUDA boundary. This catches lambda arguments and ABI drift without a GPU.
    def operand(name, dtype, shape, role="in", align=16, note=""):
        return SimpleNamespace(name=name, dtype=dtype, shape=shape, role=role,
                               align=align, note=note)

    scope = dict(Operand=operand, Scalar=lambda name: SimpleNamespace(name=name),
                 torch=SimpleNamespace(**{n: n for n in (
                     "uint8", "int8", "int32", "int64", "float16", "float32", "bfloat16")}),
                 compile_program=lambda wrapper, **kw: SimpleNamespace(**kw), __doc__="contract")
    for name in ("_Exlr8Pointers", "_Exlr8ServingPointers", "_Exlr8GroupedServingPointers",
                 "_Exlr8CompactServingPointers"):
        scope[name] = lambda g: None
    grouped = ast.parse((ROOT / "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py").read_text())
    cls = next(n for n in grouped.body if isinstance(n, ast.ClassDef) and n.name == "Exlr8GroupedKernel")
    ctas = next(n.value.value for n in cls.body if isinstance(n, ast.Assign)
                and n.targets[0].id == "CTAS_PER_SM")
    scope["Exlr8GroupedKernel"] = SimpleNamespace(CTAS_PER_SM=ctas)
    scope["SERVING_VARIANTS"] = ("k1_k23_coalesced",)
    for filename, names in (
        ("exlr8.py", {"ABI_VERSION", "CTAS_PER_SM", "DEFAULT_SERVING_VARIANT",
                      "geometry", "scratch_bytes", "compile_exlr8_a16_aot"}),
        ("exlr8_glue.py", {"INPUT_FORMATS", "INGRESS_WARPS", "_Group",
                           "compile_exlr8_group_aot", "exlr8_export_programs"}),
    ):
        path = INTEGRATION / filename
        nodes = []
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(_WithoutImports().visit(node))
            elif isinstance(node, ast.ClassDef) and node.name == "_Group":
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    scope["_GroupPrefill"] = scope["_Group"]
    return SimpleNamespace(**scope)


def programs(contract, role, **geometry):
    return {name: build() for name, _, _, build in contract.exlr8_export_programs(role, **geometry)}


def test_role_inventory(contract):
    spark = programs(contract, "spark")
    coordinator = programs(contract, "coordinator")
    all_programs = programs(contract, "all")
    assert tuple(spark) == ROUTED
    assert set(coordinator) == set(ROUTED + SHARED)
    assert set(all_programs) == set(coordinator) | {"a16_decode"}
    assert len(coordinator) == 8
    with pytest.raises(ValueError, match="role"):
        contract.exlr8_export_programs("rtx")


@pytest.mark.parametrize("hidden,units,top_k,max_groups", [(6144, 16, 8, 512), (4096, 8, 4, 256)])
def test_coordinator_reuses_spark_programs(contract, hidden, units, top_k, max_groups):
    geometry = dict(hidden=hidden, units=units, top_k=top_k, max_groups=max_groups)
    spark, coordinator = (programs(contract, role, **geometry) for role in ("spark", "coordinator"))
    for name, reference in spark.items():
        candidate = coordinator[name]
        assert candidate.key == reference.key
        assert candidate.geometry == reference.geometry
        assert candidate.operands == reference.operands
        assert candidate.scalars == reference.scalars
        assert {n: size(1) for n, size in getattr(candidate, "scratch", {}).items()} == {
            n: size(1) for n, size in getattr(reference, "scratch", {}).items()}


@pytest.mark.parametrize("name", ["a16_serving", "shared_a16_serving", "a16_grouped", "a8_grouped", "a16_grouped64"])
def test_expert_program_abi(contract, name):
    program = programs(contract, "coordinator")[name]
    grouped = "grouped" in name
    expected = POINTERS + (("unit_count", "unit_map") if grouped else ())
    assert tuple(o.name for o in program.operands) == expected
    assert tuple(s.name for s in program.scalars) == (("rows", "cta_limit", "unit_limit", "l2_bytes") if grouped
                                                   else ("rows", "cta_limit"))
    assert program.geometry["abi_version"] == 3
    assert program.geometry["ctas_per_sm"] == (2 if grouped else 4)
    assert program.operands[0].dtype == "float16"
    assert program.operands[11].dtype == "bfloat16"
    assert program.operands[11].shape == "[rows,6144]"
    wrapper = "_Exlr8GroupedServingPointers" if grouped else "_Exlr8ServingPointers"
    tree = ast.parse((INTEGRATION / "exlr8.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == wrapper)
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    assert tuple(a.arg for a in call.args.args[1:]) == expected + tuple(s.name for s in program.scalars) + ("stream",)


@pytest.mark.parametrize("name,rows,group_rows,groups,input_format,row_bytes", [
    ("group", 64, 16, 512, "fp8_k32", 6336),
    ("grouped_group", 4096, 64, 768, "fp8_k32", 6336),
    ("shared_group", 64, 16, 4, "bf16", 12288),
])
def test_grouping_preserves_wire_abi(contract, name, rows, group_rows, groups, input_format, row_bytes):
    program = programs(contract, "coordinator")[name]
    assert tuple(o.name for o in program.operands) == (
        "ids", "x", "signs_in", "group_expert", "route_slots", "route_group_lane", "group_count", "activation")
    assert tuple(s.name for s in program.scalars) == ("rows", "cta_limit")
    assert program.geometry["abi_version"] == 2
    assert program.geometry["max_rows"] == rows
    assert program.geometry["max_groups"] == groups
    assert program.geometry.get("group_rows", 16) == group_rows
    assert program.geometry["input_format"] == input_format
    assert program.geometry["input_row_bytes"] == row_bytes
    assert program.operands[1].dtype == ("bfloat16" if name == "shared_group" else "uint8")
    assert program.operands[-1].dtype == "float16"
    assert program.geometry["ctas_per_sm"] == 4


@pytest.mark.parametrize("name,fc1,intermediate,fc2", [
    ("a16_serving", 805306368, 33554432, 201326592),
    ("shared_a16_serving", 6291456, 262144, 1572864),
    ("a16_grouped", 4831838208, 201326592, 1207959552),
    ("a8_grouped", 4831838208, 201326592, 1207959552),
    ("a16_grouped64", 805306368, 33554432, 201326592),
])
def test_workspace_is_capacity_bound(contract, name, fc1, intermediate, fc2):
    program = programs(contract, "coordinator")[name]
    for rows in (1, 64, 4096):
        assert {n: size(rows) for n, size in program.scratch.items()} == dict(
            fc1=fc1, intermediate=intermediate, fc2=fc2)
    for operand in program.operands:
        if operand.name in ("fc1", "intermediate", "fc2"):
            assert operand.role == "scratch"
    if name == "a16_grouped64":
        assert program.geometry["fc2_partials"] == "f32"
    if name == "a8_grouped":
        assert program.geometry["activations"] == "a8"


def test_export_builders_have_no_device_geometry_inputs(contract):
    for name, _, _, build in contract.exlr8_export_programs("coordinator"):
        program = build()
        assert not {"sms", "sm_count", "physical_sms", "l2_bytes"} & program.geometry.keys(), name
        assert "cta_limit" in {s.name for s in program.scalars}
        assert 170 not in program.key and 188 not in program.key


def test_grouped_wrapper_forwards_l2_only_to_runtime_core():
    import __future__
    tree = ast.parse((INTEGRATION / "exlr8.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Exlr8GroupedServingPointers")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    calls = []
    scope = dict(cutlass=SimpleNamespace(const_expr=lambda v: v), cute=SimpleNamespace(
        jit=lambda f: f, make_tensor=lambda pointer, layout: pointer,
        make_layout=lambda *a, **kw: None))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "wrapper", "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    wrapper = scope[cls.name]()
    wrapper.core = SimpleNamespace(serving=lambda *args: calls.append(args))
    pointers = tuple(range(15))
    for runtime in (True, False):
        wrapper.runtime_l2 = runtime
        wrapper(*pointers, 17, 376, 8, 96 * 2**20, "stream")
    assert calls[0] == pointers[:12] + (17,) + pointers[12:] + (376, 8, 96 * 2**20, "stream")
    assert calls[1] == pointers[:12] + (17,) + pointers[12:] + (376, 8, "stream")


def test_grouped_l2_scalar_is_runtime_only(contract):
    for name in ("a16_grouped", "a8_grouped", "a16_grouped64"):
        program = programs(contract, "coordinator")[name]
        assert program.scalars[-1].name == "l2_bytes"
        assert "runtime_l2_v1" in program.key
        assert "l2_bytes" not in program.geometry
    kernel = ast.parse((ROOT / "b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py").read_text())
    core = next(n for n in kernel.body if isinstance(n, ast.ClassDef) and n.name == "Exlr8GroupedKernel")
    serving = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "serving")
    assert tuple(a.arg for a in serving.args.args[-4:]) == ("cta_limit", "unit_limit", "l2_bytes", "stream")
