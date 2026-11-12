"""CPU-only regression checks for the shared MiniMax duel manifest gate."""
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/bench_cuteafd_minimax.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "check_manifest")
namespace = dict(json=json, asdict=asdict)
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), namespace)
check_manifest = namespace["check_manifest"]


@dataclass
class Operand:
    name: str
    dtype: str
    shape: str
    role: str = "in"
    align: int = 16
    note: str = ""


@dataclass
class Scalar:
    name: str = "rows"
    ctype: str = "int32"
    note: str = ""


class NormProgram:
    name = "affine_norm"
    geometry = dict(width=6144, heads=1, eps=1e-6, zero_centered=True,
                    subtract_mean=False, bias=False, rotary=0)
    scalars = (Scalar(),)

    def __init__(self):
        self.geometry = self.geometry.copy()
        self.operands = [Operand("x", "float32", "[rows,1,6144]"),
                         Operand("weight", "bfloat16", "[6144]"),
                         Operand("bias", "bfloat16", "[6144] (ignored unless bias)"),
                         Operand("cos", "bfloat16", "[positions,0] (ignored unless rotary)"),
                         Operand("sin", "bfloat16", "[positions,0] (ignored unless rotary)"),
                         Operand("positions", "int64", "[rows] (ignored unless rotary)"),
                         Operand("out", "bfloat16", "[rows,1,6144]", "out")]

    @property
    def abi(self):
        return dict(pointers=[(o.name, o.dtype, o.shape, o.role) for o in self.operands],
                    scalars=[(s.name, s.ctype) for s in self.scalars], stream="stream")

    def scratch_bytes(self, rows):
        return {}


def baseline(tmp_path):
    args = SimpleNamespace(references=tmp_path, write_reference=True)
    check_manifest(args, "norm", NormProgram())
    args.write_reference = False
    return args


def test_identical_public_record_allows_changed_compiled_identity(tmp_path):
    args = baseline(tmp_path)
    program = NormProgram()
    program.compiled = object()
    check_manifest(args, "norm", program)


def test_dropped_norm_geometry_is_rejected(tmp_path):
    args = baseline(tmp_path)
    program = NormProgram()
    for key in ("heads", "bias", "rotary"):
        del program.geometry[key]
    with pytest.raises(AssertionError, match="manifest fields.*geometry"):
        check_manifest(args, "norm", program)


def test_shortened_norm_shapes_are_rejected(tmp_path):
    args = baseline(tmp_path)
    program = NormProgram()
    for operand in program.operands:
        if operand.name in ("x", "out"):
            operand.shape = "[rows,6144]"
        elif operand.name != "weight":
            operand.shape = "ignored"
    with pytest.raises(AssertionError, match="manifest fields.*abi.*operands"):
        check_manifest(args, "norm", program)


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("role", "out"),
                                          ("align", 8), ("name", "gamma")])
def test_all_operand_fields_are_checked(tmp_path, field, value):
    args = baseline(tmp_path)
    program = NormProgram()
    setattr(program.operands[1], field, value)
    with pytest.raises(AssertionError, match="manifest fields"):
        check_manifest(args, "norm", program)


def test_missing_reference_does_not_silently_skip_gate(tmp_path):
    args = SimpleNamespace(references=tmp_path, write_reference=False)
    with pytest.raises(FileNotFoundError):
        check_manifest(args, "norm", NormProgram())
