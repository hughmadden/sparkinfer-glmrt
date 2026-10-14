"""Arch lint: experimental kernels may not target flagship datacenter archs.

Proposal rule §1: experimental ops are for client GPU SKUs (SM12x).  This
scan forbids SM90/SM100/SM103/SM110 arch spellings, gencode flags, capability
helpers, and capability tuples anywhere under flashinfer/experimental/.

Escape hatch for a legitimately needed mention (e.g. a comment explaining a
ported heuristic): end the line with ``# arch-lint: allow``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "flashinfer" / "experimental"

ALLOW_MARKER = "arch-lint: allow"

FORBIDDEN = [
    re.compile(pattern)
    for pattern in (
        r"\bsm_?90a?\b",
        r"\bsm_?100[af]?\b",
        r"\bsm_?103a?\b",
        r"\bsm_?110a?\b",
        r"\bcompute_(90|100|103|110)\b",
        r"is_sm(90|100|103|110)a?_supported",
        r"\(\s*9\s*,\s*0\s*\)",
        r"\(\s*10\s*,\s*[03]\s*\)",
    )
]


def test_no_datacenter_arch_targets():
    violations = []
    for suffix in ("*.py", "*.cu", "*.cuh", "*.cpp", "*.h"):
        for path in sorted(EXP.rglob(suffix)):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if ALLOW_MARKER in line:
                    continue
                for pattern in FORBIDDEN:
                    if pattern.search(line):
                        violations.append(
                            f"  {path.relative_to(REPO)}:{lineno}: {line.strip()!r} "
                            f"(matched {pattern.pattern!r})"
                        )
    assert not violations, (
        "datacenter arch (SM90/SM100/SM103/SM110) references are forbidden in "
        "flashinfer/experimental (rule §1); append '# arch-lint: allow' only "
        "for justified mentions:\n" + "\n".join(violations)
    )


def test_dsv4_padding_guards_dominate_state_accesses():
    """CPU structural gate; numerical/state equality is tested on SM12x separately."""
    import ast

    root = Path(__file__).resolve().parents[2] / "b12x" / "attention"
    producer = ast.parse((root / "dsv4_producer" / "_cute.py").read_text())
    compressor = ast.parse((root / "dsv4_compressor" / "_cute.py").read_text())
    classes = {node.name: node for tree in (producer, compressor)
               for node in tree.body if isinstance(node, ast.ClassDef)}
    pack = classes["DSV4RankNormPackKV"]
    kernel = next(node for node in pack.body if isinstance(node, ast.FunctionDef) and node.name == "kernel")
    guard = next(node for node in kernel.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "slot >= Int64(0)")
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert "cache.toint()" in guarded and "words[0] = packed" in guarded
    pool = classes["DSV4CompressorPool"]
    kernel = next(node for node in pool.body if isinstance(node, ast.FunctionDef) and node.name == "kernel")
    decode = next(node for node in kernel.body if isinstance(node, ast.If)
                  and "self.mode == 'decode'" in ast.unparse(node.test))
    guard = next(node for node in decode.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "active != Int32(0)")
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert "_ld(a0, unit)" in guarded and "_state_ptr" in guarded
    mode = next(node for node in ast.walk(decode) if isinstance(node, ast.If)
                and "self.mode == 'prefill'" in ast.unparse(node.test))
    continuation = mode.orelse
    slot_guard = next(node for node in continuation if isinstance(node, ast.If))
    assert ast.unparse(slot_guard.test) == "active != Int32(0)"
    sequence_guard = next(node for node in slot_guard.body if isinstance(node, ast.If))
    guarded = ast.unparse(ast.Module(body=sequence_guard.body, type_ignores=[]))
    assert "_ld(a5, seq_slot)" in guarded and "_ld(a6, seq_slot)" in guarded
    # No metadata array may be indexed with a negative group sequence slot.
    assert "_ld(a7, seq_slot)" in ast.unparse(slot_guard)
    assert ast.unparse(continuation[1].value) == "Int32(seq_slot >= Int32(0))"
    finalize = classes["DSV4CompressorFinalize"]
    kernel = next(node for node in finalize.body if isinstance(node, ast.FunctionDef) and node.name == "kernel")
    guards = [node for node in ast.walk(kernel) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "sequence >= Int64(0)"]
    assert len(guards) == 1
    assert "_state_store" in ast.unparse(guards[0])
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_state_store"
               for node in compressor.body), "state helpers must remain module-level"
