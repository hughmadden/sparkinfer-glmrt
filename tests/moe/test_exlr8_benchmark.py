"""CPU regressions for the EXLR8 microbench's external control arm."""
import hashlib

import pytest
import torch

from benchmarks import benchmark_exlr8_a16 as bench
from b12x.integration.cuteafd import exlr8 as adapter


def test_control_cache_tracks_exact_source_bytes(tmp_path, monkeypatch):
    cache, keys = {}, []

    class Pointers:
        pass

    def compile_program(launch, *, name, key, **kwargs):
        keys.append(key)
        return cache.setdefault((name, key), launch.core.marker)

    monkeypatch.setattr(adapter, "_Exlr8Pointers", Pointers)
    monkeypatch.setattr(adapter, "compile_program", compile_program)
    path = tmp_path / "control.py"
    source = ("class Exlr8A16Kernel:\n"
              "    def __init__(self, *args):\n"
              "        self.marker = 'A1'\n")
    path.write_text(source)
    assert bench.compile_control(path) == "A1"
    assert keys[-1][-1] == hashlib.sha256(source.encode()).hexdigest()
    # Same path and length also catch stale Python bytecode, not only JIT reuse.
    source = source.replace("A1", "B2")
    path.write_text(source)
    assert bench.compile_control(path) == "B2"
    assert keys[-1][-1] == hashlib.sha256(source.encode()).hexdigest()
    assert len(cache) == 2
    other = tmp_path / "same_content.py"
    other.write_text(source)
    assert bench.compile_control(other) == "B2"
    assert keys[-1] == keys[-2]
    assert len(cache) == 2
    assert adapter._Exlr8Pointers is Pointers
    assert adapter.compile_program is compile_program


def test_control_restores_adapter_after_compile_failure(tmp_path, monkeypatch):
    class Pointers:
        pass

    def compile_program(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(adapter, "_Exlr8Pointers", Pointers)
    monkeypatch.setattr(adapter, "compile_program", compile_program)
    path = tmp_path / "control.py"
    path.write_text("class Exlr8A16Kernel:\n    def __init__(self, *args): pass\n")
    with pytest.raises(RuntimeError, match="compile failed"):
        bench.compile_control(path)
    assert adapter._Exlr8Pointers is Pointers
    assert adapter.compile_program is compile_program


@pytest.mark.parametrize("written_rows", [0, 1])
def test_control_cannot_inherit_candidate_output(written_rows):
    expected = torch.ones((2, 4))
    tensors = [None] * 7 + [torch.ones(8) for _ in range(3)] + [expected.clone()]

    class PartialControl:
        def replay(self):
            tensors[-1][:written_rows].fill_(1.0)

    with pytest.raises(RuntimeError, match="differs bitwise"):
        bench.validate_control(PartialControl(), tensors, expected, 2)


def test_control_poison_precedes_replay():
    expected = torch.ones((2, 4))
    tensors = [None] * 7 + [torch.ones(8) for _ in range(3)] + [torch.ones((3, 4))]

    class Control:
        def replay(self):
            assert all(bool(torch.isnan(t).all()) for t in tensors[7:])
            tensors[-1][:2].copy_(expected)

    bench.validate_control(Control(), tensors, expected, 2)
    assert bool(torch.isnan(tensors[-1][2:]).all())
