"""CPU-only native ABI exports; run in separate pytest processes per SM count.

TRELLIS_EXPORT_SMS selects synthetic device geometry (188 or 170), never a grid
constant. TRELLIS_EXPORT_DIR retains objects for exact-byte and spill auditing.
"""
import multiprocessing
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def offline_compiler():
    from b12x._lib.compile_pool import _initialize_worker
    sms = int(os.environ.get("TRELLIS_EXPORT_SMS", "188"))
    _initialize_worker(0, (12, 0), "trellis-export", "trellis-export", sms,
                       101376, 102400, multiprocessing.Array("i", [0, 0]))


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("codebook", ["e4m3", "f16", "mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
def test_native_trellis_export(bits, codebook, side, tmp_path):
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    destination = Path(os.environ.get("TRELLIS_EXPORT_DIR", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"trellis_k{bits}_{codebook}_s{side}"
    with exportable_compilation():
        program = compile_trellis_linear_aot(input_features=7168, output_features=12288,
            bits=bits, codebook=codebook, sharded_side=side, rotation_count=4, max_rows=4096)
    program.export_to_c(str(destination), stem, "cuteafd_" + stem)
    checked = validate_exported_header(program, destination / f"{stem}.h", "cuteafd_" + stem)
    assert checked["argument_count"] == 9
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["bf16", "f32", "output", "output_f32"])
@pytest.mark.parametrize("features", [7168, 163840])
def test_trellis_transform_export(kind, features, tmp_path):
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    destination = Path(os.environ.get("TRELLIS_EXPORT_DIR", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"trellis_transform_{kind}_{features}"
    with exportable_compilation():
        program = (compile_trellis_output_aot(features=features, output_dtype="f32" if kind == "output_f32" else "bf16") if kind.startswith("output") else
                   compile_trellis_input_aot(features=features, input_dtype=kind))
    program.export_to_c(str(destination), stem, "cuteafd_" + stem)
    checked = validate_exported_header(program, destination / f"{stem}.h", "cuteafd_" + stem)
    assert checked["argument_count"] == (6 if kind.startswith("output") else 7)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
def test_grouped_native_trellis_export(codebook, side, tmp_path):
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    destination = Path(os.environ.get("TRELLIS_EXPORT_DIR", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"trellis_k6_{codebook}_s{side}_g4"
    with exportable_compilation():
        program = compile_trellis_linear_aot(input_features=7168, output_features=6144,
            bits=6, codebook=codebook, sharded_side=side, rotation_count=4, max_rows=4096, groups=4)
    program.export_to_c(str(destination), stem, "cuteafd_" + stem)
    checked = validate_exported_header(program, destination / f"{stem}.h", "cuteafd_" + stem)
    assert checked["argument_count"] == 18
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["bf16", "f32", "output", "output_f32"])
def test_grouped_trellis_transform_export(kind, tmp_path):
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    destination = Path(os.environ.get("TRELLIS_EXPORT_DIR", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"trellis_transform_{kind}_g4"
    with exportable_compilation():
        program = (compile_trellis_output_aot(features=6144, output_dtype="f32" if kind == "output_f32" else "bf16", groups=4)
                   if kind.startswith("output") else
                   compile_trellis_input_aot(features=7168, input_dtype=kind, groups=4))
    program.export_to_c(str(destination), stem, "cuteafd_" + stem)
    checked = validate_exported_header(program, destination / f"{stem}.h", "cuteafd_" + stem)
    assert checked["argument_count"] == (15 if kind.startswith("output") else 16)
    assert not torch.cuda.is_initialized()
