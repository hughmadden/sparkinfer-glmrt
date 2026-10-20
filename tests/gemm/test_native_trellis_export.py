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
