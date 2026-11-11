"""CPU address oracles for physical DSA pages beyond signed/unsigned 32-bit offsets."""
from __future__ import annotations

import __future__
import ast
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "b12x/attention/dsa_indexer/kernel.py"


@pytest.mark.parametrize("boundary", [2**31, 2**32])
def test_scalar_key_load_widens_before_packed_page_stride(boundary):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_load_index_k_page_scalar")
    function.decorator_list = []
    env = dict(Int32=np.int32, Int64=np.int64, _PAGE_SIZE=64, _INDEX_HEAD_DIM=128,
               _PAGED_THREADS_PER_CTA=128)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), env)
    addresses, shared = [], {}

    class Cache:
        def __getitem__(self, coordinates):
            page, row, col = coordinates
            address = page * np.int32(8448) + row * np.int32(128) + col
            addresses.append(int(address))
            return address & 255

    page = np.int32(boundary // 8448 + 13)
    for thread in range(128):
        env[function.name](Cache(), page, shared, np.int32(thread))
    expected = int(page) * 8448
    assert sorted(addresses) == list(range(expected, expected + 8192))
    assert min(addresses) > boundary
    assert len(shared) == 8192


@pytest.mark.parametrize("boundary", [2**31, 2**32])
@pytest.mark.parametrize("scorer", ["DSAPagedLogitsKernel", "DSAScheduledSingleRowLogitsKernel",
                                  "DSAScheduledMultiRowLogitsKernel"])
def test_scale_load_widens_before_packed_page_stride(boundary, scorer):
    tree = ast.parse(SOURCE.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == scorer)
    loads = [node for node in ast.walk(owner) if isinstance(node, ast.Subscript)
             and isinstance(node.value, ast.Name) and node.value.id == "k_scales"]
    assert len(loads) == 1
    load = compile(ast.Expression(body=loads[0]), str(SOURCE), "eval")
    addresses = []

    class Scales:
        def __getitem__(self, coordinates):
            page, row = coordinates
            element = page * np.int32(2112) + row
            addresses.append(int(element) * 4)
            return 1.0

    # CuTe strides are in elements: exercise overflow before the F32 byte conversion.
    page = np.int32(boundary // 2112 + 13)
    for row in range(64):
        eval(load, dict(Int64=np.int64, k_scales=Scales(), page_id=page, scale_idx=np.int32(row)))
    expected = int(page) * 2112 * 4
    assert addresses == list(range(expected, expected + 256, 4))
    assert min(addresses) > boundary
