"""Step-5 canonical compensated projections with the live rows' TMA tiles.

``pointwise.compile_bf16_projection_aot(compensated=True)`` builds the K16
compensated accumulation (``Bf16PrefillKernel(compensated=True)``) and, for a
weight shape in ``COMPENSATED_TILES_BY_ROWS``, retiles it by the live rows of
the launch. Retiling only changes the M/N tile and the stage count: every
output still sees the same ordered K16 segments and the same four-FP32-op
Kahan update per segment, so a tiled launch must emit the canonical kernel's
bytes at every row count, including across a band boundary and past the last
entry.

- The table entry is what the compensated route selects.
- The outputs are byte-identical to the untitled compensated route at 1, 16,
  17, 64, 65, 256, 1024 and 2048 rows.
- One row's output is bit-invariant across those row counts: the row is a
  function of its own inputs, not of how many rows shared the launch.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

#: (name, K = input width, N = output features, output dtype): the Step-5
#: projections of ``COMPENSATED_TILES_BY_ROWS`` (heads 64 CSA, heads 32 SWA).
SHAPES = {
    "csa_out": (12288, 4096, torch.bfloat16),
    "swa_out_partial": (6144, 4096, torch.float32),
    "swa_q": (4096, 6144, torch.bfloat16),
    "swa_kv": (4096, 384, torch.bfloat16),
    "shared_gate_up": (4096, 3072, torch.bfloat16),
}
ROWS = (1, 16, 17, 64, 65, 256, 1024, 2048)
_PROGRAMS: dict = {}
_TENSORS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _cute_dtype(dtype):
    import cutlass

    return cutlass.Float32 if dtype == torch.float32 else cutlass.BFloat16


def _tiled(name: str):
    """The shipped wrapper: one compiled program, rows are a launch scalar."""
    from b12x.integration.cuteafd import pointwise

    if ("tiled", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        _PROGRAMS[("tiled", name)] = pointwise.compile_bf16_projection_aot(
            width=width, outputs=outputs, output_dtype=dtype, compensated=True)
    return _PROGRAMS[("tiled", name)]


def _fixed(name: str):
    """Compensated ``prefill_config`` without row tiles: the route every row took before."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    if ("fixed", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), tiles_by_rows=(),
                                   compensated=True)
        _PROGRAMS[("fixed", name)] = compile_program(
            launch, name="test_step5_compensated_fixed",
            operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                      Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
                      Operand("out", dtype, f"[rows,{outputs}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key())
    return _PROGRAMS[("fixed", name)]


def _weight(name: str) -> torch.Tensor:
    if ("w", name) not in _TENSORS:
        width, outputs, _ = SHAPES[name]
        gen = torch.Generator(device="cuda").manual_seed(outputs + width)
        _TENSORS[("w", name)] = (torch.randn((outputs, width), generator=gen, device="cuda") / 16).bfloat16()
    return _TENSORS[("w", name)]


def _activations(name: str, rows: int, seed: int) -> torch.Tensor:
    width = SHAPES[name][0]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, width), generator=gen, device="cuda").bfloat16()


def _project(prog, x, w, dtype) -> torch.Tensor:
    out = torch.full((x.shape[0], w.shape[0]), float("nan"), dtype=dtype, device="cuda")
    prog.launch(x, w, out, scalars=(int(x.shape[0]),))
    torch.cuda.synchronize()
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-identical: the bit patterns, so signed zeros and NaN payloads count."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def test_every_compensated_shape_selects_its_table():
    """The compensated route's row tiles are the shipped table entry, and every entry is compensated."""
    from b12x.gemm.bf16_gemv._skinny import COMPENSATED_TILES_BY_ROWS, TmaBf16Projection

    for name, (width, outputs, dtype) in SHAPES.items():
        table = COMPENSATED_TILES_BY_ROWS[(outputs, width)]
        assert table[0][0] >= 1, f"{name}: table must cover single-row decode"
        assert table[-1][0] >= 256, f"{name}: table must cover a 256-row chunk"
        launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), tiles_by_rows=table,
                                   compensated=True)
        assert launch.tiles_by_rows == table, f"{name}: compensated route does not carry its table"
        assert launch.config["compensated"] is True, f"{name}: base overrides did not reach the default"
        for entry, tile in zip(table, launch.row_tiled.tiled):
            assert tile.kernel.compensated, f"{name}: entry {entry} lost the compensated accumulation"
            assert tile.kernel.tile_m == (entry[3] if len(entry) == 6 else 16 * entry[1]), \
                f"{name}: entry {entry} built another row tile"
        assert launch.row_tiled.default.kernel.compensated, \
            f"{name}: past-the-table default lost the compensated accumulation"


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_row_tiles_give_the_bytes_of_the_fixed_compensated_route(name, rows):
    dtype = SHAPES[name][2]
    w = _weight(name)
    x = _activations(name, rows, seed=1000 * rows + w.shape[0])
    fixed = _project(_fixed(name), x, w, dtype)
    tiled = _project(_tiled(name), x, w, dtype)
    assert _same(tiled, fixed), f"{name} at {rows} rows: row tiles against the fixed compensated route"


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_a_row_is_bit_invariant_across_row_counts(name):
    """Row 0's output does not depend on how many rows shared the launch."""
    dtype = SHAPES[name][2]
    w = _weight(name)
    want = None
    for rows in ROWS:
        x = _activations(name, rows, seed=4000)
        got = _project(_tiled(name), x, w, dtype)[0]
        if want is None:
            want = got
        else:
            assert _same(got, want), f"{name}: row 0 differs at {rows} rows"
