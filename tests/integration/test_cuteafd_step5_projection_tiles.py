"""Step-5 canonical BF16 projections with the live rows' TMA tiles (``DECODE_TMA_TILES``).

The Step-5 exporter (``export_b12x_step5_aot.py --group projections``) compiles
every projection through ``pointwise.compile_bf16_projection_aot``, whose
``RoutedBf16Projection(max_skinny_rows=0)`` picks ``DECODE_TMA_TILES`` by weight
shape. These tests hold the shapes the Step-5 engine measures (heads 64 CSA,
heads 32 SWA):

- The table entry is what the canonical route selects, and it retiles one
  uncompensated accumulating MMA: the outputs are byte-identical to the
  ``prefill_config`` default route at 1, 16, 17, 64, 65 and 256 rows (each
  entry's band, one and several row tiles, and the default past the table).
- A captured CUDA graph of the canonical route replays fresh activations with
  the bytes of the eager default launch.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

#: (name, K = input width, N = output features, output dtype), the Step-5
#: projections of export_b12x_step5_aot.py at heads 64 and 32.
SHAPES = {
    "csa_out": (12288, 4096, torch.bfloat16),
    "swa_out_partial": (6144, 4096, torch.float32),
    "swa_q": (4096, 6144, torch.bfloat16),
    "swa_kv": (4096, 384, torch.bfloat16),
    "shared_gate_up": (4096, 3072, torch.bfloat16),
}
#: Shapes whose live rows ``DECODE_TMA_TILES`` retiles. The q projection
#: (4096 -> 6144) measured 1.06x cold at 80% of the roofline, below the 10%
#: retain bar, so it keeps ``prefill_config``; its canonical route is still
#: held to that default by the bitwise tests.
TILED = ("csa_out", "swa_out_partial", "swa_kv", "shared_gate_up")
ROWS = (1, 16, 17, 64, 65, 256)
_PROGRAMS: dict = {}
_TENSORS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _cute_dtype(dtype):
    import cutlass

    return cutlass.Float32 if dtype == torch.float32 else cutlass.BFloat16


def _canonical(name: str):
    """The shipped wrapper: one compiled program, rows are a launch scalar."""
    from b12x.integration.cuteafd import pointwise

    if ("canonical", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        _PROGRAMS[("canonical", name)] = pointwise.compile_bf16_projection_aot(
            width=width, outputs=outputs, output_dtype=dtype)
    return _PROGRAMS[("canonical", name)]


def _default(name: str):
    """``prefill_config`` without row tiles: the route every row took before the table."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    if ("default", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), tiles_by_rows=())
        _PROGRAMS[("default", name)] = compile_program(
            launch, name="test_step5_projection_default",
            operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                      Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
                      Operand("out", dtype, f"[rows,{outputs}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key())
    return _PROGRAMS[("default", name)]


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


def test_every_step5_shape_selects_its_table():
    """The canonical route's row tiles are the shipped ``DECODE_TMA_TILES`` entry."""
    from b12x.gemm.bf16_gemv._skinny import DECODE_TMA_TILES, RoutedBf16Projection

    assert (6144, 4096) not in DECODE_TMA_TILES, "q projection stays on prefill_config"
    for name in TILED:
        width, outputs, dtype = SHAPES[name]
        table = DECODE_TMA_TILES[(outputs, width)]
        route = RoutedBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), max_skinny_rows=0)
        assert route.tiles_by_rows == table, f"{name}: canonical route does not carry its table"
        assert table and table[0][0] >= 1, f"{name}: table must cover single-row decode"
        assert table[-1][0] >= 64, f"{name}: table must cover decode/verify widths"


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_row_tiles_give_the_bytes_of_the_default_route(name, rows):
    dtype = SHAPES[name][2]
    w = _weight(name)
    x = _activations(name, rows, seed=1000 * rows + w.shape[0])
    default = _project(_default(name), x, w, dtype)
    tiled = _project(_canonical(name), x, w, dtype)
    assert _same(tiled, default), f"{name} at {rows} rows: row tiles against prefill_config"


@pytest.mark.parametrize("rows", [16, 64, 256])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_a_captured_graph_replays_the_canonical_route(name, rows):
    """cuteafd captures one graph per row bucket: the capture's pointers and ``rows``
    then run whatever activations are in its buffers."""
    dtype = SHAPES[name][2]
    w = _weight(name)
    x = _activations(name, rows, seed=2000 + rows)
    want = _project(_default(name), x, w, dtype)
    prog = _canonical(name)
    buffer = torch.zeros_like(x)
    out = torch.zeros_like(want)
    prog.launch(buffer, w, out, scalars=(rows,))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prog.launch(buffer, w, out, scalars=(rows,))
    torch.cuda.synchronize()
    for _ in range(2):
        buffer.copy_(x)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, want), f"{name} at {rows} rows: replayed canonical route against the eager default"
