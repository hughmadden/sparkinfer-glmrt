"""Kimi K3 BF16 core's dense projections with the live rows' TMA tiles (``DECODE_TMA_TILES``).

The K3 export compiles these projections through
``pointwise.compile_bf16_projection_aot``, whose
``RoutedBf16Projection(max_skinny_rows=0)`` picks ``DECODE_TMA_TILES`` by weight
shape. The K3 decode profile's live rows are 2-6 (99% of projection calls), and
C16 verify lands in 16-64, so every shape here carries a 16-row band and a
row-64 band:

- The table entry is what the canonical route selects, and it retiles one
  uncompensated accumulating MMA: the outputs are byte-identical to the
  ``prefill_config`` default route at rows 1, 2, 5, 6, 16, 17, 32, 33, 64 and
  65 (each entry's first and last rows, one and several row tiles, and the
  default past the table).
- A captured CUDA graph of the canonical route replays fresh activations with
  the bytes of the eager default launch.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

#: (name, K = input width, N = output features, output dtype), the K3 BF16 core's
#: dense projections in the shapes the checkpoint ships.
SHAPES = {
    "kda_in": (7168, 2112, torch.bfloat16),
    "latent_down": (7168, 3584, torch.bfloat16),
    "kda_out": (7168, 6144, torch.bfloat16),
    "gate_up": (7168, 24752, torch.bfloat16),
    "output": (6144, 7168, torch.bfloat16),
    "shared_down": (3072, 7168, torch.bfloat16),
    "latent_up": (3584, 7168, torch.bfloat16),
}
ROWS = (1, 2, 5, 6, 16, 17, 32, 33, 64, 65)
#: ``tile_k`` 128 entries (the trailing seventh element): the same K order, a
#: wider pipelined K step.
TILE_K_ENTRIES = {
    # The shipped entry:
    "gate_up": ((128, 1, 4, 16, 32, 6, 128),),
    # And three that do not ship (those bands stay at tile_k 64): the same
    # seventh element on narrow, wide and deeper-stage tiles.
    "kda_in": ((128, 1, 4, 16, 32, 6, 128),),
    "kda_out": ((128, 1, 4, 16, 64, 4, 128),),
    "latent_up": ((128, 1, 4, 16, 128, 2, 128),),
}
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
            launch, name="test_kimi_projection_default",
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


def _tile_k(name: str):
    """A table with a wider ``tile_k``: one more MMA K step per pipeline stage."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    if ("tile_k", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        table = TILE_K_ENTRIES[name]
        launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), tiles_by_rows=table)
        _PROGRAMS[("tile_k", name)] = compile_program(
            launch, name="test_kimi_projection_tile_k",
            operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                      Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
                      Operand("out", dtype, f"[rows,{outputs}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key())
    return _PROGRAMS[("tile_k", name)]


@pytest.mark.parametrize("rows", [1, 6, 32, 64])
@pytest.mark.parametrize("name", sorted(TILE_K_ENTRIES))
def test_a_wide_k_step_keeps_the_default_route_bits(name, rows):
    """``tile_k`` 128 runs the K tiles in the same increasing order: the outputs
    must keep the ``prefill_config`` kernel's bytes."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection

    width, outputs, dtype = SHAPES[name]
    launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype),
                               tiles_by_rows=TILE_K_ENTRIES[name])
    kernels = [program.kernel for program in launch.row_tiled.tiled]
    assert kernels and all(kernel.tile_k == 128 for kernel in kernels), \
        f"{name}: the entry's tile_k did not reach the kernel"
    w = _weight(name)
    x = _activations(name, rows, seed=3000 + rows)
    want = _project(_default(name), x, w, dtype)
    got = _project(_tile_k(name), x, w, dtype)
    assert _same(got, want), f"{name} at {rows} rows: tile_k 128 against prefill_config"


def test_every_kimi_shape_selects_its_table():
    """The canonical route's row tiles are the shipped ``DECODE_TMA_TILES`` entry."""
    from b12x.gemm.bf16_gemv._skinny import DECODE_TMA_TILES, RoutedBf16Projection

    for name in sorted(SHAPES):
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


@pytest.mark.parametrize("rows", [16, 64])
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
