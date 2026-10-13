"""GLM 5.3 Flash decode KDA programs with the BF16 projections' TMA tiles picked by the live rows
(``compile_glmf_kda_aot(decode_tiles=True)``, ``KDA_DECODE_TILES``, ``RowTiledBf16Projection``): the bytes of
the default route.

- The in-projection ``[24896, 4096]`` and o_proj ``[4096, 8192]`` alone, ``glm_projection`` with the shape's
  table against ``glm_projection`` without, at 1, 2 and 8 rows (the skinny GEMV in both) and 9, 12, 16, 17, 32,
  33, 64, 65, 127 and 128 rows (each table entry at its first and last rows, at one and at several row tiles,
  and the rows past an entry): byte-identical outputs, and the default route's outputs within the routed
  projection's bound of the FP64 product.
- A CUDA graph of the tiled launch replays new activations written into the captured buffers with the bytes
  of the eager default launch.
- The decode KDA programs (64 and 128 rows, FP32 and BF16 state) built with ``decode_tiles`` against the
  default build: the same ABI, and byte-identical outputs, recurrent state, conv state and replay record over
  plain and speculative steps of several sequences at 16, 32, 64, 127 and 128 rows.
- Malformed tables and programs without BF16 decode projections are refused.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_bf16_state import _geometry, _weights

ROWS = (1, 2, 8, 9, 12, 16, 17, 32, 33, 64, 65, 127, 128)
POOL = 24  # state slots, more than any step's sequences
_PROGRAMS: dict = {}
_TENSORS: dict = {}

#: (program rows, step rows): a step is sequences of 8 rows, the last one shorter when the rows do not divide.
KDA_STEPS = [(64, 16), (64, 32), (64, 64), (128, 127), (128, 128)]


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _shape(name: str) -> tuple[int, int]:
    """The weight shape ``[N, K]`` of the KDA in-projection or o_proj."""
    g = _geometry()
    return {"in": (g.kda_in_width, g.hidden), "out": (g.hidden, g.kda_width)}[name]


def _tiles(name: str) -> tuple:
    from b12x.integration.cuteafd import glmf

    return glmf.KDA_DECODE_TILES[_shape(name)]


def _projection(name: str, tiled: bool):
    """``glm_projection`` of the KDA weight shape alone, with or without the shape's row tiles."""
    key = ("projection", name, tiled)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glm_kernels import glm_projection

        n, k = _shape(name)
        launch = glm_projection(n, k, row_tiles=_tiles(name) if tiled else ())
        _PROGRAMS[key] = compile_program(
            launch, name="test_glmf_kda_decode_tiles_projection",
            operands=(Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.bfloat16, f"[{n},{k}]"),
                      Operand("out", torch.bfloat16, f"[rows,{n}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key())
    return _PROGRAMS[key]


def _kda(max_rows: int, state_dtype: str, tiled: bool):
    """The decode KDA program (``fp8=True``, run with ``fp8_rows`` 0: the BF16 projections)."""
    key = ("kda", max_rows, state_dtype, tiled)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        kw = {"decode_tiles": True} if tiled else {}
        _PROGRAMS[key] = glmf.compile_glmf_kda_aot(_geometry(), max_rows=max_rows, fp8=True, state_dtype=state_dtype,
                                                   **kw)
    return _PROGRAMS[key]


def _weight(name: str) -> torch.Tensor:
    if ("w", name) not in _TENSORS:
        n, k = _shape(name)
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        _TENSORS[("w", name)] = (torch.randn((n, k), generator=gen, device="cuda") / 16).bfloat16()
    return _TENSORS[("w", name)]


def _activations(name: str, rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, _shape(name)[1]), generator=gen, device="cuda").bfloat16()


def _project(prog, x, w) -> torch.Tensor:
    out = torch.full((x.shape[0], w.shape[0]), float("nan"), dtype=torch.bfloat16, device="cuda")
    prog.launch(x, w, out, scalars=(int(x.shape[0]),))
    torch.cuda.synchronize()
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-identical: the bit patterns, so signed zeros and NaN payloads count."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _within_bound(out, x, w, chunk: int = 2048) -> bool:
    """``|out - ref| <= K * 2^-24 * sum_k |x_k w_k| + half a BF16 ulp`` of the FP64 product (the routed
    projection's contract, test_cuteafd_skinny_projection.py), a block of outputs at a time."""
    xd, xa = x.double(), x.double().abs()
    for first in range(0, w.shape[0], chunk):
        wd = w[first:first + chunk].double()
        ref = xd @ wd.t()
        bound = x.shape[1] * 2.0 ** -24 * (xa @ wd.abs().t()) + ref.abs() * 2.0 ** -8 + 1e-30
        if not bool(((out[:, first:first + chunk].double() - ref).abs() <= bound).all()):
            return False
    return True


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", ["in", "out"])
def test_row_tiles_give_the_bytes_of_the_default_route(name, rows):
    w = _weight(name)
    x = _activations(name, rows, seed=1000 * rows + w.shape[0])
    default = _project(_projection(name, False), x, w)
    assert not bool(torch.isnan(default).any()) and _within_bound(default, x, w)
    tiled = _project(_projection(name, True), x, w)
    assert _same(tiled, default), f"{name}-projection at {rows} rows: row tiles {_tiles(name)} against the default"


@pytest.mark.parametrize("rows", [16, 64, 128])
@pytest.mark.parametrize("name", ["in", "out"])
def test_a_captured_graph_replays_the_tiled_route(name, rows):
    """cuteafd captures one graph per row bucket: the captured tiled launch (pointers and ``rows`` fixed)
    then runs whatever activations are in its buffers."""
    w = _weight(name)
    x = _activations(name, rows, seed=2000 + rows)
    want = _project(_projection(name, False), x, w)
    prog = _projection(name, True)
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
        assert _same(out, want), f"{name}-projection at {rows} rows: replayed row tiles against the eager default"


def _step(rows: int) -> list[int]:
    """Rows of each sequence of a step: 8 a sequence, the last one the remainder."""
    return [8] * (rows // 8) + ([rows % 8] if rows % 8 else [])


def _tables(step) -> tuple[torch.Tensor, torch.Tensor]:
    """A step's ``slots`` (sequence i in slot ``(7 * i + 3) % POOL``, each slot one stretch of rows) and
    ``seq_first`` (the first row of each row's sequence)."""
    slots, first, row = [], [], 0
    for i, n in enumerate(step):
        slots += [(7 * i + 3) % POOL] * n
        first += [row] * n
        row += n
    return (torch.tensor(slots, dtype=torch.int32, device="cuda"),
            torch.tensor(first, dtype=torch.int32, device="cuda"))


def _pools(state_dtype: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A BF16-representable state pool ``[POOL, H, 128, 128]`` and a conv state pool (cloned by every step)."""
    if ("pools", state_dtype) not in _TENSORS:
        g = _geometry()
        gen = torch.Generator(device="cuda").manual_seed(11)
        state = (torch.randn((POOL, g.kda_heads, 128, 128), generator=gen, device="cuda") * 0.05).bfloat16()
        conv = (torch.randn((POOL, 3, 3 * g.kda_width), generator=gen, device="cuda") * 0.1).bfloat16()
        _TENSORS[("pools", state_dtype)] = (state if state_dtype == "bfloat16" else state.float(), conv)
    return _TENSORS[("pools", state_dtype)]


def _run_kda(prog, x, step, state_dtype: str, spec: int, record_rows: int):
    """One step of the KDA program: (outputs, state pool, conv state pool, replay record)."""
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    w, g = _weights(), _geometry()
    rows = int(x.shape[0])
    slots, seq_first = _tables(step)
    out = torch.zeros((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
    record = torch.zeros((kda_replay_layout(g.kda_heads, 3 * g.kda_width, record_rows)[2] // 4,), device="cuda")
    scratch = torch.zeros(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    state, conv = (pool.clone() for pool in _pools(state_dtype))
    prog.launch(x, w["w_in"], w["w_in"], w["dummy"], w["w_fg"], w["conv_w"], w["a_log"], w["dt_bias"], w["o_norm"],
                w["w_o"], w["w_o"], w["dummy"], conv, state, slots, seq_first, out, record, scratch,
                scalars=(rows, 0, spec))
    torch.cuda.synchronize()
    return out, state, conv, record


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("spec", [0, 1])
@pytest.mark.parametrize("max_rows,rows", KDA_STEPS)
def test_kda_program_with_decode_tiles_is_the_default_program(max_rows, rows, spec, state_dtype):
    """``compile_glmf_kda_aot(decode_tiles=True)``: the default program's ABI and, over a step of several
    sequences, its outputs, recurrent state, conv state and replay record."""
    g = _geometry()
    base, tiled = _kda(max_rows, state_dtype, False), _kda(max_rows, state_dtype, True)
    assert tiled.name == base.name and tiled.abi == base.abi
    assert "decode_tiles" not in base.geometry
    assert tiled.geometry["decode_tiles"] == {"in": _tiles("in"), "out": _tiles("out")}
    assert {k: v for k, v in tiled.geometry.items() if k != "decode_tiles"} == dict(base.geometry)
    assert tiled.scratch_bytes(max_rows) == base.scratch_bytes(max_rows)
    step = _step(rows)
    gen = torch.Generator(device="cuda").manual_seed(6000 + rows + spec)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    want = _run_kda(base, x, step, state_dtype, spec, max(max_rows, 64))
    got = _run_kda(tiled, x, step, state_dtype, spec, max(max_rows, 64))
    assert bool(want[0].float().abs().sum() > 0)
    for what, a, b in zip(("outputs", "state pool", "conv state pool", "replay record"), want, got):
        assert _same(a, b), f"{what}: decode_tiles against the default, {rows} rows in the {max_rows}-row program"


def test_row_tiles_and_decode_tiles_are_checked():
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection, RowTiledBf16Projection
    from b12x.integration.cuteafd import glmf
    from b12x.integration.cuteafd._glm_kernels import WIDE_SKINNY_MAX_ROWS, glm_projection

    n, k = _shape("out")
    tiles = _tiles("out")
    plain, tiled = glm_projection(n, k), glm_projection(n, k, row_tiles=tiles)
    assert plain.row_tiles == () and plain.key() == RoutedBf16Projection(n, k, max_skinny_rows=8).key()
    assert tiled.max_skinny_rows == plain.max_skinny_rows == WIDE_SKINNY_MAX_ROWS
    assert tiled.key()[:len(plain.key())] == plain.key() and tiled.key()[-1] == ("row_tiles", tiles)
    for bad, match in (
            (((32, 1, 32),), "entries"),                               # an entry is four numbers
            (((32, 1, 32, 4), (32, 4, 64, 3)), "ascending"),           # rows must ascend
            (((0, 1, 32, 4),), "ascending"),
            (((64, 4, 128, 5),), "shared memory"),                     # 123,904 B a CTA
    ):
        with pytest.raises(ValueError, match=match):
            RowTiledBf16Projection(n, k, bad)
    with pytest.raises(ValueError, match="max_skinny_rows"):
        glm_projection(n, k, row_tiles=((8, 1, 32, 4),))               # the skinny GEMV's rows
    g = _geometry()
    for kw in ({"fp8": False}, {"fp8": "prefill"}, {"fp8_only": "decode"}, {"fp8_only": "prefill"}):
        with pytest.raises(ValueError, match="decode_tiles"):
            glmf.compile_glmf_kda_aot(g, max_rows=64, decode_tiles=True, **kw)
