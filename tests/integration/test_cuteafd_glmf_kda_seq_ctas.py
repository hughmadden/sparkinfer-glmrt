"""GLM 5.3 Flash KDA recurrence with a step's sequences shared among several CTAs per head block
(``GlmfKdaRecurrent(seq_ctas=Z)``, ``compile_glmf_kda_aot(seq_ctas=Z)``): the bytes of ``seq_ctas=1``.

- ``GlmfKdaRecurrent`` alone, FP32 and BF16 state, plain (``spec`` 0) and speculative (``spec`` 1) steps, Z = 4
  and 16 against Z = 1: byte-identical outputs, state pool and replay record over steps of 16 x 8, 16 x 1, 1 x 8
  and 1 x 1 (sequences x rows), five sequences of unequal rows, 20 x 2 (more sequences than CTAs), masked rows
  (``slots`` -1) after, before and between sequences, and an all-masked step. The Z = 1 step is itself checked
  against one launch per sequence, the property that lets CTAs share the sequences.
- A CUDA graph of the Z = 16 launch captured on an all-masked step (the grid is in the graph) replays a 16 x 8
  step written into the same buffers with the bytes of the eager Z = 1 launch.
- The decode KDA programs (64 and 128 rows, both state types) built with ``seq_ctas=16`` against the default
  build: the same ABI, and byte-identical outputs, state, conv state and replay record.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_glmf_kda_bf16_state import _geometry, _recurrent_inputs, _weights

WIDE = 128  # rows of the recurrence's replay record: a step of 16 sequences x 8 rows
POOL = 24  # state slots, more than any step's sequences: some stay untouched
SEQ_CTAS = (4, 16)
_PROGRAMS: dict = {}
_POOLS: dict = {}

#: Steps as stretches of rows in step order: n > 0 is a sequence of n rows, n < 0 is -n masked rows.
STEPS = {
    "16x8": [8] * 16,
    "16x1": [1] * 16,
    "1x8": [8],
    "1x1": [1],
    "unequal": [3, 1, 8, 2, 5],
    "20x2": [2] * 20,
    "masked_tail": [4, 2, 5, -5],
    "masked_between": [-2, 3, -1, 4, 2, -3],
    "all_masked": [-8],
}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _layout(rows: int = WIDE):
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    g = _geometry()
    return kda_replay_layout(g.kda_heads, 3 * g.kda_width, rows)


def _recurrent(state_dtype: str, seq_ctas: int):
    """``GlmfKdaRecurrent`` alone over dense q|k|v, gate and beta rows and a 128-row record, its grid
    ``seq_ctas`` deep."""
    key = ("recurrent", state_dtype, seq_ctas)
    if key not in _PROGRAMS:
        import cutlass
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glmf_kernels import GlmfKdaRecurrent

        g = _geometry()
        h, d = g.kda_heads, g.kda_width
        cute_type, torch_type = {"float32": (cutlass.Float32, torch.float32),
                                 "bfloat16": (cutlass.BFloat16, torch.bfloat16)}[state_dtype]
        launch = GlmfKdaRecurrent(heads=h, lower_bound=g.gate_lower_bound, qkv_width=3 * d, g_stride=d, b_stride=h,
                                  state_dtype=cute_type, replay_rows=WIDE, seq_ctas=seq_ctas)
        _PROGRAMS[key] = compile_program(
            launch, name="test_glmf_kda_recurrent_seq_ctas",
            operands=(Operand("qkv", torch.bfloat16, f"[rows,{3 * d}]"),
                      Operand("g_raw", torch.bfloat16, f"[rows,{d}]"),
                      Operand("b_raw", torch.bfloat16, f"[rows,{h}]"),
                      Operand("a_log", torch.float32, f"[{h}]", align=4),
                      Operand("dt_bias", torch.float32, f"[{d}]", align=4),
                      Operand("state", torch_type, f"[slots,{h},128,128]", "inout"),
                      Operand("slots", torch.int32, "[rows]", align=4),
                      Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
                      Operand("replay", torch.float32, f"[{_layout()[2] // 4}]", "inout")),
            scalars=(Scalar("spec"), Scalar("rows")), key=(state_dtype, WIDE, seq_ctas))
    return _PROGRAMS[key]


def _kda(max_rows: int, state_dtype: str, seq_ctas: int):
    """The decode KDA program (``fp8=True``, run with ``fp8_rows`` 0: the BF16 projections)."""
    key = ("kda", max_rows, state_dtype, seq_ctas)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        kw = {} if seq_ctas == 1 else {"seq_ctas": seq_ctas}
        _PROGRAMS[key] = glmf.compile_glmf_kda_aot(_geometry(), max_rows=max_rows, fp8=True, state_dtype=state_dtype,
                                                   **kw)
    return _PROGRAMS[key]


def _stretches(step):
    """(first row, rows, masked) of each stretch of a step."""
    out, row = [], 0
    for n in step:
        out.append((row, abs(n), n < 0))
        row += abs(n)
    return out


def _slots(step) -> torch.Tensor:
    """A step's ``slots``: its i-th sequence in slot ``(7 * i + 3) % POOL`` (out of order, each slot in one
    stretch of rows), masked rows -1."""
    slots, sequence = [], 0
    for _, n, masked in _stretches(step):
        slots += [-1 if masked else (7 * sequence + 3) % POOL] * n
        sequence += 0 if masked else 1
    return torch.tensor(slots, dtype=torch.int32, device="cuda")


def _seq_first(step) -> torch.Tensor:
    """A step's ``seq_first``: a sequence's first row; a masked row is its own (cuteafd's padding rows)."""
    first = []
    for row, n, masked in _stretches(step):
        first += list(range(row, row + n)) if masked else [row] * n
    return torch.tensor(first, dtype=torch.int32, device="cuda")


def _pool(state_dtype: str) -> torch.Tensor:
    """A BF16-representable state pool ``[POOL, H, 128, 128]`` (cloned by every launch)."""
    if state_dtype not in _POOLS:
        gen = torch.Generator(device="cuda").manual_seed(11)
        pool = (torch.randn((POOL, _geometry().kda_heads, 128, 128), generator=gen, device="cuda") * 0.05).bfloat16()
        _POOLS[state_dtype] = pool if state_dtype == "bfloat16" else pool.float()
    return _POOLS[state_dtype]


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-identical: the bit patterns, so signed zeros and NaN payloads count."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _launch(prog, inp, slots, state, out, record, spec: int):
    rows = int(slots.numel())
    prog.launch(inp["qkv"][:rows], inp["g_raw"][:rows], inp["b_raw"][:rows], inp["a_log"], inp["dt_bias"], state,
                slots, out, record, scalars=(spec, rows))
    torch.cuda.synchronize()


def _run(prog, inp, slots, pool, spec: int):
    """One step from ``pool``: (outputs, state pool after, replay record)."""
    out = torch.zeros((int(slots.numel()), _geometry().kda_width), dtype=torch.bfloat16, device="cuda")
    record = torch.zeros((_layout()[2] // 4,), device="cuda")
    state = pool.clone()
    _launch(prog, inp, slots, state, out, record, spec)
    return out, state, record


def _run_per_stretch(prog, inp, step, slots, pool, spec: int):
    """The step as one launch per stretch (a one-sequence step each), its record rows moved to the step's."""
    g = _geometry()
    beta_off, proj_off, total = _layout()
    out = torch.zeros((int(slots.numel()), g.kda_width), dtype=torch.bfloat16, device="cuda")
    record = torch.zeros((total // 4,), device="cuda")
    rows_of = lambda r: r[: beta_off // 4].view(WIDE, g.kda_heads, 3, 128)  # noqa: E731
    beta_of = lambda r: r[beta_off // 4: proj_off // 4].view(WIDE, g.kda_heads)  # noqa: E731
    state = pool.clone()
    for row, n, _ in _stretches(step):
        part = {name: inp[name][row:row + n] for name in ("qkv", "g_raw", "b_raw")}
        part.update(a_log=inp["a_log"], dt_bias=inp["dt_bias"])
        one = torch.zeros_like(record)
        _launch(prog, part, slots[row:row + n], state, out[row:row + n], one, spec)
        rows_of(record)[row:row + n] = rows_of(one)[:n]
        beta_of(record)[row:row + n] = beta_of(one)[:n]
    return out, state, record


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("spec", [0, 1])
@pytest.mark.parametrize("name", list(STEPS))
def test_seq_ctas_gives_the_bytes_of_one_cta(name, spec, state_dtype):
    """Outputs, state pool and replay record of ``seq_ctas`` 4 and 16 against ``seq_ctas`` 1, which is one
    launch per sequence."""
    step = STEPS[name]
    slots = _slots(step)
    inp = _recurrent_inputs(int(slots.numel()), seed=4000 + 10 * list(STEPS).index(name) + spec)
    pool = _pool(state_dtype)
    one = _run(_recurrent(state_dtype, 1), inp, slots, pool, spec)
    out, state, record = one
    used = {int(slot) for slot in slots.tolist() if slot >= 0}
    assert bool(out.float().abs().sum() > 0)
    if spec:
        assert _same(state, pool) and bool(record.abs().sum() > 0)
    else:
        assert not bool(record.any())
        assert [slot for slot in range(POOL) if not _same(state[slot], pool[slot])] == sorted(used)
    apart = _run_per_stretch(_recurrent(state_dtype, 1), inp, step, slots, pool, spec)
    for what, a, b in zip(("outputs", "state pool", "replay record"), one, apart):
        assert _same(a, b), f"{what}: one launch against one per sequence, step {name}"
    for seq_ctas in SEQ_CTAS:
        got = _run(_recurrent(state_dtype, seq_ctas), inp, slots, pool, spec)
        for what, a, b in zip(("outputs", "state pool", "replay record"), one, got):
            assert _same(a, b), f"{what}: seq_ctas {seq_ctas} against 1, step {name}"


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("spec", [0, 1])
def test_a_graph_captured_on_masked_rows_replays_a_step_of_sequences(spec, state_dtype):
    """cuteafd captures its decode graphs on all-masked steps: the captured ``seq_ctas=16`` launch (grid,
    pointers, ``rows`` and ``spec`` fixed) then runs a 16 x 8 step written into the same buffers."""
    step = STEPS["16x8"]
    real = _slots(step)
    rows = int(real.numel())
    inp = _recurrent_inputs(rows, seed=5000 + spec)
    pool = _pool(state_dtype)
    want = _run(_recurrent(state_dtype, 1), inp, real, pool, spec)
    prog = _recurrent(state_dtype, 16)
    slots = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    out = torch.zeros((rows, _geometry().kda_width), dtype=torch.bfloat16, device="cuda")
    record = torch.zeros((_layout()[2] // 4,), device="cuda")
    state = pool.clone()
    _launch(prog, inp, slots, state, out, record, spec)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prog.launch(inp["qkv"], inp["g_raw"], inp["b_raw"], inp["a_log"], inp["dt_bias"], state, slots, out, record,
                    scalars=(spec, rows))
    torch.cuda.synchronize()
    assert _same(state, pool), "masked rows store no state"
    for _ in range(2):
        slots.copy_(real)
        state.copy_(pool)
        out.zero_()
        record.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for what, a, b in zip(("outputs", "state pool", "replay record"), want, (out, state, record)):
            assert _same(a, b), f"{what}: replayed seq_ctas 16 against eager seq_ctas 1"


def _run_kda(prog, x, step, conv_pool, state_pool, spec: int, record_rows: int):
    """One step of the KDA program: (outputs, state pool, conv state pool, replay record)."""
    w, g = _weights(), _geometry()
    rows = int(x.shape[0])
    out = torch.zeros((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
    record = torch.zeros((_layout(record_rows)[2] // 4,), device="cuda")
    scratch = torch.zeros(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    conv, state = conv_pool.clone(), state_pool.clone()
    prog.launch(x, w["w_in"], w["w_in"], w["dummy"], w["w_fg"], w["conv_w"], w["a_log"], w["dt_bias"], w["o_norm"],
                w["w_o"], w["w_o"], w["dummy"], conv, state, _slots(step), _seq_first(step), out, record, scratch,
                scalars=(rows, 0, spec))
    torch.cuda.synchronize()
    return out, state, conv, record


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("spec", [0, 1])
@pytest.mark.parametrize("max_rows,name", [(64, "unequal"), (64, "masked_tail"), (64, "20x2"), (128, "16x8")])
def test_kda_program_with_seq_ctas_is_the_default_program(max_rows, name, spec, state_dtype):
    """``compile_glmf_kda_aot(seq_ctas=16)``: the default program's ABI and, over a step of several
    sequences, its outputs, recurrent state, conv state and replay record."""
    g = _geometry()
    base, wide = _kda(max_rows, state_dtype, 1), _kda(max_rows, state_dtype, 16)
    assert wide.name == base.name and wide.abi == base.abi
    assert "seq_ctas" not in base.geometry and wide.geometry["seq_ctas"] == 16
    assert {k: v for k, v in wide.geometry.items() if k != "seq_ctas"} == dict(base.geometry)
    step = STEPS[name]
    rows = sum(abs(n) for n in step)
    gen = torch.Generator(device="cuda").manual_seed(6000 + rows + spec)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    conv = (torch.randn((POOL, 3, 3 * g.kda_width), generator=gen, device="cuda") * 0.1).bfloat16()
    want = _run_kda(base, x, step, conv, _pool(state_dtype), spec, max(max_rows, 64))
    got = _run_kda(wide, x, step, conv, _pool(state_dtype), spec, max(max_rows, 64))
    assert bool(want[0].float().abs().sum() > 0)
    for what, a, b in zip(("outputs", "state pool", "conv state pool", "replay record"), want, got):
        assert _same(a, b), f"{what}: seq_ctas 16 against the default, step {name}"


def test_seq_ctas_is_a_grid_dimension():
    from b12x.integration.cuteafd._glmf_kernels import GlmfKdaRecurrent

    g = _geometry()
    for bad in (0, -1, 65536):
        with pytest.raises(ValueError, match="seq_ctas"):
            GlmfKdaRecurrent(heads=g.kda_heads, lower_bound=g.gate_lower_bound, qkv_width=3 * g.kda_width,
                             g_stride=g.kda_width, b_stride=g.kda_heads, seq_ctas=bad)
