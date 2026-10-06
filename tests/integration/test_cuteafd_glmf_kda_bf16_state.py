"""GLM 5.3 Flash KDA with a BF16 recurrent state (``state_dtype="bfloat16"``): computed in FP32,
rounded to nearest even after every row of the token-sequential recurrence and of the replay
commit, and in the chunked prefill where each window of tiles stores it (or, with
``state_rounding="tile"``, after every 16-row tile).

- ``GlmfKdaRecurrent`` with a BF16 state: a window of R rows gives the bits of R serial
  single-row launches (outputs and state), R = 1..16; as a speculative verify it gives the same
  outputs and leaves the state alone, and the BF16-state commit program applied to that verify's
  replay record at k rows gives the state of k serial steps, k = 1..R.
- One row from a BF16 state gives the FP32-state outputs bit for bit and that state rounded once
  (the rounding follows the row's read-out).
- The decode and prefill KDA programs (``kda_s16_m64``, ``kda_s16_m4096``) against the FP32 ones
  from the same BF16-representable state: one decode row, and a chunked prefill inside one window
  of tiles, give the FP32 outputs bit for bit and the FP32 state rounded once; past a window, and
  with per-tile rounding, they stay within BF16 rounding of FP32.
- Per-tile rounding: a prefill split at a tile boundary gives the bits of one prefill (when the
  FP32 programs do, i.e. the projections are row-count invariant), which window rounding does not.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _geometry():
    from b12x.integration.cuteafd import GLM53_FLASH

    return GLM53_FLASH


def _recurrent(state_dtype: str):
    """``GlmfKdaRecurrent`` alone over dense q|k|v, gate and beta rows (stride ``D``, ``H``)."""
    key = ("recurrent", state_dtype)
    if key not in _PROGRAMS:
        import cutlass
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glmf_kernels import GlmfKdaRecurrent, kda_replay_layout

        g = _geometry()
        h, d = g.kda_heads, g.kda_width
        cute_type, torch_type = {"float32": (cutlass.Float32, torch.float32),
                                 "bfloat16": (cutlass.BFloat16, torch.bfloat16)}[state_dtype]
        launch = GlmfKdaRecurrent(heads=h, lower_bound=g.gate_lower_bound, qkv_width=3 * d, g_stride=d, b_stride=h,
                                  state_dtype=cute_type)
        _PROGRAMS[key] = compile_program(
            launch, name="test_glmf_kda_recurrent",
            operands=(Operand("qkv", torch.bfloat16, f"[rows,{3 * d}]"),
                      Operand("g_raw", torch.bfloat16, f"[rows,{d}]"),
                      Operand("b_raw", torch.bfloat16, f"[rows,{h}]"),
                      Operand("a_log", torch.float32, f"[{h}]", align=4),
                      Operand("dt_bias", torch.float32, f"[{d}]", align=4),
                      Operand("state", torch_type, f"[slots,{h},128,128]", "inout"),
                      Operand("slots", torch.int32, "[rows]", align=4),
                      Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
                      Operand("replay", torch.float32, f"[{kda_replay_layout(h, 3 * d)[2] // 4}]", "inout")),
            scalars=(Scalar("spec"), Scalar("rows")), key=(state_dtype,))
    return _PROGRAMS[key]


def _commit(state_dtype: str):
    key = ("commit", state_dtype)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        _PROGRAMS[key] = glmf.compile_glmf_kda_commit_aot(_geometry(), state_dtype=state_dtype)
    return _PROGRAMS[key]


def _kda(max_rows: int, fp8, state_dtype: str, state_rounding: str = "window"):
    key = ("kda", max_rows, fp8, state_dtype, state_rounding)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        _PROGRAMS[key] = glmf.compile_glmf_kda_aot(_geometry(), max_rows=max_rows, fp8=fp8, state_dtype=state_dtype,
                                                   state_rounding=state_rounding)
    return _PROGRAMS[key]


def _recurrent_inputs(rows: int, seed: int):
    """Conv outputs, gate and beta rows, decay parameters and a BF16-representable state of one
    sequence. A quarter of the gate rows sit far below zero (decay multipliers within an ulp of 1:
    the slow channels where rounding the state matters most)."""
    g = _geometry()
    h, d = g.kda_heads, g.kda_width
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device="cuda") * s  # noqa: E731
    gate = r(rows, d, s=2.0)
    gate[:, : d // 4] -= 12.0
    return dict(
        qkv=r(rows, 3 * d, s=0.5).bfloat16(), g_raw=gate.bfloat16(), b_raw=r(rows, h).bfloat16(),
        a_log=torch.log(torch.rand((h,), generator=gen, device="cuda") * 15 + 1).float(),
        dt_bias=r(d, s=0.1).float(), state=r(1, h, 128, 128, s=0.05).bfloat16())


def _replay():
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    g = _geometry()
    return torch.zeros((kda_replay_layout(g.kda_heads, 3 * g.kda_width)[2] // 4,), device="cuda")


def _run_recurrent(prog, inp, state, rows_slice: slice, spec: int, replay=None):
    rows = rows_slice.stop - rows_slice.start
    out = torch.empty((rows, _geometry().kda_width), dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    prog.launch(inp["qkv"][rows_slice], inp["g_raw"][rows_slice], inp["b_raw"][rows_slice], inp["a_log"],
                inp["dt_bias"], state, slots, out, _replay() if replay is None else replay, scalars=(spec, rows))
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("rows", range(1, 17))
def test_bf16_state_window_is_serial_steps_and_commit_replays_them(rows):
    """R rows in one launch give the bits of R single-row launches; a speculative verify gives the
    same outputs, leaves the state, and its commit at k rows stores the state of k serial steps."""
    prog = _recurrent("bfloat16")
    inp = _recurrent_inputs(rows, seed=100 + rows)
    window = inp["state"].clone()
    out_window = _run_recurrent(prog, inp, window, slice(0, rows), spec=0)
    serial = inp["state"].clone()
    out_serial, after = [], []
    for t in range(rows):
        out_serial.append(_run_recurrent(prog, inp, serial, slice(t, t + 1), spec=0))
        after.append(serial.clone())
    assert torch.equal(out_window, torch.cat(out_serial))
    assert torch.equal(window, serial)
    verify = inp["state"].clone()
    record = _replay()
    out_verify = _run_recurrent(prog, inp, verify, slice(0, rows), spec=1, replay=record)
    assert torch.equal(out_verify, out_window)
    assert torch.equal(verify, inp["state"])
    commit = _commit("bfloat16")
    g = _geometry()
    for keep in range(1, rows + 1):
        state = inp["state"].clone()
        conv = torch.zeros((1, 1, 3, 3 * g.kda_width), dtype=torch.bfloat16, device="cuda")
        tables = torch.tensor([[0], [0], [keep]], dtype=torch.int32, device="cuda")
        commit.launch(state, conv, record, tables, scalars=(1, 1, 1))
        torch.cuda.synchronize()
        assert torch.equal(state, after[keep - 1]), f"commit of {keep}/{rows} rows"


@pytest.mark.parametrize("rows", (1, 7, 16))
def test_f32_state_window_is_serial_steps(rows):
    """The same property of the FP32 state (the harness's control)."""
    prog = _recurrent("float32")
    inp = _recurrent_inputs(rows, seed=200 + rows)
    window = inp["state"].float()
    out_window = _run_recurrent(prog, inp, window, slice(0, rows), spec=0)
    serial = inp["state"].float()
    out_serial = torch.cat([_run_recurrent(prog, inp, serial, slice(t, t + 1), spec=0) for t in range(rows)])
    assert torch.equal(out_window, out_serial) and torch.equal(window, serial)


def test_bf16_state_one_row_rounds_the_f32_state_once():
    inp = _recurrent_inputs(1, seed=300)
    s32 = inp["state"].float()
    out32 = _run_recurrent(_recurrent("float32"), inp, s32, slice(0, 1), spec=0)
    s16 = inp["state"].clone()
    out16 = _run_recurrent(_recurrent("bfloat16"), inp, s16, slice(0, 1), spec=0)
    assert torch.equal(out16, out32)
    assert torch.equal(s16, s32.bfloat16())
    inp = _recurrent_inputs(16, seed=301)
    s32, s16 = inp["state"].float(), inp["state"].clone()
    out32 = _run_recurrent(_recurrent("float32"), inp, s32, slice(0, 16), spec=0)
    out16 = _run_recurrent(_recurrent("bfloat16"), inp, s16, slice(0, 16), spec=0)
    print(f"16 rows, BF16 against FP32 state: out {_rel(out16, out32):.2e} state {_rel(s16, s32):.2e}")
    assert _rel(out16, out32) < 2e-2 and _rel(s16, s32) < 2e-2


_WEIGHTS: dict = {}


def _weights():
    if _WEIGHTS:
        return _WEIGHTS
    g = _geometry()
    gen = torch.Generator(device="cuda").manual_seed(7)
    h, d, p = g.hidden, g.kda_width, g.kda_in_width
    r = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device="cuda") * s  # noqa: E731
    _WEIGHTS.update(
        w_in=r(p, h, s=0.02).bfloat16(), w_o=r(h, d, s=0.01).bfloat16(),
        w_fg=r(2, d, g.kda_head_dim, s=0.05).bfloat16(), conv_w=r(3 * d, 4, s=0.3).float(),
        a_log=torch.log(torch.rand((g.kda_heads,), generator=gen, device="cuda") * 15 + 1).float(),
        dt_bias=r(d, s=0.1).float(), o_norm=(1 + r(g.kda_head_dim, s=0.1)).bfloat16(),
        # Unread FP8 operands (fp8_rows 0 runs the BF16 weights): any aligned buffer.
        dummy=torch.zeros((1 << 16,), dtype=torch.float32, device="cuda"))
    return _WEIGHTS


def _sequence(rows: int, seed: int):
    g = _geometry()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    state = (torch.randn((1, g.kda_heads, 128, 128), generator=gen, device="cuda") * 0.01).bfloat16()
    conv = (torch.randn((1, 3, 3 * g.kda_width), generator=gen, device="cuda") * 0.1).bfloat16()
    return x, state, conv


def _run_program(prog, x, state, conv, passes, decode: bool):
    """Runs one sequence (slot 0) through ``passes`` steps of the KDA program; returns the outputs."""
    w = _weights()
    outs, start = [], 0
    for n in passes:
        slots = torch.zeros((n,), dtype=torch.int32, device="cuda")
        out = torch.empty((n, x.shape[1]), dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(prog.scratch_bytes(n)["scratch"], dtype=torch.uint8, device="cuda")
        tail = (conv, state, slots, slots, out) + ((_replay(),) if decode else ()) + (scratch,)
        prog.launch(x[start:start + n], w["w_in"], w["w_in"], w["dummy"], w["w_fg"], w["conv_w"], w["a_log"],
                    w["dt_bias"], w["o_norm"], w["w_o"], w["w_o"], w["dummy"], *tail,
                    scalars=(n, 0, 0) if decode else (n, 0))
        torch.cuda.synchronize()
        outs.append(out)
        start += n
    return torch.cat(outs)


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def test_kda_s16_decode_row_is_the_f32_row():
    """``kda_s16_m64`` against ``kda_m64``: one row, the same outputs and the state rounded once."""
    x, state, conv = _sequence(1, seed=400)
    s32, c32 = state.float(), conv.clone()
    out32 = _run_program(_kda(64, True, "float32"), x, s32, c32, [1], decode=True)
    s16, c16 = state.clone(), conv.clone()
    out16 = _run_program(_kda(64, True, "bfloat16"), x, s16, c16, [1], decode=True)
    assert torch.equal(out16, out32) and torch.equal(c16, c32)
    assert torch.equal(s16, s32.bfloat16())


def test_kda_s16_prefill_one_window_is_the_f32_prefill():
    """Inside one window of tiles (656 rows at 64 heads) the chunked recurrence keeps FP32 and the
    window's store rounds once: the FP32 program's outputs, its state rounded."""
    x, state, conv = _sequence(600, seed=500)
    s32, c32 = state.float(), conv.clone()
    out32 = _run_program(_kda(4096, "prefill", "float32"), x, s32, c32, [600], decode=False)
    s16, c16 = state.clone(), conv.clone()
    out16 = _run_program(_kda(4096, "prefill", "bfloat16"), x, s16, c16, [600], decode=False)
    assert torch.equal(out16, out32) and torch.equal(c16, c32)
    assert torch.equal(s16, s32.bfloat16())


def test_kda_s16_prefill_past_a_window_and_per_tile():
    """Past a window (rounded where it is stored) and rounding every tile: within BF16 rounding."""
    x, state, conv = _sequence(1500, seed=600)
    s32 = state.float()
    out32 = _run_program(_kda(4096, "prefill", "float32"), x, s32, conv.clone(), [1500], decode=False)
    for rounding in ("window", "tile"):
        s16 = state.clone()
        out16 = _run_program(_kda(4096, "prefill", "bfloat16", rounding), x, s16, conv.clone(), [1500],
                             decode=False)
        print(f"1500-row prefill, BF16 state ({rounding}) against FP32: out {_rel(out16, out32):.2e} "
              f"state {_rel(s16, s32):.2e}")
        assert _rel(out16, out32) < 2e-2 and _rel(s16, s32) < 2e-2


def test_kda_s16_tile_rounding_does_not_depend_on_the_chunk_plan():
    """A prefill split at a tile boundary (320 + 680 rows) against one prefill of 1000 rows."""
    x, state, conv = _sequence(1000, seed=700)

    def both(state_dtype, rounding="window"):
        prog = _kda(4096, "prefill", state_dtype, rounding)
        start = state.float() if state_dtype == "float32" else state.clone()
        runs = []
        for passes in ([1000], [320, 680]):
            s, c = start.clone(), conv.clone()
            runs.append((_run_program(prog, x, s, c, passes, decode=False), s, c))
        (o1, s1, c1), (o2, s2, c2) = runs
        same = torch.equal(o1, o2) and torch.equal(s1, s2) and torch.equal(c1, c2)
        print(f"{state_dtype} ({rounding}): one prefill against 320 + 680 rows: bitwise {same}, "
              f"out {_rel(o2, o1):.2e} state {_rel(s2, s1):.2e}")
        return same

    if not both("float32"):
        pytest.skip("the FP32 programs' projections are not row-count invariant here; nothing to isolate")
    both("bfloat16", "window")
    assert both("bfloat16", "tile")
