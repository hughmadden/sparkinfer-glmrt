"""GLM 5.3 Flash wide decode programs (128 rows; cuteafd serve ``--decode-rows 128``): the decode
programs a verify of 65 to 128 rows runs, and the replay commits over their 128-row records.

- ``GlmfKdaRecurrent`` over a 128-row record (``replay_rows=128``), with a BF16 and an FP32 state:
  a window of R rows (up to 128) gives the bits of R serial single-row launches, outputs and state;
  as a speculative verify it gives the same outputs and leaves the state alone; and the 128-row
  commit (``compile_glmf_kda_commit_aot(replay_rows=128)``) at k rows stores the state of k serial
  steps, in each layer of a two-layer record (the 128-row layer stride).
- ``kda_m128`` / ``kda_s16_m128`` (the decode structures at 128 rows) against ``kda_m64`` /
  ``kda_s16_m64`` on up to 64 rows: the same outputs, state and conv state bit for bit, and the
  same recorded rows, each in its own record layout.
- Through ``kda_m128`` / ``kda_s16_m128``: a 128-row speculative verify committed in full equals
  a plain 128-row verify (recurrent and conv state), and a commit at k rows does not depend on the
  rejected rows after k.
- ``kda_commit_c_m128``: its KDA half equals ``kda_commit_m128`` and its tail half rebuilds every
  sequence's index tail from a 128-row key | gate record, in each of two DSA layers.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

WIDE = 128
TAIL_BYTES = 1552
_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _g():
    from b12x.integration.cuteafd import GLM53_FLASH

    return GLM53_FLASH


def _layout(rows: int):
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

    g = _g()
    return kda_replay_layout(g.kda_heads, 3 * g.kda_width, rows)


def _recurrent(state_dtype: str, replay_rows: int):
    """``GlmfKdaRecurrent`` alone over dense q|k|v, gate and beta rows, with a record of ``replay_rows``."""
    key = ("recurrent", state_dtype, replay_rows)
    if key not in _PROGRAMS:
        import cutlass
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glmf_kernels import GlmfKdaRecurrent

        g = _g()
        h, d = g.kda_heads, g.kda_width
        cute_type, torch_type = {"float32": (cutlass.Float32, torch.float32),
                                 "bfloat16": (cutlass.BFloat16, torch.bfloat16)}[state_dtype]
        launch = GlmfKdaRecurrent(heads=h, lower_bound=g.gate_lower_bound, qkv_width=3 * d, g_stride=d, b_stride=h,
                                  state_dtype=cute_type, replay_rows=replay_rows)
        _PROGRAMS[key] = compile_program(
            launch, name="test_glmf_kda_recurrent_wide",
            operands=(Operand("qkv", torch.bfloat16, f"[rows,{3 * d}]"),
                      Operand("g_raw", torch.bfloat16, f"[rows,{d}]"),
                      Operand("b_raw", torch.bfloat16, f"[rows,{h}]"),
                      Operand("a_log", torch.float32, f"[{h}]", align=4),
                      Operand("dt_bias", torch.float32, f"[{d}]", align=4),
                      Operand("state", torch_type, f"[slots,{h},128,128]", "inout"),
                      Operand("slots", torch.int32, "[rows]", align=4),
                      Operand("out", torch.bfloat16, f"[rows,{d}]", "out"),
                      Operand("replay", torch.float32, f"[{_layout(replay_rows)[2] // 4}]", "inout")),
            scalars=(Scalar("spec"), Scalar("rows")), key=(state_dtype, replay_rows))
    return _PROGRAMS[key]


def _compiled(kind: str, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import glmf

        fn = {"kda": glmf.compile_glmf_kda_aot, "commit": glmf.compile_glmf_kda_commit_aot,
              "commit_c": glmf.compile_glmf_kda_commit_c_aot}[kind]
        _PROGRAMS[key] = fn(_g(), **kw)
    return _PROGRAMS[key]


def _kda(max_rows: int, state_dtype: str):
    """The decode KDA program (``fp8=True``, run with ``fp8_rows`` 0: the BF16 projections)."""
    kw = {} if state_dtype == "float32" else {"state_dtype": state_dtype}
    return _compiled("kda", max_rows=max_rows, fp8=True, **kw)


def _commit(state_dtype: str, replay_rows: int = WIDE, compact: bool = False):
    kw = {} if state_dtype == "float32" else {"state_dtype": state_dtype}
    return _compiled("commit_c" if compact else "commit", replay_rows=replay_rows, **kw)


def _recurrent_inputs(rows: int, seed: int):
    """Conv outputs, gate and beta rows, decay parameters and a BF16-representable state of one
    sequence; a quarter of the gate rows far below zero (decay multipliers within an ulp of 1)."""
    g = _g()
    h, d = g.kda_heads, g.kda_width
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device="cuda") * s  # noqa: E731
    gate = r(rows, d, s=2.0)
    gate[:, : d // 4] -= 12.0
    return dict(
        qkv=r(rows, 3 * d, s=0.5).bfloat16(), g_raw=gate.bfloat16(), b_raw=r(rows, h).bfloat16(),
        a_log=torch.log(torch.rand((h,), generator=gen, device="cuda") * 15 + 1).float(),
        dt_bias=r(d, s=0.1).float(), state=r(1, h, 128, 128, s=0.05).bfloat16())


def _run_recurrent(prog, inp, state, rows: slice, spec: int, replay):
    n = rows.stop - rows.start
    out = torch.empty((n, _g().kda_width), dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros((n,), dtype=torch.int32, device="cuda")
    prog.launch(inp["qkv"][rows], inp["g_raw"][rows], inp["b_raw"][rows], inp["a_log"], inp["dt_bias"], state, slots,
                out, replay, scalars=(spec, n))
    torch.cuda.synchronize()
    return out


def _state(inp, state_dtype: str):
    return inp["state"].clone() if state_dtype == "bfloat16" else inp["state"].float()


@pytest.mark.parametrize("state_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("rows", [65, 100, 128])
def test_a_wide_window_is_serial_steps_and_the_wide_commit_replays_them(state_dtype, rows):
    """R rows in one launch give the bits of R single-row launches; a speculative verify gives the
    same outputs and leaves the state; the 128-row commit at k rows stores the state of k serial
    steps, in each layer of a two-layer record."""
    g = _g()
    prog = _recurrent(state_dtype, WIDE)
    layers = [_recurrent_inputs(rows, seed=1000 * (layer + 1) + rows) for layer in range(2)]
    record = torch.zeros((2, _layout(WIDE)[2] // 4), device="cuda")
    after = []
    for layer, inp in enumerate(layers):
        window = _state(inp, state_dtype)
        out_window = _run_recurrent(prog, inp, window, slice(0, rows), 0, torch.zeros_like(record[layer]))
        serial = _state(inp, state_dtype)
        out_serial, states = [], []
        for t in range(rows):
            out_serial.append(_run_recurrent(prog, inp, serial, slice(t, t + 1), 0, torch.zeros_like(record[layer])))
            states.append(serial.clone())
        assert torch.equal(out_window, torch.cat(out_serial)), f"layer {layer}"
        assert torch.equal(window, serial), f"layer {layer}"
        verify = _state(inp, state_dtype)
        out_verify = _run_recurrent(prog, inp, verify, slice(0, rows), 1, record[layer])
        assert torch.equal(out_verify, out_window) and torch.equal(verify, _state(inp, state_dtype))
        after.append(states)
    commit = _commit(state_dtype)
    for keep in sorted({1, 2, 63, 64, 65, rows - 1, rows}):
        state = torch.stack([_state(inp, state_dtype)[0] for inp in layers]).unsqueeze(1)
        conv = torch.zeros((2, 1, 3, 3 * g.kda_width), dtype=torch.bfloat16, device="cuda")
        tables = torch.tensor([[0], [0], [keep]], dtype=torch.int32, device="cuda")
        commit.launch(state, conv, record, tables, scalars=(1, 2, 1))
        torch.cuda.synchronize()
        for layer in range(2):
            assert torch.equal(state[layer], after[layer][keep - 1]), f"commit of {keep}/{rows} rows, layer {layer}"


_WEIGHTS: dict = {}


def _weights():
    if _WEIGHTS:
        return _WEIGHTS
    g = _g()
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


def _sequence(rows: int, seed: int, state_dtype: str):
    g = _g()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    state = (torch.randn((1, g.kda_heads, 128, 128), generator=gen, device="cuda") * 0.01).bfloat16()
    conv = (torch.randn((1, 3, 3 * g.kda_width), generator=gen, device="cuda") * 0.1).bfloat16()
    return x, (state if state_dtype == "bfloat16" else state.float()), conv


def _step(prog, x, state, conv, record, spec: int):
    """One decode step of one sequence (slot 0) over all of ``x``; returns the outputs."""
    w, n = _weights(), x.shape[0]
    slots = torch.zeros((n,), dtype=torch.int32, device="cuda")
    out = torch.empty((n, x.shape[1]), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(prog.scratch_bytes(n)["scratch"], dtype=torch.uint8, device="cuda")
    prog.launch(x, w["w_in"], w["w_in"], w["dummy"], w["w_fg"], w["conv_w"], w["a_log"], w["dt_bias"], w["o_norm"],
                w["w_o"], w["w_o"], w["dummy"], conv, state, slots, slots, out, record, scratch, scalars=(n, 0, spec))
    torch.cuda.synchronize()
    return out


def _record(rows: int):
    return torch.zeros((_layout(rows)[2] // 4,), device="cuda")


def _record_rows(record, record_rows: int, rows: int):
    """The first ``rows`` rows of a record of ``record_rows``: k | decay | v, beta, the in-projection."""
    g = _g()
    h, c = g.kda_heads, 3 * g.kda_width
    beta, proj, _ = _layout(record_rows)
    raw = record.view(torch.uint8)
    kdv = raw[:beta].view(torch.float32).view(record_rows, h, 3, 128)[:rows]
    betas = raw[beta:proj].view(torch.float32).view(record_rows, h)[:rows]
    projs = raw[proj:proj + record_rows * c * 2].view(torch.bfloat16).view(record_rows, c)[:rows]
    return kdv, betas, projs


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rows", [1, 7, 16, 17, 64])
def test_the_wide_program_is_the_64_row_program_up_to_64_rows(state_dtype, rows):
    """The same decode structures: plain steps give the same outputs and state, speculative ones
    the same outputs and recorded rows (each program in its own record layout)."""
    narrow, wide = _kda(64, state_dtype), _kda(WIDE, state_dtype)
    for spec in (0, 1):
        x, state, conv = _sequence(rows, seed=50 + rows, state_dtype=state_dtype)
        s64, c64, r64 = state.clone(), conv.clone(), _record(64)
        s128, c128, r128 = state.clone(), conv.clone(), _record(WIDE)
        out64 = _step(narrow, x, s64, c64, r64, spec)
        out128 = _step(wide, x, s128, c128, r128, spec)
        assert torch.equal(out64, out128) and torch.equal(s64, s128) and torch.equal(c64, c128), spec
        if spec:
            for a, b in zip(_record_rows(r64, 64, rows), _record_rows(r128, WIDE, rows)):
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
def test_a_wide_verify_commits_as_a_plain_step_and_ignores_rejected_rows(state_dtype):
    """128 rows through the wide program: the full commit equals the plain verify's recurrent and
    conv state, and a commit at k rows is the same whatever the rows after k held."""
    wide, commit = _kda(WIDE, state_dtype), _commit(state_dtype)
    x, state, conv = _sequence(WIDE, seed=77, state_dtype=state_dtype)
    plain_state, plain_conv = state.clone(), conv.clone()
    _step(wide, x, plain_state, plain_conv, _record(WIDE), 0)

    def verify_and_commit(rows_in, keep):
        s, c, r = state.clone(), conv.clone(), _record(WIDE)
        out = _step(wide, rows_in, s, c, r, 1)
        assert torch.equal(s, state) and torch.equal(c, conv), "a speculative step changed the state"
        tables = torch.tensor([[0], [0], [keep]], dtype=torch.int32, device="cuda")
        commit.launch(s.unsqueeze(0), c.unsqueeze(0), r.view(1, -1), tables, scalars=(1, 1, 1))
        torch.cuda.synchronize()
        return out, s, c

    out, s, c = verify_and_commit(x, WIDE)
    assert torch.equal(s, plain_state) and torch.equal(c, plain_conv), "full commit against the plain verify"
    for keep in (1, 63, 64, 65, 100):
        changed = x.clone()
        changed[keep:] = torch.randn_like(changed[keep:].float()).bfloat16()
        out_a, s_a, c_a = verify_and_commit(x, keep)
        out_b, s_b, c_b = verify_and_commit(changed, keep)
        assert torch.equal(out_a[:keep], out_b[:keep]), f"kept outputs at {keep}"
        assert torch.equal(s_a, s_b) and torch.equal(c_a, c_b), f"committed state at {keep}"


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
def test_the_compact_wide_commit_rebuilds_tails_from_128_row_records(state_dtype):
    """``kda_commit_c_m128``: the KDA half as ``kda_commit_m128``; the tails of two DSA layers from
    their 128-row key | gate records (rows past 64 included), as the producer would leave them."""
    g = _g()
    gen = torch.Generator(device="cuda").manual_seed(99)
    d, slots = g.kda_width, 4
    # Sequences (slot, first step row, kept rows, rows held in its tail before the step).
    plan = [(0, 0, 70, 1), (1, 70, 3, 2), (2, 73, 50, 3), (3, 123, 5, 0)]
    tables = torch.tensor([[s for s, *_ in plan], [f for _, f, _, _ in plan], [k for _, _, k, _ in plan]],
                          dtype=torch.int32, device="cuda")
    state = torch.randn((1, slots, g.kda_heads, 128, 128), generator=gen, device="cuda")
    state = state.bfloat16() if state_dtype == "bfloat16" else state
    conv = torch.randn((1, slots, 3, 3 * d), generator=gen, device="cuda").bfloat16()
    replay = torch.rand((1, _layout(WIDE)[2] // 4), generator=gen, device="cuda")
    index_replay = torch.randn((2, WIDE, 256), generator=gen, device="cuda").bfloat16()
    tails = torch.zeros((2, slots, TAIL_BYTES), dtype=torch.uint8, device="cuda")
    old_rows = torch.randn((2, slots, 3, 256), generator=gen, device="cuda").bfloat16()
    for layer in range(2):
        for slot, _, _, held in plan:
            tails[layer, slot, :16].view(torch.int32)[0] = held
            rows = tails[layer, slot, 16:].view(torch.bfloat16).view(3, 256)
            rows[:held] = old_rows[layer, slot, :held]
    state_c, conv_c, tails_before = state.clone(), conv.clone(), tails.clone()
    _commit(state_dtype).launch(state, conv, replay, tables, scalars=(len(plan), 1, slots))
    _commit(state_dtype, compact=True).launch(state_c, conv_c, replay, tables, tails, index_replay,
                                              scalars=(len(plan), 1, slots, 2))
    torch.cuda.synchronize()
    word = torch.int16 if state_dtype == "bfloat16" else torch.int32
    assert torch.equal(state.view(word), state_c.view(word)) and torch.equal(conv.view(torch.int16),
                                                                             conv_c.view(torch.int16))
    for layer in range(2):
        for slot, first, keep, held in plan:
            total = held + keep
            count, opened = total % 4, total - total % 4
            expected = torch.zeros((3, 256), dtype=torch.bfloat16)
            before = tails_before[layer, slot, 16:].view(torch.bfloat16).view(3, 256).cpu()
            for e in range(3):
                if e < count:
                    offset = opened + e
                    expected[e] = index_replay[layer, first + offset - held].cpu() if offset >= held else before[e]
            tail = tails[layer, slot].cpu()
            assert tail[:16].view(torch.int32).tolist() == [count, 0, 0, 0], (layer, slot)
            assert torch.equal(tail[16:].view(torch.bfloat16).view(3, 256).view(torch.int16),
                               expected.view(torch.int16)), (layer, slot)
