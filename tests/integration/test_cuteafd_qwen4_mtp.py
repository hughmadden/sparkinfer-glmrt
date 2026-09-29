"""Speculation programs of Qwen 3.8 Flash Next (family ``qwen4``) on synthetic operands.

* ``qwen4_gdn`` with ``spec`` = 1, then ``qwen4_gdn_commit``: the verify rows'
  outputs equal a plain step's bit for bit, the state stays untouched until the
  commit, and the committed recurrent and conv state equal those of a plain
  step over the accepted rows (several sequences, different acceptances).
* ``qwen4_ple_*`` with ``spec`` = 1, then ``qwen4_ple_commit``: the same for the
  PLE conv state.
* ``qwen4_mtp_feedback`` against b12x's ``mtp_feedback`` reference.

  PYTHONPATH=. python3 tests/integration/test_cuteafd_qwen4_mtp.py
"""

from __future__ import annotations

from functools import lru_cache

import torch


def _bf16(*shape, scale=1.0, device="cuda"):
    return (torch.randn(*shape, device=device) * scale).to(torch.bfloat16)


@lru_cache(maxsize=None)
def _gdn(fp8: bool):
    from b12x.integration.cuteafd.qwen4_gdn import compile_qwen4_gdn_aot

    return compile_qwen4_gdn_aot(max_rows=64, fp8=fp8)


@lru_cache(maxsize=None)
def _commit():
    from b12x.integration.cuteafd.qwen4_gdn import compile_qwen4_gdn_commit_aot

    return compile_qwen4_gdn_commit_aot()


class GdnLayer:
    def __init__(self, g, seed: int):
        torch.manual_seed(seed)
        h, p, c, v, heads = g.hidden, g.gdn_in_width, g.gdn_conv_width, g.gdn_value_width, g.gdn_value_heads
        self.w_in = _bf16(p, h, scale=h ** -0.5)
        self.conv_w = torch.randn(c, 4, device="cuda") * 0.5
        self.a_log = torch.randn(heads, device="cuda") * 0.5
        self.dt_bias = torch.randn(heads, device="cuda") * 0.5
        self.norm_w = _bf16(128, scale=0.2) + 1
        self.w_out = _bf16(h, v, scale=v ** -0.5)
        self.fp8 = {}
        for name, w in (("w_in", self.w_in), ("w_out", self.w_out)):
            rows, cols = w.shape
            blocks = w.float().reshape(rows // 128 if rows % 128 == 0 else -1, 128, cols // 128, 128) \
                if rows % 128 == 0 else None
            if blocks is None:
                pad = (-rows) % 128
                wp = torch.nn.functional.pad(w.float(), (0, 0, 0, pad))
                blocks = wp.reshape(-1, 128, cols // 128, 128)
            amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
            scale = amax / 448.0
            q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).reshape(-1, cols)[:rows].contiguous()
            self.fp8[name] = (q, scale.contiguous().float())


def gdn_step(prog, layer, g, conv, state, x, slots, seq_first, replay, spec):
    rows = x.shape[0]
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(prog.scratch_bytes(64)["scratch"], dtype=torch.uint8, device="cuda")
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    names = [op.name for op in prog.operands]
    ops = {"x": x, "w_in": layer.w_in, "conv_w": layer.conv_w, "a_log": layer.a_log, "dt_bias": layer.dt_bias,
           "norm_w": layer.norm_w, "w_out": layer.w_out, "conv_state": conv, "state": state, "slots": i32(slots),
           "seq_first": i32(seq_first), "out": out, "replay": replay, "scratch": scratch}
    if "w_in_fp8" in names:
        ops.update(w_in_fp8=layer.fp8["w_in"][0], w_in_scale=layer.fp8["w_in"][1],
                   w_out_fp8=layer.fp8["w_out"][0], w_out_scale=layer.fp8["w_out"][1])
    prog.launch(*[ops[n] for n in names], scalars=[rows, spec])
    return out


def test_gdn_verify_by_replay(fp8: bool = True):
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    from b12x.integration.cuteafd.qwen4_gdn import gdn_replay_layout

    prog, commit = _gdn(fp8), _commit()
    layers, slots = 2, 4
    c, heads = g.gdn_conv_width, g.gdn_value_heads
    record = gdn_replay_layout(g.gdn_key_heads, heads, c)["bytes"]
    torch.manual_seed(0)
    state0 = torch.randn(layers, slots, heads, 128, 128, device="cuda") * 0.05
    conv0 = _bf16(layers, slots, 3, c)
    ref_state, ref_conv = state0.clone(), conv0.clone()
    spec_state, spec_conv = state0.clone(), conv0.clone()
    replay = torch.zeros(layers, record // 4, device="cuda")
    # Sequences in slots 2, 0, 3 verify 5, 3, 1 rows and keep 2, 3, 1 of them.
    rows_of, keep_of, slot_of = [5, 3, 1], [2, 3, 1], [2, 0, 3]
    xs = [_bf16(n, g.hidden) for n in rows_of]
    x = torch.cat(xs).contiguous()
    step_slots, seq_first, firsts = [], [], []
    for n, slot in zip(rows_of, slot_of):
        firsts.append(len(step_slots))
        seq_first += [len(step_slots)] * n
        step_slots += [slot] * n
    worst = 0.0
    for layer_id in range(layers):
        layer = GdnLayer(g, 100 + layer_id)
        spec_out = gdn_step(prog, layer, g, spec_conv[layer_id], spec_state[layer_id], x, step_slots, seq_first,
                            replay[layer_id], 1)
        assert torch.equal(spec_state, state0) and torch.equal(spec_conv, conv0), "spec step changed the state"
        # Plain steps: the full rows (outputs), then the kept rows (state).
        full_state, full_conv = state0[layer_id].clone(), conv0[layer_id].clone()
        plain_out = gdn_step(prog, layer, g, full_conv, full_state, x, step_slots, seq_first, None, 0)
        assert torch.equal(spec_out, plain_out), "spec outputs differ from a plain step"
        kept_x = torch.cat([xi[:k] for xi, k in zip(xs, keep_of)]).contiguous()
        k_slots, k_first = [], []
        for k, slot in zip(keep_of, slot_of):
            k_first += [len(k_slots)] * k
            k_slots += [slot] * k
        gdn_step(prog, layer, g, ref_conv[layer_id], ref_state[layer_id], kept_x, k_slots, k_first, None, 0)
        worst = max(worst, float((plain_out.float() - spec_out.float()).abs().max()))
    tables = torch.tensor([slot_of, firsts, keep_of], dtype=torch.int32, device="cuda")
    commit.launch(spec_state, spec_conv, replay, tables, scalars=[3, layers, slots])
    torch.cuda.synchronize()
    conv_diff = (spec_conv.float() - ref_conv.float()).abs().max().item()
    diff = (spec_state - ref_state).abs().max().item()
    if fp8:
        # FP8 decode projections (<= 16 rows) do not depend on the step's row count,
        # so the serial steps over the kept rows see the very same in-projection.
        assert torch.equal(spec_conv, ref_conv), "committed conv state differs from serial steps"
        assert torch.equal(spec_state, ref_state), f"committed state differs from serial steps (max {diff})"
        print(f"gdn verify-by-replay (fp8): outputs and committed state bit-identical")
    else:
        # BF16 projections may route 9 and 6 rows through different GEMMs.
        print(f"gdn verify-by-replay (bf16): outputs bit-identical; committed vs serial kept rows: "
              f"conv max |diff| {conv_diff:.3e}, state max |diff| {diff:.3e}")
        assert diff < 1e-2 and conv_diff < 0.1


def test_ple_verify_by_replay():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    from b12x.integration.cuteafd.qwen4 import compile_qwen4_ple_aot, compile_qwen4_ple_commit_aot

    prog, commit = compile_qwen4_ple_aot(g, fp8=False), compile_qwen4_ple_commit_aot(g)
    torch.manual_seed(1)
    c, h, slots = g.hc_width, g.hidden, 3
    table = _bf16(4096, g.ple_row_dim, scale=0.5)
    w_kv = _bf16(c + h, g.ple_dim, scale=g.ple_dim ** -0.5)
    norms = [_bf16(c, scale=0.1) for _ in range(3)]
    conv_w = torch.randn(c, g.ple_conv, device="cuda") * 0.5
    state0 = _bf16(slots, g.ple_state_rows, c)
    rows_of, keep_of, slot_of = [4, 2], [1, 2], [1, 2]
    n = sum(rows_of)
    streams0 = _bf16(n, g.hc_count, h)
    ids = torch.randint(0, 4096, (n, g.ple_rows), device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda")
    step_slots, seq_first, firsts = [], [], []
    for r, slot in zip(rows_of, slot_of):
        firsts.append(len(step_slots))
        seq_first += [len(step_slots)] * r
        step_slots += [slot] * r
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    scratch = torch.empty(prog.scratch_bytes(64)["scratch"], dtype=torch.uint8, device="cuda")

    def run(streams, state, sel, sl, sf, replay, spec):
        prog.launch(streams, ids[sel].contiguous(), table, scale, w_kv, *norms, conv_w, state, i32(sl), i32(sf),
                    replay, scratch, scalars=[streams.shape[0], spec])

    replay = torch.zeros(64, c, dtype=torch.bfloat16, device="cuda")
    spec_state, spec_streams = state0.clone(), streams0.clone()
    run(spec_streams, spec_state, torch.arange(n, device="cuda"), step_slots, seq_first, replay, 1)
    assert torch.equal(spec_state, state0)
    plain_state, plain_streams = state0.clone(), streams0.clone()
    run(plain_streams, plain_state, torch.arange(n, device="cuda"), step_slots, seq_first, None, 0)
    assert torch.equal(spec_streams, plain_streams)
    sel, k_slots, k_first = [], [], []
    for first, k, slot in zip(firsts, keep_of, slot_of):
        sel += list(range(first, first + k))
        k_first += [len(k_slots)] * k
        k_slots += [slot] * k
    sel = torch.tensor(sel, device="cuda")
    ref_state = state0.clone()
    run(streams0[sel].clone(), ref_state, sel, k_slots, k_first, None, 0)
    commit.launch(spec_state, replay, i32([slot_of, firsts, keep_of]), scalars=[len(rows_of)])
    torch.cuda.synchronize()
    assert torch.equal(spec_state, ref_state), "committed PLE conv state differs from serial steps"
    print("ple verify-by-replay: outputs and committed conv state bit-identical")


def test_mtp_feedback():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    from b12x.integration.cuteafd.qwen4 import compile_qwen4_mtp_feedback_aot
    from b12x.sequence.mtp_feedback.reference import feedback

    prog = compile_qwen4_mtp_feedback_aot(g)
    torch.manual_seed(2)
    h, n = g.hidden, g.hc_count
    for rows in (1, 5, 64, 300):
        source = _bf16(rows + 3, n, h, scale=3.0)
        pick = torch.randperm(rows + 3, device="cuda")[:rows].to(torch.int32)
        embed = _bf16(rows, h, scale=0.02)
        nh, ne = _bf16(n * h, scale=0.3), _bf16(h, scale=0.3)
        fh, fe = _bf16(h, h, scale=h ** -0.5), _bf16(h, h, scale=h ** -0.5)
        streams = torch.empty(rows, n, h, dtype=torch.bfloat16, device="cuda")
        delta = torch.empty(rows, h, dtype=torch.bfloat16, device="cuda")
        inject = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
        prog.launch(source, pick, embed, nh, ne, fh, fe, streams, delta, inject, scratch, scalars=[rows])
        ours = (streams.float() + delta.float()[:, None]).to(torch.bfloat16)
        theirs = feedback(embed, source[pick.long()].contiguous(), ne, nh, fe, fh)
        diff = (ours.float() - theirs.float()).abs()
        cos = torch.nn.functional.cosine_similarity(ours.float().flatten(), theirs.float().flatten(), dim=0).item()
        assert torch.all(inject == 1)
        assert cos > 0.99999, cos
        print(f"mtp feedback rows {rows}: cosine {cos:.7f}, max |diff| {diff.max().item():.3e}, "
              f"exact {100 * (diff == 0).float().mean().item():.2f}%")


if __name__ == "__main__":
    test_mtp_feedback()
    test_gdn_verify_by_replay(True)
    test_gdn_verify_by_replay(False)
    test_ple_verify_by_replay()
