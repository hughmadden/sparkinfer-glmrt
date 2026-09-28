"""cuteafd DSV4 compressor AOT programs vs b12x.attention.dsv4_compressor.

Two comparisons per scenario:

* port exactness: the Triton pooling/finalize kernels are run on the very
  projection the AOT program computed (read back from its scratch), so the
  caches and states isolate the CuTe ports;
* end to end: the full prepared path (torch.mm projection) versus the AOT
  program, within BF16-projection tolerance.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import check_export, checkpoint_tensor, cosine, has_checkpoint

EPS = 1.0e-6
C4_PAGE_BYTES = 37_440
C128_PAGE_BYTES = 1_728
INDEX_PAGE_BYTES = 8_448


def _weights(ratio: int, device, hidden: int = 4096):
    from b12x.attention import dsv4_compressor

    layer = 2 if ratio == 4 else 3
    if hidden == 4096 and has_checkpoint():
        g = lambda n: checkpoint_tensor(f"layers.{layer}.attn.{n}", device)  # noqa: E731
        kw = {}
        if ratio == 4:
            kw = dict(index_wkv=g("indexer.compressor.wkv.weight").contiguous(),
                      index_wgate=g("indexer.compressor.wgate.weight").contiguous(),
                      index_ape=g("indexer.compressor.ape").float().contiguous(),
                      index_norm=g("indexer.compressor.norm.weight").contiguous())
        return dsv4_compressor.pack_weights(
            g("compressor.wkv.weight").contiguous(), g("compressor.wgate.weight").contiguous(),
            g("compressor.ape").float().contiguous(), g("compressor.norm.weight").contiguous(), **kw)
    gen = torch.Generator(device="cpu").manual_seed(ratio)
    pw = 1024 if ratio == 4 else 512
    rnd = lambda *s, scale=0.02: (torch.randn(s, generator=gen) * scale).to(device)  # noqa: E731
    kw = {}
    if ratio == 4:
        kw = dict(index_wkv=rnd(256, hidden).bfloat16(), index_wgate=rnd(256, hidden).bfloat16(),
                  index_ape=rnd(4, 256, scale=0.5), index_norm=(1 + rnd(128, scale=0.1)).bfloat16())
    return dsv4_compressor.pack_weights(rnd(pw, hidden).bfloat16(), rnd(pw, hidden).bfloat16(),
                                        rnd(ratio, pw, scale=0.5), (1 + rnd(512, scale=0.1)).bfloat16(), **kw)


def _cos_sin(positions: int, device) -> torch.Tensor:
    inv = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous().to(device)


class Pools:
    """Compressed cache, index cache and states; one slot page parked high."""

    def __init__(self, ratio, sequences, device, *, high_page):
        self.ratio = ratio
        page_bytes = C4_PAGE_BYTES if ratio == 4 else C128_PAGE_BYTES
        self.pages = high_page + 256
        self.cache = torch.zeros((self.pages, page_bytes), dtype=torch.uint8, device=device)
        self.index_cache = (torch.zeros((self.pages, INDEX_PAGE_BYTES), dtype=torch.uint8, device=device)
                            if ratio == 4 else None)
        pw, rows = (1024, 16) if ratio == 4 else (512, 256)
        self.kv = torch.zeros((sequences, rows, pw), device=device)
        self.score = torch.zeros_like(self.kv)
        self.ikv = torch.zeros((sequences, 16, 256), device=device) if ratio == 4 else None
        self.iscore = torch.zeros_like(self.ikv) if ratio == 4 else None

    def clone(self):
        other = Pools.__new__(Pools)
        for name, value in vars(self).items():
            setattr(other, name, value.clone() if isinstance(value, torch.Tensor) else value)
        return other

    def kwargs(self):
        kw = dict(compressed_main_cache=self.cache, main_kv_state=self.kv, main_score_state=self.score)
        if self.ratio == 4:
            kw.update(index_cache=self.index_cache, index_kv_state=self.ikv, index_score_state=self.iscore)
        return kw

    def pointers(self, weights):
        ptrs = [weights.main_ape, weights.main_norm, self.cache, self.kv, self.score]
        if self.ratio == 4:
            ptrs += [weights.index_ape, weights.index_norm, self.index_cache, self.ikv, self.iscore]
        return ptrs


def _slot(ratio, j, high_page):
    """Distinct physical slot for group key ``j``: a few low slots, the rest
    on pages past the 2^31-byte line."""
    rows = 64 if ratio == 4 else 2
    return j if j < 3 else high_page * rows + j


def _prefill_meta(ratio, lengths, device, high_page, *, group_capacity=None):
    i32 = dict(dtype=torch.int32, device=device)
    starts, ropes, slots, offsets = [], [], [], [0]
    for s, n in enumerate(lengths):
        for j in range(n // ratio):
            starts.append(offsets[-1] + j * ratio)
            ropes.append(j * ratio)
            slots.append(_slot(ratio, 100 * s + j, high_page))
        offsets.append(offsets[-1] + n)
    groups = len(starts)
    cap = max(group_capacity or groups, 1)
    pad = cap - groups
    return dict(
        active_groups=torch.tensor([groups], **i32),
        group_source_starts=torch.tensor(starts + [0] * pad, **i32),
        group_rope_positions=torch.tensor(ropes + [0] * pad, **i32),
        compressed_slots=torch.tensor(slots + [0] * pad, **i32),
        active_sequences=torch.tensor([len(lengths)], **i32),
        sequence_offsets=torch.tensor(offsets, **i32),
        state_sequence_ids=torch.arange(len(lengths), **i32),
    )


def _continuation_meta(ratio, starts_lengths, device, high_page):
    i32 = dict(dtype=torch.int32, device=device)
    seq_slots, positions, slots, offsets = [], [], [], [0]
    for s, (start, n) in enumerate(starts_lengths):
        for p in range(start, start + n):
            if p % ratio == ratio - 1:
                seq_slots.append(s)
                positions.append(p + 1 - ratio)
                slots.append(_slot(ratio, 100 * s + p // ratio, high_page))
        offsets.append(offsets[-1] + n)
    groups = len(positions)
    cap = max(groups, 1)
    pad = cap - groups
    return dict(
        active_groups=torch.tensor([groups], **i32),
        group_sequence_slots=torch.tensor(seq_slots + [0] * pad, **i32),
        group_source_positions=torch.tensor(positions + [0] * pad, **i32),
        group_rope_positions=torch.tensor(positions + [0] * pad, **i32),
        compressed_slots=torch.tensor(slots + [0] * pad, **i32),
        active_sequences=torch.tensor([len(starts_lengths)], **i32),
        sequence_offsets=torch.tensor(offsets, **i32),
        sequence_start_positions=torch.tensor([s for s, _ in starts_lengths], **i32),
        state_sequence_ids=torch.arange(len(starts_lengths), **i32),
    )


_PREFILL_KEYS = ("active_groups", "group_source_starts", "group_rope_positions", "compressed_slots",
                 "active_sequences", "sequence_offsets", "state_sequence_ids")
_CONT_KEYS = ("active_groups", "group_sequence_slots", "group_source_positions", "group_rope_positions",
              "compressed_slots", "active_sequences", "sequence_offsets", "sequence_start_positions",
              "state_sequence_ids")


def _plan(ratio, max_tokens, device, hidden=4096):
    from b12x.attention import dsv4_compressor

    return dsv4_compressor.plan(dsv4_compressor.Caps(
        device=device, max_tokens=max_tokens, hidden=hidden, compress_ratio=ratio,
        with_indexer=ratio == 4, cache_format="fp8"))


def _scratch(plan):
    spec = plan.scratch_specs()[0]
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _triton_prefill(binding, projection=None):
    """run_prefill, optionally with a given projection instead of torch.mm."""
    from b12x.attention.dsv4_compressor import _impl

    if projection is None:
        _impl.run_dsv4_compressor_prefill(binding=binding)
        return
    binding = replace(binding, projection=projection)
    _impl._run_prefill_main(binding)
    _impl._finalize_prefill_state(binding, kv_state=binding.main_kv_state, score_state=binding.main_score_state,
                                  ape=binding.weights.main_ape, projection_offset=0,
                                  projected_width=1024 if binding.compress_ratio == 4 else 512)
    if binding.compress_ratio == 4:
        _impl._run_prefill_index(binding)
        _impl._finalize_prefill_state(binding, kv_state=binding.index_kv_state,
                                      score_state=binding.index_score_state, ape=binding.weights.index_ape,
                                      projection_offset=2048, projected_width=256)


def _triton_continuation(binding, projection=None):
    from b12x.attention.dsv4_compressor import _impl

    if projection is None:
        _impl.run_dsv4_compressor_continuation(binding=binding)
        return
    binding = replace(binding, projection=projection)
    _impl._run_continuation_main(binding)
    if binding.compress_ratio == 4:
        _impl._run_continuation_index(binding)
    _impl._finalize_continuation_state(binding, kv_state=binding.main_kv_state,
                                       score_state=binding.main_score_state, ape=binding.weights.main_ape,
                                       projection_offset=0,
                                       projected_width=1024 if binding.compress_ratio == 4 else 512)
    if binding.compress_ratio == 4:
        _impl._finalize_continuation_state(binding, kv_state=binding.index_kv_state,
                                           score_state=binding.index_score_state,
                                           ape=binding.weights.index_ape, projection_offset=2048,
                                           projected_width=256)


def _triton_decode(binding, projection=None):
    from b12x.attention.dsv4_compressor import _impl

    if projection is None:
        _impl.run_dsv4_compressor_decode(binding=binding)
        return
    binding = replace(binding, projection=projection)
    _impl._run_main(binding)
    if binding.compress_ratio == 4:
        _impl._run_index(binding)


def _aot_projection(program, scratch, rows):
    width = program.geometry["joint_width"]
    return scratch[: rows * width * 2].view(torch.bfloat16).view(rows, width)


def _written_rows(pools, ratio, slots):
    """Main (payload+scale) and index (values+scale) bytes of each slot."""
    rows_per_page = 64 if ratio == 4 else 2
    main, index = [], []
    for slot in slots:
        page, row = divmod(int(slot), rows_per_page)
        base = rows_per_page * 576
        main.append(torch.cat((pools.cache[page, row * 576:(row + 1) * 576],
                               pools.cache[page, base + row * 8: base + row * 8 + 8])))
        if ratio == 4:
            index.append(torch.cat((pools.index_cache[page, row * 128:(row + 1) * 128],
                                    pools.index_cache[page, 8192 + 4 * row: 8192 + 4 * row + 4])))
    return torch.stack(main), (torch.stack(index) if index else None)


def _compare_pools(actual, expected, ratio, slots, *, exact):
    main_a, index_a = _written_rows(actual, ratio, slots)
    main_e, index_e = _written_rows(expected, ratio, slots)
    frac = lambda a, e: float((a != e).float().mean())  # noqa: E731
    if exact:
        # Same projection in: the CuTe ports reproduce the Triton bytes.
        assert frac(main_a, main_e) == 0.0, f"main cache mismatch {frac(main_a, main_e)}"
        if ratio == 4:
            assert frac(index_a, index_e) == 0.0, f"index cache mismatch {frac(index_a, index_e)}"
    else:
        # cuBLAS vs CuTe BF16 projection: rare one-ulp inputs flip a few codes.
        assert frac(main_a, main_e) <= 0.01, f"main cache mismatch {frac(main_a, main_e)}"
        rope_a = main_a[:, 448:576].contiguous().view(torch.bfloat16).float()
        rope_e = main_e[:, 448:576].contiguous().view(torch.bfloat16).float()
        assert cosine(rope_a, rope_e) > 0.999
        if ratio == 4:
            assert frac(index_a, index_e) <= 0.02, f"index cache mismatch {frac(index_a, index_e)}"
    for name in ("kv", "score", "ikv", "iscore"):
        a, e = getattr(actual, name), getattr(expected, name)
        if a is None:
            continue
        if exact:
            assert torch.equal(a, e), name
        else:
            finite = torch.isfinite(e)
            assert torch.equal(finite, torch.isfinite(a)), name
            torch.testing.assert_close(a[finite], e[finite], rtol=1e-2, atol=1e-2)


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd import dsv4_compressor as c

    with exportable_compilation():
        out = {}
        for ratio in (4, 128):
            out[("decode", ratio)] = c.compile_dsv4_compressor_decode_aot(FLASH, ratio=ratio)
            out[("prefill", ratio)] = c.compile_dsv4_compressor_prefill_aot(FLASH, ratio=ratio)
            out[("continuation", ratio)] = c.compile_dsv4_compressor_continuation_aot(FLASH, ratio=ratio)
            out[("pro_prefill", ratio)] = c.compile_dsv4_compressor_prefill_aot(PRO, ratio=ratio)
        return out


HIGH = {4: (2**31) // C4_PAGE_BYTES + 2, 128: (2**31) // C128_PAGE_BYTES + 2}


def _hidden(rows, seed, device, hidden=4096):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((rows, hidden), generator=gen).bfloat16().to(device)


def _run_prefill_both(programs, ratio, lengths, weights, cos_sin, device, *, pools=None, hidden_size=4096):
    """Returns (aot_pools, triton_on_aot_projection_pools, full_prepared_pools, aot_projection)."""
    rows = sum(lengths)
    meta = _prefill_meta(ratio, lengths, device, HIGH[ratio])
    base = pools if pools is not None else Pools(ratio, len(lengths), device, high_page=HIGH[ratio])
    hidden = _hidden(rows, rows + ratio, device, hidden_size)
    program = programs[("prefill", ratio) if hidden_size == 4096 else ("pro_prefill", ratio)]
    aot = base.clone()
    scratch = torch.empty((program.scratch_bytes(rows)["scratch"],), dtype=torch.uint8, device=device)
    program.launch(hidden, *[meta[k] for k in _PREFILL_KEYS], cos_sin, weights.joint_projection,
                   *aot.pointers(weights), scratch,
                   scalars=(rows, int(meta["group_source_starts"].shape[0]), len(lengths)))
    projection = _aot_projection(program, scratch, rows)
    plan = _plan(ratio, rows, device, hidden_size)
    results = []
    for use_projection in (True, False):
        ref = base.clone()
        binding = plan.bind_prefill(scratch=_scratch(plan), hidden_states=hidden,
                                    compressed_cos_sin_cache=cos_sin, weights=weights, eps=EPS,
                                    initial_prefill=True, **meta, **ref.kwargs())
        _triton_prefill(binding, projection if use_projection else None)
        results.append(ref)
    torch.cuda.synchronize()
    slots = meta["compressed_slots"][: int(meta["active_groups"])].tolist()
    return aot, results[0], results[1], projection, slots


@pytest.mark.parametrize(("ratio", "lengths"), [
    (4, [37]), (4, [20, 13, 4]), (4, [1000]), (128, [300]), (128, [130, 256]), (128, [2048])])
def test_prefill_matches_triton(programs, ratio, lengths):
    device = require_b12x()
    weights = _weights(ratio, device)
    cos_sin = _cos_sin(4096, device)
    aot, exact_ref, full_ref, projection, slots = _run_prefill_both(programs, ratio, lengths, weights,
                                                                    cos_sin, device)
    _compare_pools(aot, exact_ref, ratio, slots, exact=True)
    _compare_pools(aot, full_ref, ratio, slots, exact=False)


@pytest.mark.parametrize(("ratio", "lengths"), [(4, [45, 7]), (128, [260])])
def test_pro_prefill_matches_triton(programs, ratio, lengths):
    device = require_b12x()
    weights = _weights(ratio, device, 7168)
    cos_sin = _cos_sin(4096, device)
    aot, exact_ref, full_ref, _, slots = _run_prefill_both(programs, ratio, lengths, weights, cos_sin,
                                                           device, hidden_size=7168)
    _compare_pools(aot, exact_ref, ratio, slots, exact=True)
    _compare_pools(aot, full_ref, ratio, slots, exact=False)


def test_continuation_and_decode_match_triton(programs):
    """C4 and C128: prefill, then an ordered continuation chunk, then decode rows."""
    device = require_b12x()
    cos_sin = _cos_sin(4096, device)
    for ratio, prompt, chunk, decode_steps in ((4, [22, 9], [11, 6], 5), (128, [200], [70], 130)):
        weights = _weights(ratio, device)
        aot, exact_ref, _, _, _ = _run_prefill_both(programs, ratio, prompt, weights, cos_sin, device)
        # Continue both sequences from where the prompt ended (states agree bitwise).
        starts = [(p, n) for p, n in zip(prompt, chunk)]
        meta = _continuation_meta(ratio, starts, device, HIGH[ratio])
        rows = sum(chunk)
        hidden = _hidden(rows, 77 + ratio, device)
        program = programs[("continuation", ratio)]
        scratch = torch.empty((program.scratch_bytes(rows)["scratch"],), dtype=torch.uint8, device=device)
        program.launch(hidden, *[meta[k] for k in _CONT_KEYS], cos_sin, weights.joint_projection,
                       *aot.pointers(weights), scratch,
                       scalars=(rows, int(meta["group_sequence_slots"].shape[0]), len(starts)))
        projection = _aot_projection(program, scratch, rows)
        plan = _plan(ratio, rows, device)
        binding = plan.bind_continuation(scratch=_scratch(plan), hidden_states=hidden,
                                         compressed_cos_sin_cache=cos_sin, weights=weights, eps=EPS,
                                         ordered_continuation=True, **meta, **exact_ref.kwargs())
        _triton_continuation(binding, projection)
        torch.cuda.synchronize()
        slots = meta["compressed_slots"][: int(meta["active_groups"])].tolist()
        _compare_pools(aot, exact_ref, ratio, slots, exact=True)

        # Decode: one row per sequence per step; crosses at least one group boundary.
        program = programs[("decode", ratio)]
        position = [p + n for p, n in starts]
        decode_plan = _plan(ratio, len(position), device)
        emitted = []
        for step in range(decode_steps):
            rows = len(position)
            i32 = dict(dtype=torch.int32, device=device)
            pos = torch.tensor(position, **i32)
            seq = torch.arange(rows, **i32)
            slots = torch.tensor([_slot(ratio, 100 * s + p // ratio, HIGH[ratio]) for s, p in enumerate(position)], **i32)
            hidden = _hidden(rows, 1000 + step, device)
            scratch = torch.empty((program.scratch_bytes(rows)["scratch"],), dtype=torch.uint8, device=device)
            program.launch(hidden, pos, seq, slots, cos_sin, weights.joint_projection,
                           *aot.pointers(weights), scratch, scalars=(rows,))
            projection = _aot_projection(program, scratch, rows)
            binding = decode_plan.bind_decode(
                scratch=_scratch(decode_plan), hidden_states=hidden, positions=pos, sequence_ids=seq,
                compressed_slots=slots, compressed_cos_sin_cache=cos_sin, weights=weights, eps=EPS,
                rows_are_sequence_unique=True, **exact_ref.kwargs())
            _triton_decode(binding, projection)
            emitted += [int(s) for s, p in zip(slots.tolist(), position) if p % ratio == ratio - 1]
            position = [p + 1 for p in position]
        torch.cuda.synchronize()
        assert emitted, "decode steps must complete at least one group"
        _compare_pools(aot, exact_ref, ratio, emitted, exact=True)


def test_prefill_replays_in_cuda_graph(programs):
    device = require_b12x()
    ratio, lengths = 4, [21]
    weights = _weights(ratio, device)
    cos_sin = _cos_sin(4096, device)
    meta = _prefill_meta(ratio, lengths, device, 3)
    pools = Pools(ratio, 1, device, high_page=3)
    hidden = _hidden(21, 5, device)
    program = programs[("prefill", ratio)]
    scratch = torch.empty((program.scratch_bytes(21)["scratch"],), dtype=torch.uint8, device=device)
    args = (hidden, *[meta[k] for k in _PREFILL_KEYS], cos_sin, weights.joint_projection,
            *pools.pointers(weights), scratch)
    scalars = (21, int(meta["group_source_starts"].shape[0]), 1)
    program.launch(*args, scalars=scalars)
    torch.cuda.synchronize()
    eager = pools.clone()
    for t in (pools.cache, pools.index_cache, pools.kv, pools.score, pools.ikv, pools.iscore):
        t.zero_()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        program.launch(*args, scalars=scalars)
    graph.replay()
    torch.cuda.synchronize()
    for name in ("cache", "index_cache", "kv", "score", "ikv", "iscore"):
        assert torch.equal(getattr(pools, name), getattr(eager, name)), name


@pytest.mark.parametrize("key", [("decode", 4), ("prefill", 128), ("continuation", 4)])
def test_export_to_c_signature(programs, key, tmp_path):
    info = check_export(programs[key], tmp_path, f"dsv4_compressor_{key[0]}_c{key[1]}")
    assert info["argument_count"] == len(programs[key].operands) + len(programs[key].scalars) + 1


def test_decode_completing_first_group_matches_prefill(programs):
    """A C4 decode step that completes group 0 has no previous group.

    The AOT decode reads the -inf rolling-state rows left by the prefill
    finalizer (floor modulo), so it must agree bitwise with a prefill of the
    same four tokens. (The Triton decode's C remainder addresses rows -4..-1,
    i.e. another sequence's state, for this case.)
    """
    device = require_b12x()
    ratio = 4
    weights = _weights(ratio, device)
    cos_sin = _cos_sin(64, device)
    hidden = _hidden(4, 99, device)
    i32 = dict(dtype=torch.int32, device=device)

    # Reference: prefill all four tokens into slot 0 (sequence state 1).
    ref = Pools(ratio, 2, device, high_page=0)
    meta = _prefill_meta(ratio, [4], device, 0)
    meta["state_sequence_ids"] = torch.tensor([1], **i32)
    program = programs[("prefill", ratio)]
    scratch = torch.empty((program.scratch_bytes(4)["scratch"],), dtype=torch.uint8, device=device)
    program.launch(hidden, *[meta[k] for k in _PREFILL_KEYS], cos_sin, weights.joint_projection,
                   *ref.pointers(weights), scratch, scalars=(4, 1, 1))

    # AOT: prefill two tokens (no group), then decode positions 2 and 3.
    aot = Pools(ratio, 2, device, high_page=0)
    aot.score.fill_(123.0)  # poison: previous-group rows must come from the finalizer
    meta = _prefill_meta(ratio, [2], device, 0)
    meta["state_sequence_ids"] = torch.tensor([1], **i32)
    program.launch(hidden[:2].contiguous(), *[meta[k] for k in _PREFILL_KEYS], cos_sin,
                   weights.joint_projection, *aot.pointers(weights), scratch, scalars=(2, 1, 1))
    decode = programs[("decode", ratio)]
    for position in (2, 3):
        decode.launch(hidden[position:position + 1].contiguous(), torch.tensor([position], **i32),
                      torch.tensor([1], **i32), torch.tensor([0], **i32), cos_sin,
                      weights.joint_projection, *aot.pointers(weights), scratch, scalars=(1,))
    torch.cuda.synchronize()
    main_a, index_a = _written_rows(aot, ratio, [0])
    main_e, index_e = _written_rows(ref, ratio, [0])
    assert torch.equal(main_a, main_e)
    assert torch.equal(index_a, index_e)
