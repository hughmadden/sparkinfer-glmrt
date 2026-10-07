"""GLM 5.3 Flash compact DSA index cache (``glmf_index_producer_c_m*``, ``glmf_kda_commit_c``)
against the per-token-key producer (``glmf_index_producer_m*``) it replaces.

Sequences run through both producers side by side over shared paged caches: prefill chunks
and decode steps ending at every position mod 4, decode steps of several sequences at once,
and speculative steps committed at every kept count. After every step the pooled-key cache,
the index query and the head weights are bitwise equal, and every sequence's tail holds
exactly the key | gate rows of its open pool (the per-token keys at those positions) with a
zero past its count. The KDA half of ``glmf_kda_commit_c`` equals ``glmf_kda_commit``
bitwise, over an FP32 state and over the BF16 state of ``--kda-state bf16`` (both programs'
``state_dtype="bfloat16"`` variants). With ``GLMF_SNAPSHOT`` (a GLM-5.3-Flash checkpoint directory) the same runs on
layer 3's indexer weights with real embedding rows through its input norm.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

TAIL_BYTES = 1552
PAGE = 64
KPOOL = 4
UNIT = KPOOL * PAGE
_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def P(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import GLM53_FLASH, glmf

        fn = {"keys": glmf.compile_glmf_index_producer_aot, "compact": glmf.compile_glmf_index_producer_c_aot,
              "commit": glmf.compile_glmf_kda_commit_aot, "commit_c": glmf.compile_glmf_kda_commit_c_aot}[kind]
        _PROGRAMS[key] = fn(GLM53_FLASH, **kw)
    return _PROGRAMS[key]


def _random_weights(g, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    i, q, h = g.index_heads, g.q_lora_rank, g.hidden
    rnd = lambda *shape, s=1.0: torch.randn(shape, generator=gen, device="cuda") * s  # noqa: E731
    return {"w_iq": rnd(i * 128, q, s=q ** -0.5).bfloat16(), "w_ik": rnd(256 + i, h, s=h ** -0.5).bfloat16(),
            "k_norm_w": (1.0 + rnd(128, s=0.1)).bfloat16(), "k_norm_b": rnd(128, s=0.05).bfloat16(),
            "ape": rnd(KPOOL, 128, s=0.5).bfloat16()}


class _Checkpoint:
    """Tensors of a safetensors checkpoint directory, read on demand."""

    def __init__(self, root: Path):
        from safetensors import safe_open

        self.root, self.open = root, safe_open
        self.where = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with self.open(str(self.root / self.where[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        with self.open(str(self.root / self.where[name]), framework="pt", device="cpu") as f:
            table = f.get_slice(name)
            return torch.stack([table[i:i + 1][0] for i in ids])


def _real_inputs(g, root: Path, layer: int = 3):
    """Layer ``layer``'s indexer weights, and a source of its inputs: embedding rows of random
    ids through the layer's input RMSNorm (proxies for the layer's real activations)."""
    ckpt = _Checkpoint(root)
    p = f"model.language_model.layers.{layer}."
    a = lambda name: ckpt.get(p + "self_attn.indexer." + name).cuda()  # noqa: E731
    weights = {"w_iq": a("wq_b.weight").bfloat16(),
               "w_ik": torch.cat([a("wk.weight"), a("weights_proj.weight"), a("index_kpool_compress_gate")]).bfloat16(),
               "k_norm_w": a("k_norm.weight").bfloat16(), "k_norm_b": a("k_norm.bias").bfloat16(),
               "ape": a("index_kpool_compress_ape").bfloat16().reshape(KPOOL, 128)}
    norm = ckpt.get(p + "input_layernorm.weight").cuda().float()
    rng = random.Random(7)

    def x_rows(rows: int) -> torch.Tensor:
        ids = [rng.randrange(150000) for _ in range(rows)]
        e = ckpt.rows("model.language_model.embed_tokens.weight", ids).cuda().float()
        return (e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + g.norm_eps) * norm).bfloat16()

    return weights, x_rows


class World:
    """Sequences over shared paged caches (allocation units of four 64-row MLA pages and one
    pool page, as the engine's), run through the per-token-key producer and the compact one."""

    def __init__(self, g, weights, *, slots=16, units=48, seed=0, x_rows=None):
        self.g, self.w, self.slots = g, weights, slots
        self.gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
        self.x_rows = x_rows or (lambda rows: torch.randn((rows, g.hidden), generator=self.gen,
                                                          device="cuda").bfloat16())
        self.token_keys = torch.zeros((units * UNIT, 256), dtype=torch.bfloat16, device="cuda")
        self.cache_keys = torch.zeros((units, PAGE * 132), dtype=torch.uint8, device="cuda")
        self.cache_compact = torch.zeros_like(self.cache_keys)
        self.tails = torch.zeros((slots, TAIL_BYTES), dtype=torch.uint8, device="cuda")
        self.replay = torch.zeros((64, 256), dtype=torch.bfloat16, device="cuda")
        rng = random.Random(seed)
        self.free_units = list(range(units))
        rng.shuffle(self.free_units)
        self.free_slots = list(range(slots))
        rng.shuffle(self.free_slots)
        self.rng = rng
        self.seqs = []

    def admit(self, capacity):
        seq = {"units": [self.free_units.pop() for _ in range(-(-capacity // UNIT))], "slot": self.free_slots.pop(),
               "len": 0}
        self.seqs.append(seq)
        return seq

    @staticmethod
    def record(seq, p):
        return (seq["units"][p // UNIT] * KPOOL + (p % UNIT) // PAGE) * PAGE + p % PAGE

    @staticmethod
    def pool_slot(seq, p):
        return -1 if p % KPOOL != KPOOL - 1 else seq["units"][p // UNIT] * PAGE + (p // KPOOL) % PAGE

    def step(self, parts, cap, spec=False):
        """One step over ``parts`` [(sequence, rows)] through both producers (``cap`` 64: the decode
        programs, 4096: prefill); sequences advance unless ``spec``."""
        g, w = self.g, self.w
        positions, records, pools, kda, first = [], [], [], [], []
        for seq, n in parts:
            f = len(positions)
            for p in range(seq["len"], seq["len"] + n):
                positions.append(p)
                records.append(self.record(seq, p))
                pools.append(self.pool_slot(seq, p))
                kda.append(seq["slot"])
                first.append(f)
        rows = len(positions)
        assert rows <= cap and (not spec or rows <= 64)
        t = lambda v, dtype: torch.tensor(v, dtype=dtype, device="cuda")  # noqa: E731
        x = self.x_rows(rows)
        q_resid = torch.randn((rows, g.q_lora_rank), generator=self.gen, device="cuda").bfloat16()
        keys, compact = P("keys", max_rows=cap), P("compact", max_rows=cap)
        outs = []
        for program in (keys, compact):
            scratch = torch.randint(0, 256, (program.scratch_bytes(cap)["scratch"],), dtype=torch.uint8,
                                    generator=self.gen, device="cuda")
            q_fp8 = torch.empty((rows, g.index_heads, 128), dtype=torch.uint8, device="cuda")
            head_weights = torch.empty((rows, g.index_heads), dtype=torch.float32, device="cuda")
            outs.append((scratch, q_fp8, head_weights))
        # The replay record holds garbage outside what a step writes.
        self.replay.view(torch.int16).random_(-32768, 32767, generator=self.gen)
        (s0, q0, h0), (s1, q1, h1) = outs
        tails_before = self.tails.clone()
        keys.launch(x, q_resid, t(records, torch.int64), t(pools, torch.int64), w["w_iq"], w["w_ik"], w["k_norm_w"],
                    w["k_norm_b"], w["ape"], self.token_keys, self.cache_keys, q0.view(torch.float8_e4m3fn), h0, s0,
                    scalars=(rows,))
        compact.launch(x, q_resid, t(pools, torch.int64), t(positions, torch.int64), t(kda, torch.int32),
                       t(first, torch.int32), w["w_iq"], w["w_ik"], w["k_norm_w"], w["k_norm_b"], w["ape"],
                       self.tails, self.replay, self.cache_compact, q1.view(torch.float8_e4m3fn), h1, s1,
                       scalars=(rows, int(spec)))
        torch.cuda.synchronize()
        assert torch.equal(q0, q1)
        assert torch.equal(h0.view(torch.int32), h1.view(torch.int32))
        assert torch.equal(self.cache_keys, self.cache_compact), "pooled keys differ"
        if spec:
            assert torch.equal(self.tails, tails_before), "a speculative step changed the tails"
            # Every row's key | gate row is in the replay record, as the per-token keys have it.
            assert torch.equal(self.replay[:rows].view(torch.int16),
                               self.token_keys[t(records, torch.int64)].view(torch.int16))
        else:
            for seq, n in parts:
                seq["len"] += n
            self.check_tails([seq for seq, _ in parts])

    def commit(self, parts, keeps, state_dtype="float32"):
        """``glmf_kda_commit_c`` after a speculative step over ``parts``: sequence i keeps
        ``keeps[i]`` rows. Its KDA half against ``glmf_kda_commit`` on the same random state,
        FP32 or (``state_dtype="bfloat16"``) BF16, both programs built for that state."""
        from b12x.integration.cuteafd import GLM53_FLASH as g
        from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout

        slots, firsts, f = [], [], 0
        for seq, n in parts:
            slots.append(seq["slot"])
            firsts.append(f)
            f += n
        tables = torch.tensor([slots, firsts, list(keeps)], dtype=torch.int32, device="cuda")
        d = g.kda_width
        record = kda_replay_layout(g.kda_heads, 3 * d)[2]
        state = torch.randn((1, self.slots, g.kda_heads, 128, 128), generator=self.gen, device="cuda")
        bf16 = state_dtype == "bfloat16"
        if bf16:
            state = state.bfloat16()
        conv = torch.randn((1, self.slots, 3, 3 * d), generator=self.gen, device="cuda").bfloat16()
        replay = torch.rand((1, record // 4), generator=self.gen, device="cuda")
        state_c, conv_c = state.clone(), conv.clone()
        # The FP32 programs are built without the keyword, as the exporter builds them.
        kw = {"state_dtype": state_dtype} if bf16 else {}
        P("commit", **kw).launch(state, conv, replay, tables, scalars=(len(parts), 1, self.slots))
        P("commit_c", **kw).launch(state_c, conv_c, replay, tables, self.tails.view(1, self.slots, TAIL_BYTES),
                                   self.replay.view(1, 64, 256), scalars=(len(parts), 1, self.slots, 1))
        torch.cuda.synchronize()
        word = torch.int16 if bf16 else torch.int32
        assert torch.equal(state.view(word), state_c.view(word))
        assert torch.equal(conv.view(torch.int16), conv_c.view(torch.int16))
        for (seq, _), keep in zip(parts, keeps):
            seq["len"] += keep
        self.check_tails([seq for seq, _ in parts])

    def check_tails(self, seqs):
        tails = self.tails.cpu()
        keys = self.token_keys.view(torch.int16).cpu()
        for seq in seqs:
            n = seq["len"] % KPOOL
            tail = tails[seq["slot"]]
            assert tail[:16].view(torch.int32).tolist() == [n, 0, 0, 0], (seq["len"], tail[:16].tolist())
            rows = tail[16:].view(torch.int16).view(3, 256)
            for e in range(3):
                if e < n:
                    p = seq["len"] - n + e
                    assert torch.equal(rows[e], keys[self.record(seq, p)]), (seq["len"], e)
                else:
                    assert not rows[e].any(), (seq["len"], e)


def _g():
    from b12x.integration.cuteafd import GLM53_FLASH

    return GLM53_FLASH


@pytest.mark.parametrize("cap", [64, 4096])
def test_prefill_chunks_end_at_every_position_mod_4(cap):
    world = World(_g(), _random_weights(_g(), 1), seed=1)
    seq = world.admit(1200)
    chunks = [1, 2, 3, 4, 5, 6, 7, 61, 64, 63, 1, 1, 1, 2, 3] if cap == 64 else [1, 2, 3, 70, 129, 257, 255, 3, 2, 1]
    for n in chunks:
        world.step([(seq, n)], cap)
    assert seq["len"] == sum(chunks)


def test_prefill_lanes_then_decode_steps_of_several_sequences():
    g = _g()
    world = World(g, _random_weights(g, 2), seed=2)
    seqs = [world.admit(900) for _ in range(6)]
    # Lanes: consecutive chunks of one sequence, split at multiples of 64 rows from the start.
    for i, seq in enumerate(seqs):
        start = [0, 1, 2, 3, 5, 6][i]
        if start:
            world.step([(seq, start)], 4096)
        world.step([(seq, 192)], 4096)
        world.step([(seq, 128 + i)], 4096)
    rng = random.Random(3)
    for _ in range(40):
        parts = [(seq, rng.randint(1, 5)) for seq in rng.sample(seqs, rng.randint(1, len(seqs)))]
        parts.sort(key=lambda part: seqs.index(part[0]))
        world.step(parts, 64)


@pytest.mark.parametrize("rows", [1, 3, 6])
def test_speculative_commit_at_every_kept_count(rows):
    g = _g()
    for keep in range(rows + 1):
        world = World(g, _random_weights(g, 4), seed=10 + keep)
        # Four sequences, one per start position mod 4, in one speculative step.
        seqs = [world.admit(600) for _ in range(4)]
        for r, seq in enumerate(seqs):
            world.step([(seq, 300 + r)], 4096)
        parts = [(seq, rows) for seq in seqs]
        world.step(parts, 64, spec=True)
        world.commit(parts, [keep] * len(seqs))
        # The next steps read the committed tails.
        world.step([(seq, 1 + (i % 3)) for i, seq in enumerate(seqs)], 64)
        world.step([(seq, 4) for seq in seqs], 64)


@pytest.mark.parametrize("rows", [1, 6])
def test_speculative_commit_over_a_bf16_kda_state(rows):
    """The compact cache with ``--kda-state bf16`` commits with ``glmf_kda_commit_c``'s BF16-state
    variant: its KDA half equals ``glmf_kda_commit``'s BF16-state program, its tails as with FP32."""
    g = _g()
    for keep in range(rows + 1):
        world = World(g, _random_weights(g, 4), seed=30 + keep)
        seqs = [world.admit(600) for _ in range(4)]
        for r, seq in enumerate(seqs):
            world.step([(seq, 300 + r)], 4096)
        parts = [(seq, rows) for seq in seqs]
        world.step(parts, 64, spec=True)
        world.commit(parts, [keep] * len(seqs), state_dtype="bfloat16")
        # The next step reads the committed tails.
        world.step([(seq, 1 + (i % 3)) for i, seq in enumerate(seqs)], 64)


def test_speculative_rounds_with_mixed_kept_counts():
    g = _g()
    world = World(g, _random_weights(g, 5), seed=5)
    seqs = [world.admit(800) for _ in range(5)]
    for r, seq in enumerate(seqs):
        world.step([(seq, 200 + r)], 4096)
    rng = random.Random(6)
    for _ in range(25):
        parts = [(seq, rng.randint(1, 8)) for seq in seqs]
        world.step(parts, 64, spec=True)
        world.commit(parts, [rng.randint(0, n) for _, n in parts])
        if rng.random() < 0.3:
            world.step([(seq, rng.randint(1, 3)) for seq in seqs], 64)


def test_real_layer_weights():
    root = os.environ.get("GLMF_SNAPSHOT")
    if not root:
        pytest.skip("GLMF_SNAPSHOT names a GLM-5.3-Flash checkpoint directory")
    g = _g()
    weights, x_rows = _real_inputs(g, Path(root))
    world = World(g, weights, seed=8, x_rows=x_rows)
    seqs = [world.admit(700) for _ in range(4)]
    for r, seq in enumerate(seqs):
        world.step([(seq, 1 + r)], 4096)
        world.step([(seq, 250 + r)], 4096)
    rng = random.Random(9)
    for _ in range(12):
        parts = [(seq, rng.randint(1, 6)) for seq in seqs]
        world.step(parts, 64, spec=True)
        world.commit(parts, [rng.randint(1, n) for _, n in parts])
        world.step([(seq, rng.randint(1, 3)) for seq in seqs], 64)


# Sequence boundaries in wide steps. The row completing a sequence's open pool writes its tail
# and searches for the sequence's last row; a step of up to ``max_rows`` rows needs
# ``last_row_steps(max_rows)`` halvings: 13 up to 8,193 rows, 15 at 32,768
# (tpurtell/sparkinfer-glmrt#1 review). The host check of the depth rule is
# ``test_glmf_index_tail_search_contract.py``.
WIDE = 32768


def _wide_world(seed, cap=WIDE, steps=1):
    g = _g()
    return World(g, _random_weights(g, seed), seed=seed, units=steps * cap // UNIT + 16)


def _lengths(rng, rows, count):
    cuts = sorted(rng.sample(range(1, rows), count - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, rows])]


def test_review_case_two_sequences_in_a_32768_row_step():
    # The review's SM120 reproduction: A holds 1 row, then one step carries 19,001 more rows
    # of A and 13,767 of a new sequence B. A's tail counts 2 (19,002 rows); with 13 halvings
    # its header read [1, 0, 0, 0] while the pooled keys, query and head weights stayed exact.
    world = _wide_world(20)
    a, b = world.admit(19002), world.admit(13767)
    world.step([(a, 1)], WIDE)
    world.step([(a, 19001), (b, 13767)], WIDE)


@pytest.mark.parametrize("start", [1, 2, 3])
def test_full_32768_row_steps_split_between_two_sequences(start):
    # A holds ``start`` rows, part-way into a pool, so the row completing that pool searches
    # for A's last row in the next step: a full step of A's rows then a new sequence B's,
    # split anywhere from just past the pool to B's last row.
    for n in (4, 5, 6, 4097, 8191, 8192, 8193, 16384, 16385, 19001, 24575, 32765, 32766, 32767):
        world = _wide_world(20 + start)
        a, b = world.admit(start + n), world.admit(WIDE - n)
        world.step([(a, start)], 64)
        world.step([(a, n), (b, WIDE - n)], WIDE)


def test_random_sequence_layouts_in_full_32768_row_steps():
    # Up to eight sequences at random positions mod 4 share two full steps; the second step
    # reads the tails the first one left.
    rng = random.Random(23)
    for trial in range(3):
        world = _wide_world(30 + trial, steps=2)
        count = rng.randint(2, 8)
        starts = [rng.randrange(KPOOL) for _ in range(count)]
        layouts = [_lengths(rng, WIDE, count) for _ in range(2)]
        seqs = [world.admit(starts[i] + layouts[0][i] + layouts[1][i]) for i in range(count)]
        opening = [(seq, s) for seq, s in zip(seqs, starts) if s]
        if opening:
            world.step(opening, 64)
        for lengths in layouts:
            world.step(list(zip(seqs, lengths)), WIDE)


def test_the_first_capacity_past_thirteen_halvings():
    # 8,194 rows is the smallest step that 13 halvings cannot always search (14 here): A holds
    # 3 rows, so its first row in the step completes the open pool and searches 8,193 rows on.
    world = _wide_world(40, cap=8194)
    a, b = world.admit(3 + 8193), world.admit(1)
    world.step([(a, 3)], 64)
    world.step([(a, 8193), (b, 1)], 8194)
