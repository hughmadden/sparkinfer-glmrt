"""CPU execution of MSA metadata adapters; QK/attention kernels need GPU gates."""
from __future__ import annotations

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd"


def module(name):
    tree = ast.parse((ROOT / f"{name}.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    arch = SimpleNamespace(thread_idx=lambda: (0, 0, 0), block_idx=lambda: (0, 0, 0))
    env = dict(Int32=np.int32, Int64=np.int64, Float32=SimpleNamespace(inf=np.inf),
        const_expr=bool, cutlass=SimpleNamespace(range_constexpr=range),
        cute=SimpleNamespace(arch=arch), attention_ops=SimpleNamespace(fmax=np.maximum))
    class F32:
        inf = np.inf

        def __new__(cls, value):
            return np.float32(value)
    env["Float32"] = F32
    exec(compile(tree, str(ROOT / f"{name}.py"), "exec", flags=__future__.annotations.compiler_flag), env)
    return env, arch


def execute(arch, fn, grid, threads, *args):
    for x in range(grid[0]):
        for y in range(grid[1]):
            arch.block_idx = lambda: (x, y, 0)
            for tid in range(threads):
                arch.thread_idx = lambda: (tid, 0, 0)
                fn(*args)


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("rows,pages", [(1, 1), (3, 31), (128, 16384)])
def test_index_scratch_regions_are_aligned_and_disjoint(mode, rows, pages):
    env, _ = module("minimax_indexer")
    blocks, offsets, size = env["_layout"](rows, pages, mode)
    assert blocks == (pages + 1) // 2
    assert all(x % 256 == 0 for x in offsets.values())
    assert list(offsets.values()) == sorted(offsets.values())
    assert size >= offsets["controls"] + 256
    assert env["minimax_index_scratch_bytes"](max_rows=rows, max_pages=pages, mode=mode) == size


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_attention_metadata_preserves_request_boundaries(mode):
    env, arch = module("minimax_attention")
    obj = object.__new__(env["_Attention"])
    obj.tokens = 1 if mode == "decode" else 8
    cu = np.array([0, 1, 10, 23], dtype=np.int32)
    rows, batch = 23, 3
    work = rows if mode == "decode" else (rows + 7) // 8 + batch
    requests, tiles, zeros, valid = [np.full(work, -99, dtype=np.int32) for _ in range(4)]
    execute(arch, obj.metadata, (1, 1), 128, cu, requests, tiles, zeros, valid, work, batch)
    expected = [(r, t) for r in range(batch) for t in range((cu[r + 1] - cu[r] + obj.tokens - 1) // obj.tokens)]
    assert list(zip(requests[valid == 1], tiles[valid == 1])) == expected
    assert (zeros == 0).all()
    assert (valid[len(expected):] == 0).all()


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_pool_force_and_publication_reuse_fixed_head_stride(mode):
    env, arch = module("minimax_indexer")
    obj = object.__new__(env["_Index"])
    obj.mode, obj.rows, obj.pages, obj.blocks = mode, 7, 37, 19
    rows = 3
    rng = np.random.default_rng(17)
    pages = rng.normal(size=(4, rows, obj.pages)).astype(np.float32)
    scores = rng.normal(size=(4, rows, obj.blocks)).astype(np.float32)
    original = scores.copy()
    k_start = np.array([0, 128, 256], dtype=np.int32) if mode == "prefill" else np.zeros(rows, np.int32)
    k_end = np.array([1, 17 * 128 + 5, 18 * 128 + 3], dtype=np.int32)
    starts, lengths = np.full(rows * 4, -99, np.int32), np.full(rows * 4, -99, np.int32)
    execute(arch, obj.finish_scores, (rows * 4, 1), 128, pages.ravel(), scores.ravel(),
            starts, lengths, k_start, k_end, rows, obj.pages)
    selected = np.full((4, rows, 16), -1, dtype=np.int32)
    for h in range(4):
        for r in range(rows):
            begin, end = k_start[r] // 128, (k_end[r] + 127) // 128
            assert starts[h * rows + r] == begin
            assert lengths[h * rows + r] == end - begin
            assert np.isposinf(scores[h, r, end - 1])
            for b in range(begin, end - 1):
                expected = max(pages[h, r, 2*b:2*b+2]) if mode == "decode" else original[h, r, b]
                assert scores[h, r, b] == expected
            assert np.isneginf(scores[h, r, :begin]).all()
            assert np.isneginf(scores[h, r, end:]).all()
            chosen = sorted(range(begin, end), key=lambda b: (-scores[h, r, b], b))[:16]
            chosen.sort()
            selected[h, r, :len(chosen)] = chosen
    output = np.full((4, obj.rows, 16), -88, dtype=np.int32)
    shuffled = selected[:, :, ::-1].copy()
    execute(arch, obj.publish, (2, 1), 128, shuffled.ravel(), starts, output.ravel(), rows)
    for h in range(4):
        for r in range(rows):
            expected = [int(v - k_start[r] // 128) if v >= 0 else -1 for v in selected[h, r]]
            assert output[h, r].tolist() == expected
        assert (output[h, rows:] == -88).all()


def test_union_masks_and_partial_tiles_keep_each_rows_selection():
    env, arch = module("minimax_attention")
    obj = object.__new__(env["_Attention"])
    obj.rows = 20
    cu = np.array([0, 3, 12], dtype=np.int32)
    lengths = np.array([300, 2200], dtype=np.int32)
    requests, tiles, valid = np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0]), np.array([1, 1, 1, 0])
    selected = np.full((4, obj.rows, 16), -1, dtype=np.int32)
    for h in range(4):
        for row in range(12):
            selected[h, row, :3] = [0, 1 + (row + h) % 2, 2 if row < 3 else 17]
    # A later row introduces a lower block after the first row's local block.
    selected[:, 3, :3] = [0, 7, 17]
    selected[:, 4, :3] = [0, 3, 17]
    blocks, masks = np.full((4, 4, 128), -99, np.int32), np.full((4, 4, 128), -99, np.int32)
    counts = np.full((4, 4), -99, np.int32)
    execute(arch, obj.union, (4, 4), 1, selected.ravel(), lengths, cu, requests, tiles, valid,
            blocks.ravel(), masks.ravel(), counts.ravel(), 4)
    for work in range(4):
        for h in range(4):
            if not valid[work]:
                assert counts[work, h] == 0
                continue
            req, first = requests[work], tiles[work] * 8
            qlen = cu[req + 1] - cu[req]
            union = blocks[work, h, :counts[work, h]].tolist()
            assert union == sorted(set(union))
            mapping = dict(zip(union, masks[work, h, :counts[work, h]]))
            for token in range(min(8, qlen - first)):
                row = cu[req] + first + token
                visible = lengths[req] - qlen + first + token + 1
                expected = {int(b) for b in selected[h, row] if b >= 0 and b * 128 < visible}
                assert {int(b) for b, mask in mapping.items() if mask & (1 << token)} == expected


def test_invalid_capacities_fail_before_compilation():
    for name, args in [("minimax_indexer", (0, 1, "decode")),
                       ("minimax_indexer", (1, 1, "extend")),
                       ("minimax_indexer", (1 << 20, 1 << 20, "prefill")),
                       ("minimax_attention", (0, 1, "decode")),
                       ("minimax_attention", (1, 1, "prefill"))]:
        env, _ = module(name)
        with pytest.raises(ValueError):
            env["_layout"](*args)
