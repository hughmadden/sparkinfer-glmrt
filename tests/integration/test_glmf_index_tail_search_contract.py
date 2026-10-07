"""GLM Flash's compact index tails: the depth of the search ``GlmfPoolKeysTail`` runs for a
sequence's last step row, independent of a CUDA device.

The row that writes the tail of a sequence which does not end the step finds the sequence's
last row with an unrolled binary search over ``seq_first``. ``host_last_row`` runs the
kernel's own ``_last_row`` statements on the host: its decorator dropped, ``Int64`` and
``Int32`` as Python ints and ``range_constexpr`` as ``range``. The CuTe DSL traces those
statements into the same integer operations (the loop unrolls, the ``if``s become branches),
and every value is a row index in ``[0, rows)``, where 64-bit arithmetic and floor division
agree with Python's, so the host answer is the kernel's. ``tail_counts`` mirrors by hand the
kernel's choice of the row that writes each tail and the count it stores. The rows the tail
holds are checked on a GPU (``test_cuteafd_glmf_index_compact_aot.py``).
"""

from __future__ import annotations

import ast
import inspect
import random
import textwrap
from types import SimpleNamespace

import pytest

from b12x.integration.cuteafd import _glmf_kernels
from b12x.integration.cuteafd._glmf_kernels import GlmfPoolKeysTail, last_row_steps

KPOOL = 4


def _host_method(cls, name: str):
    """``cls.name``'s statements as a host function (decorators dropped)."""
    source, line = inspect.getsourcelines(cls)
    tree = ast.parse(textwrap.dedent("".join(source)))
    ast.increment_lineno(tree, line - 1)
    fn = next(node for node in tree.body[0].body if isinstance(node, ast.FunctionDef) and node.name == name)
    fn.decorator_list = []
    namespace = {"Int64": int, "Int32": int, "cutlass": SimpleNamespace(range_constexpr=range),
                 "cute": SimpleNamespace(Tensor=object)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), inspect.getsourcefile(cls), "exec"), namespace)
    return namespace[name]


_LAST_ROW = _host_method(GlmfPoolKeysTail, "_last_row")


def host_last_row(seq_first, first: int, row: int, rows: int, steps: int) -> int:
    """``GlmfPoolKeysTail._last_row`` unrolled ``steps`` deep."""
    return _LAST_ROW(SimpleNamespace(search_steps=steps), seq_first, first, row, rows)


class _Split:
    """``seq_first`` of a step whose first sequence has rows ``0 .. at - 1`` and whose next
    sequence starts at row ``at`` (only the boundary matters to the search)."""

    def __init__(self, at: int):
        self.at = at

    def __getitem__(self, row: int) -> int:
        return 0 if row < self.at else self.at


def tail_counts(lengths, starts, steps: int) -> list[list[int]]:
    """The tail counts written for each sequence of a non-speculative step: sequence ``i``
    has ``lengths[i]`` rows starting at position ``starts[i]``. Mirrors the tail write of
    ``GlmfPoolKeysTail.kernel``: the row completing the pool open before the step writes
    (else the sequence's last row), and it stores ``(positions[end] + 1) % 4`` for the
    sequence's last row ``end``, found by the search unless the writer is that row."""
    seq_first, positions, index = [], [], {}
    for i, (n, start) in enumerate(zip(lengths, starts)):
        index[len(seq_first)] = i
        seq_first += [len(seq_first)] * n
        positions += range(start, start + n)
    rows = len(seq_first)
    written = [[] for _ in lengths]
    for token in range(rows):
        first = seq_first[token]
        held = positions[first] % KPOOL
        last = token + 1 == rows or seq_first[token + 1] != first
        opener = first + KPOOL - 1 - held
        if (token == opener or (last and opener > token)) if held > 0 else last:
            end = token if last else host_last_row(seq_first, first, token, rows, steps)
            written[index[first]].append((positions[end] + 1) % KPOOL)
    return written


def _layout(rng: random.Random, rows: int, count: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, rows), count - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, rows])]


@pytest.mark.parametrize("max_rows,steps", [(1, 13), (64, 13), (128, 13), (4096, 13), (8192, 13), (8193, 13),
                                            (8194, 14), (16385, 14), (16386, 15), (32768, 15), (65536, 16)])
def test_search_depth_follows_the_program_capacity(max_rows, steps):
    # 13 up to 8,193 rows: the 64- and 4,096-row programs keep the search they were qualified with.
    assert last_row_steps(max_rows) == steps
    assert GlmfPoolKeysTail(max_rows=max_rows).search_steps == steps


def test_the_rule_is_the_fewest_halvings_that_always_reach_the_boundary(monkeypatch):
    # Without the 13-step floor, every capacity up to 130 rows, every starting row and every
    # boundary: the rule's depth finds the sequence's last row, and one step fewer does not.
    monkeypatch.setattr(_glmf_kernels, "_LAST_ROW_STEPS", 0)
    for max_rows in range(2, 131):
        steps = last_row_steps(max_rows)
        cases = [(row, end) for row in range(max_rows - 1) for end in range(row, max_rows - 1)]
        assert all(host_last_row(_Split(end + 1), 0, row, max_rows, steps) == end for row, end in cases)
        if steps:
            assert any(host_last_row(_Split(end + 1), 0, row, max_rows, steps - 1) != end for row, end in cases)


@pytest.mark.parametrize("max_rows", [8193, 8194, 16386, 32768])
def test_every_boundary_of_a_full_step_is_found(max_rows):
    # The widest searches: from rows 0, 1 and 2 (the row completing the open pool of a step's
    # first sequence) over a full step, to every possible last row of that sequence.
    steps = last_row_steps(max_rows)
    for row in range(3):
        for end in range(row, max_rows - 1):
            assert host_last_row(_Split(end + 1), 0, row, max_rows, steps) == end, (row, end)
    assert any(host_last_row(_Split(end + 1), 0, 0, max_rows, steps - 1) != end for end in range(max_rows - 1))


def test_review_case_two_sequences_in_a_32768_row_step():
    # tpurtell/sparkinfer-glmrt#1 (review, on SM120): sequence A holds 1 row; a 32,768-row
    # step carries 19,001 more rows of A, then 13,767 of a new sequence B. A ends at 19,002
    # rows, so its tail counts 2. The 13-step search stopped one row short: count 1, the
    # header [1, 0, 0, 0] seen on the GPU.
    lengths, starts = [19001, 13767], [1, 0]
    assert tail_counts(lengths, starts, steps=13) == [[1], [3]]
    assert tail_counts(lengths, starts, steps=last_row_steps(32768)) == [[2], [3]]


@pytest.mark.parametrize("max_rows,layouts", [(64, 400), (128, 300), (4096, 100), (32768, 40)])
def test_each_sequence_gets_one_tail_write_with_its_count(max_rows, layouts):
    # Random steps of up to eight sequences at random positions mod 4: exactly one row writes
    # each sequence's tail, and the count it writes is the sequence's new length mod 4.
    rng = random.Random(max_rows)
    steps = last_row_steps(max_rows)
    for _ in range(layouts):
        rows = rng.randint(1, max_rows)
        lengths = _layout(rng, rows, rng.randint(1, min(8, rows)))
        starts = [rng.randrange(1000) for _ in lengths]
        expected = [[(start + n) % KPOOL] for start, n in zip(starts, lengths)]
        assert tail_counts(lengths, starts, steps) == expected, (lengths, starts)
