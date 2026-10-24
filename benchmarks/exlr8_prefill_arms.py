#!/usr/bin/env python3
"""Isolate one term of the grouped EXLR8 expert prefill with a source patch.

Apply to a fresh `cp -al` copy of the fork tree (patch the copy's copy of
`b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py`; never the shared tree), then
run `benchmarks/bench_exlr8_grouped_prefill.py` against that tree:

    cp -al /path/to/fork /tmp/arm && cp --remove-destination \
        /path/to/fork/b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py /tmp/arm/p.tmp
    python3 benchmarks/exlr8_prefill_arms.py halfmma /tmp/arm/p.tmp
    mv /tmp/arm/p.tmp /tmp/arm/b12x/moe/_shared/kernels/w4a16/exlr8_grouped.py

  none     unpatched A16, the shipped kernel
  mma2     every f16 MMA duplicated into its own accumulators: 2x tensor-core
           instructions with identical loads and decode, so it measures the
           MMA leg's marginal cost
  halfmma  the high-N8 MMA of every pair replaced by a 2-op consumption of its
           B words: half the MMA instructions, decode and A loads unchanged —
           the upper bound for what m16n8k32 (A8) can buy for the same FLOPs
  nomma    every MMA replaced the same way: the decode leg plus loads, fold,
           activation and reduce, with the tensor pipe idle
  fold     the FC1 split fold's global read-add dropped (the write stays)

Measured on GB10/SM121 (48 SM, 24 MiB L2), hidden 6144, units 16, top_k 8,
group_rows 64, rows 4096, max_groups 512, one warm launch per arm: A16 43.33 ms,
mma2 63.00, nomma 24.26, halfmma 34.26, fold 40.73 — decode 19.9 ms + MMA
19.1 ms + 4.4 ms of loads/fold/activation, i.e. the MMA leg is 44% and halving
its instruction count is worth 9.1 ms. An earlier "decode and MMA removed"
arm read 23.94 ms because dropping the MMA let DCE delete the decode with it;
`nomma` keeps the decode live and that arm is the one to trust.
"""
import sys
from pathlib import Path

CALL = "f16_mma_m16n8k16_f32("
FOLD = """                            else:
                                g0, g1 = ld_global_v2_f32(addr)
                                st_global_v2_f32(addr, g0 + e0, g1 + e1)
"""
FOLD_NEW = """                            else:
                                st_global_v2_f32(addr, e0, e1)
"""


def split_args(text, start):
    """Return (args, end_index_after_close_paren) for a call body at `start`."""
    depth, args, cur, i = 1, [], [], start  # depth 1: inside the call's parens
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i + 1
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    raise SystemExit("unbalanced call")


def double_mma(text):
    out, pos, n = [], 0, 0
    while True:
        hit = text.find("= " + CALL, pos)
        if hit < 0:
            out.append(text[pos:])
            return "".join(out), n
        open_at = hit + len("= ") + len(CALL)
        args, end = split_args(text, open_at)
        if len(args) != 10:
            raise SystemExit(f"unexpected MMA arity {len(args)}: {args}")
        stmt_start = text.rfind("\n", 0, hit) + 1
        line = text[stmt_start:hit]
        indent = line[: len(line) - len(line.lstrip())]
        names = [a.strip() for a in line.split(",")]
        if len(names) != 4:
            raise SystemExit(f"unexpected MMA result list: {names}")
        out.append(text[pos:end])
        out.append(
            "\n" + indent + ", ".join(names) + " = " + CALL
            + ", ".join(names + args[4:]) + ")"
        )
        pos = end
        n += 1


def drop_reassign(text, pos, names):
    """Skip a trailing `<acc...> = dX, dY, dZ, dW` line that used the MMA result."""
    import re
    m = re.compile(r"[ \t]*\n[ \t]*[^\n]+ = " + re.escape(", ".join(names)) + r"[ \t]*(\n|$)")
    hit = m.match(text, pos)
    return hit.end() if hit else None


def drop_mma(text):
    """Replace every MMA with a 2-op consumption of the decoded B word.

    Keeps the trellis decode and its loads live (the asm's B output feeds the
    accumulator) while leaving the tensor pipe idle, so the arm times the
    decode leg instead of letting DCE delete it.
    """
    out, pos, n = [], 0, 0
    while True:
        hit = text.find("= " + CALL, pos)
        if hit < 0:
            out.append(text[pos:])
            return "".join(out), n
        open_at = hit + len("= ") + len(CALL)
        args, end = split_args(text, open_at)
        if len(args) != 10:
            raise SystemExit(f"unexpected MMA arity {len(args)}: {args}")
        stmt_start = text.rfind("\n", 0, hit) + 1
        line = text[stmt_start:hit]
        indent = line[: len(line) - len(line.lstrip())]
        names = [a.strip() for a in line.split(",")]
        if len(names) != 4:
            raise SystemExit(f"unexpected MMA result list: {names}")
        out.append(text[pos:stmt_start])
        out.append(
            f"{indent}nb, _ = half2_to_float2_scaled({args[8]}, Float32(1.0))\n"
            f"{indent}{args[0]} = {args[0]} + nb\n"
        )
        pos = drop_reassign(text, end, names) or end
        n += 1


def half_mma(text):
    """Keep the low-N8 MMA, replace the high-N8 MMA (results d4..d7) with a
    2-op consumption: halves MMA instructions, keeps decode and A loads live."""
    out, pos, n = [], 0, 0
    while True:
        hit = text.find("= " + CALL, pos)
        if hit < 0:
            out.append(text[pos:])
            return "".join(out), n
        open_at = hit + len("= ") + len(CALL)
        args, end = split_args(text, open_at)
        stmt_start = text.rfind("\n", 0, hit) + 1
        line = text[stmt_start:hit]
        indent = line[: len(line) - len(line.lstrip())]
        names = [a.strip() for a in line.split(",")]
        if names[0] != "d4":
            out.append(text[pos:end])
            pos = end
            continue
        out.append(text[pos:stmt_start])
        out.append(
            f"{indent}nb2, _ = half2_to_float2_scaled({args[8]}, Float32(1.0))\n"
            f"{indent}nb3, _ = half2_to_float2_scaled({args[9]}, Float32(1.0))\n"
            f"{indent}{args[0]} = {args[0]} + nb2 + nb3\n"
        )
        pos = drop_reassign(text, end, names) or end
        n += 1


def main():
    arm, path = sys.argv[1], Path(sys.argv[2])
    text = path.read_text()
    if arm == "mma2":
        text, n = double_mma(text)
        print(f"doubled {n} MMA sites")
    elif arm == "halfmma":
        text, n = half_mma(text)
        print(f"replaced {n} high-N8 MMA sites with a decode consumption")
    elif arm == "nomma":
        text, n = drop_mma(text)
        print(f"replaced {n} MMA sites with a decode consumption")
    elif arm == "fold":
        if text.count(FOLD) != 1:
            raise SystemExit(f"fold pattern count {text.count(FOLD)} != 1")
        text = text.replace(FOLD, FOLD_NEW)
        print("removed the split-fold read-modify-write")
    elif arm != "none":
        raise SystemExit("unknown arm")
    path.write_text(text)


if __name__ == "__main__":
    main()
