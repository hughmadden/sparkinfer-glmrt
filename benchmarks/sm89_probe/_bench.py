"""Timing helpers shared by the SM89 probes."""

import torch


def time_rotating(calls, rounds: int = 20, warmup: int = 2) -> float:
    """Mean microseconds per call over back-to-back launches that rotate through
    ``calls`` (one per distinct weight copy, together larger than L2), so every
    call streams its weight from DRAM and L2 holds clean lines, as in a decode
    step walking through layers. Launch gaps are included."""
    for _ in range(warmup):
        for fn in calls:
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        for fn in calls:
            fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (rounds * len(calls))
