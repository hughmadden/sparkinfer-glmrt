"""Measure the split KDA packing tax, not end-to-end K6 speed.

Ceiling: one extra read+write of four BF16 projections (16*rows*D bytes).
Stop after one warmed graph timing at decode and 4096 rows; correctness is
qualified separately by tests/integration/test_cuteafd_kimi_split.py.
"""
import json

import torch

from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd._kda_contract import KDAGeometry
from b12x.integration.cuteafd.kimi_split import _KdaPack


def main():
    device = torch.cuda.get_device_properties(0)
    records = []
    for heads in (48, 96):
        g = KDAGeometry(hidden=7168, kda_heads=heads)
        operands = tuple(Operand(n, torch.bfloat16, "[]", "out" if n == "proj" else "in")
                         for n in ("q", "k", "v", "gate", "fab", "proj"))
        program = compile_program(_KdaPack(g), name="kimi_kda_pack_bench",
            operands=operands, scalars=(Scalar("rows"),), key=(g,))
        for rows in (1, 4096):
            tensors = [torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
                       for width in (g.kda_width,) * 4 + (128+heads, g.kda_in_width)]
            run = lambda: program.launch(*tensors, scalars=(rows,))
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    run()
            graph.replay()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); graph.replay(); end.record(); end.synchronize()
            us = begin.elapsed_time(end) * 10
            records.append(dict(heads=heads, rows=rows, microseconds=us,
                extra_projection_copy_bytes=16*rows*g.kda_width,
                total_pack_read_write_bytes=4*rows*g.kda_in_width))
    print(json.dumps(dict(gpu=device.name, sm_count=device.multi_processor_count,
                          clock_rate=getattr(device, "clock_rate", None), results=records), indent=2))


if __name__ == "__main__":
    main()
