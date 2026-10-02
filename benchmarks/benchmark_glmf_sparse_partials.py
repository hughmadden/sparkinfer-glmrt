"""Replay actual cuteafd --geometry-trace MLA inputs through both partial dtypes.

Run in the matching SM120 container on one explicitly allocated GPU. The
trace supplies fixed queries and packed KV records, so the precision arm is
the only changed input. Timings are component measurements, not model tok/s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention._shared.mla.reference import sparse_mla_reference
from b12x.integration.cuteafd._common import GLM53_FLASH
from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot


def read(path, dtype, shape):
    raw = torch.from_numpy(np.fromfile(path, dtype).copy())
    if dtype == np.uint16:
        raw = raw.view(torch.bfloat16)
    return raw.reshape(shape).cuda()


def relative(a, b):
    return float(torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(a.float()))


def capture(program, q, cache, indices, lengths):
    rows = len(q)
    out = torch.empty((rows, 64, 512), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(program.scratch_bytes(64)["scratch"], dtype=torch.uint8, device="cuda")
    args = q, cache, indices, lengths, out, scratch
    for _ in range(3):
        program.launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(*args, scalars=(rows,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0
    return graph, out, args


def duration(graph, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=300)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    path = args.trace / "wide/layer03"
    rows = json.loads((path / "mla_meta.json").read_text())["rows"]
    q = read(path / "mla_query.bin", np.uint16, (rows, 64, 512))
    cache = read(path / "mla_kv.bin", np.uint8, (-1, 64 * 528))
    indices = read(path / "mla_indices.bin", np.int32, (rows, 2112))
    lengths = read(path / "mla_lengths.bin", np.int32, (rows,))
    recorded = read(path / "mla_latent.bin", np.uint16, (rows, 64, 512))
    programs = {name: compile_glm_sparse_mla_aot(GLM53_FLASH, route="decode", max_rows=64,
                    fp32_partials=name == "fp32") for name in ("bf16", "fp32")}
    graphs = {}
    outputs = {}
    holds = []
    report = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
              "scratch": {name: p.scratch_bytes(64) for name, p in programs.items()}, "quality": {}, "timings_us": {}}
    # Freeze compilation before every live count and graph capture/replay.
    with kernel_resolution_guard("GLM Flash partial precision gate"):
        for name, program in programs.items():
            graph, out, keep = capture(program, q, cache, indices, lengths)
            graphs[name, rows] = graph
            holds.append(keep)
            outputs[name] = out.clone()
            serial = []
            for j in range(rows):
                _, one, keep = capture(program, q[j:j + 1], cache, indices[j:j + 1], lengths[j:j + 1])
                holds.append(keep)
                serial.append(one.clone())
            serial = torch.cat(serial)
            report["quality"][name] = {"serial_wide_relative_l2": relative(serial, out)}
            # Retained serial/wide baseline artifacts prove the BF16 arm has
            # the same deployed math, not a silently altered comparison path.
            if name == "bf16":
                assert torch.equal(out, recorded)
                for j in range(rows):
                    expected = read(args.trace / f"serial/row{j:02}/layer03/mla_latent.bin",
                                    np.uint16, (1, 64, 512))
                    assert torch.equal(serial[j:j + 1], expected)
            changed_q = q.clone()
            changed_q[1:] *= -2
            changed_indices = indices.clone()
            changed_indices[1:] = indices[1:].flip(1)
            _, changed, keep = capture(program, changed_q, cache, changed_indices, lengths)
            holds.append(keep)
            assert torch.equal(changed[0], out[0]), "attention depends on later query rows"
            for count in (1, 4, 8, 9, 16, 17, 64):
                repeat_q = q[:1].repeat(count, 1, 1)
                repeat_indices = indices[:1].repeat(count, 1)
                repeat_lengths = lengths[:1].repeat(count)
                graph, repeated, keep = capture(program, repeat_q, cache, repeat_indices, repeat_lengths)
                holds.append(keep)
                graphs[name, count] = graph
                assert torch.isfinite(repeated).all()
            # A paged-address gate must cross the signed-32-bit byte offset.
            page_bytes = 64 * 528
            high_page = (2 ** 31) // page_bytes + 1
            high_cache = torch.empty((high_page + len(cache), page_bytes), dtype=torch.uint8, device="cuda")
            high_cache[high_page:].copy_(cache)
            high_indices = torch.where(indices >= 0, indices + high_page * 64, indices)
            _, high_out, keep = capture(program, q, high_cache, high_indices, lengths)
            assert torch.equal(high_out, out), "paged attention changes past a 2 GiB byte offset"
            # Release the large address-only pool after all queued work drains.
            torch.cuda.synchronize()
            del keep, high_cache, high_indices, high_out
        for count in (1, rows, 64):
            samples = {name: [] for name in programs}
            for iteration in range(3):
                for name in (("bf16", "fp32") if iteration % 2 == 0 else ("fp32", "bf16")):
                    samples[name].append(duration(graphs[name, count], args.repeats))
            report["timings_us"][count] = samples
    # The oracle gathers only valid records, independent of their physical ids.
    oracle = []
    for j in range(rows):
        n = int(lengths[j])
        packed = cache.view(-1, 528)[indices[j, :n].long()].contiguous()
        logical = torch.arange(n, dtype=torch.int32, device="cuda").reshape(1, n)
        oracle.append(sparse_mla_reference(q_all=q[j:j + 1], kv_cache=packed, page_table_1=logical,
                      sm_scale=GLM53_FLASH.softmax_scale, v_head_dim=512))
    oracle = torch.cat(oracle)
    for name, out in outputs.items():
        report["quality"][name]["oracle_relative_l2"] = relative(oracle, out)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
