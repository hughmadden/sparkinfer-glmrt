#!/usr/bin/env python3
"""MiniMax coordinator duel: independent oracles, baseline bytes and warm graphs.

Run in the cuteafd coordinator dev image. Save baseline tensors with
--references DIR --write-reference; candidates use --references DIR unchanged.
Norm/router may use their existing FP64 bounds with --allow-fp64-bound.
The baseline producers derive from Hugh Madden's donated implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import itertools
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


class NvmlWindow:
    """Sample the selected GPU, retaining unsupported fields explicitly."""

    def __init__(self):
        import pynvml as nv

        self.nv = nv
        nv.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _read(self, fn):
        try:
            return fn()
        except self.nv.NVMLError:
            return None

    def _sample(self):
        n, h = self.nv, self.handle
        while not self.stop.is_set():
            self.samples.append(dict(
                monotonic_s=time.monotonic(),
                power_w=self._read(lambda: n.nvmlDeviceGetPowerUsage(h) / 1000),
                power_limit_w=self._read(lambda: n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000),
                temperature_c=self._read(lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                sm_clock_mhz=self._read(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                memory_clock_mhz=self._read(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)),
                throttle_reasons=self._read(lambda: n.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            ))
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.nv.nvmlShutdown()

    def result(self):
        n = self.nv
        result = {"samples": self.samples, "sample_period_s": 0.05}
        masks = {"power_cap": n.nvmlClocksThrottleReasonSwPowerCap,
                 "thermal": n.nvmlClocksThrottleReasonHwThermalSlowdown | n.nvmlClocksThrottleReasonSwThermalSlowdown}
        for name, mask in masks.items():
            result[name + "_sampled_s"] = sum(
                b["monotonic_s"] - a["monotonic_s"]
                for a, b in zip(self.samples, self.samples[1:])
                if a["throttle_reasons"] is not None and a["throttle_reasons"] & mask)
        return result


def measure(fn, iterations, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(times), times


def check_reference(args, key, outputs, bounded=False):
    if args.references is None:
        return {"baseline_exact": None}
    path = args.references / (key + ".pt")
    actual = [v.detach().contiguous().cpu().view(torch.uint8) for v in outputs]
    if args.write_reference:
        torch.save(actual, path)
        return {"baseline_exact": True}
    expected = torch.load(path, weights_only=True)
    exact = len(actual) == len(expected) and all(torch.equal(a, b) for a, b in zip(actual, expected))
    if not exact and not (bounded and args.allow_fp64_bound):
        raise AssertionError(f"{key}: changed baseline output bytes")
    return {"baseline_exact": exact, "fp64_bound_exception": not exact}


def check_manifest(args, name, program):
    """Pin every public contract field; compiled code/cache identities are excluded."""
    record = dict(name=program.name, abi=program.abi,
                  operands=[asdict(o) for o in program.operands],
                  scalars=[asdict(s) for s in program.scalars],
                  geometry=dict(program.geometry),
                  scratch_bytes={str(rows): program.scratch_bytes(rows) for rows in (1, 4096)})
    record = json.loads(json.dumps(record, default=str))
    if args.references is not None:
        path = args.references / (name + "-manifest.json")
        if args.write_reference:
            path.write_text(json.dumps(record, indent=2) + "\n")
        else:
            expected = json.loads(path.read_text())
            if record != expected:
                fields = sorted(k for k in record.keys() | expected.keys()
                                if record.get(k) != expected.get(k))
                raise AssertionError(f"{name}: changed baseline manifest fields: {fields}")
    return record


def producer_cases(args):
    from b12x.integration.cuteafd.minimax import (
        compile_minimax_norm_aot, compile_minimax_router_scores_aot, compile_minimax_kv_pack_aot)

    dev = "cuda"
    programs = {"norm": compile_minimax_norm_aot, "router": compile_minimax_router_scores_aot,
                "kv_pack": compile_minimax_kv_pack_aot}
    for name, compile_fn in programs.items():
        if name not in args.kernels:
            continue
        program = compile_fn()
        manifest = check_manifest(args, name, program)
        if args.manifest_only:
            yield dict(kernel=name, manifest=manifest)
            continue
        # The same first row is used at every row count, checking row invariance.
        torch.manual_seed(281)
        capacity = max(args.rows) + 1
        if name == "norm":
            x = torch.randn(capacity, 6144, device=dev)
            x[0].zero_()
            w = (torch.randn(6144, device=dev) * 0.1).bfloat16()
            out = torch.empty_like(x, dtype=torch.bfloat16)
            unused = torch.empty(1, device=dev, dtype=torch.bfloat16)
            pos = torch.zeros(capacity, device=dev, dtype=torch.int64)
            operands = (x, w, unused, unused, unused, pos, out)
            ref = (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
                   * (1 + w.double())).bfloat16()
            rtol, atol = 0.008, 0.0001
        elif name == "router":
            x = torch.randn(capacity, 6144, device=dev).bfloat16()
            w = torch.randn(128, 6144, device=dev) * 0.01
            out = torch.empty(capacity, 128, device=dev)
            operands = (x, w, out)
            ref = (x.double() @ w.double().T).float()
            rtol, atol = 2e-5, 2e-6
        else:
            k, v = [torch.randn(capacity, 4, 128, device=dev).bfloat16() for _ in range(2)]
            slots = torch.arange(capacity, device=dev, dtype=torch.int64) * 2 + 64
            slots[3::7] = -1
            batch = (torch.arange(capacity, device=dev) % 3).int()
            kd = torch.tensor([[1, .5, 2, .25], [.75, 1.5, 3, .125], [2.5, .3, 1, .2]], device=dev)
            vd = kd.flip(0).contiguous()
            caches = [torch.empty((capacity * 2 + 128, 4, 128), device=dev, dtype=torch.uint8) for _ in range(2)]
            operands = (k, v, slots, batch, kd, vd, *caches)
            refs = [(value.float() * (1 / scale[batch.long(), :, None])).to(torch.float8_e4m3fn).view(torch.uint8)
                    for value, scale in zip((k, v), (kd, vd))]
        full_bits = None
        if name != "kv_pack":
            program.launch(*operands, scalars=(max(args.rows),))
            full_bits = out[:max(args.rows)].view(torch.uint8).clone()
        first = None
        for rows in args.rows:
            if name == "kv_pack":
                for cache in caches:
                    cache.fill_(173)
            else:
                out.fill_(float("nan"))
            fn = lambda: program.launch(*operands, scalars=(rows,))
            fn()
            if name == "kv_pack":
                for cache, reference in zip(caches, refs):
                    expected = torch.full_like(cache, 173)
                    active = slots[:rows] >= 0
                    expected[slots[:rows][active]] = reference[:rows][active]
                    assert torch.equal(cache, expected), "KV byte oracle or untouched suffix"
                outputs = caches
                row_bits = [cache[64].view(torch.uint8).clone() for cache in caches]
                byte_count, flops = rows * 3072, 0
            else:
                torch.testing.assert_close(out[:rows], ref[:rows], rtol=rtol, atol=atol)
                assert torch.isnan(out[rows:]).all(), "writes beyond live rows"
                assert torch.equal(out[:rows].view(torch.uint8), full_bits[:rows]), "row-count dependent prefix"
                outputs = [out[:rows]]
                row_bits = [out[0].view(torch.uint8).clone()]
                byte_count = rows * 6144 * 6 + 6144 * 2 if name == "norm" else 128 * 6144 * 4 + rows * (6144 * 2 + 128 * 4)
                flops = rows * 2 * 6144 * 128 if name == "router" else 0
            if first is not None:
                assert all(torch.equal(a, b) for a, b in zip(first, row_bits)), "row-count dependent output"
            first = row_bits
            comparison = check_reference(args, f"{name}-{rows}", outputs, bounded=name != "kv_pack")
            us, raw = measure(fn, args.iterations, args.samples)
            yield dict(kernel=name, rows=rows, median_us=us, samples_us=raw,
                       nominal_bytes=byte_count, nominal_flops=flops, manifest=manifest, **comparison)


def gather_cases(args):
    if "gather" not in args.kernels:
        return
    from b12x.integration.cuteafd.index_cache import compile_index_cache_gather_aot

    program = compile_index_cache_gather_aot()
    for context in args.gather_contexts:
        extent = (context + 255) // 256 * 256
        live = context if args.gather_live is None else min(args.gather_live, context)
        pool = (context + 63) // 64 + 4
        cache = (torch.arange(pool * 8448, device="cuda", dtype=torch.int32) * 37 + 19).to(torch.uint8).reshape(pool, 8448)
        pages = torch.arange(pool - 2, device="cuda", dtype=torch.int32).flip(0).contiguous()
        pages[1::11] = -1
        pages[7::13] = pool
        lengths = torch.arange(max(args.rows) + 1, device="cuda", dtype=torch.int32) * 67 - 17
        keys = torch.empty((extent + 256, 128), device="cuda", dtype=torch.uint8)
        scales = torch.empty(extent + 256, device="cuda")
        starts, ends = [torch.empty_like(lengths) for _ in range(2)]
        # Independent CPU byte oracle, including invalid pages and padded suffix.
        source, table = cache.cpu(), pages.cpu().tolist()
        expected_keys = torch.zeros((extent, 128), dtype=torch.uint8)
        expected_scales = torch.zeros((extent, 4), dtype=torch.uint8)
        for token in range(live):
            col = args.gather_page_begin + token // 64
            if 0 <= col < len(table) and 0 <= table[col] < pool:
                slot = token % 64
                expected_keys[token] = source[table[col], slot * 128:(slot + 1) * 128]
                expected_scales[token] = source[table[col], 8192 + slot * 4:8192 + (slot + 1) * 4]
        for rows in args.rows:
            keys.fill_(173)
            scales.view(torch.uint8).fill_(173)
            starts.fill_(-333)
            ends.fill_(-333)
            operands = (cache, pages, lengths, keys, scales, starts, ends)
            scalars = (rows, len(table), args.gather_page_begin, pool, live, extent)
            fn = lambda: program.launch(*operands, scalars=scalars)
            fn()
            assert torch.equal(keys[:extent].cpu(), expected_keys)
            assert torch.equal(scales[:extent].view(torch.uint8).reshape(-1, 4).cpu(), expected_scales)
            assert (keys[extent:] == 173).all() and (scales[extent:].view(torch.uint8) == 173).all()
            assert (starts[:rows] == 0).all()
            assert torch.equal(ends[:rows], (lengths[:rows].long() - args.gather_page_begin * 64).clamp(0, live).int())
            assert (starts[rows:] == -333).all() and (ends[rows:] == -333).all()
            key = f"gather-{rows}-{context}-{live}-{args.gather_page_begin}"
            comparison = check_reference(args, key, (keys, scales, starts, ends))
            us, raw = measure(fn, args.iterations, args.samples)
            yield dict(kernel="gather", rows=rows, context=context, live=live, extent=extent,
                       median_us=us, samples_us=raw, nominal_bytes=(live + extent) * 132 + rows * 12,
                       nominal_flops=0, **comparison)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="1,2,4,8,16,32,64,127,257,2048,2752,4096")
    p.add_argument("--kernels", default="norm,router,kv_pack,gather")
    p.add_argument("--gather-contexts", default="8192")
    p.add_argument("--gather-live", type=int)
    p.add_argument("--gather-page-begin", type=int, default=0)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--references", type=Path)
    p.add_argument("--write-reference", action="store_true")
    p.add_argument("--allow-fp64-bound", action="store_true")
    p.add_argument("--manifest-only", action="store_true", help="compile and check producer contracts without timing")
    p.add_argument("--skip-tests", action="store_true", help="only after the unchanged #281/#282 tests ran separately")
    p.add_argument("--json", required=True, type=Path)
    args = p.parse_args()
    args.rows = [int(v) for v in args.rows.split(",")]
    args.gather_contexts = [int(v) for v in args.gather_contexts.split(",")]
    args.kernels = args.kernels.split(",")
    if (not args.rows or min(args.rows) < 1 or min(args.gather_contexts) < 1
            or args.iterations < 1 or args.samples < 1 or (args.gather_live is not None and args.gather_live < 0)):
        p.error("rows, contexts, iterations and samples must be positive; live must be nonnegative")
    if args.manifest_only and "gather" in args.kernels:
        p.error("--manifest-only supports producers only")
    if args.write_reference and args.references is None:
        p.error("--write-reference requires --references")
    if args.references is not None and args.write_reference:
        args.references.mkdir(parents=True, exist_ok=True)
    if not args.skip_tests:
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        str(root / "tests/integration/test_cuteafd_minimax_gpu.py"),
                        str(root / "tests/integration/test_cuteafd_index_cache_gather.py")], check=True)
    props = torch.cuda.get_device_properties(0)
    results = []
    with NvmlWindow() as window:
        for result in itertools.chain(producer_cases(args), gather_cases(args)):
            results.append(result)
            print(json.dumps(result), flush=True)
    payload = dict(device=dict(name=props.name, uuid=str(props.uuid), sm_count=props.multi_processor_count,
                               total_memory=props.total_memory), results=results, nvml=window.result(),
                   conditions=dict(warm_graph=True, l2_flush=False, iterations=args.iterations,
                                   samples=args.samples, tests_skipped=args.skip_tests))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
