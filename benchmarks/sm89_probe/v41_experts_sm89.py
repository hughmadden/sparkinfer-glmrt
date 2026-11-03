#!/usr/bin/env python3
"""SM89 V4.1 W4A8 routed experts: numerics and L2-cold timing of the slice pipeline.

Runs the production pipelines (``V41DraftSlicePipeline`` for the coordinator
draft slice, ``V41SlicePipeline`` for the RTX TP2/full backbone roles) on the
SM120 resident layout with V4.1 scale codes, and compares against an FP32
PyTorch reference of the same W4A8 recipe: MXFP8 K32 activations with the
1e-4 amax floor, BF16-rounded FC1 outputs, gate <= 10 and |up| <= 10, SiLU,
route weight, BF16, MXFP8 requantization, FP32 FC2 per route.

Timing replays a CUDA graph of back-to-back calls whose expert ids walk a
resident pool far larger than L2, so every call streams its weights from DRAM
(the probe's method; no L2 flush, which would leave dirty lines). Bandwidth
counts packed weight bytes (payload + scale codes) of the experts touched.
"""
import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

H = 5120
WIRE = H + H // 32
DEV = "cuda"
PEAK = 1008.0

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def stream():
    from b12x._lib.utils import current_cuda_stream
    return current_cuda_stream()


# ----------------------------------------------------------------------------- packing
def pack_plane(words):
    """Logical int32 words [E, rows, K/8] -> N256/K128 lane-major resident bytes [E, -1]."""
    e, rows, kw = words.shape
    t = words.reshape(e, rows // 256, 8, 4, 8, kw // 16, 4, 4)  # e, rg, chunk, w, g, kt, kb, c
    return t.permute(0, 1, 5, 6, 2, 4, 7, 3).contiguous().view(torch.uint8).reshape(e, -1)


def pack_raw_scales(scales):
    e, rows, kb = scales.shape
    t = scales.reshape(e, rows // 256, 256, kb // 4, 4).permute(0, 1, 3, 2, 4)
    return t.contiguous().reshape(e, -1)


def pack_experts(w1, w3, w2, s1, s3, s2, codes=True):
    """Logical MXFP4 experts -> resident [w13, s13, w2, s2] as flat uint32 tensors."""
    from b12x.moe._shared.scale_codes import pack_scale_codes
    planes = [pack_plane(torch.cat((w3, w1), 1).view(torch.int32))]
    if codes:
        planes.append(pack_scale_codes(torch.cat((s3, s1), 1), tensor_name="w13.scale",
                                       gated_half_rows=w1.shape[1]))
    else:
        planes.append(pack_raw_scales(torch.cat((s3, s1), 1)))
    planes.append(pack_plane(w2.view(torch.int32)))
    planes.append(pack_scale_codes(s2, tensor_name="w2.scale") if codes else pack_raw_scales(s2))
    return [p.contiguous().reshape(-1).view(torch.uint32) for p in planes]


def plane_bytes(inter, codes=True):
    """Per-expert resident bytes: W13, S13, W2, S2 (matches the exporter)."""
    s13 = 2 * inter * (2 * (H // 128) + 1) if codes else 2 * inter * H // 32
    s2 = H * (2 * (inter // 128) + 1) if codes else H * inter // 32
    return [inter * H, s13, inter * H // 2, s2]


class Pool:
    """Resident experts. ``logical`` experts keep their source tensors for references;
    the rest are random resident bytes with valid scale codes (timing only)."""

    def __init__(self, n, inter, logical, seed, codes=True):
        self.n, self.inter, self.codes = n, inter, codes
        self.sizes = plane_bytes(inter, codes)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(seed)
        self.planes = [torch.empty(n * size, dtype=torch.uint8, device=DEV) for size in self.sizes]
        for plane, size in zip(self.planes, self.sizes):
            view = plane.view(n, size)
            for e0 in range(0, n, 16):
                blk = view[e0:e0 + 16]
                blk.copy_(torch.randint(0, 256, blk.shape, dtype=torch.uint8, device=DEV, generator=gen))
        if codes:
            # Valid codes: row bases 112..120 (2^-15..2^-7 with offsets 0..15).
            for plane, rows, tiles in ((self.planes[1], 2 * inter, H // 128),
                                       (self.planes[3], H, inter // 128)):
                view = plane.view(n, -1)
                view[:, rows * tiles * 2:] = torch.randint(112, 121, (n, rows), dtype=torch.uint8,
                                                           device=DEV, generator=gen)
        else:
            for plane in (self.planes[1], self.planes[3]):
                plane.copy_(torch.randint(117, 125, plane.shape, dtype=torch.uint8, device=DEV, generator=gen))
        self.logical = {}
        for e in range(logical):
            self.set_logical(e, gen)

    @property
    def expert_bytes(self):
        return sum(self.sizes)

    def set_logical(self, e, gen):
        inter = self.inter
        w = {name: torch.randint(0, 256, shape, dtype=torch.uint8, device=DEV, generator=gen)
             for name, shape in (("w1", (1, inter, H // 2)), ("w3", (1, inter, H // 2)),
                                 ("w2", (1, H, inter // 2)))}
        s = {name: torch.randint(117, 124, (*w[name].shape[:-1], w[name].shape[-1] // 16),
                                 dtype=torch.uint8, device=DEV, generator=gen) for name in w}
        packed = pack_experts(w["w1"], w["w3"], w["w2"], s["w1"], s["w3"], s["w2"], self.codes)
        for plane, part, size in zip(self.planes, packed, self.sizes):
            assert part.numel() * 4 == size
            plane[e * size:(e + 1) * size] = part.view(torch.uint8)
        self.logical[e] = (w, s)

    def dequant(self, e):
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=DEV)
        w, s = self.logical[e]
        out = []
        for name in ("w1", "w3", "w2"):
            codes = w[name][0]
            values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).flatten(-2)
            scale = torch.exp2(s[name][0].float() - 127).repeat_interleave(32, -1)
            out.append(values * scale)
        return out

    def tensors(self):
        return [p.view(torch.uint32) for p in self.planes]


# ----------------------------------------------------------------------------- MX helpers
def mx_quant(x):
    """MXFP8 K32: UE8M0 pow2-ceil of max(amax, 1e-4)/448, E4M3 RN payload."""
    shp = x.shape
    b = x.float().reshape(-1, shp[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).contiguous().view(torch.int32)
    bits = torch.where(bits & 0x7FFFFF != 0, (bits + 0x800000) & 0x7F800000, bits)
    byte = ((bits >> 23) & 0xFF).to(torch.uint8)
    inv = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    q = (b * inv).clamp(-448, 448).to(torch.float8_e4m3fn)
    deq = q.float() * bits.view(torch.float32)
    return q.view(torch.uint8).reshape(shp), byte.reshape(*shp[:-1], shp[-1] // 32), deq.reshape(shp)


def mx_dequant(codes, sbytes):
    shp = codes.shape
    v = codes.reshape(-1, shp[-1] // 32, 32).view(torch.float8_e4m3fn).float()
    s = torch.exp2(sbytes.reshape(-1, shp[-1] // 32, 1).float() - 127)
    return (v * s).reshape(shp)


def make_x(m, gen):
    x = torch.randn(m, H, device=DEV, generator=gen)
    chan = torch.ones(H, device=DEV)
    chan[torch.randint(0, H, (64,), device=DEV, generator=gen)] = 8.0
    return (x * chan).to(torch.bfloat16)


def wire_rows(x):
    q, s, _ = mx_quant(x.float())
    wire = torch.empty(x.shape[0], WIRE, dtype=torch.uint8, device=DEV)
    wire[:, :H] = q
    wire[:, H:] = s
    return wire


def reference(pool, wire, ids, rw):
    """FP32 reference per original route (token-major, topk-minor)."""
    m, topk = ids.shape
    inter = pool.inter
    xdq = mx_dequant(wire[:, :H].contiguous(), wire[:, H:].contiguous())
    routes = m * topk
    out = torch.zeros(routes, H, device=DEV)
    hc = torch.zeros(routes, inter, dtype=torch.uint8, device=DEV)
    hs = torch.zeros(routes, inter // 32, dtype=torch.uint8, device=DEV)
    for e in sorted(set(ids[ids >= 0].tolist())):
        tok, k = torch.where(ids == e)
        r = tok * topk + k
        w1, w3, w2 = pool.dequant(e)
        a = xdq[tok]
        gate = (a @ w1.T).to(torch.bfloat16).float().clamp(max=10.0)
        up = (a @ w3.T).to(torch.bfloat16).float().clamp(-10.0, 10.0)
        h = (gate * torch.sigmoid(gate) * up * rw[tok, k, None]).to(torch.bfloat16).float()
        c, s, d = mx_quant(h)
        hc[r], hs[r] = c, s
        out[r] = d @ w2.T
    return out, hc, hs


# ----------------------------------------------------------------------------- pipeline
class Runner:
    """One compiled pipeline at a capacity with caller-owned scratch."""

    def __init__(self, role, capacity, pool, width=192):
        from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline, V41DraftSlicePipeline
        self.role, self.capacity, self.pool = role, capacity, pool
        inter = pool.inter
        self.draft = role == "coordinator"
        self.topk = 3 if role in ("coordinator", "dspark_tp2") else 6
        self.experts = 128 if role in ("coordinator", "dspark_tp2") else 384
        self.routes = capacity * self.topk
        self.width = width
        planes = (inter + width - 1) // width
        self.planes = planes
        self.wire = torch.zeros(capacity, WIRE, dtype=torch.uint8, device=DEV)
        self.source = torch.zeros(capacity, H, dtype=torch.bfloat16, device=DEV)
        self.ids = torch.full((self.routes,), -1, dtype=torch.int32, device=DEV)
        self.routing = torch.zeros(self.routes, dtype=torch.float32, device=DEV)
        e, r = self.experts, self.routes
        self.live = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.packed = torch.zeros(e * r, dtype=torch.int32, device=DEV)
        self.counts = torch.zeros(e, dtype=torch.int32, device=DEV)
        self.prefixes = torch.zeros(e, 2, dtype=torch.int32, device=DEV)
        self.metadata = torch.zeros(r, 19, dtype=torch.int32, device=DEV)
        self.grouped = torch.zeros(r, dtype=torch.float32, device=DEV)
        self.inverse = torch.zeros(r, dtype=torch.int32, device=DEV)
        self.partial = torch.zeros(planes, r, width // 4 + width // 32, dtype=torch.float32, device=DEV)
        self.output = torch.zeros(r, H, dtype=torch.float32, device=DEV)
        if self.draft:
            self.pipeline = V41DraftSlicePipeline(capacity, width, scale_codes=pool.codes)
        else:
            self.pipeline = V41SlicePipeline(capacity, width, experts=e, topk=self.topk,
                                             intermediate=inter, hidden=H, scale_codes=pool.codes)
        t0 = time.time()
        self.fn = cute.compile(self.pipeline, *self.args(), Int32(capacity), stream())
        self.compile_s = time.time() - t0

    def x_views(self):
        return self.wire[:, :H].view(torch.uint32), self.wire[:, H:]

    def args(self, ids=None, scratch=None):
        x, xs = self.x_views()
        sc = scratch or (self.metadata, self.grouped, self.inverse)
        tensors = [x, xs, *self.pool.tensors(), self.ids if ids is None else ids, self.routing, self.live,
                   self.packed, self.counts, self.prefixes, sc[0], sc[1], sc[2], self.partial, self.output]
        if self.draft:
            tensors.insert(0, self.source)
        return [from_dlpack(t, assumed_align=16) for t in tensors]

    def run(self, rows, ids=None):
        self.fn(*self.args(ids), Int32(rows), stream())

    def set_routes(self, ids, rw):
        m = ids.shape[0]
        self.ids.fill_(-1)
        self.ids[:m * self.topk] = ids.reshape(-1).to(torch.int32)
        self.routing.zero_()
        self.routing[:m * self.topk] = rw.reshape(-1)

    def kernel_h(self, rows):
        """Kernel FC1 output per original route: (codes [routes, I], scale bytes [routes, I/32])."""
        inter = self.pool.inter
        raw = self.partial.view(torch.uint8).reshape(-1)
        qa = raw[:self.routes * inter].view(self.routes, inter)
        qs = raw[self.routes * inter:self.routes * inter + self.routes * inter // 32].view(self.routes, inter // 32)
        live = rows * self.topk
        inv = self.inverse[:live].long()
        codes = torch.zeros(live, inter, dtype=torch.uint8, device=DEV)
        scales = torch.zeros(live, inter // 32, dtype=torch.uint8, device=DEV)
        ok = inv >= 0
        codes[ok] = qa[inv[ok]]
        scales[ok] = qs[inv[ok]]
        return codes, scales

    def groups(self, rows):
        live = rows * self.topk
        return max(1, min(live, self.experts + max(live - self.experts, 0) // 16))


def err(name, got, ref):
    d = (got.float() - ref.float()).abs()
    refmax = ref.float().abs().max().item()
    r = dict(name=name, max_abs=d.max().item(), ref_absmax=refmax,
             max_abs_over_refmax=d.max().item() / max(refmax, 1e-30),
             rel_l2=(d.norm() / ref.float().norm().clamp_min(1e-30)).item())
    print(f"  {name:28s} max|d|={r['max_abs']:.3e} max|d|/max|ref|={r['max_abs_over_refmax']:.3e} "
          f"relL2={r['rel_l2']:.3e}")
    return r


def correctness(runner, ids, seed, label):
    pool = runner.pool
    m, topk = ids.shape
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    x = make_x(m, gen)
    rw = torch.rand(m, topk, device=DEV, generator=gen) * 0.3 + 0.05
    runner.set_routes(ids.to(DEV), rw)
    runner.output.fill_(float("nan"))
    if runner.draft:
        runner.source[:m] = x
        runner.wire.zero_()
    else:
        runner.wire[:m] = wire_rows(x)
    runner.run(m)
    torch.cuda.synchronize()
    wire = runner.wire[:m].clone()
    res = dict(case=label, role=runner.role, capacity=runner.capacity, rows=m, topk=topk,
               I=pool.inter, experts=len(set(ids[ids >= 0].tolist())))
    if runner.draft:
        q, s, _ = mx_quant(x.float())
        res["input_quant_codes_equal"] = (wire[:, :H] == q).float().mean().item()
        res["input_quant_scales_equal"] = (wire[:, H:] == s).float().mean().item()
    out_ref, hc_ref, hs_ref = reference(pool, wire, ids.to(DEV), rw)
    hc, hs = runner.kernel_h(m)
    valid = (ids.reshape(-1) >= 0).to(DEV)
    res["h_codes_equal"] = (hc[valid] == hc_ref[valid]).float().mean().item()
    res["h_scales_equal"] = (hs[valid] == hs_ref[valid]).float().mean().item()
    got = runner.output[:m * topk]
    print(f"[{label}] role={runner.role} cap={runner.capacity} rows={m} I={pool.inter} "
          f"routes={int(valid.sum())} experts={res['experts']}: h codes equal {res['h_codes_equal'] * 100:.3f}%, "
          f"scales {res['h_scales_equal'] * 100:.3f}%"
          + (f"; input quant codes {res['input_quant_codes_equal'] * 100:.3f}% scales "
             f"{res['input_quant_scales_equal'] * 100:.3f}%" if runner.draft else ""))
    res["finite"] = bool(torch.isfinite(got).all())
    res["invalid_routes_zero"] = bool((got[~valid] == 0).all())
    res["route_vs_ref"] = err("route FP32 vs full ref", got[valid], out_ref[valid])
    # FC2 alone, fed with the kernel's own h.
    fc2_ref = torch.zeros_like(out_ref)
    for e in sorted(set(ids[ids >= 0].tolist())):
        tok, k = torch.where(ids.to(DEV) == e)
        r = tok * topk + k
        fc2_ref[r] = mx_dequant(hc[r], hs[r]) @ pool.dequant(e)[2].T
    res["route_vs_fc2_kernel_h"] = err("route FP32 vs FC2(kernel h)", got[valid], fc2_ref[valid])
    y = torch.zeros(m, H, device=DEV)
    y_ref = torch.zeros(m, H, device=DEV)
    for k in range(topk):
        rows_k = torch.arange(m, device=DEV) * topk + k
        y += got[rows_k].to(torch.bfloat16).float()
        y_ref += out_ref[rows_k].to(torch.bfloat16).float()
    res["token_bf16"] = err("token out BF16", y.to(torch.bfloat16), y_ref.to(torch.bfloat16))
    # Capacity-independent decode: the same rows give the same bits at another live count.
    return res


# ----------------------------------------------------------------------------- timing
class VT(ctypes.Structure):
    _fields_ = [("ref", ctypes.c_ulonglong), ("viol", ctypes.c_ulonglong)]


class NVML:
    POLICIES = {"power_cap": 0, "thermal": 1, "board_limit": 3, "reliability": 5}

    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            self.lib.nvmlInit_v2()
            self.h = ctypes.c_void_p()
            self.ok = self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) == 0
        except OSError:
            pass

    def snap(self):
        if not self.ok:
            return None
        e = ctypes.c_ulonglong()
        self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e))
        d = {"t": time.perf_counter(), "energy_mJ": e.value}
        for k, pol in self.POLICIES.items():
            v = VT()
            d[k] = v.viol if self.lib.nvmlDeviceGetViolationStatus(self.h, pol, ctypes.byref(v)) == 0 else None
        c = ctypes.c_uint()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(c))
        d["sm_MHz"] = c.value
        t = ctypes.c_uint()
        self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(t))
        d["temp_C"] = t.value
        return d

    @staticmethod
    def delta(a, b):
        if a is None or b is None:
            return {}
        dt = b["t"] - a["t"]
        out = {"window_ms": round(dt * 1e3, 1), "avg_W": round((b["energy_mJ"] - a["energy_mJ"]) / 1e3 / dt, 1),
               "sm_MHz_end": b["sm_MHz"], "temp_C_end": b["temp_C"]}
        for k in NVML.POLICIES:
            if a[k] is not None and b[k] is not None:
                out[k + "_ms"] = round((b[k] - a[k]) / 1e6, 2)
        return out


NV = None


def time_graph(fn, iters, reps=7, min_window_s=0.4):
    global NV
    if NV is None:
        NV = NVML()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(0)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for it in range(iters):
            fn(it)
    t0 = time.perf_counter()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 2
    reps = max(reps, int(math.ceil(min_window_s / max(per, 1e-6))))
    a0 = NV.snap()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / iters)
    thr = NVML.delta(a0, NV.snap())
    del graph
    return ts, thr


def decode_ids(n_experts, rows, topk, used_topk=None):
    """Token i routes to experts (k_used * i + k) % n_experts; extra top-k slots stay -1."""
    used = used_topk or topk
    ids = torch.full((rows, topk), -1, dtype=torch.int64)
    for i in range(rows):
        for k in range(used):
            ids[i, k] = (used * i + k) % n_experts
    return ids


def timing_case(runner, ids_local, label, iters=None, parts=True):
    """ids_local: [rows, topk] slot ids in 0..E-1 (or -1); each graph iteration maps them
    onto a different window of a random pool permutation."""
    pool = runner.pool
    rows, topk = ids_local.shape
    distinct = sorted(set(ids_local[ids_local >= 0].tolist()))
    n_exp = len(distinct)
    remap = {e: i for i, e in enumerate(distinct)}
    if iters is None:
        iters = max(8, min(200, (pool.n // max(n_exp, 1)) * 2))
    perm = torch.randperm(pool.n, generator=torch.Generator().manual_seed(17))
    ids_iter = []
    for it in range(iters):
        mapped = ids_local.clone()
        for e in distinct:
            mapped[ids_local == e] = int(perm[(it * n_exp + remap[e]) % pool.n])
        flat = torch.full((runner.routes,), -1, dtype=torch.int32)
        flat[:rows * topk] = mapped.reshape(-1).to(torch.int32)
        ids_iter.append(flat.to(DEV))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(5)
    rw = torch.rand(rows, topk, device=DEV, generator=gen) * 0.3 + 0.05
    runner.set_routes(ids_local.to(DEV).clamp_min(-1), rw)
    x = make_x(rows, gen)
    if runner.draft:
        runner.source[:rows] = x
    else:
        runner.wire[:rows] = wire_rows(x)
    args_iter = [runner.args(ids) for ids in ids_iter]
    ts, thr = time_graph(lambda it: runner.fn(*args_iter[it], Int32(rows), stream()), iters)
    nbytes = n_exp * pool.expert_bytes
    med = statistics.median(ts)
    out = dict(case=label, role=runner.role, I=pool.inter, capacity=runner.capacity, rows=rows,
               routes=int((ids_local >= 0).sum()), experts=n_exp, iters=iters, time_us=med,
               time_us_all=ts, weight_bytes=nbytes, GBps=nbytes / med / 1e3,
               pct_1008=nbytes / med / 1e3 / PEAK * 100, throttle=thr)
    if parts:
        # FC1 and FC2 alone, with each iteration's route plan prepared up front.
        pipe = runner.pipeline.pipeline if runner.draft else runner.pipeline
        groups = runner.groups(rows)
        plan_out = []
        for ids in ids_iter:
            md = torch.zeros_like(runner.metadata)
            gw = torch.zeros_like(runner.grouped)
            inv = torch.zeros_like(runner.inverse)
            runner.live.fill_(rows)
            plan_out.append((md, gw, inv))
        if not hasattr(runner, "plan_fn"):
            pa = [from_dlpack(t, assumed_align=16) for t in
                  (runner.ids, runner.routing, runner.live, runner.packed, runner.counts, runner.prefixes,
                   runner.metadata, runner.grouped, runner.inverse)]
            runner.plan_fn = cute.compile(pipe.plan, *pa, stream())
            x, xs = runner.x_views()
            fa = [from_dlpack(t, assumed_align=16) for t in
                  (x, xs, runner.pool.tensors()[0], runner.pool.tensors()[1], runner.grouped,
                   runner.partial, runner.metadata)]
            runner.fc1_fn = cute.compile(pipe.fc1, *fa, Int32(groups), stream())
            sa = [from_dlpack(t, assumed_align=16) for t in
                  (runner.pool.tensors()[2], runner.pool.tensors()[3], runner.partial, runner.output,
                   runner.metadata, runner.inverse, runner.live)]
            runner.fc2_fn = cute.compile(pipe.fc2, *sa, Int32(groups), stream())
        if runner.draft:
            # Quantize the source once into the wire rows the FC1 timing reads.
            runner.run(rows, ids_iter[0])
        plan_args, fc1_args, fc2_args = [], [], []
        x, xs = runner.x_views()
        for ids, (md, gw, inv) in zip(ids_iter, plan_out):
            pa = [from_dlpack(t, assumed_align=16) for t in
                  (ids, runner.routing, runner.live, runner.packed, runner.counts, runner.prefixes, md, gw, inv)]
            runner.plan_fn(*pa, stream())
            plan_args.append(pa)
            fc1_args.append([from_dlpack(t, assumed_align=16) for t in
                             (x, xs, runner.pool.tensors()[0], runner.pool.tensors()[1], gw,
                              runner.partial, md)])
            fc2_args.append([from_dlpack(t, assumed_align=16) for t in
                             (runner.pool.tensors()[2], runner.pool.tensors()[3], runner.partial,
                              runner.output, md, inv, runner.live)])
        torch.cuda.synchronize()
        g = Int32(groups)

        def both(it):
            st = stream()
            runner.fc1_fn(*fc1_args[it], g, st)
            runner.fc2_fn(*fc2_args[it], g, st)
        for part, fn in (("fc1+fc2", both),
                         ("fc1", lambda it: runner.fc1_fn(*fc1_args[it], g, stream())),
                         ("fc2", lambda it: runner.fc2_fn(*fc2_args[it], g, stream())),
                         ("plan", lambda it: runner.plan_fn(*plan_args[it], stream()))):
            t, th = time_graph(fn, iters)
            out[part + "_us"] = statistics.median(t)
            if part == "fc1+fc2":
                out["fc1+fc2_throttle"] = th
        w = pool.sizes
        out["fc1+fc2_GBps"] = nbytes / out["fc1+fc2_us"] / 1e3
        out["fc1+fc2_pct_1008"] = out["fc1+fc2_GBps"] / PEAK * 100
        out["fc1_GBps"] = n_exp * (w[0] + w[1]) / out["fc1_us"] / 1e3
        out["fc2_GBps"] = n_exp * (w[2] + w[3]) / out["fc2_us"] / 1e3
    thr = out.get("fc1+fc2_throttle") or out["throttle"]
    print(f"  {label:30s} I={pool.inter} rows={rows:3d} experts={n_exp:3d} pipeline {med:8.1f} us "
          f"{out['GBps']:6.1f} GB/s ({out['pct_1008']:5.1f}%)"
          + (f" | fc1+fc2 {out['fc1+fc2_us']:8.1f} us {out['fc1+fc2_GBps']:6.1f} GB/s "
             f"({out['fc1+fc2_pct_1008']:5.1f}%) fc1 {out['fc1_us']:.1f} us ({out['fc1_GBps']:.0f} GB/s) "
             f"fc2 {out['fc2_us']:.1f} us ({out['fc2_GBps']:.0f} GB/s) plan {out['plan_us']:.1f} us" if parts else "")
          + f" [{thr.get('avg_W', 0):.0f} W, {thr.get('sm_MHz_end', 0)} MHz, {thr.get('temp_C_end', 0)} C, "
            f"pcap {thr.get('power_cap_ms', 0):.0f} ms, therm {thr.get('thermal_ms', 0):.0f} ms]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="correct,draft")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool", type=int, default=0, help="pool experts (default: role expert count)")
    ap.add_argument("--only", default="", help="comma list of timing-case label substrings")
    args = ap.parse_args()
    only = [o for o in args.only.split(",") if o]

    def wanted(label):
        return not only or any(o in label for o in only)
    what = set(args.what.split(","))
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} sm_{props.major}{props.minor} SMs={props.multi_processor_count} "
          f"torch={torch.__version__} cutlass-dsl={getattr(cutlass, '__version__', '?')}", flush=True)
    results = dict(device=props.name, sms=props.multi_processor_count, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   what=sorted(what), tuning=os.environ.get("B12X_SM89_V41_EXPERTS", "default"),
                   correct=[], timing=[])
    t0 = time.time()
    if what & {"correct", "draft"}:
        pool = Pool(args.pool or 128, 2304, logical=32 if "correct" in what else 0, seed=1)
        print(f"draft pool: {pool.n} experts x {pool.expert_bytes / 1e6:.3f} MB "
              f"({[round(s / 1e6, 3) for s in pool.sizes]}) built in {time.time() - t0:.1f}s", flush=True)
        runners = {}

        def runner(cap):
            if cap not in runners:
                runners[cap] = Runner("coordinator", cap, pool)
                print(f"  compiled coordinator m{cap} in {runners[cap].compile_s:.1f}s", flush=True)
            return runners[cap]
        if "correct" in what:
            g = torch.Generator().manual_seed(21)
            cases = [("draft m1", 1, torch.tensor([[0, 1, 2]])),
                     ("draft m7", 16, torch.stack([torch.randperm(32, generator=g)[:3] for _ in range(7)])),
                     ("draft m16 shared", 16, torch.arange(3).expand(16, 3).clone()),
                     ("draft m40", 40, torch.stack([torch.randperm(32, generator=g)[:3] for _ in range(40)]))]
            for label, cap, ids in cases:
                results["correct"].append(correctness(runner(cap), ids, seed=22 + cap, label=label))
        if "draft" in what:
            print("[draft timing: coordinator I=2304, 128 experts, top-3]", flush=True)
            g = torch.Generator().manual_seed(3)
            for label, cap, rows in (("draft m1", 1, 1), ("draft m7", 16, 7), ("draft m16", 16, 16),
                                     ("draft m80", 80, 80)):
                ids = torch.stack([torch.randperm(128, generator=g)[:3] for _ in range(rows)])
                if wanted(label):
                    results["timing"].append(timing_case(runner(cap), ids, label))
        del runners, pool
        torch.cuda.empty_cache()
    if "dspark_tp2" in what:
        pool = Pool(args.pool or 128, 1152, logical=0, seed=3)
        print(f"dspark_tp2 pool: {pool.n} experts x {pool.expert_bytes / 1e6:.3f} MB", flush=True)
        g = torch.Generator().manual_seed(3)
        runners = {}
        for label, cap, rows in (("dspark_tp2 m1", 1, 1), ("dspark_tp2 m7", 16, 7), ("dspark_tp2 m16", 16, 16),
                                 ("dspark_tp2 m80", 80, 80)):
            ids = torch.stack([torch.randperm(128, generator=g)[:3] for _ in range(rows)])
            if cap not in runners:
                runners[cap] = Runner("dspark_tp2", cap, pool)
            if wanted(label):
                results["timing"].append(timing_case(runners[cap], ids, label))
        del runners, pool
        torch.cuda.empty_cache()
    for role, inter in (("rtx_tp2", 1152), ("rtx_backbone", 2304)):
        key = "tp2" if inter == 1152 else "whole"
        if not what & {key, key + "_correct"}:
            continue
        pool = Pool(args.pool or 384, inter, logical=24 if key + "_correct" in what else 0, seed=2)
        print(f"{role} pool: {pool.n} experts x {pool.expert_bytes / 1e6:.3f} MB", flush=True)
        run16 = Runner(role, 16, pool)
        print(f"  compiled {role} m16 in {run16.compile_s:.1f}s", flush=True)
        if key + "_correct" in what:
            results["correct"].append(correctness(run16, decode_ids(24, 6, 6), seed=23, label=f"{key} wave"))
            results["correct"].append(correctness(run16, decode_ids(8, 16, 6), seed=24, label=f"{key} m16x8"))
        if key in what:
            print(f"[{key} decode timing: I={inter}, top-6]", flush=True)
            for label, ids in ((f"{key} 1 expert", decode_ids(1, 2, 6, used_topk=1)),
                               (f"{key} 8 experts", decode_ids(8, 2, 6)),
                               (f"{key} wave 24 experts", decode_ids(24, 6, 6)),
                               (f"{key} 48 experts", decode_ids(48, 12, 6))):
                if wanted(label):
                    results["timing"].append(timing_case(run16, ids, label))
        del run16, pool
        torch.cuda.empty_cache()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
