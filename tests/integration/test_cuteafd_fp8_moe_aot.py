"""cuteafd exact FP8 routed-expert programs (``fp8_moe``) vs a torch reference.

Random E4M3 expert weights with FP32 128x128 block scales at the MiMo V2
Flash (H 4096) and GLM 5.3 (H 6144) geometries, full width (TP1, the RTX
local/MTP layers) and the TP4 Spark slice (I 512); FP8 K32 wire or BF16
input rows; decode (grouped GEMV), prefill (grouped TMA GEMM), stream
(expert-stationary streaming GEMMs over FP8 wire rows; BF16 rows quantized
to wire rows first) and auto routes, in each ``prefill`` form (auto, w8a8, w8a16). The
reference is the checkpoint semantics: ``F.linear`` of BF16 activations
with ``bf16(w * s)`` weights, BF16 gate/up, ``bf16(silu(g)) *
u``, BF16 down output, FP32 weighted route sum; ``exact=True`` takes the
weights ``w * s`` unrounded (FP64), which the W8A8 gate/up computes.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

COS = 0.9999
_WEIGHTS: dict = {}
_PROGRAMS: dict = {}


def _geometry(name, tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES[name].with_tp(tp)


def _weights(g):
    key = (g.name, g.tp)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        gen = torch.Generator(device="cuda").manual_seed(0)
        h, i, e = g.hidden, g.slice, g.experts

        def fp8(shape):
            return (torch.randn(shape, device="cuda", generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)

        def scales(shape):
            return torch.rand(shape, device="cuda", generator=gen) * 2e-4 + 1e-4

        if g.weights == "bf16":
            def bf16(shape):
                return (torch.randn(shape, device="cuda", generator=gen) * 0.02).bfloat16()

            unused = torch.empty(0, device="cuda")
            _WEIGHTS[key] = (bf16((e, i, h)), unused, bf16((e, i, h)), unused,
                             bf16((e, h, i)), unused)
        else:
            _WEIGHTS[key] = (fp8((e, i, h)), scales((e, i // 128, h // 128)), fp8((e, i, h)),
                             scales((e, i // 128, h // 128)), fp8((e, h, i)), scales((e, h // 128, i // 128)))
    return _WEIGHTS[key]


def _program(g, route, capacity, wire, prefill="auto"):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot

    key = (g, route, capacity, wire, prefill)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route=route, max_rows=capacity, wire=wire, prefill=prefill)
    return _PROGRAMS[key]


@pytest.mark.parametrize("route,wire,prefill,segmented,weight_kind", [
    ("decode", True, "auto", False, "fp8"),
    ("decode", False, "auto", False, "fp8"),
    ("prefill", False, "w8a16", False, "fp8"),
    ("stream", True, "w8a8", False, "fp8"),
    ("stream", False, "w8a8", False, "fp8"),
    ("stream", True, "w8a16", True, "fp8"),
    ("auto", True, "auto", False, "fp8"),
    ("stream", True, "auto", False, "mxfp4"),
    ("stream", True, "auto", True, "mxfp4"),
    ("stream", False, "auto", False, "nvfp4"),
    ("decode", False, "auto", False, "bf16"),
    ("prefill", False, "auto", False, "bf16"),
])
def test_core_reply_matches_opaque_bytes_and_graph(route, wire, prefill, segmented, weight_kind):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import (
        Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_reply_layout)
    from b12x.integration.cuteafd.expert_reply import compile_fp8_reply_aot
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot
    from .test_cuteafd_nvfp4_moe_aot import _weights as nvfp4_weights
    from .test_cuteafd_mxfp4_moe_aot import _weights as mxfp4_weights

    h, capacity = 1024, 129
    g = Fp8MoeGeometry("reply_" + weight_kind, hidden=h, experts=4, top_k=2,
                      intermediate=512, weights=weight_kind,
                      activations="a4" if weight_kind == "nvfp4" else "a16")
    operands = (nvfp4_weights(g, g.slice) if weight_kind == "nvfp4" else
                mxfp4_weights(g, g.slice) if weight_kind == "mxfp4" else _weights(g))
    options = dict(route=route, max_rows=capacity, wire=wire, prefill=prefill,
                   row_addresses=segmented)
    opaque = compile_fp8_moe_aot(g, **options)
    core = compile_fp8_moe_aot(g, core_only=True, **options)
    assert core.scratch_bytes(capacity) == opaque.scratch_bytes(capacity)
    epilogues = [compile_fp8_reply_aot(hidden=h, top_k=g.top_k, format=f,
                                     row_addresses=segmented) for f in range(3)]
    quantize, decode = compile_expert_reply_aot(quantize=True), compile_expert_reply_aot()
    torch.manual_seed(1166)
    x = torch.randn(capacity, h, device="cuda").bfloat16()
    input_rows = wire_rows(x)[0] if wire else x
    source = input_rows
    if segmented:
        # Keep owners live and use the same row-address input ABI as serving.
        source = torch.tensor([input_rows[r].data_ptr() for r in range(capacity)],
                              device="cuda", dtype=torch.int64)
    ids = torch.rand(capacity, g.experts, device="cuda").topk(g.top_k, -1).indices.int()
    ids[-1] = -1
    weights = torch.rand(capacity, g.top_k, device="cuda") / g.top_k
    original_weights = weights.clone()
    baseline = torch.empty_like(x)
    baseline_ptrs = torch.tensor([baseline[r].data_ptr() for r in range(capacity)],
                                device="cuda", dtype=torch.int64) if segmented else baseline
    scratch = torch.empty(core.scratch_bytes(capacity)["scratch"], device="cuda", dtype=torch.uint8)
    unused = torch.full_like(baseline, -123)

    for rows in (1, 17, 129):
        layout = fp8_moe_reply_layout(g, rows=rows, **options)
        positions = scratch[layout["pair_pos_offset"]:].view(torch.int32)
        down = scratch[layout["y_offset"]:].view(torch.bfloat16)
        for format, epilogue in enumerate(epilogues):
            row_bytes = h * 2 if format == 0 else h + h // 32 if format == 1 else h // 2 + h // 16
            canary = torch.full((capacity, row_bytes + 512), 0xA5, device="cuda", dtype=torch.uint8)
            if segmented:
                out = torch.tensor([canary[r, 256:].data_ptr() for r in range(capacity)],
                                   device="cuda", dtype=torch.int64)
            else:
                canary = torch.full((capacity * row_bytes + 512,), 0xA5, device="cuda", dtype=torch.uint8)
                out = canary[256:-256].view(capacity, row_bytes)
            expected = torch.empty((capacity, row_bytes), device="cuda", dtype=torch.uint8)
            restored, reference_restored = torch.empty_like(x), torch.empty_like(x)

            def reference():
                opaque.launch(source, ids, weights, *operands, baseline_ptrs, scratch, scalars=(rows,))
                if format:
                    quantize.launch(baseline, expected, None, scalars=(rows, h, format))
                else:
                    expected[:rows].copy_(baseline[:rows].view(torch.uint8))

            def fused():
                core.launch(source, ids, weights, *operands, unused, scratch, scalars=(rows,))
                epilogue.launch(down, positions, weights, out, scalars=(rows,))

            weights.copy_(original_weights)
            assert weights[:rows].count_nonzero().item() > 0
            reference()
            fused()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with kernel_resolution_guard("FP8 core+reply byte oracle"):
                with torch.cuda.graph(graph):
                    fused()
                for factor in (1., 0.5, -1.):
                    weights.copy_(original_weights * factor)
                    assert weights[:rows].count_nonzero().item() > 0
                    reference()
                    canary.fill_(0xA5)
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = canary[:rows, 256:256 + row_bytes] if segmented else out[:rows]
                    assert torch.equal(actual, expected[:rows]), (route, rows, format)
                    if format and factor == 1.:
                        corrupted = actual.clone()
                        corrupted[0, 0] ^= 1
                        with pytest.raises(AssertionError, match="compressed byte oracle"):
                            assert torch.equal(corrupted, expected[:rows]), "compressed byte oracle"
                    assert (unused == -123).all(), "core-only wrote opaque output"
                    if segmented:
                        assert (canary[:, :256] == 0xA5).all() and (canary[:, -256:] == 0xA5).all()
                    else:
                        assert (canary[:256] == 0xA5).all() and (canary[-256:] == 0xA5).all()
                    if format:
                        decode.launch(actual.contiguous(), restored, None, scalars=(rows, h, format))
                        decode.launch(expected, reference_restored, None, scalars=(rows, h, format))
                        assert torch.equal(restored[:rows].view(torch.int16), reference_restored[:rows].view(torch.int16))


def _quantizes_input(g, route, rows, wire, prefill):
    """Whether a BF16-input program runs these rows through its wire quantizer."""
    from b12x.integration.cuteafd.fp8_moe import auto_large_rows

    if wire:
        return False
    return route == "stream" or (route == "auto" and prefill == "w8a8"
                                 and rows > auto_large_rows(False, g.weights, prefill))


def wire_rows(x: torch.Tensor):
    """FP8 K32 wire rows of BF16 ``x`` and their exact BF16 value."""
    rows, h = x.shape
    groups = x.float().view(rows, h // 32, 32)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-127, 127)
    q = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
    wire = torch.cat([q.view(rows, h).view(torch.uint8), (exponent + 127).to(torch.uint8)], 1).contiguous()
    return wire, (q.float() * torch.exp2(exponent)[..., None]).view(rows, h).bfloat16()


def reference(x, ids, weights, w1, s1, w3, s3, w2, s2, limit=0.0, exact=False,
              activation="silu_bf16_product", native_bf16=False):
    if activation == "silu_post_clamp_fp32_product":
        return post_silu_reference(x, ids, weights, w1, s1, w3, s3, w2, s2,
                                   limit=limit, native_bf16=native_bf16)
    def dequant(w, s):
        d = w.double() * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
        return d if exact else d.float().bfloat16()

    if exact:
        x = x.double()
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float64)
        for expert in ids.unique().tolist():
            rows, slots = torch.where(ids == expert)
            rnd = lambda t: t.float().bfloat16().double()  # noqa: E731
            gate = rnd(x[rows] @ dequant(w1[expert], s1[expert]).T)
            up = rnd(x[rows] @ dequant(w3[expert], s3[expert]).T)
            if limit > 0:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            act = rnd(rnd(torch.nn.functional.silu(gate)) * up)
            y = rnd(act @ dequant(w2[expert], s2[expert]).T)
            out.index_add_(0, rows, y * weights[rows, slots][:, None].double())
        return out.float().bfloat16()
    out = torch.zeros(x.shape, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = x[rows] @ dequant(w1[expert], s1[expert]).T
        up = x[rows] @ dequant(w3[expert], s3[expert]).T
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        y = (torch.nn.functional.silu(gate) * up) @ dequant(w2[expert], s2[expert]).T
        out.index_add_(0, rows, y.float() * weights[rows, slots][:, None])
    return out.bfloat16()


def post_silu_reference(x, ids, weights, w1, s1, w3, s3, w2, s2, *, limit, native_bf16):
    """Independent CPU FP64 products; explicit BF16 stores and slot-order FP32 sum."""
    x, ids, weights = x.cpu(), ids.cpu(), weights.cpu()
    routes = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.float32)

    def operand(w, s, expert):
        value = w[expert].cpu().float()
        if not native_bf16:
            value *= s[expert].cpu().repeat_interleave(128, 0).repeat_interleave(128, 1)
        return value.bfloat16().double()

    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = (x[rows].double() @ operand(w1, s1, expert).T).float().bfloat16().float()
        up = (x[rows].double() @ operand(w3, s3, expert).T).float().bfloat16().float()
        silu = torch.nn.functional.silu(gate).clamp(max=limit)
        act = (silu * up.clamp(-limit, limit)).bfloat16()
        y = (act.double() @ operand(w2, s2, expert).T).float().bfloat16()
        routes[rows, slots] = y.float()
    out = torch.zeros(x.shape, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        out += routes[:, slot] * weights[:, slot, None]
    return out.bfloat16()


def _run(g, route, capacity, rows, wire=True, seed=1, hot=0, prefill="auto", exact=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = _weights(g)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    source, x_exact = wire_rows(x) if wire else (x, x)
    if _quantizes_input(g, route, rows, wire, prefill):
        x_exact = wire_rows(x)[1]
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0  # every row routes to the first `hot` experts (multi-chunk groups)
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    from b12x.integration.cuteafd.fp8_moe import fp8_moe_scratch_bytes

    program = _program(g, route, capacity, wire, prefill)
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity, wire, prefill), dtype=torch.uint8,
                          device="cuda")
    program.launch(source, ids, weights, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    expected = reference(x_exact, ids, weights, *w, limit=g.swiglu_limit, exact=exact,
                         activation=g.activation, native_bf16=g.weights == "bf16").to(out.device)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    return c, worst


def _assert_segment_row_invariance(g, operands, hot, ring_frames=False):
    """Ordinary STREAM bytes survive reordered mapped rows, including real ring offsets."""
    from itertools import permutations

    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_segment_scratch_bytes
    from b12x.sequence._shared.disk_table import MappedHostAllocation

    capacity, maximum = 2752, 16384
    row_counts = (2752, 2752, 2487) if ring_frames else (capacity,) * 3
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(844)
    direct = _program(g, "stream", capacity, True)
    indirect = compile_fp8_moe_aot(g, route="stream", max_rows=maximum, row_addresses=True)
    assert indirect.geometry["row_addresses"] and indirect.geometry["route"] == "stream"
    direct_scratch = torch.empty(direct.scratch_bytes(capacity)["scratch"], dtype=torch.uint8, device=device)
    scratch = torch.empty(fp8_moe_segment_scratch_bytes(g, sum(row_counts), maximum),
                          dtype=torch.uint8, device=device)
    allocations, segments = [], []
    sentinel = 173
    wire_stride = g.hidden + g.hidden // 32
    # Match the transport's page-aligned request/reply slot strides. The GPU only
    # reads payloads; metadata remains sentinel-filled to catch accidental reads/writes.
    request_stride = (96 + capacity * (40 + 12 * g.top_k + wire_stride) + 4095) & ~4095
    response_stride = (96 + capacity * (g.hidden * 2 + 4) + 4095) & ~4095
    try:
        for slot, rows in enumerate(row_counts):
            x = torch.randn(rows, g.hidden, device=device, generator=gen).bfloat16()
            source, _ = wire_rows(x)
            assert source.shape[1] == wire_stride
            scores = torch.rand(rows, g.experts, device=device, generator=gen)
            if hot:
                scores[:, :hot] += 2.0
            ids = scores.topk(g.top_k, -1).indices.int().contiguous()
            weights = torch.rand(rows, g.top_k, device=device, generator=gen)
            baseline = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device=device)
            direct.launch(source, ids, weights, *operands, baseline, direct_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.isfinite(baseline).all() and baseline.count_nonzero() > 0
            if ring_frames:
                raw_prefix = 96 + rows * 40 + rows * g.top_k * 12
                prefix = (raw_prefix + 15) & ~15
                if rows == 2487:
                    assert raw_prefix % 16 == 8 and prefix - raw_prefix == 8
                input_offset = slot * request_stride + prefix
                output_offset = slot * response_stride + 96
                input_bytes = (slot + 1) * request_stride
                output_bytes = (slot + 1) * response_stride
            else:
                input_offset, output_offset = wire_stride, g.hidden * 2
                input_bytes, output_bytes = (rows + 2) * wire_stride, (rows + 2) * g.hidden * 2
            source_allocation = MappedHostAllocation((input_bytes,), torch.uint8, device)
            allocations.append(source_allocation)
            response = MappedHostAllocation((output_bytes,), torch.uint8, device)
            allocations.append(response)
            source_allocation.host_view.fill_(sentinel)
            source_allocation.host_view[input_offset:input_offset + rows * wire_stride].copy_(source.cpu().flatten())
            input_snapshot = source_allocation.host_view.clone()
            assert (source_allocation.device_view.data_ptr() + input_offset) % 16 == 0
            assert (response.device_view.data_ptr() + output_offset) % 16 == 0
            segments.append((source_allocation, response, ids, weights, baseline.cpu(), input_snapshot,
                             input_offset, output_offset, rows))

        for count in (2, 3):
            for order in permutations(range(3), count):
                for _, response, *_ in segments:
                    response.host_view.fill_(sentinel)
                input_addresses, output_addresses, routed_ids, routed_weights = [], [], [], []
                for position, segment in enumerate(order):
                    source_allocation, response, ids, weights, _, _, input_offset, output_offset, rows = segments[segment]
                    # Change expert-sorted route order without changing a row's slot sum.
                    permutation = (torch.arange(rows, device=device) if position == 0 else
                                   torch.arange(rows - 1, -1, -1, device=device) if position == 1 else
                                   torch.randperm(rows, device=device, generator=gen))
                    input_addresses.append(source_allocation.device_view.data_ptr() + input_offset
                                           + permutation.long() * wire_stride)
                    output_addresses.append(response.device_view.data_ptr() + output_offset
                                            + permutation.long() * g.hidden * 2)
                    routed_ids.append(ids[permutation])
                    routed_weights.append(weights[permutation])
                x_addresses = torch.cat(input_addresses)
                y_addresses = torch.cat(output_addresses)
                ids = torch.cat(routed_ids)
                weights = torch.cat(routed_weights)
                live_rows = sum(row_counts[segment] for segment in order)
                indirect.launch(x_addresses, ids, weights, *operands, y_addresses, scratch,
                                scalars=(live_rows,))
                torch.cuda.synchronize()
                for segment, (source_allocation, response, _, _, baseline, snapshot, _, offset, rows) in enumerate(segments):
                    assert torch.equal(source_allocation.host_view, snapshot), "input/guards modified"
                    raw = response.host_view
                    end = offset + rows * g.hidden * 2
                    assert (raw[:offset] == sentinel).all() and (raw[end:] == sentinel).all(), "response guard overwritten"
                    if segment in order:
                        actual = raw[offset:end].view(torch.bfloat16).reshape(rows, g.hidden)
                        assert torch.isfinite(actual).all() and actual.count_nonzero() > 0
                        different = int((actual.view(torch.int16) != baseline.view(torch.int16)).sum())
                        assert different == 0, (f"{g.name} tp{g.tp} hot={hot} order={order} segment={segment}: "
                                                f"{different} BF16 words differ")
                    else:
                        assert (raw == sentinel).all(), "nonmember response modified"
                print(f"row_addresses {g.name} tp{g.tp} hot={hot} ring_frames={ring_frames} "
                      f"order={order} rows={live_rows}: byte-identical")
    finally:
        for allocation in reversed(allocations):
            allocation.close()


@pytest.mark.parametrize("ring_frames", [False, True])
@pytest.mark.parametrize("hot", [0, 8])
def test_fp8_moe_stream_segment_row_invariance(hot, ring_frames):
    g = _geometry("mimo", 2)
    _assert_segment_row_invariance(g, _weights(g), hot, ring_frames)


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (geometry, tp, route, capacity, rows)
    ("mimo", 4, "decode", 1, 1), ("mimo", 4, "decode", 16, 16), ("mimo", 4, "decode", 80, 80),
    ("mimo", 4, "auto", 1024, 1024), ("mimo", 4, "auto", 4096, 4096), ("mimo", 4, "prefill", 4096, 80),
    ("mimo", 1, "decode", 1, 1), ("mimo", 1, "decode", 16, 16), ("mimo", 1, "auto", 80, 80),
    ("mimo", 1, "auto", 1024, 1024), ("mimo", 1, "auto", 4096, 4096),
    ("glm", 1, "decode", 1, 1), ("glm", 1, "auto", 80, 80), ("glm", 1, "auto", 4096, 4096),
    ("glm", 4, "auto", 256, 256),
    ("mimo", 4, "stream", 16, 5), ("mimo", 4, "stream", 256, 200), ("mimo", 4, "stream", 4096, 4096),
    ("mimo", 2, "stream", 1024, 1024), ("glm", 4, "stream", 4096, 3000), ("glm", 2, "stream", 256, 256),
    ("glmf", 4, "stream", 4096, 4096), ("glmf", 4, "auto", 1024, 900),
    # TP6 of 2048: 3-block (384) slices, the stored width of every rank.
    ("mimo", 6, "decode", 1, 1), ("glm", 6, "decode", 16, 16), ("glmf", 6, "auto", 80, 80),
    ("mimo", 6, "stream", 1024, 1024), ("glm", 6, "stream", 4096, 4096),
]


@pytest.mark.parametrize("name,tp,route,capacity,rows", CASES)
def test_fp8_moe(name, tp, route, capacity, rows):
    g = _geometry(name, tp)
    c, worst = _run(g, route, capacity, rows)
    print(f"fp8_moe {name} tp{tp} {route} m{capacity} rows={rows}: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


def test_fp8_moe_bf16_input():
    g = _geometry("mimo", 4)
    c, worst = _run(g, "auto", 80, 33, wire=False)
    print(f"fp8_moe mimo tp4 bf16 input rows=33: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


@pytest.mark.parametrize("name,tp,route,capacity,rows,prefill", [
    ("mimo", 1, "auto", 4096, 3000, "w8a8"), ("mimo", 1, "auto", 4096, 3000, "auto"),
    ("mimo", 1, "auto", 1024, 700, "w8a8"), ("glmf", 1, "stream", 1024, 1000, "auto"),
    ("mimo", 4, "stream", 256, 200, "auto"),
])
def test_fp8_moe_bf16_input_large(name, tp, route, capacity, rows, prefill):
    """BF16 rows at large row counts: W8A8 programs quantize them to wire rows
    (the reference takes the same quantized rows), "auto" ones keep them exact."""
    g = _geometry(name, tp)
    c, worst = _run(g, route, capacity, rows, wire=False, prefill=prefill)
    print(f"fp8_moe {name} tp{tp} bf16 input {route} m{capacity} rows={rows} {prefill}: cosine {c:.7f} "
          f"worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


@pytest.mark.parametrize("name,tp,capacity,rows", [("mimo", 4, 4096, 4096), ("glm", 2, 1024, 1000),
                                                   ("glmf", 6, 1024, 1024)])
def test_fp8_moe_stream_w8a8_vs_exact_weights(name, tp, capacity, rows):
    """The W8A8 stream gate/up computes x . (w * s) with the weights unrounded:
    against the exact-weight (FP64) reference it is at least as close as the
    W8A16 form, which rounds them to bf16(w * s)."""
    g = _geometry(name, tp)
    a8 = _run(g, "stream", capacity, rows, exact=True)
    a16 = _run(g, "stream", capacity, rows, exact=True, prefill="w8a16")
    oracle_a8 = _run(g, "stream", capacity, rows)
    print(f"fp8_moe {name} tp{tp} stream rows={rows} vs exact weights: W8A8 {a8[0]:.8f} W8A16 {a16[0]:.8f}; "
          f"W8A8 vs bf16(w*s) reference {oracle_a8[0]:.8f}")
    assert a8[0] >= 0.99999 and a8[0] >= a16[0] - 1e-7 and oracle_a8[0] >= COS


def test_fp8_moe_clamp():
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    g = Fp8MoeGeometry("clamped", hidden=4096, experts=256, top_k=8, intermediate=2048, tp=4, swiglu_limit=0.5)
    c, worst = _run(g, "auto", 16, 16)
    print(f"fp8_moe clamp 0.5 rows=16: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


def test_fp8_moe_stream_hot_experts():
    g = _geometry("mimo", 4)
    c, worst = _run(g, "stream", 1024, 1000, hot=6)
    print(f"fp8_moe mimo tp4 stream hot experts rows=1000: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS and worst >= 0.999


def test_fp8_moe_stream_clamp():
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    g = Fp8MoeGeometry("clamped", hidden=4096, experts=256, top_k=8, intermediate=2048, tp=4, swiglu_limit=0.5)
    c, worst = _run(g, "stream", 256, 256)
    print(f"fp8_moe clamp 0.5 stream rows=256: cosine {c:.7f} worst row {worst:.6f}")
    assert c >= COS


@pytest.mark.parametrize("route,capacity,rows", [("decode", 1, 1), ("decode", 80, 80), ("stream", 1024, 1024)])
def test_fp8_moe_tp6_zero_padding_is_exact(route, capacity, rows):
    """A TP6 rank owning two of its three stored 128-blocks (ranks 4 and 5 of
    2048) keeps zero gate/up rows and zero down columns in the third: its
    output equals the reference over the unpadded 256-wide slice."""
    g = _geometry("glm", 6)
    w1, s1, w3, s3, w2, s2 = (t.clone() for t in _weights(g))
    for w in (w1, w3):
        w[:, 256:].zero_()
    w2[:, :, 256:].zero_()
    padded = (w1, s1, w3, s3, w2, s2)
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    source, x_exact = wire_rows(x)
    ids = torch.rand(rows, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand(rows, g.top_k, device="cuda", generator=gen).contiguous()
    from b12x.integration.cuteafd.fp8_moe import fp8_moe_scratch_bytes

    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, route, capacity), dtype=torch.uint8, device="cuda")
    _program(g, route, capacity, True).launch(source, ids, weights, *padded, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    narrow = (w1[:, :256].contiguous(), s1[:, :2].contiguous(), w3[:, :256].contiguous(), s3[:, :2].contiguous(),
              w2[:, :, :256].contiguous(), s2[:, :, :2].contiguous())
    expected = reference(x_exact, ids, weights, *narrow, limit=g.swiglu_limit)
    a, b = out.float(), expected.float()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    print(f"fp8_moe glm tp6 padded {route} rows={rows}: cosine {c:.7f}")
    assert c >= COS


@pytest.mark.parametrize("tp", [4, 6])
def test_step5_mixed_native_groups_post_silu_and_graph(tp, monkeypatch):
    """Keep all native format owners live and alternate them within one graph."""
    from b12x.integration.cuteafd import fp8_moe
    from .test_cuteafd_nvfp4_moe_aot import _weights as nvfp4_weights, reference as nvfp4_reference

    groups = []
    for name in ("step5", "step5_bf16", "step5_nvfp4"):
        g = _geometry(name, tp)
        operands = nvfp4_weights(g, g.slice) if g.weights == "nvfp4" else _weights(g)
        program = _program(g, "auto", 64, False)
        scratch = torch.empty(fp8_moe.fp8_moe_scratch_bytes(g, "auto", 64, False),
                              dtype=torch.uint8, device="cuda")
        out = torch.empty(64, g.hidden, dtype=torch.bfloat16, device="cuda")
        groups.append((g, operands, program, scratch, out))

    def no_compile(*args, **kwargs):
        raise AssertionError("live rows must reuse planned kernels")

    monkeypatch.setattr(fp8_moe, "compile_fp8_moe_aot", no_compile)
    gen = torch.Generator(device="cuda").manual_seed(274)
    # Larger inputs exercise both post-SiLU and symmetric up clamps.
    x = (torch.randn(64, groups[0][0].hidden, device="cuda", generator=gen) * 8).bfloat16()
    ids = ((torch.arange(64, device="cuda")[:, None] + torch.arange(8, device="cuda")) % 8).int()
    weights = torch.rand(64, 8, device="cuda", generator=gen)
    weights /= weights.sum(-1, keepdim=True)

    for rows in (1, 16, 64):
        def launch():
            for _, operands, program, scratch, out in groups:
                program.launch(x, ids, weights, *operands, out, scratch, scalars=(rows,))

        launch()
        torch.cuda.synchronize()
        expected = []
        eager = []
        for g, operands, _, _, out in groups:
            if g.weights == "nvfp4":
                oracle = nvfp4_reference(g, x[:rows], ids[:rows], weights[:rows], *operands, a4=True)
            else:
                oracle = post_silu_reference(x[:rows], ids[:rows], weights[:rows], *operands,
                    limit=g.swiglu_limit, native_bf16=g.weights == "bf16")
            expected.append(oracle.cpu().float())
            eager.append(out[:rows].clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(3):
            for _, _, _, _, out in groups:
                out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for (g, _, _, _, out), baseline, oracle in zip(groups, eager, expected):
                assert torch.equal(out[:rows], baseline), f"{g.name} rows={rows}: replay changed bytes"
                actual = out[:rows].cpu().float()
                assert torch.isfinite(actual).all() and oracle.norm() > 0
                worst = torch.nn.functional.cosine_similarity(actual, oracle, dim=1).min().item()
                relative = (actual - oracle).norm() / oracle.norm()
                print(f"{g.name} tp{tp} rows={rows}: worst cosine={worst:.7f} rel_l2={relative:.7f}")
                assert worst >= 0.9999 and relative <= 0.02
