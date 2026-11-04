"""Combined replies match an independent block-codec oracle byte for byte."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def reference(x, format):
    rows, width = x.shape
    block = 32 if format == 1 else 16
    values = x.float().reshape(rows, -1, block)
    amax = values.abs().amax(-1)
    if format == 1:
        # 448 = .875 * 2**9. frexp is exact at power-of-two boundaries;
        # CUDA log2 can round just above an integer and make ceil add a bit.
        mantissa, binary_exponent = torch.frexp(amax.double())
        exponent = (binary_exponent - 9 + (mantissa > .875)).clamp(-127, 127)
        exponent = torch.where(amax == 0, -127, exponent)
        scale = torch.exp2(exponent).float()
        codes = (exponent + 127).to(torch.uint8)
        packed = (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
        packed = torch.where((amax == 0).unsqueeze(-1), 0, packed).flatten(1)
        restored = packed.view(torch.float8_e4m3fn).float() * scale.repeat_interleave(32, -1)
    else:
        codes = (amax / 6).clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8)
        codes = torch.where(amax == 0, 0, codes.clamp_min(1))
        scale = codes.view(torch.float8_e4m3fn).float()
        normalized = values / torch.where(scale == 0, 1, scale).unsqueeze(-1)
        magnitude = normalized.abs()
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
        distance = (magnitude.unsqueeze(-1) - table).abs()
        # E2M1 halfway values round to the even code, including saturation.
        minimum = distance.amin(-1, keepdim=True)
        candidates = torch.arange(8, device=x.device).expand_as(distance)
        priority = candidates + (candidates % 2) * 8
        priority = torch.where(distance == minimum, priority, 32)
        code = priority.argmin(-1).to(torch.uint8)
        code |= normalized.signbit().to(torch.uint8) * 8
        code = torch.where((amax == 0).unsqueeze(-1), 0, code).flatten(1)
        packed = code[:, ::2] | (code[:, 1::2] << 4)
        restored = table[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)
        restored *= scale.repeat_interleave(16, -1)
    return torch.cat((packed, codes), -1), restored.bfloat16()


def mismatch_details(x, actual, expected, format):
    width = x.shape[1]
    block = 32 if format == 1 else 16
    value_bytes = width if format == 1 else width // 2
    changed_values = (actual[:, :value_bytes] != expected[:, :value_bytes]).reshape(
        x.shape[0], -1, block if format == 1 else block // 2).sum(-1)
    changed_scales = actual[:, value_bytes:] != expected[:, value_bytes:]
    blocks = ((changed_values != 0) | changed_scales).nonzero()
    details = []
    for row, index in blocks[:16].tolist():
        amax = float(x[row, index * block:(index + 1) * block].float().abs().max())
        import math
        details.append(dict(row=row, block=index, changed_values=int(changed_values[row, index]),
            actual_scale=int(actual[row, value_bytes + index]),
            expected_scale=int(expected[row, value_bytes + index]),
            amax_hex=amax.hex(), ratio_448_hex=(amax / 448).hex(),
            fp32_subnormal=0 < amax < 2.0 ** -126, zero_amax=amax == 0,
            cpu_exponent=max(-127, min(127, math.ceil(math.log2(amax / 448)))) if amax else -127))
    result = dict(changed_blocks=len(blocks), changed_scales=int(changed_scales.sum()), blocks=details)
    if len(blocks):
        import json
        print("reply_codec_mismatch=" + json.dumps(result, sort_keys=True))
    return result


@pytest.fixture(scope="module")
def reduction():
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_reduce_aot
    return compile_expert_reply_reduce_aot()


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot
    return (compile_expert_reply_aot(quantize=True),
            compile_expert_reply_aot(stage=True), compile_expert_reply_aot())


@pytest.mark.parametrize("width", [32, 160, 4096, 5120, 7168])
@pytest.mark.parametrize("ranks", range(1, 9))
@pytest.mark.parametrize("has_shared", [False, True])
def test_fused_reply_reduce_rank_rounding_and_dynamic_graph(programs, reduction, width, ranks, has_shared):
    from b12x._lib.runtime_control import kernel_resolution_guard
    quantize, stage, decode = programs
    rows = 65 if width < 4096 else 3
    torch.manual_seed(width + ranks)
    inputs = [(torch.randn(rows, width, device="cuda") * (2. ** (rank % 5 - 2))).bfloat16()
              for rank in range(ranks)]
    for x in inputs:
        x[0] = -0.0
        x[-1, ::32] *= 1e-5
    shared = torch.randn_like(inputs[0]) if has_shared else None
    planes = [torch.full((rows * width * 2 + 512,), 0xA5, device="cuda", dtype=torch.uint8)
              for _ in range(ranks)]
    views = [p[256:-256] for p in planes]
    guarded = torch.full_like(planes[0], 0x5A)
    output = guarded[256:-256].view(torch.bfloat16).view(rows, width)
    decoded = [torch.empty_like(x) for x in inputs]
    scratch = torch.empty((rows, width + width // 32), device="cuda", dtype=torch.uint8)
    mode = torch.zeros(1, device="cuda", dtype=torch.uint32)
    pointers = views + [None] * (8 - ranks)
    import os
    import ctypes as C
    native_path = os.environ.get("CUTEAFD_NATIVE_LIBRARY")
    native = C.CDLL(native_path) if native_path and width == 5120 else None
    native_output = torch.empty_like(output) if native else None
    if native:
        native.cuteafd_set_expert_hidden.argtypes = [C.c_uint32]
        native.cuteafd_set_expert_hidden.restype = C.c_int32
        assert native.cuteafd_set_expert_hidden(width) == 0
        native_reduce = native.cuteafd_reduce_compact_bf16_eight_planes_async
        native_reduce.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p,
                                 C.c_uint32, C.c_uint32, C.c_void_p]
        native_reduce.restype = C.c_int32

    def oracle(format):
        mode.fill_(format)
        for x, plane, restored in zip(inputs, views, decoded):
            if format == 0:
                plane.copy_(x.view(torch.uint8).flatten())
                restored.copy_(x)
            else:
                quantize.launch(x, plane, mode, scalars=(rows, width, format))
                stage.launch(plane, scratch, mode, scalars=(rows, width, -1))
                decode.launch(scratch, restored, mode, scalars=(rows, width, -1))
        expected = decoded[0].float()
        for restored in decoded[1:]:
            expected = expected + restored.float()
        if has_shared:
            expected = expected + shared.float()
        expected = expected.bfloat16()
        if native:
            addresses = (C.c_void_p * 8)(*[x.data_ptr() for x in decoded], *([None] * (8 - ranks)))
            assert native_reduce(addresses, shared.data_ptr() if has_shared else None,
                native_output.data_ptr(), rows, ranks, torch.cuda.current_stream().cuda_stream) == 0
            assert torch.equal(native_output.view(torch.int16), expected.view(torch.int16))
            expected = native_output
        return expected

    # One compiled callable and graph handles both formats, BF16 fallback,
    # and changed inputs; packed planes remain immutable throughout reduction.
    with kernel_resolution_guard("test_fused_reply_reduce"):
        expected = oracle(1)
        reduction.launch(*pointers, shared, output, mode,
                         scalars=(rows, width, ranks, int(has_shared), -1))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reduction.launch(*pointers, shared, output, mode,
                             scalars=(rows, width, ranks, int(has_shared), -1))
        for iteration, format in enumerate([1, 2, 0, 2, 1]):
            for x in inputs:
                x[1:].mul_(0.5 if iteration % 2 else -2.)
            expected = oracle(format)
            before = [p.clone() for p in planes]
            guarded.fill_(0x5A)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16)), (width, ranks, format)
            assert (guarded[:256] == 0x5A).all() and (guarded[-256:] == 0x5A).all()
            assert all(torch.equal(p, snapshot) for p, snapshot in zip(planes, before))
            assert all((p[:256] == 0xA5).all() and (p[-256:] == 0xA5).all() for p in planes)


def test_fp8_reference_exact_power_of_two_boundary(programs):
    quantize, stage, decode = programs
    boundary = torch.tensor(float.fromhex("0x1.cp-18"), device="cuda", dtype=torch.bfloat16)
    above = torch.nextafter(boundary, torch.full_like(boundary, float("inf")))
    x = torch.stack((boundary, above, torch.zeros_like(boundary))).unsqueeze(1).expand(-1, 32).contiguous()
    wire, restored = reference(x, 1)
    assert wire[:, -1].tolist() == [101, 102, 0]
    packed, scratch, output = torch.empty_like(wire), torch.empty_like(wire), torch.empty_like(x)
    mode = torch.tensor([1], device="cuda", dtype=torch.uint32)
    quantize.launch(x, packed, None, scalars=(3, 32, 1))
    stage.launch(packed, scratch, mode, scalars=(3, 32, -1))
    decode.launch(scratch, output, mode, scalars=(3, 32, -1))
    torch.cuda.synchronize()
    assert torch.equal(packed, wire), mismatch_details(x, packed, wire, 1)
    assert torch.equal(output.view(torch.int16), restored.view(torch.int16))


@pytest.mark.parametrize("width", [32, 128, 4096, 5120, 7168])
@pytest.mark.parametrize("format", [1, 2])
def test_reply_pair_exact_and_graph_dynamic_mode(programs, width, format):
    quantize, stage, decode = programs
    torch.manual_seed(width + format)
    for rows in [1, 65, 129]:
        x = (torch.randn(rows, width, device="cuda") * .3).bfloat16()
        x[0] = 0
        if rows > 1:
            x[1] *= 1e-5
        expected_wire, expected = reference(x, format)
        packed = torch.full_like(expected_wire, 0xA5)
        scratch = torch.full_like(expected_wire, 0xA5)
        out = torch.empty_like(x)
        mode = torch.tensor([format], device="cuda", dtype=torch.uint32)
        quantize.launch(x, packed, None, scalars=(rows, width, format))
        stage.launch(packed, scratch, mode, scalars=(rows, width, -1))
        decode.launch(scratch, out, mode, scalars=(rows, width, -1))
        torch.cuda.synchronize()
        mismatch = (packed != expected_wire).nonzero()
        samples = [(r, c, int(packed[r, c]), int(expected_wire[r, c]),
                    float(x[r, min(c, width - 1)])) for r, c in mismatch[:16].tolist()]
        assert not mismatch.numel(), (rows, width, format, samples,
                                      mismatch_details(x, packed, expected_wire, format))
        assert torch.equal(scratch, expected_wire)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        # The actual consumer expands into the BF16-sized landing plane only
        # after a separate launch staged the complete compressed source.
        landing = torch.full((rows * width * 2 + 512,), 0xA5, device="cuda", dtype=torch.uint8)
        plane = landing[256:-256]
        plane[:packed.numel()].copy_(packed.flatten())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage.launch(plane, scratch, mode, scalars=(rows, width, -1))
            decode.launch(scratch, plane, mode, scalars=(rows, width, -1))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(plane.view(torch.bfloat16).view_as(x).view(torch.int16), expected.view(torch.int16))
        assert (landing[:256] == 0xA5).all() and (landing[-256:] == 0xA5).all()
        # A mixed/decode wave switches the same captured programs to no-op.
        mode.zero_()
        plane.copy_(x.view(torch.uint8).flatten())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(plane, x.view(torch.uint8).flatten())


@pytest.mark.parametrize("format", [1, 2])
def test_reply_slot_scratch_is_independent_across_consumer_streams(programs, format):
    quantize, stage, decode = programs
    rows, width = 129, 4096
    torch.manual_seed(100 + format)
    inputs = [(torch.randn(rows, width, device="cuda") * scale).bfloat16()
              for scale in [.03, 3.]]
    expected = [reference(x, format) for x in inputs]
    packed = [torch.empty_like(wire) for wire, _ in expected]
    outputs = [torch.empty_like(x) for x in inputs]
    scratch = torch.empty((2, *packed[0].shape), device="cuda", dtype=torch.uint8)
    mode = torch.tensor([format], device="cuda", dtype=torch.uint32)
    for x, wire in zip(inputs, packed):
        quantize.launch(x, wire, None, scalars=(rows, width, format))
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(32):
        index = iteration % 2
        stream = streams[index]
        with torch.cuda.stream(stream):
            stage.launch(packed[index], scratch[index], mode, scalars=(rows, width, -1))
            decode.launch(scratch[index], outputs[index], mode, scalars=(rows, width, -1))
    torch.cuda.synchronize()
    for actual, (_, restored) in zip(outputs, expected):
        assert torch.equal(actual.view(torch.int16), restored.view(torch.int16))


@pytest.mark.parametrize("format", [1, 2])
def test_reply_chunking_is_byte_identical_above_arena_capacity(programs, format):
    quantize, _, _ = programs
    rows, width, capacity = 4161, 128, 4096
    torch.manual_seed(format)
    x = (torch.randn(rows, width, device="cuda") * .3).bfloat16()
    # Distinct scale regimes immediately before and after the arena boundary.
    x[capacity - 1] *= 1e-5
    x[capacity] = 0
    x[capacity + 1] *= 128
    expected, _ = reference(x, format)
    whole = torch.full_like(expected, 0xA5)
    chunked = torch.full_like(expected, 0x5A)
    quantize.launch(x, whole, None, scalars=(rows, width, format))
    for first in range(0, rows, capacity):
        count = min(capacity, rows - first)
        quantize.launch(x[first:first + count], chunked[first:first + count], None,
                        scalars=(count, width, format))
    torch.cuda.synchronize()
    assert torch.equal(chunked, whole)
    assert torch.equal(whole, expected), mismatch_details(x, whole, expected, format)


@pytest.mark.parametrize("format", [1, 2])
def test_fused_w4a16_topk_sum_reply_exact_and_frozen_graph_replay(programs, format):
    quantize, _, _ = programs
    import cuda.bindings.driver as cuda
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum

    max_rows, width, topk, experts = 129, 512, 8, 32
    row_bytes = (width + width // 32) if format == 1 else (width // 2 + width // 16)
    common_kwargs = dict(
        m=max_rows,
        topk=topk,
        hidden_size=width,
        element_dtype="fp16",
        full_rotation=True,
        full_rotation_output_dtype="bf16",
        coupled_hadamard=False,
        num_experts=experts,
        route_num_experts=experts,
        route_ids_dtype=torch.int32,
        use_expert_map=True,
        broadcast_svh=False,
        float32_output=False,
    )
    sum_bf16 = compile_w4a16_topk_sum(**common_kwargs, reply_format=0)
    sum_fused = compile_w4a16_topk_sum(**common_kwargs, reply_format=format)

    torch.manual_seed(2026 + format)
    fc2 = (torch.randn(max_rows * topk, width, device="cuda") * 0.35).half()
    # Zero token, tiny-amax token, and exact power-of-two scale regime token.
    fc2[:topk] = 0
    fc2[topk:2 * topk] *= 1e-4
    weights = torch.rand(max_rows, topk, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    ids = torch.randint(0, experts, (max_rows, topk), device="cuda", dtype=torch.int32)
    ids[2, 0] = -1
    ids[2, 1] = experts
    expert_map = torch.randperm(experts, device="cuda", dtype=torch.int32)
    expert_map[0] = -1
    svh = (torch.randn(experts, width, device="cuda") * 0.25).half()

    bf16_out = torch.empty((max_rows, width), device="cuda", dtype=torch.bfloat16)
    baseline_wire = torch.empty((max_rows, row_bytes), device="cuda", dtype=torch.uint8)
    guarded = torch.full((max_rows * row_bytes + 512,), 0xA5, device="cuda", dtype=torch.uint8)
    fused_wire = guarded[256:-256].view(max_rows, row_bytes)

    with kernel_resolution_guard("test_fused_w4a16_topk_sum_reply"):
        # Re-resolving with a different live m must hit the compiled cache entry.
        cached = compile_w4a16_topk_sum(**{**common_kwargs, "m": 17}, reply_format=format)
        assert cached.compiled is sum_fused.compiled

        for rows in (1, 17, 65, max_rows):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            sum_bf16(
                fc2[:rows * topk],
                bf16_out[:rows],
                weights[:rows],
                ids[:rows],
                expert_map,
                svh,
                rows,
                stream,
            )
            quantize.launch(
                bf16_out[:rows],
                baseline_wire[:rows],
                None,
                scalars=(rows, width, format),
            )
            ref_wire, _ = reference(bf16_out[:rows], format)
            assert torch.equal(baseline_wire[:rows], ref_wire)

            guarded.fill_(0xA5)
            sum_fused(
                fc2[:rows * topk],
                fused_wire[:rows],
                weights[:rows],
                ids[:rows],
                expert_map,
                svh,
                rows,
                stream,
            )
            torch.cuda.synchronize()
            assert torch.equal(fused_wire[:rows], baseline_wire[:rows]), mismatch_details(
                bf16_out[:rows], fused_wire[:rows], baseline_wire[:rows], format
            )
            assert (guarded[:256] == 0xA5).all()
            assert (guarded[256 + rows * row_bytes:] == 0xA5).all()

        # Verify CUDA Graph capture and replay on a non-default stream.
        work_stream = torch.cuda.Stream()
        with torch.cuda.stream(work_stream):
            guarded.fill_(0x5A)
            sum_fused(
                fc2,
                fused_wire,
                weights,
                ids,
                expert_map,
                svh,
                max_rows,
                cuda.CUstream(work_stream.cuda_stream),
            )
        work_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=work_stream):
            sum_fused(
                fc2,
                fused_wire,
                weights,
                ids,
                expert_map,
                svh,
                max_rows,
                cuda.CUstream(work_stream.cuda_stream),
            )
        with torch.cuda.stream(work_stream):
            guarded.fill_(0xA5)
            graph.replay()
        work_stream.synchronize()
        assert torch.equal(fused_wire, baseline_wire)
        assert (guarded[:256] == 0xA5).all() and (guarded[-256:] == 0xA5).all()

