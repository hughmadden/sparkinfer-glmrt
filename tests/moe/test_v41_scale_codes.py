"""Lossless MXFP4 scale storage, rejection and real-weight graph equivalence."""

import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from b12x.moe._shared.scale_codes import (
    ScaleCodeRangeError, pack_scale_codes, unpack_scale_codes,
)


@pytest.mark.parametrize("rows,columns,half", [(768, 160, 384), (1280, 160, 640), (5120, 18, None), (5120, 12, None)])
def test_scale_code_roundtrip(rows, columns, half):
    torch.manual_seed(444)
    scales = torch.randint(0, 16, (2, rows, columns), dtype=torch.uint8)
    scales += torch.randint(100, 120, (2, rows, 1), dtype=torch.uint8)
    packed = pack_scale_codes(scales, tensor_name="fixture.scale", gated_half_rows=half)
    restored = unpack_scale_codes(packed, rows=rows, columns=columns)
    assert torch.equal(restored[..., :columns], scales)
    assert torch.count_nonzero(restored[..., columns:]) == 0
    assert packed.numel() == 2 * rows * (((columns + 3) // 4) * 2 + 1)


def test_scale_codes_refuse_span_16():
    scales = torch.tensor([[[111, 127, 111, 111]]], dtype=torch.uint8)
    with pytest.raises(ScaleCodeRangeError, match=r"fixture\.w2\.scale: expert 0 row 0 UE8M0 span 16 exceeds 15; unsupported V4.1 scale codes"):
        pack_scale_codes(scales, tensor_name="fixture.w2.scale")


def _snapshot():
    path = os.environ.get("V41_SCALE_CODES_SNAPSHOT")
    if not path:
        pytest.skip("V41_SCALE_CODES_SNAPSHOT is required for real-checkpoint gates")
    return Path(path)


def _scan_scale_code_spans(snapshot):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {}
    for name, shard in index.items():
        if ".ffn.experts." in name and name.endswith(".scale"):
            selected.setdefault(shard, []).append(name)
    assert selected, "no routed/dSpark scales found"
    totals = {"backbone": {"tensors": 0, "rows": 0, "maximum_span": 0},
              "dspark": {"tensors": 0, "rows": 0, "maximum_span": 0}}
    for shard, names in sorted(selected.items()):
        with (snapshot / shard).open("rb") as source:
            header_bytes = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_bytes))
            for name in sorted(names):
                metadata = header[name]
                assert metadata["dtype"] == "F8_E8M0", (name, metadata)
                begin, end = metadata["data_offsets"]
                source.seek(8 + header_bytes + begin)
                payload = source.read(end - begin)
                assert len(payload) == end - begin, name
                scales = np.frombuffer(payload, dtype=np.uint8).reshape(metadata["shape"])
                assert scales.ndim == 2, (name, scales.shape)
                span = scales.max(-1).astype(np.int16) - scales.min(-1).astype(np.int16)
                maximum = int(span.max())
                kind = "dspark" if name.startswith("mtp.") else "backbone"
                result = totals[kind]
                result["tensors"] += 1
                result["rows"] += scales.shape[0]
                result["maximum_span"] = max(result["maximum_span"], maximum)
        print(json.dumps({"shard": shard, "totals": totals}), flush=True)
    assert all(value["tensors"] and value["rows"] for value in totals.values())
    print("full_checkpoint_scale_codes " + json.dumps({"snapshot": snapshot.name, "totals": totals}), flush=True)
    return totals


def test_full_checkpoint_scale_code_spans():
    totals = _scan_scale_code_spans(_snapshot())
    assert all(value["maximum_span"] <= 15 for value in totals.values()), totals


@pytest.mark.parametrize("environment", ["V4_FLASH_SCALE_CODES_SNAPSHOT", "V4_PRO_SCALE_CODES_SNAPSHOT"])
def test_v4_checkpoint_scale_code_span_survey(environment):
    path = os.environ.get(environment)
    if not path:
        pytest.skip(f"{environment} is required for the optional V4 survey")
    _scan_scale_code_spans(Path(path))


@pytest.mark.parametrize("prefix,intermediate,compact", [
    ("layers.3.ffn.experts.0", 384, False),
    ("layers.3.ffn.experts.0", 576, False),
    ("mtp.0.ffn.experts.0", 2304, False),
    ("layers.3.ffn.experts.0", 576, True),
    ("layers.3.ffn.experts.0", 1152, True),
])
def test_real_weights_scale_code_gemm_bitwise(prefix, intermediate, compact):
    snapshot = _snapshot()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from safetensors import safe_open
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
    from b12x.moe.fused_moe._impl import (
        _logical_weight_to_w4a8_rp_inplace as pack_weight,
        _e8m0_scale_to_w4a8_sfb_inplace as pack_scale,
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    weights, scales = {}, {}
    for projection in ("w1", "w3", "w2"):
        for suffix, target in (("weight", weights), ("scale", scales)):
            name = f"{prefix}.{projection}.{suffix}"
            with safe_open(snapshot / index[name], framework="pt", device="cpu") as source:
                tensor = source.get_tensor(name).view(torch.uint8)
                width = intermediate // (2 if suffix == "weight" else 32)
                tensor = tensor[:, :width] if projection == "w2" else tensor[:intermediate]
                target[projection] = tensor.contiguous().unsqueeze(0).cuda()
    hidden, capacity, width = 5120, 96, 192
    stored = (intermediate + 127) // 128 * 128
    s13 = torch.cat((scales["w3"], scales["w1"]), 1)
    compressed = [pack_scale_codes(s13, tensor_name=prefix + ".w13.scale", gated_half_rows=intermediate),
                  pack_scale_codes(scales["w2"], tensor_name=prefix + ".w2.scale")]
    raw = [pack_scale(s13.clone(), weight_E=1, rows=2*intermediate, k_dim=hidden, gated_half_rows=intermediate),
           pack_scale(scales["w2"].clone(), weight_E=1, rows=hidden, k_dim=intermediate)]
    payload = [pack_weight(torch.cat((weights["w3"], weights["w1"]), 1), size_k=hidden, size_n=2*intermediate, gated_half_rows=intermediate),
               pack_weight(weights["w2"].clone(), size_k=intermediate, size_n=hidden)]
    torch.manual_seed(444)
    x = torch.randn(capacity, hidden, device="cuda") * 0.5
    blocks = x.reshape(capacity, hidden // 32, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    qa = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).view(torch.uint32).reshape(capacity, hidden // 4)
    qs = (exponent + 127).to(torch.uint8)
    routing = torch.rand(capacity, device="cuda")
    # Six groups force multiple CTAs for the same real expert, up through M96.
    metadata = torch.full((6, 19), -1, dtype=torch.int32, device="cuda")
    outputs = [torch.empty(((intermediate + width - 1)//width, capacity, hidden), device="cuda") for _ in range(2)]
    compiled, arguments = [], []
    if compact:
        from b12x.moe._shared.kernels.w4a8_compact_micro import _DirectW4A8CompactLaunch
        ids = torch.zeros(capacity, dtype=torch.int32, device="cuda")
        ones = torch.ones(1, device="cuda")
        outputs = [torch.empty((capacity, hidden), device="cuda") for _ in range(2)]
        intermediates = [torch.empty(capacity * (stored + (stored // 128) * 4) // 4,
                                    device="cuda", dtype=torch.uint32) for _ in range(2)]
        projections = [torch.empty((capacity, 2 * stored), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        for coded, scale_pair, output, mid, projected in zip((False, True), (raw, compressed), outputs, intermediates, projections):
            tensors = [qa.view(torch.uint8).flatten(), qs.flatten(), payload[0].view(torch.uint32).flatten(),
                       scale_pair[0].view(torch.uint32).flatten(), mid, projected, routing, ids, ones, ones,
                       payload[1].view(torch.uint32).flatten(), scale_pair[1].view(torch.uint32).flatten(), output, ones, ones]
            args = [from_dlpack(t, assumed_align=16).iterator for t in tensors]
            kernel = _DirectW4A8CompactLaunch(max_tokens=capacity, num_topk=1, k=hidden, n=stored,
                experts=1, input_scale_count=1, down_scale_count=1, swiglu_limit=10,
                fast_math=False, native_v41=True, unit_scales=True, scale_codes=coded,
                scale_columns=intermediate // 32)
            compiled.append(cute.compile(kernel, *args, cutlass.Int32(capacity), cutlass.Int32(1), current_cuda_stream()))
            arguments.append(args)
    for coded, scale_pair, output in zip((False, True), (raw, compressed), outputs) if not compact else ():
        operands = [qa, qs, payload[0], scale_pair[0], payload[1], scale_pair[1], routing, output]
        args = [from_dlpack(t.view(torch.uint32).flatten() if i in (2, 3, 4, 5) else t, assumed_align=16)
                for i, t in enumerate(operands)]
        meta = from_dlpack(metadata, assumed_align=16)
        compiled.append(cute.compile(V41FusedSliceKernel(width, grouped=True, intermediate=intermediate, scale_codes=coded),
                                     *args, cutlass.Int32(capacity), current_cuda_stream(), meta, cutlass.Int32(6)))
        arguments.append((args, meta))
    for rows in (1, 3, 16, 64, 96):
        metadata.fill_(-1)
        groups = (rows + 15) // 16
        for group in range(groups):
            count = min(16, rows - group * 16)
            metadata[group, :3] = torch.tensor((0, count, group*16), device="cuda")
            metadata[group, 3:3+count] = torch.arange(group*16, group*16+count, device="cuda")
        graphs = []
        for launch, arguments_for_arm in zip(compiled, arguments):
            if compact:
                launch_args = (*arguments_for_arm, rows, 1, current_cuda_stream())
            else:
                args, meta = arguments_for_arm
                launch_args = (*args, rows, current_cuda_stream(), meta, groups)
            launch(*launch_args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(*launch_args)
            graphs.append(graph)
        allocated = torch.cuda.memory_allocated()
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        selected = [output[:rows] if compact else output[:, :rows] for output in outputs]
        assert torch.isfinite(selected[0]).all()
        assert torch.count_nonzero(selected[0]) > 0
        assert torch.equal(selected[0].view(torch.uint8), selected[1].view(torch.uint8)), (prefix, intermediate, compact, rows)
        print(json.dumps({"prefix": prefix, "intermediate": intermediate, "compact": compact, "rows": rows, "byte_identical": True,
                          "raw_scale_bytes": sum(t.numel()*t.element_size() for t in raw),
                          "code_scale_bytes": sum(t.numel() for t in compressed)}), flush=True)
