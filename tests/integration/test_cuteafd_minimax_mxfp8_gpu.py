"""Native vendor MXFP8 W8A8: independent group-32 QDQ and FP64 GEMM."""
import pytest

from ..conftest import require_b12x

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("k,n,side", [(6144, 512, 1), (3072, 6144, 2)])
def test_bf16_ingress_native_mxfp8_reuses_records_and_live_rows(k, n, side):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.dense_mxfp8 import (
        compile_dense_mxfp8_aot, compile_dense_mxfp8_quant_aot, record_bytes)

    capacity = 129
    generator = torch.Generator(device="cuda").manual_seed(541 + side)
    x = torch.randn((capacity, k), generator=generator, device="cuda").bfloat16()
    pattern = torch.linspace(-1, 1, k, device="cuda")
    for row, amplitude in enumerate((0., 1e-5, 1e-3, 60000.)):
        x[row].copy_(pattern * amplitude)
    table = torch.tensor([x.data_ptr()], dtype=torch.int64, device="cuda")
    weight_values = torch.randn((n, k), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    weight_scales = torch.randint(118, 125, (n, k // 32), generator=generator,
                                 dtype=torch.uint8, device="cuda")
    weight = weight_values.float() * torch.exp2(weight_scales.float() - 127).repeat_interleave(32, 1)
    payload = weight_values.view(torch.uint8)
    scales = weight_scales
    if side == 2:
        payload = payload.reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
        scales = scales.reshape(n, k // 128, 4).permute(1, 0, 2).contiguous()
    quant = compile_dense_mxfp8_quant_aot(input_features=k, input_dtype="bf16", rotation_count=1)
    core = compile_dense_mxfp8_aot(input_features=k, output_features=n, encoding="mxfp8",
                                 sharded_side=side, input_dtype="bf16", rotation_count=1)
    records = torch.empty(record_bytes(capacity, k), dtype=torch.uint8, device="cuda")
    output = torch.empty((capacity, n), dtype=torch.float32, device="cuda")
    limit = torch.cuda.get_device_properties(0).multi_processor_count * 4

    def launch(rows):
        quant.launch(table, records, scalars=(rows, 0, limit))
        core.launch(table, payload, scales, records, output, scalars=(rows, 0, limit))

    def oracle(rows):
        blocks = x[:rows].float().reshape(rows, k // 32, 32)
        amax = blocks.abs().amax(-1, keepdim=True)
        sf = torch.exp2(torch.ceil(torch.log2(amax / 448)).clamp(-127, 127))
        scaled = torch.where(amax == 0, 0., blocks / sf)
        aq = (scaled.to(torch.float8_e4m3fn).float() * sf).reshape(rows, k)
        # Decode actual MMA records without the quantizer's helper routines.
        rec = records.view(-1, k // 32, 528)
        packed = rec[:, :, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, k // 32, 32, 16)
        decoded = torch.empty((rec.shape[0], 16, k // 32, 32), device="cuda")
        for lane in range(32):
            r, c = lane // 4, 4 * (lane % 4)
            for row_off, byte_off in ((0, 0), (8, 4)):
                scale = torch.exp2(rec[:, :, 512 + r + row_off].float() - 127)
                for ko, bo in ((0, byte_off), (16, byte_off + 8)):
                    decoded[:, r + row_off, :, c + ko:c + ko + 4] = packed[:, :, lane, bo:bo + 4] * scale[:, :, None]
        assert torch.equal(decoded.reshape(-1, k)[:rows], aq)
        target = aq.double() @ weight[:64].double().T
        absolute_products = aq.double().abs() @ weight[:64].double().abs().T
        gamma = (k * 2 ** -24) / (1 - k * 2 ** -24)
        error = (output[:rows, :64].double() - target).abs()
        assert (error <= gamma * absolute_products + 1e-7).all()
        assert torch.isnan(output[rows:]).all(), "inactive capacity rows overwritten"

    with kernel_resolution_guard("MiniMax native MXFP8 live rows and graph replay"):
        launch(capacity)
        torch.cuda.synchronize()
        frozen = output.clone()
        for rows in (1, 16, 17, 65, capacity):
            output.fill_(float("nan"))
            launch(rows)
            torch.cuda.synchronize()
            assert torch.equal(output[:rows].view(torch.int32), frozen[:rows].view(torch.int32))
            oracle(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(capacity)
        for _ in range(2):
            x.neg_()
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            oracle(capacity)
