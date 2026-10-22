"""Shared operand gates against the independent E1 trellis oracle."""
import numpy as np
import pytest
import torch


def reference(codebook="e4m3"):
    from tests._reference.native_trellis import Reference
    return Reference(codebook)


class Case:
    def __init__(self, k=1152, n=256, side=1, bits=4, rows=4096, codebook="e4m3"):
        from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
        rng = np.random.default_rng(408)
        raw = rng.integers(0, 256, k*n*bits//8, dtype=np.uint8)
        ref = reference(codebook)
        if side == 1:
            weight = np.concatenate([ref.decode_unit(chunk.tobytes(), bits, "gate", k)
                                     for chunk in raw.reshape(n//128, -1)], axis=1)
        else:
            weight = np.concatenate([ref.decode_unit(chunk.tobytes(), bits, "down", n)
                                     for chunk in raw.reshape(k//128, -1)], axis=0)
        self.x = torch.tensor(rng.normal(0, .1, (rows, k)), device="cuda", dtype=torch.float16)
        self.other_x = torch.tensor(rng.normal(0, .1, (rows, k)), device="cuda", dtype=torch.float16)
        self.inputs = torch.tensor([self.other_x.data_ptr(), self.x.data_ptr()], dtype=torch.int64, device="cuda")
        self.payload = torch.tensor(raw, dtype=torch.uint8, device="cuda")
        self.scales = torch.tensor(rng.uniform(.5, 2, k//128), dtype=torch.float16, device="cuda")
        self.weight = torch.tensor(weight, device="cuda") * self.scales.float().repeat_interleave(128)[:, None]
        self.partial = torch.empty((rows, (k+1023)//1024, n), dtype=torch.float32, device="cuda")
        self.output = torch.empty((rows, n), dtype=torch.float32, device="cuda")
        self.program = compile_trellis_linear_aot(input_features=k, output_features=n,
                           sharded_side=side, bits=bits, codebook=codebook, rotation_count=2, max_rows=rows)

    def launch(self, rows, limit=188*4, rotation=1):
        self.program.launch(self.inputs, self.payload, self.scales,
                            self.partial.flatten(), self.output.flatten(),
                            scalars=(rows, rotation, limit))


@pytest.mark.parametrize("side,bits,codebook", [(1, 4, "e4m3"), (2, 4, "e4m3"), (1, 2, "e4m3"), (2, 3, "e4m3"), (1, 4, "f16"), (2, 3, "f16"), *[(side, bits, cb) for side in (1, 2) for bits in (5, 6) for cb in ("mcg", "mul1")]])
def test_accuracy_rows_and_cta_invariance(side, bits, codebook):
    c = Case(side=side, bits=bits, codebook=codebook)
    c.launch(64, rotation=0)
    other_expected = c.output[:64].clone()
    other_ref = c.other_x[:64].double() @ c.weight.double()
    other_baseline = (c.other_x[:64].bfloat16() @ c.weight.bfloat16()).double()
    assert torch.linalg.vector_norm(other_expected.double()-other_ref) <= 1.01*torch.linalg.vector_norm(other_baseline-other_ref)
    c.launch(64)
    expected = c.output[:64].clone()
    assert not torch.equal(other_expected, expected)
    ref = c.x[:64].double() @ c.weight.double()
    baseline = (c.x[:64].bfloat16() @ c.weight.bfloat16()).double()
    error = torch.linalg.vector_norm(expected.double()-ref)
    bound = torch.linalg.vector_norm(baseline-ref)
    assert error <= 1.01 * bound, (error.item(), bound.item())
    for rows in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 36, 63, 64):
        for limit in (1, 48, 170*4, 188*4):
            c.partial.fill_(float("nan"))
            c.output.fill_(float("nan"))
            c.launch(rows, limit, rotation=rows % 2)
            selected = expected if rows % 2 else other_expected
            assert torch.equal(c.output[:rows].view(torch.int32), selected[:rows].view(torch.int32))
            assert torch.isnan(c.output[rows:]).all()
    c.launch(4096)
    ref = c.x.double() @ c.weight.double()
    baseline = (c.x.bfloat16() @ c.weight.bfloat16()).double()
    assert torch.linalg.vector_norm(c.output.double()-ref) <= 1.01*torch.linalg.vector_norm(baseline-ref)
    assert torch.equal(c.output[:64].view(torch.int32), expected.view(torch.int32))


def test_graph_replay_and_invalid_launch():
    c = Case(k=128, n=128, rows=64)
    c.launch(17)
    expected = c.output[:17].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.launch(17)
    for _ in range(3):
        c.partial.fill_(float("nan"))
        c.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(c.output[:17].view(torch.int32), expected.view(torch.int32))
        assert torch.isnan(c.output[17:]).all()
    for rows, rotation, limit in ((0, 0, 1), (65, 0, 1), (1, -1, 1), (1, 2, 1), (1, 0, 0)):
        c.output.fill_(float("nan"))
        c.launch(rows, limit, rotation)
        assert torch.isnan(c.output).all()


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("codebook", ["e4m3", "f16", "mcg", "mul1"])
def test_bit_exact_decode(bits, codebook):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.gemm.trellis_linear._native import decode_tile

    @cute.kernel
    def decode(payload: cute.Tensor, output: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        values = decode_tile(Int64(payload.iterator.toint()), Int32(tile), Int32(lane), bits, codebook == "e4m3", codebook == "mul1")
        for j in cutlass.range_constexpr(4):
            output[tile*128+lane*4+j] = values[j]

    @cute.jit
    def probe(payload: cute.Tensor, output: cute.Tensor, stream):
        decode(payload, output).launch(grid=(8192, 1, 1), block=(32, 1, 1), stream=stream)

    raw = np.random.default_rng(408).integers(0, 256, 8192*32*bits, dtype=np.uint8)
    p = torch.tensor(raw, device="cuda")
    out = torch.empty(8192*128, dtype=torch.uint32, device="cuda")
    pv, ov = from_dlpack(p), from_dlpack(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    program = cute.compile(probe, pv, ov, stream)
    program(pv, ov, stream)
    ref = reference(codebook)
    assert np.unique(ref.codewords(raw.tobytes(), bits)).size == 65536
    expected = ref.decode_tiles(raw.tobytes(), bits)
    values = out.view(torch.float16).cpu().numpy().reshape(8192, 32, 8)
    for lane in range(32):
        r, c = 2*(lane % 4), lane//4
        positions = ((r,c),(r+1,c),(r+8,c),(r+9,c),(r,c+8),(r+1,c+8),(r+8,c+8),(r+9,c+8))
        for j, (r, c) in enumerate(positions):
            np.testing.assert_array_equal(values[:, lane, j], expected[:, r, c])


GLM_TP2_SHAPES = (
    (2048, 16384, 1), (16384, 6144, 2),
    (6144, 12288, 1), (12288, 6144, 2),
)


@pytest.mark.parametrize("full_k,full_n,side", GLM_TP2_SHAPES)
@pytest.mark.skipif(__import__("os").environ.get("NATIVE_GLM_TP2_SHAPES") != "1",
                    reason="opt-in full GLM TP2 geometry gate")
def test_glm_tp2_native_slices(full_k, full_n, side):
    """Both checkpoint-unit slices, bounded F64 oracle and frozen ABI-v1 replay."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot

    rng = np.random.default_rng(408)
    rows, bits = 4096, 4
    k, n = (full_k, full_n // 2) if side == 1 else (full_k // 2, full_n)
    full_raw = rng.integers(0, 256, full_k * full_n // 2, dtype=np.uint8)
    full_x = rng.normal(0, .05, (rows, full_k)).astype(np.float16)
    full_scales = rng.uniform(.5, 2., full_k // 128).astype(np.float16)
    program = compile_trellis_linear_aot(input_features=k, output_features=n,
        sharded_side=side, bits=bits, codebook="e4m3", rotation_count=4, max_rows=rows)
    limit = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count * 4
    partial = torch.empty((rows, (k + 1023) // 1024, n), device="cuda")
    output = torch.empty((rows, n), device="cuda")
    ref = reference()
    blocks = sorted({0, n // 256, n // 128 - 1})
    cols = torch.tensor([nb * 128 + i for nb in blocks for i in range(128)], device="cuda")
    sample_rows = [0, 1, 7, 15, 63, 1023, rows - 1]

    for rank in range(2):
        units = full_raw.reshape((full_n if side == 1 else full_k) // 128, -1)
        count = (n if side == 1 else k) // 128
        raw = units[rank * count:(rank + 1) * count].copy()
        start = rank * k if side == 2 else 0
        x = torch.tensor(full_x[:, start:start + k].copy(), device="cuda")
        scales = torch.tensor(full_scales[start // 128:(start + k) // 128].copy(), device="cuda")
        payload = torch.tensor(raw.flatten(), device="cuda")
        rotations = (torch.zeros_like(x), x, -x, x * 2)
        inputs = torch.tensor([v.data_ptr() for v in rotations], device="cuda", dtype=torch.int64)
        immutable = [v.clone() for v in (*rotations, payload, scales, inputs)]
        pointers = [v.data_ptr() for v in (*rotations, payload, scales, inputs, partial, output)]
        weights = []
        for nb in blocks:
            if side == 1:
                w = ref.decode_unit(raw[nb].tobytes(), bits, "gate", k)
            else:
                pieces = raw.reshape(k // 128, n // 128, 64 * 32 * bits)[:, nb]
                w = np.concatenate([ref.decode_unit(u.tobytes(), bits, "down", 128)
                                    for u in pieces])
            weights.append(torch.tensor(w, device="cuda"))
        weight = torch.cat(weights, dim=1) * scales.float().repeat_interleave(128)[:, None]

        def launch(live, rotation=1, ctas=limit):
            program.launch(inputs, payload, scales, partial.flatten(), output.flatten(),
                           scalars=(live, rotation, ctas))

        partial.fill_(float("nan"))
        output.fill_(float("nan"))
        launch(rows)
        assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
        saved = output[:64].clone()
        torch.testing.assert_close(output[sample_rows][:, cols].double(),
            x[sample_rows].double() @ weight.double(), rtol=2e-4, atol=2e-5)
        torch.cuda.synchronize()
        with kernel_resolution_guard("GLM TP2 trellis is precompiled"):
            for live in (*range(1, 65), 127, 128, 129, 256, 1024, 4096):
                output.fill_(float("nan"))
                partial.fill_(float("nan"))
                launch(live, ctas=1 if live % 2 else limit)
                prefix = min(live, 64)
                assert torch.equal(output[:prefix].view(torch.int32), saved[:prefix].view(torch.int32))
                assert torch.isnan(output[live:]).all()
            for rotation in range(4):
                launch(17, rotation)
                torch.testing.assert_close(output[:17, cols].double(),
                    rotations[rotation][:17].double() @ weight.double(), rtol=2e-4, atol=2e-5)
            launch(17)
            graph_expected = output[:17].clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(17)
            for _ in range(3):
                output.fill_(float("nan"))
                partial.fill_(float("nan"))
                graph.replay()
                assert torch.equal(output[:17].view(torch.int32), graph_expected.view(torch.int32))
                assert torch.isnan(output[17:]).all()
        for value, before in zip((*rotations, payload, scales, inputs), immutable):
            assert torch.equal(value, before)
        assert [v.data_ptr() for v in (*rotations, payload, scales, inputs, partial, output)] == pointers
        del graph


# Whole matrices and actual TP2 shards; kv_a's stored N640 is trimmed to 576.
# Latent projections and the vocabulary head stay whole on their owner GPU.
KIMI_SHAPES = sorted({
    (7168, 12288), (7168, 6144), (12288, 7168), (6144, 7168),
    (7168, 3072), (3072, 7168), (7168, 3584), (3584, 7168),
    (7168, 1536), (1536, 18432), (1536, 9216),
    (7168, 640), (7168, 163840),
})


@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("k,n", KIMI_SHAPES)
@pytest.mark.skipif(__import__("os").environ.get("NATIVE_TRELLIS_FULL_SHAPES") != "1",
                    reason="opt-in full Kimi geometry gate")
def test_kimi_shapes(k, n, side, bits, codebook):
    """Full native launch; bounded independent F64 rows/columns, never BF16 math."""
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    rng = np.random.default_rng(273)
    rows = 64 if n >= 81920 else 4096
    raw = rng.integers(0, 256, k*n*bits//8, dtype=np.uint8)
    x = torch.tensor(rng.normal(0, .05, (rows, k)), device="cuda", dtype=torch.float16)
    inputs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
    payload = torch.tensor(raw, device="cuda")
    scales = torch.tensor(rng.uniform(.5, 2., k//128), device="cuda", dtype=torch.float16)
    partial = torch.empty((rows, (k+1023)//1024, n), device="cuda", dtype=torch.float32)
    output = torch.empty((rows, n), device="cuda", dtype=torch.float32)
    program = compile_trellis_linear_aot(input_features=k, output_features=n,
        sharded_side=side, bits=bits, codebook=codebook, rotation_count=1, max_rows=rows)
    # Sample the first, middle and final N128 units, including the final ring.
    blocks = sorted({0, n//256, n//128-1})
    ref = reference(codebook)
    weights = []
    for nb in blocks:
        if side == 1:
            unit = raw.reshape(n//128, -1)[nb]
            w = ref.decode_unit(unit.tobytes(), bits, "gate", k)
        else:
            unit = raw.reshape(k//128, n//128, 64*32*bits)[:, nb]
            w = np.concatenate([ref.decode_unit(u.tobytes(), bits, "down", 128) for u in unit])
        weights.append(torch.tensor(w, device="cuda"))
    weight = torch.cat(weights, dim=1) * scales.float().repeat_interleave(128)[:, None]
    cols = torch.tensor([nb*128+i for nb in blocks for i in range(128)], device="cuda")
    sample_rows = sorted({0, 1, 7, 15, 31, rows-1})
    expected = x[sample_rows].double() @ weight.double()
    limit = torch.cuda.get_device_properties(0).multi_processor_count * 4
    program.launch(inputs, payload, scales, partial.flatten(), output.flatten(), scalars=(rows, 0, limit))
    saved = output[:64].clone()
    assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
    actual = output[sample_rows][:, cols].double()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    for live in ((1, 8, 16, 36, 64) if rows == 64 else (1, 8, 16, 36, 64, 256, 1024, 4096)):
        output.fill_(float("nan"))
        program.launch(inputs, payload, scales, partial.flatten(), output.flatten(), scalars=(live, 0, limit))
        assert torch.equal(output[:min(64, live)].view(torch.int32), saved[:min(64, live)].view(torch.int32))
        assert torch.isnan(output[live:]).all()
        chosen = [r for r in sample_rows if r < live]
        torch.testing.assert_close(output[chosen][:, cols].double(),
                                   x[chosen].double() @ weight.double(), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("codebook", ["e4m3", "f16", "mcg", "mul1"])
def test_geometry_codebooks(bits, codebook):
    from b12x.integration.cuteafd.trellis_linear import geometry
    g = geometry(7168, 12288, bits=bits, codebook=codebook)
    assert g["abi_version"] == 1 and g["splits"] == 7
    assert (g["bits"], g["codebook"]) == (bits, codebook)


@pytest.mark.parametrize("bits,codebook", [(1, "mcg"), (7, "mcg"), (6., "mcg"),
    (True, "mcg"), (6, "sqg_fp16"), (4, "sqg_e4m3"), (6, "3inst"), (6, "unknown")])
def test_geometry_fails_closed(bits, codebook):
    from b12x.integration.cuteafd.trellis_linear import geometry
    with pytest.raises(ValueError, match="trellis requires"):
        geometry(7168, 12288, bits=bits, codebook=codebook)


@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
def test_high_rate_graph_replay_and_frozen_resolution(bits, codebook):
    from b12x._lib.runtime_control import kernel_resolution_guard
    c = Case(k=128, n=128, rows=64, bits=bits, codebook=codebook)
    c.launch(64)
    expected = c.output.clone()
    torch.cuda.synchronize()
    with kernel_resolution_guard("native high-rate trellis is precompiled"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            c.launch(64)
        for _ in range(3):
            c.output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(c.output.view(torch.int32), expected.view(torch.int32))
        for rows in (1, 8, 16, 36, 64):
            c.launch(rows)
            assert torch.equal(c.output[:rows].view(torch.int32), expected[:rows].view(torch.int32))


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
def test_wide_vs_base_bitwise_equality(side, bits, codebook, monkeypatch):
    monkeypatch.setenv("TRELLIS_WIDE", "0")
    base = Case(k=1152, n=256, side=side, bits=bits, rows=128, codebook=codebook)
    monkeypatch.setenv("TRELLIS_WIDE", "1")
    wide = Case(k=1152, n=256, side=side, bits=bits, rows=128, codebook=codebook)
    for rows in (1, 8, 16, 36, 64, 128):
        base.partial.fill_(float("nan"))
        base.output.fill_(float("nan"))
        wide.partial.fill_(float("nan"))
        wide.output.fill_(float("nan"))
        base.launch(rows)
        wide.launch(rows)
        assert torch.equal(wide.output[:rows].view(torch.int32), base.output[:rows].view(torch.int32))
        assert torch.isnan(wide.output[rows:]).all()

