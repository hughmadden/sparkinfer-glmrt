"""Native scale semantics, deterministic arithmetic and graph contracts."""
import pytest
import torch


class Case:
    def __init__(self, encoding, side, k=1152, n=96, rows=4096):
        from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
        torch.manual_seed(408)
        self.x = (torch.randn(rows,k,device="cuda")*.1).bfloat16()
        if encoding == "int4_g32":
            q = torch.randint(0,16,(n,k),dtype=torch.uint8,device="cuda")
            payload = q[:,::2] | (q[:,1::2]<<4)
            scales = (torch.rand(n,k//32,device="cuda")+.01).half()
            weight = (q.float()-8)*scales.float().repeat_interleave(32,dim=1)
        else:
            q = torch.randn(n,k,device="cuda").to(torch.float8_e4m3fn)
            payload = q.view(torch.uint8)
            if encoding == "mxfp8":
                scales = torch.randint(117,132,(n,k//32),dtype=torch.uint8,device="cuda")
                weight = q.float()*torch.exp2(scales.float()-127).repeat_interleave(32,dim=1)
            else:
                scales = torch.rand(n,device="cuda")+.01
                weight = q.float()*scales[:,None]
        if side == 2:
            payload = payload.reshape(n,k//128,-1).permute(1,0,2).contiguous()
            scales = scales.reshape(n,k//128,4).permute(1,0,2).contiguous()
        self.payload,self.scales,self.weight = payload.flatten(),scales.flatten(),weight
        self.partial = torch.empty(rows,(k+1023)//1024,n,device="cuda")
        self.output = torch.empty(rows,n,device="cuda")
        self.program = compile_scaled_linear_aot(input_features=k,output_features=n,
                            encoding=encoding,sharded_side=side,max_rows=rows)

    def launch(self,rows,limit=752):
        self.program.launch(self.x.flatten(),self.payload,self.scales,self.partial.flatten(),
                            self.output.flatten(),scalars=(rows,limit))


@pytest.mark.parametrize("encoding,side,k,n", [
    ("mxfp8", 1, 256, 96), ("mxfp8", 2, 256, 96),
    ("int4_g32", 1, 256, 96), ("int4_g32", 2, 256, 96),
    ("fp8_row", 1, 256, 96),
])
def test_benchmark_reference_weight(encoding, side, k, n):
    from benchmarks.benchmark_exlr8_dense import reference_weight
    c = Case(encoding, side, k=k, n=n, rows=1)
    actual = reference_weight(encoding, k, n, side, c.payload, c.scales)
    assert torch.equal(actual, c.weight)


@pytest.mark.parametrize("encoding,side", [("mxfp8",1),("mxfp8",2),("int4_g32",1),("int4_g32",2),("fp8_row",1)])
def test_accuracy_and_row_invariance(encoding,side):
    c = Case(encoding,side)
    c.launch(4096)
    expected = c.output[:64].clone()
    ref = c.x.double()@c.weight.double().T
    baseline = (c.x@c.weight.bfloat16().T).double()
    err = torch.linalg.vector_norm(c.output.double()-ref)
    bound = torch.linalg.vector_norm(baseline-ref)
    assert err <= 1.01*bound,(err.item(),bound.item())
    for rows in range(1, 65):
        for limit in (1,48,680,752):
            c.output.fill_(float("nan"))
            c.partial.fill_(float("nan"))
            c.launch(rows,limit)
            assert torch.equal(c.output[:rows].view(torch.int32),expected[:rows].view(torch.int32))
            assert torch.isnan(c.output[rows:]).all()


@pytest.mark.parametrize("encoding,side,live", [("mxfp8",1,17), ("mxfp8",2,65), ("int4_g32",2,129)])
def test_scaled_graph_and_invalid_launch(encoding,side,live):
    capacity = max(64, live)
    c = Case(encoding,side,k=256,n=32,rows=capacity)
    c.launch(live)
    expected = c.output[:live].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.launch(live)
    for _ in range(3):
        c.output.fill_(float("nan"))
        c.partial.fill_(float("nan"))
        graph.replay()
        assert torch.equal(c.output[:live].view(torch.int32),expected.view(torch.int32))
        assert torch.isnan(c.output[live:]).all()
    for rows,limit in ((0,1),(capacity+1,1),(1,0)):
        c.output.fill_(float("nan"))
        c.launch(rows,limit)
        assert torch.isnan(c.output).all()


@pytest.mark.parametrize("side", [1, 2])
def test_ue8m0_edge_exponents(side):
    c = Case("mxfp8",side,k=256,n=32,rows=1)
    # Each K32 sum is 32, making even scale zero's 2^-127 contribution normal.
    c.x.fill_(1)
    c.payload.fill_(0x38)  # E4M3 1.0
    c.scales.zero_()
    c.launch(1)
    expected = torch.full_like(c.output,256.0*2.0**-127)
    assert torch.equal(c.output.view(torch.int32),expected.view(torch.int32))
    c.scales.fill_(255)
    c.launch(1)
    assert torch.isnan(c.output).all()


@pytest.mark.parametrize("n", [32, 64, 96, 160])
def test_fp8_row_prefill_column_tail(n):
    c = Case("fp8_row", 1, k=256, n=n, rows=129)
    c.launch(129)
    expected = c.output.clone()
    ref = c.x.double() @ c.weight.double().T
    bf16 = (c.x @ c.weight.bfloat16().T).double()
    assert torch.linalg.vector_norm(expected.double()-ref) <= 1.01*torch.linalg.vector_norm(bf16-ref)
    for rows in (1, 16, 64, 65, 129):
        for limit in (1, 7, 48, 752):
            c.output.fill_(float("nan"))
            c.launch(rows, limit)
            assert torch.equal(c.output[:rows].view(torch.int32), expected[:rows].view(torch.int32))
            assert torch.isnan(c.output[rows:]).all()


@pytest.mark.parametrize("full_k,full_n,side", [(2048, 16384, 1), (16384, 6144, 2)])
@pytest.mark.skipif(__import__("os").environ.get("NATIVE_GLM_TP2_SHAPES") != "1",
                    reason="opt-in full GLM TP2 geometry gate")
def test_glm_tp2_int4_native_slices(full_k, full_n, side):
    """Slice the physical native layout, not a converted model weight."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot

    torch.manual_seed(408)
    rows = 4096
    k, n = (full_k, full_n // 2) if side == 1 else (full_k // 2, full_n)
    full_x = (torch.randn(rows, full_k, device="cuda") * .05).bfloat16()
    packed = torch.randint(0, 256, (full_n, full_k // 2), device="cuda", dtype=torch.uint8)
    grouped = (torch.rand(full_n, full_k // 32, device="cuda") + .01).half()
    if side == 2:
        packed = packed.reshape(full_n, full_k // 128, 64).permute(1, 0, 2).contiguous()
        grouped = grouped.reshape(full_n, full_k // 128, 4).permute(1, 0, 2).contiguous()
    program = compile_scaled_linear_aot(input_features=k, output_features=n,
        encoding="int4_g32", sharded_side=side, max_rows=rows)
    limit = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count * 4
    partial = torch.full((1,), float("nan"), device="cuda")  # ABI-v1 unused scratch.
    output = torch.empty(rows, n, device="cuda")
    cols = torch.tensor(sorted({0, 1, n // 2, n - 2, n - 1}), device="cuda")
    sample_rows = [0, 1, 7, 15, 63, 1023, rows - 1]
    for rank in range(2):
        count = n if side == 1 else k // 128
        payload = packed[rank * count:(rank + 1) * count].clone()
        scales = grouped[rank * count:(rank + 1) * count].clone()
        x = full_x[:, rank * k:(rank + 1) * k].contiguous() if side == 2 else full_x
        immutable = [v.clone() for v in (x, payload, scales)]
        pointers = [v.data_ptr() for v in (x, payload, scales, partial, output)]
        p = payload if side == 1 else payload.permute(1, 0, 2).reshape(n, k // 2)
        s = scales if side == 1 else scales.permute(1, 0, 2).reshape(n, k // 32)
        chosen = p[cols]
        q = torch.stack((chosen & 15, chosen >> 4), dim=-1).reshape(len(cols), k)
        weight = (q.float() - 8) * s[cols].float().repeat_interleave(32, dim=1)

        def launch(live, ctas=limit):
            program.launch(x.flatten(), payload.flatten(), scales.flatten(), partial, output.flatten(),
                           scalars=(live, ctas))

        output.fill_(float("nan"))
        launch(rows)
        assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
        saved = output[:64].clone()
        torch.testing.assert_close(output[sample_rows][:, cols].double(),
            x[sample_rows].double() @ weight.double().T, rtol=2e-4, atol=2e-5)
        torch.cuda.synchronize()
        with kernel_resolution_guard("GLM TP2 int4 is precompiled"):
            for live in (*range(1, 65), 127, 128, 129, 256, 1024, 4096):
                output.fill_(float("nan"))
                launch(live, ctas=1 if live % 2 else limit)
                prefix = min(live, 64)
                assert torch.equal(output[:prefix].view(torch.int32), saved[:prefix].view(torch.int32))
                assert torch.isnan(output[live:]).all()
            launch(17)
            graph_expected = output[:17].clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(17)
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                assert torch.equal(output[:17].view(torch.int32), graph_expected.view(torch.int32))
                assert torch.isnan(output[17:]).all()
        assert torch.isnan(partial).all()
        for value, before in zip((x, payload, scales), immutable):
            assert torch.equal(value, before)
        assert [v.data_ptr() for v in (x, payload, scales, partial, output)] == pointers
        del graph


def test_fp8_embedding_gather():
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    torch.manual_seed(408)
    hidden,vocabulary = 258,97
    values = torch.randn(vocabulary,hidden,device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(vocabulary,device="cuda")+.01
    ids = torch.tensor([0,96,-1,97,30],dtype=torch.int32,device="cuda")
    output = torch.empty(6,hidden,device="cuda",dtype=torch.bfloat16)
    program = compile_fp8_row_embedding_aot(hidden=hidden,vocabulary=vocabulary,max_rows=6)
    expected = (values.float()*scales[:,None]).bfloat16()[ids.clamp(0,vocabulary-1).long()]
    expected[2:4] = 0
    def launch():
        program.launch(ids,values.view(torch.uint8).flatten(),scales,output.flatten(),scalars=(5,752))
    output.fill_(float("nan"))
    launch()
    assert torch.equal(output[:5],expected)
    assert torch.isnan(output[5:]).all()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:5],expected)
    assert torch.isnan(output[5:]).all()
