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
    for rows in (1,2,7,8,9,15,16,17,31,32,33,63,64):
        for limit in (1,48,680,752):
            c.output.fill_(float("nan"))
            c.partial.fill_(float("nan"))
            c.launch(rows,limit)
            assert torch.equal(c.output[:rows].view(torch.int32),expected[:rows].view(torch.int32))
            assert torch.isnan(c.output[rows:]).all()


def test_scaled_graph_and_invalid_launch():
    c = Case("mxfp8",1,k=128,n=32,rows=64)
    c.launch(17)
    expected = c.output[:17].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.launch(17)
    for _ in range(3):
        c.output.fill_(float("nan"))
        c.partial.fill_(float("nan"))
        graph.replay()
        assert torch.equal(c.output[:17].view(torch.int32),expected.view(torch.int32))
        assert torch.isnan(c.output[17:]).all()
    for rows,limit in ((0,1),(65,1),(1,0)):
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
