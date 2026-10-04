"""Exact V4.1 TP4 resident tails against padded slices and the FP32 oracle."""
import pytest
import torch
from tests.conftest import require_b12x
from tests.moe.test_v41_expert_numerics import reference


def pack(weights, scales, exact):
    """Independent tensor-layout reference, including both FC1 projection halves."""
    n, h = weights['w1'].shape[1], weights['w1'].shape[2] * 2
    padded = n if exact else (n + 127) // 128 * 128
    result = []
    for gated, scale in [(True, False), (True, True), (False, False), (False, True)]:
        names = ['w3', 'w1'] if gated else ['w2']
        planes = []
        for name in names:
            source = scales[name] if scale else weights[name].view(torch.int32)
            if gated:
                source = torch.nn.functional.pad(source, (0, 0, 0, padded - n))
            else:
                source = torch.nn.functional.pad(source, (0, (padded-n)//(32 if scale else 8)))
            planes.append(source)
        source = torch.cat(planes, 1) if gated and not exact else None
        tiles = []
        if exact:
            for plane in planes:
                for nt in range(0, plane.shape[1], 128):
                    tile = plane[:,nt:nt+128]
                    for kt in range(0, tile.shape[2], 4 if scale else 16):
                        piece = tile[:,:,kt:kt+(4 if scale else 16)]
                        if scale:
                            tiles.append(piece.contiguous().flatten(1))
                        else:
                            e, rows, cols = piece.shape
                            tiles.append(piece.reshape(e,rows//32,4,8,cols//4,4).permute(0,4,1,3,5,2).contiguous().flatten(1))
        else:
            if source is None: source = planes[0]
            for nt in range(0, source.shape[1], 256):
                for kt in range(0, source.shape[2], 4 if scale else 16):
                    piece = source[:,nt:nt+256,kt:kt+(4 if scale else 16)]
                    if scale: tiles.append(piece.contiguous().flatten(1))
                    else:
                        e, rows, cols = piece.shape
                        tiles.append(piece.reshape(e,rows//32,4,8,cols//4,4).permute(0,4,1,3,5,2).contiguous().flatten(1))
        result.append(torch.cat(tiles,1).contiguous().view(torch.uint32).flatten().cuda())
    return result


def run_pipeline(x, ids, routing, packed, width, exact, atomic=False):
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
    m, h = x.shape
    e, topk, n = 8, ids.shape[1], 576
    wire = torch.zeros(m, h+h//32, dtype=torch.uint8, device='cuda')
    blocks = x.float().reshape(m,-1,32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448)))
    wire[:,:h] = (blocks/scale[...,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(m,h)
    wire[:,h:] = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    routes=m*topk
    partial=torch.empty(((n+width-1)//width,routes,h),device='cuda',dtype=torch.float32)
    output=torch.empty(m*h if atomic else (routes,h),device='cuda',dtype=torch.float32)
    scratch=[torch.empty(shape,device='cuda',dtype=dtype) for shape,dtype in [
        ((1,),torch.int32),((e*routes,),torch.int32),((e,),torch.int32),((e,2),torch.int32),
        ((routes,19),torch.int32),((routes,),torch.float32),((routes,),torch.int32)]]
    tensors=[wire[:,:h].view(torch.uint32),wire[:,h:],*packed,ids.flatten(),routing.flatten(),*scratch,partial,output]
    def tensor(t):
        from cutlass.cute.runtime import from_dlpack
        return from_dlpack(t,assumed_align=16)
    args=[tensor(t) for t in tensors]
    fn=cute.compile(V41SlicePipeline(m,width,experts=e,topk=topk,exact_storage=exact,atomic_tokens=atomic),*args,cutlass.Int32(m),current_cuda_stream())
    fn(*args,cutlass.Int32(m),current_cuda_stream())
    # Resolve once, exercise multiple live counts, then replay with fixed storage.
    fn(*args,cutlass.Int32(1),current_cuda_stream())
    fn(*args,cutlass.Int32(m),current_cuda_stream())
    torch.cuda.synchronize()
    expected = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args,cutlass.Int32(m),current_cuda_stream())
    output.fill_(float('nan'))
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == allocated
    torch.cuda.synchronize()
    if not atomic: assert torch.equal(output, expected)
    else: torch.testing.assert_close(output, expected, atol=.002, rtol=.002)
    return output.reshape(m,topk,h).sum(1) if not atomic else output.reshape(m,h)


@pytest.mark.parametrize('m,width,atomic', [(1,64,False),(16,192,False),(80,192,False),(16,128,False),(256,192,True)])
def test_exact_slices(m,width,atomic):
    require_b12x()
    torch.manual_seed(4100+m)
    e,n,h,k=8,576,5120,6
    x=(torch.randn(m,h,device='cuda')*.5).bfloat16()
    ids=torch.rand(m,e,device='cuda').topk(k,-1).indices.int()
    routing=torch.rand(m,k,device='cuda'); routing=(routing/routing.sum(-1,keepdim=True)*1.5).float()
    weights={};scales={}
    for name,shape in [('w1',(e,n,h//2)),('w3',(e,n,h//2)),('w2',(e,h,n//2))]:
        weights[name]=torch.randint(0,256,shape,dtype=torch.uint8)
        scales[name]=torch.randint(121,124,(*shape[:-1],shape[-1]//16),dtype=torch.uint8)
    old=run_pipeline(x,ids,routing,pack(weights,scales,False),width,False,atomic)
    new=run_pipeline(x,ids,routing,pack(weights,scales,True),width,True,atomic)
    if not atomic: assert torch.equal(old,new), (old-new).abs().max().item()
    else: torch.testing.assert_close(old,new,atol=.002,rtol=.002)
    oracle=reference(x,ids,routing,{k:v.cuda() for k,v in weights.items()},{k:v.cuda() for k,v in scales.items()})
    cos=torch.nn.functional.cosine_similarity(new.flatten(),oracle.flatten(),dim=0).item()
    assert cos>.9999,cos
