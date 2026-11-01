"""Exact K32 resident tails against identical padded slices and the FP32 oracle."""
import pytest
import torch
from tests.conftest import require_b12x
from tests.moe.test_v41_expert_numerics import reference


def pack(weights, scales, exact, *, w2_pitch=None):
    """Independent tensor-layout reference, including both FC1 projection halves."""
    n, h = weights['w1'].shape[1], weights['w1'].shape[2] * 2
    padded = n if exact else (n + 127) // 128 * 128
    if w2_pitch is not None:
        assert exact and w2_pitch >= n and w2_pitch % 128 == 0
    result = []
    for gated, scale in [(True, False), (True, True), (False, False), (False, True)]:
        names = ['w3', 'w1'] if gated else ['w2']
        planes = []
        for name in names:
            source = scales[name] if scale else weights[name].view(torch.int32)
            if gated:
                source = torch.nn.functional.pad(source, (0, 0, 0, padded - n))
            else:
                stored = w2_pitch if w2_pitch is not None and not scale else padded
                source = torch.nn.functional.pad(source, (0, (stored-n)//(32 if scale else 8)))
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


def run_pipeline(x, ids, routing, packed, width, exact, atomic=False, *, n=576, coded=False, return_routes=False, fc1_k_splits=None):
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
    m, h = x.shape
    e, topk = 8, ids.shape[1]
    wire = torch.zeros(m, h+h//32, dtype=torch.uint8, device='cuda')
    blocks = x.float().reshape(m,-1,32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448)))
    wire[:,:h] = (blocks/scale[...,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(m,h)
    wire[:,h:] = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    routes=m*topk
    partial_words = 1 if atomic else ((n+width-1)//width)*routes*(width//4+width//32)
    partial_storage = torch.full((partial_words+16,), 12345., device='cuda')
    partial = partial_storage[:partial_words]
    output=torch.empty(m*h if atomic else (routes,h),device='cuda',dtype=torch.float32)
    scratch=[torch.empty(shape,device='cuda',dtype=dtype) for shape,dtype in [
        ((1,),torch.int32),((e*routes,),torch.int32),((e,),torch.int32),((e,2),torch.int32),
        ((routes,19),torch.int32),((routes,),torch.float32),((routes,),torch.int32)]]
    tensors=[wire[:,:h].view(torch.uint32),wire[:,h:],*packed,ids.flatten(),routing.flatten(),*scratch,partial,output]
    def tensor(t):
        from cutlass.cute.runtime import from_dlpack
        return from_dlpack(t,assumed_align=16)
    args=[tensor(t) for t in tensors]
    fn=cute.compile(V41SlicePipeline(m,width,experts=e,topk=topk,intermediate=n,hidden=h,exact_storage=exact,atomic_tokens=atomic,scale_codes=coded,fc1_k_splits=fc1_k_splits),*args,cutlass.Int32(m),current_cuda_stream())
    fn(*args,cutlass.Int32(m),current_cuda_stream())
    torch.cuda.synchronize()
    expected = output.clone()
    for live in sorted({1, min(17, m), m}):
        output.fill_(float("nan"))
        partial.fill_(float("nan"))
        for storage in scratch:
            storage.fill_(255)
        fn(*args,cutlass.Int32(live),current_cuda_stream())
        if not atomic:
            assert torch.equal(output[:live * topk], expected[:live * topk])
    fn(*args,cutlass.Int32(m),current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args,cutlass.Int32(m),current_cuda_stream())
    output.fill_(float('nan'))
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == allocated
    torch.cuda.synchronize()
    # The exporter exposes exactly the compact activation arena, not hidden-wide planes.
    assert bool((partial_storage[partial_words:] == 12345.).all())
    if not atomic: assert torch.equal(output, expected)
    else: torch.testing.assert_close(output, expected, atol=.002, rtol=.002)
    if return_routes:
        assert not atomic
        return output.reshape(m, topk, h)
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


# Every adjacent width at its production hidden geometry, including shipped
# equal splits and native V4 Flash TP3/TP6.
WIDTHS = [(hidden, width)
          for intermediate, hidden in ((2048, 4096), (2304, 5120), (3072, 7168))
          for width in sorted({32 * (intermediate // 32 // world + extra)
                               for world in range(2, 9)
                               for extra in (0, int(intermediate // 32 % world != 0))})]


@pytest.mark.parametrize("h,n", WIDTHS)
@pytest.mark.parametrize("coded", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_all_k32_widths_exact_storage(h, n, coded, atomic):
    require_b12x()
    from b12x.moe._shared.scale_codes import pack_scale_codes
    torch.manual_seed(928 + n)
    # Eight experts exercise nonzero expert strides and both projection halves.
    m, e, k = 17, 8, 6
    x = (torch.randn(m, h, device="cuda") * .5).bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(k, -1).indices.int()
    routing = torch.rand(m, k, device="cuda")
    routing = (routing / routing.sum(-1, keepdim=True) * 1.5).float()
    weights, scales = {}, {}
    for name, shape in [("w1", (e, n, h // 2)), ("w3", (e, n, h // 2)), ("w2", (e, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8)
    padded = pack(weights, scales, False)
    exact = pack(weights, scales, True) if n % 128 else list(padded)
    if coded:
        exact[1] = pack_scale_codes(torch.cat((scales["w3"], scales["w1"]), 1),
                                   tensor_name="w13.scale", gated_half_rows=n).view(torch.uint32).flatten().cuda()
        exact[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
    reference_output = run_pipeline(x, ids, routing, padded, 192 if n % 192 == 0 else 128,
                                    False, atomic, n=n)
    result = run_pipeline(x, ids, routing, exact, 192 if n % 192 == 0 else 128,
                          bool(n % 128), atomic, n=n, coded=coded)
    if atomic:
        torch.testing.assert_close(result, reference_output, atol=.002, rtol=.002)
    else:
        assert torch.equal(result, reference_output), (n, coded, (result - reference_output).abs().max().item())


@pytest.mark.parametrize("h,n", [(h, n) for h in (4096, 5120, 7168) for n in (256, 288)])
@pytest.mark.parametrize("coded", [False, True])
def test_ordered_tile32_decode_qualification(h, n, coded):
    require_b12x()
    import json
    from benchmarks.bench_cuteafd_fp8_moe import (
        f32_errors, mxfp4_f32_reference, native_pair_arm,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    m, e, k = 80, 8, 6
    generator = torch.Generator().manual_seed(92832 + n)
    weights, scales = {}, {}
    for name, shape in [("w1", (e, n, h // 2)), ("w3", (e, n, h // 2)), ("w2", (e, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                     dtype=torch.uint8, generator=generator)
    torch.manual_seed(92832 + h + n)
    x = (torch.randn(m, h, device="cuda") * .5).bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(k, -1).indices.int()
    routing = torch.rand(m, k, device="cuda")
    routing /= routing.sum(-1, keepdim=True)
    reference = mxfp4_f32_reference(x, ids, routing, weights, scales, 10., native=True)
    expected, results = None, []
    for capacity in (80, 16, 1):
        operands = (x[:capacity], ids[:capacity], routing[:capacity], weights, scales)
        control_tile = 192 if coded and n == 288 else 128
        control, control_output, control_owned = native_pair_arm(
            *operands, control_tile, exact_control, False, coded, qualify=True, fc1_k_splits=1)
        candidate, output, owned = native_pair_arm(
            *operands, 32, True, False, coded, qualify=True)
        control()
        candidate()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all() and torch.count_nonzero(output)
        if expected is None:
            expected = output.clone()
        assert torch.equal(output, expected[:capacity * k]), "decode-capacity row invariance"
        oracle = reference[:capacity * k]
        control_error = f32_errors(control_output, oracle)
        candidate_error = f32_errors(output, oracle)
        torch.testing.assert_close(output, control_output, atol=.01, rtol=.01)
        results.append(dict(capacity=capacity, control=control_error, candidate=candidate_error,
                            eligible=all(candidate_error[key] <= control_error[key]
                                         for key in ("max_abs", "rms"))))
    print("mxfp4_tile32_qualification " + json.dumps(dict(
        hidden=h, logical=n, compact=bool(n % 128), coded=coded,
        cached_slices=n // 32 <= 8, deterministic=True, poison_live_graph=True,
        decode_capacity_row_invariance=True, results=results)), flush=True)


@pytest.mark.parametrize("weight_seed", [92800, 92801, 92802, 92803])
def test_v41_s576_tile128_synthetic_oracle_seeds(weight_seed):
    _synthetic_oracle_seeds(weight_seed, 5120, 576, True, 128, 192,
                            "mxfp4_s576_synthetic_oracle")


@pytest.mark.parametrize("h,n,coded", [(4096, 288, False), (5120, 256, False),
                                     (7168, 288, False), (5120, 256, True),
                                     (5120, 288, True)])
@pytest.mark.parametrize("weight_seed", [92800, 92801, 92802, 92803])
def test_ordered_tile32_synthetic_oracle_seeds(weight_seed, h, n, coded):
    control_tile = 192 if coded and n == 288 else 128
    _synthetic_oracle_seeds(weight_seed, h, n, coded, 32, control_tile,
                            "mxfp4_tile32_synthetic_oracle")


@pytest.mark.parametrize("n", [384, 576])
def test_v41_fc1_k_split_capacity_invariance(n):
    require_b12x()
    from b12x.moe._shared.scale_codes import pack_scale_codes
    torch.manual_seed(928384)
    m, e, k, h = 80, 8, 6, 5120
    x = (torch.randn(m, h, device="cuda") * .5).bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(k, -1).indices.int()
    routing = torch.rand(m, k, device="cuda")
    routing /= routing.sum(-1, keepdim=True)
    weights, scales = {}, {}
    for name, shape in [("w1", (e, n, h // 2)), ("w3", (e, n, h // 2)), ("w2", (e, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8)
    packed = pack(weights, scales, True)
    packed[1] = pack_scale_codes(torch.cat((scales["w3"], scales["w1"]), 1),
                                tensor_name="w13.scale", gated_half_rows=n).view(torch.uint32).flatten().cuda()
    packed[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
    actual = run_pipeline(x, ids, routing, packed, 128, True, n=n, coded=True,
                          return_routes=True, fc1_k_splits=2)
    for capacity in (1, 16):
        prefix = run_pipeline(x[:capacity], ids[:capacity], routing[:capacity], packed, 128, True,
                              n=n, coded=True, return_routes=True, fc1_k_splits=2)
        assert torch.equal(prefix, actual[:capacity]), "K split changed with decode capacity"


@pytest.mark.parametrize("n", [384, 576])
@pytest.mark.parametrize("weight_seed", [92800, 92801, 92802, 92803])
def test_v41_k_split_synthetic_oracle_seeds(weight_seed, n):
    _synthetic_oracle_seeds(weight_seed, 5120, n, True, 128, 128,
                            "mxfp4_k_split_synthetic_oracle", k_split=2, exact_control=True)


def _synthetic_oracle_seeds(weight_seed, h, n, coded, tile, control_tile, marker,
                            k_split=1, exact_control=False):
    require_b12x()
    import json
    from benchmarks.bench_cuteafd_fp8_moe import (
        f32_errors, mxfp4_f32_reference, native_pair_arm,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    m, e, k = 80, 8, 6
    generator = torch.Generator().manual_seed(weight_seed)
    weights, scales = {}, {}
    for name, shape in [("w1", (e, n, h // 2)), ("w3", (e, n, h // 2)), ("w2", (e, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                     dtype=torch.uint8, generator=generator)
    for activation_seed in range(8):
        torch.manual_seed(activation_seed)
        x = (torch.randn(m, h, device="cuda") * .5).bfloat16()
        ids = torch.rand(m, e, device="cuda").topk(k, -1).indices.int()
        routing = torch.rand(m, k, device="cuda")
        routing /= routing.sum(-1, keepdim=True)
        reference = mxfp4_f32_reference(x, ids, routing, weights, scales, 10., native=True)
        for capacity in (1, 16, 80):
            operands = (x[:capacity], ids[:capacity], routing[:capacity], weights, scales)
            control, control_output, control_owned = native_pair_arm(
                *operands, control_tile, False, False, coded, qualify=True)
            candidate, output, owned = native_pair_arm(
                *operands, tile, True, False, coded, qualify=True, fc1_k_splits=k_split)
            control()
            candidate()
            torch.cuda.synchronize()
            assert torch.isfinite(output).all() and torch.count_nonzero(output)
            oracle = reference[:capacity * k]
            print(marker + " " + json.dumps(dict(
                weight_seed=weight_seed, activation_seed=activation_seed, capacity=capacity,
                hidden=h, logical=n, coded=coded, tile=tile, control_tile=control_tile, k_split=k_split,
                control=f32_errors(control_output, oracle), candidate=f32_errors(output, oracle),
                synthetic=True, timed=False)), flush=True)


@pytest.mark.parametrize("world,n", [(4, 576), (8, 288)])
@pytest.mark.parametrize("m", [1, 16, 80, 256, 1024, 4096])
def test_shipped_v41_width192_exact_route_bytes(world, n, m):
    require_b12x()
    import json
    from b12x.moe._shared.scale_codes import pack_scale_codes

    h, e, k, width = 5120, 8, 6, 192
    stored = (n + 127) // 128 * 128
    torch.manual_seed(9280 + n + m)
    x = (torch.randn(m, h, device="cuda") * .5).bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(k, -1).indices.int()
    routing = torch.rand(m, k, device="cuda")
    routing = (routing / routing.sum(-1, keepdim=True)).float()
    weights, scales = {}, {}
    for name, shape in [("w1", (e, n, h // 2)), ("w3", (e, n, h // 2)), ("w2", (e, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8)

    def resident(exact):
        packed = pack(weights, scales, exact)
        half = n if exact else stored
        s13 = torch.cat([torch.nn.functional.pad(scales[name], (0, 0, 0, half - n))
                         for name in ("w3", "w1")], 1)
        packed[1] = pack_scale_codes(s13, tensor_name="w13.scale", gated_half_rows=half).view(torch.uint32).flatten().cuda()
        packed[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
        return packed

    # Compare the ABI's ordered route planes, not a later sum that could hide
    # different per-route values. Width 192 and its plane fold stay unchanged.
    baseline = run_pipeline(x, ids, routing, resident(False), width, False,
                            n=n, coded=True, return_routes=True)
    candidate = run_pipeline(x, ids, routing, resident(True), width, True,
                             n=n, coded=True, return_routes=True)
    assert torch.isfinite(candidate).all() and torch.count_nonzero(candidate) > 0
    assert torch.equal(candidate.view(torch.uint8), baseline.view(torch.uint8)), (world, n, m)
    print("mxfp4_shipped_recipe " + json.dumps({
        "world": world, "logical": n, "padded": stored, "width": width,
        "planes": (n + width - 1) // width, "rows": m, "coded": True,
        "route_bytes": candidate.numel() * candidate.element_size(), "bitwise_equal": True,
    }), flush=True)
