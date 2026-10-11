"""K1 real-geometry gates for SM120/SM121, no checkpoint or engine required."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS = {}


def program(kind, family=None, **kw):
    require_b12x()
    from b12x.integration.cuteafd import GLM53, GLM53_FLASH, FLASH, PRO, exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd import context_split as k1
    g = {"glm": GLM53, "glmf": GLM53_FLASH, "flash": FLASH, "pro": PRO}.get(family)
    key = (kind, family, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        fn = getattr(k1, f"compile_{kind}_aot")
        with exportable_compilation():
            p = fn(g, **kw) if g is not None else fn(**kw)
        export = os.environ.get("K1_EXPORT_DIR")
        if export:
            target = Path(export)
            target.mkdir(parents=True, exist_ok=True)
            stem = f"k1_{kind}_{family or 'shared'}_{len(_PROGRAMS)}"
            p.export_to_c(str(target), stem, stem)
            validate_exported_header(p, target / f"{stem}.h", stem)
        _PROGRAMS[key] = p
    return _PROGRAMS[key]


def pair_order(scores, indices, k):
    """FP64 total-order oracle; -1 padding follows every real candidate."""
    out_i = torch.full((len(scores), k), -1, dtype=torch.int32)
    out_v = torch.full((len(scores), k), -float("inf"), dtype=torch.float32)
    for r in range(len(scores)):
        picks = sorted((j for j in range(indices.shape[1]) if indices[r, j] >= 0),
                       key=lambda j: (-float(scores[r, j].double()), int(indices[r, j])))[:k]
        if picks:
            out_i[r, :len(picks)] = indices[r, picks].int()
            out_v[r, :len(picks)] = scores[r, picks].float()
    return out_v, out_i


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("ties", [False, True, "mixed"])
def test_candidate_merge(k, ties):
    p = program("dsa_candidate_merge", topk=k)
    rows = 7
    indices = torch.stack([torch.randperm(8*k)[:2*k] for _ in range(rows)]).int()
    indices[0].fill_(-1)
    indices[1, 3:] = -1
    scores = torch.zeros((rows, 2*k)) if ties else torch.randn((rows, 2*k))
    if ties == "mixed":
        scores = torch.randint(-2, 3, (rows, 2*k)).float()
        scores[:, ::11] = -float("inf")
        scores[:, ::17] = float("inf")
        scores[:, ::19] = -0.0
    scores[indices < 0] = -float("inf")
    expected_v, expected_i = pair_order(scores, indices, k)
    runs = [pair_order(scores[:,s*k:(s+1)*k], indices[:,s*k:(s+1)*k], k) for s in (0,1)]
    scores = torch.cat([run[0] for run in runs],1)
    indices = torch.cat([run[1] for run in runs],1)
    indices, scores = indices.cuda(), scores.cuda()
    table = (torch.randperm(k//16)[:k//16] + 5).int().cuda()[None].expand(rows, -1).contiguous()
    ov = torch.full((rows,k), float("nan"), device="cuda")
    oi = torch.full((rows,k), -99, device="cuda", dtype=torch.int32)
    slots = oi.clone(); lens = torch.full((rows,), -99, device="cuda", dtype=torch.int32)
    for rank in (0, 1):
        p.launch(scores, indices, ov, oi, table, slots, lens, scalars=(rows,table.shape[1],rank))
        torch.cuda.synchronize()
        assert torch.equal(oi.cpu(), expected_i)
        assert torch.equal(ov.cpu().view(torch.int32), expected_v.view(torch.int32))
        repeated = tuple(t.clone() for t in (ov, oi, slots, lens))
        p.launch(scores, indices, ov, oi, table, slots, lens, scalars=(rows,table.shape[1],rank))
        assert all(torch.equal(a, b) for a, b in zip(repeated, (ov, oi, slots, lens)))
        for r in range(rows):
            selected = sorted(int(i) for i in expected_i[r] if i >= 0 and (i//64)%2 == rank)
            expected = [int(table[r,i//128])*64 + i%64 for i in selected]
            assert lens[r] == len(expected)
            assert slots[r,:len(expected)].tolist() == expected
            assert (slots[r,len(expected):] == -1).all()


@pytest.mark.parametrize("family", ["glm", "glmf", "flash", "pro"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_scored_index_shards_equal_full(family, ties, mode):
    """Integer FP8 inputs and power-of-two weights make score arithmetic exact
    in FP32 as well as FP64; boundary ties must resolve by logical position.
    Every pool access crosses the signed 32-bit byte-offset boundary.
    """
    p = program("scored_index_topk", family, max_rows=16, max_pages=192, mode=mode)
    heads, k = p.geometry["heads"], p.geometry["topk"]
    rows, pages = 6, 192
    high = 2**31 // 8448 + 3
    physical = (torch.randperm(pages) + high).int().cuda()
    cache = torch.empty((high+pages,8448), dtype=torch.uint8, device="cuda")
    keys = torch.randint(-3,4,(pages*64,128), device="cuda").to(torch.float8_e4m3fn)
    cache[physical.long(),:8192] = keys.view(torch.uint8).view(pages,8192)
    scales = torch.ones((pages,64),device="cuda")
    cache[physical.long(),8192:] = scales.view(torch.uint8).view(pages,256)
    q = torch.randint(-2,3,(rows,heads,128),device="cuda").to(torch.float8_e4m3fn)
    if ties:
        q.zero_()
    weights = torch.full((rows,heads),1/32,device="cuda")
    lengths = torch.tensor([0,1,63,70,3200,pages*64-9],device="cuda",dtype=torch.int32)
    table = physical[None].expand(rows,-1).contiguous()
    scratch = torch.zeros(p.scratch_bytes(rows)["scratch"],device="cuda",dtype=torch.uint8)
    full_i = torch.empty((rows,k),device="cuda",dtype=torch.int32)
    full_v = torch.empty((rows,k),device="cuda")
    p.launch(q,weights,cache,table[0] if mode == "prefill" else table,lengths,full_i,full_v,scratch,scalars=(rows,pages,pages,1,0))
    ref_scores = torch.einsum("rhd,nd->rhn",q.double(),keys.double()).relu().sum(1)/32
    logical = torch.arange(pages*64,device="cuda",dtype=torch.int32)[None].expand(rows,-1)
    valid_i = torch.where(logical < lengths[:,None],logical,-1)
    ev, ei = pair_order(ref_scores.cpu(),valid_i.cpu(),k)
    assert torch.equal(full_i.cpu(),ei)
    assert torch.equal(full_v.cpu(),ev)
    cv, ci = [], []
    for rank in (0,1):
        half_table = table[:,rank::2].contiguous()
        units, tail = lengths//128, lengths%128
        half_lengths = units*64 + (tail-rank*64).clamp(0,64)
        oi = torch.empty_like(full_i); ov = torch.empty_like(full_v)
        p.launch(q,weights,cache,half_table[0] if mode == "prefill" else half_table,half_lengths,oi,ov,scratch,
                 scalars=(rows,pages//2,pages//2,2,rank))
        cv.append(ov); ci.append(oi)
    merge = program("dsa_candidate_merge",topk=k)
    v, i = torch.cat(cv,1),torch.cat(ci,1)
    out_v = torch.empty_like(full_v); out_i = torch.empty_like(full_i)
    slots = out_i.clone(); lens = lengths.clone()
    for rank in (0,1):
        merge.launch(v,i,out_v,out_i,table[:,rank::2].contiguous(),slots,lens,
                     scalars=(rows,pages//2,rank))
        assert torch.equal(out_i,full_i)
        assert torch.equal(out_v,full_v)
    torch.cuda.synchronize()


def combine_reference(o0,l0,o1,l1,sink=None):
    a,b = l0.double(),l1.double()
    m = torch.maximum(a,b)
    if sink is not None:
        m = torch.maximum(m,sink.double()[None]/torch.log(torch.tensor(2.,dtype=torch.float64,device="cuda")))
    m = torch.where(torch.isfinite(m),m,0.)
    wa,wb = torch.exp2(a-m),torch.exp2(b-m)
    d = wa+wb
    if sink is not None:
        d += torch.exp2(sink.double()[None]/torch.log(torch.tensor(2.,dtype=torch.float64,device="cuda"))-m)
    v = torch.where(wa[...,None] > 0,o0.double(),0.)*wa[...,None] + torch.where(wb[...,None] > 0,o1.double(),0.)*wb[...,None]
    return torch.where(d[...,None] > 0,v/d[...,None],0.)


@pytest.mark.parametrize("heads",[32,64])
@pytest.mark.parametrize("sink",[False,True])
def test_combine_fp64_and_existing_merge(heads,sink):
    from b12x.attention._shared.mla.merge import SparseMLASplitDecodeMergeKernel, SparseMLASplitDecodeSinkMergeKernel
    from b12x._lib.utils import current_cuda_stream
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    rows = 4
    p = program("lse_combine2",heads=heads,has_sink=sink)
    o = torch.randn((rows,heads,2,512),device="cuda",dtype=torch.bfloat16)
    l = torch.randn((rows,heads,2),device="cuda")*50
    l[0].fill_(-float("inf")); l[1,:,0] = -float("inf")
    s = torch.randn(heads,device="cuda")*10 if sink else None
    out = torch.empty((rows,heads,512),device="cuda",dtype=torch.bfloat16)
    o0,o1 = o[:,:,0].contiguous(),o[:,:,1].contiguous()
    l0,l1 = l[:,:,0].contiguous(),l[:,:,1].contiguous()
    p.launch(o0,l0,o1,l1,s,out,scalars=(rows,))
    expected = combine_reference(o0,l0,o1,l1,s)
    torch.testing.assert_close(out.float(),expected.float(),rtol=.008,atol=.004)
    merged = torch.empty_like(out)
    count = torch.tensor([2],device="cuda",dtype=torch.int32)
    kernel = (SparseMLASplitDecodeSinkMergeKernel if sink else SparseMLASplitDecodeMergeKernel)(static_num_chunks=2)
    args = [from_dlpack(t,assumed_align=16) for t in (o,l,count)]
    if sink:
        args.append(from_dlpack(s,assumed_align=16))
    args += [from_dlpack(merged,assumed_align=16),current_cuda_stream()]
    compiled = cute.compile(kernel,*args)
    compiled(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(),merged.float(),rtol=.008,atol=.004)
    assert (out[0] == 0).all()
    # Empty wire partials may be poisoned; combine must not multiply NaN by 0.
    o0[1].fill_(float("nan"))
    p.launch(o0,l0,o1,l1,s,out,scalars=(rows,))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(("row_bytes","page_rows","page_bytes"),[
    (656,64,41984),(528,64,33792),(584,64,37440),(584,256,149760),(128,64,8448),(4,64,8448),
])
def test_gather_byte_exact_high_pages(row_bytes,page_rows,page_bytes):
    p = program("paged_staging_gather",row_bytes=row_bytes,page_rows=page_rows,page_bytes=page_bytes)
    high = 2**31//page_bytes+3
    pool = torch.empty((high+5,page_bytes),device="cuda",dtype=torch.uint8)
    pool[high:] = torch.randint(0,256,(5,page_bytes),device="cuda",dtype=torch.uint8)
    rows = page_rows*5-3
    stage = torch.full((10,page_bytes),0xCD,device="cuda",dtype=torch.uint8)
    table = torch.arange(high,high+5,device="cuda",dtype=torch.int32)
    for rank in (0,1):
        p.launch(pool,table,stage,scalars=(rows,2,rank))
        assert torch.equal(stage[rank::2],pool[high:])


def pack_glm(records, family):
    width = 528 if family == "glmf" else 656
    n = len(records)
    # Exactly representable FP8 values and fixed scales; BF16 RoPE is kept.
    latent = records[:,:512].to(torch.float8_e4m3fn)
    raw = torch.zeros((n,width),device="cuda",dtype=torch.uint8)
    raw[:,:512] = latent.view(torch.uint8)
    raw[:,512:528] = torch.ones((n,4),device="cuda").view(torch.uint8)
    if width == 656:
        raw[:,528:] = records[:,512:].bfloat16().contiguous().view(torch.uint8)
    return raw, torch.cat((latent.float(),records[:,512:].bfloat16().float()),dim=1) if width == 656 else latent.float()


@pytest.mark.parametrize("family",["glm","glmf","flash","pro"])
@pytest.mark.parametrize("rows",[1,8,64])
@pytest.mark.parametrize("plan_sm_count",[170,188])
def test_partial_two_shards_fp64_and_full(family,rows,plan_sm_count):
    from b12x.integration.cuteafd import GLM53,GLM53_FLASH,FLASH,PRO
    from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot
    from b12x.integration.cuteafd.dsv4_sparse_mla import compile_dsv4_sparse_mla_aot
    v4 = family in ("flash","pro")
    g = {"glm":GLM53,"glmf":GLM53_FLASH,"flash":FLASH,"pro":PRO}[family]
    from b12x.integration.cuteafd.context_split import sparse_mla_partial_split_plan
    splits = sparse_mla_partial_split_plan(g,max_rows=64,head_count=g.heads//2,sm_count=plan_sm_count)
    p = program("sparse_mla_partial",family,max_rows=64,head_begin=g.heads//2,head_count=g.heads//2,num_splits=splits)
    k = (g.index_topk if v4 else getattr(g,"sparse_topk",g.index_topk))
    width = 512 if v4 else g.latent_dim
    q = torch.randn((rows,g.heads,width),device="cuda").bfloat16()*0.5
    n = k+128
    if v4:
        from b12x.attention._shared.mla.compressed_reference import pack_compressed_sparse_mla_kv_cache_reference as pack
        records = torch.randn((n,512),device="cuda")*.5
        main = pack(records[:128,:448],records[:128,448:].bfloat16(),page_size=256,num_pages=1)
        extra = pack(records[128:,:448],records[128:,448:].bfloat16(),page_size=64,num_pages=(k+63)//64)
        # Decode the UE8M0 records for an independent FP64 oracle.
        def unpack(cache,page_rows):
            r = cache[:,:page_rows*576].reshape(-1,576)
            latent = r[:,:448].contiguous().view(torch.float8_e4m3fn).float()
            footer = cache[:,page_rows*576:page_rows*584].reshape(-1,8)
            scales = torch.exp2(footer[:,:7].float()-127).repeat_interleave(64,dim=1)
            rope = r[:,448:576].contiguous().view(torch.bfloat16).float()
            return torch.cat((latent*scales,rope),1)
        main_values,extra_values = unpack(main,256)[:128],unpack(extra,64)[:k]
        values = torch.cat((main_values,extra_values),0)
        full_ix = torch.arange(128,device="cuda",dtype=torch.int32)[None].expand(rows,-1).contiguous()
        full_ei = torch.arange(k,device="cuda",dtype=torch.int32)[None].expand(rows,-1).contiguous()
        ml = torch.full((rows,),128,device="cuda",dtype=torch.int32)
        el = torch.full((rows,),k,device="cuda",dtype=torch.int32)
        scale=512**-.5
    else:
        records = torch.randn((n,width),device="cuda")*.5
        raw,values = pack_glm(records,family)
        pages=(n+63)//64
        main=torch.zeros((pages,64*p.geometry["record_bytes"]),device="cuda",dtype=torch.uint8)
        main.view(-1,p.geometry["record_bytes"])[:n] = raw
        extra=full_ei=el=None
        full_ix=torch.arange(k,device="cuda",dtype=torch.int32)[None].expand(rows,-1).contiguous()
        ml=torch.full((rows,),k,device="cuda",dtype=torch.int32)
        values=values[:k]
        scale=g.softmax_scale
    outputs=[];lses=[]
    head_q=q[:,g.heads//2:].contiguous()
    for rank in (0,1):
        if v4:
            # Window belongs to GPU0; C4 pages alternate.
            ix=full_ix if rank==0 else torch.full_like(full_ix,-1)
            lengths=ml if rank==0 else torch.zeros_like(ml)
            ei=full_ei[:,((torch.arange(k,device="cuda")//64)%2)==rank]
            selected=torch.full_like(full_ei,-1);selected[:,:ei.shape[1]]=ei
            ei=selected; exlen=torch.full_like(el,k//2)
        else:
            selected=full_ix[:,((torch.arange(k,device="cuda")//64)%2)==rank]
            ix=torch.full_like(full_ix,-1);ix[:,:selected.shape[1]]=selected
            lengths=torch.full_like(ml,selected.shape[1]);ei=exlen=None
        # Row 0 is empty on GPU1 and nonempty on GPU0.
        if rank==1:
            lengths[0]=0;ix[0]=-1
            if v4:
                exlen[0]=0;ei[0]=-1
        out=torch.full((rows,g.heads//2,512),float("nan"),device="cuda",dtype=torch.bfloat16)
        lse=torch.full((rows,g.heads//2),float("nan"),device="cuda")
        scratch=torch.full((p.scratch_bytes(rows)["scratch"],),0xFF,device="cuda",dtype=torch.uint8)
        p.launch(q,main,ix,lengths,extra,ei,exlen,out,lse,scratch,scalars=(rows,))
        torch.cuda.synchronize()
        repeat_out, repeat_lse = out.clone(), lse.clone()
        p.launch(q,main,ix,lengths,extra,ei,exlen,out,lse,scratch,scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out,repeat_out) and torch.equal(lse,repeat_lse)
        if rank==1:
            assert (out[0]==0).all()
            assert torch.isneginf(lse[0]).all()
        outputs.append(out);lses.append(lse)
    combine=program("lse_combine2",heads=g.heads//2,has_sink=v4)
    sink=torch.randn((g.heads//2,),device="cuda") if v4 else None
    combined=torch.empty_like(outputs[0])
    combine.launch(outputs[0],lses[0],outputs[1],lses[1],sink,combined,scalars=(rows,))
    for row in range(rows):
        # Row0 contains only GPU0's shard due to the explicit empty-shard case.
        indices=torch.arange(len(values),device="cuda")
        if row==0:
            indices=indices[((indices-(128 if v4 else 0))//64)%2==0] if not v4 else indices[(indices<128)|(((indices-128)//64)%2==0)]
        vv=values[indices].double()
        logits=head_q[row].double()@vv.T*scale
        if v4:
            logits=torch.cat((logits,sink.double()[:,None]),1)
            vv=torch.cat((vv,torch.zeros((1,512),device="cuda",dtype=torch.float64)),0)
        expected=torch.softmax(logits,dim=1)@vv[:,:512]
        if v4:
            # The existing V4 decode gate includes FP8 Q quantization error;
            # its dequantized-oracle contract is cosine > .998.
            cosine = torch.nn.functional.cosine_similarity(combined[row].float().flatten(), expected.float().flatten(), dim=0)
            assert cosine > .998
        else:
            torch.testing.assert_close(combined[row].float(),expected.float(),rtol=.025,atol=.002)
    # Match row0's deliberately empty second shard in the full selection too.
    if v4:
        keep = full_ei[0, ((torch.arange(k,device="cuda")//64)%2)==0].clone()
        full_ei[0].fill_(-1); full_ei[0,:len(keep)] = keep; el[0] = len(keep)
    else:
        keep = full_ix[0, ((torch.arange(k,device="cuda")//64)%2)==0].clone()
        full_ix[0].fill_(-1); full_ix[0,:len(keep)] = keep; ml[0] = len(keep)
    # Existing full kernel agrees at its own BF16 split rounding tolerance.
    full=(compile_dsv4_sparse_mla_aot(g,route="decode",max_rows=64,indexed_width=k)
          if v4 else compile_glm_sparse_mla_aot(g,route="decode",max_rows=64))
    fo=torch.empty((rows,g.heads,512),device="cuda",dtype=torch.bfloat16)
    fs=torch.empty(full.scratch_bytes(rows)["scratch"],device="cuda",dtype=torch.uint8)
    if v4:
        all_sink=torch.cat((sink,sink))
        full.launch(q,main,full_ix,ml,extra,full_ei,el,all_sink,fo,fs,scalars=(rows,))
    else:
        full.launch(q,main,full_ix,ml,fo,fs,scalars=(rows,))
    torch.testing.assert_close(combined.float(),fo[:,g.heads//2:].float(),rtol=.025,atol=.002)
    torch.cuda.synchronize()


def test_frozen_resolution_graph_replay_live_rows():
    """One compiled set covers multiple live counts with stable workspaces."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    rows,heads,k = 16,32,512
    merge=program("dsa_candidate_merge",topk=k)
    combine=program("lse_combine2",heads=heads)
    gather=program("paged_staging_gather",row_bytes=656,page_rows=64)
    partial=program("sparse_mla_partial","glm",max_rows=16,head_begin=32,head_count=32)
    index=program("scored_index_topk","glmf",max_rows=16,max_pages=192)
    v=torch.randn((rows,2,k),device="cuda")
    v,perm=v.sort(dim=2,descending=True,stable=True)
    i=(perm+torch.arange(2,device="cuda")[None,:,None]*k).int().reshape(rows,2*k)
    v=v.reshape(rows,2*k)
    ov=torch.empty((rows,k),device="cuda");oi=torch.empty_like(ov,dtype=torch.int32)
    slots=torch.empty_like(oi);lens=torch.empty(rows,device="cuda",dtype=torch.int32)
    table=torch.arange(96,device="cuda",dtype=torch.int32)[None].expand(rows,-1).contiguous()
    o0=torch.randn((rows,heads,512),device="cuda",dtype=torch.bfloat16);o1=o0.clone()
    l=torch.zeros((rows,heads),device="cuda");out=torch.empty_like(o0)
    pool=torch.zeros((4,41984),device="cuda",dtype=torch.uint8)
    gt=torch.arange(4,device="cuda",dtype=torch.int32)
    staging=torch.empty_like(pool)
    q=torch.zeros((rows,64,576),device="cuda",dtype=torch.bfloat16)
    ix=torch.full((rows,2048),-1,device="cuda",dtype=torch.int32)
    ln=torch.zeros(rows,device="cuda",dtype=torch.int32)
    po=torch.empty_like(o0);pl=torch.empty_like(l)
    ps=torch.empty(partial.scratch_bytes(rows)["scratch"],device="cuda",dtype=torch.uint8)
    iq=torch.zeros((rows,32,128),device="cuda",dtype=torch.float8_e4m3fn)
    iw=torch.ones((rows,32),device="cuda")
    ic=torch.zeros((96,8448),device="cuda",dtype=torch.uint8)
    il=torch.zeros(rows,device="cuda",dtype=torch.int32)
    ii=torch.empty((rows,k),device="cuda",dtype=torch.int32);iv=torch.empty((rows,k),device="cuda")
    isc=torch.zeros(index.scratch_bytes(rows)["scratch"],device="cuda",dtype=torch.uint8)
    def launch(n):
        merge.launch(v,i,ov,oi,table,slots,lens,scalars=(n,96,0))
        combine.launch(o0,l,o1,l,None,out,scalars=(n,))
        gather.launch(pool,gt,staging,scalars=(n*4,1,0))
        partial.launch(q,pool,ix,ln,None,None,None,po,pl,ps,scalars=(n,))
        index.launch(iq,iw,ic,table,il,ii,iv,isc,scalars=(n,96,96,2,0))
    launch(rows)
    torch.cuda.synchronize()
    with kernel_resolution_guard("K1 graph gate"):
        for n in (1,6,16):
            launch(n)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(n)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:n],o0[:n])
            assert (po[:n]==0).all() and torch.isneginf(pl[:n]).all()
            assert (ii[:n]==-1).all() and torch.isneginf(iv[:n]).all()


@pytest.mark.parametrize("weights", ["fp8", "bf16"])
@pytest.mark.parametrize("rows", [1, 3, 8, 64])
def test_producer_query_row_writes_the_half_of_an_assembled_query(weights, rows):
    """A head-split producer with ``query_row = 2 * heads * latent`` (context decode: this GPU's
    heads into its half of the two-GPU assembled query) writes bitwise what the contiguous producer
    writes, at the strided rows, and leaves the other half alone; records and q_resid are equal."""
    require_b12x()
    import dataclasses, re
    from b12x.integration.cuteafd import GLM53, exportable_compilation
    from b12x.integration.cuteafd.glm_attention import compile_glm_producer_aot
    g = dataclasses.replace(GLM53, name="glm53_tp2", heads=GLM53.heads // 2,
                            dense_inter=GLM53.dense_inter // 2, moe_inter=GLM53.moe_inter // 2)
    row = 2 * g.heads * g.latent_dim
    kw = {"fp8_only": "decode"} if weights == "fp8" else {}
    with exportable_compilation():
        flat = compile_glm_producer_aot(g, max_rows=64, **kw)
        half = compile_glm_producer_aot(g, max_rows=64, query_row=row, **kw)
    assert half.geometry["query_row"] == row and "query_row" not in flat.geometry
    gen = torch.Generator(device="cuda").manual_seed(7 + rows)
    pages = 8
    sizes = {"rows": rows, "pages": pages, "P": 8192,
             "producer_scratch_bytes": flat.scratch_bytes(64)["scratch"]}

    def shape(text):
        return [int(sizes.get(d.strip(), d.strip())) for d in re.findall(r"[^\[\],]+", text.split(" ")[0])]

    inputs = {}
    for op in flat.operands:
        dims = shape(op.shape)
        if op.name == "positions":
            t = torch.randperm(8192, generator=gen, device="cuda")[:rows].to(torch.int64)
        elif op.name == "kv_slots":
            t = torch.randperm(pages * 64, generator=gen, device="cuda")[:rows].to(torch.int64)
        elif op.name == "cos_sin":
            angles = torch.rand((8192, 32), generator=gen, device="cuda") * 6.0
            t = torch.cat([angles.cos(), angles.sin()], 1)
        elif op.dtype == torch.float8_e4m3fn:
            t = (torch.randn(dims, generator=gen, device="cuda") * 0.5).to(op.dtype)
        elif op.dtype == torch.float32:
            t = torch.rand(dims, generator=gen, device="cuda") * 0.02 + 0.005
        elif op.dtype == torch.bfloat16:
            t = (torch.randn(dims, generator=gen, device="cuda") * 0.1).bfloat16()
        else:
            t = torch.zeros(dims, dtype=op.dtype, device="cuda")
        inputs[op.name] = t

    def run(program, query):
        cache = torch.zeros((pages, g.kv_page_bytes), dtype=torch.uint8, device="cuda")
        q_resid = torch.zeros((rows, g.q_lora_rank), dtype=torch.bfloat16, device="cuda")
        scratch = torch.zeros(program.scratch_bytes(64)["scratch"], dtype=torch.uint8, device="cuda")
        args = {**inputs, "kv_cache": cache, "query": query, "q_resid": q_resid, "scratch": scratch}
        program.launch(*(args[op.name] for op in program.operands), scalars=(rows,))
        torch.cuda.synchronize()
        return cache, q_resid

    contiguous = torch.zeros((rows, g.heads, g.latent_dim), dtype=torch.bfloat16, device="cuda")
    ref_cache, ref_resid = run(flat, contiguous)
    for rank in (0, 1):
        assembled = torch.full((rows, 2, g.heads, g.latent_dim), 7.0, dtype=torch.bfloat16, device="cuda")
        cache, resid = run(half, assembled[:, rank])
        assert torch.equal(assembled[:, rank], contiguous), f"rank {rank}: strided query rows differ"
        assert (assembled[:, 1 - rank] == 7.0).all(), f"rank {rank}: wrote the peer's half"
        assert torch.equal(cache, ref_cache) and torch.equal(resid, ref_resid)
    assert contiguous.abs().sum() > 0
