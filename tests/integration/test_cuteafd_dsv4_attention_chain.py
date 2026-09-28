"""End to end: V4 Flash layer-2 (C4) attention prefill through every AOT program
(producer -> compressor -> index producer -> top-k -> sparse MLA -> wo) versus
the prototype's DeepseekV4Layer.attention on the prepared b12x ops.

Opt-in: needs the prototype (``CUTEAFD_DSV4_PROTOTYPE``, default the
cuteafd-work checkout) and the V4 Flash snapshot mounted in the container.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import SNAPSHOT, has_checkpoint

PROTOTYPE = Path(os.environ.get(
    "CUTEAFD_DSV4_PROTOTYPE", "/home/tj/Developer/cuteafd-work/python/reference/deepseek_v4"))


def run(program, scalars, **tensors):
    names = [o.name for o in program.operands]
    missing = set(names) - set(tensors)
    assert not missing, missing
    program.launch(*[tensors[n] for n in names], scalars=scalars)


def scratch(program, rows, zero=False):
    n = program.scratch_bytes(rows)["scratch"]
    return (torch.zeros if zero else torch.empty)((n,), dtype=torch.uint8, device="cuda")


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


@pytest.mark.parametrize("T", [64, 300])
def test_flash_layer2_attention_chain_matches_prototype(T, monkeypatch):
    """Every AOT program chained exactly as the engine would, vs the prototype."""
    require_b12x()
    if not (PROTOTYPE / "b12x_model.py").is_file() or not has_checkpoint():
        pytest.skip("prototype b12x_model.py or V4 Flash snapshot not available")
    sys.path.insert(0, str(PROTOTYPE))
    import b12x_model as bm

    from b12x.integration.cuteafd import FLASH, exportable_compilation
    from b12x.integration.cuteafd import dsv4_compressor as C, dsv4_indexer as I
    from b12x.integration.cuteafd import dsv4_producer as P, dsv4_sparse_mla as M, dsv4_wo as W

    dev = torch.device("cuda", torch.cuda.current_device())
    layer = 2
    monkeypatch.setattr(bm, "load_routed_experts", lambda *a, **k: None)  # attention only
    m = bm.DeepseekV4B12x(SNAPSHOT, dev.index)
    cfg = m.cfg
    blk = m.load_layer(layer)
    w = blk.w
    meta = m.metadata(T)
    cs = m.cos_sin(cfg.ratio(layer), T)
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn((T, cfg.dim), generator=gen) * 1.0).bfloat16().to(dev)

    dbg = {}
    with torch.inference_mode():
        ref = blk.attention(x, meta, cs, dbg)
    torch.cuda.synchronize()

    with exportable_compilation():
        prod = P.compile_dsv4_producer_aot(FLASH, max_rows=T)
        comp = C.compile_dsv4_compressor_prefill_aot(FLASH, ratio=4)
        iprod = P.compile_dsv4_index_producer_aot(FLASH, max_rows=T)
        c = meta.compress[4]
        pages = c["pages"]
        topk = I.compile_dsv4_index_topk_aot(FLASH, max_rows=T, max_pages=pages, mode="prefill")
        mla = M.compile_dsv4_sparse_mla_aot(FLASH, route="prefill", max_rows=T, indexed_width=512,
                                            indexed_page_rows=64)
        wo = W.compile_dsv4_wo_projection_aot(FLASH, max_rows=T)

    i32 = dict(dtype=torch.int32, device=dev)
    main_cache = torch.zeros((meta.main_pages, 149_760), dtype=torch.uint8, device=dev)
    query = torch.empty((T, 64, 512), dtype=torch.bfloat16, device=dev)
    q_rank = torch.empty((T, 1024), dtype=torch.bfloat16, device=dev)
    pw = w.producer
    run(prod, (T,), hidden=x, positions=meta.positions, main_slots=meta.main_slots, cos_sin=cs,
        w_qkv=pw.qkv_rank.weight.values, w_qkv_scale=pw.qkv_rank.weight.scale_mma,
        w_q=pw.q.weight.values, w_q_scale=pw.q.weight.scale_mma, q_norm=pw.q_norm, kv_norm=pw.kv_norm,
        main_kv_cache=main_cache, query=query, q_rank=q_rank, scratch=scratch(prod, T))

    comp_cache = torch.zeros((pages, 37_440), dtype=torch.uint8, device=dev)
    index_cache = torch.zeros((pages, 8_448), dtype=torch.uint8, device=dev)
    kv = torch.zeros((1, 16, 1024), device=dev); score = torch.zeros_like(kv)
    ikv = torch.zeros((1, 16, 256), device=dev); iscore = torch.zeros_like(ikv)
    cw = w.compressor
    run(comp, (T, int(c["group_source_starts"].shape[0]), 1), hidden=x,
        active_groups=c["active_groups"], group_source_starts=c["group_source_starts"],
        group_rope_positions=c["group_rope_positions"], compressed_slots=c["compressed_slots"],
        active_sequences=c["active_sequences"], sequence_offsets=c["sequence_offsets"],
        state_sequence_ids=c["state_sequence_ids"], cos_sin=cs, joint_projection=cw.joint_projection,
        main_ape=cw.main_ape, main_norm=cw.main_norm, compressed_cache=comp_cache, main_kv_state=kv,
        main_score_state=score, index_ape=cw.index_ape, index_norm=cw.index_norm,
        index_cache=index_cache, index_kv_state=ikv, index_score_state=iscore, scratch=scratch(comp, T))

    index_q = torch.empty((T, 64, 128), dtype=torch.float8_e4m3fn, device=dev)
    head_w = torch.empty((T, 64), dtype=torch.float32, device=dev)
    run(iprod, (T,), q_rank=q_rank, hidden=x, positions=meta.positions, cos_sin=cs,
        w_q=w.indexer.q.weight.values, w_q_scale=w.indexer.q.weight.scale_mma,
        w_proj=w.indexer.weights_projection, query=index_q, head_weights=head_w,
        scratch=scratch(iprod, T))

    selected = torch.empty((T, 512), dtype=torch.int32, device=dev)
    run(topk, (T, pages, pages), q_fp8=index_q, weights=head_w, index_k_cache=index_cache,
        page_table=torch.arange(pages, **i32), cache_lengths=c["index_cache_lengths"],
        output_indices=selected, scratch=scratch(topk, T, zero=True))

    attn = torch.empty((T, 64, 512), dtype=torch.bfloat16, device=dev)
    run(mla, (T,), q=query, swa_cache=main_cache, swa_indices=meta.swa_indices,
        swa_lengths=meta.swa_lengths, indexed_cache=comp_cache, indexed_indices=selected,
        indexed_lengths=c["indexed_lengths"], attn_sink=w.attn_sink, out=attn, scratch=scratch(mla, T))

    out = torch.empty((T, cfg.dim), dtype=torch.bfloat16, device=dev)
    run(wo, (T,), o=attn, positions=meta.positions, cos_sin=cs, wo_a=w.wo.wo_a.values,
        wo_a_scale=w.wo.wo_a.scale_mma, wo_b=w.wo.wo_b.values, wo_b_scale=w.wo.wo_b.scale_mma,
        out=out, scratch=scratch(wo, T))
    torch.cuda.synchronize()

    sel_ref = dbg["c4_selected"]
    for a, b in zip(selected.cpu(), sel_ref.cpu()):
        assert set(a.tolist()) == set(b.tolist())
    assert torch.equal(main_cache, dbg["main_kv_cache"])
    assert cos(query, dbg["query"]) > 0.99999
    assert cos(attn, dbg["attn_out"]) > 0.9999
    # The prototype itself is >= 0.9994 cosine per layer vs the official model.
    assert cos(out, ref) > 0.9995
