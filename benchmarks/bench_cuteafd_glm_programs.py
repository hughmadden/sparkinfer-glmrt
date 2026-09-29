#!/usr/bin/env python3
"""Time the cuteafd GLM 5.x AOT programs: decode programs (m64) at 1/16/64
live rows, prefill programs (m4096) at 4096 rows. Median of CUDA-event
timings, L2 flushed before every launch (weights L2-cold), the GPU held
busy while the host enqueues (device time, no Python launch gaps), synthetic BF16
weights of the real shapes. Attention: 2048 selected slots per row; the
decode index top-k scores ``--decode-context`` cached rows per query row,
the prefill top-k is a 4096-row first chunk (causal lengths 1..4096).

Decode programs also run with FP8 weight operands (``*_m64 fp8``): E4M3
weights and FP32 block scales next to the BF16 copy.

  python benchmarks/bench_cuteafd_glm_programs.py [--iters 30] [--decode-context 8192]
"""

from __future__ import annotations

import argparse

import torch


def time_us(fn, iters: int, flush: torch.Tensor) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.max()  # read-only L2 flush: no dirty lines to write back inside fn
        torch.cuda._sleep(3_000_000)  # covers the host-side launch of fn
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--decode-context", type=int, default=8192)
    args = parser.parse_args()

    from b12x.integration.cuteafd import GLM53 as g
    from b12x.integration.cuteafd import glm_attention as attn
    from b12x.integration.cuteafd import glm_ffn as ffn
    from b12x.integration.cuteafd import glm_indexer as idx
    from b12x.integration.cuteafd import glm_sparse_mla as mla
    from b12x.integration.cuteafd.dsv4_ffn import expert_input_quant_grid

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    flush = torch.zeros(512 << 20, dtype=torch.uint8, device=dev)

    def bf16(*shape, scale=0.02):
        return (torch.randn(shape, generator=gen, device=dev) * scale).bfloat16()

    h, n, q = g.hidden, g.heads, g.q_lora_rank
    W = dict(
        w_qkv_a=bf16(g.qkv_a_width, h), q_a_norm=bf16(q, scale=1.0), kv_a_norm=bf16(512, scale=1.0),
        w_q_b=bf16(n * 256, q), w_uk=bf16(n, 512, 192), w_uv=bf16(n, 256, 512), w_o=bf16(h, n * 256),
        w_iq=bf16(32 * 128, q), w_ik=bf16(160, h), k_norm_w=bf16(128, scale=1.0), k_norm_b=bf16(128),
        norm=bf16(h, scale=1.0), router=bf16(256, h),
        ffn2048=(bf16(4096, h), bf16(h, 2048)), ffn12288=(bf16(24576, h), bf16(h, 12288)),
    )

    def fp8(w):
        rows, cols = w.shape
        scale = torch.rand(((rows + 127) // 128, cols // 128), generator=gen, device=dev) * 1e-3 + 1e-4
        return (w.float() / scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)).to(
            torch.float8_e4m3fn), scale

    F = {name: fp8(W[name]) for name in ("w_qkv_a", "w_q_b", "w_iq", "w_o")}
    F["ffn2048"] = (fp8(W["ffn2048"][0]), fp8(W["ffn2048"][1]))
    F["ffn12288"] = (fp8(W["ffn12288"][0]), fp8(W["ffn12288"][1]))
    max_ctx = max(args.decode_context, 4096) + 64
    pages = max_ctx // 64 + 2
    kv_cache = torch.randint(0, 120, (pages, 64 * 656), dtype=torch.uint8, device=dev, generator=gen)
    kv_cache.view(pages, 64, 656)[..., 512:528] = torch.tensor([0, 0, 128, 59] * 4, dtype=torch.uint8, device=dev)
    index_cache = torch.randint(0, 120, (pages, 8448), dtype=torch.uint8, device=dev, generator=gen)
    index_cache[:, 8192:].view(torch.float32).fill_(0.01)
    cs = torch.randn((max_ctx, 64), device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    progs = {}

    def prog(name, fn):
        if name not in progs:
            progs[name] = fn()
        return progs[name]

    results = []
    for cap, rows_list in ((64, (1, 16, 64)), (4096, (4096,))):
        route = "decode" if cap == 64 else "prefill"
        p_norm = prog("norm", lambda: ffn.compile_glm_norm_aot(g))
        p_router = prog("router", lambda: ffn.compile_glm_router_scores_aot(g))
        p_quant = prog("quant", lambda: ffn.compile_glm_expert_input_quant_aot(g))
        p_prod = prog(f"prod{cap}", lambda: attn.compile_glm_producer_aot(g, max_rows=cap))
        p_iprod = prog(f"iprod{cap}", lambda: attn.compile_glm_index_producer_aot(g, max_rows=cap))
        p_o = prog(f"o{cap}", lambda: attn.compile_glm_o_aot(g, max_rows=cap))
        p_f2 = prog(f"f2{cap}", lambda: ffn.compile_glm_ffn_aot(g, inter=2048, max_rows=cap))
        p_f12 = prog(f"f12{cap}", lambda: ffn.compile_glm_ffn_aot(g, inter=12288, max_rows=cap))
        p_topk = prog(f"topk{cap}", lambda: idx.compile_glm_index_topk_aot(
            g, max_rows=cap, max_pages=131072 // 64, mode=route))
        p_mla = prog(f"mla{cap}", lambda: mla.compile_glm_sparse_mla_aot(g, route=route, max_rows=cap))
        for rows in rows_list:
            x = bf16(rows, h, scale=1.0)
            res = x.clone()
            out = torch.empty_like(x)
            pos = torch.arange(rows, dtype=torch.int64, device=dev) + (args.decode_context - rows if cap == 64 else 0)
            slots = pos.clone()
            query = torch.empty((rows, n, 576), dtype=torch.bfloat16, device=dev)
            q_resid = bf16(rows, q, scale=1.0)
            q8 = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device=dev)
            hw = torch.empty((rows, 32), dtype=torch.float32, device=dev)
            latent = bf16(rows, n, 512, scale=1.0)
            logits = torch.empty((rows, 256), dtype=torch.float32, device=dev)
            wire = torch.empty((rows, h + h // 32), dtype=torch.uint8, device=dev)
            sc = lambda p: torch.zeros(p.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device=dev)  # noqa: E731
            s_prod, s_iprod, s_o, s_f2, s_f12 = sc(p_prod), sc(p_iprod), sc(p_o), sc(p_f2), sc(p_f12)
            s_topk, s_mla = sc(p_topk), sc(p_mla)
            ctx = args.decode_context if cap == 64 else 4096
            width = ctx // 64
            table = torch.arange(width, dtype=torch.int32, device=dev)
            if cap == 64:
                lengths = torch.full((rows,), ctx, dtype=torch.int32, device=dev)
                page_table, stride = table[None].expand(rows, -1).contiguous(), width
            else:
                lengths = torch.arange(1, rows + 1, dtype=torch.int32, device=dev)
                page_table, stride = table, 0
            sel = torch.stack([torch.randperm(ctx, device=dev, generator=gen)[:2048] for _ in range(min(rows, 64))])
            indices = sel.repeat(-(-rows // sel.shape[0]), 1)[:rows].int().contiguous()
            full = torch.full((rows,), 2048, dtype=torch.int32, device=dev)
            topk_out = torch.empty((rows, 2048), dtype=torch.int32, device=dev)
            cases = [
                ("glm_norm (deltas=1)", lambda: p_norm.launch(res, x, x, W["norm"], out, scalars=(rows, 1))),
                (f"glm_producer_m{cap}", lambda: p_prod.launch(
                    x, pos, slots, cs, W["w_qkv_a"], W["q_a_norm"], W["kv_a_norm"], W["w_q_b"], W["w_uk"],
                    kv_cache, query, q_resid, s_prod, scalars=(rows,))),
                (f"glm_index_producer_m{cap}", lambda: p_iprod.launch(
                    x, q_resid, pos, slots, cs, W["w_iq"], W["w_ik"], W["k_norm_w"], W["k_norm_b"],
                    index_cache, q8, hw, s_iprod, scalars=(rows,))),
                (f"glm_index_topk_{route}_m{cap}", lambda: p_topk.launch(
                    q8, hw, index_cache, page_table, lengths, topk_out, s_topk, scalars=(rows, width, stride))),
                (f"glm_sparse_mla_{route}_m{cap}", lambda: p_mla.launch(
                    query, kv_cache, indices, full, latent, s_mla, scalars=(rows,))),
                (f"glm_o_m{cap}", lambda: p_o.launch(latent, W["w_uv"], W["w_o"], out, s_o, scalars=(rows,))),
                (f"glm_ffn_i2048_m{cap}", lambda: p_f2.launch(x, *W["ffn2048"], out, s_f2, scalars=(rows,))),
                (f"glm_ffn_i12288_m{cap}", lambda: p_f12.launch(x, *W["ffn12288"], out, s_f12, scalars=(rows,))),
                ("glm_router_scores", lambda: p_router.launch(x, W["router"], logits, scalars=(rows,))),
                ("glm_expert_input_quant", lambda: p_quant.launch(
                    x, wire, wire[:, h:], wire, scalars=(rows, expert_input_quant_grid(h, rows, sms)))),
            ]
            # Producers first so the q8/query operands of later cases are real.
            if cap == 64:
                q8p = prog("prod64f", lambda: attn.compile_glm_producer_aot(g, max_rows=64, fp8=True))
                i8p = prog("iprod64f", lambda: attn.compile_glm_index_producer_aot(g, max_rows=64, fp8=True))
                o8p = prog("o64f", lambda: attn.compile_glm_o_aot(g, max_rows=64, fp8=True))
                f2p = prog("f264f", lambda: ffn.compile_glm_ffn_aot(g, inter=2048, max_rows=64, fp8=True))
                f12p = prog("f1264f", lambda: ffn.compile_glm_ffn_aot(g, inter=12288, max_rows=64, fp8=True))
                (gu2, gu2s), (d2, d2s) = F["ffn2048"]
                (gu12, gu12s), (d12, d12s) = F["ffn12288"]
                cases += [
                    ("glm_producer_m64 fp8", lambda: q8p.launch(
                        x, pos, slots, cs, W["w_qkv_a"], *F["w_qkv_a"], W["q_a_norm"], W["kv_a_norm"], W["w_q_b"],
                        *F["w_q_b"], W["w_uk"], kv_cache, query, q_resid, s_prod, scalars=(rows,))),
                    ("glm_index_producer_m64 fp8", lambda: i8p.launch(
                        x, q_resid, pos, slots, cs, W["w_iq"], *F["w_iq"], W["w_ik"], W["k_norm_w"],
                        W["k_norm_b"], index_cache, q8, hw, s_iprod, scalars=(rows,))),
                    ("glm_o_m64 fp8", lambda: o8p.launch(latent, W["w_uv"], W["w_o"], *F["w_o"], out, s_o,
                                                        scalars=(rows,))),
                    ("glm_ffn_i2048_m64 fp8", lambda: f2p.launch(x, W["ffn2048"][0], gu2, gu2s, W["ffn2048"][1],
                                                                d2, d2s, out, s_f2, scalars=(rows,))),
                    ("glm_ffn_i12288_m64 fp8", lambda: f12p.launch(
                        x, W["ffn12288"][0], gu12, gu12s, W["ffn12288"][1], d12, d12s, out, s_f12,
                        scalars=(rows,))),
                ]
            for name, fn in cases:
                results.append((name, rows, time_us(fn, args.iters, flush)))
    names = list(dict.fromkeys(name.replace("_m4096", "").replace("_m64", "").replace("_decode", "")
                               .replace("_prefill", "") for name, _, _ in results))
    table = {}
    for name, rows, us in results:
        key = name.replace("_m4096", "").replace("_m64", "").replace("_decode", "").replace("_prefill", "")
        table.setdefault(key, {})[rows] = us
    print(f"{'program':28s} {'M=1':>8s} {'M=16':>8s} {'M=64':>8s} {'M=4096':>9s}   (us, L2-cold)")
    for key in names:
        row = table[key]
        print(f"{key:28s} " + " ".join(f"{row.get(m, float('nan')):8.1f}" for m in (1, 16, 64))
              + f" {row.get(4096, float('nan')):9.1f}")


if __name__ == "__main__":
    main()
