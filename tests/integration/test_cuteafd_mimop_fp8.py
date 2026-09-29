"""cuteafd MiMo V2.6 Pro (``mimop``) programs over real V2.6 Pro weights.

The fused, TP8-interleaved ``qkv_proj`` (FP8, one 128x128 grid per row shard)
in the coordinator's layout (``[q; k; v]`` with every 192-row key zero-padded
to 256 rows): the BF16 producer against a torch reference of the QKV
projection, and the ``fp8=True`` producer (per-row x 128-K scales, the
shard grids expanded) against the BF16 program at 1/5/16 rows; the FP8 o and
dense FFN programs against the BF16 programs over the dequantized weights.
Skipped when the snapshot is not mounted.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

COS = 0.99999
SNAPSHOTS = glob.glob("/mnt/sparknest/hf-home/hub/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/*")
_FILES: dict = {}
_PROGRAMS: dict = {}


def raw(name):
    from safetensors import safe_open

    snap = Path(SNAPSHOTS[0])
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    shard = index[name]
    if shard not in _FILES:
        _FILES[shard] = safe_open(str(snap / shard), framework="pt", device="cpu")
    return _FILES[shard].get_tensor(name).cuda()


@pytest.fixture(scope="module")
def g():
    require_b12x()
    if not SNAPSHOTS:
        pytest.skip("MiMo V2.6 Pro snapshot not mounted")
    from b12x.integration.cuteafd import MIMO_V26_PRO

    torch.backends.cuda.matmul.allow_tf32 = False
    return MIMO_V26_PRO


def program(program_kind, **kw):
    key = (program_kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import MIMO_V26_PRO, mimo_attention, mimo_ffn

        fn = {"o": mimo_attention.compile_mimo_o_aot, "producer": mimo_attention.compile_mimo_producer_aot,
              "ffn": mimo_ffn.compile_mimo_ffn_aot}[program_kind]
        _PROGRAMS[key] = fn(MIMO_V26_PRO, **kw)
    return _PROGRAMS[key]


def scratch(p, rows):
    return torch.empty(max(p.scratch_bytes(rows).get("scratch", 0), 1024), dtype=torch.uint8, device="cuda")


def fused_qkv(g, layer):
    """(BF16 [W, H], E4M3 [W, H], FP32 row scales [W, H/128]) in the padded layout."""
    w = raw(f"model.layers.{layer}.self_attn.qkv_proj.weight")
    s = raw(f"model.layers.{layer}.self_attn.qkv_proj.weight_scale_inv").float()
    tp, heads, kv = 8, g.heads, 8
    q, k, v = heads // tp * 192, kv // tp * 192, kv // tp * 128
    blocks = [q // 128, -(-k // 128), v // 128]
    stride = g.qkv_k_stride
    width = heads * 192 + kv * (stride + 128)
    values = torch.zeros((width, w.shape[1]), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.zeros((width, w.shape[1] // 128), dtype=torch.float32, device="cuda")
    for shard in range(tp):
        src, grid = shard * (q + k + v), shard * sum(blocks)
        for part, (rows, dest) in enumerate([(q, shard * q), (k, heads * 192 + shard * stride),
                                             (v, heads * 192 + kv * stride + shard * v)]):
            values[dest:dest + rows] = w[src:src + rows]
            row_grid = torch.arange(rows, device="cuda") // 128 + grid
            scales[dest:dest + rows] = s[row_grid]
            src += rows
            grid += blocks[part]
    bf16 = (values.float().view(width, -1, 128) * scales[..., None]).view(width, -1).bfloat16()
    return bf16.contiguous(), values.contiguous(), scales.contiguous()


def rows_of(n, h, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, h, generator=gen, device="cuda") * 0.5).bfloat16()


def cos_sin(p, theta):
    inv = 1.0 / theta ** (torch.arange(0, 64, 2, device="cuda").float() / 64)
    ang = torch.arange(p, device="cuda").float()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], 1).contiguous()


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("rows", [1, 5, 16, 64])
def test_mimop_producer(g, layer, rows):
    kind = "full" if layer == 0 else "swa"
    bf16, q8, s8 = fused_qkv(g, layer)
    x = rows_of(rows, g.hidden, 1)
    positions = torch.arange(rows, device="cuda") + 3
    slots = torch.arange(rows, device="cuda")
    table = cos_sin(int(positions.max()) + 1, g.rope_theta(kind))
    r = g.record_elems(kind)

    def run(p, *weights, scalars):
        cache = torch.zeros((rows, r), dtype=torch.bfloat16, device="cuda")
        query = torch.empty((rows, g.heads, 192), dtype=torch.bfloat16, device="cuda")
        p.launch(x, positions, slots, table, *weights, cache, query, scratch(p, rows), scalars=scalars)
        torch.cuda.synchronize()
        return query, cache

    query, cache = run(program("producer", kind=kind, max_rows=64), bf16, scalars=(rows,))
    # Reference: the pass-through dims of q and k and the scaled values.
    qkv = (x.float() @ bf16.float().t()).bfloat16()
    q_ref = qkv[:, :g.heads * 192].view(rows, g.heads, 192)
    k_ref = torch.stack([qkv[:, g.heads * 192 + h * g.qkv_k_stride:][:, :192] for h in range(8)], 1)
    v_ref = qkv[:, g.heads * 192 + 8 * g.qkv_k_stride:].view(rows, 8, 128)
    k_got = cache[:, :8 * 192].view(rows, 8, 192)
    v_got = cache[:, 8 * 192:].view(rows, 8, 128)
    c = [cosine(query[..., 64:], q_ref[..., 64:]), cosine(k_got[..., 64:], k_ref[..., 64:]),
         cosine(v_got, (v_ref.float() * g.v_scale).bfloat16())]
    fp8 = program("producer", kind=kind, max_rows=64, fp8=True)
    q2, c2 = run(fp8, bf16, q8, s8, scalars=(rows, 16))
    c8 = cosine(torch.cat([q2.view(rows, -1), c2], 1), torch.cat([query.view(rows, -1), cache], 1))
    print(f"mimop_{kind}_producer rows={rows}: q/k/v vs torch {c[0]:.7f} {c[1]:.7f} {c[2]:.7f}; fp8 vs bf16 {c8:.7f}")
    assert min(c) >= COS and c8 >= COS


@pytest.mark.parametrize("rows", [1, 16])
def test_mimop_o_ffn_fp8(g, rows):
    from .test_cuteafd_mimo_fp8 import quantize_rows

    w = raw("model.layers.1.self_attn.o_proj.weight")
    q8, s8 = quantize_rows(w)
    deq = (q8.float().view(w.shape[0], -1, 128) * s8[..., None]).view(w.shape).bfloat16()
    attn = rows_of(rows, w.shape[1], 3)
    out, ref = (torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    program("o", max_rows=64, fp8=True).launch(attn, w, q8, s8, out, scalars=(rows, 16))
    program("o", max_rows=64).launch(attn, deq, ref, scalars=(rows,))
    names = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    parts = [(raw(n), raw(n + "_scale_inv").float()) for n in names + ["model.layers.0.mlp.down_proj.weight"]]
    expand = lambda w, s: (w, s[torch.arange(w.shape[0], device="cuda") // 128].contiguous())  # noqa: E731
    gu8 = torch.cat([p[0] for p in parts[:2]]).contiguous()
    gus = torch.cat([expand(*p)[1] for p in parts[:2]]).contiguous()
    d8, ds = expand(*parts[2])
    dq = lambda w, s: (w.float().view(w.shape[0], -1, 128) * s[..., None]).view(w.shape).bfloat16()  # noqa: E731
    gate_up, down = dq(gu8, gus).contiguous(), dq(d8, ds).contiguous()
    x = rows_of(rows, g.hidden, 2)
    f_out, f_ref = (torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    p8, p = program("ffn", max_rows=64, fp8=True), program("ffn", max_rows=64)
    p8.launch(x, gate_up, gu8, gus, down, d8, ds, f_out, scratch(p8, rows), scalars=(rows, 16))
    p.launch(x, gate_up, down, f_ref, scratch(p, rows), scalars=(rows,))
    torch.cuda.synchronize()
    co, cf = cosine(out, ref), cosine(f_out, f_ref)
    print(f"mimop o fp8 rows={rows}: {co:.7f}; ffn fp8 {cf:.7f}")
    assert co >= COS and cf >= COS
