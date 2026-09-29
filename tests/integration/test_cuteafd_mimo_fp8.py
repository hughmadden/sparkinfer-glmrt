"""cuteafd MiMo V2 Flash FP8 decode programs vs their BF16 programs.

``fp8=True`` producer / o / ffn programs and the FP8 LM head read E4M3 copies
with per-row x 128-K FP32 scales for steps of ``rows <= fp8_rows`` (<= 32; head <= 16):
the checkpoint's own E4M3 bytes with its block grids expanded per row
(q/k/v, dense gate/up/down: the BF16 programs' dequantized weights exactly),
or a BF16 weight quantized per row and 128-K block (o_proj, lm_head).
Checked: FP8 rows against the BF16 program over the same dequantized weight
(cosine >= 0.99999), rows above ``fp8_rows`` or ``fp8_rows = 0`` bitwise
equal to the BF16 program.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._mimo import cos_sin, golden_rows, raw_tensor, scale_row_of, tensor

COS = 0.99999


@pytest.fixture(scope="module")
def g():
    require_b12x()
    from b12x.integration.cuteafd import MIMO_V2_FLASH

    return MIMO_V2_FLASH


_PROGRAMS: dict = {}


def _program(program_kind, **kw):
    key = (program_kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import MIMO_V2_FLASH, mimo_attention, mimo_ffn

        fn = {"o": mimo_attention.compile_mimo_o_aot, "producer": mimo_attention.compile_mimo_producer_aot,
              "ffn": mimo_ffn.compile_mimo_ffn_aot, "head": mimo_attention.compile_mimo_head_fp8_aot}[program_kind]
        _PROGRAMS[key] = fn(MIMO_V2_FLASH, **kw)
    return _PROGRAMS[key]


def _scratch(program, rows):
    size = program.scratch_bytes(rows).get("scratch", 0)
    return torch.empty(max(size, 1024), dtype=torch.uint8, device="cuda")


def checkpoint_fp8(names):
    """E4M3 rows of ``names`` concatenated, with per-row x 128-K FP32 scales."""
    values, scales = [], []
    for name in names:
        w = raw_tensor(name).cuda()
        s = raw_tensor(name.removesuffix("weight") + "weight_scale_inv").cuda().float()
        values.append(w)
        scales.append(s[scale_row_of(w.shape[0], s.shape[0])])
    return torch.cat(values).contiguous(), torch.cat(scales).contiguous()


def quantize_rows(w):
    """BF16 [N, K] -> E4M3 and FP32 [N, K/128] scales (amax / 448 per row and K block)."""
    n, k = w.shape
    blocks = w.float().view(n, k // 128, 128)
    s = (blocks.abs().amax(-1) / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    s = torch.where(blocks.abs().amax(-1) > 0, s, torch.ones_like(s))
    q = (blocks / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k)
    return q.contiguous(), s.contiguous()


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("rows", [1, 5, 16, 17, 32, 64])
def test_mimo_producer_fp8(g, layer, rows):
    kind = "full" if layer == 0 else "swa"
    names = [f"model.layers.{layer}.self_attn.{p}_proj.weight" for p in "qkv"]
    bf16 = torch.cat([tensor(n) for n in names]).contiguous()
    q8, s8 = checkpoint_fp8(names)
    x = golden_rows(max(layer - 1, 0), rows, seed=1)
    positions = torch.arange(rows, device="cuda") + 3
    slots = torch.arange(rows, device="cuda")
    table = cos_sin(int(positions.max()) + 1, kind)
    r = g.record_elems(kind)

    def run(program, *weights, scalars):
        cache = torch.zeros((rows, r), dtype=torch.bfloat16, device="cuda")
        query = torch.empty((rows, 64, 192), dtype=torch.bfloat16, device="cuda")
        program.launch(x, positions, slots, table, *weights, cache, query, _scratch(program, rows), scalars=scalars)
        torch.cuda.synchronize()
        return torch.cat([query.view(rows, -1), cache], 1)

    ref = run(_program("producer", kind=kind, max_rows=64), bf16, scalars=(rows,))
    fp8 = _program("producer", kind=kind, max_rows=64, fp8=True)
    got = run(fp8, bf16, q8, s8, scalars=(rows, 16))
    off = run(fp8, bf16, q8, s8, scalars=(rows, 0))
    c = cosine(got, ref)
    print(f"mimo_{kind}_producer fp8 rows={rows}: cosine {c:.7f}")
    assert torch.equal(off, ref)
    if rows > 16:
        assert torch.equal(got, ref)
    assert c >= COS
    # fp8_rows 32: steps of 17-32 rows read the E4M3 copy through the two-tile GEMV.
    wide = run(fp8, bf16, q8, s8, scalars=(rows, 32))
    if rows <= 16:
        assert torch.equal(wide, got)
    elif rows <= 32:
        cw = cosine(wide, ref)
        print(f"mimo_{kind}_producer fp8 rows={rows} fp8_rows=32: cosine {cw:.7f}")
        assert cw >= COS
    else:
        assert torch.equal(wide, ref)


@pytest.mark.parametrize("rows,fp8_rows", [(1, 16), (16, 16), (17, 16), (17, 32), (32, 32)])
def test_mimo_o_fp8(g, rows, fp8_rows):
    w = tensor("model.layers.1.self_attn.o_proj.weight")
    q8, s8 = quantize_rows(w)
    deq = (q8.float().view(4096, -1, 128) * s8[..., None]).view(4096, -1).bfloat16()
    attn = (golden_rows(2, rows * 2, seed=3).view(rows, 8192) * 0.2).contiguous()
    out = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    _program("o", max_rows=64, fp8=True).launch(attn, w, q8, s8, out, scalars=(rows, fp8_rows))
    ref = torch.empty_like(out)
    _program("o", max_rows=64).launch(attn, deq if rows <= fp8_rows else w, ref, scalars=(rows,))
    torch.cuda.synchronize()
    c = cosine(out, ref)
    print(f"mimo_o fp8 rows={rows} fp8_rows={fp8_rows}: cosine {c:.7f} vs BF16 program over the dequantized weight")
    assert c >= COS


@pytest.mark.parametrize("rows,fp8_rows", [(1, 16), (16, 16), (24, 32), (32, 32)])
def test_mimo_ffn_fp8(g, rows, fp8_rows):
    names = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    gate_up = torch.cat([tensor(n) for n in names]).contiguous()
    down = tensor("model.layers.0.mlp.down_proj.weight")
    gu8, gus = checkpoint_fp8(names)
    d8, ds = checkpoint_fp8(["model.layers.0.mlp.down_proj.weight"])
    x = golden_rows(0, rows, seed=2)
    out = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    ref = torch.empty_like(out)
    p8, p = _program("ffn", max_rows=64, fp8=True), _program("ffn", max_rows=64)
    p8.launch(x, gate_up, gu8, gus, down, d8, ds, out, _scratch(p8, rows), scalars=(rows, fp8_rows))
    p.launch(x, gate_up, down, ref, _scratch(p, rows), scalars=(rows,))
    torch.cuda.synchronize()
    c = cosine(out, ref)
    print(f"mimo_ffn fp8 rows={rows} fp8_rows={fp8_rows}: cosine {c:.7f}")
    assert c >= COS


@pytest.mark.parametrize("rows", [1, 16])
def test_mimo_head_fp8(g, rows):
    w = raw_tensor("lm_head.weight").cuda()
    q8, s8 = quantize_rows(w)
    deq = (q8.float().view(w.shape[0], -1, 128) * s8[..., None]).view(w.shape)
    x = golden_rows(46, rows, seed=4)
    logits = torch.empty((rows, w.shape[0]), dtype=torch.float32, device="cuda")
    _program("head").launch(x, q8, s8, logits, scalars=(rows,))
    ref = x.float() @ deq.bfloat16().float().t()
    torch.cuda.synchronize()
    c = cosine(logits, ref)
    same = float((logits.argmax(-1) == ref.argmax(-1)).float().mean())
    print(f"mimo_head_fp8 rows={rows}: cosine {c:.7f} argmax agreement {same:.3f}")
    assert c >= COS and same == 1.0
