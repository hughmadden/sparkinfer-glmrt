"""cuteafd DSV4 mHC AOT programs versus the prepared b12x.norm.mhc path."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, cosine, has_checkpoint, prepare

EPS, HC_EPS, ITERS = 1.0e-6, 1.0e-6, 20
LAYER = 2


def _weights(hidden: int, prefix: str, device):
    """Real layer-2 Flash hc weights when mounted, else realistic random ones."""
    if hidden == 4096 and has_checkpoint():
        fn = checkpoint_tensor(f"layers.{LAYER}.hc_{prefix}_fn").float().contiguous()
        scale = checkpoint_tensor(f"layers.{LAYER}.hc_{prefix}_scale").float().contiguous()
        base = checkpoint_tensor(f"layers.{LAYER}.hc_{prefix}_base").float().contiguous()
        norm = checkpoint_tensor(f"layers.{LAYER}.{prefix}_norm.weight").contiguous()
        return fn, scale, base, norm
    gen = torch.Generator(device="cpu").manual_seed(hidden + len(prefix))
    fn = (torch.randn((24, 4 * hidden), generator=gen) * 0.02).to(device)
    scale = torch.tensor([0.08, 0.05, 0.3], device=device)
    base = (torch.randn((24,), generator=gen) * 0.5).to(device)
    norm = (1.0 + 0.1 * torch.randn((hidden,), generator=gen)).bfloat16().to(device)
    return fn, scale, base, norm


def _stream(rows: int, hidden: int, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn((rows, 4, hidden), generator=gen) * 0.7).bfloat16().to(device)


def _plan(operation: str, rows: int, hidden: int, device):
    from b12x.norm import mhc
    from b12x.preparation import FrozenMapping

    options = dict(operation=operation, output_mode="provided")
    if operation != "post":
        options.update(has_norm_weight=True, norm_weight_dtype="bfloat16", rms_eps=EPS,
                       hc_eps=HC_EPS, sinkhorn_iters=ITERS, norm_eps=EPS)
    if operation == "pre":
        options["expanded_residual"] = True
    plan = mhc.plan(mhc.Caps(device=device, max_tokens=rows, hidden_size=hidden),
                    invocation=FrozenMapping(options))
    return prepare(plan, f"test.mhc.{operation}.{hidden}.{rows}")


def _prepared_pre(residual, fn, scale, base, norm):
    from b12x.norm import mhc

    rows, _, hidden = residual.shape
    plan = _plan("pre", rows, hidden, residual.device)
    spec = plan.scratch_specs()[0]
    binding = mhc.bind(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=residual.device), tokens=rows,
        out=torch.empty_like(residual), y=torch.empty((rows, hidden), dtype=torch.bfloat16, device=residual.device),
        post=torch.empty((rows, 4), dtype=torch.float32, device=residual.device),
        comb=torch.empty((rows, 4, 4), dtype=torch.float32, device=residual.device),
    )
    return mhc.run_pre(residual, fn, scale, base, rms_eps=EPS, hc_eps=HC_EPS, sinkhorn_iters=ITERS,
                       norm_weight=norm, norm_eps=EPS, binding=binding)


def _prepared_post_pre(x, residual, prev_post, prev_comb, fn, scale, base, norm):
    from b12x.norm import mhc

    rows, _, hidden = residual.shape
    plan = _plan("post_pre", rows, hidden, residual.device)
    spec = plan.scratch_specs()[0]
    binding = mhc.bind(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=residual.device), tokens=rows,
        out=torch.empty_like(residual), y=torch.empty_like(x),
        post=torch.empty_like(prev_post), comb=torch.empty_like(prev_comb),
    )
    return mhc.run_post_pre(x, residual, prev_post, prev_comb, fn, scale, base, rms_eps=EPS,
                            hc_eps=HC_EPS, sinkhorn_iters=ITERS, norm_weight=norm, norm_eps=EPS,
                            binding=binding)


def _aot_pre(program, residual, fn, scale, base, norm, *, poison=True):
    rows, _, hidden = residual.shape
    dev = residual.device
    post = torch.full((rows, 4), float("nan"), device=dev)
    comb = torch.full((rows, 4, 4), float("nan"), device=dev)
    y = torch.full((rows, hidden), float("nan"), device=dev, dtype=torch.bfloat16)
    scratch = torch.full((program.scratch_bytes(rows)["scratch"] // 4,), float("nan"), device=dev)
    program.launch(residual, fn, scale, base, norm, post, comb, y, scratch, scalars=(rows,))
    return post, comb, y


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    from b12x.integration.cuteafd import FLASH, PRO, exportable_compilation
    from b12x.integration.cuteafd import dsv4_mhc

    with exportable_compilation():
        return {
            "pre": dsv4_mhc.compile_dsv4_mhc_pre_aot(FLASH),
            "post_pre_decode": dsv4_mhc.compile_dsv4_mhc_post_pre_aot(FLASH, max_rows=64),
            "post_pre_block_m": dsv4_mhc.compile_dsv4_mhc_post_pre_aot(FLASH, max_rows=4096),
            "post": dsv4_mhc.compile_dsv4_mhc_post_aot(FLASH),
            "head": dsv4_mhc.compile_dsv4_mhc_head_aot(FLASH),
            "pro_pre": dsv4_mhc.compile_dsv4_mhc_pre_aot(PRO),
            "pro_head": dsv4_mhc.compile_dsv4_mhc_head_aot(PRO),
        }


@pytest.mark.parametrize("rows", [1, 7, 130])
def test_pre_matches_prepared_bitwise_across_live_rows(programs, rows):
    device = require_b12x()
    fn, scale, base, norm = _weights(4096, "attn", device)
    residual = _stream(rows, 4096, 100 + rows, device)
    _, post_ref, comb_ref, y_ref = _prepared_pre(residual, fn, scale, base, norm)
    post, comb, y = _aot_pre(programs["pre"], residual, fn, scale, base, norm)
    torch.cuda.synchronize()
    assert_bitwise(post, post_ref, "post")
    assert_bitwise(comb, comb_ref, "comb")
    assert_bitwise(y, y_ref, "y")


def _mix_reference64(stream, fn, scale, base):
    """FP64 mHC post/comb coefficients of ``stream`` (Sinkhorn as in model.py)."""
    flat = stream.flatten(1).double()
    fn, scale, base = fn.double(), scale.double(), base.double()
    mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + EPS)
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax(mixes[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4), dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    return post, comb


def _run_post_pre(program, x, residual, prev_post, prev_comb, fn, scale, base, norm):
    rows, _, hidden = residual.shape
    dev = residual.device
    out = torch.empty_like(residual)
    post = torch.full((rows, 4), float("nan"), device=dev)
    comb = torch.full((rows, 4, 4), float("nan"), device=dev)
    y = torch.empty_like(x)
    scratch = torch.full((program.scratch_bytes(rows)["scratch"] // 4,), float("nan"), device=dev)
    program.launch(x, residual, prev_post, prev_comb, fn, scale, base, norm, out, post, comb, y,
                   scratch, scalars=(rows,))
    return out, post, comb, y


@pytest.mark.parametrize(("route", "rows", "exact"), [
    ("post_pre_decode", 1, True), ("post_pre_decode", 5, True), ("post_pre_decode", 64, True),
    ("post_pre_block_m", 100, True), ("post_pre_block_m", 300, True),
    # The prepared plan switches to its TF32 tensor-core projection at >= 384 rows.
    ("post_pre_block_m", 512, False),
])
def test_post_pre_matches_prepared(programs, route, rows, exact):
    device = require_b12x()
    fn_a, scale_a, base_a, norm_a = _weights(4096, "attn", device)
    fn, scale, base, norm = _weights(4096, "ffn", device)
    residual = _stream(rows, 4096, 200 + rows, device)
    prev_post, prev_comb, _ = _aot_pre(programs["pre"], residual, fn_a, scale_a, base_a, norm_a)
    gen = torch.Generator(device="cpu").manual_seed(rows)
    x = (torch.randn((rows, 4096), generator=gen) * 0.3).bfloat16().to(device)
    ref = _prepared_post_pre(x, residual, prev_post, prev_comb, fn, scale, base, norm)
    got = _run_post_pre(programs[route], x, residual, prev_post, prev_comb, fn, scale, base, norm)
    torch.cuda.synchronize()
    names = ("residual_out", "post", "comb", "y")
    assert_bitwise(got[0], ref[0], "residual_out")
    for name, a, e in zip(names[1:], got[1:], ref[1:]):
        if exact:
            assert_bitwise(a, e, name)
        elif name == "y":
            # The prepared TF32 projection perturbs the mixes; the AOT route
            # keeps the FP32 projection (checked against FP64 below).
            assert float((a.float() - e.float()).abs().max()) <= 0.02 * float(e.float().abs().max()), name
            assert cosine(a, e) > 0.99999, name
        else:
            torch.testing.assert_close(a, e, rtol=2e-3, atol=2e-3)
    if not exact:
        post64, comb64 = _mix_reference64(ref[0], fn, scale, base)
        for a, e, r in ((got[1], ref[1], post64), (got[2], ref[2], comb64)):
            assert float((a.double() - r).abs().max()) <= max(float((e.double() - r).abs().max()), 2e-4)


@pytest.mark.parametrize("rows", [1, 33])
def test_post_matches_prepared_bitwise(programs, rows):
    from b12x.norm import mhc

    device = require_b12x()
    residual = _stream(rows, 4096, 300 + rows, device)
    fn, scale, base, norm = _weights(4096, "ffn", device)
    prev_post, prev_comb, _ = _aot_pre(programs["pre"], residual, fn, scale, base, norm)
    x = _stream(rows, 4096, 301, device)[:, 0].contiguous()
    plan = _plan("post", rows, 4096, device)
    expected = mhc.run_post(x, residual, prev_post, prev_comb, plan=plan, out=torch.empty_like(residual))
    out = torch.full_like(residual, float("nan"))
    programs["post"].launch(x, residual, prev_post, prev_comb, out, scalars=(rows,))
    torch.cuda.synchronize()
    assert_bitwise(out, expected, "out")


def _head_weights(hidden, device):
    if hidden == 4096 and has_checkpoint():
        return (checkpoint_tensor("hc_head_fn").float().contiguous(),
                checkpoint_tensor("hc_head_scale").float().contiguous(),
                checkpoint_tensor("hc_head_base").float().contiguous(),
                checkpoint_tensor("norm.weight").contiguous())
    gen = torch.Generator(device="cpu").manual_seed(hidden)
    return ((torch.randn((4, 4 * hidden), generator=gen) * 0.02).to(device),
            torch.tensor([0.1], device=device), (torch.randn((4,), generator=gen)).to(device),
            (1 + 0.1 * torch.randn((hidden,), generator=gen)).bfloat16().to(device))


@pytest.mark.parametrize(("key", "hidden", "rows"), [("head", 4096, 1), ("head", 4096, 37), ("pro_head", 7168, 5)])
def test_head_matches_prepared_head(programs, key, hidden, rows):
    from b12x.norm import mhc

    device = require_b12x()
    fn, scale, base, norm = _head_weights(hidden, device)
    residual = _stream(rows, hidden, 400 + rows, device)
    expected = mhc.run_head(residual, fn, scale, base, norm, rms_eps=EPS, hc_eps=HC_EPS, norm_eps=EPS,
                            out=torch.empty((rows, hidden), dtype=torch.bfloat16, device=device))
    out = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16, device=device)
    programs[key].launch(residual, fn, scale, base, norm, None, out, scalars=(rows,))
    torch.cuda.synchronize()
    # Different FP32 reduction order than Triton: at most one BF16 ulp apart.
    torch.testing.assert_close(out, expected, rtol=2.0 ** -7, atol=1e-5)
    assert float((out != expected).float().mean()) < 0.01
    assert cosine(out, expected) > 0.99999


def test_pro_pre_matches_prepared_bitwise(programs):
    device = require_b12x()
    fn, scale, base, norm = _weights(7168, "attn", device)
    residual = _stream(9, 7168, 500, device)
    _, post_ref, comb_ref, y_ref = _prepared_pre(residual, fn, scale, base, norm)
    post, comb, y = _aot_pre(programs["pro_pre"], residual, fn, scale, base, norm)
    torch.cuda.synchronize()
    assert_bitwise(post, post_ref, "post")
    assert_bitwise(comb, comb_ref, "comb")
    assert_bitwise(y, y_ref, "y")


def test_pre_replays_in_cuda_graph(programs):
    device = require_b12x()
    fn, scale, base, norm = _weights(4096, "attn", device)
    rows = 3
    residual = _stream(rows, 4096, 600, device)
    post = torch.empty((rows, 4), device=device)
    comb = torch.empty((rows, 4, 4), device=device)
    y = torch.empty((rows, 4096), device=device, dtype=torch.bfloat16)
    scratch = torch.empty((programs["pre"].scratch_bytes(rows)["scratch"] // 4,), device=device)
    args = (residual, fn, scale, base, norm, post, comb, y, scratch)
    programs["pre"].launch(*args, scalars=(rows,))
    torch.cuda.synchronize()
    eager = y.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        programs["pre"].launch(*args, scalars=(rows,))
    y.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(y, eager, "graph y")


@pytest.mark.parametrize("key", ["pre", "post_pre_decode", "post_pre_block_m", "post", "head"])
def test_export_to_c_signature(programs, key, tmp_path):
    info = check_export(programs[key], tmp_path, f"dsv4_mhc_{key}")
    assert info["argument_count"] == len(programs[key].operands) + 2
