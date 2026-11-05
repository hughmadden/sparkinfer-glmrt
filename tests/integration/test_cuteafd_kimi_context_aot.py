"""Owned-page Kimi MLA: full-attention oracles, high IDs and graph replay."""
from functools import lru_cache
from pathlib import Path
import os
import math

import pytest
import torch

from ..conftest import require_b12x


@lru_cache(None)
def programs(cache, begin):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_mla_partial_aot
    with exportable_compilation():
        p = compile_kimi_mla_partial_aot(KimiMlaGeometry(kv_cache=cache), max_rows=16,
                                       max_context=1024, head_begin=begin)
    if os.environ.get("K1_EXPORT_DIR"):
        target = Path(os.environ["K1_EXPORT_DIR"])
        target.mkdir(parents=True, exist_ok=True)
        stem = f"kimi_context_{cache}_h{begin}"
        p.export_to_c(str(target), stem, stem)
        validate_exported_header(p, target / f"{stem}.h", stem)
    return p


def records(cache, n):
    values = (torch.randn((n, 576), device="cuda") * .25).bfloat16()
    if cache == "bf16":
        return values, values.float()
    raw = torch.zeros((n, 656), device="cuda", dtype=torch.uint8)
    latent = values[:, :512].to(torch.float8_e4m3fn)
    raw[:, :512] = latent.view(torch.uint8)
    raw[:, 512:528] = torch.ones((n, 4), device="cuda").view(torch.uint8)
    raw[:, 528:] = values[:, 512:].contiguous().view(torch.uint8)
    # The compact reader widens to BF16 before its tensor-core arithmetic.
    return raw, torch.cat((latent.bfloat16().float(), values[:, 512:].float()), dim=1)


def test_partial_compile_contract(monkeypatch):
    from b12x.integration.cuteafd import kimi_mla as k
    calls = []
    monkeypatch.setattr(k, "compile_program", lambda launch, **kw: calls.append((launch, kw)) or kw)
    g = k.KimiMlaGeometry()
    p = k.compile_kimi_mla_partial_aot(g, max_rows=16, max_context=1024)
    assert [(s.name, s.ctype) for s in p["scalars"]] == [("rows", "int32"), ("page_table_width", "int64"), ("active_splits", "int32")]
    assert p["geometry"]["lse_base"] == 2
    launch = calls[0][0]
    assert launch.forward.query_tile == 1 and launch.forward.num_heads == 48
    assert p["scratch"]["scratch"](3) == 3 * 48 * 16 * (512 * 2 + 4)
    for args in ({"num_splits": 1}, {"head_begin": 1}, {"head_count": 96},
                 {"max_rows": 0}, {"max_context": 1048577}):
        with pytest.raises(ValueError):
            k.compile_kimi_mla_partial_aot(g, **args)
    with pytest.raises(ValueError):
        k.compile_kimi_mla_partial_aot(k.MlaDraftGeometry())


def reference_merge(p, scratch, out, lse, rows, active_splits):
    """Independent invocation of the declared reference split reduction."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass import Int32, BFloat16, Float32
    from b12x._lib.utils import current_cuda_stream, make_ptr
    from b12x.integration.cuteafd.context_split import _PartialMerge
    h, s = 48, p.geometry["num_splits"]
    size = rows * h * s * 512 * 2
    po = scratch[:size].view(torch.bfloat16).view(rows, h, s, 512)
    off = (size + 1023) // 1024 * 1024
    pl = scratch[off:off + rows * h * s * 4].view(torch.float32).view(rows, h, s)
    pl[:, :, active_splits:] = -float("inf")
    args = (from_dlpack(po), from_dlpack(pl),
            make_ptr(BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            Int32(rows), current_cuda_stream())
    fn = cute.compile(_PartialMerge(h, s), *args)
    fn(*args)


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
@pytest.mark.parametrize("begin", [0, 48])
@pytest.mark.parametrize("splits", [2, 3, 8, 16])
def test_partial_combined_full_and_fp64(cache, begin, splits):
    require_b12x()
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_dense_mla_aot
    from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
    rows, n = 8, 5 * 64
    p = programs(cache, begin)
    raw, values = records(cache, n)
    pool = raw.reshape(5, 64, -1)
    q = (torch.randn((rows, 96, 576), device="cuda") * .25).bfloat16()
    live = [0, 1, 63, 64, 65, 127, 129, n - 3]
    lengths = torch.tensor(live, device="cuda", dtype=torch.int32)
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    outputs, lses = [], []
    for rank in (0, 1):
        owned = pool[rank::2].contiguous()
        table = torch.arange(len(owned), device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
        local = torch.tensor([sum(1 for i in range(x) if (i // 64) % 2 == rank) for x in live],
                             device="cuda", dtype=torch.int32)
        out = torch.full((rows, 48, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
        lse = torch.full((rows, 48), float("nan"), device="cuda")
        scratch = torch.full((p.scratch_bytes(rows)["scratch"],), 0xFF, device="cuda", dtype=torch.uint8)
        p.launch(q, owned, table, local, cu, out, lse, scratch, scalars=(rows, table.shape[1], splits))
        reference, ref_lse = torch.empty_like(out), torch.empty_like(lse)
        reference_merge(p, scratch, reference, ref_lse, rows, splits)
        assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
        assert torch.equal(lse.view(torch.uint8), ref_lse.view(torch.uint8))
        for r, length in enumerate(local.tolist()):
            if not length:
                assert (out[r] == 0).all() and torch.isneginf(lse[r]).all()
        outputs.append(out)
        lses.append(lse)
    combine = compile_lse_combine2_aot(heads=48)
    combined = torch.empty_like(outputs[0])
    combine.launch(outputs[0], lses[0], outputs[1], lses[1], None, combined, scalars=(rows,))
    # Full h96 attention uses the same causal history, but its normal LSE is ln.
    full = compile_kimi_dense_mla_aot(KimiMlaGeometry(kv_cache=cache), max_rows=16,
                                      max_context=1024, mode="decode")
    full_out = torch.zeros((rows, 96, 512), device="cuda", dtype=torch.bfloat16)
    full_lse = torch.empty((rows, 96), device="cuda")
    fs = torch.empty(full.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    ft = torch.arange(5, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    full.launch(q, pool, ft, lengths, cu, full_out, full_lse, fs,
                scalars=(rows, rows, 5, full.geometry["num_splits"]))
    torch.testing.assert_close(combined.float(), full_out[:, begin:begin+48].float(), rtol=.025, atol=.002)
    for r, length in enumerate(live):
        if not length:
            assert (combined[r] == 0).all()
            continue
        v = values[:length].double()
        logits = q[r, begin:begin+48].double() @ v.T * (192 ** -.5)
        expected = torch.softmax(logits, dim=-1) @ v[:, :512]
        torch.testing.assert_close(combined[r].float(), expected.float(), rtol=.025, atol=.002)
        for rank in (0, 1):
            take = (torch.arange(length, device="cuda") // 64) % 2 == rank
            if take.any():
                expected_lse = torch.logsumexp(logits[:, take], dim=-1) / math.log(2)
                torch.testing.assert_close(lses[rank][r].double(), expected_lse, rtol=.002, atol=.002)


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_high_pages_graph_and_frozen_resolution(cache):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    p = programs(cache, 48)
    row_bytes = 1152 if cache == "bf16" else 656
    high = 2**31 // (64 * row_bytes) + 3
    # Bounded <2.2 GiB allocation, only two live pages touched.
    pool = torch.empty((high + 2, 64, 576 if cache == "bf16" else 656), device="cuda",
                       dtype=torch.bfloat16 if cache == "bf16" else torch.uint8)
    raw, _ = records(cache, 128)
    pool[high:] = raw.reshape(2, 64, -1)
    compact = pool[high:].clone()
    rows = 16
    q = (torch.randn((rows, 96, 576), device="cuda") * .25).bfloat16()
    table = torch.tensor([high, high+1], device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    low = table - high
    lengths = torch.tensor([0, 1, 63, 64, 65, 127, 128, 3] * 2, device="cuda", dtype=torch.int32)
    cu = torch.arange(rows+1, device="cuda", dtype=torch.int32)
    out = torch.empty((rows, 48, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((rows, 48), device="cuda")
    scratch = torch.full((p.scratch_bytes(rows)["scratch"],), 0xFF, device="cuda", dtype=torch.uint8)
    p.launch(q, compact, low, lengths, cu, out, lse, scratch, scalars=(rows, 2, 2))
    expected, expected_lse = out.clone(), lse.clone()
    def launch(n):
        p.launch(q, pool, table, lengths, cu, out, lse, scratch, scalars=(n, 2, 2))
    launch(rows)
    torch.cuda.synchronize()
    addresses = [x.data_ptr() for x in (q, pool, table, lengths, cu, out, lse, scratch)]
    with kernel_resolution_guard("Kimi owned-page graph gate"):
        for n in (1, 6, 16):
            launch(n)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(n)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:n].view(torch.uint8), expected[:n].view(torch.uint8))
            assert torch.equal(lse[:n].view(torch.uint8), expected_lse[:n].view(torch.uint8))
    assert addresses == [x.data_ptr() for x in (q, pool, table, lengths, cu, out, lse, scratch)]


def test_split_plan_uses_live_work_and_device_width():
    from b12x.integration.cuteafd.kimi_mla import kimi_context_split_plan
    assert kimi_context_split_plan(owned_pages=4096, rows=1, sm_count=188) == 16
    assert kimi_context_split_plan(owned_pages=4096, rows=16, sm_count=188) == 16
    assert kimi_context_split_plan(owned_pages=16, rows=1, sm_count=188) == 2
    assert kimi_context_split_plan(owned_pages=128, rows=1, sm_count=48) == 8
    for pages in (0, 1, 16, 128, 1024, 4096):
        for rows in (1, 16, 64):
            assert 2 <= kimi_context_split_plan(owned_pages=pages, rows=rows, sm_count=170) <= 16


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_existing_fixed_chunk_path_is_byte_identical(cache, monkeypatch):
    """Compare the default compiler with the pre-change forward source, not a numerical tolerance."""
    require_b12x()
    import importlib.util
    from b12x.integration.cuteafd import kimi_mla as k
    baseline = os.environ.get("KIMI_FIXED_FORWARD_SOURCE")
    if not baseline:
        pytest.skip("KIMI_FIXED_FORWARD_SOURCE supplies the pre-change fixed kernel")
    spec = importlib.util.spec_from_file_location("b12x.attention.dense_mla._fixed_oracle", baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    g = k.KimiMlaGeometry(kv_cache=cache)
    raw, _ = records(cache, 320)
    pool = raw.reshape(5, 64, -1)
    q = (torch.randn((3, 96, 576), device="cuda") * .25).bfloat16()
    table = torch.arange(5, device="cuda", dtype=torch.int32)[None].repeat(3, 1)
    lengths = torch.tensor([1, 129, 317], device="cuda", dtype=torch.int32)
    cu = torch.arange(4, device="cuda", dtype=torch.int32)
    outputs = []
    for factory in (k.DenseMlaForwardKernel, old.DenseMlaForwardKernel):
        monkeypatch.setattr(k, "DenseMlaForwardKernel", factory)
        from b12x.integration.cuteafd import exportable_compilation
        with exportable_compilation():
            p = k.compile_kimi_dense_mla_aot(g, max_rows=3, max_context=1024)
        out = torch.empty((3, 96, 512), device="cuda", dtype=torch.bfloat16)
        lse = torch.empty((3, 96), device="cuda")
        scratch = torch.empty(p.scratch_bytes(3)["scratch"], device="cuda", dtype=torch.uint8)
        p.launch(q, pool, table, lengths, cu, out, lse, scratch,
                 scalars=(3, 3, 5, p.geometry["num_splits"]))
        torch.cuda.synchronize()
        outputs.append((out.clone(), lse.clone()))
    for before, after in zip(*outputs):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
