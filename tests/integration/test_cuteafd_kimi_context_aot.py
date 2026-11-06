"""Owned-page Kimi MLA: full-attention oracles, high IDs and graph replay."""
from functools import lru_cache
from pathlib import Path
import os
import math

import pytest
import torch

from ..conftest import require_b12x


@lru_cache(None)
def programs(cache, begin, device_splits=False, heads_per_tile=16):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_mla_partial_aot
    with exportable_compilation():
        p = compile_kimi_mla_partial_aot(KimiMlaGeometry(kv_cache=cache), max_rows=16,
                                       max_context=1024, head_begin=begin, device_splits=device_splits,
                                       heads_per_tile=heads_per_tile)
    if os.environ.get("K1_EXPORT_DIR"):
        target = Path(os.environ["K1_EXPORT_DIR"])
        target.mkdir(parents=True, exist_ok=True)
        stem = f"kimi_context_{cache}_h{begin}_t{heads_per_tile}" + ("_device" if device_splits else "")
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
    p8 = k.compile_kimi_mla_partial_aot(g, max_rows=16, max_context=1024, heads_per_tile=8)
    assert [(s.name, s.ctype) for s in p["scalars"]] == [("rows", "int32"), ("page_table_width", "int64"), ("active_splits", "int32")]
    assert p["geometry"]["lse_base"] == 2
    assert p["geometry"]["heads_per_tile"] == 16 and p8["geometry"]["heads_per_tile"] == 8
    assert p["key"] != p8["key"]
    assert k.kimi_context_heads_per_tile(rows=1) == 8
    assert k.kimi_context_heads_per_tile(rows=4) == 16
    assert k.kimi_context_heads_per_tile(rows=16) == 16
    launch = calls[0][0]
    assert launch.forward.query_tile == 1 and launch.forward.num_heads == 48
    assert launch.smem.heads_per_tile == 16 and calls[1][0].smem.heads_per_tile == 8
    assert p["scratch"]["scratch"](3) == 3 * 48 * 16 * (512 * 2 + 4)
    for args in ({"num_splits": 1}, {"head_begin": 1}, {"head_count": 96},
                 {"max_rows": 0}, {"max_context": 1048577}, {"heads_per_tile": 4}):
        with pytest.raises(ValueError):
            k.compile_kimi_mla_partial_aot(g, **args)
    with pytest.raises(ValueError):
        k.kimi_context_heads_per_tile(rows=0)
    with pytest.raises(ValueError):
        k.compile_kimi_mla_partial_aot(k.MlaDraftGeometry())


def test_draft_partial_compile_contract(monkeypatch):
    from b12x.integration.cuteafd import kimi_mla as k
    calls = []
    monkeypatch.setattr(k, "compile_program", lambda launch, **kw: calls.append((launch, kw)) or kw)
    for window in (None, 4096):
        p = k.compile_mla_draft_partial_aot(k.MlaDraftGeometry(window_size=window))
        launch = calls[-1][0]
        assert launch.forward.explicit_intervals
        assert launch.forward.query_tile == 1 and launch.forward.num_heads == 64
        assert not launch.forward.noncausal_block and launch.forward.window_size is None
        assert p["geometry"]["lse_base"] == 2
        assert p["geometry"]["runtime_attention_scale"]
        assert [(s.name, s.ctype) for s in p["scalars"]] == [
            ("rows", "int32"), ("page_table_width", "int64"),
            ("active_splits", "int32"), ("attention_scale", "float32")]
    with pytest.raises(ValueError):
        k.compile_mla_draft_partial_aot(k.KimiMlaGeometry())
    for args in ({"max_rows": 0}, {"max_context": 1048577}, {"num_splits": 1}):
        with pytest.raises(ValueError):
            k.compile_mla_draft_partial_aot(k.MlaDraftGeometry(), **args)
    target = k.compile_kimi_mla_partial_aot()
    assert not calls[-1][0].forward.explicit_intervals
    assert "runtime_attention_scale" not in target["geometry"]


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
@pytest.mark.parametrize("window", [None, 4096])
@pytest.mark.parametrize("high_ids", [False, True])
def test_draft_partial_global_window_and_subpage_intervals(cache, window, high_ids):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.kimi_mla import MlaDraftGeometry, compile_mla_draft_partial_aot
    from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
    torch.manual_seed(1354)
    anchors = [0, 1, 63, 64, 4095, 4096, 4097, 4163]
    block, rows, n = 8, len(anchors) * 8, 4224
    raw, values = records(cache, n)
    pool = raw.reshape(-1, 64, raw.shape[-1])
    q = (torch.randn((rows, 64, 576), device="cuda") * .25).bfloat16()
    with exportable_compilation():
        p = compile_mla_draft_partial_aot(MlaDraftGeometry(kv_cache=cache, window_size=window),
                                          max_rows=rows, max_context=n)
        combine = compile_lse_combine2_aot(heads=64)
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    scale = .037  # Deliberately differs from the target scale.
    outputs, lses = [], []
    for rank in (0, 1):
        local = pool[rank::2].contiguous()
        first = (2**31 // (64 * raw.element_size() * raw.shape[-1]) + 1) if high_ids else 0
        owned = torch.empty((first + len(local), *local.shape[1:]), dtype=raw.dtype, device="cuda")
        owned[first:] = local
        table = torch.arange(first, first + len(local), device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
        selected = torch.tensor([i for i in range(n) if (i // 64) % 2 == rank])
        bounds = []
        for anchor in anchors:
            for offset in range(block):
                begin = 0 if window is None else max(0, min(anchor, anchor + offset + 1 - window))
                end = anchor + block
                bounds.append((int((selected < begin).sum()), int((selected < end).sum())))
        intervals = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        out = torch.full((rows, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
        lse = torch.full((rows, 64), float("nan"), device="cuda")
        scratch = torch.full((p.scratch_bytes(rows)["scratch"],), 255, device="cuda", dtype=torch.uint8)
        p.launch(q, owned, table, intervals, cu, out, lse, scratch,
                 scalars=(rows, table.shape[1], 8, scale))
        for row, (begin, end) in enumerate(bounds):
            if begin == end:
                assert torch.count_nonzero(out[row]) == 0 and torch.isneginf(lse[row]).all()
        outputs.append(out)
        lses.append(lse)
    merged = torch.empty_like(outputs[0])
    combine.launch(outputs[0], lses[0], outputs[1], lses[1], lses[0], merged, scalars=(rows,))
    for row in range(rows):
        anchor, offset = anchors[row // block], row % block
        begin = 0 if window is None else max(0, min(anchor, anchor + offset + 1 - window))
        live = values[begin:anchor + block].double()
        scores = q[row].double() @ live.T * scale
        reference = torch.softmax(scores, dim=-1) @ live[:, :512]
        torch.testing.assert_close(merged[row].float(), reference.float(), atol=.006, rtol=.025)
    assert torch.isfinite(merged).all()


def test_k6_decode_buckets_derive_from_native_dispatch_constants():
    import ast
    source = Path(__file__).resolve().parents[2] / "b12x/gemm/trellis_linear/_native.py"
    tree = ast.parse(source.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    routes = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "K6_DECODE_ROUTE_THRESHOLDS" for t in node.targets))
    names = {node.id for node in routes.elts}
    thresholds = sorted(constants[name] for name in names)
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in ("_launch_k6_opt", "_decode_k6"):
            continue
        for branch in ast.walk(node):
            if not isinstance(branch, ast.Compare) or not isinstance(branch.left, ast.Name) or branch.left.id != "rows":
                continue
            for call in branch.comparators:
                assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Int32"
                assert len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                used.add(call.args[0].id)
    assert used == names
    buckets = sorted({1, 4, 64, *thresholds})
    assert buckets == [1, 4, 8, 16, 32, 48, 64]
    for rows in range(1, 65):
        bucket = next(b for b in buckets if b >= rows)
        assert all((rows <= edge) == (bucket <= edge) for edge in thresholds)
    integration = source.parents[2] / "integration/cuteafd/trellis_linear.py"
    text = integration.read_text()
    assert '"decode_route_thresholds": routes' in text
    assert "list(K6_DECODE_ROUTE_THRESHOLDS)" in text


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


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
@pytest.mark.parametrize("begin", [0, 48])
@pytest.mark.parametrize("heads_per_tile", [8, 16])
def test_device_splits_scalar_oracle_and_replay(cache, begin, heads_per_tile):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    scalar = programs(cache, begin, heads_per_tile=heads_per_tile)
    device = programs(cache, begin, True, heads_per_tile=heads_per_tile)
    for rows in (1, 8, 16):
        pool, _ = records(cache, 1024)
        pool = pool.reshape(16, 64, -1)
        q = (torch.randn((rows, 96, 576), device="cuda") * .25).bfloat16()
        table = torch.arange(16, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
        lengths = torch.tensor(([1024, 0, 1, 63, 64, 65, 513, 1023] * 2)[:rows],
                               device="cuda", dtype=torch.int32)
        cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
        active = torch.tensor([2], device="cuda", dtype=torch.int32)
        expected = torch.empty((rows, 48, 512), device="cuda", dtype=torch.bfloat16)
        expected_lse = torch.empty((rows, 48), device="cuda")
        out, lse = torch.empty_like(expected), torch.empty_like(expected_lse)
        scratch = torch.empty(device.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
        reference_scratch = torch.empty_like(scratch)

        def launch():
            device.launch(q, pool, table, lengths, cu, out, lse, scratch, active,
                          scalars=(rows, 16))

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard("Kimi device split graph gate"):
            with torch.cuda.graph(graph):
                launch()
            # One executable must reread the scalar, including decreasing split counts.
            for splits in (2, 4, 8, 16, 4, 2):
                active.fill_(splits)
                scratch.fill_(0xFF)
                reference_scratch.fill_(0xFF)
                scalar.launch(q, pool, table, lengths, cu, expected, expected_lse,
                              reference_scratch, scalars=(rows, 16, splits))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                assert torch.equal(lse.view(torch.uint8), expected_lse.view(torch.uint8))
                # Inactive CTAs must not touch their poisoned scratch lanes.
                size = rows * 48 * 16 * 512 * 2
                partial = scratch[:size].view(rows, 48, 16, 1024)
                assert (partial[:, :, splits:] == 0xFF).all()


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
@pytest.mark.parametrize("begin", [0, 48])
def test_device_splits_short_history_timing(cache, begin):
    require_b12x()
    scalar, device = programs(cache, begin), programs(cache, begin, True)
    pool, _ = records(cache, 128)
    pool = pool.reshape(2, 64, -1)
    q = (torch.randn((1, 96, 576), device="cuda") * .25).bfloat16()
    table = torch.arange(2, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    cu = torch.arange(2, device="cuda", dtype=torch.int32)
    active = torch.tensor([2], device="cuda", dtype=torch.int32)
    out = torch.empty((1, 48, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((1, 48), device="cuda")
    scratch = torch.empty(device.scratch_bytes(1)["scratch"], device="cuda", dtype=torch.uint8)
    common = (q, pool, table, lengths, cu, out, lse, scratch)
    graphs = []
    for p, operands, scalars in ((scalar, common, (1, 2, 2)),
                                 (device, (*common, active), (1, 2))):
        p.launch(*operands, scalars=scalars)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(128):
                p.launch(*operands, scalars=scalars)
        graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)
    times = []
    for graph in graphs:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(16):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (128 * 16))
    print(f"device_split_short cache={cache} h={begin} scalar_us={times[0]:.4f} "
          f"device_us={times[1]:.4f} ratio={times[1]/times[0]:.4f}")
    assert times[1] <= times[0] * 1.10, "fixed-grid overhead exceeds the 10% stop bar"


def test_split_plan_uses_live_work_and_device_width():
    from b12x.integration.cuteafd.kimi_mla import kimi_context_split_plan
    assert kimi_context_split_plan(owned_pages=4096, rows=1, sm_count=188) == 16
    assert kimi_context_split_plan(owned_pages=4096, rows=16, sm_count=188) == 16
    assert kimi_context_split_plan(owned_pages=16, rows=1, sm_count=188) == 2
    assert kimi_context_split_plan(owned_pages=128, rows=1, sm_count=48) == 8
    for pages in (0, 1, 16, 128, 1024, 4096):
        for rows in (1, 16, 64):
            assert kimi_context_split_plan(owned_pages=pages, rows=rows, sm_count=170) in (2, 4, 8, 16)


def test_draft_split_plan_fills_starved_rows_only():
    """The draft plan is a measured table: one split per owned page until the
    capacity bounds it, eight splits while one wave of CTA tiles still absorbs
    them, then page-density steps. Only the one-row bucket compiles the doubled
    split capacity, and that decision follows the runtime SM count: a full
    32-split wave of one row (128 CTAs) fits both 188 and 170 SMs, two rows
    (256) fit neither."""
    from b12x.integration.cuteafd.kimi_mla import (
        draft_context_split_capacity, draft_context_split_plan, kimi_context_split_plan)
    for sm in (188, 170):
        assert draft_context_split_capacity(rows=1, sm_count=sm) == 32
        assert draft_context_split_capacity(rows=2, sm_count=sm) == 16
        assert draft_context_split_capacity(rows=8, sm_count=sm) == 16
        assert draft_context_split_capacity(rows=64, sm_count=sm) == 16
    # The measured fp8/bf16 sweep cells, identical on both RTX SM counts.
    measured = {
        (8, 1): 8, (32, 1): 32, (64, 1): 32,
        (8, 2): 8, (32, 2): 16, (64, 2): 16,
        (8, 4): 8, (32, 4): 8, (64, 4): 8,
        (8, 8): 4, (32, 8): 4, (64, 8): 16,
        (8, 16): 2, (32, 16): 8, (64, 16): 8,
        (8, 64): 2, (32, 64): 2, (64, 64): 2,
    }
    for sm in (188, 170):
        for (pages, rows), splits in measured.items():
            assert draft_context_split_plan(owned_pages=pages, rows=rows,
                                            sm_count=sm) == splits
    with pytest.raises(ValueError):
        draft_context_split_plan(owned_pages=64, rows=0, sm_count=188)
    with pytest.raises(ValueError):
        draft_context_split_capacity(rows=1, sm_count=0)
    for pages in (0, 1, 16, 128, 1024, 4096):
        for rows in (1, 2, 3, 4, 5, 8, 16, 32, 64):
            for sm in (48, 170, 188):
                capacity = draft_context_split_capacity(rows=rows, sm_count=sm)
                splits = draft_context_split_plan(owned_pages=pages, rows=rows, sm_count=sm)
                assert splits in (2, 4, 8, 16, 32) and splits <= capacity
    # The plan holds on the 170-SM card because the only SM-keyed branch is the
    # capacity test, and 128 CTAs fit there too.
    assert (draft_context_split_plan(owned_pages=64, rows=16, sm_count=188)
            >= kimi_context_split_plan(owned_pages=64, rows=16, sm_count=188))


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
