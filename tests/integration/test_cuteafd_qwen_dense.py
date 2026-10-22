"""Dense operator qualification; no serving or full-model parity claim."""
from dataclasses import fields

import pytest
import torch

from ..conftest import require_b12x


def test_dense_geometry_has_no_fabricated_moe_or_hc():
    from b12x.integration.cuteafd._common import QWEN35_9B, QWEN38_27B, Qwen4Geometry
    for g, p, state in [(QWEN35_9B, 10240, 32 * 128 * 128 * 4),
                         (QWEN38_27B, 14336, 48 * 128 * 128 * 4)]:
        assert g.attn_in_width == p
        assert g.record_bytes == 4096
        assert g.gdn_state_bytes == state
        assert not {"hc_count", "routed_experts", "index_heads", "ple_dim"} & {f.name for f in fields(g)}
    g = Qwen4Geometry()
    assert g.attn_in_width == 13952 and g.gdn_in_width == 16480
    assert g.hc_width == 10240 and g.sparse_topk == 2112


@pytest.mark.parametrize("hidden", [4096, 5120])
def test_dense_norm_and_residual(hidden):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_add_aot, compile_qwen_dense_norm_aot
    g = QwenDenseGeometry(hidden=hidden)
    norm, add = compile_qwen_dense_norm_aot(g), compile_qwen_dense_add_aot(g)
    x = torch.randn(7, hidden, device="cuda", dtype=torch.bfloat16)
    weight = (0.1 * torch.randn(hidden, device="cuda")).bfloat16()
    delta = torch.randn_like(x)
    residual, out = torch.empty_like(x), torch.empty_like(x)
    add.launch(x, delta, residual, scalars=[7])
    assert torch.equal(residual, x + delta)
    norm.launch(residual, weight, out, scalars=[7])
    expected = (residual.float() * torch.rsqrt(residual.float().square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight.float())).bfloat16()
    torch.testing.assert_close(out, expected, atol=0.015625, rtol=0.008)


@pytest.mark.parametrize("hidden,intermediate", [(4096, 12288), (5120, 17408)])
def test_dense_ffn(hidden, intermediate):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_ffn_aot
    g = QwenDenseGeometry(hidden=hidden, intermediate=intermediate)
    program = compile_qwen_dense_ffn_aot(g, max_rows=64)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    up = torch.randn(2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    down = torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.01
    out = torch.empty_like(x)
    scratch = torch.empty(program.scratch_bytes(64)["scratch"], device="cuda", dtype=torch.uint8)
    program.launch(x, up, down, out, scratch, scalars=[3])
    gate, value = torch.nn.functional.linear(x, up).chunk(2, -1)
    expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * value, down)
    torch.testing.assert_close(out, expected, atol=0.002, rtol=0.04)


@pytest.mark.parametrize("hidden,heads,values,norm_dtype", [
    (4096, 16, 32, torch.float32), (5120, 24, 48, torch.bfloat16)])
@pytest.mark.parametrize("rows", [3, 65])
def test_dense_gdn_native_norm_and_silu(hidden, heads, values, norm_dtype, rows):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_gdn_aot
    g = QwenDenseGeometry(hidden=hidden, heads=heads, gdn_value_heads=values)
    program = compile_qwen_dense_gdn_aot(g, max_rows=max(64, rows), norm_dtype=norm_dtype)
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w_in = torch.randn(g.gdn_in_width, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    w_out = torch.randn(g.hidden, g.gdn_value_width, device="cuda", dtype=torch.bfloat16) * 0.01
    conv_w = torch.randn(g.gdn_conv_width, 4, device="cuda") * 0.1
    a_log = torch.randn(values, device="cuda") * 0.1
    dt_bias = torch.randn(values, device="cuda") * 0.1
    norm_w = (1 + 0.1 * torch.randn(128, device="cuda")).to(norm_dtype)
    conv = torch.zeros(1, 3, g.gdn_conv_width, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(1, values, 128, 128, device="cuda")
    slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first = torch.zeros_like(slots)
    out = torch.empty_like(x)
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    args = (x, w_in, conv_w, a_log, dt_bias, norm_w, w_out, conv, state, slots, first, out)
    if rows <= 64:
        program.launch(*args, None, scratch, scalars=[rows, 0])
    else:
        program.launch(*args, scratch, scalars=[rows])
    projected = torch.nn.functional.linear(x, w_in)
    c, v = g.gdn_conv_width, g.gdn_value_width
    qkv = projected[:, :c].T[None]
    padded = torch.nn.functional.pad(qkv, (3, 0))
    filtered = torch.nn.functional.silu(torch.nn.functional.conv1d(
        padded.float(), conv_w[:, None], groups=c).bfloat16())[0].T
    _, _, value = torch.split(filtered, [g.gdn_key_width, g.gdn_key_width, v], -1)
    # The reference L2 normalization uses sqrt(sum(x*x) + eps), not clamp(norm).
    qraw = filtered[:, :g.gdn_key_width].reshape(rows, g.gdn_key_heads, 128).float()
    kraw = filtered[:, g.gdn_key_width:2*g.gdn_key_width].reshape(rows, g.gdn_key_heads, 128).float()
    q = qraw * torch.rsqrt(qraw.square().sum(-1, keepdim=True) + 1e-6)
    k = kraw * torch.rsqrt(kraw.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(values // g.gdn_key_heads, 1) * (128 ** -0.5)
    k = k.repeat_interleave(values // g.gdn_key_heads, 1)
    value = value.reshape(rows, values, 128).float()
    b = projected[:, c + v:c + v + values]
    a = projected[:, c + v + values:]
    beta = b.sigmoid().float()
    decay = -a_log.exp()[None] * torch.nn.functional.softplus(a.float() + dt_bias)
    expected_state = torch.zeros_like(state[0])
    outputs = []
    for i in range(rows):
        expected_state *= decay[i].exp()[:, None, None]
        delta = (value[i] - (expected_state * k[i, :, None]).sum(-1)) * beta[i, :, None]
        expected_state += delta[:, :, None] * k[i, :, None]
        outputs.append((expected_state * q[i, :, None]).sum(-1).bfloat16())
    recurrent = torch.stack(outputs)
    norm = (recurrent.float() * torch.rsqrt(recurrent.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    weighted = norm * norm_w
    gated = (weighted.float() * torch.nn.functional.silu(projected[:, c:c + v].reshape(rows, values, 128).float())).bfloat16()
    expected = torch.nn.functional.linear(gated.reshape(rows, v), w_out)
    torch.testing.assert_close(state[0], expected_state, atol=2e-5, rtol=0.04)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=0.04)
    # GEMM reduction order may differ from Torch; state must retain our exact
    # projected BF16 rows, not a separately reduced approximation of them.
    actual_projection = scratch[:rows * g.gdn_in_width * 2].view(torch.bfloat16).reshape(rows, g.gdn_in_width)
    torch.testing.assert_close(actual_projection, projected, atol=0.002, rtol=0.02)
    assert torch.equal(conv[0], actual_projection[-3:, :c])
    assert next(op for op in program.operands if op.name == "norm_w").dtype == norm_dtype

    if rows > 64:
        return

    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_gdn_commit_aot
    commit = compile_qwen_dense_gdn_commit_aot(g)
    replay = torch.empty(program.geometry["replay_bytes"] // 4, device="cuda")
    conv.zero_()
    state.zero_()
    program.launch(x, w_in, conv_w, a_log, dt_bias, norm_w, w_out, conv, state, slots, first,
                   out, replay, scratch, scalars=[rows, 1])
    assert not torch.count_nonzero(conv) and not torch.count_nonzero(state)
    tables = torch.tensor([[0], [0], [2]], device="cuda", dtype=torch.int32)
    commit.launch(state, conv, replay, tables, scalars=[1, 1, 1])
    accepted_state, accepted_conv = state.clone(), conv.clone()
    state.zero_()
    conv.zero_()
    # Match live row count and projected scratch layout for exact replay math.
    program.launch(x, w_in, conv_w, a_log, dt_bias, norm_w, w_out, conv, state,
                   torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32),
                   torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32),
                   out, None, scratch, scalars=[rows, 0])
    assert torch.equal(state, accepted_state)
    assert torch.equal(conv, accepted_conv)


@pytest.mark.parametrize("heads", [16, 24])
@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
def test_dense_producer(heads, kv_format):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_producer_aot
    g = QwenDenseGeometry(heads=heads)
    program = compile_qwen_dense_producer_aot(g, max_rows=64, kv_format=kv_format)
    x = torch.randn(3, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(g.attn_in_width, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    qw = (0.1 * torch.randn(256, device="cuda")).bfloat16()
    kw = (0.1 * torch.randn(256, device="cuda")).bfloat16()
    positions = torch.tensor([[0, 0, 0], [7, 8, 9], [100, 200, 300]], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 2, -1], device="cuda", dtype=torch.int64)
    cache = torch.full((4, 2, g.kv_heads, 256), 123, device="cuda", dtype=torch.bfloat16)
    query = torch.empty(3, heads, 256, device="cuda", dtype=torch.bfloat16)
    gate = torch.empty_like(query)
    scratch = torch.empty(program.scratch_bytes(64)["scratch"], device="cuda", dtype=torch.uint8)
    if kv_format == "fp8":
        from .test_cuteafd_qwen4_fp8_kv import _pack
        reference = compile_qwen_dense_producer_aot(g, max_rows=64, kv_format="bf16")
        bf16 = torch.full((4, 2, g.kv_heads, 256), 123, device="cuda", dtype=torch.bfloat16)
        reference.launch(x, weight, qw, kw, positions, slots, bf16, query, gate, scratch, scalars=[3])
        expected, _ = _pack(bf16[[0, 2]])
        record_bytes = g.kv_record_bytes(kv_format)
        packed = torch.full((4, record_bytes), 123, device="cuda", dtype=torch.uint8)
        program.launch(x, weight, qw, kw, positions, slots, packed, query, gate, scratch, scalars=[3])
        assert torch.equal(packed[[0, 2]], expected)
        assert torch.all(packed[[1, 3]] == 123)
        high_slot = (1 << 31) // record_bytes + 1
        high = torch.empty((high_slot + 4, record_bytes), device="cuda", dtype=torch.uint8)
        high[high_slot:].fill_(123)
        high_slots = torch.tensor([high_slot, high_slot + 2, -1], device="cuda", dtype=torch.int64)
        program.launch(x, weight, qw, kw, positions, high_slots, high, query, gate, scratch, scalars=[3])
        assert torch.equal(high[high_slot:], packed)
        return
    program.launch(x, weight, qw, kw, positions, slots, cache, query, gate, scratch, scalars=[3])
    projected = torch.nn.functional.linear(x, weight)
    qg = projected[:, :2 * heads * 256].reshape(3, heads, 512)
    k, v = projected[:, 2 * heads * 256:].reshape(3, 2, g.kv_heads, 256).unbind(1)
    inv = 1 / (g.rope_theta ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    axes = torch.zeros(32, device="cuda", dtype=torch.long)
    axes[1:32:3] = 1
    axes[2:30:3] = 2
    frequency = positions[:, axes].float() * inv
    cos, sin = frequency.cos().bfloat16()[:, None], frequency.sin().bfloat16()[:, None]

    def norm_rope(t, w):
        t = (t.float() * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + 1e-6)
             * (1 + w.float())).bfloat16()
        left, right = t[..., :32], t[..., 32:64]
        rotated = torch.cat([left * cos - right * sin, right * cos + left * sin], -1)
        return torch.cat([rotated, t[..., 64:]], -1)

    torch.testing.assert_close(gate, qg[..., 256:], atol=0.002, rtol=0.02)
    torch.testing.assert_close(query, norm_rope(qg[..., :256], qw), atol=0.03125, rtol=0.02)
    expected_k = norm_rope(k, kw)
    torch.testing.assert_close(cache[[0, 2], 0], expected_k[:2], atol=0.03125, rtol=0.02)
    torch.testing.assert_close(cache[[0, 2], 1], v[:2], atol=0.002, rtol=0.02)
    assert torch.all(cache[[1, 3]] == 123)
    high_slot = (1 << 31) // g.record_bytes + 1
    high_cache = torch.empty(high_slot + 4, 2, g.kv_heads, 256, device="cuda", dtype=torch.bfloat16)
    high_cache[high_slot:].fill_(123)
    high_slots = torch.tensor([high_slot, high_slot + 2, -1], device="cuda", dtype=torch.int64)
    program.launch(x, weight, qw, kw, positions, high_slots, high_cache, query, gate, scratch, scalars=[3])
    assert torch.equal(high_cache[high_slot:], cache)


@pytest.mark.parametrize("heads", [16, 24])
def test_dense_attention_output_gate(heads):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_attn_o_aot
    g = QwenDenseGeometry(heads=heads)
    program = compile_qwen_dense_attn_o_aot(g, max_rows=64)
    width = heads * 256
    attn = torch.randn(3, width, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn_like(attn)
    weight = torch.randn(g.hidden, width, device="cuda", dtype=torch.bfloat16) * 0.01
    out = torch.empty(3, g.hidden, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(program.scratch_bytes(64)["scratch"], device="cuda", dtype=torch.uint8)
    program.launch(attn, gate, weight, out, scratch, scalars=[3])
    # Full-attention gate is sigmoid, not the GDN's SiLU.
    expected = torch.nn.functional.linear(attn * gate.sigmoid(), weight)
    torch.testing.assert_close(out, expected, atol=0.002, rtol=0.04)


@pytest.mark.parametrize("sink", [False, True])
def test_shared_split_merge_empty_rows_zero_with_nan_partials(sink):
    require_b12x()
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._mimo_kernels import MimoSplitMerge
    rows, splits, heads, width = 3, 4, 2, 128
    merge = MimoSplitMerge(heads=heads, v_head=width, sink=sink, empty_rows_zero=True)
    program = compile_program(merge, name="test_empty_split_merge", operands=(
        Operand("partial", torch.float32, "[rows,splits,heads,width]"),
        Operand("lse", torch.float32, "[rows,splits,heads]"),
        Operand("sink", torch.float32, "[heads]"),
        Operand("out", torch.bfloat16, "[rows,heads,width]", "out")),
        scalars=(Scalar("rows"), Scalar("splits")), key=(sink, "empty_rows_zero"))
    partial = torch.randn(rows, splits, heads, width, device="cuda")
    lse = torch.randn(rows, splits, heads, device="cuda")
    sinks = torch.randn(heads, device="cuda")
    lse[1].fill_(-torch.inf)
    lse[2, 1:].fill_(-torch.inf)
    partial[1].fill_(torch.nan)
    partial[2, 1:].fill_(torch.nan)
    out = torch.empty(rows, heads, width, device="cuda", dtype=torch.bfloat16)
    program.launch(partial, lse, sinks, out, scalars=[rows, splits])
    for row, live in ((0, splits), (2, 1)):
        logits = lse[row, :live].T
        if sink:
            logits = torch.cat((logits, sinks[:, None]), dim=1)
        weights = logits.softmax(-1)[:, :live].T
        expected = (partial[row, :live] * weights[..., None]).sum(0).bfloat16()
        torch.testing.assert_close(out[row], expected, atol=0.015625, rtol=0.008)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("heads", [16, 24])
def test_dense_full_gqa_is_unrestricted_and_causal(heads):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_full_gqa_aot
    g = QwenDenseGeometry(heads=heads)
    program = compile_qwen_dense_full_gqa_aot(g, max_rows=64, max_splits=4)
    rows, keys = 3, 2113
    pages = (keys + 63) // 64
    cache = torch.randn(pages * 64, 2, g.kv_heads, 256, dtype=torch.bfloat16, device="cuda") * 0.1
    query = torch.randn(rows, heads, 256, dtype=torch.bfloat16, device="cuda") * 0.1
    positions = torch.tensor([keys - 1, 0, 16], dtype=torch.int64, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda")
    out = torch.empty_like(query)
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    program.launch(query, cache, positions, table, out, scratch, scalars=[rows, 0, 4])
    expected = []
    for i, pos in enumerate(positions.tolist()):
        k = cache[:pos + 1, 0].repeat_interleave(heads // g.kv_heads, 1).transpose(0, 1)
        v = cache[:pos + 1, 1].repeat_interleave(heads // g.kv_heads, 1).transpose(0, 1)
        expected.append(torch.nn.functional.scaled_dot_product_attention(query[i, :, None], k, v)[:, 0])
    torch.testing.assert_close(out, torch.stack(expected), atol=0.002, rtol=0.02)
    # A distant key contributes: this must not silently use the QSA 2051-token cap.
    assert program.geometry["indexed"] is False
    assert torch.isfinite(out).all()
    from b12x._lib.runtime_control import kernel_resolution_guard
    # Fixed pointers/workspace survive graph capture and varying live row counts.
    with kernel_resolution_guard("dense GQA qualification"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(query, cache, positions, table, out, scratch, scalars=[rows, 0, 4])
        graph.replay()
        program.launch(query, cache, positions, table, out, scratch, scalars=[1, 0, 1])
        graph.replay()
    torch.testing.assert_close(out, torch.stack(expected), atol=0.002, rtol=0.02)
    # Poison future keys for the short query: causal output must stay identical.
    before = out[1:].clone()
    cache[17:] = 100
    program.launch(query, cache, positions, table, out, scratch, scalars=[rows, 0, 4])
    assert torch.equal(out[1:], before)

    # Park live pages beyond a signed 32-bit byte offset; untouched capacity
    # need not be initialized. The page table, not a contiguous shortcut, wins.
    high_page = (1 << 31) // g.kv_page_bytes + 1
    high_cache = torch.empty((high_page + pages) * 64, 2, g.kv_heads, 256,
                             dtype=torch.bfloat16, device="cuda")
    high_cache[high_page * 64:].copy_(cache)
    high_table = table + high_page
    baseline = out.clone()
    program.launch(query, high_cache, positions, high_table, out, scratch, scalars=[rows, 0, 4])
    assert torch.equal(out, baseline)

    # An inactive physical lane has no page table and may contain poisoned query
    # and scratch values; it cannot leak NaNs into itself or adjacent lanes.
    positions[1] = -1
    query[1].fill_(float("nan"))
    mixed_table = table.repeat(rows, 1)
    mixed_table[1].fill_(-1)
    scratch.fill_(255)
    program.launch(query, cache, positions, mixed_table, out, scratch, scalars=[rows, pages, 4])
    assert torch.equal(out[[0, 2]], baseline[[0, 2]])
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("heads", [16, 24])
@pytest.mark.parametrize("amplitude", [0.1, 8.0])
def test_dense_compact_paged_reader_matches_dequantized_bf16(heads, amplitude, monkeypatch):
    require_b12x()
    from b12x.integration.cuteafd._common import QwenDenseGeometry
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_full_gqa_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    from .test_cuteafd_qwen4_fp8_kv import _pack
    g = QwenDenseGeometry(heads=heads)
    compact = compile_qwen_dense_full_gqa_aot(g, max_rows=64, max_splits=4, kv_format="fp8")
    # Keep the dequantized oracle's reduction geometry identical: compact
    # storage changes the key tile, not the expected reconstructed values.
    import b12x.integration.cuteafd.qwen_dense as dense
    shared_attention = dense.MimoGqaAttention

    def reference_attention(**kwargs):
        attention = shared_attention(**kwargs)
        attention.tile_n = min(attention.tile_n, 32)
        attention.qk_warps = max(attention.qk_warps, 2)
        attention.tile_m = 16 * attention.qk_warps
        attention.pv_warps = 4 * attention.qk_warps
        attention.threads = 32 * attention.pv_warps
        attention.k_iters = attention.tile_n * attention.head // attention.chunk // attention.threads
        attention.v_iters = attention.tile_n * attention.v_head // attention.chunk // attention.threads
        return attention

    with monkeypatch.context() as patch:
        patch.setattr(dense, "MimoGqaAttention", reference_attention)
        reference = compile_qwen_dense_full_gqa_aot(g, max_rows=64, max_splits=4, kv_format="bf16")
    torch.manual_seed(595)
    rows, pages = 8, 35
    kv = torch.randn(pages * 64, 2, g.kv_heads, g.head_dim, device="cuda").bfloat16()
    kv[0].zero_()
    cache, scales = _pack(kv)
    payload_bytes = 2 * g.kv_heads * g.head_dim
    decoded = (cache[:, :payload_bytes].contiguous().view(torch.float8_e4m3fn).float().reshape_as(kv)
               * scales).bfloat16()
    query = (torch.randn(rows, heads, 256, device="cuda") * amplitude).bfloat16()
    positions = torch.tensor([0, 63, 64, 65, 127, 128, 2113, -1], device="cuda", dtype=torch.int64)
    query[-1].fill_(float("nan"))
    table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32).repeat(rows, 1)
    table[-1].fill_(-1)
    out, expected = torch.empty_like(query), torch.empty_like(query)
    scratch = torch.empty(compact.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    assert compact.scratch_bytes(rows) == reference.scratch_bytes(rows)
    for splits in (1, 4):
        reference.launch(query, decoded, positions, table, expected, scratch, scalars=[rows, pages, splits])
        compact.launch(query, cache, positions, table, out, scratch, scalars=[rows, pages, splits])
        assert torch.equal(out, expected)
        assert torch.isfinite(out).all() and torch.count_nonzero(out[:-1]) > 0
        assert torch.equal(out[-1], torch.zeros_like(out[-1]))
    with kernel_resolution_guard("dense compact paged GQA"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compact.launch(query, cache, positions, table, out, scratch, scalars=[rows, pages, 4])
        graph.replay()
        compact.launch(query, cache, positions, table, out, scratch, scalars=[1, pages, 1])
        graph.replay()
    assert torch.equal(out, expected)
    high_page = (1 << 31) // (64 * g.kv_record_bytes("fp8")) + 1
    high = torch.empty(((high_page + pages) * 64, g.kv_record_bytes("fp8")),
                       device="cuda", dtype=torch.uint8)
    high[high_page * 64:].copy_(cache)
    high_table = table + high_page
    high_table[-1].fill_(-1)
    compact.launch(query, high, positions, high_table, out, scratch, scalars=[rows, pages, 4])
    assert torch.equal(out, expected)
