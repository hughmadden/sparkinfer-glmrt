"""Qwen 3.8 Flash Next GDN layer, attention producer and attn_o over FP8-only
projection weights (``fp8_only``: E4M3 + FP32 128x128 block scales, no BF16
copy) against the existing programs fed the same E4M3 copies and the BF16
weights dequantized from them (``bf16(w * s)``):

* decode programs (``fp8`` vs ``fp8_only="decode"``): bitwise at every row
  count (<= 16 the same tensor-core GEMV, above it W8A16 TMA vs BF16 TMA over
  the dequantized weights);
* prefill programs (BF16 vs ``fp8_only="prefill"``): ``fp8_rows`` 0 (W8A16)
  bitwise; nonzero (W8A8) close;
* the E4M3 LM head (``qwen4_head_fp8``) run in 16-row spans: every row
  bitwise equal to the same row computed alone (target logits in spans equal
  MTP draft logits) and close to an FP32 reference over the dequantized head.

Synthetic weights (no checkpoint needed). The GDN cases compare outputs and
the recurrent, conv and replay state."""

from __future__ import annotations

import ctypes
import subprocess
import sys

import pytest
import torch

from ..conftest import require_b12x

_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _g():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT

    return QWEN38_FLASH_NEXT


def P(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import qwen4, qwen4_attention, qwen4_gdn

        fn = {"gdn": qwen4_gdn.compile_qwen4_gdn_aot, "producer": qwen4_attention.compile_qwen4_attn_producer_aot,
              "o": qwen4_attention.compile_qwen4_attn_o_aot, "head_fp8": qwen4.compile_qwen4_head_fp8_aot}[kind]
        _PROGRAMS[key] = fn(_g(), **kw)
    return _PROGRAMS[key]


def _w(n, k, seed, scale=0.02):
    """E4M3 [n, k], FP32 128x128 block scales, and the dequantized BF16 weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    nb, kb = -(-n // 128), k // 128
    s = (torch.rand((nb, kb), generator=gen, device="cuda") + 0.5) * scale
    full = s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w8, s.contiguous(), (w8.float() * full).bfloat16()


def _scratch(program, rows):
    return torch.zeros(program.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")


def _names(program):
    return [operand.name for operand in program.operands]


DECODE_ROWS = (1, 5, 16, 17, 33, 64)
PREFILL = [(r, f) for r in (64, 65, 600) for f in (0, 1)]


def _cmp(new, old, w8a8, what):
    if w8a8:
        c = torch.nn.functional.cosine_similarity(new.float().flatten(), old.float().flatten(), dim=0).item()
        print(f"{what}: W8A8 cosine {c:.7f}")
        assert c > 0.999, what
    else:
        assert torch.equal(new, old), f"{what}: max |diff| {(new.float() - old.float()).abs().max().item()}"


# ---------------------------------------------------------------------------
# GDN
# ---------------------------------------------------------------------------


class _GdnWeights:
    def __init__(self, g, seed):
        h, c, v, p, heads = g.hidden, g.gdn_conv_width, g.gdn_value_width, g.gdn_in_width, g.gdn_value_heads
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.w_in = _w(p, h, seed + 1)
        self.w_out = _w(h, v, seed + 2)
        self.conv_w = (torch.randn((c, 4), generator=gen, device="cuda") * 0.3).contiguous()
        self.a_log = (torch.randn((heads,), generator=gen, device="cuda") * 0.5).contiguous()
        self.dt_bias = (torch.randn((heads,), generator=gen, device="cuda") * 0.5).contiguous()
        self.norm_w = (1 + 0.1 * torch.randn((g.gdn_head_dim,), generator=gen, device="cuda")).bfloat16()


def _gdn_case(rows, slots_n, seq_of, cap, mode_new, fp8_rows, spec):
    g = _g()
    w = _GdnWeights(g, rows + 11 * cap)
    h, c, heads = g.hidden, g.gdn_conv_width, g.gdn_value_heads
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn((rows, h), generator=gen, device="cuda").bfloat16()
    conv0 = (torch.randn((slots_n, 3, c), generator=gen, device="cuda") * 0.5).bfloat16()
    state0 = torch.randn((slots_n, heads, 128, 128), generator=gen, device="cuda") * 0.05
    slots = torch.tensor(seq_of, dtype=torch.int32, device="cuda")
    first = {}
    seq_first = torch.tensor([first.setdefault(s, i) for i, s in enumerate(seq_of)], dtype=torch.int32, device="cuda")
    decode = cap <= 64

    def run(program, weights, scalars):
        conv, state = conv0.clone(), state0.clone()
        out = torch.empty((rows, h), dtype=torch.bfloat16, device="cuda")
        record = program.geometry.get("replay_bytes") or 0
        replay = [torch.zeros(record // 4, dtype=torch.float32, device="cuda")] if decode else []
        before, after = weights
        program.launch(x, *before, w.conv_w, w.a_log, w.dt_bias, w.norm_w, *after, conv, state, slots, seq_first,
                       out, *replay, _scratch(program, cap), scalars=scalars)
        torch.cuda.synchronize()
        return out, conv, state, replay

    (i8, i_s, i_bf), (o8, o_s, o_bf) = w.w_in, w.w_out
    if decode:
        old = run(P("gdn", max_rows=cap, fp8=True), ((i_bf, i8, i_s), (o_bf, o8, o_s)), (rows, spec))
        new_p = P("gdn", max_rows=cap, fp8_only="decode")
        new = run(new_p, ((i8, i_s), (o8, o_s)), (rows, spec))
    else:
        old = run(P("gdn", max_rows=cap), ((i_bf,), (o_bf,)), (rows,))
        new_p = P("gdn", max_rows=cap, fp8_only="prefill")
        new = run(new_p, ((i8, i_s), (o8, o_s)), (rows, fp8_rows))
    assert "w_in" not in _names(new_p) and "w_out" not in _names(new_p)
    assert torch.isfinite(new[0]).all() and torch.count_nonzero(new[0]) > 0
    w8a8 = not decode and fp8_rows != 0
    _cmp(new[0], old[0], w8a8, f"gdn out rows={rows}")
    if not w8a8:
        assert torch.equal(new[1], old[1]), "conv state"
        assert torch.equal(new[2], old[2]), "recurrent state"
        if decode:
            assert torch.equal(new[3][0], old[3][0]), "replay record"
    else:
        _cmp(new[2], old[2], True, "gdn state")


@pytest.mark.parametrize("rows", DECODE_ROWS)
@pytest.mark.parametrize("spec", [0, 1])
def test_qwen4_gdn_w8_decode(rows, spec):
    # Decode/verify rows: four sequences sharing the step (contiguous rows each).
    seq_of = [min(i * 4 // rows, 3) for i in range(rows)] if rows >= 4 else list(range(rows))
    _gdn_case(rows, 4, seq_of, 64, "decode", 0, spec)


@pytest.mark.parametrize("rows,fp8_rows", PREFILL)
def test_qwen4_gdn_w8_prefill(rows, fp8_rows):
    _gdn_case(rows, 2, [1] * rows, 4096, "prefill", fp8_rows, 0)


# ---------------------------------------------------------------------------
# Attention producer and attn_o
# ---------------------------------------------------------------------------


def _producer_case(rows, cap, fp8_rows):
    g = _g()
    h, n, d = g.hidden, g.heads, g.head_dim
    w8, s, wd = _w(g.attn_in_width, h, rows + cap)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn((rows, h), generator=gen, device="cuda").bfloat16()
    norms = [(1 + 0.1 * torch.randn((dim,), generator=gen, device="cuda")).bfloat16() for dim in (d, d, 128, 128)]
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") + 5
    rope_positions = positions.to(torch.int32)[:, None].expand(-1, 3).contiguous()
    block_rope_positions = (positions - positions % 4).to(torch.int32)[:, None].expand(-1, 3).contiguous()
    kv_slots = torch.arange(rows, dtype=torch.int64, device="cuda") + 3
    pool_slots = torch.where(positions % 4 == 3, positions // 4, torch.full_like(positions, -1))
    pages = -(-(rows + 3) // g.page_rows)
    decode = cap <= 64

    def run(program, weights, scalars):
        kv = torch.zeros((pages, g.kv_page_bytes), dtype=torch.uint8, device="cuda")
        keys = torch.zeros((pages * g.page_rows, 128), dtype=torch.bfloat16, device="cuda")
        index = torch.zeros((pages * g.page_rows, 128), dtype=torch.bfloat16, device="cuda")
        query = torch.empty((rows, n, d), dtype=torch.bfloat16, device="cuda")
        gate = torch.empty((rows, n * d), dtype=torch.bfloat16, device="cuda")
        index_q = torch.empty((rows, g.index_heads, 128), dtype=torch.bfloat16, device="cuda")
        program.launch(x, *weights, *norms, positions, rope_positions, block_rope_positions,
                       kv_slots, pool_slots, kv, keys, index, query, gate, index_q,
                       _scratch(program, cap), scalars=scalars)
        torch.cuda.synchronize()
        return torch.cat([query.view(rows, -1), gate, index_q.view(rows, -1)], 1), kv, keys, index

    if decode:
        old = run(P("producer", max_rows=cap, fp8=True), (wd, w8, s), (rows,))
        new_p = P("producer", max_rows=cap, fp8_only="decode")
        new = run(new_p, (w8, s), (rows,))
    else:
        old = run(P("producer", max_rows=cap), (wd,), (rows,))
        new_p = P("producer", max_rows=cap, fp8_only="prefill")
        new = run(new_p, (w8, s), (rows, fp8_rows))
    assert "w_in" not in _names(new_p)
    w8a8 = not decode and fp8_rows != 0
    for a, b, what in zip(new, old, ("outputs", "kv_cache", "token_keys", "index_cache")):
        if what == "kv_cache" and w8a8:
            a, b = a.view(torch.bfloat16), b.view(torch.bfloat16)
        _cmp(a, b, w8a8, f"producer {what} rows={rows}")


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_qwen4_attn_producer_w8_decode(rows):
    _producer_case(rows, 64, 0)


@pytest.mark.parametrize("rows,fp8_rows", PREFILL)
def test_qwen4_attn_producer_w8_prefill(rows, fp8_rows):
    _producer_case(rows, 4096, fp8_rows)


def _o_case(rows, cap, fp8_rows):
    g = _g()
    h, wd_ = g.hidden, g.heads * g.head_dim
    w8, s, wd = _w(h, wd_, rows + cap, scale=0.01)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    attn = torch.randn((rows, wd_), generator=gen, device="cuda").bfloat16()
    gate = torch.randn((rows, wd_), generator=gen, device="cuda").bfloat16()
    decode = cap <= 64
    old = torch.empty((rows, h), dtype=torch.bfloat16, device="cuda")
    new = torch.empty_like(old)
    if decode:
        P("o", max_rows=cap, fp8=True).launch(attn, gate, wd, w8, s, old, _scratch(P("o", max_rows=cap, fp8=True), cap),
                                              scalars=(rows,))
        new_p = P("o", max_rows=cap, fp8_only="decode")
        new_p.launch(attn, gate, w8, s, new, _scratch(new_p, cap), scalars=(rows,))
    else:
        P("o", max_rows=cap).launch(attn, gate, wd, old, _scratch(P("o", max_rows=cap), cap), scalars=(rows,))
        new_p = P("o", max_rows=cap, fp8_only="prefill")
        new_p.launch(attn, gate, w8, s, new, _scratch(new_p, cap), scalars=(rows, fp8_rows))
    torch.cuda.synchronize()
    assert "w_o" not in _names(new_p)
    assert torch.isfinite(new).all() and torch.count_nonzero(new) > 0
    _cmp(new, old, not decode and fp8_rows != 0, f"attn_o rows={rows}")


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_qwen4_attn_o_w8_decode(rows):
    _o_case(rows, 64, 0)


@pytest.mark.parametrize("rows,fp8_rows", PREFILL)
def test_qwen4_attn_o_w8_prefill(rows, fp8_rows):
    _o_case(rows, 4096, fp8_rows)


def test_qwen4_w8_export_queries_launch_device_sm_count(monkeypatch, tmp_path):
    """A compiler-side SM count must not leak into exported persistent grids."""
    from cuda.bindings import driver as cuda
    from b12x._lib import utils
    from b12x.integration.cuteafd._common import exportable_compilation
    from b12x.integration.cuteafd.qwen4_attention import compile_qwen4_attn_o_aot

    actual_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    monkeypatch.setattr(utils, "get_num_sm", lambda *args, **kwargs: 170 if actual_sms != 170 else 188)
    with exportable_compilation():
        program = compile_qwen4_attn_o_aot(_g(), max_rows=4096, fp8_only="prefill")
        program.export_to_c(str(tmp_path), "w8_runtime_sm", "w8_runtime_sm")
    wrapper = tmp_path / "wrapper.cpp"
    wrapper.write_text('''#include "w8_runtime_sm.h"
extern "C" void load(void *module) {
    w8_runtime_sm_Kernel_Module_Load((w8_runtime_sm_Kernel_Module_t *)module);
}
extern "C" void unload(void *module) {
    w8_runtime_sm_Kernel_Module_Unload((w8_runtime_sm_Kernel_Module_t *)module);
}
extern "C" int launch(void *module, void *a, void *g, void *w, void *s,
                      void *o, void *scratch, int rows, void *stream) {
    return cute_dsl_w8_runtime_sm_wrapper((w8_runtime_sm_Kernel_Module_t *)module,
                                         a, g, w, s, o, scratch, rows, 1,
                                         (cudaStream_t)stream);
}
''')
    library = tmp_path / "w8_runtime_sm.so"
    runtime_dir = subprocess.check_output(
        [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True).strip()
    subprocess.run(["g++", "-shared", "-fPIC", str(wrapper), str(tmp_path / "w8_runtime_sm.o"),
                    "-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart",
                    f"-L{runtime_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{runtime_dir}",
                    "-Wl,-z,defs", "-o", str(library)], check=True)
    exported = ctypes.CDLL(str(library))
    exported.load.argtypes = exported.unload.argtypes = [ctypes.c_void_p]
    exported.load.restype = exported.unload.restype = None
    exported.launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_void_p]
    exported.launch.restype = ctypes.c_int32
    module = ctypes.c_void_p()
    exported.load(ctypes.byref(module))
    try:
        g = _g()
        weights, scales, bf16 = _w(g.hidden, g.heads * g.head_dim, 6199, scale=0.01)
        attn = torch.randn((4096, g.heads * g.head_dim), device="cuda").bfloat16()
        gate = torch.randn_like(attn)
        scratch = _scratch(program, 4096)
        output = torch.empty((4096, g.hidden), dtype=torch.bfloat16, device="cuda")
        reference = torch.empty_like(output)
        baseline = P("o", max_rows=4096)
        baseline_scratch = _scratch(baseline, 4096)

        def launch(rows):
            pointers = [t.data_ptr() for t in (attn, gate, weights, scales, output, scratch)]
            assert exported.launch(ctypes.byref(module), *pointers, rows,
                                   torch.cuda.current_stream().cuda_stream) == 0

        for rows in (65, 600, 4096):
            launch(rows)
            baseline.launch(attn, gate, bf16, reference, baseline_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.isfinite(output[:rows]).all() and torch.count_nonzero(output[:rows]) > 0
            _cmp(output[:rows], reference[:rows], True, f"exported W8 rows={rows}")
        expected = output.clone()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            launch(4096)
        # AOT modules use driver kernels; the runtime getter rejects their functions.
        error, _, count = cuda.cuGraphGetNodes(graph.raw_cuda_graph())
        assert error == cuda.CUresult.CUDA_SUCCESS
        error, nodes, count = cuda.cuGraphGetNodes(graph.raw_cuda_graph(), count)
        assert error == cuda.CUresult.CUDA_SUCCESS
        grids = []
        for node in nodes[:count]:
            error, kind = cuda.cuGraphNodeGetType(node)
            assert error == cuda.CUresult.CUDA_SUCCESS
            if kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                error, params = cuda.cuGraphKernelNodeGetParams(node)
                assert error == cuda.CUresult.CUDA_SUCCESS
                grids.append((params.gridDimX, params.gridDimY, params.gridDimZ))
        assert (1, 1, actual_sms) in grids, (actual_sms, grids)
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        del graph
    finally:
        torch.cuda.synchronize()
        exported.unload(ctypes.byref(module))


# ---------------------------------------------------------------------------
# The shared E4M3 head in 16-row spans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 16, 37, 64])
def test_qwen4_head_fp8_spans(rows):
    g = _g()
    h, vocab = g.hidden, g.vocab
    gen = torch.Generator(device="cuda").manual_seed(rows)
    w8 = (torch.randn((vocab, h), generator=gen, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    scale = (torch.rand((vocab, h // 128), generator=gen, device="cuda") + 0.5) * 0.02
    x = torch.randn((rows, h), generator=gen, device="cuda").bfloat16()
    program = P("head_fp8")
    assert program.geometry["max_rows"] == 64
    full = torch.empty((rows, vocab), dtype=torch.float32, device="cuda")
    program.launch(x, w8, scale, full, scalars=(rows,))
    spans = torch.empty((rows, vocab), dtype=torch.float32, device="cuda")
    for first in range(0, rows, 16):
        n = min(16, rows - first)
        program.launch(x[first:first + n], w8, scale, spans[first:first + n], scalars=(n,))
    alone = torch.empty_like(spans)
    for r in range(rows):
        program.launch(x[r:r + 1], w8, scale, alone[r:r + 1], scalars=(1,))
    torch.cuda.synchronize()
    assert torch.equal(full, alone) and torch.equal(spans, alone), "a row's logits depend on its span"
    wd = (w8.float() * scale.repeat_interleave(128, 1)).bfloat16().float()
    reference = x.float() @ wd.t()
    err = (spans - reference).abs().max().item() / reference.abs().max().item()
    print(f"head spans rows={rows}: max rel err {err:.3e}")
    assert err < 1e-4
