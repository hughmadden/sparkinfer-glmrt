"""Canonical shared BF16 projections and MiMo decode row invariance.

All shared GLM/GLM Flash/MiMo BF16 shapes sweep live widths 1..128. MiMo's
production output ABI sweeps its complete 64-row verify capacity. The other
MiMo operators pin the M8/M9 boundary localization that exposed the defect.
The BF16 head test uses CUTEAFD_NATIVE_LIBRARY from a sealed engine build.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import replace
import os

import pytest
import torch

from ..conftest import require_b12x
from ._mimo import cos_sin
from .test_cuteafd_fp8_moe_aot import wire_rows
from .test_cuteafd_mimo_w8_aot import _scratch, _w


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _geometry():
    from b12x.integration.cuteafd import MIMO_V26_FLASH

    g = MIMO_V26_FLASH
    return replace(g, heads=g.heads // 2, full_kv_heads=g.full_kv_heads // 2,
                   swa_kv_heads=g.swa_kv_heads // 2, dense_inter=g.dense_inter // 2)


def _assert_rows(run, rows=(8, 9, 10)):
    """Distinct rows cover every lane, including the second skinny M tile."""
    serial = torch.cat([run(i, 1) for i in range(max(rows))])
    for count in rows:
        got = run(0, count)
        torch.cuda.synchronize()
        bits = torch.int32 if got.dtype == torch.float32 else torch.int16
        differences = int((got.view(bits) != serial[:count].view(bits)).sum())
        assert differences == 0, f"M={count}: {differences} differing output elements versus serial M=1"


@pytest.mark.parametrize("family,tp", [("flash", 1), ("flash", 2), ("pro", 1), ("pro", 2)])
def test_mimo_bf16_output_row_invariance(family, tp):
    from b12x.integration.cuteafd import MIMO_V26_FLASH, MIMO_V26_PRO
    from b12x.integration.cuteafd.mimo_attention import compile_mimo_o_aot

    root = {"flash": MIMO_V26_FLASH, "pro": MIMO_V26_PRO}[family]
    g = replace(root, heads=root.heads // tp)
    # The official Flash checkpoint has no FP8 o_proj copy. Its decode ABI
    # still has FP8 operands, ignored when fp8_rows is zero.
    program = compile_mimo_o_aot(g, max_rows=64, fp8=True)
    width = g.heads * g.v_head_dim
    x = torch.randn(64, width, device="cuda").bfloat16()
    weight = (torch.randn(g.hidden, width, device="cuda") * 0.03).bfloat16()
    dummy = torch.empty(16, device="cuda", dtype=torch.uint8)

    def run(first, count):
        out = torch.empty(count, g.hidden, device="cuda", dtype=torch.bfloat16)
        program.launch(x[first:first + count], weight, dummy, dummy, out, scalars=(count, 0))
        return out

    _assert_rows(run, range(1, 65))


def _shared_bf16_shapes():
    from b12x.integration.cuteafd import GLM53, GLM53_FLASH, MIMO_V26_FLASH, MIMO_V26_PRO

    shapes = set()
    for g in (MIMO_V26_FLASH, MIMO_V26_PRO):
        for tp in (1, 2):
            local = replace(g, heads=g.heads // tp, full_kv_heads=g.full_kv_heads // tp,
                            swa_kv_heads=g.swa_kv_heads // tp)
            shapes.update((local.qkv_width(kind), g.hidden) for kind in ("full", "swa"))
            shapes.update(((g.hidden, local.heads * g.v_head_dim),
                           (2 * g.dense_inter // tp, g.hidden), (g.hidden, g.dense_inter // tp)))
    for g in (GLM53, GLM53_FLASH):
        shapes.add((g.qkv_a_width, g.hidden))
        for tp in (1, 2):
            heads, index = g.heads // tp, g.index_heads // tp
            key_width = (256 if g is GLM53_FLASH else 128) + index
            shapes.update(((heads * g.qk_head_dim, g.q_lora_rank),
                           (g.hidden, heads * g.v_head_dim),
                           (index * g.index_head_dim, g.q_lora_rank), (key_width, g.hidden)))
            for inter in (g.dense_inter // tp, g.moe_inter // tp):
                shapes.update(((2 * inter, g.hidden), (g.hidden, inter)))
    shapes.update(((GLM53_FLASH.kda_in_width, GLM53_FLASH.hidden),
                   (GLM53_FLASH.hidden, GLM53_FLASH.kda_width)))
    return sorted(shapes)


@pytest.mark.parametrize("n,k", _shared_bf16_shapes())
def test_shared_bf16_projection_row_invariance(n, k):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._glm_kernels import glm_projection
    from b12x.integration.cuteafd.glmf import KDA_DECODE_TILES
    from .test_cuteafd_glmf_kda_decode_tiles import _within_bound

    torch.manual_seed(n + k)
    weight = (torch.randn(n, k, device="cuda") / 16).bfloat16()
    x = torch.randn(128, k, device="cuda").bfloat16()
    canonical = None
    for wide, tiles in ((False, ()), (True, KDA_DECODE_TILES.get((n, k), ()))):
        op = glm_projection(n, k, wide=wide, tiles_by_rows=tiles)
        assert op.max_skinny_rows == 0
        program = compile_program(op, name="shared_bf16_row_invariance",
                                  operands=(Operand("x", torch.bfloat16, "x"),
                                            Operand("w", torch.bfloat16, "w"),
                                            Operand("out", torch.bfloat16, "out", "out")),
                                  scalars=(Scalar("rows"),), key=op.key())

        def run(first, count):
            out = torch.empty(count, n, device="cuda", dtype=torch.bfloat16)
            program.launch(x[first:first + count], weight, out, scalars=(count,))
            return out

        full = run(0, 128)
        torch.cuda.synchronize()
        if canonical is None:
            canonical = full
            assert _within_bound(full, x, weight)
        else:
            assert torch.equal(full.view(torch.int16), canonical.view(torch.int16))
        _assert_rows(run, range(1, 129))
        if wide:
            for count in (512, 513):
                expanded = x.repeat(5, 1)[:count].contiguous()
                out = torch.empty(count, n, device="cuda", dtype=torch.bfloat16)
                program.launch(expanded, weight, out, scalars=(count,))
                torch.cuda.synchronize()
                expected = canonical.repeat(5, 1)[:count]
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("kind,kv", [("full", "int8"), ("swa", "bf16")])
def test_mimo_qkv_row_invariance(kind, kv):
    from b12x.integration.cuteafd.mimo_attention import compile_mimo_producer_aot, record_bytes

    g = _geometry()
    program = compile_mimo_producer_aot(g, kind=kind, max_rows=64, fp8_only="decode", kv=kv)
    weight, scale, _ = _w(g.qkv_width(kind), g.hidden, 718)
    x = torch.randn(10, g.hidden, device="cuda").bfloat16()
    positions = torch.arange(10, device="cuda", dtype=torch.int64) + 7
    table = cos_sin(32, kind)

    def run(first, count):
        slots = torch.arange(count, device="cuda", dtype=torch.int64)
        cache = torch.zeros(count, record_bytes(g, kind, kv), device="cuda", dtype=torch.uint8)
        query = torch.empty(count, g.heads, g.qk_head_dim, device="cuda", dtype=torch.bfloat16)
        program.launch(x[first:first + count], positions[first:first + count], slots, table,
                       weight, scale, cache, query, _scratch(program, count), scalars=(count, 32))
        return torch.cat([query.view(count, -1).view(torch.uint8), cache], 1).view(torch.int16)

    _assert_rows(run)


def test_mimo_dense_ffn_row_invariance():
    from b12x.integration.cuteafd.mimo_ffn import compile_mimo_ffn_aot

    g = _geometry()
    program = compile_mimo_ffn_aot(g, max_rows=64, fp8_only="decode")
    gu, gs, _ = _w(2 * g.dense_inter, g.hidden, 719)
    down, ds, _ = _w(g.hidden, g.dense_inter, 720)
    x = torch.randn(10, g.hidden, device="cuda").bfloat16()

    def run(first, count):
        out = torch.empty(count, g.hidden, device="cuda", dtype=torch.bfloat16)
        program.launch(x[first:first + count], gu, gs, down, ds, out,
                       _scratch(program, count), scalars=(count, 32))
        return out

    _assert_rows(run)


@pytest.mark.parametrize("wire", [False, True])
def test_mimo_mxfp4_expert_row_invariance(wire):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from .test_cuteafd_mxfp4_moe_aot import _weights

    g = GEOMETRIES["mimof"].with_tp(4)
    weights = _weights(g, g.slice)
    x = torch.randn(10, g.hidden, device="cuda").bfloat16()
    source = wire_rows(x)[0] if wire else x
    ids = torch.rand(10, g.experts, device="cuda").topk(g.top_k, -1).indices.int().contiguous()
    # Make all rows hit the same experts too: group length must not change a row's fold.
    ids[:] = ids[0].clone()
    scores = torch.rand(10, g.top_k, device="cuda")
    scores /= scores.sum(1, keepdim=True)
    programs = {cap: compile_fp8_moe_aot(g, route="auto", max_rows=cap, wire=wire) for cap in (1, 16, 80)}
    scratch = {cap: torch.empty(fp8_moe_scratch_bytes(g, "auto", cap, wire), device="cuda", dtype=torch.uint8)
               for cap in programs}

    def run(first, count, capacity):
        out = torch.empty(count, g.hidden, device="cuda", dtype=torch.bfloat16)
        programs[capacity].launch(source[first:first + count], ids[first:first + count],
                                  scores[first:first + count], *weights, out, scratch[capacity], scalars=(count,))
        return out

    serial = torch.cat([run(i, 1, 1) for i in range(10)])
    for capacity in (16, 80):
        for count in (8, 9, 10):
            got = run(0, count, capacity)
            torch.cuda.synchronize()
            differences = int((got.view(torch.int16) != serial[:count].view(torch.int16)).sum())
            assert differences == 0, f"cap={capacity}, M={count}: {differences} differing expert outputs"


def test_mimo_bf16_head_row_invariance():
    path = os.environ.get("CUTEAFD_NATIVE_LIBRARY")
    if not path:
        pytest.skip("CUTEAFD_NATIVE_LIBRARY must name a sealed cuteafd native library")
    lib = C.CDLL(path)
    ptr = C.c_void_p
    create = lib.cuteafd_vocabulary_head_create_vocab
    create.argtypes = [ptr, C.c_uint64, C.c_int32, C.c_int32, C.c_int32, C.POINTER(ptr)]
    create.restype = C.c_int32
    launch = lib.cuteafd_vocabulary_head_launch_tensor_invariant
    launch.argtypes = [ptr, ptr, ptr, ptr, C.c_int32, ptr]
    launch.restype = C.c_int32
    destroy = lib.cuteafd_v41_markov_destroy
    destroy.argtypes, destroy.restype = [ptr], C.c_int32
    g = _geometry()
    weight = (torch.randn(g.vocab_size, g.hidden, device="cuda") * 0.03).bfloat16()
    x = torch.randn(10, g.hidden, device="cuda").bfloat16()
    workspace = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
    handle = ptr()
    assert create(workspace.data_ptr(), workspace.numel(), g.hidden, 64, g.vocab_size, C.byref(handle)) == 0
    stream = torch.cuda.current_stream().cuda_stream

    def run(first, count):
        out = torch.empty(count, g.vocab_size, device="cuda", dtype=torch.float32)
        assert launch(handle, x[first:first + count].data_ptr(), weight.data_ptr(), out.data_ptr(), count, stream) == 0
        return out

    try:
        _assert_rows(run)
    finally:
        torch.cuda.synchronize()
        assert destroy(handle) == 0
