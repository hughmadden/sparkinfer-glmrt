"""Checkpoint bytes + WEIGHT_SOURCES + prep programs == the prepared packers' operands."""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise, check_export, checkpoint_tensor, has_checkpoint, u8


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """All bytes of the tensor's backing storage (views of packed buffers)."""
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage)


def _packed_operands(layer: int, ratio: int, hash_layer: bool, device) -> dict[tuple[str, str], torch.Tensor]:
    """Operands exactly as b12x_model.load_layer_weights packs them."""
    from b12x.attention import dsv4_compressor, dsv4_producer
    from b12x.gemm import block_fp8_linear, wo_projection
    from b12x.integration.cuteafd import FLASH as cfg

    g = lambda n: checkpoint_tensor(f"layers.{layer}.{n}", device)  # noqa: E731
    out: dict[tuple[str, str], torch.Tensor] = {}
    producer = dsv4_producer.pack_weights(
        g("attn.wq_a.weight"), u8(g("attn.wq_a.scale")), g("attn.wq_b.weight"), u8(g("attn.wq_b.scale")),
        g("attn.wkv.weight"), u8(g("attn.wkv.scale")), g("attn.q_norm.weight").contiguous(),
        g("attn.kv_norm.weight").contiguous())
    out.update({
        ("producer", "w_qkv"): producer.qkv_rank.weight.values,
        ("producer", "w_qkv_scale"): producer.qkv_rank.weight.scale_mma,
        ("producer", "w_q"): producer.q.weight.values,
        ("producer", "w_q_scale"): producer.q.weight.scale_mma,
        ("producer", "q_norm"): producer.q_norm, ("producer", "kv_norm"): producer.kv_norm,
        ("mhc_pre", "fn"): g("hc_attn_fn").float().contiguous(),
        ("mhc_pre", "scale"): g("hc_attn_scale").float().contiguous(),
        ("mhc_pre", "base"): g("hc_attn_base").float().contiguous(),
        ("mhc_pre", "norm"): g("attn_norm.weight").contiguous(),
        ("sparse_mla", "attn_sink"): g("attn.attn_sink").float().contiguous(),
        ("router_scores", "w"): g("ffn.gate.weight").contiguous(),
    })
    if ratio:
        kw = {}
        if ratio == 4:
            kw = dict(index_wkv=g("attn.indexer.compressor.wkv.weight").contiguous(),
                      index_wgate=g("attn.indexer.compressor.wgate.weight").contiguous(),
                      index_ape=g("attn.indexer.compressor.ape").float().contiguous(),
                      index_norm=g("attn.indexer.compressor.norm.weight").contiguous())
            indexer = dsv4_producer.pack_indexer_weights(
                g("attn.indexer.wq_b.weight"), u8(g("attn.indexer.wq_b.scale")),
                g("attn.indexer.weights_proj.weight").contiguous())
            out.update({("index_producer", "w_q"): indexer.q.weight.values,
                        ("index_producer", "w_q_scale"): indexer.q.weight.scale_mma,
                        ("index_producer", "w_proj"): indexer.weights_projection})
        comp = dsv4_compressor.pack_weights(
            g("attn.compressor.wkv.weight").contiguous(), g("attn.compressor.wgate.weight").contiguous(),
            g("attn.compressor.ape").float().contiguous(), g("attn.compressor.norm.weight").contiguous(), **kw)
        out.update({("compressor", "joint_projection"): comp.joint_projection,
                    ("compressor", "main_ape"): comp.main_ape, ("compressor", "main_norm"): comp.main_norm})
        if ratio == 4:
            out.update({("compressor", "index_ape"): comp.index_ape,
                        ("compressor", "index_norm"): comp.index_norm})
    wo = wo_projection.pack_weights(
        g("attn.wo_a.weight"), u8(g("attn.wo_a.scale")), g("attn.wo_b.weight"), u8(g("attn.wo_b.scale")),
        groups=cfg.o_groups, group_width=cfg.o_group_width, rank=cfg.o_lora_rank, hidden=cfg.hidden)
    out.update({("wo_projection", "wo_a"): wo.wo_a.values, ("wo_projection", "wo_a_scale"): wo.wo_a.scale_mma,
                ("wo_projection", "wo_b"): wo.wo_b.values, ("wo_projection", "wo_b_scale"): wo.wo_b.scale_mma})
    sw1, sw3 = g("ffn.shared_experts.w1.weight"), g("ffn.shared_experts.w3.weight")
    ss1, ss3 = u8(g("ffn.shared_experts.w1.scale")), u8(g("ffn.shared_experts.w3.scale"))
    w13 = block_fp8_linear.pack_weight(
        torch.cat((sw1.view(torch.uint8), sw3.view(torch.uint8))).view(torch.float8_e4m3fn), torch.cat((ss1, ss3)))
    w2 = block_fp8_linear.pack_weight(g("ffn.shared_experts.w2.weight"), u8(g("ffn.shared_experts.w2.scale")))
    out.update({("shared_ffn", "w13"): w13.weight.values, ("shared_ffn", "w13_scale"): w13.weight.scale_mma,
                ("shared_ffn", "w2"): w2.weight.values, ("shared_ffn", "w2_scale"): w2.weight.scale_mma})
    if hash_layer:
        out[("engine", "tid2eid")] = g("ffn.gate.tid2eid").to(torch.int32).contiguous()
    else:
        out[("engine", "gate_bias")] = g("ffn.gate.bias").float().contiguous()
    return out


@pytest.fixture(scope="module")
def prep():
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.weights import (
        compile_dsv4_block_fp8_scale_prep_aot,
        compile_dsv4_i64_to_i32_aot,
    )

    with exportable_compilation():
        return {"scale": compile_dsv4_block_fp8_scale_prep_aot(), "narrow": compile_dsv4_i64_to_i32_aot()}


def _engine_operand(entry, shape, layer, prep, device):
    """What a native loader does: raw bytes, cat along rows, then prep."""
    from b12x.integration.cuteafd.weights import block_fp8_scale_prep_args, scale_mma_bytes

    names = [t.replace("{L}", str(layer)).replace("{L1}", str(layer + 1)) for t in entry.tensors]
    raw = torch.cat([checkpoint_tensor(n, device).contiguous().view(torch.uint8).flatten() for n in names])
    if entry.prep is None:
        return raw
    if entry.prep == "block_fp8_scale":
        n, k, groups = shape
        out = torch.full((scale_mma_bytes(n, k, groups),), 0xA5, dtype=torch.uint8, device=device)
        prep["scale"].launch(raw, out, scalars=block_fp8_scale_prep_args(n, k, groups))
        return out
    assert entry.prep == "narrow_i64"
    count = raw.numel() // 8
    out = torch.full((count,), -7, dtype=torch.int32, device=device)
    prep["narrow"].launch(raw.view(torch.int64), out, scalars=(count,))
    return out.view(torch.uint8)


@pytest.mark.parametrize(("layer", "ratio", "hash_layer"), [(0, 0, True), (2, 4, True), (3, 128, False)])
def test_layer_operands_from_checkpoint_bytes(prep, layer, ratio, hash_layer):
    """Flash layers 0 (window, hash), 2 (C4, hash) and 3 (C128, score routing)."""
    from b12x.integration.cuteafd import FLASH
    from b12x.integration.cuteafd.weights import weight_sources

    device = require_b12x()
    if not has_checkpoint():
        pytest.skip("V4 Flash snapshot not mounted")
    expected = _packed_operands(layer, ratio, hash_layer, device)
    checked = set()
    for entry, shape in weight_sources(FLASH, ratio=ratio, hash_layer=hash_layer):
        key = (entry.program, entry.operand)
        if entry.program == "mhc_post_pre" or key not in expected:
            continue  # post_pre reuses the mhc_pre tensor names (hc_ffn_* / next layer), checked below
        actual = _engine_operand(entry, shape, layer, prep, device)
        torch.cuda.synchronize()
        assert_bitwise(actual, _storage_bytes(expected[key]), f"layer {layer} {key}")
        checked.add(key)
    assert checked == set(expected), set(expected) - checked
    # mhc_post_pre operands are raw hc_ffn_* / ffn_norm (after attention) tensors.
    for entry, _ in weight_sources(FLASH, ratio=ratio, hash_layer=hash_layer):
        if entry.program == "mhc_post_pre":
            name = entry.tensors[0].replace("{L}", str(layer)).replace("{L1}", str(layer + 1))
            tensor = checkpoint_tensor(name, device)
            assert tensor.dtype in (torch.float32, torch.bfloat16), name


@pytest.mark.parametrize(("n", "k", "groups"), [(1536, 7168, 1), (65536, 1536, 1), (1024, 8192, 16), (7168, 16384, 1)])
def test_scale_prep_matches_packer_pro_shapes(prep, n, k, groups):
    """Pro-sized weights with random UE8M0 scales (no Pro FP8 checkpoint mounted)."""
    from b12x.gemm._shared.wo_mxfp8 import pack_fp8_block_scaled_weight_mxfp8
    from b12x.integration.cuteafd.weights import block_fp8_scale_prep_args, scale_mma_bytes

    device = require_b12x()
    gen = torch.Generator().manual_seed(n + k + groups)
    scale = torch.randint(100, 140, (groups * n // 128, k // 128), generator=gen, dtype=torch.uint8).to(device)
    weight = torch.zeros((groups * n, k), dtype=torch.float8_e4m3fn, device=device)
    packed = pack_fp8_block_scaled_weight_mxfp8(weight, scale, m=n, k=k, num_groups=groups)
    out = torch.empty((scale_mma_bytes(n, k, groups),), dtype=torch.uint8, device=device)
    prep["scale"].launch(scale, out, scalars=block_fp8_scale_prep_args(n, k, groups))
    torch.cuda.synchronize()
    assert_bitwise(out, _storage_bytes(packed.scale_mma), "scale_mma")


def test_weight_table_covers_every_program_weight_operand():
    """Every non-activation input operand of the weight-bearing programs has a source."""
    from b12x.integration.cuteafd.weights import WEIGHT_SOURCES

    covered = {(e.program, e.operand) for e in WEIGHT_SOURCES}
    required = {
        "producer": ("w_qkv", "w_qkv_scale", "w_q", "w_q_scale", "q_norm", "kv_norm"),
        "index_producer": ("w_q", "w_q_scale", "w_proj"),
        "compressor": ("joint_projection", "main_ape", "main_norm", "index_ape", "index_norm"),
        "sparse_mla": ("attn_sink",),
        "wo_projection": ("wo_a", "wo_a_scale", "wo_b", "wo_b_scale"),
        "shared_ffn": ("w13", "w13_scale", "w2", "w2_scale"),
        "router_scores": ("w",),
        "mhc_pre": ("fn", "scale", "base", "norm"),
        "mhc_post_pre": ("fn", "scale", "base", "norm"),
        "mhc_head": ("fn", "scale", "base", "norm"),
    }
    missing = {(p, o) for p, ops in required.items() for o in ops} - covered
    assert not missing, missing


@pytest.mark.parametrize("key", ["scale", "narrow"])
def test_export_to_c_signature(prep, key, tmp_path):
    info = check_export(prep[key], tmp_path, f"dsv4_weight_prep_{key}")
    assert info["argument_count"] == len(prep[key].operands) + len(prep[key].scalars) + 1
