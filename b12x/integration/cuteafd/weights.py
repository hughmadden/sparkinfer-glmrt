"""Checkpoint -> AOT weight operands for DeepSeek V4 (Flash / Pro).

A native engine that loads raw safetensors bytes onto the GPU can build every
weight operand of every ``cuteafd`` program from :data:`WEIGHT_SOURCES` and the
two preparation programs below; the result is byte-identical to what the
prepared b12x Python packers (``dsv4_producer.pack_weights``,
``pack_indexer_weights``, ``dsv4_compressor.pack_weights``,
``wo_projection.pack_weights``, ``block_fp8_linear.pack_weight``) build.

Every FP8 GEMM weight keeps its checkpoint E4M3 bytes (``values`` is raw,
row-major ``[N, K]``; grouped ``wo_a`` stays ``[G*R, W]``). Only the
checkpoint's 128x128 UE8M0 block scales need preparation: the dense GEMM
reads them in MMA tile order (``scale_mma``), produced by
:func:`compile_dsv4_block_fp8_scale_prep_aot`. The one dtype conversion in the
checkpoint is the hash-layer ``tid2eid`` table (I64), which the prototype
narrows to int32 (:func:`compile_dsv4_i64_to_i32_aot`); no exported program
consumes it, so it is listed for the engine's hash routing.

Mapping (``{L}`` layer index; "raw" = checkpoint bytes as-is; "cat0" = the
listed tensors concatenated along rows (axis 0, in order) into one buffer;
``prep(...)`` = run :func:`compile_dsv4_block_fp8_scale_prep_aot` with those
arguments on the (concatenated) UE8M0 scale bytes)::

    program.operand                  checkpoint tensors (per layer)                      transform
    mhc_pre.fn / scale / base        layers.{L}.hc_attn_fn / hc_attn_scale / hc_attn_base  raw (F32)
    mhc_pre.norm                     layers.{L}.attn_norm.weight                           raw (BF16)
    mhc_post_pre.fn/scale/base/norm  after attention: layers.{L}.hc_ffn_* + ffn_norm.weight raw
                                     after the FFN (fused into layer L+1's pre):
                                     layers.{L+1}.hc_attn_* + attn_norm.weight              raw
    mhc_head.fn / scale / base       hc_head_fn / hc_head_scale / hc_head_base             raw (F32)
    mhc_head.norm                    norm.weight                                            raw
    producer.w_qkv                   attn.wq_a.weight, attn.wkv.weight                      cat0 (E4M3 [Q+512,H])
    producer.w_qkv_scale             attn.wq_a.scale, attn.wkv.scale                        cat0 -> prep(Q+512, H)
    producer.w_q / w_q_scale         attn.wq_b.weight / .scale                              raw / prep(N*512, Q)
    producer.q_norm / kv_norm        attn.q_norm.weight / attn.kv_norm.weight               raw
    index_producer.w_q / w_q_scale   attn.indexer.wq_b.weight / .scale                      raw / prep(8192, Q)
    index_producer.w_proj            attn.indexer.weights_proj.weight                       raw (BF16 [64,H])
    compressor(C4).joint_projection  attn.compressor.wkv.weight, attn.compressor.wgate.weight,
                                     attn.indexer.compressor.wkv.weight,
                                     attn.indexer.compressor.wgate.weight                   cat0 (BF16 [2560,H])
    compressor(C128).joint_projection attn.compressor.wkv.weight, .wgate.weight              cat0 (BF16 [1024,H])
    compressor.main_ape / main_norm  attn.compressor.ape / attn.compressor.norm.weight      raw (F32 / BF16)
    compressor.index_ape/index_norm  attn.indexer.compressor.ape / .norm.weight             raw (C4 only)
    sparse_mla.attn_sink             attn.attn_sink                                         raw (F32)
    wo.wo_a / wo_a_scale             attn.wo_a.weight / .scale                              raw / prep(R, W, groups=G)
    wo.wo_b / wo_b_scale             attn.wo_b.weight / .scale                              raw / prep(H, G*R)
    shared_ffn.w13                   ffn.shared_experts.w1.weight, .w3.weight               cat0 (gate rows first)
    shared_ffn.w13_scale             ffn.shared_experts.w1.scale, .w3.scale                 cat0 -> prep(2I, H)
    shared_ffn.w2 / w2_scale         ffn.shared_experts.w2.weight / .scale                  raw / prep(H, I)
    router_scores.w                  ffn.gate.weight                                        raw (BF16 [E,H])
    (engine) score-routing bias      ffn.gate.bias                  (non-hash layers)       raw (F32 [E])
    (engine) hash routing table      ffn.gate.tid2eid               (hash layers < n_hash)  I64 -> int32 (narrow)

Geometry: ``H`` hidden, ``Q`` q_lora_rank, ``N`` heads, ``G`` o_groups,
``R`` o_lora_rank, ``W`` = N*512/G, ``I`` moe_inter, ``E`` routed experts.
Layers with compress ratio 0 have no compressor/indexer tensors; ratio-128
layers have no indexer. Routed experts are outside this table.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import AotProgram, DSV4Geometry, Operand, Scalar, compile_program

__all__ = [
    "WEIGHT_SOURCES",
    "WeightSource",
    "block_fp8_scale_prep_args",
    "compile_dsv4_block_fp8_scale_prep_aot",
    "compile_dsv4_i64_to_i32_aot",
    "scale_mma_bytes",
    "weight_sources",
]


@dataclass(frozen=True)
class WeightSource:
    """How one program weight operand is built from checkpoint tensors.

    ``tensors`` are checkpoint names (``{L}`` = layer, ``{L1}`` = next layer)
    concatenated along axis 0 in order when there are several. ``prep`` is
    ``None`` for raw bytes, ``"block_fp8_scale"`` for
    :func:`compile_dsv4_block_fp8_scale_prep_aot` with ``prep_shape`` =
    ``(n, k, groups)`` (geometry attribute expressions), or ``"narrow_i64"``.
    ``layers`` limits the entry to ``"all"``, ``"c4"``, ``"c128"``,
    ``"compressed"`` (ratio 4 or 128), ``"hash"`` or ``"score"`` layers,
    or ``"model"`` (not per layer).
    """

    program: str
    operand: str
    tensors: tuple[str, ...]
    prep: str | None = None
    prep_shape: tuple[str, str, str] | None = None
    layers: str = "all"
    note: str = ""


_A = "layers.{L}.attn."
_F = "layers.{L}.ffn."

WEIGHT_SOURCES: tuple[WeightSource, ...] = (
    WeightSource("mhc_pre", "fn", ("layers.{L}.hc_attn_fn",)),
    WeightSource("mhc_pre", "scale", ("layers.{L}.hc_attn_scale",)),
    WeightSource("mhc_pre", "base", ("layers.{L}.hc_attn_base",)),
    WeightSource("mhc_pre", "norm", ("layers.{L}.attn_norm.weight",)),
    WeightSource("mhc_post_pre", "fn", ("layers.{L}.hc_ffn_fn",), note="after attention"),
    WeightSource("mhc_post_pre", "scale", ("layers.{L}.hc_ffn_scale",), note="after attention"),
    WeightSource("mhc_post_pre", "base", ("layers.{L}.hc_ffn_base",), note="after attention"),
    WeightSource("mhc_post_pre", "norm", ("layers.{L}.ffn_norm.weight",), note="after attention"),
    WeightSource("mhc_post_pre", "fn", ("layers.{L1}.hc_attn_fn",), note="after the FFN, into layer L+1"),
    WeightSource("mhc_post_pre", "scale", ("layers.{L1}.hc_attn_scale",), note="after the FFN, into layer L+1"),
    WeightSource("mhc_post_pre", "base", ("layers.{L1}.hc_attn_base",), note="after the FFN, into layer L+1"),
    WeightSource("mhc_post_pre", "norm", ("layers.{L1}.attn_norm.weight",), note="after the FFN, into layer L+1"),
    WeightSource("mhc_head", "fn", ("hc_head_fn",), layers="model"),
    WeightSource("mhc_head", "scale", ("hc_head_scale",), layers="model"),
    WeightSource("mhc_head", "base", ("hc_head_base",), layers="model"),
    WeightSource("mhc_head", "norm", ("norm.weight",), layers="model"),
    WeightSource("producer", "w_qkv", (_A + "wq_a.weight", _A + "wkv.weight")),
    WeightSource("producer", "w_qkv_scale", (_A + "wq_a.scale", _A + "wkv.scale"),
                 "block_fp8_scale", ("q_lora_rank + 512", "hidden", "1")),
    WeightSource("producer", "w_q", (_A + "wq_b.weight",)),
    WeightSource("producer", "w_q_scale", (_A + "wq_b.scale",),
                 "block_fp8_scale", ("heads * 512", "q_lora_rank", "1")),
    WeightSource("producer", "q_norm", (_A + "q_norm.weight",)),
    WeightSource("producer", "kv_norm", (_A + "kv_norm.weight",)),
    WeightSource("index_producer", "w_q", (_A + "indexer.wq_b.weight",), layers="c4"),
    WeightSource("index_producer", "w_q_scale", (_A + "indexer.wq_b.scale",),
                 "block_fp8_scale", ("index_heads * index_head_dim", "q_lora_rank", "1"), layers="c4"),
    WeightSource("index_producer", "w_proj", (_A + "indexer.weights_proj.weight",), layers="c4"),
    WeightSource("compressor", "joint_projection",
                 (_A + "compressor.wkv.weight", _A + "compressor.wgate.weight",
                  _A + "indexer.compressor.wkv.weight", _A + "indexer.compressor.wgate.weight"),
                 layers="c4"),
    WeightSource("compressor", "joint_projection",
                 (_A + "compressor.wkv.weight", _A + "compressor.wgate.weight"), layers="c128"),
    WeightSource("compressor", "main_ape", (_A + "compressor.ape",), layers="compressed"),
    WeightSource("compressor", "main_norm", (_A + "compressor.norm.weight",), layers="compressed"),
    WeightSource("compressor", "index_ape", (_A + "indexer.compressor.ape",), layers="c4"),
    WeightSource("compressor", "index_norm", (_A + "indexer.compressor.norm.weight",), layers="c4"),
    WeightSource("sparse_mla", "attn_sink", (_A + "attn_sink",)),
    WeightSource("wo_projection", "wo_a", (_A + "wo_a.weight",)),
    WeightSource("wo_projection", "wo_a_scale", (_A + "wo_a.scale",),
                 "block_fp8_scale", ("o_lora_rank", "o_group_width", "o_groups")),
    WeightSource("wo_projection", "wo_b", (_A + "wo_b.weight",)),
    WeightSource("wo_projection", "wo_b_scale", (_A + "wo_b.scale",),
                 "block_fp8_scale", ("hidden", "o_groups * o_lora_rank", "1")),
    WeightSource("shared_ffn", "w13", (_F + "shared_experts.w1.weight", _F + "shared_experts.w3.weight"),
                 note="gate (w1) rows first"),
    WeightSource("shared_ffn", "w13_scale", (_F + "shared_experts.w1.scale", _F + "shared_experts.w3.scale"),
                 "block_fp8_scale", ("2 * moe_inter", "hidden", "1")),
    WeightSource("shared_ffn", "w2", (_F + "shared_experts.w2.weight",)),
    WeightSource("shared_ffn", "w2_scale", (_F + "shared_experts.w2.scale",),
                 "block_fp8_scale", ("hidden", "moe_inter", "1")),
    WeightSource("router_scores", "w", (_F + "gate.weight",)),
    WeightSource("engine", "gate_bias", (_F + "gate.bias",), layers="score",
                 note="F32 [E] score-correction bias for top-k selection"),
    WeightSource("engine", "tid2eid", (_F + "gate.tid2eid",), "narrow_i64", layers="hash",
                 note="hash routing expert ids; the prototype uses int32"),
)


def _eval(expression: str, geometry: DSV4Geometry) -> int:
    names = {name: getattr(geometry, name) for name in dir(geometry) if not name.startswith("_")}
    return int(eval(expression, {"__builtins__": {}}, names))  # noqa: S307 - static table strings


def weight_sources(geometry: DSV4Geometry, *, ratio: int, hash_layer: bool) -> list[tuple[WeightSource, tuple[int, ...] | None]]:
    """Entries applying to one layer, with ``prep_shape`` evaluated to ints."""
    selected = []
    for entry in WEIGHT_SOURCES:
        applies = {
            "all": True, "model": False, "c4": ratio == 4, "c128": ratio == 128,
            "compressed": ratio in (4, 128), "hash": hash_layer, "score": not hash_layer,
        }[entry.layers]
        if not applies:
            continue
        shape = None if entry.prep_shape is None else tuple(_eval(e, geometry) for e in entry.prep_shape)
        selected.append((entry, shape))
    return selected


def scale_mma_bytes(n: int, k: int, groups: int = 1) -> int:
    """Bytes of a packed dense-GEMM scale operand for weight [groups*n, k]."""
    return int(groups) * ((int(n) + 127) // 128) * ((int(k) + 127) // 128) * 512


def block_fp8_scale_prep_args(n: int, k: int, groups: int = 1) -> tuple[int, int]:
    """``(n_blocks, k_blocks)`` scalars for the scale-prep program."""
    if n % 128 or k % 128:
        raise ValueError("DSV4 block-FP8 weights have N and K multiples of 128")
    return groups * n // 128, k // 128


class _BlockScalePrep:
    """scale_mma[t*512 + j] = scale[t // k_blocks, t % k_blocks] for every tile t.

    The MMA scale tile covers 128 rows x 4 K32 groups (= one 128x128 weight
    block), so each 512-byte tile replicates one checkpoint UE8M0 byte; tiles
    are ordered (row block, K block) with groups folded into row blocks.
    """

    @cute.jit
    def __call__(self, scale: cute.Pointer, scale_mma: cute.Pointer, n_blocks: Int32,
                 k_blocks: Int32, stream: cuda.CUstream):
        self.kernel(scale, scale_mma, k_blocks).launch(
            grid=(n_blocks * k_blocks, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, scale: cute.Pointer, scale_mma: cute.Pointer, k_blocks: Int32):
        tile = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        byte = cutlass.Uint32(cute.make_ptr(cutlass.Uint8, Int64(scale.toint()) + tile,
                                            cute.AddressSpace.gmem, assumed_align=1)[0])
        word = byte | (byte << cutlass.Uint32(8)) | (byte << cutlass.Uint32(16)) | (byte << cutlass.Uint32(24))
        cute.make_ptr(cutlass.Uint32, Int64(scale_mma.toint()) + tile * Int64(512) + Int64(4) * tidx,
                      cute.AddressSpace.gmem, assumed_align=4)[0] = word


def compile_dsv4_block_fp8_scale_prep_aot() -> AotProgram:
    """Checkpoint 128x128 UE8M0 block scales -> dense-GEMM ``scale_mma`` bytes.

    ABI::

        scale      u8 [n_blocks, k_blocks]      in   raw (concatenated) checkpoint scales
        scale_mma  u8 [n_blocks*k_blocks*512]   out  == pack_*(...).weight.scale_mma storage
        n_blocks   int32   groups * N / 128
        k_blocks   int32   K / 128

    Use :func:`block_fp8_scale_prep_args` for the scalars. Tile ``t`` (row
    block ``t // k_blocks``, K block ``t % k_blocks``) is 512 copies of scale
    byte ``t``, which is the layout ``pack_mxfp8_scales_for_dense_gemm``
    produces for 128x128 blocks (row-in-tile ``r4*32 + r32``, K32 group
    ``k4``, byte ``((tile*32 + r32)*4 + r4)*4 + k4``).
    """
    return compile_program(
        _BlockScalePrep(), name="dsv4_block_fp8_scale_prep",
        operands=(Operand("scale", torch.uint8, "[n_blocks,k_blocks]", align=1),
                  Operand("scale_mma", torch.uint8, "[n_blocks*k_blocks*512]", "out")),
        scalars=(Scalar("n_blocks"), Scalar("k_blocks")), key=(), doc=__doc__,
    )


class _NarrowI64:
    @cute.jit
    def __call__(self, source: cute.Pointer, out: cute.Pointer, count: Int32, stream: cuda.CUstream):
        self.kernel(source, out, count).launch(
            grid=((count + Int32(255)) // Int32(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, out: cute.Pointer, count: Int32):
        index = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if index < Int64(count):
            value = cute.make_ptr(cutlass.Int64, Int64(source.toint()) + Int64(8) * index,
                                  cute.AddressSpace.gmem, assumed_align=8)[0]
            cute.make_ptr(Int32, Int64(out.toint()) + Int64(4) * index, cute.AddressSpace.gmem,
                          assumed_align=4)[0] = Int32(value)


def compile_dsv4_i64_to_i32_aot() -> AotProgram:
    """Narrow ``count`` int64 values to int32 (``tid2eid``: 129280*6 entries).

    ABI: ``source`` i64 [count] in, ``out`` i32 [count] out, ``count`` int32.
    """
    return compile_program(
        _NarrowI64(), name="dsv4_i64_to_i32",
        operands=(Operand("source", torch.int64, "[count]", align=8),
                  Operand("out", torch.int32, "[count]", "out", align=4)),
        scalars=(Scalar("count"),), key=(), doc=__doc__,
    )


def _check() -> None:  # table sanity for imports/tests
    for entry in WEIGHT_SOURCES:
        if (entry.prep == "block_fp8_scale") != (entry.prep_shape is not None):
            raise AssertionError(f"{entry.program}.{entry.operand}: prep_shape mismatch")


_check()
