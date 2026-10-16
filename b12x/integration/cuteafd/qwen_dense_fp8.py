"""Native checkpoint E4M3 blocks with exact scales and BF16 GDN a/b gates.

Projection math is shared Fp8Projection. Only the mixed projection row packing
is local: neither BF16 gates nor FP8 checkpoint matrices are requantized.
"""
from __future__ import annotations

from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import QWEN38_27B, Operand, QwenDenseGeometry, Scalar, compile_program
from ._fp8_weights import Fp8Projection, fp8_only_operands, projection
from ._qwen4_attention_kernels import Qwen4AttnPost
from ._qwen4_kernels import Qwen4SwiGLU
from .qwen4_attention import compile_qwen4_attn_o_aot
from .qwen4_gdn import _Gdn, _ptr, gdn_scratch_bytes


def _align(n: int) -> int:
    return (int(n) + 1023) // 1024 * 1024


def _prefill(rows: int) -> bool:
    if int(rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(rows) > 64


def _projection(n: int, k: int, rows: int) -> Fp8Projection:
    return Fp8Projection(n, k, prefill_rows=int(rows) if _prefill(rows) else None,
                         decode_rows=64)


def _geometry(g: QwenDenseGeometry, rows: int) -> dict:
    return {"hidden": g.hidden, "max_rows": int(rows), "fp8_weights": "only",
            "weight_block": [128, 128], "mode": "prefill" if _prefill(rows) else "decode"}


class _PackGdn:
    def __init__(self, g: QwenDenseGeometry):
        self.quantized = g.gdn_conv_width + g.gdn_value_width
        self.gates = 2 * g.gdn_value_heads
        self.width = g.gdn_in_width

    @cute.jit
    def __call__(self, qkvz: cute.Pointer, ba: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(qkvz, ba, out, rows).launch(
            grid=((self.width + 255) // 256, rows, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkvz: cute.Pointer, ba: cute.Pointer, out: cute.Pointer, rows: Int32):
        col = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        row = Int64(cute.arch.block_idx()[1])
        if row < Int64(rows):
            if col < Int64(self.width):
                if col < Int64(self.quantized):
                    out[row * Int64(self.width) + col] = qkvz[row * Int64(self.quantized) + col]
                else:
                    out[row * Int64(self.width) + col] = ba[row * Int64(self.gates) + col - Int64(self.quantized)]


class _MixedGdn(_Gdn):
    def __init__(self, g: QwenDenseGeometry, max_rows: int):
        super().__init__(g, max_rows, activation="silu")
        self.quantized = self.c + self.v
        self.qkvz = _projection(self.quantized, g.hidden, max_rows)
        self.ba = projection(2 * g.gdn_value_heads, g.hidden, False)
        self.o_proj = _projection(g.hidden, self.v, max_rows)
        self.pack = _PackGdn(g)

    def key(self):
        return (self.g, self.qkvz.key(), self.ba.key(), self.o_proj.key(),
                None if self.chunked is None else self.chunked.key(), "native_bf16_gates_silu")

    def scratch_bytes(self, rows: int) -> int:
        return (gdn_scratch_bytes(self.g, rows, self.chunked_bytes())
                + _align(max(rows, 1) * self.quantized * 2)
                + _align(max(rows, 1) * 2 * self.g.gdn_value_heads * 2)
                + max(self.qkvz.prefill_scratch_bytes(rows), self.o_proj.prefill_scratch_bytes(rows)))

    @cute.jit
    def body_mixed(self, x: cute.Pointer, w_qkvz_fp8: cute.Pointer, w_qkvz_scale: cute.Pointer,
                   w_ba: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                   norm_w: cute.Pointer, w_out_fp8: cute.Pointer, w_out_scale: cute.Pointer,
                   conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                   out: cute.Pointer, replay: cute.Pointer, scratch: cute.Pointer,
                   rows: Int32, spec: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        base = Int64(scratch.toint())
        qkv_off = base + (m * Int64(self.p * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        o_off = qkv_off + (m * Int64(self.c * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        y_off = o_off + (m * Int64(self.v * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        quant_off = y_off + (m * Int64(self.v * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        quant_off = quant_off + Int64(_align(self.chunked_bytes()))
        ba_off = quant_off + (m * Int64(self.quantized * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        qscratch = ba_off + (m * Int64(4 * self.g.gdn_value_heads) + Int64(1023)) // Int64(1024) * Int64(1024)
        self.qkvz(x, w_qkvz_fp8, w_qkvz_scale, _ptr(cutlass.BFloat16, quant_off), rows,
                  Int32(255), qscratch, stream)
        self.ba(x, w_ba, w_ba, w_ba, _ptr(cutlass.BFloat16, ba_off), rows, stream)
        self.pack(_ptr(cutlass.BFloat16, quant_off), _ptr(cutlass.BFloat16, ba_off),
                  _ptr(cutlass.BFloat16, base), rows, stream)
        self.core(conv_w, a_log, dt_bias, norm_w, conv_state, state, slots, seq_first,
                  replay, scratch, rows, spec, stream)
        self.o_proj(_ptr(cutlass.BFloat16, y_off), w_out_fp8, w_out_scale, out, rows,
                    Int32(255), qscratch, stream)


class _MixedGdnDecode(_MixedGdn):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_qkvz_fp8: cute.Pointer, w_qkvz_scale: cute.Pointer,
                 w_ba: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                 norm_w: cute.Pointer, w_out_fp8: cute.Pointer, w_out_scale: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, replay: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, spec: Int32, stream: cuda.CUstream):
        self.body_mixed(x, w_qkvz_fp8, w_qkvz_scale, w_ba, conv_w, a_log, dt_bias, norm_w,
                        w_out_fp8, w_out_scale, conv_state, state, slots, seq_first, out,
                        replay, scratch, rows, spec, stream)


class _MixedGdnPrefill(_MixedGdn):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_qkvz_fp8: cute.Pointer, w_qkvz_scale: cute.Pointer,
                 w_ba: cute.Pointer, conv_w: cute.Pointer, a_log: cute.Pointer, dt_bias: cute.Pointer,
                 norm_w: cute.Pointer, w_out_fp8: cute.Pointer, w_out_scale: cute.Pointer,
                 conv_state: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.body_mixed(x, w_qkvz_fp8, w_qkvz_scale, w_ba, conv_w, a_log, dt_bias, norm_w,
                        w_out_fp8, w_out_scale, conv_state, state, slots, seq_first, out,
                        state, scratch, rows, Int32(0), stream)


def compile_qwen_dense_gdn_fp8_aot(g: QwenDenseGeometry = QWEN38_27B, *, max_rows: int):
    prefill = _prefill(max_rows)
    launch = (_MixedGdnPrefill if prefill else _MixedGdnDecode)(g, max_rows)
    replay = launch.replay["bytes"]
    return compile_program(launch, name="qwen_dense_gdn_fp8", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        *fp8_only_operands("w_qkvz", g.gdn_conv_width + g.gdn_value_width, g.hidden, prefill=prefill),
        Operand("w_ba", torch.bfloat16, f"[{2 * g.gdn_value_heads},{g.hidden}]"),
        Operand("conv_w", torch.float32, f"[{g.gdn_conv_width},4]", align=4),
        Operand("a_log", torch.float32, f"[{g.gdn_value_heads}]", align=4),
        Operand("dt_bias", torch.float32, f"[{g.gdn_value_heads}]", align=4),
        Operand("norm_w", torch.bfloat16, f"[{g.gdn_head_dim}]"),
        *fp8_only_operands("w_out", g.hidden, g.gdn_value_width, prefill=prefill),
        Operand("conv_state", torch.bfloat16, f"[slots,3,{g.gdn_conv_width}]", "inout", align=2),
        Operand("state", torch.float32, f"[slots,{g.gdn_value_heads},128,128]", "inout"),
        Operand("slots", torch.int32, "[rows]", align=4),
        Operand("seq_first", torch.int32, "[rows]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"),
        *((Operand("replay", torch.float32, f"[{replay // 4}]", "out"),) if not prefill else ()),
        Operand("scratch", torch.uint8, "[gdn_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),) if prefill else (Scalar("rows"), Scalar("spec")),
        key=(max_rows, launch.key()), geometry={**_geometry(g, max_rows),
            "key_heads": g.gdn_key_heads, "value_heads": g.gdn_value_heads, "head_dim": g.gdn_head_dim,
            "in_width": g.gdn_in_width, "eps": g.norm_eps, "output_gate": "silu",
            "norm_dtype": str(torch.bfloat16), "gate_weights": "bf16", "gate_order": ["b", "a"],
            "replay_rows": 64 if not prefill else None, "replay_bytes": replay if not prefill else None},
        scratch={"scratch": launch.scratch_bytes})


class _Producer:
    def __init__(self, g: QwenDenseGeometry, rows: int):
        self.g = g
        self.proj = _projection(g.attn_in_width, g.hidden, rows)
        self.post = Qwen4AttnPost(g, indexed=False)

    def key(self):
        return (self.g, self.proj.key(), "unrestricted")

    def scratch_bytes(self, rows: int):
        return _align(max(rows, 1) * self.g.attn_in_width * 2) + self.proj.prefill_scratch_bytes(rows)

    @cute.jit
    def __call__(self, x: cute.Pointer, w_in_fp8: cute.Pointer, w_in_scale: cute.Pointer,
                 q_norm: cute.Pointer, k_norm: cute.Pointer, rope_positions: cute.Pointer,
                 kv_slots: cute.Pointer, kv_cache: cute.Pointer, query: cute.Pointer, gate: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        projected = _ptr(cutlass.BFloat16, base)
        qscratch = base + (Int64(rows) * Int64(self.g.attn_in_width * 2) + Int64(1023)) // Int64(1024) * Int64(1024)
        self.proj(x, w_in_fp8, w_in_scale, projected, rows, Int32(255), qscratch, stream)
        self.post(projected, q_norm, k_norm, q_norm, k_norm, rope_positions, kv_slots, kv_cache,
                  query, query, gate, query, rows, stream)


def compile_qwen_dense_producer_fp8_aot(g: QwenDenseGeometry = QWEN38_27B, *, max_rows: int):
    launch = _Producer(g, max_rows)
    return compile_program(launch, name="qwen_dense_producer_fp8", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        *fp8_only_operands("w_in", g.attn_in_width, g.hidden, prefill=_prefill(max_rows)),
        Operand("q_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("k_norm", torch.bfloat16, f"[{g.head_dim}]"),
        Operand("rope_positions", torch.int32, "[rows,3]", align=4),
        Operand("kv_slots", torch.int64, "[rows]", align=8),
        Operand("kv_cache", torch.uint8, f"[record_slots,{g.record_bytes}]", "inout"),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]", "out"),
        Operand("gate", torch.bfloat16, f"[rows,{g.heads * g.head_dim}]", "out"),
        Operand("scratch", torch.uint8, "[producer_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={**_geometry(g, max_rows), "heads": g.heads, "kv_heads": g.kv_heads,
                  "head_dim": g.head_dim, "in_width": g.attn_in_width, "record_bytes": g.record_bytes,
                  "rope_dim": g.rope_dim, "rope_theta": g.rope_theta, "indexed": False},
        scratch={"scratch": launch.scratch_bytes})


class _Ffn:
    def __init__(self, g: QwenDenseGeometry, rows: int):
        self.g = g
        self.up = _projection(2 * g.intermediate, g.hidden, rows)
        self.act = Qwen4SwiGLU(g.intermediate, 2 * g.intermediate)
        self.down = _projection(g.hidden, g.intermediate, rows)

    def key(self):
        return (self.g, self.up.key(), self.down.key())

    def scratch_bytes(self, rows: int):
        return (_align(max(rows, 1) * self.g.intermediate * 4)
                + _align(max(rows, 1) * self.g.intermediate * 2)
                + max(self.up.prefill_scratch_bytes(rows), self.down.prefill_scratch_bytes(rows)))

    @cute.jit
    def __call__(self, x: cute.Pointer, w_gate_up_fp8: cute.Pointer, w_gate_up_scale: cute.Pointer,
                 w_down_fp8: cute.Pointer, w_down_scale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        hidden = base + (Int64(rows) * Int64(4 * self.g.intermediate) + Int64(1023)) // Int64(1024) * Int64(1024)
        qscratch = hidden + (Int64(rows) * Int64(2 * self.g.intermediate) + Int64(1023)) // Int64(1024) * Int64(1024)
        self.up(x, w_gate_up_fp8, w_gate_up_scale, _ptr(cutlass.BFloat16, base), rows,
                Int32(255), qscratch, stream)
        self.act(_ptr(cutlass.BFloat16, base), _ptr(cutlass.BFloat16, hidden), rows, stream)
        self.down(_ptr(cutlass.BFloat16, hidden), w_down_fp8, w_down_scale, out, rows,
                  Int32(255), qscratch, stream)


def compile_qwen_dense_ffn_fp8_aot(g: QwenDenseGeometry = QWEN38_27B, *, max_rows: int):
    launch = _Ffn(g, max_rows)
    return compile_program(launch, name="qwen_dense_ffn_fp8", operands=(
        Operand("x", torch.bfloat16, f"[rows,{g.hidden}]"),
        *fp8_only_operands("w_gate_up", 2 * g.intermediate, g.hidden, prefill=_prefill(max_rows)),
        *fp8_only_operands("w_down", g.hidden, g.intermediate, prefill=_prefill(max_rows)),
        Operand("out", torch.bfloat16, f"[rows,{g.hidden}]", "out"),
        Operand("scratch", torch.uint8, "[ffn_scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry={**_geometry(g, max_rows), "intermediate": g.intermediate},
        scratch={"scratch": launch.scratch_bytes})


def compile_qwen_dense_attn_o_fp8_aot(g: QwenDenseGeometry = QWEN38_27B, *, max_rows: int):
    mode = "prefill" if _prefill(max_rows) else "decode"
    program = compile_qwen4_attn_o_aot(g, max_rows=max_rows, fp8_only=mode)
    return replace(program, name="qwen_dense_attn_o_fp8", geometry={**program.geometry,
                   "weight_block": [128, 128]})
