"""Kimi K3 attention-residual normalization and standalone Situ AOT kernels.

The caller meets TP ranks before launch. Arena slot 8 is the prefix; slots
0..C-2 are immutable blocks. Append stores the unmixed prefix in slot C-1;
the next boundary FFN uses replace, rather than resetting during append.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
import torch

from b12x._lib.intrinsics import div_rn_f32, fmax_f32, pack_f32x2_to_bfloat2
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _bf16, _block_sum, _reduction_storage, _rsqrt


def _candidate_storage(words):
    class Storage:
        pass

    Storage.__annotations__ = {
        "values": cute.struct.Align[cute.struct.MemRange[Uint32, words], 16],
    }
    return cute.struct(Storage)


class KimiAttnResRmsNorm:
    threads = 512

    def __init__(self, hidden, candidates, prefix_mode, append, has_mix):
        if hidden != 7168 or not 1 <= candidates <= 9:
            raise ValueError("Kimi residual requires H7168 and 1..9 candidates")
        if prefix_mode not in ("keep", "add", "replace"):
            raise ValueError("prefix_mode must be keep, add or replace")
        if append and candidates == 9:
            raise ValueError("all eight immutable block slots are already occupied")
        if not has_mix and candidates != 1:
            raise ValueError("no-mix is only valid for the initial single prefix")
        self.hidden, self.candidates = hidden, candidates
        self.prefix_mode, self.append, self.has_mix = prefix_mode, append, has_mix
        self.elements = hidden // self.threads
        self.warps = self.threads // 32
        # Bound register liveness without exceeding SM120's per-CTA SMEM limit.
        self.shared_candidates = max(0, candidates - 3) if candidates >= 5 else 0

    @cute.jit
    def __call__(self, arena: cute.Pointer, delta: cute.Pointer,
                 score_weight: cute.Pointer, norm_weight: cute.Pointer,
                 out: cute.Pointer, rows: Int32, arena_rows: Int32,
                 eps: Float32, stream: cuda.CUstream):
        self.kernel(arena, delta, score_weight, norm_weight, out, arena_rows, eps).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, arena: cute.Pointer, delta: cute.Pointer,
               score_weight: cute.Pointer, norm_weight: cute.Pointer,
               out: cute.Pointer, arena_rows: Int32, eps: Float32):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int64(cute.arch.thread_idx()[0])
        stride = Int64(arena_rows) * Int64(self.hidden)
        base = row * Int64(self.hidden)
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(_reduction_storage(2 * self.candidates + 1, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout(
            (2 * self.candidates + 1, self.warps), stride=(self.warps, 1)))
        # Each lane owns its shared words; the thread-contiguous layout is
        # bank-conflict-free and never changes the FP32 reduction order.
        if cutlass.const_expr(self.shared_candidates > 0):
            parked = allocator.allocate(_candidate_storage(self.shared_candidates * self.hidden // 2))
            shared = parked.values.get_tensor(cute.make_layout(
                (self.shared_candidates, self.elements // 2, self.threads),
                stride=(self.hidden // 2, self.threads, 1)))
        values = cute.make_rmem_tensor((self.candidates - self.shared_candidates, self.elements // 2), Uint32)
        current = cute.make_rmem_tensor((self.elements,), Float32)
        scores = cute.make_rmem_tensor((self.candidates,), Float32)
        mixed = cute.make_rmem_tensor((self.elements,), Float32)
        weight = cute.make_rmem_tensor((self.elements,), Float32)
        if cutlass.const_expr(self.has_mix):
            for j in cutlass.range_constexpr(self.elements):
                weight[j] = Float32(score_weight[tid + Int64(j * self.threads)])
        for c in cutlass.range_constexpr(self.candidates):
            for j in cutlass.range_constexpr(self.elements):
                at = base + tid + Int64(j * self.threads)
                if cutlass.const_expr(c == self.candidates - 1):
                    if cutlass.const_expr(self.prefix_mode == "replace"):
                        value = Float32(delta[at])
                    else:
                        value = Float32(arena[Int64(8) * stride + at])
                        if cutlass.const_expr(self.prefix_mode == "add"):
                            value = _bf16(value + Float32(delta[at]))
                    if cutlass.const_expr(self.prefix_mode != "keep"):
                        arena[Int64(8) * stride + at] = value.to(BFloat16)
                else:
                    value = Float32(arena[Int64(c) * stride + at])
                current[j] = value
            for j in cutlass.range_constexpr(self.elements // 2):
                packed = pack_f32x2_to_bfloat2(current[2 * j], current[2 * j + 1])
                if cutlass.const_expr(c < self.shared_candidates):
                    shared[c, j, tid] = packed
                else:
                    values[c - self.shared_candidates, j] = packed
            if cutlass.const_expr(self.has_mix):
                square = Float32(0)
                for j in cutlass.range_constexpr(self.elements):
                    square = square + current[j] * current[j]
                total = _block_sum(square, sums, Int32(2 * c), self.warps)
                inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + eps)
                dot = Float32(0)
                for j in cutlass.range_constexpr(self.elements):
                    dot = dot + (current[j] * inv) * weight[j]
                scores[c] = _block_sum(dot, sums, Int32(2 * c + 1), self.warps)
        if cutlass.const_expr(self.has_mix):
            maximum = scores[0]
            for c in cutlass.range_constexpr(1, self.candidates):
                maximum = fmax_f32(maximum, scores[c])
            denominator = Float32(0)
            for c in cutlass.range_constexpr(self.candidates):
                scores[c] = cute.math.exp(scores[c] - maximum, fastmath=False)
                denominator = denominator + scores[c]
            for c in cutlass.range_constexpr(self.candidates):
                scores[c] = div_rn_f32(scores[c], denominator)
        square = Float32(0)
        for j in cutlass.range_constexpr(self.elements):
            value = current[j]
            if cutlass.const_expr(self.has_mix):
                value = Float32(0)
                for c in cutlass.range_constexpr(self.candidates):
                    if cutlass.const_expr(c < self.shared_candidates):
                        packed = shared[c, j // 2, tid]
                    else:
                        packed = values[c - self.shared_candidates, j // 2]
                    if cutlass.const_expr(j % 2 == 0):
                        candidate = _bf16_lo(packed)
                    else:
                        candidate = _bf16_hi(packed)
                    value = value + candidate * scores[c]
                value = _bf16(value)
            mixed[j] = value
            square = square + value * value
            if cutlass.const_expr(self.append):
                at = base + tid + Int64(j * self.threads)
                arena[Int64(self.candidates - 1) * stride + at] = current[j].to(BFloat16)
        total = _block_sum(square, sums, Int32(2 * self.candidates), self.warps)
        inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + eps)
        for j in cutlass.range_constexpr(self.elements):
            d = tid + Int64(j * self.threads)
            out[base + d] = (_bf16(mixed[j] * inv) * Float32(norm_weight[d])).to(BFloat16)


class KimiSitu:
    threads = 256

    def __init__(self, intermediate):
        if intermediate not in (3072, 6144, 16896, 33792):
            raise ValueError("unsupported Kimi dense/shared intermediate width")
        self.intermediate = intermediate

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(gate_up, out).launch(
            grid=((self.intermediate + self.threads - 1) // self.threads, rows, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Pointer, out: cute.Pointer):
        tile, row, _ = cute.arch.block_idx()
        d = Int64(tile) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if d < Int64(self.intermediate):
            base = Int64(row) * Int64(2 * self.intermediate)
            gate = Float32(gate_up[base + d])
            up = Float32(gate_up[base + Int64(self.intermediate) + d])
            sigmoid = div_rn_f32(Float32(1), Float32(1) + cute.math.exp(-gate, fastmath=False))
            a = Float32(4) * cute.math.tanh(div_rn_f32(gate, Float32(4)), fastmath=False) * sigmoid
            b = Float32(25) * cute.math.tanh(div_rn_f32(up, Float32(25)), fastmath=False)
            out[Int64(row) * Int64(self.intermediate) + d] = (a * b).to(BFloat16)


def compile_kimi_attn_res_rmsnorm_aot(hidden=7168, candidates=2, prefix_mode="keep",
                                    append=False, has_mix=True):
    """Out must not alias arena or delta; 0 <= rows <= arena_rows is required."""
    kernel = KimiAttnResRmsNorm(hidden, candidates, prefix_mode, append, has_mix)
    return compile_program(kernel, name="kimi_attn_res_rmsnorm", operands=(
        Operand("arena", torch.bfloat16, f"[9,arena_rows,{hidden}]", "inout"),
        Operand("delta", torch.bfloat16, f"[rows,{hidden}]"),
        Operand("score_weight", torch.float32, f"[{hidden}]"),
        Operand("norm_weight", torch.bfloat16, f"[{hidden}]"),
        Operand("out", torch.bfloat16, f"[rows,{hidden}]", "out")),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("eps", "float32")),
        key=(hidden, candidates, prefix_mode, append, has_mix),
        geometry=dict(hidden=hidden, candidates=candidates, prefix_mode=prefix_mode,
                      append=append, has_mix=has_mix),
        doc="Fixed-stride BF16 prefix update, optional unmixed append, residual mix and RMSNorm.")


def compile_kimi_situ_aot(intermediate=3072):
    return compile_program(KimiSitu(intermediate), name="kimi_situ", operands=(
        Operand("gate_up", torch.bfloat16, f"[rows,{2 * intermediate}]"),
        Operand("out", torch.bfloat16, f"[rows,{intermediate}]", "out")),
        scalars=(Scalar("rows"),), key=(intermediate,),
        geometry=dict(intermediate=intermediate),
        doc="Gate-first Situ in FP32 with one final BF16 rounding.")
