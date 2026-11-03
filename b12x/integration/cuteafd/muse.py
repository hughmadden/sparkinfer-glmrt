"""Muse Spark coordinator boundaries; fixed arithmetic independent of live rows.

BF16 materialization is intentional at each checkpoint tensor-operation boundary.
All pool/row offsets are 64-bit. Compile before capture; launch allocates nothing.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Float64, Int32, Int64

from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _bf16, _warp_sum, _warp_max
from ._qwen4_kernels import _sigmoid
from .pointwise import _add, _mul, _sqrt_rn, _view


# Router tile geometry. The grid is derived from the live row count only, so one
# compiled program serves every SM120/SM121 card; nothing here depends on an SM
# count, L2 size or wave size. The dispatch threshold is the measured crossover
# between the scalar GEMV and the register tile (16 rows scalar beats tiled by
# 1.5x, 64 rows tiled beats scalar by 1.6x on RTX PRO 6000 and Max-Q).
ROUTER_SCALAR_MAX_ROWS = 16
ROUTER_TILE_ROWS = 8
ROUTER_TILE_EXPERTS = 8
ROUTER_BLOCK_WARPS = 8
ROUTER_BLOCK_ROW_GROUPS = 4


class _RouterScalar:
    """Native F32 checkpoint weights; BF16 input widened before each product."""
    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, weight, out).launch(grid=(rows, 32, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        x, weight, out = _view(x), _view(weight), _view(out)
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        expert = Int64(cute.arch.block_idx()[1]) * Int64(8) + Int64(tid // Int32(32))
        lane = Int64(tid % Int32(32))
        total = Float32(0.0)
        for chunk in cutlass.range(256, unroll=1):
            col = Int64(chunk) * Int64(32) + lane
            total = _add(total, _mul(Float32(x[row * Int64(8192) + col]),
                                     Float32(weight[expert * Int64(8192) + col])))
        total = _warp_sum(total)
        if lane == Int64(0):
            out[row * Int64(256) + expert] = total


class _RouterTiled:
    """Register-tiled router: one warp owns a ROWS x EXPERTS output tile.

    Warps of a CTA stack ``row_groups`` tiles deep on rows and
    ``block_warps // row_groups`` wide on experts, so the CTA re-reads the same
    input lines and the same weight lines from L1 instead of L2.

    Bit-identical to :class:`_RouterScalar` by construction: for every
    ``(row, expert)`` pair, lane ``l`` still sums the columns ``32*k + l`` for
    ``k = 0..255`` in ascending order, each term a separate IEEE ``mul.rn.f32``
    followed by a separate ``add.rn.f32`` from the accumulator's initial
    ``+0.0``, and the five-stage butterfly warp sum is the shared
    ``_warp_sum``. Tiling only changes which loaded value a lane reuses; the
    per-product and per-addition rounding sequence is unchanged, so the stored
    bits are the same as the scalar kernel's for the same rows.
    """

    def __init__(self, rows_per_warp: int = ROUTER_TILE_ROWS,
                 experts_per_warp: int = ROUTER_TILE_EXPERTS,
                 block_warps: int = ROUTER_BLOCK_WARPS,
                 row_groups: int = ROUTER_BLOCK_ROW_GROUPS):
        self.rows, self.experts = int(rows_per_warp), int(experts_per_warp)
        self.warps, self.row_groups = int(block_warps), int(row_groups)
        self.expert_groups = self.warps // self.row_groups
        if self.row_groups * self.expert_groups != self.warps:
            raise ValueError("router block warps must split into row groups")
        if 256 % (self.experts * self.expert_groups):
            raise ValueError("router expert groups must divide 256 experts")

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        block_rows = self.rows * self.row_groups
        self.kernel(x, weight, out, rows).launch(
            grid=((rows + Int32(block_rows - 1)) // Int32(block_rows),
                  256 // (self.experts * self.expert_groups), 1),
            block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer, rows: Int32):
        x, weight, out = _view(x), _view(weight), _view(out)
        tid = Int32(cute.arch.thread_idx()[0])
        warp = Int64(tid) // Int64(32)
        lane = Int64(tid % Int32(32))
        row_base = (Int64(cute.arch.block_idx()[0]) * Int64(self.row_groups)
                    + warp // Int64(self.expert_groups)) * Int64(self.rows)
        expert_base = (Int64(cute.arch.block_idx()[1]) * Int64(self.expert_groups)
                       + warp % Int64(self.expert_groups)) * Int64(self.experts)
        live, last = Int64(rows), Int64(rows) - Int64(1)
        # Clamp every row slot to the last live row so a partial tile never reads
        # or writes outside `x`; the clamped slots are computed and not stored.
        col_at = cute.make_rmem_tensor(cute.make_layout((self.rows,), stride=(1,)), Int64)
        weight_at = cute.make_rmem_tensor(cute.make_layout((self.experts,), stride=(1,)), Int64)
        x_value = cute.make_rmem_tensor(cute.make_layout((self.rows,), stride=(1,)), Float32)
        w_value = cute.make_rmem_tensor(cute.make_layout((self.experts,), stride=(1,)), Float32)
        acc = cute.make_rmem_tensor(cute.make_layout((self.rows * self.experts,), stride=(1,)), Float32)
        for r in cutlass.range_constexpr(self.rows):
            col_at[r] = cutlass.min(row_base + Int64(r), last) * Int64(8192) + lane
        for e in cutlass.range_constexpr(self.experts):
            weight_at[e] = (expert_base + Int64(e)) * Int64(8192) + lane
        for i in cutlass.range_constexpr(self.rows * self.experts):
            acc[i] = Float32(0.0)
        for chunk in cutlass.range(256, unroll=1):
            offset = Int64(chunk) * Int64(32)
            for r in cutlass.range_constexpr(self.rows):
                x_value[r] = Float32(x[col_at[r] + offset])
            for e in cutlass.range_constexpr(self.experts):
                w_value[e] = Float32(weight[weight_at[e] + offset])
            for r in cutlass.range_constexpr(self.rows):
                for e in cutlass.range_constexpr(self.experts):
                    at = r * self.experts + e
                    acc[at] = _add(acc[at], _mul(x_value[r], w_value[e]))
        for i in cutlass.range_constexpr(self.rows * self.experts):
            acc[i] = _warp_sum(acc[i])
        if lane == Int64(0):
            for r in cutlass.range_constexpr(self.rows):
                if row_base + Int64(r) < live:
                    for e in cutlass.range_constexpr(self.experts):
                        at = (row_base + Int64(r)) * Int64(256) + expert_base + Int64(e)
                        out[at] = acc[r * self.experts + e]


class RouterProjection:
    """Narrow batches keep the scalar GEMV; wide ones take the register tile.

    Both kernels store the same bits, so the row-count route can never move a
    result: an arbitrary row's logits are identical whether it is launched
    alone (scalar) or inside a wide prefill unit (tiled).
    """

    def __init__(self, scalar_max_rows: int = ROUTER_SCALAR_MAX_ROWS,
                 rows_per_warp: int = ROUTER_TILE_ROWS,
                 experts_per_warp: int = ROUTER_TILE_EXPERTS,
                 block_warps: int = ROUTER_BLOCK_WARPS,
                 row_groups: int = ROUTER_BLOCK_ROW_GROUPS):
        self.scalar_max_rows = int(scalar_max_rows)
        self.scalar = _RouterScalar()
        self.tiled = _RouterTiled(rows_per_warp, experts_per_warp, block_warps, row_groups)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(self.scalar_max_rows):
            self.scalar(x, weight, out, rows, stream)
        else:
            self.tiled(x, weight, out, rows, stream)


def compile_router_aot(*, scalar_max_rows: int = ROUTER_SCALAR_MAX_ROWS,
                       rows_per_warp: int = ROUTER_TILE_ROWS,
                       experts_per_warp: int = ROUTER_TILE_EXPERTS,
                       block_warps: int = ROUTER_BLOCK_WARPS,
                       row_groups: int = ROUTER_BLOCK_ROW_GROUPS):
    """Compile the router program; the tile geometry only changes load reuse.

    Every geometry stores the same bits, so the geometry is a launch-time
    choice, never a numerics choice.
    """
    return compile_program(
        RouterProjection(scalar_max_rows, rows_per_warp, experts_per_warp, block_warps, row_groups),
        name="muse_router", operands=(
            Operand("x", torch.bfloat16, "[rows,8192]"),
            Operand("weight", torch.float32, "[256,8192]"),
            Operand("out", torch.float32, "[rows,256]", "out")),
        scalars=(Scalar("rows"),),
        key=(8192, 256, int(scalar_max_rows), int(rows_per_warp), int(experts_per_warp),
             int(block_warps), int(row_groups)),
        geometry=dict(width=8192, outputs=256, weight_dtype="f32",
                      scalar_max_rows=int(scalar_max_rows), tile_rows=int(rows_per_warp),
                      tile_experts=int(experts_per_warp), block_warps=int(block_warps),
                      block_row_groups=int(row_groups)))


class Softcap:
    def __init__(self, vocab):
        self.vocab = int(vocab)

    @cute.jit
    def __call__(self, logits: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(logits, out, rows).launch(
            grid=((Int64(rows) * Int64(self.vocab) + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, out: cute.Pointer, rows: Int32):
        logits, out = _view(logits), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < Int64(rows) * Int64(self.vocab):
            value = _bf16(_mul(_bf16(Float32(logits[i])), Float32(0.17677669529663687)))
            value = _bf16(div_rn_f32(value, Float32(20.0)))
            value = _bf16(cute.math.tanh(value, fastmath=False))
            out[i] = _bf16(_mul(value, Float32(20.0)))


def compile_softcap_aot(*, vocab=202048):
    if vocab <= 0:
        raise ValueError("Muse vocab must be positive")
    return compile_program(Softcap(vocab), name="muse_softcap", operands=(
        Operand("logits", torch.float32, f"[rows,{vocab}]"),
        Operand("out", torch.float32, f"[rows,{vocab}]", "out")),
        scalars=(Scalar("rows"),), key=(vocab,), geometry=dict(vocab=vocab, multiplier=0.17677669529663687, softcap=20.0))


class ResidualGate:
    @cute.jit
    def __call__(self, residual: cute.Pointer, x: cute.Pointer, alpha: cute.Pointer,
                 beta: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(residual, x, alpha, beta, out, rows).launch(
            grid=((rows * Int32(8192) + Int32(255)) // Int32(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, residual: cute.Pointer, x: cute.Pointer, alpha: cute.Pointer,
               beta: cute.Pointer, out: cute.Pointer, rows: Int32):
        residual, x, alpha, beta, out = _view(residual), _view(x), _view(alpha), _view(beta), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < Int64(rows) * Int64(8192):
            col = i % Int64(8192)
            out[i] = BFloat16(_add(_mul(Float32(alpha[col]), Float32(residual[i])),
                                   _mul(Float32(beta[col]), Float32(x[i]))))


def compile_residual_gate_aot():
    return compile_program(ResidualGate(), name="muse_residual_gate", operands=(
        Operand("residual", torch.bfloat16, "[rows,8192]"),
        Operand("x", torch.bfloat16, "[rows,8192]"),
        Operand("alpha", torch.float32, "[8192]"),
        Operand("beta", torch.float32, "[8192]"),
        Operand("out", torch.bfloat16, "[rows,8192]", "out")),
        scalars=(Scalar("rows"),), key=(8192, 0.3), geometry=dict(width=8192, temperature=0.3))


class RouteSelect:
    """Eight deterministic warp argmax passes; sum in descending selected order."""
    @cute.jit
    def __call__(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
                 weights: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(logits, bias, ids, weights).launch(grid=(rows, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer):
        logits, bias, ids, weights = _view(logits), _view(bias), _view(ids), _view(weights)
        row = Int64(cute.arch.block_idx()[0])
        lane = Int32(cute.arch.thread_idx()[0])
        scores = cute.make_rmem_tensor((8,), Float32)
        biased = cute.make_rmem_tensor((8,), Float32)
        selected = cute.make_rmem_tensor((8,), Int32)
        values = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            expert = lane + Int32(j * 32)
            scaled = _mul(Float32(logits[row * Int64(256) + Int64(expert)]), Float32(0.1767766952966369))
            # Widen only exp to avoid the device expf approximation moving top-k ties.
            exponent = Float32(cute.math.exp(Float64(-scaled), fastmath=False))
            scores[j] = div_rn_f32(Float32(1.0), _add(Float32(1.0), exponent))
            biased[j] = _add(scores[j], Float32(bias[expert]))
        total = Float32(0.0)
        for slot in cutlass.range_constexpr(8):
            best = -Float32.inf
            for j in cutlass.range_constexpr(8):
                best = cutlass.max(best, biased[j])
            best = _warp_max(best)
            expert = Int32(256)
            value = Float32(0.0)
            for j in cutlass.range_constexpr(8):
                candidate = lane + Int32(j * 32)
                if (biased[j] == best) & (candidate < expert):
                    expert = candidate
                    value = scores[j]
            # The lower id wins exact ties, including across lanes.
            for delta in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(expert, 1 << delta)
                expert = cutlass.min(expert, other)
            for j in cutlass.range_constexpr(8):
                if lane + Int32(j * 32) == expert:
                    value = scores[j]
                    biased[j] = -Float32.inf
            value = cute.arch.shuffle_sync(value, expert % Int32(32))
            selected[slot], values[slot] = expert, value
            total = _add(total, value)
        total = _add(total, Float32(1e-15))
        if lane == Int32(0):
            for slot in cutlass.range_constexpr(8):
                at = row * Int64(8) + Int64(slot)
                ids[at] = selected[slot]
                weights[at] = div_rn_f32(values[slot], total)


def compile_route_select_aot():
    return compile_program(RouteSelect(), name="muse_route_select", operands=(
        Operand("logits", torch.float32, "[rows,256]"),
        Operand("bias", torch.float32, "[256]"),
        Operand("ids", torch.int32, "[rows,8]", "out", align=4),
        Operand("weights", torch.float32, "[rows,8]", "out")),
        scalars=(Scalar("rows"),), key=(256, 8, 0.1767766952966369),
        geometry=dict(experts=256, topk=8, multiplier=0.1767766952966369, epsilon=1e-15))


class RowNorm:
    def __init__(self, width, mode, eps):
        self.width, self.mode, self.eps = int(width), mode, float(eps)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, weight, out).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def value(self, x: cute.Tensor, w: cute.Tensor, base: Int64, col: Int64):
        v = Float32(x[base + col])
        if cutlass.const_expr(self.mode == "input_scaled"):
            v = _bf16(_mul(v, _bf16(_add(Float32(1.0), Float32(w[col])))))
        return v

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        x, weight, out = _view(x), _view(weight), _view(out)
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        warp, lane = tid // Int32(32), tid % Int32(32)
        base = row * Int64(self.width)
        squares = Float32(0.0)
        for slab_group in cutlass.range_constexpr((self.width + 1023) // 1024):
            for j in cutlass.range_constexpr(4):
                col = Int64(slab_group * 1024) + Int64(warp) * Int64(128) + Int64(lane) * Int64(4) + Int64(j)
                if col < Int64(self.width):
                    value = self.value(x, weight, base, col)
                    squares = _add(squares, _mul(value, value))
        squares = _warp_sum(squares)
        totals = cutlass.utils.SmemAllocator().allocate_tensor(Float32, cute.make_layout((8,)))
        if lane == Int32(0):
            totals[warp] = squares
        cute.arch.sync_threads()
        total = Float32(0.0)
        for w in cutlass.range_constexpr(8):
            total = _add(total, Float32(totals[w]))
        variance = _add(div_rn_f32(total, Float32(self.width)), Float32(self.eps))
        inv = div_rn_f32(Float32(1.0), _sqrt_rn(variance))
        for chunk in cutlass.range_constexpr((self.width + 255) // 256):
            col = Int64(chunk * 256) + Int64(tid)
            if col < Int64(self.width):
                value = _mul(self.value(x, weight, base, col), inv)
                if cutlass.const_expr(self.mode == "centered"):
                    value = _mul(value, _bf16(_add(Float32(1.0), Float32(weight[col]))))
                elif cutlass.const_expr(self.mode == "scaled"):
                    value = _mul(value, Float32(weight[col]))
                out[base + col] = BFloat16(value)


class WeightlessNorm:
    def __init__(self, width, eps):
        self.norm = RowNorm(width, "weightless", eps)

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.norm(x, x, out, rows, stream)


NORMS = {
    "embed_norm": (8192, "weightless", 1e-5),
    "input_norm": (8192, "centered", 1e-5),
    "pre_ffn_norm": (8192, "centered", 1e-5),
    "pre_expert_norm": (4096, "centered", 1e-5),
    "post_attn_norm": (8192, "weightless", 1e-8),
    "post_ffn_norm": (8192, "input_scaled", 1e-8),
    "final_norm": (8192, "scaled", 1e-5),
}


def compile_norm_aot(*, op):
    if op not in NORMS:
        raise ValueError(f"unknown Muse norm: {op}")
    width, mode, eps = NORMS[op]
    operands = (Operand("x", torch.bfloat16, f"[rows,{width}]"),)
    if mode != "weightless":
        operands += (Operand("weight", torch.bfloat16, f"[{width}]"),)
    operands += (Operand("out", torch.bfloat16, f"[rows,{width}]", "out"),)
    launch = WeightlessNorm(width, eps) if mode == "weightless" else RowNorm(width, mode, eps)
    return compile_program(launch, name=f"muse_{op}", operands=operands, scalars=(Scalar("rows"),),
        key=(width, mode, eps), geometry=dict(width=width, mode=mode, eps=eps))


class HeadNorm:
    def __init__(self, heads, *, scale=1.0, rotary=False, gate=False):
        self.heads, self.scale = int(heads), float(scale)
        self.rotary, self.gate = bool(rotary), bool(gate)

    @cute.jit
    def __call__(self, x: cute.Pointer, positions: cute.Pointer, cos: cute.Pointer, sin: cute.Pointer,
                 gate: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, positions, cos, sin, gate, out).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, positions: cute.Pointer, cos: cute.Pointer, sin: cute.Pointer,
               gate: cute.Pointer, out: cute.Pointer):
        x, positions, cos, sin, gate, out = _view(x), _view(positions), _view(cos), _view(sin), _view(gate), _view(out)
        row, head = Int64(cute.arch.block_idx()[0]), Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        base = (row * Int64(self.heads) + head) * Int64(64)
        a, b = Float32(x[base + lane]), Float32(x[base + lane + Int64(32)])
        squares = Float32(0.0)
        if lane < Int64(16):
            for j in cutlass.range_constexpr(4):
                value = Float32(x[base + lane * Int64(4) + Int64(j)])
                squares = _add(squares, _mul(value, value))
        squares = _warp_sum(squares)
        inv = div_rn_f32(Float32(1.0), _sqrt_rn(_add(_mul(squares, Float32(1.0 / 64)), Float32(1e-5))))
        a, b = _bf16(_mul(a, inv)), _bf16(_mul(b, inv))
        if cutlass.const_expr(self.scale != 1.0):
            a, b = _bf16(_mul(a, Float32(self.scale))), _bf16(_mul(b, Float32(self.scale)))
        if cutlass.const_expr(self.rotary):
            at = Int64(positions[row]) * Int64(32) + lane
            ra = _bf16(_add(_bf16(_mul(a, Float32(cos[at]))), _bf16(_mul(-b, Float32(sin[at])))))
            rb = _bf16(_add(_bf16(_mul(b, Float32(cos[at]))),
                            _bf16(_mul(a, Float32(sin[at])))))
            a, b = ra, rb
        if cutlass.const_expr(self.gate):
            a = _bf16(_mul(a, _bf16(_sigmoid(Float32(gate[base + lane])))))
            b = _bf16(_mul(b, _bf16(_sigmoid(Float32(gate[base + lane + Int64(32)])))))
        out[base + lane], out[base + lane + Int64(32)] = BFloat16(a), BFloat16(b)


class KvRecordNorm:
    def __init__(self, heads, rotary):
        self.heads, self.rotary = heads, rotary

    @cute.kernel
    def kernel(self, k_raw: cute.Pointer, v: cute.Pointer, positions: cute.Pointer,
               cos: cute.Pointer, sin: cute.Pointer, slots: cute.Pointer, dst: cute.Pointer):
        k_raw, v, positions, cos, sin, slots, dst = (
            _view(p) for p in (k_raw, v, positions, cos, sin, slots, dst))
        row, head = Int64(cute.arch.block_idx()[0]), Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        slot = Int64(slots[row])
        if slot >= Int64(0):
            base = (row * Int64(self.heads) + head) * Int64(64)
            a, b = Float32(k_raw[base + lane]), Float32(k_raw[base + lane + Int64(32)])
            # Match HeadNorm's arithmetic and BF16 boundaries before scattering.
            squares = Float32(0.0)
            if lane < Int64(16):
                for j in cutlass.range_constexpr(4):
                    value = Float32(k_raw[base + lane * Int64(4) + Int64(j)])
                    squares = _add(squares, _mul(value, value))
            squares = _warp_sum(squares)
            inv = div_rn_f32(Float32(1.0), _sqrt_rn(_add(_mul(squares, Float32(1.0 / 64)), Float32(1e-5))))
            a, b = _bf16(_mul(a, inv)), _bf16(_mul(b, inv))
            if cutlass.const_expr(self.rotary):
                at = Int64(positions[row]) * Int64(32) + lane
                ra = _bf16(_add(_bf16(_mul(a, Float32(cos[at]))), _bf16(_mul(-b, Float32(sin[at])))))
                rb = _bf16(_add(_bf16(_mul(b, Float32(cos[at]))),
                                _bf16(_mul(a, Float32(sin[at])))))
                a, b = ra, rb
            at = slot * Int64(self.heads * 128) + head * Int64(64) + lane
            dst[at], dst[at + Int64(32)] = BFloat16(a), BFloat16(b)
            dst[at + Int64(self.heads * 64)] = v[base + lane]
            dst[at + Int64(self.heads * 64 + 32)] = v[base + lane + Int64(32)]


class QkNormRope:
    def __init__(self, heads, kv_heads, rotary):
        self.q = HeadNorm(heads, scale=5.4730064863875, rotary=rotary)
        self.kv = KvRecordNorm(kv_heads, rotary)

    @cute.jit
    def __call__(self, q_raw: cute.Pointer, k_raw: cute.Pointer, v: cute.Pointer,
                 positions: cute.Pointer, cos: cute.Pointer, sin: cute.Pointer,
                 kv_slots: cute.Pointer, q: cute.Pointer, kv_dst: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.q(q_raw, positions, cos, sin, q_raw, q, rows, stream)
            self.kv.kernel(k_raw, v, positions, cos, sin, kv_slots, kv_dst).launch(
                grid=(rows, self.kv.heads, 1), block=(32, 1, 1), stream=stream)


def _heads(heads):
    if heads not in (64, 128):
        raise ValueError("Muse query heads must be 128 (whole) or 64 (TP2)")
    return heads, heads // 8


def compile_qk_norm_rope_aot(*, heads=128, kind="swa"):
    heads, kv_heads = _heads(heads)
    if kind not in ("swa", "full"):
        raise ValueError("Muse attention kind must be swa or full")
    return compile_program(QkNormRope(heads, kv_heads, kind == "swa"),
        name="muse_qk_norm_rope_swa" if kind == "swa" else "muse_qk_norm_nope_full", operands=(
            Operand("q_raw", torch.bfloat16, f"[rows,{heads},64]"),
            Operand("k_raw", torch.bfloat16, f"[rows,{kv_heads},64]"),
            Operand("v", torch.bfloat16, f"[rows,{kv_heads},64]"),
            Operand("positions", torch.int64, "[rows]", align=8),
            Operand("cos", torch.bfloat16, "[P,32]"), Operand("sin", torch.bfloat16, "[P,32]"),
            Operand("kv_slots", torch.int64, "[rows]", align=8),
            Operand("q", torch.bfloat16, f"[rows,{heads},64]", "out"),
            Operand("kv_dst", torch.bfloat16, f"[records,{kv_heads * 128}]", "inout")),
        scalars=(Scalar("rows"),), key=(heads, kind), geometry=dict(heads=heads, kv_heads=kv_heads, kind=kind))


class AttentionOutput:
    def __init__(self, heads):
        self.norm = HeadNorm(heads, gate=True)

    @cute.jit
    def __call__(self, attn: cute.Pointer, gate: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.norm(attn, attn, attn, attn, gate, out, rows, stream)


def compile_attention_output_aot(*, heads=128):
    heads, _ = _heads(heads)
    return compile_program(AttentionOutput(heads), name="muse_attention_output", operands=(
        Operand("attn", torch.bfloat16, f"[rows,{heads},64]"),
        Operand("gate", torch.bfloat16, f"[rows,{heads * 64}]"),
        Operand("out", torch.bfloat16, f"[rows,{heads * 64}]", "out")),
        scalars=(Scalar("rows"),), key=(heads,), geometry=dict(heads=heads, head_dim=64, eps=1e-5))
