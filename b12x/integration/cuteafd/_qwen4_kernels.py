"""CuTe DSL kernels composed into the Qwen 3.8 Flash Next (``qwen4_*``) AOT programs.

Every kernel takes raw pointers and a runtime ``rows``. Arithmetic follows the
transformers ``modeling_qwen4_exp`` rounding points: every tensor the
reference materializes in BF16 is rounded here too. Qwen's RMSNorm is
``bf16(x * rsqrt(mean(x^2) + eps) * (1 + w))`` computed in FP32 and rounded
once; the hyper-connection norms normalize each ``hidden``-wide stream group
separately. One CTA of ``hidden / 8`` threads (320 at 2560) owns a row: thread
``t`` holds the 8 BF16 values ``[8t, 8t + 8)`` of every stream group.

Hyper-connections (``Qwen4ExpTextGatedResidual``), streams ``S`` BF16
``[rows, 4, H]``:

* ``n = hc_norm(S)``; ``d = bf16(n @ w_down^T)`` (plus the 4 injection logits
  ``bf16(n @ w_inject^T)``); ``a = bf16(silu(bf16(d / 4)))``;
  ``u = bf16(a @ w_up^T)``; ``y = bf16(mean_s bf16(bf16(sigmoid(u_s)) * n_s))``;
  ``inject_s = bf16(2 * bf16(sigmoid(bf16(e_s / 4))))``.
* after a sublayer with output ``o``: ``S_s = bf16(S_s + bf16(o * inject_s))``.

PLE (``Qwen4ExpTextPLELayer``, layer 1): ``e`` = 16 gathered n-gram rows of
160 (BF16, or E4M3 times the BF16 table scale rounded to BF16); ``k|v =
bf16(e @ [key_proj; value_proj]^T)``; per stream ``g_s = bf16(sum(bf16(
norm_key(k)_s * norm_query(S)_s)))``, ``g_s = bf16(g_s / sqrt(H))``, ``g_s =
sign(g_s) * bf16(sqrt(max(|g_s|, 1e-6)))``; ``gv_s = bf16(bf16(sigmoid(g_s)) *
v)``; ``c = bf16(silu(bf16(conv(norm_conv(gv)))))`` (depthwise, 4 taps at
dilation 3, the sequence's last 9 ``norm_conv(gv)`` rows as state);
``S = bf16(S + bf16(gv + c))``.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cvt_e4m3x4_to_f32x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    ld_global_v4_u32,
    pack_f32x2_to_bfloat2,
    st_global_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._glm_kernels import _block_sum, _reduction_storage, _rsqrt

VEC = 8


@cute.jit
def _bf16(value: Float32) -> Float32:
    """Round to BF16 (RNE) through the packing instruction: a plain truncf/extf pair may be folded away."""
    return _bf16_lo(pack_f32x2_to_bfloat2(value, value))


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return div_rn_f32(Float32(1.0), Float32(1.0) + cute.math.exp(-x, fastmath=False))


@cute.jit
def _silu(x: Float32) -> Float32:
    return div_rn_f32(x, Float32(1.0) + cute.math.exp(-x, fastmath=False))


@cute.jit
def _load8(address: Int64, values: cute.Tensor, at: cutlass.Constexpr):
    """8 BF16 at ``address`` (16-byte aligned) into ``values[at:at+8]`` as FP32."""
    words = ld_global_v4_u32(address)
    for i in cutlass.range_constexpr(4):
        values[at + 2 * i] = _bf16_lo(words[i])
        values[at + 2 * i + 1] = _bf16_hi(words[i])


@cute.jit
def _store8(address: Int64, values: cute.Tensor, at: cutlass.Constexpr):
    st_global_v4_u32(address,
                     pack_f32x2_to_bfloat2(values[at], values[at + 1]),
                     pack_f32x2_to_bfloat2(values[at + 2], values[at + 3]),
                     pack_f32x2_to_bfloat2(values[at + 4], values[at + 5]),
                     pack_f32x2_to_bfloat2(values[at + 6], values[at + 7]))


@cute.jit
def _row_scalars(values: cute.Pointer, row: Int64, n: cutlass.Constexpr) -> cute.Tensor:
    """The ``n`` BF16 scalars of ``row`` in a dense ``[rows, n]`` BF16 array."""
    at = Int64(values.toint()) + row * Int64(n * 2)
    return cute.make_tensor(cute.make_ptr(BFloat16, at, cute.AddressSpace.gmem, assumed_align=2),
                            cute.make_layout((n,)))


def _row_threads(hidden: int) -> int:
    if hidden % VEC or (hidden // VEC) % 32:
        raise ValueError("hidden / 8 must be a multiple of the warp size")
    return hidden // VEC


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------


class Qwen4HcNorm:
    """Optional post (``S_s = bf16(S_s + bf16(o * inject_s))``, written to
    ``residual_out``) then ``normed = hc_norm(S)`` BF16 ``[rows, 4H]``.

    Pre: ``(residual, weight, normed)``; post: ``(delta, residual, inject,
    weight, residual_out, normed)``. ``inject`` is BF16 ``[rows, 4]`` and may be
    overwritten later in the same program (it is read here first).
    """

    def __init__(self, hidden: int, streams: int, eps: float, post: bool):
        self.hidden, self.streams, self.eps, self.post = int(hidden), int(streams), float(eps), bool(post)
        self.threads = _row_threads(self.hidden)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, delta: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, weight: cute.Pointer,
                 residual_out: cute.Pointer, normed: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(delta, residual, inject, weight, residual_out, normed).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, delta: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, weight: cute.Pointer,
               residual_out: cute.Pointer, normed: cute.Pointer):
        h, n = self.hidden, self.streams
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(n, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((n, self.warps), stride=(self.warps, 1)))
        s_row = Int64(residual.toint()) + row * Int64(n * h * 2) + tidx * Int64(16)
        values = cute.make_rmem_tensor(cute.make_layout((n * VEC,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(n):
            _load8(s_row + Int64(s * h * 2), values, s * VEC)
        if cutlass.const_expr(self.post):
            d = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
            _load8(Int64(delta.toint()) + row * Int64(h * 2) + tidx * Int64(16), d, 0)
            inj = _row_scalars(inject, row, n)
            for s in cutlass.range_constexpr(n):
                w = Float32(inj[s])
                for k in cutlass.range_constexpr(VEC):
                    values[s * VEC + k] = _bf16(values[s * VEC + k] + _bf16(d[k] * w))
            o_row = Int64(residual_out.toint()) + row * Int64(n * h * 2) + tidx * Int64(16)
            for s in cutlass.range_constexpr(n):
                _store8(o_row + Int64(s * h * 2), values, s * VEC)
        wv = cute.make_rmem_tensor(cute.make_layout((n * VEC,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(n):
            _load8(Int64(weight.toint()) + Int64(s * h * 2) + tidx * Int64(16), wv, s * VEC)
        n_row = Int64(normed.toint()) + row * Int64(n * h * 2) + tidx * Int64(16)
        for s in cutlass.range_constexpr(n):
            square = Float32(0.0)
            for k in cutlass.range_constexpr(VEC):
                square = square + values[s * VEC + k] * values[s * VEC + k]
            total = _block_sum(square, sums, Int32(s), self.warps)
            inv = _rsqrt(total / Float32(h) + Float32(self.eps))
            for k in cutlass.range_constexpr(VEC):
                values[s * VEC + k] = values[s * VEC + k] * inv * (Float32(1.0) + wv[s * VEC + k])
            _store8(n_row + Int64(s * h * 2), values, s * VEC)


class Qwen4HcGate:
    """``a = bf16(silu(bf16(d / hc)))`` over the first ``lowrank`` columns of
    ``d`` BF16 ``[rows, d_width]``; with ``inject``, ``inject_s =
    bf16(2 * bf16(sigmoid(bf16(d[lowrank + s] / hc))))``."""

    threads = 128

    def __init__(self, lowrank: int, streams: int, inject: bool):
        self.lowrank, self.streams, self.inject = int(lowrank), int(streams), bool(inject)
        self.width = self.lowrank + (self.streams if self.inject else 0)

    @cute.jit
    def __call__(self, d: cute.Pointer, a: cute.Pointer, inject: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(cute.make_tensor(d, cute.make_layout((m, self.width), stride=(self.width, 1))),
                    cute.make_tensor(a, cute.make_layout((m, self.lowrank), stride=(self.lowrank, 1))),
                    cute.make_tensor(inject, cute.make_layout((m, self.streams), stride=(self.streams, 1)))).launch(
            grid=(rows, (self.width + self.threads - 1) // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, d: cute.Tensor, a: cute.Tensor, inject: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        scale = Float32(1.0 / self.streams)
        if col < Int64(self.lowrank):
            a[row, col] = _silu(_bf16(Float32(d[row, col]) * scale)).to(BFloat16)
        elif col < Int64(self.width):
            e = _bf16(_sigmoid(_bf16(Float32(d[row, col]) * scale)))
            inject[row, col - Int64(self.lowrank)] = (Float32(2.0) * e).to(BFloat16)


class Qwen4HcMix:
    """``y = bf16(mean_s bf16(bf16(sigmoid(u_s)) * n_s))`` from ``u`` and
    ``normed`` BF16 ``[rows, 4H]``; ``y`` BF16 ``[rows, H]``."""

    def __init__(self, hidden: int, streams: int):
        self.hidden, self.streams = int(hidden), int(streams)
        self.threads = _row_threads(self.hidden)

    @cute.jit
    def __call__(self, u: cute.Pointer, normed: cute.Pointer, y: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(u, normed, y).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, u: cute.Pointer, normed: cute.Pointer, y: cute.Pointer):
        h, n = self.hidden, self.streams
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        base = row * Int64(n * h * 2) + tidx * Int64(16)
        acc = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        uv = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        nv = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        for k in cutlass.range_constexpr(VEC):
            acc[k] = Float32(0.0)
        for s in cutlass.range_constexpr(n):
            _load8(Int64(u.toint()) + base + Int64(s * h * 2), uv, 0)
            _load8(Int64(normed.toint()) + base + Int64(s * h * 2), nv, 0)
            for k in cutlass.range_constexpr(VEC):
                acc[k] = acc[k] + _bf16(_bf16(_sigmoid(uv[k])) * nv[k])
        for k in cutlass.range_constexpr(VEC):
            acc[k] = acc[k] * Float32(1.0 / n)
        _store8(Int64(y.toint()) + row * Int64(h * 2) + tidx * Int64(16), acc, 0)


class Qwen4HcPost:
    """``out_s = bf16(S_s + bf16(o * inject_s))`` (the last layer's MLP site)."""

    def __init__(self, hidden: int, streams: int):
        self.hidden, self.streams = int(hidden), int(streams)
        self.threads = _row_threads(self.hidden)

    @cute.jit
    def __call__(self, x: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, residual, inject, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, delta: cute.Pointer, residual: cute.Pointer, inject: cute.Pointer, out: cute.Pointer):
        h, n = self.hidden, self.streams
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        d = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        v = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        _load8(Int64(delta.toint()) + row * Int64(h * 2) + tidx * Int64(16), d, 0)
        inj = _row_scalars(inject, row, n)
        base = row * Int64(n * h * 2) + tidx * Int64(16)
        for s in cutlass.range_constexpr(n):
            w = Float32(inj[s])
            _load8(Int64(residual.toint()) + base + Int64(s * h * 2), v, 0)
            for k in cutlass.range_constexpr(VEC):
                v[k] = v[k] + _bf16(d[k] * w)
            _store8(Int64(out.toint()) + base + Int64(s * h * 2), v, 0)


class Qwen4Add:
    """``out = bf16(a + b)`` over BF16 ``[rows, H]``."""

    def __init__(self, hidden: int):
        self.hidden = int(hidden)
        self.threads = _row_threads(self.hidden)

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(a, b, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer):
        at = Int64(cute.arch.block_idx()[0]) * Int64(self.hidden * 2) + Int64(cute.arch.thread_idx()[0]) * Int64(16)
        x = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        y = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        _load8(Int64(a.toint()) + at, x, 0)
        _load8(Int64(b.toint()) + at, y, 0)
        for k in cutlass.range_constexpr(VEC):
            x[k] = x[k] + y[k]
        _store8(Int64(out.toint()) + at, x, 0)


class Qwen4SwiGLU:
    """``hidden = bf16(bf16(silu(gate)) * up)`` from rows of stride ``stride``
    holding gate ``[0, I)`` then up ``[I, 2I)``; SiLU unclamped."""

    threads = 128

    def __init__(self, inter: int, stride: int):
        self.inter, self.stride = int(inter), int(stride)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, hidden: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        i = self.inter
        self.kernel(cute.make_tensor(gate_up, cute.make_layout((m, self.stride), stride=(self.stride, 1))),
                    cute.make_tensor(hidden, cute.make_layout((m, i), stride=(i, 1)))).launch(
            grid=(rows, (i + self.threads - 1) // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Tensor, hidden: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if col < Int64(self.inter):
            silu = _bf16(_silu(Float32(gate_up[row, col])))
            hidden[row, col] = (silu * Float32(gate_up[row, Int64(self.inter) + col])).to(BFloat16)


class Qwen4SharedGate:
    """Shared expert output ``out = bf16(bf16(sigmoid(g)) * o)`` where ``g`` is
    column ``gate_col`` of the gate/up projection rows (the appended
    ``shared_expert_gate`` row), ``o`` BF16 ``[rows, H]``."""

    def __init__(self, hidden: int, gate_row: int, gate_col: int):
        self.hidden, self.gate_row, self.gate_col = int(hidden), int(gate_row), int(gate_col)
        self.threads = _row_threads(self.hidden)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, o: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(cute.make_tensor(gate_up, cute.make_layout((m, self.gate_row), stride=(self.gate_row, 1))),
                    o, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Tensor, o: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        at = row * Int64(self.hidden * 2) + Int64(cute.arch.thread_idx()[0]) * Int64(16)
        g = _bf16(_sigmoid(Float32(gate_up[row, Int64(self.gate_col)])))
        x = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        _load8(Int64(o.toint()) + at, x, 0)
        for k in cutlass.range_constexpr(VEC):
            x[k] = g * x[k]
        _store8(Int64(out.toint()) + at, x, 0)


# ---------------------------------------------------------------------------
# PLE
# ---------------------------------------------------------------------------


class Qwen4PleGather:
    """``emb[r, h*D:(h+1)*D] = table[ids[r, h]]`` for the ``heads`` n-gram
    rows of each token: BF16 rows, or E4M3 rows times the BF16 table
    ``scale`` (``[1]``) rounded to BF16. ``table`` may be host-mapped memory.
    Each thread moves one 16-byte source vector."""

    def __init__(self, heads: int, row_dim: int, fp8: bool):
        self.heads, self.row_dim, self.fp8 = int(heads), int(row_dim), bool(fp8)
        self.elem = 1 if self.fp8 else 2
        self.vecs = self.row_dim * self.elem // 16
        if self.row_dim * self.elem % 16:
            raise ValueError("PLE rows must be whole 16-byte vectors")
        self.threads = self.heads * self.vecs

    @cute.jit
    def __call__(self, ids: cute.Pointer, table: cute.Pointer, scale: cute.Pointer, emb: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(cute.make_tensor(ids, cute.make_layout((m, self.heads), stride=(self.heads, 1))), table,
                    cute.make_tensor(scale, cute.make_layout((1,))), emb).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Tensor, table: cute.Pointer, scale: cute.Tensor, emb: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        head = tidx // Int64(self.vecs)
        vec = tidx % Int64(self.vecs)
        src = Int64(table.toint()) + Int64(ids[row, head]) * Int64(self.row_dim * self.elem) + vec * Int64(16)
        words = ld_global_v4_u32(src)
        dst = Int64(emb.toint()) + (row * Int64(self.heads * self.row_dim) + head * Int64(self.row_dim)) * Int64(2)
        if cutlass.const_expr(self.fp8):
            s = Float32(scale[0])
            vals = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
            for i in cutlass.range_constexpr(4):
                f0, f1, f2, f3 = cvt_e4m3x4_to_f32x4(words[i])
                vals[4 * i] = f0 * s
                vals[4 * i + 1] = f1 * s
                vals[4 * i + 2] = f2 * s
                vals[4 * i + 3] = f3 * s
            at = dst + vec * Int64(32)
            _store8(at, vals, 0)
            _store8(at + Int64(16), vals, 8)
        else:
            st_global_v4_u32(dst + vec * Int64(16), words[0], words[1], words[2], words[3])


class Qwen4PleGate:
    """Per row: ``gv_s = bf16(bf16(sigmoid(g_s)) * v)`` and ``gvn =
    norm_conv(gv)`` (see the module docstring) from ``kv`` BF16 ``[rows, 4H +
    H]`` (key then value) and the streams ``S``."""

    def __init__(self, hidden: int, streams: int, eps: float):
        self.hidden, self.streams, self.eps = int(hidden), int(streams), float(eps)
        self.threads = _row_threads(self.hidden)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, kv: cute.Pointer, streams: cute.Pointer, norm_key: cute.Pointer, norm_query: cute.Pointer,
                 norm_conv: cute.Pointer, gv: cute.Pointer, gvn: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(kv, streams, norm_key, norm_query, norm_conv, gv, gvn).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _norm(self, values: cute.Tensor, weight: cute.Pointer, sums: cute.Tensor, tidx: Int64):
        """In place: each stream group of ``values`` -> bf16(x * rsqrt(mean + eps) * (1 + w))."""
        h, n = self.hidden, self.streams
        wv = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(n):
            square = Float32(0.0)
            for k in cutlass.range_constexpr(VEC):
                square = square + values[s * VEC + k] * values[s * VEC + k]
            total = _block_sum(square, sums, Int32(s), self.warps)
            inv = _rsqrt(total / Float32(h) + Float32(self.eps))
            _load8(Int64(weight.toint()) + Int64(s * h * 2) + tidx * Int64(16), wv, 0)
            for k in cutlass.range_constexpr(VEC):
                values[s * VEC + k] = _bf16(values[s * VEC + k] * inv * (Float32(1.0) + wv[k]))
            cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, kv: cute.Pointer, streams: cute.Pointer, norm_key: cute.Pointer, norm_query: cute.Pointer,
               norm_conv: cute.Pointer, gv: cute.Pointer, gvn: cute.Pointer):
        h, n = self.hidden, self.streams
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(n, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((n, self.warps), stride=(self.warps, 1)))
        key = cute.make_rmem_tensor(cute.make_layout((n * VEC,), stride=(1,)), Float32)
        query = cute.make_rmem_tensor(cute.make_layout((n * VEC,), stride=(1,)), Float32)
        kv_row = Int64(kv.toint()) + row * Int64((n + 1) * h * 2) + tidx * Int64(16)
        s_row = Int64(streams.toint()) + row * Int64(n * h * 2) + tidx * Int64(16)
        for s in cutlass.range_constexpr(n):
            _load8(kv_row + Int64(s * h * 2), key, s * VEC)
            _load8(s_row + Int64(s * h * 2), query, s * VEC)
        value = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), Float32)
        _load8(kv_row + Int64(n * h * 2), value, 0)
        self._norm(key, norm_key, sums, tidx)
        self._norm(query, norm_query, sums, tidx)
        gates = cute.make_rmem_tensor(cute.make_layout((n,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(n):
            dot = Float32(0.0)
            for k in cutlass.range_constexpr(VEC):
                dot = dot + _bf16(key[s * VEC + k] * query[s * VEC + k])
            g = _bf16(_block_sum(dot, sums, Int32(s), self.warps))
            g = _bf16(div_rn_f32(g, Float32(math.sqrt(h))))
            mag = _bf16(cute.math.sqrt(fmax_f32(fabs_f32(g), Float32(_bf16_const(1.0e-6))), fastmath=False))
            signed = Float32(0.0)
            if g > Float32(0.0):
                signed = mag
            elif g < Float32(0.0):
                signed = -mag
            gates[s] = _bf16(_sigmoid(signed))
        cute.arch.sync_threads()
        for s in cutlass.range_constexpr(n):
            for k in cutlass.range_constexpr(VEC):
                key[s * VEC + k] = _bf16(gates[s] * value[k])
            _store8(Int64(gv.toint()) + row * Int64(n * h * 2) + tidx * Int64(16) + Int64(s * h * 2), key, s * VEC)
        self._norm(key, norm_conv, sums, tidx)
        for s in cutlass.range_constexpr(n):
            _store8(Int64(gvn.toint()) + row * Int64(n * h * 2) + tidx * Int64(16) + Int64(s * h * 2), key, s * VEC)


def _bf16_const(value: float) -> float:
    """``value`` rounded to BF16 (round to nearest even), as a Python float."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@cute.jit
def _ple_state(conv_state: cute.Pointer, slot: Int64, taps: cutlass.Constexpr, channels: cutlass.Constexpr):
    safe = slot
    if safe < Int64(0):
        safe = Int64(0)
    base = Int64(conv_state.toint()) + safe * Int64(taps * channels * 2)
    return cute.make_tensor(cute.make_ptr(BFloat16, base, cute.AddressSpace.gmem, assumed_align=2),
                            cute.make_layout((taps, channels), stride=(channels, 1)))


class Qwen4PleConv:
    """``S = bf16(S + bf16(gv + bf16(silu(bf16(conv(gvn))))))`` with the
    dilated depthwise conv ``conv(x)_t = sum_j w[c, j] x[t - dil * (K-1-j)]``;
    inputs before the step come from the sequence's ``(K-1)*dil`` state rows
    (oldest first; negative slots read zeros). Grid ``(rows, C / 256)``."""

    threads = 256

    def __init__(self, channels: int, taps: int, dilation: int):
        self.channels, self.taps, self.dilation = int(channels), int(taps), int(dilation)
        self.state_rows = (self.taps - 1) * self.dilation
        if self.channels % self.threads:
            raise ValueError("PLE conv channels must divide by the CTA width")

    @cute.jit
    def __call__(self, gv: cute.Pointer, gvn: cute.Pointer, weight: cute.Pointer, conv_state: cute.Pointer,
                 slots: cute.Pointer, seq_first: cute.Pointer, streams: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        c = self.channels
        self.kernel(cute.make_tensor(gv, cute.make_layout((m, c), stride=(c, 1))),
                    cute.make_tensor(gvn, cute.make_layout((m, c), stride=(c, 1))),
                    cute.make_tensor(weight, cute.make_layout((c, self.taps), stride=(self.taps, 1))),
                    conv_state,
                    cute.make_tensor(slots, cute.make_layout((m,))),
                    cute.make_tensor(seq_first, cute.make_layout((m,))),
                    cute.make_tensor(streams, cute.make_layout((m, c), stride=(c, 1)))).launch(
            grid=(rows, c // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gv: cute.Tensor, gvn: cute.Tensor, weight: cute.Tensor, conv_state: cute.Pointer,
               slots: cute.Tensor, seq_first: cute.Tensor, streams: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        slot = Int64(slots[row])
        rel = row - first
        state = _ple_state(conv_state, slot, self.state_rows, self.channels)
        acc = Float32(0.0)
        for j in cutlass.range_constexpr(self.taps):
            s = rel - Int64(self.dilation * (self.taps - 1 - j))
            x = Float32(0.0)
            if s >= Int64(0):
                x = Float32(gvn[first + s, ch])
            elif slot >= Int64(0):
                x = Float32(state[Int64(self.state_rows) + s, ch])
            acc = acc + Float32(weight[ch, j]) * x
        c = _bf16(_silu(_bf16(acc)))
        out = _bf16(Float32(gv[row, ch]) + c)
        streams[row, ch] = (Float32(streams[row, ch]) + out).to(BFloat16)


class Qwen4PleConvState:
    """After the conv: each sequence's last step row stores its last
    ``(K-1)*dil`` ``gvn`` rows (older ones from the previous state)."""

    threads = 256

    def __init__(self, channels: int, state_rows: int):
        self.channels, self.state_rows = int(channels), int(state_rows)

    @cute.jit
    def __call__(self, gvn: cute.Pointer, conv_state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        c = self.channels
        self.kernel(cute.make_tensor(gvn, cute.make_layout((m, c), stride=(c, 1))), conv_state,
                    cute.make_tensor(slots, cute.make_layout((m,))),
                    cute.make_tensor(seq_first, cute.make_layout((m,))), rows).launch(
            grid=(rows, c // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gvn: cute.Tensor, conv_state: cute.Pointer, slots: cute.Tensor, seq_first: cute.Tensor,
               rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        last = row + Int64(1) == Int64(rows)
        if not last:
            last = Int64(seq_first[row + Int64(1)]) != first
        slot = Int64(slots[row])
        if last and slot >= Int64(0):
            state = _ple_state(conv_state, slot, self.state_rows, self.channels)
            n = row - first + Int64(1)
            kept = cute.make_rmem_tensor(cute.make_layout((self.state_rows,), stride=(1,)), BFloat16)
            for i in cutlass.range_constexpr(self.state_rows):
                s = n - Int64(self.state_rows - i)
                if s >= Int64(0):
                    kept[i] = gvn[first + s, ch]
                else:
                    kept[i] = state[Int64(self.state_rows) + s, ch]
            for i in cutlass.range_constexpr(self.state_rows):
                state[i, ch] = kept[i]
