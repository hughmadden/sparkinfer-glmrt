"""CuTe DSL kernels composed into the Qwen 3.8 Flash Next Gated DeltaNet program (``qwen4_gdn``).

Rounding follows transformers ``modeling_qwen4_exp`` (``Qwen4ExpTextGatedDeltaNet``):

* short convolution: ``bf16(silu(bf16(sum_j w[c, j] * x[t - 3 + j, c])))`` over
  the q/k/v in-projection (BF16 ``F.conv1d`` output, then SiLU on BF16);
* recurrence per value head ``h`` (``S`` FP32 ``[v, k]``, key head ``h // r``):
  ``q, k`` L2-normalized in FP32 (eps 1e-6), ``q *= 128^-0.5``;
  ``g = -exp(A_log[h]) * softplus(a + dt_bias[h])`` (FP32, one scalar per
  head); ``beta = bf16(sigmoid(b))``; ``S *= exp(g)``,
  ``d = (v - S k) * beta``, ``S += d k^T``, ``o = bf16(S q)``;
* output: per head ``bf16(bf16(w * bf16(rmsnorm(o))) * sigmoid(z))`` (eps 1e-6,
  the gate is a sigmoid, the weight is used as ``w``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import div_rn_f32

from ._glm_kernels import _bf16, _rsqrt, _warp_sum
from ._glmf_kernels import CONV_TAPS, _conv_state, _sigmoid

HEAD = 128


class Qwen4GdnConv:
    """Causal short conv + SiLU over the q/k/v in-projection columns (BF16 rounding before SiLU).

    ``proj`` BF16 ``[rows, P]`` (q/k/v in columns ``[0, C)``), ``weight`` FP32
    ``[C, 4]``, ``conv_state`` BF16 ``[slots, 3, C]`` (the sequence's last three
    inputs before this step, oldest first; a negative slot means zeros),
    ``out`` BF16 ``[rows, C]``. Grid ``(rows, C / 256)``.
    """

    threads = 256

    def __init__(self, *, channels: int, proj_width: int):
        self.channels, self.proj_width = int(channels), int(proj_width)
        if self.channels % self.threads:
            raise ValueError("conv channels must divide by the CTA width")

    @cute.jit
    def __call__(self, proj: cute.Pointer, weight: cute.Pointer, conv_state: cute.Pointer, slots: cute.Pointer,
                 seq_first: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        c = self.channels
        self.kernel(
            cute.make_tensor(proj, cute.make_layout((m, self.proj_width), stride=(self.proj_width, 1))),
            cute.make_tensor(weight, cute.make_layout((c, CONV_TAPS), stride=(CONV_TAPS, 1))),
            conv_state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(seq_first, cute.make_layout((m,))),
            cute.make_tensor(out, cute.make_layout((m, c), stride=(c, 1))),
        ).launch(grid=(rows, c // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, proj: cute.Tensor, weight: cute.Tensor, conv_state: cute.Pointer, slots: cute.Tensor,
               seq_first: cute.Tensor, out: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        slot = Int64(slots[row])
        rel = row - first
        state = _conv_state(conv_state, slot, self.channels)
        acc = Float32(0.0)
        for j in cutlass.range_constexpr(CONV_TAPS):
            s = rel - Int64(CONV_TAPS - 1 - j)
            x = Float32(0.0)
            if s >= Int64(0):
                x = Float32(proj[first + s, ch])
            elif slot >= Int64(0):
                x = Float32(state[Int64(CONV_TAPS - 1) + s, ch])
            acc = acc + Float32(weight[ch, j]) * x
        acc = _bf16(acc)
        out[row, ch] = div_rn_f32(acc, Float32(1.0) + cute.math.exp(-acc, fastmath=False)).to(BFloat16)


class Qwen4GdnGatedNorm:
    """``y = bf16(bf16(w * bf16(rmsnorm(o))) * sigmoid(z))`` per 128-wide value head.

    ``o`` BF16 ``[rows, D]``, ``gate`` BF16 rows of stride ``gate_stride``,
    ``weight`` BF16 ``[128]``, ``y`` BF16 ``[rows, D]``. A warp per (row,
    head), grid ``(rows, heads / 8)``.
    """

    threads = 256

    def __init__(self, *, heads: int, eps: float, gate_stride: int):
        self.heads, self.eps, self.gate_stride = int(heads), float(eps), int(gate_stride)
        self.width = self.heads * HEAD
        if self.heads % 8:
            raise ValueError("GDN value heads must divide into 8 warps")

    @cute.jit
    def __call__(self, o: cute.Pointer, gate: cute.Pointer, weight: cute.Pointer, y: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        d = self.width
        self.kernel(
            cute.make_tensor(o, cute.make_layout((m, d), stride=(d, 1))),
            cute.make_tensor(gate, cute.make_layout((m, d), stride=(self.gate_stride, 1))),
            cute.make_tensor(weight, cute.make_layout((HEAD,))),
            cute.make_tensor(y, cute.make_layout((m, d), stride=(d, 1))),
        ).launch(grid=(rows, self.heads // 8, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, o: cute.Tensor, gate: cute.Tensor, weight: cute.Tensor, y: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        head = Int64(cute.arch.block_idx()[1]) * Int64(8) + Int64(tidx // Int32(32))
        lane = Int64(tidx % Int32(32))
        x = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        square = Float32(0.0)
        for i in cutlass.range_constexpr(4):
            x[i] = Float32(o[row, head * Int64(HEAD) + lane * Int64(4) + Int64(i)])
            square = square + x[i] * x[i]
        inv = _rsqrt(_warp_sum(square) / Float32(HEAD) + Float32(self.eps))
        for i in cutlass.range_constexpr(4):
            d = lane * Int64(4) + Int64(i)
            g = Float32(gate[row, head * Int64(HEAD) + d])
            v = _bf16(Float32(weight[d]) * _bf16(x[i] * inv))
            y[row, head * Int64(HEAD) + d] = (v * _sigmoid(g)).to(BFloat16)


class Qwen4GdnRecurrent:
    """Token-sequential gated delta rule over one step's rows.

    ``qkv`` BF16 ``[rows, 2K + V]`` (conv outputs q | k | v; ``K`` =
    key heads x 128, ``V`` = value heads x 128), ``a_raw`` and ``b_raw`` BF16
    rows of stride ``ab_stride`` (one value per value head), ``a_log`` and
    ``dt_bias`` FP32 ``[heads]``, ``state`` FP32 ``[slots, heads, 128 (v), 128
    (k)]`` (the b12x ``gdn_decode`` / delta-prefill layout), ``slots`` i32
    ``[rows]`` (negative: zero state, not stored), ``out`` BF16 ``[rows, V]``.

    Grid ``(heads, 128 / 32)``: a CTA owns 32 value rows of one head, each of
    its 8 warps 4 rows, each lane 4 key columns of those rows in registers.
    Rows run in order; a row whose slot differs from the previous row's stores
    the previous sequence's state and loads its own.
    """

    threads = 256
    v_block = 32

    def __init__(self, *, heads: int, key_heads: int, ab_stride: int, eps: float = 1.0e-6):
        self.heads, self.key_heads = int(heads), int(key_heads)
        self.ratio = self.heads // self.key_heads
        self.key_width = self.key_heads * HEAD
        self.width = self.heads * HEAD
        self.qkv_width = 2 * self.key_width + self.width
        self.ab_stride = int(ab_stride)
        self.eps = float(eps)
        self.scale = HEAD ** -0.5

    @cute.jit
    def __call__(self, qkv: cute.Pointer, a_raw: cute.Pointer, b_raw: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(qkv, cute.make_layout((m, self.qkv_width), stride=(self.qkv_width, 1))),
            cute.make_tensor(a_raw, cute.make_layout((m, self.heads), stride=(self.ab_stride, 1))),
            cute.make_tensor(b_raw, cute.make_layout((m, self.heads), stride=(self.ab_stride, 1))),
            cute.make_tensor(a_log, cute.make_layout((self.heads,))),
            cute.make_tensor(dt_bias, cute.make_layout((self.heads,))),
            state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(out, cute.make_layout((m, self.width), stride=(self.width, 1))),
            rows,
        ).launch(grid=(self.heads, HEAD // self.v_block, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _state_tensor(self, state: cute.Pointer, slot: Int64, head: Int64) -> cute.Tensor:
        base = Int64(state.toint()) + ((slot * Int64(self.heads) + head) * Int64(HEAD * HEAD)) * Int64(4)
        return cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=16),
                                cute.make_layout((HEAD, HEAD), stride=(HEAD, 1)))

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, a_raw: cute.Tensor, b_raw: cute.Tensor, a_log: cute.Tensor,
               dt_bias: cute.Tensor, state: cute.Pointer, slots: cute.Tensor, out: cute.Tensor, rows: Int32):
        head = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        v0 = Int64(cute.arch.block_idx()[1]) * Int64(self.v_block) + Int64(tidx // Int32(32)) * Int64(4)
        k0 = Int64(lane) * Int64(4)
        kcol = (head // Int64(self.ratio)) * Int64(HEAD)
        vcol = Int64(2 * self.key_width) + head * Int64(HEAD)
        s = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), Float32)
        q = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        k = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for j in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(4):
                s[j, i] = Float32(0.0)
        rate = cute.math.exp(Float32(a_log[head]), fastmath=False)
        bias = Float32(dt_bias[head])
        current = Int64(-1)
        for t in cutlass.range(rows, unroll=1):
            row = Int64(t)
            slot = Int64(slots[row])
            if slot != current:
                if current >= Int64(0):
                    old = self._state_tensor(state, current, head)
                    for j in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(4):
                            old[v0 + Int64(j), k0 + Int64(i)] = s[j, i]
                if slot >= Int64(0):
                    new = self._state_tensor(state, slot, head)
                    for j in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(4):
                            s[j, i] = new[v0 + Int64(j), k0 + Int64(i)]
                else:
                    for j in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(4):
                            s[j, i] = Float32(0.0)
                current = slot
            q_sq = Float32(0.0)
            k_sq = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                q[i] = Float32(qkv[row, kcol + k0 + Int64(i)])
                k[i] = Float32(qkv[row, Int64(self.key_width) + kcol + k0 + Int64(i)])
                q_sq = q_sq + q[i] * q[i]
                k_sq = k_sq + k[i] * k[i]
            q_inv = _rsqrt(_warp_sum(q_sq) + Float32(self.eps))
            k_inv = _rsqrt(_warp_sum(k_sq) + Float32(self.eps))
            for i in cutlass.range_constexpr(4):
                q[i] = q[i] * q_inv * Float32(self.scale)
                k[i] = k[i] * k_inv
            x = Float32(a_raw[row, head]) + bias
            softplus = x
            if x <= Float32(20.0):
                softplus = cute.math.log1p(cute.math.exp(x, fastmath=False), fastmath=False)
            decay = cute.math.exp(-rate * softplus, fastmath=False)
            beta = _bf16(_sigmoid(Float32(b_raw[row, head])))
            for j in cutlass.range_constexpr(4):
                v = Float32(qkv[row, vcol + v0 + Int64(j)])
                memory = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    s[j, i] = s[j, i] * decay
                    memory = memory + s[j, i] * k[i]
                delta = (v - _warp_sum(memory)) * beta
                o = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    s[j, i] = s[j, i] + k[i] * delta
                    o = o + s[j, i] * q[i]
                o = _warp_sum(o)
                if lane == Int32(j):
                    out[row, head * Int64(HEAD) + v0 + Int64(j)] = o.to(BFloat16)
        if current >= Int64(0):
            old = self._state_tensor(state, current, head)
            for j in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    old[v0 + Int64(j), k0 + Int64(i)] = s[j, i]
