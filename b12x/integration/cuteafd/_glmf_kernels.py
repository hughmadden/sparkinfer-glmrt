"""CuTe DSL kernels composed into the GLM 5.3 Flash (``glmf_*``) AOT programs.

Every kernel takes raw pointers and a runtime ``rows``. Arithmetic follows the
transformers ``modeling_glm5_next`` rounding points: every tensor the
reference materializes in BF16 is rounded here too, FP32 where it computes in
FP32 (the short convolution, the KDA gates and recurrence, the gated output
norm).

Kimi Delta Attention, one sequence's rows at a time (rows of one sequence are
contiguous in a step; ``slots[row]`` names its state slot, ``seq_first[row]``
the step row where its sequence starts):

* short convolution: ``silu(sum_j w[c, j] * x[t - 3 + j, c])`` over the q/k/v
  in-projection, FP32 weights, inputs before the step from the sequence's
  conv state (its last three in-projection rows, BF16);
* recurrence per head (``S`` FP32 ``[v, k]``): ``q, k`` L2-normalized (eps
  1e-6), ``q *= 128^-0.5``; ``g = lb * sigmoid(exp(A_log) * (f + dt_bias))``
  per key; ``beta = bf16(sigmoid(b))``; ``S *= exp(g)`` per key column,
  ``d = (v - S k) * beta``, ``S += d k^T``, ``o = S q``;
* output: per head ``w * rmsnorm(o) * sigmoid(gate)`` (eps 1e-5), BF16.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import div_rn_f32, ld_global_v4_u32, pack_f32x2_to_bfloat2, st_global_v4_u32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._glm_kernels import _bf16, _block_sum, _reduction_storage, _rsqrt, _warp_sum

KDA_HEAD = 128
CONV_TAPS = 4


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return div_rn_f32(Float32(1.0), Float32(1.0) + cute.math.exp(-x, fastmath=False))


@cute.jit
def _conv_state(conv_state: cute.Pointer, slot: Int64, channels: cutlass.Constexpr) -> cute.Tensor:
    """The ``[3, C]`` BF16 conv state of ``slot`` (clamped to 0 when negative; callers skip it)."""
    safe = slot
    if safe < Int64(0):
        safe = Int64(0)
    base = Int64(conv_state.toint()) + safe * Int64((CONV_TAPS - 1) * channels * 2)
    return cute.make_tensor(cute.make_ptr(BFloat16, base, cute.AddressSpace.gmem, assumed_align=2),
                            cute.make_layout((CONV_TAPS - 1, channels), stride=(channels, 1)))


class _RowVectors:
    """Per-row 16-byte vector loads/stores of a BF16 row of ``width`` (CTA of 256)."""

    threads = 256

    def __init__(self, width: int):
        self.width = int(width)
        if self.width % (8 * self.threads):
            raise ValueError("row width must be a multiple of 8 * CTA width")
        self.vectors = self.width // (8 * self.threads)
        self.count = 8 * self.vectors

    @cute.jit
    def load(self, address: Int64, values: cute.Tensor):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            words = ld_global_v4_u32(address + (Int64(j * self.threads) + tidx) * Int64(16))
            for i in cutlass.range_constexpr(4):
                values[8 * j + 2 * i] = _bf16_lo(words[i])
                values[8 * j + 2 * i + 1] = _bf16_hi(words[i])

    @cute.jit
    def store(self, address: Int64, values: cute.Tensor):
        tidx = Int64(cute.arch.thread_idx()[0])
        for j in cutlass.range_constexpr(self.vectors):
            st_global_v4_u32(address + (Int64(j * self.threads) + tidx) * Int64(16),
                             pack_f32x2_to_bfloat2(values[8 * j], values[8 * j + 1]),
                             pack_f32x2_to_bfloat2(values[8 * j + 2], values[8 * j + 3]),
                             pack_f32x2_to_bfloat2(values[8 * j + 4], values[8 * j + 5]),
                             pack_f32x2_to_bfloat2(values[8 * j + 6], values[8 * j + 7]))


class GlmfMeanNorm:
    """Final mHC collapse + ``model.norm``: ``out = w * bf16(rmsnorm(bf16(mean(streams))))``.

    ``streams`` BF16 ``[rows, 4, H]``, ``out`` BF16 ``[rows, H]``. One CTA per row.
    """

    threads = 256

    def __init__(self, width: int, eps: float, streams: int = 4):
        self.row = _RowVectors(width)
        self.width, self.eps, self.streams = int(width), float(eps), int(streams)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, streams: cute.Pointer, weight: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(streams, weight, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, streams: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(1, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((1, self.warps), stride=(self.warps, 1)))
        count = self.row.count
        total = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        part = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        base = Int64(streams.toint()) + token * Int64(self.streams * self.width * 2)
        self.row.load(base, total)
        for s in cutlass.range_constexpr(1, self.streams):
            self.row.load(base + Int64(s * self.width * 2), part)
            for k in cutlass.range_constexpr(count):
                total[k] = total[k] + part[k]
        square = Float32(0.0)
        for k in cutlass.range_constexpr(count):
            total[k] = _bf16(total[k] * Float32(1.0 / self.streams))
            square = square + total[k] * total[k]
        inv = _rsqrt(_block_sum(square, sums, Int32(0), self.warps) / Float32(self.width) + Float32(self.eps))
        self.row.load(Int64(weight.toint()), part)
        for k in cutlass.range_constexpr(count):
            total[k] = part[k] * _bf16(total[k] * inv)
        self.row.store(Int64(out.toint()) + token * Int64(self.width * 2), total)


class GlmfAdd:
    """``out = bf16(a + b)`` over BF16 ``[rows, H]`` (routed + shared expert outputs)."""

    threads = 256

    def __init__(self, width: int):
        self.row = _RowVectors(width)
        self.width = int(width)

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(a, b, out).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Pointer, b: cute.Pointer, out: cute.Pointer):
        offset = Int64(cute.arch.block_idx()[0]) * Int64(self.width * 2)
        count = self.row.count
        x = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        y = cute.make_rmem_tensor(cute.make_layout((count,), stride=(1,)), Float32)
        self.row.load(Int64(a.toint()) + offset, x)
        self.row.load(Int64(b.toint()) + offset, y)
        for k in cutlass.range_constexpr(count):
            x[k] = x[k] + y[k]
        self.row.store(Int64(out.toint()) + offset, x)


class GlmfKdaConv:
    """Causal short conv + SiLU over the q/k/v in-projection columns.

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
        out[row, ch] = div_rn_f32(acc, Float32(1.0) + cute.math.exp(-acc, fastmath=False)).to(BFloat16)


class GlmfKdaConvState:
    """After the conv: each sequence's last step row stores its last three
    in-projection rows (older ones from the previous state) into its slot."""

    threads = 256

    def __init__(self, *, channels: int, proj_width: int):
        self.channels, self.proj_width = int(channels), int(proj_width)

    @cute.jit
    def __call__(self, proj: cute.Pointer, conv_state: cute.Pointer, slots: cute.Pointer, seq_first: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(proj, cute.make_layout((m, self.proj_width), stride=(self.proj_width, 1))),
            conv_state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(seq_first, cute.make_layout((m,))),
            rows,
        ).launch(grid=(rows, self.channels // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, proj: cute.Tensor, conv_state: cute.Pointer, slots: cute.Tensor, seq_first: cute.Tensor,
               rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        last = row + Int64(1) == Int64(rows)
        if not last:
            last = Int64(seq_first[row + Int64(1)]) != first
        slot = Int64(slots[row])
        if last and slot >= Int64(0):
            state = _conv_state(conv_state, slot, self.channels)
            n = row - first + Int64(1)
            kept = cute.make_rmem_tensor(cute.make_layout((CONV_TAPS - 1,), stride=(1,)), BFloat16)
            for i in cutlass.range_constexpr(CONV_TAPS - 1):
                s = n - Int64(CONV_TAPS - 1 - i)
                if s >= Int64(0):
                    kept[i] = proj[first + s, ch]
                else:
                    kept[i] = state[Int64(CONV_TAPS - 1) + s, ch]
            for i in cutlass.range_constexpr(CONV_TAPS - 1):
                state[i, ch] = kept[i]


class GlmfKdaGatedNorm:
    """``y = bf16(w * rmsnorm(o) * sigmoid(gate))`` per 128-wide head.

    ``o`` BF16 ``[rows, D]``, ``gate`` BF16 rows of stride ``gate_stride``,
    ``weight`` BF16 ``[128]``, ``y`` BF16 ``[rows, D]``. A warp per (row,
    head), grid ``(rows, heads / 8)``.
    """

    threads = 256

    def __init__(self, *, heads: int, eps: float, gate_stride: int):
        self.heads, self.eps, self.gate_stride = int(heads), float(eps), int(gate_stride)
        self.width = self.heads * KDA_HEAD
        if self.heads % 8:
            raise ValueError("KDA heads must divide into 8 warps")

    @cute.jit
    def __call__(self, o: cute.Pointer, gate: cute.Pointer, weight: cute.Pointer, y: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        d = self.width
        self.kernel(
            cute.make_tensor(o, cute.make_layout((m, d), stride=(d, 1))),
            cute.make_tensor(gate, cute.make_layout((m, d), stride=(self.gate_stride, 1))),
            cute.make_tensor(weight, cute.make_layout((KDA_HEAD,))),
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
            x[i] = Float32(o[row, head * Int64(KDA_HEAD) + lane * Int64(4) + Int64(i)])
            square = square + x[i] * x[i]
        inv = _rsqrt(_warp_sum(square) / Float32(KDA_HEAD) + Float32(self.eps))
        for i in cutlass.range_constexpr(4):
            d = lane * Int64(4) + Int64(i)
            g = Float32(gate[row, head * Int64(KDA_HEAD) + d])
            y[row, head * Int64(KDA_HEAD) + d] = (Float32(weight[d]) * (x[i] * inv) * _sigmoid(g)).to(BFloat16)


class GlmfKdaRecurrent:
    """Token-sequential lower-bounded KDA over one step's rows.

    ``qkv`` BF16 ``[rows, 3D]`` (conv outputs q | k | v), ``g_raw`` BF16 rows
    of stride ``g_stride`` (``f_b(f_a(x))``), ``b_raw`` BF16 rows of stride
    ``b_stride`` (``b_proj(x)``, one per head), ``a_log`` FP32 ``[H]``,
    ``dt_bias`` FP32 ``[D]``, ``state`` FP32 ``[slots, H, 128 (v), 128 (k)]``
    (the b12x ``gdn_decode`` layout), ``slots`` i32 ``[rows]`` (negative: zero
    state, not stored), ``out`` BF16 ``[rows, D]``.

    Grid ``(H, 128 / 32)``: a CTA owns 32 value rows of one head, each of its 8
    warps 4 rows, each lane 4 key columns of those rows in registers. Rows run
    in order; a row whose slot differs from the previous row's stores the
    previous sequence's state and loads its own.
    """

    threads = 256
    v_block = 32

    def __init__(self, *, heads: int, lower_bound: float, qkv_width: int, g_stride: int, b_stride: int,
                 eps: float = 1.0e-6):
        self.heads = int(heads)
        self.width = self.heads * KDA_HEAD
        self.lower_bound = float(lower_bound)
        self.qkv_width, self.g_stride, self.b_stride = int(qkv_width), int(g_stride), int(b_stride)
        self.eps = float(eps)
        self.scale = KDA_HEAD ** -0.5

    @cute.jit
    def __call__(self, qkv: cute.Pointer, g_raw: cute.Pointer, b_raw: cute.Pointer, a_log: cute.Pointer,
                 dt_bias: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        d = self.width
        self.kernel(
            cute.make_tensor(qkv, cute.make_layout((m, self.qkv_width), stride=(self.qkv_width, 1))),
            cute.make_tensor(g_raw, cute.make_layout((m, d), stride=(self.g_stride, 1))),
            cute.make_tensor(b_raw, cute.make_layout((m, self.heads), stride=(self.b_stride, 1))),
            cute.make_tensor(a_log, cute.make_layout((self.heads,))),
            cute.make_tensor(dt_bias, cute.make_layout((d,))),
            state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(out, cute.make_layout((m, d), stride=(d, 1))),
            rows,
        ).launch(grid=(self.heads, KDA_HEAD // self.v_block, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _state_tensor(self, state: cute.Pointer, slot: Int64, head: Int64) -> cute.Tensor:
        base = Int64(state.toint()) + ((slot * Int64(self.heads) + head) * Int64(KDA_HEAD * KDA_HEAD)) * Int64(4)
        return cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=16),
                                cute.make_layout((KDA_HEAD, KDA_HEAD), stride=(KDA_HEAD, 1)))

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, g_raw: cute.Tensor, b_raw: cute.Tensor, a_log: cute.Tensor,
               dt_bias: cute.Tensor, state: cute.Pointer, slots: cute.Tensor, out: cute.Tensor, rows: Int32):
        head = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        v0 = Int64(cute.arch.block_idx()[1]) * Int64(self.v_block) + Int64(tidx // Int32(32)) * Int64(4)
        k0 = Int64(lane) * Int64(4)
        col = head * Int64(KDA_HEAD)
        d = Int64(self.width)
        s = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), Float32)
        q = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        k = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        decay = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        bias = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(4):
            bias[i] = Float32(dt_bias[col + k0 + Int64(i)])
            for j in cutlass.range_constexpr(4):
                s[j, i] = Float32(0.0)
        rate = cute.math.exp(Float32(a_log[head]), fastmath=False)
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
                q[i] = Float32(qkv[row, col + k0 + Int64(i)])
                k[i] = Float32(qkv[row, d + col + k0 + Int64(i)])
                q_sq = q_sq + q[i] * q[i]
                k_sq = k_sq + k[i] * k[i]
                gate = Float32(self.lower_bound) * _sigmoid(rate * (Float32(g_raw[row, col + k0 + Int64(i)])
                                                                    + bias[i]))
                decay[i] = cute.math.exp(gate, fastmath=False)
            q_norm = cute.math.sqrt(_warp_sum(q_sq) + Float32(self.eps), fastmath=False)
            k_norm = cute.math.sqrt(_warp_sum(k_sq) + Float32(self.eps), fastmath=False)
            for i in cutlass.range_constexpr(4):
                q[i] = div_rn_f32(q[i], q_norm) * Float32(self.scale)
                k[i] = div_rn_f32(k[i], k_norm)
            beta = _bf16(_sigmoid(Float32(b_raw[row, head])))
            for j in cutlass.range_constexpr(4):
                v = Float32(qkv[row, d + d + col + v0 + Int64(j)])
                memory = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    s[j, i] = s[j, i] * decay[i]
                    memory = memory + s[j, i] * k[i]
                delta = (v - _warp_sum(memory)) * beta
                o = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    s[j, i] = s[j, i] + k[i] * delta
                    o = o + s[j, i] * q[i]
                o = _warp_sum(o)
                if lane == Int32(j):
                    out[row, col + v0 + Int64(j)] = o.to(BFloat16)
        if current >= Int64(0):
            old = self._state_tensor(state, current, head)
            for j in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    old[v0 + Int64(j), k0 + Int64(i)] = s[j, i]


__all__ = [
    "GlmfAdd",
    "GlmfKdaConv",
    "GlmfKdaConvState",
    "GlmfKdaGatedNorm",
    "GlmfKdaRecurrent",
    "GlmfMeanNorm",
]
