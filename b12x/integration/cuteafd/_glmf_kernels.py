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

Speculative verify steps (``spec`` != 0) leave the conv and recurrent state
untouched and record each row's replay inputs instead (``kda_replay_bytes``
per layer: the normalized key, decay and value per head in FP32, beta, and
the q/k/v in-projection row); ``GlmfKdaCommit`` / ``GlmfKdaConvCommit`` then
apply a sequence's accepted rows with the recurrent step's own arithmetic, so
the state equals that of serial steps over those rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    ld_global_v4_u32,
    pack_f32x2_to_bfloat2,
    st_global_v4_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._glm_kernels import _bf16, _block_sum, _fp8_scale, _reduction_storage, _rsqrt, _warp_max, _warp_sum

KDA_HEAD = 128
CONV_TAPS = 4
#: Rows one speculative step records per layer (the decode programs' rows).
REPLAY_ROWS = 64


def kda_replay_layout(heads: int, channels: int) -> tuple[int, int, int]:
    """Byte offsets of a layer's replay record: k | decay | v FP32 ``[REPLAY_ROWS, heads, 3, 128]``
    at 0, beta FP32 ``[REPLAY_ROWS, heads]``, the in-projection q|k|v BF16 ``[REPLAY_ROWS, channels]``;
    returns (beta offset, projection offset, total bytes)."""
    beta = REPLAY_ROWS * heads * 3 * KDA_HEAD * 4
    proj = beta + REPLAY_ROWS * heads * 4
    return beta, proj, proj + REPLAY_ROWS * channels * 2


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
                 replay: cute.Pointer, spec: Int32, rows: Int32, stream: cuda.CUstream):
        """With ``spec`` != 0 every row's q/k/v in-projection goes to ``replay``
        (``[REPLAY_ROWS, channels]`` BF16) and the state stays as it was."""
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(proj, cute.make_layout((m, self.proj_width), stride=(self.proj_width, 1))),
            conv_state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(seq_first, cute.make_layout((m,))),
            cute.make_tensor(replay, cute.make_layout((m, self.channels), stride=(self.channels, 1))),
            spec, rows,
        ).launch(grid=(rows, self.channels // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, proj: cute.Tensor, conv_state: cute.Pointer, slots: cute.Tensor, seq_first: cute.Tensor,
               replay: cute.Tensor, spec: Int32, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        last = row + Int64(1) == Int64(rows)
        if not last:
            last = Int64(seq_first[row + Int64(1)]) != first
        slot = Int64(slots[row])
        if spec != Int32(0):
            replay[row, ch] = proj[row, ch]
        else:
            if last and slot >= Int64(0):
                _shift_conv_state(_conv_state(conv_state, slot, self.channels), proj, first, row - first + Int64(1),
                                  ch)


@cute.jit
def _shift_conv_state(state: cute.Tensor, proj: cute.Tensor, first: Int64, n: Int64, ch: Int64):
    """``state`` (a slot's ``[3, C]``) becomes the last three of its rows followed by
    ``proj`` rows ``first .. first + n`` (column ``ch``)."""
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
                 dt_bias: cute.Pointer, state: cute.Pointer, slots: cute.Pointer, out: cute.Pointer,
                 replay: cute.Pointer, spec: Int32, rows: Int32, stream: cuda.CUstream):
        """With ``spec`` != 0 the state is read but not stored; each row's
        normalized key, decay, value and beta go to ``replay`` (see
        ``kda_replay_layout``) for ``GlmfKdaCommit``."""
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
            replay, spec, rows,
        ).launch(grid=(self.heads, KDA_HEAD // self.v_block, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _state_tensor(self, state: cute.Pointer, slot: Int64, head: Int64) -> cute.Tensor:
        base = Int64(state.toint()) + ((slot * Int64(self.heads) + head) * Int64(KDA_HEAD * KDA_HEAD)) * Int64(4)
        return cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=16),
                                cute.make_layout((KDA_HEAD, KDA_HEAD), stride=(KDA_HEAD, 1)))

    @cute.kernel
    def kernel(self, qkv: cute.Tensor, g_raw: cute.Tensor, b_raw: cute.Tensor, a_log: cute.Tensor,
               dt_bias: cute.Tensor, state: cute.Pointer, slots: cute.Tensor, out: cute.Tensor,
               replay_ptr: cute.Pointer, spec: Int32, rows: Int32):
        head = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        beta_off, _, _ = kda_replay_layout(self.heads, 3 * self.width)
        replay = _replay_rows(Int64(replay_ptr.toint()), self.heads)
        replay_beta = _replay_beta(Int64(replay_ptr.toint()) + Int64(beta_off), self.heads)
        # Warp 0 of the head's first CTA records the row's key, decay and beta.
        recorder = Int32(cute.arch.block_idx()[1]) * Int32(self.threads) + tidx < Int32(32)
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
                    if spec == Int32(0):
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
            if spec != Int32(0):
                if recorder:
                    for i in cutlass.range_constexpr(4):
                        replay[row, head, 0, k0 + Int64(i)] = k[i]
                        replay[row, head, 1, k0 + Int64(i)] = decay[i]
                    if tidx == Int32(0):
                        replay_beta[row, head] = beta
            for j in cutlass.range_constexpr(4):
                v = Float32(qkv[row, d + d + col + v0 + Int64(j)])
                if spec != Int32(0):
                    if lane == Int32(j):
                        replay[row, head, 2, v0 + Int64(j)] = v
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
            if spec == Int32(0):
                old = self._state_tensor(state, current, head)
                for j in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        old[v0 + Int64(j), k0 + Int64(i)] = s[j, i]


@cute.jit
def _replay_rows(address: Int64, heads: cutlass.Constexpr) -> cute.Tensor:
    """A layer's recorded k | decay | v at ``address``, FP32 ``[REPLAY_ROWS, heads, 3, 128]``."""
    return cute.make_tensor(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=16),
                            cute.make_layout((REPLAY_ROWS, heads, 3, KDA_HEAD),
                                             stride=(heads * 3 * KDA_HEAD, 3 * KDA_HEAD, KDA_HEAD, 1)))


@cute.jit
def _replay_beta(address: Int64, heads: cutlass.Constexpr) -> cute.Tensor:
    """A layer's recorded beta at ``address``, FP32 ``[REPLAY_ROWS, heads]``."""
    return cute.make_tensor(cute.make_ptr(Float32, address, cute.AddressSpace.gmem, assumed_align=16),
                            cute.make_layout((REPLAY_ROWS, heads), stride=(heads, 1)))


class GlmfKdaCommit:
    """Applies each sequence's accepted rows of a speculative step to its
    recurrent state, in every KDA layer at once.

    ``state`` FP32 ``[layers, slots, H, 128, 128]``, ``replay`` the layers'
    records (``replay_bytes`` apart, see ``kda_replay_layout``), ``tables``
    i32 ``[3, sequences]``: state slot, first step row and accepted rows of
    each sequence. Grid ``(H, 128 / 32, sequences * layers)`` with the
    recurrent kernel's thread mapping and update arithmetic (decay, memory,
    delta, rank-1 update per row), so the result is what serial steps store.
    """

    threads = 256
    v_block = 32

    def __init__(self, *, heads: int, channels: int):
        self.heads = int(heads)
        self.beta_off, _, self.replay_bytes = kda_replay_layout(self.heads, int(channels))

    @cute.jit
    def __call__(self, state: cute.Pointer, replay: cute.Pointer, tables: cute.Pointer, sequences: Int32,
                 layers: Int32, slots: Int32, stream: cuda.CUstream):
        self.kernel(state, replay, cute.make_tensor(tables, cute.make_layout((3, sequences), stride=(sequences, 1))),
                    sequences, slots).launch(
            grid=(self.heads, KDA_HEAD // self.v_block, sequences * layers), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, state: cute.Pointer, replay: cute.Pointer, tables: cute.Tensor, sequences: Int32, slots: Int32):
        head = Int64(cute.arch.block_idx()[0])
        z = Int32(cute.arch.block_idx()[2])
        seq = z % sequences
        layer = Int64(z // sequences)
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        v0 = Int64(cute.arch.block_idx()[1]) * Int64(self.v_block) + Int64(tidx // Int32(32)) * Int64(4)
        k0 = Int64(lane) * Int64(4)
        slot = Int64(tables[0, seq])
        first = Int64(tables[1, seq])
        keep = Int32(tables[2, seq])
        if slot >= Int64(0):
            base = Int64(state.toint()) + (((layer * Int64(slots) + slot) * Int64(self.heads) + head)
                                           * Int64(KDA_HEAD * KDA_HEAD)) * Int64(4)
            st = cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=16),
                                  cute.make_layout((KDA_HEAD, KDA_HEAD), stride=(KDA_HEAD, 1)))
            record = Int64(replay.toint()) + layer * Int64(self.replay_bytes)
            rows = _replay_rows(record, self.heads)
            betas = _replay_beta(record + Int64(self.beta_off), self.heads)
            s = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), Float32)
            k = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            decay = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            for j in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    s[j, i] = st[v0 + Int64(j), k0 + Int64(i)]
            for t in cutlass.range(keep, unroll=1):
                row = first + Int64(t)
                for i in cutlass.range_constexpr(4):
                    k[i] = rows[row, head, 0, k0 + Int64(i)]
                    decay[i] = rows[row, head, 1, k0 + Int64(i)]
                beta = betas[row, head]
                for j in cutlass.range_constexpr(4):
                    v = rows[row, head, 2, v0 + Int64(j)]
                    memory = Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        s[j, i] = s[j, i] * decay[i]
                        memory = memory + s[j, i] * k[i]
                    delta = (v - _warp_sum(memory)) * beta
                    for i in cutlass.range_constexpr(4):
                        s[j, i] = s[j, i] + k[i] * delta
            for j in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    st[v0 + Int64(j), k0 + Int64(i)] = s[j, i]


class GlmfKdaConvCommit:
    """Shifts each sequence's accepted in-projection rows of a speculative step
    into its conv state, in every KDA layer at once: ``conv_state`` BF16
    ``[layers, slots, 3, C]``, ``replay`` as ``GlmfKdaCommit``. Grid
    ``(sequences * layers, C / 256)``."""

    threads = 256

    def __init__(self, *, heads: int, channels: int):
        self.channels = int(channels)
        _, self.proj_off, self.replay_bytes = kda_replay_layout(int(heads), self.channels)

    @cute.jit
    def __call__(self, conv_state: cute.Pointer, replay: cute.Pointer, tables: cute.Pointer, sequences: Int32,
                 layers: Int32, slots: Int32, stream: cuda.CUstream):
        self.kernel(conv_state, replay, cute.make_tensor(tables, cute.make_layout((3, sequences), stride=(sequences, 1))),
                    sequences, slots).launch(
            grid=(sequences * layers, self.channels // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, conv_state: cute.Pointer, replay: cute.Pointer, tables: cute.Tensor, sequences: Int32,
               slots: Int32):
        z = Int32(cute.arch.block_idx()[0])
        seq = z % sequences
        layer = Int64(z // sequences)
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        slot = Int64(tables[0, seq])
        first = Int64(tables[1, seq])
        keep = Int64(tables[2, seq])
        if slot >= Int64(0):
            if keep > Int64(0):
                c = self.channels
                base = Int64(conv_state.toint()) + (layer * Int64(slots) + slot) * Int64((CONV_TAPS - 1) * c * 2)
                state = cute.make_tensor(cute.make_ptr(BFloat16, base, cute.AddressSpace.gmem, assumed_align=2),
                                         cute.make_layout((CONV_TAPS - 1, c), stride=(c, 1)))
                record = Int64(replay.toint()) + layer * Int64(self.replay_bytes) + Int64(self.proj_off)
                proj = cute.make_tensor(cute.make_ptr(BFloat16, record, cute.AddressSpace.gmem, assumed_align=16),
                                        cute.make_layout((REPLAY_ROWS, c), stride=(c, 1)))
                _shift_conv_state(state, proj, first, keep, ch)


class GlmfIndexPost:
    """DSA indexer epilogue without RoPE (GLM 5.3 Flash).

    ``iq`` BF16 ``[rows, H*128]`` = ``wq_b(q_resid)``; ``kw`` BF16 ``[rows,
    128 + H + 128]`` = ``[wk(x) | weights_proj(x) | kpool_compress_gate(x)]``.
    Per (row, head): E4M3 query with scale ``amax/448`` into ``q_fp8 [rows,
    H, 128]`` and ``head_weights = bf16(proj) * H^-0.5 * 128^-0.5 * q_scale``.
    Key: ``bf16(LayerNorm(wk(x)))`` and the raw gate, stored BF16 as ``[k |
    g]`` (256 values) at ``token_keys[slots[row]]`` (negative slots skip).
    One warp per (row, head) plus one for the key; lane ``l`` owns 4l..4l+3.
    """

    def __init__(self, *, heads: int, eps: float, weight_scale: float):
        self.heads, self.eps, self.weight_scale = int(heads), float(eps), float(weight_scale)
        self.width = 128 + self.heads + 128

    @cute.jit
    def __call__(self, iq: cute.Pointer, kw: cute.Pointer, slots: cute.Pointer, k_weight: cute.Pointer,
                 k_bias: cute.Pointer, q_fp8: cute.Pointer, head_weights: cute.Pointer, token_keys: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        h = self.heads
        self.kernel(
            cute.make_tensor(iq, cute.make_layout((m, h, 128), stride=(h * 128, 128, 1))),
            cute.make_tensor(kw, cute.make_layout((m, self.width), stride=(self.width, 1))),
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(k_weight, cute.make_layout((128,))),
            cute.make_tensor(k_bias, cute.make_layout((128,))),
            q_fp8,
            cute.make_tensor(head_weights, cute.make_layout((m, h), stride=(h, 1))),
            token_keys,
        ).launch(grid=(rows, h + 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, iq: cute.Tensor, kw: cute.Tensor, slots: cute.Tensor, k_weight: cute.Tensor,
               k_bias: cute.Tensor, q_fp8: cute.Pointer, head_weights: cute.Tensor, token_keys: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        head = Int32(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        out = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        if head < Int32(self.heads):
            h64 = Int64(head)
            for e in cutlass.range_constexpr(4):
                out[e] = Float32(iq[token, h64, Int64(4) * Int64(lane) + Int64(e)])
            local = fmax_f32(fmax_f32(fabs_f32(out[0]), fabs_f32(out[1])),
                             fmax_f32(fabs_f32(out[2]), fabs_f32(out[3])))
            scale = _fp8_scale(_warp_max(local))
            packed = cvt_f32x4_to_e4m3x4(div_rn_f32(out[0], scale), div_rn_f32(out[1], scale),
                                         div_rn_f32(out[2], scale), div_rn_f32(out[3], scale))
            words = cute.make_ptr(
                Uint32, Int64(q_fp8.toint()) + (token * Int64(self.heads) + h64) * Int64(128)
                + Int64(4) * Int64(lane), cute.AddressSpace.gmem, assumed_align=4)
            words[0] = packed
            if lane == Int32(0):
                raw = Float32(kw[token, Int64(128) + h64])
                head_weights[token, h64] = raw * Float32(self.weight_scale) * scale
        else:
            total = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                out[e] = Float32(kw[token, Int64(4) * Int64(lane) + Int64(e)])
                total = total + out[e]
            mean = _warp_sum(total) / Float32(128.0)
            var = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                diff = out[e] - mean
                var = var + diff * diff
            rstd = _rsqrt(_warp_sum(var) / Float32(128.0) + Float32(self.eps))
            slot = Int64(slots[token])
            if slot >= Int64(0):
                row = cute.make_tensor(
                    cute.make_ptr(BFloat16, Int64(token_keys.toint()) + slot * Int64(512), cute.AddressSpace.gmem,
                                  assumed_align=16), cute.make_layout((256,)))
                for e in cutlass.range_constexpr(4):
                    d = Int32(4) * lane + Int32(e)
                    row[d] = ((out[e] - mean) * rstd * Float32(k_weight[d]) + Float32(k_bias[d])).to(BFloat16)
                    row[Int32(128) + d] = kw[token, Int64(128 + self.heads) + Int64(d)]


class GlmfPoolKeys:
    """Compressed key of every pool a row completes (``pool_slots[row] >= 0``).

    The pool is the row's own token and the three before it (record slots
    ``slots[row] - 3 .. slots[row]``: a pool never crosses a 64-row page).
    Per channel ``c``: ``p_t = bf16(softmax_t(g[t, c] + ape[t, c]))`` and
    ``key[c] = bf16(sum_t bf16(p_t * k[t, c]))``, as the reference's
    ``get_pooled_states``; then E4M3 with a per-pool scale ``amax/448`` into
    the index cache (64 pools x 128 E4M3 then 64 FP32 scales per page) at
    ``pool_slots[row]``. One CTA of 128 threads (one per channel) per row.
    """

    threads = 128

    def __init__(self, *, kpool: int = 4, page_rows: int = 64):
        if kpool != 4:
            raise ValueError("pool keys are built for 4-token pools")
        self.kpool, self.page_rows = int(kpool), int(page_rows)
        self.page_bytes = self.page_rows * (128 + 4)
        self.warps = self.threads // 32

    @cute.jit
    def __call__(self, slots: cute.Pointer, pool_slots: cute.Pointer, ape: cute.Pointer, token_keys: cute.Pointer,
                 cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(pool_slots, cute.make_layout((m,))),
            cute.make_tensor(ape, cute.make_layout((self.kpool, 128), stride=(128, 1))),
            token_keys, cache,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, slots: cute.Tensor, pool_slots: cute.Tensor, ape: cute.Tensor, token_keys: cute.Pointer,
               cache: cute.Pointer):
        token = Int64(cute.arch.block_idx()[0])
        c = Int32(cute.arch.thread_idx()[0])
        lane = c % Int32(32)
        warp_id = c // Int32(32)
        pool = Int64(pool_slots[token])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(1, self.warps))
        maxes = storage.sums.get_tensor(cute.make_layout((1, self.warps), stride=(self.warps, 1)))
        if pool >= Int64(0):
            first = Int64(slots[token]) - Int64(self.kpool - 1)
            keys = cute.make_tensor(
                cute.make_ptr(BFloat16, Int64(token_keys.toint()) + first * Int64(512), cute.AddressSpace.gmem,
                              assumed_align=16), cute.make_layout((self.kpool, 256), stride=(256, 1)))
            logit = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
            top = Float32(-3.0e38)
            for t in cutlass.range_constexpr(4):
                logit[t] = Float32(keys[t, Int32(128) + c]) + Float32(ape[t, c])
                top = fmax_f32(top, logit[t])
            total = Float32(0.0)
            for t in cutlass.range_constexpr(4):
                logit[t] = cute.math.exp(logit[t] - top, fastmath=False)
                total = total + logit[t]
            key = Float32(0.0)
            for t in cutlass.range_constexpr(4):
                key = key + _bf16(_bf16(div_rn_f32(logit[t], total)) * Float32(keys[t, c]))
            key = _bf16(key)
            local = _warp_max(fabs_f32(key))
            if lane == Int32(0):
                maxes[0, warp_id] = local
            cute.arch.sync_threads()
            amax = Float32(0.0)
            for w in cutlass.range_constexpr(self.warps):
                amax = fmax_f32(amax, maxes[0, w])
            scale = _fp8_scale(amax)
            page = pool // Int64(self.page_rows)
            row = pool - page * Int64(self.page_rows)
            page_base = Int64(cache.toint()) + page * Int64(self.page_bytes)
            value = cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, page_base + row * Int64(128),
                                                   cute.AddressSpace.gmem, assumed_align=1), cute.make_layout((128,)))
            value[c] = div_rn_f32(key, scale).to(cutlass.Float8E4M3FN)
            if c == Int32(0):
                scale_ptr = cute.make_ptr(Float32, page_base + Int64(self.page_rows * 128) + row * Int64(4),
                                          cute.AddressSpace.gmem, assumed_align=4)
                scale_ptr[0] = scale


class GlmfIndexExpand:
    """Selected MLA record slots per row (``SPARSE`` wide) and their count.

    Rows whose context ``p + 1 <= dense_limit`` select every earlier token
    (the reference's selection there); longer rows expand the top-k pools
    (index-cache physical slots) to their 4 tokens, then append the open tail
    pool. Record slot of position ``q``: ``table[row * stride + q / 64] * 64
    + q % 64``; the pool of index slot ``s`` is ``pool_logical[s / 64] * 64 +
    s % 64``. One CTA of 256 threads per row.
    """

    threads = 256

    def __init__(self, *, pools: int, width: int, dense_limit: int, kpool: int = 4):
        self.pools, self.width, self.dense_limit, self.kpool = int(pools), int(width), int(dense_limit), int(kpool)

    @cute.jit
    def __call__(self, positions: cute.Pointer, pools: cute.Pointer, pool_logical: cute.Pointer,
                 page_table: cute.Pointer, indices: cute.Pointer, lengths: cute.Pointer, rows: Int32, stride: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(positions, cute.make_layout((m,))),
            cute.make_tensor(pools, cute.make_layout((m, self.pools), stride=(self.pools, 1))),
            pool_logical, page_table,
            cute.make_tensor(indices, cute.make_layout((m, self.width), stride=(self.width, 1))),
            cute.make_tensor(lengths, cute.make_layout((m,))),
            stride,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _slot(self, table: cute.Pointer, base: Int64, position: Int64) -> Int32:
        entry = cute.make_ptr(Int32, Int64(table.toint()) + (base + position // Int64(64)) * Int64(4),
                              cute.AddressSpace.gmem, assumed_align=4)
        return Int32(Int64(entry[0]) * Int64(64) + position % Int64(64))

    @cute.kernel
    def kernel(self, positions: cute.Tensor, pools: cute.Tensor, pool_logical: cute.Pointer,
               page_table: cute.Pointer, indices: cute.Tensor, lengths: cute.Tensor, stride: Int32):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int64(cute.arch.thread_idx()[0])
        p = Int64(positions[row])
        base = row * Int64(stride)
        count = p + Int64(1)
        if count <= Int64(self.dense_limit):
            for it in cutlass.range_constexpr((self.width + self.threads - 1) // self.threads):
                i = Int64(it * self.threads) + tidx
                if i < Int64(self.width):
                    value = Int32(-1)
                    if i < count:
                        value = self._slot(page_table, base, i)
                    indices[row, i] = value
            if tidx == Int64(0):
                lengths[row] = Int32(count)
        else:
            for it in cutlass.range_constexpr((self.pools + self.threads - 1) // self.threads):
                j = Int64(it * self.threads) + tidx
                if j < Int64(self.pools):
                    s = Int64(pools[row, j])
                    entry = cute.make_ptr(Int32, Int64(pool_logical.toint()) + s // Int64(64) * Int64(4),
                                          cute.AddressSpace.gmem, assumed_align=4)
                    pool = Int64(entry[0]) * Int64(64) + s % Int64(64)
                    for e in cutlass.range_constexpr(4):
                        indices[row, j * Int64(4) + Int64(e)] = self._slot(page_table, base,
                                                                            pool * Int64(4) + Int64(e))
            tail_start = count // Int64(self.kpool) * Int64(self.kpool)
            tail = count - tail_start
            filled = Int64(self.pools * self.kpool)
            if tidx < Int64(self.width) - filled:
                value = Int32(-1)
                if tidx < tail:
                    value = self._slot(page_table, base, tail_start + tidx)
                indices[row, filled + tidx] = value
            if tidx == Int64(0):
                lengths[row] = Int32(filled + tail)


__all__ = [
    "GlmfAdd",
    "GlmfIndexExpand",
    "GlmfIndexPost",
    "GlmfPoolKeys",
    "GlmfKdaConv",
    "GlmfKdaConvState",
    "GlmfKdaGatedNorm",
    "GlmfKdaRecurrent",
    "GlmfMeanNorm",
]
