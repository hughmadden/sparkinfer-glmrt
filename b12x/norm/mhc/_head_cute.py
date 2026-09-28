"""CuTe DSL port of the terminal mHC head (sigmoid collapse + final RMSNorm).

Mirrors the Triton ``_mhc_head_fused_kernel`` used by ``run_head``:

    rms   = rsqrt(sum_l,h r[l,h]^2 / (4H) + rms_eps)
    mix_l = rms * sum_{l',h} fn[l, l'*H + h] * r[l',h]          (l = 0..3)
    pre_l = sigmoid(mix_l * scale[0] + bias[l]) + hc_eps
    c[h]  = bf16(sum_l pre_l * r[l,h])
    out[h] = bf16(c[h] * rsqrt(sum_h c[h]^2 / H + norm_eps) * weight[h])

One CTA per token; the reduction order differs from Triton's, so results
agree to FP32 rounding (BF16 outputs are equal or one ulp apart). The launch
takes raw pointers and a runtime token count for native AOT export.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, const_expr

_HEAD_THREADS = 256
_HEAD_REDUCTIONS = 5  # sum r^2 and four mixes


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


class MhcHeadKernel:
    """Terminal head: residual [T,4,H] BF16 -> normalized [T,H] BF16."""

    def __init__(self, *, hidden_size: int, rms_eps: float, hc_eps: float,
                 norm_eps: float, store_collapsed: bool = False):
        self.hidden = int(hidden_size)
        if self.hidden % _HEAD_THREADS:
            raise ValueError(f"mHC head hidden {self.hidden} must divide by {_HEAD_THREADS}")
        self.per_thread = self.hidden // _HEAD_THREADS
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.store_collapsed = bool(store_collapsed)
        self.warps = _HEAD_THREADS // 32

    def _storage(self):
        warps = self.warps

        class HeadStorage:
            pass

        HeadStorage.__annotations__ = {
            "sums": cute.struct.Align[
                cute.struct.MemRange[Float32, _HEAD_REDUCTIONS * warps], 16],
        }
        return cute.struct(HeadStorage)

    @cute.jit
    def __call__(self, residual: cute.Pointer, fn: cute.Pointer, scale: cute.Pointer,
                 base: cute.Pointer, norm: cute.Pointer, collapsed: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        h = self.hidden
        m = Int64(rows)
        r = cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        w = cute.make_tensor(fn, cute.make_layout((4, 4 * h), stride=(4 * h, 1)))
        s = cute.make_tensor(scale, cute.make_layout((1,)))
        b = cute.make_tensor(base, cute.make_layout((4,)))
        g = cute.make_tensor(norm, cute.make_layout((h,)))
        c = cute.make_tensor(collapsed, cute.make_layout((m, h), stride=(h, 1)))
        y = cute.make_tensor(out, cute.make_layout((m, h), stride=(h, 1)))
        self.kernel(r, w, s, b, g, c, y).launch(
            grid=(rows, 1, 1), block=(_HEAD_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, r: cute.Tensor, w: cute.Tensor, s: cute.Tensor, b: cute.Tensor,
               g: cute.Tensor, c: cute.Tensor, y: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sums = storage.sums.get_tensor(
            cute.make_layout((_HEAD_REDUCTIONS, self.warps), stride=(self.warps, 1)))

        acc = cute.make_rmem_tensor(cute.make_layout((_HEAD_REDUCTIONS,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(_HEAD_REDUCTIONS):
            acc[i] = Float32(0.0)
        hidden = Int64(self.hidden)
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _HEAD_THREADS) + Int64(tidx)
            r0 = Float32(r[token, 0, hh])
            r1 = Float32(r[token, 1, hh])
            r2 = Float32(r[token, 2, hh])
            r3 = Float32(r[token, 3, hh])
            acc[0] = acc[0] + (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3)
            for mix in cutlass.range_constexpr(4):
                acc[1 + mix] = acc[1 + mix] + (
                    Float32(w[mix, hh]) * r0
                    + Float32(w[mix, hidden + hh]) * r1
                    + Float32(w[mix, Int64(2) * hidden + hh]) * r2
                    + Float32(w[mix, Int64(3) * hidden + hh]) * r3
                )
        for i in cutlass.range_constexpr(_HEAD_REDUCTIONS):
            acc[i] = _warp_sum(acc[i])
        if lane == Int32(0):
            for i in cutlass.range_constexpr(_HEAD_REDUCTIONS):
                sums[i, warp] = acc[i]
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(_HEAD_REDUCTIONS):
            total = Float32(0.0)
            for src in cutlass.range_constexpr(self.warps):
                total = total + sums[i, src]
            acc[i] = total
        cute.arch.sync_threads()

        input_rms = cute.math.rsqrt(acc[0] / Float32(4 * self.hidden) + Float32(self.rms_eps))
        hc_scale = Float32(s[0])
        pre = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for mix in cutlass.range_constexpr(4):
            logit = acc[1 + mix] * input_rms * hc_scale + Float32(b[mix])
            pre[mix] = Float32(1.0) / (Float32(1.0) + cute.math.exp(-logit, fastmath=False)) \
                + Float32(self.hc_eps)

        kept = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _HEAD_THREADS) + Int64(tidx)
            value = (pre[0] * Float32(r[token, 0, hh]) + pre[1] * Float32(r[token, 1, hh])
                     + pre[2] * Float32(r[token, 2, hh]) + pre[3] * Float32(r[token, 3, hh]))
            rounded = value.to(BFloat16)
            if const_expr(self.store_collapsed):
                c[token, hh] = rounded
            kept[j] = Float32(rounded)
            square = square + kept[j] * kept[j]
        square = _warp_sum(square)
        if lane == Int32(0):
            sums[0, warp] = square
        cute.arch.sync_threads()
        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = total + sums[0, src]
        output_rms = cute.math.rsqrt(total / Float32(self.hidden) + Float32(self.norm_eps))
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _HEAD_THREADS) + Int64(tidx)
            y[token, hh] = (kept[j] * output_rms * Float32(g[hh])).to(BFloat16)


__all__ = ["MhcHeadKernel"]
