"""MiniMax arithmetic not shared by the other family pointwise programs."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32
from b12x.gemm.bf16_gemv._kernel import SmallNGemvKernel
from ._glm_kernels import _fp8_scale, _warp_max
from ._qwen4_kernels import _sigmoid
from .pointwise import _add, _mul, _view


@cute.jit
def _swiglu_oai(gate: Float32, up: Float32):
    gate = cutlass.min(gate, Float32(7.0))
    up = cutlass.max(cutlass.min(up, Float32(7.0)), Float32(-7.0))
    return _mul(_mul(gate, _sigmoid(_mul(Float32(1.702), gate))), _add(up, Float32(1.0)))


class SwigluOai:
    def __init__(self, width):
        self.width = int(width)

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.width)
        self.kernel(gate_up, out, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Pointer, out: cute.Pointer, count: Int64):
        gate_up, out = _view(gate_up), _view(out)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            base = (i // Int64(self.width)) * Int64(2 * self.width) + i % Int64(self.width)
            out[i] = _swiglu_oai(Float32(gate_up[base]),
                Float32(gate_up[base + Int64(self.width)])).to(BFloat16)


class RouterScores:
    def __init__(self):
        self.project = SmallNGemvKernel(128, 6144, False, False, unrolled_scalar=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, logits: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.project(x, weight, weight, logits, rows, Int64(6144), Int64(6144), Int64(128),
                     Int64(1), Int64(1), Int32(0), stream)


class KvSplit:
    @cute.jit
    def __call__(self, kv: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(512)
        self.kernel(kv, k, v, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, kv: cute.Pointer, k: cute.Pointer, v: cute.Pointer, count: Int64):
        kv, k, v = _view(kv), _view(k), _view(v)
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < count:
            base = i // Int64(512) * Int64(1024) + i % Int64(512)
            k[i], v[i] = kv[base], kv[base + Int64(512)]


class Route:
    @cute.jit
    def __call__(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
                 weights: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(logits, bias, ids, weights).launch(grid=(rows, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer):
        logits, bias, ids, weights = _view(logits), _view(bias), _view(ids), _view(weights)
        row, lane = Int64(cute.arch.block_idx()[0]), Int32(cute.arch.thread_idx()[0])
        scores = cute.make_rmem_tensor((4,), Float32)
        selected = cute.make_rmem_tensor((4,), Float32)
        for i in cutlass.range_constexpr(4):
            expert = lane + Int32(i * 32)
            scores[i] = _add(_sigmoid(Float32(logits[row * Int64(128) + Int64(expert)])),
                             Float32(bias[Int64(expert)]))
        denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(4):
            best = Float32(float("-inf"))
            for i in cutlass.range_constexpr(4):
                best = cutlass.max(best, scores[i])
            best = _warp_max(best)
            winner = Int32(128)
            for i in cutlass.range_constexpr(4):
                if scores[i] == best:
                    winner = cutlass.min(winner, lane + Int32(i * 32))
            for shift in cutlass.range_constexpr(5):
                winner = cutlass.min(winner, cute.arch.shuffle_sync_bfly(winner, offset=1 << shift))
            value = _sigmoid(Float32(logits[row * Int64(128) + Int64(winner)]))
            selected[slot] = value
            denominator = _add(denominator, value)
            if lane == Int32(0):
                ids[row * Int64(4) + Int64(slot)] = winner
            for i in cutlass.range_constexpr(4):
                if lane + Int32(i * 32) == winner:
                    scores[i] = Float32(float("-inf"))
        if lane == Int32(0):
            for slot in cutlass.range_constexpr(4):
                weights[row * Int64(4) + Int64(slot)] = _mul(
                    div_rn_f32(selected[slot], denominator), Float32(2.0))


class IndexQuant:
    def __init__(self, heads, paged=False):
        self.heads, self.paged = int(heads), bool(paged)

    @cute.jit
    def __call__(self, x: cute.Pointer, slots: cute.Pointer, values: cute.Pointer,
                 scales: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, slots, values, scales).launch(grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, slots: cute.Pointer, values: cute.Pointer, scales: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        x, slots, scales = _view(x), _view(slots), _view(scales)
        slot = Int64(row)
        if cutlass.const_expr(self.paged):
            slot = Int64(slots[Int64(row)])
        if slot >= Int64(0):
            base = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(128) + Int64(lane * Int32(4))
            held = cute.make_rmem_tensor((4,), Float32)
            amax = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                held[i] = Float32(x[base + Int64(i)])
                amax = cutlass.max(amax, cute.math.abs(held[i]))
            scale = _fp8_scale(_warp_max(amax))
            inverse = div_rn_f32(Float32(1.0), scale)
            address = Int64(values.toint())
            if cutlass.const_expr(self.paged):
                address += slot // Int64(64) * Int64(8448)
                if lane == Int32(0):
                    scale_ptr = cute.make_ptr(Float32, address + Int64(8192) + slot % Int64(64) * Int64(4),
                                              cute.AddressSpace.gmem, assumed_align=4)
                    scale_ptr[0] = scale
                address += slot % Int64(64) * Int64(128)
            else:
                address += (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(128)
                if lane == Int32(0):
                    scales[Int64(row) * Int64(self.heads) + Int64(head)] = scale
            packed = cute.make_ptr(Uint32, address, cute.AddressSpace.gmem, assumed_align=16)
            packed[lane] = cvt_f32x4_to_e4m3x4(held[0] * inverse, held[1] * inverse,
                                              held[2] * inverse, held[3] * inverse)


class KvPack:
    @cute.jit
    def __call__(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
                 batch_ids: cute.Pointer, k_descale: cute.Pointer, v_descale: cute.Pointer,
                 key_cache: cute.Pointer, value_cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(k, v, slots, batch_ids, k_descale, v_descale, key_cache, value_cache).launch(
            grid=(rows, 4, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
               batch_ids: cute.Pointer, k_descale: cute.Pointer, v_descale: cute.Pointer,
               key_cache: cute.Pointer, value_cache: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        lane = Int64(cute.arch.thread_idx()[0])
        k, v, slots, batch_ids = _view(k), _view(v), _view(slots), _view(batch_ids)
        k_descale, v_descale = _view(k_descale), _view(v_descale)
        slot = Int64(slots[Int64(row)])
        if slot >= Int64(0):
            scale_at = Int64(batch_ids[Int64(row)]) * Int64(4) + Int64(head)
            ki = div_rn_f32(Float32(1.0), Float32(k_descale[scale_at]))
            vi = div_rn_f32(Float32(1.0), Float32(v_descale[scale_at]))
            src = (Int64(row) * Int64(4) + Int64(head)) * Int64(128) + lane * Int64(4)
            dst = (slot * Int64(4) + Int64(head)) * Int64(128)
            kp = cute.make_ptr(Uint32, Int64(key_cache.toint()) + dst, cute.AddressSpace.gmem, assumed_align=16)
            vp = cute.make_ptr(Uint32, Int64(value_cache.toint()) + dst, cute.AddressSpace.gmem, assumed_align=16)
            kp[lane] = cvt_f32x4_to_e4m3x4(Float32(k[src]) * ki, Float32(k[src + Int64(1)]) * ki,
                                          Float32(k[src + Int64(2)]) * ki, Float32(k[src + Int64(3)]) * ki)
            vp[lane] = cvt_f32x4_to_e4m3x4(Float32(v[src]) * vi, Float32(v[src + Int64(1)]) * vi,
                                          Float32(v[src + Int64(2)]) * vi, Float32(v[src + Int64(3)]) * vi)
