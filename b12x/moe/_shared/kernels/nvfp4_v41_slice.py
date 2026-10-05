"""Experimental native-K16 NVFP4 route-slot slices; no cooperative grid.

Caller owns native ModelOpt payload/F8_128x4 scales, per-expert alpha vectors,
FP32 [3, capacity*6, 5120] partials and BF16 route output. Live rows are runtime
launch arguments. FC1/FC2 keep the dynamic W4A4 alpha and BF16 activation
boundary; the ordered slice sum changes FC2 FP32 accumulation grouping.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Int32, Int64, Uint32, Uint64, Float32
from b12x._lib.intrinsics import (
    cp_async4_shared_global, cp_async_u32_shared_global, get_ptr_as_int64,
    shared_ptr_to_u32, nvfp4_mma_m16n8k64_f32_e2m1,
    quantize_block_fp4_fast, fabs_f32, fmax_f32, fmin_f32,
)


class V41Nvfp4Slice:
    hidden = 5120
    logical = 576
    stored = 640
    width = 192
    topk = 6

    @cute.jit
    def __call__(self, x: cute.Tensor, ids: cute.Tensor, routing: cute.Tensor,
                 w13: cute.Tensor, s13: cute.Tensor, w2: cute.Tensor,
                 s2: cute.Tensor, input_gs: cute.Tensor, alpha: cute.Tensor,
                 down_alpha: cute.Tensor, down_gs: cute.Tensor,
                 partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, ids, routing, w13, s13, w2, s2, input_gs, alpha,
                    down_alpha, down_gs, partial).launch(
            grid=(3, rows * self.topk, 1), block=(128, 1, 1),
            min_blocks_per_mp=2, stream=stream)
        self.reduce(partial, output, rows).launch(
            grid=(rows * self.topk, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def scale_word(self, scales: cute.Tensor, expert: Int32, row: Int32,
                   k64: Int32, n: int, k: int):
        # Native F8_128x4: four adjacent K16 bytes per row and K64 slice.
        return (Int64(expert) * Int64(n * k // 16)
                + Int64(row // 128) * Int64(k // 64 * 512)
                + Int64(k64) * Int64(512)
                + Int64(row % 32 * 16 + row % 128 // 32 * 4))

    @cute.kernel
    def kernel(self, x: cute.Tensor, ids: cute.Tensor, routing: cute.Tensor,
               w13: cute.Tensor, s13: cute.Tensor, w2: cute.Tensor,
               s2: cute.Tensor, input_gs: cute.Tensor, alpha: cute.Tensor,
               down_alpha: cute.Tensor, down_gs: cute.Tensor,
               partial: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        warp = tid // 32
        c, q = lane % 4, lane // 4
        sid, route, _ = cute.arch.block_idx()
        token = route // self.topk
        expert = ids[route]
        start = sid * self.width
        smem = cutlass.utils.SmemAllocator()
        # Reuse FC1 staging for FC2: 18,048 explicit shared bytes (<18 KiB).
        b = smem.allocate_tensor(Uint32, cute.make_layout(2 * self.width * 8), byte_alignment=16)
        sf = smem.allocate_tensor(Uint32, cute.make_layout(2 * self.width), byte_alignment=16)
        qa = smem.allocate_tensor(Uint32, cute.make_layout(self.hidden // 8), byte_alignment=16)
        qs = smem.allocate_tensor(Uint32, cute.make_layout(self.hidden // 16), byte_alignment=16)
        mid = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(self.width), byte_alignment=16)
        bb, sb = shared_ptr_to_u32(b.iterator), shared_ptr_to_u32(sf.iterator)
        for block in range(tid, self.hidden // 16, 128):
            values = cute.make_rmem_tensor((16,), Float32)
            maximum = Float32(0)
            for j in cutlass.range_constexpr(16):
                value = x[Int64(token) * self.hidden + block * 16 + j].to(Float32)
                values[j] = value
                maximum = fmax_f32(maximum, fabs_f32(value))
            payload, scale = quantize_block_fp4_fast(values, maximum, input_gs[expert])
            qa[block * 2] = Uint32(payload)
            qa[block * 2 + 1] = Uint32(payload >> Uint64(32))
            qs[block] = Uint32(scale)
        cute.arch.sync_threads()
        gate = cute.make_rmem_tensor((6, 4), Float32)
        up = cute.make_rmem_tensor((6, 4), Float32)
        gate.fill(0)
        up.fill(0)
        for kt in range(self.hidden // 64):
            for half in cutlass.range_constexpr(2):
                for i in range(tid, self.width * 2, 128):
                    row, vec = i // 2, i % 2
                    nr = start + row + half * self.stored
                    src = ((Int64(expert) * (2 * self.stored) + Int64(nr))
                           * (self.hidden // 8) + kt * 8 + vec * 4)
                    cp_async4_shared_global(bb + half * self.width * 32 + row * 32 + vec * 16,
                                           get_ptr_as_int64(w13, src))
                for row in range(tid, self.width, 128):
                    nr = start + row + half * self.stored
                    offset = self.scale_word(s13, expert, nr, kt, 2 * self.stored, self.hidden)
                    cp_async_u32_shared_global(sb + (half * self.width + row) * 4,
                                              get_ptr_as_int64(s13, offset))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            # Replicate the token in all MMA rows; only row zero is published.
            a0, a2 = qa[kt * 8 + c], qa[kt * 8 + c + 4]
            sa = qs[kt * 4] | (qs[kt * 4 + 1] << 8) | (qs[kt * 4 + 2] << 16) | (qs[kt * 4 + 3] << 24)
            for nf in cutlass.range_constexpr(6):
                nr = warp * 48 + nf * 8 + q
                u0, u1, u2, u3 = nvfp4_mma_m16n8k64_f32_e2m1(
                    up[nf, 0], up[nf, 1], up[nf, 2], up[nf, 3],
                    a0, a0, a2, a2, b[nr * 8 + c], b[nr * 8 + c + 4], sa, sf[nr])
                g0, g1, g2, g3 = nvfp4_mma_m16n8k64_f32_e2m1(
                    gate[nf, 0], gate[nf, 1], gate[nf, 2], gate[nf, 3],
                    a0, a0, a2, a2, b[(self.width + nr) * 8 + c],
                    b[(self.width + nr) * 8 + c + 4], sa, sf[self.width + nr])
                up[nf, 0], up[nf, 1], up[nf, 2], up[nf, 3] = u0, u1, u2, u3
                gate[nf, 0], gate[nf, 1], gate[nf, 2], gate[nf, 3] = g0, g1, g2, g3
            cute.arch.sync_threads()
        if q == 0:
            for nf in cutlass.range_constexpr(6):
                for j in cutlass.range_constexpr(2):
                    gv = fmin_f32(gate[nf, j] * alpha[expert], Float32(10))
                    uv = fmax_f32(Float32(-10), fmin_f32(up[nf, j] * alpha[expert], Float32(10)))
                    value = gv * cute.arch.rcp_approx(Float32(1) + cute.math.exp(-gv, fastmath=True)) * uv
                    mid[warp * 48 + nf * 8 + c * 2 + j] = value.to(cutlass.BFloat16)
        cute.arch.sync_threads()
        if tid < self.width // 16:
            values = cute.make_rmem_tensor((16,), Float32)
            maximum = Float32(0)
            for j in cutlass.range_constexpr(16):
                value = mid[tid * 16 + j].to(Float32)
                values[j] = value
                maximum = fmax_f32(maximum, fabs_f32(value))
            payload, scale = quantize_block_fp4_fast(values, maximum, down_gs[expert])
            qa[tid * 2] = Uint32(payload)
            qa[tid * 2 + 1] = Uint32(payload >> Uint64(32))
            qs[tid] = Uint32(scale)
        cute.arch.sync_threads()
        for ot in range(self.hidden // 128):
            acc = cute.make_rmem_tensor((4, 4), Float32)
            acc.fill(0)
            for kt in cutlass.range_constexpr(self.width // 64):
                for i in range(tid, 256, 128):
                    nr, vec = i // 2, i % 2
                    src = ((Int64(expert) * self.hidden + Int64(ot * 128 + nr))
                           * (self.stored // 8) + (start // 64 + kt) * 8 + vec * 4)
                    cp_async4_shared_global(bb + nr * 32 + vec * 16, get_ptr_as_int64(w2, src))
                offset = self.scale_word(s2, expert, ot * 128 + tid, start // 64 + kt, self.hidden, self.stored)
                cp_async_u32_shared_global(sb + tid * 4, get_ptr_as_int64(s2, offset))
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                a0, a2 = qa[kt * 8 + c], qa[kt * 8 + c + 4]
                sa = qs[kt * 4] | (qs[kt * 4 + 1] << 8) | (qs[kt * 4 + 2] << 16) | (qs[kt * 4 + 3] << 24)
                for nf in cutlass.range_constexpr(4):
                    nr = warp * 32 + nf * 8 + q
                    d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        acc[nf, 0], acc[nf, 1], acc[nf, 2], acc[nf, 3],
                        a0, a0, a2, a2, b[nr * 8 + c], b[nr * 8 + c + 4], sa, sf[nr])
                    acc[nf, 0], acc[nf, 1], acc[nf, 2], acc[nf, 3] = d0, d1, d2, d3
                cute.arch.sync_threads()
            if q == 0:
                for nf in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(2):
                        col = ot * 128 + warp * 32 + nf * 8 + c * 2 + j
                        partial[sid, route, col] = acc[nf, j] * (down_alpha[expert] * routing[route])

    @cute.kernel
    def reduce(self, partial: cute.Tensor, output: cute.Tensor, rows: Int32):
        route = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        for col in range(tid, self.hidden, 256):
            value = partial[0, route, col] + partial[1, route, col] + partial[2, route, col]
            output[Int64(route) * self.hidden + col] = value.to(cutlass.BFloat16)
