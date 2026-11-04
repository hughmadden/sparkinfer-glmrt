"""Producer epilogues over existing core outputs; no intermediate BF16 plane."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from b12x._lib.intrinsics import ld_global_v2_u32
from b12x.gemm.bf16_gemv._skinny import _bf16_lo, _bf16_hi
from b12x.moe._shared.kernels.w4a16.reply_store import ReplyStore
from ._common import Operand, Scalar, compile_program
from ._fp8_moe_kernels import MoeCombine
from ._fp8_weights import _ld_f32
from .pointwise import _view


class _Fp8Reply:
    def __init__(self, hidden, top_k, format, row_addresses):
        self.hidden, self.top_k = hidden, top_k
        self.row_addresses = row_addresses
        self.store = ReplyStore(hidden, format)

    @cute.jit
    def __call__(self, y: cute.Pointer, pair_pos: cute.Pointer, weights: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(y, pair_pos, weights, out).launch(
            grid=(rows, self.hidden // 128, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, y: cute.Pointer, pair_pos: cute.Pointer, weights: cute.Pointer,
               out: cute.Pointer):
        row, block, _ = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        col = Int64(block) * Int64(128) + Int64(lane) * Int64(4)
        positions = cute.make_ptr(Int32, pair_pos.toint(), cute.AddressSpace.gmem, assumed_align=4)
        values = cute.make_rmem_tensor((4,), Float32)
        values.fill(Float32(0.0))
        for slot in cutlass.range_constexpr(self.top_k):
            route = Int64(row) * Int64(self.top_k) + Int64(slot)
            p = positions[route]
            if p >= Int32(0):
                weight = _ld_f32(Int64(weights.toint()) + route * Int64(4))
                lo, hi = ld_global_v2_u32(Int64(y.toint()) +
                    (Int64(p) * Int64(self.hidden) + col) * Int64(2))
                values[0] = values[0] + weight * _bf16_lo(lo)
                values[1] = values[1] + weight * _bf16_hi(lo)
                values[2] = values[2] + weight * _bf16_lo(hi)
                values[3] = values[3] + weight * _bf16_hi(hi)
        destination = out
        token = Int32(row)
        if cutlass.const_expr(self.row_addresses):
            addresses = cute.make_ptr(Int64, out.toint(), cute.AddressSpace.gmem, assumed_align=8)
            destination = cute.make_ptr(Uint8, addresses[row], cute.AddressSpace.gmem, assumed_align=16)
            token = Int32(0)
        self.store.store(_view(destination), token, Int32(block), lane,
                         values[0], values[1], values[2], values[3])


def compile_fp8_reply_aot(*, hidden, top_k, format, row_addresses=False):
    if hidden <= 0 or hidden % 128 or top_k <= 0 or format not in (0, 1, 2):
        raise ValueError("FP8 reply requires H128, positive top-k, format 0/1/2")
    launch = (MoeCombine(hidden=hidden, top_k=top_k, row_addresses=row_addresses) if format == 0 else
              _Fp8Reply(hidden, top_k, format, row_addresses))
    row_bytes = hidden * 2 if format == 0 else hidden + hidden // 32 if format == 1 else hidden // 2 + hidden // 16
    return compile_program(launch, name=f"fp8_reply_h{hidden}_k{top_k}_{format}",
        operands=(Operand("y", torch.bfloat16, f"[grouped,{hidden}]"),
                  Operand("pair_pos", torch.int32, f"[rows,{top_k}]", align=4),
                  Operand("weights", torch.float32, f"[rows,{top_k}]", align=4),
                  Operand("out", torch.int64 if row_addresses else torch.bfloat16 if format == 0 else torch.uint8,
                          "[rows]" if row_addresses else f"[rows,{hidden if format == 0 else row_bytes}]", "out", align=8 if row_addresses else 16)),
        scalars=(Scalar("rows"),), key=(hidden, top_k, format, bool(row_addresses)),
        geometry=dict(hidden=hidden, top_k=top_k, format=format, row_bytes=row_bytes,
                      row_addresses=bool(row_addresses)))


class _Exlr8Reply:
    def __init__(self, hidden, units, top_k, max_groups, variant, format):
        from b12x.moe._shared.kernels.w4a16.exlr8 import Exlr8A16Kernel, Exlr8RoutedServingKernel
        self.core = (Exlr8A16Kernel(hidden, units, top_k, max_groups, device_count=True)
                     if variant == "baseline" else
                     Exlr8RoutedServingKernel(hidden, units, top_k, max_groups, variant=variant))
        if format:
            self.core.reply_format = format
            self.core.reply_store = ReplyStore(hidden, format)

    @cute.jit
    def __call__(self, route_group_lane: cute.Pointer, route_weights: cute.Pointer,
                 fc2: cute.Pointer, signs_out: cute.Pointer, output: cute.Pointer,
                 group_count: cute.Pointer, rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(64) and cta_limit > Int32(0):
            self.core.reduce_kernel(_view(route_group_lane), _view(route_weights),
                _view(fc2), _view(signs_out), _view(output), rows, _view(group_count)).launch(
                    grid=(min((rows * Int32(self.core.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
                    block=(128, 1, 1), stream=stream)


def compile_exlr8_reply_aot(*, hidden, units, top_k, max_groups,
                            variant="k1_k23_coalesced", format=0):
    if format not in (0, 1, 2):
        raise ValueError("EXLR8 reply format must be 0/1/2")
    launch = _Exlr8Reply(hidden, units, top_k, max_groups, variant, format)
    row_bytes = hidden * 2 if format == 0 else hidden + hidden // 32 if format == 1 else hidden // 2 + hidden // 16
    return compile_program(launch, name=f"exlr8_reply_h{hidden}_k{top_k}_{variant}_{format}",
        operands=(Operand("route_group_lane", torch.int32, f"[rows,{top_k}]", align=4),
                  Operand("route_weights", torch.float32, f"[rows,{top_k}]", align=4),
                  Operand("fc2", torch.float32, "[fc2_scratch]"),
                  Operand("signs_out", torch.int8, f"[{hidden}]", align=4),
                  Operand("output", torch.bfloat16 if format == 0 else torch.uint8,
                          f"[rows,{hidden if format == 0 else row_bytes}]", "out"),
                  Operand("group_count", torch.int32, "[1]")),
        scalars=(Scalar("rows"), Scalar("cta_limit")),
        key=(hidden, units, top_k, max_groups, variant, format),
        geometry=dict(hidden=hidden, top_k=top_k, format=format, row_bytes=row_bytes,
                      variant=variant))
