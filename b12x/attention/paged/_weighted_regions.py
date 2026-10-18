"""CuTe weighted-region compression and signed provider-group scoring.

Low-level entries consume caller-owned staging/output buffers. Compression never
publishes a partial region; the cache owner publishes staged results at commit.
"""
from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import BFloat16, Float32, Float8E4M3FN, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.intrinsics import (
    block_reduce, cvt_bf16x2x2_to_e4m3x4, fmax_f32,
    mma_m16n8k32_f32_e4m3, warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _CompressRegions:
    @cute.jit
    def __call__(self, keys: cute.Pointer, logits: cute.Pointer,
                 starts: cute.Pointer, counts: cute.Pointer, output: cute.Pointer,
                 unrounded: cute.Pointer, regions: Int32, limit: Int64, stream: cuda.CUstream):
        self.kernel(keys, logits, starts, counts, output, unrounded, limit).launch(
            grid=(regions, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, keys: cute.Pointer, logits: cute.Pointer,
               starts: cute.Pointer, counts: cute.Pointer, output: cute.Pointer,
               unrounded: cute.Pointer, limit: Int64):
        region, _, _ = cute.arch.block_idx()
        dim, _, _ = cute.arch.thread_idx()
        # Uniform early exit: incomplete/unused slots remain byte-untouched.
        start = starts[region].to(Int64)
        if (counts[region] == Int32(8)) & (start >= Int64(0)) & (start + Int64(8) <= limit):
            z = cute.make_rmem_tensor((8,), Float32)
            k = cute.make_rmem_tensor((8,), Float32)
            maximum = Float32(-float("inf"))
            for token in cutlass.range_constexpr(8):
                offset = (start + Int64(token)) * Int64(256) + Int64(dim)
                z[token] = Float32(logits[offset])
                k[token] = Float32(keys[offset])
                maximum = fmax_f32(maximum, z[token])
            allocator = cutlass.utils.SmemAllocator()
            scratch = allocator.allocate_tensor(
                Float32, cute.make_layout((1, 8)), byte_alignment=16)
            maximum = block_reduce(warp_reduce(maximum, fmax_f32), fmax_f32,
                                   scratch, Float32(-float("inf")))
            for token in cutlass.range_constexpr(8):
                z[token] = cute.math.exp(z[token] - maximum, fastmath=True)
                k[token] = z[token] * k[token]
            for stride in cutlass.range_constexpr(3):
                delta = 4 >> stride
                for token in cutlass.range_constexpr(delta):
                    z[token] = z[token] + z[token + delta]
                    k[token] = k[token] + k[token + delta]
            result = Float32(0)
            if z[0] > Float32(0):
                result = k[0] / cutlass.max(z[0], Float32(1e-20))
            offset = Int64(region) * Int64(256) + Int64(dim)
            unrounded[offset] = result
            output[offset] = Float8E4M3FN(BFloat16(result))


class _WeightedRegionScore:
    def __init__(self, groups: int, capacity: int):
        self.groups = groups
        self.capacity = capacity

    @cute.jit
    def __call__(self, query: cute.Pointer, weights: cute.Pointer,
                 summaries: cute.Pointer, requests: cute.Pointer,
                 positions: cute.Pointer, scores: cute.Pointer,
                 rows: Int32, limit: Int64, stream: cuda.CUstream):
        self.kernel(query, weights, summaries, requests, positions, scores, limit).launch(
            grid=((self.capacity + 63) // 64, rows, self.groups),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def packed_query(self, pointer: cute.Pointer, offset: Int64):
        # Round Q once per packed fragment, without widening the stored FP8 K.
        words = cute.recast_ptr((pointer + offset).align(4), dtype=Uint32)
        return cvt_bf16x2x2_to_e4m3x4(words[0], words[1])

    @cute.jit
    def packed_summary(self, pointer: cute.Pointer, offset: Int64):
        return cute.recast_ptr((pointer + offset).align(4), dtype=Uint32)[0].to(Uint32)

    @cute.kernel
    def kernel(self, query: cute.Pointer, weights: cute.Pointer,
               summaries: cute.Pointer, requests: cute.Pointer,
               positions: cute.Pointer, scores: cute.Pointer, limit: Int64):
        tile, row, group = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        matrix_row = lane // Int32(4)
        matrix_pair = lane % Int32(4)
        start = Int32(tile) * Int32(64) + warp * Int32(16)
        request = requests[row].to(Int64)
        visible = positions[row].to(Int64) // Int64(8)
        base = cute.make_rmem_tensor((2,), Int64)
        valid = cute.make_rmem_tensor((2,), Int32)
        for part in cutlass.range_constexpr(2):
            region = start + matrix_row + Int32(part * 8)
            valid[part] = Int32(0)
            base[part] = Int64(0)
            if (request >= Int64(0)) & (request < limit) & (region < self.capacity) & (region.to(Int64) < visible):
                valid[part] = Int32(1)
                base[part] = (request * Int64(self.capacity) + region.to(Int64)) * Int64(256)
        d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
        for tile_d in cutlass.range_constexpr(8):
            dim = Int64(tile_d * 32) + matrix_pair.to(Int64) * Int64(4)
            a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if valid[0] != 0:
                a0 = self.packed_summary(summaries, base[0] + dim)
                a2 = self.packed_summary(summaries, base[0] + dim + Int64(16))
            if valid[1] != 0:
                a1 = self.packed_summary(summaries, base[1] + dim)
                a3 = self.packed_summary(summaries, base[1] + dim + Int64(16))
            b0, b1 = Uint32(0), Uint32(0)
            if matrix_row < Int32(4):
                qbase = ((Int64(row) * Int64(self.groups) + Int64(group)) * Int64(4)
                         + matrix_row.to(Int64)) * Int64(256) + dim
                b0 = self.packed_query(query, qbase)
                b1 = self.packed_query(query, qbase + Int64(16))
            d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        for part in cutlass.range_constexpr(2):
            first = cutlass.max(d0 if part == 0 else d2, Float32(0))
            second = cutlass.max(d1 if part == 0 else d3, Float32(0))
            score = Float32(0)
            for head in cutlass.range_constexpr(4):
                value = first if head % 2 == 0 else second
                value = cute.arch.shuffle_sync(value, (lane // Int32(4)) * Int32(4) + Int32(head // 2))
                weight = weights[(Int64(row) * Int64(self.groups) + Int64(group)) * Int64(4) + Int64(head)]
                score = score + value * Float32(weight)
            if valid[part] == 0:
                score = Float32(-float("inf"))
            region = start + matrix_row + Int32(part * 8)
            if (matrix_pair == Int32(0)) & (region < self.capacity):
                offset = (Int64(row) * Int64(self.groups) + Int64(group)) * Int64(self.capacity) + region.to(Int64)
                scores[offset] = score


_TYPES = {torch.bfloat16: BFloat16, torch.float8_e4m3fn: Float8E4M3FN,
          torch.float32: Float32, torch.int32: Int32, torch.int64: Int64}


def _ptr(tensor):
    return make_ptr(_TYPES[tensor.dtype], tensor.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=tensor.element_size())


@lru_cache(maxsize=None)
def _compile(operation, groups, capacity, device):
    if operation == "compress":
        kernel = _CompressRegions()
        types = (BFloat16, BFloat16, Int32, Int32, Float8E4M3FN, Float32)
    else:
        kernel = _WeightedRegionScore(groups, capacity)
        types = (BFloat16, Float32, Float8E4M3FN, Int32, Int64, Float32)
    key = (operation, groups, capacity, device)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    fake = tuple(make_ptr(dtype, 16, cute.AddressSpace.gmem,
                          assumed_align=max(1, dtype.width // 8)) for dtype in types)
    with torch.cuda.device(device):
        return compile_cute(kernel, *fake, Int32(1), Int64(1), current_cuda_stream(),
                            compile_spec=KernelCompileSpec.from_key(
                                "attention.weighted_regions", 3, key))


def compress_regions(*, keys, logits, starts, counts, output, unrounded):
    """Compress staged complete regions; counts other than eight leave outputs intact."""
    if keys.dtype != torch.bfloat16 or logits.dtype != torch.bfloat16:
        raise ValueError("region keys and logits must be BF16")
    if keys.shape != logits.shape or keys.ndim != 2 or keys.shape[1] != 256:
        raise ValueError("region keys/logits must have matching [tokens,256] shape")
    if starts.ndim != 1 or starts.dtype != torch.int32 or counts.dtype != torch.int32 or starts.shape != counts.shape:
        raise ValueError("region spans must be matching I32 vectors")
    if output.dtype != torch.float8_e4m3fn or unrounded.dtype != torch.float32:
        raise ValueError("summary outputs must be E4M3 and F32")
    if output.shape != (starts.numel(), 256) or unrounded.shape != output.shape:
        raise ValueError("summary output shape must be [regions,256]")
    tensors = (keys, logits, starts, counts, output, unrounded)
    _validate_storage(tensors, keys.device)
    if starts.numel() == 0:
        return
    with torch.cuda.device(keys.device):
        raw = _compile("compress", 1, 8, keys.device.index)
        run_compiled(raw, (*map(_ptr, tensors), starts.numel(), keys.shape[0], current_cuda_stream()))


def score_regions(*, query, weights, summaries, requests, positions, output):
    """Signed weighted-ReLU scores, causal history only, no scale beyond caller W."""
    if query.dtype != torch.bfloat16 or query.ndim != 4 or query.shape[2:] != (4, 256):
        raise ValueError("index query must be BF16 [rows,groups,4,256]")
    rows, groups = query.shape[:2]
    if weights.dtype != torch.float32 or weights.shape != (rows, groups, 4):
        raise ValueError("index weights must be F32 [rows,groups,4]")
    if summaries.dtype != torch.float8_e4m3fn or summaries.ndim != 3 or summaries.shape[-1] != 256:
        raise ValueError("summaries must be E4M3 [requests,capacity,256]")
    capacity = summaries.shape[1]
    if capacity <= 0 or groups <= 0:
        raise ValueError("capacity and provider groups must be positive")
    if requests.dtype != torch.int32 or positions.dtype != torch.int64:
        raise ValueError("requests/positions must be I32/I64")
    if requests.shape != (rows,) or positions.shape != (rows,):
        raise ValueError("one request and position is required per query")
    if output.dtype != torch.float32 or output.shape != (rows, groups, capacity):
        raise ValueError("scores must be F32 [rows,groups,capacity]")
    tensors = (query, weights, summaries, requests, positions, output)
    _validate_storage(tensors, query.device)
    if rows == 0:
        return
    with torch.cuda.device(query.device):
        raw = _compile("score", groups, capacity, query.device.index)
        run_compiled(raw, (*map(_ptr, tensors), rows, summaries.shape[0], current_cuda_stream()))


def _validate_storage(tensors, device):
    if device.type != "cuda" or any(t.device != device or not t.is_contiguous() for t in tensors):
        raise ValueError("weighted-region tensors must be contiguous on one CUDA device")
