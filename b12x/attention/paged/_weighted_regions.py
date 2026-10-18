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
    mma_m16n8k32_f32_e4m3, shared_ptr_to_u32, warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from ._selected_forward import _cp_async_load_128b_zfill, _pointer_as_int64


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
            grid=(rows, (self.capacity + 127) // 128, 1),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def packed_query(self, pointer: cute.Pointer, offset: Int64):
        words = cute.recast_ptr((pointer + offset).align(4), dtype=Uint32)
        return cvt_bf16x2x2_to_e4m3x4(words[0], words[1])

    @cute.jit
    def stage_summary(self, shared: cute.Pointer, summaries: cute.Pointer,
                      request: Int64, visible: Int64, start: Int32, thread: Int32):
        lane = thread % Int32(32)
        warp = thread // Int32(32)
        for step in cutlass.range_constexpr(8):
            local = warp * Int32(16) + Int32(step * 2) + lane // Int32(16)
            vector = lane % Int32(16)
            region = start + local
            valid = (region < self.capacity) & (region.to(Int64) < visible)
            base = (request * Int64(self.capacity) + region.to(Int64)) * Int64(256)
            base = cutlass.select_(valid, base, Int64(0))
            destination = local * Int32(64) + (vector ^ (local & Int32(7))) * Int32(4)
            _cp_async_load_128b_zfill(
                shared_ptr_to_u32(shared + destination),
                _pointer_as_int64(summaries, base + vector.to(Int64) * Int64(16)),
                cutlass.select_(valid, Int32(16), Int32(0)))
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(self, query: cute.Pointer, weights: cute.Pointer,
               summaries: cute.Pointer, requests: cute.Pointer,
               positions: cute.Pointer, scores: cute.Pointer, limit: Int64):
        row, tile, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        matrix_row = lane // Int32(4)
        matrix_pair = lane % Int32(4)
        start = Int32(tile) * Int32(128)
        request = requests[row].to(Int64)
        visible = positions[row].to(Int64) // Int64(8)
        if (request < Int64(0)) | (request >= limit) | (start.to(Int64) >= visible):
            for group in cutlass.range_constexpr(self.groups):
                region = start + Int32(thread)
                if region < self.capacity:
                    offset = (Int64(row) * Int64(self.groups) + Int64(group)) * Int64(self.capacity)
                    scores[offset + region.to(Int64)] = Float32(-float("inf"))
        else:
            allocator = cutlass.utils.SmemAllocator()
            staged = allocator.allocate_tensor(Uint32, cute.make_layout((8192,)), byte_alignment=128)
            rounded = allocator.allocate_tensor(
                Uint32, cute.make_layout((self.groups * 256,)), byte_alignment=16)
            result = allocator.allocate_tensor(
                Float32, cute.make_layout((self.groups * 64,)), byte_alignment=16)
            self.stage_summary(staged.iterator, summaries, request, visible, start, Int32(thread))
            # Convert every Q element once; all four warps retain B fragments.
            for chunk in cutlass.range_constexpr(self.groups * 2):
                word = Int32(thread) + Int32(chunk * 128)
                qbase = Int64(row) * Int64(self.groups * 1024) + word.to(Int64) * Int64(4)
                rounded[word] = self.packed_query(query, qbase)
            cute.arch.sync_threads()
            b = cute.make_rmem_tensor(((self.groups + 1) // 2, 8, 2), Uint32)
            w = cute.make_rmem_tensor(((self.groups + 1) // 2, 2), Float32)
            for pair in cutlass.range_constexpr((self.groups + 1) // 2):
                qhead = Int32(pair * 8) + matrix_row
                for tile_d in cutlass.range_constexpr(8):
                    for part in cutlass.range_constexpr(2):
                        b[pair, tile_d, part] = Uint32(0)
                        if qhead < Int32(self.groups * 4):
                            b[pair, tile_d, part] = rounded[
                                qhead * Int32(64) + Int32(tile_d * 8 + part * 4) + matrix_pair]
                for part in cutlass.range_constexpr(2):
                    head = Int32(pair * 8 + part) + matrix_pair * Int32(2)
                    w[pair, part] = Float32(0)
                    if head < Int32(self.groups * 4):
                        w[pair, part] = weights[Int64(row) * Int64(self.groups * 4) + head.to(Int64)]
            self.stage_summary(staged.iterator + 4096, summaries, request, visible,
                               start + Int32(64), Int32(thread))
            for stage in cutlass.range_constexpr(2):
                cute.arch.cp_async_wait_group(1 - stage)
                cute.arch.sync_threads()
                d = cute.make_rmem_tensor(((self.groups + 1) // 2, 4), Float32)
                d.fill(Float32(0))
                for tile_d in cutlass.range_constexpr(8):
                    local = warp * Int32(16) + matrix_row
                    v0 = Int32(tile_d * 2) ^ (matrix_row & Int32(7))
                    v1 = Int32(tile_d * 2 + 1) ^ (matrix_row & Int32(7))
                    base = Int32(stage * 4096) + local * Int32(64) + matrix_pair
                    a0 = staged[base + v0 * Int32(4)]
                    a1 = staged[base + Int32(8 * 64) + v0 * Int32(4)]
                    a2 = staged[base + v1 * Int32(4)]
                    a3 = staged[base + Int32(8 * 64) + v1 * Int32(4)]
                    for pair in cutlass.range_constexpr((self.groups + 1) // 2):
                        d[pair, 0], d[pair, 1], d[pair, 2], d[pair, 3] = mma_m16n8k32_f32_e4m3(
                            d[pair, 0], d[pair, 1], d[pair, 2], d[pair, 3],
                            a0, a1, a2, a3, b[pair, tile_d, 0], b[pair, tile_d, 1])
                for group in cutlass.range_constexpr(self.groups):
                    for part in cutlass.range_constexpr(2):
                        first = cutlass.max(d[group // 2, part * 2], Float32(0))
                        second = cutlass.max(d[group // 2, part * 2 + 1], Float32(0))
                        score = Float32(0)
                        # Preserve sequential FP32 multiply-adds, including signed zero.
                        for head in cutlass.range_constexpr(4):
                            source = matrix_row * Int32(4) + Int32((group % 2) * 2 + head // 2)
                            value = cute.arch.shuffle_sync(first if head % 2 == 0 else second, source)
                            weight = cute.arch.shuffle_sync(w[group // 2, head % 2], source)
                            score = score + value * weight
                        local = warp * Int32(16) + matrix_row + Int32(part * 8)
                        region = start + Int32(stage * 64) + local
                        if region.to(Int64) >= visible:
                            score = Float32(-float("inf"))
                        if matrix_pair == Int32(0):
                            result[Int32(group * 64) + local] = score
                cute.arch.sync_threads()
                for chunk in cutlass.range_constexpr((self.groups + 1) // 2):
                    index = Int32(chunk * 128) + Int32(thread)
                    group = index // Int32(64)
                    region = start + Int32(stage * 64) + index % Int32(64)
                    if (group < self.groups) & (region < self.capacity):
                        offset = (Int64(row) * Int64(self.groups) + group.to(Int64)) * Int64(self.capacity)
                        scores[offset + region.to(Int64)] = result[index]
                cute.arch.sync_threads()


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
                                "attention.weighted_regions", 4, key))


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
