"""MSA paged GQA pointer ABI: FP8 planar KV, D128, 64 Q / 4 KV heads.

Selections retain a fixed [4,max_rows,16] allocation. Cache pages contain 64
rows; each selected 128-token block addresses two logical pages. Descales are
per request and KV head. Output LSE is base 2, as in PagedForwardKernel.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr

from ._common import Operand, Scalar, compile_program


def _layout(max_rows, max_batch, mode):
    if mode not in ("decode", "extend") or min(max_rows, max_batch) <= 0:
        raise ValueError("positive capacities and decode/extend mode required")
    work = max_rows if mode == "decode" else (max_rows + 7) // 8 + max_batch
    sizes = dict(requests=work * 4, tiles=work * 4, zeros=max(work, max_batch + 1) * 4,
                 valid=work * 4, union_blocks=work * 4 * 128 * 4,
                 union_masks=work * 4 * 128 * 4, union_counts=work * 4 * 4)
    if mode == "decode":
        sizes.update(union_blocks=0, union_masks=0, union_counts=0)
    offsets, cursor = {}, 0
    for key, size in sizes.items():
        offsets[key] = cursor
        cursor += (size + 255) // 256 * 256
    return work, offsets, cursor


def minimax_attention_scratch_bytes(*, max_rows, max_batch, mode="decode"):
    return _layout(int(max_rows), int(max_batch), mode)[2]


class _Attention:
    def __init__(self, max_rows, max_batch, mode):
        from b12x.attention.paged.forward_extend_generic import PagedForwardKernel
        from b12x.attention.paged.traits import PagedForwardTraits

        self.rows, self.batch, self.mode = max_rows, max_batch, mode
        self.work, self.offsets, self.nbytes = _layout(max_rows, max_batch, mode)
        union = mode == "extend"
        self.tokens = 8 if union else 1
        # Static D128 tiles fit every supported SM120/SM121 device.
        traits = PagedForwardTraits(
            cta_tile_q=128 if union else 16, cta_tile_kv=32 if union else 64,
            num_mma_q=2 if union else 1, num_mma_kv=2 if union else 1,
            num_mma_d_qk=8, num_mma_d_vo=8, num_warps_q=4 if union else 1,
            num_warps_kv=1 if union else 4, num_threads=128,
            head_dim_qk=128, head_dim_vo=128, upcast_stride_q=16,
            upcast_stride_k=8, upcast_stride_v=8, upcast_stride_o=16,
            q_dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn, o_dtype=torch.bfloat16,
            q_smem_bytes=(128 if union else 16) * 256, shared_storage_bytes=65536,
            max_smem_per_sm=102400, num_ctas_per_sm=1, max_smem_per_threadblock=101376)
        self.forward = PagedForwardKernel(cutlass.BFloat16, cutlass.Float8E4M3FN,
            cutlass.Uint8, cutlass.BFloat16, traits=traits,
            msa_block_sparse=True, msa_union_tile=union, page_size=64)

    @cute.jit
    def _tensor(self, ptr, shape, strides):
        return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))

    @cute.jit
    def _scratch(self, scratch, offset: cutlass.Constexpr):
        return cute.recast_ptr(scratch + offset, dtype=Int32)

    @cute.jit
    def __call__(self, query: cute.Pointer, key_cache: cute.Pointer, value_cache: cute.Pointer,
                 k_descale: cute.Pointer, v_descale: cute.Pointer, page_table: cute.Pointer,
                 cache_lengths: cute.Pointer, cu_seqlens_q: cute.Pointer,
                 q2k_indices: cute.Pointer, output: cute.Pointer, lse: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, batch: Int32, table_stride: Int32,
                 pool_pages: Int64, stream: cuda.CUstream):
        work = rows if const_expr(self.mode == "decode") else (rows + 7) // 8 + batch
        requests = self._scratch(scratch, self.offsets["requests"])
        tiles = self._scratch(scratch, self.offsets["tiles"])
        zeros = self._scratch(scratch, self.offsets["zeros"])
        valid = self._scratch(scratch, self.offsets["valid"])
        blocks = self._scratch(scratch, self.offsets["union_blocks"])
        masks = self._scratch(scratch, self.offsets["union_masks"])
        counts = self._scratch(scratch, self.offsets["union_counts"])
        self.metadata(cu_seqlens_q, requests, tiles, zeros, valid, work, batch).launch(
            grid=((work + batch + 128) // 128, 1, 1), block=(128, 1, 1), stream=stream)
        union_blocks, union_masks, union_counts = None, None, None
        if const_expr(self.mode == "extend"):
            self.union(q2k_indices, cache_lengths, cu_seqlens_q, requests, tiles, valid,
                       blocks, masks, counts, work).launch(
                grid=(work, 4, 1), block=(32, 1, 1), stream=stream)
            union_blocks = self._tensor(blocks, (work, 4, 128), (512, 128, 1))
            union_masks = self._tensor(masks, (work, 4, 128), (512, 128, 1))
            union_counts = self._tensor(counts, (work, 4), (4, 1))
        k = self._tensor(key_cache, (pool_pages, 64, 4, 128), (Int64(32768), 512, 128, 1))
        v = self._tensor(value_cache, (pool_pages, 64, 4, 128), (Int64(32768), 512, 128, 1))
        dummy_float = self._tensor(cute.recast_ptr(zeros, dtype=cutlass.Float32), (1,), (1,))
        dummy_desc = self._tensor(cute.recast_ptr(zeros, dtype=Int64), (1,), (1,))
        self.forward(
            self._tensor(query, (rows, 64, 128), (8192, 128, 1)), k, v,
            self._tensor(page_table, (batch, table_stride), (Int64(table_stride), 1)),
            self._tensor(cache_lengths, (batch,), (1,)),
            self._tensor(cu_seqlens_q, (batch + 1,), (1,)),
            self._tensor(requests, (work,), (1,)), self._tensor(tiles, (work,), (1,)),
            self._tensor(zeros, (work,), (1,)), self._tensor(zeros, (batch + 1,), (1,)),
            self._tensor(zeros, (1,), (1,)), self._tensor(zeros, (batch,), (1,)),
            self._tensor(valid, (work,), (1,)),
            self._tensor(q2k_indices, (4, self.rows, 16), (self.rows * 16, 16, 1)),
            union_blocks, union_masks, union_counts, dummy_float, None,
            self._tensor(output, (rows, 64, 128), (8192, 128, 1)),
            self._tensor(lse, (rows, 64), (64, 1)),
            self._tensor(k_descale, (batch, 4), (4, 1)),
            self._tensor(v_descale, (batch, 4), (4, 1)), dummy_desc, dummy_desc, stream)

    @cute.kernel
    def metadata(self, cu, requests, tiles, zeros, valid, work: Int32, batch: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        idx = bx * 128 + tx
        if idx < work or idx <= batch:
            zeros[idx] = Int32(0)
        if idx < work:
            req, base = Int32(0), Int32(0)
            while req < batch:
                n = (cu[req + 1] - cu[req] + self.tokens - 1) // self.tokens
                if idx < base + n:
                    break
                base += n
                req += 1
            valid[idx] = Int32(req < batch)
            requests[idx] = req if req < batch else Int32(0)
            tiles[idx] = idx - base if req < batch else Int32(0)

    @cute.kernel
    def union(self, selections, lengths, cu, requests, tiles, valid,
              blocks, masks, counts, work: Int32):
        tx, _, _ = cute.arch.thread_idx()
        wi, head, _ = cute.arch.block_idx()
        if tx == 0:
            count = Int32(0)
            base = (Int64(wi) * 4 + head) * 128
            if valid[wi] != 0:
                req = requests[wi]
                start, end = cu[req], cu[req + 1]
                first = tiles[wi] * 8
                for token in cutlass.range_constexpr(8):
                    row = start + first + token
                    if row < end:
                        visible = lengths[req] - (end - start) + first + token + 1
                        for slot in cutlass.range_constexpr(16):
                            block = selections[(Int64(head) * self.rows + row) * 16 + slot]
                            if block >= 0 and Int64(block) * 128 < visible:
                                found = Int32(-1)
                                probe = Int32(0)
                                while probe < count:
                                    if blocks[base + probe] == block:
                                        found = probe
                                    probe += 1
                                if found < 0:
                                    insert = count
                                    while insert > 0:
                                        if blocks[base + insert - 1] <= block:
                                            break
                                        blocks[base + insert] = blocks[base + insert - 1]
                                        masks[base + insert] = masks[base + insert - 1]
                                        insert -= 1
                                    blocks[base + insert] = block
                                    masks[base + insert] = Int32(1 << token)
                                    count += 1
                                else:
                                    masks[base + found] = masks[base + found] | Int32(1 << token)
            counts[Int64(wi) * 4 + head] = count


def compile_minimax_attention_aot(*, max_rows, max_batch, mode="decode"):
    launch = _Attention(int(max_rows), int(max_batch), mode)
    operands = (
        Operand("query", torch.bfloat16, "[rows,64,128]"),
        Operand("key_cache", torch.uint8, "[pool_pages,64,4,128] E4M3 bytes"),
        Operand("value_cache", torch.uint8, "[pool_pages,64,4,128] E4M3 bytes"),
        Operand("k_descale", torch.float32, "[batch,4]"),
        Operand("v_descale", torch.float32, "[batch,4]"),
        Operand("page_table", torch.int32, "[batch,table_stride]", align=4),
        Operand("cache_lengths", torch.int32, "[batch] including appended queries", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1] packed query offsets; decode arange", align=4),
        Operand("q2k_indices", torch.int32, f"[4,{max_rows},16]", align=4),
        Operand("output", torch.bfloat16, "[rows,64,128]", "out"),
        Operand("lse", torch.float32, "[rows,64] base-2", "out"),
        Operand("scratch", torch.uint8, f"[{launch.nbytes}]", "scratch"))
    return compile_program(launch, name=f"minimax_attention_{mode}", operands=operands,
        scalars=(Scalar("rows"), Scalar("batch"), Scalar("table_stride"), Scalar("pool_pages", "int64")),
        key=(int(max_rows), int(max_batch), mode), scratch={"scratch": lambda rows: launch.nbytes},
        geometry=dict(max_rows=int(max_rows), max_batch=int(max_batch), page_rows=64,
                      query_heads=64, kv_heads=4, head_dim=128, block_tokens=128,
                      topk=16, mode=mode, softmax_scale=128 ** -0.5), doc=__doc__)
