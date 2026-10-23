"""Native dense routes with fixed per-output K32 accumulation and segment folds."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, cp_async4_shared_global_pred, cp_async_u32_shared_global,
    ld_global_nc_u32, ld_global_nc_v2_u32, ld_global_v2_u32, ld_shared_u32, ld_shared_v2_u32,
    ld_shared_v4_u32, shared_ptr_to_u32, st_global_v2_f32, u32_as_f32,
    ldmatrix_m8n8x4_b16, ldmatrix_m8n8x2_b16, ld_global_nc_u32_l2hint, ld_global_nc_v2_u32_l2hint,
    ld_global_nc_v4_u32_l2hint, weight_stream_policy,
)
from b12x.gemm.bf16_gemv._skinny import _ld_cached

DECODE_ROWS = 64           # decode route capacity (four 16-row tiles per warp)
PREFILL_BM = 128           # rows of one prefill CTA tile
PREFILL_STAGES = 4
DECODE_WARPS = 8           # warps of a decode CTA (slots x segments, segments <= 8)


@dsl_user_op
def _e4m3x4_bf16(word, *, loc=None, ip=None):
    """Exact BF16 of four packed E4M3 (byte 0 first) as two bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 l, h; mov.b32 {l, h}, $2; "
        "cvt.rn.bf16x2.e4m3x2 $0, l; cvt.rn.bf16x2.e4m3x2 $1, h; }",
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _int4x4_bf16(word, *, loc=None, ip=None):
    """Exact BF16 ``q - 8`` of the four low-first nibbles in ``word``'s low 16 bits."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 w, s, p0, p1, c; mov.b32 w, $2; shr.u32 s, w, 4; mov.b32 c, 0x43084308; "
        "prmt.b32 p0, w, s, 0x0400; lop3.b32 p0, p0, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $0, p0, c; "
        "prmt.b32 p1, w, s, 0x0501; lop3.b32 p1, p1, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $1, p1, c; }",
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _int4x8_bf16(word, *, loc=None, ip=None):
    """Exact BF16 ``q - 8`` of all eight low-first nibbles in ``word`` as four bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 w, s, p0, p1, p2, p3, c; mov.b32 w, $4; shr.u32 s, w, 4; mov.b32 c, 0x43084308; "
        "prmt.b32 p0, w, s, 0x0400; lop3.b32 p0, p0, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $0, p0, c; "
        "prmt.b32 p1, w, s, 0x0501; lop3.b32 p1, p1, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $1, p1, c; "
        "prmt.b32 p2, w, s, 0x0602; lop3.b32 p2, p2, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $2, p2, c; "
        "prmt.b32 p3, w, s, 0x0703; lop3.b32 p3, p3, 0x000f000f, 0x43004300, 0xea; sub.rn.bf16x2 $3, p3, c; }",
        "=r,=r,=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _quad_transpose_u8x4(raw, prmt_sel, *, loc=None, ip=None):
    """4x4 u8 transpose across the 4 lanes j = 0..3 of each quad."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(raw).ir_value(loc=loc, ip=ip), Uint32(prmt_sel).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 s0, s1, s2, s3, lo, hi; "
        "shfl.sync.idx.b32 s0, $1, 0, 0x1c1f, 0xffffffff; "
        "shfl.sync.idx.b32 s1, $1, 1, 0x1c1f, 0xffffffff; "
        "shfl.sync.idx.b32 s2, $1, 2, 0x1c1f, 0xffffffff; "
        "shfl.sync.idx.b32 s3, $1, 3, 0x1c1f, 0xffffffff; "
        "prmt.b32 lo, s0, s1, $2; "
        "prmt.b32 hi, s2, s3, $2; "
        "prmt.b32 $0, lo, hi, 0x6420; }",
        "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _quad_transpose_x4(a0, a1, a2, a3, j, *, loc=None, ip=None):
    """4x4 u32 matrix transpose across the 4 lanes j = 0..3 of each quad."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3)] + [Int32(j).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p1, p2; .reg .b32 s0, s1, t0, t1, b0, b1, b2, b3, s2, s3, t2, t3; "
        "and.b32 s0, $8, 1; setp.ne.u32 p1, s0, 0; "
        "and.b32 s1, $8, 2; setp.ne.u32 p2, s1, 0; "
        "selp.b32 s0, $4, $5, p1; selp.b32 s1, $6, $7, p1; "
        "shfl.sync.bfly.b32 t0, s0, 1, 0x1f, 0xffffffff; "
        "shfl.sync.bfly.b32 t1, s1, 1, 0x1f, 0xffffffff; "
        "selp.b32 b0, t0, $4, p1; selp.b32 b1, $5, t0, p1; "
        "selp.b32 b2, t1, $6, p1; selp.b32 b3, $7, t1, p1; "
        "selp.b32 s2, b0, b2, p2; selp.b32 s3, b1, b3, p2; "
        "shfl.sync.bfly.b32 t2, s2, 2, 0x1f, 0xffffffff; "
        "shfl.sync.bfly.b32 t3, s3, 2, 0x1f, 0xffffffff; "
        "selp.b32 $0, t2, b0, p2; selp.b32 $1, t3, b1, p2; "
        "selp.b32 $2, b2, t2, p2; selp.b32 $3, b3, t3, p2; }",
        "=r,=r,=r,=r,r,r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _ld_cached_pred(address: Int64, r: Int32, rows: Int32, *, loc=None, ip=None):
    """Predicated 16-byte global read-only load (returns 0 when r >= rows)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip),
         Int32(r).ir_value(loc=loc, ip=ip),
         Int32(rows).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.lt.s32 p, $5, $6; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _warp_uniform_i32(val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(val).ir_value(loc=loc, ip=ip)],
        "redux.sync.or.b32 $0, $1, 0xffffffff;", "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    """``a * b + c`` as one pinned fma.rn.f32."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
                  Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,0", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _mma_k32_scaled(acc0, acc1, acc2, acc3, a0, a1, a2, a3, a4, a5, a6, a7,
                    b0, b1, b2, b3, s0, s1, *, loc=None, ip=None):
    """One K32 group: two m16n8k16 steps summed in FP32, then one pinned fma.rn.f32 per output."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (s0, s1, acc0, acc1, acc2, acc3)],
        "{ .reg .f32 p0, p1, p2, p3; "
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{p0, p1, p2, p3}, {$4, $5, $6, $7}, {$12, $13}, {0f00000000, 0f00000000, 0f00000000, 0f00000000}; "
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{p0, p1, p2, p3}, {$8, $9, $10, $11}, {$14, $15}, {p0, p1, p2, p3}; "
        "fma.rn.f32 $0, p0, $16, $0; "
        "fma.rn.f32 $1, p1, $17, $1; "
        "fma.rn.f32 $2, p2, $16, $2; "
        "fma.rn.f32 $3, p3, $17, $3; }",
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r,f,f,0,1,2,3",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    """``a + b`` as one pinned add.rn.f32 (segment fold)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f16_to_f32(bits, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(bits).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $1; cvt.f32.f16 $0, h; }", "=f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _st_shared_v4(addr, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip)] + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
                    "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_f32(addr, value, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
                    "st.shared.f32 [$0], $1;", "r,f", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _e8m0(byte, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(byte).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p0, p1; .reg .b32 v; shl.b32 v, $1, 23; "
        "setp.eq.u32 p0, $1, 0; selp.b32 v, 0x00400000, v, p0; "
        "setp.eq.u32 p1, $1, 255; selp.b32 $0, 0x7fffffff, v, p1; }",
        "=f,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def decode_groups(n: int, encoding: str) -> int:
    return 2 if encoding == "fp8_row" else 1


class NativeTileLinear:
    def __init__(self, g, *, stages: int = PREFILL_STAGES):
        self.k, self.n = g["input_features"], g["output_features"]
        self.encoding, self.max_rows = g["encoding"], g["max_rows"]
        self.side2 = g["sharded_side"] == 2
        self.grouped = self.encoding != "fp8_row"
        self.int4 = self.encoding == "int4_g32"
        self.segments = int(g["segments"])
        self.groups = decode_groups(self.n, self.encoding)
        if self.segments < 1 or self.segments & (self.segments - 1):
            raise ValueError("tile segments must be a power of two")
        self.cols = 8 * self.groups
        self.slots = max(1, DECODE_WARPS // self.segments)
        self.d_threads = 32 * self.slots * self.segments
        self.stages = 2 if self.grouped else int(stages)
        self.bm = PREFILL_BM
        self.bn = 128 if (not self.grouped and self.n % 128 == 0) else 64
        self.wn = self.bn // 32
        self.wm = 2 if not self.grouped else 4
        self.mt = self.bm // self.wm // 16
        self.ng = 4
        if self.k % (128 * self.segments) or self.n % self.cols or self.n % 32:
            raise ValueError("tile core needs K % (128 * segments) == 0 and N % 32 == 0")
        self.seg_blocks = self.k // 128 // self.segments
        self.n_tiles = (self.n + self.bn - 1) // self.bn
        self.wbytes = 32 if self.int4 else 64        # payload bytes of one column x 64 K
        self.a_bytes = self.bm * 128                 # BF16 rows x 64 K
        self.w_bytes = self.bn * self.wbytes
        self.s_bytes = self.bn * 4 if self.grouped else 0
        self.stage_bytes = self.a_bytes + self.w_bytes + self.s_bytes
        self.prefill_threads = 32 * self.wm * self.wn
        self.wpb = 8 * self.groups if not self.int4 else 4 * self.groups
        self.spb = (2 if self.int4 else 1) * 2 * self.groups if self.grouped else 1
        self.decode_ctas = (self.n // self.cols + self.slots - 1) // self.slots
        # Rows of a decode tile are independent, so a shape with few CTAs splits its
        # 16-row chunks across more of them; wide shapes stay whole.
        self.decode_tiles = 4 if self.decode_ctas <= 32 else (2 if self.decode_ctas <= 96 else 1)

    # ------------------------------------------------------------------ launch
    @cute.jit
    def __call__(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                 output: cute.Tensor, rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            if rows <= Int32(DECODE_ROWS):
                if cutlass.const_expr(self.decode_tiles > 1):
                    row_tiles = max(Int32(1), min(min((rows + Int32(15)) >> Int32(4), Int32(self.decode_tiles)),
                                                  cta_limit // Int32(self.decode_ctas)))
                else:
                    row_tiles = Int32(1)
                chunks = (rows + Int32(15)) >> Int32(4)
                rows_per = ((chunks + row_tiles - Int32(1)) // row_tiles) * Int32(16)
                self.decode(x, payload, scales, output, rows, rows_per, row_tiles).launch(
                    grid=(min(Int32(self.decode_ctas) * row_tiles, cta_limit), 1, 1),
                    block=(self.d_threads, 1, 1),
                    stream=stream, min_blocks_per_mp=3 if self.side2 else 2)
            else:
                tiles = ((rows + Int32(self.bm - 1)) // Int32(self.bm)) * Int32(self.n_tiles)
                if cutlass.const_expr(self.grouped):
                    self.prefill(x, payload, scales, output, rows).launch(
                        grid=(min(tiles, cta_limit), 1, 1), block=(self.prefill_threads, 1, 1),
                        stream=stream, min_blocks_per_mp=2)
                else:
                    self.prefill_row(x, payload, scales, output, rows).launch(
                        grid=(min(tiles, cta_limit), 1, 1), block=(self.prefill_threads, 1, 1),
                        stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------ addressing
    @cute.jit
    def _w_addr(self, payload: cute.Tensor, col: Int32, k: Int32):
        k_off = (k >> Int32(1)) if self.int4 else k
        if cutlass.const_expr(self.side2):
            blk = 64 if self.int4 else 128
            index = ((k // Int32(128)).to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(blk) \
                + (k_off % Int32(blk)).to(Int64)
        else:
            index = col.to(Int64) * Int64(self.k // 2 if self.int4 else self.k) + k_off.to(Int64)
        return payload.iterator.toint() + index

    @cute.jit
    def _s_addr(self, scales: cute.Tensor, col: Int32, kb: Int32):
        if cutlass.const_expr(self.side2):
            index = (kb.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
        else:
            index = col.to(Int64) * Int64(self.k // 32) + kb.to(Int64) * Int64(4)
        return scales.iterator.toint() + index * Int64(2 if self.int4 else 1)

    @cute.jit
    def _scale(self, raw: Uint32, h):
        sh = (Int32(16 if self.int4 else 8) * Int32(h)).to(Uint32)
        if cutlass.const_expr(self.int4):
            value = _f16_to_f32((raw >> sh) & Uint32(0xFFFF))
        else:
            value = _e8m0((raw >> sh) & Uint32(0xFF))
        return value

    @cute.jit
    def _store(self, output: cute.Tensor, scales: cute.Tensor, acc0: Float32, acc1: Float32,
               row: Int32, col: Int32, rows: Int32):
        if row < rows and col < Int32(self.n):
            out = output.iterator.toint() + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
            if cutlass.const_expr(self.grouped):
                st_global_v2_f32(out, acc0, acc1)
            else:
                s0, s1 = ld_global_nc_v2_u32(scales.iterator.toint() + col.to(Int64) * Int64(4))
                st_global_v2_f32(out, acc0 * u32_as_f32(s0), acc1 * u32_as_f32(s1))

    @cute.jit
    def _mma_chunk(self, acc: cute.Tensor, f: int, xa: cute.Tensor, bw: cute.Tensor, gi: int,
                   sc: cute.Tensor, quad: cutlass.Constexpr = False):
        if cutlass.const_expr(not self.grouped):
            for t in cutlass.range_constexpr(4):
                a0, a1, a2, a3 = xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1]
                if cutlass.const_expr(quad):
                    a0, a1, a2, a3 = xa[4 * t], xa[4 * t + 1], xa[4 * t + 2], xa[4 * t + 3]
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3], a0, a1, a2, a3,
                    bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = d0, d1, d2, d3
        else:
            for h in cutlass.range_constexpr(2):
                t0 = 2 * h
                t1 = 2 * h + 1
                a0, a1, a2, a3 = xa[2 * t0], xa[8 + 2 * t0], xa[2 * t0 + 1], xa[8 + 2 * t0 + 1]
                a4, a5, a6, a7 = xa[2 * t1], xa[8 + 2 * t1], xa[2 * t1 + 1], xa[2 * t1 + 1]
                if cutlass.const_expr(quad):
                    a0, a1, a2, a3 = xa[4 * t0], xa[4 * t0 + 1], xa[4 * t0 + 2], xa[4 * t0 + 3]
                    a4, a5, a6, a7 = xa[4 * t1], xa[4 * t1 + 1], xa[4 * t1 + 2], xa[4 * t1 + 3]
                s0, s1 = sc[(gi * 2 + h) * 2], sc[(gi * 2 + h) * 2 + 1]
                acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = _mma_k32_scaled(
                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                    a0, a1, a2, a3, a4, a5, a6, a7,
                    bw[8 * gi + 2 * t0], bw[8 * gi + 2 * t0 + 1],
                    bw[8 * gi + 2 * t1], bw[8 * gi + 2 * t1 + 1],
                    s0, s1)

    # ------------------------------------------------------------------ decode
    def _decode_storage(self):
        class Storage:
            pass
        words = max(1, (self.segments - 1) * self.slots * 4 * self.groups * 32)
        Storage.__annotations__ = {
            "fold": cute.struct.Align[cute.struct.MemRange[Float32, words], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def _decode_load(self, wdst: cute.Tensor, sdst: cute.Tensor, payload: cute.Tensor,
                     scales: cute.Tensor, n0: Int32, g: Int32, j: Int32, kb: Int32, policy: Uint64):
        for c in cutlass.range_constexpr(2):
            for gi in cutlass.range_constexpr(self.groups):
                col = n0 + Int32(8 * gi) + g
                k0 = kb * Int32(128) + Int32(64 * c)
                if cutlass.const_expr(not self.grouped):
                    words = ld_global_nc_v4_u32_l2hint(
                        self._w_addr(payload, col, k0 + Int32(16) * j), policy, no_allocate=True)
                    for t in cutlass.range_constexpr(4):
                        wdst[(c * self.groups + gi) * 4 + t] = words[t]
                elif cutlass.const_expr(self.int4):
                    for h in cutlass.range_constexpr(2):
                        wdst[(c * self.groups + gi) * 2 + h] = ld_global_nc_u32_l2hint(
                            self._w_addr(payload, col, k0 + Int32(32 * h) + Int32(8) * j), policy)
                else:
                    for h in cutlass.range_constexpr(2):
                        w0, w1 = ld_global_nc_v2_u32_l2hint(
                            self._w_addr(payload, col, k0 + Int32(32 * h) + Int32(8) * j), policy)
                        wdst[(c * self.groups + gi) * 4 + 2 * h] = w0
                        wdst[(c * self.groups + gi) * 4 + 2 * h + 1] = w1
        if cutlass.const_expr(self.grouped):
            for gi in cutlass.range_constexpr(self.groups):
                for e in cutlass.range_constexpr(2):
                    col = n0 + Int32(8 * gi) + Int32(2) * j + Int32(e)
                    if cutlass.const_expr(self.int4):
                        s0, s1 = ld_global_nc_v2_u32(self._s_addr(scales, col, kb))
                        sdst[(gi * 2 + e) * 2], sdst[(gi * 2 + e) * 2 + 1] = s0, s1
                    else:
                        sdst[gi * 2 + e] = ld_global_nc_u32(self._s_addr(scales, col, kb))

    @cute.kernel
    def decode(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
               output: cute.Tensor, rows: Int32, rows_per: Int32, row_tiles: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        tid = Int32(tid)
        lane = tid & Int32(31)
        wid = _warp_uniform_i32(tid >> Int32(5)) if cutlass.const_expr(self.segments == 1) else (tid >> Int32(5))
        slot, seg = wid % Int32(self.slots), wid // Int32(self.slots)
        g, j = lane >> Int32(2), lane & Int32(3)
        if cutlass.const_expr(self.decode_tiles > 1):
            block_cta = Int32(cta) // row_tiles
            tile = Int32(cta) - block_cta * row_tiles
            row0 = tile * rows_per
            # Row tiles of a column block each stream its weights; the last evicts them.
            policy = weight_stream_policy((tile + Int32(1) == row_tiles).to(Int32))
        else:
            block_cta = Int32(cta)
            row0 = Int32(0)
            policy = weight_stream_policy(Int32(1))
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._decode_storage())
        fold = shared_ptr_to_u32(storage.fold.data_ptr())
        G = self.groups
        F = 4 * G * 4
        blocks = Int64(self.n // self.cols)
        # One group per (block CTA, slot); rows split across CTAs, never columns.
        if cutlass.const_expr(self.decode_tiles > 1):
            grid, _, _ = cute.arch.grid_dim()
            # A capped grid still owns all columns; row splitting never reduces
            # the column grid unless the budget is below its natural size.
            groups = min(Int64(grid), Int64(self.decode_ctas)) * Int64(self.slots)
            gw = block_cta.to(Int64) * Int64(self.slots) + slot.to(Int64)
        else:
            grid, _, _ = cute.arch.grid_dim()
            groups = Int64(grid) * Int64(self.slots)
            gw = Int64(cta) * Int64(self.slots) + slot.to(Int64)
        begin = (gw * blocks // groups).to(Int32)
        end = ((gw + Int64(1)) * blocks // groups).to(Int32)
        if cutlass.const_expr(self.decode_tiles > 1):
            live = (rows_per + Int32(15)) >> Int32(4)
        else:
            live = (rows + Int32(15)) >> Int32(4)
        kb0 = seg * Int32(self.seg_blocks)
        lanek = Int32(8) * j if cutlass.const_expr(self.grouped) else Int32(16) * j
        prmt_sel = j.to(Uint32) * Uint32(0x0101) + Uint32(0x0400)
        x_base = x.iterator.toint() + lanek.to(Int64) * Int64(2)
        span = end - begin
        if cutlass.const_expr(self.segments > 1):
            span = ((blocks + groups - Int64(1)) // groups).to(Int32)
        for it in cutlass.range(span, unroll=1):
            block = begin + it
            active = block < end
            n0 = block * Int32(self.cols)
            acc = cute.make_rmem_tensor((F,), Float32)
            acc.fill(Float32(0.0))
            if active:
                cur = cute.make_rmem_tensor((self.wpb,), Uint32)
                nxt = cute.make_rmem_tensor((self.wpb,), Uint32)
                scur = cute.make_rmem_tensor((self.spb,), Uint32)
                snxt = cute.make_rmem_tensor((self.spb,), Uint32)
                self._decode_load(cur, scur, payload, scales, n0, g, j, kb0, policy)
                for b in cutlass.range(self.seg_blocks, unroll=1):
                    kb = kb0 + b
                    if b + Int32(1) < Int32(self.seg_blocks):
                        self._decode_load(nxt, snxt, payload, scales, n0, g, j, kb + Int32(1), policy)
                    for c in cutlass.range_constexpr(2):
                        k0 = kb * Int32(128) + Int32(64 * c)
                        if cutlass.const_expr(self.grouped):
                            for h in cutlass.range_constexpr(2):
                                k32 = k0 + Int32(32 * h)
                                if cutlass.const_expr(not self.side2):
                                    at_m0 = x_base + ((row0 + g).to(Int64) * Int64(self.k) + k32.to(Int64)) * Int64(2)
                                    m0_lo0, m0_lo1, m0_lo2, m0_lo3 = _ld_cached_pred(at_m0, row0 + g, rows)
                                    m0_hi0, m0_hi1, m0_hi2, m0_hi3 = _ld_cached_pred(at_m0 + Int64(16 * self.k), row0 + g + Int32(8), rows)
                                if cutlass.const_expr(self.int4):
                                    t_w = _quad_transpose_u8x4(cur[c * 2 + h], prmt_sel)
                                    b0, b1, b2, b3 = _int4x8_bf16(t_w)
                                    r_s0, r_s1 = scur[c], scur[2 + c]
                                else:
                                    w0, w1 = _e4m3x4_bf16(cur[c * 4 + 2 * h])
                                    w2, w3 = _e4m3x4_bf16(cur[c * 4 + 2 * h + 1])
                                    b0, b1, b2, b3 = _quad_transpose_x4(w0, w1, w2, w3, j)
                                    r_s0 = scur[0] >> Uint32(16 * c)
                                    r_s1 = scur[1] >> Uint32(16 * c)
                                s0 = self._scale(r_s0, h)
                                s1 = self._scale(r_s1, h)
                                if cutlass.const_expr(not self.side2):
                                    m0_a0, m0_a2, m0_a4, m0_a6 = _quad_transpose_x4(m0_lo0, m0_lo1, m0_lo2, m0_lo3, j)
                                    m0_a1, m0_a3, m0_a5, m0_a7 = _quad_transpose_x4(m0_hi0, m0_hi1, m0_hi2, m0_hi3, j)
                                    acc[0], acc[1], acc[2], acc[3] = _mma_k32_scaled(
                                        acc[0], acc[1], acc[2], acc[3],
                                        m0_a0, m0_a1, m0_a2, m0_a3, m0_a4, m0_a5, m0_a6, m0_a7,
                                        b0, b1, b2, b3, s0, s1)
                                for mt in cutlass.range_constexpr(0 if self.side2 else 1, 4):
                                    if Int32(mt) < live:
                                        r0 = Int32(16 * mt) + g
                                        at0 = x_base + ((row0 + r0).to(Int64) * Int64(self.k) + k32.to(Int64)) * Int64(2)
                                        lo0, lo1, lo2, lo3 = _ld_cached_pred(at0, row0 + r0, rows)
                                        hi0, hi1, hi2, hi3 = _ld_cached_pred(at0 + Int64(16 * self.k), row0 + r0 + Int32(8), rows)
                                        a0, a2, a4, a6 = _quad_transpose_x4(lo0, lo1, lo2, lo3, j)
                                        a1, a3, a5, a7 = _quad_transpose_x4(hi0, hi1, hi2, hi3, j)
                                        f = 4 * mt
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = _mma_k32_scaled(
                                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                            a0, a1, a2, a3, a4, a5, a6, a7,
                                            b0, b1, b2, b3, s0, s1)
                        else:
                            bw = cute.make_rmem_tensor((8 * G,), Uint32)
                            sc = cute.make_rmem_tensor((4 * G,), Float32)
                            for gi in cutlass.range_constexpr(G):
                                for t in cutlass.range_constexpr(4):
                                    b0, b1 = _e4m3x4_bf16(cur[(c * G + gi) * 4 + t])
                                    bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                            for mt in cutlass.range_constexpr(4):
                                if Int32(mt) < live:
                                    xa = cute.make_rmem_tensor((16,), Uint32)
                                    xa.fill(Uint32(0))
                                    for hh in cutlass.range_constexpr(2):
                                        r = Int32(16 * mt + 8 * hh) + g
                                        if row0 + r < rows:
                                            at = x_base + ((row0 + r).to(Int64) * Int64(self.k) + k0.to(Int64)) * Int64(2)
                                            lo = _ld_cached(at)
                                            hi = _ld_cached(at + Int64(16))
                                            for i in cutlass.range_constexpr(4):
                                                xa[8 * hh + i], xa[8 * hh + 4 + i] = lo[i], hi[i]
                                    for gi in cutlass.range_constexpr(G):
                                        self._mma_chunk(acc, 4 * (mt * G + gi), xa, bw, gi, sc)
                    for i in cutlass.range_constexpr(self.wpb):
                        cur[i] = nxt[i]
                    if cutlass.const_expr(self.grouped):
                        for i in cutlass.range_constexpr(self.spb):
                            scur[i] = snxt[i]
            if cutlass.const_expr(self.segments > 1):
                for mt in cutlass.range_constexpr(4):
                    if Int32(mt) < live:
                        if seg > Int32(0):
                            base = fold + (((seg - Int32(1)) * Int32(self.slots) + slot)
                                           * Int32(4 * G * 32)) * Int32(4)
                            for i in cutlass.range_constexpr(4 * G):
                                _st_shared_f32(base + Int32(i * 128) + lane * Int32(4), acc[4 * G * mt + i])
                        cute.arch.sync_threads()
                        if seg == Int32(0):
                            for s in cutlass.range_constexpr(self.segments - 1):
                                base = fold + ((Int32(s) * Int32(self.slots) + slot) * Int32(4 * G * 32)) * Int32(4)
                                for i in cutlass.range_constexpr(4 * G):
                                    acc[4 * G * mt + i] = _add(acc[4 * G * mt + i], u32_as_f32(
                                        ld_shared_u32(base + Int32(i * 128) + lane * Int32(4))))
                        cute.arch.sync_threads()
            if seg == Int32(0) and active:
                for mt in cutlass.range_constexpr(4):
                    if Int32(mt) < live:
                        for gi in cutlass.range_constexpr(G):
                            f = 4 * (mt * G + gi)
                            col = n0 + Int32(8 * gi) + Int32(2) * j
                            self._store(output, scales, acc[f], acc[f + 1], row0 + Int32(16 * mt) + g, col, rows)
                            self._store(output, scales, acc[f + 2], acc[f + 3], row0 + Int32(16 * mt + 8) + g, col, rows)

    # ------------------------------------------------------------------ prefill
    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.stages * self.stage_bytes], 128],
            "wide": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.bn * 128 if self.grouped else 16], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def _widen_stage(self, sw: Int32, wide: Int32, tid: Int32):
        """Widen the staged W tile once per CTA into BF16 ``wide`` ([col][64 k], 16-byte
        units XOR-swizzled by col & 7)."""
        units = self.bn * 8                        # 16-byte BF16 units (8 k each)
        n_iter = (units + self.prefill_threads - 1) // self.prefill_threads
        col0 = tid >> Int32(3)
        c = tid & Int32(7)
        sw_off = col0 * Int32(32 if self.int4 else 64) + c * Int32(4 if self.int4 else 8)
        wide_off = col0 * Int32(128) + ((c ^ (col0 & Int32(7))) << Int32(4))
        if cutlass.const_expr(self.int4):
            words = cute.make_rmem_tensor((n_iter,), Uint32)
            for i in cutlass.range_constexpr(n_iter):
                words[i] = ld_shared_u32(sw + sw_off + Int32(i * 32 * 32))
            for i in cutlass.range_constexpr(n_iter):
                b0, b1, b2, b3 = _int4x8_bf16(words[i])
                _st_shared_v4(wide + wide_off + Int32(i * 32 * 128), b0, b1, b2, b3)
        else:
            w0s = cute.make_rmem_tensor((n_iter,), Uint32)
            w1s = cute.make_rmem_tensor((n_iter,), Uint32)
            for i in cutlass.range_constexpr(n_iter):
                w0s[i], w1s[i] = ld_shared_v2_u32(sw + sw_off + Int32(i * 32 * 64))
            for i in cutlass.range_constexpr(n_iter):
                b0, b1 = _e4m3x4_bf16(w0s[i])
                b2, b3 = _e4m3x4_bf16(w1s[i])
                _st_shared_v4(wide + wide_off + Int32(i * 32 * 128), b0, b1, b2, b3)

    @cute.jit
    def _issue(self, base: Int32, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
               rows: Int32, m0: Int32, n0: Int32, kc: Int32, stage: Int32, tid: Int32):
        """cp.async one 64-wide K chunk of the A (swizzled), W and scale tiles into ``stage``."""
        sa = base + stage * Int32(self.stage_bytes)
        sw = sa + Int32(self.a_bytes)
        ss = sw + Int32(self.w_bytes)
        r0 = tid >> Int32(3)
        c = tid & Int32(7)
        sa_off = sa + r0 * Int32(128) + ((c ^ (r0 & Int32(7))) << Int32(4))
        row0 = m0 + r0
        src0 = x.iterator.toint() + (row0.to(Int64) * Int64(self.k)
                                     + kc.to(Int64) * Int64(64) + (c * Int32(8)).to(Int64)) * Int64(2)
        r_step = self.prefill_threads // 8
        for i in cutlass.range_constexpr((self.a_bytes // 16 + self.prefill_threads - 1) // self.prefill_threads):
            pred = Int32(0)
            if row0 + Int32(i * r_step) < rows:
                pred = Int32(1)
            cp_async4_shared_global_pred(sa_off + Int32(i * r_step * 128),
                                         src0 + Int64(i * r_step * self.k * 2), pred)
        per_col = self.wbytes // 16
        w_chunks = self.w_bytes // 16
        for i in cutlass.range_constexpr((w_chunks + self.prefill_threads - 1) // self.prefill_threads):
            chunk = Int32(i * self.prefill_threads) + tid
            if chunk < Int32(w_chunks):
                r, cw = chunk // Int32(per_col), chunk % Int32(per_col)
                src = self._w_addr(payload, n0 + r, kc * Int32(64) + cw * Int32(32 if self.int4 else 16))
                if cutlass.const_expr(self.n % self.bn == 0):
                    cp_async4_shared_global_pred(sw + r * Int32(self.wbytes) + (cw << Int32(4)), src, Int32(1))
                else:
                    pred = Int32(0)
                    if n0 + r < Int32(self.n):
                        pred = Int32(1)
                    cp_async4_shared_global_pred(sw + r * Int32(self.wbytes) + (cw << Int32(4)), src, pred)
        if cutlass.const_expr(self.grouped):
            if tid < Int32(self.bn):
                if cutlass.const_expr(self.n % self.bn == 0):
                    src = self._s_addr(scales, n0 + tid, kc >> Int32(1))
                    if cutlass.const_expr(self.int4):
                        src = src + (kc & Int32(1)).to(Int64) * Int64(4)
                    cp_async_u32_shared_global(ss + tid * Int32(4), src)
                elif n0 + tid < Int32(self.n):
                    src = self._s_addr(scales, n0 + tid, kc >> Int32(1))
                    if cutlass.const_expr(self.int4):
                        src = src + (kc & Int32(1)).to(Int64) * Int64(4)
                    cp_async_u32_shared_global(ss + tid * Int32(4), src)

    @cute.kernel
    def prefill(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        g, j = lane >> Int32(2), lane & Int32(3)
        wm, wn = warp // Int32(self.wn), warp % Int32(self.wn)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        wide = shared_ptr_to_u32(storage.wide.data_ptr())
        m_tiles = (rows + Int32(self.bm - 1)) // Int32(self.bm)
        tiles = m_tiles * Int32(self.n_tiles)
        kcs = Int32(self.k // 64)
        seg_chunks = self.seg_blocks * 2
        MT, NG, S = self.mt, self.ng, self.stages
        l7 = lane & Int32(7)
        l15 = lane & Int32(15)
        l_sh3 = lane >> Int32(3)
        l_sh4 = lane >> Int32(4)
        s_lane_off = (wn * Int32(8 * NG) + Int32(2) * j) * Int32(4)
        b_lane_off = (wn * Int32(8 * NG) + l7) * Int32(128) + ((l_sh3 ^ l7) << Int32(4))
        a_lane_off = (wm * Int32(16 * MT) + l15) * Int32(128) + ((l_sh4 ^ l7) << Int32(4))
        while tile < tiles:
            if cutlass.const_expr(self.side2):
                m0 = (tile // Int32(self.n_tiles)) * Int32(self.bm)
                n0 = (tile % Int32(self.n_tiles)) * Int32(self.bn)
            else:
                m0 = (tile % m_tiles) * Int32(self.bm)
                n0 = (tile // m_tiles) * Int32(self.bn)
            acc = cute.make_rmem_tensor((MT * NG * 4,), Float32)
            acc.fill(Float32(0.0))
            for s in cutlass.range_constexpr(S - 1):
                if Int32(s) < kcs:
                    self._issue(base, x, payload, scales, rows, m0, n0, Int32(s), Int32(s), tid)
                cute.arch.cp_async_commit_group()
            for seg in cutlass.range(self.segments, unroll=1):
                for sc_it in cutlass.range(seg_chunks, unroll=1):
                    kc = seg * Int32(seg_chunks) + sc_it
                    cute.arch.cp_async_wait_group(S - 2)
                    cute.arch.sync_threads()
                    sa = base + (kc & Int32(S - 1)) * Int32(self.stage_bytes)
                    sw = sa + Int32(self.a_bytes)
                    ss = sw + Int32(self.w_bytes)
                    nk = kc + Int32(S - 1)
                    if nk < kcs:
                        self._issue(base, x, payload, scales, rows, m0, n0, nk, nk & Int32(S - 1), tid)
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(self.grouped):
                        self._widen_stage(sw, wide, tid)
                        cute.arch.sync_threads()
                        s_base = ss + s_lane_off
                        s_raw = cute.make_rmem_tensor((2 * NG,), Uint32)
                        for gi in cutlass.range_constexpr(NG):
                            r0, r1 = ld_shared_v2_u32(s_base + Int32(gi * 32))
                            if cutlass.const_expr(not self.int4):
                                sh = ((kc & Int32(1)) * Int32(16)).to(Uint32)
                                r0, r1 = r0 >> sh, r1 >> sh
                            s_raw[2 * gi], s_raw[2 * gi + 1] = r0, r1
                        for h in cutlass.range(2, unroll=1):
                            h_xor = h << Int32(6)
                            bw_h = cute.make_rmem_tensor((4 * NG,), Uint32)
                            sc_h = cute.make_rmem_tensor((2 * NG,), Float32)
                            b_off = wide + (b_lane_off ^ h_xor)
                            for gi in cutlass.range_constexpr(NG):
                                b = ldmatrix_m8n8x4_b16(b_off + Int32(gi * 1024))
                                for i in cutlass.range_constexpr(4):
                                    bw_h[4 * gi + i] = b[i]
                                sc_h[2 * gi] = self._scale(s_raw[2 * gi], h)
                                sc_h[2 * gi + 1] = self._scale(s_raw[2 * gi + 1], h)
                            a_off0 = sa + (a_lane_off ^ h_xor)
                            a_off1 = sa + (a_lane_off ^ (h_xor ^ Int32(32)))
                            for mt in cutlass.range_constexpr(MT):
                                xa_h = cute.make_rmem_tensor((8,), Uint32)
                                a0 = ldmatrix_m8n8x4_b16(a_off0 + Int32(mt * 2048))
                                a1 = ldmatrix_m8n8x4_b16(a_off1 + Int32(mt * 2048))
                                for i in cutlass.range_constexpr(4):
                                    xa_h[i] = a0[i]
                                    xa_h[4 + i] = a1[i]
                                for gi in cutlass.range_constexpr(NG):
                                    f = 4 * (mt * NG + gi)
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = _mma_k32_scaled(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        xa_h[0], xa_h[1], xa_h[2], xa_h[3],
                                        xa_h[4], xa_h[5], xa_h[6], xa_h[7],
                                        bw_h[4 * gi], bw_h[4 * gi + 1], bw_h[4 * gi + 2], bw_h[4 * gi + 3],
                                        sc_h[2 * gi], sc_h[2 * gi + 1])
                    else:
                        bw = cute.make_rmem_tensor((8 * NG,), Uint32)
                        sc = cute.make_rmem_tensor((4 * NG,), Float32)
                        for gi in cutlass.range_constexpr(NG):
                            col = wn * Int32(8 * NG) + Int32(8 * gi) + g
                            w = ld_shared_v4_u32(sw + col * Int32(64) + (j << Int32(4)))
                            for t in cutlass.range_constexpr(4):
                                b0, b1 = _e4m3x4_bf16(w[t])
                                bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                        for mt in cutlass.range_constexpr(MT):
                            r_lo = wm * Int32(16 * MT) + Int32(16 * mt) + g
                            xa = cute.make_rmem_tensor((16,), Uint32)
                            c_lo = Int32(2) * j
                            c_hi = Int32(2) * j + Int32(1)
                            at_lo = sa + r_lo * Int32(128)
                            at_hi = at_lo + Int32(8 * 128)
                            sw7 = r_lo & Int32(7)
                            for q in cutlass.range_constexpr(8):
                                unit = c_lo if q < 4 else c_hi
                                off = ((unit ^ sw7) << Int32(4)) + Int32(4 * (q % 4))
                                t, odd = q // 2, q % 2
                                xa[4 * t + 2 * odd] = ld_shared_u32(at_lo + off)
                                xa[4 * t + 2 * odd + 1] = ld_shared_u32(at_hi + off)
                            for gi in cutlass.range_constexpr(NG):
                                self._mma_chunk(acc, 4 * (mt * NG + gi), xa, bw, gi, sc, quad=True)
                if cutlass.const_expr(self.segments > 1):
                    first = seg == Int32(0)
                    last = seg == Int32(self.segments - 1)
                    row0 = m0 + wm * Int32(16 * MT) + g
                    col0 = n0 + wn * Int32(8 * NG) + Int32(2) * j
                    for mt in cutlass.range_constexpr(MT):
                        for hh in cutlass.range_constexpr(2):
                            r = row0 + Int32(16 * mt + 8 * hh)
                            if r < rows:
                                base_at = output.iterator.toint() + (r.to(Int64) * Int64(self.n) + col0.to(Int64)) * Int64(4)
                                for gi in cutlass.range_constexpr(NG):
                                    col = col0 + Int32(8 * gi)
                                    if col < Int32(self.n):
                                        at = base_at + Int64(gi * 32)
                                        f = 4 * (mt * NG + gi) + 2 * hh
                                        v0, v1 = acc[f], acc[f + 1]
                                        if not first:
                                            o0, o1 = ld_global_v2_u32(at)
                                            v0 = _add(u32_as_f32(o0), v0)
                                            v1 = _add(u32_as_f32(o1), v1)
                                        if cutlass.const_expr(not self.grouped):
                                            if last:
                                                s0, s1 = ld_global_nc_v2_u32(scales.iterator.toint() + col.to(Int64) * Int64(4))
                                                v0, v1 = v0 * u32_as_f32(s0), v1 * u32_as_f32(s1)
                                        st_global_v2_f32(at, v0, v1)
                    acc.fill(Float32(0.0))
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.segments == 1):
                for mt in cutlass.range_constexpr(MT):
                    for gi in cutlass.range_constexpr(NG):
                        f = 4 * (mt * NG + gi)
                        row = m0 + wm * Int32(16 * MT) + Int32(16 * mt) + g
                        col = n0 + wn * Int32(8 * NG) + Int32(8 * gi) + Int32(2) * j
                        self._store(output, scales, acc[f], acc[f + 1], row, col, rows)
                        self._store(output, scales, acc[f + 2], acc[f + 3], row + Int32(8), col, rows)
            tile += grid

    def _storage_row(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.stages * self.stage_bytes], 128],
            "wide": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 16], 128],
            "fsc": cute.struct.Align[cute.struct.MemRange[Float32, self.bn * 2], 16],
        }
        return cute.struct(Storage)

    @cute.kernel
    def prefill_row(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        g, j = lane >> Int32(2), lane & Int32(3)
        wm, wn = warp // Int32(self.wn), warp % Int32(self.wn)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage_row())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        wide = shared_ptr_to_u32(storage.wide.data_ptr())
        fsc = shared_ptr_to_u32(storage.fsc.data_ptr())
        m_tiles = (rows + Int32(self.bm - 1)) // Int32(self.bm)
        tiles = m_tiles * Int32(self.n_tiles)
        kcs = Int32(self.k // 64)
        seg_chunks = self.seg_blocks * 2
        MT, NG, S = self.mt, self.ng, self.stages
        while tile < tiles:
            m0 = (tile % m_tiles) * Int32(self.bm)
            n0 = (tile // m_tiles) * Int32(self.bn)
            acc = cute.make_rmem_tensor((MT * NG * 4,), Float32)
            acc.fill(Float32(0.0))
            for s in cutlass.range_constexpr(S - 1):
                if Int32(s) < kcs:
                    self._issue(base, x, payload, scales, rows, m0, n0, Int32(s), Int32(s), tid)
                cute.arch.cp_async_commit_group()
            for kc in cutlass.range(self.k // 64, unroll=1):
                cute.arch.cp_async_wait_group(S - 2)
                # Every warp is past the previous chunk's reads of ``wide`` (and of the
                # stage the next cp.async overwrites) once all threads arrive here.
                cute.arch.sync_threads()
                sa = base + (kc % Int32(S)) * Int32(self.stage_bytes)
                sw = sa + Int32(self.a_bytes)
                ss = sw + Int32(self.w_bytes)
                nk = kc + Int32(S - 1)
                if nk < kcs:
                    self._issue(base, x, payload, scales, rows, m0, n0, nk, nk % Int32(S), tid)
                cute.arch.cp_async_commit_group()
                bw = cute.make_rmem_tensor((8 * NG,), Uint32)
                sc = cute.make_rmem_tensor((4 * NG,), Float32)
                for gi in cutlass.range_constexpr(NG):
                    col = wn * Int32(8 * NG) + Int32(8 * gi) + g
                    sw7 = col & Int32(7)
                    if cutlass.const_expr(not self.grouped):
                        # k = 16 j + 4 t + i straight from the staged E4M3 run.
                        w = ld_shared_v4_u32(sw + col * Int32(64) + (j << Int32(4)))
                        for t in cutlass.range_constexpr(4):
                            b0, b1 = _e4m3x4_bf16(w[t])
                            bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                    else:
                        # k = 32 h + 8 j + 4 t' + i: BF16 units 4h + j.
                        for h in cutlass.range_constexpr(2):
                            w = ld_shared_v4_u32(wide + col * Int32(128) + (((Int32(4 * h) + j) ^ sw7) << Int32(4)))
                            for i in cutlass.range_constexpr(4):
                                bw[8 * gi + 4 * h + i] = w[i]
                        for e in cutlass.range_constexpr(2):
                            c0 = wn * Int32(8 * NG) + Int32(8 * gi) + Int32(2) * j + Int32(e)
                            for h in cutlass.range_constexpr(2):
                                sc[(gi * 2 + h) * 2 + e] = u32_as_f32(ld_shared_u32(fsc + (c0 * Int32(2) + Int32(h)) * Int32(4)))
                for mt in cutlass.range_constexpr(MT):
                    r_lo = wm * Int32(16 * MT) + Int32(16 * mt) + g
                    # A in MMA register order: per step t the quad (lo 2t, hi 2t, lo 2t+1,
                    # hi 2t+1) loads word by word, so no register shuffles precede the MMA.
                    xa = cute.make_rmem_tensor((16,), Uint32)
                    c_lo = j if cutlass.const_expr(self.grouped) else Int32(2) * j
                    c_hi = j + Int32(4) if cutlass.const_expr(self.grouped) else Int32(2) * j + Int32(1)
                    at_lo = sa + r_lo * Int32(128)
                    at_hi = at_lo + Int32(8 * 128)
                    sw7 = r_lo & Int32(7)
                    for q in cutlass.range_constexpr(8):
                        unit = c_lo if q < 4 else c_hi
                        off = ((unit ^ sw7) << Int32(4)) + Int32(4 * (q % 4))
                        t, odd = q // 2, q % 2
                        xa[4 * t + 2 * odd] = ld_shared_u32(at_lo + off)
                        xa[4 * t + 2 * odd + 1] = ld_shared_u32(at_hi + off)
                    for gi in cutlass.range_constexpr(NG):
                        self._mma_chunk(acc, 4 * (mt * NG + gi), xa, bw, gi, sc, quad=True)
                if cutlass.const_expr(self.segments > 1):
                    # Segment end: chain = (segments so far) + this segment, in ascending
                    # order through this CTA's own output tile (no other CTA touches it).
                    if (kc + Int32(1)) % Int32(seg_chunks) == Int32(0):
                        first = kc + Int32(1) == Int32(seg_chunks)
                        last = kc + Int32(1) == kcs
                        for mt in cutlass.range_constexpr(MT):
                            for gi in cutlass.range_constexpr(NG):
                                f = 4 * (mt * NG + gi)
                                row = m0 + wm * Int32(16 * MT) + Int32(16 * mt) + g
                                col = n0 + wn * Int32(8 * NG) + Int32(8 * gi) + Int32(2) * j
                                for hh in cutlass.range_constexpr(2):
                                    r = row + Int32(8 * hh)
                                    if r < rows and col < Int32(self.n):
                                        at = output.iterator.toint() + (r.to(Int64) * Int64(self.n)
                                                                        + col.to(Int64)) * Int64(4)
                                        v0, v1 = acc[f + 2 * hh], acc[f + 2 * hh + 1]
                                        if not first:
                                            o0, o1 = ld_global_v2_u32(at)
                                            v0 = _add(u32_as_f32(o0), v0)
                                            v1 = _add(u32_as_f32(o1), v1)
                                        if last:
                                            self._store(output, scales, v0, v1, r, col, rows)
                                        else:
                                            st_global_v2_f32(at, v0, v1)
                        acc.fill(Float32(0.0))
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.segments == 1):
                for mt in cutlass.range_constexpr(MT):
                    for gi in cutlass.range_constexpr(NG):
                        f = 4 * (mt * NG + gi)
                        row = m0 + wm * Int32(16 * MT) + Int32(16 * mt) + g
                        col = n0 + wn * Int32(8 * NG) + Int32(8 * gi) + Int32(2) * j
                        self._store(output, scales, acc[f], acc[f + 1], row, col, rows)
                        self._store(output, scales, acc[f + 2], acc[f + 3], row + Int32(8), col, rows)
            tile += grid
