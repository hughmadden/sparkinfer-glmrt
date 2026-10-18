"""Native E4M3 / int4 dense linear on tensor cores: one K order from decode to prefill.

``output[rows, N] (F32) = x[rows, K] (BF16) @ W[N, K]^T``. Weights widen exactly
to BF16 in registers (E4M3 by ``cvt``, int4 ``q - 8`` by mantissa magic); no
dequantized matrix is materialized.

K is cut into ``segments`` fixed contiguous ranges, a function of ``(N, K)`` only
(more parallel chains for small-N projections). Each output element is one FP32
chain per segment, ascending over 64-wide K chunks; the segments then add in
ascending order. Inside a chunk, ``fp8_row`` gives quad lane ``j`` the logical
``k = 16 j + 4 t + i`` of m16n8k16 step ``t`` (one 16-byte weight run per lane).
Grouped encodings (``mxfp8``, ``int4_g32``) use a group-pure mapping,
``k = 32 h + 8 j + 4 t' + i`` for steps ``2h + t'``, so one K32 group is exactly
two MMA steps whose FP32 sum is then FMA'd into the chain with the group scale
(``fma(sum, scale, chain)``, ascending groups). No live row count, tile shape or
CTA count changes a chain: the decode route (weights streamed into registers,
reused by up to four 16-row tiles) and the prefill route (cp.async-staged
128-row tiles, each weight tile widened once per CTA tile) are bitwise one
function of ``(x row, weight column)``.

``fp8_row`` multiplies the finished sum by its FP32 per-output scale.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, cp_async4_shared_global_pred, cp_async_u32_shared_global,
    ld_global_nc_u32, ld_global_nc_v2_u32, ld_global_v2_u32, ld_shared_u32, ld_shared_v2_u32,
    ld_shared_v4_u32, shared_ptr_to_u32, st_global_v2_f32, u32_as_f32,
)
from b12x.gemm.bf16_gemv._skinny import _ld_cached, _ld_stream

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
    """Exact BF16 ``q - 8`` of the four low-first nibbles in ``word``'s low 16 bits.

    A nibble placed in a BF16 mantissa of 128.0 reads 128 + q (unit ulp); one
    bf16x2 subtract of 136 leaves q - 8 exactly."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 w, a, b, c; mov.b32 w, $2; mov.b32 c, 0x43084308; "
        "and.b32 a, w, 0xF; and.b32 b, w, 0xF0; shl.b32 b, b, 12; or.b32 a, a, b; "
        "or.b32 a, a, 0x43004300; sub.rn.bf16x2 $0, a, c; "
        "shr.b32 w, w, 8; and.b32 a, w, 0xF; and.b32 b, w, 0xF0; shl.b32 b, b, 12; or.b32 a, a, b; "
        "or.b32 a, a, 0x43004300; sub.rn.bf16x2 $1, a, c; }",
        "=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    """``a * b + c`` as one pinned fma.rn.f32."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
                  Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


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


@cute.jit
def _e8m0(byte: Uint32):
    value = u32_as_f32(byte << Uint32(23))
    if byte == Uint32(0):
        value = u32_as_f32(Uint32(0x00400000))
    if byte == Uint32(255):
        value = u32_as_f32(Uint32(0x7fffffff))
    return value


def decode_groups(n: int, encoding: str) -> int:
    """8-column groups one decode warp owns (the ABI geometry mirrors this choice):
    two for row-FP8 (16-byte runs per lane), one for grouped encodings (registers)."""
    return 2 if encoding == "fp8_row" else 1


class NativeTileLinear:
    """Tile core for ``fp8_row``, ``mxfp8`` and ``int4_g32`` (side 1 ``[N, K]`` or side 2
    ``[K/128, N, 128]`` payload and ``[K/128, N, 4]`` scales, as the ABI states)."""

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
        # Decode CTAs hold DECODE_WARPS warps whatever the segment count (latency hiding).
        self.slots = max(1, DECODE_WARPS // self.segments)
        self.d_threads = 32 * self.slots * self.segments
        self.stages = int(stages)
        self.bm = PREFILL_BM
        # Prefill warp tiles: row-FP8 64x32 (8 warps over 128x128); grouped encodings
        # 32x32 over 128x64 (8 warps), leaving registers for their group partials and
        # scales. Segments fold through the CTA's own F32 output tile (exact round trip).
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
        self.wpb = 8 * self.groups if not self.int4 else 4 * self.groups   # weight words / lane / 128 K
        self.spb = (2 if self.int4 else 1) * 2 * self.groups if self.grouped else 1

    # ------------------------------------------------------------------ launch
    @cute.jit
    def __call__(self, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
                 output: cute.Tensor, rows: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            if rows <= Int32(DECODE_ROWS):
                ctas = (self.n // self.cols + self.slots - 1) // self.slots
                self.decode(x, payload, scales, output, rows).launch(
                    grid=(min(Int32(ctas), cta_limit), 1, 1), block=(self.d_threads, 1, 1), stream=stream)
            else:
                tiles = ((rows + Int32(self.bm - 1)) // Int32(self.bm)) * Int32(self.n_tiles)
                self.prefill(x, payload, scales, output, rows).launch(
                    grid=(min(tiles, cta_limit), 1, 1), block=(self.prefill_threads, 1, 1),
                    stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------ addressing
    @cute.jit
    def _w_addr(self, payload: cute.Tensor, col: Int32, k: Int32):
        """Byte address of weight (col, k), k a multiple of 8."""
        if cutlass.const_expr(self.side2):
            index = ((k // Int32(128)).to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(128) \
                + (k % Int32(128)).to(Int64)
        else:
            index = col.to(Int64) * Int64(self.k) + k.to(Int64)
        if cutlass.const_expr(self.int4):
            index = index // Int64(2)
        return payload.iterator.toint() + index

    @cute.jit
    def _s_addr(self, scales: cute.Tensor, col: Int32, kb: Int32):
        """Byte address of the four K32 group scales of (col, 128-wide K block kb)."""
        if cutlass.const_expr(self.side2):
            index = (kb.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
        else:
            index = col.to(Int64) * Int64(self.k // 32) + kb.to(Int64) * Int64(4)
        return scales.iterator.toint() + index * Int64(2 if self.int4 else 1)

    @cute.jit
    def _scale(self, raw: Uint32, h: int):
        """Scale of group ``h`` (0..1) of a chunk's raw scale word (mxfp8: the
        chunk's two bytes at bits 0..15; int4: two F16)."""
        if cutlass.const_expr(self.int4):
            value = _f16_to_f32((raw >> Uint32(16 * h)) & Uint32(0xFFFF))
        else:
            value = _e8m0((raw >> Uint32(8 * h)) & Uint32(0xFF))
        return value

    @cute.jit
    def _store(self, output: cute.Tensor, scales: cute.Tensor, acc0: Float32, acc1: Float32,
               row: Int32, col: Int32, rows: Int32):
        """F32 pair at (row, col..col+1); col is even. fp8_row applies its row scale."""
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
        """One 64-wide K chunk of one (16-row tile, 8-column group) into ``acc[f:f+4]``.
        ``quad``: ``xa`` holds step t's A operands at 4t..4t+3 (else lo 0..7, hi 8..15)."""
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
                p0, p1, p2, p3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                for tp in cutlass.range_constexpr(2):
                    t = 2 * h + tp
                    a0, a1, a2, a3 = xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1]
                    if cutlass.const_expr(quad):
                        a0, a1, a2, a3 = xa[4 * t], xa[4 * t + 1], xa[4 * t + 2], xa[4 * t + 3]
                    p0, p1, p2, p3 = bf16_mma_m16n8k16_f32(
                        p0, p1, p2, p3, a0, a1, a2, a3,
                        bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                s0, s1 = sc[(gi * 2 + h) * 2], sc[(gi * 2 + h) * 2 + 1]
                acc[f] = _fma(p0, s0, acc[f])
                acc[f + 1] = _fma(p1, s1, acc[f + 1])
                acc[f + 2] = _fma(p2, s0, acc[f + 2])
                acc[f + 3] = _fma(p3, s1, acc[f + 3])

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
                     scales: cute.Tensor, n0: Int32, g: Int32, j: Int32, kb: Int32):
        """This lane's weights (and grouped scales) of one 128-wide K block."""
        for c in cutlass.range_constexpr(2):
            for gi in cutlass.range_constexpr(self.groups):
                col = n0 + Int32(8 * gi) + g
                k0 = kb * Int32(128) + Int32(64 * c)
                if cutlass.const_expr(not self.grouped):
                    words = _ld_stream(self._w_addr(payload, col, k0 + Int32(16) * j))
                    for t in cutlass.range_constexpr(4):
                        wdst[(c * self.groups + gi) * 4 + t] = words[t]
                elif cutlass.const_expr(self.int4):
                    for h in cutlass.range_constexpr(2):
                        wdst[(c * self.groups + gi) * 2 + h] = ld_global_nc_u32(
                            self._w_addr(payload, col, k0 + Int32(32 * h) + Int32(8) * j))
                else:
                    for h in cutlass.range_constexpr(2):
                        w0, w1 = ld_global_nc_v2_u32(
                            self._w_addr(payload, col, k0 + Int32(32 * h) + Int32(8) * j))
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
               output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, wid = tid & Int32(31), tid >> Int32(5)
        slot, seg = wid % Int32(self.slots), wid // Int32(self.slots)
        g, j = lane >> Int32(2), lane & Int32(3)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._decode_storage())
        fold = shared_ptr_to_u32(storage.fold.data_ptr())
        G = self.groups
        F = 4 * G * 4
        # Balanced contiguous column blocks per warp slot across the grid; the
        # segment warps of a slot share its blocks. Placement never changes arithmetic.
        blocks = Int64(self.n // self.cols)
        slots = Int64(grid) * Int64(self.slots)
        gw = Int64(cta) * Int64(self.slots) + slot.to(Int64)
        begin = (gw * blocks // slots).to(Int32)
        end = ((gw + Int64(1)) * blocks // slots).to(Int32)
        live = (rows + Int32(15)) >> Int32(4)
        kb0 = seg * Int32(self.seg_blocks)
        lanek = Int32(8) * j if cutlass.const_expr(self.grouped) else Int32(16) * j
        x_base = x.iterator.toint() + lanek.to(Int64) * Int64(2)
        span = end - begin
        if cutlass.const_expr(self.segments > 1):
            span = ((blocks + slots - Int64(1)) // slots).to(Int32)
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
                self._decode_load(cur, scur, payload, scales, n0, g, j, kb0)
                for b in cutlass.range(self.seg_blocks, unroll=1):
                    kb = kb0 + b
                    if b + Int32(1) < Int32(self.seg_blocks):
                        self._decode_load(nxt, snxt, payload, scales, n0, g, j, kb + Int32(1))
                    for c in cutlass.range_constexpr(2):
                        k0 = kb * Int32(128) + Int32(64 * c)
                        bw = cute.make_rmem_tensor((8 * G,), Uint32)
                        for gi in cutlass.range_constexpr(G):
                            for t in cutlass.range_constexpr(4):
                                if cutlass.const_expr(self.int4):
                                    b0, b1 = _int4x4_bf16(cur[(c * G + gi) * 2 + t // 2] >> Uint32(16 * (t % 2)))
                                else:
                                    b0, b1 = _e4m3x4_bf16(cur[(c * G + gi) * 4 + t])
                                bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1] = b0, b1
                        sc = cute.make_rmem_tensor((4 * G,), Float32)
                        if cutlass.const_expr(self.grouped):
                            for gi in cutlass.range_constexpr(G):
                                for e in cutlass.range_constexpr(2):
                                    if cutlass.const_expr(self.int4):
                                        raw = scur[(gi * 2 + e) * 2 + c]
                                    else:
                                        raw = scur[gi * 2 + e] >> Uint32(16 * c)
                                    for h in cutlass.range_constexpr(2):
                                        sc[(gi * 2 + h) * 2 + e] = self._scale(raw, h)
                        for mt in cutlass.range_constexpr(4):
                            if Int32(mt) < live:
                                xa = cute.make_rmem_tensor((16,), Uint32)
                                xa.fill(Uint32(0))
                                for hh in cutlass.range_constexpr(2):
                                    r = Int32(16 * mt + 8 * hh) + g
                                    if r < rows:
                                        at = x_base + (r.to(Int64) * Int64(self.k) + k0.to(Int64)) * Int64(2)
                                        lo = _ld_cached(at)
                                        hi = _ld_cached(at + Int64(64 if self.grouped else 16))
                                        for i in cutlass.range_constexpr(4):
                                            xa[8 * hh + i], xa[8 * hh + 4 + i] = lo[i], hi[i]
                                # Grouped: xa[0..3] row lo k 8j.. of group 0, [4..7] group 1;
                                # step t = 2h + t' reads xa[2t], xa[2t+1] (k 32h + 8j + 4t' ..).
                                for gi in cutlass.range_constexpr(G):
                                    self._mma_chunk(acc, 4 * (mt * G + gi), xa, bw, gi, sc)
                    for i in cutlass.range_constexpr(self.wpb):
                        cur[i] = nxt[i]
                    if cutlass.const_expr(self.grouped):
                        for i in cutlass.range_constexpr(self.spb):
                            scur[i] = snxt[i]
            # Segments add in ascending order through shared memory; segment 0 stores.
            if cutlass.const_expr(self.segments > 1):
                for mt in cutlass.range_constexpr(4):
                    if Int32(mt) < live:
                        if seg > Int32(0):
                            base = fold + (((seg - Int32(1)) * Int32(self.slots) + slot)
                                           * Int32(4 * G * 32)) * Int32(4)
                            for i in cutlass.range_constexpr(4 * G):
                                _st_shared_f32(base + Int32(i * 128) + lane * Int32(4), acc[4 * G * mt + i])
                        cute.arch.barrier(barrier_id=Int32(1) + slot, number_of_threads=32 * self.segments)
                        if seg == Int32(0):
                            for s in cutlass.range_constexpr(self.segments - 1):
                                base = fold + ((Int32(s) * Int32(self.slots) + slot) * Int32(4 * G * 32)) * Int32(4)
                                for i in cutlass.range_constexpr(4 * G):
                                    acc[4 * G * mt + i] = _add(acc[4 * G * mt + i], u32_as_f32(
                                        ld_shared_u32(base + Int32(i * 128) + lane * Int32(4))))
                        cute.arch.barrier(barrier_id=Int32(1) + slot, number_of_threads=32 * self.segments)
            if seg == Int32(0) and active:
                for mt in cutlass.range_constexpr(4):
                    if Int32(mt) < live:
                        for gi in cutlass.range_constexpr(G):
                            f = 4 * (mt * G + gi)
                            col = n0 + Int32(8 * gi) + Int32(2) * j
                            self._store(output, scales, acc[f], acc[f + 1], Int32(16 * mt) + g, col, rows)
                            self._store(output, scales, acc[f + 2], acc[f + 3], Int32(16 * mt + 8) + g, col, rows)

    # ------------------------------------------------------------------ prefill
    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.stages * self.stage_bytes], 128],
            "wide": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.bn * 128 if self.grouped else 16], 128],
            "fsc": cute.struct.Align[cute.struct.MemRange[Float32, max(1, self.bn * 2)], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def _widen_stage(self, sw: Int32, ss: Int32, wide: Int32, fsc: Int32, kc: Int32, tid: Int32):
        """Widen the staged W tile once per CTA into BF16 ``wide`` ([col][64 k], 16-byte
        units XOR-swizzled by col & 7) and its group scales into F32 ``fsc`` ([col][2])."""
        units = self.bn * 8                        # 16-byte BF16 units (8 k each)
        for i in cutlass.range_constexpr((units + self.prefill_threads - 1) // self.prefill_threads):
            u = Int32(i * self.prefill_threads) + tid
            if u < Int32(units):
                col, c = u >> Int32(3), u & Int32(7)
                if cutlass.const_expr(self.int4):
                    word = ld_shared_u32(sw + col * Int32(32) + c * Int32(4))
                    b0, b1 = _int4x4_bf16(word)
                    b2, b3 = _int4x4_bf16(word >> Uint32(16))
                else:
                    w0, w1 = ld_shared_v2_u32(sw + col * Int32(64) + c * Int32(8))
                    b0, b1 = _e4m3x4_bf16(w0)
                    b2, b3 = _e4m3x4_bf16(w1)
                _st_shared_v4(wide + col * Int32(128) + ((c ^ (col & Int32(7))) << Int32(4)), b0, b1, b2, b3)
        if cutlass.const_expr(self.grouped):
            if tid < Int32(self.bn):
                raw = ld_shared_u32(ss + tid * Int32(4))
                if cutlass.const_expr(not self.int4):
                    raw = raw >> ((kc % Int32(2)) * Int32(16)).to(Uint32)
                for h in cutlass.range_constexpr(2):
                    _st_shared_f32(fsc + (tid * Int32(2) + Int32(h)) * Int32(4), self._scale(raw, h))

    @cute.jit
    def _issue(self, base: Int32, x: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor,
               rows: Int32, m0: Int32, n0: Int32, kc: Int32, stage: Int32, tid: Int32):
        """cp.async one 64-wide K chunk of the A (swizzled), W and scale tiles into ``stage``."""
        sa = base + stage * Int32(self.stage_bytes)
        sw = sa + Int32(self.a_bytes)
        ss = sw + Int32(self.w_bytes)
        for i in cutlass.range_constexpr((self.a_bytes // 16 + self.prefill_threads - 1) // self.prefill_threads):
            chunk = Int32(i * self.prefill_threads) + tid
            if chunk < Int32(self.a_bytes // 16):
                r, c = chunk >> Int32(3), chunk & Int32(7)
                row = m0 + r
                src = x.iterator.toint() + (row.to(Int64) * Int64(self.k)
                                            + kc.to(Int64) * Int64(64) + (c * Int32(8)).to(Int64)) * Int64(2)
                pred = Int32(0)
                if row < rows:
                    pred = Int32(1)
                cp_async4_shared_global_pred(sa + r * Int32(128) + ((c ^ (r & Int32(7))) << Int32(4)), src, pred)
        per_col = self.wbytes // 16
        for i in cutlass.range_constexpr((self.w_bytes // 16 + self.prefill_threads - 1) // self.prefill_threads):
            chunk = Int32(i * self.prefill_threads) + tid
            if chunk < Int32(self.w_bytes // 16):
                r, c = chunk // Int32(per_col), chunk % Int32(per_col)
                src = self._w_addr(payload, n0 + r, kc * Int32(64) + c * Int32(32 if self.int4 else 16))
                pred = Int32(0)
                if n0 + r < Int32(self.n):
                    pred = Int32(1)
                cp_async4_shared_global_pred(sw + r * Int32(self.wbytes) + (c << Int32(4)), src, pred)
        if cutlass.const_expr(self.grouped):
            # One 4-byte word per column: mxfp8 the block's four group bytes, int4 the
            # chunk's two F16 (the half of the block's eight bytes).
            if tid < Int32(self.bn):
                if n0 + tid < Int32(self.n):
                    src = self._s_addr(scales, n0 + tid, kc // Int32(2))
                    if cutlass.const_expr(self.int4):
                        src = src + (kc % Int32(2)).to(Int64) * Int64(4)
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
                if cutlass.const_expr(self.grouped):
                    self._widen_stage(sw, ss, wide, fsc, kc, tid)
                    cute.arch.sync_threads()
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
