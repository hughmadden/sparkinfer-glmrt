"""SM89 (Ada) DeepSeek V4.1 W4A8 routed-expert projections.

Same contract as the SM120 slice kernels in ``w4a8_v41_slice`` (route
metadata, MXFP8 K32 activations, the N256/K128 lane-major resident weights
with V4.1 scale codes or raw UE8M0 scale words, ordered FP32 route output) for
GPUs without block-scaled MMA:

* each K32 step widens both operands exactly to F16 and runs two
  ``mma.m16n8k16.f32.f16.f16.f32`` into a zero accumulator, then promotes it
  into FP32 with ``fma(d, s_w * s_a, acc)``. Both scales are UE8M0, so their
  product is an exact power of two and each K32 step adds exactly one
  rounding. Ada's FP8 ``mma.m16n8k32.e4m3`` keeps too few bits when it sums a
  block's 32 products: on real V4.1 experts a quarter of the BF16 outputs left
  the official recipe, against 0.03% for the block-scaled SM121 path. It stays
  selectable as ``mma="fp8"`` (``B12X_SM89_V41_MMA=fp8``) for comparisons;
* E2M1 weights are relabelled to E4M3 bytes in registers (lossless) in the
  nibble-to-byte order the SM120 kernels give their QMMA E2M1 containers;
* weights are the MMA A operand and live rows the B operand ("swap AB"): one
  16-byte lane load of the resident layout is exactly the A fragments of two
  16-row M tiles for one K32 block, and a route group (at most 16 rows) is one
  or two 8-row N tiles;
* every warp streams its own rows in batches of ``batch`` K32 steps (whole
  K128 tiles): one loop iteration issues a batch's loads back to back, then
  runs its MMAs. Loads placed further ahead than their batch are sunk by ptxas
  toward their use anyway; batches keep several 512-byte requests in flight
  per warp, the CTAs on an SM overlap each other, and the rolled loop keeps
  the kernel small enough for the instruction cache.

FC1: one CTA per (32 intermediate columns, route group). Its eight warps
split the hidden K range into contiguous parts of whole K128 tiles, are folded
in ascending warp order, and warp 0 applies the V4.1 activation (BF16
rounding, gate <= 10, |up| <= 10, SiLU, route weight, BF16) and the MXFP8
requantization of that one 32-column block. Eight warps keep a lone expert's
36-72 CTAs busy enough to stream at the rate of many.

FC2: one CTA (four warps) per (64 hidden rows, route group). A 32-row chunk's
K tiles are split into two contiguous halves (the first one tile longer when
the tile count is odd); each warp accumulates its half in ascending K32 order
in one FFMA chain and the halves are folded in order.

The FC1->FC2 activation scratch is internal to this pair: E4M3 bytes
[routes][I], then UE8M0 bytes [routes][I/32]. It fits in the slice-plane
scratch that the SM120 recipe reserves for the same capacity.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint32, Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x._lib.intrinsics import (
    e2m1x8_to_e4m3x8,
    f16_mma_m16n8k16_f32,
    fmax_f32,
    fmin_f32,
    fabs_f32,
    fp8x4_e4m3_to_half2x2,
    get_ptr_as_int64,
    mma_m16n8k32_f32_e4m3,
    pow2_ceil_ue8m0,
    u32_as_f32,
    ue8m0_to_output_scale,
    cvt_f32_to_e4m3,
)

WARPS = 4
THREADS = 32 * WARPS
FLOAT8_E4M3_MAX = 448.0


def _asm(result, operands, text, constraints, *, side_effects, loc=None, ip=None):
    return llvm.inline_asm(
        result, operands, text, constraints,
        has_side_effects=side_effects, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_stream_v4(addr: Int64, *, loc=None, ip=None):
    """16 weight bytes on the read-only path without L1 allocation."""
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", side_effects=True, loc=loc, ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_nc_v2(addr: Int64, *, loc=None, ip=None):
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l", side_effects=True, loc=loc, ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def ld_nc_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "ld.global.nc.u32 $0, [$1];", "=r,l", side_effects=True, loc=loc, ip=ip))


@dsl_user_op
def ld_nc_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "ld.global.nc.u16 $0, [$1];", "=r,l", side_effects=True, loc=loc, ip=ip))


@dsl_user_op
def ld_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "ld.global.nc.u8 $0, [$1];", "=r,l", side_effects=True, loc=loc, ip=ip))


@dsl_user_op
def st_u8(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
         "st.global.u8 [$0], $1;", "l,r", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def ffma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """One-rounding a * b + c (the promotion step)."""
    return Float32(_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", side_effects=False, loc=loc, ip=ip,
    ))


@dsl_user_op
def fadd_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE round-to-nearest FP32 add without contraction (ordered folds)."""
    return Float32(_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", side_effects=False, loc=loc, ip=ip,
    ))


@dsl_user_op
def fmul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", side_effects=False, loc=loc, ip=ip,
    ))


@dsl_user_op
def shfl_idx(value: Uint32, source_lane: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(
        T.i32(),
        [Uint32(value).ir_value(loc=loc, ip=ip), Int32(source_lane).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", "=r,r,r", side_effects=True, loc=loc, ip=ip,
    ))


@dsl_user_op
def shfl_xor_f32(value: Float32, mask: int, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(
        T.f32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        f"shfl.sync.bfly.b32 $0, $1, {int(mask)}, 31, -1;", "=f,f", side_effects=True,
        loc=loc, ip=ip,
    ))


def spread_scale_code(code: Uint32, base: Uint32) -> Uint32:
    """Four K32 offsets of one V4.1 scale code plus the row base, one byte each."""
    word = ((code & Uint32(0x000F)) | ((code & Uint32(0x00F0)) << Uint32(4))
            | ((code & Uint32(0x0F00)) << Uint32(8)) | ((code & Uint32(0xF000)) << Uint32(12)))
    return word + base * Uint32(0x01010101)


def ue8m0_byte_value(word: Uint32, byte: int) -> Float32:
    """2^(b - 127) for UE8M0 byte ``byte`` of ``word`` (byte 0 reads as zero)."""
    if byte == 3:
        return u32_as_f32((word >> Uint32(24)) << Uint32(23))
    return u32_as_f32(((word >> Uint32(8 * byte)) & Uint32(255)) << Uint32(23))


def relabel_a(a_lo: Uint32, a_hi: Uint32):
    """A fragments of one 16x32 tile from the packed E2M1 words of rows g, g + 8.

    Each word holds eight consecutive K values of one row; the SM120 resident
    order pairs them with activation bytes 8c..8c+7, so a0/a2 (row g) and
    a1/a3 (row g + 8) come straight from the two relabelled words.
    """
    a0, a2 = e2m1x8_to_e4m3x8(a_lo)
    a1, a3 = e2m1x8_to_e4m3x8(a_hi)
    return a0, a1, a2, a3


def promoted_mma(acc, n, m, a, b0: Uint32, b1: Uint32,
                 s_lo: Float32, s_hi: Float32, sa0: Float32, sa1: Float32):
    """acc[n, m, :] += mma(A, B, 0) * s_row * s_col for one 16x8x32 tile.

    ``s_lo``/``s_hi`` are the weight scales of rows g and g + 8, ``sa0``/``sa1``
    the activation scales of columns 2c and 2c + 1.
    """
    zero = Float32(0)
    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(zero, zero, zero, zero, *a, b0, b1)
    acc[n, m, 0] = ffma(d0, fmul_rn(s_lo, sa0), acc[n, m, 0])
    acc[n, m, 1] = ffma(d1, fmul_rn(s_lo, sa1), acc[n, m, 1])
    acc[n, m, 2] = ffma(d2, fmul_rn(s_hi, sa0), acc[n, m, 2])
    acc[n, m, 3] = ffma(d3, fmul_rn(s_hi, sa1), acc[n, m, 3])


def promoted_mma_f16(acc, n, m, a, b0: Uint32, b1: Uint32,
                     s_lo: Float32, s_hi: Float32, sa0: Float32, sa1: Float32):
    """``promoted_mma`` as two exact F16 ``m16n8k16`` MMAs instead of one FP8 ``m16n8k32``.

    Ada's FP8 MMA sums a K32 block's 32 products with limited internal
    precision, which moves FC1 accumulators enough to flip the BF16 and MXFP8
    rounding of the intermediate. E4M3 and relabelled E2M1 values widen
    exactly to F16, the products are exact in FP32, and the F16 MMA accumulates
    them like the SM120/SM121 block-scaled path. The thread's eight physical K
    values 8c..8c+7 split into two k16 MMAs over the same registers: physical
    (8c, 8c+1) and (8c+2, 8c+3) are logical (2c, 2c+1) and (2c+8, 2c+9) of the
    first, (8c+4..8c+7) the same of the second; A and B use the same map.
    """
    a0l, a0h = fp8x4_e4m3_to_half2x2(a[0])
    a1l, a1h = fp8x4_e4m3_to_half2x2(a[1])
    a2l, a2h = fp8x4_e4m3_to_half2x2(a[2])
    a3l, a3h = fp8x4_e4m3_to_half2x2(a[3])
    b0l, b0h = fp8x4_e4m3_to_half2x2(b0)
    b1l, b1h = fp8x4_e4m3_to_half2x2(b1)
    zero = Float32(0)
    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(zero, zero, zero, zero, a0l, a1l, a0h, a1h, b0l, b0h)
    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(d0, d1, d2, d3, a2l, a3l, a2h, a3h, b1l, b1h)
    acc[n, m, 0] = ffma(d0, fmul_rn(s_lo, sa0), acc[n, m, 0])
    acc[n, m, 1] = ffma(d1, fmul_rn(s_lo, sa1), acc[n, m, 1])
    acc[n, m, 2] = ffma(d2, fmul_rn(s_hi, sa0), acc[n, m, 2])
    acc[n, m, 3] = ffma(d3, fmul_rn(s_hi, sa1), acc[n, m, 3])


MMA_MODES = ("fp8", "f16")


class _ResidentPlane:
    """Byte addressing of one projection in the N256/K128 lane-major layout.

    ``rows`` is the plane's resident row count (W13: 2 * I, W2: hidden) and
    ``k`` its reduction width; both are static model geometry.
    """

    def __init__(self, rows: int, k: int, scale_codes: bool):
        # SM89 roles have 256-row aligned planes (W13 2 * I, W2 hidden), so every
        # scale-entry row group holds a full 256 rows.
        assert rows % 256 == 0 and k % 128 == 0
        self.rows = rows
        self.k = k
        self.k_tiles = k // 128
        self.scale_codes = scale_codes
        # Payload bytes and scale bytes of one expert.
        self.payload_bytes = rows * k // 2
        if scale_codes:
            self.scale_bytes = rows * (2 * self.k_tiles + 1)
        else:
            self.scale_bytes = rows * k // 32

    def chunk(self, base: Int64, expert: Int64, row0: Int32, lane: Int32) -> Int64:
        """Address of this lane's 16 bytes of K32 block 0 of 32 rows at row0.

        K32 block b of the same rows is ``+ b * 4096``: a K128 tile holds four
        consecutive K32 blocks and tiles of one 256-row group are adjacent.
        """
        return (base + expert * Int64(self.payload_bytes)
                + Int64(row0 // Int32(256)) * Int64(self.k_tiles * 16384)
                + Int64((row0 % Int32(256)) // Int32(32)) * Int64(512)
                + Int64(lane) * Int64(16))

    def scale_row(self, base: Int64, expert: Int64, row: Int32) -> Int64:
        """Address of the scale entry (code u16 or raw u32) of row, K128 tile 0.

        Tile t of the same row is ``+ t * tile_stride``.
        """
        index = Int64(row // Int32(256)) * Int64(256 * self.k_tiles) + Int64(row % Int32(256))
        return base + expert * Int64(self.scale_bytes) + index * Int64(2 if self.scale_codes else 4)

    @property
    def tile_stride(self) -> int:
        return 256 * (2 if self.scale_codes else 4)

    def base_row(self, base: Int64, expert: Int64, row: Int32) -> Int64:
        """Address of a row's UE8M0 base byte (scale codes only)."""
        return (base + expert * Int64(self.scale_bytes)
                + Int64(self.rows * self.k_tiles * 2) + Int64(row))


def load_scale_entry(address: Int64, scale_codes: bool) -> Uint32:
    """One row's K128-tile scale entry: a code (u16) or a raw UE8M0x4 word."""
    if scale_codes:
        return ld_nc_u16(address)
    return ld_nc_u32(address)


def lane_row_scales(plane: _ResidentPlane, scale_base: Int64, expert: Int64,
                    row0: Int32, lane: Int32, g: Int32):
    """Per-lane scale sources for rows g, g+8, g+16, g+24 of a 32-row chunk.

    Each lane fetches the entry of row 8c + g; quads then exchange them. Returns
    (entry address for tile 0, base bytes of the four rows or None).
    """
    c = lane % Int32(4)
    own = row0 + c * Int32(8) + g
    address = plane.scale_row(scale_base, expert, own)
    bases = None
    if cutlass.const_expr(plane.scale_codes):
        mine = ld_nc_u8(plane.base_row(scale_base, expert, own))
        bases = [shfl_idx(mine, g * Int32(4) + Int32(q)) for q in range(4)]
    return address, bases


def tile_row_words(entry: Uint32, bases, scale_codes: bool, g: Int32):
    """Four UE8M0x4 words (one per K128 tile) for rows g + 8q of the chunk."""
    words = []
    for q in range(4):
        value = shfl_idx(entry, g * Int32(4) + Int32(q))
        if scale_codes:
            value = spread_scale_code(value, bases[q])
        words.append(value)
    return words


class V41Sm89Fc1Kernel:
    """FC1, V4.1 activation and MXFP8 requantization into internal scratch."""

    def __init__(self, capacity, *, topk=6, intermediate=2304, hidden=5120,
                 scale_codes=True, batch=4, warps=8, mma="f16"):
        assert capacity > 0 and topk > 0
        assert warps in (4, 8) and hidden % (128 * warps) == 0, "warps own whole K128 tiles"
        assert mma in MMA_MODES
        self.mma = mma
        self.warps = warps
        self.threads = 32 * warps
        assert intermediate % 128 == 0, "SM89 roles use 128-aligned resident widths"
        self.capacity = capacity
        self.topk = topk
        self.routes = capacity * topk
        self.hidden = hidden
        self.intermediate = intermediate
        self.blocks = intermediate // 32
        self.plane = _ResidentPlane(2 * intermediate, hidden, scale_codes)
        self.scale_codes = scale_codes
        self.warp_tiles = hidden // 128 // warps
        self.steps = 4 * self.warp_tiles
        # A batch covers whole K128 tiles (one scale entry per row and tile): the
        # largest multiple of four up to ``batch`` that divides the warp's steps.
        self.batch = max(b for b in range(4, max(batch, 4) + 1, 4) if self.steps % b == 0)
        self.batches = self.steps // self.batch

    @cute.jit
    def __call__(self, x: cute.Tensor, xs: cute.Tensor, w13: cute.Tensor, s13: cute.Tensor,
                 routing: cute.Tensor, partial: cute.Tensor, metadata: cute.Tensor,
                 groups: Int32, stream: cuda.CUstream):
        self.kernel(x, xs, w13, s13, routing, partial, metadata).launch(
            grid=(self.blocks, groups, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, xs: cute.Tensor, w13: cute.Tensor, s13: cute.Tensor,
               routing: cute.Tensor, partial: cute.Tensor, metadata: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        c = lane % 4
        block = cute.arch.block_idx()[0]
        group = cute.arch.block_idx()[1]
        rows = Int32(metadata[group, 1])
        if rows > 0:
            expert = Int64(metadata[group, 0])
            route_base = Int64(metadata[group, 2])
            smem = cutlass.utils.SmemAllocator()
            fold = smem.allocate_tensor(Float32, cute.make_layout((self.warps - 1) * 32 * 32),
                                        byte_alignment=16)
            two = rows > Int32(8)
            topk = Int32(self.topk)
            x_base = get_ptr_as_int64(x, 0)
            xs_base = get_ptr_as_int64(xs, 0)

            def token(row):
                safe = row if row < rows else Int32(0)
                return Int32(metadata[group, 3 + safe]) // topk

            def x_row(t):
                return x_base + Int64(cute.crd2idx((t, 0), x.layout)) * Int64(4)

            def xs_row(t):
                return xs_base + Int64(cute.crd2idx((t, 0), xs.layout))

            # B fragments: row g of each N tile; scale columns 2c and 2c + 1.
            b_rows = [x_row(token(g)) + Int64(c * Int32(8)),
                      x_row(token(g + Int32(8))) + Int64(c * Int32(8))]
            a_rows = [[xs_row(token(Int32(8 * n) + Int32(2) * c + Int32(i))) for i in range(2)]
                      for n in range(2)]

            plane = self.plane
            w_base = get_ptr_as_int64(w13, 0)
            s_base = get_ptr_as_int64(s13, 0)
            up_row0 = block * Int32(32)
            gate_row0 = Int32(self.intermediate) + up_row0
            up_chunk = plane.chunk(w_base, expert, up_row0, lane)
            gate_chunk = plane.chunk(w_base, expert, gate_row0, lane)
            up_entry, up_bases = lane_row_scales(plane, s_base, expert, up_row0, lane, g)
            gate_entry, gate_bases = lane_row_scales(plane, s_base, expert, gate_row0, lane, g)

            k_tile0 = warp * Int32(self.warp_tiles)
            k_block0 = k_tile0 * Int32(4)

            weights = cute.make_rmem_tensor((self.batch, 8), Uint32)
            acts = cute.make_rmem_tensor((self.batch, 2, 2), Uint32)
            entries = cute.make_rmem_tensor((self.batch // 4, 2), Uint32)
            act_scales = cute.make_rmem_tensor((self.batch // 4, 2, 2), Uint32)
            row_words = cute.make_rmem_tensor((8,), Uint32)
            col_words = cute.make_rmem_tensor((2, 2), Uint32)
            acc = cute.make_rmem_tensor((2, 4, 4), Float32)
            acc.fill(0)
            tile_stride = Int64(plane.tile_stride)
            tile_mma = promoted_mma_f16 if self.mma == "f16" else promoted_mma
            # One iteration per batch: all of its loads back to back, then its MMAs.
            for batch in range(self.batches):
                block0 = k_block0 + batch * Int32(self.batch)
                tile0 = Int64(block0 // Int32(4))
                for u in cutlass.range_constexpr(self.batch):
                    offset = Int64(block0 + Int32(u)) * Int64(4096)
                    up = ld_stream_v4(up_chunk + offset)
                    gate = ld_stream_v4(gate_chunk + offset)
                    for i in cutlass.range_constexpr(4):
                        weights[u, i] = up[i]
                        weights[u, 4 + i] = gate[i]
                    byte = Int64(block0 + Int32(u)) * Int64(32)
                    lo = ld_nc_v2(b_rows[0] + byte)
                    acts[u, 0, 0] = lo[0]
                    acts[u, 0, 1] = lo[1]
                    if two:
                        hi = ld_nc_v2(b_rows[1] + byte)
                        acts[u, 1, 0] = hi[0]
                        acts[u, 1, 1] = hi[1]
                for q in cutlass.range_constexpr(self.batch // 4):
                    tile = tile0 + Int64(q)
                    entries[q, 0] = load_scale_entry(up_entry + tile * tile_stride, self.scale_codes)
                    entries[q, 1] = load_scale_entry(gate_entry + tile * tile_stride, self.scale_codes)
                    for i in cutlass.range_constexpr(2):
                        act_scales[q, 0, i] = ld_nc_u32(a_rows[0][i] + tile * Int64(4))
                    if two:
                        for i in cutlass.range_constexpr(2):
                            act_scales[q, 1, i] = ld_nc_u32(a_rows[1][i] + tile * Int64(4))
                for u in cutlass.range_constexpr(self.batch):
                    kb = u % 4
                    if cutlass.const_expr(kb == 0):
                        up_words = tile_row_words(entries[u // 4, 0], up_bases, self.scale_codes, g)
                        gate_words = tile_row_words(entries[u // 4, 1], gate_bases, self.scale_codes, g)
                        for q in cutlass.range_constexpr(4):
                            row_words[q] = up_words[q]
                            row_words[4 + q] = gate_words[q]
                        for n in cutlass.range_constexpr(2):
                            for i in cutlass.range_constexpr(2):
                                col_words[n, i] = act_scales[u // 4, n, i]
                    # M tiles: up rows 0-15, up 16-31, gate 0-15, gate 16-31.
                    s_rows = [ue8m0_byte_value(row_words[q], kb) for q in range(8)]
                    sa = [ue8m0_byte_value(col_words[0, i], kb) for i in range(2)]
                    frags = [relabel_a(weights[u, 2 * m], weights[u, 2 * m + 1]) for m in range(4)]
                    for m in cutlass.range_constexpr(4):
                        tile_mma(acc, 0, m, frags[m], acts[u, 0, 0], acts[u, 0, 1],
                                 s_rows[2 * m], s_rows[2 * m + 1], sa[0], sa[1])
                    if two:
                        sb = [ue8m0_byte_value(col_words[1, i], kb) for i in range(2)]
                        for m in cutlass.range_constexpr(4):
                            tile_mma(acc, 1, m, frags[m], acts[u, 1, 0], acts[u, 1, 1],
                                     s_rows[2 * m], s_rows[2 * m + 1], sb[0], sb[1])

            # Ascending warp fold of the K parts.
            if warp > 0:
                for n in cutlass.range_constexpr(2):
                    for m in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(4):
                            fold[((warp - 1) * 32 + (n * 16 + m * 4 + e)) * 32 + lane] = acc[n, m, e]
            cute.arch.sync_threads()
            if warp == 0:
                for w in cutlass.range_constexpr(self.warps - 1):
                    for n in cutlass.range_constexpr(2):
                        for m in cutlass.range_constexpr(4):
                            for e in cutlass.range_constexpr(4):
                                acc[n, m, e] = fadd_rn(
                                    acc[n, m, e], fold[(w * 32 + (n * 16 + m * 4 + e)) * 32 + lane])
                partial_base = get_ptr_as_int64(partial, 0)
                scale_bytes = partial_base + Int64(self.routes * self.intermediate)
                for n in cutlass.range_constexpr(2):
                    # The second N tile is live only above eight rows (uniform branch).
                    if Int32(8 * n) < rows:
                        h = cute.make_rmem_tensor((2, 4), Float32)
                        for mm in cutlass.range_constexpr(2):
                            for e in cutlass.range_constexpr(4):
                                row = Int32(8 * n) + Int32(2) * c + Int32(e % 2)
                                gv = acc[n, 2 + mm, e].to(cutlass.BFloat16).to(Float32)
                                uv = acc[n, mm, e].to(cutlass.BFloat16).to(Float32)
                                gv = fmin_f32(gv, Float32(10))
                                uv = fmax_f32(Float32(-10), fmin_f32(uv, Float32(10)))
                                value = Float32(0)
                                if row < rows:
                                    sigmoid = cute.arch.rcp_approx(
                                        Float32(1) + cute.math.exp(-gv, fastmath=True))
                                    value = ((gv * sigmoid * uv * routing[route_base + Int64(row)])
                                             .to(cutlass.BFloat16).to(Float32))
                                h[mm, e] = value
                        for i in cutlass.range_constexpr(2):
                            amax = fmax_f32(fmax_f32(fabs_f32(h[0, i]), fabs_f32(h[0, 2 + i])),
                                            fmax_f32(fabs_f32(h[1, i]), fabs_f32(h[1, 2 + i])))
                            for mask in (4, 8, 16):
                                amax = fmax_f32(amax, shfl_xor_f32(amax, mask))
                            row = Int32(8 * n) + Int32(2) * c + Int32(i)
                            if row < rows:
                                _, byte = pow2_ceil_ue8m0(
                                    fmax_f32(amax, Float32(1e-4)) * Float32(1.0 / FLOAT8_E4M3_MAX))
                                inverse = ue8m0_to_output_scale(byte)
                                route = route_base + Int64(row)
                                out = (partial_base + route * Int64(self.intermediate)
                                       + Int64(block * Int32(32)) + Int64(g))
                                for mm in cutlass.range_constexpr(2):
                                    for half in cutlass.range_constexpr(2):
                                        st_u8(out + Int64(16 * mm + 8 * half),
                                              cvt_f32_to_e4m3(h[mm, 2 * half + i] * inverse))
                                if g == 0:
                                    st_u8(scale_bytes + route * Int64(self.intermediate // 32)
                                          + Int64(block), byte)


class V41Sm89Fc2Kernel:
    """FC2 into ordered FP32 route rows; invalid live routes are zeroed."""

    def __init__(self, capacity, *, topk=6, intermediate=2304, hidden=5120,
                 scale_codes=True, batch=4, k_split=2, mma="f16"):
        assert capacity > 0 and topk > 0
        assert hidden % 128 == 0 and intermediate % 128 == 0
        assert k_split in (1, 2, 4)
        assert mma in MMA_MODES
        self.mma = mma
        self.capacity = capacity
        self.topk = topk
        self.routes = capacity * topk
        self.hidden = hidden
        self.intermediate = intermediate
        # Warps own 32-row chunks. With k_split > 1 a chunk's K tiles are split into
        # that many contiguous parts (earlier parts one tile longer when uneven),
        # folded in ascending order.
        self.k_split = k_split
        self.chunks = WARPS // k_split
        self.rows_per_cta = 32 * self.chunks
        self.blocks = hidden // self.rows_per_cta
        self.plane = _ResidentPlane(hidden, intermediate, scale_codes)
        self.scale_codes = scale_codes
        tiles = intermediate // 128
        assert tiles >= k_split
        self.part_tiles = tiles // k_split
        self.long_parts = tiles % k_split
        spans = [4 * (self.part_tiles + (h < self.long_parts)) for h in range(k_split)]
        # A batch covers whole K128 tiles (one scale entry per row and tile): the
        # largest multiple of four up to ``batch`` that divides every warp's steps.
        self.batch = max(b for b in range(4, max(batch, 4) + 1, 4) if all(n % b == 0 for n in spans))

    @cute.jit
    def __call__(self, w2: cute.Tensor, s2: cute.Tensor, partial: cute.Tensor, out: cute.Tensor,
                 metadata: cute.Tensor, inverse: cute.Tensor, live_rows: cute.Tensor,
                 groups: Int32, stream: cuda.CUstream):
        self.kernel(w2, s2, partial, out, metadata, inverse, live_rows).launch(
            grid=(self.blocks, groups, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, w2: cute.Tensor, s2: cute.Tensor, partial: cute.Tensor, out: cute.Tensor,
               metadata: cute.Tensor, inverse: cute.Tensor, live_rows: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        c = lane % 4
        block = cute.arch.block_idx()[0]
        group = cute.arch.block_idx()[1]
        rows = Int32(metadata[group, 1])
        chunk_id = warp % Int32(self.chunks)
        half = warp // Int32(self.chunks)
        row0 = block * Int32(self.rows_per_cta) + chunk_id * Int32(32)
        if rows > 0:
            expert = Int64(metadata[group, 0])
            route_base = Int64(metadata[group, 2])
            two = rows > Int32(8)
            partial_base = get_ptr_as_int64(partial, 0)
            scale_base = partial_base + Int64(self.routes * self.intermediate)

            def grouped_route(row):
                return route_base + Int64(row if row < rows else Int32(0))

            b_rows = [partial_base + grouped_route(g + Int32(8 * n)) * Int64(self.intermediate)
                      + Int64(c * Int32(8)) for n in range(2)]
            a_rows = [[scale_base + grouped_route(Int32(8 * n) + Int32(2) * c + Int32(i))
                       * Int64(self.intermediate // 32) for i in range(2)] for n in range(2)]

            plane = self.plane
            chunk = plane.chunk(get_ptr_as_int64(w2, 0), expert, row0, lane)
            entry, bases = lane_row_scales(plane, get_ptr_as_int64(s2, 0), expert, row0, lane, g)

            weights = cute.make_rmem_tensor((self.batch, 4), Uint32)
            acts = cute.make_rmem_tensor((self.batch, 2, 2), Uint32)
            entries = cute.make_rmem_tensor((self.batch // 4,), Uint32)
            act_scales = cute.make_rmem_tensor((self.batch // 4, 2, 2), Uint32)
            row_words = cute.make_rmem_tensor((4,), Uint32)
            col_words = cute.make_rmem_tensor((2, 2), Uint32)
            acc = cute.make_rmem_tensor((2, 2, 4), Float32)
            acc.fill(0)
            tile_stride = Int64(plane.tile_stride)
            tile_mma = promoted_mma_f16 if self.mma == "f16" else promoted_mma
            # This warp's contiguous K tiles: part ``half`` of ``k_split``.
            longer = Int32(half < Int32(self.long_parts))
            first_tile = half * Int32(self.part_tiles) + cutlass.min(half, Int32(self.long_parts))
            batches = (Int32(self.part_tiles) + longer) * Int32(4) // Int32(self.batch)
            for batch in range(batches):
                tile_index = first_tile + batch * Int32(self.batch // 4)
                block0 = tile_index * Int32(4)
                tile0 = Int64(tile_index)
                for u in cutlass.range_constexpr(self.batch):
                    w = ld_stream_v4(chunk + Int64(block0 + Int32(u)) * Int64(4096))
                    for i in cutlass.range_constexpr(4):
                        weights[u, i] = w[i]
                    byte = Int64(block0 + Int32(u)) * Int64(32)
                    lo = ld_nc_v2(b_rows[0] + byte)
                    acts[u, 0, 0] = lo[0]
                    acts[u, 0, 1] = lo[1]
                    if two:
                        hi = ld_nc_v2(b_rows[1] + byte)
                        acts[u, 1, 0] = hi[0]
                        acts[u, 1, 1] = hi[1]
                for q in cutlass.range_constexpr(self.batch // 4):
                    tile = tile0 + Int64(q)
                    entries[q] = load_scale_entry(entry + tile * tile_stride, self.scale_codes)
                    for i in cutlass.range_constexpr(2):
                        act_scales[q, 0, i] = ld_nc_u32(a_rows[0][i] + tile * Int64(4))
                    if two:
                        for i in cutlass.range_constexpr(2):
                            act_scales[q, 1, i] = ld_nc_u32(a_rows[1][i] + tile * Int64(4))
                for u in cutlass.range_constexpr(self.batch):
                    kb = u % 4
                    if cutlass.const_expr(kb == 0):
                        words = tile_row_words(entries[u // 4], bases, self.scale_codes, g)
                        for q in cutlass.range_constexpr(4):
                            row_words[q] = words[q]
                        for n in cutlass.range_constexpr(2):
                            for i in cutlass.range_constexpr(2):
                                col_words[n, i] = act_scales[u // 4, n, i]
                    s_rows = [ue8m0_byte_value(row_words[q], kb) for q in range(4)]
                    sa = [ue8m0_byte_value(col_words[0, i], kb) for i in range(2)]
                    frags = [relabel_a(weights[u, 2 * m], weights[u, 2 * m + 1]) for m in range(2)]
                    for m in cutlass.range_constexpr(2):
                        tile_mma(acc, 0, m, frags[m], acts[u, 0, 0], acts[u, 0, 1],
                                 s_rows[2 * m], s_rows[2 * m + 1], sa[0], sa[1])
                    if two:
                        sb = [ue8m0_byte_value(col_words[1, i], kb) for i in range(2)]
                        for m in cutlass.range_constexpr(2):
                            tile_mma(acc, 1, m, frags[m], acts[u, 1, 0], acts[u, 1, 1],
                                     s_rows[2 * m], s_rows[2 * m + 1], sb[0], sb[1])

            if cutlass.const_expr(self.k_split > 1):
                # Later K parts into shared memory; the first part adds them in order.
                smem = cutlass.utils.SmemAllocator()
                fold = smem.allocate_tensor(
                    Float32, cute.make_layout((self.k_split - 1) * self.chunks * 16 * 32),
                    byte_alignment=16)
                if half > 0:
                    for n in cutlass.range_constexpr(2):
                        for m in cutlass.range_constexpr(2):
                            for e in cutlass.range_constexpr(4):
                                fold[(((half - 1) * Int32(self.chunks) + chunk_id) * 16
                                      + (n * 8 + m * 4 + e)) * 32 + lane] = acc[n, m, e]
                cute.arch.sync_threads()
                if half == 0:
                    for part in cutlass.range_constexpr(self.k_split - 1):
                        for n in cutlass.range_constexpr(2):
                            for m in cutlass.range_constexpr(2):
                                for e in cutlass.range_constexpr(4):
                                    acc[n, m, e] = fadd_rn(acc[n, m, e], fold[
                                        ((part * self.chunks + chunk_id) * 16
                                         + (n * 8 + m * 4 + e)) * 32 + lane])
            if half == 0:
                for n in cutlass.range_constexpr(2):
                    if Int32(8 * n) < rows:
                        for e in cutlass.range_constexpr(4):
                            row = Int32(8 * n) + Int32(2) * c + Int32(e % 2)
                            if row < rows:
                                route = Int64(metadata[group, 3 + row])
                                for m in cutlass.range_constexpr(2):
                                    out[route, row0 + Int32(16 * m + 8 * (e // 2)) + g] = acc[n, m, e]
        # Live routes whose expert id was invalid produce zero rows.
        live_routes = max(Int32(0), min(live_rows[0] * Int32(self.topk), Int32(self.routes)))
        col_begin = block * Int32(self.rows_per_cta)
        if tid < Int32(self.rows_per_cta):
            for r in range(group, live_routes, cute.arch.grid_dim()[1]):
                if inverse[r] < 0:
                    out[Int64(r), col_begin + tid] = Float32(0)
