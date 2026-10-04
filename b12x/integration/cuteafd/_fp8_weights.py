"""FP8 E4M3 weights with FP32 128x128 block scales for the GLM programs.

GLM checkpoints store dense weights as E4M3 ``[N, K]`` with one FP32 scale
per 128x128 block (``[ceil(N/128), K/128]``). Re-quantizing those scales to
powers of two costs 0.025 nats, so these kernels apply them exactly:

``MmaFp8Gemv``   decode rows (<= 16 in the programs): m16n8k16 BF16 MMAs fed
    straight from global memory, weights widened in registers to
    ``bf16(w * s)`` (exactly the BF16 program's dequantized weight), FP32
    accumulation; reads half the weight bytes of the BF16 GEMV.
``TmaFp8Gemm``    the BF16 TMA tensor-core GEMM with E4M3 weight tiles widened
    to ``bf16(w * s)`` in shared memory; bitwise equal to the BF16 kernel over
    dequantized weights (W8A16; block, per-row or K-block-major per-row scales).
``RoutedFp8Projection``  ``MmaFp8Gemv`` up to 16 live rows, the BF16 TMA GEMM
    over the BF16 copy of the weight above.
``Fp8Projection``  a weight held only as E4M3 + scales (no BF16 copy): decode
    programs run ``MmaFp8Gemv`` then ``TmaFp8Gemm``; prefill programs W8A8
    (``BlockFp8Projection``: E4M3 activations per row and 128-K block, the
    official FP8 releases' served numerics) or ``TmaFp8Gemm`` on a launch switch.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import bf16_mma_m16n8k16_f32, ld_shared_v2_u32, shared_ptr_to_u32, st_shared_v4_u32

from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection, _ld_cached, _ld_stream

__all__ = ["FP8_GEMV_ROWS", "Fp8Projection", "decode_tile_n", "MmaFp8Gemv", "RoutedFp8Projection", "TmaFp8Gemm", "check_w8_mode",
           "fp8_only_operands", "fp8_operands", "fp8_scale_shape", "gemv_warps", "projection", "w8_scalars"]


def fp8_scale_shape(n: int, k: int) -> tuple[int, int]:
    """FP32 block-scale grid of an ``[n, k]`` E4M3 weight."""
    return (int(n) + 127) // 128, (int(k) + 127) // 128


@dsl_user_op
def _ld_f32(address, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


class RoutedFp8Projection:
    """``out = x @ w^T`` for decode rows: ``MmaFp8Gemv`` over the E4M3 weight
    and its FP32 block scales up to ``max_fp8_rows`` live rows, the BF16 TMA
    GEMM over ``w_bf16`` (the same weight dequantized) above; the branch is
    on the ``rows`` scalar."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, max_fp8_rows: int = 16):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.max_fp8_rows = int(max_fp8_rows)
        self.fp8 = MmaFp8Gemv(self.n, self.k, max_rows=self.max_fp8_rows, warps=gemv_warps(self.k),
                              out_dtype=out_dtype)
        self.large = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype)

    def key(self) -> tuple:
        return (self.fp8.key(), self.max_fp8_rows, tuple(sorted(self.large.config.items())))

    @cute.jit
    def __call__(self, x: cute.Pointer, w_bf16: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(self.max_fp8_rows):
            self.fp8(x, w_fp8, scale, out, rows, stream)
        else:
            self.large(x, w_bf16, out, rows, stream)


class _Bf16Projection:
    """The BF16 projection behind the FP8 call signature (FP8 operands unused)."""

    def __init__(self, n: int, k: int):
        from ._glm_kernels import glm_projection

        self.inner = glm_projection(n, k)

    def key(self) -> tuple:
        return self.inner.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_bf16: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.inner(x, w_bf16, out, rows, stream)


def projection(n: int, k: int, fp8: bool):
    """A GLM projection with the ``(x, w, w_fp8, scale, out, rows, stream)`` call."""
    return RoutedFp8Projection(n, k) if fp8 else _Bf16Projection(n, k)


class Fp8Projection:
    """``out = x @ dequant(w)^T`` over a weight held only as E4M3 ``[N, K]`` plus FP32 scales
    (no BF16 copy). ``row_scales``: one scale per output row and 128-wide K block instead of
    the checkpoint's 128x128 grid (``[N, K/128]`` in decode programs, K-block major
    ``[K/128, N]`` in prefill programs).

    Decode programs (``prefill_rows`` None): ``MmaFp8Gemv`` for ``rows <= min(fp8_rows,
    gemv_rows)`` (a 16-row tile, plus a ``wide_rows`` multi-tile GEMV above 16 when given),
    ``TmaFp8Gemm`` above: W8A16, bitwise the BF16 programs over ``bf16(w * s)`` weights.

    Prefill programs: W8A8 when ``fp8_rows & prefill_mask`` is nonzero (``BlockFp8Projection``:
    E4M3 activations per row and 128-K block with FP32 ``amax / 448`` scales, the official FP8
    releases' served numerics), else ``TmaFp8Gemm`` (W8A16). ``qscratch`` holds the quantized
    rows (``prefill_scratch_bytes``)."""

    def __init__(self, n: int, k: int, *, prefill_rows: int | None = None, row_scales: bool = False,
                 kmajor: bool = False, gemv_rows: int = 16, wide_rows: int = 0, warps: int | None = None,
                 groups: int = 4, prefill_mask: int = 0xFF, out_dtype=cutlass.BFloat16,
                 wide_warps: int | None = None, wide_groups: int | None = None):
        self.n, self.k = int(n), int(k)
        self.row_scales = bool(row_scales)
        # Per-row scales K-block major in decode programs too (one copy serves both).
        self.kmajor = bool(kmajor) or (self.row_scales and prefill_rows is not None)
        self.prefill_mask = int(prefill_mask)
        self.prefill = prefill_rows is not None
        self.gemv = self.gemv_wide = self.w8a8 = None
        if self.prefill:
            from ._glmf_fp8 import BlockFp8Projection

            if out_dtype != cutlass.BFloat16 and self.prefill_mask:
                raise ValueError("non-BF16 prefill output requires W8A16 (prefill_mask=0)")
            self.w8a8 = BlockFp8Projection(self.n, self.k, int(prefill_rows), row_scales=self.row_scales) \
                if self.prefill_mask else None
            # 128-row tiles: the E4M3 tile is widened once per 128 rows (SM120 at 325 W, 4096 rows,
            # 6144x16384: 4 warps 3650 us, 8 warps 2779 us; BF16 TMA 128x128 tiles 2244 us).
            self.w8a16 = TmaFp8Gemm(self.n, self.k, scales="row_kmajor" if self.row_scales else "block",
                                    compute_warps=8, num_stages=3, out_dtype=out_dtype)
            self.max_gemv_rows = 0
        else:
            warps = gemv_warps(self.k) if warps is None else int(warps)
            self.gemv = MmaFp8Gemv(self.n, self.k, max_rows=int(gemv_rows), warps=warps, groups=int(groups),
                                   row_scales=self.row_scales, kmajor=self.kmajor, out_dtype=out_dtype)
            if int(wide_rows) > int(gemv_rows):
                self.gemv_wide = MmaFp8Gemv(self.n, self.k, max_rows=int(wide_rows),
                                            warps=warps if wide_warps is None else int(wide_warps),
                                            groups=int(groups) if wide_groups is None else int(wide_groups),
                                            row_scales=self.row_scales, kmajor=self.kmajor,
                                            out_dtype=out_dtype)
            self.gemv_rows = int(gemv_rows)
            self.max_gemv_rows = max(int(gemv_rows), int(wide_rows))
            tile_n = decode_tile_n(self.n)
            self.w8a16 = TmaFp8Gemm(self.n, self.k, scales=("row_kmajor" if self.kmajor else "row")
                                    if self.row_scales else "block", tile_n=tile_n,
                                    num_stages=4 if tile_n == 128 else 6, out_dtype=out_dtype)

    def key(self) -> tuple:
        return (self.n, self.k, self.row_scales, self.kmajor, self.prefill_mask, self.max_gemv_rows,
                None if self.gemv is None else self.gemv.key(),
                None if self.gemv_wide is None else self.gemv_wide.key(),
                None if self.w8a8 is None else self.w8a8.key(), self.w8a16.key())

    def prefill_scratch_bytes(self, rows: int) -> int:
        from ._glmf_fp8 import quant_scratch_bytes

        return quant_scratch_bytes(self.k, rows) if self.prefill else 0

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, fp8_rows: Int32, qscratch: Int64, stream: cuda.CUstream):
        if cutlass.const_expr(self.prefill):
            if cutlass.const_expr(self.w8a8 is None):
                self.w8a16(x, w_fp8, scale, out, rows, stream)
            else:
                if (fp8_rows & Int32(self.prefill_mask)) != Int32(0):
                    self.w8a8(x, w_fp8, scale, out, qscratch, rows, stream)
                else:
                    self.w8a16(x, w_fp8, scale, out, rows, stream)
        else:
            limit = fp8_rows
            if limit > Int32(self.max_gemv_rows):
                limit = Int32(self.max_gemv_rows)
            if rows <= limit:
                if cutlass.const_expr(self.gemv_wide is None):
                    self.gemv(x, w_fp8, scale, out, rows, stream)
                else:
                    if rows <= Int32(self.gemv_rows):
                        self.gemv(x, w_fp8, scale, out, rows, stream)
                    else:
                        self.gemv_wide(x, w_fp8, scale, out, rows, stream)
            else:
                self.w8a16(x, w_fp8, scale, out, rows, stream)


# Most decode rows the FP8-only programs run through the tensor-core GEMV (MmaFp8Gemv's 16-row tile).
FP8_GEMV_ROWS = 16


def decode_tile_n(n: int) -> int:
    """``TmaFp8Gemm`` N tile for decode steps above the GEMV rows (17-64 rows: a single M tile,
    so the CTA count is the N tiles): 128 from 16384 outputs, 64 from 4096, else 32. SM120 at
    325 W, L2-cold, 17/64 rows, us (BF16 TMA -> FP8 128 / chosen): GLM 5.3 o_proj 6144x16384
    143/145 -> 188/191 / 125/131 (64), q_b 16384x2048 51/53 -> 37/35 (128), q_a|kv_a 2624x6144
    59/59 -> 74/76 / 37/43 (32), dense down 6144x12288 127/127 -> 145/145 / 94/100 (64)."""
    n = int(n)
    if n >= 16384:
        return 128
    return 64 if n >= 4096 and n % 64 == 0 else 32


def check_w8_mode(mode: str) -> str:
    """``fp8_only`` of the compile functions: ``"decode"`` or ``"prefill"``."""
    if mode not in ("decode", "prefill"):
        raise ValueError(f"fp8_only must be 'decode' or 'prefill', got {mode!r}")
    return mode


def w8_scalars(prefill: bool) -> tuple:
    """Launch scalars of FP8-only programs: ``rows``, plus the prefill W8A8 switch."""
    from ._common import Scalar

    if prefill:
        return (Scalar("rows"), Scalar("fp8_rows", note="prefill: nonzero runs W8A8 (E4M3 activations per "
                                                        "row and 128-K block), 0 W8A16"))
    return (Scalar("rows"),)


def fp8_only_operands(name: str, n: int, k: int, *, row_scales: bool = False, prefill: bool = False,
                      kmajor: bool = False) -> tuple:
    """The ``{name}_fp8`` E4M3 ``[n, k]`` and ``{name}_scale`` FP32 operands of an ``Fp8Projection``:
    the checkpoint's 128x128 grid ``[ceil(n/128), k/128]``, or with ``row_scales`` one scale per
    row and 128-K block (``[n, k/128]``; prefill programs, and ``kmajor`` decode programs, take
    them K-block major as ``{name}_kscale [k/128, n]``)."""
    from ._common import Operand
    import torch

    kb = -(-int(k) // 128)
    if row_scales and (prefill or kmajor):
        return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
                Operand(f"{name}_kscale", torch.float32, f"[{kb},{n}]", align=16,
                        note="per-row x 128-K FP32 scales, K-block major"))
    if row_scales:
        shape, note = f"[{n},{kb}]", "per-row x 128-K FP32 scales"
    else:
        shape, note = f"[{-(-int(n) // 128)},{kb}]", "checkpoint weight_scale_inv (FP32 128x128 block scales)"
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
            Operand(f"{name}_scale", torch.float32, shape, align=16 if prefill else 4, note=note))


def fp8_operands(name: str, n: int, k: int) -> tuple:
    """The ``{name}_fp8`` / ``{name}_scale`` pointer operands of an FP8 weight."""
    from ._common import Operand
    import torch

    rows, cols = fp8_scale_shape(n, k)
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
            Operand(f"{name}_scale", torch.float32, f"[{rows},{cols}]", align=4,
                    note="checkpoint weight_scale_inv (FP32 128x128 block scales)"))


def gemv_warps(k: int) -> int:
    """K-splitting warps per CTA (SM120, L2-cold weights, us at M=1/5/16):
    hidden-width inputs (K=6144) take 8 (q_a+kv_a 14/16/18 vs 18/18/23 with 4;
    dense gate/up 98/98/100 vs 104/104/106), longer and shorter K take 4
    (o_proj 68/70/76 vs 76/76/80; dense down 53/57/63 vs 57/59/59)."""
    return 8 if int(k) == 6144 else 4


# ---------------------------------------------------------------------------
# TMA tensor-core GEMM over FP8 weights (BF16 MMA, FP32 accumulation)
# ---------------------------------------------------------------------------


@dsl_user_op
def _e4m3x8_scaled_bf16(lo, hi, s, *, loc=None, ip=None):
    """bf16(e4m3 * s) for eight packed E4M3 values, as four bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e0, e1, e2, e3, h0, h1, h2, h3, h4, h5, h6, h7;
            .reg .b32 p0, p1, p2, p3;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            mov.b32 {e0, e1}, $4;
            mov.b32 {e2, e3}, $5;
            cvt.rn.f16x2.e4m3x2 p0, e0;
            cvt.rn.f16x2.e4m3x2 p1, e1;
            cvt.rn.f16x2.e4m3x2 p2, e2;
            cvt.rn.f16x2.e4m3x2 p3, e3;
            mov.b32 {h0, h1}, p0;
            mov.b32 {h2, h3}, p1;
            mov.b32 {h4, h5}, p2;
            mov.b32 {h6, h7}, p3;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.f32.f16 f4, h4;
            cvt.f32.f16 f5, h5;
            cvt.f32.f16 f6, h6;
            cvt.f32.f16 f7, h7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


class TmaFp8Gemm:
    """``out = x @ (w * s)^T`` over E4M3 ``w [N, K]`` with FP32 scales per 128-wide K block.

    The Bf16PrefillKernel pipeline (TMA producer warp, warp MMA m16n8k16,
    FP32 accumulators) with the weight tile fetched as E4M3 (half the bytes).
    After each stage lands, the compute warps widen it to ``bf16(w * s)``
    into a swizzled BF16 tile and run the MMAs from it, so results equal the
    BF16 program over ``bf16(w * s)`` weights (W8A16, bitwise).

    ``scales``: ``"block"`` the checkpoint's 128x128 grid ``[ceil(N/128),
    K/128]`` (one scale per 128x64 tile); ``"row"`` one scale per output row
    and K block, ``[N, K/128]``; ``"row_kmajor"`` the same scales K-block
    major, ``[K/128, N]`` (the block-FP8 prefill GEMM's layout).
    """

    tile_k = 64
    buffer_align_bytes = 1024

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, compute_warps: int = 4,
                 num_stages: int = 4, scales: str = "block", tile_n: int = 128):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        # Narrower N tiles give narrow, long-K projections more CTAs (each widens fewer rows).
        if int(tile_n) not in (32, 64, 128):
            raise ValueError("FP8 TMA GEMM tile_n is 32, 64 or 128")
        self.tile_n = int(tile_n)
        if scales not in ("block", "row", "row_kmajor"):
            raise ValueError(f"scales must be block, row or row_kmajor, got {scales!r}")
        self.scales = scales
        self.num_compute_warps = int(compute_warps)
        self.compute_threads = 32 * self.num_compute_warps
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.num_compute_warps
        self.num_stages = int(num_stages)
        if self.k % 128:
            raise ValueError("FP8 TMA GEMM needs K % 128 == 0")
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.k_blocks = self.k // 128
        self.chunks = self.tile_n * self.tile_k // 8 // self.compute_threads

    def key(self) -> tuple:
        return (self.n, self.k, str(self.out_dtype), self.num_compute_warps, self.num_stages, self.scales,
                self.tile_n)

    def _tiled_mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    @cute.jit
    def _scale(self, scale: cute.Pointer, n_tile: Int32, n: Int32, k_tile: Int32) -> Float32:
        """The scale of weight row ``n`` of ``n_tile`` at K tile ``k_tile`` (rows past N clamp to
        the last one: their E4M3 values are TMA zero fill)."""
        kb = k_tile // Int32(2)
        if cutlass.const_expr(self.scales == "block"):
            block = n_tile // Int32(128 // self.tile_n)
            return _ld_f32(Int64(scale.toint()) + (Int64(block) * Int64(self.k_blocks) + Int64(kb)) * Int64(4))
        row = n_tile * Int32(self.tile_n) + n
        if row > Int32(self.n - 1):
            row = Int32(self.n - 1)
        if cutlass.const_expr(self.scales == "row"):
            return _ld_f32(Int64(scale.toint()) + (Int64(row) * Int64(self.k_blocks) + Int64(kb)) * Int64(4))
        return _ld_f32(Int64(scale.toint()) + (Int64(kb) * Int64(self.n) + Int64(row)) * Int64(4))

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.tile_k),
            cutlass.BFloat16)
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k), order=(0, 1))
        s_w = cute.make_layout((self.tile_n, self.tile_k, self.num_stages),
                               stride=(self.tile_k, 1, self.tile_n * self.tile_k))
        return s_a, s_b, s_w

    def _storage(self, s_a, s_b, s_w):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_a)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_b)], self.buffer_align_bytes],
            "sW": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(s_w)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        a_t = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        w8 = cute.make_ptr(cutlass.Uint8, Int64(w.toint()), cute.AddressSpace.gmem, assumed_align=16)
        w_t = cute.make_tensor(w8, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        o_t = cute.make_tensor(out, cute.make_layout((rows, self.n), stride=(self.n, 1)))
        s_a, s_b, s_w = self._layouts()
        storage = self._storage(s_a, s_b, s_w)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1)
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_w, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(tma_tensor_a, tma_tensor_w, o_t, scale, tma_a, tma_w, s_a, s_b, s_w,
                    self._tiled_mma(), storage, rows).launch(
            grid=(grid_m, self.n_tiles, 1), block=[self.num_threads, 1, 1], stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, output: cute.Tensor, scale: cute.Pointer,
               tma_atom_a: cute.CopyAtom, tma_atom_w: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
               s_b_layout: cute.ComposedLayout, s_w_layout: cute.Layout, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_w)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        s_w = storage.sW.get_tensor(s_w_layout)
        tma_bytes = (self.tile_m * 2 + self.tile_n) * self.tile_k
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
        g_w = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                           cute.group_modes(g_a, 0, 2))
        t_ws, t_wg = cpasync.tma_partition(tma_atom_w, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                           cute.group_modes(g_w, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb)
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc.fill(0.0)
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            t_ssa = copy_a.partition_S(s_a)
            t_ssb = copy_b.partition_S(s_b)
            w_base = shared_ptr_to_u32(storage.sW.data_ptr())
            b_base = shared_ptr_to_u32(storage.sB.data_ptr())
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                s = self._scale(scale, n_tile, Int32(0), k_tile)
                load_pipeline.consumer_wait(consumer_state)
                stage = w_base + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
                for i in cutlass.range_constexpr(self.chunks):
                    chunk = Int32(i * self.compute_threads) + Int32(tidx)
                    n = chunk // Int32(self.tile_k // 8)
                    c = chunk % Int32(self.tile_k // 8)
                    if cutlass.const_expr(self.scales != "block"):
                        s = self._scale(scale, n_tile, n, k_tile)
                    lo, hi = ld_shared_v2_u32(stage + n * Int32(self.tile_k) + c * Int32(8))
                    v0, v1, v2, v3 = _e4m3x8_scaled_bf16(lo, hi, s)
                    # 128-byte swizzle: 16-byte chunk c of row n sits at chunk c ^ (n % 8).
                    st_shared_v4_u32(b_base + n * Int32(128) + ((c ^ (n % Int32(8))) * Int32(16)), v0, v1, v2, v3)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
                shared_a = t_ssa[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, t_ssb[None, None, 0], target_b[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                    if kk < cute.size(shared_a.shape[2]) - 1:
                        cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                        cute.copy(copy_b, t_ssb[None, None, kk + 1], target_b[None, None, kk + 1])
                    cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            for index in cutlass.range_constexpr(cute.size(acc)):
                coord = coordinates[index]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens and column < Int32(self.n):
                    output[Int64(token), Int64(column)] = acc[index].to(output.element_type)
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile)], t_as[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_w, t_wg[(None, n_tile, k_tile)], t_ws[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


# ---------------------------------------------------------------------------
# Tensor-core GEMV over FP8 weights for decode rows (<= 64)
# ---------------------------------------------------------------------------


@dsl_user_op
def _e4m3x4_scaled_bf16x2x2(word, s, *, loc=None, ip=None):
    """bf16(e4m3 * s) for four packed E4M3 (byte 0 first) as two bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip), Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23, h0, h1, h2, h3;
            .reg .b32 p01, p23;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 p01, e01;
            cvt.rn.f16x2.e4m3x2 p23, e23;
            mov.b32 {h0, h1}, p01;
            mov.b32 {h2, h3}, p23;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            mul.rn.f32 f0, f0, $3;
            mul.rn.f32 f1, f1, $3;
            mul.rn.f32 f2, f2, $3;
            mul.rn.f32 f3, f3, $3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class MmaFp8Gemv:
    """Decode-row GEMM on tensor cores straight from global memory.

    A CTA owns ``8 * groups`` output columns; its ``warps`` split K. Per
    64-wide K chunk a warp runs four m16n8k16 BF16 MMAs per (16-row tile,
    8-column group). MMA k-slot ``s`` of step ``t`` holds logical ``k =
    16 * quad(s) + 4t + idx(s)`` (``idx`` of slots ``2j, 2j+1, 2j+8, 2j+9``
    is 0..3), so each lane reads one contiguous 16-byte weight run per group
    (row ``lane/4``) and two contiguous 32-byte activation runs per tile,
    reused by every group. Weights enter the MMA as ``bf16(w * s)`` (the BF16
    program's dequantized weight; one scale per 128-wide K block since a
    CTA's columns share a 128-row block), FP32 accumulation; warps reduce
    through shared memory; output BF16 or FP32 ``[rows, N]``.

    ``row_scales``: the scale grid is ``[N, K/128]`` FP32, one scale per
    output row and 128-wide K block (per-channel or row-block quantization of
    a BF16 weight); each lane scales its own row's weights. ``kmajor``: those
    per-row scales stored K-block major, ``[K/128, N]``.
    """

    def __init__(self, n: int, k: int, *, max_rows: int = 64, warps: int = 4, groups: int = 4,
                 out_dtype=cutlass.BFloat16, row_scales: bool = False, kmajor: bool = False):
        self.row_scales = bool(row_scales)
        self.kmajor = bool(kmajor)
        if self.kmajor and not self.row_scales:
            raise ValueError("K-block-major scales are per-row scales")
        self.n, self.k = int(n), int(k)
        self.warps, self.groups = int(warps), int(groups)
        self.cols = 8 * self.groups
        self.m_tiles = (int(max_rows) + 15) // 16
        self.out_dtype = out_dtype
        if self.n % self.cols or 128 % self.cols or self.k % (128 * self.warps):
            raise ValueError("MMA FP8 GEMV needs N % (8*groups) == 0, 8*groups dividing 128 and "
                             "K a multiple of 128 * warps")
        self.k_per_warp = self.k // self.warps
        self.blocks = self.k_per_warp // 128
        self.k_blocks = self.k // 128
        self.frags = self.m_tiles * self.groups * 4

    def key(self) -> tuple:
        return (self.n, self.k, self.warps, self.groups, self.m_tiles, str(self.out_dtype), self.row_scales,
                self.kmajor)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, w, scale, out, rows).launch(
            grid=(self.n // self.cols, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load_block(self, dest: cute.Tensor, w_row: Int64, k_off: Int64):
        for chunk in cutlass.range_constexpr(2):
            for gi in cutlass.range_constexpr(self.groups):
                words = _ld_stream(w_row + Int64(gi * 8 * self.k) + k_off + Int64(chunk * 64))
                for t in cutlass.range_constexpr(4):
                    dest[(chunk * self.groups + gi) * 4 + t] = words[t]

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.frags * 32,)))
        acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.frags):
            acc[i] = Float32(0.0)
        k_begin = Int64(warp_id) * Int64(self.k_per_warp)
        w_row = Int64(w.toint()) + (n0 + Int64(g)) * Int64(self.k) + k_begin + Int64(16) * Int64(j)
        x_base = Int64(x.toint()) + (k_begin + Int64(16) * Int64(j)) * Int64(2)
        # Scale of (row group gi, K block): one per CTA column block, or per lane row.
        s_rows = self.groups if self.row_scales else 1
        s_first = (n0 + Int64(g)) if self.row_scales else (n0 // Int64(128))
        # Byte strides between a lane's scales of consecutive K blocks and row groups.
        s_kb = Int64(self.n * 4) if self.kmajor else Int64(4)
        s_gi = Int64(8 * 4) if self.kmajor else Int64(8 * self.k_blocks * 4)
        s_row = Int64(scale.toint()) + s_first * (Int64(4) if self.kmajor else Int64(self.k_blocks * 4)) \
            + (k_begin // Int64(128)) * s_kb
        live_tiles = (rows + Int32(15)) // Int32(16)
        per_block = 2 * self.groups * 4
        # Register double buffer: block b+1's weights and scales load while b computes.
        cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        s_cur = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        s_nxt = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        self._load_block(cur, w_row, Int64(0))
        for si in cutlass.range_constexpr(s_rows):
            s_cur[si] = _ld_f32(s_row + Int64(si) * s_gi)
        for block in cutlass.range(self.blocks, unroll=1):
            for si in cutlass.range_constexpr(s_rows):
                s_nxt[si] = s_cur[si]
            if block + 1 < self.blocks:
                self._load_block(nxt, w_row, Int64(block + 1) * Int64(128))
                for si in cutlass.range_constexpr(s_rows):
                    s_nxt[si] = _ld_f32(s_row + Int64(si) * s_gi + Int64(block + 1) * s_kb)
            for chunk in cutlass.range_constexpr(2):
                k_off = Int64(block) * Int64(128) + Int64(chunk * 64)
                bw = cute.make_rmem_tensor(cute.make_layout((8 * self.groups,), stride=(1,)), Uint32)
                for gi in cutlass.range_constexpr(self.groups):
                    for t in cutlass.range_constexpr(4):
                        b0, b1 = _e4m3x4_scaled_bf16x2x2(cur[(chunk * self.groups + gi) * 4 + t],
                                                          s_cur[gi if self.row_scales else 0])
                        bw[8 * gi + 2 * t] = b0
                        bw[8 * gi + 2 * t + 1] = b1
                for mt in cutlass.range_constexpr(self.m_tiles):
                    if Int32(mt) < live_tiles:
                        r_lo = Int32(16 * mt) + g
                        r_hi = r_lo + Int32(8)
                        xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                        for i in cutlass.range_constexpr(16):
                            xa[i] = Uint32(0)
                        if r_lo < rows:
                            at = x_base + (Int64(r_lo) * Int64(self.k) + k_off) * Int64(2)
                            lo = _ld_cached(at)
                            hi = _ld_cached(at + Int64(16))
                            for i in cutlass.range_constexpr(4):
                                xa[i] = lo[i]
                                xa[4 + i] = hi[i]
                        if r_hi < rows:
                            at = x_base + (Int64(r_hi) * Int64(self.k) + k_off) * Int64(2)
                            lo = _ld_cached(at)
                            hi = _ld_cached(at + Int64(16))
                            for i in cutlass.range_constexpr(4):
                                xa[8 + i] = lo[i]
                                xa[12 + i] = hi[i]
                        for gi in cutlass.range_constexpr(self.groups):
                            f = 4 * (mt * self.groups + gi)
                            for t in cutlass.range_constexpr(4):
                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                    xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                    bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                                acc[f] = d0
                                acc[f + 1] = d1
                                acc[f + 2] = d2
                                acc[f + 3] = d3
            for i in cutlass.range_constexpr(per_block):
                cur[i] = nxt[i]
            for si in cutlass.range_constexpr(s_rows):
                s_cur[si] = s_nxt[si]
        for i in cutlass.range_constexpr(self.frags):
            partial[(warp_id * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
        cute.arch.sync_threads()
        out_bytes = 2 if const_expr(self.out_dtype == cutlass.BFloat16) else 4
        # Every warp stores a share of the fragments.
        for i in cutlass.range_constexpr(self.frags):
            if Int32(i % self.warps) == warp_id:
                total = Float32(0.0)
                for src in cutlass.range_constexpr(self.warps):
                    total = total + partial[(Int32(src * self.frags + i)) * Int32(32) + lane]
                mt = i // (4 * self.groups)
                gi = (i // 4) % self.groups
                row = Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                if row < rows:
                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(out_bytes)
                    target = cute.make_ptr(self.out_dtype, address, cute.AddressSpace.gmem, assumed_align=out_bytes)
                    target[0] = total.to(self.out_dtype)
