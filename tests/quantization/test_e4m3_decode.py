"""Exact E4M3 decode helpers and the NVFP4 block quantizers built on them.

``fp8_e4m3_to_f32`` once decoded every nonzero byte as a normal number,
``2^(e-7) * (1 + m/8)``, so the E4M3 subnormals (``e == 0``: ``m/8 * 2^-6``)
came back up to 4.5x too large. ``quantize_block_fp4`` stores the correct
scale byte but forms its E2M1 thresholds from the decoded scale, so every
block whose scale falls below 2^-6 (``amax * global_scale < 0.09375``) was
quantized against the wrong grid. These tests pin all 256 codes against
``torch.float8_e4m3fn`` and the quantizers against a reference NVFP4 QDQ on
blocks whose scales are subnormal, normal and at the E4M3 maximum.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    fp4_quantize_values_torch,
    fp8_e4m3_to_f32,
    fp8_e4m3_to_f32_and_rcp,
    max_abs_16,
    quantize_block_fp4,
    quantize_block_fp4_fast,
)
from b12x.quantization.nvfp4._kernel import fp8_e4m3_to_f32_and_effective_rcp

from tests._reference.helpers import require_b12x

_SUBNORMAL_LIMIT = 2.0**-6
_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _reference_e4m3(codes: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.uint8).cpu().view(torch.float8_e4m3fn).to(torch.float32)


def _cute(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


class _DecodeKernel:
    """One thread per code: the plain decode, its reciprocal, and the NVFP4
    quantizer's effective reciprocal (with a unit global-scale reciprocal)."""

    @cute.jit
    def __call__(self, codes: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(codes, out).launch(grid=(1, 1, 1), block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, codes: cute.Tensor, out: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]
        # Garbage above the low byte must be ignored.
        code = Uint32(codes[tidx]) | Uint32(0xABCD0000)
        out[tidx, 0] = fp8_e4m3_to_f32(code)
        out[tidx, 1] = fp8_e4m3_to_f32_and_rcp(code)
        out[tidx, 2] = fp8_e4m3_to_f32_and_effective_rcp(Uint32(codes[tidx]), Float32(1.0))


def _decode_all_codes() -> tuple[torch.Tensor, torch.Tensor]:
    device = require_b12x()
    codes = torch.arange(256, dtype=torch.int32, device=device)
    out = torch.zeros(256, 3, dtype=torch.float32, device=device)
    args = (_cute(codes, cutlass.Int32), _cute(out, cutlass.Float32), _stream())
    compiled = cute.compile(_DecodeKernel(), *args)
    compiled(*args)
    torch.cuda.synchronize()
    return codes.cpu(), out.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_e4m3_to_f32_is_exact_for_all_256_codes() -> None:
    codes, out = _decode_all_codes()
    reference = _reference_e4m3(codes)
    decoded = out[:, 0]
    nan = torch.isnan(reference)
    assert nan.nonzero().flatten().tolist() == [0x7F, 0xFF]
    assert torch.isnan(decoded[nan]).all(), decoded[nan]
    finite = ~nan
    # Bitwise: also distinguishes +0 (0x00) from -0 (0x80).
    assert torch.equal(decoded[finite].view(torch.int32), reference[finite].view(torch.int32)), [
        (int(c), float(d), float(r))
        for c, d, r in zip(codes[finite], decoded[finite], reference[finite])
        if d.view(torch.int32) != r.view(torch.int32)
    ]
    # The cases that motivated the fix.
    assert decoded[0x01].item() == 0.001953125
    assert decoded[0x07].item() == 0.013671875
    assert decoded[0x08].item() == _SUBNORMAL_LIMIT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_e4m3_reciprocal_helpers_match_the_exact_decode() -> None:
    codes, out = _decode_all_codes()
    reference = _reference_e4m3(codes).to(torch.float64)
    nan = torch.isnan(reference)
    zero = reference == 0
    regular = ~nan & ~zero
    expected = 1.0 / reference[regular]
    assert torch.isnan(out[nan, 1]).all(), out[nan, 1]
    for column, name in ((1, "fp8_e4m3_to_f32_and_rcp"), (2, "effective_rcp")):
        got = out[:, column].to(torch.float64)
        assert (got[zero] == 0).all(), (name, got[zero])
        rel = ((got[regular] - expected) / expected).abs()
        # rcp.approx.ftz is within 1 ulp; div.rn is exact to rounding.
        assert rel.max().item() <= 2.0**-22, (name, rel.max().item())
        assert torch.equal(torch.sign(got[regular]), torch.sign(expected)), name


class _QuantizeBlocksKernel:
    """One thread per 16-value block: exact and fast NVFP4 block quantizers."""

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,  # [blocks, 16] f32
        global_scale: cute.Tensor,  # [1] f32
        packed: cute.Tensor,  # [blocks, 4] u32: exact lo/hi, fast lo/hi
        scales: cute.Tensor,  # [blocks, 2] i32: exact, fast scale byte
        stream: cuda.CUstream,
    ):
        blocks = values.shape[0]
        self.kernel(values, global_scale, packed, scales).launch(
            grid=((blocks + 127) // 128, 1, 1), block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        global_scale: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        block = bidx * Int32(128) + tidx
        if block < values.shape[0]:
            row = cute.make_rmem_tensor((16,), Float32)
            for i in cutlass.range_constexpr(16):
                row[i] = Float32(values[block, i])
            gs = Float32(global_scale[0])
            amax = max_abs_16(row)
            exact, exact_scale = quantize_block_fp4(row, amax, gs)
            fast, fast_scale = quantize_block_fp4_fast(row, amax, gs)
            packed[block, 0] = Uint32(exact & cutlass.Uint64(0xFFFFFFFF))
            packed[block, 1] = Uint32(exact >> cutlass.Uint64(32))
            packed[block, 2] = Uint32(fast & cutlass.Uint64(0xFFFFFFFF))
            packed[block, 3] = Uint32(fast >> cutlass.Uint64(32))
            scales[block, 0] = Int32(exact_scale)
            scales[block, 1] = Int32(fast_scale)


def _unpack(lo_hi: torch.Tensor) -> torch.Tensor:
    """[blocks, 2] u32 words -> [blocks, 16] dequantized E2M1 grid values."""
    words = lo_hi.to(torch.int64) & 0xFFFFFFFF
    packed = words[:, 0] | (words[:, 1] << 32)
    shifts = torch.arange(16, dtype=torch.int64) * 4
    nibbles = (packed.unsqueeze(1) >> shifts) & 0xF
    lut = torch.tensor(_FP4_LUT + tuple(-v for v in _FP4_LUT), dtype=torch.float32)
    return lut[nibbles]


def _reference_qdq(values: torch.Tensor, global_scale: float):
    """ModelOpt / TensorRT-LLM NVFP4 block QDQ with exact E4M3 scales."""
    amax = values.abs().amax(dim=1, keepdim=True)
    raw = (amax * global_scale / FLOAT4_E2M1_MAX).clamp(max=FLOAT8_E4M3_MAX)
    scale_byte = raw.to(torch.float8_e4m3fn)
    scale = scale_byte.to(torch.float32)
    effective = scale / global_scale
    grid = torch.where(effective > 0, values / effective, torch.zeros_like(values))
    codes = fp4_quantize_values_torch(grid)
    return codes, scale_byte.view(torch.uint8).flatten().to(torch.int32), scale


def _blocks(global_scale: float) -> torch.Tensor:
    """Blocks whose scales span subnormal codes 0x01..0x07, the boundary
    0x08 and normal/maximum codes, each with randomized members."""
    gen = torch.Generator().manual_seed(20261112)
    targets = [m / 8 * 2.0**-6 for m in range(1, 8)]  # every subnormal
    targets += [2.0**-6, 2.0**-6 * 1.125, 0.03, 0.25, 1.0, 7.0, 300.0, 448.0]
    rows = []
    for target in targets:
        for _ in range(64):
            amax = target * FLOAT4_E2M1_MAX / global_scale
            amax *= float(torch.empty(1).uniform_(0.97, 1.03, generator=gen))
            row = torch.empty(16).uniform_(-1.0, 1.0, generator=gen) * amax
            row[int(torch.randint(16, (1,), generator=gen))] = amax
            rows.append(row)
    return torch.stack(rows).to(torch.float32)


def _run_quantizers(values: torch.Tensor, global_scale: float):
    device = require_b12x()
    blocks = values.shape[0]
    v = values.to(device).contiguous()
    gs = torch.tensor([global_scale], dtype=torch.float32, device=device)
    packed = torch.zeros(blocks, 4, dtype=torch.int32, device=device)
    scales = torch.zeros(blocks, 2, dtype=torch.int32, device=device)
    args = (
        _cute(v, cutlass.Float32),
        _cute(gs, cutlass.Float32),
        _cute(packed, cutlass.Uint32),
        _cute(scales, cutlass.Int32),
        _stream(),
    )
    compiled = cute.compile(_QuantizeBlocksKernel(), *args)
    compiled(*args)
    torch.cuda.synchronize()
    return packed.cpu(), scales.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("global_scale", (1.0, 1.0 / 3.0, 2688.0))
def test_quantize_block_fp4_matches_reference_on_subnormal_scales(global_scale: float) -> None:
    values = _blocks(global_scale)
    packed, scales = _run_quantizers(values, global_scale)
    codes, ref_scale_bytes, ref_scale = _reference_qdq(values, global_scale)
    subnormal = (ref_scale.flatten() > 0) & (ref_scale.flatten() < _SUBNORMAL_LIMIT)
    assert int(subnormal.sum()) >= 7 * 64

    assert torch.equal(scales[:, 0], ref_scale_bytes)
    exact = _unpack(packed[:, 0:2])
    assert torch.equal(exact, codes), int((exact != codes).any(dim=1).sum())

    # The fast path rounds through a reciprocal and RN E2M1 conversion: its
    # codes may differ from the reference only at exact grid midpoints
    # (one ulp of the reciprocal), never by more than one E2M1 step.
    assert torch.equal(scales[:, 1], ref_scale_bytes)
    fast = _unpack(packed[:, 2:4])
    mismatch = fast != codes
    assert mismatch.float().mean().item() < 0.01, mismatch.float().mean().item()
    lut = torch.tensor(_FP4_LUT)
    step = lambda x: torch.bucketize(x.abs(), lut)  # noqa: E731
    assert (step(fast) - step(codes)).abs().max().item() <= 1

    # QDQ error is the E2M1 grid's own: at most half the widest grid step
    # (6 - 4 = 2) at the block's effective scale.
    effective = (ref_scale / global_scale)
    deq = exact * effective
    worst = ((deq - values).abs() / effective.clamp(min=1e-30)).max().item()
    assert worst <= 1.0 + 1e-6, worst
    rel = (deq - values).norm() / values.norm()
    assert math.isfinite(rel.item()) and rel.item() < 0.2
