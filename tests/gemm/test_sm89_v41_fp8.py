"""SM89 V4.1 coordinator FP8: K32 quantizers bit-exact, promoted GEMMs exact.

Runs only on an SM89 (Ada) GPU. The quantizers must reproduce the SM120 K32
contract bit for bit (E4M3 values, UE8M0 row scales, amax floor) with the
MMA-scale pointer aliasing the row scales; the block-scaled projections
(decode GEMV and prefill GEMM, through the native pointer ABI) must match the
FP64 product of the dequantized operands to BF16 output rounding.
"""

import pytest
import torch

pytest.importorskip("cutlass")
import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Int32  # noqa: E402

from b12x._lib.intrinsics import pow2_ceil_ue8m0_torch  # noqa: E402
from b12x._lib.utils import make_ptr  # noqa: E402


def require_sm89():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 GPU required")


def _ptr(dtype, tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def quant_reference(x, floor):
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // 32, 32)
    amax = blocks.abs().amax(-1)
    if floor > 0:
        amax = amax.clamp_min(floor)
    _, byte = pow2_ceil_ue8m0_torch(amax * torch.tensor(1.0 / 448.0, device=x.device))
    byte = torch.where(amax == 0, torch.full_like(byte, 127), byte)
    inv = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    values = (blocks * inv[..., None]).to(torch.float8_e4m3fn).reshape(rows, k)
    return values.view(torch.uint8), byte


def rope_reference(x, freqs):
    rows, k = x.shape
    v = x.float().reshape(rows, k // 512, 512).clone()
    tail = v[..., 448:].reshape(rows, k // 512, 32, 2)
    c = freqs[:, None, :, 0].double()
    s = freqs[:, None, :, 1].float()
    v0, v1 = tail[..., 0], tail[..., 1]
    r0 = (v0.double() * c + (v1 * s).double()).float().bfloat16().float()
    r1 = (v1.double() * c + (-(v0 * s)).double()).float().bfloat16().float()
    v[..., 448:] = torch.stack((r0, r1), -1).reshape(rows, k // 512, 64)
    return v.reshape(rows, k).bfloat16()


@pytest.mark.parametrize("k,expected_m,rows", [(5120, 1, 1), (5120, 16, 13), (1280, 4096, 300),
                                               (2304, 64, 64), (15360, 80, 77)])
def test_rows_quant_bit_exact_with_aliased_mma_scales(k, expected_m, rows):
    require_sm89()
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid

    torch.manual_seed(k + rows)
    x = (torch.randn(rows, k, device="cuda") * torch.logspace(-6, 2, rows, device="cuda")[:, None])
    x[0, :64] = 0  # floored all-zero groups
    x = x.bfloat16()
    values = torch.full((rows * k,), 0xAA, device="cuda", dtype=torch.uint8)
    scales = torch.full((rows * (k // 32) + 64,), 0xAA, device="cuda", dtype=torch.uint8)
    quant = compile_mxfp8_rows_quant_aot(size_k=k, scale_block_size=32, expected_m=expected_m,
                                         amax_floor=1e-4)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = mxfp8_rows_quant_aot_grid(size_k=k, rows=rows, expected_m=expected_m, sm_count=sms)
    quant(_ptr(cutlass.BFloat16, x), _ptr(cutlass.Uint32, values), _ptr(cutlass.Uint8, scales),
          _ptr(cutlass.Uint8, scales), Int32(rows), Int32(grid), _stream())
    torch.cuda.synchronize()
    want_v, want_s = quant_reference(x, 1e-4)
    assert torch.equal(values.view(rows, k), want_v)
    assert torch.equal(scales[: rows * (k // 32)].view(rows, k // 32), want_s)
    assert bool((scales[rows * (k // 32):] == 0xAA).all())


@pytest.mark.parametrize("rope", [False, True])
def test_grouped_quant_bit_exact(rope):
    require_sm89()
    from b12x.gemm.wo_projection._quant_cute import compile_wo_grouped_quant_aot

    groups, width, rows = 8, 4096, 21
    torch.manual_seed(7)
    x = (torch.randn(rows, groups * width, device="cuda") * 3).bfloat16()
    x[1, :32] = 0  # zero group keeps scale 127 (no floor for WO-A)
    freqs = torch.randn(rows, 32, 2, device="cuda") if rope else None
    values = torch.zeros(groups * rows * width, device="cuda", dtype=torch.uint8)
    scales = torch.zeros(groups * rows * (width // 32), device="cuda", dtype=torch.uint8)
    quant = compile_wo_grouped_quant_aot(groups=groups, group_width=width, row_frequencies=rope)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    cos_sin = _ptr(cutlass.Float32, freqs) if rope else _ptr(cutlass.BFloat16, x)
    quant(_ptr(cutlass.BFloat16, x), _ptr(cutlass.Int64, x), cos_sin, _ptr(cutlass.Uint32, values),
          _ptr(cutlass.Uint8, scales), _ptr(cutlass.Uint8, scales), Int32(rows),
          Int32(rows * 64 if rope else 1), Int32(min(rows * 32, sms * 4)), _stream())
    torch.cuda.synchronize()
    source = rope_reference(x, freqs) if rope else x
    for g in range(groups):
        want_v, want_s = quant_reference(source[:, g * width:(g + 1) * width], 0.0)
        assert torch.equal(values[g * rows * width:(g + 1) * rows * width].view(rows, width), want_v)
        got_s = scales[g * rows * (width // 32):(g + 1) * rows * (width // 32)].view(rows, width // 32)
        assert torch.equal(got_s, want_s)


def _run_projection(n, k, groups, capacity, rows, graph=False):
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot

    gn, gk = n // groups, k // groups
    torch.manual_seed(n + k + capacity)
    a = (torch.randn(groups, rows, gk, device="cuda") * 3).to(torch.float8_e4m3fn)
    sa = torch.randint(110, 135, (groups, rows, gk // 32), device="cuda", dtype=torch.uint8)
    w = (torch.randn(groups, gn, gk, device="cuda") * 2).to(torch.float8_e4m3fn)
    sb = torch.randint(115, 130, (groups, gn // 32, gk // 32), device="cuda", dtype=torch.uint8)
    out = torch.zeros(groups * capacity * gn, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(1, device="cuda")
    gemm, split = compile_dense_gemm_mxfp8_aot(
        size_m=capacity, size_n=gn, size_k=gk, num_groups=groups, expected_m=capacity,
        return_split_k_metadata=True, sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    assert split == 1
    args = [_ptr(cutlass.Float8E4M3FN, a), _ptr(cutlass.Float8E4M3FN, w),
            _ptr(cutlass.Float8E8M0FNU, sa), _ptr(cutlass.Float8E8M0FNU, sb)]
    c = _ptr(cutlass.BFloat16, out)

    def launch(stream):
        gemm(*args, c, c, c, c, _ptr(cutlass.Float32, alpha), Int32(rows), stream)

    if graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            launch(cuda.CUstream(stream.cuda_stream))
        torch.cuda.synchronize()
        out.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            launch(cuda.CUstream(stream.cuda_stream))
        g.replay()
        torch.cuda.synchronize()
        g.reset()
    else:
        launch(_stream())
        torch.cuda.synchronize()
    got = out[: groups * rows * gn].view(groups, rows, gn).double()
    assert bool((out[groups * rows * gn:] == 0).all())
    for g in range(groups):
        da = (a[g].double().view(rows, gk // 32, 32) * torch.exp2(sa[g].double() - 127)[..., None]).view(rows, gk)
        db = (w[g].double().view(gn // 32, 32, gk // 32, 32)
              * torch.exp2(sb[g].double() - 127)[:, None, :, None]).view(gn, gk)
        ref = da @ db.T
        rel = ((got[g] - ref).abs().max() / ref.abs().max()).item()
        assert rel < 8e-3, (g, rel)


@pytest.mark.parametrize("n,k,groups,capacity,rows", [
    (1280, 5120, 1, 1, 1), (512, 5120, 1, 16, 9), (8192, 32768, 8, 16, 5),
    # Verify capacities: rows on both sides of each plan's kernel split.
    (5120, 2304, 1, 64, 61), (5120, 15360, 1, 64, 23), (5120, 15360, 1, 64, 47),
    (2304, 5120, 1, 80, 80), (1280, 5120, 1, 80, 30), (8192, 32768, 8, 80, 77),
    (8192, 32768, 8, 64, 40), (32768, 1280, 1, 80, 70),
])
def test_block32_decode_verify_matches_fp64(n, k, groups, capacity, rows):
    require_sm89()
    _run_projection(n, k, groups, capacity, rows)


@pytest.mark.parametrize("n,k,groups,capacity,rows", [
    (512, 5120, 1, 256, 251), (2304, 5120, 1, 1024, 1024), (5120, 1152, 1, 4096, 777),
    (8192, 32768, 8, 256, 130), (5120, 15360, 1, 256, 90), (8192, 32768, 8, 256, 85),
    # Multistage GEMM: partial row tiles, grouped, full capacity.
    (5120, 15360, 1, 256, 200), (8192, 32768, 8, 1024, 600), (1280, 5120, 1, 4096, 4093),
])
def test_block32_prefill_matches_fp64(n, k, groups, capacity, rows):
    require_sm89()
    _run_projection(n, k, groups, capacity, rows)


@pytest.mark.parametrize("max_rows", [32, 80])
def test_block32_gemv_every_live_tile_count(max_rows):
    """Each live row-tile count runs its own specialized loop (whole-block
    register prefetch up to 32 rows, the shared-memory ring above)."""
    require_sm89()
    from b12x._lib.sm89_fp8 import Sm89Block32Gemv

    n, k = 1024, 3072
    torch.manual_seed(max_rows)
    a = (torch.randn(max_rows, k, device="cuda") * 3).to(torch.float8_e4m3fn)
    sa = torch.randint(110, 135, (max_rows, k // 32), device="cuda", dtype=torch.uint8)
    w = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    sb = torch.randint(115, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    da = (a.double().view(max_rows, k // 32, 32) * torch.exp2(sa.double() - 127)[..., None]).view(max_rows, k)
    db = (w.double().view(n // 32, 32, k // 32, 32) * torch.exp2(sb.double() - 127)[:, None, :, None]).view(n, k)
    gemv = Sm89Block32Gemv(n, k, max_rows=max_rows, warps=8, col_groups=4)
    out = torch.zeros(max_rows * n, device="cuda", dtype=torch.bfloat16)
    args = [_ptr(cutlass.Float8E4M3FN, a), _ptr(cutlass.Float8E4M3FN, w),
            _ptr(cutlass.Float8E8M0FNU, sa), _ptr(cutlass.Float8E8M0FNU, sb)]
    c = _ptr(cutlass.BFloat16, out)
    compiled = cute.compile(gemv, *args, c, c, c, c, _ptr(cutlass.Float32, out), Int32(1), _stream())
    for rows in sorted({1, 7, 16, 17, 30, 32, 33, 48, 49, 63, 64, 65, 79, 80} & set(range(1, max_rows + 1))):
        # Rows are packed with live M: the row scales' stride follows the live count.
        sa_live = sa[:rows].contiguous()
        args[2] = _ptr(cutlass.Float8E8M0FNU, sa_live)
        out.zero_()
        compiled(*args, c, c, c, c, _ptr(cutlass.Float32, out), Int32(rows), _stream())
        torch.cuda.synchronize()
        ref = da[:rows] @ db.T
        got = out[: rows * n].view(rows, n).double()
        assert ((got - ref).abs().max() / ref.abs().max()).item() < 8e-3, rows
        assert bool((out[rows * n:] == 0).all()), rows


@pytest.mark.parametrize("capacity,rows", [(16, 16), (1024, 600)])
def test_block32_projection_replays_in_cuda_graph(capacity, rows):
    require_sm89()
    _run_projection(1280, 5120, 1, capacity, rows, graph=True)


@pytest.mark.parametrize("slices", [2, 4])
def test_split_k_gemv_partial_planes_sum_to_reference(slices):
    require_sm89()
    from b12x._lib.sm89_fp8 import Sm89Block32Gemv

    n, k, rows = 512, 5120, 11
    torch.manual_seed(slices)
    a = (torch.randn(rows, k, device="cuda") * 3).to(torch.float8_e4m3fn)
    sa = torch.randint(110, 135, (rows, k // 32), device="cuda", dtype=torch.uint8)
    w = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    sb = torch.randint(115, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    planes = torch.zeros(slices * 16 * n, device="cuda")
    gemv = Sm89Block32Gemv(n, k, max_rows=16, warps=8, slices=slices)
    args = [_ptr(cutlass.Float8E4M3FN, a), _ptr(cutlass.Float8E4M3FN, w),
            _ptr(cutlass.Float8E8M0FNU, sa), _ptr(cutlass.Float8E8M0FNU, sb)]
    c = _ptr(cutlass.Float32, planes)
    compiled = cute.compile(gemv, *args, c, c, c, c, _ptr(cutlass.Float32, planes), Int32(rows), _stream())
    compiled(*args, c, c, c, c, _ptr(cutlass.Float32, planes), Int32(rows), _stream())
    torch.cuda.synchronize()
    total = planes[: slices * rows * n].view(slices, rows, n).double().sum(0)
    da = (a.double().view(rows, k // 32, 32) * torch.exp2(sa.double() - 127)[..., None]).view(rows, k)
    db = (w.double().view(n // 32, 32, k // 32, 32) * torch.exp2(sb.double() - 127)[:, None, :, None]).view(n, k)
    ref = da @ db.T
    # FP32 partials: only the tensor core K32 partial rounding and FP32 adds.
    assert ((total - ref).abs().max() / ref.abs().max()).item() < 1e-3
    assert bool((planes[slices * rows * n:] == 0).all())


def test_output_projection_rows_bitwise_identical_across_decode_and_verify():
    """o_b shares one reduction recipe for every capacity up to 80 rows: a row's
    bits must not depend on the capacity (decode 1/16, verify 64/80) nor on how
    many rows are live (each verify kernel and live row-tile count)."""
    require_sm89()
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot

    n, k = 2560, 8192
    torch.manual_seed(5)
    a = (torch.randn(80, k, device="cuda") * 3).to(torch.float8_e4m3fn)
    sa = torch.randint(110, 135, (80, k // 32), device="cuda", dtype=torch.uint8)
    w = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    sb = torch.randint(115, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gemms = {}
    for capacity in (1, 16, 64, 80):
        gemm, split = compile_dense_gemm_mxfp8_aot(
            size_m=capacity, size_n=n, size_k=k, expected_m=16, return_split_k_metadata=True,
            sm_count=sms)
        assert split == 1
        gemms[capacity] = gemm

    def run(capacity, first, rows):
        out = torch.zeros(rows, n, device="cuda", dtype=torch.bfloat16)
        c = _ptr(cutlass.BFloat16, out)
        a_rows = a[first:first + rows].contiguous()
        sa_rows = sa[first:first + rows].contiguous()
        gemms[capacity](_ptr(cutlass.Float8E4M3FN, a_rows), _ptr(cutlass.Float8E4M3FN, w),
                        _ptr(cutlass.Float8E8M0FNU, sa_rows), _ptr(cutlass.Float8E8M0FNU, sb),
                        c, c, c, c, _ptr(cutlass.Float32, out), Int32(rows), _stream())
        torch.cuda.synchronize()
        return out

    decode = torch.cat([run(1, r, 1) for r in range(80)])
    for capacity, rows in ((16, 16), (16, 9), (64, 17), (64, 33), (64, 50), (64, 64),
                           (80, 20), (80, 47), (80, 65), (80, 80)):
        assert torch.equal(run(capacity, 0, rows), decode[:rows]), (capacity, rows)


def test_mhc_projection_runs_on_sm89():
    require_sm89()
    from b12x.norm.mhc._v41_project import compile_v41_mhc_project_aot

    rows, split = 7, 8
    torch.manual_seed(3)
    r = torch.randn(rows, 20480, device="cuda").bfloat16()
    w = torch.randn(24, 20480, device="cuda")
    p = torch.zeros(rows, 24, split, 2, device="cuda")
    project = compile_v41_mhc_project_aot(split_k=split)
    project(_ptr(cutlass.BFloat16, r), _ptr(cutlass.Float32, w), _ptr(cutlass.Float32, p),
            Int32(rows), _stream())
    torch.cuda.synchronize()
    dot = p[..., 0].sum(-1).double()
    sq = p[..., 1].sum(-1).double()
    ref_dot = r.double() @ w.double().T
    ref_sq = (r.double() ** 2).sum(-1, keepdim=True).expand(-1, 24)
    assert torch.allclose(dot, ref_dot, rtol=1e-4, atol=1e-2)
    assert torch.allclose(sq, ref_sq, rtol=1e-5, atol=1e-2)
