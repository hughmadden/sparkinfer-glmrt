"""Native H128 companion oracles, row invariance and composed EXL3 forward."""
import os

import numpy as np
import pytest
import torch

from tests._reference.native_trellis import Reference
from tests.gemm.test_native_trellis_linear import KIMI_SHAPES


def hadamard(x):
    y = x.float().clone().reshape(*x.shape[:-1], -1, 128)
    # This is the primitive's exact natural-order F32 butterfly schedule.
    for width in (1, 2, 4, 8, 16, 32, 64):
        z = y.reshape(*y.shape[:-1], -1, 2, width)
        a, b = z[..., 0, :].clone(), z[..., 1, :].clone()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    return (y * np.float32(0.088388347648)).reshape_as(x)


def hadamard64(x):
    y = x.double().clone().reshape(*x.shape[:-1], -1, 128)
    for width in (1, 2, 4, 8, 16, 32, 64):
        z = y.reshape(*y.shape[:-1], -1, 2, width)
        a, b = z[..., 0, :].clone(), z[..., 1, :].clone()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    return (y / 128.**.5).reshape_as(x)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("features", [128, 7168, 12288, 163840])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_companion_bit_exact_rows_graph(dtype, features, output_dtype):
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    rows = 64 if features == 163840 else 4096
    x = (torch.randn((rows, features), device="cuda")*.03).to(torch.bfloat16 if dtype == "bf16" else torch.float32)
    suh = (torch.randn(features, device="cuda")*.2).half()
    svh = (torch.randn(features, device="cuda")*.2).half()
    rotated = torch.empty_like(x, dtype=torch.float16)
    blocks = torch.empty(features//128, device="cuda", dtype=torch.float16)
    source = torch.randn((rows, features), device="cuda")*.1
    output = torch.empty_like(x, dtype=output_dtype)
    inp = compile_trellis_input_aot(features=features, input_dtype=dtype, max_rows=rows)
    out = compile_trellis_output_aot(features=features, output_dtype="bf16" if output_dtype == torch.bfloat16 else "f32", max_rows=rows)
    expected_in = hadamard(x.float()*suh.float()).half()
    expected_out = (hadamard(source)*svh.float()).to(output_dtype)
    limit = torch.cuda.get_device_properties(0).multi_processor_count*4
    with kernel_resolution_guard("transforms prepared before replay"):
        for live in ((1, 16, 64) if rows == 64 else (1, 8, 16, 64, 256, 1024, 4096)):
            rotated.fill_(float("nan")); output.fill_(float("nan")); blocks.fill_(float("nan"))
            inp.launch(x, suh, rotated, blocks, scalars=(live, limit))
            out.launch(source, svh, output, scalars=(live, limit))
            assert torch.equal(rotated[:live].view(torch.int16), expected_in[:live].view(torch.int16))
            assert torch.equal(output[:live].view(torch.int16), expected_out[:live].view(torch.int16))
            assert torch.isnan(rotated[live:]).all() and torch.isnan(output[live:]).all()
            assert torch.equal(blocks, torch.ones_like(blocks))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            inp.launch(x, suh, rotated, blocks, scalars=(64, limit))
            out.launch(source, svh, output, scalars=(64, limit))
        for _ in range(3):
            rotated.fill_(float("nan")); output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(rotated[:64].view(torch.int16), expected_in[:64].view(torch.int16))
            assert torch.equal(output[:64].view(torch.int16), expected_out[:64].view(torch.int16))


def test_ingress_f16_default_export_unchanged(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import trellis_transform as module
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    geometry = dict(features=7168, input_dtype="bf16", max_rows=4096)
    with exportable_compilation():
        legacy = compile_program(module._Transform(7168, 4096, True), name="trellis_input",
            operands=(Operand("input", torch.bfloat16, "[rows,7168]"),
                      Operand("suh", torch.float16, "[7168]"),
                      Operand("rotated", torch.float16, "[rows,7168]", "out"),
                      Operand("input_block_scales", torch.float16, "[56]", "out")),
            scalars=(Scalar("rows"), Scalar("cta_limit")), key=tuple(geometry.values()),
            geometry=geometry, doc=module.__doc__)
        default = module.compile_trellis_input_aot(features=7168)
        explicit = module.compile_trellis_input_aot(features=7168, output_dtype="f16")
    for name, program in (("legacy", legacy), ("default", default), ("explicit", explicit)):
        directory = tmp_path/name
        directory.mkdir()
        program.export_to_c(str(directory), "ingress", "ingress")
    for suffix in ("o", "h"):
        expected = (tmp_path/"legacy"/f"ingress.{suffix}").read_bytes()
        assert (tmp_path/"default"/f"ingress.{suffix}").read_bytes() == expected
        assert (tmp_path/"explicit"/f"ingress.{suffix}").read_bytes() == expected


@pytest.mark.parametrize("features", [128, 7168])
def test_bf16_rotated_single_round(features):
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot
    x = torch.randn(256, features, device="cuda", dtype=torch.bfloat16)*.03
    suh = torch.randn(features, device="cuda", dtype=torch.float16)*.2
    rotated = torch.empty_like(x)
    blocks = torch.empty(features//128, device="cuda", dtype=torch.float16)
    program = compile_trellis_input_aot(features=features, max_rows=256, output_dtype="bf16")
    program.launch(x, suh, rotated, blocks, scalars=(256, 752))
    expected = hadamard(x.float()*suh.float()).bfloat16()
    torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
    torch.testing.assert_close(blocks, torch.ones_like(blocks), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [5, 6])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("k,n", [(1152,256)] + [
    pytest.param(k,n, marks=pytest.mark.skipif(
        os.environ.get("NATIVE_TRELLIS_FULL_SHAPES") != "1",
        reason="opt-in composed Kimi geometry gate"))
    for k,n in KIMI_SHAPES])
def test_composed_forward_f64(bits, codebook, side, k, n):
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    rows = 64
    torch.manual_seed(273)
    rng = np.random.default_rng(273)
    raw = rng.integers(0, 256, k*n*bits//8, dtype=np.uint8)
    ref = Reference(codebook)
    blocks = sorted({0, n//256, n//128-1})
    weights = []
    for nb in blocks:
        if side == 1:
            w = ref.decode_unit(raw.reshape(n//128,-1)[nb].tobytes(), bits, "gate", k)
        else:
            unit = raw.reshape(k//128,n//128,64*32*bits)[:,nb]
            w = np.concatenate([ref.decode_unit(u.tobytes(), bits, "down", 128) for u in unit])
        weights.append(torch.tensor(w, device="cuda"))
    weight = torch.cat(weights, dim=1)
    cols = torch.tensor([nb*128+i for nb in blocks for i in range(128)], device="cuda")
    x = (torch.randn((rows,k), device="cuda")*.03).bfloat16()
    suh = (torch.randn(k, device="cuda")*.2).half()
    svh = (torch.randn(n, device="cuda")*.2).half()
    rotated = torch.empty((rows,k), device="cuda", dtype=torch.float16)
    blocks = torch.empty(k//128, device="cuda", dtype=torch.float16)
    payload = torch.tensor(raw, device="cuda")
    pointers = torch.tensor([rotated.data_ptr()], device="cuda", dtype=torch.int64)
    partial = torch.empty((rows,(k+1023)//1024,n), device="cuda")
    acc = torch.empty((rows,n), device="cuda")
    output = torch.empty((rows,n), device="cuda", dtype=torch.bfloat16)
    inp = compile_trellis_input_aot(features=k, max_rows=rows)
    core = compile_trellis_linear_aot(input_features=k, output_features=n, bits=bits, codebook=codebook, sharded_side=side, rotation_count=1, max_rows=rows)
    out = compile_trellis_output_aot(features=n, max_rows=rows)
    limit = torch.cuda.get_device_properties(0).multi_processor_count*4
    inp.launch(x,suh,rotated,blocks,scalars=(rows,limit))
    core.launch(pointers,payload,blocks,partial.flatten(),acc.flatten(),scalars=(rows,0,limit))
    out.launch(acc,svh,output,scalars=(rows,limit))
    # Independent weight-side F64 reconstruction, matching EXL3's mathematical
    # H_K D H_N layout, without hiding activation-side F16 rounding differences.
    reconstructed = hadamard64(weight.T).T * suh.double()[:,None]
    reconstructed = hadamard64(reconstructed) * svh[cols].double()[None,:]
    expected = x.double() @ reconstructed
    actual = output[:,cols].double()
    delta = actual-expected
    relative_l2 = (delta.norm()/expected.norm()).item()
    maximum = delta.abs().max().item()
    print(f"K{bits} {codebook} side{side} shape={k}x{n} relative_l2={relative_l2:.8g} max_abs={maximum:.8g}")
    assert relative_l2 < .004
    torch.testing.assert_close(actual, expected, rtol=.04, atol=.0002)


@pytest.mark.parametrize("side", [1, 2])
def test_real_writer_fixture(side):
    """Optional pinned quantizer fixture, including its weight-side F16 rounds."""
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    directory = os.environ.get("NATIVE_TRELLIS_WRITER_FIXTURE")
    if not directory:
        pytest.skip("real quantizer fixture not supplied")
    tensors = load_file(str(Path(directory)/"core-k6.safetensors"))
    oracle = load_file(str(Path(directory)/"oracle.safetensors"))
    prefixes = sorted(key.removesuffix(".trellis") for key in tensors if key.endswith(".trellis"))
    assert len(prefixes) == 22
    for prefix in prefixes:
        codebook = prefix.split(".",1)[0]
        suh, svh = tensors[prefix+".suh"].cuda(), tensors[prefix+".svh"].cuda()
        k, n = suh.numel(), svh.numel()
        tiles = tensors[prefix+".trellis"]
        # The writer's [K16,N16,16*bits] grid becomes native ABI unit order;
        # codewords and packed uint32 contents are never decoded or altered.
        packed = (tiles.reshape(k//16,n//128,8,96).permute(1,0,2,3) if side == 1 else
                  tiles.reshape(k//128,8,n//128,8,96).permute(0,2,1,3,4))
        payload = packed.contiguous().cuda()
        x = oracle[prefix+".input_bf16"].cuda()
        rows = x.shape[0]
        rotated = torch.empty((rows,k), device="cuda", dtype=torch.float16)
        blocks = torch.empty(k//128, device="cuda", dtype=torch.float16)
        pointers = torch.tensor([rotated.data_ptr()], device="cuda", dtype=torch.int64)
        partial = torch.empty((rows,(k+1023)//1024,n), device="cuda")
        acc = torch.empty((rows,n), device="cuda")
        output = torch.empty((rows,n), device="cuda", dtype=torch.bfloat16)
        inp = compile_trellis_input_aot(features=k, max_rows=rows)
        core = compile_trellis_linear_aot(input_features=k, output_features=n,
            bits=6, codebook=codebook, sharded_side=side, rotation_count=1, max_rows=rows)
        out = compile_trellis_output_aot(features=n, max_rows=rows)
        limit = torch.cuda.get_device_properties(0).multi_processor_count*4
        inp.launch(x,suh,rotated,blocks,scalars=(rows,limit))
        core.launch(pointers,payload,blocks,partial.flatten(),acc.flatten(),scalars=(rows,0,limit))
        out.launch(acc,svh,output,scalars=(rows,limit))
        expected = oracle[prefix+".forward_f32"].cuda().double()
        delta = output.double()-expected
        relative_l2 = (delta.norm()/expected.norm()).item()
        maximum = delta.abs().max().item()
        print(f"writer {prefix} side{side} relative_l2={relative_l2:.8g} max_abs={maximum:.8g}")
        assert relative_l2 < .004
        # Weight-side F16 reconstruction rounds at additional boundaries. Near
        # cancellation, use a scale-relative max bound rather than relative error.
        assert maximum < .01 * expected.abs().max().item()
