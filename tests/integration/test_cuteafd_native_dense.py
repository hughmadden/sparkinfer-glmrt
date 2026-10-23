"""GLM split SwiGLU and format-neutral F32->BF16 column packing."""
import pytest
import torch

from b12x.integration.cuteafd.native_dense import compile_swiglu_split_aot, compile_f32_narrow_pack_aot
from b12x.integration.cuteafd._glm_kernels import GlmSwiGLU
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program


@pytest.mark.parametrize("inter", [6144, 12288])
def test_split_swiglu_exact(inter):
    split = compile_swiglu_split_aot(inter=inter)
    packed = compile_program(GlmSwiGLU(inter), name="test_swiglu", operands=(
        Operand("gate_up", torch.bfloat16, f"[rows,{2*inter}]"),
        Operand("hidden", torch.bfloat16, f"[rows,{inter}]", "out")),
        scalars=(Scalar("rows"),), key=(inter,))
    gate = (torch.randn(65, inter, device="cuda")*10).bfloat16()
    up = (torch.randn_like(gate)*20).bfloat16()
    both = torch.cat((gate, up), dim=1)
    out, expected = torch.empty_like(gate), torch.empty_like(gate)
    for rows in (1, 16, 64, 65):
        split.launch(gate, up, out, scalars=(rows,))
        packed.launch(both, expected, scalars=(rows,))
        assert torch.equal(out[:rows], expected[:rows])


@pytest.mark.parametrize("width,pitch,c0", [(2048,2624,0), (576,2624,2048),
    (128,160,0), (32,160,128)])
def test_narrow_pack_preserves_other_columns(width, pitch, c0):
    program = compile_f32_narrow_pack_aot(width=width, pitch=pitch)
    source = torch.randn(65, width, device="cuda")
    destination = torch.full((65, pitch), 123., device="cuda", dtype=torch.bfloat16)
    expected = destination.clone()
    for rows in (1, 16, 64, 65):
        destination.fill_(123.)
        expected.fill_(123.)
        expected[:rows, c0:c0+width] = source[:rows].bfloat16()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(source, destination, scalars=(rows, c0))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(destination, expected)
    for invalid in (-1, pitch-width+1):
        destination.fill_(123.)
        program.launch(source, destination, scalars=(65, invalid))
        assert (destination == 123.).all()
