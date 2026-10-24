"""MiniMax router scores on the GPU: the straight-line FP32 dot equals the strided loop bit for bit.

The router program issues a thread's weight loads together (``unrolled_scalar``); its
products and sums keep the strided loop's order, so every live row's logits are the same
bits, and rows past the live ones are never written. (No postponed annotations: the
reference arm's jit signature names locally imported types.)
"""
import pytest

from ..conftest import require_b12x

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def programs():
    require_b12x()
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from b12x.gemm.bf16_gemv._kernel import SmallNGemvKernel
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd.minimax import compile_minimax_router_scores_aot

    class StridedRouter:
        """RouterScores before the straight-line dot (the reference arm)."""

        def __init__(self):
            self.project = SmallNGemvKernel(128, 6144, False, False)

        @cute.jit
        def __call__(self, x: cute.Pointer, weight: cute.Pointer, logits: cute.Pointer,
                     rows: Int32, stream: cuda.CUstream):
            self.project(x, weight, weight, logits, rows, Int64(6144), Int64(6144), Int64(128),
                         Int64(1), Int64(1), Int32(0), stream)

    strided = compile_program(StridedRouter(), name="minimax_router_scores_strided_reference", operands=(
        Operand("x", torch.bfloat16, "[rows,6144]"),
        Operand("weight", torch.float32, "[128,6144]"),
        Operand("logits", torch.float32, "[rows,128]", "out")), scalars=(Scalar("rows"),), key=())
    return compile_minimax_router_scores_aot(), strided


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 9, 16, 33, 64])
def test_router_rows_match_the_strided_loop_bit_for_bit(programs, rows):
    unrolled, strided = programs
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn((rows, 6144), generator=generator, device="cuda").bfloat16()
    weight = torch.randn((128, 6144), generator=generator, device="cuda") * 0.02
    outputs = []
    for program in (unrolled, strided):
        logits = torch.full((rows + 8, 128), float("nan"), dtype=torch.float32, device="cuda")
        program.launch(x, weight, logits, scalars=(rows,))
        torch.cuda.synchronize()
        outputs.append(logits)
    assert torch.equal(outputs[0][:rows].view(torch.int32), outputs[1][:rows].view(torch.int32)), \
        "straight-line and strided logits differ"
    assert outputs[0][rows:].isnan().all(), "rows past the live ones were written"
    # FP64 reference: an FP32 matmul may run as TF32 (NGC images set the cuBLAS override).
    expected = x.double() @ weight.double().t()
    assert torch.allclose(outputs[0][:rows].double(), expected, rtol=1e-4, atol=1e-4), "logits off the FP64 dot"
