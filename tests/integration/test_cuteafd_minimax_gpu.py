"""MiniMax forward primitives: independent arithmetic, live rows and replay."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import require_b12x


def test_residual_norm_fp64_oracle_live_rows_and_replay():
    device = require_b12x()
    from b12x.integration.cuteafd.minimax import compile_minimax_norm_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_minimax_norm_aot()
    x = torch.randn(17, 6144, device=device, dtype=torch.float32)
    x[0].zero_()
    weight = (torch.randn(6144, device=device) * 0.1).bfloat16()
    out = torch.empty_like(x, dtype=torch.bfloat16)
    unused = torch.empty(1, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(17, device=device, dtype=torch.int64)
    args = (x, weight, unused, unused, unused, positions, out)

    def check(rows):
        reference = (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
                     * (1 + weight.double())).bfloat16()
        torch.testing.assert_close(out[:rows], reference[:rows], rtol=0.008, atol=0.0001)
        assert torch.isnan(out[rows:]).all()
        assert torch.count_nonzero(out[0]) == 0

    program.launch(*args, scalars=(17,))
    with kernel_resolution_guard("MiniMax residual norm live rows and graph"):
        for rows in (1, 7, 17):
            out.fill_(float("nan"))
            program.launch(*args, scalars=(rows,))
            check(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*args, scalars=(17,))
        x.neg_()
        graph.replay()
        check(17)


def test_fp32_router_weights_fp64_oracle_live_rows_and_replay():
    device = require_b12x()
    from b12x.integration.cuteafd.minimax import compile_minimax_router_scores_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_minimax_router_scores_aot()
    x = torch.randn(17, 6144, device=device).bfloat16()
    weight = torch.randn(128, 6144, device=device, dtype=torch.float32) * 0.01
    out = torch.empty(17, 128, device=device)

    def check(rows):
        reference = (x.double() @ weight.double().T).float()
        torch.testing.assert_close(out[:rows], reference[:rows], rtol=2e-5, atol=2e-6)
        assert torch.isnan(out[rows:]).all()

    program.launch(x, weight, out, scalars=(17,))
    with kernel_resolution_guard("MiniMax router live rows and graph"):
        for rows in (1, 8, 17):
            out.fill_(float("nan"))
            program.launch(x, weight, out, scalars=(rows,))
            check(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(x, weight, out, scalars=(17,))
        # The FP32 perturbation also verifies router weights were not rounded to BF16.
        weight.add_(0.000013)
        x.neg_()
        graph.replay()
        check(17)


@pytest.mark.parametrize("big_slot", [False, True])
def test_kv_pack_exact_bits_descales_negative_slots_live_rows_and_replay(big_slot):
    device = require_b12x()
    from b12x.integration.cuteafd.minimax import compile_minimax_kv_pack_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_minimax_kv_pack_aot()
    rows = 7
    k, v = [torch.randn(rows, 4, 128, device=device).bfloat16() for _ in range(2)]
    k[0].zero_()
    base = 2**31 // 512 + 64 if big_slot else 64
    slots = torch.tensor([base, base + 63, base + 64, -1, base + 129, base + 11, base + 130],
                         device=device, dtype=torch.int64)
    batch_ids = torch.tensor([0, 1, 2, 0, 2, 1, 0], device=device, dtype=torch.int32)
    kd = torch.tensor([[1, 0.5, 2, 0.25], [0.75, 1.5, 3, 0.125], [2.5, 0.3, 1, 0.2]], device=device)
    vd = kd.flip(0).contiguous()
    caches = [torch.empty((base + 192, 4, 128), dtype=torch.uint8, device=device) for _ in range(2)]
    tensors = (k, v, slots, batch_ids, kd, vd, *caches)
    # Initialize only the touched tail, keeping the >2 GiB offset test cheap.
    def reset():
        for cache in caches:
            cache[base:].fill_(173)

    def check(live):
        for value, descale, cache in zip((k, v), (kd, vd), caches):
            source = value.float().cpu().numpy()
            scales = descale[batch_ids.long()].cpu().numpy()
            scaled = source * (np.float32(1) / scales[..., None])
            reference = torch.from_numpy(scaled).to(torch.float8_e4m3fn).view(torch.uint8)
            expected = torch.full((192, 4, 128), 173, dtype=torch.uint8)
            for row in range(live):
                slot = int(slots[row].item())
                if slot >= 0:
                    expected[slot - base] = reference[row]
            assert torch.equal(cache[base:].cpu(), expected)

    reset()
    program.launch(*tensors, scalars=(rows,))
    check(rows)
    with kernel_resolution_guard("MiniMax KV pack live rows and graph"):
        for live in (1, 4, 7):
            reset()
            program.launch(*tensors, scalars=(live,))
            check(live)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*tensors, scalars=(rows,))
        k.neg_()
        v.mul_(0.5)
        reset()
        graph.replay()
        check(rows)
