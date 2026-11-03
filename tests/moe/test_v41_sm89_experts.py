"""SM89 V4.1 W4A8 expert slices: promoted MMA over the SM120 resident layout.

Runs only on an SM89 (Ada) GPU, where ``V41SlicePipeline`` selects the
promoted-MMA kernels. Ordered route planes must match the FP32 oracle of the
official BF16 -> MXFP8 -> BF16 recipe with V4.1 scale codes and raw UE8M0
scale words, stay bit-identical across live-row counts, capacities and CUDA
graph replay (``run_pipeline`` checks both), and zero routes to invalid
experts without changing the others.
"""

import pytest
import torch

pytest.importorskip("cutlass")
from tests.moe.test_v41_expert_numerics import reference  # noqa: E402
from tests.moe.test_v41_exact_slices import pack, run_pipeline  # noqa: E402

H = 5120
EXPERTS = 8  # run_pipeline's route-plan geometry


def require_sm89():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 GPU required")


def _problem(n, m, topk, seed):
    torch.manual_seed(seed)
    x = (torch.randn(m, H, device="cuda") * .5).bfloat16()
    ids = torch.rand(m, EXPERTS, device="cuda").topk(topk, -1).indices.int()
    routing = torch.rand(m, topk, device="cuda")
    routing = (routing / routing.sum(-1, keepdim=True) * 1.5).float()
    weights, scales = {}, {}
    for name, shape in [("w1", (EXPERTS, n, H // 2)), ("w3", (EXPERTS, n, H // 2)),
                        ("w2", (EXPERTS, H, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8)
    return x, ids, routing, weights, scales


def _resident(weights, scales, n, coded):
    packed = pack(weights, scales, False)
    if coded:
        from b12x.moe._shared.scale_codes import pack_scale_codes
        packed[1] = pack_scale_codes(torch.cat((scales["w3"], scales["w1"]), 1), tensor_name="w13.scale",
                                     gated_half_rows=n).view(torch.uint32).flatten().cuda()
        packed[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
    return packed


# TP2 halves (1152) and whole draft/backbone experts (2304); 16 rows fill two N tiles.
@pytest.mark.parametrize("n,m,topk", [(1152, 1, 6), (1152, 16, 6), (2304, 7, 3), (2304, 16, 3)])
@pytest.mark.parametrize("coded", [True, False])
def test_sm89_slices_match_oracle(n, m, topk, coded):
    require_sm89()
    x, ids, routing, weights, scales = _problem(n, m, topk, 8900 + n + m)
    routes = run_pipeline(x, ids, routing, _resident(weights, scales, n, coded), 192, False,
                          n=n, coded=coded, return_routes=True)
    actual = routes.bfloat16().float().sum(1)
    oracle = reference(x, ids, routing, {k: v.cuda() for k, v in weights.items()},
                       {k: v.cuda() for k, v in scales.items()})
    rel = ((actual - oracle).norm() / oracle.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), oracle.flatten(), dim=0).item()
    assert torch.isfinite(routes).all() and rel < 0.01 and cosine > 0.9999, (rel, cosine)


def test_sm89_capacity_invariance_and_invalid_routes():
    require_sm89()
    n, topk = 1152, 6
    x, ids, routing, weights, scales = _problem(n, 16, topk, 8916)
    packed = _resident(weights, scales, n, True)
    full = run_pipeline(x, ids, routing, packed, 192, False, n=n, coded=True, return_routes=True)
    one = run_pipeline(x[:1], ids[:1], routing[:1], packed, 192, False, n=n, coded=True,
                       return_routes=True)
    assert torch.equal(one[0], full[0]), "decode capacity changed a row's routes"
    dropped = ids.clone()
    dropped[:, 1] = -1
    partial = run_pipeline(x, dropped, routing, packed, 192, False, n=n, coded=True,
                           return_routes=True)
    assert bool((partial[:, 1] == 0).all()), "invalid expert routes must be zero"
    keep = [k for k in range(topk) if k != 1]
    assert torch.equal(partial[:, keep], full[:, keep]), "dropping a route changed the others"
