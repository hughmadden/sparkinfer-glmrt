"""Paired boundary prototype: compare skipped reads with a full-width control.

The control zeroes the gate post-rotation scales for the omitted H128 block,
so it computes and streams all resident blocks but contributes one fewer.
This is not yet the independent real-checkpoint, four-rank release oracle.
"""
from dataclasses import replace
import pytest
import torch
from tests.moe.test_w4a16_mixed_trellis import _prepared
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    compile_mixed_trellis, build_tiered_maps, combine_trellis_rotations,
    make_mixed_trellis_buffers, bind_mixed_trellis, run_bound_mixed_trellis,
)

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1), reason='requires Spark SM121')
@pytest.mark.parametrize('boundary', ['first', 'last'])
@pytest.mark.parametrize('tile_k', [64, 128])
@pytest.mark.parametrize('intermediate', [256, 384, 640, 768])
def test_paired_boundary_skips_match_full_width_control(boundary, tile_k, intermediate):
    torch.manual_seed(410917)
    device = torch.device('cuda', 0)
    props = torch.cuda.get_device_properties(device)
    assert (props.major, props.minor) == (12, 1)
    tile = (tile_k, 128, tile_k, 128)
    tiers = tuple(_prepared(experts=2, hidden=128, intermediate=intermediate,
        bits=bits, seed=bits*101, device=device, tile_config=tile) for bits in (3, 4))
    mapping, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    extended = torch.cat((descriptor, torch.ones(4, dtype=torch.int32, device=device)))
    extended._mt_projection_counts = descriptor._mt_projection_counts
    rotations = combine_trellis_rotations(*tiers)
    control_rotations = replace(rotations, intermediate=rotations.intermediate.clone())
    args = dict(size_m=16, hidden_size=128, intermediate_size=intermediate,
        tier0_num_experts=2, tier1_num_experts=2, top_k=2, max_m_blocks=8,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=tile, full_rotation_output_dtype='bf16', swiglu_limit=10.)
    plans=[]
    for mode, desc, rot in [(None, descriptor, control_rotations), (boundary, extended, rotations)]:
        launch = compile_mixed_trellis(**args, paired_boundary=mode)
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*tiers, mapping, desc, rot, launch)
        plans.append((binding, buffers))
    x = (torch.randn(16, 128, device=device)*.01).to(torch.bfloat16)
    ids = torch.tensor([[0,1],[3,-1],[2,3],[0,3],[2,1],[0,2],[1,3],[2,3]], device=device, dtype=torch.int32)
    ids = ids.repeat(2, 1)
    weights = torch.full((16,2), .5, device=device); weights[1,1]=0
    for rows in (1, 3, 8, 16):
        graphs=[]; outputs=[]
        for binding, buffers in plans:
            for _ in range(2): run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], binding, buffers)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out=run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], binding, buffers)
            graphs.append(graph); outputs.append(out)
        for bits in ([1,1,1,1], [0,1,0,1], [1,0,1,0], [0,0,0,0], [1,1,1,1]):
            owned=torch.tensor(bits, dtype=torch.int32, device=device)
            extended[-4:].copy_(owned)
            control_rotations.intermediate.copy_(rotations.intermediate)
            start=0 if boundary=='first' else intermediate-128
            control_rotations.intermediate.view(4, 3*intermediate)[:,start:start+128].mul_(owned[:,None])
            # Unowned gate/up outputs may be stale, including NaNs; omitted
            # blocks must not contaminate any retained down-projection input.
            plans[1][1].fc1.fill_(float('nan'))
            x.mul_(-.75)
            for graph in graphs: graph.replay()
            torch.cuda.synchronize()
            ref, got=outputs
            assert torch.isfinite(ref[:rows]).all() and ref[:rows].abs().max()>0
            assert torch.isfinite(got[:rows]).all()
            error=float((got[:rows].float()-ref[:rows].float()).abs().max()/ref[:rows].float().abs().max())
            assert error <= .004, (boundary,tile_k,rows,bits,error)
            if all(bits):
                assert torch.equal(ref[:rows], got[:rows]), (boundary, tile_k, rows)
            saved=got.clone()
            run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], *plans[1])
            torch.cuda.synchronize()
            assert torch.equal(saved[:rows], got[:rows])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1), reason="requires Spark SM121")
@pytest.mark.parametrize("intermediate", [384, 640, 768])
@pytest.mark.parametrize("tile_k", [64, 128])
@pytest.mark.parametrize("all_owned", [False, True])
def test_compact_true_width_weights_match_padded_control(intermediate, tile_k, all_owned):
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import build_compact_trellis_descriptor
    torch.manual_seed(521031)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    tile = (tile_k, 128, tile_k, 128)
    tiers = tuple(_prepared(experts=2, hidden=128, intermediate=intermediate,
        bits=bits, seed=bits * 211, device=device, tile_config=tile) for bits in (3, 4))
    mapping, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    widths = [intermediate] * 4 if all_owned else [intermediate, intermediate - 128, intermediate - 128, intermediate]
    compact_descriptor = build_compact_trellis_descriptor(descriptor, widths,
        hidden_size=128, intermediate_size=intermediate)
    compact_tiers = []
    for tier_id, tier in enumerate(tiers):
        bits = 3 + tier_id
        packed = tier.w13.reshape(2, 2, 8, intermediate // 16, 8 * bits)
        down = tier.w2.reshape(2, intermediate // 16, 8, 8 * bits)
        local_widths = widths[tier_id * 2:tier_id * 2 + 2]
        w13 = torch.cat([packed[p, e, :, :width // 16].contiguous().flatten()
            for p in range(2) for e, width in enumerate(local_widths)])
        w2 = torch.cat([down[e, :width // 16].contiguous().flatten()
            for e, width in enumerate(local_widths)])
        compact_tiers.append(replace(tier, w13=w13, w2=w2))
        assert w13.numel() + w2.numel() == sum(local_widths) * 128 * bits * 3 // 32
    rotations = combine_trellis_rotations(*tiers)
    control = replace(rotations, intermediate=rotations.intermediate.clone())
    for expert, width in enumerate(widths):
        control.intermediate.view(4, 3 * intermediate)[expert, width:intermediate] = 0
    args = dict(size_m=16, hidden_size=128, intermediate_size=intermediate,
        tier0_num_experts=2, tier1_num_experts=2, top_k=2, max_m_blocks=8,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=tile, full_rotation_output_dtype="bf16", swiglu_limit=10.)
    plans = []
    for compact, payload, desc, rot in [(False, tiers, descriptor, control),
            (True, compact_tiers, compact_descriptor, rotations)]:
        launch = compile_mixed_trellis(**args, compact_weights=compact)
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*payload, mapping, desc, rot, launch)
        plans.append((binding, buffers))
    # Fail before launch for an offset table that would address outside its bank.
    broken = compact_descriptor.clone()
    broken[4 * 4] += 4
    with pytest.raises(ValueError, match="not canonical"):
        bind_mixed_trellis(*compact_tiers, mapping, broken, rotations, plans[1][0].launch)
    x = (torch.randn(16, 128, device=device) * .01).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [3, -1], [2, 3], [0, 3], [2, 1], [0, 2], [1, 3], [2, 3]],
        device=device, dtype=torch.int32).repeat(2, 1)
    weights = torch.full((16, 2), .5, device=device)
    weights[1, 1] = 0
    for rows in (1, 3, 8, 16):
        graphs, outputs = [], []
        for binding, buffers in plans:
            for _ in range(2):
                run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], binding, buffers)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], binding, buffers)
            graphs.append(graph)
            outputs.append(out)
        for _ in range(3):
            plans[1][1].fc1.fill_(float("nan"))
            x.mul_(-.75)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            ref, got = outputs
            assert torch.isfinite(ref[:rows]).all() and ref[:rows].abs().max() > 0
            assert torch.isfinite(got[:rows]).all()
            error = float((got[:rows].float() - ref[:rows].float()).abs().max() / ref[:rows].float().abs().max())
            assert error <= .004, (intermediate, tile_k, rows, error)
            if all_owned:
                assert torch.equal(ref[:rows], got[:rows])
            saved = got.clone()
            run_bound_mixed_trellis(x[:rows], weights[:rows], ids[:rows], *plans[1])
            torch.cuda.synchronize()
            assert torch.equal(saved[:rows], got[:rows])
