"""M-RoPE post/pooling oracle, scalar-text identity and stable-table graph replay."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _programs(legacy=False):
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT as g
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._qwen4_attention_kernels import Qwen4AttnPost, Qwen4PoolKeys, _Rope, _bf16

    class ScalarRope(_Rope):
        # The old scalar arithmetic, independent of the new axis selector.
        @cute.jit
        def apply(self, y: Float32, partner: Float32, dim: Int32, position: cute.Tensor) -> Float32:
            c, s = self.cos_sin(dim % Int32(self.half), Int64(position[0]))
            rotated = partner * s
            if dim < Int32(self.half):
                rotated = -rotated
            return _bf16(_bf16(y * c) + _bf16(rotated))

    post, pool = Qwen4AttnPost(g), Qwen4PoolKeys(g)
    if legacy:
        post.rope = ScalarRope(g.rope_dim, g.rope_theta)
        pool.post.rope = ScalarRope(g.rope_dim, g.rope_theta)
    bf, i64, i32 = torch.bfloat16, torch.int64, torch.int32
    post_ops = tuple(Operand(n, dt, shape, align=align) for n, dt, shape, align in (
        ("proj", bf, "[rows,P]", 16), ("q_norm", bf, "[256]", 16), ("k_norm", bf, "[256]", 16),
        ("iq_norm", bf, "[128]", 16), ("ik_norm", bf, "[128]", 16),
        ("rope_positions", i32, "[rows,3]", 4), ("kv_slots", i64, "[rows]", 8),
        ("kv_cache", torch.uint8, "[records,2048]", 16), ("token_keys", bf, "[records,128]", 16),
        ("query", bf, "[rows,24,256]", 16), ("gate", bf, "[rows,6144]", 16),
        ("index_q", bf, "[rows,4,128]", 16)))
    pool_ops = tuple(Operand(n, dt, shape, align=align) for n, dt, shape, align in (
        ("block_rope_positions", i32, "[rows,3]", 4), ("kv_slots", i64, "[rows]", 8),
        ("pool_slots", i64, "[rows]", 8), ("ik_norm", bf, "[128]", 16),
        ("token_keys", bf, "[records,128]", 16), ("index_cache", bf, "[blocks,128]", 16)))
    return g, compile_program(post, name="mrope_test_post", operands=post_ops, scalars=(Scalar("rows"),),
                              key=(legacy,)), compile_program(pool, name="mrope_test_pool", operands=pool_ops,
                                                             scalars=(Scalar("rows"),), key=(legacy,))


def _norm_rope(x, w, positions):
    from b12x.integration.cuteafd._qwen4_attention_kernels import rope_inv_freq
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                  * (1 + w.float())).bfloat16()
    inv = torch.tensor(rope_inv_freq(64, 1e7), device="cuda")
    freq = positions.float()[:, :, None] * inv[None, None, :]
    selected = freq[:, 0].clone()
    selected[:, 1:33:3] = freq[:, 1, 1:33:3]
    selected[:, 2:30:3] = freq[:, 2, 2:30:3]
    c = torch.cat((selected.cos(), selected.cos()), -1).bfloat16()[:, None, :]
    s = torch.cat((selected.sin(), selected.sin()), -1).bfloat16()[:, None, :]
    first = normalized[..., :64]
    partner = torch.cat((-first[..., 32:], first[..., :32]), -1)
    return torch.cat(((first * c + partner * s).bfloat16(), normalized[..., 64:]), -1)


def _case(text=False, large=False):
    g, post, pool = _programs()
    rows = 8
    gen = torch.Generator(device="cuda").manual_seed(6308)
    proj = torch.randn((rows, g.attn_in_width), device="cuda", generator=gen).bfloat16()
    norms = [(0.1 * torch.randn((d,), device="cuda", generator=gen)).bfloat16() for d in (256, 256, 128, 128)]
    logical = torch.arange(rows, device="cuda", dtype=torch.int32)
    # Row two starts a 2x3 image. Block zero starts in text; block one starts
    # inside the image. Neither block position equals completing THW minus 3.
    coords = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3],
              [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]]
    positions = (logical[:, None].expand(-1, 3).contiguous() if text else
                 torch.tensor(coords, device="cuda", dtype=torch.int32))
    block_positions = positions[logical - logical % 4].contiguous()
    slot_base, pool_base = (1 << 20, 1 << 23) if large else (64, 64)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64) + slot_base
    pools = torch.where(logical % 4 == 3, logical.to(torch.int64) // 4 + pool_base, -1)
    # Only touched records are read. Large sparse offsets cross 2^31 bytes.
    kv = torch.empty((slot_base + rows, 2048), device="cuda", dtype=torch.uint8)
    keys = torch.empty((slot_base + rows, 128), device="cuda", dtype=torch.bfloat16)
    index = torch.empty((pool_base + 2, 128), device="cuda", dtype=torch.bfloat16)
    query = torch.empty((rows, 24, 256), device="cuda", dtype=torch.bfloat16)
    gate = torch.empty((rows, 6144), device="cuda", dtype=torch.bfloat16)
    iq = torch.empty((rows, 4, 128), device="cuda", dtype=torch.bfloat16)

    def launch(p=post, b=pool):
        p.launch(proj, *norms, positions, slots, kv, keys, query, gate, iq, scalars=[rows])
        b.launch(block_positions, slots, pools, norms[3], keys, index, scalars=[rows])

    def outputs():
        return [query.clone(), gate.clone(), iq.clone(), kv[slot_base:].clone(),
                keys[slot_base:].clone(), index[pool_base:].clone()]

    launch()
    torch.cuda.synchronize()
    baseline = outputs()
    if text:
        _, scalar_post, scalar_pool = _programs(legacy=True)
        launch(scalar_post, scalar_pool)
        torch.cuda.synchronize()
        for actual, expected in zip(outputs(), baseline):
            assert torch.equal(actual, expected), "text scalar path must remain byte-identical"
    else:
        qg = proj[:, :12288].reshape(rows, 24, 512)
        k, v = proj[:, 12288:12800].reshape(rows, 2, 256), proj[:, 12800:13312].reshape(rows, 2, 256)
        expected = [_norm_rope(qg[..., :256], norms[0], positions), qg[..., 256:].reshape(rows, -1),
                    _norm_rope(proj[:, 13312:13824].reshape(rows, 4, 128), norms[2], positions),
                    torch.cat((_norm_rope(k, norms[1], positions).reshape(rows, -1), v.reshape(rows, -1)), -1),
                    proj[:, 13824:]]
        actual = [query, gate, iq, kv[slot_base:].view(torch.bfloat16), keys[slot_base:]]
        # Torch and warp RMS reductions differ slightly; test actual BF16 outputs.
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0.016, atol=0.016)
        raw = proj[:, 13824:].reshape(2, 4, 128)
        mean = ((raw[:, 0].float() + raw[:, 1].float()) + (raw[:, 2].float() + raw[:, 3].float())) * 0.25
        pooled = _norm_rope(mean.bfloat16()[:, None], norms[3], positions[[0, 4]])[:, 0]
        torch.testing.assert_close(index[pool_base:], pooled, rtol=0.016, atol=0.016)
        assert not torch.equal(query, _norm_rope(qg[..., :256], norms[0], logical[:, None].expand(-1, 3)))

    # Capture once, then mutate values in the same allocation. No recapture.
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    positions.add_(17)
    block_positions.add_(17)
    graph.replay()
    torch.cuda.synchronize()
    replayed = outputs()
    launch()
    torch.cuda.synchronize()
    for a, b in zip(replayed, outputs()):
        assert torch.equal(a, b), "graph must read current rotary table values"
    assert not torch.equal(replayed[0], baseline[0])


@pytest.mark.parametrize("text,large", [(True, False), (False, False), (False, True)])
def test_qwen4_mrope_post_pool_and_replay(text, large):
    _case(text, large)
