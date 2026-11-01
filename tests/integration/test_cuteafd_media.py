"""Official media operation boundaries, exact histories and wave invariance."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x

ROWS = (1, 16, 64, 80, 2752, 4096)


def _ulp_close(actual, expected):
    expected = expected.float()
    ulp = torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(2.0 ** -126))) - 7)
    error = (actual.float() - expected).abs()
    assert (error <= ulp).all(), (error.max().item(), (error / ulp).max().item())


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dim,bf16_products", [(96, False), (128, True)])
def test_axial_rotary(rows, dim, bf16_products):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_axial_rotary_aot
    g = torch.Generator().manual_seed(123)
    q, k = [torch.randn(rows, 16, dim, generator=g).bfloat16().cuda() for _ in range(2)]
    # Pre-rounded tables deliberately vary by row, including permuted coordinates.
    phase = torch.randn(rows, dim // 2, generator=g).repeat(1, 2)
    c, s = phase.cos().bfloat16().cuda(), phase.sin().bfloat16().cuda()
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    program = compile_axial_rotary_aot(dim=dim, bf16_products=bf16_products)
    program.launch(q, k, c, s, oq, ok, scalars=[rows])
    for x, out in ((q, oq), (k, ok)):
        rotated = torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), -1)
        if bf16_products:
            ref = x * c[:, None] + rotated * s[:, None]
        else:
            ref = (x.float() * c[:, None].float() + rotated.float() * s[:, None].float()).bfloat16()
        assert torch.equal(out, ref)
    one_q, one_k = torch.empty_like(q[:1]), torch.empty_like(k[:1])
    program.launch(q[:1], k[:1], c[:1], s[:1], one_q, one_k, scalars=[1])
    assert torch.equal(one_q, oq[:1]) and torch.equal(one_k, ok[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_position_gather(rows):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_position_gather_aot
    g = torch.Generator().manual_seed(124)
    table = torch.randn(1024, 1536, generator=g).bfloat16().cuda()
    ix = torch.randint(0, 1024, (rows, 4), generator=g).int().cuda()
    w = torch.rand(rows, 4, generator=g).cuda()
    w /= w.sum(-1, keepdim=True)
    w[0, 1:] = 0
    patch = torch.randn(rows, 1536, generator=g).bfloat16().cuda()
    total = table[ix[:, 0].long()].float() * w[:, :1]
    for tap in range(1, 4):
        total = total + table[ix[:, tap].long()].float() * w[:, tap:tap+1]
    ref = patch + total.bfloat16()
    out = torch.empty_like(patch)
    program = compile_position_gather_aot()
    program.launch(table, ix, w, patch, out, scalars=[rows])
    assert torch.equal(out, ref)
    one = torch.empty_like(out[:1])
    program.launch(table, ix[:1], w[:1], patch[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("width", (256, 512, 1024, 2048))
@pytest.mark.parametrize("length", (1, 2, 5, 6, 7, 16, 64, 80))
def test_causal_conv_history(width, length):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_causal_depthwise_conv7_aot
    g = torch.Generator().manual_seed(125)
    x = torch.randn(2, length, width, generator=g).bfloat16().cuda()
    w = torch.randn(width, 7, generator=g).bfloat16().cuda()
    bias = torch.randn(width, generator=g).bfloat16().cuda()
    old = torch.randn(2, width, 6, generator=g).bfloat16().cuda()
    out, new = torch.empty_like(x), torch.empty_like(old)
    program = compile_causal_depthwise_conv7_aot(width=width)
    program.launch(x, w, bias, old, out, new, scalars=[2, length])
    full = torch.cat((old, x.transpose(1, 2)), -1)
    ref64 = torch.nn.functional.conv1d(full.double(), w[:, None].double(), bias.double(), groups=width).transpose(1, 2)
    terms = torch.nn.functional.conv1d(full.double().abs(), w[:, None].double().abs(), groups=width).transpose(1, 2)
    ulp = torch.exp2(torch.floor(torch.log2(ref64.abs().clamp_min(2.0 ** -126))) - 7)
    bound = terms * (8 * 2.0 ** -24 / (1 - 8 * 2.0 ** -24)) + ulp
    assert ((out.double() - ref64).abs() <= bound).all()
    assert torch.equal(new, full[..., -6:])
    # In-place old/new history has the same exact snapshot semantics.
    in_place = old.clone()
    again = torch.empty_like(out)
    program.launch(x, w, bias, in_place, again, in_place, scalars=[2, length])
    assert torch.equal(again, out) and torch.equal(in_place, new)
    # Every row sees the same causal history under singleton streaming chunks.
    state = old.clone()
    pieces = []
    for at in range(length):
        chunk = x[:, at:at+1].contiguous()
        y = torch.empty_like(chunk)
        program.launch(chunk, w, bias, state, y, state, scalars=[2, 1])
        pieces.append(y)
    assert torch.equal(torch.cat(pieces, 1), out)
    assert torch.equal(state, new)


@pytest.mark.parametrize("rows", ROWS)
def test_pixtral_rms_boundary(rows):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    g = torch.Generator().manual_seed(126)
    x = torch.randn(rows, 2048, generator=g).bfloat16().cuda()
    gamma = (torch.randn(2048, generator=g) * .2 + 1).bfloat16().cuda()
    ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * gamma
    out = torch.empty_like(x)
    program = compile_affine_norm_aot(width=2048, input_dtype=torch.bfloat16,
        parameter_dtype=torch.bfloat16, round_normalized_before_weight=True)
    program.launch(x, gamma, None, None, None, None, out, scalars=[rows])
    _ulp_close(out, ref)
    one = torch.empty_like(out[:1])
    program.launch(x[:1], gamma, None, None, None, None, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_existing_biased_matmul_epilogue_rounds_only_after_bias(rows):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_biased_projection_aot
    # Exactly representable dyadic products/sums isolate the epilogue boundary
    # from permitted dot-product reduction ordering.
    g = torch.Generator().manual_seed(129)
    x = (torch.randint(-4, 5, (rows, 16), generator=g).float() / 16).bfloat16().cuda()
    weight = (torch.randint(-4, 5, (256, 16), generator=g).float() / 16).bfloat16().cuda()
    bias = torch.linspace(-.05, .05, 256).bfloat16().cuda()
    # Deterministic midpoint: 1 + 1/256 rounds to 1, while adding another
    # 1/256 before the store must produce the next representable BF16 value.
    x[:, 0], x[:, 1] = 1, 1 / 256
    weight[0] = 0
    weight[0, :2] = 1
    bias[0] = 1 / 256
    expected = (x.float() @ weight.float().T + bias.float()).bfloat16()
    prematurely_rounded = (x.float() @ weight.float().T).bfloat16() + bias
    assert (expected != prematurely_rounded).any()
    out = torch.empty_like(expected)
    program = compile_biased_projection_aot(width=16, outputs=256)
    program.launch(x, weight, bias, out, scalars=[rows])
    assert torch.equal(out, expected)
    one = torch.empty_like(out[:1])
    program.launch(x[:1], weight, bias, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("width", (256, 512, 1024, 2048))
@pytest.mark.parametrize("rows", ROWS)
def test_audio_builder_residual_is_bf16(width, rows):
    require_b12x()
    from b12x.integration.cuteafd.media import audio_programs
    program = audio_programs()[f"residual_h{width}"]()
    assert all(operand.dtype == torch.bfloat16 for operand in program.operands)
    a = torch.randn(rows + 1, width, device="cuda").bfloat16()
    b = torch.randn_like(a)
    out = torch.full_like(a, 42)
    program.launch(a, b, out, scalars=[rows])
    assert torch.equal(out[:rows], a[:rows] + b[:rows])
    assert (out[rows:] == 42).all()
    one = torch.empty_like(out[:1])
    program.launch(a, b, one, scalars=[1])
    assert torch.equal(one, out[:1])
    program.launch(a, b, a, scalars=[rows])
    assert torch.equal(a[:rows], out[:rows])


@pytest.mark.parametrize("width", (1024, 2048, 4096, 8192))
def test_existing_gelu_audio_shapes(width):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_gelu_aot
    x = torch.linspace(-8, 8, width * 16, device="cuda").reshape(16, width).bfloat16()
    out = torch.empty_like(x)
    program = compile_gelu_aot(width=width)
    program.launch(x, out, scalars=[16])
    _ulp_close(out, torch.nn.functional.gelu(x))
    one = torch.empty_like(x[:1])
    program.launch(x[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("width", (256, 512, 1024, 2048, 8192))
def test_audio_plain_rms_shapes(width):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    g = torch.Generator().manual_seed(130)
    x = torch.randn(16, width, generator=g).bfloat16().cuda()
    gamma = torch.randn(width, generator=g).bfloat16().cuda()
    ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
           * gamma.float()).bfloat16()
    out = torch.empty_like(x)
    program = compile_affine_norm_aot(width=width, eps=1e-6,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16)
    program.launch(x, gamma, None, None, None, None, out, scalars=[16])
    _ulp_close(out, ref)


@pytest.mark.parametrize("width,outputs", [(588, 2048), (1536, 8960), (8960, 1536),
    (16, 256), (1536, 512), (2560, 1024), (4096, 2048), (2048, 8192)])
def test_existing_biased_projection_tower_shapes(width, outputs):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_biased_projection_aot
    # Two exact nonzero terms retain the bias midpoint even across K tiling.
    x = torch.zeros(16, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(outputs, width, device="cuda", dtype=torch.bfloat16)
    x[:, 0], x[:, -1] = 1, 1 / 256
    weight[:, 0], weight[:, -1] = 1, 1
    bias = torch.full((outputs,), 1 / 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(16, outputs, device="cuda", dtype=torch.bfloat16)
    program = compile_biased_projection_aot(width=width, outputs=outputs)
    program.launch(x, weight, bias, out, scalars=[16])
    assert (out == 1 + 1 / 128).all()


@pytest.mark.parametrize("rows", (1, 16, 80))
def test_patch_projection_matches_single_bf16_store_and_live_rows(rows):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.media import compile_patch_projection_aot
    width, outputs = 1176, 1536
    g = torch.Generator().manual_seed(132)
    x = (torch.randint(-4, 5, (rows, width), generator=g).float() / 16).bfloat16().cuda()
    weight = (torch.randint(-4, 5, (outputs, width), generator=g).float() / 16).bfloat16().cuda()
    # The K-tail term moves this midpoint up; an intermediate BF16 cast loses it.
    x[:, 0], x[:, 1], x[:, -1] = 1, 1 / 256, 1 / 256
    weight[0] = 0
    weight[0, 0], weight[0, 1], weight[0, -1] = 1, 1, 1
    expected = (x.float() @ weight.float().T).bfloat16()
    assert (expected[:, 0] == 1 + 1 / 128).all()
    program = compile_patch_projection_aot(width=width, outputs=outputs)
    out = torch.full((rows + 1, outputs), 42, device="cuda", dtype=torch.bfloat16)
    program.launch(x, weight, out, scalars=[rows])
    assert torch.equal(out[:rows], expected)
    assert (out[rows:] == 42).all()
    with kernel_resolution_guard("vision patch projection live extents"):
        for live in dict.fromkeys((1, min(rows, 16), rows)):
            program.launch(x, weight, out, scalars=[live])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(x, weight, out, scalars=[live])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:live], expected[:live])


def test_patch_projection_random_reference():
    require_b12x()
    from b12x.integration.cuteafd.media import compile_patch_projection_aot
    g = torch.Generator().manual_seed(133)
    x = torch.randn(16, 1176, generator=g).bfloat16().cuda()
    weight = torch.randn(1536, 1176, generator=g).bfloat16().cuda()
    out = torch.empty(16, 1536, device="cuda", dtype=torch.bfloat16)
    program = compile_patch_projection_aot(width=1176, outputs=1536)
    program.launch(x, weight, out, scalars=[16])
    ref64 = x.double() @ weight.double().T
    terms = x.double().abs() @ weight.double().abs().T
    ulp = torch.exp2(torch.floor(torch.log2(ref64.abs().clamp_min(2.0 ** -126))) - 7)
    error = terms * (1176 * 2.0 ** -24 / (1 - 1176 * 2.0 ** -24)) + ulp
    assert ((out.double() - ref64).abs() <= error).all()


@pytest.mark.parametrize("height,width", [(2, 2), (4, 6), (6, 4), (2, 128), (128, 2), (128, 128)])
def test_spatial_merge_adapter_matches_official_channel_major_pixel_shuffle(height, width):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.media import compile_spatial_merge_aot
    hidden, ratio = 1536, 2
    g = torch.Generator().manual_seed(131)
    x = torch.randn(height * width, hidden, generator=g).bfloat16().cuda()
    # Official Muse get_vision_pixel_shuffle_index followed by pixel_shuffle:
    # group raster spatial blocks first, then transpose C ahead of their 4 rows.
    index = torch.arange(height * width, device="cuda")
    index = index.view(height // ratio, ratio, width // ratio, ratio)
    index = index.permute(0, 2, 1, 3).reshape(-1)
    expected = x[index].view(-1, ratio * ratio, hidden).permute(0, 2, 1)
    expected = expected.reshape(-1, hidden * ratio * ratio)
    rows = expected.shape[0]
    out = torch.full((rows + 1, hidden * ratio * ratio), 42, device="cuda", dtype=torch.bfloat16)
    program = compile_spatial_merge_aot(width=hidden, ratio=ratio)
    program.launch(x, out, scalars=[rows, height, width])
    assert torch.equal(out[:rows], expected)
    assert (out[rows:] == 42).all()
    with kernel_resolution_guard("vision spatial merge live extents"):
        for live in dict.fromkeys((1, min(rows, 16), rows)):
            program.launch(x, out, scalars=[live, height, width])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(x, out, scalars=[live, height, width])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:live], expected[:live])


@pytest.mark.parametrize("dim,windowed", [(96, False), (96, True), (128, False)])
def test_vision_attention_segments(dim, windowed):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_vision_attention_aot
    g = torch.Generator().manual_seed(127)
    capacity = 16384
    # Non-tile-multiple lengths plus trailing empty segments exercise masking.
    lengths = [1, 16, 64, 80] if windowed else [80]
    n = sum(lengths)
    q, k, v = [torch.randn(capacity, 16, dim, generator=g).bfloat16().cuda() for _ in range(3)]
    out = torch.full_like(q, 42)
    lse = torch.empty(16, capacity, device="cuda")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    segments = 512 if windowed else 1
    offsets += [n] * (segments + 1 - len(offsets))
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    program = compile_vision_attention_aot(dim=dim, windowed=windowed)
    program.launch(q, k, v, out, lse, cu, scalars=[dim ** -.5])
    for start, end in zip(offsets[:len(lengths)], offsets[1:len(lengths)+1]):
        qq, kk, vv = [t[start:end].double().transpose(0, 1) for t in (q, k, v)]
        p = (qq @ kk.transpose(-1, -2) * (dim ** -.5)).softmax(-1)
        ref = (p @ vv).transpose(0, 1)
        terms = (p @ vv.abs()).transpose(0, 1)
        ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -126))) - 7)
        assert ((out[start:end].double() - ref).abs() <= terms * (2.0 ** -7) + ulp).all()
    assert (out[n:] == 42).all()
    # Same first request in a different wave mix: padded empty segments vs full wave.
    if windowed:
        singleton = out.clone().fill_(42)
        one_cu = torch.tensor([0, lengths[0]] + [lengths[0]] * 511, device="cuda", dtype=torch.int32)
        program.launch(q, k, v, singleton, lse, one_cu, scalars=[dim ** -.5])
        assert torch.equal(singleton[:lengths[0]], out[:lengths[0]])
