"""Raw paged index gather: byte oracle, bounds, padding and graph replay."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("large_offset", [False, True])
def test_index_cache_gather_exact_bytes_bounds_padding_and_replay(large_offset):
    device = require_b12x()
    from b12x.integration.cuteafd.index_cache import compile_index_cache_gather_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    program = compile_index_cache_gather_aot()
    high = 2**31 // 8448 + 1 if large_offset else 2
    pool_pages = high + 2
    cache = torch.empty((pool_pages, 8448), dtype=torch.uint8, device=device)
    # Arbitrary raw bits cover signed zero, non-power-of-two scales and NaNs;
    # staging must never interpret or canonicalize them.
    payload = (torch.arange(8448, device=device, dtype=torch.int32) * 37 + 19).to(torch.uint8)
    cache[high].copy_(payload)
    cache[1].copy_(payload.flip(0))
    pages = torch.tensor([high, -1, pool_pages, 1, 2**31 - 1, high + 1],
                         dtype=torch.int32, device=device)
    lengths = torch.tensor([-17, 0, 1, 63, 64, 65, 257, 4000], dtype=torch.int32, device=device)
    keys = torch.empty((768, 128), dtype=torch.uint8, device=device)
    scales = torch.empty(768, dtype=torch.float32, device=device)
    starts = torch.empty(8, dtype=torch.int32, device=device)
    ends = torch.empty_like(starts)
    operands = (cache, pages, lengths, keys, scales, starts, ends)

    def reset():
        keys.fill_(173)
        scales.view(torch.uint8).fill_(173)
        starts.fill_(-333)
        ends.fill_(-333)

    def check(rows, width, begin, live, extent):
        expected_keys = torch.zeros((extent, 128), dtype=torch.uint8)
        expected_scales = torch.zeros((extent, 4), dtype=torch.uint8)
        table = pages.cpu().tolist()
        for token in range(live):
            column = begin + token // 64
            if 0 <= column < width and 0 <= table[column] < pool_pages:
                source = cache[table[column]].cpu()
                slot = token % 64
                expected_keys[token] = source[slot * 128:(slot + 1) * 128]
                expected_scales[token] = source[8192 + slot * 4:8192 + (slot + 1) * 4]
        assert torch.equal(keys[:extent].cpu(), expected_keys)
        assert torch.equal(scales[:extent].view(torch.uint8).reshape(-1, 4).cpu(), expected_scales)
        assert (keys[extent:] == 173).all()
        assert (scales[extent:].view(torch.uint8) == 173).all()
        expected_ends = (lengths[:rows].cpu().long() - begin * 64).clamp(0, live).int()
        assert torch.equal(ends[:rows].cpu(), expected_ends)
        assert (starts[:rows] == 0).all()
        assert (starts[rows:] == -333).all() and (ends[rows:] == -333).all()

    cases = [(8, 6, 0, 257, 512), (3, 5, 0, 0, 256), (1, 6, 3, 73, 256),
             (7, 0, 0, 31, 256), (8, 6, -1, 127, 256), (5, 6, 5, 0, 256)]
    # First call resolves the module; all subsequent shapes reuse it.
    program.launch(*operands, scalars=(8, 6, 0, pool_pages, 257, 512))
    with kernel_resolution_guard("bounded shared index gather live scalars and graph"):
        for rows, width, begin, live, extent in cases:
            reset()
            program.launch(*operands, scalars=(rows, width, begin, pool_pages, live, extent))
            check(rows, width, begin, live, extent)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*operands, scalars=(8, 6, 0, pool_pages, 257, 512))
        cache[high].bitwise_xor_(91)
        pages[1] = 1
        lengths.add_(7)
        graph.replay()
        check(8, 6, 0, 257, 512)
