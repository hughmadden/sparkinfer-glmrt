"""SM89 probe 5: V4.1 FP4 overlay index scorer (``V41OverlayScore``).

1. ``fp4_decode_2`` for all 256 bytes against the E2M1 table (pre-Blackwell
   targets take the prmt decode: ``cvt.*.e2m1x2`` is SM100+).
2. The scorer, compiled as ``export_b12x_v41_index_aot.py`` compiles it, on
   the fixtures of tests/attention/test_v41_overlay_score.py (valid committed
   and proposal rows, masked positions, a bad page table, a malformed
   descriptor, a stale start, a masked causal window, the last strided
   proposal and an unsigned causal bound) against a torch reference that
   follows the kernel's masks, BF16 roundings and FP32 head-sum tree; only
   the 128-wide dot products are float64 there.

Usage: q5_v41_index_sm89.py
"""

import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics import fp4_decode_2
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention.dsa_indexer._v41_overlay import V41OverlayScore

DTYPES = [Uint8, Uint8, BFloat16, Uint8, Uint8, Uint32, Int64, Int64, Int64, Float32, Uint8, Uint8]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class DecodeAll:
    @cute.jit
    def __call__(self, out: cute.Pointer, stream: cuda.CUstream):
        self.kernel(out).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, out: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        tensor = cute.make_tensor(out, cute.make_layout(256))
        tensor[tidx] = fp4_decode_2(Uint32(tidx))


def check_decode():
    out = torch.zeros(256, device="cuda", dtype=torch.int32)
    compiled = compile_kernel(DecodeAll(), make_ptr(Uint32, out.data_ptr(), cute.AddressSpace.gmem,
                                                    assumed_align=16), current_cuda_stream())
    compiled(make_ptr(Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
             current_cuda_stream())
    torch.cuda.synchronize()
    halves = out.cpu().view(torch.int16).view(torch.float16).reshape(256, 2)
    byte = torch.arange(256)
    table = torch.tensor(E2M1, dtype=torch.float16)
    expected = torch.stack([table[byte & 15], table[byte >> 4]], dim=1)
    # Compare bit patterns: -0.0 must stay -0.0.
    ok = torch.equal(halves.view(torch.int16), expected.view(torch.int16))
    print(f"fp4_decode_2 all 256 bytes bit-exact={ok}", flush=True)
    return ok


def inputs(rows, width):
    torch.manual_seed(4191 + rows + width)
    q = torch.randint(256, (rows, 32, 64), device="cuda", dtype=torch.uint8)
    qs = torch.randint(120, 128, (rows, 32, 4), device="cuda", dtype=torch.uint8)
    w = torch.randn((rows, 32), device="cuda", dtype=torch.bfloat16)
    k = torch.randint(256, (256, 64), device="cuda", dtype=torch.uint8)
    ks = torch.randint(120, 128, (256, 4), device="cuda", dtype=torch.uint8)
    pages = torch.zeros(1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([256], device="cuda", dtype=torch.int64)
    meta = torch.tensor([0, 384, 256, 128, 0, 2], device="cuda", dtype=torch.int64).expand(rows, 6).contiguous()
    positions = torch.arange(width, device="cuda", dtype=torch.int64).expand(rows, width).contiguous()
    positions[:, 1::67] = -1
    out = torch.empty((rows, width), device="cuda")
    p = torch.randint(256, (256, 64), device="cuda", dtype=torch.uint8)
    ps = torch.randint(120, 128, (256, 4), device="cuda", dtype=torch.uint8)
    return [q, qs, w, k, ks, pages, lengths, meta, positions, out, p, ps]


def dequant(data, scales):
    """[..., 64] FP4 bytes and [..., 4] E8M0 scales -> BF16-rounded FP32 [..., 128]."""
    table = torch.tensor(E2M1, device=data.device)
    values = torch.stack([table[(data & 15).long()], table[(data >> 4).long()]], dim=-1)
    values = values.reshape(*data.shape[:-1], 128)
    sf = scales.long()
    scale = torch.exp2((sf - 127).double()).float()
    scale = torch.where(sf == 0, torch.tensor(2.0**-127, device=data.device), scale)
    return (values * scale.repeat_interleave(32, dim=-1)).bfloat16().float()


def bf16(x):
    return x.bfloat16().float()


def reference(t, slots, stride, capacity, proposal_capacity):
    q, qs, w, k, ks, pages, lengths, meta, positions, _, p, ps = t
    rows, width = positions.shape
    slot, causal, start, count, offset, step = meta.unbind(1)
    slot_ok = (slot >= 0) & (slot < slots)
    committed = lengths[slot.clamp(0, slots - 1)]
    desc = ((start == committed) & (start >= 0) & (start <= 1048576) & (count >= 0)
            & (count <= 1048576 - start) & ((step == 1) | (step == 2)) & (offset >= 0)
            & (offset <= proposal_capacity))
    room = (proposal_capacity - 1 - offset).clamp_min(0) // step.clamp_min(1)
    with_count = (offset < proposal_capacity) & ((step <= 0) | (count - 1 <= room))
    desc &= (count <= 0) | with_count
    pos = positions
    live = (slot_ok & desc)[:, None] & (pos >= 0)
    live &= (causal[:, None] < 0) | (pos < causal[:, None])  # unsigned causal bound
    page_rows = (pages.long() & 0xFFFFFFFF)
    in_table = pos < stride * 256
    page_index = (slot[:, None].clamp_min(0) * stride + (pos.clamp_min(0) // 256)).clamp(0, page_rows.numel() - 1)
    physical = page_rows[page_index] * 256 + pos.clamp_min(0) % 256
    committed_ok = (pos < committed[:, None]) & in_table & (physical >= 0) & (physical < capacity)
    proposal_ok = (pos >= committed[:, None]) & (pos - committed[:, None] < count[:, None])
    proposal_row = offset[:, None] + (pos - committed[:, None]) * step[:, None]
    valid = live & (committed_ok | proposal_ok)
    use_proposal = live & proposal_ok

    keys = dequant(k, ks)          # [256, 128]
    proposals = dequant(p, ps)     # [256, 128]
    queries = dequant(q, qs)       # [rows, 32, 128]
    weights = w.float()            # [rows, 32]
    out = torch.full((rows, width), float("-inf"), device=q.device)
    for r0 in range(0, rows, 64):
        r1 = min(rows, r0 + 64)
        key_rows = torch.where(use_proposal[r0:r1], proposal_row[r0:r1], physical[r0:r1]).clamp(0, 255)
        cand = torch.where(use_proposal[r0:r1, :, None], proposals[key_rows], keys[key_rows])
        dots = torch.einsum("rck,rhk->rch", cand.double(), queries[r0:r1].double()).float()
        vals = bf16(bf16(dots).clamp_min(0) * weights[r0:r1, None, :])  # [r, c, 32]
        # Head h = chunk*8 + pair*2 + half; per pair, then the lane butterfly.
        v = vals.reshape(r1 - r0, width, 4, 4, 2)  # chunk, pair, half
        x = (v[:, :, 0] + v[:, :, 2]) + (v[:, :, 1] + v[:, :, 3])  # [r, c, pair, half]
        x = (x[:, :, 0] + x[:, :, 2]) + (x[:, :, 1] + x[:, :, 3])  # [r, c, half]
        score = bf16(x[..., 0] + x[..., 1])
        out[r0:r1] = torch.where(valid[r0:r1], score, out[r0:r1])
    return out


def main():
    print(f"device={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
          f"cutlass-dsl={cutlass.__version__}", flush=True)
    ok = check_decode()
    raw = compile_kernel(V41OverlayScore(),
        *[make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in DTYPES],
        Int32(1), Int32(1), Int32(1), Int32(1), Int64(256), Int64(1), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.indexer.v41_overlay", 1, ()))
    for rows, width in ((1, 1), (6, 65), (16, 255), (80, 513), (256, 1024), (1024, 4096)):
        t = inputs(rows, width)
        for cycle in range(7):
            if cycle:
                t[0].bitwise_xor_(255)
                t[2].neg_()
            if cycle == 1: t[5].fill_(-1)
            if cycle == 2: t[7][:, 5] = 0
            if cycle == 3:
                t[5].zero_(); t[7][:, 5] = 2; t[7][:, 2] = 255
            if cycle == 4:
                t[7][:, 2] = 256; t[7][:, 1] = 0
            if cycle == 5:
                t[7][:, 1] = 384; t[8][:, 0] = 383
            if cycle == 6: t[7][:, 1] = -1
            t[9].fill_(float("nan"))
            raw(*[make_ptr(dt, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)
                  for x, dt in zip(t, DTYPES)], rows, width, 1, 1, 256, 256, current_cuda_stream())
            torch.cuda.synchronize()
            ref = reference(t, 1, 1, 256, 256)
            out = t[9]
            masks_equal = torch.equal(torch.isinf(out) & (out < 0), torch.isinf(ref) & (ref < 0))
            finite = torch.isfinite(ref)
            diff = (out[finite] - ref[finite]).abs()
            scale = ref[finite].abs().max().item() if finite.any() else 1.0
            rel = (diff.max().item() / scale) if finite.any() else 0.0
            exact = (out[finite] == ref[finite]).float().mean().item() if finite.any() else 1.0
            case_ok = masks_equal and not torch.isnan(out).any() and rel < 1e-2
            ok &= case_ok
            print(f"rows={rows:4d} width={width:4d} cycle={cycle} live={finite.float().mean().item():.3f} "
                  f"masks_equal={masks_equal} maxrel={rel:.2e} bitexact={exact:.4f} ok={case_ok}",
                  flush=True)
    print("Q5", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
