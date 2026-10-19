"""Checkpoint-backed ABI fixtures using the engine's independent E1 decoder."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


def reference_module():
    path = Path(os.environ["CUTEAFD_EXLR8_REFERENCE"])
    spec = importlib.util.spec_from_file_location("exlr8_e1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hadamard(x):
    x = x.clone()
    for width in (1, 2, 4, 8, 16, 32, 64):
        view = x.reshape(*x.shape[:-1], -1, 2, width)
        a, b = view[..., 0, :].clone(), view[..., 1, :].clone()
        view[..., 0, :], view[..., 1, :] = a + b, a - b
    return x * (128.0 ** -0.5)


def hadamard_axis0(x):
    return hadamard(x.transpose(0, 1).contiguous()).transpose(0, 1)


def make_wire(x):
    """Standard Fp8K32 rows: E4M3 bytes then UE8M0 per-32 scale bytes.

    Returns (wire uint8 [rows, h + h/32], exact dequantized float64 [rows, h]).
    """
    rows, h = x.shape
    blocks = x.float().reshape(rows, h // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    e4m3 = (blocks / scale).to(torch.float8_e4m3fn)
    wire = torch.empty((rows, h + h // 32), dtype=torch.uint8, device=x.device)
    wire[:, :h] = e4m3.reshape(rows, h).view(torch.uint8)
    wire[:, h:] = (exponent.reshape(rows, h // 32) + 127).to(torch.uint8)
    dequant = (e4m3.double() * scale.double()).reshape(rows, h)
    return wire, dequant


class Case:
    def __init__(self, snapshot, layer=40, widths=(2, 3, 4), *, device="cuda",
                 hidden=6144, units=3, nonresident=((1, 1),)):
        self.ref = reference_module()
        self.h, self.u, self.topk, self.max_groups = hidden, units, 2, 128
        self.device = torch.device(device)
        self.widths = widths
        self.owners = []
        self.decoded = {}
        pointers = torch.zeros((4, self.u, 3, 3), dtype=torch.int64)
        bits = torch.zeros((4, self.u, 3), dtype=torch.int32)
        index = self.ref.Tensors(Path(snapshot) / "experts" / f"exlr8-layer-{layer:03}.index.safetensors")
        # The layer's hidden-side rotation, shared by every routed and shared expert.
        self.signs_in = torch.from_numpy(index.array("signs.in")[:self.h].astype(np.int8).copy()).to(self.device)
        self.signs_out = torch.from_numpy(index.array("signs.out")[:self.h].astype(np.int8).copy()).to(self.device)
        self.checksums = []
        for e, source in enumerate((7, 11, 23, 256)):
            for u in range(self.u):
                if (e, u) in nonresident:
                    continue
                for p, projection in enumerate(("gate", "up", "down")):
                    width = widths[(u + p) % len(widths)]
                    if source == 256 and width == 2:
                        width = 3  # This checkpoint ships no K2 shared expert.
                    prefix = ("shared." if source == 256 else "") + f"k{width}."
                    source_e, source_u = (0 if source == 256 else source), 5 + u
                    offset, length = index.array(prefix + "index")[source_e, p, source_u]
                    with (Path(snapshot) / "experts" / f"exlr8-layer-{layer:03}.k{width}.safetensors").open("rb") as stream:
                        stream.seek(int(offset))
                        payload = stream.read(int(length))
                    channel, block = ("suh", "sv_block") if p == 2 else ("svh", "su_block")
                    c = index.array(prefix + projection + "." + channel)[source_e, source_u*128:(source_u+1)*128].copy()
                    b = index.array(prefix + projection + "." + block)[source_e].copy()
                    checksum = self.ref.xxhash64(payload + c.tobytes() + b.tobytes())
                    assert checksum == int(index.array(prefix + "checksum")[source_e, p, source_u])
                    self.checksums.append(f"{checksum:016x}")
                    # Native payloads are ordered by hidden K-block (gate/up)
                    # or hidden output block (down), so a shorter hidden prefix
                    # retains the exact tile encoding and channel-side vectors.
                    payload = payload[:self.h * 16 * width]
                    b = b[:self.h // 128]
                    decoded = self.ref.decode_unit(payload, width, projection, self.h)
                    tensors = [torch.from_numpy(a.copy()).to(self.device) for a in
                               (np.frombuffer(payload, dtype=np.uint8), c, b)]
                    self.owners.extend(tensors)
                    pointers[e, u, p] = torch.tensor([t.data_ptr() for t in tensors])
                    bits[e, u, p] = width
                    self.decoded[e, u, p] = (torch.from_numpy(decoded).to(self.device), tensors[1], tensors[2])
        self.pointers, self.bits = pointers.to(self.device), bits.to(self.device)
        self.host_bits = bits.numpy()
        rng = np.random.default_rng(408)
        self.x = torch.from_numpy(rng.standard_normal((64, self.h)).astype(np.float16)).to(self.device)
        self.weights = torch.from_numpy(rng.uniform(.05, .3, (64, self.topk)).astype(np.float32)).to(self.device)
        s = (self.h + 1023)//1024
        self.fc1 = torch.empty((self.max_groups, self.u, 2, s, 16, 128), device=self.device)
        self.intermediate = torch.empty((self.max_groups, self.u, 16, 128), dtype=torch.float16, device=self.device)
        self.fc2 = torch.empty((self.max_groups, 16, self.h), device=self.device)
        self.output = torch.empty((64, self.h), dtype=torch.bfloat16, device=self.device)

    def routing(self, rows, *, chunk=16, reverse=False):
        expert, slots = [], []
        inverse = np.full((64, self.topk), -1, dtype=np.int32)
        order = reversed(range(4)) if reverse else range(4)
        for e in order:
            routes = [r*self.topk + s for r in range(rows) for s in range(self.topk)
                      if (r+s) % 4 == e]
            for start in range(0, len(routes), chunk):
                lane_routes = routes[start:start+chunk]
                g = len(expert)
                expert.append(e)
                slots.append(lane_routes + [-1]*(16-len(lane_routes)))
                for lane, route in enumerate(lane_routes):
                    inverse.flat[route] = 16*g+lane
        self.groups = len(expert)
        # Routing contents change between steps; graph-captured addresses do not.
        if not hasattr(self, "group_expert"):
            self.group_expert = torch.empty((self.max_groups,), dtype=torch.int32, device=self.device)
            self.route_slots = torch.empty((self.max_groups, 16), dtype=torch.int32, device=self.device)
            self.inverse = torch.empty((64, self.topk), dtype=torch.int32, device=self.device)
        self.group_expert.fill_(-1)
        self.route_slots.fill_(-1)
        self.group_expert[:self.groups] = torch.tensor(expert, dtype=torch.int32, device=self.device)
        self.route_slots[:self.groups] = torch.tensor(slots, dtype=torch.int32, device=self.device)
        self.inverse.copy_(torch.from_numpy(inverse))
        return self.groups

    def tensors(self):
        return (self.x, self.pointers, self.bits, self.group_expert, self.route_slots,
                self.inverse, self.weights, self.fc1, self.intermediate, self.fc2,
                self.signs_out, self.output)

    def egress(self, rotated):
        """Hidden-domain partial from a rotated-domain sum: signs.out * H128."""
        h = hadamard(rotated.reshape(*rotated.shape[:-1], -1, 128)).reshape(rotated.shape)
        return h * self.signs_out.to(rotated.dtype)

    def hidden_weights(self, e, dtype=torch.float64):
        """Dense hidden-domain gate/up [h, 128u] and down [128u, h] of expert e
        (resident units only), composed exactly as the format defines them."""
        signs_in, signs_out = self.signs_in.to(dtype), self.signs_out.to(dtype)
        mats = ([], [], [])
        for u in range(self.u):
            if not self.host_bits[e, u, 0]:
                continue
            for p in (0, 1, 2):
                t, c, b = self.decoded[e, u, p]
                w = hadamard_axis0(hadamard(t.to(dtype)))
                side = b.to(dtype).repeat_interleave(128)
                if p < 2:
                    w = w * (signs_in * side)[:, None] * c.to(dtype)[None, :]
                else:
                    w = w * c.to(dtype)[:, None] * (signs_out * side)[None, :]
                mats[p].append(w)
        return torch.cat(mats[0], 1), torch.cat(mats[1], 1), torch.cat(mats[2], 0)

    def hidden_reference(self, x, ids, weights, *, official):
        """Hidden-domain MoE partial of the local experts for dequantized rows.

        official=False: every operation in FP64. official=True: the checkpoint's
        BF16 serving arithmetic (BF16 operands and results per op, FP32 MMA
        accumulation, routed sum in BF16), one row at a time.
        """
        dtype = torch.bfloat16 if official else torch.float64
        mats = {e: self.hidden_weights(e) for e in sorted(set(ids.flatten().tolist()))}
        if official:
            mats = {e: tuple(m.to(dtype) for m in v) for e, v in mats.items()}
        out = torch.zeros((x.shape[0], self.h), dtype=dtype, device=self.device)
        for r in range(x.shape[0]):
            a = x[r:r+1].to(dtype)
            for slot in range(ids.shape[1]):
                gate_w, up_w, down_w = mats[int(ids[r, slot])]
                gate, up = a @ gate_w, a @ up_w
                mid = torch.nn.functional.silu(gate) * up
                out[r] = out[r] + ((mid @ down_w) * weights[r, slot].to(dtype))[0]
        return out

    def oracle(self, rows, *, fp64=False, round_intermediate=True, expert=None):
        # Single-row products intentionally avoid batch-size dependent BLAS choices.
        dtype = torch.float64 if fp64 else torch.float32
        output = torch.zeros((rows, self.h), dtype=dtype, device=self.device)
        for r in range(rows):
            for slot in range(self.topk):
                e = (r+slot) % 4 if expert is None else expert
                route = torch.zeros(self.h, dtype=dtype, device=self.device)
                for u in range(self.u):
                    if not self.host_bits[e, u, 0]:
                        continue
                    projections = []
                    for p in (0, 1):
                        w, c, b = self.decoded[e, u, p]
                        splits = []
                        for start in range(0, self.h//128, 8):
                            total = torch.zeros(128, dtype=dtype, device=self.device)
                            for block in range(start, min(start+8, self.h//128)):
                                a = self.x[r, block*128:(block+1)*128].to(dtype)
                                total = total + (a @ w[block*128:(block+1)*128].to(dtype))*b[block].to(dtype)
                            splits.append(total)
                        total = splits[0]
                        for partial in splits[1:]:
                            total = total + partial
                        projections.append(hadamard(total)*c.to(dtype))
                    gate, up = projections
                    w, c, b = self.decoded[e, u, 2]
                    mid = hadamard((gate*torch.sigmoid(gate)*up)*c.to(dtype))
                    if round_intermediate:
                        mid = mid.half().to(dtype)
                    partial = mid @ w.to(dtype)
                    route = route + partial * b.to(dtype).repeat_interleave(128)
                output[r] = output[r] + route*self.weights[r, slot].to(dtype)
        return output

    def payload_bytes(self, rows):
        seen = {(r+s) % 4 for r in range(rows) for s in range(self.topk)}
        return sum(self.h*16*int(self.host_bits[e,u,p]) for e in seen
                   for u in range(self.u) for p in range(3))
