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


class Case:
    def __init__(self, snapshot, layer=40, widths=(2, 3, 4), *, device="cuda"):
        self.ref = reference_module()
        self.h, self.u, self.topk, self.max_groups = 6144, 3, 2, 128
        self.device = torch.device(device)
        self.widths = widths
        self.owners = []
        self.decoded = {}
        pointers = torch.zeros((4, self.u, 3, 3), dtype=torch.int64)
        bits = torch.zeros((4, self.u, 3), dtype=torch.int32)
        index = self.ref.Tensors(Path(snapshot) / "experts" / f"exlr8-layer-{layer:03}.index.safetensors")
        self.checksums = []
        for e, source in enumerate((7, 11, 23, 256)):
            for u, width in enumerate(widths):
                if e == 1 and u == 1:  # All three projections of one unit are nonresident.
                    continue
                if source == 256 and width == 2:
                    width = 3  # This checkpoint ships no K2 shared expert.
                for p, projection in enumerate(("gate", "up", "down")):
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
        self.x = torch.from_numpy((rng.standard_normal((64, self.h))*0.02).astype(np.float16)).to(self.device)
        self.weights = torch.from_numpy(rng.uniform(.05, .3, (64, self.topk)).astype(np.float32)).to(self.device)
        s = (self.h + 1023)//1024
        self.fc1 = torch.empty((self.max_groups, self.u, 2, s, 16, 128), device=self.device)
        self.intermediate = torch.empty((self.max_groups, self.u, 16, 128), dtype=torch.float16, device=self.device)
        self.fc2 = torch.empty((self.max_groups, 16, self.h), device=self.device)
        self.output = torch.empty((64, self.h), device=self.device)

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
        self.group_expert = torch.full((self.max_groups,), -1, dtype=torch.int32, device=self.device)
        self.route_slots = torch.full((self.max_groups, 16), -1, dtype=torch.int32, device=self.device)
        self.group_expert[:self.groups] = torch.tensor(expert, dtype=torch.int32, device=self.device)
        self.route_slots[:self.groups] = torch.tensor(slots, dtype=torch.int32, device=self.device)
        self.inverse = torch.from_numpy(inverse).to(self.device)
        return self.groups

    def tensors(self):
        return (self.x, self.pointers, self.bits, self.group_expert, self.route_slots,
                self.inverse, self.weights, self.fc1, self.intermediate, self.fc2, self.output)

    def oracle(self, rows, *, fp64=False, round_intermediate=True):
        # Single-row products intentionally avoid batch-size dependent BLAS choices.
        dtype = torch.float64 if fp64 else torch.float32
        output = torch.zeros((rows, self.h), dtype=dtype, device=self.device)
        for r in range(rows):
            for slot in range(self.topk):
                e = (r+slot) % 4
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
