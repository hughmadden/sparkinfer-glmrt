"""EXLR8 A16 unit-expert ABI v3; all tables and scratch belong to the caller.

Input is the rotated F16 activation the grouping step writes (H128 with
signs.in applied once per token); output is the BF16 hidden-domain partial
of the local experts (summed, inverse H128 and signs.out applied once).
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program

ABI_VERSION = 3
# Launch hint, not math: callers pass cta_limit = cudaDevAttr SM count * this.
# Mirrors Exlr8A16Kernel.CTAS_PER_SM (kept literal so the contract stays CUDA-free).
CTAS_PER_SM = 4
POINTERS = ("x", "pointers", "bits", "group_expert", "route_slots",
            "route_group_lane", "route_weights", "fc1", "intermediate", "fc2",
            "signs_out", "output")


def geometry(hidden=6144, units=16, top_k=8, max_groups=512):
    values = (hidden, units, top_k, max_groups)
    if any(type(v) is not int or v <= 0 or v > 2**31 - 1 for v in values) or hidden % 128:
        raise ValueError("EXLR8 requires positive int32 geometry and hidden % 128 == 0")
    result = dict(abi_version=ABI_VERSION, hidden=hidden, units=units, top_k=top_k,
                  max_groups=max_groups, max_rows=64, split_k_blocks=8,
                  fc1_splits=(hidden + 1023) // 1024, ctas_per_sm=CTAS_PER_SM)
    tasks = (max_groups * units * 2 * result["fc1_splits"],
             max_groups * units, max_groups * (hidden // 128), 64 * (hidden // 128))
    # Grid-stride increments may reach almost twice the task count before exiting.
    if max(tasks) > (2**31 - 1) // 2 or max(64 * top_k, max_groups * 16) > 2**31 - 1:
        raise ValueError("EXLR8 task or route indexing exceeds signed int32")
    if max(scratch_bytes(result).values()) > 2**63 - 1:
        raise ValueError("EXLR8 scratch exceeds signed 64-bit addressing")
    return result


def scratch_bytes(g):
    groups, units, hidden = g["max_groups"], g["units"], g["hidden"]
    return dict(fc1=groups * units * 2 * g["fc1_splits"] * g.get("group_rows", 16) * 128 * 4,
                intermediate=groups * units * g.get("group_rows", 16) * 128 * 2,
                fc2=groups * g.get("group_rows", 16) * hidden * 4)


class _Exlr8Pointers:
    def __init__(self, g):
        from b12x.moe._shared.kernels.w4a16.exlr8 import Exlr8A16Kernel
        self.core = Exlr8A16Kernel(g["hidden"], g["units"], g["top_k"], g["max_groups"])

    @cute.jit
    def __call__(self, x: cute.Pointer, pointers: cute.Pointer, bits: cute.Pointer,
                 group_expert: cute.Pointer, route_slots: cute.Pointer,
                 route_group_lane: cute.Pointer, route_weights: cute.Pointer,
                 fc1: cute.Pointer, intermediate: cute.Pointer, fc2: cute.Pointer,
                 signs_out: cute.Pointer, output: cute.Pointer, rows: cutlass.Int32,
                 groups: cutlass.Int32,
                 cta_limit: cutlass.Int32, stream: cuda.CUstream):
        # Flat views are address carriers; the core computes all runtime offsets.
        self.core(cute.make_tensor(x, cute.make_layout(1, stride=1)),
                  cute.make_tensor(pointers, cute.make_layout(1, stride=1)),
                  cute.make_tensor(bits, cute.make_layout(1, stride=1)),
                  cute.make_tensor(group_expert, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_slots, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_group_lane, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_weights, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc1, cute.make_layout(1, stride=1)),
                  cute.make_tensor(intermediate, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc2, cute.make_layout(1, stride=1)),
                  cute.make_tensor(signs_out, cute.make_layout(1, stride=1)),
                  cute.make_tensor(output, cute.make_layout(1, stride=1)), rows, groups, cta_limit, stream)


class _Exlr8ServingPointers:
    def __init__(self, g):
        from b12x.moe._shared.kernels.w4a16.exlr8 import Exlr8A16Kernel
        if g.get("max_rows", 64) > 64:
            from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
            self.core = Exlr8GroupedKernel(g["hidden"], g["units"], g["top_k"],
                                           g["max_groups"], g["group_rows"])
        else:
            self.core = Exlr8A16Kernel(g["hidden"], g["units"], g["top_k"],
                                      g["max_groups"], device_count=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, pointers: cute.Pointer, bits: cute.Pointer,
                 group_expert: cute.Pointer, route_slots: cute.Pointer,
                 route_group_lane: cute.Pointer, route_weights: cute.Pointer,
                 fc1: cute.Pointer, intermediate: cute.Pointer, fc2: cute.Pointer,
                 signs_out: cute.Pointer, output: cute.Pointer, group_count: cute.Pointer,
                 rows: cutlass.Int32, cta_limit: cutlass.Int32, stream: cuda.CUstream):
        self.core.serving(cute.make_tensor(x, cute.make_layout(1, stride=1)),
                  cute.make_tensor(pointers, cute.make_layout(1, stride=1)),
                  cute.make_tensor(bits, cute.make_layout(1, stride=1)),
                  cute.make_tensor(group_expert, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_slots, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_group_lane, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_weights, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc1, cute.make_layout(1, stride=1)),
                  cute.make_tensor(intermediate, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc2, cute.make_layout(1, stride=1)),
                  cute.make_tensor(signs_out, cute.make_layout(1, stride=1)),
                  cute.make_tensor(output, cute.make_layout(1, stride=1)), rows,
                  cute.make_tensor(group_count, cute.make_layout(1, stride=1)), cta_limit, stream)


class _Exlr8GroupedServingPointers:
    """Grouped ABI plus the compact owned-unit tables and runtime unit limit.

    `unit_count[E]` and `unit_map[E,U]` describe, per expert, which unit slots
    this rank actually owns, in ascending unit order. `unit_limit` is their
    runtime maximum; it bounds the FC1/activation grids and the compact scratch
    stride. Descriptor tables stay global, so unowned slots keep their zero
    `bits` entries and the host descriptor builder is unchanged.
    """

    def __init__(self, g):
        from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
        self.core = Exlr8GroupedKernel(g["hidden"], g["units"], g["top_k"],
                                       g["max_groups"], g["group_rows"],
                                       activations=g.get("activations", "a16"),
                                       quantize_pass=g.get("quantize_pass", True))

    @cute.jit
    def __call__(self, x: cute.Pointer, pointers: cute.Pointer, bits: cute.Pointer,
                 group_expert: cute.Pointer, route_slots: cute.Pointer,
                 route_group_lane: cute.Pointer, route_weights: cute.Pointer,
                 fc1: cute.Pointer, intermediate: cute.Pointer, fc2: cute.Pointer,
                 signs_out: cute.Pointer, output: cute.Pointer, group_count: cute.Pointer,
                 unit_count: cute.Pointer, unit_map: cute.Pointer,
                 rows: cutlass.Int32, cta_limit: cutlass.Int32,
                 unit_limit: cutlass.Int32, stream: cuda.CUstream):
        self.core.serving(cute.make_tensor(x, cute.make_layout(1, stride=1)),
                  cute.make_tensor(pointers, cute.make_layout(1, stride=1)),
                  cute.make_tensor(bits, cute.make_layout(1, stride=1)),
                  cute.make_tensor(group_expert, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_slots, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_group_lane, cute.make_layout(1, stride=1)),
                  cute.make_tensor(route_weights, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc1, cute.make_layout(1, stride=1)),
                  cute.make_tensor(intermediate, cute.make_layout(1, stride=1)),
                  cute.make_tensor(fc2, cute.make_layout(1, stride=1)),
                  cute.make_tensor(signs_out, cute.make_layout(1, stride=1)),
                  cute.make_tensor(output, cute.make_layout(1, stride=1)),
                  rows,
                  cute.make_tensor(group_count, cute.make_layout(1, stride=1)),
                  cute.make_tensor(unit_count, cute.make_layout(1, stride=1)),
                  cute.make_tensor(unit_map, cute.make_layout(1, stride=1)),
                  cta_limit, unit_limit, stream)


def compile_exlr8_a16_aot(*, hidden=6144, units=16, top_k=8, max_groups=512,
                         device_count=False, shared_expert=False, grouped=False, group_rows=64,
                         activations="a16", quantize_pass=True):
    if activations not in ("a16", "a8") or (activations == "a8" and not grouped):
        raise ValueError("EXLR8 A8 activations require grouped prefill")
    if shared_expert:
        if not device_count:
            raise ValueError("shared EXLR8 requires device-count serving ABI")
        if top_k != 1 or max_groups != 4:
            raise ValueError("shared EXLR8 requires top_k=1 and max_groups=4")
    g = geometry(hidden, units, top_k, max_groups)
    if grouped:
        if not device_count or shared_expert or group_rows not in (16, 32, 64):
            raise ValueError("EXLR8 grouped prefill requires device count and group rows 16/32/64")
        from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
        g.update(max_rows=4096, group_rows=group_rows,
                 ctas_per_sm=Exlr8GroupedKernel.CTAS_PER_SM)
        if activations != "a16":
            g["activations"] = activations
            if not quantize_pass:
                g["quantize_pass"] = False
        tasks = (max_groups * units * 2 * g["fc1_splits"] * 4,
                 max_groups * (hidden // 128) * 4, 4096 * (hidden // 128))
        if max(tasks) > (2**31 - 1) // 2 or max_groups * group_rows > 2**31 - 1:
            raise ValueError("EXLR8 grouped task or route indexing exceeds signed int32")
        if max(scratch_bytes(g).values()) > 2**63 - 1:
            raise ValueError("EXLR8 grouped scratch exceeds signed 64-bit addressing")
    r = g.get("group_rows", 16)
    h, u, t, m, s = (g[k] for k in ("hidden", "units", "top_k", "max_groups", "fc1_splits"))
    operands = (
        Operand("x", torch.float16, f"[rows,{h}]"),
        Operand("pointers", torch.int64, f"[E,{u},3,3]"),
        Operand("bits", torch.int32, f"[E,{u},3]"),
        Operand("group_expert", torch.int32, f"[{m}]"),
        Operand("route_slots", torch.int32, f"[{m},{r}]"),
        Operand("route_group_lane", torch.int32, f"[rows,{t}]"),
        Operand("route_weights", torch.float32, f"[rows,{t}]"),
        Operand("fc1", torch.float32, f"[{m},{u},2,{s},{r},128]", "scratch"),
        Operand("intermediate", torch.float16, f"[{m},{u},{r},128]", "scratch"),
        Operand("fc2", torch.float32, f"[{m},{r},{h}]", "scratch"),
        Operand("signs_out", torch.int8, f"[{h}]", align=4),
        Operand("output", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    sizes = scratch_bytes(g)
    if device_count:
        operands += (Operand("group_count", torch.int32, "[1]"),)
    if grouped:
        # Compact owned-unit tables appended after the existing prefix.
        operands += (Operand("unit_count", torch.int32, "[E]"),
                     Operand("unit_map", torch.int32, f"[E,{u}]"))
    return compile_program(_Exlr8GroupedServingPointers(g) if grouped else
                           _Exlr8ServingPointers(g) if device_count else _Exlr8Pointers(g),
                           name=(f"exlr8_{activations}_grouped" if grouped else
                                 "exlr8_shared_a16_serving" if shared_expert else
                                 "exlr8_a16_serving" if device_count else "exlr8_a16_decode"), operands=operands,
                           scalars=tuple(Scalar(n) for n in (("rows", "cta_limit", "unit_limit")
                                                            if grouped else
                                                            ("rows", "cta_limit") if device_count
                                                            else ("rows", "groups", "cta_limit"))),
                           key=((h, u, t, m, g["abi_version"], shared_expert, grouped, r,
                                 # The AOT spec is keyed on this tuple, not on kernel
                                 # source: bump the tag when decode's weight-load
                                 # cache policy changes.
                                 "decode_l2_fc2_prefetch_v9")
                                + (("grouped_compact_units",) if grouped else ())
                                + ((activations,) if grouped and activations != "a16" else ())
                                + (("fused_quantize",) if grouped and activations != "a16" and not quantize_pass else ())), geometry=g,
                           scratch={n: lambda rows, size=size: size for n, size in sizes.items()},
                           doc=__doc__)
