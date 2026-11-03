"""Pre-Hopper (SM80/SM86/SM89) cp.async mainloop pipeline.

CuTe DSL gates its mbarrier waits, ``cp.async.mbarrier.arrive`` and the
mbarrier init fence to SM90+, although the underlying PTX (``mbarrier.init``,
``mbarrier.arrive``, ``mbarrier.test_wait.parity`` and
``cp.async.mbarrier.arrive.noinc``) exists from SM80. This module provides the
subset of ``cutlass.pipeline.PipelineAsync`` that the warp-specialized
cp.async mainloops use, emitted as inline PTX, so the same producer/consumer
code can run on Ampere and Ada parts.

The producer commits with ``cp.async.mbarrier.arrive.noinc``: each producer
thread's arrival fires when all of its earlier cp.async copies have landed, so
the DMA warp never blocks on ``cp.async.wait_group`` and keeps several stages
in flight. Phases follow ``cutlass.pipeline.make_pipeline_state`` (producer
starts at phase 1, consumer at phase 0).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def target_arch_major_minor() -> tuple[int, int]:
    """(major, minor) of the architecture the DSL is compiling for: the compile
    option or ``CUTE_DSL_ARCH``, else the visible GPU. With neither (a
    CPU-only host constructing kernels) the SM120 default stands."""
    from cutlass.cutlass_dsl import CuTeDSL

    try:
        arch = CuTeDSL._get_dsl().get_arch_enum()
    except Exception:
        return 12, 0
    return int(arch.major), int(arch.minor)


def target_is_pre_hopper() -> bool:
    return target_arch_major_minor()[0] < 9


def target_smem_arch() -> str:
    """Arch key for ``cutlass.utils.get_smem_capacity_in_bytes``."""
    major, minor = target_arch_major_minor()
    return "sm_120" if major >= 12 else f"sm_{major}{minor}"


@dsl_user_op
def _mbar_init_by_thread0(addr: Int32, count: Int32, tidx: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip),
         Int32(tidx).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared.b64 [$0], $1; }",
        "r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive_lane0(addr: Int32, *, loc=None, ip=None) -> None:
    # One arrival per warp: converge the warp, then lane 0 arrives. The
    # warp's earlier shared-memory reads (ldmatrix is warp-collective) are
    # complete, and the arrive has release semantics.
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; .reg .b32 lane; .reg .b64 st; "
        "bar.warp.sync -1; "
        "mov.u32 lane, %laneid; "
        "setp.eq.u32 p, lane, 0; "
        "@p mbarrier.arrive.shared.b64 st, [$0]; }",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_mbar_arrive_noinc(addr: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared.b64 [$0];",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_wait_parity(addr: Int32, phase: Int32, *, loc=None, ip=None) -> None:
    # Labels are scoped by the enclosing braces, so repeated expansion is legal.
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; "
        "LAB_WAIT: "
        "mbarrier.test_wait.parity.shared.b64 p, [$0], $1; "
        "@!p bra LAB_WAIT; }",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class Sm80CpAsyncPipeline:
    """mbarrier ring for one cp.async producer warp (one arrival per producer
    thread) and ``consumer_warps`` consumer warps (one elected arrival per
    warp); full[i] at ``base + 8*i``, empty[i] after the fulls."""

    def __init__(self, base_addr: Int32, num_stages: int):
        self.base_addr = base_addr
        self.num_stages = num_stages

    # Carried through dynamic if/while regions like cutlass.pipeline objects.
    def __extract_mlir_values__(self):
        return [self.base_addr.ir_value()]

    def __new_from_mlir_values__(self, values):
        return Sm80CpAsyncPipeline(Int32(values[0]), self.num_stages)

    @staticmethod
    def create(*, num_stages: int, producer_threads: int, consumer_warps: int,
               barrier_storage: cute.Pointer, tidx) -> "Sm80CpAsyncPipeline":
        from b12x._lib.intrinsics import shared_ptr_to_u32

        base = shared_ptr_to_u32(barrier_storage)
        for i in range(num_stages):
            _mbar_init_by_thread0(base + Int32(8 * i), Int32(producer_threads), Int32(tidx))
            _mbar_init_by_thread0(
                base + Int32(8 * (num_stages + i)), Int32(consumer_warps), Int32(tidx)
            )
        cute.arch.sync_threads()
        return Sm80CpAsyncPipeline(base, num_stages)

    def _full(self, index):
        return self.base_addr + Int32(index) * Int32(8)

    def _empty(self, index):
        return self.base_addr + (Int32(index) + Int32(self.num_stages)) * Int32(8)

    def producer_acquire(self, state) -> None:
        _mbar_wait_parity(self._empty(state.index), Int32(state.phase))

    def producer_commit(self, state) -> None:
        _cp_async_mbar_arrive_noinc(self._full(state.index))

    def consumer_try_wait(self, state):
        # The blocking wait below re-tests the barrier first, so a separate
        # non-blocking peek buys nothing here.
        return cutlass.Boolean(False)

    def consumer_wait(self, state, try_wait_token=None) -> None:
        _mbar_wait_parity(self._full(state.index), Int32(state.phase))

    def consumer_release(self, state) -> None:
        _mbar_arrive_lane0(self._empty(state.index))

    def producer_tail(self, state) -> None:
        for _ in range(self.num_stages):
            self.producer_acquire(state)
            state.advance()
