"""Shared contract for the cuteafd DeepSeek V4 native AOT programs.

Every compile function in this package returns an :class:`AotProgram`: one
CuTe DSL program whose host ABI is ``(void *ptr..., <scalars>..., stream)``.
The ABI is recorded on the program (:attr:`AotProgram.abi`) so exporters can
validate the generated header against it, and so tests can launch the program
from torch tensors through the same pointer order the native engine uses.

Conventions shared by all programs:

* Pointers are raw device addresses; the caller owns every buffer, including
  scratch. Shapes in :class:`Operand` use the symbolic live row count ``rows``
  (the ``rows`` scalar) and the static geometry the program was compiled for.
  All buffers are dense row-major with the stated shape unless the operand
  says otherwise.
* Row and page counts are runtime scalars; no live request quantity enters a
  compile key. Pool-scaled offsets are computed in 64-bit arithmetic.
* Programs perform no allocation, host synchronization, or compilation at
  launch. Compile (and export) every program before CUDA graph capture.
* Exporting requires the retained IR, so compile under
  :func:`exportable_compilation` (it disables the b12x object caches for the
  duration of the call) before calling ``export_to_c``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

__all__ = [
    "AotProgram",
    "DSV4Geometry",
    "FLASH",
    "Operand",
    "PRO",
    "Scalar",
    "exportable_compilation",
    "validate_exported_header",
]


@dataclass(frozen=True)
class DSV4Geometry:
    """Static DeepSeek V4 model geometry baked into the exported programs.

    Only model constants live here; capacities are separate compile
    parameters and live counts are launch scalars.
    """

    name: str = "flash"
    hidden: int = 4096
    heads: int = 64
    q_lora_rank: int = 1024
    head_dim: int = 512
    nope_dim: int = 448
    rope_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    window: int = 128
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    norm_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if (self.head_dim, self.nope_dim, self.rope_dim) != (512, 448, 64):
            raise ValueError("DSV4 requires head/nope/rope dims 512/448/64")
        if self.hc_mult != 4:
            raise ValueError("DSV4 mHC requires hc_mult=4")
        if self.hidden not in (4096, 7168):
            raise ValueError("DSV4 hidden must be 4096 (Flash) or 7168 (Pro)")
        if (self.index_heads, self.index_head_dim) != (64, 128):
            raise ValueError("DSV4 indexer requires 64 heads of 128 dims")
        if self.heads % self.o_groups:
            raise ValueError("DSV4 heads must divide into o_groups")

    @property
    def mhc_split_k(self) -> int:
        """Partial rows per token of the mHC fn projection (2 * hidden / 128)."""
        return 2 * self.hidden // 128

    @property
    def o_group_width(self) -> int:
        return self.heads * self.head_dim // self.o_groups


FLASH = DSV4Geometry()
PRO = DSV4Geometry(
    name="pro", hidden=7168, heads=128, q_lora_rank=1536, o_groups=16,
    o_lora_rank=1024, index_topk=1024,
)


_TORCH_TO_CUTE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


@dataclass(frozen=True)
class Operand:
    """One pointer argument of an exported program.

    ``dtype`` is the element type the kernel reads or writes through the
    pointer, ``shape`` a human/machine readable shape using ``rows`` for the
    live row count, ``role`` one of ``in``, ``out``, ``inout`` or ``scratch``,
    and ``align`` the byte alignment the compiled code assumes.
    """

    name: str
    dtype: torch.dtype
    shape: str
    role: str = "in"
    align: int = 16
    note: str = ""

    @property
    def cute_dtype(self):
        return _TORCH_TO_CUTE[self.dtype]


@dataclass(frozen=True)
class Scalar:
    """One scalar launch argument (``int32``/``int64``/``float32``)."""

    name: str
    ctype: str = "int32"
    note: str = ""

    def placeholder(self):
        return {"int32": cutlass.Int32(1), "int64": cutlass.Int64(1),
                "float32": cutlass.Float32(1.0)}[self.ctype]

    def value(self, value):
        return {"int32": cutlass.Int32, "int64": cutlass.Int64,
                "float32": cutlass.Float32}[self.ctype](value)


@dataclass(frozen=True)
class AotProgram:
    """A compiled raw-pointer program plus its documented ABI.

    ``compiled`` is the CuTe DSL JIT executor (it provides ``export_to_c``).
    ``scratch`` maps each scratch operand name to a byte-size function of the
    live/capacity row count; ``geometry`` records the static compile inputs.
    """

    name: str
    compiled: object
    operands: tuple[Operand, ...]
    scalars: tuple[Scalar, ...]
    geometry: Mapping[str, object] = field(default_factory=dict)
    scratch: Mapping[str, Callable[[int], int]] = field(default_factory=dict)
    doc: str = ""

    @property
    def abi(self) -> dict[str, object]:
        return {
            "pointers": [(o.name, str(o.dtype).removeprefix("torch."), o.shape, o.role)
                         for o in self.operands],
            "scalars": [(s.name, s.ctype) for s in self.scalars],
            "stream": "stream",
        }

    def scratch_bytes(self, rows: int) -> dict[str, int]:
        """Byte size of every scratch pointer for ``rows`` (capacity) rows."""
        return {name: int(fn(int(rows))) for name, fn in self.scratch.items()}

    def export_to_c(self, file_path: str, file_name: str, function_prefix: str):
        return self.compiled.export_to_c(str(file_path), file_name, function_prefix)

    def launch(self, *tensors: torch.Tensor, scalars: Sequence[object], stream=None):
        """Launch from torch tensors in ABI order (test/diagnostic helper).

        ``tensors`` may contain ``None`` only for operands documented as
        ignorable; a 16-byte-aligned dummy address is passed in that case.
        """
        if len(tensors) != len(self.operands):
            raise ValueError(f"{self.name}: expected {len(self.operands)} pointers, got {len(tensors)}")
        if len(scalars) != len(self.scalars):
            raise ValueError(f"{self.name}: expected {len(self.scalars)} scalars")
        pointers = []
        for operand, tensor in zip(self.operands, tensors):
            address = 16 if tensor is None else int(tensor.data_ptr())
            if tensor is not None and address % operand.align:
                raise ValueError(f"{self.name}.{operand.name} is not {operand.align}-byte aligned")
            pointers.append(make_ptr(operand.cute_dtype, address, cute.AddressSpace.gmem,
                                     assumed_align=operand.align))
        values = [s.value(v) for s, v in zip(self.scalars, scalars)]
        self.compiled(*pointers, *values, current_cuda_stream() if stream is None else stream)


def compile_program(
    launch: object,
    *,
    name: str,
    operands: Sequence[Operand],
    scalars: Sequence[Scalar],
    key: tuple,
    geometry: Mapping[str, object] | None = None,
    scratch: Mapping[str, Callable[[int], int]] | None = None,
    doc: str = "",
    version: int = 1,
) -> AotProgram:
    """Compile ``launch(*pointers, *scalars, stream)`` into an AotProgram."""
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    full_key = (name, tuple(key), capability, device)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=full_key)
    pointers = [make_ptr(o.cute_dtype, 16, cute.AddressSpace.gmem, assumed_align=o.align)
                for o in operands]
    compiled = compile_cute(
        launch, *pointers, *(s.placeholder() for s in scalars), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(f"cuteafd.{name}", version, full_key),
    )
    return AotProgram(
        name=name, compiled=compiled, operands=tuple(operands), scalars=tuple(scalars),
        geometry=dict(geometry or {}), scratch=dict(scratch or {}), doc=doc,
    )


@contextmanager
def exportable_compilation():
    """Compile with the b12x object caches disabled so export_to_c has IR."""
    names = ("B12X_COMPILE_DISK_CACHE", "B12X_COMPILE_MEMORY_CACHE")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "0"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_C_TYPES = {"int32": "int32_t", "int64": "int64_t", "float32": "float"}


def validate_exported_header(program: AotProgram, header: str | Path, symbol: str) -> dict:
    """Check the generated wrapper signature and argument order against the ABI.

    Returns ``{"symbol": <_mlir entry>, "argument_count": n}`` for manifests.
    """
    text = Path(header).read_text() if not isinstance(header, str) or "\n" not in header else header
    expected = [f"{symbol}_Kernel_Module_t *module"]
    expected += [f"void *{o.name}" for o in program.operands]
    expected += [f"{_C_TYPES[s.ctype]} {s.name}" for s in program.scalars]
    expected += ["cudaStream_t stream"]
    signature = re.search(
        r"static inline int32_t cute_dsl_" + re.escape(symbol) + r"_wrapper\(([^)]*)\)", text)
    if signature is None:
        raise ValueError(f"{program.name}: wrapper for {symbol} not found in header")
    if re.sub(r"\s+", "", signature[1]) != re.sub(r"\s+", "", ",".join(expected)):
        raise ValueError(f"{program.name}: unexpected wrapper signature {signature[1]!r}")
    names = [o.name for o in program.operands] + [s.name for s in program.scalars] + ["stream"]
    count = len(names)
    arguments = re.search(r"void \*args\[" + str(count + 1) + r"\] = \{([^}]*)\}", text)
    if arguments is None or re.sub(r"\s+", "", arguments[1]) != ",".join("&" + n for n in names + ["ret"]):
        raise ValueError(f"{program.name}: unexpected generated argument order")
    entries = re.findall(r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\);", text)
    if len(entries) != 1:
        raise ValueError(f"{program.name}: expected one generated entry point, got {entries}")
    return {"symbol": entries[0], "argument_count": count}
