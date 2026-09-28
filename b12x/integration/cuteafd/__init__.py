"""Native AOT (raw-pointer C ABI) programs for the cuteafd DeepSeek V4 engine.

Each ``compile_*_aot`` function returns an :class:`AotProgram` wrapping one
CuTe DSL program ``(void *ptr..., scalars..., cudaStream_t)`` with live row
counts as launch scalars, so a native engine can link it via ``export_to_c``
and run without Python. See ``_common`` for the shared conventions and each
module docstring for the exact pointer list, dtypes, shapes and scratch sizes:

* ``dsv4_mhc``      mHC pre / post_pre / post / head
* ``dsv4_producer`` fused Q/KV producer and the C4 index-query producer
* ``dsv4_compressor`` C4/C128 compressor (decode, prefill, continuation)
* ``dsv4_indexer``  C4 index top-k (physical slots)
* ``dsv4_sparse_mla`` compressed sparse MLA over FP8 584-byte records + sink
* ``dsv4_wo``       inverse-RoPE grouped wo_a + wo_b

Import the submodule you need; this package does not import them eagerly.
"""

from ._common import (  # noqa: F401
    FLASH,
    PRO,
    AotProgram,
    DSV4Geometry,
    Operand,
    Scalar,
    exportable_compilation,
    validate_exported_header,
)
