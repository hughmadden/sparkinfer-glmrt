"""Native AOT (raw-pointer C ABI) programs for the cuteafd DeepSeek V4 engine.

Each ``compile_*_aot`` function returns an :class:`AotProgram` wrapping one
CuTe DSL program ``(void *ptr..., scalars..., cudaStream_t)`` with live row
counts as launch scalars, so a native engine can link it via ``export_to_c``
and run without Python. See ``_common`` for the shared conventions and each
module docstring for the exact pointer list, dtypes, shapes and scratch sizes:

* ``dsv4_mhc``         ``compile_dsv4_mhc_{pre,post_pre,post,head}_aot``
* ``dsv4_producer``    ``compile_dsv4_producer_aot`` (fused Q/KV producer) and
                       ``compile_dsv4_index_producer_aot`` (C4 index query)
* ``dsv4_compressor``  ``compile_dsv4_compressor_{decode,prefill,continuation}_aot``
                       (ratio 4 with the index compressor, or 128)
* ``dsv4_indexer``     ``compile_dsv4_index_topk_aot`` (C4 top-k, physical slots)
* ``dsv4_sparse_mla``  ``compile_dsv4_sparse_mla_aot`` (FP8 584-byte records + sink)
* ``dsv4_wo``          ``compile_dsv4_wo_projection_aot`` (inverse RoPE, wo_a, wo_b)

Exporter recipe (per program)::

    with exportable_compilation():
        program = compile_...(FLASH, ...)
    program.export_to_c(out_dir, stem, "cuteafd_" + stem)
    validate_exported_header(program, out_dir / f"{stem}.h", "cuteafd_" + stem)
    # manifest: program.abi, program.geometry, program.scratch_bytes(max_rows)

Tests: ``tests/integration/test_cuteafd_dsv4_*_aot.py`` compare every program
against the prepared b12x Python path it replaces.

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
