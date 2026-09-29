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
* ``dsv4_ffn``         ``compile_dsv4_shared_ffn_aot`` (shared expert),
                       ``compile_dsv4_router_scores_aot`` (FP32 gate scores),
                       ``compile_dsv4_expert_input_quant_aot`` (FP8 K32 wire rows)
* ``weights``          ``WEIGHT_SOURCES`` (checkpoint tensor -> operand table),
                       ``compile_dsv4_block_fp8_scale_prep_aot`` (UE8M0 block
                       scales -> scale_mma), ``compile_dsv4_i64_to_i32_aot``

GLM 5.x (``glm_moe_dsa``, geometry :data:`GLM53`), programs named ``glm_*``:

* ``glm_attention``    ``compile_glm_producer_aot`` (q/kv producer, absorbed
                       576-wide query, FP8 656-byte latent records),
                       ``compile_glm_index_producer_aot`` (DSA index q/k/weights),
                       ``compile_glm_o_aot`` (W_UV then o_proj)
* ``glm_indexer``      ``compile_glm_index_topk_aot`` (causal top-2048)
* ``glm_sparse_mla``   ``compile_glm_sparse_mla_aot`` (latent attention)
* ``glm_ffn``          ``compile_glm_norm_aot`` (residual add + RMSNorm),
                       ``compile_glm_ffn_aot`` (SwiGLU), ``compile_glm_router_scores_aot``,
                       ``compile_glm_expert_input_quant_aot``

GLM programs take BF16 weights (FP8 blocks x FP32 scales, dequantized at
load); tests: ``tests/integration/test_cuteafd_glm_*.py`` against the
transformers ``modeling_glm_moe_dsa`` modules and golden layer outputs.

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
    GLM53,
    PRO,
    AotProgram,
    DSV4Geometry,
    GLMGeometry,
    Operand,
    Scalar,
    exportable_compilation,
    validate_exported_header,
)
