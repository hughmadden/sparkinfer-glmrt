"""GLM Flash's split scratch contract, independent of a CUDA device."""

import pytest

from b12x.integration.cuteafd._common import GLM53, GLM53_FLASH
from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot, sparse_mla_scratch_bytes


@pytest.mark.parametrize("fp32", [False, True])
def test_every_live_bucket_fits_the_planned_partial_and_lse_scratch(fp32):
    buckets = ((1, 33, 1), (8, 17, 2), (64, 2, 17))
    capacity = sparse_mla_scratch_bytes(GLM53_FLASH, route="decode", rows=64,
                                       buckets=buckets, fp32_partials=fp32)
    assert capacity == (17_860_608 if fp32 else 8_947_712)
    for rows in range(1, 65):
        splits = next(s for cap, s, _ in buckets if rows <= cap)
        partial_bytes = rows * GLM53_FLASH.heads * splits * 512 * (4 if fp32 else 2)
        used = ((partial_bytes + 1023) // 1024 * 1024) + rows * GLM53_FLASH.heads * splits * 4
        advertised = sparse_mla_scratch_bytes(GLM53_FLASH, route="decode", rows=rows,
                                             buckets=buckets, fp32_partials=fp32)
        assert used <= advertised <= capacity


@pytest.mark.parametrize("geometry,route", [(GLM53, "decode"), (GLM53_FLASH, "prefill")])
def test_fp32_partials_reject_other_family_and_prefill_before_compilation(geometry, route):
    with pytest.raises(ValueError, match="only for GLM Flash decode"):
        compile_glm_sparse_mla_aot(geometry, route=route, fp32_partials=True)
