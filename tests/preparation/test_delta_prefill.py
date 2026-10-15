"""GDN/KDA prefill contracts through scoped prepared executions."""

from __future__ import annotations

import pytest
import torch

from b12x.sequence import gdn_prefill as gdn
from b12x.testing.delta_prefill_cases import (
    PrefillCase,
    assert_close,
    check_binding,
    make_inputs,
    oracle,
    prepared_binding,
    run_binding,
)


def _gpu():
    from ..conftest import require_b12x
    return require_b12x()


@pytest.mark.parametrize("ratio", (2, 3))
def test_gdn_prefill_head_ratio_matches_expanded_reference(ratio):
    from b12x.sequence.gdn_prefill.reference import prepare_chunk, recurrent_gdn

    case = PrefillCase("gdn", 2, 2 * ratio, (17,))
    tensors = make_inputs(case, device="cpu")
    q, k, v = (tensors[name] for name in ("q", "k", "v"))
    args = (v, tensors["raw_g"], tensors["raw_beta"], tensors["A_log"], tensors["dt_bias"])
    state = tensors["recurrent_state"][0]
    actual = recurrent_gdn(q, k, *args, initial_state=state, checkpoint_offset=16)
    expanded = recurrent_gdn(q.repeat_interleave(ratio, 1), k.repeat_interleave(ratio, 1),
                             *args, initial_state=state, checkpoint_offset=16)
    for got, expected in zip(actual, expanded):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    scalars = (tensors["raw_g"][:16], tensors["raw_beta"][:16],
               tensors["A_log"], tensors["dt_bias"])
    actual_chunk = prepare_chunk(q[:16], k[:16], *scalars)
    expanded_chunk = prepare_chunk(q[:16].repeat_interleave(ratio, 1),
                                   k[:16].repeat_interleave(ratio, 1), *scalars)
    for name in actual_chunk:
        torch.testing.assert_close(actual_chunk[name], expanded_chunk[name], rtol=0, atol=0)


@pytest.mark.parametrize("values", (2, 3))
def test_gdn_prefill_accepts_integral_ratios(values):
    from b12x.sequence.gdn_prefill._tuning import GdnPrefillQuery, _validate_query

    gdn.Caps(device="cuda:0", max_tokens=65, max_seqs=1, max_state_slots=3,
             key_heads=1, value_heads=values)
    query = GdnPrefillQuery(key_heads=1, value_heads=values, head_dim=128,
                           model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True,
                           checkpoint_export=False, max_tokens=65, max_seqs=1)
    _validate_query(query, None)


def test_gdn_prefill_rejects_fractional_head_ratio():
    from b12x.sequence.gdn_prefill._tuning import GdnPrefillQuery, _validate_query
    from b12x.sequence.gdn_prefill.reference import prepare_chunk, recurrent_gdn

    with pytest.raises(ValueError, match="integral"):
        gdn.Caps(device="cuda:0", max_tokens=65, max_seqs=1, max_state_slots=3,
                 key_heads=2, value_heads=3)
    query = GdnPrefillQuery(key_heads=2, value_heads=3, head_dim=128,
                           model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True,
                           checkpoint_export=False, max_tokens=65, max_seqs=1)
    with pytest.raises(ValueError, match="integral"):
        _validate_query(query, None)
    tensors = make_inputs(PrefillCase("gdn", 2, 3, (1,)), device="cpu")
    q, k, v, a, b, log, bias = (tensors[name] for name in
                               ("q", "k", "v", "raw_g", "raw_beta", "A_log", "dt_bias"))
    with pytest.raises(ValueError, match="integral"):
        recurrent_gdn(q, k, v, a, b, log, bias, initial_state=tensors["recurrent_state"][0])
    with pytest.raises(ValueError, match="integral"):
        prepare_chunk(q, k, a, b, log, bias)


@pytest.mark.parametrize("ratio", (2, 3))
def test_prepared_gdn_prefill_preserves_integral_head_ratios(ratio):
    case = PrefillCase("gdn", 2, 2 * ratio, (65, 17))
    tensors = make_inputs(case, device=_gpu())
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    with prepared_binding(case, tensors) as binding:
        gdn.run(binding)
        check_binding(case, binding, expected, state, initial)


@pytest.mark.parametrize("recipe", ("gdn", "kda"))
@pytest.mark.parametrize("lengths", ((1,), (17,), (33, 0, 17), (128, 33)))
def test_prepared_prefill_preserves_dynamic_lengths_and_slots(recipe, lengths) -> None:
    key_heads = 1 if recipe == "gdn" else 3
    value_heads = 3 if recipe == "gdn" else 3
    case = PrefillCase(recipe, key_heads, value_heads, lengths)
    tensors = make_inputs(case, device=_gpu(), max_tokens=256, max_seqs=4)
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)

    with prepared_binding(case, tensors, max_tokens=256, max_seqs=4) as binding:
        run_binding(recipe, binding)
        check_binding(case, binding, expected, state, initial)


@pytest.mark.parametrize("lengths", ((), (0,), (0, 0), (0, 17, 33)))
def test_prepared_prefill_preserves_null_and_checkpoint_slots(lengths) -> None:
    case = PrefillCase("gdn", 1, 3, lengths)
    tensors = make_inputs(case, device=_gpu())
    null = tensors["recurrent_state"].shape[0] - 1
    tensors["recurrent_state"][null].fill_(float("nan"))
    tensors["initial_state_indices"].fill_(null)
    if len(lengths) > 1:
        tensors["initial_state_indices"][1] = tensors["final_state_indices"][1]
    if lengths and lengths[-1] >= 16:
        tensors["checkpoint_offsets"][len(lengths) - 1] = 16
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors, null_state_index=null)

    with prepared_binding(case, tensors, checkpoint_export=True, null_state_index=null) as binding:
        gdn.run(binding)
        check_binding(case, binding, expected, state, initial)


def test_prepared_prefill_handles_large_int32_state_offsets() -> None:
    device = _gpu()
    case = PrefillCase("gdn", 1, 3, (17,))
    tensors = make_inputs(case, device=device)
    compact_initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    elements, stride = 3 * 128 * 128, 3 * 128 * 128 + 128
    high = (1 << 31) // stride + 1
    storage = torch.empty((high + 3) * stride, dtype=torch.float32, device=device)
    pool = torch.as_strided(storage, (high + 3, 3, 128, 128), (stride, 128 * 128, 128, 1))
    pool[high].copy_(compact_initial[0])
    pool[high + 1].fill_(float("nan"))
    pool[high + 2].fill_(5)
    tensors["recurrent_state"] = pool
    tensors["initial_state_indices"].fill_(high)
    tensors["final_state_indices"].fill_(high + 1)
    tensors["checkpoint_state_indices"].fill_(high + 2)

    with prepared_binding(case, tensors) as binding:
        gdn.run(binding)
        assert_close("large-slot output", binding.output, expected, ratio=1e-2)
        assert_close("large-slot state", pool[high + 1], state[1], ratio=5e-3)
        torch.testing.assert_close(pool[high], compact_initial[0], rtol=0, atol=0)
        assert torch.all(pool[high + 2] == 5)


def test_prepared_prefill_keeps_strided_inputs_immutable() -> None:
    case = PrefillCase("gdn", 1, 3, (63, 128))
    tensors = make_inputs(case, device=_gpu())
    for name in ("initial_state_indices", "final_state_indices", "checkpoint_state_indices"):
        tensors[name] = tensors[name].to(torch.int64)
    for name in ("raw_g", "raw_beta"):
        shape = (case.tokens, 2, 3) if name == "raw_g" else (case.tokens, 3, 2)
        storage = torch.zeros(shape, dtype=torch.bfloat16, device=tensors[name].device)
        view = storage[:, 0, :] if name == "raw_g" else storage[:, :, 0]
        view.copy_(tensors[name])
        tensors[name] = view
    initial = tensors["recurrent_state"].clone()
    saved = {name: value.clone() for name, value in tensors.items() if name not in ("output", "recurrent_state")}
    expected, state = oracle(case, tensors, qk_l2norm=False)

    with prepared_binding(case, tensors, qk_l2norm=False) as binding:
        gdn.run(binding)
        check_binding(case, binding, expected, state, initial)
    for name, value in saved.items():
        torch.testing.assert_close(tensors[name], value, rtol=0, atol=0)
