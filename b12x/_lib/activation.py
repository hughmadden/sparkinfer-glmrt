"""Shared scalar gated activation used by fused and pointer-ABI MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32


@cute.jit
def gated_activation_value(gate: Float32, up: Float32, *,
                           is_v41: cutlass.Constexpr = False,
                           has_limit: cutlass.Constexpr = False,
                           limit_value: cutlass.Constexpr = 0.0,
                           is_swigluoai: cutlass.Constexpr = False,
                           alpha: cutlass.Constexpr = 1.0,
                           offset: cutlass.Constexpr = 0.0,
                           is_situ: cutlass.Constexpr = False,
                           situ_beta: cutlass.Constexpr = 4.0,
                           situ_linear_beta: cutlass.Constexpr = 25.0,
                           fast_math: cutlass.Constexpr = False):
    """Existing dynamic MoE epilogue, before its caller's final rounding."""
    if cutlass.const_expr(is_v41):
        gate = gate.to(cutlass.BFloat16).to(Float32)
        up = up.to(cutlass.BFloat16).to(Float32)
    if cutlass.const_expr(has_limit):
        limit = Float32(limit_value)
        neg_limit = Float32(-limit_value)
        if gate > limit:
            gate = limit
        if up > limit:
            up = limit
        if up < neg_limit:
            up = neg_limit
    sigmoid_arg = gate
    up_term = up
    if cutlass.const_expr(is_swigluoai):
        sigmoid_arg = Float32(alpha) * gate
        up_term = up + Float32(offset)
    sigmoid = cute.arch.rcp_approx(
        Float32(1.0) + cute.math.exp(-sigmoid_arg, fastmath=fast_math)
    )
    if cutlass.const_expr(is_situ):
        beta = Float32(situ_beta)
        linear_beta = Float32(situ_linear_beta)
        situ_gate = beta * cute.math.tanh(gate / beta, fastmath=fast_math) * sigmoid
        situ_up = linear_beta * cute.math.tanh(up / linear_beta, fastmath=fast_math)
        return situ_gate * situ_up
    return gate * sigmoid * up_term
