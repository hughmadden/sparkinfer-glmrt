"""CUDA-independent geometry of the shared KDA programs."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class KDAGeometry:
    """KDA-only geometry, independent of a family's MLA, FFN and residuals.

    Full-rank output gates are folded into ``w_in = [q;k;v;f_a;g;b]``;
    ``w_fg`` then contains only f_b. Low-rank gates retain GLM Flash's
    ``[q;k;v;f_a;g_a;b]`` and ``[f_b;g_b]`` layout. Beta is a raw BF16
    projection converted to FP32 before the in-kernel sigmoid, as Kimi K3
    and FLA 0.5.2 specify (no BF16 rounding of the sigmoid).
    """

    hidden: int
    kda_heads: int
    kda_head_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0
    norm_eps: float = 1.0e-5
    output_gate: str = "full_rank"
    o_norm_dtype: str = "float32"

    def __post_init__(self):
        if self.hidden <= 0 or self.hidden % 128:
            raise ValueError("KDA hidden must be a positive multiple of 128")
        if self.kda_heads <= 0 or self.kda_heads % 8:
            raise ValueError("KDA heads must be a positive multiple of 8")
        if (self.kda_head_dim, self.conv_kernel) != (128, 4):
            raise ValueError("KDA programs require 128-wide heads and a 4-tap conv")
        if self.output_gate not in ("full_rank", "low_rank"):
            raise ValueError("KDA output_gate is full_rank or low_rank")
        if self.o_norm_dtype not in ("float32", "bfloat16"):
            raise ValueError("KDA o_norm_dtype is float32 or bfloat16")
        if not math.isfinite(self.gate_lower_bound) or self.gate_lower_bound >= 0:
            raise ValueError("KDA gate_lower_bound must be finite and negative")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("KDA norm_eps must be finite and positive")

    @property
    def kda_width(self):
        return self.kda_heads * self.kda_head_dim

    @property
    def kda_in_width(self):
        return 3 * self.kda_width + self.kda_head_dim + self.gate_width + self.kda_heads

    @property
    def gate_width(self):
        return self.kda_width if self.output_gate == "full_rank" else self.kda_head_dim

    @property
    def kda_state_bytes(self):
        return self.kda_heads * self.kda_head_dim * self.kda_head_dim * 4

    @property
    def conv_state_bytes(self):
        return (self.conv_kernel - 1) * 3 * self.kda_width * 2

    def head_split(self, ranks):
        if ranks <= 0 or self.kda_heads % ranks:
            raise ValueError("KDA heads must split evenly over positive ranks")
        return replace(self, kda_heads=self.kda_heads // ranks)


def kda_layout(g):
    """Static offsets and strides; GLMFGeometry keeps its original identity."""
    full = getattr(g, "output_gate", "low_rank") == "full_rank"
    d, hd = g.kda_width, g.kda_head_dim
    return {
        "full_rank_gate": full,
        "fg_batch": 1 if full else 2,
        "fg_stride": d if full else 2 * d,
        "gate_offset": 3 * d + hd if full else d,
        "gate_stride": g.kda_in_width if full else 2 * d,
        "beta_offset": 4 * d + hd if full else 3 * d + 2 * hd,
        "o_norm_dtype": getattr(g, "o_norm_dtype", "bfloat16"),
        "round_recurrent_beta": not isinstance(g, KDAGeometry),
    }
