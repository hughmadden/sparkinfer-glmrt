"""Static storage geometry for the opt-in MXFP8/MXFP4 streaming down GEMM."""
from dataclasses import dataclass
from typing import ClassVar


def mxfp8_down_row_bytes(inter: int) -> int:
    if inter <= 0 or inter % 128:
        raise ValueError("MXFP8 down rows need a positive 128-aligned intermediate slice")
    # TP6 I=384 has 12 scale bytes: pad 396 to 400 for 16-byte cp.async alignment.
    return (inter + inter // 32 + 15) // 16 * 16


@dataclass(frozen=True)
class Mxfp4DownA8Plan:
    hidden: int
    inter: int
    experts: int
    tile_m: ClassVar[int] = 128
    tile_n: ClassVar[int] = 128
    k_block: ClassVar[int] = 128
    threads: ClassVar[int] = 256

    def __post_init__(self):
        if self.hidden <= 0 or self.hidden % 128 or self.experts <= 0:
            raise ValueError("MXFP4 A8 down needs positive experts and a 128-aligned hidden size")
        mxfp8_down_row_bytes(self.inter)

    @property
    def row_bytes(self) -> int:
        return mxfp8_down_row_bytes(self.inter)

    @property
    def a_stride(self) -> int:
        return self.k_block + 16

    @property
    def w_stride(self) -> int:
        return self.k_block // 2 + 16

    @property
    def s_stride(self) -> int:
        return self.k_block // 32 + 4

    @property
    def a_bytes(self) -> int:
        return self.tile_m * self.a_stride

    @property
    def w_bytes(self) -> int:
        return self.tile_n * self.w_stride

    @property
    def s_bytes(self) -> int:
        return self.tile_m * self.s_stride

    @property
    def stage_bytes(self) -> int:
        return self.a_bytes + self.w_bytes + 2 * self.s_bytes

    @property
    def out_stride(self) -> int:
        return self.tile_n * 2 + 16

    @property
    def out_bytes(self) -> int:
        return self.tile_m * self.out_stride
