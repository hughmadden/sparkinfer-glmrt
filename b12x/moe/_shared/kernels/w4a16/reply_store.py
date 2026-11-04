"""Shared producer wire store: the existing EXL3 conversion and block layout."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .kernel import W4A16TopKSumKernel


class ReplyStore(W4A16TopKSumKernel):
    def __init__(self, hidden, format):
        if hidden <= 0 or hidden % 128 or format not in (1, 2):
            raise ValueError("compressed replies require H128 and format 1 or 2")
        self.hidden_size = int(hidden)
        self.reply_format = int(format)
        self.reply_row_bytes = (hidden + hidden // 32 if format == 1 else
                                hidden // 2 + hidden // 16)

    @cute.jit
    def store(self, output: cute.Tensor, row: Int32, block: Int32, lane: Int32,
              v0: cutlass.Float32, v1: cutlass.Float32,
              v2: cutlass.Float32, v3: cutlass.Float32):
        self._quantize_and_store_reply_slab(output, row, block,
            block * Int32(128) + lane * Int32(4), lane, v0, v1, v2, v3)
