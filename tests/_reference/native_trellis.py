"""Independent MSB-first tail-biting EXL3 oracle (no kernel imports).

Reconstruction uses F64 integer-exact scalar laws followed by one F16 rounding.
The matrix layout is the public EXL3 tensor-core permutation, not the GPU's
ring-word/funnel algorithm.
"""
import numpy as np


class Reference:
    def __init__(self, codebook):
        if codebook not in ("e4m3", "f16", "mcg", "mul1"):
            raise ValueError(codebook)
        self.codebook = codebook

    @staticmethod
    def codewords(data, bits):
        if bits not in (2, 3, 4, 5, 6) or len(data) % (32 * bits):
            raise ValueError("expected complete K2--K6 tiles")
        # Each little-endian uint32 is an MSB-first segment of the circular code.
        words = np.frombuffer(data, dtype="<u4").reshape(-1, 8 * bits)
        stream = ((words[:, :, None] >> np.arange(31, -1, -1)) & 1).reshape(-1, 256 * bits)
        ends = np.arange(1, 257) * bits - 1
        states = np.zeros((len(words), 256), dtype=np.uint32)
        for shift in range(16):
            states |= stream[:, (ends - shift) % (256 * bits)].astype(np.uint32) << shift
        return states

    def decode_tiles(self, data, bits):
        states = self.codewords(data, bits)
        if self.codebook == "mul1":
            raw = states * np.uint32(0x83DCD12D)
            summed = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
            accum = (summed + 0x6400).astype("<u2").view("<f2").astype(np.float64)
            inv, bias = np.array([0x1EEE, 0xC931], dtype="<u2").view("<f2").astype(np.float64)
            values = (accum * inv + bias).astype(np.float16).astype(np.float32)
        else:
            raw = (states * np.uint32(0xCBAC1FED) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
            lo = (raw & 65535).astype("<u2").view("<f2").astype(np.float64)
            hi = (raw >> 16).astype("<u2").view("<f2").astype(np.float64)
            values = (lo + hi).astype(np.float16).astype(np.float32)
        if self.codebook == "e4m3":
            magnitude = np.minimum(np.abs(values), 448.)
            step = np.exp2(np.floor(np.log2(np.maximum(magnitude, 2. ** -6))) - 3)
            values = np.copysign(np.minimum(np.rint(magnitude / step) * step, 448.), values)
        out = np.empty((len(states), 16, 16), dtype=np.float32)
        for lane in range(32):
            r, c = 2 * (lane % 4), lane // 4
            positions = ((r,c),(r+1,c),(r+8,c),(r+9,c),(r,c+8),(r+1,c+8),(r+8,c+8),(r+9,c+8))
            for j, (row, col) in enumerate(positions):
                out[:, row, col] = values[:, 8 * lane + j]
        return out

    def decode_unit(self, data, bits, projection, hidden):
        tiles = self.decode_tiles(data, bits)
        if projection in ("gate", "up"):
            return tiles.reshape(hidden // 16, 8, 16, 16).transpose(0, 2, 1, 3).reshape(hidden, 128)
        if projection == "down":
            return tiles.reshape(hidden // 128, 8, 8, 16, 16).transpose(1, 3, 0, 2, 4).reshape(128, hidden)
        raise ValueError(projection)
