"""Lossless per-row UE8M0 offsets in the V4.1 N256/K128 scale layout.

Each expert stores uint16 codes in the existing scale-word order, followed by
one uint8 base per resident row. Padding is excluded from the exponent range.
A caller must retain raw scales when validation refuses a tensor.
"""

import torch


class ScaleCodeRangeError(ValueError):
    """A checkpoint row cannot be represented by four-bit offsets."""


def pack_scale_codes(scales, *, tensor_name, gated_half_rows=None):
    """Pack logical uint8 [experts, rows, K/32] scales without changing bytes.

    Validation precedes output allocation. A failed tensor is never partially
    packed, so the owner can keep its raw UE8M0 path and admission unchanged.
    """
    if scales.dtype != torch.uint8 or scales.ndim != 3 or not scales.shape[-1]:
        raise ValueError(f"{tensor_name}: expected uint8 [experts, rows, K/32]")
    experts, rows, columns = scales.shape
    low = scales.amin(-1)
    high = scales.amax(-1)
    invalid = (high.to(torch.int16) - low.to(torch.int16)) > 15
    if bool(invalid.any()):
        expert, row = invalid.nonzero()[0].tolist()
        span = int(high[expert, row]) - int(low[expert, row])
        raise ScaleCodeRangeError(
            f"{tensor_name}: expert {expert} row {row} UE8M0 span {span} exceeds 15; "
            "retain raw UE8M0 scales"
        )
    if gated_half_rows is not None:
        if rows != 2 * gated_half_rows:
            raise ValueError(f"{tensor_name}: invalid gate/up row geometry")
        half = (gated_half_rows + 127) // 128 * 128
        stored_rows = 2 * half
    else:
        half = 0
        stored_rows = (rows + 255) // 256 * 256
    if stored_rows % 256:
        raise ValueError(f"{tensor_name}: resident rows must align to N256 tiles")
    tiles = (columns + 3) // 4
    offsets = torch.zeros((experts, stored_rows, tiles * 4), dtype=torch.uint8, device=scales.device)
    bases = torch.zeros((experts, stored_rows), dtype=torch.uint8, device=scales.device)
    delta = scales - low[..., None]
    if half:
        for source, destination in ((0, 0), (gated_half_rows, half)):
            offsets[:, destination:destination + gated_half_rows, :columns] = delta[:, source:source + gated_half_rows]
            bases[:, destination:destination + gated_half_rows] = low[:, source:source + gated_half_rows]
    else:
        offsets[:, :rows, :columns] = delta
        bases[:, :rows] = low
    nibble = offsets.reshape(experts, stored_rows, tiles, 4).to(torch.int32)
    codes = nibble[..., 0] | nibble[..., 1] << 4 | nibble[..., 2] << 8 | nibble[..., 3] << 12
    codes = codes.reshape(experts, stored_rows // 256, 256, tiles).permute(0, 1, 3, 2).contiguous()
    return torch.cat((codes.to(torch.uint16).view(torch.uint8).reshape(experts, -1), bases), -1).contiguous()


def unpack_scale_codes(packed, *, rows, columns):
    """Reference round trip of resident rows, including zero K-tail padding."""
    experts = packed.shape[0]
    tiles = (columns + 3) // 4
    code_bytes = rows * tiles * 2
    if rows % 256 or packed.dtype != torch.uint8 or packed.shape[1] != code_bytes + rows:
        raise ValueError("invalid V4.1 scale-code resident extent")
    codes = packed[:, :code_bytes].contiguous().view(torch.uint16).to(torch.int32)
    codes = codes.reshape(experts, rows // 256, tiles, 256).permute(0, 1, 3, 2).reshape(experts, rows, tiles)
    bases = packed[:, code_bytes:].to(torch.int32)
    values = torch.stack(tuple((codes >> (4 * i)) & 15 for i in range(4)), -1)
    values = (values + bases[..., None, None]).to(torch.uint8).reshape(experts, rows, tiles * 4)
    values[..., columns:] = 0
    return values
