"""Lossless per-row UE8M0 offsets in the V4.1 N256/K128 scale layout.

Each expert stores uint16 codes in the existing scale-word order, followed by
one uint8 base per resident row. Padding is excluded from the exponent range.
Unsupported exponent spans fail the load before any execution views publish.
"""

import torch


class ScaleCodeRangeError(ValueError):
    """A checkpoint row cannot be represented by four-bit offsets."""


def pack_scale_codes(scales, *, tensor_name, gated_half_rows=None):
    """Pack logical uint8 [experts, rows, K/32] scales without changing bytes.

    Validation precedes output allocation. A failed tensor never becomes
    execution-visible; resident admission always uses the shape-derived size.
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
            "unsupported V4.1 scale codes"
        )
    if rows <= 0 or rows % 32 or (gated_half_rows is not None and rows != 2 * gated_half_rows):
        raise ValueError(f"{tensor_name}: invalid exact K32 scale rows")
    stored_rows = rows
    tiles = (columns + 3) // 4
    offsets = torch.zeros((experts, stored_rows, tiles * 4), dtype=torch.uint8, device=scales.device)
    bases = torch.zeros((experts, stored_rows), dtype=torch.uint8, device=scales.device)
    delta = scales - low[..., None]
    offsets[:, :, :columns] = delta
    bases[:] = low
    nibble = offsets.reshape(experts, stored_rows, tiles, 4).to(torch.int32)
    codes = nibble[..., 0] | nibble[..., 1] << 4 | nibble[..., 2] << 8 | nibble[..., 3] << 12
    codes = torch.cat(tuple(codes[:, start:start + 256].transpose(1, 2).contiguous().flatten(1)
                            for start in range(0, rows, 256)), 1)
    return torch.cat((codes.to(torch.uint16).view(torch.uint8).reshape(experts, -1), bases), -1).contiguous()


def unpack_scale_codes(packed, *, rows, columns):
    """Reference round trip of resident rows, including zero K-tail padding."""
    experts = packed.shape[0]
    tiles = (columns + 3) // 4
    code_bytes = rows * tiles * 2
    if rows <= 0 or rows % 32 or columns <= 0 or packed.dtype != torch.uint8 or packed.shape[1] != code_bytes + rows:
        raise ValueError("invalid V4.1 scale-code resident extent")
    codes = packed[:, :code_bytes].contiguous().view(torch.uint16).to(torch.int32)
    codes = torch.cat(tuple(codes[:, start * tiles:(start + min(256, rows - start)) * tiles]
                            .reshape(experts, tiles, min(256, rows - start)).transpose(1, 2)
                            for start in range(0, rows, 256)), 1)
    bases = packed[:, code_bytes:].to(torch.int32)
    values = torch.stack(tuple((codes >> (4 * i)) & 15 for i in range(4)), -1)
    values = (values + bases[..., None, None]).to(torch.uint8).reshape(experts, rows, tiles * 4)
    values[..., columns:] = 0
    return values
