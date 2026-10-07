"""Static preparation contracts for native FP32 audio support stages."""
from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan, make_fixed_contract

OPERATIONS = frozenset({"layer_norm", "rms_norm", "sum_square", "softmax", "rvq_select",
    "frame", "magnitude", "log", "im2col", "bias", "gelu", "bias_gelu", "silu_product",
    "add", "rope_pack", "pack_heads", "unpack_heads", "speech_add"})


@dataclass(frozen=True, kw_only=True)
class AudioQuery:
    operation: str
    width: int
    kernel: int = 1
    stride: int = 1

    def __post_init__(self):
        if self.operation not in OPERATIONS:
            raise ValueError("unknown native audio operation")
        if any(type(value) is not int for value in (self.width, self.kernel, self.stride)):
            raise ValueError("audio geometry must use integer dimensions")
        if self.width <= 0 or self.width > 16384 or self.kernel not in (1, 2, 3) or self.stride not in (1, 2):
            raise ValueError("invalid static audio geometry")
        if self.operation != "im2col" and (self.kernel != 1 or self.stride != 1):
            raise ValueError("only im2col specializes convolution geometry")
        if self.operation in {"rope_pack", "pack_heads", "unpack_heads", "speech_add", "rvq_select"} and self.width != 1024:
            raise ValueError("audio tower requires width1024 and heads16")
        if self.operation == "frame" and self.width != 960:
            raise ValueError("audio framing requires n_fft960")
        if self.operation == "magnitude" and self.width != 481:
            raise ValueError("audio rFFT requires481 complex bins")
        if self.operation in {"layer_norm", "rms_norm", "sum_square", "softmax"} and self.width != 1024:
            raise ValueError("audio reductions require width1024")
        if self.operation == "im2col" and (self.width, self.kernel, self.stride) not in {(384, 3, 1), (3072, 3, 2), (2048, 2, 2)}:
            raise ValueError("unsupported audio convolution geometry")
        allowed = {"log": {128}, "bias": {1024, 4096}, "gelu": {1024, 16384},
                   "bias_gelu": {1024, 4096}, "silu_product": {4096}, "add": {1024}}
        if self.operation in allowed and self.width not in allowed[self.operation]:
            raise ValueError("unsupported audio activation width")


TUNING = make_fixed_contract(component_id="norm.audio", query_type=AudioQuery, backend="cute")


def compile_audio(query_payload, ordinal):
    from .audio import _compile
    query = AudioQuery(**dict(query_payload))
    return _compile(query.operation, query.width, query.kernel, query.stride, ordinal)[0]


@dataclass(frozen=True)
class AudioState:
    query: AudioQuery
    compiled: object
    types: tuple[object, ...]


def plan(query: AudioQuery, *, device, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, AudioQuery):
        raise TypeError("audio plan requires AudioQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("audio live row/sequence counts are runtime inputs")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("native audio requires CUDA")

    def materialize(selection, detected):
        del selection, detected
        from .audio import _compile
        compiled, types = _compile(query.operation, query.width, query.kernel, query.stride, target.index)
        return AudioState(query, compiled, types)

    return Plan(contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, detected: (CompileJob.create(
            "b12x.norm._audio_preparation:compile_audio", TUNING.encode_query(query), detected.ordinal),),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize, _device=target)
