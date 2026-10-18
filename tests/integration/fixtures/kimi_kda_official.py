"""Freeze one real Kimi K3 KDA layer in the pinned official BF16 runtime.

Run under the hardware queue; this fixture builder never launches containers.
Arguments: --snapshot, --reference-source (the frozen fidelity source), --output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    import torch
    import fla

    source = args.reference_source / "python/reference/families/kimi_k3/model.py"
    spec = importlib.util.spec_from_file_location("kimi_official_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    numerics = reference.configure_f32()
    official, config, quantization = reference.load_official(args.snapshot)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        layer = official.KimiDeltaAttention(config, layer_idx=0)
    weights = reference.LayerWeights(args.snapshot, quantization, kda_num_heads=96)
    # dtype=None retains native F32 conv/norm and BF16 projections.
    layer = weights.load_module(layer, "language_model.model.layers.0.self_attn")
    gen = torch.Generator(device="cuda").manual_seed(2736)
    x = torch.randn((1, 129, config.hidden_size), generator=gen, device="cuda").bfloat16()
    saved = {"weights": {name: value.detach().cpu() for name, value in layer.named_parameters()},
             "x": x.cpu(), "cases": {}, "eps": config.rms_norm_eps,
             "provenance": {"source": reference.source_identity(args.snapshot),
                            "torch": str(torch.__version__), "fla": str(fla.__version__),
                            "reference_mode": "BF16 projections with native F32 conv/norm",
                            "numerics": numerics}}
    norm_outputs = []
    hook = layer.o_norm.register_forward_hook(lambda module, inp, out: norm_outputs.append(out.detach().cpu()))
    with torch.inference_mode():
        for rows in (1, 8, 129):
            cache = official.KimiDynamicCache(config)
            out = layer(x[:, :rows], cache_params=cache)
            saved["cases"][str(rows)] = {"out": out.cpu(), "norm": norm_outputs.pop(),
                "state": cache.recurrent_states[0].cpu(),
                "conv": tuple(value.cpu() for value in cache.conv_states[0])}
    hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print(json.dumps(saved["provenance"], sort_keys=True), flush=True)
    print(f"saved official recurrent/chunk KDA fixture ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
