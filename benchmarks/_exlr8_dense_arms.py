"""Immutable dense benchmark sources; deliberately usable without CUDA imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

MODULE = "b12x/gemm/native_tile_linear.py"
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def parse_arms(values):
    arms = {}
    for value in values:
        name, sep, rev = value.partition("=")
        if not sep or not NAME.fullmatch(name) or not rev or rev.startswith("-") or name in arms:
            raise ValueError(f"invalid or duplicate arm {value!r}; expected NAME=git-rev")
        arms[name] = rev
    if not arms:
        raise ValueError("at least one --arm is required")
    return arms


def parse_references(default, values, arms):
    if default not in arms:
        raise ValueError("reference must name an arm")
    result = {}
    for value in values:
        op, sep, arm = value.partition("=")
        if not sep or not re.fullmatch(r"(?:mxfp8|int4_g32|fp8_row)/[A-Za-z0-9_]+", op) or arm not in arms or op in result:
            raise ValueError(f"invalid or duplicate op reference {value!r}")
        result[op] = arm
    return result


def program_key(arm, geometry):
    return (arm["revision"], arm["sha256"], *geometry)


def cold_copies(weight_bytes, l2_bytes):
    if weight_bytes <= 0 or l2_bytes <= 0:
        raise ValueError("weight and L2 sizes must be positive")
    return max(1, (2*l2_bytes + weight_bytes - 1)//weight_bytes)


def materialize(repo, bundle, arms):
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "arms": {}}
    for name, rev in arms.items():
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--verify", rev + "^{commit}"], text=True).strip()
        source = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{MODULE}"])
        digest = hashlib.sha256(source).hexdigest()
        relative = f"{name}/{revision}/native_tile_linear.py"
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        manifest["arms"][name] = dict(revision=revision, sha256=digest, path=relative)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["arms"]


def verify(bundle, arms):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != 1 or set(manifest["arms"]) != set(arms):
        raise ValueError("arm bundle schema or names do not match")
    result = manifest["arms"]
    for name, expected in arms.items():
        arm = result[name]
        path = (bundle / arm["path"]).resolve()
        if not path.is_relative_to(bundle) or not re.fullmatch(r"[a-f0-9]{40}", arm["revision"]):
            raise ValueError("invalid arm bundle path or revision")
        # Offline invocations must name the resolved commit, not a moving ref.
        if expected != arm["revision"]:
            raise ValueError(f"{name}: use the pinned revision {arm['revision']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != arm["sha256"]:
            raise ValueError(f"{name}: arm source digest mismatch")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--arm-bundle", type=Path, required=True)
    args = parser.parse_args()
    records = materialize(args.repo, args.arm_bundle, parse_arms(args.arm))
    for name, record in records.items():
        print(f"--arm {name}={record['revision']}")
