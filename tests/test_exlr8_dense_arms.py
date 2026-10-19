"""CPU contracts for immutable multi-arm dense measurements."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/_exlr8_dense_arms.py"
spec = importlib.util.spec_from_file_location("dense_arms", PATH)
arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arms)


def test_parse_and_key():
    assert arms.parse_arms(["base=HEAD", "candidate=deadbeef"]) == {"base": "HEAD", "candidate": "deadbeef"}
    for values in ([], ["base"], ["base="], ["base=-x"], ["../x=HEAD"], ["base=A", "base=B"]):
        with pytest.raises(ValueError):
            arms.parse_arms(values)
    a = dict(revision="a"*40, sha256="b"*64)
    assert arms.program_key(a, (128, 32)) != arms.program_key(dict(a, sha256="c"*64), (128, 32))
    assert arms.program_key(a, (128, 32)) != arms.program_key(dict(a, revision="d"*40), (128, 32))


def test_bundle_pins_and_verifies(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / arms.MODULE
    source.parent.mkdir(parents=True)
    source.write_text("class NativeTileLinear: pass\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    bundle = tmp_path / "bundle"
    records = arms.materialize(repo, bundle, {"base": "HEAD"})
    revision = records["base"]["revision"]
    assert arms.verify(bundle, {"base": revision}) == records
    with pytest.raises(ValueError, match="pinned revision"):
        arms.verify(bundle, {"base": "HEAD"})
    (bundle / records["base"]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        arms.verify(bundle, {"base": revision})
