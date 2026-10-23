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


def test_op_references():
    entries = {"round2": "a"*40, "gemini": "b"*40}
    assert arms.parse_references("round2", ["int4_g32/o=gemini"], entries) == {"int4_g32/o": "gemini"}
    for default, values in (("missing", []), ("round2", ["o=gemini"]),
                            ("round2", ["mxfp8/q_a=missing"]),
                            ("round2", ["mxfp8/q_a=gemini", "mxfp8/q_a=round2"])):
        with pytest.raises(ValueError):
            arms.parse_references(default, values, entries)


def test_cold_copies_covers_runtime_l2():
    assert arms.cold_copies(1 << 20, 128 << 20) == 256
    assert arms.cold_copies(3 << 20, 128 << 20) == 86
    assert arms.cold_copies(1 << 30, 128 << 20) == 1
    for weight, l2 in ((0, 1), (1, 0), (-1, 1)):
        with pytest.raises(ValueError):
            arms.cold_copies(weight, l2)


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


def test_trellis_references():
    assert arms.parse_references("base", ["trellis/q_b=base"], {"base": "HEAD"}) == {"trellis/q_b": "base"}


@pytest.mark.parametrize("formats", [("trellis",), ("trellis", "mxfp8"), ("int4_g32",)])
def test_per_format_modules(tmp_path, formats):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for module in (arms.MODULE, arms.TRELLIS + "/__init__.py", arms.TRELLIS + "/_native.py", arms.TRELLIS + "/_helper.py"):
        path = repo / module
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    bundle = tmp_path / "bundle"
    records = arms.materialize(repo, bundle, {"base": "HEAD"}, formats)
    pinned = {"base": records["base"]["revision"]}
    assert arms.verify(bundle, pinned, formats) == records
    files = records["base"]["files"]
    assert (arms.MODULE in files) == (set(formats) != {"trellis"})
    assert (arms.TRELLIS + "/_helper.py" in files) == ("trellis" in formats)
    if "trellis" not in formats:
        with pytest.raises(ValueError, match="missing requested formats"):
            arms.verify(bundle, pinned, ["trellis"])
    target = next(iter(files.values()))
    (bundle / target["path"]).write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        arms.verify(bundle, pinned, formats)


@pytest.mark.parametrize("rounds,replays", [(0, 1), (6, 100)])
def test_valid_timing(rounds, replays):
    arms.validate_timing(rounds, replays)


@pytest.mark.parametrize("rounds,replays", [(-1, 10), (1, 0), (0, -1), (1.5, 10), (2, 1.5)])
def test_invalid_timing(rounds, replays):
    with pytest.raises(ValueError):
        arms.validate_timing(rounds, replays)


def test_alternating_timing_order():
    names = ["base", "candidate", "reference_control"]
    assert [arms.timing_order(names, i) for i in range(4)] == [names, names[::-1], names, names[::-1]]
    assert names == ["base", "candidate", "reference_control"]


@pytest.mark.parametrize("control,passed", [(10.1, True), (10.5, False)])
def test_control_noise(control, passed):
    medians, spread, noise_pass = arms.timing_summary(
        {"base": [9., 10., 11.], "reference_control": [control]*3, "candidate": [8.]*3}, "base")
    assert medians["base"] == 10.
    assert spread == pytest.approx(abs(control/10.-1))
    assert noise_pass is passed


def test_missing_control_is_not_a_noise_pass():
    _, spread, passed = arms.timing_summary({"base": [10.]}, "base")
    assert spread is None and passed is False
