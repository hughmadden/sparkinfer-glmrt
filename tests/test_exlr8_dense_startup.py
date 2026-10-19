"""Exercise dense benchmark startup in the compiler image without a GPU."""
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from benchmarks import benchmark_exlr8_dense as bench
from benchmarks._exlr8_dense_arms import materialize


def test_device_l2_query(monkeypatch):
    monkeypatch.setattr(bench.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(bench.cuda, "cuDeviceGet", lambda index: (bench.cuda.CUresult.CUDA_SUCCESS, 0))
    monkeypatch.setattr(bench.cuda, "cuDeviceGetAttribute", lambda attr, dev: (bench.cuda.CUresult.CUDA_SUCCESS, 128 << 20))
    assert bench.copies_for(1 << 20, True) == 256
    assert bench.copies_for(1 << 20, False) == 1


def test_materialized_startup(monkeypatch, tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "b12x/gemm/native_tile_linear.py"
    source.parent.mkdir(parents=True)
    source.write_text("class NativeTileLinear: pass\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    bundle = tmp_path / "bundle"
    records = materialize(repo, bundle, {"base": "HEAD"})
    monkeypatch.setattr(sys, "argv", ["dense", "--arm", "base=" + records["base"]["revision"],
                        "--arm-bundle", str(bundle), "--reference", "base", "--accuracy-reference", "base",
                        "--op-reference", "mxfp8/q_a=base", "--require-reference-arm", "base",
                        "--rows", "1,4,16,64,4096", "--output", str(tmp_path / "result.json")])
    monkeypatch.setattr(bench.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(bench.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name="stub", multi_processor_count=170))
    def reached_device_work():
        raise RuntimeError("startup complete")
    monkeypatch.setattr(bench, "read_roofline", reached_device_work)
    with pytest.raises(RuntimeError, match="startup complete"):
        bench.main()


@pytest.mark.parametrize("encoding,side", [("mxfp8", 1), ("mxfp8", 2), ("int4_g32", 1), ("int4_g32", 2), ("fp8_row", 1)])
def test_reference_weight_cpu(encoding, side):
    torch = bench.torch
    n, k = 32, 256
    if encoding == "int4_g32":
        q = torch.randint(0, 16, (n, k), dtype=torch.uint8)
        payload = q[:, ::2] | (q[:, 1::2] << 4)
        scales = (torch.rand(n, k // 32) + .01).half()
        expected = (q.float() - 8) * scales.float().repeat_interleave(32, 1)
    else:
        q = torch.randn(n, k).to(torch.float8_e4m3fn)
        payload = q.view(torch.uint8)
        if encoding == "mxfp8":
            scales = torch.randint(117, 132, (n, k // 32), dtype=torch.uint8)
            expected = q.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
        else:
            scales = torch.rand(n) + .01
            expected = q.float() * scales[:, None]
    if side == 2:
        payload = payload.reshape(n, k // 128, -1).permute(1, 0, 2).contiguous()
        scales = scales.reshape(n, k // 128, 4).permute(1, 0, 2).contiguous()
    actual = bench.reference_weight(encoding, k, n, side, payload, scales.flatten())
    assert torch.equal(actual, expected)
