"""CPU contracts for the existing batched HQA timing harness."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/benchmark_hqa_bf16_rowinv.py'
    spec = importlib.util.spec_from_file_location('hqa_timing_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_invocation_preserves_flash_shapes_and_baseline(harness, tmp_path):
    library = tmp_path / 'native.so'
    library.touch()
    args = harness.parse_args(['--native-lib', str(library), '--output', str(tmp_path / 'result.json')])
    assert args.shapes == ((1536, 8192), (1536, 16384), (4096, 4096), (8192, 4096), (16384, 4096))
    assert harness.baseline_kwargs(args) == {'tiles_by_rows': ((16, 1, 4, 16, 64, 6),)}
    assert args.out_dtype == 'bf16' and args.cublas
    assert args.samples == 31 and args.warm_launches == 200 and args.tiles is None
    assert harness.ROWS == (1, 2, 4, 8, 16, 32, 64)
    assert harness.GATE_ROWS == (1, 4, 8, 16, 64, 128, 140, 2048)
    assert harness.OFFSETS == (0, 1, 17, 63, 64, 65, 127, 139, 511, 1023, 2047)


def test_explicit_shapes_use_actual_default_and_f32_output(harness, tmp_path):
    args = harness.parse_args(['--output', str(tmp_path / 'result.json'),
                              '--shapes', '7168,6144;8192,4096', '--out-dtype', 'f32'])
    assert args.shapes == ((6144, 7168), (4096, 8192))
    assert harness.baseline_kwargs(args) == {}
    assert not args.cublas


@pytest.mark.parametrize('shapes', ['1,2', '4096', '4096,8192,128', '4096,8192;4096,8192', ''])
def test_bad_shapes_fail_before_cuda(harness, tmp_path, shapes):
    with pytest.raises(SystemExit):
        harness.parse_args(['--output', str(tmp_path / 'result.json'),
                            '--out-dtype', 'f32', '--shapes', shapes])
