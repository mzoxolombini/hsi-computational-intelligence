"""Tests for src/optimization/ils_hed.py"""

import pytest

pytest.importorskip("torch", reason="torch not available")

from src.optimization.ils_hed import ClassicalDetectorConfig


def test_enumerate_all_configs_returns_53_single_detector_configs():
    config_space = ClassicalDetectorConfig()
    configs = config_space.enumerate_all_configs()
    assert len(configs) == 53

    for cfg in configs:
        non_zero = sum(int(cfg[key] != 0) for key in ("canny", "sobel", "laplacian", "gabor"))
        assert non_zero <= 1
