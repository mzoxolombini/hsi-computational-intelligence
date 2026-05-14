# tests/test_metrics.py
"""Tests for src/evaluation/metrics.py"""

import numpy as np
import pytest

from src.evaluation.metrics import PairedTTest, SegmentationMetrics


NUM_CLASSES = 4


@pytest.fixture
def metrics():
    return SegmentationMetrics(num_classes=NUM_CLASSES)


def test_overall_accuracy_perfect(metrics):
    y = np.array([0, 1, 2, 3, 0, 1])
    assert metrics.overall_accuracy(y, y) == pytest.approx(1.0)


def test_overall_accuracy_known(metrics):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 3 correct out of 4
    assert metrics.overall_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_mean_iou_perfect(metrics):
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    iou = metrics.mean_iou(y, y)
    assert iou == pytest.approx(1.0)


def test_compute_all_returns_expected_keys(metrics):
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    result = metrics.compute_all(y_true, y_pred)
    expected_keys = {'overall_accuracy', 'average_accuracy', 'mean_iou', 'macro_f1', 'per_class_f1'}
    assert expected_keys.issubset(result.keys())


def test_paired_ttest_returns_correct_keys():
    baseline = np.array([0.70] * 10)
    enhanced = np.array([0.75] * 10)
    result = PairedTTest.test(baseline, enhanced, alpha=0.01)
    assert {'t_statistic', 'critical_value', 'significant', 'mean_improvement', 'confidence'}.issubset(
        result.keys()
    )


def test_paired_ttest_significant_is_bool():
    baseline = np.array([0.70] * 10)
    enhanced = np.array([0.80] * 10)
    result = PairedTTest.test(baseline, enhanced, alpha=0.05)
    assert isinstance(result['significant'], (bool, np.bool_))


def test_paired_ttest_zero_difference_has_zero_tstat():
    baseline = np.array([0.75] * 10)
    enhanced = np.array([0.75] * 10)
    result = PairedTTest.test(baseline, enhanced, alpha=0.05)
    assert result['t_statistic'] == pytest.approx(0.0)
    assert result['mean_improvement'] == pytest.approx(0.0)
    assert result['significant'] is False
