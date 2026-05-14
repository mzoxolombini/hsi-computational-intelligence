# tests/test_watershed_ga.py
"""Tests for src/optimization/watershed_ga.py"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not available")

from src.optimization.watershed_ga import WatershedGA  # noqa: E402


@pytest.fixture
def wsga():
    return WatershedGA(population_size=10, max_generations=5)


DEFAULT_PARAMS = {
    'n_pca': 5,
    'sigma': 1.0,
    'tau_grad': 0.5,
    's_min': 100,
    'classifier': 1,
}


def test_encode_decode_roundtrip(wsga):
    chromosome = wsga.encode_chromosome(DEFAULT_PARAMS)
    decoded = wsga.decode_chromosome(chromosome)
    assert decoded['n_pca'] == DEFAULT_PARAMS['n_pca']
    assert abs(decoded['sigma'] - DEFAULT_PARAMS['sigma']) < 1e-5
    assert abs(decoded['tau_grad'] - DEFAULT_PARAMS['tau_grad']) < 1e-5
    assert decoded['s_min'] == DEFAULT_PARAMS['s_min']
    assert decoded['classifier'] == DEFAULT_PARAMS['classifier']


def test_initialize_population_size(wsga):
    population = wsga._initialize_population()
    assert len(population) == wsga.population_size


def test_initialize_population_chromosome_length(wsga):
    population = wsga._initialize_population()
    assert all(len(ind) == 5 for ind in population)


def test_coverage_penalty_fully_labeled(wsga):
    # A segmentation where every pixel is labeled → coverage_ratio = 1.0 → penalty = 0
    full_seg = np.ones((20, 20), dtype=int)
    penalty = wsga.coverage_penalty(full_seg)
    assert penalty == pytest.approx(0.0)


def test_coverage_penalty_partially_labeled(wsga):
    # Half the pixels unlabeled → ratio = 0.5 < 0.95 → non-zero penalty
    seg = np.zeros((20, 20), dtype=int)
    seg[:10, :] = 1
    penalty = wsga.coverage_penalty(seg)
    assert penalty > 0.0


def test_smoothness_penalty_few_regions(wsga):
    seg = np.ones((20, 20), dtype=int)
    assert wsga.smoothness_penalty(seg) == pytest.approx(0.0)


def test_encode_chromosome_shape(wsga):
    chrom = wsga.encode_chromosome(DEFAULT_PARAMS)
    assert chrom.shape == (5,)


def test_decode_chromosome_supports_knn_classifier(wsga):
    chromosome = np.array([5, 1.0, 0.5, 100, 2])
    decoded = wsga.decode_chromosome(chromosome)
    assert decoded['classifier'] == 2
