# tests/test_degree_ga.py
"""Tests for src/optimization/degree_ga.py"""

import numpy as np
import pytest

from src.optimization.degree_ga import DEGAHybrid


NUM_THRESHOLDS = 3
POP_SIZE = 10


@pytest.fixture
def dega():
    return DEGAHybrid(num_thresholds=NUM_THRESHOLDS, population_size=POP_SIZE, max_generations=5)


@pytest.fixture
def uniform_histogram():
    hist = np.ones(256, dtype=np.float32)
    hist /= hist.sum()
    return hist


def test_initialize_population_shape(dega):
    pop = dega.initialize_population()
    assert pop.shape == (POP_SIZE, NUM_THRESHOLDS)


def test_initialize_population_bounds(dega):
    pop = dega.initialize_population()
    assert pop.min() >= 0
    assert pop.max() <= 255


def test_otsu_fitness_non_negative(dega, uniform_histogram):
    thresholds = np.array([64, 128, 192])
    fitness = dega.otsu_fitness(thresholds, uniform_histogram)
    assert isinstance(float(fitness), float)
    assert fitness >= 0.0


def test_differential_mutation_within_bounds(dega):
    pop = dega.initialize_population()
    mutant = dega.differential_mutation(pop, idx=0)
    assert mutant.shape == (NUM_THRESHOLDS,)
    assert mutant.min() >= 0
    assert mutant.max() <= 255


def test_optimize_runs_without_error(dega):
    rng = np.random.default_rng(42)
    histogram = rng.integers(1, 100, size=256).astype(np.float32)
    optimal_thresholds, best_fitness = dega.optimize(histogram, verbose=False)
    assert optimal_thresholds.shape == (NUM_THRESHOLDS,)
    assert best_fitness >= 0.0
