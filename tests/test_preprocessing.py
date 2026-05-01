# tests/test_preprocessing.py
"""Tests for src/utils/preprocessing.py"""

import numpy as np
import pytest

from src.utils.preprocessing import (
    apply_pca,
    augment_training_data,
    extract_roi_spectra,
    normalize_data,
    remove_noisy_bands,
)


@pytest.fixture
def synthetic_cube():
    """Small (10, 10, 20) float cube with random values."""
    rng = np.random.default_rng(0)
    return rng.random((10, 10, 20)).astype(np.float32)


def test_normalize_data_minmax_range(synthetic_cube):
    result = normalize_data(synthetic_cube, method='minmax')
    assert result.shape == synthetic_cube.shape
    for b in range(result.shape[-1]):
        band = result[..., b]
        assert band.min() >= 0.0 - 1e-6
        assert band.max() <= 1.0 + 1e-6


def test_normalize_data_zscore_mean(synthetic_cube):
    result = normalize_data(synthetic_cube, method='zscore')
    assert result.shape == synthetic_cube.shape
    for b in range(result.shape[-1]):
        assert abs(result[..., b].mean()) < 1e-4


def test_remove_noisy_bands_removes_constant_bands():
    data = np.random.rand(8, 8, 10).astype(np.float32)
    # Make the first two bands constant
    data[:, :, 0] = 0.5
    data[:, :, 1] = 0.3
    result = remove_noisy_bands(data, threshold=0.001)
    assert result.shape[-1] == 8  # 2 constant bands removed


def test_remove_noisy_bands_keeps_all_when_varied(synthetic_cube):
    result = remove_noisy_bands(synthetic_cube, threshold=0.0)
    assert result.shape[-1] == synthetic_cube.shape[-1]


def test_apply_pca_reduces_dimensionality(synthetic_cube):
    n_components = 5
    transformed, pca_model = apply_pca(synthetic_cube, n_components=n_components)
    assert transformed.shape == (10, 10, n_components)
    assert pca_model.n_components_ == n_components


def test_extract_roi_spectra_correct_shape(synthetic_cube):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:6] = True  # 9 pixels
    spectra, indices = extract_roi_spectra(synthetic_cube, mask)
    assert spectra.shape == (9, 20)
    assert indices.shape == (9, 2)


def test_extract_roi_spectra_values_match(synthetic_cube):
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    spectra, indices = extract_roi_spectra(synthetic_cube, mask)
    np.testing.assert_array_equal(spectra[0], synthetic_cube[0, 0, :])


def test_augment_training_data_factor_2():
    rng = np.random.default_rng(1)
    spectra = rng.random((20, 15)).astype(np.float32)
    labels = np.arange(20)
    aug_s, aug_l = augment_training_data(spectra, labels, factor=2)
    assert aug_s.shape == (40, 15)
    assert aug_l.shape == (40,)
