# tests/test_data_loader.py
"""Tests for src/utils/data_loader.py"""

import numpy as np
import pytest

from src.utils.data_loader import HyperspectralDataLoader


@pytest.fixture
def loader(tmp_path):
    return HyperspectralDataLoader(data_dir=str(tmp_path))


def test_init_sets_up_five_datasets(loader):
    assert len(loader.datasets) == 5
    expected = {'indian_pines', 'pavia_university', 'salinas_valley', 'houston', 'botswana'}
    assert set(loader.datasets.keys()) == expected


def test_load_data_unknown_raises_value_error(loader):
    with pytest.raises(ValueError, match="Unknown dataset"):
        loader.load_data('nonexistent_dataset')


def test_load_data_missing_file_returns_none(loader):
    # No .mat files present in tmp_path
    data, labels = loader.load_data('indian_pines')
    assert data is None
    assert labels is None


def test_split_data_returns_correct_keys():
    loader = HyperspectralDataLoader(data_dir='./data/raw')
    # Build a synthetic label map (10×10, 2 classes, all labeled)
    rng = np.random.default_rng(42)
    labels = rng.integers(1, 3, size=(10, 10)).astype(np.int32)  # values 1 or 2
    result = loader.split_data(labels, train_ratio=0.7)
    assert {'train_mask', 'test_mask', 'train_indices', 'test_indices',
            'train_labels', 'test_labels'}.issubset(result.keys())


def test_split_data_mask_shapes():
    loader = HyperspectralDataLoader(data_dir='./data/raw')
    rng = np.random.default_rng(7)
    labels = rng.integers(1, 4, size=(8, 8)).astype(np.int32)
    result = loader.split_data(labels, train_ratio=0.7)
    assert result['train_mask'].shape == labels.shape
    assert result['test_mask'].shape == labels.shape
    # Masks should be disjoint
    assert not np.any(result['train_mask'] & result['test_mask'])


def test_get_pixel_spectra_correct_shape():
    loader = HyperspectralDataLoader(data_dir='./data/raw')
    rng = np.random.default_rng(3)
    data = rng.random((10, 10, 15)).astype(np.float32)
    indices = np.array([[0, 0], [1, 2], [5, 7]])
    spectra = loader.get_pixel_spectra(data, indices)
    assert spectra.shape == (3, 15)
