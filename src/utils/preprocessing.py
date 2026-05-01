# src/utils/preprocessing.py
"""
Preprocessing utilities for hyperspectral image data.
"""

import numpy as np
from typing import Tuple


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalise a hyperspectral cube per band.

    Args:
        data:   Hyperspectral cube of shape (H, W, B) or (N, B).
        method: 'minmax' — scale each band to [0, 1];
                'zscore' — zero-mean unit-variance per band.

    Returns:
        Normalised array with the same shape as *data*.
    """
    data = data.astype(np.float32)
    result = np.empty_like(data)

    # Work along the last axis (bands)
    n_bands = data.shape[-1]
    for b in range(n_bands):
        band = data[..., b]
        if method == 'zscore':
            mu = band.mean()
            sigma = band.std()
            result[..., b] = (band - mu) / (sigma if sigma > 0 else 1.0)
        else:  # minmax (default)
            b_min = band.min()
            b_max = band.max()
            denom = b_max - b_min
            result[..., b] = (band - b_min) / (denom if denom > 0 else 1.0)

    return result


def remove_noisy_bands(data: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """
    Remove spectral bands whose variance is below *threshold*.

    Args:
        data:      Hyperspectral cube (H, W, B) or matrix (N, B).
        threshold: Bands with variance ≤ threshold are dropped.

    Returns:
        Array with noisy (near-constant) bands removed.
    """
    # Reshape to (N, B) to compute per-band variance
    original_shape = data.shape
    n_bands = original_shape[-1]
    flat = data.reshape(-1, n_bands).astype(np.float32)

    variances = flat.var(axis=0)
    keep_mask = variances > threshold

    filtered_flat = flat[:, keep_mask]

    # Restore spatial dimensions if present
    if data.ndim == 3:
        h, w = original_shape[:2]
        return filtered_flat.reshape(h, w, -1)
    return filtered_flat


def apply_pca(
    data: np.ndarray, n_components: int
) -> Tuple[np.ndarray, object]:
    """
    Apply PCA to reduce the spectral dimensionality of a hyperspectral cube.

    Args:
        data:         Hyperspectral cube (H, W, B) or matrix (N, B).
        n_components: Target number of principal components.

    Returns:
        transformed: Reduced array — shape (H, W, n_components) or (N, n_components).
        pca_model:   Fitted ``sklearn.decomposition.PCA`` instance.
    """
    from sklearn.decomposition import PCA

    original_shape = data.shape
    n_bands = original_shape[-1]
    flat = data.reshape(-1, n_bands).astype(np.float32)

    pca = PCA(n_components=n_components)
    transformed_flat = pca.fit_transform(flat)

    if data.ndim == 3:
        h, w = original_shape[:2]
        return transformed_flat.reshape(h, w, n_components), pca
    return transformed_flat, pca


def extract_roi_spectra(
    data: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pixel spectra for all True positions in *mask*.

    Args:
        data: Hyperspectral cube (H, W, B).
        mask: Boolean mask (H, W).

    Returns:
        spectra: Spectra matrix of shape (N_masked, B).
        indices: Array of shape (N_masked, 2) with (row, col) coordinates.
    """
    indices = np.argwhere(mask)
    spectra = data[mask]
    return spectra, indices


def augment_training_data(
    spectra: np.ndarray,
    labels: np.ndarray,
    factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training spectra by adding Gaussian noise.

    Args:
        spectra: Training spectra (N, B).
        labels:  Corresponding labels (N,).
        factor:  Multiplicative factor; *factor* − 1 noisy copies are appended
                 so the output has *factor* × N samples.

    Returns:
        aug_spectra: Augmented spectra (factor * N, B).
        aug_labels:  Corresponding labels (factor * N,).
    """
    aug_spectra = [spectra]
    aug_labels = [labels]

    std = spectra.std(axis=0)

    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.01 * std, size=spectra.shape).astype(spectra.dtype)
        aug_spectra.append(spectra + noise)
        aug_labels.append(labels)

    return np.concatenate(aug_spectra, axis=0), np.concatenate(aug_labels, axis=0)
