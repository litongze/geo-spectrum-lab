"""Shared geometry and amplitude weights for complex neighbor sources."""
from __future__ import annotations

import numpy as np


def normalize_real(weight: np.ndarray) -> np.ndarray:
    """Normalize signed real coefficients while guarding singular rows."""
    denominator = weight.sum(axis=1, keepdims=True)
    return weight / np.where(
        np.abs(denominator) > 1e-12, denominator, 1.0
    )


def anisotropic_weights(
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
    radial_ratio: float,
    distance_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized anisotropic IDW weights and effective distances."""
    effective = np.sqrt(
        np.square(radial_delta * radial_ratio)
        + np.square(tangent_delta)
    )
    weight = 1.0 / np.maximum(effective, 0.05) ** distance_power
    return normalize_real(weight), effective


def moment_correct_weights(
    base: np.ndarray,
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
    ridge: float,
    strength: float,
) -> np.ndarray:
    """Reduce the weighted first spatial moment with ridge regularization."""
    if ridge <= 0:
        raise ValueError("moment ridge must be positive")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("moment strength must be in [0, 1]")

    scale = np.maximum(
        np.median(
            np.sqrt(radial_delta**2 + tangent_delta**2),
            axis=1,
            keepdims=True,
        ),
        0.3,
    )
    delta = np.stack(
        [radial_delta / scale, tangent_delta / scale], axis=1
    )
    moment = np.einsum("bdk,bk->bd", delta, base)
    covariance = np.einsum(
        "bdk,bk,bek->bde", delta, base, delta
    )
    solved = np.linalg.solve(
        covariance
        + ridge * np.eye(2, dtype=np.float64)[None],
        moment[..., None],
    )[..., 0]
    correction = base * np.einsum(
        "bdk,bd->bk", delta, solved
    )
    return normalize_real(base - strength * correction)


def equalize_amplitude(
    coefficient: np.ndarray,
    amplitude: np.ndarray,
) -> np.ndarray:
    """Equalize neighbor RMS while preserving signed/complex coefficients."""
    magnitude = np.abs(coefficient)
    target = (magnitude * amplitude).sum(axis=1, keepdims=True)
    target /= np.maximum(
        magnitude.sum(axis=1, keepdims=True), 1e-30
    )
    return coefficient * target / np.maximum(amplitude, 1e-30)
