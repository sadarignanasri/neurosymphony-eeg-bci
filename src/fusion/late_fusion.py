"""
Neuro-Symphony Phase 3 — Universal Late Fusion.

Late fusion combines modality-specific decision representations.

The fusion layer does not train a classifier. It receives
already-computed modality decision/score vectors and combines
them using configurable weights.

The implementation is dataset-independent.
"""

from __future__ import annotations

import numpy as np


def _validate_score_vector(
    scores: np.ndarray,
    name: str,
) -> np.ndarray:

    values = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(
            f"{name} score vector cannot be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} score vector contains "
            "NaN or infinite values."
        )

    return values


def late_fusion(
    eeg_scores: np.ndarray,
    fnirs_scores: np.ndarray,
    eeg_weight: float = 0.5,
    fnirs_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine EEG and fNIRS decision-score vectors.

    Both modalities must produce the same number of scores,
    for example one score per class.

    Parameters
    ----------
    eeg_scores:
        EEG decision/probability vector.

    fnirs_scores:
        fNIRS decision/probability vector.

    eeg_weight:
        Weight assigned to EEG.

    fnirs_weight:
        Weight assigned to fNIRS.

    Returns
    -------
    np.ndarray
        Fused decision representation.
    """

    eeg = _validate_score_vector(
        eeg_scores,
        "EEG",
    )

    fnirs = _validate_score_vector(
        fnirs_scores,
        "fNIRS",
    )

    if eeg.shape != fnirs.shape:
        raise ValueError(
            "EEG and fNIRS score vectors must "
            "have the same shape for late fusion."
        )

    if not np.isfinite(eeg_weight):
        raise ValueError(
            "eeg_weight must be finite."
        )

    if not np.isfinite(fnirs_weight):
        raise ValueError(
            "fnirs_weight must be finite."
        )

    if eeg_weight < 0 or fnirs_weight < 0:
        raise ValueError(
            "Fusion weights must be non-negative."
        )

    total_weight = (
        eeg_weight + fnirs_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "At least one fusion weight must be positive."
        )

    eeg_weight = (
        eeg_weight / total_weight
    )

    fnirs_weight = (
        fnirs_weight / total_weight
    )

    fused = (
        eeg_weight * eeg
        +
        fnirs_weight * fnirs
    )

    fused = np.asarray(
        fused,
        dtype=np.float64,
    )

    if not np.isfinite(fused).all():
        raise ValueError(
            "Late-fused representation contains "
            "NaN or infinite values."
        )

    return fused


__all__ = [
    "late_fusion",
]