"""
Neuro-Symphony Phase 3 — Universal Early Fusion.

Early fusion combines modality feature representations into
one joint feature vector.

The implementation is dataset-independent.

Input:
    EEG feature vector
    fNIRS feature vector

Output:
    Concatenated multimodal feature vector.
"""

from __future__ import annotations

import numpy as np


def early_fusion(
    eeg_vector: np.ndarray,
    fnirs_vector: np.ndarray,
) -> np.ndarray:
    """
    Concatenate EEG and fNIRS feature representations.

    Parameters
    ----------
    eeg_vector:
        EEG feature vector.

    fnirs_vector:
        fNIRS feature vector.

    Returns
    -------
    np.ndarray
        Combined multimodal representation.
    """

    eeg = np.asarray(
        eeg_vector,
        dtype=np.float64,
    ).reshape(-1)

    fnirs = np.asarray(
        fnirs_vector,
        dtype=np.float64,
    ).reshape(-1)

    if eeg.size == 0:
        raise ValueError(
            "EEG feature vector cannot be empty."
        )

    if fnirs.size == 0:
        raise ValueError(
            "fNIRS feature vector cannot be empty."
        )

    if not np.isfinite(eeg).all():
        raise ValueError(
            "EEG feature vector contains "
            "NaN or infinite values."
        )

    if not np.isfinite(fnirs).all():
        raise ValueError(
            "fNIRS feature vector contains "
            "NaN or infinite values."
        )

    fused = np.concatenate(
        [eeg, fnirs]
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(fused).all():
        raise ValueError(
            "Early-fused representation contains "
            "NaN or infinite values."
        )

    return fused


__all__ = [
    "early_fusion",
]
