"""
Neuro-Symphony Phase 3 — Universal Hybrid Fusion.

Hybrid fusion combines:

    1. Early feature-level fusion
    2. Late decision-level fusion

This produces a representation containing both detailed
modality features and modality-level decision information.

The implementation is dataset-independent.
"""

from __future__ import annotations

import numpy as np

from src.fusion.early_fusion import early_fusion
from src.fusion.late_fusion import late_fusion


def hybrid_fusion(
    eeg_vector: np.ndarray,
    fnirs_vector: np.ndarray,
    eeg_scores: np.ndarray,
    fnirs_scores: np.ndarray,
    eeg_weight: float = 0.5,
    fnirs_weight: float = 0.5,
) -> np.ndarray:
    """
    Create a hybrid multimodal representation.

    The output is:

        [early-fused features | late-fused scores]

    Parameters
    ----------
    eeg_vector:
        EEG feature representation.

    fnirs_vector:
        fNIRS feature representation.

    eeg_scores:
        EEG decision-score vector.

    fnirs_scores:
        fNIRS decision-score vector.

    eeg_weight:
        EEG late-fusion weight.

    fnirs_weight:
        fNIRS late-fusion weight.

    Returns
    -------
    np.ndarray
        Hybrid multimodal representation.
    """

    early = early_fusion(
        eeg_vector,
        fnirs_vector,
    )

    late = late_fusion(
        eeg_scores,
        fnirs_scores,
        eeg_weight=eeg_weight,
        fnirs_weight=fnirs_weight,
    )

    hybrid = np.concatenate(
        [early, late]
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(hybrid).all():
        raise ValueError(
            "Hybrid-fused representation contains "
            "NaN or infinite values."
        )

    return hybrid


__all__ = [
    "hybrid_fusion",
]