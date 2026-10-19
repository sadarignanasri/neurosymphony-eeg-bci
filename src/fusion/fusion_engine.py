"""
Neuro-Symphony Phase 3 — Universal Fusion Engine.

Provides one standardized interface for:

    - Early fusion
    - Late fusion
    - Hybrid fusion

No dataset-specific information is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.fusion.early_fusion import early_fusion
from src.fusion.late_fusion import late_fusion
from src.fusion.hybrid_fusion import hybrid_fusion


@dataclass(frozen=True)
class FusionResult:
    """
    Standardized fusion result.
    """

    method: str
    vector: np.ndarray

    eeg_dimension: int
    fnirs_dimension: int

    output_dimension: int


def fuse_early(
    eeg_vector: np.ndarray,
    fnirs_vector: np.ndarray,
) -> FusionResult:

    eeg = np.asarray(
        eeg_vector,
        dtype=np.float64,
    ).reshape(-1)

    fnirs = np.asarray(
        fnirs_vector,
        dtype=np.float64,
    ).reshape(-1)

    vector = early_fusion(
        eeg,
        fnirs,
    )

    return FusionResult(
        method="early",
        vector=vector,
        eeg_dimension=int(eeg.size),
        fnirs_dimension=int(fnirs.size),
        output_dimension=int(vector.size),
    )


def fuse_late(
    eeg_scores: np.ndarray,
    fnirs_scores: np.ndarray,
    eeg_weight: float = 0.5,
    fnirs_weight: float = 0.5,
) -> FusionResult:

    eeg = np.asarray(
        eeg_scores,
        dtype=np.float64,
    ).reshape(-1)

    fnirs = np.asarray(
        fnirs_scores,
        dtype=np.float64,
    ).reshape(-1)

    vector = late_fusion(
        eeg,
        fnirs,
        eeg_weight=eeg_weight,
        fnirs_weight=fnirs_weight,
    )

    return FusionResult(
        method="late",
        vector=vector,
        eeg_dimension=int(eeg.size),
        fnirs_dimension=int(fnirs.size),
        output_dimension=int(vector.size),
    )


def fuse_hybrid(
    eeg_vector: np.ndarray,
    fnirs_vector: np.ndarray,
    eeg_scores: np.ndarray,
    fnirs_scores: np.ndarray,
    eeg_weight: float = 0.5,
    fnirs_weight: float = 0.5,
) -> FusionResult:

    eeg = np.asarray(
        eeg_vector,
        dtype=np.float64,
    ).reshape(-1)

    fnirs = np.asarray(
        fnirs_vector,
        dtype=np.float64,
    ).reshape(-1)

    vector = hybrid_fusion(
        eeg,
        fnirs,
        eeg_scores,
        fnirs_scores,
        eeg_weight=eeg_weight,
        fnirs_weight=fnirs_weight,
    )

    return FusionResult(
        method="hybrid",
        vector=vector,
        eeg_dimension=int(eeg.size),
        fnirs_dimension=int(fnirs.size),
        output_dimension=int(vector.size),
    )


__all__ = [
    "FusionResult",
    "fuse_early",
    "fuse_late",
    "fuse_hybrid",
]