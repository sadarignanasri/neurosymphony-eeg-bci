"""
Neuro-Symphony Phase 3 — Universal Multimodal Representation.

Combines synchronized EEG and fNIRS feature representations.

The implementation is dataset-independent.

Input:
    EEGFeature objects
    FNIRSFeature objects

Output:
    MultimodalFeature objects containing:
        - paired metadata
        - EEG feature vector
        - fNIRS feature vector
        - concatenated multimodal vector

The EEG and fNIRS features must correspond to the same
subject, task, and event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.features.eeg_features import (
    EEGFeature,
    eeg_feature_vector,
)
from src.features.fnirs_features import (
    FNIRSFeature,
    fnirs_feature_vector,
)


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class MultimodalFeature:
    """
    Universal multimodal feature representation for one event.

    The representation preserves EEG and fNIRS vectors
    separately while also providing a concatenated vector
    for downstream fusion.
    """

    subject: str | None
    trial_type: str

    event_onset_eeg: float
    event_onset_fnirs: float

    eeg_vector: np.ndarray
    fnirs_vector: np.ndarray

    combined_vector: np.ndarray

    eeg_feature_dimension: int
    fnirs_feature_dimension: int

    feature_dimension: int

    modality: str = "EEG-fNIRS"


# ============================================================
# VALIDATION
# ============================================================

def _validate_feature_pair(
    eeg: EEGFeature,
    fnirs: FNIRSFeature,
) -> None:
    """Validate that EEG and fNIRS features can be paired."""

    if not isinstance(eeg, EEGFeature):
        raise TypeError(
            "Expected EEGFeature object."
        )

    if not isinstance(fnirs, FNIRSFeature):
        raise TypeError(
            "Expected FNIRSFeature object."
        )

    if eeg.modality != "EEG":
        raise ValueError(
            "First feature must be an EEG feature."
        )

    if fnirs.modality != "fNIRS":
        raise ValueError(
            "Second feature must be an fNIRS feature."
        )

    # --------------------------------------------------------
    # Subject consistency
    # --------------------------------------------------------

    if (
        eeg.subject is not None
        and fnirs.subject is not None
        and eeg.subject != fnirs.subject
    ):
        raise ValueError(
            "EEG and fNIRS features belong to "
            "different subjects."
        )

    # --------------------------------------------------------
    # Task consistency
    # --------------------------------------------------------

    if eeg.trial_type != fnirs.trial_type:
        raise ValueError(
            "EEG and fNIRS features have different "
            "trial types."
        )

    # --------------------------------------------------------
    # Event timing consistency
    #
    # The timestamps may belong to different clock domains,
    # so exact equality is NOT required here.
    # Synchronization is represented explicitly by the two
    # onset timestamps.
    # --------------------------------------------------------

    if not np.isfinite(eeg.event_onset):
        raise ValueError(
            "EEG event onset must be finite."
        )

    if not np.isfinite(fnirs.event_onset):
        raise ValueError(
            "fNIRS event onset must be finite."
        )


# ============================================================
# SINGLE MULTIMODAL REPRESENTATION
# ============================================================

def extract_multimodal_feature(
    eeg: EEGFeature,
    fnirs: FNIRSFeature,
) -> MultimodalFeature:
    """
    Combine one synchronized EEG feature and one fNIRS feature.

    Parameters
    ----------
    eeg:
        EEG feature representation for one event.

    fnirs:
        fNIRS feature representation for the corresponding
        event.

    Returns
    -------
    MultimodalFeature
        Combined universal multimodal representation.
    """

    _validate_feature_pair(
        eeg,
        fnirs,
    )

    eeg_vector = np.asarray(
        eeg_feature_vector(eeg),
        dtype=np.float64,
    )

    fnirs_vector = np.asarray(
        fnirs_feature_vector(fnirs),
        dtype=np.float64,
    )

    if eeg_vector.ndim != 1:
        raise ValueError(
            "EEG feature vector must be one-dimensional."
        )

    if fnirs_vector.ndim != 1:
        raise ValueError(
            "fNIRS feature vector must be one-dimensional."
        )

    combined_vector = np.concatenate(
        [
            eeg_vector,
            fnirs_vector,
        ]
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        combined_vector
    ).all():
        raise ValueError(
            "Multimodal feature vector contains "
            "NaN or infinite values."
        )

    return MultimodalFeature(
        subject=(
            eeg.subject
            if eeg.subject is not None
            else fnirs.subject
        ),
        trial_type=eeg.trial_type,
        event_onset_eeg=float(
            eeg.event_onset
        ),
        event_onset_fnirs=float(
            fnirs.event_onset
        ),
        eeg_vector=eeg_vector,
        fnirs_vector=fnirs_vector,
        combined_vector=combined_vector,
        eeg_feature_dimension=int(
            eeg_vector.size
        ),
        fnirs_feature_dimension=int(
            fnirs_vector.size
        ),
        feature_dimension=int(
            combined_vector.size
        ),
    )


# ============================================================
# MULTIPLE REPRESENTATIONS
# ============================================================

def extract_multimodal_features(
    eeg_features: Sequence[EEGFeature],
    fnirs_features: Sequence[FNIRSFeature],
) -> tuple[MultimodalFeature, ...]:
    """
    Combine corresponding EEG and fNIRS features.

    The two sequences must contain the same number of
    feature objects and must already be aligned in event order.
    """

    if len(eeg_features) != len(fnirs_features):
        raise ValueError(
            "EEG and fNIRS feature sequences must contain "
            "the same number of events."
        )

    representations = []

    for eeg, fnirs in zip(
        eeg_features,
        fnirs_features,
    ):
        representations.append(
            extract_multimodal_feature(
                eeg,
                fnirs,
            )
        )

    return tuple(
        representations
    )


# ============================================================
# FLAT REPRESENTATION
# ============================================================

def multimodal_feature_vector(
    feature: MultimodalFeature,
) -> np.ndarray:
    """
    Return the flat multimodal feature vector.

    Layout:

        [EEG features | fNIRS features]
    """

    if not isinstance(
        feature,
        MultimodalFeature,
    ):
        raise TypeError(
            "Expected MultimodalFeature object."
        )

    vector = np.asarray(
        feature.combined_vector,
        dtype=np.float64,
    )

    if vector.ndim != 1:
        raise ValueError(
            "Multimodal feature vector must be one-dimensional."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Multimodal feature vector contains "
            "NaN or infinite values."
        )

    return vector.copy()


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "MultimodalFeature",
    "extract_multimodal_feature",
    "extract_multimodal_features",
    "multimodal_feature_vector",
]
