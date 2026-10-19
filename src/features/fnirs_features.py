"""
Neuro-Symphony Phase 3 — Universal fNIRS Feature Extraction.

This module extracts reusable fNIRS features from task-locked
Phase 3 fNIRS epochs.

The implementation is dataset-independent.

Input:
    Epoch objects from src.epochs.epoch_extractor

Output:
    Standardized FNIRSFeature objects containing time-domain
    and hemodynamic-response features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.epochs.epoch_extractor import Epoch


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class FNIRSFeature:
    """
    Feature representation for one fNIRS epoch.

    Features are represented independently for every fNIRS
    channel.

    Each feature array has shape:

        (n_channels,)
    """

    subject: str | None
    trial_type: str

    event_onset: float

    channel_names: tuple[str, ...]

    sampling_rate: float

    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    peak: np.ndarray
    peak_time: np.ndarray

    baseline_mean: np.ndarray
    response_mean: np.ndarray
    response_change: np.ndarray

    feature_names: tuple[str, ...]

    modality: str = "fNIRS"


# ============================================================
# VALIDATION
# ============================================================

def _validate_epoch(epoch: Epoch) -> None:
    """Validate one fNIRS epoch."""

    if not isinstance(epoch, Epoch):
        raise TypeError(
            "fNIRS feature extraction requires an Epoch object."
        )

    if epoch.modality != "fNIRS":
        raise ValueError(
            "fNIRS feature extraction received a non-fNIRS epoch."
        )

    if epoch.data.ndim != 2:
        raise ValueError(
            "fNIRS epoch data must have shape "
            "(channels, samples)."
        )

    if epoch.data.shape[0] != len(epoch.channel_names):
        raise ValueError(
            "fNIRS channel count does not match channel_names."
        )

    if epoch.sampling_rate <= 0:
        raise ValueError(
            "fNIRS sampling rate must be positive."
        )

    if epoch.data.shape[1] < 2:
        raise ValueError(
            "fNIRS epoch must contain at least two samples."
        )

    if not np.isfinite(epoch.data).all():
        raise ValueError(
            "fNIRS epoch contains NaN or infinite values."
        )


# ============================================================
# SINGLE EPOCH
# ============================================================

def extract_fnirs_feature(
    epoch: Epoch,
) -> FNIRSFeature:
    """
    Extract fNIRS features from one task epoch.

    The feature set is intentionally dataset-independent.

    Features include:

        - mean
        - variance
        - standard deviation
        - minimum
        - maximum
        - peak response
        - peak response time
        - baseline mean
        - response mean
        - response change

    The epoch itself is task-locked by the universal Phase 3
    epoch extractor.

    Returns
    -------
    FNIRSFeature
        Standardized fNIRS feature representation.
    """

    _validate_epoch(epoch)

    data = np.asarray(
        epoch.data,
        dtype=np.float64,
    )

    n_channels, n_samples = data.shape

    sampling_rate = float(
        epoch.sampling_rate
    )

    # ========================================================
    # TIME-DOMAIN FEATURES
    # ========================================================

    mean = np.mean(
        data,
        axis=1,
    )

    variance = np.var(
        data,
        axis=1,
    )

    std = np.std(
        data,
        axis=1,
    )

    minimum = np.min(
        data,
        axis=1,
    )

    maximum = np.max(
        data,
        axis=1,
    )

    # ========================================================
    # PEAK RESPONSE
    # ========================================================

    # fNIRS epochs are represented as task-locked signals.
    #
    # The absolute maximum is used as a deterministic,
    # modality-independent response descriptor.

    peak_indices = np.argmax(
        np.abs(data),
        axis=1,
    )

    channel_indices = np.arange(
        n_channels
    )

    peak = data[
        channel_indices,
        peak_indices,
    ]

    peak_time = (
        peak_indices / sampling_rate
    )

    # ========================================================
    # BASELINE / RESPONSE FEATURES
    # ========================================================

    # The epoch configuration currently uses tmin=0 and
    # tmax=3 seconds.
    #
    # Because the universal epoch extractor does not assume
    # a dataset-specific baseline interval, use the first
    # 20% of the epoch as a generic baseline region and the
    # remaining 80% as the task-response region.
    #
    # This keeps the implementation dataset-independent.

    baseline_samples = max(
        1,
        int(
            np.floor(
                0.20 * n_samples
            )
        ),
    )

    baseline_data = data[
        :,
        :baseline_samples,
    ]

    response_data = data[
        :,
        baseline_samples:,
    ]

    baseline_mean = np.mean(
        baseline_data,
        axis=1,
    )

    response_mean = np.mean(
        response_data,
        axis=1,
    )

    response_change = (
        response_mean - baseline_mean
    )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = (
        "mean",
        "variance",
        "std",
        "minimum",
        "maximum",
        "peak",
        "peak_time",
        "baseline_mean",
        "response_mean",
        "response_change",
    )

    # ========================================================
    # INTEGRITY CHECK
    # ========================================================

    feature_arrays = (
        mean,
        variance,
        std,
        minimum,
        maximum,
        peak,
        peak_time,
        baseline_mean,
        response_mean,
        response_change,
    )

    for values in feature_arrays:

        if not np.isfinite(values).all():
            raise ValueError(
                "fNIRS feature extraction produced "
                "NaN or infinite values."
            )

    # ========================================================
    # RESULT
    # ========================================================

    return FNIRSFeature(
        subject=epoch.subject,
        trial_type=epoch.trial_type,
        event_onset=float(
            epoch.event_onset
        ),
        channel_names=tuple(
            epoch.channel_names
        ),
        sampling_rate=sampling_rate,
        mean=mean,
        variance=variance,
        std=std,
        minimum=minimum,
        maximum=maximum,
        peak=peak,
        peak_time=peak_time,
        baseline_mean=baseline_mean,
        response_mean=response_mean,
        response_change=response_change,
        feature_names=feature_names,
    )


# ============================================================
# MULTIPLE EPOCHS
# ============================================================

def extract_fnirs_features(
    epochs: Sequence[Epoch],
) -> tuple[FNIRSFeature, ...]:
    """
    Extract fNIRS features from multiple epochs.

    Parameters
    ----------
    epochs:
        Sequence of universal fNIRS Epoch objects.

    Returns
    -------
    tuple[FNIRSFeature, ...]
        One FNIRSFeature object per input epoch.
    """

    return tuple(
        extract_fnirs_feature(
            epoch
        )
        for epoch in epochs
    )


# ============================================================
# FLAT FEATURE VECTOR
# ============================================================

def fnirs_feature_vector(
    feature: FNIRSFeature,
) -> np.ndarray:
    """
    Convert one FNIRSFeature object into a flat numeric vector.

    This representation is intended for downstream
    multimodal representation and fusion.

    For N fNIRS channels, the vector contains:

        10 features × N channels
    """

    if not isinstance(
        feature,
        FNIRSFeature,
    ):
        raise TypeError(
            "fnirs_feature_vector requires "
            "an FNIRSFeature object."
        )

    values = [
        feature.mean,
        feature.variance,
        feature.std,
        feature.minimum,
        feature.maximum,
        feature.peak,
        feature.peak_time,
        feature.baseline_mean,
        feature.response_mean,
        feature.response_change,
    ]

    vector = np.concatenate(
        values
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(vector).all():
        raise ValueError(
            "fNIRS feature vector contains "
            "NaN or infinite values."
        )

    return vector


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "FNIRSFeature",
    "extract_fnirs_feature",
    "extract_fnirs_features",
    "fnirs_feature_vector",
]