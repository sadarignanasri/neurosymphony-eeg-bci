"""
Neuro-Symphony Phase 3 — Universal EEG Feature Extraction.

This module extracts reusable EEG features from task-locked
Phase 3 EEG epochs.

The implementation is dataset-independent.

Input:
    Epoch objects from src.epochs.epoch_extractor

Output:
    Standardized EEGFeature objects containing time-domain
    and frequency-domain features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import welch

from src.epochs.epoch_extractor import Epoch


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

DEFAULT_FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class EEGFeature:
    """
    Feature representation for one EEG epoch.

    Features are represented per EEG channel.

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
    rms: np.ndarray

    band_power: dict[str, np.ndarray]

    relative_band_power: dict[str, np.ndarray]

    total_power: np.ndarray

    feature_names: tuple[str, ...]

    modality: str = "EEG"


# ============================================================
# VALIDATION
# ============================================================

def _validate_epoch(epoch: Epoch) -> None:
    """Validate one EEG epoch."""

    if not isinstance(epoch, Epoch):
        raise TypeError(
            "EEG feature extraction requires an Epoch object."
        )

    if epoch.modality != "EEG":
        raise ValueError(
            "EEG feature extraction received a non-EEG epoch."
        )

    if epoch.data.ndim != 2:
        raise ValueError(
            "EEG epoch data must have shape "
            "(channels, samples)."
        )

    if epoch.data.shape[0] != len(epoch.channel_names):
        raise ValueError(
            "EEG channel count does not match channel_names."
        )

    if epoch.sampling_rate <= 0:
        raise ValueError(
            "EEG sampling rate must be positive."
        )

    if not np.isfinite(epoch.data).all():
        raise ValueError(
            "EEG epoch contains NaN or infinite values."
        )


# ============================================================
# POWER SPECTRAL DENSITY
# ============================================================

def _compute_psd(
    data: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density for every EEG channel.

    Returns
    -------
    frequencies:
        Frequency vector.

    psd:
        PSD with shape:

            (channels, frequencies)
    """

    n_samples = data.shape[1]

    # Keep Welch robust for both short and long epochs.
    nperseg = min(
        n_samples,
        int(round(sampling_rate)),
    )

    if nperseg < 2:
        raise ValueError(
            "EEG epoch is too short for PSD calculation."
        )

    frequencies, psd = welch(
        data,
        fs=sampling_rate,
        axis=1,
        nperseg=nperseg,
        detrend="constant",
    )

    return frequencies, psd


# ============================================================
# BAND POWER
# ============================================================

def _band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_frequency: float,
    high_frequency: float,
) -> np.ndarray:
    """
    Calculate absolute power inside one frequency band.

    Returns one value per EEG channel.
    """

    mask = (
        (frequencies >= low_frequency)
        & (frequencies < high_frequency)
    )

    if not np.any(mask):
        return np.zeros(
            psd.shape[0],
            dtype=np.float64,
        )

    return np.trapezoid(
        psd[:, mask],
        frequencies[mask],
        axis=1,
    )


# ============================================================
# SINGLE EPOCH
# ============================================================

def extract_eeg_feature(
    epoch: Epoch,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> EEGFeature:
    """
    Extract EEG features from one task epoch.

    Parameters
    ----------
    epoch:
        Universal Phase 3 EEG Epoch.

    frequency_bands:
        Optional custom frequency-band definition.

    Returns
    -------
    EEGFeature
        Standardized EEG feature representation.
    """

    _validate_epoch(epoch)

    if frequency_bands is None:
        frequency_bands = DEFAULT_FREQUENCY_BANDS

    data = np.asarray(
        epoch.data,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

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

    rms = np.sqrt(
        np.mean(
            np.square(data),
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    frequencies, psd = _compute_psd(
        data,
        float(epoch.sampling_rate),
    )

    band_power: dict[str, np.ndarray] = {}

    for band_name, (
        low_frequency,
        high_frequency,
    ) in frequency_bands.items():

        if low_frequency < 0:
            raise ValueError(
                f"Frequency band '{band_name}' "
                "has a negative lower frequency."
            )

        if high_frequency <= low_frequency:
            raise ValueError(
                f"Frequency band '{band_name}' "
                "has invalid frequency limits."
            )

        band_power[band_name] = _band_power(
            frequencies,
            psd,
            low_frequency,
            high_frequency,
        )

    # --------------------------------------------------------
    # Total spectral power
    #
    # Sum the configured bands rather than using arbitrary
    # frequencies outside the feature definition.
    # --------------------------------------------------------

    total_power = np.zeros(
        data.shape[0],
        dtype=np.float64,
    )

    for values in band_power.values():
        total_power += values

    # --------------------------------------------------------
    # Relative band power
    # --------------------------------------------------------

    relative_band_power: dict[str, np.ndarray] = {}

    denominator = np.maximum(
        total_power,
        np.finfo(np.float64).eps,
    )

    for band_name, values in band_power.items():

        relative_band_power[band_name] = (
            values / denominator
        )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        "mean",
        "variance",
        "std",
        "rms",
        *[
            f"{band}_power"
            for band in frequency_bands
        ],
        *[
            f"{band}_relative_power"
            for band in frequency_bands
        ],
        "total_power",
    )

    return EEGFeature(
        subject=epoch.subject,
        trial_type=epoch.trial_type,
        event_onset=float(epoch.event_onset),
        channel_names=tuple(epoch.channel_names),
        sampling_rate=float(epoch.sampling_rate),
        mean=mean,
        variance=variance,
        std=std,
        rms=rms,
        band_power=band_power,
        relative_band_power=relative_band_power,
        total_power=total_power,
        feature_names=tuple(feature_names),
    )


# ============================================================
# MULTIPLE EPOCHS
# ============================================================

def extract_eeg_features(
    epochs: Sequence[Epoch],
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> tuple[EEGFeature, ...]:
    """
    Extract EEG features from multiple epochs.

    Parameters
    ----------
    epochs:
        Sequence of universal EEG Epoch objects.

    frequency_bands:
        Optional custom frequency-band definition.

    Returns
    -------
    tuple[EEGFeature, ...]
        One EEGFeature object per input epoch.
    """

    return tuple(
        extract_eeg_feature(
            epoch,
            frequency_bands=frequency_bands,
        )
        for epoch in epochs
    )


# ============================================================
# FLAT FEATURE VECTOR
# ============================================================

def eeg_feature_vector(
    feature: EEGFeature,
) -> np.ndarray:
    """
    Convert one EEGFeature object into a flat numeric vector.

    This representation is intended for downstream
    multimodal representation and fusion.
    """

    if not isinstance(feature, EEGFeature):
        raise TypeError(
            "eeg_feature_vector requires an EEGFeature object."
        )

    values = [
        feature.mean,
        feature.variance,
        feature.std,
        feature.rms,
    ]

    for band in feature.band_power:
        values.append(
            feature.band_power[band]
        )

    for band in feature.relative_band_power:
        values.append(
            feature.relative_band_power[band]
        )

    values.append(
        feature.total_power
    )

    vector = np.concatenate(
        values
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(vector).all():
        raise ValueError(
            "EEG feature vector contains "
            "NaN or infinite values."
        )

    return vector


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "EEGFeature",
    "DEFAULT_FREQUENCY_BANDS",
    "extract_eeg_feature",
    "extract_eeg_features",
    "eeg_feature_vector",
]