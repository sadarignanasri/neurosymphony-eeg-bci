"""
Neuro-Symphony Phase 2 — Universal fNIRS Processing.

This module consumes the standardized FNIRSData object produced by
Phase 1 and performs deterministic signal validation, filtering,
artifact suppression, normalization, and quality assessment.

Phase 1 acquisition is not modified.
EEG preprocessing is not modified.
Phase 3 feature extraction is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.acquisition.fnirs_loader import FNIRSData


# =====================================================================
# QUALITY REPORT
# =====================================================================

@dataclass(frozen=True)
class FNIRSQualityReport:
    """Quality information generated during Phase 2 fNIRS processing."""

    total_channels: int
    total_samples: int
    sampling_rate: float
    duration_seconds: float

    nan_count_before: int
    inf_count_before: int

    nan_count_after: int
    inf_count_after: int

    finite_before: bool
    finite_after: bool

    shape_preserved: bool
    channel_count_preserved: bool
    channel_names_preserved: bool
    channel_types_preserved: bool
    metadata_preserved: bool

    processing_status: str

    # -------------------------------------------------------------
    # Phase 2 signal-processing metrics
    # -------------------------------------------------------------

    filtering_applied: bool = False
    filter_low_cutoff_hz: float = 0.01
    filter_high_cutoff_hz: float = 0.5

    artifact_detection_applied: bool = False
    artifact_samples_detected: int = 0
    artifact_percentage: float = 0.0

    normalization_applied: bool = False

    mean_before: float = 0.0
    std_before: float = 0.0
    mean_after: float = 0.0
    std_after: float = 0.0


# =====================================================================
# RESULT
# =====================================================================

@dataclass(frozen=True)
class FNIRSPreprocessingResult:
    """Universal Phase 2 fNIRS output."""

    data: FNIRSData
    quality_report: FNIRSQualityReport


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_LOW_CUTOFF_HZ = 0.01
DEFAULT_HIGH_CUTOFF_HZ = 0.5

FILTER_ORDER = 4

# Robust derivative threshold.
#
# A point is considered a potential motion artifact when the absolute
# first difference exceeds:
#
# median(|derivative|) + threshold * MAD
#
# This is intentionally conservative and deterministic.
ARTIFACT_THRESHOLD = 8.0

# Small epsilon prevents division by zero during normalization.
EPSILON = 1e-12


# =====================================================================
# VALIDATION
# =====================================================================

def _validate_input(fnirs: FNIRSData) -> None:
    """Validate the standardized Phase 1 fNIRS object."""

    if not isinstance(fnirs, FNIRSData):
        raise TypeError(
            "Phase 2 fNIRS preprocessing requires an FNIRSData object."
        )

    if not isinstance(fnirs.data, np.ndarray):
        raise TypeError(
            "FNIRSData.data must be a NumPy array."
        )

    if fnirs.data.ndim != 2:
        raise ValueError(
            "FNIRSData.data must have shape (channels, samples)."
        )

    if fnirs.data.shape[0] == 0:
        raise ValueError(
            "fNIRS recording contains no channels."
        )

    if fnirs.data.shape[1] == 0:
        raise ValueError(
            "fNIRS recording contains no samples."
        )

    if not np.isfinite(fnirs.sampling_rate):
        raise ValueError(
            "fNIRS sampling rate must be finite."
        )

    if fnirs.sampling_rate <= 0:
        raise ValueError(
            "fNIRS sampling rate must be positive."
        )


def _count_nonfinite(data: np.ndarray) -> tuple[int, int]:
    """Return NaN and Inf counts."""

    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    return nan_count, inf_count


def _validate_signal(
    data: np.ndarray,
    stage: str,
) -> None:
    """Ensure the signal contains only finite values."""

    nan_count, inf_count = _count_nonfinite(data)

    if nan_count > 0:
        raise ValueError(
            f"fNIRS signal contains {nan_count} NaN values at {stage}."
        )

    if inf_count > 0:
        raise ValueError(
            f"fNIRS signal contains {inf_count} infinite values at {stage}."
        )


# =====================================================================
# FILTERING
# =====================================================================

def _resolve_filter_cutoffs(
    sampling_rate: float,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
) -> tuple[float, float]:
    """
    Resolve safe band-pass cutoffs for the current sampling rate.

    The requested fNIRS baseline is 0.01–0.5 Hz.

    The Nyquist frequency is checked explicitly so that the filter
    remains valid for datasets with different sampling rates.
    """

    nyquist = sampling_rate / 2.0

    low = float(low_cutoff_hz)
    high = float(high_cutoff_hz)

    if low <= 0:
        raise ValueError(
            "fNIRS low cutoff must be positive."
        )

    if high <= low:
        raise ValueError(
            "fNIRS high cutoff must be greater than low cutoff."
        )

    if high >= nyquist:
        # Leave a safety margin below Nyquist.
        high = nyquist * 0.8

    if high <= low:
        raise ValueError(
            f"Sampling rate {sampling_rate} Hz is too low for the "
            f"requested fNIRS band {low_cutoff_hz}-{high_cutoff_hz} Hz."
        )

    return low, high


def _bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_cutoff_hz: float = DEFAULT_LOW_CUTOFF_HZ,
    high_cutoff_hz: float = DEFAULT_HIGH_CUTOFF_HZ,
) -> tuple[np.ndarray, float, float]:
    """
    Apply deterministic zero-phase Butterworth band-pass filtering.

    Input shape:
        (channels, samples)

    Output shape:
        (channels, samples)
    """

    low, high = _resolve_filter_cutoffs(
        sampling_rate=sampling_rate,
        low_cutoff_hz=low_cutoff_hz,
        high_cutoff_hz=high_cutoff_hz,
    )

    nyquist = sampling_rate / 2.0

    normalized_low = low / nyquist
    normalized_high = high / nyquist

    sos = butter(
        FILTER_ORDER,
        [normalized_low, normalized_high],
        btype="bandpass",
        output="sos",
    )

    filtered = np.empty_like(data, dtype=np.float64)

    for channel_index in range(data.shape[0]):
        channel = data[channel_index]

        # sosfiltfilt requires enough samples for stable edge handling.
        # For very short development segments, fall back to a causal
        # filtering pass rather than failing the complete Phase 2 path.
        try:
            filtered[channel_index] = sosfiltfilt(
                sos,
                channel,
            )
        except ValueError:
            from scipy.signal import sosfilt

            filtered[channel_index] = sosfilt(
                sos,
                channel,
            )

    return filtered, low, high


# =====================================================================
# ARTIFACT DETECTION / SUPPRESSION
# =====================================================================

def _robust_artifact_mask(
    data: np.ndarray,
    threshold: float = ARTIFACT_THRESHOLD,
) -> np.ndarray:
    """
    Detect abrupt temporal changes using a robust derivative criterion.

    Detection is performed independently for every channel.

    Returns:
        Boolean array with shape (channels, samples).
    """

    channels, samples = data.shape

    mask = np.zeros(
        (channels, samples),
        dtype=bool,
    )

    if samples < 3:
        return mask

    derivative = np.diff(
        data,
        axis=1,
    )

    absolute_derivative = np.abs(derivative)

    median = np.median(
        absolute_derivative,
        axis=1,
        keepdims=True,
    )

    mad = np.median(
        np.abs(
            absolute_derivative - median
        ),
        axis=1,
        keepdims=True,
    )

    # Convert MAD to a robust scale estimate.
    robust_scale = 1.4826 * mad

    threshold_value = (
        median
        + threshold * robust_scale
        + EPSILON
    )

    derivative_mask = (
        absolute_derivative > threshold_value
    )

    # A derivative corresponds to a transition between samples.
    # Mark both sides of that transition.
    mask[:, 1:] |= derivative_mask
    mask[:, :-1] |= derivative_mask

    return mask


def _interpolate_artifacts(
    data: np.ndarray,
    artifact_mask: np.ndarray,
) -> np.ndarray:
    """
    Replace detected artifact samples using linear interpolation.

    Edge artifacts are replaced with the nearest valid sample.

    If an entire channel is marked as artifact, the channel is left
    unchanged rather than inventing a complete signal.
    """

    repaired = data.copy()

    channels, samples = repaired.shape

    for channel_index in range(channels):

        channel_mask = artifact_mask[channel_index]

        if not np.any(channel_mask):
            continue

        valid_indices = np.flatnonzero(
            ~channel_mask
        )

        if valid_indices.size == 0:
            continue

        artifact_indices = np.flatnonzero(
            channel_mask
        )

        if valid_indices.size == 1:
            repaired[channel_index, artifact_indices] = (
                repaired[channel_index, valid_indices[0]]
            )
            continue

        repaired[channel_index, artifact_indices] = np.interp(
            artifact_indices,
            valid_indices,
            repaired[
                channel_index,
                valid_indices,
            ],
        )

    return repaired


# =====================================================================
# NORMALIZATION
# =====================================================================

def _normalize_channels(
    data: np.ndarray,
) -> np.ndarray:
    """
    Standardize each fNIRS channel independently.

    For every channel:

        normalized = (x - mean) / std

    Constant channels are centered but not divided by zero.
    """

    means = np.mean(
        data,
        axis=1,
        keepdims=True,
    )

    stds = np.std(
        data,
        axis=1,
        keepdims=True,
    )

    safe_stds = np.where(
        stds > EPSILON,
        stds,
        1.0,
    )

    normalized = (
        data - means
    ) / safe_stds

    return normalized


# =====================================================================
# METADATA PRESERVATION
# =====================================================================

def _preserve_metadata(
    original: FNIRSData,
    processed_data: np.ndarray,
) -> FNIRSData:
    """
    Construct a new FNIRSData object while preserving Phase 1 metadata.
    """

    return FNIRSData(
        recording_id=original.recording_id,
        recording_path=original.recording_path,
        modality=original.modality,
        layout=original.layout,
        subject_id=original.subject_id,
        session_id=original.session_id,
        task_id=original.task_id,
        run_id=original.run_id,
        source_kind=original.source_kind,
        adapter_name=original.adapter_name,
        sampling_rate=original.sampling_rate,
        channel_names=original.channel_names,
        data=processed_data,
        channel_types=original.channel_types,
        fnirs_metadata=original.fnirs_metadata,
        metadata=dict(original.metadata),
    )


# =====================================================================
# MAIN PHASE 2 PROCESSOR
# =====================================================================

def preprocess_fnirs(
    fnirs: FNIRSData,
    *,
    low_cutoff_hz: float = DEFAULT_LOW_CUTOFF_HZ,
    high_cutoff_hz: float = DEFAULT_HIGH_CUTOFF_HZ,
    artifact_threshold: float = ARTIFACT_THRESHOLD,
) -> FNIRSPreprocessingResult:
    """
    Run the Phase 2 fNIRS preprocessing pipeline.

    Pipeline:

    1. Input validation.
    2. Immutable numerical copy.
    3. Finite-value validation.
    4. Band-pass filtering.
    5. Robust temporal artifact detection.
    6. Artifact interpolation.
    7. Channel-wise normalization.
    8. Output validation.
    9. Metadata preservation.
    10. Quality-report generation.

    The Phase 1 FNIRSData object is never modified in-place.
    """

    # -------------------------------------------------------------
    # 1. Validate Phase 1 output
    # -------------------------------------------------------------

    _validate_input(fnirs)

    # -------------------------------------------------------------
    # 2. Immutable numerical copy
    # -------------------------------------------------------------

    data_before = np.asarray(
        fnirs.data,
        dtype=np.float64,
    ).copy()

    original_snapshot = data_before.copy()

    # -------------------------------------------------------------
    # 3. Validate input signal
    # -------------------------------------------------------------

    nan_before, inf_before = _count_nonfinite(
        data_before
    )

    _validate_signal(
        data_before,
        stage="input",
    )

    mean_before = float(
        np.mean(data_before)
    )

    std_before = float(
        np.std(data_before)
    )

    # -------------------------------------------------------------
    # 4. Actual fNIRS band-pass filtering
    # -------------------------------------------------------------

    filtered_data, actual_low, actual_high = (
        _bandpass_filter(
            data=data_before,
            sampling_rate=float(fnirs.sampling_rate),
            low_cutoff_hz=low_cutoff_hz,
            high_cutoff_hz=high_cutoff_hz,
        )
    )

    _validate_signal(
        filtered_data,
        stage="after filtering",
    )

    # -------------------------------------------------------------
    # 5. Motion/artifact detection
    # -------------------------------------------------------------

    artifact_mask = _robust_artifact_mask(
        filtered_data,
        threshold=artifact_threshold,
    )

    artifact_samples_detected = int(
        np.sum(artifact_mask)
    )

    total_values = int(
        artifact_mask.size
    )

    artifact_percentage = (
        100.0 * artifact_samples_detected / total_values
        if total_values > 0
        else 0.0
    )

    # -------------------------------------------------------------
    # 6. Artifact suppression
    # -------------------------------------------------------------

    artifact_corrected = _interpolate_artifacts(
        filtered_data,
        artifact_mask,
    )

    _validate_signal(
        artifact_corrected,
        stage="after artifact correction",
    )

    # -------------------------------------------------------------
    # 7. Channel-wise normalization
    # -------------------------------------------------------------

    processed_data = _normalize_channels(
        artifact_corrected
    )

    _validate_signal(
        processed_data,
        stage="after normalization",
    )

    # -------------------------------------------------------------
    # 8. Final immutability check
    # -------------------------------------------------------------

    if not np.array_equal(
        data_before,
        original_snapshot,
    ):
        raise RuntimeError(
            "Phase 2 unexpectedly modified its internal input copy."
        )

    # -------------------------------------------------------------
    # 9. Construct processed FNIRSData
    # -------------------------------------------------------------

    processed = _preserve_metadata(
        original=fnirs,
        processed_data=processed_data,
    )

    # -------------------------------------------------------------
    # 10. Validate structural preservation
    # -------------------------------------------------------------

    shape_preserved = (
        processed.data.shape == fnirs.data.shape
    )

    channel_count_preserved = (
        processed.n_channels == fnirs.n_channels
    )

    channel_names_preserved = (
        processed.channel_names == fnirs.channel_names
    )

    channel_types_preserved = (
        processed.channel_types == fnirs.channel_types
    )

    metadata_preserved = (
        processed.metadata == fnirs.metadata
        and processed.fnirs_metadata == fnirs.fnirs_metadata
    )

    if not shape_preserved:
        raise RuntimeError(
            "Phase 2 changed the fNIRS signal shape."
        )

    if not channel_count_preserved:
        raise RuntimeError(
            "Phase 2 changed the fNIRS channel count."
        )

    if not channel_names_preserved:
        raise RuntimeError(
            "Phase 2 changed fNIRS channel names."
        )

    if not channel_types_preserved:
        raise RuntimeError(
            "Phase 2 changed fNIRS channel types."
        )

    if not metadata_preserved:
        raise RuntimeError(
            "Phase 2 failed to preserve fNIRS metadata."
        )

    # -------------------------------------------------------------
    # 11. Final quality metrics
    # -------------------------------------------------------------

    nan_after, inf_after = _count_nonfinite(
        processed_data
    )

    finite_before = bool(
        np.isfinite(data_before).all()
    )

    finite_after = bool(
        np.isfinite(processed_data).all()
    )

    mean_after = float(
        np.mean(processed_data)
    )

    std_after = float(
        np.std(processed_data)
    )

    processing_status = (
        "PASS"
        if finite_after
        and shape_preserved
        and channel_count_preserved
        and channel_names_preserved
        and channel_types_preserved
        and metadata_preserved
        else "FAIL"
    )

    if processing_status != "PASS":
        raise RuntimeError(
            "Phase 2 fNIRS quality validation failed."
        )

    quality_report = FNIRSQualityReport(
        total_channels=processed.n_channels,
        total_samples=processed.n_samples,
        sampling_rate=float(
            processed.sampling_rate
        ),
        duration_seconds=float(
            processed.duration_seconds
        ),

        nan_count_before=nan_before,
        inf_count_before=inf_before,

        nan_count_after=nan_after,
        inf_count_after=inf_after,

        finite_before=finite_before,
        finite_after=finite_after,

        shape_preserved=shape_preserved,
        channel_count_preserved=channel_count_preserved,
        channel_names_preserved=channel_names_preserved,
        channel_types_preserved=channel_types_preserved,
        metadata_preserved=metadata_preserved,

        processing_status=processing_status,

        filtering_applied=True,
        filter_low_cutoff_hz=float(actual_low),
        filter_high_cutoff_hz=float(actual_high),

        artifact_detection_applied=True,
        artifact_samples_detected=artifact_samples_detected,
        artifact_percentage=float(
            artifact_percentage
        ),

        normalization_applied=True,

        mean_before=mean_before,
        std_before=std_before,

        mean_after=mean_after,
        std_after=std_after,
    )

    return FNIRSPreprocessingResult(
        data=processed,
        quality_report=quality_report,
    )


__all__ = [
    "FNIRSQualityReport",
    "FNIRSPreprocessingResult",
    "preprocess_fnirs",
]
