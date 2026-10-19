"""
Universal Phase 3 EEG-fNIRS Epoch Extraction.

This module extracts task-locked epochs from Phase 2 EEGData
and FNIRSData objects.

Dataset-specific information such as task names and event files
is supplied externally. No dataset ID is hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from src.acquisition.eeg_loader import EEGData
from src.acquisition.fnirs_loader import FNIRSData
from src.synchronization.timestamp_alignment import TimestampAlignment


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EpochConfig:
    """Universal epoch extraction configuration."""

    tmin: float = 0.0
    tmax: float = 3.0
    reject_nonfinite: bool = True

    def __post_init__(self):
        if not np.isfinite(self.tmin):
            raise ValueError("tmin must be finite.")

        if not np.isfinite(self.tmax):
            raise ValueError("tmax must be finite.")

        if self.tmax <= self.tmin:
            raise ValueError(
                "tmax must be greater than tmin."
            )


# ============================================================
# EVENT
# ============================================================

@dataclass(frozen=True)
class EpochEvent:
    """One task event."""

    onset: float
    trial_type: str
    value: Optional[int | float | str] = None
    subject: Optional[str] = None

    def __post_init__(self):
        if not np.isfinite(self.onset):
            raise ValueError(
                "Event onset must be finite."
            )

        if not self.trial_type:
            raise ValueError(
                "trial_type cannot be empty."
            )


# ============================================================
# EPOCH
# ============================================================

@dataclass(frozen=True)
class Epoch:
    """One modality-specific epoch."""

    subject: Optional[str]
    trial_type: str

    event_onset: float
    start_time: float
    end_time: float

    data: np.ndarray

    sampling_rate: float
    channel_names: tuple[str, ...]

    modality: str

    event_value: Optional[int | float | str] = None


@dataclass(frozen=True)
class MultimodalEpoch:
    """One synchronized EEG + fNIRS epoch."""

    subject: Optional[str]
    trial_type: str

    eeg_epoch: Epoch
    fnirs_epoch: Epoch

    event_onset_eeg: float
    event_onset_fnirs: float


# ============================================================
# EVENT NORMALIZATION
# ============================================================

def normalize_events(
    events: Iterable[EpochEvent | dict],
) -> tuple[EpochEvent, ...]:

    result = []

    for event in events:

        if isinstance(event, EpochEvent):
            result.append(event)
            continue

        if not isinstance(event, dict):
            raise TypeError(
                "Events must be EpochEvent objects or dictionaries."
            )

        result.append(
            EpochEvent(
                onset=float(event["onset"]),
                trial_type=str(event["trial_type"]),
                value=event.get("value"),
                subject=event.get("subject"),
            )
        )

    return tuple(result)


# ============================================================
# VALIDATION
# ============================================================

def _validate_eeg(eeg: EEGData) -> None:

    if not isinstance(eeg, EEGData):
        raise TypeError("Expected EEGData.")

    if eeg.data.ndim != 2:
        raise ValueError(
            "EEGData.data must have shape (channels, samples)."
        )

    if eeg.sampling_rate <= 0:
        raise ValueError(
            "EEG sampling rate must be positive."
        )

    if eeg.data.shape[0] != len(eeg.channel_names):
        raise ValueError(
            "EEG channel count does not match channel_names."
        )


def _validate_fnirs(fnirs: FNIRSData) -> None:

    if not isinstance(fnirs, FNIRSData):
        raise TypeError("Expected FNIRSData.")

    if fnirs.data.ndim != 2:
        raise ValueError(
            "FNIRSData.data must have shape (channels, samples)."
        )

    if fnirs.sampling_rate <= 0:
        raise ValueError(
            "fNIRS sampling rate must be positive."
        )

    if fnirs.data.shape[0] != len(fnirs.channel_names):
        raise ValueError(
            "fNIRS channel count does not match channel_names."
        )


# ============================================================
# LOW-LEVEL EXTRACTION
# ============================================================

def _extract_epoch(
    data: np.ndarray,
    sampling_rate: float,
    event_onset: float,
    config: EpochConfig,
) -> tuple[np.ndarray, float, float]:

    
    start_time = event_onset + config.tmin
    end_time = event_onset + config.tmax

    # ------------------------------------------------------------
    # Use a fixed number of samples for every epoch.
    #
    # This is important for lower-frequency modalities such as
    # fNIRS, where epoch_duration * sampling_rate may not be an
    # integer.
    # ------------------------------------------------------------

    epoch_duration = config.tmax - config.tmin

    n_samples = int(
        np.round(epoch_duration * sampling_rate)
    )

    start_sample = int(
        np.round(start_time * sampling_rate)
    )

    end_sample = start_sample + n_samples

    if start_sample < 0:
        raise ValueError(
            "Epoch starts before the recording."
        )

    if end_sample > data.shape[1]:
        raise ValueError(
            "Epoch extends beyond the recording."
        )

    if end_sample <= start_sample:
        raise ValueError(
            "Epoch contains no samples."
        )

    epoch = np.asarray(
        data[:, start_sample:end_sample],
        dtype=np.float64,
    )

    if config.reject_nonfinite:
        if not np.isfinite(epoch).all():
            raise ValueError(
                "Epoch contains NaN or infinite values."
            )

    return epoch, start_time, end_time


# ============================================================
# EEG EPOCHS
# ============================================================

def extract_eeg_epochs(
    eeg_data: EEGData,
    events: Sequence[EpochEvent | dict],
    config: EpochConfig | None = None,
) -> tuple[Epoch, ...]:

    _validate_eeg(eeg_data)

    if config is None:
        config = EpochConfig()

    events = normalize_events(events)

    epochs = []

    for event in events:

        try:
            data, start, end = _extract_epoch(
                eeg_data.data,
                float(eeg_data.sampling_rate),
                event.onset,
                config,
            )
        except ValueError:
            continue

        epochs.append(
            Epoch(
                subject=(
                    event.subject
                    if event.subject is not None
                    else eeg_data.subject_id
                ),
                trial_type=event.trial_type,
                event_onset=event.onset,
                start_time=start,
                end_time=end,
                data=data,
                sampling_rate=float(
                    eeg_data.sampling_rate
                ),
                channel_names=tuple(
                    eeg_data.channel_names
                ),
                modality="EEG",
                event_value=event.value,
            )
        )

    return tuple(epochs)


# ============================================================
# fNIRS EPOCHS
# ============================================================

def extract_fnirs_epochs(
    fnirs_data: FNIRSData,
    events: Sequence[EpochEvent | dict],
    config: EpochConfig | None = None,
) -> tuple[Epoch, ...]:

    _validate_fnirs(fnirs_data)

    if config is None:
        config = EpochConfig()

    events = normalize_events(events)

    epochs = []

    for event in events:

        try:
            data, start, end = _extract_epoch(
                fnirs_data.data,
                float(fnirs_data.sampling_rate),
                event.onset,
                config,
            )
        except ValueError:
            continue

        epochs.append(
            Epoch(
                subject=(
                    event.subject
                    if event.subject is not None
                    else fnirs_data.subject_id
                ),
                trial_type=event.trial_type,
                event_onset=event.onset,
                start_time=start,
                end_time=end,
                data=data,
                sampling_rate=float(
                    fnirs_data.sampling_rate
                ),
                channel_names=tuple(
                    fnirs_data.channel_names
                ),
                modality="fNIRS",
                event_value=event.value,
            )
        )

    return tuple(epochs)


# ============================================================
# MULTIMODAL EPOCHS
# ============================================================

def extract_multimodal_epochs(
    eeg_data: EEGData,
    fnirs_data: FNIRSData,
    events: Sequence[EpochEvent | dict],
    alignment: TimestampAlignment,
    config: EpochConfig | None = None,
) -> tuple[MultimodalEpoch, ...]:

    _validate_eeg(eeg_data)
    _validate_fnirs(fnirs_data)

    if config is None:
        config = EpochConfig()

    events = normalize_events(events)

    epochs = []

    for event in events:

        eeg_onset = float(event.onset)

        fnirs_onset = float(
            alignment.eeg_to_fnirs(
                np.asarray([eeg_onset])
            )[0]
        )

        try:

            eeg_data_epoch, eeg_start, eeg_end = (
                _extract_epoch(
                    eeg_data.data,
                    float(eeg_data.sampling_rate),
                    eeg_onset,
                    config,
                )
            )

            fnirs_data_epoch, fnirs_start, fnirs_end = (
                _extract_epoch(
                    fnirs_data.data,
                    float(fnirs_data.sampling_rate),
                    fnirs_onset,
                    config,
                )
            )

        except ValueError:
            continue

        subject = (
            event.subject
            if event.subject is not None
            else eeg_data.subject_id
        )

        eeg_epoch = Epoch(
            subject=subject,
            trial_type=event.trial_type,
            event_onset=eeg_onset,
            start_time=eeg_start,
            end_time=eeg_end,
            data=eeg_data_epoch,
            sampling_rate=float(
                eeg_data.sampling_rate
            ),
            channel_names=tuple(
                eeg_data.channel_names
            ),
            modality="EEG",
            event_value=event.value,
        )

        fnirs_epoch = Epoch(
            subject=subject,
            trial_type=event.trial_type,
            event_onset=fnirs_onset,
            start_time=fnirs_start,
            end_time=fnirs_end,
            data=fnirs_data_epoch,
            sampling_rate=float(
                fnirs_data.sampling_rate
            ),
            channel_names=tuple(
                fnirs_data.channel_names
            ),
            modality="fNIRS",
            event_value=event.value,
        )

        epochs.append(
            MultimodalEpoch(
                subject=subject,
                trial_type=event.trial_type,
                eeg_epoch=eeg_epoch,
                fnirs_epoch=fnirs_epoch,
                event_onset_eeg=eeg_onset,
                event_onset_fnirs=fnirs_onset,
            )
        )

    return tuple(epochs)


__all__ = [
    "EpochConfig",
    "EpochEvent",
    "Epoch",
    "MultimodalEpoch",
    "normalize_events",
    "extract_eeg_epochs",
    "extract_fnirs_epochs",
    "extract_multimodal_epochs",
]