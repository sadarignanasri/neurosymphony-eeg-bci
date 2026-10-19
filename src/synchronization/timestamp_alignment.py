"""
EEG–fNIRS timestamp alignment.

Phase 3 component for Neuro-Symphony.

This module provides reusable utilities for aligning EEG and fNIRS
timestamps using an experimentally determined temporal offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TimestampAlignment:
    """
    Represents the temporal relationship between two modalities.

    Parameters
    ----------
    offset_seconds:
        Amount of time that must be added to an EEG timestamp to
        express it in the fNIRS time coordinate system.
    """

    offset_seconds: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.offset_seconds):
            raise ValueError(
                "offset_seconds must be a finite number."
            )

    def eeg_to_fnirs(
        self,
        eeg_timestamps: Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Convert EEG timestamps into the fNIRS time coordinate system.
        """
        timestamps = np.asarray(
            eeg_timestamps,
            dtype=np.float64,
        )

        return timestamps + self.offset_seconds

    def fnirs_to_eeg(
        self,
        fnirs_timestamps: Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Convert fNIRS timestamps into the EEG time coordinate system.
        """
        timestamps = np.asarray(
            fnirs_timestamps,
            dtype=np.float64,
        )

        return timestamps - self.offset_seconds


def estimate_offset(
    eeg_timestamps: Iterable[float] | np.ndarray,
    fnirs_timestamps: Iterable[float] | np.ndarray,
) -> TimestampAlignment:
    """
    Estimate the temporal offset between corresponding EEG and
    fNIRS timestamps.

    The two arrays must contain corresponding events in the same
    order.

    Offset is calculated as:

        fNIRS timestamp - EEG timestamp

    The median is used because it is robust to occasional outliers.
    """

    eeg = np.asarray(
        eeg_timestamps,
        dtype=np.float64,
    )

    fnirs = np.asarray(
        fnirs_timestamps,
        dtype=np.float64,
    )

    if eeg.ndim != 1 or fnirs.ndim != 1:
        raise ValueError(
            "EEG and fNIRS timestamps must be one-dimensional."
        )

    if len(eeg) == 0 or len(fnirs) == 0:
        raise ValueError(
            "EEG and fNIRS timestamp arrays cannot be empty."
        )

    if len(eeg) != len(fnirs):
        raise ValueError(
            "EEG and fNIRS timestamp arrays must have "
            "the same length."
        )

    if not np.all(np.isfinite(eeg)):
        raise ValueError(
            "EEG timestamps contain non-finite values."
        )

    if not np.all(np.isfinite(fnirs)):
        raise ValueError(
            "fNIRS timestamps contain non-finite values."
        )

    offsets = fnirs - eeg

    return TimestampAlignment(
        offset_seconds=float(np.median(offsets))
    )


def calculate_alignment_error(
    eeg_timestamps: Iterable[float] | np.ndarray,
    fnirs_timestamps: Iterable[float] | np.ndarray,
    alignment: TimestampAlignment,
) -> dict[str, float]:
    """
    Calculate synchronization error after applying an alignment.

    Returns mean, median, standard deviation, minimum and maximum
    absolute timing errors in seconds.
    """

    eeg = np.asarray(
        eeg_timestamps,
        dtype=np.float64,
    )

    fnirs = np.asarray(
        fnirs_timestamps,
        dtype=np.float64,
    )

    aligned_eeg = alignment.eeg_to_fnirs(eeg)

    errors = fnirs - aligned_eeg
    absolute_errors = np.abs(errors)

    return {
        "mean_error_seconds": float(
            np.mean(absolute_errors)
        ),
        "median_error_seconds": float(
            np.median(absolute_errors)
        ),
        "std_error_seconds": float(
            np.std(errors)
        ),
        "max_error_seconds": float(
            np.max(absolute_errors)
        ),
    }
