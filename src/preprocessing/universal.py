
"""
Neuro-Symphony Universal Phase 2 Processing.

Routes standardized Phase 1 EEGData and FNIRSData objects
to their corresponding modality-specific Phase 2 processors.

No modality-specific preprocessing is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.acquisition.eeg_loader import EEGData
from src.acquisition.fnirs_loader import FNIRSData

from src.preprocessing.phase1_bridge import preprocess_phase1_eeg
from src.preprocessing.fnirs_bridge import preprocess_phase1_fnirs


@dataclass(frozen=True)
class UniversalPhase2Result:
    """Standardized result returned by Universal Phase 2."""

    modality: str
    processed_data: Any
    quality_report: Any


def preprocess_phase2(
    data,
    dataset_config=None,
) -> UniversalPhase2Result:
    """
    Route Phase 1 standardized data to the correct Phase 2 cleaner.

    Parameters
    ----------
    data:
        Either EEGData or FNIRSData produced by Phase 1.

    dataset_config:
        Optional dataset-specific configuration used by EEG
        preprocessing.

    Returns
    -------
    UniversalPhase2Result
        Cleaned modality data and its quality report.
    """

    if isinstance(data, EEGData):

        processed, quality_report = preprocess_phase1_eeg(
            data,
            dataset_config=dataset_config,
        )

        return UniversalPhase2Result(
            modality="EEG",
            processed_data=processed,
            quality_report=quality_report,
        )

    if isinstance(data, FNIRSData):

        result = preprocess_phase1_fnirs(
            data
        )

        return UniversalPhase2Result(
            modality="fNIRS",
            processed_data=result.data,
            quality_report=result.quality_report,
        )

    raise TypeError(
        "Universal Phase 2 expects either EEGData "
        "or FNIRSData."
    )


__all__ = [
    "UniversalPhase2Result",
    "preprocess_phase2",
]