
"""
Neuro-Symphony Phase 1 -> Phase 2 fNIRS bridge.

Converts the standardized Phase 1 FNIRSData object into
the Phase 2 fNIRS processing pipeline.

Phase 1 acquisition is not modified.
"""

from __future__ import annotations

from src.acquisition.fnirs_loader import FNIRSData
from src.preprocessing.fnirs import (
    FNIRSPreprocessingResult,
    preprocess_fnirs,
)


def preprocess_phase1_fnirs(
    fnirs_data: FNIRSData,
) -> FNIRSPreprocessingResult:
    """
    Main universal Phase 1 -> Phase 2 fNIRS handoff.

    Parameters
    ----------
    fnirs_data:
        Standardized FNIRSData returned by Phase 1.

    Returns
    -------
    FNIRSPreprocessingResult
        Processed fNIRS data and Phase 2 quality report.
    """

    if not isinstance(fnirs_data, FNIRSData):
        raise TypeError(
            "Phase 2 expects Phase 1 FNIRSData."
        )

    return preprocess_fnirs(
        fnirs_data
    )


__all__ = [
    "preprocess_phase1_fnirs",
]