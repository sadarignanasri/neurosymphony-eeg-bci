"""Universal Phase 3 epoch extraction."""

from .epoch_extractor import (
    Epoch,
    EpochConfig,
    EpochEvent,
    MultimodalEpoch,
    extract_eeg_epochs,
    extract_fnirs_epochs,
    extract_multimodal_epochs,
    normalize_events,
)

__all__ = [
    "Epoch",
    "EpochConfig",
    "EpochEvent",
    "MultimodalEpoch",
    "extract_eeg_epochs",
    "extract_fnirs_epochs",
    "extract_multimodal_epochs",
    "normalize_events",
]