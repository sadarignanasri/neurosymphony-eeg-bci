"""
Universal Phase 1 -> Phase 2 EEG bridge.

Phase 1 produces standardized EEGData.
Phase 2 receives EEGData, applies the dataset-specific
spatial configuration, performs universal preprocessing,
and returns EEGData again.

The preprocessing algorithm itself remains dataset-independent.
"""

from dataclasses import replace

import numpy as np
import mne

from src.acquisition.eeg_loader import EEGData
from src.preprocessing.montage import apply_eeg_montage
from src.preprocessing.pipeline import preprocess_eeg


def eeg_data_to_raw(
    eeg_data: EEGData,
    dataset_config=None,
) -> mne.io.BaseRaw:
    """
    Convert Phase 1 EEGData into an MNE Raw object.

    Dataset configuration is used only to restore the
    spatial/channel representation required by Phase 2.
    """

    if not isinstance(eeg_data, EEGData):
        raise TypeError(
            "Expected Phase 1 EEGData object."
        )

    data = np.asarray(eeg_data.data)

    if data.ndim != 2:
        raise ValueError(
            "EEGData.data must have shape "
            "(channels, samples)."
        )

    if data.shape[0] != len(eeg_data.channel_names):
        raise ValueError(
            "EEGData channel count does not "
            "match channel_names."
        )

    channel_types = tuple(eeg_data.channel_types)

    if (
        not channel_types
        or len(channel_types) != len(eeg_data.channel_names)
    ):
        channel_types = tuple(
            "eeg"
            for _ in eeg_data.channel_names
        )

    info = mne.create_info(
        ch_names=list(eeg_data.channel_names),
        sfreq=float(eeg_data.sampling_rate),
        ch_types=list(channel_types),
    )

    raw = mne.io.RawArray(
        data,
        info,
    )

    # ---------------------------------------------------------
    # Restore dataset-specific spatial information.
    #
    # This is the Phase-1 -> Phase-2 boundary. The universal
    # preprocessing algorithm remains inside preprocess_eeg().
    # ---------------------------------------------------------

    if dataset_config is not None:

        raw = apply_eeg_montage(
            raw,
            channel_mapping=dataset_config.get(
                "channel_mapping"
            ),
            montage_name=dataset_config.get(
                "montage"
            ),
            eeg_channel_names=dataset_config.get(
                "eeg_channel_names"
            ),
            auxiliary_channel_names=dataset_config.get(
                "auxiliary_channel_names"
            ),
            trigger_channel_names=dataset_config.get(
                "trigger_channel_names"
            ),
        )

    return raw


def raw_to_eeg_data(
    processed_raw: mne.io.BaseRaw,
    original: EEGData,
) -> EEGData:
    """
    Convert processed MNE Raw back into Phase 1 EEGData.
    """

    if not isinstance(
        processed_raw,
        mne.io.BaseRaw,
    ):
        raise TypeError(
            "processed_raw must be an MNE Raw object."
        )

    processed_data = np.asarray(
        processed_raw.get_data()
    )

    if not np.isfinite(processed_data).all():
        raise ValueError(
            "Processed EEGData contains NaN or infinite values."
        )

    metadata = dict(
        original.metadata or {}
    )

    metadata["phase2"] = {
        "status": "processed",
        "bandpass": "1-40 Hz",
        "notch": "50 Hz",
        "bad_channels": list(
            processed_raw.info["bads"]
        ),
        "interpolation": "spherical spline",
        "reference": "average",
    }

    return replace(
        original,
        data=processed_data,
        sampling_rate=float(
            processed_raw.info["sfreq"]
        ),
        channel_names=tuple(
            processed_raw.ch_names
        ),
        channel_types=tuple(
            processed_raw.get_channel_types()
        ),
        metadata=metadata,
    )


def preprocess_phase1_eeg(
    eeg_data: EEGData,
    dataset_config=None,
):
    """
    Main universal Phase 1 -> Phase 2 EEG handoff.
    """

    if not isinstance(
        eeg_data,
        EEGData,
    ):
        raise TypeError(
            "Phase 2 expects Phase 1 EEGData."
        )

    raw = eeg_data_to_raw(
        eeg_data,
        dataset_config=dataset_config,
    )

    processed_raw, quality_report = preprocess_eeg(
        raw,
        dataset_config=dataset_config,
    )

    processed_eeg_data = raw_to_eeg_data(
        processed_raw,
        eeg_data,
    )

    return (
        processed_eeg_data,
        quality_report,
    )
