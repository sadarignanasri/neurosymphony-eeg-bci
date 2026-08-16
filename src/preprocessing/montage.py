"""
Universal EEG channel normalization and montage handling.

This module is responsible only for dataset-specific spatial
preparation. The Phase 2 preprocessing algorithm remains universal.

Responsibilities:
    1. Identify configured EEG channels.
    2. Mark auxiliary channels as MISC.
    3. Mark trigger channels as STIM.
    4. Rename dataset-specific EEG channels to standard names.
    5. Apply the configured standard montage.
    6. Verify that configured EEG channels have valid positions.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import mne


def apply_eeg_montage(
    raw,
    channel_mapping: Optional[Dict[str, str]] = None,
    montage_name: Optional[str] = None,
    eeg_channel_names: Optional[Iterable[str]] = None,
    auxiliary_channel_names: Optional[Iterable[str]] = None,
    trigger_channel_names: Optional[Iterable[str]] = None,
):
    """
    Prepare an MNE Raw object for universal Phase 2 processing.

    Dataset-specific information is supplied through the arguments.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input EEG recording.

    channel_mapping : dict, optional
        Dataset channel name -> standard electrode name.

    montage_name : str, optional
        MNE standard montage name.

    eeg_channel_names : iterable, optional
        Original dataset EEG channel names.

    auxiliary_channel_names : iterable, optional
        Original auxiliary channel names.

    trigger_channel_names : iterable, optional
        Original trigger channel names.

    Returns
    -------
    mne.io.BaseRaw
        Prepared copy with normalized EEG names,
        channel types, and sensor positions.
    """

    # =========================================================
    # 1. INPUT VALIDATION
    # =========================================================

    if raw is None:
        raise ValueError(
            "EEG input cannot be None."
        )

    if not isinstance(
        raw,
        mne.io.BaseRaw,
    ):
        raise TypeError(
            "EEG input must be an MNE Raw object."
        )

    prepared = raw.copy()

    eeg_channel_names = tuple(
        eeg_channel_names or ()
    )

    auxiliary_channel_names = tuple(
        auxiliary_channel_names or ()
    )

    trigger_channel_names = tuple(
        trigger_channel_names or ()
    )

    channel_mapping = dict(
        channel_mapping or {}
    )

    # =========================================================
    # 2. VERIFY DATASET EEG CHANNELS
    # =========================================================

    if eeg_channel_names:

        missing_eeg = [
            channel
            for channel in eeg_channel_names
            if channel not in prepared.ch_names
        ]

        if missing_eeg:
            raise ValueError(
                "Configured EEG channels are missing "
                f"from the recording: {missing_eeg}"
            )

    # =========================================================
    # 3. RENAME EEG CHANNELS FIRST
    # =========================================================
    #
    # Important:
    # The standard montage expects standard electrode names.
    #
    # Therefore:
    #
    #     A28 -> Iz
    #     B21 -> TP8
    #     B27 -> P6
    #
    # must happen before spatial montage assignment.
    #
    # =========================================================

    if channel_mapping:

        valid_mapping = {
            old_name: new_name
            for old_name, new_name
            in channel_mapping.items()
            if old_name in prepared.ch_names
        }

        if valid_mapping:

            prepared.rename_channels(
                valid_mapping
            )

    # =========================================================
    # 4. DETERMINE FINAL EEG CHANNEL NAMES
    # =========================================================

    if eeg_channel_names:

        final_eeg_channels = []

        for original_name in eeg_channel_names:

            standard_name = channel_mapping.get(
                original_name,
                original_name,
            )

            if standard_name in prepared.ch_names:
                final_eeg_channels.append(
                    standard_name
                )

        if len(final_eeg_channels) != len(
            eeg_channel_names
        ):
            missing_final = [
                channel
                for channel in final_eeg_channels
                if channel not in prepared.ch_names
            ]

            raise ValueError(
                "Could not construct the complete "
                "normalized EEG channel set. "
                f"Missing: {missing_final}"
            )

    else:

        final_eeg_channels = []

    # =========================================================
    # 5. MARK ALL NON-EEG CHANNELS FIRST
    # =========================================================

    # Anything not explicitly configured as EEG,
    # auxiliary, or trigger is left untouched.
    #
    # Configured auxiliary channels are explicitly MISC.

    auxiliary_final_names = []

    for original_name in auxiliary_channel_names:

        if original_name in prepared.ch_names:

            auxiliary_final_names.append(
                original_name
            )

    if auxiliary_final_names:

        prepared.set_channel_types(
            {
                channel: "misc"
                for channel
                in auxiliary_final_names
            },
            on_unit_change="ignore",
        )

    # =========================================================
    # 6. MARK TRIGGER CHANNELS
    # =========================================================

    trigger_final_names = []

    for original_name in trigger_channel_names:

        if original_name in prepared.ch_names:

            trigger_final_names.append(
                original_name
            )

    if trigger_final_names:

        prepared.set_channel_types(
            {
                channel: "stim"
                for channel
                in trigger_final_names
            },
            on_unit_change="ignore",
        )

    # =========================================================
    # 7. FORCE FINAL EEG CHANNEL TYPES
    # =========================================================

    if final_eeg_channels:

        prepared.set_channel_types(
            {
                channel: "eeg"
                for channel
                in final_eeg_channels
            },
            on_unit_change="ignore",
        )

    # =========================================================
    # 8. APPLY STANDARD MONTAGE
    # =========================================================

    if montage_name:

        try:

            montage = (
                mne.channels.make_standard_montage(
                    montage_name
                )
            )

        except Exception as exc:

            raise ValueError(
                "Unable to create requested EEG "
                f"montage '{montage_name}': {exc}"
            ) from exc

        # Only EEG channels participate in the montage.
        #
        # Auxiliary and trigger channels remain MISC/STIM.

        prepared.set_montage(
            montage,
            match_case=False,
            on_missing="ignore",
        )

    # =========================================================
    # 9. VERIFY EEG POSITIONS
    # =========================================================

    if final_eeg_channels:

        missing_positions = []

        for channel in final_eeg_channels:

            index = prepared.ch_names.index(
                channel
            )

            position = (
                prepared.info["chs"][index]["loc"][:3]
            )

            if (
                position is None
                or len(position) != 3
                or any(
                    value != value
                    for value in position
                )
                or all(
                    value == 0
                    for value in position
                )
            ):

                missing_positions.append(
                    channel
                )

        if missing_positions:

            raise RuntimeError(
                "EEG montage assignment failed. "
                "The following EEG channels do not "
                "have valid sensor positions: "
                f"{missing_positions}"
            )

    # =========================================================
    # 10. VERIFY FINAL EEG COUNT
    # =========================================================

    if final_eeg_channels:

        actual_eeg = [
            channel
            for channel, channel_type
            in zip(
                prepared.ch_names,
                prepared.get_channel_types(),
            )
            if channel_type == "eeg"
        ]

        expected_set = set(
            final_eeg_channels
        )

        actual_set = set(
            actual_eeg
        )

        missing_eeg = sorted(
            expected_set - actual_set
        )

        unexpected_eeg = sorted(
            actual_set - expected_set
        )

        if missing_eeg or unexpected_eeg:

            raise RuntimeError(
                "Final EEG channel classification "
                "does not match the dataset configuration. "
                f"Missing EEG: {missing_eeg}; "
                f"Unexpected EEG: {unexpected_eeg}"
            )

    # =========================================================
    # 11. FINAL SANITY CHECK
    # =========================================================

    if final_eeg_channels:

        if prepared.info["dig"] is None:

            raise RuntimeError(
                "EEG montage was not attached. "
                "MNE info['dig'] is None after "
                "montage assignment."
            )

    return prepared


__all__ = [
    "apply_eeg_montage",
]