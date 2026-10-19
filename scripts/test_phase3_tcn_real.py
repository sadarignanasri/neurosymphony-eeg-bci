"""
Neuro-Symphony Phase 3 — Real TCN Feature Validation.

Validates the universal Temporal Convolutional Network on
real EEG and fNIRS task-locked epochs.

This test verifies:

    1. Real EEG loading
    2. Real fNIRS loading
    3. Real task-event loading
    4. Epoch extraction
    5. EEG TCN representation
    6. fNIRS TCN representation
    7. Representation dimensions
    8. Finite-value integrity
    9. Task-label preservation
    10. EEG/fNIRS representation pairing

No dataset-specific logic is implemented inside the TCN.
ds004514 is used only as a real validation dataset.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.acquisition.eeg_loader import EEGData
from src.acquisition.fnirs_loader import FNIRSData
from src.epochs.epoch_extractor import (
    EpochEvent,
    EpochConfig,
    extract_eeg_epochs,
    extract_fnirs_epochs,
)
from src.models.tcn import (
    TCNConfig,
    TemporalConvolutionalNetwork,
    extract_tcn_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/Users/gnanasrisadari/Desktop/ds004514"
)

SUBJECT = "sub-01"

EEG_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "eeg"
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

FNIRS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "nirs"
    / f"{SUBJECT}_task-nirs_nirs.snirf"
)

EEG_EVENTS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "eeg"
    / f"{SUBJECT}_task-eeg_events.tsv"
)

FNIRS_EVENTS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "nirs"
    / f"{SUBJECT}_task-nirs_events.tsv"
)


# ============================================================
# EVENT LOADING
# ============================================================


TASK_LABELS = {
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
}


def load_task_events(
    path: Path,
) -> list[EpochEvent]:

    table = pd.read_csv(
        path,
        sep="\t",
    )

    required_columns = {
        "onset",
        "trial_type",
    }

    missing = required_columns.difference(
        table.columns
    )

    if missing:
        raise ValueError(
            f"Missing event columns: {sorted(missing)}"
        )

    events = []

    for _, row in table.iterrows():

        trial_type = str(
            row["trial_type"]
        )

        if trial_type not in TASK_LABELS:
            continue

        onset = float(
            row["onset"]
        )

        value = (
            None
            if "value" not in table.columns
            else row["value"]
        )

        events.append(
            EpochEvent(
                onset=onset,
                trial_type=trial_type,
                value=value,
                subject=SUBJECT,
            )
        )

    return events


# ============================================================
# REAL EEG LOADING
# ============================================================


def load_eeg() -> EEGData:

    import mne

    print("[1] Loading EEG...")

    raw = mne.io.read_raw_bdf(
        EEG_FILE,
        preload=True,
        verbose="ERROR",
    )

    data = np.asarray(
        raw.get_data(),
        dtype=np.float64,
    )

    channel_names = tuple(
        raw.ch_names
    )

    channel_types = tuple(
        raw.get_channel_types()
    )

    eeg = EEGData(
        recording_id=f"{SUBJECT}_task-eeg",
        recording_path=EEG_FILE,
        modality=__import__(
            "src.acquisition.recording",
            fromlist=["RecordingModality"],
        ).RecordingModality.EEG,
        layout=__import__(
            "src.acquisition.recording",
            fromlist=["RecordingLayout"],
        ).RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=__import__(
            "src.acquisition.eeg_loader",
            fromlist=["EEGSourceKind"],
        ).EEGSourceKind.FILE,
        adapter_name="real_validation_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=channel_names,
        data=data,
        channel_types=channel_types,
        eeg_metadata=None,
        metadata={},
    )

    print(
        f"EEG channels: {eeg.n_channels}"
    )

    print(
        f"EEG sampling rate: "
        f"{eeg.sampling_rate:.6f} Hz"
    )

    print(
        f"EEG duration: "
        f"{eeg.duration_seconds:.3f} s"
    )

    return eeg


# ============================================================
# REAL fNIRS LOADING
# ============================================================


def load_fnirs() -> FNIRSData:

    import mne

    print("[2] Loading fNIRS...")

    raw = mne.io.read_raw_snirf(
        FNIRS_FILE,
        preload=True,
        verbose="ERROR",
    )

    data = np.asarray(
        raw.get_data(),
        dtype=np.float64,
    )

    channel_names = tuple(
        raw.ch_names
    )

    channel_types = tuple(
        raw.get_channel_types()
    )

    fnirs = FNIRSData(
        recording_id=f"{SUBJECT}_task-nirs",
        recording_path=FNIRS_FILE,
        modality=__import__(
            "src.acquisition.recording",
            fromlist=["RecordingModality"],
        ).RecordingModality.FNIRS,
        layout=__import__(
            "src.acquisition.recording",
            fromlist=["RecordingLayout"],
        ).RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=__import__(
            "src.acquisition.fnirs_loader",
            fromlist=["FNIRSSourceKind"],
        ).FNIRSSourceKind.FILE,
        adapter_name="real_validation_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=channel_names,
        data=data,
        channel_types=channel_types,
        fnirs_metadata=None,
        metadata={},
    )

    print(
        f"fNIRS channels: {fnirs.n_channels}"
    )

    print(
        f"fNIRS sampling rate: "
        f"{fnirs.sampling_rate:.6f} Hz"
    )

    print(
        f"fNIRS duration: "
        f"{fnirs.duration_seconds:.3f} s"
    )

    return fnirs


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 72)

    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL TCN FEATURE VALIDATION"
    )

    print("=" * 72)

    print()

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Subject: {SUBJECT}"
    )

    print()

    # --------------------------------------------------------
    # 1. Load EEG
    # --------------------------------------------------------

    eeg = load_eeg()

    print()

    # --------------------------------------------------------
    # 2. Load fNIRS
    # --------------------------------------------------------

    fnirs = load_fnirs()

    print()

    # --------------------------------------------------------
    # 3. Load events
    # --------------------------------------------------------

    print(
        "[3] Loading task events..."
    )

    eeg_events = load_task_events(
        EEG_EVENTS_FILE
    )

    fnirs_events = load_task_events(
        FNIRS_EVENTS_FILE
    )

    print(
        f"EEG task events: "
        f"{len(eeg_events)}"
    )

    print(
        f"fNIRS task events: "
        f"{len(fnirs_events)}"
    )

    if len(eeg_events) != len(fnirs_events):
        raise AssertionError(
            "EEG and fNIRS task-event counts differ."
        )

    print(
        "PASS: Corresponding task-event "
        "counts are equal."
    )

    print()

    # --------------------------------------------------------
    # 4. Extract epochs
    # --------------------------------------------------------

    print(
        "[4] Extracting EEG epochs..."
    )

    epoch_config = EpochConfig(
        tmin=0.0,
        tmax=3.0,
        reject_nonfinite=True,
    )

    eeg_epochs = extract_eeg_epochs(
        eeg,
        eeg_events,
        config=epoch_config,
    )

    print(
        f"EEG epochs: {len(eeg_epochs)}"
    )

    print(
        "[4b] Extracting fNIRS epochs..."
    )

    fnirs_epochs = extract_fnirs_epochs(
        fnirs,
        fnirs_events,
        config=epoch_config,
    )

    print(
        f"fNIRS epochs: {len(fnirs_epochs)}"
    )

    if len(eeg_epochs) != 720:
        raise AssertionError(
            f"Expected 720 EEG epochs, "
            f"received {len(eeg_epochs)}."
        )

    if len(fnirs_epochs) != 720:
        raise AssertionError(
            f"Expected 720 fNIRS epochs, "
            f"received {len(fnirs_epochs)}."
        )

    print(
        "PASS: 720 EEG and fNIRS epochs extracted."
    )

    print()

    # --------------------------------------------------------
    # 5. Convert epochs to tensors
    # --------------------------------------------------------

    print(
        "[5] Converting epochs to PyTorch tensors..."
    )

    eeg_array = np.stack(
        [
            epoch.data
            for epoch in eeg_epochs
        ],
        axis=0,
    )

    fnirs_array = np.stack(
        [
            epoch.data
            for epoch in fnirs_epochs
        ],
        axis=0,
    )

    eeg_tensor = torch.from_numpy(
        eeg_array
    ).float()

    fnirs_tensor = torch.from_numpy(
        fnirs_array
    ).float()

    print(
        f"EEG tensor shape: "
        f"{tuple(eeg_tensor.shape)}"
    )

    print(
        f"fNIRS tensor shape: "
        f"{tuple(fnirs_tensor.shape)}"
    )

    print(
        "PASS: Epoch tensors created."
    )

    print()

    # --------------------------------------------------------
    # 6. Select device
    # --------------------------------------------------------

    print(
        "[6] Selecting computation device..."
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # 7. Create EEG TCN
    # --------------------------------------------------------

    print(
        "[7] Creating EEG TCN..."
    )

    eeg_config = TCNConfig(
        input_channels=eeg.n_channels,
        hidden_channels=32,
        embedding_dimension=128,
        kernel_size=3,
        num_blocks=4,
        dropout=0.1,
        use_batch_norm=True,
    )

    eeg_tcn = TemporalConvolutionalNetwork(
        eeg_config
    ).to(device)

    print(
        f"EEG TCN input channels: "
        f"{eeg_config.input_channels}"
    )

    print(
        f"EEG embedding dimension: "
        f"{eeg_config.embedding_dimension}"
    )

    print(
        "PASS: EEG TCN created."
    )

    print()

    # --------------------------------------------------------
    # 8. Create fNIRS TCN
    # --------------------------------------------------------

    print(
        "[8] Creating fNIRS TCN..."
    )

    fnirs_config = TCNConfig(
        input_channels=fnirs.n_channels,
        hidden_channels=32,
        embedding_dimension=128,
        kernel_size=3,
        num_blocks=3,
        dropout=0.1,
        use_batch_norm=True,
    )

    fnirs_tcn = TemporalConvolutionalNetwork(
        fnirs_config
    ).to(device)

    print(
        f"fNIRS TCN input channels: "
        f"{fnirs_config.input_channels}"
    )

    print(
        f"fNIRS embedding dimension: "
        f"{fnirs_config.embedding_dimension}"
    )

    print(
        "PASS: fNIRS TCN created."
    )

    print()

    # --------------------------------------------------------
    # 9. Extract EEG temporal representations
    # --------------------------------------------------------

    print(
        "[9] Extracting EEG temporal representations..."
    )

    eeg_tensor_device = eeg_tensor.to(
        device
    )

    eeg_representation = extract_tcn_features(
        eeg_tcn,
        eeg_tensor_device,
    )

    print(
        f"EEG temporal representation shape: "
        f"{tuple(eeg_representation.shape)}"
    )

    expected_eeg_shape = (
        len(eeg_epochs),
        eeg_config.embedding_dimension,
    )

    if tuple(eeg_representation.shape) != expected_eeg_shape:
        raise AssertionError(
            "EEG TCN representation shape is incorrect."
        )

    print(
        "PASS: EEG temporal representations "
        "have the expected dimensions."
    )

    print()

    # --------------------------------------------------------
    # 10. Extract fNIRS temporal representations
    # --------------------------------------------------------

    print(
        "[10] Extracting fNIRS temporal representations..."
    )

    fnirs_tensor_device = fnirs_tensor.to(
        device
    )

    fnirs_representation = extract_tcn_features(
        fnirs_tcn,
        fnirs_tensor_device,
    )

    print(
        f"fNIRS temporal representation shape: "
        f"{tuple(fnirs_representation.shape)}"
    )

    expected_fnirs_shape = (
        len(fnirs_epochs),
        fnirs_config.embedding_dimension,
    )

    if tuple(fnirs_representation.shape) != expected_fnirs_shape:
        raise AssertionError(
            "fNIRS TCN representation shape is incorrect."
        )

    print(
        "PASS: fNIRS temporal representations "
        "have the expected dimensions."
    )

    print()

    # --------------------------------------------------------
    # 11. Integrity validation
    # --------------------------------------------------------

    print(
        "[11] Checking representation integrity..."
    )

    if not torch.isfinite(
        eeg_representation
    ).all():

        raise AssertionError(
            "EEG TCN representation contains "
            "NaN or infinite values."
        )

    if not torch.isfinite(
        fnirs_representation
    ).all():

        raise AssertionError(
            "fNIRS TCN representation contains "
            "NaN or infinite values."
        )

    print(
        "PASS: All TCN representations "
        "contain finite values."
    )

    print()

    # --------------------------------------------------------
    # 12. Task-label validation
    # --------------------------------------------------------

    print(
        "[12] Validating task labels..."
    )

    eeg_labels = [
        epoch.trial_type
        for epoch in eeg_epochs
    ]

    fnirs_labels = [
        epoch.trial_type
        for epoch in fnirs_epochs
    ]

    eeg_counts = Counter(
        eeg_labels
    )

    fnirs_counts = Counter(
        fnirs_labels
    )

    for label in sorted(
        TASK_LABELS
    ):

        print(
            f"{label:40s}: "
            f"{eeg_counts[label]}"
        )

        if eeg_counts[label] != 90:
            raise AssertionError(
                f"EEG task '{label}' does not "
                "contain 90 epochs."
            )

        if fnirs_counts[label] != 90:
            raise AssertionError(
                f"fNIRS task '{label}' does not "
                "contain 90 epochs."
            )

    print(
        "PASS: All 8 task classes contain "
        "90 temporal representations each."
    )

    print()

    # --------------------------------------------------------
    # 13. EEG/fNIRS pairing validation
    # --------------------------------------------------------

    print(
        "[13] Validating EEG/fNIRS temporal "
        "representation pairing..."
    )

    for index in range(
        len(eeg_epochs)
    ):

        if (
            eeg_epochs[index].trial_type
            != fnirs_epochs[index].trial_type
        ):

            raise AssertionError(
                "EEG/fNIRS task labels are not aligned."
            )

    print(
        "PASS: EEG and fNIRS temporal "
        "representations are correctly paired."
    )

    print()

    # --------------------------------------------------------
    # 14. First representations
    # --------------------------------------------------------

    print(
        "[14] First EEG temporal representation"
    )

    print(
        f"Subject: "
        f"{eeg_epochs[0].subject}"
    )

    print(
        f"Task: "
        f"{eeg_epochs[0].trial_type}"
    )

    print(
        f"EEG representation: "
        f"{tuple(eeg_representation[0].shape)}"
    )

    print()

    print(
        "[15] First fNIRS temporal representation"
    )

    print(
        f"Subject: "
        f"{fnirs_epochs[0].subject}"
    )

    print(
        f"Task: "
        f"{fnirs_epochs[0].trial_type}"
    )

    print(
        f"fNIRS representation: "
        f"{tuple(fnirs_representation[0].shape)}"
    )

    print()

    print("=" * 72)

    print(
        "PHASE 3 REAL TCN FEATURE VALIDATION PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
