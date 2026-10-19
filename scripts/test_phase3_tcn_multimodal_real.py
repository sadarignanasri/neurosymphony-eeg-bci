"""
Neuro-Symphony Phase 3 — Real TCN Multimodal Representation Validation.

Pipeline:

    EEG epoch
        ↓
    EEG TCN
        ↓
    128-D EEG temporal representation
        \
         → 256-D multimodal representation
        /
    128-D fNIRS temporal representation
        ↓
    fNIRS TCN

This validates the actual Phase 3 temporal representation
pipeline on the real ds004514 dataset.

The implementation remains dataset-independent.
The dataset is used only for validation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mne

from src.acquisition.eeg_loader import EEGData
from src.acquisition.fnirs_loader import FNIRSData
from src.acquisition.recording import (
    RecordingLayout,
    RecordingModality,
)
from src.acquisition.eeg_loader import EEGSourceKind
from src.acquisition.fnirs_loader import FNIRSSourceKind

from src.epochs.epoch_extractor import (
    Epoch,
    EpochConfig,
    EpochEvent,
    extract_eeg_epochs,
    extract_fnirs_epochs,
)

from src.models.tcn import (
    TCNConfig,
    TemporalConvolutionalNetwork,
    extract_tcn_features,
)

from src.models.multimodal import (
    MultimodalRepresentation,
    MultimodalRepresentationConfig,
    multimodal_feature_vector,
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


# ============================================================
# EVENT LOADING
# ============================================================

def load_task_events(
    path: Path,
) -> list[EpochEvent]:

    table = pd.read_csv(
        path,
        sep="\t",
    )

    events = []

    for _, row in table.iterrows():

        trial_type = str(
            row["trial_type"]
        )

        if trial_type not in TASK_LABELS:
            continue

        events.append(
            EpochEvent(
                onset=float(row["onset"]),
                trial_type=trial_type,
                value=(
                    row["value"]
                    if "value" in table.columns
                    else None
                ),
                subject=SUBJECT,
            )
        )

    return events


# ============================================================
# EEG
# ============================================================

def load_eeg() -> EEGData:

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

    eeg = EEGData(
        recording_id=f"{SUBJECT}_task-eeg",
        recording_path=EEG_FILE,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_validation_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=tuple(
            raw.ch_names
        ),
        data=data,
        channel_types=tuple(
            raw.get_channel_types()
        ),
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

    return eeg


# ============================================================
# fNIRS
# ============================================================

def load_fnirs() -> FNIRSData:

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

    fnirs = FNIRSData(
        recording_id=f"{SUBJECT}_task-nirs",
        recording_path=FNIRS_FILE,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_validation_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=tuple(
            raw.ch_names
        ),
        data=data,
        channel_types=tuple(
            raw.get_channel_types()
        ),
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

    return fnirs


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL TCN MULTIMODAL REPRESENTATION VALIDATION"
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

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    eeg = load_eeg()

    print()

    fnirs = load_fnirs()

    print()

    # ========================================================
    # 2. LOAD EVENTS
    # ========================================================

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

    if len(eeg_events) != 720:
        raise AssertionError(
            "Expected 720 EEG task events."
        )

    if len(fnirs_events) != 720:
        raise AssertionError(
            "Expected 720 fNIRS task events."
        )

    print(
        "PASS: 720 EEG and fNIRS task events loaded."
    )

    print()

    # ========================================================
    # 3. EXTRACT EPOCHS
    # ========================================================

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
            "EEG epoch count is not 720."
        )

    if len(fnirs_epochs) != 720:
        raise AssertionError(
            "fNIRS epoch count is not 720."
        )

    print(
        "PASS: 720 synchronized task epochs "
        "available for both modalities."
    )

    print()

    # ========================================================
    # 4. CONVERT TO TENSORS
    # ========================================================

    print(
        "[5] Creating epoch tensors..."
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
        f"EEG tensor: "
        f"{tuple(eeg_tensor.shape)}"
    )

    print(
        f"fNIRS tensor: "
        f"{tuple(fnirs_tensor.shape)}"
    )

    print()

    # ========================================================
    # 5. DEVICE
    # ========================================================

    print(
        "[6] Selecting computation device..."
    )

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    print()

    # ========================================================
    # 6. EEG TCN
    # ========================================================

    print(
        "[7] Creating EEG TCN..."
    )

    eeg_tcn_config = TCNConfig(
        input_channels=eeg.n_channels,
        hidden_channels=32,
        embedding_dimension=128,
        kernel_size=3,
        num_blocks=4,
        dropout=0.1,
        use_batch_norm=True,
    )

    eeg_tcn = TemporalConvolutionalNetwork(
        eeg_tcn_config
    ).to(device)

    print(
        "EEG TCN created."
    )

    print()

    # ========================================================
    # 7. fNIRS TCN
    # ========================================================

    print(
        "[8] Creating fNIRS TCN..."
    )

    fnirs_tcn_config = TCNConfig(
        input_channels=fnirs.n_channels,
        hidden_channels=32,
        embedding_dimension=128,
        kernel_size=3,
        num_blocks=3,
        dropout=0.1,
        use_batch_norm=True,
    )

    fnirs_tcn = TemporalConvolutionalNetwork(
        fnirs_tcn_config
    ).to(device)

    print(
        "fNIRS TCN created."
    )

    print()

    # ========================================================
    # 8. TCN REPRESENTATIONS
    # ========================================================

    print(
        "[9] Extracting EEG TCN representations..."
    )

    eeg_representation = extract_tcn_features(
        eeg_tcn,
        eeg_tensor.to(device),
    )

    print(
        f"EEG representation: "
        f"{tuple(eeg_representation.shape)}"
    )

    print(
        "[10] Extracting fNIRS TCN representations..."
    )

    fnirs_representation = extract_tcn_features(
        fnirs_tcn,
        fnirs_tensor.to(device),
    )

    print(
        f"fNIRS representation: "
        f"{tuple(fnirs_representation.shape)}"
    )

    expected_eeg = (
        720,
        128,
    )

    expected_fnirs = (
        720,
        128,
    )

    if tuple(
        eeg_representation.shape
    ) != expected_eeg:

        raise AssertionError(
            "Unexpected EEG TCN representation shape."
        )

    if tuple(
        fnirs_representation.shape
    ) != expected_fnirs:

        raise AssertionError(
            "Unexpected fNIRS TCN representation shape."
        )

    print(
        "PASS: EEG and fNIRS TCN representations "
        "have expected dimensions."
    )

    print()

    # ========================================================
    # 9. MULTIMODAL REPRESENTATION
    # ========================================================

    print(
        "[11] Creating multimodal representation..."
    )

    multimodal_config = (
        MultimodalRepresentationConfig(
            eeg_dimension=128,
            fnirs_dimension=128,
            output_dimension=None,
            dropout=0.0,
        )
    )

    multimodal_model = (
        MultimodalRepresentation(
            multimodal_config
        ).to(device)
    )

    multimodal_model.eval()

    with torch.no_grad():

        multimodal_representation = (
            multimodal_model(
                eeg_representation,
                fnirs_representation,
            )
        )

    print(
        f"Multimodal representation: "
        f"{tuple(multimodal_representation.shape)}"
    )

    expected_multimodal_shape = (
        720,
        256,
    )

    if tuple(
        multimodal_representation.shape
    ) != expected_multimodal_shape:

        raise AssertionError(
            "Unexpected multimodal representation shape."
        )

    print(
        "PASS: 256-dimensional multimodal "
        "temporal representation created."
    )

    print()

    # ========================================================
    # 10. INTEGRITY
    # ========================================================

    print(
        "[12] Checking representation integrity..."
    )

    if not torch.isfinite(
        eeg_representation
    ).all():

        raise AssertionError(
            "EEG representation contains "
            "non-finite values."
        )

    if not torch.isfinite(
        fnirs_representation
    ).all():

        raise AssertionError(
            "fNIRS representation contains "
            "non-finite values."
        )

    if not torch.isfinite(
        multimodal_representation
    ).all():

        raise AssertionError(
            "Multimodal representation contains "
            "non-finite values."
        )

    print(
        "PASS: All temporal representations "
        "contain finite values."
    )

    print()

    # ========================================================
    # 11. LABEL VALIDATION
    # ========================================================

    print(
        "[13] Validating task labels..."
    )

    eeg_labels = [
        epoch.trial_type
        for epoch in eeg_epochs
    ]

    fnirs_labels = [
        epoch.trial_type
        for epoch in fnirs_epochs
    ]

    if eeg_labels != fnirs_labels:

        raise AssertionError(
            "EEG and fNIRS labels are not "
            "in the same order."
        )

    counts = Counter(
        eeg_labels
    )

    for label in sorted(
        TASK_LABELS
    ):

        print(
            f"{label:40s}: "
            f"{counts[label]}"
        )

        if counts[label] != 90:

            raise AssertionError(
                f"Task '{label}' does not "
                "contain exactly 90 representations."
            )

    print(
        "PASS: All 8 task classes contain "
        "90 multimodal representations."
    )

    print()

    # ========================================================
    # 12. REPRESENTATION PAIRING
    # ========================================================

    print(
        "[14] Validating EEG/fNIRS representation pairing..."
    )

    if (
        eeg_representation.shape[0]
        != fnirs_representation.shape[0]
    ):

        raise AssertionError(
            "EEG and fNIRS representation "
            "sample counts differ."
        )

    if (
        eeg_representation.shape[0]
        != multimodal_representation.shape[0]
    ):

        raise AssertionError(
            "Multimodal representation sample "
            "count does not match modalities."
        )

    print(
        "PASS: EEG, fNIRS, and multimodal "
        "representations are correctly paired."
    )

    print()

    # ========================================================
    # 13. FEATURE VECTOR TEST
    # ========================================================

    print(
        "[15] Testing multimodal feature vector..."
    )

    vector = multimodal_feature_vector(
        multimodal_representation
    )

    print(
        f"Feature matrix shape: "
        f"{tuple(vector.shape)}"
    )

    if tuple(
        vector.shape
    ) != (720, 256):

        raise AssertionError(
            "Invalid multimodal feature-vector shape."
        )

    print(
        "PASS: Multimodal feature representation "
        "is valid."
    )

    print()

    # ========================================================
    # 14. FIRST REPRESENTATION
    # ========================================================

    print(
        "[16] First multimodal temporal representation"
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

    print(
        f"fNIRS representation: "
        f"{tuple(fnirs_representation[0].shape)}"
    )

    print(
        f"Combined representation: "
        f"{tuple(multimodal_representation[0].shape)}"
    )

    print()

    print("=" * 72)

    print(
        "PHASE 3 REAL TCN MULTIMODAL "
        "REPRESENTATION VALIDATION PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
