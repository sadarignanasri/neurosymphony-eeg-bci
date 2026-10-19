"""
NEURO-SYMPHONY PHASE 3
REAL DATASET EPOCH EXTRACTION VALIDATION

Validation dataset:
    ds004514

This script validates the universal epoch extractor against
real EEG/fNIRS task events.

Dataset-specific information is used only by this validation
script. The actual epoch extraction implementation remains
universal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.epochs import (
    EpochConfig,
    EpochEvent,
    extract_multimodal_epochs,
)
from src.synchronization.timestamp_alignment import (
    TimestampAlignment,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path.home() / "Desktop" / "ds004514"

# Already validated in Phase 3 timestamp alignment.
EEG_TO_FNIRS_OFFSET_SECONDS = 51.891824

# Dataset task labels already audited.
TASK_LABELS = (
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
)


# ============================================================
# FIND EVENTS
# ============================================================

def find_events_file(subject_dir: Path) -> Path:

    candidates = sorted(
        subject_dir.rglob("*events.tsv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No events.tsv found under {subject_dir}"
        )

    return candidates[0]


# ============================================================
# FIND EEG
# ============================================================

def find_eeg_file(subject_dir: Path) -> Path:

    candidates = sorted(
        subject_dir.rglob("*.bdf")
    )

    if not candidates:
        candidates = sorted(
            subject_dir.rglob("*eeg*.set")
        )

    if not candidates:
        candidates = sorted(
            subject_dir.rglob("*eeg*.edf")
        )

    if not candidates:
        raise FileNotFoundError(
            f"No EEG file found under {subject_dir}"
        )

    return candidates[0]


# ============================================================
# FIND fNIRS
# ============================================================

def find_fnirs_file(subject_dir: Path) -> Path:

    candidates = sorted(
        subject_dir.rglob("*.snirf")
    )

    if not candidates:
        candidates = sorted(
            subject_dir.rglob("*nirs*.edf")
        )

    if not candidates:
        raise FileNotFoundError(
            f"No fNIRS file found under {subject_dir}"
        )

    return candidates[0]


# ============================================================
# LOAD EVENTS
# ============================================================

def load_task_events(
    events_file: Path,
    subject: str,
) -> list[EpochEvent]:

    events = pd.read_csv(
        events_file,
        sep="\t",
    )

    required = {
        "onset",
        "trial_type",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"Events file missing columns: {missing}"
        )

    events = events[
        events["trial_type"].isin(TASK_LABELS)
    ].copy()

    result = []

    for _, row in events.iterrows():

        result.append(
            EpochEvent(
                onset=float(row["onset"]),
                trial_type=str(row["trial_type"]),
                value=(
                    row["value"]
                    if "value" in row
                    else None
                ),
                subject=subject,
            )
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NEURO-SYMPHONY PHASE 3 — REAL EPOCH EXTRACTION")
    print("=" * 72)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    subjects = sorted(
        path
        for path in DATASET_ROOT.glob("sub-*")
        if path.is_dir()
    )

    if not subjects:
        raise RuntimeError(
            "No BIDS subjects found."
        )

    print()
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Subjects found: {len(subjects)}")

    # --------------------------------------------------------
    # Validate one real subject first.
    # --------------------------------------------------------

    subject_dir = subjects[0]
    subject = subject_dir.name

    print()
    print("=" * 72)
    print(f"TEST SUBJECT: {subject}")
    print("=" * 72)

    events_file = find_events_file(
        subject_dir
    )

    eeg_file = find_eeg_file(
        subject_dir
    )

    fnirs_file = find_fnirs_file(
        subject_dir
    )

    print()
    print("Events file:")
    print(events_file)

    print()
    print("EEG file:")
    print(eeg_file)

    print()
    print("fNIRS file:")
    print(fnirs_file)

    # --------------------------------------------------------
    # Load MNE.
    # --------------------------------------------------------

    import mne

    print()
    print("[1] Loading EEG...")

    eeg_raw = mne.io.read_raw_bdf(
        eeg_file,
        preload=True,
        verbose="ERROR",
    )

    print(
        f"EEG channels: {len(eeg_raw.ch_names)}"
    )

    print(
        f"EEG sampling rate: "
        f"{eeg_raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"EEG duration: "
        f"{eeg_raw.times[-1]:.3f} s"
    )

    # --------------------------------------------------------
    # Load fNIRS.
    # --------------------------------------------------------

    print()
    print("[2] Loading fNIRS...")

    fnirs_raw = mne.io.read_raw_snirf(
        fnirs_file,
        preload=True,
        verbose="ERROR",
    )

    print(
        f"fNIRS channels: "
        f"{len(fnirs_raw.ch_names)}"
    )

    print(
        f"fNIRS sampling rate: "
        f"{fnirs_raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"fNIRS duration: "
        f"{fnirs_raw.times[-1]:.3f} s"
    )

    # --------------------------------------------------------
    # Load task events.
    # --------------------------------------------------------

    print()
    print("[3] Loading task events...")

    events = load_task_events(
        events_file,
        subject,
    )

    print(
        f"Task events: {len(events)}"
    )

    # --------------------------------------------------------
    # Convert MNE data into the standardized structures
    # expected by the universal epoch extractor.
    #
    # This is ONLY the validation boundary.
    # --------------------------------------------------------

    from src.acquisition.eeg_loader import (
        EEGData,
        EEGSourceKind,
    )

    from src.acquisition.fnirs_loader import (
        FNIRSData,
        FNIRSSourceKind,
    )

    from src.acquisition.recording import (
        RecordingLayout,
        RecordingModality,
    )

    eeg_data = EEGData(
        recording_id=f"{subject}_eeg",
        recording_path=eeg_file,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=subject,
        session_id=None,
        task_id=None,
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_dataset_validation",
        sampling_rate=float(
            eeg_raw.info["sfreq"]
        ),
        channel_names=tuple(
            eeg_raw.ch_names
        ),
        data=np.asarray(
            eeg_raw.get_data()
        ),
        channel_types=tuple(
            eeg_raw.get_channel_types()
        ),
        metadata={
            "dataset_id": "ds004514",
            "phase": 3,
        },
    )

    fnirs_data = FNIRSData(
        recording_id=f"{subject}_fnirs",
        recording_path=fnirs_file,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=subject,
        session_id=None,
        task_id=None,
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_dataset_validation",
        sampling_rate=float(
            fnirs_raw.info["sfreq"]
        ),
        channel_names=tuple(
            fnirs_raw.ch_names
        ),
        data=np.asarray(
            fnirs_raw.get_data()
        ),
        channel_types=tuple(
            fnirs_raw.get_channel_types()
        ),
        metadata={
            "dataset_id": "ds004514",
            "phase": 3,
        },
    )

    # --------------------------------------------------------
    # Timestamp alignment.
    # --------------------------------------------------------

    alignment = TimestampAlignment(
        offset_seconds=EEG_TO_FNIRS_OFFSET_SECONDS
    )

    # --------------------------------------------------------
    # Extract synchronized epochs.
    #
    # We use exactly the 3-second task duration established
    # by the real dataset audit.
    # --------------------------------------------------------

    print()
    print("[4] Extracting synchronized epochs...")

    config = EpochConfig(
        tmin=0.0,
        tmax=3.0,
    )

    multimodal_epochs = extract_multimodal_epochs(
        eeg_data=eeg_data,
        fnirs_data=fnirs_data,
        events=events,
        alignment=alignment,
        config=config,
    )

    print(
        f"Successfully extracted: "
        f"{len(multimodal_epochs)}"
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    print()
    print("[5] Validating epoch shapes...")

    if not multimodal_epochs:
        raise RuntimeError(
            "No multimodal epochs were extracted."
        )

    eeg_shapes = {
        epoch.eeg_epoch.data.shape
        for epoch in multimodal_epochs
    }

    fnirs_shapes = {
        epoch.fnirs_epoch.data.shape
        for epoch in multimodal_epochs
    }

    print(
        f"Unique EEG epoch shapes: "
        f"{eeg_shapes}"
    )

    print(
        f"Unique fNIRS epoch shapes: "
        f"{fnirs_shapes}"
    )

    # --------------------------------------------------------
    # Validate task labels.
    # --------------------------------------------------------

    extracted_labels = {
        epoch.trial_type
        for epoch in multimodal_epochs
    }

    print()
    print("[6] Validating task labels...")

    for label in TASK_LABELS:

        count = sum(
            epoch.trial_type == label
            for epoch in multimodal_epochs
        )

        print(
            f"{label:40s}: {count}"
        )

    missing_labels = (
        set(TASK_LABELS) - extracted_labels
    )

    if missing_labels:
        raise RuntimeError(
            f"Missing task labels: {missing_labels}"
        )

    # --------------------------------------------------------
    # Validate synchronization.
    # --------------------------------------------------------

    print()
    print("[7] Validating EEG → fNIRS event alignment...")

    first = multimodal_epochs[0]

    expected_fnirs_onset = (
        first.event_onset_eeg
        + EEG_TO_FNIRS_OFFSET_SECONDS
    )

    difference = abs(
        first.event_onset_fnirs
        - expected_fnirs_onset
    )

    print(
        f"EEG onset: "
        f"{first.event_onset_eeg:.6f} s"
    )

    print(
        f"fNIRS onset: "
        f"{first.event_onset_fnirs:.6f} s"
    )

    print(
        f"Expected fNIRS onset: "
        f"{expected_fnirs_onset:.6f} s"
    )

    print(
        f"Conversion error: "
        f"{difference:.9f} s"
    )

    if difference > 1e-9:
        raise RuntimeError(
            "EEG → fNIRS timestamp conversion failed."
        )

    # --------------------------------------------------------
    # Validate finite data.
    # --------------------------------------------------------

    print()
    print("[8] Checking epoch data integrity...")

    for epoch in multimodal_epochs:

        if not np.isfinite(
            epoch.eeg_epoch.data
        ).all():
            raise RuntimeError(
                "EEG epoch contains NaN/Inf."
            )

        if not np.isfinite(
            epoch.fnirs_epoch.data
        ).all():
            raise RuntimeError(
                "fNIRS epoch contains NaN/Inf."
            )

    print(
        "PASS: All extracted epochs contain "
        "finite values."
    )

    # --------------------------------------------------------
    # Show first epoch.
    # --------------------------------------------------------

    print()
    print("[9] First multimodal epoch")

    print(
        f"Subject: {first.subject}"
    )

    print(
        f"Task: {first.trial_type}"
    )

    print(
        f"EEG shape: "
        f"{first.eeg_epoch.data.shape}"
    )

    print(
        f"fNIRS shape: "
        f"{first.fnirs_epoch.data.shape}"
    )

    print(
        f"EEG onset: "
        f"{first.event_onset_eeg:.6f}"
    )

    print(
        f"fNIRS onset: "
        f"{first.event_onset_fnirs:.6f}"
    )

    # --------------------------------------------------------
    # Final.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 3 REAL EPOCH EXTRACTION VALIDATION PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()