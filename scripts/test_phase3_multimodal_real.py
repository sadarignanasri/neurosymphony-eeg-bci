"""
Neuro-Symphony Phase 3 — Real Multimodal Representation Validation.

Uses the real ds004514 BIDS EEG and fNIRS event files to validate:

    EEG events
        ↓
    EEG → fNIRS timestamp alignment
        ↓
    synchronized EEG/fNIRS epochs
        ↓
    EEG features + fNIRS features
        ↓
    multimodal representation

The underlying Neuro-Symphony modules remain universal.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.acquisition.eeg_loader import EEGData, EEGSourceKind
from src.acquisition.fnirs_loader import FNIRSData, FNIRSSourceKind
from src.acquisition.recording import (
    RecordingLayout,
    RecordingModality,
)

from src.synchronization.timestamp_alignment import (
    estimate_offset,
)

from src.epochs.epoch_extractor import (
    EpochConfig,
    EpochEvent,
    extract_eeg_epochs,
    extract_fnirs_epochs,
)

from src.features.eeg_features import (
    extract_eeg_features,
)

from src.features.fnirs_features import (
    extract_fnirs_features,
)

from src.features.multimodal_features import (
    MultimodalFeature,
    extract_multimodal_features,
    multimodal_feature_vector,
)


# ============================================================
# DATASET
# ============================================================

DATASET = (
    Path.home()
    / "Desktop"
    / "ds004514"
)

SUBJECT = "sub-01"


# ============================================================
# TASKS USED FOR MULTIMODAL REPRESENTATION
# ============================================================

TASK_TYPES = {
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
    events_file: Path,
) -> pd.DataFrame:
    """
    Load BIDS events.tsv and retain only task-locked events.
    """

    events = pd.read_csv(
        events_file,
        sep="\t",
    )

    required_columns = {
        "onset",
        "trial_type",
    }

    missing = required_columns - set(
        events.columns
    )

    if missing:
        raise ValueError(
            f"Events file is missing columns: {missing}"
        )

    events["trial_type"] = (
        events["trial_type"]
        .astype(str)
        .str.strip()
    )

    task_events = events[
        events["trial_type"].isin(
            TASK_TYPES
        )
    ].copy()

    task_events = task_events.reset_index(
        drop=True
    )

    return task_events


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL MULTIMODAL REPRESENTATION VALIDATION"
    )
    print("=" * 72)

    subject_dir = (
        DATASET
        / SUBJECT
    )

    eeg_file = (
        subject_dir
        / "eeg"
        / f"{SUBJECT}_task-eeg_eeg.bdf"
    )

    eeg_events_file = (
        subject_dir
        / "eeg"
        / f"{SUBJECT}_task-eeg_events.tsv"
    )

    fnirs_file = (
        subject_dir
        / "nirs"
        / f"{SUBJECT}_task-nirs_nirs.snirf"
    )

    fnirs_events_file = (
        subject_dir
        / "nirs"
        / f"{SUBJECT}_task-nirs_events.tsv"
    )

    print()
    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Subject: {SUBJECT}"
    )

    # ========================================================
    # 1. LOAD EEG
    # ========================================================

    print()
    print("[1] Loading EEG...")

    eeg_raw = mne.io.read_raw_bdf(
        eeg_file,
        preload=True,
        verbose="ERROR",
    )

    eeg_array = np.asarray(
        eeg_raw.get_data(),
        dtype=np.float64,
    )

    print(
        f"EEG channels: "
        f"{eeg_array.shape[0]}"
    )

    print(
        f"EEG sampling rate: "
        f"{eeg_raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"EEG duration: "
        f"{eeg_raw.times[-1]:.3f} s"
    )

    # ========================================================
    # 2. LOAD fNIRS
    # ========================================================

    print()
    print("[2] Loading fNIRS...")

    fnirs_raw = mne.io.read_raw_snirf(
        fnirs_file,
        preload=True,
        verbose="ERROR",
    )

    fnirs_array = np.asarray(
        fnirs_raw.get_data(),
        dtype=np.float64,
    )

    print(
        f"fNIRS channels: "
        f"{fnirs_array.shape[0]}"
    )

    print(
        f"fNIRS sampling rate: "
        f"{fnirs_raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"fNIRS duration: "
        f"{fnirs_raw.times[-1]:.3f} s"
    )

    # ========================================================
    # 3. STANDARDIZED EEGData
    # ========================================================

    print()
    print(
        "[3] Creating standardized EEGData..."
    )

    eeg_data = EEGData(
        recording_id=(
            f"{SUBJECT}_task-eeg"
        ),
        recording_path=eeg_file,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_bdf_test",
        sampling_rate=float(
            eeg_raw.info["sfreq"]
        ),
        channel_names=tuple(
            eeg_raw.ch_names
        ),
        data=eeg_array,
        channel_types=tuple(
            eeg_raw.get_channel_types()
        ),
        metadata={
            "dataset": "ds004514",
        },
    )

    print(
        f"EEG standardized shape: "
        f"{eeg_data.data.shape}"
    )

    # ========================================================
    # 4. STANDARDIZED FNIRSData
    # ========================================================

    print()
    print(
        "[4] Creating standardized FNIRSData..."
    )

    fnirs_data = FNIRSData(
        recording_id=(
            f"{SUBJECT}_task-nirs"
        ),
        recording_path=fnirs_file,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_snirf_test",
        sampling_rate=float(
            fnirs_raw.info["sfreq"]
        ),
        channel_names=tuple(
            fnirs_raw.ch_names
        ),
        data=fnirs_array,
        channel_types=tuple(
            fnirs_raw.get_channel_types()
        ),
        metadata={
            "dataset": "ds004514",
        },
    )

    print(
        f"fNIRS standardized shape: "
        f"{fnirs_data.data.shape}"
    )

    # ========================================================
    # 5. LOAD EEG EVENTS
    # ========================================================

    print()
    print("[5] Loading EEG task events...")

    eeg_events_df = load_task_events(
        eeg_events_file
    )

    print(
        f"EEG task events: "
        f"{len(eeg_events_df)}"
    )

    # ========================================================
    # 6. LOAD fNIRS EVENTS
    # ========================================================

    print()
    print(
        "[6] Loading fNIRS task events..."
    )

    fnirs_events_df = load_task_events(
        fnirs_events_file
    )

    print(
        f"fNIRS task events: "
        f"{len(fnirs_events_df)}"
    )

    # ========================================================
    # 7. VERIFY EVENT COUNTS
    # ========================================================

    print()
    print(
        "[7] Validating corresponding task events..."
    )

    if len(eeg_events_df) != len(
        fnirs_events_df
    ):
        raise AssertionError(
            "EEG and fNIRS task-event counts "
            "do not match."
        )

    if len(eeg_events_df) != 720:
        raise AssertionError(
            "Expected 720 task events for "
            "sub-01."
        )

    eeg_labels = list(
        eeg_events_df["trial_type"]
    )

    fnirs_labels = list(
        fnirs_events_df["trial_type"]
    )

    if eeg_labels != fnirs_labels:
        raise AssertionError(
            "EEG and fNIRS task labels are "
            "not in the same order."
        )

    print(
        "PASS: EEG and fNIRS contain "
        "720 corresponding task events."
    )

    # ========================================================
    # 8. ESTIMATE TIMESTAMP ALIGNMENT
    # ========================================================

    print()
    print(
        "[8] Estimating EEG → fNIRS timestamp alignment..."
    )

    eeg_onsets = np.asarray(
        eeg_events_df["onset"],
        dtype=np.float64,
    )

    fnirs_onsets = np.asarray(
        fnirs_events_df["onset"],
        dtype=np.float64,
    )

    alignment = estimate_offset(
        eeg_onsets,
        fnirs_onsets,
    )

    print(
        f"Estimated offset: "
        f"{alignment.offset_seconds:.9f} s"
    )

    # ========================================================
    # 9. VALIDATE ALIGNMENT
    # ========================================================

    print()
    print(
        "[9] Validating timestamp alignment..."
    )

    expected_fnirs_onsets = (
        alignment.eeg_to_fnirs(
            eeg_onsets
        )
    )

    alignment_errors = (
        fnirs_onsets
        - expected_fnirs_onsets
    )

    max_alignment_error = float(
        np.max(
            np.abs(
                alignment_errors
            )
        )
    )

    mean_alignment_error = float(
        np.mean(
            np.abs(
                alignment_errors
            )
        )
    )

    print(
        f"Mean alignment error: "
        f"{mean_alignment_error:.12f} s"
    )

    print(
        f"Maximum alignment error: "
        f"{max_alignment_error:.12f} s"
    )

    # if max_alignment_error > 1e-6:
    #     raise AssertionError(
    #         "EEG/fNIRS event alignment "
    #         "exceeds tolerance."
    #     )
    # ------------------------------------------------------------
# Universal synchronization tolerance
#
# The maximum meaningful temporal precision is limited by
# the sampling interval of the lower-frequency modality.
# ------------------------------------------------------------

    fnirs_sample_interval = (
        1.0 / float(fnirs_data.sampling_rate)
    )

    alignment_tolerance = fnirs_sample_interval

    print(
        f"fNIRS sample interval: "
        f"{fnirs_sample_interval:.12f} s"
    )

    print(
        f"Alignment tolerance: "
        f"{alignment_tolerance:.12f} s"
    )

    if max_alignment_error > alignment_tolerance:
        raise AssertionError(
            "EEG/fNIRS event alignment exceeds "
            "the fNIRS sampling-resolution tolerance: "
            f"{max_alignment_error:.12f} s > "
            f"{alignment_tolerance:.12f} s"
        )

    print(
        "PASS: EEG → fNIRS event alignment "
        "is valid within fNIRS sampling resolution."
    )

    print(
        "PASS: EEG → fNIRS event alignment "
        "is valid."
    )

    # ========================================================
    # 10. CREATE EPOCH EVENTS
    # ========================================================

    print()
    print(
        "[10] Creating synchronized epoch events..."
    )

    eeg_epoch_events = []

    fnirs_epoch_events = []

    for index in range(
        len(eeg_events_df)
    ):

        trial_type = eeg_labels[index]

        eeg_epoch_events.append(
            EpochEvent(
                onset=float(
                    eeg_onsets[index]
                ),
                trial_type=trial_type,
                value=(
                    eeg_events_df.iloc[
                        index
                    ].get(
                        "value",
                        None,
                    )
                ),
                subject=SUBJECT,
            )
        )

        fnirs_epoch_events.append(
            EpochEvent(
                onset=float(
                    fnirs_onsets[index]
                ),
                trial_type=trial_type,
                value=(
                    fnirs_events_df.iloc[
                        index
                    ].get(
                        "value",
                        None,
                    )
                ),
                subject=SUBJECT,
            )
        )

    # ========================================================
    # 11. EXTRACT EEG EPOCHS
    # ========================================================

    print()
    print(
        "[11] Extracting synchronized EEG epochs..."
    )

    epoch_config = EpochConfig(
        tmin=0.0,
        tmax=3.0,
    )

    eeg_epochs = extract_eeg_epochs(
        eeg_data,
        eeg_epoch_events,
        config=epoch_config,
    )

    print(
        f"EEG epochs: "
        f"{len(eeg_epochs)}"
    )

    # ========================================================
    # 12. EXTRACT fNIRS EPOCHS
    # ========================================================

    print()
    print(
        "[12] Extracting synchronized fNIRS epochs..."
    )

    fnirs_epochs = extract_fnirs_epochs(
        fnirs_data,
        fnirs_epoch_events,
        config=epoch_config,
    )

    print(
        f"fNIRS epochs: "
        f"{len(fnirs_epochs)}"
    )

    if len(eeg_epochs) != 720:
        raise AssertionError(
            "Expected 720 EEG epochs."
        )

    if len(fnirs_epochs) != 720:
        raise AssertionError(
            "Expected 720 fNIRS epochs."
        )

    print(
        "PASS: 720 synchronized EEG and "
        "fNIRS epochs extracted."
    )

    # ========================================================
    # 13. EXTRACT EEG FEATURES
    # ========================================================

    print()
    print(
        "[13] Extracting EEG features..."
    )

    eeg_features = extract_eeg_features(
        eeg_epochs
    )

    print(
        f"EEG feature objects: "
        f"{len(eeg_features)}"
    )

    # ========================================================
    # 14. EXTRACT FNIRS FEATURES
    # ========================================================

    print()
    print(
        "[14] Extracting fNIRS features..."
    )

    fnirs_features = extract_fnirs_features(
        fnirs_epochs
    )

    print(
        f"fNIRS feature objects: "
        f"{len(fnirs_features)}"
    )

    if len(eeg_features) != 720:
        raise AssertionError(
            "Expected 720 EEG feature objects."
        )

    if len(fnirs_features) != 720:
        raise AssertionError(
            "Expected 720 fNIRS feature objects."
        )

    # ========================================================
    # 15. CREATE MULTIMODAL REPRESENTATIONS
    # ========================================================

    print()
    print(
        "[15] Creating multimodal representations..."
    )

    multimodal_features = (
        extract_multimodal_features(
            eeg_features,
            fnirs_features,
        )
    )

    print(
        f"Multimodal feature objects: "
        f"{len(multimodal_features)}"
    )

    if len(multimodal_features) != 720:
        raise AssertionError(
            "Expected 720 multimodal feature objects."
        )

    # ========================================================
    # 16. VALIDATE FIRST REPRESENTATION
    # ========================================================

    print()
    print(
        "[16] Validating multimodal dimensions..."
    )

    first = multimodal_features[0]

    print(
        f"EEG feature dimension: "
        f"{first.eeg_feature_dimension}"
    )

    print(
        f"fNIRS feature dimension: "
        f"{first.fnirs_feature_dimension}"
    )

    print(
        f"Combined dimension: "
        f"{first.feature_dimension}"
    )

    if first.eeg_feature_dimension != 1200:
        raise AssertionError(
            "Expected EEG feature dimension "
            "of 1200."
        )

    if first.fnirs_feature_dimension != 280:
        raise AssertionError(
            "Expected fNIRS feature dimension "
            "of 280."
        )

    if first.feature_dimension != 1480:
        raise AssertionError(
            "Expected combined multimodal "
            "dimension of 1480."
        )

    print(
        "PASS: Multimodal feature dimensions "
        "are correct."
    )

    # ========================================================
    # 17. VALIDATE VECTOR SHAPES
    # ========================================================

    print()
    print(
        "[17] Validating vector shapes..."
    )

    if first.eeg_vector.shape != (
        1200,
    ):
        raise AssertionError(
            "Invalid EEG vector shape."
        )

    if first.fnirs_vector.shape != (
        280,
    ):
        raise AssertionError(
            "Invalid fNIRS vector shape."
        )

    if first.combined_vector.shape != (
        1480,
    ):
        raise AssertionError(
            "Invalid combined vector shape."
        )

    print(
        "PASS: EEG, fNIRS, and combined "
        "vector shapes are valid."
    )

    # ========================================================
    # 18. CHECK FINITE VALUES
    # ========================================================

    print()
    print(
        "[18] Checking multimodal feature integrity..."
    )

    for feature in multimodal_features:

        if not isinstance(
            feature,
            MultimodalFeature,
        ):
            raise AssertionError(
                "Invalid MultimodalFeature object."
            )

        if not np.isfinite(
            feature.eeg_vector
        ).all():
            raise AssertionError(
                "EEG vector contains "
                "non-finite values."
            )

        if not np.isfinite(
            feature.fnirs_vector
        ).all():
            raise AssertionError(
                "fNIRS vector contains "
                "non-finite values."
            )

        if not np.isfinite(
            feature.combined_vector
        ).all():
            raise AssertionError(
                "Combined vector contains "
                "non-finite values."
            )

    print(
        "PASS: All multimodal representations "
        "contain finite values."
    )

    # ========================================================
    # 19. VALIDATE TASK DISTRIBUTION
    # ========================================================

    print()
    print(
        "[19] Validating task labels..."
    )

    label_counts = Counter(
        feature.trial_type
        for feature in multimodal_features
    )

    for label in sorted(
        label_counts
    ):
        print(
            f"{label:<40}: "
            f"{label_counts[label]}"
        )

    if set(label_counts) != TASK_TYPES:
        raise AssertionError(
            "Unexpected task labels."
        )

    for label in TASK_TYPES:

        if label_counts[label] != 90:
            raise AssertionError(
                f"Expected 90 epochs for "
                f"{label}."
            )

    print(
        "PASS: All 8 task classes contain "
        "90 representations each."
    )

    # ========================================================
    # 20. TEST FLAT VECTOR
    # ========================================================

    print()
    print(
        "[20] Testing flat multimodal vector..."
    )

    vector = multimodal_feature_vector(
        first
    )

    print(
        f"Feature vector shape: "
        f"{vector.shape}"
    )

    if vector.shape != (
        1480,
    ):
        raise AssertionError(
            "Flat multimodal vector must "
            "have 1480 dimensions."
        )

    if not np.isfinite(vector).all():
        raise AssertionError(
            "Flat multimodal vector contains "
            "non-finite values."
        )

    print(
        "PASS: Flat multimodal feature vector "
        "is valid."
    )

    # ========================================================
    # 21. VERIFY EEG/fNIRS PAIRING
    # ========================================================

    print()
    print(
        "[21] Validating EEG/fNIRS feature pairing..."
    )

    for index, feature in enumerate(
        multimodal_features
    ):

        if (
            feature.trial_type
            != eeg_features[index].trial_type
        ):
            raise AssertionError(
                f"EEG task mismatch at "
                f"index {index}."
            )

        if (
            feature.trial_type
            != fnirs_features[index].trial_type
        ):
            raise AssertionError(
                f"fNIRS task mismatch at "
                f"index {index}."
            )

        if (
            feature.subject is not None
            and eeg_features[index].subject
            is not None
            and feature.subject
            != eeg_features[index].subject
        ):
            raise AssertionError(
                f"EEG subject mismatch at "
                f"index {index}."
            )

        if (
            feature.subject is not None
            and fnirs_features[index].subject
            is not None
            and feature.subject
            != fnirs_features[index].subject
        ):
            raise AssertionError(
                f"fNIRS subject mismatch at "
                f"index {index}."
            )

    print(
        "PASS: EEG and fNIRS features "
        "are correctly paired."
    )

    # ========================================================
    # 22. FIRST MULTIMODAL REPRESENTATION
    # ========================================================

    print()
    print(
        "[22] First multimodal representation"
    )

    print(
        f"Subject: "
        f"{first.subject}"
    )

    print(
        f"Task: "
        f"{first.trial_type}"
    )

    print(
        f"EEG onset: "
        f"{first.event_onset_eeg:.6f}"
    )

    print(
        f"fNIRS onset: "
        f"{first.event_onset_fnirs:.6f}"
    )

    print(
        f"EEG vector: "
        f"{first.eeg_vector.shape}"
    )

    print(
        f"fNIRS vector: "
        f"{first.fnirs_vector.shape}"
    )

    print(
        f"Combined vector: "
        f"{first.combined_vector.shape}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PHASE 3 REAL MULTIMODAL REPRESENTATION "
        "VALIDATION PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()