"""
Neuro-Symphony Phase 3 — Real EEG Feature Validation.

Validates universal EEG feature extraction using real
ds004514 task epochs.

The feature extractor itself remains dataset-independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import mne

from src.acquisition.eeg_loader import EEGData
from src.acquisition.recording import (
    RecordingLayout,
    RecordingModality,
)
from src.acquisition.eeg_loader import EEGSourceKind
from src.epochs.epoch_extractor import (
    EpochConfig,
    EpochEvent,
    extract_eeg_epochs,
)
from src.features.eeg_features import (
    extract_eeg_features,
    eeg_feature_vector,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path.home() / "Desktop" / "ds004514"

SUBJECT = "sub-01"

EPOCH_CONFIG = EpochConfig(
    tmin=0.0,
    tmax=3.0,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NEURO-SYMPHONY PHASE 3 — REAL EEG FEATURE VALIDATION")
    print("=" * 72)

    subject_root = DATASET_ROOT / SUBJECT
    eeg_root = subject_root / "eeg"

    events_file = (
        eeg_root /
        f"{SUBJECT}_task-eeg_events.tsv"
    )

    eeg_file = (
        eeg_root /
        f"{SUBJECT}_task-eeg_eeg.bdf"
    )

    print()
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Subject: {SUBJECT}")

    # ========================================================
    # 1. LOAD EEG
    # ========================================================

    print()
    print("[1] Loading EEG...")

    raw = mne.io.read_raw_bdf(
        eeg_file,
        preload=True,
        verbose="ERROR",
    )

    print(
        f"EEG channels: {len(raw.ch_names)}"
    )

    print(
        f"EEG sampling rate: "
        f"{raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"EEG duration: "
        f"{raw.times[-1]:.3f} s"
    )

    # ========================================================
    # 2. CREATE UNIVERSAL EEGData
    # ========================================================

    print()
    print("[2] Creating standardized EEGData...")

    data = raw.get_data()

    eeg_data = EEGData(
        recording_id=f"{SUBJECT}_task-eeg",
        recording_path=eeg_file,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_validation",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=tuple(
            raw.ch_names
        ),
        data=np.asarray(
            data,
            dtype=np.float64,
        ),
        channel_types=tuple(
            raw.get_channel_types()
        ),
        metadata={
            "validation_dataset": "ds004514",
        },
    )

    print(
        f"Standardized shape: "
        f"{eeg_data.data.shape}"
    )

    # ========================================================
    # 3. LOAD TASK EVENTS
    # ========================================================

    print()
    print("[3] Loading task events...")

    events_df = pd.read_csv(
        events_file,
        sep="\t",
    )

    task_names = [
        name
        for name in events_df["trial_type"].unique()
        if (
            "imagery_task" in str(name)
            or "silent_naming_task" in str(name)
        )
    ]

    task_events = events_df[
        events_df["trial_type"].isin(task_names)
    ].copy()

    print(
        f"Task events: {len(task_events)}"
    )

    events = tuple(
        EpochEvent(
            onset=float(row["onset"]),
            trial_type=str(
                row["trial_type"]
            ),
            value=row.get("value"),
            subject=SUBJECT,
        )
        for _, row in task_events.iterrows()
    )

    # ========================================================
    # 4. EXTRACT EEG EPOCHS
    # ========================================================

    print()
    print("[4] Extracting EEG epochs...")

    epochs = extract_eeg_epochs(
        eeg_data,
        events,
        config=EPOCH_CONFIG,
    )

    print(
        f"Successfully extracted: "
        f"{len(epochs)}"
    )

    if len(epochs) == 0:
        raise RuntimeError(
            "No EEG epochs were extracted."
        )

    # ========================================================
    # 5. EXTRACT FEATURES
    # ========================================================

    print()
    print("[5] Extracting EEG features...")

    features = extract_eeg_features(
        epochs
    )

    print(
        f"Feature objects: "
        f"{len(features)}"
    )

    if len(features) != len(epochs):
        raise AssertionError(
            "Number of feature objects does not "
            "match number of epochs."
        )

    # ========================================================
    # 6. VALIDATE FEATURE SHAPES
    # ========================================================

    print()
    print("[6] Validating feature shapes...")

    n_channels = len(
        epochs[0].channel_names
    )

    print(
        f"Channels: {n_channels}"
    )

    print(
        f"Mean shape: "
        f"{features[0].mean.shape}"
    )

    print(
        f"Variance shape: "
        f"{features[0].variance.shape}"
    )

    print(
        f"RMS shape: "
        f"{features[0].rms.shape}"
    )

    for feature in features:

        if feature.mean.shape != (n_channels,):
            raise AssertionError(
                "Mean feature shape is incorrect."
            )

        if feature.variance.shape != (n_channels,):
            raise AssertionError(
                "Variance feature shape is incorrect."
            )

        if feature.std.shape != (n_channels,):
            raise AssertionError(
                "STD feature shape is incorrect."
            )

        if feature.rms.shape != (n_channels,):
            raise AssertionError(
                "RMS feature shape is incorrect."
            )

    print(
        "PASS: Time-domain feature shapes are valid."
    )

    # ========================================================
    # 7. VALIDATE FREQUENCY FEATURES
    # ========================================================

    print()
    print("[7] Validating frequency-domain features...")

    expected_bands = {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    }

    for feature in features:

        if set(feature.band_power) != expected_bands:
            raise AssertionError(
                "Unexpected EEG frequency bands."
            )

        if set(feature.relative_band_power) != expected_bands:
            raise AssertionError(
                "Unexpected relative frequency bands."
            )

        for band in expected_bands:

            if feature.band_power[band].shape != (
                n_channels,
            ):
                raise AssertionError(
                    f"{band} power shape is incorrect."
                )

            if feature.relative_band_power[band].shape != (
                n_channels,
            ):
                raise AssertionError(
                    f"{band} relative power shape is incorrect."
                )

    print(
        "PASS: Delta/theta/alpha/beta/gamma "
        "features are valid."
    )

    # ========================================================
    # 8. CHECK FINITE VALUES
    # ========================================================

    print()
    print("[8] Checking feature integrity...")

    for feature in features:

        arrays = [
            feature.mean,
            feature.variance,
            feature.std,
            feature.rms,
            feature.total_power,
        ]

        arrays.extend(
            feature.band_power.values()
        )

        arrays.extend(
            feature.relative_band_power.values()
        )

        for array in arrays:

            if not np.isfinite(array).all():
                raise AssertionError(
                    "EEG features contain NaN "
                    "or infinite values."
                )

    print(
        "PASS: All EEG features contain finite values."
    )

    # ========================================================
    # 9. FLAT FEATURE VECTOR
    # ========================================================

    print()
    print("[9] Testing flat feature representation...")

    vector = eeg_feature_vector(
        features[0]
    )

    print(
        f"Feature vector shape: "
        f"{vector.shape}"
    )

    expected_feature_count = (
        n_channels * 15
    )

    if vector.shape != (
        expected_feature_count,
    ):
        raise AssertionError(
            "Unexpected flat feature vector size. "
            f"Expected {expected_feature_count}, "
            f"got {vector.shape[0]}."
        )

    if not np.isfinite(vector).all():
        raise AssertionError(
            "Flat feature vector contains "
            "non-finite values."
        )

    print(
        "PASS: Flat EEG feature vector is valid."
    )

    # ========================================================
    # 10. TASK LABEL VALIDATION
    # ========================================================

    print()
    print("[10] Validating task labels...")

    label_counts = pd.Series(
        feature.trial_type
        for feature in features
    ).value_counts()

    print(
        label_counts.to_string()
    )

    # ========================================================
    # 11. FIRST FEATURE OBJECT
    # ========================================================

    first = features[0]

    print()
    print("[11] First EEG feature object")

    print(
        f"Subject: {first.subject}"
    )

    print(
        f"Task: {first.trial_type}"
    )

    print(
        f"Event onset: "
        f"{first.event_onset:.6f}"
    )

    print(
        f"Mean shape: "
        f"{first.mean.shape}"
    )

    print(
        f"Alpha power shape: "
        f"{first.band_power['alpha'].shape}"
    )

    print(
        f"Feature vector length: "
        f"{len(vector)}"
    )

    print()
    print("=" * 72)
    print("PHASE 3 REAL EEG FEATURE VALIDATION PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()