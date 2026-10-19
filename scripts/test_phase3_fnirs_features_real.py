"""
Neuro-Symphony Phase 3 — Real fNIRS Feature Validation.

Validates universal fNIRS feature extraction using the real
ds004514 dataset.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from src.acquisition.fnirs_loader import FNIRSData, FNIRSSourceKind
from src.acquisition.recording import RecordingLayout, RecordingModality
from src.epochs.epoch_extractor import (
    EpochEvent,
    EpochConfig,
    extract_fnirs_epochs,
)
from src.features.fnirs_features import (
    FNIRSFeature,
    extract_fnirs_features,
    fnirs_feature_vector,
)


DATASET = Path.home() / "Desktop" / "ds004514"


def main():

    print("=" * 72)
    print("NEURO-SYMPHONY PHASE 3 — REAL fNIRS FEATURE VALIDATION")
    print("=" * 72)

    subject = "sub-01"
    subject_dir = DATASET / subject

    nirs_file = (
        subject_dir
        / "nirs"
        / f"{subject}_task-nirs_nirs.snirf"
    )

    events_file = (
        subject_dir
        / "eeg"
        / f"{subject}_task-eeg_events.tsv"
    )

    print()
    print(f"Dataset: {DATASET}")
    print(f"Subject: {subject}")

    # ========================================================
    # 1. LOAD fNIRS
    # ========================================================

    print()
    print("[1] Loading fNIRS...")

    raw = mne.io.read_raw_snirf(
        nirs_file,
        preload=True,
        verbose="ERROR",
    )

    data = np.asarray(
        raw.get_data(),
        dtype=np.float64,
    )

    print(
        f"fNIRS channels: {data.shape[0]}"
    )

    print(
        f"fNIRS sampling rate: "
        f"{raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"fNIRS duration: "
        f"{raw.times[-1]:.3f} s"
    )

    # ========================================================
    # 2. CREATE STANDARDIZED FNIRSData
    # ========================================================

    print()
    print("[2] Creating standardized FNIRSData...")

    channel_names = tuple(
        raw.ch_names
    )

    channel_types = tuple(
        raw.get_channel_types()
    )

    fnirs_data = FNIRSData(
        recording_id=f"{subject}_task-nirs",
        recording_path=nirs_file,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=subject,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_snirf_test",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=channel_names,
        data=data,
        channel_types=channel_types,
        metadata={
            "dataset": "ds004514",
        },
    )

    print(
        f"Standardized shape: "
        f"{fnirs_data.data.shape}"
    )

    # ========================================================
    # 3. LOAD EVENTS
    # ========================================================

    print()
    print("[3] Loading task events...")

    events_df = pd.read_csv(
        events_file,
        sep="\t",
    )

    task_events = events_df[
        events_df["trial_type"].astype(str).str.contains(
            "imagery|naming",
            case=False,
            regex=True,
        )
    ].copy()

    events = []

    for _, row in task_events.iterrows():

        events.append(
            EpochEvent(
                onset=float(
                    row["onset"]
                ),
                trial_type=str(
                    row["trial_type"]
                ),
                value=row.get(
                    "value",
                    None,
                ),
                subject=subject,
            )
        )

    print(
        f"Task events: {len(events)}"
    )

    # ========================================================
    # 4. EXTRACT fNIRS EPOCHS
    # ========================================================

    print()
    print("[4] Extracting fNIRS epochs...")

    epochs = extract_fnirs_epochs(
        fnirs_data,
        events,
        config=EpochConfig(
            tmin=0.0,
            tmax=3.0,
        ),
    )

    print(
        f"Successfully extracted: "
        f"{len(epochs)}"
    )

    if len(epochs) == 0:
        raise RuntimeError(
            "No fNIRS epochs were extracted."
        )

    # ========================================================
    # 5. EXTRACT FEATURES
    # ========================================================

    print()
    print("[5] Extracting fNIRS features...")

    features = extract_fnirs_features(
        epochs
    )

    print(
        f"Feature objects: "
        f"{len(features)}"
    )

    if len(features) != len(epochs):
        raise AssertionError(
            "Feature count does not match epoch count."
        )

    # ========================================================
    # 6. VALIDATE FEATURE SHAPES
    # ========================================================

    print()
    print("[6] Validating feature shapes...")

    first = features[0]

    n_channels = len(
        first.channel_names
    )

    print(
        f"Channels: {n_channels}"
    )

    print(
        f"Mean shape: "
        f"{first.mean.shape}"
    )

    print(
        f"Variance shape: "
        f"{first.variance.shape}"
    )

    print(
        f"Response change shape: "
        f"{first.response_change.shape}"
    )

    expected_shape = (
        n_channels,
    )

    feature_arrays = [
        first.mean,
        first.variance,
        first.std,
        first.minimum,
        first.maximum,
        first.peak,
        first.peak_time,
        first.baseline_mean,
        first.response_mean,
        first.response_change,
    ]

    for values in feature_arrays:

        if values.shape != expected_shape:
            raise AssertionError(
                f"Invalid feature shape: "
                f"{values.shape}"
            )

    print(
        "PASS: fNIRS feature shapes are valid."
    )

    # ========================================================
    # 7. FEATURE INTEGRITY
    # ========================================================

    print()
    print("[7] Checking feature integrity...")

    for feature in features:

        if not isinstance(
            feature,
            FNIRSFeature,
        ):
            raise AssertionError(
                "Invalid FNIRSFeature object."
            )

        arrays = [
            feature.mean,
            feature.variance,
            feature.std,
            feature.minimum,
            feature.maximum,
            feature.peak,
            feature.peak_time,
            feature.baseline_mean,
            feature.response_mean,
            feature.response_change,
        ]

        for values in arrays:

            if not np.isfinite(values).all():
                raise AssertionError(
                    "fNIRS feature contains "
                    "NaN or infinite values."
                )

    print(
        "PASS: All fNIRS features contain finite values."
    )

    # ========================================================
    # 8. FLAT FEATURE VECTOR
    # ========================================================

    print()
    print("[8] Testing flat feature representation...")

    vector = fnirs_feature_vector(
        first
    )

    print(
        f"Feature vector shape: "
        f"{vector.shape}"
    )

    expected_length = (
        n_channels * 10
    )

    if vector.shape != (
        expected_length,
    ):
        raise AssertionError(
            f"Expected feature vector length "
            f"{expected_length}, got {vector.shape[0]}"
        )

    if not np.isfinite(vector).all():
        raise AssertionError(
            "Flat feature vector contains "
            "NaN or infinite values."
        )

    print(
        "PASS: Flat fNIRS feature vector is valid."
    )

    # ========================================================
    # 9. VALIDATE TASK LABELS
    # ========================================================

    print()
    print("[9] Validating task labels...")

    label_counts = Counter(
        feature.trial_type
        for feature in features
    )

    for label, count in label_counts.items():

        print(
            f"{label:<40} {count}"
        )

    # ========================================================
    # 10. FIRST FEATURE OBJECT
    # ========================================================

    print()
    print("[10] First fNIRS feature object")

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
        f"Peak shape: "
        f"{first.peak.shape}"
    )

    print(
        f"Response change shape: "
        f"{first.response_change.shape}"
    )

    print(
        f"Feature vector length: "
        f"{vector.shape[0]}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PHASE 3 REAL fNIRS FEATURE VALIDATION PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()