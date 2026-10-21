
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS — ORIGINAL DATASET IS READ-ONLY
# =========================================================
ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATA_ROOT = ROOT / "major datasets" / "28955456" / "epo_data"

OUTPUT_ROOT = (
    ROOT
    / "experiments"
    / "dataset3_28955456_training"
    / "experiment139_preprocessing"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. CONFIGURATION
# =========================================================
FS = 256
WINDOW_START_SEC = 13.0
WINDOW_END_SEC = 22.0
EPS = 1e-20

CONFIGS = {
    "bandpass_8_30": {
        "low": 8.0, "high": 30.0, "notch": False
    },
    "bandpass_8_30_notch50": {
        "low": 8.0, "high": 30.0, "notch": True
    },
    "bandpass_1_40": {
        "low": 1.0, "high": 40.0, "notch": False
    },
    "bandpass_4_30": {
        "low": 4.0, "high": 30.0, "notch": False
    },
}

# =========================================================
# 3. DISCOVER DATASET FILES
# =========================================================
def find_dataset_files():
    files = []

    for group_name in ("Normals", "Patients"):
        eeg_dir = DATA_ROOT / group_name / "EEG"

        if not eeg_dir.is_dir():
            raise FileNotFoundError(
                f"Missing EEG directory: {eeg_dir}"
            )

        for path in sorted(eeg_dir.rglob("*_epo.mat")):
            participant = f"{group_name}_{path.parent.name}"
            files.append((path, group_name, participant))

    if not files:
        raise FileNotFoundError(
            f"No EEG epoch files found under {DATA_ROOT}"
        )

    participant_count = len(
        {participant for _, _, participant in files}
    )

    print(f"EEG files: {len(files)}")
    print(f"Participants: {participant_count}")

    return files


# =========================================================
# 4. LOAD MATLAB EPOCH FILE
# =========================================================
def load_epoch_file(path):
    mat = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False,
    )

    if "epo" not in mat:
        raise ValueError(f"Missing epo structure: {path}")

    epo = mat["epo"][0, 0]

    X = np.asarray(epo.X, dtype=np.float64)
    y = np.asarray(epo.y).reshape(-1).astype(int)
    fs = float(np.asarray(epo.fs).reshape(-1)[0])

    # Original layout: samples x channels x trials.
    if X.ndim != 3 or X.shape[2] != len(y):
        raise ValueError(
            f"Unexpected EEG shape or labels: {path}, {X.shape}"
        )

    if not np.isclose(fs, FS):
        raise ValueError(
            f"Expected {FS} Hz, found {fs}: {path}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"NaN/Inf values found: {path}")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Unexpected labels {np.unique(y)} in {path}"
        )

    # Convert to trials x channels x samples.
    X = np.transpose(X, (2, 1, 0))

    return X, y


# =========================================================
# 5. FILTERING AND LOG-VARIANCE FEATURES
# =========================================================
def preprocess_and_extract_log_variance(trials, config):
    start = int(round(WINDOW_START_SEC * FS))
    end = int(round(WINDOW_END_SEC * FS))

    if trials.shape[-1] < end:
        raise ValueError(
            f"Epoch has {trials.shape[-1]} samples; "
            f"at least {end} are required."
        )

    # Same candidate window for every configuration.
    window = trials[:, :, start:end].copy()

    # Optional 50 Hz notch filter.
    if config["notch"]:
        b, a = iirnotch(
            w0=50.0,
            Q=30.0,
            fs=FS,
        )
        notch_sos = tf2sos(b, a)
        window = sosfiltfilt(
            notch_sos,
            window,
            axis=-1,
        )

    # Configuration-specific Butterworth band-pass.
    sos = butter(
        4,
        [config["low"], config["high"]],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        window,
        axis=-1,
    )

    # One log-variance feature per channel.
    features = np.log(
        np.var(filtered, axis=-1) + EPS
    )

    return features


# =========================================================
# 6. BUILD FEATURE DATASETS
# =========================================================
def build_dataset():
    feature_parts = {
        name: [] for name in CONFIGS
    }

    labels = []
    participants = []
    groups = []
    source_files = []
    trial_numbers = []

    files = find_dataset_files()

    for file_index, (path, group, participant) in enumerate(
        files, start=1
    ):
        trials, y = load_epoch_file(path)

        for config_name, config in CONFIGS.items():
            features = preprocess_and_extract_log_variance(
                trials,
                config,
            )
            feature_parts[config_name].append(features)

        labels.extend(y.tolist())
        participants.extend([participant] * len(y))
        groups.extend([group] * len(y))
        source_files.extend([str(path)] * len(y))
        trial_numbers.extend(range(len(y)))

        print(
            f"[{file_index}/{len(files)}] "
            f"{participant} | {path.name} | trials={len(y)}"
        )

    X_features = {
        name: np.concatenate(parts, axis=0)
        for name, parts in feature_parts.items()
    }

    metadata = pd.DataFrame({
        "label": labels,
        "participant": participants,
        "group": groups,
        "source_file": source_files,
        "trial_number": trial_numbers,
    })

    # Save inventory in this experiment's folder.
    metadata.to_csv(
        OUTPUT_ROOT / "trial_inventory.csv",
        index=False,
    )

    return X_features, metadata


# =========================================================
# 7. LEAVE-ONE-PARTICIPANT-OUT EVALUATION
# =========================================================
def evaluate(X_features, metadata):
    y = metadata["label"].to_numpy()
    participant_ids = metadata["participant"].to_numpy()

    unique_participants = np.unique(participant_ids)

    model_template = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
    )

    fold_rows = []
    prediction_rows = []
    start_time = time.time()

    for config_name, X in X_features.items():
        print(
            f"\nEvaluating: {config_name} "
            f"| feature shape={X.shape}"
        )

        for fold_number, held_out in enumerate(
            unique_participants,
            start=1,
        ):
            test_mask = participant_ids == held_out
            train_mask = ~test_mask

            y_train = y[train_mask]
            y_test = y[test_mask]

            if np.unique(y_train).size != 2:
                raise ValueError(
                    f"Training fold lacks a class: {held_out}"
                )

            # Fresh model for every held-out participant.
            model = clone(model_template)
            model.fit(X[train_mask], y_train)

            pred = model.predict(X[test_mask])

            # Participant-level metrics.
            fold_rows.append({
                "preprocessing": config_name,
                "held_out_participant": held_out,
                "group": metadata.loc[
                    test_mask, "group"
                ].iloc[0],
                "n_trials": int(test_mask.sum()),
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, pred
                ),
                "macro_f1": f1_score(
                    y_test,
                    pred,
                    average="macro",
                    zero_division=0,
                ),
            })

            # IMPORTANT FIX:
            # pred is indexed locally within the held-out fold.
            # global_idx indexes the full dataset.
            test_indices = np.flatnonzero(test_mask)

            if len(test_indices) != len(pred):
                raise RuntimeError(
                    f"Prediction count mismatch for {held_out}"
                )

            for local_idx, global_idx in enumerate(test_indices):
                prediction_rows.append({
                    "preprocessing": config_name,
                    "participant": participant_ids[global_idx],
                    "group": metadata.iloc[global_idx]["group"],
                    "source_file": metadata.iloc[
                        global_idx
                    ]["source_file"],
                    "trial_number": int(
                        metadata.iloc[global_idx]["trial_number"]
                    ),
                    "true_label": int(y[global_idx]),
                    "predicted_label": int(pred[local_idx]),
                })

            if (
                fold_number % 5 == 0
                or fold_number == len(unique_participants)
            ):
                print(
                    f"  Completed fold {fold_number}/"
                    f"{len(unique_participants)}"
                )

    # -----------------------------------------------------
    # 8. SAVE FOLD METRICS AND PREDICTIONS
    # -----------------------------------------------------
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    folds.to_csv(
        OUTPUT_ROOT / "participant_fold_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_ROOT / "out_of_fold_predictions.csv",
        index=False,
    )

    # Verify every trial has one prediction per configuration.
    expected_predictions = len(metadata) * len(CONFIGS)

    if len(predictions) != expected_predictions:
        raise RuntimeError(
            f"Expected {expected_predictions} predictions, "
            f"found {len(predictions)}."
        )

    duplicate_count = predictions.duplicated(
        subset=["preprocessing", "source_file", "trial_number"]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate trial predictions."
        )

    # -----------------------------------------------------
    # 9. COMPARE CONFIGURATIONS
    # -----------------------------------------------------
    summary_rows = []

    for name, group in predictions.groupby(
        "preprocessing",
        sort=False,
    ):
        participant_folds = folds[
            folds["preprocessing"] == name
        ]

        summary_rows.append({
            "preprocessing": name,
            "pooled_accuracy": accuracy_score(
                group["true_label"],
                group["predicted_label"],
            ),
            "pooled_balanced_accuracy": balanced_accuracy_score(
                group["true_label"],
                group["predicted_label"],
            ),
            "pooled_macro_f1": f1_score(
                group["true_label"],
                group["predicted_label"],
                average="macro",
                zero_division=0,
            ),
            "mean_participant_accuracy":
                participant_folds["accuracy"].mean(),
            "std_participant_accuracy":
                participant_folds["accuracy"].std(),
            "mean_participant_balanced_accuracy":
                participant_folds["balanced_accuracy"].mean(),
            "mean_participant_macro_f1":
                participant_folds["macro_f1"].mean(),
            "participants":
                int(
                    participant_folds[
                        "held_out_participant"
                    ].nunique()
                ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["pooled_balanced_accuracy", "pooled_macro_f1"],
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_ROOT / "preprocessing_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 10. SAVE JSON SUMMARY
    # -----------------------------------------------------
    result = {
        "experiment": 139,
        "description": (
            "Compare log-variance EEG features under four "
            "filtering configurations using leave-one-"
            "participant-out evaluation."
        ),
        "original_dataset_modified": False,
        "sampling_rate_hz": FS,
        "window_epoch_relative_seconds": [
            WINDOW_START_SEC,
            WINDOW_END_SEC,
        ],
        "participants": int(
            metadata["participant"].nunique()
        ),
        "trials": int(len(metadata)),
        "configurations": CONFIGS,
        "prediction_rows": int(len(predictions)),
        "elapsed_seconds": round(
            time.time() - start_time, 2
        ),
        "results": summary.to_dict(orient="records"),
        "limitations": [
            "The physiological validity of the time window "
            "has not been independently established.",
            "The configurations are exploratory comparisons.",
            "The selected window is filtered separately from "
            "the full epoch, unlike Experiment 134.",
            "Selecting the best configuration on these same "
            "participants does not provide independent test "
            "performance.",
        ],
    }

    with open(
        OUTPUT_ROOT / "preprocessing_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 76)
    print("EXPERIMENT 139 COMPLETE")
    print("=" * 76)
    print(summary.to_string(index=False))
    print(f"\nPrediction rows verified: {len(predictions)}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")
    print(f"Outputs saved to:\n{OUTPUT_ROOT}")


# =========================================================
# 11. MAIN
# =========================================================
def main():
    print("=" * 76)
    print("EXPERIMENT 139 — EEG PREPROCESSING COMPARISON")
    print("=" * 76)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Output:  {OUTPUT_ROOT}")
    print("Original dataset files will only be read.\n")

    X_features, metadata = build_dataset()

    print(f"\nTotal trials: {len(metadata)}")
    print(
        f"Unique participants: "
        f"{metadata['participant'].nunique()}"
    )
    print("\nClass counts:")
    print(metadata["label"].value_counts().sort_index())

    evaluate(X_features, metadata)


if __name__ == "__main__":
    main()
