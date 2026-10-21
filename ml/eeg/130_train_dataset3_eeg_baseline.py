
"""
Experiment 130: Dataset 3 EEG motor-imagery baseline.

Task:
    Binary classification: Left vs Right motor imagery.

Pipeline:
    1. Read original MATLAB epoch files without modifying them.
    2. Apply 8-30 Hz EEG band-pass filtering.
    3. Extract the 13-22 second epoch-relative window.
    4. Extract channel-wise log-variance features.
    5. Train an RBF-SVM with scaling inside the pipeline.
    6. Evaluate using leave-one-participant-out validation.
    7. Save predictions, metrics, and run metadata.

All outputs are saved under:
    experiments/dataset3_28955456_training/

Important:
    Passing the structural audit does not independently establish
    that the selected window contains the intended brain activity.
    The window is chosen to end before the next marker described
    in the original epoch-generation code.
"""

from pathlib import Path
import csv
import json
import time
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATASET_ROOT = PROJECT_ROOT / "major datasets" / "28955456"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "dataset3_28955456_training"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS_EXPECTED = 256
EXPECTED_CHANNELS = 32
EXPECTED_TRIALS = 30

FILTER_LOW_HZ = 8.0
FILTER_HIGH_HZ = 30.0
FILTER_ORDER = 4

# Epoch-relative time. This is a candidate window, not yet
# an independently verified physiological interval.
WINDOW_START_SECONDS = 13
WINDOW_END_SECONDS = 22

RANDOM_STATE = 42

LABEL_NAMES = {
    0: "Left",
    1: "Right",
}


# ============================================================
# 2. READ MATLAB FILES
# ============================================================

def load_epoch_file(path):
    """Read X, y and fs from the MATLAB 'epo' structure."""

    mat = loadmat(
        str(path),
        squeeze_me=False,
        struct_as_record=True,
    )

    if "epo" not in mat:
        raise ValueError("Missing MATLAB variable 'epo'.")

    container = mat["epo"]

    if container.size != 1:
        raise ValueError(
            f"Expected one epo structure; found {container.shape}"
        )

    epo = container[0, 0]

    if epo.dtype.names is None:
        raise ValueError("epo is not a structured MATLAB record.")

    for field in ("X", "y", "fs"):
        if field not in epo.dtype.names:
            raise ValueError(f"Missing epo field: {field}")

    # Correct extraction: these fields already contain their data.
    X = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1).astype(int)
    fs = float(np.asarray(epo["fs"]).reshape(-1)[0])

    if X.ndim != 3:
        raise ValueError(f"Expected 3D X; found shape {X.shape}")

    # Expected MATLAB layout: samples x channels x trials.
    if X.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} channels; got {X.shape[1]}"
        )

    if X.shape[2] != EXPECTED_TRIALS:
        raise ValueError(
            f"Expected {EXPECTED_TRIALS} trials; got {X.shape[2]}"
        )

    if y.size != X.shape[2]:
        raise ValueError(
            f"Label count {y.size} does not match trial count {X.shape[2]}"
        )

    if not np.isclose(fs, FS_EXPECTED):
        raise ValueError(f"Unexpected sampling frequency: {fs}")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Unexpected labels: {np.unique(y).tolist()}")

    if not np.isfinite(X).all():
        raise ValueError("EEG array contains NaN or infinite values.")

    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))

    if label_counts != {0: 15, 1: 15}:
        raise ValueError(f"Unexpected class distribution: {label_counts}")

    return X, y, fs


# ============================================================
# 3. PREPROCESS AND EXTRACT FEATURES
# ============================================================

def extract_features(X, fs):
    """
    X shape: samples x channels x trials.

    Returns:
        features: trials x channels
        window_sample_count: number of samples in the selected window

    Filtering is performed on each complete epoch before selecting
    the candidate window, reducing filtering edge effects at the
    selected window boundaries.
    """

    if not FILTER_LOW_HZ < FILTER_HIGH_HZ < fs / 2:
        raise ValueError("Invalid band-pass frequencies.")

    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    # Filtering along the samples axis.
    filtered = sosfiltfilt(sos, X, axis=0)

    # Reproduce the original MATLAB 1-based inclusive start index:
    # MATLAB: sampleRate * startTime
    # Python: sampleRate * startTime - 1
    start_idx = int(round(fs * WINDOW_START_SECONDS)) - 1

    # MATLAB end index is inclusive; Python end index is exclusive.
    end_idx = int(round(fs * WINDOW_END_SECONDS))

    if start_idx < 0 or end_idx > filtered.shape[0]:
        raise ValueError(
            f"Window [{WINDOW_START_SECONDS}, {WINDOW_END_SECONDS}] "
            f"seconds exceeds epoch length {filtered.shape[0] / fs:.3f}s."
        )

    window = filtered[start_idx:end_idx, :, :]

    if window.shape[0] < 2:
        raise ValueError("Selected window is too short.")

    # Variance over time, separately for every channel and trial.
    # Window shape: samples x channels x trials.
    variance = np.var(window, axis=0, ddof=1).T

    # Shape: trials x channels.
    features = np.log(np.maximum(variance, 1e-20))

    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values generated.")

    return features, int(window.shape[0])


# ============================================================
# 4. COLLECT ALL PARTICIPANTS
# ============================================================

def collect_dataset():
    all_features = []
    all_labels = []
    all_groups = []
    all_file_paths = []
    all_trial_numbers = []
    all_participant_groups = []
    inventory = []

    for group_name in ("Normals", "Patients"):
        eeg_dir = DATASET_ROOT / "epo_data" / group_name / "EEG"

        if not eeg_dir.exists():
            warnings.warn(f"Directory not found: {eeg_dir}")
            continue

        paths = sorted(eeg_dir.rglob("*_epo.mat"))
        print(f"{group_name}: {len(paths)} files found")

        for file_index, path in enumerate(paths, start=1):
            try:
                X, y, fs = load_epoch_file(path)
                features, window_samples = extract_features(X, fs)

                # The subject folder is the participant identifier.
                # Group is included because subject1 in Normals is not
                # assumed to be the same person as subject1 in Patients.
                subject_folder = path.parent.name
                participant_id = f"{group_name}_{subject_folder}"

                all_features.append(features)
                all_labels.append(y)
                all_groups.extend([participant_id] * len(y))
                all_file_paths.extend([str(path)] * len(y))
                all_trial_numbers.extend(range(1, len(y) + 1))
                all_participant_groups.extend([group_name] * len(y))

                inventory.append({
                    "file": str(path),
                    "group": group_name,
                    "participant_id": participant_id,
                    "status": "PASS",
                    "trials": int(len(y)),
                    "channels": int(X.shape[1]),
                    "samples_per_epoch": int(X.shape[0]),
                    "sampling_frequency_hz": fs,
                    "window_samples": window_samples,
                    "feature_count": int(features.shape[1]),
                })

            except Exception as exc:
                inventory.append({
                    "file": str(path),
                    "group": group_name,
                    "participant_id": path.parent.name,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  SKIPPED: {path.name}: {exc}")

            if file_index % 10 == 0 or file_index == len(paths):
                print(f"  Processed {file_index}/{len(paths)}")

    inventory_path = OUTPUT_DIR / "training_file_inventory.csv"

    if inventory:
        fieldnames = sorted({
            key
            for record in inventory
            for key in record.keys()
        })

        with inventory_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory)

    if not all_features:
        raise RuntimeError("No valid EEG trials were loaded.")

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    groups_all = np.asarray(all_groups)
    files_all = np.asarray(all_file_paths)
    trials_all = np.asarray(all_trial_numbers)
    participant_groups_all = np.asarray(all_participant_groups)

    return (
        X_all,
        y_all,
        groups_all,
        files_all,
        trials_all,
        participant_groups_all,
        inventory,
    )


# ============================================================
# 5. LEAVE-ONE-PARTICIPANT-OUT MODEL EVALUATION
# ============================================================

def train_and_evaluate(
    X,
    y,
    groups,
    files,
    trial_numbers,
    participant_group_labels,
):
    logo = LeaveOneGroupOut()

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            ),
        ),
    ])

    predictions = np.full(len(y), -1, dtype=int)
    fold_results = []

    unique_participants = np.unique(groups)

    print(f"\nTotal trials: {len(y)}")
    print(f"Total participants: {len(unique_participants)}")
    print(f"Feature shape: {X.shape}")
    print("\nStarting leave-one-participant-out evaluation...")

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups),
        start=1,
    ):
        held_out_participant = groups[test_idx][0]

        # Explicit leakage safeguard.
        train_participants = set(groups[train_idx])
        test_participants = set(groups[test_idx])

        if train_participants.intersection(test_participants):
            raise RuntimeError(
                f"Participant leakage detected in fold {fold}."
            )

        if len(np.unique(y[train_idx])) < 2:
            raise RuntimeError(
                f"Training fold {fold} contains fewer than two classes."
            )

        model.fit(X[train_idx], y[train_idx])
        fold_predictions = model.predict(X[test_idx])
        predictions[test_idx] = fold_predictions

        fold_results.append({
            "fold": fold,
            "held_out_participant": held_out_participant,
            "held_out_group": participant_group_labels[test_idx][0],
            "train_trials": int(len(train_idx)),
            "test_trials": int(len(test_idx)),
            "accuracy": float(
                accuracy_score(y[test_idx], fold_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[test_idx], fold_predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y[test_idx],
                    fold_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        })

        print(
            f"Fold {fold:02d}/{len(unique_participants)} | "
            f"{held_out_participant} | "
            f"accuracy={fold_results[-1]['accuracy']:.4f}"
        )

    if np.any(predictions < 0):
        raise RuntimeError("Some trials did not receive a test prediction.")

    # Pooled out-of-fold metrics.
    cm = confusion_matrix(y, predictions, labels=[0, 1])

    summary = {
        "experiment": "130",
        "dataset": str(DATASET_ROOT),
        "task": "Left-versus-Right motor imagery",
        "feature_method": "Channel-wise log variance",
        "filter": {
            "type": "Butterworth band-pass",
            "order": FILTER_ORDER,
            "low_hz": FILTER_LOW_HZ,
            "high_hz": FILTER_HIGH_HZ,
            "implementation": "scipy.signal.sosfiltfilt",
        },
        "candidate_window_epoch_relative_seconds": [
            WINDOW_START_SECONDS,
            WINDOW_END_SECONDS,
        ],
        "window_timing_independently_verified": False,
        "classifier": {
            "type": "RBF-SVM",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
        },
        "validation": "Leave-one-participant-out",
        "participant_count": int(len(np.unique(groups))),
        "trial_count": int(len(y)),
        "feature_count": int(X.shape[1]),
        "class_counts": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "pooled_accuracy": float(accuracy_score(y, predictions)),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "pooled_macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix_labels_0_left_1_right": cm.tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Left", "Right"],
            output_dict=True,
            zero_division=0,
        ),
        "mean_participant_accuracy": float(
            np.mean([f["accuracy"] for f in fold_results])
        ),
        "mean_participant_balanced_accuracy": float(
            np.mean([f["balanced_accuracy"] for f in fold_results])
        ),
        "mean_participant_macro_f1": float(
            np.mean([f["macro_f1"] for f in fold_results])
        ),
        "fold_results": fold_results,
        "original_files_modified": False,
        "limitations": [
            "The selected window's physiological timing is not independently verified.",
            "This is a baseline experiment, not evidence of real-time BCI performance.",
            "Participant groups may differ in ways that affect generalization.",
            "Hyperparameters were not tuned on an independent validation dataset.",
        ],
    }

    # Save every out-of-fold prediction.
    prediction_path = OUTPUT_DIR / "out_of_fold_predictions.csv"

    with prediction_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "participant_id",
            "participant_group",
            "trial_number",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "correct",
        ])

        for i in range(len(y)):
            writer.writerow([
                files[i],
                groups[i],
                participant_group_labels[i],
                int(trial_numbers[i]),
                int(y[i]),
                LABEL_NAMES[int(y[i])],
                int(predictions[i]),
                LABEL_NAMES[int(predictions[i])],
                bool(y[i] == predictions[i]),
            ])

    fold_path = OUTPUT_DIR / "participant_fold_metrics.csv"

    with fold_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "held_out_participant",
                "held_out_group",
                "train_trials",
                "test_trials",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(fold_results)

    summary_path = OUTPUT_DIR / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# 6. MAIN
# ============================================================

def main():
    start_time = time.time()

    print("=" * 72)
    print("EXPERIMENT 130: DATASET 3 EEG BASELINE")
    print("=" * 72)
    print("Original dataset:", DATASET_ROOT)
    print("Output directory:", OUTPUT_DIR)
    print("Original files will not be modified.\n")

    (
        X,
        y,
        groups,
        files,
        trial_numbers,
        participant_group_labels,
        inventory,
    ) = collect_dataset()

    failed_files = sum(
        item["status"] == "FAILED"
        for item in inventory
    )

    print("\n=== DATA COLLECTION SUMMARY ===")
    print("Valid trials:", len(y))
    print("Unique participants:", len(np.unique(groups)))
    print("Feature dimensions:", X.shape)
    print("Failed files:", failed_files)

    if failed_files:
        print(
            "WARNING: Some files were skipped. Review "
            "training_file_inventory.csv before reporting results."
        )

    summary = train_and_evaluate(
        X,
        y,
        groups,
        files,
        trial_numbers,
        participant_group_labels,
    )

    summary["failed_file_count"] = int(failed_files)
    summary["elapsed_seconds"] = round(time.time() - start_time, 2)

    with (OUTPUT_DIR / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("EXPERIMENT 130 RESULTS")
    print("=" * 72)
    print(f"Participants: {summary['participant_count']}")
    print(f"Trials: {summary['trial_count']}")
    print(f"Features per trial: {summary['feature_count']}")
    print(f"Pooled accuracy: {summary['pooled_accuracy']:.4f}")
    print(
        "Pooled balanced accuracy: "
        f"{summary['pooled_balanced_accuracy']:.4f}"
    )
    print(f"Pooled macro F1: {summary['pooled_macro_f1']:.4f}")
    print(
        "Mean participant accuracy: "
        f"{summary['mean_participant_accuracy']:.4f}"
    )
    print(
        "Mean participant balanced accuracy: "
        f"{summary['mean_participant_balanced_accuracy']:.4f}"
    )
    print(
        "Mean participant macro F1: "
        f"{summary['mean_participant_macro_f1']:.4f}"
    )
    print("Elapsed seconds:", summary["elapsed_seconds"])

    print("\nReports saved under:")
    print(OUTPUT_DIR)
    print("- training_file_inventory.csv")
    print("- participant_fold_metrics.csv")
    print("- out_of_fold_predictions.csv")
    print("- training_summary.json")

    print("\nOriginal dataset files were read only.")


if __name__ == "__main__":
    main()
