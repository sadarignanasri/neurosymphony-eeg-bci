
"""
Experiment 131: Dataset 3 EEG motor-imagery classification.

Model: CSP + LDA
Validation: Leave-one-participant-out
Original dataset: Read-only
Outputs: experiments/dataset3_28955456_training/experiment131_csp_lda/

NOTE:
The 13-22 second epoch-relative window is still a candidate window.
Its physiological timing has not yet been independently verified.
"""

from pathlib import Path
import csv
import json
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from mne.decoding import CSP
except ImportError as exc:
    raise SystemExit(
        "Missing MNE. Install it using: python -m pip install mne"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATASET = ROOT / "major datasets" / "28955456"

OUT = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment131_csp_lda"
)
OUT.mkdir(parents=True, exist_ok=True)

FS = 256
CHANNELS = 32
TRIALS_PER_FILE = 30

LOW_HZ = 8.0
HIGH_HZ = 30.0
FILTER_ORDER = 4

WINDOW_START = 13
WINDOW_END = 22

N_CSP_COMPONENTS = 6
LABEL_NAMES = {0: "Left", 1: "Right"}


# ============================================================
# LOAD AND VALIDATE FILE
# ============================================================

def load_file(path):
    mat = loadmat(
        str(path),
        squeeze_me=False,
        struct_as_record=True,
    )

    if "epo" not in mat or mat["epo"].size != 1:
        raise ValueError("Missing or invalid epo structure.")

    epo = mat["epo"][0, 0]

    X = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1).astype(int)
    fs = float(np.asarray(epo["fs"]).reshape(-1)[0])

    if X.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {X.shape}")

    if X.shape[1] != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels, got {X.shape[1]}")

    if X.shape[2] != TRIALS_PER_FILE:
        raise ValueError(f"Expected {TRIALS_PER_FILE} trials, got {X.shape[2]}")

    if y.size != X.shape[2]:
        raise ValueError("Trial count and label count differ.")

    if not np.isclose(fs, FS):
        raise ValueError(f"Unexpected sampling rate: {fs}")

    if not np.isfinite(X).all():
        raise ValueError("EEG contains non-finite values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Unexpected labels: {np.unique(y)}")

    unique, counts = np.unique(y, return_counts=True)
    if dict(zip(unique.tolist(), counts.tolist())) != {0: 15, 1: 15}:
        raise ValueError("Expected 15 trials per class.")

    return X, y, fs


# ============================================================
# FILTER AND SELECT CANDIDATE WINDOW
# ============================================================

def preprocess(X, fs):
    sos = butter(
        FILTER_ORDER,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    # X: samples x channels x trials
    filtered = sosfiltfilt(sos, X, axis=0)

    # Convert MATLAB's 1-based inclusive indices to Python slicing.
    start = int(round(fs * WINDOW_START)) - 1
    end = int(round(fs * WINDOW_END))

    if start < 0 or end > filtered.shape[0] or start >= end:
        raise ValueError("Candidate window is outside the epoch.")

    window = filtered[start:end, :, :]

    # MNE CSP input: trials x channels x samples.
    return np.transpose(window, (2, 1, 0))


# ============================================================
# LOAD ALL PARTICIPANTS
# ============================================================

def collect_data():
    all_X = []
    all_y = []
    all_subjects = []
    all_groups = []
    all_files = []
    all_trial_numbers = []
    inventory = []

    for group in ("Normals", "Patients"):
        folder = DATASET / "epo_data" / group / "EEG"
        paths = sorted(folder.rglob("*_epo.mat")) if folder.exists() else []

        print(f"{group}: {len(paths)} files found")

        for idx, path in enumerate(paths, start=1):
            try:
                X, y, fs = load_file(path)
                X_window = preprocess(X, fs)

                subject_id = f"{group}_{path.parent.name}"

                all_X.append(X_window)
                all_y.append(y)
                all_subjects.extend([subject_id] * len(y))
                all_groups.extend([group] * len(y))
                all_files.extend([str(path)] * len(y))
                all_trial_numbers.extend(range(1, len(y) + 1))

                inventory.append({
                    "file": str(path),
                    "group": group,
                    "subject_id": subject_id,
                    "status": "PASS",
                    "trials": len(y),
                })

            except Exception as exc:
                inventory.append({
                    "file": str(path),
                    "group": group,
                    "subject_id": f"{group}_{path.parent.name}",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  SKIPPED {path.name}: {exc}")

            if idx % 10 == 0 or idx == len(paths):
                print(f"  Processed {idx}/{len(paths)}")

    if not all_X:
        raise RuntimeError("No valid trials loaded.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    subjects = np.asarray(all_subjects)
    groups = np.asarray(all_groups)
    files = np.asarray(all_files)
    trial_numbers = np.asarray(all_trial_numbers)

    with (OUT / "file_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = sorted({k for row in inventory for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)

    return X, y, subjects, groups, files, trial_numbers, inventory


# ============================================================
# LEAVE-ONE-SUBJECT-OUT EVALUATION
# ============================================================

def evaluate(X, y, subjects, groups, files, trial_numbers, inventory):
    logo = LeaveOneGroupOut()
    predictions = np.full(len(y), -1, dtype=int)
    fold_results = []

    unique_subjects = np.unique(subjects)

    print(f"\nTrials: {len(y)}")
    print(f"Participants: {len(unique_subjects)}")
    print(f"EEG window tensor: {X.shape}")
    print("\nTraining CSP + LDA...")

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=subjects), start=1
    ):
        # Fit CSP only on the training participants.
        csp = CSP(
            n_components=N_CSP_COMPONENTS,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
            transform_into="average_power",
        )

        # LDA receives CSP features. Standardization is fitted on
        # training data only, within the pipeline.
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

        X_train_csp = csp.fit_transform(X[train_idx], y[train_idx])
        X_test_csp = csp.transform(X[test_idx])

        classifier.fit(X_train_csp, y[train_idx])
        pred = classifier.predict(X_test_csp)
        predictions[test_idx] = pred

        fold_results.append({
            "fold": fold,
            "held_out_subject": subjects[test_idx][0],
            "group": groups[test_idx][0],
            "train_trials": int(len(train_idx)),
            "test_trials": int(len(test_idx)),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[test_idx], pred)
            ),
            "macro_f1": float(
                f1_score(y[test_idx], pred, average="macro", zero_division=0)
            ),
        })

        print(
            f"Fold {fold:02d}/{len(unique_subjects)} "
            f"{subjects[test_idx][0]} | "
            f"accuracy={fold_results[-1]['accuracy']:.4f}"
        )

    if np.any(predictions < 0):
        raise RuntimeError("Some trials were not evaluated.")

    cm = confusion_matrix(y, predictions, labels=[0, 1])

    summary = {
        "experiment": 131,
        "model": "CSP + LDA",
        "task": "Left versus Right motor imagery",
        "dataset": str(DATASET),
        "participants": int(len(unique_subjects)),
        "trials": int(len(y)),
        "channels": CHANNELS,
        "sampling_frequency_hz": FS,
        "filter_band_hz": [LOW_HZ, HIGH_HZ],
        "candidate_window_epoch_relative_seconds": [WINDOW_START, WINDOW_END],
        "window_timing_independently_verified": False,
        "csp_components": N_CSP_COMPONENTS,
        "validation": "Leave-one-participant-out",
        "class_counts": {
            LABEL_NAMES[int(k)]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "pooled_accuracy": float(accuracy_score(y, predictions)),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "pooled_macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix_labels_left_right": cm.tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Left", "Right"],
            output_dict=True,
            zero_division=0,
        ),
        "mean_participant_accuracy": float(
            np.mean([r["accuracy"] for r in fold_results])
        ),
        "mean_participant_balanced_accuracy": float(
            np.mean([r["balanced_accuracy"] for r in fold_results])
        ),
        "mean_participant_macro_f1": float(
            np.mean([r["macro_f1"] for r in fold_results])
        ),
        "fold_results": fold_results,
        "failed_files": sum(r["status"] == "FAILED" for r in inventory),
        "original_files_modified": False,
    }

    with (OUT / "participant_fold_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "fold", "held_out_subject", "group", "train_trials",
            "test_trials", "accuracy", "balanced_accuracy", "macro_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fold_results)

    with (OUT / "out_of_fold_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "subject_id", "group", "trial_number",
            "true_label", "predicted_label", "correct",
        ])
        for i in range(len(y)):
            writer.writerow([
                files[i], subjects[i], groups[i], int(trial_numbers[i]),
                int(y[i]), int(predictions[i]), bool(y[i] == predictions[i]),
            ])

    with (OUT / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    start_time = time.time()

    print("=" * 72)
    print("EXPERIMENT 131: CSP + LDA")
    print("=" * 72)
    print("Original dataset files will not be modified.")
    print("Output:", OUT)

    X, y, subjects, groups, files, trial_numbers, inventory = collect_data()

    print("\n=== DATA COLLECTION ===")
    print("Trials:", len(y))
    print("Participants:", len(np.unique(subjects)))
    print("Tensor:", X.shape)
    print("Failed files:", sum(r["status"] == "FAILED" for r in inventory))

    summary = evaluate(X, y, subjects, groups, files, trial_numbers, inventory)
    summary["elapsed_seconds"] = round(time.time() - start_time, 2)

    with (OUT / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("EXPERIMENT 131 RESULTS")
    print("=" * 72)
    print("Pooled accuracy:", f"{summary['pooled_accuracy']:.4f}")
    print("Balanced accuracy:", f"{summary['pooled_balanced_accuracy']:.4f}")
    print("Macro F1:", f"{summary['pooled_macro_f1']:.4f}")
    print("Mean participant accuracy:", f"{summary['mean_participant_accuracy']:.4f}")
    print("Mean participant balanced accuracy:", f"{summary['mean_participant_balanced_accuracy']:.4f}")
    print("Mean participant macro F1:", f"{summary['mean_participant_macro_f1']:.4f}")
    print("Elapsed seconds:", summary["elapsed_seconds"])
    print("Reports saved to:", OUT)
    print("Original dataset files were read only.")


if __name__ == "__main__":
    main()
