
"""
Experiment 132: Compare source-informed EEG time windows.

Compares RBF-SVM log-variance features using leave-one-participant-out
validation. Original dataset is read-only.

Windows are relative to epoch start:
  10-13 s: cue to 3 s after cue
  13-16 s: 3-6 s after cue
  16-19 s: 6-9 s after cue
  19-22 s: 9-12 s after cue
  13-22 s: 3-12 s after cue (source-informed candidate)

These are exploratory comparisons, not proof of physiological timing.
"""

from pathlib import Path
import csv
import json
import time

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
)

ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATASET = ROOT / "major datasets" / "28955456"
OUT = ROOT / "experiments" / "dataset3_28955456_training" / "experiment132_time_windows"
OUT.mkdir(parents=True, exist_ok=True)

FS = 256
LOW, HIGH = 8.0, 30.0

WINDOWS = {
    "cue_to_3s": (10, 13),
    "3_to_6s_after_cue": (13, 16),
    "6_to_9s_after_cue": (16, 19),
    "9_to_12s_after_cue": (19, 22),
    "3_to_12s_after_cue": (13, 22),
}


def load_file(path):
    mat = loadmat(str(path), squeeze_me=False, struct_as_record=True)
    epo = mat["epo"][0, 0]

    X = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1).astype(int)
    fs = float(np.asarray(epo["fs"]).reshape(-1)[0])

    if X.ndim != 3 or X.shape[1] != 32 or X.shape[2] != 30:
        raise ValueError(f"Unexpected EEG shape: {X.shape}")
    if fs != FS:
        raise ValueError(f"Unexpected sampling frequency: {fs}")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid EEG values or labels.")

    return X, y


def main():
    started = time.time()
    sos = butter(4, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")

    # Filter each complete epoch once, then compare windows.
    records = []
    all_windows = {name: [] for name in WINDOWS}
    all_labels = []
    all_subjects = []

    for group in ("Normals", "Patients"):
        folder = DATASET / "epo_data" / group / "EEG"
        paths = sorted(folder.rglob("*_epo.mat")) if folder.exists() else []
        print(f"{group}: {len(paths)} files")

        for i, path in enumerate(paths, 1):
            try:
                X, y = load_file(path)
                filtered = sosfiltfilt(sos, X, axis=0)
                subject = f"{group}_{path.parent.name}"

                for name, (start_s, end_s) in WINDOWS.items():
                    # Match MATLAB 1-based inclusive indexing.
                    start = int(round(start_s * FS)) - 1
                    end = int(round(end_s * FS))
                    segment = filtered[start:end, :, :]

                    # samples x channels x trials -> trials x channels
                    variance = np.var(segment, axis=0, ddof=1).T
                    features = np.log(np.maximum(variance, 1e-20))
                    all_windows[name].append(features)

                all_labels.append(y)
                all_subjects.extend([subject] * len(y))
                records.append({"file": str(path), "subject": subject, "status": "PASS"})

            except Exception as exc:
                records.append({
                    "file": str(path),
                    "subject": f"{group}_{path.parent.name}",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"Skipped {path.name}: {exc}")

            if i % 10 == 0 or i == len(paths):
                print(f"  Processed {i}/{len(paths)}")

    if not all_labels:
        raise RuntimeError("No valid trials loaded.")

    y = np.concatenate(all_labels)
    subjects = np.asarray(all_subjects)

    results = []

    for window_name, feature_blocks in all_windows.items():
        X_features = np.vstack(feature_blocks)
        predictions = np.full(len(y), -1, dtype=int)
        fold_scores = []

        cv = LeaveOneGroupOut()
        print(f"\nEvaluating window: {window_name} {X_features.shape}")

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X_features, y, groups=subjects), start=1
        ):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                )),
            ])

            model.fit(X_features[train_idx], y[train_idx])
            pred = model.predict(X_features[test_idx])
            predictions[test_idx] = pred

            fold_scores.append({
                "subject": subjects[test_idx][0],
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test_idx], pred)
                ),
                "macro_f1": float(
                    f1_score(y[test_idx], pred, average="macro", zero_division=0)
                ),
            })

        result = {
            "window": window_name,
            "epoch_relative_start_s": WINDOWS[window_name][0],
            "epoch_relative_end_s": WINDOWS[window_name][1],
            "trials": int(len(y)),
            "participants": int(len(np.unique(subjects))),
            "pooled_accuracy": float(accuracy_score(y, predictions)),
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(y, predictions)
            ),
            "pooled_macro_f1": float(
                f1_score(y, predictions, average="macro", zero_division=0)
            ),
            "mean_participant_accuracy": float(
                np.mean([r["accuracy"] for r in fold_scores])
            ),
            "mean_participant_balanced_accuracy": float(
                np.mean([r["balanced_accuracy"] for r in fold_scores])
            ),
            "mean_participant_macro_f1": float(
                np.mean([r["macro_f1"] for r in fold_scores])
            ),
            "fold_results": fold_scores,
        }
        results.append(result)

        print(
            f"{window_name}: accuracy={result['pooled_accuracy']:.4f}, "
            f"balanced_accuracy={result['pooled_balanced_accuracy']:.4f}, "
            f"macro_f1={result['pooled_macro_f1']:.4f}"
        )

    results.sort(key=lambda r: r["pooled_balanced_accuracy"], reverse=True)

    summary = {
        "experiment": 132,
        "task": "Left-versus-Right motor imagery",
        "model": "RBF-SVM with standardized channel-wise log variance",
        "filter_band_hz": [LOW, HIGH],
        "validation": "Leave-one-participant-out",
        "window_timing_status": (
            "Windows are source-informed; physiological timing is not independently verified."
        ),
        "failed_file_count": sum(r["status"] == "FAILED" for r in records),
        "ranked_results": results,
        "elapsed_seconds": round(time.time() - started, 2),
        "original_files_modified": False,
    }

    with (OUT / "window_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (OUT / "window_comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "window", "epoch_relative_start_s", "epoch_relative_end_s",
            "trials", "participants", "pooled_accuracy",
            "pooled_balanced_accuracy", "pooled_macro_f1",
            "mean_participant_accuracy", "mean_participant_balanced_accuracy",
            "mean_participant_macro_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})

    with (OUT / "file_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = sorted({k for r in records for k in r})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("\n=== EXPERIMENT 132 RANKED RESULTS ===")
    for rank, r in enumerate(results, 1):
        print(
            f"{rank}. {r['window']}: "
            f"accuracy={r['pooled_accuracy']:.4f}, "
            f"balanced_accuracy={r['pooled_balanced_accuracy']:.4f}, "
            f"macro_f1={r['pooled_macro_f1']:.4f}"
        )

    print("\nReports saved to:", OUT)
    print("Original dataset files were read only.")


if __name__ == "__main__":
    main()
