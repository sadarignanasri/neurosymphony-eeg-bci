
from pathlib import Path
import csv
import json
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATA = ROOT / "major datasets" / "28955456" / "epo_data"
OUT = ROOT / "experiments" / "dataset3_28955456_training" / "experiment133_features"
OUT.mkdir(parents=True, exist_ok=True)

FS = 256
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
WINDOW_START = 13  # seconds relative to epoch start = 3 s after cue
WINDOW_END = 22    # seconds relative to epoch start = 12 s after cue


def load_epoch(path):
    mat = loadmat(str(path), squeeze_me=False, struct_as_record=True)
    epo = mat["epo"][0, 0]
    X = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1).astype(int)
    fs = float(np.asarray(epo["fs"]).reshape(-1)[0])

    if fs != FS or X.shape != (9531, 32, 30):
        raise ValueError(f"Unexpected sampling rate/shape: {fs}, {X.shape}")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid signal or labels")
    return X, y


def make_features(segment):
    # segment: samples x channels x trials
    # Return trial x feature matrices.
    variance = np.var(segment, axis=0, ddof=1).T
    log_variance = np.log(np.maximum(variance, 1e-20))

    freqs, psd = welch(
        segment, fs=FS, nperseg=512, axis=0, detrend="constant"
    )
    # PSD shape: frequencies x channels x trials
    band_powers = []

    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() < 2:
            raise ValueError(f"Not enough PSD bins for band {low}-{high} Hz")
        power = np.trapezoid(psd[mask], freqs[mask], axis=0)
        band_powers.append(np.maximum(power.T, 1e-20))

    # trial x (channels * 5 bands)
    absolute = np.log(np.concatenate(band_powers, axis=1))
    powers = np.concatenate(band_powers, axis=1).reshape(
        len(segment[0, 0, :]), 5, segment.shape[1]
    )
    total = np.maximum(powers.sum(axis=1, keepdims=True), 1e-20)
    relative = (powers / total).reshape(len(segment[0, 0, :]), -1)

    return {
        "log_variance": log_variance,
        "log_band_power": absolute,
        "relative_band_power": relative,
        "combined": np.concatenate([log_variance, absolute, relative], axis=1),
    }


def main():
    started = time.time()
    sos = butter(4, [8, 30], btype="bandpass", fs=FS, output="sos")

    features = {name: [] for name in (
        "log_variance", "log_band_power", "relative_band_power", "combined"
    )}
    labels, groups, inventory = [], [], []

    for group in ("Normals", "Patients"):
        folder = DATA / group / "EEG"
        files = sorted(folder.rglob("*_epo.mat")) if folder.exists() else []
        print(f"{group}: {len(files)} files")

        for i, path in enumerate(files, 1):
            try:
                X, y = load_epoch(path)
                filtered = sosfiltfilt(sos, X, axis=0)

                start = WINDOW_START * FS - 1
                end = WINDOW_END * FS
                segment = filtered[start:end, :, :]

                extracted = make_features(segment)
                for name in features:
                    features[name].append(extracted[name])

                labels.append(y)
                participant = f"{group}_{path.parent.name}"
                groups.extend([participant] * len(y))
                inventory.append({
                    "file": str(path),
                    "participant": participant,
                    "status": "PASS",
                })
            except Exception as exc:
                inventory.append({
                    "file": str(path),
                    "participant": f"{group}_{path.parent.name}",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"Skipped {path}: {exc}")

            if i % 10 == 0 or i == len(files):
                print(f"  Processed {i}/{len(files)}")

    if not labels:
        raise RuntimeError("No valid epoch files were loaded.")

    y = np.concatenate(labels)
    groups = np.asarray(groups)
    cv = LeaveOneGroupOut()
    results = []

    models = {
        "RBF_SVM": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced"
            )),
        ]),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for feature_name, blocks in features.items():
        X = np.vstack(blocks)
        print(f"\nFeature family: {feature_name}; shape={X.shape}")

        for model_name, model_template in models.items():
            predictions = np.full(len(y), -1, dtype=int)
            fold_results = []

            for train, test in cv.split(X, y, groups=groups):
                # Create a fresh estimator for every held-out participant.
                from sklearn.base import clone
                model = clone(model_template)
                model.fit(X[train], y[train])
                pred = model.predict(X[test])
                predictions[test] = pred

                fold_results.append({
                    "participant": str(groups[test][0]),
                    "accuracy": float(accuracy_score(y[test], pred)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y[test], pred)
                    ),
                    "macro_f1": float(
                        f1_score(y[test], pred, average="macro", zero_division=0)
                    ),
                })

            if np.any(predictions < 0):
                raise RuntimeError("Some trials did not receive a prediction.")

            result = {
                "feature_family": feature_name,
                "model": model_name,
                "feature_count": int(X.shape[1]),
                "trials": int(len(y)),
                "participants": int(len(np.unique(groups))),
                "pooled_accuracy": float(accuracy_score(y, predictions)),
                "pooled_balanced_accuracy": float(
                    balanced_accuracy_score(y, predictions)
                ),
                "pooled_macro_f1": float(
                    f1_score(y, predictions, average="macro", zero_division=0)
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
            }
            results.append(result)
            print(
                f"{model_name}: accuracy={result['pooled_accuracy']:.4f}, "
                f"balanced_accuracy={result['pooled_balanced_accuracy']:.4f}, "
                f"macro_f1={result['pooled_macro_f1']:.4f}"
            )

    results.sort(key=lambda r: r["pooled_balanced_accuracy"], reverse=True)

    summary = {
        "experiment": 133,
        "dataset": "28955456",
        "window_epoch_relative_seconds": [WINDOW_START, WINDOW_END],
        "window_after_cue_seconds": [WINDOW_START - 10, WINDOW_END - 10],
        "filter_hz": [8, 30],
        "validation": "Leave-one-participant-out",
        "feature_families": list(features.keys()),
        "results_ranked_by_pooled_balanced_accuracy": results,
        "failed_file_count": sum(r["status"] == "FAILED" for r in inventory),
        "elapsed_seconds": round(time.time() - started, 2),
        "original_data_modified": False,
        "note": (
            "Exploratory comparison. The window was selected after earlier "
            "comparisons on these same participants, so results are not an "
            "independent final-test estimate."
        ),
    }

    with (OUT / "feature_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (OUT / "feature_comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "feature_family", "model", "feature_count", "trials", "participants",
            "pooled_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1",
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
        fields = sorted({k for row in inventory for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)

    print("\n" + "=" * 60)
    print("EXPERIMENT 133 RANKED RESULTS")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r['feature_family']} + {r['model']}: "
            f"accuracy={r['pooled_accuracy']:.4f}, "
            f"balanced_accuracy={r['pooled_balanced_accuracy']:.4f}, "
            f"macro_f1={r['pooled_macro_f1']:.4f}"
        )
    print(f"\nReports saved to: {OUT}")
    print("Original dataset files were read only.")


if __name__ == "__main__":
    main()
