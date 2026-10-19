
"""
EEG-SP-01 — RECTANGULAR VS HANN VALIDATION COMPARISON

Random Forest only.
Uses existing train/validation splits.
Never loads or evaluates test indices.
Never overwrites existing outputs.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features"
SPLITS = ROOT / "data" / "splits"

RESULTS_DIR = (
    ROOT / "results" / "eeg" / "signal_processing"
    / "EEG_SP_01_window_validation"
)

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]

SEED = 42
N_TREES = 300
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
CLASS_WEIGHT = None


def get_split_subject(subject):
    return "sub01" if subject == "sub-01" else subject


def load_labels(subject):
    if subject == "sub-01":
        path = (
            FEATURES / "temporal_spectral_sub01"
            / "sub-01_spectral_metadata.csv"
        )
        column = "label"
    else:
        path = (
            FEATURES / f"temporal_eeg_{subject}"
            / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
        )
        column = "label_id"

    if not path.exists():
        raise FileNotFoundError(path)

    metadata = pd.read_csv(path)

    if column not in metadata.columns:
        raise ValueError(f"{subject}: missing label column {column}")

    labels = metadata[column].to_numpy()

    if len(labels) != 720:
        raise ValueError(f"{subject}: expected 720 labels")

    if not np.array_equal(np.unique(labels), np.arange(8)):
        raise ValueError(f"{subject}: expected labels 0-7")

    return labels


def feature_path(subject, condition):
    if condition == "rectangular":
        directory = f"controlled_rect_spectral_{subject}"
    else:
        directory = f"controlled_hann_spectral_{subject}"

    return (
        FEATURES / directory
        / f"{subject}_{condition}_band_power.npy"
    )


def evaluate_condition(subject, condition, labels, train_idx, val_idx):
    path = feature_path(subject, condition)

    if not path.exists():
        raise FileNotFoundError(path)

    X = np.load(path, mmap_mode="r")

    if X.shape != (720, 64, 5):
        raise ValueError(f"{subject}/{condition}: bad shape {X.shape}")

    # Flatten channel-band features: 64 * 5 = 320.
    X_train = np.asarray(X[train_idx], dtype=np.float32).reshape(
        len(train_idx), -1
    )
    X_val = np.asarray(X[val_idx], dtype=np.float32).reshape(
        len(val_idx), -1
    )

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError(f"{subject}/{condition}: NaN/Inf in features")

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = {
        "subject": subject,
        "condition": condition,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "n_estimators": N_TREES,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "class_weight": CLASS_WEIGHT,
        "random_state": SEED,
    }

    del X, X_train, X_val, y_train, y_val, model, predictions
    gc.collect()

    return metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "validation_comparison.csv"
    summary_file = RESULTS_DIR / "validation_summary.csv"

    for path in (output_file, summary_file):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    rows = []

    for subject in SUBJECTS:
        print(f"\n{'=' * 65}\n{subject}\n{'=' * 65}")

        labels = load_labels(subject)
        split_dir = SPLITS / get_split_subject(subject)

        train_idx = np.load(split_dir / "train_indices.npy")
        val_idx = np.load(split_dir / "val_indices.npy")

        # Deliberately do not load test_indices.npy.
        if len(train_idx) != 504 or len(val_idx) != 108:
            raise ValueError(f"{subject}: unexpected train/validation sizes")

        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"{subject}: train/validation overlap")

        for condition in ("rectangular", "hann"):
            result = evaluate_condition(
                subject, condition, labels, train_idx, val_idx
            )
            rows.append(result)

            print(
                f"{condition:12s} | "
                f"accuracy={result['accuracy']:.4f} | "
                f"balanced_accuracy={result['balanced_accuracy']:.4f} | "
                f"macro_f1={result['macro_f1']:.4f}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(output_file, index=False)

    summary_rows = []

    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        grouped = results.groupby("condition")[metric]

        for condition in ("rectangular", "hann"):
            values = grouped.get_group(condition)
            summary_rows.append({
                "metric": metric,
                "condition": condition,
                "mean_across_subjects": values.mean(),
                "std_across_subjects": values.std(ddof=1),
                "subjects": len(values),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False)

    print(f"\n{'=' * 65}")
    print("VALIDATION SUMMARY — MEAN ACROSS 12 SUBJECTS")
    print(f"{'=' * 65}")

    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        print(f"\n{metric}:")
        subset = summary[summary["metric"] == metric]

        for _, row in subset.iterrows():
            print(
                f"  {row['condition']:12s}: "
                f"mean={row['mean_across_subjects']:.4f}, "
                f"std={row['std_across_subjects']:.4f}"
            )

    print(f"\nDetailed results: {output_file}")
    print(f"Summary: {summary_file}")
    print("\n✓ Validation comparison completed. Test data was not evaluated.")


if __name__ == "__main__":
    main()
