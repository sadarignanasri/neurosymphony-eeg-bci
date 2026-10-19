
"""Sanity-check original Step 33 vs controlled rectangular features."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features"
SPLITS = ROOT / "data" / "splits"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]


def load_labels(subject):
    if subject == "sub-01":
        path = FEATURES / "temporal_spectral_sub01" / "sub-01_spectral_metadata.csv"
        column = "label"
    else:
        path = (
            FEATURES / f"temporal_eeg_{subject}"
            / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
        )
        column = "label_id"

    df = pd.read_csv(path)
    labels = df[column].to_numpy()

    if len(labels) != 720 or not np.array_equal(np.unique(labels), np.arange(8)):
        raise ValueError(f"{subject}: label validation failed")

    return labels


def evaluate(X, labels, train_idx, val_idx):
    X_train = np.asarray(X[train_idx], dtype=np.float32).reshape(len(train_idx), -1)
    X_val = np.asarray(X[val_idx], dtype=np.float32).reshape(len(val_idx), -1)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, labels[train_idx])
    pred = model.predict(X_val)

    return (
        accuracy_score(labels[val_idx], pred),
        balanced_accuracy_score(labels[val_idx], pred),
        f1_score(labels[val_idx], pred, average="macro", zero_division=0),
    )


def main():
    rows = []

    for subject in SUBJECTS:
        split_subject = "sub01" if subject == "sub-01" else subject
        split_dir = SPLITS / split_subject

        train_idx = np.load(split_dir / "train_indices.npy")
        val_idx = np.load(split_dir / "val_indices.npy")
        labels = load_labels(subject)

        original_file = (
            FEATURES / f"temporal_spectral_{subject}"
            / f"{subject}_band_power.npy"
        )
        if subject == "sub-01":
            original_file = FEATURES / "temporal_spectral_sub01" / "sub-01_band_power.npy"

        rectangular_file = (
            FEATURES / f"controlled_rect_spectral_{subject}"
            / f"{subject}_rectangular_band_power.npy"
        )

        for name, path in (
            ("original_step33", original_file),
            ("controlled_rectangular", rectangular_file),
        ):
            if not path.exists():
                raise FileNotFoundError(path)

            X = np.load(path, mmap_mode="r")
            if X.shape != (720, 64, 5):
                raise ValueError(f"{subject}/{name}: invalid shape {X.shape}")

            metrics = evaluate(X, labels, train_idx, val_idx)
            rows.append({
                "subject": subject,
                "condition": name,
                "accuracy": metrics[0],
                "balanced_accuracy": metrics[1],
                "macro_f1": metrics[2],
            })

            print(
                f"{subject} | {name:24s} | "
                f"accuracy={metrics[0]:.4f} | "
                f"balanced_accuracy={metrics[1]:.4f} | "
                f"macro_f1={metrics[2]:.4f}"
            )
            del X

    results = pd.DataFrame(rows)
    output_dir = ROOT / "results" / "eeg" / "signal_processing" / "EEG_SP_01_window_validation"
    output_file = output_dir / "baseline_sanity_check.csv"

    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {output_file}")

    results.to_csv(output_file, index=False)

    print("\nMEAN VALIDATION METRICS ACROSS SUBJECTS")
    print(results.groupby("condition")[["accuracy", "balanced_accuracy", "macro_f1"]].mean())
    print(f"\nSaved: {output_file}")
    print("Test data was not evaluated.")


if __name__ == "__main__":
    main()
