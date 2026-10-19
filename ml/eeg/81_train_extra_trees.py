
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


FEATURE_ROOT = Path("data/features")
SPLIT_ROOT = Path("data/splits")
OUTPUT_ROOT = Path("results/eeg/advanced_models/step81_extra_trees")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLASSES = 8


def locate_files(subject_num):
    sid = f"{subject_num:02d}"

    candidates = [
        (
            FEATURE_ROOT / f"temporal_spectral_sub{sid}"
            / f"sub-{sid}_band_power.npy",
            FEATURE_ROOT / f"temporal_spectral_sub{sid}"
            / f"sub-{sid}_spectral_metadata.csv",
            "label",
        ),
        (
            FEATURE_ROOT / f"temporal_spectral_sub-{sid}"
            / f"sub-{sid}_band_power.npy",
            FEATURE_ROOT / f"temporal_eeg_sub-{sid}"
            / f"sub-{sid}_temporal_eeg_preprocessed_metadata.csv",
            "label_id",
        ),
    ]

    for feature_path, metadata_path, label_column in candidates:
        if feature_path.exists() and metadata_path.exists():
            return feature_path, metadata_path, label_column

    raise FileNotFoundError(
        f"Could not locate feature and label files for subject {sid}."
    )


def load_subject(subject_num):
    feature_path, metadata_path, label_column = locate_files(subject_num)

    X = np.load(feature_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)

    if label_column not in metadata.columns:
        raise ValueError(
            f"{metadata_path} has no '{label_column}' column. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    y = metadata[label_column].to_numpy(dtype=np.int64)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D features, received {X.shape}")

    if X.shape[0] != len(y):
        raise ValueError(
            f"Feature/label mismatch for subject {subject_num}: "
            f"{X.shape[0]} features versus {len(y)} labels"
        )

    # Convert (trials, channels, bands) into (trials, features).
    X = X.reshape(X.shape[0], -1)

    # Normalize class IDs to 0–7 for consistent evaluation.
    if set(np.unique(y)) == set(range(1, N_CLASSES + 1)):
        y = y - 1

    if set(np.unique(y)) != set(range(N_CLASSES)):
        raise ValueError(
            f"Unexpected labels for subject {subject_num}: "
            f"{np.unique(y).tolist()}"
        )

    return X, y


def load_indices(subject_num):
    sid = f"{subject_num:02d}"
    split_dir = SPLIT_ROOT / f"sub-{sid}"

    if subject_num == 1:
        split_dir = SPLIT_ROOT / "sub01"

    files = {
        name: split_dir / f"{name}_indices.npy"
        for name in ("train", "val", "test")
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split files:\n" + "\n".join(missing)
        )

    return {
        name: np.load(path).astype(int)
        for name, path in files.items()
    }


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def main():
    subject_results = []
    all_predictions = []

    for subject_num in range(1, 13):
        print(f"\n{'=' * 60}")
        print(f"STEP 81 — EXTRA TREES — SUBJECT {subject_num:02d}")
        print("=" * 60)

        X, y = load_subject(subject_num)
        splits = load_indices(subject_num)

        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]

        # Confirm indices are valid and mutually exclusive.
        for name, indices in splits.items():
            if len(indices) == 0:
                raise ValueError(f"Empty {name} split for subject {subject_num}")
            if indices.min() < 0 or indices.max() >= len(y):
                raise ValueError(f"Invalid {name} indices for subject {subject_num}")

        if (
            set(train_idx) & set(val_idx)
            or set(train_idx) & set(test_idx)
            or set(val_idx) & set(test_idx)
        ):
            raise ValueError(f"Overlapping splits for subject {subject_num}")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ])

        model.fit(X[train_idx], y[train_idx])

        for split_name, indices in (
            ("validation", val_idx),
            ("test", test_idx),
        ):
            predictions = model.predict(X[indices])
            scores = metrics(y[indices], predictions)

            print(
                f"{split_name.capitalize()} | "
                f"Accuracy: {scores['accuracy']:.4f} | "
                f"Balanced accuracy: {scores['balanced_accuracy']:.4f} | "
                f"Macro-F1: {scores['macro_f1']:.4f}"
            )

            subject_results.append({
                "subject": f"sub-{subject_num:02d}",
                "split": split_name,
                "train_trials": len(train_idx),
                "evaluation_trials": len(indices),
                **scores,
            })

            if split_name == "test":
                all_predictions.append(pd.DataFrame({
                    "subject": f"sub-{subject_num:02d}",
                    "trial_index": indices,
                    "true_label": y[indices],
                    "predicted_label": predictions,
                }))

    results = pd.DataFrame(subject_results)
    results.to_csv(OUTPUT_ROOT / "subject_results.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(
        OUTPUT_ROOT / "test_predictions.csv", index=False
    )

    aggregate_rows = []
    for split_name in ("validation", "test"):
        subset = results[results["split"] == split_name]

        for metric_name in (
            "accuracy", "balanced_accuracy", "macro_f1"
        ):
            aggregate_rows.append({
                "split": split_name,
                "metric": metric_name,
                "mean": subset[metric_name].mean(),
                "std": subset[metric_name].std(),
                "subjects": len(subset),
            })

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT_ROOT / "aggregate_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("STEP 81 COMPLETE")
    print("=" * 60)
    print(aggregate.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
