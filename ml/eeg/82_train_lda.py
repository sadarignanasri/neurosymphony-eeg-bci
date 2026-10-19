
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 82 — LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================================

FEATURE_ROOT = Path("data/features")
SPLIT_ROOT = Path("data/splits")
OUTPUT_ROOT = Path("results/eeg/advanced_models/step82_lda")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

N_CLASSES = 8
N_FEATURES = 320


# ============================================================
# LOAD FEATURES AND LABELS
# ============================================================

def load_data(subject):
    sid = f"{subject:02d}"

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
            X = np.load(feature_path).astype(np.float32)

            if X.ndim != 3:
                raise ValueError(
                    f"Expected 3D features for subject {sid}; "
                    f"received shape {X.shape}"
                )

            X = X.reshape(X.shape[0], -1)

            if X.shape[1] != N_FEATURES:
                raise ValueError(
                    f"Expected {N_FEATURES} features for subject {sid}; "
                    f"received {X.shape[1]}"
                )

            metadata = pd.read_csv(metadata_path)

            if label_column not in metadata.columns:
                raise ValueError(
                    f"Missing '{label_column}' in {metadata_path}. "
                    f"Available columns: {metadata.columns.tolist()}"
                )

            y = metadata[label_column].to_numpy(dtype=np.int64)

            if len(X) != len(y):
                raise ValueError(
                    f"Feature/label mismatch for subject {sid}: "
                    f"{len(X)} features versus {len(y)} labels"
                )

            # Convert labels 1–8 to 0–7 if necessary.
            if set(np.unique(y)) == set(range(1, N_CLASSES + 1)):
                y = y - 1

            if set(np.unique(y)) != set(range(N_CLASSES)):
                raise ValueError(
                    f"Unexpected labels for subject {sid}: "
                    f"{np.unique(y).tolist()}"
                )

            if not np.isfinite(X).all():
                raise ValueError(
                    f"Non-finite feature values found for subject {sid}"
                )

            return X, y

    raise FileNotFoundError(
        f"Feature or metadata files not found for subject {sid}"
    )


# ============================================================
# LOAD SAVED TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def load_splits(subject):
    sid = f"{subject:02d}"

    split_dir = (
        SPLIT_ROOT / "sub01"
        if subject == 1
        else SPLIT_ROOT / f"sub-{sid}"
    )

    splits = {}

    for name in ("train", "val", "test"):
        path = split_dir / f"{name}_indices.npy"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}"
            )

        splits[name] = np.load(path).astype(int)

    return splits


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    results = []
    test_predictions = []

    for subject in range(1, 13):
        sid = f"sub-{subject:02d}"

        print("\n" + "=" * 60)
        print(f"STEP 82 — LDA — {sid}")
        print("=" * 60)

        X, y = load_data(subject)
        splits = load_splits(subject)

        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]

        # Validate indices.
        for name, indices in splits.items():
            if len(indices) == 0:
                raise ValueError(f"Empty {name} split for {sid}")

            if indices.min() < 0 or indices.max() >= len(y):
                raise ValueError(f"Invalid {name} indices for {sid}")

            if len(np.unique(indices)) != len(indices):
                raise ValueError(
                    f"Duplicate indices in {name} split for {sid}"
                )

        # Ensure the three splits do not overlap.
        if (
            set(train_idx) & set(val_idx)
            or set(train_idx) & set(test_idx)
            or set(val_idx) & set(test_idx)
        ):
            raise ValueError(f"Overlapping splits for {sid}")

        # Imputation and scaling are fitted only on training data.
        # LDA uses automatic covariance shrinkage.
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            ),
        )

        model.fit(X[train_idx], y[train_idx])

        print(
            f"Trials: train={len(train_idx)}, "
            f"validation={len(val_idx)}, "
            f"test={len(test_idx)}"
        )

        for split_name, indices in (
            ("validation", val_idx),
            ("test", test_idx),
        ):
            predictions = model.predict(X[indices])

            scores = calculate_metrics(
                y[indices],
                predictions,
            )

            results.append({
                "subject": sid,
                "split": split_name,
                "train_trials": len(train_idx),
                "evaluation_trials": len(indices),
                **scores,
            })

            print(
                f"{split_name.capitalize()} | "
                f"Accuracy: {scores['accuracy']:.4f} | "
                f"Balanced accuracy: "
                f"{scores['balanced_accuracy']:.4f} | "
                f"Macro-F1: {scores['macro_f1']:.4f}"
            )

            if split_name == "test":
                test_predictions.append(
                    pd.DataFrame({
                        "subject": sid,
                        "trial_index": indices,
                        "true_label": y[indices],
                        "predicted_label": predictions,
                    })
                )

    # Save per-subject results.
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_ROOT / "subject_results.csv",
        index=False,
    )

    # Save test predictions.
    predictions_df = pd.concat(
        test_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_ROOT / "test_predictions.csv",
        index=False,
    )

    # Aggregate means and standard deviations across subjects.
    aggregate = (
        results_df.groupby("split")[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
    )

    aggregate.to_csv(
        OUTPUT_ROOT / "aggregate_results.csv"
    )

    print("\n" + "=" * 60)
    print("STEP 82 COMPLETE")
    print("=" * 60)
    print(aggregate.round(4).to_string())
    print(f"\nResults saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
