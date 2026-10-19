

"""
NeuroSymphony — Step 62: All-Subject Random Forest Analysis

Evaluates training and validation performance for all 12 subjects.
Analyzes feature importance and saves subject-wise and aggregate reports.
The test set is checked for split integrity only and is never evaluated.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "data" / "features"
SPLIT_ROOT = ROOT / "data" / "splits"
RESULT_ROOT = ROOT / "results" / "eeg" / "model_comparison"

SUBJECTS = range(1, 13)
EXPECTED_TRIALS = 720
EXPECTED_FEATURES = 320
EXPECTED_CHANNELS = 64
EXPECTED_BANDS = 5

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_NAMES = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)


def subject_paths(number):
    """Return paths for one subject."""

    sid = f"sub-{number:02d}"

    if number == 1:
        feature_dir = FEATURE_ROOT / "temporal_spectral_sub01"
        split_dir = SPLIT_ROOT / "sub01"
        result_dir = RESULT_ROOT / "sub01"
        metadata_path = feature_dir / f"{sid}_spectral_metadata.csv"
        output_dir = (
            result_dir
            / "step62_overfitting_feature_importance_all_subjects_rerun"
        )
    else:
        feature_dir = FEATURE_ROOT / f"temporal_spectral_{sid}"
        split_dir = SPLIT_ROOT / sid
        result_dir = RESULT_ROOT / sid

        metadata_path = (
            FEATURE_ROOT
            / f"temporal_eeg_{sid}"
            / f"{sid}_temporal_eeg_preprocessed_metadata.csv"
        )

        output_dir = (
            result_dir / "step62_overfitting_feature_importance"
        )

    return {
        "sid": sid,
        "feature": feature_dir / f"{sid}_band_power.npy",
        "metadata": metadata_path,
        "train": split_dir / "train_indices.npy",
        "val": split_dir / "val_indices.npy",
        "test": split_dir / "test_indices.npy",
        "model": (
            result_dir
            / "step61_random_forest_tuning"
            / "step61_selected_pipeline.joblib"
        ),
        "output": output_dir,
    }


def validate_splits(paths):
    """Validate split sizes, uniqueness, separation and coverage."""

    train_idx = np.load(paths["train"]).astype(int)
    val_idx = np.load(paths["val"]).astype(int)
    test_idx = np.load(paths["test"]).astype(int)

    expected_sizes = (504, 108, 108)
    actual_sizes = (len(train_idx), len(val_idx), len(test_idx))

    if actual_sizes != expected_sizes:
        raise ValueError(
            f"{paths['sid']}: expected split sizes {expected_sizes}, "
            f"found {actual_sizes}"
        )

    for name, indices in (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                f"{paths['sid']}: duplicate {name} indices"
            )

        if np.any(indices < 0) or np.any(indices >= EXPECTED_TRIALS):
            raise ValueError(
                f"{paths['sid']}: invalid {name} indices"
            )

    if (
        set(train_idx) & set(val_idx)
        or set(train_idx) & set(test_idx)
        or set(val_idx) & set(test_idx)
    ):
        raise ValueError(f"{paths['sid']}: split overlap detected")

    combined = np.concatenate([train_idx, val_idx, test_idx])

    if not np.array_equal(
        np.sort(combined), np.arange(EXPECTED_TRIALS)
    ):
        raise ValueError(
            f"{paths['sid']}: splits do not cover all trials"
        )

    return train_idx, val_idx, test_idx


def analyze_subject(number):
    """Analyze one subject without evaluating its test set."""

    paths = subject_paths(number)
    sid = paths["sid"]
    out = paths["output"]

    print("\n" + "=" * 78)
    print(f"STEP 62 — {sid}")
    print("=" * 78)

    for key in (
        "feature", "metadata", "train", "val", "test", "model"
    ):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"{sid}: required {key} file not found:\n"
                f"{paths[key]}"
            )

    out.mkdir(parents=True, exist_ok=True)

    # Load band-power features.
    X_raw = np.load(paths["feature"])

    if X_raw.shape != (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_BANDS,
    ):
        raise ValueError(
            f"{sid}: unexpected feature shape {X_raw.shape}; "
            "expected (720, 64, 5)"
        )

    if not np.isfinite(X_raw).all():
        raise ValueError(f"{sid}: features contain NaN or infinity")

    X = X_raw.reshape(EXPECTED_TRIALS, EXPECTED_FEATURES)

    # Load the correct trial labels for each subject.
    metadata = pd.read_csv(paths["metadata"])

    if number == 1:
        label_column = "label"
    else:
        label_column = "label_id"

    if label_column not in metadata.columns:
        raise ValueError(
            f"{sid}: column '{label_column}' not found in "
            f"{paths['metadata']}. Available columns: "
            f"{metadata.columns.tolist()}"
        )

    if len(metadata) != EXPECTED_TRIALS:
        raise ValueError(
            f"{sid}: expected 720 metadata rows, found {len(metadata)}"
        )

    y = metadata[label_column].to_numpy()

    if pd.isna(y).any() or len(np.unique(y)) != 8:
        raise ValueError(
            f"{sid}: labels are missing or do not contain 8 classes"
        )

    train_idx, val_idx, _ = validate_splits(paths)

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    # Load the selected Step 61 Random Forest pipeline.
    pipeline = joblib.load(paths["model"])

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            f"{sid}: saved model is not a scikit-learn pipeline"
        )

    if "classifier" not in pipeline.named_steps:
        raise ValueError(
            f"{sid}: pipeline does not contain a 'classifier' step"
        )

    model = pipeline.named_steps["classifier"]

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{sid}: classifier does not support feature importances"
        )

    if len(model.feature_importances_) != EXPECTED_FEATURES:
        raise ValueError(
            f"{sid}: model does not have 320 feature importances"
        )

    # Evaluate training and validation only.
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    metrics = {
        "subject": sid,
        "training_accuracy": accuracy_score(y_train, train_pred),
        "validation_accuracy": accuracy_score(y_val, val_pred),
        "training_balanced_accuracy": balanced_accuracy_score(
            y_train, train_pred
        ),
        "validation_balanced_accuracy": balanced_accuracy_score(
            y_val, val_pred
        ),
        "training_macro_f1": f1_score(
            y_train, train_pred, average="macro", zero_division=0
        ),
        "validation_macro_f1": f1_score(
            y_val, val_pred, average="macro", zero_division=0
        ),
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "class_weight": str(model.class_weight),
    }

    metrics["accuracy_gap"] = (
        metrics["training_accuracy"]
        - metrics["validation_accuracy"]
    )

    metrics["macro_f1_gap"] = (
        metrics["training_macro_f1"]
        - metrics["validation_macro_f1"]
    )

    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
            if key != "subject"
        ]
    ).to_csv(
        out / "step62_performance_metrics.csv", index=False
    )

    report = classification_report(
        y_val,
        val_pred,
        labels=np.sort(np.unique(y)),
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        out / "step62_validation_classification_report.csv"
    )

    # Analyze feature importance.
    # Flattening assumes channel-major, band-minor feature order.
    feature_rows = []
    importances = model.feature_importances_

    for index, importance in enumerate(importances):
        channel_index = index // EXPECTED_BANDS
        band_index = index % EXPECTED_BANDS

        channel = CHANNEL_NAMES[channel_index]
        band = BAND_NAMES[band_index]

        feature_rows.append({
            "subject": sid,
            "feature_index": index,
            "channel": channel,
            "band": band,
            "feature_name": f"{channel}_{band}",
            "importance": float(importance),
        })

    importance_df = pd.DataFrame(feature_rows).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    importance_df.insert(
        0, "importance_rank", np.arange(1, len(importance_df) + 1)
    )

    importance_df.to_csv(
        out / "step62_feature_importance.csv", index=False
    )

    band_df = importance_df.groupby(
        "band", as_index=False, observed=False
    )["importance"].sum()

    band_df["subject"] = sid
    band_df["band"] = pd.Categorical(
        band_df["band"], categories=BAND_NAMES, ordered=True
    )
    band_df = band_df.sort_values("band")

    band_df.to_csv(
        out / "step62_band_importance.csv", index=False
    )

    channel_df = (
        importance_df.groupby("channel", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    channel_df.insert(
        0, "importance_rank", np.arange(1, len(channel_df) + 1)
    )
    channel_df["subject"] = sid

    channel_df.to_csv(
        out / "step62_channel_importance.csv", index=False
    )

    # Plot training versus validation metrics.
    metric_labels = [
        "accuracy", "balanced_accuracy", "macro_f1"
    ]

    train_values = [
        metrics["training_accuracy"],
        metrics["training_balanced_accuracy"],
        metrics["training_macro_f1"],
    ]

    val_values = [
        metrics["validation_accuracy"],
        metrics["validation_balanced_accuracy"],
        metrics["validation_macro_f1"],
    ]

    positions = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        positions - width / 2, train_values, width, label="Training"
    )
    ax.bar(
        positions + width / 2, val_values, width, label="Validation"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Step 62 — {sid}: Training vs Validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step62_training_validation.png", dpi=160)
    plt.close(fig)

    # Plot the top 20 features.
    top_features = importance_df.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_features["feature_name"], top_features["importance"]
    )
    ax.set_xlabel("Random Forest impurity-based importance")
    ax.set_title(f"Step 62 — {sid}: Top 20 Features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step62_top20_feature_importance.png", dpi=160)
    plt.close(fig)

    # Plot importance by frequency band.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        band_df["band"].astype(str), band_df["importance"]
    )
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Summed feature importance")
    ax.set_title(f"Step 62 — {sid}: Importance by Band")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step62_band_importance.png", dpi=160)
    plt.close(fig)

    # Write subject summary.
    with open(
        out / "step62_summary.txt", "w", encoding="utf-8"
    ) as file:
        file.write(f"NEUROSYMPHONY — STEP 62 SUMMARY — {sid}\n")
        file.write("=" * 60 + "\n")
        file.write("Model: Selected Step 61 Random Forest\n")
        file.write("Feature count: 320\n")
        file.write("Training trials: 504\n")
        file.write("Validation trials: 108\n")
        file.write("Test evaluation: NOT performed\n\n")
        file.write("Metrics and model parameters:\n")

        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

        file.write("\nTop 20 features:\n")
        file.write(importance_df.head(20).to_string(index=False))
        file.write("\n\nImportance by frequency band:\n")
        file.write(band_df.to_string(index=False))
        file.write(
            "\n\nCaution: confirm channel order and band order against "
            "feature-generation code before interpreting importance.\n"
        )

    print(
        f"{sid}: train macro-F1="
        f"{metrics['training_macro_f1']:.4f} | "
        f"validation macro-F1="
        f"{metrics['validation_macro_f1']:.4f} | "
        f"accuracy gap={metrics['accuracy_gap']:.4f}"
    )
    print(f"Saved: {out}")

    return metrics, importance_df, band_df, channel_df


def main():
    all_metrics = []
    all_features = []
    all_bands = []
    all_channels = []
    failures = []

    for number in SUBJECTS:
        try:
            metrics, features, bands, channels = analyze_subject(number)

            all_metrics.append(metrics)
            all_features.append(features)
            all_bands.append(bands)
            all_channels.append(channels)

        except Exception as exc:
            sid = f"sub-{number:02d}"
            failures.append({
                "subject": sid,
                "error": str(exc),
            })
            print(f"\nERROR — {sid}: {exc}")

    aggregate_dir = (
        RESULT_ROOT / "all_subjects" / "step62_summary"
    )
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        metrics_df.to_csv(
            aggregate_dir / "step62_all_subjects_metrics.csv",
            index=False,
        )

        pd.concat(all_features, ignore_index=True).to_csv(
            aggregate_dir / "step62_all_subjects_feature_importance.csv",
            index=False,
        )

        pd.concat(all_bands, ignore_index=True).to_csv(
            aggregate_dir / "step62_all_subjects_band_importance.csv",
            index=False,
        )

        pd.concat(all_channels, ignore_index=True).to_csv(
            aggregate_dir / "step62_all_subjects_channel_importance.csv",
            index=False,
        )

        print("\nALL-SUBJECT METRICS")
        print(
            metrics_df[
                [
                    "subject",
                    "training_macro_f1",
                    "validation_macro_f1",
                    "accuracy_gap",
                    "macro_f1_gap",
                ]
            ].to_string(index=False)
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            aggregate_dir / "step62_failures.csv", index=False
        )

        print("\nSome subjects failed. See step62_failures.csv:")
        for failure in failures:
            print(f"{failure['subject']}: {failure['error']}")

        raise RuntimeError(
            f"Step 62 had {len(failures)} subject failure(s). "
            "Resolve these before considering Step 62 complete."
        )

    print("\n" + "=" * 78)
    print("STEP 62 COMPLETED FOR ALL 12 SUBJECTS")
    print("Test-set performance was not evaluated.")
    print(f"Aggregate reports: {aggregate_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
