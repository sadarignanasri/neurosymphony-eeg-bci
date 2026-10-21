
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_selection import f_classif


# ============================================================
# EXPERIMENT 106
# EEG feature separability + band-power LOSO baseline
# Dataset: ds004022
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT
    / "experiments"
    / "ds004022_experiment104_results"
    / "band_power_features.csv"
)

OUT_DIR = ROOT / "experiments" / "ds004022_experiment106_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CLASSES = {
    "0": "reaching",
    "1": "grasping",
    "2": "lifting",
    "3": "twisting",
}

RANDOM_STATE = 42


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Experiment 104 features not found: {FEATURE_PATH}"
        )

    df = pd.read_csv(FEATURE_PATH)

    required = ["trial_index", "subject", "label", "class_name"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    feature_cols = [
        col for col in df.columns
        if col.endswith("_mean_power")
        or col.endswith("_median_power")
    ]

    if not feature_cols:
        raise ValueError("No band-power feature columns found.")

    df["label"] = df["label"].astype(str)
    df["subject"] = df["subject"].astype(str)

    if df[feature_cols].isna().any().any():
        raise ValueError(
            "Feature table contains NaN values. Investigate before training."
        )

    if not np.isfinite(df[feature_cols].to_numpy()).all():
        raise ValueError(
            "Feature table contains infinite values. Investigate first."
        )

    classes = sorted(df["label"].unique().tolist())
    if len(classes) != 4:
        raise ValueError(
            f"Expected four classes, found {len(classes)}: {classes}"
        )

    print("\nEXPERIMENT 106 — FEATURE SEPARABILITY")
    print("=" * 60)
    print("Rows:", len(df))
    print("Subjects:", sorted(df["subject"].unique().tolist()))
    print("Classes:", classes)
    print("Features:", feature_cols)

    # --------------------------------------------------------
    # 1. Class counts per subject
    # --------------------------------------------------------
    count_table = pd.crosstab(
        [df["subject"]],
        df["class_name"],
    )
    count_table.to_csv(OUT_DIR / "class_counts_by_subject.csv")

    # --------------------------------------------------------
    # 2. Descriptive feature statistics by class
    # --------------------------------------------------------
    class_stats = (
        df.groupby(["label", "class_name"])[feature_cols]
        .agg(["mean", "median", "std"])
    )
    class_stats.to_csv(OUT_DIR / "feature_statistics_by_class.csv")

    subject_class_stats = (
        df.groupby(["subject", "label", "class_name"])[feature_cols]
        .mean()
        .reset_index()
    )
    subject_class_stats.to_csv(
        OUT_DIR / "feature_means_by_subject_and_class.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. One-way ANOVA F statistics
    #
    # Descriptive screening only; these p-values are not used
    # as independent confirmatory evidence.
    # --------------------------------------------------------
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["label"].to_numpy()

    f_values, p_values = f_classif(X_all, y_all)

    anova_df = pd.DataFrame({
        "feature": feature_cols,
        "f_statistic": f_values,
        "p_value_descriptive": p_values,
    }).sort_values("f_statistic", ascending=False)

    anova_df.to_csv(
        OUT_DIR / "descriptive_feature_class_association.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. LOSO baseline: standardize inside each training fold
    # --------------------------------------------------------
    groups = df["subject"].to_numpy()
    logo = LeaveOneGroupOut()

    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )

    # Each trial's prediction is made by a model that did not
    # train on that trial's subject.
    y_pred = cross_val_predict(
        model,
        X_all,
        y_all,
        groups=groups,
        cv=logo,
        n_jobs=1,
        method="predict",
    )

    accuracy = accuracy_score(y_all, y_pred)
    balanced_acc = balanced_accuracy_score(y_all, y_pred)
    macro_f1 = f1_score(
        y_all,
        y_pred,
        average="macro",
        zero_division=0,
    )

    labels = sorted(np.unique(y_all).tolist())
    class_names = [
        EXPECTED_CLASSES.get(label, label)
        for label in labels
    ]

    cm = confusion_matrix(y_all, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(OUT_DIR / "loso_confusion_matrix.csv")

    report = classification_report(
        y_all,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(
        OUT_DIR / "loso_classification_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    predictions = df[
        ["trial_index", "subject", "label", "class_name"]
    ].copy()
    predictions["predicted_label"] = y_pred
    predictions["correct"] = (
        predictions["label"].to_numpy() == y_pred
    )
    predictions.to_csv(
        OUT_DIR / "loso_trial_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Subject-wise scores
    # --------------------------------------------------------
    subject_rows = []

    for subject in sorted(df["subject"].unique()):
        mask = groups == subject
        subject_rows.append({
            "subject": subject,
            "n_trials": int(mask.sum()),
            "accuracy": float(
                accuracy_score(y_all[mask], y_pred[mask])
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_all[mask], y_pred[mask])
            ),
            "macro_f1": float(
                f1_score(
                    y_all[mask],
                    y_pred[mask],
                    average="macro",
                    zero_division=0,
                )
            ),
        })

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(
        OUT_DIR / "loso_subject_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Save summary
    # --------------------------------------------------------
    summary = {
        "experiment": 106,
        "dataset": "ds004022",
        "method": (
            "Band-power features + StandardScaler + shrinkage LDA; "
            "Leave-One-Subject-Out predictions"
        ),
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "class_counts": {
            str(label): int((y_all == label).sum())
            for label in labels
        },
        "chance_level": 0.25,
        "pooled_loso_accuracy": float(accuracy),
        "pooled_loso_balanced_accuracy": float(balanced_acc),
        "pooled_loso_macro_f1": float(macro_f1),
        "mean_subject_accuracy": float(subject_df["accuracy"].mean()),
        "mean_subject_balanced_accuracy": float(
            subject_df["balanced_accuracy"].mean()
        ),
        "mean_subject_macro_f1": float(subject_df["macro_f1"].mean()),
        "warning": (
            "ANOVA feature rankings are descriptive and computed across "
            "all trials, so they are not an unbiased feature-selection "
            "procedure. They were not used to train the LOSO baseline. "
            "Only four subjects are available, limiting generalization claims."
        ),
    }

    with open(
        OUT_DIR / "experiment106_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 106 COMPLETE")
    print("=" * 60)
    print(f"Pooled LOSO accuracy:          {accuracy:.4f}")
    print(f"Pooled balanced accuracy:      {balanced_acc:.4f}")
    print(f"Pooled macro F1:                {macro_f1:.4f}")
    print(f"Mean subject accuracy:          {subject_df['accuracy'].mean():.4f}")
    print(f"Mean subject balanced accuracy: {subject_df['balanced_accuracy'].mean():.4f}")
    print(f"Mean subject macro F1:           {subject_df['macro_f1'].mean():.4f}")
    print("\nFeature rankings:")
    print(anova_df.to_string(index=False))
    print("\nSubject-wise results:")
    print(subject_df.to_string(index=False))
    print("\nOutputs:", OUT_DIR)


if __name__ == "__main__":
    main()
