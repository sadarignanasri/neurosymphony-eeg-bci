
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment134_corrected_features"
)
OUTPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment135_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

predictions = pd.read_csv(INPUT_DIR / "out_of_fold_predictions.csv")
fold_metrics = pd.read_csv(INPUT_DIR / "participant_fold_metrics.csv")

# Select the best model by pooled accuracy from Experiment 134.
best = (
    predictions.groupby(["feature_family", "model"])
    .apply(
        lambda g: accuracy_score(g["true_label"], g["predicted_label"]),
        include_groups=False,
    )
    .sort_values(ascending=False)
)

print("\nMODEL RANKING BY POOLED ACCURACY")
print(best.to_string())

best_family, best_model = best.index[0]
print(f"\nSelected model: {best_family} + {best_model}")

df = predictions[
    (predictions["feature_family"] == best_family)
    & (predictions["model"] == best_model)
].copy()

y_true = df["true_label"].astype(int)
y_pred = df["predicted_label"].astype(int)
labels = sorted(set(y_true) | set(y_pred))

summary = {
    "feature_family": best_family,
    "model": best_model,
    "n_trials": int(len(df)),
    "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
    "pooled_balanced_accuracy": float(
        balanced_accuracy_score(y_true, y_pred)
    ),
    "pooled_macro_f1": float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    ),
}

# Overall class-wise report
report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "classwise_report.csv", index=True)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels],
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)

# Participant-wise metrics for the selected model
selected_folds = fold_metrics[
    (fold_metrics["feature_family"] == best_family)
    & (fold_metrics["model"] == best_model)
].copy()

selected_folds.to_csv(
    OUTPUT_DIR / "selected_model_participant_metrics.csv",
    index=False,
)

summary["participant_count"] = int(
    selected_folds["held_out_participant"].nunique()
)
summary["mean_participant_accuracy"] = float(
    selected_folds["accuracy"].mean()
)
summary["mean_participant_balanced_accuracy"] = float(
    selected_folds["balanced_accuracy"].mean()
)
summary["mean_participant_macro_f1"] = float(
    selected_folds["macro_f1"].mean()
)
summary["min_participant_accuracy"] = float(
    selected_folds["accuracy"].min()
)
summary["max_participant_accuracy"] = float(
    selected_folds["accuracy"].max()
)

summary["chance_level"] = 0.50
summary["accuracy_above_chance_percentage_points"] = (
    summary["pooled_accuracy"] - 0.50
) * 100
summary["interpretation_note"] = (
    "Exploratory validation on existing out-of-fold predictions; "
    "not an independent test set."
)

with open(
    OUTPUT_DIR / "validation_summary.json", "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nOVERALL VALIDATION")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nCLASS-WISE REPORT")
print(report_df.to_string())

print("\nCONFUSION MATRIX")
print(cm_df.to_string())

print("\nPARTICIPANTS WITH LOWEST ACCURACY")
print(
    selected_folds.sort_values("accuracy")[
        ["held_out_participant", "group", "n_trials",
         "accuracy", "balanced_accuracy", "macro_f1"]
    ].head(10).to_string(index=False)
)

print("\nPARTICIPANTS WITH HIGHEST ACCURACY")
print(
    selected_folds.sort_values("accuracy", ascending=False)[
        ["held_out_participant", "group", "n_trials",
         "accuracy", "balanced_accuracy", "macro_f1"]
    ].head(10).to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_DIR}")
