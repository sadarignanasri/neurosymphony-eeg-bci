
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment134_corrected_features"
)
OUTPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment138_error_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_DIR / "out_of_fold_predictions.csv")

df = df[
    (df["feature_family"] == "combined")
    & (df["model"] == "RBF_SVM")
].copy()

if df.empty:
    raise ValueError("No predictions found for combined + RBF-SVM.")

df["correct"] = df["true_label"] == df["predicted_label"]
df["error_type"] = df.apply(
    lambda row: (
        "correct"
        if row["correct"]
        else f"{int(row['true_label'])}_predicted_as_"
             f"{int(row['predicted_label'])}"
    ),
    axis=1,
)

# Overall confusion matrix
labels = sorted(set(df["true_label"]) | set(df["predicted_label"]))
cm = confusion_matrix(
    df["true_label"], df["predicted_label"], labels=labels
)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels],
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# Class-wise performance
report = classification_report(
    df["true_label"],
    df["predicted_label"],
    labels=labels,
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "classwise_report.csv"
)

# Error rates by participant
participant_errors = (
    df.groupby(["participant", "group"])
    .agg(
        trials=("correct", "size"),
        correct_trials=("correct", "sum"),
        accuracy=("correct", "mean"),
    )
    .reset_index()
)
participant_errors["error_rate"] = (
    1 - participant_errors["accuracy"]
)
participant_errors.sort_values("error_rate", ascending=False).to_csv(
    OUTPUT_DIR / "participant_error_rates.csv", index=False
)

# Error rates by group and true class
group_class_errors = (
    df.groupby(["group", "true_label"])
    .agg(
        trials=("correct", "size"),
        accuracy=("correct", "mean"),
    )
    .reset_index()
)
group_class_errors["error_rate"] = 1 - group_class_errors["accuracy"]
group_class_errors.to_csv(
    OUTPUT_DIR / "group_class_error_rates.csv", index=False
)

# Most common wrong predictions
wrong = df.loc[~df["correct"]].copy()
wrong_predictions = (
    wrong.groupby(["group", "true_label", "predicted_label"])
    .size()
    .reset_index(name="error_count")
    .sort_values("error_count", ascending=False)
)
wrong_predictions.to_csv(
    OUTPUT_DIR / "most_common_wrong_predictions.csv", index=False
)

summary = {
    "experiment": 138,
    "model": "combined + RBF-SVM",
    "total_trials": int(len(df)),
    "incorrect_predictions": int((~df["correct"]).sum()),
    "overall_error_rate": float((~df["correct"]).mean()),
    "labels": [int(label) for label in labels],
    "note": (
        "Labels 0 and 1 are retained as stored in the dataset. "
        "Their semantic class names have not been independently "
        "verified in this script."
    ),
}

with open(
    OUTPUT_DIR / "error_analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print("\nERROR ANALYSIS SUMMARY")
print(json.dumps(summary, indent=2))

print("\nCONFUSION MATRIX")
print(cm_df.to_string())

print("\nGROUP-WISE CLASS ERROR RATES")
print(group_class_errors.to_string(index=False))

print("\nMOST COMMON WRONG PREDICTIONS")
print(wrong_predictions.head(10).to_string(index=False))

print("\nHIGHEST PARTICIPANT ERROR RATES")
print(
    participant_errors.sort_values("error_rate", ascending=False)
    .head(10).to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_DIR}")
