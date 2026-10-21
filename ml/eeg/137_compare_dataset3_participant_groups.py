
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment134_corrected_features"
)
OUTPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment137_group_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_DIR / "out_of_fold_predictions.csv")
df = df[
    (df["feature_family"] == "combined")
    & (df["model"] == "RBF_SVM")
].copy()

if df.empty:
    raise ValueError("No combined + RBF-SVM predictions found.")

# Calculate one row per held-out participant.
rows = []
for participant, group in df.groupby("participant"):
    y_true = group["true_label"].astype(int)
    y_pred = group["predicted_label"].astype(int)

    rows.append({
        "participant": participant,
        "group": group["group"].iloc[0],
        "n_trials": len(group),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    })

participant_df = pd.DataFrame(rows)
participant_df.to_csv(
    OUTPUT_DIR / "participant_group_metrics.csv", index=False
)

def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, None, None
    se = values.std(ddof=1) / np.sqrt(n)
    critical = student_t.ppf(0.975, df=n - 1)
    return mean, float(mean - critical * se), float(mean + critical * se)

group_results = {}
for name, group in participant_df.groupby("group"):
    group_results[name] = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        mean, low, high = mean_ci(group[metric])
        group_results[name][metric] = {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        }
    group_results[name]["participants"] = int(len(group))
    group_results[name]["trials"] = int(group["n_trials"].sum())

normal = participant_df.loc[
    participant_df["group"] == "Normals", "accuracy"
].to_numpy()
patient = participant_df.loc[
    participant_df["group"] == "Patients", "accuracy"
].to_numpy()

if len(normal) < 2 or len(patient) < 2:
    raise ValueError("Not enough participants in one or both groups.")

t_stat, t_p = ttest_ind(normal, patient, equal_var=False)
u_stat, u_p = mannwhitneyu(
    normal, patient, alternative="two-sided"
)

result = {
    "experiment": 137,
    "model": "combined + RBF-SVM",
    "group_results": group_results,
    "normal_minus_patient_mean_accuracy": float(
        normal.mean() - patient.mean()
    ),
    "welch_t_test": {
        "statistic": float(t_stat),
        "p_value_two_sided": float(t_p),
    },
    "mann_whitney_test": {
        "statistic": float(u_stat),
        "p_value_two_sided": float(u_p),
    },
    "limitations": (
        "Exploratory group comparison on existing out-of-fold predictions. "
        "Participants were reused during model selection. The tests compare "
        "participant-level scores, not independent clinical populations."
    ),
}

with open(
    OUTPUT_DIR / "group_comparison_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(result, f, indent=2)

print("\nGROUP-WISE PERFORMANCE")
print(participant_df.groupby("group")[
    ["accuracy", "balanced_accuracy", "macro_f1"]
].agg(["count", "mean", "std"]).to_string())

print("\nGROUP COMPARISON")
print(json.dumps(result, indent=2))
print(f"\nResults saved to: {OUTPUT_DIR}")
