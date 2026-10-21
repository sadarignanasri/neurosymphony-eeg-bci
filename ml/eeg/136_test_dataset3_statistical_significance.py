
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon, binomtest
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment135_validation"
)
OUTPUT_DIR = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment136_significance"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = INPUT_DIR / "selected_model_participant_metrics.csv"
if not metrics_path.exists():
    raise FileNotFoundError(f"Missing file: {metrics_path}")

df = pd.read_csv(metrics_path)

required = {"held_out_participant", "accuracy"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df["held_out_participant"].duplicated().any():
    raise ValueError("Duplicate participant rows found.")

scores = df["accuracy"].dropna().to_numpy(dtype=float)
if len(scores) < 2:
    raise ValueError("Not enough participant scores.")

chance = 0.50
differences = scores - chance
n = len(scores)
mean_score = float(np.mean(scores))
sd_score = float(np.std(scores, ddof=1))
se = sd_score / np.sqrt(n)

# Two-sided 95% confidence interval for the mean participant accuracy.
critical = student_t.ppf(0.975, df=n - 1)
ci_low = mean_score - critical * se
ci_high = mean_score + critical * se

# Primary test: one-sided paired participant-level t-test against 50%.
t_stat, p_t_two_sided = ttest_1samp(scores, popmean=chance)
p_t_one_sided = (
    p_t_two_sided / 2
    if t_stat > 0
    else 1 - p_t_two_sided / 2
)

# Nonparametric sensitivity test.
try:
    w_stat, p_wilcoxon = wilcoxon(
        differences,
        alternative="greater",
        zero_method="wilcox",
    )
    w_stat = float(w_stat)
    p_wilcoxon = float(p_wilcoxon)
except ValueError:
    w_stat = None
    p_wilcoxon = None

# Descriptive only: pooled trial-level binomial test.
# This treats trials as independent, so it is NOT the primary inference.
pooled_correct = int(round(scores.sum() * 0))  # replaced below
pooled_path = (
    ROOT / "experiments" / "dataset3_28955456_training"
    / "experiment134_corrected_features"
    / "out_of_fold_predictions.csv"
)
pred = pd.read_csv(pooled_path)
pred = pred[
    (pred["feature_family"] == "combined")
    & (pred["model"] == "RBF_SVM")
]
pooled_correct = int(
    (pred["true_label"] == pred["predicted_label"]).sum()
)
pooled_n = len(pred)
pooled_binom_p = float(
    binomtest(
        pooled_correct, pooled_n, p=chance, alternative="greater"
    ).pvalue
)

result = {
    "experiment": 136,
    "model": "combined + RBF-SVM",
    "participants": int(n),
    "chance_level": chance,
    "mean_participant_accuracy": mean_score,
    "participant_accuracy_sd": sd_score,
    "mean_accuracy_95pct_ci": [float(ci_low), float(ci_high)],
    "participant_level_t_statistic": float(t_stat),
    "participant_level_t_test_p_one_sided": float(p_t_one_sided),
    "participant_level_wilcoxon_statistic": w_stat,
    "participant_level_wilcoxon_p_one_sided": p_wilcoxon,
    "pooled_accuracy": float(pooled_correct / pooled_n),
    "pooled_trials": int(pooled_n),
    "pooled_binomial_test_p_descriptive_only": pooled_binom_p,
    "interpretation": (
        "Exploratory participant-level inference. Participants were "
        "reused during prior model/feature comparisons, so results "
        "are not independent confirmation. The pooled binomial test "
        "does not account for within-participant trial dependence."
    ),
}

with open(
    OUTPUT_DIR / "significance_summary.json", "w", encoding="utf-8"
) as f:
    json.dump(result, f, indent=2)

df["difference_from_chance"] = df["accuracy"] - chance
df.to_csv(
    OUTPUT_DIR / "participant_scores_with_chance.csv", index=False
)

print("\nEXPERIMENT 136 — STATISTICAL RESULTS")
for key, value in result.items():
    print(f"{key}: {value}")

print(f"\nSaved results to: {OUTPUT_DIR}")
