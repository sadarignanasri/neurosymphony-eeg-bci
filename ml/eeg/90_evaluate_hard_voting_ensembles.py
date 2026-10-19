
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# NEUROSYMPHONY — STEP 90
# PRACTICAL HARD / PLURALITY VOTING ENSEMBLES
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "results"
    / "eeg"
    / "oracle_coverage"
    / "step89"
    / "sample_wise_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "eeg"
    / "ensembles"
    / "step90_hard_voting"
)

MODEL_COMBINATIONS = {
    "LR_RF": [
        "logistic_regression",
        "random_forest",
    ],
    "LR_RF_ET": [
        "logistic_regression",
        "random_forest",
        "extra_trees",
    ],
    "LR_LinearSVM_RF": [
        "logistic_regression",
        "linear_svm",
        "random_forest",
    ],
    "LR_RF_HGB": [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ],
    "All_8_Models": [
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "random_forest",
        "extra_trees",
        "knn",
        "lda",
        "hist_gradient_boosting",
    ],
}

IDENTITY_COLUMNS = [
    "sample_id",
    "subject",
    "trial_index",
    "true_label",
]


def plurality_vote(prediction_values):
    """
    Select the most frequently predicted class.
    If multiple classes tie, choose the smallest numeric label.
    This is a deterministic tie-breaking rule, not a learned rule.
    """
    values, counts = np.unique(
        np.asarray(prediction_values, dtype=int),
        return_counts=True,
    )

    return int(values[counts == counts.max()].min())


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy_percent": (
            100 * accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy_percent": (
            100 * balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Step 89 predictions not found: {INPUT_FILE}"
        )

    long_df = pd.read_csv(INPUT_FILE)

    required_columns = set(
        IDENTITY_COLUMNS
        + ["model", "predicted_label", "correct"]
    )

    missing = required_columns - set(long_df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if long_df.empty:
        raise ValueError("Prediction file is empty.")

    if long_df.duplicated(
        ["sample_id", "model"]
    ).any():
        raise ValueError(
            "Duplicate predictions found for a sample/model pair."
        )

    # Every sample must have one consistent true label and subject.
    identity_counts = (
        long_df.groupby("sample_id")[IDENTITY_COLUMNS[1:]]
        .nunique()
    )

    if (identity_counts > 1).any().any():
        raise ValueError(
            "Inconsistent sample identity or true labels detected."
        )

    # Convert long predictions into one row per test sample.
    prediction_wide = long_df.pivot(
        index="sample_id",
        columns="model",
        values="predicted_label",
    )

    metadata = (
        long_df[IDENTITY_COLUMNS]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
    )

    data = metadata.join(prediction_wide).reset_index()

    model_columns = list(MODEL_COMBINATIONS["All_8_Models"])

    missing_models = set(model_columns) - set(data.columns)

    if missing_models:
        raise ValueError(
            f"Missing model predictions: {sorted(missing_models)}"
        )

    if data[model_columns].isna().any().any():
        raise ValueError(
            "Some samples are missing one or more model predictions."
        )

    data["true_label"] = data["true_label"].astype(int)

    # --------------------------------------------------------
    # INDIVIDUAL MODEL PERFORMANCE
    # --------------------------------------------------------

    individual_rows = []

    for model_name in model_columns:
        metrics = calculate_metrics(
            data["true_label"],
            data[model_name].astype(int),
        )

        individual_rows.append({
            "method": model_name,
            "number_of_models": 1,
            "test_samples": len(data),
            **metrics,
        })

    # --------------------------------------------------------
    # PRACTICAL VOTING ENSEMBLES
    # --------------------------------------------------------

    ensemble_rows = []
    ensemble_predictions = data[
        IDENTITY_COLUMNS
    ].copy()

    for combination_name, selected_models in (
        MODEL_COMBINATIONS.items()
    ):
        prediction_matrix = data[selected_models].to_numpy(
            dtype=int
        )

        voted_predictions = np.array([
            plurality_vote(row)
            for row in prediction_matrix
        ])

        ensemble_predictions[combination_name] = (
            voted_predictions
        )

        metrics = calculate_metrics(
            data["true_label"],
            voted_predictions,
        )

        ensemble_rows.append({
            "method": combination_name,
            "models": " + ".join(selected_models),
            "number_of_models": len(selected_models),
            "test_samples": len(data),
            "tie_break": "smallest numeric class label",
            **metrics,
        })

    ensemble_predictions.to_csv(
        OUTPUT_DIR / "hard_voting_sample_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # POOLED RESULTS
    # --------------------------------------------------------

    comparison = pd.concat(
        [
            pd.DataFrame(individual_rows),
            pd.DataFrame(ensemble_rows),
        ],
        ignore_index=True,
    )

    comparison = comparison.sort_values(
        "accuracy_percent",
        ascending=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "hard_voting_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PER-SUBJECT ENSEMBLE RESULTS
    # --------------------------------------------------------

    subject_rows = []

    for subject, group in data.groupby("subject"):
        for combination_name, selected_models in (
            MODEL_COMBINATIONS.items()
        ):
            predictions = np.array([
                plurality_vote(row)
                for row in group[selected_models].to_numpy(
                    dtype=int
                )
            ])

            metrics = calculate_metrics(
                group["true_label"].to_numpy(dtype=int),
                predictions,
            )

            subject_rows.append({
                "subject": subject,
                "method": combination_name,
                "test_samples": len(group),
                **metrics,
            })

    subject_results = pd.DataFrame(subject_rows)

    subject_results.to_csv(
        OUTPUT_DIR / "hard_voting_by_subject.csv",
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("=" * 72)
    print("NEUROSYMPHONY — STEP 90")
    print("PRACTICAL HARD / PLURALITY VOTING RESULTS")
    print("=" * 72)

    print(f"Unique test samples: {len(data)}")
    print("Voting rule: most frequent predicted class")
    print("Tie rule: smallest numeric class label")

    print("\nINDIVIDUAL MODELS AND VOTING ENSEMBLES")
    print(
        comparison[
            [
                "method",
                "number_of_models",
                "accuracy_percent",
                "balanced_accuracy_percent",
                "macro_f1",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nENSEMBLE MEAN ACCURACY ACROSS SUBJECTS")
    print(
        subject_results.groupby("method")[
            "accuracy_percent"
        ].mean().sort_values(ascending=False).round(3).to_string()
    )

    print("\nSaved results:")
    print(OUTPUT_DIR.resolve())

    print("\nSTEP 90 COMPLETED.")


if __name__ == "__main__":
    main()
