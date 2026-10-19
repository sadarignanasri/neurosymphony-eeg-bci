from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# STEP 46
# FEATURE SELECTION EXPERIMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
    / "step44_sub01_band_power_only.csv"
)

RANKING_PATH = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step45"
    / "step45_all_320_feature_ranking.csv"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step45"
    / "step45_split_indices.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step46"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K_VALUES = [
    10,
    20,
    50,
    100,
    200,
    320,
]

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 46 — FEATURE SELECTION EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature matrix...")

df = pd.read_csv(
    FEATURE_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# LOAD FEATURE RANKING
# ============================================================

print(
    "\nLoading training-only "
    "feature ranking..."
)

ranking_df = pd.read_csv(
    RANKING_PATH
)

ranking_df = ranking_df.sort_values(
    "rank"
)

ranked_features = (
    ranking_df["feature"]
    .tolist()
)


if len(ranked_features) != 320:

    raise ValueError(
        f"Expected 320 ranked features, "
        f"found {len(ranked_features)}."
    )


# ============================================================
# LOAD FIXED SPLIT
# ============================================================

print(
    "\nLoading Step 45 split..."
)

split_df = pd.read_csv(
    SPLIT_PATH
)

train_indices = (
    split_df.loc[
        split_df["split"] == "train",
        "dataset_index"
    ]
    .to_numpy()
)

val_indices = (
    split_df.loc[
        split_df["split"] == "validation",
        "dataset_index"
    ]
    .to_numpy()
)

test_indices = (
    split_df.loc[
        split_df["split"] == "test",
        "dataset_index"
    ]
    .to_numpy()
)


print(
    f"\nTrain      : {len(train_indices)}"
)

print(
    f"Validation : {len(val_indices)}"
)

print(
    f"Test       : {len(test_indices)}"
)


# ============================================================
# PREPARE TARGET
# ============================================================

y = df["label"].to_numpy()


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# RUN EACH FEATURE COUNT
# ============================================================

for k in TOP_K_VALUES:

    print("\n" + "-" * 70)

    print(
        f"Testing TOP {k} FEATURES"
    )

    print("-" * 70)

    selected_features = (
        ranked_features[:k]
    )

    # --------------------------------------------------------
    # Extract selected features
    # --------------------------------------------------------

    X = df[
        selected_features
    ].to_numpy(
        dtype=np.float64
    )

    X_train = X[
        train_indices
    ]

    X_val = X[
        val_indices
    ]

    y_train = y[
        train_indices
    ]

    y_val = y[
        val_indices
    ]

    # --------------------------------------------------------
    # StandardScaler
    #
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train_scaled
    )

    val_pred = model.predict(
        X_val_scaled
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            val_pred
        )
    )

    val_macro_f1 = (
        f1_score(
            y_val,
            val_pred,
            average="macro"
        )
    )

    results.append({

        "top_k":
            k,

        "train_accuracy":
            train_accuracy,

        "validation_accuracy":
            val_accuracy,

        "validation_balanced_accuracy":
            val_balanced_accuracy,

        "validation_macro_f1":
            val_macro_f1,
    })

    print(
        f"Train accuracy       : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation accuracy  : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation balanced  : "
        f"{val_balanced_accuracy:.4f}"
    )

    print(
        f"Validation macro F1  : "
        f"{val_macro_f1:.4f}"
    )


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# FIND BEST K USING VALIDATION ACCURACY
# ============================================================

best_row = (
    results_df
    .sort_values(
        by="validation_accuracy",
        ascending=False
    )
    .iloc[0]
)

best_k = int(
    best_row["top_k"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "step46_feature_selection_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE SELECTED FEATURES FOR EACH K
# ============================================================

for k in TOP_K_VALUES:

    selected_features = (
        ranked_features[:k]
    )

    feature_df = pd.DataFrame({
        "rank":
            np.arange(k) + 1,

        "feature":
            selected_features
    })

    feature_df.to_csv(
        OUTPUT_DIR
        / f"step46_top{k}_features.csv",
        index=False
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("BEST FEATURE COUNT")
print("=" * 70)

print(
    f"\nBest K based on "
    f"validation accuracy: {best_k}"
)

print(
    f"Validation accuracy: "
    f"{best_row['validation_accuracy']:.4f}"
)

print(
    f"Validation balanced accuracy: "
    f"{best_row['validation_balanced_accuracy']:.4f}"
)

print(
    f"Validation macro F1: "
    f"{best_row['validation_macro_f1']:.4f}"
)


# ============================================================
# IMPORTANT METHODOLOGICAL NOTE
# ============================================================

print("\n" + "=" * 70)
print("METHODOLOGICAL NOTE")
print("=" * 70)

print(
    "\nThe test set was NOT used to select K."
)

print(
    "K was selected using validation "
    "performance only."
)

print(
    "The held-out test set will be used "
    "in the next experiment."
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 46 COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
)

print(
    results_path
)