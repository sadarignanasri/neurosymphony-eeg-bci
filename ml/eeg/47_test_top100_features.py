"""
STEP 47 — HELD-OUT TEST EVALUATION
===================================

Purpose
-------
Evaluate the Top-100 absolute band-power features selected in Step 46
on the completely untouched test set.

Methodology
-----------
1. Load Step 44 absolute band-power features.
2. Load Step 45 training-only ANOVA feature ranking.
3. Select the Top 100 features.
4. Load the exact Step 45 train/validation/test split.
5. Use training data to fit StandardScaler.
6. Train Logistic Regression using the Top 100 features.
7. Evaluate ONLY the held-out test set.
8. Save test metrics, confusion matrix, per-class accuracy,
   classification report, and predictions.

Important
---------
- Feature ranking was performed using training data only.
- K=100 was selected using validation performance in Step 46.
- The test set is NOT used for feature selection or K selection.
- The test set is used only for final evaluation here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TOP_K = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
    / "step44_sub01_band_power_only.csv"
)

RANKING_FILE = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step45"
    / "step45_all_320_feature_ranking.csv"
)

SPLIT_FILE = (
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
    / "step47"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 47 — HELD-OUT TEST EVALUATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. LOAD FEATURE MATRIX
    # ========================================================

    print("\nLoading feature matrix...")

    df = pd.read_csv(FEATURE_FILE)

    print(f"Dataset shape: {df.shape}")

    # ========================================================
    # 2. IDENTIFY METADATA AND FEATURE COLUMNS
    # ========================================================

    metadata_columns = [
        "subject",
        "trial_index",
        "label",
        "label_name",
        "event_sample",
        "event_onset",
        "duration",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print(f"Total features: {len(feature_columns)}")

    X = df[feature_columns].values

    # Keep labels exactly as stored in the dataset.
    # In this dataset the label column contains strings.
    y = df["label"].values

    # ========================================================
    # 3. DISPLAY LABEL INFORMATION
    # ========================================================

    print("\nLabel values:")

    unique_labels = pd.unique(y)

    for label in unique_labels:
        print(f"  {label}")

    print(f"\nNumber of classes: {len(unique_labels)}")

    # ========================================================
    # 4. LOAD TRAINING-ONLY FEATURE RANKING
    # ========================================================

    print("\nLoading Step 45 training-only feature ranking...")

    ranking = pd.read_csv(RANKING_FILE)

    ranked_features = ranking["feature"].tolist()

    # Select Top 100 based on Step 45 ranking.
    top_features = ranked_features[:TOP_K]

    print(f"Selected features: {TOP_K}")

    # ========================================================
    # 5. SAFETY CHECK FOR FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in top_features
        if feature not in feature_columns
    ]

    if missing_features:

        raise ValueError(
            "Some ranked features are missing from the feature matrix:\n"
            + "\n".join(missing_features[:20])
        )

    selected_indices = [
        feature_columns.index(feature)
        for feature in top_features
    ]

    X_selected = X[:, selected_indices]

    print(
        f"Selected feature matrix shape: "
        f"{X_selected.shape}"
    )

    # ========================================================
    # 6. LOAD STEP 45 SPLIT
    # ========================================================

    print("\nLoading Step 45 split...")

    split_df = pd.read_csv(SPLIT_FILE)

    print(
        "Split file columns:",
        split_df.columns.tolist()
    )

    # Step 45 split file uses:
    # dataset_index
    # split

    if "dataset_index" not in split_df.columns:
        raise ValueError(
            "Expected 'dataset_index' column was not found "
            "in Step 45 split file."
        )

    if "split" not in split_df.columns:
        raise ValueError(
            "Expected 'split' column was not found "
            "in Step 45 split file."
        )

    train_idx = (
        split_df.loc[
            split_df["split"] == "train",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    val_idx = (
        split_df.loc[
            split_df["split"] == "validation",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    test_idx = (
        split_df.loc[
            split_df["split"] == "test",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    print(f"Train      : {len(train_idx)}")
    print(f"Validation : {len(val_idx)}")
    print(f"Test       : {len(test_idx)}")

    # ========================================================
    # 7. SAFETY CHECK FOR SPLITS
    # ========================================================

    if len(train_idx) != 504:
        raise ValueError(
            f"Expected 504 training samples, "
            f"found {len(train_idx)}."
        )

    if len(val_idx) != 108:
        raise ValueError(
            f"Expected 108 validation samples, "
            f"found {len(val_idx)}."
        )

    if len(test_idx) != 108:
        raise ValueError(
            f"Expected 108 test samples, "
            f"found {len(test_idx)}."
        )

    # ========================================================
    # 8. SPLIT DATA
    # ========================================================

    X_train = X_selected[train_idx]
    y_train = y[train_idx]

    X_val = X_selected[val_idx]
    y_val = y[val_idx]

    X_test = X_selected[test_idx]
    y_test = y[test_idx]

    # ========================================================
    # 9. VERIFY CLASS DISTRIBUTION
    # ========================================================

    print("\nClass distribution:")

    print("\nTraining:")
    print(pd.Series(y_train).value_counts().sort_index())

    print("\nValidation:")
    print(pd.Series(y_val).value_counts().sort_index())

    print("\nTest:")
    print(pd.Series(y_test).value_counts().sort_index())

    # ========================================================
    # 10. STANDARDIZATION
    # ========================================================
    #
    # IMPORTANT:
    # Fit the scaler ONLY on training data.
    #
    # Validation and test are transformed using parameters
    # learned from training data.
    # ========================================================

    print("\nStandardizing features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    # ========================================================
    # 11. TRAIN LOGISTIC REGRESSION
    # ========================================================

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # 12. TEST PREDICTION
    # ========================================================

    y_test_pred = model.predict(
        X_test_scaled
    )

    # ========================================================
    # 13. HELD-OUT TEST METRICS
    # ========================================================

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    test_balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_test_pred
    )

    test_macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro"
    )

    print("\n" + "-" * 70)
    print("HELD-OUT TEST RESULTS")
    print("-" * 70)

    print(
        f"Test accuracy             : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Test balanced accuracy    : "
        f"{test_balanced_accuracy:.4f} "
        f"({test_balanced_accuracy * 100:.2f}%)"
    )

    print(
        f"Test macro F1             : "
        f"{test_macro_f1:.4f}"
    )

    # ========================================================
    # 14. CONFUSION MATRIX
    # ========================================================
    #
    # Use the actual string labels from the dataset.
    # This avoids the int/string mismatch from the previous
    # version of Step 47.
    # ========================================================

    class_labels = sorted(
        pd.unique(y)
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=class_labels
    )

    print("\nConfusion Matrix:")

    cm_display = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    print(cm_display)

    # ========================================================
    # 15. PER-CLASS ACCURACY
    # ========================================================

    print("\nPer-class test accuracy:")

    per_class_results = []

    for class_label in class_labels:

        class_mask = (
            y_test == class_label
        )

        class_total = int(
            class_mask.sum()
        )

        class_correct = int(
            (
                y_test_pred[class_mask]
                == class_label
            ).sum()
        )

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0.0
        )

        print(
            f"{class_label:<35} "
            f"{class_correct:>3}/{class_total:<3} "
            f"({class_accuracy * 100:.2f}%)"
        )

        per_class_results.append({
            "label": class_label,
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_accuracy,
        })

    # ========================================================
    # 16. CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        y_test_pred,
        labels=class_labels,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )

    print(report)

    # ========================================================
    # 17. SAVE TEST METRICS
    # ========================================================

    metrics_df = pd.DataFrame([{
        "top_k": TOP_K,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_macro_f1": test_macro_f1,
        "chance_accuracy": 1 / len(class_labels),
    }])

    metrics_df.to_csv(
        OUTPUT_DIR
        / "step47_test_metrics.csv",
        index=False
    )

    # ========================================================
    # 18. SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    cm_df.to_csv(
        OUTPUT_DIR
        / "step47_confusion_matrix.csv"
    )

    # ========================================================
    # 19. SAVE PER-CLASS RESULTS
    # ========================================================

    per_class_df = pd.DataFrame(
        per_class_results
    )

    per_class_df.to_csv(
        OUTPUT_DIR
        / "step47_per_class_accuracy.csv",
        index=False
    )

    # ========================================================
    # 20. SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "dataset_index": test_idx,
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })

    predictions_df.to_csv(
        OUTPUT_DIR
        / "step47_test_predictions.csv",
        index=False
    )

    # ========================================================
    # 21. SAVE SELECTED FEATURES
    # ========================================================

    selected_features_df = pd.DataFrame({
        "rank": np.arange(
            1,
            TOP_K + 1
        ),
        "feature": top_features,
    })

    selected_features_df.to_csv(
        OUTPUT_DIR
        / "step47_selected_top100_features.csv",
        index=False
    )

    # ========================================================
    # 22. SAVE CLASSIFICATION REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "step47_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # ========================================================
    # 23. FINAL METHODOLOGICAL NOTE
    # ========================================================

    print("\n" + "=" * 70)
    print("METHODOLOGICAL NOTE")
    print("=" * 70)

    print(
        "\nTop 100 features were selected using "
        "validation performance in Step 46."
    )

    print(
        "The feature ranking was generated using "
        "training data only."
    )

    print(
        "The test set was NOT used during feature ranking "
        "or K selection."
    )

    print(
        "The scaler was fitted only on the training data."
    )

    print(
        "The test set was used only for this final "
        "held-out evaluation."
    )

    # ========================================================
    # 24. FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 47 COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal held-out test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Final held-out balanced accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Final held-out macro F1: "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()