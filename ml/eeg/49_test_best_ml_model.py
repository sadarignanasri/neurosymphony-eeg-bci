"""
STEP 49 — HELD-OUT TEST EVALUATION OF BEST ML MODEL
====================================================

Best model selected in Step 48:
    Logistic Regression

Feature representation:
    320 absolute band-power features
    64 channels × 5 frequency bands

The test set is used ONLY in this step.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
    / "step44_sub01_band_power_only.csv"
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
    / "step49"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 49 — HELD-OUT TEST EVALUATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\nLoading feature matrix...")

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # Actual metadata columns from Step 44
    metadata_columns = [
        "subject",
        "trial",
        "label",
        "label_name",
        "event_row",
        "start_sample",
        "duration_seconds",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    # ========================================================
    # 2. VERIFY FEATURES
    # ========================================================

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != 320:

        raise ValueError(
            f"Expected 320 band-power features, "
            f"found {len(feature_columns)}."
        )

    X = df[feature_columns].values

    y = df["label"].values

    print(
        "Verified: 320 absolute band-power features."
    )

    print(
        f"Number of classes: "
        f"{len(np.unique(y))}"
    )

    # ========================================================
    # 3. LOAD EXACT STEP 45 SPLIT
    # ========================================================

    print("\nLoading Step 45 split...")

    split_df = pd.read_csv(
        SPLIT_FILE
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

    print(
        f"Train      : {len(train_idx)}"
    )

    print(
        f"Validation : {len(val_idx)}"
    )

    print(
        f"Test       : {len(test_idx)}"
    )

    # ========================================================
    # 4. SAFETY CHECK
    # ========================================================

    if len(train_idx) != 504:
        raise ValueError(
            "Expected 504 training samples."
        )

    if len(val_idx) != 108:
        raise ValueError(
            "Expected 108 validation samples."
        )

    if len(test_idx) != 108:
        raise ValueError(
            "Expected 108 test samples."
        )

    # ========================================================
    # 5. CREATE DATA SPLITS
    # ========================================================

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # ========================================================
    # 6. STANDARDIZATION
    # ========================================================

    print(
        "\nStandardizing features..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "Scaler fitted on training data only."
    )

    # ========================================================
    # 7. TRAIN BEST MODEL
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

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
    # 8. PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(
        X_train_scaled
    )

    y_val_pred = model.predict(
        X_val_scaled
    )

    y_test_pred = model.predict(
        X_test_scaled
    )

    # ========================================================
    # 9. TRAIN / VALIDATION / TEST METRICS
    # ========================================================

    train_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    validation_accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_test_pred
        )
    )

    test_macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro"
    )

    # ========================================================
    # 10. DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BEST MODEL — LOGISTIC REGRESSION")
    print("=" * 70)

    print(
        f"Train accuracy       : "
        f"{train_accuracy:.4f} "
        f"({train_accuracy * 100:.2f}%)"
    )

    print(
        f"Validation accuracy  : "
        f"{validation_accuracy:.4f} "
        f"({validation_accuracy * 100:.2f}%)"
    )

    print(
        f"Test accuracy        : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Test balanced        : "
        f"{test_balanced_accuracy:.4f} "
        f"({test_balanced_accuracy * 100:.2f}%)"
    )

    print(
        f"Test macro F1        : "
        f"{test_macro_f1:.4f}"
    )

    # ========================================================
    # 11. CONFUSION MATRIX
    # ========================================================

    class_labels = sorted(
        np.unique(y)
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=class_labels
    )

    print("\nConfusion Matrix:")

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    print(
        cm_df.to_string()
    )

    # ========================================================
    # 12. PER-CLASS ACCURACY
    # ========================================================

    print(
        "\nPer-class test accuracy:"
    )

    per_class_results = []

    for class_label in class_labels:

        mask = (
            y_test == class_label
        )

        total = int(
            mask.sum()
        )

        correct = int(
            (
                y_test_pred[mask]
                == class_label
            ).sum()
        )

        class_accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        print(
            f"{class_label:<35} "
            f"{correct:>2}/{total:<2} "
            f"({class_accuracy * 100:.2f}%)"
        )

        per_class_results.append({
            "label": class_label,
            "correct": correct,
            "total": total,
            "accuracy": class_accuracy,
        })

    # ========================================================
    # 13. CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

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
    # 14. SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "features": 320,
            "train_accuracy": train_accuracy,
            "validation_accuracy":
                validation_accuracy,
            "test_accuracy":
                test_accuracy,
            "test_balanced_accuracy":
                test_balanced_accuracy,
            "test_macro_f1":
                test_macro_f1,
            "chance_accuracy":
                1 / len(class_labels),
        }
    ])

    metrics_df.to_csv(
        OUTPUT_DIR
        / "step49_test_metrics.csv",
        index=False
    )

    # ========================================================
    # 15. SAVE CONFUSION MATRIX
    # ========================================================

    cm_df.to_csv(
        OUTPUT_DIR
        / "step49_confusion_matrix.csv"
    )

    # ========================================================
    # 16. SAVE PER-CLASS ACCURACY
    # ========================================================

    pd.DataFrame(
        per_class_results
    ).to_csv(
        OUTPUT_DIR
        / "step49_per_class_accuracy.csv",
        index=False
    )

    # ========================================================
    # 17. SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "dataset_index": test_idx,
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })

    predictions_df.to_csv(
        OUTPUT_DIR
        / "step49_test_predictions.csv",
        index=False
    )

    # ========================================================
    # 18. SAVE CLASSIFICATION REPORT
    # ========================================================

    with open(
        OUTPUT_DIR
        / "step49_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # ========================================================
    # 19. METHODOLOGICAL NOTE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("METHODOLOGICAL NOTE")
    print("=" * 70)

    print(
        "\nLogistic Regression was selected as the best model "
        "using validation performance in Step 48."
    )

    print(
        "The model uses all 320 absolute band-power features."
    )

    print(
        "The scaler was fitted only on the training data."
    )

    print(
        "The test set was not used for model selection."
    )

    print(
        "The test set was used only for this final "
        "held-out evaluation."
    )

    # ========================================================
    # 20. COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 49 COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal TEST ACCURACY: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Final TEST BALANCED ACCURACY: "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Final TEST MACRO F1: "
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