"""
STEP 50 — TEST ALL ML MODELS

Purpose:
    Evaluate all five ML models on the SAME untouched test set.

Dataset:
    Step 44 — sub-01 absolute band-power features

Features:
    320 = 64 EEG channels × 5 frequency bands

Split:
    Step 45 fixed split
    Train      = 504
    Validation = 108
    Test       = 108

Important:
    - Scaler fitted ONLY on training data.
    - Test data is used only for final evaluation.
    - No model is selected using test performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

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
    / "step50"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

METADATA_COLUMNS = [
    "subject",
    "trial",
    "label",
    "label_name",
    "event_row",
    "start_sample",
    "duration_seconds",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 50 — TEST ALL ML MODELS")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature matrix...")

df = pd.read_csv(FEATURE_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

feature_columns = [
    col for col in df.columns
    if col not in METADATA_COLUMNS
]

print(f"Total features: {len(feature_columns)}")

if len(feature_columns) != 320:
    raise ValueError(
        f"Expected 320 absolute band-power features, "
        f"found {len(feature_columns)}."
    )

print("Verified: 320 absolute band-power features.")


# ============================================================
# LABELS
# ============================================================

X = df[feature_columns].values
y = df["label"].values

class_labels = sorted(np.unique(y))

print(f"Number of classes: {len(class_labels)}")

if len(class_labels) != 8:
    raise ValueError(
        f"Expected 8 classes, found {len(class_labels)}."
    )


# ============================================================
# LOAD SPLIT
# ============================================================

print("\nLoading Step 45 split...")

split_df = pd.read_csv(SPLIT_FILE)

if "dataset_index" not in split_df.columns:
    raise ValueError(
        "Expected 'dataset_index' column in split file."
    )

if "split" not in split_df.columns:
    raise ValueError(
        "Expected 'split' column in split file."
    )


# ============================================================
# CREATE SPLIT MASKS
# ============================================================

train_indices = split_df.loc[
    split_df["split"] == "train",
    "dataset_index"
].astype(int).values

val_indices = split_df.loc[
    split_df["split"] == "validation",
    "dataset_index"
].astype(int).values

test_indices = split_df.loc[
    split_df["split"] == "test",
    "dataset_index"
].astype(int).values


print(f"Train      : {len(train_indices)}")
print(f"Validation : {len(val_indices)}")
print(f"Test       : {len(test_indices)}")


# ============================================================
# EXTRACT DATA
# ============================================================

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print("Scaler fitted on training data only.")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    ),

    "Linear SVM": SVC(
        kernel="linear",
        random_state=RANDOM_STATE
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
}


# ============================================================
# STORAGE
# ============================================================

results = []

all_predictions = []


# ============================================================
# TRAIN + TEST ALL MODELS
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print("Training...")

    model.fit(X_train_scaled, y_train)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_pred = model.predict(X_train_scaled)

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_pred = model.predict(X_val_scaled)

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_balanced = balanced_accuracy_score(
        y_val,
        val_pred
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred = model.predict(X_test_scaled)

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_balanced = balanced_accuracy_score(
        y_test,
        test_pred
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train accuracy       : "
        f"{train_accuracy:.4f} "
        f"({train_accuracy * 100:.2f}%)"
    )

    print(
        f"Validation accuracy  : "
        f"{val_accuracy:.4f} "
        f"({val_accuracy * 100:.2f}%)"
    )

    print(
        f"Test accuracy        : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Test balanced        : "
        f"{test_balanced:.4f} "
        f"({test_balanced * 100:.2f}%)"
    )

    print(
        f"Test macro F1         : "
        f"{test_macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({
        "model": model_name,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "validation_balanced_accuracy": val_balanced,
        "validation_macro_f1": val_macro_f1,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced,
        "test_macro_f1": test_macro_f1,
    })

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    for i, (true_label, predicted_label) in enumerate(
        zip(y_test, test_pred)
    ):

        all_predictions.append({
            "model": model_name,
            "test_index": int(test_indices[i]),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": bool(
                true_label == predicted_label
            ),
        })

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=class_labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    cm_df.to_csv(
        OUTPUT_DIR
        / f"step50_confusion_matrix_{safe_name}.csv"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        test_pred,
        labels=class_labels,
        target_names=class_labels,
        zero_division=0
    )

    with open(
        OUTPUT_DIR
        / f"step50_classification_report_{safe_name}.txt",
        "w"
    ) as f:

        f.write(report)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="test_accuracy",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR
    / "step50_all_models_test_results.csv",
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    OUTPUT_DIR
    / "step50_test_predictions.csv",
    index=False
)


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

experiment_info = pd.DataFrame({
    "parameter": [
        "dataset",
        "subjects",
        "features",
        "number_of_features",
        "classes",
        "train_samples",
        "validation_samples",
        "test_samples",
        "split_source",
        "scaler",
        "random_state",
        "test_used_for_model_selection",
    ],
    "value": [
        "DS004514 sub-01",
        "sub-01",
        "absolute band power",
        320,
        8,
        len(train_indices),
        len(val_indices),
        len(test_indices),
        "Step 45 fixed split",
        "StandardScaler fitted on training only",
        RANDOM_STATE,
        "No",
    ]
})

experiment_info.to_csv(
    OUTPUT_DIR
    / "step50_experiment_info.csv",
    index=False
)


# ============================================================
# FINAL RANKING
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST MODEL RANKING")
print("=" * 70)

for rank, (_, row) in enumerate(
    results_df.iterrows(),
    start=1
):

    print(
        f"{rank}. "
        f"{row['model']:<22} "
        f"Test Accuracy: "
        f"{row['test_accuracy'] * 100:.2f}% | "
        f"Balanced: "
        f"{row['test_balanced_accuracy'] * 100:.2f}% | "
        f"Macro F1: "
        f"{row['test_macro_f1']:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 50 COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nMain results file:")
print(
    OUTPUT_DIR
    / "step50_all_models_test_results.csv"
)