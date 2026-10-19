"""
Neuro-Symphony
============================================================
DS004514 EEG Baseline Machine Learning
============================================================

Purpose:
    Establish a baseline classification performance using the
    64 EEG channel values from the approved DS004514 CSV.

Input:
    data/processed/ds004514_eeg_approved.csv

Features:
    A1-A32
    B1-B32

Target:
    label

Human-readable target:
    label_name

Validation:
    Leave-One-Subject-Out Cross-Validation (LOSO)

Models:
    1. Logistic Regression
    2. Linear SVM
    3. LDA
    4. KNN
    5. Random Forest
    6. Extra Trees

IMPORTANT:
    - subject_id is used ONLY for grouping.
    - trial_id is NOT used as an ML feature.
    - label_name is NOT used as an ML feature.
    - label is the target.
    - Scaling is performed INSIDE the ML pipeline.
    - The approved CSV is never modified.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    +
    [f"B{i}" for i in range(1, 33)]
)

TARGET_COLUMN = "label"

LABEL_NAME_COLUMN = "label_name"

SUBJECT_COLUMN = "subject_id"


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 75)
print("NEURO-SYMPHONY")
print("DS004514 EEG BASELINE MACHINE LEARNING")
print("=" * 75)

print("\nLoading dataset...")

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("✓ Dataset loaded successfully.")

print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. VERIFY DATASET
# ============================================================

print("\n" + "=" * 75)
print("DATASET VERIFICATION")
print("=" * 75)

print(f"Samples       : {len(df)}")
print(f"EEG features  : {len(EEG_CHANNELS)}")
print(f"Subjects      : {df[SUBJECT_COLUMN].nunique()}")
print(f"Classes       : {df[TARGET_COLUMN].nunique()}")

missing = df[EEG_CHANNELS + [TARGET_COLUMN]].isnull().sum().sum()

print(f"Missing values: {missing}")

if len(EEG_CHANNELS) != 64:

    raise ValueError(
        f"Expected 64 EEG channels, found {len(EEG_CHANNELS)}"
    )

if df[TARGET_COLUMN].nunique() != 8:

    raise ValueError(
        "Expected exactly 8 target classes."
    )

if missing != 0:

    raise ValueError(
        "Missing values detected in ML data."
    )

print("✓ Dataset verification passed.")


# ============================================================
# 5. CREATE X, y AND GROUPS
# ============================================================

X = df[EEG_CHANNELS].copy()

y = df[TARGET_COLUMN].copy()

groups = df[SUBJECT_COLUMN].copy()


print("\n" + "=" * 75)
print("ML DATA")
print("=" * 75)

print(f"X shape      : {X.shape}")
print(f"y shape      : {y.shape}")
print(f"Groups       : {groups.nunique()} subjects")

print("\nTarget distribution:")

print(
    y.value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 6. LABEL MAPPING
# ============================================================

print("\n" + "=" * 75)
print("LABEL MAPPING")
print("=" * 75)

label_mapping = (
    df[
        [
            TARGET_COLUMN,
            LABEL_NAME_COLUMN
        ]
    ]
    .drop_duplicates()
    .sort_values(TARGET_COLUMN)
)

print(
    label_mapping.to_string(index=False)
)


# ============================================================
# 7. DEFINE LOSO
# ============================================================

logo = LeaveOneGroupOut()

number_of_folds = logo.get_n_splits(
    X,
    y,
    groups
)

print("\n" + "=" * 75)
print("VALIDATION STRATEGY")
print("=" * 75)

print(
    "Leave-One-Subject-Out Cross-Validation"
)

print(
    f"Number of folds: {number_of_folds}"
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    ),

    "Linear SVM": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    random_state=42
                )
            )
        ]
    ),

    "LDA": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto"
                )
            )
        ]
    ),

    "KNN": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                )
            )
        ]
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 9. STORAGE FOR RESULTS
# ============================================================

all_results = []

all_predictions = {}


# ============================================================
# 10. TRAIN AND EVALUATE EACH MODEL
# ============================================================

for model_name, model in models.items():

    print("\n")
    print("=" * 75)
    print(f"MODEL: {model_name}")
    print("=" * 75)

    start_time = time.time()

    true_labels = []

    predicted_labels = []

    fold_accuracies = []

    fold_number = 0

    for train_index, test_index in logo.split(
        X,
        y,
        groups
    ):

        fold_number += 1

        X_train = X.iloc[train_index]

        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]

        test_subject = groups.iloc[
            test_index
        ].iloc[0]

        print(
            f"\nFold {fold_number:02d}/{number_of_folds}"
            f" | Test subject: {test_subject}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Fold accuracy
        # ----------------------------------------------------

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_accuracies.append(
            fold_accuracy
        )

        print(
            f"Fold accuracy: "
            f"{fold_accuracy * 100:.2f}%"
        )

        true_labels.extend(
            y_test.tolist()
        )

        predicted_labels.extend(
            y_pred.tolist()
        )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    mean_fold_accuracy = np.mean(
        fold_accuracies
    )

    std_fold_accuracy = np.std(
        fold_accuracies
    )

    elapsed_time = (
        time.time()
        - start_time
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    all_predictions[model_name] = {
        "true": true_labels,
        "predicted": predicted_labels
    }


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 75)
    print("MODEL RESULTS")
    print("-" * 75)

    print(
        f"Overall Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision        : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Macro Recall           : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1-score         : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Mean Fold Accuracy     : "
        f"{mean_fold_accuracy * 100:.2f}%"
    )

    print(
        f"Fold Accuracy Std      : "
        f"{std_fold_accuracy * 100:.2f}%"
    )

    print(
        f"Training Time          : "
        f"{elapsed_time:.2f} seconds"
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    all_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "mean_fold_accuracy": mean_fold_accuracy,
            "std_fold_accuracy": std_fold_accuracy,
            "training_time_seconds": elapsed_time
        }
    )


# ============================================================
# 11. CREATE RESULTS TABLE
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "accuracy",
    ascending=False
)

display_columns = [
    "model",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mean_fold_accuracy",
    "std_fold_accuracy"
]

display_df = results_df[
    display_columns
].copy()

for column in [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mean_fold_accuracy",
    "std_fold_accuracy"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)

print(
    display_df.to_string(index=False)
)


# ============================================================
# 12. SAVE MODEL RESULTS
# ============================================================

results_path = (
    RESULTS_DIR
    / "baseline_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\n✓ Saved: {results_path}"
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

prediction_rows = []

for model_name, prediction_data in all_predictions.items():

    for true_value, predicted_value in zip(
        prediction_data["true"],
        prediction_data["predicted"]
    ):

        prediction_rows.append(
            {
                "model": model_name,
                "true_label": true_value,
                "predicted_label": predicted_value
            }
        )


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = (
    RESULTS_DIR
    / "baseline_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(
    f"✓ Saved: {predictions_path}"
)


# ============================================================
# 14. BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

print("\n" + "=" * 75)
print("BEST BASELINE MODEL")
print("=" * 75)

print(
    f"Model   : {best_row['model']}"
)

print(
    f"Accuracy: "
    f"{best_row['accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{best_row['f1_macro'] * 100:.2f}%"
)


# ============================================================
# 15. CHANCE LEVEL
# ============================================================

chance_accuracy = 1 / 8

print("\n" + "=" * 75)
print("CHANCE LEVEL")
print("=" * 75)

print(
    f"8-class chance accuracy: "
    f"{chance_accuracy * 100:.2f}%"
)

accuracy_improvement = (
    best_row["accuracy"]
    - chance_accuracy
)

print(
    f"Best model improvement "
    f"over chance: "
    f"{accuracy_improvement * 100:.2f} percentage points"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("BASELINE ML TRAINING COMPLETED")
print("=" * 75)

print(
    "\nThe raw 64-channel EEG baseline has been evaluated "
    "using subject-independent validation."
)

print(
    "\nNext stage:"
)

print(
    "EEG feature engineering and improved representations."
)