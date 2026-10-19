"""
Neuro-Symphony
============================================================
DS004514 EEG — Engineered Feature ML Experiment
============================================================

Purpose:
    Evaluate whether engineered spatial/statistical EEG
    features improve classification over the raw-channel
    baseline.

Validation:
    Leave-One-Subject-Out (LOSO)

Models:
    1. Logistic Regression
    2. Linear SVM
    3. LDA
    4. KNN
    5. Random Forest
    6. Extra Trees

IMPORTANT:
    subject_id is used ONLY for grouping.
    It is NEVER used as an ML feature.

    label_name is metadata.
    label is the target.

Outputs:
    results/eeg/engineered_model_results.csv
    results/eeg/engineered_predictions.csv
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ds004514_eeg_engineered.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "eeg"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    RESULTS_DIR
    / "engineered_model_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "engineered_predictions.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

METADATA_COLUMNS = [
    "subject_id",
    "trial_id",
    "label_name",
]

TARGET_COLUMN = "label"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("NEURO-SYMPHONY")
print("DS004514 EEG — ENGINEERED FEATURE ML")
print("=" * 80)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading engineered dataset...")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nEngineered dataset not found:\n"
        f"{DATA_PATH}\n\n"
        f"Run 03_feature_engineering.py first."
    )

df = pd.read_csv(DATA_PATH)

print("✓ Dataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# DATA VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

required_columns = (
    METADATA_COLUMNS
    + [TARGET_COLUMN]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("✓ Required metadata and target columns found.")

print(f"Samples       : {len(df)}")
print(f"Subjects      : {df['subject_id'].nunique()}")
print(f"Classes       : {df[TARGET_COLUMN].nunique()}")


# ============================================================
# IDENTIFY ML FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in METADATA_COLUMNS
    and column != TARGET_COLUMN
]

X = df[feature_columns].copy()

y = df[TARGET_COLUMN].copy()

groups = df["subject_id"].copy()

print(f"ML features   : {len(feature_columns)}")


# ============================================================
# FEATURE GROUP INFORMATION
# ============================================================

print("\nFeature groups detected:")


raw_features = [
    column
    for column in feature_columns
    if (
        column.startswith("A")
        and column[1:].isdigit()
        and 1 <= int(column[1:]) <= 32
    )
    or (
        column.startswith("B")
        and column[1:].isdigit()
        and 1 <= int(column[1:]) <= 32
    )
]


z_features = [
    column
    for column in feature_columns
    if column.endswith("_z")
]


stat_features = [
    column
    for column in feature_columns
    if column not in raw_features
    and column not in z_features
]


print(
    f"  Raw EEG features       : "
    f"{len(raw_features)}"
)

print(
    f"  Statistical/comparison : "
    f"{len(stat_features)}"
)

print(
    f"  Normalized EEG         : "
    f"{len(z_features)}"
)


# ============================================================
# FEATURE MATRIX SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE MATRIX SANITY CHECK")
print("=" * 80)


missing_values = X.isnull().sum().sum()

if missing_values == 0:

    print("✓ No missing values")

else:

    print(
        f"⚠ Missing values detected: "
        f"{missing_values}"
    )


numeric_values = (
    X
    .select_dtypes(include=np.number)
    .to_numpy()
)


if np.isfinite(numeric_values).all():

    print("✓ No infinite values")

else:

    raise ValueError(
        "Infinite values detected in feature matrix."
    )


if X.shape[0] != y.shape[0]:

    raise ValueError(
        "X and y have different numbers of samples."
    )


if X.shape[0] != groups.shape[0]:

    raise ValueError(
        "X and groups have different numbers of samples."
    )


print(
    "✓ Feature/target/group dimensions consistent."
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

class_counts = (
    y
    .value_counts()
    .sort_index()
)

for label, count in class_counts.items():

    print(
        f"Class {label}: {count} samples"
    )


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("SUBJECT DISTRIBUTION")
print("=" * 80)

subject_counts = (
    groups
    .value_counts()
    .sort_index()
)

for subject, count in subject_counts.items():

    print(
        f"{subject}: {count} samples"
    )


# ============================================================
# LOSO VALIDATION
# ============================================================

logo = LeaveOneGroupOut()

subjects = sorted(
    groups.unique()
)


print("\n" + "=" * 80)
print("VALIDATION STRATEGY")
print("=" * 80)

print(
    "Method: Leave-One-Subject-Out (LOSO)"
)

print(
    f"Subjects: {len(subjects)}"
)

print(
    f"LOSO folds: {len(subjects)}"
)

print("\nEach fold:")

print(
    "  Training = 11 subjects"
)

print(
    "  Testing  = 1 completely unseen subject"
)


# ============================================================
# MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
        ),
    ]),


    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    "Linear SVM": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LinearSVC(
                C=1.0,
                max_iter=5000,
                random_state=42,
            )
        ),
    ]),


    # --------------------------------------------------------
    # LDA
    # --------------------------------------------------------

    "LDA": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )
        ),
    ]),


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    "KNN": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            )
        ),
    ]),


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "Random Forest": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
        ),
    ]),


    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------

    "Extra Trees": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
        ),
    ]),
}


# ============================================================
# TRAINING
# ============================================================

all_results = []

all_predictions = []


print("\n" + "=" * 80)
print("STARTING LOSO TRAINING")
print("=" * 80)


overall_start = time.time()


for model_name, model in models.items():

    print("\n" + "-" * 80)
    print(f"MODEL: {model_name}")
    print("-" * 80)

    model_start = time.time()

    y_true_all = []

    y_pred_all = []

    fold_results = []


    # ========================================================
    # LOSO FOLDS
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        logo.split(
            X,
            y,
            groups
        ),
        start=1,
    ):

        test_subject = (
            groups
            .iloc[test_idx]
            .iloc[0]
        )


        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]


        print(
            f"Fold {fold:02d}/{len(subjects)} "
            f"| Test subject: {test_subject}",
            end=" ... ",
        )


        # ----------------------------------------------------
        # Clone model
        # ----------------------------------------------------

        fold_model = clone(model)


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        fold_model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = (
            fold_model
            .predict(X_test)
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )


        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )


        balanced_acc = (
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        y_true_all.extend(
            y_test.tolist()
        )

        y_pred_all.extend(
            predictions.tolist()
        )


        # ----------------------------------------------------
        # Store fold results
        # ----------------------------------------------------

        fold_results.append({

            "fold": fold,

            "test_subject": test_subject,

            "accuracy": accuracy,

            "macro_f1": macro_f1,

            "balanced_accuracy": balanced_acc,

        })


        # ----------------------------------------------------
        # Save individual predictions
        # ----------------------------------------------------

        for idx, true_value, pred_value in zip(
            test_idx,
            y_test,
            predictions,
        ):

            all_predictions.append({

                "model": model_name,

                "subject_id":
                    df.iloc[idx]["subject_id"],

                "trial_id":
                    df.iloc[idx]["trial_id"],

                "true_label":
                    int(true_value),

                "predicted_label":
                    int(pred_value),

            })


        print(
            f"Accuracy={accuracy:.4f} "
            f"| F1={macro_f1:.4f}"
        )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    overall_accuracy = accuracy_score(
        y_true_all,
        y_pred_all
    )


    overall_f1 = f1_score(
        y_true_all,
        y_pred_all,
        average="macro",
        zero_division=0,
    )


    overall_balanced_accuracy = (
        balanced_accuracy_score(
            y_true_all,
            y_pred_all
        )
    )


    fold_accuracies = [
        result["accuracy"]
        for result in fold_results
    ]


    fold_f1 = [
        result["macro_f1"]
        for result in fold_results
    ]


    elapsed = (
        time.time()
        - model_start
    )


    # ========================================================
    # PRINT MODEL RESULTS
    # ========================================================

    print("\nOverall results:")


    print(
        f"Accuracy           : "
        f"{overall_accuracy:.4f} "
        f"({overall_accuracy * 100:.2f}%)"
    )


    print(
        f"Macro F1           : "
        f"{overall_f1:.4f} "
        f"({overall_f1 * 100:.2f}%)"
    )


    print(
        f"Balanced Accuracy  : "
        f"{overall_balanced_accuracy:.4f} "
        f"({overall_balanced_accuracy * 100:.2f}%)"
    )


    print(
        f"Fold Accuracy Mean : "
        f"{np.mean(fold_accuracies):.4f}"
    )


    print(
        f"Fold Accuracy Std  : "
        f"{np.std(fold_accuracies):.4f}"
    )


    print(
        f"Training time      : "
        f"{elapsed:.2f} seconds"
    )


    # ========================================================
    # STORE MODEL RESULT
    # ========================================================

    all_results.append({

        "model":
            model_name,

        "accuracy":
            overall_accuracy,

        "macro_f1":
            overall_f1,

        "balanced_accuracy":
            overall_balanced_accuracy,

        "fold_accuracy_mean":
            np.mean(fold_accuracies),

        "fold_accuracy_std":
            np.std(fold_accuracies),

        "fold_f1_mean":
            np.mean(fold_f1),

        "training_time_seconds":
            elapsed,

    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SORT MODELS
# ============================================================

results_df = (
    results_df
    .sort_values(
        by="accuracy",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# CHANCE LEVEL
# ============================================================

number_of_classes = (
    df[TARGET_COLUMN]
    .nunique()
)

chance_accuracy = (
    1 / number_of_classes
)


results_df[
    "chance_accuracy"
] = chance_accuracy


results_df[
    "improvement_over_chance"
] = (
    results_df["accuracy"]
    - chance_accuracy
)


results_df[
    "improvement_percentage_points"
] = (
    results_df[
        "improvement_over_chance"
    ]
    * 100
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


predictions_df = pd.DataFrame(
    all_predictions
)


predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

overall_elapsed = (
    time.time()
    - overall_start
)


print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)


display_columns = [

    "model",

    "accuracy",

    "macro_f1",

    "balanced_accuracy",

    "improvement_percentage_points",

]


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = (
    results_df
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST ENGINEERED-FEATURE MODEL")
print("=" * 80)


print(
    f"Model               : "
    f"{best_model['model']}"
)


print(
    f"Accuracy            : "
    f"{best_model['accuracy'] * 100:.2f}%"
)


print(
    f"Macro F1            : "
    f"{best_model['macro_f1'] * 100:.2f}%"
)


print(
    f"Balanced Accuracy   : "
    f"{best_model['balanced_accuracy'] * 100:.2f}%"
)


print(
    f"Improvement vs "
    f"{chance_accuracy * 100:.1f}% chance: "
    f"{best_model['improvement_percentage_points']:.2f} "
    f"percentage points"
)


print(
    f"\nTotal experiment time: "
    f"{overall_elapsed:.2f} seconds"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    f"✓ {RESULTS_PATH}"
)


print(
    f"✓ {PREDICTIONS_PATH}"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("ENGINEERED FEATURE ML EXPERIMENT COMPLETED")
print("=" * 80)