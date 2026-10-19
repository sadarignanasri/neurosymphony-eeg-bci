"""
NEURO-SYMPHONY
fNIRS WITHIN-SUBJECT CLASSIFICATION

Purpose:
    Evaluate whether the engineered fNIRS representation
    can distinguish the 8 mental-task classes when training
    and testing within the same subject.

Evaluation:
    - 12 subjects
    - Stratified 80/20 train-test split per subject
    - 8-class classification
    - Accuracy
    - Macro F1
    - Balanced Accuracy

Models:
    - Logistic Regression
    - Linear SVM
    - LDA
    - KNN
    - Random Forest
    - Extra Trees
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# SUPPRESS NON-CRITICAL WARNINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/features/ds004514_fnirs_engineered.csv"

OUTPUT_DIR = "results/fnirs"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "within_subject_model_results.csv"
)

SUBJECT_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "within_subject_subject_results.csv"
)

BEST_SUBJECT_FILE = os.path.join(
    OUTPUT_DIR,
    "within_subject_best_per_subject.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "within_subject_predictions.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("NEURO-SYMPHONY — fNIRS WITHIN-SUBJECT MODELS")
print("=" * 70)

start_time = time.time()

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n"
        "\nMake sure you are running this command "
        "from the neuro-symphony-bci project root."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# STANDARDIZE LABEL COLUMN
# ============================================================

if "label" not in df.columns:

    if "Label" in df.columns:

        df = df.rename(
            columns={
                "Label": "label"
            }
        )

        print("\nRenamed 'Label' → 'label'")

    elif "label_name" in df.columns:

        df = df.rename(
            columns={
                "label_name": "label"
            }
        )

        print("\nRenamed 'label_name' → 'label'")

    else:

        raise ValueError(
            "No label column found.\n"
            "Expected one of: "
            "'label', 'Label', or 'label_name'."
        )


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# Required metadata
# ------------------------------------------------------------

required_metadata = [
    "subject_id",
    "trial_id",
    "label",
    "label_name"
]


# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

for col in required_metadata:

    if col not in df.columns:

        raise ValueError(
            f"Missing required column: {col}"
        )


# ------------------------------------------------------------
# Subjects
# ------------------------------------------------------------

subjects = sorted(
    df["subject_id"].unique()
)


# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

classes = sorted(
    df["label"].unique()
)


# ------------------------------------------------------------
# Numeric ML feature selection
#
# IMPORTANT:
# label_name is metadata and must NOT enter the model.
#
# Only numeric columns are allowed as ML features.
# ------------------------------------------------------------

feature_columns = [
    col
    for col in df.columns
    if col not in required_metadata
    and pd.api.types.is_numeric_dtype(df[col])
]


# ------------------------------------------------------------
# Feature matrix
# ------------------------------------------------------------

X_all = df[feature_columns]


# ============================================================
# DATA VALIDATION OUTPUT
# ============================================================

print(
    f"Samples: {len(df)}"
)

print(
    f"Subjects: {len(subjects)}"
)

print(
    f"Classes: {len(classes)}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Missing values: "
    f"{int(X_all.isna().sum().sum())}"
)


# ------------------------------------------------------------
# Infinite values
# ------------------------------------------------------------

numeric_values = X_all.select_dtypes(
    include=np.number
)

infinite_values = np.isinf(
    numeric_values.to_numpy()
).sum()

print(
    f"Infinite values: {int(infinite_values)}"
)


# ------------------------------------------------------------
# Confirm every feature is numeric
# ------------------------------------------------------------

all_numeric = all(
    pd.api.types.is_numeric_dtype(
        df[col]
    )
    for col in feature_columns
)

print(
    f"All ML features numeric: {all_numeric}"
)


# ------------------------------------------------------------
# Show excluded metadata
# ------------------------------------------------------------

print("\nColumns excluded from ML:")

for col in df.columns:

    if col not in feature_columns:

        print(
            f"  {col}"
        )


# ============================================================
# EXPECTED FEATURE COUNT CHECK
# ============================================================

EXPECTED_FEATURES = 175

if len(feature_columns) != EXPECTED_FEATURES:

    print("\nWARNING:")
    print(
        f"Expected approximately "
        f"{EXPECTED_FEATURES} engineered features, "
        f"but found {len(feature_columns)}."
    )

    print(
        "The experiment will continue using only "
        "numeric feature columns."
    )

else:

    print(
        f"\nFeature count verified: "
        f"{EXPECTED_FEATURES} ML features"
    )


# ============================================================
# SUBJECT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT DISTRIBUTION")
print("=" * 70)

subject_counts = (
    df["subject_id"]
    .value_counts()
    .sort_index()
)


for subject, count in subject_counts.items():

    print(
        f"{subject}: "
        f"{count} trials"
    )


# ============================================================
# CLASS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)


for label, count in class_counts.items():

    print(
        f"{label}: "
        f"{count}"
    )


# ============================================================
# VALIDATE EACH SUBJECT HAS ALL CLASSES
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT × CLASS VALIDATION")
print("=" * 70)

subject_class_counts = pd.crosstab(
    df["subject_id"],
    df["label"]
)

print(
    subject_class_counts.to_string()
)


# ------------------------------------------------------------
# Check each subject has every class
# ------------------------------------------------------------

expected_classes = set(classes)

for subject in subjects:

    subject_classes = set(
        df.loc[
            df["subject_id"] == subject,
            "label"
        ].unique()
    )

    missing_classes = (
        expected_classes - subject_classes
    )

    if missing_classes:

        raise ValueError(
            f"{subject} is missing classes: "
            f"{sorted(missing_classes)}"
        )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
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
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    # --------------------------------------------------------
    # LINEAR SVM
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
            "model",
            SVC(
                kernel="linear",
                random_state=RANDOM_STATE
            )
        )
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
            "model",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            )
        )
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
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),


    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    "Extra Trees": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])
}


# ============================================================
# START WITHIN-SUBJECT EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("STARTING fNIRS WITHIN-SUBJECT CLASSIFICATION")
print("=" * 70)

print(
    f"\nSubjects: {len(subjects)}"
)

print(
    f"Classes: {len(classes)}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Train/Test split: "
    f"{int((1 - TEST_SIZE) * 100)}/"
    f"{int(TEST_SIZE * 100)}"
)

print(
    f"Random state: {RANDOM_STATE}"
)


# ============================================================
# STORAGE
# ============================================================

all_results = []

all_predictions = []


# ============================================================
# LOOP THROUGH MODELS
# ============================================================

for model_name, model in models.items():

    print("\n")

    print("-" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Loop through subjects
    # --------------------------------------------------------

    for subject in subjects:

        print(
            f"\nTesting subject: {subject}"
        )


        # ----------------------------------------------------
        # Select one subject
        # ----------------------------------------------------

        subject_df = df[
            df["subject_id"] == subject
        ].copy()


        # ----------------------------------------------------
        # Features and labels
        # ----------------------------------------------------

        X = subject_df[
            feature_columns
        ]

        y = subject_df[
            "label"
        ]


        # ----------------------------------------------------
        # Stratified 80/20 split
        #
        # subject_df.index is retained so that predictions
        # can be linked back to original trials.
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
            idx_train,
            idx_test
        ) = train_test_split(

            X,
            y,
            subject_df.index,

            test_size=TEST_SIZE,

            stratify=y,

            random_state=RANDOM_STATE
        )


        # ----------------------------------------------------
        # Print split information
        # ----------------------------------------------------

        print(
            f"  Train samples: "
            f"{len(X_train)}"
        )

        print(
            f"  Test samples:  "
            f"{len(X_test)}"
        )


        # ----------------------------------------------------
        # Validate train/test classes
        # ----------------------------------------------------

        train_classes = set(
            y_train.unique()
        )

        test_classes = set(
            y_test.unique()
        )


        if train_classes != expected_classes:

            raise ValueError(
                f"{subject} training split does not "
                f"contain all classes."
            )


        if test_classes != expected_classes:

            raise ValueError(
                f"{subject} testing split does not "
                f"contain all classes."
            )


        # ----------------------------------------------------
        # TRAIN
        #
        # Imputation and scaling happen inside the pipeline.
        # Therefore test information does not influence
        # preprocessing.
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )


        # ----------------------------------------------------
        # PRINT METRICS
        # ----------------------------------------------------

        print(
            f"  Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{macro_f1 * 100:.2f}%"
        )

        print(
            f"  Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        result = {

            "model": model_name,

            "subject_id": subject,

            "accuracy": accuracy,

            "macro_f1": macro_f1,

            "balanced_accuracy":
                balanced_accuracy,

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test)
        }


        all_results.append(
            result
        )


        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        for (
            original_idx,
            true_label,
            predicted_label
        ) in zip(
            idx_test,
            y_test,
            y_pred
        ):

            row = df.loc[
                original_idx
            ]


            all_predictions.append({

                "model":
                    model_name,

                "subject_id":
                    row["subject_id"],

                "trial_id":
                    row["trial_id"],

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label
            })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# MODEL-LEVEL SUMMARY
# ============================================================

model_summary = (

    results_df

    .groupby("model")

    .agg(

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        std_accuracy=(
            "accuracy",
            "std"
        ),

        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),

        std_macro_f1=(
            "macro_f1",
            "std"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),

        std_balanced_accuracy=(
            "balanced_accuracy",
            "std"
        )
    )

    .reset_index()
)


# ============================================================
# CHANCE LEVEL
# ============================================================

chance = 1 / len(classes)


model_summary[
    "improvement_over_chance"
] = (

    model_summary[
        "mean_accuracy"
    ]

    - chance
)


# ============================================================
# SORT MODELS BY ACCURACY
# ============================================================

model_summary = (
    model_summary
    .sort_values(
        "mean_accuracy",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE SUBJECT-LEVEL RESULTS
# ============================================================

results_df.to_csv(
    SUBJECT_RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

model_summary.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL fNIRS WITHIN-SUBJECT RESULTS")
print("=" * 70)

print(
    model_summary.to_string(
        index=False
    )
)


# ============================================================
# BEST OVERALL MODEL
# ============================================================

best_idx = (
    model_summary[
        "mean_accuracy"
    ].idxmax()
)


best_model = (
    model_summary.loc[
        best_idx
    ]
)


print("\n" + "=" * 70)
print("BEST WITHIN-SUBJECT fNIRS MODEL")
print("=" * 70)

print(
    f"Model: "
    f"{best_model['model']}"
)

print(
    f"Mean Accuracy: "
    f"{best_model['mean_accuracy'] * 100:.2f}%"
)

print(
    f"Std Accuracy: "
    f"{best_model['std_accuracy'] * 100:.2f}%"
)

print(
    f"Mean Macro F1: "
    f"{best_model['mean_macro_f1'] * 100:.2f}%"
)

print(
    f"Mean Balanced Accuracy: "
    f"{best_model['mean_balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Chance Level: "
    f"{chance * 100:.2f}%"
)

print(
    f"Improvement over chance: "
    f"{best_model['improvement_over_chance'] * 100:.2f} "
    f"percentage points"
)


# ============================================================
# BEST MODEL FOR EACH SUBJECT
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL FOR EACH SUBJECT")
print("=" * 70)


best_subject_models = []


for subject in subjects:

    subject_data = results_df[
        results_df["subject_id"] == subject
    ]


    best_idx = (
        subject_data[
            "accuracy"
        ].idxmax()
    )


    best = (
        subject_data.loc[
            best_idx
        ]
    )


    best_subject_models.append({

        "subject_id":
            subject,

        "best_model":
            best["model"],

        "accuracy":
            best["accuracy"],

        "macro_f1":
            best["macro_f1"],

        "balanced_accuracy":
            best["balanced_accuracy"]
    })


    print(
        f"{subject}: "
        f"{best['model']} — "
        f"{best['accuracy'] * 100:.2f}% "
        f"(F1: "
        f"{best['macro_f1'] * 100:.2f}%)"
    )


# ============================================================
# SAVE BEST MODEL PER SUBJECT
# ============================================================

best_subject_df = pd.DataFrame(
    best_subject_models
)


best_subject_df.to_csv(
    BEST_SUBJECT_FILE,
    index=False
)


# ============================================================
# SUBJECT PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT PERFORMANCE SUMMARY")
print("=" * 70)

print(
    best_subject_df.to_string(
        index=False
    )
)


# ============================================================
# OVERALL SUBJECT STATISTICS
# ============================================================

subject_mean_accuracy = (
    best_subject_df[
        "accuracy"
    ].mean()
)

subject_std_accuracy = (
    best_subject_df[
        "accuracy"
    ].std()
)


print("\n" + "=" * 70)
print("BEST-MODEL SUBJECT STATISTICS")
print("=" * 70)

print(
    f"Mean best-subject accuracy: "
    f"{subject_mean_accuracy * 100:.2f}%"
)

print(
    f"Std best-subject accuracy: "
    f"{subject_std_accuracy * 100:.2f}%"
)

print(
    f"Chance level: "
    f"{chance * 100:.2f}%"
)

print(
    f"Improvement over chance: "
    f"{(subject_mean_accuracy - chance) * 100:.2f} "
    f"percentage points"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "1. results/fnirs/"
    "within_subject_model_results.csv"
)

print(
    "2. results/fnirs/"
    "within_subject_subject_results.csv"
)

print(
    "3. results/fnirs/"
    "within_subject_best_per_subject.csv"
)

print(
    "4. results/fnirs/"
    "within_subject_predictions.csv"
)


# ============================================================
# EXPERIMENT TIME
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print(
    f"\nTotal experiment time: "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("fNIRS WITHIN-SUBJECT EXPERIMENT COMPLETED")
print("=" * 70)