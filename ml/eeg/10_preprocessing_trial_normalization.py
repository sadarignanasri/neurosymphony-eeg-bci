"""
10_preprocessing_trial_normalization.py

Neuro-Symphony
EEG Preprocessing Experiment 3: Per-Trial Normalization

Purpose
-------
Evaluate whether normalizing the 64 EEG channel values within
each individual trial improves subject-independent EEG
classification performance.

This experiment is different from StandardScaler and RobustScaler.

StandardScaler:
    Uses training-set mean and standard deviation for each channel.

RobustScaler:
    Uses training-set median and IQR for each channel.

Per-Trial Normalization:
    Each individual trial is normalized independently across
    its 64 EEG channels.

Formula
-------
For each trial:

    z = (x - trial_mean) / trial_std

where:

    trial_mean = mean of the 64 EEG channel values
    trial_std  = standard deviation of the 64 EEG channel values

Dataset
-------
data/processed/ds004514_eeg_approved.csv

Validation
----------
Strict Leave-One-Subject-Out (LOSO) cross-validation.

Important
---------
The normalization is performed independently for each trial.
Therefore, no information from other trials or test subjects
is used during the normalization process.

Input features
--------------
A1-A32 and B1-B32 = 64 EEG channels

Target
------
label

Models
------
1. Logistic Regression
2. Linear SVM
3. LDA
4. KNN
5. Random Forest
6. Extra Trees

Metrics
-------
- Accuracy
- Macro F1
- Balanced Accuracy

Reference
---------
Raw EEG baseline:
12.91%

8-class chance:
12.50%
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/ds004514_eeg_approved.csv"

RESULTS_DIR = "results/eeg/preprocessing"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "trial_normalization_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "trial_normalization_predictions.csv"
)

# Existing raw EEG baseline
BASELINE_ACCURACY = 0.1291

# StandardScaler best accuracy
STANDARD_SCALER_BEST_ACCURACY = 0.1291

# RobustScaler best accuracy
ROBUST_SCALER_BEST_ACCURACY = 0.1304

# Eight-class chance level
CHANCE_ACCURACY = 1.0 / 8.0

RANDOM_STATE = 42


# ============================================================
# EEG CHANNELS
# ============================================================

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE
    ),

    "Linear SVM": SVC(
        kernel="linear",
        random_state=RANDOM_STATE
    ),

    "LDA": LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
}


# ============================================================
# UTILITY
# ============================================================

def print_header(title):
    """Print a formatted section header."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df):
    """Validate the EEG CSV before training."""

    print_header("DATASET VALIDATION")

    required_columns = (
        ["subject_id", "trial_id", "label"]
        + EEG_CHANNELS
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    subjects = sorted(
        df["subject_id"].astype(str).unique()
    )

    print(f"Subjects: {len(subjects)}")
    print(f"Subject IDs: {subjects}")

    if len(subjects) != 12:
        raise ValueError(
            f"Expected 12 subjects, found {len(subjects)}"
        )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = sorted(
        df["label"].unique()
    )

    print(f"Classes: {len(classes)}")
    print(f"Labels: {classes}")

    if len(classes) != 8:
        raise ValueError(
            f"Expected 8 classes, found {len(classes)}"
        )

    # --------------------------------------------------------
    # Samples per subject
    # --------------------------------------------------------

    subject_counts = (
        df.groupby("subject_id")
        .size()
    )

    print("\nSamples per subject:")
    print(subject_counts.to_string())

    if not (subject_counts == 720).all():
        raise ValueError(
            "Every subject must contain exactly 720 trials."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nClass distribution:")
    print(class_counts.to_string())

    if not (class_counts == 1080).all():
        raise ValueError(
            "Every class must contain exactly 1080 samples."
        )

    # --------------------------------------------------------
    # Subject/class distribution
    # --------------------------------------------------------

    subject_class_counts = (
        df.groupby(
            ["subject_id", "label"]
        )
        .size()
    )

    if not (subject_class_counts == 90).all():
        raise ValueError(
            "Every subject/class combination must contain 90 trials."
        )

    print(
        "\nSubject/class validation: "
        "every combination contains 90 trials."
    )

    # --------------------------------------------------------
    # Duplicate subject/trial combinations
    # --------------------------------------------------------

    duplicate_keys = df.duplicated(
        subset=["subject_id", "trial_id"]
    ).sum()

    print(
        f"Duplicate subject/trial combinations: "
        f"{duplicate_keys}"
    )

    if duplicate_keys != 0:
        raise ValueError(
            "Duplicate subject/trial combinations detected."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        df[EEG_CHANNELS]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing EEG values: {missing_values}"
    )

    if missing_values != 0:
        raise ValueError(
            "Missing EEG values detected."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite_values = np.isinf(
        df[EEG_CHANNELS].to_numpy()
    ).sum()

    print(
        f"Infinite EEG values: {infinite_values}"
    )

    if infinite_values != 0:
        raise ValueError(
            "Infinite EEG values detected."
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    non_numeric = [
        col
        for col in EEG_CHANNELS
        if not pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric EEG columns: {non_numeric}"
        )

    print(
        f"EEG features: {len(EEG_CHANNELS)}"
    )

    print("\nDataset validation PASSED.")


# ============================================================
# PER-TRIAL NORMALIZATION
# ============================================================

def normalize_trials(X):
    """
    Normalize every EEG trial independently across its
    64 channel values.

    Parameters
    ----------
    X : pandas.DataFrame
        Rows = EEG trials
        Columns = 64 EEG channels

    Returns
    -------
    pandas.DataFrame
        Trial-normalized EEG data.
    """

    X_array = X.to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Calculate mean for each trial.
    # axis=1 means across the 64 EEG channels.
    # --------------------------------------------------------

    trial_means = np.mean(
        X_array,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Calculate standard deviation for each trial.
    # --------------------------------------------------------

    trial_stds = np.std(
        X_array,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Safety check for zero-variance trials.
    # --------------------------------------------------------

    zero_std_trials = (
        trial_stds.flatten() == 0
    )

    if np.any(zero_std_trials):

        count = int(
            np.sum(zero_std_trials)
        )

        raise ValueError(
            f"{count} trials have zero "
            "standard deviation across EEG channels."
        )

    # --------------------------------------------------------
    # Z-score each trial.
    # --------------------------------------------------------

    normalized = (
        X_array - trial_means
    ) / trial_stds

    # --------------------------------------------------------
    # Return DataFrame with original columns/index.
    # --------------------------------------------------------

    return pd.DataFrame(
        normalized,
        columns=X.columns,
        index=X.index
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    start_time = time.time()

    print_header(
        "NEURO-SYMPHONY EEG PER-TRIAL NORMALIZATION EXPERIMENT"
    )

    print("Purpose:")
    print(
        "Test whether normalizing each EEG trial "
        "across its 64 channels improves "
        "subject-independent classification."
    )

    print(f"\nDataset: {DATA_PATH}")

    print(
        f"Existing raw EEG baseline: "
        f"{BASELINE_ACCURACY * 100:.2f}%"
    )

    print(
        f"StandardScaler best accuracy: "
        f"{STANDARD_SCALER_BEST_ACCURACY * 100:.2f}%"
    )

    print(
        f"RobustScaler best accuracy: "
        f"{ROBUST_SCALER_BEST_ACCURACY * 100:.2f}%"
    )

    print(
        f"8-class chance level: "
        f"{CHANCE_ACCURACY * 100:.2f}%"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print_header("LOADING DATASET")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded dataset with shape: {df.shape}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_dataset(df)

    # ========================================================
    # PREPARE RAW DATA
    # ========================================================

    X_raw = df[
        EEG_CHANNELS
    ].copy()

    y = df[
        "label"
    ].copy()

    groups = (
        df["subject_id"]
        .astype(str)
        .copy()
    )

    subjects = sorted(
        groups.unique()
    )

    # ========================================================
    # APPLY PER-TRIAL NORMALIZATION
    # ========================================================

    print_header(
        "APPLYING PER-TRIAL NORMALIZATION"
    )

    print(
        "Normalization rule:"
    )

    print(
        "For every trial:"
    )

    print(
        "    z = (x - trial_mean) / trial_std"
    )

    print(
        "\nNormalization is performed independently "
        "for every trial."
    )

    print(
        "No information from another trial or "
        "another subject is used."
    )

    X = normalize_trials(
        X_raw
    )

    # ========================================================
    # VALIDATE NORMALIZED DATA
    # ========================================================

    normalized_array = X.to_numpy()

    normalized_means = np.mean(
        normalized_array,
        axis=1
    )

    normalized_stds = np.std(
        normalized_array,
        axis=1
    )

    max_abs_mean = np.max(
        np.abs(normalized_means)
    )

    max_std_error = np.max(
        np.abs(
            normalized_stds - 1.0
        )
    )

    print(
        f"\nMaximum absolute trial mean "
        f"after normalization: "
        f"{max_abs_mean:.10f}"
    )

    print(
        f"Maximum trial std error from 1: "
        f"{max_std_error:.10f}"
    )

    if max_abs_mean > 1e-10:
        raise ValueError(
            "Trial normalization mean validation failed."
        )

    if max_std_error > 1e-10:
        raise ValueError(
            "Trial normalization standard deviation validation failed."
        )

    print(
        "Trial normalization validation PASSED."
    )

    # ========================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================

    print_header(
        "EXPERIMENT CONFIGURATION"
    )

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Subjects: {len(subjects)}")
    print(f"Classes: {y.nunique()}")

    print(
        "Validation: Leave-One-Subject-Out"
    )

    print(
        "Preprocessing: Per-Trial Z-Score Normalization"
    )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    all_predictions = []

    summary_results = []

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name, model in MODELS.items():

        print_header(
            f"MODEL: {model_name}"
        )

        subject_accuracies = []

        subject_f1s = []

        subject_balanced_accuracies = []

        model_predictions = []

        model_start = time.time()

        # ----------------------------------------------------
        # LOSO
        # ----------------------------------------------------

        for fold_number, test_subject in enumerate(
            subjects,
            start=1
        ):

            train_mask = (
                groups != test_subject
            )

            test_mask = (
                groups == test_subject
            )

            X_train = X.loc[
                train_mask
            ]

            X_test = X.loc[
                test_mask
            ]

            y_train = y.loc[
                train_mask
            ]

            y_test = y.loc[
                test_mask
            ]

            # ------------------------------------------------
            # IMPORTANT
            #
            # Per-trial normalization has already been done
            # independently within each trial.
            #
            # StandardScaler below is additionally fitted only
            # on the training subjects. This controls feature
            # scale for models that benefit from it.
            # ------------------------------------------------

            pipeline = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        model
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            y_pred = pipeline.predict(
                X_test
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

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

            subject_accuracies.append(
                accuracy
            )

            subject_f1s.append(
                macro_f1
            )

            subject_balanced_accuracies.append(
                balanced_accuracy
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            fold_predictions = pd.DataFrame({
                "subject_id": df.loc[
                    test_mask,
                    "subject_id"
                ].values,

                "trial_id": df.loc[
                    test_mask,
                    "trial_id"
                ].values,

                "true_label": y_test.values,

                "predicted_label": y_pred,

                "model": model_name,

                "fold": fold_number,

                "preprocessing": (
                    "Per-Trial Normalization"
                ),
            })

            model_predictions.append(
                fold_predictions
            )

            print(
                f"Fold {fold_number:02d}/12 | "
                f"Test Subject: {test_subject} | "
                f"Accuracy: {accuracy * 100:.2f}% | "
                f"Macro F1: {macro_f1 * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        mean_accuracy = np.mean(
            subject_accuracies
        )

        std_accuracy = np.std(
            subject_accuracies
        )

        mean_f1 = np.mean(
            subject_f1s
        )

        mean_balanced_accuracy = np.mean(
            subject_balanced_accuracies
        )

        improvement_vs_chance = (
            mean_accuracy
            - CHANCE_ACCURACY
        )

        improvement_vs_raw_baseline = (
            mean_accuracy
            - BASELINE_ACCURACY
        )

        improvement_vs_standard_scaler = (
            mean_accuracy
            - STANDARD_SCALER_BEST_ACCURACY
        )

        improvement_vs_robust_scaler = (
            mean_accuracy
            - ROBUST_SCALER_BEST_ACCURACY
        )

        result = {
            "preprocessing": (
                "Per-Trial Normalization"
            ),

            "model": model_name,

            "accuracy_mean": mean_accuracy,

            "accuracy_std": std_accuracy,

            "macro_f1": mean_f1,

            "balanced_accuracy": (
                mean_balanced_accuracy
            ),

            "improvement_vs_chance": (
                improvement_vs_chance
            ),

            "improvement_vs_raw_baseline": (
                improvement_vs_raw_baseline
            ),

            "improvement_vs_standard_scaler": (
                improvement_vs_standard_scaler
            ),

            "improvement_vs_robust_scaler": (
                improvement_vs_robust_scaler
            ),

            "chance_accuracy": (
                CHANCE_ACCURACY
            ),

            "raw_baseline_accuracy": (
                BASELINE_ACCURACY
            ),

            "standard_scaler_best_accuracy": (
                STANDARD_SCALER_BEST_ACCURACY
            ),

            "robust_scaler_best_accuracy": (
                ROBUST_SCALER_BEST_ACCURACY
            ),
        }

        summary_results.append(
            result
        )

        all_predictions.extend(
            model_predictions
        )

        elapsed_model = (
            time.time() - model_start
        )

        print(
            f"\n{model_name} completed "
            f"in {elapsed_model:.2f} seconds."
        )

        print(
            f"Mean Accuracy: "
            f"{mean_accuracy * 100:.2f}% "
            f"+/- "
            f"{std_accuracy * 100:.2f}%"
        )

        print(
            f"Macro F1: "
            f"{mean_f1 * 100:.2f}%"
        )

        print(
            f"Balanced Accuracy: "
            f"{mean_balanced_accuracy * 100:.2f}%"
        )

        print(
            f"Change vs raw baseline: "
            f"{improvement_vs_raw_baseline * 100:+.2f} pp"
        )

        print(
            f"Change vs RobustScaler best: "
            f"{improvement_vs_robust_scaler * 100:+.2f} pp"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        summary_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print_header(
        "PER-TRIAL NORMALIZATION FINAL RESULTS"
    )

    display_df = results_df.copy()

    display_df["accuracy_mean"] *= 100

    display_df["accuracy_std"] *= 100

    display_df["macro_f1"] *= 100

    display_df["balanced_accuracy"] *= 100

    display_df[
        "improvement_vs_chance"
    ] *= 100

    display_df[
        "improvement_vs_raw_baseline"
    ] *= 100

    display_df[
        "improvement_vs_standard_scaler"
    ] *= 100

    display_df[
        "improvement_vs_robust_scaler"
    ] *= 100

    print(
        display_df[
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1",
                "balanced_accuracy",
                "improvement_vs_chance",
                "improvement_vs_raw_baseline",
                "improvement_vs_standard_scaler",
                "improvement_vs_robust_scaler",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_index = results_df[
        "accuracy_mean"
    ].idxmax()

    best_model = results_df.loc[
        best_index
    ]

    print_header(
        "BEST PER-TRIAL NORMALIZATION MODEL"
    )

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"Accuracy: "
        f"{best_model['accuracy_mean'] * 100:.2f}%"
    )

    print(
        f"Accuracy Std: "
        f"{best_model['accuracy_std'] * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{best_model['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_model['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"Change vs raw baseline: "
        f"{best_model['improvement_vs_raw_baseline'] * 100:+.2f} pp"
    )

    print(
        f"Change vs RobustScaler: "
        f"{best_model['improvement_vs_robust_scaler'] * 100:+.2f} pp"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print_header("OUTPUT FILES")

    print(
        f"Results:\n{RESULTS_FILE}"
    )

    print(
        f"\nPredictions:\n{PREDICTIONS_FILE}"
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    total_time = (
        time.time() - start_time
    )

    print_header(
        "EEG PER-TRIAL NORMALIZATION EXPERIMENT COMPLETED"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()