
"""
NEURO-SYMPHONY
20 — Official Baseline Evaluation

Current CSV:
    data/processed/ds004514_eeg_approved.csv

Official baseline:
    RobustScaler + Extra Trees

Evaluation:
    Strict Leave-One-Subject-Out

Outputs:
    - fold results
    - predictions
    - classification report
    - confusion matrix
    - class distribution
    - split summary
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = (
    "data/processed/"
    "ds004514_eeg_approved.csv"
)

OUTPUT_DIR = (
    "results/eeg/baseline"
)

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_predictions.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

CONFUSION_FILE = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

CLASS_DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIR,
    "class_distribution.csv"
)

SPLIT_FILE = os.path.join(
    OUTPUT_DIR,
    "split_summary.csv"
)

SEED = 42


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):

    required = [
        "subject_id",
        "trial_id",
        "label"
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    eeg_columns = [
        column
        for column in df.columns
        if column.startswith("A")
        or column.startswith("B")
    ]

    if len(eeg_columns) != 64:

        raise ValueError(
            f"Expected 64 EEG channels, "
            f"found {len(eeg_columns)}"
        )

    if df[eeg_columns].isna().any().any():

        raise ValueError(
            "Missing EEG values detected."
        )

    if not np.isfinite(
        df[eeg_columns].values
    ).all():

        raise ValueError(
            "Infinite EEG values detected."
        )

    duplicates = df.duplicated(
        subset=[
            "subject_id",
            "trial_id"
        ]
    ).sum()

    if duplicates != 0:

        raise ValueError(
            f"Duplicate trials found: "
            f"{duplicates}"
        )

    return eeg_columns


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NEURO-SYMPHONY — OFFICIAL BASELINE")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = pd.read_csv(
        CSV_PATH
    )

    eeg_columns = validate_dataset(
        df
    )

    subjects = sorted(
        df["subject_id"].unique()
    )

    labels = sorted(
        df["label"].unique()
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"EEG channels: {len(eeg_columns)}"
    )

    print(
        f"Subjects: {len(subjects)}"
    )

    print(
        f"Classes: {len(labels)}"
    )

    print(
        "✓ DATASET VALIDATION PASSED"
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_distribution = (
        df.groupby("label")
        .size()
        .reset_index(
            name="trial_count"
        )
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_FILE,
        index=False
    )

    print()
    print("CLASS DISTRIBUTION")
    print(class_distribution)

    # --------------------------------------------------------
    # SUBJECT DISTRIBUTION
    # --------------------------------------------------------

    subject_distribution = (
        df.groupby("subject_id")
        .size()
        .reset_index(
            name="trial_count"
        )
    )

    print()
    print("SUBJECT DISTRIBUTION")
    print(subject_distribution)

    # --------------------------------------------------------
    # SUBJECT / CLASS
    # --------------------------------------------------------

    subject_class = (
        df.groupby(
            [
                "subject_id",
                "label"
            ]
        )
        .size()
        .reset_index(
            name="trial_count"
        )
    )

    print()
    print(
        "Unique subject/class trial counts:"
    )

    print(
        subject_class[
            "trial_count"
        ].unique()
    )

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    label_map = {
        label: index
        for index, label
        in enumerate(labels)
    }

    df["_label_index"] = (
        df["label"]
        .map(label_map)
    )

    all_results = []

    all_predictions = []

    split_records = []

    # ========================================================
    # LOSO
    # ========================================================

    for fold, test_subject in enumerate(
        subjects,
        start=1
    ):

        print()
        print(
            f"FOLD {fold}/{len(subjects)}"
        )

        remaining_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        validation_subject = (
            remaining_subjects[-1]
        )

        train_subjects = [
            subject
            for subject in remaining_subjects
            if subject != validation_subject
        ]

        train_df = df[
            df["subject_id"].isin(
                train_subjects
            )
        ]

        validation_df = df[
            df["subject_id"]
            == validation_subject
        ]

        test_df = df[
            df["subject_id"]
            == test_subject
        ]

        # ----------------------------------------------------
        # SAVE EXACT SPLIT
        # ----------------------------------------------------

        split_records.append({

            "fold":
                fold,

            "train_subjects":
                ",".join(train_subjects),

            "validation_subject":
                validation_subject,

            "test_subject":
                test_subject,

            "train_samples":
                len(train_df),

            "validation_samples":
                len(validation_df),

            "test_samples":
                len(test_df)
        })

        X_train = train_df[
            eeg_columns
        ].values

        X_validation = validation_df[
            eeg_columns
        ].values

        X_test = test_df[
            eeg_columns
        ].values

        y_train = train_df[
            "_label_index"
        ].values

        y_test = test_df[
            "_label_index"
        ].values

        # ----------------------------------------------------
        # FIT PREPROCESSING ON TRAINING ONLY
        # ----------------------------------------------------

        scaler = RobustScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_validation = scaler.transform(
            X_validation
        )

        X_test = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = ExtraTreesClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        train_predictions = (
            model.predict(
                X_train
            )
        )

        test_predictions = (
            model.predict(
                X_test
            )
        )

        train_accuracy = (
            accuracy_score(
                y_train,
                train_predictions
            ) * 100
        )

        test_accuracy = (
            accuracy_score(
                y_test,
                test_predictions
            ) * 100
        )

        test_f1 = (
            f1_score(
                y_test,
                test_predictions,
                average="macro",
                zero_division=0
            ) * 100
        )

        balanced = (
            balanced_accuracy_score(
                y_test,
                test_predictions
            ) * 100
        )

        all_results.append({

            "fold":
                fold,

            "test_subject":
                test_subject,

            "validation_subject":
                validation_subject,

            "train_accuracy":
                train_accuracy,

            "test_accuracy":
                test_accuracy,

            "macro_f1":
                test_f1,

            "balanced_accuracy":
                balanced
        })

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        for i in range(
            len(test_predictions)
        ):

            all_predictions.append({

                "fold":
                    fold,

                "subject_id":
                    test_subject,

                "trial_id":
                    test_df.iloc[
                        i
                    ]["trial_id"],

                "true_label":
                    y_test[i],

                "predicted_label":
                    test_predictions[i]
            })

        print(
            f"Train accuracy: "
            f"{train_accuracy:.3f}%"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy:.3f}%"
        )

        print(
            f"Macro F1: "
            f"{test_f1:.3f}%"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    split_df = pd.DataFrame(
        split_records
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    split_df.to_csv(
        SPLIT_FILE,
        index=False
    )

    # ========================================================
    # GLOBAL CLASSIFICATION REPORT
    # ========================================================

    y_true = (
        predictions_df[
            "true_label"
        ].values
    )

    y_pred = (
        predictions_df[
            "predicted_label"
        ].values
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(labels))
        ),
        target_names=[
            str(label)
            for label in labels
        ],
        digits=4,
        zero_division=0
    )

    with open(
        REPORT_FILE,
        "w"
    ) as file:

        file.write(
            "NEURO-SYMPHONY OFFICIAL BASELINE\n"
        )

        file.write(
            "Model: RobustScaler + Extra Trees\n"
        )

        file.write(
            "Evaluation: 12-fold LOSO\n\n"
        )

        file.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(labels))
        )
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"true_{label}"
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ]
    )

    matrix_df.to_csv(
        CONFUSION_FILE
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("OFFICIAL BASELINE COMPLETE")
    print("=" * 80)

    print(
        f"Train accuracy: "
        f"{results_df['train_accuracy'].mean():.3f}%"
    )

    print(
        f"Test accuracy: "
        f"{results_df['test_accuracy'].mean():.3f}% "
        f"± "
        f"{results_df['test_accuracy'].std(ddof=0):.3f}%"
    )

    print(
        f"Macro F1: "
        f"{results_df['macro_f1'].mean():.3f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{results_df['balanced_accuracy'].mean():.3f}%"
    )

    print()
    print(
        f"Results: {RESULTS_FILE}"
    )

    print(
        f"Predictions: {PREDICTIONS_FILE}"
    )

    print(
        f"Classification report: "
        f"{REPORT_FILE}"
    )

    print(
        f"Confusion matrix: "
        f"{CONFUSION_FILE}"
    )

    print(
        f"Class distribution: "
        f"{CLASS_DISTRIBUTION_FILE}"
    )

    print(
        f"Split summary: "
        f"{SPLIT_FILE}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()