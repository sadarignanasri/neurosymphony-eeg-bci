"""
STEP 35 — ALL-SUBJECT BASELINE CLASSIFIER

Model:
    Logistic Regression

Features:
    1–40 Hz EEG band power

Input per subject:
    (720 trials, 64 channels, 5 bands)

Flattened:
    64 × 5 = 320 features per trial

Splits:
    Train       = 504 / subject
    Validation  = 108 / subject
    Test        = 108 / subject

Important:
    - Splits were created in Step 34.
    - StandardScaler is fitted ONLY on training data.
    - Validation and test data are transformed using
      the training-fitted scaler.
    - Test data is evaluated only once at the end.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT /
    "data" /
    "features"
)

SPLIT_ROOT = (
    PROJECT_ROOT /
    "data" /
    "splits"
)

RESULT_ROOT = (
    PROJECT_ROOT /
    "results" /
    "baseline"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 13)
]

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_BANDS = 5

TRAIN_PER_SUBJECT = 504
VAL_PER_SUBJECT = 108
TEST_PER_SUBJECT = 108

RANDOM_SEED = 42


# ============================================================
# LABEL NAMES
# ============================================================

LABEL_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print("\n" + "=" * 90)


def load_subject_data(subject):
    """
    Load one subject's band-power features and split indices.
    """

    feature_file = (
        FEATURE_ROOT /
        f"temporal_spectral_{subject}" /
        f"{subject}_band_power.npy"
    )

    split_dir = (
        SPLIT_ROOT /
        subject
    )

    train_file = (
        split_dir /
        "train_indices.npy"
    )

    val_file = (
        split_dir /
        "val_indices.npy"
    )

    test_file = (
        split_dir /
        "test_indices.npy"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        feature_file,
        train_file,
        val_file,
        test_file,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{file_path}"
            )

    # --------------------------------------------------------
    # Load band-power data
    # --------------------------------------------------------

    X = np.load(
        feature_file,
        mmap_mode="r"
    )

    train_indices = np.load(
        train_file
    )

    val_indices = np.load(
        val_file
    )

    test_indices = np.load(
        test_file
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_BANDS,
    )

    if X.shape != expected_shape:

        raise ValueError(
            f"{subject}: unexpected feature shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found: {X.shape}"
        )

    if len(train_indices) != TRAIN_PER_SUBJECT:
        raise ValueError(
            f"{subject}: expected "
            f"{TRAIN_PER_SUBJECT} train trials."
        )

    if len(val_indices) != VAL_PER_SUBJECT:
        raise ValueError(
            f"{subject}: expected "
            f"{VAL_PER_SUBJECT} validation trials."
        )

    if len(test_indices) != TEST_PER_SUBJECT:
        raise ValueError(
            f"{subject}: expected "
            f"{TEST_PER_SUBJECT} test trials."
        )

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    y = np.repeat(
        np.arange(8),
        90
    )

    # --------------------------------------------------------
    # Extract split data
    # --------------------------------------------------------

    X_train = np.asarray(
        X[train_indices],
        dtype=np.float32
    )

    X_val = np.asarray(
        X[val_indices],
        dtype=np.float32
    )

    X_test = np.asarray(
        X[test_indices],
        dtype=np.float32
    )

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    # --------------------------------------------------------
    # Flatten:
    #
    # 64 channels × 5 bands
    # = 320 features
    # --------------------------------------------------------

    X_train = X_train.reshape(
        len(X_train),
        EXPECTED_CHANNELS * EXPECTED_BANDS
    )

    X_val = X_val.reshape(
        len(X_val),
        EXPECTED_CHANNELS * EXPECTED_BANDS
    )

    X_test = X_test.reshape(
        len(X_test),
        EXPECTED_CHANNELS * EXPECTED_BANDS
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("STEP 35 — ALL-SUBJECT BAND-POWER BASELINE")
    print_separator()

    print(
        f"Subjects              : {len(SUBJECTS)}"
    )

    print(
        f"Features per trial    : "
        f"{EXPECTED_CHANNELS * EXPECTED_BANDS}"
    )

    print(
        "Model                 : Logistic Regression"
    )

    print(
        "Scaler                : StandardScaler"
    )

    print(
        "Random seed           : 42"
    )

    print(
        "Chance accuracy       : 12.50%"
    )

    # ========================================================
    # LOAD ALL SUBJECTS
    # ========================================================

    train_X_parts = []
    val_X_parts = []
    test_X_parts = []

    train_y_parts = []
    val_y_parts = []
    test_y_parts = []

    # --------------------------------------------------------
    # Subject-level bookkeeping
    # --------------------------------------------------------

    subject_summary = []

    for subject in SUBJECTS:

        print_separator()
        print(f"Loading {subject}")
        print_separator()

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = load_subject_data(subject)

        print(
            f"Train features      : {X_train.shape}"
        )

        print(
            f"Validation features : {X_val.shape}"
        )

        print(
            f"Test features       : {X_test.shape}"
        )

        print(
            f"Train labels        : {y_train.shape}"
        )

        print(
            f"Validation labels   : {y_val.shape}"
        )

        print(
            f"Test labels         : {y_test.shape}"
        )

        # ----------------------------------------------------
        # Check finite values
        # ----------------------------------------------------

        if not np.isfinite(X_train).all():
            raise ValueError(
                f"{subject}: NaN/Inf in training features."
            )

        if not np.isfinite(X_val).all():
            raise ValueError(
                f"{subject}: NaN/Inf in validation features."
            )

        if not np.isfinite(X_test).all():
            raise ValueError(
                f"{subject}: NaN/Inf in test features."
            )

        print("✓ Feature values are finite.")

        train_X_parts.append(X_train)
        val_X_parts.append(X_val)
        test_X_parts.append(X_test)

        train_y_parts.append(y_train)
        val_y_parts.append(y_val)
        test_y_parts.append(y_test)

        subject_summary.append(
            {
                "subject": subject,
                "train_trials": len(y_train),
                "validation_trials": len(y_val),
                "test_trials": len(y_test),
            }
        )

    # ========================================================
    # COMBINE SUBJECTS
    # ========================================================

    print_separator()
    print("COMBINING ALL SUBJECTS")
    print_separator()

    X_train = np.concatenate(
        train_X_parts,
        axis=0
    )

    X_val = np.concatenate(
        val_X_parts,
        axis=0
    )

    X_test = np.concatenate(
        test_X_parts,
        axis=0
    )

    y_train = np.concatenate(
        train_y_parts,
        axis=0
    )

    y_val = np.concatenate(
        val_y_parts,
        axis=0
    )

    y_test = np.concatenate(
        test_y_parts,
        axis=0
    )

    print(
        f"Training data       : "
        f"{X_train.shape}"
    )

    print(
        f"Validation data     : "
        f"{X_val.shape}"
    )

    print(
        f"Test data           : "
        f"{X_test.shape}"
    )

    print(
        f"Training labels     : "
        f"{y_train.shape}"
    )

    print(
        f"Validation labels   : "
        f"{y_val.shape}"
    )

    print(
        f"Test labels         : "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # Verify expected pooled sizes
    # --------------------------------------------------------

    if X_train.shape != (
        6048,
        320
    ):
        raise ValueError(
            f"Unexpected training shape: "
            f"{X_train.shape}"
        )

    if X_val.shape != (
        1296,
        320
    ):
        raise ValueError(
            f"Unexpected validation shape: "
            f"{X_val.shape}"
        )

    if X_test.shape != (
        1296,
        320
    ):
        raise ValueError(
            f"Unexpected test shape: "
            f"{X_test.shape}"
        )

    print(
        "\n✓ Pooled dataset sizes verified."
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print_separator()
    print("TRAIN-ONLY STANDARDIZATION")
    print_separator()

    scaler = StandardScaler()

    print(
        "Fitting StandardScaler ONLY on "
        "training data..."
    )

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    print(
        "✓ Training scaler fitted."
    )

    print(
        "Transforming validation data "
        "using training scaler..."
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    print(
        "✓ Validation transformed."
    )

    print(
        "Transforming test data "
        "using training scaler..."
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "✓ Test transformed."
    )

    # --------------------------------------------------------
    # Verify no invalid values
    # --------------------------------------------------------

    if not np.isfinite(
        X_train_scaled
    ).all():

        raise ValueError(
            "NaN/Inf in scaled training data."
        )

    if not np.isfinite(
        X_val_scaled
    ).all():

        raise ValueError(
            "NaN/Inf in scaled validation data."
        )

    if not np.isfinite(
        X_test_scaled
    ).all():

        raise ValueError(
            "NaN/Inf in scaled test data."
        )

    print(
        "✓ Scaled datasets contain no NaN/Inf."
    )

    # ========================================================
    # MODEL
    # ========================================================

    print_separator()
    print("TRAINING LOGISTIC REGRESSION")
    print_separator()

    model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_SEED,
    )

    print(
        "Fitting model on training data..."
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "✓ Model training complete."
    )

    # ========================================================
    # TRAINING EVALUATION
    # ========================================================

    train_predictions = model.predict(
        X_train_scaled
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    print(
        f"\nTraining Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    # ========================================================
    # VALIDATION EVALUATION
    # ========================================================

    print_separator()
    print("VALIDATION EVALUATION")
    print_separator()

    val_predictions = model.predict(
        X_val_scaled
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            val_predictions
        )
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Balanced Accuracy: "
        f"{val_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro F1: "
        f"{val_macro_f1:.4f}"
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    print_separator()
    print("FINAL HELD-OUT TEST EVALUATION")
    print_separator()

    print(
        "Evaluating the model on the "
        "held-out test set..."
    )

    test_predictions = model.predict(
        X_test_scaled
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_predictions
        )
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro"
    )

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print_separator()
    print("TEST CLASSIFICATION REPORT")
    print_separator()

    report = classification_report(
        y_test,
        test_predictions,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print_separator()
    print("TEST CONFUSION MATRIX")
    print_separator()

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print(cm)

    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    print_separator()
    print("PER-CLASS TEST ACCURACY")
    print_separator()

    per_class_accuracy = (
        cm.diagonal() /
        cm.sum(axis=1)
    )

    for label_id, accuracy in enumerate(
        per_class_accuracy
    ):

        print(
            f"{label_id}: "
            f"{LABEL_NAMES[label_id]:<35} "
            f"{accuracy * 100:.2f}%"
        )

    # ========================================================
    # CHANCE LEVEL
    # ========================================================

    chance_accuracy = (
        1.0 /
        len(LABEL_NAMES)
    )

    print_separator()
    print("BASELINE SUMMARY")
    print_separator()

    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Balanced Accuracy: "
        f"{val_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro F1: "
        f"{val_macro_f1:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"Chance Accuracy: "
        f"{chance_accuracy * 100:.2f}%"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print_separator()
    print("SAVING RESULTS")
    print_separator()

    summary = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                "features": "64-channel band power, 1-40 Hz",
                "subjects": 12,
                "train_trials": len(y_train),
                "validation_trials": len(y_val),
                "test_trials": len(y_test),
                "training_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "validation_balanced_accuracy":
                    val_balanced_accuracy,
                "validation_macro_f1":
                    val_macro_f1,
                "test_accuracy": test_accuracy,
                "test_balanced_accuracy":
                    test_balanced_accuracy,
                "test_macro_f1":
                    test_macro_f1,
                "chance_accuracy":
                    chance_accuracy,
                "random_seed":
                    RANDOM_SEED,
            }
        ]
    )

    summary_file = (
        RESULT_ROOT /
        "step35_baseline_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=LABEL_NAMES,
        columns=LABEL_NAMES
    )

    cm_file = (
        RESULT_ROOT /
        "step35_test_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_file
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        test_predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_file = (
        RESULT_ROOT /
        "step35_test_classification_report.csv"
    )

    report_df.to_csv(
        report_file
    )

    # --------------------------------------------------------
    # Save subject summary
    # --------------------------------------------------------

    subject_summary_df = pd.DataFrame(
        subject_summary
    )

    subject_summary_file = (
        RESULT_ROOT /
        "step35_subject_summary.csv"
    )

    subject_summary_df.to_csv(
        subject_summary_file,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        f"\n✓ Saved summary:"
        f"\n  {summary_file}"
    )

    print(
        f"\n✓ Saved confusion matrix:"
        f"\n  {cm_file}"
    )

    print(
        f"\n✓ Saved classification report:"
        f"\n  {report_file}"
    )

    print(
        f"\n✓ Saved subject summary:"
        f"\n  {subject_summary_file}"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del X_train
    del X_val
    del X_test

    del X_train_scaled
    del X_val_scaled
    del X_test_scaled

    gc.collect()

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print_separator()
    print("STEP 35 — COMPLETE")
    print_separator()

    print(
        "✓ All 12 subjects used."
    )

    print(
        "✓ Train-only scaler used."
    )

    print(
        "✓ Validation evaluated separately."
    )

    print(
        "✓ Held-out test evaluated."
    )

    print(
        "✓ Classification report generated."
    )

    print(
        "✓ Confusion matrix generated."
    )

    print(
        "✓ Baseline results saved."
    )

    print_separator()


if __name__ == "__main__":
    main()