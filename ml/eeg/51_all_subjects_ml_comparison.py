"""
STEP 51 — ALL-SUBJECT ML MODEL COMPARISON

Purpose:
    Compare five ML models using the EXACT same
    12-subject pipeline used in Step 35.

Dataset:
    12 subjects
    720 trials / subject
    64 EEG channels
    5 frequency bands

Features:
    64 × 5 = 320 band-power features

Split per subject:
    Train       = 504
    Validation  = 108
    Test        = 108

Pooled:
    Train       = 6048
    Validation  = 1296
    Test        = 1296

Models:
    1. Logistic Regression
    2. Linear SVM
    3. RBF SVM
    4. Random Forest
    5. Gradient Boosting

Important:
    - Same Step 35 features
    - Same Step 34 subject-wise splits
    - Same train-only StandardScaler
    - Test set is evaluated only after training
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
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
    "eeg" /
    "all_subjects_ml"
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
# HELPER
# ============================================================

def print_separator():
    print("\n" + "=" * 90)


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject_data(subject):

    feature_file = (
        FEATURE_ROOT /
        f"temporal_spectral_{subject}" /
        f"{subject}_band_power.npy"
    )

    split_dir = (
        SPLIT_ROOT /
        subject
    )

    train_file = split_dir / "train_indices.npy"
    val_file = split_dir / "val_indices.npy"
    test_file = split_dir / "test_indices.npy"

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
    # Load features
    # --------------------------------------------------------

    X = np.load(
        feature_file,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    train_indices = np.load(train_file)
    val_indices = np.load(val_file)
    test_indices = np.load(test_file)

    # --------------------------------------------------------
    # Validate feature shape
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

    # --------------------------------------------------------
    # Validate split sizes
    # --------------------------------------------------------

    if len(train_indices) != TRAIN_PER_SUBJECT:
        raise ValueError(
            f"{subject}: expected "
            f"{TRAIN_PER_SUBJECT} training trials."
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
    # Extract splits
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
    # Flatten 64 × 5 → 320
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
    print("STEP 51 — ALL-SUBJECT ML MODEL COMPARISON")
    print_separator()

    print(f"Subjects              : {len(SUBJECTS)}")
    print(
        f"Features per trial    : "
        f"{EXPECTED_CHANNELS * EXPECTED_BANDS}"
    )
    print("Features              : 1–40 Hz band power")
    print("Scaler                : StandardScaler")
    print("Random seed           : 42")
    print("Chance accuracy       : 12.50%")

    # ========================================================
    # LOAD ALL SUBJECTS
    # ========================================================

    train_X_parts = []
    val_X_parts = []
    test_X_parts = []

    train_y_parts = []
    val_y_parts = []
    test_y_parts = []

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
            f"Train      : {X_train.shape}"
        )

        print(
            f"Validation : {X_val.shape}"
        )

        print(
            f"Test       : {X_test.shape}"
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

        train_X_parts.append(X_train)
        val_X_parts.append(X_val)
        test_X_parts.append(X_test)

        train_y_parts.append(y_train)
        val_y_parts.append(y_val)
        test_y_parts.append(y_test)

    # ========================================================
    # COMBINE ALL SUBJECTS
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
        f"Training data       : {X_train.shape}"
    )

    print(
        f"Validation data     : {X_val.shape}"
    )

    print(
        f"Test data           : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Verify pooled sizes
    # --------------------------------------------------------

    if X_train.shape != (6048, 320):
        raise ValueError(
            f"Unexpected training shape: {X_train.shape}"
        )

    if X_val.shape != (1296, 320):
        raise ValueError(
            f"Unexpected validation shape: {X_val.shape}"
        )

    if X_test.shape != (1296, 320):
        raise ValueError(
            f"Unexpected test shape: {X_test.shape}"
        )

    print("\n✓ Pooled dataset sizes verified.")

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print_separator()
    print("TRAIN-ONLY STANDARDIZATION")
    print_separator()

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
        "✓ StandardScaler fitted ONLY on training data."
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED
        ),

        "Linear SVM": SVC(
            kernel="linear",
            random_state=RANDOM_SEED
        ),

        "RBF SVM": SVC(
            kernel="rbf",
            random_state=RANDOM_SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED
        ),
    }

    results = []

    # ========================================================
    # TRAIN + EVALUATE
    # ========================================================

    for model_name, model in models.items():

        print_separator()
        print(f"MODEL: {model_name}")
        print_separator()

        print("Training model...")

        model.fit(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_pred = model.predict(
            X_train_scaled
        )

        train_accuracy = accuracy_score(
            y_train,
            train_pred
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_pred = model.predict(
            X_val_scaled
        )

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

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_pred = model.predict(
            X_test_scaled
        )

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

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            test_pred,
            labels=np.arange(8)
        )

        cm_df = pd.DataFrame(
            cm,
            index=LABEL_NAMES,
            columns=LABEL_NAMES
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        cm_df.to_csv(
            RESULT_ROOT /
            f"step51_confusion_matrix_{safe_name}.csv"
        )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            test_pred,
            labels=np.arange(8),
            target_names=LABEL_NAMES,
            zero_division=0
        )

        with open(
            RESULT_ROOT /
            f"step51_classification_report_{safe_name}.txt",
            "w"
        ) as f:

            f.write(report)

        gc.collect()

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(results)

    # IMPORTANT:
    # This sorting is ONLY for reporting.
    # We are NOT selecting a model based on test results.

    results_df = results_df.sort_values(
        by="test_accuracy",
        ascending=False
    )

    results_file = (
        RESULT_ROOT /
        "step51_all_subjects_model_comparison.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    # ========================================================
    # FINAL RANKING
    # ========================================================

    print_separator()
    print("STEP 51 — FINAL TEST RESULTS")
    print_separator()

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

    print_separator()
    print("STEP 51 COMPLETE")
    print_separator()

    print(
        f"\nResults saved to:\n"
        f"{RESULT_ROOT}"
    )

    print(
        f"\nMain results file:\n"
        f"{results_file}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()