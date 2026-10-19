"""
===============================================================================
Neuro-Symphony
Step 35: Sub-01 Band-Power Baseline
===============================================================================

Purpose:
    Train the first leakage-safe classification baseline
    using temporal EEG band-power features.

Input:
    data/features/temporal_spectral_sub01/
        sub-01_band_power.npy

    data/features/temporal_spectral_sub01/
        sub-01_spectral_metadata.csv

    data/splits/sub01/
        train_indices.npy
        val_indices.npy
        test_indices.npy

Feature shape:
    (720, 64, 5)

Flattened shape:
    (720, 320)

Bands:
    Delta
    Theta
    Alpha
    Beta
    Gamma

Model:
    Logistic Regression

Important leakage rules:
    - Split is already fixed in Step 34.
    - Scaler is fitted ONLY on training data.
    - Model is fitted ONLY on training data.
    - Training accuracy is evaluated on training data after fitting.
    - Validation is used for evaluation.
    - Test is evaluated only after training.
    - Test data is never used for fitting.
===============================================================================
"""


# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import sys

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


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# INPUT PATHS
# ============================================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_spectral_sub01"
    / "sub-01_band_power.npy"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_spectral_sub01"
    / "sub-01_spectral_metadata.csv"
)


# ============================================================================
# SPLIT PATHS
# ============================================================================

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "sub01"
)

TRAIN_INDICES_FILE = (
    SPLIT_DIR
    / "train_indices.npy"
)

VAL_INDICES_FILE = (
    SPLIT_DIR
    / "val_indices.npy"
)

TEST_INDICES_FILE = (
    SPLIT_DIR
    / "test_indices.npy"
)


# ============================================================================
# PARAMETERS
# ============================================================================

RANDOM_STATE = 42

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_BANDS = 5

EXPECTED_FEATURES = (
    EXPECTED_CHANNELS
    * EXPECTED_BANDS
)

EXPECTED_TRAIN = 504
EXPECTED_VALIDATION = 108
EXPECTED_TEST = 108


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("NEURO-SYMPHONY — STEP 35")
print("SUB-01 BAND-POWER LOGISTIC REGRESSION BASELINE")
print("=" * 80)


# ============================================================================
# CHECK INPUT FILES
# ============================================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

required_files = [
    FEATURE_FILE,
    METADATA_FILE,
    TRAIN_INDICES_FILE,
    VAL_INDICES_FILE,
    TEST_INDICES_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print(
        f"✓ Found: {file_path.name}"
    )


# ============================================================================
# LOAD BAND-POWER FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("LOADING BAND-POWER FEATURES")
print("=" * 80)

X = np.load(
    FEATURE_FILE
)

print(
    f"\nOriginal feature shape: {X.shape}"
)


# ============================================================================
# VALIDATE FEATURE SHAPE
# ============================================================================

expected_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    EXPECTED_BANDS,
)

if X.shape != expected_shape:

    raise RuntimeError(
        "\nUnexpected feature shape.\n"
        f"Expected: {expected_shape}\n"
        f"Found:    {X.shape}"
    )

print(
    "✓ Feature shape verified."
)


# ============================================================================
# CHECK NUMERICAL VALIDITY
# ============================================================================

if np.isnan(X).any():

    raise RuntimeError(
        "Band-power features contain NaN values."
    )

if np.isinf(X).any():

    raise RuntimeError(
        "Band-power features contain Inf values."
    )

print(
    "✓ No NaN values."
)

print(
    "✓ No Inf values."
)


# ============================================================================
# LOAD METADATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING LABELS")
print("=" * 80)

metadata = pd.read_csv(
    METADATA_FILE
)

if len(metadata) != EXPECTED_TRIALS:

    raise RuntimeError(
        f"Expected {EXPECTED_TRIALS} metadata rows, "
        f"found {len(metadata)}."
    )

if "label" not in metadata.columns:

    raise RuntimeError(
        "Metadata does not contain a 'label' column."
    )

if "label_name" not in metadata.columns:

    raise RuntimeError(
        "Metadata does not contain a 'label_name' column."
    )

y = metadata[
    "label"
].to_numpy()

label_names = (
    metadata[
        [
            "label",
            "label_name"
        ]
    ]
    .drop_duplicates()
    .sort_values("label")
)

print(
    f"\nNumber of trials: {len(y)}"
)

print(
    f"Number of classes: {len(np.unique(y))}"
)

print("\nClasses:")

for _, row in label_names.iterrows():

    print(
        f"  {int(row['label'])}: "
        f"{row['label_name']}"
    )


# ============================================================================
# FLATTEN FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("RESHAPING FEATURES")
print("=" * 80)

print(
    "\nBefore flattening:"
)

print(
    f"  {X.shape}"
)

X = X.reshape(
    EXPECTED_TRIALS,
    EXPECTED_FEATURES
)

print(
    "\nAfter flattening:"
)

print(
    f"  {X.shape}"
)

if X.shape != (
    EXPECTED_TRIALS,
    EXPECTED_FEATURES
):

    raise RuntimeError(
        "Unexpected flattened feature shape."
    )

print(
    "\n✓ 64 channels × 5 bands = 320 features/trial."
)


# ============================================================================
# LOAD FIXED SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("LOADING FIXED STEP-34 SPLIT")
print("=" * 80)

train_indices = np.load(
    TRAIN_INDICES_FILE
)

validation_indices = np.load(
    VAL_INDICES_FILE
)

test_indices = np.load(
    TEST_INDICES_FILE
)

print(
    f"\nTrain indices:      {len(train_indices)}"
)

print(
    f"Validation indices: {len(validation_indices)}"
)

print(
    f"Test indices:       {len(test_indices)}"
)


# ============================================================================
# VERIFY SPLIT SIZES
# ============================================================================

if len(train_indices) != EXPECTED_TRAIN:

    raise RuntimeError(
        "Unexpected number of training samples."
    )

if len(validation_indices) != EXPECTED_VALIDATION:

    raise RuntimeError(
        "Unexpected number of validation samples."
    )

if len(test_indices) != EXPECTED_TEST:

    raise RuntimeError(
        "Unexpected number of test samples."
    )


# ============================================================================
# CHECK SPLIT OVERLAP
# ============================================================================

train_set = set(
    train_indices
)

validation_set = set(
    validation_indices
)

test_set = set(
    test_indices
)

if train_set & validation_set:

    raise RuntimeError(
        "Train/validation overlap detected."
    )

if train_set & test_set:

    raise RuntimeError(
        "Train/test overlap detected."
    )

if validation_set & test_set:

    raise RuntimeError(
        "Validation/test overlap detected."
    )

print(
    "\n✓ No split overlap detected."
)


# ============================================================================
# CREATE TRAIN / VALIDATION / TEST DATA
# ============================================================================

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]

X_val = X[
    validation_indices
]

y_val = y[
    validation_indices
]

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]

print("\nDataset shapes:")

print(
    f"  X_train: {X_train.shape}"
)

print(
    f"  X_val:   {X_val.shape}"
)

print(
    f"  X_test:  {X_test.shape}"
)


# ============================================================================
# STANDARDIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STANDARDIZATION")
print("=" * 80)

print(
    "\nFitting StandardScaler ONLY on training data..."
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

print(
    "✓ Scaler fitted on training data only."
)

print(
    "Transforming validation data..."
)

X_val_scaled = scaler.transform(
    X_val
)

print(
    "Transforming test data..."
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "✓ Validation and test were transformed "
    "without refitting the scaler."
)


# ============================================================================
# VERIFY SCALER
# ============================================================================

print("\nScaler statistics:")

print(
    f"  Training mean: "
    f"{X_train_scaled.mean():.10f}"
)

print(
    f"  Training std:  "
    f"{X_train_scaled.std():.10f}"
)


# ============================================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 80)

model = LogisticRegression(
    max_iter=5000,
    random_state=RANDOM_STATE,
    solver="lbfgs"
)

print(
    "\nModel configuration:"
)

print(
    "  Algorithm: Logistic Regression"
)

print(
    "  Solver: lbfgs"
)

print(
    "  Max iterations: 5000"
)

print(
    f"  Random state: {RANDOM_STATE}"
)

print(
    "\nTraining..."
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "✓ Model training completed."
)


# ============================================================================
# TRAINING EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING EVALUATION")
print("=" * 80)

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


# ============================================================================
# VALIDATION EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION EVALUATION")
print("=" * 80)

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
    f"\nValidation Accuracy: "
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


# ============================================================================
# TEST EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

print(
    "\nThe model has already been trained."
)

print(
    "The scaler was fitted only on training data."
)

print(
    "The test set was not used for fitting."
)

print(
    "\nEvaluating the locked test set..."
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


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 80)
print("TEST CLASSIFICATION REPORT")
print("=" * 80)

target_names = (
    label_names[
        "label_name"
    ]
    .tolist()
)

print(
    "\n"
)

print(
    classification_report(
        y_test,
        test_predictions,
        labels=sorted(
            np.unique(y)
        ),
        target_names=target_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("TEST CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=sorted(
        np.unique(y)
    )
)

print(
    "\nRows = True class"
)

print(
    "Columns = Predicted class"
)

print(
    "\n"
)

print(cm)


# ============================================================================
# PER-CLASS TEST ACCURACY
# ============================================================================

print("\n" + "=" * 80)
print("PER-CLASS TEST ACCURACY")
print("=" * 80)

for label in sorted(
    np.unique(y)
):

    class_mask = (
        y_test == label
    )

    class_accuracy = accuracy_score(
        y_test[class_mask],
        test_predictions[class_mask]
    )

    class_name = label_names.loc[
        label_names["label"] == label,
        "label_name"
    ].iloc[0]

    print(
        f"{label}: "
        f"{class_name:<35} "
        f"{class_accuracy * 100:.2f}%"
    )


# ============================================================================
# BASELINE COMPARISON
# ============================================================================

chance_accuracy = (
    1.0
    /
    len(np.unique(y))
)

print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

print(
    f"\nNumber of classes: "
    f"{len(np.unique(y))}"
)

print(
    f"Chance accuracy: "
    f"{chance_accuracy * 100:.2f}%"
)

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 35 COMPLETE")
print("=" * 80)

print("\nInput:")

print(
    "  Temporal EEG band-power features"
)

print(
    "  64 channels × 5 frequency bands"
)

print(
    "  320 features per trial"
)

print("\nModel:")

print(
    "  Logistic Regression"
)

print("\nLeakage protection:")

print(
    "  ✓ Fixed Step-34 split"
)

print(
    "  ✓ Scaler fitted only on training data"
)

print(
    "  ✓ Model fitted only on training data"
)

print(
    "  ✓ Training evaluated after fitting"
)

print(
    "  ✓ Validation evaluated separately"
)

print(
    "  ✓ Test evaluated after training"
)

print(
    "  ✓ Test data never used for fitting"
)

print("\nResults:")

print(
    f"  Training Accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"  Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"  Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"  Test Balanced Accuracy: "
    f"{test_balanced_accuracy * 100:.2f}%"
)

print(
    f"  Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print("\nChance level:")

print(
    f"  {chance_accuracy * 100:.2f}%"
)

print(
    "\nNo test-set tuning was performed."
)

print("=" * 80)