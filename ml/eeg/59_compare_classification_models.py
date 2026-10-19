
"""
===============================================================================
NeuroSymphony — Step 59: Classification Model Comparison
===============================================================================

Purpose:
    Compare five classifiers using the original Step 35 band-power features.

Dataset:
    Subject 01
    720 trials
    64 EEG channels x 5 frequency bands = 320 features
    8 classes

Models:
    1. Logistic Regression (Step 35 reference)
    2. Linear SVM
    3. RBF SVM
    4. Random Forest
    5. k-Nearest Neighbors

Experimental rules:
    - Reuse the exact Step 35 train/validation/test indices.
    - Use the original Step 35 features and labels.
    - Fit StandardScaler on training data only.
    - Rank models using validation macro F1.
    - Use validation accuracy and balanced accuracy as
      additional selection metrics.
    - Evaluate the selected model on the test set only once.
    - Never use test performance to select the model.
    - Do not overwrite Steps 35, 55, or 58.
===============================================================================
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "features"
    / "temporal_spectral_sub01" / "sub-01_band_power.npy"
)

METADATA_FILE = (
    PROJECT_ROOT / "data" / "features"
    / "temporal_spectral_sub01" / "sub-01_spectral_metadata.csv"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "sub01"

TRAIN_FILE = SPLIT_DIR / "train_indices.npy"
VAL_FILE = SPLIT_DIR / "val_indices.npy"
TEST_FILE = SPLIT_DIR / "test_indices.npy"

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "eeg"
    / "model_comparison" / "sub01" / "step59"
)


# =============================================================================
# PARAMETERS
# =============================================================================

RANDOM_STATE = 42

EXPECTED_TRIALS = 720
EXPECTED_FEATURES = 320
EXPECTED_TRAIN = 504
EXPECTED_VAL = 108
EXPECTED_TEST = 108
EXPECTED_CLASSES = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER: EVALUATION METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# =============================================================================
# START
# =============================================================================

print("=" * 78)
print("NEUROSYMPHONY — STEP 59: CLASSIFICATION MODEL COMPARISON")
print("=" * 78)

required_files = [
    FEATURE_FILE,
    METADATA_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Found: {path}")


# =============================================================================
# LOAD AND VALIDATE FEATURES
# =============================================================================

print("\nLoading original Step 35 band-power features...")

X_raw = np.load(FEATURE_FILE)

if X_raw.shape != (EXPECTED_TRIALS, 64, 5):
    raise ValueError(
        f"Unexpected feature shape: {X_raw.shape}; "
        f"expected {(EXPECTED_TRIALS, 64, 5)}"
    )

if not np.isfinite(X_raw).all():
    raise ValueError("Features contain NaN or infinite values.")

X = X_raw.reshape(EXPECTED_TRIALS, EXPECTED_FEATURES)

print(f"Feature shape: {X.shape}")


# =============================================================================
# LOAD LABELS
# =============================================================================

metadata = pd.read_csv(METADATA_FILE)

if len(metadata) != EXPECTED_TRIALS:
    raise ValueError(
        f"Expected {EXPECTED_TRIALS} metadata rows, got {len(metadata)}"
    )

for column in ["label", "label_name"]:
    if column not in metadata.columns:
        raise ValueError(f"Metadata is missing required column: {column}")

y = metadata["label"].to_numpy()

if pd.isna(y).any():
    raise ValueError("Labels contain missing values.")

label_table = (
    metadata[["label", "label_name"]]
    .drop_duplicates()
    .sort_values("label")
)

if label_table["label"].duplicated().any():
    raise ValueError("A numeric label maps to multiple label names.")

labels = sorted(np.unique(y).tolist())

if len(labels) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, found {len(labels)}"
    )

label_names = label_table.set_index("label")["label_name"].to_dict()

print(f"Number of classes: {len(labels)}")
print("\nClass mapping:")
for label in labels:
    print(f"  {label}: {label_names[label]}")


# =============================================================================
# LOAD AND VALIDATE STEP 35 SPLITS
# =============================================================================

print("\nLoading exact Step 35 split indices...")

train_idx = np.load(TRAIN_FILE).astype(int)
val_idx = np.load(VAL_FILE).astype(int)
test_idx = np.load(TEST_FILE).astype(int)

if len(train_idx) != EXPECTED_TRAIN:
    raise ValueError(f"Expected {EXPECTED_TRAIN} training indices.")

if len(val_idx) != EXPECTED_VAL:
    raise ValueError(f"Expected {EXPECTED_VAL} validation indices.")

if len(test_idx) != EXPECTED_TEST:
    raise ValueError(f"Expected {EXPECTED_TEST} test indices.")

for name, indices in [
    ("train", train_idx),
    ("validation", val_idx),
    ("test", test_idx),
]:
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate indices found in {name} split.")

    if np.any(indices < 0) or np.any(indices >= EXPECTED_TRIALS):
        raise ValueError(f"Out-of-range index found in {name} split.")

if (
    set(train_idx) & set(val_idx)
    or set(train_idx) & set(test_idx)
    or set(val_idx) & set(test_idx)
):
    raise ValueError("Split overlap detected.")

all_indices = np.concatenate([train_idx, val_idx, test_idx])

if not np.array_equal(np.sort(all_indices), np.arange(EXPECTED_TRIALS)):
    raise ValueError(
        "The splits do not partition all 720 trials exactly once."
    )

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("Split verification passed.")


# =============================================================================
# CREATE FIXED DATA SPLITS
# =============================================================================

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# =============================================================================
# DEFINE CLASSIFIERS
# =============================================================================

# Every pipeline fits its own scaler on training data only.
# This also ensures each model receives consistently standardized features.

models = {
    "logistic_regression": LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    ),

    "linear_svm": LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=RANDOM_STATE,
    ),

    "rbf_svm": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    ),
}


# =============================================================================
# TRAIN AND EVALUATE ON TRAINING / VALIDATION SETS
# =============================================================================

print("\n" + "=" * 78)
print("TRAINING MODELS AND EVALUATING VALIDATION PERFORMANCE")
print("=" * 78)

validation_results = []
trained_pipelines = {}

for model_name, estimator in models.items():
    print(f"\n[{len(validation_results) + 1}/{len(models)}] {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clone(estimator)),
    ])

    start_time = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)

    convergence_warning = any(
        "converg" in str(w.message).lower()
        for w in caught_warnings
    )

    row = {
        "model": model_name,
        "training_accuracy": train_metrics["accuracy"],
        "validation_accuracy": val_metrics["accuracy"],
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        "validation_macro_f1": val_metrics["macro_f1"],
        "training_seconds": training_seconds,
        "convergence_warning": convergence_warning,
    }

    validation_results.append(row)
    trained_pipelines[model_name] = pipeline

    print(f"Training accuracy:          {train_metrics['accuracy']:.4f}")
    print(f"Validation accuracy:         {val_metrics['accuracy']:.4f}")
    print(f"Validation balanced accuracy:{val_metrics['balanced_accuracy']:.4f}")
    print(f"Validation macro F1:         {val_metrics['macro_f1']:.4f}")
    print(f"Training time:               {training_seconds:.2f} seconds")

    if convergence_warning:
        print("WARNING: A convergence-related warning was raised.")


# =============================================================================
# RANK MODELS USING VALIDATION MACRO F1
# =============================================================================

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by=[
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_accuracy",
    ],
    ascending=False,
).reset_index(drop=True)

results_df.insert(
    0, "validation_rank", np.arange(1, len(results_df) + 1)
)

validation_csv = OUTPUT_DIR / "step59_validation_results.csv"
results_df.to_csv(validation_csv, index=False)

print("\n" + "=" * 78)
print("STEP 59 — VALIDATION RANKING")
print("Primary ranking metric: validation macro F1")
print("=" * 78)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# SELECT WINNER USING VALIDATION ONLY
# =============================================================================

selected_name = results_df.iloc[0]["model"]
selected_pipeline = trained_pipelines[selected_name]

print("\nSelected model using validation metrics:", selected_name)
print("Validation macro F1:",
      f"{results_df.iloc[0]['validation_macro_f1']:.4f}")


# =============================================================================
# HELD-OUT TEST EVALUATION — SELECTED MODEL ONLY
# =============================================================================

print("\n" + "=" * 78)
print("HELD-OUT TEST EVALUATION")
print("=" * 78)

print("The selected model was chosen using validation data only.")
print("The test set was not used to select the model.")

test_pred = selected_pipeline.predict(X_test)
test_metrics = calculate_metrics(y_test, test_pred)

print("Selected model:", selected_name)
print(f"Test accuracy:          {test_metrics['accuracy']:.4f}")
print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
print(f"Test macro F1:           {test_metrics['macro_f1']:.4f}")


# =============================================================================
# SAVE TEST METRICS
# =============================================================================

test_result = {
    "selected_model": selected_name,
    "selection_metric": "validation_macro_f1",
    "validation_accuracy": float(
        results_df.iloc[0]["validation_accuracy"]
    ),
    "validation_balanced_accuracy": float(
        results_df.iloc[0]["validation_balanced_accuracy"]
    ),
    "validation_macro_f1": float(
        results_df.iloc[0]["validation_macro_f1"]
    ),
    "test_accuracy": test_metrics["accuracy"],
    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    "test_macro_f1": test_metrics["macro_f1"],
    "test_trial_count": len(y_test),
}

test_csv = OUTPUT_DIR / "step59_selected_model_test_result.csv"
pd.DataFrame([test_result]).to_csv(test_csv, index=False)


# =============================================================================
# SAVE CLASSIFICATION REPORT
# =============================================================================

report = classification_report(
    y_test,
    test_pred,
    labels=labels,
    target_names=[str(label_names[label]) for label in labels],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_csv = OUTPUT_DIR / "step59_test_classification_report.csv"
report_df.to_csv(report_csv)


# =============================================================================
# SAVE CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_test, test_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label_names[label]}" for label in labels],
    columns=[f"pred_{label_names[label]}" for label in labels],
)

cm_csv = OUTPUT_DIR / "step59_test_confusion_matrix.csv"
cm_df.to_csv(cm_csv)


# =============================================================================
# SAVE TEST PREDICTIONS
# =============================================================================

predictions_df = pd.DataFrame({
    "dataset_index": test_idx,
    "true_label": y_test,
    "true_label_name": [label_names[label] for label in y_test],
    "predicted_label": test_pred,
    "predicted_label_name": [label_names[label] for label in test_pred],
})

predictions_df.to_csv(
    OUTPUT_DIR / "step59_test_predictions.csv",
    index=False,
)


# =============================================================================
# SAVE SELECTED PIPELINE
# =============================================================================

pipeline_file = OUTPUT_DIR / "step59_selected_pipeline.joblib"
joblib.dump(selected_pipeline, pipeline_file)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_file = OUTPUT_DIR / "step59_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:
    file.write("NEUROSYMPHONY — STEP 59 SUMMARY\n")
    file.write("=" * 60 + "\n\n")
    file.write("Subject: 01\n")
    file.write("Features: Original Step 35 band-power features\n")
    file.write("Feature count: 320\n")
    file.write("Trials: 720\n")
    file.write("Classes: 8\n")
    file.write("Split source: Step 35 saved indices\n")
    file.write("Split sizes: 504 train / 108 validation / 108 test\n")
    file.write("Scaling: StandardScaler fitted on training data only\n")
    file.write("Selection metric: Validation macro F1\n")
    file.write("Test evaluation: Selected model only\n\n")
    file.write("Validation ranking:\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nSelected model: " + selected_name + "\n")
    file.write(f"Test accuracy: {test_metrics['accuracy']:.4f}\n")
    file.write(
        f"Test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}\n"
    )
    file.write(f"Test macro F1: {test_metrics['macro_f1']:.4f}\n")


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 59 COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nSaved outputs:")
for path in [
    validation_csv,
    test_csv,
    report_csv,
    cm_csv,
    OUTPUT_DIR / "step59_test_predictions.csv",
    pipeline_file,
    summary_file,
]:
    print(path)
