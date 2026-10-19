
"""
NeuroSymphony — Step 58
Compare 22 EEG filtering conditions using the exact Step 35 split.

Protocol:
- Reuse Step 35 train/validation/test indices.
- Fit StandardScaler on training data only.
- Rank all filtering conditions using validation metrics only.
- Evaluate only the selected filter on the held-out test set.
- Preserve the original Step 35 and Step 57 results.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step55"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "sub01"

TRAIN_INDICES_FILE = SPLIT_DIR / "train_indices.npy"
VAL_INDICES_FILE = SPLIT_DIR / "val_indices.npy"
TEST_INDICES_FILE = SPLIT_DIR / "test_indices.npy"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step58_same_split"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SUFFIXES = (
    "_delta_power",
    "_theta_power",
    "_alpha_power",
    "_beta_power",
    "_gamma_power",
)

EXPECTED_CONDITIONS = 22
EXPECTED_TRIALS = 720
EXPECTED_FEATURES = 320
EXPECTED_CLASSES = 8

EXPECTED_COUNTS = {
    "train": 504,
    "validation": 108,
    "test": 108,
}


# ---------------------------------------------------------
# 2. METRIC HELPER
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# 3. LOAD AND VALIDATE THE STEP 35 SPLIT
# ---------------------------------------------------------

print("=" * 72)
print("NEUROSYMPHONY — STEP 58: SAME-SPLIT FILTER COMPARISON")
print("=" * 72)

for path in (
    TRAIN_INDICES_FILE,
    VAL_INDICES_FILE,
    TEST_INDICES_FILE,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 35 split file not found: {path}"
        )

if not FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"Feature directory not found: {FEATURE_DIR}"
    )

train_indices = np.load(TRAIN_INDICES_FILE)
val_indices = np.load(VAL_INDICES_FILE)
test_indices = np.load(TEST_INDICES_FILE)

split_arrays = {
    "train": train_indices,
    "validation": val_indices,
    "test": test_indices,
}

# Check that every split is a one-dimensional integer array.
for name, indices in split_arrays.items():
    if indices.ndim != 1:
        raise ValueError(
            f"{name} indices must be one-dimensional."
        )

    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(
            f"{name} indices must contain integers."
        )

    if len(indices) != EXPECTED_COUNTS[name]:
        raise ValueError(
            f"{name}: expected {EXPECTED_COUNTS[name]} indices, "
            f"found {len(indices)}."
        )

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            f"Duplicate indices found in the {name} split."
        )

# Confirm that splits do not overlap.
for first_name, first_indices in split_arrays.items():
    for second_name, second_indices in split_arrays.items():
        if first_name >= second_name:
            continue

        overlap = np.intersect1d(
            first_indices, second_indices
        )

        if len(overlap):
            raise ValueError(
                f"Split overlap between {first_name} and "
                f"{second_name}: {overlap[:10]}"
            )

# These arrays must represent positions in a 720-trial dataset.
all_indices = np.concatenate(list(split_arrays.values()))

if (
    len(all_indices) != EXPECTED_TRIALS
    or not np.array_equal(
        np.sort(all_indices),
        np.arange(EXPECTED_TRIALS),
    )
):
    raise ValueError(
        "The Step 35 split indices do not partition positions "
        "0 through 719 exactly once. Inspect the Step 35 script "
        "before proceeding; do not assume these are row positions."
    )

# Create a mapping from dataset position to split name.
split_labels = np.full(
    EXPECTED_TRIALS, "", dtype=object
)

for name, indices in split_arrays.items():
    split_labels[indices] = name

split_df = pd.DataFrame({
    "dataset_index": np.arange(EXPECTED_TRIALS),
    "split": split_labels,
})

print("\nStep 35 split loaded successfully:")
print(split_df["split"].value_counts().to_string())
print(f"Train indices: {TRAIN_INDICES_FILE}")
print(f"Validation indices: {VAL_INDICES_FILE}")
print(f"Test indices: {TEST_INDICES_FILE}")


# ---------------------------------------------------------
# 4. FIND THE 22 FEATURE DATASETS
# ---------------------------------------------------------

feature_files = sorted(
    path
    for path in FEATURE_DIR.glob("step55_*.csv")
    if not path.name.endswith("_timing.csv")
    and path.name != "step55_filter_conditions.csv"
)

if len(feature_files) != EXPECTED_CONDITIONS:
    raise ValueError(
        f"Expected {EXPECTED_CONDITIONS} feature datasets, "
        f"found {len(feature_files)} in {FEATURE_DIR}"
    )

print(f"\nFeature datasets found: {len(feature_files)}")


# ---------------------------------------------------------
# 5. VALIDATE FEATURE STRUCTURE AND TRIAL ALIGNMENT
# ---------------------------------------------------------

reference_df = pd.read_csv(feature_files[0])

required_columns = {"trial_index", "label"}

if not required_columns.issubset(reference_df.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required_columns - set(reference_df.columns)}"
    )

feature_columns = [
    column
    for column in reference_df.columns
    if column.endswith(FEATURE_SUFFIXES)
]

if len(feature_columns) != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features, "
        f"found {len(feature_columns)}."
    )

if len(reference_df) != EXPECTED_TRIALS:
    raise ValueError(
        f"Expected {EXPECTED_TRIALS} trials, "
        f"found {len(reference_df)}."
    )

reference_df["trial_index"] = (
    reference_df["trial_index"].astype(int)
)

reference_df = reference_df.sort_values(
    "trial_index"
).reset_index(drop=True)

# The split arrays are positional indices. To map them safely,
# the sorted trial_index values must be exactly 0 through 719.
if not np.array_equal(
    reference_df["trial_index"].to_numpy(),
    np.arange(EXPECTED_TRIALS),
):
    raise ValueError(
        "Sorted trial_index values are not exactly 0..719. "
        "The mapping between Step 35 positions and Step 55 "
        "trial IDs must be verified before training."
    )

if reference_df["label"].nunique() != EXPECTED_CLASSES:
    raise ValueError(
        "Reference dataset does not contain exactly 8 classes."
    )

reference_labels = reference_df["label"].to_numpy()

for path in feature_files:
    check_df = pd.read_csv(path)

    if not {"trial_index", "label"}.issubset(check_df.columns):
        raise ValueError(
            f"Missing trial_index or label in {path.name}"
        )

    check_df["trial_index"] = (
        check_df["trial_index"].astype(int)
    )

    check_features = [
        column
        for column in check_df.columns
        if column.endswith(FEATURE_SUFFIXES)
    ]

    if len(check_df) != EXPECTED_TRIALS:
        raise ValueError(
            f"Unexpected row count in {path.name}"
        )

    if check_features != feature_columns:
        raise ValueError(
            f"Feature-column mismatch in {path.name}"
        )

    if check_df["trial_index"].duplicated().any():
        raise ValueError(
            f"Duplicate trial indices in {path.name}"
        )

    check_df = check_df.sort_values(
        "trial_index"
    ).reset_index(drop=True)

    if not np.array_equal(
        check_df["trial_index"].to_numpy(),
        np.arange(EXPECTED_TRIALS),
    ):
        raise ValueError(
            f"Trial-index mismatch in {path.name}"
        )

    if not np.array_equal(
        check_df["label"].to_numpy(),
        reference_labels,
    ):
        raise ValueError(
            f"Label alignment mismatch in {path.name}"
        )

print(f"Feature count per trial: {len(feature_columns)}")
print("Trial-index alignment: passed")
print("Label alignment across all 22 conditions: passed")


# ---------------------------------------------------------
# 6. TRAIN AND VALIDATE ALL 22 CONDITIONS
# ---------------------------------------------------------

validation_results = []
fitted_models = {}

for file_number, path in enumerate(feature_files, start=1):
    condition = path.stem.replace("step55_", "")

    print(
        f"\n[{file_number}/{len(feature_files)}] "
        f"Evaluating: {condition}"
    )

    df = pd.read_csv(path)

    df["trial_index"] = df["trial_index"].astype(int)

    # Attach split labels by trial ID, not by arbitrary CSV row order.
    df = df.merge(
        split_df.rename(
            columns={"dataset_index": "trial_index"}
        ),
        on="trial_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if df["split"].isna().any():
        raise ValueError(
            f"Missing split assignment in {condition}"
        )

    # Sort for deterministic indexing and alignment.
    df = df.sort_values(
        "trial_index"
    ).reset_index(drop=True)

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    split_values = df["split"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            f"NaN or infinite feature values in {condition}"
        )

    train_mask = split_values == "train"
    val_mask = split_values == "validation"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    if (
        len(X_train) != EXPECTED_COUNTS["train"]
        or len(X_val) != EXPECTED_COUNTS["validation"]
    ):
        raise ValueError(
            f"Unexpected train/validation sizes in {condition}"
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    convergence_warning = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    # Validation only: test data is not used for model selection.
    y_val_pred = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)

    validation_results.append({
        "condition": condition,
        "feature_file": path.name,
        "train_trials": len(X_train),
        "validation_trials": len(X_val),
        "validation_accuracy": metrics["accuracy"],
        "validation_balanced_accuracy": (
            metrics["balanced_accuracy"]
        ),
        "validation_macro_f1": metrics["macro_f1"],
        "convergence_warning": convergence_warning,
    })

    fitted_models[condition] = pipeline

    print(f"  Validation accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        "  Balanced accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    if convergence_warning:
        print("  WARNING: convergence warning detected.")


# ---------------------------------------------------------
# 7. RANK FILTERS USING VALIDATION METRICS ONLY
# ---------------------------------------------------------

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by=[
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

results_df.insert(
    0,
    "validation_rank",
    np.arange(1, len(results_df) + 1),
)

validation_output = (
    OUTPUT_DIR / "step58_validation_results.csv"
)

results_df.to_csv(validation_output, index=False)

print("\n" + "=" * 72)
print("STEP 58 — FILTER RANKING ON VALIDATION SET")
print("=" * 72)

print(
    results_df[
        [
            "validation_rank",
            "condition",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_macro_f1",
            "convergence_warning",
        ]
    ].to_string(index=False)
)

winner = results_df.iloc[0]
winner_condition = winner["condition"]
winner_model = fitted_models[winner_condition]

print("\nSelected filter based on validation metrics:")
print(winner_condition)
print(
    "Validation accuracy: "
    f"{winner['validation_accuracy'] * 100:.2f}%"
)


# ---------------------------------------------------------
# 8. EVALUATE THE SELECTED FILTER ON THE TEST SET
# ---------------------------------------------------------

winner_path = FEATURE_DIR / winner["feature_file"]

test_df = pd.read_csv(winner_path)
test_df["trial_index"] = test_df["trial_index"].astype(int)

test_df = test_df.merge(
    split_df.rename(
        columns={"dataset_index": "trial_index"}
    ),
    on="trial_index",
    how="left",
    validate="one_to_one",
)

test_df = test_df.loc[
    test_df["split"] == "test"
].copy()

test_df = test_df.sort_values(
    "trial_index"
).reset_index(drop=True)

if len(test_df) != EXPECTED_COUNTS["test"]:
    raise ValueError(
        f"Expected 108 test trials, found {len(test_df)}"
    )

X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
y_test = test_df["label"].to_numpy(dtype=int)

if not np.isfinite(X_test).all():
    raise ValueError(
        "NaN or infinite values in the selected filter's test features."
    )

# The fitted pipeline is not refitted on test data.
y_test_pred = winner_model.predict(X_test)
test_metrics = calculate_metrics(y_test, y_test_pred)

test_result = {
    "selected_condition": winner_condition,
    "selection_basis": (
        "validation accuracy, balanced accuracy, macro F1"
    ),
    "test_trials": len(X_test),
    "test_accuracy": test_metrics["accuracy"],
    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    "test_macro_f1": test_metrics["macro_f1"],
}

test_output = (
    OUTPUT_DIR / "step58_selected_filter_test_result.csv"
)

pd.DataFrame([test_result]).to_csv(
    test_output, index=False
)

model_output = (
    OUTPUT_DIR / "step58_selected_pipeline.joblib"
)

joblib.dump(winner_model, model_output)


# ---------------------------------------------------------
# 9. SAVE SUMMARY AND SPLIT MANIFEST
# ---------------------------------------------------------

split_manifest_output = (
    OUTPUT_DIR / "step58_split_manifest.csv"
)

split_df.to_csv(split_manifest_output, index=False)

summary_output = OUTPUT_DIR / "step58_summary.txt"

summary_lines = [
    "NEUROSYMPHONY — STEP 58 SUMMARY",
    "=" * 50,
    "Split source: Step 35 .npy index files",
    f"Train indices file: {TRAIN_INDICES_FILE}",
    f"Validation indices file: {VAL_INDICES_FILE}",
    f"Test indices file: {TEST_INDICES_FILE}",
    f"Feature datasets evaluated: {len(feature_files)}",
    f"Features per trial: {len(feature_columns)}",
    f"Training trials: {EXPECTED_COUNTS['train']}",
    f"Validation trials: {EXPECTED_COUNTS['validation']}",
    f"Test trials: {EXPECTED_COUNTS['test']}",
    "",
    "Selected filter (validation ranking):",
    winner_condition,
    f"Validation accuracy: {winner['validation_accuracy']:.6f}",
    (
        "Validation balanced accuracy: "
        f"{winner['validation_balanced_accuracy']:.6f}"
    ),
    f"Validation macro F1: {winner['validation_macro_f1']:.6f}",
    "",
    "Held-out test metrics for selected filter:",
    f"Test accuracy: {test_metrics['accuracy']:.6f}",
    (
        "Test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.6f}"
    ),
    f"Test macro F1: {test_metrics['macro_f1']:.6f}",
    "",
    "Filter ranking used validation metrics only.",
    "Test data was not used for filter selection.",
    "Results apply to subject 01.",
]

summary_output.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 72)
print("HELD-OUT TEST RESULTS — STEP 58")
print("=" * 72)
print(f"Selected filter: {winner_condition}")
print(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")
print(
    "Test balanced accuracy: "
    f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
)
print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")

print("\nSaved outputs:")
print(f"Validation ranking: {validation_output}")
print(f"Test result: {test_output}")
print(f"Pipeline: {model_output}")
print(f"Summary: {summary_output}")
print(f"Split manifest: {split_manifest_output}")

print("\nSTEP 58 COMPLETED SUCCESSFULLY.")
