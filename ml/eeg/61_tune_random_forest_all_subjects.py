
"""
NeuroSymphony — Step 61: Random Forest Hyperparameter Tuning
Subjects 02–12

Protocol:
- Use each subject's original band-power features.
- Train on the saved training split only.
- Select hyperparameters using validation macro F1.
- Use validation balanced accuracy and accuracy as tie-breakers.
- Do not evaluate test-set performance.
- Preserve existing Subject 01 results and avoid overwriting outputs.
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_BANDS = 5
EXPECTED_FEATURES = EXPECTED_CHANNELS * EXPECTED_BANDS
EXPECTED_CLASSES = 8

SUBJECTS = range(2, 13)

N_ESTIMATORS_VALUES = [100, 300, 500]
MAX_DEPTH_VALUES = [5, 10, None]
MIN_SAMPLES_LEAF_VALUES = [1, 3]
CLASS_WEIGHT_VALUES = [None, "balanced"]


# ============================================================
# HYPERPARAMETER GRID — 36 CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
    }
    for n_estimators in N_ESTIMATORS_VALUES
    for max_depth in MAX_DEPTH_VALUES
    for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES
    for class_weight in CLASS_WEIGHT_VALUES
]


# ============================================================
# SUBJECT-SPECIFIC PATHS
# ============================================================

def get_subject_paths(subject_number):
    sid = f"{subject_number:02d}"

    feature_dir = (
        ROOT
        / "data"
        / "features"
        / f"temporal_spectral_sub-{sid}"
    )

    metadata_file = (
        ROOT
        / "data"
        / "features"
        / f"temporal_eeg_sub-{sid}"
        / f"sub-{sid}_temporal_eeg_preprocessed_metadata.csv"
    )

    split_dir = ROOT / "data" / "splits" / f"sub-{sid}"

    output_dir = (
        ROOT
        / "results"
        / "eeg"
        / "model_comparison"
        / f"sub-{sid}"
        / "step61_random_forest_tuning"
    )

    return {
        "sid": sid,
        "feature": feature_dir / f"sub-{sid}_band_power.npy",
        "metadata": metadata_file,
        "train": split_dir / "train_indices.npy",
        "val": split_dir / "val_indices.npy",
        "test": split_dir / "test_indices.npy",
        "output": output_dir,
    }


# ============================================================
# LOAD AND VALIDATE ONE SUBJECT
# ============================================================

def load_subject_data(paths):
    sid = paths["sid"]

    required_files = [
        paths["feature"],
        paths["metadata"],
        paths["train"],
        paths["val"],
        paths["test"],
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Subject {sid}: required file not found:\n"
                f"{file_path}"
            )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    features = np.load(paths["feature"])

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_BANDS,
    )

    if features.shape != expected_shape:
        raise ValueError(
            f"Subject {sid}: unexpected feature shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found: {features.shape}"
        )

    X = features.reshape(
        EXPECTED_TRIALS,
        EXPECTED_FEATURES,
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Subject {sid}: features contain NaN or infinity."
        )

    # --------------------------------------------------------
    # Load trial-level labels
    # --------------------------------------------------------

    metadata = pd.read_csv(paths["metadata"])

    if len(metadata) != EXPECTED_TRIALS:
        raise ValueError(
            f"Subject {sid}: expected 720 metadata rows, "
            f"found {len(metadata)}."
        )

    if "label_id" in metadata.columns:
        label_column = "label_id"
    elif "label" in metadata.columns:
        label_column = "label"
    else:
        raise ValueError(
            f"Subject {sid}: metadata has neither "
            "'label_id' nor 'label'."
        )

    y = metadata[label_column].to_numpy()

    if pd.isna(y).any():
        raise ValueError(
            f"Subject {sid}: missing labels detected."
        )

    try:
        y_numeric = y.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Subject {sid}: labels must be integer class IDs."
        ) from exc

    if not np.array_equal(y, y_numeric):
        raise ValueError(
            f"Subject {sid}: labels contain non-integer values."
        )

    y = y_numeric

    if not np.array_equal(
        np.sort(np.unique(y)),
        np.arange(EXPECTED_CLASSES),
    ):
        raise ValueError(
            f"Subject {sid}: expected class IDs 0–7, "
            f"found {np.unique(y).tolist()}."
        )

    label_counts = pd.Series(y).value_counts().sort_index()

    if not np.all(label_counts.to_numpy() == 90):
        raise ValueError(
            f"Subject {sid}: expected 90 trials per class; "
            f"found {label_counts.to_dict()}."
        )

    # Check sequential trial identifiers when present.
    trial_column = (
        "trial_index"
        if "trial_index" in metadata.columns
        else "trial_id"
        if "trial_id" in metadata.columns
        else None
    )

    if trial_column is not None:
        trial_ids = metadata[trial_column].to_numpy()

        if not np.array_equal(
            trial_ids,
            np.arange(EXPECTED_TRIALS),
        ):
            raise ValueError(
                f"Subject {sid}: {trial_column} is not "
                "sequential from 0 to 719."
            )

    # --------------------------------------------------------
    # Load and validate saved splits
    # --------------------------------------------------------

    train_idx = np.load(paths["train"]).astype(int)
    val_idx = np.load(paths["val"]).astype(int)
    test_idx = np.load(paths["test"]).astype(int)

    expected_sizes = [
        ("training", train_idx, 504),
        ("validation", val_idx, 108),
        ("test", test_idx, 108),
    ]

    for name, indices, expected_size in expected_sizes:
        if len(indices) != expected_size:
            raise ValueError(
                f"Subject {sid}: {name} split has "
                f"{len(indices)} indices; expected {expected_size}."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                f"Subject {sid}: duplicate indices in {name} split."
            )

        if np.any(indices < 0) or np.any(
            indices >= EXPECTED_TRIALS
        ):
            raise ValueError(
                f"Subject {sid}: out-of-range index in {name} split."
            )

    if (
        set(train_idx) & set(val_idx)
        or set(train_idx) & set(test_idx)
        or set(val_idx) & set(test_idx)
    ):
        raise ValueError(
            f"Subject {sid}: overlap detected between splits."
        )

    all_indices = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    if not np.array_equal(
        np.sort(all_indices),
        np.arange(EXPECTED_TRIALS),
    ):
        raise ValueError(
            f"Subject {sid}: splits do not partition all 720 trials."
        )

    return X, y, train_idx, val_idx


# ============================================================
# TUNE RANDOM FOREST FOR ONE SUBJECT
# ============================================================

def tune_subject(subject_number):
    paths = get_subject_paths(subject_number)
    sid = paths["sid"]
    output_dir = paths["output"]

    print("\n" + "=" * 78, flush=True)
    print(f"STEP 61 — SUBJECT {sid}", flush=True)
    print("=" * 78, flush=True)

    # Create output directory without deleting existing files.
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_file = (
        output_dir / "step61_validation_results.csv"
    )
    pipeline_file = (
        output_dir / "step61_selected_pipeline.joblib"
    )
    summary_file = (
        output_dir / "step61_summary.txt"
    )

    # Do not silently overwrite previous experiment results.
    existing_outputs = [
        path
        for path in (
            ranking_file,
            pipeline_file,
            summary_file,
        )
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            f"Subject {sid}: Step 61 output already exists.\n"
            f"Inspect or back up these files before rerunning:\n"
            + "\n".join(str(path) for path in existing_outputs)
        )

    X, y, train_idx, val_idx = load_subject_data(paths)

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    print(f"Feature shape: {X.shape}", flush=True)
    print(f"Training trials: {len(train_idx)}", flush=True)
    print(f"Validation trials: {len(val_idx)}", flush=True)
    print(f"Configurations: {len(CONFIGURATIONS)}", flush=True)
    print(
        "Test split: integrity checked only; not evaluated.",
        flush=True,
    )

    results = []
    best_pipeline = None
    best_params = None
    best_key = None
    best_row = None

    # --------------------------------------------------------
    # Train and evaluate using training and validation only
    # --------------------------------------------------------

    for config_index, params in enumerate(CONFIGURATIONS):
        print(
            f"\nSubject {sid} — "
            f"configuration {config_index + 1}/"
            f"{len(CONFIGURATIONS)}",
            flush=True,
        )
        print(f"Parameters: {params}", flush=True)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    **params,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

        start_time = time.perf_counter()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)

        elapsed_seconds = time.perf_counter() - start_time

        train_predictions = pipeline.predict(X_train)
        val_predictions = pipeline.predict(X_val)

        row = {
            "configuration_index": config_index,
            **params,
            "training_accuracy": accuracy_score(
                y_train, train_predictions
            ),
            "validation_accuracy": accuracy_score(
                y_val, val_predictions
            ),
            "validation_balanced_accuracy": balanced_accuracy_score(
                y_val, val_predictions
            ),
            "validation_macro_f1": f1_score(
                y_val,
                val_predictions,
                average="macro",
                zero_division=0,
            ),
            "training_seconds": elapsed_seconds,
            "warning_count": len(caught),
        }

        results.append(row)

        # Rank by validation macro F1, then balanced accuracy,
        # then accuracy, matching the original Step 61 protocol.
        ranking_key = (
            row["validation_macro_f1"],
            row["validation_balanced_accuracy"],
            row["validation_accuracy"],
        )

        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_pipeline = pipeline
            best_params = dict(params)
            best_row = dict(row)

        print(
            f"Train accuracy: {row['training_accuracy']:.4f} | "
            f"Validation accuracy: {row['validation_accuracy']:.4f} | "
            f"Validation balanced accuracy: "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"Validation macro F1: "
            f"{row['validation_macro_f1']:.4f} | "
            f"Time: {elapsed_seconds:.1f}s",
            flush=True,
        )

    # --------------------------------------------------------
    # Save validation ranking
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "validation_rank",
        np.arange(1, len(results_df) + 1),
    )

    results_df.to_csv(ranking_file, index=False)

    # Save the already-trained selected pipeline.
    joblib.dump(best_pipeline, pipeline_file)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("NEUROSYMPHONY — STEP 61 SUMMARY\n")
        file.write("=" * 60 + "\n")
        file.write(f"Subject: {sid}\n")
        file.write("Features: Original band-power features\n")
        file.write("Feature shape: 720 x 64 x 5\n")
        file.write("Flattened feature count: 320\n")
        file.write("Training trials: 504\n")
        file.write("Validation trials: 108\n")
        file.write("Test trials: 108\n")
        file.write(
            f"Configurations evaluated: {len(CONFIGURATIONS)}\n"
        )
        file.write("Selection metric: Validation macro F1\n")
        file.write("Test evaluation: NOT performed\n\n")

        file.write("Selected hyperparameters:\n")
        for name, value in best_params.items():
            file.write(f"{name}: {value}\n")

        file.write("\nSelected configuration metrics:\n")
        for name, value in best_row.items():
            file.write(f"{name}: {value}\n")

        file.write("\nTop 10 validation configurations:\n")
        file.write(
            results_df.head(10).to_string(index=False)
        )
        file.write("\n")

    print("\n" + "-" * 78, flush=True)
    print(f"SUBJECT {sid} STEP 61 COMPLETED", flush=True)
    print(f"Selected parameters: {best_params}", flush=True)
    print(
        f"Validation macro F1: "
        f"{best_row['validation_macro_f1']:.4f}",
        flush=True,
    )
    print(f"Ranking: {ranking_file}", flush=True)
    print(f"Pipeline: {pipeline_file}", flush=True)
    print(f"Summary: {summary_file}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78, flush=True)
    print(
        "STEP 61 — RANDOM FOREST TUNING FOR SUBJECTS 02–12",
        flush=True,
    )
    print(
        "Subject 01 is excluded to preserve its existing results.",
        flush=True,
    )
    print(
        f"Configurations per subject: {len(CONFIGURATIONS)}",
        flush=True,
    )
    print(
        f"Subjects to process: {len(list(SUBJECTS))}",
        flush=True,
    )
    print(
        "Test-set performance will not be evaluated.",
        flush=True,
    )

    for subject_number in SUBJECTS:
        tune_subject(subject_number)

    print("\n" + "=" * 78, flush=True)
    print("STEP 61 COMPLETED FOR SUBJECTS 02–12", flush=True)
    print("No test-set performance was evaluated.", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
