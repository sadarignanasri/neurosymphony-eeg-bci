
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mne.decoding import CSP

warnings.filterwarnings("ignore")

# ============================================================
# EXPERIMENT 102
# Resource-conscious CSP + SVM optimization
# Dataset: ds004022
#
# Outer validation: Leave-One-Subject-Out (LOSO)
# Inner validation: Grouped cross-validation by subject
# Parallel processing: Disabled
# ============================================================

ROOT = Path(r"D:\neurosymphony-eeg-bci")

INPUT_PATH = (
    ROOT
    / "experiments"
    / "ds004022_verified_epochs"
    / "eeg_epochs.npz"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "ds004022_experiment102_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500.0
RANDOM_STATE = 42

BAND_CONFIGS = {
    "mu_beta": [(8, 13), (13, 30)],
    "wide_motor": [(7, 30)],
    "mu_only": [(8, 13)],
}

CSP_COMPONENTS = [2, 4]

PARAM_GRID = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.1, 1, 10],
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale"],
    },
]


def load_data():
    """Load existing epochs without modifying the dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"EEG dataset not found: {INPUT_PATH}"
        )

    with np.load(INPUT_PATH, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"]).reshape(-1)
    subjects = np.asarray(data["subjects"]).astype(str).reshape(-1)

    if X.ndim != 3:
        raise ValueError(
            f"Expected (trials, channels, samples), got {X.shape}"
        )

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            "EEG trials, labels, and subject IDs have different lengths."
        )

    if not np.isfinite(X).all():
        raise ValueError("EEG contains NaN or infinite values.")

    if len(np.unique(y)) != 4:
        raise ValueError(
            f"Expected four classes, found {np.unique(y)}"
        )

    if len(np.unique(subjects)) < 4:
        raise ValueError(
            "Expected at least four subjects for this experiment."
        )

    print("\n========== DATASET ==========")
    print("EEG shape:", X.shape)
    print("Class counts:", np.unique(y, return_counts=True))
    print("Subjects:", np.unique(subjects))
    print("Sampling rate:", FS, "Hz")

    return X, y, subjects


def filter_epochs(X, low, high):
    """Apply a fixed band-pass filter independently to each epoch."""

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    filtered = np.empty_like(X)

    for i in range(len(X)):
        filtered[i] = sosfiltfilt(
            sos,
            X[i],
            axis=-1,
        )

    return filtered


def build_features(X, bands):
    """
    Filter each frequency band and concatenate along the channel axis.

    This is band-concatenated CSP, not independent CSP per band.
    """

    band_arrays = []

    for low, high in bands:
        print(f"  Filtering {low}-{high} Hz")
        band_arrays.append(filter_epochs(X, low, high))

    return np.concatenate(band_arrays, axis=1)


def make_pipeline(n_components):
    """CSP is fitted inside each training fold."""

    return Pipeline([
        (
            "csp",
            CSP(
                n_components=n_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            ),
        ),
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ])


def save_results(rows):
    """Save progress after every completed subject evaluation."""

    if not rows:
        return

    fold_df = pd.DataFrame(rows)

    fold_df.to_csv(
        OUTPUT_DIR / "loso_fold_results_partial.csv",
        index=False,
    )

    summary = (
        fold_df.groupby(
            ["band_config", "bands_hz", "csp_components"],
            as_index=False,
        )
        .agg(
            completed_subject_folds=("held_out_subject", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1"],
            ascending=False,
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "configuration_summary_partial.csv",
        index=False,
    )


def main():
    X, y, subjects = load_data()
    unique_subjects = np.unique(subjects)

    partial_path = OUTPUT_DIR / "loso_fold_results_partial.csv"

    # Resume previously completed evaluations if available.
    if partial_path.exists():
        previous = pd.read_csv(partial_path)
        all_rows = previous.to_dict("records")
        print(
            f"\nPreviously completed evaluations: {len(all_rows)}"
        )
    else:
        all_rows = []

    completed_keys = {
        (
            str(row["band_config"]),
            str(row["bands_hz"]),
            int(row["csp_components"]),
            str(row["held_out_subject"]),
        )
        for row in all_rows
    }

    expected_folds = (
        len(BAND_CONFIGS)
        * len(CSP_COMPONENTS)
        * len(unique_subjects)
    )

    for band_name, bands in BAND_CONFIGS.items():
        print(f"\n========== BAND CONFIG: {band_name} ==========")
        print("Bands:", bands)

        X_band = build_features(X, bands)

        for n_components in CSP_COMPONENTS:
            print(f"\nCSP components: {n_components}")

            for held_out in unique_subjects:
                key = (
                    band_name,
                    str(bands),
                    n_components,
                    str(held_out),
                )

                if key in completed_keys:
                    print(f"Skipping completed fold: {held_out}")
                    continue

                print(f"\nHeld-out subject: {held_out}")

                train_mask = subjects != held_out
                test_mask = subjects == held_out

                X_train = X_band[train_mask]
                y_train = y[train_mask]
                groups_train = subjects[train_mask]

                X_test = X_band[test_mask]
                y_test = y[test_mask]

                # Grouped inner CV prevents trials from the same
                # subject appearing in both inner training and validation.
                inner_cv = StratifiedGroupKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                )

                search = GridSearchCV(
                    estimator=make_pipeline(n_components),
                    param_grid=PARAM_GRID,
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                    verbose=0,
                    return_train_score=False,
                )

                search.fit(
                    X_train,
                    y_train,
                    groups=groups_train,
                )

                predictions = search.predict(X_test)

                row = {
                    "band_config": band_name,
                    "bands_hz": str(bands),
                    "csp_components": n_components,
                    "held_out_subject": str(held_out),
                    "accuracy": float(
                        accuracy_score(y_test, predictions)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, predictions)
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "best_inner_balanced_accuracy": float(
                        search.best_score_
                    ),
                    "best_parameters": json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }

                all_rows.append(row)
                completed_keys.add(key)

                print(
                    f"Accuracy: {row['accuracy']:.4f} | "
                    f"Balanced accuracy: "
                    f"{row['balanced_accuracy']:.4f} | "
                    f"Macro F1: {row['macro_f1']:.4f}"
                )
                print("Best parameters:", search.best_params_)

                save_results(all_rows)

    # Produce final summaries.
    fold_df = pd.DataFrame(all_rows)

    summary = (
        fold_df.groupby(
            ["band_config", "bands_hz", "csp_components"],
            as_index=False,
        )
        .agg(
            completed_subject_folds=("held_out_subject", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_inner_balanced_accuracy=(
                "best_inner_balanced_accuracy",
                "mean",
            ),
        )
        .sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1"],
            ascending=False,
        )
    )

    complete = len(fold_df) == expected_folds

    fold_df.to_csv(
        OUTPUT_DIR / "loso_fold_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "configuration_summary.csv",
        index=False,
    )

    report = {
        "experiment": "102",
        "dataset": "ds004022",
        "validation": (
            "Outer leave-one-subject-out; inner StratifiedGroupKFold "
            "using subject IDs as groups"
        ),
        "n_subjects": int(len(unique_subjects)),
        "n_trials": int(len(y)),
        "expected_fold_evaluations": int(expected_folds),
        "completed_fold_evaluations": int(len(fold_df)),
        "all_evaluations_completed": bool(complete),
        "chance_level": 0.25,
        "band_configs": BAND_CONFIGS,
        "csp_components": CSP_COMPONENTS,
        "svm_parameter_grid": PARAM_GRID,
        "best_configuration": (
            summary.iloc[0].to_dict()
            if not summary.empty
            else {}
        ),
        "warning": (
            "The best configuration is selected from multiple comparisons. "
            "Its score is exploratory, not an independent final estimate."
        ),
    }

    with open(
        OUTPUT_DIR / "experiment102_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, default=str)

    print("\n========== EXPERIMENT 102 SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nAll expected evaluations completed:", complete)
    print("Results folder:", OUTPUT_DIR)
    print("Files:")
    print("1. loso_fold_results.csv")
    print("2. configuration_summary.csv")
    print("3. experiment102_report.json")
    print("4. loso_fold_results_partial.csv")
    print("5. configuration_summary_partial.csv")
    print("\nPrevious FBCSP + SVM benchmark: 26.46% LOSO accuracy.")


if __name__ == "__main__":
    main()
