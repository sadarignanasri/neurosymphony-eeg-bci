
from __future__ import annotations

import json
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from mne.decoding import CSP


# ============================================================
# EXPERIMENT 103
# Nested LOSO validation with TRUE Filter Bank CSP
# Dataset: ds004022
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "experiments"
    / "ds004022_verified_epochs"
    / "eeg_epochs.npz"
)

OUT_DIR = ROOT / "experiments" / "ds004022_experiment103_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
N_COMPONENTS_OPTIONS = [2, 4]

# Three candidate filter-bank configurations, defined as
# indices into the canonical frequency bands below.
CANONICAL_BANDS = (
    (8.0, 13.0),   # mu / alpha
    (13.0, 30.0),  # beta
    (7.0, 30.0),   # wider motor-imagery band
)

BAND_CONFIGURATIONS = [
    (0, 1),  # Separate mu and beta CSP feature sets
    (2,),    # Wide motor band
    (0,),    # Mu band only
]

SVM_GRID = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.1, 1.0, 10.0],
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1.0, 10.0],
        "svc__gamma": ["scale"],
    },
]

RANDOM_STATE = 42


def get_array(data, possible_names, required=True):
    """Read an array using one of several possible NPZ key names."""
    for name in possible_names:
        if name in data.files:
            return data[name]

    if required:
        raise KeyError(
            f"Could not find any of {possible_names}. "
            f"Available NPZ keys: {data.files}"
        )
    return None


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Fit a separate CSP model for each selected frequency band.
    Concatenate the resulting CSP features for classification.

    Input shape:
        (trials, frequency_bands, channels, time_samples)
    """

    def __init__(
        self,
        band_indices=(0, 1),
        n_components=4,
        reg=None,
    ):
        self.band_indices = band_indices
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        self.csp_models_ = []

        for band_index in self.band_indices:
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                norm_trace=False,
            )

            csp.fit(X[:, band_index, :, :], y)
            self.csp_models_.append((band_index, csp))

        return self

    def transform(self, X):
        features = []

        for band_index, csp in self.csp_models_:
            band_features = csp.transform(
                X[:, band_index, :, :]
            )
            features.append(band_features)

        return np.concatenate(features, axis=1)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:
        print("Available NPZ keys:", data.files)

        X = get_array(data, ["X", "epochs", "eeg_epochs"])
        y = get_array(data, ["y", "labels", "targets"])
        subjects = get_array(
            data,
            ["subjects", "subject_ids", "subject"],
        )

        class_names = get_array(
            data,
            ["class_names", "classes", "label_names"],
            required=False,
        )

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1).astype(str)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X shape (trials, channels, samples); got {X.shape}"
        )

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            "Mismatch in the number of epochs, labels, and subject IDs."
        )

    if class_names is not None:
        class_names = [str(v) for v in np.asarray(class_names).tolist()]
    else:
        class_names = [str(v) for v in np.unique(y)]

    print("\nDataset verification")
    print("--------------------")
    print("X shape:", X.shape)
    print("Labels:", len(y))
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Classes:", np.unique(y).tolist())
    print("Class names:", class_names)

    if X.shape[1] != 18 or X.shape[2] != 2500:
        print(
            "WARNING: The loaded epoch dimensions differ from the "
            "expected 18 channels x 2500 samples."
        )

    if len(np.unique(y)) != 4:
        raise ValueError(
            "Expected four classes for this experiment."
        )

    return X.astype(np.float32, copy=False), y, subjects, class_names


def prefilter_epochs(X, temp_path):
    """
    Filter each epoch independently and write to a memory-mapped
    temporary array to limit RAM usage.

    Filtering is deterministic and uses no labels or fitted
    parameters from held-out subjects.
    """
    shape = (
        X.shape[0],
        len(CANONICAL_BANDS),
        X.shape[1],
        X.shape[2],
    )

    filtered = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    print("\nPre-filtering epochs...")
    start = time.time()

    for band_index, (low, high) in enumerate(CANONICAL_BANDS):
        print(f"Filtering {low:g}-{high:g} Hz...")

        sos = butter(
            4,
            [low, high],
            btype="bandpass",
            fs=FS,
            output="sos",
        )

        # Filter one epoch at a time to limit temporary RAM use.
        for trial_index in range(X.shape[0]):
            filtered[trial_index, band_index] = sosfiltfilt(
                sos,
                X[trial_index],
                axis=-1,
            )

        filtered.flush()

    print(f"Filtering finished in {(time.time() - start) / 60:.2f} minutes.")
    return filtered


def main():
    X, y, subjects, class_names = load_dataset()

    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        raise ValueError(
            "At least three subjects are required for grouped inner CV."
        )

    temp_path = OUT_DIR / "filtered_epochs_temp.npy"

    fold_results = []
    prediction_frames = []
    all_true = []
    all_pred = []
    selected_configs = []

    try:
        X_filtered = prefilter_epochs(X, temp_path)

        for fold_number, held_out_subject in enumerate(
            unique_subjects, start=1
        ):
            print("\n" + "=" * 65)
            print(
                f"OUTER FOLD {fold_number}/{len(unique_subjects)}"
            )
            print("Held-out subject:", held_out_subject)
            print("=" * 65)

            train_mask = subjects != held_out_subject
            test_mask = subjects == held_out_subject

            X_train = X_filtered[train_mask]
            X_test = X_filtered[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            groups_train = subjects[train_mask]

            inner_cv = StratifiedGroupKFold(
                n_splits=min(3, len(np.unique(groups_train))),
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            pipeline = Pipeline([
                (
                    "fbcsp",
                    FilterBankCSP(),
                ),
                (
                    "svc",
                    SVC(),
                ),
            ])

            param_grid = [
                {
                    "fbcsp__band_indices": BAND_CONFIGURATIONS,
                    "fbcsp__n_components": N_COMPONENTS_OPTIONS,
                    **svm_params,
                }
                for svm_params in SVM_GRID
            ]

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                verbose=1,
                return_train_score=False,
                error_score="raise",
            )

            # All band, CSP, and SVM choices are selected using
            # only the outer training subjects.
            search.fit(
                X_train,
                y_train,
                groups=groups_train,
            )

            # The held-out subject is evaluated only after selection.
            y_pred = search.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            best_params = {
                key: (
                    list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value in search.best_params_.items()
            }

            fold_results.append({
                "held_out_subject": held_out_subject,
                "n_train_trials": int(len(y_train)),
                "n_test_trials": int(len(y_test)),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "macro_f1": float(macro_f1),
                "inner_best_balanced_accuracy": float(
                    search.best_score_
                ),
                "best_params": json.dumps(best_params),
            })

            selected_configs.append({
                "held_out_subject": held_out_subject,
                "best_params": best_params,
                "inner_best_balanced_accuracy": float(
                    search.best_score_
                ),
            })

            prediction_frames.append(pd.DataFrame({
                "subject": held_out_subject,
                "y_true": y_test,
                "y_pred": y_pred,
            }))

            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())

            print("Selected parameters:", best_params)
            print(f"Accuracy:          {accuracy:.4f}")
            print(f"Balanced accuracy: {balanced_acc:.4f}")
            print(f"Macro F1:          {macro_f1:.4f}")

            # Preserve progress after every outer fold.
            pd.DataFrame(fold_results).to_csv(
                OUT_DIR / "loso_fold_results_partial.csv",
                index=False,
            )

            pd.concat(prediction_frames, ignore_index=True).to_csv(
                OUT_DIR / "outer_fold_predictions_partial.csv",
                index=False,
            )

            with open(
                OUT_DIR / "selected_configurations_partial.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(selected_configs, f, indent=2)

            del search, pipeline, X_train, X_test
            gc.collect()

        # Final outputs
        fold_df = pd.DataFrame(fold_results)
        predictions_df = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        fold_df.to_csv(
            OUT_DIR / "loso_fold_results.csv",
            index=False,
        )
        predictions_df.to_csv(
            OUT_DIR / "outer_fold_predictions.csv",
            index=False,
        )

        labels = np.unique(y).tolist()
        cm = confusion_matrix(
            all_true,
            all_pred,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{v}" for v in labels],
            columns=[f"pred_{v}" for v in labels],
        )
        cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

        report = classification_report(
            all_true,
            all_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        with open(
            OUT_DIR / "classification_report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(report, f, indent=2)

        summary = {
            "experiment": 103,
            "method": "Nested LOSO with separate CSP per selected frequency band",
            "dataset": "ds004022",
            "trials": int(len(y)),
            "channels": int(X.shape[1]),
            "samples_per_epoch": int(X.shape[2]),
            "sampling_frequency_hz": FS,
            "subjects": unique_subjects.tolist(),
            "classes": class_names,
            "chance_level": 1.0 / len(labels),
            "mean_outer_accuracy": float(fold_df["accuracy"].mean()),
            "std_outer_accuracy": float(fold_df["accuracy"].std(ddof=0)),
            "mean_outer_balanced_accuracy": float(
                fold_df["balanced_accuracy"].mean()
            ),
            "mean_outer_macro_f1": float(fold_df["macro_f1"].mean()),
            "pooled_accuracy": float(
                accuracy_score(all_true, all_pred)
            ),
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(all_true, all_pred)
            ),
            "pooled_macro_f1": float(
                f1_score(
                    all_true,
                    all_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "selected_configurations_by_outer_fold": selected_configs,
            "note": (
                "All hyperparameter selection occurs within inner "
                "grouped CV. Each outer held-out subject is evaluated "
                "after selection. Results are estimates from only four "
                "subjects and should be interpreted cautiously."
            ),
        }

        with open(
            OUT_DIR / "experiment103_report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 65)
        print("EXPERIMENT 103 COMPLETE")
        print("=" * 65)
        print(f"Mean outer LOSO accuracy: "
              f"{summary['mean_outer_accuracy']:.4f}")
        print(f"Mean balanced accuracy:   "
              f"{summary['mean_outer_balanced_accuracy']:.4f}")
        print(f"Mean macro F1:             "
              f"{summary['mean_outer_macro_f1']:.4f}")
        print(f"Pooled accuracy:           "
              f"{summary['pooled_accuracy']:.4f}")
        print(f"Chance level:              "
              f"{summary['chance_level']:.4f}")
        print("\nResults saved to:")
        print(OUT_DIR)

    finally:
        # Close the memory map before deleting its temporary file.
        if "X_filtered" in locals():
            del X_filtered
            gc.collect()

        if temp_path.exists():
            try:
                temp_path.unlink()
                print("Temporary filtered data removed.")
            except OSError:
                print(
                    "Temporary filtered data could not be removed. "
                    f"You can delete it after Python has closed: {temp_path}"
                )


if __name__ == "__main__":
    main()
