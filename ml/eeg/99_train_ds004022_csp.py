
from pathlib import Path
import json
import sys
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "ds004022_verified_epochs"
    / "eeg_epochs.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "ds004022_csp_results"
)

MODEL_DIR = OUTPUT_DIR / "fold_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CSP_COMPONENTS = 4
FILTER_ORDER = 4

CLASS_MAP = {
    "0": "reaching",
    "1": "grasping",
    "2": "lifting",
    "3": "twisting",
}

FREQUENCY_BANDS = [
    ("mu_low", 8.0, 12.0),
    ("mu_beta", 12.0, 16.0),
    ("beta_low", 16.0, 22.0),
    ("beta_high", 22.0, 30.0),
]

CLASS_NAMES = [
    "reaching",
    "grasping",
    "lifting",
    "twisting",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"EEG epochs not found:\n{INPUT_FILE}\n"
            "Run script 97 first."
        )

    with np.load(INPUT_FILE, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y_raw = np.asarray(data["y"]).astype(str)
        subjects = np.asarray(data["subjects"]).astype(str)

        sampling_rate = (
            float(np.asarray(data["sampling_rate"]).item())
            if "sampling_rate" in data.files
            else 500.0
        )

        channel_names = (
            np.asarray(data["channel_names"]).astype(str).tolist()
            if "channel_names" in data.files
            else [f"EEG_{i + 1}" for i in range(X.shape[1])]
        )

    if X.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {X.shape}")

    if not (len(X) == len(y_raw) == len(subjects)):
        raise ValueError("Trial, label, and subject counts differ.")

    if not np.isfinite(X).all():
        raise ValueError("EEG contains NaN or infinite values.")

    if sampling_rate != 500.0:
        raise ValueError(
            f"Expected 500 Hz, found {sampling_rate} Hz."
        )

    # Convert numeric labels into readable class names.
    if set(np.unique(y_raw)).issubset(set(CLASS_MAP)):
        y = np.array([CLASS_MAP[label] for label in y_raw])
    else:
        y = y_raw

    if set(np.unique(y)) != set(CLASS_NAMES):
        raise ValueError(
            "Unexpected class labels: "
            f"{np.unique(y).tolist()}. "
            "Verify CLASS_MAP against the extraction metadata."
        )

    if len(np.unique(subjects)) < 2:
        raise ValueError("LOSO requires at least two subjects.")

    return X, y, subjects, sampling_rate, channel_names


# ============================================================
# 3. BAND-PASS FILTERING
# ============================================================

def bandpass_epochs(X, sampling_rate, low_hz, high_hz):
    """Filter trials along the time axis."""

    if not (0 < low_hz < high_hz < sampling_rate / 2):
        raise ValueError(
            f"Invalid frequency band: {low_hz}-{high_hz} Hz."
        )

    sos = butter(
        FILTER_ORDER,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    filtered = sosfiltfilt(sos, X, axis=-1)

    if not np.isfinite(filtered).all():
        raise ValueError("Filtering produced invalid values.")

    return filtered


# ============================================================
# 4. CSP + LDA
# ============================================================

def train_csp_lda(X_train, y_train, X_test):
    """
    CSP is fitted only on the training fold.
    LDA learns from the training fold's CSP features.
    """

    csp = CSP(
        n_components=CSP_COMPONENTS,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
        transform_into="average_power",
        component_order="mutual_info",
    )

    train_features = csp.fit_transform(X_train, y_train)
    test_features = csp.transform(X_test)

    classifier = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    )

    classifier.fit(train_features, y_train)
    predictions = classifier.predict(test_features)

    return predictions, {
        "csp": csp,
        "classifier": classifier,
    }


# ============================================================
# 5. FILTER BANK CSP + SVM
# ============================================================

def train_fbcsp_svm(
    X_train,
    y_train,
    X_test,
    sampling_rate,
):
    """
    Fit a separate CSP for each frequency band.
    Concatenate the features and train an RBF SVM.
    """

    train_blocks = []
    test_blocks = []
    csp_models = []

    for band_name, low_hz, high_hz in FREQUENCY_BANDS:
        X_train_band = bandpass_epochs(
            X_train, sampling_rate, low_hz, high_hz
        )
        X_test_band = bandpass_epochs(
            X_test, sampling_rate, low_hz, high_hz
        )

        csp = CSP(
            n_components=CSP_COMPONENTS,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
            transform_into="average_power",
            component_order="mutual_info",
        )

        train_features = csp.fit_transform(
            X_train_band, y_train
        )
        test_features = csp.transform(X_test_band)

        train_blocks.append(train_features)
        test_blocks.append(test_features)

        csp_models.append({
            "band_name": band_name,
            "low_hz": low_hz,
            "high_hz": high_hz,
            "csp": csp,
        })

    X_train_fbcsp = np.concatenate(train_blocks, axis=1)
    X_test_fbcsp = np.concatenate(test_blocks, axis=1)

    classifier = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    classifier.fit(X_train_fbcsp, y_train)
    predictions = classifier.predict(X_test_fbcsp)

    return predictions, {
        "csp_models": csp_models,
        "classifier": classifier,
    }


# ============================================================
# 6. LOSO EVALUATION
# ============================================================

def run_loso(X, y, subjects, sampling_rate, channel_names):
    models = ["CSP_LDA", "FBCSP_SVM"]

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups=subjects))

    fold_results = []
    prediction_results = []

    pooled_true = {name: [] for name in models}
    pooled_pred = {name: [] for name in models}

    best_fold_score = {name: -1.0 for name in models}

    print("\n" + "=" * 70)
    print("LEAVE-ONE-SUBJECT-OUT EVALUATION")
    print("=" * 70)
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Classes:", CLASS_NAMES)
    print("Number of folds:", len(splits))

    for fold, (train_idx, test_idx) in enumerate(
        splits, start=1
    ):
        held_out = np.unique(subjects[test_idx])

        if len(held_out) != 1:
            raise RuntimeError("LOSO test fold has multiple subjects.")

        subject = str(held_out[0])

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if set(np.unique(y_train)) != set(CLASS_NAMES):
            raise ValueError(
                f"Fold {fold} training data is missing a class."
            )

        print(
            f"\nFold {fold}/{len(splits)} | "
            f"Held-out subject: {subject} | "
            f"Train: {len(train_idx)} | Test: {len(test_idx)}"
        )

        # CSP + LDA uses the 8-30 Hz motor-imagery range.
        X_train_csp = bandpass_epochs(
            X_train, sampling_rate, 8.0, 30.0
        )
        X_test_csp = bandpass_epochs(
            X_test, sampling_rate, 8.0, 30.0
        )

        print("  Training CSP + LDA...", flush=True)

        pred_csp, csp_objects = train_csp_lda(
            X_train_csp, y_train, X_test_csp
        )

        print("  Training FBCSP + SVM...", flush=True)

        pred_fbcsp, fbcsp_objects = train_fbcsp_svm(
            X_train,
            y_train,
            X_test,
            sampling_rate,
        )

        fold_predictions = {
            "CSP_LDA": pred_csp,
            "FBCSP_SVM": pred_fbcsp,
        }

        for model_name, predictions in fold_predictions.items():
            accuracy = accuracy_score(y_test, predictions)
            balanced = balanced_accuracy_score(y_test, predictions)
            macro_f1 = f1_score(
                y_test,
                predictions,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0,
            )

            print(
                f"    {model_name}: "
                f"accuracy={accuracy:.4f}, "
                f"balanced_accuracy={balanced:.4f}, "
                f"macro_F1={macro_f1:.4f}"
            )

            fold_results.append({
                "fold": fold,
                "held_out_subject": subject,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "macro_f1": macro_f1,
            })

            pooled_true[model_name].extend(y_test.tolist())
            pooled_pred[model_name].extend(predictions.tolist())

            for local_i, original_i in enumerate(test_idx):
                prediction_results.append({
                    "fold": fold,
                    "subject": subject,
                    "trial_index": int(original_i),
                    "model": model_name,
                    "true_label": str(y_test[local_i]),
                    "predicted_label": str(predictions[local_i]),
                    "correct": bool(
                        y_test[local_i] == predictions[local_i]
                    ),
                })

            # Save best individual fold model for diagnostics.
            if balanced > best_fold_score[model_name]:
                best_fold_score[model_name] = balanced

                if model_name == "CSP_LDA":
                    saved_model = {
                        "model_name": model_name,
                        "csp": csp_objects["csp"],
                        "classifier": csp_objects["classifier"],
                        "held_out_subject": subject,
                        "class_names": CLASS_NAMES,
                        "sampling_rate": sampling_rate,
                        "broad_band_hz": [8.0, 30.0],
                        "channel_names": channel_names,
                    }
                else:
                    saved_model = {
                        "model_name": model_name,
                        **fbcsp_objects,
                        "held_out_subject": subject,
                        "class_names": CLASS_NAMES,
                        "sampling_rate": sampling_rate,
                        "frequency_bands": FREQUENCY_BANDS,
                        "channel_names": channel_names,
                    }

                joblib.dump(
                    saved_model,
                    MODEL_DIR / f"{model_name}_best_fold_model.joblib",
                )

    # Save all fold-level metrics and predictions.
    folds_df = pd.DataFrame(fold_results)
    predictions_df = pd.DataFrame(prediction_results)

    folds_df.to_csv(
        OUTPUT_DIR / "loso_fold_metrics.csv",
        index=False,
    )
    predictions_df.to_csv(
        OUTPUT_DIR / "loso_predictions.csv",
        index=False,
    )

    # Pooled metrics from every held-out subject.
    summary_rows = []

    for model_name in models:
        truth = np.asarray(pooled_true[model_name])
        pred = np.asarray(pooled_pred[model_name])

        fold_df = folds_df[folds_df["model"] == model_name]

        summary_rows.append({
            "model": model_name,
            "mean_fold_accuracy": fold_df["accuracy"].mean(),
            "std_fold_accuracy": fold_df["accuracy"].std(ddof=1),
            "mean_fold_balanced_accuracy": (
                fold_df["balanced_accuracy"].mean()
            ),
            "mean_fold_macro_f1": fold_df["macro_f1"].mean(),
            "pooled_loso_accuracy": accuracy_score(truth, pred),
            "pooled_loso_balanced_accuracy": (
                balanced_accuracy_score(truth, pred)
            ),
            "pooled_loso_macro_f1": f1_score(
                truth,
                pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0,
            ),
        })

        cm = confusion_matrix(
            truth,
            pred,
            labels=CLASS_NAMES,
        )

        pd.DataFrame(
            cm,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        ).to_csv(
            OUTPUT_DIR / f"{model_name}_confusion_matrix.csv"
        )

        report = classification_report(
            truth,
            pred,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        with open(
            OUTPUT_DIR / f"{model_name}_classification_report.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=2)

    summary = pd.DataFrame(summary_rows).sort_values(
        by=[
            "pooled_loso_balanced_accuracy",
            "pooled_loso_macro_f1",
        ],
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv",
        index=False,
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "output_directory": str(OUTPUT_DIR),
        "n_trials": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "subjects": sorted(np.unique(subjects).tolist()),
        "n_channels": int(X.shape[1]),
        "samples_per_trial": int(X.shape[2]),
        "sampling_rate": float(sampling_rate),
        "classes": CLASS_NAMES,
        "class_counts": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(y, return_counts=True)
            )
        },
        "evaluation": "Leave-One-Subject-Out cross-validation",
        "chance_level": 0.25,
        "frequency_bands": FREQUENCY_BANDS,
        "csp_components_per_band": CSP_COMPONENTS,
        "note": (
            "Only four subjects are available. Results are preliminary "
            "and have substantial uncertainty."
        ),
    }

    with open(
        OUTPUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    return summary


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NEUROSYMPHONY - DS004022 CSP EXPERIMENT")
    print("=" * 70)
    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)

    X, y, subjects, sampling_rate, channel_names = load_dataset()

    print("\nDATASET VALIDATION")
    print("-" * 70)
    print("EEG tensor shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Sampling rate:", sampling_rate, "Hz")
    print("Channels:", len(channel_names))
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Class counts:")

    for label, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {label}: {count}")

    print("\nStarting CSP and FBCSP evaluation...")

    summary = run_loso(
        X,
        y,
        subjects,
        sampling_rate,
        channel_names,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda value: f"{value:.4f}",
    ):
        print(summary.to_string(index=False))

    print("\nEXPERIMENT COMPLETE")
    print("Results saved to:", OUTPUT_DIR)
    print("Four-class chance level: 25%")
    print("Compare with Experiment 98 using the same LOSO protocol.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: CSP experiment failed.")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
