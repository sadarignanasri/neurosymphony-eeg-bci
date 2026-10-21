
"""
NeuroSymphony - Dataset 2 (ds004022)
EEG-only model training and subject-aware evaluation.

Input:
    experiments/ds004022_verified_epochs/eeg_epochs.npz

Output:
    experiments/ds004022_model_results/

Evaluation:
    Leave-One-Subject-Out (LOSO) cross-validation.

Models:
    1. Logistic Regression
    2. RBF SVM
    3. Extra Trees
    4. HistGradientBoosting
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# 1. PATHS AND CONFIGURATION
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
    / "ds004022_model_results"
)

FOLD_DIR = OUTPUT_DIR / "fold_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SELECTED_FEATURES = 80

MODEL_NAMES = [
    "LogisticRegression",
    "SVM_RBF",
    "ExtraTrees",
    "HistGradientBoosting",
]


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset(npz_path):
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{npz_path}\n"
            "Run 97_extract_ds004022_epochs.py first."
        )

    with np.load(npz_path, allow_pickle=True) as data:
        required = ["X", "y", "subjects"]

        missing = [
            key for key in required
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Missing arrays in NPZ file: {missing}"
            )

        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).astype(str)
        subjects = np.asarray(data["subjects"]).astype(str)

        if "channel_names" in data.files:
            channel_names = np.asarray(
                data["channel_names"]
            ).astype(str)
        else:
            channel_names = np.array([
                f"EEG_{i + 1:02d}"
                for i in range(X.shape[1])
            ])

        if "sampling_rate" in data.files:
            sampling_rate = float(
                np.asarray(data["sampling_rate"]).item()
            )
        else:
            sampling_rate = 500.0

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data, received {X.shape}"
        )

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            "Trial, label, and subject counts do not match."
        )

    if len(X) == 0:
        raise ValueError("Dataset contains no trials.")

    if not np.isfinite(X).all():
        raise ValueError(
            "EEG data contains NaN or infinite values."
        )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    if len(np.unique(subjects)) < 2:
        raise ValueError(
            "LOSO requires at least two subjects."
        )

    return (
        X,
        y,
        subjects,
        channel_names,
        sampling_rate,
    )


# ============================================================
# 3. EEG FEATURE EXTRACTION
# ============================================================

def extract_features(X, sampling_rate):
    """
    Extract 16 features per EEG channel:

    1. Mean
    2. Standard deviation
    3. Variance
    4. RMS
    5. Line length
    6. Zero-crossing rate
    7-11. Relative delta, theta, alpha, beta, gamma power
    12-16. Log delta, theta, alpha, beta, gamma power

    Frequency bands:
        Delta: 1-4 Hz
        Theta: 4-8 Hz
        Alpha: 8-13 Hz
        Beta: 13-30 Hz
        Gamma: 30-40 Hz
    """

    n_trials, n_channels, n_samples = X.shape

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if 40.0 >= sampling_rate / 2.0:
        raise ValueError(
            "Sampling rate is too low for the 40 Hz band."
        )

    bands = [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 40.0),
    ]

    feature_names = []

    for ch in range(n_channels):
        prefix = f"ch{ch + 1:02d}"

        feature_names.extend([
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_variance",
            f"{prefix}_rms",
            f"{prefix}_line_length",
            f"{prefix}_zero_crossing_rate",
        ])

        feature_names.extend([
            f"{prefix}_{name}_relative_power"
            for name, _, _ in bands
        ])

        feature_names.extend([
            f"{prefix}_{name}_log_power"
            for name, _, _ in bands
        ])

    features = np.zeros(
        (n_trials, n_channels * 16),
        dtype=np.float64,
    )

    frequencies = np.fft.rfftfreq(
        n_samples,
        d=1.0 / sampling_rate,
    )

    window = np.hanning(n_samples)
    window_energy = max(
        float(np.sum(window ** 2)),
        np.finfo(float).eps,
    )

    for trial_idx in range(n_trials):
        epoch = X[trial_idx]

        means = np.mean(epoch, axis=1)
        centered = epoch - means[:, None]

        stds = np.std(epoch, axis=1)
        variances = np.var(epoch, axis=1)

        rms = np.sqrt(
            np.mean(epoch ** 2, axis=1)
        )

        line_length = np.mean(
            np.abs(np.diff(epoch, axis=1)),
            axis=1,
        )

        zero_crossing_rate = np.mean(
            (
                centered[:, 1:]
                * centered[:, :-1]
            ) < 0,
            axis=1,
        )

        tapered = centered * window[None, :]

        spectrum = np.fft.rfft(
            tapered,
            axis=1,
        )

        power = (
            np.abs(spectrum) ** 2
        ) / window_energy

        band_powers = []

        for _, low, high in bands:
            mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

            if np.any(mask):
                band_power = np.mean(
                    power[:, mask],
                    axis=1,
                )
            else:
                band_power = np.zeros(
                    n_channels,
                    dtype=np.float64,
                )

            band_powers.append(
                np.maximum(band_power, 1e-12)
            )

        total_power = np.sum(
            np.stack(band_powers, axis=1),
            axis=1,
        )

        total_power = np.maximum(
            total_power,
            1e-12,
        )

        relative_powers = [
            bp / total_power
            for bp in band_powers
        ]

        log_powers = [
            np.log10(bp)
            for bp in band_powers
        ]

        trial_features = []

        for ch in range(n_channels):
            trial_features.extend([
                means[ch],
                stds[ch],
                variances[ch],
                rms[ch],
                line_length[ch],
                zero_crossing_rate[ch],
            ])

            trial_features.extend([
                rp[ch]
                for rp in relative_powers
            ])

            trial_features.extend([
                lp[ch]
                for lp in log_powers
            ])

        features[trial_idx] = np.asarray(
            trial_features,
            dtype=np.float64,
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature extraction produced invalid values."
        )

    return features, feature_names


# ============================================================
# 4. DEFINE CLASSIFICATION MODELS
# ============================================================

def build_models(n_features):
    k = min(
        N_SELECTED_FEATURES,
        n_features,
    )

    return {
        "LogisticRegression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=k,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=1.0,
                ),
            ),
        ]),

        "SVM_RBF": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=k,
                ),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=k,
                ),
            ),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        "HistGradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=k,
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


# ============================================================
# 5. LEAVE-ONE-SUBJECT-OUT EVALUATION
# ============================================================

def run_loso(
    X_features,
    y,
    subjects,
    feature_names,
    sampling_rate,
):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()

    logo = LeaveOneGroupOut()

    splits = list(
        logo.split(
            X_features,
            y_encoded,
            groups=subjects,
        )
    )

    unique_subjects = sorted(
        np.unique(subjects).tolist()
    )

    if len(splits) != len(unique_subjects):
        raise RuntimeError(
            "Unexpected number of LOSO folds."
        )

    fold_rows = []
    prediction_rows = []

    pooled_truth = {
        name: []
        for name in MODEL_NAMES
    }

    pooled_predictions = {
        name: []
        for name in MODEL_NAMES
    }

    print("\n" + "=" * 70)
    print("LEAVE-ONE-SUBJECT-OUT EVALUATION")
    print("=" * 70)

    print("Subjects:", ", ".join(unique_subjects))
    print("Classes:", ", ".join(class_names))
    print("Number of folds:", len(splits))

    best_fold_score = {
        name: -1.0
        for name in MODEL_NAMES
    }

    for fold_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        held_out_subjects = np.unique(
            subjects[test_idx]
        )

        if len(held_out_subjects) != 1:
            raise RuntimeError(
                "Each test fold must contain one subject."
            )

        held_out_subject = str(
            held_out_subjects[0]
        )

        X_train = X_features[train_idx]
        X_test = X_features[test_idx]

        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        if set(np.unique(y_train)) != set(
            range(len(class_names))
        ):
            raise ValueError(
                f"Training fold {fold_number} "
                "does not contain every class."
            )

        print(
            f"\nFold {fold_number}/{len(splits)}"
            f" | Test subject: {held_out_subject}"
            f" | Train trials: {len(train_idx)}"
            f" | Test trials: {len(test_idx)}"
        )

        models = build_models(
            X_features.shape[1]
        )

        for model_name, model in models.items():
            print(
                f"  Training {model_name}...",
                flush=True,
            )

            model.fit(
                X_train,
                y_train,
            )

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            balanced_acc = balanced_accuracy_score(
                y_test,
                y_pred,
            )

            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            fold_rows.append({
                "fold": fold_number,
                "held_out_subject": held_out_subject,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "macro_f1": macro_f1,
            })

            pooled_truth[model_name].extend(
                y_test.tolist()
            )

            pooled_predictions[model_name].extend(
                y_pred.tolist()
            )

            for local_idx, original_idx in enumerate(
                test_idx
            ):
                prediction_rows.append({
                    "fold": fold_number,
                    "subject": held_out_subject,
                    "trial_index": int(original_idx),
                    "model": model_name,
                    "true_label": class_names[
                        int(y_test[local_idx])
                    ],
                    "predicted_label": class_names[
                        int(y_pred[local_idx])
                    ],
                    "correct": bool(
                        y_test[local_idx]
                        == y_pred[local_idx]
                    ),
                })

            print(
                f"    Accuracy: {accuracy:.4f}"
                f" | Balanced accuracy: {balanced_acc:.4f}"
                f" | Macro F1: {macro_f1:.4f}"
            )

            # Save the strongest individual fold model
            # for diagnostic use only.
            if balanced_acc > best_fold_score[model_name]:
                best_fold_score[model_name] = balanced_acc

                joblib.dump(
                    {
                        "model": model,
                        "model_name": model_name,
                        "held_out_subject": held_out_subject,
                        "class_names": class_names,
                        "feature_names": feature_names,
                        "sampling_rate": sampling_rate,
                        "note": (
                            "Best individual LOSO fold model. "
                            "Not the final all-subject model."
                        ),
                    },
                    FOLD_DIR
                    / f"{model_name}_best_fold_model.joblib",
                )

    folds_df = pd.DataFrame(fold_rows)

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    folds_df.to_csv(
        OUTPUT_DIR / "loso_fold_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "loso_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. SUMMARIZE ALL HELD-OUT PREDICTIONS
    # --------------------------------------------------------

    summary_rows = []

    for model_name in MODEL_NAMES:
        model_folds = folds_df[
            folds_df["model"] == model_name
        ]

        truth = np.asarray(
            pooled_truth[model_name],
            dtype=int,
        )

        predictions = np.asarray(
            pooled_predictions[model_name],
            dtype=int,
        )

        pooled_accuracy = accuracy_score(
            truth,
            predictions,
        )

        pooled_balanced_accuracy = (
            balanced_accuracy_score(
                truth,
                predictions,
            )
        )

        pooled_macro_f1 = f1_score(
            truth,
            predictions,
            average="macro",
            zero_division=0,
        )

        summary_rows.append({
            "model": model_name,
            "mean_fold_accuracy": (
                model_folds["accuracy"].mean()
            ),
            "std_fold_accuracy": (
                model_folds["accuracy"].std(ddof=1)
            ),
            "mean_fold_balanced_accuracy": (
                model_folds["balanced_accuracy"].mean()
            ),
            "std_fold_balanced_accuracy": (
                model_folds["balanced_accuracy"].std(ddof=1)
            ),
            "mean_fold_macro_f1": (
                model_folds["macro_f1"].mean()
            ),
            "std_fold_macro_f1": (
                model_folds["macro_f1"].std(ddof=1)
            ),
            "pooled_loso_accuracy": pooled_accuracy,
            "pooled_loso_balanced_accuracy": (
                pooled_balanced_accuracy
            ),
            "pooled_loso_macro_f1": pooled_macro_f1,
        })

        matrix = confusion_matrix(
            truth,
            predictions,
            labels=np.arange(len(class_names)),
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                f"true_{name}"
                for name in class_names
            ],
            columns=[
                f"pred_{name}"
                for name in class_names
            ],
        )

        matrix_df.to_csv(
            OUTPUT_DIR
            / f"{model_name}_confusion_matrix.csv"
        )

        report = classification_report(
            truth,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        with open(
            OUTPUT_DIR
            / f"{model_name}_classification_report.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        by=[
            "pooled_loso_balanced_accuracy",
            "pooled_loso_macro_f1",
        ],
        ascending=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv",
        index=False,
    )

    best_model_name = str(
        summary_df.iloc[0]["model"]
    )

    best_metrics = summary_df.iloc[0].to_dict()

    # --------------------------------------------------------
    # 7. TRAIN SELECTED MODEL ON ALL AVAILABLE SUBJECTS
    # --------------------------------------------------------
    # This is for future inference, not unbiased evaluation.

    final_models = build_models(
        X_features.shape[1]
    )

    final_model = final_models[
        best_model_name
    ]

    final_model.fit(
        X_features,
        y_encoded,
    )

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_model_name,
            "class_names": class_names,
            "feature_names": feature_names,
            "sampling_rate": sampling_rate,
            "epoch_seconds": 5.0,
            "n_channels": int(
                X_features.shape[1] // 16
            ),
            "training_subjects": unique_subjects,
            "evaluation": (
                "Leave-One-Subject-Out cross-validation"
            ),
            "warning": (
                "Trained on all subjects. Do not report "
                "training accuracy as test accuracy."
            ),
        },
        OUTPUT_DIR
        / "best_model_trained_on_all_subjects.joblib",
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "output_directory": str(OUTPUT_DIR),
        "n_trials": int(len(y)),
        "n_subjects": int(len(unique_subjects)),
        "subjects": unique_subjects,
        "n_channels": int(
            X_features.shape[1] // 16
        ),
        "n_features": int(
            X_features.shape[1]
        ),
        "classes": class_names,
        "class_counts": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(y, return_counts=True)
            )
        },
        "evaluation": (
            "Leave-One-Subject-Out cross-validation"
        ),
        "four_class_chance_level": 0.25,
        "best_model_by_pooled_loso_balanced_accuracy": (
            best_model_name
        ),
        "best_model_metrics": {
            key: float(value)
            for key, value in best_metrics.items()
            if key != "model"
        },
        "caution": (
            "Only four subjects are available. "
            "Cross-subject estimates have substantial uncertainty."
        ),
    }

    with open(
        OUTPUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return summary_df, best_model_name


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NEUROSYMPHONY - DS004022 EEG MODEL TRAINING")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input file:   {INPUT_FILE}")
    print(f"Output dir:   {OUTPUT_DIR}")

    (
        X,
        y,
        subjects,
        channel_names,
        sampling_rate,
    ) = load_dataset(INPUT_FILE)

    print("\nDATASET VALIDATION")
    print("-" * 70)

    print("EEG tensor shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Unique subjects:", len(np.unique(subjects)))
    print(
        "Subjects:",
        ", ".join(sorted(np.unique(subjects))),
    )
    print("Sampling rate:", sampling_rate, "Hz")
    print("Channels listed:", len(channel_names))
    print(
        "Classes:",
        ", ".join(sorted(np.unique(y))),
    )

    print("Class counts:")

    for label, count in zip(
        *np.unique(y, return_counts=True)
    ):
        print(f"  {label}: {count}")

    print("\nEXTRACTING EEG FEATURES")
    print("-" * 70)

    X_features, feature_names = extract_features(
        X,
        sampling_rate,
    )

    print("Feature matrix shape:", X_features.shape)
    print("Features per channel: 16")
    print("Feature extraction completed.")

    pd.DataFrame({
        "feature_index": np.arange(
            len(feature_names)
        ),
        "feature_name": feature_names,
    }).to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False,
    )

    summary_df, best_model = run_loso(
        X_features=X_features,
        y=y,
        subjects=subjects,
        feature_names=feature_names,
        sampling_rate=sampling_rate,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format",
        lambda value: f"{value:.4f}",
    ):
        print(
            summary_df.to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best model by pooled LOSO balanced accuracy:",
        best_model,
    )

    print("Results saved to:", OUTPUT_DIR)

    print("\nGenerated files:")
    print("  model_comparison_summary.csv")
    print("  loso_fold_metrics.csv")
    print("  loso_predictions.csv")
    print("  *_confusion_matrix.csv")
    print("  *_classification_report.json")
    print("  feature_names.csv")
    print("  best_model_trained_on_all_subjects.joblib")
    print("  run_metadata.json")

    print(
        "\nFour-class chance level: 25%. "
        "Report the evaluation method with every accuracy result."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Training pipeline failed.")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
