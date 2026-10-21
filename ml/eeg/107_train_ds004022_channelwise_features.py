
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# EXPERIMENT 107
# Channel-wise log band-power + nested subject-wise validation
# Dataset: ds004022
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
)

OUT_DIR = ROOT / "experiments" / "ds004022_experiment107_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
EPSILON = 1e-30
RANDOM_STATE = 42

BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}

CLASS_NAMES = {
    0: "reaching",
    1: "grasping",
    2: "lifting",
    3: "twisting",
}


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1).astype(int)
        subjects = np.asarray(data["subjects"]).reshape(-1).astype(str)

        if "channel_names" in data.files:
            channel_names = [
                str(v) for v in data["channel_names"].tolist()
            ]
        else:
            channel_names = [
                f"channel_{i:02d}" for i in range(X.shape[1])
            ]

    if X.ndim != 3:
        raise ValueError(f"Expected (trials, channels, samples), got {X.shape}")

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError("Epoch, label, and subject counts do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite EEG values detected.")

    if set(np.unique(y)) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected class IDs: {np.unique(y)}")

    if len(channel_names) != X.shape[1]:
        raise ValueError("Channel-name count does not match EEG channels.")

    return X, y, subjects, channel_names


def extract_channelwise_log_bandpower(X, channel_names):
    """
    Produce one log-power feature per channel and frequency band.

    Each epoch is processed independently. No labels or information
    from other trials are used to calculate its features.
    """
    feature_names = [
        f"{channel}_{band_name}"
        for channel in channel_names
        for band_name in BANDS
    ]

    features = np.empty(
        (len(X), len(feature_names)),
        dtype=np.float64,
    )

    for trial_idx, epoch in enumerate(X):
        freqs, psd = welch(
            epoch,
            fs=FS,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )

        values = []
        for channel_idx in range(epoch.shape[0]):
            for low, high in BANDS.values():
                mask = (freqs >= low) & (freqs < high)

                if mask.sum() < 2:
                    raise ValueError(
                        f"Insufficient PSD bins for {low}-{high} Hz."
                    )

                power = np.trapezoid(
                    psd[channel_idx, mask],
                    x=freqs[mask],
                )

                # Log transform reduces the effect of extreme power values.
                values.append(np.log10(max(float(power), EPSILON)))

        features[trial_idx] = values

        if (trial_idx + 1) % 100 == 0:
            print(f"Extracted features for {trial_idx + 1}/{len(X)} trials")

    return features, feature_names


def candidate_models():
    return [
        (
            "shrinkage_lda",
            Pipeline([
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",
                    ),
                ),
            ]),
            [{}],
        ),
        (
            "svm",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC()),
            ]),
            [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1.0, 10.0],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1.0, 10.0],
                    "model__gamma": ["scale"],
                },
            ],
        ),
    ]


def main():
    X, y, subjects, channel_names = load_data()

    print("\nEXPERIMENT 107 — CHANNEL-WISE EEG FEATURES")
    print("=" * 60)
    print("Epoch tensor:", X.shape)
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Class counts:", {
        int(c): int((y == c).sum()) for c in np.unique(y)
    })

    features, feature_names = extract_channelwise_log_bandpower(
        X, channel_names
    )

    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values were generated.")

    pd.DataFrame({
        "feature": feature_names,
    }).to_csv(OUT_DIR / "feature_names.csv", index=False)

    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df.insert(0, "subject", subjects)
    feature_df.insert(0, "label", y)
    feature_df.insert(0, "trial_index", np.arange(len(y)))

    feature_df.to_csv(
        OUT_DIR / "channelwise_log_bandpower_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Nested LOSO: choose model and hyperparameters using only
    # the training subjects, then evaluate on the held-out one.
    # --------------------------------------------------------
    unique_subjects = np.unique(subjects)

    if len(unique_subjects) < 3:
        raise ValueError("At least three subjects are required.")

    outer_rows = []
    prediction_frames = []
    selected_configs = []

    for fold_idx, held_out in enumerate(unique_subjects, start=1):
        print("\n" + "=" * 60)
        print(f"OUTER FOLD {fold_idx}/{len(unique_subjects)}")
        print("Held-out subject:", held_out)

        train_mask = subjects != held_out
        test_mask = subjects == held_out

        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        groups_train = subjects[train_mask]

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        candidate_results = []

        for model_name, estimator, grid in candidate_models():
            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                scoring="balanced_accuracy",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                verbose=0,
                error_score="raise",
            )

            search.fit(
                X_train,
                y_train,
                groups=groups_train,
            )

            candidate_results.append({
                "name": model_name,
                "search": search,
                "inner_score": float(search.best_score_),
                "best_params": search.best_params_,
            })

            print(
                f"{model_name}: inner balanced accuracy "
                f"{search.best_score_:.4f}"
            )

        # Model selection is based exclusively on inner CV.
        winner = max(
            candidate_results,
            key=lambda item: item["inner_score"],
        )

        best_model = winner["search"].best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        print("Selected model:", winner["name"])
        print("Selected parameters:", winner["best_params"])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        outer_rows.append({
            "held_out_subject": held_out,
            "n_test_trials": int(len(y_test)),
            "selected_model": winner["name"],
            "inner_balanced_accuracy": winner["inner_score"],
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "macro_f1": float(macro_f1),
            "best_params": json.dumps(winner["best_params"]),
        })

        selected_configs.append({
            "held_out_subject": held_out,
            "selected_model": winner["name"],
            "inner_balanced_accuracy": winner["inner_score"],
            "best_params": winner["best_params"],
        })

        prediction_frames.append(pd.DataFrame({
            "trial_index": np.flatnonzero(test_mask),
            "subject": held_out,
            "y_true": y_test,
            "y_pred": y_pred,
        }))

        # Save progress after each outer fold.
        pd.DataFrame(outer_rows).to_csv(
            OUT_DIR / "loso_fold_results_partial.csv",
            index=False,
        )

        pd.concat(prediction_frames, ignore_index=True).to_csv(
            OUT_DIR / "loso_predictions_partial.csv",
            index=False,
        )

        del candidate_results, best_model
        gc.collect()

    # --------------------------------------------------------
    # Final evaluation outputs
    # --------------------------------------------------------
    fold_df = pd.DataFrame(outer_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)

    fold_df.to_csv(OUT_DIR / "loso_fold_results.csv", index=False)
    pred_df.to_csv(OUT_DIR / "loso_predictions.csv", index=False)

    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    labels = [0, 1, 2, 3]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"true_{CLASS_NAMES[c]}" for c in labels],
        columns=[f"pred_{CLASS_NAMES[c]}" for c in labels],
    ).to_csv(OUT_DIR / "confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES[c] for c in labels],
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
        "experiment": 107,
        "dataset": "ds004022",
        "feature_method": "Per-channel log band power",
        "n_trials": int(len(y)),
        "n_subjects": int(len(unique_subjects)),
        "n_channels": int(X.shape[1]),
        "n_features": int(features.shape[1]),
        "frequency_bands_hz": {
            name: list(bounds) for name, bounds in BANDS.items()
        },
        "chance_level": 0.25,
        "pooled_loso_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_loso_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "pooled_loso_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mean_subject_accuracy": float(fold_df["accuracy"].mean()),
        "mean_subject_balanced_accuracy": float(
            fold_df["balanced_accuracy"].mean()
        ),
        "mean_subject_macro_f1": float(fold_df["macro_f1"].mean()),
        "selected_models_by_subject": selected_configs,
        "validation": (
            "Nested LOSO. Each held-out subject is evaluated after "
            "model and hyperparameter selection on the other subjects."
        ),
        "caution": (
            "Only four subjects are available. Performance estimates are "
            "uncertain and do not guarantee generalization to new people."
        ),
    }

    with open(
        OUT_DIR / "experiment107_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 107 COMPLETE")
    print("=" * 60)
    print(f"Pooled LOSO accuracy:          {summary['pooled_loso_accuracy']:.4f}")
    print(f"Pooled balanced accuracy:      {summary['pooled_loso_balanced_accuracy']:.4f}")
    print(f"Pooled macro F1:                {summary['pooled_loso_macro_f1']:.4f}")
    print(f"Mean subject accuracy:          {summary['mean_subject_accuracy']:.4f}")
    print(f"Mean subject balanced accuracy: {summary['mean_subject_balanced_accuracy']:.4f}")
    print(f"Mean subject macro F1:           {summary['mean_subject_macro_f1']:.4f}")
    print("\nResults saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
