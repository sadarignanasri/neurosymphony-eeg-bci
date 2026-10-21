
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mne.decoding import CSP

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# EXPERIMENT 110: UNIFIED NESTED LOSO BENCHMARK
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
)
OUT_DIR = PROJECT / "experiments" / "ds004022_experiment110_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EXPECTED_FS = 500.0
N_CLASSES = 4
CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]

BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "wide_motor": (7.0, 30.0),
}


def bandpass_epochs(X, fs, low, high):
    sos = butter(
        4, [low, high], btype="bandpass", fs=fs, output="sos"
    )
    return sosfiltfilt(sos, X, axis=-1)


def bandpower_features(X, fs):
    """Per-channel log relative band power."""
    feature_blocks = []
    bands = [
        (1.0, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 40.0),
    ]

    for epoch in X:
        freqs, psd = welch(
            epoch, fs=fs, nperseg=min(1000, epoch.shape[-1]), axis=-1
        )
        total_mask = (freqs >= 1.0) & (freqs <= 40.0)
        total = np.trapezoid(
            psd[:, total_mask], x=freqs[total_mask], axis=1
        )
        total = np.maximum(total, 1e-30)

        trial_features = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], x=freqs[mask], axis=1
            )
            relative = power / total
            trial_features.extend(np.log(np.maximum(relative, 1e-12)))

        feature_blocks.append(trial_features)

    return np.asarray(feature_blocks, dtype=np.float64)


def build_model(name):
    if name == "CSP_LDA":
        return Pipeline([
            ("csp", CSP(
                n_components=4,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )),
            ("classifier", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto"
            )),
        ]), {
            "csp__n_components": [2, 4, 6],
        }

    if name == "CSP_RBF_SVM":
        return Pipeline([
            ("csp", CSP(
                n_components=4,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", class_weight="balanced")),
        ]), {
            "csp__n_components": [2, 4, 6],
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale"],
        }

    if name == "BandPower_LDA":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto"
            )),
        ]), {}

    if name == "BandPower_SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", class_weight="balanced"
            )),
        ]), {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale"],
        }

    raise ValueError(f"Unknown model: {name}")


def riemannian_features(X):
    """Estimate regularized covariance matrices and vectorize them.

    Uses upper-triangular covariance entries as a simple baseline,
    not a full tangent-space Riemannian classifier.
    """
    feature_rows = []
    for epoch in X:
        cov = LedoitWolf().fit(epoch.T).covariance_
        i, j = np.triu_indices(cov.shape[0])
        feature_rows.append(cov[i, j])
    return np.asarray(feature_rows)


def nested_loso(X, y, subjects, model_name, fs):
    predictions = np.full(len(y), -1, dtype=int)
    fold_rows = []
    subject_values = sorted(np.unique(subjects))

    for held_subject in subject_values:
        train_idx = np.where(subjects != held_subject)[0]
        test_idx = np.where(subjects == held_subject)[0]

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_groups = subjects[train_idx]

        if model_name in ("CSP_LDA", "CSP_RBF_SVM"):
            # Compare the predefined frequency bands. Selection occurs
            # only through grouped inner CV on the training subjects.
            candidates = []
            for band_name, (low, high) in BANDS.items():
                X_band_train = bandpass_epochs(X_train_raw, fs, low, high)
                X_band_test = bandpass_epochs(X_test_raw, fs, low, high)

                model, grid = build_model(model_name)
                inner_splits = GroupKFold(n_splits=3)

                search = GridSearchCV(
                    model,
                    param_grid=grid,
                    scoring="balanced_accuracy",
                    cv=inner_splits,
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_band_train, y_train, groups=train_groups)

                candidates.append((
                    search.best_score_,
                    band_name,
                    X_band_test,
                    search,
                ))

            _, selected_band, X_test_selected, best_search = max(
                candidates, key=lambda item: item[0]
            )
            best_model = best_search.best_estimator_
            pred = best_model.predict(X_test_selected)
            best_params = best_search.best_params_

        elif model_name in ("BandPower_LDA", "BandPower_SVM"):
            X_train = bandpower_features(X_train_raw, fs)
            X_test = bandpower_features(X_test_raw, fs)
            model, grid = build_model(model_name)

            if grid:
                search = GridSearchCV(
                    model,
                    param_grid=grid,
                    scoring="balanced_accuracy",
                    cv=GroupKFold(n_splits=3),
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train, groups=train_groups)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                best_model = clone(model)
                best_model.fit(X_train, y_train)
                best_params = {}

            pred = best_model.predict(X_test)
            selected_band = "1-40 Hz bandpower"

        else:
            # Covariance-vector baseline; tune only within training subjects.
            X_train = riemannian_features(X_train_raw)
            X_test = riemannian_features(X_test_raw)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(class_weight="balanced")),
            ])
            grid = {"classifier__C": [0.1, 1, 10]}

            search = GridSearchCV(
                model,
                param_grid=grid,
                scoring="balanced_accuracy",
                cv=GroupKFold(n_splits=3),
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=train_groups)
            pred = search.predict(X_test)
            best_params = search.best_params_
            selected_band = "regularized covariance vector"

        predictions[test_idx] = pred

        row = {
            "model": model_name,
            "held_out_subject": held_subject,
            "n_test_trials": len(test_idx),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred, labels=np.arange(N_CLASSES),
                average="macro", zero_division=0
            ),
            "selected_representation": selected_band,
            "best_params": json.dumps(best_params, default=str),
        }
        fold_rows.append(row)

        print(
            f"{model_name:18s} | {held_subject} | "
            f"accuracy={row['accuracy']:.4f} | "
            f"balanced_acc={row['balanced_accuracy']:.4f} | "
            f"macro_F1={row['macro_f1']:.4f}"
        )

    if np.any(predictions < 0):
        raise RuntimeError("Some trials did not receive an outer-fold prediction.")

    return predictions, pd.DataFrame(fold_rows)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing verified epochs: {DATA_PATH}")

    with np.load(DATA_PATH, allow_pickle=True) as d:
        X = np.asarray(d["X"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=int)
        subjects = np.asarray(d["subjects"]).astype(str)
        fs = float(np.asarray(d["sampling_rate"]).squeeze())

    if X.ndim != 3:
        raise ValueError(f"Expected 3D EEG data, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite EEG data found.")
    if fs != EXPECTED_FS:
        raise ValueError(f"Expected {EXPECTED_FS} Hz, got {fs}")
    if set(np.unique(y)) != set(range(N_CLASSES)):
        raise ValueError(f"Unexpected class labels: {np.unique(y)}")
    if len(np.unique(subjects)) < 4:
        raise ValueError("Expected at least four subjects for this benchmark.")

    print("=" * 75)
    print("EXPERIMENT 110 — UNIFIED NESTED LOSO BENCHMARK")
    print("=" * 75)
    print("X:", X.shape, "| fs:", fs)
    print("Subjects:", sorted(np.unique(subjects)))
    print("Class counts:", {
        CLASS_NAMES[c]: int(np.sum(y == c))
        for c in range(N_CLASSES)
    })
    print("Chance accuracy: 25%")
    print("No original files will be changed.")

    all_folds = []
    all_summary = []
    all_confusion = []

    models = [
        "CSP_LDA",
        "CSP_RBF_SVM",
        "BandPower_LDA",
        "BandPower_SVM",
        "Covariance_SVM",
    ]

    for model_name in models:
        print("\n" + "-" * 75)
        print("Training:", model_name)

        pred, fold_df = nested_loso(
            X, y, subjects, model_name, fs
        )
        fold_df.to_csv(
            OUT_DIR / f"{model_name}_fold_results.csv", index=False
        )
        all_folds.append(fold_df)

        pooled = {
            "model": model_name,
            "pooled_accuracy": accuracy_score(y, pred),
            "pooled_balanced_accuracy": balanced_accuracy_score(y, pred),
            "pooled_macro_f1": f1_score(
                y, pred, labels=np.arange(N_CLASSES),
                average="macro", zero_division=0
            ),
            "mean_subject_accuracy": fold_df["accuracy"].mean(),
            "mean_subject_balanced_accuracy": (
                fold_df["balanced_accuracy"].mean()
            ),
            "mean_subject_macro_f1": fold_df["macro_f1"].mean(),
        }
        all_summary.append(pooled)

        cm = confusion_matrix(y, pred, labels=np.arange(N_CLASSES))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        )
        cm_df.to_csv(OUT_DIR / f"{model_name}_confusion_matrix.csv")

        report = classification_report(
            y, pred, labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES, zero_division=0, output_dict=True
        )
        with open(
            OUT_DIR / f"{model_name}_classification_report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(report, f, indent=2)

        print("Pooled:", pooled)
        print("Confusion matrix:\n", cm_df)

    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.DataFrame(all_summary).sort_values(
        "mean_subject_balanced_accuracy", ascending=False
    )

    folds_df.to_csv(OUT_DIR / "all_model_fold_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    report = {
        "experiment": 110,
        "dataset": "ds004022",
        "n_trials": int(len(y)),
        "n_channels": int(X.shape[1]),
        "n_subjects": int(len(np.unique(subjects))),
        "sampling_rate_hz": fs,
        "chance_accuracy": 0.25,
        "validation": "outer leave-one-subject-out; inner 3-fold GroupKFold",
        "models": models,
        "results_directory": str(OUT_DIR),
        "warning": (
            "Only four subjects are available. Estimates are preliminary. "
            "Band-power and covariance-vector models are baselines; the "
            "covariance-vector model is not a full tangent-space method. "
            "The three inner folds each group subjects, leaving only three "
            "training subjects for inner tuning, so estimates are noisy."
        ),
    }
    with open(OUT_DIR / "experiment110_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75)
    print("EXPERIMENT 110 COMPLETE")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("\nResults saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
