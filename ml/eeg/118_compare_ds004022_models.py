
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
OUT = PROJECT / "experiments" / "ds004022_experiment118_results"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]
FS = 500
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    data = np.load(DATA, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(int)
    subjects = data["subjects"].astype(str)
    runs = data["runs"].astype(str)

    if not (len(X) == len(y) == len(subjects) == len(runs)):
        raise ValueError("Epochs and metadata have mismatched lengths.")
    if not np.isfinite(X).all():
        raise ValueError("EEG data contains NaN or infinite values.")
    if set(np.unique(y)) != set(range(4)):
        raise ValueError(f"Unexpected class IDs: {np.unique(y)}")

    return X, y, subjects, runs


def extract_relative_power(X):
    """Reproduce Experiment 117's 90 relative log-band-power features."""
    rows = []

    for epoch in X:
        freqs, psd = welch(
            epoch,
            fs=FS,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )

        total_mask = (freqs >= 1) & (freqs <= 40)
        total = np.trapezoid(
            psd[:, total_mask], freqs[total_mask], axis=1
        )
        total = np.maximum(total, 1e-12)

        features = []
        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], freqs[mask], axis=1
            )
            relative = power / total
            features.append(np.log(np.maximum(relative, 1e-12)))

        rows.append(np.concatenate(features))

    return np.asarray(rows)


def model_configs():
    """All preprocessing is inside each training-fold pipeline."""
    return {
        "RBF_SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    class_weight="balanced",
                )),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale"],
            },
        ),
        "Linear_SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=42,
                )),
            ]),
            {
                "clf__C": [0.01, 0.1, 1, 10],
            },
        ),
        "Shrinkage_LDA": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                )),
            ]),
            {},
        ),
        "Extra_Trees": (
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            {
                "max_features": ["sqrt", 0.5],
                "min_samples_leaf": [1, 3],
            },
        ),
    }


def main():
    X, y, subjects, runs = load_data()

    print("Extracting Experiment 117 relative-power features...")
    features = extract_relative_power(X)
    print("Feature matrix:", features.shape)
    print("Subjects:", sorted(np.unique(subjects)))

    configs = model_configs()
    outer_cv = LeaveOneGroupOut()

    fold_rows = []
    prediction_rows = []
    outer_fold_summaries = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(features, y, groups=subjects), start=1
    ):
        held_out = subjects[test_idx[0]]
        train_groups = subjects[train_idx]

        print(f"\nFold {fold}/4 | held-out subject: {held_out}")

        for model_name, (estimator, param_grid) in configs.items():
            if param_grid:
                search = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=param_grid,
                    scoring="balanced_accuracy",
                    cv=GroupKFold(n_splits=3),
                    n_jobs=1,
                    refit=True,
                )
                search.fit(
                    features[train_idx],
                    y[train_idx],
                    groups=train_groups,
                )
                fitted_model = search.best_estimator_
                best_params = search.best_params_
                inner_score = float(search.best_score_)
            else:
                fitted_model = clone(estimator)
                fitted_model.fit(features[train_idx], y[train_idx])
                best_params = {}
                inner_score = None

            pred = fitted_model.predict(features[test_idx])

            acc = accuracy_score(y[test_idx], pred)
            bal_acc = balanced_accuracy_score(y[test_idx], pred)
            macro_f1 = f1_score(
                y[test_idx], pred,
                average="macro", zero_division=0
            )

            fold_rows.append({
                "fold": fold,
                "held_out_subject": held_out,
                "model": model_name,
                "n_test_trials": len(test_idx),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "inner_best_balanced_accuracy": inner_score,
                "best_params": json.dumps(best_params),
            })

            outer_fold_summaries.append({
                "fold": fold,
                "held_out_subject": held_out,
                "model": model_name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
            })

            for idx, pred_label in zip(test_idx, pred):
                prediction_rows.append({
                    "trial_index": int(idx),
                    "subject": subjects[idx],
                    "run": runs[idx],
                    "true_class_id": int(y[idx]),
                    "true_class": CLASS_NAMES[y[idx]],
                    "predicted_class_id": int(pred_label),
                    "predicted_class": CLASS_NAMES[pred_label],
                    "correct": bool(y[idx] == pred_label),
                    "model": model_name,
                    "held_out_subject": held_out,
                })

            print(
                f"  {model_name:14s} | accuracy={acc:.3f} | "
                f"balanced_acc={bal_acc:.3f} | macro_F1={macro_f1:.3f}"
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "model_fold_results.csv", index=False)

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(OUT / "trial_level_predictions.csv", index=False)

    summary = fold_df.groupby("model").agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
    ).reset_index()

    summary = summary.sort_values(
        "mean_balanced_accuracy", ascending=False
    )
    summary.to_csv(OUT / "model_comparison_summary.csv", index=False)

    # Per-subject metrics for each model.
    subject_rows = []
    for (model_name, subject), part in pred_df.groupby(
        ["model", "held_out_subject"], sort=True
    ):
        truth = part["true_class_id"].to_numpy()
        pred = part["predicted_class_id"].to_numpy()

        subject_rows.append({
            "model": model_name,
            "held_out_subject": subject,
            "trials": len(part),
            "accuracy": accuracy_score(truth, pred),
            "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "macro_f1": f1_score(
                truth, pred, average="macro", zero_division=0
            ),
        })

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(OUT / "model_subject_results.csv", index=False)

    # Pooled classification report and confusion matrix for each model.
    report_rows = []
    cm_rows = []

    for model_name in configs:
        part = pred_df[pred_df["model"] == model_name]
        truth = part["true_class_id"].to_numpy()
        pred = part["predicted_class_id"].to_numpy()

        report = classification_report(
            truth,
            pred,
            labels=np.arange(4),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        for label, values in report.items():
            if isinstance(values, dict):
                report_rows.append({
                    "model": model_name,
                    "class_or_average": label,
                    **values,
                })

        cm = confusion_matrix(truth, pred, labels=np.arange(4))
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                cm_rows.append({
                    "model": model_name,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                })

    pd.DataFrame(report_rows).to_csv(
        OUT / "model_classification_reports.csv", index=False
    )
    pd.DataFrame(cm_rows).to_csv(
        OUT / "model_confusion_matrices.csv", index=False
    )

    best_model = str(summary.iloc[0]["model"])
    best_row = summary.iloc[0].to_dict()

    final_report = {
        "experiment": 118,
        "epochs": int(len(y)),
        "subjects": int(len(np.unique(subjects))),
        "feature_count": int(features.shape[1]),
        "evaluation": (
            "Outer LOSO; hyperparameters tuned by inner GroupKFold "
            "grouped by subject. Shrinkage LDA uses fixed shrinkage='auto'."
        ),
        "chance_level": 0.25,
        "best_by_mean_balanced_accuracy": best_model,
        "best_model_summary": {
            key: float(value) if isinstance(value, (np.floating, float))
            else value
            for key, value in best_row.items()
        },
        "limitations": [
            "Only four subjects were available for cross-subject evaluation.",
            "Four outer folds produce uncertain performance estimates.",
            "Model selection across multiple candidates may be optimistic.",
            "No specific accuracy target is guaranteed."
        ],
    }

    with open(OUT / "experiment118_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\nEXPERIMENT 118 COMPLETE")
    print("\nModel comparison (sorted by mean balanced accuracy):")
    print(summary.to_string(index=False))
    print("\nPer-subject results:")
    print(subject_df.to_string(index=False))
    print("\nFinal report:")
    print(json.dumps(final_report, indent=2))
    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
