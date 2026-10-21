
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.base import clone
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
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
OUT = PROJECT / "experiments" / "ds004022_experiment119_results"
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
        raise ValueError("Epochs, labels, subjects, and runs do not align.")
    if not np.isfinite(X).all():
        raise ValueError("EEG data contains non-finite values.")
    if set(np.unique(y)) != set(range(4)):
        raise ValueError(f"Unexpected class IDs: {np.unique(y)}")

    return X, y, subjects, runs


def extract_relative_power(X):
    """Reproduce the 90 features used in Experiment 118."""
    all_features = []

    for epoch in X:
        freqs, psd = welch(
            epoch,
            fs=FS,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )

        total_mask = (freqs >= 1) & (freqs <= 40)
        total_power = np.trapezoid(
            psd[:, total_mask], freqs[total_mask], axis=1
        )
        total_power = np.maximum(total_power, 1e-12)

        band_features = []
        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], freqs[mask], axis=1
            )
            relative = power / total_power
            band_features.append(
                np.log(np.maximum(relative, 1e-12))
            )

        all_features.append(np.concatenate(band_features))

    return np.asarray(all_features)


def get_models():
    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced")),
    ])

    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    return {
        "RBF_SVM": (
            rbf_svm,
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale"],
            },
        ),
        "Extra_Trees": (
            extra_trees,
            {
                "max_features": ["sqrt", 0.5],
                "min_samples_leaf": [1, 3],
            },
        ),
    }


def main():
    X, y, subjects, runs = load_data()
    unique_subjects = np.unique(subjects)

    if len(unique_subjects) < 4:
        raise ValueError("Expected at least four subjects for this experiment.")

    print("Extracting the same 90 relative-power features...")
    features = extract_relative_power(X)
    print("Feature shape:", features.shape)
    print("Subjects:", unique_subjects.tolist())

    # Feature extraction is fixed and trial-local. Scaling, where used,
    # is fitted inside each training fold through the model Pipeline.
    models = get_models()
    outer_cv = LeaveOneGroupOut()

    prediction_rows = []
    fold_rows = []
    fold_count = 0

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(features, y, groups=subjects), start=1
    ):
        fold_count += 1
        held_out_subject = subjects[test_idx[0]]

        # Explicit leakage checks: the held-out subject must not occur
        # in the training data for this outer fold.
        train_subject_set = set(subjects[train_idx])
        test_subject_set = set(subjects[test_idx])

        if train_subject_set & test_subject_set:
            raise RuntimeError("Subject leakage detected in outer split.")
        if len(test_subject_set) != 1:
            raise RuntimeError("Expected exactly one held-out subject.")

        print(f"\nFold {fold}/{len(unique_subjects)}: {held_out_subject}")

        for model_name, (estimator, param_grid) in models.items():
            search = GridSearchCV(
                estimator=clone(estimator),
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=GroupKFold(n_splits=3),
                n_jobs=1,
                refit=True,
            )

            # Inner folds are grouped by subject; the held-out subject
            # is never passed to GridSearchCV.
            search.fit(
                features[train_idx],
                y[train_idx],
                groups=subjects[train_idx],
            )

            pred = search.predict(features[test_idx])

            fold_rows.append({
                "fold": fold,
                "held_out_subject": held_out_subject,
                "model": model_name,
                "n_train_trials": len(train_idx),
                "n_test_trials": len(test_idx),
                "train_subjects": ",".join(sorted(train_subject_set)),
                "test_subjects": ",".join(sorted(test_subject_set)),
                "inner_best_balanced_accuracy": float(search.best_score_),
                "outer_accuracy": float(accuracy_score(y[test_idx], pred)),
                "outer_balanced_accuracy": float(
                    balanced_accuracy_score(y[test_idx], pred)
                ),
                "outer_macro_f1": float(
                    f1_score(
                        y[test_idx], pred,
                        average="macro", zero_division=0
                    )
                ),
                "best_params": json.dumps(search.best_params_),
            })

            for idx, predicted in zip(test_idx, pred):
                prediction_rows.append({
                    "trial_index": int(idx),
                    "subject": subjects[idx],
                    "run": runs[idx],
                    "true_class_id": int(y[idx]),
                    "true_class": CLASS_NAMES[y[idx]],
                    "predicted_class_id": int(predicted),
                    "predicted_class": CLASS_NAMES[predicted],
                    "correct": bool(y[idx] == predicted),
                    "model": model_name,
                    "held_out_subject": held_out_subject,
                })

            print(
                f"  {model_name}: accuracy={accuracy_score(y[test_idx], pred):.3f}, "
                f"macro-F1={f1_score(y[test_idx], pred, average='macro', zero_division=0):.3f}"
            )

    if fold_count != len(unique_subjects):
        raise RuntimeError("Unexpected number of outer LOSO folds.")

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)

    predictions.to_csv(OUT / "held_out_predictions.csv", index=False)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)

    # Every model must have exactly one held-out prediction per trial.
    coverage = {}
    for model_name, part in predictions.groupby("model"):
        coverage[model_name] = {
            "prediction_rows": int(len(part)),
            "unique_trial_indices": int(part["trial_index"].nunique()),
            "missing_predictions": int(len(y) - part["trial_index"].nunique()),
        }
        if part["trial_index"].nunique() != len(y):
            raise RuntimeError(
                f"Incomplete held-out predictions for {model_name}."
            )

    summary_rows = []
    class_rows = []
    confusion_rows = []
    subject_rows = []

    for model_name in models:
        part = predictions[predictions["model"] == model_name]
        truth = part["true_class_id"].to_numpy()
        pred = part["predicted_class_id"].to_numpy()

        summary_rows.append({
            "model": model_name,
            "pooled_accuracy": accuracy_score(truth, pred),
            "pooled_balanced_accuracy": balanced_accuracy_score(truth, pred),
            "pooled_macro_f1": f1_score(
                truth, pred, average="macro", zero_division=0
            ),
            "mean_subject_accuracy": folds.loc[
                folds["model"] == model_name, "outer_accuracy"
            ].mean(),
            "mean_subject_macro_f1": folds.loc[
                folds["model"] == model_name, "outer_macro_f1"
            ].mean(),
        })

        report = classification_report(
            truth,
            pred,
            labels=np.arange(4),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        for class_name in CLASS_NAMES:
            class_rows.append({
                "model": model_name,
                "class": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

        cm = confusion_matrix(truth, pred, labels=np.arange(4))
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                confusion_rows.append({
                    "model": model_name,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                })

        for subject, subject_part in part.groupby("subject"):
            sy = subject_part["true_class_id"].to_numpy()
            sp = subject_part["predicted_class_id"].to_numpy()
            subject_rows.append({
                "model": model_name,
                "subject": subject,
                "accuracy": accuracy_score(sy, sp),
                "balanced_accuracy": balanced_accuracy_score(sy, sp),
                "macro_f1": f1_score(
                    sy, sp, average="macro", zero_division=0
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "final_model_comparison.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        OUT / "final_class_metrics.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        OUT / "final_confusion_matrices.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(
        OUT / "final_subject_metrics.csv", index=False
    )

    report = {
        "experiment": 119,
        "epochs": int(len(y)),
        "subjects": unique_subjects.tolist(),
        "classes": CLASS_NAMES,
        "features": int(features.shape[1]),
        "evaluation": (
            "Nested LOSO; inner GroupKFold grouped by training subject. "
            "Model scaling and hyperparameter fitting occur inside training folds."
        ),
        "outer_split_leakage_check": "passed",
        "prediction_coverage": coverage,
        "chance_level": 0.25,
        "models": summary_rows,
        "limitations": [
            "The same four subjects were used in earlier experiments to choose candidates.",
            "This is a final reproducibility check, not an independent confirmation set.",
            "Independent generalization claims require additional unseen subjects.",
            "Four outer folds give uncertain performance estimates."
        ],
    }

    with open(OUT / "experiment119_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nEXPERIMENT 119 COMPLETE")
    print("\nFinal model comparison:")
    print(summary_df.to_string(index=False))
    print("\nPer-subject metrics:")
    print(pd.DataFrame(subject_rows).to_string(index=False))
    print("\nClass metrics:")
    print(pd.DataFrame(class_rows).to_string(index=False))
    print("\nLeakage and prediction coverage:")
    print(json.dumps({
        "outer_split_leakage_check": "passed",
        "prediction_coverage": coverage,
    }, indent=2))
    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
