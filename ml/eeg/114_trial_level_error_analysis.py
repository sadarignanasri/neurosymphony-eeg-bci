
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
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
AUDIT = PROJECT / "experiments" / "ds004022_verified_epochs" / "trial_audit.csv"
OUT = PROJECT / "experiments" / "ds004022_experiment114_results"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]


def load_epochs():
    data = np.load(DATA, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    subjects = data["subjects"].astype(str)
    runs = data["runs"].astype(str)

    if X.shape[0] != len(y) or len(y) != len(subjects) or len(y) != len(runs):
        raise ValueError("Epochs, labels, subjects, and runs have different lengths.")

    if not np.isfinite(X).all():
        raise ValueError("Epoch data contains NaN or infinite values.")

    audit = pd.read_csv(AUDIT)
    if len(audit) != len(y):
        raise ValueError("Audit row count does not match the epoch count.")

    return X, y, subjects, runs, audit


def extract_features(X, fs=500):
    """Same channel-wise log-relative bandpower representation as Exp. 110."""
    features = []
    for epoch in X:
        freqs, psd = welch(
            epoch,
            fs=fs,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )
        total_mask = (freqs >= 1) & (freqs <= 40)
        total_power = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
        band_features = []

        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
            relative_power = power / np.maximum(total_power, 1e-12)
            band_features.append(np.log(np.maximum(relative_power, 1e-12)))

        features.append(np.concatenate(band_features))

    return np.asarray(features)


def main():
    X, y, subjects, runs, audit = load_epochs()
    print("Extracting features...")
    features = extract_features(X)
    print("Feature matrix:", features.shape)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            decision_function_shape="ovr",
        )),
    ])
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale"],
    }

    outer_cv = LeaveOneGroupOut()
    predictions = np.full(len(y), -1, dtype=int)
    decision_scores = [None] * len(y)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(features, y, groups=subjects), start=1
    ):
        train_subjects = subjects[train_idx]
        test_subject = subjects[test_idx[0]]

        inner_cv = GroupKFold(n_splits=3)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(
            features[train_idx],
            y[train_idx],
            groups=train_subjects,
        )

        pred = search.predict(features[test_idx])
        predictions[test_idx] = pred

        # Decision scores are uncalibrated margins, not probabilities.
        scores = search.decision_function(features[test_idx])
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        for local_i, global_i in enumerate(test_idx):
            decision_scores[global_i] = scores[local_i].tolist()

        fold_rows.append({
            "fold": fold,
            "held_out_subject": test_subject,
            "n_trials": len(test_idx),
            "accuracy": accuracy_score(y[test_idx], pred),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
            "best_params": json.dumps(search.best_params_),
            "inner_best_score": search.best_score_,
        })
        print(
            f"Fold {fold}/4 | {test_subject} | "
            f"accuracy={accuracy_score(y[test_idx], pred):.4f} | "
            f"macro-F1={f1_score(y[test_idx], pred, average='macro', zero_division=0):.4f}"
        )

    if (predictions < 0).any():
        raise RuntimeError("Some trials did not receive held-out predictions.")

    # Save trial-level predictions with the aligned audit metadata.
    trial_df = audit.copy()
    trial_df["subject_from_epochs"] = subjects
    trial_df["run_from_epochs"] = runs
    trial_df["true_class_id"] = y
    trial_df["true_class"] = [CLASS_NAMES[i] for i in y]
    trial_df["predicted_class_id"] = predictions
    trial_df["predicted_class"] = [CLASS_NAMES[i] for i in predictions]
    trial_df["correct"] = predictions == y
    trial_df["decision_scores"] = [
        json.dumps(s) for s in decision_scores
    ]
    trial_df.to_csv(OUT / "trial_level_predictions.csv", index=False)

    # Pooled metrics and confusion matrix.
    cm = confusion_matrix(y, predictions, labels=np.arange(4))
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    ).to_csv(OUT / "pooled_confusion_matrix.csv")

    report = classification_report(
        y,
        predictions,
        labels=np.arange(4),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(OUT / "pooled_classification_report.csv")

    # Per-subject, per-class, and per-run diagnostics.
    subject_rows = []
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        subject_rows.append({
            "subject": subject,
            "n_trials": int(mask.sum()),
            "accuracy": accuracy_score(y[mask], predictions[mask]),
            "balanced_accuracy": balanced_accuracy_score(y[mask], predictions[mask]),
            "macro_f1": f1_score(
                y[mask], predictions[mask], average="macro", zero_division=0
            ),
        })
    pd.DataFrame(subject_rows).to_csv(
        OUT / "subject_metrics.csv", index=False
    )

    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_mask = y == class_id
        pred_mask = predictions == class_id
        tp = int(np.sum(true_mask & pred_mask))
        class_rows.append({
            "class": class_name,
            "true_trials": int(true_mask.sum()),
            "predicted_trials": int(pred_mask.sum()),
            "correct": tp,
            "precision": tp / max(int(pred_mask.sum()), 1),
            "recall": tp / max(int(true_mask.sum()), 1),
            "f1": report[class_name]["f1-score"],
        })
    pd.DataFrame(class_rows).to_csv(
        OUT / "class_metrics.csv", index=False
    )

    run_rows = []
    for subject in sorted(np.unique(subjects)):
        for run in sorted(np.unique(runs[subjects == subject])):
            mask = (subjects == subject) & (runs == run)
            if mask.sum() == 0:
                continue
            run_rows.append({
                "subject": subject,
                "run": run,
                "n_trials": int(mask.sum()),
                "accuracy": accuracy_score(y[mask], predictions[mask]),
                "balanced_accuracy": balanced_accuracy_score(y[mask], predictions[mask]),
                "macro_f1": f1_score(
                    y[mask], predictions[mask], average="macro", zero_division=0
                ),
            })
    pd.DataFrame(run_rows).to_csv(OUT / "run_metrics.csv", index=False)

    # One confusion matrix per held-out subject.
    subject_cm_rows = []
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        subject_cm = confusion_matrix(y[mask], predictions[mask], labels=np.arange(4))
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                subject_cm_rows.append({
                    "subject": subject,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(subject_cm[i, j]),
                })
    pd.DataFrame(subject_cm_rows).to_csv(
        OUT / "subject_confusion_matrices_long.csv", index=False
    )

    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)

    summary = {
        "experiment": 114,
        "model": "BandPower_SVM",
        "evaluation": "Nested LOSO; inner GroupKFold grouped by training subject",
        "n_trials": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_features": int(features.shape[1]),
        "pooled_accuracy": float(accuracy_score(y, predictions)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "pooled_macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "warning": (
            "This reruns the pipeline to capture trial-level predictions. "
            "Small numerical differences from earlier experiments can occur "
            "if their feature extraction or evaluation code differed."
        ),
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 114 COMPLETE")
    print(json.dumps(summary, indent=2))
    print("\nClass metrics:")
    print(pd.DataFrame(class_rows).to_string(index=False))
    print("\nSubject metrics:")
    print(pd.DataFrame(subject_rows).to_string(index=False))
    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
