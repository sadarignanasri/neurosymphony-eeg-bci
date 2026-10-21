
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
AUDIT_PATH = PROJECT / "experiments" / "ds004022_verified_epochs" / "trial_audit.csv"
SOURCE_DIR = PROJECT / "experiments" / "ds004022_experiment110_results"
OUT_DIR = PROJECT / "experiments" / "ds004022_experiment113_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["reaching", "grasping", "lifting", "twisting"]


def main():
    # Reuse the existing BandPower_SVM confusion matrix.
    cm_path = SOURCE_DIR / "BandPower_SVM_confusion_matrix.csv"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing prediction confusion matrix: {cm_path}")

    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.to_numpy(dtype=int)

    if cm.shape != (4, 4):
        raise ValueError(f"Expected 4x4 confusion matrix, got {cm.shape}")

    # Reconstruct predictions from the confusion matrix for class-level
    # diagnostic statistics. This cannot recover subject-wise predictions.
    y_true = []
    y_pred = []
    for true_idx in range(4):
        for pred_idx in range(4):
            count = int(cm[true_idx, pred_idx])
            y_true.extend([true_idx] * count)
            y_pred.extend([pred_idx] * count)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    class_report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(4),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(class_report).T.to_csv(
        OUT_DIR / "class_performance.csv"
    )

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    recalls = np.divide(
        np.diag(cm), row_sums,
        out=np.zeros(4, dtype=float), where=row_sums != 0
    )
    precisions = np.divide(
        np.diag(cm), col_sums,
        out=np.zeros(4, dtype=float), where=col_sums != 0
    )

    class_rows = []
    for i, name in enumerate(CLASSES):
        class_rows.append({
            "class": name,
            "true_trials": int(row_sums[i]),
            "predicted_trials": int(col_sums[i]),
            "correct_predictions": int(cm[i, i]),
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(class_report[name]["f1-score"]),
        })

    pd.DataFrame(class_rows).to_csv(
        OUT_DIR / "class_diagnostics.csv", index=False
    )

    # Read actual outer-fold results; do not infer individual subject
    # confusion matrices from pooled results.
    fold_path = SOURCE_DIR / "BandPower_SVM_fold_results.csv"
    if fold_path.exists():
        folds = pd.read_csv(fold_path)
        folds.to_csv(OUT_DIR / "subject_performance.csv", index=False)
    else:
        folds = pd.DataFrame()

    report = {
        "experiment": 113,
        "source": "Experiment 110 BandPower_SVM results",
        "trials_represented_by_confusion_matrix": int(cm.sum()),
        "pooled_accuracy_from_confusion_matrix": float(
            accuracy_score(y_true, y_pred)
        ),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "pooled_macro_f1": float(
            f1_score(
                y_true, y_pred, labels=np.arange(4),
                average="macro", zero_division=0
            )
        ),
        "class_diagnostics": class_rows,
        "subject_fold_results_available": not folds.empty,
        "important_limitation": (
            "The saved confusion matrix provides pooled class diagnostics "
            "only. It cannot reveal which classes failed for each subject. "
            "Experiment 110 did not save per-trial predictions, so subject-"
            "specific confusion matrices require rerunning that model while "
            "saving its held-out predictions."
        ),
    }

    with open(OUT_DIR / "experiment113_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nEXPERIMENT 113 COMPLETE")
    print("=" * 60)
    print("Pooled confusion matrix:")
    print(cm_df.to_string())
    print("\nClass diagnostics:")
    print(pd.DataFrame(class_rows).to_string(index=False))
    if not folds.empty:
        print("\nSubject-level results:")
        print(folds.to_string(index=False))
    print("\nReport:")
    print(json.dumps(report, indent=2))
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
