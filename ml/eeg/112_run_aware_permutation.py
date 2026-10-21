
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "experiments" / "ds004022_verified_epochs"
DATA_PATH = DATA_DIR / "eeg_epochs.npz"
AUDIT_PATH = DATA_DIR / "trial_audit.csv"
OUT_DIR = PROJECT / "experiments" / "ds004022_experiment112_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PERMUTATIONS = 1000
N_CLASSES = 4
CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]


def extract_features(X, fs):
    """Per-channel log relative power in five frequency bands."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    result = []

    for epoch in X:
        freqs, psd = welch(
            epoch,
            fs=fs,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )

        total_mask = (freqs >= 1) & (freqs <= 40)
        total = np.trapezoid(
            psd[:, total_mask], x=freqs[total_mask], axis=1
        )
        total = np.maximum(total, 1e-30)

        row = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], x=freqs[mask], axis=1
            )
            row.extend(np.log(np.maximum(power / total, 1e-12)))

        result.append(row)

    return np.asarray(result, dtype=float)


def evaluate_nested_loso(X, y, subjects):
    """Evaluate the complete tuned SVM pipeline with outer LOSO."""
    predictions = np.full(len(y), -1, dtype=int)
    fold_results = []

    for held_out in np.unique(subjects):
        train_idx = np.flatnonzero(subjects != held_out)
        test_idx = np.flatnonzero(subjects == held_out)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = subjects[train_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ])

        search = GridSearchCV(
            model,
            {
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale"],
            },
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=3),
            n_jobs=1,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train, groups=train_groups)
        pred = search.predict(X_test)
        predictions[test_idx] = pred

        fold_results.append({
            "held_out_subject": held_out,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test,
                pred,
                labels=np.arange(N_CLASSES),
                average="macro",
                zero_division=0,
            ),
            "best_params": str(search.best_params_),
        })

    if np.any(predictions < 0):
        raise RuntimeError("Missing outer-fold predictions.")

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            labels=np.arange(N_CLASSES),
            average="macro",
            zero_division=0,
        ),
    }

    return predictions, metrics, pd.DataFrame(fold_results)


def constrained_run_permutation(y, subjects, runs, rng):
    """Shuffle labels within each subject-run, preserving class counts."""
    permuted = y.copy()

    for subject in np.unique(subjects):
        for run in np.unique(runs[subjects == subject]):
            idx = np.flatnonzero(
                (subjects == subject) & (runs == run)
            )
            permuted[idx] = rng.permutation(y[idx])

            # Explicitly verify that this run's class counts are preserved.
            if not np.array_equal(
                np.bincount(permuted[idx], minlength=N_CLASSES),
                np.bincount(y[idx], minlength=N_CLASSES),
            ):
                raise RuntimeError("Permutation changed run class counts.")

    return permuted


def main():
    if not DATA_PATH.exists() or not AUDIT_PATH.exists():
        raise FileNotFoundError("Verified epochs or trial audit is missing.")

    with np.load(DATA_PATH, allow_pickle=True) as d:
        X_raw = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"], dtype=int)
        subjects = np.asarray(d["subjects"]).astype(str)
        runs = np.asarray(d["runs"]).astype(str)
        fs = float(np.asarray(d["sampling_rate"]).squeeze())

    audit = pd.read_csv(AUDIT_PATH)

    if len(X_raw) != len(audit):
        raise ValueError("Epoch count does not match audit.")
    if not np.isfinite(X_raw).all():
        raise ValueError("EEG data contains non-finite values.")
    if set(np.unique(y)) != set(range(N_CLASSES)):
        raise ValueError("Unexpected class IDs.")
    if not np.array_equal(subjects, audit["subject"].astype(str).to_numpy()):
        raise ValueError("Subject metadata does not align.")
    if not np.array_equal(runs.astype(int), audit["run"].to_numpy(int)):
        raise ValueError("Run metadata does not align.")
    if not np.array_equal(y, audit["class_id"].to_numpy(int)):
        raise ValueError("Label metadata does not align.")

    # Every subject-run must have exactly 10 examples of each class.
    for (subject, run), idx in audit.groupby(["subject", "run"]).groups.items():
        counts = np.bincount(y[np.asarray(list(idx))], minlength=N_CLASSES)
        if not np.array_equal(counts, np.repeat(10, N_CLASSES)):
            raise ValueError(
                f"Unexpected class counts in {subject}, run {run}: {counts}"
            )

    print("Extracting features...")
    X = extract_features(X_raw, fs)

    print("Evaluating observed nested LOSO score...")
    observed_pred, observed, observed_folds = evaluate_nested_loso(
        X, y, subjects
    )
    observed_folds.to_csv(OUT_DIR / "observed_fold_results.csv", index=False)

    print("Observed metrics:", observed)

    rng = np.random.default_rng(SEED)
    null_rows = []
    start_time = time.time()

    for p in range(N_PERMUTATIONS):
        y_perm = constrained_run_permutation(y, subjects, runs, rng)

        # Refit and retune the full pipeline for every permutation.
        _, metrics, _ = evaluate_nested_loso(X, y_perm, subjects)

        null_rows.append({
            "permutation": p + 1,
            **metrics,
        })

        if (p + 1) % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"Completed {p + 1}/{N_PERMUTATIONS} permutations "
                f"({elapsed / 60:.1f} minutes elapsed)"
            )

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(OUT_DIR / "permutation_null_scores.csv", index=False)

    p_values = {}
    null_summary = {}

    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        null_scores = null_df[metric].to_numpy()
        observed_score = observed[metric]

        # One-sided empirical p-value with finite-sample correction.
        p_values[metric] = float(
            (1 + np.sum(null_scores >= observed_score))
            / (N_PERMUTATIONS + 1)
        )

        null_summary[metric] = {
            "mean": float(np.mean(null_scores)),
            "std": float(np.std(null_scores, ddof=1)),
            "percentile_95": float(np.quantile(null_scores, 0.95)),
            "percentile_99": float(np.quantile(null_scores, 0.99)),
        }

    report = {
        "experiment": 112,
        "model": "BandPower + RBF SVM",
        "validation": "Nested LOSO; inner GroupKFold by subject",
        "permutation_scheme": (
            "Shuffle labels within each subject-run, preserving each "
            "run's class counts; rerun the full nested LOSO pipeline."
        ),
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "observed_metrics": {
            key: float(value) for key, value in observed.items()
        },
        "permutation_p_values": p_values,
        "null_distribution": null_summary,
        "subjects": sorted(np.unique(subjects).tolist()),
        "n_trials": int(len(y)),
        "chance_accuracy": 0.25,
        "caveat": (
            "This constrained permutation test is appropriate only if the "
            "original task randomization supports exchangeability of trial "
            "labels within each run. It preserves run-level class counts, "
            "but not every possible temporal dependency. Four subjects "
            "provide limited evidence for population-level generalization."
        ),
    }

    with open(
        OUT_DIR / "experiment112_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nEXPERIMENT 112 COMPLETE")
    print(json.dumps(report, indent=2))
    print("\nResults saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
