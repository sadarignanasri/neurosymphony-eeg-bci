
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
OUT_DIR = PROJECT / "experiments" / "ds004022_experiment111_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PERMUTATIONS = 1000
N_CLASSES = 4
CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]


def extract_features(X, fs):
    """Extract per-channel log relative band-power features."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    features = []

    for epoch in X:
        freqs, psd = welch(
            epoch, fs=fs, nperseg=min(1000, epoch.shape[-1]), axis=-1
        )
        total_mask = (freqs >= 1) & (freqs <= 40)
        total = np.trapezoid(
            psd[:, total_mask], x=freqs[total_mask], axis=1
        )
        total = np.maximum(total, 1e-30)

        row = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(psd[:, mask], x=freqs[mask], axis=1)
            row.extend(np.log(np.maximum(power / total, 1e-12)))

        features.append(row)

    return np.asarray(features, dtype=float)


def evaluate_loso(X, y, subjects):
    """Tune SVM only within training subjects, then predict held-out subject."""
    predictions = np.full(len(y), -1, dtype=int)
    fold_rows = []

    for held_out in np.unique(subjects):
        train = np.where(subjects != held_out)[0]
        test = np.where(subjects == held_out)[0]

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        groups_train = subjects[train]

        model = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ])

        search = GridSearchCV(
            model,
            {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale"]},
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=3),
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups_train)
        pred = search.predict(X_test)
        predictions[test] = pred

        fold_rows.append({
            "held_out_subject": held_out,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred, labels=np.arange(N_CLASSES),
                average="macro", zero_division=0
            ),
            "best_params": str(search.best_params_),
        })

    if np.any(predictions < 0):
        raise RuntimeError("Some trials were not predicted.")

    return predictions, pd.DataFrame(fold_rows)


def main():
    rng = np.random.default_rng(SEED)

    with np.load(DATA_PATH, allow_pickle=True) as d:
        X_raw = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"], dtype=int)
        subjects = np.asarray(d["subjects"]).astype(str)
        fs = float(np.asarray(d["sampling_rate"]).squeeze())

    if not np.isfinite(X_raw).all():
        raise ValueError("EEG data contains non-finite values.")
    if len(np.unique(subjects)) != 4:
        raise ValueError("This validation script expects the four known subjects.")

    print("Extracting features...")
    X = extract_features(X_raw, fs)

    # Actual observed performance.
    observed_pred, observed_folds = evaluate_loso(X, y, subjects)
    observed_accuracy = accuracy_score(y, observed_pred)
    observed_balanced = balanced_accuracy_score(y, observed_pred)
    observed_macro_f1 = f1_score(
        y, observed_pred, labels=np.arange(N_CLASSES),
        average="macro", zero_division=0
    )

    observed_folds.to_csv(OUT_DIR / "observed_fold_results.csv", index=False)

    # Permutation null: shuffle labels within each subject, preserving
    # each subject's class counts, then rerun the entire nested LOSO pipeline.
    null_accuracy = []
    null_balanced = []
    null_macro_f1 = []

    for p in range(N_PERMUTATIONS):
        y_perm = y.copy()

        for subject in np.unique(subjects):
            idx = np.where(subjects == subject)[0]
            y_perm[idx] = rng.permutation(y_perm[idx])

        pred_perm, _ = evaluate_loso(X, y_perm, subjects)

        null_accuracy.append(accuracy_score(y_perm, pred_perm))
        null_balanced.append(balanced_accuracy_score(y_perm, pred_perm))
        null_macro_f1.append(f1_score(
            y_perm, pred_perm, labels=np.arange(N_CLASSES),
            average="macro", zero_division=0
        ))

        if (p + 1) % 25 == 0:
            print(f"Completed {p + 1}/{N_PERMUTATIONS} permutations")

    null_accuracy = np.asarray(null_accuracy)
    null_balanced = np.asarray(null_balanced)
    null_macro_f1 = np.asarray(null_macro_f1)

    # One-sided empirical permutation p-values.
    p_accuracy = (
        1 + np.sum(null_accuracy >= observed_accuracy)
    ) / (N_PERMUTATIONS + 1)

    p_balanced = (
        1 + np.sum(null_balanced >= observed_balanced)
    ) / (N_PERMUTATIONS + 1)

    p_macro_f1 = (
        1 + np.sum(null_macro_f1 >= observed_macro_f1)
    ) / (N_PERMUTATIONS + 1)

    null_df = pd.DataFrame({
        "permutation": np.arange(1, N_PERMUTATIONS + 1),
        "accuracy": null_accuracy,
        "balanced_accuracy": null_balanced,
        "macro_f1": null_macro_f1,
    })
    null_df.to_csv(OUT_DIR / "permutation_null_scores.csv", index=False)

    report = {
        "experiment": 111,
        "model": "BandPower + RBF SVM",
        "validation": "Nested LOSO; inner GroupKFold",
        "permutations": N_PERMUTATIONS,
        "seed": SEED,
        "observed": {
            "accuracy": float(observed_accuracy),
            "balanced_accuracy": float(observed_balanced),
            "macro_f1": float(observed_macro_f1),
        },
        "permutation_p_values": {
            "accuracy": float(p_accuracy),
            "balanced_accuracy": float(p_balanced),
            "macro_f1": float(p_macro_f1),
        },
        "null_distribution": {
            "accuracy_mean": float(null_accuracy.mean()),
            "accuracy_95th_percentile": float(np.quantile(null_accuracy, 0.95)),
            "balanced_accuracy_mean": float(null_balanced.mean()),
            "balanced_accuracy_95th_percentile": float(
                np.quantile(null_balanced, 0.95)
            ),
            "macro_f1_mean": float(null_macro_f1.mean()),
            "macro_f1_95th_percentile": float(np.quantile(null_macro_f1, 0.95)),
        },
        "warning": (
            "Permutation p-values are conditional on this evaluation design. "
            "Only four subjects are available. Shuffling labels within subject "
            "preserves class counts but not temporal dependence or run structure; "
            "interpret results cautiously."
        ),
    }

    with open(OUT_DIR / "experiment111_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nEXPERIMENT 111 COMPLETE")
    print(json.dumps(report, indent=2))
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()
