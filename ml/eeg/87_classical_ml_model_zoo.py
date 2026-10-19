
"""STEP 87: Classical ML model zoo for NeuroSymphony EEG."""

from pathlib import Path
from label_utils import load_subject_labels
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "data/features"
SPLIT_ROOT = ROOT / "data/splits"
OUT = ROOT / "results/eeg/step87_classical_ml_corrected"
OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]
CLASS_NAMES = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]


def load_data():
    data = {s: {"X": [], "y": []} for s in ("train", "val", "test")}

    for subject in SUBJECTS:
        fp = (
            FEATURE_ROOT / f"temporal_spectral_{subject}"
            / f"{subject}_band_power.npy"
        )
        sd = SPLIT_ROOT / subject

        if not fp.exists():
            raise FileNotFoundError(fp)

        X = np.load(fp)
        if X.shape != (720, 64, 5) or not np.isfinite(X).all():
            raise ValueError(f"{subject}: invalid feature shape or values")

        y = load_subject_labels(subject)
        used = []

        for split, expected in [
            ("train", 504), ("val", 108), ("test", 108)
        ]:
            ip = sd / f"{split}_indices.npy"
            if not ip.exists():
                raise FileNotFoundError(ip)

            idx = np.load(ip).astype(int)
            if len(idx) != expected:
                raise ValueError(f"{subject}/{split}: unexpected count")

            if np.any(idx < 0) or np.any(idx >= 720):
                raise ValueError(f"{subject}/{split}: invalid indices")

            used.extend(idx.tolist())
            data[split]["X"].append(X[idx].reshape(len(idx), 320))
            data[split]["y"].append(y[idx])

        if len(set(used)) != 720:
            raise ValueError(f"{subject}: split overlap or missing trials")

        print(f"Validated {subject}")

    result = {}
    for split in ("train", "val", "test"):
        result[split] = {
            "X": np.concatenate(data[split]["X"]).astype(np.float32),
            "y": np.concatenate(data[split]["y"]),
        }

    expected = {"train": 6048, "val": 1296, "test": 1296}
    for split, count in expected.items():
        if len(result[split]["y"]) != count:
            raise ValueError(f"{split}: expected {count} samples")

    # Scaling is fitted on training trials only.
    scaler = StandardScaler()
    result["train"]["X"] = scaler.fit_transform(
        result["train"]["X"]
    ).astype(np.float32)

    for split in ("val", "test"):
        result[split]["X"] = scaler.transform(
            result[split]["X"]
        ).astype(np.float32)

    print("\nDataset verified:")
    for split in ("train", "val", "test"):
        print(split, result[split]["X"].shape)

    return result


def model_candidates():
    # Configurations are compared using validation macro-F1 only.
    return {
        "LogisticRegression": [
            LogisticRegression(C=c, max_iter=2000, random_state=SEED)
            for c in (0.1, 1.0, 10.0)
        ],
        "LinearSVM": [
            LinearSVC(C=c, max_iter=10000, random_state=SEED)
            for c in (0.1, 1.0, 10.0)
        ],
        "RBFSVM": [
            SVC(C=c, gamma=g, kernel="rbf", cache_size=500)
            for c, g in ((1, "scale"), (10, "scale"), (10, 0.001))
        ],
        "LDA": [
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)
            for s in ("auto", 0.1, 0.5)
        ],
        "QDA": [
            QuadraticDiscriminantAnalysis(reg_param=r)
            for r in (0.1, 0.5, 0.9)
        ],
        "KNN": [
            KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)
            for k in (5, 11, 21)
            for w in ("uniform", "distance")
        ],
        "RandomForest": [
            RandomForestClassifier(
                n_estimators=250, max_features=m, min_samples_leaf=leaf,
                class_weight="balanced", n_jobs=-1, random_state=SEED
            )
            for m, leaf in (("sqrt", 1), (0.5, 2), (None, 2))
        ],
        "ExtraTrees": [
            ExtraTreesClassifier(
                n_estimators=300, max_features=m, min_samples_leaf=leaf,
                class_weight="balanced", n_jobs=-1, random_state=SEED
            )
            for m, leaf in (("sqrt", 1), (0.5, 2), (None, 2))
        ],
        "DecisionTree": [
            DecisionTreeClassifier(
                max_depth=d, min_samples_leaf=leaf,
                class_weight="balanced", random_state=SEED
            )
            for d, leaf in ((12, 5), (24, 10), (None, 20))
        ],
        "HistGradientBoosting": [
            HistGradientBoostingClassifier(
                learning_rate=lr, max_iter=iterations, max_leaf_nodes=31,
                l2_regularization=reg, random_state=SEED
            )
            for lr, iterations, reg in (
                (0.05, 150, 1.0), (0.1, 150, 1.0), (0.05, 250, 5.0)
            )
        ],
        "GradientBoosting": [
            GradientBoostingClassifier(
                n_estimators=n, learning_rate=lr, max_depth=d,
                random_state=SEED
            )
            for n, lr, d in ((100, 0.05, 2), (150, 0.05, 2), (100, 0.1, 3))
        ],
        "GaussianNB": [
            GaussianNB(var_smoothing=v)
            for v in (1e-9, 1e-7, 1e-5)
        ],
        "MLP": [
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=layers, alpha=alpha,
                    max_iter=350, early_stopping=True,
                    validation_fraction=0.12, n_iter_no_change=20,
                    random_state=SEED
                )
            )
            for layers, alpha in (
                ((256, 128), 0.001),
                ((256, 128), 0.01),
                ((128, 64), 0.001),
            )
        ],
    }


def main():
    print("=" * 72)
    print("STEP 87: CLASSICAL ML MODEL ZOO")
    print("=" * 72)
    print("Output:", OUT.resolve())

    data = load_data()
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    results = []
    predictions = []

    for name, candidates in model_candidates().items():
        print(f"\n--- {name}: {len(candidates)} configurations ---")
        best_model = None
        best_f1 = -1.0
        best_config = None
        start = time.time()

        for i, model in enumerate(candidates, start=1):
            try:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_f1 = f1_score(
                    y_val, val_pred, average="macro", zero_division=0
                )
                val_acc = accuracy_score(y_val, val_pred)

                print(
                    f"  config {i}/{len(candidates)}: "
                    f"val_macro_f1={val_f1:.4f}, val_acc={val_acc:.4f}"
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model = model
                    best_config = model.get_params(deep=False)

            except Exception as exc:
                print(f"  config {i} failed: {type(exc).__name__}: {exc}")

        if best_model is None:
            print(f"Skipping {name}: all configurations failed")
            continue

        # Evaluate the selected configuration on test data after selection.
        test_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_bal = balanced_accuracy_score(y_test, test_pred)
        test_f1 = f1_score(
            y_test, test_pred, average="macro", zero_division=0
        )

        model_dir = OUT / name
        model_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            classification_report(
                y_test, test_pred, labels=list(range(8)),
                target_names=CLASS_NAMES, output_dict=True, zero_division=0
            )
        ).T.to_csv(model_dir / "classification_report.csv")

        pd.DataFrame(
            confusion_matrix(y_test, test_pred, labels=list(range(8))),
            index=[f"true_{i}_{CLASS_NAMES[i]}" for i in range(8)],
            columns=[f"pred_{i}_{CLASS_NAMES[i]}" for i in range(8)],
        ).to_csv(model_dir / "confusion_matrix.csv")

        pd.DataFrame({
            "true_label": y_test,
            "true_label_name": [CLASS_NAMES[i] for i in y_test],
            "predicted_label": test_pred,
            "predicted_label_name": [CLASS_NAMES[i] for i in test_pred],
            "correct": y_test == test_pred,
        }).to_csv(model_dir / "test_predictions.csv", index=False)

        with open(model_dir / "selected_config.json", "w") as f:
            json.dump(
                {k: str(v) for k, v in best_config.items()},
                f, indent=2
            )

        elapsed = time.time() - start
        results.append({
            "model": name,
            "selected_validation_macro_f1": best_f1,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bal,
            "test_macro_f1": test_f1,
            "runtime_seconds": elapsed,
        })

        predictions.append(pd.DataFrame({
            "model": name,
            "true_label": y_test,
            "predicted_label": test_pred,
        }))

        print(
            f"SELECTED {name}: val_macro_f1={best_f1:.4f} | "
            f"test_acc={test_acc:.4f} | test_macro_f1={test_f1:.4f} | "
            f"time={elapsed:.1f}s"
        )

    result_df = pd.DataFrame(results)
    result_df.sort_values(
        "selected_validation_macro_f1", ascending=False, inplace=True
    )
    result_df.to_csv(OUT / "step87_model_comparison.csv", index=False)

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            OUT / "step87_test_predictions.csv", index=False
        )

    with open(OUT / "step87_metadata.json", "w") as f:
        json.dump({
            "step": 87,
            "subjects": 12,
            "features_per_trial": 320,
            "split_sizes": {"train": 6048, "validation": 1296, "test": 1296},
            "label_order": CLASS_NAMES,
            "scaler": "StandardScaler fitted on training data only",
            "model_selection": "validation macro-F1",
            "seed": SEED,
        }, f, indent=2)

    print("\n" + "=" * 72)
    print("STEP 87 COMPLETE")
    print("=" * 72)
    if not result_df.empty:
        print(result_df.to_string(index=False))
    print("\nResults saved to:", OUT.resolve())
    print("Existing experiments were not overwritten.")


if __name__ == "__main__":
    main()
