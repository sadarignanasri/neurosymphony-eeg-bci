
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
OUT = PROJECT / "experiments" / "ds004022_experiment117_results"
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
        raise ValueError("Epoch and metadata lengths do not match.")
    if not np.isfinite(X).all():
        raise ValueError("EEG data contains NaN or infinite values.")
    if set(np.unique(y)) != set(range(4)):
        raise ValueError(f"Unexpected class IDs: {np.unique(y)}")

    return X, y, subjects, runs


def extract_feature_sets(X):
    """
    Create feature sets:
      relative_power: 90 channel-wise log-relative band powers
      absolute_power: 90 channel-wise log absolute band powers
      time_domain:    36 channel-wise log variance and log RMS features
      combined:       all 216 features concatenated
    """
    relative_rows = []
    absolute_rows = []
    time_rows = []

    for epoch in X:
        # Welch PSD, matching the current 1-40 Hz band-power baseline.
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

        relative_features = []
        absolute_features = []

        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], freqs[mask], axis=1
            )
            power = np.maximum(power, 1e-12)

            relative_features.append(
                np.log(np.maximum(power / total_power, 1e-12))
            )
            absolute_features.append(np.log(power))

        relative_rows.append(np.concatenate(relative_features))
        absolute_rows.append(np.concatenate(absolute_features))

        # Time-domain features, independently for each EEG channel.
        variance = np.var(epoch, axis=1)
        rms = np.sqrt(np.mean(epoch ** 2, axis=1))
        time_rows.append(np.concatenate([
            np.log(np.maximum(variance, 1e-20)),
            np.log(np.maximum(rms, 1e-20)),
        ]))

    relative_power = np.asarray(relative_rows)
    absolute_power = np.asarray(absolute_rows)
    time_domain = np.asarray(time_rows)

    return {
        "relative_power": relative_power,
        "absolute_power": absolute_power,
        "time_domain": time_domain,
        "combined": np.concatenate(
            [relative_power, absolute_power, time_domain], axis=1
        ),
    }


def make_search():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            decision_function_shape="ovr",
        )),
    ])

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale"],
        },
        scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=3),
        n_jobs=1,
        refit=True,
    )


def main():
    X, y, subjects, runs = load_data()

    print(f"Dataset: {X.shape}")
    print(f"Subjects: {sorted(np.unique(subjects))}")
    print("Extracting feature representations...")

    feature_sets = extract_feature_sets(X)
    for name, features in feature_sets.items():
        print(f"  {name}: {features.shape}")

    outer_cv = LeaveOneGroupOut()
    result_rows = []
    selected_rows = []
    selected_predictions = np.full(len(y), -1, dtype=int)
    selected_representations = [""] * len(y)

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=subjects), start=1
    ):
        held_out_subject = subjects[test_idx[0]]
        fold_candidates = []

        # Compare each fixed feature representation on the same outer fold.
        # Inner CV uses only the three training subjects.
        for name, features in feature_sets.items():
            search = make_search()
            search.fit(
                features[train_idx],
                y[train_idx],
                groups=subjects[train_idx],
            )

            pred = search.predict(features[test_idx])

            result_rows.append({
                "fold": fold,
                "held_out_subject": held_out_subject,
                "representation": name,
                "n_features": int(features.shape[1]),
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

            fold_candidates.append((search.best_score_, name, search))

        # Choose representation using inner-CV scores only.
        fold_candidates.sort(key=lambda item: item[0], reverse=True)
        inner_score, chosen_name, chosen_search = fold_candidates[0]

        chosen_features = feature_sets[chosen_name]
        chosen_pred = chosen_search.predict(chosen_features[test_idx])
        selected_predictions[test_idx] = chosen_pred

        for idx in test_idx:
            selected_representations[idx] = chosen_name

        selected_rows.append({
            "fold": fold,
            "held_out_subject": held_out_subject,
            "selected_representation": chosen_name,
            "n_features": int(chosen_features.shape[1]),
            "inner_best_balanced_accuracy": float(inner_score),
            "outer_accuracy": float(accuracy_score(y[test_idx], chosen_pred)),
            "outer_balanced_accuracy": float(
                balanced_accuracy_score(y[test_idx], chosen_pred)
            ),
            "outer_macro_f1": float(
                f1_score(
                    y[test_idx], chosen_pred,
                    average="macro", zero_division=0
                )
            ),
            "best_params": json.dumps(chosen_search.best_params_),
        })

        print(
            f"Fold {fold}/4 | {held_out_subject} | "
            f"selected={chosen_name} | "
            f"accuracy={accuracy_score(y[test_idx], chosen_pred):.3f}"
        )

    if (selected_predictions < 0).any():
        raise RuntimeError("Some trials are missing held-out predictions.")

    # Save fold-level comparison.
    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(OUT / "feature_outer_fold_results.csv", index=False)

    # Each subject contributes equally to these mean fold metrics.
    summary = comparison.groupby("representation").agg(
        mean_accuracy=("outer_accuracy", "mean"),
        std_accuracy=("outer_accuracy", "std"),
        mean_balanced_accuracy=("outer_balanced_accuracy", "mean"),
        mean_macro_f1=("outer_macro_f1", "mean"),
        feature_count=("n_features", "first"),
    ).reset_index().sort_values("mean_accuracy", ascending=False)

    summary.to_csv(OUT / "feature_summary.csv", index=False)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(
        OUT / "nested_feature_selection_results.csv", index=False
    )

    pd.DataFrame({
        "subject": subjects,
        "run": runs,
        "true_class_id": y,
        "true_class": [CLASS_NAMES[i] for i in y],
        "predicted_class_id": selected_predictions,
        "predicted_class": [CLASS_NAMES[i] for i in selected_predictions],
        "selected_representation": selected_representations,
        "correct": selected_predictions == y,
    }).to_csv(OUT / "nested_feature_selection_predictions.csv", index=False)

    final_report = {
        "experiment": 117,
        "dataset_epochs": int(len(y)),
        "subjects": int(len(np.unique(subjects))),
        "classes": CLASS_NAMES,
        "evaluation": "Nested LOSO with inner GroupKFold by subject",
        "feature_dimensions": {
            name: int(features.shape[1])
            for name, features in feature_sets.items()
        },
        "nested_selected_accuracy": float(
            accuracy_score(y, selected_predictions)
        ),
        "nested_selected_balanced_accuracy": float(
            balanced_accuracy_score(y, selected_predictions)
        ),
        "nested_selected_macro_f1": float(
            f1_score(
                y, selected_predictions,
                average="macro", zero_division=0
            )
        ),
        "chance_level": 0.25,
        "limitations": [
            "Only four subjects are available for cross-subject evaluation.",
            "Absolute power may be affected by recording scale and artifacts.",
            "Feature representations are selected using inner training-subject CV.",
            "Results do not establish generalization to a broader population."
        ],
    }

    with open(OUT / "experiment117_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\nEXPERIMENT 117 COMPLETE")
    print("\nFeature comparison:")
    print(summary.to_string(index=False))
    print("\nNested feature selection by held-out subject:")
    print(selected_df.to_string(index=False))
    print("\nFinal report:")
    print(json.dumps(final_report, indent=2))
    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
