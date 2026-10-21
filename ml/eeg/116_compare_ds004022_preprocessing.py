
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
OUT = PROJECT / "experiments" / "ds004022_experiment116_results"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["reaching", "grasping", "lifting", "twisting"]
FS = 500
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]

# The input epochs were already bandpass-filtered at 1–40 Hz during
# extraction. Other variants below narrow this existing frequency range.
VARIANTS = {
    "baseline_1_40": None,
    "narrow_4_30": (4, 30),
    "motor_7_30": (7, 30),
    "alpha_beta_8_30": (8, 30),
}


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    data = np.load(DATA, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(int)
    subjects = data["subjects"].astype(str)
    runs = data["runs"].astype(str)

    if X.shape[0] != len(y) or len(y) != len(subjects) or len(y) != len(runs):
        raise ValueError("Mismatch between epochs and metadata.")

    if not np.isfinite(X).all():
        raise ValueError("EEG epochs contain NaN or infinite values.")

    if set(np.unique(y)) != set(range(4)):
        raise ValueError(f"Unexpected class IDs: {np.unique(y)}")

    return X, y, subjects, runs


def filter_epochs(X, low_high):
    """Apply a fixed zero-phase filter independently to each epoch."""
    if low_high is None:
        return X.copy()

    low, high = low_high
    sos = butter(
        4, [low, high], btype="bandpass", fs=FS, output="sos"
    )
    return np.stack([
        sosfiltfilt(sos, epoch, axis=-1)
        for epoch in X
    ])


def extract_features(X):
    """Extract 90 channel-wise log-relative band-power features."""
    result = []

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

        parts = []
        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = np.trapezoid(
                psd[:, mask], freqs[mask], axis=1
            )
            relative_power = power / total_power
            parts.append(np.log(np.maximum(relative_power, 1e-12)))

        result.append(np.concatenate(parts))

    return np.asarray(result)


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            decision_function_shape="ovr",
        )),
    ])


def make_search():
    return GridSearchCV(
        estimator=make_model(),
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
    print(f"Dataset: {X.shape}; subjects={len(np.unique(subjects))}")
    print("Preparing preprocessing variants...")

    features_by_variant = {}
    for name, band in VARIANTS.items():
        filtered = filter_epochs(X, band)
        features_by_variant[name] = extract_features(filtered)
        print(f"  {name}: {features_by_variant[name].shape}")

    outer_cv = LeaveOneGroupOut()
    comparison_rows = []
    selected_rows = []
    selected_predictions = np.full(len(y), -1, dtype=int)
    selected_variants = [""] * len(y)

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=subjects), start=1
    ):
        held_out = subjects[test_idx[0]]
        fold_candidates = []

        # Every preprocessing/model choice is evaluated using only
        # the training subjects' inner grouped cross-validation.
        for name, features in features_by_variant.items():
            search = make_search()
            search.fit(
                features[train_idx],
                y[train_idx],
                groups=subjects[train_idx],
            )

            pred = search.predict(features[test_idx])
            row = {
                "fold": fold,
                "held_out_subject": held_out,
                "variant": name,
                "inner_best_balanced_accuracy": search.best_score_,
                "outer_accuracy": accuracy_score(y[test_idx], pred),
                "outer_balanced_accuracy": balanced_accuracy_score(
                    y[test_idx], pred
                ),
                "outer_macro_f1": f1_score(
                    y[test_idx], pred, average="macro", zero_division=0
                ),
                "best_params": json.dumps(search.best_params_),
            }
            comparison_rows.append(row)
            fold_candidates.append((search.best_score_, name, search))

        # Select the preprocessing variant from inner CV only.
        fold_candidates.sort(key=lambda item: item[0], reverse=True)
        best_inner_score, chosen_name, chosen_search = fold_candidates[0]

        chosen_features = features_by_variant[chosen_name]
        chosen_pred = chosen_search.predict(chosen_features[test_idx])
        selected_predictions[test_idx] = chosen_pred

        for idx, pred in zip(test_idx, chosen_pred):
            selected_variants[idx] = chosen_name

        selected_rows.append({
            "fold": fold,
            "held_out_subject": held_out,
            "selected_variant": chosen_name,
            "inner_best_balanced_accuracy": best_inner_score,
            "outer_accuracy": accuracy_score(y[test_idx], chosen_pred),
            "outer_balanced_accuracy": balanced_accuracy_score(
                y[test_idx], chosen_pred
            ),
            "outer_macro_f1": f1_score(
                y[test_idx], chosen_pred, average="macro", zero_division=0
            ),
            "best_params": json.dumps(chosen_search.best_params_),
        })

        print(
            f"Fold {fold}/4 | {held_out} | selected={chosen_name} | "
            f"accuracy={accuracy_score(y[test_idx], chosen_pred):.3f}"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "variant_outer_fold_results.csv", index=False)

    # Descriptive comparison: each variant's outer-fold performance.
    summary = comparison.groupby("variant").agg(
        mean_accuracy=("outer_accuracy", "mean"),
        std_accuracy=("outer_accuracy", "std"),
        mean_balanced_accuracy=("outer_balanced_accuracy", "mean"),
        mean_macro_f1=("outer_macro_f1", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "variant_summary.csv", index=False)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT / "nested_selection_fold_results.csv", index=False)

    pd.DataFrame({
        "subject": subjects,
        "run": runs,
        "true_class_id": y,
        "true_class": [CLASS_NAMES[i] for i in y],
        "predicted_class_id": selected_predictions,
        "predicted_class": [CLASS_NAMES[i] for i in selected_predictions],
        "selected_variant": selected_variants,
        "correct": selected_predictions == y,
    }).to_csv(OUT / "nested_selection_predictions.csv", index=False)

    final_report = {
        "experiment": 116,
        "dataset_epochs": int(len(y)),
        "subjects": int(len(np.unique(subjects))),
        "classes": CLASS_NAMES,
        "evaluation": "LOSO; preprocessing/model selection by inner GroupKFold",
        "variants": list(VARIANTS.keys()),
        "nested_selected_accuracy": float(
            accuracy_score(y, selected_predictions)
        ),
        "nested_selected_balanced_accuracy": float(
            balanced_accuracy_score(y, selected_predictions)
        ),
        "nested_selected_macro_f1": float(
            f1_score(y, selected_predictions, average="macro", zero_division=0)
        ),
        "important_caveat": (
            "Epochs were already filtered at 1–40 Hz during extraction. "
            "This experiment only compares additional narrower filters, "
            "not alternative filters applied to the original continuous raw "
            "recordings. Four subjects provide limited generalization evidence."
        ),
    }

    with open(OUT / "experiment116_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\nEXPERIMENT 116 COMPLETE")
    print("\nOuter-fold variant comparison:")
    print(summary.sort_values("mean_accuracy", ascending=False).to_string(index=False))
    print("\nNested preprocessing selection:")
    print(selected_df.to_string(index=False))
    print("\nFinal report:")
    print(json.dumps(final_report, indent=2))
    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
