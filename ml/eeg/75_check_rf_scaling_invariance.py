
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path.cwd()

OUTPUT_DIR = (
    ROOT
    / "results"
    / "eeg"
    / "signal_processing"
    / "EEG_SP_01_window_validation"
)

# Preserve the previous float32 report.
OUTPUT_FILE = OUTPUT_DIR / "rf_scaling_invariance_float64.csv"


def get_original_feature_path(subject_num):
    if subject_num == 1:
        folder = ROOT / "data/features/temporal_spectral_sub01"
    else:
        folder = ROOT / (
            f"data/features/temporal_spectral_sub-{subject_num:02d}"
        )

    return folder / f"sub-{subject_num:02d}_band_power.npy"


def get_rectangular_feature_path(subject_num):
    folder = ROOT / (
        f"data/features/controlled_rect_spectral_sub-{subject_num:02d}"
    )

    return folder / (
        f"sub-{subject_num:02d}_rectangular_band_power.npy"
    )


def get_label_path(subject_num):
    if subject_num == 1:
        return (
            ROOT / "data/features/temporal_spectral_sub01"
            / "sub-01_spectral_metadata.csv"
        )

    return (
        ROOT / "data/features/temporal_eeg_sub-"
        f"{subject_num:02d}"
        / f"sub-{subject_num:02d}_temporal_eeg_preprocessed_metadata.csv"
    )


def get_split_path(subject_num, split_name):
    if subject_num == 1:
        folder = ROOT / "data/splits/sub01"
    else:
        folder = ROOT / f"data/splits/sub-{subject_num:02d}"

    return folder / f"{split_name}_indices.npy"


def load_labels(subject_num):
    metadata = pd.read_csv(get_label_path(subject_num))

    label_column = "label" if subject_num == 1 else "label_id"

    if label_column not in metadata.columns:
        raise ValueError(
            f"Missing label column '{label_column}' "
            f"for subject {subject_num}"
        )

    return metadata[label_column].to_numpy(dtype=int)


def load_split(subject_num, split_name):
    return np.load(get_split_path(subject_num, split_name))


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Refusing to overwrite the existing report:\n"
            f"{OUTPUT_FILE}\n"
            "Rename or remove that report only if you intend to rerun."
        )

    rows = []

    for subject_num in range(1, 13):
        subject = f"sub-{subject_num:02d}"

        original_path = get_original_feature_path(subject_num)
        rectangular_path = get_rectangular_feature_path(subject_num)

        if not original_path.exists():
            raise FileNotFoundError(
                f"Original features not found: {original_path}"
            )

        if not rectangular_path.exists():
            raise FileNotFoundError(
                f"Rectangular features not found: {rectangular_path}"
            )

        # Use float64 for this precision diagnostic.
        original = np.load(original_path).astype(np.float64)
        rectangular = np.load(rectangular_path).astype(np.float64)

        labels = load_labels(subject_num)
        train_idx = load_split(subject_num, "train")
        val_idx = load_split(subject_num, "val")

        if original.shape != rectangular.shape:
            raise ValueError(
                f"{subject}: feature shapes differ: "
                f"{original.shape} vs {rectangular.shape}"
            )

        if original.ndim != 3 or original.shape[1:] != (64, 5):
            raise ValueError(
                f"{subject}: unexpected feature shape {original.shape}"
            )

        if original.shape[0] != len(labels):
            raise ValueError(
                f"{subject}: feature count {original.shape[0]} "
                f"does not match label count {len(labels)}"
            )

        if not (
            np.isfinite(original).all()
            and np.isfinite(rectangular).all()
        ):
            raise ValueError(f"{subject}: non-finite feature values found")

        # Verify the known normalization relationship.
        expected_rectangular = original / 6142.0

        if not np.allclose(
            expected_rectangular,
            rectangular,
            rtol=2e-6,
            atol=1e-12,
        ):
            raise ValueError(
                f"{subject}: expected scaling relationship failed"
            )

        # Verify split indices before training.
        for name, indices in (
            ("train", train_idx),
            ("val", val_idx),
        ):
            if indices.ndim != 1:
                raise ValueError(f"{subject}: {name} indices must be 1-D")

            if np.any(indices < 0) or np.any(indices >= len(labels)):
                raise ValueError(f"{subject}: invalid {name} indices")

        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"{subject}: train/validation overlap detected")

        X_original = original.reshape(original.shape[0], -1)
        X_rectangular = rectangular.reshape(rectangular.shape[0], -1)

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        model_original = make_model()
        model_rectangular = make_model()

        model_original.fit(X_original[train_idx], y_train)
        model_rectangular.fit(X_rectangular[train_idx], y_train)

        pred_original = model_original.predict(X_original[val_idx])
        pred_rectangular = model_rectangular.predict(
            X_rectangular[val_idx]
        )

        agreement = float(np.mean(pred_original == pred_rectangular))

        row = {
            "subject": subject,
            "input_dtype": "float64",
            "original_accuracy": accuracy_score(y_val, pred_original),
            "rectangular_accuracy": accuracy_score(
                y_val, pred_rectangular
            ),
            "original_macro_f1": f1_score(
                y_val,
                pred_original,
                average="macro",
                zero_division=0,
            ),
            "rectangular_macro_f1": f1_score(
                y_val,
                pred_rectangular,
                average="macro",
                zero_division=0,
            ),
            "prediction_agreement": agreement,
            "identical_predictions": bool(
                np.array_equal(pred_original, pred_rectangular)
            ),
        }

        rows.append(row)

        print(
            f"{subject} | "
            f"agreement={agreement:.2%} | "
            f"original_F1={row['original_macro_f1']:.4f} | "
            f"rectangular_F1={row['rectangular_macro_f1']:.4f} | "
            f"identical={row['identical_predictions']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\nMEAN VALIDATION RESULTS")

    metric_columns = [
        "original_accuracy",
        "rectangular_accuracy",
        "original_macro_f1",
        "rectangular_macro_f1",
        "prediction_agreement",
    ]

    print(report[metric_columns].mean().to_string())

    print(f"\nSaved report: {OUTPUT_FILE}")
    print("Test data was not evaluated.")


if __name__ == "__main__":
    main()
