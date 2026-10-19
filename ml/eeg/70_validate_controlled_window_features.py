
"""Validate controlled window features, labels, and existing splits."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features"
SPLITS = ROOT / "data" / "splits"

N_TRIALS = 720
N_CHANNELS = 64
N_BANDS = 5


def subject_paths(subject):
    # Subject 01 has a legacy directory naming convention.
    split_subject = "sub01" if subject == "sub-01" else subject

    if subject == "sub-01":
        label_file = (
            FEATURES / "temporal_spectral_sub01"
            / "sub-01_spectral_metadata.csv"
        )
        label_column = "label"
    else:
        label_file = (
            FEATURES / f"temporal_eeg_{subject}"
            / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
        )
        label_column = "label_id"

    return split_subject, label_file, label_column


def main():
    total = 0

    for number in range(1, 13):
        subject = f"sub-{number:02d}"
        split_subject, label_file, label_column = subject_paths(subject)

        print(f"\n{subject}")

        if not label_file.exists():
            raise FileNotFoundError(label_file)

        metadata = pd.read_csv(label_file)

        if label_column not in metadata.columns:
            raise ValueError(
                f"{subject}: missing label column {label_column}"
            )

        labels = metadata[label_column].to_numpy()

        if len(labels) != N_TRIALS:
            raise ValueError(
                f"{subject}: expected {N_TRIALS} labels, got {len(labels)}"
            )

        if not np.array_equal(np.unique(labels), np.arange(8)):
            raise ValueError(
                f"{subject}: expected labels 0-7, got {np.unique(labels)}"
            )

        split_dir = SPLITS / split_subject
        train = np.load(split_dir / "train_indices.npy")
        val = np.load(split_dir / "val_indices.npy")
        test = np.load(split_dir / "test_indices.npy")

        if (
            len(np.unique(np.concatenate([train, val, test])))
            != N_TRIALS
        ):
            raise ValueError(f"{subject}: split overlap or missing indices")

        if (
            min(train.min(), val.min(), test.min()) < 0
            or max(train.max(), val.max(), test.max()) >= N_TRIALS
        ):
            raise ValueError(f"{subject}: split index out of range")

        for condition in ("rectangular", "hann"):
            feature_file = (
                FEATURES
                / f"controlled_{'rect' if condition == 'rectangular' else 'hann'}_spectral_{subject}"
                / f"{subject}_{condition}_band_power.npy"
            )

            if not feature_file.exists():
                raise FileNotFoundError(feature_file)

            X = np.load(feature_file, mmap_mode="r")

            if X.shape != (N_TRIALS, N_CHANNELS, N_BANDS):
                raise ValueError(
                    f"{subject}/{condition}: invalid shape {X.shape}"
                )

            if not np.isfinite(np.asarray(X[:2])).all():
                raise ValueError(
                    f"{subject}/{condition}: invalid sample values"
                )

            print(
                f"  {condition}: shape={X.shape}; "
                f"train/val/test={len(train)}/{len(val)}/{len(test)}"
            )

            del X

        print(f"  Labels: PASS; unique classes: {len(np.unique(labels))}")
        print("  Splits: PASS; no overlap and all indices in range")
        total += 1

    print(f"\nPASS: {total}/12 subjects validated.")


if __name__ == "__main__":
    main()
