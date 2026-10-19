
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

OUTPUT_DIR = (
    ROOT
    / "results"
    / "eeg"
    / "signal_processing"
    / "EEG_SP_01_window_validation"
)

OUTPUT_FILE = OUTPUT_DIR / "fft_normalization_verification.csv"
WINDOW_LENGTH = 6142


def get_original_path(subject_num):
    if subject_num == 1:
        folder = ROOT / "data/features/temporal_spectral_sub01"
    else:
        folder = ROOT / (
            f"data/features/temporal_spectral_sub-{subject_num:02d}"
        )

    return folder / f"sub-{subject_num:02d}_band_power.npy"


def get_rectangular_path(subject_num):
    folder = ROOT / (
        f"data/features/controlled_rect_spectral_sub-{subject_num:02d}"
    )

    return folder / (
        f"sub-{subject_num:02d}_rectangular_band_power.npy"
    )


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Report already exists; refusing to overwrite: {OUTPUT_FILE}"
        )

    rows = []

    for subject_num in range(1, 13):
        original_path = get_original_path(subject_num)
        rectangular_path = get_rectangular_path(subject_num)

        if not original_path.exists():
            raise FileNotFoundError(f"Missing: {original_path}")

        if not rectangular_path.exists():
            raise FileNotFoundError(f"Missing: {rectangular_path}")

        original = np.load(original_path).astype(np.float64)
        rectangular = np.load(rectangular_path).astype(np.float64)

        if original.shape != rectangular.shape:
            raise ValueError(
                f"Shape mismatch for subject {subject_num}: "
                f"{original.shape} vs {rectangular.shape}"
            )

        if not (
            np.isfinite(original).all()
            and np.isfinite(rectangular).all()
        ):
            raise ValueError(
                f"Non-finite values found for subject {subject_num}"
            )

        expected_rectangular = original / WINDOW_LENGTH
        scaling_error = rectangular - expected_rectangular

        relative_error = (
            np.linalg.norm(scaling_error)
            / max(np.linalg.norm(expected_rectangular), 1e-30)
        )

        max_error = np.max(np.abs(scaling_error))

        ratio = rectangular / np.maximum(
            np.abs(original), 1e-30
        )

        row = {
            "subject": f"sub-{subject_num:02d}",
            "shape": str(original.shape),
            "expected_ratio": 1.0 / WINDOW_LENGTH,
            "median_observed_ratio": float(np.median(ratio)),
            "expected_scaling_relative_error": float(relative_error),
            "max_scaling_error": float(max_error),
        }

        rows.append(row)

        print(
            f"{row['subject']} | "
            f"expected_ratio={row['expected_ratio']:.9f} | "
            f"observed_ratio={row['median_observed_ratio']:.9f} | "
            f"relative_error={row['expected_scaling_relative_error']:.9e}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\nReport saved: {OUTPUT_FILE}")
    print("No classifier was trained and no test metrics were evaluated.")


if __name__ == "__main__":
    main()
