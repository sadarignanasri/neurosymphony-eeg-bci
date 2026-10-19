
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

OUTPUT_FILE = OUTPUT_DIR / "feature_equivalence_diagnostic.csv"


def original_feature_path(subject_num):
    if subject_num == 1:
        folder = ROOT / "data/features/temporal_spectral_sub01"
        filename = "sub-01_band_power.npy"
    else:
        folder = (
            ROOT
            / f"data/features/temporal_spectral_sub-{subject_num:02d}"
        )
        filename = f"sub-{subject_num:02d}_band_power.npy"

    return folder / filename


def rectangular_feature_path(subject_num):
    folder = (
        ROOT
        / f"data/features/controlled_rect_spectral_sub-{subject_num:02d}"
    )
    filename = (
        f"sub-{subject_num:02d}_rectangular_band_power.npy"
    )
    return folder / filename


def compare_arrays(original, rectangular):
    original = np.asarray(original, dtype=np.float64)
    rectangular = np.asarray(rectangular, dtype=np.float64)

    result = {
        "original_shape": str(original.shape),
        "rectangular_shape": str(rectangular.shape),
        "shape_match": original.shape == rectangular.shape,
        "original_finite": bool(np.isfinite(original).all()),
        "rectangular_finite": bool(np.isfinite(rectangular).all()),
    }

    if original.shape != rectangular.shape:
        return result

    if not (
        np.isfinite(original).all()
        and np.isfinite(rectangular).all()
    ):
        return result

    difference = rectangular - original
    

    abs_original = np.abs(original)

    result["original_mean"] = float(np.mean(original))
    result["rectangular_mean"] = float(np.mean(rectangular))
    result["original_std"] = float(np.std(original))
    result["rectangular_std"] = float(np.std(rectangular))
    result["mean_absolute_difference"] = float(
        np.mean(np.abs(difference))
    )
    result["max_absolute_difference"] = float(
        np.max(np.abs(difference))
    )

    denominator = np.linalg.norm(original)
    result["relative_l2_difference"] = (
        float(np.linalg.norm(difference) / denominator)
        if denominator > 0
        else np.nan
    )

    # A ratio is meaningful only where the original is not near zero.
    threshold = max(float(np.max(abs_original)) * 1e-8, 1e-12)
    mask = abs_original > threshold

    if mask.any():
        ratios = rectangular[mask] / original[mask]
        result["ratio_median"] = float(np.median(ratios))
        result["ratio_p05"] = float(np.percentile(ratios, 5))
        result["ratio_p95"] = float(np.percentile(ratios, 95))
    else:
        result["ratio_median"] = np.nan
        result["ratio_p05"] = np.nan
        result["ratio_p95"] = np.nan

    # Correlation helps assess whether both pipelines preserve
    # the relative variation among corresponding feature values.
    x = original.ravel()
    y = rectangular.ravel()

    if np.std(x) > 0 and np.std(y) > 0:
        result["value_correlation"] = float(np.corrcoef(x, y)[0, 1])
    else:
        result["value_correlation"] = np.nan

    return result


def main():
    rows = []

    for subject_num in range(1, 13):
        original_path = original_feature_path(subject_num)
        rectangular_path = rectangular_feature_path(subject_num)

        if not original_path.exists():
            raise FileNotFoundError(
                f"Missing original features: {original_path}"
            )

        if not rectangular_path.exists():
            raise FileNotFoundError(
                f"Missing rectangular features: {rectangular_path}"
            )

        original = np.load(original_path, mmap_mode="r")
        rectangular = np.load(rectangular_path, mmap_mode="r")

        metrics = compare_arrays(original, rectangular)
        metrics["subject"] = f"sub-{subject_num:02d}"
        rows.append(metrics)

        print(
            f"{metrics['subject']} | "
            f"shape_match={metrics['shape_match']} | "
            f"relative_L2="
            f"{metrics.get('relative_l2_difference', np.nan):.6g} | "
            f"correlation="
            f"{metrics.get('value_correlation', np.nan):.6f} | "
            f"median_ratio="
            f"{metrics.get('ratio_median', np.nan):.6g}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {OUTPUT_FILE}"
        )

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved diagnostic report: {OUTPUT_FILE}")
    print("\nInterpretation:")
    print("- Correlation near 1 and a stable ratio suggest a scaling difference.")
    print("- Large relative differences suggest the feature pipelines differ.")
    print("- This diagnostic does not evaluate or use test-set labels.")


if __name__ == "__main__":
    main()
