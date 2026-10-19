
"""
EEG-SP-01 — Extract five-band features from Hann-windowed FFT power.

Input:
    data/features/hann_fft_sub-XX/
        sub-XX_hann_fft_power.npy
        sub-XX_hann_fft_frequencies.npy

Output:
    data/features/hann_spectral_sub-XX/
        sub-XX_hann_band_power.npy
        sub-XX_hann_spectral_metadata.csv

Output shape per subject: (720, 64, 5)
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
CHUNK_SIZE = 32

# Half-open bands prevent assigning a shared boundary frequency
# to two adjacent bands. The final band includes its upper limit.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def process_subject(subject):
    print("\n" + "=" * 75)
    print(f"EEG-SP-01 — Hann band features: {subject}")
    print("=" * 75)

    input_dir = FEATURES_ROOT / f"hann_fft_{subject}"
    power_file = input_dir / f"{subject}_hann_fft_power.npy"
    frequency_file = (
        input_dir / f"{subject}_hann_fft_frequencies.npy"
    )

    if not power_file.exists() or not frequency_file.exists():
        raise FileNotFoundError(
            f"Missing Hann FFT input for {subject}: {input_dir}"
        )

    power = np.load(power_file, mmap_mode="r")
    frequencies = np.load(frequency_file, mmap_mode="r")

    if power.shape[:2] != (EXPECTED_TRIALS, EXPECTED_CHANNELS):
        raise ValueError(
            f"{subject}: unexpected FFT shape {power.shape}"
        )

    if power.shape[2] != len(frequencies):
        raise ValueError(
            f"{subject}: frequency vector does not match FFT bins"
        )

    if not np.isfinite(frequencies).all():
        raise ValueError(f"{subject}: invalid frequency vector")

    output_dir = FEATURES_ROOT / f"hann_spectral_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_file = output_dir / f"{subject}_hann_band_power.npy"
    metadata_file = (
        output_dir / f"{subject}_hann_spectral_metadata.csv"
    )

    for path in (feature_file, metadata_file):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Rename the previous output before rerunning."
            )

    feature_shape = (EXPECTED_TRIALS, EXPECTED_CHANNELS, len(BANDS))

    features = np.lib.format.open_memmap(
        feature_file,
        mode="w+",
        dtype=np.float32,
        shape=feature_shape,
    )

    try:
        band_masks = []
        for band_name, (low, high) in BANDS.items():
            if band_name == "gamma":
                mask = (frequencies >= low) & (frequencies <= high)
            else:
                mask = (frequencies >= low) & (frequencies < high)

            if not np.any(mask):
                raise ValueError(
                    f"{subject}: no frequency bins for {band_name}"
                )

            band_masks.append(mask)
            print(
                f"{band_name}: {low:g}–{high:g} Hz, "
                f"{int(mask.sum())} bins"
            )

        for start in range(0, EXPECTED_TRIALS, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, EXPECTED_TRIALS)
            print(f"Trials {start + 1}-{end}/{EXPECTED_TRIALS}")

            for band_index, mask in enumerate(band_masks):
                band_power = np.asarray(
                    power[start:end, :, mask],
                    dtype=np.float32,
                )

                features[start:end, :, band_index] = np.mean(
                    band_power, axis=2, dtype=np.float64
                ).astype(np.float32)

                del band_power

            features.flush()

        if not np.isfinite(np.asarray(features[:10])).all():
            raise ValueError(
                f"{subject}: non-finite extracted features"
            )

    finally:
        del features
        gc.collect()

    # Reopen and verify saved output.
    saved = np.load(feature_file, mmap_mode="r")

    if saved.shape != feature_shape:
        raise ValueError(
            f"{subject}: expected {feature_shape}, got {saved.shape}"
        )

    if not np.isfinite(np.asarray(saved[:10])).all():
        raise ValueError(f"{subject}: invalid saved feature values")

    metadata = pd.DataFrame([{
        "subject": subject,
        "trials": EXPECTED_TRIALS,
        "channels": EXPECTED_CHANNELS,
        "bands": len(BANDS),
        "features_per_trial": EXPECTED_CHANNELS * len(BANDS),
        "output_shape": str(feature_shape),
        "window": "symmetric Hann",
        "power_normalization": "squared magnitude / sum(window**2)",
        "band_boundary_policy": "lower-inclusive, upper-exclusive; gamma upper-inclusive",
    }])

    metadata.to_csv(metadata_file, index=False)

    print(f"Feature shape: {saved.shape}")
    print(f"Saved: {feature_file}")
    print("Feature validation passed.")

    del saved, power, frequencies
    gc.collect()


def main():
    print("EEG-SP-01 — HANN-WINDOWED FIVE-BAND FEATURES")

    for subject in SUBJECTS:
        process_subject(subject)

    print("\nAll 12 subjects completed successfully.")


if __name__ == "__main__":
    main()
