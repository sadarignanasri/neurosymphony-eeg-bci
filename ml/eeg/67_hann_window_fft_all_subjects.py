
"""
EEG-SP-01 — HANN-WINDOWED TEMPORAL FFT

Experimental pipeline for NeuroSymphony.
Does not modify the original Step 32 baseline.

Input per subject:
    data/features/temporal_eeg_sub-XX/
        sub-XX_temporal_eeg_preprocessed.npy

Input shape:
    (720 trials, 64 EEG channels, 6142 time samples)

Output:
    data/features/hann_fft_sub-XX/
        sub-XX_hann_fft_power.npy
        sub-XX_hann_fft_frequencies.npy
        hann_fft_metadata.csv

The Hann window is applied only along the temporal axis.
Power is divided by the squared sum of the window to make the
power scaling comparable in definition to the unwindowed FFT.
This remains raw window-normalized spectral power, not PSD.
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
EXPECTED_SAMPLES = 6142
SAMPLING_RATE = 2048.0
CHUNK_SIZE = 16


def process_subject(subject):
    print("\n" + "=" * 75)
    print(f"EEG-SP-01: Hann-windowed FFT — {subject}")
    print("=" * 75)

    input_file = (
        FEATURES_ROOT
        / f"temporal_eeg_{subject}"
        / f"{subject}_temporal_eeg_preprocessed.npy"
    )

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    X = np.load(input_file, mmap_mode="r")

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"{subject}: expected {expected_shape}, found {X.shape}"
        )

    if X.dtype != np.float32:
        raise ValueError(
            f"{subject}: expected float32, found {X.dtype}"
        )

    # Symmetric Hann window; identical window for every trial/channel.
    window = np.hanning(EXPECTED_SAMPLES).astype(np.float32)

    # Window-energy normalization for the squared FFT magnitude.
    window_energy = float(np.sum(window.astype(np.float64) ** 2))

    if not np.isfinite(window_energy) or window_energy <= 0:
        raise ValueError("Invalid Hann window energy.")

    frequencies = np.fft.rfftfreq(
        EXPECTED_SAMPLES,
        d=1.0 / SAMPLING_RATE,
    )

    output_dir = FEATURES_ROOT / f"hann_fft_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    power_file = output_dir / f"{subject}_hann_fft_power.npy"
    frequency_file = (
        output_dir / f"{subject}_hann_fft_frequencies.npy"
    )
    metadata_file = output_dir / "hann_fft_metadata.csv"

    # Refuse to overwrite an earlier experiment.
    for path in (power_file, frequency_file, metadata_file):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Move or rename the previous experimental output "
                "before rerunning."
            )

    fft_bins = len(frequencies)
    output = np.lib.format.open_memmap(
        power_file,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_TRIALS, EXPECTED_CHANNELS, fft_bins),
    )

    try:
        for start in range(0, EXPECTED_TRIALS, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, EXPECTED_TRIALS)
            print(f"{subject}: trials {start + 1}-{end}/{EXPECTED_TRIALS}")

            chunk = np.asarray(X[start:end], dtype=np.float32)

            if not np.isfinite(chunk).all():
                raise ValueError(
                    f"{subject}: NaN/Inf in input trials {start}:{end}"
                )

            # Apply Hann window along time only (axis=2).
            windowed = chunk * window[None, None, :]

            fft_result = np.fft.rfft(windowed, axis=2)

            # Normalize squared magnitude by window energy.
            # The same convention must be used for the baseline
            # if absolute band-power scales are directly compared.
            power = (
                np.abs(fft_result) ** 2 / window_energy
            ).astype(np.float32)

            if not np.isfinite(power).all():
                raise ValueError(
                    f"{subject}: NaN/Inf in FFT output."
                )

            output[start:end] = power
            output.flush()

            del chunk, windowed, fft_result, power
            gc.collect()

        np.save(frequency_file, frequencies)

        metadata = pd.DataFrame([{
            "subject": subject,
            "window": "symmetric Hann",
            "trials": EXPECTED_TRIALS,
            "channels": EXPECTED_CHANNELS,
            "samples_per_trial": EXPECTED_SAMPLES,
            "sampling_rate_hz": SAMPLING_RATE,
            "fft_bins": fft_bins,
            "frequency_resolution_hz": float(
                frequencies[1] - frequencies[0]
            ),
            "window_energy": window_energy,
            "power_normalization": "squared magnitude / sum(window**2)",
            "power_is_psd": False,
            "chunk_size": CHUNK_SIZE,
        }])

        metadata.to_csv(metadata_file, index=False)
        output.flush()

    finally:
        del output
        gc.collect()

    # Reopen outputs and validate them.
    saved_power = np.load(power_file, mmap_mode="r")
    saved_freq = np.load(frequency_file, mmap_mode="r")

    if saved_power.shape != (
        EXPECTED_TRIALS, EXPECTED_CHANNELS, fft_bins
    ):
        raise ValueError(f"{subject}: saved power shape is incorrect.")

    if saved_freq.shape != (fft_bins,):
        raise ValueError(f"{subject}: saved frequency shape is incorrect.")

    if not np.isfinite(np.asarray(saved_power[:2])).all():
        raise ValueError(f"{subject}: invalid saved power values.")

    print(f"Power shape: {saved_power.shape}")
    print(f"Frequency bins: {len(saved_freq)}")
    print(f"Saved: {power_file}")
    print("Validation passed.")

    del saved_power, saved_freq, X
    gc.collect()


def main():
    print("EEG-SP-01 — HANN-WINDOWED FFT")
    print(f"Subjects: {len(SUBJECTS)}")
    print("Original Step 32 outputs will not be modified.")

    for subject in SUBJECTS:
        process_subject(subject)

    print("\nEEG-SP-01 FFT extraction completed for all subjects.")


if __name__ == "__main__":
    main()
