"""
STEP 32 — TRUE TEMPORAL FFT FOR ALL SUBJECTS

Purpose
-------
Apply a real temporal FFT to the preprocessed EEG trials generated
in Step 31.

Input per subject:
    (720 trials, 64 EEG channels, 6142 time samples)

FFT is performed ONLY along the temporal axis:
    axis=2

Outputs per subject:
    - FFT magnitude
    - FFT power
    - frequency vector
    - metadata

Subjects are processed one at a time to control RAM usage.
"""

from pathlib import Path
import gc
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "features"


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_SAMPLES = 6142

SAMPLING_RATE = 2048.0

# EEG frequency range retained later for feature extraction
EEG_MIN_FREQ = 1.0
EEG_MAX_FREQ = 40.0


# ============================================================
# HELPER
# ============================================================

def print_separator():
    print("\n" + "=" * 90)


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):
    print_separator()
    print(f"{subject} — TEMPORAL FFT")
    print_separator()

    input_dir = TEMPORAL_DIR / f"temporal_eeg_{subject}"

    input_file = (
        input_dir /
        f"{subject}_temporal_eeg_preprocessed.npy"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Temporal EEG file not found:\n{input_file}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = OUTPUT_ROOT / f"temporal_fft_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    magnitude_file = (
        output_dir /
        f"{subject}_fft_magnitude.npy"
    )

    power_file = (
        output_dir /
        f"{subject}_fft_power.npy"
    )

    frequency_file = (
        output_dir /
        f"{subject}_fft_frequencies.npy"
    )

    metadata_file = (
        output_dir /
        "fft_metadata.csv"
    )

    # --------------------------------------------------------
    # Load temporal EEG
    # --------------------------------------------------------

    print("\nLoading temporal EEG...")

    X = np.load(input_file, mmap_mode="r")

    print(f"Input shape : {X.shape}")
    print(f"Input dtype : {X.dtype}")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {subject}.\n"
            f"Expected: {expected_shape}\n"
            f"Found   : {X.shape}"
        )

    if X.dtype != np.float32:
        raise ValueError(
            f"Expected float32 input, found {X.dtype}"
        )

    print("✓ Input shape verified.")
    print("✓ Input dtype verified.")

    # --------------------------------------------------------
    # Check sampled data
    # --------------------------------------------------------

    print("\nChecking input for NaN / Inf...")

    sample = np.asarray(X[:10], dtype=np.float32)

    if not np.isfinite(sample).all():
        raise ValueError(
            f"{subject}: NaN or Inf detected in temporal EEG."
        )

    print("✓ Sampled temporal data contains no NaN or Inf.")

    # --------------------------------------------------------
    # Frequency vector
    # --------------------------------------------------------

    print("\nComputing frequency vector...")

    frequencies = np.fft.rfftfreq(
        EXPECTED_SAMPLES,
        d=1.0 / SAMPLING_RATE
    )

    print(f"Frequency bins : {len(frequencies)}")
    print(f"Minimum freq   : {frequencies[0]:.6f} Hz")
    print(f"Maximum freq   : {frequencies[-1]:.6f} Hz")
    print(
        f"Resolution     : "
        f"{frequencies[1] - frequencies[0]:.6f} Hz"
    )

    # --------------------------------------------------------
    # Validate frequency vector
    # --------------------------------------------------------

    expected_fft_bins = EXPECTED_SAMPLES // 2 + 1

    if len(frequencies) != expected_fft_bins:
        raise ValueError(
            f"Unexpected number of FFT bins.\n"
            f"Expected: {expected_fft_bins}\n"
            f"Found   : {len(frequencies)}"
        )

    print("✓ Frequency vector verified.")

    # --------------------------------------------------------
    # Save frequency vector
    # --------------------------------------------------------

    np.save(frequency_file, frequencies)

    print(f"✓ Saved frequencies:")
    print(f"  {frequency_file}")

    # --------------------------------------------------------
    # Create output arrays on disk
    # --------------------------------------------------------

    fft_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        expected_fft_bins,
    )

    print("\nFFT output shape:")
    print(f"  {fft_shape}")

    print("\nCreating disk-backed FFT arrays...")

    magnitude = np.lib.format.open_memmap(
        magnitude_file,
        mode="w+",
        dtype=np.float32,
        shape=fft_shape,
    )

    power = np.lib.format.open_memmap(
        power_file,
        mode="w+",
        dtype=np.float32,
        shape=fft_shape,
    )

    # --------------------------------------------------------
    # Process trials in chunks
    # --------------------------------------------------------

    TRIAL_CHUNK_SIZE = 32

    print("\nPerforming temporal FFT...")
    print(f"Trial chunk size: {TRIAL_CHUNK_SIZE}")

    for start in range(
        0,
        EXPECTED_TRIALS,
        TRIAL_CHUNK_SIZE
    ):

        end = min(
            start + TRIAL_CHUNK_SIZE,
            EXPECTED_TRIALS
        )

        print(
            f"\nTrials {start + 1}-{end}/{EXPECTED_TRIALS}"
        )

        # Load only a small block
        chunk = np.asarray(
            X[start:end],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # TRUE TEMPORAL FFT
        #
        # axis=2 = time
        # ----------------------------------------------------

        fft_result = np.fft.rfft(
            chunk,
            axis=2
        )

        # Magnitude
        fft_magnitude = np.abs(
            fft_result
        ).astype(np.float32)

        # Power
        fft_power = (
            fft_magnitude.astype(np.float32) ** 2
        ).astype(np.float32)

        # ----------------------------------------------------
        # Validate chunk
        # ----------------------------------------------------

        if not np.isfinite(fft_magnitude).all():
            raise ValueError(
                f"{subject}: NaN/Inf detected in FFT magnitude."
            )

        if not np.isfinite(fft_power).all():
            raise ValueError(
                f"{subject}: NaN/Inf detected in FFT power."
            )

        # ----------------------------------------------------
        # Write directly to disk
        # ----------------------------------------------------

        magnitude[start:end] = fft_magnitude
        power[start:end] = fft_power

        magnitude.flush()
        power.flush()

        print("  ✓ FFT completed")
        print(f"  ✓ Output shape: {fft_magnitude.shape}")

        # Release temporary arrays
        del chunk
        del fft_result
        del fft_magnitude
        del fft_power

        gc.collect()

    # --------------------------------------------------------
    # Flush output
    # --------------------------------------------------------

    magnitude.flush()
    power.flush()

    del magnitude
    del power
    del X

    gc.collect()

    # --------------------------------------------------------
    # Re-open and verify
    # --------------------------------------------------------

    print("\nRe-opening FFT outputs for validation...")

    magnitude_check = np.load(
        magnitude_file,
        mmap_mode="r"
    )

    power_check = np.load(
        power_file,
        mmap_mode="r"
    )

    frequency_check = np.load(
        frequency_file,
        mmap_mode="r"
    )

    print(f"Magnitude shape : {magnitude_check.shape}")
    print(f"Power shape     : {power_check.shape}")
    print(f"Frequency shape : {frequency_check.shape}")

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if magnitude_check.shape != fft_shape:
        raise ValueError(
            "FFT magnitude shape validation failed."
        )

    if power_check.shape != fft_shape:
        raise ValueError(
            "FFT power shape validation failed."
        )

    if frequency_check.shape != (expected_fft_bins,):
        raise ValueError(
            "Frequency shape validation failed."
        )

    print("✓ FFT output shapes verified.")

    # --------------------------------------------------------
    # Sample validation
    # --------------------------------------------------------

    print("\nChecking FFT outputs for NaN / Inf...")

    magnitude_sample = np.asarray(
        magnitude_check[:10],
        dtype=np.float32
    )

    power_sample = np.asarray(
        power_check[:10],
        dtype=np.float32
    )

    if not np.isfinite(magnitude_sample).all():
        raise ValueError(
            "NaN/Inf detected in magnitude output."
        )

    if not np.isfinite(power_sample).all():
        raise ValueError(
            "NaN/Inf detected in power output."
        )

    print("✓ FFT magnitude contains no NaN or Inf.")
    print("✓ FFT power contains no NaN or Inf.")

    # --------------------------------------------------------
    # Frequency range check
    # --------------------------------------------------------

    selected_frequency_mask = (
        (frequency_check >= EEG_MIN_FREQ)
        &
        (frequency_check <= EEG_MAX_FREQ)
    )

    selected_frequencies = (
        frequency_check[selected_frequency_mask]
    )

    print("\nEEG frequency range check:")
    print(
        f"Selected range : "
        f"{EEG_MIN_FREQ:.1f}–{EEG_MAX_FREQ:.1f} Hz"
    )
    print(
        f"Selected bins  : "
        f"{len(selected_frequencies)}"
    )

    if len(selected_frequencies) == 0:
        raise ValueError(
            "No FFT frequency bins found in 1–40 Hz."
        )

    print(
        f"First selected : "
        f"{selected_frequencies[0]:.6f} Hz"
    )

    print(
        f"Last selected  : "
        f"{selected_frequencies[-1]:.6f} Hz"
    )

    print("✓ 1–40 Hz frequency range available.")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "subject": [subject],
            "trials": [EXPECTED_TRIALS],
            "channels": [EXPECTED_CHANNELS],
            "samples_per_trial": [EXPECTED_SAMPLES],
            "sampling_rate_hz": [SAMPLING_RATE],
            "fft_bins": [expected_fft_bins],
            "frequency_resolution_hz": [
                frequencies[1] - frequencies[0]
            ],
            "min_frequency_hz": [
                frequencies[0]
            ],
            "max_frequency_hz": [
                frequencies[-1]
            ],
            "selected_min_frequency_hz": [
                selected_frequencies[0]
            ],
            "selected_max_frequency_hz": [
                selected_frequencies[-1]
            ],
            "selected_frequency_bins": [
                len(selected_frequencies)
            ],
        }
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    print(f"\n✓ Saved metadata:")
    print(f"  {metadata_file}")

    # --------------------------------------------------------
    # Final subject summary
    # --------------------------------------------------------

    print_separator()
    print(f"{subject} — FFT COMPLETE")
    print_separator()

    print(f"Trials              : {EXPECTED_TRIALS}")
    print(f"Channels            : {EXPECTED_CHANNELS}")
    print(f"Time samples        : {EXPECTED_SAMPLES}")
    print(f"FFT bins            : {expected_fft_bins}")
    print(
        f"Frequency resolution: "
        f"{frequencies[1] - frequencies[0]:.6f} Hz"
    )
    print(
        f"1–40 Hz bins        : "
        f"{len(selected_frequencies)}"
    )

    print("\nOutputs:")
    print(f"  Magnitude: {magnitude_file}")
    print(f"  Power    : {power_file}")
    print(f"  Frequency: {frequency_file}")
    print(f"  Metadata : {metadata_file}")

    print("\n✓ Subject FFT dataset verified.")

    # Close memory maps
    del magnitude_check
    del power_check
    del frequency_check

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("STEP 32 — TRUE TEMPORAL FFT")
    print("ALL 12 SUBJECTS")
    print_separator()

    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(
        f"Input shape per subject: "
        f"({EXPECTED_TRIALS}, "
        f"{EXPECTED_CHANNELS}, "
        f"{EXPECTED_SAMPLES})"
    )

    completed = []
    failed = []

    for subject in SUBJECTS:

        try:

            process_subject(subject)

            completed.append(subject)

        except Exception as exc:

            print("\n" + "!" * 90)
            print(f"FAILED: {subject}")
            print("!" * 90)

            print(
                f"{type(exc).__name__}: {exc}"
            )

            failed.append(subject)

            # Stop immediately so we don't silently
            # continue with an inconsistent dataset.
            raise

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_separator()
    print("STEP 32 — FINAL SUMMARY")
    print_separator()

    print(
        f"Subjects requested : {len(SUBJECTS)}"
    )

    print(
        f"Subjects completed : {len(completed)}"
    )

    print(
        f"Subjects failed    : {len(failed)}"
    )

    if failed:

        print("\nFAILED:")
        for subject in failed:
            print(f"  ✗ {subject}")

    else:

        for subject in completed:
            print(f"  ✓ {subject}")

        print("\n" + "=" * 90)
        print("✓ ALL 12 SUBJECTS COMPLETED SUCCESSFULLY")
        print("=" * 90)

        print(
            "\nNext step: "
            "1–40 Hz temporal spectral feature extraction."
        )


if __name__ == "__main__":
    main()