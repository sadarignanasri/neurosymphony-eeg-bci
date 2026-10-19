"""
STEP 33 — TEMPORAL SPECTRAL FEATURE EXTRACTION
ALL 12 SUBJECTS

Input:
    Step 32 FFT outputs

    FFT magnitude:
        (720, 64, 3072)

    FFT power:
        (720, 64, 3072)

Output:
    1–40 Hz spectral magnitude:
        (720, 64, 117)

    1–40 Hz spectral power:
        (720, 64, 117)

    Band power:
        (720, 64, 5)

Bands:
    Delta : 1–4 Hz
    Theta : 4–8 Hz
    Alpha : 8–13 Hz
    Beta  : 13–30 Hz
    Gamma : 30–40 Hz

Subjects are processed one at a time.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "data" / "features"


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_FFT_BINS = 3072

MIN_FREQ = 1.0
MAX_FREQ = 40.0

TRIAL_CHUNK_SIZE = 32


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


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
    print(f"{subject} — TEMPORAL SPECTRAL FEATURE EXTRACTION")
    print_separator()

    # --------------------------------------------------------
    # Input directory
    # --------------------------------------------------------

    input_dir = FEATURE_ROOT / f"temporal_fft_{subject}"

    magnitude_file = (
        input_dir /
        f"{subject}_fft_magnitude.npy"
    )

    power_file = (
        input_dir /
        f"{subject}_fft_power.npy"
    )

    frequency_file = (
        input_dir /
        f"{subject}_fft_frequencies.npy"
    )

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    for file_path in [
        magnitude_file,
        power_file,
        frequency_file,
    ]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required Step 32 file not found:\n"
                f"{file_path}"
            )

    print("\n✓ All Step 32 input files found.")

    # --------------------------------------------------------
    # Load frequency vector
    # --------------------------------------------------------

    frequencies = np.load(
        frequency_file,
        mmap_mode="r"
    )

    print("\nFrequency vector:")
    print(f"  Shape: {frequencies.shape}")
    print(f"  First: {frequencies[0]:.6f} Hz")
    print(f"  Last : {frequencies[-1]:.6f} Hz")

    if frequencies.shape != (EXPECTED_FFT_BINS,):

        raise ValueError(
            f"Unexpected frequency shape.\n"
            f"Expected: ({EXPECTED_FFT_BINS},)\n"
            f"Found: {frequencies.shape}"
        )

    # --------------------------------------------------------
    # Create 1–40 Hz mask
    # --------------------------------------------------------

    frequency_mask = (
        (frequencies >= MIN_FREQ)
        &
        (frequencies <= MAX_FREQ)
    )

    selected_frequencies = (
        np.asarray(frequencies)[frequency_mask]
    )

    n_selected = len(selected_frequencies)

    print("\nSelected EEG frequency range:")
    print(f"  Range: {MIN_FREQ}–{MAX_FREQ} Hz")
    print(f"  Bins : {n_selected}")
    print(
        f"  First: "
        f"{selected_frequencies[0]:.6f} Hz"
    )
    print(
        f"  Last : "
        f"{selected_frequencies[-1]:.6f} Hz"
    )

    if n_selected != 117:

        raise ValueError(
            f"Expected 117 frequency bins in 1–40 Hz, "
            f"found {n_selected}."
        )

    print("✓ 1–40 Hz frequency selection verified.")

    # --------------------------------------------------------
    # Load FFT arrays as memory maps
    # --------------------------------------------------------

    print("\nLoading FFT arrays...")

    fft_magnitude = np.load(
        magnitude_file,
        mmap_mode="r"
    )

    fft_power = np.load(
        power_file,
        mmap_mode="r"
    )

    print(
        f"Magnitude shape: "
        f"{fft_magnitude.shape}"
    )

    print(
        f"Power shape    : "
        f"{fft_power.shape}"
    )

    expected_fft_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        EXPECTED_FFT_BINS,
    )

    if fft_magnitude.shape != expected_fft_shape:

        raise ValueError(
            f"Unexpected magnitude shape.\n"
            f"Expected: {expected_fft_shape}\n"
            f"Found: {fft_magnitude.shape}"
        )

    if fft_power.shape != expected_fft_shape:

        raise ValueError(
            f"Unexpected power shape.\n"
            f"Expected: {expected_fft_shape}\n"
            f"Found: {fft_power.shape}"
        )

    print("✓ FFT array shapes verified.")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        FEATURE_ROOT /
        f"temporal_spectral_{subject}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    spectral_magnitude_file = (
        output_dir /
        f"{subject}_spectral_magnitude_1_40hz.npy"
    )

    spectral_power_file = (
        output_dir /
        f"{subject}_spectral_power_1_40hz.npy"
    )

    band_power_file = (
        output_dir /
        f"{subject}_band_power.npy"
    )

    frequency_output_file = (
        output_dir /
        f"{subject}_selected_frequencies_1_40hz.npy"
    )

    metadata_file = (
        output_dir /
        "spectral_feature_metadata.csv"
    )

    # --------------------------------------------------------
    # Output shapes
    # --------------------------------------------------------

    spectral_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        n_selected,
    )

    band_shape = (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        len(BANDS),
    )

    print("\nOutput shapes:")
    print(
        f"  Spectral magnitude: "
        f"{spectral_shape}"
    )
    print(
        f"  Spectral power    : "
        f"{spectral_shape}"
    )
    print(
        f"  Band power        : "
        f"{band_shape}"
    )

    # --------------------------------------------------------
    # Create disk-backed outputs
    # --------------------------------------------------------

    spectral_magnitude = np.lib.format.open_memmap(
        spectral_magnitude_file,
        mode="w+",
        dtype=np.float32,
        shape=spectral_shape,
    )

    spectral_power = np.lib.format.open_memmap(
        spectral_power_file,
        mode="w+",
        dtype=np.float32,
        shape=spectral_shape,
    )

    band_power = np.lib.format.open_memmap(
        band_power_file,
        mode="w+",
        dtype=np.float32,
        shape=band_shape,
    )

    # Save selected frequencies
    np.save(
        frequency_output_file,
        selected_frequencies
    )

    print("\n✓ Output arrays created.")

    # --------------------------------------------------------
    # Selected frequency indices
    # --------------------------------------------------------

    selected_indices = np.where(
        frequency_mask
    )[0]

    # --------------------------------------------------------
    # Process trial chunks
    # --------------------------------------------------------

    print("\nExtracting spectral features...")

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
            f"\nTrials "
            f"{start + 1}-{end}/{EXPECTED_TRIALS}"
        )

        # ----------------------------------------------------
        # Load small FFT blocks
        # ----------------------------------------------------

        magnitude_chunk = np.asarray(
            fft_magnitude[
                start:end,
                :,
                selected_indices
            ],
            dtype=np.float32
        )

        power_chunk = np.asarray(
            fft_power[
                start:end,
                :,
                selected_indices
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate input chunk
        # ----------------------------------------------------

        if not np.isfinite(
            magnitude_chunk
        ).all():

            raise ValueError(
                f"{subject}: NaN/Inf in "
                f"spectral magnitude chunk."
            )

        if not np.isfinite(
            power_chunk
        ).all():

            raise ValueError(
                f"{subject}: NaN/Inf in "
                f"spectral power chunk."
            )

        # ----------------------------------------------------
        # Store spectral features
        # ----------------------------------------------------

        spectral_magnitude[
            start:end
        ] = magnitude_chunk

        spectral_power[
            start:end
        ] = power_chunk

        # ----------------------------------------------------
        # Calculate band power
        # ----------------------------------------------------

        band_results = []

        for band_name, (
            band_low,
            band_high
        ) in BANDS.items():

            band_mask = (
                (selected_frequencies >= band_low)
                &
                (selected_frequencies <= band_high)
            )

            band_values = power_chunk[
                :,
                :,
                band_mask
            ]

            # Mean spectral power across
            # frequencies belonging to the band.
            band_mean = np.mean(
                band_values,
                axis=2
            ).astype(np.float32)

            band_results.append(
                band_mean
            )

        # ----------------------------------------------------
        # Stack five bands
        # ----------------------------------------------------

        band_chunk = np.stack(
            band_results,
            axis=2
        ).astype(np.float32)

        # ----------------------------------------------------
        # Validate band features
        # ----------------------------------------------------

        if not np.isfinite(
            band_chunk
        ).all():

            raise ValueError(
                f"{subject}: NaN/Inf detected "
                f"in band-power features."
            )

        # ----------------------------------------------------
        # Write band power
        # ----------------------------------------------------

        band_power[
            start:end
        ] = band_chunk

        # ----------------------------------------------------
        # Flush to disk
        # ----------------------------------------------------

        spectral_magnitude.flush()
        spectral_power.flush()
        band_power.flush()

        print("  ✓ 1–40 Hz spectral features extracted")
        print(
            f"  ✓ Spectral shape: "
            f"{magnitude_chunk.shape}"
        )
        print(
            f"  ✓ Band-power shape: "
            f"{band_chunk.shape}"
        )

        # ----------------------------------------------------
        # Release temporary memory
        # ----------------------------------------------------

        del magnitude_chunk
        del power_chunk
        del band_chunk
        del band_results

        gc.collect()

    # --------------------------------------------------------
    # Flush and release output maps
    # --------------------------------------------------------

    spectral_magnitude.flush()
    spectral_power.flush()
    band_power.flush()

    del spectral_magnitude
    del spectral_power
    del band_power

    del fft_magnitude
    del fft_power
    del frequencies

    gc.collect()

    # --------------------------------------------------------
    # Re-open outputs for validation
    # --------------------------------------------------------

    print("\nRe-opening outputs for final validation...")

    magnitude_check = np.load(
        spectral_magnitude_file,
        mmap_mode="r"
    )

    power_check = np.load(
        spectral_power_file,
        mmap_mode="r"
    )

    band_check = np.load(
        band_power_file,
        mmap_mode="r"
    )

    selected_frequency_check = np.load(
        frequency_output_file,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if magnitude_check.shape != spectral_shape:

        raise ValueError(
            "Spectral magnitude shape validation failed."
        )

    if power_check.shape != spectral_shape:

        raise ValueError(
            "Spectral power shape validation failed."
        )

    if band_check.shape != band_shape:

        raise ValueError(
            "Band-power shape validation failed."
        )

    if selected_frequency_check.shape != (
        n_selected,
    ):

        raise ValueError(
            "Selected-frequency shape validation failed."
        )

    print("✓ Spectral magnitude shape verified.")
    print("✓ Spectral power shape verified.")
    print("✓ Band-power shape verified.")
    print("✓ Selected-frequency shape verified.")

    # --------------------------------------------------------
    # Sample validation
    # --------------------------------------------------------

    print("\nChecking final outputs for NaN / Inf...")

    magnitude_sample = np.asarray(
        magnitude_check[:10],
        dtype=np.float32
    )

    power_sample = np.asarray(
        power_check[:10],
        dtype=np.float32
    )

    band_sample = np.asarray(
        band_check[:10],
        dtype=np.float32
    )

    if not np.isfinite(
        magnitude_sample
    ).all():

        raise ValueError(
            "NaN/Inf found in final spectral magnitude."
        )

    if not np.isfinite(
        power_sample
    ).all():

        raise ValueError(
            "NaN/Inf found in final spectral power."
        )

    if not np.isfinite(
        band_sample
    ).all():

        raise ValueError(
            "NaN/Inf found in final band power."
        )

    print(
        "✓ Spectral magnitude contains no NaN or Inf."
    )

    print(
        "✓ Spectral power contains no NaN or Inf."
    )

    print(
        "✓ Band-power features contain no NaN or Inf."
    )

    # --------------------------------------------------------
    # Band statistics
    # --------------------------------------------------------

    print("\nBand-power summary:")

    # Use a small sample to avoid loading
    # the complete dataset into RAM.
    stats_sample = np.asarray(
        band_check[:32],
        dtype=np.float32
    )

    for index, band_name in enumerate(BANDS):

        values = stats_sample[
            :,
            :,
            index
        ]

        print(
            f"  {band_name:>5} "
            f"{BANDS[band_name][0]:>5.1f}–"
            f"{BANDS[band_name][1]:<5.1f} Hz : "
            f"mean={values.mean():.10f}, "
            f"std={values.std():.10f}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "subject": [subject],
            "trials": [EXPECTED_TRIALS],
            "channels": [EXPECTED_CHANNELS],
            "original_fft_bins": [
                EXPECTED_FFT_BINS
            ],
            "selected_min_frequency_hz": [
                float(selected_frequencies[0])
            ],
            "selected_max_frequency_hz": [
                float(selected_frequencies[-1])
            ],
            "selected_frequency_bins": [
                n_selected
            ],
            "delta_range_hz": ["1-4"],
            "theta_range_hz": ["4-8"],
            "alpha_range_hz": ["8-13"],
            "beta_range_hz": ["13-30"],
            "gamma_range_hz": ["30-40"],
        }
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_separator()
    print(f"{subject} — SPECTRAL FEATURES COMPLETE")
    print_separator()

    print(
        f"Trials              : "
        f"{EXPECTED_TRIALS}"
    )

    print(
        f"Channels            : "
        f"{EXPECTED_CHANNELS}"
    )

    print(
        f"1–40 Hz spectral bins: "
        f"{n_selected}"
    )

    print(
        f"Band-power features  : "
        f"{len(BANDS)}"
    )

    print("\nBands:")

    for band_name, (
        low,
        high
    ) in BANDS.items():

        print(
            f"  {band_name.capitalize():<6}: "
            f"{low:.0f}–{high:.0f} Hz"
        )

    print("\nOutputs:")

    print(
        f"  Spectral magnitude:"
        f"\n    {spectral_magnitude_file}"
    )

    print(
        f"  Spectral power:"
        f"\n    {spectral_power_file}"
    )

    print(
        f"  Band power:"
        f"\n    {band_power_file}"
    )

    print(
        f"  Frequencies:"
        f"\n    {frequency_output_file}"
    )

    print(
        f"  Metadata:"
        f"\n    {metadata_file}"
    )

    print("\n✓ Subject spectral features verified.")

    # Release validation maps
    del magnitude_check
    del power_check
    del band_check
    del selected_frequency_check

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("STEP 33 — TEMPORAL SPECTRAL FEATURE EXTRACTION")
    print("ALL 12 SUBJECTS")
    print_separator()

    print(f"Subjects: {len(SUBJECTS)}")

    print(
        f"Input per subject: "
        f"({EXPECTED_TRIALS}, "
        f"{EXPECTED_CHANNELS}, "
        f"{EXPECTED_FFT_BINS})"
    )

    print(
        f"Frequency range: "
        f"{MIN_FREQ}–{MAX_FREQ} Hz"
    )

    print(
        f"Trial chunk size: "
        f"{TRIAL_CHUNK_SIZE}"
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

            # Stop immediately.
            raise

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_separator()
    print("STEP 33 — FINAL SUMMARY")
    print_separator()

    print(
        f"Subjects requested : "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Subjects completed : "
        f"{len(completed)}"
    )

    print(
        f"Subjects failed    : "
        f"{len(failed)}"
    )

    for subject in completed:
        print(f"  ✓ {subject}")

    if failed:

        print("\nFAILED:")

        for subject in failed:
            print(f"  ✗ {subject}")

    else:

        print("\n" + "=" * 90)
        print(
            "✓ ALL 12 SUBJECTS COMPLETED SUCCESSFULLY"
        )
        print("=" * 90)

        print(
            "\nNext step: "
            "create verified train/validation/test splits."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()