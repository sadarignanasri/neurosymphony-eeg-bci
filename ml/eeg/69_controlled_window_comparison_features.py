
"""
EEG-SP-01 — CONTROLLED WINDOW COMPARISON
NeuroSymphony

Compares rectangular and symmetric Hann windows under the
same window-energy-normalized spectral power convention.

Original Steps 32, 33, 67 and 68 remain untouched.

Outputs are saved in separate experiment directories.
Existing output files are never overwritten.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]

N_TRIALS = 720
N_CHANNELS = 64
N_SAMPLES = 6142
FS = 2048.0
CHUNK_SIZE = 16

MIN_FREQ = 1.0
MAX_FREQ = 40.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def get_band_masks(frequencies):
    """Use [low, high) bands; include 40 Hz in gamma."""
    masks = {}

    for name, (low, high) in BANDS.items():
        if name == "gamma":
            mask = (frequencies >= low) & (frequencies <= high)
        else:
            mask = (frequencies >= low) & (frequencies < high)

        if not np.any(mask):
            raise ValueError(f"Frequency band {name} has no bins.")

        masks[name] = mask

    return masks


def process_subject(subject):
    print("\n" + "=" * 75)
    print(f"Controlled window comparison: {subject}")
    print("=" * 75)

    input_file = (
        FEATURES_ROOT
        / f"temporal_eeg_{subject}"
        / f"{subject}_temporal_eeg_preprocessed.npy"
    )

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    X = np.load(input_file, mmap_mode="r")

    expected_shape = (N_TRIALS, N_CHANNELS, N_SAMPLES)
    if X.shape != expected_shape:
        raise ValueError(
            f"{subject}: expected {expected_shape}, found {X.shape}"
        )

    if X.dtype != np.float32:
        raise ValueError(
            f"{subject}: expected float32, found {X.dtype}"
        )

    frequencies = np.fft.rfftfreq(N_SAMPLES, d=1.0 / FS)
    frequency_mask = (
        (frequencies >= MIN_FREQ) & (frequencies <= MAX_FREQ)
    )
    selected_frequencies = frequencies[frequency_mask]
    band_masks = get_band_masks(selected_frequencies)

    # Rectangular window has energy N; Hann has its own energy.
    windows = {
        "rectangular": np.ones(N_SAMPLES, dtype=np.float32),
        "hann": np.hanning(N_SAMPLES).astype(np.float32),
    }
    energies = {
        name: float(np.sum(window.astype(np.float64) ** 2))
        for name, window in windows.items()
    }

    output_dirs = {
        "rectangular": FEATURES_ROOT / f"controlled_rect_spectral_{subject}",
        "hann": FEATURES_ROOT / f"controlled_hann_spectral_{subject}",
    }

    output_files = {}
    for name, output_dir in output_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_file = output_dir / f"{subject}_{name}_band_power.npy"
        metadata_file = output_dir / f"{subject}_{name}_metadata.csv"

        for path in (feature_file, metadata_file):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {path}"
                )

        output_files[name] = (feature_file, metadata_file)

    output_maps = {}
    for name, (feature_file, _) in output_files.items():
        output_maps[name] = np.lib.format.open_memmap(
            feature_file,
            mode="w+",
            dtype=np.float32,
            shape=(N_TRIALS, N_CHANNELS, len(BANDS)),
        )

    try:
        for start in range(0, N_TRIALS, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N_TRIALS)
            print(f"{subject}: trials {start + 1}-{end}/{N_TRIALS}")

            chunk = np.asarray(X[start:end], dtype=np.float32)

            if not np.isfinite(chunk).all():
                raise ValueError(
                    f"{subject}: non-finite input in trials {start}:{end}"
                )

            for name, window in windows.items():
                windowed = chunk * window[None, None, :]
                fft_result = np.fft.rfft(windowed, axis=2)

                # Common estimator convention:
                # squared FFT magnitude / sum(window**2).
                power = (
                    np.abs(fft_result) ** 2 / energies[name]
                ).astype(np.float32)

                selected_power = power[:, :, frequency_mask]
                band_features = []

                for band_name in BANDS:
                    band_mean = np.mean(
                        selected_power[:, :, band_masks[band_name]],
                        axis=2,
                    )
                    band_features.append(band_mean)

                features = np.stack(band_features, axis=2).astype(
                    np.float32
                )

                if features.shape != (
                    end - start, N_CHANNELS, len(BANDS)
                ):
                    raise ValueError(
                        f"{subject}/{name}: unexpected feature shape "
                        f"{features.shape}"
                    )

                if not np.isfinite(features).all():
                    raise ValueError(
                        f"{subject}/{name}: non-finite features"
                    )

                output_maps[name][start:end] = features
                output_maps[name].flush()

                del windowed, fft_result, power, selected_power
                del band_features, features
                gc.collect()

            del chunk
            gc.collect()

    finally:
        for output in output_maps.values():
            output.flush()
        output_maps.clear()
        gc.collect()

    # Write metadata only after feature extraction has completed.
    for name, (_, metadata_file) in output_files.items():
        metadata = pd.DataFrame([{
            "subject": subject,
            "window": (
                "symmetric Hann" if name == "hann"
                else "rectangular"
            ),
            "trials": N_TRIALS,
            "channels": N_CHANNELS,
            "samples_per_trial": N_SAMPLES,
            "sampling_rate_hz": FS,
            "frequency_min_hz": MIN_FREQ,
            "frequency_max_hz": MAX_FREQ,
            "frequency_bins_selected": len(selected_frequencies),
            "power_normalization": "squared FFT magnitude / sum(window**2)",
            "window_energy": energies[name],
            "power_is_psd": False,
            "band_boundary_rule": (
                "[low, high), except final gamma band includes 40 Hz"
            ),
            "feature_order": "delta, theta, alpha, beta, gamma",
        }])
        metadata.to_csv(metadata_file, index=False)

    # Reopen and validate the saved feature arrays.
    for name, (feature_file, _) in output_files.items():
        saved = np.load(feature_file, mmap_mode="r")

        if saved.shape != (N_TRIALS, N_CHANNELS, len(BANDS)):
            raise ValueError(
                f"{subject}/{name}: saved shape mismatch {saved.shape}"
            )

        if not np.isfinite(np.asarray(saved[:2])).all():
            raise ValueError(
                f"{subject}/{name}: saved features contain NaN/Inf"
            )

        print(
            f"{name}: shape={saved.shape}, "
            f"mean(first 2 trials)="
            f"{np.asarray(saved[:2]).mean():.8e}"
        )
        del saved

    del X
    gc.collect()
    print(f"✓ {subject}: both controlled feature sets validated.")


def main():
    print("EEG-SP-01 — CONTROLLED WINDOW COMPARISON")
    print(f"Subjects: {len(SUBJECTS)}")
    print("Original baseline files will not be modified.")

    for subject in SUBJECTS:
        process_subject(subject)

    print("\n✓ All 12 subjects completed successfully.")


if __name__ == "__main__":
    main()
