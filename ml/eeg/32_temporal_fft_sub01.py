
"""
Neuro-Symphony
Step 32: True Temporal FFT Analysis — sub-01

Input:
    Preprocessed temporal EEG
    Shape: (720, 64, 6142)

FFT is applied along the TEMPORAL axis.

    6142 time samples
          ↓
        rFFT
          ↓
    frequency-domain representation

No FFT is performed across channels.
No model training.
No train/test split.

Output:
    Temporal FFT magnitude
    Temporal FFT power
    Frequency vector
    Metadata
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# INPUTS
# ============================================================================

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_eeg_sub01"
)

INPUT_FILE = (
    INPUT_DIR
    / "sub-01_temporal_eeg_preprocessed.npy"
)

METADATA_FILE = (
    INPUT_DIR
    / "sub-01_temporal_eeg_preprocessed_metadata.csv"
)

CHANNEL_FILE = (
    INPUT_DIR
    / "preprocessed_channel_names.txt"
)


# ============================================================================
# OUTPUTS
# ============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_fft_sub01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FFT_MAGNITUDE_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_fft_magnitude.npy"
)

FFT_POWER_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_fft_power.npy"
)

FREQUENCIES_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_fft_frequencies.npy"
)

METADATA_OUTPUT_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_fft_metadata.csv"
)


# ============================================================================
# PARAMETERS
# ============================================================================

SFREQ = 2048.0

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_SAMPLES = 6142


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("NEURO-SYMPHONY — TRUE TEMPORAL FFT")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================================
# VALIDATE INPUTS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATING INPUT FILES")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Temporal EEG file not found:\n{INPUT_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

if not CHANNEL_FILE.exists():
    raise FileNotFoundError(
        f"Channel file not found:\n{CHANNEL_FILE}"
    )

print("\n✓ Temporal EEG file found.")
print("✓ Metadata file found.")
print("✓ Channel file found.")


# ============================================================================
# LOAD TEMPORAL EEG
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PREPROCESSED TEMPORAL EEG")
print("=" * 80)

X = np.load(
    INPUT_FILE,
    mmap_mode="r"
)

print("\nLoaded shape:")
print(X.shape)

print(f"\nData type: {X.dtype}")


# ============================================================================
# VALIDATE SHAPE
# ============================================================================

expected_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLES
)

print("\nExpected shape:")
print(expected_shape)


if X.shape != expected_shape:
    raise RuntimeError(
        f"Unexpected temporal EEG shape: "
        f"{X.shape}. "
        f"Expected {expected_shape}."
    )


print("\n✓ Temporal EEG shape verified.")


# ============================================================================
# LOAD METADATA
# ============================================================================

metadata = pd.read_csv(
    METADATA_FILE
)

print("\nMetadata shape:")
print(metadata.shape)


if len(metadata) != EXPECTED_TRIALS:
    raise RuntimeError(
        f"Expected {EXPECTED_TRIALS} metadata rows, "
        f"found {len(metadata)}."
    )


# ============================================================================
# LOAD CHANNEL NAMES
# ============================================================================

with open(
    CHANNEL_FILE,
    "r",
    encoding="utf-8"
) as f:

    channel_names = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"\nChannel names loaded: "
    f"{len(channel_names)}"
)


if len(channel_names) != EXPECTED_CHANNELS:
    raise RuntimeError(
        f"Expected {EXPECTED_CHANNELS} channel names, "
        f"found {len(channel_names)}."
    )


# ============================================================================
# FFT INFORMATION
# ============================================================================

print("\n" + "=" * 80)
print("TEMPORAL FFT CONFIGURATION")
print("=" * 80)

print(
    f"\nSampling frequency: "
    f"{SFREQ} Hz"
)

print(
    f"Time samples per trial: "
    f"{EXPECTED_SAMPLES}"
)

print(
    f"Trial duration: "
    f"{EXPECTED_SAMPLES / SFREQ:.6f} seconds"
)


# rFFT returns:

# floor(N / 2) + 1

# frequency bins for a real-valued signal.

N = EXPECTED_SAMPLES

N_FREQS = (
    N // 2
) + 1


print(
    f"\nExpected rFFT frequency bins: "
    f"{N_FREQS}"
)


# ============================================================================
# COMPUTE FREQUENCY VECTOR
# ============================================================================

frequencies = np.fft.rfftfreq(
    N,
    d=1.0 / SFREQ
)


print("\nFrequency vector:")

print(
    f"Number of frequencies: "
    f"{len(frequencies)}"
)

print(
    f"Minimum frequency: "
    f"{frequencies[0]:.6f} Hz"
)

print(
    f"Maximum frequency: "
    f"{frequencies[-1]:.6f} Hz"
)

if len(frequencies) != N_FREQS:
    raise RuntimeError(
        "Unexpected frequency-vector length."
    )


# ============================================================================
# FREQUENCY RESOLUTION
# ============================================================================

frequency_resolution = (
    SFREQ / N
)


print(
    f"\nFrequency resolution: "
    f"{frequency_resolution:.6f} Hz"
)


# ============================================================================
# TRUE TEMPORAL FFT
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING TRUE TEMPORAL FFT")
print("=" * 80)

print(
    "\nFFT axis: time axis"
)

print(
    "Input dimensions:"
)

print(
    "  trials × channels × time"
)

print(
    f"  {EXPECTED_TRIALS} × "
    f"{EXPECTED_CHANNELS} × "
    f"{EXPECTED_SAMPLES}"
)

print(
    "\nFFT operation:"
)

print(
    "  np.fft.rfft(X, axis=2)"
)

print(
    "\nThis means the FFT is performed independently "
    "for every trial and every EEG channel."
)


# ============================================================================
# PROCESS IN TRIAL BATCHES
# ============================================================================

# We process trials in batches so that the complete complex FFT
# does not unnecessarily remain in memory.

BATCH_SIZE = 30

magnitude_batches = []
power_batches = []


for start in range(
    0,
    EXPECTED_TRIALS,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        EXPECTED_TRIALS
    )

    print(
        f"Processing trials "
        f"{start + 1}-{end}/{EXPECTED_TRIALS}"
    )


    batch = np.asarray(
        X[start:end],
        dtype=np.float32
    )


    # ------------------------------------------------------------------------
    # TRUE TEMPORAL FFT
    # ------------------------------------------------------------------------

    fft_complex = np.fft.rfft(
        batch,
        axis=2
    )


    # ------------------------------------------------------------------------
    # MAGNITUDE
    # ------------------------------------------------------------------------

    magnitude = np.abs(
        fft_complex
    )


    # ------------------------------------------------------------------------
    # POWER
    # ------------------------------------------------------------------------

    power = (
        magnitude ** 2
    )


    magnitude_batches.append(
        magnitude.astype(
            np.float32
        )
    )

    power_batches.append(
        power.astype(
            np.float32
        )
    )


# ============================================================================
# STACK FFT FEATURES
# ============================================================================

fft_magnitude = np.concatenate(
    magnitude_batches,
    axis=0
)

fft_power = np.concatenate(
    power_batches,
    axis=0
)


# ============================================================================
# VALIDATE FFT SHAPES
# ============================================================================

print("\n" + "=" * 80)
print("FFT VALIDATION")
print("=" * 80)

expected_fft_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    N_FREQS
)


print("\nExpected FFT shape:")
print(expected_fft_shape)

print("\nMagnitude shape:")
print(fft_magnitude.shape)

print("\nPower shape:")
print(fft_power.shape)


if fft_magnitude.shape != expected_fft_shape:
    raise RuntimeError(
        f"Unexpected magnitude shape: "
        f"{fft_magnitude.shape}. "
        f"Expected {expected_fft_shape}."
    )


if fft_power.shape != expected_fft_shape:
    raise RuntimeError(
        f"Unexpected power shape: "
        f"{fft_power.shape}. "
        f"Expected {expected_fft_shape}."
    )


# ============================================================================
# NUMERICAL VALIDATION
# ============================================================================

print("\nMagnitude statistics:")

print(
    f"Minimum: "
    f"{np.min(fft_magnitude):.10f}"
)

print(
    f"Maximum: "
    f"{np.max(fft_magnitude):.10f}"
)

print(
    f"Mean: "
    f"{np.mean(fft_magnitude):.10f}"
)

print(
    f"Std: "
    f"{np.std(fft_magnitude):.10f}"
)


print("\nPower statistics:")

print(
    f"Minimum: "
    f"{np.min(fft_power):.10f}"
)

print(
    f"Maximum: "
    f"{np.max(fft_power):.10f}"
)

print(
    f"Mean: "
    f"{np.mean(fft_power):.10f}"
)

print(
    f"Std: "
    f"{np.std(fft_power):.10f}"
)


# ============================================================================
# NaN / INF CHECK
# ============================================================================

magnitude_nan = np.isnan(
    fft_magnitude
).sum()

magnitude_inf = np.isinf(
    fft_magnitude
).sum()

power_nan = np.isnan(
    fft_power
).sum()

power_inf = np.isinf(
    fft_power
).sum()


print("\nNumerical integrity:")

print(
    f"Magnitude NaN: {magnitude_nan}"
)

print(
    f"Magnitude Inf: {magnitude_inf}"
)

print(
    f"Power NaN: {power_nan}"
)

print(
    f"Power Inf: {power_inf}"
)


if magnitude_nan != 0:
    raise RuntimeError(
        "NaN values detected in FFT magnitude."
    )

if magnitude_inf != 0:
    raise RuntimeError(
        "Infinite values detected in FFT magnitude."
    )

if power_nan != 0:
    raise RuntimeError(
        "NaN values detected in FFT power."
    )

if power_inf != 0:
    raise RuntimeError(
        "Infinite values detected in FFT power."
    )


print("\n✓ FFT numerical integrity verified.")


# ============================================================================
# IMPORTANT FREQUENCY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("FREQUENCY CHECK")
print("=" * 80)

frequency_points = [
    0,
    1,
    4,
    8,
    10,
    12,
    20,
    30,
    40,
    50,
    60,
    100,
    500,
    1000,
]


print("\nSelected frequency bins:")

for target_frequency in frequency_points:

    index = np.argmin(
        np.abs(
            frequencies
            - target_frequency
        )
    )

    print(
        f"Requested: "
        f"{target_frequency:>4} Hz"
        f"  →  "
        f"Bin: {index:>4}"
        f"  →  "
        f"Actual: "
        f"{frequencies[index]:.6f} Hz"
    )


# ============================================================================
# METADATA
# ============================================================================

fft_metadata = metadata.copy()

fft_metadata["sampling_frequency_hz"] = SFREQ
fft_metadata["n_time_samples"] = N
fft_metadata["n_frequency_bins"] = N_FREQS
fft_metadata["frequency_resolution_hz"] = frequency_resolution
fft_metadata["trial_duration_sec"] = N / SFREQ


# ============================================================================
# SAVE
# ============================================================================

print("\n" + "=" * 80)
print("SAVING TEMPORAL FFT DATA")
print("=" * 80)


np.save(
    FFT_MAGNITUDE_FILE,
    fft_magnitude
)


np.save(
    FFT_POWER_FILE,
    fft_power
)


np.save(
    FREQUENCIES_FILE,
    frequencies.astype(
        np.float32
    )
)


fft_metadata.to_csv(
    METADATA_OUTPUT_FILE,
    index=False
)


print("\n✓ Saved FFT magnitude:")
print(FFT_MAGNITUDE_FILE)

print("\n✓ Saved FFT power:")
print(FFT_POWER_FILE)

print("\n✓ Saved frequency vector:")
print(FREQUENCIES_FILE)

print("\n✓ Saved FFT metadata:")
print(METADATA_OUTPUT_FILE)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 32 COMPLETE")
print("=" * 80)

print("\nOriginal temporal representation:")
print(
    f"  {EXPECTED_TRIALS} trials"
)

print(
    f"  {EXPECTED_CHANNELS} channels"
)

print(
    f"  {EXPECTED_SAMPLES} time samples"
)


print("\nFrequency representation:")
print(
    f"  {EXPECTED_TRIALS} trials"
)

print(
    f"  {EXPECTED_CHANNELS} channels"
)

print(
    f"  {N_FREQS} frequency bins"
)


print(
    f"\nFrequency resolution: "
    f"{frequency_resolution:.6f} Hz"
)

print(
    f"Frequency range: "
    f"{frequencies[0]:.6f}–"
    f"{frequencies[-1]:.6f} Hz"
)


print("\nFFT axis:")
print("  Temporal axis only (axis=2)")


print("\nNo model was trained.")
print("No train/test split was performed.")
print("No classification result was calculated.")


print(
    "\nThe true temporal frequency-domain EEG "
    "representation is now ready."
)

print("=" * 80)