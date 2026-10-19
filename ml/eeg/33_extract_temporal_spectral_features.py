"""
Neuro-Symphony
Step 33: Extract Temporal Spectral Features — sub-01

Input:
    True temporal FFT representation

    Shape:
        (720 trials, 64 channels, 3072 frequency bins)

The FFT was computed along the temporal axis from the preprocessed
EEG waveform.

This step:

    True Temporal FFT
          ↓
    Keep 1–40 Hz
          ↓
    Full spectral magnitude
    Full spectral power
          ↓
    EEG frequency-band power
          ↓
    Delta / Theta / Alpha / Beta / Gamma

Output:

    Full 1–40 Hz magnitude
        (720, 64, N_1_40)

    Full 1–40 Hz power
        (720, 64, N_1_40)

    Band-power features
        (720, 64, 5)

No model training.
No train/test split.
No test-set information is used.
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
# INPUT DIRECTORY
# ============================================================================

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_fft_sub01"
)


# ============================================================================
# INPUT FILES
# ============================================================================

FFT_MAGNITUDE_FILE = (
    INPUT_DIR
    / "sub-01_temporal_fft_magnitude.npy"
)

FFT_POWER_FILE = (
    INPUT_DIR
    / "sub-01_temporal_fft_power.npy"
)

FREQUENCIES_FILE = (
    INPUT_DIR
    / "sub-01_temporal_fft_frequencies.npy"
)

METADATA_FILE = (
    INPUT_DIR
    / "sub-01_temporal_fft_metadata.csv"
)


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_spectral_sub01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

SPECTRAL_MAGNITUDE_FILE = (
    OUTPUT_DIR
    / "sub-01_spectral_magnitude_1_40hz.npy"
)

SPECTRAL_POWER_FILE = (
    OUTPUT_DIR
    / "sub-01_spectral_power_1_40hz.npy"
)

BAND_POWER_FILE = (
    OUTPUT_DIR
    / "sub-01_band_power.npy"
)

FREQUENCIES_1_40_FILE = (
    OUTPUT_DIR
    / "sub-01_frequencies_1_40hz.npy"
)

BAND_NAMES_FILE = (
    OUTPUT_DIR
    / "band_names.txt"
)

METADATA_OUTPUT_FILE = (
    OUTPUT_DIR
    / "sub-01_spectral_metadata.csv"
)


# ============================================================================
# EXPECTED DATASET PARAMETERS
# ============================================================================

EXPECTED_TRIALS = 720
EXPECTED_CHANNELS = 64
EXPECTED_FFT_BINS = 3072

SFREQ = 2048.0


# ============================================================================
# FREQUENCY RANGE
# ============================================================================

LOW_FREQUENCY = 1.0
HIGH_FREQUENCY = 40.0


# ============================================================================
# EEG FREQUENCY BANDS
# ============================================================================

# The final Gamma range is limited to 30–40 Hz because the original
# preprocessing pipeline uses a 1–40 Hz band-pass filter.

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


BAND_NAMES = list(
    BANDS.keys()
)


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("NEURO-SYMPHONY — TEMPORAL SPECTRAL FEATURE EXTRACTION")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================================
# VALIDATE INPUT FILES
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATING INPUT FILES")
print("=" * 80)


required_files = [
    FFT_MAGNITUDE_FILE,
    FFT_POWER_FILE,
    FREQUENCIES_FILE,
    METADATA_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n{file_path}"
        )

    print(
        f"✓ Found: {file_path.name}"
    )


# ============================================================================
# LOAD FFT DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING TEMPORAL FFT DATA")
print("=" * 80)


fft_magnitude = np.load(
    FFT_MAGNITUDE_FILE,
    mmap_mode="r"
)


fft_power = np.load(
    FFT_POWER_FILE,
    mmap_mode="r"
)


frequencies = np.load(
    FREQUENCIES_FILE
)


metadata = pd.read_csv(
    METADATA_FILE
)


# ============================================================================
# DISPLAY INPUT SHAPES
# ============================================================================

print("\nFFT magnitude shape:")
print(fft_magnitude.shape)


print("\nFFT power shape:")
print(fft_power.shape)


print("\nFrequency vector shape:")
print(frequencies.shape)


print("\nMetadata shape:")
print(metadata.shape)


# ============================================================================
# VALIDATE INPUT SHAPES
# ============================================================================

expected_fft_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    EXPECTED_FFT_BINS,
)


if fft_magnitude.shape != expected_fft_shape:

    raise RuntimeError(
        "Unexpected FFT magnitude shape: "
        f"{fft_magnitude.shape}. "
        f"Expected {expected_fft_shape}."
    )


if fft_power.shape != expected_fft_shape:

    raise RuntimeError(
        "Unexpected FFT power shape: "
        f"{fft_power.shape}. "
        f"Expected {expected_fft_shape}."
    )


if len(metadata) != EXPECTED_TRIALS:

    raise RuntimeError(
        f"Expected {EXPECTED_TRIALS} metadata rows, "
        f"found {len(metadata)}."
    )


print(
    "\n✓ FFT dimensions verified."
)


# ============================================================================
# FREQUENCY VECTOR VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATING FREQUENCY VECTOR")
print("=" * 80)


print(
    f"\nMinimum frequency: "
    f"{frequencies[0]:.6f} Hz"
)


print(
    f"Maximum frequency: "
    f"{frequencies[-1]:.6f} Hz"
)


print(
    f"Number of frequency bins: "
    f"{len(frequencies)}"
)


frequency_resolution = (
    frequencies[1]
    - frequencies[0]
)


print(
    f"Frequency resolution: "
    f"{frequency_resolution:.6f} Hz"
)


# ============================================================================
# SELECT 1–40 Hz
# ============================================================================

print("\n" + "=" * 80)
print("SELECTING 1–40 Hz")
print("=" * 80)


frequency_mask = (
    (frequencies >= LOW_FREQUENCY)
    &
    (frequencies <= HIGH_FREQUENCY)
)


selected_indices = np.where(
    frequency_mask
)[0]


if len(selected_indices) == 0:

    raise RuntimeError(
        "No frequency bins found in the 1–40 Hz range."
    )


frequencies_1_40 = (
    frequencies[selected_indices]
)


print(
    f"\nSelected frequency bins: "
    f"{len(selected_indices)}"
)


print(
    f"First selected frequency: "
    f"{frequencies_1_40[0]:.6f} Hz"
)


print(
    f"Last selected frequency: "
    f"{frequencies_1_40[-1]:.6f} Hz"
)


print(
    f"Frequency resolution: "
    f"{frequency_resolution:.6f} Hz"
)


# ============================================================================
# EXTRACT FULL 1–40 Hz SPECTRUM
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING FULL 1–40 Hz SPECTRUM")
print("=" * 80)


print(
    "\nSelecting frequency axis only."
)


print(
    "No additional filtering is being performed."
)


print(
    "No labels are used during spectral extraction."
)


spectral_magnitude = np.asarray(
    fft_magnitude[
        :,
        :,
        selected_indices
    ],
    dtype=np.float32
)


spectral_power = np.asarray(
    fft_power[
        :,
        :,
        selected_indices
    ],
    dtype=np.float32
)


print("\nSpectral magnitude shape:")
print(
    spectral_magnitude.shape
)


print("\nSpectral power shape:")
print(
    spectral_power.shape
)


expected_spectral_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    len(selected_indices),
)


if spectral_magnitude.shape != expected_spectral_shape:

    raise RuntimeError(
        "Unexpected 1–40 Hz magnitude shape: "
        f"{spectral_magnitude.shape}. "
        f"Expected {expected_spectral_shape}."
    )


if spectral_power.shape != expected_spectral_shape:

    raise RuntimeError(
        "Unexpected 1–40 Hz power shape: "
        f"{spectral_power.shape}. "
        f"Expected {expected_spectral_shape}."
    )


print(
    "\n✓ Full 1–40 Hz spectral representation verified."
)


# ============================================================================
# BAND POWER EXTRACTION
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING EEG BAND-POWER FEATURES")
print("=" * 80)


print(
    "\nFrequency bands:"
)


for band_name, (
    band_low,
    band_high
) in BANDS.items():

    print(
        f"  {band_name:<8} "
        f"{band_low:>5.1f}–"
        f"{band_high:>5.1f} Hz"
    )


# Allocate:

# trials × channels × bands

band_power = np.zeros(
    (
        EXPECTED_TRIALS,
        EXPECTED_CHANNELS,
        len(BANDS),
    ),
    dtype=np.float32,
)


for band_index, (
    band_name,
    (
        band_low,
        band_high
    )
) in enumerate(
    BANDS.items()
):

    # ------------------------------------------------------------------------
    # Find bins belonging to this frequency band
    # ------------------------------------------------------------------------

    band_mask = (
        (frequencies_1_40 >= band_low)
        &
        (frequencies_1_40 <= band_high)
    )


    band_indices = np.where(
        band_mask
    )[0]


    if len(band_indices) == 0:

        raise RuntimeError(
            f"No FFT bins found for "
            f"{band_name} band."
        )


    band_frequencies = (
        frequencies_1_40[
            band_indices
        ]
    )


    print(
        f"\n{band_name.upper()}:"
    )


    print(
        f"  Requested range: "
        f"{band_low:.1f}–"
        f"{band_high:.1f} Hz"
    )


    print(
        f"  Actual range: "
        f"{band_frequencies[0]:.6f}–"
        f"{band_frequencies[-1]:.6f} Hz"
    )


    print(
        f"  Number of bins: "
        f"{len(band_indices)}"
    )


    # ------------------------------------------------------------------------
    # Extract power for this band
    # ------------------------------------------------------------------------

    band_power_values = (
        spectral_power[
            :,
            :,
            band_indices
        ]
    )


    # ------------------------------------------------------------------------
    # Integrate power across frequency bins
    # ------------------------------------------------------------------------
    #
    # We use the mean power across the bins in the band.
    #
    # This gives one band-power feature for every:
    #
    #     trial × channel × band
    #
    # and keeps the scale comparable across bands with different
    # numbers of frequency bins.
    #

    mean_band_power = np.mean(
        band_power_values,
        axis=2
    )


    band_power[
        :,
        :,
        band_index
    ] = mean_band_power.astype(
        np.float32
    )


# ============================================================================
# VALIDATE BAND POWER
# ============================================================================

print("\n" + "=" * 80)
print("BAND-POWER VALIDATION")
print("=" * 80)


expected_band_shape = (
    EXPECTED_TRIALS,
    EXPECTED_CHANNELS,
    len(BANDS),
)


print("\nBand-power shape:")
print(
    band_power.shape
)


print("\nExpected:")
print(
    expected_band_shape
)


if band_power.shape != expected_band_shape:

    raise RuntimeError(
        f"Unexpected band-power shape: "
        f"{band_power.shape}. "
        f"Expected {expected_band_shape}."
    )


print(
    "\n✓ Band-power shape verified."
)


# ============================================================================
# NUMERICAL VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("NUMERICAL VALIDATION")
print("=" * 80)


print("\nFull spectral magnitude:")

print(
    f"Minimum: "
    f"{np.min(spectral_magnitude):.10f}"
)

print(
    f"Maximum: "
    f"{np.max(spectral_magnitude):.10f}"
)

print(
    f"Mean: "
    f"{np.mean(spectral_magnitude):.10f}"
)

print(
    f"Std: "
    f"{np.std(spectral_magnitude):.10f}"
)


print("\nFull spectral power:")

print(
    f"Minimum: "
    f"{np.min(spectral_power):.10f}"
)

print(
    f"Maximum: "
    f"{np.max(spectral_power):.10f}"
)

print(
    f"Mean: "
    f"{np.mean(spectral_power):.10f}"
)

print(
    f"Std: "
    f"{np.std(spectral_power):.10f}"
)


print("\nBand power:")

print(
    f"Minimum: "
    f"{np.min(band_power):.10f}"
)

print(
    f"Maximum: "
    f"{np.max(band_power):.10f}"
)

print(
    f"Mean: "
    f"{np.mean(band_power):.10f}"
)

print(
    f"Std: "
    f"{np.std(band_power):.10f}"
)


# ============================================================================
# NaN / INF CHECK
# ============================================================================

spectral_magnitude_nan = np.isnan(
    spectral_magnitude
).sum()

spectral_magnitude_inf = np.isinf(
    spectral_magnitude
).sum()


spectral_power_nan = np.isnan(
    spectral_power
).sum()

spectral_power_inf = np.isinf(
    spectral_power
).sum()


band_power_nan = np.isnan(
    band_power
).sum()

band_power_inf = np.isinf(
    band_power
).sum()


print("\nNumerical integrity:")

print(
    f"Spectral magnitude NaN: "
    f"{spectral_magnitude_nan}"
)

print(
    f"Spectral magnitude Inf: "
    f"{spectral_magnitude_inf}"
)

print(
    f"Spectral power NaN: "
    f"{spectral_power_nan}"
)

print(
    f"Spectral power Inf: "
    f"{spectral_power_inf}"
)

print(
    f"Band power NaN: "
    f"{band_power_nan}"
)

print(
    f"Band power Inf: "
    f"{band_power_inf}"
)


if spectral_magnitude_nan != 0:

    raise RuntimeError(
        "NaN values detected in spectral magnitude."
    )


if spectral_magnitude_inf != 0:

    raise RuntimeError(
        "Infinite values detected in spectral magnitude."
    )


if spectral_power_nan != 0:

    raise RuntimeError(
        "NaN values detected in spectral power."
    )


if spectral_power_inf != 0:

    raise RuntimeError(
        "Infinite values detected in spectral power."
    )


if band_power_nan != 0:

    raise RuntimeError(
        "NaN values detected in band power."
    )


if band_power_inf != 0:

    raise RuntimeError(
        "Infinite values detected in band power."
    )


print(
    "\n✓ All spectral features contain finite values."
)


# ============================================================================
# BAND-WISE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BAND-POWER SUMMARY")
print("=" * 80)


for band_index, band_name in enumerate(
    BAND_NAMES
):

    values = band_power[
        :,
        :,
        band_index
    ]


    print(
        f"\n{band_name.upper()}"
    )


    print(
        f"  Mean: "
        f"{np.mean(values):.10f}"
    )


    print(
        f"  Std: "
        f"{np.std(values):.10f}"
    )


    print(
        f"  Min: "
        f"{np.min(values):.10f}"
    )


    print(
        f"  Max: "
        f"{np.max(values):.10f}"
    )


# ============================================================================
# METADATA
# ============================================================================

spectral_metadata = metadata.copy()


spectral_metadata[
    "sampling_frequency_hz"
] = SFREQ


spectral_metadata[
    "spectral_low_frequency_hz"
] = LOW_FREQUENCY


spectral_metadata[
    "spectral_high_frequency_hz"
] = HIGH_FREQUENCY


spectral_metadata[
    "n_spectral_frequency_bins"
] = len(
    selected_indices
)


spectral_metadata[
    "frequency_resolution_hz"
] = frequency_resolution


spectral_metadata[
    "n_band_features"
] = len(BANDS)


# ============================================================================
# SAVE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SPECTRAL FEATURES")
print("=" * 80)


np.save(
    SPECTRAL_MAGNITUDE_FILE,
    spectral_magnitude
)


np.save(
    SPECTRAL_POWER_FILE,
    spectral_power
)


np.save(
    BAND_POWER_FILE,
    band_power
)


np.save(
    FREQUENCIES_1_40_FILE,
    frequencies_1_40.astype(
        np.float32
    )
)


spectral_metadata.to_csv(
    METADATA_OUTPUT_FILE,
    index=False
)


with open(
    BAND_NAMES_FILE,
    "w",
    encoding="utf-8"
) as f:

    for band_name in BAND_NAMES:

        f.write(
            band_name + "\n"
        )


print("\n✓ Saved full 1–40 Hz magnitude:")
print(SPECTRAL_MAGNITUDE_FILE)


print("\n✓ Saved full 1–40 Hz power:")
print(SPECTRAL_POWER_FILE)


print("\n✓ Saved band-power features:")
print(BAND_POWER_FILE)


print("\n✓ Saved 1–40 Hz frequency vector:")
print(FREQUENCIES_1_40_FILE)


print("\n✓ Saved band names:")
print(BAND_NAMES_FILE)


print("\n✓ Saved spectral metadata:")
print(METADATA_OUTPUT_FILE)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 33 COMPLETE")
print("=" * 80)


print("\nOriginal temporal EEG:")
print(
    "  720 trials × 64 channels × 6142 time samples"
)


print("\nOriginal temporal FFT:")
print(
    "  720 trials × 64 channels × 3072 frequency bins"
)


print("\nSelected spectral range:")
print(
    f"  {LOW_FREQUENCY:.1f}–"
    f"{HIGH_FREQUENCY:.1f} Hz"
)


print("\nFull spectral magnitude:")
print(
    f"  {spectral_magnitude.shape}"
)


print("\nFull spectral power:")
print(
    f"  {spectral_power.shape}"
)


print("\nBand-power representation:")
print(
    f"  {band_power.shape}"
)


print("\nBands:")

for band_name, (
    low,
    high
) in BANDS.items():

    print(
        f"  {band_name}: "
        f"{low:.1f}–"
        f"{high:.1f} Hz"
    )


print("\nNo model was trained.")
print("No train/test split was performed.")
print("No labels were used to construct the spectral features.")


print(
    "\nThe 1–40 Hz temporal spectral features "
    "are now ready for controlled classification experiments."
)


print("=" * 80)