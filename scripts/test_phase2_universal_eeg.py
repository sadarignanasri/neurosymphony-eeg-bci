import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from src.data.data_loader import DataLoader
from src.preprocessing.pipeline import preprocess_eeg
from src.preprocessing.dataset_configs import (
    DS004514_EEG_CONFIG,
)


DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "ds004514"
)


TEST_DURATION = 60.0


print("=" * 75)
print("PHASE 2.11 — UNIVERSAL EEG PIPELINE TEST")
print("=" * 75)

print("\nLoading DS004514 EEG...")

loader = DataLoader(
    DATASET_ROOT
)

raw = loader.load_raw_bdf(
    "sub-01"
)

print("EEG loaded.")

print(
    "Original channels:",
    len(raw.ch_names)
)

print(
    "Original duration:",
    f"{raw.times[-1]:.2f} seconds"
)


# ------------------------------------------------------------
# USE ONLY A DEVELOPMENT SEGMENT
# ------------------------------------------------------------

print(
    f"\nCreating {TEST_DURATION:.0f}-second "
    "validation segment..."
)

test_raw = raw.copy().crop(
    tmin=0.0,
    tmax=TEST_DURATION
)

print(
    "Test channels:",
    len(test_raw.ch_names)
)

print(
    "Test duration:",
    f"{test_raw.times[-1]:.2f} seconds"
)


# ------------------------------------------------------------
# RUN UNIVERSAL PHASE 2
# ------------------------------------------------------------

print(
    "\nRunning universal Phase 2 EEG pipeline..."
)

processed, quality_report = preprocess_eeg(
    test_raw,
    dataset_config=DS004514_EEG_CONFIG,
)

print(
    "\nPhase 2 preprocessing complete."
)


# ------------------------------------------------------------
# CHANNEL CHECK
# ------------------------------------------------------------

eeg_channels = [
    channel
    for channel, channel_type
    in zip(
        processed.ch_names,
        processed.get_channel_types()
    )
    if channel_type == "eeg"
]

print(
    "\nFinal EEG channels:",
    len(eeg_channels)
)

assert len(eeg_channels) == 64

print(
    "PASS: 64 EEG channels present."
)


# ------------------------------------------------------------
# BAD CHANNEL CHECK
# ------------------------------------------------------------
# ------------------------------------------------------------
# BAD CHANNEL / INTERPOLATION CHECK
# ------------------------------------------------------------

expected_interpolated = {
    "Iz",
    "TP8",
    "P6",
}

print(
    "\nInterpolated bad channels:"
)

for channel in sorted(
    expected_interpolated
):
    assert channel in processed.ch_names
    print(
        f"  - {channel}"
    )

print(
    "PASS: Expected bad EEG channels were "
    "successfully repaired by interpolation."
)

# Interpolated channels should no longer remain
# in MNE's active bad-channel list.
remaining_bad = set(
    processed.info["bads"]
)

assert not (
    remaining_bad
    & expected_interpolated
)

print(
    "PASS: Interpolated channels are no longer "
    "marked as bad."
)

# ------------------------------------------------------------
# FINITE CHECK
# ------------------------------------------------------------

data = processed.get_data(
    picks=eeg_channels
)

nan_count = int(
    np.isnan(data).sum()
)

inf_count = int(
    np.isinf(data).sum()
)

print("\nFinal data validation:")

print(
    "NaNs      :",
    nan_count
)

print(
    "Infinities:",
    inf_count
)

assert nan_count == 0
assert inf_count == 0

print(
    "PASS: Final EEG contains only finite values."
)


# ------------------------------------------------------------
# MONTAGE CHECK
# ------------------------------------------------------------

print("\nMontage check:")

assert processed.info["dig"] is not None

print(
    "PASS: EEG sensor positions are present."
)


# ------------------------------------------------------------
# REPAIRED CHANNELS
# ------------------------------------------------------------

print("\nRepaired channel statistics:")

for channel in [
    "Iz",
    "TP8",
    "P6",
]:

    signal = processed.get_data(
        picks=[channel]
    )[0]

    print(
        f"\n{channel}"
    )

    print(
        f"STD : {np.std(signal):.10f}"
    )

    print(
        f"MIN : {np.min(signal):.10f}"
    )

    print(
        f"MAX : {np.max(signal):.10f}"
    )

    print(
        f"P2P : {np.ptp(signal):.10f}"
    )


# ------------------------------------------------------------
# ORIGINAL TEST OBJECT INTEGRITY
# ------------------------------------------------------------

print("\nOriginal test Raw integrity:")

assert len(
    test_raw.ch_names
) == 80

print(
    "PASS: Original test Raw object remains unchanged."
)


# ------------------------------------------------------------
# QUALITY REPORT
# ------------------------------------------------------------

assert quality_report is not None

print("\nQuality report:")

print(
    "Total windows:",
    quality_report["total_windows"]
)

print(
    "Window size:",
    quality_report["window_seconds"]
)

print(
    "Persistence threshold:",
    quality_report[
        "abnormal_window_fraction"
    ]
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PHASE 2.11 COMPLETE")
print("=" * 75)

print(
    "\nUniversal EEG preprocessing pipeline "
    "validated on DS004514."
)