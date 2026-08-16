

import os
import sys

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from src.data.data_loader import DataLoader
from src.preprocessing.eeg import prepare_eeg_channels
from src.preprocessing.full_recording_quality import (
    assess_full_recording_quality
)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "ds004514"
)


# ------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------

print("=" * 75)
print("PHASE 2.10 — FULL-RECORDING EEG QUALITY TEST")
print("=" * 75)

print("\nLoading EEG...")

loader = DataLoader(
    DATASET_ROOT
)

raw = loader.load_raw_bdf(
    "sub-01"
)

print("EEG loaded.")

print(
    "Original duration:",
    f"{raw.times[-1]:.2f} seconds"
)

print(
    "Original channels:",
    len(raw.ch_names)
)


# ------------------------------------------------------------
# PREPARE EEG CHANNEL TYPES
# ------------------------------------------------------------

print("\nPreparing EEG channels...")

raw = prepare_eeg_channels(
    raw
)

eeg_count = len(
    [
        channel
        for channel in raw.get_channel_types()
        if channel == "eeg"
    ]
)

print(
    "EEG channels:",
    eeg_count
)


# ------------------------------------------------------------
# RUN FULL-RECORDING QUALITY ASSESSMENT
# ------------------------------------------------------------

print("\nRunning full-recording quality assessment...")

bad_channels, report = (
    assess_full_recording_quality(
        raw,
        window_seconds=30.0,
        abnormal_window_fraction=0.25
    )
)


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL PERSISTENT BAD CHANNELS")
print("=" * 75)

if bad_channels:

    for channel in bad_channels:
        print(
            f"  - {channel}"
        )

else:

    print(
        "  None"
    )

print(
    "\nNumber of persistent bad channels:",
    len(bad_channels)
)

print(
    "Total windows:",
    report["total_windows"]
)

print(
    "Window size:",
    report["window_seconds"],
    "seconds"
)

print(
    "Persistence threshold:",
    f"{report['abnormal_window_fraction'] * 100:.1f}%"
)


# ------------------------------------------------------------
# PER-CHANNEL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CHANNEL PERSISTENCE SUMMARY")
print("=" * 75)

print(
    f"{'CHANNEL':<10}"
    f"{'BAD WINDOWS':>15}"
    f"{'TOTAL WINDOWS':>17}"
    f"{'FRACTION':>15}"
    f"{'STATUS':>12}"
)

print("-" * 75)

channels = report["channels"]

for channel, info in channels.items():

    status = (
        "BAD"
        if info["persistent_bad"]
        else "OK"
    )

    print(
        f"{channel:<10}"
        f"{info['abnormal_windows']:>15}"
        f"{info['total_windows']:>17}"
        f"{info['abnormal_fraction']:>15.3f}"
        f"{status:>12}"
    )


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("VALIDATION")
print("=" * 75)

assert report["total_windows"] > 0

assert len(
    report["channels"]
) == eeg_count

for channel in bad_channels:

    assert channel in channels

print(
    "PASS: Full-recording quality assessment completed."
)

print(
    "PASS: Every EEG channel has a persistence record."
)

print("\n" + "=" * 75)
print("FULL-RECORDING QUALITY TEST COMPLETE")
print("=" * 75)