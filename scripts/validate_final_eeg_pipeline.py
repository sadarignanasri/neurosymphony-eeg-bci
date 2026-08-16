
import os
import sys
import numpy as np
import mne

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.data.data_loader import DataLoader
from src.preprocessing.eeg import prepare_eeg_channels


# ============================================================
# DATASET
# ============================================================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "ds004514"
)


# ============================================================
# BIOSEMI → STANDARD 10-20 MAPPING
# ============================================================

CHANNEL_MAPPING = {
    "A1": "Fp1",
    "A2": "AF7",
    "A3": "AF3",
    "A4": "F1",
    "A5": "F3",
    "A6": "F5",
    "A7": "F7",
    "A8": "FT7",
    "A9": "FC5",
    "A10": "FC3",
    "A11": "FC1",
    "A12": "C1",
    "A13": "C3",
    "A14": "C5",
    "A15": "T7",
    "A16": "TP7",
    "A17": "CP5",
    "A18": "CP3",
    "A19": "CP1",
    "A20": "P1",
    "A21": "P3",
    "A22": "P5",
    "A23": "P7",
    "A24": "P9",
    "A25": "PO7",
    "A26": "PO3",
    "A27": "O1",
    "A28": "Iz",
    "A29": "Oz",
    "A30": "POz",
    "A31": "Pz",
    "A32": "CPz",

    "B1": "Fpz",
    "B2": "Fp2",
    "B3": "AF8",
    "B4": "AF4",
    "B5": "AFz",
    "B6": "Fz",
    "B7": "F2",
    "B8": "F4",
    "B9": "F6",
    "B10": "F8",
    "B11": "FT8",
    "B12": "FC6",
    "B13": "FC4",
    "B14": "FC2",
    "B15": "FCz",
    "B16": "Cz",
    "B17": "C2",
    "B18": "C4",
    "B19": "C6",
    "B20": "T8",
    "B21": "TP8",
    "B22": "CP6",
    "B23": "CP4",
    "B24": "CP2",
    "B25": "P2",
    "B26": "P4",
    "B27": "P6",
    "B28": "P8",
    "B29": "P10",
    "B30": "PO8",
    "B31": "PO4",
    "B32": "O2",
}


# ============================================================
# FINAL EEG CONFIGURATION
# ============================================================

BAD_CHANNELS = [
    "Iz",
    "TP8",
    "P6",
]

LOW_CUT = 1.0
HIGH_CUT = 40.0
NOTCH_FREQ = 50.0


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PHASE 2.7 — FINAL EEG PIPELINE VALIDATION")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

print("\nLoading EEG...")

loader = DataLoader(DATASET_ROOT)

raw = loader.load_raw_bdf("sub-01")

print("EEG loaded.")

print(
    f"Original channels : {len(raw.ch_names)}"
)

print(
    f"Original duration : {raw.times[-1]:.2f} seconds"
)


# ============================================================
# KEEP ORIGINAL COPY FOR COMPARISON
# ============================================================

original_raw = raw.copy()


# ============================================================
# PREPARE EEG CHANNELS
# ============================================================

print("\nPreparing EEG channels...")

raw = prepare_eeg_channels(raw)

print("EEG channels identified.")


# ============================================================
# RENAME BIOSEMI CHANNELS
# ============================================================

print("\nApplying BioSemi channel mapping...")

raw.rename_channels(CHANNEL_MAPPING)

print("BioSemi channels mapped.")


# ============================================================
# APPLY MONTAGE
# ============================================================

print("\nApplying BioSemi-64 montage...")

montage = mne.channels.make_standard_montage(
    "biosemi64"
)

raw.set_montage(
    montage,
    on_missing="ignore"
)

print("BioSemi-64 montage applied.")


# ============================================================
# VERIFY EEG CHANNEL COUNT
# ============================================================

eeg_picks = mne.pick_types(
    raw.info,
    eeg=True,
    exclude=[]
)

eeg_channels = [
    raw.ch_names[index]
    for index in eeg_picks
]

print(
    f"\nEEG channels after preparation: "
    f"{len(eeg_channels)}"
)

assert len(eeg_channels) == 64, (
    f"Expected 64 EEG channels, "
    f"found {len(eeg_channels)}"
)

print("PASS: Exactly 64 EEG channels found.")


# ============================================================
# VERIFY REQUIRED CHANNELS
# ============================================================

print("\nChecking required channels...")

required_channels = [
    "Iz",
    "TP8",
    "P6",
]

missing = [
    channel
    for channel in required_channels
    if channel not in raw.ch_names
]

assert not missing, (
    f"Missing required channels: {missing}"
)

print("PASS: Iz, TP8 and P6 are present.")


# ============================================================
# APPLY 1–40 Hz BAND-PASS
# ============================================================

print("\nApplying 1–40 Hz band-pass...")

raw.filter(
    LOW_CUT,
    HIGH_CUT,
    picks=eeg_picks,
    fir_design="firwin"
)

print("Band-pass complete.")


# ============================================================
# APPLY 50 Hz NOTCH
# ============================================================

print("\nApplying 50 Hz notch...")

raw.notch_filter(
    NOTCH_FREQ,
    picks=eeg_picks
)

print("Notch complete.")


# ============================================================
# CHECK DATA BEFORE INTERPOLATION
# ============================================================

print("\nChecking filtered EEG before interpolation...")

data_before = raw.get_data(
    picks=eeg_picks
)

nan_before = np.isnan(data_before).sum()
inf_before = np.isinf(data_before).sum()

print(
    f"NaNs before interpolation : {nan_before}"
)

print(
    f"Infinities before interpolation : {inf_before}"
)

assert nan_before == 0, "NaNs detected before interpolation."
assert inf_before == 0, "Infinities detected before interpolation."

print("PASS: Filtered EEG contains no NaNs or infinities.")


# ============================================================
# SAVE ORIGINAL BAD-CHANNEL VALUES
# ============================================================

print("\nRecording bad-channel statistics before repair...")

before_stats = {}

for channel in BAD_CHANNELS:

    signal = raw.get_data(
        picks=[channel]
    )[0]

    before_stats[channel] = {
        "std": np.std(signal),
        "range": np.ptp(signal),
        "max_abs": np.max(np.abs(signal)),
    }

    print(
        f"{channel:<5} "
        f"STD={np.std(signal):.8f} "
        f"RANGE={np.ptp(signal):.8f}"
    )


# ============================================================
# MARK FINAL BAD CHANNELS
# ============================================================

print("\nMarking final EEG bad channels...")

raw.info["bads"] = BAD_CHANNELS.copy()

print(
    "Final bad channels:",
    raw.info["bads"]
)


# ============================================================
# INTERPOLATE
# ============================================================

print("\nApplying spherical-spline interpolation...")

raw.interpolate_bads(
    reset_bads=False,
    method=dict(
        eeg="spline"
    )
)

print("Interpolation complete.")


# ============================================================
# VERIFY BAD CHANNELS STILL EXIST
# ============================================================

for channel in BAD_CHANNELS:

    assert channel in raw.ch_names, (
        f"{channel} disappeared after interpolation."
    )

print(
    "PASS: All repaired channels remain in the EEG."
)


# ============================================================
# CHECK FINAL DATA
# ============================================================

print("\nChecking final EEG data...")

final_data = raw.get_data(
    picks=eeg_picks
)

nan_after = np.isnan(final_data).sum()
inf_after = np.isinf(final_data).sum()

print(
    f"Final NaNs       : {nan_after}"
)

print(
    f"Final infinities : {inf_after}"
)

assert nan_after == 0, (
    "NaNs detected after interpolation."
)

assert inf_after == 0, (
    "Infinities detected after interpolation."
)

print(
    "PASS: Final EEG contains no NaNs or infinities."
)


# ============================================================
# CHECK REPAIRED CHANNELS
# ============================================================

print("\nFinal repaired-channel statistics:")

for channel in BAD_CHANNELS:

    signal = raw.get_data(
        picks=[channel]
    )[0]

    print(
        f"\n{channel}"
    )

    print(
        f"STD   : {np.std(signal):.10f}"
    )

    print(
        f"MIN   : {np.min(signal):.10f}"
    )

    print(
        f"MAX   : {np.max(signal):.10f}"
    )

    print(
        f"P2P   : {np.ptp(signal):.10f}"
    )


# ============================================================
# CHECK ALL CHANNELS ARE FINITE
# ============================================================

print("\nChecking every EEG channel...")

for channel in eeg_channels:

    signal = raw.get_data(
        picks=[channel]
    )[0]

    assert np.all(np.isfinite(signal)), (
        f"Invalid values found in {channel}"
    )

print(
    "PASS: All 64 EEG channels contain finite values."
)


# ============================================================
# CHECK MONTAGE
# ============================================================

print("\nChecking montage...")

assert raw.get_montage() is not None

print(
    "PASS: EEG montage is present."
)


# ============================================================
# CHECK CHANNEL COUNT AFTER INTERPOLATION
# ============================================================

final_eeg_picks = mne.pick_types(
    raw.info,
    eeg=True,
    exclude=[]
)

print(
    f"\nFinal EEG channel count: "
    f"{len(final_eeg_picks)}"
)

assert len(final_eeg_picks) == 64

print(
    "PASS: Final pipeline contains exactly 64 EEG channels."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FINAL EEG PIPELINE VALIDATION SUMMARY")
print("=" * 75)

print(
    "\nOriginal channels       : "
    f"{len(original_raw.ch_names)}"
)

print(
    "Final EEG channels      : "
    f"{len(final_eeg_picks)}"
)

print(
    "Band-pass               : "
    "1–40 Hz"
)

print(
    "Notch                   : "
    "50 Hz"
)

print(
    "Bad channels repaired   : "
    f"{BAD_CHANNELS}"
)

print(
    "Interpolation           : "
    "Spherical spline"
)

print(
    "NaNs                    : "
    f"{nan_after}"
)

print(
    "Infinities              : "
    f"{inf_after}"
)


# ============================================================
# FINAL PASS
# ============================================================

print("\n")
print("=" * 75)
print("PHASE 2.7 VALIDATION PASSED")
print("=" * 75)

print(
    "\nEEG preprocessing pipeline is internally valid."
)

print(
    "No original dataset files were modified."
)

print(
    "No GitHub changes were made."
)

print(
    "\nNEXT STEP:"
)

print(
    "Integrate the finalized EEG configuration "
    "into src/preprocessing/eeg.py."
)