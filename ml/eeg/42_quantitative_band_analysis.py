from pathlib import Path

import mne
import numpy as np
import pandas as pd


# ============================================================
# STEP 42 — QUANTITATIVE BAND-WISE EEG ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BDF_PATH = PROJECT_ROOT / "data" / "raw" / "sub-01" / "sub-01_task-eeg_eeg.bdf"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step42"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SFREQ = 2048.0
TRIAL_SAMPLES = 6142

CHANNEL = "B6"          # Original BioSemi channel
DISPLAY_CHANNEL = "Fz"  # B6 corresponds to Fz

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40),
}


# Actual labels found in DS004514
CLASS_LABELS = {
    "animal_auditory_imagery_task": "Animal - Auditory",
    "animal_silent_naming_task": "Animal - Silent",
    "animal_tactile_imagery_task": "Animal - Tactile",
    "animal_visual_imagery_task": "Animal - Visual",
    "tool_auditory_imagery_task": "Tool - Auditory",
    "tool_silent_naming_task": "Tool - Silent",
    "tool_tactile_imagery_task": "Tool - Tactile",
    "tool_visual_imagery_task": "Tool - Visual",
}


# ------------------------------------------------------------
# Load BDF
# ------------------------------------------------------------

print("=" * 70)
print("STEP 42 — QUANTITATIVE BAND-WISE EEG ANALYSIS")
print("=" * 70)

print("\nLoading:", BDF_PATH)

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=True,
    verbose=False
)

print(f"Sampling frequency : {raw.info['sfreq']} Hz")
print(f"Number of channels : {len(raw.ch_names)}")
print(f"Recording duration : {raw.times[-1]:.3f} s")


# ------------------------------------------------------------
# Find Status channel and events
# ------------------------------------------------------------

events = mne.find_events(
    raw,
    stim_channel="Status",
    shortest_event=1,
    verbose=False
)

print(f"\nTotal event rows found: {len(events)}")


# ------------------------------------------------------------
# Find task events using event IDs
# ------------------------------------------------------------

event_id = {
    label: idx
    for idx, label in enumerate(
        CLASS_LABELS.keys(),
        start=1
    )
}

# ------------------------------------------------------------
# IMPORTANT:
# We use the event timing information already established
# for the 720 task trials.
#
# Step 42 uses the same first-occurrence representative
# trial per class as Step 36 / 41.
# ------------------------------------------------------------

# These are the representative onset samples established
# during the previous analysis.
REPRESENTATIVE_ONSETS = {
    "Animal - Auditory": 800706,
    "Animal - Silent": 807238,
    "Animal - Tactile": 787597,
    "Animal - Visual": 794152,
    "Tool - Auditory": 836748,
    "Tool - Silent": 843280,
    "Tool - Tactile": 823642,
    "Tool - Visual": 830197,
}


# ------------------------------------------------------------
# Extract one 3-second trial
# ------------------------------------------------------------

def extract_trial(signal, start_sample, n_samples):
    end_sample = start_sample + n_samples

    if start_sample < 0 or end_sample > len(signal):
        raise ValueError(
            f"Trial outside recording: "
            f"{start_sample} -> {end_sample}"
        )

    return signal[start_sample:end_sample]


# ------------------------------------------------------------
# Calculate quantitative features
# ------------------------------------------------------------

def calculate_features(signal):
    mean_value = np.mean(signal)

    std_value = np.std(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    peak_to_peak = (
        np.max(signal) -
        np.min(signal)
    )

    variance = np.var(signal)

    return {
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "peak_to_peak": peak_to_peak,
        "variance": variance,
    }


# ------------------------------------------------------------
# Get channel data
# ------------------------------------------------------------

if CHANNEL not in raw.ch_names:
    raise ValueError(
        f"{CHANNEL} not found in BDF channels."
    )

channel_index = raw.ch_names.index(CHANNEL)

signal = raw.get_data(
    picks=[channel_index]
)[0]

print(
    f"\nUsing channel: {CHANNEL} "
    f"({DISPLAY_CHANNEL})"
)

print(
    f"Signal samples: {len(signal)}"
)


# ------------------------------------------------------------
# Process all 8 classes × 5 bands
# ------------------------------------------------------------

results = []

print("\nCalculating band-wise features...\n")

for class_name, start_sample in REPRESENTATIVE_ONSETS.items():

    trial = extract_trial(
        signal,
        start_sample,
        TRIAL_SAMPLES
    )

    for band_name, (low_freq, high_freq) in BANDS.items():

        print(
            f"Processing "
            f"{class_name:20s} | "
            f"{band_name:6s} "
            f"({low_freq}-{high_freq} Hz)"
        )

        # ----------------------------------------------------
        # Create temporary RawArray for filtering
        # ----------------------------------------------------

        info = mne.create_info(
            ch_names=[CHANNEL],
            sfreq=SFREQ,
            ch_types=["eeg"]
        )

        trial_raw = mne.io.RawArray(
            trial[np.newaxis, :],
            info,
            verbose=False
        )

        # ----------------------------------------------------
        # Band-pass filter
        # ----------------------------------------------------

        filtered_raw = trial_raw.copy().filter(
            l_freq=low_freq,
            h_freq=high_freq,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False
        )

        filtered_signal = filtered_raw.get_data()[0]

        # ----------------------------------------------------
        # Calculate features
        # ----------------------------------------------------

        features = calculate_features(
            filtered_signal
        )

        results.append({
            "class": class_name,
            "band": band_name,
            "low_frequency_hz": low_freq,
            "high_frequency_hz": high_freq,
            "channel": DISPLAY_CHANNEL,
            "sampling_frequency_hz": SFREQ,
            "samples": len(filtered_signal),
            "duration_seconds": len(filtered_signal) / SFREQ,
            **features,
        })


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(results)


# ------------------------------------------------------------
# Save complete results
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "step42_band_features_sub-01_Fz.csv"

df.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# Create compact comparison tables
# ------------------------------------------------------------

rms_table = df.pivot(
    index="class",
    columns="band",
    values="rms"
)

std_table = df.pivot(
    index="class",
    columns="band",
    values="std"
)

ptp_table = df.pivot(
    index="class",
    columns="band",
    values="peak_to_peak"
)


rms_table.to_csv(
    OUTPUT_DIR / "step42_rms_comparison.csv"
)

std_table.to_csv(
    OUTPUT_DIR / "step42_std_comparison.csv"
)

ptp_table.to_csv(
    OUTPUT_DIR / "step42_peak_to_peak_comparison.csv"
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6e}"
)

print("\n" + "=" * 70)
print("RMS COMPARISON")
print("=" * 70)

print(rms_table)


print("\n" + "=" * 70)
print("STANDARD DEVIATION COMPARISON")
print("=" * 70)

print(std_table)


print("\n" + "=" * 70)
print("PEAK-TO-PEAK COMPARISON")
print("=" * 70)

print(ptp_table)


print("\n" + "=" * 70)
print("STEP 42 COMPLETE")
print("=" * 70)

print("\nOutput files:")

print(csv_path)

print(
    OUTPUT_DIR /
    "step42_rms_comparison.csv"
)

print(
    OUTPUT_DIR /
    "step42_std_comparison.csv"
)

print(
    OUTPUT_DIR /
    "step42_peak_to_peak_comparison.csv"
)