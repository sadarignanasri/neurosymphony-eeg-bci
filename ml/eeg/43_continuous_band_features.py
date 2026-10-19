from pathlib import Path

import mne
import numpy as np
import pandas as pd


# ============================================================
# STEP 43 — CONTINUOUS EEG BAND-PASS FEATURE ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_eeg.bdf"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step43"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SFREQ = 2048.0

CHANNEL = "B6"
DISPLAY_CHANNEL = "Fz"

TRIAL_SAMPLES = 6142

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40),
}


# Same representative trials used in Steps 36–42
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


# ============================================================
# LOAD ORIGINAL BDF
# ============================================================

print("=" * 70)
print("STEP 43 — CONTINUOUS EEG BAND-PASS FEATURE ANALYSIS")
print("=" * 70)

print("\nLoading:")
print(BDF_PATH)

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=True,
    verbose=False
)

print(
    f"\nSampling frequency : "
    f"{raw.info['sfreq']} Hz"
)

print(
    f"Number of channels : "
    f"{len(raw.ch_names)}"
)

print(
    f"Recording duration : "
    f"{raw.times[-1]:.3f} s"
)


# ============================================================
# GET FZ / B6 SIGNAL
# ============================================================

if CHANNEL not in raw.ch_names:
    raise ValueError(
        f"{CHANNEL} not found in recording."
    )

channel_index = raw.ch_names.index(CHANNEL)

signal = raw.get_data(
    picks=[channel_index]
)[0]

print(
    f"\nUsing channel: "
    f"{CHANNEL} ({DISPLAY_CHANNEL})"
)

print(
    f"Total samples: {len(signal)}"
)


# ============================================================
# FEATURE FUNCTION
# ============================================================

def calculate_features(signal):

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    peak_to_peak = (
        np.max(signal)
        - np.min(signal)
    )

    variance = np.var(signal)

    return {
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "peak_to_peak": peak_to_peak,
        "variance": variance,
    }


# ============================================================
# PROCESS EACH BAND
# ============================================================

all_results = []


for band_name, (low_freq, high_freq) in BANDS.items():

    print("\n" + "-" * 70)

    print(
        f"Filtering continuous EEG: "
        f"{band_name} "
        f"({low_freq}-{high_freq} Hz)"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Create one-channel continuous RawArray
    # --------------------------------------------------------

    info = mne.create_info(
        ch_names=[CHANNEL],
        sfreq=SFREQ,
        ch_types=["eeg"]
    )

    continuous_raw = mne.io.RawArray(
        signal[np.newaxis, :],
        info,
        verbose=False
    )

    # --------------------------------------------------------
    # Filter the CONTINUOUS signal
    # --------------------------------------------------------

    filtered_raw = continuous_raw.copy().filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False
    )

    filtered_signal = (
        filtered_raw.get_data()[0]
    )

    print(
        f"Filtered samples: "
        f"{len(filtered_signal)}"
    )

    # --------------------------------------------------------
    # Extract representative trials
    # --------------------------------------------------------

    for class_name, start_sample in (
        REPRESENTATIVE_ONSETS.items()
    ):

        end_sample = (
            start_sample
            + TRIAL_SAMPLES
        )

        if end_sample > len(filtered_signal):

            raise ValueError(
                f"Trial exceeds recording "
                f"for {class_name}"
            )

        trial = filtered_signal[
            start_sample:end_sample
        ]

        features = calculate_features(
            trial
        )

        all_results.append({
            "class": class_name,
            "band": band_name,
            "low_frequency_hz": low_freq,
            "high_frequency_hz": high_freq,
            "channel": DISPLAY_CHANNEL,
            "sampling_frequency_hz": SFREQ,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "samples": len(trial),
            "duration_seconds":
                len(trial) / SFREQ,
            **features,
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

complete_csv = (
    OUTPUT_DIR
    / "step43_continuous_band_features_sub-01_Fz.csv"
)

df.to_csv(
    complete_csv,
    index=False
)


# ============================================================
# CREATE COMPARISON TABLES
# ============================================================

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

variance_table = df.pivot(
    index="class",
    columns="band",
    values="variance"
)


rms_table.to_csv(
    OUTPUT_DIR
    / "step43_rms_comparison.csv"
)

std_table.to_csv(
    OUTPUT_DIR
    / "step43_std_comparison.csv"
)

ptp_table.to_csv(
    OUTPUT_DIR
    / "step43_peak_to_peak_comparison.csv"
)

variance_table.to_csv(
    OUTPUT_DIR
    / "step43_variance_comparison.csv"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

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
print("VARIANCE COMPARISON")
print("=" * 70)

print(variance_table)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 43 COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(complete_csv)

print(
    OUTPUT_DIR
    / "step43_rms_comparison.csv"
)

print(
    OUTPUT_DIR
    / "step43_std_comparison.csv"
)

print(
    OUTPUT_DIR
    / "step43_peak_to_peak_comparison.csv"
)

print(
    OUTPUT_DIR
    / "step43_variance_comparison.csv"
)