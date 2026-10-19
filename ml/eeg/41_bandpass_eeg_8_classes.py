"""
Step 41 — EEG band-pass comparison for 8 classes

Subject:
    sub-01

Channel:
    Fz = BioSemi B6

Purpose:
    Compare the same representative EEG trials after
    different band-pass filters.

Bands:
    Delta : 1–4 Hz
    Theta : 4–8 Hz
    Alpha : 8–13 Hz
    Beta  : 13–30 Hz
    Gamma : 30–40 Hz

Important:
    - Original BDF is used.
    - Same first occurrence of each class as previous steps.
    - No FFT.
    - No ML.
    - Raw data is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT = "sub-01"

CHANNEL = "B6"
STANDARD_CHANNEL = "Fz"

SFREQ = 2048
TRIAL_SAMPLES = 6142
DURATION = TRIAL_SAMPLES / SFREQ


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step41"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta_1_4Hz": (1.0, 4.0),
    "theta_4_8Hz": (4.0, 8.0),
    "alpha_8_13Hz": (8.0, 13.0),
    "beta_13_30Hz": (13.0, 30.0),
    "gamma_30_40Hz": (30.0, 40.0),
}


# ============================================================
# ACTUAL SUB-01 LABELS
# ============================================================

CLASS_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


DISPLAY_NAMES = {
    "animal_auditory_imagery_task":
        "Animal - Auditory",

    "animal_silent_naming_task":
        "Animal - Silent",

    "animal_tactile_imagery_task":
        "Animal - Tactile",

    "animal_visual_imagery_task":
        "Animal - Visual",

    "tool_auditory_imagery_task":
        "Tool - Auditory",

    "tool_silent_naming_task":
        "Tool - Silent",

    "tool_tactile_imagery_task":
        "Tool - Tactile",

    "tool_visual_imagery_task":
        "Tool - Visual",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("STEP 41 — EEG BAND-PASS COMPARISON")
print("=" * 75)

print(f"Subject       : {SUBJECT}")
print(f"Channel       : {STANDARD_CHANNEL} ({CHANNEL})")
print(f"Sampling rate : {SFREQ} Hz")
print(f"Trial samples : {TRIAL_SAMPLES}")
print(f"Duration      : {DURATION:.6f} seconds")
print()

print("Frequency bands:")

for band_name, (low, high) in BANDS.items():
    print(
        f"  {band_name:<18} "
        f"{low:.1f}–{high:.1f} Hz"
    )

print()


# ============================================================
# FILE PATHS
# ============================================================

BDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_events.tsv"
)


if not BDF_PATH.exists():
    raise FileNotFoundError(
        f"BDF not found:\n{BDF_PATH}"
    )

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"Events file not found:\n{EVENTS_PATH}"
    )


# ============================================================
# LOAD EVENTS
# ============================================================

events_df = pd.read_csv(
    EVENTS_PATH,
    sep="\t"
)

task_df = events_df[
    events_df["trial_type"].isin(CLASS_NAMES)
].copy()

task_df = task_df.reset_index(
    drop=True
)

print(
    f"Task trials found: {len(task_df)}"
)

if len(task_df) != 720:
    raise ValueError(
        f"Expected 720 task trials, "
        f"found {len(task_df)}."
    )

# Validate each class
counts = task_df["trial_type"].value_counts()

for class_name in CLASS_NAMES:

    count = int(
        counts.get(class_name, 0)
    )

    if count != 90:
        raise ValueError(
            f"{class_name}: expected 90, "
            f"found {count}."
        )

print("✓ 720 task trials")
print("✓ 8 classes")
print("✓ 90 trials per class")


# ============================================================
# LOAD ORIGINAL BDF
# ============================================================

print("\nLoading original BDF...")

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=False,
    verbose="ERROR"
)

print(raw)

if raw.info["sfreq"] != SFREQ:
    raise ValueError(
        f"Expected {SFREQ} Hz, "
        f"found {raw.info['sfreq']} Hz."
    )

if CHANNEL not in raw.ch_names:
    raise ValueError(
        f"{CHANNEL} not found in BDF."
    )


# ============================================================
# EXTRACT SAME 8 REPRESENTATIVE TRIALS
# ============================================================

channel_index = raw.ch_names.index(
    CHANNEL
)

trials = {}

print("\nSelecting first occurrence of each class...")

for class_name in CLASS_NAMES:

    rows = task_df[
        task_df["trial_type"] == class_name
    ]

    first_row = rows.iloc[0]

    onset = float(
        first_row["onset"]
    )

    start = int(
        round(onset * SFREQ)
    )

    stop = start + TRIAL_SAMPLES

    data, _ = raw[
        [channel_index],
        start:stop
    ]

    signal = data[0]

    if len(signal) != TRIAL_SAMPLES:
        raise ValueError(
            f"{class_name}: incorrect "
            f"trial length."
        )

    trials[class_name] = {
        "signal": signal,
        "onset": onset,
        "start": start,
        "event_row": int(first_row.name),
    }

    print(
        f"  {DISPLAY_NAMES[class_name]:<20} "
        f"onset={onset:.6f}s"
    )


# ============================================================
# TIME AXIS
# ============================================================

time = (
    np.arange(TRIAL_SAMPLES)
    / SFREQ
)


# ============================================================
# FILTER + PLOT
# ============================================================

metadata_rows = []


for band_name, (low, high) in BANDS.items():

    print("\n" + "-" * 75)
    print(
        f"Processing {band_name}: "
        f"{low}–{high} Hz"
    )
    print("-" * 75)

    # --------------------------------------------------------
    # Create one combined figure for this band
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(14, 17),
        sharex=True
    )

    for ax, class_name in zip(
        axes,
        CLASS_NAMES
    ):

        signal = trials[
            class_name
        ]["signal"]

        # ----------------------------------------------------
        # Create temporary MNE RawArray
        # ----------------------------------------------------

        info = mne.create_info(
            ch_names=[STANDARD_CHANNEL],
            sfreq=SFREQ,
            ch_types=["eeg"]
        )

        temp_raw = mne.io.RawArray(
            signal[np.newaxis, :],
            info,
            verbose="ERROR"
        )

        # ----------------------------------------------------
        # Apply band-pass filter
        #
        # Zero-phase FIR filtering.
        # Original BDF remains untouched.
        # ----------------------------------------------------

        filtered = temp_raw.copy().filter(
            l_freq=low,
            h_freq=high,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="ERROR"
        )

        filtered_signal = (
            filtered.get_data()[0]
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mean_value = float(
            np.mean(filtered_signal)
        )

        std_value = float(
            np.std(filtered_signal)
        )

        min_value = float(
            np.min(filtered_signal)
        )

        max_value = float(
            np.max(filtered_signal)
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        ax.plot(
            time,
            filtered_signal,
            linewidth=0.8
        )

        ax.set_title(
            DISPLAY_NAMES[class_name],
            loc="left",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Amplitude (V)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.set_xlim(
            0,
            DURATION
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_rows.append(
            {
                "subject": SUBJECT,
                "class": class_name,
                "display_name":
                    DISPLAY_NAMES[class_name],
                "channel": STANDARD_CHANNEL,
                "band": band_name,
                "low_hz": low,
                "high_hz": high,
                "sampling_rate_hz": SFREQ,
                "trial_samples": TRIAL_SAMPLES,
                "duration_seconds": DURATION,
                "event_row":
                    trials[class_name]["event_row"],
                "onset_seconds":
                    trials[class_name]["onset"],
                "mean":
                    mean_value,
                "std":
                    std_value,
                "min":
                    min_value,
                "max":
                    max_value,
            }
        )

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    fig.suptitle(
        f"Band-Pass EEG — {band_name.replace('_', ' ').title()}\n"
        f"Subject: {SUBJECT} | "
        f"Channel: {STANDARD_CHANNEL} | "
        f"{low:.0f}–{high:.0f} Hz",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    axes[-1].set_xlabel(
        "Time (seconds)"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    output_path = (
        OUTPUT_DIR
        / f"step41_{band_name}_"
          f"{SUBJECT}_{STANDARD_CHANNEL}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_path = (
    OUTPUT_DIR
    / "step41_bandpass_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 41 COMPLETE")
print("=" * 75)

print(f"Subject       : {SUBJECT}")
print(f"Channel       : {STANDARD_CHANNEL}")
print("Classes       : 8")
print("Bands         : 5")
print(f"Samples/trial : {TRIAL_SAMPLES}")
print(f"Duration      : {DURATION:.6f} seconds")

print("\nGenerated band-pass figures:")

for path in sorted(
    OUTPUT_DIR.glob(
        "step41_*.png"
    )
):
    print(f"  {path}")

print("\nMetadata:")
print(f"  {metadata_path}")

print("\n✓ Original BDF preserved")
print("✓ Same 8 representative trials")
print("✓ Delta / Theta / Alpha / Beta / Gamma")
print("✓ Zero-phase FIR filtering")
print("✓ No FFT")
print("✓ No ML")