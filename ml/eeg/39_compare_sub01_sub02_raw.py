
"""
Step 39 — Cross-subject raw EEG comparison

Compare sub-01 and sub-02 using:

    Original BDF
    Channel: Fz (BioSemi B6)
    Same 8 classes
    First occurrence of each class
    6142 samples
    ~3 seconds
    No filtering
    No FFT
    No ML

Purpose:
    Study subject-to-subject variability in raw EEG.
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

SUBJECTS = ["sub-01", "sub-02"]

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
    / "cross_subject"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ACTUAL DATASET LABELS
# ============================================================

CLASS_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# ------------------------------------------------------------
# Subject-specific actual event labels
# ------------------------------------------------------------

SUBJECT_LABELS = {

    "sub-01": {
        "animal_auditory_imagery_task":
            "animal_auditory_imagery_task",

        "animal_silent":
            "animal_silent_naming_task",

        "animal_tactile_imagery_task":
            "animal_tactile_imagery_task",

        "animal_visual_imagery_task":
            "animal_visual_imagery_task",

        "tool_auditory_imagery_task":
            "tool_auditory_imagery_task",

        "tool_silent":
            "tool_silent_naming_task",

        "tool_tactile_imagery_task":
            "tool_tactile_imagery_task",

        "tool_visual_imagery_task":
            "tool_visual_imagery_task",
    },

    "sub-02": {
        "animal_auditory_imagery_task":
            "animal_auditory_imagery_task",

        "animal_silent":
            "animal_silent_naming_task",

        "animal_tactile_imagery_task":
            "animal_tactile_imagery_task",

        "animal_visual_imagery_task":
            "animal_visual_imagery_task",

        "tool_auditory_imagery_task":
            "tool_auditory_imagery_task",

        "tool_silent":
            "tool_silent_naming_task",

        "tool_tactile_imagery_task":
            "tool_tactile_imagery_task",

        "tool_visual_imagery_task":
            "tool_visual_imagery_task",
    }
}

# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "animal_auditory_imagery_task":
        "Animal - Auditory",

    "animal_silent":
        "Animal - Silent",

    "animal_tactile_imagery_task":
        "Animal - Tactile",

    "animal_visual_imagery_task":
        "Animal - Visual",

    "tool_auditory_imagery_task":
        "Tool - Auditory",

    "tool_silent":
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
print("STEP 39 — CROSS-SUBJECT RAW EEG COMPARISON")
print("=" * 75)

print("Subjects      : sub-01 vs sub-02")
print("Channel       : Fz (B6)")
print(f"Sampling rate : {SFREQ} Hz")
print(f"Samples/trial : {TRIAL_SAMPLES}")
print(f"Duration      : {DURATION:.6f} seconds")
print("Filtering     : NONE")
print("FFT           : NONE")
print("ML            : NONE")
print()


# ============================================================
# FUNCTION: LOAD SUBJECT DATA
# ============================================================

def load_subject(subject):

    print("-" * 75)
    print(f"Loading {subject}")
    print("-" * 75)

    bdf_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / subject
        / f"{subject}_task-eeg_eeg.bdf"
    )

    events_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / subject
        / f"{subject}_task-eeg_events.tsv"
    )

    if not bdf_path.exists():
        raise FileNotFoundError(
            f"BDF not found:\n{bdf_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found:\n{events_path}"
        )

    events = pd.read_csv(
        events_path,
        sep="\t"
    )

    print(f"Total event rows: {len(events)}")

    label_mapping = SUBJECT_LABELS[subject]

    # --------------------------------------------------------
    # Find all 8 task classes
    # --------------------------------------------------------

    actual_labels = list(
        label_mapping.values()
    )

    task_df = events[
        events["trial_type"].isin(actual_labels)
    ].copy()

    task_df = task_df.reset_index(drop=True)

    print(
        f"Task trials found: {len(task_df)}"
    )

    if len(task_df) != 720:
        raise ValueError(
            f"{subject}: expected 720 task trials, "
            f"found {len(task_df)}"
        )

    # --------------------------------------------------------
    # Load BDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=False,
        verbose="ERROR"
    )

    print(
        f"Sampling frequency: "
        f"{raw.info['sfreq']} Hz"
    )

    if raw.info["sfreq"] != SFREQ:
        raise ValueError(
            f"{subject}: unexpected sampling rate."
        )

    if CHANNEL not in raw.ch_names:
        raise ValueError(
            f"{subject}: channel {CHANNEL} not found."
        )

    # --------------------------------------------------------
    # Extract first occurrence of each class
    # --------------------------------------------------------

    results = {}

    channel_index = raw.ch_names.index(
        CHANNEL
    )

    for class_key in CLASS_NAMES:

        actual_label = label_mapping[
            class_key
        ]

        rows = task_df[
            task_df["trial_type"] == actual_label
        ]

        if len(rows) != 90:
            raise ValueError(
                f"{subject} / {actual_label}: "
                f"expected 90 trials, "
                f"found {len(rows)}"
            )

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
                f"{subject} / {class_key}: "
                f"incorrect trial length."
            )

        results[class_key] = {
            "signal": signal,
            "onset": onset,
            "start_sample": start,
            "event_row": int(first_row.name),
        }

    print(
        f"✓ {subject}: "
        f"8 classes extracted"
    )

    return results


# ============================================================
# LOAD BOTH SUBJECTS
# ============================================================

subject_data = {}

for subject in SUBJECTS:
    subject_data[subject] = load_subject(
        subject
    )

print("\n✓ Both subjects loaded successfully.")


# ============================================================
# TIME AXIS
# ============================================================

time = (
    np.arange(TRIAL_SAMPLES)
    / SFREQ
)


# ============================================================
# COMPARE EACH CLASS
# ============================================================

summary_rows = []


print("\n")
print("=" * 75)
print("CROSS-SUBJECT STATISTICS")
print("=" * 75)


for class_key in CLASS_NAMES:

    signal_01 = subject_data[
        "sub-01"
    ][class_key]["signal"]

    signal_02 = subject_data[
        "sub-02"
    ][class_key]["signal"]

    display_name = DISPLAY_NAMES[
        class_key
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_01 = float(
        np.mean(signal_01)
    )

    mean_02 = float(
        np.mean(signal_02)
    )

    std_01 = float(
        np.std(signal_01)
    )

    std_02 = float(
        np.std(signal_02)
    )

    min_01 = float(
        np.min(signal_01)
    )

    min_02 = float(
        np.min(signal_02)
    )

    max_01 = float(
        np.max(signal_01)
    )

    max_02 = float(
        np.max(signal_02)
    )

    # --------------------------------------------------------
    # Pearson correlation
    #
    # This measures similarity in temporal shape.
    # It does NOT mean the subjects have identical EEG.
    # --------------------------------------------------------

    correlation = float(
        np.corrcoef(
            signal_01,
            signal_02
        )[0, 1]
    )

    # --------------------------------------------------------
    # RMSE
    #
    # Measures absolute waveform difference.
    # --------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(
                (signal_01 - signal_02) ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\n{display_name}"
    )

    print(
        f"  sub-01 mean : {mean_01:.6e}"
    )

    print(
        f"  sub-02 mean : {mean_02:.6e}"
    )

    print(
        f"  sub-01 std  : {std_01:.6e}"
    )

    print(
        f"  sub-02 std  : {std_02:.6e}"
    )

    print(
        f"  correlation : {correlation:.4f}"
    )

    print(
        f"  RMSE        : {rmse:.6e}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        time,
        signal_01,
        linewidth=0.8,
        label="sub-01"
    )

    ax.plot(
        time,
        signal_02,
        linewidth=0.8,
        label="sub-02"
    )

    ax.set_title(
        f"{display_name}\n"
        f"Raw EEG — Fz"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude (V)"
    )

    ax.set_xlim(
        0,
        DURATION
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    safe_name = (
        class_key
        .replace("/", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"step39_{safe_name}_"
          f"sub01_vs_sub02.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    summary_rows.append(
        {
            "class": class_key,
            "display_name": display_name,

            "sub01_mean": mean_01,
            "sub02_mean": mean_02,

            "sub01_std": std_01,
            "sub02_std": std_02,

            "sub01_min": min_01,
            "sub02_min": min_02,

            "sub01_max": max_01,
            "sub02_max": max_02,

            "pearson_correlation":
                correlation,

            "rmse":
                rmse,

            "sub01_event_row":
                subject_data[
                    "sub-01"
                ][class_key]["event_row"],

            "sub02_event_row":
                subject_data[
                    "sub-02"
                ][class_key]["event_row"],

            "sub01_onset":
                subject_data[
                    "sub-01"
                ][class_key]["onset"],

            "sub02_onset":
                subject_data[
                    "sub-02"
                ][class_key]["onset"],
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "step39_sub01_vs_sub02_raw_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# CREATE SUMMARY BAR CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

x = np.arange(
    len(summary_df)
)

ax.bar(
    x - 0.2,
    summary_df["sub01_std"],
    width=0.4,
    label="sub-01"
)

ax.bar(
    x + 0.2,
    summary_df["sub02_std"],
    width=0.4,
    label="sub-02"
)

ax.set_xticks(x)

ax.set_xticklabels(
    summary_df["display_name"],
    rotation=30,
    ha="right"
)

ax.set_ylabel(
    "Signal Standard Deviation (V)"
)

ax.set_title(
    "Raw EEG Variability — sub-01 vs sub-02\n"
    "Fz Channel"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

std_plot_path = (
    OUTPUT_DIR
    / "step39_sub01_vs_sub02_std_comparison.png"
)

plt.savefig(
    std_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 39 COMPLETE")
print("=" * 75)

print("\nGenerated comparison plots:")
for path in sorted(
    OUTPUT_DIR.glob(
        "step39_*sub01_vs_sub02.png"
    )
):
    print(f"  {path}")

print("\nSummary:")
print(f"  {summary_path}")

print("\nStandard deviation comparison:")
print(f"  {std_plot_path}")

print("\n✓ Same channel: Fz")
print("✓ Same trial length")
print("✓ Same sampling rate")
print("✓ Same class-selection rule")
print("✓ Original raw BDF")
print("✓ No filtering")
print("✓ No FFT")
print("✓ No ML")